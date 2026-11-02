# SPDX-License-Identifier: Apache-2.0
"""Node-cache daemon: one per node, shared by every SGLang rank and instance.

    python -m sglang_node_cache.daemon --shm-size-gb 768 \
        --ssd-dir /mnt/nvme/node_cache --ssd-size-gb 6000

The arena is an anonymous memfd handed to clients over the socket, so it never
appears under /dev/shm: it needs no shared /dev/shm between containers, is not
bounded by that tmpfs, and is freed when the daemon and its last client exit.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import mmap
import os
import signal
import socket
import socketserver
import struct
import threading
import uuid

from sglang_node_cache.protocol import (
    Hello,
    Op,
    Request,
    Response,
    recv_request,
    send_fd,
    send_frame,
    socket_address,
)
from sglang_node_cache.ssd_tier import SsdTier
from sglang_node_cache.store import NodeCacheStore

logger = logging.getLogger(__name__)

DEFAULT_SOCKET = "/dev/shm/sglang_node_cache/default.sock"
# Arbitrary headroom left to the OS and the engines when sizing the arena.
_MEMORY_HEADROOM_BYTES = 16 << 30


class NodeCacheDaemon:
    def __init__(
        self,
        *,
        socket_path: str,
        shm_bytes: int,
        chunk_bytes: int,
        ssd_dir: str | None,
        ssd_bytes: int,
        clean_ratio: float,
        fill_wait_s: float,
        prefault: bool = False,
        health_port: int | None = None,
    ):
        self.socket_path = socket_path
        self._socket_on_disk = not socket_path.startswith("@")
        self.arena_id = uuid.uuid4().hex
        self._arena_fd = _create_arena(nbytes=shm_bytes, prefault=prefault)
        self._mmap = mmap.mmap(self._arena_fd, shm_bytes, flags=mmap.MAP_SHARED)
        ssd = None
        if ssd_dir:
            ssd = SsdTier(root=ssd_dir, capacity_bytes=ssd_bytes)
        self.store = NodeCacheStore(
            arena=memoryview(self._mmap),
            chunk_bytes=chunk_bytes,
            ssd=ssd,
            clean_ratio=clean_ratio,
            fill_wait_s=fill_wait_s,
        )
        self._hello = Hello(arena_bytes=shm_bytes, arena_id=self.arena_id)
        self._conn_ids = itertools.count(1)
        if self._socket_on_disk:
            os.makedirs(os.path.dirname(socket_path) or ".", exist_ok=True)
            if os.path.exists(socket_path):
                os.unlink(socket_path)
        self._server = _Server(socket_address(socket_path), _make_handler(self))
        self._health = None if health_port is None else _open_health_port(health_port)
        logger.info(
            "Node cache: arena %.1f GiB (prefault=%s), SSD %s (%.1f GiB), "
            "socket %s, health port %s",
            shm_bytes / (1 << 30),
            prefault,
            ssd_dir or "disabled",
            ssd_bytes / (1 << 30) if ssd_dir else 0,
            self.socket_path,
            health_port,
        )

    @property
    def arena_fd(self) -> int:
        return self._arena_fd

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread

    def shutdown(self) -> None:
        """Stop serve_forever(); call from a thread other than the serving one."""
        self._server.shutdown()

    def close(self) -> None:
        self._server.server_close()
        if self._health is not None:
            self._health.close()
        self.store.close()
        if self._socket_on_disk and os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def next_conn_id(self) -> int:
        return next(self._conn_ids)

    def handle(self, *, conn: int, request: Request) -> Response:
        store = self.store
        op = request.op
        if op == Op.HELLO:
            return Response(hello=self._hello)
        if op == Op.EXISTS:
            return Response(flags=store.exists(request.keys))
        if op == Op.ACQUIRE:
            return Response(
                offsets=store.acquire(
                    conn=conn, keys=request.keys, nbytes=request.nbytes
                )
            )
        if op == Op.RELEASE:
            store.release(conn=conn, keys=request.keys)
            return Response()
        if op == Op.RESERVE:
            return Response(
                offsets=store.reserve(
                    conn=conn, keys=request.keys, nbytes=request.nbytes
                )
            )
        if op == Op.COMMIT:
            store.commit(conn=conn, keys=request.keys)
            return Response()
        if op == Op.ABORT:
            store.abort(conn=conn, keys=request.keys)
            return Response()
        if op == Op.STATS:
            return Response(stats=store.stats())
        if op == Op.CLEAR:
            store.clear()
            return Response()
        return Response(error=f"unknown op {op!r}")


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def _make_handler(daemon: NodeCacheDaemon) -> type[socketserver.BaseRequestHandler]:
    class _Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            conn = daemon.next_conn_id()
            sock: socket.socket = self.request
            if not _same_user(sock):
                # An abstract socket is reachable by every user on the node.
                logger.warning("Rejected a node-cache client run by another user")
                return
            try:
                while True:
                    request = recv_request(sock)
                    if request is None:
                        return
                    try:
                        response = daemon.handle(conn=conn, request=request)
                    except ValueError as e:
                        response = Response(error=str(e))
                    send_frame(sock=sock, payload=response)
                    if request.op == Op.HELLO:
                        send_fd(sock=sock, fd=daemon.arena_fd)
            except (ConnectionError, OSError):
                pass
            finally:
                daemon.store.disconnect(conn)

    return _Handler


def _same_user(sock: socket.socket) -> bool:
    creds = sock.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
    )
    _, uid, _ = struct.unpack("3i", creds)
    return uid == os.getuid()


def _create_arena(*, nbytes: int, prefault: bool) -> int:
    available = _mem_available_bytes()
    if nbytes + _MEMORY_HEADROOM_BYTES > available:
        raise MemoryError(
            f"Arena of {nbytes / (1 << 30):.1f} GiB plus "
            f"{_MEMORY_HEADROOM_BYTES >> 30} GiB headroom does not fit in "
            f"{available / (1 << 30):.1f} GiB of available memory"
        )
    fd = os.memfd_create("sglang_node_cache", os.MFD_CLOEXEC)
    os.ftruncate(fd, nbytes)
    if prefault:
        # A sparse arena is filled on first write; if the node runs out of
        # memory by then, the writing engine dies with SIGBUS instead.
        os.posix_fallocate(fd, 0, nbytes)
    return fd


def _mem_available_bytes() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def _open_health_port(port: int) -> socket.socket:
    """A TCP port that accepts and closes connections, for launchers that wait on one."""
    listener = socket.create_server(("", port))

    def _accept_loop() -> None:
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            conn.close()

    threading.Thread(target=_accept_loop, name="node-cache-health", daemon=True).start()
    return listener


def main() -> None:
    parser = argparse.ArgumentParser(description="SGLang HiCache node-cache daemon")
    parser.add_argument(
        "--socket",
        default=DEFAULT_SOCKET,
        help="Unix socket path, or @name for an abstract socket that containers "
        "sharing the host network reach without a shared filesystem.",
    )
    parser.add_argument("--shm-size-gb", type=float, required=True)
    parser.add_argument(
        "--prefault",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allocate the whole arena before serving; slow for hundreds of GiB "
        "since the kernel zeroes it single-threaded.",
    )
    parser.add_argument(
        "--health-port",
        type=int,
        default=None,
        help="Open this TCP port once the daemon is ready to serve.",
    )
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--ssd-dir", default=None)
    parser.add_argument("--ssd-size-gb", type=float, default=0)
    parser.add_argument(
        "--clean-ratio",
        type=float,
        default=0.1,
        help="Fraction of the arena, oldest first, kept persisted on SSD.",
    )
    parser.add_argument(
        "--fill-wait-s",
        type=float,
        default=2.0,
        help="How long a reader waits for a page another client is writing.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    if args.ssd_dir and args.ssd_size_gb <= 0:
        parser.error("--ssd-size-gb must be positive when --ssd-dir is set")

    daemon = NodeCacheDaemon(
        socket_path=args.socket,
        shm_bytes=int(args.shm_size_gb * (1 << 30)),
        chunk_bytes=args.chunk_mb << 20,
        ssd_dir=args.ssd_dir,
        ssd_bytes=int(args.ssd_size_gb * (1 << 30)),
        clean_ratio=args.clean_ratio,
        fill_wait_s=args.fill_wait_s,
        prefault=args.prefault,
        health_port=args.health_port,
    )

    def _stop(signum, frame):
        threading.Thread(target=daemon.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    daemon.serve_forever()
    daemon.close()


if __name__ == "__main__":
    main()

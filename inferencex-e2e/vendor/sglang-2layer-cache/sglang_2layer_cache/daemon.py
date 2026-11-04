# SPDX-License-Identifier: Apache-2.0
"""Two-layer cache daemon: one per node, shared by every SGLang rank and instance.

Layer 1 is the shared DRAM arena; layer 2 is either SSD files or a Mooncake
store reached over TCP or RDMA:

    python -m sglang_2layer_cache.daemon --shm-size-gb 768 \
        --layer2 mooncake --mooncake-master 10.0.0.5:50051 --mooncake-protocol rdma

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

import msgspec

from sglang_2layer_cache.lower_tier import LowerTier
from sglang_2layer_cache.mooncake_tier import MooncakeTier, MooncakeTierConfig
from sglang_2layer_cache.protocol import (
    Hello,
    Op,
    Request,
    Response,
    recv_request,
    send_fd,
    send_frame,
    socket_address,
)
from sglang_2layer_cache.ssd_tier import SsdTier
from sglang_2layer_cache.store import TwoLayerStore, arena_address

logger = logging.getLogger(__name__)

DEFAULT_SOCKET = "/dev/shm/sglang_2layer_cache/default.sock"
# Arbitrary headroom left to the OS and the engines when sizing the arena.
_MEMORY_HEADROOM_BYTES = 16 << 30


class SsdTierConfig(msgspec.Struct, frozen=True):
    root: str
    capacity_bytes: int


Layer2Config = SsdTierConfig | MooncakeTierConfig


class TwoLayerDaemon:
    def __init__(
        self,
        *,
        socket_path: str,
        shm_bytes: int,
        chunk_bytes: int,
        layer2: Layer2Config | None,
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
        arena = memoryview(self._mmap)
        self.store = TwoLayerStore(
            arena=arena,
            chunk_bytes=chunk_bytes,
            lower=_make_lower(config=layer2, arena=arena),
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
            "Two-layer cache: arena %.1f GiB (prefault=%s), layer 2 %s, "
            "socket %s, health port %s",
            shm_bytes / (1 << 30),
            prefault,
            layer2,
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


def _make_handler(daemon: TwoLayerDaemon) -> type[socketserver.BaseRequestHandler]:
    class _Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            conn = daemon.next_conn_id()
            sock: socket.socket = self.request
            if not _same_user(sock):
                # An abstract socket is reachable by every user on the node.
                logger.warning("Rejected a two-layer-cache client run by another user")
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


def _make_lower(*, config: Layer2Config | None, arena: memoryview) -> LowerTier | None:
    if config is None:
        return None
    if isinstance(config, SsdTierConfig):
        return SsdTier(root=config.root, capacity_bytes=config.capacity_bytes)
    return MooncakeTier(
        config=config, arena_addr=arena_address(arena), arena_bytes=len(arena)
    )


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
    fd = os.memfd_create("sglang_2layer_cache", os.MFD_CLOEXEC)
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

    threading.Thread(
        target=_accept_loop, name="two-layer-cache-health", daemon=True
    ).start()
    return listener


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGLang HiCache two-layer cache daemon"
    )
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
    parser.add_argument("--layer2", choices=["none", "ssd", "mooncake"], default="none")
    parser.add_argument("--ssd-dir", default=None)
    parser.add_argument("--ssd-size-gb", type=float, default=0)
    parser.add_argument(
        "--mooncake-master", default=None, help="host:port of mooncake_master"
    )
    parser.add_argument("--mooncake-protocol", choices=["tcp", "rdma"], default="tcp")
    parser.add_argument("--mooncake-metadata-server", default="P2PHANDSHAKE")
    parser.add_argument(
        "--mooncake-local-hostname",
        default="127.0.0.1",
        help="Address other Mooncake clients use to reach this daemon.",
    )
    parser.add_argument(
        "--mooncake-device", default="", help="RDMA devices, e.g. mlx5_0,mlx5_1"
    )
    parser.add_argument(
        "--mooncake-segment-gb",
        type=float,
        default=0,
        help="DRAM this daemon adds to the Mooncake pool; 0 for a pure client "
        "of an external Mooncake cluster.",
    )
    parser.add_argument(
        "--mooncake-ssd-offload-dir",
        default=None,
        help="Let Mooncake spill this daemon's segment to this SSD directory.",
    )
    parser.add_argument(
        "--clean-ratio",
        type=float,
        default=0.1,
        help="Fraction of the arena, oldest first, kept demoted to layer 2.",
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
    daemon = TwoLayerDaemon(
        socket_path=args.socket,
        shm_bytes=int(args.shm_size_gb * (1 << 30)),
        chunk_bytes=args.chunk_mb << 20,
        layer2=_layer2_from_args(parser=parser, args=args),
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


def _layer2_from_args(
    *, parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Layer2Config | None:
    if args.layer2 == "ssd":
        if not args.ssd_dir or args.ssd_size_gb <= 0:
            parser.error("--layer2 ssd needs --ssd-dir and a positive --ssd-size-gb")
        return SsdTierConfig(
            root=args.ssd_dir, capacity_bytes=int(args.ssd_size_gb * (1 << 30))
        )
    if args.layer2 == "mooncake":
        if not args.mooncake_master:
            parser.error("--layer2 mooncake needs --mooncake-master")
        return MooncakeTierConfig(
            master_server_address=args.mooncake_master,
            protocol=args.mooncake_protocol,
            metadata_server=args.mooncake_metadata_server,
            local_hostname=args.mooncake_local_hostname,
            device_name=args.mooncake_device,
            segment_bytes=int(args.mooncake_segment_gb * (1 << 30)),
            ssd_offload_path=args.mooncake_ssd_offload_dir,
        )
    return None


if __name__ == "__main__":
    main()

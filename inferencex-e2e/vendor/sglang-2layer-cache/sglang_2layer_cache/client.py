# SPDX-License-Identifier: Apache-2.0
"""Client side of the two-layer cache: RPC to the daemon plus a mapping of its arena."""

from __future__ import annotations

import logging
import os
import socket
import threading

import torch

from sglang_2layer_cache.protocol import (
    Op,
    Request,
    Response,
    StoreStats,
    recv_fd,
    recv_response,
    send_frame,
    socket_address,
)

logger = logging.getLogger(__name__)


class TwoLayerClient:
    """Thread-safe: each calling thread gets its own daemon connection, and
    pins are owned by that connection."""

    def __init__(self, *, socket_path: str, timeout_s: float = 30.0):
        self.socket_path = socket_path
        self.timeout_s = timeout_s
        self._local = threading.local()
        self._map_lock = threading.Lock()
        self._arena_id: str | None = None
        self.arena: torch.Tensor | None = None
        self._connect()

    def exists(self, keys: list[str]) -> list[bool]:
        return self._call(Request(op=Op.EXISTS, keys=keys)).flags

    def acquire(self, *, keys: list[str], nbytes: int) -> list[int]:
        return self._call(Request(op=Op.ACQUIRE, keys=keys, nbytes=nbytes)).offsets

    def release(self, keys: list[str]) -> None:
        self._call(Request(op=Op.RELEASE, keys=keys))

    def reserve(self, *, keys: list[str], nbytes: int) -> list[int]:
        return self._call(Request(op=Op.RESERVE, keys=keys, nbytes=nbytes)).offsets

    def commit(self, keys: list[str]) -> None:
        self._call(Request(op=Op.COMMIT, keys=keys))

    def abort(self, keys: list[str]) -> None:
        self._call(Request(op=Op.ABORT, keys=keys))

    def stats(self) -> StoreStats:
        return self._call(Request(op=Op.STATS)).stats

    def clear(self) -> None:
        self._call(Request(op=Op.CLEAR))

    def close(self) -> None:
        sock = self._local.__dict__.pop("sock", None)
        if sock is not None:
            sock.close()

    def _call(self, request: Request) -> Response:
        sock = self._local.__dict__.get("sock")
        if sock is None:
            sock = self._connect()
        try:
            send_frame(sock=sock, payload=request)
            response = recv_response(sock)
        except (ConnectionError, OSError):
            # The daemon dropped every pin of this connection; a retry would
            # hand back offsets the caller never asked for, so fail the call.
            self._local.__dict__.pop("sock", None)
            sock.close()
            raise
        if response.error:
            raise RuntimeError(f"two-layer-cache daemon: {response.error}")
        return response

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_s)
        sock.connect(socket_address(self.socket_path))
        send_frame(sock=sock, payload=Request(op=Op.HELLO))
        hello = recv_response(sock).hello
        arena_fd = recv_fd(sock)
        try:
            self._map_arena(hello=hello, arena_fd=arena_fd)
        finally:
            os.close(arena_fd)
        self._local.sock = sock
        return sock

    def _map_arena(self, *, hello, arena_fd: int) -> None:
        with self._map_lock:
            if hello.arena_id == self._arena_id:
                return
            if self._arena_id is not None:
                logger.warning("Node-cache daemon restarted; remapping its arena")
            # torch maps by path; the memfd is reachable only through this process's fd table.
            self.arena = torch.from_file(
                f"/proc/self/fd/{arena_fd}",
                shared=True,
                size=hello.arena_bytes,
                dtype=torch.uint8,
            )
            self._arena_id = hello.arena_id

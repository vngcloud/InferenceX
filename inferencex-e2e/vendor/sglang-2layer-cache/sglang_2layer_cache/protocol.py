# SPDX-License-Identifier: Apache-2.0
"""Wire protocol between the two-layer-cache daemon and its clients.

One Unix stream socket per client thread; every message is a 4-byte
big-endian length followed by a msgpack-encoded struct. Page bytes never cross
the socket: the HELLO reply is followed by one byte carrying the arena memfd
(SCM_RIGHTS), both sides map it, and later messages only carry keys and
arena offsets.
"""

from __future__ import annotations

import enum
import os
import socket
import struct

import msgspec

_HEADER = struct.Struct(">I")
# Arbitrary; one request carries at most a few hundred keys.
MAX_FRAME_BYTES = 64 << 20

# Offsets returned by acquire/reserve that are not arena positions.
MISSING = -1
PRESENT = -2
NO_SPACE = -3


class Op(str, enum.Enum):
    HELLO = "hello"
    EXISTS = "exists"
    ACQUIRE = "acquire"
    RELEASE = "release"
    RESERVE = "reserve"
    COMMIT = "commit"
    ABORT = "abort"
    STATS = "stats"
    CLEAR = "clear"


class Request(msgspec.Struct, frozen=True, omit_defaults=True):
    op: Op
    keys: list[str] = []
    nbytes: int = 0


class Hello(msgspec.Struct, frozen=True):
    arena_bytes: int
    # Changes on every daemon start; a client seeing a new id must remap.
    arena_id: str


class StoreStats(msgspec.Struct, frozen=True):
    arena_bytes: int
    arena_carved_bytes: int
    shm_pages: int
    shm_bytes: int
    dirty_bytes: int
    lower_tier: str
    lower_pages: int
    lower_bytes: int
    lower_capacity_bytes: int
    shm_hits: int
    lower_hits: int
    misses: int
    inserts: int
    no_space: int
    evictions: int
    demotions: int
    demote_failures: int


class Response(msgspec.Struct, frozen=True, omit_defaults=True):
    offsets: list[int] = []
    flags: list[bool] = []
    hello: Hello | None = None
    stats: StoreStats | None = None
    error: str = ""


_encoder = msgspec.msgpack.Encoder()
_request_decoder = msgspec.msgpack.Decoder(Request)
_response_decoder = msgspec.msgpack.Decoder(Response)


def socket_address(path: str) -> str:
    """``@name`` is a Linux abstract socket: it lives in the network namespace,
    so containers sharing the host network reach it without a shared filesystem."""
    return "\0" + path[1:] if path.startswith("@") else path


def send_frame(*, sock: socket.socket, payload: msgspec.Struct) -> None:
    body = _encoder.encode(payload)
    sock.sendall(_HEADER.pack(len(body)) + body)


def send_fd(*, sock: socket.socket, fd: int) -> None:
    socket.send_fds(sock, [b"F"], [fd])


def recv_fd(sock: socket.socket) -> int:
    data, fds, _, _ = socket.recv_fds(sock, 1, 1)
    if data != b"F" or len(fds) != 1:
        for fd in fds:
            os.close(fd)
        raise ConnectionError(
            "two-layer-cache daemon did not pass the arena descriptor"
        )
    return fds[0]


def recv_request(sock: socket.socket) -> Request | None:
    body = _recv_frame(sock)
    return None if body is None else _request_decoder.decode(body)


def recv_response(sock: socket.socket) -> Response:
    body = _recv_frame(sock)
    if body is None:
        raise ConnectionError("two-layer-cache daemon closed the connection")
    return _response_decoder.decode(body)


def _recv_frame(sock: socket.socket) -> bytes | None:
    header = _recv_exact(sock=sock, size=_HEADER.size)
    if header is None:
        return None
    (size,) = _HEADER.unpack(header)
    if size > MAX_FRAME_BYTES:
        raise ConnectionError(
            f"two-layer-cache frame of {size} bytes exceeds the limit"
        )
    body = _recv_exact(sock=sock, size=size)
    if body is None:
        raise ConnectionError("two-layer-cache connection closed mid-frame")
    return body


def _recv_exact(*, sock: socket.socket, size: int) -> bytes | None:
    buf = bytearray(size)
    view = memoryview(buf)
    got = 0
    while got < size:
        n = sock.recv_into(view[got:])
        if n == 0:
            if got == 0:
                return None
            raise ConnectionError("two-layer-cache connection closed mid-frame")
        got += n
    return bytes(buf)

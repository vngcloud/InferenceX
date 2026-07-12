#!/usr/bin/env python3
"""Per-turn session-splitting router for LMCache P2P benchmarking.

Sits in front of N vLLM instances and routes each turn of a conversation to a
DIFFERENT backend, so consecutive turns of the same session land on different
instances. This forces the LMCache P2P cross-instance KV lookup/transfer path
that ordinary session-sticky load balancing never exercises.

Routing:
  - Sessions are keyed on the X-Correlation-ID (fallback X-Session-ID) header
    that AIPerf stamps on every request.
  - The first time a session is seen it is assigned a base backend via global
    round-robin (so sessions stay balanced across instances).
  - Turn k of that session goes to backend (base + k) % N. With two backends
    this yields A, B, A, B ... within every session.
  - Requests without a correlation header (health checks, /v1/models) go to
    backend 0 and do not disturb any counter.

Streaming is passed through transparently (chunks are relayed as they arrive)
so TTFT / SSE timing is preserved.

Config via env:
  BACKENDS    comma-separated backend base URLs
              (default "http://127.0.0.1:8000,http://127.0.0.1:8001")
  ROUTER_PORT listen port (default 8080)
  ROUTER_LOG  "1" to log one line per routed turn (default "1")
"""

from __future__ import annotations

import os
import sys

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

BACKENDS = [
    u.strip().rstrip("/")
    for u in os.environ.get(
        "BACKENDS", "http://127.0.0.1:8000,http://127.0.0.1:8001"
    ).split(",")
    if u.strip()
]
PORT = int(os.environ.get("ROUTER_PORT", "8080"))
LOG = os.environ.get("ROUTER_LOG", "1") not in ("0", "", "false", "False")

# Hop-by-hop headers (RFC 7230) plus framing headers the proxy must re-derive.
HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

# corr_id -> [base_backend_index, turns_seen]
_sessions: dict[str, list[int]] = {}
_rr = 0  # global round-robin used only to assign each new session's base


def _pick_backend(request: web.Request) -> tuple[int, str | None, int | None]:
    """Return (backend_index, corr_id, turn_index) for this request."""
    global _rr
    n = len(BACKENDS)
    corr = request.headers.get("X-Correlation-ID") or request.headers.get(
        "X-Session-ID"
    )
    if not corr:
        return 0, None, None
    # Single-threaded asyncio: this read-modify-write has no await, so it is
    # atomic across concurrent sessions.
    state = _sessions.get(corr)
    if state is None:
        base = _rr % n
        _rr += 1
        state = [base, 0]
        _sessions[corr] = state
    base, count = state
    state[1] = count + 1
    return (base + count) % n, corr, count


async def handle(request: web.Request) -> web.StreamResponse:
    idx, corr, turn = _pick_backend(request)
    backend = BACKENDS[idx]
    target = backend + request.raw_path
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    body = await request.read()

    if LOG and corr is not None:
        sys.stderr.write(
            f"[router] corr={corr[:8]} turn={turn} -> backend#{idx} {backend}\n"
        )
        sys.stderr.flush()

    client: ClientSession = request.app["client"]
    try:
        async with client.request(
            request.method,
            target,
            headers=headers,
            data=body,
            allow_redirects=False,
        ) as upstream:
            resp = web.StreamResponse(status=upstream.status)
            for k, v in upstream.headers.items():
                if k.lower() not in HOP:
                    resp.headers[k] = v
            await resp.prepare(request)
            async for chunk in upstream.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp
    except Exception as e:  # noqa: BLE001 - surface any upstream failure as 502
        return web.Response(status=502, text=f"router upstream error: {e}")


async def _on_startup(app: web.Application) -> None:
    app["client"] = ClientSession(
        timeout=ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=None),
        connector=TCPConnector(limit=0, ttl_dns_cache=300),
        auto_decompress=False,
    )


async def _on_cleanup(app: web.Application) -> None:
    await app["client"].close()


def main() -> None:
    if len(BACKENDS) < 2:
        sys.stderr.write(f"[router] WARNING: only {len(BACKENDS)} backend(s): {BACKENDS}\n")
    sys.stderr.write(f"[router] listening on :{PORT} -> {BACKENDS}\n")
    sys.stderr.flush()
    app = web.Application(client_max_size=1024**3)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    app.router.add_route("*", "/{tail:.*}", handle)
    web.run_app(app, host="0.0.0.0", port=PORT, access_log=None, print=None)


if __name__ == "__main__":
    main()

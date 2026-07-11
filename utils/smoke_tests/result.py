"""Shared result type for smoke-test probes."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProbeResult:
    ok: bool
    detail: str
    data: dict = field(default_factory=dict)

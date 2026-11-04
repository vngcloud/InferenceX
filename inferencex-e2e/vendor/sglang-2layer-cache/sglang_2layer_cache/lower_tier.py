# SPDX-License-Identifier: Apache-2.0
"""Interface of layer 2, the tier below the shared DRAM arena.

Every call is batched: layer 2 may sit behind a network round trip (Mooncake),
and the store never calls it while holding its own lock.
"""

from __future__ import annotations

from typing import Protocol

import msgspec


class ArenaSlot(msgspec.Struct, frozen=True):
    """One page-sized region of the arena, as a view and as a raw address."""

    view: memoryview
    addr: int


class LowerTier(Protocol):
    name: str
    # 0 when layer 2 does not report its capacity.
    capacity_bytes: int

    def contains_many(self, keys: list[str]) -> list[bool]: ...

    def read_many(self, *, keys: list[str], slots: list[ArenaSlot]) -> list[bool]:
        """Fill each slot with its page; False when absent or of another size."""
        ...

    def write_many(self, *, keys: list[str], slots: list[ArenaSlot]) -> list[bool]: ...

    def clear(self) -> None: ...

    def usage(self) -> tuple[int, int]:
        """(pages, bytes) held, or (0, 0) when layer 2 does not report it."""
        ...

    def close(self) -> None: ...

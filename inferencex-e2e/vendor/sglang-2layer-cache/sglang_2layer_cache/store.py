# SPDX-License-Identifier: Apache-2.0
"""Index and allocator of the two-layer cache, owned by the daemon.

Layer 1 is one shared DRAM arena (an anonymous memfd) carved into fixed-size
slots per page size. Clients copy page bytes in and out of the arena
themselves; this store only hands out offsets and tracks who holds which slot.
Layer 2 (SSD files or a Mooncake store) sits below it.

Life of a page:

    reserve -> (client copies host page into the slot) -> commit -> ready
    ready, dirty --cleaner demotes to layer 2--> ready, clean --LRU--> evicted
    evicted, in layer 2 --acquire--> promoted back into a slot

Only clean pages are evicted, so a page leaves the arena only after layer 2
has it (or when no layer 2 is configured). The cleaner keeps the oldest
``clean_ratio`` of the arena clean, so eviction rarely has to wait for it.
Layer 2 is never called with the lock held: it may be a network round trip.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from collections import Counter, OrderedDict

import msgspec

from sglang_2layer_cache.lower_tier import ArenaSlot, LowerTier
from sglang_2layer_cache.protocol import MISSING, NO_SPACE, PRESENT, StoreStats
from sglang_2layer_cache.ssd_tier import validate_key

logger = logging.getLogger(__name__)

# Slot sizes are rounded to this so every slot is page- and dtype-aligned.
SLOT_ALIGN = 4096
# Arbitrary; bounds how long the cleaner holds pins on one pass.
_CLEANER_BATCH = 64


class _Entry(msgspec.Struct):
    offset: int
    nbytes: int
    # A filling entry is reserved by ``owner`` and invisible to readers.
    ready: bool = False
    owner: int = 0
    pins: int = 0
    # May be dropped from the arena without losing the only copy.
    clean: bool = False


class _SizeClass(msgspec.Struct):
    slot_bytes: int
    free: list[int] = []
    num_slots: int = 0


class TwoLayerStore:
    def __init__(
        self,
        *,
        arena: memoryview,
        chunk_bytes: int,
        lower: LowerTier | None,
        clean_ratio: float = 0.1,
        fill_wait_s: float = 2.0,
        start_cleaner: bool = True,
    ):
        self.arena = arena
        self.arena_bytes = len(arena)
        self.arena_addr = arena_address(arena)
        self.chunk_bytes = chunk_bytes
        self.lower = lower
        self.clean_target_bytes = int(self.arena_bytes * clean_ratio)
        self.fill_wait_s = fill_wait_s
        self._cond = threading.Condition(threading.Lock())
        # key -> entry, least recently used first.
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._classes: dict[int, _SizeClass] = {}
        self._carved_bytes = 0
        self._conn_pins: dict[int, Counter[str]] = {}
        self._shm_bytes = 0
        self._dirty_bytes = 0
        self._counters = Counter()
        self._closed = False
        self._cleaner = None
        if lower is not None and start_cleaner:
            self._cleaner = threading.Thread(
                target=self._cleaner_loop, name="two-layer-cleaner", daemon=True
            )
            self._cleaner.start()

    # ------------------------------------------------------------ queries

    def exists(self, keys: list[str]) -> list[bool]:
        with self._cond:
            flags = [self._is_ready(key) for key in keys]
        missing = [i for i, hit in enumerate(flags) if not hit]
        if self.lower is not None and missing:
            in_lower = self.lower.contains_many([keys[i] for i in missing])
            for i, hit in zip(missing, in_lower):
                flags[i] = hit
        return flags

    def acquire(self, *, conn: int, keys: list[str], nbytes: int) -> list[int]:
        """Pin the leading run of present pages for reading, promoting layer-2 hits."""
        offsets = self._pin_arena_run(conn=conn, keys=keys, nbytes=nbytes)
        start = len(offsets)
        if start < len(keys) and self.lower is not None:
            offsets += self._promote_run(conn=conn, keys=keys[start:], nbytes=nbytes)
        if len(offsets) < len(keys):
            with self._cond:
                self._counters["misses"] += 1
        return offsets + [MISSING] * (len(keys) - len(offsets))

    def release(self, *, conn: int, keys: list[str]) -> None:
        with self._cond:
            for key in keys:
                self._unpin(conn=conn, key=key)

    # ------------------------------------------------------------ writes

    def reserve(self, *, conn: int, keys: list[str], nbytes: int) -> list[int]:
        """A slot per key the caller must fill and commit, or PRESENT/NO_SPACE."""
        for key in keys:
            validate_key(key)
        with self._cond:
            for key in keys:
                if key in self._entries:
                    self._entries.move_to_end(key)
            candidates = [key for key in keys if key not in self._entries]
        in_lower = set()
        if self.lower is not None and candidates:
            in_lower = {
                key
                for key, hit in zip(candidates, self.lower.contains_many(candidates))
                if hit
            }
        offsets = []
        with self._cond:
            for key in keys:
                if key in self._entries or key in in_lower:
                    offsets.append(PRESENT)
                else:
                    entry = self._insert_filling(conn=conn, key=key, nbytes=nbytes)
                    offsets.append(NO_SPACE if entry is None else entry.offset)
        return offsets

    def commit(self, *, conn: int, keys: list[str]) -> None:
        with self._cond:
            for key in keys:
                entry = self._entries.get(key)
                if entry is None or entry.ready or entry.owner != conn:
                    continue
                entry.ready = True
                entry.clean = self.lower is None
                if not entry.clean:
                    self._dirty_bytes += entry.nbytes
                self._unpin(conn=conn, key=key)
            self._cond.notify_all()

    def abort(self, *, conn: int, keys: list[str]) -> None:
        with self._cond:
            for key in keys:
                entry = self._entries.get(key)
                if entry is not None and not entry.ready and entry.owner == conn:
                    self._unpin(conn=conn, key=key)
                    self._remove(key=key, entry=entry)
            self._cond.notify_all()

    # ------------------------------------------------------------ lifecycle

    def disconnect(self, conn: int) -> None:
        """Drop every pin and unfinished reservation of a closed client."""
        with self._cond:
            for key, count in self._conn_pins.pop(conn, Counter()).items():
                entry = self._entries.get(key)
                if entry is None:
                    continue
                entry.pins -= count
                if not entry.ready and entry.owner == conn:
                    self._remove(key=key, entry=entry)
            self._cond.notify_all()

    def clear(self) -> None:
        with self._cond:
            for key, entry in list(self._entries.items()):
                if entry.ready and entry.pins == 0:
                    self._remove(key=key, entry=entry)
        if self.lower is not None:
            self.lower.clear()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if self._cleaner is not None:
            self._cleaner.join()
        if self.lower is not None:
            self.lower.close()

    def stats(self) -> StoreStats:
        lower_pages, lower_bytes = (0, 0) if self.lower is None else self.lower.usage()
        with self._cond:
            return StoreStats(
                arena_bytes=self.arena_bytes,
                arena_carved_bytes=self._carved_bytes,
                shm_pages=len(self._entries),
                shm_bytes=self._shm_bytes,
                dirty_bytes=self._dirty_bytes,
                lower_tier="none" if self.lower is None else self.lower.name,
                lower_pages=lower_pages,
                lower_bytes=lower_bytes,
                lower_capacity_bytes=(
                    0 if self.lower is None else self.lower.capacity_bytes
                ),
                shm_hits=self._counters["shm_hits"],
                lower_hits=self._counters["lower_hits"],
                misses=self._counters["misses"],
                inserts=self._counters["inserts"],
                no_space=self._counters["no_space"],
                evictions=self._counters["evictions"],
                demotions=self._counters["demotions"],
                demote_failures=self._counters["demote_failures"],
            )

    # ------------------------------------------------------------ acquire helpers

    def _pin_arena_run(self, *, conn: int, keys: list[str], nbytes: int) -> list[int]:
        """Offsets of the leading keys already ready in the arena, pinned."""
        offsets = []
        with self._cond:
            for key in keys:
                entry = self._wait_for_fill(key)
                if entry is None or not entry.ready or entry.nbytes != nbytes:
                    break
                self._pin(conn=conn, key=key, entry=entry)
                self._entries.move_to_end(key)
                self._counters["shm_hits"] += 1
                offsets.append(entry.offset)
        return offsets

    def _promote_run(self, *, conn: int, keys: list[str], nbytes: int) -> list[int]:
        """Offsets of the leading keys found in layer 2, copied into new slots."""
        in_lower = self.lower.contains_many(keys)
        run = keys[: in_lower.index(False) if False in in_lower else len(keys)]
        # Each item is (key, entry, promote): promote is False when another
        # client put the page in the arena since the arena pass.
        planned: list[tuple[str, _Entry, bool]] = []
        with self._cond:
            for key in run:
                entry = self._entries.get(key)
                if entry is not None:
                    if not entry.ready or entry.nbytes != nbytes:
                        break
                    self._pin(conn=conn, key=key, entry=entry)
                    planned.append((key, entry, False))
                    continue
                entry = self._insert_filling(conn=conn, key=key, nbytes=nbytes)
                if entry is None:
                    break
                planned.append((key, entry, True))
        to_read = [(key, entry) for key, entry, promote in planned if promote]
        read_ok = self.lower.read_many(
            keys=[key for key, _ in to_read],
            slots=[self._slot(entry) for _, entry in to_read],
        )
        ok_by_key = {key: ok for (key, _), ok in zip(to_read, read_ok)}
        return self._finish_promotion(conn=conn, planned=planned, ok_by_key=ok_by_key)

    def _finish_promotion(
        self,
        *,
        conn: int,
        planned: list[tuple[str, _Entry, bool]],
        ok_by_key: dict[str, bool],
    ) -> list[int]:
        offsets = []
        run_open = True
        with self._cond:
            for key, entry, promote in planned:
                ok = not promote or ok_by_key[key]
                if promote and ok:
                    entry.ready = True
                    entry.clean = True
                    self._counters["lower_hits"] += 1
                elif promote:
                    self._unpin(conn=conn, key=key)
                    self._remove(key=key, entry=entry)
                if run_open and ok:
                    offsets.append(entry.offset)
                    continue
                run_open = False
                # Past the first failure the run is over: keep the pages that
                # did arrive cached, but release them.
                if ok:
                    self._unpin(conn=conn, key=key)
            self._cond.notify_all()
        return offsets

    def _wait_for_fill(self, key: str) -> _Entry | None:
        """The entry for ``key`` once no other client is filling it. Lock held."""
        deadline = time.monotonic() + self.fill_wait_s
        while True:
            entry = self._entries.get(key)
            if entry is None or entry.ready:
                return entry
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return entry
            self._cond.wait(remaining)

    # ------------------------------------------------------------ slot management

    def _insert_filling(self, *, conn: int, key: str, nbytes: int) -> _Entry | None:
        """A pinned, not-ready entry owned by ``conn``; None when out of space."""
        offset = self._alloc_slot(nbytes)
        if offset is None:
            self._counters["no_space"] += 1
            self._cond.notify_all()
            return None
        entry = _Entry(offset=offset, nbytes=nbytes, owner=conn)
        self._entries[key] = entry
        self._pin(conn=conn, key=key, entry=entry)
        self._shm_bytes += nbytes
        self._counters["inserts"] += 1
        return entry

    def _alloc_slot(self, nbytes: int) -> int | None:
        slot_bytes = -(-nbytes // SLOT_ALIGN) * SLOT_ALIGN
        size_class = self._classes.get(slot_bytes)
        if size_class is None:
            size_class = self._classes[slot_bytes] = _SizeClass(slot_bytes=slot_bytes)
        if size_class.free or self._carve(size_class) or self._evict_one(slot_bytes):
            return size_class.free.pop()
        return None

    def _carve(self, size_class: _SizeClass) -> bool:
        slot_bytes = size_class.slot_bytes
        num_slots = max(self.chunk_bytes // slot_bytes, 1)
        num_slots = min(
            num_slots, (self.arena_bytes - self._carved_bytes) // slot_bytes
        )
        if num_slots <= 0:
            return False
        base = self._carved_bytes
        size_class.free.extend(base + i * slot_bytes for i in range(num_slots))
        size_class.num_slots += num_slots
        self._carved_bytes += num_slots * slot_bytes
        return True

    def _evict_one(self, slot_bytes: int) -> bool:
        for key, entry in self._entries.items():
            if (
                entry.ready
                and entry.clean
                and entry.pins == 0
                and self._slot_bytes(entry) == slot_bytes
            ):
                self._remove(key=key, entry=entry)
                self._counters["evictions"] += 1
                return True
        return False

    def _remove(self, *, key: str, entry: _Entry) -> None:
        del self._entries[key]
        self._shm_bytes -= entry.nbytes
        if entry.ready and not entry.clean:
            self._dirty_bytes -= entry.nbytes
        self._classes[self._slot_bytes(entry)].free.append(entry.offset)

    def _slot(self, entry: _Entry) -> ArenaSlot:
        return ArenaSlot(
            view=self.arena[entry.offset : entry.offset + entry.nbytes],
            addr=self.arena_addr + entry.offset,
        )

    @staticmethod
    def _slot_bytes(entry: _Entry) -> int:
        return -(-entry.nbytes // SLOT_ALIGN) * SLOT_ALIGN

    def _is_ready(self, key: str) -> bool:
        entry = self._entries.get(key)
        return entry is not None and entry.ready

    def _pin(self, *, conn: int, key: str, entry: _Entry) -> None:
        entry.pins += 1
        self._conn_pins.setdefault(conn, Counter())[key] += 1

    def _unpin(self, *, conn: int, key: str) -> None:
        pins = self._conn_pins.get(conn)
        if pins is None or pins[key] <= 0:
            return
        pins[key] -= 1
        if pins[key] == 0:
            del pins[key]
        entry = self._entries.get(key)
        if entry is not None:
            entry.pins -= 1

    # ------------------------------------------------------------ cleaner

    def clean_once(self) -> int:
        """Demote one batch of the oldest dirty pages to layer 2; returns how many."""
        with self._cond:
            batch = self._pick_dirty()
        if not batch:
            return 0
        try:
            written = self.lower.write_many(
                keys=[key for key, _ in batch],
                slots=[self._slot(entry) for _, entry in batch],
            )
        except Exception:
            logger.exception("Failed to demote %d pages to layer 2", len(batch))
            written = [False] * len(batch)
        with self._cond:
            for (key, entry), ok in zip(batch, written):
                entry.pins -= 1
                # A page layer 2 refused is marked clean anyway, so a full or
                # failing layer 2 cannot pin the whole arena; it is then simply
                # lost on eviction.
                if self._entries.get(key) is entry and not entry.clean:
                    entry.clean = True
                    self._dirty_bytes -= entry.nbytes
                    self._counters["demotions" if ok else "demote_failures"] += 1
            self._cond.notify_all()
        return len(batch)

    def _cleaner_loop(self) -> None:
        while True:
            with self._cond:
                if self._closed:
                    return
            if self.clean_once() == 0:
                with self._cond:
                    if not self._closed:
                        self._cond.wait(0.5)

    def _pick_dirty(self) -> list[tuple[str, _Entry]]:
        """Pin the dirty pages among the oldest ``clean_target_bytes``. Lock held."""
        batch = []
        scanned_bytes = 0
        for key, entry in self._entries.items():
            if scanned_bytes >= self.clean_target_bytes or len(batch) >= _CLEANER_BATCH:
                break
            if not entry.ready:
                continue
            scanned_bytes += entry.nbytes
            if not entry.clean:
                entry.pins += 1
                batch.append((key, entry))
        return batch


def arena_address(arena: memoryview) -> int:
    """Address of the arena's first byte, for zero-copy layer-2 transfers."""
    return ctypes.addressof(ctypes.c_char.from_buffer(arena))

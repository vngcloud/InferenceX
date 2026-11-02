# SPDX-License-Identifier: Apache-2.0
"""Index and allocator of the node cache, owned by the daemon.

The hot tier is one shared arena (a file under /dev/shm) carved into fixed-size
slots per page size. Clients copy page bytes in and out of the arena
themselves; this store only hands out offsets and tracks who holds which slot.

Life of a page:

    reserve -> (client copies host page into the slot) -> commit -> ready
    ready, dirty --cleaner writes <key>.bin--> ready, clean --LRU--> evicted
    evicted, on SSD --acquire--> promoted back into a slot

Only clean pages are evicted, so a page leaves the arena only after it has a
copy on SSD (or when no SSD tier is configured). The cleaner keeps the oldest
``clean_ratio`` of the arena clean, so eviction rarely has to wait for it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, OrderedDict

import msgspec

from sglang_node_cache.protocol import MISSING, NO_SPACE, PRESENT, StoreStats
from sglang_node_cache.ssd_tier import SsdTier, validate_key

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


class NodeCacheStore:
    def __init__(
        self,
        *,
        arena: memoryview,
        chunk_bytes: int,
        ssd: SsdTier | None,
        clean_ratio: float = 0.1,
        fill_wait_s: float = 2.0,
        start_cleaner: bool = True,
    ):
        self.arena = arena
        self.arena_bytes = len(arena)
        self.chunk_bytes = chunk_bytes
        self.ssd = ssd
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
        if ssd is not None and start_cleaner:
            self._cleaner = threading.Thread(
                target=self._cleaner_loop, name="node-cache-cleaner", daemon=True
            )
            self._cleaner.start()

    # ------------------------------------------------------------ queries

    def exists(self, keys: list[str]) -> list[bool]:
        with self._cond:
            in_shm = [self._is_ready(key) for key in keys]
        return [
            hit or (self.ssd is not None and self.ssd.contains(key))
            for hit, key in zip(in_shm, keys)
        ]

    def acquire(self, *, conn: int, keys: list[str], nbytes: int) -> list[int]:
        """Pin the leading run of present pages for reading, promoting SSD hits."""
        offsets = []
        for key in keys:
            offset = self._acquire_one(conn=conn, key=key, nbytes=nbytes)
            offsets.append(offset)
            if offset < 0:
                break
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
        offsets = []
        with self._cond:
            for key in keys:
                if key in self._entries:
                    self._entries.move_to_end(key)
                    offsets.append(PRESENT)
                elif self.ssd is not None and self.ssd.contains(key):
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
                entry.clean = self.ssd is None
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
        if self.ssd is not None:
            self.ssd.clear()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if self._cleaner is not None:
            self._cleaner.join()

    def stats(self) -> StoreStats:
        ssd_pages, ssd_bytes = (0, 0) if self.ssd is None else self.ssd.usage()
        with self._cond:
            return StoreStats(
                arena_bytes=self.arena_bytes,
                arena_carved_bytes=self._carved_bytes,
                shm_pages=len(self._entries),
                shm_bytes=self._shm_bytes,
                dirty_bytes=self._dirty_bytes,
                ssd_pages=ssd_pages,
                ssd_bytes=ssd_bytes,
                ssd_capacity_bytes=0 if self.ssd is None else self.ssd.capacity_bytes,
                shm_hits=self._counters["shm_hits"],
                ssd_hits=self._counters["ssd_hits"],
                misses=self._counters["misses"],
                inserts=self._counters["inserts"],
                no_space=self._counters["no_space"],
                evictions=self._counters["evictions"],
                demotions=self._counters["demotions"],
            )

    # ------------------------------------------------------------ acquire helpers

    def _acquire_one(self, *, conn: int, key: str, nbytes: int) -> int:
        with self._cond:
            entry = self._wait_for_fill(key)
            if entry is not None:
                if entry.ready and entry.nbytes == nbytes:
                    self._pin(conn=conn, key=key, entry=entry)
                    self._entries.move_to_end(key)
                    self._counters["shm_hits"] += 1
                    return entry.offset
                self._counters["misses"] += 1
                return MISSING
            if self.ssd is None or self.ssd.size_of(key) != nbytes:
                self._counters["misses"] += 1
                return MISSING
            entry = self._insert_filling(conn=conn, key=key, nbytes=nbytes)
            if entry is None:
                return MISSING
        return self._promote(conn=conn, key=key, entry=entry)

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

    def _promote(self, *, conn: int, key: str, entry: _Entry) -> int:
        """Copy an SSD page into its new slot; the caller keeps the pin."""
        try:
            ok = self.ssd.read_into(key=key, dst=self._slot(entry))
        except OSError:
            logger.exception("Failed to promote node-cache page %s", key)
            ok = False
        with self._cond:
            if not ok:
                self._unpin(conn=conn, key=key)
                self._remove(key=key, entry=entry)
                self._counters["misses"] += 1
                self._cond.notify_all()
                return MISSING
            entry.ready = True
            entry.clean = True
            self._counters["ssd_hits"] += 1
            self._cond.notify_all()
            return entry.offset

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

    def _slot(self, entry: _Entry) -> memoryview:
        return self.arena[entry.offset : entry.offset + entry.nbytes]

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
        """Persist one batch of the oldest dirty pages; returns how many."""
        with self._cond:
            batch = self._pick_dirty()
        for key, entry in batch:
            self._demote(key=key, entry=entry)
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

    def _demote(self, *, key: str, entry: _Entry) -> None:
        try:
            self.ssd.write(key=key, src=self._slot(entry))
            written = True
        except OSError:
            # Treat the page as clean anyway so a full or failing SSD cannot
            # pin the whole arena; the page is then simply lost on eviction.
            logger.exception("Failed to demote node-cache page %s", key)
            written = False
        with self._cond:
            entry.pins -= 1
            if self._entries.get(key) is entry and not entry.clean:
                entry.clean = True
                self._dirty_bytes -= entry.nbytes
                if written:
                    self._counters["demotions"] += 1
            self._cond.notify_all()

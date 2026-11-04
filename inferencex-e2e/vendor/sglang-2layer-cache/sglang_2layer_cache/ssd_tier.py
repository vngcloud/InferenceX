# SPDX-License-Identifier: Apache-2.0
"""Layer 2 on local SSD files: one ``<key>.bin`` file per page on SSD/NVMe.

Files are written to a temporary name and renamed, so a crash never leaves a
torn page behind, and the index is rebuilt from the directory on start-up.
The tier is bounded by ``capacity_bytes`` and drops least-recently-used files.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import zlib
from collections import OrderedDict

from sglang_2layer_cache.lower_tier import ArenaSlot

logger = logging.getLogger(__name__)

_KEY_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,200}")
_SUFFIX = ".bin"
_TMP_SUFFIX = ".tmp"


def validate_key(key: str) -> None:
    # Keys become file names, so they must not carry separators or dots-only names.
    if not _KEY_PATTERN.fullmatch(key) or key in (".", ".."):
        raise ValueError(f"Invalid two-layer-cache key: {key!r}")


class SsdTier:
    name = "ssd"

    def __init__(self, *, root: str, capacity_bytes: int):
        if capacity_bytes <= 0:
            raise ValueError(
                f"SSD tier capacity must be positive, got {capacity_bytes}"
            )
        self.root = root
        self.capacity_bytes = capacity_bytes
        self._lock = threading.Lock()
        # key -> nbytes, oldest first.
        self._files: OrderedDict[str, int] = OrderedDict()
        self._used_bytes = 0
        os.makedirs(root, exist_ok=True)
        self._rebuild_index()

    def contains_many(self, keys: list[str]) -> list[bool]:
        with self._lock:
            return [key in self._files for key in keys]

    def read_many(self, *, keys: list[str], slots: list[ArenaSlot]) -> list[bool]:
        results = []
        for key, slot in zip(keys, slots):
            try:
                results.append(self.read_into(key=key, dst=slot.view))
            except OSError:
                logger.exception("Failed to read SSD page %s", key)
                results.append(False)
        return results

    def write_many(self, *, keys: list[str], slots: list[ArenaSlot]) -> list[bool]:
        results = []
        for key, slot in zip(keys, slots):
            try:
                self.write(key=key, src=slot.view)
                results.append(True)
            except OSError:
                logger.exception("Failed to write SSD page %s", key)
                results.append(False)
        return results

    def close(self) -> None:
        pass

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._files

    def size_of(self, key: str) -> int | None:
        with self._lock:
            return self._files.get(key)

    def read_into(self, *, key: str, dst: memoryview) -> bool:
        """Fill ``dst`` with the page; False when absent or of another size."""
        with self._lock:
            nbytes = self._files.get(key)
            if nbytes is None or nbytes != len(dst):
                return False
            self._files.move_to_end(key)
        try:
            with open(self._path(key), "rb", buffering=0) as f:
                got = f.readinto(dst)
                # Each page is read once per promotion; keep it out of the page cache.
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except FileNotFoundError:
            # Evicted between the index lookup and the open.
            return False
        return got == len(dst)

    def write(self, *, key: str, src: memoryview) -> list[str]:
        """Persist one page and return the keys evicted to make room."""
        nbytes = len(src)
        if nbytes > self.capacity_bytes:
            return []
        path = self._path(key)
        tmp_path = f"{path}.{threading.get_ident()}{_TMP_SUFFIX}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp_path, "wb", buffering=0) as f:
            written = f.write(src)
        if written != nbytes:
            os.unlink(tmp_path)
            raise OSError(
                f"Short write for two-layer-cache page {key}: {written}/{nbytes}"
            )
        os.replace(tmp_path, path)
        with self._lock:
            previous = self._files.pop(key, None)
            if previous is not None:
                self._used_bytes -= previous
            self._files[key] = nbytes
            self._used_bytes += nbytes
            victims = self._pop_over_capacity()
        for victim in victims:
            self._unlink(victim)
        return victims

    def clear(self) -> list[str]:
        with self._lock:
            keys = list(self._files)
            self._files.clear()
            self._used_bytes = 0
        for key in keys:
            self._unlink(key)
        return keys

    def usage(self) -> tuple[int, int]:
        with self._lock:
            return len(self._files), self._used_bytes

    def _pop_over_capacity(self) -> list[str]:
        victims = []
        while self._used_bytes > self.capacity_bytes and self._files:
            key, nbytes = self._files.popitem(last=False)
            self._used_bytes -= nbytes
            victims.append(key)
        return victims

    def _path(self, key: str) -> str:
        bucket = f"{zlib.crc32(key.encode()) & 0xFF:02x}"
        return os.path.join(self.root, bucket, key + _SUFFIX)

    def _unlink(self, key: str) -> None:
        try:
            os.unlink(self._path(key))
        except FileNotFoundError:
            pass

    def _rebuild_index(self) -> None:
        found = []
        for bucket in os.scandir(self.root):
            if not bucket.is_dir():
                continue
            for entry in os.scandir(bucket.path):
                if entry.name.endswith(_TMP_SUFFIX):
                    os.unlink(entry.path)
                    continue
                if not entry.name.endswith(_SUFFIX):
                    continue
                stat = entry.stat()
                found.append((stat.st_mtime, entry.name[: -len(_SUFFIX)], stat.st_size))
        for _, key, nbytes in sorted(found):
            self._files[key] = nbytes
            self._used_bytes += nbytes
        victims = self._pop_over_capacity()
        for victim in victims:
            self._unlink(victim)
        logger.info(
            "SSD tier at %s: %d pages, %.1f GiB recovered",
            self.root,
            len(self._files),
            self._used_bytes / (1 << 30),
        )

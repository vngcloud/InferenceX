# SPDX-License-Identifier: Apache-2.0
"""Layer 2 on a Mooncake store, reached over TCP or RDMA.

The daemon is one Mooncake client for the whole node. It registers the arena
with the Mooncake transfer engine, so pages move between the arena and
Mooncake zero-copy (batch_put_from / batch_get_into on arena addresses).

Two deployments:
- external: a Mooncake cluster whose storage nodes keep the pages (their own
  DRAM plus SSD offload). The daemon contributes no segment (segment_bytes=0).
- self-hosted: the daemon contributes a DRAM segment and spills it to a local
  SSD through Mooncake's own offload (ssd_offload_path).
"""

from __future__ import annotations

import logging

import msgspec

from sglang_2layer_cache.lower_tier import ArenaSlot

logger = logging.getLogger(__name__)

# Same value as sglang's MooncakeStore: zero-copy calls never use it.
_LOCAL_BUFFER_BYTES = 16 << 20


class MooncakeTierConfig(msgspec.Struct, frozen=True):
    master_server_address: str
    protocol: str = "tcp"
    metadata_server: str = "P2PHANDSHAKE"
    local_hostname: str = "127.0.0.1"
    device_name: str = ""
    # Bytes of DRAM this daemon adds to the Mooncake pool; 0 for a pure client.
    segment_bytes: int = 0
    ssd_offload_path: str | None = None


class MooncakeTier:
    name = "mooncake"
    capacity_bytes = 0

    def __init__(
        self, *, config: MooncakeTierConfig, arena_addr: int, arena_bytes: int
    ):
        from mooncake.store import MooncakeDistributedStore

        if config.protocol not in ("tcp", "rdma"):
            raise ValueError(
                f"Mooncake protocol must be tcp or rdma, got {config.protocol!r}"
            )
        self.config = config
        self._store = MooncakeDistributedStore()
        _setup(store=self._store, config=config)
        ret = self._store.register_buffer(arena_addr, arena_bytes)
        if ret != 0:
            raise RuntimeError(f"Mooncake register_buffer of the arena failed: {ret}")
        logger.info(
            "Layer 2 = Mooncake at %s over %s (segment %.1f GiB, SSD offload %s)",
            config.master_server_address,
            config.protocol,
            config.segment_bytes / (1 << 30),
            config.ssd_offload_path or "off",
        )

    def contains_many(self, keys: list[str]) -> list[bool]:
        if not keys:
            return []
        return [ret == 1 for ret in self._store.batch_is_exist(keys)]

    def read_many(self, *, keys: list[str], slots: list[ArenaSlot]) -> list[bool]:
        if not keys:
            return []
        sizes = [len(slot.view) for slot in slots]
        # Each result is the bytes read, or a negative error code.
        results = self._store.batch_get_into(keys, [s.addr for s in slots], sizes)
        return [ret == size for ret, size in zip(results, sizes)]

    def write_many(self, *, keys: list[str], slots: list[ArenaSlot]) -> list[bool]:
        if not keys:
            return []
        sizes = [len(slot.view) for slot in slots]
        # Each result is 0 on success, or a negative error code.
        results = self._store.batch_put_from(keys, [s.addr for s in slots], sizes)
        failed = [(key, ret) for key, ret in zip(keys, results) if ret != 0]
        if failed:
            logger.debug(
                "Mooncake put failed for %d pages, e.g. %s", len(failed), failed[0]
            )
        return [ret == 0 for ret in results]

    def clear(self) -> None:
        # Mooncake may be shared by other nodes; the store only calls this on
        # an explicit clear request.
        self._store.remove_all()

    def usage(self) -> tuple[int, int]:
        return 0, 0

    def close(self) -> None:
        self._store.close()


def _setup(*, store, config: MooncakeTierConfig) -> None:
    kwargs = {}
    if config.ssd_offload_path:
        kwargs = {
            "enable_ssd_offload": True,
            "ssd_offload_path": config.ssd_offload_path,
        }
    ret = store.setup(
        config.local_hostname,
        config.metadata_server,
        config.segment_bytes,
        _LOCAL_BUFFER_BYTES,
        config.protocol,
        config.device_name,
        config.master_server_address,
        **kwargs,
    )
    if ret != 0:
        raise RuntimeError(f"Mooncake setup failed with code {ret}")

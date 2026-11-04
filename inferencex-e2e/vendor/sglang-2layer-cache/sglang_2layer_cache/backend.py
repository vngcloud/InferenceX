# SPDX-License-Identifier: Apache-2.0
"""HiCache storage backend that talks to the two-layer-cache daemon.

Loaded by SGLang as a dynamic backend; no SGLang source change is needed:

    --hicache-storage-backend dynamic
    --hicache-storage-backend-extra-config '{
        "backend_name": "two_layer_cache",
        "module_path": "sglang_2layer_cache.backend",
        "class_name": "TwoLayerCacheStorage",
        "interface_v1": 1,
        "two_layer_socket": "/dev/shm/sglang_2layer_cache/default.sock"}'

``two_layer_socket`` falls back to the TWO_LAYER_CACHE_SOCKET environment variable.

Page bytes move by memcpy between the rank's host pool (L2) and the shared
arena; the daemon only exchanges keys and offsets. Pages that are identical on
every rank (MLA KV, DSA indexer, MLA draft) get rank-free keys, so one copy on
the node serves every rank, DP group and SGLang instance of the same model.
"""

from __future__ import annotations

import logging
import os
import re
import zlib
from typing import TYPE_CHECKING, Any, List, Optional

import msgspec
import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

from sglang_2layer_cache.client import TwoLayerClient
from sglang_2layer_cache.daemon import DEFAULT_SOCKET
from sglang_2layer_cache.protocol import PRESENT

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool_host import HostKVCache

logger = logging.getLogger(__name__)

# Read when the extra config names no socket, e.g. for launchers that only
# template the environment.
SOCKET_ENV = "TWO_LAYER_CACHE_SOCKET"
_KV_POOL = str(PoolName.KV)
# Pools that hold per-rank shards even when the KV pool is MLA-replicated.
_RANK_SHARDED_POOLS = frozenset({str(PoolName.MAMBA)})
# Pools whose layout follows their own host pool class rather than the target's.
_LAYOUT_FOLLOWS_POOL = frozenset(
    {str(PoolName.DRAFT), str(PoolName.DRAFT_SWA), str(PoolName.SWA)}
)


class TwoLayerCacheConfig(msgspec.Struct, frozen=True):
    socket_path: str
    namespace: str
    rpc_timeout_s: float
    clear_on_flush: bool

    @classmethod
    def from_storage_config(
        cls, storage_config: HiCacheStorageConfig
    ) -> TwoLayerCacheConfig:
        extra = storage_config.extra_config or {}
        default_socket = os.environ.get(SOCKET_ENV, DEFAULT_SOCKET)
        return cls(
            socket_path=extra.get("two_layer_socket", default_socket),
            namespace=_namespace(
                model_name=storage_config.model_name,
                tag=extra.get("two_layer_namespace", ""),
            ),
            rpc_timeout_s=float(extra.get("two_layer_rpc_timeout_s", 30.0)),
            clear_on_flush=_as_bool(extra.get("two_layer_clear_on_flush", False)),
        )


class _PageGeometry(msgspec.Struct, frozen=True):
    dtype: torch.dtype
    nbytes: int
    # Appended to every key of the pool; empty when all ranks share the page.
    rank_tag: str


class TwoLayerCacheStorage(HiCacheStorage):
    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        kwargs: Optional[dict] = None,
        client: Optional[TwoLayerClient] = None,
    ):
        self.storage_config = storage_config
        self.config = TwoLayerCacheConfig.from_storage_config(storage_config)
        self.client = client or TwoLayerClient(
            socket_path=self.config.socket_path, timeout_s=self.config.rpc_timeout_s
        )
        self.mem_pool_host: Optional[HostKVCache] = None
        self.registered_pools: dict[str, HostKVCache] = {}
        self._geometry: dict[str, _PageGeometry] = {}
        logger.info(
            "Node-cache HiCache backend: socket %s, namespace %s",
            self.config.socket_path,
            self.config.namespace,
        )

    # ------------------------------------------------------------ registration

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self.mem_pool_host = mem_pool_host
        self._register_pool(name=_KV_POOL, host_pool=mem_pool_host)

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        self._register_pool(name=str(host_pool_name), host_pool=host_pool)

    # ------------------------------------------------------------ existence

    def exists(self, key: str) -> bool:
        return self.batch_exists([key]) == 1

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        flags = self._exists_flags(self._store_keys(pool_name=_KV_POOL, keys=keys))
        return _leading_true(flags)

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        transfers = pool_transfers or []
        query = self._store_keys(pool_name=_KV_POOL, keys=keys)
        for transfer in transfers:
            query += self._store_keys(pool_name=str(transfer.name), keys=keys)
        flags = self._exists_flags(query)

        kv_pages = _leading_true(flags[: len(keys)])
        hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}
        restorable = list(range(1, kv_pages + 1))
        for i, transfer in enumerate(transfers):
            if not restorable:
                break
            start = (i + 1) * len(keys)
            present = flags[start : start + kv_pages]
            pool_restorable = _restorable_prefixes(present=present, transfer=transfer)
            if pool_restorable:
                hit_count[transfer.name] = pool_restorable[-1]
            allowed = set(pool_restorable)
            restorable = [p for p in restorable if p in allowed]
        final_pages = restorable[-1] if restorable else 0
        return PoolTransferResult(final_pages, hit_count, restorable)

    # ------------------------------------------------------------ zero-copy IO

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        return self._get_pages(
            pool_name=_KV_POOL,
            host_pool=self.mem_pool_host,
            keys=keys,
            host_indices=host_indices,
        )

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        return self._put_pages(
            pool_name=_KV_POOL,
            host_pool=self.mem_pool_host,
            keys=keys,
            host_indices=host_indices,
        )

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return {
            transfer.name: self._get_pages(
                pool_name=str(transfer.name),
                host_pool=self.registered_pools[str(transfer.name)],
                keys=transfer.keys or [],
                host_indices=transfer.host_indices,
            )
            for transfer in transfers
        }

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return {
            transfer.name: self._put_pages(
                pool_name=str(transfer.name),
                host_pool=self.registered_pools[str(transfer.name)],
                keys=transfer.keys or [],
                host_indices=transfer.host_indices,
            )
            for transfer in transfers
        }

    # ------------------------------------------------------------ legacy API

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        raise NotImplementedError(
            "two_layer_cache needs interface_v1=1 in the extra config"
        )

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        raise NotImplementedError(
            "two_layer_cache needs interface_v1=1 in the extra config"
        )

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError(
            "two_layer_cache needs interface_v1=1 in the extra config"
        )

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError(
            "two_layer_cache needs interface_v1=1 in the extra config"
        )

    # ------------------------------------------------------------ lifecycle

    def clear(self) -> None:
        # The cache is shared by every rank and instance on the node, so one
        # server's flush must not wipe it unless explicitly configured.
        if self.config.clear_on_flush:
            self.client.clear()

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------ helpers

    def _register_pool(self, *, name: str, host_pool: HostKVCache) -> None:
        self.registered_pools[name] = host_pool
        sample = host_pool.get_data_page(0, flat=True)
        self._geometry[name] = _PageGeometry(
            dtype=sample.dtype,
            nbytes=sample.numel() * sample.element_size(),
            rank_tag=self._rank_tag(name=name, host_pool=host_pool),
        )

    def _rank_tag(self, *, name: str, host_pool: HostKVCache) -> str:
        cfg = self.storage_config
        shared = cfg.is_mla_model and name not in _RANK_SHARDED_POOLS
        if name in _LAYOUT_FOLLOWS_POOL and isinstance(host_pool, MHATokenToKVPoolHost):
            shared = False
        tag = "" if shared else f".tp{cfg.tp_rank}of{cfg.tp_size}"
        if cfg.pp_size > 1:
            tag += f".pp{cfg.pp_rank}of{cfg.pp_size}"
        if cfg.attn_cp_size > 1:
            tag += f".cp{cfg.attn_cp_rank}of{cfg.attn_cp_size}"
        return tag

    def _store_keys(self, *, pool_name: str, keys: List[str]) -> List[str]:
        rank_tag = self._geometry[pool_name].rank_tag
        prefix = f"{self.config.namespace}.{pool_name}."
        return [f"{prefix}{key}{rank_tag}" for key in keys]

    def _exists_flags(self, store_keys: List[str]) -> List[bool]:
        if not store_keys:
            return []
        try:
            return self.client.exists(store_keys)
        except (ConnectionError, OSError):
            logger.warning("Node-cache daemon unreachable; treating pages as missing")
            return [False] * len(store_keys)

    def _get_pages(
        self,
        *,
        pool_name: str,
        host_pool: HostKVCache,
        keys: List[str],
        host_indices: torch.Tensor,
    ) -> List[bool]:
        if not keys:
            return []
        geometry = self._geometry[pool_name]
        store_keys = self._store_keys(pool_name=pool_name, keys=keys)
        try:
            offsets = self.client.acquire(keys=store_keys, nbytes=geometry.nbytes)
        except (ConnectionError, OSError):
            logger.warning("Node-cache daemon unreachable; prefetch skipped")
            return [False] * len(keys)
        hits = [offset >= 0 for offset in offsets]
        arena = self.client.arena
        page_size = host_pool.page_size
        try:
            for i, offset in enumerate(offsets):
                if offset < 0:
                    break
                page = arena[offset : offset + geometry.nbytes].view(geometry.dtype)
                host_pool.set_from_flat_data_page(
                    int(host_indices[i * page_size]), page
                )
        finally:
            self._release([key for key, hit in zip(store_keys, hits) if hit])
        return hits

    def _put_pages(
        self,
        *,
        pool_name: str,
        host_pool: HostKVCache,
        keys: List[str],
        host_indices: torch.Tensor,
    ) -> List[bool]:
        if not keys:
            return []
        geometry = self._geometry[pool_name]
        store_keys = self._store_keys(pool_name=pool_name, keys=keys)
        try:
            offsets = self.client.reserve(keys=store_keys, nbytes=geometry.nbytes)
        except (ConnectionError, OSError):
            logger.warning("Node-cache daemon unreachable; backup skipped")
            return [False] * len(keys)
        arena = self.client.arena
        page_size = host_pool.page_size
        filled: List[str] = []
        try:
            for i, offset in enumerate(offsets):
                if offset < 0:
                    continue
                page = host_pool.get_data_page(
                    int(host_indices[i * page_size]), flat=True
                )
                arena[offset : offset + geometry.nbytes].view(geometry.dtype).copy_(
                    page
                )
                filled.append(store_keys[i])
        except BaseException:
            filled_set = set(filled)
            self.client.abort(
                [
                    key
                    for key, offset in zip(store_keys, offsets)
                    if offset >= 0 and key not in filled_set
                ]
            )
            raise
        finally:
            if filled:
                self.client.commit(filled)
        return [offset >= 0 or offset == PRESENT for offset in offsets]

    def _release(self, store_keys: List[str]) -> None:
        if not store_keys:
            return
        try:
            self.client.release(store_keys)
        except (ConnectionError, OSError):
            # The daemon drops every pin of a broken connection by itself.
            pass


def _namespace(*, model_name: Optional[str], tag: str) -> str:
    raw = f"{model_name or 'model'}{'@' + tag if tag else ''}"
    readable = re.sub(r"[^A-Za-z0-9_-]", "-", raw)[-48:]
    # Truncation can merge two models' prefixes; the checksum keeps them apart.
    return f"{readable}-{zlib.crc32(raw.encode()):08x}"


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def _leading_true(flags: List[bool]) -> int:
    return flags.index(False) if False in flags else len(flags)


def _restorable_prefixes(*, present: List[bool], transfer: PoolTransfer) -> List[int]:
    """Prefix lengths this pool can serve, ascending; mirrors MooncakeStore."""
    if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
        return list(range(1, _leading_true(present) + 1))
    if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
        trailing = max(1, len(transfer.keys) if transfer.keys else 1)
        return [
            prefix_len
            for prefix_len in range(1, len(present) + 1)
            if all(present[max(0, prefix_len - trailing) : prefix_len])
        ]
    raise ValueError(f"Unsupported pool hit policy: {transfer.hit_policy}")

# sglang-2layer-cache

A node-shared, two-layer HiCache L3 for SGLang. SGLang sees one storage
backend. Behind it, one daemon per node manages two layers.

```
GPU (L1)  <-  L2, private to each rank (--hicache-host-memory-mode buffer_only)
                 ^ memcpy (the backend, inside each rank)
     +---------------- two-layer cache daemon (one per node) ----------------+
     |  layer 1: memfd DRAM arena, mapped by every rank                     |
     |     v demote before eviction                  ^ promote on hit       |
     |  layer 2: Mooncake store over TCP or RDMA (zero-copy from the arena) |
     |           or local SSD files (<key>.bin)                             |
     +-----------------------------------------------------------------------+
```

Derived from `sglang-node-cache`: layer 1, the socket protocol and the SGLang
backend are the same. What is new is layer 2 behind a batched `LowerTier`
interface, with a Mooncake implementation.

## Why the Mooncake client lives in the daemon

Only the daemon knows when a page is about to leave layer 1, so only it can
demote that page to layer 2 before dropping it. A Mooncake master evicts
silently and tells no client. Placing the client in the daemon also means:

- **One Mooncake client per node, not one per rank.** The whole arena is
  registered once with the Mooncake transfer engine. `batch_put_from` and
  `batch_get_into` then move pages between the arena and Mooncake zero-copy.
- **SGLang ranks never talk to Mooncake.** They only memcpy against the
  arena. A layer-2 hit is first promoted into the arena, then copied to the
  rank's L2.
- **Layer 2 is never called with the store lock held.** Every layer-2 call is
  batched, so one prefetch or backup batch costs one round trip per call.

## Behavior

| Event | What happens |
|---|---|
| backup | `reserve` -> rank copies its host page into the slot -> `commit`. A key already in layer 1 or layer 2 returns `PRESENT` and is not copied. One `batch_is_exist` covers the batch. |
| prefetch | `acquire` pins the leading run of keys that are in layer 1. For the rest, one `batch_is_exist` finds the run that layer 2 has, then one `batch_get_into` promotes those pages into new slots. |
| demotion | A cleaner thread keeps the oldest `--clean-ratio` of the arena already written to layer 2 (`batch_put_from`, up to 64 pages per call), so eviction never waits. |
| eviction | LRU among clean pages only. A page layer 2 refused (`demote_failures`) is still marked clean, so a full or failing layer 2 cannot pin the arena. |
| sharing | Rank-replicated pages (MLA KV, DSA indexer, MLA draft) have rank-free keys. Every rank, DP group and instance of the model on the node shares one copy, and with Mooncake every node sharing the master shares it too. |

## Layer 2 on Mooncake

There are two deployments.

**External Mooncake cluster.** Storage nodes hold the pages in their own DRAM
and SSD offload. The daemon is a pure client:

```bash
python -m sglang_2layer_cache.daemon --socket @sglang_2layer_cache --shm-size-gb 768 \
    --layer2 mooncake --mooncake-master 10.0.0.5:50051 \
    --mooncake-protocol rdma --mooncake-device mlx5_0,mlx5_1 \
    --mooncake-local-hostname 10.0.0.21
```

**Self-hosted on the node.** A local `mooncake_master` runs on the node. The
daemon contributes a DRAM segment, which Mooncake spills to a local SSD:

```bash
mooncake_master --port=50051 --eviction_high_watermark_ratio=0.95 &
python -m sglang_2layer_cache.daemon --socket @sglang_2layer_cache --shm-size-gb 768 \
    --layer2 mooncake --mooncake-master 127.0.0.1:50051 --mooncake-protocol tcp \
    --mooncake-segment-gb 64 --mooncake-ssd-offload-dir /mnt/nvme/mooncake
```

| Flag | Default | Meaning |
|---|---|---|
| `--mooncake-master` | required | `host:port` of `mooncake_master` |
| `--mooncake-protocol` | `tcp` | `tcp` or `rdma` |
| `--mooncake-device` | `""` | RDMA devices, e.g. `mlx5_0,mlx5_1` |
| `--mooncake-local-hostname` | `127.0.0.1` | address other Mooncake clients use to reach this daemon. Use the node IP whenever the daemon holds a segment that remote clients read. |
| `--mooncake-metadata-server` | `P2PHANDSHAKE` | transfer-engine metadata service |
| `--mooncake-segment-gb` | `0` | DRAM this daemon adds to the Mooncake pool |
| `--mooncake-ssd-offload-dir` | off | Mooncake SSD offload of this daemon's segment |

With `rdma`, registering a 768 GiB arena pins all of it. The daemon then needs
an unlimited `memlock` limit, and registration faults in the whole arena at
start-up.

## Layer 2 on local SSD files

```bash
python -m sglang_2layer_cache.daemon --shm-size-gb 768 \
    --layer2 ssd --ssd-dir /mnt/nvme/sglang_2layer_cache --ssd-size-gb 3000
```

Files are written to a temporary name, then renamed. The SSD index is rebuilt
from the directory on restart.

## Common daemon flags

| Flag | Meaning |
|---|---|
| `--socket` | A Unix socket path (shorter than 108 bytes), or `@name` for an abstract socket that containers sharing the host network reach without a shared filesystem. The daemon rejects clients run by another uid. |
| `--shm-size-gb` | Arena size. Start-up fails unless this plus 16 GiB fits in `MemAvailable`. |
| `--prefault` | Allocate the whole arena before serving. Off by default: the kernel zeroes it single-threaded. |
| `--health-port` | A TCP port opened once the daemon is ready. |
| `--clean-ratio` | Fraction of the arena, oldest first, kept demoted to layer 2 (default 0.1). |

## SGLang

```bash
TWO_LAYER_CACHE_SOCKET=@sglang_2layer_cache \
PYTHONPATH=/path/to/sglang-2layer-cache python -m sglang.launch_server ... \
  --enable-hierarchical-cache --hicache-size 48 \
  --hicache-io-backend direct --hicache-write-policy write_through \
  --hicache-mem-layout page_first_direct --hicache-host-memory-mode buffer_only \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{
      "backend_name": "two_layer_cache",
      "module_path": "sglang_2layer_cache.backend",
      "class_name": "TwoLayerCacheStorage",
      "interface_v1": 1}'
```

| Extra-config key | Default | Meaning |
|---|---|---|
| `two_layer_socket` | `$TWO_LAYER_CACHE_SOCKET`, else `/dev/shm/sglang_2layer_cache/default.sock` | daemon socket, path or `@name` |
| `two_layer_namespace` | `""` | extra key namespace, on top of the model name |
| `two_layer_rpc_timeout_s` | `30` | socket timeout |
| `two_layer_clear_on_flush` | `false` | let `/flush_cache` wipe both layers, including a shared Mooncake |

SGLang only switches built-in backends to a page-first layout by itself, so
pass `--hicache-mem-layout page_first_direct`. Without it, every page copy is
a gather across layers instead of one memcpy. If the daemon is unreachable,
every lookup is a miss and every backup fails; the server keeps serving.

## Tests

```bash
PYTHONPATH=.:<sglang>/python python -m pytest tests
```

`tests/test_mooncake_tier.py` replaces `mooncake.store` with an in-memory store
that copies through the raw addresses the tier passes. It checks setup
arguments, demotion, promotion and failure handling without a Mooncake
install. A real Mooncake run still has to be checked on the cluster.

## Not verified yet

- Whether `MooncakeDistributedStore.setup` accepts a 0-byte segment for a pure
  client. If it does not, give `--mooncake-segment-gb` a small value.
- The exact return code Mooncake gives for a put of a key that already exists.
  The daemon counts it as a demote failure, which is harmless.

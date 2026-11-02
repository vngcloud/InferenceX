# sglang-node-cache

A node-shared HiCache L3 for SGLang. SGLang sees one storage backend. Behind
it, a per-node daemon keeps two tiers.

```
GPU (L1)  <-  L2, private to each rank (--hicache-host-memory-mode buffer_only)
                 ^ memcpy (the backend, inside each rank)
     +---------- node cache daemon (one per node) ----------+
     |  hot : memfd arena, mapped by every rank              |
     |        v demote before eviction    ^ promote on hit   |
     |  cold: SSD/NVMe, one <key>.bin file per page          |
     +-------------------------------------------------------+
```

- **No SGLang fork.** The backend loads with `--hicache-storage-backend dynamic`
  and `interface_v1: 1`, so SGLang uses its zero-copy `batch_get_v1/batch_set_v1`
  path. Hybrid side pools (DSA indexer, draft, SWA) go through `batch_*_v2`.
- **Page bytes never cross the socket.** The arena is an anonymous memfd. The
  daemon passes its descriptor to each rank on connect (SCM_RIGHTS), and the
  rank maps it. After that the socket only carries keys and offsets. Nothing is
  created under `/dev/shm`, so the containers do not need a shared `/dev/shm`.
  The memory is freed when the daemon and its last client exit.
- **One copy per node.** Pages that are identical on every rank use keys with
  no rank tag: MLA KV, DSA indexer, MLA draft and MLA SWA. One copy then serves
  every TP rank, every DP group, and every SGLang instance of the same model on
  the node. With MLA, SGLang already writes only from attention-TP rank 0.
  Rank-sharded pools (MHA KV, Mamba) carry `.tp<rank>of<size>`. PP and CP
  ranks are tagged as well.

## Daemon

| Behavior | Detail |
|---|---|
| backup | `reserve` -> rank memcpys host page into the slot -> `commit`. A key already in the arena or on SSD returns `PRESENT` and nothing is copied. |
| prefetch | `acquire` pins the leading run of present pages, promoting SSD hits into the arena. The rank memcpys them into its L2, then `release`s them. |
| eviction | LRU, and only pages that are *clean*. A page is clean once it is on SSD, or always when no SSD tier is configured. |
| demotion | A cleaner thread keeps the oldest `--clean-ratio` of the arena persisted on SSD, so eviction never waits for a write. |
| concurrency | A reader waits up to `--fill-wait-s` for a page another rank is writing. When a client disconnects, the daemon drops its pins and unfinished reservations. |
| restart | The SSD index is rebuilt from the directory, so the SSD tier survives a daemon restart. The arena does not. Clients remap after reconnecting. |
| access | The daemon rejects a client whose uid differs from its own (`SO_PEERCRED`). |

The arena is carved into fixed-size slots, one size class per page size (KV
page, indexer page, ...). Each class grows by `--chunk-mb` until the arena is
full. A class never borrows slots from another class.

```bash
python -m sglang_node_cache.daemon \
    --socket @sglang_node_cache --shm-size-gb 768 \
    --ssd-dir /mnt/nvme/sglang_node_cache --ssd-size-gb 3000
```

| Flag | Meaning |
|---|---|
| `--socket` | A Unix socket path (must be shorter than 108 bytes), or `@name` for an abstract socket. An abstract socket lives in the network namespace, so containers that share the host network reach it without a shared filesystem. |
| `--shm-size-gb` | Arena size. Start-up fails unless this plus 16 GiB fits in `MemAvailable`. |
| `--prefault` | Allocate the whole arena before serving. Off by default: the kernel zeroes it single-threaded, which takes minutes for hundreds of GiB. When off, a node that runs out of memory later kills the writing engine with SIGBUS. |
| `--health-port` | A TCP port opened once the daemon is ready, for launchers that wait on a port. |

## SGLang

```bash
PYTHONPATH=/path/to/sglang-node-cache python -m sglang.launch_server ... \
  --enable-hierarchical-cache --hicache-size 32 \
  --hicache-io-backend direct --hicache-write-policy write_through \
  --hicache-mem-layout page_first_direct \
  --hicache-host-memory-mode buffer_only \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{
      "backend_name": "node_cache",
      "module_path": "sglang_node_cache.backend",
      "class_name": "NodeCacheStorage",
      "interface_v1": 1,
      "node_cache_socket": "/dev/shm/sglang_node_cache/default.sock"}'
```

| Extra-config key | Default | Meaning |
|---|---|---|
| `node_cache_socket` | `$NODE_CACHE_SOCKET`, else `/dev/shm/sglang_node_cache/default.sock` | daemon socket, path or `@name` |
| `node_cache_namespace` | `""` | extra key namespace, on top of the model name |
| `node_cache_rpc_timeout_s` | `30` | socket timeout |
| `node_cache_clear_on_flush` | `false` | let `/flush_cache` wipe the shared cache |

If the daemon is unreachable, every lookup is a miss and every backup fails.
The server keeps serving.

## Tests

```bash
PYTHONPATH=.:<sglang>/python python -m pytest tests
```

## Known limits

- DeepSeek V4 logical KV anchors (a host pool without `kv_buffer`) are not
  supported.
- Page copies are per page, in Python, through `get_data_page` /
  `set_from_flat_data_page`. SGLang only switches the layout to `page_first*`
  on its own for built-in backends, so pass `--hicache-mem-layout page_first_direct`.
  That keeps every copy one contiguous memcpy instead of a gather across layers.

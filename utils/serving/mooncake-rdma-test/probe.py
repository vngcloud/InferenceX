"""Mooncake store client probe: zero-copy batch put/get through one transfer protocol, timed and verified.

Run inside the sglang image (it ships the mooncake wheel) after a master and a store service are up:
    python3 probe.py --proto rdma --device rocep41s0f0 --host 10.200.3.1 \
        --master 10.200.3.1:50051 --metadata http://10.200.3.1:8080/metadata

Like a HiCache engine, the client contributes no segment of its own (global_segment_size 0): every object lands
in the store service's segment, and the client only registers its source and destination buffers with the NIC.
Prints one "RESULT ..." line per round and exits non-zero on any failed put, short get or data mismatch.
"""

import argparse
import sys
import time

import torch
from mooncake.store import MooncakeDistributedStore

LOCAL_BUFFER = 16 * 1024 * 1024  # what sglang's MooncakeStore passes; zero-copy transfers do not use it


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--proto", default="rdma", choices=["rdma", "tcp"])
    p.add_argument("--device", default="rocep41s0f0")
    p.add_argument("--host", required=True, help="this client's address (the NIC's IPv4)")
    p.add_argument("--master", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--objects", type=int, default=64)
    p.add_argument("--object-mb", type=int, default=64, help="object size; 64 MB ~ 1.2k tokens of GLM-5.3 KV")
    p.add_argument("--rounds", type=int, default=3)
    a = p.parse_args()

    store = MooncakeDistributedStore()
    ret = store.setup(a.host, a.metadata, 0, LOCAL_BUFFER, a.proto, a.device, a.master)
    if ret != 0:
        print(f"FAIL setup returned {ret}", flush=True)
        return 1

    # small copy through the local buffer first (what sglang's warmup does)
    if store.put("probe-warmup", b"x" * 4096) != 0 or store.get("probe-warmup") != b"x" * 4096:
        print("FAIL warmup put/get", flush=True)
        return 1

    size = a.object_mb * 1024 * 1024
    total = size * a.objects
    src = torch.randint(0, 256, (total,), dtype=torch.uint8)
    dst = torch.zeros(total, dtype=torch.uint8)
    for name, t in (("src", src), ("dst", dst)):
        r = store.register_buffer(t.data_ptr(), total)
        if r != 0:
            print(f"FAIL register_buffer({name}, {total / 2**30:.1f} GiB) returned {r}", flush=True)
            return 1
    print(f"registered 2 x {total / 2**30:.1f} GiB, {a.objects} objects of {a.object_mb} MB, proto={a.proto}", flush=True)

    ok = True
    for rnd in range(a.rounds):
        keys = [f"probe-{rnd}-{i}" for i in range(a.objects)]
        sptrs = [src.data_ptr() + i * size for i in range(a.objects)]
        dptrs = [dst.data_ptr() + i * size for i in range(a.objects)]
        sizes = [size] * a.objects
        dst.zero_()
        t0 = time.perf_counter()
        put = store.batch_put_from(keys, sptrs, sizes)
        t1 = time.perf_counter()
        got = store.batch_get_into(keys, dptrs, sizes)
        t2 = time.perf_counter()
        put_fail = [r for r in put if r != 0]
        get_short = [r for r in got if r != size]
        same = torch.equal(src, dst)
        ok &= not put_fail and not get_short and same
        print(f"RESULT round={rnd} proto={a.proto} device={a.device} bytes={total} "
              f"put_GBps={total / (t1 - t0) / 1e9:.2f} get_GBps={total / (t2 - t1) / 1e9:.2f} "
              f"put_fail={len(put_fail)} get_short={len(get_short)} verify={'OK' if same else 'MISMATCH'}",
              flush=True)
    store.remove_all()
    print("PROBE " + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

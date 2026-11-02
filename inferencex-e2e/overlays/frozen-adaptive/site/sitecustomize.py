"""Swap two sglang modules for the patched copies next to this file (Frozen-KV MTP adaptive steps).

The ai1rc image ships sglang v0.5.22 without adaptive support for FROZEN_KV_MTP. The finder is
inherited by spawned scheduler processes through PYTHONPATH; drop it once the patch is upstream.
"""
import importlib.abc
import importlib.util
import pathlib
import sys

_OVERLAY = pathlib.Path(__file__).resolve().parent.parent
_MODULES = {
    "sglang.srt.speculative.frozen_kv_mtp_worker_v2": "frozen_kv_mtp_worker_v2.py",
    "sglang.srt.speculative.adaptive_spec_params": "adaptive_spec_params.py",
}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        name = _MODULES.get(fullname)
        if name is None:
            return None
        return importlib.util.spec_from_file_location(fullname, _OVERLAY / name)


sys.meta_path.insert(0, _Finder())

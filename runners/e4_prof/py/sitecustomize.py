"""e4 profiling hooks, first on PYTHONPATH (set by runners/e4_prof/wrap.sh).

Python imports only the first sitecustomize on the path, so this one runs the mentor's
runners/sglang_gemma4_bcg/sitecustomize.py itself; without E4_TORCH_TRIGGER nothing else changes.

E4_TORCH_TRIGGER=<path>: every scheduler rank polls <path> twice a second from its event loop and,
once the file exists, starts its torch profiler for E4_TORCH_STEPS forward steps through the same
ProfileReq handler that /start_profile reaches. The Rust frontend has no /start_profile, so this is
the one trigger both frontends share; the scheduler code under it is identical in both arms.
"""
import functools
import importlib.abc
import importlib.util
import os
import runpy
import sys
import time

runpy.run_path(os.path.join(os.path.dirname(__file__), "..", "..", "sglang_gemma4_bcg", "sitecustomize.py"))

TARGET = "sglang.srt.managers.scheduler"
POLL_S = 0.5


def _patch(mod):
    from sglang.srt.managers.io_struct import ProfileReq

    trigger = os.environ["E4_TORCH_TRIGGER"]
    req = dict(
        output_dir=os.environ["E4_TORCH_DIR"],
        num_steps=int(os.environ["E4_TORCH_STEPS"]),
        activities=["CPU", "GPU"],
        profile_id=os.environ["E4_TORCH_ID"],
    )
    orig = mod.Scheduler.process_input_requests
    state = {"next_poll": 0.0, "fired": False}

    @functools.wraps(orig)
    def process_input_requests(self, recv_reqs):
        if not state["fired"] and time.monotonic() >= state["next_poll"]:
            state["next_poll"] = time.monotonic() + POLL_S
            if os.path.exists(trigger):
                state["fired"] = True
                # A failed start must not take the serving run down with it.
                try:
                    out = self.profiler_manager._profile(ProfileReq(**req))
                except Exception as e:
                    out = f"FAILED {type(e).__name__}: {e}"
                print(f"[e4prof] torch profiler start pid={os.getpid()} wall={time.time():.6f} -> {out}",
                      file=sys.stderr, flush=True)
        return orig(self, recv_reqs)

    mod.Scheduler.process_input_requests = process_input_requests
    print(f"[e4prof] patched {TARGET}.Scheduler.process_input_requests (trigger {trigger})",
          file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if os.environ.get("E4_TORCH_TRIGGER"):
    sys.meta_path.insert(0, _Finder())

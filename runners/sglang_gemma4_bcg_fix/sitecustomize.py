"""Make Gemma-4 follow the prefill CUDA graph runner's contract (supersedes gemma4_bcg_patch).

BCG / Full prefill graphs capture only layer_model.forward, called with input_ids AND the static
input_embeds slot; on replay they refill the static slots (input_ids, positions, input_embeds) and
ignore every other argument the eager wrapper passes. So the captured body must be a function of
those slots. Gemma-4 breaks that twice:

(1) Gemma4TextModel.forward requires exactly one of input_ids / input_embeds (capture raises), and
    PLE variants (hidden_size_per_layer_input > 0, e.g. E2B/E4B) get per_layer_inputs from the
    multimodal wrapper as a kwarg, which replay drops. Fix: given both, keep the embeds and compute
    per_layer_inputs inside the body from input_ids (multimodal placeholder ids -> pad, as the
    wrapper does), so the graph reads them from the refilled input_ids slot.
(2) Gemma4ForConditionalGeneration.forward does `positions += 1` in place. Under BCG the eager tail
    gets the static positions slot, so the graph sees the shift; under Full it gets the raw batch,
    so the graph reads unshifted positions (prefill KV off by one from decode). Fix: hand the Full
    tail a view of the static positions slot.

Loaded via PYTHONPATH (inherited by the scheduler subprocesses); patches each module on import.
"""
import functools
import importlib.abc
import importlib.util
import sys


def _log(msg):
    print(f"[gemma4_bcg_fix] {msg}", file=sys.stderr, flush=True)


def _patch_causal(mod):
    import torch

    cls = mod.Gemma4TextModel
    orig = cls.forward

    @functools.wraps(orig)
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, per_layer_inputs=None,
                pp_proxy_tensors=None, **kwargs):
        if input_ids is not None and input_embeds is not None and self.pp_group.is_first_rank:
            if per_layer_inputs is None and self.hidden_size_per_layer_input:
                ple_ids = input_ids
                for tid in getattr(self, "_bcg_fix_mm_token_ids", ()):
                    ple_ids = torch.where(ple_ids == tid, self._bcg_fix_pad_id, ple_ids)
                per_layer_inputs = self.get_per_layer_inputs(ple_ids)
            input_ids = None
        return orig(self, input_ids, positions, forward_batch, input_embeds, per_layer_inputs,
                    pp_proxy_tensors, **kwargs)

    cls.forward = forward
    _log("patched Gemma4TextModel.forward")


def _patch_mm(mod):
    cls = mod.Gemma4ForConditionalGeneration
    orig = cls.__init__

    @functools.wraps(orig)
    def __init__(self, config, *args, **kwargs):
        orig(self, config, *args, **kwargs)
        lm = self.language_model
        lm._bcg_fix_mm_token_ids = tuple(
            t for t in (getattr(config, f"{m}_token_id", None) for m in ("image", "video", "audio"))
            if t is not None
        )
        lm._bcg_fix_pad_id = config.text_config.pad_token_id

    cls.__init__ = __init__
    _log("patched Gemma4ForConditionalGeneration.__init__")


def _patch_runner(mod):
    cls = mod.PrefillCudaGraphRunner
    orig = cls._execute_body_capture

    @functools.wraps(orig)
    def _execute_body_capture(self, forward_batch, static_forward_batch, static_num_tokens,
                              raw_num_tokens, shape_key, **kwargs):
        if not self._is_full_backend:
            return orig(self, forward_batch, static_forward_batch, static_num_tokens,
                        raw_num_tokens, shape_key, **kwargs)
        raw_positions = forward_batch.positions
        forward_batch.positions = static_forward_batch.positions[:raw_num_tokens]
        try:
            return orig(self, forward_batch, static_forward_batch, static_num_tokens,
                        raw_num_tokens, shape_key, **kwargs)
        finally:
            forward_batch.positions = raw_positions

    cls._execute_body_capture = _execute_body_capture
    _log("patched PrefillCudaGraphRunner._execute_body_capture")


TARGETS = {
    "sglang.srt.models.gemma4_causal": _patch_causal,
    "sglang.srt.models.gemma4_mm": _patch_mm,
    "sglang.srt.model_executor.runner.prefill_cuda_graph_runner": _patch_runner,
}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        patch = TARGETS.get(name)
        if patch is None:
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
            patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


sys.meta_path.insert(0, _Finder())

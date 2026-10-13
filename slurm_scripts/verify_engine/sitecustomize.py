"""Scratch instrumentation for the Gemma-4 DFlash engine checks (never edits the engine venvs).

Put this directory on PYTHONPATH: Python imports it in every process, including the engines'
spawned workers. A daemon thread waits for the draft-model module to be imported, then wraps
the one call that produces the draft's block embedding.

Env:
  DFV_LOG    jsonl file; first calls log the per-token L2 norm of the draft's block embedding
  DFV_SCALE  vLLM: multiply that embedding by this factor (test C, default 1)
  DFV_SGLANG_NOISE_SCALE  sglang: define Gemma-4 get_dflash_noise_embedding_scale() returning it
"""
import functools
import json
import os
import sys
import threading
import time

LOG = os.environ.get("DFV_LOG")
SCALE = float(os.environ.get("DFV_SCALE", "1"))
SG_SCALE = os.environ.get("DFV_SGLANG_NOISE_SCALE")
MAX_LOGS = 6
_n = {"calls": 0, "logged": 0}


def _log(tag, embeds, ids=None):
    import torch

    if torch.cuda.is_current_stream_capturing() or _n["logged"] >= MAX_LOGS:
        return
    if embeds.dim() != 2 or embeds.shape[0] % 16 != 0 or embeds.shape[0] == 0:
        return
    norms = embeds.detach().float().norm(dim=-1)
    if norms.max().item() == 0.0:  # dummy warmup buffers
        return
    rec = {
        "tag": tag,
        "pid": os.getpid(),
        "shape": list(embeds.shape),
        "norm_first_block": [round(x, 3) for x in norms[:16].tolist()],
        "norm_mean": round(norms.mean().item(), 3),
    }
    if ids is not None:
        rec["ids_first_block"] = ids.reshape(-1)[:16].tolist()
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    _n["logged"] += 1


def _patch_vllm(mod):
    # the proposer feeds input_ids to the inner model, which embeds them itself (job 1203's x73
    # run patched the outer ForCausalLM method and never reached the draft)
    cls = mod.DFlashQwen3Model
    orig = cls.embed_input_ids

    def wrapped(self, input_ids):
        out = orig(self, input_ids)
        if SCALE != 1.0:
            out = out * SCALE
        if LOG:
            try:
                _log("vllm.embed_input_ids", out, input_ids)
            except Exception as e:  # never break serving
                print(f"[dfv] log failed: {e}", file=sys.stderr)
        return out

    cls.embed_input_ids = wrapped
    print(f"[dfv] patched vLLM {cls.__name__}.embed_input_ids scale={SCALE}", file=sys.stderr)


def _patch_sglang_draft(mod):
    cls = mod.DFlashDraftModel
    orig = cls.forward

    @functools.wraps(orig)  # sglang inspects forward's signature for input_embeds
    def wrapped(self, *args, **kwargs):
        emb = kwargs.get("input_embeds")
        if LOG and emb is not None:
            try:
                _log("sglang.DFlashDraftModel.forward", emb)
            except Exception as e:
                print(f"[dfv] log failed: {e}", file=sys.stderr)
        return orig(self, *args, **kwargs)

    cls.forward = wrapped
    print("[dfv] patched sglang DFlashDraftModel.forward (logging)", file=sys.stderr)


def _patch_sglang_target(mod):
    n = 0
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and name.startswith("Gemma4For"):
            obj.get_dflash_noise_embedding_scale = lambda self, _s=float(SG_SCALE): _s
            n += 1
    print(f"[dfv] sglang noise embedding scale={SG_SCALE} set on {n} Gemma4 classes", file=sys.stderr)


AUDIT = os.environ.get("DFV_AUDIT")  # jsonl: vLLM draft weight-load audit + positions seen by the draft
DRAFT_DIR = os.environ.get("DFV_DRAFT_DIR")
_pos_logged = {"fwd": 0, "ctx": 0}


def _audit_write(rec):
    with open(AUDIT, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _audit_weights(self):
    """Compare every loaded draft parameter with the safetensors (undoing qkv / gate_up fusing)."""
    import torch
    from safetensors import safe_open

    params = dict(self.model.named_parameters())
    cfg = self.model.config
    hd = cfg.head_dim
    q_sz, kv_sz = cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd
    inter = cfg.intermediate_size
    seen, bad = set(), []
    n = 0
    with safe_open(os.path.join(DRAFT_DIR, "model.safetensors"), "pt", device="cpu") as sf:
        for k in sf.keys():
            if "t2d" in k or "d2t" in k or "lm_head" in k:
                continue
            name = k.replace("midlayer.", "layers.0.")
            src = sf.get_tensor(k)
            sl = None
            for tag, new, lo, hi in (
                (".q_proj", ".qkv_proj", 0, q_sz),
                (".k_proj", ".qkv_proj", q_sz, q_sz + kv_sz),
                (".v_proj", ".qkv_proj", q_sz + kv_sz, q_sz + 2 * kv_sz),
                (".gate_proj", ".gate_up_proj", 0, inter),
                (".up_proj", ".gate_up_proj", inter, 2 * inter),
            ):
                if tag in name:
                    name, sl = name.replace(tag, new), (lo, hi)
                    break
            p = params.get(name)
            if p is None:
                bad.append({"key": k, "problem": "no vLLM param"})
                continue
            seen.add(name)
            dst = p.data if sl is None else p.data[sl[0] : sl[1]]
            if dst.shape != src.shape:
                bad.append({"key": k, "problem": f"shape {tuple(dst.shape)} vs {tuple(src.shape)}"})
                continue
            d = (dst.float().cpu() - src.float()).abs().max().item()
            n += 1
            if d != 0.0:
                bad.append({"key": k, "problem": "values differ", "max_abs_diff": d})
    unloaded = sorted(set(params) - seen)
    rec = {"audit": "weights", "compared": n, "mismatches": bad, "vllm_params_not_in_safetensors": unloaded}
    # frozen-target-head / embedding facts
    rec["has_lm_head_param"] = any("lm_head" in x for x in params)
    rec["dtype"] = str(next(iter(params.values())).dtype)
    # rope: compare layer-0 cache with the textbook default rope (theta 1e6)
    attn0 = self.model.layers[0].self_attn
    rope = attn0.rotary_emb
    cache = rope.cos_sin_cache
    theta = float(cfg.rope_parameters["rope_theta"])
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    t = torch.arange(1024, dtype=torch.float32)
    fr = torch.outer(t, inv)
    exp = torch.cat([fr.cos(), fr.sin()], dim=-1)
    rec["rope"] = {
        "cache_shape": list(cache.shape),
        "cache_dtype": str(cache.dtype),
        "is_neox": bool(rope.is_neox_style),
        "rotary_dim": int(rope.rotary_dim),
        "max_abs_diff_vs_default_theta1e6": (cache[:1024].float().cpu() - exp).abs().max().item(),
    }
    # fused context-KV buffers vs the safetensors
    try:
        m = self.model
        rec["fused_kv_weight_shape"] = list(m._fused_kv_weight.shape)
        with safe_open(os.path.join(DRAFT_DIR, "model.safetensors"), "pt", device="cpu") as sf:
            ks = [sf.get_tensor(f"layers.{i}.self_attn.k_proj.weight") for i in range(cfg.num_hidden_layers)]
            vs = [sf.get_tensor(f"layers.{i}.self_attn.v_proj.weight") for i in range(cfg.num_hidden_layers)]
            hn = sf.get_tensor("hidden_norm.weight")
        fw = m._fused_kv_weight.float().cpu()
        want_a = torch.cat([torch.cat([k, v], 0) for k, v in zip(ks, vs)], 0).float()
        want_b = torch.cat(ks + vs, 0).float()
        rec["fused_kv_diff_layerwise_kv"] = (fw - want_a).abs().max().item() if fw.shape == want_a.shape else "shape"
        rec["fused_kv_diff_all_k_then_all_v"] = (fw - want_b).abs().max().item() if fw.shape == want_b.shape else "shape"
        rec["hidden_norm_alias_diff"] = (m._hidden_norm_weight.float().cpu() - hn.float()).abs().max().item()
    except Exception as e:
        rec["fused_kv_check_error"] = repr(e)
    _audit_write(rec)


def _patch_vllm_audit(mod):
    cls = mod.DFlashQwen3ForCausalLM
    orig_load = cls.load_weights

    def load_weights(self, weights):
        out = orig_load(self, weights)
        try:
            _audit_weights(self)
        except Exception as e:
            import traceback

            _audit_write({"audit": "weights", "error": traceback.format_exc()})
        return out

    cls.load_weights = load_weights

    inner = mod.DFlashQwen3Model
    orig_fwd = inner.forward

    def fwd(self, input_ids, positions, input_embeds=None):
        if _pos_logged["fwd"] < 4 and positions.numel() % 16 == 0 and positions.numel() > 0:
            pl = positions.reshape(-1).tolist()
            _audit_write({"audit": "draft_forward", "n": len(pl), "ids_first16": input_ids.reshape(-1)[:16].tolist(),
                          "positions_first20": pl[:20], "positions_last20": pl[-20:]})
            _pos_logged["fwd"] += 1
        return orig_fwd(self, input_ids, positions, input_embeds)

    inner.forward = fwd

    orig_ctx = inner.precompute_and_store_context_kv

    def ctx(self, context_states, context_positions, context_slot_mapping=None):
        if _pos_logged["ctx"] < 6 and context_slot_mapping is not None and context_positions.numel() > 0:
            pl = context_positions.reshape(-1).tolist()
            _audit_write({"audit": "context_kv", "n": len(pl), "positions_first8": pl[:8], "positions_last8": pl[-8:],
                          "state_norm_mean": round(context_states.float().norm(dim=-1).mean().item(), 3),
                          "state_dtype": str(context_states.dtype)})
            _pos_logged["ctx"] += 1
        return orig_ctx(self, context_states, context_positions, context_slot_mapping)

    inner.precompute_and_store_context_kv = ctx
    print("[dfv] vLLM draft audit hooks installed", file=sys.stderr)


DUMP = os.environ.get("DFV_DUMP")  # path prefix: save the first prefill's raw aux hidden states (pre-fc)
_dumped = {"done": False}


def _dump_aux(tag, hs):
    import torch

    if _dumped.get("n", 0) >= 8 or hs.dim() != 2 or hs.shape[0] < 32:
        return
    if hs.float().abs().max().item() == 0.0:
        return
    i = _dumped["n"] = _dumped.get("n", 0) + 1
    torch.save(hs.detach().cpu(), f"{DUMP}.{tag}.{i}.pt")
    with open(f"{DUMP}.{tag}.txt", "a") as f:
        f.write(f"{tag} #{i} shape={list(hs.shape)} dtype={hs.dtype}\n")


def _patch_vllm_dump(mod):
    cls = mod.DFlashQwen3ForCausalLM
    orig = cls.combine_hidden_states

    def combine(self, hidden_states):
        try:
            _dump_aux("vllm", hidden_states)
        except Exception as e:
            print(f"[dfv] dump failed: {e}", file=sys.stderr)
        return orig(self, hidden_states)

    cls.combine_hidden_states = combine
    print("[dfv] vLLM aux-hidden dump installed", file=sys.stderr)


def _patch_sglang_dump(mod):
    for name in dir(mod):
        cls = getattr(mod, name)
        if isinstance(cls, type) and "project_target_hidden" in cls.__dict__:
            orig = cls.project_target_hidden

            def proj(self, target_hidden, _orig=orig):
                try:
                    _dump_aux("sglang", target_hidden)
                except Exception as e:
                    print(f"[dfv] dump failed: {e}", file=sys.stderr)
                return _orig(self, target_hidden)

            cls.project_target_hidden = proj
            print(f"[dfv] sglang aux-hidden dump installed on {name}", file=sys.stderr)


LAYERS = os.environ.get("DFV_SG_LAYERS")  # path prefix: save every sglang Gemma4 decoder layer's input/output (first long prefill)
_lay = {"rows": None, "out": {}, "saved": False}


def _patch_sglang_layers(mod):
    import torch

    cls = mod.Gemma4DecoderLayer
    orig = cls.forward
    nl = int(os.environ.get("DFV_NLAYERS", "60"))
    clone = bool(os.environ.get("DFV_SG_CLONE"))  # test fix: layer works on a copy of its input, so the aux reference is never mutated
    keep = {2, 16, 30, 44, 58}  # layers whose input buffer is re-read after the layer ran (aliasing probe)

    def fwd(self, positions, hidden_states, per_layer_input, forward_batch, **kw):
        n = hidden_states.shape[0]
        rec = bool(LAYERS) and n >= 1000 and not _lay["saved"] and _lay["rows"] in (None, n)
        if rec and self.layer_id == 0:
            _lay["rows"], _lay["out"] = n, {"emb": hidden_states.detach().cpu()}
        inp = hidden_states
        if clone:
            hidden_states = hidden_states.clone()
        pre = inp.detach().cpu() if rec and self.layer_id in keep else None
        out = orig(self, positions, hidden_states, per_layer_input, forward_batch, **kw)
        if rec:
            _lay["out"][self.layer_id] = out[0].detach().cpu()
            if pre is not None:
                _lay["out"][f"in{self.layer_id}_before"] = pre
                _lay["out"][f"in{self.layer_id}_after"] = inp.detach().cpu()
            if self.layer_id == nl - 1:
                _lay["saved"] = True
                torch.save(_lay["out"], f"{LAYERS}.sglang.pt")
                print(f"[dfv] saved {len(_lay['out'])} sglang layer tensors ({n} rows)", file=sys.stderr)
        return out

    cls.forward = fwd
    print(f"[dfv] sglang per-layer hook installed (clone={clone})", file=sys.stderr)


_TARGETS = []
if os.environ.get("DFV_LOG") or SCALE != 1.0:
    _TARGETS.append(("vllm.model_executor.models.qwen3_dflash", _patch_vllm))
    _TARGETS.append(("sglang.srt.models.dflash", _patch_sglang_draft))
if DUMP:
    _TARGETS.append(("vllm.model_executor.models.qwen3_dflash", _patch_vllm_dump))
    _TARGETS.append(("sglang.srt.models.dflash", _patch_sglang_dump))
if LAYERS or os.environ.get("DFV_SG_CLONE"):
    _TARGETS.append(("sglang.srt.models.gemma4_causal", _patch_sglang_layers))
if AUDIT:
    _TARGETS.append(("vllm.model_executor.models.qwen3_dflash", _patch_vllm_audit))
if SG_SCALE:
    _TARGETS.append(("sglang.srt.models.gemma4_causal", _patch_sglang_target))
    _TARGETS.append(("sglang.srt.models.gemma4_mm", _patch_sglang_target))


def _watch():
    pending = list(_TARGETS)
    deadline = time.time() + 1800
    while pending and time.time() < deadline:
        for item in list(pending):
            name, fn = item
            mod = sys.modules.get(name)
            # the module is fully imported once its last class exists
            if mod is not None and getattr(mod, "__spec__", None) and not getattr(
                mod.__spec__, "_initializing", False
            ):
                try:
                    fn(mod)
                except Exception as e:
                    print(f"[dfv] patch {name} failed: {e!r}", file=sys.stderr)
                pending.remove(item)
        time.sleep(0.02)


if _TARGETS:
    threading.Thread(target=_watch, daemon=True).start()

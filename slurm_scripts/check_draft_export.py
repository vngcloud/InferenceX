"""Check that an exported draft (hf-draft/model.safetensors) matches the model SpecForge builds.

Reads only the safetensors header (no weights), builds the draft on the meta device from the run
config, and reports missing keys, unexpected keys, shape mismatches and dtypes.

    python scripts/check_draft_export.py configs/specforge/gemma4-dflash-smoke-overfit.yaml \
        /shared/phucnlt2/checkpoints/gemma4-dflash-smoke-overfit/hf-draft/model.safetensors
"""
import json
import struct
import sys


def header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return {k: (tuple(v["shape"]), v["dtype"]) for k, v in h.items()}


def main(cfg_path, st_path):
    import torch
    from specforge.application.composition import resolve_run
    from specforge.config.schema import load_config
    from specforge.modeling.auto import AutoDraftModel
    from specforge.training.model_loading import resolve_draft_config

    cfg = load_config(cfg_path)
    provider = resolve_run(cfg).algorithm.providers.model.draft_config
    with torch.device("meta"):
        draft = AutoDraftModel.from_config(resolve_draft_config(cfg, provider=provider), torch_dtype=torch.bfloat16)
    want = {k: tuple(v.shape) for k, v in draft.state_dict().items()}
    got = header(st_path)
    missing = sorted(set(want) - set(got))
    unexpected = sorted(set(got) - set(want))
    shape_bad = sorted(k for k in set(want) & set(got) if want[k] != got[k][0])
    dtypes = sorted({d for _, d in got.values()})
    print(f"{type(draft).__name__}: model keys {len(want)}, file keys {len(got)}, dtypes {dtypes}")
    print(f"missing {len(missing)}  unexpected {len(unexpected)}  shape mismatches {len(shape_bad)}")
    for name, items in (("missing", missing), ("unexpected", unexpected), ("shape", shape_bad)):
        for k in items[:10]:
            print(f"  {name}: {k}", want.get(k), got.get(k))
    extra = [k for k in got if "selector" in k or "pair" in k or "unary" in k]
    print(f"DFlash2 selector-like keys in file: {len(extra)} {extra[:6]}")
    sys.exit(1 if (missing or unexpected or shape_bad) else 0)


if __name__ == "__main__":
    main(*sys.argv[1:3])

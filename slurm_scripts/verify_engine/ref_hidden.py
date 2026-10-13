"""Reference target hidden states for the first forced-prefix request (HF transformers, no engine).

Rebuilds the exact token ids that forced_client.py sends first, runs the FP8-block target with
HF, saves the 5 aux layers (output of layer id == hidden_states[id + 1]) as [T, 5*5376] bf16 in the
same layout as the engine dumps, then compares it to the sglang / vLLM dumps.
"""
import argparse
import json
import sys

import torch

sys.path.insert(0, "scripts/verify_engine")
from accept_client import MARK  # noqa: E402

LAYERS = [1, 15, 29, 43, 57]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmp", nargs="*", default=[], help="name=path.pt dumps to compare against")
    ap.add_argument("--layers-cmp", default=None, help="sglang per-layer dump (layers.sglang.pt)")
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--cuts", default="0.1,0.3,0.5,0.7")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    r = json.loads(open(a.data).readline())
    i = r["text"].rfind(MARK)
    p = tok(r["text"][: i + len(MARK)], add_special_tokens=False)["input_ids"]
    rec = tok(r["text"][i + len(MARK):], add_special_tokens=False)["input_ids"]
    ids = None
    for c in [float(x) for x in a.cuts.split(",")]:
        k = int(len(rec) * c)
        if k + a.gen <= len(rec):
            ids = p + rec[:k]
            break
    print(f"[ref] token count {len(ids)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0")
    print(f"[ref] loaded {type(model).__name__}", flush=True)
    with torch.no_grad():
        out = model(torch.tensor([ids], device="cuda:0"), output_hidden_states=True)
    hs = out.hidden_states
    print(f"[ref] n_hidden_states={len(hs)}", flush=True)
    ref = torch.cat([hs[l + 1][0] for l in LAYERS], dim=-1).cpu()
    torch.save(ref, a.out)

    F = torch.nn.functional
    if a.layers_cmp:
        sg = torch.load(a.layers_cmp)
        print("[layers] sglang vs HF per layer (cos / rel / norm sg / norm ref)", flush=True)
        for k in ["emb"] + list(range(59)):
            u = sg[k].float().cuda()
            v = (hs[0 if k == "emb" else k + 1][0]).float()
            cs = F.cosine_similarity(u, v, dim=-1).mean().item()
            rel = ((u - v).norm(dim=-1) / v.norm(dim=-1)).mean().item()
            print(f"[layers] {k}: cos {cs:.5f} rel {rel:.4f} norm {u.norm(dim=-1).mean():.1f} / {v.norm(dim=-1).mean():.1f}", flush=True)
    for spec in a.cmp:
        name, path = spec.split("=", 1)
        x = torch.load(path).float()
        if x.shape != ref.shape:
            print(f"[cmp] {name}: shape {tuple(x.shape)} != ref {tuple(ref.shape)}", flush=True)
            continue
        for j, l in enumerate(LAYERS):
            s = slice(j * 5376, (j + 1) * 5376)
            u, v = x[:, s], ref[:, s].float()
            cs = F.cosine_similarity(u, v, dim=-1).mean().item()
            rel = ((u - v).norm(dim=-1) / v.norm(dim=-1)).mean().item()
            print(f"[cmp] {name} L{l}: cos {cs:.5f} rel {rel:.4f} norm {u.norm(dim=-1).mean():.1f} ref {v.norm(dim=-1).mean():.1f}", flush=True)


if __name__ == "__main__":
    main()

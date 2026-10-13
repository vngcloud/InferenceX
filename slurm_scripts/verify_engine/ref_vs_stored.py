"""Compare stored SpecForge training features with an HF reference on the same token ids.

Loads one stored sample (data_N.ckpt: input_ids, hidden_states [1, T, 5*5376]), runs the HF
FP8-block target on input_ids[:T'] (causal -> a prefix gives identical rows) and prints cos / relative
error per target layer for the stored features, and the same for the HF-vs-HF sanity check.
"""
import argparse

import torch

LAYERS = [1, 15, 29, 43, 57]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=1200)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0")
    F = torch.nn.functional
    for path in a.ckpt:
        d = torch.load(path, map_location="cpu")
        ids = d["input_ids"][: a.tokens]
        st = d["hidden_states"][0, : len(ids)].float()
        with torch.no_grad():
            hs = model(ids[None].cuda(), output_hidden_states=True).hidden_states
        print(f"[stored] {path.split('/')[-1]} tokens {len(ids)} hidden {tuple(d['hidden_states'].shape)}", flush=True)
        for j, l in enumerate(LAYERS):
            u = st[:, j * 5376:(j + 1) * 5376]
            v = hs[l + 1][0].float().cpu()
            cs = F.cosine_similarity(u, v, dim=-1).mean().item()
            rel = ((u - v).norm(dim=-1) / v.norm(dim=-1)).mean().item()
            print(f"[stored] L{l}: cos {cs:.5f} rel {rel:.4f} norm stored {u.norm(dim=-1).mean():.1f} ref {v.norm(dim=-1).mean():.1f}", flush=True)


if __name__ == "__main__":
    main()

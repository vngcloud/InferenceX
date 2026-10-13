"""Forced-prefix acceptance check: removes greedy drift from the engine measurement.

Each prompt is extended with the first k tokens of the RECORDED reply (the text the overfit draft
memorised), then the target continues greedily for --gen tokens. Acceptance is aggregated only over
continuations that still equal the recorded text (faithful), which is the engine-side analogue of
the training-side teacher-forced walk (~15.8). Cuts are at several fractions of the reply.
"""
import argparse
import json
import time

from accept_client import MARK, post, vllm_counters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="google/gemma-4-31B-it")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--cuts", default="0.1,0.3,0.5,0.7")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = [json.loads(l) for l in open(a.data)][: a.n]
    cuts = [float(c) for c in a.cuts.split(",")]
    reqs = []
    for r in rows:
        i = r["text"].rfind(MARK)
        p = tok(r["text"][: i + len(MARK)], add_special_tokens=False)["input_ids"]
        rec = tok(r["text"][i + len(MARK):], add_special_tokens=False)["input_ids"]
        for c in cuts:
            k = int(len(rec) * c)
            if k + a.gen > len(rec):
                continue
            reqs.append({"ids": p + rec[:k], "ref": tok.decode(rec[k:k + a.gen]), "cut": c})

    first = reqs[0]["ids"]
    for _ in range(2):  # warm up
        if a.engine == "vllm":
            post(a.base_url + "/v1/completions", {"model": a.model, "prompt": first, "max_tokens": 16, "temperature": 0})
        else:
            post(a.base_url + "/generate", {"input_ids": first, "sampling_params": {"max_new_tokens": 16, "temperature": 0}})

    t0 = time.time()
    for q in reqs:
        if a.engine == "vllm":
            c0 = vllm_counters(a.base_url)
            r = post(a.base_url + "/v1/completions",
                     {"model": a.model, "prompt": q["ids"], "max_tokens": a.gen, "temperature": 0})
            c1 = vllm_counters(a.base_url)
            q["tokens"] = r["usage"]["completion_tokens"]
            q["drafts"] = c1["drafts"] - c0["drafts"]
            q["accepted"] = c1["accepted"] - c0["accepted"]
            q["per_pos"] = [c1["per_pos"].get(i, 0.0) - c0["per_pos"].get(i, 0.0) for i in range(15)]
            q["text"] = r["choices"][0]["text"]
        else:
            r = post(a.base_url + "/generate",
                     {"input_ids": q["ids"], "sampling_params": {"max_new_tokens": a.gen, "temperature": 0}})
            q["tokens"] = r["meta_info"]["completion_tokens"]
            q["drafts"] = r["meta_info"].get("spec_verify_ct", 0)
            q["spec_meta"] = {k: v for k, v in r["meta_info"].items() if k.startswith("spec_")}
            q["text"] = r["text"]
        n = 0
        while n < min(len(q["text"]), len(q["ref"])) and q["text"][n] == q["ref"][n]:
            n += 1
        q["match_frac"] = n / max(1, len(q["ref"]))
        del q["ids"]

    def agg(sel):
        t = sum(q["tokens"] for q in sel)
        d = sum(q["drafts"] for q in sel)
        return {"n": len(sel), "tokens": t, "drafts": d, "mean_accept_len": round(t / d, 3) if d else None}

    # P(position i accepted | round) over faithful requests: vLLM per-pos counters; sglang histogram
    # (histogram[k] = rounds that accepted k draft tokens), so P(i) = sum_{k>i} hist[k] / rounds
    pos = [0.0] * 15
    rounds = 0.0
    for q in reqs:
        if q["match_frac"] < 0.95:
            continue
        if "per_pos" in q:
            pos = [x + y for x, y in zip(pos, q["per_pos"])]
            rounds += q["drafts"]
        elif isinstance(q.get("spec_meta", {}).get("spec_accept_histogram"), list):
            h = q["spec_meta"]["spec_accept_histogram"]
            for i in range(15):
                pos[i] += sum(h[i + 1:])
            rounds += sum(h)
    per_position = [round(x / rounds, 3) for x in pos] if rounds else None

    res = {"per_position_accept_rate": per_position, "engine": a.engine, "data": a.data, "gen": a.gen, "seconds": round(time.time() - t0, 1),
           "all": agg(reqs),
           "faithful_ge_0.95": agg([q for q in reqs if q["match_frac"] >= 0.95]),
           "faithful_ge_0.8": agg([q for q in reqs if q["match_frac"] >= 0.8]),
           "per_cut": {str(c): agg([q for q in reqs if q["cut"] == c and q["match_frac"] >= 0.95]) for c in cuts},
           "requests": reqs}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "requests"}, indent=1))


if __name__ == "__main__":
    main()

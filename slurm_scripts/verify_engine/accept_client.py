"""Acceptance-length client for the Gemma-4 DFlash engine checks (greedy, token-id prompts).

Rows are the smoke jsonl `text` fields cut at the generation prompt, so the target answers the
same prompts the draft was trained on (overfit) or never saw (heldout).
vLLM: acceptance from /metrics counters. sglang: per-request meta_info (tokens / verify steps).
"""
import argparse
import json
import re
import time
import urllib.request

MARK = "<|turn>model\n<|channel>thought\n<channel|>"


def post(url, body, timeout=900):
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def vllm_counters(base):
    txt = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    out = {"drafts": 0.0, "accepted": 0.0, "per_pos": {}}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"vllm:spec_decode_num_drafts_total\{.*\}\s+(\S+)", line)
        if m:
            out["drafts"] += float(m.group(1))
        m = re.match(r"vllm:spec_decode_num_accepted_tokens_total\{.*\}\s+(\S+)", line)
        if m:
            out["accepted"] += float(m.group(1))
        m = re.match(r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{.*position="(\d+)".*\}\s+(\S+)', line)
        if m:
            out["per_pos"][int(m.group(1))] = out["per_pos"].get(int(m.group(1)), 0.0) + float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", required=True, help="vLLM served model name")
    ap.add_argument("--tokenizer", default="google/gemma-4-31B-it")
    ap.add_argument("--data", required=True, help="smoke jsonl (overfit / heldout)")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = [json.loads(l) for l in open(a.data)][: a.n]
    prompts, recorded = [], []
    for r in rows:
        i = r["text"].rfind(MARK)
        assert i >= 0, "generation-prompt marker missing"
        recorded.append(r["text"][i + len(MARK):])
        prompts.append(tok(r["text"][: i + len(MARK)], add_special_tokens=False)["input_ids"])

    # warm up (compile / cuda graphs) so the counters below cover steady state only
    for p in prompts[:2]:
        if a.engine == "vllm":
            post(a.base_url + "/v1/completions", {"model": a.model, "prompt": p, "max_tokens": 32, "temperature": 0})
        else:
            post(a.base_url + "/generate", {"input_ids": p, "sampling_params": {"max_new_tokens": 32, "temperature": 0}})

    t0 = time.time()
    texts, toks, verify, per_req = [], 0, 0, []
    before = vllm_counters(a.base_url) if a.engine == "vllm" else None
    for p in prompts:
        c0 = vllm_counters(a.base_url) if a.engine == "vllm" else None
        if a.engine == "vllm":
            r = post(a.base_url + "/v1/completions",
                     {"model": a.model, "prompt": p, "max_tokens": a.max_tokens, "temperature": 0})
            texts.append(r["choices"][0]["text"])
            toks += r["usage"]["completion_tokens"]
            c1 = vllm_counters(a.base_url)
            per_req.append({"tokens": r["usage"]["completion_tokens"], "drafts": c1["drafts"] - c0["drafts"],
                            "accepted": c1["accepted"] - c0["accepted"]})
        else:
            r = post(a.base_url + "/generate",
                     {"input_ids": p, "sampling_params": {"max_new_tokens": a.max_tokens, "temperature": 0}})
            texts.append(r["text"])
            mi = r["meta_info"]
            toks += mi["completion_tokens"]
            verify += mi.get("spec_verify_ct", 0)
            per_req.append({"tokens": mi["completion_tokens"], "drafts": mi.get("spec_verify_ct", 0)})
    res = {"engine": a.engine, "data": a.data, "n": len(prompts), "completion_tokens": toks,
           "seconds": round(time.time() - t0, 1), "first_output": texts[0][:200]}
    if a.engine == "vllm":
        after = vllm_counters(a.base_url)
        d = after["drafts"] - before["drafts"]
        acc = after["accepted"] - before["accepted"]
        res.update(drafts=d, accepted=acc, mean_accept_len=round(1 + acc / d, 3) if d else None,
                   accepted_per_pos={k: round((after["per_pos"].get(k, 0) - before["per_pos"].get(k, 0)) / d, 3)
                                     for k in sorted(after["per_pos"])} if d else {})
    else:
        res.update(verify_steps=verify, mean_accept_len=round(toks / verify, 3) if verify else None)
    for q, t, rec in zip(per_req, texts, recorded):
        n = 0
        while n < min(len(t), len(rec)) and t[n] == rec[n]:
            n += 1
        q["match_chars"], q["out_chars"], q["rec_chars"] = n, len(t), len(rec)
    res["per_request"] = per_req
    res["texts"] = texts
    res["recorded"] = recorded
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "texts"}, indent=1))


if __name__ == "__main__":
    main()

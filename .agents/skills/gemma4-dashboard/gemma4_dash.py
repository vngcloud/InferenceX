#!/usr/bin/env python3
"""Gemma optimization journey dashboard: import run artifacts into docs/gemma4-configs.json and build
docs/gemma-optimization-journey.html. Scenarios: SpeedBench high/low entropy 8k (import) and tau2 bench (import-tau2).

Run with `python3 -I` (stdlib only, needs `gh` logged in). Subcommands: list, new-config, import, import-tau2, build.
"""
import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import zipfile
from collections import OrderedDict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIGS = ROOT / "docs" / "gemma4-configs.json"
OUT_HTML = ROOT / "docs" / "gemma-optimization-journey.html"
TEMPLATE = HERE / "dashboard_template.html"
REPO = "vngcloud/InferenceX"
CONCS = (1, 8, 32, 64)
T2_CONCS = (1, 8, 16, 32)
T2_CAT = "t2"
CATS = ("hi", "lo")
ARMS = ("base", "mtp")
MODEL = "RedHatAI/gemma-4-31B-it-FP8-block"
DRAFT = "google/gemma-4-31B-it-assistant"
NAME_RE = re.compile(r"^(speedbench_aiperf|server_logs)_gemma4sba(.*?)(base|mtp)(hi|lo)_(.*)$")


# ---------------------------------------------------------------- statistics
def pct(sorted_vals, q):
    """Linear-interpolated percentile (same as numpy.percentile default)."""
    k = (len(sorted_vals) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def stats(vals):
    if not vals:
        return None
    s = sorted(vals)
    return dict(avg=sum(s) / len(s), p50=pct(s, 50), p90=pct(s, 90), p95=pct(s, 95), p99=pct(s, 99), max=s[-1])


def request_stats(lines):
    """aiperf profile_export.jsonl -> avg/p50/p90/p95/p99/max of ttft (s), tpot (ms), itl (ms), e2e (s).

    TPOT is per request, (e2e - ttft) / (osl - 1); ITL pools every inter-chunk gap of every request."""
    ttft, tpot, itl, e2e = [], [], [], []
    for line in lines:
        r = json.loads(line)
        if r["metadata"].get("was_cancelled"):
            continue
        m = r["metrics"]
        if "time_to_first_token" not in m or "request_latency" not in m:
            continue
        t, e = m["time_to_first_token"]["value"], m["request_latency"]["value"]
        n = (m.get("output_sequence_length") or m.get("output_token_count") or {}).get("value")
        ttft.append(t / 1000)
        e2e.append(e / 1000)
        if n and n > 1:
            tpot.append((e - t) / (n - 1))
        itl.extend((m.get("inter_chunk_latency") or {}).get("value") or [])
    return dict(ttft=stats(ttft), tpot=stats(tpot), itl=stats(itl), e2e=stats(e2e)), len(e2e)


# ---------------------------------------------------------------- server logs
def log_stats(txt, engine):
    ints = lambda pat: [int(x) for x in re.findall(pat, txt)]
    if engine == "vllm":
        pool = re.search(r"GPU KV cache size: ([\d,]+) tokens", txt)
        running = ints(r"Running: (\d+) reqs")
        return dict(full=int(pool.group(1).replace(",", "")) if pool else None, swa=None,
                    retract=len(re.findall(r"[Pp]reempt", txt)), peak=max(running) if running else None,
                    acc=sum(vacc) / len(vacc) if (vacc := [float(x) for x in re.findall(r"Mean acceptance length: ([\d.]+)", txt)]) else None)
    full = re.search(r"Full KV Cache is allocated.*?#tokens: (\d+)", txt)
    swa = re.search(r"SWA KV Cache is allocated.*?#tokens: (\d+)", txt)
    running = ints(r"Decode batch, #running-req: (\d+)")
    acc = [float(x) for x in re.findall(r"accept len: ([\d.]+)", txt)]
    return dict(full=int(full.group(1)) if full else None, swa=int(swa.group(1)) if swa else None,
                retract=len(re.findall(r"Retract requests|KV cache pool is full\. Retract", txt)),
                peak=max(running) if running else None, acc=sum(acc) / len(acc) if acc else None)


# ---------------------------------------------------------------- gh helpers
def gh_api(*args):
    p = subprocess.run(["gh", "api", *args], capture_output=True)
    if p.returncode:
        raise SystemExit(f"gh api {' '.join(args)} failed: {p.stderr.decode()[:300]}")
    return p.stdout


def run_artifacts(run):
    out = gh_api("--paginate", f"repos/{REPO}/actions/runs/{run}/artifacts?per_page=100", "--jq",
                 '.artifacts[]|"\\(.id) \\(.expired) \\(.name)"').decode()
    arts = []
    for line in out.splitlines():
        aid, expired, name = line.split(" ", 2)
        arts.append((aid, expired == "true", name))
    return arts


def download(aid):
    return zipfile.ZipFile(io.BytesIO(gh_api(f"repos/{REPO}/actions/artifacts/{aid}/zip")))


# ---------------------------------------------------------------- config file
def load():
    return json.loads(CONFIGS.read_text(), object_pairs_hook=OrderedDict)


def save(data):
    CONFIGS.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")


def find_config(data, cid):
    for c in data["configs"]:
        if c["id"] == cid:
            return c
    raise SystemExit(f"unknown config id {cid!r}; known: {', '.join(c['id'] for c in data['configs'])}")


def key_of(cell):
    return (cell["cfg"], cell["arm"], cell["cat"], cell["conc"])


# ---------------------------------------------------------------- commands
def cmd_list(_):
    data = load()
    have = {key_of(c) for c in data["cells"]}
    for c in data["configs"]:
        if c.get("scenario") == "tau2":
            want = [("mtp", T2_CAT, n) for n in T2_CONCS]
        else:
            want = [(a, k, n) for a in ARMS for k in CATS for n in CONCS]
        missing = [f"{a}/{k}/c{n}" for a, k, n in want if (c["id"], a, k, n) not in have]
        print(f"{c['id']:<10} {c['short']:<24} {c['engine']:<7} {c.get('scenario', 'speedbench'):<10} cells {len(want) - len(missing)}/{len(want)}"
              + (f"  missing: {' '.join(missing)}" if missing else ""))


def set_path(d, path, raw):
    try:
        val = json.loads(raw)
    except ValueError:
        val = raw
    d[path] = val


def cmd_new_config(a):
    data = load()
    if any(c["id"] == a.id for c in data["configs"]):
        raise SystemExit(f"config {a.id} already exists")
    cfg = OrderedDict(json.loads(json.dumps(find_config(data, a.copy_from)))) if a.copy_from else OrderedDict(id=a.id)
    cfg["id"] = a.id
    cfg["runs"] = ""
    cfg.pop("commit", None)
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        set_path(cfg, k, v)
    for need in ("short", "label", "engine", "context"):
        if need not in cfg:
            raise SystemExit(f"missing field {need}: pass --copy-from or --set {need}=...")
    data["configs"].append(cfg)
    save(data)
    print(f"added config {a.id} (copy of {a.copy_from or '-'}); edit docs/gemma4-configs.json for the rest, then import runs")


def parse_artifacts(arts):
    """{(key, arm, cat, suffix): {'aiperf': (id, name), 'logs': (id, name)}} for non-expired speedbench artifacts."""
    out = {}
    for aid, expired, name in arts:
        m = NAME_RE.match(name)
        if not m:
            continue
        kind = "aiperf" if m.group(1) == "speedbench_aiperf" else "logs"
        out.setdefault((m.group(2), m.group(3), m.group(4), m.group(5)), {})[kind] = (aid, expired, name)
    return out


def cmd_import(a):
    data = load()
    cfg = find_config(data, a.config)
    have = {key_of(c): i for i, c in enumerate(data["cells"])}
    added, notes = [], []
    for run in a.runs:
        groups = parse_artifacts(run_artifacts(run))
        keys = sorted({g[0] for g in groups})
        if not groups:
            notes.append(f"run {run}: no speedbench artifacts (expired or not a SpeedBench run)")
            continue
        if a.key is None and len(keys) > 1:
            raise SystemExit(f"run {run} holds several configs {keys}; pick one with --key")
        if a.key is not None and a.key not in keys:
            raise SystemExit(f"run {run} has no key {a.key!r}; its keys are {keys} (text after 'gemma4sba' in the artifact name)")
        for (key, arm, cat, suffix), parts in sorted(groups.items()):
            if a.key is not None and key != a.key:
                continue
            if "aiperf" not in parts or parts["aiperf"][1]:
                notes.append(f"run {run} {key}/{arm}/{cat}: aiperf artifact missing or expired")
                continue
            zf = download(parts["aiperf"][0])
            res = json.loads(zf.read(next(n for n in zf.namelist() if n.endswith("result.json"))))
            conc = res["max_concurrency"]
            status = (res.get("benchmark_outcome") or {}).get("status")
            if status != "passed" and not a.include_failed:
                notes.append(f"run {run} {arm}/{cat}/c{conc}: status {status}, skipped (use --include-failed)")
                continue
            lines = io.TextIOWrapper(io.BytesIO(zf.read(next(n for n in zf.namelist() if n.endswith("profile_export.jsonl")))),
                                     encoding="utf-8")
            st, n = request_stats(lines)
            cell = OrderedDict(cfg=cfg["id"], arm=arm, cat=cat, conc=conc, tput=res["output_throughput"],
                               ttot=res.get("total_token_throughput"),
                               acc=res.get("acceptance_length"), full=None, swa=None, retract=None, peak=None, run=run, n=n)
            if "logs" in parts and not parts["logs"][1]:
                lz = download(parts["logs"][0])
                logname = next((x for x in lz.namelist() if x.endswith("server.log")), None)
                if logname:
                    ls = log_stats(lz.read(logname).decode("utf-8", "ignore"), cfg["engine"])
                    cell.update(full=ls["full"], swa=ls["swa"], retract=ls["retract"], peak=ls["peak"])
                    if cell["acc"] is None and arm == "mtp":
                        cell["acc"] = ls["acc"]
            else:
                notes.append(f"run {run} {arm}/{cat}/c{conc}: no server log; pool/retract/peak left empty")
            cell["stats"] = st
            k = key_of(cell)
            if k in have and not a.replace:
                notes.append(f"{'/'.join(map(str, k))}: already imported (run {data['cells'][have[k]].get('run')}), skipped; use --replace")
                continue
            if k in have:
                data["cells"][have[k]] = cell
            else:
                have[k] = len(data["cells"])
                data["cells"].append(cell)
            added.append(cell)
        if run not in cfg.get("runs", ""):
            cfg["runs"] = (cfg.get("runs", "") + " / " + run).strip(" /")
    data["cells"].sort(key=lambda c: ([x["id"] for x in data["configs"]].index(c["cfg"]), c["arm"], c["cat"] != "hi", c["conc"]))
    save(data)
    f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
    print(f"{'cell':<16}{'tput':>8}{'TTFT p90':>10}{'TPOT p90':>10}{'ITL p99':>9}{'E2E p90':>9}{'acc':>6}{'retr':>6}{'peak':>6}{'n':>5}")
    for c in sorted(added, key=lambda c: (c["arm"], c["cat"], c["conc"])):
        s = c["stats"]
        print(f"{c['arm']}/{c['cat']}/c{c['conc']:<8}{f(c['tput']):>8}{f(s['ttft']['p90']):>10}{f(s['tpot']['p90']):>10}"
              f"{f(s['itl']['p99'], 0):>9}{f(s['e2e']['p90']):>9}{f(c['acc'], 2):>6}{f(c['retract'], 0):>6}{f(c['peak'], 0):>6}{c['n']:>5}")
    for n in notes:
        print("NOTE:", n)
    cmd_list(None)


T2_RE = re.compile(r"^(agentic|bmk_agentic|server_logs)_gemma4tau2_.*-([0-9a-f]{20})$")


def cmd_import_tau2(a):
    """tau2 (agentic) runs: one artifact triple per conc: agentic_* (aiperf files), bmk_agentic_* (aggregate json), server_logs_*."""
    data = load()
    cfg = find_config(data, a.config)
    if cfg.get("scenario") != "tau2":
        raise SystemExit(f"config {a.config} is not a tau2 config (set scenario=tau2 with new-config --set scenario=tau2)")
    have = {key_of(c): i for i, c in enumerate(data["cells"])}
    added, notes = [], []
    for run in a.runs:
        groups = {}
        for aid, expired, name in run_artifacts(run):
            m = T2_RE.match(name)
            if m:
                groups.setdefault(m.group(2), {})[m.group(1)] = (aid, expired)
        if not groups:
            notes.append(f"run {run}: no gemma4tau2 artifacts (expired or not a tau2 run)")
            continue
        for h, parts in sorted(groups.items()):
            if any(k not in parts or parts[k][1] for k in ("agentic", "bmk_agentic")):
                notes.append(f"run {run} {h}: aiperf or aggregate artifact missing or expired, skipped")
                continue
            bz = download(parts["bmk_agentic"][0])
            agg = json.loads(bz.read(next(n for n in bz.namelist() if n.endswith(".json"))))
            conc = agg["conc"]
            if conc not in T2_CONCS and not a.any_conc:
                notes.append(f"run {run} c{conc}: not in the tau2 ladder {T2_CONCS}, skipped (--any-conc to keep)")
                continue
            if agg["num_requests_successful"] == 0:
                notes.append(f"run {run} c{conc}: no successful requests, skipped")
                continue
            az = download(parts["agentic"][0])
            raw = az.read("aiperf_artifacts/profile_export.jsonl").decode("utf-8", "ignore").splitlines()
            recs = [json.loads(l) for l in raw]
            prof = [l for l, r in zip(raw, recs) if r["metadata"].get("benchmark_phase") == "profiling"]
            # errors per phase: nearly all errored requests are warmup ones (cold start), which the harness drops
            perr = sum(1 for r in recs if r.get("error") and r["metadata"].get("benchmark_phase") == "profiling")
            werr = sum(1 for r in recs if r.get("error") and r["metadata"].get("benchmark_phase") != "profiling")
            st, n = request_stats(prof)
            sm, th = agg.get("server_metrics") or {}, agg["request_metrics"]["throughput"]["per_gpu"]
            cache = (sm.get("cache") or {})
            cell = OrderedDict(cfg=cfg["id"], arm="mtp", cat=T2_CAT, conc=conc, tput=th["output_tput_tps"], ttot=th["total_tput_tps"],
                               hit=cache.get("overall_cache_hit_rate"), ok=agg["num_requests_successful"],
                               err=perr, werr=werr, power=sm.get("avg_power_w"),
                               acc=None, full=sm.get("kv_cache_pool_tokens"), swa=None, retract=None, peak=None, run=run, n=n)
            if "server_logs" in parts and not parts["server_logs"][1]:
                lz = download(parts["server_logs"][0])
                logname = next((x for x in lz.namelist() if x.endswith("server.log")), None)
                if logname:
                    ls = log_stats(lz.read(logname).decode("utf-8", "ignore"), cfg["engine"])
                    cell.update(full=ls["full"] or cell["full"], swa=ls["swa"], retract=ls["retract"], peak=ls["peak"], acc=ls["acc"])
            else:
                notes.append(f"run {run} c{conc}: no server log; pool/retract/peak left empty")
            cell["stats"] = st
            k = key_of(cell)
            if k in have and not a.replace:
                notes.append(f"{'/'.join(map(str, k))}: already imported (run {data['cells'][have[k]].get('run')}), skipped; use --replace")
                continue
            if k in have:
                data["cells"][have[k]] = cell
            else:
                have[k] = len(data["cells"])
                data["cells"].append(cell)
            added.append(cell)
            agg_lat = agg["request_metrics"]["latency"]
            notes.append(f"c{conc} cross-check against the harness aggregate (profiling phase): TTFT p90 {agg_lat['ttft']['p90']:.3f}s vs ours {st['ttft']['p90']:.3f}s; "
                         f"E2E p90 {agg_lat['e2el']['p90']:.3f}s vs ours {st['e2e']['p90']:.3f}s; {agg['num_requests_successful']} ok vs {n} records")
        if run not in cfg.get("runs", ""):
            cfg["runs"] = (cfg.get("runs", "") + " / " + run).strip(" /")
    order = [x["id"] for x in data["configs"]]
    data["cells"].sort(key=lambda c: (order.index(c["cfg"]), c["arm"], c["cat"] != "hi", c["conc"]))
    save(data)
    f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
    print(f"{'cell':<10}{'out t/s':>9}{'tot t/s':>9}{'TTFT p90':>10}{'TPOT p90':>10}{'E2E p90':>9}{'hit':>6}{'acc':>6}{'retr':>6}{'peak':>6}{'pool':>9}{'ok':>6}{'err':>5}")
    for c in sorted(added, key=lambda c: c["conc"]):
        s = c["stats"]
        print(f"c{c['conc']:<9}{f(c['tput']):>9}{f(c['ttot'], 0):>9}{f(s['ttft']['p90'], 2):>10}{f(s['tpot']['p90']):>10}{f(s['e2e']['p90'], 2):>9}"
              f"{f(c['hit'], 3):>6}{f(c['acc'], 2):>6}{f(c['retract'], 0):>6}{f(c['peak'], 0):>6}{f(c['full'], 0):>9}{c['ok']:>6}{c['err']:>5}")
    for n in notes:
        print("NOTE:", n)
    cmd_list(None)


# ---------------------------------------------------------------- serve command text
def serve_cmd_tau2(cfg, conc):
    """Serve command of the tau2 recipes (gemma4tau2_fp8_h200_mtp.sh for vLLM, gemma4tau2_fp8_h200_sglang_mtp.sh for SGLang)."""
    cc = "$CONC" if conc is None else str(conc)
    name = "google/gemma-4-31b-it"
    if cfg["engine"] == "vllm":
        return (f"# image {cfg['image']}, 1×H200 (TP1), MTP4. VLLM_ATTENTION_BACKEND is ignored by v0.30.0 (logged as unknown); vLLM auto-selects FLASH_ATTN, FlashAttention 4.\n"
                "VLLM_DISABLE_COMPILE_CACHE=1 NCCL_P2P_LEVEL=NVL VLLM_ATTENTION_BACKEND=FLASHINFER \\\n"
                f"vllm serve {MODEL} --served-model-name {name} gemma-4-31b-it --host 0.0.0.0 --port $PORT \\\n"
                f"    --trust-remote-code --tensor-parallel-size 1 --gpu-memory-utilization {cfg['mem']} \\\n"
                f"    --max-model-len {cfg['context']} --max-num-seqs 64 --max-num-batched-tokens 16384 \\\n"
                f"    --enable-chunked-prefill --long-prefill-token-threshold 8192 \\\n"
                f"    --speculative-config '{{\"method\": \"mtp\", \"model\": \"{DRAFT}\", \"num_speculative_tokens\": 4}}' \\\n"
                f"    --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4")
    cap = cfg.get("mtp_max_running", 32)
    maxrun = f"min($CONC, {cap})" if conc is None else str(min(conc, cap))
    ratio = (f"--swa-full-tokens-ratio {cfg['ratio']}" if cfg.get("ratio_flag", True)
             else "# --swa-full-tokens-ratio not passed: engine default 0.8")
    return (f"# image {cfg['image']}, 1×H200 (TP1), MTP4\n"
            f"SGLANG_SWA_EVICTION_INTERVAL={cfg['evict']} \\\n"
            f"python3 -m sglang.launch_server --model-path {MODEL} --served-model-name {name} \\\n"
            f"    --host 0.0.0.0 --port $PORT --trust-remote-code --tp 1 \\\n"
            f"    --mem-fraction-static {cfg['mem']} {ratio} --context-length {cfg['context']} \\\n"
            f"    --max-running-requests {maxrun} --chunked-prefill-size {cfg['chunk']} \\\n"
            f"    --attention-backend fa4 --schedule-policy hrrn --enable-metrics --enable-cache-report \\\n"
            f"    --speculative-algorithm NEXTN --speculative-draft-model-path {DRAFT} \\\n"
            f"    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \\\n"
            f"    --tool-call-parser gemma4 --reasoning-parser gemma4")


def serve_cmd(cfg, arm, conc):
    if cfg.get("scenario") == "tau2":
        return serve_cmd_tau2(cfg, conc)
    cc = "$CONC" if conc is None else str(conc)
    if cfg["engine"] == "vllm":
        pc = "--enable-prefix-caching" if cfg["radix"] else "--no-enable-prefix-caching"
        spec = (f"--speculative-config '{{\"method\": \"mtp\", \"model\": \"{DRAFT}\", \"num_speculative_tokens\": 4}}' \\\n    "
                if arm == "mtp" else "")
        return ("# image vllm/vllm-openai:v0.30.0, 1×H200 (TP1). VLLM_ATTENTION_BACKEND is ignored by v0.30.0 (logged as unknown); vLLM auto-selects FLASH_ATTN, FlashAttention 4.\n"
                "VLLM_DISABLE_COMPILE_CACHE=1 NCCL_P2P_LEVEL=NVL VLLM_ATTENTION_BACKEND=FLASHINFER \\\n"
                f"vllm serve {MODEL} --host 0.0.0.0 --port $PORT --served-model-name {MODEL} \\\n"
                f"    --trust-remote-code --tensor-parallel-size 1 --gpu-memory-utilization 0.92 \\\n"
                f"    --max-model-len {cfg['context']} --max-num-seqs {cc} --max-num-batched-tokens 16384 \\\n"
                f"    --enable-chunked-prefill --long-prefill-token-threshold 8192 {pc} \\\n"
                f"    {spec}--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4")
    radix = f"--swa-full-tokens-ratio {cfg['ratio']}" if cfg["radix"] else "--disable-radix-cache"
    evict, maxrun, cap = cfg.get("evict"), cc, cfg.get("mtp_max_running", 32)
    if arm == "mtp" and conc is None:
        maxrun = f"min($CONC, {cap})" if cap < 64 else cc
    elif arm == "mtp" and conc > cap:
        maxrun = str(cap)  # the FROZEN_KV_MTP worst-case SWA pool does not fit above this many slots
        evict = evict if evict is not None else 32
    env = f"SGLANG_SWA_EVICTION_INTERVAL={evict} \\\n" if evict is not None else ""
    img = cfg.get("image_base") if arm == "base" and cfg.get("image_base") else cfg["image"]
    spec = (f"--speculative-algorithm NEXTN --speculative-draft-model-path {DRAFT} \\\n    "
            "--speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \\\n    ") if arm == "mtp" else ""
    extra = (cfg["extra"] + " ") if cfg.get("extra") else ""
    return (f"# image {img}, 1×H200 (TP1)\n" + env
            + f"python3 -m sglang.launch_server --model-path {MODEL} --served-model-name {MODEL} \\\n"
            f"    --host 0.0.0.0 --port $PORT --trust-remote-code --tp 1 \\\n"
            f"    --mem-fraction-static {cfg['mem']} --context-length {cfg['context']} \\\n"
            f"    --max-running-requests {maxrun} --chunked-prefill-size {cfg['chunk']} \\\n"
            f"    {radix} --attention-backend fa4 --enable-metrics {extra}\\\n"
            f"    {spec}--tool-call-parser gemma4 --reasoning-parser gemma4")


def cmd_backfill_ttot(a):
    """Fill `ttot` (total_token_throughput of result.json) into SpeedBench cells imported before it was stored."""
    data = load()
    t2 = {c["id"] for c in data["configs"] if c.get("scenario") == "tau2"}
    todo = [c for c in data["cells"] if c["cfg"] not in t2 and c.get("ttot") is None]
    done = miss = 0
    cache = {}

    def results_of(run):
        if run not in cache:
            cache[run] = []
            for _, parts in sorted(parse_artifacts(run_artifacts(run)).items()):
                if "aiperf" in parts and not parts["aiperf"][1]:
                    zf = download(parts["aiperf"][0])
                    cache[run].append(json.loads(zf.read(next(n for n in zf.namelist() if n.endswith("result.json")))))
        return cache[run]

    for c in todo:
        runs = [c["run"]] if c.get("run") else find_config(data, c["cfg"]).get("runs", "").split(" / ")
        hit = [r for run in runs for r in results_of(run) if r["max_concurrency"] == c["conc"] and r.get("sb_arm", c["arm"]) == c["arm"]
               and abs(r["output_throughput"] - c["tput"]) <= 1e-6 * c["tput"] and r.get("total_token_throughput") is not None]
        if hit:
            c["ttot"] = hit[0]["total_token_throughput"]
            done += 1
        else:
            miss += 1
            print(f"no match: {c['cfg']}/{c['arm']}/{c['cat']}/c{c['conc']} (artifact expired?)")
    save(data)
    print(f"backfilled {done} cells, {miss} without a match")


def cmd_build(a):
    data = load()
    by_id = {c["id"]: c for c in data["configs"]}
    out = OrderedDict(meta=dict(data["meta"], generated=date.today().isoformat()), configs=[], cells=[])
    for c in data["configs"]:
        c = OrderedDict(c)
        c["cmds"] = {arm: serve_cmd(c, arm, None) for arm in (("mtp",) if c.get("scenario") == "tau2" else ARMS)}
        out["configs"].append(c)
    for cell in data["cells"]:
        if cell["cfg"] not in by_id:
            raise SystemExit(f"cell references unknown config {cell['cfg']}")
        cell = OrderedDict(cell)
        cell["cmd"] = serve_cmd(by_id[cell["cfg"]], cell["arm"], cell["conc"])
        out["cells"].append(cell)
    tpl = TEMPLATE.read_text()
    if "/*__DATA__*/null" not in tpl:
        raise SystemExit("template placeholder /*__DATA__*/null not found")
    OUT_HTML.write_text(tpl.replace("/*__DATA__*/null", json.dumps(out, ensure_ascii=False)))
    print(f"built {OUT_HTML} ({len(out['configs'])} configs, {len(out['cells'])} cells)")
    for d in a.copy_to or []:
        shutil.copy(OUT_HTML, Path(d) / OUT_HTML.name)
        print(f"copied to {Path(d) / OUT_HTML.name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="configs and cell coverage").set_defaults(fn=cmd_list)
    n = sub.add_parser("new-config", help="add a config entry")
    n.add_argument("--id", required=True)
    n.add_argument("--copy-from")
    n.add_argument("--set", action="append", metavar="FIELD=VALUE", help="JSON value or plain string")
    n.set_defaults(fn=cmd_new_config)
    i = sub.add_parser("import", help="import the SpeedBench artifacts of runs into a config")
    i.add_argument("--config", required=True)
    i.add_argument("--key", help="exp-name token (e.g. at15) when a run holds several configs")
    i.add_argument("--replace", action="store_true")
    i.add_argument("--include-failed", action="store_true")
    i.add_argument("runs", nargs="+")
    i.set_defaults(fn=cmd_import)
    t = sub.add_parser("import-tau2", help="import the tau2 (agentic) artifacts of runs into a tau2 config")
    t.add_argument("--config", required=True)
    t.add_argument("--replace", action="store_true")
    t.add_argument("--any-conc", action="store_true", help="keep conc values outside the ladder 1/8/16/32")
    t.add_argument("runs", nargs="+")
    t.set_defaults(fn=cmd_import_tau2)
    sub.add_parser("backfill-ttot", help="add total throughput to SpeedBench cells imported earlier").set_defaults(fn=cmd_backfill_ttot)
    b = sub.add_parser("build", help="write docs/gemma-optimization-journey.html")
    b.add_argument("--copy-to", action="append", metavar="DIR", help="also copy the html into DIR (e.g. the main checkout's docs/)")
    b.set_defaults(fn=cmd_build)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Gemma4 SpeedBench dashboard: import run artifacts into docs/gemma4-configs.json and build docs/gemma4-dashboard.html.

Run with `python3 -I` (stdlib only, needs `gh` logged in). Subcommands: list, new-config, import, build.
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
OUT_HTML = ROOT / "docs" / "gemma4-dashboard.html"
TEMPLATE = HERE / "dashboard_template.html"
REPO = "vngcloud/InferenceX"
CONCS = (1, 8, 32, 64)
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
                    retract=len(re.findall(r"[Pp]reempt", txt)), peak=max(running) if running else None, acc=None)
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
        missing = [f"{a}/{k}/c{n}" for a in ARMS for k in CATS for n in CONCS if (c["id"], a, k, n) not in have]
        print(f"{c['id']:<10} {c['short']:<16} {c['engine']:<7} cells {16 - len(missing)}/16"
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


# ---------------------------------------------------------------- serve command text
def serve_cmd(cfg, arm, conc):
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


def cmd_build(a):
    data = load()
    by_id = {c["id"]: c for c in data["configs"]}
    out = OrderedDict(meta=dict(data["meta"], generated=date.today().isoformat()), configs=[], cells=[])
    for c in data["configs"]:
        c = OrderedDict(c)
        c["cmds"] = {arm: serve_cmd(c, arm, None) for arm in ARMS}
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
    b = sub.add_parser("build", help="write docs/gemma4-dashboard.html")
    b.add_argument("--copy-to", action="append", metavar="DIR", help="also copy the html into DIR (e.g. the main checkout's docs/)")
    b.set_defaults(fn=cmd_build)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

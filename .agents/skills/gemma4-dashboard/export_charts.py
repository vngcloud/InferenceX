"""Export the two headline charts (SpeedBench high entropy, tau2 bench) as PNG + SVG.

One line per engine: the best SGLang configuration and the vLLM reference of meta.best.<hi|t2>, MTP arm,
output throughput per GPU vs concurrency. Usage: python3 -I export_charts.py [out_dir]
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
data = json.load(open(os.path.join(root, "docs", "gemma4-configs.json")))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "docs")
cfgs = {c["id"]: c for c in data["configs"]}
SGL, VLLM = "#2e7d32", "#c2410c"


def series(cfg, arm, cat):
    cs = sorted((c for c in data["cells"] if c["cfg"] == cfg and c["arm"] == arm and c["cat"] == cat and c["tput"] is not None), key=lambda c: c["conc"])
    return cs


PANELS = [("ttot", "Total throughput, input + output (tok/s per GPU)"), ("e2e", "E2E latency p90 (s)")]


def value(c, m):
    return c[m] if m == "ttot" else c["stats"][m]["p90"]


def above(c, m, b, cat):
    """id of the config whose point is higher at this concurrency (its label goes above, the other below)"""
    vs = {}
    for cid in (b["cfg"], b["ref"]):
        o = next((x for x in series(cid, b["arm"], cat) if x["conc"] == c["conc"]), None)
        if o is not None and (m == "ttot" or o.get("stats", {}).get(m)):
            vs[cid] = value(o, m)
    return max(vs, key=vs.get)


def chart(key, cat, title, tag, note, fname):
    b = data["meta"]["best"][key]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    for ax, (m, ylab) in zip(axes, PANELS):
        for cid, col, name in ((b["cfg"], SGL, "SGLang"), (b["ref"], VLLM, "vLLM")):
            cs = [c for c in series(cid, b["arm"], cat) if (m == "ttot" and c.get("ttot") is not None) or (m != "ttot" and c.get("stats", {}).get(m))]
            ax.plot([c["conc"] for c in cs], [value(c, m) for c in cs], "-^", color=col, lw=2.4, ms=8, label=cfgs[cid]["short"].replace(" MTP4", "") + " · MTP4")
            for c in cs:
                v = value(c, m)
                ax.annotate(f"{v:.0f}" if m == "ttot" else f"{v:.1f}", (c["conc"], v), textcoords="offset points",
                            xytext=(0, 9 if above(c, m, b, cat) == cid else -16), ha="center", fontsize=8, color=col)
        ax.set_xscale("log", base=2)
        ticks = sorted({c["conc"] for c in data["cells"] if c["cat"] == cat})
        ax.set_xticks(ticks); ax.set_xticklabels([f"c{t}" for t in ticks]); ax.minorticks_off()
        ax.set_xlabel("Concurrency"); ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0); ax.grid(alpha=.3)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.9, tag, fontsize=10.5, fontweight="bold", va="center", bbox=dict(boxstyle="round,pad=0.4", fc="#fff7e0", ec="#d9a400"))
    fig.text(0.14 if len(tag) < 12 else 0.2, 0.9, note, fontsize=9.5, color="#444", va="center")
    fig.text(0.01, 0.01, "Gemma-4 31B FP8-block · 1×H200 · MTP depth 4 · 256k context", fontsize=7.5, color="#777")
    fig.tight_layout(rect=(0, .02, 1, .88))
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out, f"{fname}.{ext}"))
    plt.close(fig)
    print("wrote", fname)


chart("hi", "hi", "SpeedBench (high entropy, 8k in / 1k out)", "NO CACHE",
      "Every prompt is unique: no prefix-cache reuse.", "gemma4-speedbench-hi")
hits = [c["hit"] for c in data["cells"] if c["cat"] == "t2" and c["cfg"] in (data["meta"]["best"]["t2"]["cfg"], data["meta"]["best"]["t2"]["ref"]) and c.get("hit") is not None]
chart("t2", "t2", "tau2 bench (agentic customer-service replay)", "AGENTIC · WITH CACHE",
      f"Multi-turn agent traces (~14.7k in / 200 out): prefix-cache hit {min(hits)*100:.0f}-{max(hits)*100:.0f}%.", "gemma4-tau2")

"""Render report data (extract.build_data) as blocks, then markdown or HTML."""

from __future__ import annotations

import re
from dataclasses import dataclass

from extract import PCTS, parse_grid, sla_text, tokens

DASH = "—"
KEY_FINDINGS_PLACEHOLDER = "<!-- KEY_FINDINGS: write 3-6 bullets from data.json, then re-run with --findings (see SKILL.md) -->"
THROUGHPUT = [("RPS", ("rps",)), ("Input tok/s", ("input_tps",)), ("Output tok/s", ("output_tps",)),
              ("Total tok/s", ("total_tps",)), ("Output tok/s/GPU", ("output_tps_per_gpu",))]
CACHE = [("Hit overall", "overall"), ("Hit GPU (HBM)", "gpu"), ("Hit CPU (DRAM)", "cpu"),
         ("Hit external (L3)", "external"), ("Hit router", "router")]
LATENCY = [("TTFT", "ttft"), ("ITL", "itl"), ("E2E latency", "e2e")]
FIXED_MATRICES = [("Output tok/s", ("metrics", "throughput", "output_tps"), True),
                  ("TTFT p99 (ms)", ("metrics", "ttft", "p99"), False),
                  ("ITL p99 (ms)", ("metrics", "itl", "p99"), False)]


@dataclass
class Table:
    headers: list[str]
    rows: list[list[tuple[str, str | None]]]


def num(v: float | None) -> str:
    if v is None:
        return DASH
    a = abs(v)
    return f"{v:,.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}"


def pct(v: float | None) -> str:
    return DASH if v is None else f"{v * 100:.1f}%"


def err_pct(v: float | None) -> str:
    return DASH if v is None else f"{v:.2f}%"


def _tok(v: float) -> str:
    return f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.1f}k" if v >= 1e3 else f"{v:.0f}"


def get(obj, *path):
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _delta(value: float, base: float, higher_better: bool) -> tuple[str, str | None]:
    d = (value - base) / abs(base) * 100
    tone = None if abs(d) < 0.5 else ("good" if (d > 0) == higher_better else "bad")
    return f" ({d:+.1f}%)", tone


def cell(value, base, fmt, higher_better) -> tuple[str, str | None]:
    text = fmt(value)
    if value is None or base is None or base == 0:
        return text, None
    suffix, tone = _delta(value, base, higher_better)
    return text + suffix, tone


def _arm_order(data) -> dict[str, int]:
    return {a["label"]: i for i, a in enumerate(data["arms"])}


def _scenario_sort(s):
    g = parse_grid(s)
    if g:
        return (0, tokens(g["isl"]), tokens(g["osl"]), g["cache"], s)
    return (1 if s else 2, 0, 0, 0, s or "")


def _label(data, arm: str) -> str:
    return arm + (" (baseline)" if arm == data["baseline"] and len(data["arms"]) > 1 else "")


def _conc(p) -> str:
    return DASH if p.get("conc") is None else str(p["conc"])


def _base(data, pts, p):
    if p["arm"] == data["baseline"]:
        return None
    for q in pts:
        if q["arm"] == data["baseline"] and q["scenario"] == p["scenario"] and (
                p["mode"] == "search" or q["conc"] == p["conc"]):
            return q
    return None


def drop_empty(headers, rows, droppable: set[int]) -> Table:
    keep = [i for i in range(len(headers)) if i not in droppable or any(r[i][0] != DASH for r in rows)]
    return Table([headers[i] for i in keep], [[r[i] for i in keep] for r in rows])


def _link(s: str) -> str:
    m = re.search(r"/runs/(\d+)$", s)
    return f"[run {m.group(1)}]({s})" if m else f"`{s}`"


def runs_table(data) -> Table:
    rows = []
    for a in data["arms"]:
        i = a["identity"]
        rows.append([(_label(data, a["label"]), None)] + [
            (str(v) if v is not None else DASH, None) for v in
            (i.get("model"), i.get("hw"), i.get("framework"), i.get("precision"), i.get("num_gpus"))
        ] + [(f"{i.get('tp') or DASH}/{i.get('ep') or DASH}", None), (i.get("image") or DASH, None),
             (", ".join(_link(s) for s in a["sources"]), None)])
    return Table(["Arm", "Model", "HW", "Framework", "Precision", "GPUs", "TP/EP", "Image", "Sources"], rows)


def throughput_table(data, pts) -> list:
    headers = ["Arm", "CCU", *(h for h, _ in THROUGHPUT), *(h for h, _ in CACHE), "Cached tokens by tier"]
    rows = []
    for p in pts:
        b = _base(data, pts, p)
        row = [(_label(data, p["arm"]), None), (_conc(p), None)]
        row += [cell(get(p, "metrics", "throughput", *k), get(b, "metrics", "throughput", *k), num, True)
                for _, k in THROUGHPUT]
        row += [cell(get(p, "cache", k), get(b, "cache", k), pct, True) for _, k in CACHE]
        if get(p, "cache", "source") == "client":
            text, tone = row[2 + len(THROUGHPUT)]
            row[2 + len(THROUGHPUT)] = (text + " (client)", tone)
        tiers = get(p, "cache", "tokens_by_tier") or {}
        row.append((" · ".join(f"{k} {_tok(v)}" for k, v in tiers.items()) or DASH, None))
        rows.append(row)
    table = drop_empty(headers, rows, set(range(2 + len(THROUGHPUT), len(headers))))
    blocks = [("h", 3, "Throughput & cache"), ("table", table)]
    if not any(h in table.headers for h, _ in CACHE):
        blocks.append(("p", "No server metrics logged; cache hit rates unavailable."))
    return blocks


def latency_tables(data, pts) -> list:
    blocks = []
    for title, key in LATENCY:
        headers = ["Arm", "CCU", *PCTS] + (["Out tok/s/user", "Errors"] if key == "e2e" else [])
        rows = []
        for p in pts:
            b = _base(data, pts, p)
            row = [(_label(data, p["arm"]), None), (_conc(p), None)]
            row += [cell(get(p, "metrics", key, s), get(b, "metrics", key, s), num, False) for s in PCTS]
            if key == "e2e":
                row.append(cell(get(p, "metrics", "tps_per_user"), get(b, "metrics", "tps_per_user"), num, True))
                row.append(cell(get(p, "metrics", "error_rate_pct"), get(b, "metrics", "error_rate_pct"), err_pct, False))
            rows.append(row)
        blocks += [("h", 3, f"{title} (ms)"), ("table", Table(headers, rows))]
    warnings = [f"⚠ {p['arm']} c{p['conc']}: {n}" for p in pts for n in p["notes"] if n.startswith("validator dropped")]
    if warnings:  # spec: error-count disagreement goes right under the E2E table
        blocks.append(("p", "\n".join(f"- {w}" for w in warnings)))
    return blocks


def search_blocks(data, pts) -> list:
    blocks = []
    for p in pts:
        s = p.get("search")
        if p["mode"] != "search" or not s:
            continue
        max_conc = s["max_conc"] if s["max_conc"] is not None else DASH
        head = f"**{p['arm']}**: max CCU = **{max_conc}** · state `{s['state']}` · SLA: {', '.join(s['sla']) or DASH}"
        if s["binding"]:
            observed = f"{s['breach_observed']:.2f}" if s["breach_observed"] is not None else DASH
            head += f" · binding: {s['binding']} (observed {observed} at c{s['infeasible_min']})"
        if s["best_trials_conc"] is not None and s["best_trials_conc"] != s["max_conc"]:
            head += f" · highest objective at c{s['best_trials_conc']}"
        rows = []
        for pr in sorted(s["probes"], key=lambda pr: (pr["conc"] is None, pr["conc"] or 0)):
            marks = " ".join(m for m, on in (("★ max CCU", pr["winner"]), ("◆ best objective", pr["best_objective"])) if on)
            rows.append([(str(pr["iter"]), None), (str(pr["conc"]), None),
                         ("✓", "good") if pr["feasible"] else ("✗", "bad"),
                         *[(num(pr["observed"].get(f"{f['metric_tag']}.{f['stat']}")), None) for f in s["sla_filters"]],
                         (num(pr["output_tps"]), None), (pct(pr["cache_overall"]), None), (marks, None)])
        headers = ["Iter", "CCU", "Pass", *s["sla"], "Output tok/s", "Hit overall", "Mark"]
        blocks += [("h", 3, f"Search: {p['arm']}"), ("p", head), ("table", Table(headers, rows))]
    return blocks


def _grid_points(data, kind):
    return [p for p in data["points"] if p.get("grid") and p["grid"]["kind"] == kind]


def search_matrix(data) -> list:
    pts = [p for p in _grid_points(data, "search") if p.get("search")]
    if not pts:
        return []
    workloads = sorted({(p["grid"]["isl"], p["grid"]["osl"]) for p in pts}, key=lambda k: (tokens(k[0]), tokens(k[1])))
    caches = sorted({p["grid"]["cache"] for p in pts})
    arms = [a["label"] for a in data["arms"] if any(p["arm"] == a["label"] for p in pts)]
    find = lambda arm, isl, osl, c: next((p for p in pts if p["arm"] == arm and
                                          (p["grid"]["isl"], p["grid"]["osl"], p["grid"]["cache"]) == (isl, osl, c)), None)
    rows = []
    for isl, osl in workloads:
        row = [(f"ISL {isl} · OSL {osl}", None)]
        for c in caches:
            base, texts, tone = find(data["baseline"], isl, osl, c), [], None
            for arm in arms:
                p = find(arm, isl, osl, c)
                if p is None:
                    texts.append(DASH)
                    continue
                s = p["search"]
                if s["state"] == "no_pass_in_range":
                    texts.append("✗")
                    continue
                text = f"{s['max_conc']}{'*' if s['state'] == 'cliff' else ''} · {num(get(p, 'metrics', 'throughput', 'output_tps'))}"
                if base is not None and p is not base and get(base, "search", "max_conc"):
                    suffix, tone = _delta(s["max_conc"], base["search"]["max_conc"], True)
                    text += suffix
                texts.append(text)
            row.append((" / ".join(texts), tone))
        rows.append(row)
    legend = ("Cell = max CCU under SLA · output tok/s at that CCU"
              + (f" (arms: {' / '.join(arms)})" if len(arms) > 1 else "")
              + ". `✗` no CCU passed the SLA, `*` cliff boundary.")
    return [("h", 3, "Max CCU under SLA"), ("p", legend),
            ("table", Table(["Workload", *(f"cache {c}%" for c in caches)], rows))]


def fixed_matrices(data) -> list:
    pts = _grid_points(data, "fixed")
    if not pts:
        return []
    order = _arm_order(data)
    concs = sorted({p["conc"] for p in pts if p["conc"] is not None})
    keys = sorted({(p["scenario"], p["arm"]) for p in pts}, key=lambda k: (_scenario_sort(k[0]), order[k[1]]))
    find = lambda scen, arm, c: next((q for q in pts if (q["scenario"], q["arm"], q["conc"]) == (scen, arm, c)), None)
    blocks = [("h", 3, "Fixed CCU grid")]
    for title, path, higher_better in FIXED_MATRICES:
        rows = []
        for scen, arm in keys:
            row = [(scen, None), (arm, None)]
            for c in concs:
                p = find(scen, arm, c)
                b = find(scen, data["baseline"], c) if arm != data["baseline"] else None
                row.append(cell(get(p, *path), get(b, *path), num, higher_better) if p else (DASH, None))
            rows.append(row)
        blocks += [("h", 4, title), ("table", Table(["Scenario", "Arm", *(f"c{c}" for c in concs)], rows))]
    return blocks


def cache_check(data) -> list:
    pts = [p for p in data["points"] if p.get("cache_check")]
    if not pts:
        return []
    order = _arm_order(data)
    rows = []
    for p in sorted(pts, key=lambda p: (_scenario_sort(p["scenario"]), order[p["arm"]], p["conc"] or 0)):
        cc = p["cache_check"]
        rows.append([(p["scenario"], None), (p["arm"], None), (_conc(p), None), (pct(cc["nominal"]), None),
                     (pct(cc["measured"]), None),
                     ("⚠ >10 pts off", "bad") if cc["mismatch"] else ("", None)])
    return [("h", 4, "Cache hit: nominal vs measured (overall)"),
            ("table", Table(["Scenario", "Arm", "CCU", "Nominal", "Measured", "Check"], rows))]


def config_blocks(data) -> list:
    blocks = []
    for a in data["arms"]:
        body = []
        for name, text in (a.get("config") or {}).items():
            if text:
                body += [("p", f"`{name}`"), ("code", "yaml" if name.endswith(".yaml") else "bash", text)]
        if body:
            blocks.append(("details", f"Config: {a['label']}", body))
    return [("h", 2, "Configs"), *blocks] if blocks else []


def notes_blocks(data) -> list:
    notes = list(data["notes"])
    points = data["points"]
    if any(get(p, "cache", "engine") == "vllm" for p in points):
        notes.append("vLLM: CPU and L3 hit rates are not separable; offload hits are shown under external (L3).")
    if any(get(p, "cache", "source") == "client" for p in points):
        notes.append("Hit overall marked (client) is aiperf's client-side prompt cache read %, not tiered.")
    if any(get(p, "metrics", "source") in ("aggregate", "search_aggregate") for p in points):
        notes.append("Rows without an aiperf export use the aggregate JSON: latency from seconds, p99 unavailable.")
    if len(data["arms"]) > 1:
        notes.append(f"Deltas are against the baseline arm `{data['baseline']}` at the same scenario and CCU "
                     "(search: same scenario).")
    return [("h", 2, "Notes"), ("p", "\n".join(f"- {n}" for n in notes))] if notes else []


def chart_blocks(data, pts) -> list:
    return []  # Task 8


def build_blocks(data, charts: bool = False) -> list:
    inputs = ", ".join(_link(i) for i in data["inputs"]) or DASH
    blocks = [
        ("h", 1, data["title"]),
        ("p", f"Generated {data['generated']} · inputs: {inputs} · downloaded {data.get('bytes_fetched', 0) / 1e6:.2f} MB"),
        ("h", 2, "Key findings"), ("p", data.get("findings") or KEY_FINDINGS_PLACEHOLDER),
        ("h", 2, "Runs"), ("table", runs_table(data)),
    ]
    grid = search_matrix(data) + fixed_matrices(data) + cache_check(data)
    if grid:
        blocks += [("h", 2, "Grid summary"), *grid]
    order = _arm_order(data)
    for scen in sorted({p["scenario"] for p in data["points"]}, key=_scenario_sort):
        pts = sorted([p for p in data["points"] if p["scenario"] == scen],
                     key=lambda p: (p["conc"] if p["conc"] is not None else -1, order[p["arm"]]))
        body = throughput_table(data, pts) + latency_tables(data, pts) + search_blocks(data, pts)
        if charts:
            body += chart_blocks(data, pts)
        title = f"Scenario: {scen or '(unknown)'}"
        grid_info = parse_grid(scen)
        if grid_info and grid_info["kind"] == "fixed":
            blocks.append(("details", title, body))
        else:
            blocks += [("h", 2, title), *body]
    return blocks + config_blocks(data) + notes_blocks(data)


def _md_table(t: Table) -> str:
    esc = lambda s: s.replace("|", "\\|")
    lines = ["| " + " | ".join(map(esc, t.headers)) + " |", "|" + "---|" * len(t.headers)]
    lines += ["| " + " | ".join(esc(text) for text, _ in row) + " |" for row in t.rows]
    return "\n".join(lines)


def to_markdown(blocks) -> str:
    out = []
    for block in blocks:
        kind = block[0]
        if kind == "h":
            out.append("#" * block[1] + " " + block[2])
        elif kind == "p":
            out.append(block[1])
        elif kind == "table":
            out.append(_md_table(block[1]))
        elif kind == "code":
            out.append(f"```{block[1]}\n{block[2].rstrip()}\n```")
        elif kind == "details":
            out.append(f"<details><summary>{block[1]}</summary>\n\n{to_markdown(block[2]).rstrip()}\n\n</details>")
        # "chart" blocks are HTML-only
    return "\n\n".join(out) + "\n"

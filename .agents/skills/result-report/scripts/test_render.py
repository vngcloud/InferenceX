import copy
import unittest
from pathlib import Path

import extract
import fetch
import render

FIX = Path(__file__).resolve().parent / "fixtures"


def data_for(*names, baseline=None):
    cfgs = [c for n in names for c in fetch.local_configs(fetch.parse_source(str(FIX / n)))]
    return extract.build_data(cfgs, [], baseline=baseline, inputs=[str(FIX / n) for n in names], generated="2026-10-23")


def grid_point(arm, scenario, conc, out_tps, ttft_p99, state=None, measured=0.5):
    g = extract.parse_grid(scenario)
    p = {"rf": scenario, "source": "s", "arm": arm, "scenario": scenario, "grid": g, "conc": conc,
         "mode": g["kind"], "identity": {}, "notes": [], "cache": {"overall": measured, "source": "aggregate"},
         "cache_check": {"nominal": g["cache"] / 100, "measured": measured,
                         "mismatch": abs(measured - g["cache"] / 100) > 0.10},
         "metrics": extract._empty_metrics("aiperf"), "search": None}
    p["metrics"]["throughput"]["output_tps"] = out_tps
    p["metrics"]["ttft"]["p99"] = ttft_p99
    p["metrics"]["itl"]["p99"] = ttft_p99 / 10 if ttft_p99 is not None else None
    if state:
        p["search"] = {"state": state, "max_conc": conc, "sla": [], "sla_filters": [], "binding": None,
                       "breach_observed": None, "best_trials_conc": None, "probes": [], "infeasible_min": None}
    return p


def grid_data(points, arms):
    return {"title": "t", "generated": "d", "inputs": [], "baseline": arms[0], "notes": [], "bytes_fetched": 0,
            "findings": None, "arms": [{"label": a, "identity": {}, "sources": [], "config": {}} for a in arms],
            "points": points}


class FormatTest(unittest.TestCase):
    def test_num_and_pct(self):
        self.assertEqual((render.num(None), render.num(78369.2), render.num(49.16), render.num(0.357)),
                         ("—", "78,369", "49.2", "0.36"))
        self.assertEqual(render.pct(0.43174), "43.2%")

    def test_delta_tone_follows_direction(self):
        self.assertEqual(render.cell(110.0, 100.0, render.num, True), ("110 (+10.0%)", "good"))
        self.assertEqual(render.cell(110.0, 100.0, render.num, False), ("110 (+10.0%)", "bad"))
        self.assertEqual(render.cell(110.0, None, render.num, True), ("110", None))


class ReportTest(unittest.TestCase):
    def test_fixed_report_hides_empty_cache_columns(self):
        md = render.to_markdown(render.build_blocks(data_for("gh_fixed")))
        self.assertIn("## Key findings", md)
        self.assertIn(render.KEY_FINDINGS_PLACEHOLDER, md)
        self.assertIn("## Scenario: agentx-cc-062126", md)
        self.assertIn("Hit CPU (DRAM)", md)
        self.assertIn("43.2%", md)
        self.assertNotIn("Hit external (L3)", md)
        self.assertIn("### TTFT (ms)", md)
        self.assertIn("723,840", md)
        self.assertIn("validator dropped 397", md)

    def test_aggregate_only_shows_dash_p99_and_note(self):
        md = render.to_markdown(render.build_blocks(data_for("aggonly")))
        self.assertIn("p99 unavailable", md)
        ttft = md.split("### TTFT (ms)")[1].split("###")[0]
        self.assertIn("| — |", ttft)

    def test_raw_folder_has_no_server_metrics(self):
        md = render.to_markdown(render.build_blocks(data_for("raw_aiperf")))
        self.assertIn("No server metrics logged", md)

    def test_search_headline_and_probe_table(self):
        md = render.to_markdown(render.build_blocks(data_for("search_local")))
        self.assertIn("max CCU = **60**", md)
        self.assertIn("binding: ITL p90 < 50 ms (observed 50.85 at c61)", md)
        self.assertIn("highest objective at c48", md)
        self.assertIn("★ max CCU", md)
        self.assertIn("## Grid summary", md)
        self.assertIn("| ISL 8k · OSL 1k | 60 · 1,335 |", md)

    def test_two_arms_get_deltas_against_baseline(self):
        data = data_for("gh_fixed")
        twin = copy.deepcopy(data["points"][0])
        twin["arm"] = "B"
        twin["metrics"]["throughput"]["output_tps"] *= 1.1
        data["points"].append(twin)
        data["arms"].append({"label": "B", "identity": {}, "sources": [], "config": {}})
        md = render.to_markdown(render.build_blocks(data))
        self.assertIn("(+10.0%)", md)
        self.assertIn("(baseline)", md)

    def test_search_grid_matrix_marks(self):
        pts = [grid_point("A", "sweep-isl8k-osl1k-cache0", 64, 2000.0, None, state="ok"),
               grid_point("A", "sweep-isl8k-osl1k-cache50", 0, None, None, state="no_pass_in_range"),
               grid_point("A", "sweep-isl32k-osl1k-cache0", 16, 900.0, None, state="cliff")]
        md = render.to_markdown(render.build_blocks(grid_data(pts, ["A"])))
        self.assertIn("| ISL 8k · OSL 1k | 64 · 2,000 | ✗ |", md)
        self.assertIn("| ISL 32k · OSL 1k | 16* · 900 | — |", md)

    def test_fixed_grid_matrices_and_cache_check(self):
        pts = [grid_point("A", "fixed-isl8k-osl1k-cache0", 8, 500.0, 900.0, measured=0.02),
               grid_point("A", "fixed-isl8k-osl1k-cache0", 32, 1500.0, 2500.0, measured=0.03),
               grid_point("A", "fixed-isl8k-osl1k-cache50", 8, 600.0, 700.0, measured=0.30)]
        md = render.to_markdown(render.build_blocks(grid_data(pts, ["A"])))
        self.assertIn("| Scenario | Arm | c8 | c32 |", md)
        self.assertIn("| fixed-isl8k-osl1k-cache50 | A | 600 | — |", md)
        self.assertIn("⚠", md)
        self.assertIn("<details><summary>Scenario: fixed-isl8k-osl1k-cache0</summary>", md)


if __name__ == "__main__":
    unittest.main()


class HtmlTest(unittest.TestCase):
    def test_line_chart_has_series_and_threshold(self):
        svg = render.line_chart("ITL", {"A": [(8, 40.0), (32, 55.0)]}, "CCU", "ms", threshold=50.0)
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("<polyline", svg)
        self.assertIn('stroke-dasharray="4 3"', svg)
        self.assertEqual(render.line_chart("x", {}, "a", "b"), "")

    def test_html_report_has_tables_tones_and_probe_chart(self):
        data = data_for("search_local")
        page = render.to_html(render.build_blocks(data, charts=True), data["title"])
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertIn("<table>", page)
        self.assertIn('class="good"', page)
        self.assertIn("<svg", page)
        self.assertIn("by probe CCU", page)

    def test_fixed_ladder_gets_three_charts(self):
        pts = [grid_point("A", "fixed-isl8k-osl1k-cache0", 8, 500.0, 900.0),
               grid_point("A", "fixed-isl8k-osl1k-cache0", 32, 1500.0, 2500.0)]
        blocks = render.chart_blocks(grid_data(pts, ["A"]), pts)
        self.assertEqual(len(blocks), 3)

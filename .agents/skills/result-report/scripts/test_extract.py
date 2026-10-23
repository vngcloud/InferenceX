import copy
import json
import tempfile
import unittest
from pathlib import Path

import extract
import fetch

FIX = Path(__file__).resolve().parent / "fixtures"
load = lambda rel: json.loads((FIX / rel).read_text())
GLM_AGG = "gh_fixed/bmk_agentic_glm53_c108/glm53_c108.json"
GLM_EXPORT = "gh_fixed/agentic_glm53_c108/aiperf_artifacts/profile_export_aiperf.json"
PROBE = "search_local/ag/aiperf_artifacts/search_iter_{:04d}/profile_runs/run_0001/"


class GridTest(unittest.TestCase):
    def test_parse_grid(self):
        self.assertEqual(extract.parse_grid("fixed-isl8k-osl1k-cache50"),
                         {"kind": "fixed", "isl": "8k", "osl": "1k", "cache": 50})
        self.assertEqual(extract.parse_grid("sweep-isl128k-osl512-cache0")["kind"], "search")
        self.assertIsNone(extract.parse_grid("agentx-cc-062126"))
        self.assertIsNone(extract.parse_grid(None))

    def test_tokens(self):
        self.assertEqual((extract.tokens("8k"), extract.tokens("512")), (8192, 512))


class AiperfTest(unittest.TestCase):
    def test_latency_ms_with_p99_and_errors(self):
        m = extract.from_aiperf(load(GLM_EXPORT))
        self.assertAlmostEqual(m["ttft"]["p99"], 723840.0278512397, places=3)
        self.assertAlmostEqual(m["throughput"]["input_tps"], 78369.27687594405, places=3)
        self.assertAlmostEqual(m["error_rate_pct"], 0.5054759898904801, places=6)
        self.assertEqual(m["errors"], 18)
        self.assertAlmostEqual(m["client_cache_hit"], 0.945903879813367, places=6)
        self.assertEqual(m["source"], "aiperf")
        self.assertIsNone(m["throughput"]["output_tps_per_gpu"])

    def test_seconds_are_converted(self):
        m = extract.from_aiperf({"time_to_first_token": {"unit": "sec", "avg": 1.5, "p99": 2.0}})
        self.assertEqual((m["ttft"]["avg"], m["ttft"]["p99"]), (1500.0, 2000.0))

    def test_no_error_metrics_means_zero_errors(self):
        m = extract.from_aiperf({"request_count": {"unit": "requests", "avg": 417.0}})
        self.assertEqual((m["error_rate_pct"], m["errors"]), (0.0, 0))


class AggregateTest(unittest.TestCase):
    def test_fallback_latency_seconds_to_ms_without_p99(self):
        agg = load("aggonly/bmk_agentic_gemma4_c96/gemma4_c96.json")
        m = extract.from_aggregate(agg)
        self.assertAlmostEqual(m["ttft"]["avg"], 357.13, places=2)
        self.assertIsNone(m["ttft"]["p99"])
        self.assertAlmostEqual(m["throughput"]["rps"], 11765 / 1202.89864, places=4)
        self.assertEqual(m["source"], "aggregate")

    def test_sglang_tiers_from_aggregate(self):
        c = extract.aggregate_tiers(load(GLM_AGG))
        self.assertEqual((c["gpu"], c["cpu"], c["overall"]), (0.51417, 0.43174, 0.9459))
        self.assertIsNone(c["external"])
        self.assertIsNone(c["router"])
        self.assertEqual(c["tokens_by_tier"], {"device": 146673664, "host": 123159104})

    def test_total_label_is_not_double_counted(self):
        agg = copy.deepcopy(load(GLM_AGG))
        agg["server_metrics"]["cache"]["cached_tokens_by_source"] = {"device": 100, "host": 50, "total": 150}
        agg["server_metrics"]["tokens"]["prompt_total"] = 300
        c = extract.aggregate_tiers(agg)
        self.assertEqual(c["overall"], 0.5)
        self.assertNotIn("total", c["tokens_by_tier"])

    def test_vllm_cpu_tier_is_not_separable(self):
        agg = copy.deepcopy(load(GLM_AGG))
        agg["server_metrics"]["adapter"] = "vllm"
        self.assertIsNone(extract.aggregate_tiers(agg)["cpu"])

    def test_search_aggregate_metrics(self):
        m = extract.from_search_aggregate(load("search_local/bmk/gemma4_search.json"))
        self.assertAlmostEqual(m["throughput"]["output_tps"], 1334.90684, places=4)
        self.assertAlmostEqual(m["itl"]["p90"], 49.15992, places=4)
        self.assertIsNone(m["itl"]["p99"])


class ServerMetricsTest(unittest.TestCase):
    def test_sglang_tiers_router_and_total_label(self):
        c = extract.tiers_from_server_metrics(load("sglang_server_metrics.json"))
        self.assertEqual((c["gpu"], c["cpu"], c["external"], c["overall"], c["router"]), (0.5, 0.2, 0.1, 0.8, 0.9))
        self.assertNotIn("total", c["tokens_by_tier"])
        self.assertEqual(c["engine"], "sglang")

    def test_vllm_probe(self):
        c = extract.tiers_from_server_metrics(load(PROBE.format(10) + "server_metrics_export.json"))
        self.assertAlmostEqual(c["gpu"], 0.48894, places=4)
        self.assertAlmostEqual(c["overall"], 0.48894, places=4)
        self.assertEqual(c["external"], 0.0)
        self.assertIsNone(c["cpu"])
        self.assertEqual(c["engine"], "vllm")
        self.assertNotIn("local_compute", c["tokens_by_tier"])


if __name__ == "__main__":
    unittest.main()


def _cfg(name):
    [cfg] = fetch.local_configs(fetch.parse_source(str(FIX / name)))
    return cfg


class SearchSummaryTest(unittest.TestCase):
    # source of the cases: infx/results/sweep.py:32 state order
    def test_states(self):
        ok = {"boundary_summary": {"feasible_max": {"value": 60}, "infeasible_min": {"value": 61},
                                   "boundary_type": "smooth"}, "convergence_reason": "precision", "iterations": [1, 2]}
        self.assertEqual(extract.summarize_search(ok)["state"], "ok")
        cliff = copy.deepcopy(ok); cliff["boundary_summary"]["boundary_type"] = "cliff"
        self.assertEqual(extract.summarize_search(cliff)["state"], "cliff")
        none = {"boundary_summary": {"feasible_max": None, "infeasible_min": {"value": 1}}}
        self.assertEqual(extract.summarize_search(none)["state"], "no_pass_in_range")
        self.assertEqual(extract.summarize_search({})["state"], "no_data")

    def test_sla_text(self):
        f = {"metric_tag": "inter_token_latency", "stat": "p90", "op": "lt", "threshold": 50.0}
        self.assertEqual(extract.sla_text(f), "ITL p90 < 50 ms")


class BuildPointTest(unittest.TestCase):
    def test_fixed_gh_layout(self):
        p = extract.build_point(_cfg("gh_fixed"))
        self.assertEqual((p["mode"], p["conc"], p["scenario"]), ("fixed", 108, "agentx-cc-062126"))
        self.assertIsNone(p["grid"])
        self.assertEqual(p["metrics"]["source"], "aiperf")
        self.assertAlmostEqual(p["metrics"]["throughput"]["output_tps_per_gpu"],
                               p["metrics"]["throughput"]["output_tps"] / 8)
        self.assertEqual(p["cache"]["cpu"], 0.43174)
        self.assertTrue(any("397" in n and "18" in n for n in p["notes"]))
        self.assertIn("--concurrency 108", p["config"]["benchmark_command.txt"])

    def test_aggregate_only_notes_missing_p99(self):
        p = extract.build_point(_cfg("aggonly"))
        self.assertEqual((p["metrics"]["source"], p["conc"]), ("aggregate", 96))
        self.assertIsNone(p["metrics"]["ttft"]["p99"])
        self.assertTrue(any("p99 unavailable" in n for n in p["notes"]))

    def test_search(self):
        p = extract.build_point(_cfg("search_local"))
        s = p["search"]
        self.assertEqual((p["mode"], p["conc"], s["state"]), ("search", 60, "ok"))
        self.assertEqual(p["scenario"], "sweep-isl8k-osl1k-cache50")
        self.assertEqual(p["grid"], {"kind": "search", "isl": "8k", "osl": "1k", "cache": 50})
        self.assertEqual(s["binding"], "ITL p90 < 50 ms")
        self.assertAlmostEqual(s["breach_observed"], 50.84856, places=4)
        self.assertEqual((s["infeasible_min"], s["best_trials_conc"]), (61, 48))
        self.assertEqual(len(s["probes"]), 12)
        winner = [pr for pr in s["probes"] if pr["winner"]]
        self.assertEqual([pr["conc"] for pr in winner], [60])
        self.assertAlmostEqual(p["metrics"]["throughput"]["output_tps"], 1334.9068416429348, places=6)
        self.assertAlmostEqual(winner[0]["observed"]["inter_token_latency.p90"], 49.1599168688172, places=6)
        self.assertAlmostEqual(p["cache"]["gpu"], 0.48894, places=4)
        self.assertEqual(p["cache_check"]["nominal"], 0.5)
        self.assertFalse(p["cache_check"]["mismatch"])

    def test_raw_aiperf(self):
        p = extract.build_point(_cfg("raw_aiperf"))
        self.assertEqual((p["mode"], p["conc"], p["scenario"]), ("fixed", 16, None))
        self.assertIsNone(p["cache"])
        self.assertEqual(p["metrics"]["source"], "aiperf")


class ArmsTest(unittest.TestCase):
    def _dir_cfg(self, root, name, scenario, conc):
        d = Path(root) / name
        (d / "aiperf_artifacts").mkdir(parents=True)
        (d / "serving.yaml").write_text(
            f"# comment {name}\nname: gemma4-hicache-1gpu-{scenario.removeprefix('fixed-')}\n"
            f"image: img\nserve: |\n  python3 -m sglang.launch_server --tp 1\nscenario: {scenario}\nconc: [8, 32]\n")
        (d / "benchmark_command.txt").write_text(
            f"aiperf profile --config /workspace/scenarios/{scenario}.yaml --concurrency {conc}\n")
        (d / "aiperf_artifacts/profile_export_aiperf.json").write_text(
            json.dumps({"output_token_throughput": {"unit": "tokens/sec", "avg": 100.0 + conc}}))
        return fetch.Config(rf=name, source=fetch.parse_source(root), agg=None, files=fetch.DirReader(d))

    def test_name_suffix_scenario_and_conc_do_not_split_an_arm(self):
        with tempfile.TemporaryDirectory() as root:
            cfgs = [self._dir_cfg(root, "a", "fixed-isl8k-osl1k-cache0", 8),
                    self._dir_cfg(root, "b", "fixed-isl8k-osl1k-cache50", 32)]
            data = extract.build_data(cfgs, [], inputs=[root], generated="2026-10-23")
        self.assertEqual([a["label"] for a in data["arms"]], ["gemma4-hicache-1gpu"])
        self.assertEqual(data["baseline"], "gemma4-hicache-1gpu")
        self.assertEqual({(p["scenario"], p["conc"]) for p in data["points"]},
                         {("fixed-isl8k-osl1k-cache0", 8), ("fixed-isl8k-osl1k-cache50", 32)})
        self.assertNotIn("config", data["points"][0])
        self.assertIn("serving.yaml", data["arms"][0]["config"])

    def test_aggregate_only_joins_the_yaml_arm_with_same_identity(self):
        full = _cfg("gh_fixed")
        only = fetch.Config(rf="x_c120", source=full.source, agg=copy.deepcopy(full.agg), files=None)
        data = extract.build_data([only, full], [])
        self.assertEqual(len(data["arms"]), 1)
        self.assertEqual(len(data["points"]), 2)

    def test_unknown_baseline(self):
        with self.assertRaises(ValueError):
            extract.build_data([_cfg("gh_fixed")], [], baseline="nope")


class FixRound1Test(unittest.TestCase):
    def test_sla_text_missing(self):
        self.assertEqual(extract.sla_text({"metric_tag": "inter_token_latency"}), "ITL — — —")
        self.assertEqual(extract.sla_text({"metric_tag": "x", "stat": "p90", "op": "lt", "threshold": 3}), "x p90 < 3")

    def test_arm_key_ignores_trailing_conc_and_indented_lists(self):
        def key(serving):
            p = {"config": {"serving.yaml": serving}, "identity": {}}
            return extract.arm_of(p, fetch.Config(rf="r", source=fetch.parse_source("/tmp"), agg=None, files=None))[0]
        base = "name: a\nimage: i\n"
        self.assertEqual(key(base + "conc: 1"), key(base + "conc: 2"))
        self.assertEqual(key(base + "scenario:\n  - x\nimage2: z\n"), key(base + "scenario:\n  - y\n  - w\nimage2: z\n"))

    def test_empty_search_history_point(self):
        with tempfile.TemporaryDirectory() as root:
            d = Path(root) / "s"
            d.mkdir()
            (d / "search_history.json").write_text(json.dumps(
                {"boundary_summary": {"feasible_max": None, "infeasible_min": {"value": 1}}}))
            cfg = fetch.Config(rf="s", source=fetch.parse_source(root), agg=None, files=fetch.DirReader(d))
            p = extract.build_point(cfg)
            self.assertEqual((p["mode"], p["search"]["state"], p["conc"]), ("search", "no_pass_in_range", None))
            self.assertTrue(any("winning probe export missing" in n for n in p["notes"]))
            data = extract.build_data([cfg], [])
            self.assertIn("c—", data["notes"][-1])

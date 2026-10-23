import copy
import json
import unittest
from pathlib import Path

import extract

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

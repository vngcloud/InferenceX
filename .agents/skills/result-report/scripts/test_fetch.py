import tempfile
import unittest
from pathlib import Path

import fetch

FIX = Path(__file__).resolve().parent / "fixtures"


class ParseSourceTest(unittest.TestCase):
    def test_run_url(self):
        src = fetch.parse_source("https://github.com/vngcloud/InferenceX/actions/runs/39558375380/job/1")
        self.assertEqual((src.kind, src.repo, src.run_id), ("run", "vngcloud/InferenceX", "39558375380"))
        self.assertEqual(src.link, "https://github.com/vngcloud/InferenceX/actions/runs/39558375380")

    def test_bare_id_uses_default_repo(self):
        src = fetch.parse_source("123", default_repo="o/r")
        self.assertEqual((src.kind, src.repo, src.run_id), ("run", "o/r", "123"))

    def test_dir(self):
        src = fetch.parse_source(str(FIX / "gh_fixed"))
        self.assertEqual(src.kind, "dir")
        self.assertEqual(src.path, (FIX / "gh_fixed").resolve())

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            fetch.parse_source("/no/such/dir")


class LocalConfigsTest(unittest.TestCase):
    def configs(self, name):
        return fetch.local_configs(fetch.parse_source(str(FIX / name)))

    def test_gh_layout_pairs_on_rf(self):
        [cfg] = self.configs("gh_fixed")
        self.assertEqual(cfg.rf, "glm53_c108")
        self.assertEqual(cfg.agg["conc"], 108)
        self.assertIn(b"--concurrency 108", cfg.files.read("benchmark_command.txt"))
        self.assertIsNone(cfg.files.read("missing.txt"))

    def test_hand_named_layout_pairs_the_lone_aggregate_with_the_lone_root(self):
        [cfg] = self.configs("search_local")
        self.assertEqual(cfg.agg["state"], "ok")
        self.assertIsNotNone(cfg.files.read("aiperf_artifacts/search_history.json"))

    def test_raw_aiperf_dir_is_a_root_without_aggregate(self):
        [cfg] = self.configs("raw_aiperf")
        self.assertIsNone(cfg.agg)
        self.assertIsNotNone(cfg.files.read("profile_export_aiperf.json"))
        self.assertIn("ccu16", cfg.rf)

    def test_aggregate_only(self):
        [cfg] = self.configs("aggonly")
        self.assertIsNone(cfg.files)
        self.assertEqual(cfg.agg["conc"], 96)

    def test_empty_dir_gives_a_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            configs, notes = fetch.discover(fetch.parse_source(tmp), Path(tmp) / ".cache", 100)
        self.assertEqual(configs, [])
        self.assertIn("no aggregate JSON or aiperf export", notes[0])


if __name__ == "__main__":
    unittest.main()

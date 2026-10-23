import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import result_report

FIX = Path(__file__).resolve().parent / "fixtures"


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = result_report.main([str(a) for a in argv])
    return code, out.getvalue()


class CliTest(unittest.TestCase):
    def test_writes_report_data_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run(FIX / "gh_fixed", FIX / "search_local", "--out", tmp, "--html", "--title", "demo")
            self.assertEqual(code, 0, out)
            md = (Path(tmp) / "report.md").read_text()
            data = json.loads((Path(tmp) / "data.json").read_text())
            page = (Path(tmp) / "report.html").read_text()
        self.assertIn("# demo", md)
        self.assertIn("## Scenario: agentx-cc-062126", md)
        self.assertEqual(len(data["arms"]), 2)
        self.assertIn("<svg", page)
        self.assertIn("report.md", out)

    def test_findings_replace_the_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            findings = Path(tmp) / "findings.md"
            findings.write_text("- GLM c108 reaches 734 output tok/s.\n")
            code, out = run(FIX / "gh_fixed", "--out", tmp, "--findings", findings)
            md = (Path(tmp) / "report.md").read_text()
        self.assertEqual(code, 0, out)
        self.assertIn("- GLM c108 reaches 734 output tok/s.", md)
        self.assertNotIn("KEY_FINDINGS", md)

    def test_bad_input_and_empty_dir(self):
        self.assertEqual(run("/no/such/dir")[0], 2)
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            self.assertEqual(run(empty, "--out", Path(tmp) / "o")[0], 1)

    def test_unknown_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run(FIX / "gh_fixed", "--out", tmp, "--baseline", "nope")[0], 2)


class FetchedBytesTest(unittest.TestCase):
    def test_cumulative_bytes_survive_a_cached_rerun(self):
        import fetch
        real = fetch.discover

        def fake(src, cache, mb):
            fetch.FETCHED["bytes"] += 3_000_000
            return real(src, cache, mb)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(fetch, "discover", fake):
            code, out = run(FIX / "gh_fixed", "--out", tmp)
            self.assertIn("downloaded 3.00 MB this run (total 3.00 MB)", out)
            self.assertEqual(json.loads((Path(tmp) / ".cache/fetched.json").read_text()), {"bytes": 3_000_000})
            with mock.patch.object(fetch, "discover", real):
                code, out = run(FIX / "gh_fixed", "--out", tmp)
            data = json.loads((Path(tmp) / "data.json").read_text())
            md = (Path(tmp) / "report.md").read_text()
        self.assertEqual(data["bytes_fetched"], 3_000_000)
        self.assertIn("downloaded 0.00 MB this run (total 3.00 MB)", out)
        self.assertIn("downloaded 3.00 MB", md)

    def test_corrupt_fetched_json_is_treated_as_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".cache").mkdir()
            (Path(tmp) / ".cache/fetched.json").write_text("{not json")
            code, out = run(FIX / "gh_fixed", "--out", tmp)
        self.assertEqual(code, 0, out)
        self.assertIn("total 0.00 MB", out)

    def test_null_bytes_in_fetched_json_is_treated_as_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".cache").mkdir()
            (Path(tmp) / ".cache/fetched.json").write_text('{"bytes": null}')
            code, out = run(FIX / "gh_fixed", "--out", tmp)
        self.assertEqual(code, 0, out)

    def test_local_folder_creates_no_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            run(FIX / "gh_fixed", "--out", tmp)
            self.assertFalse((Path(tmp) / ".cache").exists())


if __name__ == "__main__":
    unittest.main()

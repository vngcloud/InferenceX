import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

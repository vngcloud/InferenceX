import json

import pytest

from infx.serving import specdec

TEXT = "\n".join([
    "# HELP vllm:spec_decode_num_drafts_total drafts",
    "# TYPE vllm:spec_decode_num_drafts_total counter",
    'vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 1.5e+03',
    'vllm:spec_decode_num_drafts_total{engine="1",model_name="m"} 500',
    "vllm:spec_decode_num_accepted_tokens_total 4000",
    "vllm:spec_decode_num_draft_tokens_total 12000",
    "vllm:num_requests_running 3",
    "vllm:bad nan",
])


def test_parse_sums_label_sets_and_skips_non_finite():
    totals = specdec.parse_prometheus(TEXT)
    assert totals["vllm:spec_decode_num_drafts_total"] == 2000
    assert "vllm:bad" not in totals


def test_sglang_gauge_is_averaged_over_nonzero_series():
    text = "\n".join([
        'sglang:spec_accept_length{dp_rank="0"} 3.5',
        'sglang:spec_accept_length{dp_rank="1"} 4.5',
        'sglang:spec_accept_length{dp_rank="2"} 0.0',
        'sglang:num_running_reqs{dp_rank="0"} 2',
        'sglang:num_running_reqs{dp_rank="1"} 3',
    ])
    totals = specdec.parse_prometheus(text)
    assert totals["sglang:spec_accept_length"] == pytest.approx(4.0)
    assert totals["sglang:num_running_reqs"] == 5
    assert specdec.parse_prometheus('sglang:spec_accept_length{dp_rank="0"} 0.0')["sglang:spec_accept_length"] == 0.0


def test_snapshot_averages_the_gauge_across_urls(monkeypatch):
    pages = {"a": "sglang:spec_accept_length 3.0\nvllm:spec_decode_num_drafts_total 10",
             "b": "sglang:spec_accept_length 5.0\nvllm:spec_decode_num_drafts_total 30"}

    class Response:
        def __init__(self, url):
            self.body = pages[url].encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self.body

    monkeypatch.setattr(specdec.urllib.request, "urlopen", lambda url, timeout: Response(url))
    snap = specdec.snapshot(["a", "b"])
    assert snap["sglang:spec_accept_length"] == pytest.approx(4.0)
    assert snap["vllm:spec_decode_num_drafts_total"] == 40


def test_acceptance_is_one_plus_accepted_per_draft_event():
    before = {"vllm:spec_decode_num_accepted_tokens_total": 100.0, "vllm:spec_decode_num_drafts_total": 50.0,
              "vllm:spec_decode_num_draft_tokens_total": 300.0}
    after = {"vllm:spec_decode_num_accepted_tokens_total": 1100.0, "vllm:spec_decode_num_drafts_total": 550.0,
             "vllm:spec_decode_num_draft_tokens_total": 3300.0}
    result = specdec.acceptance(before, after)
    assert result["acceptance_length"] == pytest.approx(3.0)
    assert result["acceptance_rate"] == pytest.approx(1000 / 3000)
    assert result["spec_draft_events"] == 500 and result["draft_depth"] == 6


def test_no_series_means_no_speculative_decoding():
    assert specdec.acceptance({}, {}) is None
    assert specdec.acceptance({"vllm:num_requests_running": 1.0}, {}) is None


def test_sglang_counters_give_tokens_per_verify_call():
    before = {"sglang:spec_verify_calls": 100.0, "sglang:generation_tokens": 300.0,
              "sglang:spec_accept_length": 3.0}
    after = {"sglang:spec_verify_calls": 1100.0, "sglang:generation_tokens": 3800.0,
             "sglang:spec_accept_length": 14.0}  # gauge ignored when counters exist
    result = specdec.acceptance(before, after)
    assert result["acceptance_length"] == pytest.approx(3.5)
    assert result["spec_verify_calls"] == 1000 and result["spec_generated_tokens"] == 3500


def test_sglang_without_speculation_has_generation_tokens_only():
    assert specdec.acceptance({"sglang:generation_tokens": 0.0},
                              {"sglang:generation_tokens": 5000.0, "sglang:spec_accept_length": 0.0}) is None


def test_sglang_gauge_is_read_after_the_run():
    result = specdec.acceptance({}, {"sglang:spec_accept_length": 2.4})
    assert result["acceptance_length"] == 2.4


def test_sglang_gauge_of_zero_is_no_speculative_decoding():
    assert specdec.acceptance({"sglang:spec_accept_length": 0.0}, {"sglang:spec_accept_length": 0.0}) is None


def test_summarize_and_merge(tmp_path):
    before, after, agg = tmp_path / "b.json", tmp_path / "a.json", tmp_path / "agg.json"
    assert specdec.summarize(before, after) is None
    specdec.save(before, {"vllm:spec_decode_num_drafts_total": 0.0})
    specdec.save(after, {"vllm:spec_decode_num_drafts_total": 10.0, "vllm:spec_decode_num_accepted_tokens_total": 20.0})
    agg.write_text(json.dumps({"scenario": "x"}))
    specdec.merge(agg, specdec.summarize(before, after))
    assert json.loads(agg.read_text())["spec_decode"]["acceptance_length"] == pytest.approx(3.0)
    specdec.merge(tmp_path / "missing.json", {})  # no aggregate, nothing to do

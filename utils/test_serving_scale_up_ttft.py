import json

import pytest

from serving.scale_up_ttft import percentile, report

S = 10**9
JOIN = 1000 * S
ENGINE1, ENGINE2 = "http://127.0.0.1:9200", "http://127.0.0.1:9201"


def record(corr, start_ns, ttft_ms, phase="profiling"):
    return {"metadata": {"benchmark_phase": phase, "request_start_ns": start_ns, "x_correlation_id": corr,
                         "conversation_id": f"room-{corr}", "turn_index": 0},
            "metrics": {"time_to_first_token": {"value": ttft_ms, "unit": "ms"}}}


@pytest.fixture
def results(tmp_path):
    # room a stays on engine 1; room b moves to engine 2 at the join (slow first turn there);
    # room c starts after the join on engine 2.
    requests = [  # (corr, start, ttft ms, engine)
        ("a", JOIN - 300 * S, 100, ENGINE1), ("b", JOIN - 200 * S, 200, ENGINE1),
        ("a", JOIN + 10 * S, 120, ENGINE1), ("b", JOIN + 20 * S, 900, ENGINE2),
        ("b", JOIN + 100 * S, 80, ENGINE2), ("c", JOIN + 400 * S, 300, ENGINE2),
    ]
    (tmp_path / "aiperf_artifacts").mkdir()
    (tmp_path / "aiperf_artifacts/profile_export.jsonl").write_text("".join(
        json.dumps(record(corr, start, ttft)) + "\n" for corr, start, ttft, _ in requests))
    (tmp_path / "scale_up.jsonl").write_text(json.dumps({"url": ENGINE2, "added_ns": JOIN, "response": "ok"}) + "\n")
    (tmp_path / "router.log").write_text("".join(
        f"\x1b[2mINFO\x1b[0m Consistent hash routing: key='header:x-correlation-id:{corr}' -> worker='{engine}' (index=0)\n"
        "Consistent hash routing: key='request:null' -> worker='x' (index=0)\n"
        for corr, _, _, engine in requests))
    return tmp_path


def lines(text):
    return {line[:34].strip(): line[34:].split() for line in text.splitlines() if len(line) > 34}


def test_percentile_matches_numpy_linear():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10], 95) == 10
    assert percentile([], 95) is None


def test_report_windows_engines_and_slo(results):
    text = report(results, slo_ms=500, bucket=60)
    table = lines(text)
    # reqs err p50 p95 p99 [verdict]
    assert table["before join"][:2] == ["2", "0"]
    assert table["+0..+60 s"][:2] == ["2", "0"] and table["+0..+60 s"][-2:] == ["OVER", "SLO"]
    assert table["+300..end s"][:3] == ["1", "0", "0.30"]
    assert table["stayed on the first engine(s)"][:3] == ["1", "0", "0.12"]
    assert table["moved room, first turn there"][:3] == ["1", "0", "0.90"]
    assert table["moved room, later turns"][:3] == ["1", "0", "0.08"]
    assert table["room started after the join"][:3] == ["1", "0", "0.30"]
    assert "join: http://127.0.0.1:9201 at 300 s into profiling" in text
    assert "skipped" not in text


def test_report_without_router_log_skips_engine_split(results):
    (results / "router.log").unlink()
    assert "by engine" not in report(results, slo_ms=None, bucket=60)

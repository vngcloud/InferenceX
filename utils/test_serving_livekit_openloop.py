import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "utils/serving/livekit_openloop.py"


def trace_rows():
    """Three rooms; a dial warm-up row first (no delay), then turns with logged delays."""
    rows = []
    for room in ("room-a", "room-b", "room-c"):
        msgs = [{"role": "system", "content": "shared head"}, {"role": "user", "content": f"{room} hello"}]
        rows.append({"session_id": room, "messages": msgs, "output_length": 1})
        rows.append({"session_id": room, "messages": msgs, "output_length": 20, "delay": 3000})
        rows.append({"session_id": room, "messages": msgs + [{"role": "assistant", "content": "hi"},
                                                             {"role": "user", "content": "more"}],
                     "output_length": 30, "delay": 6500})
    return rows


def generate(tmp_path, *extra, name="out"):
    src = tmp_path / "trace.jsonl"
    if not src.exists():
        src.write_text("".join(json.dumps(r) + "\n" for r in trace_rows()))
    dst, sidecar = tmp_path / f"{name}.jsonl", tmp_path / f"{name}.json"
    args = ["--ccu", "8", "--call-duration-s", "40", "--duration-s", "600", "--ramp-s", "60", "--tail-s", "60", *extra]
    out = subprocess.run([sys.executable, str(SCRIPT), str(src), str(dst), str(sidecar), *args],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return [json.loads(line) for line in dst.read_text().splitlines()], json.loads(sidecar.read_text())


def test_same_seed_same_trace_other_seed_differs(tmp_path):
    a, _ = generate(tmp_path, name="a")
    b, _ = generate(tmp_path, name="b")
    c, _ = generate(tmp_path, "--seed", "7", name="c")
    assert a == b
    assert a != c


def test_openings_get_arrival_times_later_turns_keep_their_gap(tmp_path):
    rows, side = generate(tmp_path)
    by_call: dict[str, list[dict]] = {}
    for r in rows:
        by_call.setdefault(r["session_id"], []).append(r)
    assert len(by_call) == side["arrivals"]
    horizon_ms = (side["duration_s"] - side["tail_s"]) * 1000
    for call in by_call.values():
        assert len(call) == 3
        assert "delay" not in call[0] and 0 <= call[0]["timestamp"] < horizon_ms
        assert [r["delay"] for r in call[1:]] == [3000, 6500]
        assert all("timestamp" not in r for r in call[1:])


def test_replayed_calls_are_renamed_and_their_own_context_goes_cold(tmp_path):
    rows, side = generate(tmp_path)
    assert side["max_play"] >= 2   # 8 calls / 40 s over 540 s needs ~100 arrivals from 3 rooms
    replay = [r for r in rows if r["session_id"] == "room-a~2"]
    assert len(replay) == 3
    for r in replay:
        assert r["messages"][0] == {"role": "system", "content": "shared head"}
        assert r["messages"][1]["content"] == "play2 room-a hello"
    assert all(r["messages"][1]["content"] == "room-a hello" for r in rows if r["session_id"] == "room-a")


def test_rate_and_scoring_window(tmp_path):
    rows, side = generate(tmp_path)
    assert side["calls_per_s"] == pytest.approx(8 / 40)
    # linear 60 s ramp then steady until 600 - 60: expected 0.2 * (30 + 480) = 102 arrivals
    assert 70 <= side["arrivals"] <= 135
    t0 = side["first_arrival_s"]
    assert min(r["timestamp"] for r in rows if "timestamp" in r) == round(t0 * 1000)
    assert side["score_window_s"] == pytest.approx([60 - t0, 540 - t0])


def test_duration_from_the_scenario(tmp_path):
    src = tmp_path / "trace.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in trace_rows()))
    out = subprocess.run([sys.executable, str(SCRIPT), str(src), str(tmp_path / "o.jsonl"), str(tmp_path / "o.json"),
                          "--ccu", "4", "--scenario",
                          str(REPO / "scenarios/custom-agentic-livekit111868-v1.1-warm-openloop.yaml")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    side = json.loads((tmp_path / "o.json").read_text())
    assert (side["duration_s"], side["ramp_s"], side["tail_s"], side["call_duration_s"]) == (2280, 180, 300, 127)


def test_no_scored_window_is_an_error(tmp_path):
    src = tmp_path / "trace.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in trace_rows()))
    out = subprocess.run([sys.executable, str(SCRIPT), str(src), str(tmp_path / "o.jsonl"), str(tmp_path / "o.json"),
                          "--ccu", "4", "--duration-s", "300", "--ramp-s", "180", "--tail-s", "300"],
                         capture_output=True, text=True)
    assert out.returncode != 0
    assert "no scored window" in out.stderr


def test_openloop_trace_rewrites_the_dataset_and_lifts_the_session_cap(tmp_path):
    src = tmp_path / "trace.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in trace_rows()))
    script = f'''
        set -e
        export INFMAX_CONTAINER_WORKSPACE="$PWD" AIPERF_CLI=aiperf
        export IX_CUSTOM_DATASET_FILE={src} CONC=8 PORT=8888 MODEL=m SERVING_SEARCH=0
        export SERVING_SCENARIO=custom-agentic-livekit111868-v1.1-warm-openloop IX_OPENLOOP_DIR={tmp_path}/ol
        export IX_OPENLOOP_SEED=3
        source benchmarks/benchmark_lib.sh
        export AIPERF_PYTHON={sys.executable}   # benchmark_lib.sh points it at the container venv
        openloop_trace {tmp_path}
        build_replay_cmd_yaml {tmp_path}
        printf '%s\\n%s\\n' "$IX_CUSTOM_DATASET_FILE" "$REPLAY_CMD"
    '''
    out = subprocess.run(["bash", "-c", script], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    dataset, cmd = out.stdout.splitlines()[-2:]
    assert dataset == f"{tmp_path}/ol/custom-agentic-livekit111868-v1.1-warm-openloop-c8.jsonl"
    assert Path(dataset).stat().st_size > 0
    assert " --concurrency 32 " in cmd
    side = json.loads((tmp_path / "openloop.json").read_text())
    assert (side["ccu"], side["seed"]) == (8, 3)

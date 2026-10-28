import json
import subprocess
import sys
from pathlib import Path

import pytest

from infx.bench_serving.greenspec_aiperf import (
    ALLOWLIST,
    MAX_BYTES,
    META_KEYS,
    SERVER_LOG_STUB,
    ConsoleFilter,
    guard,
    payload,
    preset,
    select,
)


def _row(qid: str, category: str, bucket: str, lite: bool, **body) -> dict:
    return {
        "question_id": qid,
        "category": category,
        "bucket": bucket,
        "lite": lite,
        "isl": 100,
        "osl": 10,
        "thinking_src": "none",
        "recorded": {"temperature": 0.6, "max_completion_tokens": 4096},
        "messages": [{"role": "user", "content": f"q {qid}"}],
        **body,
    }


ROWS = [
    _row("m/aaaa", "agentic_tools", "agentic_tools__0-4k", True, tools=[{"type": "function"}]),
    _row("m/bbbb", "agentic_tools", "agentic_tools__4k-16k", False),
    _row("m/cccc", "llm_task", "llm_task__16k+", True),
]


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    p = tmp_path / "greenspec.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    return p


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("diverse", False, None), ["m/aaaa", "m/bbbb", "m/cccc"]),
        (("diverse", True, None), ["m/aaaa", "m/cccc"]),
        (("agentic_tools", False, None), ["m/aaaa", "m/bbbb"]),
        (("agentic_tools", True, None), ["m/aaaa"]),
        (("diverse", False, "llm_task__16k+"), ["m/cccc"]),
    ],
)
def test_select(dataset: Path, args: tuple, expected: list[str]) -> None:
    assert [r["question_id"] for r in select(dataset, *args)] == expected


def test_unknown_subset(dataset: Path) -> None:
    with pytest.raises(ValueError, match="unknown GreenSpec subset"):
        select(dataset, "coding", False, None)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("diverse", False, None, False), "greenspec"),
        (("diverse", True, None, True), "greenspec_lite_recorded"),
        (("agentic_tools", True, None, False), "greenspec_agentic_tools_lite"),
        (("diverse", False, "agentic_coding__64k+", False), "greenspec_agentic_coding_64k_plus"),
        (("diverse", False, "llm_task__0-1k", False), "greenspec_llm_task_0_1k"),
        (("diverse", False, "llm_task__0-1k", True), None),  # no recorded bucket presets
    ],
)
def test_preset(args: tuple, expected: str | None) -> None:
    assert preset(*args) == expected


def test_payload_target_default() -> None:
    body = payload(ROWS[0], "served", extra={"max_tokens": 8192})
    assert list(body)[0] == "model" and body["model"] == "served"
    assert not set(META_KEYS) & set(body)
    assert body["max_tokens"] == 8192 and "temperature" not in body
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["tools"] == [{"type": "function"}]


def test_payload_recorded_cap_only_lowers() -> None:
    low = payload(ROWS[0], "m", recorded=True, extra={"max_tokens": 2048})
    assert low["max_completion_tokens"] == 2048 and "max_tokens" not in low
    high = payload(ROWS[0], "m", recorded=True, extra={"max_tokens": 8192})
    assert high["max_completion_tokens"] == 4096 and high["temperature"] == 0.6


CONSOLE = (
    "14:27:38.612 INFO     Starting AIPerf System (_single_run.py:64)\n"
    "Profiling:   0% |          | [00:00<?]\r"
    "Profiling:  40% |####      | [00:20<00:30]\r"
    "Profiling:  90% |######### | [00:50<00:05]14:27:46.424 WARNING  OSL mismatch: got 18 tokens\n"
    "14:27:46.500 ERROR    Error in SSE response: ValueError('secret prompt m/aaaa')\n"
    "Traceback (most recent call last):\n"
    "  ValueError: secret prompt m/aaaa\n"
    "\x1b[32mProfiling: 100% |##########| [00:55<00:00]\x1b[0m\r"
    "Processing Records: 40/40 |##########| 100% [00:01<00:00]\n"
    "                     NVIDIA AIPerf | Error Summary                     \n"
    "┏━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓\n"
    "┃ Code ┃        Type ┃ Message                      ┃ Count ┃\n"
    "┡━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩\n"
    "│  400 │ Bad Request │ {\"input_value\": \"secret m/b │ 1,203 │\n"
    "│      │             │ bbb\"}                        │       │\n"
    "│  N/A │         N/A │ timeout                      │     2 │\n"
    "└──────┴─────────────┴──────────────────────────────┴───────┘\n"
    "            NVIDIA AIPerf | LLM Metrics            \n"
    "│ Request Latency (ms) │ 12.00 │\n"
)


def _filtered(text: str, interval: float) -> list[str]:
    out: list[str] = []
    ticks = iter(range(0, 1000, 10))  # each bar line is 10 s after the previous one
    console = ConsoleFilter(out.append, interval, clock=lambda: next(ticks))
    for segment in text.replace("\r", "\n").split("\n"):
        console.feed(segment)
    console.close()
    return out


def test_console_withholds_error_text_and_throttles_bars() -> None:
    out = _filtered(CONSOLE, interval=15)
    assert "secret" not in "\n".join(out) and "aaaa" not in "\n".join(out)
    assert out == [
        "14:27:38.612 INFO     Starting AIPerf System (_single_run.py:64)",
        "GreenSpec progress: Profiling (0%) elapsed 00:00 eta ?",
        "GreenSpec progress: Profiling (90%) elapsed 00:50 eta 00:05",
        "14:27:46.424 WARNING  OSL mismatch: got 18 tokens",
        "14:27:46.500 ERROR    <withheld: may echo server error text>",
        "GreenSpec progress: Profiling (100%) elapsed 00:55 eta 00:00",
        "GreenSpec progress: Processing Records 40/40 (100%) elapsed 00:01 eta 00:00",
        "GreenSpec: aiperf error summary withheld (messages can echo requests): 400 x1203, N/A x2",
        "            NVIDIA AIPerf | LLM Metrics",
        "│ Request Latency (ms) │ 12.00 │",
    ]


def test_console_keeps_an_unterminated_error_table_dropped() -> None:
    cut = CONSOLE[: CONSOLE.index("│  N/A")]
    out = _filtered(cut, interval=0)
    assert "secret" not in "\n".join(out)
    assert out[-1] == "GreenSpec: aiperf error summary withheld (messages can echo requests): 400 x1203"


def test_cli_console_filters_stdin() -> None:
    cmd = [sys.executable, "-m", "infx.bench_serving.greenspec_aiperf", "console", "--interval", "0"]
    res = subprocess.run(cmd, input=CONSOLE.encode(), capture_output=True, check=True,
                         cwd=Path(__file__).resolve().parents[1])
    lines = res.stdout.decode().splitlines()
    assert "secret" not in res.stdout.decode()
    assert "GreenSpec progress: Profiling (40%) elapsed 00:20 eta 00:30" in lines
    assert lines[-1] == "│ Request Latency (ms) │ 12.00 │"


def test_guard_prunes_redacts_and_withholds_server_log(dataset: Path, tmp_path: Path) -> None:
    art = tmp_path / "results" / "greenspec_aiperf"
    (art / "logs").mkdir(parents=True)
    for name in ALLOWLIST:
        (art / name).write_text("{}")
    export = {"error_summary": [{"count": 1, "error_details": {"code": 400, "message": "x"}}]}
    (art / "profile_export_aiperf.json").write_text(json.dumps(export))
    (art / "inputs.json").write_text("prompts")
    (art / "profile_export.jsonl").write_text("{}")
    (art / "logs" / "aiperf.log").write_text("log")
    log = tmp_path / "server.log"
    log.write_text("engine output")
    keep = tmp_path / "kept"

    assert guard(dataset, art, [], [log], keep_log_dir=keep) == []
    assert sorted(p.name for p in art.rglob("*")) == sorted(ALLOWLIST)
    redacted = json.loads((art / "profile_export_aiperf.json").read_text())
    assert redacted["error_summary"][0]["error_details"] == {"code": 400, "message": "<withheld>"}
    assert log.read_text() == SERVER_LOG_STUB
    assert (keep / "server.log").read_text() == "engine output"


def test_guard_stub_survives_a_server_still_writing(dataset: Path, tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text("engine output")
    with log.open("r+") as server:  # like `> server.log`: no O_APPEND
        server.seek(0, 2)
        assert guard(dataset, tmp_path / "none", [], [log]) == []
        server.write("late engine output")
    assert log.read_text() == SERVER_LOG_STUB


def test_guard_fails_on_a_row_id_or_a_big_file(dataset: Path, tmp_path: Path) -> None:
    art = tmp_path / "greenspec_aiperf"
    art.mkdir()
    (art / "result.json").write_text(json.dumps({"note": "m/bbbb"}))
    big = tmp_path / "result_cell.json"
    big.write_text("x" * (MAX_BYTES + 1))
    problems = guard(dataset, art, [big], [])
    assert any("question_id" in p for p in problems)
    assert any("bytes" in p for p in problems)


def test_cli_prepare_writes_raw_payload_lines(dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "sel.jsonl"
    cmd = [
        sys.executable, "-m", "infx.bench_serving.greenspec_aiperf", "prepare",
        "--file", str(dataset), "--subset", "agentic_tools", "--lite", "1",
        "--out", str(out), "--model", "served", "--extra", '{"max_tokens": 1024}',
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert res.stdout.strip() == "1"
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines == [payload(ROWS[0], "served", extra={"max_tokens": 1024})]

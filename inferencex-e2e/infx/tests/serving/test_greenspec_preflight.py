import json
from pathlib import Path

import pytest

from infx.serving import client
from infx.serving import greenspec_preflight as pf


def _body(n: int, **extra) -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": "x" * n}], **extra}


def _count(body) -> int:
    text = body["messages"][0]["content"]
    if text.startswith("BAD"):
        raise pf.RejectedError("TemplateError: System message must be at the beginning.")
    return len(text)


@pytest.mark.parametrize("tokens, cap, framework, expected", [
    (980, 8192, "sglang", None),  # SGLang ignores the output cap
    (999, 8192, "sglang", pf.NEAR),
    (1000, 8192, "sglang", pf.OVER),  # SGLang needs room for one output token
    (995, 0, "sglang", pf.NEAR),  # within 1%
    (900, 50, "vllm", None),
    (900, 101, "vllm", pf.OVER),  # vLLM needs room for the whole output cap
    (900, 95, "vllm", pf.NEAR),
])
def test_classify(tokens, cap, framework, expected):
    assert pf.classify(tokens, cap, 1000, framework) == expected


def test_output_cap_takes_either_key():
    assert pf.output_cap({"max_tokens": 10}) == 10
    assert pf.output_cap({"max_completion_tokens": 7, "max_tokens": 3}) == 7
    assert pf.output_cap({}) == 0


def test_engine_messages_parse_tool_arguments_without_mutating():
    messages = [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}},
                                                     {"function": {"name": "g", "arguments": "{cut"}}]}]
    out = pf.engine_messages(messages)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    assert out[0]["tool_calls"][1]["function"]["arguments"] == "{cut"
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_check_groups_lines_by_reason():
    bodies = [_body(10), _body(1200), {"messages": [{"role": "user", "content": "BAD"}]}, _body(995)]
    report = pf.check(bodies, _count, 1000, "/v1/models", "sglang")
    assert report.rows == 4
    assert report.reasons == {pf.OVER: [1], pf.TEMPLATE: [2], pf.NEAR: [3]}
    assert report.flagged(*pf.DROPPED) == [1, 2]
    assert report.summary(dropped=2) == {
        "context": 1000, "context_source": "/v1/models", "rows": 4, "over_context": 1,
        "near_context": 1, "template_rejected": 1, "dropped": 2,
    }


def test_log_lines_carry_line_numbers_and_safe_template_text_only():
    bodies = [_body(10), _body(1200), {"messages": [{"role": "user", "content": "BAD secret"}]}]
    report = pf.check(bodies, _count, 1000, "serve command", "vllm")
    text = "\n".join(pf.log_lines(report, "warn", 0))
    assert "selection lines 1" in text and "selection lines 2" in text
    assert "System message must be at the beginning." in text
    assert "secret" not in text and "set overrides.preflight: drop" in text


def test_drop_lines_rewrites_the_selection_and_its_ids(tmp_path: Path):
    selection = tmp_path / "selection.jsonl"
    selection.write_text("".join(json.dumps(_body(i)) + "\n" for i in (1, 2, 3)))
    ids = tmp_path / "selection.ids"
    ids.write_text("a\nb\nc\n")
    assert pf.drop_lines(selection, [1], ids) == 2
    assert [len(b["messages"][0]["content"]) for b in pf.iter_bodies(selection)] == [1, 3]
    assert ids.read_text() == "a\nc\n"


def test_private_report_names_the_flagged_rows(tmp_path: Path):
    ids = tmp_path / "selection.ids"
    ids.write_text("a\nb\n")
    report = pf.check([_body(10), _body(2000)], _count, 1000, "/v1/models", "sglang")
    out = tmp_path / "private" / "report.json"
    pf.private_report(report, ids, out)
    data = json.loads(out.read_text())
    assert data["flagged"] == [{"line": 1, "question_id": "b", "reason": pf.OVER,
                                "input_tokens": 2000, "detail": None}]


def test_served_context_falls_back_to_the_serve_command():
    unreachable = "http://127.0.0.1:9"
    assert pf.served_context(unreachable, {}, "sglang serve --context-length 262144", 1) == (262144, "serve command")
    assert pf.served_context(unreachable, {}, "vllm serve --max-model-len=32768", 1) == (32768, "serve command")
    assert pf.served_context(unreachable, {}, "sglang serve", 1) == (None, "")


def test_run_preflight_drop_leaves_the_rejected_lines_out(tmp_path: Path, monkeypatch):
    selection = tmp_path / "selection.jsonl"
    selection.write_text("".join(json.dumps(_body(n)) + "\n" for n in (10, 1200, 20)))
    selection.with_suffix(".ids").write_text("a\nb\nc\n")
    monkeypatch.setattr(pf, "served_context", lambda *a, **k: (1000, "/v1/models"))
    monkeypatch.setattr(pf, "tokenizer_counter", lambda name, framework="": _count)
    env = {"MODEL": "m", "PORT": "8000", "FRAMEWORK": "sglang"}
    summary = client.run_preflight(env, selection, "drop", tmp_path)
    assert summary["over_context"] == 1 and summary["dropped"] == 1
    assert sum(1 for _ in pf.iter_bodies(selection)) == 2
    private = json.loads((tmp_path / client.PRIVATE_LOGS / "greenspec_preflight.json").read_text())
    assert private["flagged"][0]["question_id"] == "b"


def test_run_preflight_warn_keeps_every_line_and_skips_without_a_tokenizer(tmp_path: Path, monkeypatch):
    selection = tmp_path / "selection.jsonl"
    selection.write_text("".join(json.dumps(_body(n)) + "\n" for n in (10, 1200)))
    monkeypatch.setattr(pf, "served_context", lambda *a, **k: (1000, "/v1/models"))
    monkeypatch.setattr(pf, "tokenizer_counter", lambda name, framework="": _count)
    env = {"MODEL": "m", "PORT": "8000"}
    assert client.run_preflight(env, selection, "warn", tmp_path)["dropped"] == 0
    assert sum(1 for _ in pf.iter_bodies(selection)) == 2

    def broken(name, framework=""):
        raise OSError("no tokenizer")

    monkeypatch.setattr(pf, "tokenizer_counter", broken)
    assert client.run_preflight(env, selection, "warn", tmp_path) is None


def test_label_result_files_the_preflight_counts(tmp_path: Path):
    aggregate = tmp_path / "r.json"
    aggregate.write_text("{}")
    client.label_result(aggregate, "greenspec-diverse", greenspec_preflight={"dropped": 2})
    assert json.loads(aggregate.read_text())["greenspec_preflight"] == {"dropped": 2}


def test_engine_tools_match_each_engines_dump():
    body = {"tools": [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]}
    assert pf.engine_tools(body, "sglang") == [{
        "type": "function",
        "function": {"description": "d", "name": "f", "parameters": {"type": "object"}, "strict": False},
        "defer_loading": None,
    }]
    assert list(pf.engine_tools(body, "sglang")[0]["function"]) == ["description", "name", "parameters", "strict"]
    assert pf.engine_tools(body, "vllm") == [
        {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}
    ]
    assert pf.engine_tools(body, "") == body["tools"]
    assert pf.engine_tools({**body, "tool_choice": "none"}, "sglang") is None
    assert pf.engine_tools({"tools": []}, "sglang") is None
    assert pf.engine_tools({}, "vllm") is None

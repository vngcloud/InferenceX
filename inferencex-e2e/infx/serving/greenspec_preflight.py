"""GreenSpec runtime preflight: which request bodies will the served model reject, before the replay.

A GreenSpec release is cut once for every model; whether a row fits depends on the served model's
tokenizer, chat template and context. So the check runs on the client, once the engine serves and
before aiperf starts, against the very bodies the replay sends:

- the context limit comes from the server (``/v1/models``: ``max_model_len``), else from the serve
  command (``--context-length`` / ``--max-model-len``);
- each body is rendered with the model's chat template (``apply_chat_template``, messages + tools +
  the body's ``chat_template_kwargs``) and its tokens counted;
- a body is ``over`` when the engine would refuse it: SGLang needs room for one output token, vLLM
  for the whole output cap (``max_tokens`` / ``max_completion_tokens``); ``near`` when it is within
  ``NEAR_MARGIN`` of that limit (the client's count can differ from the server's by a few tokens);
  ``template`` when the chat template raises.

``warn`` lists them, ``drop`` also leaves ``over`` and ``template`` bodies out of the selection
file. The job log and the result JSON get counts and line numbers of the selection only; the
question_ids and token counts go to a node-local report (the benchmark repo is public).
"""

from __future__ import annotations

import contextlib
import copy
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OVER = "over"
NEAR = "near"
TEMPLATE = "template"
DROPPED = (OVER, TEMPLATE)
NEAR_MARGIN = 0.01
MAX_TOKEN_KEYS = ("max_tokens", "max_completion_tokens")
CONTEXT_FLAGS = re.compile(r"--(?:context-length|max-model-len)[ =](\d+)")
SHOWN = 20  # line numbers listed per reason in the job log


class RejectedError(Exception):
    """The chat template raised on a body; the message is safe to log (no row text)."""


@dataclass
class Report:
    context: int
    source: str
    framework: str
    rows: int = 0
    reasons: dict[str, list[int]] = field(default_factory=dict)  # reason -> 0-based selection lines
    tokens: dict[int, int] = field(default_factory=dict)  # line -> input tokens, flagged lines only
    details: dict[int, str] = field(default_factory=dict)  # line -> template error, safe text only

    def flagged(self, *reasons: str) -> list[int]:
        return sorted({line for r in reasons for line in self.reasons.get(r, [])})

    def summary(self, dropped: int = 0) -> dict[str, Any]:
        """Counts only: this goes into the uploaded result JSON."""
        return {
            "context": self.context,
            "context_source": self.source,
            "rows": self.rows,
            "over_context": len(self.reasons.get(OVER, [])),
            "near_context": len(self.reasons.get(NEAR, [])),
            "template_rejected": len(self.reasons.get(TEMPLATE, [])),
            "dropped": dropped,
        }


def engine_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Messages as an engine hands them to the template: tool-call ``arguments`` strings parsed."""
    out = copy.deepcopy(messages)
    for message in out:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                with contextlib.suppress(ValueError):  # not JSON: left as is, the template decides
                    function["arguments"] = json.loads(arguments)
    return out


def engine_tools(body: Mapping[str, Any], framework: str) -> list[dict[str, Any]] | None:
    """``tools`` as the engine hands them to the template, after its pydantic model dumps them.

    The defaults the dump adds are tokens too: on a row with 428 tools the raw list counts 4,407
    tokens fewer than SGLang's own count. SGLang (0.5.22: ``Tool.model_dump()``) adds
    ``function.strict: false`` and a top-level ``defer_loading: null`` and orders the function as
    description, name, parameters, strict; this matches its 400 messages to the token on Qwen3.8.
    vLLM dumps ``{type, function: {name, description, parameters}}``. Neither renders tools when
    ``tool_choice`` is ``none``.
    """
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools or body.get("tool_choice") == "none":
        return None
    out = []
    for tool in tools:
        function = (tool.get("function") if isinstance(tool, dict) else None) or {}
        if framework == "sglang":
            out.append(
                {
                    "type": tool.get("type", "function"),
                    "function": {
                        "description": function.get("description"),
                        "name": function.get("name"),
                        "parameters": function.get("parameters"),
                        "strict": bool(function.get("strict", False)),
                    },
                    "defer_loading": tool.get("defer_loading"),
                }
            )
        elif framework == "vllm":
            out.append(
                {
                    "type": tool.get("type", "function"),
                    "function": {
                        "name": function.get("name"),
                        "description": function.get("description"),
                        "parameters": function.get("parameters"),
                    },
                }
            )
        else:
            out.append(tool)
    return out


def _reason(error: Exception) -> str:
    """A template error's message (written by the template author), else only the error type:
    other errors (TypeError, KeyError, ...) can quote values from the row."""
    try:
        from jinja2 import TemplateError
    except ImportError:  # pragma: no cover - jinja2 ships with transformers' chat templates
        TemplateError = ()  # type: ignore[assignment,misc]  # noqa: N806
    if isinstance(error, TemplateError) and type(error).__name__ == "TemplateError":
        return f"TemplateError: {str(error)[:120]}"
    return type(error).__name__


def tokenizer_counter(name: str, framework: str = "") -> Callable[[Mapping[str, Any]], int]:
    """Input tokens of a body with the model's tokenizer and chat template; raises ``RejectedError``."""
    from transformers import AutoTokenizer  # in the AgentX venv (aiperf's tokenizer)

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    def count(body: Mapping[str, Any]) -> int:
        kwargs = body.get("chat_template_kwargs") or {}
        try:
            text = tokenizer.apply_chat_template(
                engine_messages(body["messages"]),
                tools=engine_tools(body, framework),
                add_generation_prompt=True,
                tokenize=False,
                **kwargs,
            )
        except Exception as error:  # noqa: BLE001 - any template failure is a rejection
            raise RejectedError(_reason(error)) from None
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


def output_cap(body: Mapping[str, Any]) -> int:
    caps = [body[k] for k in MAX_TOKEN_KEYS if isinstance(body.get(k), int)]
    return max(caps) if caps else 0


def classify(
    tokens: int, cap: int, context: int, framework: str, margin: float = NEAR_MARGIN
) -> str | None:
    """``over`` if the engine refuses the body, ``near`` within ``margin`` of that, else None."""
    needed = tokens + (cap if framework == "vllm" else 1)
    if needed > context:
        return OVER
    if needed > context * (1 - margin):
        return NEAR
    return None


def check(
    bodies: Iterable[Mapping[str, Any]],
    count: Callable[[Mapping[str, Any]], int],
    context: int,
    source: str,
    framework: str,
    margin: float = NEAR_MARGIN,
) -> Report:
    report = Report(context, source, framework)
    for line, body in enumerate(bodies):
        report.rows += 1
        try:
            tokens = count(body)
        except RejectedError as error:
            report.reasons.setdefault(TEMPLATE, []).append(line)
            report.details[line] = str(error)
            continue
        reason = classify(tokens, output_cap(body), context, framework, margin)
        if reason:
            report.reasons.setdefault(reason, []).append(line)
            report.tokens[line] = tokens
    return report


def served_context(
    base_url: str, headers: Mapping[str, str], serve_cmd: str, timeout: float = 30.0
) -> tuple[int | None, str]:
    """The served context and where it came from: the server's ``/v1/models``, else the serve command."""
    request = urllib.request.Request(f"{base_url}/v1/models", headers=dict(headers))  # noqa: S310 -- http(s) only
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            models = json.loads(response.read()).get("data") or []
        for model in models:
            value = model.get("max_model_len") or model.get("context_length")
            if isinstance(value, int) and value > 0:
                return value, "/v1/models"
    except (urllib.error.URLError, OSError, ValueError, AttributeError):
        pass
    if match := CONTEXT_FLAGS.search(serve_cmd or ""):
        return int(match.group(1)), "serve command"
    return None, ""


def iter_bodies(path: Path) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def drop_lines(selection: Path, lines: Iterable[int], ids: Path | None = None) -> int:
    """Rewrite the selection (and its ids file) without ``lines``; return how many remain."""
    gone = set(lines)
    kept = [text for i, text in enumerate(_lines(selection)) if i not in gone]
    selection.write_text("".join(kept), encoding="utf-8")
    if ids is not None and ids.exists():
        ids.write_text(
            "".join(t for i, t in enumerate(_lines(ids)) if i not in gone), encoding="utf-8"
        )
    return len(kept)


def _lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line for line in f if line.strip()]


def log_lines(report: Report, mode: str, dropped: int) -> list[str]:
    """The job-log lines: counts and selection line numbers, nothing from the rows."""
    s = report.summary(dropped)
    head = (
        f"GreenSpec preflight: {s['rows']} requests against context {s['context']} "
        f"({s['context_source']}, {report.framework or 'unknown engine'}): "
        f"{s['over_context']} over, {s['near_context']} near, "
        f"{s['template_rejected']} rejected by the chat template"
    )
    lines = [head]
    for reason, label in ((OVER, "over context"), (NEAR, "near context"), (TEMPLATE, "template")):
        flagged = report.reasons.get(reason, [])
        if flagged:
            shown = ", ".join(str(i) for i in flagged[:SHOWN])
            more = f" (+{len(flagged) - SHOWN} more)" if len(flagged) > SHOWN else ""
            lines.append(f"WARNING GreenSpec preflight {label}: selection lines {shown}{more}")
    lines.extend(
        f"WARNING GreenSpec preflight template error: {message}"
        for message in sorted(set(report.details.values()))
    )
    if mode == "drop" and dropped:
        lines.append(
            f"GreenSpec preflight: left out {dropped} requests (overrides.preflight: drop)"
        )
    elif report.flagged(*DROPPED):
        lines.append(
            "GreenSpec preflight: these requests stay in the run and will fail; "
            "set overrides.preflight: drop to leave them out"
        )
    return lines


def private_report(report: Report, ids: Path | None, out: Path) -> None:
    """question_ids, token counts and reasons of every flagged line, node-local only."""
    names = _lines(ids) if ids is not None and ids.exists() else []
    flagged = [
        {
            "line": line,
            "question_id": names[line].strip() if line < len(names) else None,
            "reason": reason,
            "input_tokens": report.tokens.get(line),
            "detail": report.details.get(line),
        }
        for reason, lines in sorted(report.reasons.items())
        for line in lines
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**report.summary(), "flagged": flagged}, indent=2))

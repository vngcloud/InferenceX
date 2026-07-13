#!/usr/bin/env python3
"""Tool-calling probe: send a real tool-enabled chat completion and verify
the server actually returns a tool_calls response, not just any 200.

Usage:
    python3 utils/smoke_tests/probes/tool_calling.py \
        --base-url http://.../sglang-vanilla \
        --endpoint /v1/chat/completions \
        --model RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8

Exits 0 and prints the response's tool_calls on success. Exits 1 (with the
raw response body on stderr for debugging) if the server errors, or replies
without invoking the tool.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from result import ProbeResult  # noqa: E402

REQUEST_TIMEOUT_S = 30

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get the weather for.",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

# Deliberately unambiguous: a well-behaved tool-calling model has no reason
# not to call get_current_weather here.
PROMPT = "What is the current weather in Hanoi? Use the available tool to find out."


def run(base_url: str, endpoint: str, model: str) -> ProbeResult:
    url = base_url.rstrip("/") + endpoint
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": TOOLS,
        # "required", not "auto": confirmed live (see
        # project_toolcalling_limitation.md) that at temperature=0.0 this
        # model deterministically never attempts a tool call under "auto" --
        # a stable decoding-time property of a model that was never
        # tool-call-instruction-tuned, not a parser bug and not fixable from
        # this probe's side. This check's job is to verify the server
        # correctly parses and returns a well-formed call once the model is
        # forced to attempt one -- not to evaluate whether the model would
        # spontaneously choose to call a tool, which "auto" would test but
        # this model will never pass regardless of server-side fixes.
        "tool_choice": "required",
        "temperature": 0.0,
        "max_tokens": 256,
    }

    resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_S)
    if resp.status_code != 200:
        return ProbeResult(
            ok=False,
            detail=f"HTTP {resp.status_code}",
            data={"response_text": resp.text},
        )

    body = resp.json()
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError) as exc:
        return ProbeResult(
            ok=False, detail=f"unexpected response shape: {exc}", data=body
        )

    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return ProbeResult(
            ok=False,
            detail="server did not invoke the tool -- got a plain content "
            "response instead of tool_calls",
            data=message,
        )

    return ProbeResult(ok=True, detail="tool invoked", data={"tool_calls": tool_calls})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.base_url, args.endpoint, args.model)

    print(json.dumps(result.data, indent=2))
    if not result.ok:
        print(f"::error::{result.detail}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

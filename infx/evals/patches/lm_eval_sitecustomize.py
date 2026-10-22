"""Runtime compatibility hooks for lm-eval 0.4.9.2."""

import json
from typing import Any

from lm_eval.models import api_models
from lm_eval.models.openai_completions import (
    LocalChatCompletion,
    OpenAIChatCompletion,
)


def _stream_result(
    content: list[str],
    reasoning_content: list[str],
    finish_reason: str | None,
    usage: dict[str, Any] | None,
    model: str | None,
) -> dict[str, Any]:
    return {
        "id": "stream-accumulated",
        "object": "chat.completion",
        "model": model or "",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(content),
                    "reasoning_content": "".join(reasoning_content),
                },
                "finish_reason": finish_reason or "stop",
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _consume_sse_data(data: str, state: dict[str, Any]) -> bool:
    if data.strip() == "[DONE]":
        return True
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return False
    if chunk.get("usage"):
        state["usage"] = chunk["usage"]
    if chunk.get("model"):
        state["model"] = chunk["model"]
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("reasoning_content"):
            state["reasoning_content"].append(delta["reasoning_content"])
        if delta.get("content"):
            state["content"].append(delta["content"])
        if choice.get("finish_reason"):
            state["finish_reason"] = choice["finish_reason"]
    return False


def _new_stream_state() -> dict[str, Any]:
    return {
        "content": [],
        "reasoning_content": [],
        "finish_reason": None,
        "usage": None,
        "model": None,
    }


def _parse_sse_stream(response: Any) -> dict[str, Any]:
    state = _new_stream_state()
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: ") and _consume_sse_data(line[6:], state):
            break
    return _stream_result(**state)


async def _parse_sse_stream_async(response: Any) -> dict[str, Any]:
    state = _new_stream_state()
    buffer = ""
    async for raw_chunk in response.content:
        buffer += raw_chunk.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line.startswith("data: ") and _consume_sse_data(line[6:], state):
                return _stream_result(**state)
    if buffer.startswith("data: "):
        _consume_sse_data(buffer[6:].rstrip("\r"), state)
    return _stream_result(**state)


_openai_create_payload = OpenAIChatCompletion._create_payload  # noqa: SLF001


def _create_streaming_payload(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _openai_create_payload(self, *args, **kwargs)
    payload["stream"] = True
    return payload


OpenAIChatCompletion._create_payload = _create_streaming_payload  # noqa: SLF001
api_models._parse_sse_stream = _parse_sse_stream  # noqa: SLF001
api_models._parse_sse_stream_async = _parse_sse_stream_async  # noqa: SLF001


def _parse_generations(outputs: Any, **kwargs: Any) -> list[Any]:  # noqa: ARG001
    results = []
    if not isinstance(outputs, list):
        outputs = [outputs]
    for output in outputs or []:
        try:
            choices = output.get("choices", [])
            parsed = ["" for _ in choices]
            for choice in choices:
                index = choice.get("index", 0)
                message = choice.get("message") or {}
                content = message.get("content")
                if content in (None, "", []):
                    content = message.get("reasoning_content") or ""
                parsed[index] = content
        except Exception:  # noqa: BLE001
            parsed = [""]
        results.extend(parsed)
    return results


LocalChatCompletion.parse_generations = staticmethod(_parse_generations)

try:
    from lm_eval.models.api_models import JsonChatStr, TemplateAPI
except ImportError:
    JsonChatStr = None
    TemplateAPI = None

if TemplateAPI is not None and JsonChatStr is not None:

    def _apply_chat_template(
        self: Any,
        chat_history: list[dict[str, Any]],
        add_generation_prompt: bool = True,
    ) -> Any:
        if self.tokenizer_backend == "huggingface" and self.tokenized_requests:
            return self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        if self.tokenizer_backend == "remote" and self.tokenized_requests:
            return chat_history
        return JsonChatStr(json.dumps(list(chat_history), ensure_ascii=False))

    TemplateAPI.apply_chat_template = _apply_chat_template

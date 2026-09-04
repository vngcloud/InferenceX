#!/usr/bin/env bash
set -euo pipefail
set -x

# CPU runner for quality-eval benchmarks. No salloc, no Docker image, no
# squashfs — this box only drives benchmark scripts against an externally-
# managed inference endpoint (QUALITY_ENDPOINT).
#
# This script sets up the Python environments and clones the external
# benchmark repos (if not already present) before dispatching to the
# per-benchmark script under benchmarks/single_node/quality/.
#
# Venvs and cloned repos live under $QUALITY_CACHE_DIR (persistent across
# CI runs on a self-hosted runner).  Job output (results, logs) lives under
# $QUALITY_WORKSPACE (= $GITHUB_WORKSPACE) and is uploaded as artifacts.
#
# Required env (set by benchmark-tmpl.yml):
#   QUALITY_BENCHMARK_NAME  e.g. gpqa, mmlu_pro, hle, livecodebench, bfcl,
#                           scicode, swebench_pro, deepswe
#   QUALITY_ENDPOINT        e.g. https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1
#   QUALITY_API_KEY         API key for the endpoint
#   QUALITY_MODEL_NAME      e.g. openai/z-ai/glm-5.2
# Optional env:
#   RUN_ID, LIMIT (from EVAL_LIMIT), HF_TOKEN, SMOKE

# --- Paths ---------------------------------------------------------------
# Job output: per-run workspace (cleaned by GitHub Actions each run)
export QUALITY_WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"

# Persistent cache: venvs + cloned repos survive across runs.
# On a self-hosted runner $HOME is stable.  Fall back to /tmp for ephemeral CI.
export QUALITY_CACHE_DIR="${QUALITY_CACHE_DIR:-${HOME:-/tmp}/.quality-eval-cache}"
mkdir -p "$QUALITY_CACHE_DIR"

export RESULT_DIR="${QUALITY_WORKSPACE}/results"

BENCH_SCRIPT="benchmarks/single_node/quality/run_${QUALITY_BENCHMARK_NAME}.sh"

if [[ ! -f "$BENCH_SCRIPT" ]]; then
    echo "ERROR: Unknown quality benchmark '${QUALITY_BENCHMARK_NAME}'" >&2
    echo "Expected script: $BENCH_SCRIPT" >&2
    exit 1
fi

# Map InferenceX's EVAL_LIMIT to the LIMIT env var that all benchmark
# scripts read for subset/smoke runs.
export LIMIT="${EVAL_LIMIT:-${LIMIT:-}}"

# Map NUM_CONCURRENT to per-benchmark concurrency env vars.
# Each benchmark script reads its own var; NUM_CONCURRENT is the unified knob.
if [[ -n "${NUM_CONCURRENT:-}" ]]; then
    export NUM_CONCURRENT="$NUM_CONCURRENT"       # gpqa, mmlu_pro, hle (lm-eval)
    export MULTIPROCESS="$NUM_CONCURRENT"           # livecodebench
    export NUM_THREADS="$NUM_CONCURRENT"            # bfcl
    export MAX_CONNECTIONS="$NUM_CONCURRENT"        # scicode
    export WORKERS="$NUM_CONCURRENT"                # swebench_pro
    export CCU="$NUM_CONCURRENT"                    # deepswe
fi

# Set RUN_ID from the experiment name if not already set.
export RUN_ID="${RUN_ID:-${EXP_NAME:-quality-eval}}"

# Ensure uv is available (GitHub Actions runners may not have it pre-installed).
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---------------------------------------------------------------------------
# Per-benchmark environment setup
# ---------------------------------------------------------------------------
# Each function ensures the venv and external repo are ready.
# Venvs/repos live under $QUALITY_CACHE_DIR and are reused across runs.
# First run: creates everything from scratch (~5-15 min depending on benchmark).
# Subsequent runs: skips setup entirely (just exports paths).

setup_lmeval() {
    local VENV="$QUALITY_CACHE_DIR/.venv-lmeval"
    if [[ ! -x "$VENV/bin/lm-eval" ]] || ! "$VENV/bin/python" -c "import tenacity, PIL" 2>/dev/null; then
        echo "=== Setting up lm-eval venv ==="
        uv venv --clear --seed "$VENV"
        uv pip install --python "$VENV/bin/python" \
            "lm-eval[api]>=0.4.5" "openai>=1.59.0" "Pillow>=10.0.0"
    fi
    # Patch openai_completions.py + api_models.py for streaming
    local OAI_COMP="$VENV/lib/python3.12/site-packages/lm_eval/models/openai_completions.py"
    if [[ -f "$OAI_COMP" ]] && ! grep -q '"stream": True' "$OAI_COMP" 2>/dev/null; then
        echo "=== Patching lm-eval openai_completions.py for streaming ==="
        python3 - "$OAI_COMP" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
# Add stream=True to chat completions payload
old = '''            "seed": seed,
            **gen_kwargs,
        }

    def parse_generations(self, outputs: dict | list[dict], **kwargs) -> list[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            try:
                tmp = [None] * len(out["choices"])
                for choices in out["choices"]:
                    content = choices["message"]["content"]'''
new = '''            "seed": seed,
            "stream": True,
            **gen_kwargs,
        }

    def parse_generations(self, outputs: dict | list[dict], **kwargs) -> list[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            try:
                tmp = [None] * len(out["choices"])
                for choices in out["choices"]:
                    content = choices["message"]["content"]'''
if old in src and '"stream": True' not in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
PY
    fi
    local API_MODELS="$VENV/lib/python3.12/site-packages/lm_eval/models/api_models.py"
    if [[ -f "$API_MODELS" ]] && ! grep -q '_parse_sse_stream' "$API_MODELS" 2>/dev/null; then
        echo "=== Patching lm-eval api_models.py for streaming ==="
        python3 - "$API_MODELS" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
# Add SSE helpers after imports
old = '''from lm_eval.api.model import TemplateLM'''
new = '''from lm_eval.api.model import TemplateLM


def _parse_sse_stream(response):
    """Parse SSE stream from requests.Response into a dict matching non-stream format."""
    import json as _json
    content = ""
    finish_reason = None
    usage = None
    model = None
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        try:
            chunk = _json.loads(data)
        except _json.JSONDecodeError:
            continue
        if "usage" in chunk and chunk["usage"]:
            usage = chunk["usage"]
        if "model" in chunk and chunk["model"]:
            model = chunk["model"]
        if "choices" in chunk and chunk["choices"]:
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                content += delta["content"]
            if chunk["choices"][0].get("finish_reason"):
                finish_reason = chunk["choices"][0]["finish_reason"]
    return {
        "id": "stream-accumulated",
        "object": "chat.completion",
        "model": model or "",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason or "stop"}],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _parse_sse_stream_async(response):
    """Parse SSE stream from aiohttp response into a dict matching non-stream format."""
    import json as _json
    content = ""
    finish_reason = None
    usage = None
    model = None
    async for raw_line in response.content:
        line = raw_line.decode("utf-8").strip()
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        try:
            chunk = _json.loads(data)
        except _json.JSONDecodeError:
            continue
        if "usage" in chunk and chunk["usage"]:
            usage = chunk["usage"]
        if "model" in chunk and chunk["model"]:
            model = chunk["model"]
        if "choices" in chunk and chunk["choices"]:
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                content += delta["content"]
            if chunk["choices"][0].get("finish_reason"):
                finish_reason = chunk["choices"][0]["finish_reason"]
    return {
        "id": "stream-accumulated",
        "object": "chat.completion",
        "model": model or "",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason or "stop"}],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }'''
if old in src and "_parse_sse_stream" not in src:
    src = src.replace(old, new, 1)
    # Patch model_call to use streaming
    old2 = '''            response = requests.post(
                self.base_url,
                json=self._create_payload(
                    self.create_message(messages),
                    generate=generate,
                    gen_kwargs=gen_kwargs,
                    seed=self._seed,
                    eos=self.eos_string,
                    **kwargs,
                ),
                headers=self.header,
                verify=self.verify_certificate,
                timeout=self.timeout,
            )
            if not response.ok:
                eval_logger.warning(
                    f"API request failed with error message: {response.text}. Retrying..."
                )
            response.raise_for_status()
            return response.json()'''
    new2 = '''            payload = self._create_payload(
                self.create_message(messages),
                generate=generate,
                gen_kwargs=gen_kwargs,
                seed=self._seed,
                eos=self.eos_string,
                **kwargs,
            )
            is_stream = payload.get("stream", False)
            response = requests.post(
                self.base_url,
                json=payload,
                headers=self.header,
                verify=self.verify_certificate,
                timeout=self.timeout,
                stream=is_stream,
            )
            if not response.ok:
                eval_logger.warning(
                    f"API request failed with error message: {response.text}. Retrying..."
                )
            response.raise_for_status()
            if is_stream:
                return _parse_sse_stream(response)
            return response.json()'''
    if old2 in src:
        src = src.replace(old2, new2, 1)
    # Patch amodel_call to use streaming
    old3 = '''                response.raise_for_status()
                outputs = await response.json()'''
    new3 = '''                response.raise_for_status()
                if payload.get("stream", False):
                    outputs = await _parse_sse_stream_async(response)
                else:
                    outputs = await response.json()'''
    if old3 in src:
        src = src.replace(old3, new3, 1)
    p.write_text(src)
PY
    fi
    export QUALITY_VENV="$VENV"
}

setup_livecodebench() {
    local LCB_DIR="$QUALITY_CACHE_DIR/LiveCodeBench"
    local VENV="$LCB_DIR/.venv-lcb"
    if [[ ! -d "$LCB_DIR" ]]; then
        echo "=== Cloning LiveCodeBench (first time) ==="
        git clone --depth 1 https://github.com/LiveCodeBench/LiveCodeBench.git "$LCB_DIR"
    fi
    # Patch lcb_runner files that import HUMAN_PROMPT/AI_PROMPT without
    # try/except fallback (anthropic>=0.42 removed them).  code_generation.py
    # already has a fallback; self_repair.py and test_output_prediction.py do not.
    for f in lcb_runner/prompts/self_repair.py lcb_runner/prompts/test_output_prediction.py; do
        if [[ -f "$LCB_DIR/$f" ]] && ! grep -q "HUMAN_PROMPT = None" "$LCB_DIR/$f" 2>/dev/null; then
            echo "=== Patching $f for anthropic>=0.42 compatibility ==="
            python3 - "$LCB_DIR/$f" <<'PY'
import pathlib, sys, re
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = "from anthropic import HUMAN_PROMPT, AI_PROMPT"
new = """try:
    from anthropic import HUMAN_PROMPT, AI_PROMPT
except ImportError:
    HUMAN_PROMPT = None
    AI_PROMPT = None"""
if old in src and "HUMAN_PROMPT = None" not in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
PY
        fi
    done
    # Patch code_generation.py: load_dataset needs config name = release_version,
    # otherwise datasets looks for 'default' config which doesn't exist in cache.
    local CG_FILE="$LCB_DIR/lcb_runner/benchmarks/code_generation.py"
    if ! grep -q 'release_version, split=' "$CG_FILE" 2>/dev/null; then
        echo "=== Patching code_generation.py load_dataset config name ==="
        python3 - "$CG_FILE" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = 'load_dataset("livecodebench/code_generation_lite", split="test", version_tag=release_version, trust_remote_code=True)'
new = 'load_dataset("livecodebench/code_generation_lite", release_version, split="test", version_tag=release_version, trust_remote_code=True)'
if old in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
PY
    fi
    # Patch main.py: add LCB_LIMIT env var support to slice benchmark
    local MAIN_FILE="$LCB_DIR/lcb_runner/runner/main.py"
    if ! grep -q "LCB_LIMIT" "$MAIN_FILE" 2>/dev/null; then
        echo "=== Patching main.py with LCB_LIMIT support ==="
        python3 - "$MAIN_FILE" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = '    if args.debug:'
new = '''    _lcb_limit = int(os.environ.get("LCB_LIMIT", "0"))
    if _lcb_limit > 0:
        print(f"LCB_LIMIT={_lcb_limit}: slicing benchmark from {len(benchmark)} to {_lcb_limit} instances")
        benchmark = benchmark[:_lcb_limit]
    if args.debug:'''
if old in src and "LCB_LIMIT" not in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
PY
    fi
    # Patch lm_styles.py to add z-ai/glm-5.2 as an OpenAIChat model
    # (LCB has a hardcoded LanguageModelStore dict; our model isn't in it)
    local LM_STYLES="$LCB_DIR/lcb_runner/lm_styles.py"
    if ! grep -q '"z-ai/glm-5.2"' "$LM_STYLES" 2>/dev/null; then
        echo "=== Patching lm_styles.py with z-ai/glm-5.2 model ==="
        python3 - "$LM_STYLES" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
entry = '''    LanguageModel(
        "z-ai/glm-5.2",
        "GLM-5.2",
        LMStyle.OpenAIChat,
        datetime(2024, 12, 1),
        "https://huggingface.co/z-ai",
    ),
'''
marker = "\n]\n\nLanguageModelStore"
idx = src.find(marker)
if idx == -1:
    raise SystemExit("marker not found")
src = src[:idx] + "\n" + entry + src[idx:]
p.write_text(src)
PY
    fi
    # Patch oai_runner.py to use streaming (avoid proxy timeouts on long generations)
    local OAI_RUNNER="$LCB_DIR/lcb_runner/runner/oai_runner.py"
    if ! grep -q 'stream=True' "$OAI_RUNNER" 2>/dev/null; then
        echo "=== Patching oai_runner.py for streaming ==="
        python3 - "$OAI_RUNNER" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = '''            response = OpenAIRunner.client.chat.completions.create(
                messages=prompt,
                **self.client_kwargs,
            )
        except ('''
new = '''            response = OpenAIRunner.client.chat.completions.create(
                messages=prompt,
                stream=True,
                **self.client_kwargs,
            )
            contents = [""] * self.client_kwargs.get("n", 1)
            for chunk in response:
                if not chunk.choices:
                    continue
                for choice in chunk.choices:
                    if choice.delta.content:
                        idx = choice.index if choice.index < len(contents) else 0
                        contents[idx] += choice.delta.content
        except ('''
if old in src and "stream=True" not in src:
    src = src.replace(old, new, 1)
    old2 = '        return [c.message.content for c in response.choices]'
    new2 = '        return contents'
    if old2 in src:
        src = src.replace(old2, new2, 1)
    p.write_text(src)
PY
    fi
    # Cache-bust: check livecodebench import works
    if [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -c "import lcb_runner" 2>/dev/null; then
        echo "=== Setting up LiveCodeBench venv (first time) ==="
        uv venv --clear --seed "$VENV"
        uv pip install --python "$VENV/bin/python" -e "$LCB_DIR"
    fi
    export QUALITY_LCB_VENV="$VENV"
    export QUALITY_LCB_DIR="$LCB_DIR"
}

setup_bfcl() {
    local BFCL_DIR="$QUALITY_CACHE_DIR/BFCL/berkeley-function-call-leaderboard"
    local VENV="$BFCL_DIR/.venv-bfcl"
    if [[ ! -d "$QUALITY_CACHE_DIR/BFCL" ]]; then
        echo "=== Cloning BFCL (first time) ==="
        git clone --depth 1 https://github.com/ShishirPatil/gorilla.git "$QUALITY_CACHE_DIR/BFCL"
    fi
    # Inject z-ai/glm-5.2 model config if not already present
    local MC_FILE="$BFCL_DIR/bfcl_eval/constants/model_config.py"
    if ! grep -q "z-ai/glm-5.2-FC" "$MC_FILE" 2>/dev/null; then
        echo "=== Patching BFCL model_config.py with z-ai/glm-5.2 ==="
        python3 - "$MC_FILE" <<'PY'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
block = '''    "z-ai/glm-5.2-FC": ModelConfig(
        model_name="z-ai/glm-5.2",
        display_name="GLM-5.2 (FC, OpenAI-compatible)",
        url="https://tokenplan.api.greennode.ai",
        org="z-ai",
        license="Proprietary",
        model_handler=OpenAICompletionsHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "z-ai/glm-5.2-PROMPT": ModelConfig(
        model_name="z-ai/glm-5.2",
        display_name="GLM-5.2 (Prompt, OpenAI-compatible)",
        url="https://tokenplan.api.greennode.ai",
        org="z-ai",
        license="Proprietary",
        model_handler=OpenAICompletionsHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
'''
marker = 'api_inference_model_map = {'
idx = src.find(marker)
if idx == -1:
    print("ERROR: could not find api_inference_model_map marker", file=sys.stderr)
    sys.exit(1)
insert_at = src.find('{', idx) + 1
p.write_text(src[:insert_at] + '\n' + block + src[insert_at:])
PY
    fi
    # Patch openai_completion.py to use streaming (avoid proxy timeouts)
    local OAI_COMP="$BFCL_DIR/bfcl_eval/model_handler/api_inference/openai_completion.py"
    if ! grep -q 'stream.*True' "$OAI_COMP" 2>/dev/null; then
        echo "=== Patching BFCL openai_completion.py for streaming ==="
        python3 - "$OAI_COMP" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = '''    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time'''
new = '''    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        kwargs["stream"] = True
        api_response = self.client.chat.completions.create(**kwargs)
        accumulated = self._accumulate_stream(api_response)
        end_time = time.time()

        return accumulated, end_time - start_time

    @staticmethod
    def _accumulate_stream(stream):
        content = ""
        reasoning_content = ""
        tool_calls = {}
        finish_reason = None
        usage = None
        for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls[idx]["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls[idx]["function"]["arguments"] += tc.function.arguments
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        from types import SimpleNamespace
        msg = SimpleNamespace(content=content if content else None, reasoning_content=reasoning_content if reasoning_content else None, tool_calls=None)
        if tool_calls:
            msg.tool_calls = [
                SimpleNamespace(id=tc["id"], type=tc["type"], function=SimpleNamespace(name=tc["function"]["name"], arguments=tc["function"]["arguments"]))
                for tc in tool_calls.values()
            ]
        choice = SimpleNamespace(index=0, message=msg, finish_reason=finish_reason)
        resp = SimpleNamespace(choices=[choice], usage=usage)
        return resp'''
if old in src and "stream" not in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
PY
    fi
    if [[ ! -x "$VENV/bin/bfcl" ]] || ! "$VENV/bin/python" -c "import soundfile" 2>/dev/null; then
        echo "=== Setting up BFCL venv (first time) ==="
        uv venv --clear --seed "$VENV"
        uv pip install --python "$VENV/bin/python" -e "$BFCL_DIR" "soundfile>=0.12.0"
    fi
    export QUALITY_BFCL_VENV="$VENV"
    export QUALITY_BFCL_DIR="$BFCL_DIR"
}

setup_scicode() {
    local SCICODE_DIR="$QUALITY_CACHE_DIR/SciCode"
    local VENV="$QUALITY_CACHE_DIR/.venv-scicode"
    if [[ ! -d "$SCICODE_DIR" ]]; then
        echo "=== Cloning SciCode (first time) ==="
        git clone --depth 1 https://github.com/scicode-bench/SciCode.git "$SCICODE_DIR"
    fi
    # Cache-bust: check scicode + inspect_ai import works
    if [[ ! -x "$VENV/bin/inspect" ]] || ! "$VENV/bin/python" -c "import scicode; import inspect_ai" 2>/dev/null; then
        echo "=== Setting up SciCode venv (first time) ==="
        uv venv --clear --seed "$VENV"
        # SciCode pyproject pins unpinned "datasets" → resolver picks 2.14.4,
        # but inspect-ai requires datasets>=2.16.  datasets 2.16.1 has a bug
        # with SciCode1/SciCode dataset (TypeError in generate_from_dict).
        # Pin datasets==5.0.1 + pyarrow==25.0.1 (known good, same as LCB/swebench).
        uv pip install --python "$VENV/bin/python" \
            "datasets==5.0.1" "pyarrow==25.0.1" "openai>=3.1" "anthropic" "config" \
            "litellm" "inspect-ai" "rich" "pytest" "pytest-cov" \
            "matplotlib" "scipy" "sympy" "h5py" "jsonlines" \
            "google-generativeai"
        uv pip install --python "$VENV/bin/python" --no-deps -e "$SCICODE_DIR"
    fi
    # Patch inspect_ai OpenAI provider for streaming (avoid proxy timeouts)
    local OAI_PROVIDER="$VENV/lib/python3.12/site-packages/inspect_ai/model/_providers/openai.py"
    if [[ -f "$OAI_PROVIDER" ]] && ! grep -q 'stream.*True' "$OAI_PROVIDER" 2>/dev/null; then
        echo "=== Patching inspect_ai openai.py for streaming ==="
        python3 - "$OAI_PROVIDER" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = '''            # generate completion
            completion: ChatCompletion = await self.client.chat.completions.create(
                **request
            )

            # save response for model_call
            response = completion.model_dump()

            # parse out choices
            choices = self._chat_choices_from_response(completion, tools)

            # return output and call
            return ModelOutput(
                model=completion.model,
                choices=choices,
                usage=(
                    ModelUsage(
                        input_tokens=completion.usage.prompt_tokens,
                        output_tokens=completion.usage.completion_tokens,
                        total_tokens=completion.usage.total_tokens,
                    )
                    if completion.usage
                    else None
                ),
            ), model_call()'''
new = '''            # generate completion (streaming to avoid proxy timeouts on long generations)
            request["stream"] = True
            stream = await self.client.chat.completions.create(**request)

            content = ""
            tool_calls_map = {}
            finish_reason = None
            model_name = self.model_name
            usage = None
            async for chunk in stream:
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage = chunk.usage
                if hasattr(chunk, "model") and chunk.model:
                    model_name = chunk.model
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    content += delta.content
                if delta and hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                        if tc.id:
                            tool_calls_map[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_map[idx]["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_map[idx]["function"]["arguments"] += tc.function.arguments
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall
            from openai.types.chat.chat_completion import Choice

            msg = ChatCompletionMessage(content=content if content else None)
            if tool_calls_map:
                msg.tool_calls = [
                    ChatCompletionMessageToolCall(id=tc["id"], type=tc["type"], function=tc["function"])
                    for tc in tool_calls_map.values()
                ]

            choice = Choice(index=0, message=msg, finish_reason=finish_reason or "stop")
            completion = ChatCompletion(
                id="stream-accumulated",
                model=model_name,
                choices=[choice],
                created=0,
                object="chat.completion",
                usage=usage,
            )

            # save response for model_call
            response = completion.model_dump()

            # parse out choices
            choices = self._chat_choices_from_response(completion, tools)

            # return output and call
            return ModelOutput(
                model=completion.model,
                choices=choices,
                usage=(
                    ModelUsage(
                        input_tokens=completion.usage.prompt_tokens,
                        output_tokens=completion.usage.completion_tokens,
                        total_tokens=completion.usage.total_tokens,
                    )
                    if completion.usage
                    else None
                ),
            ), model_call()'''
if old in src and "stream" not in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
PY
    fi
    export QUALITY_SCICODE_VENV="$VENV"
    export QUALITY_SCICODE_DIR="$SCICODE_DIR"
}

setup_swebench_pro() {
    local SWEBENCH_DIR="$QUALITY_CACHE_DIR/SWE-bench_Pro-os"
    local VENV="$QUALITY_CACHE_DIR/.venv-swebenchpro"
    if [[ ! -d "$SWEBENCH_DIR" ]]; then
        echo "=== Cloning SWE-bench Pro (first time, with submodules) ==="
        git clone --recurse-submodules --depth 1 https://github.com/scaleapi/SWE-bench_Pro-os.git "$SWEBENCH_DIR"
    fi
    # Ensure SWE-agent submodule is present (cache may have shallow clone without it)
    if [[ ! -d "$SWEBENCH_DIR/SWE-agent/.git" ]]; then
        echo "=== Initializing SWE-agent submodule ==="
        git -C "$SWEBENCH_DIR" submodule update --init --recursive
    fi
    # Generate instances.yaml if missing (required by run_swebench_pro.py)
    local INSTANCES_YAML="$SWEBENCH_DIR/SWE-agent/data/instances.yaml"
    if [[ ! -f "$INSTANCES_YAML" ]]; then
        echo "=== Generating instances.yaml from HuggingFace dataset ==="
        if [[ ! -x "$VENV/bin/python" ]]; then
            uv venv --clear --seed "$VENV"
            uv pip install --python "$VENV/bin/python" \
                -r "$SWEBENCH_DIR/requirements.txt" \
                "mini-swe-agent" "litellm" "rich" "pyyaml" "datasets" "tqdm"
        fi
        "$VENV/bin/python" "$SWEBENCH_DIR/helper_code/generate_sweagent_instances.py" \
            --dockerhub_username "${DOCKERHUB_USERNAME:-jefzda}" \
            --output_path "$INSTANCES_YAML"
    fi
    if [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -c "import yaml" 2>/dev/null; then
        echo "=== Setting up SWE-bench Pro venv (first time) ==="
        uv venv --clear --seed "$VENV"
        uv pip install --python "$VENV/bin/python" \
            -r "$SWEBENCH_DIR/requirements.txt" \
            "mini-swe-agent" "litellm" "rich" "pyyaml"
    fi
    # Patch litellm_model.py for streaming (shared by SWE-bench Pro + DeepSWE)
    local LITELLM_MODEL="$VENV/lib/python3.12/site-packages/minisweagent/models/litellm_model.py"
    if [[ -f "$LITELLM_MODEL" ]] && ! grep -q 'stream.*True' "$LITELLM_MODEL" 2>/dev/null; then
        echo "=== Patching litellm_model.py for streaming ==="
        python3 - "$LITELLM_MODEL" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
# Add stream=True to _query
old1 = '                tools=[BASH_TOOL],\n                **(self.config.model_kwargs | kwargs),'
new1 = '                tools=[BASH_TOOL],\n                stream=True,\n                **(self.config.model_kwargs | kwargs),'
if old1 in src:
    src = src.replace(old1, new1, 1)
# Add accumulator function after imports
old2 = 'from minisweagent.exceptions import FormatError'
new2 = '''

def _accumulate_litellm_stream(stream):
    """Accumulate a litellm streaming response into a single ModelResponse."""
    content = ""
    tool_calls_map = {}
    finish_reason = None
    usage = None
    model = None
    for chunk in stream:
        if hasattr(chunk, "usage") and chunk.usage is not None:
            usage = chunk.usage
        if hasattr(chunk, "model") and chunk.model:
            model = chunk.model
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            content += delta.content
        if delta and getattr(delta, "tool_calls", None):
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                if tc.id:
                    tool_calls_map[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls_map[idx]["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_calls_map[idx]["function"]["arguments"] += tc.function.arguments
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
    from types import SimpleNamespace
    msg = SimpleNamespace(content=content if content else None, tool_calls=None)
    if tool_calls_map:
        msg.tool_calls = [
            SimpleNamespace(id=tc["id"], type=tc["type"], function=SimpleNamespace(name=tc["function"]["name"], arguments=tc["function"]["arguments"]))
            for tc in tool_calls_map.values()
        ]
    msg.model_dump = lambda mode=None: {"content": msg.content, "tool_calls": None}
    choice = SimpleNamespace(index=0, message=msg, finish_reason=finish_reason or "stop")
    resp = SimpleNamespace(choices=[choice], usage=usage, model=model or "")
    resp.model_dump = lambda mode=None: {"choices": [{"message": msg.model_dump(), "finish_reason": choice.finish_reason}], "usage": None, "model": resp.model}
    return resp


from minisweagent.exceptions import FormatError'''
if old2 in src and "_accumulate_litellm_stream" not in src:
    src = src.replace(old2, new2, 1)
# Add accumulation call in query()
old3 = '                response = self._query(self._prepare_messages_for_api(messages), **kwargs)\n        cost_output = self._calculate_cost(response)'
new3 = '                response = self._query(self._prepare_messages_for_api(messages), **kwargs)\n        response = _accumulate_litellm_stream(response)\n        cost_output = self._calculate_cost(response)'
if old3 in src and "_accumulate_litellm_stream(response)" not in src:
    src = src.replace(old3, new3, 1)
p.write_text(src)
PY
    fi
    # Also patch litellm_textbased_model.py
    local LITELLM_TEXT="$VENV/lib/python3.12/site-packages/minisweagent/models/litellm_textbased_model.py"
    if [[ -f "$LITELLM_TEXT" ]] && ! grep -q 'stream.*True' "$LITELLM_TEXT" 2>/dev/null; then
        echo "=== Patching litellm_textbased_model.py for streaming ==="
        python3 - "$LITELLM_TEXT" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = 'model=self.config.model_name, messages=messages, **(self.config.model_kwargs | kwargs)'
new = 'model=self.config.model_name, messages=messages, stream=True, **(self.config.model_kwargs | kwargs)'
if old in src and "stream" not in src:
    src = src.replace(old, new, 1)
# Import accumulator
old2 = 'from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig'
new2 = 'from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig, _accumulate_litellm_stream'
if old2 in src and "_accumulate_litellm_stream" not in src:
    src = src.replace(old2, new2, 1)
p.write_text(src)
PY
    fi
    export QUALITY_SWEBENCHPRO_VENV="$VENV"
    export QUALITY_SWEBENCH_DIR="$SWEBENCH_DIR"
}

setup_deepswe() {
    local DEEPSWE_DIR="$QUALITY_CACHE_DIR/deep-swe"
    if [[ ! -d "$DEEPSWE_DIR" ]]; then
        echo "=== Cloning DeepSWE (first time) ==="
        git clone --depth 1 https://github.com/datacurve-ai/deep-swe.git "$DEEPSWE_DIR"
    fi
    export QUALITY_DEEPSWE_DIR="$DEEPSWE_DIR"
}

# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------
# After the benchmark script runs, copy result files from the per-benchmark
# output directory to the workspace root so that benchmark-tmpl.yml's
# upload-artifact step (which globs for results*.json, *.traj*, etc. at
# workspace root) and validate_scores.py can find them.
#
# Also creates meta_env.json with the model prefix for threshold validation.

collect_results() {
    local BENCH="$1"
    local OUT_BASE="$QUALITY_WORKSPACE/jobs/$RUN_ID/$BENCH"
    local DEST="$QUALITY_WORKSPACE"

    echo "=== Collecting results from $OUT_BASE ==="

    # Create meta_env.json for validate_scores.py
    local MODEL_PREFIX="${MODEL_PREFIX:-${EXP_NAME%%_*}}"
    cat > "$DEST/meta_env.json" <<EOF
{"infmax_model_prefix": "${MODEL_PREFIX}", "benchmark": "${BENCH}", "run_id": "${RUN_ID}"}
EOF

    # Copy result files to workspace root (flatten, don't preserve dir structure)
    # Patterns cover all 8 benchmarks' output formats.
    local COPIED=0

    # results*.json — lm-eval (GPQA, MMLU-Pro, HLE) + general
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -name 'results*.json' -print0 2>/dev/null || true)

    # result.json (singular) — DeepSWE / pier output; copy as results.json
    # so benchmark-tmpl.yml's `ls results*.json` check passes.
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/results.json"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -maxdepth 2 -type f -name 'result.json' ! -path '*/ipython-session-bundle-*' -print0 2>/dev/null || true)

    # sample*.jsonl — lm-eval logged samples
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -name 'sample*.jsonl' -print0 2>/dev/null || true)

    # eval_results*.json — SWE-bench Pro, SciCode
    # Also copy as results.json so benchmark-tmpl.yml's `ls results*.json` check passes.
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        if [[ ! -f "$DEST/results.json" ]]; then
            cp -f "$f" "$DEST/results.json"
        fi
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -name 'eval_results*.json' -print0 2>/dev/null || true)

    # inspect-ai log files (SciCode) — .json in logs/ subdir (with --log-format json)
    # Copy first .json log as results.json so benchmark-tmpl.yml's glob matches.
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        if [[ ! -f "$DEST/results.json" ]]; then
            cp -f "$f" "$DEST/results.json"
        fi
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -path '*/logs/*' -name '*.json' -print0 2>/dev/null || true)

    # predictions.jsonl, agent_preds.json — SWE-bench, agentic
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f \( -name 'predictions.jsonl' -o -name 'agent_preds.json' \) -print0 2>/dev/null || true)

    # swebench_report_*.json
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -name 'swebench_report_*.json' -print0 2>/dev/null || true)

    # *.traj* — DeepSWE, SWE-bench trajectories
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -name '*.traj*' -print0 2>/dev/null || true)

    # BFCL score CSVs
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -name '*.csv' -print0 2>/dev/null || true)

    # LiveCodeBench result JSONs/JSONLs
    # LCB writes to output/<model>/<scenario>_<n>_<temp>.json and _eval.json
    # Copy first .json as results.json so benchmark-tmpl.yml's glob matches.
    local LCB_FIRST_JSON=""
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f \( -name '*.jsonl' -o -name 'lcb_results*.json' \) -print0 2>/dev/null || true)
    # Also pick up LCB's output/*.json files
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        if [[ -z "$LCB_FIRST_JSON" ]]; then
            LCB_FIRST_JSON="$f"
        fi
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -name '*.json' ! -name 'results*.json' ! -name 'eval_results*.json' -print0 2>/dev/null || true)
    if [[ -n "$LCB_FIRST_JSON" && ! -f "$DEST/results.json" ]]; then
        cp -f "$LCB_FIRST_JSON" "$DEST/results.json"
    fi

    echo "  Copied $COPIED result file(s) to $DEST"
    if [[ "$COPIED" -eq 0 ]]; then
        echo "  WARNING: no result files found in $OUT_BASE" >&2
        # List what IS there for debugging
        find "$OUT_BASE" -type f 2>/dev/null | head -20 || echo "  (directory empty or missing)"
    fi
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
echo "=== Quality-eval setup: ${QUALITY_BENCHMARK_NAME} ==="
echo "  Workspace (output) : $QUALITY_WORKSPACE"
echo "  Cache (venv/repos) : $QUALITY_CACHE_DIR"
echo "  Run ID             : $RUN_ID"
echo

case "${QUALITY_BENCHMARK_NAME}" in
    gpqa|mmlu_pro|hle)
        setup_lmeval
        ;;
    livecodebench)
        setup_livecodebench
        ;;
    bfcl)
        setup_bfcl
        ;;
    scicode)
        setup_scicode
        ;;
    swebench_pro)
        setup_swebench_pro
        ;;
    deepswe)
        setup_deepswe
        ;;
    *)
        echo "ERROR: Unknown quality benchmark '${QUALITY_BENCHMARK_NAME}'" >&2
        exit 1
        ;;
esac

echo "=== Dispatching to $BENCH_SCRIPT ==="
bash "$BENCH_SCRIPT"

echo "=== Collecting results for artifact upload ==="
collect_results "${QUALITY_BENCHMARK_NAME}"

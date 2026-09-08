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

# Benchmark-specific output ceilings. MAX_GEN_TOKENS is the unified CI/direct
# override; MAX_TOKENS is the spelling used by LiveCodeBench and SciCode.
if [[ -z "${MAX_GEN_TOKENS:-}" ]]; then
    case "$QUALITY_BENCHMARK_NAME" in
        gpqa|mmlu_pro|bfcl) MAX_GEN_TOKENS=8192 ;;
        hle|livecodebench|scicode) MAX_GEN_TOKENS=16384 ;;
        swebench_pro|deepswe) MAX_GEN_TOKENS=32768 ;;
    esac
fi
export MAX_GEN_TOKENS
export MAX_TOKENS="${MAX_TOKENS:-$MAX_GEN_TOKENS}"

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
    # Install the repository-owned runtime hooks into the persistent venv on
    # every run. This repairs already-cached environments and keeps reasoning
    # delta assembly independent of the installed lm-eval source layout.
    local SITE_PACKAGES
    SITE_PACKAGES="$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
    cp "$QUALITY_WORKSPACE/utils/evals/patches/lm_eval_sitecustomize.py" \
        "$SITE_PACKAGES/sitecustomize.py"
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
    # Cache generations per endpoint. LiveCodeBench's upstream cache key only
    # contains model/scenario/sampling settings, so using the same model name
    # against a different API endpoint can otherwise reuse unrelated answers.
    # Keeping the endpoint namespace still allows interrupted runs to resume.
    local PATH_UTILS="$LCB_DIR/lcb_runner/utils/path_utils.py"
    if [[ -f "$PATH_UTILS" ]] && ! grep -q 'LCB_CACHE_NAMESPACE' "$PATH_UTILS" 2>/dev/null; then
        echo "=== Patching path_utils.py with endpoint-scoped generation cache ==="
        python3 - "$PATH_UTILS" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = '    path = f"cache/{model_repr}/{scenario}_{n}_{temperature}.json"'
new = '''    cache_namespace = os.environ.get("LCB_CACHE_NAMESPACE", "")
    if cache_namespace:
        import hashlib
        namespace_hash = hashlib.sha256(cache_namespace.encode("utf-8")).hexdigest()[:16]
        path = f"cache/endpoints/{namespace_hash}/{model_repr}/{scenario}_{n}_{temperature}.json"
    else:
        path = f"cache/{model_repr}/{scenario}_{n}_{temperature}.json"'''
if old not in src:
    raise SystemExit(f"expected LiveCodeBench cache path not found in {p}")
if "import os" not in src:
    src = "import os\n" + src
p.write_text(src.replace(old, new, 1))
PY
    fi
    # Patch path_utils.py: use LCB_OUTPUT_DIR env var as base for output path
    # so results land in $OUT_BASE (where collect_results looks), not $LCB_DIR/output/
    if ! grep -q 'LCB_OUTPUT_DIR' "$PATH_UTILS" 2>/dev/null; then
        echo "=== Patching path_utils.py with LCB_OUTPUT_DIR support ==="
        python3 - "$PATH_UTILS" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = '    path = f"output/{model_repr}/{scenario}_{n}_{temperature}{cot_suffix}.json"'
new = '''    _base = os.environ.get("LCB_OUTPUT_DIR", "")
    path = f"{_base}output/{model_repr}/{scenario}_{n}_{temperature}{cot_suffix}.json" if _base else f"output/{model_repr}/{scenario}_{n}_{temperature}{cot_suffix}.json"'''
if old in src and "LCB_OUTPUT_DIR" not in src:
    src = src.replace(old, new, 1)
    if "import os" not in src:
        src = "import os\n" + src
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
    # BFCL writes CSV rows without quoting fields. Commas in a display name
    # shift every subsequent column and corrupt the score metadata.
    if grep -q 'GLM-5.2 (FC, OpenAI-compatible)' "$MC_FILE" 2>/dev/null; then
        echo "=== Repairing BFCL CSV-safe model display names ==="
        python3 - "$MC_FILE" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
src = src.replace("GLM-5.2 (FC, OpenAI-compatible)", "GLM-5.2 (FC; OpenAI-compatible)")
src = src.replace("GLM-5.2 (Prompt, OpenAI-compatible)", "GLM-5.2 (Prompt; OpenAI-compatible)")
p.write_text(src)
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
    # A streamed Chat Completions response has no usage object unless usage is
    # requested, and some compatible endpoints omit it even when requested.
    # Token accounting must never invalidate an otherwise valid tool call.
    if [[ -f "$OAI_COMP" ]] && ! grep -q 'BFCL_OPTIONAL_STREAM_USAGE' "$OAI_COMP" 2>/dev/null; then
        echo "=== Patching BFCL optional streamed usage handling ==="
        python3 - "$OAI_COMP" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
create_anchor = '''        kwargs["stream"] = True
        api_response = self.client.chat.completions.create(**kwargs)'''
create_replacement = '''        kwargs["stream"] = True
        # BFCL_OPTIONAL_STREAM_USAGE: OpenAI emits usage on the final empty
        # stream chunk only when include_usage is requested.
        kwargs.setdefault("stream_options", {"include_usage": True})
        api_response = self.client.chat.completions.create(**kwargs)'''
if create_anchor not in src:
    raise SystemExit(f"expected BFCL streaming request not found in {p}")
src = src.replace(create_anchor, create_replacement, 1)
usage_anchor = '''            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,'''
usage_replacement = '''            "input_token": getattr(api_response.usage, "prompt_tokens", 0),
            "output_token": getattr(api_response.usage, "completion_tokens", 0),'''
if usage_anchor not in src:
    raise SystemExit(f"expected BFCL usage parser not found in {p}")
p.write_text(src.replace(usage_anchor, usage_replacement, 1))
PY
    fi
    # BFCL has no CLI output-token option. Apply the repository-owned default
    # at its final OpenAI request boundary while keeping it env-overridable.
    if [[ -f "$OAI_COMP" ]] && ! grep -q 'BFCL_MAX_GEN_TOKENS' "$OAI_COMP" 2>/dev/null; then
        echo "=== Patching BFCL output-token ceiling ==="
        python3 - "$OAI_COMP" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
anchor = '        kwargs["stream"] = True\n'
replacement = '''        # BFCL_MAX_GEN_TOKENS: bounded, benchmark-specific generation budget.
        kwargs.setdefault("max_tokens", int(__import__("os").environ.get("MAX_GEN_TOKENS", "8192")))
        kwargs["stream"] = True
'''
if anchor not in src:
    raise SystemExit(f"expected BFCL streaming request not found in {p}")
p.write_text(src.replace(anchor, replacement, 1))
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
            "google-generativeai" "gdown>=5.2,<6"
        uv pip install --python "$VENV/bin/python" --no-deps -e "$SCICODE_DIR"
    fi

    # SciCode keeps its numeric reference outputs outside the git repository.
    # Cache them beside the persistent checkout so later CI runs can reuse the
    # 1 GiB file instead of downloading it again. Validate before reuse and
    # before atomically installing a new download; otherwise a missing or
    # interrupted download is silently converted by SciCode into zero scores.
    if ! "$VENV/bin/python" -c "import gdown" 2>/dev/null; then
        echo "=== Installing SciCode data downloader ==="
        uv pip install --python "$VENV/bin/python" "gdown>=5.2,<6"
    fi
    local SCICODE_DATA_DIR="$SCICODE_DIR/eval/data"
    local SCICODE_DATA_FILE="$SCICODE_DATA_DIR/test_data.h5"
    local SCICODE_DATA_URL="${SCICODE_DATA_URL:-https://drive.google.com/uc?id=17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW}"
    local SCICODE_DATA_SHA256="${SCICODE_DATA_SHA256:-48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890}"
    mkdir -p "$SCICODE_DATA_DIR"

    validate_scicode_data() {
        "$VENV/bin/python" - "$1" "$SCICODE_DATA_SHA256" <<'PY'
import hashlib
import pathlib
import sys

import h5py

path = pathlib.Path(sys.argv[1])
expected_sha256 = sys.argv[2]
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)
try:
    with h5py.File(path, "r") as data:
        if len(data) == 0:
            raise ValueError("HDF5 file contains no reference-data groups")
except (OSError, ValueError):
    raise SystemExit(1)
digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != expected_sha256:
    raise SystemExit(1)
PY
    }

    if validate_scicode_data "$SCICODE_DATA_FILE"; then
        echo "=== Reusing cached SciCode numeric test data ==="
        echo "  Data file: $SCICODE_DATA_FILE"
    else
        echo "=== Downloading SciCode numeric test data (first time) ==="
        local SCICODE_DATA_TMP
        SCICODE_DATA_TMP="$(mktemp "$SCICODE_DATA_DIR/.test_data.h5.part.XXXXXX")"
        if ! "$VENV/bin/python" -m gdown --no-cookies \
            "$SCICODE_DATA_URL" -O "$SCICODE_DATA_TMP"; then
            rm -f "$SCICODE_DATA_TMP"
            echo "ERROR: Failed to download SciCode numeric test data" >&2
            exit 1
        fi
        if ! validate_scicode_data "$SCICODE_DATA_TMP"; then
            rm -f "$SCICODE_DATA_TMP"
            echo "ERROR: Downloaded SciCode numeric test data is not a valid, non-empty HDF5 file" >&2
            exit 1
        fi
        mv -f "$SCICODE_DATA_TMP" "$SCICODE_DATA_FILE"
        echo "  Cached data file: $SCICODE_DATA_FILE"
    fi
    export QUALITY_SCICODE_DATA_FILE="$SCICODE_DATA_FILE"

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

            msg = ChatCompletionMessage(
                role="assistant",
                content=content if content else None,
            )
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
    # Repair environments cached with the original streaming patch. OpenAI SDK
    # 2.x requires ChatCompletionMessage.role; without it response assembly
    # raises after the full stream has already completed.
    if [[ -f "$OAI_PROVIDER" ]] && grep -q 'ChatCompletionMessage(content=content if content else None)' "$OAI_PROVIDER" 2>/dev/null; then
        echo "=== Repairing inspect_ai streaming response role ==="
        python3 - "$OAI_PROVIDER" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = 'msg = ChatCompletionMessage(content=content if content else None)'
new = 'msg = ChatCompletionMessage(role="assistant", content=content if content else None)'
if old not in src:
    raise SystemExit(f"expected streaming response constructor not found in {p}")
p.write_text(src.replace(old, new, 1))
PY
    fi

    # SciCode otherwise hides all generation exceptions and silently writes a
    # dummy response, which can make a zero-score smoke run appear healthy.
    local SCICODE_TASK="$SCICODE_DIR/eval/inspect_ai/scicode.py"
    if [[ -f "$SCICODE_TASK" ]] && grep -q '^                except:$' "$SCICODE_TASK" 2>/dev/null; then
        echo "=== Patching SciCode to report generation exceptions ==="
        python3 - "$SCICODE_TASK" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = '''                except:
                    print(f"Failed to generate response for problem {prob_id} step {idx+1}.")'''
new = '''                except Exception as exc:
                    print(
                        f"Failed to generate response for problem {prob_id} step {idx+1}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )'''
if old not in src:
    raise SystemExit(f"expected SciCode exception handler not found in {p}")
p.write_text(src.replace(old, new, 1))
PY
    fi
    # SciCode's upstream extractor assumes a lowercase Python fence. Some
    # models return valid Python without a fence or surround it with reasoning;
    # upstream then writes that prose into the .py file and poisons later steps.
    local SCICODE_MODELS="$SCICODE_DIR/src/scicode/gen/models.py"
    if [[ -f "$SCICODE_MODELS" ]] && ! grep -q 'SCICODE_ROBUST_CODE_EXTRACTION' "$SCICODE_MODELS" 2>/dev/null; then
        echo "=== Patching SciCode robust Python response extraction ==="
        python3 - "$SCICODE_MODELS" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = '''def extract_python_script(response: str):
    # We will extract the python script from the response
    if '```' in response:
        python_script = response.split("```python")[1].split("```")[0] if '```python' in response else response.split('```')[1].split('```')[0]
    else:
        print("Fail to extract python code from specific format.")
        python_script = response
    python_script = re.sub(r'^\\s*(import .*|from .*\\s+import\\s+.*)', '', python_script, flags=re.MULTILINE)
    return python_script'''
new = '''def extract_python_script(response: str):
    # SCICODE_ROBUST_CODE_EXTRACTION
    if not isinstance(response, str) or not response.strip():
        print("SciCode extraction error: model returned no text.", flush=True)
        return ""

    # Accept normal fence variations and prefer a block containing a definition.
    blocks = re.findall(
        r"```[ \\t]*(?:python|py)?[ \\t]*\\r?\\n?(.*?)```",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    python_script = ""
    if blocks:
        python_script = next(
            (block for block in blocks if re.search(r"(?m)^\\s*(?:async\\s+def|def|class)\\s+", block)),
            blocks[0],
        )
    else:
        # Reasoning should arrive separately, but compatible endpoints can embed
        # it in content. Remove only explicit reasoning wrappers.
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.IGNORECASE | re.DOTALL).strip()
        candidates = [cleaned]
        code_start = re.search(
            r"(?m)^(?=\\s*(?:#\\s*Background:|@|async\\s+def|def|class)\\b)",
            cleaned,
        )
        if code_start and code_start.start() > 0:
            candidates.insert(0, cleaned[code_start.start():])

        # Accept valid unfenced Python. If prose follows the program, trim only
        # trailing lines until a syntactically valid program remains.
        python_script = candidates[0]
        recovered = False
        for candidate in candidates:
            lines = candidate.splitlines()
            for end in range(len(lines), 0, -1):
                possible = "\\n".join(lines[:end]).strip()
                try:
                    compile(possible, "<scicode-response>", "exec")
                except (SyntaxError, ValueError, TypeError):
                    continue
                python_script = possible
                recovered = True
                break
            if recovered:
                break
        if recovered:
            print("SciCode: recovered valid Python from an unfenced response.", flush=True)
        else:
            print("SciCode extraction error: response contains no valid Python block.", flush=True)

    python_script = re.sub(r'^\\s*(import .*|from .*\\s+import\\s+.*)', '', python_script, flags=re.MULTILINE)
    return python_script'''
if old not in src:
    raise SystemExit(f"expected SciCode extractor not found in {p}")
p.write_text(src.replace(old, new, 1))
PY
    fi

    # Avoid parsing each response twice (and emitting duplicate diagnostics).
    if [[ -f "$SCICODE_TASK" ]] && ! grep -q 'SCICODE_SINGLE_EXTRACTION' "$SCICODE_TASK" 2>/dev/null; then
        echo "=== Patching SciCode to extract each response once ==="
        python3 - "$SCICODE_TASK" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old_register = '''        self.previous_llm_code[num_steps - 1] = extract_python_script(response)
        self.save_response_with_steps(
            prob_data,
            response,
            previous_code,
            num_steps,
        )'''
new_register = '''        # SCICODE_SINGLE_EXTRACTION
        python_code = extract_python_script(response)
        self.previous_llm_code[num_steps - 1] = python_code
        self.save_response_with_steps(
            prob_data,
            response,
            previous_code,
            num_steps,
            python_code=python_code,
        )'''
old_signature = '        previous_code: str,' + ' \n' + '''        num_steps: int
    ) -> None:'''
new_signature = '''        previous_code: str,
        num_steps: int,
        python_code: str | None = None,
    ) -> None:'''
old_extract = '        python_code = extract_python_script(response)\n        output_file_path.write_text'
new_extract = '        python_code = python_code if python_code is not None else extract_python_script(response)\n        output_file_path.write_text'
for old_text, new_text, label in (
    (old_register, new_register, "register_previous_response"),
    (old_signature, new_signature, "save_response_with_steps signature"),
    (old_extract, new_extract, "save_response_with_steps extraction"),
):
    if old_text not in src:
        raise SystemExit(f"expected SciCode {label} not found in {p}")
    src = src.replace(old_text, new_text, 1)
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
    def _message_dump(mode=None):
        serialized_tool_calls = None
        if msg.tool_calls:
            serialized_tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return {"role": "assistant", "content": msg.content, "tool_calls": serialized_tool_calls}
    msg.model_dump = _message_dump
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
    # Repair cached environments created by the original accumulator, which
    # discarded tool calls when serializing the reconstructed response.
    if [[ -f "$LITELLM_MODEL" ]] && grep -q 'lambda mode=None: {"content": msg.content, "tool_calls": None}' "$LITELLM_MODEL" 2>/dev/null; then
        echo "=== Repairing mini-swe-agent streaming tool-call serialization ==="
        python3 - "$LITELLM_MODEL" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = '    msg.model_dump = lambda mode=None: {"content": msg.content, "tool_calls": None}'
new = '''    def _message_dump(mode=None):
        serialized_tool_calls = None
        if msg.tool_calls:
            serialized_tool_calls = [
                {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        return {"role": "assistant", "content": msg.content, "tool_calls": serialized_tool_calls}
    msg.model_dump = _message_dump'''
if old not in src:
    raise SystemExit(f"expected broken tool-call serializer not found in {p}")
p.write_text(src.replace(old, new, 1))
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
    # Convert inspect-ai eval log to validate_scores.py format:
    #   {"results": {"scicode": {"sub_problem_correctness": 0.0, "Problem Correctness/mean": 0.0}}}
    # so validate_scores.py can check thresholds.
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
        # Convert inspect-ai log to results.json with validate_scores.py-compatible format
        if [[ "$BENCH" == "scicode" ]]; then
            python3 - "$f" "$DEST/results.json" <<'PY' || true
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
results = {}
# Extract scores from reductions (inspect-ai summary)
for red in data.get("reductions", []):
    scorer_name = red.get("scorer", "unknown")
    for sample in red.get("samples", []):
        pass  # individual sample scores
# Extract from results.scores (aggregate metrics)
for score_entry in data.get("results", {}).get("scores", []):
    scorer = score_entry.get("scorer", score_entry.get("name", "unknown"))
    metrics = score_entry.get("metrics", {})
    task_key = f"scicode/{scorer}"
    results[task_key] = {}
    for metric_name, metric_val in metrics.items():
        val = metric_val.get("value") if isinstance(metric_val, dict) else metric_val
        if isinstance(val, (int, float)):
            results[task_key][metric_name] = val
# Also extract per-sample Problem Correctness from samples
for sample in data.get("samples", []):
    sid = sample.get("id", "unknown")
    for scorer_name, score_obj in sample.get("scores", {}).items():
        val = score_obj.get("value", {})
        if isinstance(val, dict) and "Problem Correctness" in val:
            results[f"scicode/problem_{sid}"] = {"Problem Correctness": val["Problem Correctness"]}
if results:
    out = {"results": results}
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Converted inspect-ai log to results.json with {len(results)} tasks")
PY
        fi
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

    # BFCL native generation records and inference audit. These are needed to
    # distinguish wrong tool calls from adapter/endpoint failures.
    if [[ "$BENCH" == "bfcl" ]]; then
        local BFCL_RAW_INDEX=0
        while IFS= read -r -d '' f; do
            cp -f "$f" "$DEST/bfcl_result_${BFCL_RAW_INDEX}_$(basename "$f")"
            BFCL_RAW_INDEX=$((BFCL_RAW_INDEX + 1))
            COPIED=$((COPIED + 1))
        done < <(find "$OUT_BASE/result" -type f -name '*.json' -print0 2>/dev/null || true)
        if [[ -f "$OUT_BASE/bfcl_inference_audit.json" ]]; then
            cp -f "$OUT_BASE/bfcl_inference_audit.json" "$DEST/"
            COPIED=$((COPIED + 1))
        fi
    fi

    # Preserve Inspect logs, prompts, and generated programs as one artifact so
    # SciCode parser/scorer failures can be audited after the runner cleans up.
    if [[ "$BENCH" == "scicode" && -d "$OUT_BASE" ]]; then
        tar -czf "$DEST/scicode_debug.tar.gz" -C "$OUT_BASE" .
        COPIED=$((COPIED + 1))
    fi

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

    # Convert LCB _eval.json (list format) to validate_scores.py-compatible dict format:
    #   LCB output: [{"pass@1": 1.0, "detail": {...}}, ...]
    #   validate_scores.py expects: {"results": {"livecodebench": {"pass@1": 1.0}}}
    if [[ "$BENCH" == "livecodebench" ]]; then
        local LCB_EVAL_JSON=""
        while IFS= read -r -d '' f; do
            LCB_EVAL_JSON="$f"
            break
        done < <(find "$OUT_BASE" -type f -name '*_eval.json' -print0 2>/dev/null || true)
        if [[ -n "$LCB_EVAL_JSON" ]]; then
            python3 - "$LCB_EVAL_JSON" "$DEST/results.json" <<'PY' || true
import json, os, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
results = {}
# LCB _eval.json is a list; first element has pass@k scores
if isinstance(data, list) and data and isinstance(data[0], dict):
    scores = data[0]
    task_key = "livecodebench"
    results[task_key] = {}
    for k, v in scores.items():
        if isinstance(v, (int, float)):
            results[task_key][k] = v
elif isinstance(data, dict):
    # Already dict format — check for results key
    if "results" in data:
        results = data["results"]
    else:
        results["livecodebench"] = {k: v for k, v in data.items() if isinstance(v, (int, float))}
if results:
    out = {"results": results}
    effective = int(os.environ.get("LIMIT") or 0)
    if effective:
        out["n-samples"] = {"livecodebench": {"effective": effective}}
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Converted LCB eval to results.json with {len(results)} tasks")
PY
        fi
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
set +e
bash "$BENCH_SCRIPT"
BENCH_STATUS=$?
set -e

echo "=== Collecting results for artifact upload ==="
collect_results "${QUALITY_BENCHMARK_NAME}"
exit "$BENCH_STATUS"

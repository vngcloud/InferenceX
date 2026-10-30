import importlib.util
import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = REPO_ROOT / "benchmarks/single_node/quality/tasks/hle"
RUN_SCRIPT = REPO_ROOT / "benchmarks/single_node/quality/run_hle.sh"


def _load_hle_utils():
    spec = importlib.util.spec_from_file_location("hle_task_utils", TASK_DIR / "utils.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _filter_pattern(name: str) -> str:
    config = yaml.load(
        (TASK_DIR / "hle_multiple_choice.yaml").read_text(),
        Loader=yaml.BaseLoader,
    )
    filter_config = next(item for item in config["filter_list"] if item["name"] == name)
    return filter_config["filter"][0]["regex_pattern"]


def test_hle_multiple_choice_extracts_last_final_answer_through_j():
    pattern = _filter_pattern("custom-extract")
    response = "A and E are considered first. The answer is (C). The answer is (J)"

    assert re.findall(pattern, response)[-1] == "J"


def test_hle_flexible_extract_ignores_standalone_reasoning_letters():
    pattern = _filter_pattern("flexible-extract")
    response = "A and E are considered first; the final choice is (I)."

    assert re.findall(pattern, response)[-1] == "I"


def test_hle_prompts_require_unambiguous_final_lines():
    task_utils = _load_hle_utils()

    exact_prompt = task_utils.doc_to_text(
        {"question": "Q", "answer_type": "exactMatch"}
    )
    choice_prompt = task_utils.doc_to_text(
        {"question": "Q", "answer_type": "multipleChoice"}
    )

    assert 'exactly one final line in the form "#### <answer>"' in exact_prompt
    assert 'exactly one final line in the form "The answer is (X)"' in choice_prompt
    assert "A through J" in choice_prompt


def test_hle_uses_long_request_timeout_for_streamed_reasoning():
    script = RUN_SCRIPT.read_text()

    assert 'REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"' in script
    assert "timeout=${REQUEST_TIMEOUT}" in script

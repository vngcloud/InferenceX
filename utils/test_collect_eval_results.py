"""Tests for eval result aggregation."""

import json
from pathlib import Path

from collect_eval_results import build_row, collect_eval_rows


def test_build_row_preserves_sequence_lengths() -> None:
    row = build_row(
        {
            "infmax_model_prefix": "gptoss",
            "hw": "h100",
            "framework": "vllm",
            "precision": "fp4",
            "isl": "1024",
            "osl": "1024",
        },
        {"task": "gsm8k"},
    )

    assert row["isl"] == 1024
    assert row["osl"] == 1024


def _write_lm_eval_result(path: Path, score: float) -> None:
    path.write_text(json.dumps({
        "lm_eval_version": "0.4.0",
        "model_name": "test-model",
        "results": {
            "gsm8k": {
                "exact_match,strict-match": score,
                "exact_match_stderr,strict-match": 0.01,
            },
        },
        "configs": {
            "gsm8k": {
                "metric_list": [{"metric": "exact_match"}],
                "filter_list": [{"name": "strict-match"}],
            },
        },
        "n-samples": {"gsm8k": {"effective": 10}},
    }))


def test_collect_eval_rows_expands_batched_concurrencies(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "eval_batch"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(json.dumps({
        "is_multinode": True,
        "infmax_model_prefix": "gptoss",
        "hw": "gb200",
        "framework": "dynamo-sglang",
        "precision": "fp8",
        "spec_decoding": "none",
        "isl": 8192,
        "osl": 1024,
        "prefill_tp": 4,
        "prefill_ep": 1,
        "prefill_num_workers": 1,
        "decode_tp": 8,
        "decode_ep": 1,
        "decode_num_workers": 2,
        "eval_concs": [4, 16],
        "completed_eval_concs": [4, 16],
        "failed_eval_concs": [],
        "conc": 4,
    }))
    _write_lm_eval_result(
        artifact_dir / "results_test_conc4.json",
        0.90,
    )
    _write_lm_eval_result(
        artifact_dir / "results_test_conc16.json",
        0.91,
    )

    rows = collect_eval_rows(tmp_path)

    assert [row["conc"] for row in rows] == [4, 16]
    assert [row["score"] for row in rows] == [0.90, 0.91]


def test_collect_eval_rows_ignores_failed_batch_points(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "eval_batch"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(json.dumps({
        "is_multinode": True,
        "eval_concs": [4, 16],
        "completed_eval_concs": [4],
        "failed_eval_concs": [16],
        "conc": 4,
    }))
    _write_lm_eval_result(
        artifact_dir / "results_test_conc4.json",
        0.90,
    )
    _write_lm_eval_result(
        artifact_dir / "results_test_conc16.json",
        0.91,
    )

    rows = collect_eval_rows(tmp_path)

    assert [row["conc"] for row in rows] == [4]


def test_collect_eval_rows_normalizes_custom_quality_wrappers(
    tmp_path: Path,
) -> None:
    fixtures = {
        "livecodebench": ({
            "results": {"livecodebench": {"pass@1": 0.75}},
            "n-samples": {"livecodebench": {"effective": 4}},
        }, 0.75, 4),
        "scicode": ({
            "results": {
                "scicode/scicode_scorer": {"mean": 0.5},
                "scicode/problem_11": {"Problem Correctness": 0},
                "scicode/problem_77": {"Problem Correctness": 1},
            },
        }, 0.5, 2),
        "swebench_pro": ({
            "results": {"swebench_pro": {"exact_match,resolved": 1.0}},
            "n-samples": {"swebench_pro": {"effective": 1}},
        }, 1.0, 1),
        "bfcl": ({
            "benchmark": "bfcl",
            "scores": [{"Overall Acc": "62.50%", "Model": "GLM-5.2"}],
        }, 0.625, 8),
        "deepswe": ({
            "result": {"reward": 0.9},
            "completed_tasks": 5,
        }, 0.9, 5),
    }

    for benchmark, (result, _score, _n_eff) in fixtures.items():
        artifact = tmp_path / f"eval_{benchmark}"
        artifact.mkdir()
        (artifact / "meta_env.json").write_text(json.dumps({
            "benchmark": benchmark,
            "infmax_model_prefix": "glm5.2",
        }))
        (artifact / "results.json").write_text(json.dumps(result))
        if benchmark == "bfcl":
            (artifact / "bfcl_inference_audit.json").write_text(json.dumps({
                "checked_responses": 8,
                "inference_errors": 0,
            }))

    rows = {row["task"]: row for row in collect_eval_rows(tmp_path)}

    assert set(rows) == set(fixtures)
    for benchmark, (_result, score, n_eff) in fixtures.items():
        assert rows[benchmark]["score"] == score
        assert rows[benchmark]["n_eff"] == n_eff

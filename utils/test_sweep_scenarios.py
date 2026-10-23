"""Tests for utils/serving/gen_sweep_scenarios.py (sweep-scenarios Task 3).

Expected values hand-computed per the brief: 65536x90% -> 58982,
8192x50% -> 4096, 131072x90% -> 117965, 32768x0% -> 0.
"""

from __future__ import annotations

import yaml

from utils.serving.gen_sweep_scenarios import (
    FIXED_HEADER,
    GRID_CACHE,
    SCENARIOS_DIR,
    _scenario_text,
    build_scenario,
    grid,
    scenario_name,
    shared_prefix_tokens,
)


def test_prefix_tokens_rounding() -> None:
    assert shared_prefix_tokens(65536, 90) == 58982
    assert shared_prefix_tokens(8192, 50) == 4096
    assert shared_prefix_tokens(131072, 90) == 117965
    assert shared_prefix_tokens(32768, 0) == 0


def test_grid_is_the_24_chosen_points() -> None:
    points = grid()
    assert len(points) == 24
    assert {p[0] for p in points} == {"8k", "32k", "64k", "128k"}
    assert {p[2] for p in points} == {"512", "1k"}
    assert {p[4] for p in points} == set(GRID_CACHE) == {0, 50, 90}
    assert len({scenario_name(p[0], p[2], p[4]) for p in points}) == 24


def test_name() -> None:
    assert scenario_name("64k", "1k", 90) == "sweep-isl64k-osl1k-cache90"


def test_built_scenario_carries_grid_values_and_search_budget() -> None:
    doc = yaml.safe_load(yaml.safe_dump(build_scenario(65536, 1024, 90)))
    benchmark = doc["benchmark"]
    # SWEEP_KEY (spike): sweep sits top-level, NOT under benchmark.
    sweep = doc["sweep"]
    assert benchmark["endpoint"]["extra"]["ignore_eos"] is True
    assert "scenario" not in benchmark
    assert sweep["search_space"][0]["lo"] == 1
    assert sweep["search_space"][0]["hi"] == 512
    assert sweep["max_iterations"] == 16
    assert sweep["planner"] == "smooth_isotonic"
    assert sweep["sla_precision"] == "normal"
    # objectives required by AIPerf (spike field-name deviation #2)
    assert sweep["objectives"] == [
        {"metric": "output_token_throughput", "stat": "avg", "direction": "maximize"}
    ]
    filters = {(f["metric_tag"], f["stat"], f["op"], f["threshold"]) for f in sweep["sla_filters"]}
    assert filters == {
        ("time_to_first_token", "avg", "lt", 8000.0),
        ("inter_token_latency", "p90", "lt", 50.0),
    }
    dataset = benchmark["datasets"][0]
    assert dataset["osl"] == 1024
    assert dataset["randomSeed"] == 42
    # ISL_INCLUDES_PREFIX=false (spike): prefix is added on top of the dataset's
    # isl, so total prompt tokens = prefix + isl = the grid ISL target.
    prefix = dataset["prefix_prompts"]["shared_system_length"]
    assert prefix == 58982
    assert prefix + dataset["isl"] == 65536
    phase = benchmark["phases"][0]
    assert phase["duration"] == 300.0
    assert phase["gracePeriod"] == 30.0
    assert phase["failedRequestThreshold"] == 0.1
    assert doc["randomSeed"] == 42


def test_cache_zero_has_no_shared_prefix_and_no_cache_bust() -> None:
    doc = build_scenario(32768, 512, 0)
    dataset = doc["benchmark"]["datasets"][0]
    assert "prefix_prompts" not in dataset
    assert "prompts" not in dataset
    # positive-cache datasets do carry the shared prefix
    doc50 = build_scenario(32768, 512, 50)
    assert doc50["benchmark"]["datasets"][0]["prefix_prompts"]["shared_system_length"] == 16384


def test_isl_accounting_across_grid() -> None:
    for il, isl, ol, osl, cache in grid():
        doc = build_scenario(isl, osl, cache)
        dataset = doc["benchmark"]["datasets"][0]
        if cache:
            assert dataset["prefix_prompts"]["shared_system_length"] == round(isl * cache / 100)
        assert (
            dataset["isl"] + dataset.get("prefix_prompts", {}).get("shared_system_length", 0) == isl
        )
        assert dataset["osl"] == osl


def test_rendered_template_matches_dict_builder_for_all_24_points() -> None:
    """main() writes from _scenario_text(), not build_scenario(); guard against drift."""
    for il, isl, ol, osl, cache in grid():
        assert yaml.safe_load(_scenario_text(isl, osl, cache)) == build_scenario(isl, osl, cache)


def test_fixed_name() -> None:
    assert scenario_name("8k", "1k", 50, kind="fixed") == "fixed-isl8k-osl1k-cache50"


def test_fixed_twin_has_no_sweep_block_and_same_workload() -> None:
    for il, isl, ol, osl, cache in grid():
        fixed = build_scenario(isl, osl, cache, search=False)
        assert "sweep" not in fixed
        assert fixed["benchmark"] == build_scenario(isl, osl, cache)["benchmark"]
        assert yaml.safe_load(_scenario_text(isl, osl, cache, search=False)) == fixed


def test_fixed_files_on_disk_match_generator() -> None:
    for il, isl, ol, osl, cache in grid():
        path = SCENARIOS_DIR / f"{scenario_name(il, ol, cache, kind='fixed')}.yaml"
        assert path.read_text() == FIXED_HEADER + _scenario_text(isl, osl, cache, search=False)


def test_fixed_scenario_is_not_a_search() -> None:
    from infx.serving.to_matrix import is_sweep

    assert not is_sweep("fixed-isl8k-osl1k-cache50")
    assert is_sweep("sweep-isl8k-osl1k-cache50")

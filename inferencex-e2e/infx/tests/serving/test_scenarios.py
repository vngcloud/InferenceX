import copy
from pathlib import Path

import pytest
import yaml

from infx.serving.scenarios import FAMILIES, EmbedPoint, Point, RerankPoint, ScenarioError, render

LEGACY = Path(__file__).parent / "legacy_scenarios"
GEN = {"top_p": 0.95, "temperature": 1.0}

# (legacy file, family, overrides, point, search, expected label)
CASES = [
    ("agentx-cc-062126", "agentx", {}, None, False, "agentx-cc-062126"),
    ("agentx-cc-062126-smoke", "agentx", {"duration": 180}, None, False, "agentx-cc-062126-smoke"),
    ("agentx-cc-062126-256k", "agentx", {"dataset": "semianalysis_cc_traces_weka_062126_256k"}, None, False,
     "agentx-cc-062126-256k"),
    ("agentx-cc-062126-1m", "agentx", {"max_context_length": None}, None, False, "agentx-cc-062126-1m"),
    ("agentx-cc-062126-topp95", "agentx", {"generation": GEN}, None, False, "agentx-cc-062126-topp95"),
    ("custom-agentic-tau2", "tau2", {}, None, False, "custom-agentic-tau2"),
    ("custom-agentic-tau2-smoke", "tau2", {"duration": 180}, None, False, "custom-agentic-tau2-smoke"),
    ("custom-agentic-livekit111868-v1.1-warm", "livekit", {}, None, False,
     "custom-agentic-livekit111868-v1.1-warm"),
    ("custom-agentic-livekit111868-v1", "livekit", {"dataset": "livekit-111868-v1"}, None, False,
     "custom-agentic-livekit111868-v1"),
    ("custom-agentic-livekit111868-v2", "livekit", {"dataset": "livekit-111868-v2"}, None, False,
     "custom-agentic-livekit111868-v2"),
    ("custom-agentic-livekit111868-smoke", "livekit", {"dataset": "livekit-111868-v2", "duration": 180}, None,
     False, "custom-agentic-livekit111868-v2-smoke"),
    ("agentx-shared-prefix-e2", "shared-prefix-e2", {}, None, False, "agentx-shared-prefix-e2"),
    ("agentx-hidden-prefix-e3", "hidden-prefix-e3", {}, None, False, "agentx-hidden-prefix-e3"),
    ("agentx-hidden-prefix-e3-smoke", "hidden-prefix-e3", {"duration": 180}, None, False,
     "agentx-hidden-prefix-e3-smoke"),
    ("sweep-isl8k-osl512-cache50", "synthetic", {}, Point("8k", "512", 50), True, "sweep-isl8k-osl512-cache50"),
    ("sweep-isl8k-osl1k-cache90", "synthetic", {}, Point("8k", "1k", 90), True, "sweep-isl8k-osl1k-cache90"),
    ("sweep-isl128k-osl512-cache0", "synthetic", {}, Point("128k", "512", 0), True, "sweep-isl128k-osl512-cache0"),
    ("sweep-isl128k-osl1k-cache0", "synthetic", {}, Point("128k", "1k", 0), True, "sweep-isl128k-osl1k-cache0"),
    ("fixed-isl8k-osl1k-cache50", "synthetic", {}, Point("8k", "1k", 50), False, "fixed-isl8k-osl1k-cache50"),
    ("fixed-isl8k-osl1k-cache0", "synthetic", {}, Point("8k", "1k", 0), False, "fixed-isl8k-osl1k-cache0"),
]


def without_default_warmup(doc):
    """The doc minus the excluded warmup a search gets by default; the legacy files predate it."""
    doc = copy.deepcopy(doc)
    if "sweep" in doc:
        warmup, profiling = doc["benchmark"]["phases"]
        assert warmup["kind"] == "warmup" and profiling.pop("seamless") is True
        doc["benchmark"]["phases"] = [profiling]
        del doc["sweep"]["sla_warmup_seconds"]
    return doc


@pytest.mark.parametrize("legacy, family, overrides, point, search, label", CASES, ids=[c[0] for c in CASES])
def test_base_plus_variant_equals_the_legacy_file(legacy, family, overrides, point, search, label):
    rendered = render(family, overrides, point, search)
    legacy_doc = yaml.safe_load((LEGACY / f"{legacy}.yaml").read_text())
    assert without_default_warmup(rendered.doc) == legacy_doc
    assert rendered.label == label
    assert yaml.safe_load(rendered.yaml()) == rendered.doc


def test_every_legacy_fixture_is_covered():
    assert {path.stem for path in LEGACY.glob("*.yaml")} == {case[0] for case in CASES}


def test_kinds_and_datasets():
    assert render("agentx").kind == "public"
    tau2 = render("tau2")
    assert (tau2.kind, tau2.dataset, tau2.duration) == ("file", "tau2-cs", 1200.0)
    assert render("livekit").dataset == "livekit-111868-v1.1-warm"
    sweep = render("synthetic", point=Point("8k", "1k", 0), search=True)
    assert (sweep.kind, sweep.search) == ("synthetic", True)


def test_slo_replaces_the_sweep_filters():
    doc = render("synthetic", point=Point("8k", "1k", 0), search=True,
                 slo="time_to_first_token:p95:lt:500;inter_token_latency:p50:lt:50").doc
    assert doc["sweep"]["sla_filters"] == [
        {"metric_tag": "time_to_first_token", "stat": "p95", "op": "lt", "threshold": 500.0},
        {"metric_tag": "inter_token_latency", "stat": "p50", "op": "lt", "threshold": 50.0},
    ]


@pytest.mark.parametrize("args, message", [
    (("agentx", {"warmup": 1}), "overrides: unknown key"),
    (("agentx", {"dataset": "tau2-cs"}), "overrides.dataset: agentx takes a public"),
    (("tau2", {"dataset": "semianalysis_cc_traces_weka_062126"}), "is not in the dataset lock"),
    (("agentx", {"duration": 0}), "overrides.duration"),
    (("agentx", {"generation": {"top_k": 5}}), "overrides.generation"),
    (("synthetic", {}), "workload: synthetic needs"),
    (("synthetic", {"dataset": "x"}, Point("8k", "1k", 0)), "overrides: synthetic only takes duration, warmup"),
    (("synthetic", {"warmup": -1}, Point("8k", "1k", 0)), "overrides.warmup"),
    (("synthetic", {}, Point("16k", "1k", 0)), "workload.isl"),
    (("synthetic", {}, Point("8k", "1k", 0), False, "time_to_first_token:avg:lt:1"), "slo: only applies"),
    (("agentx", {}, Point("8k", "1k", 0)), "workload/slo: only for scenario synthetic"),
    (("nope",), "scenario: 'nope' is not one of"),
])
def test_bad_variants_fail_with_one_line(args, message):
    with pytest.raises(ScenarioError, match=message):
        render(*args)


def test_greenspec_renders_the_selection_and_has_no_duration():
    rendered = render("greenspec", {"subset": "agentic_tools", "lite": True, "requests": 50})
    dataset = rendered.doc["benchmark"]["datasets"][0]
    assert (rendered.kind, rendered.dataset, rendered.label) == ("file", "greenspec-v1.1.1", "greenspec-agentic_tools-lite-n50")
    assert dataset["format"] == "raw_payload" and dataset["path"] == "${IX_CUSTOM_DATASET_FILE}"
    [phase] = rendered.doc["benchmark"]["phases"]
    assert phase["requests"] == 50 and "duration" not in phase and rendered.duration == 0.0
    assert "ignore_eos" not in rendered.doc["benchmark"]["endpoint"].get("extra", {})
    assert render("greenspec").label == "greenspec-diverse"


@pytest.mark.parametrize("args, message", [
    (("greenspec", {"subset": "coding"}), "overrides.subset"),
    (("greenspec", {"duration": 180}), "greenspec takes"),
    (("greenspec", {}, Point("8k", "1k", 0)), "workload/slo: only for scenario synthetic"),
])
def test_bad_greenspec_variants_fail_with_one_line(args, message):
    with pytest.raises(ScenarioError, match=message):
        render(*args)


def test_speedbench_renders_the_config_dataset_and_has_no_duration():
    rendered = render("speedbench", {"config": "throughput_8k", "category": "mixed", "requests": 30})
    dataset = rendered.doc["benchmark"]["datasets"][0]
    assert (rendered.kind, rendered.dataset, rendered.label) == (
        "file", "speedbench-throughput-8k", "speedbench-throughput_8k-mixed-n30")
    assert dataset["name"] == "speedbench-throughput-8k"
    assert dataset["format"] == "raw_payload" and dataset["path"] == "${IX_CUSTOM_DATASET_FILE}"
    [phase] = rendered.doc["benchmark"]["phases"]
    assert phase["requests"] == 30 and "duration" not in phase and rendered.duration == 0.0
    assert "ignore_eos" not in rendered.doc["benchmark"]["endpoint"].get("extra", {})
    default = render("speedbench")
    assert (default.label, default.dataset) == ("speedbench-qualitative", "speedbench-qualitative")


@pytest.mark.parametrize("args, message", [
    (("speedbench", {"category": "mixed"}), "overrides.category"),
    (("speedbench", {"subset": "diverse"}), "speedbench takes"),
    (("speedbench", {}, Point("8k", "1k", 0)), "workload/slo: only for scenario synthetic, not speedbench"),
])
def test_bad_speedbench_variants_fail_with_one_line(args, message):
    with pytest.raises(ScenarioError, match=message):
        render(*args)


def test_warmup_requests_per_lane_override():
    def profiling(doc):
        return next(p for p in doc["benchmark"]["phases"] if p.get("kind") == "profiling")

    off = render("livekit", {"warmup_requests_per_lane": 0})
    assert "warmupRequestsPerLane" not in profiling(off.doc)
    assert off.label == "custom-agentic-livekit111868-v1.1-warm-nowarmup"
    on = render("livekit", {"warmup_requests_per_lane": 3})
    assert profiling(on.doc)["warmupRequestsPerLane"] == 3
    assert on.label == "custom-agentic-livekit111868-v1.1-warm-warm3"
    with pytest.raises(ScenarioError, match="overrides.warmup_requests_per_lane"):
        render("livekit", {"warmup_requests_per_lane": True})


def test_families_have_base_files():
    from infx.serving.scenarios import SCENARIOS_DIR
    assert {path.stem for path in SCENARIOS_DIR.glob("*.yaml")} == set(FAMILIES)


def test_synthetic_search_warms_up_by_default_and_hands_off_seamlessly():
    slo = "time_to_first_token:p95:lt:8000"
    rendered = render("synthetic", {}, Point("8k", "1k", 50), True, slo)
    warmup, profiling = rendered.doc["benchmark"]["phases"]
    assert (warmup["kind"], warmup["duration"]) == ("warmup", 120.0)
    assert profiling["kind"] == "profiling" and profiling["seamless"] is True
    assert rendered.doc["sweep"]["sla_warmup_seconds"] == 120.0
    assert rendered.label == "sweep-isl8k-osl1k-cache50"
    cold = render("synthetic", {"warmup": 0}, Point("8k", "1k", 50), True, slo)
    assert [p["kind"] for p in cold.doc["benchmark"]["phases"]] == ["profiling"]
    assert cold.doc["sweep"]["sla_warmup_seconds"] == 0 and cold.label.endswith("-w0")
    fixed = render("synthetic", {"duration": 180}, Point("8k", "1k", 0))
    assert [p["kind"] for p in fixed.doc["benchmark"]["phases"]] == ["profiling"]
    warm = render("synthetic", {"duration": 180, "warmup": 120}, Point("8k", "1k", 0))
    assert warm.doc["benchmark"]["phases"][0]["kind"] == "warmup"
    assert warm.label == "fixed-isl8k-osl1k-cache0-smoke-w120"


def test_embed_search_point_fills_prompts_and_searches_request_throughput():
    rendered = render("embed", {}, EmbedPoint("2k", 32), search=True)
    dataset = rendered.doc["benchmark"]["datasets"][0]
    assert dataset["prompts"] == {"isl": 2048, "batchSize": 32}
    assert rendered.doc["benchmark"]["endpoint"] == {"type": "embeddings", "path": "/v1/embeddings"}
    sweep = rendered.doc["sweep"]
    assert sweep["objectives"] == [
        {"metric": "request_throughput", "stat": "avg", "direction": "maximize"}
    ]
    assert sweep["sla_filters"] == [
        {"metric_tag": "request_latency", "stat": "p90", "op": "lt", "threshold": 500.0}
    ]
    assert (rendered.label, rendered.kind, rendered.search) == ("sweep-embed-isl2k-b32", "synthetic", True)


def test_rerank_fixed_point_fills_rankings_without_a_sweep():
    rendered = render("rerank", {"duration": 60}, RerankPoint("512", 40))
    dataset = rendered.doc["benchmark"]["datasets"][0]
    assert dataset["rankings"] == {"queryTokens": 32, "passages": 40, "passageTokens": 512}
    assert rendered.doc["benchmark"]["endpoint"] == {"type": "cohere_rankings", "path": "/v1/rerank"}
    assert "sweep" not in rendered.doc
    assert (rendered.label, rendered.duration) == ("fixed-rerank-p512-n40-d60", 60.0)


def test_rerank_search_uses_its_own_default_slo_and_accepts_an_override():
    default = render("rerank", {}, RerankPoint("2k", 10), search=True)
    assert default.doc["sweep"]["sla_filters"][0]["threshold"] == 1000.0
    custom = render("rerank", {}, RerankPoint("2k", 10), search=True, slo="request_latency:p99:lt:200")
    assert custom.doc["sweep"]["sla_filters"] == [
        {"metric_tag": "request_latency", "stat": "p99", "op": "lt", "threshold": 200.0}
    ]


def test_pooling_sizes_beyond_the_nim_grid_render():
    embed = render("embed", {}, EmbedPoint("2k", 16), search=True)
    assert embed.doc["benchmark"]["datasets"][0]["prompts"] == {"isl": 2048, "batchSize": 16}
    assert embed.label == "sweep-embed-isl2k-b16"
    rerank = render("rerank", {}, RerankPoint("512", 100), search=True)
    assert rerank.doc["benchmark"]["datasets"][0]["rankings"]["passages"] == 100
    assert rerank.label == "sweep-rerank-p512-n100"


@pytest.mark.parametrize(
    "family, point, slo, message",
    [
        ("embed", EmbedPoint("4k", 1), None, "workload.isl: '4k'"),
        ("embed", EmbedPoint("2k", 0), None, "workload.batch: 0"),
        ("rerank", RerankPoint("1k", 10), None, "workload.passage: '1k'"),
        ("rerank", RerankPoint("512", 0), None, "workload.docs: 0"),
        ("embed", EmbedPoint("2k", 1), "request_latency:p90:lt:5", "slo: only applies"),
        ("rerank", EmbedPoint("2k", 1), None, "workload: rerank takes passage, docs"),
        ("embed", None, None, "workload: embed takes isl, batch"),
    ],
)
def test_pooling_rejects_points_outside_the_grid(family, point, slo, message):
    with pytest.raises(ScenarioError, match=message):
        render(family, {}, point, False, slo)


def test_pooling_takes_only_the_duration_override():
    with pytest.raises(ScenarioError, match="overrides: embed only takes duration"):
        render("embed", {"dataset": "x"}, EmbedPoint("512", 1))

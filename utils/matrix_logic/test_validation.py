"""Comprehensive tests for validation.py"""
import pytest
from validation import (
    Fields,
    SingleNodeMatrixEntry,
    MultiNodeMatrixEntry,
    SingleNodeAgenticMatrixEntry,
    MultiNodeAgenticMatrixEntry,
    SingleNodeAgenticReplayMatrixEntry,
    WorkerConfig,
    SingleNodeSearchSpaceEntry,
    MultiNodeSearchSpaceEntry,
    SingleNodeSeqLenConfig,
    MultiNodeSeqLenConfig,
    AgenticCodingConfig,
    AgenticReplayConfig,
    SingleNodeMasterConfigEntry,
    MultiNodeMasterConfigEntry,
    validate_matrix_entry,
    validate_agentic_replay_matrix_entry,
    validate_master_config,
    validate_runner_config,
    load_config_files,
    load_runner_file,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def valid_single_node_matrix_entry():
    """Valid single node matrix entry based on dsr1-fp4-mi355x-sglang config."""
    return {
        "image": "rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi35x-20250915",
        "model": "amd/DeepSeek-R1-0528-MXFP4-Preview",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "sglang",
        "spec-decoding": "none",
        "runner": "mi355x",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "ep": 1,
        "dp-attn": False,
        "conc": 4,
        "max-model-len": 2248,
        "exp-name": "dsr1_1k1k",
        "disagg": False,
        "run-eval": False,
    }


@pytest.fixture
def valid_multinode_matrix_entry():
    """Valid multinode matrix entry based on dsr1-fp4-gb200-dynamo-trt config."""
    return {
        "image": "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:0.5.1-rc0.pre3",
        "model": "deepseek-r1-fp4",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "dynamo-trt",
        "spec-decoding": "none",
        "runner": "gb200",
        "isl": 1024,
        "osl": 1024,
        "prefill": {
            "num-worker": 5,
            "tp": 4,
            "ep": 4,
            "dp-attn": True,
            "additional-settings": [
                "PREFILL_MAX_NUM_TOKENS=8448",
                "PREFILL_MAX_BATCH_SIZE=1",
            ],
        },
        "decode": {
            "num-worker": 1,
            "tp": 8,
            "ep": 8,
            "dp-attn": True,
            "additional-settings": [
                "DECODE_MAX_NUM_TOKENS=256",
                "DECODE_MAX_BATCH_SIZE=256",
                "DECODE_GPU_MEM_FRACTION=0.8",
                "DECODE_MTP_SIZE=0",
            ],
        },
        "conc": [2150],
        "max-model-len": 2248,
        "exp-name": "dsr1_1k1k",
        "disagg": True,
        "run-eval": False,
    }


@pytest.fixture
def valid_single_node_master_config():
    """Valid single node master config based on dsr1-fp8-mi300x-sglang."""
    return {
        "image": "rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi30x-20250915",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model-prefix": "dsr1",
        "precision": "fp8",
        "framework": "sglang",
        "runner": "mi300x",
        "multinode": False,
        "scenarios": {
            "fixed-seq-len": [

                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {"tp": 8, "conc-start": 4, "conc-end": 64}
                    ]
                }
            ]
        }
    }


@pytest.fixture
def valid_multinode_master_config():
    """Valid multinode master config based on dsr1-fp4-gb200-dynamo-trt."""
    return {
        "image": "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:0.5.1-rc0.pre3",
        "model": "deepseek-r1-fp4",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "dynamo-trt",
        "runner": "gb200",
        "multinode": True,
        "disagg": True,
        "scenarios": {
            "fixed-seq-len": [

                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {
                            "prefill": {
                                "num-worker": 5,
                                "tp": 4,
                                "ep": 4,
                                "dp-attn": True,
                                "additional-settings": [
                                    "PREFILL_MAX_NUM_TOKENS=8448",
                                    "PREFILL_MAX_BATCH_SIZE=1",
                                ],
                            },
                            "decode": {
                                "num-worker": 1,
                                "tp": 8,
                                "ep": 8,
                                "dp-attn": True,
                                "additional-settings": [
                                    "DECODE_MAX_NUM_TOKENS=256",
                                    "DECODE_MAX_BATCH_SIZE=256",
                                ],
                            },
                            "conc-list": [2150],
                        }
                    ]
                }
            ]
        }
    }


@pytest.fixture
def valid_runner_config():
    """Valid runner config based on .github/configs/runners.yaml."""
    return {
        "h100": ["h100-cr_0", "h100-cr_1", "h100-cw_0", "h100-cw_1"],
        "h200": ["h200-cw_0", "h200-cw_1", "h200-nb_0", "h200-nb_1"],
        "b200": ["b200-nvd_0", "b200-nvd_1", "b200-dgxc_1"],
        "mi300x": ["mi300x-amd_0", "mi300x-amd_1", "mi300x-cr_0"],
        "gb200": ["gb200-nv_0"],
    }


# =============================================================================
# Test Fields Enum
# =============================================================================

class TestFieldsEnum:
    """Tests for Fields enum."""

    def test_field_values_are_strings(self):
        """All field values should be strings."""
        for field in Fields:
            assert isinstance(field.value, str)

    def test_key_fields_exist(self):
        """Key fields should be defined."""
        assert Fields.IMAGE.value == "image"
        assert Fields.MODEL.value == "model"
        assert Fields.TP.value == "tp"
        assert Fields.MULTINODE.value == "multinode"
        assert Fields.CONC.value == "conc"
        assert Fields.BENCHMARK_CLIENT.value == "benchmark-client"
        assert Fields.SPEC_DECODING.value == "spec-decoding"
        assert Fields.PREFILL.value == "prefill"
        assert Fields.DECODE.value == "decode"


# =============================================================================
# Test WorkerConfig
# =============================================================================

class TestWorkerConfig:
    """Tests for WorkerConfig model."""

    def test_valid_worker_config(self):
        """Valid worker config should pass."""
        config = WorkerConfig(**{
            "num-worker": 5,
            "tp": 4,
            "ep": 4,
            "dp-attn": True,
        })
        assert config.num_worker == 5
        assert config.tp == 4
        assert config.ep == 4
        assert config.dp_attn is True

    def test_worker_config_with_additional_settings(self):
        """Worker config with additional settings should pass."""
        config = WorkerConfig(**{
            "num-worker": 1,
            "tp": 8,
            "ep": 8,
            "dp-attn": True,
            "additional-settings": [
                "DECODE_MAX_NUM_TOKENS=256",
                "DECODE_MAX_BATCH_SIZE=256",
                "DECODE_GPU_MEM_FRACTION=0.8",
            ],
        })
        assert len(config.additional_settings) == 3
        assert "DECODE_MAX_NUM_TOKENS=256" in config.additional_settings

    def test_worker_config_missing_required_field(self):
        """Missing required field should fail."""
        with pytest.raises(Exception):
            WorkerConfig(**{
                "num-worker": 2,
                "tp": 4,
                # Missing ep and dp-attn
            })

    def test_worker_config_extra_field_forbidden(self):
        """Extra fields should be forbidden."""
        with pytest.raises(Exception):
            WorkerConfig(**{
                "num-worker": 2,
                "tp": 4,
                "ep": 1,
                "dp-attn": False,
                "unknown-field": "value",
            })


# =============================================================================
# Test SingleNodeMatrixEntry
# =============================================================================

class TestSingleNodeMatrixEntry:
    """Tests for SingleNodeMatrixEntry model."""

    def test_valid_entry(self, valid_single_node_matrix_entry):
        """Valid entry should pass validation."""
        entry = SingleNodeMatrixEntry(**valid_single_node_matrix_entry)
        assert entry.image == "rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi35x-20250915"
        assert entry.tp == 8
        assert entry.conc == 4
        assert entry.framework == "sglang"
        assert entry.benchmark_client == "inferencex_native"

    def test_benchmark_client_aiperf(self, valid_single_node_matrix_entry):
        """AIPerf should validate as benchmark client, not framework."""
        valid_single_node_matrix_entry["benchmark-client"] = "aiperf"
        entry = SingleNodeMatrixEntry(**valid_single_node_matrix_entry)
        assert entry.benchmark_client == "aiperf"
        assert entry.framework == "sglang"

    def test_invalid_benchmark_client(self, valid_single_node_matrix_entry):
        """Invalid benchmark client value should fail."""
        valid_single_node_matrix_entry["benchmark-client"] = "foo"
        with pytest.raises(Exception):
            SingleNodeMatrixEntry(**valid_single_node_matrix_entry)

    def test_conc_as_list(self, valid_single_node_matrix_entry):
        """Conc can be a list of integers."""
        valid_single_node_matrix_entry["conc"] = [4, 8, 16, 32, 64]
        entry = SingleNodeMatrixEntry(**valid_single_node_matrix_entry)
        assert entry.conc == [4, 8, 16, 32, 64]

    def test_spec_decoding_values(self, valid_single_node_matrix_entry):
        """Spec decoding should accept valid literal values."""
        for value in ["mtp", "draft_model", "none"]:
            valid_single_node_matrix_entry["spec-decoding"] = value
            entry = SingleNodeMatrixEntry(**valid_single_node_matrix_entry)
            assert entry.spec_decoding == value

    def test_invalid_spec_decoding(self, valid_single_node_matrix_entry):
        """Invalid spec decoding value should fail."""
        valid_single_node_matrix_entry["spec-decoding"] = "invalid"
        with pytest.raises(Exception):
            SingleNodeMatrixEntry(**valid_single_node_matrix_entry)

    def test_missing_required_field(self, valid_single_node_matrix_entry):
        """Missing required field should fail validation."""
        del valid_single_node_matrix_entry["model"]
        with pytest.raises(Exception):
            SingleNodeMatrixEntry(**valid_single_node_matrix_entry)

    def test_extra_field_forbidden(self, valid_single_node_matrix_entry):
        """Extra fields should be forbidden."""
        valid_single_node_matrix_entry["extra-field"] = "value"
        with pytest.raises(Exception):
            SingleNodeMatrixEntry(**valid_single_node_matrix_entry)


# =============================================================================
# Test MultiNodeMatrixEntry
# =============================================================================

class TestMultiNodeMatrixEntry:
    """Tests for MultiNodeMatrixEntry model."""

    def test_valid_entry(self, valid_multinode_matrix_entry):
        """Valid entry should pass validation."""
        entry = MultiNodeMatrixEntry(**valid_multinode_matrix_entry)
        assert entry.model == "deepseek-r1-fp4"
        assert entry.conc == [2150]
        assert entry.disagg is True
        assert entry.benchmark_client == "inferencex_native"

    def test_benchmark_client_aiperf(self, valid_multinode_matrix_entry):
        """AIPerf should validate as benchmark client for multinode entries."""
        valid_multinode_matrix_entry["benchmark-client"] = "aiperf"
        entry = MultiNodeMatrixEntry(**valid_multinode_matrix_entry)
        assert entry.benchmark_client == "aiperf"

    def test_invalid_benchmark_client(self, valid_multinode_matrix_entry):
        """Invalid benchmark client value should fail for multinode entries."""
        valid_multinode_matrix_entry["benchmark-client"] = "foo"
        with pytest.raises(Exception):
            MultiNodeMatrixEntry(**valid_multinode_matrix_entry)

    def test_prefill_decode_worker_configs(self, valid_multinode_matrix_entry):
        """Prefill and decode should be WorkerConfig objects."""
        entry = MultiNodeMatrixEntry(**valid_multinode_matrix_entry)
        assert entry.prefill.num_worker == 5
        assert entry.prefill.tp == 4
        assert entry.decode.tp == 8
        assert entry.decode.dp_attn is True

    def test_conc_must_be_list(self, valid_multinode_matrix_entry):
        """Conc must be a list for multinode."""
        valid_multinode_matrix_entry["conc"] = 2150  # Single int, not list
        with pytest.raises(Exception):
            MultiNodeMatrixEntry(**valid_multinode_matrix_entry)

    def test_missing_prefill(self, valid_multinode_matrix_entry):
        """Missing prefill should fail."""
        del valid_multinode_matrix_entry["prefill"]
        with pytest.raises(Exception):
            MultiNodeMatrixEntry(**valid_multinode_matrix_entry)

    def test_missing_decode(self, valid_multinode_matrix_entry):
        """Missing decode should fail."""
        del valid_multinode_matrix_entry["decode"]
        with pytest.raises(Exception):
            MultiNodeMatrixEntry(**valid_multinode_matrix_entry)


# =============================================================================
# Test validate_matrix_entry function
# =============================================================================

class TestValidateMatrixEntry:
    """Tests for validate_matrix_entry function."""

    def test_valid_single_node(self, valid_single_node_matrix_entry):
        """Valid single node entry should return the entry."""
        result = validate_matrix_entry(valid_single_node_matrix_entry, is_multinode=False)
        assert result == valid_single_node_matrix_entry

    def test_valid_multinode(self, valid_multinode_matrix_entry):
        """Valid multinode entry should return the entry."""
        result = validate_matrix_entry(valid_multinode_matrix_entry, is_multinode=True)
        assert result == valid_multinode_matrix_entry

    def test_invalid_single_node_raises_valueerror(self, valid_single_node_matrix_entry):
        """Invalid single node entry should raise ValueError."""
        del valid_single_node_matrix_entry["tp"]
        with pytest.raises(ValueError) as exc_info:
            validate_matrix_entry(valid_single_node_matrix_entry, is_multinode=False)
        assert "failed validation" in str(exc_info.value)

    def test_invalid_multinode_raises_valueerror(self, valid_multinode_matrix_entry):
        """Invalid multinode entry should raise ValueError."""
        del valid_multinode_matrix_entry["prefill"]
        with pytest.raises(ValueError) as exc_info:
            validate_matrix_entry(valid_multinode_matrix_entry, is_multinode=True)
        assert "failed validation" in str(exc_info.value)


# =============================================================================
# Test Agentic Matrix Entries
# =============================================================================

class TestAgenticMatrixEntries:
    """Tests for benchmark-client validation on agentic matrix entries."""

    def test_single_node_agentic_benchmark_client_default(self):
        entry = SingleNodeAgenticMatrixEntry(**{
            "image": "test-image",
            "model": "test-model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h100",
            "tp": 8,
            "ep": 1,
            "dp-attn": False,
            "conc": 8,
            "offloading": "none",
            "exp-name": "agentic_test",
            "scenario-type": "agentic-coding",
        })
        assert entry.benchmark_client == "inferencex_native"

    def test_single_node_agentic_benchmark_client_aiperf(self):
        entry = SingleNodeAgenticMatrixEntry(**{
            "image": "test-image",
            "model": "test-model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "benchmark-client": "aiperf",
            "runner": "h100",
            "tp": 8,
            "ep": 1,
            "dp-attn": False,
            "conc": 8,
            "offloading": "none",
            "exp-name": "agentic_test",
            "scenario-type": "agentic-coding",
        })
        assert entry.benchmark_client == "aiperf"

    def test_single_node_agentic_invalid_benchmark_client(self):
        with pytest.raises(Exception):
            SingleNodeAgenticMatrixEntry(**{
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "benchmark-client": "foo",
                "runner": "h100",
                "tp": 8,
                "ep": 1,
                "dp-attn": False,
                "conc": 8,
                "offloading": "none",
                "exp-name": "agentic_test",
                "scenario-type": "agentic-coding",
            })

    def test_multinode_agentic_benchmark_client_default(self):
        entry = MultiNodeAgenticMatrixEntry(**{
            "image": "test-image",
            "model": "test-model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "spec-decoding": "none",
            "runner": "h100",
            "prefill": {"num-worker": 1, "tp": 4, "ep": 1, "dp-attn": False},
            "decode": {"num-worker": 1, "tp": 4, "ep": 1, "dp-attn": False},
            "conc": 8,
            "exp-name": "agentic_test",
            "disagg": True,
            "scenario-type": "agentic-coding",
        })
        assert entry.benchmark_client == "inferencex_native"

    def test_multinode_agentic_benchmark_client_aiperf(self):
        entry = MultiNodeAgenticMatrixEntry(**{
            "image": "test-image",
            "model": "test-model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "benchmark-client": "aiperf",
            "spec-decoding": "none",
            "runner": "h100",
            "prefill": {"num-worker": 1, "tp": 4, "ep": 1, "dp-attn": False},
            "decode": {"num-worker": 1, "tp": 4, "ep": 1, "dp-attn": False},
            "conc": 8,
            "exp-name": "agentic_test",
            "disagg": True,
            "scenario-type": "agentic-coding",
        })
        assert entry.benchmark_client == "aiperf"

    def test_multinode_agentic_invalid_benchmark_client(self):
        with pytest.raises(Exception):
            MultiNodeAgenticMatrixEntry(**{
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "vllm",
                "benchmark-client": "foo",
                "spec-decoding": "none",
                "runner": "h100",
                "prefill": {"num-worker": 1, "tp": 4, "ep": 1, "dp-attn": False},
                "decode": {"num-worker": 1, "tp": 4, "ep": 1, "dp-attn": False},
                "conc": 8,
                "exp-name": "agentic_test",
                "disagg": True,
                "scenario-type": "agentic-coding",
            })

# =============================================================================
# Test Agentic-Replay Entries
# =============================================================================

class TestAgenticReplayMatrixEntries:
    """Tests for the single-node agentic-replay matrix entry and validator."""

    def _entry(self, **overrides):
        entry = {
            "image": "vllm/vllm-openai:v0.21.0",
            "model": "Qwen/Qwen3.5-4B",
            "model-prefix": "qwen3.5-4b",
            "precision": "bf16",
            "framework": "vllm",
            "benchmark-client": "aiperf",
            "runner": "h100-2x",
            "tp": 1,
            "ep": 1,
            "dp-attn": False,
            "conc": 2,
            "isl": 4096,
            "osl": 512,
            "max-model-len": 8192,
            "input-file": "benchmarks/single_node/agentic/datasets/qwen3.5-4b-smoke.jsonl",
            "custom-dataset-type": "mooncake_trace",
            "exp-name": "qwen3.5-4b_tp1_conc2",
            "disagg": False,
            "scenario-type": "agentic-replay",
        }
        entry.update(overrides)
        return entry

    def test_valid_entry(self):
        entry = SingleNodeAgenticReplayMatrixEntry(**self._entry())
        assert entry.benchmark_client == "aiperf"
        assert entry.input_file.endswith("qwen3.5-4b-smoke.jsonl")
        assert entry.custom_dataset_type == "mooncake_trace"
        assert entry.duration == 1800  # default
        assert entry.fixed_schedule is False
        assert entry.max_context_length is None

    def test_benchmark_client_defaults_to_aiperf(self):
        raw = self._entry()
        del raw["benchmark-client"]

        entry = SingleNodeAgenticReplayMatrixEntry(**raw)

        assert entry.benchmark_client == "aiperf"

    def test_remote_allowed(self):
        entry = SingleNodeAgenticReplayMatrixEntry(**self._entry(remote={
            "url": "http://remote:8000",
            "server-metrics-url": "http://remote:8000/metrics",
            "gpu-telemetry-url": "http://remote:9400/metrics",
        }))

        assert entry.remote.url == "http://remote:8000"

    def test_remote_api_key_secret_name_allowed(self):
        """A remote config may name the GitHub secret holding its API key; the
        reusable workflow resolves that name dynamically at run time."""
        entry = SingleNodeAgenticReplayMatrixEntry(**self._entry(remote={
            "url": "http://remote:8000",
            "api-key-secret-name": "MAAS_HCM_API_KEY",
        }))

        assert entry.remote.api_key_secret_name == "MAAS_HCM_API_KEY"

    def test_remote_api_key_secret_name_defaults_none(self):
        entry = SingleNodeAgenticReplayMatrixEntry(**self._entry(remote={
            "url": "http://remote:8000",
        }))

        assert entry.remote.api_key_secret_name is None

    def test_remote_server_command_is_optional_metadata(self):
        command = "python3 -m sglang.launch_server --tp-size 8 --ep-size 8"
        entry = SingleNodeAgenticReplayMatrixEntry(**self._entry(remote={
            "url": "http://remote:8000",
            "server-command": command,
        }))

        assert entry.remote.server_command == command

    def test_remote_url_list_joined_to_comma_separated_string(self):
        """A model hosted across multiple endpoints may be declared as a list;
        RemoteConfig normalizes it to aiperf's comma-separated multi-URL
        syntax."""
        entry = SingleNodeAgenticReplayMatrixEntry(**self._entry(remote={
            "url": ["http://a:8000", "http://b:8000"],
            "server-metrics-url": ["http://a:8000/metrics", "http://b:8000/metrics"],
            "gpu-telemetry-url": "http://a:9400/metrics",
        }))

        assert entry.remote.url == "http://a:8000,http://b:8000"
        assert entry.remote.server_metrics_url == "http://a:8000/metrics,http://b:8000/metrics"
        assert entry.remote.gpu_telemetry_url == "http://a:9400/metrics"

    def test_remote_url_empty_list_rejected(self):
        with pytest.raises(Exception):
            SingleNodeAgenticReplayMatrixEntry(**self._entry(remote={"url": []}))

    def test_validator_passes(self):
        # validator returns the original dict on success
        e = self._entry()
        assert validate_agentic_replay_matrix_entry(e) is e

    def test_missing_input_file_rejected(self):
        bad = self._entry()
        del bad["input-file"]
        with pytest.raises(ValueError):
            validate_agentic_replay_matrix_entry(bad)

    def test_extra_field_rejected(self):
        with pytest.raises(Exception):
            SingleNodeAgenticReplayMatrixEntry(**self._entry(offloading="none"))

    def test_invalid_benchmark_client_rejected(self):
        with pytest.raises(Exception):
            SingleNodeAgenticReplayMatrixEntry(**self._entry(**{"benchmark-client": "foo"}))

    def test_native_benchmark_client_rejected(self):
        with pytest.raises(Exception):
            SingleNodeAgenticReplayMatrixEntry(**self._entry(**{"benchmark-client": "inferencex_native"}))


class TestAgenticReplayConfig:
    """Tests for the input-side agentic-replay scenario config."""

    def _config(self, **overrides):
        cfg = {
            "input-file": "benchmarks/single_node/agentic/datasets/qwen3.5-4b-smoke.jsonl",
            "custom-dataset-type": "mooncake_trace",
            "max-model-len": 8192,
            "benchmark-client": ["aiperf"],
            "search-space": [{"tp": 1, "conc-list": [2]}],
        }
        cfg.update(overrides)
        return cfg

    def test_valid_config(self):
        cfg = AgenticReplayConfig(**self._config())
        assert cfg.custom_dataset_type == "mooncake_trace"
        assert cfg.max_model_len == 8192
        assert cfg.benchmark_client == ["aiperf"]

    def test_fixed_schedule_defaults(self):
        cfg = AgenticReplayConfig(**self._config())

        assert cfg.fixed_schedule is False
        assert cfg.max_context_length is None

    def test_fixed_schedule_fields_serialize_with_aliases(self):
        cfg = AgenticReplayConfig(**self._config(**{
            "fixed-schedule": True,
            "max-context-length": 100000,
        }))

        dumped = cfg.model_dump(by_alias=True)

        assert dumped["fixed-schedule"] is True
        assert dumped["max-context-length"] == 100000

    @pytest.mark.parametrize("max_context_length", [0, -1])
    def test_max_context_length_must_be_positive(self, max_context_length):
        with pytest.raises(Exception):
            AgenticReplayConfig(**self._config(**{
                "max-context-length": max_context_length,
            }))

    def test_unknown_field_rejected(self):
        with pytest.raises(Exception):
            AgenticReplayConfig(**self._config(unknown=True))

    def test_benchmark_client_defaults_to_aiperf(self):
        raw = self._config()
        del raw["benchmark-client"]

        cfg = AgenticReplayConfig(**raw)

        assert cfg.benchmark_client == ["aiperf"]

    def test_native_benchmark_client_rejected(self):
        with pytest.raises(Exception):
            AgenticReplayConfig(**self._config(**{"benchmark-client": ["inferencex_native"]}))

    def test_requires_input_file(self):
        bad = self._config()
        del bad["input-file"]
        with pytest.raises(Exception):
            AgenticReplayConfig(**bad)

    def test_weka_defaults_to_public_dataset(self):
        raw = self._config(**{"custom-dataset-type": "weka_trace"})
        del raw["input-file"]

        cfg = AgenticReplayConfig(**raw)

        assert cfg.public_dataset == "semianalysis_cc_traces_weka_with_subagents_060826"
        assert cfg.input_file is None

    def test_weka_accepts_explicit_public_dataset(self):
        raw = self._config(**{
            "custom-dataset-type": "weka_trace",
            "public-dataset": "semianalysis_cc_traces_weka_with_subagents_060826",
        })
        del raw["input-file"]

        cfg = AgenticReplayConfig(**raw)

        assert cfg.public_dataset == "semianalysis_cc_traces_weka_with_subagents_060826"

    def test_weka_accepts_generic_hf_repo(self):
        raw = self._config(**{
            "custom-dataset-type": "weka_trace",
            "public-dataset": "weka_hf",
            "hf-weka-repo": "semianalysisai/cc-traces-weka-062126",
        })
        del raw["input-file"]

        cfg = AgenticReplayConfig(**raw)

        assert cfg.public_dataset == "weka_hf"
        assert cfg.hf_weka_repo == "semianalysisai/cc-traces-weka-062126"

    def test_hf_weka_repo_requires_generic_hf_loader(self):
        raw = self._config(**{
            "custom-dataset-type": "weka_trace",
            "public-dataset": "semianalysis_cc_traces_weka_with_subagents_060826",
            "hf-weka-repo": "semianalysisai/cc-traces-weka-062126",
        })
        del raw["input-file"]

        with pytest.raises(Exception):
            AgenticReplayConfig(**raw)

    def test_rejects_both_input_file_and_public_dataset(self):
        with pytest.raises(Exception):
            AgenticReplayConfig(**self._config(**{
                "public-dataset": "semianalysis_cc_traces_weka_with_subagents_060826",
            }))

    def test_public_dataset_rejected_for_non_weka(self):
        raw = self._config(**{
            "public-dataset": "semianalysis_cc_traces_weka_with_subagents_060826",
        })
        del raw["input-file"]
        with pytest.raises(Exception):
            AgenticReplayConfig(**raw)

    def test_conc_range_or_list_required(self):
        with pytest.raises(Exception):
            AgenticReplayConfig(**self._config(**{"search-space": [{"tp": 1}]}))


# =============================================================================
# Test SingleNodeSearchSpaceEntry
# =============================================================================

class TestSingleNodeSearchSpaceEntry:
    """Tests for SingleNodeSearchSpaceEntry model."""

    def test_valid_with_conc_range(self):
        """Valid entry with conc range should pass (like mi300x config)."""
        entry = SingleNodeSearchSpaceEntry(**{
            "tp": 8,
            "conc-start": 4,
            "conc-end": 64,
        })
        assert entry.tp == 8
        assert entry.conc_start == 4
        assert entry.conc_end == 64

    def test_valid_with_conc_list(self):
        """Valid entry with conc list should pass."""
        entry = SingleNodeSearchSpaceEntry(**{
            "tp": 4,
            "conc-list": [4, 8, 16, 32, 64, 128],
        })
        assert entry.conc_list == [4, 8, 16, 32, 64, 128]

    def test_cannot_have_both_range_and_list(self):
        """Cannot specify both conc range and list."""
        with pytest.raises(Exception) as exc_info:
            SingleNodeSearchSpaceEntry(**{
                "tp": 4,
                "conc-start": 4,
                "conc-end": 64,
                "conc-list": [4, 8, 16],
            })
        assert "Cannot specify both" in str(exc_info.value)

    def test_must_have_range_or_list(self):
        """Must specify either conc range or list."""
        with pytest.raises(Exception) as exc_info:
            SingleNodeSearchSpaceEntry(**{
                "tp": 8,
            })
        assert "Must specify either" in str(exc_info.value)

    def test_conc_start_must_be_lte_conc_end(self):
        """conc-start must be <= conc-end."""
        with pytest.raises(Exception) as exc_info:
            SingleNodeSearchSpaceEntry(**{
                "tp": 8,
                "conc-start": 64,
                "conc-end": 4,
            })
        assert "must be <=" in str(exc_info.value)

    def test_conc_list_values_must_be_positive(self):
        """conc-list values must be > 0."""
        with pytest.raises(Exception) as exc_info:
            SingleNodeSearchSpaceEntry(**{
                "tp": 4,
                "conc-list": [4, 0, 16],
            })
        assert "must be greater than 0" in str(exc_info.value)

    def test_optional_fields_defaults(self):
        """Optional fields should have correct defaults."""
        entry = SingleNodeSearchSpaceEntry(**{
            "tp": 8,
            "conc-list": [4, 8],
        })
        assert entry.ep is None
        assert entry.dp_attn is None
        assert entry.spec_decoding == "none"

    def test_with_ep_and_dp_attn(self):
        """Entry with ep and dp-attn like b200-sglang config."""
        entry = SingleNodeSearchSpaceEntry(**{
            "tp": 4,
            "ep": 4,
            "dp-attn": True,
            "conc-start": 4,
            "conc-end": 128,
        })
        assert entry.ep == 4
        assert entry.dp_attn is True

    def test_with_spec_decoding_mtp(self):
        """Entry with mtp spec decoding."""
        entry = SingleNodeSearchSpaceEntry(**{
            "tp": 8,
            "spec-decoding": "mtp",
            "conc-list": [1, 2, 4],
        })
        assert entry.spec_decoding == "mtp"


# =============================================================================
# Test MultiNodeSearchSpaceEntry
# =============================================================================

class TestMultiNodeSearchSpaceEntry:
    """Tests for MultiNodeSearchSpaceEntry model."""

    def test_valid_with_conc_list(self):
        """Valid multinode search space with list (like gb200 config)."""
        entry = MultiNodeSearchSpaceEntry(**{
            "prefill": {
                "num-worker": 5,
                "tp": 4,
                "ep": 4,
                "dp-attn": True,
                "additional-settings": ["PREFILL_MAX_NUM_TOKENS=8448"],
            },
            "decode": {
                "num-worker": 1,
                "tp": 8,
                "ep": 8,
                "dp-attn": True,
                "additional-settings": ["DECODE_MAX_NUM_TOKENS=256"],
            },
            "conc-list": [2150],
        })
        assert entry.prefill.num_worker == 5
        assert entry.decode.tp == 8

    def test_valid_with_conc_range(self):
        """Valid multinode search space with range."""
        entry = MultiNodeSearchSpaceEntry(**{
            "prefill": {
                "num-worker": 1,
                "tp": 4,
                "ep": 4,
                "dp-attn": False,
            },
            "decode": {
                "num-worker": 4,
                "tp": 8,
                "ep": 8,
                "dp-attn": False,
            },
            "conc-start": 1,
            "conc-end": 64,
        })
        assert entry.conc_start == 1
        assert entry.conc_end == 64

    def test_with_spec_decoding_mtp(self):
        """Multinode entry with mtp spec decoding."""
        entry = MultiNodeSearchSpaceEntry(**{
            "spec-decoding": "mtp",
            "prefill": {
                "num-worker": 1,
                "tp": 4,
                "ep": 4,
                "dp-attn": False,
            },
            "decode": {
                "num-worker": 4,
                "tp": 8,
                "ep": 8,
                "dp-attn": False,
            },
            "conc-list": [1, 2, 4, 8, 16, 36],
        })
        assert entry.spec_decoding == "mtp"

    def test_missing_conc_specification(self):
        """Missing conc specification should fail."""
        with pytest.raises(Exception):
            MultiNodeSearchSpaceEntry(**{
                "prefill": {
                    "num-worker": 2,
                    "tp": 4,
                    "ep": 4,
                    "dp-attn": False,
                },
                "decode": {
                    "num-worker": 2,
                    "tp": 4,
                    "ep": 4,
                    "dp-attn": False,
                },
                # Missing conc specification
            })


# =============================================================================
# Test SeqLenConfig models
# =============================================================================

class TestSeqLenConfigs:
    """Tests for sequence length config models."""

    def test_single_node_seq_len_config_1k1k(self):
        """Valid single node seq len config for 1k/1k."""
        config = SingleNodeSeqLenConfig(**{
            "isl": 1024,
            "osl": 1024,
            "search-space": [
                {"tp": 8, "conc-start": 4, "conc-end": 64}
            ]
        })
        assert config.isl == 1024
        assert config.osl == 1024
        assert len(config.search_space) == 1
        assert config.benchmark_client == ["inferencex_native"]

    def test_single_node_benchmark_client_roundtrip(self):
        """Single-node seq-len configs should accept both benchmark clients."""
        config = SingleNodeSeqLenConfig(**{
            "isl": 1024,
            "osl": 1024,
            "benchmark-client": ["inferencex_native", "aiperf"],
            "search-space": [
                {"tp": 8, "conc-start": 4, "conc-end": 64}
            ]
        })
        assert config.benchmark_client == ["inferencex_native", "aiperf"]
        assert config.model_dump(by_alias=True)["benchmark-client"] == ["inferencex_native", "aiperf"]

    def test_single_node_invalid_benchmark_client(self):
        """Invalid single-node benchmark client values should fail."""
        with pytest.raises(Exception):
            SingleNodeSeqLenConfig(**{
                "isl": 1024,
                "osl": 1024,
                "benchmark-client": ["foo"],
                "search-space": [
                    {"tp": 8, "conc-start": 4, "conc-end": 64}
                ]
            })

    def test_single_node_seq_len_config_8k1k(self):
        """Valid single node seq len config for 8k/1k."""
        config = SingleNodeSeqLenConfig(**{
            "isl": 8192,
            "osl": 1024,
            "search-space": [
                {"tp": 8, "conc-start": 4, "conc-end": 64}
            ]
        })
        assert config.isl == 8192
        assert config.osl == 1024

    def test_multinode_seq_len_config(self):
        """Valid multinode seq len config."""
        config = MultiNodeSeqLenConfig(**{
            "isl": 1024,
            "osl": 1024,
            "search-space": [
                {
                    "prefill": {
                        "num-worker": 5,
                        "tp": 4,
                        "ep": 4,
                        "dp-attn": True,
                    },
                    "decode": {
                        "num-worker": 1,
                        "tp": 8,
                        "ep": 8,
                        "dp-attn": True,
                    },
                    "conc-list": [2150],
                }
            ]
        })
        assert config.isl == 1024
        assert config.osl == 1024
        assert config.benchmark_client == ["inferencex_native"]

    def test_multinode_benchmark_client_roundtrip(self):
        """Multinode seq-len configs should accept both benchmark clients."""
        config = MultiNodeSeqLenConfig(**{
            "isl": 1024,
            "osl": 1024,
            "benchmark-client": ["inferencex_native", "aiperf"],
            "search-space": [
                {
                    "prefill": {
                        "num-worker": 5,
                        "tp": 4,
                        "ep": 4,
                        "dp-attn": True,
                    },
                    "decode": {
                        "num-worker": 1,
                        "tp": 8,
                        "ep": 8,
                        "dp-attn": True,
                    },
                    "conc-list": [2150],
                }
            ]
        })
        assert config.benchmark_client == ["inferencex_native", "aiperf"]
        assert config.model_dump(by_alias=True)["benchmark-client"] == ["inferencex_native", "aiperf"]

    def test_multinode_invalid_benchmark_client(self):
        """Invalid multinode benchmark client values should fail."""
        with pytest.raises(Exception):
            MultiNodeSeqLenConfig(**{
                "isl": 1024,
                "osl": 1024,
                "benchmark-client": ["foo"],
                "search-space": [
                    {
                        "prefill": {
                            "num-worker": 5,
                            "tp": 4,
                            "ep": 4,
                            "dp-attn": True,
                        },
                        "decode": {
                            "num-worker": 1,
                            "tp": 8,
                            "ep": 8,
                            "dp-attn": True,
                        },
                        "conc-list": [2150],
                    }
                ]
            })

class TestAgenticCodingConfig:
    """Tests for agentic coding scenario config benchmark clients."""

    def test_benchmark_client_default(self):
        config = AgenticCodingConfig(**{
            "search-space": [
                {"tp": 8, "conc-start": 4, "conc-end": 4}
            ]
        })
        assert config.benchmark_client == ["inferencex_native"]

    def test_benchmark_client_roundtrip(self):
        config = AgenticCodingConfig(**{
            "benchmark-client": ["inferencex_native", "aiperf"],
            "search-space": [
                {"tp": 8, "conc-start": 4, "conc-end": 4}
            ]
        })
        assert config.benchmark_client == ["inferencex_native", "aiperf"]
        assert config.model_dump(by_alias=True)["benchmark-client"] == ["inferencex_native", "aiperf"]

    def test_invalid_benchmark_client(self):
        with pytest.raises(Exception):
            AgenticCodingConfig(**{
                "benchmark-client": ["foo"],
                "search-space": [
                    {"tp": 8, "conc-start": 4, "conc-end": 4}
                ]
            })


# =============================================================================
# Test MasterConfigEntry models
# =============================================================================

class TestMasterConfigEntries:
    """Tests for master config entry models."""

    def test_single_node_master_config(self, valid_single_node_master_config):
        """Valid single node master config."""
        config = SingleNodeMasterConfigEntry(**valid_single_node_master_config)
        assert config.multinode is False
        assert config.model_prefix == "dsr1"
        assert config.runner == "mi300x"
        assert config.framework == "sglang"

    def test_multinode_master_config(self, valid_multinode_master_config):
        """Valid multinode master config."""
        config = MultiNodeMasterConfigEntry(**valid_multinode_master_config)
        assert config.multinode is True
        assert config.model_prefix == "dsr1"
        assert config.runner == "gb200"
        assert config.disagg is True

    def test_single_node_cannot_have_multinode_true(self, valid_single_node_master_config):
        """Single node config must have multinode=False."""
        valid_single_node_master_config["multinode"] = True
        with pytest.raises(Exception):
            SingleNodeMasterConfigEntry(**valid_single_node_master_config)

    def test_multinode_cannot_have_multinode_false(self, valid_multinode_master_config):
        """Multinode config must have multinode=True."""
        valid_multinode_master_config["multinode"] = False
        with pytest.raises(Exception):
            MultiNodeMasterConfigEntry(**valid_multinode_master_config)

    def test_disagg_default_false(self, valid_single_node_master_config):
        """Disagg should default to False."""
        config = SingleNodeMasterConfigEntry(**valid_single_node_master_config)
        assert config.disagg is False


# =============================================================================
# Test validate_master_config function
# =============================================================================

class TestValidateMasterConfig:
    """Tests for validate_master_config function."""

    def test_valid_single_node_config(self, valid_single_node_master_config):
        """Valid single node config should pass."""
        configs = {"dsr1-fp8-mi300x-sglang": valid_single_node_master_config}
        result = validate_master_config(configs)
        assert result == configs

    def test_valid_single_node_config_with_benchmark_client_opt_in(self, valid_single_node_master_config):
        """Master config validation should allow scenario-level benchmark-client lists."""
        valid_single_node_master_config["scenarios"]["fixed-seq-len"][0]["benchmark-client"] = [
            "inferencex_native",
            "aiperf",
        ]
        configs = {"dsr1-fp8-mi300x-sglang": valid_single_node_master_config}
        result = validate_master_config(configs)
        assert result == configs

    def test_remote_requires_agentic_replay_only(self, valid_single_node_master_config):
        valid_single_node_master_config["remote"] = {"url": "http://remote:8000"}
        configs = {"remote-fixed-seq": valid_single_node_master_config}

        with pytest.raises(ValueError):
            validate_master_config(configs)

    def test_remote_agentic_replay_config(self):
        configs = {
            "remote-replay": {
                "image": "python:3.12-slim",
                "model": "served-model",
                "model-prefix": "served",
                "precision": "fp8",
                "framework": "sglang",
                "runner": "h200-greennode_01",
                "multinode": False,
                "remote": {"url": "http://remote:8000"},
                "scenarios": {
                    "agentic-replay": [{
                        "custom-dataset-type": "weka_trace",
                        "max-model-len": 8192,
                        "search-space": [{"tp": 1, "conc-list": [2]}],
                    }]
                },
            }
        }

        assert validate_master_config(configs) == configs

    def test_valid_multinode_config(self, valid_multinode_master_config):
        """Valid multinode config should pass."""
        configs = {"dsr1-fp4-gb200-dynamo-trt": valid_multinode_master_config}
        result = validate_master_config(configs)
        assert result == configs

    def test_mixed_configs(self, valid_single_node_master_config, valid_multinode_master_config):
        """Mixed single and multinode configs should pass."""
        configs = {
            "dsr1-fp8-mi300x-sglang": valid_single_node_master_config,
            "dsr1-fp4-gb200-dynamo-trt": valid_multinode_master_config,
        }
        result = validate_master_config(configs)
        assert len(result) == 2

    def test_invalid_config_raises_valueerror(self, valid_single_node_master_config):
        """Invalid config should raise ValueError with key name."""
        del valid_single_node_master_config["model"]
        configs = {"broken-config": valid_single_node_master_config}
        with pytest.raises(ValueError) as exc_info:
            validate_master_config(configs)
        assert "broken-config" in str(exc_info.value)
        assert "failed validation" in str(exc_info.value)


# =============================================================================
# Test validate_runner_config function
# =============================================================================

class TestValidateRunnerConfig:
    """Tests for validate_runner_config function."""

    def test_valid_runner_config(self, valid_runner_config):
        """Valid runner config should pass."""
        result = validate_runner_config(valid_runner_config)
        assert result == valid_runner_config

    def test_value_must_be_list(self):
        """Runner config values must be lists."""
        config = {
            "h100": "h100-cr_0",  # Not a list
        }
        with pytest.raises(ValueError) as exc_info:
            validate_runner_config(config)
        assert "must be a list" in str(exc_info.value)

    def test_list_must_contain_strings(self):
        """Runner config lists must contain only strings."""
        config = {
            "h100": ["h100-cr_0", 123],  # Contains non-string
        }
        with pytest.raises(ValueError) as exc_info:
            validate_runner_config(config)
        assert "must contain only strings" in str(exc_info.value)

    def test_list_cannot_be_empty(self):
        """Runner config lists cannot be empty."""
        config = {
            "mi355x": [],
        }
        with pytest.raises(ValueError) as exc_info:
            validate_runner_config(config)
        assert "cannot be an empty list" in str(exc_info.value)

    def test_multiple_runner_types(self, valid_runner_config):
        """Multiple runner types should work."""
        result = validate_runner_config(valid_runner_config)
        assert "h100" in result
        assert "h200" in result
        assert "mi300x" in result
        assert "gb200" in result


# =============================================================================
# Test load_config_files
# =============================================================================

class TestLoadConfigFiles:
    """Tests for load_config_files function."""

    def test_load_single_file_with_validation(self, tmp_path, valid_single_node_master_config):
        """Should load and validate a single config file."""
        config_file = tmp_path / "config.yaml"
        import yaml
        config_file.write_text(yaml.dump({"test-config": valid_single_node_master_config}))
        result = load_config_files([str(config_file)])
        assert "test-config" in result
        assert result["test-config"]["image"] == valid_single_node_master_config["image"]

    def test_load_single_file_without_validation(self, tmp_path):
        """Should load a single config file without validation when validate=False."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
test-config:
  image: test-image
  model: test-model
""")
        result = load_config_files([str(config_file)], validate=False)
        assert "test-config" in result
        assert result["test-config"]["image"] == "test-image"

    def test_load_multiple_files(self, tmp_path):
        """Should merge multiple config files."""
        config1 = tmp_path / "config1.yaml"
        config1.write_text("""
config-one:
  value: 1
""")
        config2 = tmp_path / "config2.yaml"
        config2.write_text("""
config-two:
  value: 2
""")
        result = load_config_files([str(config1), str(config2)], validate=False)
        assert "config-one" in result
        assert "config-two" in result

    def test_duplicate_keys_raise_error(self, tmp_path):
        """Duplicate keys across files should raise error."""
        config1 = tmp_path / "config1.yaml"
        config1.write_text("""
duplicate-key:
  value: 1
""")
        config2 = tmp_path / "config2.yaml"
        config2.write_text("""
duplicate-key:
  value: 2
""")
        with pytest.raises(ValueError) as exc_info:
            load_config_files([str(config1), str(config2)], validate=False)
        assert "Duplicate configuration keys" in str(exc_info.value)

    def test_nonexistent_file_raises_error(self):
        """Nonexistent file should raise error."""
        with pytest.raises(ValueError) as exc_info:
            load_config_files(["nonexistent.yaml"])
        assert "does not exist" in str(exc_info.value)

    def test_validation_runs_by_default(self, tmp_path):
        """Validation should run by default and catch invalid configs."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
invalid-config:
  image: test-image
  # Missing required fields like model, model-prefix, precision, etc.
""")
        with pytest.raises(ValueError) as exc_info:
            load_config_files([str(config_file)])
        assert "failed validation" in str(exc_info.value)


# =============================================================================
# Test load_runner_file
# =============================================================================

class TestLoadRunnerFile:
    """Tests for load_runner_file function."""

    def test_load_runner_file_with_validation(self, tmp_path):
        """Should load and validate runner config file."""
        runner_file = tmp_path / "runners.yaml"
        runner_file.write_text("""
h100:
- h100-node-0
- h100-node-1
""")
        result = load_runner_file(str(runner_file))
        assert "h100" in result
        assert len(result["h100"]) == 2

    def test_load_runner_file_without_validation(self, tmp_path):
        """Should load runner config file without validation when validate=False."""
        runner_file = tmp_path / "runners.yaml"
        runner_file.write_text("""
h100:
- h100-node-0
- h100-node-1
""")
        result = load_runner_file(str(runner_file), validate=False)
        assert "h100" in result
        assert len(result["h100"]) == 2

    def test_nonexistent_runner_file(self):
        """Nonexistent runner file should raise error."""
        with pytest.raises(ValueError) as exc_info:
            load_runner_file("nonexistent.yaml")
        assert "does not exist" in str(exc_info.value)

    def test_validation_runs_by_default(self, tmp_path):
        """Validation should run by default and catch invalid configs."""
        runner_file = tmp_path / "runners.yaml"
        runner_file.write_text("""
h100: not-a-list
""")
        with pytest.raises(ValueError) as exc_info:
            load_runner_file(str(runner_file))
        assert "must be a list" in str(exc_info.value)

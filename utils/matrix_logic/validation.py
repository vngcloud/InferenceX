from pydantic import BaseModel, Field, ValidationError, ConfigDict, model_validator
from typing import List, Optional, Union, Literal
from enum import Enum

import pprint
import yaml

"""
    The below class defines the field names expected to be present in the JSON entries
    for both single-node and multi-node configurations.
"""


class Fields(Enum):
    # Field name constants
    # Top-level config fields
    IMAGE = 'image'
    MODEL = 'model'
    MODEL_PREFIX = 'model-prefix'
    PRECISION = 'precision'
    FRAMEWORK = 'framework'
    BENCHMARK_CLIENT = 'benchmark-client'
    RUNNER = 'runner'
    SCENARIOS = 'scenarios'
    MULTINODE = 'multinode'

    # Scenario type keys
    FIXED_SEQ_LEN = 'fixed-seq-len'
    AGENTIC_CODING = 'agentic-coding'
    AGENTIC_REPLAY = 'agentic-replay'

    # Seq-len-config fields
    ISL = 'isl'
    OSL = 'osl'
    SEARCH_SPACE = 'search-space'

    # Search-space/benchmark fields
    TP = 'tp'
    CONC_START = 'conc-start'
    CONC_END = 'conc-end'
    CONC_LIST = 'conc-list'
    EP = 'ep'
    DP_ATTN = 'dp-attn'
    MAX_NUM_BATCHED_TOKENS = 'max-num-batched-tokens'
    NUM_SPECULATIVE_TOKENS = 'num-speculative-tokens'

    # Multinode-specific fields (when MULTINODE = true)
    SPEC_DECODING = 'spec-decoding'
    PREFILL = 'prefill'
    DECODE = 'decode'
    NUM_WORKER = 'num-worker'
    BATCH_SIZE = 'batch-size'
    MAX_NUM_TOKENS = 'max-num-tokens'
    ADDITIONAL_SETTINGS = 'additional-settings'

    # Agentic coding fields
    OFFLOADING = 'offloading'
    DURATION = 'duration'

    # Agentic replay fields (mooncake_trace on official AIPerf)
    INPUT_FILE = 'input-file'
    CUSTOM_DATASET_TYPE = 'custom-dataset-type'
    # Optional HF tokenizer id for AIPerf token counting; defaults to the served
    # model when unset (the standard flow). Set only when served-name != HF id.
    TOKENIZER = 'tokenizer'
    # Agentic-replay Mode 1 (capacity sweep) fields
    NO_FIXED_SCHEDULE = 'no-fixed-schedule'
    NUM_WARMUP_SESSIONS = 'num-warmup-sessions'
    REQUEST_COUNT = 'request-count'
    STRIP_TRACE_DELAYS = 'strip-trace-delays'

    # Matrix entry fields
    CONC = 'conc'
    MAX_MODEL_LEN = 'max-model-len'
    EXP_NAME = 'exp-name'
    DISAGG = 'disagg'
    SCENARIO_TYPE = 'scenario-type'

    # Eval
    RUN_EVAL = 'run-eval'
    EVAL_ONLY = 'eval-only'
    EVAL_CONC = 'eval-conc'


"""
    Below is the validation logic for the OUTPUT of utils/matrix_logic/generate_sweep_configs.py, i.e., 
    the input to the actual workflow files. The validation enforces a strict set of rules on the structure
    of the generated matrix entries to ensure correctness before proceeding with benchmarking. This ensures
    that no validation has to happen in the workflow itself, i.e., at runtime, it is assumed that all inputs
    are valid. Threfore, there should not be any default values set in these Pydantic models. Any missing value
    should raise a validation error.
"""


class SingleNodeMatrixEntry(BaseModel):
    """Pydantic model for validating single node matrix entry structure.
    This validates the input that should be expected to .github/workflows/benchmark-tmpl.yml"""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    benchmark_client: Literal["inferencex_native", "aiperf"] = Field(
        default="inferencex_native", alias=Fields.BENCHMARK_CLIENT.value
    )
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        alias=Fields.SPEC_DECODING.value
    )
    num_speculative_tokens: Optional[int] = Field(
        default=None, alias=Fields.NUM_SPECULATIVE_TOKENS.value
    )
    runner: str
    isl: int
    osl: int
    tp: int
    ep: int
    dp_attn: bool = Field(alias=Fields.DP_ATTN.value)
    conc: Union[int, List[int]]
    max_model_len: int = Field(alias=Fields.MAX_MODEL_LEN.value)
    max_num_batched_tokens: Optional[int] = Field(
        default=None, alias=Fields.MAX_NUM_BATCHED_TOKENS.value
    )
    exp_name: str = Field(alias=Fields.EXP_NAME.value)
    disagg: bool
    run_eval: bool = Field(alias=Fields.RUN_EVAL.value)
    eval_only: bool = Field(alias=Fields.EVAL_ONLY.value, default=False)


class WorkerConfig(BaseModel):
    """Pydantic model for validating worker configuration in multinode entries."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    num_worker: int = Field(alias=Fields.NUM_WORKER.value)
    tp: int
    ep: int
    dp_attn: bool = Field(alias=Fields.DP_ATTN.value)
    additional_settings: Optional[List[str]] = Field(
        default=[], alias=Fields.ADDITIONAL_SETTINGS.value)


class MultiNodeMatrixEntry(BaseModel):
    """Pydantic model for validating multinode matrix entry structure.
    This validates the input that should be expected to .github/workflows/benchmark-multinode-tmpl.yml"""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    benchmark_client: Literal["inferencex_native", "aiperf"] = Field(
        default="inferencex_native", alias=Fields.BENCHMARK_CLIENT.value
    )
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        alias=Fields.SPEC_DECODING.value
    )
    runner: str
    isl: int
    osl: int
    prefill: WorkerConfig
    decode: WorkerConfig
    conc: List[int]
    max_model_len: int = Field(alias=Fields.MAX_MODEL_LEN.value)
    exp_name: str = Field(alias=Fields.EXP_NAME.value)
    disagg: bool
    run_eval: bool = Field(alias=Fields.RUN_EVAL.value)
    eval_only: bool = Field(alias=Fields.EVAL_ONLY.value, default=False)
    eval_conc: Optional[int] = Field(default=None, alias=Fields.EVAL_CONC.value)


class SingleNodeAgenticMatrixEntry(BaseModel):
    """Pydantic model for validating single-node agentic coding matrix entries."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    benchmark_client: Literal["inferencex_native", "aiperf"] = Field(
        default="inferencex_native", alias=Fields.BENCHMARK_CLIENT.value
    )
    runner: str
    tp: int
    ep: int
    dp_attn: bool = Field(alias=Fields.DP_ATTN.value)
    conc: int
    offloading: Literal["none", "cpu", "ssd"] = Field(alias=Fields.OFFLOADING.value)
    duration: int = Field(default=1800, alias=Fields.DURATION.value)
    exp_name: str = Field(alias=Fields.EXP_NAME.value)
    scenario_type: str = Field(alias=Fields.SCENARIO_TYPE.value)


class MultiNodeAgenticMatrixEntry(BaseModel):
    """Pydantic model for validating multinode agentic coding matrix entries."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    benchmark_client: Literal["inferencex_native", "aiperf"] = Field(
        default="inferencex_native", alias=Fields.BENCHMARK_CLIENT.value
    )
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        alias=Fields.SPEC_DECODING.value
    )
    runner: str
    prefill: WorkerConfig
    decode: WorkerConfig
    conc: int
    duration: int = Field(default=1800, alias=Fields.DURATION.value)
    exp_name: str = Field(alias=Fields.EXP_NAME.value)
    disagg: bool
    scenario_type: str = Field(alias=Fields.SCENARIO_TYPE.value)


AgenticMatrixEntry = Union[SingleNodeAgenticMatrixEntry, MultiNodeAgenticMatrixEntry]


class SingleNodeAgenticReplayMatrixEntry(BaseModel):
    """Pydantic model for validating single-node agentic-replay matrix entries.

    Agentic-replay drives a recorded mooncake_trace JSONL through official AIPerf
    (the ``aiperf`` benchmark client) and rides the standard ``process_result.py``
    aggregation. ``isl``/``osl`` are placeholders required by downstream env
    checks; the trace defines the real per-request lengths."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    benchmark_client: Literal["aiperf"] = Field(
        default="aiperf", alias=Fields.BENCHMARK_CLIENT.value
    )
    runner: str
    tp: int
    ep: int
    dp_attn: bool = Field(alias=Fields.DP_ATTN.value)
    conc: int
    isl: int
    osl: int
    max_model_len: int = Field(alias=Fields.MAX_MODEL_LEN.value)
    input_file: str = Field(alias=Fields.INPUT_FILE.value)
    custom_dataset_type: str = Field(alias=Fields.CUSTOM_DATASET_TYPE.value)
    duration: int = Field(default=1800, alias=Fields.DURATION.value)
    # Mode 1 (capacity sweep) controls; defaults preserve single-replay behavior.
    no_fixed_schedule: bool = Field(
        default=False, alias=Fields.NO_FIXED_SCHEDULE.value)
    num_warmup_sessions: Optional[int] = Field(
        default=None, alias=Fields.NUM_WARMUP_SESSIONS.value)
    request_count: Optional[int] = Field(
        default=None, alias=Fields.REQUEST_COUNT.value)
    strip_trace_delays: bool = Field(
        default=False, alias=Fields.STRIP_TRACE_DELAYS.value)
    tokenizer: Optional[str] = Field(
        default=None, alias=Fields.TOKENIZER.value)
    exp_name: str = Field(alias=Fields.EXP_NAME.value)
    disagg: bool
    scenario_type: str = Field(alias=Fields.SCENARIO_TYPE.value)


def validate_agentic_matrix_entry(entry: dict) -> dict:
    """Validate that an agentic matrix entry matches the expected structure."""
    try:
        if Fields.PREFILL.value in entry:
            MultiNodeAgenticMatrixEntry(**entry)
        else:
            SingleNodeAgenticMatrixEntry(**entry)
    except ValidationError as e:
        raise ValueError(
            f"The following parsed agentic matrix entry failed validation:\n{pprint.pformat(entry)}\n{e}")
    return entry


def validate_agentic_replay_matrix_entry(entry: dict) -> dict:
    """Validate that an agentic-replay matrix entry matches the expected structure."""
    try:
        SingleNodeAgenticReplayMatrixEntry(**entry)
    except ValidationError as e:
        raise ValueError(
            f"The following parsed agentic-replay matrix entry failed validation:\n{pprint.pformat(entry)}\n{e}")
    return entry


def validate_matrix_entry(entry: dict, is_multinode: bool) -> dict:
    """Validate that matrix_values entries match the expected structure.

    Raises ValueError if any entry fails validation.
    Returns the original list if all entries are valid.
    """
    try:
        if is_multinode:
            MultiNodeMatrixEntry(**entry)
        else:
            SingleNodeMatrixEntry(**entry)
    except ValidationError as e:
        raise ValueError(
            f"The following parsed matrix entry failed validation:\n{pprint.pformat(entry)}\n{e}")
    return entry


"""
    Below is the validation logic for the INPUT to utils/matrix_logic/generate_sweep_configs.py, i.e., 
    the master configuration files found in .github/configs. The validation enforces a strict set of 
    rules on the structure of the master configuration files to ensure correctness before proceeding 
    with matrix generation.
"""


def _validate_conc_fields(self):
    """Ensure either (conc_start AND conc_end) OR conc_list is provided, but not both."""
    has_range = self.conc_start is not None and self.conc_end is not None
    has_list = self.conc_list is not None and len(self.conc_list) > 0

    if has_range and has_list:
        raise ValueError(
            f"Cannot specify both '{Fields.CONC_LIST.value}' list and "
            f"'{Fields.CONC_START.value}'/'{Fields.CONC_END.value}'. "
            "Use either a list or a range, not both."
        )

    if not has_range and not has_list:
        raise ValueError(
            f"Must specify either '{Fields.CONC_LIST.value}' list or both "
            f"'{Fields.CONC_START.value}' and '{Fields.CONC_END.value}'."
        )

    if has_range:
        if self.conc_start is None or self.conc_end is None:
            raise ValueError(
                f"Both '{Fields.CONC_START.value}' and '{Fields.CONC_END.value}' "
                "must be provided together."
            )

        if self.conc_start > self.conc_end:
            raise ValueError(
                f"'{Fields.CONC_START.value}' ({self.conc_start}) must be <= "
                f"'{Fields.CONC_END.value}' ({self.conc_end})."
            )

    if has_list:
        if not all(x > 0 for x in self.conc_list):
            raise ValueError(
                f"Input '{Fields.CONC_LIST.value}' entries must be greater than 0."
            )

    return self


class SingleNodeSearchSpaceEntry(BaseModel):
    """Single node search space configuration."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    tp: int
    ep: Optional[int] = None
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        default="none", alias=Fields.SPEC_DECODING.value)
    # N for speculative decoding (num_speculative_tokens / drafter depth).
    # Only meaningful when spec_decoding != "none". None means the bench
    # script picks its own default (typically the model's native MTP depth).
    num_speculative_tokens: Optional[int] = Field(
        default=None, alias=Fields.NUM_SPECULATIVE_TOKENS.value)
    dp_attn: Optional[bool] = Field(
        default=None, alias=Fields.DP_ATTN.value)
    # Chunked-prefill ceiling. None means the engine default is used.
    max_num_batched_tokens: Optional[int] = Field(
        default=None, alias=Fields.MAX_NUM_BATCHED_TOKENS.value)
    conc_start: Optional[int] = Field(
        default=None, alias=Fields.CONC_START.value)
    conc_end: Optional[int] = Field(
        default=None, alias=Fields.CONC_END.value)
    conc_list: Optional[List[int]] = Field(
        default=None, alias=Fields.CONC_LIST.value)

    @model_validator(mode='after')
    def validate_conc_fields(self):
        return _validate_conc_fields(self)

    @model_validator(mode='after')
    def validate_spec_consistency(self):
        if self.num_speculative_tokens is not None and self.spec_decoding == "none":
            raise ValueError(
                f"'{Fields.NUM_SPECULATIVE_TOKENS.value}' is only meaningful "
                f"when '{Fields.SPEC_DECODING.value}' is not 'none'."
            )
        return self


class MultiNodeSearchSpaceEntry(BaseModel):
    """Multinode search space configuration."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        default="none", alias=Fields.SPEC_DECODING.value)
    prefill: WorkerConfig
    decode: WorkerConfig
    conc_start: Optional[int] = Field(
        default=None, alias=Fields.CONC_START.value)
    conc_end: Optional[int] = Field(
        default=None, alias=Fields.CONC_END.value)
    conc_list: Optional[List[int]] = Field(
        default=None, alias=Fields.CONC_LIST.value)

    @model_validator(mode='after')
    def validate_conc_fields(self):
        return _validate_conc_fields(self)


class SingleNodeSeqLenConfig(BaseModel):
    """Single node sequence length configuration."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    isl: int
    osl: int
    benchmark_client: List[Literal["inferencex_native", "aiperf"]] = Field(
        default=["inferencex_native"], alias=Fields.BENCHMARK_CLIENT.value)
    search_space: List[SingleNodeSearchSpaceEntry] = Field(
        alias=Fields.SEARCH_SPACE.value)


class MultiNodeSeqLenConfig(BaseModel):
    """Multinode sequence length configuration."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    isl: int
    osl: int
    benchmark_client: List[Literal["inferencex_native", "aiperf"]] = Field(
        default=["inferencex_native"], alias=Fields.BENCHMARK_CLIENT.value)
    search_space: List[MultiNodeSearchSpaceEntry] = Field(
        alias=Fields.SEARCH_SPACE.value)


class AgenticCodingSearchSpaceEntry(BaseModel):
    """Agentic coding search space configuration."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    tp: Optional[int] = None
    ep: Optional[int] = None
    dp_attn: Optional[bool] = Field(default=None, alias=Fields.DP_ATTN.value)
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        default="none", alias=Fields.SPEC_DECODING.value)
    prefill: Optional[WorkerConfig] = None
    decode: Optional[WorkerConfig] = None
    offloading: Literal["none", "cpu", "ssd"] = Field(default="none", alias=Fields.OFFLOADING.value)
    conc_start: Optional[int] = Field(default=None, alias=Fields.CONC_START.value)
    conc_end: Optional[int] = Field(default=None, alias=Fields.CONC_END.value)
    conc_list: Optional[List[int]] = Field(default=None, alias=Fields.CONC_LIST.value)

    @model_validator(mode='after')
    def validate_conc_fields(self):
        return _validate_conc_fields(self)

    @model_validator(mode='after')
    def validate_topology_fields(self):
        has_single_node = self.tp is not None
        has_any_multinode_field = self.prefill is not None or self.decode is not None
        has_complete_multinode = self.prefill is not None and self.decode is not None
        if has_single_node:
            valid = not has_any_multinode_field
        else:
            valid = has_complete_multinode
        if not valid:
            raise ValueError("Agentic search-space entries must specify either tp or both prefill and decode")
        return self


class AgenticCodingConfig(BaseModel):
    """Agentic coding scenario configuration for trace replay benchmarks."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    search_space: List[AgenticCodingSearchSpaceEntry] = Field(alias=Fields.SEARCH_SPACE.value)
    benchmark_client: List[Literal["inferencex_native", "aiperf"]] = Field(
        default=["inferencex_native"], alias=Fields.BENCHMARK_CLIENT.value)
    duration: int = Field(default=1800, alias=Fields.DURATION.value)


class AgenticReplaySearchSpaceEntry(BaseModel):
    """Single-node agentic-replay search space configuration."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    tp: int
    ep: Optional[int] = None
    dp_attn: Optional[bool] = Field(default=None, alias=Fields.DP_ATTN.value)
    conc_start: Optional[int] = Field(default=None, alias=Fields.CONC_START.value)
    conc_end: Optional[int] = Field(default=None, alias=Fields.CONC_END.value)
    conc_list: Optional[List[int]] = Field(default=None, alias=Fields.CONC_LIST.value)

    @model_validator(mode='after')
    def validate_conc_fields(self):
        return _validate_conc_fields(self)


class AgenticReplayConfig(BaseModel):
    """Single-node agentic-replay scenario: a recorded trace replayed once
    through official AIPerf. ``input-file`` is a repo-relative mooncake_trace
    JSONL; the run is bounded by the record count, not a duration sweep."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    input_file: str = Field(alias=Fields.INPUT_FILE.value)
    custom_dataset_type: str = Field(alias=Fields.CUSTOM_DATASET_TYPE.value)
    max_model_len: int = Field(alias=Fields.MAX_MODEL_LEN.value)
    benchmark_client: List[Literal["aiperf"]] = Field(
        default=["aiperf"], alias=Fields.BENCHMARK_CLIENT.value)
    duration: int = Field(default=1800, alias=Fields.DURATION.value)
    # Mode 1 (capacity sweep): drive the trace by --concurrency back-pressure
    # rather than replaying it once on a fixed schedule. Defaults preserve the
    # original single-replay behavior.
    no_fixed_schedule: bool = Field(
        default=False, alias=Fields.NO_FIXED_SCHEDULE.value)
    num_warmup_sessions: Optional[int] = Field(
        default=None, alias=Fields.NUM_WARMUP_SESSIONS.value)
    request_count: Optional[int] = Field(
        default=None, alias=Fields.REQUEST_COUNT.value)
    strip_trace_delays: bool = Field(
        default=False, alias=Fields.STRIP_TRACE_DELAYS.value)
    tokenizer: Optional[str] = Field(
        default=None, alias=Fields.TOKENIZER.value)
    search_space: List[AgenticReplaySearchSpaceEntry] = Field(
        alias=Fields.SEARCH_SPACE.value)

    @model_validator(mode='after')
    def validate_request_count_vs_conc(self):
        """AIPerf requires request-count >= concurrency. When an explicit
        request-count is set, it must cover the largest swept concurrency."""
        if self.request_count is None:
            return self
        max_conc = 0
        for entry in self.search_space:
            if entry.conc_list:
                max_conc = max(max_conc, max(entry.conc_list))
            if entry.conc_end is not None:
                max_conc = max(max_conc, entry.conc_end)
        if max_conc and self.request_count < max_conc:
            raise ValueError(
                f"request-count ({self.request_count}) must be >= the largest "
                f"swept concurrency ({max_conc}); AIPerf rejects "
                "request-count < concurrency.")
        return self


class SingleNodeScenarios(BaseModel):
    """Scenarios wrapper for single-node configs."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    fixed_seq_len: Optional[List[SingleNodeSeqLenConfig]] = Field(
        default=None, alias=Fields.FIXED_SEQ_LEN.value)
    agentic_coding: Optional[List[AgenticCodingConfig]] = Field(
        default=None, alias=Fields.AGENTIC_CODING.value)
    agentic_replay: Optional[List[AgenticReplayConfig]] = Field(
        default=None, alias=Fields.AGENTIC_REPLAY.value)

    @model_validator(mode='after')
    def at_least_one_scenario(self):
        if not self.fixed_seq_len and not self.agentic_coding and not self.agentic_replay:
            raise ValueError("At least one scenario type must be specified")
        return self


class MultiNodeScenarios(BaseModel):
    """Scenarios wrapper for multinode configs."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    fixed_seq_len: Optional[List[MultiNodeSeqLenConfig]] = Field(
        default=None, alias=Fields.FIXED_SEQ_LEN.value)
    agentic_coding: Optional[List[AgenticCodingConfig]] = Field(
        default=None, alias=Fields.AGENTIC_CODING.value)

    @model_validator(mode='after')
    def at_least_one_scenario(self):
        if not self.fixed_seq_len and not self.agentic_coding:
            raise ValueError("At least one scenario type must be specified")
        return self


class SingleNodeMasterConfigEntry(BaseModel):
    """Top-level single node master configuration entry."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    runner: str
    multinode: Literal[False]
    disagg: bool = Field(default=False)
    scenarios: SingleNodeScenarios


class MultiNodeMasterConfigEntry(BaseModel):
    """Top-level multinode master configuration entry."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    runner: str
    multinode: Literal[True]
    disagg: bool = Field(default=False)
    scenarios: MultiNodeScenarios


def validate_master_config(master_configs: dict) -> List[dict]:
    """Validate input master configuration structure."""
    for key, entry in master_configs.items():
        is_multinode = entry.get('multinode', False)

        try:
            if is_multinode:
                MultiNodeMasterConfigEntry(**entry)
            else:
                SingleNodeMasterConfigEntry(**entry)
        except ValidationError as e:
            raise ValueError(
                f"Master config entry '{key}' failed validation:\n{e}")
    return master_configs

# Runner Config Validation


def validate_runner_config(runner_configs: dict) -> List[dict]:
    """Validate input master configuration structure."""
    for key, value in runner_configs.items():
        if not isinstance(value, list):
            raise ValueError(
                f"Runner config entry '{key}' must be a list, got {type(value).__name__}")

        if not all(isinstance(item, str) for item in value):
            raise ValueError(
                f"Runner config entry '{key}' must contain only strings")

        if not value:
            raise ValueError(
                f"Runner config entry '{key}' cannot be an empty list")

    return runner_configs


"""
    Below is the validation logic for the changelog entries found in perf-changelog.yaml.
    This ensures that the changelog entries conform to the expected structure before
    proceeding with processing.
"""


class ChangelogEntry(BaseModel):
    """Pydantic model for validating changelog entry structure."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    config_keys: list[str] = Field(alias="config-keys", min_length=1)
    description: list[str] = Field(min_length=1)
    pr_link: str = Field(alias="pr-link")
    evals_only: bool = Field(alias="evals-only", default=False)
    scenario_type: Optional[List[str]] = Field(
        alias="scenario-type", default=None,
        description="Restrict to specific scenario types (e.g., ['fixed-seq-len', 'agentic-coding'])"
    )


class ChangelogMetadata(BaseModel):
    """Pydantic model for validating changelog metadata structure."""
    model_config = ConfigDict(extra="forbid")

    base_ref: str
    head_ref: str
    entries: list[ChangelogEntry]


class ChangelogMatrixEntry(BaseModel):
    """Pydantic model for validating final changelog matrix entry structure.
    This imposes a strict contract on the output of process_changelog.py, dictated by
    the expected input to the run-sweep.yml workflow file.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    single_node: dict[str, list[Union[SingleNodeMatrixEntry, SingleNodeAgenticMatrixEntry, SingleNodeAgenticReplayMatrixEntry]]
                      ] = Field(default_factory=dict)
    multi_node: dict[str, list[Union[MultiNodeMatrixEntry, MultiNodeAgenticMatrixEntry]]
                     ] = Field(default_factory=dict)
    evals: list[SingleNodeMatrixEntry] = Field(default_factory=list)
    multinode_evals: list[MultiNodeMatrixEntry] = Field(default_factory=list)
    changelog_metadata: ChangelogMetadata


# =============================================================================
# File Loading Functions
# =============================================================================


def load_config_files(config_files: List[str], validate: bool = True) -> dict:
    """Load and merge configuration files.

    Args:
        config_files: List of paths to YAML configuration files.
        validate: If True, run validate_master_config on loaded data. Defaults to True.

    Returns:
        Merged configuration dictionary.

    Raises:
        ValueError: If file doesn't exist, isn't a dict, or has duplicate keys.
    """
    all_config_data = {}
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                assert isinstance(
                    config_data, dict), f"Config file '{config_file}' must contain a dictionary"

                # Don't allow '*' wildcard in master config keys as we need to reserve these
                # for expansion in process_changelog.py
                for key in config_data.keys():
                    if "*" in key:
                        raise ValueError(
                            f" Wildcard '*' is not allowed in master config keys: '{key}'")

                # Check for duplicate keys
                duplicate_keys = set(all_config_data.keys()) & set(
                    config_data.keys())
                if duplicate_keys:
                    raise ValueError(
                        f"Duplicate configuration keys found in '{config_file}': {', '.join(sorted(duplicate_keys))}"
                    )

                all_config_data.update(config_data)
        except FileNotFoundError:
            raise ValueError(f"Input file '{config_file}' does not exist.")

    if validate:
        validate_master_config(all_config_data)

    return all_config_data


def load_runner_file(runner_file: str, validate: bool = True) -> dict:
    """Load runner configuration file.

    Args:
        runner_file: Path to the runner YAML configuration file.
        validate: If True, run validate_runner_config on loaded data. Defaults to True.

    Returns:
        Runner configuration dictionary.

    Raises:
        ValueError: If file doesn't exist or fails validation.
    """
    try:
        with open(runner_file, 'r') as f:
            runner_config = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(
            f"Runner config file '{runner_file}' does not exist.")

    if validate:
        validate_runner_config(runner_config)

    return runner_config

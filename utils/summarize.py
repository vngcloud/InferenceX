import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional
from tabulate import tabulate

# Header constants
MODEL = "Model"
SERVED_MODEL = "Served Model"
HARDWARE = "Hardware"
FRAMEWORK = "Framework"
PRECISION = "Precision"
ISL = "ISL"
OSL = "OSL"
TP = "TP"
EP = "EP"
DP_ATTENTION = "DP Attention"
CONC = "Conc"
# Latency columns (TTFT / TPOT / Intvty / E2EL) are generated for the full
# mean + p50/p75/p90/p95/p99 distribution by the _*_headers/_*_cells helpers below.
TPUT_PER_GPU = "TPUT per GPU"
OUTPUT_TPUT_PER_GPU = "Output TPUT per GPU"
INPUT_TPUT_PER_GPU = "Input TPUT per GPU"
# Two tokens/Watt conventions (see process_result.py): total counts input+output
# tokens (prefill-dominated, reads high); output counts decoded tokens only.
TOK_PER_WATT_TOTAL = "Token/Watt total (tok/s/W)"
TOK_PER_WATT_OUTPUT = "Token/Watt output (tok/s/W)"
POWER_MEAN = "Power Mean (W)"
PREFILL_TP = "Prefill TP"
PREFILL_EP = "Prefill EP"
PREFILL_DP_ATTN = "Prefill DP Attn"
PREFILL_WORKERS = "Prefill Workers"
PREFILL_GPUS = "Prefill GPUs"
DECODE_TP = "Decode TP"
DECODE_EP = "Decode EP"
DECODE_DP_ATTN = "Decode DP Attn"
DECODE_WORKERS = "Decode Workers"
DECODE_GPUS = "Decode GPUs"

# Eval constants
TASK = "Task"
SCORE = "Score"
EM_STRICT = "EM Strict"
EM_FLEXIBLE = "EM Flexible"
N_EFF = "N (eff)"
SPEC_DECODING = "Spec Decode"

# Latency percentiles surfaced in the summary tables (mean is always shown too).
LATENCY_PCTLS = ("p50", "p75", "p90", "p95", "p99")


def _ms_headers(label: str) -> list:
    """mean + percentile headers for a millisecond latency metric (e.g. TTFT)."""
    return [f"{label} Mean (ms)"] + [f"{label} {p.upper()} (ms)" for p in LATENCY_PCTLS]


def _ms_cells(r: dict, key: str) -> list:
    """mean + percentile cells; values are stored in seconds, rendered as ms."""
    return [f"{r.get(f'mean_{key}', 0) * 1000:.4f}"] + [
        f"{r.get(f'{p}_{key}', 0) * 1000:.4f}" for p in LATENCY_PCTLS
    ]


def _sec_headers(label: str) -> list:
    """mean + percentile headers for a second-scale latency metric (e.g. E2EL)."""
    return [f"{label} Mean (s)"] + [f"{label} {p.upper()} (s)" for p in LATENCY_PCTLS]


def _sec_cells(r: dict, key: str) -> list:
    return [f"{r.get(f'mean_{key}', 0):.4f}"] + [
        f"{r.get(f'{p}_{key}', 0):.4f}" for p in LATENCY_PCTLS
    ]


def _intvty_headers() -> list:
    """Interactivity (tok/s/user) is 1/TPOT, so each column tracks a TPOT pctl."""
    return ["Intvty Mean (tok/s/user)"] + [
        f"Intvty at {p.upper()} TPOT (tok/s/user)" for p in LATENCY_PCTLS
    ]


def _intvty_cells(r: dict) -> list:
    return [f"{r.get('mean_intvty', 0):.4f}"] + [
        f"{r.get(f'{p}_intvty', 0):.4f}" for p in LATENCY_PCTLS
    ]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file and return dict, or None on error."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize.py <results_dir>")
        sys.exit(1)

    results = []
    results_dir = Path(sys.argv[1])
    for result_path in results_dir.rglob('*.json'):
        result = load_json(result_path)
        if result and 'is_multinode' in result:
            results.append(result)

    single_node_results = [r for r in results if not r['is_multinode'] and r.get('scenario_type') != 'agentic-coding']
    multinode_results = [r for r in results if r['is_multinode'] and r.get('scenario_type') != 'agentic-coding']
    agentic_results = [r for r in results if r.get('scenario_type') == 'agentic-coding']

    # Single-node and multi-node results have different fields and therefore need to be printed separately
    if single_node_results:
        single_node_results.sort(key=lambda r: (
            r['infmax_model_prefix'], r['hw'], r['framework'], r['precision'], r['isl'], r['osl'], r['tp'], r['ep'], r['conc']))

        single_node_headers = [
            MODEL, SERVED_MODEL, HARDWARE, FRAMEWORK, PRECISION, ISL, OSL, TP, EP, DP_ATTENTION,
            CONC,
            *_ms_headers("TTFT"),
            *_ms_headers("TPOT"),
            *_intvty_headers(),
            *_sec_headers("E2EL"),
            TPUT_PER_GPU, OUTPUT_TPUT_PER_GPU, INPUT_TPUT_PER_GPU,
            TOK_PER_WATT_TOTAL, TOK_PER_WATT_OUTPUT, POWER_MEAN
        ]

        single_node_rows = [
            [
                r['infmax_model_prefix'],
                r['model'],
                r['hw'].upper(),
                r['framework'].upper(),
                r['precision'].upper(),
                r['isl'],
                r['osl'],
                r['tp'],
                r['ep'],
                r['dp_attention'],
                r['conc'],
                *_ms_cells(r, "ttft"),
                *_ms_cells(r, "tpot"),
                *_intvty_cells(r),
                *_sec_cells(r, "e2el"),
                f"{r['tput_per_gpu']:.4f}",
                f"{r['output_tput_per_gpu']:.4f}",
                f"{r['input_tput_per_gpu']:.4f}",
                f"{r.get('tok_per_watt_total') or 0:.4f}",
                f"{r.get('tok_per_watt_output') or 0:.4f}",
                f"{r.get('mean_power_w') or 0:.2f}",
            ]
            for r in single_node_results
        ]

        print("## Single-Node Results\n")
        print("Only [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) repo contains the Official InferenceX™ result, all other forks & repos are Unofficial. The benchmark setup & quality of machines/clouds in unofficial repos may be differ leading to subpar benchmarking. Unofficial must be explicitly labelled as Unofficial. Forks may not remove this disclaimer.\n")
        print(tabulate(single_node_rows, headers=single_node_headers, tablefmt="github"))
        print("\n")

    if multinode_results:
        multinode_results.sort(key=lambda r: (r['infmax_model_prefix'], r['hw'], r['framework'], r['precision'], r['isl'],
                            r['osl'], r['prefill_tp'], r['prefill_ep'], r['decode_tp'], r['decode_ep'], r['conc']))

        multinode_headers = [
            MODEL, SERVED_MODEL, HARDWARE, FRAMEWORK, PRECISION, ISL, OSL,
            PREFILL_TP, PREFILL_EP, PREFILL_DP_ATTN, PREFILL_WORKERS, PREFILL_GPUS,
            DECODE_TP, DECODE_EP, DECODE_DP_ATTN, DECODE_WORKERS, DECODE_GPUS,
            CONC,
            *_ms_headers("TTFT"),
            *_ms_headers("TPOT"),
            *_intvty_headers(),
            *_sec_headers("E2EL"),
            TPUT_PER_GPU, OUTPUT_TPUT_PER_GPU, INPUT_TPUT_PER_GPU,
            TOK_PER_WATT_TOTAL, TOK_PER_WATT_OUTPUT, POWER_MEAN
        ]

        multinode_rows = [
            [
                r['infmax_model_prefix'],
                r['model'],
                r['hw'].upper(),
                r['framework'].upper(),
                r['precision'].upper(),
                r['isl'],
                r['osl'],
                r['prefill_tp'],
                r['prefill_ep'],
                r['prefill_dp_attention'],
                r['prefill_num_workers'],
                r['num_prefill_gpu'],
                r['decode_tp'],
                r['decode_ep'],
                r['decode_dp_attention'],
                r['decode_num_workers'],
                r['num_decode_gpu'],
                r['conc'],
                *_ms_cells(r, "ttft"),
                *_ms_cells(r, "tpot"),
                *_intvty_cells(r),
                *_sec_cells(r, "e2el"),
                f"{r['tput_per_gpu']:.4f}",
                f"{r['output_tput_per_gpu']:.4f}",
                f"{r['input_tput_per_gpu']:.4f}",
                f"{r.get('tok_per_watt_total') or 0:.4f}",
                f"{r.get('tok_per_watt_output') or 0:.4f}",
                f"{r.get('mean_power_w') or 0:.2f}",
            ]
            for r in multinode_results
        ]

        print("## Multi-Node Results\n")
        print("Only [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) repo contains the Official InferenceX™ result, all other forks & repos are Unofficial. The benchmark setup & quality of machines/clouds in unofficial repos may be differ leading to subpar benchmarking. Unofficial must be explicitly labelled as Unofficial. Forks may not remove this disclaimer.\n")
        print(tabulate(multinode_rows, headers=multinode_headers, tablefmt="github"))


if __name__ == "__main__":
    main()

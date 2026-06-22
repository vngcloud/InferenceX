import sys
import json
import os
import csv
from datetime import datetime
from pathlib import Path


def mean_total_power_w(csv_path, num_gpus, window_s=None):
    """Mean total GPU power (W) over the benchmark window from an nvidia-smi CSV.

    The CSV is the one written by start_gpu_monitor() in benchmark_lib.sh
    (`nvidia-smi --query-gpu=timestamp,index,power.draw,... --format=csv -l 1`).

    Sums the per-GPU mean power of the `num_gpus` busiest GPUs so idle GPUs on a
    shared host (e.g. an unused second card on a TP=1 run) don't inflate the
    figure. When `window_s` is given, only samples within the last `window_s`
    seconds are used so model-load/warmup power (the monitor starts before the
    server) is excluded. Returns None when no usable data is found.
    """
    try:
        with open(csv_path, newline='') as f:
            rows = list(csv.reader(f))
    except (FileNotFoundError, OSError):
        return None

    if len(rows) < 2:
        return None

    header = [h.strip() for h in rows[0]]

    def col(prefix):
        for i, h in enumerate(header):
            if h.startswith(prefix):
                return i
        return None

    ts_i, idx_i, pw_i = col('timestamp'), col('index'), col('power.draw')
    if ts_i is None or idx_i is None or pw_i is None:
        return None

    samples = []  # (ts_seconds, gpu_index, power_w)
    for r in rows[1:]:
        if len(r) <= max(ts_i, idx_i, pw_i):
            continue
        try:
            ts = datetime.strptime(
                r[ts_i].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            gpu = r[idx_i].strip()
            # values look like "350.12 W"; "[N/A]" and similar are skipped
            power = float(r[pw_i].strip().split()[0])
        except (ValueError, IndexError):
            continue
        samples.append((ts, gpu, power))

    if not samples:
        return None

    if window_s and window_s > 0:
        end = max(s[0] for s in samples)
        windowed = [s for s in samples if s[0] >= end - window_s]
        if windowed:
            samples = windowed

    per_gpu = {}
    for _, gpu, power in samples:
        per_gpu.setdefault(gpu, []).append(power)

    gpu_means = sorted(
        (sum(v) / len(v) for v in per_gpu.values()), reverse=True)
    n = max(1, min(num_gpus, len(gpu_means)))
    return sum(gpu_means[:n])


def get_required_env_vars(required_vars):
    """Load and validate required environment variables."""
    env_values = {}
    missing_env_vars = []

    for var_name in required_vars:
        value = os.environ.get(var_name)
        if value is None:
            missing_env_vars.append(var_name)
        env_values[var_name] = value

    if missing_env_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_env_vars)}")

    return env_values


# Base required env vars
base_env = get_required_env_vars([
    'RUNNER_TYPE', 'FRAMEWORK', 'PRECISION', 'SPEC_DECODING',
    'RESULT_FILENAME', 'ISL', 'OSL', 'DISAGG', 'MODEL_PREFIX', 'IMAGE'
])

hw = base_env['RUNNER_TYPE']
model_prefix = base_env['MODEL_PREFIX']
framework = base_env['FRAMEWORK']
precision = base_env['PRECISION']
spec_decoding = base_env['SPEC_DECODING']
disagg = base_env['DISAGG'].lower() == 'true'
result_filename = base_env['RESULT_FILENAME']
isl = base_env['ISL']
osl = base_env['OSL']
image = base_env['IMAGE']
benchmark_client = os.environ.get('BENCHMARK_CLIENT', 'inferencex_native')


def _opt_int(env_name):
    """Read an optional env var as int; '' or unset → None (engine default)."""
    raw = os.environ.get(env_name, '')
    return int(raw) if raw not in ('', '0') else None


num_speculative_tokens = _opt_int('NUM_SPECULATIVE_TOKENS')
max_num_batched_tokens = _opt_int('MAX_NUM_BATCHED_TOKENS')

with open(f'{result_filename}.json') as f:
    bmk_result = json.load(f)

data = {
    'hw': hw,
    'conc': int(bmk_result['max_concurrency']),
    'image': image,
    'model': bmk_result['model_id'],
    'infmax_model_prefix': model_prefix,
    'framework': framework,
    'benchmark_client': benchmark_client,
    'precision': precision,
    'spec_decoding': spec_decoding,
    'num_speculative_tokens': num_speculative_tokens,
    'max_num_batched_tokens': max_num_batched_tokens,
    'disagg': disagg,
    'isl': int(isl),
    'osl': int(osl),
}

is_multinode = os.environ.get('IS_MULTINODE', 'false').lower() == 'true'

if is_multinode:
    # TODO: Eventually will have to have a separate condition in here for multinode disagg and
    # multinode agg. For now, just assume that multinode implies disagg.

    multinode_env = get_required_env_vars(['PREFILL_GPUS', 'DECODE_GPUS', 'PREFILL_NUM_WORKERS', 'PREFILL_TP',
                                          'PREFILL_EP', 'PREFILL_DP_ATTN', 'DECODE_NUM_WORKERS', 'DECODE_TP', 'DECODE_EP', 'DECODE_DP_ATTN'])
    prefill_gpus = int(multinode_env['PREFILL_GPUS'])
    decode_gpus = int(multinode_env['DECODE_GPUS'])
    prefill_num_workers = int(multinode_env['PREFILL_NUM_WORKERS'])
    prefill_tp = int(multinode_env['PREFILL_TP'])
    prefill_ep = int(multinode_env['PREFILL_EP'])
    prefill_dp_attn = multinode_env['PREFILL_DP_ATTN']
    decode_num_workers = int(multinode_env['DECODE_NUM_WORKERS'])
    decode_tp = int(multinode_env['DECODE_TP'])
    decode_ep = int(multinode_env['DECODE_EP'])
    decode_dp_attn = multinode_env['DECODE_DP_ATTN']

    total_gpus = prefill_gpus + decode_gpus
    num_gpus_used = total_gpus
    if total_gpus <= 0:
        raise ValueError("Multinode results require at least one GPU.")
    if prefill_gpus <= 0:
        raise ValueError("Multinode results require at least one prefill GPU.")

    output_tput_denominator = decode_gpus if decode_gpus > 0 else total_gpus
    output_decode_tp = decode_tp if decode_gpus > 0 else 0
    output_decode_ep = decode_ep if decode_gpus > 0 else 0

    multi_node_data = {
        'is_multinode': True,
        'prefill_tp': prefill_tp,
        'prefill_ep': prefill_ep,
        'prefill_dp_attention': prefill_dp_attn,
        'prefill_num_workers': prefill_num_workers,
        'decode_tp': output_decode_tp,
        'decode_ep': output_decode_ep,
        'decode_dp_attention': decode_dp_attn,
        'decode_num_workers': decode_num_workers,
        'num_prefill_gpu': prefill_gpus,
        'num_decode_gpu': decode_gpus,
        'tput_per_gpu': float(bmk_result['total_token_throughput']) / total_gpus,
        'output_tput_per_gpu': float(bmk_result['output_throughput']) / output_tput_denominator,
        'input_tput_per_gpu': (float(bmk_result['total_token_throughput']) - float(bmk_result['output_throughput'])) / prefill_gpus,
    }

    data = data | multi_node_data
else:
    if disagg:
        raise ValueError("Disaggregated mode requires multinode setup.")

    single_node_env = get_required_env_vars(['TP', 'EP_SIZE', 'DP_ATTENTION'])
    tp_size = int(single_node_env['TP'])
    ep_size = int(single_node_env['EP_SIZE'])
    dp_attention = single_node_env['DP_ATTENTION']
    num_gpus_used = tp_size

    single_node_data = {
        'is_multinode': False,
        'tp': tp_size,
        'ep': ep_size,
        'dp_attention': dp_attention,
        'tput_per_gpu': float(bmk_result['total_token_throughput']) / tp_size,
        'output_tput_per_gpu': float(bmk_result['output_throughput']) / tp_size,
        'input_tput_per_gpu': (float(bmk_result['total_token_throughput']) - float(bmk_result['output_throughput'])) / tp_size,
    }

    data = data | single_node_data

for key, value in bmk_result.items():
    if key.endswith('ms'):
        data[key.replace('_ms', '')] = float(value) / 1000.0
    if 'tpot' in key:
        data[key.replace('_ms', '').replace(
            'tpot', 'intvty')] = 1000.0 / float(value)

# Energy efficiency: tokens per watt (tok/s/W) from the nvidia-smi power log.
# Power telemetry is best-effort; a missing/empty CSV leaves the fields null.
gpu_metrics_csv = os.environ.get('GPU_METRICS_CSV', 'gpu_metrics.csv')
window_s = bmk_result.get('duration')
mean_power = mean_total_power_w(gpu_metrics_csv, num_gpus_used, window_s)
data['mean_power_w'] = round(mean_power, 2) if mean_power else None
# Two tokens/Watt conventions, both emitted so reports aren't ambiguous:
#   - total  = (input + output) tokens / W. Dominated by prefill on input-heavy
#     or no-prefix-cache workloads, so it reads high.
#   - output = decoded tokens only / W. The stricter "useful work per Watt".
# `tok_per_watt` is kept as an alias of the total for backward compatibility.
total_tput = float(bmk_result['total_token_throughput'])
output_tput = float(bmk_result['output_throughput'])
data['tok_per_watt_total'] = round(total_tput / mean_power, 4) if mean_power else None
data['tok_per_watt_output'] = round(output_tput / mean_power, 4) if mean_power else None
data['tok_per_watt'] = data['tok_per_watt_total']

print(json.dumps(data, indent=2))

with open(f'agg_{result_filename}.json', 'w') as f:
    json.dump(data, f, indent=2)

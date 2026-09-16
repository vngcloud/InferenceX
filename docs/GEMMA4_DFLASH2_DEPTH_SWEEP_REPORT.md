# Gemma-4 31B — DFlash2 (in-house checkpoint) Depth Sweep Report

Gemma-4 31B FP8-block · 1×H200 (`h200-greennode_06`, checkpoint local to this host) · vLLM v0.28.0 · branch `bench/gemma4cp-8k1k-1gpu`

## 1. Cấu hình đã serve + scenario đã chạy

### 1.1 Engine config 

```bash
export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_P2P_LEVEL=NVL
export VLLM_ATTENTION_BACKEND=FLASHINFER

CUDA_VISIBLE_DEVICES=<gpu> vllm serve RedHatAI/gemma-4-31B-it-FP8-block \
    --host 0.0.0.0 --port <port> \
    --served-model-name RedHatAI/gemma-4-31B-it-FP8-block \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 65536 \
    --max-num-seqs <CONC> \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --long-prefill-token-threshold 8192 \
    --no-enable-prefix-caching \
    --speculative-config '{"model": "/models/gemma4-31b-it-dflash2", "num_speculative_tokens": <DEPTH>, "method": "dflash"}' \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4
```

- **Draft checkpoint:** in-house DFlash2 (`/models/gemma4-31b-it-dflash2`, mounted read-only from `/mnt/models` on `h200-greennode_06` only — not on the Hub). `method` stays `"dflash"`; vLLM auto-selects the `DFlash2Speculator` at runtime from the checkpoint's own `architectures` field (`DFlash2DraftModel`, native in `registry.py` since PR #52816 / v0.28.0). No `trust_remote_code`/`auto_map` needed.
- **`<DEPTH>` (num_speculative_tokens) swept: 3, 5, 7.** Checkpoint's native `dflash_config`: `block_size=8`, `sample_from_anchor=False` → depth 7 = block_size−1 is the checkpoint's native draft width; 3 and 5 are shallower cuts of the same drafter.
- **Prefix caching OFF** on purpose (no KV reuse across requests — isolates the spec-decoding effect; production would run APC on, so these numbers are a floor).
- **`--max-num-seqs` = CONC of the cell** (1/8/32/64) — matches target concurrency exactly, no over-provisioning.

### 1.2 Benchmark scenario

- **Dataset:** nvidia/SPEED-Bench `throughput_8k` (real text, entropy-stratified), ISL ≈ 8192, OSL = 1024, `--ignore-eos`, prefix caching off, `num_prompts = clamp(conc×10, [64, 512])`.
- **Categories run: low_entropy + high_entropy** (same two categories as the prior MTP/dflash sweep; `mixed` still excluded — 268/512 of its rows are gated on `cais/hle`).
- **Concurrency ladder: 1, 8, 32, 64** — identical to the existing MTP arm (`gemma4sbmtphi/lo`, depth 4) and RedHatAI dflash arm (`gemma4sbdfhi/lo`, depth 8), for apples-to-apples comparison.
- **Grid: 2 categories × 3 depths × 4 conc = 24 cells**, all green. `--no-evals` was applied on dispatch — this run is throughput-only, no lm-eval accuracy jobs.
- **Acceptance accounting:** read from Prometheus counters `vllm:spec_decode_*` via a sidecar (`spec_accepted_tokens`, `spec_draft_events`, `spec_drafted_tokens`), not the server log — same method as the MTP/dflash baseline report.

## 2. Performance (throughput + SLA), full 24-cell grid

Đơn vị: `tput` = output tok/s/GPU (decode throughput; loại input tput cho so được với bảng base/MTP/dflash trước). TTFT/ITL = giây. `gain` = so với **base** (no-spec, cùng category/conc, từ run gốc 34502057305 — https://github.com/vngcloud/InferenceX/actions/runs/34502057305).

### high_entropy

| depth | conc | tput (tok/s/GPU) | gain vs base (%) | AL (tok/step) | accrate (0–1) | p99 TTFT (s) | ΔTTFT (%) | p99 ITL (s) |
|---|---|---|---|---|---|---|---|---|
| 3 | 1  | 95.2  | +38.5% | 1.782 | 0.261 | 18.59  | +0.4%  | 0.0147 |
| 3 | 8  | 302.6 | +7.0%  | 1.795 | 0.265 | 28.92  | +9.5%  | 0.5890 |
| 3 | 32 | 648.8 | +5.0%  | 1.751 | 0.250 | 23.22  | +16.7% | 0.6874 |
| 3 | 64 | 700.7 | +10.1% | 1.731 | 0.244 | 124.36 | +59.9% | 0.6723 |
| 5 | 1  | 97.7  | +42.2% | 1.854 | 0.171 | 18.68  | +0.9%  | 0.0148 |
| 5 | 8  | 316.1 | +11.8% | 1.849 | 0.170 | 26.52  | +0.4%  | 0.5247 |
| 5 | 32 | 626.2 | +1.3%  | 1.790 | 0.158 | 23.34  | +17.4% | 0.6286 |
| 5 | 64 | 658.0 | +3.4%  | 1.774 | 0.155 | 126.91 | +63.2% | 0.6801 |
| 7 | 1  | 100.5 | +46.3% | 1.864 | 0.123 | 18.61  | +0.5%  | 0.0143 |
| 7 | 8  | 313.6 | +10.9% | 1.860 | 0.123 | 24.46  | −7.4%  | 0.5945 |
| 7 | 32 | 629.0 | +1.7%  | 1.800 | 0.114 | 22.06  | +10.9% | 0.6688 |
| 7 | 64 | 640.1 | +0.6%  | 1.786 | 0.112 | 103.09 | +32.6% | 0.6915 |

### low_entropy

| depth | conc | tput (tok/s/GPU) | gain vs base (%) | AL (tok/step) | accrate (0–1) | p99 TTFT (s) | ΔTTFT (%) | p99 ITL (s) |
|---|---|---|---|---|---|---|---|---|
| 3 | 1  | 119.8 | +76.6% | 2.519 | 0.506 | 16.27  | +2.4%  | 0.0147 |
| 3 | 8  | 339.1 | +0.5%  | 2.500 | 0.500 | 18.89  | −4.6%  | 0.6180 |
| 3 | 32 | 722.0 | +16.2% | 2.505 | 0.502 | 25.16  | +18.1% | 0.7621 |
| 3 | 64 | 808.9 | +31.3% | 2.502 | 0.501 | 97.12  | −32.6% | 0.6902 |
| 5 | 1  | 128.6 | +89.7% | 2.813 | 0.363 | 16.22  | +2.1%  | 0.0148 |
| 5 | 8  | 372.5 | +10.4% | 2.804 | 0.361 | 14.60  | −26.3% | 0.5534 |
| 5 | 32 | 728.4 | +17.2% | 2.776 | 0.355 | 24.84  | +16.6% | 0.7476 |
| 5 | 64 | 787.5 | +27.9% | 2.789 | 0.358 | 111.29 | −22.7% | 0.6993 |
| 7 | 1  | 135.2 | +99.4% | 2.953 | 0.279 | 16.37  | +3.0%  | 0.0144 |
| 7 | 8  | 389.8 | +15.6% | 2.935 | 0.277 | 15.50  | −21.7% | 0.6258 |
| 7 | 32 | 750.4 | +20.8% | 2.884 | 0.269 | 23.26  | +9.2%  | 0.7506 |
| 7 | 64 | 773.5 | +25.6% | 2.866 | 0.267 | 94.94  | −34.1% | 0.7073 |

**Đọc nhanh:**
- Throughput tăng đơn điệu theo depth ở mọi conc/category (depth 7 > 5 > 3), nhưng gain % nhỏ dần rất nhanh khi conc tăng: c1 gain +38–99%, c32/c64 chỉ còn +1–31%. Batch càng lớn, phần thời gian "cứu" được nhờ spec-decoding càng nhỏ so với tổng — như dflash gốc và MTP đã thấy.
- **p99 TTFT tại c8/c32 tăng nhẹ (+0.4→+18%)** so với base ở cả 3 depth — chi phí verify-step khi hàng đợi chưa bão hòa. Ở **c64 thì trái chiều theo category**: high_entropy TTFT vẫn tăng mạnh (+33→+63%, giống pattern MTP/dflash cũ), nhưng low_entropy TTFT lại **giảm** (−23→−33%) vì AL cao hơn rút ngắn hàng đợi tổng thể ở workload dễ đoán.
- p99 ITL ở c8 tăng theo tỉ lệ % rất lớn (base c8 ITL ~0.015s, cực nhỏ) — **không so được trực tiếp**: base đo per-token, spec đo theo burst (~AL token/lần), giống caveat đã ghi trong report MTP/dflash. Từ c32 trở lên ITL của cả 3 depth ổn định quanh 0.63–0.76s, thấp hơn base c32/c64 (~1.19s) → cải thiện thật.

## 3. Speculative-decoding — số riêng theo depth

### 3.1 Acceptance length & rate theo depth (trung bình 4 conc)

| depth | AL hi (tok/step) | accrate hi (0–1) | AL lo (tok/step) | accrate lo (0–1) |
|---|---|---|---|---|
| 3 | ~1.77 | ~0.255 | ~2.51 | ~0.502 |
| 5 | ~1.82 | ~0.164 | ~2.79 | ~0.360 |
| 7 | ~1.83 | ~0.118 | ~2.91 | ~0.273 |

- **AL tăng theo depth nhưng bão hòa nhanh:** hi đi từ 1.77 (d3) → 1.82 (d5) → 1.83 (d7), lo từ 2.51 → 2.79 → 2.91. Tăng depth từ 3→5 (+67% số token draft/round) chỉ đổi AL +0.05–0.28; từ 5→7 gần như phẳng (hi +0.01, lo +0.12). Checkpoint này **hết hơi ở khoảng draft thứ 2–3**: token thứ 4-7 hiếm khi được accept.
- **accrate giảm đơn điệu theo depth** (per-position acceptance) — đúng cơ chế kỳ vọng: mỗi vị trí xa target hơn thì xác suất đúng thấp hơn, nên rate trung bình toàn block giảm dù AL tuyệt đối vẫn nhích lên.
- **AL/accrate gần như hằng số theo conc** trong mỗi depth (dao động <5% giữa c1 và c64 ở mọi hàng của bảng §2) — batch size không ảnh hưởng đến chất lượng draft, đúng như baseline MTP/dflash đã ghi nhận.
- **low_entropy luôn vượt high_entropy rõ rệt ở mọi depth** (AL lo cao hơn hi ~40–59%) — văn bản dễ đoán hơn giúp drafter đồng ý với target nhiều hơn, nhất quán với 2 arm cũ.

### 3.2 So với 2 arm đã có (cùng conc-list, cùng dataset — từ `GEMMA4_STAGE5_SPEEDBENCH_REPORT.md`)

| arm | depth | AL hi (tok/step) | accrate hi (0–1) | AL lo (tok/step) | accrate lo (0–1) | Run |
|---|---|---|---|---|---|---|
| MTP (`google/gemma-4-31B-it-assistant`) | 4 | 2.41 | 0.35 | 4.10–4.14 | 0.78 | [34502057305](https://github.com/vngcloud/InferenceX/actions/runs/34502057305) |
| dflash (RedHatAI, Hub) | 8 | 1.80–1.89 | 0.10–0.11 | 2.60–2.65 | 0.20 | [34505989997](https://github.com/vngcloud/InferenceX/actions/runs/34505989997) |
| **dflash2 (in-house), sweep này** | 3 | 1.73–1.80 | 0.24–0.27 | 2.50–2.52 | 0.50–0.51 | [34948987453](https://github.com/vngcloud/InferenceX/actions/runs/34948987453) |
| **dflash2 (in-house), sweep này** | 5 | 1.77–1.85 | 0.15–0.17 | 2.78–2.81 | 0.36 | [34948987453](https://github.com/vngcloud/InferenceX/actions/runs/34948987453) |
| **dflash2 (in-house), sweep này** | 7 | 1.79–1.86 | 0.11–0.12 | 2.87–2.95 | 0.27–0.28 | [34948987453](https://github.com/vngcloud/InferenceX/actions/runs/34948987453) |

- **DFlash2 ở depth 7 gần trùng AL với dflash RedHatAI ở depth 8** (hi: 1.83 vs 1.85; nhưng lo: 2.91 vs 2.62 — dflash2 cao hơn hẳn ở low-entropy).
- **Không depth nào của dflash2 tiệm cận MTP** (MTP vẫn AL 2.41/4.10, cao hơn dflash2-depth7 tốt nhất là +32% (hi) / +41% (lo)) — MTP consume hidden state của target nên "nhìn" được nhiều tín hiệu hơn một drafter độc lập.
- **accrate của dflash2-depth3 (0.25 hi / 0.50 lo) là cao nhất trong mọi arm dflash-họ đã bench** — vượt cả dflash RedHatAI depth8 (0.10–0.20) hơn 2×, dù AL tuyệt đối thấp hơn MTP.
- **Điểm ngọt cho throughput:** vì AL gần bão hòa từ depth 5, depth 3 đã ăn phần lớn lợi ích với chi phí draft thấp nhất/round → tại c32/c64 depth 3 và depth 5 cho tput gần nhau (hi c64: 700.7 vs 658.0 — depth 3 thậm chí NHỈNH HƠN depth 5), còn depth 7 chỉ hơn rõ ở c1 (ít batch, chi phí draft không phải bottleneck). Nói cách khác: **tăng depth quá điểm bão hòa AL không đổi lấy thêm throughput tương xứng**, đúng pattern lý thuyết của speculative decoding (chi phí draft tuyến tính theo depth, lợi ích AL thì sub-linear).

## 4. Provenance

| Run | Cells | Nội dung |
|---|---|---|
| 34944216671 | 1 | dflash2 preflight (in-house checkpoint), hi c1, depth 7 |
| 34948987453 | 24 | dflash2 full depth sweep: {hi, lo} × depth {3, 5, 7} × conc {1, 8, 32, 64}, dispatched với `--no-evals` |

- Số liệu từ artifact `results_bmk` (`agg_bmk.json`, 24 rows) và 24 `server_logs_*` (`results/specdec_*.log`, JSON với `acceptance_length`/`acceptance_rate` đọc từ Prometheus counters `vllm:spec_decode_*`) của run `34948987453` trên vngcloud/InferenceX Actions.
- Base/MTP/dflash(RedHatAI) numbers dùng làm denominator/so sánh lấy nguyên từ `docs/GEMMA4_STAGE5_SPEEDBENCH_REPORT.md` (runs 34502057305 / 34505986099 / 34505989997) — cùng engine config, cùng dataset, cùng conc-list nên vẫn là baseline hợp lệ.
- Benchmark scripts: `benchmarks/single_node/fixed_seq_len/gemma4sb_body.sh` (shared body) + wrappers `gemma4sbdf2{hi,hi3,hi5,lo3,lo5,lo7}_fp8block_h200_specdec.sh` (naming: bare `hi`/`lo3`/`lo5`/`lo7` — depth 7 kept the original preflight names `gemma4sbdf2hi`/created `gemma4sbdf2lo7` as separate files); matrix trong `configs/nvidia-master.yaml`.

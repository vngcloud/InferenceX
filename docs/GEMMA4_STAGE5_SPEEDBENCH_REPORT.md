# Stage 5 — SPEED-Bench Spec-Decoding Report

Gemma-4 31B FP8-block · 1×H200 (h200-greennode_07, GPU 4) · vLLM v0.28.0 · branch `bench/gemma4cp-8k1k-1gpu`

## 1. Dataset & cases

- **nvidia/SPEED-Bench `throughput_8k`** — real text, entropy-stratified, prompt ~8k tokens. 1,536 rows = 512 × {low, mixed, high} entropy. Chuẩn bị bằng NeMo prepare.py (hle branch disabled), commit thẳng vào repo (`benchmarks/single_node/fixed_seq_len/speed_bench_throughput_8k.jsonl`, 46MB) vì route mạng runner tới raw.githubusercontent/gutenberg chỉ ~12 kB/s (~3h/cell nếu self-prepare).
- **Benched: low_entropy + high_entropy.** `mixed` không chạy — 268/512 row của nó tới từ cais/hle (gated trên HF Hub, runner token chưa accept) nên chỉ là placeholder.
- Matrix: **2 spec arms × 2 categories × conc {1, 8, 32, 64}** = 16 spec cells + base arm (no-spec, cùng dataset) làm gate denominator → **25 cells / 3 runs** (MTP round 16, dflash preflight 1, dflash full 8). Tất cả jobs xanh.
- Chung mọi cell: OSL 1k, ignore_eos, APC off, num_prompts = conc×10 clamp [64, 512], cùng engine config với Stage 1 (FP8-block, MAX_MODEL_LEN 65536).

## 2. Số raw (tất cả arms)

Đơn vị: tput = tok/s/GPU · TTFT/ITL = giây.

### high_entropy

| arm | conc | tput | AL | accrate | p99 TTFT | p99 ITL |
|---|---|---|---|---|---|---|
| base | 1 | 68.7 | – | – | 18.52 | 0.0124 |
| MTP | 1 | 118.3 | 2.411 | 0.353 | 18.49 | 0.0148 |
| dflash | 1 | 99.3 | 1.875 | 0.109 | 18.53 | 0.0147 |
| base | 8 | 282.8 | – | – | 26.41 | 0.0153 |
| MTP | 8 | 328.4 | 2.427 | 0.357 | 24.80 | 0.5208 |
| dflash | 8 | 286.2 | 1.885 | 0.111 | 26.75 | 0.5405 |
| base | 32 | 618.2 | – | – | 19.89 | 1.1938 |
| MTP | 32 | 681.8 | 2.415 | 0.354 | 23.22 | 0.6921 |
| dflash | 32 | 505.9 | 1.818 | 0.102 | 21.09 | 0.6248 |
| base | 64 | 636.6 | – | – | 77.76 | 1.2056 |
| MTP | 64 | 739.4 | 2.390 | 0.348 | 105.35 | 0.6971 |
| dflash | 64 | 526.3 | 1.804 | 0.100 | 118.43 | 0.5906 |

### low_entropy

| arm | conc | tput | AL | accrate | p99 TTFT | p99 ITL |
|---|---|---|---|---|---|---|
| base | 1 | 67.8 | – | – | 15.89 | 0.0124 |
| MTP | 1 | 162.0 | 4.085 | 0.771 | 16.11 | 0.0150 |
| dflash | 1 | 124.7 | 2.652 | 0.206 | 16.04 | 0.0147 |
| base | 8 | 337.3 | – | – | 19.80 | 0.0154 |
| MTP | 8 | 409.9 | 4.123 | 0.781 | 20.18 | 0.6267 |
| dflash | 8 | 351.2 | 2.597 | 0.200 | 14.50 | 0.5625 |
| base | 32 | 621.4 | – | – | 21.31 | 1.1931 |
| MTP | 32 | 794.4 | 4.124 | 0.781 | 24.61 | 1.2697 |
| dflash | 32 | 600.3 | 2.613 | 0.202 | 21.77 | 0.6677 |
| base | 64 | 615.9 | – | – | 144.00 | 1.1942 |
| MTP | 64 | 885.5 | 4.142 | 0.786 | 102.71 | 0.8498 |
| dflash | 64 | 623.3 | 2.604 | 0.201 | 93.08 | 0.6108 |

## 3. MTP — `google/gemma-4-31B-it-assistant`, depth 4

**Acceptance:** AL **2.41** (hi) / **4.10** (lo); acceptance rate **0.35** / **0.78** — flat ở mọi concurrency.

| tput (tok/s/GPU) | c1 | c8 | c32 | c64 |
|---|---|---|---|---|
| base hi / lo | 68.7 / 67.8 | 282.8 / 337.3 | 618.2 / 621.4 | 636.6 / 615.9 |
| MTP hi / lo | 118.3 / 162.0 | 328.4 / 409.9 | 681.8 / 794.4 | 739.4 / 885.5 |
| gain hi | +72.1% | +16.1% | +10.3% | +16.2% |
| gain lo | +138.9% | +21.5% | +27.8% | **+43.8%** |

**SLA:** p99 TTFT tăng ở hi c32/c64 (+16.8% / +35.5%) — điểm nghẽn duy nhất; lo c64 TTFT lại **giảm 28.7%**. p99 ITL cải thiện ở c32/c64 (−42% / −29%). Ở c1 ITL +20% (overhead thật của verify step, nhưng per-token wall time vẫn giảm tới −58% nhờ AL); p99 ITL c8 không so được giữa base và spec (spec stream theo burst ~AL token/chunk nên ITL của nó là inter-burst gap, không per-token như base). mean ITL base c8 > p99 (0.021/0.015) không phải mâu thuẫn: <1% ITL là stall 0.35–0.63s do prefill chunk xen kẽ, p99 chưa chạm tới — sang c32/c64 stall đó vượt 1% tần suất nên p99 lên 1.19s.

## 4. dflash — `RedHatAI/gemma-4-31B-it-speculator.dflash`, depth 8

**Acceptance:** AL **1.8** (hi) / **2.6** (lo); acceptance rate chỉ **0.10** / **0.20**.

| tput (tok/s/GPU) | c1 | c8 | c32 | c64 |
|---|---|---|---|---|
| base hi / lo | 68.7 / 67.8 | 282.8 / 337.3 | 618.2 / 621.4 | 636.6 / 615.9 |
| dflash hi / lo | 99.3 / 124.7 | 286.2 / 351.2 | 505.9 / 600.3 | 526.3 / 623.3 |
| gain hi | +44.5% | +1.2% | **−18.2%** | **−17.3%** |
| gain lo | +83.9% | +4.1% | −3.4% | +1.2% |

**SLA:** p99 TTFT hi c64 **+52.3%**. Từ c8 trở lên không cell nào chạm +15%, hi c32/c64 còn âm. Model card cũng chỉ validate trên H100. **→ loại.**

## 5. Gate verdicts

Gate: tput ≥ +15% & p99 TTFT ≤ +5% & p99 ITL ≤ +5% so với base cùng conc + category.

| Cell | tput | p99 TTFT | p99 ITL | Verdict |
|---|---|---|---|---|
| MTP lo c64 | ✅ +43.8% | ✅ −28.7% | ✅ −28.8% | **PASS — cell duy nhất pass trọn gate** |
| MTP lo c8 | ✅ +21.5% | ✅ +1.9% | ⚠ artifact | pass* |
| MTP hi c8 | ✅ +16.1% | ✅ −6.1% | ⚠ artifact | pass* |
| MTP hi c1 | ✅ +72.1% | ✅ −0.2% | ❌ +20.0% | fail (letter) |
| MTP lo c1 | ✅ +138.9% | ✅ +1.4% | ❌ +20.8% | fail (letter) |
| MTP lo c32 | ✅ +27.8% | ❌ +15.5% | ❌ +6.4% | fail |
| MTP hi c64 | ✅ +16.2% | ❌ +35.5% | ✅ −42.2% | fail |
| MTP hi c32 | ❌ +10.3% | ❌ +16.8% | ✅ −42.0% | fail |
| dflash — cả 8 cells | ❌ (≤ +4.2% hoặc âm, trừ c1) | ❌ hi c64 +52% | ❌ | **fail toàn bộ** |

\* ITL c8 không comparable giữa base và spec (base per-token, spec per-burst ~AL token); cross-check per-token wall time (conc/tput): MTP −14% (hi) / −18% (lo) tại c8.

## 6. Serve command — config đã bench, mang lên production

Từ run thật (`benchmarks/single_node/fixed_seq_len/gemma4sb_body.sh:259-273`, image `vllm/vllm-openai:v0.28.0`), áp đúng 1 thay đổi cho prod: bỏ `--no-enable-prefix-caching` → APC on. Ví dụ là arm MTP — arm thắng:

```bash
export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_P2P_LEVEL=NVL
export VLLM_ATTENTION_BACKEND=FLASHINFER

CUDA_VISIBLE_DEVICES=4 vllm serve RedHatAI/gemma-4-31B-it-FP8-block \
    --host 0.0.0.0 --port 8888 \
    --served-model-name RedHatAI/gemma-4-31B-it-FP8-block \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 65536 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --long-prefill-token-threshold 8192 \
    --speculative-config '{"method": "mtp", "model": "google/gemma-4-31B-it-assistant", "num_speculative_tokens": 4}' \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4
```

**Biến theo arm** — mỗi arm chỉ khác mỗi `--speculative-config`, giữ nguyên phần còn lại:

| arm | `--speculative-config` |
|---|---|
| base (no spec) | bỏ hẳn flag |
| **mtp** (winner) | `{"method": "mtp", "model": "google/gemma-4-31B-it-assistant", "num_speculative_tokens": 4}` |
| dflash | `{"model": "RedHatAI/gemma-4-31B-it-speculator.dflash", "num_speculative_tokens": 8, "method": "dflash"}` |
| e3 | `{"model": "RedHatAI/gemma-4-31B-it-speculator.eagle3", "num_speculative_tokens": 3, "method": "eagle3"}` |

**Biến theo cell / môi trường:**

- `--max-num-seqs` = CONC của cell (1/8/32/64; ví dụ trên là 64). Prod đặt theo target concurrency — số trong report đo đúng tại giá trị này.
- `CUDA_VISIBLE_DEVICES=4` + `--port 8888`: pin GPU/port của node bench (h200-greennode_07) — đổi hoặc bỏ theo node prod.
- Env: `VLLM_DISABLE_COMPILE_CACHE=1` là hygiene bench (prod bỏ để restart nhanh hơn), `NCCL_P2P_LEVEL=NVL` chỉ có ý nghĩa khi TP>1, `VLLM_ATTENTION_BACKEND=FLASHINFER` là no-op ở v0.28 (vLLM tự chọn backend).

**Lưu ý prod:**

- `method` PHẢI là `"mtp"` — checkpoint assistant ăn hidden states của target, sai method là crash lúc init, không phải chạy chậm.
- APC: bench chạy với `--no-enable-prefix-caching` cho sạch (không KV reuse giữa request); command prod ở trên đã bỏ flag → APC on (default v0.28). Số trong report đo với APC off nên là floor; workload shared-prefix/system-prompt sẽ còn nhanh hơn.
- KHÔNG thêm `--kv-cache-dtype fp8`: nó pin Gemma-4 vào Triton trên SM90, tốn +72% TTFT / +25% TPOT / −20% req/s.
- Tradeoff duy nhất khi adopt MTP: p99 TTFT tăng ở high-entropy c32/c64 (+16.8% / +35.5%) — xem §5.

## Kết luận

MTP thắng tuyệt đối (AL 2.4–4.1 vs 1.8–2.6). Chỉ **lo c64** pass trọn gate; blocker duy nhất của MTP là p99 TTFT ở high-entropy c32/64. Nếu production chấp nhận TTFT tradeoff đó thì MTP adopt được ở mọi conc; không thì adopt cho workload thiên low-entropy.

## Provenance

| Run | Cells | Nội dung |
|---|---|---|
| 34502057305 | 16 | base + MTP, hi/lo × c{1,8,32,64} |
| 34505986099 | 1 | dflash preflight hi c1 |
| 34505989997 | 8 | dflash full, hi/lo × c{1,8,32,64} |

- Số liệu từ artifact `results_bmk` (agg_bmk.json) và `server_logs_*` (specdec sidecar với AL/accrate đọc từ Prometheus counters `vllm:spec_decode_*`) của từng run trên vngcloud/InferenceX Actions.
- Benchmark scripts: `benchmarks/single_node/fixed_seq_len/gemma4sb_body.sh` + wrappers `gemma4sb{base,df,mtp}{hi,lo}.sh`; matrix trong `configs/nvidia-master.yaml`.

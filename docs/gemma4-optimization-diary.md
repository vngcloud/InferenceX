# Nhật ký tối ưu Gemma-4 31B (FP8-block) trên SPEED-Bench

Model: `RedHatAI/gemma-4-31B-it-FP8-block`, 1×H200 (TP1). Workload: SPEED-Bench `throughput_8k` qua aiperf 0.12.0, ISL 8192 / OSL 1024, ignore-eos, dataset `Noridom1/speed-bench-prepared@6e5c3a81`.
Ma trận đầy đủ gồm 16 cell cho mỗi engine: {base, MTP depth 4} × {high entropy, low entropy} × conc {1, 8, 32, 64}. Mọi run đi qua workflow `e2e-tests.yml` của `vngcloud/InferenceX` (không chạy slurm thủ công).

Cập nhật lần cuối: 2026-10-09. Số liệu lấy từ artifact `results_bmk` / `result.json` / `server.log` của từng run.

## Cách đọc

- **Trạng thái**: `FULL` = đã chạy đủ cell trong ma trận, `PARTIAL` = thiếu cell, `PROBE` = chỉ chạy một cell để dò hành vi (một mẫu, không dùng làm số chính thức), `SMOKE` = chạy để kiểm tra khởi động/OOM.
- tput = output tok/s trên 1 GPU. TTFT tính bằng giây; TPOT, ITL tính bằng ms. ITL là median, `p99 ITL` là p99.
- Hi/lo = high/low entropy. Chữ `*` sau c64 của SGLang MTP: chỉ chạy 32 slot đồng thời (xem mục SGLang).
- Mỗi cấu hình chỉ có một lần chạy, chưa có lặp lại nên chưa có khoảng tin cậy. Chênh lệch dưới vài % không đáng tin.

## Mục lục: đã làm gì, theo thứ tự thời gian

| # | Việc đã làm | Kết luận chính | Xem mục |
|---|---|---|---|
| 1 | vLLM v0.30.0, 16 cell, mem 0.92, 65k, run 37259992014 | Có baseline đầy đủ. c64 bị KV chặn ở base | "Phần 1: vLLM" |
| 2 | SGLang base 8 cell, image 2d74309, mem 0.88, run 37298154251 | Base chạy được. MTP trên image này lỗi | "Kết quả đầy đủ" (SGLang) |
| 3 | Bug MTP + FA4 `CUDA illegal memory access`, bản fix (nhánh `fix/fa-frozen-kv-mtp-swa-pool`, image B) | MTP chạy ổn trên image B, 8 cell MTP có số | "Bug MTP + FA4", file `gemma4-frozen-kv-mtp-fa-bug-report.md` |
| 4 | Probe KV pool base c64 (a, b, c), smoke MTP mem 0.92 (d, e) | Pool c64 nhỏ vì cap-mode; mem 0.92 chạy được | "KV pool ở base c64", "Smoke MTP hi ở mem 0.92" |
| 5 | Smoke tuned f, g (0.92 + evict 32 + chunk 8192), rồi 16 cell tuned (commit e46c9d30) | Có cell lo c8/c32 thụt lùi, cần điều tra | "SGLang tuned, 16 cell" |
| 6 | Điều tra base lo c8/c32, sweep mem-only 16 cell (commit 28272142, chunk 16384) | Lỗi do SWA pool nhỏ ở chunk 8192; mem-only hết thụt lùi | "Điều tra base lo c8/c32 và sweep mem-only", "So pool và retract" |
| 7 | Probe A-G trên mem 0.92 (commit 1aee9614), một cell mỗi cấu hình | Số một mẫu, chưa dùng làm số chính thức. Probe G dùng sai cờ, chưa chạy lại | "Probe A-G trên mem 0.92" |
| 8 | Chuyển sang 256k + prefix cache / radix cho cả hai engine (commit b2c9f949) | Pool tính theo ratio, không còn theo conc | "SGLang 256k + radix cache" |
| 9 | SGLang 32 cell: A (ratio 0.3, chunk 16384, run 37880075115 / 37880078321) và B (cấu hình tau2, commit b0208a66, run 37884261982 / 37884264773) | Base c64 +21-27% so mem-only; B không retract ở MTP; prefix cache hit chỉ 1-3% | cùng mục trên |
| 10 | vLLM 256k + prefix cache, 16 cell (commit 4bd28f8c, run 37911942823 / 37911946936, Greennode `1x_01/_02`). Lần dispatch trên hcm-8x/1x không có runner nhận nên đã hủy | 16/16 passed. Ngang vLLM 65k trong khoảng nhiễu (prefix hit 0-2%); base c64 vLLM cao hơn SGLang B 14-15%, c1 thấp hơn 25-32% | "vLLM 256k + prefix cache" |
| 11 | Đối chiếu run tau2 37879258874 (agentic) với cấu hình SpeedBench | B trùng pool với tau2 (365,387 / 73,077) | cuối mục "SGLang 256k + radix cache" |

Chưa làm, chờ quyết định: chọn mặc định A hay B; chạy lại probe G với `--fp8-gemm-backend`; FP8 KV qua triton; cell MTP depth khác 4; kiểm tra nhãn runner hcm; chờ maintainer gắn nhãn `run-ci` cho PR upstream sgl-project/sglang#43542 (đã ready for review, CI bị `pr-gate` chặn khi còn draft).

## Tổng quan trạng thái

| Engine | Cấu hình | Run | Trạng thái |
|---|---|---|---|
| vLLM v0.30.0 | 16 cell, mem 0.92 | 37259992014 (`8x_01`) | FULL, 16/16 passed |
| SGLang base (image 2d74309) | 8 cell, mem 0.88 | 37298154251 (`8x_02`) | FULL cho base, 8/8 passed (cả run đỏ vì các cell MTP trên image này lỗi, đã chạy lại bằng image B) |
| SGLang MTP (image B) | 8 cell | 37738313355, 37738317822 (Greennode 1x_01/_02), 37735723272 (hcm `8x_02`), 37601519336 | FULL về số cell: 8/8 có số. Run hcm 37735723272 còn hi c64 chưa chạy xong (đã có số tương ứng từ Greennode) |
| SGLang KV-pool probe base c64 | a, c cũ, c mới | 37745925506, 37745935978, 37754283939 | PROBE, một mẫu mỗi cấu hình |
| SGLang KV-pool probe b | mem 0.92 + eviction 32 | 37745930415 | Đã hủy (trùng với c cũ) |
| SGLang MTP smoke mem 0.92 | d: c32 (32 slot), e: c64 (64 slot) | 37760820543, 37760825959 | SMOKE, xong cả hai |
| SGLang tuned smoke (0.92 + evict 32 + chunk 8192) | f: MTP hi c32, g: base hi c8 | 37767038996, 37767042768 | SMOKE, xong cả hai (commit 341b0b51) |
| SGLang 256k + radix, cấu hình A (ratio 0.3, chunk 16384) | 16 cell, Greennode `1x_01/_02` | 37880075115, 37880078321 | FULL, 16/16 passed (commit b2c9f949) |
| SGLang 256k + radix, cấu hình B (tau2: ratio 0.2, chunk 8192, hrrn, evict 32) | 16 cell, Greennode `1x_01/_02` | 37884261982, 37884264773 | FULL, 16/16 passed (commit b0208a66) |
| vLLM 256k + prefix cache | 16 cell, Greennode `1x_01/_02` | 37911942823, 37911946936 | FULL, 16/16 passed (commit 4bd28f8c) |

---

# Phần 1: vLLM

## Cấu hình

| Tham số | Giá trị |
|---|---|
| Image | vLLM v0.30.0 |
| `--gpu-memory-utilization` | 0.92 (tính cả activation và cuda graph) |
| `--max-model-len` | 65536 |
| `--max-num-seqs` | = conc |
| `--max-num-batched-tokens` | 16384 (budget mỗi step, gồm cả decode) |
| Prefix caching | tắt |
| Attention | FA4 do vLLM tự chọn (`VLLM_ATTENTION_BACKEND=FLASHINFER` trong recipe bị v0.30.0 bỏ qua, xem ghi chú dưới), KV BF16 |
| MTP | `num_speculative_tokens=4`, draft `google/gemma-4-31B-it-assistant` |
| Runner | `hardware-hcm-8x_01` |

Ghi nhận khi khởi động (base c64): activation đỉnh 3.46 GiB, graph pool 0.39 GiB, KV 92.89 GiB, `GPU KV cache size: 197,683 tokens`.

## Kết quả đầy đủ (FULL, run 37259992014)

### Base

| Entropy | conc | tput | TTFT (s) | TPOT (ms) | ITL (ms) | p99 ITL (ms) | E2E (s) |
|---|---|---|---|---|---|---|---|
| hi | 1 | 53.1 | 0.69 | 18 | 18 | 20 | 19.2 |
| hi | 8 | 323.8 | 4.40 | 20 | 19 | 21 | 25.1 |
| hi | 32 | 649.3 | 12.82 | 37 | 29 | 99 | 50.4 |
| hi | 64 | 695.5 | 12.85 | 75 | 45 | 1260 | 89.7 |
| lo | 1 | 52.6 | 0.73 | 18 | 18 | 21 | 19.6 |
| lo | 8 | 317.3 | 4.27 | 21 | 20 | 39 | 25.7 |
| lo | 32 | 634.2 | 12.98 | 38 | 29 | 143 | 51.7 |
| lo | 64 | 694.2 | 13.94 | 74 | 45 | 1219 | 89.0 |

### MTP depth 4

| Entropy | conc | tput | TTFT (s) | TPOT (ms) | ITL (ms) | p99 ITL (ms) | E2E (s) | Accept length | Accept rate |
|---|---|---|---|---|---|---|---|---|---|
| hi | 1 | 94.4 | 0.72 | 10 | 23 | 25 | 11.0 | 2.31 | 32.8% |
| hi | 8 | 455.3 | 0.79 | 16 | 26 | 446 | 17.0 | 2.36 | 34.0% |
| hi | 32 | 752.2 | 0.95 | 40 | 45 | 668 | 42.7 | 2.38 | 34.6% |
| hi | 64 | 785.7 | 12.45 | 68 | 76 | 629 | 83.0 | 2.40 | 35.0% |
| lo | 1 | 156.9 | 0.78 | 6 | 23 | 25 | 6.7 | 4.13 | 78.4% |
| lo | 8 | 620.3 | 0.85 | 11 | 27 | 495 | 12.2 | 4.17 | 79.2% |
| lo | 32 | 915.2 | 1.60 | 31 | 46 | 1127 | 34.2 | 4.16 | 79.1% |
| lo | 64 | 942.3 | 12.66 | 52 | 78 | 777 | 68.1 | 4.14 | 78.4% |

Accept length ở vLLM lấy từ `spec_accepted_tokens / spec_draft_events + 1`, accept rate = accepted / drafted.

## vLLM 256k + prefix cache (FULL, commit 4bd28f8c, run 37911942823 / 37911946936, Greennode `1x_01` hi và `1x_02` lo, 2026-10-09)

Cùng ma trận 16 cell, nhưng `--max-model-len 262144` và `--enable-prefix-caching` (hai thay đổi cùng lúc so với run 37259992014, nên chỉ so định tính; ngoài ra node khác: hcm `8x_01` ở run cũ). Các tham số còn lại giữ nguyên: mem 0.92, `max-num-batched-tokens 16384`, `--max-num-seqs` = conc. 16/16 cell `passed` (0 failed).

**Cách đọc bảng dưới đây:** tput = output tok/s mỗi GPU; TTFT = **mean**, giây; TPOT = **median**, ms; ITL = **mean**, ms (đo theo chunk, xem mục "Vì sao ITL của MTP cao hơn base"); p99 ITL, ms; E2E = mean, giây. Bảng vLLM 65k phía trên ghi ITL là median và có thể TTFT cũng là median (median TTFT của MTP hi c8 ở run mới là 0.82 s, gần với 0.79 của bảng cũ, còn mean là 2.23 s), nên **chỉ so tput và TPOT giữa hai bảng, đừng so TTFT và ITL**.

### Base (vLLM 256k)

| Entropy | conc | tput | TTFT (s) | TPOT (ms) | ITL (ms) | p99 ITL (ms) | E2E (s) |
|---|---|---|---|---|---|---|---|
| hi | 1 | 53.7 | 0.76 | 18 | 18 | 20 | 19.1 |
| hi | 8 | 329.7 | 4.00 | 20 | 21 | 22 | 24.8 |
| hi | 32 | 654.5 | 12.21 | 37 | 44 | 137 | 50.0 |
| hi | 64 | 717.2 | 16.70 | 73 | 91 | 1223 | 89.8 |
| lo | 1 | 52.7 | 0.75 | 18 | 18 | 21 | 19.4 |
| lo | 8 | 320.3 | 3.82 | 21 | 22 | 39 | 25.6 |
| lo | 32 | 651.4 | 12.12 | 37 | 48 | 486 | 50.3 |
| lo | 64 | 708.2 | 17.70 | 73 | 101 | 1056 | 91.4 |

### MTP depth 4 (vLLM 256k)

| Entropy | conc | tput | TTFT (s) | TPOT (ms) | ITL (ms) | p99 ITL (ms) | E2E (s) | Accept length | Accept rate |
|---|---|---|---|---|---|---|---|---|---|
| hi | 1 | 90.1 | 0.73 | 11 | 24 | 25 | 11.4 | 2.32 | 32.9% |
| hi | 8 | 450.0 | 2.23 | 16 | 35 | 447 | 17.6 | 2.36 | 33.9% |
| hi | 32 | 766.9 | 3.86 | 39 | 89 | 672 | 42.1 | 2.39 | 34.6% |
| hi | 64 | 792.9 | 16.10 | 67 | 153 | 620 | 81.4 | 2.39 | 34.8% |
| lo | 1 | 153.1 | 0.73 | 6 | 24 | 27 | 6.7 | 4.14 | 78.5% |
| lo | 8 | 616.0 | 2.22 | 10 | 44 | 546 | 13.0 | 4.18 | 79.6% |
| lo | 32 | 947.9 | 4.20 | 30 | 122 | 1226 | 34.1 | 4.14 | 78.6% |
| lo | 64 | 961.7 | 15.65 | 51 | 209 | 756 | 67.2 | 4.13 | 78.2% |

### KV pool, peak running và prefix cache hit (vLLM 256k, đọc từ log server)

| Arm | Entropy | conc | KV pool (token) | Peak running | Peak waiting | Peak KV usage | Prefix hit cuối run | Preempt |
|---|---|---|---|---|---|---|---|---|
| base | hi | 1 | 533,316 | 1 | 0 | 1.9% | 2.3% | 0 |
| base | hi | 8 | 531,415 | 8 | 3 | 34.4% | 2.0% | 0 |
| base | hi | 32 | 530,800 | 32 | 16 | 75.4% | 0.7% | 0 |
| base | hi | 64 | 531,709 | 59 | 48 | 98.9% | 0.9% | 0 |
| base | lo | 1 | 532,883 | 1 | 0 | 9.5% | 1.9% | 0 |
| base | lo | 8 | 530,982 | 8 | 4 | 35.1% | 2.0% | 0 |
| base | lo | 32 | 530,367 | 32 | 23 | 66.7% | 2.0% | 0 |
| base | lo | 64 | 531,275 | 58 | 50 | 98.3% | 1.7% | 0 |
| mtp | hi | 1 | 523,840 | 1 | 0 | 1.9% | 1.6% | 0 |
| mtp | hi | 8 | 523,016 | 8 | 0 | 31.2% | 1.3% | 0 |
| mtp | hi | 32 | 520,863 | 32 | 27 | 70.6% | 0.0% | 0 |
| mtp | hi | 64 | 519,675 | 56 | 54 | 99.5% | 0.2% | 0 |
| mtp | lo | 1 | 523,407 | 1 | 0 | 9.3% | 1.2% | 0 |
| mtp | lo | 8 | 522,582 | 8 | 0 | 26.5% | 1.3% | 0 |
| mtp | lo | 32 | 520,430 | 32 | 18 | 78.5% | 1.3% | 0 |
| mtp | lo | 64 | 519,242 | 56 | 50 | 99.1% | 0.7% | 0 |

Peak running / waiting / KV usage lấy từ các dòng `loggers.py` mỗi 10 giây nên có thể sót đỉnh thật. "Prefix hit cuối run" là giá trị cộng dồn ở dòng log cuối cùng. Preempt = số dòng chứa `preempt` trong log (không có dòng nào).

**Nhận xét.**
- **Pool**: `GPU KV cache size` là 519k đến 533k token (so với 193k đến 198k ở 65k), đúng như công thức đã ghi ở mục "Tổng hợp KV pool" (giá trị này tính theo `max_model_len`; MTP hi c64 in ra đúng 519,675 token như dự đoán). Đây không phải thêm bộ nhớ, chỉ đổi đơn vị.
- **Không preempt, nhưng c64 vẫn chạm trần KV**: KV usage đỉnh 98.3-99.5% ở c64, peak running 56-59 trên 64, waiting 48-54. Giống run 65k (peak 55-58): c64 bị KV chặn ở vLLM.
- **Prefix cache gần như không có tác dụng**: hit rate cuối run chỉ 0.0-2.3%, cùng kết luận với SGLang (1-3%). Vì vậy số 256k và 65k ngang nhau trong khoảng nhiễu.
- **So với run vLLM 65k (tput)**: base c1/c8/c32/c64 hi 53.1 / 323.8 / 649.3 / 695.5 lên 53.7 / 329.7 / 654.5 / 717.2 (c64 +3.1%); lo 52.6 / 317.3 / 634.2 / 694.2 lên 52.7 / 320.3 / 651.4 / 708.2 (c64 +2.0%). MTP hi 94.4 / 455.3 / 752.2 / 785.7 lên 90.1 / 450.0 / 766.9 / 792.9; MTP lo 156.9 / 620.3 / 915.2 / 942.3 lên 153.1 / 616.0 / 947.9 / 961.7. Chênh lệch trong khoảng -4.6% (MTP hi c1) đến +3.6% (MTP lo c32), phần lớn dưới 3%, tức coi như không đổi.
- **So với SGLang 256k cấu hình B (tput, tok/s)**:
  - c1: SGLang nhanh hơn 25-32% (base 66.9 / 66.5 so với 53.7 / 52.7, MTP 118.9 / 198.6 so với 90.1 / 153.1).
  - c8: base ngang nhau (333 so với 330, 334 so với 320); MTP SGLang cao hơn 6% (hi 479 so với 450) và 11% (lo 685 so với 616).
  - c32: ngang nhau trong khoảng 3% (base 653 / 647 so với 655 / 651; MTP hi 773 so với 767, lo 972 so với 948).
  - c64: base vLLM cao hơn 14-15% (717 / 708 so với 623 / 623, do SGLang base c64 bị KV chặn); MTP ngang nhau (793 / 962 so với 776 / 975), nhưng SGLang MTP chỉ chạy 32 slot còn vLLM chạy tới 56.
  - Cả hai engine đều chạy FA4 (log vLLM: `Using FLASH_ATTN attention backend`, `Using FlashAttention version 4`), nên chênh lệch không đến từ attention backend. Mỗi cell chỉ một mẫu, nên đây là so sánh định tính.

### Vì sao ITL của MTP cao hơn base

Kiểm tra trên file raw `profile_export.jsonl` của SGLang 256k·c8k (B), hi, c1 (run 37884261982, 64 request × 1024 token):

| | base | MTP depth 4 |
|---|---|---|
| Token mỗi chunk | 1.00 | 2.31 |
| Chunk mỗi request | 1020.7 | 442.5 |
| ITL (inter-chunk trung bình, ms) | 14.3 | 17.9 |
| TPOT = (e2e - TTFT) / (OSL - 1) (ms) | 14.3 | 7.7 |

ITL trong `result.json` (`mean_itl_ms`) là inter-chunk latency: khoảng cách giữa hai chunk stream. Với MTP mỗi bước verify gửi nhiều token trong một chunk (không có gap nào dưới 3 ms), nên ITL là thời gian một bước draft + verify, dài hơn một bước decode của base. Metric `inter_token_latency` của aiperf (7.72 ms) lại bằng TPOT. Khi so MTP với base hãy dùng TPOT, E2E hoặc tput, không dùng ITL. Chỉ kiểm tra SGLang c1; vLLM dùng cùng định nghĩa ITL trong `result.json` nên chắc chắn tương tự, nhưng chưa mở raw của vLLM.

## Nhật ký vLLM

- Các run sweep đầu (37255960787, 37255963667, 37256173172) bị hủy. Sau đó chuyển sang v0.30.0, có một run import probe lỗi (37257811513) trước khi run đầy đủ 37259992014 chạy thành công. Chưa ghi lại nguyên nhân từng lần hủy/lỗi.
- Chưa có thử nghiệm điều chỉnh tham số nào cho vLLM (mem, max-num-batched-tokens...). Số hiện tại là cấu hình gốc.
- Chạy lại ở 256k + prefix cache: xem mục "vLLM 256k + prefix cache" phía trên (16/16 passed).

---

# Phần 2: SGLang

## Cấu hình

| Tham số | vLLM tương ứng | SGLang |
|---|---|---|
| Image | v0.30.0 | `ghcr.io/noridom1/sglang:v0.5.21-gemma4-fa4-swapool-3a7d5ad` (image B = v0.5.21 + FA4 PR #42019 + swa-pool fix, không có aux). Base ban đầu chạy trên image cũ `...-gemma4-2d74309` |
| Memory | `--gpu-memory-utilization 0.92` | `--mem-fraction-static 0.88` (mặc định). SGLang chỉ tính weights + KV, không tính activation/graph |
| Context | `--max-model-len 65536` | `--context-length 65536` |
| Concurrency | `--max-num-seqs` | `--max-running-requests` = conc (MTP c64: 32, xem dưới) |
| Prefill | `--max-num-batched-tokens 16384` | `--chunked-prefill-size 16384` (gần tương đương, không giống hệt) |
| Prefix cache | tắt | `--disable-radix-cache` |
| Attention | FA4 (vLLM tự chọn `FLASH_ATTN`, FA version 4) | `--attention-backend fa4`, KV BF16 |

> **Ghi chú backend vLLM.** Recipe `gemma4sba_body.sh` và recipe tau2 có `export VLLM_ATTENTION_BACKEND=FLASHINFER`, nhưng log server in `Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND`: biến này bị vLLM v0.30.0 bỏ qua. vLLM tự chọn `FLASH_ATTN` với FlashAttention version 4 (cũng thấy một dòng `TRITON_FLASH_ATTN`, nhiều khả năng là phần encoder phụ, chưa kiểm chứng). Đã đọc log của vLLM 256k (base hi, base lo, MTP hi) và run tau2 37800187468. Run vLLM 65k cùng image và cùng recipe nên chắc cũng vậy, chưa đọc log. Trước đây diary ghi nhầm là FlashInfer. Smoke thử ép FA4 (run 37944457459, ref `gemma4sba-vllm-fa4`) đã hủy vì vô nghĩa: hai biến env của nó cũng bị bỏ qua.
| MTP | `num_speculative_tokens=4` | NEXTN, `num-steps 4`, `topk 1`, `draft-tokens 5` (Gemma4 assistant draft, tự đổi sang FROZEN_KV_MTP) |

Bảng trên và mọi kết quả cũ hơn ngày 2026-10-09 dùng context 65536 và radix tắt. Từ 2026-10-09 cả vLLM và SGLang chạy `max-model-len` / `context-length` 262144 và bật prefix cache / radix (xem mục "SGLang 256k + radix cache"); vì vậy số cũ và mới không so được một chiều (hai thay đổi cùng lúc).

Image A (`...-gemma4-aux-fa4-swapool-8c1384a`) cho kết quả sai khác dưới 1.5% so với B ở hi c1/c8 (116.9 so với 115.7 tok/s ở c1, 448.7 so với 448.2 ở c8).

## Kết quả đầy đủ

### Base (FULL, run 37298154251, image 2d74309, `hardware-hcm-8x_02`, mem 0.88)

| Entropy | conc | tput | TTFT (s) | TPOT (ms) | ITL (ms) | p99 ITL (ms) | E2E (s) |
|---|---|---|---|---|---|---|---|
| hi | 1 | 66.2 | 0.67 | 14 | 14 | 20 | 15.3 |
| hi | 8 | 324.7 | 3.23 | 21 | 19 | 26 | 24.8 |
| hi | 32 | 647.5 | 5.89 | 43 | 28 | 102 | 50.5 |
| hi | 64 | 361.9 | 136.4 | 29 | 24 | 32 | 165.0 |
| lo | 1 | 66.0 | 0.71 | 14 | 14 | 20 | 15.4 |
| lo | 8 | 324.8 | 3.56 | 21 | 19 | 26 | 25.2 |
| lo | 32 | 634.2 | 6.20 | 44 | 28 | 142 | 51.7 |
| lo | 64 | 349.6 | 146.2 | 31 | 24 | 70 | 178.6 |

Base c64 thấp hơn hẳn vLLM (362 so với 696 tok/s) vì KV pool nhỏ: xem mục "KV pool" bên dưới. Đây là hạn chế cấu hình, không phải giới hạn của engine.

### MTP depth 4 (image B, mem 0.88)

| Entropy | conc | Run / node | tput | TTFT (s) | TPOT (ms) | ITL (ms) | p99 ITL (ms) | E2E (s) | Accept length | Accept rate |
|---|---|---|---|---|---|---|---|---|---|---|
| hi | 1 | 37601519336, hcm `8x_02` | 116.9 | 0.67 | 8 | 18 | 23 | 8.5 | 2.32 | 33.1% |
| hi | 8 | 37601519336, hcm `8x_02` | 448.7 | 0.72 | 15 | 25 | 651 | 16.3 | 2.36 | 34.0% |
| hi | 32 | 37738313355, Greennode `1x_01` | 721.7 | 13.36 | 29 | 35 | 722 | 43.9 | 2.37 | 34.3% |
| hi | 64* | 37738313355, Greennode `1x_01` | 787.5 | 42.03 | 40 | 46 | 742 | 82.6 | 2.38 | 34.6% |
| lo | 1 | 37738313355, Greennode `1x_01` | 199.5 | 0.72 | 4 | 18 | 23 | 5.2 | 4.15 | 78.6% |
| lo | 8 | 37738317822, Greennode `1x_02` | 601.6 | 0.78 | 10 | 25 | 718 | 11.3 | 4.19 | 79.7% |
| lo | 32 | 37738317822, Greennode `1x_02` | 925.7 | 12.20 | 22 | 35 | 825 | 34.4 | 4.14 | 78.5% |
| lo | 64* | 37738317822, Greennode `1x_02` | 971.7 | 34.44 | 31 | 46 | 1184 | 66.1 | 4.11 | 77.9% |

Kiểm chứng chéo lo cell trên hcm `8x_02` (run 37735723272): c1 190.2, c8 598.7, c32 891.5, c64 948.9 tok/s, lệch dưới 4% so với Greennode. Accept length tương đương (4.12–4.14).
Cell hi c64 trên hcm (cùng run, xong sau): 778.5 tok/s, TTFT 41.6 s (median), TPOT 39 ms, ITL median 45.7 ms, p99 ITL 736 ms, E2E 81.5 s, accept length 2.39; lệch 1.1% so với Greennode (787.5). Cả 8 cell MTP của run này đã xong.

Accept length tính bằng tổng output tokens / `sglang:spec_verify_calls`, rate = (length − 1) / 4. Trường `spec_accept_length` trong `result.json` chỉ là gauge của batch cuối, không phải trung bình, nên không dùng.

**Lưu ý c64 (`*`)**: với MTP, SWA pool được tính theo trường hợp xấu nhất tại `--max-running-requests`. Ở 64 slot cần khoảng 107 GiB nên startup abort. Vì vậy MTP c64 chạy với `--max-running-requests 32` và `SGLANG_SWA_EVICTION_INTERVAL=32`, tức thực chất chỉ 32 request đồng thời. Khi so với vLLM c64 (64 slot) phải ghi chú điều này; số c64 của MTP không phải so sánh công bằng.

## Bug MTP + FA4 (`CUDA illegal memory access`) và bản fix

**Triệu chứng.** Trên image 2d74309 (v0.5.21 + PR aux #39286 + PR FA4 #42019), cả 8 cell MTP của sweep gốc (run 37298154251, `hardware-hcm-8x_02`) đều chết giữa chừng với `CUDA illegal memory access`. 8 cell base cùng run chạy bình thường. FA4 bắt buộc phải giữ nên không đổi backend để né lỗi.

**Nguyên nhân.** Khi bật MTP, SGLang tự nâng NEXTN thành `FROZEN_KV_MTP`: draft `google/gemma-4-31B-it-assistant` không có KV riêng mà đọc thẳng KV của target. `FlashAttentionBackend.__init__` suy ra `use_sliding_window_kv_pool` từ pool của chính nó. Với draft worker, đó là `MHATokenToKVPool` thường nên cờ thành `False`. Trong khi pool thật của target là `SWAKVPool` (Gemma-4 có lớp sliding window), page table chứa slot theo đánh số full-pool. Backend không dịch slot full-pool sang slot SWA, nên index full-pool bị dùng để đọc buffer SWA nhỏ hơn: truy cập ngoài biên. Các backend flashinfer, trtllm_mha và aiter đã lấy pool từ allocator cho draft frozen-KV; chỉ FA còn sót.

**Fix** (`python/sglang/srt/layers/attention/flashattention_backend.py`, commit `8c1384a796` trên nhánh `gemma4-v0.5.21-aux-fa4-swapool`, cũng có ở `fix/fa-frozen-kv-mtp-swa-pool`): thêm `_resolve_swa_kv_pool`, giải quyết pool SWA một lần trong `__init__` (lấy từ allocator khi là draft của FROZEN_KV_MTP, các draft khác giữ hành vi cũ) và dùng `self._swa_kv_pool` ở 13 chỗ dịch slot. Có thêm unit test `test/registered/unit/spec/test_resolve_swa_kv_pool.py`.

**Kiểm chứng.**
- Hai job 1 GPU H200 (cuda graph và eager): 0 lỗi illegal memory, accept length khoảng 3.0–3.4. Bench ngắn, không so sánh được với bảng 16 cell.
- Trên 8 cell MTP của sweep (image B, `3a7d5ad`, không có aux): tất cả chạy hết ma trận, không còn crash. Số liệu ở bảng MTP phía trên.
- Image A (`...-aux-fa4-swapool-8c1384a`, có aux) và B (không aux) chênh nhau dưới 1.5%: bản fix này đủ, PR aux không ảnh hưởng đến hiệu năng.

**Kiểm chứng trên `main` + fix (PR upstream #43542).** Nhánh `fix/fa-frozen-kv-mtp-swa-pool-main` (commit `d639cf1b57`, từ `main` đã có FA4 PR #42019, không có aux), H200 1 GPU, `RedHatAI/gemma-4-31B-it-FP8-block`, `--attention-backend fa4`. Lỗi là latent bug của FA backend, FA4 PR chỉ làm nó lộ ra. Fix dùng `self._swa_kv_pool` ở 11 chỗ.
- GSM8K (lm-eval, temp 0): no-spec 0.9712 ±0.0046, MTP4 0.9719 ±0.0045, tức hai arm cho điểm như nhau; 0 dòng illegal memory.
- SpeedBench throughput_8k, tok/s (no-spec so với MTP4): high entropy c8 261.9 / 328.0, c32 620.3 / 733.8; low entropy c8 254.7 / 438.7, c32 615.4 / 922.2. Accept length MTP4: high 2.19 (c8) và 2.33 (c32), low 4.35 và 3.49; accept rate suy ra `(len-1)/4`, đã đối chiếu với `/metrics` (4.257 ra 0.814). 0 dòng illegal memory ở cả 6 cell.
- Cột TTFT p50 của arm no-spec (4.2 s ở c8, MTP4 là 0.7 s) chưa giải thích được, không dùng để kết luận.
- Trạng thái PR: đã mở draft rồi chuyển ready for review; label `[]`, `pr-gate` cần `run-ci` do maintainer gắn. Branch chỉ out-of-date so với base, không conflict; khi cần cập nhật thì rebase, không dùng "Update branch". Số liệu thô: `/shared/phucnlt2/sgl-swapool-t/` (job 2738-2743).

**Chưa làm.** Chờ `run-ci` để xem CI chạy test (lint đã xong). `SWAPOOL_FINDINGS.md` không còn trong repo sglang `main`, không cần cập nhật.

## KV pool ở base c64: nguyên nhân và probe

Hybrid SWA của Gemma-4 31B: 10 lớp full attention (~76 KB/token) và 50 lớp sliding window (cửa sổ 1024, ~763 KB/token). Trong SGLang v0.5.21, SWA pool được tính theo trường hợp xấu nhất, không theo `--swa-full-tokens-ratio`:

```
per_request = window + SGLANG_SWA_EVICTION_INTERVAL × draft_tokens + page + decode_alloc
swa_pool    = max_running_requests × per_request + 2 × chunked_prefill_size
```

Phần bộ nhớ còn lại cho full pool. Ở base c64 mặc định, SWA pool chiếm khoảng 81 GB trong ~90 GB pool nhưng thực tế chỉ dùng ~11%, nên full pool chỉ còn 115k tokens (vLLM có 197k).
`SGLANG_SWA_EVICTION_INTERVAL` (mặc định 128) là số token tối thiểu đã trượt khỏi cửa sổ trước khi KV SWA được giải phóng. Giá trị nhỏ hơn thì SWA pool nhỏ hơn, không ảnh hưởng độ chính xác model, chỉ tốn thêm chút công eviction. Với MTP, `draft_tokens = 5` nên interval có ảnh hưởng gấp 5 lần.

### Kết quả probe (base, hi, c64; PROBE, một mẫu mỗi cấu hình)

| | Baseline (0.88, evict 128, chunk 16384) | a: mem 0.92 | c cũ: 0.92 + evict 32 (chunk vẫn 16384) | c mới: 0.92 + evict 32 + chunk 8192 |
|---|---|---|---|---|
| Run | 37298154251 | 37745925506 | 37745935978 | 37754283939 |
| Node | `8x_02`, image 2d74309 | `8x_01`, image B | `8x_01`, image B | `8x_01`, image B |
| Full pool (tokens) | 115,525 | 188,502 | 249,942 | 413,782 |
| SWA pool (tokens) | 106,561 | 106,561 | 100,417 | 84,033 |
| tput (tok/s) | 362 | 510 | 565 | 614 |
| TTFT (s) | 136 | 84 | 61 | 24 |
| TPOT (ms) | 29 | 35 | 43 | 65 |
| ITL (ms) | 24 | 24 | 27 | 37 |
| p99 ITL (ms) | 32 | 62 | 616 | 738 |
| E2E (s) | 165 | 119 | 106 | 93 |
| KV-full retract | 45 | 27 | 35 | 51 |
| Bộ nhớ trống sau startup | 16.6 GB | 10.8 GB | 10.8 GB | 10.8 GB |

Đọc kết quả:

- Tăng mem 0.92 (+5 GB) dồn hết vào full pool (+63% tokens), vì SWA pool bị giới hạn theo công thức trên.
- Mỗi bước tăng throughput: mem 0.92 +41%, eviction 32 thêm +11%, chunk 8192 thêm +9% (tổng +70%).
- TPOT và p99 ITL tăng theo, vì nhiều request chạy song song hơn và prefill chunk nhỏ xen vào decode nhiều hơn.
- c64 vẫn có retract (51), tức vẫn bị giới hạn KV (413k tokens đủ khoảng 45 request đầy ngữ cảnh).
- Không có OOM ở mem 0.92 (bộ nhớ trống còn 10.8 GB).
- Lưu ý khi so sánh: baseline chạy trên node và image khác; một mẫu mỗi cấu hình; `--chunked-prefill-size 8192` không có tương đương trực tiếp trong vLLM, phải ghi rõ nếu đưa vào bảng so sánh.
- Phát hiện sai sót: lần chạy "c cũ" thực tế dùng chunk 16384 do lỗi script (hard-code sau một lệnh `sed`), nên nó đo mem 0.92 + eviction 32. Đã sửa (commit 68aef722) và chạy lại thành "c mới".

## Dự kiến cho MTP ở mem 0.92

Log 6 cell MTP ở mem 0.88 (không có OOM): bộ nhớ trống sau startup c1 34.5 GB, c8 31.8 GB, c32 22.6 GB, c64 (32 slot) 26.0 GB; KV pool c1 590k, c8 506k, c32 219k, c64 329k tokens. Nâng lên 0.92 lấy thêm khoảng 5.6 GB, nên c32 còn khoảng 17 GB: ước lượng là không OOM (chưa kiểm chứng).
Ước lượng cho 64 slot: SWA pool tăng thêm khoảng 41k tokens × 763 KB ≈ 31 GB, trong khi mem 0.92 chỉ thêm 5.6 GB, nên gần như hết full pool. Dự đoán startup abort hoặc kém hơn 32 slot. Hai smoke bên dưới để kiểm chứng.

### Smoke MTP hi ở mem 0.92 (SMOKE, đã xong)

| Smoke | Run | Node | Cấu hình | Full pool | SWA pool | Tput (tok/s) | TTFT med (s) | TPOT (ms) | ITL med (ms) | p99 ITL (ms) | E2E (s) | Accept len | Retract |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d | 37760820543 | Greennode `1x_01` | c32, max-running 32, evict mặc định 128 | 271,422 | 86,337 | 739.0 | 6.27 | 37 | 44 | 739 | 43.9 | 2.37 | 8 |
| e | 37760825959 | Greennode `1x_02` | c64, max-running 64, evict 32 | 107,512 | 109,185 | 533.8 | 100.0 | 20 | 31.7 | 674 | 120.4 | 2.38 | 9 |

Cả hai không OOM và không abort. Bộ nhớ trống sau startup: d 18.6 GB, e 13.3 GB.
Smoke e (64 slot) kém hơn hẳn: SWA pool chiếm gần hết bộ nhớ nên full pool chỉ còn 107k tokens (so với 329k ở 32 slot mem 0.88), TTFT 100 s. TPOT thấp (20 ms) vì chỉ một phần request đang decode cùng lúc, còn phần còn lại xếp hàng chờ. Kết luận: giữ 32 slot cho MTP c64, đúng như dự đoán ở trên. Quá trình capture CUDA graph ở e mất 505 s, lâu bất thường.
Smoke d so với mem 0.88 (721.7 tok/s, TTFT 13.36 s, TPOT 29 ms, ITL 35 ms, 218,616 tokens): pool +24%, tput +2.4%, TTFT giảm một nửa, TPOT/ITL cao hơn. Mỗi bên chỉ một mẫu, khác node, nên sai khác nhỏ chỉ mang tính tham khảo. Accept length tính bằng tổng output token chia `spec_verify_calls`.

Đối chiếu ở mem 0.88: c32 có 218,616 tokens, c64 (32 slot) có 329,009 tokens.

### Smoke tuned (mem 0.92 + evict 32 + chunk 8192) (SMOKE, đã xong)

| Smoke | Run | Node | Cấu hình | Full pool | SWA pool | Tput (tok/s) | TTFT mean (s) | TPOT (ms) | ITL med / mean (ms) | p99 ITL (ms) | E2E (s) | Accept len | Retract |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f | 37767038996 | Greennode `1x_01` | MTP hi c32, 32 slot | 498,164 | 54,593 | 749.9 | 2.04 | 40 | 45.8 / 95.8 | 727 | 43.0 | 2.39 | 20 |
| g | 37767042768 | Greennode `1x_02` | base hi c8 | 1,004,709 | 24,841 | 323.8 | 2.64 | 22 | 18.6 / 22.2 | 27 | 25.3 | n/a | 12 |

Không OOM. Bộ nhớ trống sau startup: f 25.7 GB, g 10.9 GB.
- f so với smoke d (0.92, evict 128, chunk 16384: 739.0 tok/s, TTFT 6.27 s, pool 271k) và mem 0.88 (721.7 tok/s, TTFT 13.36 s, pool 219k): full pool 498k (gấp 2.3 lần mem 0.88), tput +1.5% so với d và +3.9% so với 0.88, TTFT mean 2.0 s (cột TTFT của d và 0.88 ở trên là median, nên số TTFT này chỉ so sánh tham khảo). TPOT 40 ms cao hơn 29 ms của 0.88 và ITL mean 95.8 ms bị kéo lên bởi 20 lần retract; p99 ITL không đổi (727 ms so với 722 ms).
- g so với base hi c8 ở 0.88 (324.7 tok/s, TTFT 3.23 s, TPOT 21 ms, ITL 19 ms, p99 ITL 26 ms): gần như không đổi (323.8, 2.64 s, 22 ms, 22.2 ms mean, 27 ms). Tức cấu hình tuned không làm xấu conc thấp, nhưng cũng không giúp gì ở c8 vì pool vốn đã dư.
- Mỗi smoke một mẫu, khác node với baseline, nên chênh dưới vài % không có ý nghĩa. Kết luận: mem 0.92 + evict 32 + chunk 8192 an toàn trên cả MTP c32 và base c8; lợi ích rõ chỉ ở cell bị giới hạn KV pool (base c64: 362 lên 614 tok/s).

### SGLang tuned, 16 cell (FULL, commit e46c9d30, image B, mem 0.92 + evict 32 + chunk 8192)

Chạy trên Greennode `1x_01` (c1, c64) và `1x_02` (c8, c32), 8 run: 37768891961, 37768896365, 37768901567, 37768906322, 37768911378, 37768916464, 37768921291, 37768926650. 16/16 cell passed, không OOM. MTP c64 dùng 32 slot. Nhãn "SGLang tuned", không so like-for-like với vLLM (chunk 8192 không có tương đương). Cột "so với 0.88" là tput của bảng gốc (Base: run 37298154251 trên hcm; MTP: bảng ở trên).

| Arm | Entropy | conc | Tput | so với 0.88 | TTFT med (s) | TPOT (ms) | ITL med (ms) | p99 ITL (ms) | E2E (s) | Accept | Full / SWA pool | Retract |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | hi | 1 | 66.9 | 66.2 | 0.70 | 14 | 14 | 20 | 15.3 | | 1,079,692 / 17,442 | 0 |
| base | hi | 8 | 323.2 | 324.7 | 2.58 | 21 | 19 | 27 | 25.3 | | 1,004,709 / 24,841 | 13 |
| base | hi | 32 | 610.6 | 647.5 | 2.67 | 47 | 28 | 1254 | 53.6 | | 751,029 / 50,209 | 51 |
| base | hi | 64 | 615.7 | 361.9 | 23.33 | 65 | 37 | 790 | 103.0 | | 413,782 / 84,033 | 60 |
| base | lo | 1 | 66.6 | 66.0 | 0.74 | 14 | 14 | 21 | 15.4 | | 1,079,692 / 17,442 | 0 |
| base | lo | 8 | **133.1** | 324.8 | 2.45 | 22 | 19 | 818 | 61.4 | | 1,004,709 / 24,841 | **475** |
| base | lo | 32 | **98.6** | 634.2 | 2.61 | 321 | 29 | 1607 | 331.1 | | 751,029 / 50,209 | **3547** |
| base | lo | 64 | 591.4 | 349.6 | 26.02 | 66 | 37 | 931 | 107.6 | | 413,782 / 84,033 | 71 |
| mtp | hi | 1 | 119.6 | 116.9 | 0.70 | 8 | 18 | 24 | 8.6 | 2.33 | 762,550 / 17,579 | 0 |
| mtp | hi | 8 | 463.9 | 448.7 | 0.73 | 16 | 25 | 594 | 17.1 | 2.34 | 702,140 / 25,937 | 10 |
| mtp | hi | 32 | 768.5 | 721.7 | 0.78 | 40 | 45 | 711 | 42.0 | 2.37 | 497,455 / 54,593 | 9 |
| mtp | hi | 64 | 769.0 | 787.5 | 42.72 | 41 | 46 | 721 | 82.6 | 2.39 | 498,164 / 54,593 | 13 |
| mtp | lo | 1 | 200.1 | 199.5 | 0.74 | 4 | 18 | 23 | 5.1 | 4.17 | 762,550 / 17,579 | 0 |
| mtp | lo | 8 | 613.2 | 601.6 | 0.78 | 11 | 26 | 600 | 13.1 | 4.15 | 702,140 / 25,937 | 16 |
| mtp | lo | 32 | 918.0 | 925.7 | 0.86 | 32 | 46 | 832 | 35.3 | 4.13 | 497,455 / 54,593 | 19 |
| mtp | lo | 64 | 931.4 | 971.7 | 35.51 | 33 | 46 | 839 | 68.4 | 4.10 | 498,164 / 54,593 | 9 |

Bộ nhớ trống sau startup: base c1 10.9 GB, c64 10.8 GB; MTP c1 34.2 GB, c64 25.7 GB.

Kết luận:
- **Base c64 tăng mạnh**: hi 362 lên 616 tok/s (+70%), lo 350 lên 591 (+69%), TTFT 136 s xuống 23 s. Đây là cell bị nghẽn KV pool nên đúng như probe.
- **MTP: gần như không đổi** (trong khoảng -4% đến +7%, một mẫu mỗi cell, khác node với baseline). TTFT c8/c32 rất thấp (0.7-0.9 s) vì chunk 8192, TPOT/ITL cao hơn nhẹ ở c32/c64.
- **Base lo c8 và c32 bị thụt lùi nặng**: 133 tok/s (so với 325) và 98.6 tok/s (so với 634), kèm 475 và 3547 lần retract. Log cho thấy SWA token usage 0.66-0.79 trong khi full token usage chỉ 0.1: SWA pool nhỏ (24.8k ở c8, 50.2k ở c32) mới là nút cổ chai, request liên tục bị retract rồi prefill lại. Base hi c8 và c32 cùng cấu hình chỉ 13 và 51 retract (c32 giảm 6% tput), nên hiện tượng này không xảy ra đều giữa các cell; chưa xác định nguyên nhân (nghi ngờ mức độ ngẫu nhiên của thời điểm request đến cùng SWA pool quá sát). Cần chạy lại các cell này, hoặc nới SWA pool (tăng `--max-running-requests` hay eviction lớn hơn), trước khi tin vào cấu hình tuned cho base c8/c32.
- Vì vậy cấu hình tuned **chưa nên là mặc định chung**: tốt cho base c64, an toàn cho MTP, nhưng gây lỗi hiệu năng ở base lo c8/c32.

### Điều tra base lo c8/c32 và sweep mem-only (FULL cho mem-only, PROBE cho phần còn lại)

Image B, Greennode `1x_01/_02`. Cột "0.88" là bảng gốc, "tuned" là sweep chunk 8192 ở trên. Mem-only = mem 0.92, eviction mặc định, chunk 16384 (commit 4a3a5431, run 37807812093 / 37807817494 / 37807823121 / 37807828318, 16/16 cell passed).

| Arm | Entropy | conc | 0.88 | tuned (0.92 + evict 32 + chunk 8192) | **mem-only (0.92 + chunk 16384)** | TTFT (s) | TPOT (ms) | ITL mean (ms) | p99 ITL (ms) | E2E (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| base | hi | 1 | 66.2 | 66.9 | 67.5 | 0.68 | 14.1 | 14.2 | 19 | 15.2 |
| base | hi | 8 | 324.7 | 323.2 | 325.4 | 3.38 | 21.4 | 21.4 | 26 | 25.2 |
| base | hi | 32 | 647.5 | 610.6 | 641.5 | 6.30 | 43.9 | 43.9 | 98 | 51.1 |
| base | hi | 64 | 361.9 | 615.7 | 515.1 | 82.5 | 34.1 | 37.9 | 76 | 121.2 |
| base | lo | 1 | 66.0 | 66.6 | 66.8 | 0.70 | 14.3 | 14.4 | 20 | 15.3 |
| base | lo | 8 | 324.8 | **133.1** | 328.7 | 3.38 | 21.1 | 21.2 | 26 | 24.9 |
| base | lo | 32 | 634.2 | **98.6** | 632.9 | 6.04 | 44.8 | 44.9 | 124 | 51.8 |
| base | lo | 64 | 349.6 | 591.4 | 489.1 | 86.3 | 35.4 | 40.8 | 74 | 127.8 |
| mtp | hi | 1 | 116.9 | 119.6 | 120.9 | 0.68 | 7.7 | 17.6 | 23 | 8.5 |
| mtp | hi | 8 | 448.7 | 463.9 | 492.7 | 1.14 | 14.9 | 34.7 | 654 | 16.1 |
| mtp | hi | 32 | 721.7 | 768.5 | 746.3 | 6.99 | 36.4 | 83.5 | 730 | 43.1 |
| mtp | hi | 64 | 787.5 | 769.0 | 794.6 | 40.5 | 39.7 | 91.5 | 734 | 79.8 |
| mtp | lo | 1 | 199.5 | 200.1 | 197.8 | 0.71 | 4.4 | 17.9 | 23 | 5.2 |
| mtp | lo | 8 | 601.6 | 613.2 | 672.4 | 1.20 | 10.4 | 43.4 | 728 | 11.9 |
| mtp | lo | 32 | 925.7 | 918.0 | 910.7 | 7.58 | 27.8 | 113.1 | 1199 | 35.5 |
| mtp | lo | 64 | 971.7 | 931.4 | 956.9 | 34.1 | 31.6 | 130.5 | 1207 | 66.5 |

Các cột tput là output tok/s mỗi GPU; TTFT là mean (s); TPOT là median (ms). Pool size, số retract và accept length của sweep mem-only chưa trích từ server log.

Điều tra base lo c8/c32 (mỗi cấu hình một mẫu):

| Run | Cấu hình | base lo c8 | base lo c32 |
|---|---|---|---|
| 37805406642 | tuned chạy lại (0.92 + evict 32 + chunk 8192) | 127.5 | 118.5 |
| 37805413643 | biến thể (0.92 + evict 32 + chunk 16384) | 329.4 | 636.8 |

- **Lỗi thụt lùi lặp lại ở chunk 8192** (127.5 và 118.5 tok/s, ITL mean 61 và 267 ms, p99 ITL 830 và 1546 ms), nên không phải ngẫu nhiên.
- **Chunk 16384 hết lỗi**: biến thể chỉ khác tuned ở chunk (vẫn evict 32, mem 0.92) mà quay về 329 và 637 tok/s, ngang 0.88 và mem-only. Vậy thủ phạm là chunk 8192, không phải mem 0.92 hay evict 32. Nhất quán với công thức pool: SWA pool có số hạng `2 × chunked_prefill_size`, chunk nhỏ làm pool nhỏ (24.8k ở c8, 50.2k ở c32) nên retract liên tục. Cơ chế vì sao chỉ entropy thấp bị nặng (base hi cùng cấu hình chỉ mất 6% ở c32) vẫn chưa giải thích được.
- **Mem-only (chunk 16384) là cấu hình an toàn**: base c1/c8/c32 ngang 0.88 (trong ±1-2%), base c64 tăng 42% (hi 362 lên 515, lo 350 lên 489), MTP không giảm (MTP hi/lo c8 tăng 10%/12%, các cell khác trong ±3%). Tuned vẫn nhỉnh hơn ở base c64 (616/591) nhưng đánh đổi bằng lỗi ở c8/c32.

### So pool và retract: 0.88 và mem-only (FULL, đọc từ 12 log server base, chunk 16384 cả hai)

Baseline 0.88 là run 37298154251 (`8x_02`, image 2d74309); mem-only là các run 37807812093 / 37807823121 (Greennode, image B).

| Cell | Full pool 0.88 / mem 0.92 | SWA pool (cả hai) | Retract 0.88 / 0.92 | Peak running 0.88 / 0.92 |
|---|---|---|---|---|
| hi c8 | 761,205 / 834,182 | 41,993 | 8 / 8 | 8 / 8 |
| lo c8 | 761,205 / 833,189 | 41,993 | 4 / 4 | 8 / 8 |
| hi c32 | 484,485 / 557,462 | 69,665 | 13 / 10 | 32 / 32 |
| lo c32 | 484,485 / 556,469 | 69,665 | 13 / 13 | 32 / 32 |
| hi c64 | 115,525 / 188,502 | 106,561 | 45 / 25 | 22 / 28 |
| lo c64 | 115,525 / 187,509 | 106,561 | 27 / 33 | 21 / 28 |

- Mem 0.92 chỉ tăng full pool, SWA pool không đổi (bị công thức chặn). Ở c8/c32 full pool đã rất lớn nên không đổi gì (tput ngang nhau); ở c64 full pool +63%, request chạy đồng thời 22 lên 28, đó là nguồn của +42% tput.
- c64 vẫn bị giới hạn KV (retract 25-33, peak running 28 trên 64). Retract ở c8 (4-8 lần) có ở cả 0.88 nên không phải hồi quy. Dòng log `KV cache pool is full` không nói pool nào đầy; nút cổ chai thật của c64 (full hay SWA) chưa tách được.

**Đối chiếu vLLM c64 (base, run 37259992014 hi, log đã đọc; vLLM v0.30.0 mem 0.92).** `GPU KV cache size: 197,683 tokens` (con số này là `3.02 × max_model_len 65,536`, một năng lực quy đổi chứ không phải số token đếm được, nên nó chỉ tình cờ gần full pool SGLang 188k), nhưng peak running 58 (hi) và 58 (lo), peak KV usage 99.5% / 98.3%, 0 dòng preempt trong log. SGLang cùng pool cỡ này chỉ chạy đồng thời 28 request và retract 25-33 lần. Hai con số "tokens" không so trực tiếp được: vLLM dùng hybrid KV manager chung một pool block cho các nhóm lớp, còn SGLang tách full pool và SWA pool với SWA pool tính theo trường hợp xấu nhất. Đọc code (SGLang nhánh `gemma4-v0.5.21-aux-fa4-swapool` trong `/home/phucnlt2/sglang`, vLLM tag v0.30.0) thì giả thuyết "do scheduler" chỉ đúng một phần:

- **Admission của hai bên đều lạc quan về output.** SGLang (`PrefillAdder`, `SWAPrefillBudget.check_prefill`) cộng `max_new_tokens × new_token_ratio` (init 0.7, giảm dần về 0.098 sau 600 bước) vào offset rồi `retract_decode` khi `check_decode_mem` thất bại; log thấy ratio nhảy `0.098 -> 0.5-0.9` sau mỗi lần retract. vLLM mặc định `scheduler_reserve_full_isl=True` (`allocate_slots`, `full_sequence_must_fit`), tức bắt buộc cả prompt (ISL) vừa, SWA bị chặn ở cửa sổ, nhưng không dành chỗ cho OSL, `watermark=0`, và vẫn preempt khi decode hết block. Vậy vLLM cũng có thể preempt: peak KV usage 99.5% cho thấy nó chạy sát đầy, chỉ là log không có dòng preempt nên chưa biết đếm được bao nhiêu (cần metrics, chưa có).
- **Khác biệt lớn là cách chia pool.** vLLM dùng một block pool chung cho các nhóm lớp; block SWA ngoài cửa sổ được giải phóng ngay mỗi bước (`remove_skipped_blocks`), nên mỗi request chỉ giữ khoảng 8.2k+1k token cho 10 lớp full (76 KB/token) cộng 1024 token cho 50 lớp SWA (763 KB/token), tức khoảng 1.5 GB/request, ~100 GB chứa được khoảng 65 request (khớp peak running 58). SGLang chia cứng lúc khởi động: SWA pool = 64 × ~1.15k + `2 × chunk 16384` = 106,561 token ≈ 81 GB, full pool chỉ còn 14 GB = 188k token ≈ 20 request đầy ngữ cảnh. Phần `2 × chunk` (32.8k token ≈ 25 GB) là phần lớn nhất và chỉ dùng cho prefill chunk, nên giảm chunk 8192 làm full pool 188k lên 413k (đúng bảng probe), còn chunk 16384 thì SWA pool vẫn chiếm 81 GB mà thực tế chỉ dùng ~11% vì chỉ có 20-28 request chạy.
- Số liệu 1.5 GB/request và 65 request là ước lượng từ log (KV size, 76/763 KB/token), chưa đo trực tiếp ở vLLM. Giả thuyết này chưa kiểm chứng bằng thí nghiệm; `peak running` ở hai bên là đỉnh, không phải trung bình. Lưu ý vLLM c64 không tốt hoàn toàn: p99 ITL 1260 ms và TPOT 75 ms, tput 695 so với SGLang mem-only 515.

### Probe A-G trên mem 0.92 (PROBE, một cell mỗi cấu hình, commit 1aee9614)

| Probe | Thay đổi | Run | Kết quả |
|---|---|---|---|
| A | `--kv-cache-dtype fp8_e4m3` (base lo c8/c32; MTP hi c32) | 37817243239, 37817250468 | **Fail khi khởi động**: `FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe`. FP8 KV với FA4 không chạy trên H200 (SM90) và SGLang không fallback. Log run MTP chưa đọc, suy ra từ run base. |
| B | `--schedule-conservativeness 1.3` + `SGLANG_RETRACT_DECODE_STEPS=40` (base lo c8/c32) | 37817257022 | c8 329.0, c32 634.2 tok/s, y hệt mem-only (328.7, 632.9). Không có lỗi retract ở mem-only nên probe này không có gì để cải thiện; chưa kết luận được tác dụng khi bị retract. |
| C | FA4 prefill + flashinfer decode (base hi c8; MTP hi c32) | 37817263225, 37817269333 | **Fail khi khởi động**: `Gemma4 only supports trtllm_mha, triton, fa4, ascend, intel_xpu, intel_amx, or aiter attention backend, got prefill=fa4, decode=flashinfer`. Flashinfer bị chặn cho Gemma-4. Log run MTP chưa đọc, suy ra từ run base. |
| D | MTP depth 2 / 3 (hi c32, lo c8) | 37817274835, 37817280979, 37817286778, 37817292497 | Xem bảng dưới. |
| E | `--cuda-graph-max-bs-decode 128` (base hi c64) | 37817299637 | 515.0 tok/s, TTFT 82.1 s, TPOT 33.9 ms; y hệt mem-only (515.1). Không có tác dụng. |
| F | Bật radix cache (base hi c8) | 37817305924 | 333.1 tok/s so với 325.4 (+2.4%), TTFT 3.49 s, TPOT 20.5 ms. Một mẫu, nằm trong nhiễu; chưa đọc hit rate. |
| G | `--fp8-gemm-runner-backend cutlass` / `triton` (base hi c8) | 37817312675, 37817319069 | **Fail khi parse tham số do lỗi của tôi**: flag trong image tên `--fp8-gemm-backend` (tên lấy từ source local lệch với image). Chưa đo được gì; cần dispatch lại với tên đúng. Log run triton chưa đọc, suy ra từ run cutlass. |

Probe D (MTP depth, output tok/s; depth 4 là mem-only):

| Entropy | conc | depth 2 | depth 3 | depth 4 | p99 ITL d2 / d3 / d4 (ms) |
|---|---|---|---|---|---|
| hi | 32 | **818.8** | 812.0 | 746.3 | 726 / 733 / 730 |
| lo | 8 | 589.8 | 641.7 | **672.4** | 634 / 710 / 728 |

- Entropy cao nên dùng depth thấp (depth 2 hơn depth 4 khoảng 10% ở c32, TTFT 2.1 s so với 7.0 s), entropy thấp nên dùng depth sâu (depth 4 hơn depth 2 khoảng 14% ở c8). Mỗi cell một mẫu. Chưa chạy c1/c64 và chưa tính accept length.
- Bài học: kiểm tra tên flag bằng `launch_server --help` của chính image trước khi dispatch, không dựa vào source local; và kiểm tra backend mà model cho phép (Gemma-4 chỉ có `fa4`, `triton`, `trtllm_mha` trên NVIDIA).

### SGLang 256k + radix cache: hai cấu hình, 32 cell (FULL, commit b2c9f949 và b0208a66, 2026-10-09)

Từ 2026-10-09 mọi cell phục vụ `--context-length 262144` và bật prefix cache (vLLM `--enable-prefix-caching`, SGLang radix, không còn `--disable-radix-cache`). Với SGLang việc bật radix làm mất cách tính pool theo số request (xem mục "Pool khi bật radix" dưới), pool giờ tính theo `--swa-full-tokens-ratio`. Hai cấu hình, mỗi cấu hình 16 cell, base/MTP × hi/lo × conc 1/8/32/64, image B, Greennode `1x_01` (hi) và `1x_02` (lo):

- **A** (commit b2c9f949, run 37880075115 / 37880078321): mem 0.92, chunk 16384, ratio **0.3**, eviction mặc định, FCFS. Là mem-only cũ cộng 256k, radix và ratio 0.3 (ratio 0.3 do tôi chọn, chưa được đối chiếu trước đó).
- **B** (commit b0208a66, run 37884261982 / 37884264773): copy cấu hình serving của run tau2 37879258874 (SGLang MTP4, commit cd19d215): mem 0.92, ratio **0.2**, chunk **8192**, `--schedule-policy hrrn`, `--enable-cache-report`, `SGLANG_SWA_EVICTION_INTERVAL=32` cho mọi cell. `--max-running-requests` vẫn theo conc (tau2 cố định 32), MTP c64 vẫn 32 slot.

32/32 cell `passed` (320/320 request mỗi cell, 0 failed).

**Đọc cột trong bảng hiệu năng dưới đây:**
- Ba cột tên run (**mem-only**, **A**, **B**) đều là **tput = output throughput mỗi GPU, đơn vị tok/s** (token sinh ra trên giây trên 1 GPU; càng cao càng tốt). Cùng cell (arm, entropy, conc), chỉ khác cấu hình serving.
- "mem-only" là bảng cũ phía trên (khác node, 65k context, radix tắt, cap-mode), nên A và B so với nó là hai thay đổi cùng lúc (256k + radix) cộng ratio.
- Các cột còn lại, mỗi cột ghi `A / B`: TTFT = mean, giây (s); TPOT = median, mili-giây (ms); ITL mean = mili-giây (ms); p99 ITL = mili-giây (ms); E2E = mean latency cả request, giây (s); Accept = accept length trung bình của MTP, đơn vị token trên mỗi bước verify (chỉ có ở arm mtp).

| Arm | Entropy | conc | tput mem-only (tok/s; 0.92, chunk 16384, radix tắt) | tput **A** (tok/s; 256k + radix, ratio 0.3, chunk 16384) | tput **B** (tok/s; cấu hình tau2) | TTFT A / B (s) | TPOT A / B (ms) | ITL mean A / B (ms) | p99 ITL A / B (ms) | E2E A / B (s) | Accept A / B (token/bước) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | hi | 1 | 67.5 | 67.0 | 66.9 | 0.68 / 0.70 | 14 / 14 | 14 / 14 | 20 / 20 | 15.3 / 15.3 |  |
| base | hi | 8 | 325.4 | 333.4 | 332.5 | 3.48 / 3.34 | 20 / 21 | 21 / 21 | 25 / 25 | 24.6 / 24.6 |  |
| base | hi | 32 | 641.5 | 625.0 | 652.7 | 11.08 / 11.05 | 38 / 38 | 39 / 38 | 37 / 36 | 50.7 / 50.2 |  |
| base | hi | 64 | 515.1 | 639.1 | 622.7 | 57.58 / 43.72 | 40 / 56 | 41 / 56 | 37 / 109 | 99.4 / 100.9 |  |
| base | lo | 1 | 66.8 | 66.6 | 66.5 | 0.69 / 0.70 | 14 / 14 | 14 / 14 | 20 / 21 | 15.4 / 15.4 |  |
| base | lo | 8 | 328.7 | 334.7 | 333.6 | 3.47 / 3.27 | 20 / 21 | 21 / 21 | 26 / 26 | 24.5 / 24.5 |  |
| base | lo | 32 | 632.9 | 619.8 | 647.4 | 11.40 / 10.91 | 39 / 39 | 39 / 39 | 37 / 36 | 51.1 / 50.6 |  |
| base | lo | 64 | 489.1 | 617.9 | 623.3 | 58.46 / 44.23 | 40 / 56 | 41 / 56 | 37 / 83 | 100.6 / 100.9 |  |
| mtp | hi | 1 | 120.9 | 119.2 | 118.9 | 0.69 / 0.71 | 8 / 8 | 18 / 18 | 23 / 24 | 8.6 / 8.6 | 2.31 / 2.32 |
| mtp | hi | 8 | 492.7 | 485.4 | 479.0 | 1.15 / 1.17 | 15 / 16 | 35 / 35 | 660 / 607 | 16.4 / 16.6 | 2.35 / 2.35 |
| mtp | hi | 32 | 746.3 | 755.5 | 772.9 | 5.52 / 2.05 | 38 / 40 | 86 / 92 | 734 / 716 | 42.6 / 41.7 | 2.37 / 2.37 |
| mtp | hi | 64 | 794.6 | 763.1 | 776.4 | 45.43 / 41.55 | 38 / 41 | 88 / 93 | 737 / 719 | 83.1 / 81.7 | 2.38 / 2.38 |
| mtp | lo | 1 | 197.8 | 202.5 | 198.6 | 0.69 / 0.70 | 4 / 4 | 18 / 18 | 24 / 24 | 5.1 / 5.2 | 4.17 / 4.08 |
| mtp | lo | 8 | 672.4 | 694.0 | 685.0 | 1.19 / 1.12 | 10 / 10 | 42 / 43 | 707 / 602 | 11.6 / 11.7 | 4.15 / 4.17 |
| mtp | lo | 32 | 910.7 | 948.5 | 972.1 | 5.70 / 2.12 | 28 / 30 | 114 / 125 | 842 / 824 | 34.1 / 33.3 | 4.10 / 4.11 |
| mtp | lo | 64 | 956.9 | 943.1 | 975.0 | 37.99 / 33.63 | 28 / 31 | 118 / 127 | 1170 / 796 | 67.3 / 65.3 | 4.12 / 4.12 |

| Arm | Entropy | conc | Full / SWA pool A | Full / SWA pool B | Peak running A / B | Queue max A / B | Retract A / B | Peak full / SWA usage A | Peak full / SWA usage B |
|---|---|---|---|---|---|---|---|---|---|
| base | hi | 1 | 313,515 / 94,054 | 418,020 / 83,604 | 1 / 1 | 0 / 0 | 0 / 0 | 0.04 / 0.02 | 0.03 / 0.05 |
| base | hi | 8 | 313,515 / 94,054 | 418,020 / 83,604 | 8 / 8 | 5 / 6 | 0 / 0 | 0.27 / 0.23 | 0.20 / 0.21 |
| base | hi | 32 | 313,515 / 94,054 | 418,020 / 83,604 | 32 / 32 | 27 / 28 | 9 / 0 | 1.00 / 0.67 | 0.77 / 0.81 |
| base | hi | 64 | 313,515 / 94,054 | 418,020 / 83,604 | 33 / 45 | 59 / 59 | 21 / 46 | 1.00 / 0.69 | 1.00 / 1.00 |
| base | lo | 1 | 313,267 / 93,980 | 417,689 / 83,537 | 1 / 1 | 0 / 0 | 0 / 0 | 0.04 / 0.02 | 0.03 / 0.04 |
| base | lo | 8 | 313,267 / 93,980 | 417,689 / 83,537 | 8 / 8 | 5 / 6 | 0 / 0 | 0.28 / 0.24 | 0.21 / 0.21 |
| base | lo | 32 | 313,267 / 93,980 | 417,689 / 83,537 | 32 / 32 | 30 / 28 | 12 / 0 | 1.00 / 0.68 | 0.79 / 0.84 |
| base | lo | 64 | 313,267 / 93,980 | 417,689 / 83,537 | 34 / 45 | 58 / 60 | 22 / 34 | 1.00 / 0.71 | 1.00 / 1.00 |
| mtp | hi | 1 | 282,570 / 84,771 | 365,679 / 73,135 | 1 / 1 | 0 / 0 | 0 / 0 | 0.04 / 0.02 | 0.03 / 0.06 |
| mtp | hi | 8 | 282,570 / 84,771 | 365,679 / 73,135 | 8 / 8 | 5 / 6 | 0 / 0 | 0.29 / 0.25 | 0.22 / 0.32 |
| mtp | hi | 32 | 282,570 / 84,771 | 365,679 / 73,135 | 30 / 32 | 27 / 29 | 5 / 0 | 1.00 / 0.69 | 0.86 / 0.99 |
| mtp | hi | 64 | 282,570 / 84,771 | 365,679 / 73,135 | 30 / 32 | 59 / 59 | 4 / 0 | 1.00 / 0.66 | 0.87 / 1.00 |
| mtp | lo | 1 | 282,344 / 84,703 | 365,387 / 73,077 | 1 / 1 | 0 / 0 | 0 / 0 | 0.04 / 0.03 | 0.03 / 0.05 |
| mtp | lo | 8 | 282,344 / 84,703 | 365,387 / 73,077 | 8 / 8 | 5 / 6 | 0 / 0 | 0.30 / 0.26 | 0.23 / 0.29 |
| mtp | lo | 32 | 282,344 / 84,703 | 365,387 / 73,077 | 30 / 32 | 27 / 29 | 2 / 0 | 1.00 / 0.70 | 0.89 / 0.90 |
| mtp | lo | 64 | 282,344 / 84,703 | 365,387 / 73,077 | 30 / 32 | 59 / 59 | 6 / 0 | 0.99 / 0.69 | 0.90 / 0.88 |

Retract = số dòng `Retract requests` trong server log. Peak running / queue max / peak usage lấy từ các dòng `Decode batch` định kỳ (đỉnh có thể sót). Với radix bật, "full token usage 1.00" gồm cả cây radix có thể evict, không có nghĩa tất cả là request đang chạy.

**Pool khi bật radix.** Cả hai cấu hình có full / SWA pool **không đổi theo conc** (A: base 313,515 / 94,054, MTP 282,570 / 84,771; B: base 418,020 / 83,604, MTP 365,679 / 73,135, đều là hi; lo lệch dưới 0.2%), vì pool giờ là `swa = full × ratio` chứ không còn `max_running × 1,153 + 2 × chunk`. Tỉ lệ thực đo được là 0.300 và 0.200, đúng cờ. So với mem-only (radix tắt): ở c1/c8/c32 SWA pool **lớn hơn nhiều** (94k so với 34k / 42k / 70k) nhưng full pool nhỏ hơn (313k so với 915k / 834k / 557k); ở c64 base full pool **tăng** từ 188k lên 313k (A) và 418k (B). Ước tính 330k / 100k của tôi trước khi chạy hơi cao (thực đo 313k / 94k ở ratio 0.3).

**Hiệu năng.**
- **Base c64 (cell bị KV chặn) tăng thêm**: hi 515 lên 639 (A) và 623 (B), lo 489 lên 618 (A) và 623 (B), tức +21% đến +27%. TTFT 82-86 s xuống 58 s (A) và 44 s (B). B chạy được 45 request đồng thời so với 33-34 của A (và 28 của mem-only), nhưng tput gần như ngang A vì TPOT tăng 40 lên 56 ms (nhiều request hơn trên mỗi bước decode); p99 ITL của B (83-109 ms) cao hơn A (37 ms).
- **Base c1/c8/c32 ngang mem-only** trong khoảng ±3% (B c32: 653 / 647, A c32: 625 / 620, mem-only: 642 / 633). Không còn thụt lùi ở lo c8/c32 như chunk 8192 cũ: B cũng dùng chunk 8192 nhưng ở đây base lo c8/c32 đạt 334 / 647 tok/s với **0 retract**. Nhất quán với giả thuyết cũ: lỗi do SWA pool nhỏ ở cap-mode (24.8k / 50.2k); với radix + ratio 0.2 SWA pool là 83.6k nên hết. Lưu ý B đổi cả cách tính pool lẫn radix và hrrn, nên chưa tách riêng được yếu tố nào.
- **MTP**: B nhỉnh hơn A ở c32/c64 (hi 773 / 776 so với 756 / 763, lo 972 / 975 so với 949 / 943) và có **0 retract ở mọi cell**, trong khi A còn 2-6 lần ở c32/c64. Hơn mem-only: MTP lo c32 910.7 lên 949 (A) và 972 (B), c64 hi hơi thấp hơn mem-only 794.6 (763 và 776). Accept length ngang nhau (hi 2.3-2.4, lo 4.1-4.2).
- **MTP c64 chỉ 32 slot** nên c64 chỉ cho cùng tput với c32; TTFT c64 (34-45 s) là hàng chờ.
- **Prefix cache gần như không có tác dụng với SPEED-Bench**: tỉ lệ cached-token / (cached + new) ở các dòng `Prefill batch` chỉ 1-3% (đây là tổng ước lượng từ log, các prompt khác nhau nên không dùng lại tiền tố). Cell không còn lợi từ prefill reuse, nên các chênh lệch trên chủ yếu do pool và scheduling, không phải do cache hit.
- So với vLLM 256k + prefix cache (cùng ngày, 16/16 passed): xem mục "vLLM 256k + prefix cache" ở Phần 1. Tóm tắt: SGLang B nhanh hơn ở c1 (25-32%), ngang ở c32, vLLM base nhanh hơn ở c64 (14-15%).

Rủi ro đã ghi nhận, chưa kiểm chứng riêng: FROZEN_KV_MTP chạy cùng radix: cả 16 log MTP đều có dòng "promoting ... to FROZEN_KV_MTP" và không abort của hai cấu hình (không abort, accept length giống trước đó), nên coi là hoạt động, nhưng chưa có thí nghiệm bật/tắt radix riêng cho MTP để xác nhận không đổi về tính đúng đắn đầu ra (không chạy eval).

**Đối chiếu với run tau2 37879258874** (agentic, MTP4, c32, ratio 0.2, chunk 8192, hrrn, evict 32, 256k): pool full 365,387, SWA 73,077 (KV 83.6 GB), trùng với cấu hình B MTP lo (365,387 / 73,077). Số này xác nhận B đúng là cùng cấu hình serving với tau2. Điểm khác còn lại: tau2 đặt `--max-running-requests 32` cố định, B đặt theo conc.

## Tổng hợp KV pool: vLLM và SGLang theo từng ccu

Nguồn: dòng `Full KV Cache is allocated` / `SWA KV Cache is allocated` và `#running-req` / `Retract requests` trong log server của từng cell (FULL, đọc trực tiếp); với vLLM là `GPU KV cache size`, `Running:` và `GPU KV cache usage:`. Số SGLang là số token thật của từng pool (`#tokens` trong log). Số vLLM **không phải số token đếm trực tiếp**: theo `get_kv_cache_capacity` (`kv_cache_utils.py`, v0.30.0), `GPU KV cache size = int(max_concurrency × max_model_len)` với `max_concurrency = num_blocks / (tổng số block một request dài đúng max_model_len cần, cộng qua các nhóm lớp)`. Nên con số này phụ thuộc `max_model_len` (các run này 65,536; một cấu hình với 262,144 sẽ in số khác, ví dụ "519,675 tokens ... 262,144 tokens per request: 1.98x", không có trong log nào của chúng ta) và dòng "Maximum concurrency 3.02x" là cho request dài 65,536 token, không phải 9.2k. Hai bên **không so trực tiếp được**: vLLM dùng một pool block chung cho 6 nhóm lớp (log: `kv cache group sizes [16, 16, 16, 16, 16, 32]`), SGLang tách full pool và SWA pool. Mem: SGLang 0.88 là baseline (image 2d74309 cho base, image B cho MTP), mem-only là mem 0.92 + chunk 16384, tuned là mem 0.92 + evict 32 + chunk 8192; vLLM mem 0.92.

### SGLang base: full / SWA pool (retract hi/lo)

| ccu (`max_running`) | 0.88, chunk 16384 | mem-only 0.92, chunk 16384 | tuned 0.92, chunk 8192 |
|---|---|---|---|
| 1 | 841,915 / 33,922 (0/0) | 914,892 / 33,922 (0/0) | 1,079,692 / 17,442 (0/0) |
| 8 | 761,205 / 41,993 (8/4) | 834,182 / 41,993 (8/4) | 1,004,709 / 24,841 (13/475) |
| 32 | 484,485 / 69,665 (13/13) | 557,462 / 69,665 (10/13) | 751,029 / 50,209 (51/3547) |
| 64 | 115,525 / 106,561 (45/27) | 188,502 / 106,561 (25/33) | 413,782 / 84,033 (60/71) |
| Peak running ở c64 (hi/lo) | 22 / 21 | 28 / 28 | 48 / 46 |

### SGLang MTP depth 4: full / SWA pool (retract hi/lo)

| ccu (`max_running`) | 0.88, chunk 16384 | mem-only 0.92, chunk 16384 | tuned 0.92, chunk 8192 |
|---|---|---|---|
| 1 | 589,966 / 34,443 (0/0) | 641,383 / 34,443 (0/0) | 762,550 / 17,579 (0/0) |
| 8 | 506,266 / 46,161 (0/1) | 557,683 / 46,161 (0/1) | 702,140 / 25,937 (10/16) |
| 32 | 219,295 / 86,337 (9/5) | 270,712 / 86,337 (6/5) | 497,455 / 54,593 (9/19) |
| 64 (chỉ 32 slot) | 329,009 / 70,977 (1/1) | 380,426 / 70,977 (1/3) | 498,164 / 54,593 (13/9) |

MTP c64 chạy 32 slot nên pool bằng c32 về SWA; peak running chạm đúng 32 ở c64, 24-30 ở c32 (c32 hi 0.88: 25, mem-only: 30).

### vLLM v0.30.0 (mem 0.92, run 37259992014)

| ccu (`--max-num-seqs`) | Base: KV pool | Base: peak running / waiting / KV usage | MTP depth 4: KV pool | MTP: peak running / waiting / KV usage |
|---|---|---|---|---|
| 1 | 198,275 | 1 / 0 / 2-10% | 194,753 | 1 / 0 / 7.5% |
| 8 | 197,569 | 8 / 3-4 / 34-35% | 194,446 | 8 / 0-3 / 20-32% |
| 32 | 197,340 | 32 / 21-26 / 72-75% | 193,641 | 32 / 18-21 / 77% |
| 64 | 197,683 | 58 / 49-52 / 98-99.5% | 193,204 | 55-56 / 50-59 / 99.8-99.9% |

vLLM 256k + prefix cache (run 37911942823 / 37911946936, `max-model-len 262144`): KV pool 530-533k (base) và 519-524k (MTP) token ở mọi ccu, peak running 1 / 8 / 32 / 56-59 ở c1 / c8 / c32 / c64, peak KV usage 98.3-99.5% ở c64; chi tiết trong mục "vLLM 256k + prefix cache". Số token lớn hơn bảng 65k chỉ vì công thức tính theo `max_model_len`.

Log vLLM không có dòng preempt nào ở cell nào; vì c64 chạm 98-100% KV nên chưa kết luận được là không preempt (cần metrics). KV usage là đỉnh của các dòng log định kỳ, có thể sót đỉnh thật.

### Đọc bảng

- **SGLang: pool phụ thuộc `max_running_requests`; vLLM: gần như cố định.** vLLM dao động dưới 0.5% ở base (198,275 xuống 197,340), 2.5% ở MTP, và MTP thấp hơn base 2-3%. SGLang SWA pool theo công thức `max_running × ~1,153 + 2 × chunk`: ở chunk 16384 là 33,922 / 41,993 / 69,665 / 106,561 ở c1 / c8 / c32 / c64 (khớp tuyến tính, hệ số góc 1,153 token/request, hằng số 32,769 = 2 × 16384 + 1). SWA pool không phụ thuộc mem; mem chỉ làm full pool to ra (phần còn lại của bộ nhớ).
- **Full pool SGLang giảm mạnh khi tăng ccu** (base 0.88: 842k, 761k, 484k, 116k) vì SWA pool lớn dần và lấy chỗ. Ở c64 full pool chỉ còn 116-188k token, đủ khoảng 12-20 request đầy ngữ cảnh (9.2k token), nên đó là điểm duy nhất SGLang bị KV chặn ở base.
- **Ở c1/c8/c32 SGLang có nhiều chỗ hơn hẳn mức cần** (full pool 484k đến 915k so với 198k của vLLM), nhưng vì số token không so được nên đây không phải ưu thế thật; chỉ cho thấy KV không phải nút cổ chai ở các ccu đó, kể cả retract ở c8 (4-13 lần) là do SWA pool nhỏ (42k).
- **Chunk 8192 đổi chỗ**: SWA pool nhỏ đi (17k, 25k, 50k, 84k) nên full pool lớn lên (1.08M, 1.0M, 751k, 414k) và c64 chạy được 46-48 request (so với 22-28); đổi lại SWA pool c8/c32 nhỏ nên base lo retract 475 và 3547 lần.
- **MTP c64 trên SGLang chỉ có 32 slot** để startup không abort (SWA pool xấu nhất 64 slot cần khoảng 107 GiB), trong khi vLLM chạy MTP 64 slot, peak running 55-56. Đây là lý do thực tế SGLang MTP ở c64 không so công bằng được với vLLM.
- Chưa đo: số lần preempt của vLLM, và KV pool thực dùng trung bình (cả hai bên); các số retract là số lần dòng log, không phải số request bị đẩy ra (một dòng retract có thể nhiều request, thường là 1).

## Nhật ký SGLang

1. Chạy sweep gốc trên image 2d74309 (run 37297954388, 37298154251): base 8/8 passed, các cell MTP lỗi do bug MTP + FA4 (xem mục "Bug MTP + FA4").
2. Fix bug MTP + FA4 (swa-pool, commit 8c1384a796), build image A (có aux) và B (không aux), A/B ở hi c1/c8: sai khác dưới 1.5%.
3. MTP c64 abort ở startup do SWA pool xấu nhất ~107 GiB: giảm xuống 32 slot + eviction 32. Chạy lại 8 cell MTP: 6 cell trên Greennode `1x_01/_02` (xong), 4 cell lo trên hcm `8x_02` (xong, dùng để đối chiếu chéo), hi c64 trên hcm (xong, 778.5 tok/s).
4. Phát hiện KV pool base c64 nhỏ hơn vLLM. Probe a, b, c để thử mem, eviction, chunk (b đã hủy, c chạy lại do lỗi script).
5. Hai smoke MTP ở mem 0.92: d (c32) an toàn, +24% pool, tput gần như không đổi; e (c64, 64 slot) không abort nhưng kém hơn (534 tok/s, TTFT 100 s), nên giữ 32 slot.
6. Hai smoke tuned (0.92 + evict 32 + chunk 8192): f (MTP hi c32) 749.9 tok/s, full pool 498k, g (base hi c8) 323.8 tok/s, đều không OOM; g không đổi so với 0.88, f nhỉnh hơn một chút.
7. Chạy lại 16 cell ở cấu hình tuned (8 run, 1x_01/1x_02): 16/16 passed. Base c64 tăng khoảng 70%, MTP gần như không đổi, nhưng base lo c8 và c32 thụt lùi nặng (133 và 98.6 tok/s, hàng nghìn retract do SWA pool nhỏ).
8. Điều tra base lo c8/c32: chạy lại tuned vẫn thụt lùi (127.5 / 118.5), biến thể chunk 16384 hết lỗi (329.4 / 636.8), nên thủ phạm là chunk 8192. Sweep mem-only (mem 0.92 + chunk 16384, 16 cell) passed: base c64 +42%, phần còn lại ngang 0.88.
9. Probe A-G (13 run): A (FP8 KV với FA4) và C (flashinfer decode) fail khi khởi động vì H200 / Gemma-4 không hỗ trợ; G fail do sai tên flag của tôi; B, E, F không đổi so với mem-only; D cho thấy depth 2 tốt hơn ở entropy cao, depth 4 tốt hơn ở entropy thấp. Đã hủy sweep mem 0.92 + evict 32 (run 37815910995, 37815916948) vì probe B và D bao nội dung đó.
10. Chuyển sang phục vụ 256k và bật prefix cache cho cả hai engine (commit b2c9f949). Chạy 16 cell SGLang cấu hình A (ratio 0.3, chunk 16384, run 37880075115 / 37880078321) và 16 cell cấu hình B copy từ tau2 (ratio 0.2, chunk 8192, hrrn, evict 32; commit b0208a66, run 37884261982 / 37884264773): 32/32 passed. Base c64 tăng thêm 21-27% so với mem-only, MTP ngang hoặc tốt hơn, B không retract ở MTP. vLLM 16 cell: dispatch lần đầu trên `hcm-8x` rồi `hcm-1x` nhưng job `queued` nhiều giờ không runner nhận, đã hủy và xóa; dispatch lại trên Greennode `1x_01/_02` (commit 4bd28f8c, run 37911942823 / 37911946936): 16/16 passed.

## Việc còn lại

- vLLM 256k + prefix cache: đã xong (xem Phần 1). Còn lại: nguyên nhân hcm runner không nhận job. Hai lần dispatch trước trên `hardware-hcm-8x_*` và `hardware-hcm-1x_*` không có runner nhận job (nghi thiếu label tên runner), đã hủy.
- Chọn cấu hình SGLang mặc định mới giữa A (ratio 0.3, chunk 16384) và B (tau2: ratio 0.2, chunk 8192, hrrn, evict 32). B tốt hơn ở MTP (0 retract, tput ngang hoặc cao hơn 2-3%), A và B ngang nhau ở base c64 (639 / 618 so với 623 / 623), A có p99 ITL base c64 thấp hơn (37 ms so với 83-109 ms). Cần một lần lặp lại để biết chênh lệch dưới vài % có thật không.

- Quyết định mặc định SGLang: đề xuất mem 0.92 + chunk 16384 (mem-only, đã có đủ 16 cell). Không dùng chunk 8192 làm mặc định vì gây lỗi ở base lo c8/c32. Cần xác nhận.
- Dispatch lại probe G với `--fp8-gemm-backend` (cutlass, triton).
- FP8 KV trên H200 chỉ còn khả thi với `--attention-backend triton`; cần baseline triton BF16 để tách tác động backend khỏi FP8, và một lần GSM8K (dispatch hiện dùng `--no-evals`) để kiểm tra chất lượng.
- Chạy thêm MTP depth 2/3 ở các conc còn lại để chọn depth theo entropy; tính accept length và retract cho sweep mem-only và probe D.
- Radix cache (probe F): +2.4% ở c8 là trong nhiễu; đọc hit rate nếu muốn đánh giá lại.
- Bảng so sánh vLLM với SGLang ở 256k đã có (tput, xem mục vLLM 256k); chưa có bảng đầy đủ các metric khác (lưu ý khác biệt: image khác nhau, SGLang base c64 bị giới hạn KV, MTP c64 chỉ 32 slot; attention backend của hai bên cùng là FA4).
- PR upstream #43542 đã mở và ready; chờ maintainer gắn `run-ci`, sau đó theo dõi CI và review.
- Đề xuất cho script tổng hợp: tính accept length từ `spec_verify_calls`; tên artifact A và B trùng nhau nên tải theo artifact id.

## Tham chiếu

- Wrapper và body SGLang: `benchmarks/single_node/fixed_seq_len/gemma4sba_sglang_body.sh`, `gemma4sbaprobe{a,b,c}_*`, `gemma4sbamtpprobe{d,e}_*`.
- Ref đã đẩy: `refs/bench/gemma4-sglang-kvprobe` (68aef722), `refs/bench/gemma4-sglang-mtpprobe` (f1ba45cf), `refs/bench/gemma4-sglang-mtp-gnslurm` (e08a33ad), `refs/bench/gemma4-speedbench-aiperf-hcm-v030` (vLLM).

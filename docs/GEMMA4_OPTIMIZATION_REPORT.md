# Gemma-4 31B (FP8-block) — Báo cáo tối ưu serving trên 2×H200

**Ngày:** 2026-08-24 · **Nhánh:** `vng-benchmark-gemma4-sgl`
**Model:** `RedHatAI/gemma-4-31B-it-FP8-block` · **Phần cứng:** 2× H200 (single node)

## TL;DR

- **Cấu hình tốt nhất hiện tại: vLLM TP1×DP2 + EAGLE3.**
- Đây là topology đơn giản nhất mà đạt **concurrency cao nhất trong ngưỡng SLO**, nhờ EAGLE3 kéo TPOT xuống mạnh trên workload decode-bound.
- **Frontier hiện tại ≈ CCU 32** đạt SLO thoải mái; CCU 48 đang được dò bằng **MNBT sweep** (đang chạy — run [32702712255](https://github.com/vngcloud/InferenceX/actions/runs/32702712255)).

## Bối cảnh & SLO

- **Workload proxy:** 8k ISL / 1k OSL, nội dung random, cache-hit ≈ 0 — mô phỏng bảo thủ traffic production (ISL ≤ 8k, OSL ~1k, cache ~0).
- **SLO:** TTFT trung bình **< 8s**, TPOT trung bình **< 50ms**.
- Workload này **decode/TPOT-bound**: TTFT còn dư ~4× ngân sách ở vùng CCU hữu ích, nên **TPOT là ràng buộc quyết định** max concurrency.

## Các cấu hình đã chạy

| # | Cấu hình | GPUs | Spec-decode | Kết quả |
|---|----------|------|-------------|---------|
| 1 | vLLM TP2 | 2 | — | Baseline; decode chậm do TP comm |
| 2 | vLLM TP1×DP2 | 2 | — | Tốt hơn TP2 (KV riêng mỗi GPU, không có TP comm) |
| 3 | vLLM 2-replica + router | 2 | — | ≈ DP2, không có lợi thêm |
| 4 | **vLLM TP1×DP2 + EAGLE3** ★ | 2 | eagle3 | **Tốt nhất** |
| 5 | vLLM 2-replica + router + EAGLE3 | 2 | eagle3 | ≈ #4, nhưng phức tạp hơn (thêm replica + hop router) |

### So sánh nhanh tại điểm CCU 32 (điểm chung có đủ dữ liệu)

Throughput ở đây là **tổng toàn endpoint (2 GPU)** — đúng con số benchmark client đo. (Xem chú thích basis ở cuối.)

| Cấu hình | TPOT avg (ms) | Output tput (tok/s, 2 GPU) | Ghi chú |
|----------|--------------:|---------------------------:|---------|
| vLLM TP2 | 45.5 | 669 | sát trần SLO |
| vLLM TP1×DP2 | 47.8 | 631 | sát trần SLO |
| **vLLM TP1×DP2 + EAGLE3** ★ | **32.2** | **902** | biên an toàn lớn |
| vLLM 2-rep + router + EAGLE3 | 32.6 | 887 | ≈ #4 |

**Vì sao #4 thắng:**
1. **EAGLE3 là yếu tố quyết định:** so với DP2 thuần (cùng basis), EAGLE3 giảm TPOT **~33%** (47.8 → 32.2 ms) và tăng output throughput **~43%** (631 → 902 tok/s) — đúng bản chất workload decode-bound.
2. **Không có EAGLE, TP2 ≈ TP1×DP2** ở vùng SLO (c32: TPOT 45.5 vs 47.8 ms; output 669 vs 631 tok/s) — chọn TP hay DP gần như không đổi kết quả; DP2 được chọn làm nền để ghép EAGLE3 (chưa chạy điểm TP2+EAGLE3 để đối chứng).
3. **Router không thêm giá trị** so với DP2 (DP2 đã tự cân tải nội bộ): router+eagle ≈ DP2+eagle (887 vs 902 tok/s, 32.6 vs 32.2 ms @ c32) — nhưng tốn thêm 1 replica + 1 hop định tuyến.

→ **TP1×DP2 + EAGLE3** là topology gọn nhất đạt CCU-tại-SLO cao nhất (biên TPOT lớn nhất tại cùng throughput).

## Kết quả chi tiết — cấu hình tốt nhất (TP1×DP2 + EAGLE3)

Nguồn: run [32392516834](https://github.com/vngcloud/InferenceX/actions/runs/32392516834) · 8k1k · 2×H200 (`h200-greennode_04`)

Output tput = **tổng toàn endpoint (2 GPU)** — con số benchmark client đo trực tiếp.

| CCU | TTFT avg (s) | TPOT avg (ms) | Output tput (tok/s, 2 GPU) | E2E latency avg (s) | SLO |
|----:|-------------:|--------------:|---------------------------:|--------------------:|:---:|
| 8   | 1.16 | 19.0 | 378   | 18.7  | ✅ |
| 16  | 1.46 | 23.4 | 611   | 22.8  | ✅ |
| 32  | 1.86 | 32.2 | 902   | 31.6  | ✅ |
| 64  | 2.78 | 54.5 | 1,081 | 53.0  | ❌ (TPOT > 50ms) |
| 128 | 4.80 | 102.3 | 1,167 | 98.8  | ❌ |
| 160 | 5.56 | 124.7 | 1,205 | 120.1 | ❌ |
| 192 | 6.74 | 147.3 | 1,226 | 142.1 | ❌ |

**Đọc bảng:**
- **CCU 32 là điểm vận hành an toàn:** TPOT 32.2ms (dư 18ms so với ngưỡng 50ms), TTFT 1.86s (chỉ dùng ~23% ngân sách 8s).
- **Ràng buộc là TPOT**, không phải TTFT: TTFT vẫn trong ngưỡng tới tận CCU 192 (6.74s), nhưng TPOT vượt 50ms ngay từ CCU 64 (54.5ms).
- **Frontier thực tế nằm giữa CCU 32 và 64** (~48) — chính là lý do có MNBT sweep bên dưới.

## Bước tiếp theo (đang chạy)

**MNBT sweep** — dò `--max-num-batched-tokens` ∈ {2048, 4096, 8192} × CCU {32, 48, 64}: chia nhỏ prefill 8k để nó xen kẽ với decode → giảm TPOT bằng cách "tiêu" phần TTFT đang dư → kỳ vọng nâng max-CCU-tại-SLO từ 32 lên ~48.
Run: [32702712255](https://github.com/vngcloud/InferenceX/actions/runs/32702712255) (8192 là control = giá trị production hiện tại).

> **Ghi chú acceptance rate:** run tốt nhất (#4, ngày 20/08) chưa log tỉ lệ chấp nhận draft của EAGLE3 (dưới DP2, vLLM chặn dòng "SpecDecoding metrics" trên stdout). MNBT sweep đang chạy đã bổ sung scrape từ endpoint Prometheus `/metrics` (DP2-safe), nên **draft acceptance rate + mean acceptance length sẽ có trong log từng job** của sweep này.

---

## Chú thích về basis của throughput

Benchmark client chỉ đo ở **1 endpoint** nên mọi số throughput (`output_throughput`, `total_token_throughput`) là **tổng toàn hệ thống (gộp mọi GPU)**; TTFT/TPOT/latency là thống kê per-request. Sau đó `utils/process_result.py` chia ra field `*_per_gpu` bằng `num_gpus = tp × pp × pcp` — **không tính DP và EP**. Vì các config DP2 ở đây có `tp=1` nên `num_gpus=1`, tức field `output_tput_per_gpu` trong `agg_bmk.json` của chúng **thực chất là tổng 2 GPU**, không phải per-GPU (chỉ config TP2 mới được chia đúng). Mọi số throughput trong report này đã quy về **tổng toàn endpoint (2 GPU)** để so sánh nhất quán. Muốn ra per-GPU thật thì chia cho 2.

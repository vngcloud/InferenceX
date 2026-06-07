# Báo cáo Benchmark — Agentic CCU Sweep (smoke): Qwen3-4B trên vLLM / H100

**Run ID:** [26933973732](https://github.com/vngcloud/InferenceX/actions/runs/26933973732) ·
`conclusion: success` · **Ngày:** 2026-06-04 · **Kết quả:** PASS (3/3 leg, 50/50 request, 0 lỗi)
**Phân loại:** Smoke — kiểm định pipeline. **Các chỉ số dưới đây KHÔNG dùng làm số chính thức** (xem §5).

---

## Tóm tắt

Báo cáo đánh giá đợt chạy **Mode 1 (CCU capacity sweep)** đầu tiên của luồng
agentic-replay: replay trace coding-agent dưới áp lực concurrency thuần (bỏ
think-time), quét concurrency 8 → 16 → 32 trên `Qwen3-4B-Instruct-2507` / vLLM /
1× H100.

Kết quả chính: **pipeline chạy đúng end-to-end** (cả ba mức concurrency hoàn tất
trọn vẹn, không lỗi, qua gate fail-closed), và hệ thống **mở rộng sạch** — throughput
tăng gần tuyến tính trong khi TTFT/TPOT vẫn thấp. **Điểm gãy (knee) chưa xuất hiện ở
conc=32**, cho thấy còn nhiều dư địa tải. Vì đây là smoke (50 request, chạy một lần),
các giá trị tuyệt đối — đặc biệt là **Power và Token/Watt** — chưa đủ tin cậy để trích
dẫn; giá trị của đợt chạy nằm ở việc xác nhận luồng đo và xu hướng scale.

---

## 1. Cấu hình & phương pháp

| Hạng mục | Giá trị |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` (dense transformer) |
| Precision | BF16 |
| Engine | vLLM `v0.21.0` |
| Phần cứng | H100-2X runner — **TP=1, dùng 1× H100** (card thứ 2 để trống) |
| Song song hoá | TP=1 · EP=1 · DP-Attention=false |
| Workload | Agentic-replay trace `qwen3.5-4b-smoke.jsonl` (12 record) |
| Số request | 50 (AIPerf resample từ 12 record) |
| Chế độ đo | **Mode 1**: `--no-fixed-schedule`, delays stripped, think-time = 0 |
| Thang concurrency | 8, 16, 32 (mỗi mức một job server riêng, `--max-num-seqs = conc`) |
| Client đo | AIPerf `0.9.0` (gate fail-closed: request đủ & 0 lỗi mới ghi kết quả) |
| Artifact | `agg_bmk.json` trong artifact của run |

> **Lưu ý đọc số:** cột **ISL=4096 / OSL=512** trong bảng gốc là **placeholder** để qua
> kiểm tra downstream — không phản ánh workload thật. Trace agentic có độ dài input/output
> thay đổi theo từng lượt và **context tích lũy** qua các lượt của cùng một session.

---

## 2. Kết quả đo

| Concurrency | 8 | 16 | 32 |
|---|---:|---:|---:|
| **TTFT mean** (ms) | 36.3 | 77.2 | 139.2 |
| TTFT P50 (ms) | 30.5 | 49.2 | 79.7 |
| TTFT P90 (ms) | 64.5 | 181.3 | 225.8 |
| **TTFT P99** (ms) | 83.8 | 181.6 | 226.1 |
| **TPOT mean** (ms) | 5.34 | 5.98 | 6.92 |
| TPOT P90 (ms) | 5.90 | 6.18 | 7.88 |
| **TPOT P99** (ms) | 6.80 | 8.68 | 9.28 |
| Interactivity mean (tok/s/user) | 187.3 | 167.2 | 144.6 |
| **E2E mean** (s) | 0.51 | 0.60 | 0.74 |
| E2E P99 (s) | 0.97 | 1.02 | 1.16 |
| **Total tput/GPU** (tok/s) | 24,987 | 37,412 | 57,680 |
| **Output tput/GPU** (tok/s) | 1,261 | 1,912 | 3,120 |
| Input tput/GPU (tok/s) | 23,726 | 35,500 | 54,560 |
| Power mean (W) † | 277.1 | 239.6 | 122.9 |
| Token/Watt total (tok/s/W) † | 90.2 | 156.1 | 469.3 |

† *Không đáng tin trên smoke — xem §5.2.*

---

## 3. Phân tích

### 3.1 Throughput — mở rộng gần tuyến tính

- **Output tput/GPU** tăng 1,261 → 1,912 → 3,120 tok/s, tức ~**2.5×** khi concurrency
  tăng 4× (8→32). Độ dưới-tuyến-tính nhẹ là bình thường do overhead batching.
- **Total tput/GPU** cao hơn output ~**18–20 lần** (24,987 vs 1,261 ở conc=8). Nguyên
  nhân: trace agentic mang **context tích lũy lớn** (nhiều token input) nhưng output mỗi
  lượt nhỏ → phần lớn token là **prefill (input)**, không phải decode.

> **Quy ước báo cáo:** phải nêu rõ dùng *total* hay *output* throughput. **Output
> tput/GPU** phản ánh công việc decode hữu ích; *total* bị phần input lấn át nên luôn
> cao hơn nhiều lần.

### 3.2 Latency — thấp và ổn định, chưa nghẽn

- **TTFT** tăng theo tải (36 → 77 → 139 ms mean; P99 ≤ 226 ms) nhưng vẫn rất thấp.
  Hàng đợi prefill dài dần ở conc=32 song chưa tới mức nghẽn.
- **TPOT (= ITL)** gần như phẳng (5.34 → 5.98 → 6.92 ms mean; P99 ≤ 9.3 ms) → tốc độ
  sinh token mượt, ít nhạy với tải. Đây là chỉ số khoẻ nhất của đợt chạy.
- **E2E** tăng nhẹ (0.51 → 0.74 s mean), phần lớn phân vị vẫn dưới ~1 giây.
- **Interactivity** giảm dần theo TPOT (187 → 145 tok/s/user) nhưng vẫn cao hơn nhiều
  tốc độ đọc của con người.

### 3.3 Xu hướng tổng thể

Throughput tăng đều trong khi latency tăng chậm và chưa có dấu hiệu bão hoà →
**knee chưa xuất hiện ở conc=32**. Ở quy mô trace này, hệ thống còn dư địa tải; điểm
gãy thực sự nằm ở mức concurrency cao hơn nhưng smoke này không đủ dữ liệu để xác định.

---

## 4. Kết luận

1. **Pipeline Mode 1 hợp lệ end-to-end.** Cả ba leg (8/16/32) hoàn tất 50/50 request,
   0 lỗi, vượt gate fail-closed của adapter — luồng config → matrix → launcher →
   AIPerf → adapter hoạt động đúng.
2. **Scale sạch, chưa chạm giới hạn.** Throughput tăng gần tuyến tính, TTFT/TPOT thấp
   và ổn định; conc=32 chưa phải biên capacity.
3. **Chưa thể trích dẫn số tuyệt đối.** Quy mô smoke và các artifact đo năng lượng
   chưa đủ tin (xem §5).

---

## 5. Giới hạn & độ tin cậy

### 5.1 Quy mô smoke
50 request resample từ trace 12 record là quá ít để TTFT/TPOT hội tụ; mỗi điểm chỉ
chạy **một lần** (không có mean ± std). Kết quả chỉ phản ánh **tính đúng của pipeline**
và **xu hướng**, không phải số định lượng để so sánh.

### 5.2 Power và Token/Watt không đáng tin
Power mean đi **ngược logic** (277 → 240 → **123 W**, giảm khi tải tăng), kéo theo
Token/Watt total phình giả (90 → 156 → 469). Đây là **artifact của cửa sổ đo** trên
run quá ngắn (dưới vài giây), không phải hiệu suất năng lượng thật. Hai cột này chỉ tin
được trên sweep dài.

### 5.3 ISL/OSL là placeholder
Giá trị 4096/512 không phải độ dài thật của trace; không diễn giải theo nghĩa đen.

### 5.4 TP=1
Mọi chỉ số "per GPU" là của **một H100**; card thứ hai trên runner không tham gia.

---

## 6. Khuyến nghị cho đợt đo chính thức

Để chuyển từ smoke sang số trích dẫn được: dùng trace config150s thật
(`agentic-coding-64k-5variants-config150s-seed42-20260605-131906.jsonl`,
`max-model-len 73728`), nâng `request-count` lên
~2000–3000 với ~16 warmup session, mở rộng thang concurrency lên `[8, 16, 32, 64,
128, 256]` và dừng ở leg đầu tiên vỡ SLA TTFT/ITL, đồng thời chạy lặp để có mean ± std.
Khi đó các chỉ số Power/Token-Watt mới đáng tin. Quy trình chi tiết:
[`AGENTIC_CCU_SWEEP_GUIDE_VI.md`](AGENTIC_CCU_SWEEP_GUIDE_VI.md).

---

## Phụ lục — thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| **CCU / Conc** | Concurrency — số request đồng thời |
| **TTFT** | Time To First Token — độ trễ tới token đầu tiên |
| **TPOT** (= ITL) | Time Per Output Token — thời gian sinh mỗi token tiếp theo |
| **E2EL** | End-to-End Latency — tổng thời gian một request |
| **Interactivity** | tok/s/user ≈ 1000 / TPOT — tốc độ "đọc" mà một user cảm nhận |
| **Total tput/GPU** | Throughput input+output trên mỗi GPU |
| **Output tput/GPU** | Throughput chỉ tính token decode (công việc hữu ích) |
| **P50/P90/P99** | Phân vị — P99 = 99% request nhanh hơn ngưỡng này (đuôi xấu nhất) |
| **knee** | Điểm gãy: mức concurrency mà tăng thêm thì latency vọt lên, throughput bão hoà |
</content>

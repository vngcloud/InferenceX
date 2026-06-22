# MiniMax-M2.5 · SGLang · 8×H200 — AgentX-MVP (cc-traces-weka) vs. Mode-1 (mooncake)

> File centralize giải thích **chính xác** workload vừa chạy (AgentX-MVP trên corpus
> `cc-traces-weka`) và **khác gì** so với report cũ Mode-1 mooncake
> (`aiperf-service-docs/reports/20260606_minimax-m2.5_H200_agentic-coding/`).
>
> Mọi quy tắc "scenario khóa" trong file này lấy từ **code** (`ScenarioSpec` trong
> `utils/aiperf/src/aiperf/common/scenario/inferencex_agentx_mvp.py`), KHÔNG phải từ
> tài liệu tutorial — vì doc upstream có chỗ đã lỗi thời (xem [§7](#7-cảnh-báo-doc-upstream-lỗi-thời)).

---

## Mục lục

1. [TL;DR — vừa chạy cái gì](#1-tldr--vừa-chạy-cái-gì)
2. [Run đã submit](#2-run-đã-submit)
3. [AgentX-MVP là gì & scenario KHÓA gì (canonical)](#3-agentx-mvp-là-gì--scenario-khóa-gì-canonical)
4. [Núm team thêm (KHÔNG bị scenario khóa)](#4-núm-team-thêm-không-bị-scenario-khóa)
5. [3 sửa đổi trong session này](#5-3-sửa-đổi-trong-session-này)
6. [So sánh với workload cũ (Mode-1 mooncake 20260606)](#6-so-sánh-với-workload-cũ-mode-1-mooncake-20260606)
7. [Cảnh báo: doc upstream lỗi thời](#7-cảnh-báo-doc-upstream-lỗi-thời)
8. [submission_valid nghĩa là gì](#8-submission_valid-nghĩa-là-gì)
9. [Nguồn dữ liệu](#9-nguồn-dữ-liệu)

---

## 1. TL;DR — vừa chạy cái gì

Ta chạy **benchmark canonical "InferenceX AgentX-MVP" của SemiAnalysis**, lần đầu trên
hạ tầng team, cho **MiniMax-M2.5 FP8 / SGLang v0.5.12 / 8×H200 (TP8/EP8)**.

- **Workload**: replay **949 phiên Claude Code thật** (corpus
  `semianalysisai/cc-traces-weka-no-subagents-051226`) qua scenario aiperf
  `inferencex-agentx-mvp` — multi-turn, có warmup KV-cache, recycle FIFO, cache-bust
  mỗi lượt phát, clamp think-time 60s.
- **Tải**: concurrency `[16, 24, 32]`, mỗi điểm đo **1800s**.
- **Kết quả kỳ vọng**: `submission_valid = true` (≥900s, không phá luật scenario) —
  tức là số liệu **hợp lệ để nộp/so sánh liên-team**, khác hẳn các run nội bộ trước đó.
- **Điểm mới quan trọng**: lần đầu đo được **cache-hit** —
  `theoretical_cache_hit_rate` (client-side, từ hash_id của trace) và
  `server_gpu_cache_hit_rate` (scrape `/metrics` của SGLang). Đây là *lý do tồn tại*
  của corpus weka: đo tái dùng prefix/KV.

> **Khác biệt một dòng so với report cũ**: report cũ là dataset *tổng hợp* replay
> bằng mode `mooncake_trace` (không luật, không validity stamp). Run này là *trace
> thật* chạy bằng *scenario có luật* → là bài benchmark **canonical**, không phải
> bài đo nội bộ.

---

## 2. Run đã submit

| Field | Value |
| --- | --- |
| Model | `MiniMaxAI/MiniMax-M2.5` (FP8) |
| Engine | SGLang `v0.5.12` (`lmsysorg/sglang:v0.5.12`) |
| GPU | 8× NVIDIA H200 — `h200-greennode_00` |
| Parallelism | `tp=8 ep=8` · `dpa=false` · `disagg=false` · `spec=none` |
| Harness | AIPerf · scenario `inferencex-agentx-mvp` · timing `agentic_replay` |
| Corpus | `semianalysisai/cc-traces-weka-no-subagents-051226` (949 traces) |
| Concurrency | 16 / 24 / 32 |
| Duration | 1800s/điểm (≥900 → submission-valid) |
| Branch | `exp/minimax-2.5-sglang-8xh200-semianalysis_cc_traces_weka` |
| Run ID | `27096125894` (workflow_dispatch; KHÔNG perf-changelog/PR) |
| Config key | `minimaxm2.5-weka-h200-sglang-8x` (`.github/configs/nvidia-master.yaml`) |
| Launcher | `benchmarks/single_node/agentic/minimaxm2.5-weka_fp8_h100_sglang.sh` |

**Lệnh serving** (giống hệt report cũ, chỉ thêm `--enable-metrics`):

```bash
python3 -m sglang.launch_server \
  --model-path MiniMaxAI/MiniMax-M2.5 --served-model-name MiniMaxAI/MiniMax-M2.5 \
  --host 0.0.0.0 --port 8888 \
  --tp 8 --ep 8 --context-length 147456 \
  --tool-call-parser minimax-m2 --reasoning-parser minimax \
  --mem-fraction-static 0.85 --page-size 64 --chunked-prefill-size 16384 \
  --hicache-size 1200 \
  --enable-metrics \            # <-- thêm mới: bật /metrics cho aiperf scrape cache-hit
  --trust-remote-code
```

---

## 3. AgentX-MVP là gì & scenario KHÓA gì (canonical)

AgentX-MVP = "công thức" replay do SemiAnalysis đề xuất: gói toàn bộ luật vào **một cờ
duy nhất** `--scenario inferencex-agentx-mvp`, để hai team trên hai server khác nhau ra
số **so sánh được**. Núm tải duy nhất người dùng chỉnh là `--concurrency`.

Các luật scenario **khóa cứng** (nguồn: `ScenarioSpec`, code authoritative):

| Luật khóa | Giá trị | Ý nghĩa |
| --- | --- | --- |
| `timing_mode` | `AGENTIC_REPLAY` | Scheduler multi-turn: warmup → steady-state, recycle FIFO, clamp 60s. |
| `require_ignore_eos` | `true` | Server bị buộc sinh đủ độ dài yêu cầu (không tự dừng sớm). |
| `require_use_think_time_only` | `true` | Delay giữa lượt chỉ dùng "think time" đã ghi, bỏ "send-to-send" (vốn lẫn thời gian phản hồi của server gốc). |
| `forbid_input_truncation` | `true` | Cấm `--synthesis-max-isl` — cắt prompt phía client là làm sai workload. |
| `require_loader` | `semianalysis_cc_traces_weka` \| `weka_trace` | Buộc đúng corpus weka hash-verifiable. |
| `min_benchmark_duration_seconds` | `900` | Run ngắn hơn 900s là noise → bị từ chối (trừ khi `--unsafe-override`). |
| `inter_turn_delay_cap_seconds` | `60.0` | Delay đơn lẻ >60s bị clamp về 60s (bỏ "coffee-break" 10 phút). |
| `require_cache_bust` | **`FIRST_TURN_PREFIX`** | Mỗi lượt phát của 1 trace được chèn marker `[rid:…]` duy nhất → ngăn cache ấm dồn giả tạo khi recycle. ⚠️ doc ghi nhầm `system_prefix`, xem [§7](#7-cảnh-báo-doc-upstream-lỗi-thời). |

**Cách scenario chạy** (tóm tắt từ tutorial, phần đúng):
- **Warmup**: chọn `--concurrency` trajectory; mỗi trajectory bốc điểm bắt đầu `k_i`
  ngẫu nhiên (xác định bởi `--random-seed`), gửi 1 request kèm prefix history → làm ấm
  KV-cache trước khi đo.
- **Profiling**: mỗi trajectory replay tiếp từ `k_i+1`, tôn trọng think-time (clamp 60s).
  Trace xong quay lại hàng đợi FIFO recycle; recycle bắt đầu lại từ turn 0, marker mới.
- **Kết thúc khi `--benchmark-duration` hết.** ⇒ **số trace KHÔNG quyết định thời lượng run.**
- **Concurrency phải ≤ pool size** (mỗi lane gắn 1 trajectory riêng). 949 trace → conc
  16/24/32 thừa sức; nếu chỉ 100 trace thì trần concurrency thấp hơn nhiều.

---

## 4. Núm team thêm (KHÔNG bị scenario khóa)

`build_replay_cmd()` trong `benchmarks/benchmark_lib.sh` thêm các núm sau. **Tất cả đều
nằm ngoài `ScenarioSpec`** ⇒ không phá luật, `submission_valid` không bị ảnh hưởng.

| Núm | Default upstream | Giá trị ta dùng | Phân loại | Ảnh hưởng số? |
| --- | --- | --- | --- | --- |
| `--num-dataset-entries` | 100 (subset) | **949** (full corpus) | **Đúng chuẩn** — doc: "for a canonical submission, load full corpus" | Đại diện hơn; KHÔNG làm run lâu hơn |
| `AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES` | `0` (pre-canned) | **`1`** (live) | Tùy chọn có ghi trong doc | Cache-hit phản ánh tái dùng KV **thực** (đánh đổi: hash-id sau turn 0 không khớp byte-to-byte) |
| `--trajectory-start-min/max-ratio` | `0.0` / `0.7` | **`0.25` / `0.75`** | **Tweak riêng của team** (không bị khóa) | Dời điểm warmup khỏi turn 0 (cache lạnh) → giảm nhiễu steady-state |
| `--use-server-token-count` | off | **on** | Doc nói "safe to add" | Lossless; tránh pin CPU khi tokenize client |
| `--slice-duration` | — | **1.0** | Hạ tầng | Time-series 1s cho plot KV/cache/throughput |
| `--random-seed` | random (auto) | **42** | Hạ tầng | Tái lập trajectory/`k_i` |
| `--failed-request-threshold` | — | **0.05** | Hạ tầng | Abort nếu fail thật >5% |
| `--enable-metrics` (SGLang) + patch NaN | — | **on** | Của ta | Bật scrape `server_gpu_cache_hit_rate` |

**Kết luận về tính đúng đắn**: chỉ **1 núm** là tweak thuần team không có trong doc
(`trajectory-start-ratio`); nó giảm nhiễu chứ không làm sai. Các núm còn lại hoặc là
**đúng chuẩn canonical** (949), hoặc **được doc cho phép** (live-assistant,
use-server-token-count), hoặc thuần hạ tầng. ⇒ Run **không bị làm sai kết quả**; chỉ cần
nhớ: so số của ta với **các run cùng cấu hình `build_replay_cmd`**, đừng so byte-to-byte
với một run upstream chạy default trần.

---

## 5. 3 sửa đổi trong session này

Trước session này, smoke đầu tiên (`27090511286`) chạy được nhưng **cả 2 chỉ số cache-hit
đều null**. Đã sửa qua 3 thay đổi (branch
`exp/minimax-2.5-sglang-8xh200-semianalysis_cc_traces_weka`):

1. **Bug tên dataset** (commit `4e213c8`) — `_HF_DATASET` trong
   `utils/process_agentic_result.py` trỏ sai `cc-traces-weka-042026`; đổi sang
   `cc-traces-weka-no-subagents-051226`. → `theoretical_cache_hit_rate`: null → **0.286**.
2. **Map cache-hit SGLang** (commit `4e213c8`) — SGLang phát tên metric khác vLLM; thêm
   nhánh đọc `sglang:cached_tokens(_total)`/`prompt_tokens`, fallback gauge
   `sglang:cache_hit_rate`. + 2 unit test.
3. **Patch NaN của aiperf** (commit `158ee16`) — SGLang phát `sglang:fwd_occupancy=NaN`
   trước forward pass đầu; filter aiperf chỉ bắt `inf` → NaN lọt → orjson serialize thành
   `null` → validation reject → **rớt toàn bộ bản scrape** (kéo cache-hit theo). Fix 1
   dòng `not math.isfinite(...)`. Vì **không có quyền push fork `vngcloud/aiperf`**, ship
   dạng `.patch` apply lúc runtime trong launcher (idempotent), submodule gitlink giữ
   nguyên `7d880a1e`.

**Smoke thứ 3 (`27095454993`) xác minh** cả 3 fix: `theoretical=0.286`,
`server_gpu_cache_hit_rate=0.505`, `server_metrics_export.json` đầy đủ, hết
`ValidationError`. → mới dựng run đầy đủ.

---

## 6. So sánh với workload cũ (Mode-1 mooncake 20260606)

| Khía cạnh | **Cũ — Mode-1 (report 20260606)** | **Mới — AgentX-MVP (run 27096125894)** |
| --- | --- | --- |
| Harness/mode | AIPerf `mooncake_trace` replay | AIPerf scenario `inferencex-agentx-mvp` (`agentic_replay`) |
| Dataset | **Tổng hợp/sinh ra**: `agentic_coding_{64k,128k}_1l1variant_config150s_seed42_*.jsonl` | **Trace thật**: `cc-traces-weka-no-subagents-051226` (949 phiên Claude Code) |
| Cấu trúc | Flat replay 1 file jsonl, bucket theo độ dài (64k/128k) | Multi-turn có warmup trajectory + recycle FIFO + cache-bust mỗi lượt |
| Luật benchmark | Không (chỉ `ignore_eos:true`) | 8 luật scenario khóa cứng ([§3](#3-agentx-mvp-là-gì--scenario-khóa-gì-canonical)) |
| `submission_valid` | **Không có** (chạy không scenario) | **`true`** (canonical, nộp được) |
| Cache-hit đo? | Không | **Có** — theoretical + server GPU |
| live-assistant | Không áp dụng | **Có** (cache-hit phản ánh tái dùng KV thực) |
| Think-time giữa lượt | Không | Có (clamp 60s) |
| Duration/điểm | ~150s (`config150s`) | **1800s** |
| Concurrency | 16/24/32 | 16/24/32 (cố ý giữ để cùng thang) |
| Serving config | Giống hệt (TP8/EP8, ctx 147456, …) | Giống hệt **+ `--enable-metrics`** |
| Hardware | 8×H200 `h200-greennode_00` | 8×H200 `h200-greennode_00` |
| # Request | 64k: 2,913 · 128k: 2,732 | Theo 1800s × conc (đo xong sẽ điền) |

### Vì sao KHÔNG so số trực tiếp được

Hai bài đo **không cùng workload** nên throughput/TTFT/latency không so 1:1:

1. **Dataset khác bản chất**: cũ là synthetic bucket 64k/128k cố định; mới là 949 trace
   thật, độ dài phân tán, multi-turn có think-time.
2. **Discipline replay khác**: cũ flat; mới có warmup làm ấm cache + cache-bust chống ấm
   giả + live-assistant → đường cong cache-hit/prefill khác hẳn.
3. **Duration khác** (150s vs 1800s): cửa sổ steady-state dài hơn nhiều.
4. **Mục đích khác**: cũ = đo nội bộ tìm trần throughput theo bucket ngữ cảnh; mới =
   bài **canonical submission-valid** đo trải nghiệm agentic thật + tái dùng cache.

> Điểm chung duy nhất so sánh được: **cùng serving config + cùng hardware + cùng thang
> concurrency [16,24,32]**. Vì thế giữ nguyên conc ladder — để khi đọc hai report cạnh
> nhau, ít nhất trục tải là chung.

---

## 7. Cảnh báo: doc upstream lỗi thời

Khi viết file này tôi đối chiếu **code vs tutorial** và thấy lệch:

- Tutorial `agentx-mvp.md` (dòng 151, 280) ghi scenario khóa **`--cache-bust system_prefix`**.
- Nhưng `ScenarioSpec` thực tế (`inferencex_agentx_mvp.py:16`) khóa
  **`require_cache_bust=CacheBustTarget.FIRST_TURN_PREFIX`**.

⇒ **Code là nguồn chân lý**: marker cache-bust được chèn ở **first-turn-prefix**, không
phải system-prefix. Comment trong `benchmark_lib.sh` (đã ghi `first_turn_prefix`) khớp
code. Khi đọc tutorial, lưu ý điểm này.

(Tutorial cũng ghi corpus mặc định là `cc-traces-weka-042026` / 739 traces — đó là bản
demo cũ; ta dùng bản `no-subagents-051226` / 949 traces qua `resolve_trace_source`.)

---

## 8. submission_valid nghĩa là gì

Stamp trong `metadata` của `profile_export.json` (chỉ có khi chạy `--scenario`):

- **`true`** — tuân thủ mọi luật + chạy sạch. Đây là cái ta muốn (run 1800s này).
- **`false`** — phá luật/ép buộc; kèm `submission_invalid_reasons`, ví dụ:
  - `unsafe_override` — đã `--unsafe-override` + phá ít nhất 1 luật (các smoke 120s của ta
    rơi vào đây vì <900s).
  - `context_overflow_rate_exceeded` — >1% request bị server báo tràn ngữ cảnh (thường do
    `--max-model-len`/`--context-length` đặt thấp hơn yêu cầu corpus). ⚠️ Đáng theo dõi
    với ta vì dùng `--context-length 147456` (~144k) chứ không phải default model.
- **Vắng field** — chạy không `--scenario`. (← chính là tình trạng report cũ Mode-1.)

---

## 9. Nguồn dữ liệu

| Mục | Đường dẫn |
| --- | --- |
| Scenario spec (authoritative) | `utils/aiperf/src/aiperf/common/scenario/inferencex_agentx_mvp.py` |
| Validator luật scenario | `utils/aiperf/src/aiperf/common/scenario/validator.py` |
| Default trajectory-start ratio | `utils/aiperf/src/aiperf/common/config/loadgen_config.py:210-242` |
| Tutorial (có chỗ lỗi thời) | `utils/aiperf/docs/tutorials/agentx-mvp.md` |
| Wiring replay của team | `benchmarks/benchmark_lib.sh` (`resolve_trace_source`, `build_replay_cmd`) |
| Launcher | `benchmarks/single_node/agentic/minimaxm2.5-weka_fp8_h100_sglang.sh` |
| Config key | `.github/configs/nvidia-master.yaml` → `minimaxm2.5-weka-h200-sglang-8x` |
| Patch NaN | `benchmarks/single_node/agentic/patches/aiperf-skip-nonfinite-server-metrics.patch` |
| Report cũ Mode-1 | `aiperf-service-docs/reports/20260606_minimax-m2.5_H200_agentic-coding/` |
| Run mới | https://github.com/vngcloud/InferenceX/actions/runs/27096125894 |
| Smoke xác minh fix | https://github.com/vngcloud/InferenceX/actions/runs/27095454993 |

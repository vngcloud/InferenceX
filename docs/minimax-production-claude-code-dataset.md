# Dataset MiniMax Claude Code production (v4 weka)

Dataset replay cho agentic coding, dựng từ traffic production thật của MiniMax M2.5 trên gateway. Lần này dùng weka trace format thay vì mooncake, vì weka cho phép mô hình hóa context compaction thông qua `reset_context` (đổi hash_ids mới khi context bị nén), điều mà mooncake DELTAS không làm được.

Trace base: `InferenceX/benchmarks/single_node/agentic/datasets/minimax_cc_v4_weka/` (234 file JSON, mỗi file một CC session). Đã merge vào branch `dev` của `vngcloud/InferenceX`, commit `fb1b5ff8`.

## Nguồn dữ liệu

Pull trực tiếp từ vMonitor Log API, project `006bf4f0-527b-43d1-87d2-dd0a0733387f`. Cửa sổ thời gian: 10/6 đến 17/6/2026 (tuần hackathon Claw-a-thon).

Chiến lược pull: query per-session. Terms aggregation lấy tất cả 260 session ID có header `x-claude-code-session-id`, rồi mỗi session pull riêng với pagination 500/page. Cách này tránh được giới hạn `from + size <= 10000` của Elasticsearch và đảm bảo capture 100% traffic.

Pull script: `price_research/pull_cc_sessions.py`. Raw data lưu tại `/tmp/cc_v4_full.jsonl` (28,234 records, 260 sessions).

### Traffic MiniMax trên gateway (June 10 to 17)

Tổng minimax-m2.5 traffic trong tuần hackathon: 86,370 requests. Chỉ 30% là Claude Code, phần còn lại là API calls từ các client khác.

| Nhóm client | Requests | Tỉ lệ |
| --- | ---: | ---: |
| Claude Code (claude-cli/*) | 26,644 | 30.8% |
| OpenAI/Python, AsyncOpenAI/Python | ~25,000 | ~29% |
| python-httpx, python-requests, node, Go-http-client | ~13,000 | ~15% |
| Anthropic/Python, agent-framework-python | ~5,400 | ~6% |
| opencode, codex, các client khác | ~16,300 | ~19% |

Dataset chỉ lấy phần Claude Code vì chỉ CC gửi `x-claude-code-session-id`, cho phép reconstruct multi-turn conversations có think-time. Các client khác (OpenAI/Python, httpx, node) gửi prompt ngắn, 1-shot, không có session continuity.

Việc chọn CC-only có hệ quả trực tiếp lên ISL: CC duy trì context qua 30+ turns nên ISL cao (p50 = 63K), trong khi API calls từ các client khác có ISL thấp (kéo median toàn traffic xuống 7.5K). Đây là thuộc tính đúng của workload agentic coding, không phải sai số dataset.

### Filter

| Bước | Requests | Sessions |
| --- | ---: | ---: |
| Raw pull (tất cả status) | 28,234 | 260 |
| status=200, lat>0, started>0 | 23,572 | 260 |
| Turns >= 2 (bỏ one-shot) | 23,571 | 234 |
| Sau burst linearization | 21,449 | 234 |

26 sessions bị drop vì chỉ có 1 request (one-shot hoặc bị rate-limit toàn bộ). 2,123 requests bị drop do burst linearization (chọn 1 request đại diện cho mỗi nhóm concurrent).

## Dataset đã tạo

| Property | Giá trị |
| --- | ---: |
| Format | Weka trace (block_size=64) |
| Traces | 234 (1 file JSON per CC session) |
| Total requests | 21,449 |
| Prompt content | Synthetic (gateway mask content) |
| Session IDs | Native `x-claude-code-session-id` |
| Turn order | Chronological (sort theo `started_at`) |
| ISL/OSL | Native gateway usage (`prompt_tokens`, `completion_tokens`) |
| Think time | Raw, đo từ gap giữa consecutive requests |
| Latency | `latencies.request` (wall-clock đầy đủ, bao gồm gateway + LLM) |
| Cache model | LCP-based hash_ids + reset_context cho compaction |
| Pydantic validation | 234/234 pass |
| Hash_id sufficiency | 21,449/21,449 requests có đủ blocks |
| Token fidelity | Input 97.9%, Output 90.4% so raw gateway |

### Thống kê chính

| Metric | p50 | p90 | p99 | Max |
| --- | ---: | ---: | ---: | ---: |
| Turns/trace | 37 | 240 | 548 | 924 |
| ISL (tokens) | 62,725 | 130,411 | 161,346 | 176,000 |
| OSL (tokens) | 121 | 547 | 2,374 | 8,192 |
| Think time (s) | 0.8 | 47.6 | 619.1 | 244,938 (68h) |
| API time (s) | 4.5 | 15.2 | 35.5 | 274.2 |
| Cache hit (LCP > 0) | 86.5% | | | |
| Compaction resets | 6.2% | | | |

## Số turn mỗi session

234 traces, phân phối đuôi nặng. Một vài session coding dài gánh phần lớn token volume.

| Turns/session | Số session | Tỉ lệ |
| --- | ---: | ---: |
| 2 to 5 | 47 | 20.1% |
| 6 to 10 | 18 | 7.7% |
| 11 to 25 | 33 | 14.1% |
| 26 to 50 | 33 | 14.1% |
| 51 to 100 | 32 | 13.7% |
| 101 to 500 | 59 | 25.2% |
| 501+ (tối đa 924) | 12 | 5.1% |

## Tỉ lệ input/output

Tính trên 21,449 requests trong dataset:

| Metric | Giá trị |
| --- | ---: |
| Input mean | 63,326 tokens |
| Output mean | 255 tokens |
| Aggregate ratio (sum input / sum output) | 248 : 1 |
| Per-request ratio p50 | 515 : 1 |

Coding context-heavy: prompt tích lũy theo turn (system prompt + history + tool results), output mỗi turn ngắn (tool-call response, code snippet). Output ngắn vì phần lớn turn là tool-calling, không phải text-answer dài.

## Think time

Think time = gap giữa end request trước và start request sau, trong cùng session. Tính bằng `latencies.request` (wall-clock đầy đủ), không phải `llm_latency` (chỉ thời gian LLM inference).

```text
think_time = max(0, current_started_at - (previous_started_at + previous_latencies_request))
```

Dataset lưu think time raw, không cap. Max = 68 giờ (session bị bỏ hoang qua đêm). 8.8% entries vượt 60 giây. Runtime cap do scenario config quyết định — chuẩn InferenceX là `--inter-turn-delay-cap-seconds 60` (bao phủ p90 = 47.6s).

Phân phối think time cho thấy hai chế độ hoạt động rõ rệt: tool-call turns (p50 = 0.8s, CC gửi tool result ngay) và user-think turns (p90 = 47.6s, user đọc response rồi mới tiếp tục).

## Hash_ids và cache modeling

Mỗi request có danh sách `hash_ids`, mỗi block 64 tokens có một ID. Cache structure được mô hình hóa qua Longest Common Prefix (LCP) giữa hash_ids của consecutive turns:

- Turn N phát triển bình thường (prompt tăng): extend hash_ids của turn N-1. LCP cao, engine cache-hit phần prefix chung.
- Turn N shallow change (prompt đổi < 10%): trim hash_ids của turn N-1. LCP vẫn cao.
- Turn N bị compaction (prompt đổi > 10%): hash_ids hoàn toàn mới. LCP = 0, engine phải prefill lại từ đầu.

Ngưỡng compaction 10% dựa trên `|prompt_current - prompt_prev| / prompt_prev`. Khi Claude Code nén history (ở khoảng 150K tokens, prompt tụt còn 20 to 40K), tỉ lệ đổi vượt 10%, trigger reset.

Cache hit ước lượng (LCP > 0 giữa consecutive turns): 86.5%. Con số này cao hơn weka reference dataset (72%) vì sessions ngắn hơn (p50 = 37 vs 56 turns), nên ít compaction events hơn. Cache hit thực tế cần validate bằng Prometheus server-side metrics sau benchmark run, vì gateway-level cache hit bị trộn với non-CC traffic.

## Burst linearization

31% sessions có concurrent bursts: nhiều request chạy chồng lên nhau (started trước khi request trước kết thúc). Đây là subagent hoặc utility call chạy song song.

Dataset linearize bằng cách chọn 1 request đại diện cho mỗi burst (request có prompt tokens cao nhất, tức context nhiều nhất). Các request song song bị drop. Token loss: 0.7% input, nhưng output loss cao hơn (9.6%) vì các request bị drop có thể có output đáng kể.

Lý do linearize: weka replay client chạy tuần tự, không hỗ trợ concurrent requests trong cùng trace. Linearize giữ đúng thứ tự tuần tự để replay faithful.

## Compaction detection

```text
ratio = abs(prompt_current - prompt_prev) / max(prompt_prev, 1)
if ratio > 0.10: reset_context (hash_ids mới, LCP = 0)
```

Tỉ lệ compaction resets: 1,322 / 21,215 deltas (6.2%). Trong weka reference dataset (no-subagents), tỉ lệ này là 2.77%. Số cao hơn có thể vì gateway data noisy hơn, hoặc vì threshold 10% nhạy hơn threshold mà weka dùng.

## Build và run

Rebuild từ `price_research/`:

```bash
# 1. Pull data từ vMonitor (cần config ở ~/.claude/skills/vmonitor-data-analysis/config.json)
uv run python3 pull_cc_sessions.py    # output: /tmp/cc_v4_full.jsonl

# 2. Build weka traces
python3 build_minimax_cc_v4_weka.py /tmp/cc_v4_full.jsonl /tmp/minimax_cc_v4_weka
```

Run bằng AIPerf (fork thangquang09, branch `benchtool/agentx-weka`):

```bash
# Clone fork có WekaTraceLoader + weka fixes
git clone -b benchtool/agentx-weka https://github.com/thangquang09/aiperf.git
cd aiperf
pip install -e .
```

```bash
aiperf profile \
  --model <served-model-name> \
  --tokenizer <tokenizer-id> \
  --url http://127.0.0.1:8000 \
  --endpoint-type chat \
  --streaming \
  --input-file benchmarks/single_node/agentic/datasets/minimax_cc_v4_weka/ \
  --custom-dataset-type weka_trace \
  --no-fixed-schedule \
  --concurrency 1,4,8,16,24,32 \
  --benchmark-duration 300 \
  --benchmark-grace-period 120 \
  --inter-turn-delay-cap-seconds 60 \
  --warmup-request-count 20
```

Lưu ý:

- `--custom-dataset-type weka_trace` yêu cầu fork aiperf có `WekaTraceLoader`. Trên upstream NVIDIA aiperf `main`, loader này nằm trong PR #1053 "AgentX" (chưa merge). Dùng fork `thangquang09/aiperf` branch `benchtool/agentx-weka` (đây cũng là submodule tại `utils/aiperf-mooncake` trong `vngcloud/InferenceX`).
- `--no-fixed-schedule` bắt buộc. weka_trace lưu timestamps, nên aiperf mặc định bật fixed-schedule mode khi load dataset — replay theo timestamp gốc, ignore `--concurrency`. Flag này tắt auto fixed-schedule để dùng concurrency mode.
- `--inter-turn-delay-cap-seconds 60` cap think time ở 60s (chuẩn InferenceX). Dataset lưu raw (max 68h), không cap thì worker ngủ hàng giờ. 60s đủ bao phủ p90 (47.6s) mà không stall benchmark.
- Dataset 106MB (234 files). CI checkout xử lý bình thường.
- `--input-file` trỏ vào thư mục chứa 234 file JSON. WekaTraceLoader đọc tất cả `.json` trong thư mục.

## So sánh với dataset cũ (v3 mooncake)

| | v3 mooncake (cũ) | v4 weka (mới) |
| --- | --- | --- |
| Format | mooncake_trace (incremental delta) | weka trace (cumulative + hash_ids) |
| Source | Static dump `minimax-m2.5.jsonl` | vMonitor live pull (per-session) |
| Sessions | 249 | 234 (June 10-17 only) |
| Requests | 19,662 | 21,449 |
| Segments/traces | 5,626 segments (forest) | 234 traces (1 per session) |
| Turn/segment p50 | 2 | 37 |
| Compaction | Không mô hình hóa được | reset_context (6.2%) |
| Cache model | Warm-prefix share estimate (82.6%) | LCP-based hash_ids (86.5%) |
| Think time | Capped 60s | Raw (no cap) |
| Session continuity | Best-parent forest (derived) | Native session ID (direct) |

Lý do đổi format: mooncake DELTAS mode cộng dồn `input_length` qua các turn. Khi context bị compaction (prompt tụt từ 150K xuống 30K), mooncake không có cơ chế giảm accumulated input. Chia session tại compaction point thì mỗi segment chỉ có p50 = 2 turns, quá ngắn để benchmark có ý nghĩa. Weka giải quyết bằng `reset_context`: hash_ids đổi hoàn toàn, engine reset accumulated context, nhưng trace vẫn là 1 file liên tục với 37+ turns.

## Caveat

- Workload phục vụ capacity/cost/latency, không phải quality eval. Prompt/response thật bị mask, content synthetic.
- Think time raw có outlier lớn (max 68h). Phải cap ở runtime, nếu không benchmark stall.
- Output token fidelity 90.4%. 9.6% loss đến từ burst linearization (drop parallel requests). Input fidelity 97.9%.
- Cache hit 86.5% là ước lượng LCP-based, không phải measured cache hit. Validate bằng Prometheus sau run.
- Subagent sessions emit thành traces riêng, không nest vào parent. Trong production, subagent chạy concurrent với main agent, nhưng weka replay không hỗ trợ concurrent trong 1 trace.
- Chart SVG cũ (`minimax-source-pie.svg`, `minimax-production-histogram.svg`, `minimax-dataset-shape.svg`) tham chiếu data v3, cần regenerate cho v4.

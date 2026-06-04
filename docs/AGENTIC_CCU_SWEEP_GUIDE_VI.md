# Hướng dẫn chạy Agentic CCU Sweep (Mode 1) — Tiếng Việt

Tài liệu này hướng dẫn cách dùng InferenceX ở branch hiện tại để **benchmark khả năng
chịu tải (capacity) của một model bất kỳ** bằng cách replay lại trace của agent
coding, đẩy concurrency (CCU) lên dần để tìm điểm gãy throughput/latency.

> Đây là bản tóm tắt thực hành bằng tiếng Việt. Tài liệu gốc (chi tiết kỹ thuật,
> tiếng Anh):
> - [`AGENTIC_MODE1_CCU_SWEEP.md`](AGENTIC_MODE1_CCU_SWEEP.md) — Mode 1 capacity sweep
> - [`AIPERF_INTEGRATION.md`](AIPERF_INTEGRATION.md) — tích hợp AIPerf, sizing, verify
> - [`adr/0001-agentic-on-official-aiperf.md`](adr/0001-agentic-on-official-aiperf.md) — quyết định kiến trúc

---

## 1. Mode 1 là gì?

`agentic-replay` có 2 chế độ. **Mode 1 (capacity sweep)** là chế độ chính để ra số
báo cáo:

| | Mặc định (single-replay) | **Mode 1 (capacity sweep)** |
|---|---|---|
| Cách tính thời gian | Theo `timestamp`/`delay` trong trace | **`--no-fixed-schedule`** — chỉ chạy theo áp lực concurrency |
| Think-time (thời gian "suy nghĩ" giữa các lượt) | Giữ nguyên `delay` | **Bỏ đi** (`strip-trace-delays`) → think-time = 0 |
| `request-count` | = số dòng trong dataset (replay đúng 1 lần) | Đặt rõ; AIPerf **lấy mẫu lại (resample)** các session để đủ số đó |
| Concurrency | Là trần (ceiling) số session đồng thời | Là **biến được quét** (tăng dần tới khi vỡ SLA) |
| Mục đích | Tái hiện đúng hình dạng traffic đã ghi | **Tìm điểm gãy capacity/latency (số báo cáo)** |

Nói ngắn gọn: Mode 1 = "ép" model phục vụ nhiều request đồng thời nhất có thể, bỏ
hết thời gian nghỉ, rồi xem ở mức CCU nào thì latency tăng vọt.

---

## 2. Dataset — trace là gì?

Dataset là file JSONL kiểu **mooncake_trace**: mỗi dòng là một lượt (turn) với
`session_id`, `input_length`, `output_length`, `hash_ids`, `delay`.

Các trace đã commit sẵn trong repo (`benchmarks/single_node/agentic/datasets/`):

| Dataset | Số record | **max-model-len phải dùng** |
|---|---|---|
| `qwen3.5-4b-smoke.jsonl` | 12 | **8192** (chỉ để smoke test) |
| `agentic-coding-64k.jsonl` | 18,595 | **73728** |
| `agentic-coding-128k.jsonl` | 16,957 | **147456** |

Có thể lấy **một phần đầu** của trace bằng hậu tố `#N` trên đường dẫn, ví dụ
`agentic-coding-64k.jsonl#2000` = chỉ lấy 2000 dòng đầu. `max-model-len` không đổi
theo N (xem bảng trong [`AIPERF_INTEGRATION.md`](AIPERF_INTEGRATION.md#context-length-requirements-per-dataset-size-max-model-len-from-the-session-cumulative-max)).

### ⚠️ Quan trọng nhất: sizing `max-model-len`

Trace được replay như **hội thoại nhiều lượt**: context **tích lũy** qua từng lượt
(mỗi lượt mang theo toàn bộ lượt trước + câu trả lời làm prefix). Vì vậy độ dài
prompt thực tế **KHÔNG** phải là `input_length` của một dòng, mà là tổng dồn
`sum(input_length + output_length)` của cả session tới lượt đó.

→ Luôn lấy `max-model-len` từ **giá trị tổng-dồn theo session** (cột trong bảng trên),
**không** lấy theo từng dòng. Nếu lấy sai (quá nhỏ), server sẽ trả **HTTP 400** cho
các request vượt context → cả leg bị loại (xem mục 7).

---

## 3. Kịch bản test diễn ra thế nào?

Với mỗi mức concurrency trong `conc-list`, CI tạo **một job riêng**:

1. Launcher khởi động **server mới** (vLLM/SGLang...) với `--max-num-seqs = CONC`
   (mỗi mức CCU được đo "nguội" và độc lập).
2. Launcher xử lý dataset: cắt `#N` nếu có → bỏ `delay` nếu `strip-trace-delays`.
3. AIPerf replay trace ở đúng mức concurrency đó, resample session cho đủ
   `request-count`, giữ CCU request in-flight (back-pressure, zero think-time).
4. Sau khi chạy xong, **adapter kiểm tra fail-closed**: chỉ ghi kết quả nếu
   `request_count == kỳ vọng` **và** `error_request_count == 0`.
5. Kết quả tổng hợp vào `agg_bmk.json` (TTFT/TPOT/ITL/E2E p50–p99, throughput/GPU,
   tok/Watt...).

Vì context tích lũy, **prefix caching** (vLLM) / **RadixAttention** (SGLang) rất quan
trọng ở đây — các lượt sau "ấm" sẽ hit cache cao, giảm tải prefill.

---

## 4. Cách submit: model bất kỳ, runner bất kỳ, CCU bất kỳ

Có **3 bước**: (1) chuẩn bị launcher script, (2) thêm config key, (3) dispatch.

### Bước 1 — Launcher script cho model của bạn

Launcher được resolve theo công thức:

```
benchmarks/single_node/${model-prefix}_${precision}_h100_${framework}.sh
```

- Cách nhanh nhất: **copy** script mẫu sẵn có `qwen3-4b-2507_bf16_h100_vllm.sh`
  (đã hỗ trợ đầy đủ Mode 1 qua env: `STRIP_TRACE_DELAYS`, `REQUEST_COUNT`,
  `NO_FIXED_SCHEDULE`, `NUM_WARMUP_SESSIONS`), chỉ đổi tên file theo `model-prefix`
  mới và sửa `SERVED_MODEL_NAME` cùng các cờ serving cho phù hợp.
- ⚠️ **Dùng `model-prefix` riêng cho agentic-replay.** Nếu prefix trùng với một
  script fixed-seq-len đã tồn tại, launcher sẽ resolve nhầm sang script đó và fail
  (vì script fixed-seq-len không xử lý `INPUT_FILE`). Hãy đặt prefix riêng kiểu
  `mymodel-agentic`.

> Mẹo: script mẫu `qwen3-4b-2507_bf16_h100_vllm.sh` đã viết theo kiểu generic cho
> vLLM bf16 — phần lớn model dense text-only chỉ cần đổi `SERVED_MODEL_NAME` là chạy.

### Bước 2 — Thêm config key vào `.github/configs/nvidia-master.yaml`

Đây là template Mode 1 đầy đủ cho dataset 64k. **Đổi các phần in hoa** theo model/runner/CCU của bạn:

```yaml
MYMODEL-agentic-mode1-h100-vllm:          # <- tên config key (tùy bạn)
  image: vllm/vllm-openai:v0.21.0          # <- image serving
  model: ORG/MyModel-Instruct              # <- model trên HuggingFace
  model-prefix: mymodel-agentic            # <- phải khớp tên script ở Bước 1
  runner: h100-2x                          # <- runner (xem Bước 4)
  precision: bf16
  framework: vllm
  multinode: false
  scenarios:
    agentic-replay:
    - input-file: benchmarks/single_node/agentic/datasets/agentic-coding-64k.jsonl#2000
      custom-dataset-type: mooncake_trace
      max-model-len: 73728                 # <- theo bảng sizing (64k→73728, 128k→147456)
      benchmark-client: [aiperf]
      no-fixed-schedule: true              # Mode 1
      strip-trace-delays: true             # Mode 1 (bắt buộc)
      request-count: 2000                  # >= max(conc-list); xem mục 5
      num-warmup-sessions: 16              # làm ấm cache/CUDA graph trước khi đo
      search-space:
      - { tp: 1, conc-list: [16, 32, 64, 128, 256] }   # <- thang CCU bạn muốn quét
```

4 field Mode 1 (`no-fixed-schedule`, `strip-trace-delays`, `request-count`,
`num-warmup-sessions`) đều **optional, mặc định tắt** — không ảnh hưởng config cũ.

### Bước 3 — Dispatch (không cần PR)

```bash
gh api -X POST /repos/vngcloud/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='exp/mode1-vs-mode3-agentic-64k' \
  -f 'inputs[ref]=exp/mode1-vs-mode3-agentic-64k' \
  -f 'inputs[test-name]=MyModel agentic CCU sweep (conc 16..256)' \
  -f 'inputs[generate-cli-command]=test-config --config-keys MYMODEL-agentic-mode1-h100-vllm --config-files .github/configs/nvidia-master.yaml --no-evals --scenario-type agentic-replay' \
  -f 'inputs[duration-override]='
```

- `ref` (top-level): branch chứa **wiring `agentic-replay`** — hiện tại là
  `exp/mode1-vs-mode3-agentic-64k` (sau khi merge thì dùng `main`).
- `inputs[ref]`: branch chứa **code/config** muốn test (thường giống `ref`).
- ⚠️ Nếu dispatch từ branch **không có** wiring `agentic-replay`, job sẽ rơi vào
  bucket `single-node` thường và **fail ngay** (không set `INPUT_FILE`). Kiểm tra
  tên job: `agentic-replay /...` (đúng) vs `single-node /...` (sai branch).

### Bước 4 — Chọn runner

`runner` trong config quyết định phần cứng (từ `.github/configs/runners.yaml`):

| `runner` | Phần cứng GreenNode | Dùng cho |
|---|---|---|
| `h100-1x` | `h100-greennode_00` — **1× H100** | TP=1 |
| `h100-2x` | `h100-greennode_01` — **2× H100** | TP=1 hoặc TP=2 |

- TP=2 **bắt buộc** dùng `h100-2x` (nếu rơi vào `h100-1x` sẽ lỗi
  `World size (2) > available GPUs (1)`).
- Có các runner khác trong `runners.yaml` (`h200`, `b200`, ...) nhưng không phải của
  GreenNode — chỉ dùng nếu bạn chắc chắn có quyền truy cập.

---

## 5. Chọn tham số: request-count, CCU ladder, warmup

### Phân biệt `#N` (hồ session) vs `request-count` (số request) — đọc trước

Hai con số này **khác nhau**, đừng nhầm:

- **`#N`** trên `input-file` = **kích thước "hồ" session** mà AIPerf được phép lấy
  mẫu (chỉ là `head -n N` của file). Trace đã được trộn ngẫu nhiên sẵn nên N dòng
  đầu vẫn đại diện đúng cho cả tập.
- **`request-count`** = **số request thực sự gửi đi** mỗi leg; AIPerf **resample**
  (lặp lại) các session trong hồ cho đủ số này.

→ Quy tắc: đặt **`#N` ≥ `request-count`** để mỗi request đến từ một session khác
nhau, tránh lặp lại làm cache hit bị thổi phồng giả tạo. Ví dụ `request-count: 2000`
thì dùng `...64k.jsonl#2500` trở lên.

### `request-count` — bao nhiêu thì hội tụ mà KHÔNG cần full dataset?

Không cần chạy hết 18,595 request. Có **2 ràng buộc**, lấy cái lớn hơn:

1. **Đại diện workload:** phân phối độ dài prompt của mẫu đã khớp ~hoàn hảo với cả
   tập **từ N ≈ 500** (đo thực tế trên trace 64k: p50/p90/p95/p99 lệch < 2%). Tức là
   về mặt "mẫu có giống tập gốc không" thì rất nhanh đạt.
2. **Ổn định percentile latency (p99):** đây mới là ràng buộc quyết định. Để ước
   lượng p99 ổn định cần đủ mẫu ở đuôi: ~**10 mẫu đuôi → N ≥ 1000**; ~**25–30 mẫu
   đuôi → N ≈ 2500–3000** (p99 mượt hơn nhiều).
3. **Cộng thêm ramp theo concurrency:** ở CCU cao, ~CCU request đầu/cuối là giai
   đoạn fill/drain, không steady-state. Cộng thêm ~1×CCU vào N.

Gộp lại (đã tính cả ramp):

| CCU lớn nhất | Tối thiểu (hard) | **Khuyến nghị (hội tụ p99)** | Ghi chú |
|---|---|---|---|
| ≤ 64 | 64 | **~1000** | workload đã bão hòa từ ~500; 1000 đủ cho p99 |
| 128 | 128 | **~1500–2000** | ~25 mẫu đuôi sau ramp |
| 256 | 256 | **~2500–3000** | đuôi ổn định kể cả sau ramp 256 |

- **Bắt buộc (hard rule):** `request-count >= max(conc-list)`, nếu không config bị từ
  chối trước khi dispatch (`validate_request_count_vs_conc`).
- **Kết luận:** ~**3000 là trần thực dụng** cho mọi thang tới 256 — chạy full 18,595
  tốn ~6× thời gian mà p99 không ổn định thêm. Chỉ nâng N nếu cần đo tới **p99.9**.

### `conc-list` (thang CCU)
- Thang điển hình tìm capacity: `[8, 16, 32, 64, 128, 256]`.
- Dừng ở leg đầu tiên vỡ SLA TTFT/ITL — đó là **biên capacity**.
- Muốn khoanh điểm gãy chính xác → thêm bậc dày quanh đó (vd `160, 192, 224`).

### `num-warmup-sessions`
- Làm ấm prefix cache / CUDA graph trước khi đo. Smoke dùng `1`; sweep thật dùng
  `16` (hoặc `32` cho thang cao).

---

## 6. Lưu ý đặc biệt cho 128k

Dataset 128k chạy được nhưng **nặng hơn nhiều**:

1. `max-model-len: 147456` (lấy từ session-cumulative max ≈ 133,851).
2. **Áp lực KV cache rất lớn ở CCU cao.** Đã quan sát: 128k ở conc=32 đã có
   preemption. Khi đó hãy **hạ concurrency**, **không** hạ context window (hạ window
   sẽ gây HTTP 400 over-context).
3. Cân nhắc TP cao hơn (TP=2 trên `h100-2x`) để có thêm KV, hoặc dùng subset `#N`
   nhỏ hơn để giảm thời gian chạy.
4. Model phải hỗ trợ context ≥ 147k (đa số model hiện đại OK; kiểm tra config model).

---

## 7. Kiểm tra một run có hợp lệ không

Job xanh chỉ chứng minh mọi request đã hoàn tất. Vẫn phải đọc `server.log`:

```bash
# 1. Tỷ lệ HTTP 200/400 — mọi request benchmark phải 200
grep -oE '"POST /v1/chat/completions HTTP/1.1" [0-9]+' server.log | awk '{print $NF}' | sort | uniq -c

# 2. Đếm preemption (tín hiệu stress hợp lệ)
grep -ic preempt server.log
```

- Toàn **400** → sizing `max-model-len` sai (over-context) → **không** phải kết quả
  capacity thật → sửa context window, chạy lại.
- **5xx / connection** → server OOM/crash → đây mới là giới hạn phần cứng thật.
- Nhiều `preempt` nhưng 0 lỗi HTTP → leg hợp lệ, latency phản ánh đúng mức stress.
- Kiểm tra cache thực sự hit: vLLM log `Prefix cache hit rate: NN%`; SGLang log
  `#new-token / #cached-token` mỗi prefill-batch.

---

## 8. Một số lưu ý theo loại model

- **Dense, text-only (vd Qwen3-4B):** sạch nhất, engine chạy backend tối ưu mặc
  định → đây là lựa chọn an toàn nhất cho sweep.
- **Model có context dài:** đảm bảo model hỗ trợ context ≥ `max-model-len` bạn đặt
  (64k→73728, 128k→147456); nếu không, request sẽ bị từ chối over-context (HTTP 400).
- **Hybrid-Mamba (vd Qwen3.5-4B gốc):** vLLM **tự tắt prefix caching** → mất ý nghĩa
  của agentic-replay (vốn dựa vào cache lượt sau). Tránh dùng cho sweep này.

---

## 9. Checklist nhanh trước khi dispatch

- [ ] Có launcher script đúng tên `${model-prefix}_${precision}_h100_${framework}.sh`
      với `model-prefix` riêng cho agentic.
- [ ] `max-model-len` lấy đúng từ bảng sizing (64k→73728, 128k→147456).
- [ ] `request-count >= max(conc-list)`, và đủ lớn (~10× CCU max) cho hội tụ.
- [ ] `no-fixed-schedule: true` + `strip-trace-delays: true` (Mode 1).
- [ ] `runner` đúng (`h100-2x` nếu TP=2).
- [ ] Dispatch từ branch có wiring `agentic-replay` (kiểm tra tên job sau khi chạy).
- [ ] Sau khi chạy: đọc `server.log` để phân biệt preemption vs fail thật.

---

## 10. Test nhanh tại local (không qua CI)

Chạy thử adapter trực tiếp với một vLLM server đang chạy:

```bash
source .venv/bin/activate
uv run python utils/bench_serving/aiperf_adapter.py \
  --model ORG/MyModel-Instruct --url http://0.0.0.0:8000 --endpoint-type chat \
  --concurrency 16 --request-count 200 \
  --input-file benchmarks/single_node/agentic/datasets/agentic-coding-64k.jsonl#500 \
  --custom-dataset-type mooncake_trace \
  --result-filename mymodel-test --result-dir /tmp/mymodel-test
```

> Lưu ý: local dùng **một server cho mọi mức** → cache ấm dồn sang leg sau, làm "đẹp"
> các mức CCU cao. Để so sánh apples-to-apples (mỗi leg server nguội riêng), dùng
> đường CI matrix.
</content>
</invoke>

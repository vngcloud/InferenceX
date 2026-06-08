# AIPerf Search Recipes — Push & Dispatch Guide

Hướng dẫn tập trung để chạy **tìm điểm vận hành tối ưu** bằng AIPerf trên GitHub
Actions. Thay vì benchmark một mức concurrency cố định, AIPerf dùng **Bayesian
Optimization (BO)** tự dò các mức concurrency trong khoảng `[min, max]` và trả về
điểm tốt nhất còn thỏa SLA.

> Phần fixed-sequence (concurrency cố định) xem ở
> [`AIPERF_INTEGRATION.md`](AIPERF_INTEGRATION.md). File này chỉ nói về chế độ
> `--search-recipe`.

- Adapter: `utils/bench_serving/aiperf_adapter.py` (chỉ ủy quyền — BO do AIPerf tự
  chạy).
- AIPerf version CI cài: **0.9.0** (PyPI), chốt trong `benchmarks/benchmark_lib.sh`.
  (Clone cục bộ ở `/Users/lap15120/greennode-code/aiperf` là 0.10.0 — **lệch
  version**, đừng dùng làm chuẩn cho layout artifact.)
- SLA được AIPerf áp ở **p95** mặc định (`SLAFilter.stat = "p95"`).
- Mỗi điểm BO dò có thể đo theo **số request** (mặc định) hoặc theo **thời gian**
  (`benchmark-duration`, opt-in) — xem [mục 3.5](#35-chế-độ-đo-theo-thời-gian-duration-based).

---

## 1. Hai chế độ đang hỗ trợ

| Recipe | Mục tiêu tối ưu | SLA nhận được | Số ràng buộc |
|---|---|---|---|
| `max-throughput-itl-sla` | Throughput cao nhất | **chỉ ITL** (`sla-ms`) | đúng 1 |
| `max-concurrency-under-sla` | **CCU cao nhất** | ITL (`sla-ms`) **và/hoặc** TTFT (`ttft-sla-ms`) | ≥ 1 (tới 2) |

Điểm mấu chốt:

- **Muốn tối đa throughput** → dùng `max-throughput-itl-sla`. Recipe này theo bản
  chất AIPerf **chỉ nhận 1 ràng buộc** (ITL). Không ép kèm TTFT được.
- **Muốn ép đồng thời 2 ràng buộc** (ví dụ ITL + TTFT) → bắt buộc dùng
  `max-concurrency-under-sla`; nhưng lúc đó mục tiêu là **tối đa CCU**, không phải
  throughput. Đọc throughput tại điểm winner nếu cần.

> AIPerf 0.9.0 còn có `max-throughput-ttft-sla` và `max-goodput-under-slo` (ép cả
> 3: TTFT+TPOT+E2E), nhưng adapter hiện **chưa expose**. Khi cần mở thêm, thêm vào
> registry `SEARCH_RECIPES` trong `aiperf_adapter.py`.

---

## 2. Quy đổi SLA (ms ⇄ tok/s/user)

AIPerf nhận SLA theo **mili-giây**, không theo tok/s. Quy đổi:

```
ITL (ms) = 1000 / (tok/s/user)
tok/s/user = 1000 / ITL (ms)
```

| Mục tiêu | Giá trị `sla-ms` |
|---|---|
| ≥ 20 tok/s/user | **50** |
| ≥ 25 tok/s/user | 40 |
| ≥ 50 tok/s/user | 20 |

`ttft-sla-ms` là mili-giây trực tiếp: TTFT ≤ 5s → `ttft-sla-ms: 5000`.

---

## 3. Config mặc định hiện tại

Trong `.github/configs/nvidia-master.yaml`, config
`qwen3-4b-2507-bf16-h200-greennode-vllm-search-recipe`:

```yaml
qwen3-4b-2507-bf16-h200-greennode-vllm-search-recipe:
  image: vllm/vllm-openai:v0.21.0
  model: Qwen/Qwen3-4B-Instruct-2507
  model-prefix: qwen3-4b-2507
  runner: h200-greennode_00      # dùng trực tiếp làm runs-on, không cần runners.yaml
  precision: bf16
  framework: vllm
  multinode: false
  scenarios:
    fixed-seq-len:
    - isl: 1024
      osl: 128
      benchmark-client: [aiperf]          # BẮT BUỘC để đi qua aiperf, không phải native
      search-space:
      - { tp: 1, conc-start: 8, conc-end: 32, search-recipe: max-throughput-itl-sla, sla-ms: 50, search-max-iterations: 6, benchmark-duration: 20, benchmark-grace-period: 10 }
```

Ý nghĩa các field trong `search-space`:

| Field | Map sang AIPerf | Ý nghĩa |
|---|---|---|
| `conc-start` | `--concurrency-min` | cận dưới khoảng BO dò |
| `conc-end` | `--concurrency-max` | cận trên khoảng BO dò |
| `search-recipe` | `--search-recipe` | tên recipe (xem mục 1) |
| `sla-ms` | `--itl-sla-ms` | SLA ITL p95 (ms) — = 20 tok/s/user khi để 50 |
| `ttft-sla-ms` | `--ttft-sla-ms` | SLA TTFT p95 (ms) — chỉ dùng cho `max-concurrency-under-sla` |
| `search-max-iterations` | `--search-max-iterations` | trần số trial BO |
| `benchmark-duration` | `--benchmark-duration` | (opt-in) đo mỗi điểm theo **giây** thay vì số request — xem mục 3.5 |
| `benchmark-grace-period` | `--benchmark-grace-period` | (opt-in) thời gian rút (giây) cho request đang dang dở sau khi hết duration |

> ⚠️ **`search-max-iterations` phải > 5.** BO seed mặc định **5 điểm Sobol**
> (`n_initial_points`) trước khi fit GP, và AIPerf bắt buộc
> `n_initial_points < max_iterations`. Đặt `≤ 5` sẽ bị reject ngay ở khâu validate
> config (`iter=3` đã từng fail vì lý do này). Smoke dùng `6`; production nên
> **10–20** (AIPerf khuyến nghị).

### Ví dụ entry cho chế độ max-CCU 2 ràng buộc

```yaml
      - { tp: 1, conc-start: 8, conc-end: 64, search-recipe: max-concurrency-under-sla, sla-ms: 50, ttft-sla-ms: 5000, search-max-iterations: 16 }
```

→ tìm CCU lớn nhất mà **p95 ITL ≤ 50ms (≥20 tok/s/user) VÀ p95 TTFT ≤ 5000ms**.

---

## 3.5. Chế độ đo theo thời gian (duration-based)

Mặc định, mỗi điểm concurrency BO dò sẽ đo theo **số request cố định**
(`request-count = concurrency × 10`, warmup `× 2`). Nhược điểm: với OSL dài,
mỗi request mất rất lâu → một iteration có thể vài phút, BO 6 iteration mất ~38
phút (smoke OSL=1024 đã gặp). Với OSL ngắn lại ít sample.

**Chế độ duration-based** (opt-in) đo mỗi điểm theo **một khoảng thời gian cố
định** thay vì đếm request — wall-clock mỗi iteration ổn định, không phụ thuộc
throughput. Đây là cách đo đúng chuẩn steady-state của team (`benchmarking-plan.md`:
≥5 phút steady-state).

### Bật như thế nào

Đây là **option A — opt-in, mặc định vẫn là request-count.** Chỉ cần thêm 2 field
vào entry `search-space`:

```yaml
- { tp: 1, conc-start: 8, conc-end: 32, search-recipe: max-throughput-itl-sla, sla-ms: 50, search-max-iterations: 6, benchmark-duration: 20, benchmark-grace-period: 10 }
```

- Có `benchmark-duration` → adapter bỏ `--request-count`/`--warmup-request-count`,
  truyền `--benchmark-duration`/`--benchmark-grace-period` cho AIPerf.
- **Loại trừ lẫn nhau (mã ép buộc):** đúng **một** trong `request-count` /
  `benchmark-duration` được set. Đặt cả hai (hoặc không cái nào) → adapter
  `parser.error`. Trong `nvidia-master.yaml`, KHÔNG có field nào → mặc định
  request-count.
- `benchmark-duration` **kết hợp được với** `--search-recipe` (BO áp duration cho
  **từng điểm** nó dò). Chỉ `--convergence-metric` và `--variant` mới xung đột với
  search-recipe, duration thì không.

### Đặt duration bao nhiêu?

Đơn vị là **giây**. Quy tắc dựa trên thời gian decode mỗi request tại SLA:

```
decode_mỗi_request ≈ osl × ITL_SLA(s) = osl × (sla-ms / 1000)
benchmark-duration   ≥ 5 × decode_mỗi_request      (gom ≥5 "lớp" request liên tiếp)
benchmark-grace-period ≥ 1 × decode_mỗi_request    (đủ để request cuối hoàn tất)
```

Ví dụ smoke (`osl=128`, `sla-ms=50` → ITL 0.05s): decode ≈ `128×0.05 = 6.4s`.
→ chọn `duration=20` (~3 lớp, đủ cho smoke nhanh), `grace=10`.

| Mục đích | osl | duration | grace | Ghi chú |
|---|---|---|---|---|
| **Smoke nhanh** (validate plumbing) | 128 | **20** | **10** | mỗi iter ~30s; BO 6 iter ~3 phút |
| **Đo thật** (steady-state) | 1024 | **≥ 300** | **60** | ≥5 phút/điểm theo `benchmarking-plan.md` |

> ⚠️ Smoke `duration=20s` chỉ để kiểm tra đường ống chạy thông, **số liệu không
> dùng để kết luận hiệu năng** (quá ngắn, ít sample). Đo thật phải ≥300s/điểm.

### Vì sao smoke nhanh hơn hẳn

| | request-count (cũ) | duration (mới) |
|---|---|---|
| Đơn vị mỗi điểm | `conc×10` request | thời gian cố định |
| Wall-clock 1 iteration | phụ thuộc OSL/throughput (OSL=1024 → ~6 phút) | **cố định** (smoke: 30s) |
| BO 6 iteration | ~38 phút | **~3 phút** |

---

## 4. Quy trình: sửa config → validate → push → dispatch

### Bước 1 — Sửa config

Thêm/sửa entry trong `search-space` của `nvidia-master.yaml` (mục 3).

### Bước 2 — Validate cục bộ trước khi đẩy

```bash
cd /Users/lap15120/greennode-code/InferenceX
uv run --no-project python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --model-prefix qwen3-4b-2507
```

Kiểm tra entry sinh ra có `"benchmark-client": "aiperf"`, đúng `search-recipe`,
`concurrency-min/max`, `sla-ms`, `search-max-iterations`. Nếu dùng duration-based,
phải thấy `"benchmark-duration"` và `"benchmark-grace-period"` được emit ra (chúng
chỉ xuất hiện khi có trong source config — mặc định không có).

### Bước 3 — Push lên feature branch

```bash
git add .github/configs/nvidia-master.yaml
git commit -m "feat(aiperf-search): <mô tả>"
git push origin feat/aiperf-search-recipe
```

> Quy ước hiện tại: **chỉ push branch, không mở PR.** Dispatch thủ công để smoke.

### Bước 4 — Dispatch lên Actions

> 🔴 **`ref` (top-level) BẮT BUỘC là feature branch**, KHÔNG phải `main`.
> Workflow YAML (`e2e-tests.yml` + `benchmark-tmpl.yml`) chứa plumbing
> `benchmark-client`/`search-recipe` **chỉ tồn tại trên feature branch**. Nếu để
> `ref=main`, workflow cũ chạy → bỏ qua aiperf, rớt về native ở concurrency cố
> định (đây đúng là lỗi đã gặp). `inputs[ref]` cũng để feature branch để checkout
> đúng config + adapter.

```bash
gh api -X POST \
  /repos/vngcloud/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='feat/aiperf-search-recipe' \
  -f 'inputs[ref]=feat/aiperf-search-recipe' \
  -f 'inputs[test-name]=qwen3-4b-2507 aiperf BO search smoke' \
  -f 'inputs[generate-cli-command]=full-sweep --config-files .github/configs/nvidia-master.yaml --model-prefix qwen3-4b-2507' \
  -f 'inputs[duration-override]='
```

POST không trả body/run-id.

### Bước 5 — Theo dõi

```bash
RUN_ID=$(gh run list --repo vngcloud/InferenceX --workflow e2e-tests.yml \
  --event workflow_dispatch --limit 1 --json databaseId,headBranch \
  --jq '.[0] | "\(.databaseId) \(.headBranch)"')
echo "$RUN_ID"   # headBranch PHẢI = feat/aiperf-search-recipe

gh run watch "${RUN_ID%% *}" --repo vngcloud/InferenceX --exit-status
gh run view  "${RUN_ID%% *}" --repo vngcloud/InferenceX --log-failed
```

---

## 5. Đọc kết quả

```bash
gh run download <RUN_ID> --repo vngcloud/InferenceX -n results_bmk -D ./results
```

Trong file kết quả của adapter, các field do search bổ sung:

| Field | Ý nghĩa |
|---|---|
| `benchmark_client` | phải là `aiperf` (nếu là `inferencex_native` → sai ref/plumbing) |
| `search_recipe` | tên recipe đã chạy |
| `max_concurrency` | concurrency winner do BO chọn |
| `sla_met` | `true` nếu winner thật sự thỏa SLA; `false` nếu BO không tìm được điểm nào thỏa (fallback best-effort) |
| `total_token_throughput`, `p95_itl_ms`, `p95_ttft_ms`, ... | metric tại điểm winner |

### Artifact BO thô (`aiperf_search_<RESULT_FILENAME>`)

Ngoài `results_bmk`, mỗi run search còn upload riêng artifact `aiperf_search_*`
(thêm ở `benchmark-tmpl.yml`, chỉ chạy khi `search-recipe != ''`). Tải về để debug
BO mà không kéo nguyên cây multi-GB:

```bash
gh run download <RUN_ID> --repo vngcloud/InferenceX -n aiperf_search_<RESULT_FILENAME> -D ./search
```

Gồm: `search_history.json`, mọi `profile_export_aiperf.json`,
`profile_export_aiperf_aggregate.json`, và `logs/aiperf.log`. **Cố tình loại** các
file thô khổng lồ (`profile_export_raw.jsonl`, `inputs.json`).

**Layout artifact 0.9.0 (đã verify trên source — KHÔNG phải `concurrency_<v>/`):**

```
<result>_aiperf/
├── search_history.json
├── aggregate/profile_export_aiperf_aggregate.json
└── search_iter_<NNNN>/                 # <NNNN> = iteration_idx, 4 chữ số 0-pad
    └── profile_runs/
        └── run_<MMMM>/                 # <MMMM> = trial_index+1, 4 chữ số 0-pad
            └── profile_export_aiperf.json
```

> ⚠️ **Cell identity của BO là ITERATION (`iteration_idx`), KHÔNG phải
> concurrency.** Adapter map winner qua `search_history.json →
> best_trials[0].iteration_idx`, rồi đọc
> `search_iter_<idx>/profile_runs/run_*/profile_export_aiperf.json`. Bản cũ giả
> định đường dẫn theo concurrency (`concurrency_<v>/...`) — layout đó **chưa từng
> tồn tại**, đã gây `FileNotFoundError` ở smoke đầu (fix ở commit `e73ab5c`).
> `search_history.json.best_trials[0]` chứa `iteration_idx`, `objective_values`,
> `variation_values` (key chấm như `phases.profiling.concurrency`), `feasible`,
> `feasible_count`, `pareto_rank`.

---

## 6. Lỗi thường gặp

| Triệu chứng | Nguyên nhân | Khắc phục |
|---|---|---|
| `benchmark_client = inferencex_native` | dispatch với `ref=main` | dispatch lại với `ref=feat/aiperf-search-recipe` |
| `n_initial_points (5) must be < max_iterations` | `search-max-iterations ≤ 5` | đặt ≥ 6 (production 10–20) |
| recipe báo thiếu SLA | thiếu `sla-ms`/`ttft-sla-ms` bắt buộc | `max-throughput-itl-sla` cần `sla-ms`; `max-concurrency-under-sla` cần ≥1 trong hai |
| `sla_met = false` | không mức conc nào trong `[min,max]` thỏa SLA | nới SLA, hạ `conc-min`, hoặc xem lại serving params |
| `exactly one of --request-count or --benchmark-duration` | set cả hai hoặc không cái nào | bỏ bớt — duration-based chỉ cần `benchmark-duration`; mặc định không set field nào (request-count tự suy ra) |
| `FileNotFoundError` khi đọc winner export | đang chạy adapter cũ (map theo `concurrency_<v>/`) | dùng adapter đã fix (`e73ab5c`+): map winner qua `iteration_idx` → `search_iter_<NNNN>/` |
| số liệu duration "lạ"/ít sample | `benchmark-duration` quá ngắn (vd smoke 20s) | đo thật phải ≥300s/điểm (mục 3.5) |

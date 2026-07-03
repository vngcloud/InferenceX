# Hướng dẫn chạy benchmark với remote endpoint(s)

Tài liệu này hướng dẫn cách khai báo config và dispatch một job benchmark khi
model **đã được serve sẵn trên một host khác** (không phải trên chính runner
chạy client). Đây gọi là chế độ **remote** (`remote:` block trong master
config), khác với chế độ mặc định là runner tự serve model rồi tự benchmark.

Tham khảo thêm:
- [`docs/REMOTE_AIPERF_DOCKER.md`](REMOTE_AIPERF_DOCKER.md) — chi tiết kỹ thuật
  về cách client remote chạy AIPerf và tuỳ chọn `aiperf-docker-image`.
- [`.github/configs/CONFIGS.md`](../.github/configs/CONFIGS.md) — format tổng
  quát của master config.
- [`.github/workflows/README.md`](../.github/workflows/README.md) — cách dùng
  `generate_sweep_configs.py` và dispatch `e2e-tests.yml` nói chung.

## 1. Điều kiện của chế độ remote

Theo `utils/matrix_logic/validation.py` (`RemoteConfig`,
`SingleNodeMasterConfigEntry.remote_requires_agentic_replay_only`):

- `remote:` **chỉ** được dùng cho config single-node có **duy nhất** scenario
  `agentic-replay` (không được kèm `fixed-seq-len` hay `agentic-coding`).
  Khai báo `remote:` cùng với các scenario khác sẽ bị validation chặn ngay khi
  load config.
- `benchmark-client` của scenario `agentic-replay` phải là `aiperf`.
- `runner` của config remote thường là `benchmark-client` — vì runner này
  **không serve model**, nó chỉ chạy AIPerf client bắn request tới endpoint ở
  xa.

## 2. Các field cần khai báo trong `remote:`

Khai báo trong `.github/configs/nvidia-master.yaml` (hoặc file master config
tương ứng), bên trong entry của bạn, ngang hàng với `scenarios:`:

| Field | Bắt buộc | Ý nghĩa |
|---|---|---|
| `url` | Có | (Các) endpoint OpenAI-compatible của model đang serve, ví dụ `http://192.168.4.13:8000`. Có thể là 1 string hoặc 1 list nhiều URL. |
| `server-metrics-url` | Không | (Các) URL Prometheus metrics của server serving, cùng format với `url`. |
| `gpu-telemetry-url` | Không | URL DCGM exporter để lấy GPU telemetry. |
| `aiperf-docker-image` | Không | Tên:tag một image AIPerf đã build sẵn trên runner `benchmark-client` (ví dụ `aiperf:0.8.0`). **Lưu ý:** trường này hiện đang *inert* — xem mục "Giới hạn hiện tại" bên dưới. |

Nếu `url` (hoặc `server-metrics-url`) là một **list nhiều URL** (model được
serve bởi nhiều instance), `validation.py` sẽ tự nối chúng thành chuỗi
comma-separated theo đúng format mà AIPerf hiểu — AIPerf sẽ round-robin request
qua các endpoint đó.

### Ví dụ 1 — một endpoint duy nhất

```yaml
qwen3-4b-weka-bf16-bench-client-sglang-remote-smoke:
  image: lmsysorg/sglang:v0.5.14-cu130
  model: Qwen/Qwen3-4B
  model-prefix: qwen3-4b-weka
  runner: benchmark-client
  precision: bf16
  framework: sglang
  multinode: false
  remote:
    url: http://192.168.4.13:8000
    server-metrics-url: http://192.168.4.13:8000/metrics
    gpu-telemetry-url: http://192.168.4.13:9400/metrics
  scenarios:
    agentic-replay:
    - custom-dataset-type: weka_trace
      public-dataset: semianalysis_cc_traces_weka_with_subagents_060826
      max-model-len: 40960
      benchmark-client: [aiperf]
      tokenizer: Qwen/Qwen3-4B
      duration: 90
      search-space:
      - { tp: 1, ep: 1, conc-list: [1] }
```

### Ví dụ 2 — nhiều endpoint (multi-instance)

```yaml
deepseek-coder-v2-lite-weka-fp8-bench-client-vllm-remote-multi-endpoint-smoke:
  image: vllm/vllm-openai:v0.21.0
  model: deepseek-coder-v2-lite-fp8
  model-prefix: deepseek-coder-v2-lite-weka
  runner: benchmark-client
  precision: fp8
  framework: vllm
  multinode: false
  remote:
    url:
    - http://192.168.4.13:8000
    - http://192.168.4.13:8001
    server-metrics-url:
    - http://192.168.4.13:8000/metrics
    - http://192.168.4.13:8001/metrics
    gpu-telemetry-url: http://192.168.4.13:9400/metrics
  scenarios:
    agentic-replay:
    - custom-dataset-type: weka_trace
      public-dataset: semianalysis_cc_traces_weka_with_subagents_060826
      max-model-len: 40960
      benchmark-client: [aiperf]
      tokenizer: RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8
      num-dataset-entries: 949
      duration: 900
      search-space:
      - { tp: 1, ep: 1, conc-list: [2, 4, 8, 16] }
```

Ghi chú các field khác trong `scenarios.agentic-replay`:
- `custom-dataset-type: weka_trace` + để trống `input-file`/`public-dataset`
  → dùng bộ dataset public mặc định
  `semianalysis_cc_traces_weka_with_subagents_060826`. Khai báo
  `public-dataset` để chọn dataset public khác, hoặc `input-file` cho trace
  nội bộ (repo-relative JSONL).
- `tokenizer`: HF id của tokenizer — nên khớp với model đang serve ở remote
  endpoint.
- `tp`/`ep` trong `search-space` không dùng để serve model (vì model đã được
  serve sẵn) — chúng chỉ đi vào tên kết quả/label, không ảnh hưởng job.
- `image` vẫn phải khai báo dù runner không serve model — vì trong chế độ
  remote, `image` là container **host cho việc chạy AIPerf client**, không
  phải server image (xem `docs/REMOTE_AIPERF_DOCKER.md`).

## 3. Validate config trước khi dispatch

Luôn kiểm tra local trước khi bắn job lên CI:

```bash
# Kiểm tra YAML hợp lệ
python3 -c "import yaml; yaml.safe_load(open('.github/configs/nvidia-master.yaml'))"

# In ra matrix entry mà generate_sweep_configs.py sẽ sinh ra cho đúng config key của bạn
python3 utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-keys <your-config-key> \
  --config-files .github/configs/nvidia-master.yaml
```

Nếu thiếu field bắt buộc, sai kiểu dữ liệu, hoặc khai báo `remote:` cùng
scenario không phải `agentic-replay`, lệnh trên sẽ báo lỗi validation ngay —
không cần chờ CI chạy mới biết.

## 4. Dispatch job lên GitHub Actions

Job được dispatch qua workflow `e2e-tests.yml` (`workflow_dispatch`), workflow
này luôn chạy từ nhánh `main`, còn code/config cần test nằm ở field
`ref` (nhánh của bạn):

```bash
gh api --method POST -H "Accept: application/vnd.github+json" \
  /repos/vngcloud/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref=main \
  -f 'inputs[ref]=<tên-nhánh-của-bạn>' \
  -f 'inputs[generate-cli-command]=test-config --config-keys <your-config-key> --config-files .github/configs/nvidia-master.yaml' \
  -f 'inputs[test-name]=<tên-hiển-thị-trên-Actions-UI>'
```

Giải thích input:
- `ref` (top-level, luôn `main`): nhánh chứa workflow file để chạy.
- `inputs[ref]`: nhánh/commit chứa **config và code cần test** (branch của
  bạn, ví dụ `exp/my-remote-config`). Nếu bỏ trống, mặc định dùng SHA của
  chính lần dispatch.
- `inputs[generate-cli-command]` (bắt buộc): CLI command truyền cho
  `generate_sweep_configs.py` — nên test local trước bằng lệnh ở mục 3.
  Có thể dùng `full-sweep`, `runner-model-sweep`, hoặc `test-config`
  (khuyến nghị cho một config cụ thể, hỗ trợ wildcard `*`/`?` trong
  `--config-keys`).
- `inputs[test-name]`: tên hiển thị trong GitHub Actions UI cho dễ tìm.
- `inputs[duration-override]` (optional): override `duration` (giây) của mọi
  entry trong matrix, để trống thì dùng giá trị trong config.

Lệnh POST không trả về run ID — tìm run vừa tạo bằng:

```bash
gh run list -R vngcloud/InferenceX -w e2e-tests.yml -L 5
```

Có thể dispatch trực tiếp qua GitHub Actions UI: vào tab **Actions** → chọn
workflow **End-to-End Tests** → **Run workflow** → điền các input tương ứng.

## 5. Theo dõi và debug job

```bash
gh run view <run-id> --repo vngcloud/InferenceX --json status,jobs \
  -q '.jobs[] | "\(.status)/\(.conclusion // "-")  \(.name)"'
```

Một số lỗi thường gặp:
- **Fail ngay ở bước "Launch job script" (~20s, exit 127)** — thiếu hoặc sai
  tên script trong `benchmarks/`. Không liên quan riêng đến chế độ remote.
- **Fail lâu hơn** — thường là lỗi kết nối tới remote endpoint (`url` sai/
  không reachable từ runner `benchmark-client`), hoặc lỗi khi cài AIPerf
  (xem log `install_agentic_deps` trong `benchmark_lib.sh`).
- Log server không đọc được giữa chừng job (`BlobNotFound`) — chỉ đọc được
  sau khi job hoàn tất.

## 6. Secret cần thiết

Nếu remote endpoint yêu cầu xác thực, biến môi trường `REMOTE_API_KEY` được
set từ secret `REMOTE_ENDPOINT_API_KEY` khi `remote-url` khác rỗng (xem
`.github/workflows/benchmark-tmpl.yml`). Nếu endpoint remote không cần API
key, có thể để trống secret này — hệ thống sẽ fallback về giá trị `EMPTY`.

## 7. Giới hạn hiện tại (aiperf-docker-image)

Field `aiperf-docker-image` đã được plumbing đầy đủ từ config → matrix →
workflow inputs → env var `AIPERF_DOCKER_IMAGE` → `install_agentic_deps` trong
`benchmark_lib.sh`, nhưng **hiện tại không có tác dụng thực tế**:
`runners/launch_remote.sh` (script khởi chạy container client) chưa forward
biến `AIPERF_DOCKER_IMAGE` vào bên trong container, nên `install_agentic_deps`
luôn thấy biến này unset và đi theo nhánh pip-install như cũ. Vì vậy:

- Có thể khai báo `aiperf-docker-image` trong config mà **không gây lỗi hay
  ảnh hưởng gì** đến job hiện tại (an toàn, nhưng vô tác dụng).
- Cho tới khi tính năng này được nối dây đầy đủ (xem phần "Future work" trong
  `docs/REMOTE_AIPERF_DOCKER.md`), mọi job remote đều sẽ tự cài AIPerf qua pip
  trên mỗi lần chạy như bình thường.

## 8. Checklist nhanh

1. Xác nhận model đã được serve và endpoint (`url`) đang reachable từ runner
   `benchmark-client`.
2. Thêm entry mới vào master config với `remote:` block + `scenarios.agentic-replay`
   duy nhất, `runner: benchmark-client`.
3. Validate YAML + chạy thử `generate_sweep_configs.py test-config` local.
4. Dispatch qua `gh api ... e2e-tests.yml/dispatches` hoặc Actions UI, dùng
   nhánh của bạn ở `inputs[ref]`.
5. Theo dõi run bằng `gh run view`, kiểm tra log nếu fail.

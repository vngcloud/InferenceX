# Bug report: `CUDA illegal memory access` với FROZEN_KV_MTP + FlashAttention trên Gemma-4

## 1. Tóm tắt

- **Triệu chứng:** khi chạy Gemma-4 31B với MTP (draft `Gemma4AssistantForCausalLM`) trên backend FlashAttention (FA4, từ PR sgl-project/sglang#42019), server chết giữa chừng với `CUDA illegal memory access`.
- **Bản chất:** đây là bug **có sẵn** trong `FlashAttentionBackend` (không do kernel FA4 gây ra). Nó chưa từng lộ ra vì trước đây không có đường nào chạy Gemma-4 trên FA: `model_hook.py` chặn backend này, và head_dim 512 khiến FA3 không chạy được. PR FA4 (#42019) gỡ chặn đó, nên bug mới bị dính.
- **Fix:** giải quyết pool SWA từ allocator cho draft FROZEN_KV_MTP, đúng theo cách flashinfer, trtllm_mha và aiter đã làm. Commit `8c1384a796`.
- **Kết quả:** 8 cell MTP trước đó chết nay chạy hết; cả 16 cell pass.

## 2. Môi trường

| | |
|---|---|
| Model | `RedHatAI/gemma-4-31B-it-FP8-block`, draft `google/gemma-4-31B-it-assistant` |
| Engine | SGLang v0.5.21 + PR aux #39286 + PR FA4 #42019 (image `2d74309`) |
| Workload | SPEED-Bench qua aiperf, ISL 8192 / OSL 1024, MTP depth 4 |
| Phần cứng | H200, runner `hardware-hcm-8x_02` |

## 3. Run làm lộ bug

- **Run `37298154251`** (e2e-tests.yml, `vngcloud/InferenceX`), image `2d74309`.
- Cả **8 cell MTP** (high/low entropy × conc 1/8/32/64) chết giữa chừng với `CUDA illegal memory access`.
- 8 cell **base** cùng run chạy bình thường, nên lỗi gắn với đường MTP, không phải FA4 nói chung.
- FA4 bắt buộc phải giữ (cho head_dim 512), nên không thể đổi backend để né.

## 4. Nguyên nhân

### 4.1 Cơ chế

1. Khi bật MTP, SGLang tự nâng NEXTN/EAGLE thành `FROZEN_KV_MTP`: draft Gemma-4 không có KV riêng mà **đọc thẳng KV của target**.
2. Gemma-4 là hybrid SWA: 10 lớp full attention, 50 lớp sliding window. Pool KV của target là `SWAKVPool`, gồm một pool full và một pool SWA nhỏ hơn. Page table chứa slot theo **đánh số full-pool**, và backend phải dịch sang slot SWA bằng `translate_loc_from_full_to_swa`.
3. `FlashAttentionBackend.__init__` suy ra `use_sliding_window_kv_pool` từ `model_runner.token_to_kv_pool` **của chính nó**:
   ```python
   isinstance(model_runner.token_to_kv_pool, SWAKVPool)
   ```
4. Với draft worker, pool riêng là `MHATokenToKVPool` thường, nên điều kiện là `False`. Backend bỏ qua bước dịch full→SWA.
5. Index full-pool bị dùng để đọc buffer SWA nhỏ hơn → truy cập ngoài biên → `CUDA illegal memory access`.

### 4.2 Vì sao nói đây là bug có sẵn, không phải do kernel FA4

- Logic dịch slot nằm ở tầng backend (`flashattention_backend.py`), không nằm trong kernel. Đổi kernel FA3/FA4 không ảnh hưởng.
- Các backend khác trên upstream main **đã sửa đúng điểm này** bằng `_resolve_swa_kv_pool(model_runner)`: flashinfer, trtllm_mha, aiter. Chỉ `flashattention_backend` còn dùng `isinstance(token_to_kv_pool, SWAKVPool)`.
- Bug chưa dính vì Gemma-4 chưa chạy được trên FA:
  - `python/sglang/srt/arg_groups/model_hook.py` (~634–657) chỉ cho phép trtllm_mha, triton, ascend, intel_xpu, intel_amx, aiter với Gemma-4. FA3/FA4 bị từ chối.
  - Mọi biến thể Gemma-4 có `global_head_dim = 512`, FA3 không hỗ trợ.
  - PR #42019 thêm `fa4` vào danh sách và hỗ trợ head_dim 512 trên SM90. Đây là lúc đường code này lần đầu có thể đi tới.
- Vậy bug **tiềm ẩn trên main, không tái hiện end-to-end được trên main**; chỉ tới được qua #42019.

### 4.3 Khoảng trống test

PR #42019 không có test bao phủ tổ hợp này: test e2e B200 dùng Qwen3-30B EAGLE3; `test_fa4.py` có `runner_fa4_frozen_kv_mtp_verify_chain` nhưng dùng pool phẳng (không có `sliding_window_size`). Tổ hợp SWA hybrid pool + draft FROZEN_KV là điểm chưa được kiểm.

## 5. Cách fix

**Nhánh chính:** [`Noridom1/sglang@gemma4-v0.5.21-fa4-swapool`](https://github.com/Noridom1/sglang/tree/gemma4-v0.5.21-fa4-swapool) (`3a7d5ad`): release v0.5.21 + FA4 #42019 + fix, **không có aux**. Đây là bản sạch nhất để tham chiếu vì chỉ gồm FA4 và fix, khớp với phạm vi của bug và của PR upstream.

Nhánh phụ: `gemma4-v0.5.21-aux-fa4-swapool` (commit fix `8c1384a796`) là cùng nội dung cộng PR aux #39286. Đây là stack đã tạo ra run lỗi gốc nên giữ để đối chiếu. Aux (sửa hidden-state capture) đã được audit là không liên quan tới bug này (xem mục 6).

**File:** `python/sglang/srt/layers/attention/flashattention_backend.py`

Thêm hàm giải quyết pool, gọi một lần trong `__init__`:

```python
@staticmethod
def _resolve_swa_kv_pool(model_runner):
    active_pool = model_runner.token_to_kv_pool
    if isinstance(active_pool, SWAKVPool):
        return active_pool
    if model_runner.is_draft_worker:
        if not model_runner.spec_algorithm.is_frozen_kv_mtp():
            return None          # draft khác: giữ hành vi cũ
    kvcache = model_runner.token_to_kv_pool_allocator.get_kvcache()
    return kvcache if isinstance(kvcache, SWAKVPool) else None
```

```python
self._swa_kv_pool = self._resolve_swa_kv_pool(model_runner)
self.use_sliding_window_kv_pool = (
    self._swa_kv_pool is not None and self._swa_kv_pool.swa_layer_nums > 0
)
```

Toàn bộ chỗ dịch slot (`translate_loc_from_full_to_swa`, `full_to_swa_index_mapping`) dùng `self._swa_kv_pool` thay vì `token_to_kv_pool` của backend. Chỉ draft FROZEN_KV_MTP đổi hành vi; mọi trường hợp khác giữ nguyên.

**Test:** thêm `flashattention` vào `_RESOLVERS` trong `test/registered/unit/spec/test_resolve_swa_kv_pool.py` (kiểm resolver bằng mock, chạy được không cần GPU).

## 6. Kiểm chứng

| Kiểm tra | Kết quả |
|---|---|
| Sweep gốc, image `2d74309` (chưa fix), run `37298154251` | 8 cell MTP chết `illegal memory access` |
| 2 job 1 GPU H200 (cuda graph và eager) sau fix | 0 lỗi illegal memory, accept length ~3.0–3.4 |
| Sweep 8 cell MTP sau fix trên nhánh chính (image B, `3a7d5ad`, không aux) | chạy hết ma trận, không còn crash (cùng 8 cell base → 16 cell pass) |
| Image A (có aux, `...-aux-fa4-swapool-8c1384a`) vs B (không aux, `3a7d5ad`) | chênh < 1.5% → fix đủ, aux không ảnh hưởng hiệu năng |

Lưu ý trung thực: crash gốc chỉ được quan sát trên stack có aux (`2d74309`). Fix được kiểm trên cả hai stack, và cơ chế (dịch slot full→SWA trong backend) không liên quan tới aux, nhưng chưa có lần chạy FA4 không-aux, không-fix để xác nhận crash tái hiện đúng y như vậy.

Số liệu hiệu năng MTP đầy đủ nằm ở `docs/gemma4-optimization-diary.md`.

## 7. Còn lại / lưu ý

- Unit test chưa chạy trên nhánh dựa trên upstream `main` (máy local không có torch, mới kiểm cú pháp và ruff/isort/codespell).
- Đề xuất bổ sung case `runner_fa4_frozen_kv_mtp_verify_chain_swa` trong `test_fa4.py` (pool SWA + draft frozen-KV) để vá khoảng trống test ở mục 4.3.
- PR upstream: nên mở dạng draft, mô tả là lỗi đồng bộ giữa các backend và nêu rõ "exposed by #42019"; hoặc để đồng nghiệp gộp commit `8c1384a796` vào #42019.

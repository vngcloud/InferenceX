# GLM-5.2 DSpark 智能体基准测试报告

[English](GLM52_DSPARK_AGENTIC_REPORT.md) | [中文](GLM52_DSPARK_AGENTIC_REPORT_zh.md)

状态：已验证的 H200 智能体推理配置的基准证据与服务指导。

## 结论摘要

对于 CCU24 智能体工作负载，在以整体服务能力和尾延迟为目标时，已验证的 DSpark block-7 配置是当前测得最强的配置。与拓扑完全相同的 DSpark block-4 运行相比，它实现了：

- 输出吞吐量提高 4.58%，总吞吐量提高 6.28%；
- 成功完成的 profiling 请求增加 5.54%；
- TTFT p90 降低 50.15%，TTFT p99 降低 44.75%；
- 端到端延迟 p99 降低 12.46%；
- 每次投机步骤平均接受 3.38 个 token，p90 为 4.47 个 token。

不能孤立地解读 34.14% 的 accept rate。block 7 每步约生成 7.96 个 draft token，而 block 4 为 4.97 个；block 7 的实际 accept length 更高，只是 accept-rate 的分母更大。代价是 draft token 数增加 60.18%，每活跃用户输出吞吐量降低 5.03%。对于 CCU24 容量目标，建议保留 block 7；若严格优化单会话解码效率，block 4 仍是有价值的回退配置。

## 1. 范围与来源

本文记录生产路径实验，而不是合成微基准。三个运行均使用 GLM-5.2 智能体回放工作负载和同一 H200 GreenNode 系列。

| 项目 | 值 |
|---|---|
| 目标模型 | `zai-org/GLM-5.2-FP8`，服务路径 `/models/PhalaCloud/GLM-5.2-W4AFP8` |
| 目标量化 | W4AFP8（`--quantization w4afp8`） |
| DSpark 草稿模型 | `/models/RedHatAI/GLM-5.2-speculator.dspark` |
| 草稿量化 | 显式设置为 `unquant`（BF16 checkpoint，不继承目标模型量化） |
| 服务镜像 | `thangquang0909/sglang:v0.5.16-dspark-v2-g4` |
| Recipe commit | `310fca9287ce363146cd7718fbf739a8eab63c01` |
| 硬件 | H200 GreenNode，8 张 GPU |
| 工作负载 | SemiAnalysis Weka 256k 智能体编码回放 |
| 并发 | CCU24 |
| 时长 | 3,600 秒 profiling 窗口，另含 warmup/teardown |

运行链接：

- [EAGLE 参考运行，30163138459](https://github.com/vngcloud/InferenceX/actions/runs/30163138459)
- [DSpark block 4，30690287203](https://github.com/vngcloud/InferenceX/actions/runs/30690287203)
- [DSpark block 7，30693457521](https://github.com/vngcloud/InferenceX/actions/runs/30693457521)

原始基准测试行由 AIPerf metrics extractor 统一提取。仓库提交 recipe 和测试；大型原始基准产物保存在基准测试工作区中，通过 run ID 引用，不直接提交到 Git。

### Checkpoint 来源与一个已知的注意事项

已在服务主机（`h200-greennode_01`）上直接核实草稿 checkpoint：挂载的 `model.safetensors` 指向 `RedHatAI/GLM-5.2-speculator.dspark` 的 snapshot `8bc9ac46fbf507f3ee3ad82304116a1f63e9edb4`，与 recipe A/B 测试中选定的 "current" checkpoint 一致。该主机上不存在任何其他草稿 checkpoint（此前探索过、后已放弃的 `robertgshaw2-afk/GLM-5.2-DSpark`）。

Red Hat AI 官方 model card 将该 checkpoint 描述为"初步版本（可能变更）"，且仅在 B200 硬件上验证过。其 config 中 `speculators_config.verifier.name_or_path` 为 `RedHatAI/GLM-5.2-NVFP4-FP8`，并非本基准测试实际服务的目标模型 `zai-org/GLM-5.2-FP8`——草稿模型训练时对齐的目标量化与此处实际验证的目标不同。这一 mismatch 尚未被隔离验证或排除为第 4 节 accept-length 数据的影响因素；本 recipe 开发过程中记录的 accept-length 提升来自修复两个集成 bug（草稿 anchor-window 选择、辅助 hidden-state layer-id 偏移),而非确认了目标一致性。下文的 block-4/block-7 对比在内部仍然有效（同一 checkpoint、同一 mismatch，两行均适用），但不应将其中的 accept-length 绝对值解读为该 checkpoint 在目标一致情况下的真实上限。

## 2. 基准测试前完成的修复

DSpark 生产路径经过以下调试过程：

1. 初始路径让草稿 checkpoint 继承目标 W4AFP8 量化，导致草稿 proposal 没有有效接受。
2. launcher 增加 `--speculative-draft-model-quantization unquant`，将目标模型和草稿模型的量化配置解耦。
3. 拓扑与参考运行对齐：TP8/DP8、DP Attention、启用 DP LM head，以及 cache-aware SGLang router。
4. `mem-fraction-static` 从 0.85 调整为 0.75，为长智能体 prefill 和 DSpark verification graph 保留 headroom。
5. launcher 请求 32,768 token 的 chunked prefill size；在 DP Attention 下，SGLang 会将 effective runtime value 调整为 4,096，并记录 `max_prefill_tokens=16384`，该调整可在 server log 中看到。
6. recipe 测试现在检查生成后的真实命令，包括目标 `w4afp8`、草稿 `unquant`、TP8/DP8/DPA 和选定的 DSpark block size。

这些变更只作用于受影响的 launcher 与回归测试，不修改 SGLang 的通用量化继承逻辑。

## 3. 服务配置

生产配置的关键参数如下：

```text
--model-path /models/PhalaCloud/GLM-5.2-W4AFP8
--quantization w4afp8
--tp 8 --dp 8 --enable-dp-attention --enable-dp-lm-head
--chunked-prefill-size 32768
--mem-fraction-static 0.75
--kv-cache-dtype fp8_e4m3
--enable-hierarchical-cache --hicache-ratio 2
--speculative-algorithm DSPARK
--speculative-draft-model-path /models/RedHatAI/GLM-5.2-speculator.dspark
--speculative-draft-model-quantization unquant
--speculative-dspark-block-size 7
--schedule-policy lpm
```

benchmark 通过 public port 访问 router，通过相邻的 SGLang backend port 采集 metrics。router 使用 cache-aware 和 DP-aware 策略。generated server command 确认了请求的 launch flags；启动日志确认 effective TP8/DP8/DPA topology、cache 配置和 DSpark initialization，并记录了 DP Attention 将请求的 32,768 chunked prefill 调整为 effective 4,096。

### Block size 术语

在该 recipe 中，`--speculative-dspark-block-size 7` 选择 DSpark gamma 7。server arguments 显示 `speculative_num_draft_tokens=8`，而 draft checkpoint 声明 native block size 为 8。因此 SGLang 会记录 gamma/configuration mismatch 警告（`gamma=7`、draft `block_size=8`），同时仍使用请求的 gamma-7 服务路径。这是该配置的预期行为，并不表示 DSpark 被禁用。

block-7 运行实际测得每步平均 draft token 数为 7.957。accept-length metrics 表示每个投机步骤接受的 token 数，应理解为有效 token 数，而不是百分比。

## 4. 结果

以下均为 CCU24。block 7 的 `ok_with_errors` 来自一次 warmup 阶段空内容响应；profiling 窗口本身没有错误。

| 指标 | EAGLE | DSpark block 4 | DSpark block 7 |
|---|---:|---:|---:|
| 成功 profiling 请求 | 2,745 | 2,746 | **2,898** |
| Input TPS | 81,574.88 | 82,506.49 | **87,700.56** |
| Output TPS | 629.49 | 647.48 | **677.11** |
| Total TPS | 82,204.38 | 83,153.96 | **88,377.66** |
| Output TPS/user | **27.50** | 28.92 | 27.46 |
| 请求 QPS | 0.7541 | 0.7544 | **0.7962** |
| TTFT p50 | **0.908 s** | 1.102 s | 1.023 s |
| TTFT p90 | **3.422 s** | 12.106 s | 6.035 s |
| TTFT p99 | **17.346 s** | 83.871 s | 46.340 s |
| E2E p90 | 83.527 s | 94.704 s | 88.530 s |
| E2E p99 | 294.259 s | 282.372 s | **247.181 s** |
| ITL p90 | 68.089 ms | 69.085 ms | **67.818 ms** |
| Queue time p90 | **3.841 s** | 26.822 s | 21.992 s |
| Queue time p99 | **29.183 s** | 154.686 s | 87.453 s |
| Token usage p90 | 0.73 | 0.97 | 0.97 |
| Response cache hit | 96.364% | 96.246% | **96.398%** |
| Draft token/step | 3.976 | 4.968 | **7.957** |
| Accept rate avg | **52.922%** | 51.105% | 34.140% |
| Accept rate p90 | **87.083%** | 65.789% | 49.580% |
| Accept length avg | 2.582 | 3.038 | **3.384** |
| Accept length p90 | 3.613 | 3.632 | **4.471** |
| Prefill forward p90 | 2.939 s | 2.160 s | **1.977 s** |
| GPU utilization avg | 97.024% | 95.171% | 95.030% |
| GPU power avg | 420.14 W | 424.02 W | **417.52 W** |
| GPU memory avg | 145.23 GB | 146.36 GB | **143.84 GB** |
| GPU memory max | 145.78 GB | 149.75 GB | 150.04 GB |

### Block 7 与 block 4 的比较

这是最干净的对比：topology、image、target、draft、workload、CCU 和 memory setting 均保持不变，只改变 block size。

- Output TPS: **+4.58%**
- Total TPS: **+6.28%**
- 成功请求数：**+5.54%**
- TTFT p90/p99: **-50.15% / -44.75%**
- E2E p99: **-12.46%**
- Accept length 平均值/p90：**+11.41% / +23.10%**
- 每步 draft token：**+60.18%**
- Output TPS/user: **-5.03%**

结果表明，block 7 更适合服务 CCU24 的 closed-loop 请求群体，但不代表每条独立 decode stream 的成本更低。它完成了更多智能体工作并降低 queue/TTFT 尾延迟，代价是更长的 draft verification。

### Block 7 与 EAGLE 的比较

Block 7 的 output TPS 比 EAGLE 高 7.56%，total TPS 高 7.51%，而 output TPS/user 几乎相同（-0.15%）。它的 E2E p99 和 ITL 也更低；EAGLE 在 TTFT 和 queue time 上仍有明显优势。

该对比很有价值，但 memory budget 并不完全相同：EAGLE 使用 `mem-fraction-static=0.85`，DSpark block 4 和 block 7 使用 0.75 以保留 prefill/speculation headroom。因此不应将 latency 差异解释为纯粹的算法差异。

## 5. 如何理解 DSpark acceptance

Acceptance 有三个相关但不同的含义：

1. **Accept rate** 是 target model 接受的 draft token 比例。block 越长，分母越大，因此 rate 通常会下降。
2. **Accept length** 是每个 speculative step 产生的有效 output 数量，更直接地衡量一次投机步骤节省了多少 target-model token。
3. **Accept length p90** 描述稳定分布的上部。block 7 的 p90 为 4.47，block 4 为 3.63，EAGLE 为 3.61。

因此，block 7 的 34.14% accept rate 仍然与良好的 speculative 能力相符：每步提出接近 8 个 token，平均接受 3.38 个 token。只比较百分比会低估长 block，因为它在每次 verification 中完成了更多有用工作。

八个 DP rank 的直接 server metrics 显示，block 7 的 accept-length p90 约为 3.98 至 4.47，没有持续的零 acceptance 平台。直接 profile 包含 2,898 条成功记录；唯一错误来自 warmup 阶段没有实际 content 的 response。

## 6. 安全性与正确性检查

Run 30693457521 通过了 workflow 和 artifact 检查：

- GitHub Actions 页面记录 conclusion `success`。
- Recipe commit: `310fca9287ce363146cd7718fbf739a8eab63c01`.
- CSV extraction：87 条唯一 row，没有 identity duplicate。
- Block-7 profiling：2,898 次成功、0 次 profiling error；run 本身未被 cancel，但 timeout cleanup 阶段取消了 7 个正在运行的 request。
- Server metrics 和 DCGM telemetry：齐全。
- 没有 server OOM、traceback、严重 CUDA/NCCL 错误或非零 GPU XID。
- AIPerf aggregate 与标准化 CSV 的 output TPS 相差 0.304%。

warmup 阶段唯一的 `InvalidInferenceResultError` 表示一个 response 只有 usage/metadata，没有 content。该 record 不在 profiling success set 中，也不表示 server crash。

## 7. 复现 checklist

运行已提交的 recipe 和 test：

```bash
python -m pytest utils/test_glm52_dspark_recipe.py -v
```

dispatch 长时间 run 前检查：

1. Target image/model 为 W4AFP8，draft 包含 `config.json` 和 `model.safetensors`。
2. Generated command 分别包含 target `w4afp8` 和 draft `unquant`。
3. Topology 为 TP8/DP8，并包含 `--enable-dp-attention` 和 `--enable-dp-lm-head`。
4. HiCache ratio、FP8 KV cache、memory fraction 和 chunked prefill 符合 capacity budget；注意 DP Attention 下 effective chunked prefill 为 4096。
5. 在发送 traffic 前，server log 已记录 DSpark initialization。
6. 从 `sglang:spec_accept_rate` 和 `sglang:spec_accept_length` 读取 acceptance，不要只从 throughput 推断。
7. Artifact 包含 AIPerf profile export、server metrics、GPU telemetry 和准确的 startup command。

## 8. 建议与后续工作

对于该 image 和 workload，保留 block 7 作为 CCU24 production candidate。保留 block 4 作为 draft cost 更低的 fallback 和后续变更的 control。promote 最终 image 时必须保留 `unquant` draft override 以及 DP LM head/topology assertions。

下一项有价值的实验是在相同 memory budget 下，于 CCU8、CCU16、CCU24 和 CCU32 重复 block-7/block-4 对比。一次长时间 run 已建立当前结果，但仍需要重复 ladder 来量化低负载和高负载下的 variance。

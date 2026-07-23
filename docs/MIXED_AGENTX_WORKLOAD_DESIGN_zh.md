<div align="center">

[English](./MIXED_AGENTX_WORKLOAD_DESIGN.md) | **中文**

</div>

# AgentX 混合工作负载设计

状态：待评审。

## 目标

构建一套内部私有、可复现的 AgentX 数据集，以可配置比例混合编程、短多轮对话和
RAG 流量，测量推理引擎的服务容量。首个参考配置面向
`zai-org/GLM-5.2-FP8`。

该工作负载测量真实提示词结构自然产生的缓存行为，不会为了匹配生产环境约 70%
的观测值而强行调整整体缓存命中率。

## 范围

基准测试复用现有 `inferencex-agentx-mvp` 场景，包括预热、会话树并发、会话回收、
首轮缓存隔离，以及编程 trace 的 10 秒空闲间隔上限。

这是引擎级基准测试。当前启动器直接向
`http://localhost:$PORT/v1/chat/completions` 发送请求，不复现生产网关的负载均衡
或跨副本缓存局部性。

AgentX 会话是客户端侧的调度和提示词构造单元。SGLang 接收携带累积消息的无状态
HTTP 请求。Correlation ID 仅用于可观测性，不建立路由亲和性。

## 固定输入

| 输入 | Revision | 用途 |
|---|---|---|
| `semianalysisai/cc-traces-weka-062126` | `23f152f6f0f9399a85901b89a6458def0ef16729` | 编程会话 |
| `allenai/WildChat-1M` | `7d6490e462285cf85d91eabea0f9a954fbddcd1f` | 短多轮对话 |
| `nvidia/ChatRAG-Bench` | `22ece8bb870ddcf3f7aacfd5b6b0446d112a1e92` | 多轮 RAG |
| `zai-org/GLM-5.2-FP8` | `70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1` | 参考 tokenizer 和对话模板 |

参考服务端上下文上限为 500,000 token，与现有 GLM-5.2 H200 SGLang 启动器保持一致。
每个请求必须满足：

```text
templated_input_tokens + requested_output_tokens <= 500000
```

任一请求超出限制时，丢弃整个会话，不进行截断。

## 架构

使用一个轻量混合 loader 和一个不可变的按模型 manifest。

- Weka 数据原样委托给现有 Weka loader。
- WildChat 和 ChatRAG 转换为现有 AgentX `Conversation`/turn 表示。
- manifest 保存有序逻辑会话实例、数据源 revision、源行 ID、tokenizer/模板哈希、
  权重、时间策略、上下文上限和构建随机种子。
- 跨推理框架和硬件比较时固定 manifest。修改权重会生成新的 manifest，不会在运行
  中修改工作负载。

不新增服务场景、路由抽象或数据集框架。

## 工作负载类别

### 编程

逐字节保留现有 Weka 重建结果，包括父代理/子代理树和当前时间语义。

编程数据按会话输入 token 十分位，以及是否包含子代理进行分层选择。

### 短多轮对话

使用完整且干净的 WildChat 会话，要求：

- 至少两条 assistant 回复；
- user/assistant 角色严格交替；
- 内容为非空字符串；
- 不包含 toxic 或 redacted 会话标记；
- assistant 完成时间单调递增；
- 所有请求均不超过目标上下文。

请求 `n` 发送截止到第 `n` 个 user turn 的累积历史。后续提示词使用数据集中记录的
assistant 消息构造；基准测试期间的实时回复只用于测量，随后丢弃。`max_tokens`
取被省略目标 assistant 回复经 GLM tokenizer 计算后的长度。

该类别有意保留会话内自然的前缀复用。HTTP 无状态不等于推理无缓存：后续累积提示词
仍可能命中同一推理引擎的前缀缓存。

WildChat 按轮数、语言和源模型进行分层选择。

### RAG

使用 ChatRAG 原始结构，不合成检索文档：

- 保留前五个 `ctxs`，与官方评估默认值一致；
- 将 RAG 指令和检索文档放入结构化 system 消息；
- 保留最近七条历史消息，与源评估的历史窗口一致；
- 保留各源数据集对应的回答指令；
- 使用 `answers[0]` 作为省略的目标回复；
- 将非空 `answers[0]` 经固定 GLM tokenizer 计算后的长度设为 `max_tokens`，
  不再增加输出上限；
- 不采用源评估器的 4,096-token 截断，只应用参考模型的 500,000-token 上限。

ChatRAG 提供的是累积快照，但没有 conversation ID。按以下规则重建父子关系：

```text
child.messages[:-2] == parent.messages
```

先对完全相同的快照去重。只有上述表达式恰好匹配一个剩余快照时才建立父子关系。
父快照缺失或匹配不唯一时，将该记录作为带部分历史的轨迹起点，其内嵌历史仍原样发送。
这可避免在消息历史相同、但检索上下文或元数据不同时进行任意拼接。

system/context 消息、各数据集回答指令、七消息窗口和 GLM 对话模板属于明确的目标模型
适配。应用这些变换前，保留一份源角色和内容的规范副本。

完成许可证审查后的内部默认 subset 池为：

- Doc2Dial；
- QuAC；
- QReCC；
- DoQA cooking、movies 和 travel；
- ConvFinQA。

| Subset | 来源条款 | 必须处理 |
|---|---|---|
| [Doc2Dial](https://huggingface.co/datasets/IBM/doc2dial) | CC BY 3.0 | 保留署名和许可证声明 |
| [QuAC](https://quac.ai/datasheet.pdf) | MIT；要求引用论文 | 保留声明和引用 |
| [QReCC](https://github.com/apple/ml-qrecc#license) | 数据集 CC BY-SA 3.0；检索网页仍受原来源权利约束 | 保留署名、相同方式共享声明和来源信息 |
| [DoQA](https://ixa.eus/node/12931) | CC BY-SA 4.0；派生自 Stack Exchange | 保留署名、相同方式共享声明和来源信息 |
| [ConvFinQA](https://github.com/czyssrs/ConvFinQA) | MIT | 保留版权和许可证声明 |

实际使用仍须遵循组织内部法律政策。TopiOCQA 因非商业条款默认排除。INSCIT、CoQA、
HybriDialogue 和 SQA 在数据集级复用条款或来源信息获批前排除。完整下载可继续用于
分析，但默认 manifest 不会选择这些 subset。

## 时间模型

### WildChat

用户思考时间估算为：

```text
(completion_time(A[i+1]) - completion_time(A[i]))
  - source_service_time_estimate(A[i+1])
```

`A[i+1]` 的源服务时间按其源模型和输出 token 分桶后取 p10；源模型 token 分桶使用
`cl100k_base`。该残差估算从 `A[i]` 完成到下一次 user 请求提交之间的延迟。负残差
归零后再应用上限。

```text
WILDCHAT_THINK_TIME_CAP_SECONDS=10
```

### ChatRAG

ChatRAG 不包含时间戳。使用 manifest 随机种子，从已经接受并完成 10 秒封顶的
WildChat 残差分布中确定性采样轮间延迟。manifest 明确记录
`timing_source: wildchat_proxy`，不得将其描述为源数据时间。获得生产 RAG OTel
trace 后，用真实时间替换该代理。

## 混合比例

权重是可配置、非负且总和为 1 的输入 token 占比。首个参考配置为：

```yaml
mix:
  coding: 0.70
  short_chat: 0.15
  rag: 0.15
```

以完整轨迹为单位进行确定性选择。生成后的估算输入 token 比例与目标之间的误差不得
超过 0.5 个百分点。由于长 RAG 提示词可能用较少请求即可达到 token 目标，还必须报告
请求占比和会话占比。

服务端观测 token 比例只用于诊断。与目标偏差超过 3 个百分点时告警，但绝不自动改写
manifest。

## 预期缓存行为

不添加额外的逐轮 salt。AgentX 首轮标记负责隔离逻辑会话；后续复用来自真实重复前缀。

使用 GLM-5.2 tokenizer 分析 ChatRAG，可同时观察到两类预期的 RAG 模式：

| Subset | 输入 token 中位数 | 无限缓存条件下的公共前缀上界 |
|---|---:|---:|
| ConvFinQA | 943 | 70.25% |
| DoQA 各变体 | 294-301 | 68.12-68.91% |
| Doc2Dial | 1,855 | 15.03% |
| QReCC | 2,679 | 14.29% |
| QuAC | 2,526 | 20.37% |
| INSCIT（默认排除） | 754 | 9.49% |
| TopiOCQA（默认排除） | 781 | 10.60% |

每个 subset 的最后一列按以下方式计算：

```text
sum(LCP(parent_prompt, child_prompt)，仅统计父快照唯一的子请求)
----------------------------------------------------------------
sum(所有请求的 input_tokens)
```

提示词采用前五个 context、最近七条消息、上述适配和固定 GLM 对话模板。父快照缺失或
匹配不唯一时不计入 LCP 分子。离线统计不包含 AgentX 首轮标记、缓存块取整、淘汰和
容量限制。因此，该值是可复现的父请求公共前缀上界，而不是服务端缓存命中率预测。
检索文档变化会在提示词靠前位置破坏前缀；文档集合稳定时则保留较多复用。

## 指标

整体和各类别分别报告：

- 请求数、会话数、输入/输出 token 和 TPM；
- TTFT、TPOT、token 间延迟和端到端延迟；
- 目标、估算和服务端观测的输入 token 比例；
- 预期前缀复用。

整体缓存命中率使用预热排空后、仅覆盖 profiling 窗口的服务端 Prometheus counter
增量：

```text
overall_cache_hit_rate =
  delta(cache_read_input_tokens) / delta(prompt_input_tokens)
```

对于 SGLang，对 `sglang:cached_tokens` 的各 `cache_source` 标签计算可处理 counter
重置的增量并求和，再除以对应的 `sglang:prompt_tokens` 增量。每个去重后的推理 worker
只累加一次，不得将同一 counter 的 router/frontend 副本重复计入。profiling 开始前
立即采集起始快照，在全部 profiling 请求排空后采集结束快照，排除 warmup 和 cooldown。
其他后端使用等价的 token counter；如果不存在，则将 token 缓存命中率报告为不可用，
不得用全程 gauge 替代。

只有端点返回逐请求 cached-token 数量时，才报告各类别的观测缓存命中率。不得根据
全局计数器推导分类缓存命中率。

## 校验

出现数据源 revision 不匹配、tokenizer/模板哈希不匹配、角色非法、时间数据异常、
源行缺失、上下文溢出或比例误差超限时，构建必须失败。

最低验证要求：

- 过滤、时间估算和完整会话选择具备确定性；
- Weka 重建结果逐字节一致；
- 适配前的规范源角色/内容完全一致，并对每一种允许的 context、指令、窗口和模板变换
  进行精确测试；
- 可解析的 ChatRAG 快照父子关系精确重建；
- 使用固定 GLM tokenizer 校验上下文和输出 token；
- 不保留无关隐私或 moderation 字段；
- 完成 AgentX 集成 replay；
- 总体统计等于各类别统计之和。

## 交付

原始数据和生成的私有产物保存在本地 `datasets/` 目录，不提交到仓库。仅对 loader
代码、测试、manifest schema 和文档进行版本管理。

首版实现仅覆盖三类引擎级基准测试。网关路由模拟、自动拟合生产比例、实际执行检索和
公开发布数据集均不在范围内。

# Serving-YAML 基准测试流程

<div align="center">

[English](./serving-yaml-procedures.md) | **中文**

</div>

使用本页在 VNG GPU Runner 上对你自己的服务配置（`vllm serve` / `sglang` 命令，或 Dynamo frontend + `dynamo.sglang` 组合）进行基准测试，无需编写 recipe 或修改 `configs/`。输入一个 YAML 文件，产出一次 GitHub Actions 运行。[`infx/serving/schema.py`](../infx/serving/schema.py) 中的 Schema 与 Agent 技能 [`.agents/skills/serving-bench/SKILL.md`](../.agents/skills/serving-bench/SKILL.md) 为权威来源；本页说明谁可以运行，以及基础设施提供了什么。

## 流程索引

| 流程 | 适用场景 |
| --- | --- |
| [检查前置条件](#检查前置条件) | 首次运行 serving YAML |
| [了解 Runner 提供的资源](#了解-runner-提供的资源) | 判断你的镜像和模型能否使用 |
| [编写并派发 serving YAML](#编写并派发-serving-yaml) | 对某个配置进行基准测试 |
| [Sweep 场景](#sweep-场景) | 在合成 ISL/OSL/缓存网格上进行基准测试 |
| [查看结果与失败](#查看结果与失败) | 运行结束或失败 |
| [升级给基础设施负责人](#升级给基础设施负责人) | 阻塞点在节点上，而不在你的 YAML 中 |

## 检查前置条件

| 要求 | 检查方式 |
| --- | --- |
| 对 `vngcloud/InferenceX` 有写权限 | `gh repo view vngcloud/InferenceX` 成功，且可以推送分支 |
| `gh` 已认证 | `gh auth status` |
| 已安装 `uv` | `uv --version`（派发脚本通过 `uv run` 校验 YAML） |
| 分支已推送，且本地 `HEAD` 等于 `origin/<branch>` | 否则 `utils/serving/dispatch.sh` 会拒绝派发，因为 CI 运行的是已推送的代码 |

你不需要 Slurm 账号、GPU 节点的 SSH 权限或镜像仓库凭据。无论由谁派发，每个任务都以 Runner 的服务用户（`stackops`）身份在节点上运行。

## 了解 Runner 提供的资源

| 资源 | 提供方 | 对你的影响 |
| --- | --- | --- |
| 容器镜像拉取 | Runner 用户的 enroot 凭据（`stackops` 的 `~/.config/enroot/.credentials`） | 公共镜像可用。`vcr.vngcloud.vn` 镜像仅在 Runner 上配置的 VCR 账号可读取该仓库时可用。其他私有仓库会在 `enroot import` 时失败；见[升级给基础设施负责人](#升级给基础设施负责人)。 |
| 模型权重 | `serve_generic.sh` 在启动引擎前执行 `hf download <model>` | 模型无需预先放在节点上。若已在节点的 HF 缓存中，下载不做任何事；否则会自动拉取。`model:` 必须是 HF 仓库 ID，本地路径会被拒绝。 |
| HF 缓存位置 | 由各 Runner 池的启动脚本决定（见下表） | 不能在 `env:` 中设置 `HF_HUB_CACHE`（保留变量）。 |
| 受限或私有模型 | 来自仓库 Secret `INFERENCEX_OFFICIAL_RO_HF_TOKEN` 的 `HF_TOKEN` | 该 Token 对应的 HF 账号必须有权访问该模型（已接受许可）。 |
| 锁定版本的场景数据集（tau2） | Runner `.env` 中的 `GREENLAB_HF_TOKEN` | 无需设置；切勿在 YAML 中写入 Token。 |
| Dynamo 服务发现、端口、指标接线 | `runners/launch_hardware-hcm.sh` | 省略 `DYN_*`、`ETCD_ENDPOINTS` 与 NATS 设置。 |

| Runner 池 | `runner:` 取值 | GPU 数 | 主机 HF 缓存 | Router（`router:`） |
| --- | --- | --- | --- | --- |
| `cluster:h200-greennode-slurm` | `h200-greennode-slurm-1x_*` | 1 | `/data/inferencex-<uid>/hf`，或 Runner `.env` 中的 `IX_HF_CACHE_HOST` | 否 |
| `cluster:hardware-hcm` | `hardware-hcm-8x_01`、`hardware-hcm-8x_02` | 8 | `/mnt/hf_hub_cache` | 是 |

大模型首次运行需要额外的下载时间，并占用节点磁盘（例如 GLM-5.3 W4AFP8 约 400 GB）。请据此安排首次运行。

## 编写并派发 serving YAML

1. 创建 `serving/<name>.yaml`。字段说明与转换规则（模型路径 → HF ID、环境变量、Router 拆分、上下文长度）见 [`.agents/skills/serving-bench/SKILL.md`](../.agents/skills/serving-bench/SKILL.md)。可用示例：[`serving/qwen38-bf16-tp1.yaml`](../serving/qwen38-bf16-tp1.yaml)（仅引擎）、[`serving/glm53-w4afp8-dynamo.yaml`](../serving/glm53-w4afp8-dynamo.yaml)（Dynamo Router + 引擎）与 [`serving/glm53-w4afp8-sglang-router.yaml`](../serving/glm53-w4afp8-sglang-router.yaml)（sglang-router + SGLang 引擎）。
2. 从 `scenarios/` 中选择场景（`agentx-cc-062126`、`agentx-cc-062126-smoke`、`custom-agentic-tau2`、`custom-agentic-tau2-smoke`）。新配置先用 `-smoke` 场景。生成的 `sweep-isl<isl>-osl<osl>-cache<cache>.yaml` 系列会自适应地搜索并发，且始终以 `conc: 1` 运行——见 [Sweep 场景](#sweep-场景)。
3. 校验并派发：

   ```bash
   utils/serving/dispatch.sh serving/<name>.yaml
   ```

   脚本执行与 CI 相同的校验。出现 `ERROR <field>: <fix>` 时不会派发任何任务：修改 YAML 后重新运行。成功时会输出运行 URL。使用 `--ref <branch>` 可派发当前分支以外的分支。
4. 如果脚本提示运行未出现，不要重复派发；先执行 `gh run list --repo vngcloud/InferenceX -w e2e-tests.yml -L 5` 检查。

### 副本（replicas）

「**副本（replicas）**」：同一引擎的多个副本（如 2×TP1）。设置 `replicas: N`，`gpus` = N × 每个引擎的 GPU 数；路由命令用 `{{engine_urls}}`（空格分隔的 URL）列出全部引擎。不支持 Dynamo frontend。

```yaml
gpus: 2
replicas: 2
router:
  image: vllm/vllm-openai:v0.25.1
  metrics_port: 29000
  serve: >-
    bash -c "python3 -m pip install -q vllm-router==0.1.14
    && exec vllm-router --host 0.0.0.0 --port {{router_port}}
    --worker-urls {{engine_urls}} --policy cache_aware
    --prometheus-host 127.0.0.1 --prometheus-port 29000
    --request-timeout-secs 14400 --disable-retries"
```

### 负载下扩容（scale_up）

`scale_up:` 让最后 N 个副本不进入 `{{engine_urls}}`。它们与其他副本一起加载并通过健康检查。aiperf profiling 阶段开始 `after` 秒后，通过路由的 `POST /add_worker` 加入；vllm-router 与 sglang-router 有该接口，Dynamo frontend 没有。结果中的 `scale_up.jsonl` 记录每个副本的加入时间。

```yaml
replicas: 2
scale_up:
  replicas: 1   # 晚加入的副本数
  after: 600    # profiling 开始后的秒数；须在场景时长之内
```

用 `python3 utils/serving/scale_up_ttft.py <agentic_ 产物目录> [--slo-ms 1000]` 读取结果：按加入前后的时间窗输出 TTFT，并根据 consistent_hash 的 `router.log` 区分留下与迁移的请求。示例：`serving/gemma4-livekit-2tp4-chash-mooncake-scaleup.yaml`。

vLLM 的 `MooncakeStoreConnector` 从文件读取存储配置。改为在 `env: IX_MOONCAKE_STORE_CONFIG` 中给出 JSON（可用 `{{storage_master_port}}`）。recipe 将其写入 `mooncake_store_config.json` 并设置 `MOONCAKE_CONFIG_PATH`。

## Sweep 场景

`scenarios/sweep-*.yaml` 系列（由 [`utils/serving/gen_sweep_scenarios.py`](../utils/serving/gen_sweep_scenarios.py) 生成，请勿手改）在合成的 ISL/OSL/缓存命中网格上对同一个 serving 配置进行基准测试：ISL {8k, 32k, 64k, 128k} × OSL {512, 1k} × 前缀缓存命中 {0, 50, 90}% —— 共 24 个场景，命名为 `sweep-isl<isl>-osl<osl>-cache<cache>`。每个场景带有 AIPerf `adaptive_search` sweep：没有固定的 `conc` 值，AIPerf 在任务内探测并发区间并搜索满足 SLO 的最高并发，因此 serving YAML 以 `conc: 1` 运行它们。

- 默认 SLO（每个场景相同）：TTFT `avg < 8000 ms`，ITL `p90 < 50 ms`。
- `IX_SLO` 覆盖：将该环境变量设为以 `;` 连接的 `metric:stat:op:threshold` 元组（stat 取 `avg`、`p50`、`p90`、`p95`、`p99`；op 取 `lt`、`le`、`gt`、`ge`），例如 `IX_SLO='time_to_first_token:avg:lt:8000;inter_token_latency:p90:lt:50'`。它在该次运行中替换场景的默认 SLO；启动脚本通过 `python -m infx.serving.slo` 校验，格式错误时在引擎启动前直接中止。（CLI 的 `--search-sla` 参数在 `sweep:` 块旁边会被静默忽略，因此用环境变量。）
- 用一个 serving YAML 派发整个网格（或其子集）：

  ```bash
  utils/serving/sweep_dispatch.sh serving/<name>.yaml                  # 全部 24 个
  utils/serving/sweep_dispatch.sh serving/<name>.yaml --only 'sweep-isl8k-*'
  ```

  脚本为每个场景写一份临时 YAML 副本（替换 `scenario:`、置 `conc: 1`、在 `name:` 后追加场景短名），再通过 [`utils/serving/dispatch.sh`](../utils/serving/dispatch.sh) 逐个派发。
- 一个 run 而非 24 个：`scenario:` 也接受列表。列出多个场景的 serving YAML 会派发为**一个 workflow run，每个场景一个 job**（`conc:` 必须恰好一个值）；每个 job 的 `exp-name` 为 `<name>-<scenario>`，逐 job 的 scenario/loader/search 存放在以 `exp-name` 为键的 `scenario-map` 中（列表时扁平的 `serving.scenario` 保持为空，缺失条目会响亮失败，而不是把同一个场景跑 24 遍）：

  ```yaml
  scenario: [sweep-isl8k-osl512-cache0, sweep-isl8k-osl512-cache50]
  conc: 1
  ```

  用普通的 [`utils/serving/dispatch.sh`](../utils/serving/dispatch.sh) 派发即可 —— 无需临时副本，只产生一个 run URL。
- 读取结果：每个任务的产物包含 `search_history.json` 和一个结果 JSON，其 `state` 取以下之一：
  - `ok` —— 收敛；`max_conc` 满足 SLO，
  - `cliff` —— 可行最大值正好落在失败边界上，
  - `no_pass_in_range` —— 没有任何探测并发满足 SLO，
  - `no_failure_in_range` —— 所有探测并发都满足；应放宽搜索区间，
  - `max_iterations` —— 迭代预算耗尽，未得到干净边界，
  - `no_data` —— 搜索根本没有产生边界。

  所有状态都是结果字段，不是失败的任务。结果还包含 `max_conc`、`binding_constraint`、`rps`、`tpm`、`ttft_avg_ms` 与 `itl_p90_ms`。用 `python utils/serving/collect_sweep_table.py <artifact-dir>` 把 24 个结果汇成一张表。
- 缓存命中校验：标称命中率来自场景名的 `cacheNN` 后缀。实测 GPU 缓存命中率为尽力而为：存在每探测点或整轮运行的 `server_metrics_export.json` 时 `measured_cache_hit` 会取到值，否则 `measured_cache_hit` 与 `cache_mismatch` 保持 `null`。
- 指标命名：AIPerf 报告 `inter_token_latency`（ITL）；vLLM/SGLang 文档把同一量称为 time-per-output-token（TPOT）。同一测量，不同名称。

## 查看结果与失败

- 状态：`gh run view <run-id> --repo vngcloud/InferenceX`。每个 `conc` 值对应一个任务。除场景时长外，镜像拉取、模型加载与预热约需 20 分钟。
- 产物（`gh run download <run-id> --repo vngcloud/InferenceX -n <artifact>`）以 `<name>_c<conc>` 开头；`results/` 包含 `serving.yaml`、`server.log`、`benchmark.log`、`router.log`（Router 路径）与 `aiperf_artifacts/`。
- 失败时先看 `server.log`，再看 `benchmark.log`。Router 路径下，若任务在 aiperf 启动前失败，先看 `router.log`。`gh run view <run-id> --repo vngcloud/InferenceX --log-failed | tail -100` 显示启动脚本侧（Slurm、enroot）的日志。
- 引擎崩溃反映的是原样提交的配置。是否修改参数由配置所有者决定，而不是由基准测试决定。

## 升级给基础设施负责人

出现以下情况时，请联系 Runner 负责人（集群频道），而不是修改 YAML：

- 镜像 `enroot import` 返回 401/403：Runner 用户需要该仓库的凭据，或需将镜像预先导入为 squash（`hardware-hcm` 上的 `/mnt/containers`）。
- 模型下载返回 401/403：`INFERENCEX_OFFICIAL_RO_HF_TOKEN` 对应的 HF 账号需要该模型的访问权限。
- 因磁盘不足导致下载失败，或任务长时间在 Slurm 队列中等待（如有权限可执行 `ssh slurm-client squeue -p hardware-hcm`）。

## 权威来源链接

| 来源 | 负责内容 |
| --- | --- |
| [`infx/serving/schema.py`](../infx/serving/schema.py) | YAML 字段、校验规则、保留环境变量、允许的 Runner |
| [`infx/serving/to_matrix.py`](../infx/serving/to_matrix.py) | YAML → 基准测试矩阵与运行标题 |
| [`utils/serving/dispatch.sh`](../utils/serving/dispatch.sh) | 本地校验、推送检查、派发 |
| [`benchmarks/single_node/agentic/serve_generic.sh`](../benchmarks/single_node/agentic/serve_generic.sh) | 模型预下载、引擎启动、就绪检查 |
| [`runners/launch_h200-greennode-slurm-1x.sh`](../runners/launch_h200-greennode-slurm-1x.sh)、[`runners/launch_hardware-hcm.sh`](../runners/launch_hardware-hcm.sh) | 资源分配、镜像导入、挂载、Router sidecar |
| [`docs/superpowers/specs/`](./superpowers/specs/) | 设计决策（serving-YAML 派发、tau2、Dynamo Router） |

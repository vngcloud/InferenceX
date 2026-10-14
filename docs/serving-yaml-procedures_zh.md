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

1. 创建 `serving/<name>.yaml`。字段说明与转换规则（模型路径 → HF ID、环境变量、Router 拆分、上下文长度）见 [`.agents/skills/serving-bench/SKILL.md`](../.agents/skills/serving-bench/SKILL.md)。可用示例：[`serving/qwen38-bf16-tp1.yaml`](../serving/qwen38-bf16-tp1.yaml)（仅引擎）与 [`serving/glm53-w4afp8-dynamo.yaml`](../serving/glm53-w4afp8-dynamo.yaml)（Dynamo Router + 引擎）。
2. 从 `scenarios/` 中选择场景（`agentx-cc-062126`、`agentx-cc-062126-smoke`、`custom-agentic-tau2`、`custom-agentic-tau2-smoke`）。新配置先用 `-smoke` 场景。
3. 校验并派发：

   ```bash
   utils/serving/dispatch.sh serving/<name>.yaml
   ```

   脚本执行与 CI 相同的校验。出现 `ERROR <field>: <fix>` 时不会派发任何任务：修改 YAML 后重新运行。成功时会输出运行 URL。使用 `--ref <branch>` 可派发当前分支以外的分支。
4. 如果脚本提示运行未出现，不要重复派发；先执行 `gh run list --repo vngcloud/InferenceX -w e2e-tests.yml -L 5` 检查。

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

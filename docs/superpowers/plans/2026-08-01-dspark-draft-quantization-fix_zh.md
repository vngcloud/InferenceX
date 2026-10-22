# DSpark 草稿模型量化修复实施计划

> **对于智能体执行器：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务执行本计划。步骤使用复选框语法跟踪。

**目标：** 确保 GLM-5.2 W4AFP8 智能体 launcher 以 BF16 加载 DSpark checkpoint，并在本地验证非零接受率。

**架构：** 不修改 SGLang 通用行为，仅修复受影响的 InferenceX launcher。通过临时仓库布局和 shell stub 执行真实脚本，再在 H200 上验证 DP4/EP8 CUDA graph production path。

**技术栈：** Bash、pytest、Docker、SGLang metrics。

## 全局约束

- 目标模型保留 `--quantization w4afp8`。
- 草稿模型使用 `--speculative-draft-model-quantization unquant`。
- 自动化测试验证生成命令，而不是搜索源码文本。
- 本地接受率测试使用 DP4/EP8 与 CUDA graph。

---

### 任务 1：保护生成的 launcher 命令

**文件：**
- 新建：`utils/test_glm52_dspark_recipe.py`
- 修改：`benchmarks/single_node/agentic/glm5.2dspark_fp4_h200_sglang.sh:27-35`

先编写失败测试：在临时目录执行真实 launcher，提供 no-op `benchmark_lib.sh`、伪 `nvidia-smi` 和临时 draft 文件，验证生成命令同时包含目标 W4AFP8 与草稿 `unquant`。确认 RED 后，使 `DRAFT_MODEL_PATH` 可由环境覆盖并加入 draft quantization flag。随后运行聚焦测试及 `utils/test_benchmark_lib.py`，全部通过后提交。

### 任务 2：验证本地接受率

在 H200 上使用 W4AFP8 target、DP4/EP8、DP Attention、CUDA graph、gamma 4 和 draft `unquant` 启动服务。发送至少生成 128 token 的确定性 chat request，直接读取 `Decode batch` 与 Prometheus metrics；活跃 DP rank 必须满足 accept rate 大于零且 accept length 大于 1。最后停止精确的 debug container，并确认没有遗留 GPU 进程。

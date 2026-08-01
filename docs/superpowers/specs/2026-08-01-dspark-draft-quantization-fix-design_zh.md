# DSpark 草稿模型量化修复

[English](2026-08-01-dspark-draft-quantization-fix-design.md) | [中文](2026-08-01-dspark-draft-quantization-fix-design_zh.md)

## 问题

GLM-5.2 W4AFP8 智能体基准测试脚本未单独指定草稿模型量化方式。因此，SGLang 将目标模型的 `w4afp8` 配置继承给 BF16 DSpark checkpoint，导致 TP8 与 DP4/EP8 下的草稿 token 接受率均为零。

## 设计

保留 SGLang 通用的量化继承机制，仅在 GLM-5.2 DSpark 脚本中加入 `--speculative-draft-model-quantization unquant`。目标模型继续使用 W4AFP8，草稿模型则按其声明的 BF16 dtype 加载。

新增脚本回归测试：通过受控 shell stub 执行真实 launcher，并验证生成的 SGLang 命令包含显式的草稿模型量化覆盖。随后运行聚焦自动化测试，并在本地 H200 上使用 DP4/EP8 与 CUDA graph 发起 smoke request，直接从服务端指标和日志读取接受率。

## 成功标准

- 修改脚本前回归测试失败，修改后通过。
- 生成命令保留目标模型 `--quantization w4afp8`，并为草稿模型加入 `unquant`。
- 本地 production-path smoke test 的投机解码接受率大于零。

# 专家分布诊断

[English](DISTRIBUTION.md)

在研究分支调度 e2e-tests.yml，设置 start-from=distribution_b_ep1。
仅在 h200-2 上串行执行 EP1、EP8 两个诊断任务，不运行性能筛选。
固定原镜像和模型，关闭 EAGLE、HiCache、radix cache、CUDA graphs 和 EPLB；
EP8 使用 DeepEP normal。per_pass 记录器增加阶段和本地输入张量长度元数据，
上传修改前后的源码哈希。每个 CI 任务上限 60 分钟，探针硬上限 20 分钟，
15 分钟后不再启动新案例；单次请求超时五分钟。

先校准单请求 8、64、512 个输入 token。每个有效 MoE 层必须满足
分配数 = 8 × 路由 token 数；失败立即停止，禁止按猜测比例修正。
然后测 prefill：512/8192/32768 token，batch 1/8；decode：512/8192 上下文，
batch 1/8/16，输出 33 token。固定输入 token IDs 并保存哈希。
逐层、逐 forward mode 分析；总计数必须满足
8 × batch × (输入 token + 输出 token - 1)。

关注：分配守恒、活跃专家数、空专家比例、每专家 token 数 M、最热专家占比、
熵对应的有效专家数、rank 最大/平均负载、rank 负载变异系数和平均/最大负载比。
单 token 的 top-k=8 天然导致至少 248/256 专家为空，这不等于负载不均。
均值/最大负载比是计数代理，不能当作 GPU 利用率或性能提升。

保存所有 rank 的原始 per-pass 数据和专家映射。验证 EP1 副本后只保留一份；
EP8 使用实际物理专家映射计算所有者。EP1 的 EP8 rank 负载只是反事实投影。
缺失 rank、映射错误、副本不一致或守恒失败时不输出可解释的结论。
案例总量可能掩盖瞬态热点，不假设跨 rank 的 pass ID 已对齐。

本轮固定重复代码/文字，是形状和计数诊断；相同批次输入属于相关路由压力案例，
不能代表 AgentX 生产流量。守恒通过后，再用真实轨迹分析逐层、对齐 pass 的
p50/p95/最大负载，并在单独 profile 中关联 GEMM、通信及等待时间。

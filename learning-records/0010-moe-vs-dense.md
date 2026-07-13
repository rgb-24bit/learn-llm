# 0010: MoE（混合专家）vs Dense——架构差异与 minimind 实现

**日期**: 2026-07-13
**来源**: 第三阶段教学——GPT 架构扩展

## 关键要点

### 新学概念
- ✅ **Dense vs MoE 的本质差异**：只差在 FFN 层，Attention 不变
- ✅ **Router 机制**：Linear(hidden, num_experts) → softmax → Top-K → 加权合并
- ✅ **专家角色自然涌现**：非人为预设，由训练数据驱动分化
- ✅ **每层独立专家**：Block 0 的专家跟 Block 1 的专家完全无关
- ✅ **Load Balancing Loss**：(负载 × 平均概率).sum() × num_experts × coef
- ✅ **MoE 不影响 KV Cache**（因为 KV Cache 在 Attention 层）

### minimind 具体实现
- 配置：4 专家，Top-1，无噪声，无容量限制
- Router = Linear(768, 4)
- 专家池 = ModuleList([FeedForward × 4])
- aux_loss 系数 5e-4
- `MiniMindBlock.mlp = FeedForward or MOEFeedForward`——一行切换

### 关键认知
- MoE 不是 Dense 的替代，而是 FFN 的替代
- 总参数量大但激活参数少 → 更快推理、更大显存需求
- DeepSeek 的改进：细粒度专家（64~160 个）+ 共享专家

## 关联笔记
- `lessons/0002-moe-vs-dense.html`（正式 Lesson）
- `reference/moe-cheatsheet.html`（速查表）
- 前序学习记录：`0009-kv-cache.md`、`0008-gpt-model-architecture-block.md`

## 下一步建议
- Pretrain 训练循环（minimind 的 train_pretrain.py）
- 或 DeepSeek 的 MLA（Multi-Head Latent Attention）
- 或更深入：Expert Parallelism 如何分布式部署 MoE


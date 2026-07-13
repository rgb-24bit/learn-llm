# 0009: KV Cache——生成推理的核心优化

**日期**: 2026-07-13
**来源**: 第三阶段教学——GPT 推理优化

## 关键要点

### 新学概念
- ✅ **KV Cache 的本质**：把每步算过的 K 和 V 存起来，避免重复计算历史位置
- ✅ **Prefill vs Decode**：第一个 token 整段并行算，后续逐 token 追加
- ✅ **内存代价**：每 token 每层存 2×head_dim×num_heads 个数字，与序列长度成正比
- ✅ **复杂度**：KV 生成从 O(n²) 降到 O(n)
- ✅ **Q 不 Cache**：每次生成新 token 需要全新的 Q
- ✅ **Cache 生命周期**：一次对话/生成任务的整个周期，不跨层共享

### 关键认知
- Prefill 是 compute-bound，Decode 是 memory-bound
- 大模型上下文窗口受限的根本原因之一是 KV Cache 内存
- MQA/GQA 通过让多个 Q 头共享 K/V 来减少 cache 量

### 与旧知识的连接
- RoPE 旋转在入 Cache 前完成
- Causal mask 在生成时自然满足
- 每层 Block 有独立 KV Cache
- Attention 的 QKV 计算中只有 Q 是全重算的

## 关联笔记
- `lessons/0001-kv-cache.html`（正式 Lesson）
- `reference/kv-cache-cheatsheet.html`（速查表）
- 前序学习记录：`0008-gpt-model-architecture-block.md`、`0006-six-step-full-pipeline.md`

## 下一步建议
- Pretrain 训练循环的具体实现
- 或深入 Prefill vs Decode 的详细对比
- 或 Flash Attention 如何加速 Attention


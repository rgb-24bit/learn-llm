# 02: Embedding——把 token 变成向量

**日期**: 2026-07-09
**来源**: minimind model + LLMs-from-scratch ch02

## 关键要点

### 两种 Embedding（不是一回事）
- **内部 Token Embedding**：LLM 第一层，token ID → 向量，维度 = hidden_size
- **独立 Embedding 模型**（如 BGE/E5）：任意文本 → 向量，用于检索

### 权重绑定（tie_word_embeddings）
输入层（token→向量）和输出层（向量→token概率）共享同一套参数。节省约 8% 参数。新模型（LLaMA/Qwen/DeepSeek）大多不绑定了。

### Embedding 占比合理值
5%-15%。模型越小占比越大。GPT-2 的 31% 偏高。

## 关联笔记
- `notes/阶段0-基础八问.md` — Q4 Embedding 维度
- `notes/阶段0-2-基础八问续.md` — Q9 Embedding 模型 + Q10 占比

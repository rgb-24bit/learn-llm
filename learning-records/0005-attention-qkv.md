# 05: Attention 机制——QKV 本质

**日期**: 2026-07-09
**来源**: minimind `MiniMindAttention` + LLMs-from-scratch ch03

## 关键要点

### QKV 三兄弟
- **Q（Query）**：「我在找什么？」— 当前 token 的查询
- **K（Key）**：「我有什么？」— 每个 token 的标签
- **V（Value）**：「我能提供什么？」— 每个 token 的内容

### 计算流程
1. 每个 token × W_Q → Q，× W_K → K，× W_V → V
2. Q 和所有 K 做点积 → 关注度分数
3. 除 √d_k（防梯度爆炸）+ softmax（归一化到 0-1）
4. 用关注度对 V 做加权平均 → 得到上下文物化的新向量

### Causal Attention（因果注意力）
LLM 只看当前 token 左边的内容。通过 mask 矩阵实现：右上角全遮掉。

### Multi-Head Attention（多头注意力）
多个 QKV 投影并行（如 8 头），每个头学到不同的关注模式。最后拼接+线性投影。

### KV Cache
推理时把已经算过的 K 和 V 缓存下来，每步只算新 token 的 Q 和 K，大幅加速。代价是多占显存。

## 关联笔记
- `notes/阶段2-Attention代码逐行解读.md`
- `notes/阶段2-多头注意力的分工与实现.md`

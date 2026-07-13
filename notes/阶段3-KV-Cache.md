# 🗂️ KV Cache——LLM 推理核心优化

> 第三阶段 — GPT 推理优化
> 2026-07-13

## 核心问题

训练是**并行**（整段一起算），推理是**串行**（逐词生成）。序列每步 +1，如果每步重算历史所有位置的 Q 和 K，计算量 O(n²)。

## 解法

**存起来复用。**

每步只算新 token 的 K_new, V_new → 追加到 cache。Attention 时用 Q_new × [K_cached, K_new]。

## Prefill vs Decode

| | Prefill（首个 token） | Decode（后续） |
|---|---|---|
| 输入 | 整条 prompt（200 token） | 1 个新 token |
| Cache | 空 → 填满 | 已有 → 追加 1 个 |
| 瓶颈 | 算力（compute-bound） | 内存带宽（memory-bound） |

## 内存代价

LLaMA 7B（32×32×128），float16，4K seq：
`2 × 32 × 32 × 128 × 4096 × 2 = 2 GB`

## 关键要点

- K 和 V 被 Cache，Q 不 Cache（用完就扔）
- RoPE 旋转在入 Cache 前完成
- 每层 Block 有独立 Cache
- MQA/GQA 减小 cache：多个 Q 头共享 K/V

## 复杂度的「欺骗性」

Attention 的 Q×K^T 点数计算没减少（每步仍是 n 个点积），但 K/V 本身的**产生**（Linear 投影）从 O(n²) 降到 O(n)。实际推理中，K/V 投影矩阵乘是主要瓶颈。

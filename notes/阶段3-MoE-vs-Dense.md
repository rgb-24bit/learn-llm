# 🏗️ MoE vs Dense——混合专家模型

> 第三阶段 — GPT 架构扩展
> 2026-07-13

## 核心差异

Dense 和 MoE 只差在 **FFN 层**，Attention 部分完全一致。

```
Dense:  Pre-Norm → Attn → +残差 → Pre-Norm → [    1 个 FFN     ] → +残差
MoE:    Pre-Norm → Attn → +残差 → Pre-Norm → [Router → Top-2专家] → +残差
```

## Router 机制

线性层打分：`x @ W_router(768×4)` → softmax → Top-K → 加权合并

## 专家角色非人为预设

随机初始化 → 训练中相似 token 走同专家 → 参数朝该方向调整 → 自然分化

## 每层独立

Block 0 的 8 个专家 ≠ Block 1 的 8 个专家。32 层 = 256 个独立专家。

## 代价

- ✅ 推理更快（激活参数少）
- ❌ 显存更大（所有专家须加载）
- ❌ 负载不均衡（Router 垄断 → aux_loss 惩罚）

## minimind 实现特色

- 4 专家、Top-1（比工业界的 Top-2 更激进）
- 28 行核心逻辑，一行 `use_moe` 开关
- 不含噪声/容量限制——专为教学简化

## 对比汇总

| | Dense | MoE |
|---|---|---|
| FFN | 1 个 | N 个专家 |
| 激活 | 100% | Top-K/N |
| 显存 | 小 | 大 |
| 速度 | 基准 | 快 |
| KV Cache | 有 | 同左（无影响） |

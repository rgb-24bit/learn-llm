# 0014: Pretrain 训练循环

**日期**: 2026-07-16
**来源**: minimind `trainer/train_pretrain.py` + `model/model_minimind.py`

## 关键要点

### 训练循环六步走
一个训练步（step）的完整流程：
1. **取数据**：从数据集中取一个 batch，拿到 input_ids 和 labels
2. **前向传播**：input_ids → Embedding → N 层 Transformer Block → lm_head → logits
3. **算 Loss**：logits 和 labels 算交叉熵（只算预测位置的 loss，填充位忽略）
4. **反向传播**：loss → 每层算梯度（gradient），梯度累积多步后再更新
5. **梯度裁剪**：限制梯度最大值，防梯度爆炸
6. **优化器更新**：AdamW 用梯度更新参数 → 清空梯度

### 关键设计
- **梯度累积**：batch_size=32 + accumulation_steps=8 → 实际有效 batch = 256。先攒 8 步的梯度再更新，等效于用更大的 batch 训练，但显存需求不变
- **混合精度训练**：用 bfloat16 计算正向传播，float32 维护优化器状态。又快又省显存
- **余弦学习率衰减**：学习率从 5e-4 开始，按余弦曲线慢慢降到 5e-5（最高值的 10%）。前几步没专门 warmup

### minimind 64M 模型的实际参数
| 参数 | 值 |
|:-----|:---|
| 层数 | 8 |
| 隐藏维度 | 768 |
| 注意力头数 | 12 |
| 词表大小 | 6400 |
| 最大序列长度 | 340 token |
| 参数量 | 64M |
| 训练数据 | ~2.1B token |
| batch size | 32（累积后 256） |
| 学习率 | 5e-4 |
| 余弦衰减终值 | 5e-5（最高值 10%） |
| 梯度裁剪阈值 | 1.0 |
| 一轮 epoch | ~24,000 step |
| 总训练步 | ~48,000 step（2 epochs） |
| 训练时间 | 1-2 天（单卡 RTX 3090） |

### Loss 计算细节
- `logits[..., :-1, :]` — 每个位置的输出预测的是**下一个 token**
- `labels[..., 1:]` — 真实标签是右移一位的输入
- `ignore_index=-100` — 填充位不参与 loss 计算
- 返回两个 loss：`logits_loss`（预测损失）+ `aux_loss`（MoE 的负载均衡损失，Dense 模型为 0）

## 数学直觉
交叉熵 loss 的直观理解：模型预测每个位置下一个 token 的概率分布，越接近 one-hot 真实分布，loss 越低。

假设词表 6400，模型对一个位置的输出：
- 正确 token 概率 0.2（理想是 1.0）
- 其他 6399 个 token 概率合计 0.8
- loss ≈ -ln(0.2) ≈ 1.6
    
训练结束时，正确 token 概率可能到 0.8-0.9，loss 降到 0.1-0.2。

## 关联笔记
- `0000-training-three-phases.md` — 三阶段训练框架
- `0008-gpt-model-architecture-block.md` — Block 结构（前向传播中每一层的结构）
- `0009-kv-cache.md` — 推理优化（训练时不用 KV Cache）
- `0010-moe-vs-dense.md` — MoE 的 aux_loss 在训练中的作用

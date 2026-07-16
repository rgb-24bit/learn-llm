# 0015: SFT 与 LoRA——从"会预测"到"会回答"

**日期**: 2026-07-16
**来源**: minimind `trainer/train_full_sft.py` + `trainer/train_lora.py` + `model/model_lora.py`

## 关键要点

### SFT（监督微调）的核心创新
- **目标不同**：Pretrain 学"下一个词是什么"，SFT 学"别人问问题要回答"
- **数据格式**：多轮对话（user → assistant → user → assistant），用 chat template 拼成一段文本
- **Loss masking**：只计算 assistant 回复部分的 loss，user 提问部分 ignore（label=-100）。这是 SFT 与 Pretrain 最大的结构性差异
- **学习率极低**：SFT 的 lr=1e-5，是 Pretrain（5e-4）的 1/50。因为模型已经学会了语言，只需要微调行为模式

### LoRA（低秩适配）
- **动机**：全量 SFT 需要更新所有 64M 参数，显存需求大。LoRA 只训练极少量参数
- **原理**：冻结原始权重 W，插入两个小矩阵 A（768→r）和 B（r→768），输出 = Wx + BAx
- **秩 r**：minimind 用 r=16，每个 LoRA 参数 = 2×768×16 = 24,576，不到原始 W（768×768=589,824）的 5%
- **初始化**：A 高斯初始化，B 全 0 → 初始 LoRA 输出为 0，不破坏模型已有能力
- **应用位置**：只挂在 Attention 的 Q/K/V/O 四个线性层上（in_features==out_features 的 Linear）
- **总参数量**：8层×4个×24,576 ≈ 79 万参数，仅占 64M 模型的 1.2%

### Full SFT vs LoRA 决策

| 维度 | Full SFT | LoRA SFT |
|:-----|:---------|:---------|
| 训练参数 | 全部参数（如 64M） | 仅 LoRA 参数（~0.79M） |
| 学习率 | 1e-5 | 1e-4（可以更大，因为参数少、从头学） |
| 显存需求 | 高 | 低（节省约 30-50%） |
| 保存体积 | 全模型（~130MB） | 仅 LoRA（~1.6MB） |
| 训练数据量 | 几千~几万条 | 几百~几千条 |
| 多任务共存 | 一个模型一个任务 | 一个模型多套 LoRA，随时切换 |
| 能力上限 | 更高（全参数更新） | 略低（表达能力受秩限制） |

### SFT 数据样例（JSONL 格式）
```json
{"conversations": [
  {"role": "system", "content": "你是一个有帮助的助手。"},
  {"role": "user", "content": "什么是机器学习？"},
  {"role": "assistant", "content": "机器学习是人工智能的一个分支..."}
]}
```

## 关联笔记
- `0014-pretrain-training-loop.md` — SFT 的训练循环结构几乎和 Pretrain 一样，只是数据不同
- `0000-training-three-phases.md` — 三阶段训练框架中的第二阶段

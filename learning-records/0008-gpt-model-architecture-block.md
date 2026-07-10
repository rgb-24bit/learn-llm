# 0008: GPT 模型架构——Transformer Block 完整结构

**日期**: 2026-07-10
**来源**: 第三阶段教学——GPT 架构

## 关键要点

### 新学概念
- ✅ **Transformer Block = Pre-Norm → Attention → +残差 → Pre-Norm → FFN → +残差**
- ✅ **残差连接**：x + F(x)，让梯度直通不衰减，8 层共 16 个残差
- ✅ **RMSNorm**：除 RMS 归一化，不减均值（比 LayerNorm 快 2 倍）
- ✅ **FFN (SwiGLU)**：W_gate 检测特征 + W_up 产生候选 + W_down 压缩
- ✅ **Attention vs FFN**：Attention 收集邻居信息（开会），FFN 加工信息（消化）
- ✅ **整体组装**：8 层 Block + 最终 RMSNorm + lm_head 投影到词表

### 前馈神经网络基础
- 前馈 = 数据单向流动，无循环
- 三层结构：输入层 → 隐藏层（+激活）→ 输出层
- 激活函数（SiLU）注入非线性
- 前馈网络是深度学习的"基础积木"

### 与旧知识的连接
- 之前学的 7 步 Attention 是 Block 的前半段
- QKV 计算 → softmax@V → 残差 → 归一化 → FFN → 残差

## 关联笔记
- `notes/阶段3-GPT模型架构-完整Block.md`（本日新写）
- `notes/从文本到输出-LLM七步完整流程.md`
- 前序学习记录：`0006-six-step-full-pipeline.md`、`0007-rope-deep.md`

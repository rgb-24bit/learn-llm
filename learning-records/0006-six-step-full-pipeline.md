# 0006: 从文本到输出——LLM 六步完整流程串讲

**日期**: 2026-07-10
**来源**: 主人复述理解 + Fairy 复查修正

## 关键要点

### 主人已掌握（复查确认）
1. ✅ Tokenizer 训练：BPE 按词频合并相邻 pair，直至凑够词表大小
2. ✅ Embedding：token id 查表得向量
3. ✅ 绝对位置 vs RoPE：一个加在输入，一个旋转 Q/K
4. ✅ QKV 投影：三个独立矩阵 × token 向量
5. ✅ Q×K^T → 注意力分数
6. ✅ softmax(分数) @ V → 加权混合后的新向量

### 修正内容
- **BPE 推理**是「按频率顺序应用合并规则」，**不是**「最长前缀匹配」
- 「最长前缀匹配」是 Unigram Model (SentencePiece) 的做法
- 对 GPT-2/LLaMA 用的 BPE，推理时逐条检查 merge rules 能否应用于当前序列

### 维度流转型的重点强化
- Q×K^T 的结果是 [seq×seq] **矩阵**，不是向量
- 这个矩阵的「列数 = token 个数」对齐 V 的「行数 = token 个数」
- [seq×seq] @ [seq×d] = [seq×d] — 维度自然对上

## 关联笔记
- `notes/从文本到输出-LLM六步完整流程.md`（本日新写）
- `notes/QKV和RoPE-最土的理解.md`
- `notes/注意力维度-Softmax矩阵乘V的对齐方式.md`
- 前序学习记录：`0005-attention-qkv.md`

## 下一步建议
- 进入**第三阶段：GPT 完整架构**
  - LayerNorm / RMSNorm
  - Feed-Forward Network (FFN)
  - 残差连接
  - 8 层 Block 组装 + 输出层

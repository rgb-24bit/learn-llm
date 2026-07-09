# 01: Tokenizer 与词表设计

**日期**: 2026-07-09
**来源**: minimind `train_tokenizer.py` + LLMs-from-scratch ch02

## 关键要点

### BPE 分词原理
- 起始词表 = 256 个 byte → 反复合并最高频相邻对 → 直到目标词表大小
- ByteLevel 预分词保证任何语言都不会遇到不认识的字
- 词表大小影响：中文 vs 英文、模型规模、Embedding 占比

### 词表大小权衡
| 方向 | 好处 | 坏处 |
|:--|:-----|:-----|
| 大词表 | token 数少→序列短 | Embedding 层大→多参数 |
| 小词表 | Embedding 层小 | 序列长→训练慢 |

主流模型 32k-128k。词表增长不是线性的：前 10k 覆盖 95%+ 常见词。

## 深层理解

GPT-2 的 vocab_size=50257 对 124M 模型是历史设计失误——31% 参数卡在 Embedding 层。后续模型（LLaMA 等）用 32k 词表配大模型。

## 关联笔记

- `notes/阶段0-基础八问.md` — Q3 Tokenizer + Q11 词表大小
- `notes/阶段0-2-基础八问续.md` — Q10 Embedding 占比

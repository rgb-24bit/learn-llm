# RESOURCES: LLM 学习资源

## 核心教材

### LLMs-from-scratch
- **作者**: Sebastian Raschka
- **GitHub**: https://github.com/rasbt/LLMs-from-scratch
- **用途**: 基础原理学习，手把手从零写 GPT
- **覆盖**: Tokenizer → Attention → GPT → Pretrain → SFT → RLHF/DPO → Reasoning
- **推荐理由**: 代码逐行解释，适合跟学

### minimind
- **作者**: jingyaogong
- **GitHub**: https://github.com/jingyaogong/minimind
- **用途**: 对照实现，看真实的教学级 LLM 怎么写
- **特色**: Dense 64M / MoE 198M，对齐 Qwen3 架构，3 块钱可训练
- **覆盖**: Tokenizer → Pretrain → SFT/LoRA → DPO/PPO/GRPO → Agentic RL → 蒸馏

## 辅助教材

### The Annotated Transformer (Harvard NLP)
- **链接**: http://nlp.seas.harvard.edu/2018/04/03/attention.html
- **用途**: Attention Is All You Need 论文的逐行代码实现
- **推荐理由**: 经典中的经典

### 3Blue1Brown — Neural Networks
- **链接**: https://www.3blue1brown.com/topics/neural-networks
- **用途**: 直观理解神经网络基础
- **推荐理由**: 动画讲解，直觉建立首选

### Karpathy — Let's build GPT
- **链接**: https://www.youtube.com/watch?v=kCc8FmEb1nY
- **用途**: 从零手写 GPT，2 小时视频
- **推荐理由**: Karpathy 风格，干净直接

## 论文（后续参考）

| 论文 | 主题 | 重要性 |
|------|------|:------:|
| Attention Is All You Need (2017) | Transformer 原始论文 | ⭐⭐⭐ |
| RoFormer: Enhanced Transformer with Rotary Position Embedding (2023) | RoPE | ⭐⭐⭐ |
| YaRN: Efficient Context Window Extension (2023) | 上下文扩展 | ⭐⭐ |
| DeepSeek-R1 (2025) | GRPO + Reasoning | ⭐⭐⭐ |
| Llama 2: Open Foundation and Fine-Tuned Chat Models (2023) | SFT + RLHF | ⭐⭐ |

## 评估基准（阶段 5）

### MMLU
- **论文**: https://arxiv.org/abs/2009.03300 — Hendrycks et al., 2020
- **核心**: 15,908 道四选一，57 学科，5-shot
- **特点**: 随机基线 25%，人类 ~89%，前沿 ~93%

### GSM8K
- **论文**: https://arxiv.org/abs/2110.14168 — Cobbe et al., 2021
- **核心**: 8,500 道小学数学应用题，2–8 步推理
- **特点**: 推动了 Chain-of-Thought 技术，前沿已达 99%

### HumanEval
- **论文**: https://arxiv.org/abs/2107.03374 — Chen et al., 2021
- **核心**: 164 道 Python 函数题，pass@k 评分
- **特点**: 前沿已达 90%+，不反映真实工程能力

### SWE-bench
- **论文**: https://arxiv.org/abs/2310.06770 — Jimenez et al., 2024
- **核心**: 2,294 道真实 GitHub Issue 修复
- **特点**: 至今无人满分，当前最难代码基准

### Chatbot Arena
- **地址**: https://lmarena.ai/
- **核心**: 匿名对战 + 人类投票 + Elo 评分

## 视觉/多模态

### ViT — Vision Transformer (2021)
- **论文**: https://arxiv.org/abs/2010.11929
- **核心贡献**: 把图片切 patches → 线性投影 → 当 token 序列用 Transformer 处理
- **重要性**: 多模态模型的基础视觉编码器

### CLIP — Contrastive Language-Image Pre-training (2021)
- **论文**: https://arxiv.org/abs/2103.00020
- **核心贡献**: 图片编码器 + 文本编码器，对比学习对齐视觉和语言空间
- **重要性**: 多模态 LLM 的"翻译官"

### LLaVA (2023)
- **论文**: https://arxiv.org/abs/2304.08485
- **GitHub**: https://github.com/haotian-liu/LLaVA
- **核心贡献**: 最简单的多模态方案——CLIP 视觉编码器 + 线性投影层 + LLM
- **重要性**: 证明了"投影层就够了"，引发了多模态 LLM 热潮

### Qwen2-VL (2024)
- **论文**: https://arxiv.org/abs/2409.12191
- **核心贡献**: 动态分辨率（每张图 token 数不一样）、多模态旋转位置编码、P2P 压缩
- **重要性**: 工业级多模态模型的最佳实践案例

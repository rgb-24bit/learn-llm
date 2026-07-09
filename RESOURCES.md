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

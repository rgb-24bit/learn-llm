# 🧠 LLM 学习笔记

> 基于 [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)（基础原理）+
> [minimind](https://github.com/jingyaogong/minimind)（实现参考）的学习记录
>
> 以阅读了解为主，不做动手编码

---

## 📖 学习计划总览

学习路线分为 **7 个阶段**，从 LLM 全貌概览逐步深入到前沿主题，每个阶段同时标注 LLMs-from-scratch 的对应章节（打基础）和 minimind 的对应代码（看实现）。

| 阶段 | 主题 | LLMs-from-scratch | minimind |
|------|------|-------------------|----------|
| 一 | LLM 概览与文本表示 | ch01, ch02 | dataset/ + tokenizer |
| 二 | 注意力机制 | ch03 | model/model_minimind.py (Attention 部分) |
| 三 | GPT 模型架构 | ch04 | model/model_minimind.py (完整模型) |
| 四 | 预训练 Pretrain | ch05 | trainer/train_pretrain.py |
| 五 | 监督微调 SFT & LoRA | ch06 | trainer/train_full_sft.py + train_lora.py |
| 六 | 人类对齐 RLHF/DPO | ch07 | trainer/train_dpo.py + train_ppo.py + train_grpo.py |
| 七 | 前沿进阶 | appendix + reasoning | Agentic RL + 蒸馏 + MoE |

---

## 第一阶段：LLM 概览与文本表示

### 学习目标
理解 LLM 是什么、能做什么，以及模型如何把人类语言变成计算机能处理的形式。

### 📘 LLMs-from-scratch

#### ch01：理解大语言模型
| 文件 | 要点 |
|------|------|
| [ch01/README.md](../LLMs-from-scratch/ch01/README.md) | LLM 发展历程、什么是大语言模型、训练三阶段（Pretrain → SFT → RLHF） |
| [ch01/reading-recommendations.md](../LLMs-from-scratch/ch01/reading-recommendations.md) | 阅读建议与学习路径 |

#### ch02：文本数据处理
| 文件 | 要点 |
|------|------|
| [ch02/01_main-chapter-code/](../LLMs-from-scratch/ch02/01_main-chapter-code/) | Tokenizer（BPE）、词嵌入（Embedding）、位置编码 |
| [ch02/02_bonus_bytepair-encoder/](../LLMs-from-scratch/ch02/02_bonus_bytepair-encoder/) | BPE 算法详解 |
| [ch02/04_bonus_dataloader-intuition/](../LLMs-from-scratch/ch02/04_bonus_dataloader-intuition/) | 数据加载与批处理 |

**核心技术点：**
- Tokenization（分词）：文本 → token ID
- Byte Pair Encoding（BPE）：子词分词算法
- Embedding 层：将 token ID 映射为稠密向量
- 位置编码：让模型感知 token 的先后顺序

### 🔧 minimind 对照实现

| 文件 | 对应概念 |
|------|---------|
| [model/tokenizer.json](../minimind/model/tokenizer.json) | 训练好的 BPE + ByteLevel 词表 |
| [dataset/lm_dataset.py](../minimind/dataset/lm_dataset.py) | 数据加载与 tokenize |
| [dataset/dataset.md](../minimind/dataset/dataset.md) | 数据集说明 |
| [trainer/train_tokenizer.py](../minimind/trainer/train_tokenizer.py) | 分词器训练代码 |

**minimind 特色：** 支持 `<tool_call>`、`<think>` 等特殊标记，为后续 agent 和推理能力预留。

---

## 第二阶段：注意力机制（Attention）

### 学习目标
理解 Transformer 的核心——注意力机制如何让模型关注输入中的重要部分。

### 📘 LLMs-from-scratch

#### ch03：编码注意力机制
| 文件 | 要点 |
|------|------|
| [ch03/01_main-chapter-code/](../LLMs-from-scratch/ch03/01_main-chapter-code/) | 从零实现注意力：因果注意力（Causal Attention）、多头注意力（Multi-Head Attention） |
| [ch03/02_bonus_efficient-multihead-attention/](../LLMs-from-scratch/ch03/02_bonus_efficient-multihead-attention/) | 高效多头注意力实现 |
| [ch03/03_understanding-buffers/](../LLMs-from-scratch/ch03/03_understanding-buffers/) | PyTorch 缓冲区理解 |

**核心技术点：**
- 自注意力（Self-Attention）：QKV 计算
- 因果注意力（Causal Attention）：LLM 只看到前面的 token
- 多头注意力（Multi-Head Attention）：多个注意力头并行
- Dropout 与权重初始化

### 🔧 minimind 对照实现

| 文件 | 对应概念 |
|------|---------|
| [model/model_minimind.py](../minimind/model/model_minimind.py) | `MiniMindAttention` 类——从零实现的多头自注意力 |
| 搜索 `class MiniMindAttention` | QKV 投影、注意力计算、RoPE、KV Cache |

**minimind 对比特色：**
- 使用 RoPE（旋转位置编码）替代传统位置编码
- 支持 KV Cache 加速推理
- MoE 版本使用分组注意力

---

## 第三阶段：GPT 模型架构

### 学习目标
理解完整的 GPT/Transformer Decoder 架构，看各个组件如何组装成完整的 LLM。

### 📘 LLMs-from-scratch

#### ch04：从零实现 GPT 模型
| 文件 | 要点 |
|------|------|
| [ch04/01_main-chapter-code/](../LLMs-from-scratch/ch04/01_main-chapter-code/) | GPT 架构：LayerNorm、GELU、Feed Forward、残差连接、整体组装 |
| [ch04/02_performance-analysis/](../LLMs-from-scratch/ch04/02_performance-analysis/) | 模型性能分析 |
| [ch04/03_kv-cache/](../LLMs-from-scratch/ch04/03_kv-cache/) | KV Cache——推理加速核心 |
| [ch04/04_gqa/](../LLMs-from-scratch/ch04/04_gqa/) | Grouped Query Attention（GQA） |
| [ch04/05_mla/](../LLMs-from-scratch/ch04/05_mla/) | Multi-head Latent Attention（MLA）— DeepSeek 用的 |
| [ch04/06_swa/](../LLMs-from-scratch/ch04/06_swa/) | Sliding Window Attention |
| [ch04/07_moe/](../LLMs-from-scratch/ch04/07_moe/) | Mixture of Experts（MoE） |
| [ch04/08_deltanet/](../LLMs-from-scratch/ch04/08_deltanet/) ~ [ch04/10_kv-sharing/](../LLMs-from-scratch/ch04/10_kv-sharing/) | 更多架构变体 |

**核心技术点：**
- Transformer Decoder 架构：LayerNorm → Attention → FFN（残差包围）
- GELU 激活函数
- Layer Normalization 与 RMSNorm
- 参数配置（n_layer, n_head, n_embd 等含义）
- KV Cache 原理与 GQA/MLA 等注意力变体

### 🔧 minimind 对照实现

| 文件 | 对应概念 |
|------|---------|
| [model/model_minimind.py](../minimind/model/model_minimind.py) | 完整 Dense + MoE 模型 |
| 搜索 `class MiniMindBlock` | Transformer Block（Attention + FFN + RMSNorm） |
| 搜索 `class MiniMindMoE` | MoE 专家混合层设计 |
| 搜索 `class MiniMindForCausalLM` | 完整 CausalLM 封装 |

**minimind 对比特色：**
- 主线对齐 Qwen3/Qwen3-MoE 架构——与实际工业模型接轨
- Dense 版本约 64M 参数，MoE 版本约 198M（激活 64M）
- 使用 RMSNorm 代替 LayerNorm
- 完整的 forward 流程：embed → blocks → norm → lm_head

---

## 第四阶段：预训练（Pretrain）

### 学习目标
理解语言模型预训练的核心概念：自回归训练、损失函数、学习率策略、训练循环。

### 📘 LLMs-from-scratch

#### ch05：预训练
| 文件 | 要点 |
|------|------|
| [ch05/01_main-chapter-code/](../LLMs-from-scratch/ch05/01_main-chapter-code/) | 训练循环、损失计算、模型保存与加载 |
| [ch04/04_learning_rate_schedulers/](../LLMs-from-scratch/ch05/04_learning_rate_schedulers/) | 学习率调度策略 |
| [ch05/05_bonus_hparam_tuning/](../LLMs-from-scratch/ch05/05_bonus_hparam_tuning/) | 超参数调优 |
| [ch05/07_gpt_to_llama/](../LLMs-from-scratch/ch05/07_gpt_to_llama/) | GPT → LLaMA 架构演进 |
| [ch05/10_llm-training-speed/](../LLMs-from-scratch/ch05/10_llm-training-speed/) | 训练加速技巧 |
| [ch05/11_qwen3/](../LLMs-from-scratch/ch05/11_qwen3/) ~ [ch05/18_muon/](../LLMs-from-scratch/ch05/18_muon/) | 其他 LLM 的预训练分析 |

**核心技术点：**
- 自回归（Autoregressive）训练：预测下一个 token
- Cross-entropy Loss
- 学习率 warmup + cosine decay
- 训练循环与梯度更新
- 模型评估（perplexity）

### 🔧 minimind 对照实现

| 文件 | 对应概念 |
|------|---------|
| [trainer/train_pretrain.py](../minimind/trainer/train_pretrain.py) | 预训练完整流程 |
| [trainer/trainer_utils.py](../minimind/trainer/trainer_utils.py) | 训练工具函数 |

**minimind 对比特色：**
- 从零实现的 pretrain 循环，不依赖 transformers.Trainer
- 数据量极小（几百万条简中 token），3 块钱就能跑完
- 支持 DDP（分布式数据并行）
- 支持 wandb/swanlab 可视化

---

## 第五阶段：监督微调（SFT）与 LoRA

### 学习目标
理解如何用对话数据微调预训练模型，以及参数高效微调（PEFT）的原理。

### 📘 LLMs-from-scratch

#### ch06：分类微调
| 文件 | 要点 |
|------|------|
| [ch06/01_main-chapter-code/](../LLMs-from-scratch/ch06/01_main-chapter-code/) | 微调策略、分类头、完整微调流程 |
| [ch06/02_bonus_additional-experiments/](../LLMs-from-scratch/ch06/02_bonus_additional-experiments/) | 更多微调实验 |

> ch06 主要聚焦分类微调，ch07 才涉及指令微调（chat 格式），但 SFT 的核心思想（加载预训练权重 → 在小数据集上继续训练）是一致的。

**核心技术点：**
- 全参数微调（Full Fine-tuning）
- 冻结 vs 解冻不同层
- 分类头（Classification Head）
- 微调 vs 预训练的区别

### 🔧 minimind 对照实现

| 文件 | 对应概念 |
|------|---------|
| [trainer/train_full_sft.py](../minimind/trainer/train_full_sft.py) | 全量 SFT 微调 |
| [trainer/train_lora.py](../minimind/trainer/train_lora.py) | LoRA 微调（从零实现，不依赖 peft） |
| [model/model_lora.py](../minimind/model/model_lora.py) | LoRA 层代码 |
| [scripts/convert_model.py](../minimind/scripts/convert_model.py) | LoRA 权重合并 |

**minimind 对比特色：**
- LoRA 从零用 PyTorch 实现——看清低秩适配本质
- 对话式 SFT 数据（chat template）
- 支持 `chat_template + <think>` 模板，为 reasoning 能力准备
- 支持工具调用（tool_call）数据混入

---

## 第六阶段：人类对齐（RLHF / DPO / PPO / GRPO）

### 学习目标
理解如何让 LLM 对齐人类偏好，这是 ChatGPT 效果惊艳背后的关键技术。

### 📘 LLMs-from-scratch

#### ch07：人类反馈微调
| 文件 | 要点 |
|------|------|
| [ch07/01_main-chapter-code/](../LLMs-from-scratch/ch07/01_main-chapter-code/) | 指令微调、RLHF、DPO |
| [ch04/04_preference-tuning-with-dpo/](../LLMs-from-scratch/ch07/04_preference-tuning-with-dpo/) | DPO 算法 |
| [ch07/03_model-evaluation/](../LLMs-from-scratch/ch07/03_model-evaluation/) | 模型评估方法 |

**核心技术点：**
- RLHF 三阶段：SFT → Reward Model → PPO
- DPO：无需 Reward Model 的直接偏好优化
- PPO 的核心思想
- 指令数据格式与模板

### 🔧 minimind 对照实现

| 文件 | 对应概念 |
|------|---------|
| [trainer/train_dpo.py](../minimind/trainer/train_dpo.py) | DPO 从零实现 |
| [trainer/train_ppo.py](../minimind/trainer/train_ppo.py) | PPO 从零实现 |
| [trainer/train_grpo.py](../minimind/trainer/train_grpo.py) | GRPO 从零实现（DeepSeek-R1 风格） |

**minimind 对比特色：**
- DPO / PPO / GRPO 全部从零实现——看清每种算法的核心细节
- 包含 RLAIF（AI 反馈强化学习）数据集
- GRPO 是 DeepSeek-R1 训练路线的关键
- 三者对比：DPO 最简单，PPO 最经典，GRPO 最前沿

---

## 第七阶段：前沿进阶

### 学习目标
了解 MoE、Agentic RL、模型蒸馏、推理（Reasoning）等前沿主题。

### 📘 LLMs-from-scratch 参考

| 文件 | 要点 |
|------|------|
| [ch04/07_moe/](../LLMs-from-scratch/ch04/07_moe/) | MoE 原理 |
| [appendix-A/](../LLMs-from-scratch/appendix-A/) ~ [appendix-E/](../LLMs-from-scratch/appendix-E/) | 附录：环境配置、模型对比、训练技巧等 |
| [reasoning-from-scratch/](../LLMs-from-scratch/reasoning-from-scratch/) | 推理模型（空白，待 Raschka 补完） |

### 🔧 minimind 对照实现

| 文件 | 对应概念 |
|------|---------|
| [model/model_minimind.py](../minimind/model/model_minimind.py) | MoE 实现（`MiniMindMoE`、`MoEGate`） |
| [trainer/train_agent.py](../minimind/trainer/train_agent.py) | Agentic RL——Tool Use 场景下的 GRPO/CISPO |
| [trainer/train_distillation.py](../minimind/trainer/train_distillation.py) | 模型蒸馏（白盒） |
| [trainer/rollout_engine.py](../minimind/trainer/rollout_engine.py) | Rollout 引擎——多轮 Tool-Use 生成 |
| [trainer/train_tokenizer.py](../minimind/trainer/train_tokenizer.py) | 分词器训练 |

**热门话题速览：**
| 主题 | minimind 实现 | 概念简介 |
|------|--------------|----------|
| **MoE** | `MiniMindMoE` + `MoEGate` | 多个专家 FFN + 门控路由，激活少量专家 |
| **Agentic RL** | `train_agent.py` | 在 Tool Use 场景下用强化学习训练 agent |
| **模型蒸馏** | `train_distillation.py` | 大模型知识迁移到小模型 |
| **Reasoning** | `<think>` template + GRPO | 模型学会先思考再回答（类 DeepSeek-R1） |
| **dLM（扩散语言模型）** | Discussion #618 | 离散扩散语言模型（实验性） |
| **Linear Attention** | Discussion #704 | 线性注意力替代 Softmax 注意力（实验性） |
| **视觉/多模态** | MiniMind-V / MiniMind-O | 拓展到视觉与多模态领域 |

---

## 📚 阅读建议

1. **每个阶段的理解层次**：先用 LLMs-from-scratch 看懂原理 → 再看 minimind 如何实现 → 两者对比加深理解
2. **不要纠结代码细节**：核心是理解**为什么这么设计**，而不是抄代码
3. **善用搜索**：在 minimind 代码中搜索 `class` 关键词快速定位核心组件
4. **随时记笔记**：在 `notes/` 目录下按阶段建立自己的总结文件

---

*Happy Learning! 🚀*

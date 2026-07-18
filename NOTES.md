# NOTES: 教学偏好与约定

## 教学风格

- **最土的比喻优先**—比如 RoPE 像转风扇，QKV 像查字典，KV Cache 像写论文贴墙上
- **具体数字算例**—每一步：输入什么 → 怎么算 → 输出什么
- **表格对比**—多个概念并列时用表格
- **不要跳过基础概念**—如果牵涉到数学概念，先解释数学再解释 ML
- **不要贴代码块**—微信聊天看不见代码，必须转成自然语言讲解

## 笔记规范

- 笔记存在 `notes/` 目录，markdown 格式
- 每轮 Q&A 记得 git add + commit + push
- 文件名用 `阶段X-主题.md`

## 知识现状

### 已掌握
- ✅ LLM 训练三阶段：Pretrain → SFT → RLHF/DPO
- ✅ Tokenizer：BPE 原理、ByteLevel、词表大小权衡
- ✅ Embedding：Token Embedding vs 独立 Embedding 模型、权重绑定
- ✅ RoPE：旋转位置编码、二维配对旋转、高频 vs 低频、窗口扩展
- ✅ NTK/YaRN：上下文窗口扩展原理、维度频率分布
- ✅ QKV：Query/Key/Value 的计算与 Attention 本质
- ✅ Causal Attention：因果 mask 的原理
- ✅ Multi-Head Attention：多头并行的分工与实现
- ✅ GPT Transformer Block：Pre-Norm → Attention + 残差 → FFN + 残差
- ✅ RMSNorm：除 RMS 归一化，不减均值
- ✅ FFN/SwiGLU：门控前馈网络，W_gate/W_up/W_down 三权重
- ✅ **KV Cache**：推理时不重算历史 K/V，用内存换速度
- ✅ **图片/视频 Token 化**：ViT patch embedding，三种多模态连接方式
- ✅ **多模态训练机制**：CLIP 对比学习、对齐→SFT 三阶段、模型三件套架构
- ✅ **Pretrain 训练循环**：前向→Loss→反向→梯度累积→AdamW→学习率衰减，每个 step 的完整流程
- ✅ **SFT 与 LoRA**：Loss masking、Full SFT vs LoRA 对比、低秩分解原理
- ✅ **SFT 深层机制**：Pretrain（可能的token）→ SFT（应该的token）→ DPO（更好的token），三个层面的生效方式（Attention指令识别、FFN知识路由、概率压低错误项），SFT不改变知识只改变使用方式
- ✅ **Scaling Law**：PPL ∝ N^(-0.076)，Chinchilla 1:20 法则，边际递减

### 待学（按优先级）
1. ✅ LLM 基准全景（MMLU/GSM8K/HumanEval/SWE-bench/Chatbot Arena）
2. ✅ SFT / LoRA
3. ✅ DPO / GRPO

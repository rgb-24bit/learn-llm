# NOTES: 教学偏好与约定

## 教学风格

- **最土的比喻优先**—比如 RoPE 像转风扇，QKV 像查字典
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
- LLM 训练三阶段：Pretrain → SFT → RLHF/DPO
- Tokenizer：BPE 原理、ByteLevel、词表大小权衡
- Embedding：Token Embedding vs 独立 Embedding 模型、权重绑定
- RoPE：旋转位置编码、二维配对旋转、高频 vs 低频
- NTK/YaRN：上下文窗口扩展原理、维度频率分布
- QKV：Query/Key/Value 的计算与 Attention 本质
- Causal Attention：因果 mask 的原理
- Multi-Head Attention：多头并行的分工与实现

### 待学（按优先级）
1. GPT 完整架构（LayerNorm、FFN、残差连接）
2. KV Cache 原理
3. Pretrain 训练循环
4. SFT / LoRA
5. DPO / GRPO

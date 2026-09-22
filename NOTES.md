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
- ✅ **解码采样与温度**：模型只输出概率分布，选词靠多项式采样（掷偏心骰子）；temperature/top_k/top_p 三个旋钮的机制与顺序；贪心 vs 采样取舍
- ✅ **DeepSeek 特例**：思考模式（V4 默认开启）不支持 temperature/top_p/penalties（官方确认，传入不生效），真正旋钮是 reasoning_effort（low/high/max）；frequency/presence penalty 已 deprecated
- ✅ **API 参数全景**：五大组（生成旋钮/惩罚三兄弟/输出控制/思考与工具/工程细节）；注意 max_tokens 含思考链额度（实测 300 被思考吃光→content 空）；user_id 用于 KV Cache 隔离；DeepSeek 无 seed
- ✅ **视觉记忆专题**（跨领域对照）：人类三档（图标~0.3s / 短时 3~4 件 / 长期 1 万张认对 83%）靠"重绘"会失真；AI 三层（权重 / 上下文+KV Cache / 外部长期）看图为真、存画面刚起步（MemLens 基准、EVM、MemOCR）
- ✅ **专题：Jev/Laya 决策模型**（2026-09-22 热点）：不写字的 System One 模型——choice/score/noul 三原语 + 校准概率；非自回归一次前向 vs 逐 token；RLCD vs RLHF；「零幻觉」= 输出结构受限 ≠ 判断正确

### 待学（按优先级）
1. ✅ LLM 基准全景（MMLU/GSM8K/HumanEval/SWE-bench/Chatbot Arena）
2. ✅ SFT / LoRA
3. ✅ DPO / GRPO

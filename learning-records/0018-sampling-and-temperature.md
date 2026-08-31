# 0018: 解码采样与温度（含 DeepSeek 特例）

**日期**: 2026-08-31
**来源**: 阶段5 教学内容——推理侧采样机制；用户追问 DeepSeek temperature 不起作用

## 关键要点

### 相同输入 → 不同输出的根本原因
- 模型只输出**概率分布**（如 好=0.35 / 晴=0.25 / 不错=0.15 …），真正选词靠**多项式采样（掷偏心骰子）**
- 两次调用 = 摇两次同一枚偏心骰子 → 结果不同是设计而非 bug

### 三个旋钮（minimind generate 实现顺序）
1. **temperature**：logits ÷ T 再 softmax。T<1 尖峰更尖（确定），T>1 更平（随机）
2. **top_k**：只留前 K 个词，其余出局
3. **top_p**：概率从大到小累加，满 p 截断（砍长尾）

数值算例：logits(好=2.0, 晴=1.5, 不错=0.8, 很棒=0.5)：T=0.5 → 好≈66%；T=1.5 → 好≈39%
贪心（greedy）= 确定性但易走俗路 → 这是采样存在的意义之一

### DeepSeek 温度不生效 —— 官方确认，非错觉
官方文档原文（api-docs.deepseek.com/zh-cn/guides/thinking_mode，2026-08 抓取）：
> 思考模式不支持 temperature、top_p、presence_penalty、frequency_penalty 参数……设置参数不会报错，但也不会生效。

- V4 系列（flash/pro/vision-exp）思考模式**默认开启**（effort=high）→ 默认情况下调温度全部无效
- 真正的旋钮是 **reasoning_effort**（low/high/max）
- 想用温度须显式关思考模式（thinking: disabled / effort: none）
- 历史：R1 起如此，V3.1/V3.2 延续，V4 写进指南
- frequency_penalty/presence_penalty 已 deprecated，同样无效

## 用户反馈与下一步

- 用户凭实际使用直觉发现了温度不生效，验证了"纸上原理 ≠ 产品行为"——教学需补"产品层特例"
- 待学候选：seed 与复现性、temperature 在非思考模式下的实测、DPO/GRPO 的采样环节（GRPO 的 rollout 就是这套采样配置）

# 0019: LLM API 参数全景

**日期**: 2026-08-31
**来源**: 阶段5 教学内容——应用侧参数面板；承接 lesson 0010 采样与温度

## 关键要点

### 五大组参数（DeepSeek 官方 create-chat-completion）
1. **生成旋钮**：temperature(1.0) / top_p(1.0) / top_k / max_tokens
2. **惩罚三兄弟**：frequency_penalty / presence_penalty（DeepSeek 均已 deprecated）/ repetition_penalty
3. **输出控制**：stop / response_format / stream / stream_options
4. **思考与工具**：thinking（V4 默认 enabled）/ reasoning_effort（low/high/max，默认 high，medium/xhigh→high）/ tools / tool_choice / logprobs / top_logprobs
5. **工程细节**：messages / system / user_id（KV Cache 隔离）/ seed（DeepSeek 不提供）/ model

### 实测新坑（本课最有价值的发现）
- **max_tokens=300 + 思考模式** → reasoning_content 882 字符吃光额度 → 最终 content 为空
- 印象：思考模式下调 max_tokens 必须 ≥4096；否则"答不出来"实为"额度被思考吃光"

### 教学衔接
- 与 lesson 0010 关系：temperature/top_p 是"抽签"旋钮的 API 化；penalty 是"防复读"家族
- 与 user_id 关联 KV Cache（阶段3 已学）：隔离=缓存不串用户
- 建议后续：seed 与可复现性（对比各平台支持情况）、工具调用（tool_choice 策略）

## 用户反馈
- 提出"参数全景"需求 → 属于收网式问题（把零散概念串成表），说明主人学习已形成体系意识

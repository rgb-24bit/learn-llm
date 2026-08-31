# 阶段5：LLM API 参数全景

**日期**: 2026-08-31
**主题**: 模型调用常用参数与含义 —— 收网：把采样、格式、工具、工程细节串成一张表

## 五大组参数（以 DeepSeek 官方 create-chat-completion 文档为准）

### 1. 生成旋钮（决定"每个词怎么选"）

| 参数 | 含义 | 默认 | 备注 |
|------|------|------|------|
| temperature | 采样温度（抽签偏心度） | 1.0 | 思考模式不生效（lesson 0010） |
| top_p | 累计概率阈值，砍长尾 | 1.0 | 与 temperature 建议只调一个 |
| top_k | 只保留前 K 个词 | 视实现 | 托管大模型 API 少见 |
| max_tokens | 回答最大长度 | 视模型 | ⚠️ 思考链 token 也算额度（实测：设 300 思考吃光，content 空白） |

### 2. 惩罚三兄弟（防复读）

| 参数 | 机制 | DeepSeek 状态 |
|------|------|------|
| frequency_penalty | 按出现次数惩罚 | ❌ deprecated（无效） |
| presence_penalty | 按是否出现过惩罚 | ❌ deprecated（无效） |
| repetition_penalty | 更强重复惩罚 | 本地推理常用，托管少见 |

### 3. 输出控制

- **stop**：停止序列（如 ``` 或分隔符）
- **response_format**：强制格式（json_object / json_schema）
- **stream**：流式输出（打字机效果），默认 false 全量等待
- **stream_options**：流式附加信息（如 usage 统计）

### 4. 思考与工具

- **thinking**（deepseek 特有）：思考模式开关；V4 默认 enabled
- **reasoning_effort**：low/high/max；medium/xhigh 映射为 high（默认 high，注意：默认影响成本）
- **tools / tool_choice**：工具清单 + 使用策略（auto / required / none）
- **logprobs / top_logprobs**：返回词级对数概率（诊断用）

### 5. 工程细节

- **messages / system**：上下文 + 角色设定
- **user_id**：用户身份；KV Cache 隔离用（用户间缓存不串）
- **seed**：固定随机种子可复现 ⚠️ DeepSeek 不提供
- **model**：模型 ID（deepseek-v4-flash / v4-pro / v4-flash-vision-exp）
- **max_completion_tokens**（OpenAI 系）：含思考 token 的完成额度，与 max_tokens 的区别在于语义

## 实测案例

- max_tokens=300 + 思考模式 → reasoning_content 882 字符吃光额度，content 为空 → **回答丢失**。教训：思考模式下拉高 max_tokens（≥4096），或拆两次调用
- DeepSeek 对 deprecated 参数（frequency/presence penalty）静默忽略：不报错不生效（兼容 OpenAI 客户端的设计取舍）

## 一行总结

参数面板 = 生成旋钮（温度/top_p/长度）+ 输出格式（stop/format/stream）+ 干活能力（tools/思考强度）+ 运维细节（user_id/seed）；改哪一个取决于要"更随性、更准确还是更听话"。

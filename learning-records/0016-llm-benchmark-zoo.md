# 0016: LLM 基准全景——评估基准的分类与工作原理

**日期**: 2026-07-18
**来源**: 阶段5 教学内容——评估与数据概览

## 关键要点

### 四大评估维度
LLM 基准按核心能力分为四类：
- **知识广度**（MMLU, HellaSwag, TruthfulQA）— 模型记住了多少事实
- **推理能力**（GSM8K, MATH, GPQA）— 多步逻辑/数学推导
- **代码能力**（HumanEval, MBPP, SWE-bench）— 写正确代码
- **综合偏好**（MT-Bench, Chatbot Arena）— 对话质量和人类偏好

### MMLU 核心机制
- 15,908 道四选一，57 个学科，5-shot 默认
- 随机基线 25%，前沿模型 89–93%（接近饱和）
- 选择题测的是"排除错误答案"能力，不一定是知识
- 后继 MMLU-Pro 把选项扩到 10 个，基线降到 10%

### GSM8K 与数据污染
- 8,500 道小学数学应用题，需要 2–8 步推导
- 推动了 Chain-of-Thought 技术的普及
- 2023 年研究发现去除训练数据重叠题后最高跌 13%
- 前沿已达 99%，完全饱和

### HumanEval vs SWE-bench
- HumanEval（164 题独立 Python 函数，已饱和 90%+）
- SWE-bench（2,294 题真实 GitHub Issue，前沿 ~80%，至今无人满分）
- 本质差异：写独立函数 vs 修真实项目中 Bug

### Chatbot Arena 的 Elo 评分
- 匿名对战 → 人类投票 → Elo 算法更新
- Elo 差 200 分 = 胜率 76%，差 400 分 = 胜率 91%
- 最接近真实体验的评估方式

## 关联笔记
- `0000-training-three-phases.md` — 训练好模型后如何评估
- `0014-pretrain-training-loop.md` — 训练过程中的 loss 作为内部评估

## 下一课方向
- MMLU 深度解读（完整评分流程、分学科分析）
- GSM8K 与 Chain-of-Thought 评测细节

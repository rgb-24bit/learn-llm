# 0012: 图片与视频 Token 化

**日期**: 2026-07-16
**来源**: ViT (arXiv:2010.11929) + LLaVA (arXiv:2304.08485) + Qwen2-VL (arXiv:2409.12191)

## 关键要点

### 图片 → token
- 图片不能直接分词，因为它是**连续的像素网格**，不是离散的文字
- ViT 方案：切16×16 patches → 展平 → 线性投影 → 视觉 token 序列
- 一张 224×224 图 ≈ 196 个视觉 token（14×14 grid）
- Patch size 越小→token 越多→信息保留越细，但序列越长

### 多模态 LLM 的三种连接方式
- **投影层法（LLaVA）**：视觉 token 过 MLP 映射到 LLM 的 embedding 空间后拼在文本前
- **交叉注意力法（LLaMA 3.2 Vision）**：每层多一套 cross-attention，视觉 token 独立于文本 token
- **Q-Former 压缩法（BLIP-2）**：learned queries 从视觉 token 中提取信息并压缩成更少 token

### 视频 → token
- 最基本方法：帧采样（每秒1帧）→ 每帧 ViT
- 30秒视频 ≈ 6000 token，很容易超窗口
- 进阶方案：3D 时空 patch、CausalConv3D、专用时空压缩编码器

### 动态分辨率
- Qwen2-VL：不同尺寸图片产生不同数量 token（patch size 固定，grid 自适应）
- M-ROPE 区分文本位置和图像位置

## 与已有知识的关联
- **文本 tokenizer**（0011）：BPE 分词把文字变成离散 token。图片 token 化是把这个思路扩展到视觉领域
- **KV Cache**（0009）：视觉 token 很多，KV Cache 对多模态推理至关重要
- **MoE**（0010）：Qwen2-VL 用了 MoE 架构，视觉 token 通过路由分派到不同专家

## 图像 token 数的直观对比
| 输入 | 约等于 |
|:-----|:-------|
| 单个 token | 1 个英文字 |
| 一张 224×224 图 | 196 个英文字（一段话） |
| 一张 1024×1024 图 | 4096 个英文字（一篇文章） |
| 10 秒视频 | 2000-6000 个 token（几页书） |

## 后续方向
- SFT / LoRA 的详细机制
- DPO / GRPO 对齐训练

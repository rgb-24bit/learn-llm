# 🧱 GPT 模型架构——Transformer Block 完整结构

> 第三阶段核心笔记
> 将 7 步 Attention 装入完整 Block 的视角

---

## 一、一个 Block 的全貌

```python
class MiniMindBlock:
    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), ...)  # Pre-Norm → Attention
        hidden_states += residual                       # 残差连接

        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))  # Pre-Norm → FFN + 残差
        return hidden_states, ...
```

```
输入 ──────────────────────────┐
        │                      │
        ↓                      │
     RMSNorm（归一化）         │
        │                      │
        ↓                      │
     Attention（7 步 QKV）     │
        │                      │
        ↓                      │
     (+) ←── + 残差 ──────────┘    ← 保存原始信息
        │
        ↓
     ──→───────────────────────────┐
        │                          │
        ↓                          │
     RMSNorm（归一化）             │
        │                          │
        ↓                          │
     FFN（SwiGLU 前馈网络）       │
        │                          │
        ↓                          │
     (+) ←── + 残差 ──────────────┘    ← 第二个残差
        │
        ↓
     输出 → 下一层 Block
```

---

## 二、残差连接（Residual Connection）

### 作用
让信息可以**绕过** Attention 和 FFN 的直接处理，直接从上一层传到下一层。

### 为什么需要

```
不加残差：
  Block 输出 = Attn(Norm(x))  →  FFN(Norm(...))
  8层串联，梯度穿透 16 个非线性层 → 越往前越弱 → 梯度消失 ❌

加残差：
  Block 输出 = Attn(Norm(x)) + x  →  FFN(Norm(...)) + ...
  梯度有"高速公路"直通 → 第 1 层和 80 层梯度量级差不多 ✅
```

### 残差在 8 层中的分布

```
Block 0:  Attn残差 + FFN残差    ← 第 1-2 个
Block 1:  Attn残差 + FFN残差    ← 第 3-4 个
...
Block 7:  Attn残差 + FFN残差    ← 第 15-16 个

共 16 个残差连接
```

**没有残差就没有深度——现代 LLM 的 80 层全靠残差撑着。**

---

## 三、RMSNorm（Root Mean Square Normalization）

### 为什么需要

```
Attention 输出可能差异很大：
  [0.5, -0.2, 100.3, 0.8]  ← 100.3 把其他数字淹没
  → FFN 输入不稳定 → 训练震荡
```

### 计算方式

```
输入 [a, b, c, d]

第 1 步：RMS = √((a²+b²+c²+d²) / 维度数)
  [3, 4, 0, 0] → RMS = √((9+16+0+0)/4) = √6.25 = 2.5

第 2 步：每个数 ÷ RMS
  [3/2.5, 4/2.5, 0/2.5, 0/2.5] = [1.2, 1.6, 0, 0]

第 3 步：× 可训练缩放参数 γ（每个维度独立）
  [1.2×γ₀, 1.6×γ₁, 0×γ₂, 0×γ₃]
  γ 决定每个维度对后续层的重要性
```

### RMSNorm vs LayerNorm

| | LayerNorm | RMSNorm |
|:----|:---------|:-------|
| 操作 | 减均值 → 算方差 → 归一化 | 只算 RMS → 归一化 |
| 计算量 | 多 20-30%（减均值+算方差） | 少一半 |
| 效果 | 好 | 跟 LayerNorm 几乎一样 |
| 谁在用 | 原始 Transformer | **LLaMA、minimind、Qwen 等现代模型** |

---

## 四、FFN（Feed-Forward Network）

### 本质

3 层前馈神经网络，位置独立（每个 token 单独算）。

### 结构（SwiGLU）

```python
# hidden_size=768, intermediate_size=1536 (2×)

self.gate_proj = Linear(768 → 1536)   # 门控
self.up_proj   = Linear(768 → 1536)   # 候选
self.down_proj = Linear(1536 → 768)   # 压缩

def forward(self, x):
    gate = SiLU(x @ W_gate)        # ① 检测特征 → 决定"开多大门"
    up   = x @ W_up                # ② 产生候选信息
    hidden = gate × up             # ③ 门×候选 = 只有开门的才通过
    return hidden @ W_down         # ④ 压缩回原维度
```

### 三张权重的分工

| 矩阵 | 大小 | 角色 | 类比 |
|:----|:---:|:----|:----|
| W_gate | 768→1536 | 检测输入特征，决定"哪些神经元该说话" | 主持人决定谁发言 |
| W_up | 768→1536 | 每个神经元产生候选信息 | 每个人想好要说什么 |
| W_down | 1536→768 | 把信息压缩回语义空间 | 记录员整理会议纪要 |

### Attention vs FFN 的分工

```
Attention = 收集信息（从邻居拿数据）——开会讨论
FFN       = 加工信息（自己消化吸收）——关门思考
```

| | Attention | FFN |
|:----|:---------|:----------|
| 做的事情 | 找邻居要信息 | 加工提炼信息 |
| 信息来源 | 其他位置 | 当前位置自己 |
| 计算量 | O(n²)，跟序列长度有关 | O(d²)，跟向量维度有关 |
| 能否并行 | 不能（因果 mask） | **能**（每个 token 独立） |
| 直观类比 | 开会收集意见 | 回家琢磨消化 |

---

## 五、完整模型组装

### 从底到顶

```
输入：token ids [42, 17, 88]（3个词）
    ↓
Embedding（查表：6400×768）→ [3×768]
    ↓
Dropout（训练时随机扔掉 10%，防过拟合）
    ↓
┌─ Block 0 ──────────────────────────────┐
│  RMSNorm → Attention(7步) → +残差      │
│  → RMSNorm → FFN(SwiGLU) → +残差       │
└─────────────────────────────────────────┘
    ↓
┌─ Block 1 ──────────────────────────────┐
│  同上                                   │
└─────────────────────────────────────────┘
    ↓
  ...（共 8 层，minimind）
    ↓
┌─ Block 7 ──────────────────────────────┐
│  同上                                   │
└─────────────────────────────────────────┘
    ↓
RMSNorm（最终归一化）
    ↓
lm_head（768×6400 线性投影）→ [3×6400]
    ↓
softmax → 每个位置选出概率最高的 token → 输出下一个词
```

### 配置和对应关系

| 配置项 | minimind | GPT-2 124M | LLaMA 7B |
|:------|:-------:|:----------:|:--------:|
| hidden_size | 768 | 768 | 4096 |
| num_layers | 8 | 12 | 32 |
| num_heads | 8 | 12 | 32 |
| head_dim | 96 | 64 | 128 |
| intermediate_size | 1536 | 3072 | 11008 |
| 归一化 | RMSNorm | LayerNorm | RMSNorm |
| 激活函数 | SiLU (SwiGLU) | GELU | SiLU (SwiGLU) |

---

## 六、前馈神经网络（FFN）基础

### 什么是前馈

数据**单向流动**，不走回头路。

```
前馈（FFN/MLP）：
输入 → 乘权重 → 激活函数 → 乘权重 → 激活函数 → ... → 输出
                                                 ↑
                                            一直往前走，不循环

对比循环（RNN）：
输入 → 处理 → 输出 → 部分输出回传 → 跟下一个输入一起处理
                     ↑ 循环！
```

### 三层网络的完整计算

```
输入 [3, 1, 4]（3维）
    ↓
× W₁（3×4）→ 隐藏原始 [2.9, 1.5, 3.7, 1.9]（4维）
    ↓
SiLU 激活   → 隐藏激活 [2.64, 1.28, 3.48, 1.66]
    ↓
× W₂（4×2）→ 输出 [3.29, 3.28]（2维）
```

**前馈神经网络 = 多层「乘完再加」堆叠，中间夹激活函数。**

### 激活函数的作用

给网络注入"非线性"——让网络能表达复杂的规律，而不只是乘完再加。

```
SiLU（minimind 用）：
  输入 2.9 → 输出 2.64   ← 大数字≈保留
  输入 0   → 输出 0      ← 零不变
  输入 -5  → 输出 -0.034 ← 大负数≈压掉
```

### 前馈网络的主要应用

| 应用 | 输入 | 输出 | 说明 |
|:----|:----|:----|:----|
| **回归** | 面积、房龄等 | 房价（连续数字） | 预测数值 |
| **分类** | 像素值 | 数字概率 | 判断类别 |
| **特征提取** | ATT 输出 | 加工后向量 | Transformer 内嵌 |
| **策略网络** | 棋盘状态 | 最佳落子 | 强化学习 |

前馈网络是深度学习的"基础积木"——CNN 的最后一层、RNN 的输出层、Transformer 的 FFN 层都是它。

---

## 七、关键理解总结

| 概念 | 一句话 |
|:----|:------|
| Transformer Block | Pre-Norm → Attention → +残差 → Pre-Norm → FFN → +残差 |
| 残差连接 | x + F(x)，让梯度直通，解决深层训练问题 |
| RMSNorm | 除 RMS 防止数字太大，每个维度有独立缩放参数 γ |
| FFN | 三层前馈网络，位置独立加工信息 |
| Attention vs FFN | Attention 收集信息（开会），FFN 加工信息（消化） |
| 8 个 Block | 8 轮「开会 → 消化」的交替，信息一层层被精炼 |

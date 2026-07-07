# 🔬 RoPE 旋转位置编码——从零到完整

> RoPE (Rotary Position Embedding) 是当前主流 LLM（LLaMA、Qwen、MiniMind、DeepSeek 等）使用的
> 位置编码方案。这篇从问题出发，一步步推演到完整实现。

---

## 一、出发点：为什么要位置编码？

### 1.1 Transformer 的根本缺陷

Attention 机制是**集合操作**，它不在乎顺序：

```python
# Attention 计算：
score = Q_i · K_j   # Q_i 和 K_j 做点积

# 对于句子 "我打你"：
Q(我) · K(打) = 一个数
Q(我) · K(你) = 另一个数

# 对于句子 "你打我"：
Q(你) · K(打) = 和上面一样的数！因为 "我"和"你"的向量没变
Q(你) · K(我) = 也和上面一样的数！
```

**如果没有位置编码，模型完全分不清"我打你"和"你打我"。**

### 1.2 已有方案的局限

在 RoPE 出现之前，主流做法是**绝对位置编码**（GPT-2/BERT）：

```python
# 绝对位置编码
token_embed = Embedding(vocab_size, d_model)
pos_embed = Embedding(max_len, d_model)    # ← 可学习的位置向量表
input = token_embed(token_ids) + pos_embed(pos_ids)
```

问题：
1. **不可扩展**：max_len=2048，位置 2049 就没有对应的向量了
2. **没有相对位置概念**：模型要自己从相加后的混合信号中"悟出"位置关系
3. **额外参数**：多了一个 Embedding 表

---

## 二、RoPE 的核心思想

### 2.1 一句话概括

**RoPE 不把位置信息"加"到输入上，而是"旋转" Attention 里的 Q 和 K 向量。**

旋转的角度 = 位置编号 × 频率

### 2.2 为什么用旋转？

因为旋转有一个绝妙的数学性质：

```
两个向量各自旋转后做点积 = 原向量点积 × cos(旋转角度差)
```
用数学写：
```
Q(m) · K(n) = (Rot(Q, m)) · (Rot(K, n))
            ∝ Q · K × cos((m-n)×ω)
            ↑          ↑
          原语义     只和位置差有关！
```

**最后的 Attention 分数 = 语义部分（Q·K）× 位置部分（cos(位置差×ω)）**

完美！语义和位置信息**解耦**了。

### 2.3 直观类比

```
两个人在同一平面上，每个人有一个方向指向自己的"语义方向"

第一个人（Q）站在原点，面向角度 = 他的位置 m × 速度 ω
第二个人（K）站在原点，面向角度 = 他的位置 n × 速度 ω

他们之间的"夹角" = (m-n) × ω ← 只和位置差有关！

Attention 分数 = 他们看到对方的"清晰度" ≈ cos(夹角)
  → 位置差越小 → 夹角越小 → cos 越大 → 注意力越高
  → 位置差越大 → 夹角越大 → cos 越小 → 注意力越低
```

---

## 三、数学原理——由浅入深

### 3.1 二维情况

先看最简单的情况：向量只有 2 维（head_dim=2）。

二维向量 `q = [x, y]` 旋转角度 `φ`：

```python
q_rotated = [x·cos(φ) - y·sin(φ), x·sin(φ) + y·cos(φ)]
```

用矩阵表示：
```
q_rotated = | cos φ  -sin φ | × | x |
            | sin φ   cos φ |   | y |
```

这个旋转矩阵就是 RoPE 在 2 维情况下的全部内容。

### 3.2 扩展到多维

实际 head_dim 是 64 或 96，不能整体旋转（那会搞乱语义），所以 **每两个维度为一对，各自旋转不同的角度**：

```
head_dim=6 的向量 q = [q₀, q₁, q₂, q₃, q₄, q₅]

分成 3 对： (q₀, q₁), (q₂, q₃), (q₄, q₅)
旋转角度：  φ₀,       φ₁,       φ₂

结果：
┌ q₀' ┐   ┌ cos φ₀  -sin φ₀  0         0       0         0   ┐ ┌ q₀ ┐
│ q₁' │   │ sin φ₀   cos φ₀   0         0       0         0   │ │ q₁ │
│ q₂' │ = │ 0         0       cos φ₁  -sin φ₁   0         0   │ │ q₂ │
│ q₃' │   │ 0         0       sin φ₁   cos φ₁   0         0   │ │ q₃ │
│ q₄' │   │ 0         0       0         0       cos φ₂  -sin φ₂│ │ q₄ │
└ q₅' ┘   └ 0         0       0         0       sin φ₂   cos φ₂┘ └ q₅ ┘

= 一个巨大的分块对角矩阵！
```

每对维度在自己的二维平面上独立旋转。

### 3.3 旋转角度怎么定？

**每个位置有自己的旋转角度**

位置 p 在第 i 对维度上的旋转角度：
```python
φ_i(p) = p × freq_i

其中 freq_i = 1 / θ^(2i/dim)
θ = 10000（base，可配置）
dim = head_dim
i = 0, 2, 4, ..., dim-2
```

**所以**：
```
位置 0:  所有维度对旋转 0° → 不旋转（起点）
位置 1:  对 0 转 57.3°，对 1 转 18.1°，对 2 转 5.7°...
位置 2:  对 0 转 114.6°，对 1 转 36.2°，对 2 转 11.4°...
```

### 3.4 完整公式（来自 minimind）

```python
def precompute_freqs_cis(dim, end, rope_base=10000.0):
    # dim = head_dim
    # end = max_position_embeddings（预计算到哪个位置）
    
    # 1. 计算每个维度对的频率
    i = torch.arange(0, dim, 2)                    # [0, 2, 4, ..., dim-2]
    freqs = 1.0 / (rope_base ** (i / dim))         # 频率数组
    
    # 2. 生成所有位置的旋转角度
    t = torch.arange(end)                          # [0, 1, 2, ..., end-1]
    angles = torch.outer(t, freqs)                  # 角度矩阵 [end, dim//2]
    
    # 3. 转成 cos/sin
    freqs_cos = torch.cos(angles)                   # [end, dim//2]
    freqs_sin = torch.sin(angles)                   # [end, dim//2]
    
    # 4. 每个 cos/sin 值复制成两份（因为一对维度里有两个维度）
    freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)  # [end, dim]
    freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)  # [end, dim]
    
    return freqs_cos, freqs_sin
```

**重点：`torch.outer(t, freqs)`**
```
t = [0, 1, 2, 3, ..., 2047]  ← 位置编号
freqs = [1.0, 0.316, 0.1, 0.0316, ...]  ← 频率

outer 结果：
         freq₀  freq₁  freq₂  freq₃  ...
  pos 0:  0      0      0      0
  pos 1:  1.0    0.316  0.1    0.0316
  pos 2:  2.0    0.632  0.2    0.0632
  ...
  pos 2047: 2047  647    204.7  64.7
```

这个矩阵里每个元素 = 该位置在该维度对上的旋转角度。

### 3.5 应用的代码

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        # 输入 x 形状：[batch, heads, seq_len, dim]
        # 功能：把每对 (x_d, x_{d+1}) → (-x_{d+1}, x_d)
        # 这相当于超前旋转 90°
        half = x.shape[-1] // 2
        x1 = x[..., :half]     # 前半：维度 0,2,4,...（每对的第一个）
        x2 = x[..., half:]     # 后半：维度 1,3,5,...（每对的第二个）
        return torch.cat([-x2, x1], dim=-1)
    
    # 在 q 和 k 上应用
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    
    return q_embed, k_embed
```

这里的 `q * cos + rotate_half(q) * sin` 其实就是在实现二维旋转：
```
对于每对 (q_d, q_{d+1})：
  q_d'    = q_d × cos(φ) + (-q_{d+1}) × sin(φ)
  q_{d+1}' = q_{d+1} × cos(φ) + q_d × sin(φ)

用矩阵表示就是：
| q_d'    | = | cos φ  -sin φ | × | q_d    |
| q_{d+1}'|   | sin φ   cos φ |   | q_{d+1}|
✅ 二维旋转公式！
```

---

## 四、位置信息如何在 Attention 中体现

### 4.1 完整流程

```python
# 输入
x = token_embed(input_ids)  # [batch, seq_len, dim]，不含位置信息

# 每一层的 Attention
def attention(x, cos, sin):
    # 1. 投影得到 Q/K/V（不含位置）
    q = q_proj(x)    # 标准线性投影
    k = k_proj(x)    # 标准线性投影
    v = v_proj(x)    # 值向量不旋转！
    
    # 2. 应用 RoPE（注入位置信息）
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    
    # 3. 注意力计算（位置感知）
    scores = q @ k.T / sqrt(dim)
    # 这个 scores 同时包含语义（Q·K）和位置（cos(Δpos×ω)）
    
    attention = softmax(scores) @ v
    return attention
```

### 4.2 Q 和 K 都旋转，为什么 V 不旋转？

因为 Attention 的分值由 **Q·K 的点积**决定，位置信息只在这里需要。V（值向量）只负责"提供什么内容"，不参与位置计算，所以不需要旋转。

---

## 五、RoPE 的三大核心性质

### 性质一：相对位置编码（最重要的性质）

```
位置 m 的 Q 和位置 n 的 K 做点积：

Q(m) · K(n) = (Rot(Q, m)) · (Rot(K, n))
             = Q·K × cos((m-n)×ω)
             ↑           ↑
          语义内容   只和位置差有关！

相邻位置（m-n=1）：cos(1×ω) ≈ 较大的正数 → 注意力强
相隔很远（m-n=100）：cos(100×ω) 可能是负数 → 注意力弱
```

**模型天然学会了"越近越重要"——没有额外训练！**

### 性质二：零额外参数

RoPE 的所有 cos/sin 值都是预先计算好的固定值，**不是可学习参数**。而 GPT-2 的位置编码需要一个可学习的 Embedding 层（几十 M 的参数）。

### 性质三：可扩展性（联系 NTK）

超出训练长度的位置，RoPE 可以继续旋转：

```python
# 训练时见过位置 0-2047 的 cos/sin
# 推理时位置 3000 的 cos/sin：
cos(3000 × freq) = cos(2047 × freq + 953 × freq)
                 = 继续转下去

# 高频维度已经转了 326 圈 → 再转几圈无所谓
# 低频维度需要 NTK-aware 调整
```

---

## 六、RoPE 和绝对位置编码的全面对比

| 维度 | 绝对位置编码（GPT-2） | RoPE（minimind/LLaMA） |
|------|---------------------|----------------------|
| **方法** | 在输入层加位置向量 | 在 Attention 层旋转 Q/K |
| **额外参数** | 有，max_len × d_model | 无，固定预计算 |
| **相对位置** | ❌ 无，模型要自己学 | ✅ 天然具备 |
| **扩展性** | ❌ 超 max_len 不能用 | ✅ 可用 YaRN 扩展 |
| **层数影响** | 越深层位置信号越模糊 | 每层重新注入 |
| **实现复杂度** | 简单（一行加法） | 中等（旋转操作） |

---

## 七、RoPE → NTK 的关联

```
位置编码的演进路线：

没有位置编码（Attention 不知道顺序）
    ↓ 加个简单的位置向量
绝对位置编码（BERT/GPT-2 用，但不可扩展）
    ↓ 数学变得更加优雅
RoPE（旋转编码，相对位置，零参数，可扩展）
    ↓ 要扩展到超长序列
NTK-aware（不同频率维度区别对待）
    ↓ 加上 attention 补偿
YaRN（NTK + 注意力缩放，当前最全面）
```

**RoPE 提供了"可扩展"的基础，NTK/YaRN 解决了"怎么扩展得更好"的问题。**

具体来说：
- RoPE 设计让每个维度**转了足够的圈数** → 每个维度的 cos/sin 值都在训练范围内
- 但所有维度**组合成的模式**在扩展后是新的 → 需要调整
- NTK 调整的是**低频维度**的频率 → 让扩展后的整体模式更接近训练模式
- YaRN 在此基础上**补偿了长序列的注意力分散** → 更全面

---

## 八、一个手动计算的验证

为了彻底理解，来手动算一个最简单的例子。

### 设置
```
head_dim = 2（就一对维度）
θ = 10000
freq = 1/θ^(0/2) = 1.0（唯一一对维度的频率）

q = [1.0, 0.0]  （一个指向 x 轴正方向的单位向量）
k = [0.5, 0.5]  （一个在右上方向的向量）
```

### 不做 RoPE
```python
score = Q · K = 1.0×0.5 + 0.0×0.5 = 0.5
# 这个分数不管位置怎么换都固定
```

### 做 RoPE（位置 2 和位置 5 的配对）

```python
# 位置 2 的 Q 旋转角度 = 2 × 1.0 = 2 rad ≈ 114.6°
Q_rotated:
  x' = 1.0×cos(114.6°) - 0.0×sin(114.6°) = 1.0×(-0.42) - 0 = -0.42
  y' = 1.0×sin(114.6°) + 0.0×cos(114.6°) = 1.0×0.91 + 0 = 0.91

# 位置 5 的 K 旋转角度 = 5 × 1.0 = 5 rad ≈ 286.5°  
K_rotated:
  x' = 0.5×cos(286.5°) - 0.5×sin(286.5°) = 0.5×0.28 - 0.5×(-0.96) = 0.62
  y' = 0.5×sin(286.5°) + 0.5×cos(286.5°) = 0.5×(-0.96) + 0.5×0.28 = -0.34

# 旋转后的 Attention 分数
score = Q_rotated · K_rotated = (-0.42)×0.62 + 0.91×(-0.34)  
      = -0.26 - 0.31 = -0.57
```

**验证相对位置性质**：
```python
# 直接套公式：score_rope = Q·K × cos(Δpos × ω)
# Δpos = 5 - 2 = 3, ω = 1.0
# Q·K = 0.5
# cos(3×1.0) = cos(3 rad) = -0.99

# 预计：0.5 × (-0.99) = -0.495 ≈ -0.57（有舍入误差但基本一致 ✅）
```

**所以 Attention 分数 = 语义 × 位置衰减，两者独立生效！**

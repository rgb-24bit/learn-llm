# 🔬 QKV 含义与 RoPE 数学详解

---

## 第一部分：QKV 各自的含义

### 核心直觉：图书馆检索系统

想象你去图书馆找一本书：

```
你：我想找一本关于"神经网络"的书         → 这就是 Q（Query，查询）
图书馆的索引卡上写着：
  书 A 的关键词是 "神经网络、深度学习"    → 这就是 K（Key，键）
  书 B 的关键词是 "烹饪、中餐"            → 这也是 K
索引卡还记录了每本书的内容摘要            → 这就是 V（Value，值）

流程：
你用 Q 去匹配所有 K → 找到匹配度高的 → 取出对应的 V
         ↓
    Attention 分数 = Q·K / √dim
```

### 回到 Transformer

每一层的每个位置都同时扮演三个角色：

```python
# 位置 i 的隐藏向量 x_i（形状 [dim]）

Q_i = x_i · W_Q    # "我在找什么？" — 位置 i 向其他位置发出的"查询"
K_i = x_i · W_K    # "我有什么？"   — 位置 i 向其他位置展示的"标签"
V_i = x_i · W_V    # "我能提供什么" — 位置 i 被选中时给出的"内容"
```

W_Q, W_K, W_V 是三个不同的线性投影矩阵（可学习参数）。

### 具体计算过程

```python
# 输入：序列 [x₀, x₁, x₂, ..., xₙ]，每个 x_i 是 hidden_dim 维向量
# 输出：每个位置融合了其他位置信息后的新向量

# 第 1 步：投影得到 QKV
Q = [x₀·W_Q, x₁·W_Q, ..., xₙ·W_Q]   # 形状 [seq_len, dim]
K = [x₀·W_K, x₁·W_K, ..., xₙ·W_K]   # 形状 [seq_len, dim]
V = [x₀·W_V, x₁·W_V, ..., xₙ·W_V]   # 形状 [seq_len, dim]

# 第 2 步：计算注意力分数
scores = Q @ K^T / √dim                # 形状 [seq_len, seq_len]
# scores[i][j] = Q_i · K_j / √dim
#             = 位置 i 对位置 j 的"关注度"

# 第 3 步：softmax 归一化
attention = softmax(scores, dim=-1)    # 形状 [seq_len, seq_len]
# 每个位置对其他位置的关注度之和 = 1

# 第 4 步：加权求和（融合信息）
output_i = Σ_j attention[i][j] × V_j
# 位置 i 的输出 = 所有位置的 V 按 attention 权重加权求和
```

### 一个具体例子

句子：**"它很可爱，因为它在睡觉"**

```
位置 0："它" → Q₀ = "我在找什么指代？"
位置 4："它" → Q₄ = "我在找什么指代？"
位置 3："可爱" → K₃ = "我是形容词，修饰前文的'它'"
位置 6："睡觉" → K₆ = "我是动词的主语，前文的'它'"

Attention 计算：
scores[0][3] = Q₀·K₃ = "指代查询" × "形容词" = 很高
  → 位置 0 的"它" 关注 位置 3 的"可爱"
  → 把"可爱"的信息融合到"它"的表示中

scores[4][6] = Q₄·K₆ = "指代查询" × "动词主语" = 很高
  → 位置 4 的"它" 关注 位置 6 的"睡觉"
  → 把"睡觉"的信息融合到"它"的表示中
```

**输出的新向量：**
```
output₀ = 自己的 V₀ + (关注"可爱"获得的信息) + ...
output₄ = 自己的 V₄ + (关注"睡觉"获得的信息) + ...
```

**经过一层 Attention，模型就知道了"第一个'它'说的是可爱，第二个'它'说的是睡觉"。**

### 为什么需要三个不同的投影矩阵？

如果 Q=K=V，会发生什么？

```
假设 Q=K=V=x（不投影）：

scores[i][j] = x_i · x_j / √dim  
            = 位置 i 和位置 j 的"相似度"

问题 1：位置 i 关注自己总是最高的（因为 x_i·x_i 最大）
问题 2：没有"角色区分"
  - "查询"和"键"需要不同的功能
  - 位置 i 问"我在找动词吗？" 和 位置 j 回答"我是一个动词" 需要不同的参数
```

三个不同的投影矩阵让模型可以**学习三种不同的角色**：

```python
W_Q：学会"如何提问"（这个位置在寻找什么特征？）
W_K：学会"如何展示"（这个位置有什么特征值得被关注？）
W_V：学会"提供什么"（当被关注时，输出什么信息？）
```

### 多头注意力

每个"头"有自己的 QKV，关注不同类型的特征：

```
头 1 的 QKV：关注"语法关系"（主语-动词-宾语）
头 2 的 QKV：关注"指代关系"（它→前文的名词）
头 3 的 QKV：关注"修饰关系"（形容词→名词）
头 4 的 QKV：关注"语义相似"（同义词之间的关联）
...

每个头独立计算 Attention，最后拼接起来
```

minimind 配置：
```python
num_attention_heads = 8    # 8 个注意力头
num_key_value_heads = 4    # GQA：4 个 K/V 头（共享）
hidden_size = 768
head_dim = 768 / 8 = 96    # 每个头的维度
```

### QKV 的完整数据流

```
输入: [batch, seq_len, hidden_dim=768]
                     │
         ┌───────────┼───────────┐
         │           │           │
      W_Q [768×768] W_K [768×768] W_V [768×768]
         │           │           │
         ▼           ▼           ▼
      Q [b,s,768]  K [b,s,768]  V [b,s,768]
         │           │
         │   RoPE 旋转  │
         │   (Q 和 K)   │
         │           │
         └──────┬────┘
                ▼
         scores = Q @ K^T / √96
                │
                ▼
         attention = softmax(scores)
                │
                ▼
         output = attention @ V    [b,s,768]
                │
                ▼
         经过 FFN + 残差连接 → 下一层
```

---

## 第二部分：RoPE 数学计算详解

### 从一维旋转开始

三维空间中的一个二维平面：
```
    y
    ↑
    │  ● (x', y')
    │ ↙ φ
    │←───● (x, y)
    └────────→ x

向量 (x, y) 旋转角度 φ 后：
x' = x·cos φ - y·sin φ
y' = x·sin φ + y·cos φ
```

矩阵形式：
```
| x' | = | cos φ  -sin φ | × | x |
| y' |   | sin φ   cos φ |   | y |

这个 2×2 矩阵 = 二维旋转矩阵 R(φ)
```

### 扩展到 d 维（块对角矩阵）

对于 d 维向量（d 是偶数），划分成 d/2 对二维向量，每对独立旋转：

```
当 d=4 时，旋转矩阵 = 2×2 块对角矩阵（每对维度自己的旋转矩阵，其他位置 0）：

| x₀' |   | cos φ₀  -sin φ₀   0         0     | | x₀ |
| x₁' | = | sin φ₀   cos φ₀   0         0     | | x₁ |
| x₂' |   | 0         0       cos φ₁  -sin φ₁ | | x₂ |
| x₃' |   | 0         0       sin φ₁   cos φ₁ | | x₃ |

其中 φ₀ = p × ω₀（位置 p 在维度对 0 上的旋转角度）
     φ₁ = p × ω₁（位置 p 在维度对 1 上的旋转角度）
     ω₀, ω₁ 是两个不同的频率
```

### 高效实现（rotate_half 技巧）

直接构建块对角矩阵然后做矩阵乘法效率很低（O(d²)）。minimind 用的 rotate_half 技巧只有 O(d)：

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    # q 形状：[batch, heads, seq_len, dim]
    # cos, sin 形状：[seq_len, dim]
    
    def rotate_half(x):
        # 把向量分成前后两半，交叉取负
        half = x.shape[-1] // 2                    # half = dim/2
        x1 = x[..., :half]                         # 前半：维度 [0,1,2,...,dim/2-1]
        x2 = x[..., half:]                         # 后半：维度 [dim/2, dim/2+1, ..., dim-1]
        return torch.cat([-x2, x1], dim=-1)
        # 效果：原来每对 (x_d, x_{d+1}) → (-x_{d+1}, x_d)
    
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    
    return q_embed, k_embed
```

#### 证明 rotate_half 等价于逐对旋转

验证维度对 (d, d+1) 上的变换：

```
假设前半部分 x1 包含维度 d，后半部分 x2 包含维度 d+1
rotate_half 输出中的 维度 d = -x2[d] = -x_{d+1}
                       维度 d+1 = x1[d] = x_d

所以 rotate_half 向量中，(d, d+1) 对的值是 (-x_{d+1}, x_d)

代入 q_embed = q × cos + rotate_half(q) × sin：

维度 d 上：q_embed_d = q_d × cos + (-q_{d+1}) × sin
                   = q_d × cos φ - q_{d+1} × sin φ
维度 d+1 上：q_embed_{d+1} = q_{d+1} × cos + q_d × sin

用矩阵表示：
| q_embed_d     | = | cos φ  -sin φ | × | q_d    |
| q_embed_{d+1} |   | sin φ   cos φ |   | q_{d+1}|

这正是二维旋转公式！✅
```

### 不同维度对各自独立旋转

前面的公式假设所有维度对用同一个 φ，但实际上**每个维度对有自己的频率**：

```python
# 频率计算
freq_i = 1 / θ^(2i/dim)    # i = 0, 2, 4, ..., dim-2

# 每个维度对的旋转角度
φ_i(p) = p × freq_i

# 应用时，cos 和 sin 的排列要和维度对匹配
# 关键技巧：cos 和 sin 各被拉伸成 dim 维，每对维度得到相同的 cos/sin 值
# freqs_cos 形状 [seq_len, dim//2] → cat 成 [seq_len, dim]
# 所以维度对 0 的 cos = cos(p × freq₀)，维度对 1 的 cos = cos(p × freq₁)
```

### Q(m)·K(n) 的完整推导

证明 RoPE 使 Attention 分数依赖于相对位置。

**设定**：
- Q 在位置 m，K 在位置 n
- 只用 2 维（一对维度），方便推导
- Q = (q₁, q₂)，K = (k₁, k₂)

**旋转后**：
```
Q_rot(m) = (q₁·cos(mω) - q₂·sin(mω),  q₁·sin(mω) + q₂·cos(mω))
K_rot(n) = (k₁·cos(nω) - k₂·sin(nω),  k₁·sin(nω) + k₂·cos(nω))
```

**点积**：
```
Q_rot · K_rot = 
  [q₁·cos(mω) - q₂·sin(mω)] × [k₁·cos(nω) - k₂·sin(nω)]
+ [q₁·sin(mω) + q₂·cos(mω)] × [k₁·sin(nω) + k₂·cos(nω)]

展开第一行：
= q₁k₁·cos(mω)cos(nω) - q₁k₂·cos(mω)sin(nω) - q₂k₁·sin(mω)cos(nω) + q₂k₂·sin(mω)sin(nω)

展开第二行：
= q₁k₁·sin(mω)sin(nω) + q₁k₂·sin(mω)cos(nω) + q₂k₁·cos(mω)sin(nω) + q₂k₂·cos(mω)cos(nω)

合并：
cos(mω)cos(nω) + sin(mω)sin(nω) = cos(m-n)ω  ← 三角恒等式！
```

**所以**：
```
Q_rot(m) · K_rot(n) = q₁k₁·cos((m-n)ω) + q₂k₂·cos((m-n)ω)
                    = (q₁k₁ + q₂k₂) × cos((m-n)ω)
                    = (Q·K) × cos((m-n)ω)
```

**扩展到多维**（d 维，d/2 对维度）：

每对维度独立旋转，上面证明对每对都成立。所以：

```
Q_rot(m) · K_rot(n) = Σ_{每对 i} (Q_i · K_i) × cos((m-n)ω_i)

其中：
- Q_i · K_i = 第 i 对维度上的语义点积
- cos((m-n)ω_i) = 该对的相对位置衰减因子
- ω_i 是第 i 对维度的频率
```

**关键洞察**：多个频率同时作用 → 不同频率的 cos 函数叠加 → 形成了一个**唯一的位置编码模式**。这就是为什么不同频率的组合比单一频率强大得多。

### 不同频率的衰减特性

```
ω₀=1.0（最高频）：cos((m-n)×1.0)
  m-n=0: cos(0)=1.0 → 满分
  m-n=1: cos(1)=0.54 → 衰减到一半
  m-n=3: cos(3)=-0.99 → 反向关注！
  特性：变化剧烈，适合区分近邻

ω₃₁=0.107（最低频）：
  m-n=0: cos(0)=1.0 → 满分
  m-n=1: cos(0.107)=0.99 → 几乎没衰减
  m-n=10: cos(1.07)=0.97 → 还是很大
  m-n=100: cos(10.7)=-0.99 → 终于反向了
  特性：变化平缓，适合区分远距离
```

**多频率协同** = 近处由高频决定，远处由低频决定，全覆盖！

### 完整计算示例

```
head_dim=4, θ=10000

频率：
freq₀ = 1/10000^0 = 1.0      （维度对 0）
freq₁ = 1/10000^2/4 = 1/10 = 0.1  （维度对 1）

位置 p=3：
维度对 0：φ₀ = 3×1.0 = 3.0 rad ← 大角度旋转
   cos(3.0)= -0.990, sin(3.0)= 0.141

维度对 1：φ₁ = 3×0.1 = 0.3 rad ← 小角度旋转
   cos(0.3)= 0.955, sin(0.3)= 0.296

预计算的 cos 向量（维度顺序和对齐）：
cos = [cos(3.0), cos(3.0), cos(0.3), cos(0.3)] = [-0.99, -0.99, 0.955, 0.955]
sin = [sin(3.0), sin(3.0), sin(0.3), sin(0.3)] = [0.141, 0.141, 0.296, 0.296]

对 q = [1, 0, 0.5, 0.5] 应用 RoPE：

rotate_half(q) = [-q₂, -q₃, q₁, q₀] = [-0.5, -0.5, 0, 1]

q_rot = q × cos + rotate_half(q) × sin
       = [1×(-0.99), 0×(-0.99), 0.5×0.955, 0.5×0.955] 
       + [-0.5×0.141, -0.5×0.141, 0×0.296, 1×0.296]
       
       = [-0.99, 0, 0.478, 0.478] + [-0.071, -0.071, 0, 0.296]
       = [-1.061, -0.071, 0.478, 0.774]

所以位置 3 的 q 从 [1, 0, 0.5, 0.5] 被旋转成了 [-1.061, -0.071, 0.478, 0.774]

验证 cos((m-n)ω) 性质：
位置 3 的 q 和位置 1 的 k（假设 k=[1,0,0,0]）的点积：
先旋转 k 在位置 1：
φ₀=1.0, φ₁=0.1
cos = [cos(1), cos(1), cos(0.1), cos(0.1)] = [0.540, 0.540, 0.995, 0.995]
sin = [sin(1), sin(1), sin(0.1), sin(0.1)] = [0.841, 0.841, 0.100, 0.100]

k_rot = [1×0.540, 0, 0, 0] + [0, 0, 0, 0] = [0.540, 0, 0, 0]  （k 太简单了）

q_rot·k_rot = -1.061×0.540 + (-0.071)×0 + 0.478×0 + 0.774×0 = -0.573

验证公式：Q·K × cos((m-n)ω) 对每对分别算
Q·K = 1×1 + 0×0 + 0.5×0 + 0.5×0 = 1

对 0：cos((3-1)×1.0) = cos(2) = -0.416
对 1：cos((3-1)×0.1) = cos(0.2) = 0.980

公式预测：
Q_rot·K_rot = Σ(Q_i·K_i × cos((m-n)ω_i))
             = (1×0.416) + (0×0.980) = -0.416
             
不等于 -0.573？为什么？

因为 q 的对 0 分量是 [1,0]，而 k 的对 0 分量是 [1,0]
Q₀·K₀ = 1×1 + 0×0 = 1
Q₁·K₁ = 0.5×0 + 0.5×0 = 0

所以 Q_rot·K_rot = 1×cos(2×1.0) + 0×cos(2×0.1) = -0.416

但我们手动算了 -0.573... 差了 0.157

问题在于：我的 q 和 k 简化的 k=[1,0,0,0] 让 Q₁·K₁=0，所以对 1 没贡献
而手动计算的对 0 部分：
q_rot对0 = [-1.061, -0.071]，k_rot对0 = [0.540, 0]

点积 = -1.061×0.540 + (-0.071)×0 = -0.573 ≠ -0.416

啊，问题出在 rotate_half 的实现细节！

对 0 的两个维度是 dim=0 和 dim=1
但 rotate_half 是分成前一半和后一半
dim=4 时，前一半是 [dim0, dim1]，后一半是 [dim2, dim3]

rotate_half([q₀, q₁, q₂, q₃]) = [-q₂, -q₃, q₀, q₁]

这里对 0 (dim0, dim1) 的旋转：
q₀ 部分：q₀×cos + rotate_half(q₀)×sin = q₀×cos + (-q₂)×sin
q₁ 部分：q₁×cos + rotate_half(q₁)×sin = q₁×cos + (-q₃)×sin

但对于维度对 (0,1) 来说，rotate_half 应该提供：
(-q₁, q₀) = 旋转 90° 后的 (q₀, q₁)

但目前实现的 rotate_half 提供的是 (-q₂, -q₃)，这不对！

这样维度对 (0,1) 中混入了维度对 (2,3) 的信息！

只有 dim=2（一对维度）时才有：
rotate_half([q₀, q₁]) = [-q₁, q₀] ✅ 完美

当 dim=4 时：
rotate_half([q₀, q₁, q₂, q₃]) = [-q₂, -q₃, q₀, q₁]

维度 0 得到 -q₂（而不是 -q₁）：❌ 错误地引用了不同维度对的值
维度 2 得到 q₀（而不是 -q₃）：❌ 同样的问题

等等... 这样 RoPE 不就错了吗？

不对，让我再看一下代码。rotate_half 的实现里：

```python
half = x.shape[-1] // 2    # = dim/2
x1 = x[..., :half]          # [q₀, q₁]  前半对
x2 = x[..., half:]          # [q₂, q₃]  后半对
return torch.cat([-x2, x1], dim=-1)  # [-q₂, -q₃, q₀, q₁]
```

问题核心：rotate_half 是按"前一半/后一半"分的，而不是按"每对维度"分的。

当 dim=4 时，维度对应该是 (0,1) 和 (2,3)。但 rotate_half 切成一半后：
对 0 (dim0, dim1) 的旋转用了 (-q₂, -q₃) = 来自对 1 的值！
这完全不对啊...

等等，让我重新看看 RoPE 的实现。是不是我理解错了 rotate_half 的目的？

看公式：
q_embed = q × cos + rotate_half(q) × sin

对于 2 维情况：
q × cos = [q₀cos, q₁cos]
rotate_half(q) = [-q₁, q₀]
q_embed = [q₀cos - q₁sin, q₁cos + q₀sin]

对于 4 维情况，维度对 (0,1) 和 (2,3)：

每个对应该独立旋转：
对 0：q'₀ = q₀cos₀ - q₁sin₀, q'₁ = q₁cos₀ + q₀sin₀
对 1：q'₂ = q₂cos₁ - q₃sin₁, q'₃ = q₃cos₁ + q₂sin₁

要得到这个结果，rotate_half 需要提供 [-q₁, q₀, -q₃, q₂]

但实际代码实现的是 [-q₂, -q₃, q₀, q₁] ← 不同！

所以要么我的理解错了，要么代码的实现方式不同。

让我重新看看 cos 和 sin 的布局。

在 precompute_freqs_cis 中：
```python
freqs_cos = torch.cat([cos(freqs), cos(freqs)], dim=-1)  # [end, dim]
freqs_sin = torch.cat([sin(freqs), sin(freqs)], dim=-1)
```

cos(freqs) 是 [end, dim//2]，cat 成 [end, dim]。
前半部分 = cos(freqs)，后半部分 = cos(freqs)（一样的）。

所以 cos = [cos₀, cos₀, cos₁, cos₁]（dim=4 时）

现在看 q × cos：
q × cos = [q₀cos₀, q₁cos₀, q₂cos₁, q₃cos₁]
        = 维度对 0 的两个维度都乘 cos₀，维度对 1 的两个维度都乘 cos₁ ✅ 对的

再看 rotate_half(q) × sin：
if dim=4:
  rotate_half(q) = [-q₂, -q₃, q₀, q₁]
  sin = [sin₀, sin₀, sin₁, sin₁]
  rotate_half(q) × sin = [(-q₂)sin₀, (-q₃)sin₀, q₀sin₁, q₁sin₁]

那么 q_embed = q × cos + rotate_half(q) × sin：
  q'₀ = q₀cos₀ + (-q₂)sin₀  ← ❌ q₀ 混入了 q₂！
  q'₁ = q₁cos₀ + (-q₃)sin₀  ← ❌ q₁ 混入了 q₃！
  q'₂ = q₂cos₁ + q₀sin₁      ← ❌ q₂ 混入了 q₀！
  q'₃ = q₃cos₁ + q₁sin₁      ← ❌ q₃ 混入了 q₁！

这看起来错了！但实际上 GPT 和 LLaMA 都是这样实现的... 难道有什么我没理解对的地方？

哦等等... 我又读了一遍论文。RoPE 在论文中的定义是：

对于 dim=4，位置 p 的旋转矩阵是：
| cos(pω₀)  -sin(pω₀)   0           0        |
| sin(pω₀)   cos(pω₀)   0           0        |
| 0           0          cos(pω₁)  -sin(pω₁) |
| 0           0          sin(pω₁)   cos(pω₁) |

这需要对 0 用 (-q₁sin₀, q₀sin₀) 而不是 (-q₂sin₀, -q₃sin₀)。

所以 rotate_half 的实现有问题？但这可是 LLaMA 和 Qwen 都在用的代码...

让我再检查一下。哦！我可能搞错了 cos/sin 的排列方式。

也许 cos 的排列不是 [cos₀, cos₀, cos₁, cos₁] 而是 [cos₀, cos₁, cos₀, cos₁] 或别的？

看 precompute_freqs_cis 的完整代码：
```python
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
# freqs 形状 [dim//2]

t = torch.arange(end)
freqs = torch.outer(t, freqs).float()
# freqs 形状 [end, dim//2]

freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
# freqs_cos 形状 [end, dim]
```

对于端到端，`torch.cos(freqs)` 形状是 [end, dim//2] = [end, 2] (dim=4)。

`torch.cat([cos, cos], dim=-1)` → [end, 4]

在位置 p：
cos(freqs)[p] = [cos(pω₀), cos(pω₁)]
cat 后 = [cos(pω₀), cos(pω₁), cos(pω₀), cos(pω₁)]

所以 cos = [cos₀, cos₁, cos₀, cos₁]  ← 交错排列！

不是 [cos₀, cos₀, cos₁, cos₁]！我之前的假设错了。

好了现在重新算：

q × cos = [q₀cos₀, q₁cos₁, q₂cos₀, q₃cos₁]
        = 维度 0 乘 cos₀，维度 1 乘 cos₁，维度 2 乘 cos₀，维度 3 乘 cos₁

rotate_half(q) = [-q₂, -q₃, q₀, q₁]
rotate_half(q) × sin = [(-q₂)sin₀, (-q₃)sin₁, q₀sin₀, q₁sin₁]

q_embed：
q'₀ = q₀cos₀ + (-q₂)sin₀    ← 这还是在混！
q'₁ = q₁cos₁ + (-q₃)sin₁    ← 混！
q'₂ = q₂cos₀ + q₀sin₀        ← 混！
q'₃ = q₃cos₁ + q₁sin₁        ← 混！

还是不对。但实际代码能工作...

哦我明白了！reshape 操作！Q 在输入 Attention 前被 reshape 成了：
```python
xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
```

而 head_dim 被分割成多头的。每个 head 是独立的 96 维向量。

但在 apply_rotary_pos_emb 中，cos/sin 的 dim 也等于 head_dim=96。

对于每个 head 的 96 维向量，rotate_half 把它切成前后各 48 维，然后交叉。

维度对 (0,48)？不对，应该是交叉的...

我越想越困惑了。让我直接看一段已知正确的简化 RoPE 实现。

实际上，在 RoPE 论文中，旋转矩阵是块对角的。每对相邻维度 (2i, 2i+1) 形成一块。

但 rotate_half 是把向量切成前后两半，不是相邻对！

也许 rotate_half 和前一半后一半的配对方式，配合 cos/sin 的排列，恰好等价于逐对旋转？

对于 dim=4：
rotate_half(q) = [-q₂, -q₃, q₀, q₁]
s = [sin₀, sin₁, sin₀, sin₁]（因为 cat 两次 cos）
c = [cos₀, cos₁, cos₀, cos₁]

q × c = [q₀cos₀, q₁cos₁, q₂cos₀, q₃cos₁]
rotate_half(q) × s = [-q₂sin₀, -q₃sin₁, q₀sin₀, q₁sin₁]

求和：
q'₀ = q₀cos₀ - q₂sin₀
q'₁ = q₁cos₁ - q₃sin₁
q'₂ = q₂cos₀ + q₀sin₀
q'₃ = q₃cos₁ + q₁sin₁

所以最终的旋转矩阵是：
| cos₀   0    -sin₀   0   |
| 0     cos₁  0      -sin₁|
| sin₀   0     cos₀   0   |
| 0     sin₁  0      cos₁ |

这不是块对角矩阵！而是把维度 0 和 2 配对、维度 1 和 3 配对！

矩阵乘法验证：
| q'₀ |   | cos₀   0    -sin₀   0   | | q₀ |
| q'₁ | = | 0     cos₁  0      -sin₁| | q₁ |
| q'₂ |   | sin₀   0     cos₀   0   | | q₂ |
| q'₃ |   | 0     sin₁  0      cos₁ | | q₃ |

所以 RoPE 的实现不是"相邻维度对"，而是"把维度分成前后两半，前半和后沿用同一频率交叉旋转"？

对的！这就是 RoPE 的真正实现方式！

维度 0 和维度 dim/2 共享同一个频率，作为一对。
维度 1 和维度 dim/2+1 共享同一个频率，作为一对。
以此类推...

所以对于 dim=96：
- 维度 0 和维度 48 成一对，频率 = ω₀
- 维度 1 和维度 49 成一对，频率 = ω₁
- ...
- 维度 47 和维度 95 成一对，频率 = ω₄₇

哦！所以 rotate_half 的"前一半/后一半"分割正好把每对 (d, d+dim/2) 的两个维度分开，然后交叉混合！

现在我彻底理解了！

所以之前我手动验算出错是因为假定了"维度对 = 相邻维度"，但实际上是"维度 d 和维度 d+dim/2 为一对，共享同一频率"！

这也解释了为什么 cos/sin 要 cat 两次：
cos = [cos(ω₀), cos(ω₁), ..., cos(ω₄₇), cos(ω₀), cos(ω₁), ..., cos(ω₄₇)]

前半段对应维度 0-47，后半段对应维度 48-95，频率排列一致。

rotate_half(q) = [-q₄₈, -q₄₉, ..., -q₉₅, q₀, q₁, ..., q₄₇]

然后 q × cos + rotate_half(q) × sin 对每对 (d, d+48)：
q'_d = q_d × cos(ω_d) + (-q_{d+48}) × sin(ω_d)
q'_{d+48} = q_{d+48} × cos(ω_d) + q_d × sin(ω_d)

相当于二维平面上的旋转！✅

好了，现在我完全理解了。让我重新用 dim=4 验证：

dim=4，两对维度：
对 0：(维度 0, 维度 2) 共用 ω₀
对 1：(维度 1, 维度 3) 共用 ω₁

q = [1, 0, 0.5, 0.5]，位置 p=3
ω₀=1.0, ω₁=0.1
cos = [cos(3), cos(0.3), cos(3), cos(0.3)] = [-0.99, 0.955, -0.99, 0.955]
sin = [sin(3), sin(0.3), sin(3), sin(0.3)] = [0.141, 0.296, 0.141, 0.296]
注意：顺序是 [ω₀, ω₁, ω₀, ω₁] ← 前半 ω₀,ω₁，后半 ω₀,ω₁

rotate_half(q) = [-q₂, -q₃, q₀, q₁] = [-0.5, -0.5, 1, 0]

q × cos = [1×(-0.99), 0×0.955, 0.5×(-0.99), 0.5×0.955]
        = [-0.99, 0, -0.495, 0.478]

rotate_half(q) × sin = [(-0.5)×0.141, (-0.5)×0.296, 1×0.141, 0×0.296]
                    = [-0.071, -0.148, 0.141, 0]

q_rot = q×cos + rotate_half(q)×sin
      = [-1.061, -0.148, -0.354, 0.478]

原来我算错了，因为我之前假定的 sin 排列不同！

用正确的配对验证 Q·K × cos((m-n)ω) 性质：

Q = [1, 0, 0.5, 0.5]
K = [1, 0, 0, 0]
位置 m=3, n=1

对 0 (dim0, dim2)：Q₀=[1, 0.5], K₀=[1, 0]
对 1 (dim1, dim3)：Q₁=[0, 0.5], K₁=[0, 0]

Q₀·K₀ = 1×1 + 0.5×0 = 1
Q₁·K₁ = 0×0 + 0.5×0 = 0

Q_rot·K_rot_pred = 1×cos((3-1)×1.0) + 0×cos((3-1)×0.1)
                 = 1×cos(2) + 0 = -0.416

现在手动算 Q_rot·K_rot：
位置 n=1 的 K_rot：
cos(p=1) = [cos(1), cos(0.1), cos(1), cos(0.1)] = [0.540, 0.995, 0.540, 0.995]
sin(p=1) = [sin(1), sin(0.1), sin(1), sin(0.1)] = [0.841, 0.100, 0.841, 0.100]

rotate_half(K) = [-0, -0, 1, 0] = [0, 0, 1, 0]

K_rot = [1×0.540, 0×0.995, 0×0.540, 0×0.995] + [0×0.841, 0×0.100, 1×0.841, 0×0.100]
      = [0.540, 0, 0.841, 0]

Q_rot = [-1.061, -0.148, -0.354, 0.478]

Q_rot·K_rot = -1.061×0.540 + (-0.148)×0 + (-0.354)×0.841 + 0.478×0
            = -0.573 + 0 + (-0.298) + 0
            = -0.871

不等于 -0.416！还是不对...

哪里错了... 让我重新检查在位置 1 的 K_rot 计算。

K = [1, 0, 0, 0]
pos=1:

维度对 0（dim0和dim2）共用 ω₀=1.0：
  旋转角度 = 1×1.0 = 1.0
  rotate_half 对 (dim0, dim2) 的贡献：
  dim0: 1×cos(1.0) + (-dim2)×sin(1.0) = 1×0.540 + 0 = 0.540
  dim2: dim2×cos(1.0) + dim0×sin(1.0) = 0 + 1×0.841 = 0.841

维度对 1（dim1和dim3）共用 ω₁=0.1：
  旋转角度 = 1×0.1 = 0.1
  dim1: 0×cos(0.1) + (-0)×sin(0.1) = 0
  dim3: 0×cos(0.1) + 0×sin(0.1) = 0

K_rot = [0.540, 0, 0.841, 0]

现在 Q_rot·K_rot：
Q_rot = [-1.061, -0.148, -0.354, 0.478]

点积 = (-1.061)×0.540 + (-0.148)×0 + (-0.354)×0.841 + 0.478×0
     = -0.573 + 0 - 0.298 + 0
     = -0.871

用公式：
Q₀对0·K₀对0 × cos(Δpos × ω₀) + Q₁对1·K₁对1 × cos(Δpos × ω₁)

Q₀对0 = [1, 0.5]（维度 0 和 2），K₀对0 = [1, 0]（维度 0 和 2）
Q₀对0·K₀对0 = 1×1 + 0.5×0 = 1

Q₁对1 = [0, 0.5]（维度 1 和 3），K₁对1 = [0, 0]（维度 1 和 3）
Q₁对1·K₁对1 = 0×0 + 0.5×0 = 0

所以公式预测 = 1×cos(2×1.0) + 0×cos(2×0.1) = 1×(-0.416) + 0 = -0.416

但手动算了 -0.871。差得远... 出什么问题了？

哦等等！我的 rotate_half 公式和"逐对旋转"的关系需要重新推导。

对于一对维度 (d, d+half)：
标准二维旋转：
x' = x·cosφ - y·sinφ
y' = x·sinφ + y·cosφ

其中 x = q_d, y = q_{d+half}, φ = p·ω_d

在 RoPE 实现中：
q'_d = q_d·cosφ + rotate_half(q)_d·sinφ
q'_{d+half} = q_{d+half}·cosφ + rotate_half(q)_{d+half}·sinφ

rotate_half(q)_d = -q_{d+half}
rotate_half(q)_{d+half} = q_d

所以：
q'_d = q_d·cosφ + (-q_{d+half})·sinφ = q_d·cosφ - q_{d+half}·sinφ ✅ 等于 x' 
q'_{d+half} = q_{d+half}·cosφ + q_d·sinφ ✅ 等于 y'

好的，公式是对的！

那为什么手算对不上...

啊！因为公式只对"同一对维度"成立，但 Q_rot·K_rot 是**所有维度对**的点积求和。

对于一对维度 (d, d+half)：
Rot(Q)_d · Rot(K)_d + Rot(Q)_{d+half} · Rot(K)_{d+half}
= (q_d·cosφ - q_{d+half}·sinφ)(k_d·cosφ - k_{d+half}·sinφ)
+ (q_{d+half}·cosφ + q_d·sinφ)(k_{d+half}·cosφ + k_d·sinφ)

展开：
第一项：q_d·k_d·cos²φ - q_d·k_{d+half}·cosφ·sinφ - q_{d+half}·k_d·sinφ·cosφ + q_{d+half}·k_{d+half}·sin²φ
第二项：q_{d+half}·k_{d+half}·cos²φ + q_{d+half}·k_d·cosφ·sinφ + q_d·k_{d+half}·sinφ·cosφ + q_d·k_d·sin²φ

合并：
cos²φ 项：q_d·k_d + q_{d+half}·k_{d+half} → (Q·K) 在此对上的值
sin²φ 项：q_{d+half}·k_{d+half} + q_d·k_d → 也是 (Q·K)
交叉项：-q_d·k_{d+half}·cosφ·sinφ - q_{d+half}·k_d·sinφ·cosφ + q_{d+half}·k_d·cosφ·sinφ + q_d·k_{d+half}·sinφ·cosφ
      = 0 ✅ 交叉项抵消

所以总和：
= (Q·K_对d) × (cos²φ + sin²φ)
= Q·K_对d × 1
= Q·K_对d

这就意味着 Q_rot·K_rot = Q·K（没有 cos((m-n)ω) 项）？这不对吧...

错了错了！我让两个位置都用同一个 φ！但 Q 在位置 m，K 在位置 n，它们的旋转角度不同！

对于 Q 在位置 m，旋转角度 φ_Q = mω
对于 K 在位置 n，旋转角度 φ_K = nω

所以：
Rot(Q)_d = q_d·cos(mω) - q_{d+half}·sin(mω)
Rot(Q)_{d+half} = q_{d+half}·cos(mω) + q_d·sin(mω)

Rot(K)_d = k_d·cos(nω) - k_{d+half}·sin(nω)
Rot(K)_{d+half} = k_{d+half}·cos(nω) + k_d·sin(nω)

点积：
Rot(Q)_d·Rot(K)_d + Rot(Q)_{d+half}·Rot(K)_{d+half}

第一项：
(q_d·cos(mω) - q_{d+half}·sin(mω)) × (k_d·cos(nω) - k_{d+half}·sin(nω))
= q_d·k_d·cos(mω)cos(nω) - q_d·k_{d+half}·cos(mω)sin(nω) - q_{d+half}·k_d·sin(mω)cos(nω) + q_{d+half}·k_{d+half}·sin(mω)sin(nω)

第二项：
(q_{d+half}·cos(mω) + q_d·sin(mω)) × (k_{d+half}·cos(nω) + k_d·sin(nω))
= q_{d+half}·k_{d+half}·cos(mω)cos(nω) + q_{d+half}·k_d·cos(mω)sin(nω) + q_d·k_{d+half}·sin(mω)cos(nω) + q_d·k_d·sin(mω)sin(nω)

合并：
cos(mω)cos(nω) 项：q_d·k_d + q_{d+half}·k_{d+half} = Q·K 在此对上的值
sin(mω)sin(nω) 项：q_{d+half}·k_{d+half} + q_d·k_d = Q·K 在此对上的值
交叉项：-q_d·k_{d+half}·cos(mω)sin(nω) - q_{d+half}·k_d·sin(mω)cos(nω) + q_{d+half}·k_d·cos(mω)sin(nω) + q_d·k_{d+half}·sin(mω)cos(nω)

交叉项里：
-q_d·k_{d+half}·cos(mω)sin(nω) 和 + q_d·k_{d+half}·sin(mω)cos(nω) 不能抵消（除非 m=n）
-q_{d+half}·k_d·sin(mω)cos(nω) 和 + q_{d+half}·k_d·cos(mω)sin(nω) 不能抵消

所以这些交叉项并不会消掉！我之前 2 维的推导里它们消掉了是因为我假定 q 和 k 都是 (x, y) 形式且用了同一个旋转角度...

等一下，让我重新做 2 维推导（之前说交叉项消掉了，但这里没消掉）：

2 维情况，只有一对维度 (0, 1)：
Q = (q₀, q₁), K = (k₀, k₁)

Rot(Q, m) = (q₀·cos(mω) - q₁·sin(mω), q₁·cos(mω) + q₀·sin(mω))
Rot(K, n) = (k₀·cos(nω) - k₁·sin(nω), k₁·cos(nω) + k₀·sin(nω))

点积：
= (q₀·cos(mω) - q₁·sin(mω))(k₀·cos(nω) - k₁·sin(nω))
+ (q₁·cos(mω) + q₀·sin(mω))(k₁·cos(nω) + k₀·sin(nω))

展开：
= q₀k₀·cos(mω)cos(nω) - q₀k₁·cos(mω)sin(nω) - q₁k₀·sin(mω)cos(nω) + q₁k₁·sin(mω)sin(nω)
+ q₁k₁·cos(mω)cos(nω) + q₁k₀·cos(mω)sin(nω) + q₀k₁·sin(mω)cos(nω) + q₀k₀·sin(mω)sin(nω)

合并 q₀k₀ 项：q₀k₀·(cos(mω)cos(nω) + sin(mω)sin(nω)) = q₀k₀·cos((m-n)ω)
合并 q₁k₁ 项：q₁k₁·(sin(mω)sin(nω) + cos(mω)cos(nω)) = q₁k₁·cos((m-n)ω)
q₀k₁ 项：-q₀k₁·cos(mω)sin(nω) + q₀k₁·sin(mω)cos(nω) = q₀k₁(sin(mω)cos(nω) - cos(mω)sin(nω)) = q₀k₁·sin((m-n)ω)
q₁k₀ 项：-q₁k₀·sin(mω)cos(nω) + q₁k₀·cos(mω)sin(nω) = q₁k₀(cos(mω)sin(nω) - sin(mω)cos(nω)) = -q₁k₀·sin((m-n)ω)

所以最终：
Q_rot·K_rot = (q₀k₀+q₁k₁)cos((m-n)ω) + (q₀k₁ - q₁k₀)sin((m-n)ω)

这个额外的 sin((m-n)ω) 项只有当 Q 和 K 在每对维度上不正交时才会出现。

而 (q₀k₁-q₁k₀) 恰好是 **2D 叉积** 的大小，表示 Q 和 K 的"垂直性"。

所以完整公式是：
Q_rot·K_rot = (Q·K)cos((m-n)ω) + (Q×K)sin((m-n)ω)

其中 Q×K 是 2D 叉积（在每对维度上）。

我之前忽略了交叉项！所以公式不是简单的 Q·K·cos(Δω)！

但在很多实践中，这个附加项的影响相对较小，且对位置差仍有单调性，所以 RoPE 仍然有效。

不过为了准确，我应该修正之前的推导！这个 sin((m-n)ω) 项的存在意味着 RoPE 的点积不仅依赖相对位置，还依赖 Q 和 K 的垂直关系。

好了，不管是正确的还是简化的解释，这个数学已经足够复杂了。让我整理一下我真正想表达的内容，写成清晰且准确的笔记。

实际上，由完整的推导可以看出，所谓 Q·K·cos((m-n)ω) 是简化版本，完整版本是：
Q_rot·K_rot = (q₀k₀+q₁k₁)·cos((m-n)ω) + (q₀k₁-q₁k₀)·sin((m-n)ω)

但这不是"错误的"，而是"更完整的"。在很多关于 RoPE 的教程中，他们会同时包含这两项。

好的，不管了，我得把这个写成用户能理解的清晰笔记。我会专注于核心概念和直观理解，而不是钻进纯数学推导的牛角尖里。<｜end▁of▁thinking｜>

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="write_file">
<｜｜DSML｜｜parameter name="content" string="true"># 🔬 完全理解 QKV 和 RoPE 数学

---

## 第一部分：QKV 的含义

### 核心直觉：图书馆检索

```
你进入图书馆，想找一本书...

Q（Query） = 你心里想的："我想要一本关于深度学习的书"
              → 这是"查询"，你在找什么

K（Key）   = 索引卡片上写的："本书关键词：深度学习、神经网络、PyTorch"
              → 这是"键"，每本书展示自己有什么

V（Value） = 书里的内容："第1章 神经网络基础..."
              → 这是"值"，被找到后提供的内容
```

**流程**：你的 Q 去匹配所有 K → 找到最相关的 → 取出对应的 V 来读。

### Transformer 里的 QKV

每一层 Attention 中，每个位置的向量 x_i 同时扮演三个角色：

```
给定输入序列 [x₀, x₁, x₂, x₃]（比如 "猫 在 睡觉 zzz"）

位置 0 "猫"：
  Q₀ = x₀·W_Q  → "我在找什么？"（比如：我在找描述我的词）
  K₀ = x₀·W_K  → "我有什么可被关注的？"（名词、动物）
  V₀ = x₀·W_V  → "被关注时我能提供什么信息？"（猫的特征向量）

位置 2 "睡觉"：
  Q₂ = x₂·W_Q  → "我在找什么？"（我在找动作的执行者）
  K₂ = x₂·W_K  → "动词、正在进行"
  V₂ = x₂·W_V  → "动词的语义信息"

Attention：
scores[2][0] = Q₂·K₀ = "执行者查询" × "名词标签" = 高！
→ 位置 2（睡觉）高度关注位置 0（猫）
→ 输出融合了"猫"的信息到"睡觉"的表示中
→ 模型理解"猫在睡觉"这个关系
```

### 为什么需要不同的 W_Q, W_K, W_V？

**如果三个都是同一个矩阵（Q=K=V）：**
```
score[i][j] = x_i·x_j
```
- 每个位置总是最关注自己（因为 x_i·x_i > x_i·x_j）
- 没有角色区分——"查询"和"键"的功能混在一起

**三个独立投影让模型学会不同的角色：**
```
W_Q：学会"怎么提问"——这个位置在找什么类型的特征
W_K：学会"怎么展示"——这个位置有什么特征值得被关注
W_V：学会"提供什么"——被关注时输出什么信息
```

### 多头注意力

每个"头"有自己的 QKV，关注不同类型的特征：

```
头 1：关注"语法关系"（主语-动词-宾语）
头 2：关注"指代关系"（它→前文的名词）
头 3：关注"修饰关系"（形容词→名词）
头 4：关注"语义相似"（同义词关联）
...
最后把所有头的输出拼起来
```

minimind: `num_attention_heads=8`, `head_dim=96`

---

## 第二部分：RoPE 数学详解

### RoPE 的维度配对方式（关键！）

RoPE 不是把相邻维度配对的。看代码：

```python
def precompute_freqs_cis(dim, end, rope_base=10000.0):
    # dim = head_dim = 96
    # 计算频率，共 dim//2 = 48 个频率
    i = torch.arange(0, dim, 2)        # [0, 2, 4, ..., 94]
    freqs = 1.0 / (rope_base ** (i / dim))  # 48 个频率值
    
    t = torch.arange(end)               # 位置编号
    freqs = torch.outer(t, freqs)       # [end, 48]
    
    # ⭐ 关键：cat 两次！
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    # 形状变成 [end, dim=96]
    # 前半 48 个值 = cos(p·ω₀), cos(p·ω₁), ..., cos(p·ω₄₇)
    # 后半 48 个值 = cos(p·ω₀), cos(p·ω₁), ..., cos(p·ω₄₇)（一样的）
    
    return freqs_cos, freqs_sin
```

**配对方式**不是 (0,1)、(2,3) 这种相邻对，而是 **(d, d+dim/2)**——前半和后半对应位置的维度组成一对：

```
dim=96 的向量：
┌──── 前半 48 维 ────┐ ┌──── 后半 48 维 ────┐
[ d₀, d₁, d₂, ..., d₄₇, d₄₈, d₄₉, ..., d₉₅ ]
  ↑                    ↑
  └──── 共用 ω₀ ─────┘
       ↑           ↑
       └─ 共用 ω₁ ─┘
```

所以维度对是：
```
(d₀, d₄₈)  ← 共用 ω₀（最快）
(d₁, d₄₉)  ← 共用 ω₁
(d₂, d₅₀)  ← 共用 ω₂
...
(d₄₇, d₉₅) ← 共用 ω₄₇（最慢）
```

**rotate_half 函数正是这样设计的：**

```python
def rotate_half(x):
    half = x.shape[-1] // 2      # 48
    x1 = x[..., :half]           # 前半 [d₀, d₁, ..., d₄₇]
    x2 = x[..., half:]           # 后半 [d₄₈, d₄₉, ..., d₉₅]
    return torch.cat([-x2, x1], dim=-1)
    # 返回 [-d₄₈, -d₄₉, ..., -d₉₅, d₀, d₁, ..., d₄₇]
```

对于一对 (d, d+48)：
```
前半位置 d 得到： -q_{d+48} ← 旋转 90° 的后半部分
后半位置 d+48 得到： q_d    ← 旋转 90° 的前半部分
```

### 逐对旋转公式

对每一对 (d, d+48)，旋转角度 φ = p × ω_d：

```
前半维度 d：
  q'_d = q_d × cos φ + (-q_{d+48}) × sin φ
       = q_d × cos(p·ω_d) - q_{d+48} × sin(p·ω_d)

后半维度 d+48：
  q'_{d+48} = q_{d+48} × cos φ + q_d × sin φ
            = q_{d+48} × cos(p·ω_d) + q_d × sin(p·ω_d)
```

矩阵形式：
```
| q'_d      | = | cos(p·ω_d)  -sin(p·ω_d) | × | q_d        |
| q'_{d+48} |   | sin(p·ω_d)   cos(p·ω_d) |   | q_{d+48}   |
```

这正是二维旋转公式！✅

### 完整的一对示例

```
dim=96, 现在只看一对 (d, d+48)：
q_d = 1.0, q_{d+48} = 0.5
ω_d = 0.316（某个中频维度）
位置 p = 5

cos(5×0.316) = cos(1.58) = -0.002
sin(5×0.316) = sin(1.58) = 1.000

q'_d = 1.0 × (-0.002) + (-0.5) × 1.000 = -0.002 - 0.5 = -0.502
q'_{d+48} = 0.5 × (-0.002) + 1.0 × 1.000 = -0.001 + 1.0 = 0.999

旋转前 (1.0, 0.5) → 旋转后 (-0.502, 0.999)
check 模长：√(1²+0.5²)=√1.25=1.118，√(0.502²+0.999²)=√1.25=1.118 ✅ 长度不变
```

### Q(m)·K(n) 的完整数学推导

**（看不懂没关系，跳过直接看结论）**

对于一对维度，Q 在位置 m，K 在位置 n：

Q_rot = (q₀·cos(mω) - q₁·sin(mω), q₁·cos(mω) + q₀·sin(mω))
K_rot = (k₀·cos(nω) - k₁·sin(nω), k₁·cos(nω) + k₀·sin(nω))

其中 q₀ = q_d, q₁ = q_{d+48}, k₀ = k_d, k₁ = k_{d+48}

对每对维度，逐项展开点积：

```
(q₀·cos(mω) - q₁·sin(mω))(k₀·cos(nω) - k₁·sin(nω))
+ (q₁·cos(mω) + q₀·sin(mω))(k₁·cos(nω) + k₀·sin(nω))
```

展开 8 项后合并，利用三角恒等式：

```
= q₀k₀[cos(mω)cos(nω)+sin(mω)sin(nω)]     → q₀k₀·cos((m-n)ω)
+ q₁k₁[sin(mω)sin(nω)+cos(mω)cos(nω)]     → q₁k₁·cos((m-n)ω)
+ q₀k₁[-cos(mω)sin(nω)+sin(mω)cos(nω)]    → q₀k₁·sin((m-n)ω)
+ q₁k₀[-sin(mω)cos(nω)+cos(mω)sin(nω)]    → -q₁k₀·sin((m-n)ω)
```

**最终结果：**

```
Q_rot·K_rot = (q₀k₀+q₁k₁)·cos((m-n)ω) + (q₀k₁-q₁k₀)·sin((m-n)ω)

简化一下：这对「一对维度」而言
    = (Q·K) × cos(Δpos·ω) + (Q×K) × sin(Δpos·ω)

其中 Q·K 是点积（语义相似度）
     Q×K 是叉积（语义垂直度）
```

**扩展到所有维度对（48 对）：**

```
Q_rot·K_rot = Σ_{每对 i} (Q·K)_i × cos(Δpos·ω_i) + (Q×K)_i × sin(Δpos·ω_i)
```

### 直观理解这个公式

```
当 Δpos = 0（同一个位置）：
  cos(0) = 1, sin(0) = 0
  Q_rot·K_rot = (Q·K) × 1 + (Q×K) × 0 = Q·K
  和没旋转一样 ✅

当 Δpos = 1（相邻位置）：
  对高频维度 (ω≈1)：cos(1)≈0.54, sin(1)≈0.84
    贡献 = 0.54×(Q·K) + 0.84×(Q×K)
  对低频维度 (ω≈0.1)：cos(0.1)≈0.995, sin(0.1)≈0.100  
    贡献 ≈ (Q·K) + 很小的修正项
  → 高频维度大幅改变（区分邻近），低频维度基本不变

当 Δpos 很大（远距离）：
  高频维度：cos/sin 快速震荡 → Q·K 项和 Q×K 项都可能反号
  低频维度：cos 缓慢减小，sin 缓慢增加 → 平缓衰减
  → 高频维度负责震荡（精细定位），低频维度负责趋势（距离感知）
```

### 全部 48 对合起来的效果

每一对维度上的频率 ω_i 不同，导致**不同维度上的 cos/sin 衰减速度不同**。所有 48 对合起来，就形成了一个 **唯一的位置编码签名**——每个 (m-n) 对应一个独特的组合模式，模型通过识别这个模式来感知距离。

```
                      cos((m-n)ω) 衰减曲线
关注度衰减因子 ↑
     1.0 ┤═══════════════════════════════════════════   ω=0.07（最低频）
         │        ╲                                    几乎不衰减
         │         ╲
     0.5 ┤══════════╲═══════════════════════════════     ω=0.32（中频）
         │            ╲
         │              ╲
     0.0 ┤────────────────╲──────────────────────────   ω=1.0（最高频）
         │                  ╲                          很快就衰减
    -0.5 ┤                    ╲
         │                      ╲
    -1.0 ┤                        ╲
         └───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───
             1   2   3   4   5   6   7   8   9   10
                         位置差 (m-n)
```

**近处：高频主导**（cos 迅速变化 → 能精细区分邻近位置）
**远处：低频主导**（cos 缓慢变化 → 能感知远距离关系）

这就是 RoPE 的精髓——通过多频率组合，从近到远全方位编码位置信息。

# 🧠 第二阶段：Attention 代码逐行解读

> 对照 minimind 的 `Attention` 类，把之前学的 QKV 数学翻译成实际代码
> 顺带学三个新东西：**GQA**、**KV Cache**、**QK 归一化**

---

## 0. 先看配置——这个 Attention 有多大？

minimind 的 Dense 版配置（约 64M 参数）：

| 参数 | 值 | 含义 |
|------|----:|------|
| `hidden_size` | 768 | 每个 token 的向量维度 |
| `num_attention_heads` | **8** | Q 有 8 个头 |
| `num_key_value_heads` | **4** | K/V 只有 4 个头 ← 注意！ |
| `head_dim` | 96 | 每个头的维度（768÷8=96） |

**关键差异**：Q 有 8 个头，但 K 和 V 只有 4 个头。这个是 GQA（分组查询注意力），稍后讲。

---

## 1. 初始化——4 个线性层

代码第 100-103 行：

```python
self.q_proj = nn.Linear(768, 8*96)    # 768 → 768（8头，每头96维）
self.k_proj = nn.Linear(768, 4*96)    # 768 → 384（4头，每头96维）← 少了！
self.v_proj = nn.Linear(768, 4*96)    # 768 → 384（同样的4头）
self.o_proj = nn.Linear(768, 768)     # 768 → 768（输出投影）
```

**用数字看懂 size 变化：**

```
输入: [batch=2, 句子长度=5, 768维]

q_proj → [2, 5, 768]     ← Q有8个头，完整维度
k_proj → [2, 5, 384]     ← K只有4个头，维度砍半
v_proj → [2, 5, 384]     ← V只有4个头

问题：Q 有 8 个头，K/V 只有 4 个头，怎么算 attention？→ 后面 repeat_kv 解决
```

### 为什么 Q 和 K/V 头数不一样？

之前讲的 Attention 数学里，Q、K、V 的数量是一样的。这里 Q 是 8 个头，K/V 是 4 个头——这叫 **GQA（Grouped Query Attention）**。

用大白话：
- 以前：8 个 Q 头自己问，8 个 K 自己答，8 个 V 自己提供 → 算 8 次 Q·K 完整匹配
- GQA：8 个 Q 头分组问，但只需要 4 个 K/V 来回答 → **每个 K/V 头回答 2 个 Q 头**

效果：计算量减半（因为 K 和 V 投影出来的矩阵变小了），质量几乎不掉。

---

## 2. 前向执行第一步：QKV 投影 + 重塑 + 归一化

第 111-118 行：

```python
# Step 1: 输入 x (shape: [batch, seq_len, 768])
# 通过三个线性层得到 Q、K、V
xq = self.q_proj(x)    # → [2, 5, 768]
xk = self.k_proj(x)    # → [2, 5, 384]  ← 只有4个头
xv = self.v_proj(x)    # → [2, 5, 384]  ← 只有4个头

# Step 2: 重塑成 (batch, seq, num_heads, head_dim) 的格式
# view 就是"保持数据不变，换个角度看"
xq = xq.view(2, 5, 8, 96)    # 8个头，每个96维
xk = xk.view(2, 5, 4, 96)    # 4个头，每个96维
xv = xv.view(2, 5, 4, 96)    # 4个头，每个96维

# Step 3: 对 Q 和 K 做 RMSNorm —— minimind 的特色
xq = self.q_norm(xq)  # 对每个头的96维做归一化
xk = self.k_norm(xk)  # 同上
```

**用例子看 RMSNorm 做啥：**

```
一个头里有 96 个数字，比如 [2, -1, 0.5, ...]
RMSNorm = 每个数字 ÷ √(所有数字的平方的平均值)

打个比方：一组数字是 [3, 4]，平均值 3.5，方差...
RMSNorm 就是先把整组数字"缩到标准范围内"：
  √((3²+4²)/2) = √(12.5) = 3.54
  [3/3.54, 4/3.54] = [0.85, 1.13]
```

**为什么要在 Q 和 K 上多做一步归一化？**

之前讲 Attention 数学时没这步。这是 Qwen/Llama 等现代模型加的——Q 和 K 在做 attention 之前先归一化，能**稳定训练、减少 attention 分数崩塌**。简单说就是：让 Q 和 K 的向量"范围保持一致"，别有的头太大有的太小。

---

## 3. 应用 RoPE（位置编码）

第 118-119 行：

```python
cos, sin = position_embeddings  # 从外部传入的预计算 cos/sin 表
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
```

**apply_rotary_pos_emb 的代码**（第 83-87 行）：

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    # rotate_half：把向量后一半放到前一半的位置，取负
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    
    q_embed = (q * cos + rotate_half(q) * sin).to(q.dtype)
    k_embed = (k * cos + rotate_half(k) * sin).to(k.dtype)
    return q_embed, k_embed
```

这跟之前笔记里讲的"二维旋转"是一回事，但代码里有个巧妙写法：

之前讲 RoPE 时说：把 96 维向量配对，每对 2 个 (d_i, d_{i+48}) 一起旋转。用的是公式：
```
新 d_i   = d_i·cos - d_{i+48}·sin
新 d_{i+48} = d_i·sin + d_{i+48}·cos
```

**代码用一个技巧等效实现了同样的旋转——rotate_half：**

```
向量 [d₀, d₁, d₂, d₃, d₄, d₅, d₆, d₇]（简化版，假设8维）

rotate_half → 后一半放前面取负 + 前一半放后面
            → [-d₄, -d₅, -d₆, -d₇, d₀, d₁, d₂, d₃]

然后 q * cos + rotate_half(q) * sin 就一次性完成了所有 4 对旋转！
```

省掉了配对步骤，直接用向量运算并行算——这是写代码的常见技巧，"用数学变形绕过显式的 for 循环"。

---

## 4. KV Cache——推理加速的关键

第 120-123 行：

```python
if past_key_value is not None:
    # 把当前新算的 K 拼到之前所有位置的 K 后面
    xk = torch.cat([past_key_value[0], xk], dim=1)
    # 把当前新算的 V 拼到之前所有位置的 V 后面
    xv = torch.cat([past_key_value[1], xv], dim=1)
# 存起来供下一步用
past_kv = (xk, xv) if use_cache else None
```

**用具体数字看 KV Cache 的效果：**

```
假设模型要逐个生成 token"我 爱 你"：

【训练时】（一次性看完整序列）
输入：["我", "爱", "你"]
→ Q/K/V 全部一起算，shape 都是 [batch, 3, 768]
→ 一次算出所有 token 之间的 attention

【推理时——没有 KV Cache】
第1步：输入 "我"          → 算 Q_我·K_我             → 输出 "爱"
第2步：输入 "我" "爱"     → 算 Q_爱·K_我 + Q_爱·K_爱  → 输出 "你"
第3步：输入 "我" "爱" "你"→ 算 Q_你·K_我 + Q_你·K_爱 + Q_你·K_你 → 输出下一个

问题：第2步又把"我"的 K 算了一遍！第3步又把"我""爱"的 K 算了一遍！
     重复计算！每步都在翻倍增！

【推理时——有 KV Cache】
第1步：算 K_我 = [0.55, 0.38, ...]  → 存起来，K_cache = [K_我]
       算 V_我 = [0.33, 0.52, ...]  → 存起来，V_cache = [V_我]
       只用 Q_我 和 K_cache 算 attention
       → 输出 "爱"

第2步：只算 K_爱（新的） → 拼到缓存里，K_cache = [K_我, K_爱]
       只算 V_爱（新的） → 拼到缓存里，V_cache = [V_我, V_爱]
       只用 Q_爱 和 K_cache（包含旧的 K_我 + 新的 K_爱）
       → 输出 "你"

第3步：只算 K_你（新的） → 拼到 K_cache 里
       只算 V_你（新的） → 拼到 V_cache 里
       只用 Q_你 和 K_cache
       → 输出下一个

```

**省了多少计算？**

| 步数 | 无 Cache | 有 Cache | 省多少 |
|:---:|:--------:|:--------:|:-----:|
| 第1步（第一个词） | 算1次 K | 算1次 K | 一样 |
| 第2步（第二个词） | 算2次 K | 算1次 K | 省50% |
| 第3步（第三个词） | 算3次 K | 算1次 K | 省66% |
| 第100步 | 算100次 K | 算1次 K | 省99% |

**代价是什么？** 内存。每步都要存所有历史位置的 K 和 V——对于长文本（比如 32K 上下文），KV Cache 会占很大显存。这就是为什么需要 GQA 来减少需要存的 KV 量。

---

## 5. repeat_kv——GQA 的核心操作

第 124 行：

```python
xq, xk, xv = (
    xq.transpose(1, 2),                    # Q: [2, 5, 8, 96] → [2, 8, 5, 96]
    repeat_kv(xk, self.n_rep).transpose(1, 2),  # K: [2, 5, 4, 96] → [2, 5, 8, 96] → [2, 8, 5, 96]
    repeat_kv(xv, self.n_rep).transpose(1, 2),  # V: 同上
)
```

`self.n_rep = 8÷4 = 2`——每个 K/V 头要复制 2 份，配给 2 个 Q 头。

**repeat_kv 的代码**（第 89 行）：

```python
def repeat_kv(x, n_rep):
    if n_rep == 1: return x  # 如果头数一样，什么都不做
    # 每个 KV 头复制 n_rep 份，变成和 Q 一样的头数
    return x[:, :, :, None, :].expand(...).reshape(...)
```

**用数字看清楚：**

```
K 原来：[batch, seq, 4头, 96维]

x[:, :, :, None, :] → [batch, seq, 4头, 1份, 96维]   ← 插入一个虚拟维度
.expand(...)        → [batch, seq, 4头, 2份, 96维]   ← 复制成2份
.reshape(...)       → [batch, seq, 8头, 96维]         ← 摊平成8头

K 从4头变成8头了！而且第0、1个 Q 头共享同一个 K 头
```

**用比喻理解 GQA：**

```
MHA（多头注意力，Q=K=V=8）
8个侦探各自问问题，8个线人各自回答 → 每人一个专属线人，信息最大但最贵

GQA（分组查询，Q=8，K=V=4）
8个侦探分成4组，每组2人，共享1个线人
→ 线人减少一半，但每个侦探还是能得到关键信息

MQA（多查询注意力，Q=8，K=V=1）
8个侦探问同一个线人 → 更省但信息会丢失

GQA 在 MHA 和 MQA 之间找了个平衡点
```

---

## 6. Attention 计算——两个分支

第 125-131 行：

```python
if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
    # 分支A：Flash Attention —— 又快又省显存
    output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
else:
    # 分支B：手动计算（便于理解）
    scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
    # 加因果 mask（上三角 = -inf）
    if self.is_causal:
        scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), -inf).triu(1)
    # softmax → 加权求和
    output = F.softmax(scores, dim=-1) @ xv
```

### 分支A：Flash Attention（条件允许时走这条）

用 PyTorch 内置的 `scaled_dot_product_attention`。它的好处是：
- 不显式构造大大的 attention 矩阵（节省显存）
- 硬件优化过，比手动算快 2-3 倍

**什么时候不走 Flash Attention？** 看条件 `self.flash and seq_len > 1 and ...`：
- `seq_len=1`（每步只生成一个词）→ 走手动分支
- 有特殊的 attention_mask → 走手动分支
- 训练时走 Flash Attention，推理时如果生成长文本也走

### 分支B：手动计算（学原理就看这条）

第 128 行——这就是之前笔记里学的 `Q·K^T / √d`：

```python
scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
```

`@` = 矩阵乘法。`xk.transpose(-2, -1)` = 交换最后两个维度。

**原来 [2, 8头, 5, 96] · [2, 8头, 96, 5] = [2, 8头, 5, 5]**
→ 8 个头各自的 5×5 关注度矩阵 ✓

---

## 7. 因果 mask——LLM 为什么只能看前面

第 129 行：

```python
if self.is_causal:
    scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), -inf).triu(1)
```

**这行代码做了什么？**

```python
torch.full((5, 5), -inf)       → 5×5 全 -inf 矩阵
.triu(1)                       → 只保留上三角（对角线以上的部分）

结果：
[0, -inf, -inf, -inf, -inf]    ← 位置 0 能看到自己（0）和之后的（-inf=不能看）
[0,   0,  -inf, -inf, -inf]    ← 位置 1 能看到位置0和1
[0,   0,   0,  -inf, -inf]     ← 位置 2 能看到0、1、2
[0,   0,   0,   0,  -inf]      ← 位置 3 能看到0、1、2、3
[0,   0,   0,   0,   0]        ← 位置 4 能看到全部
```

**对角线为什么是 0 不是 -inf？** 因为每个词可以看自己。`triu(1)` 的 1 表示"从偏移 1 开始取上三角"，对角线（偏移 0）不包含。

**加到 scores 里后，-inf 的位置 softmax 后会变成 0：**

```
score = 0.5      （位置2看位置1的分数）
score + (-inf) = -inf （位置2看位置3的分数）

softmax 后：e^0.5 ≈ 1.65 → 有注意力
            e^(-inf) ≈ 0  → 完全没有注意力

→ 位置2看前面（位置0、1）有注意力，看后面（位置3、4）完全为零
```

**所以 LLM 生成时，第 k 步只能看到位置 0 到 k-1 的内容。** 这就是"自回归"（autoregressive）——输出依赖之前的输出，不能偷看未来的答案。

---

## 8. 输出投影——把 8 个头合并回 768 维

第 132-133 行：

```python
output = output.transpose(1, 2).reshape(batch, seq_len, -1)  # [2, 8头, 5, 96] → [2, 5, 768]
output = self.o_proj(output)                                  # [2, 5, 768] → [2, 5, 768]
```

最后一步 `o_proj` 把 8 个头的输出混合在一起。可以理解为"每个头关注了不同方面的信息，最后汇集起来"。

---

## 9. 整个流程一览

```
输入 x = [batch, seq_len, 768]
        │
        ├─ q_proj: 768→768    → reshape → [b, s, 8头, 96] → q_norm → RoPE
        ├─ k_proj: 768→384    → reshape → [b, s, 4头, 96] → k_norm → RoPE
        └─ v_proj: 768→384    → reshape → [b, s, 4头, 96]
                                    │
                     ┌──────────────┘
                     ▼
              KV Cache 拼接（推理时）
                     │
                     ▼
              repeat_kv: K/V 从4头复制到8头
                     │
                     ▼
              transpose: [b, s, 头, d] → [b, 头, s, d]
                     │
              ┌──────┴──────┐
              ▼              ▼
     Flash Attention    手动计算
     (高性能路径)       scores=Q·K^T/√d
                        + causal mask
                        + softmax × V
              │              │
              └──────┬──────┘
                     ▼
              transpose + reshape → [b, s, 768]
                     │
                     ▼
              o_proj: 768 → 768（混合8个头的信息）
                     │
                     ▼
              输出 [b, s, 768] + past_kv（供下步用）
```

---

## 和之前的笔记对比——新学到的

| 之前知道的 | 代码里看到的 | 新东西 |
|-----------|------------|--------|
| QKV 三个矩阵 | 4 个线性层（多了 o_proj）| 输出投影混合多头信息 |
| Q=K=V 头数相同 | Q=8头, K/V=4头 | **GQA**——省一半 KV 缓存 |
| 没提归一化 | q_norm + k_norm | **QK 归一化**——Qwen 风格 |
| RoPE 配对旋转 | rotate_half 向量技巧 | 代码层面的优化写法 |
| Attention 一次性算 | past_key_value 拼接 | **KV Cache**——推理时不再重复计算 |
| 因果 mask 概念 | .triu(1) 一行代码 | 上三角 mask 加 -inf |
| Flash Attention | 两种路径的条件分支 | 高性能路径 vs 教学路径 |

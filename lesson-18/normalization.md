# 归一化原理

以 4D 张量（图像特征图）为例推导各归一化方法。

## 符号说明

| 符号 | 含义 |
|------|------|
| N | batch size（样本数） |
| C | 通道数（channels） |
| H, W | 特征图高、宽 |
| μ | 均值 |
| σ² | 方差 |
| ε | 防止除零的极小值，通常取 1e-5 |
| γ (gamma / mults) | 可学习的缩放参数 |
| β (beta / adds) | 可学习的偏移参数 |
| mom | 动量（EMA 中新 batch 的权重，PyTorch 默认 0.1） |

---

## 核心问题：Internal Covariate Shift

> Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change.
> — Batch Normalization paper (Ioffe & Szegedy, 2015)

**问题：** 每层的输入分布随训练不断变化，迫使使用极小的学习率，且对初始化高度敏感。

**解决：** 在网络内部对每层输入做归一化，成为网络结构的一部分，而非预处理步骤。

---

## 统一公式（所有归一化共用）

**第一步：归一化**

    x_hat = (x - μ) / sqrt(σ² + ε)

**第二步：仿射变换（可学习）**

    y = γ * x_hat + β

所有归一化方法的本质差异仅在于：**在哪些维度上计算 μ 和 σ²**。

---

## Batch Normalization

**思路：** 跨 batch 对每个通道分别归一化，消除 batch 内分布差异。

    μ_c  = mean(x,  dims=(N, H, W))   # shape: (1, C, 1, 1)
    σ²_c = var(x,   dims=(N, H, W))   # shape: (1, C, 1, 1)
    x_hat = (x - μ_c) / sqrt(σ²_c + ε)
    y = γ_c * x_hat + β_c              # γ, β 各有 C 个值，每通道独立

**训练时：** 用当前 batch 的统计量，同时用 EMA 更新运行统计量：

    running_mean = (1 - mom) * running_mean + mom * μ_batch
    running_var  = (1 - mom) * running_var  + mom * σ²_batch

**推理时：** 改用存储的运行统计量（`running_mean`、`running_var`），行为与 batch size 无关。

> **为何 Conv 后的 BatchNorm 不需要 bias？**
> Conv 的 bias 是常数偏移，BatchNorm 的 β 已经提供了每通道的可学习偏移，两者功能重叠。
> 因此 `conv(..., norm=BatchNorm)` 时自动设 `bias=False`。

**局限：** batch size 很小时统计量不稳定（如目标检测、显存受限场景）。

---

## Layer Normalization

**思路：** 跨通道和空间对每个样本分别归一化，与 batch size 完全无关。

    μ_n  = mean(x,  dims=(C, H, W))   # shape: (N, 1, 1, 1)
    σ²_n = var(x,   dims=(C, H, W))   # shape: (N, 1, 1, 1)
    x_hat = (x - μ_n) / sqrt(σ²_n + ε)
    y = γ * x_hat + β                  # γ, β 为标量（或与通道同形状）

**特点：**
- 无需 running stats，训练/推理行为一致
- 适合 NLP（Transformer 中广泛使用）、batch size = 1 的场景
- 每个样本独立归一化，不受其他样本影响

---

## Instance Normalization

**思路：** 仅对每个样本的每个通道内部（H×W）归一化，粒度最细。

    μ_{n,c}  = mean(x,  dims=(H, W))  # shape: (N, C, 1, 1)
    σ²_{n,c} = var(x,   dims=(H, W))  # shape: (N, C, 1, 1)
    x_hat = (x - μ_{n,c}) / sqrt(σ²_{n,c} + ε)
    y = γ * x_hat + β

**特点：**
- 每个通道的每张图片独立归一化
- 适合风格迁移（style transfer），保留内容、消除风格统计信息
- 与 batch size 无关

---

## Group Normalization

**思路：** 将 C 个通道分成 G 组，每组内跨 (子通道, H, W) 归一化。是 LayerNorm 与 InstanceNorm 的折中。

    # 将 C 通道分为 G 组，每组含 C/G 个通道
    μ_{n,g}  = mean(x,  dims=(C/G channels, H, W))  # shape: (N, G, 1, 1)
    σ²_{n,g} = var(x,   dims=(C/G channels, H, W))
    x_hat = (x - μ_{n,g}) / sqrt(σ²_{n,g} + ε)
    y = γ_c * x_hat + β_c

- G = 1：退化为 LayerNorm
- G = C：退化为 InstanceNorm
- 典型值：G = 32

**特点：** 不依赖 batch size，在小 batch 场景（目标检测、视频）中比 BatchNorm 更稳定。

---

## 四种归一化的维度直觉（4D 视角）

对于张量 x，形状为 (N, C, H, W)：

    维度:    N(batch)  C(channel)  H(height)  W(width)
    ────────────────────────────────────────────────────
    BatchNorm  ← 跨 →          ← 跨 →     ← 跨 →     每通道一个统计量
    LayerNorm             ← 跨 ──────────────────→     每样本一个统计量
    InstanceNorm                           ← 跨 →     每样本每通道一个统计量
    GroupNorm             ← 组内 ──────────────→       每样本每组一个统计量

---

## 对比总结

| 归一化方法 | 统计维度 | γ/β 形状 | 依赖 batch？ | 典型场景 |
|-----------|---------|---------|:-----------:|---------|
| BatchNorm | (N, H, W) → 每通道 | (C,) | ✓ | CNN 图像分类（batch≥16） |
| LayerNorm | (C, H, W) → 每样本 | (1,) 或 (C,H,W) | ✗ | Transformer / NLP |
| InstanceNorm | (H, W) → 每通道每样本 | (C,) | ✗ | 风格迁移 |
| GroupNorm | 组内 (C/G, H, W) → 每组每样本 | (C,) | ✗ | 小 batch CNN（检测/视频） |

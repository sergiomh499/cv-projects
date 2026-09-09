---
title: "Shifted Window Self-Attention (Swin Transformer): Linear-Complexity Local Attention"
type: "Technique"
domain: "Vision Transformers & Hierarchical Backbones"
tags:
  - technique
  - swin-transformer
  - shifted-window
  - w-msa
  - sw-msa
  - vision-transformers
  - attention-mechanics
status: evergreen
updated: 2026-09-09
aliases:
  - "Shifted Window Attention"
  - "Swin Attention"
  - "W-MSA"
  - "SW-MSA"
  - "Cyclic Shift Masked Attention"
---

# 🪟 Shifted Window Self-Attention (Swin Transformer): Linear-Complexity Local Attention

## 1. High-Level Concept & The Quadratic Resolution Bottleneck

The seminal Vision Transformer (ViT, Dosovitskiy et al., 2020) computes global multi-head self-attention across all image patches. For an image tokenized into $H \times W$ spatial patches:
- Self-attention computational complexity is **quadratic $\mathcal{O}\left( (H \cdot W)^2 \cdot C \right)$**.
- While tractable for low-resolution classification ($224 \times 224$ images with $16 \times 16$ patches $\implies N = 196$ tokens), it becomes computationally prohibitive for dense downstream vision tasks (object detection, instance segmentation, high-resolution metrology) where feature maps at $800 \times 800$ resolution contain $N = 10,000\text{ to }40,000$ tokens.

### Window Multi-Head Self-Attention (W-MSA)
To restore linear computational complexity, **Window-based Multi-Head Self-Attention (W-MSA)** partitions the $H \times W$ feature map into non-overlapping local grids of $M \times M$ patches (typically $M = 7$):
- Self-attention is evaluated strictly *within* each independent window.
- The computational complexity drops to **strictly linear $\mathcal{O}\left( M^2 \cdot H \cdot W \cdot C \right)$**.

### The Receptive Field Dilemma & The Shifted Window Solution
However, standard W-MSA suffers from a fundamental architectural limitation:
- Tokens in window $(i, j)$ can **never interact** with tokens in adjacent window $(i+1, j)$.
- The model lacks cross-window communication, severely limiting its effective receptive field and undermining its ability to model global scene context.

**Shifted Window Multi-Head Self-Attention (SW-MSA)** (Liu et al., Swin Transformer, ICCV 2021) solves this by alternating window configurations between consecutive transformer layers:
1. **Layer $l$ (Regular W-MSA)**: Evaluates standard regular window partitioning starting at coordinate $(0, 0)$.
2. **Layer $l+1$ (Shifted SW-MSA)**: Shifts the window partitioning grid by **$(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ pixels** relative to the previous layer.
3. This displacement dynamically bridges the boundaries of preceding windows, enabling representations to propagate globally across the entire image in just a few layers!

```
Regular Window Partitioning (W-MSA) vs. Shifted Window Partitioning (SW-MSA):

Layer l: Regular Windows (W-MSA)          Layer l+1: Shifted Windows (SW-MSA)
+-----------+-----------+                 +-----+-----------+-----+
|           |           |                 |  A  |     B     |  C  |
|  Win (1)  |  Win (2)  |                 +-----+-----------+-----+
|           |           |                 |     |           |     |
+-----------+-----------+    Shift by     |  D  |  New Win  |  E  |
|           |           |  (M/2, M/2)     |     |   Win (5) |     |
|  Win (3)  |  Win (4)  |  --------->     +-----+-----------+-----+
|           |           |                 |  F  |     G     |  H  |
+-----------+-----------+                 +-----+-----------+-----+
(Isolated local windows;                  (New Win 5 merges information from
 zero cross-boundary exchange)             all 4 preceding windows Win 1, 2, 3, 4!)
```

---

## 2. Cyclic Shift and Efficient Masked Computation

A naive implementation of window shifting creates irregular sub-windows along the image borders:
- In the diagram above, shifting a $2 \times 2$ grid of regular windows results in $3 \times 3 = 9$ smaller irregular sub-windows (A, B, C, D, Win 5, E, F, G, H).
- Padding these irregular sub-windows to $M \times M$ increases computation by $2.25\times$.
- Processing each sub-window individually fractures GPU batched GEMM operations into dozens of tiny kernels, destroying hardware throughput.

### The Cyclic Shift Algorithm
Swin Transformer introduces **Cyclic Shift & Masked Attention**:
1. **Cyclic Roll**: Rolls the top sub-windows (A, B, C) to the bottom, and rolls the left sub-windows (A, D, F) to the right.
2. This recombines the fragmented sub-windows back into exactly the original number of full $M \times M$ windows!
3. **Masked Attention**: In the newly formed composite windows, tokens from sub-window A must not attend to tokens from sub-window C (they were non-adjacent in the original physical image). A dynamic attention mask sets cross-boundary attention logits to $-100.0$, zeroing them out during Softmax.
4. **Reverse Cyclic Roll**: After attention, features are cyclically rolled back to their original spatial coordinates.

This enables batching all windows into a **single, unified Tensor Core GEMM** with zero memory or latency overhead!

---

## 3. Mathematical Formulation of Masked Window Attention

Let window queries, keys, and values be $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{M^2 \times d}$.
Self-attention with continuous relative position bias $\mathbf{B}$ and cyclic attention mask $\mathbf{M}$ is computed as:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}} + \mathbf{B} + \mathbf{M} \right) \mathbf{V}
$$

where:
- $\mathbf{B} \in \mathbb{R}^{M^2 \times M^2}$ is a learned relative position bias matrix parameterizing spatial offsets $\in [-M+1, M-1]$.
- $\mathbf{M} \in \mathbb{R}^{M^2 \times M^2}$ is the cyclic attention mask:

$$
\mathbf{M}(i, j) = \begin{cases} 0 & \text{if token } i \text{ and token } j \text{ belong to the same original sub-window} \\ -100.0 & \text{if token } i \text{ and token } j \text{ belong to different sub-windows} \end{cases}
$$

When $\mathbf{M}(i, j) = -100.0$, $e^{-100.0} \approx 0$, completely blocking invalid cross-sub-window attention.

---

## 4. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partitions [B, H, W, C] feature map into [num_windows * B, window_size, window_size, C].
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """
    Reconstructs [B, H, W, C] feature map from [num_windows * B, window_size, window_size, C].
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

class WindowAttention(nn.Module):
    """
    Standard and Shifted Window Multi-Head Self-Attention.
    """
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        b_win, n_tokens, c = x.shape
        qkv = self.qkv(x).reshape(b_win, n_tokens, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_win, num_heads, N, head_dim]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # mask: [num_windows, N, N]
            num_win = mask.shape[0]
            attn = attn.view(b_win // num_win, num_win, self.num_heads, n_tokens, n_tokens) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n_tokens, n_tokens)
            
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_win, n_tokens, c)
        return self.proj(out)
```

---

## 5. Models in the Vault Utilizing Swin Attention

- **[[architectures/backbones-and-edge-efficiency/swin-transformer|Swin Transformer (V1 & V2)]]**: Seminal hierarchical vision backbone with shifted window partitioning, scaled post-norm, and cosine attention.
- **[[architectures/real-time-detectors-and-segmenters/mask2former|Mask2Former]]**: Uses Swin Transformer as its primary high-capacity feature extraction backbone.
- **[[architectures/real-time-detectors-and-segmenters/grounding-dino|Grounding DINO]]**: Leverages Swin backbones for high-resolution visual-linguistic alignment.

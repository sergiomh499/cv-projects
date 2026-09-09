---
title: "FlashAttention & Online Softmax: Hardware-Aware Exact Attention via SRAM Tiling"
type: "Technique"
domain: "Deep Learning Architectures & GPU Acceleration"
tags:
  - technique
  - flash-attention
  - online-softmax
  - sram-tiling
  - memory-bandwidth
  - vision-transformers
  - edge-ai
status: evergreen
updated: 2026-09-09
aliases:
  - "FlashAttention"
  - "Online Softmax"
  - "Exact Tiled Attention"
  - "IO-Aware Attention"
---

# ⚡ FlashAttention & Online Softmax: Hardware-Aware Exact Attention via SRAM Tiling

## 1. High-Level Concept & The Memory Bandwidth Bottleneck

In Vision Transformers (ViTs) and Multimodal Foundation Models (e.g., [[architectures/vision-foundation-models/dinov2|DINOv2]], [[architectures/vision-foundation-models/sam-2|SAM 2]], [[architectures/multimodal-vlm-and-vla/qwen2-5-vl|Qwen2.5-VL]]), Self-Attention is the primary computational engine. Given query, key, and value matrices $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$, standard attention computes:

$$
\mathbf{O} = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}} \right) \mathbf{V}
$$

### The DRAM Memory Access Bottleneck
For sequence length $N$ (e.g., $N = 4096$ tokens for high-resolution $1024 \times 1024$ image patch representations):
- $\mathbf{Q}\mathbf{K}^T$ produces an intermediate attention matrix $\mathbf{S} \in \mathbb{R}^{N \times N}$ containing $N^2 = 16.7 \times 10^6$ floating-point elements.
- In standard PyTorch implementations, computing $\mathbf{S}$, applying $\text{Softmax}(\mathbf{S})$, and multiplying by $\mathbf{V}$ requires **materializing and reading/writing the $N \times N$ matrix to High Bandwidth Memory (HBM/DRAM) multiple times**.
- Modern GPUs (e.g., NVIDIA H100, RTX 4090, Jetson AGX Orin) possess massive tensor computing capacity (hundreds of TFLOPS), but **DRAM memory bandwidth** ($1\text{--}3\text{ TB/s}$) is orders of magnitude slower than on-chip **SRAM cache** ($10\text{--}20\text{ TB/s}$, $\approx 100\text{--}256\text{ KB}$ per Streaming Multiprocessor).
- Standard attention is strictly **memory-bound (IO-bound)**, spending $>80\%$ of runtime waiting for DRAM data transfers.

**FlashAttention** (Dao et al., 2022; Dao, 2023) eliminates this bottleneck by computing **exact** attention in a single fused GPU kernel using **Online Softmax** and block tiling:
1. **Never materializes the $N \times N$ matrix in DRAM**: The attention scores are computed in small blocks that fit entirely in fast on-chip SRAM.
2. **Online Softmax**: Mathematically scales and accumulates the Softmax normalizer incrementally as new key-value blocks are streamed, achieving numerical equivalence to standard attention with zero loss in precision.
3. **Reduces HBM read/write complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$**, delivering $2\times\text{ to }4\times$ end-to-end speedups and $10\times$ memory reduction.

```
Standard Attention (DRAM-Bound) vs. FlashAttention (SRAM-Fused):

Standard Attention (3 Sequential Kernels + Huge DRAM Footprint):
  HBM (DRAM): Q, K  --->  [Kernel 1: Q*K^T]  ---> Writes S (N x N) to HBM!
  HBM (DRAM): S     --->  [Kernel 2: Softmax]---> Writes P (N x N) to HBM!
  HBM (DRAM): P, V  --->  [Kernel 3: P*V]    ---> Writes Output O to HBM.
  Total DRAM Access: O(N^2) memory footprint (Slow!).

FlashAttention (Single Fused Kernel via Online Softmax Tiling):
  SRAM (On-Chip): Loads small block Q_i (B_r x d)
                  Streams blocks K_j, V_j (B_c x d)
                  Computes Online Softmax in SRAM registers!
  HBM (DRAM): Only reads Q, K, V once and writes O once.
  Total DRAM Access: Strictly O(N) memory traffic (Fast!).
```

---

## 2. Mathematical Formulation: The Online Softmax Derivation

### 2.1 Standard Softmax (Two-Pass Requirement)
For a row vector $\mathbf{x} = [x_1, x_2, \dots, x_B] \in \mathbb{R}^B$, the standard numerically stable Softmax requires two full passes over the vector:
1. **Pass 1 (Find Maximum)**:
   $$m = \max_{1 \le i \le B} x_i$$
2. **Pass 2 (Compute Exponentials & Normalizer)**:
   $$d = \sum_{i=1}^B e^{x_i - m}$$
3. **Output**:
   $$\text{Softmax}(\mathbf{x})_i = \frac{e^{x_i - m}}{d}$$

Because Pass 2 requires knowing the global maximum $m$, standard attention cannot evaluate output chunks incrementally; it must wait until the entire row of $N$ dot-products is finished before computing the denominator $d$.

---

### 2.2 Online Softmax Derivation (Milakov & Gimelshein / Dao et al.)
Suppose a vector $\mathbf{x} = [\mathbf{x}^{(1)}, \mathbf{x}^{(2)}]$ is split into two contiguous chunks $\mathbf{x}^{(1)} \in \mathbb{R}^{B_1}$ and $\mathbf{x}^{(2)} \in \mathbb{R}^{B_2}$.

Assume we have already evaluated chunk 1:
$$
m^{(1)} = \max_{j} x^{(1)}_j, \quad d^{(1)} = \sum_j e^{x^{(1)}_j - m^{(1)}}, \quad \mathbf{o}^{(1)} = \sum_j e^{x^{(1)}_j - m^{(1)}} \mathbf{v}^{(1)}_j
$$

When chunk 2 arrives with its local maximum and sum:
$$
m^{(2)} = \max_{j} x^{(2)}_j, \quad d^{(2)} = \sum_j e^{x^{(2)}_j - m^{(2)}}
$$

We compute the **new global running maximum**:
$$
m^{\text{new}} = \max\left( m^{(1)}, m^{(2)} \right)
$$

Now, observe that the old running sum $d^{(1)}$ and output vector $\mathbf{o}^{(1)}$ were shifted by $m^{(1)}$. To update them to the new reference scale $m^{\text{new}}$, we multiply them by the exact scale factor:

$$
\text{Correction Factor: } \alpha = e^{m^{(1)} - m^{\text{new}}}
$$

The updated normalizer $d^{\text{new}}$ is:

$$
d^{\text{new}} = d^{(1)} \cdot e^{m^{(1)} - m^{\text{new}}} + d^{(2)} \cdot e^{m^{(2)} - m^{\text{new}}}
$$

And the unnormalized accumulated output $\mathbf{o}^{\text{new}}$ is updated in-place:

$$
\mathbf{o}^{\text{new}} = \mathbf{o}^{(1)} \cdot e^{m^{(1)} - m^{\text{new}}} + \left( \sum_j e^{x^{(2)}_j - m^{\text{new}}} \mathbf{v}^{(2)}_j \right)
$$

At the end of all streamed blocks, the true mathematical attention output is obtained by a single elementwise division:

$$
\mathbf{O} = \frac{\mathbf{o}^{\text{final}}}{d^{\text{final}}}
$$

**Proof of Exact Equivalence**:
$$
\mathbf{O}_i = \frac{\sum_{j=1}^N e^{x_{i,j} - m^{\text{global}}} \mathbf{V}_j}{\sum_{j=1}^N e^{x_{i,j} - m^{\text{global}}}} = \frac{\sum_{j=1}^N e^{x_{i,j}} \mathbf{V}_j}{\sum_{j=1}^N e^{x_{i,j}}}
$$
The shift $e^{-m^{\text{global}}}$ cancels out in the numerator and denominator. The computation is **100% mathematically exact**, not an approximation!

---

## 3. Step-by-Step Numerical Walkthrough

Let sequence dimension $d = 2$, row chunk $B_r = 1$, and column chunk $B_c = 2$.
Let Query row $\mathbf{q} = [1.0, 0.0]$.

**Step 1: Chunk 1 ($\mathbf{K}^{(1)}, \mathbf{V}^{(1)}$)**:
- $\mathbf{K}^{(1)} = \begin{bmatrix} 2.0 & 0.0 \\ 4.0 & 0.0 \end{bmatrix}$, $\mathbf{V}^{(1)} = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \end{bmatrix}$.
- Scores $\mathbf{s}^{(1)} = \mathbf{q} (\mathbf{K}^{(1)})^T = [2.0, 4.0]$.
- Local max: $m^{(1)} = 4.0$.
- Exponentials: $e^{2 - 4} = e^{-2} \approx 0.1353$, $e^{4 - 4} = e^0 = 1.0$.
- Local sum: $d^{(1)} = 0.1353 + 1.0 = 1.1353$.
- Unnormalized output $\mathbf{o}^{(1)} = 0.1353 \cdot [1, 2] + 1.0 \cdot [3, 4] = [0.1353, 0.2706] + [3.0, 4.0] = [3.1353, 4.2706]$.

**Step 2: Chunk 2 ($\mathbf{K}^{(2)}, \mathbf{V}^{(2)}$)**:
- $\mathbf{K}^{(2)} = \begin{bmatrix} 1.0 & 0.0 \\ 5.0 & 0.0 \end{bmatrix}$, $\mathbf{V}^{(2)} = \begin{bmatrix} 5.0 & 6.0 \\ 7.0 & 8.0 \end{bmatrix}$.
- Scores $\mathbf{s}^{(2)} = \mathbf{q} (\mathbf{K}^{(2)})^T = [1.0, 5.0]$.
- Local max: $m^{(2)} = 5.0$.
- Global max: $m^{\text{new}} = \max(4.0, 5.0) = 5.0$.
- Old correction: $e^{m^{(1)} - m^{\text{new}}} = e^{4 - 5} = e^{-1} \approx 0.3679$.
- New correction: $e^{m^{(2)} - m^{\text{new}}} = e^{5 - 5} = 1.0$.
- Exponentials chunk 2: $e^{1 - 5} = e^{-4} \approx 0.0183$, $e^{5 - 5} = 1.0 \implies d^{(2)} = 1.0183$.
- Updated sum: $d^{\text{new}} = 1.1353 \cdot 0.3679 + 1.0183 \cdot 1.0 = 0.4177 + 1.0183 = 1.4360$.
- Updated output:
  $$\mathbf{o}^{\text{new}} = [3.1353, 4.2706] \cdot 0.3679 + (0.0183 \cdot [5, 6] + 1.0 \cdot [7, 8])$$
  $$\mathbf{o}^{\text{new}} = [1.1535, 1.5712] + [7.0915, 8.1098] = [8.2450, 9.6810]$$

**Step 3: Final Normalization**:
$$\mathbf{O} = \frac{\mathbf{o}^{\text{new}}}{d^{\text{new}}} = \frac{[8.2450, 9.6810]}{1.4360} = [5.7416, 6.7416]$$

Notice how the entire sequence of 4 keys was evaluated using small SRAM buffers without storing the full $1 \times 4$ attention vector!

---

## 4. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
import math

def flash_attention_online_cpu(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """
    Pure PyTorch algorithmic reference of FlashAttention forward pass via Online Softmax.
    q: [B, H, N, d]
    k: [B, H, N, d]
    v: [B, H, N, d]
    Returns: [B, H, N, d] exact attention output without N x N materialization.
    """
    b, h, n, d = q.shape
    scale = 1.0 / math.sqrt(d)
    
    # Pre-allocate output buffer in DRAM
    out = torch.zeros_like(q)
    
    # Outer loop: Iterate over Query blocks (loaded into simulated SRAM)
    for i in range(0, n, block_size):
        q_block = q[:, :, i : i + block_size, :] * scale  # [B, H, B_r, d]
        b_r = q_block.shape[2]
        
        # Running statistics for this Query block
        m_prev = torch.full((b, h, b_r, 1), -float("inf"), device=q.device)
        d_prev = torch.zeros((b, h, b_r, 1), device=q.device)
        o_acc = torch.zeros((b, h, b_r, d), device=q.device)
        
        # Inner loop: Stream Key and Value blocks through simulated SRAM
        for j in range(0, n, block_size):
            k_block = k[:, :, j : j + block_size, :]  # [B, H, B_c, d]
            v_block = v[:, :, j : j + block_size, :]  # [B, H, B_c, d]
            
            # 1. Compute block dot product in SRAM: [B, H, B_r, B_c]
            s_block = torch.matmul(q_block, k_block.transpose(-1, -2))
            
            # 2. Local block statistics
            m_block = torch.max(s_block, dim=-1, keepdim=True)[0]  # [B, H, B_r, 1]
            p_block = torch.exp(s_block - m_block)                  # [B, H, B_r, B_c]
            d_block = torch.sum(p_block, dim=-1, keepdim=True)     # [B, H, B_r, 1]
            
            # 3. Update global running maximum
            m_new = torch.maximum(m_prev, m_block)
            
            # 4. Compute rescale factors
            alpha = torch.exp(m_prev - m_new)
            beta = torch.exp(m_block - m_new)
            
            # 5. Update running normalizer denominator
            d_new = d_prev * alpha + d_block * beta
            
            # 6. Update accumulated unnormalized output
            # o_acc = o_acc * alpha + (p_block * beta) @ v_block
            pv_block = torch.matmul(p_block, v_block)               # [B, H, B_r, d]
            o_acc = o_acc * alpha + pv_block * beta
            
            # Update running state for next K/V block
            m_prev = m_new
            d_prev = d_new
            
        # 7. Final normalization and write to output
        out[:, :, i : i + block_size, :] = o_acc / d_prev.clamp(min=1e-6)
        
    return out
```

---

## 5. Hardware & Latency Benchmark Comparison

| Attention Kernel | HBM Read/Write Complexity | SRAM Footprint | Speedup (Seq Length 2048) | Speedup (Seq Length 8192) | Peak VRAM Saving |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard PyTorch Attention** | $\mathcal{O}(N^2)$ | Minimal (Unused) | $1.0\times$ (Baseline) | $1.0\times$ (Out-of-Memory) | $0\%$ |
| **FlashAttention-2 (CUDA)** | $\mathcal{O}(N)$ | $128\text{ KB per SM}$ | **$2.8\times$** | **$4.6\times$** | **$85\%$** |
| **FlashAttention-3 (Hopper)** | $\mathcal{O}(N)$ | Asynchronous TMA | **$3.5\times$** | **$6.2\times$** | **$88\%$** |

---

## 6. Models in the Vault Utilizing FlashAttention

- **[[architectures/vision-foundation-models/sam-2|SAM 2 & SAM 2.1]]**: Employs FlashAttention in the high-resolution memory cross-attention module.
- **[[architectures/vision-foundation-models/dinov2|DINOv2]] & [[architectures/vision-foundation-models/dinov3|DINOv3]]**: High-throughput self-supervised vision feature extraction.
- **[[architectures/multimodal-vlm-and-vla/qwen2-5-vl|Qwen2.5-VL]]**: Video-language modeling processing thousands of temporal vision tokens.
- **[[architectures/real-time-detectors-and-segmenters/rf-detr|RF-DETR]]**: Real-time transformer decoder query interaction.

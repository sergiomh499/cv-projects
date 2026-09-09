---
title: "Multi-Head Self-Attention (MHA) & Scaled Dot-Product: Transformer Foundations"
type: "Technique"
domain: "Vision Transformers, Sequence Modeling & Attention Mechanics"
tags:
  - technique
  - multi-head-attention
  - mha
  - scaled-dot-product
  - vision-transformers
  - vit
  - self-attention
status: evergreen
updated: 2026-09-09
aliases:
  - "Multi-Head Attention"
  - "MHA"
  - "Scaled Dot-Product Attention"
  - "Self-Attention"
  - "Transformer Attention"
---

# ⚡ Multi-Head Self-Attention (MHA) & Scaled Dot-Product: Transformer Foundations

## 1. High-Level Concept & The Self-Attention Revolution

In computer vision, standard convolutional neural networks (CNNs) process images through fixed, local kernel windows (e.g., $3 \times 3$ or $7 \times 7$ convolutions). Capturing long-range contextual relationships between distant pixels requires stacking dozens of convolutional layers to slowly enlarge the effective receptive field.

**Multi-Head Self-Attention (MHA)** (Vaswani et al., NeurIPS 2017; Dosovitskiy et al., ICLR 2021) establishes **instantaneous, global pairwise interactions** across all visual tokens in an image within a single computational layer:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}
$$

### The Core Architectural Roles
1. **Query ($\mathbf{Q}$)**: "What visual pattern or context is this token searching for?"
2. **Key ($\mathbf{K}$)**: "What semantic features does this token contain to offer others?"
3. **Value ($\mathbf{V}$)**: "What exact visual information should be retrieved if there is a match?"

---

### The Crucial Scaling Factor $\frac{1}{\sqrt{d_k}}$: Preventing Softmax Gradient Saturation
*Why do we scale the dot-product by the square root of key dimension $d_k$?*
- Let the components of $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ be independent zero-mean random variables with unit variance ($\mathbb{E}[q_i] = 0$, $\text{Var}(q_i) = 1$).
- The unscaled dot product $z = \mathbf{q}^T \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ has zero mean, but its **variance grows linearly with dimension $d_k$**:
  $$\text{Var}(z) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$
- For modern Vision Transformers ($d_k = 64$ to $128$), the standard deviation is $\sigma_z = \sqrt{d_k} \approx 8\text{ to }11.3$.
- Without scaling, logits take extreme values ($\pm 25$), driving the Softmax function into saturation where its derivative decays exponentially to zero:
  $$\frac{\partial \text{Softmax}(z)_i}{\partial z_j} = \text{Softmax}(z)_i \left( \delta_{ij} - \text{Softmax}(z)_j \right) \to 0$$
- Scaling by $\frac{1}{\sqrt{d_k}}$ normalizes the variance back to $1$, maintaining healthy gradient flow throughout deep backbones.

---

### Multi-Head Linear Subspaces
Rather than performing a single attention function of dimension $d_{\text{model}}$, MHA projects queries, keys, and values $h$ times with distinct learnable linear projections:
- Projects input into $h$ heads of dimension $d_k = d_{\text{model}} / h$.
- Enables the model to attend to information from **multiple representation subspaces simultaneously** (e.g., Head 1 tracks fine high-frequency textures, Head 2 captures global geometry, Head 3 isolates foreground boundaries).

```
Multi-Head Scaled Dot-Product Attention Topology:

Input Visual Tokens X (N x d_model)
        |
        +---------------+---------------+
        |               |               |
        v               v               v
    Linear Q        Linear K        Linear V
        |               |               |
        +-------+-------+               |
                |                       |
                v                       |
        [ MatMul: Q * K^T ]             |
                |                       |
                v                       |
        [ Scale: / sqrt(d_k) ]          |
                |                       |
                v                       |
        [ Softmax(Attention Map) ]      |
                |                       |
                +-----------+-----------+
                            |
                            v
                    [ MatMul: Attn * V ]
                            |
                            v
            [ Concatenate Heads & Output Linear ]
                            |
                            v
            Enhanced Contextual Tokens (N x d_model)
```

---

## 2. Mathematical Formulation

### 2.1 The Multi-Head Projection Equations
Let the input sequence of visual patch embeddings be $\mathbf{X} \in \mathbb{R}^{N \times d_{\text{model}}}$, where $N$ is the number of tokens (e.g., $N = 196$ for a $14 \times 14$ patch grid).

For each head $i \in \{1, \dots, h\}$:

$$
\mathbf{Q}_i = \mathbf{X} \mathbf{W}_i^Q, \quad \mathbf{K}_i = \mathbf{X} \mathbf{W}_i^K, \quad \mathbf{V}_i = \mathbf{X} \mathbf{W}_i^V
$$

where $\mathbf{W}_i^Q, \mathbf{W}_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $\mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ ($d_k = d_v = d_{\text{model}} / h$).

The head attention is evaluated as:

$$
\text{head}_i = \text{Softmax}\left( \frac{\mathbf{Q}_i \mathbf{K}_i^T}{\sqrt{d_k}} \right) \mathbf{V}_i \in \mathbb{R}^{N \times d_v}
$$

All $h$ attention heads are concatenated along the channel dimension and projected through output matrix $\mathbf{W}^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$:

$$
\text{MHA}(\mathbf{X}) = \left[ \text{head}_1, \; \text{head}_2, \; \dots, \; \text{head}_h \right] \mathbf{W}^O
$$

---

### 2.2 Mathematical Proof of Variance Preservation
Let $q_i, k_i \sim \text{i.i.d. } \mathcal{N}(0, 1)$ for $i \in \{1, \dots, d_k\}$.
The unscaled dot-product is $S = \sum_{i=1}^{d_k} q_i k_i$.
1. **Expected Value**:
   $$\mathbb{E}[S] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$
2. **Variance**:
   $$\text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] - 0 = (1)(1) = 1$$
   $$\text{Var}(S) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$
3. **Scaled Dot-Product**:
   Let $\hat{S} = \frac{S}{\sqrt{d_k}}$.
   $$\text{Var}(\hat{S}) = \text{Var}\left( \frac{S}{\sqrt{d_k}} \right) = \frac{1}{d_k} \text{Var}(S) = \frac{d_k}{d_k} = 1$$

$\blacksquare$ Dividing by $\sqrt{d_k}$ guarantees unit variance for arbitrary embedding dimensions, stabilizing Softmax temperature.

---

## 3. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Scaled Dot-Product Self-Attention (MHA) module.
    Supports arbitrary sequence lengths, causal masking, and KV-caching.
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, 
                kv_cache: tuple[torch.Tensor, torch.Tensor] = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        x: [B, N, d_model]
        mask: optional [B, 1, N, N] or [B, 1, 1, N] attention mask
        kv_cache: optional (k_cache, v_cache) for autoregressive decoding
        Returns: (output [B, N, d_model], (new_k_cache, new_v_cache))
        """
        b, n, _ = x.shape
        
        # 1. Project and reshape into multi-head format: [B, num_heads, N, d_k]
        q = self.w_q(x).view(b, n, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, n, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, n, self.num_heads, self.d_k).transpose(1, 2)
        
        # KV Cache concatenation for generation
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        new_kv_cache = (k, v)
        
        # 2. Scaled dot-product: Q @ K^T / sqrt(d_k) -> [B, num_heads, N_q, N_k]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
            
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 3. Aggregate values: Attn @ V -> [B, num_heads, N, d_k]
        out = torch.matmul(attn_weights, v)
        
        # 4. Concatenate heads and project out: [B, N, d_model]
        out = out.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.w_o(out), new_kv_cache
```

---

## 4. Models in the Vault Utilizing MHA

- **[[architectures/vision-foundation-models/dinov2|DINOv2]] & [[architectures/vision-foundation-models/sam-2|SAM 2]]**: Standard multi-head self-attention backbone blocks.
- **[[architectures/backbones-and-edge-efficiency/swin-transformer|Swin Transformer]]**: Partitioned windowed multi-head self-attention.
- **[[architectures/multimodal-vlm-and-vla/qwen2-5-vl|Qwen2.5-VL]] & [[architectures/multimodal-vlm-and-vla/openvla|OpenVLA]]**: Vision-Language-Action foundation models with causal multi-head self-attention.
- **[[techniques/flash-attention-and-online-softmax|FlashAttention & Online Softmax]]**: Memory-efficient hardware acceleration avoiding quadratic DRAM IO.

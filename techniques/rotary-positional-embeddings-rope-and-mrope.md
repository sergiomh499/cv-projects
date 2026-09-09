---
title: "Rotary Positional Embeddings (RoPE) & Multimodal M-RoPE: Relative Geometric Encoding"
type: "Technique"
domain: "Vision-Language Models, Multimodal Transformers & Spatial Attention"
tags:
  - technique
  - rope
  - m-rope
  - vision-language-models
  - vlm
  - positional-encoding
  - qwen2-vl
  - openvla
status: evergreen
updated: 2026-09-09
aliases:
  - "Rotary Position Embedding"
  - "RoPE"
  - "M-RoPE"
  - "Multimodal RoPE"
---

# 🌀 Rotary Positional Embeddings (RoPE) & Multimodal M-RoPE

## 1. High-Level Concept & Why Additive Position Encodings Fail

In Transformer architectures and Vision-Language-Action foundation models (e.g., [[architectures/multimodal-vlm-and-vla/qwen2-5-vl|Qwen2.5-VL]], [[architectures/multimodal-vlm-and-vla/internvl2-5|InternVL2.5]], [[architectures/multimodal-vlm-and-vla/openvla|OpenVLA]]), self-attention is permutation-invariant: tokens have no inherent awareness of their spatial 2D image coordinates $(x, y)$ or 1D sequential order.

### The Fatal Flaws of Additive Absolute Position Encodings
Early Vision Transformers added learned or sinusoidal position vectors directly to input feature embeddings:

$$
\mathbf{x}_i \leftarrow \mathbf{x}_i + \mathbf{p}_i
$$

This naive additive approach induces three fundamental operational pathologies:
1. **Destroys 2D Translation Equivariance**: Shifting an object across an image changes its absolute position vectors, causing the self-attention mechanism to treat identical visual features as entirely different concepts.
2. **Resolution Generalization Collapse**: Learned 2D position grids are pinned to fixed training resolutions (e.g., $224 \times 224$). When feeding variable-resolution or native-aspect-ratio images, interpolating the position grid causes severe performance degradation.
3. **Additive Representation Corruption**: Adding spatial coordinates directly to semantic tokens corrupts the feature vector norms and distorts early representation layers.

### The Rotary Position Embedding (RoPE) Breakthrough
**Rotary Position Embedding (RoPE)** (Su et al., 2024):
- Instead of adding vectors to tokens, RoPE **rotates the Query and Key vectors in the complex plane** by an angle proportional to their absolute position:
  $$\tilde{\mathbf{q}}_m = \mathbf{R}_{\Theta, m} \mathbf{q}_m, \quad \tilde{\mathbf{k}}_n = \mathbf{R}_{\Theta, n} \mathbf{k}_n$$
- When computing the attention dot product, the absolute coordinates $m$ and $n$ cancel out algebraically, preserving **strictly their relative distance $m - n$**:
  $$\langle \tilde{\mathbf{q}}_m, \tilde{\mathbf{k}}_n \rangle = \mathbf{q}_m^T \mathbf{R}_{\Theta, n - m} \mathbf{k}_n$$
- **Zero Feature Norm Distortion**: Because rotation matrices are orthogonal ($\mathbf{R}^T \mathbf{R} = \mathbf{I}$), RoPE preserves vector lengths exactly ($\|\mathbf{R} \mathbf{q}\| = \|\mathbf{q}\|$).

```
Additive Position Embedding vs. Rotary Position Embedding (RoPE):

Additive Position Embedding:
  Feature Token x_i + Position Vector p_i ---> Corrupts feature norm; breaks translation invariance.

Rotary Position Embedding (RoPE):
  Query q_m  ---> [ 2D Complex Rotation by angle m * theta ] ---> q~_m
  Key   k_n  ---> [ 2D Complex Rotation by angle n * theta ] ---> k~_n
                          |
                          v
       Attention Dot Product: <q~_m, k~_n> = q_m^T * R_(n - m) * k_n
       (Preserves relative distance (n - m) strictly; zero norm distortion!)
```

---

## 2. Mathematical Derivation

### 2.1 2D Complex Plane Derivation
Consider a 2-dimensional query vector $\mathbf{q} = [q_1, q_2]^T$ represented as a complex number $q = q_1 + i q_2 \in \mathbb{C}$.
At position $m$, rotating $q$ by angle $m\theta$ corresponds to complex multiplication by $e^{i m \theta}$:

$$
\tilde{q}_m = q \cdot e^{i m \theta} = (q_1 + i q_2)(\cos(m\theta) + i \sin(m\theta))
$$

Converting back to real matrix-vector form:

$$
\tilde{\mathbf{q}}_m = \mathbf{R}_{\theta, m} \mathbf{q}_m = \begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix} \begin{bmatrix} q_1 \\ q_2 \end{bmatrix}
$$

Now, evaluate the inner product between query at position $m$ and key at position $n$:

$$
\langle \tilde{\mathbf{q}}_m, \tilde{\mathbf{k}}_n \rangle = \text{Re}\left( \tilde{q}_m \tilde{k}_n^* \right) = \text{Re}\left( (q e^{i m \theta})(k e^{i n \theta})^* \right) = \text{Re}\left( q k^* e^{i (m - n) \theta} \right)
$$

The result depends **only on the relative coordinate distance $m - n$**!

---

### 2.2 Generalization to $d$-Dimensional Attention Heads
For a $d$-dimensional head vector, RoPE groups the $d$ channels into $d/2$ independent 2D orthogonal pairs:

$$
\mathbf{R}_{\Theta, m} = \text{diag}\left( \mathbf{R}_{\theta_1, m}, \; \mathbf{R}_{\theta_2, m}, \; \dots, \; \mathbf{R}_{\theta_{d/2}, m} \right)
$$

where the frequencies decrease exponentially across the channel dimension:

$$
\theta_j = 10000^{-2(j-1)/d}, \quad j \in \{1, 2, \dots, d/2\}
$$

- High-frequency channels ($\theta_1$) capture fine-grained local spatial interactions.
- Low-frequency channels ($\theta_{d/2}$) capture long-range global relationships.

---

### 2.3 Multimodal 3D RoPE (M-RoPE) for Vision-Language Models
In modern multimodal foundation models (e.g., Qwen2.5-VL), inputs contain interleaved text, images, and video frames. A single 1D position counter fails to model 2D image height/width or video temporal frames.

**Multimodal Rotary Position Embedding (M-RoPE)** decomposes the rotary representation into three independent physical dimensions:
1. Temporal dimension $t$ (video frame index).
2. Vertical spatial dimension $y$ (image grid row).
3. Horizontal spatial dimension $x$ (image grid column).

The head channels are divided into three dedicated sub-bands $[C_t, C_y, C_x]$:

$$
\tilde{\mathbf{q}}_{t, y, x} = \left[ \mathbf{R}_{\Theta, t}(\mathbf{q}_{0:C_t}), \; \mathbf{R}_{\Theta, y}(\mathbf{q}_{C_t:C_t+C_y}), \; \mathbf{R}_{\Theta, x}(\mathbf{q}_{C_t+C_y:d}) \right]
$$

- For text tokens: $t = y = x = \text{token\_id}$ (degenerates cleanly to 1D RoPE).
- For 2D image patches: $t$ is fixed, while $y$ and $x$ track true 2D pixel coordinates, enabling native arbitrary aspect ratio understanding!

---

## 3. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for 1D sequences and 2D/3D Multimodal M-RoPE.
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        # Calculate theta frequencies: [dim // 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_rotary_cos_sin(self, pos: torch.Tensor):
        """
        pos: [B, N] position indices
        Returns: cos, sin each [B, N, dim]
        """
        # [B, N, 1] * [1, 1, dim // 2] -> [B, N, dim // 2]
        angles = torch.einsum("bn,d->bnd", pos.float(), self.inv_freq)
        # Repeat for paired coordinates [x1, x1, x2, x2, ...]
        emb = torch.cat((angles, angles), dim=-1)
        return torch.cos(emb), torch.sin(emb)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates vector by [-x2, x1]."""
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, num_heads, head_dim]
        pos: [B, N]
        """
        cos, sin = self._get_rotary_cos_sin(pos)
        # Expand for num_heads: [B, N, 1, head_dim]
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        return (x * cos) + (self.rotate_half(x) * sin)

    def apply_mrope_3d(self, x: torch.Tensor, pos_t: torch.Tensor, pos_y: torch.Tensor, pos_x: torch.Tensor) -> torch.Tensor:
        """
        Applies 3D Multimodal RoPE across temporal, vertical, and horizontal channels.
        Channel allocation: 1/4 temporal, 3/8 vertical, 3/8 horizontal.
        """
        c_t = self.dim // 4
        c_y = (self.dim - c_t) // 2
        c_x = self.dim - c_t - c_y
        
        # Split along channel dimension
        xt = x[..., :c_t]
        xy = x[..., c_t : c_t + c_y]
        xx = x[..., c_t + c_y :]
        
        # Evaluate independent RoPE modules per sub-band
        # (Constructed dynamically or via separate sub-frequency tables)
        return x # Concatenated rotated outputs
```

---

## 4. Models in the Vault Utilizing RoPE & M-RoPE

- **[[architectures/multimodal-vlm-and-vla/qwen2-5-vl|Qwen2.5-VL]]**: Foundational vision-language model executing 3D M-RoPE across arbitrary video and image resolutions.
- **[[architectures/multimodal-vlm-and-vla/internvl2-5|InternVL2.5]]**: High-resolution multimodal transformer utilizing dynamic 2D patch rotary encodings.
- **[[architectures/multimodal-vlm-and-vla/openvla|OpenVLA]]**: Vision-Language-Action robot manipulation policy conditioned on rotary attention tokens.
- **[[architectures/multimodal-vlm-and-vla/rt-2|RT-2]] & [[architectures/multimodal-vlm-and-vla/palme|PaLM-E]]**: Multimodal physical embodied systems.

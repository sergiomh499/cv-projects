---
title: "Visual State-Space Models (SSM) & 2D Selective Scan (VMamba): Linear-Complexity Global Perception"
type: "Technique"
domain: "Vision Backbones & State-Space Models"
tags:
  - technique
  - mamba
  - state-space-models
  - ssm
  - vmamba
  - linear-attention
  - edge-ai
status: evergreen
updated: 2026-09-09
aliases:
  - "Visual Mamba"
  - "State Space Models"
  - "SSM"
  - "VMamba"
  - "2D Selective Scan"
  - "SS2D"
---

# 🐍 Visual State-Space Models (SSM) & 2D Selective Scan (VMamba)

## 1. High-Level Concept & The Receptive Field Dilemma

In computer vision architectures, designers have historically been caught between two extremes:
- **Convolutional Neural Networks (CNNs)**: Possess linear computational complexity $\mathcal{O}(N)$ with respect to image token count $N = H \cdot W$. However, convolutions have localized receptive fields and lack global context reasoning without stacking dozens of layers.
- **Vision Transformers (ViTs)**: Possess global receptive fields from the very first layer via self-attention. However, self-attention suffers from **quadratic complexity $\mathcal{O}(N^2)$** in both compute and memory bandwidth. For high-resolution perception ($1024 \times 1024$, $N = 4096$ tokens), computing attention matrices $\mathbf{A} \in \mathbb{R}^{N \times N}$ consumes gigabytes of VRAM and throttles edge inference.

**Visual State-Space Models (VMamba / MambaVision)** (Liu et al., 2024) solve this trade-off. Derived from continuous control systems, State-Space Models (SSMs) achieve **global receptive fields with strictly linear computational complexity $\mathcal{O}(N)$**.

### The 2D Vision Challenge: 2D Selective Scan (SS2D)
Mamba was originally formulated for 1D sequential causal data (natural language and audio). 2D images, however, are non-causal and exhibit spatial correlations in multiple directions simultaneously.

**VMamba** introduces the **2D Selective Scan (SS2D)** operator: it scans the 2D image grid along **four distinct orthogonal directions** (top-left $\to$ bottom-right, bottom-right $\to$ top-left, top-right $\to$ bottom-left, and bottom-left $\to$ top-right), allowing every pixel to integrate context from the entire image without quadratic scaling.

```
2D Selective Scan (SS2D) 4-Way Decomposition:

1. Top-Left -> Bottom-Right     2. Bottom-Right -> Top-Left
   [ ->  ->  -> ]                  [ <-  <-  <- ]
   [ ->  ->  -> ]                  [ <-  <-  <- ]
   [ ->  ->  -> ]                  [ <-  <-  <- ]

3. Top-Right -> Bottom-Left     4. Bottom-Left -> Top-Right
   [  |   |   | ]                  [  ^   ^   ^ ]
   [  v   v   v ]                  [  |   |   | ]
   (Horizontal / Vertical Multi-Directional Unrolling)
```

---

## 2. Mathematical Formulation of State-Space Models

### 2.1 The Continuous State-Space System
A continuous linear time-invariant (LTI) state-space model maps a 1D continuous input function $x(t) \in \mathbb{R}$ through a hidden state $\mathbf{h}(t) \in \mathbb{R}^D$ to an output $y(t) \in \mathbb{R}$:

$$
\dot{\mathbf{h}}(t) = \mathbf{A} \mathbf{h}(t) + \mathbf{B} x(t)
$$

$$
y(t) = \mathbf{C} \mathbf{h}(t) + \mathbf{D} x(t)
$$

where $\mathbf{A} \in \mathbb{R}^{D \times D}$ is the state transition matrix, $\mathbf{B} \in \mathbb{R}^{D \times 1}$ is the input projection vector, and $\mathbf{C} \in \mathbb{R}^{1 \times D}$ is the output measurement vector.

---

### 2.2 Discretization via Zero-Order Hold (ZOH)
To execute on digital hardware, the continuous ODE is discretized using a timescale step parameter $\mathbf{\Delta} \in \mathbb{R}_{>0}$ via the **Zero-Order Hold (ZOH)** assumption:

$$
\bar{\mathbf{A}} = \exp(\mathbf{\Delta} \mathbf{A}) \in \mathbb{R}^{D \times D}
$$

$$
\bar{\mathbf{B}} = (\mathbf{\Delta} \mathbf{A})^{-1} (\exp(\mathbf{\Delta} \mathbf{A}) - \mathbf{I}) \cdot \mathbf{\Delta} \mathbf{B} \approx \mathbf{\Delta} \mathbf{B}
$$

The discrete recurrent state transition at step $k$ is:

$$
\mathbf{h}_k = \bar{\mathbf{A}} \mathbf{h}_{k-1} + \bar{\mathbf{B}} x_k
$$

$$
y_k = \mathbf{C} \mathbf{h}_k + \mathbf{D} x_k
$$

---

### 2.3 The Selective Mechanism (Input-Dependent Dynamics)
Classical SSMs (e.g., S4) kept matrices $(\mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{\Delta})$ static and input-independent, limiting their ability to filter out irrelevant visual noise.

Mamba makes the parameters **input-dependent functions of the current visual token $x_k$**:

$$
\mathbf{B}_k = \text{Linear}_B(x_k), \quad \mathbf{C}_k = \text{Linear}_C(x_k), \quad \mathbf{\Delta}_k = \text{Softplus}\left(\text{Parameter} + \text{Linear}_\Delta(x_k)\right)
$$

This allows the network to dynamically decide whether to remember past context or reset its memory at semantic object boundaries.

---

## 3. Computational & Memory Comparison

| Architecture Family | Global Receptive Field? | Time Complexity | KV-Cache Memory per Step | Edge Real-Time Feasibility |
| :--- | :---: | :---: | :---: | :---: |
| **Standard 2D Conv** | No (Local $3\times 3$) | $\mathcal{O}(N)$ | None (Feedforward) | Native |
| **Self-Attention (ViT)** | Yes (All-to-All) | $\mathcal{O}(N^2)$ | $\mathcal{O}(N \cdot D)$ | Prohibitive at high resolutions |
| **Visual SSM (VMamba)** | **Yes (All-to-All via SS2D)** | **$\mathcal{O}(N)$** | **$\mathcal{O}(D)$ (Constant $O(1)$ state)** | **Optimal for edge real-time** |

---

## 4. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SS2DBlock(nn.Module):
    """
    2D Selective Scan (SS2D) block for Visual Mamba architectures.
    Unrolls 2D feature maps along 4 spatial trajectories, processes them
    with linear SSM recurrences, and merges the reconstructed feature maps.
    """
    def __init__(self, d_model: int = 96, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # In-projection to expand channel capacity
        self.in_proj = nn.Linear(d_model, d_model * 2)
        
        # Learnable SSM continuous parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Dynamic parameter predictors
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Out-projection
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1) # [B, H, W, C]
        
        # 1. Generate 4 scanning directions
        # Dir 1: Row-major forward
        s1 = x_flat.reshape(b, h * w, c)
        # Dir 2: Row-major backward
        s2 = torch.flip(s1, dims=[1])
        # Dir 3: Col-major forward
        s3 = x_flat.transpose(1, 2).reshape(b, w * h, c)
        # Dir 4: Col-major backward
        s4 = torch.flip(s3, dims=[1])
        
        # Stack 4 scanning streams: [4, B, L, C]
        streams = torch.stack([s1, s2, s3, s4], dim=0)
        
        # 2. Linear SSM selective scan (simplified vectorized recurrent step)
        # Project inputs to dynamic B, C, Delta
        u = streams.mean(dim=0) # [B, L, C]
        delta_logits = self.dt_proj(u)
        delta = F.softplus(delta_logits) # [B, L, C]
        
        # Discretize A: exp(delta * -exp(A_log))
        a = -torch.exp(self.A_log) # [C, D]
        
        # Scan accumulation across sequence L
        hidden = torch.zeros(b, c, self.d_state, device=x.device)
        ys = []
        for t in range(h * w):
            xt = u[:, t, :] # [B, C]
            dt = delta[:, t, :].unsqueeze(-1) # [B, C, 1]
            a_bar = torch.exp(dt * a) # [B, C, D]
            b_bar = dt # [B, C, 1]
            hidden = a_bar * hidden + b_bar * xt.unsqueeze(-1)
            yt = (hidden.sum(dim=-1) + xt * self.D)
            ys.append(yt)
            
        y = torch.stack(ys, dim=1).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x + y
```

---

## 5. Models in the Vault Utilizing Visual State-Space Models

- **[[architectures/backbones-and-edge-efficiency/vmamba|VMamba]]**: Landmark 2D selective scan visual backbone.
- **[[architectures/backbones-and-edge-efficiency/mambavision|MambaVision]]**: Hybrid visual state-space and transformer foundation backbone.
- **[[architectures/backbones-and-edge-efficiency/swin-transformer|Swin Transformer]]**: Evaluated as the classical shifted-window attention counterpart to global SSMs.

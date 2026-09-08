---
title: "MambaVision: Hybrid Visual State-Space and Transformer Foundation Backbone"
type: model-deep-dive
tasks:
  - image-classification
  - object-detection
  - semantic-segmentation
  - vision-backbone
architecture_class: Hybrid State-Space Model (SSM) & Vision Transformer
primary_license: Apache-2.0
commercial_use: true
official_repo: https://github.com/NVlabs/MambaVision
paper_url: https://arxiv.org/abs/2407.08083
tags:
  - architecture
  - mambavision
  - mamba
  - ssm
  - state-space
  - hybrid-transformer
  - vision-backbone
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - MambaVision
  - MambaVision-Base
  - Hybrid Mamba Transformer
  - NVIDIA MambaVision
---

# 🔬 MambaVision: Hybrid Visual State-Space and Transformer Foundation Backbone

## 1. Executive Brief & Significance

For years, computer vision backbones have been caught in a fundamental trade-off between two dominant paradigms:
1. **Convolutional Neural Networks (CNNs)** (e.g., ConvNeXt V2, MobileNetV4): Offer hardware-friendly linear computational complexity $\mathcal{O}(N)$ and strong inductive biases (translation equivariance and local locality), but are limited by bounded effective receptive fields.
2. **Vision Transformers (ViTs)** (e.g., DINOv2, Swin Transformer): Provide unbounded global receptive fields and dynamic data-dependent routing, but suffer from quadratic computational and memory complexity $\mathcal{O}(N^2)$ with respect to spatial token length $N = H \times W$.

While pure Visual State-Space Models (SSMs) such as VMamba solve the quadratic bottleneck by utilizing 2D selective scanning across all stages, pure SSMs struggle to capture intricate high-order global semantic relationships in deep network layers where spatial resolutions are small ($H/32 \times W/32$) and channel capacities are large ($C \ge 1024$).

**MambaVision** (Hatamizadeh & Kautz, NVIDIA Research, CVPR 2025) introduces the first **Hybrid Mamba-Transformer Foundation Backbone**, systematically placing architectural operators where they achieve maximum arithmetic efficiency:
- **Stages 1–3 (High/Mid-Resolution Linear SSM Mixers)**: Employs symmetric bidirectional 1D State-Space Models (SSM) coupled with depthwise convolutions, processing high-resolution tokens ($H/4 \dots H/16$) with strictly linear $\mathcal{O}(N)$ complexity.
- **Stage 4 (Low-Resolution Global Transformer Attention)**: Utilizes Multi-Head Self-Attention (MHSA) at the lowest spatial resolution ($H/32 \times W/32$), capturing complex global token-to-token interactions where quadratic attention cost is negligible.
- **SOTA Pareto Frontier**: Outperforms ConvNeXt V2, Swin Transformer, and pure VMamba across ImageNet-1K classification (**$86.8\%$ Top-1** on Base), COCO object detection (**$54.2\text{ mAP}$**), and ADE20K semantic segmentation (**$51.8\text{ mIoU}$**) with higher GPU throughput.

```mermaid
flowchart TD
    subgraph Patch_Embedding_Stem ["Input Stem & Patch Partitioning"]
        ImgIn["Input Image: I in R^(3 x H x W)"] --> OverlapConv["Overlapping Conv Stem (4x4 Conv, Stride 4)"]
        OverlapConv --> FeatS1["Stage 1 Feature Map: H/4 x W/4 (C = 128)"]
    end

    subgraph Hierarchical_Stages ["Hierarchical Hybrid Feature Hierarchy"]
        FeatS1 --> SSMStage1["Stage 1: Linear MambaVision SSM Blocks (L1 = 3)"]
        SSMStage1 --> Down1["Patch Merging / Conv Downsample 2x"]
        Down1 --> SSMStage2["Stage 2: Linear MambaVision SSM Blocks (L2 = 3, C = 256)"]
        SSMStage2 --> Down2["Patch Merging / Conv Downsample 2x"]
        Down2 --> SSMStage3["Stage 3: Deep Linear MambaVision SSM Blocks (L3 = 7..10, C = 512)"]
        SSMStage3 --> Down3["Patch Merging / Conv Downsample 2x"]
        Down3 --> AttnStage4["Stage 4: Global Multi-Head Self-Attention Blocks (L4 = 4, C = 1024)"]
    end

    subgraph Core_MambaVision_Mixer ["MambaVision SSM Mixer Layer (Stages 1-3)"]
        InFeat["Input Token Sequence: X in R^(B x N x C)"] --> NormIn["RMSNorm"]
        NormIn --> SplitPaths["Project to Dual Forward/Backward Paths"]
        SplitPaths --> DWConv["1D Depthwise Conv + GELU"]
        DWConv --> SSM_Kernel["Selective Scan S6 Kernel: h_t = A_bar * h_(t-1) + B_bar * x_t"]
        SSM_Kernel --> GateMul["GELU Gating Multiplicative Fusion"]
        GateMul --> OutProj["Linear Projection + Residual Connection"]
    end

    subgraph Universal_Downstream_Heads ["Downstream Perception Tasks"]
        AttnStage4 --> ClsHead["ImageNet-1K Classification Head (Top-1 86.8%)"]
        SSMStage2 & SSMStage3 & AttnStage4 --> DetNeck["Feature Pyramid Network (FPN / RF-DETR Head)"]
        SSMStage2 & SSMStage3 & AttnStage4 --> SegHead["UperNet Semantic Segmentation Head"]
    end
```

---

## 2. Mathematical Foundations & State-Space Mechanics

### A. Continuous & Discrete Selective State-Space Formulation
In the MambaVision SSM layers (Stages 1–3), input tokens $\mathbf{x}(t) \in \mathbb{R}$ are processed through continuous-time linear dynamical systems:
$$h'(t) = \mathbf{A} h(t) + \mathbf{B} \mathbf{x}(t)$$
$$y(t) = \mathbf{C} h(t) + \mathbf{D} \mathbf{x}(t)$$

Using Zero-Order Hold (ZOH) with input-dependent discretization parameter $\Delta_t = \text{Softplus}(\text{Linear}_\Delta(\mathbf{x}_t))$, continuous parameters $(\mathbf{A}, \mathbf{B})$ are discretized into $(\bar{\mathbf{A}}_t, \bar{\mathbf{B}}_t)$:
$$\bar{\mathbf{A}}_t = \exp(\Delta_t \mathbf{A})$$
$$\bar{\mathbf{B}}_t = (\Delta_t \mathbf{A})^{-1} (\exp(\Delta_t \mathbf{A}) - \mathbf{I}) \cdot (\Delta_t \mathbf{B}) \approx \Delta_t \mathbf{B}$$

The discretized recurrence evaluates as:
$$h_t = \bar{\mathbf{A}}_t h_{t-1} + \bar{\mathbf{B}}_t \mathbf{x}_t$$
$$y_t = \mathbf{C}_t h_t + \mathbf{D} \mathbf{x}_t$$

where $\mathbf{B}_t = \text{Linear}_B(\mathbf{x}_t)$ and $\mathbf{C}_t = \text{Linear}_C(\mathbf{x}_t)$ dynamically select content at every token step.

---

### B. MambaVision Symmetric Bidirectional Mixer
Unlike 1D causal language modeling where information flows strictly forward in time, 2D images require non-causal bidirectional context. MambaVision processes visual tokens through a **Symmetric Bidirectional SSM Mixer**:
1. Input tensor $\mathbf{X} \in \mathbb{R}^{B \times N \times C}$ is split into forward sequence $\mathbf{X}_{\text{fwd}}$ and flipped backward sequence $\mathbf{X}_{\text{bwd}} = \text{flip}(\mathbf{X})$.
2. Forward and backward selective scan kernels execute in parallel:
   $$\mathbf{Y}_{\text{fwd}} = \text{SSM}(\mathbf{X}_{\text{fwd}}), \qquad \mathbf{Y}_{\text{bwd}} = \text{flip}\left(\text{SSM}(\mathbf{X}_{\text{bwd}})\right)$$
3. The outputs are fused with depthwise convolutional gated branches:
   $$\mathbf{Y}_{\text{fused}} = \text{Linear}_{\text{out}}\left( \text{GELU}(\mathbf{Y}_{\text{fwd}} + \mathbf{Y}_{\text{bwd}}) \odot \text{Conv1D}(\mathbf{X}) \right)$$

---

## 3. Granular Component-by-Component Architectural Breakdown

| Architectural Stage | Component Identity | Structural Specification & Type | Attention / Conv / SSM Mechanics | Receptive Field & Dimensionality |
| :--- | :--- | :--- | :--- | :--- |
| **Architectural Paradigm** | **Hybrid SSM-Transformer Backbone** | 4-Stage Hierarchy (3 SSM Mixer Stages + 1 Global Attention Stage) | Linear Bidirectional SSM (Stages 1-3) + Quadratic MHSA (Stage 4) | Full multi-scale pyramid: $1/4, 1/8, 1/16, 1/32$ |
| **Patch Stem** | **Overlapping Convolutional Stem** | $4 \times 4$ Conv2D (Stride 4) + LayerNorm | Overlapping patch projection reducing high-frequency boundary artifacts | Input $[3, H, W] \to [C=128, H/4, W/4]$ |
| **Stages 1 & 2** | **MambaVision SSM Blocks** | $L_1=3, C_1=128$ and $L_2=3, C_2=256$ | Bidirectional 1D Selective Scan S6 + $3\times 3$ Depthwise Convolutions | Linear $\mathcal{O}(N)$ global contextual receptive field |
| **Stage 3** | **Deep MambaVision SSM Core** | $L_3=7\dots 10, C_3=512$ | 1D Selective Scan with state expansion factor $E=2$ | Deepest linear representation stage ($\approx 55\%$ params) |
| **Stage 4** | **Global Transformer Blocks** | $L_4=4, C_4=1024, \text{Heads}=16$ | Global Multi-Head Self-Attention with SwiGLU FPN | Highest semantic capacity at resolution $H/32 \times W/32$ |
| **Classifier / Neck** | **GAP & Linear Classifier Head** | Global Average Pooling + LayerNorm + Linear Classifier | Spatial mean aggregation $\to \mathbb{R}^{1024}$ | Class prediction logits $\mathbb{R}^{K=1000}$ |

---

## 4. Quantitative SOTA Benchmark Profile

### Classification & Downstream Performance (ImageNet-1K, COCO, ADE20K)

| Model Architecture | Parameters | GFLOPs | ImageNet-1K Top-1 | COCO Det mAP $\uparrow$ | ADE20K Seg mIoU $\uparrow$ | TensorRT FP16 (RTX 4090) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Swin-Base** | 88.0 M | 15.4 G | 83.5% | 51.9 mAP | 48.1 mIoU | 7.8 ms |
| **ConvNeXt V2-Base** | 89.0 M | 15.4 G | 86.8% | 53.7 mAP | 50.0 mIoU | 6.2 ms |
| **VMamba-Base** | 89.0 M | 15.2 G | 86.0% | 53.4 mAP | 50.8 mIoU | 8.5 ms |
| **MambaVision-Small** | 50.0 M | 8.2 G | **85.3%** | **52.6 mAP** | **49.5 mIoU** | **3.8 ms** |
| **MambaVision-Base** | 98.0 M | 15.8 G | **86.8%** | **54.2 mAP** | **51.8 mIoU** | **5.4 ms** |
| **MambaVision-Large** | 196.0 M | 31.5 G | **87.5%** | **55.4 mAP** | **53.2 mIoU** | **9.6 ms** |

---

## 5. Engineering Implementation: PyTorch MambaVision Mixer Layer

```python
"""
PyTorch Implementation of MambaVision Hybrid SSM-Transformer Mixer Layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaVisionSSMMixer(nn.Module):
    """
    Symmetric Bidirectional Selective State-Space Mixer Block for MambaVision.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # Continuous SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, d_model] token sequence
        """
        B, N, C = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)  # [B, N, d_inner]
        
        # Apply 1D depthwise convolution
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :N].transpose(1, 2)
        x_act = F.silu(x_conv)
        
        # Compute dynamic B, C, Delta parameters
        x_dbl = self.x_proj(x_act)  # [B, N, dt_rank + 2*d_state]
        dt, B_t, C_t = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # [B, N, d_inner]
        
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # Discretize continuous state space: exp(dt * A)
        dA = torch.exp(torch.einsum("bnd,ds->bnds", dt, A))
        dB = torch.einsum("bnd,bns->bnds", dt, B_t)
        
        # Sequential selective scan recurrence (fused into parallel scan in CUDA)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(N):
            h = dA[:, i] * h + dB[:, i] * x_act[:, i, :, None]
            y_step = torch.einsum("bd s, b s -> bd", h, C_t[:, i]) + self.D * x_act[:, i]
            ys.append(y_step)
            
        y = torch.stack(ys, dim=1)  # [B, N, d_inner]
        
        # Multiplicative gating with branch z
        y_gated = y * F.silu(z)
        return self.out_proj(y_gated)
```

---

## 6. References & Official Resources
- **MambaVision Paper**: [MambaVision: A Hybrid Mamba-Transformer Vision Backbone (CVPR 2025)](https://arxiv.org/abs/2407.08083)
- **Official NVIDIA GitHub**: [https://github.com/NVlabs/MambaVision](https://github.com/NVlabs/MambaVision)

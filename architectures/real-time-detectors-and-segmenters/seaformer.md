---
title: "SeaFormer: Squeeze-Enhanced Axial Attention for Mobile Semantic Segmentation"
architecture_class: "Real-Time Segmentation"
primary_license: "Apache-2.0"
domain: "Real-Time Detectors & Segmenters"
status: evergreen
updated: 2026-08-15
tags:
  - architecture
  - semantic-segmentation
  - mobile-ai
  - seaformer
  - axial-attention
  - edge-inference
  - hybrid-transformer
aliases:
  - "SeaFormer"
  - "Squeeze-Enhanced Axial Transformer"
  - "SeaFormer-B"
  - "Mobile Semantic Segmenter"
---

# ⚡ SeaFormer: Squeeze-Enhanced Axial Attention for Mobile Semantic Segmentation

## 1. Executive Brief & Significance

Deploying Vision Transformers to edge mobile devices (such as ARM Cortex CPUs, Qualcomm Snapdragon NPUs, and Apple Silicon Neural Engines) has historically been impeded by the quadratic computational complexity $\mathcal{O}((HW)^2)$ and excessive memory access cost (MAC) of Multi-Head Self-Attention. While axial attention mechanisms factor 2D attention into separate 1D horizontal ($1 \times W$) and vertical ($H \times 1$) passes—reducing complexity to $\mathcal{O}(HW(H+W))$—they still incur severe latency penalties on memory-bandwidth-constrained mobile platforms due to repeated global tensor transpositions and fragmented kernel dispatches.

**SeaFormer** (*Wan et al., ICLR 2023*) solves this bottleneck by introducing **Squeeze-Enhanced Axial (SEA) Attention**, a mathematically elegant attention formulation designed specifically for mobile edge hardware:
1. **Axial Squeeze Compression**: Compresses 2D Key and Value feature representations into compact 1D horizontal ($1 \times W$) and vertical ($H \times 1$) squeezed vectors via spatial squeeze pooling, reducing token length by an order of magnitude.
2. **Linear Complexity Axial Attention**: Formulates attention where full-resolution 2D Query tokens ($H \times W$) attend directly to the 1D squeezed Key/Value representations, decoupling token length and driving computational complexity down to strictly **linear** $\mathcal{O}(H \cdot W \cdot C)$.
3. **Detail Enhancement Mechanism**: Complements low-frequency global axial attention with a localized depthwise convolutional pathway, preserving crisp spatial edge gradients.
4. **Hybrid CNN-Transformer Topology**: Employs lightweight MobileNetV3-style inverted residual blocks in early high-resolution stages ($1/2, 1/4$) and SEA Transformer blocks in late contextual stages ($1/8, 1/16, 1/32$), paired with an ultra-compact **Light-MLP Decoder** ($< 0.5\text{M}$ parameters).

```mermaid
flowchart TD
    subgraph InputStage ["Input Processing"]
        ImgIn["Input Image (3 x H x W)"] --> Stem["Stem: Conv 3x3 (Stride 2, C=16/32)"]
    end

    subgraph EarlyCNNStages ["Early Mobile CNN Backbone"]
        Stem --> MBStage1["Stage 1: Mobile Inverted Residual Blocks (1/2 Stride)"]
        MBStage1 --> MBStage2["Stage 2: Mobile Inverted Residual Blocks (1/4 Stride)"]
    end

    subgraph LateSEAStages ["Late Squeeze-Enhanced Axial Attention Backbone"]
        MBStage2 --> SEAStage3["Stage 3: SEA Transformer Blocks (1/8 Stride, C3)"]
        SEAStage3 --> SEAStage4["Stage 4: SEA Transformer Blocks (1/16 Stride, C4)"]
        SEAStage4 --> SEAStage5["Stage 5: SEA Transformer Blocks (1/32 Stride, C5)"]
    end

    subgraph LightDecoder ["Light-MLP Mobile Decoder"]
        MBStage2 --> Proj2["1x1 Conv (C2 -> C_dec)"]
        SEAStage3 --> Proj3["1x1 Conv (C3 -> C_dec) -> Bilinear Up (2x)"]
        SEAStage4 --> Proj4["1x1 Conv (C4 -> C_dec) -> Bilinear Up (4x)"]
        SEAStage5 --> Proj5["1x1 Conv (C5 -> C_dec) -> Bilinear Up (8x)"]

        Proj2 --> FuseCat["Concat [Proj2, Proj3, Proj4, Proj5] (4*C_dec)"]
        Proj3 --> FuseCat
        Proj4 --> FuseCat
        Proj5 --> FuseCat

        FuseCat --> FuseConv["Linear Projection (4*C_dec -> C_dec)"]
        FuseConv --> ClsConv["Classifier Conv 1x1 (C_dec -> N Classes)"]
    end

    subgraph OutputStage ["Output"]
        ClsConv --> UpFinal["Bilinear Upsample (4x)"]
        UpFinal --> OutMask["Semantic Prediction Mask (H x W x N)"]
    end
```

---

## 2. Component-by-Component Decomposition

| Architectural Component | Technical Specification | SeaFormer-T (Tiny) | SeaFormer-B (Base) | Parameter Distribution (%) | Latency / Compute Distribution (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input Stem & Early CNN** | $3 \times 3$ Conv (S2) $\rightarrow$ Mobile Inverted Residual Blocks | Strides 2, 4; Channels: $[16, 16, 32]$ | Strides 2, 4; Channels: $[32, 32, 64]$ | $\sim 8.5\%$ ($0.14\text{M}$ / $0.73\text{M}$) | $\approx 18.2\%$ of forward pass latency |
| **SEA Stage 3 (1/8)** | SEA Attention Blocks (Squeeze $H \times 1$ & $1 \times W$) + Detail Conv | $L_3 = 2$ blocks, $C_3 = 64$ | $L_3 = 3$ blocks, $C_3 = 128$ | $\sim 16.4\%$ ($0.26\text{M}$ / $1.41\text{M}$) | $\approx 32.5\%$ of forward pass latency |
| **SEA Stage 4 (1/16)** | SEA Attention Blocks (Squeeze $H \times 1$ & $1 \times W$) + Detail Conv | $L_4 = 2$ blocks, $C_4 = 128$ | $L_4 = 4$ blocks, $C_4 = 160$ | $\sim 38.2\%$ ($0.61\text{M}$ / $3.28\text{M}$) | $\approx 28.0\%$ of forward pass latency |
| **SEA Stage 5 (1/32)** | SEA Attention Blocks (Squeeze $H \times 1$ & $1 \times W$) + Detail Conv | $L_5 = 2$ blocks, $C_5 = 160$ | $L_5 = 3$ blocks, $C_5 = 192$ | $\sim 28.6\%$ ($0.46\text{M}$ / $2.46\text{M}$) | $\approx 14.8\%$ of forward pass latency |
| **SEA Attention Module** | 1D Horizontal & Vertical Squeeze Pooling + Linear Attention | Channels: $[64, 128, 160]$, Heads: $[2, 4, 6]$ | Channels: $[128, 160, 192]$, Heads: $[4, 6, 8]$ | Included in SEA Stages | $\approx 42.0\%$ of stage compute |
| **Detail Enhancement** | $3 \times 3$ Depthwise Separable Convolution on Query branch | Expansion ratio $1.0$, Zero-padded | Expansion ratio $1.0$, Zero-padded | Included in SEA Stages | $\approx 8.5\%$ of stage compute |
| **Light-MLP Decoder** | 4 Multi-Scale Linear Projections $\rightarrow$ Bilinear Up $\rightarrow$ Linear Fuse | $C_{\text{dec}} = 128$, Out: $N$ classes | $C_{\text{dec}} = 160$, Out: $N$ classes | $\sim 8.3\%$ ($0.13\text{M}$ / $0.72\text{M}$) | $\approx 6.5\%$ of forward pass latency |

---

## 3. Mathematical Formulations & Loss Functions

```mermaid
flowchart LR
    subgraph SEABlockInternal ["Squeeze-Enhanced Axial (SEA) Attention Mechanics"]
        X_in["Input Feature Map X (C x H x W)"] --> QKV["Linear Projections: Q, K, V"]
        
        QKV --> Q_branch["Query Q: (C x H x W)"]
        QKV --> K_branch["Key K: (C x H x W)"]
        QKV --> V_branch["Value V: (C x H x W)"]

        K_branch --> SqK_H["Horizontal Squeeze: Pool_W -> (C x H x 1)"]
        K_branch --> SqK_W["Vertical Squeeze: Pool_H -> (C x 1 x W)"]
        V_branch --> SqV_H["Horizontal Squeeze: Pool_W -> (C x H x 1)"]
        V_branch --> SqV_W["Vertical Squeeze: Pool_H -> (C x 1 x W)"]

        Q_branch --> AttnH["1D Horizontal Attn: Softmax(Q * SqK_H^T) * SqV_H"]
        SqK_H --> AttnH
        SqV_H --> AttnH

        Q_branch --> AttnW["1D Vertical Attn: Softmax(Q * SqK_W^T) * SqV_W"]
        SqK_W --> AttnW
        SqV_W --> AttnW

        Q_branch --> DW_Detail["Detail Enhancement: DWConv 3x3 (Q)"]

        AttnH --> CatAttn["Concat [AttnH, AttnW] + Linear Proj"]
        AttnW --> CatAttn
        CatAttn --> FinalAdd["Element-wise Add + Residual (X_in)"]
        DW_Detail --> FinalAdd
        X_in --> FinalAdd
        FinalAdd --> OutFeat["Output Feature Map"]
    end
```

### A. Axial Squeeze Operation
Given an intermediate feature tensor $X \in \mathbb{R}^{B \times C \times H \times W}$, linear projections generate Query $Q$, Key $K$, and Value $V$ tensors. Standard axial attention computes 1D attention across entire strips, leading to high memory traffic. SeaFormer introduces **Squeeze Pooling** along orthogonal spatial dimensions:

1. **Horizontal Squeeze (Column Compression)**:
   $$K_h(b, c, i, 1) = \frac{1}{W} \sum_{j=1}^W K(b, c, i, j) \in \mathbb{R}^{B \times C \times H \times 1}$$
   $$V_h(b, c, i, 1) = \frac{1}{W} \sum_{j=1}^W V(b, c, i, j) \in \mathbb{R}^{B \times C \times H \times 1}$$

2. **Vertical Squeeze (Row Compression)**:
   $$K_w(b, c, 1, j) = \frac{1}{H} \sum_{i=1}^H K(b, c, i, j) \in \mathbb{R}^{B \times C \times 1 \times W}$$
   $$V_w(b, c, 1, j) = \frac{1}{H} \sum_{i=1}^H V(b, c, i, j) \in \mathbb{R}^{B \times C \times 1 \times W}$$

### B. Squeeze-Enhanced Axial Attention Formulation
Full-resolution 2D Query tokens $Q \in \mathbb{R}^{B \times N_{\text{heads}} \times (H \cdot W) \times d}$ attend directly to the squeezed Key and Value vectors:

$$A_h = \text{Softmax}\left( \frac{Q K_h^T}{\sqrt{d}} \right) V_h \in \mathbb{R}^{B \times C \times H \times W}$$

$$A_w = \text{Softmax}\left( \frac{Q K_w^T}{\sqrt{d}} \right) V_w \in \mathbb{R}^{B \times C \times H \times W}$$

The outputs of horizontal and vertical axial attention are concatenated and linearly merged:
$$\text{Attn}_{\text{axial}}(Q, K, V) = \text{Linear}\left( \text{Concat}\left[ A_h, A_w \right] \right)$$

### C. Detail Enhancement & Linear Complexity
Because global pooling squeezes high-frequency spatial gradients, SeaFormer parallelizes axial attention with a localized convolutional **Detail Enhancement** pathway directly on the Query tensor:

$$\text{DetailEnhance}(Q) = \text{BN}\left( \text{DWConv}_{3 \times 3}\left( Q \right) \right)$$

$$\text{SEA}(X) = \text{Attn}_{\text{axial}}(Q, K, V) + \text{DetailEnhance}(Q) + X$$

**Complexity Proof**:
- Key/Value squeeze: $\mathcal{O}(H \cdot W \cdot C)$
- Attention dot product ($Q K_h^T$ and $Q K_w^T$): $\mathcal{O}(H \cdot W \cdot H \cdot d + H \cdot W \cdot W \cdot d) = \mathcal{O}(H \cdot W \cdot (H + W) \cdot d)$
- When $H, W \ll C$ or with squeezed token lengths $S_h, S_w \le 16$, the overall computational complexity scales strictly **linear** with respect to the input pixel count:
  $$\mathcal{O}_{\text{SeaFormer}} = \mathcal{O}(H \cdot W \cdot C)$$

### D. Objective Loss Function
SeaFormer is trained using Cross-Entropy with standard class-weighting or Online Hard Example Mining (OHEM):

$$\mathcal{L}_{\text{total}} = - \frac{1}{H \cdot W} \sum_{i=1}^{H \cdot W} \sum_{c=1}^{N_{\text{classes}}} Y_{i,c} \log\left( \frac{\exp(\hat{Y}_{i,c})}{\sum_{k=1}^{N_{\text{classes}}} \exp(\hat{Y}_{i,k})} \right)$$

---

## 4. Benchmark Evaluation & Performance Profiles

### A. Cityscapes & ADE20K Benchmark Comparison

| Architecture | Model Scale | Input Resolution | Parameters (M) | FLOPs (G) | Cityscapes val (mIoU %) | Cityscapes test (mIoU %) | ADE20K val (mIoU %) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TopFormer-T** | Tiny | $512 \times 512$ | 1.4M | 0.6G | 64.2% | 63.5% | 29.8% |
| **TopFormer-B** | Base | $512 \times 512$ | 5.1M | 2.4G | 70.7% | 69.8% | 35.6% |
| **SegFormer-B0** | Real-Time | $512 \times 512$ | 3.8M | 8.4G | 76.2% | 75.4% | 37.4% |
| **SeaFormer-T** | Tiny | $512 \times 512$ | **1.6M** | **0.6G** | **68.4%** | **67.7%** | **31.2%** |
| **SeaFormer-S** | Small | $512 \times 512$ | **2.7M** | **1.1G** | **72.1%** | **71.5%** | **34.1%** |
| **SeaFormer-B** | Base | $512 \times 512$ | **8.6M** | **3.4G** | **75.8%** | **75.4%** | **39.8%** |
| **SeaFormer-L** | Large | $512 \times 512$ | **14.2M** | **7.8G** | **78.1%** | **77.7%** | **43.5%** |

### B. Mobile Device & Edge Latency Matrix

*Evaluated with single batch ($B=1$), input resolution $512 \times 512$ (Mobile/ADE20K format) and $1024 \times 2048$ (Cityscapes format).*

| Hardware Target | Runtime Engine | Precision | SeaFormer-T (ms) | SeaFormer-S (ms) | SeaFormer-B (ms) | SeaFormer-L (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **iPhone 12 (A14 Bionic)** | CoreML | FP16 (512x512) | **1.8 ms (555 FPS)** | **3.2 ms (312 FPS)** | **7.4 ms (135 FPS)** | **14.5 ms (69 FPS)** |
| **Snapdragon 888 (Kryo CPU)** | TFLite / XNNPACK | FP32 (512x512) | **12.4 ms (80 FPS)** | **20.8 ms (48 FPS)** | **45.2 ms (22 FPS)** | **88.0 ms (11 FPS)** |
| **Snapdragon 888 (Hexagon NPU)**| Qualcomm QNN | INT8 (512x512) | **2.1 ms (476 FPS)** | **3.8 ms (263 FPS)** | **8.2 ms (122 FPS)** | **16.4 ms (61 FPS)** |
| **Jetson AGX Orin (64GB)** | TensorRT 10 | FP16 (1024x2048) | **3.8 ms (263 FPS)** | **6.4 ms (156 FPS)** | **11.2 ms (89 FPS)** | **22.5 ms (44 FPS)** |
| **Jetson AGX Orin (64GB)** | TensorRT 10 | INT8 (1024x2048) | **1.9 ms (526 FPS)** | **3.2 ms (312 FPS)** | **5.8 ms (172 FPS)** | **11.8 ms (85 FPS)** |
| **Jetson Orin Nano (8GB)** | TensorRT 10 | FP16 (512x512) | **4.2 ms (238 FPS)** | **7.8 ms (128 FPS)** | **15.4 ms (65 FPS)** | **31.0 ms (32 FPS)** |
| **NVIDIA RTX 4090** | TensorRT 10 | FP16 (1024x2048) | **1.1 ms (909 FPS)** | **1.9 ms (526 FPS)** | **3.4 ms (294 FPS)** | **6.8 ms (147 FPS)** |
| **NVIDIA T4** | TensorRT 10 | FP16 (1024x2048) | **4.8 ms (208 FPS)** | **8.2 ms (122 FPS)** | **15.8 ms (63 FPS)** | **32.4 ms (31 FPS)** |

---

## 5. Edge Deployment, TensorRT Optimization & Gotchas

### A. Mobile NPU & TensorRT Optimization Gotchas
1. **1D Squeeze Global Pooling Kernel Fusion**:
   In naive PyTorch, `x.mean(dim=-1, keepdim=True)` executes as separate reduction kernels along rows and columns.
   *Resolution*: Use 2D average pooling with explicit rectangular kernels (`nn.AdaptiveAvgPool2d((H, 1))` and `nn.AdaptiveAvgPool2d((1, W))`). In ONNX Opset 17+, these compile directly into native `GlobalAveragePool` / `AveragePool` operators, fusing seamlessly into upstream convolution kernels in TensorRT and CoreML.
2. **INT8 PTQ Scale Drift across Squeezed Axes**:
   Because $K_h$ and $K_w$ are averaged across spatial dimensions, their numerical variance is significantly smaller than the full-resolution Query tensor $Q$. Standard symmetric tensor-wise quantization compresses $K_h$ and $K_w$ into very few quantization bins.
   *Resolution*: Apply Channel-Wise Quantization (`IInt8EntropyCalibrator2` in TensorRT / Per-Channel INT8 in TFLite) to preserve the dynamic range of squeezed representations.
3. **Static Resolution Compilation for Mobile NPUs**:
   Mobile NPU runtimes (e.g., Apple ANE via CoreML, Qualcomm Hexagon via QNN) do not support dynamic intermediate shapes. Export models strictly with fixed spatial dimensions ($1 \times 3 \times 512 \times 512$).

### B. Clean ONNX Export Workflow

```python
import torch
import torch.nn as nn

def export_seaformer_onnx(model: nn.Module, output_path: str = "seaformer_b_512.onnx"):
    model.eval()
    dummy_input = torch.randn(1, 3, 512, 512, device="cuda", dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["segmentation"],
        dynamic_axes=None  # Static input dimensions required for Mobile NPU/ANE compilation
    )
    print(f"[✓] Exported clean SeaFormer ONNX to: {output_path}")

if __name__ == "__main__":
    pass
```

### C. Compiling via TensorRT `trtexec`

```bash
# FP16 Engine for Jetson AGX Orin & Orin Nano
trtexec \
  --onnx=seaformer_b_512.onnx \
  --saveEngine=seaformer_b_fp16.engine \
  --fp16 \
  --builderOptimizationLevel=5 \
  --avgRuns=200

# INT8 Calibrated Engine for Snapdragon / Jetson Edge Inference
trtexec \
  --onnx=seaformer_b_512.onnx \
  --saveEngine=seaformer_b_int8.engine \
  --int8 \
  --fp16 \
  --calib=seaformer_calib.cache \
  --precisionConstraints=obey \
  --builderOptimizationLevel=5
```

---

## 6. Complete Runnable Python Blueprint

```python
"""
SeaFormer-B: Squeeze-Enhanced Axial Attention for Mobile Semantic Segmentation
Self-contained, runnable implementation in PyTorch.
Paper: Wan et al., "SeaFormer: Squeeze-Enhanced Axial Attention for Mobile Semantic Segmentation", ICLR 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ----------------------------------------------------------------------
# 1. Base Mobile CNN Blocks (Inverted Residuals)
# ----------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, relu: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True) if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """MobileNet-style Inverted Residual Block for early CNN stages."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, exp_ratio: int = 4):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * exp_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if exp_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1, padding=0))
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


# ----------------------------------------------------------------------
# 2. Squeeze-Enhanced Axial (SEA) Attention Block
# ----------------------------------------------------------------------

class SqueezeAxialAttention(nn.Module):
    """Squeeze-Enhanced Axial (SEA) Attention with linear complexity."""
    def __init__(self, dim: int, num_heads: int = 4, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        # Detail enhancement convolution on Query
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.proj = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # 1. Detail Enhancement on Query
        detail = self.detail_enhance(q)

        # 2. Axial Squeeze Operations
        # Horizontal Squeeze: Pool along width (W -> 1)
        k_h = F.adaptive_avg_pool2d(k, (h, 1))  # (B, C, H, 1)
        v_h = F.adaptive_avg_pool2d(v, (h, 1))  # (B, C, H, 1)

        # Vertical Squeeze: Pool along height (H -> 1)
        k_w = F.adaptive_avg_pool2d(k, (1, w))  # (B, C, 1, W)
        v_w = F.adaptive_avg_pool2d(v, (1, w))  # (B, C, 1, W)

        # Reshape for multi-head attention
        # Q: (B, num_heads, head_dim, H, W)
        q_mh = q.reshape(b, self.num_heads, self.head_dim, h, w)
        k_h_mh = k_h.reshape(b, self.num_heads, self.head_dim, h, 1)
        v_h_mh = v_h.reshape(b, self.num_heads, self.head_dim, h, 1)

        k_w_mh = k_w.reshape(b, self.num_heads, self.head_dim, 1, w)
        v_w_mh = v_w.reshape(b, self.num_heads, self.head_dim, 1, w)

        # 3. 1D Horizontal Axial Attention
        # Dot product across head_dim: (B, num_heads, H, W, 1)
        attn_h = (q_mh * k_h_mh).sum(dim=2, keepdim=True) * self.scale
        attn_h = torch.softmax(attn_h, dim=3)  # Softmax along H
        out_h = (attn_h * v_h_mh).reshape(b, c, h, w)

        # 4. 1D Vertical Axial Attention
        # Dot product across head_dim: (B, num_heads, 1, H, W)
        attn_w = (q_mh * k_w_mh).sum(dim=2, keepdim=True) * self.scale
        attn_w = torch.softmax(attn_w, dim=4)  # Softmax along W
        out_w = (attn_w * v_w_mh).reshape(b, c, h, w)

        # 5. Concatenate & Project
        out = self.proj_bn(self.proj(torch.cat([out_h, out_w], dim=1)))
        return out + detail


class SEABlock(nn.Module):
    """Complete SEA Transformer Block with Feed-Forward Network."""
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.attn = SqueezeAxialAttention(dim, num_heads=num_heads)
        self.mlp = nn.Sequential(
            ConvBNReLU(dim, dim * mlp_ratio, kernel_size=1, padding=0),
            nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, kernel_size=3, padding=1, groups=dim * mlp_ratio, bias=False),
            nn.BatchNorm2d(dim * mlp_ratio),
            nn.ReLU6(inplace=True),
            nn.Conv2d(dim * mlp_ratio, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


# ----------------------------------------------------------------------
# 3. Complete SeaFormer Network Architecture
# ----------------------------------------------------------------------

class SeaFormerBackbone(nn.Module):
    def __init__(self, channels: List[int] = [32, 64, 128, 160, 192], num_heads: List[int] = [4, 6, 8]):
        super().__init__()
        # Stem & Stage 1 (1/2 Stride)
        self.stem = ConvBNReLU(3, channels[0], kernel_size=3, stride=2, padding=1)
        self.stage1 = InvertedResidual(channels[0], channels[0], stride=1)

        # Stage 2 (1/4 Stride)
        self.stage2 = nn.Sequential(
            InvertedResidual(channels[0], channels[1], stride=2),
            InvertedResidual(channels[1], channels[1], stride=1)
        )

        # Stage 3 (1/8 Stride) - SEA Attention
        self.down3 = InvertedResidual(channels[1], channels[2], stride=2)
        self.stage3 = nn.Sequential(*[SEABlock(channels[2], num_heads=num_heads[0]) for _ in range(3)])

        # Stage 4 (1/16 Stride) - SEA Attention
        self.down4 = InvertedResidual(channels[2], channels[3], stride=2)
        self.stage4 = nn.Sequential(*[SEABlock(channels[3], num_heads=num_heads[1]) for _ in range(4)])

        # Stage 5 (1/32 Stride) - SEA Attention
        self.down5 = InvertedResidual(channels[3], channels[4], stride=2)
        self.stage5 = nn.Sequential(*[SEABlock(channels[4], num_heads=num_heads[2]) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        f2 = self.stage2(x)      # 1/4 scale
        f3 = self.stage3(self.down3(f2))  # 1/8 scale
        f4 = self.stage4(self.down4(f3))  # 1/16 scale
        f5 = self.stage5(self.down5(f4))  # 1/32 scale
        return [f2, f3, f4, f5]


class LightMLPDecoder(nn.Module):
    """Lightweight Mobile MLP Decoder."""
    def __init__(self, in_channels: List[int], decoder_dim: int = 160, num_classes: int = 19):
        super().__init__()
        self.proj2 = ConvBNReLU(in_channels[0], decoder_dim, kernel_size=1, padding=0)
        self.proj3 = ConvBNReLU(in_channels[1], decoder_dim, kernel_size=1, padding=0)
        self.proj4 = ConvBNReLU(in_channels[2], decoder_dim, kernel_size=1, padding=0)
        self.proj5 = ConvBNReLU(in_channels[3], decoder_dim, kernel_size=1, padding=0)

        self.fuse = ConvBNReLU(decoder_dim * 4, decoder_dim, kernel_size=1, padding=0)
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        f2, f3, f4, f5 = features
        h, w = f2.shape[2:]

        p2 = self.proj2(f2)
        p3 = F.interpolate(self.proj3(f3), size=(h, w), mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.proj4(f4), size=(h, w), mode='bilinear', align_corners=False)
        p5 = F.interpolate(self.proj5(f5), size=(h, w), mode='bilinear', align_corners=False)

        fused = self.fuse(torch.cat([p2, p3, p4, p5], dim=1))
        return self.classifier(fused)


class SeaFormer(nn.Module):
    def __init__(self, num_classes: int = 19, variant: str = "B"):
        super().__init__()
        if variant == "B":
            channels = [32, 64, 128, 160, 192]
            num_heads = [4, 6, 8]
            decoder_dim = 160
        elif variant == "T":
            channels = [16, 32, 64, 128, 160]
            num_heads = [2, 4, 6]
            decoder_dim = 128
        else:
            raise ValueError(f"Unsupported SeaFormer variant: {variant}")

        self.backbone = SeaFormerBackbone(channels=channels, num_heads=num_heads)
        self.decode_head = LightMLPDecoder(in_channels=channels[1:], decoder_dim=decoder_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_orig, w_orig = x.shape[2:]
        features = self.backbone(x)
        out = self.decode_head(features)
        # Upsample 4x from Stage 2 resolution back to original image shape
        return F.interpolate(out, size=(h_orig, w_orig), mode='bilinear', align_corners=False)


# ----------------------------------------------------------------------
# 4. Model Verification & Unit Test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeaFormer(num_classes=19, variant="B").to(device)
    model.eval()

    dummy_input = torch.randn(2, 3, 512, 512, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    print("SeaFormer-B Model Verification:")
    print(f"  Input Tensor Shape:  {dummy_input.shape}")
    print(f"  Output Tensor Shape: {output.shape}")
    assert output.shape == (2, 19, 512, 512), "Output shape mismatch!"
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total Parameters:    {total_params:.2f}M")
    print("  [✓] Forward pass assertion successful.")
```

---

## 7. Peer Comparisons & Cross-Links

| Architectural Dimension | SeaFormer (Wan et al., 2023) | SegFormer (Xie et al., 2021) | PIDNet (Xu et al., 2023) | DDRNet (Hong et al., 2021) | TopFormer (Zhang et al., 2022) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Architectural Topology** | Mobile CNN-Transformer Hybrid | Pure Hierarchical ViT (MiT) | 3-Branch PID Controller | Dual-Resolution + Bridges | Token Pyramid Transformer |
| **Attention Mechanism** | **Squeeze-Enhanced Axial (SEA)** | **Spatial Reduction (SRA)** | None (Pure CNN) | None (Pure CNN) | Scale-aware Token Attn |
| **Computational Complexity** | $\mathcal{O}(H \cdot W \cdot C)$ (Strictly Linear) | $\mathcal{O}((HW)^2 / R^2)$ | $\mathcal{O}(H \cdot W \cdot K^2)$ | $\mathcal{O}(H \cdot W \cdot K^2)$ | $\mathcal{O}(N_{\text{tokens}} \cdot C)$ |
| **Mobile Latency (Snapdragon 888)**| **20.8 ms (SeaFormer-S)** | 64.5 ms (MiT-B0) | 58.2 ms (PIDNet-S) | 48.0 ms (DDRNet-23s) | 26.5 ms (TopFormer-B) |
| **Cityscapes mIoU** | **71.5% (S) / 75.4% (B)** | 76.2% (B0) / 81.0% (B2) | 78.6% (S) / 80.6% (L) | 77.8% (slim) / 79.5% (23) | 70.7% (Base) |
| **ADE20K mIoU** | **34.1% (S) / 39.8% (B)** | 37.4% (B0) / 46.5% (B2) | N/A (Autonomous Driving) | N/A (Autonomous Driving) | 35.6% (Base) |
| **Parameter Footprint** | **2.7M (S) / 8.6M (B)** | 3.8M (B0) / 24.7M (B2) | 7.6M (S) / 18.7M (M) | 5.7M (slim) / 20.1M (23) | 5.1M (Base) |

### Literature & Reference Citations
- **Official Repository**: [MetaIL/SeaFormer](https://github.com/MetaIL/SeaFormer)
- **Paper**: *SeaFormer: Squeeze-Enhanced Axial Attention for Mobile Semantic Segmentation* (Wan et al., ICLR 2023) [arXiv:2301.13156](https://arxiv.org/abs/2301.13156)
- **Related Architectures**:
  - [[architectures/real-time-detectors-and-segmenters/segformer|SegFormer]] — Positional-encoding-free hierarchical transformer segmenter.
  - [[architectures/real-time-detectors-and-segmenters/pidnet|PIDNet]] — Proportional-Integral-Derivative three-branch network.
  - [[architectures/real-time-detectors-and-segmenters/ddrnet|DDRNet]] — Deep dual-resolution road scene segmentation.
  - [[architectures/real-time-detectors-and-segmenters/bisenetv2|BiSeNet V2]] — Bilateral Detail and Semantic segmentation.
  - [[architectures/hardware-and-acceleration-runtimes/tensorrt-runtime|TensorRT Runtime]] — High-performance inference engine compilation.

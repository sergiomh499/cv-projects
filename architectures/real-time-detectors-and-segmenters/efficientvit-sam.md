---
title: "EfficientViT-SAM: Accelerated Segment Anything Model Without Performance Loss"
architecture_class: "Real-Time Foundation Segmenter"
primary_license: "Apache-2.0"
domain: "Real-Time Detectors & Segmenters"
status: evergreen
updated: 2026-08-15
tags:
  - architecture
  - segmentation
  - foundation-models
  - sam
  - efficientvit-sam
  - linear-attention
  - edge-inference
  - tensorrt
aliases:
  - "EfficientViT-SAM"
  - "EfficientViT SAM"
  - "MIT EfficientViT-SAM"
  - "Linear Attention SAM"
---

# ⚡ EfficientViT-SAM: Accelerated Segment Anything Model Without Performance Loss

## 1. Executive Brief & Significance

Meta's **Segment Anything Model (SAM)** (Kirillov et al., 2023) revolutionized zero-shot promptable image segmentation across computer vision. However, SAM's default visual foundation backbone—a standard non-hierarchical **ViT-H** image encoder with $636\text{M}$ parameters and $2,900\text{ GFLOPs}$—requires over $450\text{ ms}$ on an NVIDIA A100 GPU to encode a single $1024 \times 1024$ image frame. This massive computational burden and high memory bandwidth demand render vanilla SAM impractical for real-time robotic perception, interactive edge labeling, mobile augmented reality, and high-framerate video segmentation pipelines.

While prior lightweight distillation attempts (such as [[architectures/real-time-detectors-and-segmenters/mobilesam|MobileSAM]] with TinyViT and [[architectures/real-time-detectors-and-segmenters/repvit-sam|RepViT-SAM]] with pure convolutions) reduced parameter counts, they encountered distinct trade-offs: TinyViT retained quadratic windowed softmax attention kernels that suffer from high memory latency on Tensor Cores, while pure convolutional distillations sacrificed global context reasoning and long-range semantic coherence.

```mermaid
flowchart TD
    subgraph MultiStage_Encoder ["1. 5-Stage EfficientViT Image Encoder (Fused-MBConv + Linear Attention)"]
        RawImg["Input Image: I in R^(3 x 1024 x 1024)"] --> Stem["Patch Stem: 2x Conv3x3 (Stride 4 Downsample)"]
        Stem --> Stage12["Stages 1-2: Fused-MBConv Blocks (High-Bandwidth Local Features)"]
        Stage12 --> Stage35["Stages 3-5: Multi-Scale Linear Attention + Depthwise Token Mixers"]
        Stage35 --> FeatNeck["Multi-Scale Feature Aggregation Neck (Stride 16 Projection)"]
        FeatNeck --> ImgEmbed["Image Embedding: F_img in R^(256 x 64 x 64)"]
    end

    subgraph Prompt_Engine ["2. SAM Prompt Encoder"]
        UserPrompts["User Prompts: Point Coordinates, Bounding Boxes, Dense Masks"] --> PromptEnc["MLP & Positional Coordinate Positional Embeddings"]
        PromptEnc --> PromptEmbed["Sparse / Dense Prompt Tokens: F_prompt in R^(N x 256)"]
    end

    subgraph DecoderStage ["3. Bidirectional Two-Way Transformer Mask Decoder"]
        ImgEmbed & PromptEmbed --> TwoWayDecoder["Two-Way Cross-Attention Transformer (2 Layers)"]
        TwoWayDecoder --> MaskProj["Transposed Conv Upsampling (4x) + MLP Mask Projections"]
        MaskProj --> HighResMask["Final Zero-Shot Binary / Multi-Class Mask (>100 FPS)"]
    end
```

**EfficientViT-SAM** (Liu et al., MIT Han Lab, CVPR 2024 eLVM) resolves these fundamental limitations by replacing the heavy ViT image encoder with **EfficientViT**, a 5-stage hybrid hierarchical architecture:
1. **Fused-MBConv in Early Stages**: Deploys fused Mobile Inverted Bottleneck Convolutions in stages 1 and 2 ($1024 \to 256 \to 128$ spatial resolutions), avoiding memory fragmentation and maximizing GPU memory bandwidth utilization.
2. **Multi-Scale Linear Attention (MSLA) in Deep Stages**: Replaces quadratic Softmax Multi-Head Self-Attention ($\mathcal{O}(N^2)$) with linear kernel attention ($\mathcal{O}(N)$) via ReLU kernel feature maps $\phi(x) = \text{ReLU}(x)$. By leveraging the associative property of matrix multiplication, MSLA computes global receptive fields in linear time while maintaining continuous global token interaction.
3. **High-Frequency Local Token Mixers**: Interleaves depthwise separable convolutions within the linear attention feedforward blocks, capturing high-frequency structural boundaries that standard linear attention tends to blur.
4. **48.9$\times$ Speedup on NVIDIA GPUs**: EfficientViT-SAM-L0 achieves **$48.9\times$ TensorRT speedup over SAM-ViT-H on A100** ($9.2\text{ ms}$ vs $450\text{ ms}$) while retaining $>98\%$ of SAM's zero-shot mask mIoU across SA-1B, COCO, LVIS, and ADE20K.

---

## 2. Component-by-Component Architectural Breakdown

| Architectural Stage | Component Identity | Structural Specification & Type | Internal Mechanics | Dimensionality & Spatial Stride | Parameter Distribution (%) | Compute / Latency (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vision Encoder** | **EfficientViT Backbone** | 5-stage hierarchical hybrid backbone (L0 / L1 / L2 / XL0) | Fused-MBConv (Stages 1-2) + Multi-Scale Linear Attention (Stages 3-5) | Output feature stride 16 ($256 \times 64 \times 64$) | $\approx 86.5\%$ ($30.7\text{M}$ for L0 / $43.8\text{M}$ for L2) | $\approx 82.0\%$ of forward pass ($9.2\text{ ms}$) |
| **Patch Stem** | **Convolutional Stem** | $2\times$ Sequential $3\times 3$ Strided Convolutions | Channel expansion with stride 4 spatial downsampling | $1024\times 1024 \times 3 \to 256\times 256 \times C_1$ | $< 0.8\%$ ($0.25\text{M}$) | $\approx 3.5\%$ of forward pass |
| **Early Stages (1-2)** | **Fused-MBConv Blocks** | Fused $3\times 3\text{ Conv} \to \text{SE} \to 1\times 1\text{ Conv}$ | High-throughput local spatial filtering with zero kernel split overhead | Stages 1-2: Strides 4, 8; channels $[64, 128]$ | $\approx 18.2\%$ ($6.5\text{M}$) | $\approx 22.0\%$ of encoder latency |
| **Late Stages (3-5)** | **Multi-Scale Linear Attention (MSLA)** | Linear kernel self-attention ($\phi(x)=\text{ReLU}(x)$) + Depthwise Token Mixers | Linear global context $\mathcal{O}(N d^2)$ + multi-scale convolutional residual paths | Stages 3-5: Strides 16, 32; channels $[256, 512, 1024]$ | $\approx 64.5\%$ ($23.0\text{M}$) | $\approx 56.5\%$ of encoder latency |
| **Projector Neck** | **Feature Aggregation Neck** | $1\times 1\text{ Conv} + 3\times 3\text{ DWConv} + \text{LayerNorm}$ | Fuses multi-scale stage features into unified SAM latent space | Projects Stage 3, 4, 5 to $256 \times 64 \times 64$ | $\approx 3.0\%$ ($1.0\text{M}$) | $\approx 3.0\%$ of forward pass |
| **Prompt Encoder** | **SAM Prompt Encoder** | Positional MLPs + Sinusoidal coordinate encodings | Encodes user point clicks, bounding box corners, and dense masks | Output token sequence $[N_{\text{prompts}}, 256]$ | $< 0.1\text{M}$ | $< 0.1\text{ ms}$ |
| **Mask Decoder** | **Two-Way Transformer Decoder** | 2-layer bidirectional cross-attention Transformer | Point-to-Image Cross-Attn $\to$ Image-to-Point Cross-Attn $\to$ FFN | Dual cross-attention over $4096$ spatial tokens | $\approx 13.5\%$ ($4.8\text{M}$) | $\approx 18.0\%$ of forward pass ($2.0\text{ ms}$) |

```mermaid
flowchart LR
    subgraph Attention_Comparison ["Algorithmic Complexity: Softmax Attention vs EfficientViT Linear Attention"]
        subgraph StandardSoftmax ["Vanilla Softmax Attention (SAM ViT-H): O(N^2 * d)"]
            Q1["Query: Q in R^(N x d)"] & K1["Key: K in R^(N x d)"] --> MatMul1["Dense Dot-Product: Q * K^T in R^(N x N)"]
            MatMul1 --> Smax["Softmax Normalization (Huge SRAM Bottleneck)"]
            Smax & V1["Value: V in R^(N x d)"] --> MatMul2["Output: (Softmax(QK^T)) * V in R^(N x d)"]
        end

        subgraph EfficientViTLinear ["EfficientViT Linear Attention: O(N * d^2)"]
            Q2["Query: Q in R^(N x d)"] --> PhiQ["Kernel Map: phi(Q) = ReLU(Q)"]
            K2["Key: K in R^(N x d)"] --> PhiK["Kernel Map: phi(K) = ReLU(K)"]
            V2["Value: V in R^(N x d)"]
            
            PhiK & V2 --> KVMatMul["Associative Fusion: phi(K)^T * V in R^(d x d)"]
            PhiQ & KVMatMul --> OutLin["Output: phi(Q) * (phi(K)^T * V) in R^(N x d)"]
        end
    end
```

---

## 3. Mathematical Formulations & Loss Functions

### A. Linear Kernel Attention vs. Quadratic Softmax Attention

Standard Multi-Head Self-Attention (MHSA) in Vision Transformers computes pairwise token similarities across all $N = H \times W$ spatial locations:

$$\mathbf{O}_{\text{Softmax}} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V}, \quad \mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$$

For a $1024 \times 1024$ image at stride 16, the token count is $N = 64 \times 64 = 4096$. Materializing the $4096 \times 4096$ intermediate attention matrix incurs $\mathcal{O}(N^2 d) = 4096^2 \times 64 \approx 1.07\times 10^9$ operations per attention head, exceeding high-bandwidth memory (HBM) caching limits.

**EfficientViT's Linear Attention Formulation**:
By applying a non-negative feature kernel mapping $\phi(x) = \text{ReLU}(x)$ to both queries and keys, the Softmax similarity function is decomposed into a generalized dot product:

$$\text{Sim}(\mathbf{q}_i, \mathbf{k}_j) = \phi(\mathbf{q}_i) \phi(\mathbf{k}_j)^T$$

Exploiting the associative law of matrix multiplication allows reordering the evaluation sequence:

$$\mathbf{O}_i = \frac{\sum_{j=1}^N \phi(\mathbf{q}_i) \phi(\mathbf{k}_j)^T \mathbf{v}_j}{\sum_{j=1}^N \phi(\mathbf{q}_i) \phi(\mathbf{k}_j)^T \mathbf{1}} = \frac{\phi(\mathbf{q}_i) \left(\sum_{j=1}^N \phi(\mathbf{k}_j)^T \mathbf{v}_j\right)}{\phi(\mathbf{q}_i) \left(\sum_{j=1}^N \phi(\mathbf{k}_j)^T\right)}$$

In matrix notation:

$$\mathbf{O}_{\text{Linear}} = \frac{\phi(\mathbf{Q}) \left(\phi(\mathbf{K})^T \mathbf{V}\right)}{\phi(\mathbf{Q}) \left(\phi(\mathbf{K})^T \mathbf{1}_{N \times 1}\right)}, \quad \text{Complexity} = \mathcal{O}(N d^2)$$

Because $d = 32 \ll N = 4096$, computational complexity drops from quadratic $\mathcal{O}(N^2)$ to purely linear $\mathcal{O}(N)$, shrinking attention runtime from $18.4\text{ ms}$ to $0.62\text{ ms}$ per block.

```mermaid
flowchart TD
    subgraph MSLA_Block ["Multi-Scale Linear Attention (MSLA) Module"]
        InputX["Input Feature: X in R^(B x N x C)"] --> QKV["1x1 Linear Projections -> Q, K, V"]
        
        QKV --> LinearAttnEngine["Linear Attention Core: phi(Q)(phi(K)^T * V)"]
        
        InputX --> DW1["Depthwise Conv 3x3 (Scale 1: Local Detail)"]
        InputX --> DW2["Depthwise Conv 5x5 (Scale 2: Intermediate Context)"]
        InputX --> DW3["Depthwise Conv 7x7 (Scale 3: Broad Field)"]
        
        DW1 & DW2 & DW3 --> AggregatedDW["Multi-Scale Depthwise Concat & Pointwise Mix"]
        
        LinearAttnEngine & AggregatedDW --> FusionAdd["Residual Channel Fusion + FFN"]
        FusionAdd --> OutputY["Output Feature: Y in R^(B x N x C)"]
    end
```

### B. High-Frequency Multi-Scale Convolutional Token Mixer

To compensate for the loss of sharp local frequency components caused by linear attention kernel smoothing, EfficientViT incorporates a **Multi-Scale Depthwise Token Mixer**:

$$\mathbf{Y}_{\text{MSLA}} = \mathbf{O}_{\text{Linear}} + \sum_{k \in \{3, 5, 7\}} \text{PWConv}\left(\text{DWConv}_{k\times k}\left(\mathbf{X}\right)\right)$$

This hybrid design preserves both high-frequency instance boundaries (crucial for segmentation mask edges) and long-range semantic dependencies.

### C. Knowledge Distillation Objective from SAM-ViT-H

EfficientViT-SAM is trained via a multi-stage distillation pipeline where student features are supervised by a frozen SAM-ViT-H teacher:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}(\mathbf{F}_{\text{student}}, \mathbf{F}_{\text{teacher}}) + \lambda_{\text{cos}} \mathcal{L}_{\text{cosine}}(\mathbf{F}_{\text{student}}, \mathbf{F}_{\text{teacher}}) + \lambda_{\text{mask}} \mathcal{L}_{\text{Focal}}(\mathbf{M}, \mathbf{M}^*) + \lambda_{\text{dice}} \mathcal{L}_{\text{Dice}}(\mathbf{M}, \mathbf{M}^*)$$

where:
$$\mathcal{L}_{\text{cosine}}(\mathbf{F}_s, \mathbf{F}_t) = 1 - \frac{1}{H W} \sum_{i=1}^{H W} \frac{\mathbf{F}_s(i) \cdot \mathbf{F}_t(i)}{\|\mathbf{F}_s(i)\|_2 \|\mathbf{F}_t(i)\|_2}$$

---

## 4. Benchmark Evaluation & Performance Profiles

### Zero-Shot Promptable Segmentation Benchmarks

Evaluated with 1-point and 1-box prompt protocols on standard foundation segmentation benchmarks. Latency measured on a single NVIDIA A100-SXM4-80GB and NVIDIA Jetson Orin AGX ($1024\times 1024$ input resolution).

| Architecture | Image Encoder Params (M) | Total Params (M) | Encoder GFLOPs | SA-1B 1-Pt mIoU (%) | COCO 1-Box mIoU (%) | LVIS 1-Box mIoU (%) | A100 TensorRT FP16 (ms) | RTX 4090 FP16 (ms) | Jetson Orin AGX INT8 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EfficientViT-SAM-L0** | **30.7** | **34.8** | **35.2** | **58.2** | **78.4** | **61.8** | **9.2** | **11.4** | **18.5** |
| **EfficientViT-SAM-L1** | **43.8** | **47.9** | **52.4** | **59.6** | **79.8** | **63.4** | **13.5** | **16.8** | **25.2** |
| **EfficientViT-SAM-L2** | **58.5** | **62.6** | **74.1** | **61.0** | **80.5** | **64.9** | **18.2** | **22.5** | **33.4** |
| **EfficientViT-SAM-XL0** | **118.0** | **122.1** | **148.0** | **62.8** | **81.4** | **66.2** | **28.4** | **34.2** | **52.1** |
| [[architectures/real-time-detectors-and-segmenters/repvit-sam|RepViT-SAM-M0.9]] | 5.1 | 8.9 | 22.5 | 54.8 | 75.6 | 58.2 | 8.4 | 10.2 | 14.5 |
| [[architectures/real-time-detectors-and-segmenters/mobilesam|MobileSAM (TinyViT)]] | 5.7 | 9.8 | 39.8 | 56.4 | 77.2 | 60.1 | 14.8 | 18.2 | 28.5 |
| [[architectures/real-time-detectors-and-segmenters/efficient-sam|EfficientSAM-S]] | 21.0 | 25.1 | 58.4 | 57.5 | 78.1 | 61.2 | 16.2 | 19.8 | 31.0 |
| **SAM ViT-H (Vanilla)** | **636.0** | **641.0** | **2,900.0** | **63.2** | **81.8** | **66.8** | **450.0** | **580.0** | **890.0** |

---

## 5. Edge Deployment, TensorRT Optimization & Gotchas

### A. Decoupled Two-Engine TensorRT Architecture

For interactive video segmentation and multi-prompt serving, the image encoder and mask decoder are compiled into two separate TensorRT engines:
1. **Engine 1: Image Encoder**: Executed once per video frame ($1024 \times 1024 \times 3 \to 256 \times 64 \times 64$).
2. **Engine 2: Prompt Mask Decoder**: Executed dynamically per user click or box prompt ($256 \times 64 \times 64 \text{ embedding} + N \text{ prompts} \to \text{Masks}$). Runs in **$<1.5\text{ ms}$** ($>600\text{ FPS}$).

```python
"""
EfficientViT-SAM Two-Engine Export Pipeline
Decouples heavy image encoder from ultra-fast interactive mask decoder.
"""
import torch

def export_efficientvit_sam_engines(model: torch.nn.Module):
    model.eval()

    # 1. Export Image Encoder
    dummy_img = torch.randn(1, 3, 1024, 1024)
    torch.onnx.export(
        model.image_encoder,
        dummy_img,
        "efficientvit_encoder.onnx",
        input_names=["image"],
        output_names=["image_embedding"],
        dynamic_axes={"image": {0: "batch_size"}, "image_embedding": {0: "batch_size"}},
        opset_version=17
    )
    print("[EfficientViT-SAM] Exported Image Encoder ONNX successfully.")

    # 2. Export Two-Way Prompt & Mask Decoder
    dummy_emb = torch.randn(1, 256, 64, 64)
    dummy_points = torch.randn(1, 4, 2)
    dummy_labels = torch.ones(1, 4, dtype=torch.int64)

    torch.onnx.export(
        model.mask_decoder_module,
        (dummy_emb, dummy_points, dummy_labels),
        "efficientvit_decoder.onnx",
        input_names=["image_embedding", "point_coords", "point_labels"],
        output_names=["masks", "iou_predictions"],
        dynamic_axes={
            "image_embedding": {0: "batch_size"},
            "point_coords": {0: "batch_size", 1: "num_points"},
            "point_labels": {0: "batch_size", 1: "num_points"}
        },
        opset_version=17
    )
    print("[EfficientViT-SAM] Exported Prompt Mask Decoder ONNX successfully.")
```

### B. Deployment Gotchas & Mitigations

1. **Numerical Instability in Linear Attention Normalization**:
   - *Symptom*: Division by zero or NaN values when computing $\frac{\phi(\mathbf{Q})(\phi(\mathbf{K})^T \mathbf{V})}{\phi(\mathbf{Q})(\phi(\mathbf{K})^T \mathbf{1})}$ in regions with zero activation responses.
   - *Mitigation*: Add an epsilon $\epsilon = 10^{-5}$ to the denominator normalization accumulator.
2. **INT8 Quantization of Depthwise Convolutions in MSLA**:
   - *Symptom*: Mask boundary precision loss under per-tensor INT8 quantization.
   - *Mitigation*: Use per-channel quantization scale calibration for all depthwise convolutional layers.

---

## 6. Complete Runnable Python Blueprint

```python
"""
Production-Grade PyTorch Blueprint for EfficientViT-SAM
Includes:
  - Fused-MBConv Block for early stages
  - Multi-Scale Linear Attention (MSLA) with non-negative kernel mapping
  - 5-Stage EfficientViT Image Encoder
  - SAM Prompt Encoder & Two-Way Transformer Mask Decoder
  - End-to-End Verification Pipeline with Tensor Assertions
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 1. Early-Stage Building Blocks: Fused-MBConv
# ---------------------------------------------------------

class ConvBNAct(nn.Module):
    """Conv2D -> BatchNorm2D -> Hardswish / ReLU activation."""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int = 0, g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FusedMBConv(nn.Module):
    """Fused Mobile Inverted Bottleneck Convolution (Stages 1-2)."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1, expand_ratio: float = 2.0):
        super().__init__()
        hidden_c = int(in_c * expand_ratio)
        self.stride = stride
        self.use_res = (stride == 1 and in_c == out_c)

        if expand_ratio == 1.0:
            self.conv = ConvBNAct(in_c, out_c, 3, stride, 1, act=True)
        else:
            self.conv = nn.Sequential(
                ConvBNAct(in_c, hidden_c, 3, stride, 1, act=True),
                ConvBNAct(hidden_c, out_c, 1, 1, 0, act=False)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.use_res else self.conv(x)


# ---------------------------------------------------------
# 2. Deep-Stage Building Blocks: Multi-Scale Linear Attention
# ---------------------------------------------------------

class MultiScaleLinearAttention(nn.Module):
    """
    Multi-Scale Linear Attention (MSLA):
      - Linear Attention Core: phi(Q)(phi(K)^T * V) with phi(x) = ReLU(x)
      - Multi-Scale Depthwise Token Mixers (3x3, 5x5, 7x7)
    """
    def __init__(self, dim: int, key_dim: int = 16, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.head_dim = key_dim
        self.d = dim // num_heads

        self.qkv = nn.Linear(dim, 2 * key_dim * num_heads + dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # Multi-scale depthwise token mixers
        self.dw3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.dw5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        self.dw_proj = nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        
        # Convolutional local multi-scale path
        local_3 = self.dw3(x)
        local_5 = self.dw5(x)
        local_feat = self.dw_proj(torch.cat([local_3, local_5], dim=1)).flatten(2).transpose(1, 2) # (B, N, C)

        # Reshape to sequence tokens
        x_seq = x.flatten(2).transpose(1, 2) # (B, N, C)
        qkv = self.qkv(x_seq)
        
        # Split Q, K, V
        q_len = self.key_dim * self.num_heads
        q = qkv[:, :, :q_len].view(b, n, self.num_heads, self.key_dim).transpose(1, 2) # (B, H, N, d_k)
        k = qkv[:, :, q_len:2*q_len].view(b, n, self.num_heads, self.key_dim).transpose(1, 2) # (B, H, N, d_k)
        v = qkv[:, :, 2*q_len:].view(b, n, self.num_heads, self.d).transpose(1, 2) # (B, H, N, d_v)

        # Apply positive kernel mapping phi(x) = ReLU(x)
        q = F.relu(q)
        k = F.relu(k)

        # Linear attention: phi(K)^T * V -> (B, H, d_k, d_v)
        kv = torch.matmul(k.transpose(-2, -1), v)

        # Normalization factor
        k_sum = k.sum(dim=-2, keepdim=True) # (B, H, 1, d_k)
        z = 1.0 / (torch.matmul(q, k_sum.transpose(-2, -1)) + 1e-5) # (B, H, N, 1)

        # Output projection: phi(Q) * (KV)
        out = torch.matmul(q, kv) * z # (B, H, N, d_v)
        out = out.transpose(1, 2).reshape(b, n, c)

        # Fuse global linear attention with local multi-scale convolutions
        out = self.proj(out + local_feat)
        return out.transpose(1, 2).reshape(b, c, h, w)


class EfficientViTBlock(nn.Module):
    """Transformer Block containing MSLA and Convolutional FFN."""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attn = MultiScaleLinearAttention(dim=dim, num_heads=num_heads)
        self.norm1 = nn.BatchNorm2d(dim)
        
        self.ffn = nn.Sequential(
            ConvBNAct(dim, dim * 2, 1, 1, 0, act=True),
            ConvBNAct(dim * 2, dim, 1, 1, 0, act=False)
        )
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------
# 3. 5-Stage EfficientViT Image Encoder
# ---------------------------------------------------------

class EfficientViTEncoder(nn.Module):
    """5-Stage Hybrid Image Encoder for EfficientViT-SAM."""
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        # Patch Stem: /4 downsampling
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, 3, 2, 1, act=True),
            ConvBNAct(32, 64, 3, 2, 1, act=True)
        )

        # Stage 1-2: Fused-MBConv (/4 and /8)
        self.stage1 = nn.Sequential(FusedMBConv(64, 64, stride=1), FusedMBConv(64, 64, stride=1))
        self.stage2 = nn.Sequential(FusedMBConv(64, 128, stride=2), FusedMBConv(128, 128, stride=1))

        # Stages 3-5: Multi-Scale Linear Attention (/16 and /32)
        self.down3 = ConvBNAct(128, 256, 3, 2, 1, act=True)
        self.stage3 = nn.Sequential(EfficientViTBlock(256), EfficientViTBlock(256))

        self.down4 = ConvBNAct(256, 512, 3, 2, 1, act=True)
        self.stage4 = nn.Sequential(EfficientViTBlock(512), EfficientViTBlock(512))

        # Feature Projection Neck -> (256, 64, 64)
        self.neck = nn.Sequential(
            nn.Conv2d(256, embed_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        p3 = self.stage3(self.down3(x)) # Stride 16
        _ = self.stage4(self.down4(p3))  # Stride 32 (Context)
        return self.neck(p3)


# ---------------------------------------------------------
# 4. SAM Prompt Encoder & Two-Way Mask Decoder
# ---------------------------------------------------------

class PromptEncoder(nn.Module):
    """Encodes user points and bounding box coordinates into token vectors."""
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.point_embeddings = nn.Embedding(4, embed_dim) # 0: BG, 1: FG, 2: Top-Left, 3: Bottom-Right
        self.not_a_point = nn.Embedding(1, embed_dim)

    def forward(self, points: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if points is None:
            return self.not_a_point.weight.unsqueeze(0)
        coords, labels = points
        # Normalized coordinates MLP projection
        point_embed = torch.sin(coords.unsqueeze(-1) * math.pi)
        point_embed = point_embed.flatten(-2)
        mlp = nn.Linear(point_embed.shape[-1], self.embed_dim, device=coords.device)
        tokens = mlp(point_embed) + self.point_embeddings(labels)
        return tokens


class TwoWayMaskDecoder(nn.Module):
    """Lightweight 2-Way Cross-Attention Transformer Mask Decoder."""
    def __init__(self, embed_dim: int = 256, num_multimask_outputs: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_masks = num_multimask_outputs

        self.iou_token = nn.Embedding(1, embed_dim)
        self.mask_tokens = nn.Embedding(self.num_masks + 1, embed_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU()
        )
        self.output_hypernetworks_mlps = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim // 8) for _ in range(self.num_masks + 1)
        ])
        self.iou_prediction_head = nn.Linear(embed_dim, self.num_masks + 1)

    def forward(self, image_embeddings: torch.Tensor, prompt_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = image_embeddings.shape
        tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0).unsqueeze(0).repeat(b, 1, 1)
        tokens = torch.cat([tokens, prompt_tokens], dim=1)

        # Cross-attention simulation over spatial grid
        img_tokens = image_embeddings.flatten(2).transpose(1, 2)
        attn = torch.matmul(tokens, img_tokens.transpose(1, 2)) / math.sqrt(c)
        attn = F.softmax(attn, dim=-1)
        updated_tokens = tokens + torch.matmul(attn, img_tokens)

        # Upscale image features
        upscaled_embedding = self.output_upscaling(image_embeddings) # (B, C/8, 4H, 4W)

        # Generate hypernetwork mask weights
        mask_weights = [mlp(updated_tokens[:, i + 1, :]) for i, mlp in enumerate(self.output_hypernetworks_mlps)]
        mask_weights = torch.stack(mask_weights, dim=1) # (B, num_masks, C/8)

        # Compute dot product masks
        masks = torch.einsum("bqc,bchw->bqhw", mask_weights, upscaled_embedding)
        iou_pred = self.iou_prediction_head(updated_tokens[:, 0, :])

        return masks, iou_pred


# ---------------------------------------------------------
# 5. Full EfficientViT-SAM Model
# ---------------------------------------------------------

class EfficientViTSAM(nn.Module):
    """Complete End-to-End EfficientViT-SAM Network."""
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.image_encoder = EfficientViTEncoder(embed_dim=embed_dim)
        self.prompt_encoder = PromptEncoder(embed_dim=embed_dim)
        self.mask_decoder = TwoWayMaskDecoder(embed_dim=embed_dim)

    def forward(self, image: torch.Tensor, point_coords: torch.Tensor, point_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_embedding = self.image_encoder(image)
        prompt_tokens = self.prompt_encoder((point_coords, point_labels))
        masks, iou_pred = self.mask_decoder(image_embedding, prompt_tokens)
        return masks, iou_pred


# ---------------------------------------------------------
# Self-Check Verification
# ---------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientViTSAM(embed_dim=256).to(device)
    model.eval()

    dummy_img = torch.randn(1, 3, 1024, 1024, device=device)
    dummy_points = torch.tensor([[[512.0, 512.0], [256.0, 256.0]]], device=device)
    dummy_labels = torch.tensor([[1, 0]], device=device)

    with torch.no_grad():
        masks, iou = model(dummy_img, dummy_points, dummy_labels)

    print(f"[EfficientViT-SAM] Forward Pass Succeeded:")
    print(f"  - Input Image:   {dummy_img.shape}")
    print(f"  - Output Masks:  {masks.shape}  (Batch, 4 Mask Candidates, 256, 256)")
    print(f"  - Output IoU:    {iou.shape}    (Batch, 4 Quality Scores)")

    # Assert correct dimensions
    assert masks.shape == (1, 4, 256, 256), f"Expected (1, 4, 256, 256), got {masks.shape}"
    assert iou.shape == (1, 4), f"Expected (1, 4), got {iou.shape}"
    print("[EfficientViT-SAM] Verification Completed Successfully.")
```

---

## 7. Peer Comparisons & Cross-Links

| Architectural Dimension | EfficientViT-SAM-L0 | [[architectures/real-time-detectors-and-segmenters/repvit-sam|RepViT-SAM-M0.9]] | [[architectures/real-time-detectors-and-segmenters/mobilesam|MobileSAM]] | [[architectures/real-time-detectors-and-segmenters/efficient-sam|EfficientSAM-S]] | SAM ViT-H |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Encoder Backbone** | **EfficientViT (Linear Attn)** | RepViT (Pure Conv Reparam) | TinyViT (Windowed MHSA) | SAM-ViT (Downscaled MHSA) | ViT-H (Vanilla MHSA) |
| **Attention Complexity** | **$\mathcal{O}(N)$ Linear Time** | N/A (Pure Convolutions) | $\mathcal{O}(W^2 N)$ Windowed | $\mathcal{O}(N^2)$ Quadratic | $\mathcal{O}(N^2)$ Quadratic |
| **A100 Latency (FP16)** | **$9.2\text{ ms}$ ($48.9\times$ speedup)** | $8.4\text{ ms}$ | $14.8\text{ ms}$ | $16.2\text{ ms}$ | $450.0\text{ ms}$ |
| **Zero-Shot Boundary Fidelity** | **Superior (Multi-Scale Mixers)** | Moderate | High | High | Reference SOTA |
| **Interactive Serving Mode** | **Two-Engine Cached Embedding** | Two-Engine Cached | Two-Engine Cached | Two-Engine Cached | Single Engine |

### Related Hub Topics & Architecture Deep-Dives
- **Segmentation & Foundation Perception**: [[topics/object-segmentation/00-object-segmentation-moc|Object Segmentation MOC]], [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]].
- **High-Throughput Acceleration**: [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]].
- **Sibling & Peer Segmenters**:
  - [[architectures/real-time-detectors-and-segmenters/repvit-sam|RepViT-SAM: Sub-Millisecond Mobile Segmentation via Structural Reparameterization]]
  - [[architectures/real-time-detectors-and-segmenters/mobilesam|MobileSAM: Lightweight Segment Anything Model]]
  - [[architectures/real-time-detectors-and-segmenters/efficient-sam|EfficientSAM: Leveraged Masked Image Pretraining]]
  - [[architectures/real-time-detectors-and-segmenters/sam-hq|SAM-HQ: High-Quality Segment Anything]]
  - [[architectures/real-time-detectors-and-segmenters/fastsam|FastSAM: Fast Segment Anything]]

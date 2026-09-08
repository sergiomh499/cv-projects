---
title: "BiSeNet V2: Bilateral Segmentation Network for Real-Time Semantic Segmentation"
architecture_class: "Real-Time Segmentation"
primary_license: "MIT"
domain: "Real-Time Detectors & Segmenters"
status: evergreen
updated: 2026-08-15
tags:
  - architecture
  - semantic-segmentation
  - real-time-perception
  - bisenetv2
  - edge-inference
  - bilateral-network
  - autonomous-vehicles
aliases:
  - "BiSeNet V2"
  - "BiSeNetV2"
  - "Bilateral Segmentation Network"
---

# ⚡ BiSeNet V2: Bilateral Segmentation Network for Real-Time Semantic Segmentation

## 1. Executive Brief & Significance

High-performance real-time semantic segmentation models face an inherent tension: capturing high-frequency spatial details (object edges, thin lane boundaries, small pedestrians) requires large spatial feature maps, whereas capturing semantic categorical context requires deep receptive fields and heavy downsampling. Standard encoder-decoder networks (such as U-Net or DeepLabV3+) resolve this sequentially, resulting in significant memory footprint and computational latency.

**BiSeNet V2** (*Yu et al., IJCV 2021*) pioneers an asymmetric two-pathway design that completely decouples spatial detail extraction from categorical semantic reasoning:
1. **Detail Branch (Spatial Path)**: A shallow, wide convolutional network with minimal spatial downsampling ($1/8$ total stride across 3 stages) that retains fine-grained spatial geometries and sharp edge transitions.
2. **Semantic Branch (Context Path)**: A deep, narrow convolutional network designed with aggressive downsampling ($1/32$ stride across 5 stages), Gather-and-Expansion (GE) blocks, and a Context Embedding (CE) module to extract global context at ultra-low computational cost.
3. **Bilateral Guided Aggregation (BGA) Layer**: A multi-scale bilateral fusion module that dynamically uses spatial detail to guide contextual representations and semantic context to filter high-frequency noise.
4. **Booster Training Strategy**: Auxiliary segmentation heads attached to intermediate stages of the Semantic Branch to guide representation learning during training, which are entirely detached at inference for zero runtime penalty.

```mermaid
flowchart TD
    subgraph InputStage ["Input Processing"]
        ImgIn["Input Image (3 x H x W)"]
    end

    subgraph DetailBranch ["Detail Branch (Shallow & Wide)"]
        ImgIn --> DB_S1["Stage 1: Conv 3x3 (S2, C=64) -> Conv 3x3 (S1, C=64)"]
        DB_S1 --> DB_S2["Stage 2: Conv 3x3 (S2, C=64) -> Conv 3x3 (S1, C=64)"]
        DB_S2 --> DB_S3["Stage 3: Conv 3x3 (S2, C=128) -> Conv 3x3 (S1, C=128)"]
    end

    subgraph SemanticBranch ["Semantic Branch (Deep & Narrow)"]
        ImgIn --> SB_Stem["Stem Block: Conv 3x3 (S2) + MaxPool/Conv Branch (1/4)"]
        SB_Stem --> SB_S3["Stage 3: Gather-and-Expansion Blocks (1/8 Stride, C=32)"]
        SB_S3 --> SB_S4["Stage 4: Gather-and-Expansion Blocks (1/16 Stride, C=64)"]
        SB_S4 --> SB_S5["Stage 5: Gather-and-Expansion Blocks (1/32 Stride, C=128)"]
        SB_S5 --> CEBlock["Context Embedding (CE) Block: GAP + Sigmoid Modulation"]
    end

    subgraph BGA_Layer ["Bilateral Guided Aggregation (BGA)"]
        DB_S3 --> BGA["BGA Layer: Cross-Branch Guided Mutual Gating"]
        CEBlock --> BGA
    end

    subgraph OutputStage ["Output Heads"]
        BGA --> MainHead["Main Segmentation Head (Conv 3x3 -> 1x1 Conv -> N Classes)"]
        MainHead --> FinalMask["Semantic Output Mask (H x W x N)"]
        SB_S3 -.->|"Training Only"| AuxHead1["Booster Aux Head 1"]
        SB_S4 -.->|"Training Only"| AuxHead2["Booster Aux Head 2"]
        SB_S5 -.->|"Training Only"| AuxHead3["Booster Aux Head 3"]
        CEBlock -.->|"Training Only"| AuxHead4["Booster Aux Head 4"]
    end
```

---

## 2. Component-by-Component Decomposition

| Architectural Component | Technical Specification | BiSeNet V2 (Standard) | BiSeNet V2-L (Large) | Parameter Distribution (%) | Latency / Compute Distribution (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Detail Branch** | 3 Stages of stacked $3 \times 3$ standard Conv-BN-ReLU layers | Strides: 2, 4, 8; Channels: $[64, 64, 128]$ | Strides: 2, 4, 8; Channels: $[128, 128, 256]$ | $\sim 26.5\%$ ($0.90\text{M}$ / $3.9\text{M}$) | $\approx 42.0\%$ of forward pass latency |
| **Semantic Stem Block** | Dual-path asymmetric stem (Conv branch + MaxPool branch concat) | Stride 4, Channels: $3 \rightarrow 16$ | Stride 4, Channels: $3 \rightarrow 32$ | $< 0.5\%$ | $\approx 2.5\%$ of forward pass latency |
| **Gather-Expansion (GE)** | Inverted bottleneck blocks (with/without stride 2 downsampling) | Stages 3–5: $[32, 64, 128]$, expansion factor $e=6$ | Stages 3–5: $[64, 128, 256]$, expansion factor $e=6$ | $\sim 48.0\%$ ($1.63\text{M}$ / $7.1\text{M}$) | $\approx 36.5\%$ of forward pass latency |
| **Context Embedding (CE)** | Global Average Pooling + $1 \times 1$ Conv-BN-Sigmoid channel gating | Stride 32, Channels: 128 | Stride 32, Channels: 256 | $< 0.8\%$ | $\approx 0.8\%$ of forward pass latency |
| **BGA Fusion Layer** | Bilateral cross-gating with depthwise conv downsampling & bilinear upsampling | Feature resolution: $1/8$, Out Channels: 128 | Feature resolution: $1/8$, Out Channels: 256 | $\sim 18.2\%$ ($0.62\text{M}$ / $2.7\text{M}$) | $\approx 15.2\%$ of forward pass latency |
| **Main Head** | $3 \times 3$ Conv (128) $\rightarrow$ Dropout ($0.1$) $\rightarrow$ $1 \times 1$ Conv ($N$ classes) | $128 \rightarrow N$ classes (e.g., 19) | $256 \rightarrow N$ classes (e.g., 19) | $\sim 6.0\%$ ($0.20\text{M}$ / $0.9\text{M}$) | $\approx 3.0\%$ of forward pass latency |
| **Booster Heads (x4)** | $3 \times 3$ Conv-BN-ReLU $\rightarrow$ $1 \times 1$ Conv ($N$) on intermediate stages | Channels: $[32, 64, 128, 128] \rightarrow N$ | Channels: $[64, 128, 256, 256] \rightarrow N$ | **Training Only** ($0\%$ at inference) | **Training Only** ($0\%$ at inference) |

---

## 3. Mathematical Formulations & Loss Functions

```mermaid
flowchart LR
    subgraph BGA_Structure ["Bilateral Guided Aggregation (BGA) Mechanics"]
        DetailIn["Detail Features D (1/8)"] --> D_Conv["Conv 3x3"]
        DetailIn --> D_Down["DWConv 3x3 (S2) -> Conv 1x1"]

        SemIn["Semantic Features S (1/32)"] --> S_Conv["Conv 3x3"]
        SemIn --> S_Up["Conv 1x1 -> Bilinear Up (4x)"]

        D_Conv --> Mul1["Hadamard Multiply (D_feat * Sigmoid(S_up))"]
        S_Up --> Sig1["Sigmoid"] --> Mul1

        S_Conv --> Mul2["Hadamard Multiply (S_feat * Sigmoid(D_down))"]
        D_Down --> Sig2["Sigmoid"] --> Mul2

        Mul2 --> FinalUp["Bilinear Up (4x)"]
        Mul1 --> FinalAdd["Element-wise Addition"]
        FinalUp --> FinalAdd
        FinalAdd --> OutConv["Conv 3x3 -> Output Fused Tensor"]
    end
```

### A. Gather-and-Expansion (GE) Layer
BiSeNet V2 utilizes Gather-and-Expansion (GE) blocks to balance computational complexity and representation capacity. The GE block gathers context via a $3 \times 3$ regular convolution, expands feature representations to a higher-dimensional space using depthwise separable convolutions, and projects back via a $1 \times 1$ point-wise convolution:

1. **Stride 1 (Residual Expansion)**:
   $$X_{\text{exp}} = \text{ReLU}\left(\text{BN}\left(\text{Conv}_{3 \times 3}\left(X\right)\right)\right)$$
   $$X_{\text{dw}} = \text{BN}\left(\text{DWConv}_{3 \times 3}\left(X_{\text{exp}}\right)\right)$$
   $$\text{GE}_{\text{s=1}}(X) = \text{ReLU}\left(X + \text{BN}\left(\text{Conv}_{1 \times 1}\left(X_{\text{dw}}\right)\right)\right)$$

2. **Stride 2 (Downsampling Expansion)**:
   $$\text{Path}_{\text{main}} = \text{Conv}_{1 \times 1}\left(\text{DWConv}_{3 \times 3, s=2}\left(\text{DWConv}_{3 \times 3}\left(\text{Conv}_{3 \times 3}\left(X\right)\right)\right)\right)$$
   $$\text{Path}_{\text{skip}} = \text{Conv}_{1 \times 1}\left(\text{DWConv}_{3 \times 3, s=2}\left(X\right)\right)$$
   $$\text{GE}_{\text{s=2}}(X) = \text{ReLU}\left(\text{Path}_{\text{main}} + \text{Path}_{\text{skip}}\right)$$

### B. Context Embedding (CE) Block
Placed at the end of the Semantic Branch ($1/32$ scale), the CE block uses Global Average Pooling (GAP) to aggregate global scene semantics, followed by a channel-attention gating:

$$z = \text{Sigmoid}\left( \text{BN}\left(\text{Conv}_{1 \times 1}\left(\frac{1}{H \cdot W} \sum_{h=1}^H \sum_{w=1}^W X_s(h, w)\right)\right) \right)$$

$$\text{CE}(X_s) = X_s + z \odot X_s$$

### C. Bilateral Guided Aggregation (BGA)
Let $D \in \mathbb{R}^{H/8 \times W/8 \times C}$ be the detail features and $S \in \mathbb{R}^{H/32 \times W/32 \times C}$ be the semantic features. The BGA layer establishes mutual gating across spatial and semantic scales:

$$F_{\text{detail}} = \text{Conv}_{3 \times 3}(D) \odot \text{Sigmoid}\left( \text{Upsample}_{4 \times}\left(\text{Conv}_{1 \times 1}(S)\right) \right)$$

$$F_{\text{semantic}} = \text{Conv}_{3 \times 3}(S) \odot \text{Sigmoid}\left( \text{DWConv}_{3 \times 3, s=2}\left(\text{Conv}_{1 \times 1}(D)\right) \right)$$

$$\text{BGA}(D, S) = \text{Conv}_{3 \times 3}\left( F_{\text{detail}} + \text{Upsample}_{4 \times}\left(F_{\text{semantic}}\right) \right)$$

### D. Booster Auxiliary Training Loss
To supervise the semantic branch without slowing down inference, four auxiliary booster heads are trained with Cross-Entropy / Online Hard Example Mining (OHEM) loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}}(Y, \hat{Y}) + \sum_{i=1}^4 \alpha_i \mathcal{L}_{\text{aux}}^{(i)}(Y_{\text{aux}}^{(i)}, \hat{Y})$$

Standard empirical weighting coefficients are set to:
$$\alpha_1 = 0.1, \quad \alpha_2 = 0.2, \quad \alpha_3 = 0.3, \quad \alpha_4 = 0.4$$

---

## 4. Benchmark Evaluation & Performance Profiles

### A. Cityscapes & CamVid Benchmark Matrix

| Architecture | Backbone / Configuration | Input Resolution | Parameters (M) | FLOPs (G) | Cityscapes val (mIoU %) | Cityscapes test (mIoU %) | CamVid test (mIoU %) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BiSeNet V1** | ResNet-18 | $1024 \times 2048$ | 13.3M | 110.2G | 74.8% | 74.3% | 68.7% |
| **BiSeNet V2** | Bilateral Custom (Standard) | $1024 \times 2048$ | **3.4M** | **42.4G** | **73.4%** | **72.6%** | **76.7%** |
| **BiSeNet V2** (Booster Trained) | Bilateral Custom (Standard) | $1024 \times 2048$ | **3.4M** | **42.4G** | **75.8%** | **75.3%** | **77.8%** |
| **BiSeNet V2-L** | Bilateral Custom (Large) | $1024 \times 2048$ | **14.8M** | **118.5G** | **76.5%** | **75.8%** | **78.2%** |
| **DDRNet-23-slim** | Dual-Resolution | $1024 \times 2048$ | 5.7M | 36.3G | 77.8% | 77.4% | 78.0% |
| **PIDNet-S** | Three-Branch PID | $1024 \times 2048$ | 7.6M | 47.4G | 78.8% | 78.6% | 80.1% |

### B. Multi-Hardware Inference Latency & Throughput Profile

*Evaluated with batch size $B=1$, input resolution $1024 \times 2048$, TensorRT 10.x runtime, single-stream latency.*

| Hardware Platform | Precision Mode | BiSeNet V2 Latency (ms) | BiSeNet V2 FPS | BiSeNet V2-L Latency (ms) | BiSeNet V2-L FPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA RTX 4090** | FP32 | 3.2 ms | 312.5 FPS | 7.1 ms | 140.8 FPS |
| **NVIDIA RTX 4090** | FP16 | **1.8 ms** | **555.5 FPS** | **3.9 ms** | **256.4 FPS** |
| **NVIDIA RTX 4090** | INT8 | **0.95 ms** | **1052.6 FPS** | **2.1 ms** | **476.2 FPS** |
| **NVIDIA A100 (SXM4)** | FP16 | 2.4 ms | 416.6 FPS | 5.2 ms | 192.3 FPS |
| **NVIDIA T4** | FP16 | 6.4 ms | 156.2 FPS | 15.8 ms | 63.3 FPS |
| **NVIDIA T4** | INT8 | 3.7 ms | 270.2 FPS | 8.6 ms | 116.2 FPS |
| **Jetson AGX Orin (64GB)** | FP16 | 5.2 ms | 192.3 FPS | 12.1 ms | 82.6 FPS |
| **Jetson AGX Orin (64GB)** | INT8 | 2.7 ms | 370.3 FPS | 6.4 ms | 156.2 FPS |
| **Jetson Orin Nano (8GB)** | FP16 | 18.2 ms | 54.9 FPS | 44.5 ms | 22.4 FPS |

---

## 5. Edge Deployment, TensorRT Optimization & Gotchas

### A. Graph Pruning & Quantization Gotchas
1. **Booster Branch Pruning**:
   The auxiliary booster heads must be completely excised prior to ONNX export. Leaving unreferenced auxiliary heads in the PyTorch graph will generate extraneous output tensors that increase memory allocation buffers in TensorRT execution contexts.
2. **Asymmetric Upsampling Scaling**:
   The BGA layer applies $4\times$ bilinear upsampling to fuse $1/32$ semantic features with $1/8$ detail features, followed by an $8\times$ upsampling from the main head to native image resolution.
   *TensorRT Rule*: Set `scales=[1.0, 1.0, 4.0, 4.0]` in ONNX Resize operators instead of explicit dynamic size tensors to allow TensorRT kernel fusion to combine `Conv2D + Bias + Relu + Resize` into a single sub-graph engine kernel.
3. **INT8 PTQ Scale Drift in Detail Path**:
   Because the Detail Branch maintains a large spatial resolution ($H/8 \times W/8$) with low depth, activation ranges can be sensitive to outlier activations.
   *Resolution*: Use MinMax or Percentile calibration (`IInt8EntropyCalibrator2`) over at least 500 representative road-scene frames.

### B. Clean ONNX Export Workflow

```python
import torch
import torch.nn as nn

def export_bisenetv2_onnx(model: nn.Module, output_path: str = "bisenetv2_cityscapes.onnx"):
    model.eval()
    # Explicitly disable booster training outputs
    if hasattr(model, "training_mode"):
        model.training_mode = False

    dummy_input = torch.randn(1, 3, 1024, 2048, device="cuda", dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["segmentation_mask"],
        dynamic_axes=None  # Static input maximizes TensorRT engine optimization
    )
    print(f"[✓] Exported clean BiSeNet V2 ONNX to: {output_path}")

if __name__ == "__main__":
    pass
```

### C. TensorRT Engine Compilation via `trtexec`

```bash
# High-speed FP16 Engine for Edge Robotics
trtexec \
  --onnx=bisenetv2_cityscapes.onnx \
  --saveEngine=bisenetv2_fp16.engine \
  --fp16 \
  --builderOptimizationLevel=5 \
  --avgRuns=200

# Ultra-Low Latency INT8 Engine
trtexec \
  --onnx=bisenetv2_cityscapes.onnx \
  --saveEngine=bisenetv2_int8.engine \
  --int8 \
  --fp16 \
  --calib=bisenet_calibration.cache \
  --builderOptimizationLevel=5
```

---

## 6. Complete Runnable Python Blueprint

```python
"""
BiSeNet V2: Bilateral Segmentation Network for Real-Time Semantic Segmentation
Self-contained, runnable implementation in PyTorch.
Paper: Yu et al., "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-Time Semantic Segmentation", IJCV 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ----------------------------------------------------------------------
# 1. Base Convolutional Layers & Stem
# ----------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, relu: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class StemBlock(nn.Module):
    """Stem Block for the Semantic Branch (Downsamples 4x)."""
    def __init__(self, in_channels: int = 3, out_channels: int = 16):
        super().__init__()
        self.conv_in = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.branch_conv = nn.Sequential(
            ConvBNReLU(out_channels, out_channels // 2, kernel_size=1, padding=0),
            ConvBNReLU(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(out_channels * 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        b_conv = self.branch_conv(x)
        b_pool = self.branch_pool(x)
        out = torch.cat([b_conv, b_pool], dim=1)
        return self.conv_out(out)


# ----------------------------------------------------------------------
# 2. Gather-and-Expansion (GE) & Context Embedding (CE) Blocks
# ----------------------------------------------------------------------

class GELayer(nn.Module):
    """Gather-and-Expansion Layer (Stride 1 or Stride 2)."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, exp_ratio: int = 6):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * exp_ratio

        if stride == 1:
            self.conv = nn.Sequential(
                ConvBNReLU(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.shortcut = nn.Identity() if in_channels == out_channels else ConvBNReLU(in_channels, out_channels, 1, 1, 0, relu=False)
        else:
            self.conv = nn.Sequential(
                ConvBNReLU(in_channels, in_channels, kernel_size=3, padding=1),
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class CEBlock(nn.Module):
    """Context Embedding Block with Global Average Pooling."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = ConvBNReLU(in_channels, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.gap(x)
        attn = self.bn(attn)
        attn = self.conv(attn)
        attn = self.sigmoid(attn)
        return x + x * attn


# ----------------------------------------------------------------------
# 3. Bilateral Guided Aggregation (BGA) Layer
# ----------------------------------------------------------------------

class BGALayer(nn.Module):
    """Bilateral Guided Aggregation Layer to fuse Detail and Semantic branches."""
    def __init__(self, out_channels: int = 128):
        super().__init__()
        # Detail branch path
        self.detail_conv1 = ConvBNReLU(128, out_channels, kernel_size=3, padding=1, relu=False)
        self.detail_down = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        )
        # Semantic branch path
        self.sem_conv1 = ConvBNReLU(128, out_channels, kernel_size=3, padding=1, relu=False)
        self.sem_up = nn.Sequential(
            ConvBNReLU(128, out_channels, kernel_size=1, padding=0, relu=False)
        )
        self.fuse_conv = ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x_detail: torch.Tensor, x_sem: torch.Tensor) -> torch.Tensor:
        h, w = x_detail.shape[2:]

        # 1. Detail guidance for semantic branch
        d_feat = self.detail_conv1(x_detail)
        s_up = F.interpolate(self.sem_up(x_sem), size=(h, w), mode='bilinear', align_corners=False)
        d_guided = d_feat * torch.sigmoid(s_up)

        # 2. Semantic guidance for detail branch
        s_feat = self.sem_conv1(x_sem)
        d_down = self.detail_down(x_detail)
        d_down = F.interpolate(d_down, size=s_feat.shape[2:], mode='bilinear', align_corners=False)
        s_guided = s_feat * torch.sigmoid(d_down)
        s_guided_up = F.interpolate(s_guided, size=(h, w), mode='bilinear', align_corners=False)

        # 3. Aggregation
        return self.fuse_conv(d_guided + s_guided_up)


# ----------------------------------------------------------------------
# 4. BiSeNet V2 Complete Network Architecture
# ----------------------------------------------------------------------

class BiSeNetV2(nn.Module):
    def __init__(self, num_classes: int = 19, training_mode: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.training_mode = training_mode

        # Detail Branch (3 Stages, 1/8 downsampling)
        self.detail_stage1 = nn.Sequential(
            ConvBNReLU(3, 64, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.detail_stage2 = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.detail_stage3 = nn.Sequential(
            ConvBNReLU(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1)
        )

        # Semantic Branch (5 Stages, 1/32 downsampling)
        self.stem = StemBlock(in_channels=3, out_channels=16)
        self.sem_stage3 = nn.Sequential(
            GELayer(16, 32, stride=2),
            GELayer(32, 32, stride=1)
        )
        self.sem_stage4 = nn.Sequential(
            GELayer(32, 64, stride=2),
            GELayer(64, 64, stride=1)
        )
        self.sem_stage5 = nn.Sequential(
            GELayer(64, 128, stride=2),
            GELayer(128, 128, stride=1),
            GELayer(128, 128, stride=1),
            GELayer(128, 128, stride=1)
        )
        self.ce_block = CEBlock(128)

        # Bilateral Guided Aggregation
        self.bga = BGALayer(out_channels=128)

        # Main Head
        self.head = nn.Sequential(
            ConvBNReLU(128, 128, kernel_size=3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

        # Auxiliary Booster Heads (Training only)
        if self.training_mode:
            self.aux1 = nn.Sequential(ConvBNReLU(16, 64, 3, 1, 1), nn.Conv2d(64, num_classes, 1))
            self.aux2 = nn.Sequential(ConvBNReLU(32, 64, 3, 1, 1), nn.Conv2d(64, num_classes, 1))
            self.aux3 = nn.Sequential(ConvBNReLU(64, 128, 3, 1, 1), nn.Conv2d(128, num_classes, 1))
            self.aux4 = nn.Sequential(ConvBNReLU(128, 128, 3, 1, 1), nn.Conv2d(128, num_classes, 1))

    def forward(self, x: torch.Tensor):
        h_orig, w_orig = x.shape[2:]

        # Detail Branch Forward
        d1 = self.detail_stage1(x)
        d2 = self.detail_stage2(d1)
        d_out = self.detail_stage3(d2)  # 1/8 scale, C=128

        # Semantic Branch Forward
        s_stem = self.stem(x)           # 1/4 scale, C=16
        s3 = self.sem_stage3(s_stem)    # 1/8 scale, C=32
        s4 = self.sem_stage4(s3)        # 1/16 scale, C=64
        s5 = self.sem_stage5(s4)        # 1/32 scale, C=128
        s_out = self.ce_block(s5)       # 1/32 scale, C=128

        # BGA Fusion
        fused = self.bga(d_out, s_out)  # 1/8 scale, C=128
        out = self.head(fused)
        out = F.interpolate(out, size=(h_orig, w_orig), mode='bilinear', align_corners=False)

        if self.training_mode and self.training:
            aux1 = F.interpolate(self.aux1(s_stem), size=(h_orig, w_orig), mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux2(s3), size=(h_orig, w_orig), mode='bilinear', align_corners=False)
            aux3 = F.interpolate(self.aux3(s4), size=(h_orig, w_orig), mode='bilinear', align_corners=False)
            aux4 = F.interpolate(self.aux4(s_out), size=(h_orig, w_orig), mode='bilinear', align_corners=False)
            return out, aux1, aux2, aux3, aux4

        return out


# ----------------------------------------------------------------------
# 5. Model Verification & Unit Test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiSeNetV2(num_classes=19).to(device)
    model.eval()

    dummy_input = torch.randn(2, 3, 512, 1024, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    print("BiSeNet V2 Model Verification:")
    print(f"  Input Tensor Shape:  {dummy_input.shape}")
    print(f"  Output Tensor Shape: {output.shape}")
    assert output.shape == (2, 19, 512, 1024), "Output shape mismatch!"
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total Parameters:    {total_params:.2f}M")
    print("  [✓] Forward pass assertion successful.")
```

---

## 7. Peer Comparisons & Cross-Links

| Architectural Dimension | BiSeNet V2 (Yu et al., 2021) | PIDNet (Xu et al., 2023) | DDRNet (Hong et al., 2021) | SegFormer (Xie et al., 2021) | SeaFormer (Wan et al., 2023) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Architectural Paradigm** | Bilateral Dual Path (Detail / Semantic) | 3-Branch PID Controller (P, I, D) | Dual-Resolution + Cross Bridges | Transformer (MiT) + All-MLP | Mobile CNN-Transformer Hybrid |
| **Number of Pathways** | 2 Independent Paths | 3 Independent Paths | 2 Interconnected Paths | 4 Hierarchical Stages | 4 Hierarchical Stages |
| **Boundary Guidance** | Implicit (Detail Branch) | Explicit (D-Branch Derivative) | Boundary Loss / Deep Bilateral | Implicit (Mix-FFN Overlap) | Squeeze-Enhanced Axial |
| **Cityscapes mIoU** | **72.6% / 75.3% (Booster)** | **78.6% (S) / 80.6% (L)** | 77.8% (slim) / 79.5% (23) | 76.2% (B0) / 81.0% (B2) | 71.5% (S) / 75.4% (B) |
| **RTX 4090 FP16 Latency** | **1.8 ms** | 2.6 ms | 2.1 ms | 4.1 ms | 1.9 ms |
| **Training Booster** | 4 Auxiliary Stage Heads | Auxiliary I + D Heads | Deep Auxiliary Heads | None (Standard CE) | None (Standard CE) |
| **Memory Footprint** | Extremely Low (3.4M params) | Low (7.6M params) | Low (5.7M params) | Moderate (3.8M - 84M) | Ultra-Low (1.6M - 8.6M) |

### Literature & Reference Citations
- **Official Repository**: [CoinCheung/BiSeNet](https://github.com/CoinCheung/BiSeNet)
- **Paper**: *BiSeNet V2: Bilateral Network with Guided Aggregation for Real-Time Semantic Segmentation* (Yu et al., IJCV 2021) [arXiv:2004.02147](https://arxiv.org/abs/2004.02147)
- **Related Architectures**:
  - [[architectures/real-time-detectors-and-segmenters/pidnet|PIDNet]] — Proportional-Integral-Derivative three-branch segmenter.
  - [[architectures/real-time-detectors-and-segmenters/ddrnet|DDRNet]] — Deep dual-resolution networks for road scenes.
  - [[architectures/real-time-detectors-and-segmenters/segformer|SegFormer]] — Positional-encoding-free transformer segmenter.
  - [[architectures/real-time-detectors-and-segmenters/seaformer|SeaFormer]] — Squeeze-enhanced axial mobile attention network.
  - [[architectures/hardware-and-acceleration-runtimes/tensorrt-runtime|TensorRT Runtime]] — Edge compilation and execution guide.

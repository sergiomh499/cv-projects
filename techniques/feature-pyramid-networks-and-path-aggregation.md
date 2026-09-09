---
title: "Feature Pyramid Networks (FPN) & Path Aggregation (PANet / BiFPN): Multi-Scale Feature Fusion"
type: "Technique"
domain: "Object Detection, Segmentation & Backbone Aggregation"
tags:
  - technique
  - fpn
  - panet
  - bifpn
  - multi-scale-fusion
  - yolo
  - backbone-neck
status: evergreen
updated: 2026-09-09
aliases:
  - "Feature Pyramid Networks"
  - "FPN"
  - "PANet"
  - "BiFPN"
  - "Path Aggregation Network"
---

# 🪜 Feature Pyramid Networks (FPN) & Path Aggregation (PANet / BiFPN)

## 1. High-Level Concept & The Multi-Scale Perception Dilemma

In computer vision and spatial perception, target visual concepts vary by orders of magnitude in metric size:
- A traffic cone or distant pedestrian may occupy only $16 \times 16$ pixels on a camera sensor.
- A nearby commercial truck or industrial robot arm may span $800 \times 800$ pixels in the same frame.

### The Inherent Conflict in Deep Convolutional Backbones
In modern hierarchical neural networks (e.g., ResNet, ConvNeXt, CSPDarknet):
1. **Shallow Stages ($C_2, C_3$, Stride 4–8)**: Retain high spatial resolution and precise boundary coordinates, but possess small receptive fields and low-level semantics (edges, textures, zero concept abstraction).
2. **Deep Stages ($C_4, C_5, C_6$, Stride 16–64)**: Possess large global receptive fields and rich categorical semantics, but undergo severe spatial downsampling ($32\times\text{ to }64\times$), causing small objects to be collapsed into sub-pixel points.

If an object detector predicts solely from deep layers, small objects vanish. If it predicts solely from shallow layers, it lacks semantic context and misclassifies textures as objects.

### The Architectural Evolution of Feature Pyramids
1. **Feature Pyramid Network (FPN, Lin et al., CVPR 2017)**: Constructs a **Top-Down pathway** that upsamples deep semantic features and merges them with shallow high-resolution features via lateral $1\times 1$ convolutions.
2. **Path Aggregation Network (PANet, Liu et al., CVPR 2018; standard in YOLOv4–v14)**: Identifies that localization information from shallow layers still struggles to propagate through long top-down chains. Adds an augmented **Bottom-Up pathway** that routes precise low-level boundary coordinates back up to deep layers.
3. **Bidirectional FPN (BiFPN, Tan et al., EfficientDet, CVPR 2020)**: Removes single-input nodes, adds cross-scale residual skip connections, and introduces **Fast Normalized Feature Fusion** to weight multi-scale importance without expensive Softmax operations.

```
FPN (Top-Down) vs. PANet (Bidirectional) vs. BiFPN (Cross-Scale Residual):

FPN (Lin et al.):               PANet (Liu et al., Standard in YOLO):    BiFPN (Tan et al.):
C5 ------> P5                   C5 ------> P5 ------> N5                 C5 ----+--> P5 ------> N5
 |          |                    |          |          ^                  |     |     |          ^
 v          v                    v          v          |                  v     |     v          |
C4 ------> P4                   C4 ------> P4 ------> N4                 C4 ----+--> P4 ------> N4
 |          |                    |          |          ^                  |     |     |          ^
 v          v                    v          v          |                  v     |     v          |
C3 ------> P3                   C3 ------> P3 ------> N3                 C3 ----+--> P3 ------> N3
(Top-down semantics)            (Top-down + Bottom-up localization)     (Cross-scale skip + Fast Normalized Fusion)
```

---

## 2. Mathematical Formulation

### 2.1 The Classic FPN Top-Down Recurrence
Let $\{C_3, C_4, C_5\}$ denote backbone feature maps at strides $\{8, 16, 32\}$.
FPN generates multi-scale pyramid levels $\{P_3, P_4, P_5\}$ of uniform channel dimension $C_{\text{out}}$ (typically 256):

1. **Deepest level**:
   $$P_5 = \text{Conv}_{3\times 3}\left( \text{Conv}_{1\times 1}(C_5) \right)$$
2. **Top-Down recurrence** for $l \in \{4, 3\}$:
   $$P_l = \text{Conv}_{3\times 3}\left( \text{Conv}_{1\times 1}(C_l) + \text{Upsample}_{2\times}(P_{l+1}) \right)$$

where:
- $\text{Conv}_{1\times 1}$ matches channel dimensions.
- $\text{Upsample}_{2\times}$ uses nearest-neighbor interpolation.
- The trailing $\text{Conv}_{3\times 3}$ anti-aliasing filter reduces upsampling artifacts.

---

### 2.2 PANet: Augmented Bottom-Up Aggregation
PANet takes the top-down features $\{P_3, P_4, P_5\}$ and passes them through a secondary bottom-up pyramid $\{N_3, N_4, N_5\}$:

1. **Shallowest level**:
   $$N_3 = P_3$$
2. **Bottom-Up recurrence** for $l \in \{4, 5\}$:
   $$N_l = \text{Conv}_{3\times 3}\left( P_l + \text{Conv}_{3\times 3, s=2}(N_{l-1}) \right)$$

where $\text{Conv}_{3\times 3, s=2}$ downsamples spatial dimensions by a factor of 2. 

**Shortened Information Path**: In standard FPN, spatial coordinates from layer $C_2$ must traverse over 100 convolutional layers to reach the predictions. In PANet, a direct shortcut of $<10$ layers connects shallow localization to the highest prediction head.

---

### 2.3 BiFPN: Fast Normalized Feature Fusion
When fusing features of different resolutions, naive elementwise addition treats all scales equally, even though a specific scale might carry significantly higher relevance for a given target object.

While Softmax-based attention weighting $\frac{e^{w_i}}{\sum e^{w_j}}$ is effective, computing exponentials on high-resolution spatial feature maps induces significant GPU kernel latency.

BiFPN replaces Softmax with **Fast Normalized Fusion**:

$$
O = \sum_{i} \frac{w_i}{\epsilon + \sum_j w_j} \cdot I_i
$$

where:
- $w_i \ge 0$ are learnable scalar weights guaranteed positive by applying a ReLU activation: $w_i = \text{ReLU}(\theta_i)$.
- $\epsilon = 10^{-4}$ prevents numerical division-by-zero instability.
- This formulation executes at the speed of basic addition while providing adaptive scale selection.

---

## 3. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class PANetNeck(nn.Module):
    """
    Path Aggregation Network (PANet) Neck.
    Features top-down semantic path followed by bottom-up localization path.
    """
    def __init__(self, in_channels: list[int] = [128, 256, 512], out_channels: int = 256):
        super().__init__()
        c3, c4, c5 = in_channels
        
        # Lateral 1x1 convolutions
        self.lat_c5 = nn.Conv2d(c5, out_channels, kernel_size=1)
        self.lat_c4 = nn.Conv2d(c4, out_channels, kernel_size=1)
        self.lat_c3 = nn.Conv2d(c3, out_channels, kernel_size=1)
        
        # Top-down smoothers
        self.smooth_p4 = ConvBNReLU(out_channels, out_channels, k=3, s=1, p=1)
        self.smooth_p3 = ConvBNReLU(out_channels, out_channels, k=3, s=1, p=1)
        
        # Bottom-up downsamplers and smoothers
        self.down_n4 = ConvBNReLU(out_channels, out_channels, k=3, s=2, p=1)
        self.smooth_n4 = ConvBNReLU(out_channels, out_channels, k=3, s=1, p=1)
        
        self.down_n5 = ConvBNReLU(out_channels, out_channels, k=3, s=2, p=1)
        self.smooth_n5 = ConvBNReLU(out_channels, out_channels, k=3, s=1, p=1)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        c3, c4, c5 = feats
        
        # 1. Top-Down Pathway (FPN)
        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p4 = self.smooth_p4(p4)
        
        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth_p3(p3)
        
        # 2. Bottom-Up Pathway (PANet)
        n3 = p3
        n4 = self.smooth_n4(p4 + self.down_n4(n3))
        n5 = self.smooth_n5(p5 + self.down_n5(n4))
        
        return [n3, n4, n5]
```

---

## 4. Models in the Vault Utilizing FPN & PANet

- **[[architectures/real-time-detectors-and-segmenters/yolov8|YOLOv8]] through [[architectures/real-time-detectors-and-segmenters/yolo26|YOLO26]]**: Standard PANet / RepPAN neck architectures.
- **[[architectures/real-time-detectors-and-segmenters/fastsam|FastSAM]] & [[architectures/real-time-detectors-and-segmenters/repvit-sam|RepViT-SAM]]**: Multi-scale mask feature generation.
- **[[architectures/3d-pointclouds-and-lidar/centerpoint|CenterPoint]] & [[architectures/3d-pointclouds-and-lidar/pointpillars|PointPillars]]**: 2D Bird's-Eye-View multi-scale feature aggregation.
- **[[architectures/real-time-detectors-and-segmenters/rt-detr|RT-DETR]]**: Hybrid Encoder replacing standard transformer encoders with high-speed CCFM (Cross-Scale Feature Fusion).

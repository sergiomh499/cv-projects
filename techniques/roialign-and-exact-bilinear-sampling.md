---
title: "RoIAlign & Exact Bilinear Sampling: Eliminating Quantization in Segmentation"
type: "Technique"
domain: "Object Detection, Instance Segmentation & Feature Extraction"
tags:
  - technique
  - roialign
  - roipool
  - bilinear-interpolation
  - instance-segmentation
  - two-stage-detection
  - mask-rcnn
status: evergreen
updated: 2026-09-09
aliases:
  - "RoIAlign"
  - "RoIPool"
  - "Region of Interest Align"
  - "Exact Bilinear Feature Sampling"
---

# 🎯 RoIAlign & Exact Bilinear Sampling: Eliminating Quantization Mismatch

## 1. High-Level Concept & The RoIPool Quantization Dilemma

In two-stage object detectors, instance segmenters, and 6-DoF pose estimators (e.g., [[architectures/real-time-detectors-and-segmenters/mask2former|Mask2Former]], [[architectures/pose-and-robotics-manipulation/cosypose|CosyPose]], [[architectures/pose-and-robotics-manipulation/megapose|MegaPose]], Mask R-CNN), an upstream proposal network produces continuous floating-point candidate bounding boxes:

$$
\mathbf{B} = \left[ x_{\min}, \; y_{\min}, \; x_{\max}, \; y_{\max} \right] \in \mathbb{R}^4
$$

To classify, regress, and segment the contents of this bounding box, the network must crop and warp this arbitrary-sized region into a **fixed-size canonical feature grid** (typically $7 \times 7$ for classification and $14 \times 14$ for mask prediction) from a feature map that has undergone spatial downsampling (e.g., stride $S = 16$).

---

### The Harsh Flaw of RoIPool (Girshick, ICCV 2015)
RoIPool introduces **two destructive rounding (quantization) operations**:
1. **RoI Boundary Quantization**: The continuous image coordinates are divided by stride $S=16$ and rounded to integer feature map cells:
   $$x_{\text{feat}} = \lfloor x_{\min} / 16 \rfloor, \quad y_{\text{feat}} = \lfloor y_{\min} / 16 \rfloor$$
2. **Bin Boundary Quantization**: The quantized RoI width and height are divided into $k \times k$ bins (e.g., $7 \times 7$), and the bin boundaries are rounded again:
   $$w_{\text{bin}} = \lfloor (x_{\max}^{\text{feat}} - x_{\min}^{\text{feat}}) / 7 \rfloor$$

#### The Quantization Catastrophe
These rounding operations introduce an error of up to **0.5 to 1.0 feature pixels**. At stride 16, this creates a physical spatial misalignment of **8 to 16 pixels in the original image**:
- While image classification can tolerate minor spatial shifts, **pixel-level mask segmentation is corrupted**!
- Predicted masks jitter, fail to align with true object boundaries, and lose thin structures (cables, limbs, bicycle spokes).

---

### The RoIAlign Solution (He et al., ICCV 2017)
**RoIAlign** enforces a strict **Zero-Quantization Invariant**:
1. **Continuous Floating-Point Bins**: RoI coordinates and bin boundaries are never rounded ($x / 16.0$ is preserved as an exact floating-point value).
2. **Continuous Regular Sampling Points**: Each bin is divided into a continuous sub-grid (typically $2 \times 2 = 4$ sampling points per bin).
3. **Exact Bilinear Interpolation**: Feature values at each sampling point $(x, y)$ are evaluated via **bilinear interpolation** from the 4 surrounding discrete feature grid cells.
4. **Max / Average Pooling**: The 4 sampled values in each bin are aggregated into a single feature vector.
5. Boosts instance mask Average Precision by **$+3\text{ to }+5\text{ mAP}$** without adding any learnable parameters!

```
RoIPool vs. RoIAlign Sampling Topology:

RoIPool (Double Quantization & Spatial Drift):
   Continuous RoI (Float) -----> Floor(RoI / 16) -----> Floor(Bin / 7)
   (Severe misalignments up to 16 original image pixels!)

RoIAlign (Exact Sub-Pixel Bilinear Interpolation):
   Feature Map Grid (Discrete Integers)
         (x0, y0)             (x1, y0)
            +--------------------+
            |                    |
            |     o (x, y)       |  <--- Continuous Sampling Point (Float)
            |   (Bilinear        |       Evaluated from 4 neighbors!
            |  Interpolation)    |
            +--------------------+
         (x0, y1)             (x1, y1)
   (Zero quantization, exact sub-pixel mask alignment!)
```

---

## 2. Mathematical Formulation

### 2.1 The Bilinear Interpolation Operator
Let $(x, y) \in \mathbb{R}^2$ be a continuous sampling location on feature map $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$.
Let the four neighboring discrete integer lattice points be:

$$
x_0 = \lfloor x \rfloor, \quad x_1 = x_0 + 1, \quad y_0 = \lfloor y \rfloor, \quad y_1 = y_0 + 1
$$

The interpolated feature vector $\mathbf{f}(x, y) \in \mathbb{R}^C$ is:

$$
\mathbf{f}(x, y) = \sum_{i \in \{0, 1\}} \sum_{j \in \{0, 1\}} w(x, x_i) \cdot w(y, y_j) \cdot \mathbf{X}[:, y_j, x_i]
$$

where the 1D linear interpolation kernel is:

$$
w(a, a_k) = \max(0, \; 1 - |a - a_k|)
$$

Expanding explicitly:

$$
\mathbf{f}(x, y) = (x_1 - x)(y_1 - y) \mathbf{X}_{y_0, x_0} + (x - x_0)(y_1 - y) \mathbf{X}_{y_0, x_1} + (x_1 - x)(y - y_0) \mathbf{X}_{y_1, x_0} + (x - x_0)(y - y_0) \mathbf{X}_{y_1, x_1}
$$

Because the weights sum to $1$ ($\sum w_i = 1$), the interpolation is smooth and differentiable.

---

### 2.2 Backward Pass & Gradient Flow
During backpropagation, the loss gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{f}(x, y)}$ is distributed back to the four neighboring discrete feature pixels:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{X}[:, y_j, x_i]} \mathrel{+}= w(x, x_i) \cdot w(y, y_j) \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{f}(x, y)}
$$

This continuous gradient distribution allows the network to learn localization features via backpropagation.

---

## 3. Python Reference Implementation

```python
import torch
import torch.nn as nn

class CustomRoIAlign(nn.Module):
    """
    RoIAlign Reference Implementation.
    Extracts fixed (pooled_h, pooled_w) features from continuous floating-point boxes
    without spatial coordinate quantization.
    """
    def __init__(self, output_size: tuple[int, int] = (7, 7), spatial_scale: float = 1.0 / 16.0, 
                 sampling_ratio: int = 2):
        super().__init__()
        self.pooled_h, self.pooled_w = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio # 2x2 = 4 points per bin

    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        features: [B, C, H, W]
        rois: [N, 5] (batch_idx, x1, y1, x2, y2) in input image coordinate scale
        Returns: [N, C, pooled_h, pooled_w]
        """
        num_rois = rois.shape[0]
        c, h_feat, w_feat = features.shape[1], features.shape[2], features.shape[3]
        output = torch.zeros((num_rois, c, self.pooled_h, self.pooled_w), 
                             device=features.device, dtype=features.dtype)
        
        for n in range(num_rois):
            b_idx = int(rois[n, 0].item())
            # Scale coordinates to feature map without rounding!
            roi_x1 = rois[n, 1] * self.spatial_scale
            roi_y1 = rois[n, 2] * self.spatial_scale
            roi_x2 = rois[n, 3] * self.spatial_scale
            roi_y2 = rois[n, 4] * self.spatial_scale
            
            roi_w = max(roi_x2 - roi_x1, 1.0)
            roi_h = max(roi_y2 - roi_y1, 1.0)
            
            bin_size_w = roi_w / self.pooled_w
            bin_size_h = roi_h / self.pooled_h
            
            feat_map = features[b_idx] # [C, H, W]
            
            for ph in range(self.pooled_h):
                for pw in range(self.pooled_w):
                    # Sampling points within this bin
                    bin_x0 = roi_x1 + pw * bin_size_w
                    bin_y0 = roi_y1 + ph * bin_size_h
                    
                    val_sum = torch.zeros(c, device=features.device, dtype=features.dtype)
                    
                    for iy in range(self.sampling_ratio):
                        y = bin_y0 + (iy + 0.5) * (bin_size_h / self.sampling_ratio)
                        for ix in range(self.sampling_ratio):
                            x = bin_x0 + (ix + 0.5) * (bin_size_w / self.sampling_ratio)
                            
                            # Bilinear interpolation
                            x0 = torch.clamp(torch.floor(x).long(), 0, w_feat - 1)
                            x1 = torch.clamp(x0 + 1, 0, w_feat - 1)
                            y0 = torch.clamp(torch.floor(y).long(), 0, h_feat - 1)
                            y1 = torch.clamp(y0 + 1, 0, h_feat - 1)
                            
                            wx1 = x - x0.float()
                            wx0 = 1.0 - wx1
                            wy1 = y - y0.float()
                            wy0 = 1.0 - wy1
                            
                            sample = (wx0 * wy0 * feat_map[:, y0, x0] +
                                      wx1 * wy0 * feat_map[:, y0, x1] +
                                      wx0 * wy1 * feat_map[:, y1, x0] +
                                      wx1 * wy1 * feat_map[:, y1, x1])
                            val_sum += sample
                            
                    # Average pool sampled points
                    output[n, :, ph, pw] = val_sum / (self.sampling_ratio * self.sampling_ratio)
                    
        return output
```

---

## 4. Models in the Vault Utilizing RoIAlign

- **[[architectures/real-time-detectors-and-segmenters/mask2former|Mask2Former]]**: High-resolution mask query feature extraction.
- **[[architectures/pose-and-robotics-manipulation/cosypose|CosyPose]] & [[architectures/pose-and-robotics-manipulation/megapose|MegaPose]]**: Sub-pixel RoI feature extraction for 6-DoF render-and-compare pose hypothesis matching.
- **[[architectures/real-time-detectors-and-segmenters/grounding-dino|Grounding DINO]]**: Region feature extraction for text-token cross-attention grounding.

---
title: "All-Pairs Cross-Correlation Pyramids: Dense Visual Motion & Optical Flow Mechanics"
type: "Technique"
domain: "Visual Motion & Optical Flow"
tags:
  - technique
  - optical-flow
  - correlation-pyramid
  - raft
  - visual-tracking
  - motion-estimation
status: evergreen
updated: 2026-09-09
aliases:
  - "All-Pairs Correlation"
  - "Correlation Pyramid"
  - "RAFT Flow Mechanics"
  - "4D Correlation Volume"
---

# 🌊 All-Pairs Cross-Correlation Pyramids: Dense Visual Motion & Optical Flow

## 1. High-Level Concept & The Coarse-to-Fine Warping Failure

Estimating dense optical flow—the 2D velocity vector $\mathbf{f}(u, v) = (f_x, f_y)$ for every pixel between consecutive frames $I_1$ and $I_2$—is a core pillar of autonomous navigation, visual odometry, and dynamic obstacle tracking.

For over two decades, both classical variational algorithms and early deep neural networks (e.g., FlowNet, PWC-Net) relied on **coarse-to-fine feature warping pyramids**:
- They downsampled images into multi-scale pyramids, estimated large motions at low resolutions, and warped higher-resolution feature maps using the coarse estimates.

### The Fatal Flaw of Coarse-to-Fine Warping
Small, fast-moving objects (such as tennis balls, thin cables, traffic poles, or distant pedestrians) **completely vanish at coarse pyramid levels** (e.g., $1/16$ or $1/32$ scale). Once a small object is lost at the coarse stage, the error compounds across all subsequent scales, leaving the network incapable of ever recovering the true motion trajectory.

**All-Pairs Correlation Pyramids** (Teed & Deng, RAFT, ECCV 2020) resolved this failure mode:
1. **Full 4D Correlation Volume**: Computes the full pairwise dot product between *all pairs* of visual features across both frames at a single high resolution ($1/8$ scale), producing a complete 4D matching tensor $\mathbf{C} \in \mathbb{R}^{H \times W \times H \times W}$.
2. **Multi-Scale Correlation Pyramid**: Pools only the *target frame dimensions* across multiple scales, capturing both large displacements ($>100\text{ pixels}$) and sub-pixel micro-motions simultaneously.
3. **Iterative GRU Residual Updates**: Instead of regressing optical flow in a single pass, a lightweight recurrent unit (GRU) indexes local correlation features around the current flow estimate, incrementally optimizing the flow field like an analytical second-order solver.

```
Coarse-to-Fine Warping vs. All-Pairs Correlation:

Coarse-to-Fine Warping (PWC-Net):      All-Pairs Correlation Pyramid (RAFT):
   Level 4 (Small objects disappear!)         [ Frame 1 Feats ] x [ Frame 2 Feats ]
            |                                                |
            v                                                v
   Level 3 (Error compounds)                  [ Full 4D All-Pairs Volume ]
            |                                                |
            v                                                v
   Level 2 (Cannot recover)                   [ Correlation Pyramid Pool (1x, 2x, 4x, 8x) ]
            |                                                |
            v                                                v
   Level 1 (Failed motion)                    [ Iterative Recurrent GRU Updates ]
                                                 (Preserves small fast objects!)
```

---

## 2. Mathematical Formulation

### 2.1 4D Correlation Tensor Construction
Given two consecutive RGB video frames $I_1, I_2 \in \mathbb{R}^{3 \times H_{\text{img}} \times W_{\text{img}}}$, a shared convolutional feature extractor produces dense feature maps at $1/8$ spatial resolution:

$$
\mathbf{f}_1 = g_\theta(I_1) \in \mathbb{R}^{C \times H \times W}, \quad \mathbf{f}_2 = g_\theta(I_2) \in \mathbb{R}^{C \times H \times W}
$$

where typically $C = 256$, $H = H_{\text{img}}/8$, and $W = W_{\text{img}}/8$.

The **4D Correlation Volume** $\mathbf{C} \in \mathbb{R}^{H \times W \times H \times W}$ evaluates the normalized inner product between every pixel feature vector in $\mathbf{f}_1$ and every pixel feature vector in $\mathbf{f}_2$:

$$
\mathbf{C}(i, j, k, l) = \frac{1}{\sqrt{C}} \sum_{c=1}^C \mathbf{f}_1(c, i, j) \cdot \mathbf{f}_2(c, k, l)
$$

This matrix multiplication is vectorized in PyTorch as a single batched tensor contraction:

$$
\mathbf{C} = \frac{1}{\sqrt{C}} \mathbf{f}_1^T \mathbf{f}_2
$$

---

### 2.2 Correlation Pyramid Construction
To capture both broad global motion and fine local displacements without exploding memory, the last two dimensions (corresponding to frame 2) are pooled using 2D average pooling with kernel sizes and strides of $\{1, 2, 4, 8\}$:

$$
\mathbf{C}^{(m)} = \text{AvgPool2D}_{2^{m-1}}\left( \mathbf{C} \right) \in \mathbb{R}^{H \times W \times \frac{H}{2^{m-1}} \times \frac{W}{2^{m-1}}}, \quad m \in \{1, 2, 3, 4\}
$$

- Level 1 provides high-resolution sub-pixel local correlation.
- Level 4 provides low-resolution global matching context covering massive camera translations.

---

### 2.3 The Local Correlation Lookup Operator
Let $\mathbf{f}_k \in \mathbb{R}^{2 \times H \times W}$ denote the current optical flow estimate at recurrent step $k$.
For each source pixel $\mathbf{x} = (i, j)$, its estimated destination in frame 2 is:

$$
\mathbf{x}' = \mathbf{x} + \mathbf{f}_k(\mathbf{x})
$$

We define a local search neighborhood grid $\mathcal{N}_r(\mathbf{x}')$ of radius $r$ (typically $r = 4$, yielding $(2r+1)^2 = 81$ grid points):

$$
\mathcal{N}_r(\mathbf{x}') = \left\{ \mathbf{x}' + \mathbf{d} \;\middle|\; \mathbf{d} \in \mathbb{Z}^2, \; \|\mathbf{d}\|_\infty \le r \right\}
$$

For each pyramid level $m$, the destination coordinate is scaled by $2^{-(m-1)}$, and the correlation values are sampled using **bilinear interpolation**:

$$
\text{Lookup}^{(m)}(\mathbf{x}) = \text{BilinearSample}\left( \mathbf{C}^{(m)}(i, j, \cdot, \cdot), \; \frac{\mathcal{N}_r(\mathbf{x}')}{2^{m-1}} \right)
$$

Concatenating the lookups across all 4 pyramid levels yields a compact feature vector of size $4 \times (2r + 1)^2 = 324$ values per pixel, providing the recurrent unit with comprehensive multi-scale motion feedback.

---

## 3. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CorrelationVolume4D(nn.Module):
    """
    4D All-Pairs Correlation Volume and Multi-Scale Pyramid.
    Computes pairwise inner products between all feature tokens and performs
    bilinear local neighborhood lookup around current flow estimates.
    """
    def __init__(self, num_levels: int = 4, radius: int = 4):
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius

    def forward(self, f1: torch.Tensor, f2: torch.Tensor):
        # f1, f2: [B, C, H, W]
        b, c, h, w = f1.shape
        f1_flat = f1.view(b, c, h * w)
        f2_flat = f2.view(b, c, h * w)
        
        # 1. 4D Correlation Volume: [B, H, W, H, W]
        corr = torch.bmm(f1_flat.transpose(1, 2), f2_flat) / (c ** 0.5)
        corr = corr.view(b, h, w, 1, h, w)
        
        # 2. Build Multi-Scale Pyramid by pooling over frame 2 coordinates
        pyramid = [corr.squeeze(3)]
        curr_corr = corr.view(b * h * w, 1, h, w)
        
        for _ in range(self.num_levels - 1):
            curr_corr = F.avg_pool2d(curr_corr, kernel_size=2, stride=2)
            pyramid.append(curr_corr.view(b, h, w, curr_corr.shape[-2], curr_corr.shape[-1]))
            
        return pyramid

    def lookup(self, pyramid: list, flow: torch.Tensor) -> torch.Tensor:
        """
        Samples local neighborhood of radius r around (coords + flow) across all pyramid levels.
        flow: [B, 2, H, W]
        Returns: [B, num_levels * (2r+1)^2, H, W]
        """
        b, _, h, w = flow.shape
        # Base meshgrid
        y, x = torch.meshgrid(torch.arange(h, device=flow.device), torch.arange(w, device=flow.device), indexing="ij")
        coords = torch.stack([x, y], dim=0).float().unsqueeze(0).repeat(b, 1, 1, 1)
        coords_dest = coords + flow  # [B, 2, H, W]
        
        # Displacement offsets grid
        dx = torch.linspace(-self.radius, self.radius, 2 * self.radius + 1, device=flow.device)
        dy = torch.linspace(-self.radius, self.radius, 2 * self.radius + 1, device=flow.device)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), dim=-1).view(-1, 2) # [K, 2]
        
        out_pyramid = []
        for i, corr_level in enumerate(pyramid):
            # Scale coordinates to pyramid level
            scale = 2 ** i
            sample_coords = (coords_dest.permute(0, 2, 3, 1).unsqueeze(-2) + delta) / scale
            # Normalize to [-1, 1] for grid_sample
            h_lvl, w_lvl = corr_level.shape[-2:]
            grid_norm = torch.empty_like(sample_coords)
            grid_norm[..., 0] = 2.0 * sample_coords[..., 0] / max(w_lvl - 1, 1) - 1.0
            grid_norm[..., 1] = 2.0 * sample_coords[..., 1] / max(h_lvl - 1, 1) - 1.0
            
            # Sample correlation values
            corr_flat = corr_level.view(b * h * w, 1, h_lvl, w_lvl)
            grid_flat = grid_norm.view(b * h * w, 1, (2 * self.radius + 1)**2, 2)
            sampled = F.grid_sample(corr_flat, grid_flat, align_corners=True, mode="bilinear")
            out_pyramid.append(sampled.view(b, h, w, -1).permute(0, 3, 1, 2))
            
        return torch.cat(out_pyramid, dim=1)
```

---

## 4. Models in the Vault Utilizing All-Pairs Correlation

- **[[architectures/visual-tracking-and-flow/raft|RAFT]]**: Pioneering all-pairs optical flow architecture.
- **[[cookbooks/19-raft-optical-flow/README|Cookbook 19: RAFT Optical Flow]]**: Standalone runnable script for dense bidirectional pixel motion estimation.
- **[[architectures/visual-tracking-and-flow/tapir|TAPIR]]**: Tracking Any Point with all-pairs correlation pyramids and uncertainty estimation.
- **[[architectures/spatial-radiance-and-slam/droid-slam|DROID-SLAM]]**: Employs correlation pyramids inside recurrent visual bundle adjustment.

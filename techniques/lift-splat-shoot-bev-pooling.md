---
title: "Lift-Splat-Shoot (LSS) & Frustum BEV Voxel Pooling: Multi-Camera 3D Transformation"
type: "Technique"
domain: "Autonomous Driving & Multi-Modal Sensor Fusion"
tags:
  - technique
  - bev-pooling
  - lift-splat-shoot
  - lss
  - sensor-fusion
  - autonomous-driving
  - 3d-perception
status: evergreen
updated: 2026-09-09
aliases:
  - "Lift Splat Shoot"
  - "LSS"
  - "BEV Pooling"
  - "Frustum Voxel Pooling"
  - "Camera-to-BEV Transformation"
---

# 🚗 Lift-Splat-Shoot (LSS) & Frustum BEV Voxel Pooling: Multi-Camera 3D Transformation

## 1. High-Level Concept & The Perspective-to-Metric Dilemma

In autonomous vehicles and mobile robotics, surround cameras capture images in **perspective projection space** $(u, v) \in \mathbb{R}^2$. However, motion planning, trajectory tracking, and collision avoidance operate in a **metric, gravity-aligned ego-vehicle coordinate frame** $(X, Y, Z) \in \mathbb{R}^3$, specifically in the **Bird's-Eye-View (BEV)** plane $(X, Y)$.

Transforming 2D pixels into 3D metric coordinates is fundamentally ill-posed: along any line of sight through a pixel $(u, v)$, there are infinite 3D points $(X, Y, Z)$ that could have generated that pixel.

**Lift-Splat-Shoot (LSS)** (Philion & Baker, ECCV 2020) resolved this ambiguity by introducing a differentiable, end-to-end multi-view geometric transform:
1. **Lift**: Predict a categorical depth probability distribution across discrete depth bins for every pixel, "lifting" each 2D feature vector into a 1D frustum ray of 3D points.
2. **Splat**: Project the frustum points into discrete metric 3D voxels using calibrated camera intrinsics $\mathbf{K}$ and extrinsics $(\mathbf{R}, \mathbf{t})$.
3. **Shoot / Pool**: Aggregate (sum/max pool) all features falling within the same metric BEV pillar $(X, Y)$ into a unified spatial tensor, ready for downstream 3D detection and map segmentation.

```
Lift-Splat-Shoot Pipeline:

[ 2D Feature Map ]            [ Depth Distribution ]
 (C x H x W)                   (D x H x W)
         \                         /
          v                       v
      [ Lift: Outer Product F x D ]
         (C x D x H x W Frustum Point Cloud)
                     |
                     v
      [ Splat: Extrinsic & Intrinsic Projection ]
         (Map each 3D point to metric Voxel Index: ix, iy, iz)
                     |
                     v
      [ Shoot: Fast Cumulative Sum Pooling ]
         (Aggregate all points in same (ix, iy) column)
                     |
                     v
      [ Unified Metric BEV Grid (C x X_dim x Y_dim) ]
```

---

## 2. Mathematical Formulation

### 2.1 Coordinate Frame Transformations
Let $\mathbf{K} \in \mathbb{R}^{3 \times 3}$ denote the calibrated camera intrinsic matrix and $(\mathbf{R}, \mathbf{t}) \in \mathrm{SE}(3)$ denote the camera-to-ego extrinsic transformation.

A 2D pixel coordinate $(u, v)$ with assumed scalar metric depth $d$ unprojects to a 3D point in the camera frame $\mathbf{p}_{\text{cam}}$:

$$
\mathbf{p}_{\text{cam}}(u, v, d) = d \cdot \mathbf{K}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

Transforming from camera coordinates to vehicle ego coordinates $\mathbf{p}_{\text{ego}} = [X, Y, Z]^T$:

$$
\mathbf{p}_{\text{ego}}(u, v, d) = \mathbf{R}_{\text{cam} \to \text{ego}} \mathbf{p}_{\text{cam}}(u, v, d) + \mathbf{t}_{\text{cam} \to \text{ego}}
$$

---

### 2.2 The "Lift" Step: Categorical Depth Distribution
Rather than estimating a single deterministic depth scalar per pixel (which is fragile under reflection, transparent glass, or darkness), LSS discretizes continuous depth $[d_{\min}, d_{\max}]$ into $D$ uniform or logarithmic depth bins:

$$
d_k = d_{\min} + k \cdot \Delta d, \quad k \in \{0, 1, \dots, D-1\}
$$

A convolutional head predicts both:
1. Context features $\mathbf{f}(u, v) \in \mathbb{R}^C$.
2. Categorical depth probabilities $\mathbf{p}(u, v) \in \Delta^D$, where $\sum_{k=0}^{D-1} p_k(u, v) = 1$.

The lifted 3D frustum feature point $\mathbf{c}(u, v, k)$ at depth bin $d_k$ is the outer product:

$$
\mathbf{c}(u, v, k) = p_k(u, v) \cdot \mathbf{f}(u, v) \in \mathbb{R}^C
$$

---

### 2.3 The "Splat" & "Shoot" Step: Voxel Quantization & Pooling
The metric BEV space is discretized into a uniform grid defined by spatial boundaries $[X_{\min}, X_{\max}] \times [Y_{\min}, Y_{\max}] \times [Z_{\min}, Z_{\max}]$ with voxel resolutions $(\Delta X, \Delta Y, \Delta Z)$.

Each 3D frustum point $\mathbf{p}_{\text{ego}}(u, v, d_k) = [X, Y, Z]^T$ is assigned to a discrete discrete voxel coordinate $(i, j, k)$:

$$
i = \left\lfloor \frac{X - X_{\min}}{\Delta X} \right\rfloor, \quad j = \left\lfloor \frac{Y - Y_{\min}}{\Delta Y} \right\rfloor, \quad k = \left\lfloor \frac{Z - Z_{\min}}{\Delta Z} \right\rfloor
$$

All features from all surround cameras that map to the same horizontal BEV pillar $(i, j)$ are pooled:

$$
\mathbf{F}_{\text{BEV}}(i, j) = \sum_{(u, v, k) \in \mathcal{V}_{(i, j)}} \mathbf{c}(u, v, k)
$$

---

## 3. The "Cumsum Trick" for Real-Time Execution

In naive GPU implementations, accumulating millions of frustum points into sparse voxels requires atomic operations (`atomicAdd`), causing severe thread divergence and memory serialization:

| Implementation | Aggregation Mechanism | Latency (6 Cameras, 4K) | Edge Feasibility |
| :--- | :--- | :---: | :--- |
| **Naive LSS (2020)** | PyTorch `scatter_add` / CPU loops | $\approx 45.0\text{ ms}$ | Non-viable for real-time |
| **Cumsum Trick (LSS)** | Sort points by voxel ID, compute 1D cumulative sum, take boundary differences | $\approx 12.5\text{ ms}$ | Barely viable (30 FPS) |
| **Fast BEV Pooling (BEVFusion)** | Pre-computed static spatial hash index + specialized fused CUDA kernel | **$< 3.2\text{ ms}$** | **Optimal for edge real-time** |

### Mathematical Principle of the Cumsum Trick
1. Flatten each voxel coordinate $(i, j)$ into a single 1D index: $\text{rank} = i \cdot N_y + j$.
2. Sort all frustum points in ascending order of their rank.
3. Compute the 1D cumulative sum (`cumsum`) along the sorted features: $\mathbf{S}_m = \sum_{n=1}^m \mathbf{c}_n$.
4. Find boundary indices where the rank changes ($\text{rank}_m \ne \text{rank}_{m+1}$). The sum of features within a voxel spanning indices $[a, b]$ is simply:
   $$
   \sum_{n=a}^b \mathbf{c}_n = \mathbf{S}_b - \mathbf{S}_{a-1}
   $$
This replaces millions of random-access atomic writes with a single coalesced parallel prefix sum.

---

## 4. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FrustumBEVPooling(nn.Module):
    """
    Lift-Splat-Shoot Frustum Voxel Pooling module.
    Transforms multi-view 2D camera features into a metric 3D Bird's-Eye-View grid.
    """
    def __init__(self, x_bounds=(-50.0, 50.0, 0.5), y_bounds=(-50.0, 50.0, 0.5), z_bounds=(-5.0, 5.0, 10.0), d_bounds=(1.0, 60.0, 1.0)):
        super().__init__()
        self.x_min, self.x_max, self.dx = x_bounds
        self.y_min, self.y_max, self.dy = y_bounds
        self.z_min, self.z_max, self.dz = z_bounds
        self.d_min, self.d_max, self.dd = d_bounds
        
        self.nx = int((self.x_max - self.x_min) / self.dx)
        self.ny = int((self.y_max - self.y_min) / self.dy)
        self.d_bins = torch.arange(self.d_min, self.d_max, self.dd, dtype=torch.float32)
        self.num_depth_bins = len(self.d_bins)

    def lift_features(self, feat_2d: torch.Tensor, depth_logits: torch.Tensor):
        """
        Outer-product between context features and categorical depth probabilities.
        feat_2d: [B, N_cam, C, H, W]
        depth_logits: [B, N_cam, D, H, W]
        Returns: [B, N_cam, C, D, H, W]
        """
        depth_probs = F.softmax(depth_logits, dim=2)
        # Outer product via broadcasting
        return feat_2d.unsqueeze(3) * depth_probs.unsqueeze(2)

    def splat_to_bev(self, frustum_features: torch.Tensor, geom_coords: torch.Tensor):
        """
        Pools lifted points into discrete BEV voxels.
        geom_coords: [B, N_cam, D, H, W, 3] containing integer (ix, iy, iz) indices
        frustum_features: [B, N_cam, C, D, H, W]
        Returns: [B, C, ny, nx] metric BEV feature map
        """
        b, n_cam, c, d, h, w = frustum_features.shape
        bev_grid = torch.zeros((b, c, self.ny, self.nx), dtype=frustum_features.dtype, device=frustum_features.device)
        
        # Flatten spatial points
        feats_flat = frustum_features.permute(0, 2, 1, 3, 4, 5).reshape(b, c, -1)
        coords_flat = geom_coords.reshape(b, -1, 3)
        
        # Filter points within valid BEV bounds
        valid_mask = (
            (coords_flat[..., 0] >= 0) & (coords_flat[..., 0] < self.nx) &
            (coords_flat[..., 1] >= 0) & (coords_flat[..., 1] < self.ny)
        )
        
        for batch_idx in range(b):
            valid = valid_mask[batch_idx]
            if not valid.any():
                continue
            x_idx = coords_flat[batch_idx, valid, 0]
            y_idx = coords_flat[batch_idx, valid, 1]
            f_vals = feats_flat[batch_idx, :, valid]
            
            # Scatter addition into BEV grid
            bev_grid[batch_idx].index_put_((slice(None), y_idx, x_idx), f_vals, accumulate=True)
            
        return bev_grid
```

---

## 5. Models & Cookbooks Utilizing BEV Pooling

- **[[architectures/3d-pointclouds-and-lidar/bevfusion|BEVFusion]]**: Modern camera-LiDAR fusion standard executing Fast BEV Pooling in $<4\text{ ms}$.
- **[[cookbooks/14-bev-voxel-pooling-projection/README|Cookbook 14: BEV Voxel Pooling]]**: Standalone runnable Python recipe for multi-camera frustum projection.
- **[[architectures/3d-pointclouds-and-lidar/stream-petr|StreamPETR]] & [[architectures/3d-pointclouds-and-lidar/sparse4d|Sparse4D]]**: Temporal extensions utilizing sparse query sampling over BEV features.
- **[[topics/sensor-fusion/00-sensor-fusion-moc|Sensor Fusion MOC]]**: Connected as the primary multi-camera projection backbone.

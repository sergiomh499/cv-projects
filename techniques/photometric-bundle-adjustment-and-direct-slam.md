---
title: "Photometric Bundle Adjustment & Direct SLAM: Continuous Intensity Optimization"
type: "Technique"
domain: "Robotics, Spatial Radiance & Visual Odometry"
tags:
  - technique
  - direct-slam
  - photometric-bundle-adjustment
  - dso
  - droid-slam
  - visual-odometry
  - se3-optimization
status: evergreen
updated: 2026-09-09
aliases:
  - "Direct SLAM"
  - "Photometric Bundle Adjustment"
  - "DSO"
  - "Photometric Error"
  - "Direct Visual Odometry"
---

# 👁️ Photometric Bundle Adjustment & Direct SLAM: Continuous Intensity Optimization

## 1. High-Level Concept & Feature-Based vs. Direct Paradigms

Classical Visual SLAM and Structure-from-Motion systems (e.g., ORB-SLAM3, COLMAP) operate on **sparse geometric features** (ORB, SIFT, SuperPoint):
- They detect corner points, compute high-dimensional descriptor vectors, match correspondences between image pairs, and minimize geometric **reprojection error**:
  $$E_{\text{geom}} = \sum_{(i, j)} \rho_{\text{Huber}}\left( \left\| \mathbf{x}_{ij} - \pi(\mathbf{T}_{ji} \mathbf{X}_i) \right\|_2 \right)$$
- **The Catastrophic Failure Modes of Feature-Based Methods**:
  1. **Low-Texture Environments**: Clean industrial corridors, drywall rooms, tunnels, and open roads lack distinct high-frequency corners, causing feature matchers to return zero valid correspondences.
  2. **Motion Blur & Illumination Fluctuations**: Rapid camera rotations blur corner descriptors, leading to tracking loss.

---

### The Direct Photometric SLAM Paradigm (Engel et al., PAMI 2017; Teed et al., NeurIPS 2021)
Direct Visual SLAM (**DSO**, **DROID-SLAM**, **3DGS-SLAM**) dispenses with keypoints and descriptors entirely!
Instead, it optimizes camera motion $\mathbf{T}_{ji} \in \mathrm{SE}(3)$ and scene geometry directly on **raw pixel brightness values**:

$$
E_{\text{photo}} = \sum_{\mathbf{p} \in \Omega} \left( I_j(\mathbf{p}') - I_i(\mathbf{p}) \right)^2
$$

Every pixel with non-zero image gradient $\nabla I(\mathbf{p}) \ne \mathbf{0}$ contributes to camera tracking, enabling dense, drift-free odometry across smooth, texture-sparse surfaces.

---

### Affine Brightness Transfer & Auto-Exposure Compensation
Physical cameras continuously alter their electronic shutter exposure time and analog sensor gain. A simple brightness difference $I_j - I_i$ fails when lighting changes.
DSO models photometric calibration by assigning each frame $i$ two **affine brightness parameters** $(a_i, b_i) \in \mathbb{R}^2$:

$$
I_{\text{calibrated}}(\mathbf{p}) = \frac{I(\mathbf{p}) - b_i}{e^{a_i}}
$$

The robust photometric error evaluates:

$$
r_{\mathbf{p}} = \left( I_j(\mathbf{p}') - b_j \right) - \frac{e^{a_j}}{e^{a_i}} \left( I_i(\mathbf{p}) - b_i \right)
$$

```
Photometric Bundle Adjustment Pipeline:

Reference Frame I_i                                 Target Frame I_j
[ Pixel p = (u, v) ]                               [ Warped Pixel p' = pi(T_ji * P_i) ]
        |                                                   |
        v Inverse Depth d_p                                 v Interpolate Intensity
[ 3D Point P_i = K^-1 * [u, v, 1]^T / d_p ]                 [ Bilinear Sample I_j(p') ]
        |                                                   |
        +-------------------------+-------------------------+
                                  |
                                  v
              [ Affine Brightness Correction (a, b) ]
                                  |
                                  v
              [ Photometric Residual r_p = I_j(p') - I_i(p) ]
                                  |
                                  v
              [ Analytical Jacobian J = [grad_I * d_pi * d_SE3] ]
                                  |
                                  v
              [ Gauss-Newton Update on Lie Tangent Twist xi in se(3) ]
```

---

## 2. Mathematical Formulation

### 2.1 The 3D Warping Operator
Let point $\mathbf{p} = [u, v]^T$ in frame $i$ have inverse depth $d_{\mathbf{p}} = 1/z$.
Its 3D Euclidean coordinate in camera frame $i$ is:

$$
\mathbf{P}_i = \frac{1}{d_{\mathbf{p}}} \mathbf{K}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

Transforming to target camera frame $j$ via rigid transformation $\mathbf{T}_{ji} = (\mathbf{R}, \mathbf{t}) \in \mathrm{SE}(3)$:

$$
\mathbf{P}_j = \mathbf{R} \mathbf{P}_i + \mathbf{t} = \begin{bmatrix} X_j \\ Y_j \\ Z_j \end{bmatrix}
$$

The projected 2D coordinate in frame $j$ is evaluated via pinhole projection $\pi$:

$$
\mathbf{p}' = \pi(\mathbf{P}_j) = \begin{bmatrix} f_x \frac{X_j}{Z_j} + c_x \\ f_y \frac{Y_j}{Z_j} + c_y \end{bmatrix}
$$

---

### 2.2 Analytical Photometric Jacobians Over $\mathfrak{se}(3)$
Camera pose perturbation $\boldsymbol{\xi} = [\boldsymbol{\upsilon}, \boldsymbol{\omega}]^T \in \mathbb{R}^6$ is parameterized on the Lie algebra $\mathfrak{se}(3)$ using the matrix exponential $\mathbf{T}_{ji} \leftarrow \exp(\boldsymbol{\xi}^\wedge) \mathbf{T}_{ji}$.

By applying the chain rule:

$$
\mathbf{J}_{\boldsymbol{\xi}} = \frac{\partial r_{\mathbf{p}}}{\partial \boldsymbol{\xi}} = \frac{\partial I_j}{\partial \mathbf{p}'} \cdot \frac{\partial \mathbf{p}'}{\partial \mathbf{P}_j} \cdot \frac{\partial \mathbf{P}_j}{\partial \boldsymbol{\xi}}
$$

1. **Image Gradient**: $\frac{\partial I_j}{\partial \mathbf{p}'} = \nabla I_j(\mathbf{p}') = \begin{bmatrix} \frac{\partial I_j}{\partial u'} & \frac{\partial I_j}{\partial v'} \end{bmatrix} \in \mathbb{R}^{1 \times 2}$.
2. **Pinhole Projection Jacobian**:
   $$
   \frac{\partial \mathbf{p}'}{\partial \mathbf{P}_j} = \begin{bmatrix} \frac{f_x}{Z_j} & 0 & -\frac{f_x X_j}{Z_j^2} \\ 0 & \frac{f_y}{Z_j} & -\frac{f_y Y_j}{Z_j^2} \end{bmatrix} \in \mathbb{R}^{2 \times 3}
   $$
3. **Rigid Transformation Jacobian**:
   $$
   \frac{\partial \mathbf{P}_j}{\partial \boldsymbol{\xi}} = \begin{bmatrix} \mathbf{I}_{3 \times 3} & -\mathbf{P}_j^\wedge \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 & Z_j & -Y_j \\ 0 & 1 & 0 & -Z_j & 0 & X_j \\ 0 & 0 & 1 & Y_j & -X_j & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 6}
   $$

The image intensity gradient $\nabla I_j(\mathbf{p}')$ acts directly as the steering force guiding Gauss-Newton convergence:

$$
\left( \sum \mathbf{J}^T \mathbf{W} \mathbf{J} \right) \Delta\boldsymbol{\xi} = -\sum \mathbf{J}^T \mathbf{W} r_{\mathbf{p}}
$$

---

## 3. PyTorch Reference Implementation

```python
import torch
import torch.nn.functional as F

def pinhole_warp(ref_img: torch.Tensor, target_img: torch.Tensor, depth_ref: torch.Tensor,
                 K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Direct Photometric warping and residual computation.
    ref_img, target_img: [B, 1, H, W]
    depth_ref: [B, 1, H, W]
    K: [B, 3, 3] intrinsic matrix
    R: [B, 3, 3] rotation matrix
    t: [B, 3, 1] translation vector
    Returns: (photometric_residual [B, 1, H, W], valid_mask [B, 1, H, W])
    """
    B, _, H, W = ref_img.shape
    device = ref_img.device

    # 1. Generate 2D pixel grid coordinates
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    p2d = torch.stack([x, y, torch.ones_like(x)], dim=0).float().unsqueeze(0).repeat(B, 1, 1, 1) # [B, 3, H, W]

    # 2. Unproject to 3D camera coordinates in reference frame
    K_inv = torch.inverse(K)
    p3d_ref = torch.matmul(K_inv, p2d.view(B, 3, -1)).view(B, 3, H, W) * depth_ref # [B, 3, H, W]

    # 3. Transform to target camera frame: P_target = R * P_ref + t
    p3d_target = torch.matmul(R, p3d_ref.view(B, 3, -1)) + t
    p3d_target = p3d_target.view(B, 3, H, W)

    # 4. Project back onto target 2D image canvas
    Z_target = p3d_target[:, 2:3, :, :].clamp(min=1e-3)
    p2d_target = torch.matmul(K, p3d_target.view(B, 3, -1)).view(B, 3, H, W)
    u_norm = (p2d_target[:, 0:1, :, :] / Z_target) / (W - 1) * 2.0 - 1.0
    v_norm = (p2d_target[:, 1:2, :, :] / Z_target) / (H - 1) * 2.0 - 1.0
    grid = torch.cat([u_norm, v_norm], dim=1).permute(0, 2, 3, 1) # [B, H, W, 2]

    # 5. Bilinear sample target image intensity
    warped_target = F.grid_sample(target_img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    # 6. Photometric Residual
    residual = warped_target - ref_img

    # Valid mask (points inside frustum with positive depth)
    valid_mask = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0) & \
                 (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0) & \
                 (p3d_target[:, 2, :, :] > 0.1)
    valid_mask = valid_mask.unsqueeze(1).float()

    return residual * valid_mask, valid_mask
```

---

## 4. Models in the Vault Utilizing Photometric Bundle Adjustment

- **[[architectures/spatial-radiance-and-slam/droid-slam|DROID-SLAM]]**: Deep recurrent visual SLAM utilizing correlation-driven photometric bundle adjustment layers.
- **[[architectures/spatial-radiance-and-slam/3dgs-slam|3DGS-SLAM]] & [[architectures/spatial-radiance-and-slam/splatam|SplaTAM]]**: Tracking camera poses by backpropagating photometric rendering losses through 3D Gaussian radiance fields.
- **[[topics/slam-and-spatial-perception/README|SLAM & Spatial Perception Playbook]]**: Cataloged as the primary continuous intensity odometry formulation.
- **[[techniques/bundle-adjustment-and-schur-complement|Bundle Adjustment & Schur Complement]]**: Landmark marginalization mechanics for large-scale camera graphs.

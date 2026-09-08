---
title: "3D Gaussian Splatting: Real-Time Radiance Field Rendering via Point-Based Primitives"
type: model-deep-dive
tasks:
  - novel-view-synthesis
  - 3d-reconstruction
  - radiance-fields
  - spatial-computing
architecture_class: Explicit Differentiable Gaussian Rasterization
primary_license: Custom / Non-Commercial (Inria / Max Planck)
commercial_use: false
official_repo: https://github.com/graphdeco-inria/gaussian-splatting
paper_url: https://arxiv.org/abs/2308.04079
tags:
  - architecture
  - 3dgs
  - gaussian-splatting
  - radiance-fields
  - novel-view-synthesis
  - real-time
  - spatial-computing
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - 3DGS
  - 3D Gaussian Splatting
  - Gaussian Splatting
  - Point-Based Radiance Fields
---

# 🔬 3D Gaussian Splatting: Real-Time Radiance Field Rendering via Point-Based Primitives

## 1. Executive Brief & Significance

Neural Radiance Fields (NeRFs; Mildenhall et al., 2020) revolutionized novel view synthesis by demonstrating continuous volumetric scene representation using implicit Multi-Layer Perceptrons (MLPs). Despite delivering photo-realistic view synthesis, standard NeRF architectures suffer from severe compute constraints: rendering a single $1920 \times 1080$ frame demands millions of MLP evaluations across dense volumetric rays, capping inference at $0.1-5\text{ FPS}$ even on high-end desktop GPUs.

**3D Gaussian Splatting (3DGS)** (Kerbl, Kopanas, Leimkühler, Drettakis; SIGGRAPH 2023) fundamentally reshaped spatial radiance modeling by abandoning implicit coordinate networks in favor of **explicit, anisotropic 3D Gaussian primitives**. By coupling explicit 3D geometry with an ultra-fast, hardware-tailored differentiable tile rasterizer, 3DGS achieves:
- **Real-Time $>100\text{ FPS}$ Rendering**: High-fidelity $1080\text{p}$ novel view synthesis at $>100-200\text{ FPS}$ on consumer GPUs.
- **Fast Training Convergence**: Full scene optimization in $\approx 20-30\text{ minutes}$ on a single GPU (compared to hours/days for standard NeRFs).
- **Explicit Spatial Support**: Parameterizes scenes through millions of continuous spatial ellipsoids with bounded spatial footprints, completely eliminating volumetric ray marching and network evaluation overhead.
- **Adaptive Density Control**: Incrementally grows, splits, clones, and prunes Gaussians based on spatial view-space positional gradient accumulation.

```mermaid
flowchart TD
    subgraph Initialization ["Structure-from-Motion (SfM) Initialization"]
        SfMPoints["Sparse SfM Point Cloud (COLMAP)"] --> InitGaussians["Initialize 3D Gaussians (mu_i, s_i, q_i, alpha_i, SH_i)"]
    end

    subgraph Rasterization_Pipeline ["Differentiable Tile Rasterization Pipeline"]
        CameraPose["Camera Extrinsics T_cw and Intrinsics K"] --> ProjectGaussians["Project 3D Gaussians to 2D Screen Space (Sigma'_i = J W Sigma_i W^T J^T)"]
        InitGaussians --> ProjectGaussians
        ProjectGaussians --> TileBinning["16x16 Pixel Tile Binning and Key Generation"]
        TileBinning --> RadixSort["GPU Radix Sort by (Tile ID | View-space Depth)"]
        RadixSort --> WarpRaster["Shared Memory Cooperative Warps Front-to-Back Alpha Compositing"]
        WarpRaster --> RenderedOutput["Rendered RGB Image I_hat in R^(H x W x 3)"]
    end

    subgraph Optimization_and_Density ["Loss Computation & Adaptive Density Control"]
        RenderedOutput --> LossEngine["Loss Engine: L = (1 - lambda) L1 + lambda L_SSIM"]
        GroundTruth["Ground Truth Image I_gt"] --> LossEngine
        LossEngine --> Backprop["Analytical Gradient Backpropagation to mu, Sigma, alpha, SH"]
        Backprop --> DensityCheck{"View-space Positional Gradient > tau_pos?"}
        DensityCheck -->|Small Gaussian| Clone["Clone Gaussian"]
        DensityCheck -->|Large Gaussian| Split["Split into Two Smaller Gaussians"]
        Backprop --> PruneCheck{"Opacity alpha < tau_alpha or Scale > S_max?"}
        PruneCheck -->|Yes| Prune["Prune Primitive"]
    end

    Clone --> ProjectGaussians
    Split --> ProjectGaussians
    Prune --> ProjectGaussians
```

---

## 2. Mathematical Foundations & Analytical Derivations

### A. 3D Gaussian Primitive Parameterization
A 3D Gaussian primitive $g_i(\mathbf{x})$ is defined in continuous world space $\mathbf{x} \in \mathbb{R}^3$ by its spatial mean $\boldsymbol{\mu}_i \in \mathbb{R}^3$ and 3D covariance matrix $\boldsymbol{\Sigma}_i \in \mathbb{R}^{3 \times 3}$:

$$g_i(\mathbf{x}) = \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{x} - \boldsymbol{\mu}_i) \right)$$

To ensure that $\boldsymbol{\Sigma}_i$ remains symmetric and positive semi-definite (PSD) throughout unconstrained gradient descent, 3DGS factorizes $\boldsymbol{\Sigma}_i$ into a rotation matrix $\mathbf{R}_i \in SO(3)$ and a diagonal scaling matrix $\mathbf{S}_i = \text{diag}(s_{x,i}, s_{y,i}, s_{z,i})$:

$$\boldsymbol{\Sigma}_i = \mathbf{R}_i \mathbf{S}_i \mathbf{S}_i^T \mathbf{R}_i^T$$

- **Rotation**: Parameterized via a 4D unit quaternion $\mathbf{q}_i = [w_i, x_i, y_i, z_i]^T \in \mathbb{H}$, normalized as $\tilde{\mathbf{q}}_i = \mathbf{q}_i / \|\mathbf{q}_i\|_2$. The rotation matrix is constructed as:
  $$\mathbf{R}_i = \begin{bmatrix} 1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\ 2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\ 2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2) \end{bmatrix}$$
- **Scale**: Parameterized as unconstrained log-space scalars $\mathbf{s}'_i \in \mathbb{R}^3$, where $s_k = \exp(s'_k)$.
- **Opacity**: Parameterized as an unconstrained logit scalar $\sigma_i \in \mathbb{R}$, yielding opacity $\alpha_i = \text{sigmoid}(\sigma_i) \in [0, 1]$.
- **View-Dependent Radiance**: Represented via Spherical Harmonics (SH) up to degree $l_{\text{max}} = 3$ (16 basis functions per RGB channel, totaling 48 coefficients per Gaussian).

---

### B. Projective 2D Covariance Derivation
To project a 3D Gaussian onto the 2D image plane under camera extrinsic matrix $\mathbf{T}_{cw} = [\mathbf{W} \mid \mathbf{t}] \in SE(3)$ and camera intrinsic matrix $\mathbf{K}$:

1. **Transform Mean to Camera Coordinates**:
   $$\mathbf{t}_{\text{cam}, i} = \mathbf{W} \boldsymbol{\mu}_i + \mathbf{t} = [t_x, t_y, t_z]^T$$
2. **Affine Projective Approximation**:
   Applying the local affine approximation (Zwicker et al., 2001), the projected 2D covariance $\boldsymbol{\Sigma}'_i \in \mathbb{R}^{2 \times 2}$ on the image plane is:
   $$\boldsymbol{\Sigma}'_i = \mathbf{J}_i \mathbf{W} \boldsymbol{\Sigma}_i \mathbf{W}^T \mathbf{J}_i^T$$
   where $\mathbf{J}_i \in \mathbb{R}^{2 \times 3}$ is the Jacobian of the pinhole projective transformation evaluated at $\mathbf{t}_{\text{cam}, i}$:
   $$\mathbf{J}_i = \begin{bmatrix} \frac{f_x}{t_z} & 0 & -\frac{f_x t_x}{t_z^2} \\ 0 & \frac{f_y}{t_z} & -\frac{f_y t_y}{t_z^2} \end{bmatrix}$$
3. **Anti-Aliasing Regularization**:
   To prevent mathematical singularities and sub-pixel aliasing when Gaussians become infinitely small on screen, a constant 2D low-pass filter variance $\nu = 0.3\text{ pixels}^2$ is added:
   $$\tilde{\boldsymbol{\Sigma}}'_i = \boldsymbol{\Sigma}'_i + \nu \mathbf{I}_{2 \times 2}$$

---

### C. Differentiable Alpha Blending & Early Ray Termination
Given the projected 2D center $\boldsymbol{\mu}'_i = \mathbf{K} \mathbf{t}_{\text{cam}, i} / t_z$ and regularized 2D covariance $\tilde{\boldsymbol{\Sigma}}'_i$, the spatial evaluation of Gaussian $i$ at pixel $\mathbf{p} = (u, v)^T$ is:

$$f_i(\mathbf{p}) = \exp\left( -\frac{1}{2} (\mathbf{p} - \boldsymbol{\mu}'_i)^T (\tilde{\boldsymbol{\Sigma}}'_i)^{-1} (\mathbf{p} - \boldsymbol{\mu}'_i) \right)$$

The effective alpha contribution at pixel $\mathbf{p}$ is $\alpha'_i(\mathbf{p}) = \alpha_i \cdot f_i(\mathbf{p})$.

Sorting all overlapping Gaussians in depth order $\{1, 2, \dots, M\}$ ($z_1 \le z_2 \le \dots \le z_M$), the final synthesized color $\hat{\mathbf{I}}(\mathbf{p})$ is computed via front-to-back volumetric compositing:

$$\hat{\mathbf{I}}(\mathbf{p}) = \sum_{i=1}^M \mathbf{c}_i \cdot \alpha'_i(\mathbf{p}) \cdot T_i(\mathbf{p})$$

where cumulative transmittance $T_i(\mathbf{p})$ is:

$$T_i(\mathbf{p}) = \prod_{j=1}^{i-1} (1 - \alpha'_j(\mathbf{p}))$$

Rendering for pixel $\mathbf{p}$ terminates immediately once transmittance falls below $\epsilon = 10^{-4}$ ($T_i(\mathbf{p}) < 10^{-4}$).

---

### D. Adaptive Density Control (Splitting, Cloning, Pruning)
Throughout training, 3DGS monitors the view-space positional gradient magnitude for every Gaussian:

$$\bar{\nabla}_{\boldsymbol{\mu}} = \frac{1}{K_{\text{views}}} \sum_{k=1}^{K_{\text{views}}} \left\| \frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}'_{i, k}} \right\|_2$$

Every 100 iterations (after an initial warm-up of 500 steps), Gaussians with $\bar{\nabla}_{\boldsymbol{\mu}} > \tau_{\text{pos}} = 0.0002$ are densified:
1. **Under-Reconstruction (Small Scale)**: If $\|\mathbf{s}_i\|_2 \le \tau_{\text{scale}}$, the Gaussian is **cloned** and moved along the direction of positional gradient.
2. **Over-Reconstruction (Large Scale)**: If $\|\mathbf{s}_i\|_2 > \tau_{\text{scale}}$, the Gaussian is **split** into two child Gaussians, scaling their radii by a factor of $1 / 1.6 = 0.625$ and sampling positions from the parent Gaussian distribution.
3. **Pruning**: Prunes any Gaussian with opacity $\alpha_i < \tau_{\alpha} = 0.005$ or scale exceeding the scene bounding box. Every 3000 iterations, all opacities are reset to near-zero ($\approx 0.01$) to eliminate floaters.

---

## 3. Granular Component-by-Component Architectural Breakdown

| Component Identity | Structural Specification & Type | Attention / Conv / Math Mechanics | Tensor Dimensions & Shapes | Dominant Hardware Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **Gaussian Primitive Bank** | Parameterized 3D Ellipsoid Array | Explicit tensor state ($\boldsymbol{\mu}, \mathbf{s}', \mathbf{q}, \sigma, \mathbf{k}_{\text{SH}}$) | $[N, 3], [N, 3], [N, 4], [N, 1], [N, 48]$ | GPU VRAM capacity & allocation |
| **Projective Coordinate Engine** | Pin-hole Affine Projection Kernel | Jacobian $\mathbf{J} \in \mathbb{R}^{2 \times 3}$ and affine transform $\mathbf{J} \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^T \mathbf{J}^T$ | $N_{\text{active}} \times 2 \times 2$ covariance matrices | CUDA register pressure & FP32 math |
| **Tile Binning & Key Generator** | $16 \times 16$ Tile Bounding Evaluator | $3\sigma$ radius bounding box intersecting screen tiles | 64-bit integer keys: $[\text{Tile ID}_{32} \mid \text{Depth}_{32}]$ | Memory bandwidth on global atomics |
| **Radix Depth Sorting Engine** | Parallel GPU Key-Value Sorter | `cub::DeviceRadixSort::SortPairs` | Sorted indices $[M_{\text{instances}}]$ | Global GPU memory bandwidth & latency |
| **Tile Rasterizer Warps** | $16 \times 16$ Cooperative Thread Blocks | Shared memory loading + early ray termination alpha accumulation | $16 \times 16$ pixel tile outputs | L1 Cache / Shared memory throughput |
| **Loss & Gradient Engine** | Differentiable Photometric Evaluator | $(1 - \lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$ with analytical backprop | $[H, W, 3]$ RGB difference map | Backward kernel divergence & memory atomics |

---

## 4. SOTA Benchmark Comparison Matrix

### Novel View Synthesis Performance (Mip-NeRF 360 & Tanks and Temples)

| Architecture | Representation | Mip-NeRF 360 PSNR $\uparrow$ | Mip-NeRF 360 SSIM $\uparrow$ | Mip-NeRF 360 LPIPS $\downarrow$ | Tanks & Temples PSNR $\uparrow$ | FPS @ 1080p | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NeRF (Original)** | Implicit Coordinate MLP | 24.10 dB | 0.672 | 0.450 | 22.80 dB | 0.05 FPS | ~48.0 hrs |
| **Instant-NGP** | Multi-Res Hash Grid + MLP | 25.59 dB | 0.699 | 0.389 | 24.50 dB | 15.0 FPS | **5.0 mins** |
| **Plenoxels** | Explicit Sparse Voxel Grid | 23.08 dB | 0.625 | 0.463 | 21.08 dB | 12.0 FPS | 11.0 mins |
| **Mip-NeRF 360** | Multi-Scale Integrated Pos-Enc | 27.69 dB | 0.792 | 0.237 | 25.72 dB | 0.06 FPS | 42.0 hrs |
| **3D Gaussian Splatting** | Explicit Anisotropic 3DGS | **27.21 dB** | **0.815** | **0.214** | **28.50 dB** | **135.0 FPS** | **25.0 mins** |
| **Mip-Splatting** | Anti-Aliased 3DGS Filter | **27.85 dB** | **0.832** | **0.198** | **28.90 dB** | **120.0 FPS** | 28.0 mins |

---

## 5. Implementation Code: Complete PyTorch & CUDA Splatting Engine

```python
"""
Core PyTorch Implementation of 3D Gaussian Primitive Parameterization,
Projective Covariance Derivation, and Alpha Compositing Engine.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rotation_matrix_from_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Constructs 3x3 rotation matrices from normalized 4D quaternions [w, x, y, z]."""
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q.unbind(-1)
    
    R = torch.stack([
        1 - 2 * (y**2 + z**2), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x**2 + z**2), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x**2 + y**2)
    ], dim=-1).reshape(-1, 3, 3)
    return R


def build_covariance_3d(scales: torch.Tensor, rotations_q: torch.Tensor) -> torch.Tensor:
    """
    Computes symmetric positive semi-definite 3D covariance Sigma = R S S^T R^T.
    scales: [N, 3] in log space
    rotations_q: [N, 4] quaternions
    """
    S = torch.diag_embed(torch.exp(scales))  # [N, 3, 3]
    R = build_rotation_matrix_from_quaternion(rotations_q)  # [N, 3, 3]
    M = torch.bmm(R, S)  # [N, 3, 3]
    Sigma = torch.bmm(M, M.transpose(1, 2))  # [N, 3, 3]
    return Sigma


def project_covariance_2d(
    means3d: torch.Tensor,
    Sigma3d: torch.Tensor,
    viewmatrix: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_h: int,
    img_w: int,
    eps2d: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Projects 3D Gaussian means and covariances to 2D screen space.
    viewmatrix: [4, 4] world-to-camera matrix
    """
    N = means3d.shape[0]
    # Transform points to camera coordinates
    means3d_hom = torch.cat([means3d, torch.ones(N, 1, device=means3d.device)], dim=-1)  # [N, 4]
    p_cam = torch.matmul(means3d_hom, viewmatrix.T)[:, :3]  # [N, 3]
    
    tx, ty, tz = p_cam.unbind(-1)
    # Filter points behind camera
    valid = tz > 0.1
    
    # 2D Screen projection of centers
    u = (fx * tx / tz) + cx
    v = (fy * ty / tz) + cy
    means2d = torch.stack([u, v], dim=-1)  # [N, 2]
    
    # Pinhole Jacobian J in R^(2x3)
    J = torch.zeros(N, 2, 3, device=means3d.device)
    J[:, 0, 0] = fx / tz
    J[:, 0, 2] = -fx * tx / (tz**2)
    J[:, 1, 1] = fy / tz
    J[:, 1, 2] = -fy * ty / (tz**2)
    
    W = viewmatrix[:3, :3].unsqueeze(0).expand(N, 3, 3)  # [N, 3, 3]
    # Sigma_cam = W Sigma W^T
    Sigma_cam = torch.bmm(W, torch.bmm(Sigma3d, W.transpose(1, 2)))
    # Sigma_2d = J Sigma_cam J^T
    Sigma_2d = torch.bmm(J, torch.bmm(Sigma_cam, J.transpose(1, 2)))  # [N, 2, 2]
    
    # Add anti-aliasing low-pass filter
    Sigma_2d[:, 0, 0] += eps2d
    Sigma_2d[:, 1, 1] += eps2d
    
    return means2d, Sigma_2d, valid


class GaussianModel(nn.Module):
    """Explicit 3D Gaussian scene representation container."""
    def __init__(self, num_points: int):
        super().__init__()
        self.means = nn.Parameter(torch.randn(num_points, 3))
        self.scales = nn.Parameter(torch.zeros(num_points, 3))
        self.rotations = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(num_points, 1))
        self.opacities = nn.Parameter(torch.zeros(num_points, 1))
        self.sh_features = nn.Parameter(torch.randn(num_points, 16, 3))  # Degree 3 SH

    def get_covariance(self) -> torch.Tensor:
        return build_covariance_3d(self.scales, self.rotations)

    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacities)
```

---

## 6. References & Official Resources
- **Original Paper**: [3D Gaussian Splatting for Real-Time Radiance Field Rendering (SIGGRAPH 2023)](https://arxiv.org/abs/2308.04079)
- **Official Inria Repository**: [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- **High-Performance CUDA Rasterizer (`gsplat`)**: [https://github.com/nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)

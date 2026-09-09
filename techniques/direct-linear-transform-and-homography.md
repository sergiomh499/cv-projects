---
title: "Direct Linear Transform (DLT) & Planar Homography: 2D Projective Mapping"
type: "Technique"
domain: "Two-View Geometry, 6-DoF Calibration & Homography Estimation"
tags:
  - technique
  - dlt
  - direct-linear-transform
  - homography
  - planar-homography
  - projective-geometry
  - two-view-geometry
status: evergreen
updated: 2026-09-09
aliases:
  - "Direct Linear Transform"
  - "DLT"
  - "Planar Homography"
  - "Projective Transformation"
  - "Hartley Normalization"
---

# 📐 Direct Linear Transform (DLT) & Planar Homography: 2D Projective Mapping

## 1. High-Level Concept & The Projective Plane Mapping

In geometric computer vision, multi-camera calibration, and autonomous driving BEV perception:
When a camera observes a **planar physical surface** in the 3D world (e.g., flat road surface, checkerboard calibration pattern, building facade, or floor plane) OR when a camera undergoes **pure rotation around its optical center without translation** ($\mathbf{t} = \mathbf{0}$):

The transformation between corresponding 2D homogeneous pixel coordinates in View 1 ($\mathbf{x} = [u, v, 1]^T$) and View 2 ($\mathbf{x}' = [u', v', 1]^T$) is governed by an invertible $3 \times 3$ **Planar Homography Matrix** $\mathbf{H} \in \mathbb{R}^{3 \times 3}$:

$$
\mathbf{x}' \sim \mathbf{H} \mathbf{x} \iff \begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

where actual measured 2D image coordinates evaluate to:
$$u' = \frac{x'}{w'} = \frac{h_{11}u + h_{12}v + h_{13}}{h_{31}u + h_{32}v + h_{33}}, \quad v' = \frac{y'}{w'} = \frac{h_{21}u + h_{22}v + h_{23}}{h_{31}u + h_{32}v + h_{33}}$$

Because coordinates are homogeneous, multiplying $\mathbf{H}$ by any non-zero scalar $\lambda \ne 0$ leaves projected coordinates unchanged. Thus, a homography possesses **8 degrees of freedom** (DoF) and requires a minimum of **$N \ge 4$ point correspondences** (no three collinear).

---

### The Cross-Product Constraint Formulation
Because $\mathbf{x}'$ and $\mathbf{H}\mathbf{x}$ have identical directions in projective space $\mathbb{P}^2$, their **vector cross-product must be exactly zero**:

$$
\mathbf{x}' \times (\mathbf{H} \mathbf{x}) = \mathbf{0}
$$

Expanding the cross-product eliminates the unknown projective divisor $w'$, yielding **two linearly independent equations per correspondence pair**:

$$
\begin{bmatrix} \mathbf{0}^T & -w_i' \mathbf{x}_i^T & y_i' \mathbf{x}_i^T \\ w_i' \mathbf{x}_i^T & \mathbf{0}^T & -x_i' \mathbf{x}_i^T \end{bmatrix} \begin{bmatrix} \mathbf{h}_1 \\ \mathbf{h}_2 \\ \mathbf{h}_3 \end{bmatrix} = \mathbf{0}
$$

where $\mathbf{h} = [h_{11}, h_{12}, \dots, h_{33}]^T \in \mathbb{R}^9$.

---

### The Catastrophic Failure of Unnormalized DLT
In raw camera coordinates, typical pixel positions take values like $u \approx 1920, v \approx 1080$.
Products like $u \cdot u'$ evaluate to $\sim 4 \times 10^6$, while constant terms equal $1$.
- Stacking raw pixel numbers causes matrix $\mathbf{A}$ to have an extreme condition number ($\kappa(\mathbf{A}) \approx 10^{10}$).
- Standard SVD solvers suffer catastrophic numerical cancellation and yield wild geometric errors!

#### Hartley Isotropic Normalization (Hartley & Zisserman, 2004)
Before forming $\mathbf{A}$, transform point sets in both views independently:
1. Translate points so their centroid lies at $(0, 0)$.
2. Uniformly scale coordinates so their average Euclidean distance to the origin is $\sqrt{2}$:
   $$\mathbf{T}_{\text{norm}} = \begin{bmatrix} s & 0 & -s \bar{u} \\ 0 & s & -s \bar{v} \\ 0 & 0 & 1 \end{bmatrix}, \quad s = \frac{\sqrt{2}}{\frac{1}{N} \sum_i \sqrt{(u_i - \bar{u})^2 + (v_i - \bar{v})^2}}$$
3. Solve normalized homography $\tilde{\mathbf{H}}$, then un-normalize:
   $$\mathbf{H} = \mathbf{T}'^{-1} \tilde{\mathbf{H}} \mathbf{T}$$

```
Direct Linear Transform (DLT) Execution Pipeline:

2D Correspondence Points {x_i} <---> {x_i'}
             |
             v
[ 1. Hartley Isotropic Normalization (Centroid = 0, Dist = sqrt(2)) ]
             |
             v Normalized Points x_tilde, x_tilde'
[ 2. Assemble 2N x 9 Linear Coefficient Matrix A ]
             |
             v
[ 3. Singular Value Decomposition (SVD): A = U * Sigma * V^T ]
             |
             v Singular vector corresponding to smallest singular value (Last col of V)
[ 4. Reshape Vector h* into 3x3 Matrix H_tilde ]
             |
             v
[ 5. Denormalize: H = inv(T') * H_tilde * T ]
```

---

## 2. Mathematical Formulation & SVD Nullspace Proof

Given $N \ge 4$ point correspondences, stack the $2 \times 9$ blocks into a tall $2N \times 9$ linear system:

$$
\mathbf{A} \mathbf{h} = \mathbf{0}
$$

Due to real-world camera sensor noise and lens distortion, exact equality $\mathbf{A}\mathbf{h} = \mathbf{0}$ cannot be satisfied. We seek the optimal vector $\mathbf{h}^*$ that minimizes algebraic residual norm under unit norm constraint:

$$
\mathbf{h}^* = \arg\min_{\|\mathbf{h}\|_2 = 1} \|\mathbf{A} \mathbf{h}\|_2^2 = \arg\min_{\|\mathbf{h}\|_2 = 1} \mathbf{h}^T (\mathbf{A}^T \mathbf{A}) \mathbf{h}
$$

### Proof via Rayleigh Quotient
Let $\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ be the SVD of $\mathbf{A}$, where $\mathbf{\Sigma} = \text{diag}(\sigma_1, \dots, \sigma_9)$ with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_9 \ge 0$.
The objective expands to:

$$
\mathbf{h}^T \mathbf{A}^T \mathbf{A} \mathbf{h} = \mathbf{h}^T \mathbf{V} \mathbf{\Sigma}^2 \mathbf{V}^T \mathbf{h} = \mathbf{y}^T \mathbf{\Sigma}^2 \mathbf{y} = \sum_{k=1}^9 \sigma_k^2 y_k^2
$$

where $\mathbf{y} = \mathbf{V}^T \mathbf{h}$ with $\|\mathbf{y}\|_2 = \|\mathbf{h}\|_2 = 1$.
Because $\sigma_9 \le \sigma_k$ for all $k$, the sum is minimized when $y_9 = 1$ and $y_1 = \dots = y_8 = 0$.
Therefore:

$$
\mathbf{h}^* = \mathbf{V} \mathbf{y}^* = \mathbf{V} [0, \dots, 0, 1]^T = \mathbf{v}_9 \quad (\text{the 9th column of } \mathbf{V})
$$

---

## 3. Python Reference Implementation

```python
import numpy as np

def compute_normalization_matrix(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Hartley isotropic normalization for numerical conditioning.
    points: [N, 2]
    Returns: (normalized_points [N, 2], T [3, 3])
    """
    mean = np.mean(points, axis=0)
    std_dist = np.mean(np.sqrt(np.sum((points - mean) ** 2, axis=1)))
    scale = np.sqrt(2.0) / (std_dist + 1e-8)

    T = np.array([
        [scale, 0.0, -scale * mean[0]],
        [0.0, scale, -scale * mean[1]],
        [0.0, 0.0, 1.0]
    ])
    
    pts_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    pts_norm = (T @ pts_homo.T).T
    return pts_norm[:, :2], T


def find_homography_dlt(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Normalized Direct Linear Transform (DLT) solving planar homography H.
    src_points: [N, 2]
    dst_points: [N, 2] (N >= 4)
    Returns: H [3, 3] such that dst ~ H * src
    """
    n = src_points.shape[0]
    assert n >= 4, "DLT requires at least 4 point correspondences"

    # 1. Hartley Normalization
    src_norm, T_src = compute_normalization_matrix(src_points)
    dst_norm, T_dst = compute_normalization_matrix(dst_points)

    # 2. Build 2N x 9 Linear Matrix A
    A = np.zeros((2 * n, 9))
    for i in range(n):
        u, v = src_norm[i]
        u_prime, v_prime = dst_norm[i]
        A[2 * i] = [-u, -v, -1.0, 0.0, 0.0, 0.0, u * u_prime, v * u_prime, u_prime]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -u, -v, -1.0, u * v_prime, v * v_prime, v_prime]

    # 3. SVD Nullspace Solve
    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    # 4. Denormalize: H = inv(T_dst) * H_norm * T_src
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    return H / H[2, 2] # Normalize scale
```

---

## 4. Models in the Vault Utilizing Homographies

- **[[techniques/epipolar-geometry-essential-matrix|Epipolar Geometry & Essential Matrix]]**: Planar homography decomposition for relative camera motion initialization.
- **[[techniques/perspective-n-point-epnp-pose-estimation|Perspective-n-Point (EPnP)]]**: Planar camera pose estimation and intrinsic camera calibration.
- **[[architectures/3d-pointclouds-and-lidar/bevfusion|BEVFusion]]**: Ground-plane homography lifting in camera-to-bird's-eye-view feature projection.
- **[[topics/6dof-pose-estimation/README|6-DoF Pose Estimation Playbook]]**: Surface planar correspondence tracking for augmented reality registration.

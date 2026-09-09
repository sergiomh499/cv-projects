---
title: "Perspective-n-Point (PnP) & EPnP: Linear-Complexity O(N) 6-DoF Pose Estimation"
type: "Technique"
domain: "6-DoF Object Pose Estimation, Robotics & Spatial Localization"
tags:
  - technique
  - pnp
  - epnp
  - 6dof-pose
  - pose-estimation
  - robotics
  - camera-calibration
status: evergreen
updated: 2026-09-09
aliases:
  - "Perspective-n-Point"
  - "PnP"
  - "EPnP"
  - "Efficient PnP"
  - "2D-3D Pose Estimation"
---

# 📐 Perspective-n-Point (PnP) & EPnP: Linear-Complexity $\mathcal{O}(N)$ Pose Estimation

## 1. High-Level Concept & The 2D-to-3D Localization Problem

In robotic grasping, augmented reality, and 6-DoF object tracking (e.g., [[architectures/pose-and-robotics-manipulation/foundationpose|FoundationPose]], [[architectures/pose-and-robotics-manipulation/megapose|MegaPose]], [[architectures/pose-and-robotics-manipulation/cosypose|CosyPose]]), a calibrated camera with known intrinsic matrix $\mathbf{K}$ observes an object with known 3D geometry.

Given $N \ge 4$ paired correspondences between:
- 3D reference coordinates in the object/world frame: $\mathbf{p}_i^w = [x_i^w, y_i^w, z_i^w]^T \in \mathbb{R}^3$.
- 2D projected pixel locations on the camera sensor: $\mathbf{u}_i = [u_i, v_i]^T \in \mathbb{R}^2$.

The **Perspective-n-Point (PnP)** problem seeks the optimal rigid transformation $(\mathbf{R}, \mathbf{t}) \in \mathrm{SE}(3)$ (where $\mathbf{R} \in \mathrm{SO}(3)$ and $\mathbf{t} \in \mathbb{R}^3$) that relates the world frame to the camera frame:

$$
s_i \begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix} = \mathbf{K} \left( \mathbf{R} \mathbf{p}_i^w + \mathbf{t} \right)
$$

where $s_i$ is the unknown projective depth scale of point $i$.

### Limitations of Classical PnP Formulations
1. **Minimal P3P ($N = 3$)**: Yields up to 4 ambiguous mathematical solutions, is hyper-sensitive to pixel noise, and cannot leverage redundant correspondences when $N > 3$.
2. **Direct Linear Transform (DLT)**: Solves the linear system without enforcing the non-linear orthogonality constraint $\mathbf{R}^T \mathbf{R} = \mathbf{I}$, producing unphysical distorted matrices.
3. **Iterative Non-Linear Optimization (Levenberg-Marquardt)**: While accurate, iterative solvers require near-perfect initializations; otherwise, they fall into local minima or take $>20\text{ ms}$ to converge.

### The EPnP (Efficient PnP) Solution
**EPnP** (Lepetit, Moreno-Noguer, Fua, IJCV 2009) solves PnP in **strictly linear computational time $\mathcal{O}(N)$**:
1. **Virtual Control Points**: Expresses each 3D point $\mathbf{p}_i^w$ as a barycentric linear combination of **4 virtual non-coplanar control points** $\mathbf{c}_j^w \in \mathbb{R}^3$.
2. **Euclidean Invariance**: Barycentric coordinates $\alpha_{ij}$ are **strictly invariant under Euclidean rigid transformations**! In the camera coordinate frame, the identical weights reconstruct the camera-frame points: $\mathbf{p}_i^c = \sum \alpha_{ij} \mathbf{c}_j^c$.
3. **Linear System Assembly**: Sets up a $2N \times 12$ matrix $\mathbf{M} \mathbf{x} = \mathbf{0}$ where the unknown vector $\mathbf{x} \in \mathbb{R}^{12}$ contains the camera coordinates of the 4 control points.
4. **Rigidity Constraint Enforcement**: Solves for $\mathbf{x}$ using SVD and enforces constant pairwise distance between control points ($\|\mathbf{c}_j^c - \mathbf{c}_k^c\|^2 = \|\mathbf{c}_j^w - \mathbf{c}_k^w\|^2$).
5. **Closed-Form Pose Recovery**: Recovers $(\mathbf{R}^*, \mathbf{t}^*)$ via Orthogonal Procrustes alignment in **$<1\text{ ms}$**!

```
EPnP Algorithmic Workflow:

World 3D Points p_i^w --------> Express via 4 Control Points c_j^w (Barycentric coords alpha_ij)
                                                    |
                                                    v
Image 2D Projections u_i -----> Assemble Linear System M * x = 0 (x = camera control points c_j^c)
                                                    |
                                                    v
                                  [ SVD Nullspace of M (12 x 12) ]
                                                    |
                                                    v
                             [ Enforce Pairwise Distances: ||c_j^c - c_k^c||^2 = ||c_j^w - c_k^w||^2 ]
                                                    |
                                                    v
                             [ Closed-Form Procrustes Alignment: c_j^c ~ R * c_j^w + t ]
                                                    |
                                                    v
                                  Exact (R*, t*) in <1 millisecond!
```

---

## 2. Mathematical Formulation

### 2.1 Barycentric Control Point Decomposition
Select 4 non-coplanar virtual control points in the world frame $\{\mathbf{c}_1^w, \mathbf{c}_2^w, \mathbf{c}_3^w, \mathbf{c}_4^w\}$.
Typically, $\mathbf{c}_1^w$ is the centroid of all 3D points, and $\mathbf{c}_2^w, \mathbf{c}_3^w, \mathbf{c}_4^w$ are aligned along the principal axes of the point cloud via PCA.

Each 3D point $\mathbf{p}_i^w$ is expressed as:

$$
\mathbf{p}_i^w = \sum_{j=1}^4 \alpha_{ij} \mathbf{c}_j^w \quad \text{with} \quad \sum_{j=1}^4 \alpha_{ij} = 1
$$

The barycentric coordinates $\boldsymbol{\alpha}_i = [\alpha_{i1}, \alpha_{i2}, \alpha_{i3}, \alpha_{i4}]^T$ are uniquely calculated by inverting the $4 \times 4$ control matrix:

$$
\begin{bmatrix} \alpha_{i1} \\ \alpha_{i2} \\ \alpha_{i3} \\ \alpha_{i4} \end{bmatrix} = \begin{bmatrix} \mathbf{c}_1^w & \mathbf{c}_2^w & \mathbf{c}_3^w & \mathbf{c}_4^w \\ 1 & 1 & 1 & 1 \end{bmatrix}^{-1} \begin{bmatrix} \mathbf{p}_i^w \\ 1 \end{bmatrix}
$$

Because affine barycentric coordinates are preserved under any linear transformation and translation:

$$
\mathbf{p}_i^c = \sum_{j=1}^4 \alpha_{ij} \mathbf{c}_j^c = \sum_{j=1}^4 \alpha_{ij} \begin{bmatrix} x_j^c \\ y_j^c \\ z_j^c \end{bmatrix}
$$

---

### 2.2 Constructing the Linear System $\mathbf{M} \mathbf{x} = \mathbf{0}$
Under the pinhole camera projection:

$$
u_i = f_x \frac{x_i^c}{z_i^c} + c_x, \quad v_i = f_y \frac{y_i^c}{z_i^c} + c_y
$$

Substituting $\mathbf{p}_i^c = \sum \alpha_{ij} \mathbf{c}_j^c$ and eliminating $z_i^c$ yields two linear equations per point:

$$
\sum_{j=1}^4 \left( \alpha_{ij} f_x x_j^c + \alpha_{ij} (c_x - u_i) z_j^c \right) = 0
$$

$$
\sum_{j=1}^4 \left( \alpha_{ij} f_y y_j^c + \alpha_{ij} (c_y - v_i) z_j^c \right) = 0
$$

Let the 12-dimensional unknown vector be $\mathbf{x} = [(\mathbf{c}_1^c)^T, (\mathbf{c}_2^c)^T, (\mathbf{c}_3^c)^T, (\mathbf{c}_4^c)^T]^T \in \mathbb{R}^{12}$.
Stacking equations for all $N$ correspondences forms the $2N \times 12$ matrix system:

$$
\mathbf{M} \mathbf{x} = \mathbf{0}
$$

---

### 2.3 Solving the Nullspace via SVD
The solution $\mathbf{x}$ lies in the right nullspace of $\mathbf{M}^T \mathbf{M} \in \mathbb{R}^{12 \times 12}$.
We compute the eigenvectors $\mathbf{v}_i$ of $\mathbf{M}^T \mathbf{M}$ corresponding to the smallest eigenvalues:

$$
\mathbf{x} = \sum_{k=1}^K \beta_k \mathbf{v}_{12 - K + k}
$$

Depending on camera focal length and noise, the nullspace dimension $K \in \{1, 2, 3, 4\}$ (typically $K = 1$ for well-conditioned general geometry).

---

### 2.4 Distance Constraint Enforcement
The control point positions must preserve their physical 3D pairwise distances between world and camera frames:

$$
\left\| \mathbf{c}_j^c - \mathbf{c}_k^c \right\|^2 = \left\| \mathbf{c}_j^w - \mathbf{c}_k^w \right\|^2, \quad \forall j, k \in \{1, 2, 3, 4\}
$$

Substituting $\mathbf{x} = \beta \mathbf{v}_{12}$ for $K=1$:

$$
\beta^2 \left\| \mathbf{v}_{12}^{[j]} - \mathbf{v}_{12}^{[k]} \right\|^2 = \left\| \mathbf{c}_j^w - \mathbf{c}_k^w \right\|^2
$$

This directly determines the scalar scale $\beta = \sqrt{\frac{\|\mathbf{c}_j^w - \mathbf{c}_k^w\|^2}{\|\mathbf{v}_{12}^{[j]} - \mathbf{v}_{12}^{[k]}\|^2}}$, fixing the sign of $\beta$ such that control points have positive depth ($z_j^c > 0$).

---

### 2.5 Closed-Form Pose Extraction via Orthogonal Procrustes
With the 3D control point coordinates known in both the world frame $\{\mathbf{c}_j^w\}$ and camera frame $\{\mathbf{c}_j^c\}$:
The final optimal rotation $\mathbf{R}^* \in \mathrm{SO}(3)$ and translation $\mathbf{t}^* \in \mathbb{R}^3$ are computed in closed form using **[[techniques/orthogonal-procrustes-and-umeyama-sim3|Orthogonal Procrustes / Umeyama Alignment]]**:

$$
\mathbf{R}^*, \; \mathbf{t}^* = \arg\min_{\mathbf{R}, \mathbf{t}} \sum_{j=1}^4 \left\| \mathbf{c}_j^c - (\mathbf{R} \mathbf{c}_j^w + \mathbf{t}) \right\|^2
$$

---

## 3. Python Reference Implementation

```python
import numpy as np

def epnp_solve(pts_3d: np.ndarray, pts_2d: np.ndarray, K: np.ndarray):
    """
    EPnP: Efficient Perspective-n-Point pose estimation in O(N) linear time.
    pts_3d: [N, 3] world landmark points
    pts_2d: [N, 2] image pixel coordinates
    K: [3, 3] camera intrinsic matrix
    Returns: (R [3, 3], t [3])
    """
    n = pts_3d.shape[0]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # 1. Define 4 virtual control points in world coordinates
    c1 = np.mean(pts_3d, axis=0)
    # Principal axes via SVD of centered points
    _, _, Vt = np.linalg.svd(pts_3d - c1)
    c2 = c1 + Vt[0] * 1.0
    c3 = c1 + Vt[1] * 1.0
    c4 = c1 + Vt[2] * 1.0
    Cw = np.vstack([c1, c2, c3, c4]) # [4, 3]
    
    # 2. Compute barycentric coordinates alpha: [N, 4]
    C_homo = np.vstack([Cw.T, np.ones(4)]) # [4, 4]
    inv_C = np.linalg.inv(C_homo)
    P_homo = np.hstack([pts_3d, np.ones((n, 1))]).T # [4, N]
    alphas = (inv_C @ P_homo).T # [N, 4]
    
    # 3. Assemble linear system M * x = 0: [2N, 12]
    M = np.zeros((2 * n, 12))
    for i in range(n):
        u, v = pts_2d[i]
        a = alphas[i]
        # Row 2i: x-projection
        for j in range(4):
            M[2 * i, j * 3 + 0] = a[j] * fx
            M[2 * i, j * 3 + 1] = 0.0
            M[2 * i, j * 3 + 2] = a[j] * (cx - u)
        # Row 2i+1: y-projection
        for j in range(4):
            M[2 * i + 1, j * 3 + 0] = 0.0
            M[2 * i + 1, j * 3 + 1] = a[j] * fy
            M[2 * i + 1, j * 3 + 2] = a[j] * (cy - v)
            
    # 4. Nullspace of M^T M: [12, 12]
    _, _, Vt_M = np.linalg.svd(M)
    v12 = Vt_M[-1] # Smallest singular vector [12]
    
    # 5. Enforce distance constraint between control points
    # Control points in camera frame: Cc = beta * v12
    v12_pts = v12.reshape(4, 3)
    dist_w = np.linalg.norm(Cw[0] - Cw[1])
    dist_v = np.linalg.norm(v12_pts[0] - v12_pts[1])
    beta = dist_w / (dist_v + 1e-7)
    
    Cc = beta * v12_pts
    # Ensure positive depth
    if np.mean(Cc[:, 2]) < 0:
        Cc *= -1.0
        
    # 6. Orthogonal Procrustes alignment: Cc ~ R * Cw + t
    mu_w = np.mean(Cw, axis=0)
    mu_c = np.mean(Cc, axis=0)
    
    H = (Cc - mu_c).T @ (Cw - mu_w)
    U, _, Vt_H = np.linalg.svd(H)
    R = U @ Vt_H
    if np.linalg.det(R) < 0:
        U[:, 2] *= -1.0
        R = U @ Vt_H
        
    t = mu_c - R @ mu_w
    return R, t
```

---

## 4. Models in the Vault Utilizing PnP & EPnP

- **[[architectures/pose-and-robotics-manipulation/foundationpose|FoundationPose]]**: Fast initial 6-DoF pose hypothesis generation from predicted 2D keypoint heatmaps.
- **[[architectures/pose-and-robotics-manipulation/megapose|MegaPose]] & [[architectures/pose-and-robotics-manipulation/cosypose|CosyPose]]**: Coarse 6-DoF pose estimation via EPnP followed by neural mesh render-and-compare refinement.
- **[[topics/visual-guidance-and-robotics/00-visual-guidance-and-robotics-moc|Visual Guidance & Robotics MOC]]**: Connected as the foundational 2D-to-3D visual registration solver.

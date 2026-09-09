---
title: "Epipolar Geometry & Essential Matrix: The Normalized 8-Point Algorithm & Cheirality"
type: "Technique"
domain: "Two-View Geometry, Visual Odometry & Multi-View Perception"
tags:
  - technique
  - epipolar-geometry
  - essential-matrix
  - fundamental-matrix
  - eight-point-algorithm
  - visual-odometry
  - visual-slam
status: evergreen
updated: 2026-09-09
aliases:
  - "Epipolar Geometry"
  - "Essential Matrix"
  - "8-Point Algorithm"
  - "Two-View Motion Estimation"
  - "Cheirality Recovery"
---

# 📐 Epipolar Geometry & Essential Matrix: The Normalized 8-Point Algorithm & Cheirality

## 1. High-Level Concept & The Two-View Geometric Constraint

In visual odometry, visual SLAM (e.g., [[architectures/spatial-radiance-and-slam/droid-slam|DROID-SLAM]]), and multi-view 3D tracking (e.g., [[architectures/visual-tracking-and-flow/tapir|TAPIR]], [[architectures/pose-and-robotics-manipulation/cosypose|CosyPose]]), a primary problem is estimating the relative 3D motion—a rotation $\mathbf{R} \in \mathrm{SO}(3)$ and translation direction $\mathbf{t} \in \mathbb{R}^3$—between two calibrated camera keyframes observing a set of matching visual features.

### The Epipolar Constraint
Let $\mathbf{C}_1$ and $\mathbf{C}_2$ be the optical centers of Camera 1 and Camera 2 separated by baseline $\mathbf{t}$.
When both cameras observe a 3D landmark $\mathbf{X} \in \mathbb{R}^3$:
- The two camera centers $\mathbf{C}_1, \mathbf{C}_2$ and the physical point $\mathbf{X}$ define a single 2D plane called the **Epipolar Plane**.
- The ray from $\mathbf{C}_1$ through image point $\mathbf{x}_1$ and the ray from $\mathbf{C}_2$ through image point $\mathbf{x}_2$ must intersect at $\mathbf{X}$ and therefore lie on the epipolar plane.
- The intersection of the baseline with the image planes defines the **Epipoles** $\mathbf{e}_1, \mathbf{e}_2$.
- The projection of the entire ray $\mathbf{C}_1 \mathbf{X}$ onto Camera 2 forms the **Epipolar Line** $\mathbf{l}_2$. Given a point $\mathbf{x}_1$ in image 1, its corresponding match $\mathbf{x}_2$ in image 2 is **geometrically constrained to lie on line $\mathbf{l}_2$**, reducing a 2D spatial search to a 1D line search.

```
Epipolar Geometry & The Coplanarity Constraint:

               3D Landmark X
                 /        \
                /          \
               /  Epipolar  \
              /     Plane    \
             v                v
      Image 1 (x_1)      Image 2 (x_2 lies on epipolar line l_2!)
            |                  |
            v                  v
       Camera C_1 ========> Camera C_2 (Relative motion: R, t)
                 Baseline t
```

---

## 2. Mathematical Formulation

### 2.1 The Essential Matrix $\mathbf{E}$ vs. Fundamental Matrix $\mathbf{F}$
Let normalized camera coordinates be $\mathbf{x} = \mathbf{K}^{-1} \mathbf{p}$, where $\mathbf{p} = [u, v, 1]^T$ is the homogeneous pixel coordinate and $\mathbf{K}$ is the camera intrinsic calibration matrix.

A 3D point $\mathbf{X}_2$ in camera 2 coordinates is related to camera 1 by the rigid transformation:

$$
\mathbf{X}_2 = \mathbf{R} \mathbf{X}_1 + \mathbf{t}
$$

Taking the vector cross product with the translation vector $\mathbf{t}$:

$$
\mathbf{t} \times \mathbf{X}_2 = \mathbf{t} \times \left( \mathbf{R} \mathbf{X}_1 \right)
$$

Taking the dot product with $\mathbf{X}_2$:

$$
\mathbf{X}_2^T \left( \mathbf{t} \times \mathbf{X}_2 \right) = \mathbf{X}_2^T \left( \mathbf{t} \times (\mathbf{R} \mathbf{X}_1) \right) = 0
$$

Writing the cross product as the skew-symmetric matrix $[\mathbf{t}]_\times$:

$$
\mathbf{X}_2^T [\mathbf{t}]_\times \mathbf{R} \mathbf{X}_1 = 0 \implies \mathbf{x}_2^T \mathbf{E} \mathbf{x}_1 = 0
$$

The **Essential Matrix** $\mathbf{E} \in \mathbb{R}^{3 \times 3}$ is defined as:

$$
\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}
$$

For uncalibrated pixel coordinates, the **Fundamental Matrix** $\mathbf{F}$ satisfies:

$$
\mathbf{p}_2^T \mathbf{F} \mathbf{p}_1 = 0, \quad \mathbf{F} = \mathbf{K}_2^{-T} \mathbf{E} \mathbf{K}_1^{-1}
$$

---

### 2.2 The Normalized 8-Point Algorithm (Hartley's Isotropic Scaling)
Given $N \ge 8$ feature correspondences $\mathbf{x}_{1, i} \leftrightarrow \mathbf{x}_{2, i}$:
Expanding $\mathbf{x}_2^T \mathbf{E} \mathbf{x}_1 = 0$ yields a linear equation per match:

$$
x_{2} x_{1} e_{11} + x_{2} y_{1} e_{12} + x_{2} e_{13} + y_{2} x_{1} e_{21} + y_{2} y_{1} e_{22} + y_{2} e_{23} + x_{1} e_{31} + y_{1} e_{32} + e_{33} = 0
$$

#### The Failure of Naive Estimation
In raw pixel coordinates $(u, v) \approx 1000$:
- The terms $u_1 u_2 \approx 10^6$, whereas the constant term is $1.0$.
- The linear system $\mathbf{A} \mathbf{e} = \mathbf{0}$ has an enormous matrix condition number ($>10^8$), causing the solution to diverge on minor pixel noise.

#### Hartley's Normalization
Hartley (1997) transforms coordinates by an affine matrix $\mathbf{T}$ such that:
1. The centroid of all points is at the origin $(0, 0)$.
2. The average Euclidean distance from the origin is $\sqrt{2}$.

$$
\mathbf{T} = \begin{bmatrix} s & 0 & -s \cdot \bar{u} \\ 0 & s & -s \cdot \bar{v} \\ 0 & 0 & 1 \end{bmatrix}, \quad s = \frac{\sqrt{2}}{\frac{1}{N} \sum_{i=1}^N \sqrt{(u_i - \bar{u})^2 + (v_i - \bar{v})^2}}
$$

Normalized points $\hat{\mathbf{x}} = \mathbf{T} \mathbf{x}$ produce an exceptionally well-conditioned matrix ($\text{cond} \approx 10$).

---

### 2.3 SVD Rank-2 Constraint Projection
A mathematically valid Essential Matrix must have **two equal non-zero singular values and one zero singular value**:

$$
\mathbf{\Sigma} = \text{diag}(\sigma, \; \sigma, \; 0)
$$

The linear solve on $\mathbf{A} \mathbf{e} = \mathbf{0}$ yields a raw estimate $\mathbf{E}_{\text{raw}}$ that violates this constraint due to noise.
We compute SVD: $\mathbf{E}_{\text{raw}} = \mathbf{U} \mathbf{D} \mathbf{V}^T$, where $\mathbf{D} = \text{diag}(d_1, d_2, d_3)$.
We project $\mathbf{E}$ onto the essential matrix manifold by replacing $\mathbf{D}$ with $\text{diag}(1, 1, 0)$:

$$
\mathbf{E} = \mathbf{U} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix} \mathbf{V}^T
$$

---

### 2.4 Cheirality Recovery: Four-Fold Ambiguity Resolution
Factorizing $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$ produces **four possible mathematical solutions** $(\mathbf{R}, \mathbf{t})$:

Let $\mathbf{W} = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$. The four candidate pairs are:
1. $\mathbf{R}_1 = \mathbf{U} \mathbf{W} \mathbf{V}^T, \quad \mathbf{t}_1 = +\mathbf{u}_3$
2. $\mathbf{R}_1 = \mathbf{U} \mathbf{W} \mathbf{V}^T, \quad \mathbf{t}_2 = -\mathbf{u}_3$
3. $\mathbf{R}_2 = \mathbf{U} \mathbf{W}^T \mathbf{V}^T, \quad \mathbf{t}_1 = +\mathbf{u}_3$
4. $\mathbf{R}_2 = \mathbf{U} \mathbf{W}^T \mathbf{V}^T, \quad \mathbf{t}_2 = -\mathbf{u}_3$

(If $\det(\mathbf{R}) = -1$, negate the matrix to ensure valid $\mathrm{SO}(3)$ rotation).

#### The Cheirality Invariant
In physical reality, visual sensors cannot capture light from behind the camera.
We triangulate a test correspondence $\mathbf{x}_1 \leftrightarrow \mathbf{x}_2$ for each of the 4 solutions:
- **The true physical solution is the unique pair where the 3D triangulated point has positive optical depth ($Z > 0$) in both Camera 1 and Camera 2!**

---

## 3. Python Reference Implementation

```python
import numpy as np

def compute_essential_matrix(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Computes Essential Matrix from normalized camera coordinates via the 8-point algorithm.
    pts1, pts2: [N, 2] corresponding normalized coordinates
    """
    n = pts1.shape[0]
    # Build linear system A * e = 0
    A = np.zeros((n, 9))
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]
    
    A[:, 0] = x2 * x1
    A[:, 1] = x2 * y1
    A[:, 2] = x2
    A[:, 3] = y2 * x1
    A[:, 4] = y2 * y1
    A[:, 5] = y2
    A[:, 6] = x1
    A[:, 7] = y1
    A[:, 8] = 1.0
    
    # Solve via SVD
    _, _, Vt = np.linalg.svd(A)
    E_raw = Vt[-1].reshape(3, 3)
    
    # Force rank-2 constraint: singular values (1, 1, 0)
    U, _, Vt = np.linalg.svd(E_raw)
    E = U @ np.diag([1.0, 1.0, 0.0]) @ Vt
    return E

def decompose_essential_matrix(E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray):
    """
    Recovers relative pose (R, t) resolving the 4-fold ambiguity via the Cheirality check.
    """
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
        
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # 4 candidate solutions
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    
    candidates = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    
    # Test cheirality on first correspondence
    x1 = np.array([pts1[0, 0], pts1[0, 1], 1.0])
    x2 = np.array([pts2[0, 0], pts2[0, 1], 1.0])
    
    best_pair = candidates[0]
    max_positive = -1
    
    for R_cand, t_cand in candidates:
        # Linear triangulation for depth Z1
        # [R_cand * x1, -x2] * [Z1, Z2]^T = -t_cand
        M = np.column_stack([R_cand @ x1, -x2])
        depths, _, _, _ = np.linalg.lstsq(M, -t_cand, rcond=None)
        if depths[0] > 0 and depths[1] > 0:
            best_pair = (R_cand, t_cand)
            break
            
    return best_pair[0], best_pair[1]
```

---

## 4. Models in the Vault Utilizing Epipolar Geometry

- **[[architectures/spatial-radiance-and-slam/droid-slam|DROID-SLAM]]**: Employs epipolar geometry to initialize visual bundle adjustment keyframes.
- **[[architectures/visual-tracking-and-flow/tapir|TAPIR]]**: Constrains multi-point trajectory search along epipolar lines.
- **[[architectures/pose-and-robotics-manipulation/cosypose|CosyPose]]**: Multiview 6-DoF object pose hypothesis aggregation.
- **[[topics/slam-and-spatial-perception/00-slam-and-spatial-perception-moc|SLAM & Spatial Perception MOC]]**: Connected as the foundational two-view geometric formulation.

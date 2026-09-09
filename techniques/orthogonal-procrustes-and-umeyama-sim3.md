---
title: "Orthogonal Procrustes & Umeyama Alignment: Closed-Form Sim(3) / SE(3) Rigid Registration"
type: "Technique"
domain: "3D Point Clouds, 6-DoF Pose Estimation & Spatial SLAM"
tags:
  - technique
  - procrustes
  - umeyama
  - sim3
  - se3
  - svd
  - 6dof-pose
  - pointclouds
status: evergreen
updated: 2026-09-09
aliases:
  - "Orthogonal Procrustes"
  - "Umeyama Algorithm"
  - "Sim(3) Alignment"
  - "Absolute Orientation Problem"
---

# 📐 Orthogonal Procrustes & Umeyama Alignment: Closed-Form $\mathrm{Sim}(3)$ / $\mathrm{SE}(3)$ Registration

## 1. High-Level Concept & Point Set Registration

In 6-DoF object pose estimation (e.g., [[architectures/pose-and-robotics-manipulation/foundationpose|FoundationPose]], [[architectures/pose-and-robotics-manipulation/megapose|MegaPose]], [[architectures/pose-and-robotics-manipulation/cosypose|CosyPose]]), sensor extrinsic calibration (hand-eye and LiDAR-to-camera calibration), and visual SLAM trajectory evaluation (Absolute Trajectory Error - ATE), a core problem is registering two corresponding 3D point sets.

Given $N$ paired 3D point correspondences between a source coordinate frame $\mathbf{X} = \{\mathbf{x}_i\}_{i=1}^N$ and a target coordinate frame $\mathbf{Y} = \{\mathbf{y}_i\}_{i=1}^N$ (where $\mathbf{x}_i, \mathbf{y}_i \in \mathbb{R}^3$), we seek the optimal similarity transformation $(s, \mathbf{R}, \mathbf{t}) \in \mathrm{Sim}(3)$—consisting of a scalar metric scale $s > 0$, 3D rotation matrix $\mathbf{R} \in \mathrm{SO}(3)$, and 3D translation vector $\mathbf{t} \in \mathbb{R}^3$—that minimizes the mean squared Euclidean error:

$$
\mathcal{E}(s, \mathbf{R}, \mathbf{t}) = \frac{1}{N} \sum_{i=1}^N \left\| \mathbf{y}_i - (s \mathbf{R} \mathbf{x}_i + \mathbf{t}) \right\|^2
$$

### The Failure of Naive Gradient Descent & Iterative Solvers
Attempting to solve this problem via iterative gradient descent or non-linear least squares presents severe operational drawbacks:
1. **Local Minima**: Optimization can easily get trapped in local saddle points, particularly in symmetrical objects or when rotations exceed $90^\circ$.
2. **Manifold Violation**: Unconstrained optimization breaks the strict $\mathrm{SO}(3)$ orthogonality constraints ($\mathbf{R}^T \mathbf{R} = \mathbf{I}, \det(\mathbf{R}) = +1$).
3. **Reflection Degeneracy**: A common failure mode in unconstrained SVD is recovering an improper rotation matrix with $\det(\mathbf{R}) = -1$ (a mirror reflection rather than a valid physical rotation).

### The Umeyama Closed-Form Solution
The **Umeyama Algorithm** (Umeyama, IEEE TPAMI 1991; building on Schönemann's Orthogonal Procrustes):
1. **Decouples Translation via Centroids**: Subtracts the respective center of masses $\boldsymbol{\mu}_x, \boldsymbol{\mu}_y$, reducing the problem strictly to scale and rotation.
2. **Singular Value Decomposition (SVD)**: Decomposes the spatial cross-covariance matrix $\mathbf{\Sigma}_{yx} = \mathbf{U} \mathbf{D} \mathbf{V}^T$.
3. **Determinant Reflection Correction**: Checks $\det(\mathbf{U}\mathbf{V}^T)$ to resolve planar reflection ambiguity, guaranteeing that $\mathbf{R} \in \mathrm{SO}(3)$ always has $\det(\mathbf{R}) = +1$.
4. **Delivers the exact, globally optimal $(s^*, \mathbf{R}^*, \mathbf{t}^*)$ in a single closed-form pass in $<10\ \mu\text{s}$**!

```
Point Cloud Registration via Umeyama Sim(3):

Source Point Cloud X               Target Point Cloud Y
        |                                  |
        +---> Centroid Shift mu_x          +---> Centroid Shift mu_y
        |                                  |
        +---------------- Cross-Covariance Sigma_yx -----------------+
                                           |
                                           v
                             [ SVD: Sigma_yx = U * D * V^T ]
                                           |
                                           v
                             [ Determinant Sign Correction S ]
                                           |
                                           v
       Closed-Form Optimal Transform: R* = U S V^T, scale s*, translation t*
                     (Exact global minimum in <10 microseconds!)
```

---

## 2. Mathematical Formulation & Step-by-Step Derivation

### 2.1 Centroid Decoupling
Compute the sample barycenters of both point clouds:

$$
\boldsymbol{\mu}_x = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i, \quad \boldsymbol{\mu}_y = \frac{1}{N} \sum_{i=1}^N \mathbf{y}_i
$$

Compute the zero-centered coordinates and source variance $\sigma_x^2$:

$$
\mathbf{x}'_i = \mathbf{x}_i - \boldsymbol{\mu}_x, \quad \mathbf{y}'_i = \mathbf{y}_i - \boldsymbol{\mu}_y, \quad \sigma_x^2 = \frac{1}{N} \sum_{i=1}^N \|\mathbf{x}'_i\|^2
$$

Once centered, the optimal translation vector is uniquely determined by:

$$
\mathbf{t}^* = \boldsymbol{\mu}_y - s^* \mathbf{R}^* \boldsymbol{\mu}_x
$$

Substituting $\mathbf{t}^*$ back into the error objective eliminates translation, reducing the loss to:

$$
\mathcal{E}(s, \mathbf{R}) = \sigma_y^2 + s^2 \sigma_x^2 - 2 s \cdot \text{tr}\left( \mathbf{R} \mathbf{\Sigma}_{yx} \right)
$$

where $\mathbf{\Sigma}_{yx} \in \mathbb{R}^{3 \times 3}$ is the spatial cross-covariance matrix:

$$
\mathbf{\Sigma}_{yx} = \frac{1}{N} \sum_{i=1}^N \mathbf{y}'_i (\mathbf{x}'_i)^T = \frac{1}{N} \mathbf{Y}' (\mathbf{X}')^T
$$

---

### 2.2 SVD and the Orthogonal Procrustes Theorem
To minimize $\mathcal{E}(s, \mathbf{R})$, we must maximize $\text{tr}(\mathbf{R} \mathbf{\Sigma}_{yx})$ subject to $\mathbf{R} \in \mathrm{SO}(3)$.

Compute the Singular Value Decomposition (SVD) of the cross-covariance:

$$
\mathbf{\Sigma}_{yx} = \mathbf{U} \mathbf{D} \mathbf{V}^T
$$

where $\mathbf{U}, \mathbf{V} \in \mathrm{O}(3)$ are orthogonal matrices, and $\mathbf{D} = \text{diag}(d_1, d_2, d_3)$ contains the singular values ordered $d_1 \ge d_2 \ge d_3 \ge 0$.

Let $\mathbf{M} = \mathbf{V}^T \mathbf{R}^T \mathbf{U}$. Then:

$$
\text{tr}\left( \mathbf{R} \mathbf{\Sigma}_{yx} \right) = \text{tr}\left( \mathbf{R} \mathbf{U} \mathbf{D} \mathbf{V}^T \right) = \text{tr}\left( \mathbf{D} \mathbf{M} \right) = \sum_{i=1}^3 d_i M_{ii}
$$

Because $\mathbf{M}$ is an orthogonal matrix, its diagonal entries satisfy $|M_{ii}| \le 1$. The maximum is achieved when $M_{ii} = 1$ (i.e., $\mathbf{M} = \mathbf{I}$), yielding $\mathbf{R} = \mathbf{U} \mathbf{V}^T$.

---

### 2.3 The Reflection Correction Matrix $\mathbf{S}$
However, $\det(\mathbf{U} \mathbf{V}^T) = \det(\mathbf{U}) \det(\mathbf{V}) = \pm 1$.
- If $\det(\mathbf{U} \mathbf{V}^T) = +1$, then $\mathbf{R}^* = \mathbf{U} \mathbf{V}^T$ is a valid physical rotation in $\mathrm{SO}(3)$.
- If $\det(\mathbf{U} \mathbf{V}^T) = -1$, the transformation contains an improper mirror reflection. 

To enforce a valid rotation without flipping the two largest principal components, the sign of the smallest singular value $d_3$ is inverted:

$$
\mathbf{S} = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & \det(\mathbf{U}) \det(\mathbf{V})
\end{bmatrix}
$$

The **globally optimal rotation matrix** is:

$$
\mathbf{R}^* = \mathbf{U} \mathbf{S} \mathbf{V}^T
$$

---

### 2.4 Optimal Scale and Translation
With rotation $\mathbf{R}^*$ determined, setting $\frac{\partial \mathcal{E}}{\partial s} = 0$ yields the optimal scale $s^*$:

$$
s^* = \frac{1}{\sigma_x^2} \text{tr}\left( \mathbf{D} \mathbf{S} \right) = \frac{d_1 + d_2 + \det(\mathbf{U}\mathbf{V}^T) \cdot d_3}{\sigma_x^2}
$$

*(Note: For strict $\mathrm{SE}(3)$ rigid alignment without scaling, simply fix $s^* = 1.0$.)*

The optimal translation $\mathbf{t}^*$ is:

$$
\mathbf{t}^* = \boldsymbol{\mu}_y - s^* \mathbf{R}^* \boldsymbol{\mu}_x
$$

---

## 3. Python Reference Implementation

```python
import numpy as np

def umeyama_alignment(X: np.ndarray, Y: np.ndarray, with_scale: bool = True):
    """
    Computes closed-form optimal similarity transform (s, R, t) in Sim(3) 
    aligning source point cloud X to target point cloud Y: Y ~ s * R * X + t.
    X: [N, 3] source points
    Y: [N, 3] target points
    with_scale: True for Sim(3), False for SE(3) rigid registration.
    Returns: (s, R, t)
    """
    assert X.shape == Y.shape and X.shape[1] == 3, "Point clouds must be [N, 3]"
    n = X.shape[0]
    
    # 1. Centroids
    mu_x = np.mean(X, axis=0)
    mu_y = np.mean(Y, axis=0)
    
    # 2. Centered coordinates
    Xc = X - mu_x
    Yc = Y - mu_y
    
    # 3. Source variance
    var_x = np.mean(np.sum(Xc ** 2, axis=1))
    
    # 4. Cross-covariance matrix Sigma_yx: [3, 3]
    Sigma_yx = (Yc.T @ Xc) / n
    
    # 5. SVD
    U, D, Vt = np.linalg.svd(Sigma_yx)
    V = Vt.T
    
    # 6. Reflection check: det(U) * det(V)
    det_UV = np.linalg.det(U) * np.linalg.det(V)
    S = np.eye(3)
    if det_UV < 0:
        S[2, 2] = -1.0
        
    # 7. Optimal Rotation in SO(3)
    R = U @ S @ V.T
    
    # 8. Optimal Scale
    if with_scale:
        scale = (1.0 / var_x) * np.trace(np.diag(D) @ S)
    else:
        scale = 1.0
        
    # 9. Optimal Translation
    t = mu_y - scale * (R @ mu_x)
    
    return scale, R, t
```

---

## 4. Models in the Vault Utilizing Procrustes / Umeyama Alignment

- **[[architectures/pose-and-robotics-manipulation/foundationpose|FoundationPose]]**: 6-DoF model-based tracking aligning 3D mesh vertices to neural depth fields.
- **[[architectures/pose-and-robotics-manipulation/megapose|MegaPose]] & [[architectures/pose-and-robotics-manipulation/cosypose|CosyPose]]**: Object pose hypothesis generation via iterative Procrustes refinement.
- **[[architectures/spatial-radiance-and-slam/droid-slam|DROID-SLAM]]**: Metric scale and $\mathrm{Sim}(3)$ gauge alignment against ground-truth trajectories.
- **[[topics/slam-and-spatial-perception/00-slam-and-spatial-perception-moc|SLAM Evaluation Tools]]**: Core mathematical engine for evaluating Absolute Trajectory Error (ATE).

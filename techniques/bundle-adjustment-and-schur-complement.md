---
title: "Bundle Adjustment & The Schur Complement: Scalable SLAM Optimization"
type: "Technique"
domain: "Visual SLAM, 3D Reconstruction & Spatial Perception"
tags:
  - technique
  - bundle-adjustment
  - schur-complement
  - visual-slam
  - non-linear-optimization
  - structure-from-motion
  - factor-graphs
status: evergreen
updated: 2026-09-09
aliases:
  - "Bundle Adjustment"
  - "Schur Complement"
  - "Normal Equations Marginalization"
  - "Reduced Camera System"
---

# 📐 Bundle Adjustment & The Schur Complement: Scalable SLAM Optimization

## 1. High-Level Concept & The Multi-View Optimization Problem

In visual SLAM and Structure-from-Motion (SfM) (e.g., [[architectures/spatial-radiance-and-slam/droid-slam|DROID-SLAM]], [[architectures/spatial-radiance-and-slam/3dgs-slam|3DGS-SLAM]], COLMAP), a camera observes a physical scene from multiple viewpoints.
We are given $M$ keyframes with unknown camera poses $\mathbf{T}_i \in \mathrm{SE}(3)$ and $N$ unknown 3D world landmark points $\mathbf{X}_j \in \mathbb{R}^3$.

**Bundle Adjustment (BA)** jointly refines all camera poses and all 3D point positions by minimizing the total non-linear **reprojection error**:

$$
\min_{\{\mathbf{T}_i\}, \{\mathbf{X}_j\}} \sum_{i=1}^M \sum_{j=1}^N v_{ij} \cdot \rho\left( \left\| \mathbf{p}_{ij} - \pi(\mathbf{T}_i \mathbf{X}_j) \right\|_{\mathbf{\Sigma}_{ij}^{-1}}^2 \right)
$$

where:
- $v_{ij} \in \{0, 1\}$ indicates whether point $j$ is visible in camera $i$.
- $\mathbf{p}_{ij} \in \mathbb{R}^2$ is the observed 2D pixel coordinate of point $j$ in frame $i$.
- $\pi: \mathbb{R}^3 \to \mathbb{R}^2$ is the camera pinhole projection function.
- $\rho(\cdot)$ is a robust M-estimator (e.g., Huber loss) to suppress outlier correspondences.

### The Massive Dimensionality Barrier
In real-world robotics and autonomous mapping:
- Number of keyframes: $M \approx 100$ (giving $6 \times 100 = 600$ pose parameters).
- Number of 3D landmarks: $N \approx 50,000\text{ to }500,000$ (giving $3 \times 50,000 = 150,000$ coordinate parameters).
- Total parameter dimension: $D = 6M + 3N \approx 150,600$.

Solving the Gauss-Newton or Levenberg-Marquardt normal equations requires solving:

$$
\mathbf{H} \Delta \mathbf{x} = -\mathbf{b}
$$

A direct linear solver (e.g., standard Cholesky or LU decomposition) on a $150,600 \times 150,600$ matrix has cubic computational complexity $\mathcal{O}(D^3)$, requiring hours to compute a single optimization iteration!

### The Schur Complement Solution
Notice that **3D points are conditionally independent given the camera poses**:
- An observation of point $j$ in camera $i$ creates a constraint between camera $i$ and point $j$.
- There is **zero direct constraint between point $j$ and point $k$**.
- This imposes an **Arrowhead / Bipartite block sparsity structure** on the Hessian $\mathbf{H}$.

The **Schur Complement** analytically marginalizes all $N$ 3D points:
1. Inverts the 3D landmark block in strictly **linear time $\mathcal{O}(N)$** (because it consists of $N$ decoupled $3 \times 3$ diagonal blocks).
2. Reduces the system to a compact **Reduced Camera System** involving only the $6M$ camera pose variables (a $600 \times 600$ matrix).
3. Solves the reduced camera update in **$<5\text{ ms}$**, then recovers all 50,000 point updates in parallel via back-substitution!

```
Hessian Block Sparsity & The Schur Complement Reduction:

Full Hessian H (150,600 x 150,600):              Reduced Camera System (600 x 600):
   Poses (6M)    Points (3N)
  +-----------+----------------+
  |           |                |
  |  B (6Mx6M)|    E (6Mx3N)   |
  |           |                |
  +-----------+----------------+               [ B - E * C^-1 * E^T ] * Delta_pose = v - E * C^-1 * w
  |           | C1             |                                 |
  |   E^T     |    C2          |                                 v
  |  (3Nx6M)  |       C3       |                 Solves camera poses Delta_pose in <5 ms!
  |           |          ...CN |                                 |
  +-----------+----------------+                                 v
  (Point block C is Block-Diagonal!        Back-Substitution: Delta_points = C^-1 * (w - E^T * Delta_pose)
   Inversion takes O(N) linear time!)                    (Solved in parallel!)
```

---

## 2. Mathematical Formulation

### 2.1 The Block Partitioned Normal Equations
Partition the perturbation vector $\Delta \mathbf{x}$ into camera pose increments $\Delta \mathbf{x}_p \in \mathbb{R}^{6M}$ and 3D point increments $\Delta \mathbf{x}_l \in \mathbb{R}^{3N}$:

$$
\begin{bmatrix}
\mathbf{B} & \mathbf{E} \\
\mathbf{E}^T & \mathbf{C}
\end{bmatrix}
\begin{bmatrix}
\Delta \mathbf{x}_p \\
\Delta \mathbf{x}_l
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{v} \\
\mathbf{w}
\end{bmatrix}
$$

where:
- $\mathbf{B} = \mathbf{J}_p^T \mathbf{J}_p \in \mathbb{R}^{6M \times 6M}$ is the camera pose block.
- $\mathbf{E} = \mathbf{J}_p^T \mathbf{J}_l \in \mathbb{R}^{6M \times 3N}$ is the camera-landmark correlation block.
- $\mathbf{C} = \mathbf{J}_l^T \mathbf{J}_l \in \mathbb{R}^{3N \times 3N}$ is the landmark block.
- $\mathbf{v} = -\mathbf{J}_p^T \mathbf{r} \in \mathbb{R}^{6M}$ and $\mathbf{w} = -\mathbf{J}_l^T \mathbf{r} \in \mathbb{R}^{3N}$ are the residual gradient vectors.

---

### 2.2 The Block-Diagonal Structure of $\mathbf{C}$
Because each landmark error depends only on that landmark's 3D position:

$$
\mathbf{C} = \text{diag}\left( \mathbf{C}_1, \; \mathbf{C}_2, \; \dots, \; \mathbf{C}_N \right)
$$

where each sub-block $\mathbf{C}_j \in \mathbb{R}^{3 \times 3}$ is:

$$
\mathbf{C}_j = \sum_{i \in \text{Obs}(j)} \mathbf{J}_{ij, l}^T \mathbf{\Sigma}_{ij}^{-1} \mathbf{J}_{ij, l}
$$

The inverse $\mathbf{C}^{-1}$ is trivially evaluated by inverting each $3 \times 3$ block independently:

$$
\mathbf{C}^{-1} = \text{diag}\left( \mathbf{C}_1^{-1}, \; \mathbf{C}_2^{-1}, \; \dots, \; \mathbf{C}_N^{-1} \right)
$$

Inverting a $3 \times 3$ matrix has a closed-form analytical solution via Cramer's rule ($<0.05\ \mu\text{s}$). The entire matrix $\mathbf{C}^{-1}$ is computed in $\mathcal{O}(N)$ time.

---

### 2.3 The Schur Complement Derivation
From the second row of the normal equations:

$$
\mathbf{E}^T \Delta \mathbf{x}_p + \mathbf{C} \Delta \mathbf{x}_l = \mathbf{w} \implies \Delta \mathbf{x}_l = \mathbf{C}^{-1} \left( \mathbf{w} - \mathbf{E}^T \Delta \mathbf{x}_p \right)
$$

Substitute $\Delta \mathbf{x}_l$ into the first row:

$$
\mathbf{B} \Delta \mathbf{x}_p + \mathbf{E} \left[ \mathbf{C}^{-1} \left( \mathbf{w} - \mathbf{E}^T \Delta \mathbf{x}_p \right) \right] = \mathbf{v}
$$

Rearranging terms yields the **Reduced Camera System**:

$$
\left( \mathbf{B} - \mathbf{E} \mathbf{C}^{-1} \mathbf{E}^T \right) \Delta \mathbf{x}_p = \mathbf{v} - \mathbf{E} \mathbf{C}^{-1} \mathbf{w}
$$

Let:
- $\mathbf{S} = \mathbf{B} - \mathbf{E} \mathbf{C}^{-1} \mathbf{E}^T \in \mathbb{R}^{6M \times 6M}$ be the **Schur Complement**.
- $\mathbf{g} = \mathbf{v} - \mathbf{E} \mathbf{C}^{-1} \mathbf{w} \in \mathbb{R}^{6M}$ be the modified camera gradient.

We solve the compact $6M \times 6M$ linear system:

$$
\mathbf{S} \Delta \mathbf{x}_p = \mathbf{g} \implies \Delta \mathbf{x}_p = \mathbf{S}^{-1} \mathbf{g}
$$

---

### 2.4 Parallel Back-Substitution for Landmark Points
Once the optimal camera pose update $\Delta \mathbf{x}_p$ is determined, the landmark updates $\Delta \mathbf{x}_l$ are recovered in parallel without solving any linear system:

$$
\Delta \mathbf{x}_l = \mathbf{C}^{-1} \left( \mathbf{w} - \mathbf{E}^T \Delta \mathbf{x}_p \right)
$$

For each 3D point $j \in \{1, \dots, N\}$:

$$
\Delta \mathbf{X}_j = \mathbf{C}_j^{-1} \left( \mathbf{w}_j - \sum_{i \in \text{Obs}(j)} \mathbf{J}_{ij, l}^T \mathbf{\Sigma}_{ij}^{-1} \mathbf{J}_{ij, p} \Delta \mathbf{T}_i \right)
$$

---

## 3. Python Reference Implementation

```python
import numpy as np

def schur_complement_solve(B: np.ndarray, E: np.ndarray, C_blocks: list[np.ndarray], 
                           v: np.ndarray, w: np.ndarray):
    """
    Solves normal equations via the Schur Complement.
    B: [6M, 6M] camera pose block
    E: [6M, 3N] camera-landmark cross block
    C_blocks: list of N [3, 3] landmark diagonal blocks
    v: [6M] camera residual gradient
    w: [3N] landmark residual gradient
    Returns: (delta_p [6M], delta_l [3N])
    """
    num_cameras = B.shape[0] // 6
    num_points = len(C_blocks)
    
    # 1. Invert C block-by-block in O(N) linear time
    C_inv_blocks = [np.linalg.inv(C_j + 1e-6 * np.eye(3)) for C_j in C_blocks]
    
    # Construct C_inv @ w and E @ C_inv
    C_inv_w = np.zeros_like(w)
    E_C_inv = np.zeros_like(E)
    
    for j in range(num_points):
        p_idx = j * 3
        # Inverted landmark multiply
        C_inv_w[p_idx : p_idx + 3] = C_inv_blocks[j] @ w[p_idx : p_idx + 3]
        # Column slice of E
        E_C_inv[:, p_idx : p_idx + 3] = E[:, p_idx : p_idx + 3] @ C_inv_blocks[j]
        
    # 2. Formulate Schur Complement: S = B - E @ C^-1 @ E^T
    S = B - (E_C_inv @ E.T)
    # Formulate modified RHS: g = v - E @ C^-1 @ w
    g = v - (E @ C_inv_w)
    
    # 3. Solve compact 6M x 6M reduced camera system
    delta_p = np.linalg.solve(S + 1e-6 * np.eye(B.shape[0]), g)
    
    # 4. Back-substitution for landmark points
    delta_l = C_inv_w - (E_C_inv.T @ delta_p)
    
    return delta_p, delta_l
```

---

## 4. Models in the Vault Utilizing the Schur Complement

- **[[architectures/spatial-radiance-and-slam/droid-slam|DROID-SLAM]]**: Differentiable recurrent visual SLAM solving dense bundle adjustment with Schur complement marginalization.
- **[[architectures/spatial-radiance-and-slam/3dgs-slam|3DGS-SLAM]]**: Joint optimization of camera keyframes and 3D Gaussian radiance fields.
- **[[architectures/spatial-radiance-and-slam/splatam|SplaTAM]]**: Sub-centimeter camera tracking and volumetric mapping.
- **[[topics/slam-and-spatial-perception/00-slam-and-spatial-perception-moc|SLAM & Spatial Perception MOC]]**: Connected as the foundational mathematical engine for visual bundle adjustment.

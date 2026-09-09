---
title: "Iterative Closest Point (ICP & Generalized ICP): Point-to-Plane 3D Registration"
type: "Technique"
domain: "3D Point Clouds, LiDAR SLAM & Spatial Localization"
tags:
  - technique
  - icp
  - generalized-icp
  - point-to-plane
  - lidar-slam
  - 3d-registration
  - pointclouds
status: evergreen
updated: 2026-09-09
aliases:
  - "Iterative Closest Point"
  - "ICP"
  - "Point-to-Plane ICP"
  - "Generalized ICP"
  - "G-ICP"
---

# 📐 Iterative Closest Point (ICP & Generalized ICP): Point-to-Plane 3D Registration

## 1. High-Level Concept & The Geometric Alignment Challenge

In LiDAR SLAM, autonomous navigation, and 3D robotic inspection (e.g., [[topics/slam-and-spatial-perception/README|LiDAR SLAM]], [[architectures/spatial-radiance-and-slam/3dgs-slam|3DGS-SLAM]], [[architectures/pose-and-robotics-manipulation/foundationpose|FoundationPose]]), a sensor captures consecutive 3D point cloud scans: a source scan $\mathcal{P} = \{\mathbf{p}_i\}_{i=1}^N$ and a target scan $\mathcal{Q} = \{\mathbf{q}_j\}_{j=1}^M$.

The goal of **Point Cloud Registration** is to compute the optimal rigid transformation $(\mathbf{R}, \mathbf{t}) \in \mathrm{SE}(3)$ that aligns the source scan into the target coordinate frame:

$$
\mathbf{R}^*, \; \mathbf{t}^* = \arg\min_{\mathbf{R} \in \mathrm{SO}(3), \; \mathbf{t} \in \mathbb{R}^3} \sum_{i=1}^N d\left( \mathbf{R} \mathbf{p}_i + \mathbf{t}, \; \mathcal{Q} \right)
$$

### The Failure of Point-to-Point ICP (Besl & McKay, 1992)
Standard point-to-point ICP minimizes the Euclidean distance between closest discrete points:

$$
E_{\text{pt2pt}}(\mathbf{R}, \mathbf{t}) = \sum_{i=1}^N \left\| \mathbf{R} \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i \right\|^2
$$

*The Fundamental Flaw*: LiDAR pulses sample continuous physical surfaces (walls, roads, building facades) at arbitrary, irregular spatial locations. **Exact point-to-point coincidence almost never exists in reality!**
- Point-to-point ICP penalizes source points that slide tangentially along a flat wall.
- In long corridors, highways, or flat rooms, point-to-point ICP suffers from severe geometric under-constraining, converging slowly ($>50\text{ iterations}$) and drifting uncontrollably along the trajectory direction.

---

### The Point-to-Plane ICP Solution (Chen & Medioni, 1992)
Instead of forcing point-to-point coincidence, **Point-to-Plane ICP** projects the registration residual along the target surface's **estimated normal vector $\mathbf{n}_i$**:

$$
e_i = \mathbf{n}_i^T \left( \mathbf{R} \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i \right)
$$

*The Geometric Advantage*: Source points are permitted to **slide frictionlessly along flat tangent surfaces** without incurring artificial error penalties!
- Normalizes errors perpendicular to the surface.
- Accelerates numerical convergence by an order of magnitude (typically converging in $5\text{--}8\text{ iterations}$).
- Prevents corridor drift in structured indoor and highway environments.

---

### Generalized ICP (G-ICP) (Segal et al., RSS 2009)
G-ICP unifies point-to-point and point-to-plane by modeling both source and target points as local 3D Gaussian probability distributions: $\mathbf{p}_i \sim \mathcal{N}(\hat{\mathbf{p}}_i, \mathbf{C}_i^P)$ and $\mathbf{q}_i \sim \mathcal{N}(\hat{\mathbf{q}}_i, \mathbf{C}_i^Q)$.
It minimizes the plane-to-plane Mahalanobis distance:

$$
E_{\text{G-ICP}}(\mathbf{R}, \mathbf{t}) = \sum_{i=1}^N \mathbf{d}_i^T \left( \mathbf{C}_i^Q + \mathbf{R} \mathbf{C}_i^P \mathbf{R}^T \right)^{-1} \mathbf{d}_i
$$

where $\mathbf{d}_i = \mathbf{R} \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i$. For planar surfaces, the covariance matrix has high variance along tangent axes and near-zero variance along the normal axis, reproducing point-to-plane physics probabilistically.

```
Point-to-Point vs. Point-to-Plane Registration:

Point-to-Point ICP:                              Point-to-Plane ICP:
Source Point p --------> Target Point q          Source Point p ------+
(Forced to coincide at discrete point!                |               | Projected along
 Fails on flat continuous surfaces!)                  v               v surface normal n_i!
                                                --------------------------------- Target Plane
                                                (Point slides freely along tangent plane!
                                                 Faster convergence, zero corridor drift!)
```

---

## 2. Mathematical Formulation

### 2.1 Linearized Optimization on the Lie Algebra $\mathfrak{so}(3)$
Let $\Delta \mathbf{x} = [\boldsymbol{\omega}, \mathbf{t}]^T \in \mathbb{R}^6$ be the minimal tangent perturbation vector.
Under the small-angle rotation approximation:

$$
\mathbf{R} \approx \mathbf{I} + [\boldsymbol{\omega}]_\times = \begin{bmatrix} 1 & -\omega_z & \omega_y \\ \omega_z & 1 & -\omega_x \\ -\omega_y & \omega_x & 1 \end{bmatrix}
$$

The transformed source point is:

$$
\mathbf{p}_i' \approx \mathbf{p}_i + \boldsymbol{\omega} \times \mathbf{p}_i + \mathbf{t} = \mathbf{p}_i - [\mathbf{p}_i]_\times \boldsymbol{\omega} + \mathbf{t}
$$

Substitute into the point-to-plane error equation:

$$
e_i = \mathbf{n}_i^T \left( \mathbf{p}_i - [\mathbf{p}_i]_\times \boldsymbol{\omega} + \mathbf{t} - \mathbf{q}_i \right) = \mathbf{n}_i^T (\mathbf{p}_i - \mathbf{q}_i) + \mathbf{n}_i^T \mathbf{t} + (\mathbf{p}_i \times \mathbf{n}_i)^T \boldsymbol{\omega}
$$

---

### 2.2 Constructing the Linear System $\mathbf{J}_i \Delta \mathbf{x} = -r_i$
Define the $1 \times 6$ Jacobian row vector $\mathbf{J}_i$:

$$
\mathbf{J}_i = \begin{bmatrix} (\mathbf{p}_i \times \mathbf{n}_i)^T & \mathbf{n}_i^T \end{bmatrix} \in \mathbb{R}^{1 \times 6}
$$

and the scalar residual $r_i = \mathbf{n}_i^T (\mathbf{p}_i - \mathbf{q}_i)$.

The non-linear least squares problem is solved via the Gauss-Newton normal equations:

$$
\mathbf{H} \Delta \mathbf{x} = -\mathbf{b}
$$

$$
\mathbf{H} = \sum_{i=1}^N \mathbf{J}_i^T \mathbf{J}_i \in \mathbb{R}^{6 \times 6}, \quad \mathbf{b} = \sum_{i=1}^N \mathbf{J}_i^T r_i \in \mathbb{R}^6
$$

The compact $6 \times 6$ system is solved in $<5\ \mu\text{s}$ using Cholesky factorization:

$$
\Delta \mathbf{x} = -\mathbf{H}^{-1} \mathbf{b}
$$

Pose updates are composed using the matrix exponential on $\mathrm{SE}(3)$:

$$
\mathbf{T}_{k+1} = \exp\left( \Delta \mathbf{x}^\wedge \right) \mathbf{T}_k
$$

---

## 3. Python Reference Implementation

```python
import numpy as np
from scipy.spatial import cKDTree

def point_to_plane_icp(source_pts: np.ndarray, target_pts: np.ndarray, 
                       target_normals: np.ndarray, max_iterations: int = 20, 
                       tolerance: float = 1e-5):
    """
    Point-to-Plane ICP Registration.
    source_pts: [N, 3] source point cloud
    target_pts: [M, 3] target point cloud
    target_normals: [M, 3] estimated target normal vectors
    Returns: (R [3, 3], t [3])
    """
    kdtree = cKDTree(target_pts)
    
    R = np.eye(3)
    t = np.zeros(3)
    current_pts = source_pts.copy()
    
    for iteration in range(max_iterations):
        # 1. Nearest-neighbor correspondence association
        dists, indices = kdtree.query(current_pts, k=1)
        matched_targets = target_pts[indices]
        matched_normals = target_normals[indices]
        
        # 2. Compute point-to-plane residuals and Jacobians
        # r_i = n_i^T * (p_i - q_i)
        diffs = current_pts - matched_targets
        residuals = np.sum(matched_normals * diffs, axis=1) # [N]
        
        # J_i = [(p_i x n_i)^T, n_i^T]
        cross_pn = np.cross(current_pts, matched_normals) # [N, 3]
        J = np.hstack([cross_pn, matched_normals]) # [N, 6]
        
        # 3. Assemble Gauss-Newton system: H * dx = -b
        H = J.T @ J # [6, 6]
        b = J.T @ residuals # [6]
        
        # Solve linear system
        delta_x = -np.linalg.solve(H + 1e-6 * np.eye(6), b)
        
        omega = delta_x[:3]
        v = delta_x[3:]
        
        # 4. Small-angle rotation update
        theta = np.linalg.norm(omega)
        if theta > 1e-7:
            axis = omega / theta
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            delta_R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        else:
            delta_R = np.eye(3)
            
        delta_t = v
        
        # Update state
        R = delta_R @ R
        t = delta_R @ t + delta_t
        current_pts = (delta_R @ current_pts.T).T + delta_t
        
        if np.linalg.norm(delta_x) < tolerance:
            break
            
    return R, t
```

---

## 4. Models in the Vault Utilizing ICP

- **[[topics/slam-and-spatial-perception/README|LiDAR SLAM]]**: High-rate LiDAR-inertial odometry registering raw points directly to continuous spatial maps.
- **[[architectures/spatial-radiance-and-slam/droid-slam|DROID-SLAM]] & [[architectures/spatial-radiance-and-slam/3dgs-slam|3DGS-SLAM]]**: 3D geometric loop closure and dense frame alignment.
- **[[architectures/pose-and-robotics-manipulation/foundationpose|FoundationPose]]**: Refines 6-DoF object poses against CAD models using point-to-plane residual optimization.

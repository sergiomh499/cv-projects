---
title: "Lie Algebra se(3) & SE(3) Geometry: Analytical Photometric Tracking for SLAM & Robotics"
type: "Technique"
domain: "Robotics Kinematics & Visual SLAM"
tags:
  - technique
  - lie-algebra
  - se3
  - so3
  - visual-slam
  - pose-estimation
  - robotics
  - procrustes
status: evergreen
updated: 2026-09-09
aliases:
  - "Lie Algebra se(3)"
  - "SE(3) Pose Tracking"
  - "Lie Groups in Robotics"
  - "Photometric Residual Tracking"
---

# 📐 Lie Algebra $\mathfrak{se}(3)$ & $\mathrm{SE}(3)$ Geometry: Analytical Photometric Tracking

## 1. High-Level Concept & The Rotation Parameterization Bottleneck

In robotics, 6-DoF object manipulation, and visual SLAM, representing 3D spatial motion requires estimating a rigid transformation $(\mathbf{R}, \mathbf{t}) \in \mathrm{SE}(3)$, consisting of a 3D rotation $\mathbf{R}$ and translation vector $\mathbf{t} \in \mathbb{R}^3$.

However, parameterizing 3D rotations in standard gradient-based optimization poses severe mathematical hazards:
1. **Direct Matrix Parameterization ($\mathbb{R}^{3 \times 3}$)**: A $3\times 3$ matrix has 9 elements, but rotation possesses only 3 degrees of freedom. Optimizing 9 unconstrained weights violates the strict **orthogonality constraints**: $\mathbf{R}^T \mathbf{R} = \mathbf{I}$ and $\det(\mathbf{R}) = +1$.
2. **Euler Angles ($\alpha, \beta, \gamma$)**: Suffer from **gimbal lock** (loss of a degree of freedom when pitch approaches $\pm 90^\circ$) and non-unique, discontinuous representations.
3. **Unit Quaternions ($\mathbf{q} \in \mathbb{H}$)**: Contain 4 elements subject to a unit-norm constraint $\|\mathbf{q}\| = 1$, requiring projection steps that destabilize Gauss-Newton second-order solvers.

### The Lie Theory Solution
The **Special Euclidean Group $\mathrm{SE}(3)$** is a smooth Riemannian **Lie Group** (a continuous mathematical manifold).
Its associated **Lie Algebra $\mathfrak{se}(3)$** is the tangent vector space defined at the identity transformation.

Any small 6-DoF pose adjustment is parameterized as an **unconstrained 6-dimensional vector** $\boldsymbol{\xi} = [\boldsymbol{\upsilon}, \boldsymbol{\omega}]^T \in \mathbb{R}^6$ (3 translational velocities $\boldsymbol{\upsilon}$ and 3 rotational velocities $\boldsymbol{\omega}$). Optimization proceeds in unconstrained tangent space $\mathbb{R}^6$, and is mapped back to the manifold via the closed-form **Matrix Exponential**:

$$
\mathbf{T} \leftarrow \exp(\boldsymbol{\xi}^\wedge) \mathbf{T}
$$

This guarantees that the updated transformation $\mathbf{T}$ **never leaves the $\mathrm{SE}(3)$ manifold**, maintaining perfect orthogonality with zero drift.

```
Manifold SE(3) vs. Tangent Space se(3):

                Tangent Space se(3) ~ R^6 (Unconstrained optimization)
                        ^  xi = [v, omega]^T
                       /
                      /  Matrix Exponential: exp(xi^)
                     v
             +--------------------+
            /      Manifold        \
           /        SE(3)           \
          +--------------------------+
          (Guaranteed valid rigid transformation: R^T R = I, det(R) = 1)
```

---

## 2. Mathematical Formulation

### 2.1 The Lie Groups $\mathrm{SO}(3)$ and $\mathrm{SE}(3)$
The Special Orthogonal Group $\mathrm{SO}(3)$ and Special Euclidean Group $\mathrm{SE}(3)$ are defined as:

$$
\mathrm{SO}(3) = \left\{ \mathbf{R} \in \mathbb{R}^{3 \times 3} \;\middle|\; \mathbf{R}^T \mathbf{R} = \mathbf{I}, \; \det(\mathbf{R}) = +1 \right\}
$$

$$
\mathrm{SE}(3) = \left\{ \mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 4} \;\middle|\; \mathbf{R} \in \mathrm{SO}(3), \; \mathbf{t} \in \mathbb{R}^3 \right\}
$$

---

### 2.2 The Hat Operator $(\cdot)^\wedge$ and Lie Algebra $\mathfrak{se}(3)$
For an angular velocity vector $\boldsymbol{\omega} = [\omega_1, \omega_2, \omega_3]^T \in \mathbb{R}^3$, the skew-symmetric "hat" operator $(\cdot)^\wedge$ produces a matrix in $\mathfrak{so}(3)$:

$$
\boldsymbol{\omega}^\wedge = \begin{bmatrix}
0 & -\omega_3 & \omega_2 \\
\omega_3 & 0 & -\omega_1 \\
-\omega_2 & \omega_1 & 0
\end{bmatrix} \in \mathfrak{so}(3)
$$

For a 6-DoF twist vector $\boldsymbol{\xi} = [\boldsymbol{\upsilon}, \boldsymbol{\omega}]^T \in \mathbb{R}^6$ (where $\boldsymbol{\upsilon} \in \mathbb{R}^3$ is linear velocity and $\boldsymbol{\omega} \in \mathbb{R}^3$ is angular velocity), its matrix representation in $\mathfrak{se}(3)$ is:

$$
\boldsymbol{\xi}^\wedge = \begin{bmatrix}
\boldsymbol{\omega}^\wedge & \boldsymbol{\upsilon} \\
\mathbf{0}^T & 0
\end{bmatrix} \in \mathbb{R}^{4 \times 4}
$$

---

### 2.3 The Exponential Map: Closed-Form Rodrigues Formula
The mapping from tangent space $\mathfrak{se}(3)$ to the manifold $\mathrm{SE}(3)$ is given by the matrix exponential $\exp(\boldsymbol{\xi}^\wedge)$:

$$
\exp(\boldsymbol{\xi}^\wedge) = \begin{bmatrix}
\exp(\boldsymbol{\omega}^\wedge) & \mathbf{V} \boldsymbol{\upsilon} \\
\mathbf{0}^T & 1
\end{bmatrix}
$$

Let $\theta = \|\boldsymbol{\omega}\|$. The rotation matrix $\exp(\boldsymbol{\omega}^\wedge) \in \mathrm{SO}(3)$ is evaluated via **Rodrigues' Formula**:

$$
\exp(\boldsymbol{\omega}^\wedge) = \mathbf{I} + \frac{\sin\theta}{\theta} \boldsymbol{\omega}^\wedge + \frac{1 - \cos\theta}{\theta^2} (\boldsymbol{\omega}^\wedge)^2
$$

The left Jacobian translation matrix $\mathbf{V} \in \mathbb{R}^{3 \times 3}$ is:

$$
\mathbf{V} = \mathbf{I} + \frac{1 - \cos\theta}{\theta^2} \boldsymbol{\omega}^\wedge + \frac{\theta - \sin\theta}{\theta^3} (\boldsymbol{\omega}^\wedge)^2
$$

---

## 3. Analytical Photometric Residual Tracking

In direct visual SLAM (e.g., [[architectures/spatial-radiance-and-slam/3dgs-slam|3DGS-SLAM]], [[architectures/spatial-radiance-and-slam/droid-slam|DROID-SLAM]]), the camera pose is tracked by minimizing the photometric pixel intensity residual between a current live frame $I_{\text{live}}$ and a projected 3D reference scene $I_{\text{ref}}$:

$$
r_i(\boldsymbol{\xi}) = I_{\text{live}}\left( \pi\left( \exp(\boldsymbol{\xi}^\wedge) \mathbf{T} \mathbf{P}_i \right) \right) - I_{\text{ref}}(\mathbf{u}_i)
$$

where $\mathbf{P}_i = [X, Y, Z, 1]^T$ is a 3D point and $\pi(X, Y, Z) = [f_x \frac{X}{Z} + c_x, f_y \frac{Y}{Z} + c_y]^T$ is the camera pinhole projection.

### Chain Rule Jacobian Decomposition
The $1 \times 6$ analytical Jacobian $\mathbf{J}_i = \frac{\partial r_i}{\partial \boldsymbol{\xi}}$ decomposes into three distinct physical derivatives:

$$
\mathbf{J}_i = \underbrace{\frac{\partial I_{\text{live}}}{\partial \mathbf{u}}}_{\text{Image Gradient } (1 \times 2)} \cdot \underbrace{\frac{\partial \mathbf{u}}{\partial \mathbf{P}_{\text{cam}}}}_{\text{Camera Projection } (2 \times 3)} \cdot \underbrace{\frac{\partial \mathbf{P}_{\text{cam}}}{\partial \boldsymbol{\xi}}}_{\mathrm{SE}(3) \text{ Motion } (3 \times 6)}
$$

where:
1. $\frac{\partial I_{\text{live}}}{\partial \mathbf{u}} = [\nabla I_x, \nabla I_y]$ (spatial image Sobel gradients).
2. $\frac{\partial \mathbf{u}}{\partial \mathbf{P}_{\text{cam}}} = \begin{bmatrix} \frac{f_x}{Z} & 0 & -\frac{f_x X}{Z^2} \\ 0 & \frac{f_y}{Z} & -\frac{f_y Y}{Z^2} \end{bmatrix}$.
3. $\frac{\partial \mathbf{P}_{\text{cam}}}{\partial \boldsymbol{\xi}} = \begin{bmatrix} \mathbf{I}_{3\times 3} & -(\mathbf{P}_{\text{cam}})^\wedge \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 & Z & -Y \\ 0 & 1 & 0 & -Z & 0 & X \\ 0 & 0 & 1 & Y & -X & 0 \end{bmatrix}$.

---

### 3.2 The Second-Order Gauss-Newton Step
The optimal pose perturbation $\boldsymbol{\xi}^* \in \mathbb{R}^6$ is solved directly:

$$
\mathbf{H} \boldsymbol{\xi}^* = \mathbf{b} \implies \boldsymbol{\xi}^* = \mathbf{H}^{-1} \mathbf{b}
$$

where the $6 \times 6$ Approximate Hessian $\mathbf{H}$ and gradient vector $\mathbf{b}$ are:

$$
\mathbf{H} = \sum_{i=1}^N \mathbf{J}_i^T \mathbf{J}_i, \quad \mathbf{b} = - \sum_{i=1}^N \mathbf{J}_i^T r_i
$$

Because $\mathbf{H}$ is strictly $6 \times 6$, inverting $\mathbf{H}$ takes **$<5\ \mu\text{s}$**, achieving sub-millisecond real-time camera tracking.

---

## 4. PyTorch Reference Implementation

```python
import torch

def skew_symmetric(omega: torch.Tensor) -> torch.Tensor:
    """Computes the 3x3 hat operator matrix from 3D vector omega."""
    wx, wy, wz = omega.unbind(-1)
    zeros = torch.zeros_like(wx)
    return torch.stack([
        zeros, -wz,    wy,
        wz,    zeros, -wx,
        -wy,   wx,    zeros
    ], dim=-1).reshape(*omega.shape[:-1], 3, 3)

def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """
    Closed-form exponential map from tangent space se(3) to manifold SE(3).
    xi: [B, 6] twist vector [v_x, v_y, v_z, w_x, w_y, w_z]
    Returns: [B, 4, 4] rigid transformation matrix T in SE(3)
    """
    b = xi.shape[0]
    v = xi[:, :3]
    w = xi[:, 3:]
    
    theta = torch.norm(w, dim=-1, keepdim=True).clamp(min=1e-7)
    w_hat = skew_symmetric(w)
    w_hat2 = torch.bmm(w_hat, w_hat)
    
    # Rodrigues Formula for SO(3)
    I3 = torch.eye(3, device=xi.device).unsqueeze(0).repeat(b, 1, 1)
    sin_term = torch.sin(theta).unsqueeze(-1) / theta.unsqueeze(-1)
    cos_term = (1.0 - torch.cos(theta)).unsqueeze(-1) / (theta.unsqueeze(-1) ** 2)
    R = I3 + sin_term * w_hat + cos_term * w_hat2
    
    # Left Jacobian V
    inv_theta3 = (theta - torch.sin(theta)).unsqueeze(-1) / (theta.unsqueeze(-1) ** 3)
    V = I3 + cos_term * w_hat + inv_theta3 * w_hat2
    t = torch.bmm(V, v.unsqueeze(-1)).squeeze(-1)
    
    # Assemble 4x4 matrix
    T = torch.eye(4, device=xi.device).unsqueeze(0).repeat(b, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T
```

---

## 5. Models & Cookbooks Utilizing Lie Algebra $\mathfrak{se}(3)$

- **[[cookbooks/18-3dgs-lie-slam/README|Cookbook 18: 3DGS Lie SLAM]]**: Analytical $\mathfrak{se}(3)$ camera tracking directly on Gaussian radiance fields.
- **[[architectures/spatial-radiance-and-slam/droid-slam|DROID-SLAM]]**: Differentiable recurrent visual SLAM with Lie algebra bundle adjustment.
- **[[architectures/pose-and-robotics-manipulation/foundationpose|FoundationPose]] & [[architectures/pose-and-robotics-manipulation/megapose|MegaPose]]**: Zero-shot 6-DoF object pose tracking with unconstrained Lie group updates.
- **[[topics/slam-and-spatial-perception/00-slam-and-spatial-perception-moc|SLAM & Spatial Perception MOC]]**: Connected as the foundational mathematical formulation for geometric camera estimation.

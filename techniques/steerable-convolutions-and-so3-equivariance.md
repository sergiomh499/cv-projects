---
title: "Equivariant Neural Networks: SO(3) & SE(3) Steerable Convolutions"
type: "Technique"
domain: "Non-Euclidean Geometric Deep Learning, 3D Vision & Equivariance"
tags:
  - technique
  - equivariance
  - steerable-convolutions
  - so3-equivariance
  - se3-equivariance
  - spherical-harmonics
  - geometric-deep-learning
  - 3d-vision
status: evergreen
updated: 2026-09-09
aliases:
  - "Equivariant Neural Networks"
  - "SO(3) Equivariance"
  - "SE(3) Steerable Convolutions"
  - "Spherical Harmonics"
  - "Wigner D-Matrices"
  - "Clebsch-Gordan Tensor Product"
---

# 🌐 Equivariant Neural Networks: SO(3) & SE(3) Steerable Convolutions

## 1. High-Level Concept & The Rotational Invariance Trap

In robotic manipulation (AnyGrasp), 6-DoF object pose estimation (FoundationPose), autonomous 3D LiDAR perception, and molecular geometry:
Physical objects and spatial scenes exist in continuous three-dimensional space governed by the **Special Euclidean Group $\mathrm{SE}(3) = \mathbb{R}^3 \rtimes \mathrm{SO}(3)$** (3D translations and 3D rotations).

### The Breakdown of Standard Vision Networks Under 3D Rotation
Standard Convolutional Networks and Vision Transformers exhibit a critical mathematical flaw:
- **Translation Equivariance Only**: Convolutions commute with $2\text{D}$ translations ($\mathcal{T}_{\mathbf{t}} \circ f = f \circ \mathcal{T}_{\mathbf{t}}$).
- **Chaos Under 3D Rotations**: When an object is rotated in $3\text{D}$ space by rotation matrix $\mathbf{R} \in \mathrm{SO}(3)$, the intermediate feature activations of standard neural networks **change unpredictably and chaotically**:
$$f(\mathbf{R} \cdot \mathbf{X}) \ne \mathbf{R} \cdot f(\mathbf{X})$$

#### The Brittle Data Augmentation Band-Aid
Practitioners conventionally attempt to patch this flaw via brute-force **rotational data augmentation** (randomly rotating point clouds or CAD meshes during training).
- Augmentation requires **$50\times\text{ to }100\times$ more training data and parameters** to memorize all rotational angles.
- It provides **zero formal mathematical guarantees** on unseen orientations, causing robot grippers to drop parts at unfamiliar angles.

---

### The Group Equivariant Paradigm (Cohen & Welling, 2016; Weiler et al., 2018)
An Equivariant Neural Network guarantees by mathematical construction that **transforming the input transforms the output in an identical, predictable manner**:

$$
f\left(\mathbf{T}_g \mathbf{x}\right) = \mathbf{T}'_g f(\mathbf{x}), \quad \forall g \in G
$$

where $\mathbf{T}_g$ and $\mathbf{T}'_g$ are group representations acting on the input and output vector spaces.

For 3D vision, continuous rotation equivariance is achieved through **Steerable Convolutions**:
1. Feature maps are decomposed into **Irreducible Representations (Irreps)** of $\mathrm{SO}(3)$ characterized by angular frequency degree $l \in \{0, 1, 2, \dots\}$.
2. Convolutional kernels are constrained to the space of **Spherical Harmonics $Y_{lm}(\hat{\mathbf{r}})$**.
3. Feature channels interact strictly via **Clebsch-Gordan Tensor Products**, mathematically guaranteeing **exact rotational equivariance with zero data augmentation required**!

```
Equivariance vs. Standard Invariance:

Standard CNN / ViT (Rotational Fragility):
Input Mesh X -------------> [ Standard Network f(X) ] -----------> Feature Vector Z_1
      |                                                                   |  (Z_1 != Z_2!)
      v Rotate by R in SO(3)                                              v Unpredictable drift!
Rotated Mesh R*X ---------> [ Standard Network f(R*X) ] ---------> Feature Vector Z_2

Equivariant Steerable Network (Exact Symmetry Guarantee):
Input Mesh X -------------> [ Steerable Network f(X) ] ----------> Feature Field F(X)
      |                                                                   |
      v Rotate by R in SO(3)                                              v Rotated by Wigner Matrix:
Rotated Mesh R*X ---------> [ Steerable Network f(R*X) ] --------> D^l(R) * F(X)  (Strict Equality!)
```

---

## 2. Mathematical Formulation & Representation Theory

### 2.1 Irreducible Representations (Irreps) of $\mathrm{SO}(3)$
Any linear representation of the continuous 3D rotation group $\mathrm{SO}(3)$ decomposes into a direct sum of orthogonal **Irreducible Representations** labeled by integer degree $l \ge 0$, with dimension $2l + 1$:

- **$l = 0$ (Scalar, Dim 1)**: Invariant under rotation (e.g., density, mass, distance, classification logits):
  $$\mathbf{D}^0(\mathbf{R}) = [1]$$
- **$l = 1$ (Vector, Dim 3)**: Transforms identical to standard 3D Euclidean coordinates (e.g., 3D position, velocity, surface normal $\mathbf{n}$):
  $$\mathbf{D}^1(\mathbf{R}) = \mathbf{R} \in \mathrm{SO}(3)$$
- **$l = 2$ (Traceless Symmetric Matrix, Dim 5)**: Represents directional stress, inertia tensors, and quadrupole moments.

Under rotation $\mathbf{R}(\alpha, \beta, \gamma)$ parameterized by Euler angles, an irrep of degree $l$ transforms via the **Wigner D-Matrix** $\mathbf{D}^l(\mathbf{R}) \in \mathbb{R}^{(2l+1) \times (2l+1)}$:

$$
\mathbf{f}' = \mathbf{D}^l(\mathbf{R}) \mathbf{f}
$$

---

### 2.2 The Steerable Kernel Constraint
A continuous convolution kernel $K: \mathbb{R}^3 \to \mathbb{R}^{(2l_{\text{out}}+1) \times (2l_{\text{in}}+1)}$ mapping from an input irrep $l_{\text{in}}$ to output irrep $l_{\text{out}}$ must satisfy the **Kernel Equivariance Constraint**:

$$
K(\mathbf{R} \mathbf{x}) = \mathbf{D}^{l_{\text{out}}}(\mathbf{R}) K(\mathbf{x}) \mathbf{D}^{l_{\text{in}}}(\mathbf{R})^{-1}, \quad \forall \mathbf{R} \in \mathrm{SO}(3), \, \mathbf{x} \in \mathbb{R}^3
$$

By the Wigner-Eckart theorem, the solution to this constraint factorizes into a **radial profile function** and **Spherical Harmonics**:

$$
K(\mathbf{x}) = \sum_{J = |l_{\text{out}} - l_{\text{in}}|}^{l_{\text{out}} + l_{\text{in}}} R_J(\|\mathbf{x}\|) \sum_{M = -J}^J C_{(l_{\text{in}}, m_1)(l_{\text{out}}, m_2)}^{(J, M)} Y_{JM}\left(\frac{\mathbf{x}}{\|\mathbf{x}\|}\right)
$$

where:
- $R_J(r)$ is a learnable 1D radial basis function (e.g., Gaussian radial basis or Bessel functions).
- $Y_{JM}(\hat{\mathbf{r}})$ is the spherical harmonic of degree $J$ and order $M$.
- $C_{(l_{\text{in}}, m_1)(l_{\text{out}}, m_2)}^{(J, M)}$ are analytical **Clebsch-Gordan coefficients** coupling angular momenta.

---

### 2.3 The Clebsch-Gordan Tensor Product
To combine two equivariant geometric feature vectors $\mathbf{f}_1$ (degree $l_1$) and $\mathbf{f}_2$ (degree $l_2$), the standard elementwise multiplication is invalid because it violates rotational symmetry.
Features must be combined via the **Clebsch-Gordan Tensor Product** $\otimes_{\text{CG}}$:

$$
\left( \mathbf{f}_1 \otimes_{\text{CG}} \mathbf{f}_2 \right)_{(l_3, m_3)} = \sum_{m_1=-l_1}^{l_1} \sum_{m_2=-l_2}^{l_2} C_{(l_1, m_1)(l_2, m_2)}^{(l_3, m_3)} f_{1, (l_1, m_1)} f_{2, (l_2, m_2)}
$$

for all output degrees $l_3 \in \{|l_1 - l_2|, \dots, l_1 + l_2\}$.

```
Clebsch-Gordan Coupling Rules (Geometric Examples):
1. Vector (l=1) (x) Vector (l=1):
   - l_3 = 0 (Scalar, Dim 1): Dot product f_1 . f_2 (Invariance)
   - l_3 = 1 (Vector, Dim 3): Cross product f_1 x f_2 (Orthogonal Vector)
   - l_3 = 2 (Tensor, Dim 5): Traceless symmetric outer product 1/2(f_1*f_2^T + f_2*f_1^T) - 1/3(f_1 . f_2)I
```

---

## 3. Python Reference Implementation

```python
import torch
import torch.nn as nn
import numpy as np

def compute_spherical_harmonics_l1(r_norm: torch.Tensor) -> torch.Tensor:
    """
    Computes real spherical harmonics of degree l=1 for unit direction vectors.
    r_norm: [N, 3] normalized unit vectors [x, y, z]
    Returns: [N, 3] spherical harmonics Y_1m (m = -1, 0, 1)
    """
    x, y, z = r_norm[:, 0], r_norm[:, 1], r_norm[:, 2]
    # Y_1,-1 = sqrt(3/(4*pi)) * y
    # Y_1,0  = sqrt(3/(4*pi)) * z
    # Y_1,1  = sqrt(3/(4*pi)) * x
    c = np.sqrt(3.0 / (4.0 * np.pi))
    return torch.stack([c * y, c * z, c * x], dim=-1)


def clebsch_gordan_vector_dot(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Clebsch-Gordan coupling of two l=1 vectors into an l=0 scalar (Dot Product).
    Guarantees strict SO(3) rotational invariance.
    v1, v2: [N, 3]
    Returns: [N, 1] invariant scalar
    """
    return (v1 * v2).sum(dim=-1, keepdim=True)


def clebsch_gordan_vector_cross(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Clebsch-Gordan coupling of two l=1 vectors into an l=1 vector (Cross Product).
    Guarantees strict SO(3) rotational equivariance: cross(R*v1, R*v2) = R*cross(v1, v2).
    v1, v2: [N, 3]
    Returns: [N, 3] equivariant vector
    """
    return torch.cross(v1, v2, dim=-1)


class SteerableEquivariantLayer(nn.Module):
    """
    Minimal steerable layer processing coupled scalar (l=0) and vector (l=1) features.
    Guarantees exact SO(3) equivariance across 3D point rotations.
    """
    def __init__(self, in_scalars: int, in_vectors: int, out_scalars: int, out_vectors: int):
        super().__init__()
        # Invariant scalar path (Standard Linear)
        self.scalar_linear = nn.Linear(in_scalars + in_vectors, out_scalars)
        
        # Equivariant vector path (Gated linear combination)
        self.vector_gate = nn.Linear(in_scalars, out_vectors)
        self.vector_linear = nn.Linear(in_vectors, out_vectors, bias=False) # Bias must be 0 for equivariance!

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        s: [N, in_scalars] l=0 rotational invariants
        v: [N, in_vectors, 3] l=1 3D vectors
        Returns: (s_out [N, out_scalars], v_out [N, out_vectors, 3])
        """
        # 1. Extract invariant vector norms via Clebsch-Gordan dot product
        v_norms = torch.norm(v, dim=-1) # [N, in_vectors] (l=0 invariant)
        
        # 2. Update scalar channel
        s_combined = torch.cat([s, v_norms], dim=-1)
        s_out = torch.relu(self.scalar_linear(s_combined))
        
        # 3. Update vector channel via gated linear transformation
        # v_out = gate(s) * (W * v)
        # Multiplying vectors by scalar invariants maintains exact equivariance!
        v_transformed = self.vector_linear(v.permute(0, 2, 1)).permute(0, 2, 1) # [N, out_vectors, 3]
        gate = torch.sigmoid(self.vector_gate(s)).unsqueeze(-1) # [N, out_vectors, 1]
        v_out = gate * v_transformed
        
        return s_out, v_out
```

---

## 4. Models in the Vault Utilizing Equivariant Mechanics

- **[[architectures/pose-and-robotics-manipulation/foundationpose|FoundationPose]]**: 6-DoF object pose estimation utilizing $\mathrm{SO}(3)$ invariant descriptor spaces for zero-shot tracking.
- **[[architectures/multimodal-vlm-and-vla/anygrasp|AnyGrasp]]**: 6-DoF robotic grasp detection equivariant to arm approach angles.
- **[[techniques/lie-algebra-se3-pose-tracking|Lie Algebra se(3) Pose Tracking]]**: Continuous manifold parameterization of 3D rigid body transformations.
- **[[topics/6dof-pose-estimation/README|6-DoF Pose Estimation Playbook]]**: Cataloged as the primary mathematical framework for rotation-robust spatial AI.

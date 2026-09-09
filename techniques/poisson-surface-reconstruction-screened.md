---
title: "Poisson Surface Reconstruction: Screened Implicit Watermark Meshing"
type: "Technique"
domain: "3D Point Clouds, Spatial Radiance & Surface Meshing"
tags:
  - technique
  - poisson-reconstruction
  - screened-poisson
  - point-cloud-meshing
  - indicator-function
  - watertight-mesh
  - 3d-geometry
status: evergreen
updated: 2026-09-09
aliases:
  - "Poisson Surface Reconstruction"
  - "Screened Poisson"
  - "Indicator Function Meshing"
  - "Watertight Surface Reconstruction"
  - "Point Cloud to Mesh"
---

# 🗿 Poisson Surface Reconstruction: Screened Implicit Watermark Meshing

## 1. High-Level Concept & The Unorganized Point Cloud Challenge

Raw 3D point clouds captured by LiDAR, Structured Light, and Multi-View Stereo (COLMAP, 3D Gaussian Splatting) are:
- **Discrete & Unorganized**: Arbitrary collections of XYZ coordinates lacking topological connectivity.
- **Noisy & Incomplete**: Vulnerable to occlusion shadows, sensor dropouts, and non-uniform spatial sampling densities.

Downstream robotic grasping (AnyGrasp), aerodynamic CFD simulations, physical collision checking, and 3D printing demand **watertight (closed), 2-manifold, continuous triangle meshes** bounding solid 3D volumes.

### Why Local Surface Meshing Fails
Traditional local meshing algorithms (e.g., Delaunay triangulation, Ball Pivoting Algorithm / BPA):
- Interpolate points locally by rolling a virtual sphere over nearby points.
- When point density drops or noise exceeds sphere radius, BPA creates large artificial holes, disconnected fragments, and non-manifold self-intersecting topology.

---

### The Poisson Reconstruction Paradigm (Kazhdan et al., 2006)
Poisson Surface Reconstruction reformulates surface extraction as a **global spatial PDE boundary-value problem**:

1. **The Indicator Function**: Define an implicit 3D indicator function $\chi: \mathbb{R}^3 \to \mathbb{R}$ representing a solid volume $M$:
   $$\chi(\mathbf{x}) = \begin{cases} 1 & \text{if } \mathbf{x} \in \text{Interior of Object } M \\ 0 & \text{if } \mathbf{x} \in \text{Exterior Space} \end{cases}$$
2. **Gradient as Surface Dirac Delta**: By the Divergence Theorem, the gradient of the indicator function $\nabla \chi$ is zero everywhere except directly on the object's boundary surface $\partial M$, where it acts as a smoothed vector Dirac delta pointing inward along the surface normals:
   $$\nabla \chi(\mathbf{x}) \approx -\mathbf{V}(\mathbf{x})$$
   where $\mathbf{V}(\mathbf{x})$ is a continuous vector field formed by interpolating input sample point normals $\mathcal{P} = \{(\mathbf{p}_i, \mathbf{n}_i)\}$.
3. **The Spatial Poisson Equation**: Taking the divergence ($\nabla \cdot$) of both sides yields the Poisson equation:
   $$\Delta \chi \equiv \nabla \cdot \nabla \chi = \nabla \cdot \mathbf{V}$$
4. **The Screened Poisson Formulation (Kazhdan & Hoppe, 2013)**:
   Classical Poisson smoothed across fine details and could drift away from measured point positions.
   **Screened Poisson** introduces an explicit point-fidelity penalty:
   $$\min_{\chi} \int_{\Omega} \|\nabla \chi(\mathbf{x}) - \mathbf{V}(\mathbf{x})\|^2 \, d\mathbf{x} + \alpha \sum_{i} \left( \chi(\mathbf{p}_i) - 0.5 \right)^2$$
   The screening parameter $\alpha$ anchors the reconstructed $0.5$ iso-surface directly to the measured 3D point positions!
5. **Iso-Surface Extraction**: The watertight polygonal surface is extracted at iso-value $\chi(\mathbf{x}) = 0.5$ using an Octree-based Marching Cubes or Dual Contouring solver.

```
Screened Poisson Surface Reconstruction Pipeline:

Oriented Point Cloud {p_i, n_i} (Noisy, variable density)
               |
               v
[ 1. Spatial Vector Field Construction V(x) ]
     (Convolve point normals with Gaussian smoothing kernel)
               |
               v
[ 2. Octree Spatial Discretization (Depth d = 8 to 12) ]
     (Subdivide domain adaptively: high resolution near points, coarse elsewhere)
               |
               v
[ 3. Screened Poisson PDE Solve ]
     Solve: (Laplacian + alpha * Screen) * chi = div(V)
     (Sparse symmetric positive-definite linear system)
               |
               v Continuous Scalar Indicator Field chi(x) in [0, 1]
[ 4. Iso-Surface Polygonization at chi(x) = 0.5 (Marching Cubes) ]
               |
               v
Watertight, 2-Manifold Triangle Mesh (No holes, guaranteed manifold)
```

---

## 2. Mathematical Formulation & Variational Derivation

### 2.1 Vector Field Integration
Let $\mathcal{P} = \{(\mathbf{p}_1, \mathbf{n}_1), \dots, (\mathbf{p}_N, \mathbf{n}_N)\}$ be the input set of 3D points and unit-length outward normals.
Define the smoothed normal vector field $\mathbf{V}(\mathbf{x})$:

$$
\mathbf{V}(\mathbf{x}) = \sum_{i=1}^N \mathbf{n}_i \cdot K_{\sigma}(\mathbf{x} - \mathbf{p}_i)
$$

where $K_{\sigma}(\mathbf{x}) = \frac{1}{(2\pi \sigma^2)^{3/2}} \exp\left(-\frac{\|\mathbf{x}\|^2}{2\sigma^2}\right)$ is a normalized 3D Gaussian smoothing filter.

---

### 2.2 Screened Variational Energy Minimization
Discretize the volume $\Omega$ using an adaptive Octree with finite element basis functions $\{B_j(\mathbf{x})\}$.
Represent the indicator function as a linear combination of nodal basis weights:

$$
\chi(\mathbf{x}) = \sum_{j=1}^M c_j B_j(\mathbf{x})
$$

The variational screening energy functional is:

$$
E(\mathbf{c}) = \int_{\Omega} \left\| \sum_{j=1}^M c_j \nabla B_j(\mathbf{x}) - \mathbf{V}(\mathbf{x}) \right\|^2 \, d\mathbf{x} + \alpha \sum_{i=1}^N \left( \sum_{j=1}^M c_j B_j(\mathbf{p}_i) - 0.5 \right)^2
$$

Setting the gradient $\nabla_{\mathbf{c}} E(\mathbf{c}) = \mathbf{0}$ yields the linear system:

$$
(\mathbf{L} + \alpha \mathbf{W}) \mathbf{c} = \mathbf{b} + 0.5 \alpha \mathbf{w}
$$

where:
- $\mathbf{L}_{jk} = \int_{\Omega} \langle \nabla B_j(\mathbf{x}), \nabla B_k(\mathbf{x}) \rangle \, d\mathbf{x}$ is the discrete symmetric positive-definite stiffness Laplacian matrix.
- $\mathbf{W}_{jk} = \sum_{i=1}^N B_j(\mathbf{p}_i) B_k(\mathbf{p}_i)$ is the point interpolation mass matrix.
- $\mathbf{b}_j = \int_{\Omega} \langle \nabla B_j(\mathbf{x}), \mathbf{V}(\mathbf{x}) \rangle \, d\mathbf{x} = -\int_{\Omega} B_j(\mathbf{x}) (\nabla \cdot \mathbf{V}(\mathbf{x})) \, d\mathbf{x}$.

Because $(\mathbf{L} + \alpha \mathbf{W})$ is sparse and symmetric positive-definite, it is solved using preconditioned Conjugate Gradient (PCG) methods.

---

## 3. Python Reference Implementation

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class ScreenedPoissonReconstructor1D:
    """
    1D conceptual demonstration of Screened Poisson surface reconstruction.
    Solves for an indicator function chi(x) whose gradient matches normal vectors
    while anchoring values at sample points to 0.5.
    """
    def __init__(self, num_nodes: int = 100, alpha: float = 10.0):
        self.M = num_nodes
        self.alpha = alpha
        self.dx = 1.0 / (num_nodes - 1)

    def solve(self, sample_points: np.ndarray, sample_normals: np.ndarray) -> np.ndarray:
        """
        sample_points: [N] normalized positions in [0, 1]
        sample_normals: [N] surface gradients (+1 or -1)
        Returns: reconstructed indicator function chi [M]
        """
        M = self.M
        dx = self.dx

        # 1. Assemble 1D Finite Difference Laplacian (Stiffness Matrix L)
        main_diag = 2.0 * np.ones(M) / (dx ** 2)
        off_diag = -1.0 * np.ones(M - 1) / (dx ** 2)
        L = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(M, M), format='lil')
        # Dirichlet boundary conditions
        L[0, :] = 0; L[0, 0] = 1.0
        L[-1, :] = 0; L[-1, -1] = 1.0
        L = L.tocsr()

        # 2. Build divergence vector b = div(V)
        b = np.zeros(M)
        for pt, norm in zip(sample_points, sample_normals):
            idx = int(np.clip(pt / dx, 0, M - 1))
            b[idx] += norm / dx

        # 3. Assemble Screening Matrix W and target vector w_rhs
        W = sp.lil_matrix((M, M))
        w_rhs = np.zeros(M)
        for pt in sample_points:
            idx = int(np.clip(pt / dx, 0, M - 1))
            W[idx, idx] += 1.0
            w_rhs[idx] += 0.5

        W = W.tocsr()

        # 4. Form Screened Linear System: (L + alpha * W) * chi = b + alpha * w_rhs
        A_sys = L + self.alpha * W
        rhs = b + self.alpha * w_rhs

        # Solve sparse linear system via conjugate gradient / spsolve
        chi = spla.spsolve(A_sys, rhs)
        return chi
```

---

## 4. Models in the Vault Utilizing Poisson Reconstruction

- **[[techniques/marching-cubes-and-signed-distance-functions|Marching Cubes & Signed Distance Functions]]**: Iso-surface extraction polygonizing the Poisson indicator field $\chi = 0.5$.
- **[[architectures/spatial-radiance-and-slam/3dgs-slam|3DGS-SLAM]]**: High-fidelity mesh extraction from densified 3D Gaussian point centers.
- **[[architectures/pose-and-robotics-manipulation/foundationpose|FoundationPose]]**: CAD object model mesh reconstruction from unorganized multi-view depth point clouds.
- **[[topics/lidar-perception/README|LiDAR Perception Playbook]]**: Cataloged as the primary watertight environmental meshing pipeline.

---
title: "Multiresolution Spatial Hash Encodings & Dynamic Sparse Voxelization"
type: "Technique"
domain: "3D Spatial Perception, Radiance Fields & LiDAR Processing"
tags:
  - technique
  - hash-encoding
  - instant-ngp
  - sparse-voxelization
  - dsvt
  - pointclouds
  - lidar
  - spatial-computing
status: evergreen
updated: 2026-09-09
aliases:
  - "Multiresolution Hash Encoding"
  - "Spatial Hashing"
  - "Dynamic Sparse Voxelization"
  - "Instant-NGP Hashing"
---

# 🧊 Multiresolution Spatial Hash Encodings & Dynamic Sparse Voxelization

## 1. High-Level Concept & The Cubic 3D Memory Explosion

Physical 3D perception—including Neural Radiance Fields (NeRF), LiDAR point cloud perception, and metric volumetric mapping—operates over continuous spatial coordinates $\mathbf{x} = (x, y, z) \in \mathbb{R}^3$.

### The Cubic Scaling Barrier
Representing 3D geometry using uniform dense voxel grids suffers from cubic memory scaling $\mathcal{O}(N^3)$:
- A modest $1024 \times 1024 \times 1024$ voxel volume requires over **1.07 billion grid cells**. Storing an 8-dimensional feature per cell consumes $>34\text{ GB}$ of VRAM!
- Yet in real-world robotics and autonomous driving, **over 99% of 3D space is empty air**. Storing dense tensors wastes memory bandwidth and throttles compute.

On the other hand, purely continuous coordinate networks (e.g., standard NeRF MLPs) avoid grid storage by querying a large MLP at every 3D coordinate. However, because deep MLPs suffer from **spectral bias** (failing to represent high frequencies), they require dozens of sinusoidal frequency projections ($\sin(2^k \pi x)$) and deep networks, limiting training and inference to $<1\text{ FPS}$.

### The Multiresolution Spatial Hash Solution
**Multiresolution Hash Encoding** (Instant-NGP, Müller et al., SIGGRAPH 2022) and **Dynamic Sparse Voxelization** (DSVT / FlatFormer):
1. **Multiresolution Spatial Hashing**: Discretizes 3D space into $L$ independent resolution pyramids. Instead of allocating dense grid pointers, maps corner grid vertices to fixed-size hash tables of size $T \ll 2^{3d}$ using a spatial prime-xor hash function.
2. **Compact d-linear interpolation**: Samples 8 corner hash entries and performs trilinear interpolation, producing expressive continuous representations.
3. **Collision Disambiguation via Multi-Level Stacking**: While spatial hash functions induce hash collisions, stacking $L$ different geometric resolutions naturally resolves ambiguities—if two distinct physical coordinates collide at resolution level $l$, they map to completely different hash slots at levels $l-1$ and $l+1$.
4. **Dynamic Sparse Voxelization (DSVT)**: In LiDAR processing, non-empty voxels are indexed dynamically and grouped into fixed-size equal-work windows via parallel coordinate sorting, eliminating zero-padding while maintaining $100\%$ GPU warp occupancy.

```
Dense Voxel Grid vs. Multiresolution Spatial Hash Table:

Dense Voxel Grid (O(N^3) Memory Explosion):
  [ 1024 x 1024 x 1024 Grid ] ---> 1,073,741,824 Cells (34 GB VRAM!)
  99% of memory wasted on empty air.

Multiresolution Hash Table (O(T) Constant Memory Footprint):
  Coordinate (x, y, z)
       |
       +---> Level 1 (Coarse): Hash Table 1 (Size T = 2^19) ---> Trilinear Interp \
       +---> Level 2 (Mid):    Hash Table 2 (Size T = 2^19) ---> Trilinear Interp ---> Concat -> Output (Few MBs!)
       +---> Level L (Fine):   Hash Table L (Size T = 2^19) ---> Trilinear Interp /
```

---

## 2. Mathematical Formulation

### 2.1 Spatial Prime-XOR Hash Function
For a continuous 3D coordinate $\mathbf{x} \in \mathbb{R}^3$, the integer grid coordinates at resolution level $l$ with grid spacing $\Delta_l$ are:

$$
\mathbf{v} = \lfloor \mathbf{x} / \Delta_l \rfloor = (v_x, v_y, v_z) \in \mathbb{Z}^3
$$

When the number of grid points exceeds the allocated hash table size $T$ (typically $T = 2^{14}\text{ to }2^{22}$ entries), vertices are indexed using the spatial prime-xor hash function:

$$
h(\mathbf{v}) = \left( \bigoplus_{i=1}^3 v_i \cdot \pi_i \right) \pmod T
$$

where:
- $\bigoplus$ is the bitwise XOR operator.
- $\pi_1 = 1$, $\pi_2 = 2\,654\,435\,761$ ($2^{32} \cdot \frac{\sqrt{5}-1}{2}$, Knuth's Golden Ratio prime), and $\pi_3 = 805\,459\,861$ are large distinct prime numbers.

If the number of grid vertices $(N_l + 1)^3 \le T$, the mapping is exact and $1:1$ (no hash collisions occur). When $(N_l + 1)^3 > T$, the prime-xor function uniformly pseudo-randomizes collisions across the table.

---

### 2.2 Multiresolution Geometric Growth
The resolution levels $l \in [0, L-1]$ are distributed geometrically between the coarsest resolution $N_{\min}$ (e.g., $16$) and finest resolution $N_{\max}$ (e.g., $2048$ to $4096$):

$$
N_l = \left\lfloor N_{\min} \cdot b^l \right\rfloor
$$

where the growth factor $b$ is:

$$
b = \exp\left( \frac{\ln N_{\max} - \ln N_{\min}}{L - 1} \right)
$$

---

### 2.3 Trilinear Feature Interpolation
For a query point $\mathbf{x}$, let $\mathbf{v} = \lfloor \mathbf{x} / \Delta_l \rfloor$ be the floor grid corner, and let $\mathbf{w} = (\mathbf{x} / \Delta_l) - \mathbf{v} \in [0, 1)^3$ be the relative local intra-voxel offsets.

The interpolated feature vector $\mathbf{f}_l(\mathbf{x}) \in \mathbb{R}^F$ (typically $F = 2\text{ to }8$ channels per level) is computed by trilinearly blending the $2^3 = 8$ corner hash table entries:

$$
\mathbf{f}_l(\mathbf{x}) = \sum_{c_x \in \{0, 1\}} \sum_{c_y \in \{0, 1\}} \sum_{c_z \in \{0, 1\}} w_x(c_x) w_y(c_y) w_z(c_z) \cdot \mathbf{\Theta}_l\left[ h(v_x + c_x, v_y + c_y, v_z + c_z) \right]
$$

where:
- $w_i(1) = w_i$ and $w_i(0) = 1 - w_i$.
- $\mathbf{\Theta}_l \in \mathbb{R}^{T \times F}$ is the learnable feature lookup table for level $l$.

The final multi-scale representation is formed by concatenating all $L$ level features:

$$
\mathbf{y}(\mathbf{x}) = \left[ \mathbf{f}_0(\mathbf{x}), \; \mathbf{f}_1(\mathbf{x}), \; \dots, \; \mathbf{f}_{L-1}(\mathbf{x}) \right] \in \mathbb{R}^{L \cdot F}
$$

---

### 2.4 Analytical Gradient and Collision Disambiguation
During backpropagation, the gradient with respect to hash table entry $\mathbf{\Theta}_l[k]$ is the sum of weighted gradients from all query points $\mathbf{x}_m$ whose bounding cube vertices mapped to slot $k$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{\Theta}_l[k]} = \sum_{m} \sum_{\mathbf{c} \in \{0, 1\}^3} \mathbf{1}_{\{h(\mathbf{v}_m + \mathbf{c}) = k\}} \cdot \left( \prod_{i=1}^3 w_{m, i}(c_i) \right) \frac{\partial \mathcal{L}}{\partial \mathbf{f}_l(\mathbf{x}_m)}
$$

Because the downstream loss drives $\mathbf{y}(\mathbf{x})$ to explain the true observations, the dominant geometric signal automatically overrides the unstructured pseudo-random collision noise across different resolution levels!

---

## 3. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
import math

class HashEncoding3D(nn.Module):
    """
    Multiresolution Spatial Hash Encoding for 3D continuous coordinates.
    Computes prime-xor spatial hashing with continuous trilinear interpolation.
    """
    def __init__(self, num_levels: int = 8, features_per_level: int = 4,
                 log2_hashmap_size: int = 16, base_resolution: int = 16,
                 max_resolution: int = 512):
        super().__init__()
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size
        
        # Calculate geometric growth factor b
        self.b = math.exp((math.log(max_resolution) - math.log(base_resolution)) / (num_levels - 1))
        self.base_resolution = base_resolution
        
        # Spatial prime constants
        self.primes = [1, 2654435761, 805459861]
        
        # Learnable hash tables: [L, T, F]
        self.embeddings = nn.Parameter(
            torch.empty(num_levels, self.hashmap_size, features_per_level)
        )
        nn.init.uniform_(self.embeddings, -1e-4, 1e-4)

    def hash_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Computes prime-xor hash: coords is [..., 3] integer grid."""
        # Bitwise XOR across dimensions
        h = (coords[..., 0] * self.primes[0]) ^ \
            (coords[..., 1] * self.primes[1]) ^ \
            (coords[..., 2] * self.primes[2])
        return torch.remainder(h, self.hashmap_size).long()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3] continuous coordinates in [0, 1]
        Returns: [B, num_levels * features_per_level] multi-scale hash features
        """
        b_size = x.shape[0]
        out_levels = []
        
        for lvl in range(self.num_levels):
            # 1. Grid resolution at this level
            res = math.floor(self.base_resolution * (self.b ** lvl))
            scaled_x = x * res
            
            # 2. Corner integer coordinates and local offsets
            v0 = torch.floor(scaled_x).long()
            w = scaled_x - v0.float()  # [B, 3] in [0, 1)
            
            # 8 corners of the bounding cube
            c000 = self.hash_coords(v0)
            c100 = self.hash_coords(v0 + torch.tensor([1, 0, 0], device=x.device))
            c010 = self.hash_coords(v0 + torch.tensor([0, 1, 0], device=x.device))
            c110 = self.hash_coords(v0 + torch.tensor([1, 1, 0], device=x.device))
            c001 = self.hash_coords(v0 + torch.tensor([0, 0, 1], device=x.device))
            c101 = self.hash_coords(v0 + torch.tensor([1, 0, 1], device=x.device))
            c011 = self.hash_coords(v0 + torch.tensor([0, 1, 1], device=x.device))
            c111 = self.hash_coords(v0 + torch.tensor([1, 1, 1], device=x.device))
            
            # Lookup features: [B, F]
            emb = self.embeddings[lvl]
            f000 = emb[c000]
            f100 = emb[c100]
            f010 = emb[c010]
            f110 = emb[c110]
            f001 = emb[c001]
            f101 = emb[c101]
            f011 = emb[c011]
            f111 = emb[c111]
            
            # 3. Trilinear interpolation
            wx = w[:, 0:1]
            wy = w[:, 1:2]
            wz = w[:, 2:3]
            
            c00 = f000 * (1 - wx) + f100 * wx
            c10 = f010 * (1 - wx) + f110 * wx
            c01 = f001 * (1 - wx) + f101 * wx
            c11 = f011 * (1 - wx) + f111 * wx
            
            c0 = c00 * (1 - wy) + c10 * wy
            c1 = c01 * (1 - wy) + c11 * wy
            
            feat_lvl = c0 * (1 - wz) + c1 * wz
            out_levels.append(feat_lvl)
            
        return torch.cat(out_levels, dim=-1)
```

---

## 4. Models in the Vault Utilizing Sparse Voxels & Hashing

- **[[architectures/spatial-radiance-and-slam/instant-ngp|Instant-NGP]]**: Seminal multiresolution spatial hash encoding framework rendering gigapixel scenes in real-time.
- **[[architectures/3d-pointclouds-and-lidar/dsvt|DSVT: Dynamic Sparse Voxel Transformer]]**: Eliminates zero-padding in LiDAR processing via dynamic sparse window voxelization.
- **[[architectures/3d-pointclouds-and-lidar/flatformer|FlatFormer]]**: Equal-work grouping transformer sorting non-empty 3D voxels to maximize GPU warp occupancy.
- **[[architectures/3d-pointclouds-and-lidar/pointpillars|PointPillars]]**: Fast pillar-based 2D projection of sparse 3D point cloud data.

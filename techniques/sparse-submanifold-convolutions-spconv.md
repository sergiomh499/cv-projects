---
title: "3D Sparse & Submanifold Convolutions (SpConv): Efficient Non-Euclidean Tensor Compute"
type: "Technique"
domain: "3D Point Clouds, LiDAR Perception & Voxel Architectures"
tags:
  - technique
  - spconv
  - sparse-convolution
  - submanifold-convolution
  - 3d-pointclouds
  - lidar
  - autonomous-driving
status: evergreen
updated: 2026-09-09
aliases:
  - "SpConv"
  - "Submanifold Sparse Convolution"
  - "SSC"
  - "Sparse Convolutions"
  - "Minkowski Engine"
---

# 🧊 3D Sparse & Submanifold Convolutions (SpConv): Non-Euclidean Tensor Compute

## 1. High-Level Concept & The 3D Voxel Dilation Dilemma

In 3D autonomous driving, robotics, and spatial perception (e.g., [[architectures/3d-pointclouds-and-lidar/centerpoint|CenterPoint]], [[architectures/3d-pointclouds-and-lidar/pv-rcnn|PV-RCNN]], [[architectures/3d-pointclouds-and-lidar/dsvt|DSVT]], [[architectures/3d-pointclouds-and-lidar/bevfusion|BEVFusion]]), continuous 3D scenes are discretized into fine 3D cubic voxel grids (e.g., $1024 \times 1024 \times 40$ voxels).

### The Extreme Sparsity of Physical Space
Outdoor LiDAR scans and depth sensor measurements are **$>99\%$ empty space**:
- Out of 40 million candidate voxels in an automotive perception volume, typically fewer than $50,000$ to $100,000$ contain valid point reflections ($<0.2\%$ occupancy).
- Representing this space as a dense 3D tensor requires tens of gigabytes of GPU VRAM and wastes trillions of FLOPs convolving over empty air.

### The Catastrophic Voxel Dilation Barrier
When standard sparse convolutions are applied to 3D point clouds:
1. **Regular Sparse Convolution**: A kernel of size $K = 3 \times 3 \times 3$ has 27 spatial weights. When centered on an active voxel, it produces non-zero outputs for all 27 neighboring positions, even if those neighbors were initially empty air.
2. **The Dilation Explosion**: In a deep neural network with 10 to 20 convolutional layers, active voxels spread outward exponentially like an infectious wave. By layer 5, the entire 3D tensor becomes fully dense, obliterating the original spatial manifold and crashing GPU memory!

### The Submanifold Sparse Convolution (SSC) Solution (Graham et al., CVPR 2018)
**Submanifold Sparse Convolution** establishes a strict geometric invariant:
- **An output site $\mathbf{p}$ is evaluated if and only if the central input site was already active**:
  $$\text{Active}(\mathbf{y}_{\mathbf{p}}) = 1 \iff \text{Active}(\mathbf{x}_{\mathbf{p}}) = 1$$
- Sparse manifolds (surfaces of cars, road surfaces, pedestrians) remain strictly sparse throughout arbitrarily deep networks.
- **Rulebook Generation & Gather-GEMM-Scatter**:
  Active voxels are stored as a coordinate index table $\mathbf{C} \in \mathbb{N}^{N \times 4}$ (batch, $x, y, z$) and feature tensor $\mathbf{F} \in \mathbb{R}^{N \times C_{\text{in}}}$.
  Before execution, a **Rulebook** pre-computes the active input-output correspondence pairs for each of the $K^3 = 27$ kernel offsets.
  The operation executes via high-throughput **cuBLAS GEMM** (Gather $\to$ GEMM $\to$ Scatter), delivering a **$50\times$ speedup** and **$95\%$ VRAM reduction** over dense 3D convolutions.

```
Regular Sparse Conv vs. Submanifold Sparse Conv (SSC):

Input Voxels (3 active points)
   . . . . . . . . . .
   . . * . . . . * . .
   . . . . * . . . . .
   . . . . . . . . . .

Regular Sparse Conv (3x3 Kernel):       Submanifold Sparse Conv (SSC):
(Active sites dilate outward!)          (Preserves exact input sparsity!)
   . * * * . . * * * .                     . . . . . . . . . .
   . * * * * * * * * .                     . . * . . . . * . .
   . * * * * * * * * .                     . . . . * . . . . .
   . . . * * * . . . .                     . . . . . . . . . .
(Memory explodes across layers!)        (Constant memory in arbitrarily deep nets!)
```

---

## 2. Mathematical Formulation

### 2.1 Coordinate Hashing & Sparse Tensor Representation
A sparse tensor is defined by two tensors:
1. **Coordinate Table** $\mathbf{C} \in \mathbb{Z}^{N \times 4}$: Contains the integer grid indices $[b, x, y, z]$ for all $N$ non-empty active voxels.
2. **Feature Matrix** $\mathbf{F} \in \mathbb{R}^{N \times C_{\text{in}}}$: Stores the corresponding feature representations.

To query whether a coordinate $\mathbf{p} = [b, x, y, z]$ is active in $\mathcal{O}(1)$ time, coordinates are mapped using a spatial hash table with MurmurHash or prime-multiplication hashing:

$$
h(b, x, y, z) = \left( b \cdot p_1 \oplus x \cdot p_2 \oplus y \cdot p_3 \oplus z \cdot p_4 \right) \pmod T
$$

---

### 2.2 The Submanifold Convolution Equation
Let $\mathcal{K} = \{-1, 0, 1\}^3$ be the set of 27 spatial offset vectors for a $3 \times 3 \times 3$ kernel.
For each active output voxel coordinate $\mathbf{p}_{\text{out}} \in \mathbf{C}_{\text{in}}$:

$$
\mathbf{y}_{\mathbf{p}_{\text{out}}} = \sum_{\mathbf{k} \in \mathcal{K}} \mathbf{W}_{\mathbf{k}} \mathbf{x}_{\mathbf{p}_{\text{out}} + \mathbf{k}} \cdot \mathbb{I}\left( \mathbf{p}_{\text{out}} + \mathbf{k} \in \mathbf{C}_{\text{in}} \right)
$$

where:
- $\mathbf{W}_{\mathbf{k}} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}}}$ is the convolution weight slice corresponding to spatial offset $\mathbf{k}$.
- $\mathbb{I}(\cdot)$ is the indicator function confirming that the input neighbor at $\mathbf{p}_{\text{out}} + \mathbf{k}$ is an active site.
- If $\mathbf{p}_{\text{out}} + \mathbf{k}$ is empty space, it contributes zero without computation or memory allocation.

---

### 2.3 The Rulebook & Gather-GEMM-Scatter Execution
Evaluating the summation directly with scattered memory access results in severe GPU memory latency. SpConv decouples the computation into:

1. **Rulebook Generation**: A 2D integer array $\mathbf{R}_{\mathbf{k}} \in \mathbb{Z}^{M_{\mathbf{k}} \times 2}$ is constructed for each kernel offset $\mathbf{k} \in \mathcal{K}$. Each row contains $(i_{\text{in}}, i_{\text{out}})$, indicating that the $i_{\text{in}}$-th active input feature connects to the $i_{\text{out}}$-th output feature through kernel weight $\mathbf{W}_{\mathbf{k}}$.
2. **Gather**: Contiguous feature vectors are gathered into a matrix $\mathbf{A}_{\mathbf{k}} \in \mathbb{R}^{M_{\mathbf{k}} \times C_{\text{in}}}$ using indices $i_{\text{in}}$.
3. **GEMM (General Matrix Multiply)**: Evaluated at peak hardware Tensor Core throughput:
   $$\mathbf{B}_{\mathbf{k}} = \mathbf{A}_{\mathbf{k}} \mathbf{W}_{\mathbf{k}}^T \in \mathbb{R}^{M_{\mathbf{k}} \times C_{\text{out}}}$$
4. **Scatter-Add**: Output features are accumulated into the output buffer:
   $$\mathbf{Y}[i_{\text{out}}, :] \mathrel{+}= \mathbf{B}_{\mathbf{k}}[m, :]$$

---

## 3. Python Reference Implementation

```python
import torch
import torch.nn as nn

class SimpleSubmanifoldSparseConv(nn.Module):
    """
    Submanifold Sparse Convolution (SSC) Reference Implementation.
    Preserves exact input sparsity across 3D coordinates.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # 27 spatial offsets for 3x3x3 kernel
        self.offsets = []
        r = kernel_size // 2
        for dz in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    self.offsets.append((dx, dy, dz))
        self.num_offsets = len(self.offsets) # 27
        
        # Weights: [27, out_channels, in_channels]
        self.weights = nn.Parameter(
            torch.randn(self.num_offsets, out_channels, in_channels) * 0.05
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def build_rulebook(self, coords: torch.Tensor):
        """
        Builds input-to-output index mapping for each kernel offset.
        coords: [N, 3] integer 3D coordinates (x, y, z)
        """
        n = coords.shape[0]
        device = coords.device
        # Fast hash table mapping coordinate tuple to linear index
        coord_map = {tuple(coords[i].tolist()): i for i in range(n)}
        
        rulebook = {}
        for k_idx, (dx, dy, dz) in enumerate(self.offsets):
            in_indices = []
            out_indices = []
            for out_idx in range(n):
                target = (coords[out_idx, 0].item() + dx,
                          coords[out_idx, 1].item() + dy,
                          coords[out_idx, 2].item() + dz)
                if target in coord_map:
                    in_indices.append(coord_map[target])
                    out_indices.append(out_idx)
                    
            if in_indices:
                rulebook[k_idx] = (
                    torch.tensor(in_indices, dtype=torch.long, device=device),
                    torch.tensor(out_indices, dtype=torch.long, device=device)
                )
        return rulebook

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        features: [N, in_channels]
        coords: [N, 3] integer coordinates
        Returns: [N, out_channels] preserving identical N active sites!
        """
        n = features.shape[0]
        rulebook = self.build_rulebook(coords)
        out_features = torch.zeros((n, self.out_channels), 
                                   device=features.device, dtype=features.dtype)
        
        # Gather-GEMM-Scatter for each offset
        for k_idx, (in_idx, out_idx) in rulebook.items():
            # 1. Gather
            gathered_x = features[in_idx] # [M, in_channels]
            # 2. GEMM: gathered_x @ W_k^T
            w_k = self.weights[k_idx] # [out_channels, in_channels]
            gemm_out = torch.matmul(gathered_x, w_k.t()) # [M, out_channels]
            # 3. Scatter-Add
            out_features.index_add_(0, out_idx, gemm_out)
            
        return out_features + self.bias
```

---

## 4. Models in the Vault Utilizing SpConv

- **[[architectures/3d-pointclouds-and-lidar/centerpoint|CenterPoint]]**: Standard 3D LiDAR backbone using 3D sparse submanifold residual blocks.
- **[[architectures/3d-pointclouds-and-lidar/pv-rcnn|PV-RCNN]]**: Point-Voxel feature fusion backbone powered by submanifold convolutions.
- **[[architectures/3d-pointclouds-and-lidar/bevfusion|BEVFusion]]**: Sparse LiDAR encoder transforming 3D point clouds into high-resolution BEV representations.
- **[[architectures/3d-pointclouds-and-lidar/dsvt|DSVT]] & [[architectures/3d-pointclouds-and-lidar/flatformer|FlatFormer]]**: Hybrid sparse convolution and voxel window transformers.

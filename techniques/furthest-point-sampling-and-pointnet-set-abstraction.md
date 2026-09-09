---
title: "PointNet++ Hierarchical Set Abstraction: FPS & Ball Query"
type: "Technique"
domain: "3D Point Clouds, LiDAR Perception & Geometric Deep Learning"
tags:
  - technique
  - pointnet-plus-plus
  - furthest-point-sampling
  - ball-query
  - set-abstraction
  - point-clouds
  - lidar
status: evergreen
updated: 2026-09-09
aliases:
  - "PointNet++"
  - "Set Abstraction"
  - "Furthest Point Sampling"
  - "FPS"
  - "Ball Query"
  - "Point Cloud Grouping"
---

# 🌐 PointNet++ Hierarchical Set Abstraction: FPS & Ball Query

## 1. High-Level Concept & The Irregular 3D Geometric Challenge

In standard 2D image processing, convolutional neural networks exploit the strict, ordered pixel grid structure ($\mathbb{R}^{H \times W \times C}$) using localized sliding kernels (e.g., $3 \times 3$).
In contrast, 3D LiDAR sensors and RGB-D cameras produce **unordered, unstructured point sets**:

$$
\mathcal{P} = \{ \mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_N \} \subset \mathbb{R}^3
$$

A point cloud is fundamentally **permutation-invariant**: swapping the order of any two points in the array does not change the physical geometric object.

---

### The Fundamental Flaw of Original PointNet (Qi et al., CVPR 2017)
The original PointNet applied point-wise multi-layer perceptrons (MLPs) followed by a single global symmetric pooling operator:

$$
f(\mathcal{P}) = \gamma\left( \max_{i=1 \dots N} \{ h(\mathbf{p}_i) \} \right)
$$

While mathematically permutation-invariant, **PointNet lacks local context**:
- It learns either isolated per-point features or one global scene-wide descriptor.
- It cannot capture **local geometric structures** (fine curvature variations, surface normals, sharp edges) across varying spatial scales—a capability that CNNs achieve naturally through pooling and progressive receptive field expansion.

---

### The PointNet++ Set Abstraction Revolution (Qi et al., NeurIPS 2017)
**PointNet++** introduces a hierarchical architecture that constructs local spatial neighborhoods across multiple metric scales through three sequential operations:

#### 1. Sampling Layer: Furthest Point Sampling (FPS)
Selects a subset of $N'$ points $\{ \mathbf{c}_1, \dots, \mathbf{c}_{N'} \}$ from the original $N$ points such that each newly selected centroid is maximally distant from all previously selected centroids. This guarantees **uniform surface coverage** over the entire 3D geometry regardless of sensor beam sparsity.

#### 2. Grouping Layer: Metric Ball Query
Given each centroid $\mathbf{c}_k$, finds all neighbor points within a fixed physical metric radius $r$:

$$
\mathcal{N}(\mathbf{c}_k) = \{ \mathbf{p}_i \in \mathcal{P} \mid \|\mathbf{p}_i - \mathbf{c}_k\|_2 \le r \}
$$

**Critical Translation Invariance**: All neighbor coordinates are expressed relative to the centroid:
$$\mathbf{p}_i' = \mathbf{p}_i - \mathbf{c}_k$$
This ensures the learned local representations are invariant to global spatial shifts.

#### 3. PointNet Layer: Local Feature Aggregation
Applies shared mini-MLPs followed by channel-wise max-pooling over each neighborhood to produce a unified feature vector for centroid $\mathbf{c}_k$.

```
PointNet++ Hierarchical Set Abstraction Pipeline:

Raw Point Cloud P (N points in R^3)
              |
              v
[ 1. Furthest Point Sampling (FPS) ] ---> Selects N' evenly distributed centroids
              |
              v
[ 2. Metric Ball Query (Radius r) ]  ---> Groups K neighbors per centroid
              |
              | Local relative coordinates: p_i - c_k (Translation Invariant!)
              v
[ 3. Local PointNet Layer ]          ---> Shared MLPs + Max-Pooling
              |
              v
Downsampled Geometric Set (N' points with C-dimensional deep features!)
```

---

## 2. Mathematical Formulation

### 2.1 Furthest Point Sampling (FPS) Algorithm
Let $\mathcal{P} = \{ \mathbf{p}_1, \dots, \mathbf{p}_N \} \subset \mathbb{R}^3$.
We seek a subset $\mathcal{S} \subset \mathcal{P}$ of size $N'$.

1. Initialize $\mathcal{S} = \{ \mathbf{p}_{\text{start}} \}$ with an arbitrary point (e.g., index 0).
2. Maintain a distance array $\mathbf{D} \in \mathbb{R}^N$ where $D_i = \min_{\mathbf{c} \in \mathcal{S}} \|\mathbf{p}_i - \mathbf{c}\|_2^2$.
3. For step $k = 2$ to $N'$:
   $$\mathbf{c}_k = \arg\max_{\mathbf{p}_i \in \mathcal{P}} D_i$$
   $$\mathcal{S} \leftarrow \mathcal{S} \cup \{ \mathbf{c}_k \}$$
   Update distance vector:
   $$D_i \leftarrow \min\left( D_i, \|\mathbf{p}_i - \mathbf{c}_k\|_2^2 \right), \quad \forall i \in \{1, \dots, N\}$$

FPS provides significantly better spatial coverage than uniform random sampling, which tends to over-sample dense clusters and miss sparse boundary points.

---

### 2.2 Local Set Abstraction Operator
Let centroid $\mathbf{c} \in \mathbb{R}^3$ have $K$ neighbor points $\{ \mathbf{p}_1, \dots, \mathbf{p}_K \}$ with corresponding feature vectors $\{ \mathbf{f}_1, \dots, \mathbf{f}_K \} \subset \mathbb{R}^C$.

The input to the local MLP is the concatenated relative coordinate and feature:

$$
\mathbf{x}_j = \left[ (\mathbf{p}_j - \mathbf{c}) \;\|\; \mathbf{f}_j \right] \in \mathbb{R}^{3 + C}
$$

The updated centroid feature vector $\mathbf{f}' \in \mathbb{R}^{C'}$ is evaluated via:

$$
\mathbf{f}' = \max_{j=1 \dots K} \left( \text{MLP}(\mathbf{x}_j) \right)
$$

---

## 3. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn

def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Furthest Point Sampling (FPS) on batch of point clouds.
    xyz: [B, N, 3] coordinates
    npoint: target number of centroids N'
    Returns: [B, npoint] indices of sampled points
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        
    return centroids


def ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Finds up to nsample points within metric radius for each centroid.
    xyz: [B, N, 3] all points
    new_xyz: [B, npoint, 3] centroid coordinates
    Returns: [B, npoint, nsample] neighbor indices
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    
    # Pairwise squared Euclidean distance: [B, S, N]
    sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)
    
    # Sort indices by distance
    group_idx = torch.argsort(sqrdists, dim=-1)[:, :, :nsample]
    
    # Mask out points outside radius by replacing with first point in neighborhood
    radius_sq = radius ** 2
    mask = sqrdists.gather(dim=-1, index=group_idx) > radius_sq
    group_idx[mask] = group_idx[:, :, 0:1].repeat(1, 1, nsample)[mask]
    
    return group_idx


class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction (Sampling + Ball Query + Local PointNet)."""
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: list[int]):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        
        # Build shared MLP layers
        layers = []
        last_c = in_channel + 3 # 3 for relative coordinates (p_i - c)
        for out_c in mlp:
            layers.append(nn.Conv2d(last_c, out_c, kernel_size=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            last_c = out_c
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        xyz: [B, N, 3] coordinates
        points: [B, N, C] optional feature channels
        Returns: (new_xyz [B, npoint, 3], new_points [B, npoint, C'])
        """
        B, N, _ = xyz.shape
        fps_idx = furthest_point_sample(xyz, self.npoint)
        new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).repeat(1, 1, 3))
        
        # Group neighbors via Ball Query
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz) # [B, npoint, nsample]
        
        # Gather coordinates and express relative to centroid
        grouped_xyz = torch.gather(xyz.unsqueeze(1).repeat(1, self.npoint, 1, 1), 2, 
                                   idx.unsqueeze(-1).repeat(1, 1, 1, 3))
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2) # [B, npoint, nsample, 3]
        
        if points is not None:
            grouped_points = torch.gather(points.unsqueeze(1).repeat(1, self.npoint, 1, 1), 2, 
                                          idx.unsqueeze(-1).repeat(1, 1, 1, points.shape[-1]))
            grouped_features = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            grouped_features = grouped_xyz_norm
            
        # Reshape to [B, C, nsample, npoint] for 2D Conv
        grouped_features = grouped_features.permute(0, 3, 2, 1)
        new_points = self.mlp(grouped_features) # [B, C', nsample, npoint]
        new_points = torch.max(new_points, 2)[0].transpose(1, 2) # [B, npoint, C']
        
        return new_xyz, new_points
```

---

## 4. Models in the Vault Utilizing Set Abstraction

- **[[architectures/3d-pointclouds-and-lidar/pv-rcnn|PV-RCNN]]**: Uses Set Abstraction to integrate voxel CNN features into keypoint representations across the 3D scene.
- **[[architectures/3d-pointclouds-and-lidar/pointpillars|PointPillars]]**: Utilizes PointNet feature encoders to convert raw vertical point pillars into dense 2D pseudo-image feature maps.
- **[[architectures/pose-and-robotics-manipulation/foundationpose|FoundationPose]] & [[architectures/pose-and-robotics-manipulation/megapose|MegaPose]]**: Hierarchical 3D point sampling for CAD mesh alignment and 6-DoF pose refinement.

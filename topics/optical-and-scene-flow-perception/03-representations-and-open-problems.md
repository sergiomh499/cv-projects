---
title: "Optical & Scene Flow: Representations & Open Frontiers"
type: production-playbook
domain: Optical Flow & Scene Flow Perception
tags:
  - optical-flow
  - scene-flow
  - 3d-motion
  - point-cloud
  - open-problems
updated: 2026-09-08
aliases:
  - Flow Representations & Frontiers
---

# 🔬 Optical & Scene Flow: Representations & Open Frontiers

Dense 3D Point Cloud Scene Flow, non-rigid dynamic reconstruction, RAFT correlation pyramid details, and research frontiers.

Related notes: [[topics/optical-and-scene-flow-perception/00-optical-and-scene-flow-perception-moc|Optical Flow MOC]].

---

## 1. 2D Optical Flow vs. 3D Scene Flow

| Property | 2D Optical Flow | 3D Scene Flow |
| :--- | :--- | :--- |
| **Input Modality** | Monocular RGB sequence $(I_t, I_{t+1})$ | RGB-D video or successive LiDAR point clouds $(P_t, P_{t+1})$ |
| **Vector Space** | Tangent image plane: $[u, v]^T \in \mathbb{R}^2$ (pixels/frame) | Metric physical 3D world: $[v_x, v_y, v_z]^T \in \mathbb{R}^3$ (m/s) |
| **Depth Ambiguity** | Severe (scale unknown without calibrated depth) | **Zero — explicit metric 3D displacement** |
| **Primary Use Cases** | Frame interpolation, video stabilization, tracking | AV obstacle velocity estimation, non-rigid cloth simulation |
| **Key Failure Mode** | Occlusion, aperture problem, large displacement | Point cloud sparsity, sensor ego-motion contamination |

---

## 2. RAFT 4D Correlation Volume: Why It Works

The RAFT architecture (Teed & Deng, 2020) deserves deep analysis because it is the conceptual foundation of all modern flow architectures. The all-pairs correlation volume is the decisive innovation:

### Correlation Volume Construction

Given feature maps $\mathbf{g}_1, \mathbf{g}_2 \in \mathbb{R}^{H' \times W' \times C}$ (with $H' = H/8$, $W' = W/8$, $C = 256$):

$$C(\mathbf{x}_1, \mathbf{x}_2) = \sum_{k=1}^{C} g_1(\mathbf{x}_1)_k \cdot g_2(\mathbf{x}_2)_k \in \mathbb{R}$$

The complete 4D volume $\mathbf{C} \in \mathbb{R}^{H' \times W' \times H' \times W'}$ stores the matching score between every pixel in frame 1 with every pixel in frame 2. At $480\times640$ resolution this volume has $60 \times 80 \times 60 \times 80 = 23.0\,\text{M}$ entries requiring $92\,\text{MB}$ in float32 — fit within GPU L2 cache on modern hardware.

### Multi-Scale Pyramid

Average-pool the last two (source) dimensions at factors $\{1, 2, 4, 8\}$:

$$\mathbf{C}^s = \text{AvgPool}_{s\times s}(\mathbf{C}[\ldots, :, :]) \in \mathbb{R}^{H' \times W' \times (H'/s) \times (W'/s)}$$

At $s=8$: the correlation map at each query pixel collapses to just $8\times 8 = 64$ entries — enabling cheap coarse global matching to initialize flow for large displacements.

### Local Lookup Strategy

At GRU iteration $k$ with current flow estimate $\mathbf{f}^k$, the lookup retrieves a $r=4$ radius neighborhood around the predicted correspondence in each pyramid level:

$$\mathbf{r}^k(\mathbf{x}_1) = \text{concat}_{s \in \{1,2,4,8\}}\left\{ C^s\!\left(\mathbf{x}_1,\; \mathbf{x}_1 + \frac{\mathbf{f}^k(\mathbf{x}_1)}{s} + \Delta\mathbf{x}\right) \;\middle|\; \|\Delta\mathbf{x}\|_\infty \le r \right\}$$

The concatenated lookup vector $\mathbf{r}^k \in \mathbb{R}^{4 \times (2r+1)^2} = \mathbb{R}^{4 \times 81} = \mathbb{R}^{324}$ captures local correlation context at 4 spatial scales simultaneously. This is the input to the ConvGRU that produces the flow update $\Delta\mathbf{f}^{k+1}$.

---

## 3. GMFlow Global Matching via Cross-Attention

GMFlow (Xu et al., CVPR 2022) formulates optical flow as a **feature matching problem** solved by transformer cross-attention, eliminating the iterative GRU entirely:

### Cross-Attention Flow Matching

Feature maps from frame 1 and frame 2 are processed by a standard ViT-style transformer with alternating self- and cross-attention layers:

**Self-attention** (within each frame, for context):
$$\text{SA}(\mathbf{g}) = \text{softmax}\!\left(\frac{\mathbf{g} W_Q (\mathbf{g} W_K)^T}{\sqrt{d}}\right) \mathbf{g} W_V$$

**Cross-attention** (between frames, for matching):
$$\mathbf{g}_1' = \text{softmax}\!\left(\frac{\mathbf{g}_1 W_Q (\mathbf{g}_2 W_K)^T}{\sqrt{d}}\right) \mathbf{g}_2 W_V$$

The attention matrix $\mathbf{A} \in \mathbb{R}^{H'W' \times H'W'}$ directly encodes the matching probability between every pixel pair. Flow is extracted from the **soft argmax** of each row of $\mathbf{A}$:

$$\mathbf{f}(\mathbf{x}_1) = \sum_{\mathbf{x}_2} A(\mathbf{x}_1, \mathbf{x}_2) \cdot (\mathbf{x}_2 - \mathbf{x}_1)$$

This global operation resolves the aperture problem: even a featureless sky region finds its match because the entire frame 2 is visible in the attention computation, not just a local window.

**Computational cost**: Attention is $O((H'W')^2 \cdot d)$. At $1/8$ downsampling of $480\times640$: $(60\times80)^2 \times 256 = 7.37\,\text{G}$ FLOPs. Using window-partitioned attention (Swin-style) reduces this to $O(HW \cdot w^2 \cdot d)$ with window size $w=7$.

---

## 4. Mathematical Formulation: 3D Point Cloud Scene Flow

Given two successive point clouds $P = \{p_i\}_{i=1}^{N}$ at time $t$ and $Q = \{q_j\}_{j=1}^{M}$ at time $t+1$ where $p_i, q_j \in \mathbb{R}^3$:

The goal is to find displacement vectors $D = \{d_i\}_{i=1}^{N}$ such that $p_i' = p_i + d_i$ matches the true physical correspondence in $Q$.

### Self-Supervised Chamfer & Cycle-Consistency Objective

In practical deployments without ground-truth 3D motion labels:

$$\mathcal{L}_{\text{SceneFlow}} = \mathcal{L}_{\text{Chamfer}}(P + D, Q) + \lambda_s\,\mathcal{L}_{\text{Laplacian}}(D) + \lambda_c\,\mathcal{L}_{\text{Cycle}}(P, Q, D)$$

**Chamfer Distance** — pulls warped points toward nearest target:
$$\mathcal{L}_{\text{Chamfer}} = \sum_{p' \in P'} \min_{q \in Q} \|p' - q\|_2^2 + \sum_{q \in Q} \min_{p' \in P'} \|q - p'\|_2^2$$

**Laplacian Smoothness** — rigid-body consistency within a connected component:
$$\mathcal{L}_{\text{Laplacian}} = \sum_{p_i \in P} \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \|d_i - d_j\|_2^2$$

**Cycle Consistency** — enforces that forward + backward flow sum to zero:
$$\mathcal{L}_{\text{Cycle}} = \sum_i \|d_i^{P \to Q} + d_{\pi(i)}^{Q \to P}\|_2^2$$

where $\pi(i) = \text{argmin}_j \|p_i + d_i - q_j\|_2$ is the nearest neighbor correspondence after warping.

### Scene Flow Benchmarks

| Method | FlyingThings3D AEPE | KITTI-SF AEPE | Inference (ms) |
| :--- | :--- | :--- | :--- |
| FlowNet3D (2019) | 0.114 m | 0.177 m | 240 ms |
| PointPWC-Net (2021) | 0.059 m | 0.069 m | 117 ms |
| FlowStep3D (2022) | 0.045 m | 0.051 m | 520 ms |
| **FastFlow3D (2023)** | **0.038 m** | **0.043 m** | **32 ms** |

FastFlow3D uses a BEV (Bird's-Eye View) voxelized representation with 2D convolutions rather than point-wise operations, achieving real-time performance on the NVIDIA DRIVE AGX platform for autonomous vehicle scene flow.

---

## 5. Open Frontiers (2025–2026)

### 4D Gaussian Splatting Dynamics

Inferring continuous scene flow directly on 3D Gaussian Splats (4D-GS, Wu et al. 2024): each Gaussian has a velocity attribute $v_i \in \mathbb{R}^3$; scene flow is the continuous-time interpolation of the Gaussian positions. Enables instantaneous zero-lag rendering of dynamic fluid and cloth deformations at 30+ FPS without per-frame reconstruction.

### Event-Assisted Optical Flow

Merging standard 30 FPS RGB frames with asynchronous neuromorphic event streams (Prophesee EVK4) to resolve microsecond motion blur during extreme angular velocity camera maneuvers ($>1000°/\text{s}$). EvFlowNet (2024) achieves $<0.5\,\text{px}$ EPE on the MultiFlow-Events benchmark — representing a 5× improvement over pure RGB flow at high angular velocities.

### Foundation Flow Models (FlowFormer++, 2025)

Pretrained on 4 billion frame pairs from synthetic + real video; zero-shot generalization to medical endoscopy (bowel peristalsis), satellite imagery (cloud tracking), and industrial conveyor flow estimation. EPE on Sintel Final: $1.04\,\text{px}$ — approaching the ground-truth noise floor of the Sintel dataset itself ($\sim 0.5\,\text{px}$ inherent rendering error).

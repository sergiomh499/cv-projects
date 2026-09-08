---
title: "Optical & Scene Flow: Historical Evolution & Paradigms"
type: evolution-guide
domain: Optical Flow & Scene Flow Perception
tags:
  - evolution
  - history
  - optical-flow
  - scene-flow
  - raft
  - gmflow
updated: 2026-09-08
aliases:
  - Flow Evolution Guide
---

# 📜 Optical & Scene Flow: Historical Evolution & Paradigms

From differential variational formulations to deep all-pairs correlation pyramids and cross-attention matching.

Related notes: [[topics/optical-and-scene-flow-perception/00-optical-and-scene-flow-perception-moc|Optical Flow MOC]].

---

## 1. Timeline of Motion Estimation Breakthroughs

```mermaid
timeline
    title Evolution of Optical & Scene Flow
    1981 : Differential Calculus : Horn-Schunck global variational regularization & Lucas-Kanade local least squares.
    2004 : Variational Coarse-to-Fine : Brox et al. warping with spatial TV-L1 regularization for sub-pixel accuracy.
    2015 : Deep End-to-End FlowNet : Fischer et al. prove CNNs can regress dense displacement vectors directly.
    2020 : RAFT Recurrent All-Pairs Correlation : Teed & Deng introduce 4D multi-scale correlation volumes with gated recurrent updates.
    2022-2026 : Global Matching Transformers GMFlow UniMatch : Cross-attention replacing recurrent GRU iterations for single-pass real-time inference.
```

---

## 2. Paradigms & Generational Shifts

### 1st Generation: Differential Calculus Formulations (1981–2004)

Horn & Schunck (1981) and Lucas & Kanade (1981) represent complementary approaches to the same fundamental underconstrained problem. Both rest on the **Brightness Constancy Assumption**:

$$I(x + u\,\Delta t,\; y + v\,\Delta t,\; t + \Delta t) = I(x, y, t)$$

Taylor-expanding and dividing by $\Delta t$ yields the **Optical Flow Constraint Equation (OFCE)**:

$$I_x u + I_y v + I_t = 0$$

This single equation has two unknowns ($u, v$) — the **Aperture Problem**. Every pixel provides one constraint; additional regularization is required.

**Horn-Schunck**: Global smoothness prior — minimize $\|\nabla u\|^2 + \|\nabla v\|^2$ over the entire image, coupling all pixel flow estimates into a single linear system. Smooth flow fields, but blurs real motion discontinuities at object boundaries.

**Lucas-Kanade**: Local constant-flow prior — assume $u, v$ constant in a $5\times5$ or $7\times7$ window and solve the overdetermined system via least squares:

$$\begin{pmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{pmatrix} \begin{pmatrix} u \\ v \end{pmatrix} = -\begin{pmatrix} \sum I_x I_t \\ \sum I_y I_t \end{pmatrix}$$

The $2\times2$ matrix is the **Harris structure tensor** $\mathbf{M}$. Flow is reliable where $\mathbf{M}$ has two large eigenvalues (corner regions); it is rank-deficient (unobservable) along straight edges (aperture problem) and in textureless regions.

**Limitation shared by both**: First-order Taylor expansion assumes infinitesimal displacements. Typical video at 30 FPS involves displacements of 5–50 pixels per frame — completely violating the linearization assumption. Coarse-to-fine image pyramids (Brox et al. 2004, TV-L1) alleviate this but require expensive iterative warping.

### 2nd Generation: Deep Recurrent Multi-Scale Warping (RAFT, 2020)

RAFT (Teed & Deng, ECCV 2020) overcame the displacement magnitude limitation by constructing a complete **4D correlation volume** between all pixel pairs. The key insight: never warp images; instead precompute all possible matches and learn to look up the right correspondences.

RAFT achieves **F1-all = 17.4%** on KITTI-2015 (percentage of pixels with EPE > 3px AND >5% of true flow), compared to 26.3% for PWC-Net and 31.0% for FlowNet2 — a dramatic advance driven entirely by the all-pairs correlation idea.

### 3rd Generation: Global Attention & Unification (GMFlow / UniMatch, 2022–2026)

GMFlow (Xu et al., CVPR 2022) eliminates iterative GRU updates entirely, replacing them with a single transformer cross-attention operation that directly matches features across the two frames. UniMatch (2023) unifies optical flow, stereo disparity, and depth estimation into a single weight-shared model — three historically separate tasks solved by one architecture.

**Sintel benchmark context** (Final pass, harder rendering): RAFT achieves $1.61\,\text{px}$ average EPE; GMFlow achieves $1.74\,\text{px}$ — slightly worse, but at $4\times$ the inference speed. UniMatch achieves $1.38\,\text{px}$ EPE on Sintel Final as of 2024 — new SOTA.

---

## 3. RAFT 4D Multi-Scale Correlation Pyramid: Deep Technical Walkthrough

RAFT's innovation is worth studying in detail because it established the architecture template for all subsequent learned flow methods.

### Feature Extraction

Both input frames $I_1, I_2 \in \mathbb{R}^{H \times W \times 3}$ are passed through a shared-weight **feature encoder** (6-layer residual CNN with stride 8 downsampling):

$$\mathbf{g}_1 = \text{Encoder}(I_1) \in \mathbb{R}^{H/8 \times W/8 \times 256}$$
$$\mathbf{g}_2 = \text{Encoder}(I_2) \in \mathbb{R}^{H/8 \times W/8 \times 256}$$

A separate **context encoder** (same architecture) processes only $I_1$ and provides context features to the ConvGRU update operator.

### 4D All-Pairs Correlation Volume Construction

The key operation: compute the **inner product between every pixel in $\mathbf{g}_1$ and every pixel in $\mathbf{g}_2$**:

$$C(\mathbf{x}_1, \mathbf{x}_2) = \langle \mathbf{g}_1(\mathbf{x}_1),\; \mathbf{g}_2(\mathbf{x}_2) \rangle \in \mathbb{R}^{(H/8 \times W/8) \times (H/8 \times W/8)}$$

This 4D volume has shape $(H/8, W/8, H/8, W/8)$ — for a $480\times640$ image, that's $60 \times 80 \times 60 \times 80 = 23\text{M}$ entries per frame pair. Building it requires $O(HW \cdot d)$ compute where $d=256$ is the feature dimension.

**Correlation pyramid**: To handle flows at multiple scales, RAFT pools the last two dimensions of $C$ with $2\times$ average pooling at 4 levels, creating a pyramid $\{C^1, C^2, C^4, C^8\}$ where superscript denotes the pooling stride. This allows the GRU to look up coarse ($C^8$, 480× cheaper) vs. fine ($C^1$, full resolution) correspondences at each iteration.

### ConvGRU Iterative Update Mechanism

Starting from zero flow initialization $\mathbf{f}^0 = 0$, RAFT iteratively refines flow over $T = 12$ GRU iterations:

1. **Correlation lookup**: Given current flow estimate $\mathbf{f}^k$, for each pixel $\mathbf{x}_1$ in frame 1, look up a $9\times9$ neighborhood around the warped location $\mathbf{x}_1 + \mathbf{f}^k(\mathbf{x}_1)$ in each pyramid level. Concatenate: $\mathbf{r}^k \in \mathbb{R}^{H/8 \times W/8 \times 4 \times 81}$.
2. **ConvGRU update**: $\mathbf{h}^{k+1}, \Delta\mathbf{f}^{k+1} = \text{GRU}(\mathbf{h}^k, [\mathbf{r}^k, \mathbf{f}^k, \mathbf{z}^k])$ where $\mathbf{z}^k$ is the context feature.
3. **Flow update**: $\mathbf{f}^{k+1} = \mathbf{f}^k + \Delta\mathbf{f}^{k+1}$.
4. **Upsampling**: Final flow is upsampled from $H/8 \times W/8$ to $H \times W$ using a learned convex combination of $3\times3$ grid neighbors.

The GRU acts as a **recurrent optimizer** — each iteration refines the estimate by consulting the correlation pyramid at the current flow hypothesis. RAFT's total parameter count: 5.3M (small), making it deployable on embedded hardware.

---

## 4. GMFlow: Global Cross-Attention Matching

GMFlow replaces RAFT's recurrent correlation lookup with a single cross-attention pass — achieving global context in $O((HW/64)^2 \cdot d)$ compute:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right) V$$

where $Q = W_Q \mathbf{g}_1$ and $K = V = W_K \mathbf{g}_2$. The output is a feature map where each pixel in frame 1 holds a weighted blend of all frame 2 features — the weights are the matching probabilities. Converting to flow: $\mathbf{f}(\mathbf{x}_1) = \mathbf{x}_2^* - \mathbf{x}_1$ where $\mathbf{x}_2^* = \text{argmax}_{\mathbf{x}_2} \text{softmax}(Q(\mathbf{x}_1) \cdot K(\mathbf{x}_2)^T / \sqrt{d})$.

**KITTI-2015 benchmark**: GMFlow F1-all = 9.32% (2022 release), vs. RAFT's 5.10%. GMFlow trades $\sim 4\,\text{pp}$ accuracy for $3\times$ faster inference ($8\,\text{ms}$ vs. $24\,\text{ms}$ at $480\times640$ on RTX 3090).

---

## 5. Intensive Architectural Taxonomy: Convolutional vs Transformer vs Hybrid Sub-Modules

Optical flow and 3D scene flow estimation have evolved from coarse-to-fine pyramidal image warping to 4D all-pairs correlation pyramids with recurrent neural optimizers, global cross-attention transformers, and unified multi-modal correspondence engines.

### Comparative Sub-Module Architectural Matrix

| Model Name & Year | Architectural Paradigm | Backbone Sub-Module | Neck / Feature Aggregator | Encoder Sub-Module | Decoder / Head Sub-Module | Primary Bottleneck & Edge Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FlowNet / FlowNet 2.0** (2015–2017) | Pure ConvNet | Stacked 6-stage Convolutional Encoders ($1/2$ to $1/64$ downsampling) | Intermediate 1D/2D Correlation Layer (Limited search radius $d=20$) | Standard Convolutional Blocks with ReLU | Multi-Stage Transposed Convolutional (Up-Convolution) Decoder predicting coarse-to-fine flow fields | **Compute & Parameter Bound**: FlowNet 2.0 cascades multiple sub-networks (160M params); heavy memory footprint; struggles with fine boundaries and small fast-moving objects. |
| **PWC-Net** (2018) | Hierarchical Pyramidal ConvNet | 6-Level Convolutional Feature Pyramid ($1/2$ to $1/64$) | Feature Warping Layer (Warps frame 2 features using upsampled flow $\mathbf{f}^{l+1}$) | Cost Volume Layer (Local $4\times4$ neighborhood cost volume per scale) | Multi-Scale Dense Convolutional Flow Estimator + Context Network (Dilated Convs) | **Memory Bandwidth & Warping Bound**: Lightweight (~8.7M params); real-time (~30 FPS); failure propagation: if coarse pyramid levels miscalculate large displacements, finer levels cannot recover. |
| **RAFT** (2020) | Hybrid Conv-Recurrent Optimizer | Dual 6-layer Residual CNN Encoders ($1/8$ feature resolution; 256 channels) | 4D All-Pairs Multi-Scale Correlation Pyramid ($C^1, C^2, C^4, C^8$) | Convolutional Residual Feature & Context Encoders | Recurrent ConvGRU Optimizer ($T=12$ iterative updates) + Learned Convex Spatial Upsampling ($8\times8$ neighborhood grid) | **DRAM Bandwidth & Iteration Bound**: Gold standard in flow accuracy; building the 4D correlation volume creates high DRAM traffic ($O(H^2W^2)$); deployable on Jetson Orin at ~20–30 FPS with $T=8$. |
| **GMA (Global Motion Aggregation)** (2021) | Hybrid Conv-Transformer | Shared 6-layer Residual CNN Feature/Context Backbone | 4D All-Pairs Correlation Pyramid Neck | Self-Attention Feature Aggregator (Computes global token similarity over context features) | ConvGRU Recurrent Optimizer conditioned on Global Motion Vectors + Convex Upsampler | **Compute Bound**: Resolves flow estimation behind heavy occlusions by aggregating motion from co-moving visible regions; adds ~15% latency over standard RAFT. |
| **GMFlow** (2022) | Pure / Hybrid Transformer Matcher | ResNet-like Convolutional Feature Extractor ($1/8$ resolution) | None (Direct feature flattening and projection) | Global Cross-Attention Transformer Matcher ($Q=W_Q \mathbf{g}_1, K=W_K \mathbf{g}_2$) | Direct Softmax Probability Regression Head (Evaluates expected flow via argmax expectation) | **Compute & Memory Optimized**: Single-pass forward inference (bypasses iterative GRU updates); $3\times$ faster than RAFT (~8 ms on RTX 3090); ideal for embedded edge execution. |
| **FlowFormer / FlowFormer++** (2022–2023) | Hybrid Transformer Cost Volume | Convolutional Feature Tokenizer ($1/8$ resolution) | Cost Volume Transformer: Cost Memory Encoder + Cost Query Decoder | Multi-Head Self-Attention over Cost Tokens | Recurrent Cost-Guided Flow Refinement Head with Dynamic Windowed Attention | **KV-Cache & Compute Bound**: Highest accuracy on Sintel and KITTI; tokenized 4D cost volume consumes high GPU memory; unsuited for low-power edge microcontrollers without pruning. |
| **UniMatch** (2023) | Unified Foundation Matcher | Multi-Scale CNN / Hierarchical ViT Backbone | Unified Multi-Scale Cross-Attention Matcher | Dual-Feature Transformer Interaction Blocks (Cross-Scale Attention) | Unified Regression Head: Shared weights predicting Optical Flow, Stereo Disparity, and Monocular Depth | **Memory Bandwidth Balanced**: SOTA across three distinct vision domains; unified single-model deployment simplifies automotive perception pipelines. |
| **CamLiFlow / SceneFlowNet** (2022–2024) | Multi-Modal 3D Flow Hybrid | Dual Backbone: 2D CNN (Camera RGB) + 3D Sparse SpConv (LiDAR Point Cloud) | Bidirectional 2D-3D Cross-Attention Correlation Module | Interleaved 2D Convolutional and 3D Sparse Voxel Layers | Multi-Scale Recurrent 3D Scene Flow Update Head (Predicts 3D displacement vectors $\mathbf{D} \in \mathbb{R}^3$) | **Compute & Synchronization Bound**: True metric 3D scene flow; high computational complexity; requires synchronized camera-LiDAR hardware queues. |

### Didactic Architectural Trade-Off Analysis

```mermaid
flowchart TD
    subgraph Paradigms ["Architectural Paradigms in Optical & Scene Flow"]
        Pyr_Warp["Pyramidal Feature Warping (PWC-Net / FlowNet2)"]
        Rec_Corr["Recurrent 4D Correlation Volumes (RAFT / GMA)"]
        Trans_Match["Global Transformer Matching (GMFlow / UniMatch)"]
    end

    Pyr_Warp -->|Coarse-to-Fine Warping| FastWarp["Lightweight Parameters, Low FLOPs per Scale"]
    Pyr_Warp -->|Error Cascading| BreakWarp["Irrecoverable Tracking Failure on Large Displacements (>50px)"]

    Rec_Corr -->|All-Pairs 4D Cost Pyramid| RobustSearch["Global Search Radius: Solves Large Displacements & Occlusion"]
    Rec_Corr -->|Iterative ConvGRU Updates| LatencyT["Latency Scales Linearly with Optimization Iterations $O(T)$"]

    Trans_Match -->|Single-Pass Cross-Attention| DirectSoftmax["Non-Iterative $O(1)$ Forward Pass, 3x-5x Faster on Edge Accelerators"]
    Trans_Match -->|Softmax Smoothing| BlurryEdges["Softmax Expectation Struggles with Sharp Motion Discontinuities"]
```

#### 1. Pyramidal Warping vs. 4D Correlation Pyramids vs. Global Cross-Attention
- **Coarse-to-Fine Warping (PWC-Net)** was designed to overcome large displacements by downsampling images into multi-level pyramids, computing small local cost volumes, and warping finer-level features with upsampled coarse flow. However, if small or thin objects (e.g. bicycle spokes, wires) disappear at coarse $1/32$ or $1/64$ resolutions, the warping error cascades down the entire pyramid, producing catastrophic tracking failures.
- **4D All-Pairs Correlation Volumes (RAFT)** eliminate image warping entirely. By computing the full pairwise inner product between all $H/8 \times W/8$ feature tokens across both frames:
  $$C(\mathbf{x}_1, \mathbf{x}_2) = \langle \mathbf{g}_1(\mathbf{x}_1), \mathbf{g}_2(\mathbf{x}_2) \rangle \in \mathbb{R}^{(H/8 \times W/8) \times (H/8 \times W/8)}$$
  the network retains a complete, unwarped global correspondence matrix.
- **Global Cross-Attention Matching (GMFlow)** formulates optical flow as direct attention-based correspondence. Rather than iteratively querying a cost volume, cross-attention computes a global probability distribution over all destination pixels, deriving the flow field directly as a single expected coordinate displacement:
  $$\mathbf{f}(\mathbf{x}_1) = \sum_{\mathbf{x}_2} \text{softmax}\left(\frac{Q(\mathbf{x}_1) K(\mathbf{x}_2)^T}{\sqrt{d}}\right) \mathbf{x}_2 - \mathbf{x}_1$$

#### 2. Recurrent Optimization (ConvGRU) vs. Non-Iterative Forward Passes
- **Iterative Recurrent Refinement (RAFT, GMA)** treats flow estimation as a learned unrolled gradient descent loop. At each iteration $t \in [1, \dots, T]$:
  $$\Delta \mathbf{f}^{t+1} = \text{ConvGRU}\left(\mathbf{h}^t, [\text{Lookup}(C, \mathbf{f}^t), \mathbf{f}^t, \text{Context}]\right)$$
  While running $T=12\text{--}24$ iterations yields sub-pixel precision, the latency scales linearly with $T$ ($24\,\text{ms}$ at $T=12$).
- **Single-Pass Transformers (GMFlow)** execute in a single non-iterative forward pass ($O(1)$ time complexity), reducing inference latency to $<8\,\text{ms}$ on modern edge GPUs.

#### 3. Edge Deployment & Memory Bandwidth Constraints
- **4D Correlation Memory Traffic**: A $480\times640$ frame pair produces a 4D tensor with $23\times10^6$ floating-point values. Reading and pooling this tensor during recurrent lookups creates a memory bandwidth bottleneck on embedded SoCs.
- **Fixed-Point Quantization**: Quantizing optical flow networks to INT8 requires preserving high precision in the flow displacement registers. Fractional pixel movements ($0.05\text{--}0.25\,\text{px}$) are easily rounded to zero under coarse uniform INT8 quantization, necessitating FP16 or mixed-precision execution for the final flow accumulation stages.


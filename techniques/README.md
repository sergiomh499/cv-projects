---
title: "Techniques & Algorithmic Mechanics Vault"
type: MOC
domain: "Computer Vision & AI Mechanics"
tags:
  - moc
  - technique
  - algorithms
  - mathematics
  - signal-processing
  - edge-ai
status: evergreen
updated: 2026-09-09
aliases:
  - "Techniques Directory"
  - "Vision Mechanics"
---

# 🧬 Techniques & Algorithmic Mechanics Vault

> **Navigation**: [[README|🏠 Central Knowledge Hub]] / **Techniques & Algorithmic Mechanics**
> **Map of Content**: Access the full visual graph at [[techniques/00-techniques-moc|Techniques MOC]].

This directory catalogs standalone, mathematically rigorous, and code-grounded explanations of foundational techniques, signal transforms, variational solvers, and neural operators across computer vision, multimodal physical AI, and edge deployment.

---

## 📚 Core Documents

| Document | Category | Key Concept & Operator | Link |
| :--- | :--- | :--- | :--- |
| **Fourier Domain Adaptation** | Signal Processing & Sim2Real | Frequency-space amplitude swapping preserving exact geometric phase | [[techniques/fourier-domain-adaptation\|Fourier Domain Adaptation Guide]] |
| **Deformable Convolutions** | Convolutional Operators | DCNv1 to DCNv4: Continuous 2D offsets, modulation, and FlashDeformable kernels | [[techniques/deformable-convolutions\|Deformable Convolutions Guide]] |
| **Hypergraph Computation** | Non-Euclidean Graph Perception | Spectral hypergraph convolution capturing high-order semantic cliques | [[techniques/hypergraph-computation\|Hypergraph Computation Guide]] |
| **Gradient Reversal & DANN** | Adversarial Transfer Learning | Minimax single-pass domain invariance via gradient negation | [[techniques/gradient-reversal-and-dann\|Gradient Reversal & DANN Guide]] |
| **DFL vs. Direct Regression** | Bounding Box Regression | Statistical Softmax bin expectation vs. continuous metric regression | [[techniques/distribution-focal-loss-vs-direct-regression\|DFL vs Direct Regression Guide]] |
| **Structural Reparameterization** | Edge Acceleration & In-Place Folding | Multi-branch training fused into single-path $3\times 3$ kernel at inference | [[techniques/structural-reparameterization\|Structural Reparameterization Guide]] |
| **Bipartite Matching & Hungarian** | Set Prediction & NMS-Free Detection | Global 1:1 optimal assignment matching eliminating heuristic NMS | [[techniques/bipartite-matching-and-hungarian-assigner\|Bipartite Matching & Hungarian Assigner Guide]] |
| **Lift-Splat-Shoot & BEV Pooling** | Autonomous Driving & Sensor Fusion | Depth probability lifting, ego splatting, and fast cumsum voxel aggregation | [[techniques/lift-splat-shoot-bev-pooling\|Lift-Splat-Shoot & BEV Pooling Guide]] |
| **Visual State-Space Models (SSM)** | Sequence & Attention Alternatives | 2D Selective Scan (SS2D) achieving global receptive field in linear $\mathcal{O}(N)$ | [[techniques/visual-state-space-mamba\|Visual State-Space Models (SSM) Guide]] |
| **Continuous Flow Matching** | Generative Diffusion & Trajectories | Optimal Transport straight vector fields with 1–4 Euler ODE sampling steps | [[techniques/continuous-flow-matching\|Continuous Flow Matching Guide]] |
| **Action Chunking with C-VAE** | Physical AI & Robot Manipulation | $H$-step future trajectory prediction and exponential temporal ensembling | [[techniques/action-chunking-cvae\|Action Chunking with C-VAE Guide]] |
| **3D Gaussian Splatting Rasterization** | Spatial Radiance & Neural Rendering | Explicit 3D Gaussians, 2D EWA projection, and tile Radix sorting at $>100\text{ FPS}$ | [[techniques/3d-gaussian-splatting-rasterization\|3D Gaussian Splatting Guide]] |
| **Lie Algebra se(3) Pose Tracking** | Robotics & Visual SLAM | Unconstrained 6D tangent space optimization and analytical photometric Jacobians | [[techniques/lie-algebra-se3-pose-tracking\|Lie Algebra se(3) Pose Tracking Guide]] |
| **All-Pairs Correlation Pyramids** | Visual Motion & Optical Flow | Full 4D pairwise dot-product volume and local bilinear lookup operators | [[techniques/all-pairs-correlation-pyramids\|All-Pairs Correlation Pyramids Guide]] |
| **PTQ & Outlier Smoothing** | Quantization & Precision | SmoothQuant activation-weight difficulty migration for INT8 GEMM execution | [[techniques/post-training-quantization-and-outlier-smoothing\|PTQ & Outlier Smoothing Guide]] |
| **FlashAttention & Online Softmax** | Hardware Acceleration & Attention | SRAM block tiling and online maximum/denominator rescaling avoiding DRAM IO | [[techniques/flash-attention-and-online-softmax\|FlashAttention & Online Softmax Guide]] |
| **Contrastive SigLIP vs. InfoNCE** | Vision-Language Pretraining | Decoupled pairwise binary Sigmoid loss scaling to batch size 1,000,000+ | [[techniques/contrastive-learning-infonce-vs-siglip\|InfoNCE vs SigLIP Contrastive Learning Guide]] |
| **Spatial Hash Encodings & Voxels** | 3D Perception & Radiance Fields | Prime-XOR spatial hashing delivering dense representations in $\mathcal{O}(1)$ memory | [[techniques/multiresolution-hash-encodings-and-sparse-voxels\|Multiresolution Spatial Hash Encodings Guide]] |
| **Error-State Kalman Filter (ESKF)** | Sensor Fusion & Inertial Navigation | Minimal non-singular $15\times 15$ covariance on $\mathfrak{so}(3)$ manifolds | [[techniques/error-state-kalman-filter-eskf-vio\|Error-State Kalman Filter (ESKF) Guide]] |
| **Orthogonal Procrustes & Umeyama** | 6-DoF Registration & Pose Tracking | SVD cross-covariance decomposition for closed-form globally optimal $\mathrm{Sim}(3)$ | [[techniques/orthogonal-procrustes-and-umeyama-sim3\|Orthogonal Procrustes & Umeyama Guide]] |
| **Control Barrier Functions (CBF)** | Provable Safety & Control Theory | Online convex QP filter minimally perturbing uncertified neural policy actions | [[techniques/control-barrier-functions-safe-control\|Control Barrier Functions (CBF) Guide]] |
| **Phase-Shifting Profilometry** | Active 3D Sensing & Metrology | Multi-frequency heterodyne phase unwrapping for $<10\ \mu\text{m}$ surface depth | [[techniques/phase-shifting-profilometry-structured-light\|Phase-Shifting Profilometry Guide]] |
| **Variational Optical Flow (TV-L1)** | Visual Motion & Optimization | Total Variation regularization and primal-dual soft-thresholding shrinkage | [[techniques/variational-optical-flow-tv-l1\|Variational Optical Flow TV-L1 Guide]] |
| **Pillar Feature Encoding** | 3D LiDAR & Point Clouds | PointPillars 9D augmentation and 2D pseudo-image scatter projection | [[techniques/point-cloud-pillar-feature-encoding\|Point Cloud Pillar Feature Encoding Guide]] |
| **Bundle Adjustment & Schur** | Visual SLAM & Optimization | Arrowhead block Hessian marginalization solving reduced camera system in $<5\text{ ms}$ | [[techniques/bundle-adjustment-and-schur-complement\|Bundle Adjustment & Schur Complement Guide]] |
| **Epipolar Geometry & Essential Matrix** | Two-View Geometry & Odometry | Coplanarity constraint, Hartley 8-point normalization, SVD rank-2, and Cheirality | [[techniques/epipolar-geometry-essential-matrix\|Epipolar Geometry & Essential Matrix Guide]] |
| **Perspective-n-Point (EPnP)** | 6-DoF Pose & Spatial Localization | 4 virtual control points with barycentric Euclidean invariance for $\mathcal{O}(N)$ pose | [[techniques/perspective-n-point-epnp-pose-estimation\|Perspective-n-Point (EPnP) Guide]] |
| **Bounding Box Losses (CIoU / NWD)** | Bounding Box Regression | Scale-invariant metrics: GIoU, DIoU, CIoU, and Gaussian Wasserstein distance | [[techniques/bounding-box-losses-giou-ciou-nwd\|Bounding Box Regression Losses Guide]] |
| **Focal Loss & Class Imbalance** | Classification & Detection Losses | Dynamic modulating factor $(1-p_t)^\gamma$ and Quality Focal Loss for continuous IoU | [[techniques/focal-loss-and-class-imbalance\|Focal Loss & Class Imbalance Guide]] |
| **FPN & Path Aggregation (BiFPN)** | Multi-Scale Feature Fusion | Top-down semantics, bottom-up localization, and fast normalized weighted fusion | [[techniques/feature-pyramid-networks-and-path-aggregation\|FPN & Path Aggregation Guide]] |
| **Multi-Scale Deformable Attention** | Vision Transformers & Attention | Reference points with learned 2D sampling offsets and linear $\mathcal{O}(N_q C)$ complexity | [[techniques/multi-scale-deformable-attention\|Multi-Scale Deformable Attention Guide]] |
| **Shifted Window Attention (Swin)** | Hierarchical Vision Backbones | Linear-complexity local window attention (W-MSA) with cyclic shift and masking | [[techniques/shifted-window-attention-swin\|Shifted Window Self-Attention Guide]] |
| **Rotary Positional Embeddings (RoPE)** | Multimodal Attention & VLMs | Complex 2D orthogonal rotation matrices and 3D M-RoPE decomposing $(t, y, x)$ axes | [[techniques/rotary-positional-embeddings-rope-and-mrope\|Rotary Positional Embeddings Guide]] |
| **Implicit Neural Representations** | Neural Fields & 3D Radiance | Sinusoidal frequency projection $\gamma(\mathbf{v})$ and SIREN periodic activations | [[techniques/implicit-neural-representations-fourier-features\|Implicit Neural Representations Guide]] |
| **Sparse Submanifold Convolutions** | 3D Point Clouds & Voxels | Submanifold invariant $\text{Active}(\mathbf{y}) = \text{Active}(\mathbf{x})$ and Rulebook Gather-GEMM-Scatter | [[techniques/sparse-submanifold-convolutions-spconv\|Sparse Submanifold Convolutions Guide]] |
| **Iterative Closest Point (G-ICP)** | 3D Point Clouds & Registration | Surface normal projection $e_i = \mathbf{n}_i^T (\mathbf{R}\mathbf{p}_i + \mathbf{t} - \mathbf{q}_i)$ and Gaussian covariance matching | [[techniques/iterative-closest-point-and-generalized-icp\|Iterative Closest Point Guide]] |
| **Two-Stage Association (ByteTrack)** | Video Tracking & MOT | High/low confidence threshold partitioning ($\mathcal{D}_{\text{high}}, \mathcal{D}_{\text{low}}$) recovering occluded tracklets | [[techniques/two-stage-association-bytetrack\|Two-Stage Association Guide]] |
| **Spatial Pyramid Pooling (SPPF / ASPP)** | Multi-Scale Receptive Fields | Serial cascade equivalence and atrous dilated convolutions without downsampling | [[techniques/spatial-pyramid-pooling-spp-sppf-aspp\|Spatial Pyramid Pooling Guide]] |
| **RoIAlign & Bilinear Sampling** | Instance Segmentation & Extraction | Zero-quantization continuous sampling eliminating 8–16 pixel spatial drift | [[techniques/roialign-and-exact-bilinear-sampling\|RoIAlign & Exact Bilinear Sampling Guide]] |
| **Denoising Diffusion (DDPM / DDIM)** | Generative Vision & Trajectories | Closed-form forward Gaussian jump and deterministic non-Markovian ODE sampling | [[techniques/denoising-diffusion-ddpm-and-ddim\|Denoising Diffusion (DDPM & DDIM) Guide]] |

---

## 🎯 Quick Implementation Reference

All 40 documents provide:
1. **Mathematical derivations** and formal analytical proofs.
2. **Intuitive visual diagrams** (ASCII & Mermaid).
3. **Pure PyTorch / NumPy runnable reference modules** without external dependencies.
4. **Hardware efficiency profiles** for TensorRT, CUDA, and edge embedded targets.

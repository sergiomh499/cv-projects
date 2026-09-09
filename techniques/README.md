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

This directory catalogs standalone, mathematically rigorous, and code-grounded explanations of foundational techniques and operators across computer vision, multimodal physical AI, and edge deployment.

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

---

## 🎯 Quick Implementation Reference

All documents provide:
1. **Mathematical derivations** and signal processing proofs.
2. **Intuitive visual diagrams** (ASCII & Mermaid).
3. **Pure PyTorch runnable reference modules** without external dependencies.
4. **Hardware efficiency profiles** for TensorRT and edge embedded targets.

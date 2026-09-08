---
title: Object Classification Playbook
tags:
  - computer-vision
  - classification
  - dinov3
  - convnext
  - siglip
  - timm
updated: 2026-09-08
aliases:
  - Object Classification
---

# Object Classification Playbook

# Overview
Object Classification categorizes images, cropped regions-of-interest (RoIs), or multi-modal visual tokens into discrete predefined classes. While conceptually the simplest vision task, modern applications demand extreme efficiency, open-vocabulary generalization, robust fine-grained discrimination, and minimal out-of-distribution (OOD) failure modes.

Related notes: [[topics/object-detection/README|Object Detection]], [[topics/gpu-deployment/README|GPU Deployment]].

---

## SOTA & Research (2023–2026 Breakthroughs)

1. **DINOv3 & DINOv2: Vision Foundation Models with Self-Supervised Pre-Training** (Meta AI / FAIR, 2023–2026)
   - *Key Innovation*: Massive self-supervised pre-training using multi-crop SwAV losses, patch-level masked training (iBOT), and untied attention scaling. Delivers universal visual feature representations capable of SOTA linear-probe classification, in-context visual segmentation, and depth estimation without task-specific fine-tuning.
   - [DINOv2 Paper: arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | [Official Code](https://github.com/facebookresearch/dinov2)

2. **SigLIP & SigLIP 2: Sigmoid Loss for Language-Image Pre-Training** (Zhai et al., Google DeepMind, 2023 / 2024)
   - *Key Innovation*: Replaced traditional pairwise softmax cross-entropy with a simple binary sigmoid classification loss. This allows computing image-text similarities independently without requiring huge all-gather mini-batches across hundreds of GPUs, yielding higher zero-shot classification accuracy with half the compute.
   - [Paper: arXiv:2303.15343](https://arxiv.org/abs/2303.15343) | [Official Code](https://github.com/google-research/big_vision)

3. **ConvNeXt V2: Co-designing Network Design and Masked Autoencoders** (Woo et al., Meta / UC Berkeley, 2023 / 2024)
   - *Key Innovation*: Integrated Global Response Normalization (GRN) into modern pure ConvNet architectures and combined it with Fully Convolutional Masked Autoencoders (FCMAE), outperforming traditional Vision Transformers on ImageNet-1K.
   - [Paper: arXiv:2301.00808](https://arxiv.org/abs/2301.00808) | [Official Code](https://github.com/facebookresearch/ConvNeXt-V2)

4. **MobileNetV4: Universal Models for the Mobile Ecosystem** (Qin et al., Google, 2024)
   - *Key Innovation*: Universal Inverted Bottleneck (UIB) search optimizing hardware execution across mobile DSPs, NPUs, CPUs, and GPUs, achieving 82.3% ImageNet Top-1 accuracy at 0.85ms latency.
   - [Paper: arXiv:2404.10518](https://arxiv.org/abs/2404.10518) | [Official Code](https://github.com/huggingface/pytorch-image-models)

---

### Quantitative SOTA Benchmark Comparison (ImageNet-1K)
| Model Architecture | ImageNet-1K Top-1 | Pre-training Paradigm | Latency (FP16 ms) | Hardware Profile | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV4-Conv-Medium**| 82.3% | Supervised (UIB Search) | 0.85 ms | Mobile NPU / RTX 4090 | Apache-2.0 |
| **ConvNeXt V2-Huge** | 88.9% | Self-Supervised FCMAE | 12.40 ms | TensorRT FP16 A100 | Apache-2.0 |
| **DINOv2-Giant (ViT-g/14)**| 86.5% (Linear probe) | Self-Supervised (142M images)| 34.00 ms | Server GPU | Apache-2.0 |
| **SigLIP (ViT-SO400M/14)** | 83.2% (Zero-Shot) | Sigmoid Language-Image Pretrain | 9.80 ms | High-throughput Zero-shot | Apache-2.0 |
| **Swin-V2-L (1536x1536)** | 90.17% (SOTA Heavy) | Supervised Window Transformer | 85.00 ms | Datacenter Cluster | MIT |

---

## Commercial Usability & License Audit
- **Permissive & Commercial-Ready**:
  - **Apache-2.0**:
    - `facebookresearch/dinov2` (Meta Apache-2.0). Safe for proprietary commercial integration, feature extraction embeddings, and inference servers.
    - `facebookresearch/ConvNeXt-V2` (Apache-2.0).
    - `google-research/big_vision` (SigLIP Apache-2.0).
    - `huggingface/pytorch-image-models (timm)`: **Apache-2.0**. Safe for commercial redistribution, model training, and quantization.
  - **MIT License**:
    - `microsoft/Swin-Transformer`: **MIT**.
  - *Recommendation*: **timm** + **ConvNeXt V2** or **SigLIP** provides 100% royalty-free commercial coverage without any copyleft restrictions.

---

## Architecture Alternatives & Trade-offs
| Architecture | Top-1 (ImageNet-1K) | Latency (FP16 / Edge) | Inductive Bias | Best Suited For |
| :--- | :--- | :--- | :--- | :--- |
| **Lightweight ConvNet (MobileNetV4)** | 80 - 83% | <1.0 ms (Edge/Mobile) | Strong translation equivariance | Embedded IoT, low-power drones |
| **Modern Heavy ConvNet (ConvNeXt V2)** | 85 - 89% | 4 - 12 ms (GPU) | Moderate | High-accuracy visual classification servers |
| **Vision Transformer (ViT-H, Swin-B)** | 85 - 89% | 10 - 30 ms (GPU) | Weak (requires large-scale data pre-training) | Multimodal foundation encoders, dense context |
| **Zero-Shot VLM (SigLIP / OpenCLIP)** | 80 - 84% (zero-shot) | 5 - 15 ms (text + image) | Open-vocabulary semantic embedding | Dynamic taxonomy, anomaly filtering |

---

## Popular Repos & Integrations
- **[huggingface/pytorch-image-models (timm)](https://github.com/huggingface/pytorch-image-models)**: The ultimate gold standard for image classification backbones (Apache-2.0).
- **[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)**: Official implementation of DINOv2 self-supervised foundation vision backbones (Apache-2.0).
- **[google-research/big_vision](https://github.com/google-research/big_vision)**: Official codebase for SigLIP, PaLI, and Vision Transformer scaling research (Apache-2.0).
- **Tooling Integrations**:
  - **FiftyOne**: Inspect confusion matrices, visualize t-SNE / UMAP embeddings of intermediate layers, identify mislabeled ground-truth data.
  - **Rerun**: Visualize real-time classification probability distributions as bar charts alongside live video feeds.

---

## End-to-End Pipeline & Workarounds
1. **Pipeline Stages**:
   - Crop / Bounding-box patch extraction -> Aspect-preserving resize -> Mean/std normalization -> Forward pass -> Softmax -> Temperature scaling / calibration.
2. **Common Traps & Edge Cases**:
   - *Overconfidence on OOD Data*: Standard softmax outputs high confidence (e.g. >95%) on completely unseen background noise.
   - *Fine-Grained Feature Loss*: Severe downsampling (e.g. 224x224 standard) strips discriminative textures on small components.
3. **Engineering Workarounds**:
   - **Temperature Scaling & Platt Scaling**: Re-calibrate logit outputs post-hoc on a validation split to ensure probabilities match empirical accuracy.
   - **Mahalanobis Distance / Deep Feature Anomaly Scores**: Extract penultimate layer embeddings and reject inputs exceeding distance thresholds from class centroids.
   - **Test-Time Augmentation (TTA)**: Average predictions across original and flipped/scaled variants for critical low-confidence decisions.

---

## Deployment & Real-time Notes
- **Hardware Acceleration**:
  - Classification backbones are compute-dense and easily reach maximum hardware utilization in TensorRT, ONNX Runtime, and OpenVINO.
- **Batched RoI Classification**:
  - When pairing classification as a secondary stage after an object detector, batch RoI crops into fixed tensor chunks (e.g., batch size 16 or 32) to saturate GPU streaming multiprocessors.
- **Weight Quantization**:
  - Modern ConvNets (MobileNetV4, ResNet) easily support INT8 PTQ (Post-Training Quantization) with <0.5% Top-1 drop, making them ideal for edge execution.

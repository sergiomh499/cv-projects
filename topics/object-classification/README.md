---
title: Object Classification Master Index & Directory
tags:
  - computer-vision
  - classification
  - index
  - sota
  - models
  - playbooks
updated: 2026-09-08
aliases:
  - Object Classification Playbook
  - Classification Index
---

# Object Classification: Master Index & Domain Guide

> **Obsidian Users**: Access the unified Map of Content at [[topics/object-classification/00-object-classification-moc|Object Classification MOC]].

## 📌 Executive Brief
Object Classification categorizes images, cropped regions-of-interest (RoIs), or multi-modal visual tokens into discrete predefined classes. Modern visual classification focuses on **foundation visual representations** (DINOv2/v3), **zero-shot vision-language alignment** (SigLIP), and **hardware-tuned pure ConvNets** (MobileNetV4, ConvNeXt V2) that execute at sub-millisecond speeds.

---

## 🧭 Topic Organization & File Structure

Following the **MOC + Central Architecture Vault** design:

```text
topics/object-classification/
├── 00-object-classification-moc.md               # 🗺️ Master Map of Content & Cross-Domain Hub
├── 01-historical-evolution-and-paradigms.md       # 📜 Historical Evolution (LeNet -> ResNet -> ViT -> DINOv3)
├── 02-production-pipeline-and-workarounds.md      # 🛠️ Production Pipeline, Temperature Calibration & OOD Traps
└── README.md                                     # 🧭 Master Directory Index for Git / Web Browsers

architectures/                                    # 🔬 Shared Multi-Task Architecture Vault
├── foundation-models/
│   ├── dinov2-and-dinov3.md                      # DINOv2 & DINOv3: Foundation Self-Supervised ViT (Apache-2.0)
│   └── siglip.md                                 # SigLIP & SigLIP 2: Sigmoid Loss Vision-Language Encoders (Apache-2.0)
└── real-time-unified/
    └── convnext-and-mobilenet.md                 # ConvNeXt V2 & MobileNetV4: Modern High-Efficiency ConvNets (Apache-2.0)
```

---

## 📚 Navigation Directory

| Note Title | Document Type | Description | Link |
| :--- | :---: | :--- | :---: |
| **Object Classification MOC** | `MOC` | Master navigational hub, benchmark matrix & license audit | [Open MOC](00-object-classification-moc.md) <br> `[[topics/object-classification/00-object-classification-moc|00-object-classification-moc]]` |
| **Historical Evolution & Paradigms** | `Evolution Guide` | Evolution from LeNet to AlexNet, ResNet residual math, ViTs, and DINO self-distillation | [Open Evolution](01-historical-evolution-and-paradigms.md) <br> `[[topics/object-classification/01-historical-evolution-and-paradigms|01-historical-evolution-and-paradigms]]` |
| **Production Pipeline & Workarounds** | `Playbook` | RoI batching, softmax overconfidence, temperature scaling, Mahalanobis OOD detection | [Open Playbook](02-production-pipeline-and-workarounds.md) <br> `[[topics/object-classification/02-production-pipeline-and-workarounds|02-production-pipeline-and-workarounds]]` |
| **DINOv2 & DINOv3 Deep-Dive** | `Architecture Vault` | Multi-crop student-teacher self-distillation, iBOT masked modeling (Apache-2.0) | [Open DINOv2/v3](../../architectures/foundation-models/dinov2-and-dinov3.md) <br> `[[architectures/foundation-models/dinov2-and-dinov3|dinov2-and-dinov3]]` |
| **SigLIP & SigLIP 2 Deep-Dive** | `Architecture Vault` | Pairwise binary sigmoid loss eliminating cross-GPU all-gather bottlenecks (Apache-2.0) | [Open SigLIP](../../architectures/foundation-models/siglip.md) <br> `[[architectures/foundation-models/siglip|siglip]]` |
| **ConvNeXt V2 & MobileNetV4 Deep-Dive** | `Architecture Vault` | Modernized pure ConvNets with Global Response Normalization (GRN) and UIB search | [Open ConvNeXt/MobileNet](../../architectures/real-time-unified/convnext-and-mobilenet.md) <br> `[[architectures/real-time-unified/convnext-and-mobilenet|convnext-and-mobilenet]]` |

---

## 📊 Summary SOTA Benchmark Comparison (ImageNet-1K)
| Model Architecture | Parameters | ImageNet-1K Top-1 | Pre-training Paradigm | Latency (FP16 ms, A100) | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV4-Conv-Medium**| 9.7 M | 82.3% | Supervised (UIB Search) | 0.85 ms (Edge NPU) | Apache-2.0 |
| **ConvNeXt V2-Huge** | 660.0 M | 88.9% (SOTA) | Self-Supervised FCMAE | 12.40 ms | Apache-2.0 |
| **DINOv2-Giant (ViT-g/14)**| 1.1 B | 86.5% (Linear probe) | Self-Supervised (142M images)| 34.00 ms | Apache-2.0 |
| **SigLIP (ViT-SO400M/14)** | 880.0 M | 83.2% (Zero-Shot) | Sigmoid Language-Image Pretrain | 9.80 ms | Apache-2.0 |

---

## ⚖️ Commercial Usability Quick-Audit
- **100% Commercial-Safe (Permissive: Apache-2.0 / MIT)**:
  - `huggingface/pytorch-image-models (timm)`: **Apache-2.0**. Universal production standard for classification backbones.
  - `facebookresearch/dinov2`: **Apache-2.0**.
  - `google-research/big_vision` (SigLIP): **Apache-2.0**.
  - `facebookresearch/ConvNeXt-V2`: **Apache-2.0**.
  - *Recommendation*: The entire classification and foundation backbone ecosystem is royalty-free and safe for closed-source proprietary commercial integration.

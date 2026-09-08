---
title: Object Classification MOC
type: MOC
domain: Object Classification
tags:
  - moc
  - computer-vision
  - classification
  - backbones
  - dinov2
  - siglip
  - convnext
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - Object Classification MOC
  - Classification Hub
  - Classification MOC
---

# 🗺️ Object Classification MOC (Map of Content)

> **Navigation**: [[README|🏠 Central Knowledge Hub]] / **Object Classification
tags:
  - moc
  - computer-vision
  - classification
  - backbones
  - dinov2
  - siglip
  - convnext
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - Object Classification MOC
  - Classification Hub
  - Classification MOC
---

# 🗺️ Object Classification MOC (Map of Content)

## 📌 Domain Overview & Scope
Object Classification categorizes images, cropped regions-of-interest (RoIs), or multi-modal visual tokens into discrete predefined classes. Beyond standard softmax prediction, modern visual classification focuses on:
1. **Foundation Visual Representations**: Self-supervised dense representations that serve as universal backbones across detection, segmentation, and tracking.
2. **Zero-Shot Open-Vocabulary Alignment**: Projecting visual semantics and natural language into a shared embedding space (SigLIP).
3. **Hardware-Tuned ConvNets**: Delivering sub-millisecond classification on edge mobile NPUs and DSPs (MobileNetV4, ConvNeXt V2).

This Map of Content connects the historical evolution, production playbooks, and shared backbone architectures across the classification ecosystem.

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/object-classification/01-historical-evolution-and-paradigms|Object Classification: Historical Lineage & Backbones]]
- **Production Implementation Playbook**: [[topics/object-classification/02-production-pipeline-and-workarounds|Object Classification: Production Pipeline, Traps & Calibration Workarounds]]

### 🔬 Core Architecture Deep-Dives (Central Architecture Vault)
| Model / System | Architecture Class | Primary Innovation | License | Dedicated Deep-Dive Note |
| :--- | :--- | :--- | :---: | :---: |
| **DINOv2 & DINOv3** | Self-Supervised ViT | Multi-crop student-teacher self-distillation with iBOT masked modeling | **Apache-2.0** | [[architectures/vision-foundation-models/dinov2-and-dinov3|DINOv2 & DINOv3 Deep-Dive]] |
| **SigLIP & SigLIP 2**| Vision-Language Dual Encoder | Pairwise binary sigmoid loss eliminating cross-GPU all-gather bottlenecks | **Apache-2.0** | [[architectures/vision-foundation-models/siglip|SigLIP & SigLIP 2 Deep-Dive]] |
| **ConvNeXt V2 & MobileNetV4** | Modernized ConvNet | Global Response Normalization (GRN) and Universal Inverted Bottlenecks (UIB) | **Apache-2.0** | [[architectures/backbones-and-edge-efficiency/convnext-and-mobilenet|ConvNeXt V2 & MobileNetV4 Deep-Dive]] |

---

## 📊 Standardized SOTA Benchmark Comparison (ImageNet-1K)
| Model Architecture | Parameters | ImageNet-1K Top-1 | Pre-training Paradigm | Latency (FP16 ms, A100) | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV4-Conv-Medium**| 9.7 M | 82.3% | Supervised (UIB Search) | 0.85 ms (Edge NPU) | Apache-2.0 |
| **ConvNeXt V2-Huge** | 660.0 M | 88.9% (SOTA) | Self-Supervised FCMAE | 12.40 ms | Apache-2.0 |
| **DINOv2-Giant (ViT-g/14)**| 1.1 B | 86.5% (Linear probe) | Self-Supervised (142M images)| 34.00 ms | Apache-2.0 |
| **SigLIP (ViT-SO400M/14)** | 880.0 M | 83.2% (Zero-Shot) | Sigmoid Language-Image Pretrain | 9.80 ms | Apache-2.0 |

---

## ⚖️ Commercial Usability & License Audit
- **100% Commercial-Safe (Permissive: Apache-2.0 / MIT)**:
  - `huggingface/pytorch-image-models (timm)`: **Apache-2.0**. Universal production standard for classification backbones.
  - `facebookresearch/dinov2`: **Apache-2.0**.
  - `google-research/big_vision` (SigLIP): **Apache-2.0**.
  - `facebookresearch/ConvNeXt-V2`: **Apache-2.0**.
  - *Recommendation*: The entire classification and foundation backbone ecosystem is royalty-free and safe for closed-source proprietary commercial integration.

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/object-detection/00-object-detection-moc|Object Detection MOC]]
- [[topics/object-segmentation/00-object-segmentation-moc|Object Segmentation MOC]]
- [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]]**

## 📌 Domain Overview & Scope
Object Classification categorizes images, cropped regions-of-interest (RoIs), or multi-modal visual tokens into discrete predefined classes. Beyond standard softmax prediction, modern visual classification focuses on:
1. **Foundation Visual Representations**: Self-supervised dense representations that serve as universal backbones across detection, segmentation, and tracking.
2. **Zero-Shot Open-Vocabulary Alignment**: Projecting visual semantics and natural language into a shared embedding space (SigLIP).
3. **Hardware-Tuned ConvNets**: Delivering sub-millisecond classification on edge mobile NPUs and DSPs (MobileNetV4, ConvNeXt V2).

This Map of Content connects the historical evolution, production playbooks, and shared backbone architectures across the classification ecosystem.

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/object-classification/01-historical-evolution-and-paradigms|01 Historical Evolution And Paradigms]]
- **Production Implementation Playbook**: [[topics/object-classification/02-production-pipeline-and-workarounds|02 Production Pipeline And Workarounds]]
- **Modern Architecture & Open Problems**: [[topics/object-classification/03-backends-and-open-problems|03 Backends And Open Problems]]
- **Classical, Geometric & Hybrid Baselines**: [[topics/object-classification/04-classical-and-hybrid-methods|04 Classical And Hybrid Methods]]

### 🔬 Core Architecture Deep-Dives (Central Architecture Vault)
| Model / System | Architecture Class | Primary Innovation | License | Dedicated Deep-Dive Note |
| :--- | :--- | :--- | :---: | :---: |
| **DINOv2 & DINOv3** | Self-Supervised ViT | Multi-crop student-teacher self-distillation with iBOT masked modeling | **Apache-2.0** | [[architectures/vision-foundation-models/dinov2-and-dinov3|DINOv2 & DINOv3 Deep-Dive]] |
| **SigLIP & SigLIP 2**| Vision-Language Dual Encoder | Pairwise binary sigmoid loss eliminating cross-GPU all-gather bottlenecks | **Apache-2.0** | [[architectures/vision-foundation-models/siglip|SigLIP & SigLIP 2 Deep-Dive]] |
| **ConvNeXt V2 & MobileNetV4** | Modernized ConvNet | Global Response Normalization (GRN) and Universal Inverted Bottlenecks (UIB) | **Apache-2.0** | [[architectures/backbones-and-edge-efficiency/convnext-and-mobilenet|ConvNeXt V2 & MobileNetV4 Deep-Dive]] |

---

## 📊 Standardized SOTA Benchmark Comparison (ImageNet-1K)
| Model Architecture | Parameters | ImageNet-1K Top-1 | Pre-training Paradigm | Latency (FP16 ms, A100) | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV4-Conv-Medium**| 9.7 M | 82.3% | Supervised (UIB Search) | 0.85 ms (Edge NPU) | Apache-2.0 |
| **ConvNeXt V2-Huge** | 660.0 M | 88.9% (SOTA) | Self-Supervised FCMAE | 12.40 ms | Apache-2.0 |
| **DINOv2-Giant (ViT-g/14)**| 1.1 B | 86.5% (Linear probe) | Self-Supervised (142M images)| 34.00 ms | Apache-2.0 |
| **SigLIP (ViT-SO400M/14)** | 880.0 M | 83.2% (Zero-Shot) | Sigmoid Language-Image Pretrain | 9.80 ms | Apache-2.0 |

---

## ⚖️ Commercial Usability & License Audit
- **100% Commercial-Safe (Permissive: Apache-2.0 / MIT)**:
  - `huggingface/pytorch-image-models (timm)`: **Apache-2.0**. Universal production standard for classification backbones.
  - `facebookresearch/dinov2`: **Apache-2.0**.
  - `google-research/big_vision` (SigLIP): **Apache-2.0**.
  - `facebookresearch/ConvNeXt-V2`: **Apache-2.0**.
  - *Recommendation*: The entire classification and foundation backbone ecosystem is royalty-free and safe for closed-source proprietary commercial integration.

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/object-detection/00-object-detection-moc|Object Detection MOC]]
- [[topics/object-segmentation/00-object-segmentation-moc|Object Segmentation MOC]]
- [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]]

---

## 📂 All Notes in This Domain

```dataview
TABLE type AS "Note Type", updated AS "Last Updated", status AS "Status"
WHERE contains(file.folder, "topics/object-classification")
SORT file.name ASC
```

---
title: Object Segmentation Master Index & Directory
tags:
  - computer-vision
  - segmentation
  - index
  - sota
  - models
  - playbooks
updated: 2026-09-08
aliases:
  - Object Segmentation Playbook
  - Segmentation Index
---

# Object Segmentation: Master Index & Domain Guide

> **Obsidian Users**: Access the unified Map of Content at [[topics/object-segmentation/00-object-segmentation-moc|Object Segmentation MOC]].

## 📌 Executive Brief
Object Segmentation partitions visual scenes down to individual pixel coordinates across **Semantic Segmentation** (per-pixel class assignment), **Instance Segmentation** (discrete instance delineation), and **Panoptic Segmentation** (unifying background stuff and foreground things). Modern applications demand real-time interactive foundation models, boundary precision, and memory-efficient mask streaming.

---

## 🧭 Topic Organization & File Structure

Following the **MOC + Central Architecture Vault** architecture:

```text
topics/object-segmentation/
├── 00-object-segmentation-moc.md                 # 🗺️ Master Map of Content & Cross-Domain Hub
├── 01-historical-evolution-and-paradigms.md       # 📜 Historical Evolution (Otsu -> U-Net -> Mask R-CNN -> SAM 2)
├── 02-production-pipeline-and-workarounds.md      # 🛠️ Production Pipeline, Memory Traps & PointRend Workarounds
└── README.md                                     # 🧭 Master Directory Index for Git / Web Browsers

architectures/                                    # 🔬 Shared Multi-Task Architecture Vault
├── foundation-models/
│   ├── sam-2.md                                  # SAM 2 & 2.1: Foundation Video & Image Segmentation (Apache-2.0)
│   └── depth-anything-v2.md                      # Depth Anything V2: Metric Depth & Surface Segmentation (Apache-2.0)
└── real-time-unified/
    └── mask2former.md                            # Mask2Former: Universal Query Transformer (Apache-2.0)
```

---

## 📚 Navigation Directory

| Note Title | Document Type | Description | Link |
| :--- | :---: | :--- | :---: |
| **Object Segmentation MOC** | `MOC` | Master navigational hub, benchmark matrix & license audit | [Open MOC](00-object-segmentation-moc.md) <br> `[[topics/object-segmentation/00-object-segmentation-moc|00-object-segmentation-moc]]` |
| **Historical Evolution & Paradigms** | `Evolution Guide` | Historical progression from energy cuts to U-Net, prototype matrices, and SAM 2 | [Open Evolution](01-historical-evolution-and-paradigms.md) <br> `[[topics/object-segmentation/01-historical-evolution-and-paradigms|01-historical-evolution-and-paradigms]]` |
| **Production Pipeline & Workarounds** | `Playbook` | Sensor ingestion, RLE mask compression, PointRend boundary refinement | [Open Playbook](02-production-pipeline-and-workarounds.md) <br> `[[topics/object-segmentation/02-production-pipeline-and-workarounds|02-production-pipeline-and-workarounds]]` |
| **SAM 2 & SAM 2.1 Deep-Dive** | `Architecture Vault` | Spatial-temporal streaming memory bank, 44 FPS video propagation (Apache-2.0) | [Open SAM 2](../../architectures/foundation-models/sam-2.md) <br> `[[architectures/foundation-models/sam-2|sam-2]]` |
| **Depth Anything V2 Deep-Dive** | `Architecture Vault` | Foundation monocular depth and surface segmentation via synthetic distillation | [Open Depth Anything](../../architectures/foundation-models/depth-anything-v2.md) <br> `[[architectures/foundation-models/depth-anything-v2|depth-anything-v2]]` |
| **Mask2Former Deep-Dive** | `Architecture Vault` | Universal masked-attention query transformer for panoptic, semantic, and instance tasks | [Open Mask2Former](../../architectures/real-time-unified/mask2former.md) <br> `[[architectures/real-time-unified/mask2former|mask2former]]` |

---

## 📊 Summary SOTA Benchmark Comparison
| Architecture Model | Task Domain | Benchmark Dataset | Metric | Latency (FP16 ms, A100) | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SAM 2.1 (Hiera-B+)** | Promptable / Video | SA-V Video Benchmark | 75.0 J&F | 22.8 ms (43.8 FPS) | Apache-2.0 |
| **SAM 2.1 (Hiera-Tiny)**| Promptable / Video | SA-V Video Benchmark | 71.5 J&F | 14.7 ms (68.0 FPS) | Apache-2.0 |
| **Mask2Former (Swin-L)** | Panoptic / Semantic | COCO Panoptic / ADE20K | 58.3% PQ / 57.7% mIoU | 185.0 ms | Apache-2.0 |
| **Depth Anything V2-Large**| Metric Depth / Surface | NYUv2 / KITTI | 0.075 Rel Error | 24.2 ms | Apache-2.0 |
| **YOLOv8x-Seg** | Real-Time Instance | COCO val2017 Instance | 43.4% Mask AP | 13.5 ms | AGPL-3.0 / Commercial |

---

## ⚖️ Commercial Usability Quick-Audit
- **Commercial Permissive (Safe for Proprietary Products)**:
  - **Apache-2.0**: `facebookresearch/sam2`, `facebookresearch/Mask2Former`, `DepthAnything/Depth-Anything-V2`, `ChaoningZhang/MobileSAM`. Safe for proprietary software, edge devices, and cloud APIs.
- **Copyleft (Caution Required)**:
  - **AGPL-3.0 (Ultralytics YOLOv8-Seg / YOLO11-Seg)**: Mandates open-sourcing client backends if distributed over a network unless an enterprise license is purchased.
  - *Recommendation*: For commercial projects, use **SAM 2 (Tiny/Small)** for interactive segmentation or **MMSegmentation / DeepLabV3+** for semantic segmentation to stay strictly within Apache-2.0 terms.

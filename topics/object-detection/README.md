---
title: Object Detection Master Index & Playbook
tags:
  - computer-vision
  - object-detection
  - index
  - sota
  - models
  - playbooks
updated: 2026-09-08
aliases:
  - Object Detection Playbook
  - Detection Index
---

# Object Detection: Master Index & Domain Guide

> **Obsidian Users**: Access the unified Map of Content at [[topics/object-detection/00-object-detection-moc|Object Detection MOC]].

## 📌 Executive Brief
Object Detection localizes visual instances with 2D bounding boxes (axis-aligned or oriented) and assigns class probabilities to each. Modern production systems demand sub-10ms latency, zero-shot open-vocabulary capability, robust performance under dense clutter, and deterministic execution profiles without erratic post-processing delays.

---

## 🧭 Topic Organization & File Structure

This topic is organized into dedicated, modular Obsidian notes following the **MOC + Zettelkasten** methodology:

```text
topics/object-detection/
├── 00-object-detection-moc.md                    # 🗺️ Master Map of Content & Cross-Domain Hub
├── 01-historical-evolution-and-paradigms.md       # 📜 Historical Evolution (Haar -> Faster R-CNN -> YOLOv1-26 -> DETR)
├── 02-production-pipeline-and-workarounds.md      # 🛠️ Production Pipeline, Engineering Traps & SAHI Workarounds
└── models/                                       # 🔬 In-Depth Model Architectures
    ├── rf-detr.md                                 # RF-DETR: NAS-Optimized Real-Time Transformers with DINOv2
    ├── rt-detr.md                                 # RT-DETRv2 & RT-DETRv3: Hybrid Encoders & Dense Positive Supervision
    ├── yolo-lineage.md                            # Modern YOLO Lineage: YOLOv10, YOLO11, and Flagship YOLO26
    └── grounding-dino.md                          # Grounding DINO: Open-Vocabulary Zero-Shot Detection
```

---

## 📚 Navigation Directory

| Note Title | Document Type | Description | Link |
| :--- | :---: | :--- | :---: |
| **Object Detection MOC** | `MOC` | Master navigational hub, benchmark matrix & license audit | [Open MOC](00-object-detection-moc.md) <br> `[[topics/object-detection/00-object-detection-moc|00-object-detection-moc]]` |
| **Historical Evolution & Paradigms** | `Evolution Guide` | Complete didactic breakdown across the 3 historical eras with 4 Mermaid diagrams | [Open Evolution](01-historical-evolution-and-paradigms.md) <br> `[[topics/object-detection/01-historical-evolution-and-paradigms|01-historical-evolution-and-paradigms]]` |
| **Production Pipeline & Workarounds** | `Playbook` | Sensor ingestion, TensorRT pipelines, SAHI tiling & CUDA streaming workarounds | [Open Playbook](02-production-pipeline-and-workarounds.md) <br> `[[topics/object-detection/02-production-pipeline-and-workarounds|02-production-pipeline-and-workarounds]]` |
| **RF-DETR Deep-Dive** | `Model Guide` | DINOv2 distillation, SuperNet NAS, sub-6ms latency (Apache-2.0) | [Open RF-DETR](models/rf-detr.md) <br> `[[topics/object-detection/models/rf-detr|models/rf-detr]]` |
| **RT-DETRv2 / v3 Deep-Dive** | `Model Guide` | CCFM hybrid encoders, operator cleanup, hierarchical supervision (Apache-2.0) | [Open RT-DETR](models/rt-detr.md) <br> `[[topics/object-detection/models/rt-detr|models/rt-detr]]` |
| **YOLO Lineage Deep-Dive** | `Model Guide` | YOLOv10, YOLO11, and flagship YOLO26; dual-label assignments, AGPL-3.0 audit | [Open YOLO](models/yolo-lineage.md) <br> `[[topics/object-detection/models/yolo-lineage|models/yolo-lineage]]` |
| **Grounding DINO Deep-Dive** | `Model Guide` | Multi-modal cross-attention, zero-shot open-vocabulary queries (Apache-2.0) | [Open Grounding DINO](models/grounding-dino.md) <br> `[[topics/object-detection/models/grounding-dino|models/grounding-dino]]` |

---

## 📊 Summary SOTA Benchmark Comparison (COCO val2017)
| Model Architecture | Parameters | FLOPs | AP (0.50:0.95) | TensorRT Latency (FP16 ms) | NMS Required | Primary Open License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RF-DETR-Base** | 28.4 M | 86.0 G | 53.8% | 5.20 ms | No | Apache-2.0 |
| **RT-DETRv3-L** | 31.8 M | 108.0 G | 54.3% | 8.80 ms | No | Apache-2.0 |
| **RT-DETRv2-X** | 67.0 M | 234.0 G | 55.1% | 13.60 ms | No | Apache-2.0 |
| **YOLO26-S** | 7.4 M | 22.1 G | 47.9% | 2.30 ms | No | AGPL-3.0 / Commercial |
| **YOLOv10-X** | 29.5 M | 160.4 G | 54.4% | 10.70 ms | No | AGPL-3.0 / Commercial |
| **Grounding DINO (Swin-T)**| 172.0 M | 340.0 G | 52.5% (Zero-Shot)| 78.00 ms | Yes | Apache-2.0 |
| **Co-Deformable-DETR (Swin-L)**| 217.0 M | 1280.0 G | 66.0% (SOTA Heavy)| 145.00 ms | Yes | Apache-2.0 |

---

## ⚖️ Commercial Usability Quick-Audit
- **Commercial Permissive (Safe for Closed Products)**:
  - **Apache-2.0**: `roboflow/rf-detr`, `lyuwenyu/RT-DETR` (v1/v2), `clxia12/RT-DETRv3`, `IDEA-Research/GroundingDINO`. Safe for proprietary software and embedded binaries without revealing source code.
- **Copyleft Warning (Requires Careful Review)**:
  - **AGPL-3.0 (Ultralytics YOLOv8 / YOLO11 / YOLO26)**: Mandates open-sourcing client backends if distributed or hosted over a network unless an enterprise commercial license is purchased.
  - *Recommendation*: For commercial projects, standardize on **RF-DETR** or **RT-DETRv2/v3** (Apache-2.0).

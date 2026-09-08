---
title: Object Detection MOC
type: MOC
domain: Object Detection
tags:
  - moc
  - computer-vision
  - object-detection
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - Object Detection MOC
  - Object Detection Hub
  - Detection MOC
---

# 🗺️ Object Detection MOC (Map of Content)

## 📌 Domain Overview & Scope
Object Detection localizes visual instances with 2D bounding boxes (axis-aligned or oriented) and assigns class probabilities to each. Modern production systems demand sub-10ms latency, zero-shot open-vocabulary capability, robust performance under dense clutter, and deterministic execution profiles without erratic post-processing delays.

This Map of Content connects the historical evolution, production playbooks, and individual model deep-dives across the object detection landscape.

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/object-detection/01-historical-evolution-and-paradigms|Object Detection: Historical Lineage & Paradigms]]
- **Production Implementation Playbook**: [[topics/object-detection/02-production-pipeline-and-workarounds|Object Detection: Production Pipeline, Traps & Workarounds]]

### 🔬 In-Depth Model & Architecture Notes
| Model / System | Architecture Class | Primary Innovation | License | Dedicated Deep-Dive Note |
| :--- | :--- | :--- | :---: | :---: |
| **RF-DETR** | Transformer + DINOv2 | Weight-sharing NAS over DINOv2 backbone for real-time edge | **Apache-2.0** | [[topics/object-detection/models/rf-detr|RF-DETR Deep-Dive]] |
| **RT-DETRv2 / v3**| Transformer (CCFM) | NMS-free, hybrid encoder, hierarchical dense positive supervision | **Apache-2.0** | [[topics/object-detection/models/rt-detr|RT-DETR Deep-Dive]] |
| **YOLO Lineage** | Real-Time CNN | Evolution from YOLOv1 to YOLOv10, YOLO11, and flagship YOLO26 | **AGPL-3.0** | [[topics/object-detection/models/yolo-lineage|YOLO Lineage Deep-Dive]] |
| **Grounding DINO** | Vision-Language OVD | 3-stage visual-text cross-attention for zero-shot open-vocabulary | **Apache-2.0** | [[topics/object-detection/models/grounding-dino|Grounding DINO Deep-Dive]] |

---

## 📊 Standardized SOTA Benchmark Comparison (COCO val2017)
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

## ⚖️ Commercial Usability & License Matrix
- **Commercial Permissive (Safe for Closed Products)**:
  - **Apache-2.0**: `roboflow/rf-detr`, `lyuwenyu/RT-DETR` (v1/v2), `clxia12/RT-DETRv3`, `IDEA-Research/GroundingDINO`. Safe for proprietary software and embedded binaries without revealing source code.
- **Copyleft Warning (Requires Careful Review)**:
  - **AGPL-3.0 (Ultralytics YOLOv8 / YOLO11 / YOLO26)**: Mandates open-sourcing client backends if distributed or hosted over a network unless an enterprise commercial license is purchased.
  - *Recommendation*: For commercial projects, standardize on **RF-DETR** or **RT-DETRv2/v3** (Apache-2.0).

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/object-segmentation/00-object-segmentation-moc|Object Segmentation MOC]]
- [[topics/video-tracking/00-video-tracking-moc|Video Tracking MOC]]
- [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]]
- [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]]

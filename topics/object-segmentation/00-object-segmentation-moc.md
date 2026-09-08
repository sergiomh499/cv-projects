---
title: Object Segmentation MOC
type: MOC
domain: Object Segmentation
tags:
  - moc
  - computer-vision
  - segmentation
  - semantic-segmentation
  - instance-segmentation
  - panoptic-segmentation
  - sam2
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - Object Segmentation MOC
  - Segmentation Hub
  - Segmentation MOC
---

# 🗺️ Object Segmentation MOC (Map of Content)

> **Navigation**: [[README|🏠 Central Knowledge Hub]] / **Object Segmentation
tags:
  - moc
  - computer-vision
  - segmentation
  - semantic-segmentation
  - instance-segmentation
  - panoptic-segmentation
  - sam2
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - Object Segmentation MOC
  - Segmentation Hub
  - Segmentation MOC
---

# 🗺️ Object Segmentation MOC (Map of Content)

## 📌 Domain Overview & Scope
Object Segmentation partitions visual scenes down to individual pixel coordinates. It branches into:
1. **Semantic Segmentation**: Assigns a semantic class category (e.g., road, sky, vehicle) to every pixel in the image.
2. **Instance Segmentation**: Detects and delineates individual object boundaries (e.g., Person #1, Person #2).
3. **Panoptic Segmentation**: Unifies background**

## 📌 Domain Overview & Scope
Object Segmentation partitions visual scenes down to individual pixel coordinates. It branches into:
1. **Semantic Segmentation**: Assigns a semantic class category (e.g., road, sky, vehicle) to every pixel in the image.
2. **Instance Segmentation**: Detects and delineates individual object boundaries (e.g., Person #1, Person #2).
3. **Panoptic Segmentation**: Unifies background "stuff" and foreground "things" into an exhaustive per-pixel representation.

This Map of Content connects the historical evolution, production playbooks, and shared cross-cutting model architectures across the segmentation ecosystem.

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/object-segmentation/01-historical-evolution-and-paradigms|01 Historical Evolution And Paradigms]]
- **Production Implementation Playbook**: [[topics/object-segmentation/02-production-pipeline-and-workarounds|02 Production Pipeline And Workarounds]]
- **Modern Architecture & Open Problems**: [[topics/object-segmentation/03-decoders-and-open-problems|03 Decoders And Open Problems]]
- **Classical, Geometric & Hybrid Baselines**: [[topics/object-segmentation/04-classical-and-hybrid-methods|04 Classical And Hybrid Methods]]

### 🔬 Core Architecture Deep-Dives (Central Architecture Vault)
| Model / System | Architecture Class | Primary Innovation | License | Dedicated Deep-Dive Note |
| :--- | :--- | :--- | :---: | :---: |
| **SAM 2 & SAM 2.1** | Foundation Model (Hiera ViT) | Streaming spatial-temporal memory bank, 44 FPS video propagation | **Apache-2.0** | [[architectures/foundation-models/sam-2|SAM 2 & 2.1 Deep-Dive]] |
| **Depth Anything V2** | Foundation DINOv2 Distillation | Metric-scale monocular depth & surface segmentation via synthetic distillation | **Apache-2.0** | [[architectures/foundation-models/depth-anything-v2|Depth Anything V2 Deep-Dive]] |
| **Mask2Former** | Universal Query Transformer | Masked cross-attention unifying semantic, instance, and panoptic segmentation | **Apache-2.0** | [[architectures/real-time-unified/mask2former|Mask2Former Deep-Dive]] |
| **YOLO-Seg Lineage** | Prototype Matrix CNN | Real-time proto-mask matrix multiplication at 60–120 FPS | **AGPL-3.0** | [[topics/object-detection/models/yolo-lineage|YOLO-Seg Deep-Dive]] |

---

## 📊 Standardized SOTA Benchmark Comparison
| Architecture Model | Task Domain | Benchmark Dataset | Metric | Latency (FP16 ms, A100) | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SAM 2.1 (Hiera-B+)** | Promptable / Video | SA-V Video Benchmark | 75.0 J&F | 22.8 ms (43.8 FPS) | Apache-2.0 |
| **SAM 2.1 (Hiera-Tiny)**| Promptable / Video | SA-V Video Benchmark | 71.5 J&F | 14.7 ms (68.0 FPS) | Apache-2.0 |
| **Mask2Former (Swin-L)** | Panoptic / Semantic | COCO Panoptic / ADE20K | 58.3% PQ / 57.7% mIoU | 185.0 ms | Apache-2.0 |
| **Depth Anything V2-Large**| Metric Depth / Surface | NYUv2 / KITTI | 0.075 Rel Error | 24.2 ms | Apache-2.0 |
| **YOLOv8x-Seg** | Real-Time Instance | COCO val2017 Instance | 43.4% Mask AP | 13.5 ms | AGPL-3.0 / Commercial |

---

## ⚖️ Commercial Usability & License Audit
- **Commercial Permissive (Safe for Proprietary Products)**:
  - **Apache-2.0**: `facebookresearch/sam2`, `facebookresearch/Mask2Former`, `DepthAnything/Depth-Anything-V2`, `ChaoningZhang/MobileSAM`. Safe for proprietary software, edge devices, and cloud APIs.
- **Copyleft (Caution Required)**:
  - **AGPL-3.0 (Ultralytics YOLOv8-Seg / YOLO11-Seg)**: Mandates open-sourcing client backends if distributed over a network unless an enterprise license is purchased.
  - *Recommendation*: For commercial projects, use **SAM 2 (Tiny/Small)** for interactive segmentation or **MMSegmentation / DeepLabV3+** for semantic segmentation to stay strictly within Apache-2.0 terms.

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/object-detection/00-object-detection-moc|Object Detection MOC]]
- [[topics/video-tracking/00-video-tracking-moc|Video Tracking MOC]]
- [[topics/sensor-fusion/00-sensor-fusion-moc|Sensor Fusion MOC]]
- [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]]

---

## 📂 All Notes in This Domain

```dataview
TABLE type AS "Note Type", updated AS "Last Updated", status AS "Status"
WHERE contains(file.folder, "topics/object-segmentation")
SORT file.name ASC
```

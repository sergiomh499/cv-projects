---
title: Computer Vision & Perception Projects Knowledge Hub
tags:
  - computer-vision
  - perception
  - deep-learning
  - sota
  - obsidian-vault
  - licenses
  - architecture-evolution
  - index
updated: 2026-09-08
aliases:
  - Hub Index
  - CV Projects
---

# Computer Vision & Perception Projects Knowledge Hub

A curated, didactic engineering repository and cookbook covering latest research, historical evolutions, end-to-end pipelines, trade-off analyses, and production deployment strategies for **Computer Vision (CV)**, **Machine Learning (ML)**, **Sensor Fusion**, and **Real-Time Systems**.

Focuses on rigorous, state-of-the-art architectures (2023–2026) with demonstrable performance, verified open-source codebases, comprehensive license audits for commercial product readiness, and multi-backend acceleration across **NVIDIA CUDA / TensorRT**, **Vulkan Compute**, and **FPGA (AMD/Xilinx Vitis AI & FINN)**. Fully indexed and compatible with **Obsidian** for visual graph exploration and bidirectional knowledge browsing.

---

## 🧭 Repository Structure & Obsidian Map

```text
cv-projects/
├── topics/                     # Comprehensive domain playbooks & historical evolutions
│   ├── object-detection/       # README.md (Playbook) & EVOLUTION.md (YOLOv1->YOLO26 & DETR)
│   ├── object-segmentation/    # README.md (Playbook) & EVOLUTION.md (U-Net, SAM 2, Depth V2)
│   ├── object-classification/  # README.md (Playbook) & EVOLUTION.md (ResNet to DINOv3, SigLIP)
│   ├── video-tracking/         # README.md (Playbook) & EVOLUTION.md (SORT, ByteTrack, CoTracker)
│   ├── 6dof-pose-estimation/   # README.md (Playbook) & EVOLUTION.md (PnP, Symmetry, FoundationPose)
│   ├── lidar-perception/       # README.md (Playbook) & EVOLUTION.md (PointNet, SpConv, DSVT)
│   ├── sensor-fusion/          # README.md (Playbook) & EVOLUTION.md (Late, Early, BEVFusion, UniAD)
│   ├── fpga-deployment/        # README.md (Playbook) & EVOLUTION.md (DPU vs Streaming Dataflow)
│   ├── gpu-deployment/         # README.md (Playbook) & EVOLUTION.md (Memory Wall & FlashAttention)
│   └── real-time-systems/      # README.md (Playbook) & EVOLUTION.md (Zero-Copy Iceoryx 2 & RT)
├── templates/
│   └── topic-template.md       # Standard playbook schema and contribution guide
├── resources/
│   └── ecosystem-tools.md      # FiftyOne, Rerun, CVAT, OpenMMLab, inference runtimes
├── scripts/
│   └── validate_repo.py        # Automated repository integrity, SOTA & license validator
├── pyproject.toml              # UV-managed Python environment, dependencies & tools
└── .pre-commit-config.yaml     # Pre-commit hooks (ruff, commitzen, formatters)
```

---

## 📚 Playbooks, Historical Guides & Topic Catalog

| # | Topic Domain | SOTA Models (2023–2026) | Commercial License Audit | Production Playbook | Historical Evolution & Architecture Guide |
| :---: | :--- | :--- | :--- | :---: | :---: |
| 1 | **Object Detection** | RF-DETR, RT-DETRv2/v3, YOLOv10/YOLO26 | Apache-2.0 (RF-DETR, RT-DETR), AGPL-3.0 (YOLO) | [Playbook](topics/object-detection/README.md) | [Evolution & Lineage](topics/object-detection/EVOLUTION.md) |
| 2 | **Object Segmentation** | SAM 2/2.1, Depth Anything V2, Mask2Former | Apache-2.0 (SAM 2, Mask2Former, Depth Anything) | [Playbook](topics/object-segmentation/README.md) | [Evolution & Lineage](topics/object-segmentation/EVOLUTION.md) |
| 3 | **Object Classification** | DINOv3/v2, SigLIP 2, ConvNeXt V2, MobileNetV4 | Apache-2.0 (timm, DINOv2, SigLIP, ConvNeXt) | [Playbook](topics/object-classification/README.md) | [Evolution & Lineage](topics/object-classification/EVOLUTION.md) |
| 4 | **Video Tracking** | CoTracker3, BoT-SORT, ByteTrack, TAPIR | MIT (BoT-SORT, ByteTrack), Apache-2.0 (CoTracker) | [Playbook](topics/video-tracking/README.md) | [Evolution & Lineage](topics/video-tracking/EVOLUTION.md) |
| 5 | **6-DoF Pose Estimation** | FoundationPose, MegaPose, GDR-Net | Apache-2.0 (MegaPose, GDR-Net), Non-Commercial (FoundationPose) | [Playbook](topics/6dof-pose-estimation/README.md) | [Evolution & Lineage](topics/6dof-pose-estimation/EVOLUTION.md) |
| 6 | **LiDAR Perception** | DSVT, FlatFormer, CenterPoint, SpConv 2.x | Apache-2.0 (OpenPCDet, FlatFormer, SpConv) | [Playbook](topics/lidar-perception/README.md) | [Evolution & Lineage](topics/lidar-perception/EVOLUTION.md) |
| 7 | **Sensor Fusion** | BEVFusion, Sparse4D v3, UniAD, PointPainting | Apache-2.0 (BEVFusion, Sparse4D, UniAD) | [Playbook](topics/sensor-fusion/README.md) | [Evolution & Lineage](topics/sensor-fusion/EVOLUTION.md) |
| 8 | **FPGA Deployment** | Vitis AI 3.5 DPU, FINN v0.10, Brevitas QAT | Apache-2.0 (Vitis-AI, FINN, Brevitas) | [Playbook](topics/fpga-deployment/README.md) | [Evolution & Lineage](topics/fpga-deployment/EVOLUTION.md) |
| 9 | **GPU Deployment** | TensorRT 10, FlashAttention-2/3, Vulkan NCNN | Apache-2.0 (TensorRT), BSD-3 (Triton, NCNN) | [Playbook](topics/gpu-deployment/README.md) | [Evolution & Lineage](topics/gpu-deployment/EVOLUTION.md) |
| 10 | **Real-Time Systems** | Iceoryx 2 (Rust), ROS 2 Jazzy, PREEMPT_RT | Apache-2.0 / MIT (Iceoryx 2, ROS 2) | [Playbook](topics/real-time-systems/README.md) | [Evolution & Lineage](topics/real-time-systems/EVOLUTION.md) |

---

## 🛠️ Commercial Product Readiness & Licensing Quick-Guide

When developing commercial computer vision products, licensing compliance is paramount to avoid proprietary IP contamination:

- **100% Commercial-Safe (Permissive: Apache-2.0, MIT, BSD-3-Clause)**:
  - Can be embedded into closed-source binaries, cloud SaaS backends, and firmware without forced source-code disclosure.
  - *Recommended Stack*: **RF-DETR / RT-DETRv2** (Detection), **SAM 2 / Depth Anything V2** (Segmentation), **timm + ConvNeXt V2** (Classification), **ByteTrack / BoT-SORT** (Tracking), **MegaPose** (6-DoF), **OpenPCDet** (LiDAR), **MIT BEVFusion** (Sensor Fusion), **TensorRT / NCNN** (GPU), **Iceoryx 2** (Real-Time IPC).
- **Copyleft (AGPL-3.0, GPL-3.0)**:
  - Requires open-sourcing the entire caller product under the same license if distributed or accessed over a network.
  - *Watch Out*: **Ultralytics YOLOv8/v10/YOLO26** (AGPL-3.0), **BoxMOT** (GPL-3.0). Use with an enterprise commercial license or choose permissive alternatives listed above.
- **Non-Commercial / Research-Only**:
  - Prohibits all revenue-generating or product usage.
  - *Watch Out*: **FoundationPose** (NVIDIA Source Code License - Research Only). Use **MegaPose** (Apache-2.0) instead for commercial robotics.

---

## 🧪 Repository Verification

Run the comprehensive validator checking relative links, section schema, modern SOTA citations, benchmark tables, and commercial license audits:

```bash
python scripts/validate_repo.py
```

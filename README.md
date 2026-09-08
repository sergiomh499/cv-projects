---
title: Computer Vision & Perception Projects Knowledge Hub
tags:
  - computer-vision
  - perception
  - deep-learning
  - sota
  - obsidian-vault
  - licenses
  - index
updated: 2026-09-08
aliases:
  - Hub Index
  - CV Projects
---

# Computer Vision & Perception Projects Knowledge Hub

A curated, didactic engineering repository and cookbook covering latest research, end-to-end pipelines, trade-off analyses, and production deployment strategies for **Computer Vision (CV)**, **Machine Learning (ML)**, **Sensor Fusion**, and **Real-Time Systems**.

Focuses on rigorous, state-of-the-art architectures (2023–2026) with demonstrable performance, verified open-source codebases, comprehensive license audits for commercial product readiness, and multi-backend acceleration across **NVIDIA CUDA / TensorRT**, **Vulkan Compute**, and **FPGA (AMD/Xilinx Vitis AI & FINN)**. Fully indexed and compatible with **Obsidian** for visual graph exploration and bidirectional knowledge browsing.

---

## 🧭 Repository Structure & Obsidian Map

```text
cv-projects/
├── topics/                     # Comprehensive domain playbooks (Obsidian notes)
│   ├── object-detection/       # RF-DETR, RT-DETRv2/v3, YOLOv10, Grounding DINO, Co-DETR
│   ├── object-segmentation/    # SAM 2/2.1, Depth Anything V2, Mask2Former, MobileSAM
│   ├── object-classification/  # DINOv3/v2, SigLIP/SigLIP 2, ConvNeXt V2, MobileNetV4, timm
│   ├── video-tracking/         # CoTracker3, BoT-SORT, ByteTrack, TAPIR, SAM 2 Video
│   ├── 6dof-pose-estimation/   # FoundationPose, MegaPose, GDR-Net, BOP Benchmark
│   ├── lidar-perception/       # DSVT, FlatFormer, CenterPoint, SpConv 2.x
│   ├── sensor-fusion/          # BEVFusion, Sparse4D v3, UniAD, PTP Time-Sync
│   ├── fpga-deployment/        # Vitis AI 3.5 DPU, FINN v0.10, Brevitas QAT, LUTNet
│   ├── gpu-deployment/         # TensorRT 10, FlashAttention-2/3, Vulkan NCNN, Triton Server
│   └── real-time-systems/      # Iceoryx 2 (Rust), ROS 2 Jazzy, PREEMPT_RT, Lock-Free IPC
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

## 📚 Playbooks & Topic Catalog

| # | Topic | Core SOTA Architectures (2023–2026) | Commercial License Audit | Playbook Link (Obsidian & Git) |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Object Detection** | RF-DETR, RT-DETRv2/v3, YOLOv10, Grounding DINO | Apache-2.0 (RF-DETR, RT-DETR), AGPL-3.0 (YOLO) | [Read Playbook](topics/object-detection/README.md) <br> `[[topics/object-detection/README\|Object Detection]]` |
| 2 | **Object Segmentation** | SAM 2/2.1, Depth Anything V2, Mask2Former, MobileSAM | Apache-2.0 (SAM 2, Mask2Former, Depth Anything) | [Read Playbook](topics/object-segmentation/README.md) <br> `[[topics/object-segmentation/README\|Object Segmentation]]` |
| 3 | **Object Classification** | DINOv3/v2, SigLIP 2, ConvNeXt V2, MobileNetV4 | Apache-2.0 (timm, DINOv2, SigLIP, ConvNeXt) | [Read Playbook](topics/object-classification/README.md) <br> `[[topics/object-classification/README\|Object Classification]]` |
| 4 | **Video Tracking** | CoTracker3, BoT-SORT, ByteTrack, TAPIR, SAM 2 | MIT (BoT-SORT, ByteTrack), Apache-2.0 (CoTracker) | [Read Playbook](topics/video-tracking/README.md) <br> `[[topics/video-tracking/README\|Video Tracking]]` |
| 5 | **6-DoF Pose Estimation** | FoundationPose, MegaPose, GDR-Net, BOP Benchmark | Apache-2.0 (MegaPose, GDR-Net), Non-Commercial (FoundationPose) | [Read Playbook](topics/6dof-pose-estimation/README.md) <br> `[[topics/6dof-pose-estimation/README\|6-DoF Pose]]` |
| 6 | **LiDAR Perception** | DSVT, FlatFormer, CenterPoint, SpConv 2.x | Apache-2.0 (OpenPCDet, FlatFormer, SpConv) | [Read Playbook](topics/lidar-perception/README.md) <br> `[[topics/lidar-perception/README\|LiDAR Perception]]` |
| 7 | **Sensor Fusion** | BEVFusion, Sparse4D v3, UniAD, PointPainting | Apache-2.0 (BEVFusion, Sparse4D, UniAD) | [Read Playbook](topics/sensor-fusion/README.md) <br> `[[topics/sensor-fusion/README\|Sensor Fusion]]` |
| 8 | **FPGA Deployment** | Vitis AI 3.5 DPU, FINN v0.10, Brevitas QAT | Apache-2.0 (Vitis-AI, FINN, Brevitas) | [Read Playbook](topics/fpga-deployment/README.md) <br> `[[topics/fpga-deployment/README\|FPGA Deployment]]` |
| 9 | **GPU Deployment** | TensorRT 10, FlashAttention-2/3, Vulkan NCNN | Apache-2.0 (TensorRT), BSD-3 (Triton, NCNN) | [Read Playbook](topics/gpu-deployment/README.md) <br> `[[topics/gpu-deployment/README\|GPU Deployment]]` |
| 10 | **Real-Time Systems** | Iceoryx 2 (Rust), ROS 2 Jazzy, PREEMPT_RT | Apache-2.0 / MIT (Iceoryx 2, ROS 2) | [Read Playbook](topics/real-time-systems/README.md) <br> `[[topics/real-time-systems/README\|Real-Time Systems]]` |

---

## 🛠️ Commercial Product Readiness & Licensing Quick-Guide

When developing commercial computer vision products, licensing compliance is paramount to avoid proprietary IP contamination:

- **100% Commercial-Safe (Permissive: Apache-2.0, MIT, BSD-3-Clause)**:
  - Can be embedded into closed-source binaries, cloud SaaS backends, and firmware without forced source-code disclosure.
  - *Recommended Stack*: **RF-DETR / RT-DETRv2** (Detection), **SAM 2 / Depth Anything V2** (Segmentation), **timm + ConvNeXt V2** (Classification), **ByteTrack / BoT-SORT** (Tracking), **MegaPose** (6-DoF), **OpenPCDet** (LiDAR), **MIT BEVFusion** (Sensor Fusion), **TensorRT / NCNN** (GPU), **Iceoryx 2** (Real-Time IPC).
- **Copyleft (AGPL-3.0, GPL-3.0)**:
  - Requires open-sourcing the entire caller product under the same license if distributed or accessed over a network.
  - *Watch Out*: **Ultralytics YOLOv8/v10** (AGPL-3.0), **BoxMOT** (GPL-3.0). Use with an enterprise commercial license or choose permissive alternatives listed above.
- **Non-Commercial / Research-Only**:
  - Prohibits all revenue-generating or product usage.
  - *Watch Out*: **FoundationPose** (NVIDIA Source Code License - Research Only). Use **MegaPose** (Apache-2.0) instead for commercial robotics.

---

## 🧪 Repository Verification

Run the comprehensive validator checking relative links, section schema, modern SOTA citations, benchmark tables, and commercial license audits:

```bash
python scripts/validate_repo.py
```

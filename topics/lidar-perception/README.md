---
title: LiDAR Perception Master Index & Directory
tags:
  - computer-vision
  - lidar
  - index
  - sota
  - models
  - playbooks
updated: 2026-09-08
aliases:
  - LiDAR Perception Playbook
  - LiDAR Index
---

# LiDAR Perception: Master Index & Domain Guide

> **Obsidian Users**: Access the unified Map of Content at [[topics/lidar-perception/00-lidar-perception-moc|LiDAR Perception MOC]].

## 📌 Executive Brief
LiDAR perception processes high-bandwidth 3D point cloud streams (2.5M points/sec) to provide metric 3D object detection, semantic segmentation, and SLAM/odometry invariant to lighting variations. Modern paradigms have advanced from hand-crafted voxelization to **Dynamic Sparse Window Transformers** ([[architectures/3d-pointclouds-and-lidar/dsvt-and-flatformer|DSVT & FlatFormer]]) deployable natively on TensorRT without non-standard SpConv operations.

---

## 🧭 Topic Organization & File Structure

```text
topics/lidar-perception/
├── 00-lidar-perception-moc.md                    # 🗺️ Master Map of Content & Cross-Domain Hub
├── 01-historical-evolution-and-paradigms.md       # 📜 Historical Evolution (PointNet -> SpConv -> PointPillars -> DSVT)
├── 02-production-pipeline-and-workarounds.md      # 🛠️ Production Pipeline, Deskewing Traps & AF_XDP Workarounds
└── README.md                                     # 🧭 Master Directory Index for Git / Web Browsers

architectures/                                    # 🔬 Shared Multi-Task Architecture Vault
└── real-time-unified/
    └── dsvt-and-flatformer.md                    # DSVT & FlatFormer: Dynamic Sparse Window Attention for 3D LiDAR (Apache-2.0)
```

---

## 📚 Navigation Directory

| Note Title | Document Type | Description | Link |
| :--- | :---: | :--- | :---: |
| **LiDAR Perception MOC** | `MOC` | Master navigational hub, benchmark matrix & license audit | [Open MOC](00-lidar-perception-moc.md) <br> `[[topics/lidar-perception/00-lidar-perception-moc|00-lidar-perception-moc]]` |
| **Historical Evolution & Paradigms** | `Evolution Guide` | Evolution from PointNet symmetric pooling to Submanifold SpConv, 2D Pillars, and dynamic sparse window transformers | [Open Evolution](01-historical-evolution-and-paradigms.md) <br> `[[topics/lidar-perception/01-historical-evolution-and-paradigms|01-historical-evolution-and-paradigms]]` |
| **Production Pipeline & Workarounds** | `Playbook` | AF_XDP zero-copy networking, IMU motion deskewing, rain backscatter filtering | [Open Playbook](02-production-pipeline-and-workarounds.md) <br> `[[topics/lidar-perception/02-production-pipeline-and-workarounds|02-production-pipeline-and-workarounds]]` |
| **DSVT & FlatFormer Deep-Dive** | `Architecture Vault` | Dynamic sparse window attention compiling natively to TensorRT without SpConv | [Open DSVT/FlatFormer](../../architectures/3d-pointclouds-and-lidar/dsvt-and-flatformer.md) <br> `[[architectures/3d-pointclouds-and-lidar/dsvt-and-flatformer|dsvt-and-flatformer]]` |

---

## 📊 Summary SOTA Benchmark Comparison (Waymo Open Dataset L2)
| Model Architecture | WOD Vehicle (mAP / APH L2) | WOD Pedestrian (mAP / APH L2) | Latency (FP16 ms, A100) | TensorRT Deployable | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PointPillars (Baseline)** | 63.8 / 63.1 | 58.2 / 51.5 | 14.5 ms | Native | Apache-2.0 |
| **CenterPoint (Voxel)** | 71.8 / 71.2 | 72.1 / 67.0 | 45.0 ms | Requires SpConv | Apache-2.0 |
| **FlatFormer** | 77.2 / 76.7 | 78.5 / 74.8 | 21.0 ms | **Native** | Apache-2.0 |
| **DSVT (Voxel Transformer)**| **78.9 / 78.4** (SOTA) | **81.8 / 78.2** (SOTA) | 27.0 ms | **Native** | Apache-2.0 |

---

## ⚖️ Commercial Usability Quick-Audit
- **100% Commercial Permissive (Apache-2.0)**:
  - `Haiyang-W/DSVT`: **Apache-2.0**.
  - `open-mmlab/OpenPCDet`: **Apache-2.0**.
  - *Recommendation*: Standardize on DSVT or FlatFormer for production stacks to ensure native TensorRT execution without proprietary custom kernel dependencies.

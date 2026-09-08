---
title: SLAM & Spatial Perception Master Index & Directory
tags:
  - computer-vision
  - slam
  - index
  - sota
  - models
  - playbooks
updated: 2026-09-08
aliases:
  - SLAM Playbook
  - Spatial Perception Index
---

# SLAM & Spatial Perception: Master Index & Domain Guide

> **Obsidian Users**: Access the unified Map of Content at [[topics/slam-and-spatial-perception/00-slam-and-spatial-perception-moc|SLAM & Spatial Perception MOC]].

## 📌 Executive Brief
Simultaneous Localization and Mapping (SLAM) allows autonomous agents to navigate unknown 3D environments by jointly estimating 6-DoF trajectories and reconstructing environmental geometry. The domain has expanded from classical sparse keypoint optimization to **real-time differentiable 3D Gaussian Splatting SLAM** ([[architectures/spatial-radiance-and-slam/3dgs-slam-and-monogs|3DGS SLAM & MonoGS]]) delivering dense, photo-realistic maps at 30 FPS.

---

## 🧭 Topic Organization & File Structure

```text
topics/slam-and-spatial-perception/
├── 00-slam-and-spatial-perception-moc.md          # 🗺️ Master Map of Content & Cross-Domain Hub
├── 01-historical-evolution-and-paradigms.md       # 📜 Historical Evolution (EKF -> PTAM -> ORB-SLAM3 -> 3DGS SLAM)
├── 02-production-pipeline-and-workarounds.md      # 🛠️ Production Pipeline, Scale Drift & Loop Closure Workarounds
└── README.md                                     # 🧭 Master Directory Index for Git / Web Browsers

architectures/                                    # 🔬 Shared Multi-Task Architecture Vault
└── real-time-unified/
    └── 3dgs-slam-and-monogs.md                   # 3D Gaussian Splatting SLAM & MonoGS: Radiance Field Odometry (Apache-2.0 / MIT)
```

---

## 📚 Navigation Directory

| Note Title | Document Type | Description | Link |
| :--- | :---: | :--- | :---: |
| **SLAM & Spatial Perception MOC** | `MOC` | Master navigational hub, benchmark matrix & license audit | [Open MOC](00-slam-and-spatial-perception-moc.md) <br> `[[topics/slam-and-spatial-perception/00-slam-and-spatial-perception-moc|00-slam-and-spatial-perception-moc]]` |
| **Historical Evolution & Paradigms** | `Evolution Guide` | Progression from EKF-SLAM to PTAM thread separation, DSO direct tracking, and 3D Gaussian splatting | [Open Evolution](01-historical-evolution-and-paradigms.md) <br> `[[topics/slam-and-spatial-perception/01-historical-evolution-and-paradigms|01-historical-evolution-and-paradigms]]` |
| **Production Pipeline & Workarounds** | `Playbook` | IMU continuous pre-integration, monocular scale drift fixes, degeneracy eigenvalue checks | [Open Playbook](02-production-pipeline-and-workarounds.md) <br> `[[topics/slam-and-spatial-perception/02-production-pipeline-and-workarounds|02-production-pipeline-and-workarounds]]` |
| **3DGS SLAM & MonoGS Deep-Dive** | `Architecture Vault` | Differentiable photometric Gaussian optimization for dense real-time spatial odometry | [Open 3DGS SLAM](../../architectures/spatial-radiance-and-slam/3dgs-slam-and-monogs.md) <br> `[[architectures/spatial-radiance-and-slam/3dgs-slam-and-monogs|3dgs-slam-and-monogs]]` |

---

## 📊 Summary SOTA Benchmark Comparison
| SLAM Framework | Sensor Modality | Map Output | Trajectory ATE RMSE (cm) | Real-Time Frame Rate | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ORB-SLAM3** | Visual-Inertial RGB | Sparse Keypoints | **1.2 cm** | 45 FPS | GPL-3.0 |
| **FAST-LIO2** | LiDAR-Inertial | Metric Voxel Grid | **1.5 cm** | 100+ Hz | GPL-2.0 |
| **NICE-SLAM** | Neural NeRF RGB-D | Continuous Implicit | 2.8 cm | 1.5 FPS | Apache-2.0 |
| **MonoGS** | Monocular RGB | Dense 3D Gaussians | 1.8 cm | 28.0 FPS | **Apache-2.0** |

---

## ⚖️ Commercial Usability Quick-Audit
- **Commercial Permissive (Apache-2.0 / MIT)**:
  - `muskie82/MonoGS` & `spla-tam/SplaTAM`: **Apache-2.0 / MIT**.
- **Commercial Warning**:
  - `UZ-SLAMLab/ORB_SLAM3`: **GPL-3.0**. Requires commercial license from the University of Zaragoza for closed-source proprietary distribution.

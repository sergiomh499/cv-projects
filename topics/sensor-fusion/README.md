---
title: Sensor Fusion Master Index & Directory
tags:
  - computer-vision
  - sensor-fusion
  - index
  - sota
  - models
  - playbooks
updated: 2026-09-08
aliases:
  - Sensor Fusion Playbook
  - Sensor Fusion Index
---

# Sensor Fusion: Master Index & Domain Guide

> **Obsidian Users**: Access the unified Map of Content at [[topics/sensor-fusion/00-sensor-fusion-moc|Sensor Fusion MOC]].

## 📌 Executive Brief
Sensor Fusion unifies complementary physical sensing modalities—Cameras (dense semantics), LiDAR (metric 3D geometry), and Radar (direct Doppler velocity)—to overcome the physical failure modes of any single sensor. The domain has evolved to unified **Bird's-Eye-View (BEV) multi-modal transformers** ([[architectures/3d-pointclouds-and-lidar/bevfusion|BEVFusion & Sparse4D]]) executing under 45 ms.

---

## 🧭 Topic Organization & File Structure

```text
topics/sensor-fusion/
├── 00-sensor-fusion-moc.md                       # 🗺️ Master Map of Content & Cross-Domain Hub
├── 01-historical-evolution-and-paradigms.md       # 📜 Historical Evolution (Late EKF -> PointPainting -> LSS -> BEVFusion)
├── 02-production-pipeline-and-workarounds.md      # 🛠️ Production Pipeline, IEEE 1588 PTP & Safety Fallbacks
└── README.md                                     # 🧭 Master Directory Index for Git / Web Browsers

architectures/                                    # 🔬 Shared Multi-Task Architecture Vault
└── real-time-unified/
    └── bevfusion-and-sparse4d.md                 # BEVFusion & Sparse4D: Multi-Modal Camera-LiDAR Transformers (Apache-2.0)
```

---

## 📚 Navigation Directory

| Note Title | Document Type | Description | Link |
| :--- | :---: | :--- | :---: |
| **Sensor Fusion MOC** | `MOC` | Master navigational hub, benchmark matrix & license audit | [Open MOC](00-sensor-fusion-moc.md) <br> `[[topics/sensor-fusion/00-sensor-fusion-moc|00-sensor-fusion-moc]]` |
| **Historical Evolution & Paradigms** | `Evolution Guide` | Progression from late EKF tracking to PointPainting, Lift-Splat-Shoot (LSS), and BEVFusion | [Open Evolution](01-historical-evolution-and-paradigms.md) <br> `[[topics/sensor-fusion/01-historical-evolution-and-paradigms|01-historical-evolution-and-paradigms]]` |
| **Production Pipeline & Workarounds** | `Playbook` | IEEE 1588 PTP microsecond sync, thermal calibration drift, ISO 26262 safety fallbacks | [Open Playbook](02-production-pipeline-and-workarounds.md) <br> `[[topics/sensor-fusion/02-production-pipeline-and-workarounds|02-production-pipeline-and-workarounds]]` |
| **BEVFusion & Sparse4D Deep-Dive** | `Architecture Vault` | Multi-modal BEV perception with hardware-accelerated coordinate caching and 4D anchor queries | [Open BEVFusion/Sparse4D](../../architectures/3d-pointclouds-and-lidar/bevfusion-and-sparse4d.md) <br> `[[architectures/3d-pointclouds-and-lidar/bevfusion|bevfusion-and-sparse4d]]` |

---

## 📊 Summary SOTA Benchmark Comparison (nuScenes 3D Detection)
| Model Architecture | Modality | nuScenes NDS | nuScenes mAP | Latency (ms, RTX 3090) | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PointPainting (Early)** | Camera + LiDAR | 61.5% | 54.1% | 120.0 ms | Apache-2.0 |
| **BEVFormer (Camera)** | 6x Cameras Only | 56.9% | 48.1% | 130.0 ms | Apache-2.0 |
| **Sparse4D v3 (Camera)**| 6x Cameras Only | 61.2% | 51.5% | 32.0 ms | **Apache-2.0** |
| **BEVFusion (MIT)** | Camera + LiDAR | **72.9%** (SOTA) | **70.2%** (SOTA) | **41.0 ms** | **Apache-2.0** |

---

## ⚖️ Commercial Usability Quick-Audit
- **100% Commercial Permissive (Apache-2.0 / MIT)**:
  - `mit-han-lab/bevfusion`: **Apache-2.0**. Safe for commercial automotive fleets and industrial robotics.
  - `HorizonRobotics/Sparse4D`: **Apache-2.0**.

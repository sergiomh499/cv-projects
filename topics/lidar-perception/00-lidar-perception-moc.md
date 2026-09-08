---
title: LiDAR Perception MOC
type: MOC
domain: LiDAR Perception
tags:
  - moc
  - computer-vision
  - lidar
  - point-clouds
  - 3d-detection
  - autonomous-driving
  - dsvt
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - LiDAR Perception MOC
  - LiDAR Hub
  - 3D Point Cloud MOC
---

# 🗺️ LiDAR Perception MOC (Map of Content)

## 📌 Domain Overview & Scope
LiDAR (Light Detection and Ranging) systems emit pulsed laser beams to capture precise 3D spatial geometry ($\approx 2.5\text{M points/sec}$) invariant to ambient lighting and shadows. LiDAR perception forms the spatial backbone of autonomous driving, mobile robotics, and aerial surveying. 

Key technical requirements include:
- Ingesting multi-gigabit/s UDP raw packet point streams under strict latency bounds ($<100\text{ ms}$).
- Structuring unordered, sparse 3D point sets into voxels or pillar columns.
- Modern sparse window attention architectures ([[architectures/real-time-unified/dsvt-and-flatformer|DSVT & FlatFormer]]) deployable via TensorRT without custom CUDA non-standard layers.

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/lidar-perception/01-historical-evolution-and-paradigms|LiDAR Perception: Historical Evolution & Paradigms]]
- **Production Implementation Playbook**: [[topics/lidar-perception/02-production-pipeline-and-workarounds|LiDAR Perception: Production Pipeline, Traps & Ring Buffer Workarounds]]

### 🔬 Core Architecture Deep-Dives (Central Vault)
| Model / System | Voxelization Paradigm | Primary Innovation | License | Dedicated Note Link |
| :--- | :--- | :--- | :---: | :---: |
| **DSVT & FlatFormer** | Dynamic Sparse Window Transformer | Replaces SpConv with hardware-friendly dense window attention | **Apache-2.0** | [[architectures/real-time-unified/dsvt-and-flatformer\|DSVT & FlatFormer Deep-Dive]] |
| **OpenPCDet Toolbox** | Unified 3D Framework | Industry-standard modular codebase for point cloud models | **Apache-2.0** | [[resources/ecosystem-tools\|Ecosystem Tools]] |

---

## 📊 Standardized SOTA Benchmark Comparison (Waymo Open Dataset L2)
| Model Architecture | WOD Vehicle (mAP / APH L2) | WOD Pedestrian (mAP / APH L2) | Latency (FP16 ms, A100) | TensorRT Deployable | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PointPillars (Baseline)** | 63.8 / 63.1 | 58.2 / 51.5 | 14.5 ms | Native | Apache-2.0 |
| **CenterPoint (Voxel)** | 71.8 / 71.2 | 72.1 / 67.0 | 45.0 ms | Requires SpConv | Apache-2.0 |
| **FlatFormer** | 77.2 / 76.7 | 78.5 / 74.8 | 21.0 ms | **Native** | Apache-2.0 |
| **DSVT (Voxel Transformer)**| **78.9 / 78.4** (SOTA) | **81.8 / 78.2** (SOTA) | 27.0 ms | **Native** | Apache-2.0 |

---

## ⚖️ Commercial Usability & License Audit
- **100% Commercial Permissive (Apache-2.0)**:
  - `Haiyang-W/DSVT`: **Apache-2.0**.
  - `open-mmlab/OpenPCDet`: **Apache-2.0**.
  - `traveller59/spconv`: **Apache-2.0**.
  - *Recommendation*: Standardize on DSVT or FlatFormer for new autonomous projects to ensure clean native TensorRT deployment without reliance on proprietary external plugins.

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/sensor-fusion/00-sensor-fusion-moc|Sensor Fusion MOC]]
- [[topics/6dof-pose-estimation/00-6dof-pose-estimation-moc|6-DoF Pose Estimation MOC]]
- [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]]

---
title: SLAM & Spatial Perception MOC
type: MOC
domain: SLAM & Spatial Perception
tags:
  - moc
  - computer-vision
  - slam
  - odometry
  - 3dgs
  - point-clouds
  - robotics
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - SLAM MOC
  - Spatial Perception Hub
  - 3DGS SLAM MOC
---

# 🗺️ SLAM & Spatial Perception MOC (Map of Content)

> **Navigation**: [[README|🏠 Central Knowledge Hub]] / **SLAM & Spatial Perception
tags:
  - moc
  - computer-vision
  - slam
  - odometry
  - 3dgs
  - point-clouds
  - robotics
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - SLAM MOC
  - Spatial Perception Hub
  - 3DGS SLAM MOC
---

# 🗺️ SLAM & Spatial Perception MOC (Map of Content)

## 📌 Domain Overview & Scope
Simultaneous Localization and Mapping (SLAM) allows autonomous platforms (mobile robots, drones, AR headsets, and self-driving vehicles) to build a geometric map of an unknown environment while estimating their own 6-DoF trajectory within that map.

Key evolutionary paradigms include:
- **Sparse Feature & Direct Visual SLAM**: ORB-SLAM3, DSO, and bundle adjustment.
- **LiDAR SLAM**: Direct iterative closest point (FAST-LIO2, LIO-SAM) paired with high-rate IMU pre-integration.
- **Radiance Field SLAM (3DGS / NeRF)**: Dense, photo-realistic mapping via differentiable 3D Gaussian splatting ([[architectures/spatial-radiance-and-slam/3dgs-slam|3DGS SLAM & MonoGS]]).

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/slam-and-spatial-perception/01-historical-evolution-and-paradigms|SLAM: Historical Lineage & Paradigms]]
- **Production Implementation Playbook**: [[topics/slam-and-spatial-perception/02-production-pipeline-and-workarounds|SLAM: Production Pipeline, Scale Drift Traps & Loop Closure Workarounds]]

### 🔬 Core Architecture Deep-Dives (Central Vault)
| Model / Framework | SLAM Modality | Primary Innovation | License | Dedicated Note Link |
| :--- | :--- | :--- | :---: | :---: |
| **3DGS SLAM & MonoGS** | Visual-Dense 3DGS | Photorealistic dense map optimization via differentiable Gaussian rasterization | **Apache-2.0 / MIT** | [[architectures/spatial-radiance-and-slam/3dgs-slam|3DGS SLAM & MonoGS]] |
| **FAST-LIO2** | Direct LiDAR-Inertial | Incremental k-d tree (ik-d tree) updating sparse points without re-building | **GPL-2.0 / Academic**| [[topics/lidar-perception/00-lidar-perception-moc|LiDAR MOC]] |

---

## 📊 Standardized SOTA Benchmark Comparison (Replica & KITTI)
| SLAM Framework | Sensor Modality | Map Output | Trajectory ATE RMSE (cm) | Real-Time Frame Rate | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ORB-SLAM3** | Visual-Inertial RGB | Sparse Keypoints | **1.2 cm** | 45 FPS | GPL-3.0 |
| **FAST-LIO2** | LiDAR-Inertial | Metric Voxel Grid | **1.5 cm** | 100+ Hz | GPL-2.0 |
| **NICE-SLAM** | Neural NeRF RGB-D | Continuous Implicit | 2.8 cm | 1.5 FPS | Apache-2.0 |
| **MonoGS** | Monocular RGB | Dense 3D Gaussians | 1.8 cm | 28.0 FPS | **Apache-2.0** |

---

## ⚖️ Commercial Usability & License Audit
- **Commercial Permissive (Apache-2.0 / MIT)**:
  - `muskie82/MonoGS` & `spla-tam/SplaTAM`: **Apache-2.0 / MIT**.
  - `UZ-SLAMLab/ORB_SLAM3`: **GPL-3.0** (Caution: requires purchasing commercial license for closed-source software).
  - *Recommendation*: For commercial robotics, standardize on Apache-2.0 / BSD visual odometry engines (e.g., OpenVINS, MonoGS) or purchase commercial exceptions.

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/lidar-perception/00-lidar-perception-moc|LiDAR Perception MOC]]
- [[topics/sensor-fusion/00-sensor-fusion-moc|Sensor Fusion MOC]]
- [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]]**

## 📌 Domain Overview & Scope
Simultaneous Localization and Mapping (SLAM) allows autonomous platforms (mobile robots, drones, AR headsets, and self-driving vehicles) to build a geometric map of an unknown environment while estimating their own 6-DoF trajectory within that map.

Key evolutionary paradigms include:
- **Sparse Feature & Direct Visual SLAM**: ORB-SLAM3, DSO, and bundle adjustment.
- **LiDAR SLAM**: Direct iterative closest point (FAST-LIO2, LIO-SAM) paired with high-rate IMU pre-integration.
- **Radiance Field SLAM (3DGS / NeRF)**: Dense, photo-realistic mapping via differentiable 3D Gaussian splatting ([[architectures/spatial-radiance-and-slam/3dgs-slam|3DGS SLAM & MonoGS]]).

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/slam-and-spatial-perception/01-historical-evolution-and-paradigms|01 Historical Evolution And Paradigms]]
- **Production Implementation Playbook**: [[topics/slam-and-spatial-perception/02-production-pipeline-and-workarounds|02 Production Pipeline And Workarounds]]
- **Modern Architecture & Open Problems**: [[topics/slam-and-spatial-perception/03-frontends-and-open-problems|03 Frontends And Open Problems]]
- **Classical, Geometric & Hybrid Baselines**: [[topics/slam-and-spatial-perception/04-classical-and-hybrid-methods|04 Classical And Hybrid Methods]]

### 🔬 Core Architecture Deep-Dives (Central Vault)
| Model / Framework | SLAM Modality | Primary Innovation | License | Dedicated Note Link |
| :--- | :--- | :--- | :---: | :---: |
| **3DGS SLAM & MonoGS** | Visual-Dense 3DGS | Photorealistic dense map optimization via differentiable Gaussian rasterization | **Apache-2.0 / MIT** | [[architectures/spatial-radiance-and-slam/3dgs-slam|3DGS SLAM & MonoGS]] |
| **FAST-LIO2** | Direct LiDAR-Inertial | Incremental k-d tree (ik-d tree) updating sparse points without re-building | **GPL-2.0 / Academic**| [[topics/lidar-perception/00-lidar-perception-moc|LiDAR MOC]] |

---

## 📊 Standardized SOTA Benchmark Comparison (Replica & KITTI)
| SLAM Framework | Sensor Modality | Map Output | Trajectory ATE RMSE (cm) | Real-Time Frame Rate | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ORB-SLAM3** | Visual-Inertial RGB | Sparse Keypoints | **1.2 cm** | 45 FPS | GPL-3.0 |
| **FAST-LIO2** | LiDAR-Inertial | Metric Voxel Grid | **1.5 cm** | 100+ Hz | GPL-2.0 |
| **NICE-SLAM** | Neural NeRF RGB-D | Continuous Implicit | 2.8 cm | 1.5 FPS | Apache-2.0 |
| **MonoGS** | Monocular RGB | Dense 3D Gaussians | 1.8 cm | 28.0 FPS | **Apache-2.0** |

---

## ⚖️ Commercial Usability & License Audit
- **Commercial Permissive (Apache-2.0 / MIT)**:
  - `muskie82/MonoGS` & `spla-tam/SplaTAM`: **Apache-2.0 / MIT**.
  - `UZ-SLAMLab/ORB_SLAM3`: **GPL-3.0** (Caution: requires purchasing commercial license for closed-source software).
  - *Recommendation*: For commercial robotics, standardize on Apache-2.0 / BSD visual odometry engines (e.g., OpenVINS, MonoGS) or purchase commercial exceptions.

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/lidar-perception/00-lidar-perception-moc|LiDAR Perception MOC]]
- [[topics/sensor-fusion/00-sensor-fusion-moc|Sensor Fusion MOC]]
- [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]]

---

## 📂 All Notes in This Domain

```dataview
TABLE type AS "Note Type", updated AS "Last Updated", status AS "Status"
WHERE contains(file.folder, "topics/slam-and-spatial-perception")
SORT file.name ASC
```

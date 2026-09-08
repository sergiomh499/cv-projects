---
title: 6-DoF Pose Estimation MOC
type: MOC
domain: 6-DoF Pose Estimation
tags:
  - moc
  - computer-vision
  - 6dof-pose
  - robotics
  - cad-tracking
  - bop-challenge
  - foundationpose
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - 6-DoF Pose Estimation MOC
  - Pose Estimation Hub
  - 6-DoF MOC
---

# 🗺️ 6-DoF Pose Estimation MOC (Map of Content)

> **Navigation**: [[README|🏠 Central Knowledge Hub]] / **6-DoF Pose Estimation
tags:
  - moc
  - computer-vision
  - 6dof-pose
  - robotics
  - cad-tracking
  - bop-challenge
  - foundationpose
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - 6-DoF Pose Estimation MOC
  - Pose Estimation Hub
  - 6-DoF MOC
---

# 🗺️ 6-DoF Pose Estimation MOC (Map of Content)

## 📌 Domain Overview & Scope
6-Degrees-of-Freedom (6-DoF) Pose Estimation determines the complete 3D position ($X, Y, Z$) and 3D orientation ($\text{roll, pitch, yaw} \in SO(3)$) of objects relative to a calibrated sensor frame. It is the core perception capability required for:
- Autonomous industrial robotic bin-picking and precision assembly.
- Surgical tool tracking and robotic surgery guidance.
- Augmented reality virtual asset rendering and spatial anchoring.

The field has transitioned from per-object dense keypoint predictors (PVNet, DenseFusion) to zero-shot foundation models driven by 3D CAD meshes ([[architectures/pose-and-robotics-manipulation/foundationpose-and-megapose|FoundationPose & MegaPose]]).

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/6dof-pose-estimation/01-historical-evolution-and-paradigms|6-DoF Pose: Historical Evolution & Paradigms]]
- **Production Implementation Playbook**: [[topics/6dof-pose-estimation/02-production-pipeline-and-workarounds|6-DoF Pose: Production Pipeline, Symmetry Traps & PnP Workarounds]]

### 🔬 Core Architecture Deep-Dives (Central Vault)
| Model / System | Architecture Class | Primary Innovation | License | Dedicated Note Link |
| :--- | :--- | :--- | :---: | :---: |
| **FoundationPose** | Foundation 3D ViT | Zero-shot pose estimation and 32 ms tracking from untextured CAD mesh | ⚠️ **Non-Commercial** | [[architectures/pose-and-robotics-manipulation/foundationpose-and-megapose|FoundationPose Deep-Dive]] |
| **MegaPose** | Iterative Refinement Network | Multi-view render-and-compare transformer for novel object categories | **Apache-2.0** | [[architectures/pose-and-robotics-manipulation/foundationpose-and-megapose|MegaPose Deep-Dive]] |

---

## 📊 Standardized SOTA Benchmark Comparison (BOP Challenge)
| Model Architecture | Paradigm | YCB-V (ADD-S AUC) | Linemod-Occluded (ADD-0.1d) | Latency (ms) | Commercial License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PVNet (Classic)** | Per-Object Instance | 72.8% | 40.8% | 80.0 ms | Apache-2.0 |
| **GDR-Net** | Per-Object Direct | 91.6% | 62.2% | 45.0 ms | Apache-2.0 |
| **MegaPose** | Zero-Shot CAD-based | 88.5% | 71.0% | 150.0 ms | **Apache-2.0** |
| **FoundationPose** | Zero-Shot Foundation | **96.2%** (SOTA) | **89.5%** (SOTA) | **32.0 ms** | ⚠️ Non-Commercial (NVIDIA) |

---

## ⚖️ Commercial Usability & License Audit
- **Commercial Safe (Permissive: Apache-2.0)**:
  - `facebookresearch/megapose`: **Apache-2.0**. Recommended for enterprise warehouse bin-picking and industrial robotics without legal risk.
  - `ethnhe/pvnet`: **MIT / Apache-2.0**.
- **Commercial Prohibited**:
  - `NVlabs/FoundationPose`: **NVIDIA Non-Commercial Source License**. Commercial deployment requires a paid enterprise agreement from NVIDIA.

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/object-detection/00-object-detection-moc|Object Detection MOC]]
- [[topics/lidar-perception/00-lidar-perception-moc|LiDAR Perception MOC]]
- [[topics/sensor-fusion/00-sensor-fusion-moc|Sensor Fusion MOC]]**

## 📌 Domain Overview & Scope
6-Degrees-of-Freedom (6-DoF) Pose Estimation determines the complete 3D position ($X, Y, Z$) and 3D orientation ($\text{roll, pitch, yaw} \in SO(3)$) of objects relative to a calibrated sensor frame. It is the core perception capability required for:
- Autonomous industrial robotic bin-picking and precision assembly.
- Surgical tool tracking and robotic surgery guidance.
- Augmented reality virtual asset rendering and spatial anchoring.

The field has transitioned from per-object dense keypoint predictors (PVNet, DenseFusion) to zero-shot foundation models driven by 3D CAD meshes ([[architectures/pose-and-robotics-manipulation/foundationpose-and-megapose|FoundationPose & MegaPose]]).

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/6dof-pose-estimation/01-historical-evolution-and-paradigms|01 Historical Evolution And Paradigms]]
- **Production Implementation Playbook**: [[topics/6dof-pose-estimation/02-production-pipeline-and-workarounds|02 Production Pipeline And Workarounds]]
- **Modern Architecture & Open Problems**: [[topics/6dof-pose-estimation/03-solvers-and-open-problems|03 Solvers And Open Problems]]
- **Classical, Geometric & Hybrid Baselines**: [[topics/6dof-pose-estimation/04-classical-and-hybrid-methods|04 Classical And Hybrid Methods]]

### 🔬 Core Architecture Deep-Dives (Central Vault)
| Model / System | Architecture Class | Primary Innovation | License | Dedicated Note Link |
| :--- | :--- | :--- | :---: | :---: |
| **FoundationPose** | Foundation 3D ViT | Zero-shot pose estimation and 32 ms tracking from untextured CAD mesh | ⚠️ **Non-Commercial** | [[architectures/pose-and-robotics-manipulation/foundationpose-and-megapose|FoundationPose Deep-Dive]] |
| **MegaPose** | Iterative Refinement Network | Multi-view render-and-compare transformer for novel object categories | **Apache-2.0** | [[architectures/pose-and-robotics-manipulation/foundationpose-and-megapose|MegaPose Deep-Dive]] |

---

## 📊 Standardized SOTA Benchmark Comparison (BOP Challenge)
| Model Architecture | Paradigm | YCB-V (ADD-S AUC) | Linemod-Occluded (ADD-0.1d) | Latency (ms) | Commercial License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PVNet (Classic)** | Per-Object Instance | 72.8% | 40.8% | 80.0 ms | Apache-2.0 |
| **GDR-Net** | Per-Object Direct | 91.6% | 62.2% | 45.0 ms | Apache-2.0 |
| **MegaPose** | Zero-Shot CAD-based | 88.5% | 71.0% | 150.0 ms | **Apache-2.0** |
| **FoundationPose** | Zero-Shot Foundation | **96.2%** (SOTA) | **89.5%** (SOTA) | **32.0 ms** | ⚠️ Non-Commercial (NVIDIA) |

---

## ⚖️ Commercial Usability & License Audit
- **Commercial Safe (Permissive: Apache-2.0)**:
  - `facebookresearch/megapose`: **Apache-2.0**. Recommended for enterprise warehouse bin-picking and industrial robotics without legal risk.
  - `ethnhe/pvnet`: **MIT / Apache-2.0**.
- **Commercial Prohibited**:
  - `NVlabs/FoundationPose`: **NVIDIA Non-Commercial Source License**. Commercial deployment requires a paid enterprise agreement from NVIDIA.

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/object-detection/00-object-detection-moc|Object Detection MOC]]
- [[topics/lidar-perception/00-lidar-perception-moc|LiDAR Perception MOC]]
- [[topics/sensor-fusion/00-sensor-fusion-moc|Sensor Fusion MOC]]

---

## 📂 All Notes in This Domain

```dataview
TABLE type AS "Note Type", updated AS "Last Updated", status AS "Status"
WHERE contains(file.folder, "topics/6dof-pose-estimation")
SORT file.name ASC
```

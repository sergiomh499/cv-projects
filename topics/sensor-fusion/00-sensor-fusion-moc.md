---
title: Sensor Fusion MOC
type: MOC
domain: Sensor Fusion
tags:
  - moc
  - computer-vision
  - sensor-fusion
  - multi-modal
  - bevfusion
  - sparse4d
  - nuscenes
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - Sensor Fusion MOC
  - Sensor Fusion Hub
  - Multi-Modal MOC
---

# 🗺️ Sensor Fusion MOC (Map of Content)

> **Navigation**: [[README|🏠 Central Knowledge Hub]] / **Sensor Fusion
tags:
  - moc
  - computer-vision
  - sensor-fusion
  - multi-modal
  - bevfusion
  - sparse4d
  - nuscenes
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - Sensor Fusion MOC
  - Sensor Fusion Hub
  - Multi-Modal MOC
---

# 🗺️ Sensor Fusion MOC (Map of Content)

## 📌 Domain Overview & Scope
Sensor Fusion unifies complementary physical sensing modalities to overcome the fundamental vulnerabilities of individual sensors:
- **Cameras**: Dense visual texture, fine color, and semantic classification (degraded by nighttime darkness, direct sun glare, and inclement weather).
- **LiDAR**: Metric 3D spatial geometry and ranging invariant to ambient light (degraded by airborne backscatter in fog and heavy rain).
- **Radar**: Direct Doppler velocity vector measurements and penetration through adverse weather.

The field has evolved from heuristic Extended Kalman Filters (EKF) and late track fusion to end-to-end **Bird**

## 📌 Domain Overview & Scope
Sensor Fusion unifies complementary physical sensing modalities to overcome the fundamental vulnerabilities of individual sensors:
- **Cameras**: Dense visual texture, fine color, and semantic classification (degraded by nighttime darkness, direct sun glare, and inclement weather).
- **LiDAR**: Metric 3D spatial geometry and ranging invariant to ambient light (degraded by airborne backscatter in fog and heavy rain).
- **Radar**: Direct Doppler velocity vector measurements and penetration through adverse weather.

The field has evolved from heuristic Extended Kalman Filters (EKF) and late track fusion to end-to-end **Bird's-Eye-View (BEV) multi-modal transformers** ([[architectures/real-time-unified/bevfusion-and-sparse4d|BEVFusion & Sparse4D]]).

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/sensor-fusion/01-historical-evolution-and-paradigms|01 Historical Evolution And Paradigms]]
- **Production Implementation Playbook**: [[topics/sensor-fusion/02-production-pipeline-and-workarounds|02 Production Pipeline And Workarounds]]
- **Modern Architecture & Open Problems**: [[topics/sensor-fusion/03-uniad-and-open-problems|03 Uniad And Open Problems]]
- **Classical, Geometric & Hybrid Baselines**: [[topics/sensor-fusion/04-classical-and-hybrid-methods|04 Classical And Hybrid Methods]]

### 🔬 Core Architecture Deep-Dives (Central Vault)
| Model / System | Fusion Paradigm | Primary Innovation | License | Dedicated Note Link |
| :--- | :--- | :--- | :---: | :---: |
| **BEVFusion (MIT)** | Multi-Modal BEV Transformer | Fast GPU BEV pooling with pre-computed coordinate caching ($8.8\times$ faster) | **Apache-2.0** | [[architectures/real-time-unified/bevfusion-and-sparse4d|BEVFusion Deep-Dive]] |
| **Sparse4D** | Sparse Temporal Anchor Queries | Replaces dense 3D grids with adaptive 4D anchor queries through space/time | **Apache-2.0** | [[architectures/real-time-unified/bevfusion-and-sparse4d|Sparse4D Deep-Dive]] |

---

## 📊 Standardized SOTA Benchmark Comparison (nuScenes 3D Detection)
| Model Architecture | Modality | nuScenes NDS | nuScenes mAP | Latency (ms, RTX 3090) | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PointPainting (Early)** | Camera + LiDAR | 61.5% | 54.1% | 120.0 ms | Apache-2.0 |
| **BEVFormer (Camera)** | 6x Cameras Only | 56.9% | 48.1% | 130.0 ms | Apache-2.0 |
| **Sparse4D v3 (Camera)**| 6x Cameras Only | 61.2% | 51.5% | 32.0 ms | **Apache-2.0** |
| **BEVFusion (MIT)** | Camera + LiDAR | **72.9%** (SOTA) | **70.2%** (SOTA) | **41.0 ms** | **Apache-2.0** |

---

## ⚖️ Commercial Usability & License Audit
- **100% Commercial Permissive (Apache-2.0 / MIT)**:
  - `mit-han-lab/bevfusion`: **Apache-2.0**.
  - `HorizonRobotics/Sparse4D`: **Apache-2.0**.
  - *Recommendation*: Standardize on BEVFusion or Sparse4D for enterprise robotics and autonomous vehicle stacks.

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/object-detection/00-object-detection-moc|Object Detection MOC]]
- [[topics/lidar-perception/00-lidar-perception-moc|LiDAR Perception MOC]]
- [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]]

---

## 📂 All Notes in This Domain

```dataview
TABLE type AS "Note Type", updated AS "Last Updated", status AS "Status"
WHERE contains(file.folder, "topics/sensor-fusion")
SORT file.name ASC
```

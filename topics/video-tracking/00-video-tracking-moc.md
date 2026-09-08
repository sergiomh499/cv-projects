---
title: Video Tracking MOC
type: MOC
domain: Video Tracking
tags:
  - moc
  - computer-vision
  - video-tracking
  - mot
  - cotracker
  - botsort
  - bytetrack
  - sam2
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - Video Tracking MOC
  - Tracking Hub
  - Tracking MOC
---

# 🗺️ Video Tracking MOC (Map of Content)

> **Navigation**: [[README|🏠 Central Knowledge Hub]] / **Video Tracking
tags:
  - moc
  - computer-vision
  - video-tracking
  - mot
  - cotracker
  - botsort
  - bytetrack
  - sam2
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - Video Tracking MOC
  - Tracking Hub
  - Tracking MOC
---

# 🗺️ Video Tracking MOC (Map of Content)

## 📌 Domain Overview & Scope
Video Tracking maintains consistent spatial, semantic, and instance identity across continuous image sequences despite severe occlusions, viewpoint changes, motion blur, and camera ego-motion. The domain has bifurcated into two major operational paradigms:
1. **Real-Time Multiple Object Tracking (MOT)**: Tracking-by-detection pairing high-frame-rate 2D detectors with lightweight CPU Kalman filters and appearance ReID matching ([[architectures/visual-tracking-and-flow/botsort-and-bytetrack|BoT-SORT & ByteTrack]]).
2. **Dense Physical Trajectory & Mask Propagation**: End-to-end transformers predicting continuous point trajectories ([[architectures/visual-tracking-and-flow/cotracker|CoTracker3]]) and promptable video mask propagation ([[architectures/vision-foundation-models/sam-2|SAM 2]]).

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/video-tracking/01-historical-evolution-and-paradigms|Video Tracking: Historical Evolution & Paradigms]]
- **Production Implementation Playbook**: [[topics/video-tracking/02-production-pipeline-and-workarounds|Video Tracking: Production Pipeline, Traps & Identity Switch Workarounds]]

### 🔬 Core Architecture Deep-Dives (Central Vault)
| Model / System | Tracking Paradigm | Primary Innovation | License | Dedicated Note Link |
| :--- | :--- | :--- | :---: | :---: |
| **BoT-SORT & ByteTrack** | Bounding Box MOT | Global Motion Compensation (GMC) + Two-stage low-score association | **MIT** | [[architectures/visual-tracking-and-flow/botsort-and-bytetrack|BoT-SORT & ByteTrack]] |
| **CoTracker & CoTracker3** | Dense Point Trajectory | Spatial-temporal cross-point attention tracking 70k points with occlusion flags | **Apache-2.0** | [[architectures/visual-tracking-and-flow/cotracker|CoTracker Deep-Dive]] |
| **SAM 2 (Video Engine)** | Promptable Mask Tracking | Spatial-temporal streaming memory bank with 44 FPS mask propagation | **Apache-2.0** | [[architectures/vision-foundation-models/sam-2|SAM 2 Video Engine]] |

---

## 📊 Standardized SOTA Benchmark Comparison (MOT17 & TAP-Vid)
| Model Architecture | Benchmark Dataset | Primary Metric | Secondary Metric | Latency / FPS | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ByteTrack** | MOT17 Test (MOT) | 80.3 MOTA | 63.1 HOTA | 1.2 ms (CPU only) | MIT |
| **BoT-SORT + ReID** | MOT17 Test (MOT) | **81.1 MOTA** | **65.6 HOTA** | 6.8 ms (CPU/GPU) | MIT |
| **CoTracker3** | TAP-Vid Kinetics (Points)| **68.2 AJ** | 73.8 Davis AJ | 28.5 ms (GPU) | Apache-2.0 |
| **SAM 2 (Hiera-Base+)**| SA-V Benchmark (Masks) | **75.0 J&F** | 77.2 J-Score | 22.8 ms (44 FPS) | Apache-2.0 |

---

## ⚖️ Commercial Usability & License Audit
- **100% Commercial-Safe (Permissive: MIT / Apache-2.0)**:
  - `ifzhang/ByteTrack` & `NirAharon/BoT-SORT`: **MIT**. Safe for proprietary surveillance and edge devices.
  - `facebookresearch/co-tracker`: **Apache-2.0**.
  - `facebookresearch/sam2`: **Apache-2.0**.
  - *Recommendation*: Avoid older DeepSORT repos that carry GPL-3.0 licenses; standardize strictly on ByteTrack/BoT-SORT for boxes and SAM 2 for masks.

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/object-detection/00-object-detection-moc|Object Detection MOC]]
- [[topics/object-segmentation/00-object-segmentation-moc|Object Segmentation MOC]]
- [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]]**

## 📌 Domain Overview & Scope
Video Tracking maintains consistent spatial, semantic, and instance identity across continuous image sequences despite severe occlusions, viewpoint changes, motion blur, and camera ego-motion. The domain has bifurcated into two major operational paradigms:
1. **Real-Time Multiple Object Tracking (MOT)**: Tracking-by-detection pairing high-frame-rate 2D detectors with lightweight CPU Kalman filters and appearance ReID matching ([[architectures/visual-tracking-and-flow/botsort-and-bytetrack|BoT-SORT & ByteTrack]]).
2. **Dense Physical Trajectory & Mask Propagation**: End-to-end transformers predicting continuous point trajectories ([[architectures/visual-tracking-and-flow/cotracker|CoTracker3]]) and promptable video mask propagation ([[architectures/vision-foundation-models/sam-2|SAM 2]]).

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/video-tracking/01-historical-evolution-and-paradigms|01 Historical Evolution And Paradigms]]
- **Production Implementation Playbook**: [[topics/video-tracking/02-production-pipeline-and-workarounds|02 Production Pipeline And Workarounds]]
- **Modern Architecture & Open Problems**: [[topics/video-tracking/03-association-and-open-problems|03 Association And Open Problems]]
- **Classical, Geometric & Hybrid Baselines**: [[topics/video-tracking/04-classical-and-hybrid-methods|04 Classical And Hybrid Methods]]

### 🔬 Core Architecture Deep-Dives (Central Vault)
| Model / System | Tracking Paradigm | Primary Innovation | License | Dedicated Note Link |
| :--- | :--- | :--- | :---: | :---: |
| **BoT-SORT & ByteTrack** | Bounding Box MOT | Global Motion Compensation (GMC) + Two-stage low-score association | **MIT** | [[architectures/visual-tracking-and-flow/botsort-and-bytetrack|BoT-SORT & ByteTrack]] |
| **CoTracker & CoTracker3** | Dense Point Trajectory | Spatial-temporal cross-point attention tracking 70k points with occlusion flags | **Apache-2.0** | [[architectures/visual-tracking-and-flow/cotracker|CoTracker Deep-Dive]] |
| **SAM 2 (Video Engine)** | Promptable Mask Tracking | Spatial-temporal streaming memory bank with 44 FPS mask propagation | **Apache-2.0** | [[architectures/vision-foundation-models/sam-2|SAM 2 Video Engine]] |

---

## 📊 Standardized SOTA Benchmark Comparison (MOT17 & TAP-Vid)
| Model Architecture | Benchmark Dataset | Primary Metric | Secondary Metric | Latency / FPS | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ByteTrack** | MOT17 Test (MOT) | 80.3 MOTA | 63.1 HOTA | 1.2 ms (CPU only) | MIT |
| **BoT-SORT + ReID** | MOT17 Test (MOT) | **81.1 MOTA** | **65.6 HOTA** | 6.8 ms (CPU/GPU) | MIT |
| **CoTracker3** | TAP-Vid Kinetics (Points)| **68.2 AJ** | 73.8 Davis AJ | 28.5 ms (GPU) | Apache-2.0 |
| **SAM 2 (Hiera-Base+)**| SA-V Benchmark (Masks) | **75.0 J&F** | 77.2 J-Score | 22.8 ms (44 FPS) | Apache-2.0 |

---

## ⚖️ Commercial Usability & License Audit
- **100% Commercial-Safe (Permissive: MIT / Apache-2.0)**:
  - `ifzhang/ByteTrack` & `NirAharon/BoT-SORT`: **MIT**. Safe for proprietary surveillance and edge devices.
  - `facebookresearch/co-tracker`: **Apache-2.0**.
  - `facebookresearch/sam2`: **Apache-2.0**.
  - *Recommendation*: Avoid older DeepSORT repos that carry GPL-3.0 licenses; standardize strictly on ByteTrack/BoT-SORT for boxes and SAM 2 for masks.

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/object-detection/00-object-detection-moc|Object Detection MOC]]
- [[topics/object-segmentation/00-object-segmentation-moc|Object Segmentation MOC]]
- [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]]

---

## 📂 All Notes in This Domain

```dataview
TABLE type AS "Note Type", updated AS "Last Updated", status AS "Status"
WHERE contains(file.folder, "topics/video-tracking")
SORT file.name ASC
```

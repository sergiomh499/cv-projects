---
title: Video Tracking Master Index & Directory
tags:
  - computer-vision
  - video-tracking
  - index
  - sota
  - models
  - playbooks
updated: 2026-09-08
aliases:
  - Video Tracking Playbook
  - Tracking Index
---

# Video Tracking: Master Index & Domain Guide

> **Obsidian Users**: Access the unified Map of Content at [[topics/video-tracking/00-video-tracking-moc|Video Tracking MOC]].

## 📌 Executive Brief
Video Tracking maintains consistent object identity across sequential video frames despite camera motion, object deformation, lighting fluctuations, and severe occlusion. Modern paradigms span **real-time multiple object tracking (MOT)** using motion-compensated Kalman filters and **dense physical point/mask foundation models** (CoTracker3, SAM 2).

---

## 🧭 Topic Organization & File Structure

Following the **MOC + Central Architecture Vault** structure:

```text
topics/video-tracking/
├── 00-video-tracking-moc.md                      # 🗺️ Master Map of Content & Cross-Domain Hub
├── 01-historical-evolution-and-paradigms.md       # 📜 Historical Evolution (Optical Flow -> SORT -> ByteTrack -> CoTracker)
├── 02-production-pipeline-and-workarounds.md      # 🛠️ Production Pipeline, ID-Switch Gates & Track Pools
└── README.md                                     # 🧭 Master Directory Index for Git / Web Browsers

architectures/                                    # 🔬 Shared Multi-Task Architecture Vault
├── transformer-detectors/
│   └── cotracker.md                              # CoTracker & CoTracker3: Dense Point Trajectory Transformers (Apache-2.0)
├── real-time-unified/
│   └── botsort-and-bytetrack.md                  # BoT-SORT & ByteTrack: Production Multi-Object Tracking (MIT)
└── foundation-models/
    └── sam-2.md                                  # SAM 2 Video Engine: Streaming Spatial-Temporal Memory (Apache-2.0)
```

---

## 📚 Navigation Directory

| Note Title | Document Type | Description | Link |
| :--- | :---: | :--- | :---: |
| **Video Tracking MOC** | `MOC` | Master navigational hub, benchmark matrix & license audit | [Open MOC](00-video-tracking-moc.md) <br> `[[topics/video-tracking/00-video-tracking-moc|00-video-tracking-moc]]` |
| **Historical Evolution & Paradigms** | `Evolution Guide` | Evolution from Lucas-Kanade optical flow to SORT, ByteTrack low-score retention, and dense point transformers | [Open Evolution](01-historical-evolution-and-paradigms.md) <br> `[[topics/video-tracking/01-historical-evolution-and-paradigms|01-historical-evolution-and-paradigms]]` |
| **Production Pipeline & Workarounds** | `Playbook` | Motion-compensated ingestion, gated ReID matching, hysteresis state machines, zero-allocation track pools | [Open Playbook](02-production-pipeline-and-workarounds.md) <br> `[[topics/video-tracking/02-production-pipeline-and-workarounds|02-production-pipeline-and-workarounds]]` |
| **BoT-SORT & ByteTrack Deep-Dive** | `Architecture Vault` | Global Motion Compensation (GMC) + two-stage low-score Hungarian association (MIT) | [Open BoT-SORT/ByteTrack](../../architectures/visual-tracking-and-flow/botsort-and-bytetrack.md) <br> `[[architectures/visual-tracking-and-flow/bytetrack|botsort-and-bytetrack]]` |
| **CoTracker & CoTracker3 Deep-Dive** | `Architecture Vault` | Spatial-temporal cross-point attention tracking 70k points with occlusion flags (Apache-2.0) | [Open CoTracker](../../architectures/visual-tracking-and-flow/cotracker.md) <br> `[[architectures/visual-tracking-and-flow/cotracker|cotracker]]` |
| **SAM 2 Video Engine Deep-Dive** | `Architecture Vault` | Streaming spatial-temporal memory bank with 44 FPS promptable video mask propagation | [Open SAM 2](../../architectures/vision-foundation-models/sam-2.md) <br> `[[architectures/vision-foundation-models/sam-2|sam-2]]` |

---

## 📊 Summary SOTA Benchmark Comparison
| Model Architecture | Benchmark Dataset | Primary Metric | Secondary Metric | Latency / FPS | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ByteTrack** | MOT17 Test (MOT) | 80.3 MOTA | 63.1 HOTA | 1.2 ms (CPU only) | MIT |
| **BoT-SORT + ReID** | MOT17 Test (MOT) | **81.1 MOTA** | **65.6 HOTA** | 6.8 ms (CPU/GPU) | MIT |
| **CoTracker3** | TAP-Vid Kinetics (Points)| **68.2 AJ** | 73.8 Davis AJ | 28.5 ms (GPU) | Apache-2.0 |
| **SAM 2 (Hiera-Base+)**| SA-V Benchmark (Masks) | **75.0 J&F** | 77.2 J-Score | 22.8 ms (44 FPS) | Apache-2.0 |

---

## ⚖️ Commercial Usability Quick-Audit
- **100% Commercial Permissive (MIT / Apache-2.0)**:
  - `ifzhang/ByteTrack` and `NirAharon/BoT-SORT`: **MIT License**. Safe for proprietary CCTV, traffic analytics, and edge deployment.
  - `facebookresearch/co-tracker` and `facebookresearch/sam2`: **Apache-2.0**.
  - *Recommendation*: Standardize on ByteTrack for $<2\text{ ms}$ CPU tracking and CoTracker3/SAM 2 for high-precision physical deformation and mask propagation.

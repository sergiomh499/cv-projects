---
title: 6-DoF Pose Estimation Master Index & Directory
tags:
  - computer-vision
  - 6dof-pose
  - index
  - sota
  - models
  - playbooks
updated: 2026-09-08
aliases:
  - 6-DoF Pose Playbook
  - Pose Estimation Index
---

# 6-DoF Pose Estimation: Master Index & Domain Guide

> **Obsidian Users**: Access the unified Map of Content at [[topics/6dof-pose-estimation/00-6dof-pose-estimation-moc|6-DoF Pose Estimation MOC]].

## 📌 Executive Brief
6-Degrees-of-Freedom (6-DoF) Pose Estimation determines the 3D translation ($X,Y,Z$) and 3D orientation ($SO(3)$) of objects relative to a calibrated sensor frame. The domain has transitioned from per-object dense keypoint estimators (PVNet, GDR-Net) to zero-shot CAD foundation models ([[architectures/real-time-unified/foundationpose-and-megapose|FoundationPose & MegaPose]]) achieving sub-centimeter robotic grasp accuracy at 30 FPS.

---

## 🧭 Topic Organization & File Structure

```text
topics/6dof-pose-estimation/
├── 00-6dof-pose-estimation-moc.md                # 🗺️ Master Map of Content & Cross-Domain Hub
├── 01-historical-evolution-and-paradigms.md       # 📜 Historical Evolution (SIFT-PnP -> PVNet -> MegaPose -> FoundationPose)
├── 02-production-pipeline-and-workarounds.md      # 🛠️ Production Pipeline, Symmetry Traps & ICP Workarounds
└── README.md                                     # 🧭 Master Directory Index for Git / Web Browsers

architectures/                                    # 🔬 Shared Multi-Task Architecture Vault
└── real-time-unified/
    └── foundationpose-and-megapose.md            # FoundationPose & MegaPose: Zero-Shot CAD-Driven 6-DoF (Apache-2.0 / NVIDIA)
```

---

## 📚 Navigation Directory

| Note Title | Document Type | Description | Link |
| :--- | :---: | :--- | :---: |
| **6-DoF Pose Estimation MOC** | `MOC` | Master navigational hub, benchmark matrix & license audit | [Open MOC](00-6dof-pose-estimation-moc.md) <br> `[[topics/6dof-pose-estimation/00-6dof-pose-estimation-moc|00-6dof-pose-estimation-moc]]` |
| **Historical Evolution & Paradigms** | `Evolution Guide` | Evolution from SIFT-PnP solvers to PVNet directional voting, render-and-compare, and zero-shot transformers | [Open Evolution](01-historical-evolution-and-paradigms.md) <br> `[[topics/6dof-pose-estimation/01-historical-evolution-and-paradigms|01-historical-evolution-and-paradigms]]` |
| **Production Pipeline & Workarounds** | `Playbook` | RGB-D sensor ingestion, ADD-S rotational symmetry loss, depth inpainting, GPU-ICP refinement | [Open Playbook](02-production-pipeline-and-workarounds.md) <br> `[[topics/6dof-pose-estimation/02-production-pipeline-and-workarounds|02-production-pipeline-and-workarounds]]` |
| **FoundationPose & MegaPose Deep-Dive** | `Architecture Vault` | Foundation zero-shot 6-DoF estimation and 32 ms tracking from untextured CAD models | [Open FoundationPose/MegaPose](../../architectures/real-time-unified/foundationpose-and-megapose.md) <br> `[[architectures/real-time-unified/foundationpose-and-megapose|foundationpose-and-megapose]]` |

---

## 📊 Summary SOTA Benchmark Comparison (BOP Challenge)
| Model Architecture | Paradigm | YCB-V (ADD-S AUC) | Linemod-Occluded (ADD-0.1d) | Latency (ms) | Commercial License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PVNet (Classic)** | Per-Object Instance | 72.8% | 40.8% | 80.0 ms | Apache-2.0 |
| **GDR-Net** | Per-Object Direct | 91.6% | 62.2% | 45.0 ms | Apache-2.0 |
| **MegaPose** | Zero-Shot CAD-based | 88.5% | 71.0% | 150.0 ms | **Apache-2.0** |
| **FoundationPose** | Zero-Shot Foundation | **96.2%** (SOTA) | **89.5%** (SOTA) | **32.0 ms** | ⚠️ Non-Commercial (NVIDIA) |

---

## ⚖️ Commercial Usability Quick-Audit
- **Commercial Permissive (Apache-2.0)**:
  - `facebookresearch/megapose`: **Apache-2.0**. Recommended for commercial industrial robotics.
- **Commercial Warning**:
  - `NVlabs/FoundationPose`: **Non-Commercial Research License**. Contact NVIDIA for commercial licensing.

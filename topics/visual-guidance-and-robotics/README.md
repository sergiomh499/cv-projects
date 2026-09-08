---
title: Visual Guidance & Robotics Master Index & Directory
tags:
  - computer-vision
  - visual-guidance
  - robotics
  - index
  - sota
  - models
  - playbooks
updated: 2026-09-08
aliases:
  - Visual Guidance Playbook
  - Robotics Guidance Index
---

# Visual Guidance & Robotics: Master Index & Domain Guide

> **Obsidian Users**: Access the unified Map of Content at [[topics/visual-guidance-and-robotics/00-visual-guidance-and-robotics-moc|Visual Guidance & Robotics MOC]].

## 📌 Executive Brief
Visual Guidance translates real-time visual streams into physical robot control commands, trajectory waypoints, and manipulation grasps. The domain has evolved from handcrafted visual servoing to **Vision-Language-Action (VLA) foundation models** ([[architectures/multimodal-vlm-and-vla/anygrasp-and-openvla|OpenVLA & AnyGrasp]]) that map natural language and camera frames directly into continuous 7-DoF robot motor control.

---

## 🧭 Topic Organization & File Structure

```text
topics/visual-guidance-and-robotics/
├── 00-visual-guidance-and-robotics-moc.md         # 🗺️ Master Map of Content & Cross-Domain Hub
├── 01-historical-evolution-and-paradigms.md       # 📜 Historical Evolution (Visual Servoing -> GraspNet -> OpenVLA)
├── 02-production-pipeline-and-workarounds.md      # 🛠️ Production Pipeline, Kinematic Traps & Action Chunking
└── README.md                                     # 🧭 Master Directory Index for Git / Web Browsers

architectures/                                    # 🔬 Shared Multi-Task Architecture Vault
└── real-time-unified/
    └── anygrasp-and-openvla.md                   # AnyGrasp & OpenVLA: Robotic Visual Guidance & Action Models (Apache-2.0)
```

---

## 📚 Navigation Directory

| Note Title | Document Type | Description | Link |
| :--- | :---: | :--- | :---: |
| **Visual Guidance & Robotics MOC** | `MOC` | Master navigational hub, benchmark matrix & license audit | [Open MOC](00-visual-guidance-and-robotics-moc.md) <br> `[[topics/visual-guidance-and-robotics/00-visual-guidance-and-robotics-moc|00-visual-guidance-and-robotics-moc]]` |
| **Historical Evolution & Paradigms** | `Evolution Guide` | Progression from IBVS/PBVS visual servoing to contact grasp nets and autoregressive VLA models | [Open Evolution](01-historical-evolution-and-paradigms.md) <br> `[[topics/visual-guidance-and-robotics/01-historical-evolution-and-paradigms|01-historical-evolution-and-paradigms]]` |
| **Production Pipeline & Workarounds** | `Playbook` | Action chunking, damped least-squares inverse kinematics, and control barrier function safety shields | [Open Playbook](02-production-pipeline-and-workarounds.md) <br> `[[topics/visual-guidance-and-robotics/02-production-pipeline-and-workarounds|02-production-pipeline-and-workarounds]]` |
| **OpenVLA & AnyGrasp Deep-Dive** | `Architecture Vault` | Detailed breakdown of autoregressive action tokenization and dense 6-DoF grasp synthesis | [Open OpenVLA/AnyGrasp](../../architectures/multimodal-vlm-and-vla/anygrasp-and-openvla.md) <br> `[[architectures/multimodal-vlm-and-vla/anygrasp-and-openvla|anygrasp-and-openvla]]` |

---

## 📊 Summary SOTA Benchmark Comparison (Robotic Manipulation)
| Architecture | Model Parameters | Task Domain | Open-X Embodiment Success Rate | Inference Latency | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Octo-Small** | 27 M | VLA Diffusion Policy | 68.2% | 12.0 ms | Apache-2.0 |
| **Octo-Base** | 93 M | VLA Diffusion Policy | 74.5% | 28.0 ms | Apache-2.0 |
| **OpenVLA 7B** | 7 B | VLA Autoregressive | **84.7%** (SOTA) | 180.0 ms (FP8) | **Apache-2.0** |
| **AnyGrasp** | 35 M | Dense 6-DoF Grasping | **92.4%** Grasp Success | 45.0 ms | Non-Commercial |

---

## ⚖️ Commercial Usability Quick-Audit
- **Commercial Permissive (Apache-2.0)**:
  - `openvla/openvla`: **Apache-2.0**. Fully approved for commercial robotics and factory automation.
  - `octo-models/octo`: **Apache-2.0**.

---
title: Visual Guidance & Robotics MOC
type: MOC
domain: Visual Guidance & Robotics
tags:
  - moc
  - computer-vision
  - robotics
  - visual-guidance
  - grasping
  - vla
  - openvla
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - Visual Guidance MOC
  - Robotics Guidance Hub
  - VLA Guidance MOC
---

# 🗺️ Visual Guidance & Robotics MOC (Map of Content)

## 📌 Domain Overview & Scope
Visual Guidance translates real-time visual streams into physical robot control commands, trajectory waypoints, and end-effector manipulation actions. Key application domains include:
- Visual Servoing (PBVS & IBVS) for precision docking and drone landing.
- Dense 6-DoF grasp synthesis on unstructured physical objects.
- End-to-end **Vision-Language-Action (VLA) foundation policies** ([[architectures/real-time-unified/anygrasp-and-openvla|AnyGrasp & OpenVLA]]) that directly predict continuous robotic joint velocities from natural language instructions.

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/visual-guidance-and-robotics/01-historical-evolution-and-paradigms|Visual Guidance: Historical Evolution & Paradigms]]
- **Production Implementation Playbook**: [[topics/visual-guidance-and-robotics/02-production-pipeline-and-workarounds|Visual Guidance: Production Pipeline, Singularity Traps & Latency Workarounds]]

### 🔬 Core Architecture Deep-Dives (Central Vault)
| Model / System | Operational Paradigm | Primary Innovation | License | Dedicated Note Link |
| :--- | :--- | :--- | :---: | :---: |
| **OpenVLA & Octo** | Vision-Language-Action (VLA) | Autoregressive tokenization mapping visual tokens to 7-DoF joint actions | **Apache-2.0** | [[architectures/real-time-unified/anygrasp-and-openvla\|OpenVLA & AnyGrasp Deep-Dive]] |
| **AnyGrasp & GraspNet** | Dense 6-DoF Grasp Network | Evaluates billions of candidate grasps directly on dense point clouds | **Research / Non-Commercial** | [[architectures/real-time-unified/anygrasp-and-openvla\|OpenVLA & AnyGrasp Deep-Dive]] |

---

## 📊 Standardized SOTA Benchmark Comparison (Robotic Manipulation)
| Architecture | Model Parameters | Task Domain | Open-X Embodiment Success Rate | Inference Latency | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Octo-Small** | 27 M | VLA Diffusion Policy | 68.2% | 12.0 ms | Apache-2.0 |
| **Octo-Base** | 93 M | VLA Diffusion Policy | 74.5% | 28.0 ms | Apache-2.0 |
| **OpenVLA 7B** | 7 B | VLA Autoregressive | **84.7%** (SOTA) | 180.0 ms (FP8) | **Apache-2.0** |
| **AnyGrasp** | 35 M | Dense 6-DoF Grasping | **92.4%** Grasp Success | 45.0 ms | Non-Commercial |

---

## ⚖️ Commercial Usability & License Audit
- **Commercial Permissive (Apache-2.0)**:
  - `openvla/openvla`: **Apache-2.0**. Safe for commercial robotics and industrial automation.
  - `octo-models/octo`: **Apache-2.0**.
- **Commercial Caution**:
  - `graspnet/anygrasp_sdk`: Non-commercial research license. For commercial 6-DoF grasping, use open grasp synthesis pipelines (e.g. Contact-GraspNet or Dex-Net).

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/6dof-pose-estimation/00-6dof-pose-estimation-moc|6-DoF Pose Estimation MOC]]
- [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]]
- [[topics/slam-and-spatial-perception/00-slam-and-spatial-perception-moc|SLAM MOC]]

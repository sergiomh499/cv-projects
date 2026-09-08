---
title: Real-Time Systems Master Index & Directory
tags:
  - computer-vision
  - real-time
  - systems
  - index
  - sota
  - models
  - playbooks
updated: 2026-09-08
aliases:
  - Real-Time Systems Playbook
  - Real-Time Index
---

# Real-Time Systems: Master Index & Domain Guide

> **Obsidian Users**: Access the unified Map of Content at [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]].

## 📌 Executive Brief
Real-Time Computer Vision and Perception Systems operate under strict temporal deadlines where latency determinism and zero-jitter execution are mission-critical. Key capabilities include configuring **Linux PREEMPT_RT** kernel environments, locking memory against page faults (`mlockall`), and using lock-free shared memory zero-copy IPC ([[architectures/real-time-unified/iceoryx2-and-zenoh|Iceoryx2 & Zenoh]]) delivering sub-microsecond frame transport.

---

## 🧭 Topic Organization & File Structure

```text
topics/real-time-systems/
├── 00-real-time-systems-moc.md                   # 🗺️ Master Map of Content & Cross-Domain Hub
├── 01-historical-evolution-and-paradigms.md       # 📜 Historical Evolution (Monolithic Loops -> ROS 1/2 -> PREEMPT_RT -> Iceoryx2)
├── 02-production-pipeline-and-workarounds.md      # 🛠️ Production Pipeline, Page Fault Traps & mlockall Workarounds
└── README.md                                     # 🧭 Master Directory Index for Git / Web Browsers

architectures/                                    # 🔬 Shared Multi-Task Architecture Vault
└── real-time-unified/
    └── iceoryx2-and-zenoh.md                     # Iceoryx2 & Zenoh: Ultra-Low-Latency Zero-Copy Middleware (Apache-2.0 / MIT)
```

---

## 📚 Navigation Directory

| Note Title | Document Type | Description | Link |
| :--- | :---: | :--- | :---: |
| **Real-Time Systems MOC** | `MOC` | Master navigational hub, benchmark matrix & license audit | [Open MOC](00-real-time-systems-moc.md) <br> `[[topics/real-time-systems/00-real-time-systems-moc\|00-real-time-systems-moc]]` |
| **Historical Evolution & Paradigms** | `Evolution Guide` | Evolution from OpenCV loops to ROS 1/2, PREEMPT_RT priority inheritance, and pointer-exchange IPC | [Open Evolution](01-historical-evolution-and-paradigms.md) <br> `[[topics/real-time-systems/01-historical-evolution-and-paradigms\|01-historical-evolution-and-paradigms]]` |
| **Production Pipeline & Workarounds** | `Playbook` | Page fault traps, `mlockall` virtual memory locking, `isolcpus` core pinning, C-state disabling | [Open Playbook](02-production-pipeline-and-workarounds.md) <br> `[[topics/real-time-systems/02-production-pipeline-and-workarounds\|02-production-pipeline-and-workarounds]]` |
| **Iceoryx2 & Zenoh Deep-Dive** | `Architecture Vault` | Lock-free shared memory pointer passing with sub-microsecond transport latency | [Open Iceoryx2/Zenoh](../../architectures/real-time-unified/iceoryx2-and-zenoh.md) <br> `[[architectures/real-time-unified/iceoryx2-and-zenoh\|iceoryx2-and-zenoh]]` |

---

## 📊 Summary SOTA Benchmark Comparison (Middleware Latency)
| Middleware / Protocol | Payload Size | Latency | CPU Utilization (30 FPS) | Deterministic Real-Time Safe | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard ROS 2 (FastDDS)** | 10 MB (Image) | 12.80 ms | 45% (High Copy Load) | No | Apache-2.0 |
| **ZeroMQ (IPC Socket)** | 10 MB (Image) | 4.20 ms | 22% | No | MPL-2.0 |
| **Eclipse Zenoh** | 10 MB (Image) | 1.85 ms | 8% | Soft Real-Time | Apache-2.0 |
| **Eclipse Iceoryx2** | 10 MB (Image) | **0.0008 ms (<1 $\mu$s)**| **<1% (Zero-Copy)** | **Hard Real-Time** | **Apache-2.0 / MIT** |

---

## ⚖️ Commercial Usability Quick-Audit
- **100% Commercial Permissive (Apache-2.0 / MIT)**:
  - `eclipse-iceoryx/iceoryx2`: **Apache-2.0 / MIT**.
  - `eclipse-zenoh/zenoh`: **Apache-2.0 / EPL-2.0**.

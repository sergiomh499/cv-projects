---
title: Real-Time Systems MOC
type: MOC
domain: Real-Time Systems
tags:
  - moc
  - systems-engineering
  - real-time
  - iceoryx2
  - zenoh
  - ros2
  - preempt-rt
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - Real-Time Systems MOC
  - Real-Time Hub
  - Low-Latency Systems MOC
---

# 🗺️ Real-Time Systems MOC (Map of Content)

## 📌 Domain Overview & Scope
Real-Time Computer Vision and Perception Systems operate under strict temporal deadlines: correctness depends not only on the logical output of the neural network or tracking filter, but also on the deterministic instant at which the result is delivered. 

Key technical requirements include:
- Incurring zero unbounded latencies ($\le 33\text{ ms}$ for 30 FPS cameras, $\le 10\text{ ms}$ for high-speed robotics).
- Eliminating Linux kernel scheduling jitter using the **PREEMPT_RT** real-time patch.
- Zero-copy shared memory IPC ([[architectures/real-time-unified/iceoryx2-and-zenoh|Iceoryx2 & Zenoh]]) delivering sub-microsecond point cloud and video frame transfers.
- Hardware timestamp telemetry to detect latency anomalies across sensor-to-actuator control loops.

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/real-time-systems/01-historical-evolution-and-paradigms|Real-Time Systems: Historical Evolution & Paradigms]]
- **Production Implementation Playbook**: [[topics/real-time-systems/02-production-pipeline-and-workarounds|Real-Time Systems: Production Pipeline, Jitter Traps & PREEMPT_RT Workarounds]]

### 🔬 Core Architecture Deep-Dives (Central Vault)
| Framework / Engine | Operational Domain | Primary Innovation | License | Dedicated Note Link |
| :--- | :--- | :--- | :---: | :---: |
| **Eclipse Iceoryx2** | Lock-Free Shared Memory IPC | Pure Rust/C++ pointer-passing architecture delivering sub-microsecond latency ($<1\mu\text{s}$) | **Apache-2.0 / MIT** | [[architectures/real-time-unified/iceoryx2-and-zenoh|Iceoryx2 & Zenoh Deep-Dive]] |
| **Eclipse Zenoh** | Zero-Overhead Pub/Sub | Replaces heavy DDS discovery with 5-byte wire protocol for robotics | **Apache-2.0** | [[architectures/real-time-unified/iceoryx2-and-zenoh|Iceoryx2 & Zenoh Deep-Dive]] |

---

## 📊 Standardized SOTA Benchmark Comparison (Middleware Latency)
| Middleware / Protocol | Payload Size | Latency | CPU Utilization (30 FPS) | Deterministic Real-Time Safe | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard ROS 2 (FastDDS)** | 10 MB (Image) | 12.80 ms | 45% (High Copy Load) | No | Apache-2.0 |
| **ZeroMQ (IPC Socket)** | 10 MB (Image) | 4.20 ms | 22% | No | MPL-2.0 |
| **Eclipse Zenoh** | 10 MB (Image) | 1.85 ms | 8% | Soft Real-Time | Apache-2.0 |
| **Eclipse Iceoryx2** | 10 MB (Image) | **0.0008 ms (<1 $\mu$s)**| **<1% (Zero-Copy)** | **Hard Real-Time** | **Apache-2.0 / MIT** |

---

## ⚖️ Commercial Usability & License Audit
- **100% Commercial Permissive (Apache-2.0 / MIT)**:
  - `eclipse-iceoryx/iceoryx2`: **Apache-2.0 / MIT**.
  - `eclipse-zenoh/zenoh`: **Apache-2.0 / EPL-2.0**.
  - *Recommendation*: Pair Linux PREEMPT_RT with Iceoryx2 for all vision IPC on robots and vehicles to avoid DDS latency spikes.

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]]
- [[topics/sensor-fusion/00-sensor-fusion-moc|Sensor Fusion MOC]]
- [[topics/video-tracking/00-video-tracking-moc|Video Tracking MOC]]

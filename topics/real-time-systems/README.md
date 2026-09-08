---
title: Real-Time Systems Playbook
tags:
  - systems-engineering
  - real-time
  - iceoryx2
  - ros2
  - preempt-rt
  - zero-copy
  - deterministic
updated: 2026-09-08
aliases:
  - Real-Time Systems
---

# Real-Time Systems Playbook

# Overview
Real-Time Computer Vision and Perception Systems operate under strict temporal deadlines where correctness depends not only on algorithmic accuracy, but also on the deterministic instant at which results are delivered. Applications range from autonomous vehicle emergency braking ($<30\text{ ms}$) and high-speed robotic sorting ($<5\text{ ms}$) to closed-loop drone flight control and surgical robotic tracking.

Related notes: [[topics/gpu-deployment/README|GPU Deployment]], [[topics/fpga-deployment/README|FPGA Deployment]], [[topics/video-tracking/README|Video Tracking]].

---

## SOTA & Research (2023–2026 Breakthroughs)

1. **Iceoryx 2: Modular, Lock-Free Zero-Copy IPC in Rust and C++** (Eclipse Foundation, 2023–2025)
   - *Key Innovation*: Completely re-architected in Rust, introducing lock-free shared memory queues and direct zero-copy pointer loans. Enables sub-microsecond latency transfer of gigabyte-scale multi-camera and LiDAR streams without kernel context switches or dynamic heap allocations.
   - [Official Code & Docs](https://github.com/eclipse-iceoryx/iceoryx2)

2. **ROS 2 Jazzy Jalisco & Iron Irwini Real-Time Middleware** (Open Robotics, 2023 / 2024)
   - *Key Innovation*: Standardized loaned messages across DDS endpoints, deterministic callback scheduling groups, and integration with the Two-Level Segregated Fit (`tlsf`) deterministic memory allocator.
   - [Official Code](https://github.com/ros2/ros2)

3. **Cyclone DDS Shared Memory Zero-Copy Transport** (Eclipse Foundation, 2023 / 2024)
   - *Key Innovation*: Deep integration of POSIX shared memory and Iceoryx into OMG DDS specifications, dropping multi-node perception message latency from milliseconds down to single-digit microseconds.
   - [Official Code](https://github.com/eclipse-cyclonedds/cyclonedds)

4. **PREEMPT_RT Kernel Mainline Upstreaming** (Linux Foundation, 2024 / 2025)
   - *Key Innovation*: Final upstream merge of the real-time patchset into the mainline Linux kernel, providing deterministic priority inheritance and bounded interrupt latency without maintaining out-of-tree custom kernels.

---

### Quantitative SOTA Benchmark Comparison (Real-Time Middleware & IPC)
| Middleware Architecture | Mechanism | Glass-to-Glass Latency Jitter | Inter-Process Latency (1080p Image) | Zero-Copy Support | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Iceoryx 2 (Rust / C++)** | Lock-Free Shared Memory Ring | < 2.5 µs | 0.82 µs | Yes (Direct Pointer Loan) | Apache-2.0 / MIT |
| **ROS 2 Jazzy (Cyclone + Shm)**| Loaned Messages over POSIX Shm | < 12.0 µs | 2.10 µs | Yes | Apache-2.0 |
| **Standard Linux TCP/Socket IPC**| Kernel TCP Loopback / Sockets | > 850.0 µs | 3,400.0 µs (3.4 ms) | No (Multiple Kernel Copies) | N/A |
| **PREEMPT_RT Kernel (SCHED_FIFO)**| Real-Time Scheduler / Core Isolation | < 15.0 µs (Max WCET) | N/A (OS Scheduling Layer) | N/A | GPL-2.0 |

---

## Commercial Usability & License Audit
- **Commercial Permissive (Safe)**:
  - **Eclipse Iceoryx 2 (`eclipse-iceoryx/iceoryx2`)**: Dual-licensed under **Apache-2.0** or **MIT**. Safe for proprietary commercial robotic platforms and automotive ECUs.
  - **ROS 2 (`ros2/ros2`)**: Licensed under **Apache-2.0**. Commercial distribution in closed-source robotics products is standard industry practice.
  - **Eclipse Cyclone DDS (`eclipse-cyclonedds/cyclonedds`)**: **Eclipse Public License 2.0 (EPL-2.0) / Eclipse Distribution License 1.0 (EDL-1.0 - BSD-3-Clause equivalent)**. Safe for commercial deployment.
  - *Commercial Strategy*: Permissive licenses allow building proprietary perception pipelines without exposing business logic or algorithmic IP.

---

## Architecture Alternatives & Trade-offs
| Architecture Pattern | Latency Jitter | Throughput | Computational Overhead | Primary Domain |
| :--- | :--- | :--- | :--- | :--- |
| **Lock-Free Shared Memory (Iceoryx 2)** | Sub-microsecond | Extremely high (100+ GB/s) | Zero memory copy | High-bandwidth multi-camera/LiDAR robotics |
| **ROS 2 Zero-Copy Intra-Process** | Low-to-Moderate | High | Very Low | Modular multi-node robotic perception stacks |
| **PREEMPT_RT Hard Real-Time Loops** | Lowest (<50 microseconds jitter) | Deterministic | Moderate (requires lock-free design) | Closed-loop industrial control & actuation |
| **Hardware-Triggered FPGA Pipelines** | Truly Deterministic (Zero OS jitter) | Line-Rate | High hardware engineering cost | High-speed sorting, defense, aerospace |

---

## Popular Repos & Integrations
- **[eclipse-iceoryx/iceoryx2](https://github.com/eclipse-iceoryx/iceoryx2)**: SOTA zero-copy inter-process communication mechanism written in Rust and C++ with lock-free queues (Apache-2.0 / MIT).
- **[ros2/ros2](https://github.com/ros2/ros2)**: Production robotics middleware with real-time DDS middleware and lifecycle nodes (Apache-2.0).
- **[eclipse-cyclonedds/cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds)**: High-performance, low-latency DDS implementation (EPL-2.0 / EDL-1.0).
- **Tooling Integrations**:
  - **FiftyOne**: Log time-to-detection and confidence drifts across varying frame-rate streams to detect degradation during camera throttling.
  - **Rerun**: Native high-speed telemetry viewer capable of logging time-series latency plots, frame intervals, and queue depths alongside visual feeds.

---

## End-to-End Pipeline & Workarounds
1. **Pipeline Stages**:
   - Hardware sensor trigger -> Kernel DMA driver ring buffer -> Zero-copy shared memory transfer -> Asynchronous GPU inference -> Lock-free queue dispatch -> Actuation / downstream consumer.
2. **Common Traps & Edge Cases**:
   - *Memory Allocation During Runtime*: Dynamic `malloc()` or `new` invocations inside the critical loop trigger unpredictable OS page faults and memory defragmentation pauses.
   - *Thread Priority Inversion*: High-priority perception threads blocked waiting on a mutex held by a low-priority logging thread.
   - *Buffer Bloat & Queue Stalling*: Unbounded FIFO queues buffering stale frames when inference momentarily throttles, causing increasing latency lag.
3. **Engineering Workarounds**:
   - **Pre-Allocation & Memory Locking**: Pre-allocate all frame buffers and call `mlockall(MCL_CURRENT | MCL_FUTURE)` to lock process pages into physical RAM, preventing swapping and major page faults.
   - **Lock-Free Circular Queues (Single-Producer Single-Consumer - SPSC)**: Use atomic pointer swaps and ring buffers to exchange data between capture and inference threads without mutex locks.
   - **Drop-Oldest Queue Policy**: When the inference worker is busy, discard arriving frames immediately or overwrite the oldest unconsumed buffer to ensure the system always processes the freshest sensory state.

---

## Deployment & Real-time Notes
- **PREEMPT_RT Configuration**:
  - Set thread scheduling policy to `SCHED_FIFO` or `SCHED_RR` with high priority:
    ```c
    struct sched_param param;
    param.sched_priority = 80;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    ```
- **CPU Core Pinning & Isolation**:
  - Isolate dedicated CPU cores from the OS scheduler using the Linux kernel boot parameter `isolcpus=2,3` and bind real-time capture/inference threads using `pthread_setaffinity_np`.
- **Latency Telemetry**:
  - Embed hardware nanosecond timestamps in frame metadata at sensor capture, DMA ingress, GPU kernel launch, and inference completion to isolate the exact source of any latency spike.

---
title: Real-Time Systems - Production Pipeline & Engineering Workarounds
type: production-playbook
domain: Real-Time Systems
tags:
  - playbook
  - engineering
  - production
  - workarounds
  - preempt-rt
  - jitter
  - memory-locking
  - real-time
updated: 2026-09-08
aliases:
  - Real-Time Playbook
  - Systems Playbook
---

# 🛠️ Real-Time Systems: Production Pipeline, Traps & Workarounds

A practitioner's guide to eliminating latency jitter, configuring Linux PREEMPT_RT kernel environments, and achieving deterministic execution for robotics and autonomous vision.

Related notes: [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]], [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]].

---

## 1. End-to-End Production Pipeline Architecture

```mermaid
flowchart LR
    Sensor["Hardware Camera / Sensor Interrupt"] --> RealTimeThread["PREEMPT_RT Thread: SCHED_FIFO Priority 95"]
    RealTimeThread --> Mlock["mlockall: Pinned RAM Pages"]
    Mlock --> ShmPool["Iceoryx2 Zero-Copy Shared Memory Segment"]
    ShmPool --> TRT["Async TensorRT Forward Pass on CUDA Stream"]
    TRT --> Actuator["Zero-Copy IPC Event to CAN / EtherCAT Actuator Bus"]

```

---

## 2. Common Traps, Edge Cases & Hardware Bottlenecks

### Trap 1: Linux Memory Page Faults (Major Page Fault Jitter)
- **Problem**: When a C++ or Rust thread allocates memory dynamically via `malloc` or accesses an unpaged virtual memory address, the CPU triggers a Page Fault interrupt. The Linux kernel must find a physical RAM page, zero it out, and update page tables, introducing a **$10-50\text{ ms}$ latency spike**.

### Trap 2: CPU Dynamic Frequency Scaling (C-States & P-States)
- **Problem**: Modern CPUs enter low-power sleep states (C-states like C1E/C6) when idle. Transitioning the CPU core back to full frequency takes up to **100 microseconds**, introducing variable jitter into vision pipelines.

### Trap 3: Memory Bandwidth Contention from Background Processes
- **Problem**: An unisolated logging or monitoring daemon running on a neighboring CPU core can saturate the shared L3 cache or memory controller bus, degrading the deterministic execution of the real-time perception pipeline by up to 50%.

---

## 3. Battle-Tested Engineering Workarounds

### Workaround 1: Total Virtual Memory Locking (`mlockall`)
During process startup, lock the application's entire virtual address space into physical RAM to prevent any future page faults:

```c++
#include <sys/mman.h>
#include <sched.h>
#include <iostream>

void lock_realtime_memory() {
    // Lock all current and future mapped pages into physical RAM
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        std::cerr << "Failed to lock memory via mlockall!" << std::endl;
    }
    
    // Pre-fault stack to ensure memory pages are physically committed
    unsigned char dummy[8 * 1024 * 1024]; // 8 MB stack pre-allocation
    std::fill(std::begin(dummy), std::end(dummy), 0);
}
```

### Workaround 2: CPU Core Isolation and Real-Time Scheduling
Isolate dedicated CPU cores for vision inference using the Linux kernel `isolcpus` boot parameter:
```text
# Append to kernel cmdline:
isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3
```
In application code, bind the perception thread to the isolated core and set real-time priority:
```c++
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(2, &cpuset);
pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

struct sched_param param;
param.sched_priority = 95;
pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
```

### Workaround 3: Disabling CPU Power C-States
Force the CPU to run at locked maximum frequency without entering power-saving states:
```bash
# Disable CPU idle states for minimum wake latency
sudo cpupower idle-set -D 0
sudo cpupower frequency-set -g performance
```

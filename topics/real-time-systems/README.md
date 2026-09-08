# Real-Time Systems Playbook

# Overview
Real-Time Computer Vision and Perception Systems operate under strict temporal deadlines where correctness depends not only on the logical output of the neural network or algorithm, but also on the deterministic instant at which the result is delivered. Applications range from autonomous vehicle emergency braking ($<30\text{ ms}$) and high-speed robotic sorting ($<5\text{ ms}$) to surgical tracking and closed-loop drone flight control.

## SOTA & Research
- **Seminal & Modern Papers & Frameworks**:
  - *Cyclic Asynchronous Pipelines* (Kopetz, Real-Time Systems Principles): Time-triggered vs. event-triggered architectures in mission-critical embedded computing.
  - *ROS 2 (Robot Operating System 2)*: Built on Data Distribution Service (DDS) with Quality of Service (QoS) profiles guaranteeing deterministic transport, deadline management, and zero-copy intra-process communication.
  - *PREEMPT_RT Real-Time Linux*: Kernel patchset converting standard Linux into a hard real-time operating system with deterministic interrupt response and priority inheritance.
  - *Anytime Neural Networks & Early-Exit Architectures* (Teerapittayanon et al., 2016; Huang et al., 2018): Multi-scale networks capable of halting execution early and returning intermediate valid predictions when approaching strict deadline expirations.
- **Evaluation Benchmarks & Metrics**:
  - Worst-Case Execution Time (WCET), Latency Jitter (standard deviation of execution latency), Frame Drop Rate.
  - Deadline Miss Ratio (% of frames exceeding the deadline budget).
  - Glass-to-Glass Latency: Total time elapsed from photons hitting the camera CMOS sensor to actuation or display output.

## Architecture Alternatives & Trade-offs
| Architecture Pattern | Latency Jitter | Throughput | Computational Overhead | Primary Domain |
| :--- | :--- | :--- | :--- | :--- |
| **Pipelined Multi-Threaded Ring Buffers** | Moderate (governed by OS scheduler) | Highest | Low | Standard high-frame-rate video analytics |
| **ROS 2 Zero-Copy Intra-Process (Loaned Messages)** | Low-to-Moderate | High | Very Low | Multi-node robotic perception stacks |
| **PREEMPT_RT Hard Real-Time Loops** | Lowest (<50 microseconds jitter) | Deterministic | Moderate (requires careful lock-free design) | Closed-loop industrial control & actuation |
| **Hardware-Triggered FPGA Pipelines** | Truly Deterministic (Zero OS jitter) | Line-Rate | High hardware engineering cost | High-speed sorting, defense, aerospace |

## Popular Repos & Integrations
- **[ROS 2 (Humble / Iron / Jazzy)](https://github.com/ros2/ros2)**: Production robotics middleware with real-time DDS middleware and lifecycle nodes.
- **[Cyclone DDS](https://github.com/eclipse-cyclonedds/cyclonedds)**: High-performance, low-latency DDS implementation with shared-memory (Iceoryx) zero-copy transport.
- **[Iceoryx](https://github.com/eclipse-iceoryx/iceoryx)**: True zero-copy inter-process communication mechanism based on shared memory for high-bandwidth camera/LiDAR streams.
- **Tooling Integrations**:
  - **FiftyOne**: Log time-to-detection and confidence drifts across varying frame-rate streams to detect degradation during camera throttling.
  - **Rerun**: Native high-speed telemetry viewer capable of logging time-series latency plots, frame intervals, and queue depths alongside visual feeds.

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

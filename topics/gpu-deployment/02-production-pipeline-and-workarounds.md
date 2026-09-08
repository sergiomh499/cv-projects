---
title: GPU Deployment - Production Pipeline & Engineering Workarounds
type: production-playbook
domain: GPU Deployment
tags:
  - playbook
  - engineering
  - production
  - workarounds
  - cuda-graphs
  - fp8
  - gpu
updated: 2026-09-08
aliases:
  - GPU Playbook
  - Acceleration Playbook
---

# 🛠️ GPU Deployment: Production Pipeline, Traps & Workarounds

A practitioner's guide to profiling, compiling, and deploying low-latency vision engines using TensorRT, CUDA Graphs, and multi-stream pipelines.

Related notes: [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]], [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]].

---

## 1. End-to-End Production Pipeline Architecture

```mermaid
flowchart LR
    HostMem[Pinned Host Buffer: cudaHostAlloc] --> AsyncH2D[Asynchronous H2D Copy on Stream 1]
    AsyncH2D --> TRTGraph[TensorRT Executed inside Fused CUDA Graph]
    TRTGraph --> AsyncD2H[Asynchronous D2H Copy on Stream 2]
    AsyncD2H --> Downstream[Zero-Copy IPC Stream to Shared Memory]
```

---

## 2. Common Traps, Edge Cases & Hardware Bottlenecks

### Trap 1: Synchronous `cudaMemcpy` CPU Pipeline Stalls
- **Problem**: Calling standard `cudaMemcpy()` blocks the host CPU thread until the physical PCIe data transfer completes. This serializes execution, preventing the CPU from preparing the next video frame while the GPU computes.

### Trap 2: Dynamic Shape Re-allocation Jitter
- **Problem**: Deploying TensorRT engines with dynamic batch or image shapes (`-1x3x-1x-1`) forces TensorRT to dynamically reallocate intermediate activation scratchpads when shapes change, introducing $50-100\text{ ms}$ latency spikes.

### Trap 3: Context Switching Overhead across Multiple Processes
- **Problem**: Running multiple separate OS processes accessing the same GPU forces the GPU hardware scheduler to continuously context-switch time slices, cutting overall throughput by 40%.

---

## 3. Battle-Tested Engineering Workarounds

### Workaround 1: Static Profile Compilation with Double Buffering
Always compile TensorRT engines with fixed static shapes matching production camera feeds. Implement double-buffered asynchronous execution:

```c++
// Double-buffering pattern across independent CUDA streams
cudaStream_t stream[2];
cudaStreamCreate(&stream[0]);
cudaStreamCreate(&stream[1]);

// Frame 0 executes on Stream 0 while Frame 1 transfers on Stream 1
cudaMemcpyAsync(d_input[0], h_input[0], size, cudaMemcpyHostToDevice, stream[0]);
context->enqueueV3(stream[0]);

cudaMemcpyAsync(d_input[1], h_input[1], size, cudaMemcpyHostToDevice, stream[1]);
context->enqueueV3(stream[1]);
```

### Workaround 2: Capturing CUDA Graphs for Sub-Millisecond Launch
Eliminate CPU launch overhead by capturing inference into a static executable graph:

```python
import torch

# Warmup model
stream = torch.cuda.Stream()
model_trt.warmup(dummy_input)

# Capture graph
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(graph, stream=stream):
    static_output = model_trt(static_input)

# In production loop: single atomic hardware launch
def inference(real_input):
    static_input.copy_(real_input)  # In-place copy to static buffer
    graph.replay()                  # 0 microsecond CPU scheduling latency
    return static_output
```

### Workaround 3: NVIDIA MPS (Multi-Process Service)
When multiple client processes must query a shared GPU, enable **NVIDIA MPS**:
```bash
# Enable MPS control daemon
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
```
MPS multiplexes CUDA contexts onto the GPU simultaneously, eliminating context-switch overhead.

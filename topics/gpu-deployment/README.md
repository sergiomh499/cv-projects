---
title: GPU Deployment Playbook
tags:
  - hardware-deployment
  - gpu
  - tensorrt
  - vulkan
  - ncnn
  - flash-attention
  - cuda
updated: 2026-09-08
aliases:
  - GPU Deployment
---

# GPU Deployment Playbook

# Overview
GPU Deployment focuses on maximizing inference throughput, minimizing end-to-end latency, and optimizing memory efficiency across desktop workstations, cloud datacenters, and embedded edge systems (NVIDIA RTX/Orin, AMD ROCm, and Vulkan-enabled mobile/integrated GPUs).

Related notes: [[topics/fpga-deployment/README|FPGA Deployment]], [[topics/real-time-systems/README|Real-Time Systems]], [[topics/object-detection/README|Object Detection]].

---

## SOTA & Research (2023–2026 Breakthroughs)

1. **TensorRT 10.x & TensorRT-LLM Execution Engine** (NVIDIA, 2023–2025)
   - *Key Innovation*: Modernized compilation graph eliminating engine rebuilds for dynamic shapes, automated native FP8 (E4M3 / E5M2) execution on Ada Lovelace and Hopper architectures, and seamless compilation via `torch.compile(backend="tensorrt")`.
   - [Official Documentation & Code](https://github.com/NVIDIA/TensorRT)

2. **FlashAttention-2 & FlashAttention-3** (Dao et al., Princeton / Stanford, 2023 / 2024)
   - *Key Innovation*: Re-architects attention computation to optimize GPU SRAM memory hierarchy, utilizing warp-specialized asynchronous hardware copies and FP8 Tensor Cores to achieve up to 75% theoretical peak FLOPs.
   - [Paper: arXiv:2307.08691](https://arxiv.org/abs/2307.08691) | [Official Code](https://github.com/Dao-AILab/flash-attention)

3. **Tencent NCNN Vulkan Compute Shaders** (2023 / 2024)
   - *Key Innovation*: High-performance, cross-vendor neural network inference framework utilizing low-overhead Vulkan SPIR-V compute shaders. Provides GPU acceleration across non-CUDA hardware (Intel Arc, AMD Radeon, Qualcomm Adreno, Apple Silicon).
   - [Official Code](https://github.com/Tencent/ncnn)

4. **Triton Inference Server Architecture v2** (NVIDIA, 2023 / 2024)
   - *Key Innovation*: Production serving infrastructure with dynamic batching, zero-copy shared memory IPC, concurrent model execution, and multi-GPU routing.
   - [Official Code](https://github.com/triton-inference-server/server)

---

### Quantitative SOTA Benchmark Comparison (NVIDIA RTX 4090 / Jetson Orin)
| Engine / Runtime | Model Architecture | Batch Size | Latency (FP16 ms) | Throughput (FPS) | VRAM Footprint | License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TensorRT 10 (CUDA)** | YOLOv8-X (640x640) | 1 | 2.85 ms | 350 FPS | 420 MB | Apache-2.0 |
| **TensorRT 10 (FP8)** | ViT-Base (224x224) | 32 | 4.10 ms | 7,800 FPS | 1.1 GB | Apache-2.0 |
| **Triton Server (TRT Engine)**| ResNet-50 | 64 | 3.20 ms | 20,000 FPS | 2.4 GB | BSD-3-Clause |
| **NCNN Vulkan Compute** | YOLOv8-S (640x640) | 1 | 8.40 ms | 119 FPS | 280 MB | BSD-3-Clause |
| **ONNX Runtime (CUDA EP)** | YOLOv8-X (640x640) | 1 | 4.90 ms | 204 FPS | 780 MB | MIT |

---

## Commercial Usability & License Audit
- **Commercial Permissive (Safe)**:
  - **NVIDIA TensorRT (`NVIDIA/TensorRT`)**: Open-source components are licensed under **Apache-2.0**. Commercial distribution with NVIDIA driver binaries is fully supported and standard industry practice.
  - **Triton Inference Server (`triton-inference-server/server`)**: **BSD-3-Clause**. Completely free for proprietary cloud/enterprise serving.
  - **Tencent NCNN (`Tencent/ncnn`)**: **BSD-3-Clause**. Safe for proprietary mobile apps and embedded devices without source disclosure.
  - **FlashAttention (`Dao-AILab/flash-attention`)**: **BSD-3-Clause**.
  - **ONNX Runtime (`microsoft/onnxruntime`)**: **MIT**.

---

## Architecture Alternatives & Trade-offs
| Runtime Engine | Target Hardware | Precision Support | Ecosystem Strengths | Key Constraints |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA TensorRT** | NVIDIA (GeForce, RTX, Jetson, Hopper) | FP32, FP16, INT8, FP8 | Peak performance, deepest kernel fusion | Tied strictly to NVIDIA hardware |
| **Vulkan Compute (NCNN / Kompute)**| Cross-Platform (Intel, AMD, Mali, Adreno) | FP32, FP16 | Universal portability across mobile & embedded | Manual memory management, fewer auto-tuned transformer kernels |
| **ONNX Runtime (CUDA / TensorRT EP)**| Cross-Platform / NVIDIA | FP32, FP16, INT8 | Clean multi-backend API, rapid prototyping | Slight abstraction layer overhead vs pure TensorRT C++ API |
| **Triton Inference Server** | Cloud Datacenter & Edge Clusters | Multi-backend (TRT, PyTorch, ONNX) | Production routing, dynamic batching, metrics | Overhead for small standalone single-board setups |

---

## Popular Repos & Integrations
- **[NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)**: Open-source components, plugins, and parsers for high-performance deep learning inference (Apache-2.0).
- **[triton-inference-server/server](https://github.com/triton-inference-server/server)**: Scalable enterprise serving engine for cloud and edge AI (BSD-3-Clause).
- **[Tencent/ncnn](https://github.com/Tencent/ncnn)**: High-performance neural network inference framework optimized for cross-platform mobile and embedded GPUs via Vulkan (BSD-3-Clause).
- **[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)**: Fast, memory-efficient exact attention algorithms for CUDA and PyTorch (BSD-3-Clause).
- **Tooling Integrations**:
  - **FiftyOne**: Connect remote dataset evaluation jobs directly to GPU-accelerated inference endpoints to benchmark mAP vs batch size.
  - **Rerun**: Visualize real-time GPU inference streams, memory throughput, and timing marks over high-speed C++ and Python SDKs.

---

## End-to-End Pipeline & Workarounds
1. **Pipeline Stages**:
   - PyTorch checkpoint export -> ONNX graph simplification (`onnxsim`) -> TensorRT Engine compilation (`trtexec`) -> C++ / Python zero-copy buffer binding -> Asynchronous stream execution (`cudaStream_t`) -> Host-to-Device asynchronous copy.
2. **Common Traps & Edge Cases**:
   - *Host-Device Memory Transfers*: Copying unpinned pageable memory between CPU RAM and GPU VRAM over PCIe often consumes more time than the actual neural network forward pass.
   - *Dynamic Batching Jitter*: Using unbounded dynamic shapes causes TensorRT to re-allocate scratchpad memory dynamically or select suboptimal kernels.
   - *CPU Post-Processing Bottlenecks*: Running Python-based NMS or OpenCV warping on CPU after a 2ms GPU forward pass ruins end-to-end frame rate.
3. **Engineering Workarounds**:
   - **Pinned Memory (`cudaHostAlloc`) & CUDA Streams**: Allocate page-locked host memory to enable simultaneous bidirectional PCIe DMA transfers overlapping with kernel execution.
   - **Fused Post-Processing Plugins**: Embed bounding box decoding and NMS directly into the TensorRT engine graph using `EfficientNMS_TRT`.
   - **Fixed-Profile Engine Optimization**: Profile and compile the TensorRT engine for specific operational dimensions (`min`, `opt`, and `max` shapes) rather than fully unconstrained dynamic axes.

---

## Deployment & Real-time Notes
- **Profiling with Nsight Systems**:
  - Run `nsys profile -t cuda,nvtx,osrt --stats=true ./inference_app` to identify kernel bubbles, synchronization stalls, and memory bandwidth ceilings.
- **Vulkan Cross-Platform Deployment**:
  - When targeting non-NVIDIA embedded systems (e.g., Raspberry Pi 5 with VideoCore VII, or Intel Arc integrated GPUs), compile models with Tencent NCNN using Vulkan compute shaders for 5-10x acceleration over CPU execution.
- **CUDA Multi-Stream Concurrency**:
  - Create multiple independent CUDA streams to handle concurrent video decoding, model execution, and visualization pipelines in parallel without blocking the main event loop.

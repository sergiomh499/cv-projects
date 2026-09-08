---
title: GPU Deployment Playbook
tags:
  - hardware-deployment
  - gpu
  - tensorrt
  - vulkan
  - cuda
  - triton
  - real-time
updated: 2026-09-08
aliases:
  - GPU Deployment
---

# GPU Deployment Playbook

# Overview
GPU Deployment focuses on maximizing inference throughput, minimizing end-to-end latency, and optimizing memory efficiency across desktop, workstation, cloud datacenter, and embedded edge graphics processors (e.g., NVIDIA RTX/Datacenter, Jetson Orin, AMD ROCm, and Vulkan-enabled mobile/integrated GPUs).

Related notes: [[topics/fpga-deployment/README|FPGA Deployment]], [[topics/real-time-systems/README|Real-Time Systems]], [[topics/object-detection/README|Object Detection]].

## SOTA & Research
- **Recent Breakthroughs (2023–2026)**:
  - *TensorRT 10.x & TensorRT-LLM* (NVIDIA, 2023 / 2024): Modernized graph compilation engine featuring native dynamic shape compilation without re-building engines, automated FP8 (E4M3/E5M2) execution, and seamless PyTorch integration via `torch.compile(backend="tensorrt")`.
  - *FlashAttention-2 & FlashAttention-3* (Dao et al., 2023 / 2024) - [arXiv:2307.08691](https://arxiv.org/abs/2307.08691): SOTA exact attention optimization leveraging warp-specialized asynchronous hardware copy instructions on Hopper and Ada Lovelace GPUs, reaching up to 75% theoretical peak FLOPs.
  - *Vulkan Kompute & NCNN SPIR-V Shaders* (2023 / 2024): Portable, cross-platform shader kernels providing non-CUDA GPU compute across Intel Arc, AMD Radeon, ARM Mali, and Qualcomm Adreno architectures without driver lock-in.
  - *FasterTransformer / TensorRT Inference Server v2* (NVIDIA, 2023 / 2024): Scalable asynchronous execution queues and zero-copy shared memory IPC mechanisms across multi-model pipelines.

### Quantitative SOTA Benchmark Comparison (NVIDIA RTX 4090 / Jetson Orin)
| Engine / Runtime | Model Architecture | Batch Size | Latency (FP16 ms) | Throughput (FPS) | VRAM Footprint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TensorRT 10 (CUDA)** | YOLOv8-X (640x640) | 1 | 2.85 ms | 350 FPS | 420 MB |
| **TensorRT 10 (FP8)** | ViT-Base (224x224) | 32 | 4.10 ms | 7,800 FPS | 1.1 GB |
| **Triton Server (TRT Engine)**| ResNet-50 | 64 | 3.20 ms | 20,000 FPS | 2.4 GB |
| **NCNN Vulkan Compute** | YOLOv8-S (640x640) | 1 | 8.40 ms | 119 FPS | 280 MB |
| **ONNX Runtime (CUDA EP)** | YOLOv8-X (640x640) | 1 | 4.90 ms | 204 FPS | 780 MB |

## Architecture Alternatives & Trade-offs
| Runtime Engine | Target Hardware | Precision Support | Ecosystem Strengths | Key Constraints |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA TensorRT** | NVIDIA (GeForce, RTX, Jetson, Hopper) | FP32, FP16, INT8, FP8 | Peak performance, deepest kernel fusion | Tied strictly to NVIDIA hardware |
| **Vulkan Compute (NCNN / Kompute)**| Cross-Platform (Intel, AMD, Mali, Adreno) | FP32, FP16 | Universal portability across mobile & embedded | Manual memory management, fewer auto-tuned transformer kernels |
| **ONNX Runtime (CUDA / TensorRT EP)**| Cross-Platform / NVIDIA | FP32, FP16, INT8 | Clean multi-backend API, rapid prototyping | Slight abstraction layer overhead vs pure TensorRT C++ API |
| **Triton Inference Server** | Cloud Datacenter & Edge Clusters | Multi-backend (TRT, PyTorch, ONNX) | Production routing, dynamic batching, metrics | Overhead for small standalone single-board setups |

## Popular Repos & Integrations
- **[NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)**: Open-source repository for TensorRT parsers, open-source plugins, and deep learning samples.
- **[triton-inference-server/server](https://github.com/triton-inference-server/server)**: Scalable enterprise serving engine for cloud and edge AI.
- **[Tencent/ncnn](https://github.com/Tencent/ncnn)**: High-performance neural network inference framework optimized for cross-platform mobile and embedded GPUs via Vulkan.
- **[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)**: Fast, memory-efficient exact attention algorithms for CUDA and PyTorch.
- **Tooling Integrations**:
  - **FiftyOne**: Connect remote dataset evaluation jobs directly to GPU-accelerated inference endpoints to benchmark mAP vs batch size.
  - **Rerun**: Visualize real-time GPU inference streams, memory throughput, and timing marks over high-speed C++ and Python SDKs.

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

## Deployment & Real-time Notes
- **Profiling with Nsight Systems**:
  - Run `nsys profile -t cuda,nvtx,osrt --stats=true ./inference_app` to identify kernel bubbles, synchronization stalls, and memory bandwidth ceilings.
- **Vulkan Cross-Platform Deployment**:
  - When targeting non-NVIDIA embedded systems (e.g., Raspberry Pi 5 with VideoCore VII, or Intel Arc integrated GPUs), compile models with Tencent NCNN using Vulkan compute shaders for 5-10x acceleration over CPU execution.
- **CUDA Multi-Stream Concurrency**:
  - Create multiple independent CUDA streams to handle concurrent video decoding, model execution, and visualization pipelines in parallel without blocking the main event loop.

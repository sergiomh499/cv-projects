# GPU Deployment Playbook

# Overview
GPU Deployment focuses on maximizing inference throughput, minimizing end-to-end latency, and optimizing memory efficiency across desktop, workstation, cloud datacenter, and embedded edge graphics processors (e.g., NVIDIA RTX/Datacenter, Jetson Orin, AMD ROCm, and Vulkan-enabled mobile/integrated GPUs).

## SOTA & Research
- **Seminal & Modern Papers & Frameworks**:
  - *TensorRT* (NVIDIA): Compiler and execution engine optimizing neural network graphs through kernel fusion, precision calibration (FP32, FP16, INT8, FP8), and hardware-specific kernel auto-tuning.
  - *Triton Inference Server* (NVIDIA): Production-grade multi-model, multi-GPU serving engine supporting dynamic batching, concurrent model execution, and model pipelining.
  - *FlashAttention / FlashAttention-2* (Dao et al., 2022, 2023): Fast, memory-efficient exact attention algorithms optimizing GPU SRAM memory accesses.
  - *Vulkan Kompute & NCNN* (Tencent / Khronos): Cross-vendor, cross-platform GPU compute utilizing modern low-overhead Vulkan API SPIR-V shaders on non-CUDA hardware.
  - *FP8 Format for Deep Learning* (Micikevicius et al., 2022): Introduction of E4M3 and E5M2 floating-point representations doubling compute throughput on Hopper/Ada Lovelace architectures.
- **Evaluation Benchmarks & Metrics**:
  - Latency (p50, p95, p99 in milliseconds), Throughput (Inferences per second / FPS).
  - GPU Utilization (Compute SM activity vs Memory Bandwidth saturation via Nsight Systems), VRAM footprint.

## Architecture Alternatives & Trade-offs
| Runtime Engine | Target Hardware | Precision Support | Ecosystem Strengths | Key Constraints |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA TensorRT** | NVIDIA (GeForce, RTX, Jetson, Hopper) | FP32, FP16, INT8, FP8 | Peak performance, deepest kernel fusion | Tied strictly to NVIDIA hardware |
| **Vulkan Compute (NCNN / Kompute)**| Cross-Platform (Intel, AMD, Mali, Adreno, Apple) | FP32, FP16 | Universal portability across mobile & embedded | Manual memory management, fewer auto-tuned transformer kernels |
| **ONNX Runtime (CUDA / TensorRT EP)**| Cross-Platform / NVIDIA | FP32, FP16, INT8 | Clean multi-backend API, rapid prototyping | Slight abstraction layer overhead vs pure TensorRT C++ API |
| **Triton Inference Server** | Cloud Datacenter & Edge Clusters | Multi-backend (TRT, PyTorch, ONNX, vLLM) | Production routing, dynamic batching, metrics | Overhead for small standalone single-board setups |

## Popular Repos & Integrations
- **[NVIDIA TensorRT](https://github.com/NVIDIA/TensorRT)**: Open-source components, plugins, and parsers for high-performance deep learning inference.
- **[Triton Inference Server](https://github.com/triton-inference-server/server)**: Scalable, enterprise serving engine for cloud and edge AI.
- **[Tencent NCNN](https://github.com/Tencent/ncnn)**: High-performance neural network inference framework optimized for mobile platforms via Vulkan.
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

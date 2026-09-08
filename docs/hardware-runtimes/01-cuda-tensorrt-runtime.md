---
title: "NVIDIA CUDA & TensorRT Deep Architecture, Compilation & Testing Reality"
type: hardware-runtime-guide
domain: GPU Deployment & Acceleration
tags:
  - hardware-runtime
  - cuda
  - tensorrt
  - jetson-orin
  - layer-fusion
  - nsys
  - trtexec
updated: 2026-09-08
aliases:
  - CUDA & TensorRT Runtime Guide
---

# 🟢 NVIDIA CUDA & TensorRT: Deep Architecture, Compilation & Testing Reality

## 1. Physical Hardware Execution Model
- **Streaming Multiprocessors (SMs)**: Hardware execution units comprising ALU INT32/FP32 cores, Tensor Cores (4th/5th gen for FP8/INT8/INT4/FP16), and Special Function Units (SFU for transcendental math).
- **Warp Schedulers**: Instructions execute in SIMT (Single Instruction, Multiple Threads) groups of 32 threads (warps).
- **Memory Hierarchy**:
  - Registers (fastest, private to thread).
  - Shared Memory / L1 Cache ($128-256\,\text{KB}$ per SM, software-managed low-latency scratchpad).
  - L2 Cache ($32-96\,\text{MB}$ shared across all SMs).
  - Device Global Memory (VRAM: GDDR6X or LPDDR5 on Jetson).

---

## 2. TensorRT Optimizing Compiler Workflow
TensorRT transforms an unoptimized ONNX computation graph into an optimized execution engine (`.engine`):
1. **Vertical Layer Fusion**:
   - `Conv2D + BatchNorm + ReLU` fused into a single compound kernel (`CBR`), eliminating round-trip writes to global VRAM between layers.
2. **Horizontal Layer Fusion**:
   - Multi-head self-attention projections ($W_q, W_k, W_v$) with identical input tensors are combined into a single matrix multiply kernel.
3. **Kernel Auto-Tuning**:
   - Evaluates multiple candidate GEMM/convolution algorithms (Winograd, implicit GEMM, direct convolution) live on target silicon and selects the lowest-latency implementation.
4. **Quantization & Dynamic Range Calibration**:
   - TensorRT 10 explicit Q/DQ nodes scale activations and weights to INT8/FP8 with hardware saturation.

---

## 3. Real Physical Testing & Benchmarking Workflow

```bash
# 1. Exporting & Profiling ONNX model with trtexec directly on target GPU
trtexec \
  --onnx=model.onnx \
  --saveEngine=model.engine \
  --fp16 \
  --int8 \
  --dumpProfile \
  --exportTimes=profile_trace.json \
  --exportLayerInfo=layer_info.json

# 2. Detailed Nsight Systems Execution Trace (profiling memory transfers and kernel overlaps)
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=report_nsys \
  trtexec --loadEngine=model.engine

# 3. Nsight Compute Kernel Profiler (profiling SM occupancy and Tensor Core utilization)
ncu \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
  --target-processes all \
  trtexec --loadEngine=model.engine --iterations=10
```

---

## 4. Certification & Safety Reality
- **Standard CUDA/TensorRT**: Non-deterministic driver scheduler with dynamic memory allocations (`cudaMalloc`).
- **NVIDIA DriveOS / DriveWorks Safety**: Required for ISO 26262 ASIL-D automotive certification on Jetson AGX Orin Industrial, replacing open-source drivers with certified static memory drivers.

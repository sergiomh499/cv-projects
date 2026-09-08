---
title: GPU Deployment - Historical Evolution & Paradigms
type: evolution-guide
domain: GPU Deployment
tags:
  - evolution
  - history
  - architecture
  - cuda
  - tensorrt
  - flashattention
  - triton
  - gpu
updated: 2026-09-08
aliases:
  - GPU Evolution
  - Inference Acceleration History
---

# 📜 GPU Deployment: Historical Evolution & Paradigms

A didactic review charting the journey of GPU acceleration: from early general-purpose computing with OpenGL/Cg shaders to native CUDA, layer-by-layer framework runtimes, static deep learning graph compilers (TensorRT), and automated kernel code generation (Triton).

Related notes: [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]], [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]].

---

## 1. Evolution Timeline: From GPGPU Graphics Shaders to Triton & Blackwell

```mermaid
timeline
    title Evolution of GPU Computing & Inference
    2001-2006 : GPGPU : Mapping linear algebra onto 2D texture rendering via OpenGL
    2007 : CUDA 1.0 : Native C/C++ thread hierarchies (Grids, Blocks, Warps)
    2014 : cuDNN : NVIDIA primitives for 2D convolution and pooling
    2016-2018 : TensorRT 1-4 : Early layer fusion and FP16/INT8 PTQ calibration
    2020 : Ampere & TensorRT 8 : 3rd Gen Tensor Cores, Sparsity (2:4 structured), and CUDA Graphs
    2022 : FlashAttention : Dao et al. : Tiling attention computation in SRAM to avoid HBM memory traffic
    2023 : OpenAI Triton : Python-like domain-specific language generating peak CUDA/ROCm kernels
    2024-2026 : TensorRT 10 & Blackwell : Native FP8/FP4 micro-scaling formats and Transformer Engines
```

---

## 2. Core Paradigm Breakthroughs Explained

### Breakthrough A: Memory Hierarchy and the Roofline Model
GPU performance is fundamentally governed by two limits:
1. **Compute Bound**: The arithmetic processing ALUs/Tensor Cores are fully saturated (FLOPs/s).
2. **Memory Bandwidth Bound**: The ALUs are starved because data cannot be transferred from High-Bandwidth Memory (HBM/VRAM) to fast on-chip SRAM/Registers fast enough.

Most deep learning layers (LayerNorm, Softmax, ReLU, elementwise addition) are severely **memory bandwidth bound**. Stacking these layers sequentially wastes 80% of GPU clock cycles transferring intermediate tensors back and forth across the VRAM memory bus.

---

### Breakthrough B: FlashAttention Kernel Fusion (2022–2024)
Standard Multi-Head Attention computes $S = Q K^T$ and $P = \text{Softmax}(S)$, writing the massive $N \times N$ attention matrix out to slow HBM memory before multiplying by $V$.

Tri Dao et al. introduced **FlashAttention**:
- Splits queries, keys, and values into tiles that fit entirely inside fast on-chip **SRAM** (Shared Memory).
- Computes softmax incrementally using a running scaling factor (online softmax trick).
- Eliminates writing the intermediate $N \times N$ matrix to HBM entirely, accelerating attention by **$2\times - 4\times$** while cutting memory consumption from quadratic $O(N^2)$ to linear $O(N)$.

```mermaid
flowchart TD
    subgraph Traditional Attention (HBM Memory Bottleneck)
        Q1[Q, K Tensors in HBM] --> Matmul1[Matmul: Q x K^T]
        Matmul1 --> Write1[Write N x N Matrix to High-Latency HBM]
        Write1 --> Read1[Read N x N Matrix from HBM to compute Softmax]
        Read1 --> Write2[Write Softmax Matrix to HBM]
    end
    subgraph FlashAttention (Fused On-Chip SRAM Tiling)
        Q2[Q, K, V Loaded in SRAM Tiles] --> FusedKernel[Fused Online Softmax + GEMM in SRAM]
        FusedKernel --> Out[Write Final Output Directly to HBM: Zero N x N Intermediate Storage]
    end
```

---

### Breakthrough C: CUDA Graphs (Zero CPU Launch Overhead)
In high-frame-rate or real-time pipelines, launching hundreds of small CUDA kernels sequentially from Python introduces severe CPU overhead ($5-10\mu\text{s}$ per kernel launch).

**CUDA Graphs** allow capturing the entire sequence of GPU kernels, memory copies, and stream events once. At runtime, the host CPU executes a single instruction:
```c++
cudaGraphLaunch(instance, stream);
```
The GPU executes the complete pipeline end-to-end without waiting for CPU instruction scheduling.

---
title: GPU Deployment MOC
type: MOC
domain: GPU Deployment
tags:
  - moc
  - hardware-deployment
  - gpu
  - tensorrt
  - vulkan
  - flashattention
  - triton
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - GPU Deployment MOC
  - GPU Hub
  - Acceleration MOC
---

# 🗺️ GPU Deployment MOC (Map of Content)

## 📌 Domain Overview & Scope
GPU Deployment focuses on maximizing inference throughput, minimizing end-to-end latency, and optimizing memory efficiency across desktop workstations, cloud datacenters, and embedded edge graphics processors (NVIDIA RTX/Blackwell, Jetson Orin, AMD ROCm, Apple Silicon, and Vulkan mobile GPUs).

Key focus areas include:
- Graph optimization and kernel fusion engines ([[architectures/real-time-unified/tensorrt-and-vulkan|TensorRT 10 & Vulkan Kompute]]).
- Sub-byte quantization: FP16, INT8 PTQ/QAT, and FP8 formats.
- Overcoming CPU launch latency via **CUDA Graphs**.
- Multi-stream concurrency and Triton Inference Server clustering.

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/gpu-deployment/01-historical-evolution-and-paradigms|GPU Deployment: Historical Evolution & Paradigms]]
- **Production Implementation Playbook**: [[topics/gpu-deployment/02-production-pipeline-and-workarounds|GPU Deployment: Production Pipeline, Traps & CUDA Graph Workarounds]]

### 🔬 Core Architecture Deep-Dives (Central Vault)
| Framework / Engine | Target Platforms | Primary Innovation | License | Dedicated Note Link |
| :--- | :--- | :--- | :---: | :---: |
| **TensorRT 10** | NVIDIA Desktop / Jetson | FP8 quantization, FlashAttention fusion, CUDA Graph compilation | **Apache-2.0 / EULA** | [[architectures/real-time-unified/tensorrt-and-vulkan|TensorRT & Vulkan Deep-Dive]] |
| **Vulkan / NCNN** | Cross-Platform (AMD, Intel, ARM) | SPIR-V compute shaders for cross-vendor GPU execution | **BSD-3 / Apache-2.0** | [[architectures/real-time-unified/tensorrt-and-vulkan|TensorRT & Vulkan Deep-Dive]] |

---

## 📊 Standardized SOTA Benchmark Comparison (Latency on RTX 4090)
| Model Architecture | Input Resolution | PyTorch FP32 (Base) | TensorRT FP16 | TensorRT FP8 | Vulkan NCNN (Mobile) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | $3 \times 224 \times 224$ | 2.80 ms | 0.45 ms | **0.25 ms** | 4.80 ms |
| **YOLOv8x** | $3 \times 640 \times 640$ | 18.50 ms | 5.10 ms | **3.10 ms** | 42.0 ms |
| **DINOv2-Base (ViT)** | $3 \times 224 \times 224$ | 24.00 ms | 6.80 ms | **3.80 ms** | 68.0 ms |
| **RF-DETR (Large)** | $3 \times 640 \times 640$ | 32.00 ms | 8.20 ms | **4.90 ms** | 85.0 ms |

---

## ⚖️ Commercial Usability & License Audit
- **Commercial Permissive**:
  - `NVIDIA/TensorRT`: **Apache-2.0** (OSS repo) + NVIDIA EULA (free for commercial hardware).
  - `Tencent/ncnn` & `Kompute`: **BSD-3-Clause** / **Apache-2.0**.
  - `triton-inference-server/server`: **BSD-3-Clause**.
  - *Recommendation*: Standardize on TensorRT 10 for NVIDIA ecosystem hardware; use Vulkan/NCNN for multi-vendor embedded systems.

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]]
- [[topics/fpga-deployment/00-fpga-deployment-moc|FPGA Deployment MOC]]
- [[topics/object-detection/00-object-detection-moc|Object Detection MOC]]

---
title: GPU Deployment Master Index & Directory
tags:
  - computer-vision
  - gpu
  - index
  - sota
  - models
  - playbooks
updated: 2026-09-08
aliases:
  - GPU Deployment Playbook
  - GPU Index
---

# GPU Deployment: Master Index & Domain Guide

> **Obsidian Users**: Access the unified Map of Content at [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]].

## 📌 Executive Brief
GPU Deployment focuses on maximizing deep learning inference throughput, minimizing latency, and optimizing memory efficiency across workstations, cloud clusters, and edge embedded devices (NVIDIA RTX, Jetson Orin, Vulkan mobile). Modern production deployments leverage **TensorRT 10** with FP8 quantization and FlashAttention kernel fusion alongside cross-platform **Vulkan compute pipelines**.

---

## 🧭 Topic Organization & File Structure

```text
topics/gpu-deployment/
├── 00-gpu-deployment-moc.md                      # 🗺️ Master Map of Content & Cross-Domain Hub
├── 01-historical-evolution-and-paradigms.md       # 📜 Historical Evolution (GPGPU -> CUDA -> TensorRT -> FlashAttention)
├── 02-production-pipeline-and-workarounds.md      # 🛠️ Production Pipeline, CUDA Graphs & Double-Buffering
└── README.md                                     # 🧭 Master Directory Index for Git / Web Browsers

architectures/                                    # 🔬 Shared Multi-Task Architecture Vault
└── real-time-unified/
    └── tensorrt-and-vulkan.md                    # TensorRT 10 & Vulkan Kompute: SOTA GPU Inference Runtimes (Apache-2.0 / BSD)
```

---

## 📚 Navigation Directory

| Note Title | Document Type | Description | Link |
| :--- | :---: | :--- | :---: |
| **GPU Deployment MOC** | `MOC` | Master navigational hub, benchmark matrix & license audit | [Open MOC](00-gpu-deployment-moc.md) <br> `[[topics/gpu-deployment/00-gpu-deployment-moc|00-gpu-deployment-moc]]` |
| **Historical Evolution & Paradigms** | `Evolution Guide` | Evolution from early GPGPU shaders to CUDA, roofline memory models, FlashAttention, and CUDA Graphs | [Open Evolution](01-historical-evolution-and-paradigms.md) <br> `[[topics/gpu-deployment/01-historical-evolution-and-paradigms|01-historical-evolution-and-paradigms]]` |
| **Production Pipeline & Workarounds** | `Playbook` | Synchronous `cudaMemcpy` traps, double-buffered CUDA streams, CUDA Graph replay, and NVIDIA MPS | [Open Playbook](02-production-pipeline-and-workarounds.md) <br> `[[topics/gpu-deployment/02-production-pipeline-and-workarounds|02-production-pipeline-and-workarounds]]` |
| **TensorRT & Vulkan Deep-Dive** | `Architecture Vault` | Comprehensive breakdown of layer fusion, FP8 formats, and cross-vendor Vulkan shaders | [Open TensorRT/Vulkan](../../architectures/hardware-and-acceleration-runtimes/tensorrt-and-vulkan.md) <br> `[[architectures/hardware-and-acceleration-runtimes/tensorrt-runtime|tensorrt-and-vulkan]]` |

---

## 📊 Summary SOTA Benchmark Comparison (Latency on RTX 4090)
| Model Architecture | Input Resolution | PyTorch FP32 (Base) | TensorRT FP16 | TensorRT FP8 | Vulkan NCNN (Mobile) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | $3 \times 224 \times 224$ | 2.80 ms | 0.45 ms | **0.25 ms** | 4.80 ms |
| **YOLOv8x** | $3 \times 640 \times 640$ | 18.50 ms | 5.10 ms | **3.10 ms** | 42.0 ms |
| **DINOv2-Base (ViT)** | $3 \times 224 \times 224$ | 24.00 ms | 6.80 ms | **3.80 ms** | 68.0 ms |
| **RF-DETR (Large)** | $3 \times 640 \times 640$ | 32.00 ms | 8.20 ms | **4.90 ms** | 85.0 ms |

---

## ⚖️ Commercial Usability Quick-Audit
- **Commercial Permissive**:
  - `NVIDIA/TensorRT`: **Apache-2.0** (OSS) + NVIDIA EULA.
  - `Tencent/ncnn`: **BSD-3-Clause**.
  - `triton-inference-server/server`: **BSD-3-Clause**.

---
title: FPGA Deployment Master Index & Directory
tags:
  - computer-vision
  - fpga
  - index
  - sota
  - models
  - playbooks
updated: 2026-09-08
aliases:
  - FPGA Deployment Playbook
  - FPGA Index
---

# FPGA Deployment: Master Index & Domain Guide

> **Obsidian Users**: Access the unified Map of Content at [[topics/fpga-deployment/00-fpga-deployment-moc|FPGA Deployment MOC]].

## 📌 Executive Brief
Field-Programmable Gate Arrays (FPGAs) provide deterministic low latency, sub-microsecond jitter, and high power efficiency for edge computer vision in robotics, defense, and industrial automation. Deployments center on coarse-grained instruction-driven DPU overlays (**Vitis AI 3.5**) for rapid turnaround and spatial dataflow streaming (**FINN**) for ultra-low latency sub-byte quantized networks.

---

## 🧭 Topic Organization & File Structure

```text
topics/fpga-deployment/
├── 00-fpga-deployment-moc.md                     # 🗺️ Master Map of Content & Cross-Domain Hub
├── 01-historical-evolution-and-paradigms.md       # 📜 Historical Evolution (RTL -> HLS -> DPU Overlays -> FINN)
├── 02-production-pipeline-and-workarounds.md      # 🛠️ Production Pipeline, Operator Traps & DMA-BUF Workarounds
└── README.md                                     # 🧭 Master Directory Index for Git / Web Browsers

architectures/                                    # 🔬 Shared Multi-Task Architecture Vault
└── real-time-unified/
    └── vitis-ai-and-finn.md                      # Vitis AI 3.5 & FINN: Quantized Neural Inference on FPGAs (Apache-2.0)
```

---

## 📚 Navigation Directory

| Note Title | Document Type | Description | Link |
| :--- | :---: | :--- | :---: |
| **FPGA Deployment MOC** | `MOC` | Master navigational hub, benchmark matrix & license audit | [Open MOC](00-fpga-deployment-moc.md) <br> `[[topics/fpga-deployment/00-fpga-deployment-moc|00-fpga-deployment-moc]]` |
| **Historical Evolution & Paradigms** | `Evolution Guide` | Evolution from RTL VHDL/Verilog to Vivado HLS, DPU processor overlays, and sub-byte LUT-Net | [Open Evolution](01-historical-evolution-and-paradigms.md) <br> `[[topics/fpga-deployment/01-historical-evolution-and-paradigms|01-historical-evolution-and-paradigms]]` |
| **Production Pipeline & Workarounds** | `Playbook` | DPU operator fallback traps, CMA memory pool sizing, direct DMA-BUF zero-copy ingestion | [Open Playbook](02-production-pipeline-and-workarounds.md) <br> `[[topics/fpga-deployment/02-production-pipeline-and-workarounds|02-production-pipeline-and-workarounds]]` |
| **Vitis AI & FINN Deep-Dive** | `Architecture Vault` | Detailed breakdown of DPU instruction overlays vs. spatial dataflow streaming pipelines | [Open Vitis AI/FINN](../../architectures/hardware-and-acceleration-runtimes/vitis-ai-and-finn.md) <br> `[[architectures/hardware-and-acceleration-runtimes/vitis-ai-and-finn|vitis-ai-and-finn]]` |

---

## 📊 Summary SOTA Benchmark Comparison (FPGA Platforms)
| Framework | Target Silicon | Architecture | Quantization | Throughput (FPS) | Latency (ms) | Board Power |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vitis AI 3.5** | Kria KV260 | ResNet-50 | INT8 | 105.0 FPS | 9.5 ms | 10 W |
| **Vitis AI 3.5** | Kria KV260 | YOLOv8-Nano | INT8 | 85.0 FPS | 11.7 ms | 11 W |
| **Vitis AI 3.5** | Alveo V70 (PCIe) | ResNet-50 | INT8 | 2,850.0 FPS | 0.35 ms | 75 W |
| **FINN** | Kria KV260 | CNV ConvNet | 2-bit (QNN) | **1,450.0 FPS**| **0.68 ms** | 7 W |

---

## ⚖️ Commercial Usability Quick-Audit
- **Commercial Permissive**:
  - `Xilinx/Vitis-AI`: **Apache-2.0** runtime; DPU IP cores royalty-free under AMD Xilinx EULA.
  - `Xilinx/finn` & `Xilinx/brevitas`: **Apache-2.0**.

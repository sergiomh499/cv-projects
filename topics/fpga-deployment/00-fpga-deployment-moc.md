---
title: FPGA Deployment MOC
type: MOC
domain: FPGA Deployment
tags:
  - moc
  - hardware-deployment
  - fpga
  - vitis-ai
  - finn
  - edge-ai
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - FPGA Deployment MOC
  - FPGA Hub
  - Hardware Acceleration MOC
---

# 🗺️ FPGA Deployment MOC (Map of Content)

> **Navigation**: [[README|🏠 Central Knowledge Hub]] / **FPGA Deployment
tags:
  - moc
  - hardware-deployment
  - fpga
  - vitis-ai
  - finn
  - edge-ai
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - FPGA Deployment MOC
  - FPGA Hub
  - Hardware Acceleration MOC
---

# 🗺️ FPGA Deployment MOC (Map of Content)

## 📌 Domain Overview & Scope
Field-Programmable Gate Arrays (FPGAs) target deterministic low latency, sub-microsecond jitter, and high power efficiency for real-time computer vision in embedded robotics, defense, industrial sorting, and automotive edge devices. Unlike GPUs which share an instruction-fetch pipeline, FPGAs reconfigure silicon gates directly into deep neural network processing architectures.

Key hardware platforms include AMD/Xilinx Zynq UltraScale+ MPSoCs, Kria KV260/KR260 vision starter kits, Versal Adaptive SoCs, and Intel Agilex FPGAs.

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/fpga-deployment/01-historical-evolution-and-paradigms|FPGA Deployment: Historical Evolution & Paradigms]]
- **Production Implementation Playbook**: [[topics/fpga-deployment/02-production-pipeline-and-workarounds|FPGA Deployment: Production Pipeline, Thermal Traps & Zero-Copy DMA]]

### 🔬 Core Architecture Deep-Dives (Central Vault)
| System / Framework | Paradigm | Primary Innovation | License | Dedicated Note Link |
| :--- | :--- | :--- | :---: | :---: |
| **AMD Xilinx Vitis AI 3.5** | Instruction-Driven DPU | Overlay DPU cores executing compiled xmodel instructions without resynthesis | **Apache-2.0 / EULA** | [[architectures/hardware-and-acceleration-runtimes/vitis-ai-and-finn|Vitis AI & FINN Deep-Dive]] |
| **FINN & Brevitas** | Spatial Dataflow Streaming | Compiles quantized neural networks (1-4 bit) into dedicated hardware LUT pipelines | **Apache-2.0** | [[architectures/hardware-and-acceleration-runtimes/vitis-ai-and-finn|Vitis AI & FINN Deep-Dive]] |

---

## 📊 Standardized SOTA Benchmark Comparison (FPGA Platforms)
| Framework | Target Silicon | Architecture | Quantization | Throughput (FPS) | Latency (ms) | Board Power |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vitis AI 3.5** | Kria KV260 | ResNet-50 | INT8 | 105.0 FPS | 9.5 ms | 10 W |
| **Vitis AI 3.5** | Kria KV260 | YOLOv8-Nano | INT8 | 85.0 FPS | 11.7 ms | 11 W |
| **Vitis AI 3.5** | Alveo V70 (PCIe) | ResNet-50 | INT8 | 2,850.0 FPS | 0.35 ms | 75 W |
| **FINN** | Kria KV260 | CNV ConvNet | 2-bit (QNN) | **1,450.0 FPS**| **0.68 ms** | 7 W |

---

## ⚖️ Commercial Usability & License Audit
- **Commercial Permissive**:
  - `Xilinx/Vitis-AI`: **Apache-2.0** for runtime code; DPU IP cores are royalty-free under AMD Xilinx EULA for Xilinx silicon.
  - `Xilinx/finn` & `Xilinx/brevitas`: **Apache-2.0**.
  - *Recommendation*: Standardize on Vitis AI 3.5 for quick deployment of INT8 CNNs/YOLO models; reserve FINN for extreme sub-millisecond line-rate inspection using 2-4 bit quantization.

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]]
- [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]]
- [[topics/object-detection/00-object-detection-moc|Object Detection MOC]]**

## 📌 Domain Overview & Scope
Field-Programmable Gate Arrays (FPGAs) target deterministic low latency, sub-microsecond jitter, and high power efficiency for real-time computer vision in embedded robotics, defense, industrial sorting, and automotive edge devices. Unlike GPUs which share an instruction-fetch pipeline, FPGAs reconfigure silicon gates directly into deep neural network processing architectures.

Key hardware platforms include AMD/Xilinx Zynq UltraScale+ MPSoCs, Kria KV260/KR260 vision starter kits, Versal Adaptive SoCs, and Intel Agilex FPGAs.

---

## 🧭 Navigation & Knowledge Graph
- **Historical Lineage & Evolution**: [[topics/fpga-deployment/01-historical-evolution-and-paradigms|01 Historical Evolution And Paradigms]]
- **Production Implementation Playbook**: [[topics/fpga-deployment/02-production-pipeline-and-workarounds|02 Production Pipeline And Workarounds]]
- **Modern Architecture & Open Problems**: [[topics/fpga-deployment/03-silicon-and-open-problems|03 Silicon And Open Problems]]
- **Classical, Geometric & Hybrid Baselines**: [[topics/fpga-deployment/04-classical-and-hybrid-methods|04 Classical And Hybrid Methods]]

### 🔬 Core Architecture Deep-Dives (Central Vault)
| System / Framework | Paradigm | Primary Innovation | License | Dedicated Note Link |
| :--- | :--- | :--- | :---: | :---: |
| **AMD Xilinx Vitis AI 3.5** | Instruction-Driven DPU | Overlay DPU cores executing compiled xmodel instructions without resynthesis | **Apache-2.0 / EULA** | [[architectures/hardware-and-acceleration-runtimes/vitis-ai-and-finn|Vitis AI & FINN Deep-Dive]] |
| **FINN & Brevitas** | Spatial Dataflow Streaming | Compiles quantized neural networks (1-4 bit) into dedicated hardware LUT pipelines | **Apache-2.0** | [[architectures/hardware-and-acceleration-runtimes/vitis-ai-and-finn|Vitis AI & FINN Deep-Dive]] |

---

## 📊 Standardized SOTA Benchmark Comparison (FPGA Platforms)
| Framework | Target Silicon | Architecture | Quantization | Throughput (FPS) | Latency (ms) | Board Power |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vitis AI 3.5** | Kria KV260 | ResNet-50 | INT8 | 105.0 FPS | 9.5 ms | 10 W |
| **Vitis AI 3.5** | Kria KV260 | YOLOv8-Nano | INT8 | 85.0 FPS | 11.7 ms | 11 W |
| **Vitis AI 3.5** | Alveo V70 (PCIe) | ResNet-50 | INT8 | 2,850.0 FPS | 0.35 ms | 75 W |
| **FINN** | Kria KV260 | CNV ConvNet | 2-bit (QNN) | **1,450.0 FPS**| **0.68 ms** | 7 W |

---

## ⚖️ Commercial Usability & License Audit
- **Commercial Permissive**:
  - `Xilinx/Vitis-AI`: **Apache-2.0** for runtime code; DPU IP cores are royalty-free under AMD Xilinx EULA for Xilinx silicon.
  - `Xilinx/finn` & `Xilinx/brevitas`: **Apache-2.0**.
  - *Recommendation*: Standardize on Vitis AI 3.5 for quick deployment of INT8 CNNs/YOLO models; reserve FINN for extreme sub-millisecond line-rate inspection using 2-4 bit quantization.

---

## 🔗 Related MOCs & Interconnected Domains
- [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]]
- [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]]
- [[topics/object-detection/00-object-detection-moc|Object Detection MOC]]

---

## 📂 All Notes in This Domain

```dataview
TABLE type AS "Note Type", updated AS "Last Updated", status AS "Status"
WHERE contains(file.folder, "topics/fpga-deployment")
SORT file.name ASC
```

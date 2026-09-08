---
title: "NVIDIA DRIVE Thor: Blackwell Architecture, 1000-2000 TFLOPS & Centralized Automotive Superchip"
type: "Hardware Architecture"
domain: "Autonomous Vehicles & Centralized Automotive Superchips"
status: evergreen
updated: 2026-08-15
tags:
  - hardware
  - drive-thor
  - blackwell-gpu
  - nvfp4
  - transformer-engine
  - neoverse-v3ae
  - asil-d
  - autonomous-vehicles
  - automotive-soc
aliases:
  - NVIDIA DRIVE Thor
  - DRIVE Thor
  - Thor Automotive SoC
  - Thor Superchip
  - Blackwell Automotive
---

# 🚗 NVIDIA DRIVE Thor: Blackwell Architecture, 1000-2000 TFLOPS & Centralized Automotive Superchip

## 1. Executive Summary & Hardware Typology

**NVIDIA DRIVE Thor** is NVIDIA's flagship centralized automotive System-on-Chip (SoC), designed to unify the entirety of autonomous vehicle computing workloads onto a single monolithic silicon platform. Fabricated on a customized TSMC 4N process node, DRIVE Thor consolidates high-throughput autonomous driving perception (Multi-Camera Vision Transformers, Bird's-Eye-View [BEV] spatio-temporal representations, LiDAR/Radar point cloud backbones, 3D voxel occupancy networks), end-to-end trajectory planning, multi-modal in-cabin generative AI (multimodal Large Vision-Language Models [VLMs]), digital cockpit instrument clustering, and hard real-time ISO 26262 ASIL-D vehicle dynamics actuation.

DRIVE Thor integrates the **NVIDIA Blackwell GPU architecture** (featuring 5th Generation Tensor Cores with native 4-bit microscaling [NVFP4] arithmetic and 2nd Generation Transformer Engine), server-class **ARM Neoverse V3AE** 64-bit application CPU cores, a dedicated **ISO 26262 ASIL-D Safety Island**, and an ultra-high-bandwidth **LPDDR5X/LPDDR5T memory subsystem** delivering between **1,000 and 2,000 TFLOPS of FP4 compute** in single-SoC configurations and up to **3,000+ TFLOPS** in dual-SoC configurations.

```mermaid
flowchart TD
    subgraph Thor_SoC ["NVIDIA DRIVE Thor Automotive Monolithic SoC"]
        subgraph Neoverse_V3AE_Complex ["Server-Class Application CPU Subsystem"]
            CPU_ARRAY["24x to 32x ARM Neoverse V3AE 64-bit Cores @ 3.0 GHz<br/>Dual 256-bit Scalable Vector Extension 2 'SVE2'<br/>ARM System Memory Management Unit 'SMMUv3'"]
            CPU_SYS_CACHE["32 MB Shared System Level Cache 'SLC'"]
        end

        subgraph Blackwell_GPU_Complex ["Blackwell Architecture GPU Subsystem"]
            SM_BLACKWELL["Blackwell SM Array: 64 to 128 Streaming Multiprocessors<br/>5th Gen Tensor Cores with Native NVFP4 & FP8 Units<br/>2nd Gen Transformer Engine with FlashAttention-3"]
            GPU_L2_CACHE["64 MB Ultra-High-Throughput GPU L2 Cache"]
        end

        subgraph ASIL_D_Safety_Island ["Hardened Real-Time Safety Island 'ASIL-D'"]
            R52_LOCKSTEP["Quad ARM Cortex-R52 in Dual Lockstep Mode<br/>Fail-Operational Actuation & Steering / Braking Gate"]
            SAFETY_WATCHDOG["Hardware Diagnostic & Voltage / Clock Supervisors"]
        end

        subgraph Vision_IO_Subsystem ["High-Speed Sensor Ingestion & Networking"]
            CSI_GMSL["Multi-Port MIPI CSI-2 / GMSL3 Deserializer Hub 'Up to 16x 8MP Cameras'"]
            TSN_100G["100GbE / 10GbE Automotive TSN Ethernet Controller"]
            PCIE_GEN6["PCIe Gen 6 x16 / CXL Interconnect Interface"]
        end

        subgraph Memory_UMA ["Unified Memory Architecture Subsystem"]
            LPDDR5X_CTRL["512-bit LPDDR5X-8533 / LPDDR5T Controller<br/>546 GB/s Aggregate UMA Bandwidth with In-Line ECC"]
            SCF_MESH["High-Speed Coherent Crossbar Fabric 'SCF 2.0'"]
        end
    end

    Neoverse_V3AE_Complex <--> SCF_MESH
    Blackwell_GPU_Complex <--> SCF_MESH
    ASIL_D_Safety_Island <--> SCF_MESH
    Vision_IO_Subsystem <--> SCF_MESH
    SCF_MESH <--> LPDDR5X_CTRL
```

### Hardware Typology Matrix: DRIVE Thor SKU Profiles

| Architectural Dimension | DRIVE Thor Standard | DRIVE Thor High-Throughput | DRIVE Thor Dual-SoC Module |
| :--- | :--- | :--- | :--- |
| **GPU Architecture** | Blackwell (64 SMs) | Blackwell (96 SMs) | Dual Blackwell (192 SMs) |
| **Tensor Cores (5th Gen)** | 256 Tensor Cores | 384 Tensor Cores | 768 Tensor Cores |
| **Peak FP4 Compute (Dense / Sparse)** | **1,000 / 2,000 TFLOPS** | **1,500 / 3,000 TFLOPS** | **3,000 / 6,000 TFLOPS** |
| **Peak FP8 Compute (Dense / Sparse)** | 500 / 1,000 TFLOPS | 750 / 1,500 TFLOPS | 1,500 / 3,000 TFLOPS |
| **Application CPU Complex** | 16x ARM Neoverse V3AE | 24x ARM Neoverse V3AE | 48x ARM Neoverse V3AE |
| **Safety Real-Time Cores** | Dual Lockstep Cortex-R52 | Quad Lockstep Cortex-R52 | Dual Quad Cortex-R52 Islands |
| **Memory Configuration** | 64 GB LPDDR5X (256-bit) | 128 GB LPDDR5X (512-bit) | 256 GB LPDDR5X (Dual 512-bit) |
| **Memory Bandwidth (Peak)** | 273 GB/s | **546.1 GB/s** | **1,092.2 GB/s** |
| **Thermal Dissipation (TDP)** | 100W – 140W | 150W – 200W | 300W – 400W |
| **Cooling Methodology** | Sealed Forced Air / Cold Plate | Automotive Liquid (Water/Glycol) | Automotive Liquid (Dual Plate) |
| **Target Deployment** | L2+/L3 Premium ADAS | L3/L4 Centralized AD + Cockpit | L4/L5 Autonomous Robotaxi Fleets |

---

## 2. Compute Core & Memory Hierarchy

DRIVE Thor overcomes the historical "automotive memory wall" by implementing a monolithic **Unified Memory Architecture (UMA)**. Instead of isolating CPU memory, GPU VRAM, and safety buffers across separate physical busses, Thor multiplexes an ultra-wide LPDDR5X-8533 bus delivering **$546.1\text{ GB/s}$** of coherent memory bandwidth with hardware Single Error Correction, Double Error Detection (SECDED) inline ECC.

```mermaid
flowchart LR
    subgraph Physical_DRAM ["LPDDR5X-8533 Physical Memory Array"]
        DRAM_ARRAY["128 GB LPDDR5X Unified Memory Array<br/>546.1 GB/s Bandwidth with Inline SECDED ECC"]
    end

    subgraph Coherent_Fabric ["System Coherence Fabric 2.0 'SCF'"]
        SCF_CROSS["Multi-Terabit Hardware Cache-Coherent Interconnect<br/>Direct Hardware ARM SMMUv3 Translation Engines"]
    end

    subgraph CPU_Tier ["Neoverse V3AE Cache Hierarchy"]
        V3AE_L1["64KB I-Cache / 64KB D-Cache per Core"]
        V3AE_L2["1MB Dedicated Private L2 per Core"]
        SYS_SLC["32MB Shared System Level Cache 'SLC'"]
    end

    subgraph Blackwell_Tier ["Blackwell GPU Cache Hierarchy"]
        SM_L1_SRAM["256KB Unified L1 Cache / Shared Memory per SM"]
        GPU_L2_ARRAY["64MB Ultra-Wide GPU L2 Cache"]
    end

    subgraph Hardware_Engines ["Dedicated Hardware Blocks"]
        ISP_ENGINE["ISP v6: 16x 4K HDR 120fps Real-Time Streams"]
        SAFETY_ISLAND_SRAM["Dedicated 8MB Lockstep Safety SRAM"]
    end

    Physical_DRAM <--> Coherent_Fabric
    Coherent_Fabric <--> CPU_Tier
    Coherent_Fabric <--> Blackwell_Tier
    Coherent_Fabric <--> Hardware_Engines
```

### Die Layout and Subsystem Specifications

1. **ARM Neoverse V3AE Application Processors**:
   - Implements ARMv9.2-A with dual 256-bit **Scalable Vector Extension 2 (SVE2)** pipelines per core.
   - Server-grade branch predictors and high single-thread integer throughput enable executing non-linear Model Predictive Control (MPC) trajectory optimizers, trajectory smoothing spline solvers, and C++ ROS 2 / AUTOSAR Adaptive middleware with sub-millisecond execution bounds.
   
2. **Blackwell Streaming Multiprocessors (SM)**:
   - Configured with 256 KB of dynamically partitionable shared memory / L1 data cache per SM.
   - Features asynchronous transaction barriers (`arrive-on` and `mbarrier`) that allow direct memory-to-memory data exchange across SM shared memories without host CPU synchronization.

3. **Zero-Copy Camera Ingestion Path**:
   - GMSL3 / MIPI CSI-2 camera frames are captured by the on-chip deserializer hub, processed through the high-dynamic-range Image Signal Processor (ISP v6), and written directly into GPU L2 cache or shared LPDDR5X memory via Direct Memory Access (DMA).
   - Downstream Vision Transformers (e.g. BEVFormer, Sparse4D) consume raw Bayer or YUV422 tensors without single-byte memory copy operations.

---

## 3. Micro-Architectural Mechanics & Data Paths

### 5th Generation Tensor Cores & Native NVFP4 Microscaling

DRIVE Thor's Blackwell GPU introduces native hardware matrix execution for **NVFP4 (4-bit Microscaling Floating-Point)** arithmetic alongside 2nd Generation Transformer Engine acceleration.

```mermaid
flowchart TD
    subgraph Input_Tensors ["4-Bit Quantized Weight & Activation Vectors"]
        T4_WEIGHTS["FP4 Weights 'E2M1 Format: 1s, 2e, 1m'"]
        T4_ACTS["FP4 Activations 'E2M1 Format: 1s, 2e, 1m'"]
        SCALE_BLOCK["16-Element Block Scaling Exponents 'E8M0 FP8 Scales'"]
    end

    subgraph Transformer_Engine_v2 ["Transformer Engine v2 Hardware Pipeline"]
        SCALE_EXP["Hardware Dynamic Range Scale Dequantizer"]
        MMA_BLACKWELL["5th Gen Tensor Core MMA Engine<br/>Native 4x4x4 FP4 Microscaled Matrix Dot Products"]
        FLASH_ATTN3["FlashAttention-3 Hardware Asynchronous Execution Unit"]
    end

    subgraph Accumulator_Stage ["High-Precision Accumulation"]
        FP32_ACC["IEEE 754 FP32 Accumulators"]
        SOFTMAX_NORM["Fused LayerNorm / RMSNorm & RoPE Unit"]
    end

    Input_Tensors --> Transformer_Engine_v2
    SCALE_BLOCK --> SCALE_EXP
    T4_WEIGHTS & T4_ACTS --> MMA_BLACKWELL
    SCALE_EXP --> MMA_BLACKWELL
    MMA_BLACKWELL --> Accumulator_Stage
    Accumulator_Stage --> FLASH_ATTN3
```

#### NVFP4 Microscaling Mathematical Formulation
Standard scalar quantization applies a single scaling factor $\alpha$ across an entire tensor or row ($W \approx \alpha \cdot Q$). In contrast, NVFP4 microscaling applies independent floating-point scaling factors $s_b$ across small, 16-element sub-vectors (blocks):
$$X_{i, j} = s_b \cdot q_{i, j}, \quad \text{where } b = \lfloor j / 16 \rfloor, \quad q_{i, j} \in \text{FP4 (E2M1)}, \quad s_b \in \text{FP8 (E8M0)}$$
- **Mantissa/Exponent Layout**: FP4 (E2M1) provides 1 sign bit, 2 exponent bits, and 1 mantissa bit, representing values $\pm \{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0\}$.
- **Block Scaling**: Hardware decodes the 16-element block scale factor $s_b$ directly inside the Tensor Core register stage, eliminating memory transfer overhead while preserving dynamic range across outlier activations in large Vision Transformers.

### Hardware Multi-Tenancy & Spatial MIG Partitioning

Automotive domain consolidation demands mathematical isolation between safety-critical perception and non-critical infotainment. Thor implements hardware-enforced **Multi-Instance GPU (MIG)** partitioning:

```mermaid
flowchart TD
    subgraph Physical_Blackwell_GPU ["DRIVE Thor Blackwell GPU '64 SMs / 128GB LPDDR5X'"]
        subgraph MIG_Instance_1 ["MIG Instance 1: ASIL-D ADAS Perception '32 SMs'"]
            PERCEPT_VLM["BEV Transformer & Occupancy Grid Pipeline"]
            CAM_IN["Direct 16-Camera Real-Time Ingestion Hub"]
        end

        subgraph MIG_Instance_2 ["MIG Instance 2: Trajectory & Physics '16 SMs'"]
            DIFF_POLICY["Diffusion Policy & MPC Trajectory Optimization"]
        end

        subgraph MIG_Instance_3 ["MIG Instance 3: Cockpit & In-Cabin AI '16 SMs'"]
            COCKPIT_LLM["In-Cabin Multimodal Assistant 'Llama-3-8B FP4'"]
            INSTRUMENT_UI["Instrument Cluster 3D Graphics 'Unreal Engine'"]
        end
    end

    subgraph Hardware_Firewall ["Hardware Isolation Firewall & QoS Arbiter"]
        QOS_GUARD["Memory Bandwidth Throttling & Fault Containment Logic"]
    end

    MIG_Instance_1 <--> Hardware_Firewall
    MIG_Instance_2 <--> Hardware_Firewall
    MIG_Instance_3 <--> Hardware_Firewall
```

- **Fault Containment**: A kernel panic, out-of-memory exception, or unbounded memory loop in the in-cabin LLM (Instance 3) cannot degrade the determinism or latency of the ASIL-D ADAS perception pipeline (Instance 1).
- **Guaranteed QoS**: Hardware memory arbiters allocate guaranteed DRAM bandwidth slices and L2 cache partitions to safety instances.

---

## 4. Numerical Precision & Quantization

DRIVE Thor provides multi-precision execution capability, allowing mixed-precision deployment where feed-forward layers execute in NVFP4, attention layers maintain FP8 precision, and trajectory optimization runs in FP32.

```mermaid
flowchart LR
    Model_Input["FP32 Model Graph"] --> TRT_Quant["TensorRT 11 NVFP4 Quantizer"]
    
    TRT_Quant --> Layer_Analysis{"Layer Sensitivity Metric"}
    Layer_Analysis -->|Attention QK^T & Norms| FP8_Plan["FP8 'E4M3 / E5M2' Layer Plan"]
    Layer_Analysis -->|Feed-Forward & Projections| NVFP4_Plan["NVFP4 'Block Size 16' Layer Plan"]
    
    subgraph Execution_Core ["Blackwell 5th Gen Tensor Core Execution"]
        NVFP4_Plan --> FP4_Core["Native FP4 MMA: 2x Throughput vs FP8"]
        FP8_Plan --> FP8_Core["Native FP8 MMA: 2x Throughput vs FP16"]
    end
```

### Precision Capabilities Matrix (DRIVE Thor Single-SoC 64-SM Configuration)

| Numerical Format | Sign / Exp / Mantissa | Block Size | Hardware Acceleration | Peak Compute (TFLOPS) | Relative Memory Footprint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVFP4 (E2M1)** | 1 Sign, 2 Exp, 1 Mant | 16 elements (E8M0 Scale) | 5th Gen Tensor Core | **1,500 TFLOPS** | 0.125x (vs FP32) |
| **FP8 (E4M3)** | 1 Sign, 4 Exp, 3 Mant | Scalar | 5th Gen Tensor Core | **750 TFLOPS** | 0.25x |
| **FP8 (E5M2)** | 1 Sign, 5 Exp, 2 Mant | Scalar | 5th Gen Tensor Core | **750 TFLOPS** | 0.25x |
| **INT8 / INT4** | Two's Complement Integer | Scalar | 5th Gen Tensor Core | **750 / 1,500 TOPS** | 0.25x / 0.125x |
| **BF16 / FP16** | IEEE Standard Formats | Scalar | 5th Gen Tensor Core | **375 TFLOPS** | 0.5x |
| **TF32** | 1 Sign, 8 Exp, 10 Mant | Scalar | 5th Gen Tensor Core | **187.5 TFLOPS** | 0.5x (Memory 1.0x) |
| **FP32** | IEEE Standard 754 | Scalar | Blackwell CUDA Pipeline| **23.4 TFLOPS** | 1.0x (Baseline) |

---

## 5. Software Stack, SDKs & Driver Interface

The software ecosystem for DRIVE Thor relies on **NVIDIA DRIVE OS 7.x**, integrating a safety-certified **QNX Neutrino Hypervisor**, **TensorRT 11.x**, **CUDA 13.x**, and the **DRIVE AV** autonomous driving perception stack.

```mermaid
flowchart TD
    AppLayer["Automotive Perception & Planning 'AUTOSAR Adaptive / ROS 2'"] --> DRIVE_AV["NVIDIA DRIVE AV Stack"]
    DRIVE_AV --> TRT11["TensorRT 11.x 'NVFP4 Engine + FlashAttention-3'"]
    TRT11 --> CUDA13["CUDA 13.x Driver Runtime"]
    CUDA13 --> QNX_HYP["Safety RTOS / QNX Neutrino Hypervisor"]
    QNX_HYP --> MIG_HW["Blackwell Partitioned Hardware 'MIG / ASIL-D Island'"]
```

### Production Toolchain Workflows & CLI Commands

#### 1. NVFP4 Engine Compilation via TensorRT `trtexec`
```bash
# Build an NVFP4 quantized BEV Transformer engine with FlashAttention-3 plugins
trtexec --onnx=bev_transformer_v3.onnx \
        --saveEngine=bev_transformer_thor_fp4.engine \
        --fp4 \
        --nvfp4Scales=calib_scales.json \
        --useCustomAllReduce=enable \
        --builderOptimizationLevel=5 \
        --profilingVerbosity=detailed
```

#### 2. Hardware MIG Partitioning for Mixed Criticality
```bash
# Query available GPU instances on DRIVE Thor
nvidia-smi mig -lgi

# Partition GPU into ASIL-D ADAS instance (48 SMs) and Infotainment instance (16 SMs)
nvidia-smi mig -cgi 19,9 -C
```

#### 3. C++ Zero-Copy UMA Allocation with Hardware Coherency
```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// Allocate unified memory accessible by Neoverse V3AE CPU and Blackwell GPU
void* allocate_thor_uma_buffer(size_t tensor_bytes) {
    void* unified_ptr = nullptr;
    
    // Allocate zero-copy memory backed by LPDDR5X with hardware coherency
    cudaError_t status = cudaMallocManaged(&unified_ptr, tensor_bytes, cudaMemAttachGlobal);
    if (status != cudaSuccess) {
        throw std::runtime_error("Failed to allocate Thor UMA memory");
    }
    
    // Advise runtime to optimize for GPU read/write while maintaining CPU prefetch
    cudaMemAdvise(unified_ptr, tensor_bytes, cudaMemAdviseSetPreferredLocation, 0);
    cudaMemAdvise(unified_ptr, tensor_bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
    
    return unified_ptr;
}
```

---

## 6. Functional Safety, Real-Time Latency & Thermal Profiles

### ISO 26262 ASIL-D Safety Island Architecture

DRIVE Thor embeds a fully isolated, fail-operational **ASIL-D Safety Enclave** driven by quad ARM Cortex-R52 cores running in dual lockstep mode.

```mermaid
flowchart TD
    subgraph ASIL_D_Enclave ["Thor ASIL-D Safety Enclave"]
        R52_PRIMARY["Cortex-R52 Core Pair 'Primary'"]
        R52_SHADOW["Cortex-R52 Core Pair 'Shadow'"]
        LOGIC_CMP["Cycle-by-Cycle Hardware Comparator"]
        
        R52_PRIMARY --> LOGIC_CMP
        R52_SHADOW --> LOGIC_CMP
        
        LOGIC_CMP -->|Fault Detected| WATCHDOG_CTRL["Hardware Watchdog & Safety Supervisor"]
    end

    subgraph Primary_Compute_Complex ["Blackwell GPU & Neoverse V3AE 'ASIL-B / QM'"]
        PERCEPTION_EXEC["Perception Neural Networks & Trajectory Proposals"]
    end

    PERCEPTION_EXEC -->|Send Trajectory Proposals| ASIL_D_Enclave
    WATCHDOG_CTRL -->|Pass Validation| CAN_FLEX["Automotive CAN-XL / FlexRay Bus 'Actuation'"]
    WATCHDOG_CTRL -->|Fail Validation| FAIL_SAFE["Engage Redundant Safe Stop Maneuver"]
```

### Latency and Thermal Dissipation Specifications

| Platform Implementation | Thermal Solution | System TDP | Perception Stack Latency | VLM (7B) Decode Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **DRIVE Thor ECU (Single-SoC)**| Liquid Cooled (Water/Glycol) | **175 W** | 2.1 ms (Sparse4D) | 120 tokens / sec |
| **DRIVE Thor ECU (Dual-SoC)** | Liquid Cooled Dual Cold Plate | **320 W** | 1.1 ms (Full L4 Stack) | 240 tokens / sec |

---

## 7. Comparative Platform Matrix

| Architectural Dimension | NVIDIA DRIVE Thor | AMD Versal AI Edge Gen 2 (VE2808) | Qualcomm Snapdragon Ride Flex (SA8775P) |
| :--- | :--- | :--- | :--- |
| **Manufacturing Process** | TSMC 4N Customized | TSMC 4nm / 5nm | 4nm FinFET |
| **AI Inference Architecture**| Blackwell GPU + 5th Gen TC | AIE-ML v2 Tiles (Systolic) | Qualcomm Hexagon NPU + Adreno |
| **Peak Low-Bit AI Compute** | **1,500 TFLOPS (NVFP4)** | 760 TFLOPS (MX-FP4) | 300 TOPS (INT8 / INT4) |
| **Application Processor** | 24x ARM Neoverse V3AE | 8x ARM Cortex-A78AE | 16x ARM Cortex-A78AE / X-Cores |
| **Safety Real-Time Cores** | Quad Cortex-R52 Lockstep | Quad Cortex-R52 Lockstep | Dual Cortex-R52 Lockstep |
| **Memory Bandwidth** | **546.1 GB/s (LPDDR5X)** | 136.5 GB/s (LPDDR5X) | 102.4 GB/s (LPDDR5) |
| **Hardware Multi-Tenancy** | Hardware MIG + Spatial Slicing | PL Partitioning + NoC Virtual Ch | Hypervisor Software Slicing |
| **Software Framework** | TensorRT 11, DRIVE OS, DRIVE AV | Vitis AI 5.x, AMD Quark | Snapdragon Neural Processing SDK |
| **Functional Safety Rating**| ISO 26262 ASIL-D Complete | ISO 26262 ASIL-D Complete | ISO 26262 ASIL-D Capable |
| **Typical Board Power** | 120W – 200W (Liquid Cooled) | 45W – 75W (Air Cooled) | 50W – 90W |
| **Primary Advantage** | Highest raw AI compute + VLM serving | Adaptable logic for custom sensor I/O| Lower power envelope |
| **Primary Limitation** | Requires active automotive liquid cooling| Complex multi-engine toolchain | Lower peak compute for large VLMs |

---

## 8. Cross-References & Related Frameworks

- [[hardware/nvidia-jetson-thor|NVIDIA Jetson Thor Robotics Superchip]]
- [[hardware/nvidia-blackwell-b200|NVIDIA Blackwell B200 & Data Center GPU]]
- [[hardware/arm-neoverse-v3ae|ARM Neoverse V3AE Automotive CPU]]
- [[hardware/arm-cortex-r52|ARM Cortex-R52 Real-Time Lockstep Safety Core]]
- [[hardware/amd-versal-ai-edge-gen2|AMD Versal AI Edge Gen 2 Architecture]]
- [[topics/safety-verification-and-robustness/00-safety-verification-and-robustness-moc|Safety Verification & Robustness]]

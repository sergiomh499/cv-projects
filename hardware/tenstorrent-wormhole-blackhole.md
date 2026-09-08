---
title: "Tenstorrent Wormhole & Blackhole: Tensix Core Spatial Dataflow & Direct Mesh Architecture"
type: "Hardware Architecture"
domain: "RISC-V, Spatial Dataflow & Scalable AI Infrastructure"
status: evergreen
updated: 2026-08-15
tags:
  - hardware
  - tenstorrent
  - wormhole
  - blackhole
  - tensix-core
  - spatial-dataflow
  - risc-v
  - tt-metalium
  - tt-buda
  - scale-out
aliases:
  - Tenstorrent Wormhole
  - Tenstorrent Blackhole
  - Tenstorrent Architecture
  - Tensix Core
  - TT-Metalium
  - Spatial Dataflow Processor
---

# 🌀 Tenstorrent Wormhole & Blackhole: Tensix Core Spatial Dataflow & Direct Mesh Architecture

## 1. Executive Summary & Hardware Typology

**Tenstorrent Wormhole** and **Tenstorrent Blackhole** represent a fundamental departure from monolithic GPU computing paradigms by implementing **RISC-V-controlled Spatial Dataflow Architectures**. Pioneered by Tenstorrent, this architecture eliminates expensive proprietary interconnect switches, bloated monolithic hardware schedulers, and closed instruction set architectures (ISAs) in favor of programmable 2D Network-on-Chip (NoC) grids of **Tensix compute cores** coupled to high-bandwidth memory and direct chip-to-chip optical ethernet links.

1. **Tenstorrent Wormhole**: Fabricated on a 12nm FinFET node, Wormhole integrates a $12 \times 10$ grid of Tensix cores (72 to 80 active), 6-channel GDDR6 memory (288 GB/s), and **16 integrated 100GbE optical Ethernet NoC ports**, allowing cards to connect directly into 2D toroidal meshes without external PCIe or InfiniBand switches.
2. **Tenstorrent Blackhole**: Fabricated on a 6nm FinFET node, Blackhole integrates **140 Tensix cores**, **16 general-purpose 64-bit SiFive RISC-V CPU cores** (enabling standalone, host-free Linux boot), 8-channel GDDR7 memory delivering **$> 1.0\text{ TB/s}$ bandwidth**, and **14 integrated 400GbE Ethernet ports**, yielding over **780 TFLOPS of FP8/BF16 compute** within a **$275\text{ W} - 350\text{ W}$** envelope.

```mermaid
flowchart TD
    subgraph Tenstorrent_Blackhole_Die ["Tenstorrent Blackhole Monolithic AI Processor"]
        subgraph Host_CPU_Complex ["General-Purpose Control Domain"]
            SiFive_Cluster["16x 64-bit SiFive RISC-V Application CPU Cores 'Host-Free Boot'"]
            PCIeGen5["PCIe Gen 5 x16 Interface / CXL Bridge"]
        end

        subgraph Spatial_Tensix_Grid ["140-Core Tensix Compute Grid '2D Mesh NoC'"]
            T00["Tensix (0,0)"] <--> T01["Tensix (0,1)"] <--> T02["Tensix (0,2)"]
            T10["Tensix (1,0)"] <--> T11["Tensix (1,1)"] <--> T12["Tensix (1,2)"]
            T20["Tensix (2,0)"] <--> T21["Tensix (2,1)"] <--> T22["Tensix (2,2)"]
        end

        subgraph Direct_Mesh_Networking ["Integrated Scale-Out Fabric"]
            Eth_NoC["14x 400GbE Optical Ethernet Macro Ports 'Direct Chip-to-Chip'"]
        end

        subgraph HighSpeed_Memory ["High-Bandwidth Memory Interface"]
            GDDR7["8-Channel GDDR7 Memory Controllers '> 1.0 TB/s Bandwidth'"]
        end
    end

    SiFive_Cluster <--> Spatial_Tensix_Grid
    PCIeGen5 <--> Spatial_Tensix_Grid
    Spatial_Tensix_Grid <--> Direct_Mesh_Networking
    Spatial_Tensix_Grid <--> HighSpeed_Memory
```

### Architectural Lineage & Typology Matrix

| Parameter / Silicon | Tenstorrent Wormhole | Tenstorrent Blackhole |
| :--- | :--- | :--- |
| **Manufacturing Process** | 12nm FinFET (TSMC) | **6nm FinFET (TSMC)** |
| **Compute Core Count** | 72–80 Tensix Cores | **140 Tensix Cores** |
| **Integrated Host CPU** | External Host Required (x86/ARM via PCIe)| **16x 64-bit SiFive RISC-V Cores (Standalone OS)**|
| **Peak FP8 / BF16 Compute**| 350 TFLOPS | **780 TFLOPS** |
| **Peak INT8 Compute** | 350 TOPS | **780 TOPS** |
| **On-Chip Scratchpad SRAM**| ~120 MB (1.5 MB per Tensix core) | **210 MB (1.5 MB per Tensix core)** |
| **External Memory & Bandwidth**| 6-Ch GDDR6 (288 GB/s) | **8-Ch GDDR7 (> 1,024 GB/s)** |
| **Direct Scale-Out Fabric** | 16x 100GbE Direct Mesh | **14x 400GbE Direct Mesh (5.6 Tbps Aggregate)** |
| **Thermal Power Envelope** | 225 W – 300 W TDP | **275 W – 350 W TDP** |
| **Primary Software Framework**| TT-Metalium / TT-Buda / TT-Forge | TT-Metalium / TT-Forge / PyTorch 2.x |

---

## 2. Compute Core & Memory Hierarchy

The Tenstorrent memory hierarchy eliminates traditional hardware-managed data cache hierarchies in favor of a distributed, software-managed **Circular Buffer (CB)** model executed across on-chip SRAM scratchpads.

```mermaid
flowchart LR
    subgraph External_GDDR7 ["GDDR7 Memory Subsystem"]
        GDDR_Pool["32 GB GDDR7 (> 1.0 TB/s Port Bandwidth)"]
    end

    subgraph Tensix_Core_SRAM ["Tensix Core Local Memory (1.5 MB per Core)"]
        CB_In["Input Circular Buffer (L1 SRAM Scratchpad)"]
        CB_Inter["Intermediate Feature Buffer"]
        CB_Out["Output Circular Buffer"]
    end

    subgraph Tensix_Compute_Pipeline ["Tensix Processing Engine"]
        UnpackUnit["Unpacker (Decompress & Format Conversion)"]
        MatrixEngine["Matrix Multiplication Engine (FPU / Math)"]
        VectorEngine["Vector SIMD Engine (Non-linear LUT / Act)"]
        PackerUnit["Packer (Compress & Format Conversion)"]
    end

    subgraph NoC_Routers ["2D Torus Network-on-Chip"]
        NoC0["NoC 0: High-Bandwidth Inbound Stream"]
        NoC1["NoC 1: High-Bandwidth Outbound Stream"]
    end

    External_GDDR7 <--> NoC0
    NoC0 --> CB_In
    CB_In --> UnpackUnit
    UnpackUnit --> MatrixEngine
    MatrixEngine --> VectorEngine
    VectorEngine --> PackerUnit
    PackerUnit --> CB_Out
    CB_Out --> NoC1
    NoC1 <--> External_GDDR7
```

### Tensix Core Anatomy
Each Tensix core contains:
1. **5 RISC-V Micro-Controllers (Baby RISC-V)**: Tiny, specialized RISC-V cores programmed to manage the Unpacker, Math Engine, Packer, and NoC 0 / NoC 1 data streaming channels independently.
2. **1.5 MB Local L1 SRAM**: Configured as multi-page circular buffers where data tiles ($32 \times 32$ matrices) are streamed and consumed directly without CPU thread interrupts.
3. **Decoupled Access-Execute (DAE)**: Memory movement over the 2D NoC runs fully decoupled from mathematical matrix execution, completely hiding memory latency.

---

## 3. Micro-Architectural Mechanics: Tensix Execution & Dataflow Routing

```mermaid
flowchart TD
    subgraph Dataflow_Stage_1 ["Stage 1: Tile Stream Unpack"]
        InTile["Input Tile from NoC 0 (32x32 Data Format)"] --> Unpack["Unpack Unit: Converts Block Float / FP8 to Native Format"]
    end

    subgraph Dataflow_Stage_2 ["Stage 2: Systolic Matrix & Vector Compute"]
        Unpack --> MathEngine["Matrix FPU: Dense Matrix Multiply-Accumulate"]
        MathEngine --> VecEngine["Vector SIMD Unit: Fused Activation / Softmax / RMSNorm"]
    end

    subgraph Dataflow_Stage_3 ["Stage 3: Pack & Direct Mesh Route"]
        VecEngine --> PackUnit["Pack Unit: Compresses Result into Target Numerical Precision"]
        PackUnit --> OutTile["Writeback to Output Circular Buffer"]
        OutTile --> MeshRoute["Stream over 2D NoC to Adjacent Tensix Core or 400GbE Port"]
    end
```

### Direct Mesh Scale-Out Topology
Unlike traditional GPU clusters requiring thousands of dollars per port in external InfiniBand switches, Tenstorrent processors connect directly to neighboring chips using integrated QSFP-DD optical ethernet interfaces. The 2D NoC extends seamlessly across circuit boards and rack chassis, creating a single massive spatial computing grid.

---

## 4. Numerical Precision & Block Floating Point Formats

Tenstorrent silicon provides native hardware acceleration for standard floating-point types alongside specialized **Block Floating Point (BFP)** representations.

```mermaid
graph TD
    subgraph Formats ["Supported Numerical Representations"]
        BF16["Bfloat16 (1 Sign, 8 Exp, 7 Mantissa)"]
        FP8["FP8 E4M3 & E5M2"]
        BFP8["BFP8 (Shared 8-bit Exponent per 16/32 Elements)"]
        BFP4["BFP4 (Shared Exponent + 4-bit Mantissas)"]
        BFP2["BFP2 (Ultra-Compressed 2-bit Mantissas)"]
        INT8["INT8 Two's Complement"]
    end
```

### Block Floating Point (BFP) Mathematical Formulation
In BFP formats, a $32 \times 32$ tile of numbers shares a common maximum exponent $E_{\text{shared}}$, allowing individual elements to store only low-bit mantissas:
$$X_{i, j} = (-1)^{s_{i, j}} \cdot 2^{E_{\text{shared}}} \cdot (0.m_{i, j})$$
This preserves dynamic floating-point range while slashing SRAM footprint and memory transfer energy by up to **$75\%$**.

---

## 5. Software Stack, TT-Metalium & TT-Buda Toolchains

Tenstorrent provides a two-tiered software ecosystem:
1. **TT-Metalium**: A low-level, bare-metal C++ framework that exposes direct access to Tensix core circular buffers, RISC-V control micro-engines, and NoC routers.
2. **TT-Buda / TT-Forge**: Graph-level compilers translating PyTorch, ONNX, and JAX models directly into spatial dataflow pipelines.

```mermaid
flowchart TD
    subgraph High_Level_Stack ["High-Level Framework Integration"]
        PyTorch["PyTorch 2.x Models (Transformers / CNNs / LLMs)"]
        ONNX["ONNX Computational Graph"]
    end

    subgraph Compiler_Layer ["Tenstorrent Compilation Toolchain"]
        TT_Forge["TT-Forge / TT-Buda Graph Partitioning Engine"]
        Spatial_Placer["Spatial Tensor Placement & NoC Route Optimizer"]
    end

    subgraph Low_Level_Runtime ["Bare-Metal Runtime"]
        TT_Metalium["TT-Metalium C++ Kernel Programming Interface"]
        Kernel_Binaries["RISC-V Firmware & Tensix Micro-Op Binaries"]
    end

    subgraph Silicon_Fabric ["Tenstorrent Silicon"]
        Wormhole_HW["Tenstorrent Wormhole / Blackhole Processor Mesh"]
    end

    PyTorch --> TT_Forge
    ONNX --> TT_Forge
    TT_Forge --> Spatial_Placer
    Spatial_Placer --> TT_Metalium
    TT_Metalium --> Kernel_Binaries
    Kernel_Binaries --> Wormhole_HW
```

### TT-Metalium C++ Low-Level Kernel Example

```cpp
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/bfloat16.hpp"

using namespace tt::tt_metal;

// Launch a Custom Spatial Dataflow Kernel on Tenstorrent Tensix Core
void launch_tensix_dataflow_kernel(Device* device) {
    CoreCoord core = {0, 0}; // Target Tensix Core (0,0)
    Program program = CreateProgram();

    // 1. Configure Circular Buffers in Tensix L1 SRAM
    uint32_t cb_index = CB::c_in0;
    uint32_t num_input_tiles = 2;
    uint32_t tile_size_bytes = 2048; // 32x32 tile in BF16
    
    CircularBufferConfig cb_config = CircularBufferConfig(num_input_tiles * tile_size_bytes, {{cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(cb_index, tile_size_bytes);
    CreateCircularBuffer(program, core, cb_config);

    // 2. Load and Compile Data Movement & Compute Kernels onto Baby RISC-V Cores
    KernelHandle reader_kernel = CreateKernel(
        program, "kernels/dataflow/reader_unary.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0}
    );

    KernelHandle compute_kernel = CreateKernel(
        program, "kernels/compute/eltwise_binary.cpp", core,
        ComputeConfig{.math_approx_mode = false, .fp32_dest_acc_en = false}
    );

    // 3. Execute Program across the NoC
    EnqueueProgram(device->command_queue(), program, false);
    Finish(device->command_queue());
}
```

---

## 6. Comparative AI Acceleration Matrix

| Architectural Dimension | Tenstorrent Blackhole | Tenstorrent Wormhole | NVIDIA B200 (Blackwell) | Groq LP-1 (TSP) |
| :--- | :--- | :--- | :--- | :--- |
| **Architecture Paradigm** | **Spatial Dataflow (RISC-V)** | Spatial Dataflow (RISC-V) | SIMT GPU (Dual-Die) | Deterministic Tensor Stream |
| **Fabrication Node** | **6nm FinFET** | 12nm FinFET | TSMC 4NP | 14nm FinFET |
| **Core Composition** | **140 Tensix + 16 SiFive Cores**| 72–80 Tensix Cores | 160–192 SMs | 1 Large TSP Engine |
| **Peak Low-Bit Compute** | **780 TFLOPS (FP8/BF16)** | 350 TFLOPS (FP8/BF16) | 20.0 PFLOPS (NVFP4) | 750 TOPS (INT8) |
| **Memory Bandwidth** | **> 1,024 GB/s (GDDR7)** | 288 GB/s (GDDR6) | 8.0 TB/s (HBM3e) | 80 TB/s (SRAM-only) |
| **On-Chip SRAM Cache** | **210 MB L1 Scratchpad** | 120 MB L1 Scratchpad | 128 MB L2 Cache | 230 MB Static SRAM |
| **Direct Scale-Out Mesh** | **14x 400GbE Optical Ethernet**| 16x 100GbE Optical Ethernet| NVLink 5 (Requires Switches)| 16x Real-Time Chip-to-Chip |
| **Thermal Power (TDP)** | **275 W – 350 W** | 225 W – 300 W | 1,000 W (SXM) | 300 W |
| **Open-Source Software** | **TT-Metalium (Fully Open)** | TT-Metalium (Fully Open) | Proprietary CUDA/TRT | Closed Proprietary Compiler |

---

## 7. Cross-References & Related Frameworks

- [[hardware/esperanto-et-soc-1|Esperanto ET-SoC-1 Many-Core RISC-V Processor]]
- [[hardware/nvidia-blackwell-b200|NVIDIA Blackwell B200 Data Center GPU]]
- [[hardware/intel-xeon-amx|Intel Xeon 6th Gen AMX Datacenter Processor]]
- [[docs/edge-ai-quantization-playbook|Edge AI Quantization & Compression Playbook]]

---
title: "NVIDIA Blackwell B200 & GB200: Dual-Die Architecture, NVLink 5 & Hyperscale NVFP4 AI"
type: "Hardware Architecture"
domain: "Data Center, High-Performance AI Infrastructure & Frontier Model Training"
status: evergreen
updated: 2026-08-15
tags:
  - hardware
  - blackwell
  - b200
  - gb200
  - nvlink-5
  - nvfp4
  - transformer-engine
  - hbm3e
  - data-center
  - nvidia
aliases:
  - NVIDIA Blackwell B200
  - NVIDIA B200
  - NVIDIA GB200
  - GB200 NVL72
  - Blackwell Data Center
  - NVFP4 Microscaling
---

# 🏢 NVIDIA Blackwell B200 & GB200: Dual-Die Architecture, NVLink 5 & Hyperscale NVFP4 AI

## 1. Executive Summary & Hardware Typology

The **NVIDIA Blackwell Architecture**—embodied in the **B200**, **B100**, and **GB200 NVL72** rack-scale systems—represents a foundational breakthrough in high-performance datacenter computing, foundation model pre-training, and hyperscale Vision-Language-Action (VLA) inference. Fabricated on a customized TSMC 4NP process node, Blackwell breaks through single-die photolithographic reticle limits by implementing a **dual-die NV-HBI (High-Bandwidth Interface)** packaging architecture containing 208 billion transistors that behaves as a single, fully coherent monolithic GPU.

Blackwell introduces **5th Generation Tensor Cores** with native **NVFP4 (4-bit Microscaling)** precision, a **2nd Generation Transformer Engine** with hardware FlashAttention-3 acceleration, an ultra-high-bandwidth **8.0 TB/s HBM3e memory subsystem**, a dedicated 800 GB/s hardware **Decompression Engine**, **NVLink 5** delivering 1.8 TB/s bidirectional bandwidth per GPU, and coherent coupling to the **NVIDIA Grace CPU** via 900 GB/s NVLink-C2C.

```mermaid
flowchart TD
    subgraph GB200_Superchip ["NVIDIA GB200 NVL72 Compute Node Architecture"]
        subgraph Grace_CPU_Node ["Host CPU Subsystem: NVIDIA Grace"]
            GRACE_CORES["72x ARM Neoverse V2 64-bit Cores @ 3.4 GHz<br/>Scalable Vector Extension 'SVE2 4x 128-bit'"]
            LPDDR5X_HOST["Up to 480 GB LPDDR5X Memory (512 GB/s Bandwidth)"]
        end

        subgraph NVLink_C2C_Bus ["NVLink Chip-to-Chip 'NVLink-C2C'"]
            C2C_BUS["900 GB/s Bidirectional Cache-Coherent Interconnect"]
        end

        subgraph Blackwell_Dual_Die_GPU ["Dual-Die Coherent Blackwell GPU 'B200'"]
            subgraph Die_0 ["Blackwell Die 0 (104B Transistors)"]
                SM_ARRAY_0["80 to 96 Streaming Multiprocessors 'SMs'"]
                HBM3E_0["96 GB HBM3e Memory (4.0 TB/s Bandwidth)"]
                L2_CACHE_0["64 MB High-Throughput L2 Cache"]
            end

            subgraph NV_HBI_Link ["NV-HBI Cross-Die Link"]
                HBI["10 TB/s Bidirectional Low-Latency Die Interconnect"]
            end

            subgraph Die_1 ["Blackwell Die 1 (104B Transistors)"]
                SM_ARRAY_1["80 to 96 Streaming Multiprocessors 'SMs'"]
                HBM3E_1["96 GB HBM3e Memory (4.0 TB/s Bandwidth)"]
                L2_CACHE_1["64 MB High-Throughput L2 Cache"]
            end
        end

        subgraph NVLink5_Network ["NVLink 5 Scale-Up Interconnect"]
            NVL5_PORTS["1.8 TB/s Bidirectional Bandwidth (18 Links @ 100 GB/s)"]
            NV_SWITCH["NVLink Switch System (SHARP v4 In-Network Reductions)"]
        end
    end

    Grace_CPU_Node <--> NVLink_C2C_Bus
    NVLink_C2C_Bus <--> Blackwell_Dual_Die_GPU
    Die_0 <--> NV_HBI_Link
    NV_HBI_Link <--> Die_1
    Blackwell_Dual_Die_GPU <--> NVLink5_Network
```

### Hardware Typology Matrix: Blackwell Datacenter Lineup

| Platform / SKU | Die Configuration | Transistors | Memory Capacity & Type | Memory Bandwidth | Peak FP4 Tensor Compute | Peak FP8 Tensor Compute | Thermal TDP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B200 Tensor Core** | Dual-Die Coherent | 208 Billion | 192 GB HBM3e (8-Hi) | **8.0 TB/s** | **20.0 PFLOPS** | 10.0 PFLOPS | 1,000 W (SXM) |
| **B100 Tensor Core** | Dual-Die Coherent | 208 Billion | 192 GB HBM3e | **8.0 TB/s** | **14.0 PFLOPS** | 7.0 PFLOPS | 700 W (Air-Cooled)|
| **GB200 (Single Superchip)**| 1x Grace + 2x B200 | 488 Billion | 384 GB HBM3e + 480 GB LPDDR5X | 16.0 TB/s (HBM) + 512 GB/s | **40.0 PFLOPS** | 20.0 PFLOPS | 2,700 W |
| **GB200 NVL72 (Rack-Scale)**| 36 Grace + 72 B200 | 17.5 Trillion| 13.8 TB HBM3e + 17.2 TB LPDDR5X| **576 TB/s Aggregate** | **1.44 EFLOPS** | 720 PFLOPS | 120 kW (Liquid) |

---

## 2. Compute Core & Memory Hierarchy

Blackwell resolves multi-GPU communication and memory bandwidth bottlenecks by treating an entire 72-GPU liquid-cooled rack (the **GB200 NVL72**) as a single unified computing domain sharing 13.8 TB of fast HBM3e memory over a **130 TB/s bisection bandwidth NVLink 5 fabric**.

```mermaid
flowchart LR
    subgraph High_Speed_HBM3e ["HBM3e Memory Stack Subsystem"]
        HBM_BANKS["192 GB HBM3e Stack (8x 24GB Stacks)<br/>8.0 TB/s Aggregate Memory Bandwidth"]
    end

    subgraph GPU_Internal_Caches ["Monolithic Dual-Die Cache System"]
        L2_CACHE["128 MB Shared L2 Cache (Unified over 10 TB/s NV-HBI)"]
        SM_L1_SHARED["Up to 48 MB Aggregate SM Shared Memory / L1"]
    end

    subgraph SM_Register_Files ["SM Register Hierarchy"]
        REG_FILE["65,536 x 32-bit Registers per SM<br/>Total SM Array Register File > 50 MB"]
    end

    subgraph Host_Memory_Pool ["Grace CPU Host Memory 'NVLink-C2C'"]
        GRACE_RAM["480 GB LPDDR5X @ 512 GB/s<br/>Zero-Copy Heterogeneous Unified Address Space"]
    end

    High_Speed_HBM3e <--> GPU_Internal_Caches
    GPU_Internal_Caches <--> SM_Register_Files
    GPU_Internal_Caches <-->|900 GB/s NVLink-C2C| Host_Memory_Pool
```

### Die Layout and Interconnect Specifications
1. **NV-HBI (NVIDIA High-Bandwidth Interface)**:
   - Connects the two reticle-sized silicon dies across a passive silicon bridge with **10 TB/s bidirectional bandwidth**.
   - Maintains full cache coherency across both dies: CUDA threads executing on Die 0 access L2 cache partitions and HBM stacks on Die 1 with sub-10ns cross-die latency.
2. **HBM3e Memory Subsystem**:
   - 8 stacks of 8-high / 12-high HBM3e operating across an 8192-bit interface delivering **$8.0\text{ TB/s}$** per B200 GPU.
   - Eliminates memory bandwidth stalls during multi-head attention (MHA) decoding in trillion-parameter mixture-of-experts (MoE) foundation models.

---

## 3. Micro-Architectural Mechanics: 5th Gen Tensor Cores & NVFP4

Blackwell's 5th Generation Tensor Cores introduce native hardware acceleration for **NVFP4 (4-bit Microscaling)** floating-point matrix multiplication alongside 2nd Generation Transformer Engine optimizations.

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
Standard scalar quantization applies a single global scale factor $\alpha$ across an entire matrix row ($W \approx \alpha \cdot Q$). NVFP4 microscaling applies independent floating-point scale factors $s_b$ across small, 16-element sub-vectors (blocks):
$$X_{i, j} = s_b \cdot q_{i, j}, \quad \text{where } b = \lfloor j / 16 \rfloor, \quad q_{i, j} \in \text{FP4 (E2M1)}, \quad s_b \in \text{FP8 (E8M0)}$$
- **E2M1 Format**: Provides 1 sign bit, 2 exponent bits, and 1 mantissa bit, representing values $\pm \{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0\}$.
- **Throughput Advantage**: NVFP4 Tensor Core execution delivers **$2\times$ higher throughput** than FP8 Tensor Core execution and **$4\times$ higher throughput** than FP16/BF16.

---

## 4. Numerical Precision & Arithmetic Throughput

The Blackwell B200 delivers dense and sparse computational throughput across floating-point and integer precisions.

```mermaid
graph TD
    subgraph Precision_Types ["Blackwell Precision Formats"]
        NVFP4_P["NVFP4 (E2M1 Block 16) - 20.0 PFLOPS"]
        FP8_P["FP8 (E4M3 / E5M2) - 10.0 PFLOPS"]
        INT8_P["INT8 / INT4 - 10.0 PFLOPS / 20.0 POPS"]
        BF16_P["BF16 / FP16 Tensor - 5.0 PFLOPS"]
        TF32_P["TF32 Tensor - 2.5 PFLOPS"]
        FP32_P["FP32 CUDA Standard - 90.0 TFLOPS"]
        FP64_P["FP64 Tensor / Vector - 45.0 TFLOPS"]
    end
```

### Precision Capabilities Matrix (NVIDIA B200 SXM Single GPU)

| Numerical Precision | Mantissa / Exponent | Block Size | Hardware Acceleration | Peak Dense Compute | Peak Sparse (2:4) Compute |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVFP4 (E2M1)** | 1 Sign, 2 Exp, 1 Mant | 16 elements (E8M0 Scale) | 5th Gen Tensor Core | **10.0 PFLOPS** | **20.0 PFLOPS** |
| **FP8 (E4M3 / E5M2)**| 1 Sign, 4/5 Exp, 3/2 Mant | Scalar | 5th Gen Tensor Core | **5.0 PFLOPS** | **10.0 PFLOPS** |
| **INT8 / INT4** | Two's Complement Integer | Scalar | 5th Gen Tensor Core | **5.0 / 10.0 POPS** | **10.0 / 20.0 POPS** |
| **BF16 / FP16** | IEEE Standard Formats | Scalar | 5th Gen Tensor Core | **2.5 PFLOPS** | **5.0 PFLOPS** |
| **TF32** | 1 Sign, 8 Exp, 10 Mant | Scalar | 5th Gen Tensor Core | **1.25 PFLOPS** | **2.5 PFLOPS** |
| **FP32** | IEEE Standard 754 | Scalar | Blackwell CUDA Core | **90.0 TFLOPS** | N/A |
| **FP64** | IEEE Standard 754 | Scalar | Blackwell FP64 Unit | **45.0 TFLOPS** | N/A |

---

## 5. Software Stack, SDKs & Toolchains

The software ecosystem for Blackwell datacenter accelerators integrates **TensorRT-LLM**, **PyTorch 2.x native FP4**, **NVIDIA Megatron-Core**, **CUDA 13.x**, and **NVLink SHARP v4** in-network reductions.

```mermaid
flowchart TD
    subgraph Framework_Layer ["Frontier AI Frameworks"]
        Megatron["Megatron-Core (Pipeline + Tensor Parallelism)"]
        TRT_LLM["TensorRT-LLM (Continuous Batching + NVFP4 Engine)"]
        vLLM_TRT["vLLM Blackwell Backend"]
    end

    subgraph CUDA_Runtime ["CUDA 13.x & Communication Layer"]
        FlashAttn3["FlashAttention-3 Asynchronous CUDA Kernel"]
        NCCL_Blackwell["NCCL 2.22+ (NVLink 5 Rail-Optimized)"]
        TMA_Driver["Tensor Memory Accelerator (TMA v2) Driver"]
    end

    subgraph Silicon_Hardware ["GB200 NVL72 Rack Hardware"]
        NVL72_Fabric["72x B200 GPUs + 36x Grace CPUs via NVLink 5 Switch"]
    end

    Megatron --> TRT_LLM
    TRT_LLM --> FlashAttn3
    TRT_LLM --> NCCL_Blackwell
    FlashAttn3 --> TMA_Driver
    NCCL_Blackwell --> NVL72_Fabric
    TMA_Driver --> NVL72_Fabric
```

### Production Toolchain Recipes

#### 1. Compiling a Trillion-Parameter MoE Model for B200 in NVFP4
```bash
# Export and Quantize MoE Model to NVFP4 using TensorRT-LLM
python3 tensorrt_llm/quantization/quantize.py \
    --model_dir /models/deepseek-v3/ \
    --dtype bfloat16 \
    --qformat nvfp4 \
    --calib_dataset /data/c4_calibration.json \
    --output_dir /models/deepseek-v3-nvfp4/

# Build NVLink 5 Optimized TensorRT Engine across 8x B200 GPUs
trtllm-build \
    --checkpoint_dir /models/deepseek-v3-nvfp4/ \
    --output_dir /engines/deepseek-v3-b200/ \
    --gemm_plugin nvfp4 \
    --tp_size 8 \
    --pp_size 1 \
    --max_batch_size 256 \
    --max_num_tokens 8192
```

---

## 6. Scale-Out Network & Thermal Topologies

- **NVLink 5 Scale-Up Domain**: 1.8 TB/s bidirectional bandwidth per GPU across 18 high-speed links. The NVLink Network Switch System aggregates up to 72 GPUs into a single shared-memory address space with SHARP v4 in-network hardware reduction.
- **Thermal Management**: GB200 NVL72 operates with full direct-to-chip liquid cooling (water/glycol), dissipating 120 kW per rack while operating at coolant inlet temperatures up to 45°C.

---

## 7. Comparative Datacenter Accelerator Matrix

| Architectural Dimension | NVIDIA B200 | NVIDIA H100 (Hopper) | AMD Instinct MI325X | Google TPU v5p |
| :--- | :--- | :--- | :--- | :--- |
| **Manufacturing Node** | TSMC 4NP (Dual-Die) | TSMC 4N (Monolithic) | TSMC 5nm/6nm (Chiplets) | 3nm Custom |
| **Transistor Count** | **208 Billion** | 80 Billion | 153 Billion | Undisclosed |
| **Memory Capacity & Type** | **192 GB HBM3e** | 80 GB HBM3 | **256 GB HBM3e** | 95 GB HBM3 |
| **Memory Bandwidth** | **8.0 TB/s** | 3.35 TB/s | 6.0 TB/s | 4.8 TB/s |
| **Peak Low-Bit Compute** | **20.0 PFLOPS (NVFP4)**| 4.0 PFLOPS (FP8) | 5.2 PFLOPS (FP8) | ~918 TFLOPS (BF16/INT8) |
| **Scale-Up Interconnect** | **NVLink 5 (1.8 TB/s)**| NVLink 4 (900 GB/s) | Infinity Fabric (896 GB/s) | ICI 4.8 Tbps Optical |
| **Transformer Engine** | **2nd Gen (FlashAttn-3)**| 1st Gen (FP8) | Software ROCm Engine | XLA Compiler Graph |
| **Thermal TDP** | 1,000 W (SXM) | 700 W (SXM) | 1,000 W (OAM) | Liquid-Cooled Pod |

---

## 8. Cross-References & Related Frameworks

- [[hardware/nvidia-drive-thor|NVIDIA DRIVE Thor Automotive Superchip]]
- [[hardware/nvidia-jetson-thor|NVIDIA Jetson Thor Robotics Module]]
- [[hardware/intel-xeon-amx|Intel Xeon 6th Gen AMX Datacenter Processor]]
- [[hardware/tenstorrent-wormhole-blackhole|Tenstorrent Wormhole & Blackhole AI Processors]]
- [[docs/edge-ai-quantization-playbook|Edge AI Quantization & Compression Playbook]]

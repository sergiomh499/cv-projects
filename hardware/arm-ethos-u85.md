---
title: "Arm Ethos-U85: Micro-NPU for Edge Vision, Transformers & 4 TOPS Embedded AI Acceleration"
type: "Hardware Architecture"
domain: "Micro-NPUs, TinyML & Edge Vision Transformers"
status: evergreen
updated: 2026-08-15
tags:
  - hardware
  - arm
  - ethos-u85
  - micro-npu
  - edge-vision
  - transformers
  - tinyml
  - vela-compiler
  - int4
aliases:
  - Arm Ethos-U85
  - Ethos-U85
  - U85 MicroNPU
  - Arm Edge Transformer NPU
  - TinyML Transformer Accelerator
---

# 👁️ Arm Ethos-U85: Micro-NPU for Edge Vision, Transformers & 4 TOPS Embedded AI Acceleration

## 1. Executive Summary & Hardware Typology

The **Arm Ethos-U85** is Arm's third-generation, highest-performance **Micro-NPU (uNPU)**, engineered to bring native **Vision Transformer (ViT)** acceleration, 2D Matrix Multiplication (GEMM), and sub-watt deep learning to microcontrollers (Cortex-M55, Cortex-M85) and embedded application processors (Cortex-A320, Cortex-A520).

Operating within a thermal envelope of **$20\text{ mW} - 450\text{ mW}$**, the Ethos-U85 scales from 128 up to **2048 Multiply-Accumulate (MAC) units per clock cycle**. Running at 1.0 GHz, the 2048-MAC configuration delivers **4.0 INT8 TOPS** and **8.0 INT4 TOPS**, representing a $4\times$ throughput increase and $20\%$ higher energy efficiency over the preceding Ethos-U65 generation.

```mermaid
flowchart TD
    subgraph Host_Microcontroller ["Host Microcontroller / CPU Subsystem"]
        CPU["Cortex-M85 / Cortex-M55 / Cortex-A Core"]
        TCM["DTCM / ITCM (Zero-Wait State Local Memory)"]
        NVIC["Nested Vectored Interrupt Controller 'NVIC'"]
    end

    subgraph Ethos_U85_Silicon ["Arm Ethos-U85 Micro-NPU 'Up to 2048 MACs/clk'"]
        CCU["Command Stream Controller & Autonomous DMA Engine"]
        WeightEngine["Hardware Weight Decompression & Bit-Unpacking Unit"]
        
        subgraph MAC_Matrix_Grid ["Systolic MAC Engine Grid"]
            MAC0["Tile 0: 512 INT8/INT4 MACs"]
            MAC1["Tile 1: 512 INT8/INT4 MACs"]
            MAC2["Tile 2: 512 INT8/INT4 MACs"]
            MAC3["Tile 3: 512 INT8/INT4 MACs"]
        end

        VectorUnit["Non-Linear Vector ALU 'Softmax, SiLU, LayerNorm, SwiGLU'"]
        LineBuffers["Internal Activation Line Buffers & Accumulators"]
    end

    subgraph Memory_Topology ["Embedded Storage & Buses"]
        AXI_M["Dual 64/128-bit AXI5 Master Bus"]
        SRAM["On-Chip System SRAM (512KB - 4MB)"]
        Flash["Octal SPI / HyperFlash (NVM Models & Compressed Weights)"]
    end

    CPU -->|Issue Command Stream Address & IRQ Setup| CCU
    CCU -->|Fetch Pre-Compiled Vela Command Stream| Flash
    CCU -->|DMA Compressed Weight Stream| WeightEngine
    WeightEngine --> MAC0
    CCU -->|DMA Activation Stream| LineBuffers
    LineBuffers <--> MAC0
    MAC0 --> VectorUnit
    VectorUnit --> LineBuffers
    LineBuffers -->|Writeback Output Tensors| SRAM
    CCU -->|Assert Inference Complete IRQ| NVIC
    CPU <--> TCM
    CPU <--> AXI_M
    AXI_M <--> SRAM
    AXI_M <--> Flash
```

### Ethos-U85 Hardware Configurations Matrix

| Metric / Parameter | Ethos-U85-128 | Ethos-U85-512 | Ethos-U85-1024 | Ethos-U85-2048 |
| :--- | :--- | :--- | :--- | :--- |
| **Compute Units (MACs/cycle)**| 128 MACs/clk | 512 MACs/clk | 1024 MACs/clk | **2048 MACs/clk** |
| **Peak INT8 Throughput (at 1 GHz)**| 0.256 TOPS | 1.024 TOPS | 2.048 TOPS | **4.096 TOPS** |
| **Peak INT4 Throughput (at 1 GHz)**| 0.512 TOPS | 2.048 TOPS | 4.096 TOPS | **8.192 TOPS** |
| **Supported Precisions** | INT8, INT16 | INT8, INT16, INT4 | INT8, INT16, INT4 | **INT4, INT8, INT16** |
| **Transformer Operators** | Softmax, LayerNorm | Softmax, RMSNorm, GEMM| Native ViT / SwiGLU | **Full ViT & TinyVLM Pipeline** |
| **System Bus Interface** | 64-bit AXI5 | 64-bit AXI5 | 128-bit AXI5 | **Dual 128-bit AXI5 Interfaces** |
| **Typical Power Dissipation** | 20 mW – 50 mW | 80 mW – 150 mW | 150 mW – 280 mW | **200 mW – 450 mW** |
| **Energy Efficiency** | $\sim 10 \text{ TOPS/W}$ | $\sim 11.5 \text{ TOPS/W}$ | $\sim 12 \text{ TOPS/W}$ | **Up to 12.5 INT8 TOPS/W (25 INT4 TOPS/W)**|

---

## 2. Compute Core & Memory Hierarchy

The Ethos-U85 is architected to eliminate off-chip memory traffic, which represents the single largest power consumer in battery-powered edge vision devices.

```mermaid
flowchart LR
    subgraph Storage_Memory ["Flash / NVM Memory Tier"]
        VelaWeights["Compressed INT8/INT4 Model Weights"]
        CmdStream["Pre-Compiled Vela Command Stream"]
    end

    subgraph Internal_SRAM_Domain ["SRAM Memory Tier"]
        SystemSRAM["On-Chip SRAM 'Working Activation Tiling Arena'"]
        LineBuf["Ethos-U85 Internal Line Buffers 'Ultra-Wide Low-Power SRAM'"]
    end

    subgraph Processing_Pipelines ["Ethos-U85 Silicon Units"]
        DecUnit["Hardware Weight Decompression Engine"]
        MAC_Array["Systolic MAC Compute Grid"]
        Act_Pipe["Element-Wise Vector & LUT Unit"]
    end

    CmdStream -->|Direct DMA Fetch| DecUnit
    VelaWeights -->|Stream On-the-Fly| DecUnit
    DecUnit -->|Uncompressed Weights - Direct Injection| MAC_Array
    SystemSRAM -->|Tiled Activations DMA| LineBuf
    LineBuf <--> MAC_Array
    MAC_Array --> Act_Pipe
    Act_Pipe --> LineBuf
```

### Memory Bandwidth Optimization Mechanics
1. **On-the-Fly Hardware Weight Decompression**:
   - Neural network weights are compressed offline by the **Arm Vela Compiler** using run-length and Huffman-based entropy coding.
   - During inference, the Ethos-U85 hardware decompression engine streams packed weights directly from external Flash storage into the MAC registers, decompressing in silicon at bus speed without intermediate SRAM buffering. This reduces off-chip memory traffic by up to **$65\%$**.
2. **Double-Buffering & Spatial Activation Tiling**:
   - The compiler tiles feature maps into spatial blocks ($H_t \times W_t \times C_t$) sized to fit within small internal SRAM partitions. While the MAC array processes Tile $N$, the autonomous DMA engine pre-fetches activations for Tile $N+1$.
3. **Weight-Only INT4 Quantization**:
   - Native support for 4-bit integer weights enables executing quantized Vision Transformers (ViT) and tiny Vision-Language Models (TinyVLM) within microcontrollers containing $\le 2\text{ MB}$ Flash storage.

---

## 3. Micro-Architectural Mechanics: Native Transformer Acceleration

While earlier micro-NPUs focused exclusively on 2D convolutions, the Ethos-U85 incorporates specialized hardware execution units for Transformer attention layers.

```mermaid
flowchart TD
    subgraph Transformer_Layer_Input ["Vision Transformer Token Embeddings"]
        Q_Tokens["Query Matrix Q (INT8)"]
        K_Tokens["Key Matrix K (INT8)"]
        V_Tokens["Value Matrix V (INT8)"]
    end

    subgraph Ethos_U85_Hardware_Pipeline ["Ethos-U85 Hardware Execution Units"]
        GEMM_QK["2D Systolic Array: GEMM Q * K^T"]
        SOFTMAX_ALU["Hardware Vector Softmax & Scaling Engine"]
        GEMM_AV["2D Systolic Array: Attention * V"]
        RMS_NORM["Hardware RMSNorm / LayerNorm Unit"]
        SWIGLU_ALU["Fused SwiGLU / GeLU Activation Unit"]
    end

    subgraph Output_Tokens ["Layer Output Tokens"]
        Output_Feats["Output Feature Representation (Written to SRAM)"]
    end

    Q_Tokens & K_Tokens --> GEMM_QK
    GEMM_QK --> SOFTMAX_ALU
    SOFTMAX_ALU & V_Tokens --> GEMM_AV
    GEMM_AV --> RMS_NORM
    RMS_NORM --> SWIGLU_ALU
    SWIGLU_ALU --> Output_Feats
```

- **Hardware Softmax & Normalization**: The vector engine computes non-linear mathematical operations (exponential functions, reciprocal square roots for LayerNorm/RMSNorm) via high-precision Look-Up Tables (LUTs) with linear interpolation, eliminating CPU round-trips.

---

## 4. Numerical Precision & Quantization

The mathematical execution pipeline of Ethos-U85 handles asymmetric and symmetric quantized integer formats with 32-bit accumulation.

```mermaid
graph TD
    subgraph Input_Representation ["Quantized Input Tensors"]
        INT8_Act["INT8 Activation: q_a (Asymmetric with zero-point z_a)"]
        INT4_W["INT4 Weight: q_w (Symmetric zero-point = 0)"]
        INT16_Sens["INT16 High-Precision Sensor / Depth Feature"]
    end

    subgraph MAC_Math ["Hardware MAC Core"]
        Formula["Product = (q_a - z_a) * q_w"]
        Acc["32-bit Signed Integer Accumulator (Prevent Overflow)"]
    end

    subgraph Requantization ["Requantize & Activation Stage"]
        Scale["Fixed-Point Multiplication (Multiplier M0 * 2^-n)"]
        Clamp["Saturation Clamp [-128, 127] or [0, 255]"]
    end

    INT8_Act --> Formula
    INT4_W --> Formula
    INT16_Sens --> Formula
    Formula --> Acc
    Acc --> Scale
    Scale --> Clamp
```

### Quantization Mathematical Formulation
For standard quantized convolutional and fully-connected layers, the floating-point operation $Y = X \cdot W + B$ is realized in Ethos-U85 hardware as:
$$q_y = z_y + \text{QuantizeMultiplier}\left( \sum_{k} (q_{x, k} - z_x) \cdot q_{w, k} + b_k \right)$$
- **Accumulator Bit-Width**: 32-bit signed integers ($[-2^{31}, 2^{31}-1]$).
- **Scale Computation**: High-precision fixed-point multiplication using 32-bit scale multipliers with bit shifts, eliminating floating-point division hardware.

---

## 5. Software Stack, Vela Compiler & Toolchains

The software compilation pipeline translates PyTorch, TensorFlow Lite, and ExecuTorch models into optimized Ethos-U85 command streams.

```mermaid
flowchart TD
    subgraph Frameworks ["Model Training & Quantization"]
        Torch["PyTorch (ExecuTorch)"]
        TFLite_Quant["Post-Training Quantization (INT8 / INT4 TFLite Model)"]
    end

    subgraph Offline_Optimization ["Arm Vela Neural Network Compiler"]
        Vela["Arm Vela Optimizer (vela model.tflite)"]
        Graph_Partition["Graph Partitioning (Ethos-U85 Ops vs CPU Fallbacks)"]
        Scheduling["SRAM Allocation, Stride Folding & DMA Command Generation"]
        VelaModel["Optimized .tflite containing Custom Ethos-U85 Command Stream"]
    end

    subgraph Target_Firmware ["Microcontroller Firmware Stack"]
        CMSIS["CMSIS-NN / CMSIS-Core"]
        EthosDriver["Ethos-U Core Driver (ethosu_driver.c)"]
        TFLM["TensorFlow Lite for Microcontrollers (TFLM) Runtime"]
        ExecuTorch_U["ExecuTorch Micro-Runtime Delegate"]
    end

    Torch --> TFLite_Quant
    TFLite_Quant --> Vela
    Vela --> Graph_Partition
    Graph_Partition --> Scheduling
    Scheduling --> VelaModel
    VelaModel --> TFLM
    VelaModel --> ExecuTorch_U
    CMSIS --> EthosDriver
    TFLM --> EthosDriver
    ExecuTorch_U --> EthosDriver
```

### Practical Toolchain Usage: Compiling for Ethos-U85

```bash
# Install Arm Vela Compiler
pip install ethos-u-vela

# Compile INT8/INT4 TFLite Model for Ethos-U85 2048-MAC Configuration
vela models/vit_tiny_patch16_int8.tflite \
    --accelerator-config ethos-u85-2048 \
    --system-config Ethos_U85_High_Performance \
    --memory-mode Shared_Sram \
    --arena-cache-size 1048576 \
    --output-dir build/vela_output/

# Inspect Vela Memory Optimization & Operator Distribution Report
cat build/vela_output/vit_tiny_patch16_int8_vela_summary.csv
```

---

## 6. Functional Safety, Real-Time Latency & Thermal Profiles

### Determinism and Power Metrics
- **Cycle-Accurate Worst-Case Execution Time (WCET)**: Because Ethos-U85 operates exclusively with statically scheduled scratchpad SRAM line buffers (no non-deterministic hardware cache eviction policies), inference runtimes are cycle-deterministic to within $\pm 0.1\%$.
- **Active Power Dissipation**: $180\text{ mW} - 450\text{ mW}$ at 1.0 GHz (2048 MACs).
- **Sleep & Standby Leakage**: Automatic autonomous clock-gating drops consumption to $< 50\ \mu\text{W}$ immediately when the command queue is drained.

---

## 7. Comparative Architecture Matrix

| Metric / Parameter | Arm Ethos-U85 | Arm Ethos-U65 | STMicroelectronics STM32N6 (Neural-ART) | NXP eIQ Neutron NPU (MCX-N94x) |
| :--- | :--- | :--- | :--- | :--- |
| **Compute Core Type** | Micro-NPU Co-Processor | Micro-NPU Co-Processor | Integrated Proprietary Neural-ART NPU | Embedded Neutron NPU Architecture |
| **Peak Throughput** | **4.0 TOPS INT8 / 8.0 TOPS INT4** | 1.0 TOPS INT8 | ~0.6 TOPS INT8 | ~0.5 TOPS INT8 |
| **Transformer Native Acceleration**| **Yes (Softmax, RMSNorm, SwiGLU)**| Limited (Element-wise vector) | No (Layer-by-Layer CPU Fallbacks) | Limited (Basic GEMM) |
| **Numeric Formats** | **INT4, INT8, INT16** | INT8, INT16 | INT8, INT16 | INT8, INT16, FP16 |
| **Tightly Coupled Host Core** | Cortex-M55 / Cortex-M85 / A320 | Cortex-M55 / Cortex-A53 | Dual Cortex-M55 | Dual Cortex-M33 |
| **Compiler & Toolchain** | **Arm Vela + CMSIS-NN / ExecuTorch**| Arm Vela + CMSIS-NN | STM32Cube.AI | NXP eIQ Toolkit + TFLM |
| **Typical Power Envelope** | **$20\text{ mW} - 450\text{ mW}$** | $50\text{ mW} - 250\text{ mW}$ | $100\text{ mW} - 350\text{ mW}$ | $50\text{ mW} - 200\text{ mW}$ |
| **Target Application** | Micro Edge Vision, TinyVLM, Smart Audio| Smart Home, Voice Recognition | Industrial Smart Cameras, Robotics | Wearable Sensors, Predictive Maintenance |

---

## 8. Cross-References & Related Frameworks

- [[hardware/arm-ethos-u65|Arm Ethos-U65 Microcontroller NPU]]
- [[hardware/arm-cortex-a78ae|Arm Cortex-A78AE Application Processor]]
- [[hardware/intel-npu|Intel NPU 4 & 5 Architecture]]
- [[docs/edge-ai-quantization-playbook|TinyML Quantization & Compression Playbook]]

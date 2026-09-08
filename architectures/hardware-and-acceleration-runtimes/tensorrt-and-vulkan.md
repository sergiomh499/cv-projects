---
title: "TensorRT 10 & Vulkan 1.3: High-Throughput GPU Inference Runtimes"
type: model-deep-dive
tasks:
  - gpu-deployment
  - low-latency-inference
  - edge-workstation-serving
  - cross-platform-compute
architecture_class: Deep Learning Compiler & Cross-Platform Compute Runtimes
primary_license: Apache-2.0 / NVIDIA Proprietary EULA / Apache-2.0 (Vulkan)
commercial_use: true
official_repo: https://github.com/NVIDIA/TensorRT
paper_url: https://developer.nvidia.com/tensorrt
tags:
  - architecture
  - gpu
  - tensorrt
  - vulkan
  - flashattention
  - zero-copy
  - fp8
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - TensorRT
  - Vulkan Kompute
  - TensorRT 10
  - Vulkan 1.3 Compute
---

# 🔬 TensorRT 10 & Vulkan 1.3: Production GPU Inference Engines

## 1. Executive Brief & Significance

Production computer vision and multimodal foundation models require execution runtimes that saturate hardware arithmetic units (NVIDIA Tensor Cores, AMD Matrix Cores, Apple Neural Engine, and Qualcomm Adreno GPUs) while eliminating host-device serialization overhead.

Two primary runtimes define modern production deployments:
1. **NVIDIA TensorRT 10 / 10.x**: The industry-standard deep learning optimization compiler and runtime engine for NVIDIA architectures (Blackwell, Hopper, Ada Lovelace, Ampere, and Jetson Orin). TensorRT performs polyhedral loop optimization, horizontal and vertical layer fusions, FP8 (E4M3/E5M2) mixed-precision calibration, and native CUDA Graph execution to achieve sub-millisecond latencies.
2. **Vulkan 1.3 Compute / Kompute / NCNN**: The open, vendor-agnostic cross-platform compute standard. By compiling neural graph operations into SPIR-V intermediate bytecode, Vulkan enables hardware-accelerated deep learning across AMD GPUs, Intel Arc, Apple Silicon (via MoltenVK), Raspberry Pi 5 VideoCore VII, and Android Mali/Adreno GPUs without vendor driver lock-in.

```mermaid
flowchart TD
    Model["PyTorch / JAX / ONNX Model"] --> ModelExport["Standardized ONNX / Strong-Typed Graph"]
    ModelExport --> TargetRouting{"Target Hardware Platform?"}

    subgraph NVIDIA_Ecosystem ["NVIDIA Hardware Ecosystem (TensorRT 10.x)"]
        TargetRouting -->|NVIDIA GPU / Jetson Orin| TRTBuilder["TensorRT 10 IBuilder & INetworkDefinition"]
        TRTBuilder --> GraphOpt["Graph Rewriting: Vertical + Horizontal Layer Fusion"]
        GraphOpt --> FP8Quant["FP8 (E4M3 / E5M2) & INT8 PTQ Calibration"]
        FP8Quant --> PluginV3["Custom IPluginV3 Engine Integration"]
        PluginV3 --> PlanFile["Serialized Engine Plan (.engine / .plan)"]
        PlanFile --> CUDAGraphs["IExecutionContext + CUDA Graph Instantiation"]
        CUDAGraphs --> NVExecution["Direct Tensor Core FP8 / FlashAttention-3 Kernel Launch"]
    end

    subgraph Cross_Platform_Vulkan ["Cross-Platform Open Ecosystem (Vulkan 1.3)"]
        TargetRouting -->|AMD / Intel / ARM / Mobile / Pi| SPIRVCompiler["GLSL / HLSL -> SPIR-V Bytecode Compilation"]
        SPIRVCompiler --> VkPipeline["Vulkan 1.3 Compute Pipeline (VkPipelineLayout)"]
        VkPipeline --> DescriptorSets["Descriptor Sets & Push Constants Setup"]
        DescriptorSets --> DMABUF["Zero-Copy Linux DMA-BUF Unified Memory Buffers"]
        DMABUF --> VkDispatch["vkCmdDispatch (Subgroup Wavefront Acceleration)"]
        VkDispatch --> OpenExecution["Cross-Vendor GPU Hardware Execution"]
    end
```

---

## 2. TensorRT 10 Deep Architectural Mechanics

### A. Graph Compilation & Optimizer Workflow
The TensorRT 10 compilation pipeline operates over three distinct interface layers:
1. **`IBuilder` & `INetworkDefinition`**: Ingests ONNX nodes using the Strongly Typed network flag (`NetworkDefinitionCreationFlag::kSTRONGLY_TYPED`), preserving explicit precision bindings across layers.
2. **`IBuilderConfig`**: Defines optimization heuristics, memory budgets (`MemoryPoolType::kWORKSPACE`), hardware timing profiles, and tactic sources (`TacticSource::kCUBLAS`, `kCUDNN`, `kEDGE_MASK_CONVOLUTIONS`).
3. **`IRuntime` & `IExecutionContext`**: Deserializes the engine binary into device memory and handles asynchronous kernel dispatch (`enqueueV3`) bound to specific `cudaStream_t` queues.

---

### B. TensorRT 10 Custom Plugin Architecture: `IPluginV3`
TensorRT 10 introduced the modernized `IPluginV3` interface, replacing legacy `IPluginV2DynamicExt`. The `IPluginV3` specification decouples compilation-time tensor shape inference, memory resource management, and execution into distinct, modular interfaces:

- **`IPluginV3OneCore`**: Defines core plugin identity, capability queries, and format support (`PluginCapabilityId::kCORE`).
- **`IPluginV3OneBuild`**: Configures shape negotiation (`getOutputDataTypes`, `getOutputShapes`) and workspace resource sizing (`getWorkspaceSize`) during the engine building phase without requiring CUDA runtime initialization.
- **`IPluginV3OneRuntime`**: Manages low-overhead execution (`attachToContext`, `enqueue`) and thread-safe scratchpad buffer allocation.

```
+--------------------------------------------------------------------+
|                         IPluginV3 Interface                        |
+---------------------------------+----------------------------------+
| IPluginV3OneBuild               | IPluginV3OneRuntime              |
| - getOutputShapes()             | - attachToContext()              |
| - getOutputDataTypes()          | - enqueue(PluginTensorDesc, ...) |
| - getWorkspaceSize()            | - clone()                        |
+---------------------------------+----------------------------------+
| IPluginV3OneCore                                                   |
| - getPluginName() / getPluginVersion() / getPluginNamespace()     |
| - supportsFormatCombination(pos, inOut, inOutFmts)                |
+--------------------------------------------------------------------+
```

---

### C. FP8 Floating-Point Format (E4M3 vs. E5M2) & FlashAttention-3
TensorRT 10 fully exploits Hopper and Ada Lovelace 8-bit floating-point (FP8) Tensor Cores:
- **E4M3 (1 sign bit, 4 exponent bits, 3 mantissa bits)**: Saturated dynamic range $[-448, 448]$. Ideal for activation tensors and weight matrices where fine quantization resolution is essential to prevent degradation in vision transformers (ViT) and object detectors.
- **E5M2 (1 sign bit, 5 exponent bits, 2 mantissa bits)**: Matches IEEE FP16 dynamic range with reduced precision. Used for gradients and attention softmax probability maps where wide dynamic range prevents numerical underflow.

For transformer attention blocks, TensorRT fuses multi-head queries, keys, and values into hardware **FlashAttention-2 / FlashAttention-3** kernels, keeping intermediate $S = Q K^T$ matrices entirely within on-chip **Shared Memory (SRAM)** ($228\text{ KB per SM}$ on Ada Lovelace / Hopper), eliminating global HBM memory bandwidth round-trips.

---

## 3. Vulkan 1.3 Cross-Platform Compute Architecture

### A. SPIR-V Bytecode & Compute Pipeline
Vulkan compute decouples model execution from vendor-specific proprietary toolchains by using **Standard Portable Intermediate Representation (SPIR-V)**:
1. Neural network operations (convolutions, matrix multiplications, layer norms) are authored in GLSL/HLSL compute shaders.
2. The shader code is compiled ahead-of-time (AOT) into binary SPIR-V bytecode modules (`vkCreateShaderModule`).
3. At application launch, the Vulkan driver compiles SPIR-V directly into native GPU machine instructions tailored to the exact installed hardware (AMD RDNA, Intel Xe, Apple AGX, ARM Immortalis).

---

### B. Descriptor Sets, Push Constants & Subgroup Wavefronts
- **Push Constants**: Up to $128-256\text{ bytes}$ of uniform parameters (tensor dimensions, quantization scales, activation biases) written directly into the GPU command buffer without allocating device descriptor tables.
- **Descriptor Sets (`VkDescriptorSet`)**: Bind storage buffers (`VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`) to shader binding registers.
- **Subgroup Operations (`VK_KHR_shader_subgroup_arithmetic`)**: Executes SIMD parallel reductions across GPU execution warps (NVIDIA: 32 threads, AMD: 64 threads) using hardware shuffle instructions (`subgroupAdd`, `subgroupInclusiveMul`), bypassing shared memory synchronization barriers.

---

### C. Zero-Copy Linux DMA-BUF & Unified Memory Pipeline
In embedded computer vision (e.g., robotics, drone perception), camera frames captured via V4L2 or CSI MIPI interfaces should not be copied across user-space memory buffers.

Vulkan 1.3 and TensorRT support zero-copy sharing via Linux **DMA-BUF (`dma_buf`)** file descriptors:
1. The camera driver allocates a contiguous physical memory buffer via V4L2 memory-mapped buffers (`V4L2_MEMORY_MMAP` or `DMABUF`).
2. The file descriptor is imported into Vulkan using `VkImportMemoryFdInfoKHR` or into CUDA using `cudaImportExternalMemory`.
3. The neural network processes pixels directly from the physical DMA frame buffer.
4. Output detections or rendered segmentation masks are passed directly to hardware video encoders (NVENC / VA-API) via the same DMA-BUF handle, achieving a **0-copy, sub-10-microsecond** ingestion pipeline.

---

## Granular Component-by-Component Architectural Breakdown

### Architectural Taxonomy & Structural Elements

| Architectural Stage | Component Identity | Structural Specification & Type | Attention / Conv Mechanics | Receptive Field & Memory Space |
| :--- | :--- | :--- | :--- | :--- |
| **Architectural Paradigm** | **Graph Compiler & Compute Runtime** | AOT Graph Rewriting, Kernel Fusion & Low-Level Hardware Dispatch | Direct hardware micro-kernel synthesis (Tensor Cores / Subgroups) | Host RAM $\leftrightarrow$ GPU Unified Memory / VRAM |
| **Frontend Parser** | **ONNX / Strongly-Typed Importer** | Native ONNX graph parser with topological node validation | Static shape verification and dynamic dimension profiling | Multi-dimensional tensor definitions |
| **Optimization Neck** | **Vertical / Horizontal Layer Fusion** | CBR Fusion (`Conv + Bias + ReLU`), Multi-Head $QKV$ Projection Fusion | Polyhedral loop merging into unified CUDA / SPIR-V kernels | Kernel Shared Memory (SRAM) boundary |
| **Quantization Encoder** | **FP8 / INT8 PTQ & QAT Engine** | E4M3 / E5M2 dual-format scaling + Symmetric INT8 KL-Divergence | Per-channel weight quantization & per-tensor activation scaling | 8-bit dynamic integer / float representation |
| **Runtime Execution Head** | **CUDA Graph / Vulkan Command Dispatch** | Pre-instantiated executable execution context (`IExecutionContext`) | Direct asynchronous launch on hardware stream queues | Zero-copy mapped DMA-BUF buffers |

---

### Computational Latency & Resource Distribution

| Stage / Subsystem | Parameter Share (%) | Inference Latency (%) | Computational Complexity ($\text{FLOPs}$) | Dominant Hardware Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **Graph Kernel Dispatch** | <1% (Overhead) | ~2% (with CUDA Graphs) | $\mathcal{O}(1)$ dispatch overhead | PCIe Host-to-Device command queue latency |
| **Matrix Multiply (GEMM / Conv)** | ~80% | ~75% | $\mathcal{O}(2 M N K)$ | Tensor Core MAC arithmetic intensity |
| **FlashAttention Kernel** | ~15% | ~16% | $\mathcal{O}(N_{\text{ctx}}^2 D)$ (SRAM Fused) | GPU Shared Memory bandwidth bound |
| **Normalization & Activations** | ~5% | ~7% | $\mathcal{O}(N)$ (Element-wise) | Global Memory (HBM/VRAM) bandwidth bound |

---

## 3. Quantitative SOTA Benchmark Profile

### Inference Latency Across GPU Runtimes (Batch Size = 1)

| Model Architecture | Input Shape | PyTorch FP32 (RTX 4090) | TensorRT FP16 (RTX 4090) | TensorRT FP8 (RTX 4090) | TensorRT FP16 (Jetson Orin 64GB) | Vulkan NCNN (AMD RX 7900 XTX) | Vulkan NCNN (Apple M3 Max) | Vulkan NCNN (RPi 5 VideoCore) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | $3 \times 224 \times 224$ | 2.80 ms | 0.42 ms | **0.22 ms** | 1.85 ms | 0.65 ms | 0.82 ms | 32.0 ms |
| **YOLOv8x** | $3 \times 640 \times 640$ | 18.50 ms | 4.80 ms | **2.95 ms** | 18.20 ms | 6.40 ms | 7.80 ms | 145.0 ms |
| **DINOv2-Base** | $3 \times 224 \times 224$ | 24.00 ms | 6.50 ms | **3.60 ms** | 22.40 ms | 8.90 ms | 10.20 ms | 210.0 ms |
| **SAM 2 (Hiera-S)** | $3 \times 1024 \times 1024$ | 45.00 ms | 12.20 ms | **7.40 ms** | 42.00 ms | 16.80 ms | 19.50 ms | 480.0 ms |
| **RF-DETR (Large)** | $3 \times 640 \times 640$ | 32.00 ms | 7.90 ms | **4.60 ms** | 29.50 ms | 11.20 ms | 13.40 ms | 280.0 ms |

---

## 4. Engineering Implementation: TensorRT 10 Execution with CUDA Graphs & Custom IPluginV3

### A. Python TensorRT 10 Async Execution Pipeline with CUDA Graphs
```python
"""
TensorRT 10 High-Throughput Inference Engine with CUDA Graph Capture
"""

import tensorrt as trt
import torch


class TRT10InferenceEngine:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None

    def allocate_buffers(self, input_shape: tuple, output_shape: tuple, dtype=torch.float16):
        """Allocates static PyTorch device tensors for CUDA Graph capture."""
        self.static_input = torch.zeros(input_shape, dtype=dtype, device="cuda")
        self.static_output = torch.zeros(output_shape, dtype=dtype, device="cuda")
        
        # Set tensor addresses for TensorRT 10 execution context
        self.context.set_tensor_address("input", self.static_input.data_ptr())
        self.context.set_tensor_address("output", self.static_output.data_ptr())

    def capture_cuda_graph(self):
        """Warm up and capture execution graph to eliminate launch latency."""
        # 1. Warm-up iterations
        for _ in range(3):
            self.context.execute_async_v3(self.stream.cuda_stream)
        
        torch.cuda.synchronize()
        
        # 2. Graph Capture
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph, stream=self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Runs sub-millisecond graph-captured inference."""
        self.static_input.copy_(input_tensor)
        self.cuda_graph.replay()
        self.stream.synchronize()
        return self.static_output
```

---

### B. Custom TensorRT 10 `IPluginV3` C++ Header Definition
```cpp
#pragma once
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <vector>

namespace nvinfer1 {

class SwishActivationPluginV3 : public IPluginV3,
                                public IPluginV3OneCore,
                                public IPluginV3OneBuild,
                                public IPluginV3OneRuntime {
public:
    SwishActivationPluginV3() = default;
    virtual ~SwishActivationPluginV3() = default;

    // --- IPluginV3 Methods ---
    IPluginCapability* getCapabilityInterface(PluginCapabilityId capability) noexcept override {
        switch (capability) {
            case PluginCapabilityId::kCORE: return static_cast<IPluginV3OneCore*>(this);
            case PluginCapabilityId::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
            case PluginCapabilityId::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
            default: return nullptr;
        }
    }

    IPluginV3* clone() noexcept override { return new SwishActivationPluginV3(*this); }

    // --- IPluginV3OneCore Methods ---
    AsciiChar const* getPluginName() const noexcept override { return "SwishActivationPluginV3"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

    // --- IPluginV3OneBuild Methods ---
    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs
    ) const noexcept override {
        outputTypes[0] = inputTypes[0]; // Output type matches input type (FP16/FP32/FP8)
        return 0;
    }

    int32_t getOutputShapes(
        DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder
    ) noexcept override {
        outputs[0] = inputs[0]; // Output dimensions strictly match input dimensions
        return 0;
    }

    size_t getWorkspaceSize(
        DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs
    ) const noexcept override {
        return 0; // In-place activation requires 0 scratchpad workspace
    }

    // --- IPluginV3OneRuntime Methods ---
    int32_t enqueue(
        PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream
    ) noexcept override;
};

} // namespace nvinfer1
```

---

## 5. Commercial Usability & License Audit

- **TensorRT**: The open-source parser and plugin components are licensed under **Apache-2.0**. The binary drivers and compilation libraries are governed by the **NVIDIA TensorRT Software License Agreement** (free for royalty-free commercial distribution on NVIDIA desktop, datacenter, and Jetson hardware).
- **Vulkan / Kompute / NCNN**: **Apache-2.0** / **BSD-3-Clause**. Completely open and permissive for commercial integration across all silicon vendors.
- **Official Repositories**:
  - `NVIDIA/TensorRT`: [https://github.com/NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)
  - `Tencent/ncnn`: [https://github.com/Tencent/ncnn](https://github.com/Tencent/ncnn)
  - `Kompute`: [https://github.com/KomputeServer/kompute](https://github.com/KomputeServer/kompute)

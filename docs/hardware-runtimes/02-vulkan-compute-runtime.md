---
title: "Khronos Vulkan Standard Compute: Architecture, Shaders & Edge Testing"
type: hardware-runtime-guide
domain: Cross-Platform Edge Compute
tags:
  - hardware-runtime
  - vulkan
  - spirv
  - compute-shaders
  - ncnn
  - kompute
updated: 2026-09-08
aliases:
  - Vulkan Compute Guide
---

# 🔴 Khronos Vulkan Standard Compute: Architecture, Shaders & Edge Testing

## 1. Physical Hardware Execution Model
- **Cross-Vendor Portability**: Targets any GPU with a compliant Vulkan 1.2/1.3 driver (NVIDIA, AMD, Intel, ARM Mali, Qualcomm Adreno, Apple Silicon via MoltenVK).
- **Execution Units**:
  - `VkInstance` / `VkPhysicalDevice`: Host hardware abstraction.
  - `VkDevice` / `VkQueue`: Logical interface for dispatching work.
  - `VkCommandBuffer`: Pre-recorded sequence of execution commands.
  - `VkPipeline`: State object encapsulating the compute shader bytecode, descriptor set layouts, and push constants.

---

## 2. Compute Shader Execution Flow
1. **Shader Authoring**: Compute kernels are written in GLSL (OpenGL Shading Language) or HLSL:
   ```glsl
   #version 450
   layout(local_size_x = 16, local_size_y = 16) in;
   layout(binding = 0) readonly buffer InputBuffer { float in_data[]; };
   layout(binding = 1) writeonly buffer OutputBuffer { float out_data[]; };
   void main() {
       uint idx = gl_GlobalInvocationID.x;
       out_data[idx] = max(0.0, in_data[idx]); // ReLU
   }
   ```
2. **SPIR-V Compilation**: Offline or build-time compilation into binary SPIR-V bytecode:
   ```bash
   glslc -fshader-stage=compute relu.comp -o relu.spv
   ```
3. **Dispatch**: Submitting `vkCmdDispatch(cmd, group_x, group_y, group_z)` to the compute queue.

---

## 3. Real Physical Testing & Benchmarking Workflow

```bash
# 1. Query all physical GPU capabilities and compute queue families
vulkaninfo --summary

# 2. Check supported extensions (e.g. float16, int8, subgroup operations)
vulkaninfo | grep -E "VK_KHR_shader_float16_int8|VK_KHR_subgroup_extended_types"

# 3. Compile and validate SPIR-V shaders with Khronos validator
glslangValidator -V shader.comp -o shader.spv
spirv-val shader.spv

# 4. Profile Vulkan GPU commands via RenderDoc or VK_KHR_performance_query
renderdoccmd capture ./vulkan_inference_binary
```

---

## 4. Production Edge Deployment
- **Tencent NCNN**: High-performance mobile inference engine converting CNN/Transformer layers directly into optimized Vulkan compute shaders with FP16 packing.
- **ONNX Runtime Vulkan Execution Provider**: Executes ONNX models on non-NVIDIA GPUs (Intel/AMD/ARM) with zero CUDA dependency.

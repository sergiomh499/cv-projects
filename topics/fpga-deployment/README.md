# FPGA Deployment Playbook

# Overview
FPGA (Field-Programmable Gate Array) Deployment targets reconfigurable silicon architectures for high-throughput, deterministic, ultra-low latency, and power-efficient computer vision and deep learning inference. FPGAs excel in edge perception, aerospace, defense, automotive ADAS, and industrial vision where millisecond jitter, thermal constraints, and custom hardware sensor interfaces (MIPI-CSI, PCIe, GigE Vision) dominate.

## SOTA & Research
- **Seminal & Modern Papers**:
  - *FINN* (Umuroglu et al., 2017): Quantized Neural Network accelerator framework generating customized streaming dataflow architectures tailored to specific topologies.
  - *Brevitas* (Blott et al., 2018): PyTorch library for Quantization-Aware Training (QAT) targeting sub-8-bit and binary/ternary representations for FPGA hardware.
  - *Vitis AI / DPU Architecture* (AMD / Xilinx): Coarse-grained reconfigurable Deep Learning Processing Unit (DPU) overlays optimized for convolutional and transformer workloads.
  - *LUTNet* (Wang et al., 2019): Directly mapping quantized weights and activations into native FPGA Look-Up Tables (LUTs) rather than DSP blocks.
- **Evaluation Benchmarks & Metrics**:
  - Latency (microseconds per frame), Throughput (FPS), Power Efficiency (FPS/Watt or GOPS/Watt).
  - Resource Utilization: Look-Up Tables (LUTs), Flip-Flops (FFs), Block RAMs / UltraRAMs (BRAM/URAM), Digital Signal Processors (DSP48/DSP58 slices).

## Architecture Alternatives & Trade-offs
| FPGA Architecture Paradigm | Implementation Style | Latency & Jitter | Flexibility / Model Updates | Ideal Target |
| :--- | :--- | :--- | :--- | :--- |
| **Streaming Dataflow (FINN)** | Dedicated pipeline stage per layer | Lowest latency (<1 ms, deterministic) | Low (requires FPGA bitstream re-synthesis per model change) | Ultra-high-speed sorting, line-rate triggers |
| **Instruction-Set DPU Overlay (Vitis AI)** | General-purpose tensor processor on FPGA | Low-to-Moderate (3-10 ms) | High (swapping models is as easy as loading an `.xmodel` file) | Embedded edge vision, multi-model robotics |
| **Hybrid CPU-FPGA SoC (Zynq, Kria)** | Preprocessing in FPGA fabric, inference on DPU/ARM | Flexible system design | High | Complete autonomous perception edge boxes |

## Popular Repos & Integrations
- **[AMD / Xilinx Vitis AI](https://github.com/Xilinx/Vitis-AI)**: Complete development stack for hardware-accelerated AI inference on Xilinx platforms (Kria SOMs, Zynq UltraScale+, Alveo).
- **[Xilinx FINN](https://github.com/Xilinx/finn)**: Fast, scalable Quantized Neural Network inference compiler on FPGAs.
- **[Brevitas](https://github.com/Xilinx/brevitas)**: PyTorch library for Quantization-Aware Training (QAT) supporting 1-bit, 2-bit, 4-bit, and 8-bit networks.
- **Tooling Integrations**:
  - **FiftyOne**: Compare model accuracy discrepancies between floating-point PyTorch baseline and integer-quantized FPGA simulated outputs.
  - **Rerun**: Visualize low-latency detections and sensor feeds transmitted over UDP/Ethernet directly from the FPGA edge board.

## End-to-End Pipeline & Workarounds
1. **Pipeline Stages**:
   - PyTorch Model Definition -> Quantization-Aware Training (Brevitas QAT) -> Model Export (ONNX) -> Vitis AI Quantizer & Compiler -> Target Compilation (`.xmodel` or Vivado HLS IP core) -> Bitstream deployment on Zynq/Kria MPSoC -> Hardware runtime execution (VART / XRT).
2. **Common Traps & Edge Cases**:
   - *Unsupported Operators*: Complex activations (SiLU/GELU), non-standard pooling, or dynamic reshaping often cannot map to DPU hardware and fall back to slow ARM CPU emulation.
   - *DSP Slice Exhaustion*: Models with high channel counts without aggressive channel pruning exhaust available DSP multipliers.
   - *Memory Bandwidth Saturation*: Repeatedly fetching intermediate activation maps from external DDR RAM creates thermal and latency bottlenecks.
3. **Engineering Workarounds**:
   - **Operator Replacement**: Swap SiLU with hard-sigmoid/ReLU or LeakyReLU during model design, which map directly to FPGA LUTs.
   - **Channel Pruning & Distillation**: Prune redundant filters with tools like PyTorch Pruning before quantization to reduce DSP requirements by 50%.
   - **Layer Fusion**: Keep intermediate feature maps in on-chip UltraRAM (URAM) caches between back-to-back convolution and pooling layers to eliminate DDR round-trips.

## Deployment & Real-time Notes
- **Vitis AI Runtime (VART)**:
  - Use VART C++ API for direct zero-copy buffer sharing between hardware video capture DMA blocks and the DPU input tensor buffer.
- **Quantization Calibration**:
  - Perform Post-Training Quantization (PTQ) calibration with at least 1,000 diverse representative images covering extreme lighting and noise conditions.
- **Thermal & Clock Throttling**:
  - Monitor FPGA junction temperature (`xbutil` on Alveo, or sysfs on Zynq/Kria) to ensure clock frequencies do not throttle under continuous sustained inference.

---
title: "Edge-AI Quantization Playbook: PTQ vs QAT, Mixed-Precision & Outlier Suppression"
type: production-guide
domain: Hardware Acceleration & Quantization
tags:
  - quantization
  - ptq
  - qat
  - tensorrt-10
  - vitis-ai-5
  - smoothquant
  - outlier-suppression
  - mixed-precision
  - amd-quark
updated: 2026-09-08
aliases:
  - Edge-AI Quantization Guide
  - PTQ vs QAT Calibration Manual
---

# ⚙️ Edge-AI Quantization Playbook: PTQ vs QAT, Mixed-Precision & Outlier Suppression

A comprehensive, production-grade guide to low-precision deep learning quantization—covering **Post-Training Quantization (PTQ)** vs. **Quantization-Aware Training (QAT)**, mathematical outlier suppression (**SmoothQuant**, **Outlier Suppression+**), **Explicit Q/DQ Graph Execution in TensorRT 10**, and **AMD Quark / NPU Workflows in Vitis AI 5.x**.

Related notes: [[topics/gpu-deployment/03-compilers-and-open-problems|GPU Compilers & Precision]], [[topics/fpga-deployment/03-silicon-and-open-problems|FPGA Silicon Frontiers]], [[docs/critical-scenarios-hardware-matrix|Critical Scenarios Hardware Matrix]].

---

## 1. Quantization Spectrum: PTQ vs. QAT

```mermaid
flowchart TD
    FloatModel[Full Precision PyTorch Model: FP32 / BF16] --> Path{Quantization Route}
    Path -->|Route A: Fast Calibration PTQ| PTQ[PTQ: Feed 512 Representative Samples -> Compute Scale s & Zero-Point z]
    Path -->|Route B: Fine-Tuned Gradient QAT| QAT[QAT: Insert FakeQuant Nodes -> Retrain with Straight-Through Estimator STE]
    PTQ --> Check{Accuracy Degradation > 1.0%?}
    Check -->|No: Stable CNNs / ConvNeXt| DeployPTQ[Export INT8 Calibration Cache -> Compile Engine]
    Check -->|Yes: Transformer / Softmax Drift| Mitigate{Apply Outlier Suppression}
    Mitigate -->|SmoothQuant Migration| PTQ2[Scale Channels s = diag(w/a) -> Re-calibrate PTQ]
    Mitigate -->|Fails Complex SOTA| QAT
    QAT --> DeployQAT[Export Explicit Q/DQ ONNX -> Compile Engine]
```

### Comparative Strategy Matrix
| Feature | Post-Training Quantization (PTQ) | Quantization-Aware Training (QAT) | Mixed-Precision PTQ |
| :--- | :--- | :--- | :--- |
| **Compute Cost** | Low (5–15 minutes calibration) | High (Hours to days retraining) | Moderate (Hessian sensitivity sweep) |
| **Labeled Data Needed?**| **No** (Unlabeled representative images) | **Yes** (Ground-truth labels required) | No |
| **Sub-Byte Viability** | Fails below INT8 ($<8$-bit collapses) | **Viable at INT4 / FP4 / 2-bit** | Viable (Sensitive layers stay FP16) |
| **Tooling Pipeline** | TensorRT `IInt8EntropyCalibrator2`, Quark | PyTorch `torch.ao.quantization`, Brevitas| AMD Quark, TensorRT Model Optimizer |
| **Best-Fit Tasks** | Standard CNNs, YOLOv10, MobileNetV4 | RF-DETR, Vision Transformers, VLAs | Qwen2-VL, SAM 2.1, 3DGS-SLAM |

---

## 2. Mathematical Formulations: Uniform Quantization & Outlier Migration

### 1. Symmetric Affine Uniform Quantization:
Given a continuous real tensor $x \in [\alpha, \beta]$, a uniform $b$-bit integer quantizer maps $x$ to $q \in [-2^{b-1}, 2^{b-1} - 1]$ (for signed INT8, $[-128, 127]$):
$$q = \text{clamp}\left( \left\lfloor \frac{x}{s} \right\rceil, -2^{b-1}, 2^{b-1} - 1 \right)$$
Where the scale factor $s$ is determined by the maximum absolute dynamic range:
$$s = \frac{\max(|x|)}{2^{b-1} - 1}$$
Dequantization reconstructs the real value: $\hat{x} = q \cdot s$.

### 2. The Outlier Problem in Vision Transformers (SmoothQuant):
In Vision Transformers (ViTs, DETR, VLMs), activation channels exhibit **systematic outliers**: $0.1\%$ of channels possess activation magnitudes $100\times$ larger than other channels, but are concentrated in specific fixed channels across all tokens.
Direct activation quantization truncates or squashes the remaining $99.9\%$ of channels.

#### The SmoothQuant Mathematical Transformation:
Because matrix multiplication satisfies $Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \hat{W}$, SmoothQuant migrates quantization difficulty from activations to weights using a per-channel migration factor $s_j$:
$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1 - \alpha}} \quad \text{with migration hyperparameter } \alpha \in [0, 1]$$
- Setting $\alpha = 0.5$ balances the dynamic ranges equally between activations and weights.
- Multiplies activation channels by $s_j^{-1}$ (suppressing outliers) and pre-scales weights by $s_j$ offline.
- **Inference Overhead: ZERO.** The scaling factor is absorbed into the previous layer's LayerNorm or linear bias.

---

## 3. Production Workflows: TensorRT 10 vs. Vitis AI 5.x / AMD Quark

### Workflow A: TensorRT 10 Explicit Q/DQ Compilation
In TensorRT 10+, the legacy implicit calibration API is deprecated in favor of **Explicit Quantize/Dequantize (Q/DQ) graphs**:

```python
# Modern TensorRT 10 Model Optimizer (modelopt) Flow
import modelopt.torch.quantization as mtq

# Step 1: Wrap PyTorch model with SmoothQuant + INT8 Q/DQ nodes
config = mtq.INT8_DEFAULT_CONFIG
model_quantized = mtq.quantize(model, config, forward_loop=calibration_dataloader)

# Step 2: Export ONNX containing explicit 'QuantizeLinear' & 'DequantizeLinear' operators
mtq.export_onnx(model_quantized, dummy_input, "model_int8_qdq.onnx")
```

```bash
# Step 3: Build engine with zero-copy FP16 layer fallback for sensitive Softmax heads
trtexec --onnx=model_int8_qdq.onnx \
        --saveEngine=model_int8.engine \
        --int8 --fp16 \
        --builderOptimizationLevel=5
```

---

### Workflow B: AMD Vitis AI 5.x & Quark NPU Compilation
For AMD Versal Gen 2 AIE-ML v2 / NPU hardware:

```python
# AMD Quark Quantizer for Vitis AI 5.x
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import Config, QuantizationConfig

# Configure MX9 / INT8 Mixed-Precision targeting Versal NPU IP
quant_config = QuantizationConfig(
    calibrate_method="minmax",
    weight_type="int8",
    activation_type="int8",
    layers_to_exclude=["model.head.classification_layer"] # Keep head in FP16
)
quantizer = ModelQuantizer(quant_config)
quantized_model = quantizer.quantize_model(model, calibration_dataloader)
quantizer.export_onnx(quantized_model, "versal_npu_int8.onnx")
```

---

## 4. The 5 Golden Rules for Quantizing Computer Vision Models
1. **Never Quantize Softmax or LayerNorm to INT8**: Always keep activation normalization and attention Softmax layers in FP16/BF16 to prevent probability collapse.
2. **Exclude Coordinate Regression Layers**: Object detection bounding box heads and 6-DoF translation regressors suffer $>5\%$ mAP drop if quantized below FP16.
3. **Use at Least 512 Diverse Calibration Samples**: Avoid calibrating on consecutive video frames; sample uniformly across weather, lighting, and occlusion variations.
4. **Prefer Micro-Scaling (MX6/MX9) Over INT8 When Targeting Versal Gen 2**: Micro-scaling handles dynamic outliers natively without manual channel smoothing.
5. **Verify Accuracy on Full Validation Split**: Never trust calibration loss alone; always verify mAP / Top-1 accuracy post-quantization against the full validation baseline.

---
title: "Post-Training Quantization (PTQ) & Outlier Smoothing: INT8 / FP8 Acceleration Without Retraining"
type: "Technique"
domain: "Hardware Acceleration & Edge Neural Execution"
tags:
  - technique
  - ptq
  - int8
  - fp8
  - smoothquant
  - awq
  - tensorrt
  - quantization
  - edge-ai
status: evergreen
updated: 2026-09-09
aliases:
  - "Post-Training Quantization"
  - "PTQ"
  - "SmoothQuant"
  - "Outlier Smoothing"
  - "INT8 Quantization"
---

# ⚡ Post-Training Quantization (PTQ) & Outlier Smoothing: INT8 / FP8 Acceleration

## 1. High-Level Concept & The Activation Outlier Wall

On modern edge hardware (NVIDIA Orin, AMD Versal FPGA, Apple Silicon, Qualcomm NPU), neural network inference throughput is governed by the **Memory Bandwidth Wall**:
- Loading floating-point parameters (FP32: 4 bytes/param, FP16: 2 bytes/param) from external DRAM into on-chip cache consumes over $70\%$ of total execution latency and power.
- Quantizing models to **INT8** (1 byte/param) or **FP8** halves the required memory bandwidth, doubles cache utilization, and activates specialized INT8 Tensor Cores / DPUs capable of executing at $2\times\text{ to }4\times$ higher TOPS (Tera-Operations Per Second).

### The Activation Outlier Problem in Vision Transformers & Modern ConvNets
Standard Post-Training Quantization (PTQ) calibrates models using a small unlabelled dataset (e.g., 512 images) without fine-tuning weights.
While model **weights** $\mathbf{W}$ are uniformly distributed and trivially quantized to INT8, **activations** $\mathbf{X}$ in modern Vision Transformers (ViT, [[architectures/vision-foundation-models/dinov2|DINOv2]], [[architectures/vision-foundation-models/sam-2|SAM 2]], [[architectures/multimodal-vlm-and-vla/qwen2-5-vl|Qwen2.5-VL]]) display extreme **systematic activation outliers**:
- Out of 1024 activation channels, a small handful (e.g., channels 12, 87, 412) consistently exhibit magnitudes up to $100\times$ higher than normal channels (e.g., normal features hover around $\pm 1.5$, while outlier channels spike to $\pm 150.0$).
- If quantization scales are determined per-tensor, the outlier channel stretches the dynamic range, causing the remaining 99.7% of critical feature values to be crushed into zero or one integer bucket, inducing complete catastrophic accuracy collapse.

```
Activation Outlier Collapse vs. SmoothQuant Mathematical Migration:

Standard Per-Tensor INT8 Quantization:
Channel 1-1020: [  | | | ] (Crushed into 0 buckets!)
Channel 1021:   [========================================] (Outlier 150.0)
==> Catastrophic model failure!

SmoothQuant Activation Smoothing:
   Divide Activation Outlier by S:   X_smooth = X * diag(S)^-1   (Dynamic range reduced)
   Multiply Weight Channel by S:     W_smooth = diag(S) * W      (Absorbed offline into weights)
   Result: Y = X * W = X_smooth * W_smooth (100% mathematically exact, clean INT8 execution!)
```

---

## 2. Mathematical Formulation

### 2.1 Uniform Symmetric INT8 Quantization
A continuous floating-point tensor $\mathbf{x} \in \mathbb{R}$ is mapped to signed 8-bit integer $q \in [-128, 127]$ using a single scalar scale factor $s > 0$:

$$
q = \text{clamp}\left( \left\lfloor \frac{\mathbf{x}}{s} \right\rceil, -128, 127 \right)
$$

where $\lfloor \cdot \rceil$ denotes round-to-nearest-integer, and the optimal scale factor $s$ is calibrated from the maximum absolute tensor value:

$$
s = \frac{\max(|\mathbf{x}|)}{127}
$$

Dequantization reconstructs the floating-point approximation:

$$
\hat{\mathbf{x}} = q \cdot s
$$

---

### 2.2 SmoothQuant: Mathematical Migration of Quantization Difficulty
Given a linear layer $\mathbf{Y} = \mathbf{X} \mathbf{W}$, SmoothQuant (Xiao et al., 2023) inserts a positive diagonal scaling matrix $\mathbf{S} = \text{diag}(s_1, s_2, \dots, s_C) \in \mathbb{R}^{C \times C}$:

$$
\mathbf{Y} = \mathbf{X} \mathbf{W} = \left( \mathbf{X} \mathbf{S}^{-1} \right) \left( \mathbf{S} \mathbf{W} \right) = \hat{\mathbf{X}} \hat{\mathbf{W}}
$$

Notice that this transformation is **mathematically identical and exact**.

The per-channel scaling factor $s_j$ is formulated to migrate quantization difficulty from activations to weights according to migration hyperparameter $\alpha \in [0, 1]$:

$$
s_j = \frac{\max(|\mathbf{X}_j|)^\alpha}{\max(|\mathbf{W}_j|)^{1 - \alpha}}
$$

where:
- $\max(|\mathbf{X}_j|)$ is the maximum activation across calibration tokens for channel $j$.
- $\max(|\mathbf{W}_j|)$ is the maximum weight across output dimensions for channel $j$.
- $\alpha = 0.5$ balances the quantization difficulty equally between activations and weights.

### The Offline Advantage
Because $\mathbf{S}$ is constant after calibration:
1. $\hat{\mathbf{W}} = \mathbf{S} \mathbf{W}$ is computed **offline once** and quantized to INT8 weights.
2. At inference time, $\mathbf{S}^{-1}$ is folded directly into the preceding LayerNorm or linear bias with **zero additional runtime operations**.
3. Both $\hat{\mathbf{X}}$ and $\hat{\mathbf{W}}$ are cleanly quantized into INT8, enabling 100% native INT8 GEMM execution on hardware Tensor Cores.

---

## 3. Comparison of Quantization Paradigms

| Quantization Technique | Retraining Required | VRAM Reduction | Speedup (NVIDIA Orin / Jetson) | Accuracy Drop on ViTs |
| :--- | :---: | :---: | :---: | :---: |
| **FP16 Baseline** | None | $1.0\times$ | $1.0\times$ | $0.0\%$ |
| **Naive INT8 PTQ** | None | $2.0\times$ | $2.1\times$ | $>15.0\%$ (Catastrophic collapse) |
| **Quantization-Aware Training (QAT)** | Full Training Suite | $2.0\times$ | $2.2\times$ | $<0.3\%$ |
| **SmoothQuant / AWQ (PTQ)** | **None (Minutes)** | **$2.0\times$** | **$2.3\times$** | **$<0.5\%$** |

---

## 4. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn

class SmoothQuantLinear(nn.Module):
    """
    Applies SmoothQuant per-channel outlier smoothing across linear layers.
    Migrates activation outliers into weight matrices for high-fidelity INT8 PTQ.
    """
    def __init__(self, linear: nn.Linear, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        
        self.weight = nn.Parameter(linear.weight.data.clone())
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None

    @torch.no_grad()
    def smooth(self, act_max: torch.Tensor):
        """
        Calibrates and applies diagonal scaling factor S to weights and layer scales.
        act_max: [C_in] maximum absolute activation per channel from calibration set.
        """
        weight_max = self.weight.abs().max(dim=0)[0].clamp(min=1e-5)
        act_max = act_max.clamp(min=1e-5)
        
        # Calculate per-channel smoothing scale: s = act_max^alpha / weight_max^(1 - alpha)
        scales = (act_max ** self.alpha) / (weight_max ** (1.0 - self.alpha))
        scales = scales.clamp(min=1e-4)
        
        # 1. Scale weights offline: W_smooth = S * W (broadcast over rows)
        self.weight.mul_(scales.view(1, -1))
        
        # 2. Return inverse scales S^-1 to be folded into previous LayerNorm/Activation
        return 1.0 / scales

    def quantize_int8(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """Uniform symmetric INT8 clamp and round."""
        q = torch.clamp(torch.round(x / scale), -128, 127)
        return q * scale  # Simulated fake quantization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Evaluate standard linear layer with smoothed weights
        return nn.functional.linear(x, self.weight, self.bias)
```

---

## 5. Models & Runtimes in the Vault Utilizing PTQ & Smoothing

- **[[architectures/hardware-and-acceleration-runtimes/onnxruntime-runtime|ONNX Runtime]]**: High-throughput cross-platform engine executing INT8 PTQ and EP execution.
- **[[architectures/hardware-and-acceleration-runtimes/vulkan-sc-runtime|Vulkan SC Runtime]]**: Safety-critical graphics and compute runtime utilizing INT8 integer pipelines.
- **[[architectures/vision-foundation-models/dinov2|DINOv2]] & [[architectures/vision-foundation-models/sam-2|SAM 2]]**: ViT foundation models requiring activation smoothing for INT8 edge deployment.
- **[[architectures/backbones-and-edge-efficiency/mambavision|MambaVision]]**: Hybrid SSM-transformer utilizing PTQ for real-time edge robotics.

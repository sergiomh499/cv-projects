---
title: "Mixed-Precision Training (AMP): Dynamic Loss Scaling & Numerical Dynamics"
type: "Technique"
domain: "Edge Acceleration, Quantization & Architecture Efficiency"
tags:
  - technique
  - mixed-precision
  - amp
  - fp16
  - bf16
  - loss-scaling
  - tensor-cores
  - numerical-stability
status: evergreen
updated: 2026-09-09
aliases:
  - "Mixed-Precision Training"
  - "AMP"
  - "Automatic Mixed Precision"
  - "Dynamic Loss Scaling"
  - "FP16 vs BF16"
---

# ⚡ Mixed-Precision Training (AMP): Dynamic Loss Scaling & Numerical Dynamics

## 1. High-Level Concept & The Floating-Point Underflow Dilemma

Modern deep learning vision backbones, transformers, and multimodal physical AI models possess tens of millions to billions of parameters. Executing forward and backward passes strictly in standard IEEE 754 32-bit Single Precision (**FP32**) introduces massive memory bandwidth bottlenecks, saturates GPU VRAM, and underutilizes specialized hardware arithmetic units (NVIDIA Tensor Cores).

### Floating-Point Bit Allocations (IEEE 754 vs. Bfloat16)

```
Floating-Point Format Structural Comparison:

FP32 (Single Precision, 32 bits):
[ Sign: 1b ][ Exponent: 8b (Range: ~10^-38 to 10^38) ][ Mantissa: 23b (~7 decimal digits) ]

FP16 (Half Precision, 16 bits):
[ Sign: 1b ][ Exponent: 5b (Range: 6e-5 to 65,504)   ][ Mantissa: 10b (~3 decimal digits) ]
                                ^ Danger: Tiny dynamic range!

BF16 (Brain Floating Point, 16 bits):
[ Sign: 1b ][ Exponent: 8b (Range: ~10^-38 to 10^38) ][ Mantissa: 7b (~2 decimal digits)  ]
              ^ Same dynamic range as FP32! No underflow!
```

---

### The Catastrophic Underflow Barrier in FP16
In deep neural networks during backpropagation:
- Over **$80\%$ of backpropagated gradient values** have absolute magnitudes smaller than $2^{-14} \approx 6.1 \times 10^{-5}$ (the smallest representable positive normalized number in FP16).
- If gradients are directly computed in raw FP16:
  $$\text{If } |g| < 2^{-24} \approx 5.96 \times 10^{-8} \implies g \xrightarrow{\text{flush to zero}} 0.0$$
- Small weight updates completely vanish, stalling training convergence and causing significant degradation in task metrics.

---

### The Automatic Mixed Precision (AMP) Solution (Micikevicius et al., ICLR 2018)
AMP combines the computational speed of 16-bit Tensor Cores with the numerical stability of 32-bit master weights using three complementary techniques:

#### 1. FP32 Master Weights
Parameters $\mathbf{W}_{\text{master}} \in \mathbb{R}^D$ are persistently stored in FP32. For each forward pass, weights are dynamically cast to FP16: $\mathbf{W}_{\text{half}} = \text{cast}_{\text{FP16}}(\mathbf{W}_{\text{master}})$. Activations, convolutions, and GEMM matrix multiplications execute at $2\times\text{ to }4\times$ speed on Tensor Cores.

#### 2. Dynamic Loss Scaling
To prevent gradient underflow during backward propagation:
Multiply the scalar loss $\mathcal{L}$ by a large scaling factor $S$ (e.g., $S = 2^{16} = 65,536$) **before invoking the backward pass**:
$$\mathcal{L}_{\text{scaled}} = S \cdot \mathcal{L}$$

By the linear property of the gradient operator:
$$\nabla_{\mathbf{W}} \mathcal{L}_{\text{scaled}} = \nabla_{\mathbf{W}} (S \cdot \mathcal{L}) = S \cdot \nabla_{\mathbf{W}} \mathcal{L}$$

Scaling the loss shifts the entire gradient magnitude histogram to the right by $\log_2(S)$ bit positions, moving small gradients out of the subnormal underflow zone and into the normal FP16 representable range!

#### 3. Unscaling & Adaptive Overflow Detection
Before updating the optimizer:
1. Divide gradients by $S$: $\mathbf{g} = \frac{1}{S} \nabla_{\mathbf{W}} \mathcal{L}_{\text{scaled}}$.
2. Check for `NaN` or `Inf` (overflow beyond $65,504$).
   - **If overflow detected**: Discard the weight update step entirely, and decrease the scale factor: $S \leftarrow \max(1, S \cdot \beta_{\text{backoff}})$ (e.g., $\beta_{\text{backoff}} = 0.5$).
   - **If no overflow occurs for $N$ consecutive steps** (e.g., $N = 2,000$): Increase the scale factor: $S \leftarrow S \cdot \beta_{\text{growth}}$ (e.g., $\beta_{\text{growth}} = 2.0$).

```
Automatic Mixed Precision (AMP) Execution Loop:

[ FP32 Master Weights W_master ]
              |
              v (Cast to FP16)
[ FP16 Forward Pass (Convolutions & Attention on Tensor Cores) ] ---> Loss L
                                                                           |
                                                                           v
                                                 [ Multiply by Scale Factor S ]
                                                                           |
                                                                           v
                                              [ FP16 Backward Pass: S * grad_L ]
                                                                           |
                                                                           v
                                              [ Unscale Gradients: grad / S ]
                                                                           |
                                        +----------------------------------+
                                        | Check for Inf / NaN ?
                                        |
                 +----------------------+----------------------+
                 | Yes (Overflow)                             | No (Valid)
                 v                                            v
        [ Skip Optimizer Step ]                      [ Optimizer Update on FP32 ]
        [ Scale Factor S /= 2 ]                      [ W_master -= lr * grad    ]
                                                     [ If 2000 steps clean: S *= 2 ]
```

---

## 2. Mathematical Formulation

### 2.1 Gradient Scale Invariance
Let forward network output be $\hat{\mathbf{y}} = f(\mathbf{x}; \mathbf{W})$ and loss be $\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y})$.
The gradient with respect to weight tensor $\mathbf{W}$ is:

$$
\mathbf{g} = \frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \left( \frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{W}} \right)^T \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}}
$$

When scaling by factor $S > 0$:

$$
\mathbf{g}_{\text{scaled}} = \frac{\partial (S \cdot \mathcal{L})}{\partial \mathbf{W}} = S \cdot \left( \frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{W}} \right)^T \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} = S \cdot \mathbf{g}
$$

The exact gradient is recovered through scalar division in FP32 precision:

$$
\mathbf{g}_{\text{recovered}} = \frac{1}{S} \mathbf{g}_{\text{scaled}} = \mathbf{g}
$$

---

### 2.2 Numerical Range & Precision Bounds
- **FP16 Underflow Bound**: Smallest positive normal number is $2^{-14} \approx 6.1035 \times 10^{-5}$. Denormals extend to $2^{-24} \approx 5.9605 \times 10^{-8}$.
- **FP16 Overflow Bound**: Largest finite positive number is $(2 - 2^{-10}) \times 2^{15} = 65,504$.
- **BF16 Advantage**: Range matches FP32 ($\approx 3.4 \times 10^{38}$), eliminating the need for loss scaling in most vision backbones, though its lower mantissa (7 bits) yields higher relative truncation errors ($\epsilon_{\text{mach}} = 2^{-7} \approx 7.8 \times 10^{-3}$ vs. $2^{-10} \approx 9.8 \times 10^{-4}$ in FP16).

---

## 3. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn

class CustomGradScaler:
    """
    Standalone implementation of dynamic loss scaling for FP16 training.
    """
    def __init__(self, init_scale: float = 65536.0, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scales loss before backward pass."""
        return loss * self.scale

    def step_and_update(self, optimizer: torch.optim.Optimizer, model_params: list[torch.nn.Parameter]) -> bool:
        """
        Unscales gradients, checks for Inf/NaN, steps optimizer, and updates scale factor.
        Returns: True if step was taken, False if skipped due to overflow.
        """
        inv_scale = 1.0 / self.scale
        has_overflow = False

        # 1. Unscale gradients and inspect for non-finite values
        for p in model_params:
            if p.grad is not None:
                p.grad.data.mul_(inv_scale)
                if not torch.isfinite(p.grad.data).all():
                    has_overflow = True
                    break

        # 2. Adaptive Scale Update
        if has_overflow:
            # Clear invalid gradients to prevent corrupted optimizer state
            optimizer.zero_grad(set_to_none=True)
            self.scale = max(1.0, self.scale * self.backoff_factor)
            self._growth_tracker = 0
            return False
        else:
            # Valid step: update weights
            optimizer.step()
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_tracker = 0
            return True


def demo_amp_training_step():
    """Demonstrates standard PyTorch native AMP execution."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 10)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    inputs = torch.randn(32, 512, device=device)
    targets = torch.randint(0, 10, (32,), device=device)
    criterion = nn.CrossEntropyLoss()

    # Forward pass under autocast context
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    # Scaled backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

---

## 4. Models in the Vault Utilizing Mixed Precision

- **[[architectures/vision-foundation-models/sam-2|SAM 2]] & [[architectures/vision-foundation-models/dinov2|DINOv2]]**: Pretrained using BF16/FP16 mixed-precision attention and GEMM.
- **[[architectures/real-time-detectors-and-segmenters/yolo26|YOLO26]] & [[architectures/real-time-detectors-and-segmenters/yolov8|YOLOv8]]**: Native AMP training preventing regression loss underflow.
- **[[architectures/multimodal-vlm-and-vla/openvla|OpenVLA]]**: Vision-Language-Action fine-tuning requiring BF16 to fit 7B parameters into multi-GPU clusters.
- **[[frameworks/tensorrt|NVIDIA TensorRT Playbook]]**: FP16/BF16 engine compilation maximizing hardware Tensor Core occupancy.

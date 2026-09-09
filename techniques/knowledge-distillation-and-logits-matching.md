---
title: "Knowledge Distillation (KD): Logits Matching & Dark Knowledge"
type: "Technique"
domain: "Edge Acceleration, Model Compression & Teacher-Student Learning"
tags:
  - technique
  - knowledge-distillation
  - teacher-student
  - logits-matching
  - dark-knowledge
  - model-compression
  - edge-ai
status: evergreen
updated: 2026-09-09
aliases:
  - "Knowledge Distillation"
  - "KD"
  - "Teacher-Student Distillation"
  - "Logits Matching"
  - "Dark Knowledge"
---

# 🎓 Knowledge Distillation (KD): Logits Matching & Dark Knowledge

## 1. High-Level Concept & The "Dark Knowledge" Discovery

In standard supervised deep learning, classification models are trained with discrete **one-hot target vectors**: $\mathbf{y} \in \{0, 1\}^C$ where $y_{\text{true}} = 1$ and $y_k = 0$ for all $k \ne \text{true}$.
- **The Information Limitation of Hard Labels**: Hard labels treat all non-ground-truth classes as equally improbable ($0.0$). A one-hot vector provides zero information about the semantic geometry of the visual world—it conveys that a "sports car" is not a "truck", but fails to convey that a "sports car" shares vastly more visual, geometric, and semantic features with a "sedan" than with a "refrigerator".

### The Knowledge Distillation Paradigm (Hinton et al., NeurIPS 2014)
A high-capacity, computationally intensive **Teacher Network** ($f_{\text{teacher}}$) transfers its learned representations to a compact, real-time **Student Network** ($f_{\text{student}}$).

#### Temperature Softening
Standard Softmax produces peaked, near-binary probability distributions where true classes approach $1.0$ and negative classes approach $0.0$.
By introducing a temperature parameter $T > 1$:

$$
q_i = \frac{\exp(z_i / T)}{\sum_{j=1}^C \exp(z_j / T)}
$$

As temperature increases ($T \in [2, 8]$), the probability distribution flattens, exposing the relative log-odds between negative classes. These subtle relative probabilities are what Geoffrey Hinton termed **"Dark Knowledge"**: the latent metric structure of the learned manifold!

---

### The Dual Distillation Objective
The student network is supervised simultaneously by two signals:
1. **Task Loss ($\mathcal{L}_{\text{task}}$)**: Cross-Entropy between student output at $T=1$ and ground-truth one-hot labels.
2. **Distillation Loss ($\mathcal{L}_{\text{KD}}$)**: Kullback-Leibler (KL) Divergence between softened teacher probabilities $\mathbf{p}^T$ and softened student probabilities $\mathbf{q}^T$.

$$\mathcal{L}_{\text{total}} = (1 - \alpha) \mathcal{L}_{\text{CE}}(\mathbf{y}, \mathbf{q}^{T=1}) + \alpha T^2 \mathcal{D}_{\text{KL}}(\mathbf{p}^T \;\|\; \mathbf{q}^T)$$

The **$T^2$ scaling factor** is mathematically essential: as $T$ increases, gradients produced by the soft targets scale as $1/T^2$. Multiplying by $T^2$ ensures that the gradient contribution from the teacher remains balanced with the hard task loss regardless of temperature!

```
Knowledge Distillation Topology:

Input Image x
      |
      +-----------------------------+
      |                             |
      v                             v
[ Heavy Teacher Model ]       [ Lightweight Student Model ]
      |                             |
      v Logits v_i                  v Logits z_i
[ Softmax(v / T) ]            [ Softmax(z / T) ]  ---> [ Softmax(z / 1) ]
      |                             |                           |
      v Soft Teacher Targets p_i    v Soft Student Targets q_i  v Student Preds
      +--------------+--------------+                           |
                     |                                          v
                     v                                    [ Cross-Entropy ] <--- One-Hot Label y
        [ KL Divergence: D_KL(p || q) ]                         |
                     |                                          |
                     v (* T^2)                                  v (* (1 - alpha))
                     +---------------------+--------------------+
                                           |
                                           v
                             Total Loss L = L_CE + T^2 * L_KD
```

---

## 2. Mathematical Formulation

### 2.1 The $T^2$ Gradient Proof
Let $z_i$ be the student logits and $v_i$ be the teacher logits. Let softened probabilities be:

$$
q_i = \frac{\exp(z_i / T)}{\sum_k \exp(z_k / T)}, \quad p_i = \frac{\exp(v_i / T)}{\sum_k \exp(v_k / T)}
$$

The Kullback-Leibler divergence is:

$$
\mathcal{D}_{\text{KL}}(\mathbf{p} \;\|\; \mathbf{q}) = \sum_{i=1}^C p_i \log \frac{p_i}{q_i} = \sum_{i=1}^C p_i \log p_i - \sum_{i=1}^C p_i \log q_i
$$

Differentiating with respect to student logit $z_i$:

$$
\frac{\partial \mathcal{D}_{\text{KL}}}{\partial z_i} = -\sum_{j=1}^C p_j \frac{\partial \log q_j}{\partial z_i} = -\sum_{j=1}^C p_j \frac{1}{q_j} \frac{\partial q_j}{\partial z_i}
$$

Since $\frac{\partial q_j}{\partial z_i} = \frac{1}{T} q_j (\delta_{ij} - q_i)$:

$$
\frac{\partial \mathcal{D}_{\text{KL}}}{\partial z_i} = -\sum_{j=1}^C \frac{p_j}{q_j} \cdot \frac{q_j (\delta_{ij} - q_i)}{T} = \frac{1}{T} \left( q_i \sum_{j=1}^C p_j - p_i \right) = \frac{1}{T} (q_i - p_i)
$$

Now, consider high temperature relative to logits ($T \gg |z_i|, |v_i|$). Using Taylor series expansion $\exp(x) \approx 1 + x$:

$$
q_i \approx \frac{1 + z_i / T}{C + \sum_k z_k / T} = \frac{1 + z_i / T}{C (1 + \bar{z} / T)} \approx \frac{1}{C} \left( 1 + \frac{z_i - \bar{z}}{T} \right)
$$

where $\bar{z} = \frac{1}{C} \sum_k z_k$. Similarly for teacher logits: $p_i \approx \frac{1}{C} \left( 1 + \frac{v_i - \bar{v}}{T} \right)$.
Substituting into the gradient:

$$
\frac{\partial \mathcal{D}_{\text{KL}}}{\partial z_i} \approx \frac{1}{T} \left[ \frac{1}{C} \left( \frac{z_i - \bar{z}}{T} - \frac{v_i - \bar{v}}{T} \right) \right] = \frac{1}{C T^2} \left( (z_i - \bar{z}) - (v_i - \bar{v}) \right)
$$

The gradient scales inversely with $T^2$ ($\propto \frac{1}{T^2}$).
Therefore, multiplying the distillation loss by $T^2$ eliminates temperature dependency, matching the gradient magnitude of standard cross-entropy!

---

### 2.2 Feature-Based Distillation (Hinton + FitNets)
Beyond output logits, intermediate feature representations can be supervised:

$$
\mathcal{L}_{\text{feat}} = \frac{1}{2} \left\| \mathbf{F}_{\text{teacher}} - \mathbf{W}_{\text{proj}} \mathbf{F}_{\text{student}} \right\|_F^2
$$

where $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{C_{\text{teacher}} \times C_{\text{student}}}$ is a $1 \times 1$ convolution aligning channel dimensions.

---

## 3. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Hinton Knowledge Distillation Loss combining Cross-Entropy and Temperature KL-Divergence.
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        student_logits: [B, C] unnormalized logits
        teacher_logits: [B, C] unnormalized teacher logits (detached)
        targets: [B] integer class labels
        """
        # 1. Hard Cross-Entropy Loss (T = 1)
        loss_ce = self.ce_loss(student_logits, targets)
        
        # 2. Soft Distillation Loss (Scaled by T)
        # Log-Softmax for student, Softmax for teacher
        log_prob_student = F.log_softmax(student_logits / self.T, dim=-1)
        prob_teacher = F.softmax(teacher_logits / self.T, dim=-1)
        
        # KL divergence multiplied by T^2 to balance gradient magnitudes
        loss_kd = self.kl_div(log_prob_student, prob_teacher) * (self.T ** 2)
        
        # Total balanced loss
        return (1.0 - self.alpha) * loss_ce + self.alpha * loss_kd


class FeatureDistillationLayer(nn.Module):
    """Aligns intermediate feature maps between student and teacher."""
    def __init__(self, in_channels_student: int, in_channels_teacher: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels_student, in_channels_teacher, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels_teacher)
        )

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """Mean squared error over projected feature maps."""
        proj_student = self.proj(student_feat)
        return F.mse_loss(proj_student, teacher_feat.detach())
```

---

## 4. Models in the Vault Utilizing Knowledge Distillation

- **[[architectures/backbones-and-edge-efficiency/fastvit|FastViT]]**: Leverages structural reparameterization and teacher distillation from large ViTs.
- **[[architectures/vision-foundation-models/depth-anything-v2|Depth Anything v2]]**: Distills high-capacity monocular depth teachers into lightweight real-time student variants.
- **[[architectures/vision-foundation-models/dinov2|DINOv2]]**: Uses self-distillation with a momentum teacher to stabilize representation learning.
- **[[architectures/real-time-detectors-and-segmenters/yolov10|YOLOv10]]**: Employs dual-label assignments and teacher-guided distillation for NMS-free training.

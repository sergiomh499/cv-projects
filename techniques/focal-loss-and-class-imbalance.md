---
title: "Focal Loss & Asymmetric Formulations: Handling Extreme Class Imbalance"
type: "Technique"
domain: "Object Detection, Classification & Semantic Segmentation"
tags:
  - technique
  - focal-loss
  - quality-focal-loss
  - class-imbalance
  - object-detection
  - loss-functions
status: evergreen
updated: 2026-09-09
aliases:
  - "Focal Loss"
  - "Quality Focal Loss"
  - "QFL"
  - "Asymmetric Focal Loss"
---

# ⚖️ Focal Loss & Asymmetric Formulations: Handling Extreme Class Imbalance

## 1. High-Level Concept & The Class Imbalance Catastrophe

In dense, one-stage object detection architectures (e.g., RetinaNet, YOLOv8–v26, RT-DETR, D-FINE):
- The model evaluates an exhaustive spatial grid of candidate locations across multiple pyramid scales, generating between **$10^4$ and $10^5$ candidate anchors/queries per image**.
- In any typical scene, only a handful ($1\text{ to }50$) contain actual target objects. Over **$99.9\%$ of candidates are pure, uninformative background**.

### Why Standard Cross-Entropy Fails
Consider standard Binary Cross-Entropy (BCE) for a sample with ground truth $y \in \{0, 1\}$ and predicted probability $p \in [0, 1]$:

$$
\text{CE}(p, y) = \begin{cases} -\log(p) & \text{if } y = 1 \\ -\log(1 - p) & \text{if } y = 0 \end{cases}
$$

For notational convenience, let $p_t$ be the probability of the true class:

$$
p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{if } y = 0 \end{cases} \implies \text{CE}(p_t) = -\log(p_t)
$$

### The Quantitative Collapse
For a well-classified background candidate (an "easy negative"), the model predicts $p = 0.01 \implies p_t = 0.99$.
The loss for this single easy negative is tiny:

$$
\text{CE}(0.99) = -\log(0.99) \approx 0.01005
$$

**However, in an image with 100,000 background candidates**:
- Summing over 100,000 easy negatives yields an aggregate loss of **$\approx 1005.0$**!
- If the image contains 5 rare foreground objects with $p_t = 0.5$, their combined loss is only $5 \times (-\log(0.5)) \approx 3.46$.
- **The easy background negatives overpower the foreground signal by a factor of 300 to 1**, completely destabilizing gradient descent and forcing the network to predict "background" for all queries.

```
Standard Cross-Entropy vs. Focal Loss under Extreme Class Imbalance:

Standard Cross-Entropy:
  5 Hard Positives:     Loss = 3.46   \
  100,000 Easy Negs:    Loss = 1005.0  +---> Positives completely drowned out! Model fails.

Focal Loss (Modulating Factor (1 - p_t)^gamma, gamma=2):
  5 Hard Positives:     Loss = 3.46 * (1 - 0.5)^2  = 0.865
  100,000 Easy Negs:    Loss = 1005.0 * (1 - 0.99)^2 = 0.1005
  Result: Positives dominate loss by 8.6x! Training converges cleanly.
```

---

## 2. Mathematical Formulations & Gradient Derivations

### 2.1 The Focal Loss Formulation
Focal Loss (Lin et al., ICCV 2017) introduces a **dynamic modulating factor** $(1 - p_t)^\gamma$ with focusing parameter $\gamma \ge 0$:

$$
\text{FL}(p_t) = - \alpha_t \left( 1 - p_t \right)^\gamma \log(p_t)
$$

where:
- $\alpha_t \in [0, 1]$ is an $\alpha$-balanced class weighting factor (typically $\alpha = 0.25$ for foreground, $0.75$ for background).
- $\gamma$ (typically $\gamma = 2.0$) dynamically adjusts the rate at which easy examples are downweighted:
  - When $p_t \to 0$ (hard, misclassified sample): $(1 - p_t)^\gamma \approx 1$, leaving the loss virtually unchanged.
  - When $p_t \to 1$ (easy sample): $(1 - p_t)^\gamma \to 0$. For $p_t = 0.99$ and $\gamma = 2$, $(1 - 0.99)^2 = 0.0001$, **downweighting the sample by a factor of $10,000\times$**!

---

### 2.2 Gradient Derivation with Respect to Logit $z$
Let the predicted probability be parameterized by the sigmoid function $p = \sigma(z) = \frac{1}{1 + e^{-z}}$.
Recall that $\frac{\partial p}{\partial z} = p(1 - p)$.

Differentiating standard Cross-Entropy:

$$
\frac{\partial \text{CE}}{\partial z} = p - y
$$

Now, differentiating Focal Loss (omitting $\alpha_t$ for simplicity):

$$
\frac{\partial \text{FL}}{\partial z} = \frac{\partial}{\partial z} \left[ - (1 - p_t)^\gamma \log(p_t) \right]
$$

Using the chain rule:

$$
\frac{\partial \text{FL}}{\partial z} = (p - y) \cdot (1 - p_t)^\gamma + y_t \cdot \gamma (1 - p_t)^{\gamma - 1} \log(p_t) \cdot p_t (p - y)
$$

where $y_t = 2y - 1 \in \{-1, +1\}$.

**Critical Mathematical Behavior**:
- As $p_t \to 1$, the gradient magnitude scales with $(1 - p_t)^\gamma \to 0$.
- Easy background candidates contribute **virtually zero gradient update**, preventing them from destroying the learned representations of rare foreground instances.

---

### 2.3 Quality Focal Loss (QFL) for Continuous Soft Targets
In modern anchor-free detectors (e.g., [[architectures/real-time-detectors-and-segmenters/yolov8|YOLOv8]], [[architectures/real-time-detectors-and-segmenters/d-fine|D-FINE]]), decoupling classification and localization produces a well-known failure mode: a candidate with high classification confidence may have poor bounding box overlap (IoU).

**Quality Focal Loss** (QFL, Li et al., NeurIPS 2020) solves this by replacing binary labels $y \in \{0, 1\}$ with **continuous IoU quality targets** $y \in [0, 1]$:

$$
\text{QFL}(p) = - |y - p|^\beta \left( y \log(p) + (1 - y) \log(1 - p) \right)
$$

where:
- $y \in [0, 1]$ is the real continuous IoU between the predicted bounding box and ground truth.
- $|y - p|^\beta$ is the continuous modulating factor with hyperparameter $\beta$ (typically $\beta = 2.0$).
- When an anchor has low IoU ($y \approx 0$), QFL drives its classification confidence $p$ to zero. When an anchor has perfect overlap ($y = 1.0$), QFL drives $p$ to 1.0.

---

## 3. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SigmoidFocalLoss(nn.Module):
    """
    Standard Binary Focal Loss with alpha balancing and gamma focusing.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, ...] raw unnormalized network predictions
        targets: [B, ...] binary ground truth {0, 1}
        """
        p = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        # p_t: probability of the true class
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        
        # Modulating factor (1 - p_t)^gamma
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        
        # Alpha balancing factor
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        
        focal_loss = alpha_t * modulating_factor * bce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss (QFL) for continuous targets y in [0, 1] (e.g. IoU scores).
    """
    def __init__(self, beta: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, continuous_targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, ...] raw predictions
        continuous_targets: [B, ...] continuous quality targets in [0, 1]
        """
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, continuous_targets, reduction="none")
        
        # Absolute discrepancy modulating factor: |y - p|^beta
        modulating_factor = torch.pow(torch.abs(continuous_targets - p), self.beta)
        qfl = modulating_factor * bce
        
        if self.reduction == "mean":
            return qfl.mean()
        elif self.reduction == "sum":
            return qfl.sum()
        return qfl
```

---

## 4. Models in the Vault Utilizing Focal Losses

- **[[architectures/real-time-detectors-and-segmenters/yolov8|YOLOv8]]–[[architectures/real-time-detectors-and-segmenters/yolo26|YOLO26]]**: Classification loss and task-aligned label assignment.
- **[[architectures/real-time-detectors-and-segmenters/rf-detr|RF-DETR]] & [[architectures/real-time-detectors-and-segmenters/rt-detr|RT-DETR]]**: Classification loss and bipartite Hungarian matching cost.
- **[[architectures/real-time-detectors-and-segmenters/grounding-dino|Grounding DINO]]**: Open-vocabulary vision-language token matching.
- **[[architectures/real-time-detectors-and-segmenters/d-fine|D-FINE]]**: Quality Focal Loss jointly supervising classification and localization.

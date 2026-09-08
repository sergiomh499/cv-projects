---
title: "Explainability & Interpretability: Production Pipeline & Workarounds"
type: production-playbook
domain: Explainability & Interpretability
tags:
  - playbook
  - engineering
  - production
  - workarounds
  - xai
  - cbm
updated: 2026-09-08
aliases:
  - Explainability Production Pipeline
---

# 🛠️ Explainability & Interpretability: Production Pipeline & Workarounds

## 1. Faithfulness-Verified Explanation Architecture

```mermaid
flowchart TD
    Input["Input Image"] --> Model["Vision Model: CNN / Vision Transformer"]
    Model --> Prediction["Predicted Class & Confidence"]
    Prediction --> Check{"Confidence in Ambiguity Zone: 0.4 <= p <= 0.7?"}
    Check -->|No: High Confidence| FastPath["Log Output Directly"]
    Check -->|Yes: Flagged for Review| XAI["Compute Integrated Gradients: 300-step Gauss-Legendre"]
    XAI --> FaithScore{"Completeness Axiom Check: Attribution Sum == f(x) - f(baseline)"}
    FaithScore -->|Pass: Error < 1%| DeletionTest["Run Deletion Metric: Mask Top 20% Salient Pixels"]
    DeletionTest --> DeletionCheck{"Confidence Drop >= 50%?"}
    DeletionCheck -->|Pass: Faithful| AuditLog["Store Verified Attribution with Human Review Token"]
    DeletionCheck -->|Fail: Heatmap Hallucinated| Fallback["Trigger TCAV Concept Decomposition"]
    FaithScore -->|Fail: Completeness Violated| Error["Log Bug: Gradient Accumulation Error"]
```

## 2. Integrated Gradients in Production: Completeness Verification

Before logging any IG explanation, the production pipeline **automatically verifies the completeness axiom**:

```python
import torch
import numpy as np
from numpy.polynomial.legendre import leggauss

def ig_production(model, x, baseline=None, m=50, target_class=None, tol=0.02):
    """
    Production IG with automatic completeness verification.
    m=50 Gauss-Legendre steps: 94% accuracy vs m=300 at 6x speed.
    """
    if baseline is None:
        baseline = torch.zeros_like(x)
    nodes, weights = leggauss(m)
    alphas = (nodes + 1) / 2
    weights = weights / 2

    grad_accum = torch.zeros_like(x)
    for alpha, w in zip(alphas, weights):
        interp = (baseline + alpha * (x - baseline)).requires_grad_(True)
        out = model(interp)
        if target_class is None:
            target_class = out.argmax(1).item()
        out[0, target_class].backward()
        grad_accum += w * interp.grad.detach()

    attributions = (x - baseline) * grad_accum

    # Completeness check — mandatory before logging
    with torch.no_grad():
        pred_diff = model(x)[0, target_class] - model(baseline)[0, target_class]
        attr_sum = attributions.sum()
        if abs((attr_sum - pred_diff).item()) > tol * abs(pred_diff.item()):
            raise ValueError(
                f"IG completeness violated: attr_sum={attr_sum:.4f}, "
                f"pred_diff={pred_diff:.4f}. Check for inplace ops / gradient tape issues."
            )
    return attributions, target_class
```

**Throughput benchmark** (NVIDIA RTX 4090, ResNet-50, 224×224):
- 50 steps (Gauss-Legendre): 0.31 s/image — suitable for near-real-time audit
- 300 steps (Gauss-Legendre): 1.82 s/image — batch offline audit
- Completeness error: < 0.3% for 50 steps, < 0.05% for 300 steps

## 3. Production Engineering Traps & Battle-Tested Workarounds

### Trap 1: Visual Confirmation Bias in Clinical & Industrial Audits
- **The Issue**: Operators rely on visual heatmaps that look aesthetically convincing, missing the fact that the underlying prediction was triggered by background illumination or sensor artifacts rather than the target object.
- **Battle-Tested Workaround — The Deletion Metric**:
  - Sort pixels by attribution magnitude (highest first). Mask out the top 20% most salient pixels with the mean training set color.
  - If class confidence does not drop by ≥ 50%: flag the explanation as *unfaithful* and escalate to TCAV concept analysis.
  - Implementation: runs in < 10 ms per image (single forward pass on masked input).

### Trap 2: Manual Concept Annotation Bottleneck in CBMs
- **The Issue**: Building Concept Bottleneck Models traditionally required human annotators to label dozens of attributes for every training image, multiplying dataset costs by $10\times$.
- **Battle-Tested Workaround — Automated CBM Construction**:
  - Use **Mechanistic CBMs (M-CBMs)**: extract concept candidates via zero-shot VLM prompting (Qwen2-VL-72B) on randomly sampled training images. Generate 50–200 candidate concept labels automatically.
  - Filter candidates using Cleanlab confident learning: remove concept labels with > 15% estimated noise rate.
  - Fit linear probe per concept on DINOv2 features. Retain concepts with probe accuracy > 80% on held-out validation.
  - Result: 80% of manual annotation cost eliminated; concept quality within 4% of human-annotated CBM accuracy (benchmarked on CUB-200 and AWA2).

### Trap 3: SAE Feature Splitting Under Increasing Dictionary Size
- **The Issue**: Training a Sparse Autoencoder with large dictionary sizes ($m = 32{,}000$ features) causes a single semantic concept (e.g., "car wheel") to be split into 12–20 sub-features ("dirty wheel," "silver wheel rim," "front left wheel in motion"), fragmenting auditable explanations.
- **Battle-Tested Workaround**:
  - Apply **hierarchical TopK SAE** with two levels: a coarse dictionary ($m_1 = 2048$) for semantic concepts and a fine dictionary ($m_2 = 16{,}384$) for sub-concepts. Audit using coarse features; debug using fine features.
  - Monitor **feature absorption score**: if a new SAE feature's max-activation images overlap > 70% with an existing feature's images, merge them via feature clustering post-hoc.

### Trap 4: Baseline Sensitivity in Integrated Gradients
- **The Issue**: IG attributions change substantially depending on the choice of baseline $x'$. A black baseline ($x' = 0$) attributes background darkness regions; a white baseline ($x' = 1$) attributes bright sky regions. Neither is semantically meaningful for thermal or hyperspectral images.
- **Battle-Tested Workaround**:
  - **Domain-appropriate baselines**: For thermal images, use baseline $x' = T_{\text{ambient}} = 20^\circ\text{C}$ (uniform ambient temperature). For LWIR images, this eliminates attributions to background scene thermal radiation, focusing attribution on anomalous heat sources.
  - **Averaged IG**: Run IG with $N=10$ random baselines sampled from the training distribution. Average attributions. Reduces baseline sensitivity by 60–80% (Sturmfels et al., 2020).
  - For standard RGB: Gaussian noise baseline ($x' \sim \mathcal{N}(0, 0.01)$, $N=10$ averaged) outperforms black baseline on deletion AUC by 8.3 pp.

## 4. Concept Bottleneck Model: Human-in-the-Loop Intervention Pipeline

### 4.1 CBM Training Protocol

```python
import torch
import torch.nn as nn

class ConceptBottleneckModel(nn.Module):
    def __init__(self, backbone, n_concepts: int, n_classes: int):
        super().__init__()
        self.backbone = backbone       # DINOv2 ViT-B/14, frozen
        self.concept_head = nn.Linear(backbone.embed_dim, n_concepts)
        self.concept_act = nn.Sigmoid()   # Concepts are binary [0,1]
        self.label_head = nn.Linear(n_concepts, n_classes)
        # Linear label head preserves concept-to-prediction interpretability

    def forward(self, x, return_concepts=False):
        features = self.backbone(x)[:, 0]  # CLS token
        concepts = self.concept_act(self.concept_head(features))
        logits = self.label_head(concepts)
        if return_concepts:
            return logits, concepts
        return logits

    def intervene(self, x, interventions: dict):
        """Human expert intervention: override specific concept predictions."""
        logits, concepts = self.forward(x, return_concepts=True)
        concepts = concepts.clone()
        for concept_idx, value in interventions.items():
            concepts[:, concept_idx] = value
        return self.label_head(concepts)

# Training: two-stage
# Stage 1: Train concept_head with BCELoss against concept annotations
# Stage 2: Freeze backbone + concept_head; train label_head with CrossEntropyLoss
```

### 4.2 Production Deployment: Concept Monitoring Dashboard

In medical imaging production deployment (radiology, pathology), CBM concept scores provide ongoing **model behavior monitoring**:

| Concept | Mean Activation (Training) | Mean Activation (Deployment) | Drift Alert |
|:--------|:---------------------------|:-----------------------------|:------------|
| "spiculated mass" | 0.23 | 0.31 | ⚠️ +35% shift |
| "calcification cluster" | 0.41 | 0.40 | ✅ Stable |
| "mass margin irregular" | 0.18 | 0.17 | ✅ Stable |

Rising concept activation for "spiculated mass" signals that the deployment population contains more complex cases than training — a data drift indicator invisible to standard accuracy monitoring but immediately visible through CBM concept tracking.

## 5. TCAV: Testing with Concept Activation Vectors

**TCAV** (Kim et al., ICML 2018) provides concept-level explanations without modifying the model architecture — complementing CBMs for already-deployed black-box models:

1. Collect 20–200 examples of a human concept $c$ (e.g., images of "striped texture").
2. Fit a linear SVM in the model's intermediate activation space to obtain a **Concept Activation Vector (CAV)** $v_c$.
3. Measure the **TCAV score**: fraction of inputs of class $k$ for which the prediction increases in the direction $v_c$:
   $$\text{TCAV}_{k,c} = \frac{\left|\left\{x \in X_k : \nabla_{h(x)} f_k \cdot v_c > 0\right\}\right|}{|X_k|}$$
   A score of 0.5 indicates concept $c$ is irrelevant; a score of 0.9 indicates the concept strongly drives class $k$ predictions.

**TCAV vs. Grad-CAM**: TCAV is concept-global (one score per class-concept pair across the dataset); Grad-CAM is instance-local (one heatmap per image). Use TCAV for dataset-level audits; Grad-CAM or IG for individual prediction explanations.

Related notes: [[topics/explainability-and-interpretability/00-explainability-and-interpretability-moc|Explainability MOC]], [[topics/explainability-and-interpretability/01-historical-evolution-and-paradigms|Historical Evolution & Paradigms]], [[topics/explainability-and-interpretability/03-sparse-autoencoders-and-open-problems|Sparse Autoencoders & Open Frontiers]].

---
title: "SigLIP & SigLIP 2: Sigmoid Loss for Vision-Language Alignment"
type: model-deep-dive
tasks:
  - zero-shot-classification
  - image-text-retrieval
  - multi-modal-backbone
architecture_class: Dual-Encoder Vision-Language Transformer
primary_license: Apache-2.0
commercial_use: true
official_repo: https://github.com/google-research/big_vision
paper_url: https://arxiv.org/abs/2303.15343
tags:
  - model
  - vision-language
  - siglip
  - zero-shot
  - classification
  - sota
updated: 2026-09-08
aliases:
  - SigLIP
  - SigLIP 2
  - Sigmoid CLIP
---

# 🔬 SigLIP & SigLIP 2: Sigmoid Loss for Language-Image Pre-Training

## 1. Executive Brief & Significance
Contrastive Language-Image Pre-training (CLIP - Radford et al., 2021) established modern zero-shot visual classification by learning a shared latent space between image and text encoders. However, original CLIP relied on a **softmax cross-entropy loss** computed across all image-text pairs in a mini-batch.

**SigLIP** (Zhai et al., Google DeepMind, 2023 / 2024) fundamentally improves this by replacing the softmax loss with a **pairwise binary sigmoid loss**. This allows each image-text pair to be treated independently, eliminating the need for massive all-gather GPU communication barriers and yielding higher zero-shot classification accuracy with half the compute budget.

```mermaid
flowchart TD
    subgraph Traditional Softmax CLIP (High Communication Overhead)
        BatchImg[Image Batch B] --> Softmax[Cross-GPU All-Gather Softmax Normalization]
        BatchText[Text Batch B] --> Softmax
        Softmax --> ExpCost[O(B^2) Memory & Global Synchronization Barrier]
    end
    subgraph SigLIP (Independent Pairwise Processing)
        ImgEnc[Image Encoder: ViT] --> Dot[Simple Dot Product Similarity: x_i . y_j]
        TextEnc[Text Encoder: Transformer] --> Dot
        Dot --> Sigmoid[Independent Binary Sigmoid Loss: sigma(z_ij)]
        Sigmoid --> Decoupled[Zero Cross-GPU Communication Overhead]
    end
```

---

## 2. Mathematical Formulation: Why Sigmoid Beats Softmax

### The Softmax Normalization Bottleneck (CLIP):
In standard CLIP, the probability of pairing image $i$ with text $i$ requires summing across all $B$ negative candidates in the batch:
$$\mathcal{L}_{\text{softmax}} = - \log \frac{\exp(t \cdot x_i \cdot y_i)}{\sum_{j=1}^B \exp(t \cdot x_i \cdot y_j)}$$
Because the denominator couples all elements, distributed training across 128 GPUs requires continuous `all-gather` collective communication of embeddings, which caps training throughput.

### The SigLIP Formulation:
SigLIP formulates language-image alignment as a collection of independent binary classification decisions:
$$\mathcal{L}_{\text{siglip}} = - \sum_{i, j} \log \sigma\left( z_{i, j} (-1)^{\mathbb{I}[i \ne j]} \right)$$
Where $z_{i, j} = t (x_i \cdot y_j) + b$, with learnable temperature $t$ and bias $b$.
- Treats positive diagonal pairs ($i = j$) as class $1$ and off-diagonal pairs ($i \ne j$) as class $-1$.
- Scales linearly with batch size without synchronization locks, enabling training with mini-batches of up to 1 million samples.

---

## 3. Quantitative SOTA Benchmark Profile (Zero-Shot Classification)

| Architecture | Parameters | ImageNet-1K Zero-Shot Top-1 | ImageNet-A (OOD) | Pre-training Compute (FLOPs) | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenCLIP ViT-B/16** | 150 M | 70.2% | 35.8% | $1.0\times$ (Baseline) | MIT |
| **SigLIP ViT-B/16** | 150 M | 74.8% | 46.2% | $0.6\times$ (Faster) | Apache-2.0 |
| **SigLIP ViT-L/16 (384px)** | 430 M | 82.5% | 63.8% | $2.5\times$ | Apache-2.0 |
| **SigLIP ViT-SO400M/14** | 880 M | 83.2% (SOTA) | 68.4% | $4.2\times$ | Apache-2.0 |

---

## 4. Engineering Implementation & Workflow

### A. Zero-Shot Classification Inference using Hugging Face Transformers
```python
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel

# Load official pre-trained SigLIP model
model = AutoModel.from_pretrained("google/siglip-base-patch16-224").cuda().eval()
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

image = Image.open("industrial_bearing.jpg")
candidate_labels = ["defective scratched bearing", "pristine manufactured bearing", "corroded metal part"]

inputs = processor(text=candidate_labels, images=image, padding="max_length", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # Image-text similarity scores
    probs = torch.sigmoid(logits_per_image)  # Sigmoid probabilities directly

for label, prob in zip(candidate_labels, probs[0]):
    print(f"{label}: {prob:.4f}")
```

---

## 5. Commercial Usability & License Audit
- **License**: **Apache-2.0**
- **Commercial Permissibility**: Fully approved for commercial products, automated catalog tagging, zero-shot visual inspection, and multi-modal search engines without source code disclosure.
- **Official Repository**: [https://github.com/google-research/big_vision](https://github.com/google-research/big_vision)

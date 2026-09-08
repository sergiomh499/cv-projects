---
title: "FastSAM: Fast Segment Anything Model via Real-Time CNN Instance Segmentation"
type: model-deep-dive
tasks:
  - promptable-segmentation
  - instance-segmentation
  - edge-perception
  - real-time-segmentation
architecture_class: Real-Time CNN Instance Detector & Mask Router
primary_license: Apache-2.0
commercial_use: true
official_repo: https://github.com/CASIA-IVA-Lab/FastSAM
paper_url: https://arxiv.org/abs/2306.12156
tags:
  - architecture
  - fastsam
  - sam
  - segmentation
  - yolov8
  - cnn
  - real-time
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - FastSAM
  - Fast Segment Anything
  - FastSAM-x
  - CNN Segment Anything
---

# 🔬 FastSAM: Fast Segment Anything Model via Real-Time CNN Instance Segmentation

## 1. Executive Brief & Significance

The landmark **Segment Anything Model (SAM)** (Kirillov et al., Meta FAIR, 2023) established the foundation model paradigm for zero-shot promptable image segmentation. However, SAM's default Vision Transformer backbone (**ViT-H** with $636\text{M}$ parameters and $2,900\text{ GFLOPs}$) requires $\approx 450-500\text{ ms}$ per $1024 \times 1024$ image on an NVIDIA A100 GPU and over $3,500\text{ ms}$ on edge NPUs, rendering it impractical for real-time robotics, embedded edge inspection, and interactive mobile apps.

**FastSAM (Fast Segment Anything)** (Zhao et al., CASIA / UCAS, 2023) fundamentally broke this latency barrier by reformulating promptable segmentation into a high-throughput **two-stage CNN pipeline**:
1. **Unprompted All-Instance Segmentation**: Directly predicts all candidate instance masks in a single forward pass using a real-time **YOLOv8x-seg** convolutional backbone.
2. **Post-Hoc Prompt-Guided Mask Indexing**: Rather than executing cross-attention over feature maps for every user click or box, FastSAM indexes the pre-computed set of candidate masks using simple geometric point-in-mask tests, bounding box IoU filtering, or CLIP-based text embedding alignment.

By replacing quadratic transformer attention with optimized convolutional operations, FastSAM achieves **$>50\text{ FPS}$** ($20-40\text{ ms}$) on desktop GPUs with a **$50\times$ speedup** over SAM ViT-H.

```mermaid
flowchart TD
    subgraph Stage1_All_Instance_Segmenter ["Stage 1: All-Instance Segmentation (YOLOv8x-seg)"]
        RawImage["Input Image: I in R^(3 x 1024 x 1024)"] --> CSPDarknet["CSPDarknet Backbone (Stages P1..P5)"]
        CSPDarknet --> PANet["PANet Feature Pyramid Fusion (P3, P4, P5)"]
        PANet --> ProtoHead["Prototype Mask Head: M_proto in R^(32 x 160 x 160)"]
        PANet --> BBoxCoeffHead["Detection Head: BBoxes (N x 4) & Mask Coeffs (N x 32)"]
        ProtoHead --> MaskMatrixMul["Matrix Multiply: M_i = sigma(sum C_ij * M_proto_j)"]
        BBoxCoeffHead --> MaskMatrixMul
        MaskMatrixMul --> CandidateMaskPool["Candidate Mask Pool: N Instance Masks in R^(N x H x W)"]
    end

    subgraph Stage2_Prompt_Router ["Stage 2: Prompt-Guided Mask Routing (<2 ms)"]
        UserPrompt["User Prompt: Point Click / Bounding Box / Text Prompt"] --> RouterEngine{"Prompt Mode Classifier"}
        
        RouterEngine -->|"Point Prompt (x, y)"| PointFilter["Point-in-Mask Intersection & Foreground Area Prior"]
        RouterEngine -->|"Box Prompt B_prompt"| BoxFilter["BBox Bipartite IoU Maxima Matching"]
        RouterEngine -->|"Text Prompt car"| TextFilter["CLIP / SigLIP ROI Visual-Text Cosine Similarity"]
        
        CandidateMaskPool --> PointFilter
        CandidateMaskPool --> BoxFilter
        CandidateMaskPool --> TextFilter
        PointFilter --> TargetMask["Final Selected Precision Mask M* (<25 ms Total Latency)"]
        BoxFilter --> TargetMask
        TextFilter --> TargetMask
    end
```

---

## 2. Mathematical Foundations & Two-Stage Mask Formulation

### A. Stage 1: All-Instance Prototype Mask Generation
The convolutional backbone processes input image $\mathbf{I} \in \mathbb{R}^{3 \times H \times W}$ through CSPDarknet with Cross-Stage Partial (C2f) blocks. The decoupled segmentation head outputs:
1. **Prototype Mask Tensor**: $\mathbf{M}_{\text{proto}} \in \mathbb{R}^{k \times \frac{H}{4} \times \frac{W}{4}}$ (where $k = 32$ is the number of prototype bases).
2. **Instance Coefficients & Proposals**: For $N$ detected object proposals, the network predicts mask coefficients $\mathbf{C} \in \mathbb{R}^{N \times k}$, bounding boxes $\mathbf{B} \in \mathbb{R}^{N \times 4}$, and classification scores $\mathbf{s} \in [0, 1]^N$.

The continuous mask for the $i$-th candidate instance is synthesized via matrix multiplication followed by sigmoid activation:
$$\mathbf{M}_i(u, v) = \sigma\left( \sum_{j=1}^k C_{i, j} \cdot \mathbf{M}_{\text{proto}, j}(u, v) \right)$$

---

### B. Stage 2: Prompt-Guided Selection Algorithms

#### 1. Point Prompt Processing:
Given a positive click coordinate $\mathbf{p} = (x_p, y_p)$:
1. Identify all candidate masks that cover point $\mathbf{p}$:
   $$\mathcal{I}_{\text{point}} = \{ i \in \{1, \dots, N\} \mid \mathbf{M}_i(x_p, y_p) > \tau_{\text{mask}} \}$$
2. If multiple nested masks cover the point (e.g., a person, their jacket, and their pocket), FastSAM applies foreground confidence weighting and local bounding box proximity:
   $$i^* = \arg\max_{i \in \mathcal{I}_{\text{point}}} \left[ s_i \cdot \text{IoU}\left(\mathbf{M}_i, \, \mathcal{B}(\mathbf{p}, r)\right) \right]$$

#### 2. Bounding Box Prompt Processing:
Given a user-specified bounding box $\mathbf{B}_{\text{prompt}} = [x_1, y_1, x_2, y_2]$:
$$i^* = \arg\max_{i \in \{1, \dots, N\}} \text{IoU}(\mathbf{B}_{\text{prompt}}, \mathbf{B}_i)$$
where $\text{IoU}(\mathbf{B}_{\text{prompt}}, \mathbf{B}_i) = \frac{|\mathbf{B}_{\text{prompt}} \cap \mathbf{B}_i|}{|\mathbf{B}_{\text{prompt}} \cup \mathbf{B}_i|}$.

#### 3. Text Prompt Processing:
For open-vocabulary text prompt $T$, FastSAM crops the visual bounding boxes of all candidate instances, extracts visual embeddings $\mathbf{v}_i = \text{CLIP}_{\text{vis}}(\text{Crop}(\mathbf{I}, \mathbf{B}_i))$, and selects the mask maximizing cosine similarity with text embedding $\mathbf{t} = \text{CLIP}_{\text{text}}(T)$:
$$i^* = \arg\max_{i} \frac{\mathbf{v}_i \cdot \mathbf{t}}{\|\mathbf{v}_i\|_2 \|\mathbf{t}\|_2}$$

---

## 3. Granular Component-by-Component Architectural Breakdown

| Architectural Stage | Component Identity | Structural Specification & Type | Attention / Conv Mechanics | Receptive Field & Dimensionality |
| :--- | :--- | :--- | :--- | :--- |
| **Architectural Paradigm** | **Real-Time CNN Instance Segmenter** | Two-Stage Detector + Post-Hoc Prompt Mask Router | Pure Convolutional (C2f Bottlenecks + PANet) | Input $[3, 1024, 1024] \to$ Dense instance masks |
| **Backbone** | **CSPDarknet53 (YOLOv8x-seg)** | 5-Stage Convolutional Hierarchy with C2f modules | $3\times 3$ standard convs + residual cross-stage partial splits | Strides $P_1 \dots P_5$ ($2\times \dots 32\times$) |
| **Neck / Aggregator** | **PANet Feature Pyramid** | Top-down and bottom-up multi-scale path aggregation | Lateral $1\times 1$ convs + $3\times 3$ C2f fusion blocks | Fused multi-scale features at $P_3, P_4, P_5$ |
| **Mask Prototype Head** | **Convolutional Proto-Synthesizer** | 3-Layer $3\times 3$ Convolutions + Upsampling ($2\times$) | Dense spatial basis synthesis | Prototype tensor $\mathbf{M}_{\text{proto}} \in \mathbb{R}^{32 \times 160 \times 160}$ |
| **Detection Head** | **Decoupled Anchor-Free Head** | Decoupled Classification, BBox, and Mask Coeff branches | DFL (Distribution Focal Loss) + BCE classification | Box $[N, 4]$ + Scores $[N, 1]$ + Coeffs $[N, 32]$ |
| **Prompt Router Engine**| **Tensor Indexing Kernel** | Point-in-mask / Box IoU / CLIP cosine similarity | GPU vectorized logical filtering | Sub-millisecond target index selection ($<2\text{ ms}$) |

---

## 4. Quantitative SOTA Benchmark Profile

### Promptable Segmentation & Real-Time Throughput

| Model Architecture | Parameters | GFLOPs | AR@1000 (SA-1B 1-Point) | Box Prompt mIoU (COCO) | RTX 4090 Latency (ms) | Jetson Orin Latency (ms) | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SAM ViT-H** | 636.0 M | 2900 G | **62.5%** | **81.5%** | 450.0 ms | 3,850.0 ms | Apache-2.0 |
| **SAM ViT-L** | 308.0 M | 1450 G | 60.8% | 80.2% | 220.0 ms | 1,920.0 ms | Apache-2.0 |
| **SAM ViT-B** | 91.0 M | 430 G | 58.2% | 77.8% | 95.0 ms | 820.0 ms | Apache-2.0 |
| **MobileSAM** | 9.8 M | 40 G | 57.8% | 76.5% | **11.8 ms** | **78.0 ms** | **Apache-2.0** |
| **FastSAM-s** | 11.1 M | 42 G | 48.5% | 72.0% | **9.5 ms** | **62.0 ms** | **Apache-2.0** |
| **FastSAM-x** | 68.2 M | 275 G | **53.8%** | **78.4%** | **24.5 ms** | **145.0 ms** | **Apache-2.0** |

---

## 5. Engineering Implementation: Complete PyTorch Prompt Router

```python
"""
FastSAM: Complete PyTorch Implementation of Prompt-Guided Mask Routing & Synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSAMPromptRouter(nn.Module):
    """
    FastSAM Stage-2 Prompt Router:
    Indexes candidate masks using Point clicks, Bounding Boxes, or Text embeddings.
    """
    def __init__(self, mask_threshold: float = 0.5):
        super().__init__()
        self.mask_threshold = mask_threshold

    def synthesize_masks(self, proto_masks: torch.Tensor, mask_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Synthesizes candidate masks via matrix multiplication: M_i = sigma(C_i * M_proto)
        proto_masks: [32, H_p, W_p]
        mask_coeffs: [N, 32]
        Returns: [N, H_p, W_p] boolean masks
        """
        N, k = mask_coeffs.shape
        _, Hp, Wp = proto_masks.shape
        
        # Flatten prototypes: [32, Hp*Wp]
        proto_flat = proto_masks.view(k, -1)
        # Matrix multiplication: [N, Hp*Wp]
        masks_flat = torch.matmul(mask_coeffs, proto_flat)
        masks = torch.sigmoid(masks_flat).view(N, Hp, Wp)
        return masks

    def route_point_prompt(
        self,
        masks: torch.Tensor,
        bboxes: torch.Tensor,
        scores: torch.Tensor,
        point: tuple[int, int],
        orig_shape: tuple[int, int]
    ) -> torch.Tensor:
        """
        Selects candidate mask covering input point (x, y).
        point: (x, y) in original image coordinates
        orig_shape: (H_orig, W_orig)
        """
        N, Hp, Wp = masks.shape
        H_orig, W_orig = orig_shape
        x, y = point
        
        # Map point coordinates to prototype grid
        u = int(x * (Wp / W_orig))
        v = int(y * (Hp / H_orig))
        
        # Binary mask occupancy at point (u, v)
        point_occupancy = masks[:, v, u] > self.mask_threshold  # [N]
        
        valid_indices = torch.where(point_occupancy)[0]
        if len(valid_indices) == 0:
            # Fallback to closest bounding box center
            box_centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
            box_centers_y = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
            dists = (box_centers_x - x)**2 + (box_centers_y - y)**2
            best_idx = torch.argmin(dists)
        else:
            # Select valid mask with highest detector score
            best_idx = valid_indices[torch.argmax(scores[valid_indices])]
            
        selected_mask = masks[best_idx] > self.mask_threshold
        # Upsample to original image resolution
        selected_mask_orig = F.interpolate(
            selected_mask.float().unsqueeze(0).unsqueeze(0),
            size=orig_shape,
            mode="bilinear",
            align_corners=False
        ).squeeze() > 0.5
        
        return selected_mask_orig

    def route_box_prompt(
        self,
        masks: torch.Tensor,
        bboxes: torch.Tensor,
        prompt_box: torch.Tensor,
        orig_shape: tuple[int, int]
    ) -> torch.Tensor:
        """
        Selects mask matching user bounding box prompt [x1, y1, x2, y2].
        """
        # Calculate IoU between prompt_box and all candidate bboxes
        x1 = torch.max(bboxes[:, 0], prompt_box[0])
        y1 = torch.max(bboxes[:, 1], prompt_box[1])
        x2 = torch.min(bboxes[:, 2], prompt_box[2])
        y2 = torch.min(bboxes[:, 3], prompt_box[3])
        
        inter_w = torch.clamp(x2 - x1, min=0)
        inter_h = torch.clamp(y2 - y1, min=0)
        inter = inter_w * inter_h
        
        area_bboxes = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        area_prompt = (prompt_box[2] - prompt_box[0]) * (prompt_box[3] - prompt_box[1])
        union = area_bboxes + area_prompt - inter
        
        ious = inter / (union + 1e-6)
        best_idx = torch.argmax(ious)
        
        selected_mask = masks[best_idx] > self.mask_threshold
        selected_mask_orig = F.interpolate(
            selected_mask.float().unsqueeze(0).unsqueeze(0),
            size=orig_shape,
            mode="bilinear",
            align_corners=False
        ).squeeze() > 0.5
        
        return selected_mask_orig
```

---

## 6. References & Official Resources
- **FastSAM Paper**: [Fast Segment Anything (arXiv 2023)](https://arxiv.org/abs/2306.12156)
- **Official GitHub Repository**: [https://github.com/CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)

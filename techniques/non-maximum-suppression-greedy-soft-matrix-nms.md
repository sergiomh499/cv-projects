---
title: "Non-Maximum Suppression (NMS): Greedy, Soft-NMS & Parallel Matrix NMS"
type: "Technique"
domain: "Object Detection, 2D/3D Bounding Box Pruning & Post-Processing"
tags:
  - technique
  - nms
  - non-maximum-suppression
  - soft-nms
  - matrix-nms
  - object-detection
  - bounding-box-pruning
status: evergreen
updated: 2026-09-09
aliases:
  - "NMS"
  - "Non-Maximum Suppression"
  - "Soft-NMS"
  - "Matrix NMS"
  - "Bounding Box Pruning"
---

# 📦 Non-Maximum Suppression (NMS): Greedy, Soft-NMS & Parallel Matrix NMS

## 1. High-Level Concept & The Redundant Bounding Box Challenge

In dense anchor-based and anchor-free object detectors (e.g., [[architectures/real-time-detectors-and-segmenters/yolov8|YOLOv8]] through [[architectures/real-time-detectors-and-segmenters/yolo26|YOLO26]], [[architectures/3d-pointclouds-and-lidar/centerpoint|CenterPoint]], RetinaNet), an image is evaluated across dense feature pyramid grids.
Because neighboring spatial grid cells share overlapping receptive fields, a single physical target (e.g., a pedestrian or vehicle) produces **dozens or hundreds of redundant candidate bounding boxes** with varying confidence scores.

The objective of **Non-Maximum Suppression (NMS)** is to prune duplicate proposals and retain exactly **one optimal bounding box per physical ground-truth object**:

$$
\mathcal{B}_{\text{raw}} \in \mathbb{R}^{N \times 6} \; (x_1, y_1, x_2, y_2, \text{score}, \text{class}) \xrightarrow{\text{NMS}} \mathcal{B}_{\text{filtered}} \in \mathbb{R}^{K \times 6}
$$

---

### The Evolution: Greedy NMS $\to$ Soft-NMS $\to$ Matrix NMS

#### 1. Classical Greedy NMS (Hard NMS)
1. Sorts all candidate boxes descending by classification score: $\mathcal{B} = \{b_1, \dots, b_N\}$.
2. Pops the top-scoring box $M = b_1$ and adds it to the final output list $\mathcal{D}$.
3. Computes Intersection-over-Union (IoU) between $M$ and every remaining box $b_i$. If $\text{IoU}(M, b_i) \ge N_t$ (e.g., $N_t = 0.65$), $b_i$ is discarded.
4. Repeats sequentially until no boxes remain.

*The Severe Flaws of Hard NMS*:
- **The Occlusion Failure**: In crowded scenes (e.g., dense crowds, parked vehicles), two distinct objects genuinely overlap with $\text{IoU} > 0.65$. Hard NMS eliminates the second object, causing high false-negative rates!
- **Sequential Latency Bottleneck**: Because the removal of $b_i$ depends sequentially on prior selected boxes, standard NMS cannot be fully vectorized on GPUs, causing severe thread divergence.

---

#### 2. Soft-NMS (Bodla et al., ICCV 2017)
Instead of setting a neighboring box's score to 0 when $\text{IoU} \ge N_t$, Soft-NMS **continuously decays its score** as a monotonic function of IoU:
- **Gaussian Continuous Decay**:
  $$s_i = s_i \cdot \exp\left( -\frac{\text{IoU}(M, b_i)^2}{\sigma} \right)$$
- If a neighboring box represents a true occluded object, its score is slightly reduced rather than discarded. True targets remain above the detection threshold, boosting Average Precision by **$+1.5\text{ to }+2.5\text{ mAP}$**.

---

#### 3. Parallel Matrix NMS (Wang et al., SOLOv2)
*The Breakthrough*: Eliminates sequential loops entirely!
- Evaluates the complete **$N \times N$ pairwise IoU matrix** in parallel on GPU Tensor Cores in a single CUDA kernel dispatch.
- Evaluates the probability that candidate box $j$ is a duplicate of a higher-scoring box $i$:
  $$f(\text{IoU}_{ij}) = \min_{k: s_k > s_j} \frac{1 - \text{IoU}_{ij}}{1 - \text{IoU}_{ik}}$$
- Runs in strictly **$<0.5\text{ ms}$** regardless of object density, unlocking predictable, real-time post-processing on edge devices.

```
Comparison of NMS Execution Models:

Greedy Hard-NMS (Sequential & Destructive):
   [Sort Boxes] ---> [Select Max Box M] ---> [IoU >= 0.65 ?] --Yes--> [DISCARD BOX!]
                            ^                         |
                            |                         No
                            +-------------------------+ (Sequential loop: slow on GPU!)

Soft-NMS (Continuous Gaussian Decay):
   [Select Max Box M] ---> Decay score: s_i = s_i * exp(-IoU^2 / sigma) ---> Keeps occluded targets!

Matrix NMS (Single-Pass Fully Parallel Tensor GEMM):
   All N Boxes =====> [ Compute Full N x N IoU Matrix in 1 GPU Step ]
                                       |
                                       v
                      [ Elementwise Decay Matrix Multiplication ]
                                       |
                                       v
                      Exact Pruned Output in <0.5 milliseconds!
```

---

## 2. Mathematical Formulation

### 2.1 The Soft-NMS Formulations
Let $M$ be the current maximum-scoring bounding box and $b_i$ be an overlapping candidate box.

#### Linear Decay
$$
s_i = \begin{cases} s_i, & \text{IoU}(M, b_i) < N_t \\ s_i \left( 1 - \text{IoU}(M, b_i) \right), & \text{IoU}(M, b_i) \ge N_t \end{cases}
$$

#### Continuous Gaussian Decay
$$
s_i = s_i \cdot \exp\left( -\frac{\text{IoU}(M, b_i)^2}{\sigma} \right), \quad \forall b_i \notin \mathcal{D}
$$

The Gaussian formulation introduces no hard threshold boundaries, producing smooth and continuous score distributions across all overlap levels (typically $\sigma = 0.5$).

---

### 2.2 Matrix NMS Parallel Derivation
Let $\mathbf{S} \in \mathbb{R}^N$ be the sorted classification scores ($s_1 \ge s_2 \ge \dots \ge s_N$) and $\mathbf{M} \in \mathbb{R}^{N \times N}$ be the pairwise IoU matrix ($M_{ij} = \text{IoU}(b_i, b_j)$).
Let the upper-triangular matrix representing higher-scoring pairs be:

$$
\mathbf{U}_{ij} = \begin{cases} M_{ij}, & \text{if } i < j \\ 0, & \text{otherwise} \end{cases}
$$

The probability that box $j$ is suppressed by any higher-scoring box $i$ is evaluated as:

$$
\text{decay}_j = \min_{i < j} \left( \exp\left( -\frac{M_{ij}^2}{\sigma} \right) \right) \quad \text{or} \quad \text{decay}_j = \min_{i < j} \left( 1 - M_{ij} \right)
$$

The decayed score vector is computed via column-wise matrix reduction in parallel:

$$
\mathbf{S}_{\text{decayed}} = \mathbf{S} \odot \min_{\text{row}} \left( \mathbf{F}(\mathbf{U}) \right)
$$

---

## 3. Python Reference Implementation

```python
import numpy as np
import torch

def box_iou_matrix(boxes: torch.Tensor) -> torch.Tensor:
    """Computes pairwise IoU matrix for N boxes [N, 4] (x1, y1, x2, y2)."""
    n = boxes.shape[0]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    lt = torch.max(boxes[:, None, :2], boxes[None, :, :2]) # [N, N, 2]
    rb = torch.min(boxes[:, None, 2:], boxes[None, :, 2:]) # [N, N, 2]
    
    wh = (rb - lt).clamp(min=0) # [N, N, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area[:, None] + area[None, :] - inter
    return inter / (union + 1e-7)


def matrix_nms(boxes: torch.Tensor, scores: torch.Tensor, 
               sigma: float = 0.5, score_thresh: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parallel Matrix NMS in a single GPU pass without sequential while-loops.
    boxes: [N, 4] bounding boxes
    scores: [N] confidence scores
    Returns: (filtered_boxes, filtered_scores)
    """
    # 1. Sort descending by score
    scores, sort_idx = torch.sort(scores, descending=True)
    boxes = boxes[sort_idx]
    
    # 2. Pairwise IoU matrix: [N, N]
    ious = box_iou_matrix(boxes)
    
    # 3. Extract upper triangular part (where score_i > score_j)
    ious = torch.triu(ious, diagonal=1)
    
    # 4. Column-wise maximum IoU: [N]
    max_ious, _ = torch.max(ious, dim=0)
    
    # Matrix decay formulation: decay = exp(-ious^2 / sigma) / exp(-max_ious^2 / sigma)
    decay_matrix = torch.exp(-ious.pow(2) / sigma)
    # Compensate for false suppression
    decay, _ = torch.min(decay_matrix, dim=0)
    
    decayed_scores = scores * decay
    
    # Filter by threshold
    keep = decayed_scores >= score_thresh
    return boxes[keep], decayed_scores[keep]


def greedy_nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.65) -> list[int]:
    """Standard CPU sequential Greedy Hard NMS."""
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
        
    return keep
```

---

## 4. Models in the Vault Utilizing NMS

- **[[architectures/real-time-detectors-and-segmenters/yolov8|YOLOv8]] through [[architectures/real-time-detectors-and-segmenters/yolo11|YOLO11]]**: Primary output post-processing anchor filter.
- **[[architectures/real-time-detectors-and-segmenters/fastsam|FastSAM]]**: Mask-based instance segmentation proposal suppression.
- **[[architectures/3d-pointclouds-and-lidar/centerpoint|CenterPoint]]**: 3D oriented bounding box rotated NMS.
- **[[topics/object-detection/README|Object Detection Playbook]]**: Cataloged as the primary detection post-processor.

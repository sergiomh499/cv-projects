---
title: "Two-Stage Association & Kalman Filtering (ByteTrack): Robust MOT Under Occlusion"
type: "Technique"
domain: "Video Tracking & Multi-Object Tracking (MOT)"
tags:
  - technique
  - bytetrack
  - multi-object-tracking
  - mot
  - kalman-filter
  - two-stage-association
  - hungarian-matching
status: evergreen
updated: 2026-09-09
aliases:
  - "ByteTrack"
  - "Two-Stage Association"
  - "Kalman Filter Tracking"
  - "SORT"
  - "BoT-SORT"
---

# 🎯 Two-Stage Association & Kalman Filtering (ByteTrack): Robust MOT Under Occlusion

## 1. High-Level Concept & The False-Negative Discard Dilemma

In Multi-Object Tracking (MOT) for robotics, traffic surveillance, and autonomous driving (e.g., [[topics/video-tracking/README|Video Tracking]], BoT-SORT, [[architectures/visual-tracking-and-flow/cotracker|CoTracker]], [[architectures/visual-tracking-and-flow/tapir|TAPIR]]), a tracking system must maintain stable object identities across continuous video frames despite severe camera motion, heavy occlusion, motion blur, and lighting changes.

### The Classical MOT Dilemma (SORT, DeepSORT, JDE)
Traditional tracking-by-detection frameworks operate with a single high detection threshold (typically $\tau_{\text{high}} \approx 0.60$):
- **The Motivation**: High thresholds filter out background false positives (clutter, shadows, image noise).
- **The Catastrophic Flaw**: When an object is partially occluded, blurred by rapid motion, or enters a shadow, its detector confidence drops sharply into the range $[0.10, 0.55]$.
- Discarding all detections below $\tau_{\text{high}}$ immediately drops occluded objects, breaking track continuity, causing severe **Track Fragmentation**, and dramatically inflating **ID Switches (IDSW)**.

### The ByteTrack Solution (Zhang et al., ECCV 2022)
**ByteTrack** breaks this trade-off by partitioning detections into two confidence tiers:
1. **High-Score Detections** $\mathcal{D}_{\text{high}} = \{ d \mid \text{score}(d) \ge \tau_{\text{high}} \}$ (e.g., $\ge 0.60$).
2. **Low-Score Detections** $\mathcal{D}_{\text{low}} = \{ d \mid \tau_{\text{low}} \le \text{score}(d) < \tau_{\text{high}} \}$ (e.g., $0.10 \le \text{score} < 0.60$).

#### The Two-Stage Association Algorithm
- **Stage 1 (Primary Association)**: High-score detections $\mathcal{D}_{\text{high}}$ are matched with predicted track positions using the Hungarian algorithm based on Intersection-over-Union (IoU) or Re-ID feature distance.
- **Stage 2 (Recovery Association)**: Unmatched tracklets from Stage 1 are matched against the **low-score detections** $\mathcal{D}_{\text{low}}$!
  - Because low-score detections contain genuine target objects undergoing occlusion, this step recovers occluded tracks.
  - Crucially: **Unmatched low-score detections are immediately discarded** and *never* allowed to initialize new tracks, preventing false-positive background clutter from polluting the tracker!
- Operates at **$>100\text{ FPS}$** on edge CPUs without requiring deep neural Re-ID feature extractors.

```
ByteTrack Two-Stage Association Flow:

Raw Detections from Frame t
           |
           +---------------------------------+
           |                                 |
           v (score >= 0.60)                 v (0.10 <= score < 0.60)
[ High-Score Detections D_high ]   [ Low-Score Detections D_low ]
           |                                 |
           v                                 |
[ Stage 1: Match D_high with Tracks ]        |
     |                     |                 |
     | (Matched)           | (Unmatched)     |
     v                     v                 v
[ Update Tracks ]    [ Stage 2: Match Unmatched Tracks with D_low ]
                           |                           |
                           v (Matched)                 v (Unmatched)
                     [ Recover Track! ]          [ Mark Track Lost ]
                                                 (Discard remaining D_low)
```

---

## 2. Mathematical Formulation

### 2.1 The 8D Kalman Filter State Dynamics
The tracker models each object's 2D bounding box motion using a discrete linear constant-velocity Kalman filter. The 8-dimensional state vector $\mathbf{x} \in \mathbb{R}^8$ is:

$$
\mathbf{x} = \left[ x, \; y, \; a, \; h, \; \dot{x}, \; \dot{y}, \; \dot{a}, \; \dot{h} \right]^T
$$

where:
- $(x, y)$ is the 2D bounding box center coordinate.
- $a = w / h$ is the bounding box aspect ratio.
- $h$ is the bounding box height.
- $(\dot{x}, \dot{y}, \dot{a}, \dot{h})$ are the respective instantaneous velocities.

#### State Transition Model
Between frames separated by time step $\Delta t$:

$$
\mathbf{x}_{k|k-1} = \mathbf{F} \mathbf{x}_{k-1|k-1}, \quad \mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}
$$

The transition matrix $\mathbf{F} \in \mathbb{R}^{8 \times 8}$ is:

$$
\mathbf{F} = \begin{bmatrix} \mathbf{I}_{4 \times 4} & \Delta t \mathbf{I}_{4 \times 4} \\ \mathbf{0}_{4 \times 4} & \mathbf{I}_{4 \times 4} \end{bmatrix}
$$

#### Measurement Model
A detected bounding box provides direct observations of the first 4 parameters: $\mathbf{z}_k = [x_d, y_d, a_d, h_d]^T \in \mathbb{R}^4$.

$$
\mathbf{H} = \begin{bmatrix} \mathbf{I}_{4 \times 4} & \mathbf{0}_{4 \times 4} \end{bmatrix} \in \mathbb{R}^{4 \times 8}
$$

The measurement update is evaluated as:

$$
\mathbf{y}_k = \mathbf{z}_k - \mathbf{H} \mathbf{x}_{k|k-1}
$$

$$
\mathbf{S}_k = \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R}
$$

$$
\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^T \mathbf{S}_k^{-1}
$$

$$
\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + \mathbf{K}_k \mathbf{y}_k, \quad \mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1}
$$

---

### 2.2 Cost Matrix & Optimal Bipartite Association
Let the active predicted tracks be $\mathcal{T} = \{ t_1, \dots, t_M \}$ and the candidate detections be $\mathcal{D} = \{ d_1, \dots, d_N \}$.

The assignment cost matrix $\mathbf{C} \in \mathbb{R}^{M \times N}$ is computed based on negative Intersection-over-Union (IoU):

$$
C_{ij} = 1 - \text{IoU}\left( \text{box}(t_i), \; \text{box}(d_j) \right)
$$

The optimal matching assignment is solved globally in polynomial time $\mathcal{O}(M^2 N)$ via the **[[techniques/bipartite-matching-and-hungarian-assigner|Kuhn-Munkres Hungarian Algorithm]]**:

$$
\min_{\mathbf{X}} \sum_{i=1}^M \sum_{j=1}^N C_{ij} X_{ij} \quad \text{s.t.} \quad \sum_{j} X_{ij} \le 1, \; \sum_{i} X_{ij} \le 1, \; X_{ij} \in \{0, 1\}
$$

Matches with cost exceeding threshold $\theta_{\text{IoU}} = 0.5$ ($C_{ij} > 0.5$) are rejected.

---

## 3. Python Reference Implementation

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes IoU between two sets of boxes [N, 4] and [M, 4] (x1, y1, x2, y2)."""
    n, m = boxes_a.shape[0], boxes_b.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m))
        
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    
    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    union_area = area_a[:, None] + area_b[None, :] - inter_area
    return inter_area / (union_area + 1e-7)

def linear_assignment(cost_matrix: np.ndarray, thresh: float = 0.5):
    """Solves Hungarian assignment with gating threshold."""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))
    
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            unmatched_rows.discard(r)
            unmatched_cols.discard(c)
            
    return np.array(matches), list(unmatched_rows), list(unmatched_cols)

def bytetrack_associate(track_boxes: np.ndarray, det_boxes: np.ndarray, 
                        det_scores: np.ndarray, high_thresh: float = 0.6, 
                        low_thresh: float = 0.1, iou_thresh: float = 0.5):
    """
    ByteTrack Two-Stage Association Algorithm.
    Returns: (matched_tracks_to_dets, unmatched_track_indices, unmatched_det_indices)
    """
    # Partition detections
    high_mask = det_scores >= high_thresh
    low_mask = (det_scores >= low_thresh) & (det_scores < high_thresh)
    
    idx_high = np.where(high_mask)[0]
    idx_low = np.where(low_mask)[0]
    
    boxes_high = det_boxes[idx_high]
    boxes_low = det_boxes[idx_low]
    
    # 1. Stage 1: Associate High-Score Detections with Tracks
    cost_stage1 = 1.0 - box_iou(track_boxes, boxes_high)
    matches_1, unmatched_tracks_1, unmatched_dets_1 = linear_assignment(cost_stage1, iou_thresh)
    
    matched_results = []
    for r, c in matches_1:
        matched_results.append((r, idx_high[c]))
        
    # 2. Stage 2: Associate Unmatched Tracks with Low-Score Detections
    if len(unmatched_tracks_1) > 0 and len(boxes_low) > 0:
        track_boxes_remain = track_boxes[unmatched_tracks_1]
        cost_stage2 = 1.0 - box_iou(track_boxes_remain, boxes_low)
        matches_2, unmatched_tracks_2, _ = linear_assignment(cost_stage2, iou_thresh)
        
        for r, c in matches_2:
            matched_results.append((unmatched_tracks_1[r], idx_low[c]))
            
        final_unmatched_tracks = [unmatched_tracks_1[r] for r in unmatched_tracks_2]
    else:
        final_unmatched_tracks = unmatched_tracks_1
        
    final_unmatched_dets = [idx_high[c] for c in unmatched_dets_1]
    return matched_results, final_unmatched_tracks, final_unmatched_dets
```

---

## 4. Models in the Vault Utilizing Two-Stage Association

- **[[topics/video-tracking/README|Video Tracking Playbook]]**: Primary multi-object tracking algorithm.
- **[[cookbooks/12-bytetrack-two-stage-association/README|Cookbook 12: ByteTrack Association]]**: Production runnable implementation.
- **[[architectures/visual-tracking-and-flow/cotracker|CoTracker]] & [[architectures/visual-tracking-and-flow/tapir|TAPIR]]**: Dense persistent point and object trajectory tracking.
- **[[architectures/3d-pointclouds-and-lidar/centerpoint|CenterPoint]]**: Center-based 3D bounding box Kalman velocity tracking.

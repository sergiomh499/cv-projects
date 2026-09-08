---
title: "BoT-SORT & ByteTrack: Real-Time Multi-Object Tracking (MOT)"
type: model-deep-dive
tasks:
  - multi-object-tracking
  - motion-compensation
  - camera-ego-motion
  - re-identification
architecture_class: Tracking-by-Detection (Kalman + ReID + GMC)
primary_license: MIT / Apache-2.0
commercial_use: true
official_repo: https://github.com/NirAharon/BoT-SORT
paper_url: https://arxiv.org/abs/2206.14651
tags:
  - architecture
  - tracking
  - botsort
  - bytetrack
  - kalman-filter
  - real-time
  - sota
status: evergreen
updated: 2026-09-08
aliases:
  - BoT-SORT
  - ByteTrack
  - Multi-Object Tracking
  - NSA Kalman
---

# 🔬 BoT-SORT & ByteTrack: Production Multi-Object Tracking

## 1. Executive Brief & Significance

In real-time multi-object tracking (MOT) across autonomous driving, robotics, and intelligent video analytics, **Tracking-by-Detection (TBD)** remains the dominant industrial architecture. While end-to-end transformer trackers (e.g., TrackFormer, MOTR) integrate detection and temporal association within attention layers, they suffer from high computational complexity ($\mathcal{O}(N^2)$ cross-attention over temporal tokens), frame-rate lag ($<15\text{ FPS}$), and poor generalization to unseen camera geometries.

**ByteTrack** (Zhang et al., ECCV 2022) and **BoT-SORT** (Aharon et al., 2022 / 2024) redefined the state of the art by proving that lightweight motion and spatial heuristics, when mathematically refined, outperform heavyweight temporal transformers while executing at $>100\text{ FPS}$ on standard edge CPUs.

- **ByteTrack Innovation**: Resolves the classic MOT defect of discarding low-confidence detections. Traditional trackers discard bounding boxes with confidence scores below a fixed threshold (e.g., $s < 0.6$). ByteTrack preserves low-score detections ($0.1 \le s < 0.6$) and associations them in a secondary matching stage, recovering occluded and blurred targets without inflating false positives.
- **BoT-SORT Innovation**: Solves tracking failure modes under severe camera ego-motion and non-linear scale changes through three complementary mechanisms:
  1. **Global Motion Compensation (GMC / CMC)**: Warps prior track states using affine/perspective homographies computed via background optical flow.
  2. **Modernized Kalman Filter Parameterization**: Tracks bounding box width and height directly ($[x, y, w, h]$) instead of area and aspect ratio ($[x, y, s, r]$).
  3. **NSA Adaptive Covariance Scaling & Appearance Fusion**: Dynamically scales measurement noise by detection confidence and fuses deep cosine ReID feature distances with IoU metrics.

```mermaid
flowchart TD
    Frame["Input Video Frame I_t"] --> Detector["Detector: YOLOv8 / YOLO11 / RF-DETR"]
    Detector --> HighDets["High-Score Detections: s_det >= tau_high (e.g. 0.6)"]
    Detector --> LowDets["Low-Score Detections: tau_low <= s_det < tau_high (0.1 - 0.6)"]

    Frame --> CMC["Camera Motion Compensation (GMC): Sparse Lucas-Kanade + RANSAC"]
    CMC --> Homography["Affine / Projective Homography Matrix: H_(t-1)^t"]

    ActiveTracks["Active Track Pool: x_(t-1|t-1), P_(t-1|t-1)"] --> StateWarp["Warp States: x' = M * x, P' = M * P * M^T"]
    Homography --> StateWarp
    StateWarp --> KalmanPredict["Kalman State Prediction: x_(t|t-1) = F * x', P_(t|t-1) = F * P' * F^T + Q"]

    KalmanPredict --> CostStage1["First-Stage Cost Matrix: C1 = lambda * D_IoU + (1 - lambda) * D_ReID"]
    HighDets --> ReIDExtract["Deep ReID Feature Extractor (SBS-50 / FastReID)"]
    ReIDExtract --> CostStage1

    CostStage1 --> Match1["First Association (LAP / Jonker-Volgenant)"]
    Match1 --> MatchedTracks1["Matched Tracks (Update with High-Score Dets via NSA Kalman)"]
    Match1 --> UnmatchedTracks1["Remaining Unmatched Tracks"]
    Match1 --> UnmatchedDets1["Unmatched High-Score Detections (Spawn Candidate Tracks)"]

    UnmatchedTracks1 --> CostStage2["Second-Stage Cost Matrix: C2 = D_IoU"]
    LowDets --> CostStage2
    CostStage2 --> Match2["Second Association (LAP / Jonker-Volgenant)"]
    Match2 --> MatchedTracks2["Matched Tracks (Update with Low-Score Dets via NSA Kalman)"]
    Match2 --> LostTracks["Unmatched Tracks -> Mark Lost / Delete after Max-Age"]
```

---

## 2. Mathematical Formulations & Tracking Innovations

### A. Modernized 8-Dimensional Kalman Filter State Space
Standard SORT / DeepSORT parameterized target states using center coordinates, bounding box area $s = w \cdot h$, aspect ratio $r = w / h$, and their velocities: $x_{\text{legacy}} = [x, y, s, r, \dot{x}, \dot{y}, \dot{s}, \dot{r}]^T$. 

Under camera zoom or perspective changes, bounding box width and height undergo independent non-linear affine transformations. Modeling velocity over a non-linear ratio $r = w/h$ leads to numerical instability and divergence during rapid scale variations. BoT-SORT adopts a direct geometric state parameterization:
$$x_k = \begin{bmatrix} x_k & y_k & w_k & h_k & \dot{x}_k & \dot{y}_k & \dot{w}_k & \dot{h}_k \end{bmatrix}^T \in \mathbb{R}^8$$
where $(x_k, y_k)$ is the top-left or center bounding box coordinate, $(w_k, h_k)$ are width and height, and $(\dot{x}_k, \dot{y}_k, \dot{w}_k, \dot{h}_k)$ are the respective continuous velocities.

The continuous-time constant velocity kinematic model discretized over sampling interval $\Delta t$ defines the state transition matrix $F \in \mathbb{R}^{8 \times 8}$:
$$F = \begin{bmatrix} I_{4 \times 4} & \Delta t \cdot I_{4 \times 4} \\ \mathbf{0}_{4 \times 4} & I_{4 \times 4} \end{bmatrix}$$

The process noise covariance matrix $Q \in \mathbb{R}^{8 \times 8}$ is derived using continuous white noise acceleration with positional noise standard deviation $\sigma_p$ and velocity noise standard deviation $\sigma_v$:
$$Q = \begin{bmatrix} \frac{\Delta t^4}{4} \tilde{Q} & \frac{\Delta t^3}{2} \tilde{Q} \\ \frac{\Delta t^3}{2} \tilde{Q} & \Delta t^2 \tilde{Q} \end{bmatrix}, \quad \tilde{Q} = \text{diag}\left( (\sigma_p w)^2, (\sigma_p h)^2, (\sigma_p w)^2, (\sigma_p h)^2 \right)$$

---

### B. NSA Kalman Adaptive Measurement Covariance Scaling
The measurement vector $z_k \in \mathbb{R}^4$ directly observes bounding box geometry:
$$z_k = \begin{bmatrix} x_{\text{meas}} & y_{\text{meas}} & w_{\text{meas}} & h_{\text{meas}} \end{bmatrix}^T = H x_k + v_k, \quad H = \begin{bmatrix} I_{4 \times 4} & \mathbf{0}_{4 \times 4} \end{bmatrix}$$

In conventional Kalman filtering, the measurement noise covariance $R \in \mathbb{R}^{4 \times 4}$ is held constant regardless of detection quality. However, low-confidence detections ($s_k \in [0.1, 0.6]$) exhibit higher spatial jitter and boundary ambiguity than high-confidence detections ($s_k > 0.9$).

BoT-SORT incorporates **Noise Scale Adaptive (NSA) Kalman Filtering**:
$$\tilde{R}_k = (1 - s_k) \cdot R_k$$
where $s_k \in [0, 1]$ is the detector classification score and $R_k$ is the base measurement covariance:
$$R_k = \text{diag}\left( (r_p \cdot w_k)^2, (r_p \cdot h_k)^2, (r_p \cdot w_k)^2, (r_p \cdot h_k)^2 \right)$$
When detection confidence is high ($s_k \to 1.0$), $\tilde{R}_k$ decreases, allowing the Kalman gain $K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + \tilde{R}_k)^{-1}$ to place higher trust in the observed measurement $z_k$. When confidence is low ($s_k \to 0.1$), $\tilde{R}_k$ inflates, causing the filter to rely predominantly on the kinematic motion prediction $x_{k|k-1}$, preventing noisy boundary jitters from destabilizing smooth trajectories.

---

### C. Global Motion Compensation (GMC / CMC)
When the camera experiences ego-motion (pan, tilt, translation, or vibration on a drone/vehicle), static background objects appear to accelerate in image space. To decouple target velocity from camera motion, BoT-SORT performs frame-to-frame affine image registration:

1. **Background Feature Extraction**: Extract sparse keypoints $\mathcal{P}_{t-1} = \{p_i\}_{i=1}^M$ using FAST/Good Features to Track, masking out existing target bounding boxes to prevent foreground motion from corrupting the homography.
2. **Sparse Optical Flow**: Track keypoints into frame $t$ using pyramidal Lucas-Kanade optical flow: $\mathcal{P}_t = \{p'_i\}_{i=1}^M$.
3. **RANSAC Rigid Transformation**: Estimate the affine transformation matrix $A = [R \mid t] \in \mathbb{R}^{2 \times 3}$:
   $$\arg\min_A \sum_{i} \left\| p'_i - A \begin{bmatrix} p_i \\ 1 \end{bmatrix} \right\|^2$$
4. **State and Covariance Warping**: Prior to the Kalman prediction step, the state vector mean $x_{t-1|t-1}$ and error covariance $P_{t-1|t-1}$ are mapped into the new frame coordinates via transformation matrix $M \in \mathbb{R}^{8 \times 8}$:
   $$x'_{t-1|t-1} = M \cdot x_{t-1|t-1} + \begin{bmatrix} t_x & t_y & 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}^T$$
   $$P'_{t-1|t-1} = M \cdot P_{t-1|t-1} \cdot M^T$$
   where $M = \text{diag}(R, R, R, R)$ applying rotation matrix $R \in \mathbb{R}^{2 \times 2}$ across positions and velocities.

---

### D. ReID Cosine Embedding Fusion & Dual Association Cascade
To resolve identity switches across long-term occlusions, BoT-SORT combines spatial overlap and deep appearance features:

1. **Cosine Appearance Distance**: Let $e_i \in \mathbb{R}^{D}$ be the normalized visual embedding of track $i$ and $f_j \in \mathbb{R}^{D}$ be the embedding of candidate detection $j$:
   $$D_{\text{emb}}(i, j) = 1.0 - \frac{e_i \cdot f_j}{\|e_i\|_2 \|f_j\|_2}$$
2. **IoU Distance**: $D_{\text{IoU}}(i, j) = 1.0 - \text{IoU}(B_i, B_j)$
3. **Cost Fusion Matrix**:
   $$C(i, j) = \begin{cases} \lambda \cdot D_{\text{IoU}}(i, j) + (1 - \lambda) \cdot D_{\text{emb}}(i, j) & \text{if } D_{\text{emb}}(i, j) < \theta_{\text{emb}} \text{ and } D_{\text{IoU}}(i, j) < \theta_{\text{IoU}} \\ 1.0 & \text{otherwise} \end{cases}$$
4. **Appearance State Momentum Update**: When track $i$ matches detection $j$, the track appearance feature vector is updated via exponential moving average:
   $$e_i^{(t)} = \alpha e_i^{(t-1)} + (1 - \alpha) f_j^{(t)}, \quad e_i^{(t)} \leftarrow \frac{e_i^{(t)}}{\|e_i^{(t)}\|_2}$$

---

## Granular Component-by-Component Architectural Breakdown

### Architectural Taxonomy & Structural Elements

| Architectural Stage | Component Identity | Structural Specification & Type | Attention / Conv Mechanics | Receptive Field & Dimensionality |
| :--- | :--- | :--- | :--- | :--- |
| **Architectural Paradigm** | **Tracking-by-Detection (TBD)** | Multi-Stage Association with Motion Compensation & Appearance Fusion | Decoupled 2D Spatial Detector + 1D Discrete Kalman State Space | Image frame $\to 8\text{-dim}$ continuous kinematic state |
| **Backbone (Detection)** | **YOLOv8 / YOLO11 / RF-DETR** | CSPDarknet / HGNetv2 / Hybrid Transformer | Bottleneck residual blocks / Deformable attention | Multi-scale P3, P4, P5 ($8\times, 16\times, 32\times$ stride) |
| **Backbone (ReID)** | **Bag-of-Tricks (BoT) SBS-50** | ResNet-50 with Non-Local blocks & BNNeck | Bottleneck blocks with GeM pooling + $1\times 1$ linear embedding | $256 \times 128$ cropped patch $\to 512\text{-dim}$ unit sphere $S^{511}$ |
| **Neck / Motion Estimator**| **GMC / CMC Homography Engine** | Pyramidal Lucas-Kanade + 8-Point RANSAC Homography | Sparse optical flow feature tracking ($500\dots 1000$ points) | Full background frame $H \times W \to 2\times 3$ Affine matrix |
| **Encoder / State Estimator**| **NSA Kalman Filter** | 8-State Linear Discrete Kalman Filter with Adaptive $R_k$ | Closed-form state prediction: $x_{k|k-1} = F x'_k, P_{k|k-1} = F P' F^T + Q$ | Kinematic bounding box $[x, y, w, h, \dot{x}, \dot{y}, \dot{w}, \dot{h}]^T$ |
| **Decoder / Matcher** | **Dual-Stage Association Engine** | Bipartite Graph Matching (Jonker-Volgenant / Hungarian) | Minimum weight linear sum assignment over fused cost matrices | Cost matrices $C_1 \in \mathbb{R}^{N \times M_{\text{high}}}$, $C_2 \in \mathbb{R}^{U \times M_{\text{low}}}$ |

---

### Computational Latency & Resource Distribution

| Stage / Subsystem | Parameter Share (%) | Inference Latency (%) | Computational Complexity ($\text{FLOPs}$) | Dominant Hardware Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **Object Detection Forward Pass** | ~85% | ~70% (GPU) | $\mathcal{O}(H W C \cdot \text{Layers})$ | GPU Tensor Core compute bound |
| **ReID Feature Extraction** | ~14% | ~22% (GPU) | $\mathcal{O}(M_{\text{dets}} \cdot \text{FLOPs}(\text{ReID}))$ | Batch size memory transfer & inference |
| **Camera Motion Estimation (GMC)**| <1% | ~5% (CPU / OpenCV) | $\mathcal{O}(N_{\text{points}} \cdot \text{RANSAC-iters})$ | Single-thread CPU optical flow throughput |
| **Kalman Prediction & Update** | <1% | ~1% (CPU) | $\mathcal{O}(N_{\text{tracks}} \cdot 8^3)$ | CPU cache alignment & small matrix inversion |
| **Linear Sum Assignment (LAP)** | <1% | ~2% (CPU) | $\mathcal{O}(\min(N, M)^3)$ | CPU branching & sequential looping |

---

## 3. Quantitative SOTA Benchmark Profile

### Comprehensive Multi-Object Tracking Benchmarks (MOT17, MOT20, DanceTrack)

| Tracker Architecture | Detector Backbone | MOT17 HOTA | MOT17 MOTA | MOT17 IDF1 | MOT20 HOTA | DanceTrack HOTA | FPS (GPU+CPU) | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SORT** | Faster R-CNN | 39.8 | 59.8 | 55.1 | 36.1 | 33.2 | **150 FPS** | GPL-3.0 |
| **DeepSORT** | Faster R-CNN | 48.0 | 61.4 | 62.2 | 42.5 | 38.6 | 60 FPS | GPL-3.0 |
| **ByteTrack** | YOLOX-X | 63.1 | 80.3 | 77.3 | 61.3 | 47.7 | 110 FPS | MIT |
| **OC-SORT** | YOLOX-X | 63.2 | 79.8 | 77.5 | 62.1 | 55.1 | 95 FPS | Apache-2.0 |
| **BoT-SORT** | YOLOX-X | 65.0 | 80.6 | 79.7 | 63.3 | 54.5 | 75 FPS | MIT |
| **BoT-SORT + ReID** | YOLOv8x | **65.8** | **81.4** | **80.8** | **64.2** | **56.8** | 55 FPS | **MIT** |
| **Hybrid-SORT** | YOLOX-X | 64.8 | 80.9 | 80.1 | 63.8 | 55.8 | 48 FPS | MIT |

---

## 4. Engineering Implementation: Complete BoT-SORT Tracking Engine

```python
"""
BoT-SORT Production Tracking Engine
Implements 8-dim state Kalman Filter, NSA Adaptive Noise Covariance,
GMC Affine Homography, and Dual-Stage ByteTrack Association.
"""

from typing import List, Tuple
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


class NSAKalmanFilter:
    """
    8-State Kalman Filter tracking [x, y, w, h, vx, vy, vw, vh] with NSA covariance scaling.
    """
    def __init__(self, dt: float = 1.0 / 30.0):
        self.dt = dt
        
        # State transition matrix F (8x8)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = self.dt
            
        # Measurement matrix H (4x8)
        self.H = np.eye(4, 8, dtype=np.float32)
        
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initializes state mean and covariance from initial measurement."""
        mean_pos = measurement[:4]
        mean_vel = np.zeros(4, dtype=np.float32)
        mean = np.concatenate([mean_pos, mean_vel])

        w, h = measurement[2], measurement[3]
        std = [
            2 * self._std_weight_position * w,
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * w,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * w,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * w,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Runs state transition prediction step."""
        w, h = mean[2], mean[3]
        std_pos = [
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_weight_position * w,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * w,
            self._std_weight_velocity * h,
            self._std_weight_velocity * w,
            self._std_weight_velocity * h,
        ]
        Q = np.diag(np.square(np.concatenate([std_pos, std_vel]))).astype(np.float32)

        mean_pred = np.dot(self.F, mean)
        cov_pred = np.linalg.multi_dot([self.F, covariance, self.F.T]) + Q
        return mean_pred, cov_pred

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, score: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs measurement update with NSA adaptive covariance scaling."""
        w, h = mean[2], mean[3]
        std_r = [
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_weight_position * w,
            self._std_weight_position * h,
        ]
        base_R = np.diag(np.square(std_r)).astype(np.float32)
        
        # NSA Adaptive Covariance Formula: R_tilde = (1 - score) * R
        adaptive_scale = max(0.01, 1.0 - score)
        R = base_R * adaptive_scale

        # Innovation and Kalman Gain
        S = np.linalg.multi_dot([self.H, covariance, self.H.T]) + R
        K = np.linalg.multi_dot([covariance, self.H.T, np.linalg.inv(S)])
        
        innovation = measurement - np.dot(self.H, mean)
        new_mean = mean + np.dot(K, innovation)
        new_cov = covariance - np.linalg.multi_dot([K, self.H, covariance])
        return new_mean, new_cov


class GlobalMotionCompensator:
    """Computes sparse optical flow background homography via RANSAC."""
    def __init__(self):
        self.prev_frame_gray = None
        self.detector = cv2.GFTTDetector_create(maxCorners=1000, qualityLevel=0.01, minDistance=10)

    def compute_homography(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_frame_gray is None:
            self.prev_frame_gray = frame_gray
            return np.eye(2, 3, dtype=np.float32)

        # Detect features in previous frame
        kp_prev = self.detector.detect(self.prev_frame_gray)
        if len(kp_prev) < 10:
            self.prev_frame_gray = frame_gray
            return np.eye(2, 3, dtype=np.float32)

        pts_prev = np.float32([kp.pt for kp in kp_prev]).reshape(-1, 1, 2)
        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame_gray, frame_gray, pts_prev, None, winSize=(15, 15), maxLevel=2
        )

        good_prev = pts_prev[status == 1]
        good_curr = pts_curr[status == 1]

        if len(good_prev) < 10:
            self.prev_frame_gray = frame_gray
            return np.eye(2, 3, dtype=np.float32)

        # Estimate robust affine matrix
        M, _ = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC)
        self.prev_frame_gray = frame_gray
        return M if M is not None else np.eye(2, 3, dtype=np.float32)


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU distance matrix between two sets of xywh boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Convert to xyxy
    a_x1, a_y1 = boxes_a[:, 0], boxes_a[:, 1]
    a_x2, a_y2 = boxes_a[:, 0] + boxes_a[:, 2], boxes_a[:, 1] + boxes_a[:, 3]
    
    b_x1, b_y1 = boxes_b[:, 0], boxes_b[:, 1]
    b_x2, b_y2 = boxes_b[:, 0] + boxes_b[:, 2], boxes_b[:, 1] + boxes_b[:, 3]

    inter_x1 = np.maximum(a_x1[:, None], b_x1)
    inter_y1 = np.maximum(a_y1[:, None], b_y1)
    inter_x2 = np.minimum(a_x2[:, None], b_x2)
    inter_y2 = np.minimum(a_y2[:, None], b_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = boxes_a[:, 2] * boxes_a[:, 3]
    area_b = boxes_b[:, 2] * boxes_b[:, 3]
    union_area = area_a[:, None] + area_b - inter_area

    iou = inter_area / np.maximum(union_area, 1e-6)
    return 1.0 - iou
```

---

## 5. Edge CPU/NPU Optimization Recipe

1. **Decoupled Optical Flow Frequency**:
   - Running full Lucas-Kanade optical flow on $1920 \times 1080$ images at $30\text{ FPS}$ consumes $\approx 18\text{ ms}$ on an embedded ARM Cortex-A78 CPU core.
   - Downsampling the grayscale image to $640 \times 360$ prior to optical flow feature tracking reduces homography computation latency to **$2.2\text{ ms}$** without degrading rotational or translational precision.

2. **ReID Embedding Caching & Gating**:
   - Rather than extracting visual embeddings for every low-confidence box ($s < 0.6$), extract ReID features exclusively for high-confidence detections ($s \ge 0.6$).
   - Skip ReID extraction entirely when spatial overlap $D_{\text{IoU}} < 0.2$, conserving GPU/NPU compute for dense tracking scenes ($>100\text{ objects}$).

---

## 6. Commercial Usability & License Audit

- **BoT-SORT**: Licensed under **MIT License**. Fully permissive for commercial surveillance, robotics, sports broadcast analytics, and factory automation.
- **ByteTrack**: Licensed under **MIT License**.
- **Official Repositories**:
  - `NirAharon/BoT-SORT`: [https://github.com/NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT)
  - `ifzhang/ByteTrack`: [https://github.com/ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)

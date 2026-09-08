---
title: Video Tracking Historical Evolution & Multi-Object Paradigms
tags:
  - computer-vision
  - video-tracking
  - mot
  - deepsort
  - bytetrack
  - cotracker
  - didactic
updated: 2026-09-08
aliases:
  - Tracking Evolution
---

# Video Tracking: Historical Evolution & Association Paradigms

A didactic deep-dive examining the progression of object and point tracking in video: from classical optical flow and correlation filters to Tracking-by-Detection (SORT to BoT-SORT), joint detection-and-embedding networks, and modern long-range transformer point trackers (CoTracker3).

Related notes: [[topics/video-tracking/README|Video Tracking Playbook]], [[topics/real-time-systems/README|Real-Time Systems]].

---

## 1. The Core Paradigm Shift

```mermaid
flowchart TD
    A[Classical Motion & Flow: Lucas-Kanade, KCF 1981-2015] --> B[Tracking-by-Detection: SORT, DeepSORT 2016-2018]
    B --> C[Joint Detection & Embedding: FairMOT, JDE 2019-2021]
    B --> D[Motion-Centric Resurgence: ByteTrack, OC-SORT, BoT-SORT 2021-2024]
    C --> E[Foundation Memory Trackers: SAM 2 Video Engine 2024-2026]
    D --> E
    A --> F[Dense Point Trajectory Transformers: CoTracker, TAPIR 2023-2026]
```

---

## 2. Tracking-by-Detection Architecture & The ByteTrack Breakthrough

The dominant paradigm in Multi-Object Tracking (MOT) decouples spatial detection from temporal association across sequential frames.

```mermaid
flowchart LR
    FrameT[Frame t] --> Detector[Object Detector: Bounding Boxes + Scores]
    PastTracks[Prior Tracks from t-1] --> KalmanPred[Kalman Filter Velocity Prediction]
    Detector --> HighConf[High Confidence Detections: Score > 0.6]
    Detector --> LowConf[Low Confidence Detections: 0.1 < Score < 0.6]
    
    HighConf --> Stage1[Stage 1: IoU + ReID Hungarian Matching]
    KalmanPred --> Stage1
    
    Stage1 --> MatchedTracks[Active Matched Tracks]
    Stage1 --> UnmatchedTracks[Unmatched Tracks]
    
    UnmatchedTracks --> Stage2[Stage 2: Match Unmatched with Low-Score Boxes]
    LowConf --> Stage2
    
    Stage2 --> Recovered[Recovered Occluded Tracks]
    Stage2 --> LostTracks[Lost Tracks Marked for Deletion]
```

### The ByteTrack Innovation (Zhang et al., 2022)
Prior tracking algorithms discarded all bounding box detections below a strict confidence threshold (e.g. $score < 0.5$).
- **The Problem**: When an object becomes partially occluded by a tree, pillar, or another person, its detection confidence naturally dips from $0.9$ down to $0.2$. Discarding these low-score boxes creates fragmented trajectories and frequent **ID switches**.
- **The ByteTrack Solution**: Keep *all* detections.
  1. Match high-confidence boxes first ($score > 0.6$).
  2. For the tracks that failed to match, perform a second Hungarian association pass using the low-confidence detections ($0.1 < score < 0.6$).
  3. This simple two-stage matching recovers true occluded objects while filtering out background false positives (since random background noise rarely exhibits continuous spatial motion alignment).

---

## 3. Camera Motion Compensation (CMC) in BoT-SORT

In moving cameras (drones, autonomous vehicles, bodycams), ego-motion violates the standard constant-velocity assumption of the linear Kalman filter.

```mermaid
flowchart TD
    FramePrev[Frame t-1] --> FeatureMatch[Sparse Feature Extraction & Matching: ORB/GFTT]
    FrameCurr[Frame t] --> FeatureMatch
    FeatureMatch --> RANSAC[RANSAC Homography / Affine Estimator]
    RANSAC --> MatrixM[Camera Motion Matrix M_t]
    PastKalman[Prior Kalman State x_t-1] --> Warp[Affine Coordinate Transformation via M_t]
    MatrixM --> Warp
    Warp --> CorrectedKalman[Compensated Track State for Hungarian Matching]
```
By warping the prior state into the current frame coordinate system before running the Kalman prediction, **BoT-SORT** prevents track divergence during rapid camera panning or vehicle turning.

---

## 4. The Point Tracking Revolution: CoTracker3 and TAPIR

While MOT tracks bounding box centroids, dense point tracking follows exact physical surface coordinates through non-rigid deformations and multiple seconds of complete occlusion.

- **Classical Optical Flow (Farneback, RAFT)**: Computes displacement vectors only between adjacent consecutive frames ($t$ and $t+1$). Errors accumulate rapidly across long sequences, causing catastrophic drift.
- **CoTracker3 (Meta FAIR, 2024–2026)**:
  - Tracks up to 70,000 points simultaneously using sliding-window spatial-temporal Transformers.
  - Allows points to share trajectory context (e.g. if one point on an arm is occluded, surrounding visible points on the forearm inform the attention layers of the occluded point's physical location).
  - Explicitly predicts a continuous **visibility flag** $v \in [0, 1]$ alongside coordinate offsets $(\Delta x, \Delta y)$.

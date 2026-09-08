---
title: Video Tracking Playbook
tags:
  - computer-vision
  - video-tracking
  - mot
  - bytetrack
  - cotracker
  - real-time
updated: 2026-09-08
aliases:
  - Video Tracking
---

# Video Tracking Playbook

# Overview
Video Tracking maintains consistent object identity across sequential video frames despite camera motion, object deformation, lighting fluctuations, and severe occlusion. Tracking spans Multiple Object Tracking (MOT - tracking all detected instances), Single Object Tracking (SOT - following an arbitrary bounding-box prompt), and Point Tracking (dense or sparse physical surface point trajectory tracking).

Related notes: [[topics/object-detection/README|Object Detection]], [[topics/object-segmentation/README|Object Segmentation]], [[topics/real-time-systems/README|Real-Time Systems]].

## SOTA & Research
- **Recent Breakthroughs (2023–2026)**:
  - *CoTracker: It is Better to Track Together: Dense Point Tracking by Jointly Modeling Points and Frames* (Karaev et al., Meta FAIR, 2023 / 2024) - [arXiv:2307.07635](https://arxiv.org/abs/2307.07635): SOTA transformer tracking up to 70,000 points jointly through long occlusions and extreme deformations.
  - *OC-SORT: Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking* (Cao et al., 2023) - [arXiv:2203.14360](https://arxiv.org/abs/2203.14360): Fixed linear Kalman filter momentum accumulation errors during non-linear occlusion intervals.
  - *BoT-SORT: Robust Associations Multi-Pedestrian Tracker* (Aharon et al., 2022 / 2023) - [arXiv:2206.14651](https://arxiv.org/abs/2206.14651): Combined Camera Motion Compensation (CMC), optimized Kalman state vectors, and fused appearance ReID embeddings, achieving top ranks on MOT17/MOT20.
  - *TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement* (Doersch et al., Google DeepMind, 2023 / 2024) - [arXiv:2306.08637](https://arxiv.org/abs/2306.08637): Fast, robust point tracking capable of tracking physical surface points with occlusion prediction.

### Quantitative SOTA Benchmark Comparison
| Tracker | Target Benchmark | HOTA | MOTA | IDF1 | Latency / FPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BoT-SORT** | MOT17 Test | 65.0 | 80.6 | 79.5 | 18.0 FPS (GPU + CMC) |
| **ByteTrack** | MOT17 Test | 63.1 | 80.3 | 77.3 | 30.0+ FPS (Lightweight) |
| **OC-SORT** | DanceTrack Test | 55.1 | 89.4 | 54.2 | 28.0 FPS |
| **CoTracker2** | TAP-Vid Kinetics (Point) | 68.2 (AJ) | - | - | 35.0 FPS (512 points) |
| **TAPIR** | TAP-Vid Davis (Point) | 67.3 (AJ) | - | - | 22.0 FPS |

## Architecture Alternatives & Trade-offs
| Tracking Paradigm | Algorithms | Latency / Overhead | Robustness to Occlusion | Ideal Deployment |
| :--- | :--- | :--- | :--- | :--- |
| **Motion-Only Association** | ByteTrack, OC-SORT | <1 ms association overhead | Moderate (breaks under prolonged camera rotation/stops) | High-speed edge tracking, high frame-rate video |
| **Motion + Appearance ReID** | BoT-SORT, DeepSORT | 5-15 ms (ReID feature extraction) | High (recovers identity after lengthy occlusion) | Multi-camera tracking, urban surveillance |
| **Dense Point Trackers** | CoTracker, TAPIR | 30-100 ms | Exceptional physical surface fidelity | Video editing, VFX, deformation analysis |
| **Memory-Prompt Video Trackers** | SAM 2 Video | 25-45 ms | High pixel-accurate mask tracking | Interactive tracking, surgical tool tracking |

## Popular Repos & Integrations
- **[facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker)**: Official PyTorch implementation of CoTracker and CoTracker3.
- **[NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT)**: Official BoT-SORT multi-object tracker with ReID and camera motion compensation.
- **[ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)**: SOTA real-time multi-object tracking implementation.
- **[mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot)**: Pluggable modular tracking suite packaging ByteTrack, BoT-SORT, DeepOCSORT, and StrongSORT in Python.
- **Tooling Integrations**:
  - **FiftyOne**: Visualize video datasets with synchronized track IDs, color-coded trajectory paths, and ID switch inspection.
  - **Rerun**: Stream live trajectories as 2D/3D historical polyline paths with interactive timeline scrubbing.

## End-to-End Pipeline & Workarounds
1. **Pipeline Stages**:
   - Frame ingest -> 2D Detection -> Camera Motion Compensation (Affine/Homography via ORB/GFTT) -> Kalman Filter state prediction -> Cost matrix computation (IoU + ReID distance) -> Two-stage Hungarian assignment -> Track lifecycle management (New, Active, Lost, Removed).
2. **Common Traps & Edge Cases**:
   - *ID Switching*: Passing instances swap identifiers during dense crossings or partial overlap.
   - *Camera Shake / Rapid Panning*: Instantaneous camera movement causes standard linear Kalman filter velocity predictions to diverge drastically.
3. **Engineering Workarounds**:
   - **Camera Motion Compensation (CMC)**: Extract optical flow or sparse feature matches between frame $t-1$ and $t$ to warp the prior state into current camera coordinates.
   - **Two-Stage Matching (ByteTrack principle)**: First match high-confidence detections ($score > 0.6$); for unassigned tracks, match with low-confidence detections ($0.1 < score < 0.6$) instead of discarding them.
   - **Velocity Direction Consistency**: Penalize association candidates that exhibit instantaneous $180^\circ$ direction reversals.

## Deployment & Real-time Notes
- **Decoupled Tracker Threading**:
  - Run heavy neural detection on the GPU worker thread at e.g. 15-30 FPS, while running lightweight Kalman filter state updates and optical-flow interpolation on the CPU thread at 60-120 FPS.
- **ReID Embedding Caching**:
  - Calculate ReID cosine similarities only when IoU overlap is ambiguous ($0.2 < \text{IoU} < 0.7$). Bypassing the ReID network for straightforward associations reduces computational load by over 60%.
- **Zero-Allocation Track Pools**:
  - Pre-allocate track object buffers in memory to avoid dynamic heap allocations on every new track creation and deletion.

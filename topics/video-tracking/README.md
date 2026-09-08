---
title: Video Tracking Playbook
tags:
  - computer-vision
  - video-tracking
  - mot
  - bytetrack
  - botsort
  - cotracker
  - sam2-video
updated: 2026-09-08
aliases:
  - Video Tracking
---

# Video Tracking Playbook

# Overview
Video Tracking maintains consistent identity and spatial trajectories for instances across sequential video frames. The primary paradigms include:
1. **Multiple Object Tracking (MOT)**: Tracks all detected objects (e.g. pedestrians, vehicles) simultaneously using detection-association pipelines.
2. **Single Object Tracking (SOT)**: Follows a specific target prompted by an initial bounding box.
3. **Dense Point Tracking**: Tracks tens of thousands of physical surface points through long video sequences, non-rigid deformation, and occlusions.

Related notes: [[topics/object-detection/README|Object Detection]], [[topics/object-segmentation/README|Object Segmentation]], [[topics/real-time-systems/README|Real-Time Systems]].

---

## SOTA & Research (2023–2026 Breakthroughs)

1. **CoTracker & CoTracker3: Dense Point Tracking by Jointly Modeling Points and Frames** (Karaev et al., Meta FAIR, 2023–2025)
   - *Key Innovation*: Joint sliding-window Transformer architecture modeling cross-point and cross-frame attention simultaneously. Eliminates drifting and tracks up to 70,000 points through prolonged multi-second occlusions.
   - [Paper: arXiv:2307.07635](https://arxiv.org/abs/2307.07635) | [Official Code](https://github.com/facebookresearch/co-tracker)

2. **SAM 2 Video Object Tracking Engine** (Ravi et al., Meta FAIR, 2024 / 2025)
   - *Key Innovation*: Integrates a streaming spatial-temporal memory bank that stores past frame features, object pointers, and user interaction clicks. Re-identifies targets across severe occlusions and camera cuts with zero-shot generalization.
   - [Paper: arXiv:2408.00714](https://arxiv.org/abs/2408.00714) | [Official Code](https://github.com/facebookresearch/sam2)

3. **BoT-SORT & OC-SORT** (Aharon et al., 2023 / Cao et al., 2023)
   - *Key Innovation*: BoT-SORT integrates Camera Motion Compensation (CMC) using image feature matching and combines IoU distance with appearance ReID cosine distances. OC-SORT prevents error accumulation in linear Kalman filters during prolonged non-linear occlusions.
   - [BoT-SORT Paper](https://arxiv.org/abs/2206.14651) | [OC-SORT Paper](https://arxiv.org/abs/2203.14360)

4. **TAPIR: Tracking Any Point with Per-Frame Initialization** (Doersch et al., Google DeepMind, 2023 / 2024)
   - *Key Innovation*: Two-stage point tracker that initializes point estimates independently per frame and refines trajectories over temporal windows, running in real-time.
   - [Paper: arXiv:2306.08637](https://arxiv.org/abs/2306.08637) | [Official Code](https://github.com/google-deepmind/tapnet)

---

### Quantitative SOTA Benchmark Comparison
| Tracker Algorithm | Target Benchmark | HOTA | MOTA | IDF1 | Latency / FPS | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BoT-SORT** | MOT17 Test | 65.0 | 80.6 | 79.5 | 18.0 FPS (GPU + CMC) | MIT |
| **ByteTrack** | MOT17 Test | 63.1 | 80.3 | 77.3 | 35.0+ FPS (Lightweight) | MIT |
| **OC-SORT** | DanceTrack Test | 55.1 | 89.4 | 54.2 | 28.0 FPS | MIT |
| **SAM 2 Video Engine**| SA-V Video Tracking | 75.0 J&F | - | - | 43.8 FPS (Hiera-B+) | Apache-2.0 |
| **CoTracker3** | TAP-Vid Kinetics (Points) | 68.2 (AJ)| - | - | 35.0 FPS (512 points) | Apache-2.0 |
| **TAPIR** | TAP-Vid Davis (Points) | 67.3 (AJ)| - | - | 22.0 FPS | Apache-2.0 |

---

## Commercial Usability & License Audit
- **Permissive (Safe for Commercial Product Integration)**:
  - **MIT License**:
    - `NirAharon/BoT-SORT`: **MIT**. Safe for proprietary commercial integration.
    - `ifzhang/ByteTrack`: **MIT**. Royalty-free, no copyleft obligations.
    - `mikel-brostrom/boxmot`: **GNU GPL-3.0** (Note: BoxMOT is GPL-3.0; use individual ByteTrack/BoT-SORT repos directly if building proprietary closed-source applications to avoid GPL viral requirements).
  - **Apache-2.0**:
    - `facebookresearch/sam2` (Meta Apache-2.0).
    - `facebookresearch/co-tracker` (Apache-2.0).
    - `google-deepmind/tapnet` (Apache-2.0).
  - *Recommendation*: Pair **ByteTrack** or **BoT-SORT** directly (MIT) with **RF-DETR** or **RT-DETRv2** (Apache-2.0) for a 100% permissively licensed, copyleft-free commercial tracking product.

---

## Architecture Alternatives & Trade-offs
| Tracking Paradigm | Algorithms | Latency / Overhead | Robustness to Occlusion | Ideal Deployment |
| :--- | :--- | :--- | :--- | :--- |
| **Motion-Only Association** | ByteTrack, OC-SORT | <1 ms association overhead | Moderate (breaks under prolonged camera rotation/stops) | High-speed edge tracking, high frame-rate video |
| **Motion + Appearance ReID** | BoT-SORT, DeepSORT | 5-15 ms (ReID feature extraction) | High (recovers identity after lengthy occlusion) | Multi-camera tracking, urban surveillance |
| **Dense Point Trackers** | CoTracker, TAPIR | 30-100 ms | Exceptional physical surface fidelity | Video editing, VFX, deformation analysis |
| **Memory-Prompt Video Trackers** | SAM 2 Video | 25-45 ms | High pixel-accurate mask tracking | Interactive tracking, surgical tool tracking |

---

## Popular Repos & Integrations
- **[facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker)**: Official CoTracker repository (Apache-2.0).
- **[NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT)**: Official BoT-SORT multi-object tracker (MIT).
- **[ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)**: SOTA real-time multi-object tracking implementation (MIT).
- **[google-deepmind/tapnet](https://github.com/google-deepmind/tapnet)**: Official TAPIR implementation (Apache-2.0).
- **Tooling Integrations**:
  - **FiftyOne**: Visualize video datasets with synchronized track IDs, color-coded trajectory paths, and ID switch inspection.
  - **Rerun**: Stream live trajectories as 2D/3D historical polyline paths with interactive timeline scrubbing.

---

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

---

## Deployment & Real-time Notes
- **Decoupled Tracker Threading**:
  - Run heavy neural detection on the GPU worker thread at e.g. 15-30 FPS, while running lightweight Kalman filter state updates and optical-flow interpolation on the CPU thread at 60-120 FPS.
- **ReID Embedding Caching**:
  - Calculate ReID cosine similarities only when IoU overlap is ambiguous ($0.2 < \text{IoU} < 0.7$). Bypassing the ReID network for straightforward associations reduces computational load by over 60%.
- **Zero-Allocation Track Pools**:
  - Pre-allocate track object buffers in memory to avoid dynamic heap allocations on every new track creation and deletion.

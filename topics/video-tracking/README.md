# Video Tracking Playbook

# Overview
Video Tracking maintains consistent object identity across sequential video frames despite camera motion, object deformation, lighting fluctuations, and severe occlusion. Tracking spans Multiple Object Tracking (MOT - tracking all detected instances), Single Object Tracking (SOT - following an arbitrary bounding-box prompt), and Point Tracking (dense or sparse physical surface point trajectory tracking).

## SOTA & Research
- **Seminal & Modern Papers**:
  - *SORT* (Bewley et al., 2016) & *DeepSORT* (Wojke et al., 2017): Established the modern Tracking-by-Detection paradigm pairing Kalman Filters with Hungarian matching and appearance re-identification (ReID) embeddings.
  - *ByteTrack* (Zhang et al., 2022): Simple yet remarkably effective association method preserving low-score detection boxes to prevent track fragmentation during occlusion.
  - *OC-SORT / BoT-SORT* (Cao et al., 2023; Aharon et al., 2022): Enhanced motion modeling with observation-centric momentum and camera motion compensation (CMC).
  - *CoTracker / TAPIR* (Karaev et al., 2023; Doersch et al., 2023): Dense point tracking in video using sliding-window transformer architectures.
  - *SAM 2* (Ravi et al., 2024): Foundation model for promptable video segmentation and tracking with temporal memory attention.
- **Evaluation Benchmarks & Metrics**:
  - MOT17, MOT20, DanceTrack (severe non-linear motion), TAO, BDD100K.
  - MOTA (Multi-Object Tracking Accuracy), IDF1 (ID preservation consistency), HOTA (Higher Order Tracking Accuracy - balancing detection vs association).

## Architecture Alternatives & Trade-offs
| Tracking Paradigm | Algorithms | Latency / Overhead | Robustness to Occlusion | Ideal Deployment |
| :--- | :--- | :--- | :--- | :--- |
| **Motion-Only Association** | ByteTrack, OC-SORT | <1 ms association overhead | Moderate (breaks under prolonged camera rotation/stops) | High-speed edge tracking, high frame-rate video |
| **Motion + Appearance ReID** | BoT-SORT, DeepSORT | 5-15 ms (ReID feature extraction) | High (recovers identity after lengthy occlusion) | Multi-camera tracking, urban surveillance |
| **End-to-End Joint Detect-Track** | FairMOT, TrackFormer | 15-30 ms | High | Unified server-side pipelines |
| **Dense Point Trackers** | CoTracker, TAPIR | 30-100 ms | Exceptional physical surface fidelity | Video editing, VFX, deformation analysis |

## Popular Repos & Integrations
- **[ByteTrack](https://github.com/ifzhang/ByteTrack)**: Canonical SOTA real-time multi-object tracking implementation.
- **[BoT-SORT](https://github.com/NirAharon/BoT-SORT)**: Top-ranked MOT tracker with camera motion compensation and fused ReID.
- **[CoTracker (Meta)](https://github.com/facebookresearch/co-tracker)**: Tracking any point in long video sequences via transformers.
- **[BoxMOT](https://github.com/mikel-brostrom/boxmot)**: Pluggable Python library packaging ByteTrack, BoT-SORT, DeepOCSORT, and StrongSORT.
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

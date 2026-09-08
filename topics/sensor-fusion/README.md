# Sensor Fusion Playbook

# Overview
Sensor Fusion unifies complementary physical sensing modalities—primarily Cameras (dense visual texture and color), LiDAR (accurate metric 3D geometry), RADAR (direct Doppler velocity and penetration through adverse weather), and IMU/Odometry (high-frequency motion dynamics). By fusing modalities, perception systems overcome single-sensor failure modes and build a coherent, robust representation of the 3D environment.

## SOTA & Research
- **Seminal & Modern Papers**:
  - *PointPainting / PointAugmenting* (Vora et al., 2020; Wang et al., 2021): Early-to-mid fusion projecting 2D camera semantic segmentation logits onto raw 3D LiDAR point clouds.
  - *BEVDet / BEVDepth* (Huang et al., 2021; Li et al., 2023): Transforming multi-view camera images into unified Bird's-Eye View (BEV) space using explicit depth supervision.
  - *BEVFusion* (Liu et al., 2023; Liang et al., 2022): State-of-the-art dual-branch fusion uniting multi-camera BEV features and LiDAR voxel BEV features in a unified latent space.
  - *UniAD* (Hu et al., 2023): Unified Autonomous Driving framework integrating tracking, mapping, prediction, and planning directly from multi-sensor BEV representations.
- **Evaluation Benchmarks & Metrics**:
  - nuScenes Detection and Tracking Benchmark (NDS, mAP, mAOE, mAVE).
  - Waymo Open Dataset (multi-sensor track).
  - Robustness benchmarks under sensor degradation (e.g., nuScenes-C: camera dirt, missing camera views, rainy LiDAR returns).

## Architecture Alternatives & Trade-offs
| Fusion Level | Paradigm / Algorithms | Latency Overhead | Robustness to Calibration Drift | Failure Mode Tolerance |
| :--- | :--- | :--- | :--- | :--- |
| **Late Fusion (Track / Object Level)** | Independent detectors + EKF / GNN association | Lowest (<5 ms overhead) | High (isolated sensor pipelines) | Fails if both sensors miss a degraded target independently |
| **Early Fusion (Data / Point Level)** | PointPainting | Low-to-Moderate (15-25 ms) | Extremely sensitive to extrinsic calibration errors | Camera latency blocks entire LiDAR pipeline |
| **Deep Feature BEV Fusion** | BEVFusion (Multi-Cam + LiDAR) | Moderate-to-High (30-50 ms) | Moderate | Retains LiDAR geometry even if camera is blinded by glare |

## Popular Repos & Integrations
- **[BEVFusion (MIT HAN Lab)](https://github.com/mit-han-lab/bevfusion)**: SOTA efficient multi-modal 3D perception framework with TensorRT acceleration.
- **[MMDetection3D Fusion Models](https://github.com/open-mmlab/mmdetection3d)**: Implementations of PointPainting, TransFusion, and BEVDet.
- **[OpenCalib](https://github.com/HV/OpenCalib)**: Multi-sensor calibration tools for Camera-LiDAR, Camera-Radar, and IMU extrinsics.
- **Tooling Integrations**:
  - **FiftyOne**: Inspect multi-modal samples with synchronized camera viewpoints, 3D LiDAR clouds, and ground-truth 3D boxes.
  - **Rerun**: Simultaneously log multi-camera pinhole feeds, 3D point clouds, extrinsic coordinate transforms, and BEV bounding boxes in a single synchronized interactive canvas.

## End-to-End Pipeline & Workarounds
1. **Pipeline Stages**:
   - Hardware PTP/PPS Time Synchronization -> Extrinsic calibration projection -> Multi-camera feature extraction -> Depth estimation & LSS (Lift-Splat-Shoot) -> LiDAR Voxel feature extraction -> Dynamic BEV pooling & feature concatenation -> Detection/Tracking head.
2. **Common Traps & Edge Cases**:
   - *Time Asynchrony (Temporal Jitter)*: Camera exposures and LiDAR sweeps firing at slightly different timestamps cause projected points on moving vehicles to miss targets.
   - *Extrinsic Thermal Calibration Drift*: Thermal expansion and chassis vibration cause subtle extrinsic camera-to-LiDAR rotation shifts ($<0.5^\circ$), ruining point-to-pixel projection.
   - *Camera Dropouts / Lens Soiling*: Deep fusion networks trained on pristine pairs may fail completely if one camera feed disconnects.
3. **Engineering Workarounds**:
   - **Hardware PTP / PPS Sync**: Synchronize all sensor clocks to sub-millisecond precision via IEEE 1588 PTP over Ethernet.
   - **Online Auto-Calibration / Targetless Calibration**: Run continuous background optimization minimizing edge alignment error between camera photometric gradients and LiDAR depth discontinuities.
   - **Sensor Dropout Augmentation (Modal Dropout)**: Randomly drop out the camera branch or LiDAR branch during training with 15% probability so the model maintains graceful degradation.

## Deployment & Real-time Notes
- **BEV Pooling Optimization**:
  - The classic Lift-Splat-Shoot (LSS) camera-to-BEV transformation involves irregular memory access. Replacing standard scatter-add with optimized CUDA kernels (e.g. MIT BEVFusion's fast BEV pooling) delivers an 8x speedup.
- **Heterogeneous Execution Pipeline**:
  - Distribute camera image backbones across GPU hardware while handling sensor stream deserialization, timestamp alignment, and Kalman tracker states on dedicated CPU threads.
- **Safety Fallbacks**:
  - In safety-critical systems (ISO 26262), always maintain a parallel rule-based Late Fusion / EKF track validator to catch anomalous feature-level hallucinations.

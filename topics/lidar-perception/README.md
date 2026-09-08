---
title: LiDAR Perception Playbook
tags:
  - computer-vision
  - 3d-perception
  - lidar
  - point-clouds
  - autonomous-vehicles
  - sparse-conv
updated: 2026-09-08
aliases:
  - LiDAR Perception
---

# LiDAR Perception Playbook

# Overview
LiDAR (Light Detection and Ranging) Perception processes 3D point cloud measurements to perform 3D object detection, semantic segmentation, scene flow estimation, and SLAM/odometry. LiDAR provides metric depth and spatial geometry invariant to ambient lighting, making it an indispensable sensor modality in autonomous vehicles, off-road robotics, and aerial surveying.

Related notes: [[topics/sensor-fusion/README|Sensor Fusion]], [[topics/6dof-pose-estimation/README|6-DoF Pose Estimation]], [[topics/gpu-deployment/README|GPU Deployment]].

## SOTA & Research
- **Recent Breakthroughs (2023–2026)**:
  - *DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets* (Wang et al., CVPR 2023) - [arXiv:2301.06051](https://arxiv.org/abs/2301.06051): Pure transformer 3D backbone replacing custom sparse convolutions with hardware-friendly dynamic sparse window attention, highly parallelizable in TensorRT.
  - *FlatFormer: Flattened Vision Transformer for 3D Point Cloud Processing* (Liu et al., 2023 / 2024) - [arXiv:2301.08739](https://arxiv.org/abs/2301.08739): Equal-length window partitioning yielding ultra-fast 3D transformer execution with 4.6x speedup over standard 3D voxel backbones.
  - *HEDNet: A Hierarchical Encoder-Decoder Network for 3D Object Detection* (Zhang et al., 2023 / 2024) - [arXiv:2301.12151](https://arxiv.org/abs/2301.12151): Addressed feature dissipation in deep sparse networks with hierarchical residual dense connections.
  - *CenterPoint: Center-based 3D Object Detection and Tracking* (Yin et al., 2021 / production updates): Still the gold standard industrial architecture anchor for anchor-free 3D oriented bounding box regression.

### Quantitative SOTA Benchmark Comparison (Waymo Open Dataset & nuScenes)
| Architecture Model | Primary Benchmark | mAP / NDS | Latency (FP16 ms) | Hardware Profile |
| :--- | :--- | :--- | :--- | :--- |
| **DSVT (Voxel)** | Waymo Val Level 2 | 74.9% mAPH | 27.0 ms | TensorRT NVIDIA Orin / A100 |
| **FlatFormer** | Waymo Val Level 2 | 74.1% mAPH | 18.5 ms | High-throughput Edge GPU |
| **CenterPoint (Voxel)** | nuScenes Test | 67.3% NDS / 60.3% mAP | 22.0 ms | Jetson AGX Orin |
| **PointPillars** | nuScenes Test | 59.7% NDS / 40.1% mAP | 8.2 ms | Embedded Automotive GPU |
| **PV-RCNN++** | Waymo Val Level 2 | 75.3% mAPH | 55.0 ms | High-end Server GPU |

## Architecture Alternatives & Trade-offs
| Architecture | Representation | Latency (GPU) | 3D Accuracy (mAP) | Hardware Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **PointPillars** | 2D Pillar BEV Grid | 8-12 ms | Moderate | Lightweight GPU / Edge Compute |
| **CenterPoint (Voxel / Pillar)** | Sparse 3D Voxels | 15-25 ms | High (Industry standard) | CUDA GPU with SparseConv support |
| **DSVT / FlatFormer** | Dynamic Sparse Transformer | 18-28 ms | Highest accuracy & scaling | Modern Tensor-Core GPU |
| **PV-RCNN++** | Hybrid Point-Voxel | 40-70 ms | Highest point-level localization | High-end Server / In-vehicle GPU |

## Popular Repos & Integrations
- **[open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet)**: SOTA open-source codebase for LiDAR-based 3D object detection including DSVT, CenterPoint, and PointPillars.
- **[mit-han-lab/flatformer](https://github.com/mit-han-lab/flatformer)**: Official FlatFormer repository optimized for fast 3D point cloud transformer inference.
- **[traveller59/spconv](https://github.com/traveller59/spconv)**: Highly optimized spatial sparse convolution library for CUDA and TensorRT.
- **Tooling Integrations**:
  - **FiftyOne**: Native 3D scene visualizer supporting point clouds (`.pcd`, `.bin`), 3D cuboid annotations, and velocity vectors.
  - **Rerun**: Stream dense LiDAR scans (`rr.Points3D`) with intensity color maps, ego-vehicle transforms, and predicted 3D bounding boxes (`rr.Boxes3D`).

## End-to-End Pipeline & Workarounds
1. **Pipeline Stages**:
   - Packet capture (UDP Ethernet) -> Motion deskewing (ego-motion compensation via IMU) -> ROI cropping / ground plane removal -> Voxelization / Pillarization -> 3D Backbone (SparseConv) -> BEV Dense Flattening -> Center Head -> 3D NMS.
2. **Common Traps & Edge Cases**:
   - *Motion Distortion (Skew)*: Rotating LiDAR beams take 50-100 ms to complete a single $360^\circ$ sweep; during vehicle motion, points appear smeared or warped.
   - *Sparsity at Range*: Point density drops quadratically with distance ($1/r^2$), leaving distant objects represented by only 2-5 points.
   - *Adverse Weather (Rain, Fog, Dust)*: Atmospheric backscatter produces thousands of false-positive floating points near the sensor.
3. **Engineering Workarounds**:
   - **Linear Motion Deskewing**: Interpolate high-rate IMU/odometry poses across the microsecond timestamps of each individual firing pulse to project all points into a single reference frame.
   - **Temporal Point Cloud Accumulation**: Concatenate multiple motion-compensated sweeps (e.g. 5-10 frames) into a dense aggregated cloud.
   - **Intensity Dynamic Filtering**: Threshold high-variance low-intensity clusters caused by rain spray and vehicle exhaust.

## Deployment & Real-time Notes
- **Voxel Hash Table Acceleration**:
  - Sparse convolution performance hinges on hash table lookups. Use optimized CUDA kernels (like SpConv 2.x) that cache neighbor indices.
- **TensorRT Plugin Integration**:
  - For embedded deployment (e.g. NVIDIA Jetson Orin), compile custom TensorRT plugins for Voxelization, Sparse 3D Submanifold Convolutions, and 3D Rotate NMS.
- **Throughput & Bandwidth**:
  - A 128-beam LiDAR produces over 2.4 million points/second (approx. 50 MB/s raw stream). Ingest directly via zero-copy socket buffers (e.g., AF_XDP or kernel ring buffers).

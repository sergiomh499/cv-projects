# LiDAR Perception Playbook

# Overview
LiDAR (Light Detection and Ranging) Perception processes 3D point cloud measurements to perform 3D object detection, semantic segmentation, scene flow estimation, and SLAM/odometry. LiDAR provides metric depth and spatial geometry invariant to ambient lighting, making it an indispensable sensor modality in autonomous vehicles, off-road robotics, and aerial surveying.

## SOTA & Research
- **Seminal & Modern Papers**:
  - *PointNet / PointNet++* (Qi et al., 2017): Direct processing of unordered point sets using permutation-invariant symmetric pooling and hierarchical spatial grouping.
  - *VoxelNet & SECOND* (Zhou & Tuzel, 2018; Yan et al., 2018): Discretized 3D space into voxels and introduced Spatially Sparse Convolutions (SparseConv), solving the computational quadratic cost of 3D dense convolution.
  - *PointPillars* (Lang et al., 2019): Collapsed 3D point clouds into vertical 2D pillars, enabling standard 2D CNN inference at ultra-high frame rates (>60 FPS).
  - *CenterPoint* (Yin et al., 2021): Represented 3D objects as bounding box centers followed by regression of dimensions, orientations, and velocities.
  - *PV-RCNN / PV-RCNN++* (Shi et al., 2020, 2023): Point-voxel integrated network combining voxel feature abstraction with keypoint-based set abstraction.
- **Evaluation Benchmarks & Metrics**:
  - KITTI 3D Object Detection Benchmark, nuScenes 3D Detection (NDS - nuScenes Detection Score, mAP), Waymo Open Dataset (mAP, mAPH taking heading into account), SemanticKITTI.
  - Metrics: 3D IoU, Average Precision with 3D Oriented Bounding Box, Bird's-Eye View (BEV) AP.

## Architecture Alternatives & Trade-offs
| Architecture | Representation | Latency (GPU) | 3D Accuracy (mAP) | Hardware Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **PointPillars** | 2D Pillar BEV Grid | 10-15 ms | Moderate | Lightweight GPU / Edge Compute |
| **CenterPoint (Voxel / Pillar)** | Sparse 3D Voxels | 15-30 ms | High (Industry standard) | CUDA GPU with SparseConv support |
| **PV-RCNN++** | Hybrid Point-Voxel | 40-70 ms | Highest accuracy | High-end Server / In-vehicle GPU |
| **Range-View (SqueezeSeg, RangeNet)**| 2D Spherical Projection | 8-12 ms | Lower at long distances | Edge systems, standard 2D accelerators |

## Popular Repos & Integrations
- **[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)**: Clear, modular open-source codebase for LiDAR-based 3D object detection.
- **[spconv (Sparse Convolution)](https://github.com/traveller59/spconv)**: Highly optimized spatial sparse convolution library for CUDA and TensorRT.
- **[MMDetection3D](https://github.com/open-mmlab/mmdetection3d)**: Unified 3D perception platform supporting multi-modal and LiDAR pipelines.
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

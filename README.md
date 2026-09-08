---
title: Computer Vision & Perception Projects Knowledge Hub
tags:
  - computer-vision
  - perception
  - deep-learning
  - sota
  - obsidian-vault
  - licenses
  - index
updated: 2026-09-08
aliases:
  - Hub Index
  - CV Projects
---

# 🧠 Computer Vision & Perception Projects Knowledge Hub

A curated research engineering knowledge base, didactic repository, and production playbook vault covering end-to-end computer vision, multi-modal sensor fusion, edge deployment (CUDA/Vulkan/FPGA), and microsecond real-time robotics.

> **Obsidian Users**: This repository is designed to be opened directly as an **Obsidian Vault**. It implements a hybrid **PARA + Zettelkasten + Maps of Content (MOC)** structure with bidirectional wikilinks (`[[...]]`), structured YAML properties, and central architecture notes under `architectures/`.

---

## 🧭 Master Knowledge Map & Topic MOCs

| Domain Topic | Obsidian MOC Note | Core SOTA Breakthroughs | Primary Repositories | Latency / FPS | Safe Commercial License |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **1. Object Detection** | [[topics/object-detection/00-object-detection-moc\|Detection MOC]] ([Directory](topics/object-detection/README.md)) | RF-DETR, RT-DETRv2/v3, YOLOv10-26, Grounding DINO | `roboflow/rf-detr`, `lyuwenyu/RT-DETR` | 5.2 ms (53.8% AP) | **Apache-2.0** |
| **2. Object Segmentation** | [[topics/object-segmentation/00-object-segmentation-moc\|Segmentation MOC]] ([Directory](topics/object-segmentation/README.md)) | SAM 2.1, Depth Anything V2, Mask2Former | `facebookresearch/sam2`, `DepthAnything` | 22.8 ms (44 FPS) | **Apache-2.0** |
| **3. Object Classification** | [[topics/object-classification/00-object-classification-moc\|Classification MOC]] ([Directory](topics/object-classification/README.md)) | DINOv2/v3, SigLIP 2, ConvNeXt V2, MobileNetV4 | `facebookresearch/dinov2`, `google/big_vision` | 0.85 ms (82.3% Top-1) | **Apache-2.0** |
| **4. Video Tracking** | [[topics/video-tracking/00-video-tracking-moc\|Tracking MOC]] ([Directory](topics/video-tracking/README.md)) | BoT-SORT, ByteTrack, CoTracker3, SAM 2 | `ifzhang/ByteTrack`, `facebookresearch/co-tracker` | 1.2 ms (80.3 MOTA) | **MIT / Apache-2.0** |
| **5. 6-DoF Pose Estimation**| [[topics/6dof-pose-estimation/00-6dof-pose-estimation-moc\|6-DoF Pose MOC]] ([Directory](topics/6dof-pose-estimation/README.md)) | FoundationPose, MegaPose, GDR-Net | `NVlabs/FoundationPose`, `facebookresearch/megapose` | 32.0 ms (96.2% ADD-S) | **Apache-2.0** (MegaPose) |
| **6. LiDAR Perception** | [[topics/lidar-perception/00-lidar-perception-moc\|LiDAR MOC]] ([Directory](topics/lidar-perception/README.md)) | DSVT, FlatFormer, PointPillars, CenterPoint | `Haiyang-W/DSVT`, `open-mmlab/OpenPCDet` | 27.0 ms (78.9% mAP) | **Apache-2.0** |
| **7. Sensor Fusion** | [[topics/sensor-fusion/00-sensor-fusion-moc\|Sensor Fusion MOC]] ([Directory](topics/sensor-fusion/README.md)) | MIT BEVFusion, Sparse4D v3, UniAD | `mit-han-lab/bevfusion`, `HorizonRobotics/Sparse4D` | 41.0 ms (72.9% NDS) | **Apache-2.0** |
| **8. FPGA Deployment** | [[topics/fpga-deployment/00-fpga-deployment-moc\|FPGA MOC]] ([Directory](topics/fpga-deployment/README.md)) | AMD Vitis AI 3.5, FINN & Brevitas (QNN) | `Xilinx/Vitis-AI`, `Xilinx/finn` | 0.68 ms (1,450 FPS) | **Apache-2.0 / EULA** |
| **9. GPU Deployment** | [[topics/gpu-deployment/00-gpu-deployment-moc\|GPU Deployment MOC]] ([Directory](topics/gpu-deployment/README.md)) | TensorRT 10 (FP8), Vulkan NCNN, FlashAttention | `NVIDIA/TensorRT`, `Tencent/ncnn` | 0.25 ms (ResNet-50) | **Apache-2.0 / BSD** |
| **10. Real-Time Systems** | [[topics/real-time-systems/00-real-time-systems-moc\|Real-Time MOC]] ([Directory](topics/real-time-systems/README.md)) | Linux PREEMPT_RT, Eclipse Iceoryx2, Zenoh | `eclipse-iceoryx/iceoryx2`, `eclipse-zenoh/zenoh` | <1 $\mu$s Zero-Copy | **Apache-2.0 / MIT** |
| **11. SLAM & Spatial Perception** | [[topics/slam-and-spatial-perception/00-slam-and-spatial-perception-moc\|SLAM MOC]] ([Directory](topics/slam-and-spatial-perception/README.md)) | 3DGS SLAM, MonoGS, SplaTAM, FAST-LIO2 | `muskie82/MonoGS`, `spla-tam/SplaTAM` | 28.0 FPS (1.8 cm ATE) | **Apache-2.0 / MIT** |
| **12. Visual Guidance & Robotics** | [[topics/visual-guidance-and-robotics/00-visual-guidance-and-robotics-moc\|Visual Guidance MOC]] ([Directory](topics/visual-guidance-and-robotics/README.md)) | OpenVLA 7B, Octo, AnyGrasp | `openvla/openvla`, `octo-models/octo` | 180 ms (84.7% Success) | **Apache-2.0** (OpenVLA) |

---

## 🏛️ Central Architecture Vault (`architectures/`)

Cross-cutting models and multi-task foundation backbones shared across topics:
- **[[architectures/foundation-models/depth-anything-v2|Depth Anything V2: Metric Depth & Surface Segmentation Foundation Model]]**: Foundation Vision Transformer (DINOv2 Distillation) (Apache-2.0).
- **[[architectures/foundation-models/dinov2-and-dinov3|DINOv2 & DINOv3: Self-Supervised Vision Foundation Backbones]]**: Self-Supervised Vision Transformer (ViT) (Apache-2.0).
- **[[architectures/foundation-models/sam-2|SAM 2 & SAM 2.1: Segment Anything in Images and Videos]]**: Foundation Model (Hierarchical Vision Transformer) (Apache-2.0).
- **[[architectures/foundation-models/siglip|SigLIP & SigLIP 2: Sigmoid Loss for Vision-Language Alignment]]**: Dual-Encoder Vision-Language Transformer (Apache-2.0).
- **[[architectures/real-time-unified/3dgs-slam-and-monogs|3D Gaussian Splatting SLAM & MonoGS: Real-Time Radiance Field Odometry]]**: Differentiable 3D Gaussian Primitive Optimization (Apache-2.0 / MIT).
- **[[architectures/real-time-unified/anygrasp-and-openvla|AnyGrasp & OpenVLA: Foundation Models for Robotic Visual Guidance]]**: Foundation Vision-Language-Action (VLA) & Dense Grasp Network (Apache-2.0 / Non-Commercial).
- **[[architectures/real-time-unified/bevfusion-and-sparse4d|BEVFusion & Sparse4D: Multi-Modal Camera-LiDAR Sensor Fusion]]**: Unified Bird's-Eye-View (BEV) Transformer (Apache-2.0 / MIT).
- **[[architectures/real-time-unified/botsort-and-bytetrack|BoT-SORT & ByteTrack: Real-Time Multi-Object Tracking (MOT)]]**: Tracking-by-Detection (Kalman + ReID + GMC) (MIT / Apache-2.0).
- **[[architectures/real-time-unified/convnext-and-mobilenet|ConvNeXt V2 & MobileNetV4: Modern Edge & Workstation Visual Backbones]]**: Modern Pure ConvNet (Inverted Bottleneck & GRN) (Apache-2.0).
- **[[architectures/real-time-unified/dsvt-and-flatformer|DSVT & FlatFormer: Dynamic Sparse Window Transformers for 3D LiDAR]]**: Sparse Voxel Transformer (Apache-2.0).
- **[[architectures/real-time-unified/foundationpose-and-megapose|FoundationPose & MegaPose: 6-DoF Zero-Shot Object Pose Estimation]]**: Foundation 3D Vision Transformer (Render-and-Compare + Score Network) (Custom Non-Commercial (FoundationPose) / Apache-2.0 (MegaPose)).
- **[[architectures/real-time-unified/iceoryx2-and-zenoh|Iceoryx2 & Zenoh: Ultra-Low-Latency Real-Time Middleware]]**: Lock-Free Shared Memory Inter-Process Communication (IPC) (Apache-2.0 / MIT).
- **[[architectures/real-time-unified/mask2former|Mask2Former: Masked-Attention Mask Transformer for Universal Image Segmentation]]**: Universal Query Transformer (Apache-2.0).
- **[[architectures/real-time-unified/tensorrt-and-vulkan|TensorRT 10 & Vulkan Kompute: Cross-Platform High-Throughput GPU Inference]]**: Deep Learning Compiler & Cross-Platform Compute Runtimes (Apache-2.0 / NVIDIA Proprietary EULA).
- **[[architectures/real-time-unified/vitis-ai-and-finn|Vitis AI 3.5 & FINN: Quantized Neural Inference on FPGAs]]**: Reconfigurable Logic (DPU Cores & Dataflow Streaming Engines) (Apache-2.0 / Xilinx EULA).
- **[[architectures/transformer-detectors/cotracker|CoTracker & CoTracker3: Dense Point Trajectory Transformers]]**: Spatial-Temporal Point Transformer (Apache-2.0).

---

## ⚖️ Global Commercial Usability & License Matrix

| Category | Safe for Proprietary Software? | Key Repositories & Models |
| :--- | :---: | :--- |
| **Permissive (Apache-2.0 / MIT / BSD)** | **YES** | `rf-detr`, `RT-DETR (v2/v3)`, `SAM 2`, `Depth-Anything-V2`, `DINOv2/v3`, `SigLIP`, `ConvNeXt-V2`, `ByteTrack`, `BoT-SORT`, `CoTracker3`, `MegaPose`, `DSVT`, `OpenPCDet`, `BEVFusion`, `Sparse4D`, `Vitis-AI`, `FINN`, `TensorRT`, `ncnn`, `iceoryx2`, `zenoh`, `MonoGS`, `SplaTAM`, `OpenVLA`, `Octo` |
| **Copyleft Warning (AGPL-3.0 / GPL-3.0)** | ⚠️ **NO** | `Ultralytics YOLOv8 / YOLO11` (AGPL-3.0), `DeepSORT` (GPL-3.0), `ORB-SLAM3` (GPL-3.0), `FAST-LIO2` (GPL-2.0). Using copyleft code over network APIs requires open-sourcing client callers. Standardize on **RF-DETR**, **MegaPose**, **MonoGS**, and **OpenVLA** to eliminate legal risks. |
| **Non-Commercial / Research-Only** | ⛔ **NO** | `NVlabs/FoundationPose` (NVIDIA Source Code License), `AnyGrasp` (Research license). Standardize on **MegaPose** and **Contact-GraspNet** for proprietary robotic deployment. |

---

## 🔄 Automated Documentation Trigger & Validation

To automatically synchronize all cross-domain backlinks and validate the vault:

```bash
# Auto-sync architecture indices and backlinks
python scripts/update_docs.py

# Run comprehensive schema validation
python scripts/validate_repo.py
```

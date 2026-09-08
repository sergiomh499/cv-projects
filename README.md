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

---

## 🏛️ Central Architecture Vault (`architectures/`)

Cross-cutting models and multi-task foundation backbones shared across topics:
- **[[architectures/foundation-models/sam-2|SAM 2 & 2.1]]**: Promptable image segmentation and 44 FPS video tracking memory bank (Apache-2.0).
- **[[architectures/foundation-models/depth-anything-v2|Depth Anything V2]]**: Continuous metric depth and geometric surface parsing (Apache-2.0).
- **[[architectures/foundation-models/dinov2-and-dinov3|DINOv2 & DINOv3]]**: Self-supervised dense representations for detection, classification, and depth (Apache-2.0).
- **[[architectures/foundation-models/siglip|SigLIP & SigLIP 2]]**: Sigmoid binary loss for zero-shot open-vocabulary alignment (Apache-2.0).
- **[[architectures/real-time-unified/mask2former|Mask2Former]]**: Unified query transformer for semantic, instance, and panoptic segmentation (Apache-2.0).
- **[[architectures/real-time-unified/botsort-and-bytetrack|BoT-SORT & ByteTrack]]**: Production real-time multi-object tracking with motion compensation (MIT).
- **[[architectures/transformer-detectors/cotracker|CoTracker3]]**: Dense spatial-temporal point trajectory transformer tracking 70k points (Apache-2.0).
- **[[architectures/real-time-unified/foundationpose-and-megapose|FoundationPose & MegaPose]]**: Zero-shot CAD-driven 6-DoF object pose tracking (Apache-2.0 / NVIDIA).
- **[[architectures/real-time-unified/dsvt-and-flatformer|DSVT & FlatFormer]]**: Dynamic sparse window transformers compiling natively to TensorRT without SpConv (Apache-2.0).
- **[[architectures/real-time-unified/bevfusion-and-sparse4d|BEVFusion & Sparse4D]]**: Camera-LiDAR Bird's-Eye-View fusion with fast coordinate caching (Apache-2.0).
- **[[architectures/real-time-unified/vitis-ai-and-finn|Vitis AI & FINN]]**: Quantized neural network acceleration on FPGAs (Apache-2.0).
- **[[architectures/real-time-unified/tensorrt-and-vulkan|TensorRT 10 & Vulkan]]**: Deep learning compilation, FP8, and vendor-agnostic compute shaders (Apache-2.0 / BSD).
- **[[architectures/real-time-unified/iceoryx2-and-zenoh|Iceoryx2 & Zenoh]]**: Zero-copy shared memory IPC delivering sub-microsecond latency (Apache-2.0 / MIT).

---

## ⚖️ Global Commercial Usability & License Matrix

| Category | Safe for Proprietary Software? | Key Repositories & Models |
| :--- | :---: | :--- |
| **Permissive (Apache-2.0 / MIT / BSD)** | **YES** | `rf-detr`, `RT-DETR (v2/v3)`, `SAM 2`, `Depth-Anything-V2`, `DINOv2/v3`, `SigLIP`, `ConvNeXt-V2`, `ByteTrack`, `BoT-SORT`, `CoTracker3`, `MegaPose`, `DSVT`, `OpenPCDet`, `BEVFusion`, `Sparse4D`, `Vitis-AI`, `FINN`, `TensorRT`, `ncnn`, `iceoryx2`, `zenoh` |
| **Copyleft Warning (AGPL-3.0 / GPL-3.0)** | ⚠️ **NO** | `Ultralytics YOLOv8 / YOLO11` (AGPL-3.0), `DeepSORT` (GPL-3.0). Using AGPL-3.0 over network APIs requires open-sourcing the caller. Standardize on **RF-DETR** or **RT-DETRv2/v3** to avoid legal risk. |
| **Non-Commercial / Research-Only** | ⛔ **NO** | `NVlabs/FoundationPose` (NVIDIA Source Code License). For commercial 6-DoF pose estimation, standardize on **MegaPose** (Apache-2.0). |

---

## 🚀 Environment Setup & Validation

This project is configured with `uv` for deterministic dependency management.

```bash
# Sync dependencies
uv sync

# Run repository schema and link validator
uv run python scripts/validate_repo.py
```

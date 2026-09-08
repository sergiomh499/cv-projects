---
title: Object Segmentation Playbook
tags:
  - computer-vision
  - object-segmentation
  - sam2
  - foundation-models
  - mask2former
  - mobilenet
updated: 2026-09-08
aliases:
  - Object Segmentation
---

# Object Segmentation Playbook

# Overview
Object Segmentation classifies visual scenes down to individual pixel coordinates. It branches into:
1. **Semantic Segmentation**: Assigns a semantic class category (e.g., road, sky, vehicle) to every pixel in the image.
2. **Instance Segmentation**: Detects and delineates individual object boundaries (e.g., Person #1, Person #2).
3. **Panoptic Segmentation**: Unifies background "stuff" and foreground "things" into an exhaustive per-pixel representation.

Related notes: [[topics/object-detection/README|Object Detection]], [[topics/video-tracking/README|Video Tracking]], [[topics/sensor-fusion/README|Sensor Fusion]].

---

## SOTA & Research (2023–2026 Breakthroughs)

1. **SAM 2 & SAM 2.1: Segment Anything in Images and Videos** (Ravi et al., Meta FAIR, 2024 / 2025)
   - *Key Innovation*: Foundation segmentation architecture with streaming spatial-temporal memory attention. Capable of promptable zero-shot image segmentation and video object mask propagation at 44 FPS.
   - [Paper: arXiv:2408.00714](https://arxiv.org/abs/2408.00714) | [Official Code](https://github.com/facebookresearch/sam2)

2. **Depth Anything V2: A Foundation Model for Monocular Depth Estimation** (Yang et al., 2024 / 2025)
   - *Key Innovation*: Trains powerful visual priors using synthetic data distillation, delivering metric-scale continuous depth segmentation and geometric boundary sharpness vastly superior to MiDaS.
   - [Paper: arXiv:2406.09414](https://arxiv.org/abs/2406.09414) | [Official Code](https://github.com/DepthAnything/Depth-Anything-V2)

3. **Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation** (Cheng et al., Meta / UIUC, 2022 / 2023)
   - *Key Innovation*: Universal architecture replacing per-pixel classification heads with localized masked cross-attention queries. Still holds top-tier accuracy on ADE20K and Cityscapes.
   - [Paper: arXiv:2112.01527](https://arxiv.org/abs/2112.01527) | [Official Code](https://github.com/facebookresearch/Mask2Former)

4. **FastSAM & MobileSAM** (Zhao et al., 2023 / Zhang et al., 2023)
   - *Key Innovation*: Distills SAM's expensive ViT image encoder into a lightweight CNN prototype architecture and tiny ViT-T backbones, enabling real-time promptable segmentation on mobile NPUs and edge GPUs.
   - [FastSAM Paper](https://arxiv.org/abs/2306.12156) | [MobileSAM Paper](https://arxiv.org/abs/2306.14289)

---

### Quantitative SOTA Benchmark Comparison
| Architecture | Benchmark Dataset | Metric | Latency / FPS | Target Hardware | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SAM 2 (Hiera-B+)** | SA-V Video Benchmark | 75.0 J&F | 43.8 FPS | NVIDIA A100 / RTX 4090 | Apache-2.0 |
| **SAM 2 (Hiera-Tiny)**| SA-V Video Benchmark | 71.5 J&F | 68.0 FPS | NVIDIA RTX 3090 / Jetson Orin | Apache-2.0 |
| **Mask2Former (Swin-L)**| ADE20K Val Semantic | 57.7% mIoU | 5.2 FPS | Server GPU (A100) | Apache-2.0 |
| **Depth Anything V2-Large**| NYUv2 / KITTI Depth | 0.075 Rel Error | 22.0 ms | TensorRT FP16 | Apache-2.0 |
| **YOLOv8x-Seg** | COCO val2017 Instance | 43.4% Mask AP | 13.5 ms | TensorRT FP16 | AGPL-3.0 / Commercial |
| **MobileSAM (ViT-Tiny)**| COCO Zero-shot sample | 58.2% AP@50 | 9.8 ms | PyTorch CUDA / Edge NPU | Apache-2.0 |

---

## Commercial Usability & License Audit
- **Commercial-Friendly (Apache-2.0 / MIT)**:
  - **Meta SAM 2 (`facebookresearch/sam2`)**: Licensed under **Apache-2.0**. Safe for commercial SaaS and edge products without exposing caller code. Note: Check model checkpoint weights terms for specific training datasets.
  - **Mask2Former (`facebookresearch/Mask2Former`)**: Licensed under **Apache-2.0**.
  - **Depth Anything V2 (`DepthAnything/Depth-Anything-V2`)**: Licensed under **Apache-2.0**.
  - **MobileSAM (`ChaoningZhang/MobileSAM`)**: Licensed under **Apache-2.0**.
- **Copyleft (Caution Required)**:
  - **YOLOv8-Seg / YOLO11-Seg (Ultralytics)**: **AGPL-3.0**. Requires open-sourcing client applications or buying an Ultralytics enterprise commercial license.
  - *Recommendation*: For commercial projects, use **SAM 2 (Tiny/Small)** for promptable segmentation or **MMSegmentation / DeepLabV3+** for semantic masks to stay strictly within Apache-2.0 permissive terms.

---

## Architecture Alternatives & Trade-offs
| Architecture Family | Examples | Latency / Hardware Fit | Boundary Sharpness | Ideal Product Fit |
| :--- | :--- | :--- | :--- | :--- |
| **Promptable Foundation Segmenters** | SAM 2, MobileSAM | 10-25 ms (Tiny/Small) | Exceptional (sub-pixel) | Interactive annotation, surgical robotics, dynamic tracking |
| **Universal Query Transformers** | Mask2Former | 40-150 ms | Very high | Offline mapping, medical scan diagnostic analysis |
| **Real-Time Prototype CNNs** | YOLOv8-Seg, FastSAM | 8-15 ms | Moderate | Autonomous driving lane/drivable space, obstacle masks |

---

## Popular Repos & Integrations
- **[facebookresearch/sam2](https://github.com/facebookresearch/sam2)**: Official Meta SAM 2 repository (Apache-2.0).
- **[DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)**: SOTA monocular geometric depth and surface segmentation (Apache-2.0).
- **[facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former)**: Official universal segmentation engine (Apache-2.0).
- **[ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)**: Real-time edge mobile implementation (Apache-2.0).
- **[open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)**: Comprehensive modular toolbox for over 60 segmentation networks (Apache-2.0).
- **Tooling Integrations**:
  - **FiftyOne**: Inspect false positive mask leaks, compute per-pixel IoU histograms, filter instances with high boundary error.
  - **Rerun**: Log 2D segmentation masks and alpha overlays synchronized with raw camera feeds using `rr.SegmentationImage`.

---

## End-to-End Pipeline & Workarounds
1. **Pipeline Stages**:
   - Frame ingest -> Resize/pad -> Neural inference -> Proto-mask matrix multiplication -> Polygon contour extraction -> Binary mask compression (RLE).
2. **Common Traps & Edge Cases**:
   - *Memory Blowup*: Storing high-resolution uncompressed binary masks ($H \times W \times N$) in RAM quickly triggers OOM errors.
   - *Coarse Edge Artifacts*: Low-resolution prototype grids cause jagged or floating boundaries on thin structures (e.g. cables, limbs).
3. **Engineering Workarounds**:
   - **Run-Length Encoding (RLE)**: Immediately compress masks using pycocotools RLE or bitmasks before IPC or serial transmission.
   - **PointRend & Guided Filter Post-Processing**: Refine boundary pixels dynamically on top of prototype masks without running expensive full-resolution decoders.
   - **Downsampled Mask Buffers**: Maintain mask outputs at $1/4$ or $1/8$ spatial resolution and upscale using hardware bilinear interpolation during display/rendering.

---

## Deployment & Real-time Notes
- **TensorRT Mask Matmul Fusion**:
  - Fusing the mask coefficients ($k \times N$) and prototype map ($k \times h \times w$) via custom TensorRT GEMM plugins bypasses CPU-GPU memory bottlenecks.
- **Vulkan Acceleration**:
  - Lightweight models deploy cleanly on embedded mobile GPUs via Vulkan shaders in Tencent NCNN.
- **Memory Footprint**:
  - Pre-allocate contiguous pinned host/device buffers for the maximum expected number of instances to prevent dynamic allocation spikes during live video processing.

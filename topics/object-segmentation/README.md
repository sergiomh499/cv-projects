---
title: Object Segmentation Playbook
tags:
  - computer-vision
  - object-segmentation
  - instance-segmentation
  - sam2
  - foundation-models
updated: 2026-09-08
aliases:
  - Object Segmentation
---

# Object Segmentation Playbook

# Overview
Object Segmentation partitions an image into semantically meaningful regions down to the pixel level. It spans Semantic Segmentation (assigning class labels to every pixel), Instance Segmentation (delineating discrete individual object instances), and Panoptic Segmentation (unifying things and stuff). Key practical applications include autonomous driving free-space detection, medical imaging lesion tracing, surgical robotics, and background matting.

Related notes: [[topics/object-detection/README|Object Detection]], [[topics/video-tracking/README|Video Tracking]], [[topics/sensor-fusion/README|Sensor Fusion]].

## SOTA & Research
- **Recent Breakthroughs (2023–2026)**:
  - *SAM 2: Segment Anything in Images and Videos* (Ravi et al., Meta FAIR, 2024) - [arXiv:2408.00714](https://arxiv.org/abs/2408.00714): Extends promptable foundation segmentation to streaming video with spatial-temporal memory attention and real-time inference (44 FPS).
  - *Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation* (Cheng et al., 2022 / 2023) - [arXiv:2112.01527](https://arxiv.org/abs/2112.01527): Unified architecture outperforming specialized models on semantic (ADE20K 57.7% mIoU), instance (COCO 50.1% AP), and panoptic segmentation.
  - *FastSAM: Fast Segment Anything* (Zhao et al., 2023) - [arXiv:2306.12156](https://arxiv.org/abs/2306.12156): Reformulated promptable segmentation into a two-stage CNN prototype architecture executing at 50 FPS on a single RTX 3090.
  - *MobileSAM: Faster Segment Anything Anywhere* (Zhang et al., 2023) - [arXiv:2306.14289](https://arxiv.org/abs/2306.14289): Distilled ViT image encoder into a lightweight decoupled network running at sub-10ms latency on edge hardware.

### Quantitative SOTA Benchmark Comparison
| Architecture | Benchmark Dataset | Metric | Latency / FPS | Target Hardware |
| :--- | :--- | :--- | :--- | :--- |
| **SAM 2 (Hiera-B+)** | SA-V Video Benchmark | 75.0 J&F | 43.8 FPS | NVIDIA A100 / RTX 4090 |
| **SAM 2 (Hiera-Tiny)**| SA-V Video Benchmark | 71.5 J&F | 68.0 FPS | NVIDIA RTX 3090 / Jetson AGX |
| **Mask2Former (Swin-L)**| ADE20K Val / COCO Panoptic| 57.7% mIoU / 58.3% PQ | 5.2 FPS | Server GPU (A100) |
| **YOLOv8x-Seg** | COCO val2017 Instance | 43.4% Mask AP | 13.5 ms | TensorRT FP16 |
| **FastSAM (YOLOv8x base)**| SA-1B sample set | 63.7% AP@50 | 25.0 ms | PyTorch CUDA |

## Architecture Alternatives & Trade-offs
| Architecture | Complexity / FPS | Boundary Precision | Use-Case |
| :--- | :--- | :--- | :--- |
| **YOLO-Seg (Prototype-based)** | 60-120+ FPS (TensorRT) | Moderate (limited by proto mask resolution) | Embedded robotics, real-time tracking |
| **Mask2Former** | 10-25 FPS (Server GPU) | Very high across complex multi-class scenes | Offline analytics, high-accuracy mapping |
| **SAM 2 (Video/Image)** | 30-44 FPS (streaming prompt) | Highest zero-shot mask quality | Interactive annotation, prompt-based tracking |
| **Lightweight Semantic (BiSeNet V2, PIDNet)** | 100+ FPS (Vulkan/Edge) | Coarse boundaries, high spatial recall | Autonomous vehicle driveable area / lane line detection |

## Popular Repos & Integrations
- **[facebookresearch/sam2](https://github.com/facebookresearch/sam2)**: Official implementation of Meta Segment Anything Model 2 for real-time video and image segmentation.
- **[facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former)**: Official PyTorch/Detectron2 implementation of Mask2Former universal segmentation.
- **[ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)**: Lightweight, real-time mobile implementation of SAM.
- **[open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)**: SOTA semantic segmentation toolbox supporting over 60 networks.
- **Tooling Integrations**:
  - **FiftyOne**: Inspect false positive mask leaks, compute per-pixel IoU histograms, filter instances with high boundary error.
  - **Rerun**: Log 2D segmentation masks and alpha overlays synchronized with raw camera feeds using `rr.SegmentationImage`.

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

## Deployment & Real-time Notes
- **TensorRT Mask Matmul Fusion**:
  - Fusing the mask coefficients ($k \times N$) and prototype map ($k \times h \times w$) via custom TensorRT GEMM plugins bypasses CPU-GPU memory bottlenecks.
- **Vulkan Acceleration**:
  - PIDNet and BiSeNet models deploy cleanly on embedded mobile GPUs via Vulkan shaders in Tencent NCNN.
- **Memory Footprint**:
  - Pre-allocate contiguous pinned host/device buffers for the maximum expected number of instances to prevent dynamic allocation spikes during live video processing.

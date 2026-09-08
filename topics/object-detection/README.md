---
title: Object Detection Playbook
tags:
  - computer-vision
  - object-detection
  - sota
  - yolo
  - transformers
  - tensorrt
updated: 2026-09-08
aliases:
  - Object Detection
---

# Object Detection Playbook

# Overview
Object Detection involves localizing visual concepts with 2D axis-aligned or oriented bounding boxes and classifying instances simultaneously. It forms the backbone of autonomous navigation, surveillance, industrial defect inspection, robotics, and augmented reality. The key challenge lies in balancing multi-scale localization precision, dense spatial clutter handling, and strict inference latency constraints.

Related notes: [[topics/object-segmentation/README|Object Segmentation]], [[topics/video-tracking/README|Video Tracking]], [[topics/gpu-deployment/README|GPU Deployment]].

## SOTA & Research
- **Recent Breakthroughs (2023–2026)**:
  - *YOLOv10: Real-Time End-to-End Object Detection* (Wang et al., 2024) - [arXiv:2405.14458](https://arxiv.org/abs/2405.14458): Eliminates NMS latency bottlenecks using consistent dual assignments for NMS-free training and inference.
  - *RT-DETR: DETRs Beat YOLOs on Real-time Object Detection* (Lv et al., 2023 / 2024) - [arXiv:2304.08069](https://arxiv.org/abs/2304.08069): First real-time end-to-end Transformer detector incorporating efficient hybrid encoders and uncertainty-minimal query selection.
  - *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection* (Liu et al., 2023 / 2024) - [arXiv:2303.05499](https://arxiv.org/abs/2303.05499): SOTA open-vocabulary detector unifying text-to-image cross-attention for zero-shot generalization.
  - *Co-DETR: Collaborative Hybrid Assignments for Transformer-based Object Detection* (Zong et al., 2023) - [arXiv:2211.12860](https://arxiv.org/abs/2211.12860): Overcame sparse query learning by introducing versatile collaborative auxiliary heads, achieving 66.0% mAP on COCO test-dev.

### Quantitative SOTA Benchmark Comparison (COCO val2017)
| Model Architecture | Parameters | FLOPs | AP (0.50:0.95) | TensorRT Latency (T4/A100 FP16) | NMS-Free |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv10-S** | 7.2 M | 21.6 G | 46.3% | 2.49 ms | Yes |
| **YOLOv10-X** | 29.5 M | 160.4 G | 54.4% | 10.70 ms | Yes |
| **RT-DETR-L** | 32.0 M | 110.0 G | 53.0% | 9.00 ms | Yes |
| **RT-DETR-X** | 67.0 M | 234.0 G | 54.8% | 13.80 ms | Yes |
| **Grounding DINO (Swin-T)** | 172.0 M | 340.0 G | 52.5% (Zero-Shot) | 78.00 ms | No |
| **Co-Deformable-DETR (Swin-L)** | 217.0 M | 1280.0 G | 66.0% | 145.00 ms | No |

## Architecture Alternatives & Trade-offs
| Architecture Family | Examples | Latency / Hardware Fit | Strengths | Drawbacks / Failure Modes |
| :--- | :--- | :--- | :--- | :--- |
| **NMS-Free Real-time CNN** | YOLOv10 | <5ms on GPU, runs on Edge TPU/FPGA | Zero post-processing NMS latency, deterministic inference | Sensitive to extreme scale variance without feature pyramid tuning |
| **Real-time Transformer** | RT-DETR, DINO-DETR | 8-15ms on modern GPU (TensorRT) | Global multi-scale attention, superior large-field context | Higher VRAM footprint during dynamic shape inference |
| **Open-Vocabulary Detectors** | Grounding DINO, OWLV2 | 50-200ms | Detects unannotated arbitrary concepts from natural language | Prohibitive computational cost for embedded robotics |

## Popular Repos & Integrations
- **[THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)**: Official implementation of YOLOv10 with pre-trained PyTorch checkpoints and ONNX/TensorRT export.
- **[lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)**: SOTA real-time Transformer detector repository with PyTorch and Paddle implementations.
- **[IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)**: Official open-set zero-shot object detector implementation.
- **[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)**: Industry-standard repository supporting YOLOv8, YOLOv9, and YOLOv10 workflows.
- **Tooling Integrations**:
  - **FiftyOne**: Inspect false positives, label noise, and mAP distribution per slice (e.g. bounding box area vs recall).
  - **Rerun**: Stream live inference boxes with confidence scores overlaid on 2D camera feeds.

## End-to-End Pipeline & Workarounds
1. **Pipeline Stages**:
   - Camera frame ingestion -> Letterbox resizing / normalization -> Tensor transfer to GPU -> FP16/INT8 forward pass -> NMS / Thresholding -> Coordinate restoration.
2. **Common Traps & Edge Cases**:
   - *Tiny Objects*: Downsampling stages in deep backbones destroy features of objects <16x16 pixels.
   - *Heavy Occlusion*: Non-Maximum Suppression (NMS) suppressing legitimate overlapping instances.
   - *Aspect Ratio Distortions*: Naive stretching degrades bounding box regressor performance.
3. **Engineering Workarounds**:
   - **SAHI (Slicing Aided Hyper Inference)**: Crop high-resolution frames into overlapping patches, infer locally, and merge using Non-Maximum Merging (NMM).
   - **NMS-Free Architectures**: Switch to RT-DETR or YOLOv10 consistent dual assignments to remove non-deterministic NMS kernels.
   - **Letterbox Padding**: Preserve aspect ratio with minimum grey-padding to reduce unnecessary computation.

## Deployment & Real-time Notes
- **TensorRT Optimization**:
  - Export PyTorch model to ONNX with dynamic or fixed batch shapes.
  - Compile with `trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 --int8`.
  - Replace custom PyTorch NMS with `EfficientNMS_TRT` plugin inside the ONNX graph for fused GPU post-processing.
- **Vulkan / Edge Acceleration**:
  - For mobile/embedded x86/ARM devices without NVIDIA GPUs, export to NCNN or ONNX Runtime with Vulkan Execution Provider.
- **Zero-Copy Memory**:
  - Ingest directly from V4L2/GStreamer into DMA-BUF or CUDA unified memory to eliminate host-to-device CPU overhead.

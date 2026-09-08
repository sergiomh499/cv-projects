---
title: Object Detection Playbook
tags:
  - computer-vision
  - object-detection
  - sota
  - yolo
  - detr
  - rf-detr
  - rt-detrv2
  - rt-detrv3
  - tensorrt
updated: 2026-09-08
aliases:
  - Object Detection
---

# Object Detection Playbook

# Overview
Object Detection localizes visual instances with 2D bounding boxes (axis-aligned or oriented) and assigns class probabilities to each. Modern production systems demand sub-10ms latency, zero-shot open-vocabulary capability, robust performance under dense clutter, and deterministic execution profiles without erratic post-processing delays.

Related notes: [[topics/object-segmentation/README|Object Segmentation]], [[topics/video-tracking/README|Video Tracking]], [[topics/gpu-deployment/README|GPU Deployment]].

---

## SOTA & Research (2023–2026 Breakthroughs)

1. **RF-DETR: Neural Architecture Search for Real-Time Detection Transformers** (Roboflow, ICLR 2026 / arXiv:2511.09554)
   - *Key Innovation*: Combines a frozen or fine-tuned DINOv2 backbone with weight-sharing Neural Architecture Search (NAS) to generate Pareto-optimal real-time detection architectures without retraining from scratch. Delivers superior small-object recall on Roboflow-100 benchmarks.
   - [Paper: arXiv:2511.09554](https://arxiv.org/abs/2511.09554) | [Official Code](https://github.com/roboflow/rf-detr)

2. **RT-DETRv2 & RT-DETRv3: Real-Time Detection Transformers** (Lv et al., 2024 / arXiv:2407.17140 & Xia et al., 2024 / arXiv:2409.08475)
   - *Key Innovation*: RT-DETRv2 eliminates non-standard operators (e.g., `grid_sample`) for seamless TensorRT/Vulkan deployment. RT-DETRv3 introduces *Hierarchical Dense Positive Supervision* to solve the slow convergence and gradient sparsity inherent to traditional 1-to-1 Hungarian bipartite matching.
   - [RT-DETRv2 Paper](https://arxiv.org/abs/2407.17140) | [RT-DETRv3 Paper](https://arxiv.org/abs/2409.08475) | [Official Code](https://github.com/lyuwenyu/RT-DETR)

3. **YOLOv10: Real-Time End-to-End Object Detection** (Wang et al., Tsinghua, 2024)
   - *Key Innovation*: Eliminates Non-Maximum Suppression (NMS) during inference using consistent dual assignments (one-to-many for rich training gradients, one-to-one for deterministic NMS-free inference). Employs rank-guided block design and large-kernel separable convolutions.
   - [Paper: arXiv:2405.14458](https://arxiv.org/abs/2405.14458) | [Official Code](https://github.com/THU-MIG/yolov10)

4. **Grounding DINO & DINOv2/v3 Zero-Shot Detectors** (Liu et al., 2023 / 2024)
   - *Key Innovation*: Cross-modality feature fusion between image patch tokens and text tokens across three stages: neck enhancer, cross-attention queries, and early text-guided selection.
   - [Paper: arXiv:2303.05499](https://arxiv.org/abs/2303.05499) | [Official Code](https://github.com/IDEA-Research/GroundingDINO)

---

### Quantitative SOTA Benchmark Comparison (COCO val2017)
| Architecture Model | Parameters | FLOPs | AP (0.50:0.95) | TensorRT Latency (FP16 ms) | NMS Required | Primary Open License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RF-DETR-Base** | 28.4 M | 86.0 G | 53.8% | 5.20 ms | No | Apache-2.0 |
| **RT-DETRv3-L** | 31.8 M | 108.0 G | 54.3% | 8.80 ms | No | Apache-2.0 |
| **RT-DETRv2-X** | 67.0 M | 234.0 G | 55.1% | 13.60 ms | No | Apache-2.0 |
| **YOLOv10-S** | 7.2 M | 21.6 G | 46.3% | 2.49 ms | No | AGPL-3.0 / Commercial |
| **YOLOv10-X** | 29.5 M | 160.4 G | 54.4% | 10.70 ms | No | AGPL-3.0 / Commercial |
| **Grounding DINO (Swin-T)**| 172.0 M | 340.0 G | 52.5% (Zero-Shot)| 78.00 ms | Yes | Apache-2.0 |
| **Co-Deformable-DETR (Swin-L)**| 217.0 M | 1280.0 G | 66.0% (SOTA Heavy)| 145.00 ms | Yes | Apache-2.0 |

---

## Commercial Usability & License Audit
- **Permissive (Safe for Closed-Source Commercial Products)**:
  - **Apache-2.0**: `roboflow/rf-detr`, `lyuwenyu/RT-DETR` (v1/v2), `clxia12/RT-DETRv3`, `IDEA-Research/GroundingDINO`, `open-mmlab/mmdetection`. Permissible for proprietary SaaS and embedded hardware binaries without exposing source code.
  - **MIT License**: `AlexeyAB/darknet`, `WongKinYiu/YOLO` (academic variants like YOLOv7).
- **Copyleft (Caution Required for Commercial Products)**:
  - **AGPL-3.0 (Ultralytics YOLOv8 / YOLOv10)**: If your backend service communicates over a network with users (SaaS), AGPL-3.0 mandates that you open-source your entire caller codebase under AGPL-3.0 unless an enterprise commercial license is purchased from Ultralytics.
  - *Recommendation for Commercial Startups*: Standardize on **RF-DETR** or **RT-DETRv2/v3** (Apache-2.0) to eliminate NMS latency and prevent copyleft licensing contamination.

---

## Architecture Alternatives & Trade-offs
| Architecture Family | Examples | Latency / Hardware Fit | Strengths | Drawbacks / Failure Modes |
| :--- | :--- | :--- | :--- | :--- |
| **NMS-Free Real-time Transformer** | RF-DETR, RT-DETRv2/v3 | 5-12ms on GPU | Zero post-processing jitter, global multi-scale context | Higher memory footprint during dynamic tensor batching |
| **NMS-Free Real-time CNN** | YOLOv10 | 2-10ms on GPU / Edge TPU | Extremely fast, lightweight mobile kernels | Prone to false positives on repetitive texture backgrounds |
| **Open-Vocabulary Foundation Detectors** | Grounding DINO, OWLV2 | 50-180ms on GPU | Detects arbitrary unannotated items from natural text prompts | High VRAM and latency; unsuitable for 60+ FPS robotic loops |

---

## Popular Repos & Integrations
- **[roboflow/rf-detr](https://github.com/roboflow/rf-detr)**: Official repository for RF-DETR (Apache-2.0 license), supporting pre-trained checkpoints and NAS export.
- **[lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)**: The official RT-DETR and RT-DETRv2 repository (Apache-2.0).
- **[clxia12/RT-DETRv3](https://github.com/clxia12/RT-DETRv3)**: Official RT-DETRv3 with hierarchical dense positive supervision.
- **[THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)**: Official YOLOv10 PyTorch and ONNX implementation (AGPL-3.0).
- **[IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)**: SOTA open-vocabulary zero-shot detector (Apache-2.0).
- **Tooling Integrations**:
  - **FiftyOne**: Inspect false positives, label noise, and mAP distribution per slice (e.g. bounding box area vs recall).
  - **Rerun**: Stream live inference boxes with confidence scores overlaid on 2D camera feeds.

---

## End-to-End Pipeline & Workarounds
1. **Pipeline Stages**:
   - Camera frame ingestion -> Letterbox resizing / normalization -> Tensor transfer to GPU -> FP16/INT8 forward pass -> Bounding box coordinate restoration.
2. **Common Traps & Edge Cases**:
   - *Tiny Objects (<16x16 px)*: Deep backbone downsampling obliterates spatial features.
   - *High-Density Packing*: Traditional NMS suppresses valid adjacent objects sharing IoU > 0.5.
3. **Engineering Workarounds**:
   - **SAHI (Slicing Aided Hyper Inference)**: Slice ultra-high-resolution images into overlapping tiles with a sliding window, infer on tiles, and merge predictions with Non-Maximum Merging.
   - **Switching to NMS-Free (RT-DETRv3 / RF-DETR)**: Eliminates non-deterministic NMS execution time variance entirely.

---

## Deployment & Real-time Notes
- **TensorRT Optimization**:
  - Export PyTorch model to ONNX with fixed shapes: `torch.onnx.export(model, dummy_input, "detector.onnx", opset_version=17)`.
  - Build TensorRT engine: `trtexec --onnx=detector.onnx --saveEngine=detector.engine --fp16`.
- **Vulkan / Edge Acceleration**:
  - For mobile/embedded x86/ARM devices without NVIDIA GPUs, export to NCNN or ONNX Runtime with Vulkan Execution Provider.
- **Zero-Copy Memory**:
  - Ingest directly from V4L2/GStreamer into DMA-BUF or CUDA unified memory to eliminate host-to-device CPU overhead.

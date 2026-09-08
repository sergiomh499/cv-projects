---
title: Object Detection Playbook & Technology Index
tags:
  - computer-vision
  - object-detection
  - index
  - sota
  - models
  - playbooks
updated: 2026-09-08
aliases:
  - Object Detection Playbook
---

# Object Detection: Domain Playbook & In-Depth Technology Index

# Overview
Object Detection localizes visual instances with 2D bounding boxes (axis-aligned or oriented) and assigns class probabilities to each. Modern production systems demand sub-10ms latency, zero-shot open-vocabulary capability, robust performance under dense clutter, and deterministic execution profiles without erratic post-processing delays.

This document serves as the **central domain brief and master index** linking to dedicated architectural guides, model breakdowns, and the historical evolution of the field.

Related notes: [[topics/object-segmentation/README|Object Segmentation]], [[topics/video-tracking/README|Video Tracking]], [[topics/gpu-deployment/README|GPU Deployment]].

---

## 🧭 In-Depth Technology & Model Directory

Explore the dedicated standalone guides for each primary model family:

```text
topics/object-detection/
├── README.md               # Master Playbook & Technology Index (This file)
├── EVOLUTION.md            # Complete Historical Evolution: Haar/HOG -> R-CNN -> YOLOv1-YOLO26 -> DETR
└── models/
    ├── rf-detr.md          # RF-DETR: NAS-optimized Real-Time Transformers with DINOv2 (Apache-2.0)
    ├── rt-detr.md          # RT-DETRv2 & RT-DETRv3: Hybrid Encoders & Hierarchical Supervision (Apache-2.0)
    ├── yolo-lineage.md     # Modern YOLO Lineage: YOLOv10, YOLO11, and YOLO26 (AGPL-3.0 / Commercial)
    └── grounding-dino.md   # Grounding DINO: Open-Vocabulary Multi-Modal Zero-Shot Detection (Apache-2.0)
```

| Technology / Model | Primary Innovation | Ideal Production Use Case | License | In-Depth Guide Link |
| :--- | :--- | :--- | :---: | :---: |
| **RF-DETR** | Weight-sharing NAS over DINOv2 backbone | Edge & server real-time detection, high small-object recall | **Apache-2.0** | [Read RF-DETR Guide](models/rf-detr.md) <br> `[[topics/object-detection/models/rf-detr\|RF-DETR Guide]]` |
| **RT-DETRv2 / v3**| NMS-free, CCFM hybrid encoder, dense positive supervision | Commercial robotics, autonomous vehicles, NMS-free loops | **Apache-2.0** | [Read RT-DETR Guide](models/rt-detr.md) <br> `[[topics/object-detection/models/rt-detr\|RT-DETR Guide]]` |
| **YOLOv10 / 11 / 26**| Consistent dual assignments, C3k2/C2PSA blocks, MuSGD | High-speed surveillance, embedded edge microprocessors | **AGPL-3.0** | [Read YOLO Guide](models/yolo-lineage.md) <br> `[[topics/object-detection/models/yolo-lineage\|YOLO Guide]]` |
| **Grounding DINO** | 3-stage visual-language cross-attention fusion | Zero-shot open-vocabulary detection, auto-annotation | **Apache-2.0** | [Read Grounding DINO Guide](models/grounding-dino.md) <br> `[[topics/object-detection/models/grounding-dino\|Grounding DINO Guide]]` |
| **Lineage & History**| Evolution from sliding windows to set prediction | Didactic reference and architectural intuition | N/A | [Read Evolution Guide](EVOLUTION.md) <br> `[[topics/object-detection/EVOLUTION\|Detection Evolution]]` |

---

## SOTA & Research (2023–2026 Breakthroughs)

1. **RF-DETR: Neural Architecture Search for Real-Time Detection Transformers** (Roboflow, ICLR 2026 / arXiv:2511.09554)
   - [Paper: arXiv:2511.09554](https://arxiv.org/abs/2511.09554) | [Official Code](https://github.com/roboflow/rf-detr) | [Deep Dive](models/rf-detr.md)
2. **RT-DETRv2 & RT-DETRv3: Real-Time Detection Transformers** (Lv et al., 2024 & Xia et al., 2024)
   - [RT-DETRv2 Paper](https://arxiv.org/abs/2407.17140) | [RT-DETRv3 Paper](https://arxiv.org/abs/2409.08475) | [Official Code](https://github.com/lyuwenyu/RT-DETR) | [Deep Dive](models/rt-detr.md)
3. **YOLOv10 & YOLO26: End-to-End Real-Time Detectors** (Tsinghua 2024 / Ultralytics 2026)
   - [Paper: arXiv:2405.14458](https://arxiv.org/abs/2405.14458) | [Official Code](https://github.com/ultralytics/ultralytics) | [Deep Dive](models/yolo-lineage.md)
4. **Grounding DINO: Marrying DINO with Grounded Language Pre-Training** (Liu et al., 2023 / 2024)
   - [Paper: arXiv:2303.05499](https://arxiv.org/abs/2303.05499) | [Official Code](https://github.com/IDEA-Research/GroundingDINO) | [Deep Dive](models/grounding-dino.md)

---

### Quantitative SOTA Benchmark Comparison (COCO val2017)
| Architecture Model | Parameters | FLOPs | AP (0.50:0.95) | TensorRT Latency (FP16 ms) | NMS Required | Primary Open License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RF-DETR-Base** | 28.4 M | 86.0 G | 53.8% | 5.20 ms | No | Apache-2.0 |
| **RT-DETRv3-L** | 31.8 M | 108.0 G | 54.3% | 8.80 ms | No | Apache-2.0 |
| **RT-DETRv2-X** | 67.0 M | 234.0 G | 55.1% | 13.60 ms | No | Apache-2.0 |
| **YOLO26-S** | 7.4 M | 22.1 G | 47.9% | 2.30 ms | No | AGPL-3.0 / Commercial |
| **YOLOv10-X** | 29.5 M | 160.4 G | 54.4% | 10.70 ms | No | AGPL-3.0 / Commercial |
| **Grounding DINO (Swin-T)**| 172.0 M | 340.0 G | 52.5% (Zero-Shot)| 78.00 ms | Yes | Apache-2.0 |
| **Co-Deformable-DETR (Swin-L)**| 217.0 M | 1280.0 G | 66.0% (SOTA Heavy)| 145.00 ms | Yes | Apache-2.0 |

---

## Commercial Usability & License Audit
- **Permissive (Safe for Closed-Source Commercial Products)**:
  - **Apache-2.0**: `roboflow/rf-detr`, `lyuwenyu/RT-DETR` (v1/v2), `clxia12/RT-DETRv3`, `IDEA-Research/GroundingDINO`. Safe for proprietary software and embedded binaries without revealing source code.
- **Copyleft (Caution Required for Commercial Products)**:
  - **AGPL-3.0 (Ultralytics YOLOv8 / YOLO11 / YOLO26)**: Mandates open-sourcing client backends if distributed or hosted over a network unless an enterprise license is purchased.
  - *Recommendation*: For commercial projects, standardize on **RF-DETR** or **RT-DETRv2/v3** (Apache-2.0).

---

## Architecture Alternatives & Trade-offs
| Architecture Family | Examples | Latency / Hardware Fit | Strengths | Drawbacks / Failure Modes |
| :--- | :--- | :--- | :--- | :--- |
| **NMS-Free Real-time Transformer** | RF-DETR, RT-DETRv2/v3 | 5-12ms on GPU | Zero post-processing jitter, global multi-scale context | Higher memory footprint during dynamic tensor batching |
| **NMS-Free Real-time CNN** | YOLOv10, YOLO26 | 2-10ms on GPU / Edge TPU | Extremely fast, lightweight mobile kernels | Prone to false positives on repetitive texture backgrounds |
| **Open-Vocabulary Foundation Detectors** | Grounding DINO, OWLV2 | 50-180ms on GPU | Detects arbitrary unannotated items from natural text prompts | High VRAM and latency; unsuitable for 60+ FPS robotic loops |

---

## Popular Repos & Integrations
- **[roboflow/rf-detr](https://github.com/roboflow/rf-detr)**: Official RF-DETR repository (Apache-2.0).
- **[lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)**: Official RT-DETR / RT-DETRv2 repository (Apache-2.0).
- **[clxia12/RT-DETRv3](https://github.com/clxia12/RT-DETRv3)**: Official RT-DETRv3 with dense positive supervision (Apache-2.0).
- **[THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)**: Official YOLOv10 implementation (AGPL-3.0).
- **[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)**: Official Ultralytics repository covering YOLO11 & YOLO26 (AGPL-3.0).
- **[IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)**: SOTA open-vocabulary zero-shot detector (Apache-2.0).
- **Tooling Integrations**:
  - **FiftyOne**: Inspect false positives, label noise, and mAP distribution per slice.
  - **Rerun**: Stream live inference boxes with confidence scores overlaid on 2D camera feeds.

---

## End-to-End Pipeline & Workarounds
1. **Pipeline Stages**:
   - Camera frame ingestion -> Letterbox resizing / normalization -> Tensor transfer to GPU -> FP16/INT8 forward pass -> Coordinate restoration.
2. **Common Traps & Edge Cases**:
   - *Tiny Objects (<16x16 px)*: Deep backbone downsampling obliterates spatial features.
   - *High-Density Packing*: Traditional NMS suppresses valid adjacent objects sharing IoU > 0.5.
3. **Engineering Workarounds**:
   - **SAHI (Slicing Aided Hyper Inference)**: Slice ultra-high-resolution images into overlapping tiles, infer locally, and merge using Non-Maximum Merging.
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

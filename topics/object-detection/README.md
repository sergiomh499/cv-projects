# Object Detection Playbook

# Overview
Object Detection involves localizing visual concepts with 2D axis-aligned or oriented bounding boxes and classifying instances simultaneously. It forms the backbone of autonomous navigation, surveillance, industrial defect inspection, robotics, and augmented reality. The key challenge lies in balancing multi-scale localization precision, dense spatial clutter handling, and strict inference latency constraints.

## SOTA & Research
- **Seminal & Modern Papers**:
  - *Faster R-CNN* (Ren et al., 2015): Canonical two-stage architecture using Region Proposal Networks (RPN).
  - *YOLOv1-v10* (Redmon et al., 2016 -> Wang et al., 2024): Evolution of single-stage real-time detectors from anchor-based grid splits to anchor-free NMS-free dual-label assignments.
  - *DETR* (Carion et al., 2020) & *Deformable DETR* (Zhu et al., 2020): Transformer-based end-to-end detection reformulating detection as direct bipartite matching with set prediction loss.
  - *RT-DETR* (Lv et al., 2023): First real-time end-to-end Transformer detector eliminating post-processing NMS latency.
  - *DINO* (Zhang et al., 2022): State-of-the-art contrastive denoising training for DETR-like detectors.
- **Evaluation Benchmarks & Metrics**:
  - COCO (mAP@0.50:0.95, mAP@0.50, mAP_s, mAP_m, mAP_l).
  - Pascal VOC, Objects365, OpenImages V7.
  - Latency: End-to-end frame processing time including pre-processing, forward pass, and NMS/bipartite post-processing (FPS).

## Architecture Alternatives & Trade-offs
| Architecture Family | Examples | Latency / Hardware Fit | Strengths | Drawbacks / Failure Modes |
| :--- | :--- | :--- | :--- | :--- |
| **Real-time CNN (Single-Stage)** | YOLOv8/v9/v10, YOLOX | <5ms on GPU, runs on Edge TPU/FPGA | Extremely fast, simple export, mature quantization pipelines | Susceptible to extreme dense clustering, NMS bottlenecks in v8/v9 |
| **Real-time Transformer** | RT-DETR, Co-DETR | 8-15ms on modern GPU (TensorRT) | NMS-free, superior global context, higher small-object AP | Higher memory footprint during dynamic resolution scaling |
| **Two-Stage / Multi-Stage** | Faster R-CNN, Cascade R-CNN | >30ms (Server GPUs) | Robust bounding-box localization in complex scenes | Poor edge performance, multi-step proposal overhead |
| **Open-Vocabulary Detectors** | Grounding DINO, OWLV2 | 50-200ms | Zero-shot detection based on arbitrary natural language prompts | Unsuitable for sub-10ms real-time control loops |

## Popular Repos & Integrations
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)**: Industry-standard repository for training, validation, ONNX/TensorRT export.
- **[MMDetection](https://github.com/open-mmlab/mmdetection)**: Comprehensive modular toolbox containing 100+ architectures (Cascade, Deformable DETR, DINO).
- **[RT-DETR (PaddleDetection / Lyuwen Yu PyTorch)](https://github.com/lyuwenyu/RT-DETR)**: SOTA real-time transformer detector implementation.
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

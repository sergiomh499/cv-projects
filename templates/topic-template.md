# {{TOPIC_TITLE}} Playbook

# Overview
Brief executive summary of the vision/perception domain, common mission-critical constraints, and why this topic is essential in production AI/CV.

## SOTA & Research
- **Seminal & Modern Papers**: Foundational works and recent breakthroughs.
- **Evaluation Benchmarks & Metrics**: Standard datasets (e.g., COCO, nuScenes, KITTI, ImageNet) and metrics (mAP, mIoU, MOTA, ADD-S, latency/throughput).
- **Emerging Trends**: Transformers, diffusion/generative priors, self-supervised pre-training, or edge quantizations.

## Architecture Alternatives & Trade-offs
| Architecture / Paradigm | Pros | Cons / Latency Bottlenecks | Primary Use-Case |
| :--- | :--- | :--- | :--- |
| Option A (e.g., Heavy Transformer) | High accuracy, multi-scale attention | High VRAM, slow on edge | Server-side, offline research |
| Option B (e.g., Lightweight CNN/Mamba) | Sub-millisecond latency, low thermal | Sensitive to heavy occlusion | Edge, mobile, microcontrollers |

## Popular Repos & Integrations
- **Official / Community Repos**: GitHub links, license notes, maturity assessment.
- **Framework Ecosystem**: PyTorch, Hugging Face, Ultralytics, MMDetection, or OpenCV.
- **Dataset & Visualization Tooling**:
  - **FiftyOne**: Curation, error analysis, false-positive mining.
  - **Rerun**: Temporal logging, real-time 2D/3D stream visualization.

## End-to-End Pipeline & Workarounds
1. **Data Ingestion & Augmentation**:
   - Pipeline stages, caching, sensor calibration.
2. **Common Traps & Edge Cases**:
   - Class imbalance, motion blur, lighting changes, small object decay.
3. **Engineering Workarounds**:
   - Tiling (SAHI), temporal smoothing, optical flow assist, synthetic data augmentations.

## Deployment & Real-time Notes
- **Target Runtimes**: TensorRT (CUDA), Vulkan (Kompute/NCNN), ONNX Runtime, FPGA (Vitis AI / FINN).
- **Quantization & Precision**: FP32 vs. FP16 vs. INT8 PTQ/QAT trade-offs.
- **Real-Time Guarantees**: Zero-copy memory pipelines, thread scheduling, latency budgets (e.g., <33ms for 30 FPS, <10ms for robotics).

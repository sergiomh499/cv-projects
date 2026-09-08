---
id: "{{id}}"
title: "{{TOPIC_TITLE}}"
type: topic
domain: "{{domain}}"
tags:
  - topic
  - computer-vision
  - "{{domain_tag}}"
status: evergreen
updated: YYYY-MM-DD
aliases:
  - "{{TOPIC_TITLE}}"
---

# {{TOPIC_TITLE}}

> [!NOTE] Domain Context
> Brief executive summary of the vision/perception domain, common mission-critical constraints, and why this topic is essential in production AI/CV. Link to the parent MOC: [[{{domain_path}}/{{domain}}-moc|{{domain}} Map of Content]].

## SOTA & Research

- **Seminal & Modern Papers**: Foundational works and recent breakthroughs.
- **Evaluation Benchmarks & Metrics**: Standard datasets (e.g., COCO, nuScenes, KITTI, ImageNet) and metrics (mAP, mIoU, MOTA, ADD-S, latency/throughput).
- **Emerging Trends**: Transformers, diffusion/generative priors, self-supervised pre-training, edge quantization.

## Architecture Alternatives & Trade-offs

> [!NOTE] Architecture Notes
> Choose the paradigm based on target hardware budget and latency SLA. Transformer variants dominate accuracy; lightweight CNNs/Mamba variants dominate edge.

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

> [!WARNING] Common Failure Modes
> Class imbalance, motion blur, lighting changes, and small-object decay are the top production failure drivers. Validate your augmentation pipeline against each before deployment.

1. **Data Ingestion & Augmentation**:
   - Pipeline stages, caching, sensor calibration.
2. **Common Traps & Edge Cases**:
   - Class imbalance, motion blur, lighting changes, small object decay.
3. **Engineering Workarounds**:
   - Tiling (SAHI), temporal smoothing, optical flow assist, synthetic data augmentations.

## Deployment & Real-Time Notes

> [!TIP] Zero-Copy & Quantization
> Prefer CUDA pinned memory + DMA transfers to eliminate host-device copies. INT8 PTQ on TensorRT typically yields 2–4× speedup at <1% mAP drop when calibrated on a representative dataset. For sub-byte quantization (FP4/INT4), use QAT.

- **Target Runtimes**: TensorRT (CUDA), Vulkan (Kompute/NCNN), ONNX Runtime, FPGA (Vitis AI / FINN).
- **Quantization & Precision**: FP32 vs. FP16 vs. INT8 PTQ/QAT trade-offs.
- **Real-Time Guarantees**: Zero-copy memory pipelines, thread scheduling, latency budgets (e.g., <33ms for 30 FPS, <10ms for robotics).

## Related Notes

```dataview
TABLE type AS "Note Type", updated AS "Last Updated", status AS "Status"
WHERE contains(domain, "{{domain}}")
SORT file.name ASC
```

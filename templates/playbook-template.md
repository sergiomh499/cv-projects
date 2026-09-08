---
id: "{{id}}"
title: "{{domain}} - Production Pipeline & Engineering Workarounds"
type: production-playbook
domain: "{{domain}}"
tags:
  - playbook
  - engineering
  - production
  - workarounds
  - "{{domain_tag}}"
status: evergreen
updated: YYYY-MM-DD
aliases:
  - "{{domain}} Playbook"
---

# 🛠️ {{domain}}: Production Pipeline, Engineering Traps & Workarounds

> [!NOTE] Guide Scope
> A practitioner's guide to building, tuning, and shipping end-to-end {{domain}} systems in mission-critical environments. Link back to parent MOC: [[{{domain_path}}/{{domain}}-moc|{{domain}} Map of Content]].

## 1. End-to-End Pipeline Stages

```mermaid
flowchart LR
    Ingest[Sensor Ingest & DMA Buffers] --> Preproc[Preprocessing & Normalization]
    Preproc --> Infer[Asynchronous Forward Pass]
    Infer --> Postproc[Fused Postprocessing]
    Postproc --> Dispatch[Lock-free Zero-Copy Dispatch]
```

> [!TIP] Zero-Copy Pipeline Design
> Allocate sensor buffers with `cudaMallocHost` (pinned memory) at startup. Use CUDA streams to pipeline DMA transfer, inference, and postprocessing concurrently. Target <5% CPU utilization from the dispatch thread on Jetson-class hardware.

---

## 2. Common Traps, Edge Cases & Hardware Ceilings

> [!WARNING] Critical Engineering Gotchas
> These are the failure modes that cause silent accuracy regressions or production outages. Validate each against your specific hardware and sensor configuration before shipping.

- **Edge Case 1**: ...
- **Edge Case 2**: ...
- **Hardware Bottlenecks**: PCIe bus contention, thread priority inversion, DRAM bandwidth saturation on unified-memory SoCs (Jetson/Apple Silicon).
- **Thermal Throttling**: Sustained inference on embedded devices triggers governor-controlled clock reduction; profile at thermal steady-state, not cold-start.

---

## 3. Battle-Tested Engineering Workarounds

> [!TIP] Performance Optimizations
> Profile before optimizing. Use `nsys` / `nvtx` for CUDA kernel timelines and `perf` for CPU-bound stages. Tiling (SAHI) and temporal smoothing add latency — gate them behind config flags so they are disabled in latency-constrained paths.

- **Workaround 1**: ...
- **Workaround 2**: ...

---

## 4. Edge & Embedded Deployment Recipes

> [!WARNING] Embedded Hardware Constraints
> INT8 calibration on Jetson requires a representative calibration dataset (≥500 images from your deployment distribution). Calibrating on COCO for a domain-specific model causes systematic quantization error.

- **TensorRT (CUDA)**:
- **Vulkan / NCNN (Cross-Platform Mobile/Embedded)**:
- **Zero-Copy Memory Transport**:

---

## 5. Observability & Failure Recovery

> [!NOTE] Production Monitoring
> Instrument inference latency (p50/p95/p99), model confidence score distributions, and sensor health signals. Sudden distribution shift in confidence scores is the fastest indicator of sensor degradation or domain shift before accuracy metrics degrade.

- **Latency Budgets**: Define per-stage SLOs (e.g., <33ms end-to-end for 30 FPS, <10ms for robotics control loops).
- **Fallback Strategy**: ...
- **Online Drift Detection**: ...

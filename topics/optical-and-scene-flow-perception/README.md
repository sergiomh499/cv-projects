---
title: "Optical Flow & Scene Flow Master Playbook"
type: production-playbook
domain: Optical Flow & Scene Flow Perception
tags:
  - optical-flow
  - scene-flow
  - raft
  - gmflow
  - unimatch
  - motion-estimation
updated: 2026-09-08
aliases:
  - Optical Flow Playbook
  - Scene Flow Playbook
---

# 🌊 Optical Flow & Scene Flow Master Playbook

# Overview
Optical Flow ($2\text{D}$ pixel displacement vectors) and Scene Flow ($3\text{D}$ velocity vectors) provide dense motion fields across dynamic video and point-cloud streams. Rather than tracking isolated bounding boxes, flow fields capture full continuous motion dynamics—enabling sub-pixel video stabilization, independent moving object (IMO) segmentation, autonomous collision avoidance, and non-rigid surgical tracking.

Related notes: [[topics/optical-and-scene-flow-perception/00-optical-and-scene-flow-perception-moc|Optical Flow MOC]], [[topics/video-tracking/00-video-tracking-moc|Video Tracking MOC]].

## SOTA & Research
- **Global Matching Transformers**:
  - **GMFlow & UniMatch (CVPR / TPAMI 2024–2026)**: Replacing multi-iteration recurrent GRU updates with single-pass transformer cross-attention, enabling single-shot optical and scene flow at $>60\,\text{FPS}$.
- **RAFT (Recurrent All-Pairs Field Transforms)**:
  - Constructing $4\text{D}$ multi-scale correlation volumes followed by gated recurrent ConvGRU updates.
- **Self-Supervised 3D Scene Flow**:
  - Training flow estimators directly on unannotated autonomous driving LiDAR datasets (nuScenes, Waymo) via Chamfer and cycle-consistency objectives.

## Architecture Alternatives & Trade-offs

| Architecture | Paradigm | 2D End-Point-Error (EPE) | 1080p Inference Time | Hardware Requirements | Primary Deployment Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GMFlow / UniMatch** | Cross-Attention Matching | **0.85 px (Sintel Final)** | $\sim 15\,\text{ms}$ | 1x RTX 4080 / Orin | Production real-time perception (>60 FPS) |
| **RAFT (32 iters)** | Recurrent All-Pairs Correlation | 0.90 px | $\sim 85\,\text{ms}$ | 1x RTX 3090 / 4090 | Offline high-accuracy video editing |
| **FlowNet2** | Cascaded Encoder-Decoder CNN | 1.85 px | $\sim 28\,\text{ms}$ | 1x RTX 3080 | Legacy edge architectures |
| **Classical Farnebäck** | Quadratic Polynomial Expansion | 3.50 px | $\mathbf{<4\,\text{ms}}$ | CPU Multi-core | Low-power microcontrollers & drones |

## Popular Repos & Integrations
- `princeton-vl/RAFT`: **RAFT**: Official PyTorch implementation of Recurrent All-Pairs Field Transforms for Optical Flow.
- `haofeixu/gmflow`: **GMFlow**: SOTA global matching optical flow and stereo matching transformer.
- `haofeixu/unimatch`: **UniMatch**: Unified cross-attention architecture for Optical Flow, Stereo, and Depth Estimation.

## End-to-End Pipeline & Workarounds

```mermaid
flowchart LR
    FrameT[Image t] --> Model[GMFlow / RAFT Backbone]
    FrameT1[Image t+1] --> Model
    Model --> Flow[Dense Vector Field (u, v)]
    Flow --> FWD_BWD[Forward-Backward Consistency Filter]
    FWD_BWD --> EgoMotion[Epipolar Decoupling & IMU Subtraction]
    EgoMotion --> CollisionAlert[Independent Dynamic Moving Obstacle Alert]
```

### Production Workarounds for Optical Flow:
1. **Handling Sudden Illumination Flashes & Headlight Glare**:
   - Direct headlights completely violate the Brightness Constancy Assumption, causing optical flow vectors to explode radially outwards. **Workaround**: Train feature extractors with **Census Transform** or DINOv2 self-supervised patch tokens, which are robust to monotonic illumination shifts.
2. **Eliminating Motion Drift in Video Stabilization**:
   - Integrating dense optical flow vectors over hundreds of frames causes cumulative coordinate drift. **Workaround**: Anchor flow vectors periodically (every 10 frames) to sparse feature keypoints (SuperPoint / LightGlue) locked to the static background.

## Deployment & Real-Time Notes
- **TensorRT Optimization**: Replace dynamic ConvGRU loops with fixed-iteration static execution graphs ($N = 8$ iterations rather than 32) when converting RAFT to ONNX/TensorRT, reducing engine build latency from minutes to seconds and guaranteeing bounded execution times.

---
title: "SAM 2 & SAM 2.1: Segment Anything in Images and Videos"
type: model-deep-dive
tasks:
  - instance-segmentation
  - video-tracking
  - panoptic-segmentation
  - promptable-segmentation
architecture_class: Foundation Model (Hierarchical Vision Transformer)
primary_license: Apache-2.0
commercial_use: true
official_repo: https://github.com/facebookresearch/sam2
paper_url: https://arxiv.org/abs/2408.00714
tags:
  - model
  - foundation-model
  - sam2
  - segmentation
  - video-tracking
  - sota
updated: 2026-09-08
aliases:
  - SAM 2
  - SAM 2.1
  - Segment Anything Model 2
---

# 🔬 SAM 2 & SAM 2.1: Segment Anything in Images and Videos

## 1. Executive Brief & Significance
**SAM 2 & SAM 2.1** (Ravi et al., Meta FAIR, 2024 / 2025) unify image and streaming video perception into a single foundation model. Prior foundation segmentation (SAM 1) was confined to static images and suffered from high memory overhead (~100ms per image on a high-end GPU), making real-time interactive tracking impossible.

SAM 2 introduces a **hierarchical Hiera image/video encoder** coupled with a **streaming spatial-temporal memory bank**. It enables promptable zero-shot image segmentation, dense instance masking, and real-time video mask propagation at **44 FPS**, serving as a unified backbone across:
- [[topics/object-segmentation/00-object-segmentation-moc|Object Segmentation MOC]] (Promptable & Instance Masks)
- [[topics/video-tracking/00-video-tracking-moc|Video Tracking MOC]] (Zero-Shot Video Object Tracking & Re-identification)

```mermaid
flowchart TD
    Video[Video Frames / Streaming Ingestion] --> Hiera[Hiera Multiscale Image/Video Encoder]
    Hiera --> Feat[Frame Embeddings]
    Prompt[Point / Box / Mask Prompts] --> PromptEnc[Prompt Encoder]
    MemoryBank[(Streaming Memory Bank: Past Frame Features & Object Pointers)] --> MemAttn[Memory Cross-Attention]
    Feat --> Decoder[Two-Way Lightweight Transformer Decoder]
    PromptEnc --> Decoder
    MemAttn --> Decoder
    Decoder --> Out[Real-Time Masks + Quality Estimation at 44 FPS]
    Out --> MemoryBank
```

---

## 2. Core Architectural Mechanics

### A. Hierarchical Hiera Backbone
Unlike SAM 1's standard non-hierarchical ViT, SAM 2 adopts **Hiera**—a hierarchical vision transformer that produces multi-scale feature maps ($1/4, 1/8, 1/16, 1/32$). Hiera uses windowed local attention without complicated spatial shift operations, significantly reducing forward-pass latency and VRAM utilization.

### B. Streaming Spatial-Temporal Memory Architecture
When tracking objects across continuous video frames, SAM 2 maintains three components in memory:
1. **Spatial Feature Memory**: A FIFO queue holding the spatial embeddings of the recent $N$ frames (typically $N=6$ or $N=8$).
2. **Object Pointer Memory**: Compact vector tokens summarizing object presence and high-level identity per object instance across past frames.
3. **Prompted Conditioning Memory**: Uncompressed spatial representations of frames where the user provided explicit interactive corrections (clicks/boxes).

During inference on frame $t$, the decoder queries the memory bank using cross-attention blocks, allowing the model to smoothly recover object identity after multi-second occlusions and camera cuts without full model re-initialization.

---

## 3. Quantitative SOTA Benchmark Profile

| Model Variant | Backbone | SA-V Video J&F (Tracking) | Zero-Shot Image AP (COCO) | Latency (FP16 ms, A100) | Throughput (FPS) | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SAM 2.1-Hiera-Tiny** | Hiera-T | 71.5 | 42.1% | 14.7 ms | 68.0 FPS | Apache-2.0 |
| **SAM 2.1-Hiera-Small**| Hiera-S | 73.2 | 44.5% | 20.4 ms | 49.0 FPS | Apache-2.0 |
| **SAM 2.1-Hiera-Base+**| Hiera-B+ | 75.0 | 46.8% | 22.8 ms | 43.8 FPS | Apache-2.0 |
| **SAM 2.1-Hiera-Large**| Hiera-L | 76.2 | 48.9% | 34.0 ms | 29.4 FPS | Apache-2.0 |

---

## 4. Engineering Implementation & Deployment Recipe

### A. Minimal Inference Snippet (Video & Image)
```python
import torch
from sam2.build_sam import build_sam2_video_predictor

# Load architecture and pre-trained weights
checkpoint = "checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda")

# Initialize streaming session on video directory
inference_state = predictor.init_state(video_path="video_frames_dir/")

# Add prompt click (frame 0, point coordinates [x, y], label 1=positive)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=[[500, 375]],
    labels=[1],
)

# Propagate across video in real-time
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
    # Mask is available at 49 FPS
```

### B. TensorRT FP16 Compilation
```bash
# Export the lightweight prompt decoder to ONNX
python tools/export_decoder_onnx.py --model sam2.1_hiera_s --output sam2_decoder.onnx

# Compile with TensorRT
trtexec --onnx=sam2_decoder.onnx --saveEngine=sam2_decoder.engine --fp16
```

---

## 5. Commercial Readiness & License Audit
- **License**: **Apache-2.0**
- **Commercial Permissibility**: Permitted for commercial use, enterprise SaaS platforms, robotic perception stacks, and closed-source tools without disclosing caller source code.
- **Weights Licensing Notice**: Meta officially released SAM 2 and SAM 2.1 code and model checkpoints under Apache-2.0. Ensure training on proprietary data adheres to internal data governance.
- **Official Repository**: [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)

# Object Segmentation Playbook

# Overview
Object Segmentation partitions an image into semantically meaningful regions down to the pixel level. It spans Semantic Segmentation (assigning class labels to every pixel), Instance Segmentation (delineating discrete individual object instances), and Panoptic Segmentation (unifying things and stuff). Key practical applications include autonomous driving free-space detection, medical imaging lesion tracing, surgical robotics, and background matting.

## SOTA & Research
- **Seminal & Modern Papers**:
  - *FCN* (Long, Shelhamer, Darrell, 2015) & *U-Net* (Ronneberger et al., 2015): Foundational encoder-decoder architectures with skip connections for dense pixel classification.
  - *Mask R-CNN* (He et al., 2017): Extended Faster R-CNN with a parallel FCN mask prediction branch and RoIAlign.
  - *Segment Anything Model (SAM / SAM 2)* (Kirillov et al., 2023; Ravi et al., 2024): Foundation models for promptable zero-shot image and video segmentation with memory banks.
  - *Mask2Former* (Cheng et al., 2022): Universal architecture handling semantic, instance, and panoptic segmentation with masked-attention cross-entropy queries.
  - *YOLOv8-Seg / FastSAM* (2023-2024): Real-time instance segmentation predicting prototype masks and linear coefficient vectors.
- **Evaluation Benchmarks & Metrics**:
  - COCO (Mask mAP@0.50:0.95), Cityscapes (mIoU), ADE20K, SA-1B (SAM dataset).
  - Boundary IoU (evaluates contour delineation accuracy vs interior pixels).

## Architecture Alternatives & Trade-offs
| Architecture | Complexity / FPS | Boundary Precision | Use-Case |
| :--- | :--- | :--- | :--- |
| **YOLO-Seg (Prototype-based)** | 60-120+ FPS (TensorRT) | Moderate (limited by proto mask resolution) | Embedded robotics, real-time tracking |
| **Mask2Former** | 10-25 FPS (Server GPU) | Very high across complex multi-class scenes | Offline analytics, high-accuracy mapping |
| **SAM 2 (Video/Image)** | 30-40 FPS (streaming prompt) | Highest zero-shot mask quality | Interactive annotation, prompt-based tracking |
| **Lightweight Semantic (BiSeNet V2, PIDNet)** | 100+ FPS (Vulkan/Edge) | Coarse boundaries, high spatial recall | Autonomous vehicle driveable area / lane line detection |

## Popular Repos & Integrations
- **[Meta Segment Anything 2 (SAM 2)](https://github.com/facebookresearch/sam2)**: Real-time video and image promptable segmentation.
- **[MMSegmentation](https://github.com/open-mmlab/mmsegmentation)**: SOTA semantic segmentation toolbox supporting over 60 networks.
- **[Ultralytics YOLO-Seg](https://github.com/ultralytics/ultralytics)**: Turnkey instance segmentation training, validation, and C++ inference.
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

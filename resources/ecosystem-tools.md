# Curated Tooling & Ecosystem Resources

This document catalogs cross-cutting tools, libraries, and runtime frameworks for end-to-end Computer Vision, Perception, and Edge Deployment.

---

## 1. Visual Debugging, Curation & Dataset Management
- **[FiftyOne](https://github.com/voxel51/fiftyone)**: Standard open-source tool for building high-quality datasets and computer vision workflows. Integrates with PyTorch, Ultralytics, and Hugging Face for error analysis, hard-negative mining, and embeddings visualization.
- **[Rerun.io](https://github.com/rerun-io/rerun)**: High-performance SDK to log, visualize, and inspect multimodal spatial-temporal data (2D bounding boxes, 3D point clouds, camera pinholes, transforms, and sensor feeds) in real time over native Rust/C++/Python and web viewers.
- **[CVAT](https://github.com/cvat-ai/cvat)**: Full-featured image and video annotation tool for 2D bounding boxes, polygons, 3D cuboids, and semantic segmentation masks.

---

## 2. Core Frameworks & Model Hubs
- **PyTorch**: Primary training and research ecosystem with TorchVision, TorchScript, and LibTorch.
- **OpenMMLab (MMCV, MMDetection, MMSegmentation, MMDeploy)**: Modular open-source frameworks for dense visual perception and cross-target export.
- **Ultralytics**: High-speed, production-proven object detection, segmentation, and pose models (YOLO series).
- **Hugging Face Transformers / Diffusers**: State-of-the-art vision-language foundation models (CLIP, Florence-2, SigLIP, Depth Anything, SAM).

---

## 3. High-Performance Inference & Deployment Engines
- **NVIDIA TensorRT / Triton Inference Server**: Maximum GPU throughput, FP16/INT8 kernel auto-tuning, multi-instance GPU (MIG) execution.
- **Tencent NCNN / Vulkan**: High-performance neural network inference framework optimized for mobile and cross-platform devices via Vulkan Compute shaders.
- **ONNX Runtime**: Cross-platform acceleration across CUDA, DirectML, OpenVINO, and CPU.
- **AMD / Xilinx Vitis AI & FINN**: Quantized neural network (QNN) synthesis, DPU cores, and streaming dataflow execution on Xilinx FPGAs and Zynq UltraScale+ MPSoCs.

# Curated Tooling & Ecosystem Resources

This document catalogs cross-cutting tools, libraries, and runtime frameworks for end-to-end Computer Vision, Perception, and Edge Deployment.

---

## 1. Visual Debugging, Curation & Dataset Management
- **[[frameworks/fiftyone|FiftyOne]]**: Standard open-source tool for building high-quality datasets and computer vision workflows. Integrates with PyTorch, Ultralytics, and vector databases for error analysis, hard-negative mining, and embeddings visualization.
- **[[frameworks/rerun|Rerun.io]]**: High-performance SDK to log, visualize, and inspect multimodal spatial-temporal data (2D bounding boxes, 3D point clouds, camera pinholes, transforms, and sensor feeds) in real time over native Rust/C++/Python and web viewers.
- **CVAT**: Full-featured image and video annotation tool for 2D bounding boxes, polygons, 3D cuboids, and semantic segmentation masks.

---

## 2. Core Frameworks & Model Hubs
- **[[frameworks/pytorch|PyTorch Core]]** & **[[frameworks/torchvision|TorchVision]]**: Primary training and research ecosystem with TorchDynamo, TorchInductor, and FlexAttention.
- **[[frameworks/lerobot|Hugging Face LeRobot]]** & **[[frameworks/robomimic|Robomimic]]**: Physical AI, imitation learning, and real-time robotic teleoperation pipelines.
- **OpenMMLab (MMCV, MMDetection, MMSegmentation, MMDeploy)**: Modular open-source frameworks for dense visual perception and cross-target export.
- **Ultralytics**: High-speed, production-proven object detection, segmentation, and pose models.
- **Hugging Face Transformers / Diffusers**: State-of-the-art vision-language foundation models.

---

## 3. High-Performance Inference & Deployment Engines
- **[[frameworks/tensorrt|NVIDIA TensorRT 10.x]]**: Maximum GPU throughput, FP8/FP4 microscaling, kernel auto-tuning, and CUDA Graph execution.
- **[[frameworks/triton-inference-server|NVIDIA Triton Inference Server]]**: Dynamic batching, BLS model pipelining, concurrent model execution, and CUDA IPC shared memory.
- **[[frameworks/vulkan|Vulkan Compute]]** & **[[frameworks/vulkan-sc|Vulkan SC 2.0 (ASIL-D / DAL A)]]**: Cross-platform compute shaders and safety-certified offline compiled pipelines.
- **[[frameworks/openvino|Intel OpenVINO]]**: Cross-architecture acceleration across Intel CPUs, iGPUs, discrete GPUs, and NPU 4/5.
- **[[frameworks/onnxruntime|ONNX Runtime]]**: Universal pluggable Execution Providers (TensorRT, CUDA, OpenVINO, DirectML, QNN).
- **[[frameworks/vitis-ai|AMD Vitis AI]]** & **[[frameworks/quark|AMD Quark Quantization]]**: DPU cores, sub-byte FP8/MX6/INT4 quantization, and FPGA streaming execution.
- **[[frameworks/apache-tvm|Apache TVM & Relax]]**: End-to-end tensor compiler with symbolic shape inference and MetaSchedule.
- **[[frameworks/iceoryx2|Eclipse Iceoryx2]]** & **[[frameworks/zenoh|Eclipse Zenoh]]**: Zero-copy shared memory IPC and decentralized micro-broker middleware for robotics.

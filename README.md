---
title: Computer Vision & Perception Projects Knowledge Hub
tags:
  - computer-vision
  - perception
  - deep-learning
  - sota
  - obsidian-vault
  - index
updated: 2026-09-08
aliases:
  - Hub Index
  - CV Projects
---

# Computer Vision & Perception Projects Knowledge Hub

A curated repository and engineering cookbook covering research, end-to-end pipelines, trade-off analyses, and production deployment strategies for **Computer Vision (CV)**, **Machine Learning (ML)**, **Sensor Fusion**, and **Real-Time Systems**.

Targeting state-of-the-art architectures, practical edge cases, engineering workarounds, and multi-backend acceleration across **NVIDIA CUDA / TensorRT**, **Vulkan Compute**, and **FPGA (AMD/Xilinx Vitis AI & FINN)**. Fully indexed and compatible with **Obsidian** for visual graph exploration and bidirectional knowledge browsing.

---

## 🧭 Repository Structure & Obsidian Map

```text
cv-projects/
├── topics/                     # Comprehensive domain playbooks (Obsidian notes)
│   ├── object-detection/       # 2D/Oriented detection, SOTA transformers & real-time CNNs
│   ├── object-segmentation/    # Semantic, instance, panoptic, SAM 2 foundation models
│   ├── object-classification/  # ConvNeXt V2, ViT, SigLIP zero-shot & calibration
│   ├── video-tracking/         # MOT (ByteTrack, BoT-SORT), point tracking (CoTracker)
│   ├── 6dof-pose-estimation/   # 3D spatial orientation, PnP, FoundationPose, CAD matching
│   ├── lidar-perception/       # 3D point cloud detection, DSVT, FlatFormer, CenterPoint
│   ├── sensor-fusion/          # Camera-LiDAR-RADAR-IMU, BEVFusion, LSS, spatial calibration
│   ├── fpga-deployment/        # Vitis AI DPU, FINN dataflow, Brevitas QAT, LUTNet
│   ├── gpu-deployment/         # TensorRT 10, Triton, Vulkan NCNN, zero-copy streams, FP8/INT8
│   └── real-time-systems/      # PREEMPT_RT, Iceoryx 2 zero-copy, ROS 2, lock-free queues
├── templates/
│   └── topic-template.md       # Standard playbook schema and contribution guide
├── resources/
│   └── ecosystem-tools.md      # FiftyOne, Rerun, CVAT, OpenMMLab, inference runtimes
├── scripts/
│   └── validate_repo.py        # Automated repository integrity & schema validator
├── pyproject.toml              # UV-managed Python environment, dependencies & tools
└── .pre-commit-config.yaml     # Pre-commit hooks (ruff, commitzen, formatters)
```

---

## 📚 Playbooks & Topic Catalog

| # | Topic | Core SOTA Architectures | Hardware & Runtimes | Playbook Link (Obsidian & Git) |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Object Detection** | YOLOv10, RT-DETR, Grounding DINO, Co-DETR | TensorRT, Vulkan NCNN, ONNX Runtime | [Read Playbook](topics/object-detection/README.md) <br> `[[topics/object-detection/README\|Object Detection]]` |
| 2 | **Object Segmentation** | SAM 2, Mask2Former, FastSAM, MobileSAM, YOLOv8-Seg | TensorRT, Vulkan Shaders, TorchScript | [Read Playbook](topics/object-segmentation/README.md) <br> `[[topics/object-segmentation/README\|Object Segmentation]]` |
| 3 | **Object Classification** | ConvNeXt V2, MobileNetV4, SigLIP, DINOv2 | TensorRT INT8, OpenVINO, CPU/Edge | [Read Playbook](topics/object-classification/README.md) <br> `[[topics/object-classification/README\|Object Classification]]` |
| 4 | **Video Tracking** | BoT-SORT, ByteTrack, CoTracker2, TAPIR, SAM 2 | Multi-threaded CPU/GPU, Optical Flow | [Read Playbook](topics/video-tracking/README.md) <br> `[[topics/video-tracking/README\|Video Tracking]]` |
| 5 | **6-DoF Pose Estimation** | FoundationPose, MegaPose, GDR-Net, BOP Benchmark | Batched CUDA PnP, GPU ICP, TensorRT | [Read Playbook](topics/6dof-pose-estimation/README.md) <br> `[[topics/6dof-pose-estimation/README\|6-DoF Pose]]` |
| 6 | **LiDAR Perception** | DSVT, FlatFormer, CenterPoint, PointPillars | CUDA Sparse 3D Kernels, Jetson Orin | [Read Playbook](topics/lidar-perception/README.md) <br> `[[topics/lidar-perception/README\|LiDAR Perception]]` |
| 7 | **Sensor Fusion** | BEVFusion, Sparse4D, UniAD, PointPainting | Heterogeneous CPU-GPU, Fast BEV Pooling | [Read Playbook](topics/sensor-fusion/README.md) <br> `[[topics/sensor-fusion/README\|Sensor Fusion]]` |
| 8 | **FPGA Deployment** | Vitis AI 3.5 DPU, FINN Streaming Dataflow, Brevitas QAT | Xilinx Zynq UltraScale+, Kria SOM, Alveo | [Read Playbook](topics/fpga-deployment/README.md) <br> `[[topics/fpga-deployment/README\|FPGA Deployment]]` |
| 9 | **GPU Deployment** | TensorRT 10 FP8/INT8, FlashAttention-2, Vulkan NCNN | NVIDIA RTX/Jetson, AMD ROCm, Vulkan GPUs | [Read Playbook](topics/gpu-deployment/README.md) <br> `[[topics/gpu-deployment/README\|GPU Deployment]]` |
| 10 | **Real-Time Systems** | Iceoryx 2 Lock-Free IPC, ROS 2 Jazzy, PREEMPT_RT | Linux RT Kernel, Core Isolation, DMA | [Read Playbook](topics/real-time-systems/README.md) <br> `[[topics/real-time-systems/README\|Real-Time Systems]]` |

---

## 🛠️ Environment Setup & Tooling

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic Python environment management, alongside **FiftyOne** for dataset visualization and **Rerun** for spatial-temporal perception debugging.

### Prerequisites
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Python 3.11 or 3.12
- Git

### 1. Create Virtual Environment & Install Dependencies
```bash
# Initialize and sync dependencies
uv sync --extra dev --extra deployment
```

### 2. Pre-Commit & Commitzen
```bash
# Install git hooks
uv run pre-commit install --install-hooks
uv run pre-commit install --hook-type commit-msg

# Commit changes using Commitzen conventional commit standard
uv run cz commit
```

### 3. Visual Debugging with FiftyOne & Rerun
- **FiftyOne**: Inspect datasets, label noise, and model evaluation slices:
  ```python
  import fiftyone as fo
  dataset = fo.load_dataset("my_dataset")
  session = fo.launch_app(dataset)
  ```
- **Rerun**: Log 2D/3D camera feeds, bounding boxes, LiDAR points, and transforms:
  ```python
  import rerun as rr
  rr.init("perception_stream", spawn=True)
  rr.log("world/camera", rr.Pinhole(...))
  rr.log("world/points", rr.Points3D(...))
  ```

---

## 🧪 Repository Verification

To verify that all topics, required sections, SOTA benchmarks, and relative markdown links adhere to the project standards, run the repository validator:

```bash
python scripts/validate_repo.py
```

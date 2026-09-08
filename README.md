# Computer Vision & Perception Projects Knowledge Hub

A curated repository and engineering cookbook covering research, end-to-end pipelines, trade-off analyses, and production deployment strategies for **Computer Vision (CV)**, **Machine Learning (ML)**, **Sensor Fusion**, and **Real-Time Systems**.

Targeting state-of-the-art architectures, practical edge cases, engineering workarounds, and multi-backend acceleration across **NVIDIA CUDA / TensorRT**, **Vulkan Compute**, and **FPGA (AMD/Xilinx Vitis AI & FINN)**.

---

## 🧭 Repository Structure

```text
cv-projects/
├── topics/                     # Comprehensive domain playbooks
│   ├── object-detection/       # 2D/Oriented detection, SOTA transformers & real-time CNNs
│   ├── object-segmentation/    # Semantic, instance, panoptic, SAM 2 foundation models
│   ├── object-classification/  # ConvNeXt, ViT, CLIP/SigLIP zero-shot & calibration
│   ├── video-tracking/         # MOT (ByteTrack, BoT-SORT), point tracking (CoTracker)
│   ├── 6dof-pose-estimation/   # 3D spatial orientation, PnP, FoundationPose, CAD matching
│   ├── lidar-perception/       # 3D point cloud detection, CenterPoint, PointPillars, deskewing
│   ├── sensor-fusion/          # Camera-LiDAR-RADAR-IMU, BEVFusion, LSS, spatial calibration
│   ├── fpga-deployment/        # Vitis AI DPU, FINN dataflow, Brevitas QAT, LUTNet
│   ├── gpu-deployment/         # TensorRT, Triton, Vulkan NCNN, zero-copy streams, FP8/INT8
│   └── real-time-systems/      # PREEMPT_RT, ROS 2 zero-copy, lock-free queues, microsecond jitter
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

| # | Topic | Core SOTA Architectures | Hardware & Runtimes | Playbook Link |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Object Detection** | YOLOv8-v10, RT-DETR, Deformable DETR, Grounding DINO | TensorRT, Vulkan NCNN, ONNX Runtime | [Read Playbook](topics/object-detection/README.md) |
| 2 | **Object Segmentation** | SAM 2, Mask2Former, YOLO-Seg, PIDNet, BiSeNet | TensorRT, Vulkan Shaders, TorchScript | [Read Playbook](topics/object-segmentation/README.md) |
| 3 | **Object Classification** | ConvNeXt V2, Swin Transformer, MobileNetV4, CLIP / SigLIP | TensorRT INT8, OpenVINO, CPU/Edge | [Read Playbook](topics/object-classification/README.md) |
| 4 | **Video Tracking** | ByteTrack, BoT-SORT, CoTracker, TAPIR, SAM 2 Video | Multi-threaded CPU/GPU, Optical Flow | [Read Playbook](topics/video-tracking/README.md) |
| 5 | **6-DoF Pose Estimation** | FoundationPose, GDR-Net, PVN3D/FFB6D, CosyPose, EPnP | Batched CUDA PnP, GPU ICP, TensorRT | [Read Playbook](topics/6dof-pose-estimation/README.md) |
| 6 | **LiDAR Perception** | PointPillars, CenterPoint, PV-RCNN++, SparseConv (spconv) | CUDA Sparse 3D Kernels, Jetson Orin | [Read Playbook](topics/lidar-perception/README.md) |
| 7 | **Sensor Fusion** | BEVFusion, PointPainting, BEVDet, UniAD, PTP Time-Sync | Heterogeneous CPU-GPU, Fast BEV Pooling | [Read Playbook](topics/sensor-fusion/README.md) |
| 8 | **FPGA Deployment** | Vitis AI DPU, FINN Streaming Dataflow, Brevitas QAT | Xilinx Zynq UltraScale+, Kria SOM, Alveo | [Read Playbook](topics/fpga-deployment/README.md) |
| 9 | **GPU Deployment** | TensorRT FP8/INT8, Triton Server, Vulkan NCNN/Kompute | NVIDIA RTX/Jetson, AMD ROCm, Vulkan GPUs | [Read Playbook](topics/gpu-deployment/README.md) |
| 10 | **Real-Time Systems** | ROS 2 Cyclone DDS/Iceoryx, PREEMPT_RT, Lock-free Ring Buffers | Linux RT Kernel, Core Isolation, DMA | [Read Playbook](topics/real-time-systems/README.md) |

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

To verify that all topics, required sections, and relative markdown links adhere to the project standards, run the repository validator:

```bash
uv run python scripts/validate_repo.py
```

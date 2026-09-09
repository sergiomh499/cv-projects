---
title: "Runnable Cookbooks Vault & Production Reference Scripts MOC"
type: MOC
domain: Computer Vision, Robotics & AI Systems
status: evergreen
updated: 2026-08-15
tags:
  - moc
  - cookbooks
  - python
  - rust
  - runnable
aliases:
  - Cookbooks Hub
  - Cookbooks MOC
  - Production Scripts Hub
---

# 🍳 Runnable Cookbooks Vault & Production Reference Scripts MOC

The **Cookbooks Vault** contains 24 standalone, production-structured implementation scripts in Python and Rust. Every cookbook is engineered with zero-failure execution: each script executes directly (`python <script>.py` or `cargo check`), incorporates numerical assertions, logs hardware diagnostics, and includes a pure CPU/NumPy fallback when physical hardware accelerators or proprietary SDKs are absent.

---

## 🗂️ Master Catalog of the 24 Runnable Cookbooks

| # | Domain Focus | Directory Path | Main Script | Technology Stack | Hardware Target |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 01 | **Object Detection** | `cookbooks/01-rfdetr-tensorrt/` | `export_rfdetr_tensorrt.py` | PyTorch, ONNX, TensorRT 10 | NVIDIA GPU / Jetson Orin |
| 02 | **Segmentation** | `cookbooks/02-sam2-video-stream/` | `sam2_video_stream.py` | PyTorch, SAM 2 Memory Bank | GPU / CUDA Unified VRAM |
| 03 | **Zero-Copy IPC** | `cookbooks/03-iceoryx2-zero-copy-ipc/` | `main.rs` (Rust) | Rust, Iceoryx2 0.6, POSIX SHM | Linux `/dev/shm` IPC |
| 04 | **Data Quality** | `cookbooks/04-cleanlab-dataset-auditing/` | `audit_dataset.py` | Cleanlab, Confident Learning | CPU / Multithreaded |
| 05 | **Formal Verification**| `cookbooks/05-alpha-beta-crown-verification/` | `bound_verification.py` | Alpha-Beta-CROWN, LiRPA | CPU / GPU Verifier |
| 06 | **Explainability** | `cookbooks/06-mechanistic-cbm-attribution/` | `concept_intervention.py` | Concept Bottleneck, PyTorch | CPU / GPU Inference |
| 07 | **Neuromorphic Vision**| `cookbooks/07-neuromorphic-event-flow/` | `event_flow.py` | Asynchronous Event Surfaces | CPU / Low-Power Embedded |
| 08 | **Control Barriers** | `cookbooks/08-control-barrier-filter/` | `cbf_qp_filter.py` | Control Barrier Functions, QP | Hard Real-Time Robotics |
| 09 | **Sim2Real Safety** | `cookbooks/09-dann-sim2real-safety-gate/` | `dann_safety_gate.py` | DANN Adversarial Adaptation | CPU / Edge Accelerator |
| 10 | **Domain Alignment** | `cookbooks/10-contrastive-sim2real-alignment/` | `contrastive_alignment.py` | Contrastive InfoNCE Loss | PyTorch GPU / CPU |
| 11 | **Sensor Calibration**| `cookbooks/11-multi-sensor-cross-calibration/` | `cross_calibration.py` | Kabsch / ICP / SVD Solver | CPU Analytical Solver |
| 12 | **Video Tracking** | `cookbooks/12-bytetrack-two-stage-association/` | `bytetrack_association.py` | ByteTrack Two-Stage Matching | High-Speed CPU (<1 ms) |
| 13 | **6-DoF Pose** | `cookbooks/13-epnp-analytical-pose-solver/` | `epnp_solver.py` | EPnP O(n) Analytical Solver | CPU (<0.2 ms) |
| 14 | **BEV Sensor Fusion** | `cookbooks/14-bev-voxel-pooling-projection/` | `bev_pooling.py` | LSS / Voxel Pooling Kernels | GPU VRAM / CPU Fallback |
| 15 | **Real-Time OS** | `cookbooks/15-realtime-rms-edf-scheduler/` | `realtime_scheduler.py` | RMS / EDF Priority Scheduler | PREEMPT_RT Linux Kernel |
| 16 | **GPU Deployment** | `cookbooks/16-tensorrt-cuda-graphs/` | `engine_cuda_graphs.py` | TensorRT 10, CUDA Graphs | NVIDIA Ada / Blackwell / Orin |
| 17 | **FPGA Deployment** | `cookbooks/17-amd-quark-versal-ptq/` | `quark_versal_ptq.py` | AMD Quark, AWQ, FP8/INT4 | AMD Versal Gen 2 AIE-ML v2 |
| 18 | **SLAM & Spatial AI** | `cookbooks/18-3dgs-lie-slam/` | `3dgs_lie_slam.py` | 3DGS Lie Algebra SE(3) | PyTorch GPU / CPU Solver |
| 19 | **Optical Flow** | `cookbooks/19-raft-optical-flow/` | `raft_optical_flow.py` | RAFT 4D Correlation Volume | PyTorch CUDA / CPU |
| 20 | **Physical AI VLA** | `cookbooks/20-act-trajectory-chunking/` | `act_trajectory_chunking.py` | ACT C-VAE 50 Hz Ensembling | Robotics Edge Controller |
| 21 | **Spatial Visualization**| `cookbooks/21-rerun-spatial-sensor-stream/` | `rerun_sensor_stream.py` | Rerun.io SDK, Arrow Log | WebGPU / 3D Desktop GUI |
| 22 | **Dataset Curation** | `cookbooks/22-fiftyone-dataset-auditing/` | `fiftyone_auditing.py` | FiftyOne, Embedding Search | Desktop / Browser GUI |
| 23 | **Cross-Platform EP** | `cookbooks/23-onnxruntime-iobinding/` | `ort_iobinding.py` | ONNX Runtime 1.20+, IOBinding | TensorRT / QNN / OpenVINO |
| 24 | **Safety-Critical Compute**| `cookbooks/24-vulkan-sc-safety-critical/` | `vulkan_sc_pipeline.py` | Vulkan SC 2.0, PCC Offline AOT | ISO 26262 ASIL-D / DO-178C |

---

## 🧩 Shared Sensor & Geometric Simulation Library (`cookbooks/common/`)

The `cookbooks/common/` module provides shared, zero-dependency mathematical and sensor simulation primitives utilized across the cookbook implementations:

| Component | Primary Class / Utility | Key Capabilities |
| :--- | :--- | :--- |
| **Camera Geometry** | `PinholeCamera` | Intrinsic matrix $K$, perspective projection, unprojection from depth, boundary and positive depth gating. |
| **Rigid Body SE(3)** | `SE3Transform` | Group operations in $SE(3)$, Euler angles (XYZ degrees) to $SO(3)$, inverse transformations, homogeneous $4 \times 4$ matrices. |
| **LiDAR Simulation** | `SyntheticLidar` | Multi-beam (16/32/64) spinning and solid-state point cloud generation with ground planes and 3D bounding box obstacle clusters. |
| **Calibration Targets** | `SyntheticCalibrationTarget` | 3D metric checkerboard corner generators, 3D oriented bounding box corner synthesis with yaw. |
---

## ⚡ Execution Verification

Run the complete smoke test suite across all cookbooks:
```bash
uv run pytest tests/test_cookbooks_smoke.py -v
```

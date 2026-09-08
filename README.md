---
title: Computer Vision & Perception Projects Knowledge Hub
tags:
  - computer-vision
  - perception
  - deep-learning
  - sota
  - obsidian-vault
  - licenses
  - index
updated: 2026-09-08
aliases:
  - Hub Index
  - CV Projects
---

# 🧠 Computer Vision & Perception Projects Knowledge Hub

A curated research engineering knowledge base, didactic repository, and production playbook vault covering end-to-end computer vision, multi-modal sensor fusion, edge deployment (CUDA/Vulkan/FPGA), and microsecond real-time robotics.

> **Obsidian Users**: This repository is designed to be opened directly as an **Obsidian Vault**. It implements a hybrid **PARA + Zettelkasten + Maps of Content (MOC)** structure with bidirectional wikilinks (`[[...]]`), structured YAML properties, and central architecture notes under `architectures/`.

---

## 🧭 Master Knowledge Map & Topic MOCs

| Domain Topic | Obsidian MOC Note | Core SOTA Breakthroughs | Primary Repositories | Latency / FPS | Safe Commercial License |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **1. Object Detection** | [[topics/object-detection/00-object-detection-moc|Detection MOC]] ([Directory](topics/object-detection/README.md)) | RF-DETR, RT-DETRv2/v3, YOLOv10-26, Grounding DINO | `roboflow/rf-detr`, `lyuwenyu/RT-DETR` | 5.2 ms (53.8% AP) | **Apache-2.0** |
| **2. Object Segmentation** | [[topics/object-segmentation/00-object-segmentation-moc|Segmentation MOC]] ([Directory](topics/object-segmentation/README.md)) | SAM 2.1, Depth Anything V2, Mask2Former | `facebookresearch/sam2`, `DepthAnything` | 22.8 ms (44 FPS) | **Apache-2.0** |
| **3. Object Classification** | [[topics/object-classification/00-object-classification-moc|Classification MOC]] ([Directory](topics/object-classification/README.md)) | DINOv2/v3, SigLIP 2, ConvNeXt V2, MobileNetV4 | `facebookresearch/dinov2`, `google/big_vision` | 0.85 ms (82.3% Top-1) | **Apache-2.0** |
| **4. Video Tracking** | [[topics/video-tracking/00-video-tracking-moc|Tracking MOC]] ([Directory](topics/video-tracking/README.md)) | BoT-SORT, ByteTrack, CoTracker3, SAM 2 | `ifzhang/ByteTrack`, `facebookresearch/co-tracker` | 1.2 ms (80.3 MOTA) | **MIT / Apache-2.0** |
| **5. 6-DoF Pose Estimation**| [[topics/6dof-pose-estimation/00-6dof-pose-estimation-moc|6-DoF Pose MOC]] ([Directory](topics/6dof-pose-estimation/README.md)) | FoundationPose, MegaPose, GDR-Net | `NVlabs/FoundationPose`, `facebookresearch/megapose` | 32.0 ms (96.2% ADD-S) | **Apache-2.0** (MegaPose) |
| **6. LiDAR Perception** | [[topics/lidar-perception/00-lidar-perception-moc|LiDAR MOC]] ([Directory](topics/lidar-perception/README.md)) | DSVT, FlatFormer, PointPillars, CenterPoint | `Haiyang-W/DSVT`, `open-mmlab/OpenPCDet` | 27.0 ms (78.9% mAP) | **Apache-2.0** |
| **7. Sensor Fusion** | [[topics/sensor-fusion/00-sensor-fusion-moc|Sensor Fusion MOC]] ([Directory](topics/sensor-fusion/README.md)) | MIT BEVFusion, Sparse4D v3, UniAD | `mit-han-lab/bevfusion`, `HorizonRobotics/Sparse4D` | 41.0 ms (72.9% NDS) | **Apache-2.0** |
| **8. FPGA Deployment** | [[topics/fpga-deployment/00-fpga-deployment-moc|FPGA MOC]] ([Directory](topics/fpga-deployment/README.md)) | AMD Vitis AI 3.5, FINN & Brevitas (QNN) | `Xilinx/Vitis-AI`, `Xilinx/finn` | 0.68 ms (1,450 FPS) | **Apache-2.0 / EULA** |
| **9. GPU Deployment** | [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]] ([Directory](topics/gpu-deployment/README.md)) | TensorRT 10 (FP8), Vulkan NCNN, FlashAttention | `NVIDIA/TensorRT`, `Tencent/ncnn` | 0.25 ms (ResNet-50) | **Apache-2.0 / BSD** |
| **10. Real-Time Systems** | [[topics/real-time-systems/00-real-time-systems-moc|Real-Time MOC]] ([Directory](topics/real-time-systems/README.md)) | Linux PREEMPT_RT, Eclipse Iceoryx2, Zenoh | `eclipse-iceoryx/iceoryx2`, `eclipse-zenoh/zenoh` | <1 $\mu$s Zero-Copy | **Apache-2.0 / MIT** |
| **11. SLAM & Spatial Perception** | [[topics/slam-and-spatial-perception/00-slam-and-spatial-perception-moc|SLAM MOC]] ([Directory](topics/slam-and-spatial-perception/README.md)) | 3DGS SLAM, MonoGS, SplaTAM, FAST-LIO2 | `muskie82/MonoGS`, `spla-tam/SplaTAM` | 28.0 FPS (1.8 cm ATE) | **Apache-2.0 / MIT** |
| **12. Visual Guidance & Robotics** | [[topics/visual-guidance-and-robotics/00-visual-guidance-and-robotics-moc|Visual Guidance MOC]] ([Directory](topics/visual-guidance-and-robotics/README.md)) | OpenVLA 7B, Octo, AnyGrasp | `openvla/openvla`, `octo-models/octo` | 180 ms (84.7% Success) | **Apache-2.0** (OpenVLA) |
| **13. Data Quality & Verification** | [[topics/data-quality-and-verification/00-data-quality-and-verification-moc|Data Quality MOC]] ([Directory](topics/data-quality-and-verification/README.md)) | Cleanlab Confident Learning, Great Expectations, FiftyOne | `cleanlab/cleanlab`, `voxel51/fiftyone` | <10 ms per item | **Apache-2.0** |
| **14. Safety & Robustness** | [[topics/safety-verification-and-robustness/00-safety-verification-and-robustness-moc|Safety Verification MOC]] ([Directory](topics/safety-verification-and-robustness/README.md)) | $\alpha,\beta$-CROWN, ISO 21448 / ISO/PAS 8800, Control Barrier Functions | `Verified-Intelligence/alpha-beta-CROWN` | <100 $\mu$s CBF Filter | **GPL-3.0 / Apache-2.0** |
| **15. Explainability & Interpretability** | [[topics/explainability-and-interpretability/00-explainability-and-interpretability-moc|Explainability MOC]] ([Directory](topics/explainability-and-interpretability/README.md)) | Mechanistic CBMs, Sparse Autoencoders, Faithfulness Saliency | `jacobgil/pytorch-grad-cam`, `yewsiang/ConceptBottleneck` | 12 ms Attribution | **MIT / Apache-2.0** |
| **16. Event-Based Neuromorphic Vision** | [[topics/event-based-neuromorphic-vision/00-event-based-neuromorphic-vision-moc|Event Vision MOC]] ([Directory](topics/event-based-neuromorphic-vision/README.md)) | Prophesee Metavision 5.x, Spikformers, ESVO, Benosman Flow | `prophesee-ai/metavision_sdk`, `fangwei123456/spikingjelly` | <1 $\mu$s Temporal Res (>120 dB) | **Apache-2.0** |
| **17. Thermal & Hyperspectral Vision** | [[topics/thermal-and-hyperspectral-vision/00-thermal-and-hyperspectral-vision-moc|Thermal Vision MOC]] ([Directory](topics/thermal-and-hyperspectral-vision/README.md)) | TherA, UniCD, Frequency-Guided Cross-Attention, VOx Microbolometers | `FLIR/flirpy`, `spectral/spectral` | 800 FPS NUC (<1.5 ms) | **MIT / Apache-2.0** |
| **18. Active 3D Sensing & Structured Light** | [[topics/active-3d-sensing-and-structured-light/00-active-3d-sensing-and-structured-light-moc|Active 3D Sensing MOC]] ([Directory](topics/active-3d-sensing-and-structured-light/README.md)) | Sinusoidal Phase Profilometry, Indirect/Direct ToF SPAD, RealSense | `IntelRealSense/librealsense`, `isl-org/Open3D` | <10 $\mu$m Metrology (90 FPS) | **Apache-2.0 / MIT** |
| **19. Vision-Language-Action & Physical AI** | [[topics/vla-and-physical-ai-robotics/00-vla-and-physical-ai-robotics-moc|VLA Robotics MOC]] ([Directory](topics/vla-and-physical-ai-robotics/README.md)) | $\pi_0$, OpenVLA, Octo, Continuous Flow Matching, Cartesian Impedance | `openvla/openvla`, `huggingface/lerobot` | 50 Hz Control (20 ms) | **Apache-2.0 / MIT** |
| **20. Optical Flow & Scene Flow Perception** | [[topics/optical-and-scene-flow-perception/00-optical-and-scene-flow-perception-moc|Optical Flow MOC]] ([Directory](topics/optical-and-scene-flow-perception/README.md)) | GMFlow, UniMatch, RAFT, 3D Point Scene Flow, Horn-Schunck Variational | `princeton-vl/RAFT`, `haofeixu/gmflow` | >60 FPS GMFlow (<15 ms) | **MIT / Apache-2.0** |
---

## 🏛️ Central Architecture Vault (`architectures/`)

Dedicated landmark architectures and foundation models (each in an independent reference document):
- **[[architectures/3d-pointclouds-and-lidar/bevfusion|BEVFusion: Multi-Task Multi-Sensor Camera-LiDAR Fusion in Bird's-Eye View]]**: Unified Multi-Modal Bird's-Eye-View (BEV) Network (Apache-2.0).
- **[[architectures/3d-pointclouds-and-lidar/dsvt|DSVT: Dynamic Sparse Voxel Transformer for 3D LiDAR Object Detection]]**: Dynamic Sparse Window Voxel Transformer (Apache-2.0).
- **[[architectures/3d-pointclouds-and-lidar/flatformer|FlatFormer: Equal-Work Grouping 3D Voxel Transformer for Real-Time Point Cloud Perception]]**: Equal-Work Group Voxel Transformer (Apache-2.0).
- **[[architectures/3d-pointclouds-and-lidar/sparse4d|Sparse4D: Multi-Camera Temporal 4D Sparse Object Detection and Tracking]]**: Sparse 4D Temporal Deformable Transformer (Apache-2.0).
- **[[architectures/backbones-and-edge-efficiency/convnext-v2|ConvNeXt V2: Co-designing Pure ConvNets and Masked Autoencoders with Global Response Normalization]]**: Modern Pure ConvNet (Inverted Bottleneck & GRN) (Apache-2.0).
- **[[architectures/backbones-and-edge-efficiency/mambavision|MambaVision: Hybrid Visual State-Space and Transformer Foundation Backbone]]**: Hybrid State-Space Model (SSM) & Vision Transformer (Apache-2.0).
- **[[architectures/backbones-and-edge-efficiency/mobilenetv4|MobileNetV4: Universal Models for Efficient On-Device Computer Vision]]**: Universal Inverted Bottleneck (UIB) & Mobile-MQA (Apache-2.0).
- **[[architectures/backbones-and-edge-efficiency/vmamba|VMamba: Visual State-Space Model with 2D Selective Scan (SS2D)]]**: Visual State-Space Model (SS2D) (Apache-2.0).
- **[[architectures/hardware-and-acceleration-runtimes/finn-qnn|FINN & Brevitas: Dataflow Streaming Synthesis for Quantized Neural Networks on FPGAs]]**: Spatial Dataflow Streaming & MVTU Logic Synthesis (Apache-2.0).
- **[[architectures/hardware-and-acceleration-runtimes/iceoryx2-ipc|Iceoryx2: Ultra-Low-Latency Zero-Copy Shared Memory IPC Middleware for Robotics & Vision]]**: Lock-Free Shared Memory Inter-Process Communication (IPC) (Apache-2.0 / MIT).
- **[[architectures/hardware-and-acceleration-runtimes/tensorrt-runtime|TensorRT 10: High-Throughput Deep Learning Inference Compiler & Runtime Engine]]**: Deep Learning Compiler & Execution Engine (Apache-2.0 / NVIDIA Proprietary EULA).
- **[[architectures/hardware-and-acceleration-runtimes/vitis-ai-dpu|Vitis AI 3.5: DPU Hardware Overlays & AI Engine Execution on FPGAs & ACAPs]]**: Instruction-Driven DPU Overlay & AI Engine Vector Tiles (Apache-2.0 / Xilinx EULA).
- **[[architectures/hardware-and-acceleration-runtimes/vulkan-runtime|Vulkan 1.3 Compute: Vendor-Agnostic Cross-Platform GPU Inference Runtime]]**: Open Cross-Platform GPU Compute Runtime (Apache-2.0).
- **[[architectures/hardware-and-acceleration-runtimes/zenoh-router|Zenoh: Ultra-Low-Overhead Micro-Broker & Edge-to-Cloud Middleware for Robotics & Perception]]**: Distributed Micro-Broker & Key-Expression Router (Apache-2.0 / EPL-2.0).
- **[[architectures/multimodal-vlm-and-vla/anygrasp|AnyGrasp: Zero-Shot Dense 6-DoF Robotic Grasp Pose Detection]]**: Dense Point Cloud 6-DoF Grasp Detection Network (Custom / Research Non-Commercial).
- **[[architectures/multimodal-vlm-and-vla/diffusion-policy|Diffusion Policy: Visuomotor Robot Control via Denoising Diffusion Models]]**: Denoising Diffusion Probabilistic Model (DDPM / DDIM) (MIT).
- **[[architectures/multimodal-vlm-and-vla/florence-2|Florence-2: Unified Sequence-to-Sequence Vision Foundation Model]]**: Real-Time Unified (MIT).
- **[[architectures/multimodal-vlm-and-vla/internvl2-5|InternVL 2.5: High-Resolution Multimodal Vision-Language Foundation Model]]**: Foundation Model (Apache-2.0).
- **[[architectures/multimodal-vlm-and-vla/openvla|OpenVLA: Open-Source 7B Vision-Language-Action Foundation Model]]**: Autoregressive Vision-Language-Action (VLA) Transformer (Apache-2.0).
- **[[architectures/multimodal-vlm-and-vla/pi0|Physical Intelligence π0 (Pi-Zero): Flow Matching Vision-Language-Action Foundation Model]]**: Flow Matching Vision-Language-Action Policy (Apache-2.0).
- **[[architectures/multimodal-vlm-and-vla/qwen2-5-vl|Qwen2.5-VL: Dynamic-Resolution Vision-Language & Visual-Agent Foundation Model]]**: Foundation Model (Apache-2.0).
- **[[architectures/multimodal-vlm-and-vla/qwen2-vl|Qwen2-VL: Dynamic Resolution Vision-Language Foundation Model]]**: Real-Time Unified (Apache-2.0).
- **[[architectures/pose-and-robotics-manipulation/foundationpose|FoundationPose: Unified 6D Pose Estimation and Tracking for Novel Objects]]**: Foundation Render-and-Compare Vision Transformer (Custom Non-Commercial (NVIDIA)).
- **[[architectures/pose-and-robotics-manipulation/megapose|MegaPose: 6D Pose Estimation of Novel Objects with Differentiable Render-and-Compare Refinement]]**: Coarse-to-Fine Render-and-Compare 6D Pose Refiner (Apache-2.0).
- **[[architectures/real-time-detectors-and-segmenters/fastsam|FastSAM: Fast Segment Anything Model via Real-Time CNN Instance Segmentation]]**: Real-Time CNN Instance Detector & Mask Router (Apache-2.0).
- **[[architectures/real-time-detectors-and-segmenters/mask2former|Mask2Former: Masked-Attention Mask Transformer for Universal Image Segmentation]]**: Universal Query Transformer (Apache-2.0).
- **[[architectures/real-time-detectors-and-segmenters/mobilesam|MobileSAM: Faster Segment Anything Model via Decoupled Distillation]]**: Decoupled TinyViT Distilled Foundation Model (Apache-2.0).
- **[[architectures/real-time-detectors-and-segmenters/rf-detr|RF-DETR: Real-Time Detection Transformers via Neural Architecture Search]]**: Real-Time Unified (Apache-2.0).
- **[[architectures/real-time-detectors-and-segmenters/yolov12|YOLOv12: Attention-Centric Real-Time Detection Architecture]]**: Real-Time Unified (AGPL-3.0).
- **[[architectures/spatial-radiance-and-slam/3d-gaussian-splatting|3D Gaussian Splatting: Real-Time Radiance Field Rendering via Point-Based Primitives]]**: Explicit Differentiable Gaussian Rasterization (Custom / Non-Commercial (Inria / Max Planck)).
- **[[architectures/spatial-radiance-and-slam/3dgs-slam|3DGS-SLAM: Real-Time Dense Radiance Field Tracking & Mapping with 3D Gaussians]]**: Differentiable 3D Gaussian Primitive Optimization (MIT).
- **[[architectures/spatial-radiance-and-slam/monogs|MonoGS: Monocular 3D Gaussian Splatting SLAM with Geometric Depth Priors]]**: Monocular Differentiable Radiance Field Optimization (Apache-2.0).
- **[[architectures/spatial-radiance-and-slam/splatam|SplaTAM: Dense RGB-D SLAM with Explicit 3D Gaussian Radiance Fields]]**: Differentiable 3D Gaussian Optimization (MIT).
- **[[architectures/vision-foundation-models/depth-anything-v2|Depth Anything V2: Metric Depth & Surface Segmentation Foundation Model]]**: Foundation Vision Transformer (DINOv2 Distillation) (Apache-2.0).
- **[[architectures/vision-foundation-models/dinov2|DINOv2: Learning Robust Visual Features with Self-Supervised Vision Transformers]]**: Self-Supervised Isotropic Vision Transformer (ViT) (Apache-2.0).
- **[[architectures/vision-foundation-models/dinov3|DINOv3: Hierarchical Multimodal Vision Foundation Model with 2D-RoPE and Dense Pre-Training]]**: Hierarchical Vision Transformer with 2D-RoPE (Apache-2.0).
- **[[architectures/vision-foundation-models/sam-2|SAM 2, SAM 2.1 & SAM 3: Segment Anything in Images, Videos & Open Concepts]]**: Foundation Model (Hierarchical Vision Transformer) (Apache-2.0).
- **[[architectures/vision-foundation-models/siglip|SigLIP & SigLIP 2: Sigmoid Loss for Vision-Language Alignment]]**: Dual-Encoder Vision-Language Transformer (Apache-2.0).
- **[[architectures/visual-tracking-and-flow/botsort|BoT-SORT: Robust Multi-Object Tracking with Motion Compensation and Re-Identification]]**: Tracking-by-Detection (Kalman + ReID + GMC) (MIT).
- **[[architectures/visual-tracking-and-flow/bytetrack|ByteTrack: Multi-Object Tracking by Associating Every Detection Box]]**: Tracking-by-Detection (Two-Stage Bipartite Matching) (MIT).
- **[[architectures/visual-tracking-and-flow/cotracker|CoTracker & CoTracker3: Dense Point Trajectory Transformers]]**: Spatial-Temporal Point Transformer (Apache-2.0).

---

## 🍳 Production Cookbooks & Runnable Recipes (`cookbooks/`)

Self-contained, runnable recipes demonstrating state-of-the-art deployment, IPC, verification, and data quality:
- **[[cookbooks/01-rfdetr-tensorrt/export_rfdetr_tensorrt.py|01-rfdetr-tensorrt]]**: PyTorch to ONNX export with dynamic batch axes and TensorRT FP16 engine builder script.
- **[[cookbooks/02-sam2-video-stream/sam2_video_stream.py|02-sam2-video-stream]]**: Real-time temporal video mask propagation with SAM 2 and positive/negative interactive prompt points.
- **[[cookbooks/03-iceoryx2-zero-copy-ipc/main.rs|03-iceoryx2-zero-copy-ipc]]**: Rust-based sub-microsecond lock-free shared memory transport for 4K video frames.
- **[[cookbooks/04-cleanlab-dataset-auditing/audit_dataset.py|04-cleanlab-dataset-auditing]]**: Confident Learning implementation estimating joint noise distribution to find corrupted dataset labels.
- **[[cookbooks/05-alpha-beta-crown-verification/bound_verification.py|05-alpha-beta-crown-verification]]**: Linear relaxation neural network interval bound propagation (IBP) certifying $L_\infty$ robustness.
- **[[cookbooks/06-mechanistic-cbm-attribution/concept_intervention.py|06-mechanistic-cbm-attribution]]**: Concept Bottleneck Model (CBM) with test-time human-in-the-loop concept intervention.
- **[[cookbooks/07-neuromorphic-event-flow/event_flow.py|07-neuromorphic-event-flow]]**: Asynchronous Surface of Active Events (SAE) optical flow estimation via local plane fitting in $\mathcal{O}(1)$ time.
- **[[cookbooks/08-control-barrier-filter/cbf_qp_filter.py|08-control-barrier-filter]]**: Real-time Control Barrier Function (CBF) Quadratic Program safety filter enforcing forward invariance on unverified deep policy actions.
- **[[cookbooks/09-dann-sim2real-safety-gate/dann_safety_gate.py|09-dann-sim2real-safety-gate]]**: Sim2Real Domain Adaptation & Critical Mission Safety Gate rejecting ambiguous, high-entropy, or out-of-distribution deep/generative actions.
- **[[cookbooks/10-contrastive-sim2real-alignment/contrastive_alignment.py|10-contrastive-sim2real-alignment]]**: Supervised Contrastive Learning (SupCon / InfoNCE) projecting synthetic UE5 and scarce real physical crops into a unified domain-invariant manifold.
- **[[cookbooks/11-multi-sensor-cross-calibration/cross_calibration.py|11-multi-sensor-cross-calibration]]**: Multi-sensor extrinsic cross-calibration (Horn SVD procrustes solver) and sub-pixel metric depth parallax projection between RGB and Thermal/ToF.
- **[[cookbooks/12-bytetrack-two-stage-association/bytetrack_association.py|12-bytetrack-two-stage-association]]**: ByteTrack two-stage data association algorithm matching high-confidence detections first, then associating low-confidence occluded detections to preserve track continuity.
- **[[cookbooks/13-epnp-analytical-pose-solver/epnp_solver.py|13-epnp-analytical-pose-solver]]**: Efficient Perspective-n-Point (EPnP) analytical $O(n)$ 6-DoF pose solver using 4 virtual control points, SVD nullspace kernel estimation, and Kabsch alignment.
- **[[cookbooks/14-bev-voxel-pooling-projection/bev_pooling.py|14-bev-voxel-pooling-projection]]**: Camera-to-Bird's-Eye-View (BEV) voxel pooling projection (Lift-Splat-Shoot / BEVFusion style) unprojecting camera frustums into vehicle ego frame and voxelizing into BEV grids.
- **[[cookbooks/15-realtime-rms-edf-scheduler/realtime_scheduler.py|15-realtime-rms-edf-scheduler]]**: Hard real-time multi-task perception pipeline task scheduler implementing Rate Monotonic (RMS) and Earliest Deadline First (EDF) with Liu & Layland schedulability bounds.
---

---

## ⚡ Hardware Platforms & Silicon Acceleration Vault (`hardware/`)

---

## ⚡ Hardware Platforms & Silicon Acceleration Vault (`hardware/`)

Dedicated silicon guides covering compute, memory, and functional safety (each in an independent document):

- **[[hardware/amd-versal-ai-edge-gen1|AMD Versal AI Edge Gen 1: Architecture, AIE-ML v1 & Heterogeneous Compute]]**: FPGA & ACAP Compute Platforms.
- **[[hardware/amd-versal-ai-edge-gen2|AMD Versal AI Edge Gen 2: Architecture, AIE-ML v2, Microscaling & ASIL-D Compute]]**: FPGA & ACAP Compute Platforms.
- **[[hardware/amd-zynq-ultrascale-plus|AMD Zynq UltraScale+ MPSoC: Architecture, DPUCZDX8G & Embedded Vision]]**: FPGA & Embedded SoC Platforms.
- **[[hardware/arm-cortex-a78ae|Arm Cortex-A78AE: Split-Lock Architecture, Hybrid Clusters & ISO 26262 ASIL-D Microarchitecture]]**: Functional Safety & High-Performance Automotive CPU.
- **[[hardware/arm-cortex-r52|Arm Cortex-R52: Hard Real-Time Lockstep Safety Core, MPU & Fast Interrupt Handling]]**: Hard Real-Time Control & ISO 26262 ASIL-D Safety Cores.
- **[[hardware/arm-ethos-u65|Arm Ethos-U65: Micro-NPU for Embedded Edge, Microcontrollers & IoT Acceleration]]**: Embedded Micro-NPUs, TinyML & Microcontroller Acceleration.
- **[[hardware/arm-ethos-u85|Arm Ethos-U85: Micro-NPU for Edge Vision, Transformers & 4 TOPS Embedded AI Acceleration]]**: Micro-NPUs, TinyML & Edge Vision Transformers.
- **[[hardware/arm-neoverse-v3ae|Arm Neoverse V3AE: Automotive Server-Class High-Throughput CPU, SVE2 & AMBA CHI Architecture]]**: Automotive High-Performance Compute & Centralized SDV Infrastructure.
- **[[hardware/coreavi-cots-safety-hardware|CoreAVI COTS Safety Hardware: DO-254 DAL A, VkCoreSC & Avionics GPU Architectures]]**: Avionics, Defense & High-Integrity Safety Systems.
- **[[hardware/esperanto-et-soc-1|Esperanto ET-SoC-1: 1088-Core Energy-Efficient RISC-V Tensor Inference Architecture]]**: RISC-V Many-Core Compute & Low-Power Datacenter Acceleration.
- **[[hardware/intel-loihi-2|Intel Loihi 2: Asynchronous Neuromorphic Spiking Neural Processor & Microcode Architecture]]**: Neuromorphic Computing, Spiking Neural Networks & Event-Driven Intelligence.
- **[[hardware/intel-npu|Intel NPU 4 & 5: Lunar Lake, Arrow Lake & Panther Lake Neural Processing Units]]**: Client SoCs, Edge AI & Integrated Neural Processing Units.
- **[[hardware/intel-xeon-amx|Intel Xeon 6th Gen: AMX TMUL Matrix Acceleration & Datacenter AI Architecture]]**: Datacenter Compute, High-Performance CPU & Matrix Acceleration.
- **[[hardware/nvidia-blackwell-b200|NVIDIA Blackwell B200 & GB200: Dual-Die Architecture, NVLink 5 & Hyperscale NVFP4 AI]]**: Data Center, High-Performance AI Infrastructure & Frontier Model Training.
- **[[hardware/nvidia-drive-thor|NVIDIA DRIVE Thor: Blackwell Architecture, 1000-2000 TFLOPS & Centralized Automotive Superchip]]**: Autonomous Vehicles & Centralized Automotive Superchips.
- **[[hardware/nvidia-jetson-orin|NVIDIA Jetson Orin: Architecture, Ampere Tensor Cores, NVDLA 2.0 & Autonomous Systems]]**: Edge GPU & Autonomous Robotics Platforms.
- **[[hardware/nvidia-jetson-thor|NVIDIA Jetson Thor: Blackwell Architecture, Physical AI & Compact Robotics Superchip]]**: Physical AI, Humanoid Robotics & Edge Autonomy.
- **[[hardware/qualcomm-hexagon-npu|Qualcomm Hexagon NPU: Fused Scalar, Vector (HVX) & Matrix (HMX) Tensor Architecture]]**: Edge AI, Mobile & Automotive Central Compute.
- **[[hardware/synsense-speck|SynSense Speck & Xylo: Sub-Milliwatt Neuromorphic Dynamic Vision & Audio Processors]]**: Neuromorphic Vision, Event-Based Sensors & Ultra-Low-Power SNN SoCs.
- **[[hardware/tenstorrent-wormhole-blackhole|Tenstorrent Wormhole & Blackhole: Tensix Core Spatial Dataflow & Direct Mesh Architecture]]**: RISC-V, Spatial Dataflow & Scalable AI Infrastructure.

---

---

## 🚀 Software Frameworks, Compilers & Inference Runtimes (`frameworks/`)

Dedicated guides covering compilation pipelines, zero-copy memory, and runtime models (each in an independent document):

- **[[frameworks/apache-tvm|Apache TVM Unity & Relax: Symbolic Shapes, TensorIR Schedule Primitives & Multi-Target CodeGen]]**: Machine Learning Compilers, Kernel Optimization & Multi-Target Code Generation.
- **[[frameworks/fiftyone|Voxel51 FiftyOne: Dataset Curation, Multimodal Embeddings & Vector Search Architecture]]**: Computer Vision Data Curation, Embeddings Indexing & Dataset Quality.
- **[[frameworks/iceoryx2|Eclipse Iceoryx2: Zero-Copy Lock-Free Shared Memory Inter-Process Communication]]**: Real-Time Robotics Middleware & Deterministic IPC.
- **[[frameworks/lerobot|HuggingFace LeRobot: Physical AI, Robot Teleoperation & Visuomotor Policy Pipelines]]**: Physical AI, Robot Learning & Visuomotor Imitation Learning.
- **[[frameworks/onnxruntime|ONNX Runtime 1.20+: Cross-Platform Execution Provider Architecture & Memory Primitives]]**: Cross-Platform Inference & Execution Providers.
- **[[frameworks/openvino|Intel OpenVINO 2025.x / 2026.x: nGraph IR, NPU Acceleration & Heterogeneous Scheduling]]**: Heterogeneous Edge Inference & NPU Acceleration.
- **[[frameworks/pytorch|PyTorch 2.5 / 2.6 Core: TorchDynamo, AOTAutograd, TorchInductor & FlexAttention]]**: Deep Learning Compilers, GPU Acceleration & JIT Code Generation.
- **[[frameworks/quark|AMD Quark: Unified Quantization, Sub-Byte Formats (FP8, MX6, MX9, INT4) & Hardware Calibration]]**: Model Quantization, Compression & Low-Precision Compilation.
- **[[frameworks/rerun|Rerun.io SDK: Multimodal Spatial-Temporal Columnar Visualization & Time-Series Engine]]**: Spatial Computing, Robotics Visualization & Multimodal Stream Debugging.
- **[[frameworks/robomimic|Robomimic: Modular Imitation Learning, Offline RL Benchmarks & RoboSuite Integration]]**: Robot Learning, Offline Reinforcement Learning & Imitation Learning Benchmarks.
- **[[frameworks/ros2-rmw|ROS 2 RMW Architecture: rmw_zenoh, rmw_iceoryx2, and CycloneDDS Zero-Copy Middleware]]**: Robotics Middleware & Real-Time Communication Abstraction.
- **[[frameworks/tensorrt|NVIDIA TensorRT 10.x / 10.8+: Deep Learning Inference Compiler & Engine Runtime]]**: GPU Acceleration & High-Throughput Inference.
- **[[frameworks/torchvision|TorchVision: C++/CUDA Vision Operators, Transforms v2, GPU Video I/O & Model Zoo]]**: Computer Vision Infrastructure, Custom CUDA Operators & Data Augmentation.
- **[[frameworks/triton-inference-server|NVIDIA Triton Inference Server: Dynamic Batching, BLS & Zero-Copy IPC Shared Memory]]**: High-Throughput Model Serving & Microservices.
- **[[frameworks/vitis-ai|AMD Vitis AI 3.5: DPU Compilation, XIR Graph IR & AIE-ML Systolic Array Mapping]]**: FPGA & NPU Neural Acceleration (Versal AI Core / Kria SOM / Zynq MPSoC).
- **[[frameworks/vulkan-sc|Vulkan SC 2.0: Safety-Critical GPU Compute, ISO 26262 ASIL-D & Deterministic Pipelines]]**: Safety-Critical GPU Compute, Automotive ADAS & Avionics.
- **[[frameworks/vulkan|Vulkan 1.3 / 1.4 Compute: Explicit GPU Acceleration, SPIR-V & Cooperative Matrix]]**: Cross-Platform GPU Acceleration & Low-Overhead Compute.
- **[[frameworks/zenoh|Eclipse Zenoh & Zenoh-Pico: Decentralized Edge-to-Cloud Communication Framework]]**: Edge Robotics Middleware, Distributed Pub/Sub & Geodistributed Querying.

---

## 📐 Formal Mathematical Proofs & Hardware Acceleration Matrices

Dedicated deep engineering and didactic theoretical references:
- **[[docs/critical-scenarios-hardware-matrix.md|Critical Scenarios Hardware Benchmarking & Acceleration Matrix]]**: Comparative evaluation across **AMD Versal AI Edge Gen 2 (AIE-ML v2 / NPU IP with Vitis AI 5.x)**, **CoreAVI DO-178C / DO-254 DAL-A Vulkan SC**, **NVIDIA Jetson AGX Orin Industrial**, **NVIDIA RTX 4090**, and **AMD Alveo U50 / Xilinx Kria**.
- **[[docs/edge-ai-quantization-playbook.md|Edge-AI Quantization Playbook: PTQ vs QAT, Mixed-Precision & Outlier Suppression]]**: Production guide covering SmoothQuant mathematical outlier suppression, TensorRT 10 explicit Q/DQ compilation, and AMD Vitis AI 5.x / Quark NPU workflows.
- **[[docs/unreal-engine-sim2real-deep-guide.md|Unreal Engine Sim2Real Deep Guide: Synthetic Generation, Domain Randomization & Low-Real-Data Adaptation]]**: Complete architectural blueprint for training with massive UE5 Nanite/Lumen synthetic worlds and certifying against strictly isolated, scarce physical validation sets via DANN and Optimal Transport.
- **[[docs/hardware-runtimes-and-certification-reality.md|Hardware Perception & Inference Reality: CUDA, TensorRT, Vulkan, Vulkan SC & Vitis AI]]**: Ground-truth engineering guide detailing how each runtime actually executes, how each is tested, and the exact compilation workflows.
- **[[docs/hardware-runtimes/00-hardware-runtimes-moc.md|Modular Hardware Perception Runtimes Hub]]**: Dedicated deep guides for [[docs/hardware-runtimes/01-cuda-tensorrt-runtime|01. CUDA & TensorRT]], [[docs/hardware-runtimes/02-vulkan-compute-runtime|02. Vulkan 1.3 Compute]], [[docs/hardware-runtimes/03-vulkan-sc-safety-runtime|03. Vulkan SC & CoreAVI (DO-178C/ISO 26262)]], and [[docs/hardware-runtimes/04-vitis-ai-versal-npu-runtime|04. AMD Vitis AI & Versal NPU]].
---

## ⚖️ Global Commercial Usability & License Matrix

| Category | Safe for Proprietary Software? | Key Repositories & Models |
| :--- | :---: | :--- |
| **Permissive (Apache-2.0 / MIT / BSD)** | **YES** | `rf-detr`, `RT-DETR (v2/v3)`, `SAM 2`, `Depth-Anything-V2`, `DINOv2/v3`, `SigLIP`, `ConvNeXt-V2`, `ByteTrack`, `BoT-SORT`, `CoTracker3`, `MegaPose`, `DSVT`, `OpenPCDet`, `BEVFusion`, `Sparse4D`, `Vitis-AI`, `FINN`, `TensorRT`, `ncnn`, `iceoryx2`, `zenoh`, `MonoGS`, `SplaTAM`, `OpenVLA`, `Octo` |
| **Copyleft Warning (AGPL-3.0 / GPL-3.0)** | ⚠️ **NO** | `Ultralytics YOLOv8 / YOLO11` (AGPL-3.0), `DeepSORT` (GPL-3.0), `ORB-SLAM3` (GPL-3.0), `FAST-LIO2` (GPL-2.0). Using copyleft code over network APIs requires open-sourcing client callers. Standardize on **RF-DETR**, **MegaPose**, **MonoGS**, and **OpenVLA** to eliminate legal risks. |
| **Non-Commercial / Research-Only** | ⛔ **NO** | `NVlabs/FoundationPose` (NVIDIA Source Code License), `AnyGrasp` (Research license). Standardize on **MegaPose** and **Contact-GraspNet** for proprietary robotic deployment. |

---

## 🔄 Automated Documentation Trigger & Validation

To automatically synchronize all cross-domain backlinks and validate the vault:

```bash
# Auto-sync architecture indices and backlinks
python scripts/update_docs.py

# Run comprehensive schema validation
python scripts/validate_repo.py
```

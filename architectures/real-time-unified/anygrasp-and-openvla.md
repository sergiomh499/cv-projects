---
title: "AnyGrasp & OpenVLA: Foundation Models for Robotic Visual Guidance"
type: model-deep-dive
tasks:
  - visual-guidance
  - robotic-grasping
  - vision-language-action
  - trajectory-generation
architecture_class: Foundation Vision-Language-Action (VLA) & Dense Grasp Network
primary_license: Apache-2.0 / Non-Commercial
commercial_use: true
official_repo: https://github.com/openvla/openvla
paper_url: https://arxiv.org/abs/2406.09246
tags:
  - model
  - visual-guidance
  - robotics
  - openvla
  - anygrasp
  - vla
updated: 2026-09-08
aliases:
  - OpenVLA
  - AnyGrasp
  - Visual Guidance
---

# 🔬 AnyGrasp & OpenVLA: Robotic Visual Guidance & Action Models

## 1. Executive Brief & Significance
Robotic visual guidance historically required rigid multi-stage heuristics: detecting 2D object bounding boxes, estimating 6-DoF poses, generating collision-free inverse kinematics (IK) paths, and executing hardcoded grasp vectors.

Modern robotic perception bypasses these hand-crafted abstractions using:
1. **Dense 6-DoF Grasp Synthesis (AnyGrasp / GraspNet)**: Evaluates billions of 3D grasp proposals directly on dense point clouds, predicting gripper width, collision scores, and approach vectors.
2. **Vision-Language-Action (VLA) Foundation Models (OpenVLA, Octo)**: Direct end-to-end policy execution. Autoregressively maps natural language prompts ("pick up the red mug by the handle") and camera images directly into continuous 7-DoF robot arm joint displacements ($\Delta x, \Delta y, \Delta z, \Delta \text{roll}, \Delta \text{pitch}, \Delta \text{yaw}, \text{gripper\_action}$).

```mermaid
flowchart LR
    Prompt[Language Prompt: 'wipe the spilled coffee'] --> VLA[OpenVLA 7B Multi-Modal Transformer]
    Cam[Eye-in-Hand Camera Frame] --> VLA
    VLA --> Tokens[Autoregressive Action Token Prediction: 1x7 vector]
    Tokens --> Dequant[Dequantize to Metric Joint Velocities]
    Dequant --> Actuator[Zero-Copy Shared Memory to Robot Joint Controllers (Zenoh / EtherCAT)]
```

---

## 2. Core Architectural Mechanics: OpenVLA Action Discretization

### Autoregressive Action Tokenization:
OpenVLA (Kim et al., 2024) fine-tunes an open 7B parameter Vision-Language Model (Prismatic / Llama-2 backbone):
1. Continuous robot joint movements are quantized into 256 discrete bins per degree of freedom.
2. The model consumes visual patch tokens (from DINOv2 and SigLIP) concatenated with instruction text tokens.
3. It outputs 7 consecutive action tokens corresponding to metric arm translation, orientation delta, and binary gripper trigger:
   $$A_t = [\Delta x, \Delta y, \Delta z, \Delta R_x, \Delta R_y, \Delta R_z, \text{gripper}] \in [0, 255]^7$$

---

## 3. Quantitative SOTA Benchmark Profile (Robotic Manipulation)

| Architecture | Model Parameters | Task Domain | Open-X Embodiment Success Rate | Inference Latency | Open License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Octo-Small** | 27 M | VLA Diffusion Policy | 68.2% | 12.0 ms | Apache-2.0 |
| **Octo-Base** | 93 M | VLA Diffusion Policy | 74.5% | 28.0 ms | Apache-2.0 |
| **OpenVLA 7B** | 7 B | VLA Autoregressive | **84.7%** (SOTA) | 180.0 ms (FP8) | **Apache-2.0** |
| **AnyGrasp** | 35 M | Dense 6-DoF Grasping | **92.4%** Grasp Success | 45.0 ms | Non-Commercial |

---

## 4. Engineering Implementation: OpenVLA Action Sampling

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

# Load quantized OpenVLA model
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).cuda().eval()

image = Image.open("workbench_camera.jpg")
prompt = "In: What action should the robot take to grasp the yellow screwdriver?\nOut:"

inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)

# Predict metric joint action
with torch.no_grad():
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    # action: 7-element vector [x, y, z, roll, pitch, yaw, grasp]
```

---

## 5. Commercial Usability & License Audit
- **OpenVLA**: **Apache-2.0**. Fully permissive for commercial robotics, warehouse order-picking, and surgical assistance.
- **Octo**: **Apache-2.0**.
- **AnyGrasp**: Custom research license; commercial use requires explicit licensing.
- **Official Repositories**:
  - `openvla/openvla`: [https://github.com/openvla/openvla](https://github.com/openvla/openvla)
  - `octo-models/octo`: [https://github.com/octo-models/octo](https://github.com/octo-models/octo)

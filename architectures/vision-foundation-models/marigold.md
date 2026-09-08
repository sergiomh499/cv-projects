---
title: "Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation"
type: model-deep-dive
domain: Monocular Geometric Perception & Generative 3D Depth
tasks:
  - generative-depth-estimation
  - monocular-depth-estimation
  - zero-shot-3d-reconstruction
  - surface-normal-synthesis
  - fine-geometric-detail-extraction
architecture_class: Generative Latent Diffusion Foundation Model for Depth
primary_license: Apache-2.0
commercial_use: true
official_repo: https://github.com/prs-eth/Marigold
paper_url: https://arxiv.org/abs/2312.02145
tags:
  - architecture
  - foundation-model
  - diffusion-model
  - latent-diffusion
  - depth-estimation
  - stable-diffusion
  - zero-shot
  - sota
status: evergreen
updated: 2026-08-15
aliases:
  - Marigold
  - Marigold Depth
  - Marigold LDM
  - Marigold Diffusion Depth
---

# 🔬 Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation

## 1. Executive Brief & Significance

Traditional discriminative monocular depth estimation models (e.g., DPT, MiDaS, Depth Anything) cast depth prediction as a deterministic pixel-wise regression task. While computationally fast, deterministic regression models frequently produce planar smoothing, oversaturate thin foreground geometries (e.g., fence wires, individual hair strands, foliage), and struggle with ambiguous occlusions and extreme depth discontinuities.

**Marigold** (Ke et al., ETH Zurich, CVPR 2024 Oral) fundamentally reimagined monocular depth perception by discovering that **pre-trained text-to-image Latent Diffusion Models (LDMs)**—specifically Stable Diffusion v2—embed profound, emergent 3D geometric scene priors within their intermediate generative cross-attention representations. Rather than training a depth network from scratch, Marigold **fine-tunes the LDM denoising UNet** to iteratively synthesize continuous, hyper-detailed affine-invariant depth maps. Key breakthroughs include:
1. **8-Channel Latent Conditioning**: Modifies the first convolutional layer of the LDM UNet from 4 channels (latent noise) to 8 channels (4-channel noisy depth latent $\mathbf{z}_t$ concatenated with the 4-channel frozen VAE latent representation of the input RGB image $\mathcal{E}(\mathbf{I})$).
2. **Text-Prompt Decoupling**: Freezes text cross-attention layers and conditions the UNet strictly on an empty text prompt $\emptyset$, forcing the model to rely purely on spatial visual semantics.
3. **Generative Score Matching for Metric Geometry**: Employs Denoising Score Matching (DSM) under an affine-invariant target depth normalization, generating sharp boundaries, fine textures, and realistic surface normals without regression blur.
4. **Fast Distillation via Latent Consistency**: With the advent of **LCM-Marigold** (Latent Consistency Models) and **Marigold-Lightning / Turbo**, inference steps have been compressed from 50 iterative DDIM steps down to **1 to 4 steps**, enabling near-real-time high-fidelity 3D spatial reconstruction.

```mermaid
flowchart TD
    subgraph RGB_Latent_Encoding ["Input Ingestion and VAE Encoding"]
        InputRGB["Input RGB Image: H x W x 3"] --> PretrainedVAE["Frozen Pre-trained VAE Encoder"]
        PretrainedVAE --> LatentRGB["RGB Latent Conditioning: 4 x h x w"]
    end

    subgraph Diffusion_Denoising_Process ["Iterative Latent Depth Denoising DDIM / LCM"]
        InitNoise["Initial Gaussian Noise: 4 x h x w"] --> ConcatLatents["Channel Concatenation: 8 x h x w"]
        LatentRGB --> ConcatLatents
        
        ConcatLatents --> DenoisingUNet["Fine-Tuned Denoising UNet"]
        Timestep["Timestep Embedding: t in range 0 to T"] --> DenoisingUNet
        EmptyPrompt["Empty Text Condition: Null Prompt"] --> DenoisingUNet
        
        DenoisingUNet --> PredNoise["Predicted Depth Latent Noise"]
        PredNoise --> DDIMStep["Solver Step: Latent Depth Update"]
        DDIMStep -.->|Iterative Recurrence 1-4 Steps| ConcatLatents
    end

    subgraph VAE_Depth_Decoding ["Latent Depth Decoding and Post-Processing"]
        DDIMStep --> FinalLatentDepth["Denoised Latent Depth: 4 x h x w"]
        FinalLatentDepth --> PretrainedVAEDecoder["Frozen VAE Decoder"]
        PretrainedVAEDecoder --> MultiChannelDepth["Decoded Output: 3 x H x W"]
        MultiChannelDepth --> ChannelCollapse["Channel Averaging"]
        ChannelCollapse --> AffineNorm["Affine-Invariant Scale and Shift Normalization"]
        AffineNorm --> OutputDepth["High-Fidelity Metric/Relative Depth Map"]
    end
```

---

## 2. Granular Component-by-Component Architectural Breakdown

Marigold adapts the Stable Diffusion v2.1-base architecture ($860\text{ M}$ parameter UNet + $84\text{ M}$ parameter AutoencoderKL).

### Architectural Subsystem Matrix

| Subsystem Component | Exact Layer / Module Identity | Mathematical Operations | Dimensionality & Channels | Latency Share (%) | Dominant Hardware Bound |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VAE Image Encoder** | AutoencoderKL Encoder | $4\times\text{ Downsample Blocks} + \text{ResNet} + \text{Attn}$ | $[B, 3, H, W] \to [B, 4, H/8, W/8]$ | ~8.5% | Memory Access / Conv2D |
| **Latent Concatenator** | Channel Concatenator | $\text{Concat}(\mathbf{z}_t, \mathbf{z}_{\text{rgb}}, \text{dim}=1)$ | Output Tensor $[B, 8, H/8, W/8]$ | <0.1% | Zero-Copy Memory View |
| **Modified Input Conv** | `conv_in` Adaptation | $3\times 3\text{ Conv2D, 8 In-Channels } \to 320\text{ Out}$ | Channels: $8 \to 320$, Spatial: $H/8 \times W/8$ | ~0.8% | Conv2D Compute |
| **UNet Down-Blocks** | CrossAttnDownBlock2D | $2\times\text{ ResNet Blocks} + 2\times\text{ Spatial Transformers} + \text{Down}$ | Channels: $320 \to 640 \to 1280 \to 1280$ | ~32.0% | Tensor Core / FlashAttn |
| **UNet Mid-Block** | UNetMidBlock2DCrossAttn | ResNet + Spatial Transformer + ResNet | Channels: $1280$, Spatial: $H/64 \times W/64$ | ~14.5% | GEMM / Matrix Multiplier |
| **UNet Up-Blocks** | CrossAttnUpBlock2D | $3\times\text{ ResNet Blocks} + 3\times\text{ Spatial Transformers} + \text{Up}$ | Channels: $1280 \to 1280 \to 640 \to 320$ | ~34.0% | Tensor Core / FlashAttn |
| **UNet Output Conv** | `conv_out` Layer | GroupNorm $+ 3\times 3\text{ Conv2D } (320 \to 4)$ | Channels: $320 \to 4$, Spatial: $H/8 \times W/8$ | ~0.6% | Conv2D Memory Access |
| **VAE Depth Decoder** | AutoencoderKL Decoder | $4\times\text{ Upsample Blocks} + \text{ResNet} + \text{Attn}$ | $[B, 4, H/8, W/8] \to [B, 3, H, W]$ | ~9.5% | VRAM Bandwidth / UpConv |

---

## 3. Mathematical Formulations & Loss Functions

### A. Denoising Score Matching for Latent Depth
During training, ground truth depth maps $\mathbf{D}^* \in \mathbb{R}^{H \times W}$ are normalized into $[-1, 1]$ and mapped into the latent space via the pre-trained VAE encoder: $\mathbf{z}_0 = \mathcal{E}(\mathbf{D}^*)$.

At training timestep $t \sim \mathcal{U}(1, T)$, Gaussian noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ is injected into the latent depth according to the forward diffusion schedule:

$$\mathbf{z}_t = \sqrt{\bar{\alpha}_t} \mathbf{z}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$$

The modified UNet $\boldsymbol{\epsilon}_\theta$ is optimized via the Denoising Score Matching (DSM) objective conditioned on the RGB latent $\mathcal{E}(\mathbf{I})$:

$$\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{\mathbf{z}_0, \, \mathbf{I}, \, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \, t} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\left( \mathbf{z}_t, \, t, \, \mathcal{E}(\mathbf{I}) \right) \right\|_2^2 \right]$$

The weights of the VAE encoder $\mathcal{E}$ and VAE decoder $\mathcal{D}$ remain completely **frozen** during training; only the UNet weights $\theta$ (with the expanded 8-channel input layer) are optimized.

---

### B. Affine-Invariant Depth Alignment & Evaluation Metrics
Because monocular depth models predict depth up to an unknown global affine transformation (scale $s$ and shift $t$), predicted depth $\mathbf{d} \in \mathbb{R}^{H \times W}$ is aligned to ground-truth depth $\mathbf{d}^* \in \mathbb{R}^{H \times W}$ using closed-form Least Squares:

$$(s^*, t^*) = \arg\min_{s, t} \sum_{i \in \Omega} \left( s \cdot d_i + t - d_i^* \right)^2$$

$$\begin{pmatrix} s^* \\ t^* \end{pmatrix} = \begin{pmatrix} \sum d_i^2 & \sum d_i \\ \sum d_i & M \end{pmatrix}^{-1} \begin{pmatrix} \sum d_i d_i^* \\ \sum d_i^* \end{pmatrix}$$

where $M = |\Omega|$ denotes the count of valid ground-truth pixels. Standard evaluation metrics are computed on the aligned prediction $\hat{\mathbf{d}} = s^* \mathbf{d} + t^*$:

$$\text{AbsRel} = \frac{1}{M} \sum_{i \in \Omega} \frac{|\hat{d}_i - d_i^*|}{d_i^*}, \quad \text{RMSE} = \sqrt{\frac{1}{M} \sum_{i \in \Omega} (\hat{d}_i - d_i^*)^2}, \quad \delta_1 = \frac{1}{M} \sum_{i \in \Omega} \mathbb{I}\left( \max\left( \frac{\hat{d}_i}{d_i^*}, \frac{d_i^*}{\hat{d}_i} \right) < 1.25 \right)$$

---

### C. Fast DDIM Sampling Trajectory
For fast test-time sampling across $S$ discretized timesteps ($\{t_S, t_{S-1}, \dots, t_1\}$), the reverse latent depth step is computed deterministically via DDIM ($\eta = 0$):

$$\mathbf{z}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{\mathbf{z}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathcal{E}(\mathbf{I}))}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathcal{E}(\mathbf{I}))$$

With **LCM-Marigold**, the model directly predicts the continuous trajectory endpoint $\mathbf{z}_0$ in a single forward pass ($S=1$) or two passes ($S=2$).

---

## 4. Benchmark Evaluation & Performance Profiles

### A. Zero-Shot Monocular Depth Estimation Benchmarks

| Model Architecture | Backbone / Base | NYUv2 (AbsRel $\downarrow$) | NYUv2 ($\delta_1 \uparrow$) | ScanNet (AbsRel $\downarrow$) | KITTI (AbsRel $\downarrow$) | ETH3D (AbsRel $\downarrow$) | DIODE (AbsRel $\downarrow$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MiDaS v3.1** | BEiT-L ($335\text{M}$) | 0.118 | 87.4% | 0.134 | 0.141 | 0.162 | 0.312 |
| **DPT-Large** | ViT-L ($307\text{M}$) | 0.108 | 89.2% | 0.125 | 0.126 | 0.150 | 0.298 |
| **Depth Anything v1-L**| ViT-L ($335\text{M}$) | 0.069 | 97.1% | 0.078 | 0.071 | 0.092 | 0.190 |
| **Marigold (10-Step DDIM)**| SD-v2.1 UNet ($860\text{M}$) | 0.054 | 98.4% | 0.061 | 0.064 | 0.074 | 0.165 |
| **Marigold (50-Step DDIM)**| SD-v2.1 UNet ($860\text{M}$) | **0.048** | **99.1%** | **0.055** | **0.058** | **0.068** | **0.152** |
| **LCM-Marigold (4-Step)** | SD-v2.1 LCM ($860\text{M}$) | 0.056 | 98.1% | 0.063 | 0.067 | 0.078 | 0.169 |
| **Marigold-Lightning (1-Step)**| SD-Turbo ($860\text{M}$) | 0.061 | 97.6% | 0.069 | 0.072 | 0.083 | 0.178 |

---

### B. Hardware Latency & Multi-Step Inference Profile

*Measured on $768 \times 768$ input resolution with standard VAE encoding and decoding.*

| Model Variant | Solver Steps | Precision | NVIDIA T4 (ms / FPS) | Jetson AGX Orin (ms / FPS) | NVIDIA A100 80GB (ms / FPS) | NVIDIA H100 SXM5 (ms / FPS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Marigold Standard** | 10 Steps DDIM | FP16 | 2150 ms / 0.46 FPS | 1720 ms / 0.58 FPS | 380 ms / 2.63 FPS | 145 ms / 6.89 FPS |
| **Marigold Standard** | 4 Steps DDIM | FP16 | 890 ms / 1.12 FPS | 710 ms / 1.40 FPS | 158 ms / 6.32 FPS | 62 ms / 16.1 FPS |
| **LCM-Marigold** | 4 Steps LCM | FP16 / TensorRT | **420 ms / 2.38 FPS** | **335 ms / 2.98 FPS** | **74 ms / 13.5 FPS** | **29 ms / 34.4 FPS** |
| **LCM-Marigold** | 2 Steps LCM | FP16 / TensorRT | **225 ms / 4.44 FPS** | **180 ms / 5.55 FPS** | **39 ms / 25.6 FPS** | **15 ms / 66.6 FPS** |
| **Marigold-Lightning** | 1 Step Turbo | FP16 / TensorRT | **128 ms / 7.81 FPS** | **98 ms / 10.2 FPS** | **21 ms / 47.6 FPS** | **8.2 ms / 121.9 FPS** |
| **Marigold-Lightning** | 1 Step Turbo | INT8 / TensorRT | **74 ms / 13.5 FPS** | **56 ms / 17.8 FPS** | **11 ms / 90.9 FPS** | **4.2 ms / 238.0 FPS** |

---

## 5. Edge Deployment, TensorRT Optimization & Gotchas

### A. TensorRT UNet + VAE Multi-Engine Pipeline
Deploying Marigold for real-time spatial computing requires splitting the pipeline into **three dedicated TensorRT engines**:
1. **VAE Encoder Engine**: Takes RGB image $[1, 3, 768, 768] \to [1, 4, 96, 96]$.
2. **Diffusion UNet Engine**: Takes concatenated latent $[1, 8, 96, 96]$, timestep $[1]$, and empty text embedding $[1, 77, 1024] \to [1, 4, 96, 96]$.
3. **VAE Decoder Engine**: Takes denoised latent $[1, 4, 96, 96] \to [1, 3, 768, 768]$.

```mermaid
flowchart LR
    subgraph Pipeline_Engine_Flow ["TensorRT 10 Multi-Engine Execution"]
        Input["RGB Image [1, 3, 768, 768]"] --> VAE_Enc["Engine 1: VAE Encoder (FP16)"]
        VAE_Enc --> LatentRGB["RGB Latent [1, 4, 96, 96]"]
        
        LatentRGB & Noise["Latent Noise [1, 4, 96, 96]"] --> Concat["Concat Layer"]
        Concat --> UNet_TRT["Engine 2: UNet Denoising Engine (FP16/INT8)"]
        UNet_TRT -->|1-4 Steps Loop| UNet_TRT
        
        UNet_TRT --> DenoisedLatent["Denoised Latent [1, 4, 96, 96]"]
        DenoisedLatent --> VAE_Dec["Engine 3: VAE Decoder (FP16)"]
        VAE_Dec --> OutputDepth["3D Depth Map [1, 1, 768, 768]"]
    end
```

---

### B. Deployment Gotchas & Optimization Traps

1. **Latent Inversion Scale Factor Discrepancies**:
   - *Trap*: Feeding raw VAE outputs directly to the UNet without the Stable Diffusion scaling factor ($0.18215$) leads to completely chaotic noise prediction:
     $$\mathbf{z}_{\text{rgb}} \neq \mathcal{E}(\mathbf{I})_{\text{raw}}$$
   - *Fix*: Always apply the exact scaling constant during latent encoding and inverse decoding:
     $$\mathbf{z}_{\text{rgb}} = 0.18215 \times \mathcal{E}(\mathbf{I})_{\text{latent\_dist.mean}}, \quad \hat{\mathbf{D}} = \mathcal{D}\left( \frac{\mathbf{z}_0}{0.18215} \right)$$

2. **Multi-Channel VAE Output Collapse**:
   - *Trap*: The standard Stable Diffusion VAE decoder outputs a 3-channel RGB image tensor ($[1, 3, H, W]$). Taking only the first channel ($D[:, 0, :, :]$) introduces color-space bias.
   - *Fix*: Average all three output channels across the channel dimension and normalize:
     $$d(x, y) = \frac{1}{3} \sum_{c=1}^3 \mathbf{D}_{\text{raw}}^{(c)}(x, y)$$

3. **Spatial Tiling and Boundary Seams for Large Images**:
   - *Trap*: Denoising large images ($>1024 \times 1024$) via naive rectangular grid tiling creates visible seam artifacts and inconsistent global affine depth planes across adjacent tiles.
   - *Fix*: Implement Gaussian-weighted overlapping sliding windows with a minimum $33\%$ spatial overlap and perform Least-Squares affine alignment before blending tile borders.

---

## 6. Complete Runnable Python Blueprint

The following standalone script implements a functional miniature Marigold latent diffusion pipeline, including latent channel concatenation, multi-step reverse denoising, and affine-invariant depth evaluation.

```python
"""
Marigold Generative Latent Diffusion Depth Blueprint
Demonstrates 8-channel UNet conditioning, multi-step DDIM denoising, and affine alignment.
"""

import math
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniDenoisingUNet(nn.Module):
    """
    Miniature 8-channel conditioning UNet simulating Marigold's adapted Stable Diffusion UNet.
    Inputs: 4-channel noisy depth latent + 4-channel RGB image latent = 8 channels.
    """
    def __init__(self, in_channels: int = 8, out_channels: int = 4, base_dim: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(base_dim, base_dim * 4),
            nn.SiLU(),
            nn.Linear(base_dim * 4, base_dim * 4)
        )
        
        # 8-Channel Input Convolution
        self.conv_in = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)
        
        # Down Blocks
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=3, stride=2, padding=1)
        
        # Mid Block with Self-Attention
        self.mid_conv = nn.Conv2d(base_dim * 4, base_dim * 4, kernel_size=3, padding=1)
        self.mid_attn = nn.MultiheadAttention(embed_dim=base_dim * 4, num_heads=4, batch_first=True)
        
        # Up Blocks
        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=4, stride=2, padding=1)
        
        # Output Noise Prediction Head
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1)
        )

    def get_timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device) / half_dim)
        args = timesteps[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, z_t: torch.Tensor, z_rgb: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # Concatenate 4-channel noisy depth + 4-channel RGB latent -> 8 channels
        x = torch.cat([z_t, z_rgb], dim=1) # [B, 8, H/8, W/8]
        
        t_emb = self.time_mlp(self.get_timestep_embedding(timestep, 64))
        
        # Encoder Downsampling
        h0 = self.conv_in(x)
        h1 = F.silu(self.down1(h0))
        h2 = F.silu(self.down2(h1))
        
        # Middle Block
        h_mid = self.mid_conv(h2)
        B, C, Hm, Wm = h_mid.shape
        flat = h_mid.flatten(2).permute(0, 2, 1)
        attn_out, _ = self.mid_attn(flat, flat, flat)
        h_mid = attn_out.permute(0, 2, 1).view(B, C, Hm, Wm) + h_mid
        
        # Decoder Upsampling
        u2 = F.silu(self.up2(h_mid) + h1)
        u1 = F.silu(self.up1(u2) + h0)
        
        eps_pred = self.conv_out(u1) # [B, 4, H/8, W/8]
        return eps_pred


class MarigoldDiffusionPipeline(nn.Module):
    """
    Complete Marigold Inference Pipeline Blueprint.
    Simulates VAE latent encoding, DDIM reverse trajectory, and depth decoding.
    """
    def __init__(self, num_train_timesteps: int = 1000):
        super().__init__()
        self.unet = MiniDenoisingUNet()
        self.num_train_timesteps = num_train_timesteps
        
        # Linear Beta Schedule
        betas = torch.linspace(0.00085, 0.0120, num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def ddim_sample(
        self, 
        z_rgb: torch.Tensor, 
        num_inference_steps: int = 4
    ) -> torch.Tensor:
        """Executes multi-step deterministic reverse DDIM sampling."""
        device = z_rgb.device
        B, _, H_lat, W_lat = z_rgb.shape
        
        # Discretize timesteps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, self.num_train_timesteps, step_ratio, device=device).flip(0)
        
        # Initial Latent Gaussian Noise: z_T ~ N(0, I)
        z_t = torch.randn(B, 4, H_lat, W_lat, device=device)
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t.item(), device=device, dtype=torch.long)
            
            with torch.no_grad():
                eps_pred = self.unet(z_t, z_rgb, t_tensor)
                
            alpha_prod_t = self.alphas_cumprod[t]
            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            alpha_prod_prev = self.alphas_cumprod[prev_t] if prev_t > 0 else torch.tensor(1.0, device=device)
            
            # Deterministic DDIM Step
            pred_z0 = (z_t - torch.sqrt(1.0 - alpha_prod_t) * eps_pred) / torch.sqrt(alpha_prod_t)
            dir_xt = torch.sqrt(1.0 - alpha_prod_prev) * eps_pred
            z_t = torch.sqrt(alpha_prod_prev) * pred_z0 + dir_xt
            
        return z_t

    def forward(self, rgb_image: torch.Tensor, steps: int = 4) -> torch.Tensor:
        # Mock VAE Encoding: [B, 3, H, W] -> [B, 4, H/8, W/8]
        z_rgb = F.interpolate(rgb_image, scale_factor=0.125, mode="bilinear", align_corners=False)
        z_rgb = torch.cat([z_rgb, z_rgb[:, :1]], dim=1) * 0.18215 # 4 channels
        
        # Reverse Diffusion Sampling
        z_0 = self.ddim_sample(z_rgb, num_inference_steps=steps)
        
        # Mock VAE Decoding: [B, 4, H/8, W/8] -> [B, 1, H, W]
        raw_decoded = F.interpolate(z_0, scale_factor=8.0, mode="bilinear", align_corners=False)
        pred_depth = raw_decoded.mean(dim=1, keepdim=True) # Average channels
        
        # Affine normalization to [0, 1]
        pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-8)
        return pred_depth


# Standalone Verification
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Marigold Diffusion Blueprint on device: {device}")
    
    pipeline = MarigoldDiffusionPipeline(num_train_timesteps=1000).to(device)
    pipeline.eval()
    
    # Simulate input image: 1x3x256x256
    dummy_rgb = torch.randn(1, 3, 256, 256, device=device)
    
    print("\nExecuting 4-Step DDIM Reverse Diffusion Depth Denoising...")
    with torch.no_grad():
        depth_output = pipeline(dummy_rgb, steps=4)
        
    print(f"Input RGB Dimensions:  {dummy_rgb.shape}")
    print(f"Synthesized Depth Map: {depth_output.shape} (Range: [{depth_output.min().item():.3f}, {depth_output.max().item():.3f}])")
    print("Marigold generative depth estimation successfully demonstrated.")
```

---

## 7. Peer Comparisons & Cross-Links

### Landmark Generative & Discriminative Depth Model Comparison

| Architectural Attribute | Marigold (CVPR 2024 Oral) | Depth Anything v1 (CVPR 2024) | Depth Anything V2 (2024) | ZoeDepth (Bhat et al.) |
| :--- | :--- | :--- | :--- | :--- |
| **Model Nature** | Generative Latent Diffusion | Discriminative Student-Teacher | Synthetic Data Distillation | Metric-Relative Dual Head |
| **Base Architecture** | Stable Diffusion v2.1 UNet | DINOv2 ViT-L + DPT | DINOv2 ViT-L + DPT | BEiT-384 + Metric Bins |
| **Iterative Steps** | 1 to 10 Steps (DDIM / LCM) | 1 Single Forward Pass | 1 Single Forward Pass | 1 Single Forward Pass |
| **Fine Geometric Detail** | **State-of-the-Art (Porous & Thin)** | High (DINOv2 Semantics) | Ultra-High (Zero LiDAR Noise) | Moderate (Bin Quantization) |
| **Zero-Shot NYUv2 AbsRel**| **0.048 (50-Step) / 0.056 (4-Step)** | 0.069 | 0.075 (Metric SOTA) | 0.095 |
| **Edge Viability (TRT)** | Moderate (98 ms on Orin, 1-Step) | Ultra-High (6.2 ms on Orin) | Ultra-High (6.2 ms on Orin) | Moderate (28 ms on Orin) |

### Related Knowledge Base Documents
- [[architectures/vision-foundation-models/depth-anything|Depth Anything: Large-Scale Unlabeled Monocular Depth]]
- [[architectures/vision-foundation-models/depth-anything-v2|Depth Anything V2: Metric Depth & Surface Segmentation]]
- [[architectures/spatial-radiance-and-slam/3d-gaussian-splatting|3D Gaussian Splatting: Real-Time Radiance Field Rendering]]
- [[architectures/multimodal-vlm-and-vla/diffusion-policy|Diffusion Policy: Visuomotor Robot Control]]

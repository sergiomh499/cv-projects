---
title: "Denoising Diffusion Probabilistic Models (DDPM & DDIM): Trajectory Generation"
type: "Technique"
domain: "Generative Vision, Robot Trajectory Synthesis & Physical AI"
tags:
  - technique
  - diffusion-models
  - ddpm
  - ddim
  - trajectory-generation
  - physical-ai
  - score-matching
status: evergreen
updated: 2026-09-09
aliases:
  - "DDPM"
  - "DDIM"
  - "Denoising Diffusion"
  - "Diffusion Policy"
  - "Score-Based Generative Modeling"
---

# 🌫️ Denoising Diffusion (DDPM & DDIM): Score-Based Trajectory Generation

## 1. High-Level Concept & The Multimodal Generation Challenge

In physical robotics, robot manipulation, and generative vision (e.g., [[architectures/multimodal-vlm-and-vla/pi0|pi0]], [[architectures/multimodal-vlm-and-vla/openvla|OpenVLA]], Diffusion Policy, Marigold), an autonomous agent must predict complex continuous action trajectories:

$$
\mathbf{A}_{t:t+H} = \left[ \mathbf{a}_t, \; \mathbf{a}_{t+1}, \; \dots, \; \mathbf{a}_{t+H-1} \right] \in \mathbb{R}^{H \times D}
$$

### The Catastrophic Failure of Mean Regression
Physical action sequences are inherently **multimodal**:
- When encountering an obstacle, a mobile robot can safely steer **left** or **right**. Both actions are valid.
- If a standard neural network is trained with Mean Squared Error ($L_2$ regression), it minimizes expected loss by predicting the **arithmetic average** of the two modes.
- The average of "steer left" and "steer right" is **"steer straight into the obstacle"**, causing catastrophic collisions!

### The Denoising Diffusion Solution
**Denoising Diffusion Probabilistic Models (DDPM)** (Sohl-Dickstein et al. 2015, Ho et al. NeurIPS 2020) frame generation as a reverse thermodynamic diffusion process:
1. **Forward Process (Noise Injection)**: Gradually corrupts structured data $\mathbf{x}_0$ by injecting isotropic Gaussian noise across $T$ discrete timesteps until it becomes pure Gaussian noise $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
2. **Reverse Process (Learned Denoising)**: Trains a neural network $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ to predict the added noise vector at each timestep, which corresponds to the score function $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$.
3. Generates complex, multimodal distributions without mode collapse or adversarial instability.

### The DDIM Acceleration Breakthrough (Song et al., ICLR 2021)
*The DDPM 1,000-Step Bottleneck*: Standard DDPM relies on Markovian assumptions, requiring $T \approx 1,000$ sequential network evaluations to generate a single trajectory, taking several seconds and preventing high-rate ($50\text{ Hz}$) robot control.

**Denoising Diffusion Implicit Models (DDIM)** generalizes the inference process to non-Markovian forward distributions:
- Produces **identical marginal distributions** $q(\mathbf{x}_t \mid \mathbf{x}_0)$ at all timesteps.
- When the forward variance parameter $\sigma_t \to 0$, the reverse process becomes a **deterministic Ordinary Differential Equation (ODE)**!
- Generates high-fidelity trajectories in **10 to 50 steps** (and down to 2–4 steps with modern distillation/flow matching), enabling real-time edge execution.

```
DDPM / DDIM Forward and Reverse Trajectory Diffusion:

Forward Process (Gaussian Noise Injection, Analytic Closed-Form):
   Clean Data x_0 --------> x_1 --------> x_t --------> Pure Noise x_T ~ N(0, I)
      q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

Reverse Generation Process (Learned Score-Based Vector Field):
   Pure Noise x_T --------> x_t --------> x_1 --------> Clean Trajectory x_0
      DDPM: Stochastic Markovian Random Walk (1000 steps, ~5 seconds)
      DDIM: Deterministic Non-Markovian ODE   (10-50 steps, <50 milliseconds!)
```

---

## 2. Mathematical Formulation

### 2.1 The Forward Diffusion Process & Closed-Form Jump
Let $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ be a clean trajectory or image.
Given a variance schedule $\beta_1, \dots, \beta_T \in (0, 1)$:

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left( \mathbf{x}_t; \; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \; \beta_t \mathbf{I} \right)
$$

Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.
Using the reparameterization trick recursively:

$$
\mathbf{x}_t = \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1} = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Any noisy sample $\mathbf{x}_t$ at arbitrary timestep $t \in \{1, \dots, T\}$ is generated **in a single closed-form step without simulating intermediate states**:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\left( \mathbf{x}_t; \; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, \; (1 - \bar{\alpha}_t) \mathbf{I} \right)
$$

---

### 2.2 The Reverse Process & Simplified Objective
The true posterior $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ is Gaussian:

$$
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left( \mathbf{x}_{t-1}; \; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \; \tilde{\beta}_t \mathbf{I} \right)
$$

$$
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t
$$

Substituting $\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon} \right)$ expresses the mean purely in terms of the added noise $\boldsymbol{\epsilon}$:

$$
\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right)
$$

Ho et al. train a neural network $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ using the **Simplified Mean Squared Error Loss**:

$$
\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\left( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \; t \right) \right\|^2 \right]
$$

---

### 2.3 The Deterministic DDIM Sampling Update
DDIM (Song et al., 2021) constructs a non-Markovian sampling trajectory parameterized by variance parameter $\sigma_t$:

$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \sigma_t \boldsymbol{\epsilon}
$$

Setting $\sigma_t = 0$ makes sampling completely deterministic:
1. **Predicted $\mathbf{x}_0$**: $\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}$
2. **Direction pointing to $\mathbf{x}_t$**: $\mathbf{d}_t = \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$
3. **Step update**: $\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{\mathbf{x}}_0 + \mathbf{d}_t$

This ODE allows skipping timesteps (e.g., from $t=1000$ down to $t=0$ in 20 uniform leaps: $t \in [1000, 950, 900, \dots, 0]$).

---

## 3. Python Reference Implementation

```python
import torch
import torch.nn as nn

class DiffusionScheduler:
    """
    DDPM and DDIM noise scheduler and sampling engine.
    Supports both 1000-step stochastic DDPM and 10-50 step deterministic DDIM ODE.
    """
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, 
                 beta_end: float = 0.02, device: str = "cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form forward jump: q(x_t | x_0)."""
        noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    @torch.no_grad()
    def ddim_sample(self, model: nn.Module, shape: tuple[int, ...], 
                    num_inference_steps: int = 20) -> torch.Tensor:
        """
        Fast deterministic DDIM ODE sampling in 20 steps.
        shape: (Batch, Action_Horizon, Action_Dim)
        """
        # Select sub-sequence of timesteps
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = torch.arange(0, self.num_timesteps, step_ratio, device=self.device).flip(0)
        
        x = torch.randn(shape, device=self.device) # Initial Gaussian noise
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            pred_noise = model(x, t_batch)
            
            alpha_bar_t = self.alphas_cumprod[t]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            alpha_bar_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)
            
            # 1. Predict clean x_0
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            
            # 2. Compute direction to x_t (deterministic ODE, sigma=0)
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
            
            # 3. Deterministic step
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
            
        return x
```

---

## 4. Models in the Vault Utilizing Diffusion

- **[[architectures/multimodal-vlm-and-vla/pi0|pi0]]**: Flow matching and diffusion trajectory planning for dexterous robot control.
- **[[architectures/multimodal-vlm-and-vla/openvla|OpenVLA]]**: Vision-Language-Action foundation models with multimodal policy heads.
- **[[techniques/continuous-flow-matching|Continuous Flow Matching (CFM)]]**: High-speed generalization replacing curved Brownian noise paths with straight optimal transport lines.
- **[[cookbooks/20-act-trajectory-chunking/README|Cookbook 20: Action Chunking & Trajectory Generation]]**: Production trajectory generation playbook.

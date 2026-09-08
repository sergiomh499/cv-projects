---
title: "Vision-Language-Action: Historical Evolution & Paradigms"
type: evolution-guide
domain: Vision-Language-Action & Physical AI Robotics
tags:
  - evolution
  - history
  - vla
  - rt1
  - rt2
  - openvla
  - pi0
updated: 2026-09-08
aliases:
  - VLA Evolution Guide
---

# 📜 Vision-Language-Action: Historical Evolution & Paradigms

From behavioral cloning with CNNs to large-scale Vision-Language-Action foundation models.

Related notes: [[topics/vla-and-physical-ai-robotics/00-vla-and-physical-ai-robotics-moc|VLA Robotics MOC]].

---

## 1. Timeline of Physical AI Breakthroughs

```mermaid
timeline
    title Evolution of Vision-Language-Action
    2018 : Deep Imitation Learning : Behavior cloning over low-dimensional states; fragile to distribution shifts.
    2022 : RT-1 (Robotics Transformer) : Transformer backbone over tokenized actions (256 bins) at 3 Hz.
    2023 : RT-2 : Co-fine-tuning PaLM-E / PaLI-X vision-language models to emit action tokens directly.
    2024 : OpenVLA & Octo : Open-source 7B foundation VLA; Octo introduces modular diffusion policies.
    2025-2026 : pi0 and Flow Matching : Continuous flow matching replacing discrete tokens for dexterous, fluid manipulation.
```

---

## 2. Paradigms & Generational Shifts

### 1st Generation: Discrete Tokenization (RT-1, RT-2, Early OpenVLA)

Google's RT-1 (2022) established the Transformer-for-robotics blueprint. Continuous end-effector velocities $\Delta x, \Delta y, \Delta z, \Delta\theta_{roll}, \Delta\theta_{pitch}, \Delta\theta_{yaw}$ were each discretized into **256 uniform bins**, letting the model reuse the cross-entropy language loss:

$$\mathcal{L}_{\text{RT-1}} = -\sum_{d=1}^{D} \sum_{b=1}^{256} \mathbf{1}[a^*_d = b] \log p_\theta(b \mid o_{1:T}, \ell)$$

where $D=11$ action dimensions, $o_{1:T}$ are image observations, and $\ell$ is the language instruction. RT-1 achieved 97% success across 700 real-robot tasks on 130,000 demonstration episodes — remarkable, but jerky: the **discretization error** of $\approx 1.3\,\text{mm}$ per bin made fine-precision assembly tasks (USB insertion, circuit board placement) physically impossible.

RT-2 (2023) scaled to PaLI-X (55B parameters) and co-fine-tuned the vision-language backbone to emit action tokens, demonstrating emergent semantic reasoning: "move the Coke can to where the country that drinks the most wine is" correctly mapped to France on an embedded map. However, inference latency climbed to $\sim 1.8\,\text{s/step}$, limiting deployment to slow pick-and-place tasks.

**Benchmark context**: On SimplerEnv (2024), RT-2 achieves 54.7% task success averaged across visual matching and object manipulation tasks, compared to 29.8% for RT-1, confirming the scaling benefit.

### 2nd Generation: Diffusion Policies (Octo, Chi et al. 2023)

Chi et al. framed action generation as reverse denoising: given noisy action $a_k$ at DDPM step $k$, predict the denoised action:

$$a_{k-1} = \frac{1}{\sqrt{\alpha_k}}\left(a_k - \frac{1-\alpha_k}{\sqrt{1-\bar\alpha_k}}\varepsilon_\theta(a_k, k, c)\right) + \sigma_k z$$

This naturally models **multimodal action distributions** — given "pick the block", the robot might reach from left or right; both modes are preserved in the denoising distribution. Octo (2024) extended this to a 93M parameter open-source transformer backbone pretrained on 800k robot demonstrations from Open X-Embodiment.

**Key limitation**: DDPM requires 20–50 denoising steps → $\sim 200\,\text{ms}$ inference at 30 Hz video, creating a temporal aliasing problem for high-speed manipulation.

### 3rd Generation: Continuous Flow Matching & Action Chunking (π₀, 2024–2026)

Physical Intelligence's π₀ replaces stochastic diffusion with a deterministic **Continuous Flow Matching (CFM)** objective, solving an ODE to generate smooth continuous action chunks of $H = 32$ timesteps in as few as 4 Euler integration steps, enabling $50\,\text{Hz}$ dexterous manipulation.

The flow matching formulation (detailed in [[topics/vla-and-physical-ai-robotics/03-representations-and-open-problems|Flow Matching & Frontiers]]) sets a straight probability path between Gaussian noise and ground-truth action trajectories — in contrast to DDPM's curved Brownian motion paths — dramatically reducing the number of integration steps required.

---

## 3. SOTA Benchmarks: Where Models Stand (2025–2026)

Three canonical benchmarks now define the competitive landscape:

### SimplerEnv (Google DeepMind, 2024)
A photorealistic simulation environment with a calibrated sim-to-real transfer gap, providing standardized evaluation across 25 manipulation tasks in four categories: pick-and-place, object rotation, drawer manipulation, and long-horizon sequences.

| Model | SimplerEnv Avg. | Pick-Place | Long-Horizon |
| :--- | :--- | :--- | :--- |
| RT-1 | 29.8% | 41.2% | 4.7% |
| RT-2 (55B) | 54.7% | 68.1% | 22.3% |
| OpenVLA (7B) | 56.3% | 71.4% | 28.1% |
| **π₀ (3B + CFM)** | **71.8%** | **84.6%** | **49.2%** |

### LIBERO (Liu et al., 2023)
Four structured suites testing spatial generalization (LIBERO-Spatial), object generalization (LIBERO-Object), semantic reasoning (LIBERO-Goal), and long-horizon sequencing (LIBERO-Long). Models are evaluated zero-shot after pretraining, probing generalization rather than task-specific overfitting.

π₀ achieves 76% on LIBERO-Long compared to 44% for Octo and 51% for OpenVLA, confirming that flow-matched action chunks preserve temporal coherence over 8-step sequences where discrete-token methods diverge.

### BridgeData v2 (Walke et al., 2023)
60,096 real robot trajectories collected on a WidowX arm across kitchen and tabletop scenarios. Used as a standard fine-tuning benchmark; models are evaluated on 24 held-out tasks. OpenVLA achieves 56.7% vs. Octo's 50.1% on this benchmark, both substantially ahead of Diffusion Policy's 41.3% when fine-tuned from scratch.

---

## 4. The Action Chunking Horizon Trade-Off: H=16 vs H=64

A critical and often underappreciated design decision is the **action chunk horizon** $H$. Longer horizons provide temporal consistency but reduce responsiveness to environmental feedback:

- **Short horizon ($H = 8$–$16$)**: Policy queries the VLA every $160$–$320\,\text{ms}$ at $5\,\text{Hz}$. High responsiveness to unexpected contact events. Risk: motion discontinuities between chunks appear as micro-jitter at joint level.
- **Medium horizon ($H = 32$)**: The π₀ default. Balances smooth motion (3-4 spline segments) with re-planning every $640\,\text{ms}$. Optimal for lab manipulation benchmarks.
- **Long horizon ($H = 64$)**: Used in quadruped locomotion (Isaac Lab). Policy only queries the brain every $1.28\,\text{s}$ — during which the robot runs fully open-loop. Efficient but catastrophic if a contact disturbance occurs mid-chunk.

The theoretical variance of the final action in a chunk grows as $\sigma^2 H$ under a constant noise model, meaning long-horizon chunks accumulate more trajectory error. Production systems apply **Receding Horizon Execution (RHE)**: predict $H$ steps, execute only the first $K < H$, overlap with the next inference, and blend transitions using cubic Hermite spline interpolation (see [[topics/vla-and-physical-ai-robotics/02-production-pipeline-and-workarounds|Production Pipeline]]).

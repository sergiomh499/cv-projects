---
title: "Implicit Neural Representations (INR) & Fourier Features: Overcoming Spectral Bias"
type: "Technique"
domain: "Neural Fields, 3D Radiance & Coordinate-Based Networks"
tags:
  - technique
  - implicit-neural-representations
  - inr
  - fourier-features
  - positional-encoding
  - nerf
  - siren
  - neural-radiance-fields
status: evergreen
updated: 2026-09-09
aliases:
  - "Fourier Features"
  - "NeRF Positional Encoding"
  - "Implicit Neural Representations"
  - "INR"
  - "SIREN"
---

# 🌐 Implicit Neural Representations & Fourier Features: Overcoming Spectral Bias

## 1. High-Level Concept & The Spectral Bias Catastrophe

In modern 3D vision, neural rendering, and spatial representations (e.g., [[architectures/spatial-radiance-and-slam/instant-ngp|Instant-NGP]], [[architectures/spatial-radiance-and-slam/3dgs-slam|3DGS-SLAM]], NeRF, [[architectures/3d-pointclouds-and-lidar/stream-petr|StreamPETR]]), continuous physical scenes are parameterized as **Coordinate-Based Neural Networks (Implicit Neural Representations - INRs)**:

$$
f_\theta: \mathbf{x} \in \mathbb{R}^d \to \mathbf{y} \in \mathbb{R}^m
$$

For example, a neural radiance field maps a 3D coordinate and 2D viewing direction to RGB color and volume density: $(\mathbf{x}, \mathbf{d}) \mapsto (\mathbf{c}, \sigma)$.

### The Spectral Bias Phenomenon
When a standard Multi-Layer Perceptron (MLP) with ReLU, GELU, or Sigmoid activations is trained directly on raw low-dimensional spatial coordinates $\mathbf{x} = [x, y, z]^T$, it suffers from **Spectral Bias** (Rahaman et al., ICML 2019):
- Under gradient descent, the eigenvalues of the **Neural Tangent Kernel (NTK)** decay exponentially as a function of frequency.
- The network prioritizes learning low-frequency components (overall smooth scene shapes and bulk geometry) at an exponential rate, while requiring an intractable number of iterations to fit high-frequency components.
- The visual result is catastrophic blurring: fine textures, geometric edges, wireframes, specular glints, and surface foliage are smoothed away.

### The Fourier Feature Solution
To overcome spectral bias, the raw continuous spatial coordinates $\mathbf{v} \in [-1, 1]^d$ are projected into a higher-dimensional frequency space via sinusoidal basis functions before passing into the network:

$$
\gamma(\mathbf{v}) = \left[ \sin(2^0 \pi \mathbf{v}), \; \cos(2^0 \pi \mathbf{v}), \; \dots, \; \sin(2^{L-1} \pi \mathbf{v}), \; \cos(2^{L-1} \pi \mathbf{v}) \right]^T
$$

By lifting coordinates to orthogonal Fourier bases:
1. The effective NTK is transformed into a **shift-invariant stationary convolution kernel**.
2. The network's bandwidth is tuned to capture millimeter-level spatial details and sharp step discontinuities.
3. Allows compact 8-layer MLPs to reconstruct photorealistic scenes without blur.

```
Coordinate Encoding Workflow:

Raw Spatial Point (x, y, z) in R^3
             |
             v
[ Fourier Feature / Positional Encoding: gamma(v) ]
- Transforms R^3 into R^(2 * L * 3) (e.g. L=10 -> 60 dimensions)
- sin(2^k * pi * v) and cos(2^k * pi * v)
             |
             v
[ Fully-Connected MLP / Coordinate Network ]
- High-frequency NTK bandwidth enabled
             |
             v
Exact High-Frequency Output: Density sigma, Radiance RGB, Signed Distance Function (SDF)
```

---

## 2. Mathematical Formulation

### 2.1 NeRF Multi-Scale Positional Encoding
Let $\mathbf{p} \in \mathbb{R}^3$ be a normalized spatial coordinate. Mildenhall et al. (ECCV 2020) define the deterministic logarithmic frequency mapping $\gamma: \mathbb{R} \to \mathbb{R}^{2L}$:

$$
\gamma(p) = \left( \sin\left(2^0 \pi p\right), \; \cos\left(2^0 \pi p\right), \; \dots, \; \sin\left(2^{L-1} \pi p\right), \; \cos\left(2^{L-1} \pi p\right) \right)
$$

For a 3D coordinate vector $\mathbf{p} = [x, y, z]^T$, $\gamma$ is applied component-wise, expanding the 3-dimensional coordinate into a $3 \times 2L = 6L$-dimensional feature vector (typically $L=10$ for spatial coordinates, yielding $60\text{ dimensions}$, and $L=4$ for 2D viewing directions, yielding $24\text{ dimensions}$).

---

### 2.2 Random Fourier Features (RFF) & NTK Bandwidth
Tancik et al. (NeurIPS 2020) generalize positional encoding using random Gaussian matrices. Let $\mathbf{B} \in \mathbb{R}^{m \times d}$ be a matrix whose entries are sampled from a zero-mean isotropic Gaussian:

$$
\mathbf{B}_{ij} \sim \mathcal{N}(0, \; \sigma^2)
$$

The mapping $\gamma_{\mathbf{B}}(\mathbf{v}): \mathbb{R}^d \to \mathbb{R}^{2m}$ is given by:

$$
\gamma_{\mathbf{B}}(\mathbf{v}) = \begin{bmatrix} \cos(2\pi \mathbf{B} \mathbf{v}) \\ \sin(2\pi \mathbf{B} \mathbf{v}) \end{bmatrix}
$$

#### Stationary NTK Kernel Derivation
The inner product between two encoded points $\mathbf{v}_1, \mathbf{v}_2 \in \mathbb{R}^d$ satisfies:

$$
k(\mathbf{v}_1, \mathbf{v}_2) = \frac{1}{m} \gamma_{\mathbf{B}}(\mathbf{v}_1)^T \gamma_{\mathbf{B}}(\mathbf{v}_2) = \frac{1}{m} \sum_{i=1}^m \cos\left(2\pi \mathbf{b}_i^T (\mathbf{v}_1 - \mathbf{v}_2)\right)
$$

By Bochner's Theorem, as $m \to \infty$, this inner product converges to a stationary Gaussian radial basis function (RBF) kernel:

$$
k(\mathbf{v}_1, \mathbf{v}_2) \approx \exp\left( -2\pi^2 \sigma^2 \left\| \mathbf{v}_1 - \mathbf{v}_2 \right\|^2 \right)
$$

The hyperparameter $\sigma$ directly controls the **spectral bandwidth** of the neural representation:
- Small $\sigma$ ($\sigma \le 1.0$): Under-fits high frequencies (smooth, blurred outputs).
- Optimal $\sigma$ ($\sigma \in [10, 30]$): Accurately reconstructs complex textures, geometric edges, and fine specularities.
- Excessively high $\sigma$ ($\sigma > 100$): Over-fits to high-frequency noise, causing aliasing artifacts.

---

### 2.3 Periodic Activation Functions: SIREN
Instead of encoding the inputs, Sitzmann et al. (NeurIPS 2020) introduce **SIREN (Sinusoidal Representation Networks)**, which use sine non-linearities at every layer:

$$
\mathbf{y}^{(l)} = \sin\left( \omega_0 \left( \mathbf{W}^{(l)} \mathbf{y}^{(l-1)} + \mathbf{b}^{(l)} \right) \right)
$$

#### Mathematical Properties of SIREN
1. **Differentiability**: The derivative of a sine wave is a cosine wave (a phase-shifted sine wave). Therefore, the spatial gradient $\nabla_\mathbf{x} f_\theta(\mathbf{x})$, Hessian $\nabla^2_\mathbf{x} f_\theta(\mathbf{x})$, and Laplacian $\Delta f_\theta(\mathbf{x})$ are themselves SIREN networks of identical capacity!
2. This makes SIREN the premier architecture for physics-informed neural networks (PINNs), boundary value problems (Helmholtz and Wave equations), and Signed Distance Function (SDF) normal supervision without finite differences.

---

## 3. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class NeRFPositionalEncoding(nn.Module):
    """
    Standard NeRF Sinusoidal Positional Encoding.
    Maps [B, in_dim] -> [B, in_dim * 2 * num_frequencies]
    """
    def __init__(self, in_dim: int = 3, num_frequencies: int = 10):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        # Frequency bands: 2^0, 2^1, ..., 2^(L-1)
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        # output: [B, in_dim * 2 * L]
        scaled = x.unsqueeze(-1) * self.freq_bands * torch.pi # [B, in_dim, L]
        sin_feat = torch.sin(scaled)
        cos_feat = torch.cos(scaled)
        encoded = torch.cat([sin_feat, cos_feat], dim=-1) # [B, in_dim, 2L]
        return encoded.view(x.shape[0], -1)


class SirenLayer(nn.Module):
    """
    SIREN Layer with periodic sine activation and special uniform initialization.
    """
    def __init__(self, in_features: int, out_features: int, 
                 omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer initialization: [-1/in, 1/in]
                bound = 1.0 / self.linear.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Hidden layer initialization to preserve variance: [-sqrt(6/in)/w0, sqrt(6/in)/w0]
                bound = np.sqrt(6.0 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))
```

---

## 4. Models in the Vault Utilizing Implicit Encodings

- **[[architectures/spatial-radiance-and-slam/instant-ngp|Instant-NGP]]**: Fuses multi-resolution hash encodings with frequency basis projections for sub-millisecond radiance field queries.
- **[[architectures/spatial-radiance-and-slam/3dgs-slam|3DGS-SLAM]] & [[architectures/spatial-radiance-and-slam/splatam|SplaTAM]]**: Uses coordinate basis representations for visual tracking and geometric depth completion.
- **[[architectures/3d-pointclouds-and-lidar/stream-petr|StreamPETR]]**: 3D perception and spatial modeling with continuous neural coordinate queries.
- **[[techniques/multiresolution-hash-encodings-and-sparse-voxels|Multiresolution Spatial Hash Encodings]]**: Spatial hashing alternative delivering $\mathcal{O}(1)$ query time.

---
title: "Sinusoidal Phase-Shifting Profilometry & Multi-Frequency Phase Unwrapping: 3D Metrology"
type: "Technique"
domain: "Active 3D Sensing, Structured Light & Industrial Metrology"
tags:
  - technique
  - structured-light
  - phase-shifting
  - phase-unwrapping
  - 3d-metrology
  - active-sensing
  - fringe-projection
status: evergreen
updated: 2026-09-09
aliases:
  - "Phase-Shifting Profilometry"
  - "PSP"
  - "Fringe Projection Profilometry"
  - "Multi-Frequency Phase Unwrapping"
---

# 💡 Sinusoidal Phase-Shifting Profilometry & Multi-Frequency Phase Unwrapping

## 1. High-Level Concept & The Sub-Pixel Metrology Requirement

In industrial defect inspection, semiconductor metrology, robotics bin-picking, and dental 3D scanning, commodity active sensors (e.g., consumer Time-of-Flight or passive stereo cameras) fail because their depth accuracy is bounded to millimeters ($\pm 1\text{--}5\text{ mm}$), and their point clouds degrade near dark surfaces, specular reflections, or textureless regions.

In contrast, **Phase-Shifting Profilometry (PSP)** (or Fringe Projection Profilometry) achieves **sub-10-micrometer ($<10\ \mu\text{m}$) depth precision**:
1. **Periodic Sinusoidal Projection**: A digital projector casts high-frequency sinusoidal light fringe patterns onto the physical target scene.
2. **Phase Modulation by 3D Topography**: As light strikes a non-planar 3D object, surface height variations geometrically distort and phase-shift the sinusoidal waves observed by a calibrated camera.
3. **Pointwise Trigonometric Decoding**: By projecting $N$ phase-shifted fringe patterns ($\delta_n = \frac{2\pi n}{N}$), the surface phase $\phi(x, y)$ is extracted independently at every single pixel via closed-form arctangent trigonometry.
4. **Complete Invariance to Surface Reflectance**: Ambient room illumination $A(x, y)$ and uneven surface texture/albedo $B(x, y)$ cancel out mathematically in the arctangent numerator and denominator!
5. **Multi-Frequency Temporal Phase Unwrapping**: Resolves the $[-\pi, +\pi)$ phase wrapping ambiguity using synthetic beat frequencies, producing absolute continuous phase $\Phi(x, y)$ mapped directly to metric coordinates $(X, Y, Z)$ via calibrated ray-plane triangulation.

```
Phase-Shifting Profilometry Pipeline:

Projector: Casts N Sinusoidal Fringes with Phase Shifts delta_n = 2*pi*n/N
                               |
                               v
                       [ 3D Object Surface ] (Distorts fringes according to topography)
                               |
                               v
Camera: Captures N Images I_0, I_1, ..., I_N-1
                               |
                               v
[ Wrapped Phase Calculation: phi(x,y) = -arctan( sum I_n sin(delta_n) / sum I_n cos(delta_n) ) ]
      (Surface texture & ambient lighting cancel out mathematically!)
                               |
                               v
[ Multi-Frequency Heterodyne Unwrapping: Phi(x,y) = phi(x,y) + 2*pi*k(x,y) ]
                               |
                               v
[ Ray-Plane Triangulation ] ---> Sub-10-Micrometer Dense Metric Point Cloud!
```

---

## 2. Mathematical Formulation

### 2.1 The N-Step Phase-Shifted Intensity Model
The intensity observed by a camera pixel $(x, y)$ for the $n$-th projected fringe pattern ($n \in \{0, 1, \dots, N-1\}$) is modeled as:

$$
I_n(x, y) = A(x, y) + B(x, y) \cos\left( \phi(x, y) - \delta_n \right)
$$

where:
- $A(x, y)$ is the background ambient illumination plus DC projector intensity offset.
- $B(x, y)$ is the fringe modulation amplitude (governed by surface albedo, camera exposure, and projector focus).
- $\phi(x, y)$ is the topography-dependent phase value to be retrieved.
- $\delta_n = \frac{2\pi n}{N}$ is the calibrated phase-shift angle.

Using the cosine angle addition formula $\cos(\alpha - \beta) = \cos\alpha \cos\beta + \sin\alpha \sin\beta$:

$$
I_n(x, y) = A(x, y) + B(x, y) \cos\phi(x, y) \cos\delta_n + B(x, y) \sin\phi(x, y) \sin\delta_n
$$

---

### 2.2 Closed-Form Wrapped Phase Extraction
Multiplying $I_n$ by $\sin\delta_n$ and $\cos\delta_n$ and summing over all $N$ steps eliminates $A(x, y)$ via the orthogonality of sinusoidal basis functions:

$$
S(x, y) = \sum_{n=0}^{N-1} I_n(x, y) \sin\delta_n = - \frac{N}{2} B(x, y) \sin\phi(x, y)
$$

$$
C(x, y) = \sum_{n=0}^{N-1} I_n(x, y) \cos\delta_n = \frac{N}{2} B(x, y) \cos\phi(x, y)
$$

Taking the ratio eliminates the surface reflectivity $B(x, y)$:

$$
\frac{-S(x, y)}{C(x, y)} = \frac{\sin\phi(x, y)}{\cos\phi(x, y)} = \tan\phi(x, y)
$$

The wrapped phase $\phi(x, y) \in [-\pi, +\pi)$ is computed directly using the four-quadrant `arctan2` function:

$$
\phi(x, y) = \text{atan2}\left( -\sum_{n=0}^{N-1} I_n(x, y) \sin\left(\frac{2\pi n}{N}\right), \; \sum_{n=0}^{N-1} I_n(x, y) \cos\left(\frac{2\pi n}{N}\right) \right)
$$

#### 4-Step Standard Formulation ($N = 4, \delta = [0, \pi/2, \pi, 3\pi/2]$):
$$
\phi(x, y) = \text{atan2}\left( I_3 - I_1, \; I_0 - I_2 \right)
$$

#### Modulation Quality Metric
The modulation amplitude $B(x, y)$ serves as a confidence metric to reject saturated pixels or shadows:

$$
B(x, y) = \frac{2}{N} \sqrt{ \left( \sum_{n=0}^{N-1} I_n \sin\delta_n \right)^2 + \left( \sum_{n=0}^{N-1} I_n \cos\delta_n \right)^2 }
$$

---

### 2.3 Multi-Frequency Heterodyne Phase Unwrapping
Because the arctangent function is periodic, $\phi(x, y)$ is wrapped into $[-\pi, +\pi)$. The true continuous absolute phase $\Phi(x, y)$ is related to the wrapped phase by:

$$
\Phi(x, y) = \phi(x, y) + 2\pi \cdot k(x, y)
$$

where $k(x, y) \in \mathbb{Z}$ is the unknown integer fringe order.

To unwrap the phase without spatial path integration errors across depth discontinuities, **multi-frequency heterodyne projection** projects two high-frequency fringe patterns with periods $\lambda_1$ and $\lambda_2$ (where $\lambda_1 > \lambda_2$).

A synthetic beat wavelength $\lambda_{12}$ is generated:

$$
\lambda_{12} = \frac{\lambda_1 \lambda_2}{|\lambda_1 - \lambda_2|}
$$

The synthetic beat phase $\phi_{12}(x, y)$ spans the entire measurement range without wrapping:

$$
\phi_{12}(x, y) = \begin{cases} \phi_1(x, y) - \phi_2(x, y) & \text{if } \phi_1 \ge \phi_2 \\ \phi_1(x, y) - \phi_2(x, y) + 2\pi & \text{if } \phi_1 < \phi_2 \end{cases}
$$

The integer fringe order $k_1(x, y)$ for high-frequency fringe 1 is determined point-wise:

$$
k_1(x, y) = \text{round}\left( \frac{\frac{\lambda_{12}}{\lambda_1} \phi_{12}(x, y) - \phi_1(x, y)}{2\pi} \right)
$$

The true absolute continuous phase is then:

$$
\Phi(x, y) = \phi_1(x, y) + 2\pi k_1(x, y)
$$

---

## 3. Python Reference Implementation

```python
import numpy as np

def four_step_phase_shift(I0: np.ndarray, I1: np.ndarray, I2: np.ndarray, I3: np.ndarray):
    """
    Computes wrapped phase and modulation amplitude from 4 phase-shifted fringe images.
    delta = [0, pi/2, pi, 3*pi/2]
    Returns: (phi_wrapped in [-pi, pi], modulation amplitude B)
    """
    # Numerator = I3 - I1
    num = I3.astype(np.float64) - I1.astype(np.float64)
    # Denominator = I0 - I2
    den = I0.astype(np.float64) - I2.astype(np.float64)
    
    # Wrapped phase via atan2
    phi = np.arctan2(num, den)
    
    # Modulation amplitude
    B = 0.5 * np.sqrt(num ** 2 + den ** 2)
    return phi, B

def multi_frequency_heterodyne_unwrap(phi1: np.ndarray, phi2: np.ndarray, lambda1: float, lambda2: float):
    """
    Unwraps high-frequency phase phi1 using synthetic heterodyne beat phase phi12.
    lambda1, lambda2: fringe periods in pixels (e.g., lambda1 = 70, lambda2 = 64)
    """
    # 1. Beat period
    lambda12 = (lambda1 * lambda2) / np.abs(lambda1 - lambda2)
    
    # 2. Beat phase phi12 in [0, 2*pi)
    phi12 = np.where(phi1 >= phi2, phi1 - phi2, phi1 - phi2 + 2.0 * np.pi)
    
    # 3. Fringe order integer k1
    k1 = np.round(((lambda12 / lambda1) * phi12 - phi1) / (2.0 * np.pi))
    
    # 4. Absolute continuous phase
    Phi = phi1 + 2.0 * np.pi * k1
    return Phi
```

---

## 4. Models & Playbooks Utilizing Structured Light

- **[[topics/active-3d-sensing-and-structured-light/00-active-3d-sensing-and-structured-light-moc|Active 3D Sensing MOC]]**: Complete system design for sub-10-micron industrial surface metrology.
- **[[topics/thermal-and-hyperspectral-vision/README|Thermal & Active Sensing]]**: Fusing active phase-shifting patterns with long-wave infrared sensors for non-destructive testing.

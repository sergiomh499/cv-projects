---
title: "Optical & Scene Flow: Classical Variational & Hybrid Methods"
type: production-playbook
domain: Optical Flow & Scene Flow Perception
tags:
  - optical-flow
  - variational
  - horn-schunck
  - epipolar
  - hybrid-pipelines
updated: 2026-09-08
aliases:
  - Flow Classical & Hybrid Methods
---

# 📐 Optical & Scene Flow: Classical Variational & Hybrid Methods

Rigorous mathematical derivations of classical differential optical flow (Horn-Schunck, Lucas-Kanade), RAFT correlation volume construction, and hybrid ego-motion epipolar decoupling.

Related notes: [[topics/optical-and-scene-flow-perception/00-optical-and-scene-flow-perception-moc|Optical Flow MOC]], [[topics/slam-and-spatial-perception/04-classical-and-hybrid-methods|SLAM Classical Foundations]].

---

## 1. Classical Differential Calculus: Horn-Schunck vs. Lucas-Kanade

### The Optical Flow Brightness Constancy Constraint

Assuming a physical surface patch retains constant irradiance over an infinitesimal time step $\Delta t$:

$$I(x + \Delta x, y + \Delta y, t + \Delta t) = I(x, y, t)$$

Performing a first-order Taylor series expansion and canceling $I(x,y,t)$:

$$\frac{\partial I}{\partial x}\Delta x + \frac{\partial I}{\partial y}\Delta y + \frac{\partial I}{\partial t}\Delta t + \mathcal{O}(\Delta^2) = 0$$

Dividing by $\Delta t$ and taking the limit $\Delta t \to 0$ yields the fundamental **Optical Flow Constraint Equation (OFCE)**:

$$\boxed{I_x u + I_y v + I_t = 0 \quad \text{or equivalently} \quad \nabla I \cdot \mathbf{v} + I_t = 0}$$

Where $\mathbf{v} = [u, v]^T = [\dot{x}, \dot{y}]^T$ is the 2D optical flow velocity. This single scalar equation is underconstrained (1 equation, 2 unknowns) — the **Aperture Problem**.

### Horn-Schunck: Global Variational Functional

Horn & Schunck introduced a global **spatial smoothness regularizer** $\|\nabla u\|_2^2 + \|\nabla v\|_2^2$ to resolve underconstrained flow:

$$E(u, v) = \iint \left[ \underbrace{(I_x u + I_y v + I_t)^2}_{\text{data term}} + \alpha^2 \underbrace{(u_x^2 + u_y^2 + v_x^2 + v_y^2)}_{\text{smoothness term}} \right] dx\, dy$$

The hyperparameter $\alpha^2$ balances data fidelity vs. spatial coherence. Large $\alpha$ → very smooth flow (oversmooths motion boundaries); small $\alpha$ → noisy but sharp.

Applying the **Euler-Lagrange equations** to minimize $E$ with respect to $(u, v)$:

$$I_x(I_x u + I_y v + I_t) - \alpha^2 \Delta u = 0$$
$$I_y(I_x u + I_y v + I_t) - \alpha^2 \Delta v = 0$$

where $\Delta u = u_{xx} + u_{yy}$ is the Laplacian of $u$. Approximating $\Delta u \approx \bar{u} - u$ (local 4-connected average minus center):

$$u^{k+1} = \bar{u}^k - \frac{I_x(I_x \bar{u}^k + I_y \bar{v}^k + I_t)}{\alpha^2 + I_x^2 + I_y^2}$$
$$v^{k+1} = \bar{v}^k - \frac{I_y(I_x \bar{u}^k + I_y \bar{v}^k + I_t)}{\alpha^2 + I_x^2 + I_y^2}$$

**Convergence**: Gauss-Seidel iteration; typically 100–1000 iterations required. Dominated by the denominator $\alpha^2 + I_x^2 + I_y^2$ — in low-gradient regions ($I_x \approx I_y \approx 0$), the smoothness term dominates and flow propagates from textured neighbors.

**Physical interpretation of $\alpha$**: In a pixel with $|I_x| = |I_y| = 5\,\text{DN/px}$ and $\alpha = 1$: the data term weight is $\approx 0.96$, and smoothness weight is $\approx 0.04$ — data dominates. In a textureless pixel with $|I_x| = |I_y| = 0.1$: data weight $\approx 0.002$, smoothness $\approx 0.998$ — the estimate is essentially the neighborhood average.

### Lucas-Kanade: Local Least-Squares Estimation

Lucas & Kanade assume $(u, v)$ is constant in a $w \times w$ window ($w = 7$ typical). The $w^2$ OFCE equations form the overdetermined system:

$$\begin{pmatrix} I_x(x_1,y_1) & I_y(x_1,y_1) \\ I_x(x_2,y_2) & I_y(x_2,y_2) \\ \vdots & \vdots \end{pmatrix} \begin{pmatrix} u \\ v \end{pmatrix} = -\begin{pmatrix} I_t(x_1,y_1) \\ I_t(x_2,y_2) \\ \vdots \end{pmatrix}$$

The weighted least-squares solution (weights $W_k$ from a Gaussian window):

$$\begin{pmatrix} u \\ v \end{pmatrix} = \mathbf{M}^{-1} \mathbf{b}, \quad \mathbf{M} = \begin{pmatrix} \sum W_k I_x^2 & \sum W_k I_x I_y \\ \sum W_k I_x I_y & \sum W_k I_y^2 \end{pmatrix}, \quad \mathbf{b} = -\begin{pmatrix} \sum W_k I_x I_t \\ \sum W_k I_y I_t \end{pmatrix}$$

**Reliability criterion**: $\mathbf{M}$ is the Harris structure tensor. Flow is reliable when both eigenvalues $\lambda_1, \lambda_2$ are large (corner regions). The matrix is **singular** (aperture problem) when:
- $\lambda_1 \approx 0$: textureless region — both eigenvalues near zero.
- $\lambda_2 \gg \lambda_1 \approx 0$: straight edge — only normal component computable.

**LK vs. HS comparison**:
- HS: global solve, handles textureless regions via diffusion, blurs motion boundaries. $O(N \cdot n_{\text{iter}})$ where $N$ = pixel count.
- LK: local solve, fails in textureless regions (singular $\mathbf{M}$), preserves sharp motion boundaries. $O(N \cdot w^2)$ per iteration, parallelizable.
- Production: LK with coarse-to-fine pyramid (KLT tracker) remains the standard for **sparse feature tracking** (e.g., OpenCV `calcOpticalFlowPyrLK`); HS-variants used in variational dense flow (TV-L1, DeepFlow).

```python
import cv2
import numpy as np

# Production Lucas-Kanade sparse tracker (KLT)
def klt_track(frame_prev: np.ndarray, frame_curr: np.ndarray,
              corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Track sparse corners using pyramidal LK. Returns (tracked_pts, valid_mask)."""
    lk_params = dict(
        winSize=(21, 21),         # 21x21 window for robustness
        maxLevel=4,               # 4-level pyramid for large displacements
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        minEigThreshold=1e-4      # Reject near-singular M matrices
    )
    pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
        frame_prev, frame_curr, corners.astype(np.float32), None, **lk_params)
    # Forward-backward verification
    pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
        frame_curr, frame_prev, pts_curr, None, **lk_params)
    fb_error = np.abs(corners - pts_back).max(axis=1)
    valid = (status.ravel() == 1) & (status_back.ravel() == 1) & (fb_error < 1.0)
    return pts_curr[valid], valid
```

---

## 2. Production Hybrid Pattern: Epipolar Ego-Motion Decoupling

In autonomous driving and drone navigation, total optical flow $F_{\text{total}}$ superimposes two independent sources:
1. **Camera Ego-Motion Rigid Flow**: Caused by vehicle/drone velocity and rotation.
2. **Independent Object Motion Non-Rigid Flow**: Pedestrians, other vehicles, moving machinery.

```mermaid
flowchart LR
    TotalFlow["Deep Optical Flow Field: RAFT / GMFlow"] --> Epipolar["Step 1: Classical 8-Point Essential Matrix + IMU Gyro"]
    Epipolar --> RigidFlow["Analytical Rigid Ego-Motion Flow Field"]
    TotalFlow --> Subtraction["Step 2: Vector Subtraction F_residual = F_total - F_rigid"]
    RigidFlow --> Subtraction
    Subtraction --> ResidualThreshold["Step 3: Chi-Square Residual Thresholding"]
    ResidualThreshold --> MovingObjects["Pure Independent Moving Object Segmentation"]
```

### Exact Rigid Flow Formulation

Given camera translational velocity $v = [v_x, v_y, v_z]^T$ and rotational velocity $\omega = [\omega_x, \omega_y, \omega_z]^T$ from the vehicle CAN-bus + IMU at $200\,\text{Hz}$, and depth $Z$ at each pixel from a LiDAR or stereo depth map, the **rigid ego-motion flow** at image point $(x, y)$ is computed analytically via the differential motion field equations:

$$u_{\text{rigid}} = \frac{x\,v_z - f\,v_x}{Z} + \frac{xy}{f}\,\omega_x - \!\left(f + \frac{x^2}{f}\right)\!\omega_y + y\,\omega_z$$

$$v_{\text{rigid}} = \frac{y\,v_z - f\,v_y}{Z} + \!\left(f + \frac{y^2}{f}\right)\!\omega_x - \frac{xy}{f}\,\omega_y - x\,\omega_z$$

Where $f$ is the camera focal length in pixels and $(x,y)$ are pixel coordinates relative to the principal point.

Subtracting $F_{\text{rigid}}$ from the deep flow immediately isolates true independent moving obstacles:

$$F_{\text{residual}}(x,y) = F_{\text{total}}(x,y) - [u_{\text{rigid}}(x,y),\; v_{\text{rigid}}(x,y)]^T$$

Pixels where $\|F_{\text{residual}}\|_2 > \tau_{\text{moving}}$ ($\tau = 2.5\,\text{px}$ typical) are classified as independently moving objects.

**Production result**: On KITTI Moving Object Segmentation benchmark, this hybrid approach achieves 91.3% IoU for moving vehicle detection, compared to 78.4% for pure deep segmentation (Mask2Former) — because the rigid flow subtraction directly computes physical 3D motion rather than relying on appearance-based features.

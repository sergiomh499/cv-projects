---
title: "Variational Optical Flow & Total Variation Regularization: Horn-Schunck to TV-L1"
type: "Technique"
domain: "Visual Motion, Classical Vision & Convex Optimization"
tags:
  - technique
  - optical-flow
  - total-variation
  - tv-l1
  - horn-schunck
  - primal-dual
  - convex-optimization
status: evergreen
updated: 2026-09-09
aliases:
  - "Variational Optical Flow"
  - "TV-L1 Optical Flow"
  - "Horn-Schunck Flow"
  - "Total Variation Motion"
---

# 🌀 Variational Optical Flow & Total Variation Regularization: Horn-Schunck to TV-$L^1$

## 1. High-Level Concept & The Aperture Problem

Estimating the 2D apparent velocity field $\mathbf{u}(x, y) = (u, v)$ of physical objects across consecutive image frames $I(x, y, t)$ and $I(x, y, t+1)$ is governed by the **Brightness Constancy Assumption**:

$$
I(x + u, y + v, t + 1) = I(x, y, t)
$$

Taking the first-order Taylor expansion yields the fundamental **Optical Flow Constraint Equation (OFCE)**:

$$
I_x u + I_y v + I_t = \nabla I \cdot \mathbf{u} + I_t = 0
$$

where $I_x = \frac{\partial I}{\partial x}$, $I_y = \frac{\partial I}{\partial y}$, and $I_t = \frac{\partial I}{\partial t}$.

### The Aperture Problem
At every pixel, the OFCE provides a single linear scalar equation with two unknowns $(u, v)$. 
- It constrains motion strictly along the image gradient direction $\nabla I$ (normal to edges).
- Motion parallel to edges is mathematically unconstrained and unobservable from local intensity changes alone.

```
The Aperture Problem:
                       Edge Motion
                         |
                 ========|========  <--- Straight Edge
                         |
                         v
     (We observe normal motion; motion along the edge is invisible!)
```

---

## 2. Evolution of Variational Regularization

### 2.1 The Horn-Schunck Formulation (1981)
To make the ill-posed system solvable, Horn and Schunck introduced a global energy minimization framework with **quadratic Dirichlet smoothness**:

$$
E_{\text{HS}}(u, v) = \iint_{\Omega} \underbrace{\left( I_x u + I_y v + I_t \right)^2}_{\text{Quadratic Data Term } (L^2)} + \; \alpha^2 \underbrace{\left( \|\nabla u\|^2 + \|\nabla v\|^2 \right)}_{\text{Dirichlet Smoothness } (H^1)} dx dy
$$

#### The Failure of Quadratic Regularization
While mathematically elegant and solvable via Jacobi relaxation, the squared gradient penalty $\|\nabla u\|^2 = u_x^2 + u_y^2$ severely penalizes sharp motion transitions:
- Across physical object boundaries (e.g., a foreground car moving past a static building), the true velocity field exhibits an instantaneous step discontinuity ($u_x \to \infty$).
- Horn-Schunck heavily penalizes this step, forcing the motion of the moving car to **oversmooth and bleed into the static background**, wiping out crisp object contours.

---

### 2.2 The Modern TV-$L^1$ Formulation (Zach, Pock, Bischof, 2007)
The **TV-$L^1$ model** revolutionizes variational motion estimation by replacing both the quadratic data fidelity and quadratic smoothness terms with **$L^1$ and Total Variation (TV) penalties**:

$$
E_{\text{TV}-L^1}(\mathbf{u}) = \int_{\Omega} \underbrace{\|\nabla \mathbf{u}\|}_{\text{Total Variation Regularizer}} dx + \; \lambda \int_{\Omega} \underbrace{|\rho(\mathbf{u})|}_{L^1 \text{ Data Fidelity}} dx
$$

where:
1. **$L^1$ Data Fidelity**: $|\rho(\mathbf{u})| = |I_x u + I_y v + I_t|$ is exceptionally robust to outliers, specular glares, and partial occlusions (unlike squared $L^2$ which explodes on outliers).
2. **Total Variation (TV)**: $\|\nabla \mathbf{u}\| = \sqrt{u_x^2 + u_y^2} + \sqrt{v_x^2 + v_y^2}$. Because TV penalizes gradients linearly rather than quadratically, it **permits sharp jump discontinuities**, preserving razor-sharp object boundaries!

```
Horn-Schunck (Quadratic) vs. TV-L1 (Total Variation) at Boundaries:

True Motion Field:       Horn-Schunck (L2):        TV-L1 (Total Variation):
      |                        /                        |
10 px |-------           10 px/                 10 px   |-------
      |                      /                          |
 0 px |_______            0 px/                  0 px   |_______
    Boundary                 Boundary                 Boundary
(Step Discontinuity)     (Severely Oversmoothed!)   (Preserved Razor-Sharp!)
```

---

## 3. Primal-Dual Numerical Optimization (Chambolle-Pock)

Because both the TV term and $L^1$ norm are non-differentiable at zero, standard gradient descent diverges. 
The **Chambolle-Pock Primal-Dual Algorithm** decouples the optimization using an auxiliary flow field $\mathbf{v}$ and dual variables $\mathbf{p} = (p_x, p_y)$:

$$
\min_{\mathbf{u}, \mathbf{v}} \int_{\Omega} \|\nabla \mathbf{u}\| + \frac{1}{2\theta} \|\mathbf{u} - \mathbf{v}\|^2 + \lambda \int_{\Omega} |\rho(\mathbf{v})| dx
$$

### Step 1: Update $\mathbf{u}$ with Fixed $\mathbf{v}$ (Dual Projection)
The dual variable $\mathbf{p}$ is updated via gradient ascent and projected onto the $L^\infty$ unit ball:

$$
\mathbf{p}^{n+1} = \frac{\mathbf{p}^n + \sigma \nabla \mathbf{u}^n}{\max\left(1, \; \|\mathbf{p}^n + \sigma \nabla \mathbf{u}^n\|\right)}
$$

The flow $\mathbf{u}$ is updated via the divergence operator:

$$
\mathbf{u}^{n+1} = \mathbf{v} + \theta \cdot \text{div}(\mathbf{p}^{n+1})
$$

---

### Step 2: Update $\mathbf{v}$ with Fixed $\mathbf{u}$ (Pointwise Soft-Thresholding / Shrinkage)
With $\mathbf{u}$ fixed, each pixel solves the decoupled 1D optimization problem:

$$
\min_{\mathbf{v}} \frac{1}{2\theta} \|\mathbf{v} - \mathbf{u}\|^2 + \lambda |\rho(\mathbf{v})|
$$

This has an exact **closed-form analytical solution** via pointwise soft-thresholding:

Let residual $\rho(\mathbf{u}) = I_x (u - u_0) + I_y (v - v_0) + I_t$ and gradient magnitude $\gamma = I_x^2 + I_y^2$.

$$
\mathbf{v} = \mathbf{u} + \begin{cases}
\lambda \theta \nabla I & \text{if } \rho(\mathbf{u}) < -\lambda \theta \gamma \\
-\lambda \theta \nabla I & \text{if } \rho(\mathbf{u}) > \lambda \theta \gamma \\
-\frac{\rho(\mathbf{u})}{\gamma} \nabla I & \text{if } |\rho(\mathbf{u})| \le \lambda \theta \gamma \quad (\gamma > 0)
\end{cases}
$$

This requires zero matrix inversions and executes in parallel at over **$120\text{ FPS}$ on edge GPUs**.

---

## 4. Python Reference Implementation

```python
import numpy as np

def tv_l1_optical_flow_step(u: np.ndarray, v: np.ndarray, p: np.ndarray, 
                            Ix: np.ndarray, Iy: np.ndarray, It: np.ndarray,
                            theta: float = 0.3, lambda_val: float = 40.0, sigma: float = 0.25):
    """
    One iteration of Primal-Dual TV-L1 optical flow solver.
    u: [H, W, 2] flow field (u, v)
    p: [H, W, 2, 2] dual variables (px_u, py_u, px_v, py_v)
    Ix, Iy, It: [H, W] spatial and temporal image gradients
    """
    h, w, _ = u.shape
    grad_norm_sq = Ix**2 + Iy**2 + 1e-6
    
    # 1. Forward differences of u: grad_u [H, W, 2, 2]
    # grad_u[..., 0] = [ux, uy], grad_u[..., 1] = [vx, vy]
    ux = np.pad(u[:, 1:, 0] - u[:, :-1, 0], ((0, 0), (0, 1)), mode="constant")
    uy = np.pad(u[1:, :, 0] - u[:-1, :, 0], ((0, 1), (0, 0)), mode="constant")
    vx = np.pad(u[:, 1:, 1] - u[:, :-1, 1], ((0, 0), (0, 1)), mode="constant")
    vy = np.pad(u[1:, :, 1] - u[:-1, :, 1], ((0, 1), (0, 0)), mode="constant")
    
    # 2. Dual update with projection onto unit ball
    p[..., 0, 0] = (p[..., 0, 0] + sigma * ux) / np.maximum(1.0, np.abs(p[..., 0, 0] + sigma * ux))
    p[..., 1, 0] = (p[..., 1, 0] + sigma * uy) / np.maximum(1.0, np.abs(p[..., 1, 0] + sigma * uy))
    p[..., 0, 1] = (p[..., 0, 1] + sigma * vx) / np.maximum(1.0, np.abs(p[..., 0, 1] + sigma * vx))
    p[..., 1, 1] = (p[..., 1, 1] + sigma * vy) / np.maximum(1.0, np.abs(p[..., 1, 1] + sigma * vy))
    
    # 3. Divergence of dual variables: div_p [H, W, 2]
    div_u = np.pad(p[:, 1:, 0, 0] - p[:, :-1, 0, 0], ((0, 0), (1, 0)), mode="constant") + \
            np.pad(p[1:, :, 1, 0] - p[:-1, :, 1, 0], ((1, 0), (0, 0)), mode="constant")
    div_v = np.pad(p[:, 1:, 0, 1] - p[:, :-1, 0, 1], ((0, 0), (1, 0)), mode="constant") + \
            np.pad(p[1:, :, 1, 1] - p[:-1, :, 1, 1], ((1, 0), (0, 0)), mode="constant")
            
    # u update
    u[..., 0] += theta * div_u
    u[..., 1] += theta * div_v
    
    # 4. Pointwise Soft-Thresholding / Shrinkage update
    rho = Ix * u[..., 0] + Iy * u[..., 1] + It
    threshold = lambda_val * theta * grad_norm_sq
    
    step = np.zeros_like(u)
    mask1 = rho < -threshold
    mask2 = rho > threshold
    mask3 = ~mask1 & ~mask2
    
    step[mask1, 0] = lambda_val * theta * Ix[mask1]
    step[mask1, 1] = lambda_val * theta * Iy[mask1]
    step[mask2, 0] = -lambda_val * theta * Ix[mask2]
    step[mask2, 1] = -lambda_val * theta * Iy[mask2]
    step[mask3, 0] = - (rho[mask3] / grad_norm_sq[mask3]) * Ix[mask3]
    step[mask3, 1] = - (rho[mask3] / grad_norm_sq[mask3]) * Iy[mask3]
    
    u += step
    return u, p
```

---

## 5. Comparative Paradigm Analysis

| Optical Flow Formulation | Regularization Type | Motion Discontinuities | Outlier Robustness | GPU Real-Time Suitability |
| :--- | :--- | :--- | :--- | :--- |
| **Horn-Schunck (1981)** | Quadratic Dirichlet ($H^1$) | Blurred / Bleeds | Poor ($L^2$ error) | High (Linear System) |
| **Lucas-Kanade (1981)** | Local Window Constancy | Piecewise Constant | Moderate | Very High |
| **TV-$L^1$ (2007)** | **Total Variation ($L^1$)** | **Preserved Razor-Sharp** | **High ($L^1$ residual)** | **Very High ($>100\text{ FPS}$)** |
| **RAFT (2020)** | Neural Recurrent GRU | Learned Prior | Very High | High ($30\text{--}60\text{ FPS}$) |

---

## 6. Models & Topics in the Vault Utilizing TV-$L^1$

- **[[topics/optical-and-scene-flow-perception/00-optical-and-scene-flow-perception-moc|Optical Flow MOC]]**: Classical foundational benchmark for dense motion estimation.
- **[[topics/thermal-and-hyperspectral-vision/00-thermal-and-hyperspectral-vision-moc|Thermal Vision MOC]]**: Motion stabilization across low-contrast long-wave infrared sensors.
- **[[techniques/all-pairs-correlation-pyramids|All-Pairs Correlation Pyramids]]**: Evaluated as the modern deep-learning counterpart to variational flow.

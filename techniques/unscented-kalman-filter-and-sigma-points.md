---
title: "Unscented Kalman Filter (UKF) & Sigma Points: Derivative-Free Non-Linear Estimation"
type: "Technique"
domain: "Sensor Fusion, State Estimation & Non-Linear Filtering"
tags:
  - technique
  - ukf
  - unscented-kalman-filter
  - sigma-points
  - non-linear-filtering
  - sensor-fusion
  - state-estimation
status: evergreen
updated: 2026-09-09
aliases:
  - "UKF"
  - "Unscented Kalman Filter"
  - "Unscented Transform"
  - "Sigma Points"
  - "Derivative-Free Kalman Filter"
---

# 🔮 Unscented Kalman Filter (UKF) & Sigma Points: Derivative-Free Non-Linear Estimation

## 1. High-Level Concept & The Failure Modes of the EKF

In multi-sensor perception, robotic localization, and autonomous vehicle navigation:
Dynamic state transitions and measurement models are inherently non-linear:

$$
\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_{k-1}) + \mathbf{w}_{k-1}, \quad \mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k
$$

Common examples include radar/sonar range-bearing conversions ($r = \sqrt{x^2 + y^2}, \theta = \arctan(y/x)$), quaternion kinematics, and non-holonomic wheeled vehicle models.

### The Breakdown of the Extended Kalman Filter (EKF)
The classical EKF approximates non-linearities using a **first-order Taylor series expansion** linearized at the current estimate:
$$\mathbf{F}_k = \left.\frac{\partial f}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}_{k-1}}, \quad \mathbf{H}_k = \left.\frac{\partial h}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}_k^-}$$

This first-order approximation suffers from severe drawbacks:
1. **Severe Linearization Bias**: Under strong non-linearities or large initial uncertainty, discarding 2nd- and higher-order terms causes the posterior mean to shift and the estimated covariance to underestimate true dispersion, frequently causing **filter divergence**!
2. **Analytical Jacobian Derivation Hell**: Computing Jacobians $\frac{\partial h}{\partial \mathbf{x}}$ analytically is tedious, prone to manual mathematical errors, and **completely impossible for discontinuous, non-differentiable, or black-box physics simulation models**.

---

### The Unscented Transform (UT) Paradigm (Julier & Uhlmann, 1997)
The foundational philosophy of the Unscented Kalman Filter states:
> *"It is easier to approximate a probability distribution than it is to approximate an arbitrary non-linear function or transformation."*

Rather than approximating the non-linear function $f(\cdot)$ with a crude linear tangent plane:
The Unscented Transform deterministically samples a minimal set of **$2n + 1$ Sigma Points** ($\boldsymbol{\chi}_i$) that **exactly capture the true mean $\hat{\mathbf{x}}$ and covariance matrix $\mathbf{P}$** of an $n$-dimensional Gaussian distribution:

1. **Deterministic Sigma Point Generation**: Compute $2n + 1$ points symmetrically clustered around the mean using the matrix square root (Cholesky factorization) of covariance $\mathbf{P}$.
2. **True Non-Linear Propagation**: Pass **each individual sigma point directly through the exact, unabridged non-linear function**:
   $$\boldsymbol{\mathcal{Y}}_i = f(\boldsymbol{\chi}_i)$$
3. **Statistical Recombination**: Reconstruct the predicted posterior mean and covariance via a weighted sum of the propagated points.

```
Unscented Transform (UT) vs. Extended Kalman Filter (EKF):

EKF: Linearize Function (Taylor Series)        UKF: Transform Deterministic Sigma Points
---------------------------------------        -----------------------------------------
         Non-linear Function                            Non-linear Function
              /                                               /
             /   <-- Tangent Line Linearization              /   <-- Exact function evaluated
            /        (Errors grow with distance)            /        at 2n+1 sigma points
           /                                               /
[Prior Gaussian] ---> [Posterior Estimate]     [Prior Gaussian] ---> [Posterior Estimate]
                      (1st order accurate,                           (3rd order accurate,
                       biased mean)                                   zero Jacobians)
```

#### Theoretical Advantages of UKF over EKF
- **3rd-Order Accuracy**: For Gaussian distributions, the UKF captures posterior mean and covariance **accurate to the 3rd order** of the Taylor series (EKF achieves only 1st order).
- **Completely Derivative-Free**: Requires zero Jacobian or Hessian derivations ($\nabla f, \nabla h$).
- **Identical Computational Complexity**: Both algorithms operate with $\mathcal{O}(n^3)$ matrix operations (Cholesky factorization for UKF, matrix multiplication for EKF).

---

## 2. Mathematical Formulation

Let the state vector have dimension $n$. Define the scaling parameter $\lambda = \alpha^2 (n + \kappa) - n$:
- $\alpha \in (0, 1]$ controls the spread of sigma points (typically $10^{-3}$).
- $\kappa \ge 0$ is a secondary scaling parameter (typically $0$ or $3 - n$).
- $\beta \ge 0$ incorporates prior knowledge of the distribution ($\beta = 2$ is optimal for Gaussians).

### 2.1 Sigma Point Calculation
Let $\mathbf{L} = \text{chol}(\mathbf{P})$ be the lower-triangular Cholesky decomposition such that $\mathbf{P} = \mathbf{L} \mathbf{L}^T$.
The $2n + 1$ sigma vectors are:

$$
\begin{aligned}
\boldsymbol{\chi}_0 &= \hat{\mathbf{x}} \\
\boldsymbol{\chi}_i &= \hat{\mathbf{x}} + \left( \sqrt{n + \lambda} \, \mathbf{L} \right)_i, \quad i = 1, \dots, n \\
\boldsymbol{\chi}_{i+n} &= \hat{\mathbf{x}} - \left( \sqrt{n + \lambda} \, \mathbf{L} \right)_i, \quad i = 1, \dots, n
\end{aligned}
$$

The corresponding scalar weights for mean ($W^{(m)}$) and covariance ($W^{(c)}$) evaluate to:

$$
W_0^{(m)} = \frac{\lambda}{n + \lambda}, \quad W_0^{(c)} = \frac{\lambda}{n + \lambda} + (1 - \alpha^2 + \beta)
$$

$$
W_i^{(m)} = W_i^{(c)} = \frac{1}{2(n + \lambda)}, \quad \forall i = 1, \dots, 2n
$$

---

### 2.2 Time Update (Prediction Step)
1. Propagate each sigma point through system dynamics:
   $$\boldsymbol{\chi}_{k|k-1}^{*(i)} = f(\boldsymbol{\chi}_{k-1}^{(i)}, \mathbf{u}_{k-1})$$
2. Compute predicted state mean and covariance:
   $$\hat{\mathbf{x}}_{k}^- = \sum_{i=0}^{2n} W_i^{(m)} \boldsymbol{\chi}_{k|k-1}^{*(i)}$$
   $$\mathbf{P}_k^- = \sum_{i=0}^{2n} W_i^{(c)} \left( \boldsymbol{\chi}_{k|k-1}^{*(i)} - \hat{\mathbf{x}}_k^- \right) \left( \boldsymbol{\chi}_{k|k-1}^{*(i)} - \hat{\mathbf{x}}_k^- \right)^T + \mathbf{Q}$$

---

### 2.3 Measurement Update (Correction Step)
1. Instantiate new sigma points from $\hat{\mathbf{x}}_k^-, \mathbf{P}_k^-$ and propagate through measurement model:
   $$\boldsymbol{\mathcal{Z}}_i = h(\boldsymbol{\chi}_i^-)$$
2. Compute predicted measurement $\hat{\mathbf{z}}$, innovation covariance $\mathbf{S}_k$, and cross-covariance $\mathbf{P}_{xz}$:
   $$\hat{\mathbf{z}} = \sum_{i=0}^{2n} W_i^{(m)} \boldsymbol{\mathcal{Z}}_i$$
   $$\mathbf{S}_k = \sum_{i=0}^{2n} W_i^{(c)} \left( \boldsymbol{\mathcal{Z}}_i - \hat{\mathbf{z}} \right) \left( \boldsymbol{\mathcal{Z}}_i - \hat{\mathbf{z}} \right)^T + \mathbf{R}$$
   $$\mathbf{P}_{xz} = \sum_{i=0}^{2n} W_i^{(c)} \left( \boldsymbol{\chi}_i^- - \hat{\mathbf{x}}_k^- \right) \left( \boldsymbol{\mathcal{Z}}_i - \hat{\mathbf{z}} \right)^T$$
3. Kalman Gain and State Update:
   $$\mathbf{K}_k = \mathbf{P}_{xz} \mathbf{S}_k^{-1}$$
   $$\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + \mathbf{K}_k (\mathbf{z}_k - \hat{\mathbf{z}}), \quad \mathbf{P}_k = \mathbf{P}_k^- - \mathbf{K}_k \mathbf{S}_k \mathbf{K}_k^T$$

---

## 3. Python Reference Implementation

```python
import numpy as np
import scipy.linalg

class UnscentedKalmanFilter:
    """
    Standard Scaled Unscented Kalman Filter for non-linear state estimation.
    """
    def __init__(self, dim_x: int, dim_z: int, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.n = dim_x
        self.m = dim_z
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        self.lam = (alpha ** 2) * (self.n + kappa) - self.n
        self._compute_weights()

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x) * 0.01
        self.R = np.eye(dim_z) * 0.1

    def _compute_weights(self):
        n, lam = self.n, self.lam
        num_points = 2 * n + 1
        self.Wm = np.full(num_points, 1.0 / (2.0 * (n + lam)))
        self.Wc = np.full(num_points, 1.0 / (2.0 * (n + lam)))

        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1.0 - self.alpha ** 2 + self.beta)

    def generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generates 2n+1 sigma points using Cholesky square root."""
        n, lam = self.n, self.lam
        sigma_pts = np.zeros((2 * n + 1, n))
        sigma_pts[0] = x

        # Lower triangular Cholesky factor: P = L * L^T
        L = scipy.linalg.cholesky((n + lam) * P, lower=True)
        for i in range(n):
            sigma_pts[i + 1] = x + L[:, i]
            sigma_pts[i + 1 + n] = x - L[:, i]
        return sigma_pts

    def predict(self, f_func):
        """f_func: non-linear state transition function Callable(x) -> x_next."""
        sigmas = self.generate_sigma_points(self.x, self.P)
        sigmas_f = np.array([f_func(s) for s in sigmas])

        # Predicted mean
        self.x = np.sum(self.Wm[:, None] * sigmas_f, axis=0)

        # Predicted covariance
        y = sigmas_f - self.x
        self.P = np.sum(self.Wc[:, None, None] * (y[:, :, None] @ y[:, None, :]), axis=0) + self.Q

    def update(self, z: np.ndarray, h_func):
        """h_func: non-linear measurement function Callable(x) -> z."""
        sigmas = self.generate_sigma_points(self.x, self.P)
        sigmas_h = np.array([h_func(s) for s in sigmas])

        # Predicted measurement mean
        z_pred = np.sum(self.Wm[:, None] * sigmas_h, axis=0)

        # Innovation covariance S and cross-covariance Pxz
        y_z = sigmas_h - z_pred
        y_x = sigmas - self.x

        S = np.sum(self.Wc[:, None, None] * (y_z[:, :, None] @ y_z[:, None, :]), axis=0) + self.R
        Pxz = np.sum(self.Wc[:, None, None] * (y_x[:, :, None] @ y_z[:, None, :]), axis=0)

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)

        # State & covariance correction
        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ S @ K.T
```

---

## 4. Models in the Vault Utilizing UKF & Non-Linear Filters

- **[[techniques/extended-kalman-filter-nonlinear-estimation|Extended Kalman Filter (EKF)]]**: Linearized Taylor benchmark superseded by UKF under severe non-linearities.
- **[[techniques/error-state-kalman-filter-eskf-vio|Error-State Kalman Filter (ESKF)]]**: Lie group tangent space formulation for visual-inertial odometry.
- **[[topics/sensor-fusion/README|Sensor Fusion Playbook]]**: Radar-Camera-LiDAR tracking in autonomous driving.
- **[[topics/video-tracking/README|Video Tracking Playbook]]**: Non-linear object trajectory prediction through occlusions.

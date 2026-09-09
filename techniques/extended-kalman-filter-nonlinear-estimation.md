---
title: "Extended Kalman Filter (EKF): Non-Linear Estimation & Covariance Propagation"
type: "Technique"
domain: "Sensor Fusion, State Estimation & Navigation"
tags:
  - technique
  - kalman-filter
  - ekf
  - sensor-fusion
  - state-estimation
  - jacobian-linearization
  - tracking
status: evergreen
updated: 2026-09-09
aliases:
  - "EKF"
  - "Extended Kalman Filter"
  - "Non-Linear Estimation"
  - "Jacobian Linearization"
  - "Covariance Propagation"
---

# 📡 Extended Kalman Filter (EKF): Non-Linear Estimation & Covariance Propagation

## 1. High-Level Concept & The Non-Linear Gaussian Dilemma

The classical Linear Kalman Filter (LKF) provides the mathematically optimal, minimum-variance state estimator under the strict assumption of **linear system dynamics and linear measurement models**:

$$
\mathbf{x}_{k+1} = \mathbf{F}_k \mathbf{x}_k + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_k, \quad \mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k
$$

In real-world robotic navigation, autonomous driving, target tracking, and multi-sensor perception:
Physical kinematics and sensor observations are **inherently non-linear**:

$$
\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k) + \mathbf{w}_k, \quad \mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k
$$

Examples include:
- **Radar / Sonar Polar Measurements**: Range $r = \sqrt{x^2 + y^2}$, bearing $\theta = \text{atan2}(y, x)$, and radial Doppler velocity $\dot{r} = \frac{x\dot{x} + y\dot{y}}{\sqrt{x^2 + y^2}}$.
- **Pinhole Camera Projections**: 2D pixel coordinates $u = f_x \frac{X}{Z} + c_x, v = f_y \frac{Y}{Z} + c_y$.

---

### The Gaussian Distortion Trap
When a Gaussian random variable $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})$ is passed through a non-linear operator $h(\mathbf{x})$:
- The output distribution is **no longer Gaussian**!
- It becomes skewed, multimodal, and asymmetrical. Evaluating the exact posterior requires solving infinite-dimensional Fokker-Planck partial differential equations.

---

### The EKF First-Order Linearization Solution
The Extended Kalman Filter linearizes non-linear functions $f(\mathbf{x})$ and $h(\mathbf{x})$ locally around the current state estimate using **first-order Taylor series approximations**:

$$
f(\mathbf{x}) \approx f(\hat{\mathbf{x}}) + \mathbf{F}_k (\mathbf{x} - \hat{\mathbf{x}}), \quad h(\mathbf{x}) \approx h(\hat{\mathbf{x}}^-) + \mathbf{H}_k (\mathbf{x} - \hat{\mathbf{x}}^-)
$$

where $\mathbf{F}_k$ and $\mathbf{H}_k$ are the analytical Jacobian matrices:

$$
\mathbf{F}_k = \left. \frac{\partial f}{\partial \mathbf{x}} \right|_{\hat{\mathbf{x}}_k}, \quad \mathbf{H}_k = \left. \frac{\partial h}{\partial \mathbf{x}} \right|_{\hat{\mathbf{x}}_k^-}
$$

```
Extended Kalman Filter Recursive Cycle:

Initial State x_0, P_0
          |
          v
[ 1. Time Propagation / Predict Step ]
- Propagate State via Non-Linear Kinematics:  x_k^- = f(x_{k-1}, u_{k-1})
- Propagate Covariance via Jacobian F:        P_k^- = F * P_{k-1} * F^T + Q
          |
          v
[ 2. Measurement Update / Correct Step ]
- Compute Innovation Residual:                y_k = z_k - h(x_k^-)
- Compute Measurement Jacobian H:             H_k = dh / dx at x_k^-
- Innovation Covariance:                      S_k = H * P_k^- * H^T + R
- Optimal Kalman Gain:                        K_k = P_k^- * H^T * S_k^-1
- Posterior State Update:                     x_k = x_k^- + K_k * y_k
- Joseph-Form Covariance Update:              P_k = (I - K*H) * P_k^- * (I - K*H)^T + K*R*K^T
          |
          +----------------------------- Loop back to Step 1
```

---

## 2. Mathematical Formulation

### 2.1 The Two-Step Discrete EKF Equations

#### Step 1: Time Update (Prediction)
1. **A Priori State Estimate**:
   $$
   \hat{\mathbf{x}}_k^- = f(\hat{\mathbf{x}}_{k-1}, \mathbf{u}_{k-1})
   $$
2. **A Priori Error Covariance**:
   $$
   \mathbf{P}_k^- = \mathbf{F}_{k-1} \mathbf{P}_{k-1} \mathbf{F}_{k-1}^T + \mathbf{Q}_{k-1}
   $$

---

#### Step 2: Measurement Update (Correction)
1. **Measurement Residual (Innovation)**:
   $$
   \tilde{\mathbf{y}}_k = \mathbf{z}_k - h(\hat{\mathbf{x}}_k^-)
   $$
2. **Innovation Covariance**:
   $$
   \mathbf{S}_k = \mathbf{H}_k \mathbf{P}_k^- \mathbf{H}_k^T + \mathbf{R}_k
   $$
3. **Optimal Kalman Gain**:
   $$
   \mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}_k^T \mathbf{S}_k^{-1}
   $$
4. **A Posteriori State Update**:
   $$
   \hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + \mathbf{K}_k \tilde{\mathbf{y}}_k
   $$
5. **A Posteriori Covariance (Joseph Form)**:
   $$
   \mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_k^- (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T
   $$
   The Joseph form ensures that $\mathbf{P}_k$ remains strictly symmetric and positive semi-definite despite floating-point rounding errors.

---

### 2.2 Radar Measurement Jacobian Derivation
For a 2D radar measuring range $r$ and azimuth $\theta$ from state $\mathbf{x} = [p_x, p_y, v_x, v_y]^T$:

$$
h(\mathbf{x}) = \begin{bmatrix} \sqrt{p_x^2 + p_y^2} \\ \text{atan2}(p_y, p_x) \end{bmatrix}
$$

The analytical measurement Jacobian evaluates to:

$$
\mathbf{H} = \begin{bmatrix} \frac{p_x}{\sqrt{p_x^2 + p_y^2}} & \frac{p_y}{\sqrt{p_x^2 + p_y^2}} & 0 & 0 \\ -\frac{p_y}{p_x^2 + p_y^2} & \frac{p_x}{p_x^2 + p_y^2} & 0 & 0 \end{bmatrix}
$$

---

## 3. Python Reference Implementation

```python
import numpy as np

class ExtendedKalmanFilter2D:
    """
    Extended Kalman Filter tracking 2D kinematics [x, y, vx, vy]
    fusing non-linear polar radar measurements [range, azimuth].
    """
    def __init__(self, dt: float = 0.05, q_sigma: float = 0.2, r_range: float = 0.5, r_theta: float = 0.02):
        self.dt = dt
        # State: [px, py, vx, vy]
        self.x = np.zeros(4)
        self.P = np.eye(4) * 10.0

        # Process noise covariance Q (Constant Velocity model)
        q_pos = (dt**3) / 3.0 * (q_sigma**2)
        q_vel = dt * (q_sigma**2)
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])

        # Measurement noise covariance R
        self.R = np.diag([r_range**2, r_theta**2])

    def predict(self):
        """Linear constant-velocity state propagation."""
        F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        # x = F * x
        self.x = F @ self.x
        # P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_meas: np.ndarray):
        """
        Non-linear radar update: z = [r, theta].
        """
        px, py = self.x[0], self.x[1]
        range_pred = np.sqrt(px**2 + py**2)
        theta_pred = np.arctan2(py, px)

        if range_pred < 1e-4:
            return # Singularity at origin

        # Predicted measurement h(x)
        z_pred = np.array([range_pred, theta_pred])

        # Measurement Jacobian H = dh / dx
        H = np.array([
            [px / range_pred, py / range_pred, 0.0, 0.0],
            [-py / (range_pred**2), px / (range_pred**2), 0.0, 0.0]
        ])

        # Innovation: y = z - h(x)
        y = z_meas - z_pred
        # Normalize angle difference to [-pi, pi]
        y[1] = (y[1] + np.pi) % (2.0 * np.pi) - np.pi

        # Innovation covariance: S = H * P * H^T + R
        S = H @ self.P @ H.T + self.R

        # Kalman Gain: K = P * H^T * inv(S)
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Joseph-form covariance update
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
```

---

## 4. Models in the Vault Utilizing EKF & State Estimation

- **[[topics/slam-and-spatial-perception/README|LiDAR-Inertial Odometry (FAST-LIO)]]**: Iterated Error-State Kalman Filter on $\mathrm{SE}(3)$ Lie groups for direct LiDAR-inertial odometry.
- **[[topics/sensor-fusion/README|Sensor Fusion Playbook]]**: Cataloged as the primary non-linear sensor fusion framework.
- **[[techniques/error-state-kalman-filter-eskf-vio|Error-State Kalman Filter (ESKF)]]**: Tangent space rotation parameterization eliminating quaternion singularities.
- **[[topics/video-tracking/README|Video Tracking Playbook]]**: Classical Kalman state prediction for Bounding Box bounding trackers (SORT / ByteTrack).

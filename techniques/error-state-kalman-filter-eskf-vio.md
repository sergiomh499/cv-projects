---
title: "Error-State Kalman Filter (ESKF) & Manifold Kinematics in Visual-Inertial Odometry"
type: "Technique"
domain: "Sensor Fusion, State Estimation & Visual-Inertial SLAM"
tags:
  - technique
  - eskf
  - sensor-fusion
  - visual-inertial-odometry
  - vio
  - imu
  - lie-algebra
  - kalman-filter
status: evergreen
updated: 2026-09-09
aliases:
  - "Error-State Kalman Filter"
  - "ESKF"
  - "Manifold IMU Integration"
  - "VIO State Estimation"
---

# 🛰️ Error-State Kalman Filter (ESKF) & Manifold Kinematics in Visual-Inertial Odometry

## 1. High-Level Concept & The Quaternion Covariance Collapse

In autonomous vehicles, drones, and visual-inertial SLAM (e.g., [[topics/sensor-fusion/00-sensor-fusion-moc|FAST-LIO2]], OpenVINS), estimating a platform's 6-DoF trajectory requires fusing high-frequency ($200\text{--}1000\text{ Hz}$) inertial measurements (accelerations and angular rates from an IMU) with lower-frequency ($10\text{--}30\text{ Hz}$) external observations (camera keypoints, LiDAR scans, or GNSS fixes).

### Why Standard Extended Kalman Filters (EKF) Fail on Rotations
Traditional EKFs maintain the full state vector directly in Euclidean space. However, tracking 3D orientation directly presents mathematical pathologies:
1. **Euler Angles**: Suffer from trigonometric singularities (gimbal lock at pitch $\pm 90^\circ$).
2. **Direction Cosine Matrices ($\mathbb{R}^{3 \times 3}$)**: Optimizing 9 unconstrained elements violates the strict $\mathrm{SO}(3)$ orthogonality constraint ($\mathbf{R}^T \mathbf{R} = \mathbf{I}$).
3. **Unit Quaternions ($\mathbf{q} \in \mathbb{H}$)**: Contain 4 parameters subject to the unit-norm constraint $\|\mathbf{q}\| = 1$. The standard EKF linear covariance update $\mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}$ treats all state dimensions as unconstrained Euclidean lines. This forces the $4 \times 4$ rotation block of the covariance matrix $\mathbf{P}$ to become mathematically **singular** (rank-deficient along the quaternion radial direction), triggering numerical covariance collapse and divergence.

### The Error-State Kalman Filter (ESKF) Solution
The **Error-State Kalman Filter** (Sola, 2017) completely circumvents this by decomposing the system into three distinct entities:
1. **Nominal State $\mathbf{x}$**: Large-signal, non-linear kinematic state (position $\mathbf{p}$, velocity $\mathbf{v}$, unit quaternion $\mathbf{q}$, accelerometer bias $\mathbf{a}_b$, gyro bias $\boldsymbol{\omega}_b$). It is integrated forward in time at full IMU frequency without linear approximations.
2. **True State $\mathbf{x}_t$**: The physical ground-truth system state.
3. **Error State $\delta \mathbf{x} \in \mathbb{R}^{15}$**: The small discrepancy between the true state and nominal state. Critically, rotational error is parameterized as an unconstrained 3D tangent vector $\delta \boldsymbol{\theta} \in \mathfrak{so}(3)$!

```
Nominal State Integration vs. Error-State Reset:

IMU Measurements (1000 Hz: a_m, omega_m)
       |
       v
[ Nominal Kinematic Integration: Non-Linear Continuous ODE ] ---> Position, Velocity, Quat
                                                                         ^
Camera / LiDAR Update (30 Hz: y_obs)                                     | State Injection
       |                                                                 |
       v                                                                 |
[ Error-State Kalman Update (Linear 15x15 Covariance P) ] ---> delta_x = [dp, dv, dtheta, ...]
       |                                                                 |
       +---------------------- Reset delta_x to 0 -----------------------+
```

### Why the ESKF is Mathematically Superior
- **Near-Zero Linearization**: Because the error state $\delta \mathbf{x}$ is reset to zero after every measurement update, it constantly hovers around the origin. High-order terms in Taylor expansions genuinely vanish, making linear approximations exceptionally accurate.
- **Minimal, Non-Singular Covariance**: Rotations are represented in 3-dimensional tangent space $\mathfrak{so}(3)$, producing a strictly positive-definite, non-singular $15 \times 15$ covariance matrix $\mathbf{P}$.

---

## 2. Mathematical Formulation

### 2.1 The Kinematic State Vectors
- **Nominal State** ($\mathbf{x} \in \mathbb{R}^{16}$):
  $$\mathbf{x} = \left[ \mathbf{p}^T, \; \mathbf{v}^T, \; \mathbf{q}^T, \; \mathbf{a}_b^T, \; \boldsymbol{\omega}_b^T \right]^T$$
- **Error State** ($\delta \mathbf{x} \in \mathbb{R}^{15}$):
  $$\delta \mathbf{x} = \left[ \delta \mathbf{p}^T, \; \delta \mathbf{v}^T, \; \delta \boldsymbol{\theta}^T, \; \delta \mathbf{a}_b^T, \; \delta \boldsymbol{\omega}_b^T \right]^T$$
- True composition:
  $$\mathbf{p}_t = \mathbf{p} + \delta \mathbf{p}, \quad \mathbf{v}_t = \mathbf{v} + \delta \mathbf{v}, \quad \mathbf{q}_t = \mathbf{q} \otimes \exp\left( \frac{1}{2} \delta \boldsymbol{\theta} \right), \quad \mathbf{a}_{bt} = \mathbf{a}_b + \delta \mathbf{a}_b$$

---

### 2.2 Continuous Nominal Kinematics
Given raw IMU accelerometer $\mathbf{a}_m$ and gyroscope $\boldsymbol{\omega}_m$:

$$
\dot{\mathbf{p}} = \mathbf{v}, \quad \dot{\mathbf{v}} = \mathbf{R}(\mathbf{q}) (\mathbf{a}_m - \mathbf{a}_b) + \mathbf{g}, \quad \dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes (\boldsymbol{\omega}_m - \boldsymbol{\omega}_b), \quad \dot{\mathbf{a}}_b = \mathbf{0}, \quad \dot{\boldsymbol{\omega}}_b = \mathbf{0}
$$

where $\mathbf{R}(\mathbf{q}) \in \mathrm{SO}(3)$ is the rotation matrix of $\mathbf{q}$, and $\mathbf{g} = [0, 0, -9.81]^T$ is the gravity vector.

---

### 2.3 Continuous Error-State Linear Dynamics
Differentiating the error state yields the continuous-time linear differential equation:

$$
\dot{\delta \mathbf{x}}(t) = \mathbf{F}_t \delta \mathbf{x}(t) + \mathbf{G}_t \mathbf{w}(t)
$$

where the $15 \times 15$ system Jacobian matrix $\mathbf{F}_t$ is defined in analytical blocks:

$$
\mathbf{F}_t = \begin{bmatrix}
\mathbf{0}_{3\times 3} & \mathbf{I}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} \\
\mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & -\left[ \mathbf{R}(\mathbf{q})(\mathbf{a}_m - \mathbf{a}_b) \right]^\wedge & -\mathbf{R}(\mathbf{q}) & \mathbf{0}_{3\times 3} \\
\mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & -\left[ \boldsymbol{\omega}_m - \boldsymbol{\omega}_b \right]^\wedge & \mathbf{0}_{3\times 3} & -\mathbf{I}_{3\times 3} \\
\mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} \\
\mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3}
\end{bmatrix}
$$

where $[\cdot]^\wedge$ is the $3 \times 3$ skew-symmetric hat operator.

---

### 2.4 Discrete Covariance Propagation
For IMU sampling period $\Delta t$, the discrete state transition matrix $\mathbf{\Phi}_k$ is approximated via second-order Taylor expansion:

$$
\mathbf{\Phi}_k = \exp(\mathbf{F}_t \Delta t) \approx \mathbf{I}_{15\times 15} + \mathbf{F}_t \Delta t + \frac{1}{2} \mathbf{F}_t^2 \Delta t^2
$$

The error covariance $\mathbf{P} \in \mathbb{R}^{15 \times 15}$ is propagated:

$$
\mathbf{P}_{k|k-1} = \mathbf{\Phi}_k \mathbf{P}_{k-1|k-1} \mathbf{\Phi}_k^T + \mathbf{Q}_d
$$

where $\mathbf{Q}_d$ is the discrete IMU noise covariance (accelerometer noise $\sigma_a^2$, gyro noise $\sigma_\omega^2$, and random walk biases $\sigma_{ab}^2, \sigma_{\omega b}^2$).

---

### 2.5 Measurement Update & Manifold State Injection
When an external observation $\mathbf{y}$ arrives (e.g., 3D camera position measurement $\mathbf{y} = \mathbf{p}_t + \mathbf{v}_{\text{obs}}$):
1. **Measurement Residual**:
   $$\mathbf{r} = \mathbf{y} - h(\mathbf{x})$$
2. **Measurement Matrix** $\mathbf{H} = \frac{\partial h}{\partial \delta \mathbf{x}} \in \mathbb{R}^{3 \times 15}$:
   $$\mathbf{H} = \begin{bmatrix} \mathbf{I}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} \end{bmatrix}$$
3. **Kalman Gain**:
   $$\mathbf{K} = \mathbf{P} \mathbf{H}^T \left( \mathbf{H} \mathbf{P} \mathbf{H}^T + \mathbf{R}_{\text{obs}} \right)^{-1}$$
4. **Error State Correction**:
   $$\delta \hat{\mathbf{x}} = \mathbf{K} \mathbf{r} = \left[ \delta \hat{\mathbf{p}}^T, \; \delta \hat{\mathbf{v}}^T, \; \delta \hat{\boldsymbol{\theta}}^T, \; \delta \hat{\mathbf{a}}_b^T, \; \delta \hat{\boldsymbol{\omega}}_b^T \right]^T$$
5. **State Injection into Nominal Manifold**:
   $$\mathbf{p} \leftarrow \mathbf{p} + \delta \hat{\mathbf{p}}, \quad \mathbf{v} \leftarrow \mathbf{v} + \delta \hat{\mathbf{v}}, \quad \mathbf{a}_b \leftarrow \mathbf{a}_b + \delta \hat{\mathbf{a}}_b, \quad \boldsymbol{\omega}_b \leftarrow \boldsymbol{\omega}_b + \delta \hat{\boldsymbol{\omega}}_b$$
   $$\mathbf{q} \leftarrow \mathbf{q} \otimes \begin{bmatrix} 1 \\ \frac{1}{2} \delta \hat{\boldsymbol{\theta}} \end{bmatrix}, \quad \mathbf{q} \leftarrow \frac{\mathbf{q}}{\|\mathbf{q}\|}$$
6. **Error State Reset**:
   $$\delta \mathbf{x} \leftarrow \mathbf{0}, \quad \mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K} \mathbf{H}) \mathbf{P}$$

---

## 3. Python Reference Implementation

```python
import numpy as np

def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3D vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])

class ESKFFilter:
    """
    15-state Error-State Kalman Filter for Visual-Inertial Odometry.
    States: Position(3), Velocity(3), Orientation Tangent(3), AccelBias(3), GyroBias(3).
    """
    def __init__(self):
        # Nominal state: p(3), v(3), q(4), ab(3), wb(3)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) # [w, x, y, z]
        self.ab = np.zeros(3)
        self.wb = np.zeros(3)
        self.g = np.array([0.0, 0.0, -9.81])
        
        # Error covariance: 15x15
        self.P = np.eye(15) * 1e-3

    def predict_imu(self, am: np.ndarray, wm: np.ndarray, dt: float):
        """Propagates nominal kinematics and error covariance."""
        R = quat_to_rot(self.q)
        acc_unbiased = am - self.ab
        gyro_unbiased = wm - self.wb
        
        # 1. Propagate nominal kinematics
        self.p += self.v * dt + 0.5 * (R @ acc_unbiased + self.g) * (dt ** 2)
        self.v += (R @ acc_unbiased + self.g) * dt
        
        # Quaternion integration
        theta = np.linalg.norm(gyro_unbiased) * dt
        if theta > 1e-6:
            axis = gyro_unbiased / np.linalg.norm(gyro_unbiased)
            dq = np.array([np.cos(theta/2), *(axis * np.sin(theta/2))])
            # Quat multiply self.q (w, x, y, z)
            w1, x1, y1, z1 = self.q
            w2, x2, y2, z2 = dq
            self.q = np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])
            self.q /= np.linalg.norm(self.q)
            
        # 2. Build 15x15 Error State Transition Matrix Phi
        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -skew(R @ acc_unbiased)
        F[3:6, 9:12] = -R
        F[6:9, 6:9] = -skew(gyro_unbiased)
        F[6:9, 12:15] = -np.eye(3)
        
        Phi = np.eye(15) + F * dt + 0.5 * (F @ F) * (dt ** 2)
        Q_d = np.eye(15) * 1e-4 * dt
        
        self.P = Phi @ self.P @ Phi.T + Q_d

    def update_position(self, pos_measured: np.ndarray, R_noise: np.ndarray):
        """Measurement update using 3D position (Camera/LiDAR/GNSS)."""
        # Residual
        r = pos_measured - self.p
        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)
        
        # Kalman Gain
        S = H @ self.P @ H.T + R_noise
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Error state correction
        delta_x = K @ r
        
        # Inject into nominal state
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]
        dtheta = delta_x[6:9]
        self.ab += delta_x[9:12]
        self.wb += delta_x[12:15]
        
        # Quaternion injection via Rodrigues
        w1, x1, y1, z1 = self.q
        dq = np.array([1.0, *(0.5 * dtheta)])
        w2, x2, y2, z2 = dq
        self.q = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        self.q /= np.linalg.norm(self.q)
        
        # Update and reset covariance
        self.P = (np.eye(15) - K @ H) @ self.P
```

---

## 4. Models & Playbooks Utilizing ESKF

- **[[topics/sensor-fusion/00-sensor-fusion-moc|FAST-LIO2]]**: Direct LiDAR-inertial odometry using an error-state iterated Kalman filter on $\mathrm{SO}(3)$ manifolds.
- **[[topics/slam-and-spatial-perception/00-slam-and-spatial-perception-moc|SLAM & Spatial Perception MOC]]**: Primary state-estimation architecture for fusing visual keyframes with inertial motion.
- **[[topics/real-time-systems/README|Real-Time Systems Playbook]]**: Deterministic $<1\text{ ms}$ state propagation on automotive ECUs.

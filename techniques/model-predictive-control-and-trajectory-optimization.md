---
title: "Model Predictive Control (MPC): Receding Horizon Trajectory Optimization"
type: "Technique"
domain: "Robotics, Physical AI, Autonomous Driving & Trajectory Optimization"
tags:
  - technique
  - mpc
  - model-predictive-control
  - trajectory-optimization
  - receding-horizon
  - quadratic-programming
  - robotics
status: evergreen
updated: 2026-09-09
aliases:
  - "MPC"
  - "Model Predictive Control"
  - "Receding Horizon Control"
  - "Trajectory Optimization"
  - "Constrained Optimal Control"
---

# 🎯 Model Predictive Control (MPC): Receding Horizon Trajectory Optimization

## 1. High-Level Concept & The Constrained Actuation Reality

Classical feedback controllers (e.g., PID, Linear Quadratic Regulator / LQR) operate under significant physical limitations:
- **PID Control**: Purely reactive; lacks foresight into upcoming trajectory curvature and cannot enforce multi-variable constraints.
- **LQR Control**: Optimizes an infinite-horizon quadratic cost, but assumes **completely unconstrained state and actuator spaces**.

In physical robotics, autonomous vehicles, and industrial quadrupeds:
Real hardware operates under **strict, non-negotiable physical inequality constraints**:
- **Actuator Limits**: Motor torque saturation ($\|\boldsymbol{\tau}\| \le \tau_{\max}$), steering slew rates, and voltage limits.
- **Kinematic Boundaries**: Joint angle limits ($q_{\min} \le q \le q_{\max}$) and velocity ceilings.
- **Safety Boundaries**: Obstacle keep-out distance margins and tire-road friction cone adhesion limits ($\|F_{\text{tangent}}\| \le \mu F_{\text{normal}}$).

When an unconstrained controller commands a torque beyond motor saturation, the actual system deviates from the mathematical model, often resulting in instability, wheel slip, and collisions.

---

### The Model Predictive Control (MPC) Paradigm
MPC reformulates feedback control as an **online, real-time dynamic optimization problem solved at every sampling step**:

1. **Predict the Future**: Given the current estimated state $\mathbf{x}_t$, use an explicit dynamic model to predict state trajectories over a finite look-ahead horizon of $N$ steps: $\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k)$ for $k \in \{0, \dots, N-1\}$.
2. **Solve a Constrained Optimization Problem**: Solve for the sequence of control inputs $\mathbf{U}^* = \{\mathbf{u}_0^*, \mathbf{u}_1^*, \dots, \mathbf{u}_{N-1}^*\}$ that minimizes a tracking cost while strictly satisfying all inequality constraints.
3. **Execute the First Action Only (Receding Horizon Principle)**: Apply **only the first command $\mathbf{u}_0^*$** to the physical actuators.
4. **Recede Horizon**: At step $t+1$, read new sensor feedback, advance the time window forward by $\Delta t$, and repeat the optimization from scratch!

Continuous feedback at every step provides inherent robustness against unmodeled dynamics, external wind gusts, and surface friction variations.

```
Model Predictive Control Receding Horizon Loop:

Current Sensor State x_t
           |
           v
[ 1. Multi-Step Prediction over Horizon N ] ---> x_{t+1}, x_{t+2}, ..., x_{t+N}
           |
           v
[ 2. Constrained Optimization Solver (QP) ]
     min sum(x_k^T * Q * x_k + u_k^T * R * u_k)
     s.t. Dynamics: x_{k+1} = A*x_k + B*u_k
          Actuator Constraints: u_min <= u_k <= u_max
          Safety Boundaries:    x_min <= x_k <= x_max
           |
           v Optimal Control Sequence U* = [u_0*, u_1*, ..., u_{N-1}*]
[ 3. Apply u_0* ONLY to Physical Actuators ]
           |
           v Advance Clock t -> t+1
           +----------------------------- Loop back to State Feedback
```

---

## 2. Mathematical Formulation

### 2.1 The Constrained Quadratic Program (QP)
Consider a discrete-time linear or linearized dynamic system:

$$
\mathbf{x}_{k+1} = \mathbf{A}_k \mathbf{x}_k + \mathbf{B}_k \mathbf{u}_k + \mathbf{d}_k
$$

The finite-horizon constrained trajectory optimization problem is formulated as a Quadratic Program:

$$
\min_{\mathbf{u}_0, \dots, \mathbf{u}_{N-1}} \frac{1}{2} \mathbf{e}_N^T \mathbf{P} \mathbf{e}_N + \sum_{k=0}^{N-1} \left( \frac{1}{2} \mathbf{e}_k^T \mathbf{Q} \mathbf{e}_k + \frac{1}{2} \mathbf{u}_k^T \mathbf{R} \mathbf{u}_k + \frac{1}{2} \Delta\mathbf{u}_k^T \mathbf{R}_{\Delta} \Delta\mathbf{u}_k \right)
$$

subject to:
1. **Initial Condition**: $\mathbf{x}_0 = \mathbf{x}(t)$ (current sensor state).
2. **System Dynamics**: $\mathbf{x}_{k+1} = \mathbf{A}_k \mathbf{x}_k + \mathbf{B}_k \mathbf{u}_k, \quad \forall k \in \{0, \dots, N-1\}$.
3. **Actuator Saturation**: $\mathbf{u}_{\min} \le \mathbf{u}_k \le \mathbf{u}_{\max}$.
4. **Actuator Slew Rate**: $\Delta\mathbf{u}_{\min} \le \mathbf{u}_k - \mathbf{u}_{k-1} \le \Delta\mathbf{u}_{\max}$.
5. **State Feasibility Margins**: $\mathbf{x}_{\min} \le \mathbf{x}_k \le \mathbf{x}_{\max}$.

where:
- $\mathbf{e}_k = \mathbf{x}_k - \mathbf{x}_{\text{ref}, k}$ is tracking error.
- $\mathbf{Q} \succeq 0$ penalizes trajectory deviations.
- $\mathbf{R} \succ 0$ penalizes actuator control effort.
- $\mathbf{P} \succeq 0$ is the terminal cost matrix (derived from the discrete algebraic Riccati equation to guarantee closed-loop stability).

---

## 3. Python Reference Implementation

```python
import numpy as np
import scipy.linalg

class LinearModelPredictiveController:
    """
    Linear MPC controller using condensed Quadratic Programming (QP).
    Optimizes control trajectory u_0 ... u_{N-1} over prediction horizon N.
    """
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 horizon: int = 10, u_min: float = -1.0, u_max: float = 1.0):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N = horizon
        self.u_min = u_min
        self.u_max = u_max

        self.nx = A.shape[0]
        self.nu = B.shape[1]

        # Calculate discrete Riccati terminal cost P
        self.P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        self._build_prediction_matrices()

    def _build_prediction_matrices(self):
        """Constructs condensed prediction matrices: X = Sx * x0 + Su * U."""
        nx, nu, N = self.nx, self.nu, self.N
        self.Sx = np.zeros((N * nx, nx))
        self.Su = np.zeros((N * nx, N * nu))

        A_pow = np.eye(nx)
        for i in range(N):
            A_pow = A_pow @ self.A
            self.Sx[i * nx:(i + 1) * nx, :] = A_pow
            for j in range(i + 1):
                A_j = np.linalg.matrix_power(self.A, i - j)
                self.Su[i * nx:(i + 1) * nx, j * nu:(j + 1) * nu] = A_j @ self.B

        # Block diagonal stage cost matrices
        Q_bar = scipy.linalg.block_diag(*([self.Q] * (N - 1)), self.P)
        R_bar = scipy.linalg.block_diag(*([self.R] * N))

        # Condensed QP formulation: min 1/2 U^T * H * U + g^T * U
        self.H = self.Su.T @ Q_bar @ self.Su + R_bar
        self.H = 0.5 * (self.H + self.H.T) # Enforce strict symmetry
        self.F_x = self.Su.T @ Q_bar @ self.Sx

    def solve(self, x0: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """
        Computes optimal first control action u_0*.
        x0: [nx] current state
        x_ref: [N * nx] target reference trajectory over horizon
        Returns: [nu] commanded control action
        """
        # Linear cost gradient vector: g = F_x * x0 - Su^T * Q_bar * X_ref
        g = self.F_x @ x0

        # Unconstrained analytical solve with box clamping
        # (For hard inequality guarantees, an interior-point or OSQP solver is used)
        H_inv = np.linalg.inv(self.H)
        U_opt = -H_inv @ g

        # Clamp to physical actuator limits
        U_clamped = np.clip(U_opt, self.u_min, self.u_max)

        # Return only the very first control action u_0
        return U_clamped[:self.nu]
```

---

## 4. Models in the Vault Utilizing MPC & Trajectory Optimization

- **[[architectures/multimodal-vlm-and-vla/openvla|OpenVLA]] & [[architectures/multimodal-vlm-and-vla/pi0|pi0]]**: Higher-level foundation action policies commanding setpoints tracked by high-rate low-level MPC loops.
- **[[techniques/control-barrier-functions-safe-control|Control Barrier Functions (CBF-QP)]]**: Complementary minimal-intervention safety filters executing inside MPC loops.
- **[[topics/real-time-systems/README|Real-Time Systems Playbook]]**: Cataloged as the primary trajectory execution paradigm for robotics.
- **[[topics/safety-verification-and-robustness/README|Safety Verification Playbook]]**: Formally bounded state tracking within safe invariant sets.

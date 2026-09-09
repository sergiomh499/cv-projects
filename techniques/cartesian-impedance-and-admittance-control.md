---
title: "Cartesian Impedance & Admittance Control: Compliant Manipulation"
type: "Technique"
domain: "Robotics, Physical AI & Force-Torque Control Theory"
tags:
  - technique
  - impedance-control
  - admittance-control
  - compliant-manipulation
  - operational-space
  - physical-ai
  - robotics
status: evergreen
updated: 2026-09-09
aliases:
  - "Impedance Control"
  - "Admittance Control"
  - "Cartesian Compliance"
  - "Operational Space Control"
  - "Force Control"
---

# 🦾 Cartesian Impedance & Admittance Control: Compliant Manipulation

## 1. High-Level Concept & The Rigid Position Control Hazard

Classical industrial robot manipulators operate under **high-gain joint position control**:
The internal servo loop drives actuators to track reference joint angles $\mathbf{q}_d$ with maximum stiffness ($K_p \to \infty$).
- **The Catastrophic Contact Failure**: When interacting with the physical environment (e.g., peg-in-hole assembly, wiping a table, polishing, door opening, or human-robot physical handover), even a **$0.5\text{ mm}$ geometric perception error** causes contact forces to skyrocket beyond thousands of Newtons! This triggers mechanical fractures, emergency E-stop trips, and motor drive destruction.

---

### The Impedance Control Paradigm (Hogan, ASME 1985)
Instead of treating motion and force as isolated independent control objectives, **Impedance Control establishes a dynamic relationship between motion deviations and contact forces**:

The robot's end-effector is commanded to behave as a virtual **programmable mass-spring-damper system**:

$$
\mathbf{M}_d (\ddot{\mathbf{x}} - \ddot{\mathbf{x}}_d) + \mathbf{D}_d (\dot{\mathbf{x}} - \dot{\mathbf{x}}_d) + \mathbf{K}_d (\mathbf{x} - \mathbf{x}_d) = \mathbf{F}_{\text{ext}}
$$

- When the end-effector encounters an unexpected physical surface, it yields elastically with programmable stiffness $\mathbf{K}_d$.
- In free space ($\mathbf{F}_{\text{ext}} = \mathbf{0}$), it tracks trajectory $\mathbf{x}_d$ smoothly.
- Upon physical contact, the steady-state deflection directly governs contact force: $\mathbf{F}_{\text{ext}} = \mathbf{K}_d (\mathbf{x} - \mathbf{x}_d)$.

---

### Impedance Control vs. Admittance Control

```
Dual Paradigms of Compliant Manipulation:

1. Cartesian Impedance Control (Torque-Controlled Robots, e.g., Franka Panda, KUKA iiwa):
   [ Position Deflection e = x - x_d ] ---> [ Virtual Spring-Damper ] ---> [ Motor Torques tau = J^T * F ]
   * Causality: Motion In ---> Force Out (Inherently passive & safe!)

2. Cartesian Admittance Control (Position-Controlled Robots with 6-Axis Wrist F/T Sensor, e.g., UR5e):
   [ Force Sensor F_ext ] ---> [ Integrate M*ddot(x) + D*dot(x) + K*x = F ] ---> [ Compliant Position Command x_c ]
   * Causality: Force In ---> Motion Out (Requires high-rate inner position loop!)
```

---

## 2. Mathematical Formulation

### 2.1 Operational Space Dynamics (Khatib, 1987)
The joint-space equation of motion for an $n$-degree-of-freedom manipulator is:

$$
\mathbf{M}(\mathbf{q}) \ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}}) \dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau} - \mathbf{J}(\mathbf{q})^T \mathbf{F}_{\text{ext}}
$$

where $\mathbf{M}(\mathbf{q})$ is the joint inertia matrix, $\mathbf{C}$ represents Coriolis/centrifugal forces, and $\mathbf{g}$ is gravity.

By mapping dynamics through the task-space Jacobian $\mathbf{J}(\mathbf{q}) = \frac{\partial \mathbf{x}}{\partial \mathbf{q}}$:

#### Operational Space Inertia Matrix:
$$
\mathbf{\Lambda}(\mathbf{q}) = \left( \mathbf{J}(\mathbf{q}) \mathbf{M}(\mathbf{q})^{-1} \mathbf{J}(\mathbf{q})^T \right)^{-1} \in \mathbb{R}^{6 \times 6}
$$

The end-effector dynamics in Cartesian space evaluate to:

$$
\mathbf{\Lambda}(\mathbf{q}) \ddot{\mathbf{x}} + \boldsymbol{\mu}(\mathbf{q}, \dot{\mathbf{q}}) + \mathbf{p}(\mathbf{q}) = \mathbf{F} - \mathbf{F}_{\text{ext}}
$$

---

### 2.2 Impedance Control Law with Nullspace Posture Regulation
To decouple the dynamics and enforce target Cartesian stiffness $\mathbf{K}_d$ and damping $\mathbf{D}_d$:

The commanded wrench is:

$$
\mathbf{F}_c = \mathbf{K}_d (\mathbf{x}_d - \mathbf{x}) + \mathbf{D}_d (\dot{\mathbf{x}}_d - \dot{\mathbf{x}})
$$

The joint torque command is synthesized as:

$$
\boldsymbol{\tau}_c = \mathbf{J}^T \mathbf{F}_c + \mathbf{g}(\mathbf{q}) + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{N}^T \boldsymbol{\tau}_{\text{null}}
$$

where $\mathbf{N} = \mathbf{I} - \mathbf{J}^\dagger \mathbf{J}$ is the **nullspace projection matrix**.
- Any joint torque $\boldsymbol{\tau}_{\text{null}} = \mathbf{K}_{q} (\mathbf{q}_{\text{nominal}} - \mathbf{q}) - \mathbf{D}_q \dot{\mathbf{q}}$ projected through $\mathbf{N}^T$ reconfigures redundant arm postures without exerting any wrench at the end-effector!

---

## 3. Python Reference Implementation

```python
import numpy as np

class CartesianImpedanceController:
    """
    Cartesian Impedance Controller with dynamic gravity compensation
    and redundant nullspace posture optimization.
    """
    def __init__(self, K_pos: np.ndarray, D_pos: np.ndarray, K_null: float = 10.0):
        # Diagonal stiffness and damping matrices: 6x6 (3 translation, 3 rotation)
        self.K_d = np.diag(K_pos) # e.g. [500, 500, 500, 50, 50, 50]
        self.D_d = np.diag(D_pos) # e.g. [2 * sqrt(K)]
        self.K_null = K_null

    def compute_torque(self, q: np.ndarray, dq: np.ndarray,
                       x: np.ndarray, dx: np.ndarray,
                       x_d: np.ndarray, dx_d: np.ndarray,
                       J: np.ndarray, M: np.ndarray,
                       g: np.ndarray, q_nominal: np.ndarray) -> np.ndarray:
        """
        q, dq: [n] joint positions and velocities
        x, dx: [6] current end-effector pose (position + rotation vector) and twist
        x_d, dx_d: [6] desired pose and twist
        J: [6, n] manipulator Jacobian
        M: [n, n] joint inertia matrix
        g: [n] gravity vector
        q_nominal: [n] preferred resting joint configuration
        Returns: [n] commanded joint torques tau_c
        """
        # 1. Cartesian Error
        e_x = x_d - x
        e_dx = dx_d - dx

        # 2. Desired Virtual Impedance Wrench: F_c = K_d * e + D_d * de
        F_c = self.K_d @ e_x + self.D_d @ e_dx

        # 3. Dynamically Consistent Generalized Pseudoinverse: J_bar = M^-1 * J^T * Lambda
        M_inv = np.linalg.inv(M)
        Lambda = np.linalg.inv(J @ M_inv @ J.T)
        J_bar = M_inv @ J.T @ Lambda

        # 4. Primary Task Torques
        tau_task = J.T @ F_c

        # 5. Nullspace Projector: N = I - J^T * J_bar^T
        n_dof = q.shape[0]
        N_transpose = np.eye(n_dof) - J.T @ J_bar.T

        # 6. Secondary Nullspace Objective (Joint limit centering / posture comfort)
        tau_null = self.K_null * (q_nominal - q) - 2.0 * np.sqrt(self.K_null) * dq
        tau_posture = N_transpose @ tau_null

        # Total commanded torque with gravity compensation
        tau_command = tau_task + tau_posture + g
        return tau_command
```

---

## 4. Models in the Vault Utilizing Impedance Control

- **[[architectures/multimodal-vlm-and-vla/openvla|OpenVLA]] & [[architectures/multimodal-vlm-and-vla/pi0|pi0]]**: Foundation robot models commanding compliant Cartesian delta setpoints rather than hard joint angles.
- **[[architectures/multimodal-vlm-and-vla/rt-2|RT-2]]**: Vision-Language-Action policies interacting with novel articulated objects via admittance-filtered low-level primitives.
- **[[architectures/multimodal-vlm-and-vla/anygrasp|AnyGrasp]]**: Compliant grasping execution absorbing impact collisions during high-speed pick-and-place.
- **[[techniques/control-barrier-functions-safe-control|Control Barrier Functions (CBF-QP)]]**: Bounded impedance tracking with formal collision-avoidance guarantees.

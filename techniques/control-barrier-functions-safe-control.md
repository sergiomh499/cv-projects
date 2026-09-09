---
title: "Control Barrier Functions (CBF) & Quadratic Programming: Provable Safety Filters for Robotics"
type: "Technique"
domain: "Robotics Safety, Control Theory & Physical AI"
tags:
  - technique
  - control-barrier-functions
  - cbf
  - quadratic-programming
  - physical-ai
  - robotics-safety
  - iso-21448
  - sotif
status: evergreen
updated: 2026-09-09
aliases:
  - "Control Barrier Functions"
  - "CBF-QP Safety Filter"
  - "Provable Robot Safety"
  - "Forward Invariance"
---

# 🛡️ Control Barrier Functions (CBF) & Quadratic Programming: Provable Safety Filters

## 1. High-Level Concept & The Neural Network Safety Void

Deep reinforcement learning and Vision-Language-Action (VLA) foundation models (e.g., [[architectures/multimodal-vlm-and-vla/openvla|OpenVLA]], [[architectures/multimodal-vlm-and-vla/pi0|π₀]], [[architectures/multimodal-vlm-and-vla/act|ACT]]) demonstrate remarkable generalization in open-world robot manipulation and autonomous navigation.

### The Fatal Flaw of Purely Neural Control
Despite their performance, **deep neural network policies provide ZERO formal mathematical safety guarantees**:
- Under out-of-distribution visual inputs, specular reflections, or sudden sensor occlusion, a neural network can output an erratic motor torque command that collides a manipulator into a workspace barrier or drives an autonomous mobile robot into a human worker.
- Penalizing collisions in the training loss (e.g., via reward shaping) cannot guarantee constraint satisfaction at runtime; it only produces probabilistic minimization.

### The Control Barrier Function (CBF) Solution
**Control Barrier Functions** (Ames et al., IEEE TAC 2016) solve this by establishing **set-theoretic forward invariance**:
1. **Safe Set Definition**: Defines an admissible safe operating region in the robot's state space $\mathcal{C} = \{\mathbf{x} \in \mathbb{R}^n \mid h(\mathbf{x}) \ge 0\}$ (e.g., maintaining minimum metric distance from obstacles).
2. **Forward Invariance Condition**: Guarantees that if the system initiates within the safe set $\mathcal{C}$, state trajectories **can never leave $\mathcal{C}$ for all future time**: $\dot{h}(\mathbf{x}) + \alpha(h(\mathbf{x})) \ge 0$.
3. **The Real-Time CBF-QP Safety Filter**: Operates as a lightweight mathematical gatekeeper between the raw neural network output $\mathbf{u}_{\text{nom}}$ and the physical motor actuators. It solves a convex **Quadratic Program (QP)**:
   - When the neural command is safe, the QP outputs it untouched ($\mathbf{u}^* = \mathbf{u}_{\text{nom}}$).
   - If the neural policy attempts an unsafe action, the QP minimally modifies the control vector to strictly enforce the safe barrier boundary.
4. Solves in **$<50\ \mu\text{s}$ deterministically**, satisfying ASIL-D ISO 26262 and ISO 21448 (SOTIF) safety-critical mandates.

```
Vision-Language-Action Policy with CBF-QP Safety Filter:

Camera Image / VLM Goal
       |
       v
[ Neural Policy (OpenVLA / ACT) ] ---> Nominal Control u_nom (Uncertified!)
                                                |
                                                v
[ Real-Time CBF-QP Safety Filter: min 0.5 ||u - u_nom||^2  s.t. L_f h + L_g h * u + gamma * h >= 0 ]
                                                |
                                                v
                               Safe Certified Control u* (<50 microseconds!)
                                                |
                                                v
                                   [ Physical Robot Actuators ]
```

---

## 2. Mathematical Formulation

### 2.1 Non-Linear Control-Affine Dynamical Systems
Consider a robotic system governed by non-linear control-affine dynamics:

$$
\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}) + \mathbf{g}(\mathbf{x}) \mathbf{u}
$$

where $\mathbf{x} \in \mathcal{D} \subset \mathbb{R}^n$ is the state vector (positions, velocities), $\mathbf{u} \in \mathcal{U} \subset \mathbb{R}^m$ is the control input (joint torques, wheel accelerations), and $\mathbf{f}, \mathbf{g}$ are locally Lipschitz continuous vector fields.

---

### 2.2 The Zero-Superlevel Safe Set $\mathcal{C}$
Let safety be encoded by a continuously differentiable scalar function $h(\mathbf{x}): \mathcal{D} \to \mathbb{R}$. The safe set $\mathcal{C}$, its boundary $\partial \mathcal{C}$, and unsafe exterior are:

$$
\mathcal{C} = \left\{ \mathbf{x} \in \mathcal{D} \;\middle|\; h(\mathbf{x}) \ge 0 \right\}
$$

$$
\partial \mathcal{C} = \left\{ \mathbf{x} \in \mathcal{D} \;\middle|\; h(\mathbf{x}) = 0 \right\}
$$

$$
\text{Unsafe} = \left\{ \mathbf{x} \in \mathcal{D} \;\middle|\; h(\mathbf{x}) < 0 \right\}
$$

**Physical Example**: For a mobile robot at position $\mathbf{p} \in \mathbb{R}^2$ avoiding an obstacle at $\mathbf{p}_{\text{obs}}$ with safe radius $R_{\text{safe}}$:
$$
h(\mathbf{x}) = \|\mathbf{p} - \mathbf{p}_{\text{obs}}\|^2 - R_{\text{safe}}^2
$$

---

### 2.3 The Forward Invariance Condition (Nagumo's Theorem)
A set $\mathcal{C}$ is **forward invariant** if for every initial state $\mathbf{x}(0) \in \mathcal{C}$, the trajectory satisfies $\mathbf{x}(t) \in \mathcal{C}$ for all $t \ge 0$.

According to Nagumo's theorem, as the state approaches the boundary $\partial \mathcal{C}$ ($h(\mathbf{x}) \to 0$), the time derivative $\dot{h}(\mathbf{x})$ must not point outward into the unsafe set.

$h(\mathbf{x})$ is a **Control Barrier Function (CBF)** if there exists an extended class-$\mathcal{K}_\infty$ function $\alpha(r) = \gamma \cdot r$ (with $\gamma > 0$) such that for all $\mathbf{x} \in \mathcal{C}$:

$$
\sup_{\mathbf{u} \in \mathcal{U}} \left[ L_{\mathbf{f}} h(\mathbf{x}) + L_{\mathbf{g}} h(\mathbf{x}) \mathbf{u} + \gamma \cdot h(\mathbf{x}) \right] \ge 0
$$

where the **Lie derivatives** represent directional derivatives along system dynamics:

$$
L_{\mathbf{f}} h(\mathbf{x}) = \nabla h(\mathbf{x}) \cdot \mathbf{f}(\mathbf{x}) = \sum_{i=1}^n \frac{\partial h}{\partial x_i} f_i(\mathbf{x})
$$

$$
L_{\mathbf{g}} h(\mathbf{x}) = \nabla h(\mathbf{x}) \cdot \mathbf{g}(\mathbf{x}) = \left[ \nabla h(\mathbf{x}) \cdot \mathbf{g}_1(\mathbf{x}), \; \dots, \; \nabla h(\mathbf{x}) \cdot \mathbf{g}_m(\mathbf{x}) \right] \in \mathbb{R}^{1 \times m}
$$

---

### 2.4 The CBF Quadratic Program (CBF-QP)
Given an uncertified nominal control command $\mathbf{u}_{\text{nom}} \in \mathbb{R}^m$ produced by an end-to-end vision model or neural policy, the safety-filtered command $\mathbf{u}^*$ is obtained by solving:

$$
\mathbf{u}^* = \arg\min_{\mathbf{u} \in \mathbb{R}^m} \frac{1}{2} \left\| \mathbf{u} - \mathbf{u}_{\text{nom}} \right\|^2
$$

$$\text{subject to: } \quad - L_{\mathbf{g}} h(\mathbf{x}) \cdot \mathbf{u} \le L_{\mathbf{f}} h(\mathbf{x}) + \gamma \cdot h(\mathbf{x})$$

$$\mathbf{u}_{\min} \le \mathbf{u} \le \mathbf{u}_{\max}$$

Because the objective is strictly convex quadratic and the barrier condition is strictly **linear in the control input $\mathbf{u}$**, this is a canonical Quadratic Program (QP) with closed-form dual solutions or solvable via active-set methods in $<50\ \mu\text{s}$.

---

## 3. Python Reference Implementation

```python
import numpy as np

class CBFSafetyFilter:
    """
    Control Barrier Function Quadratic Program (CBF-QP) Filter.
    Minimally perturbs nominal neural control commands to guarantee
    obstacle avoidance and forward set invariance.
    """
    def __init__(self, gamma: float = 2.0, r_safe: float = 0.5):
        self.gamma = gamma
        self.r_safe = r_safe

    def filter_control(self, p: np.ndarray, v: np.ndarray, p_obs: np.ndarray, u_nom: np.ndarray) -> np.ndarray:
        """
        Double-integrator robot dynamics: p_dot = v, v_dot = u
        p: [2] current position
        v: [2] current velocity
        p_obs: [2] obstacle position
        u_nom: [2] nominal acceleration command from neural network
        Returns: [2] certified safe acceleration command u*
        """
        # Relative displacement
        rel_p = p - p_obs
        dist_sq = np.sum(rel_p ** 2)
        
        # 1. Barrier function h(x) = ||p - p_obs||^2 - R_safe^2
        h = dist_sq - (self.r_safe ** 2)
        
        # 2. First derivative h_dot = 2 * rel_p^T * v
        h_dot = 2.0 * np.dot(rel_p, v)
        
        # 3. For relative degree 2 (acceleration control), construct higher-order barrier:
        # B(x) = h_dot + alpha1 * h
        alpha1 = 2.0
        B = h_dot + alpha1 * h
        
        # B_dot = 2 * ||v||^2 + 2 * rel_p^T * u + alpha1 * h_dot
        # Condition: B_dot + gamma * B >= 0
        # 2 * rel_p^T * u + (2 * ||v||^2 + alpha1 * h_dot + gamma * B) >= 0
        # a_cbf * u <= b_cbf
        a_cbf = - 2.0 * rel_p  # [2]
        b_cbf = 2.0 * np.sum(v ** 2) + alpha1 * h_dot + self.gamma * B
        
        # 4. Check if u_nom satisfies constraint
        if np.dot(a_cbf, u_nom) <= b_cbf:
            return u_nom  # Completely safe, output nominal command directly!
            
        # 5. Closed-form 1D active constraint projection:
        # min 0.5 ||u - u_nom||^2  s.t. a^T u = b
        # Lagrange multiplier: lambda = (a^T u_nom - b) / ||a||^2
        norm_a_sq = np.sum(a_cbf ** 2) + 1e-6
        lam = (np.dot(a_cbf, u_nom) - b_cbf) / norm_a_sq
        u_safe = u_nom - lam * a_cbf
        
        return u_safe
```

---

## 4. Safety Standards & Models Utilizing CBFs

- **[[topics/safety-verification-and-robustness/00-safety-verification-and-robustness-moc|Safety Verification MOC]]**: Primary architectural filter for ISO 21448 (SOTIF) compliance.
- **[[topics/vla-and-physical-ai-robotics/00-vla-and-physical-ai-robotics-moc|VLA & Physical AI MOC]]**: Deployed as the hardware-in-the-loop safety barrier on high-torque robot arms.
- **[[topics/visual-guidance-and-robotics/00-visual-guidance-and-robotics-moc|Visual Guidance MOC]]**: Real-time Cartesian workspace constraint enforcer.

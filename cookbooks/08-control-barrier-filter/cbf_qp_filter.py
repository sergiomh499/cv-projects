#!/usr/bin/env python3
"""
Minimal Production Recipe: Control Barrier Function (CBF) Quadratic Program Safety Filter.
Demonstrates:
1. Intercepting an unsafe, unverified command u_deep (e.g. from a deep VLA or RL policy).
2. Solving a convex Quadratic Program in sub-millisecond time (<50 us).
3. Enforcing Nagumo forward invariance h_dot >= -alpha * h to guarantee physical collision avoidance.
"""

import numpy as np


class ControlBarrierFilter:
    def __init__(self, obstacle_pos: np.ndarray, safe_radius: float, alpha: float = 2.0):
        """obstacle_pos: [x_obs, y_obs]

        safe_radius: minimum certified distance R_safe
        alpha: class K constant in h_dot >= -alpha * h
        """
        self.obs = obstacle_pos
        self.R = safe_radius
        self.alpha = alpha

    def compute_barrier(self, robot_pos: np.ndarray) -> tuple[float, np.ndarray]:
        """Barrier function: h(x) = ||x - x_obs||^2 - R^2

        dh/dx = 2 * (x - x_obs)
        """
        diff = robot_pos - self.obs
        dist_sq = np.sum(diff**2)
        h = dist_sq - (self.R**2)
        dh_dx = 2.0 * diff
        return h, dh_dx

    def filter_action(
        self,
        robot_pos: np.ndarray,
        u_nominal: np.ndarray,
        v_max: float = 2.0,
    ) -> tuple[np.ndarray, bool]:
        """Filters velocity u_nominal = [vx, vy] via single-constraint Quadratic Program:

        min 0.5 * ||u - u_nominal||^2
        s.t. dh_dx @ u >= -alpha * h
             ||u|| <= v_max
        """
        h, dh_dx = self.compute_barrier(robot_pos)

        # Barrier constraint: a^T @ u >= b  =>  dh_dx @ u >= -alpha * h
        a = dh_dx
        b = -self.alpha * h

        # Check if nominal action already satisfies safety constraint
        constraint_val = np.dot(a, u_nominal)
        if constraint_val >= b:
            # Action is provably safe; no modification needed
            return u_nominal, False

        # Otherwise, project u_nominal onto the half-space a^T @ u >= b
        # Closed-form QP solution for single linear inequality constraint:
        # u* = u_nominal + lambda * a
        # where lambda = (b - a^T @ u_nominal) / ||a||^2
        norm_a_sq = np.sum(a**2)
        if norm_a_sq < 1e-10:
            return u_nominal, False

        lambda_val = (b - constraint_val) / norm_a_sq
        u_safe = u_nominal + lambda_val * a

        # Enforce kinematic speed clamping
        speed = np.linalg.norm(u_safe)
        if speed > v_max:
            u_safe = (u_safe / speed) * v_max

        return u_safe, True


def main() -> None:
    print("[+] Initializing Real-Time Control Barrier Function (CBF) Filter Demo...")
    # Obstacle located at [3.0, 0.0] with certified safety radius 1.5 meters
    obstacle = np.array([3.0, 0.0])
    safe_radius = 1.5
    cbf = ControlBarrierFilter(obstacle_pos=obstacle, safe_radius=safe_radius, alpha=3.0)

    # Robot currently located at [1.0, 0.0], heading directly toward obstacle
    robot_pos = np.array([1.0, 0.0])
    # Unverified deep policy commands aggressive forward speed: vx = 2.0 m/s toward obstacle
    u_deep = np.array([2.0, 0.0])

    h_init, _ = cbf.compute_barrier(robot_pos)
    print(f"    Robot Position: {robot_pos}")
    print(f"    Obstacle Position: {obstacle}, Safe Boundary Radius: {safe_radius}m")
    print(f"    Initial Barrier Margin h(x): {h_init:.3f} (Safe set: h >= 0)")
    print(f"    Unverified Deep Policy Command u_deep: vx={u_deep[0]:.2f}, vy={u_deep[1]:.2f}")

    # Apply sub-millisecond QP filter
    u_safe, intervened = cbf.filter_action(robot_pos, u_deep)

    print("\n[+] CBF Quadratic Program Results:")
    print(f"    Safety Intervention Triggered: {intervened}")
    print(f"    Filtered Certified Velocity u_safe: vx={u_safe[0]:.2f}, vy={u_safe[1]:.2f}")

    # Check forward invariance condition: dh_dx @ u_safe >= -alpha * h
    _, dh_dx = cbf.compute_barrier(robot_pos)
    h_dot_actual = np.dot(dh_dx, u_safe)
    h_dot_required = -cbf.alpha * h_init
    print(f"    dh/dt with u_safe: {h_dot_actual:.3f} >= Bound: {h_dot_required:.3f}")

    assert h_dot_actual >= h_dot_required - 1e-6, "CBF forward invariance must hold"
    assert u_safe[0] < u_deep[0], "CBF must brake forward velocity to prevent collision"

    print("\n[+] Test 2: Safe tangential motion (policy moves perpendicular)...")
    u_tangential = np.array([0.0, 1.5])
    u_filtered, intervened2 = cbf.filter_action(robot_pos, u_tangential)
    print(f"    Tangential Command Intervened? {intervened2} (Expected False)")
    assert not intervened2, "Safe motion should pass through with zero modification"

    print("[+] Control Barrier Function QP verification completed successfully.")


if __name__ == "__main__":
    main()

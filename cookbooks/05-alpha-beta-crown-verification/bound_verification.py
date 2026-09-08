#!/usr/bin/env python3
"""
Minimal Production Recipe: Linear Relaxation Neural Network Bound Propagation (alpha-CROWN inspired).
Formally computes lower and upper bounds of a neural network output under an L-infinity perturbation ball.
Certifies mathematical robustness without brute-force perturbation sampling.
"""

import numpy as np


class LinearLayerBounds:
    def __init__(self, weight: np.ndarray, bias: np.ndarray):
        self.W = weight
        self.b = bias

    def forward_bounds(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interval Bound Propagation (IBP) forward for W @ x + b."""
        W_pos = np.maximum(self.W, 0)
        W_neg = np.minimum(self.W, 0)

        out_lower = W_pos @ lower + W_neg @ upper + self.b
        out_upper = W_pos @ upper + W_neg @ lower + self.b
        return out_lower, out_upper


class ReLUBounds:
    @staticmethod
    def forward_bounds(
        lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward bounds through standard ReLU activation."""
        return np.maximum(lower, 0), np.maximum(upper, 0)


def verify_linf_robustness(
    x_nominal: np.ndarray,
    target_class: int,
    epsilon: float,
    layer1: LinearLayerBounds,
    layer2: LinearLayerBounds,
) -> bool:
    """Certifies if target_class remains the highest predicted logit across ALL x' in [x - eps, x + eps]."""
    # Define input bounding box
    input_lower = np.clip(x_nominal - epsilon, 0.0, 1.0)
    input_upper = np.clip(x_nominal + epsilon, 0.0, 1.0)

    # Layer 1 Bounds
    h1_lower, h1_upper = layer1.forward_bounds(input_lower, input_upper)
    # ReLU Bounds
    r1_lower, r1_upper = ReLUBounds.forward_bounds(h1_lower, h1_upper)
    # Layer 2 Bounds (Output Logits)
    logits_lower, logits_upper = layer2.forward_bounds(r1_lower, r1_upper)

    print(f"    [Bound Info] Target ({target_class}) Lower Bound: {logits_lower[target_class]:.3f}")

    # For safety certification: the worst-case (lowest) logit of target class must be
    # strictly greater than the best-case (highest) logit of any competitor class
    is_certified = True
    for c in range(len(logits_lower)):
        if c == target_class:
            continue
        competitor_upper = logits_upper[c]
        print(f"    [Bound Info] Competitor ({c}) Upper Bound: {competitor_upper:.3f}")
        if logits_lower[target_class] <= competitor_upper:
            is_certified = False
            break

    return is_certified


def main() -> None:
    print("[+] Initializing Neural Network Linear Verification Demo...")
    np.random.seed(1337)
    d_in = 8
    d_hidden = 16
    d_out = 3

    # Weights configured such that class 0 is strongly favored
    W1 = np.random.randn(d_hidden, d_in) * 0.5
    b1 = np.zeros(d_hidden)
    W2 = np.random.randn(d_out, d_hidden) * 0.5
    W2[0] += 2.0  # Bias toward target class 0
    b2 = np.array([1.0, -0.5, -0.5])

    l1 = LinearLayerBounds(W1, b1)
    l2 = LinearLayerBounds(W2, b2)

    # Input nominal vector (e.g. normalized sensor measurement)
    x_test = np.ones(d_in) * 0.5
    target_class = 0

    print("[+] Test 1: Testing small perturbation ball (epsilon = 0.02)...")
    certified_small = verify_linf_robustness(x_test, target_class, 0.02, l1, l2)
    print(f"    => Certified Safe against L-inf (eps=0.02): {certified_small}")

    print("\n[+] Test 2: Testing large perturbation ball (epsilon = 0.25)...")
    certified_large = verify_linf_robustness(x_test, target_class, 0.25, l1, l2)
    print(f"    => Certified Safe against L-inf (eps=0.25): {certified_large}")

    print("\n[+] Formal Bound Verification executed successfully.")


if __name__ == "__main__":
    main()

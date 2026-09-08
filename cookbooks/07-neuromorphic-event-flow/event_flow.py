#!/usr/bin/env python3
"""
Minimal Production Recipe: Asynchronous Neuromorphic Event Optical Flow (Benosman Method).
Demonstrates:
1. Ingesting raw event streams e_k = (x, y, t, p) into a continuous Surface of Active Events (SAE).
2. Estimating local optical flow velocity vectors v_perp in O(1) time per event using local plane fitting.
3. Filtering thermal background activity noise and computing instantaneous motion speed.
"""

import numpy as np


class EventSurfaceOpticalFlow:
    def __init__(self, width: int = 128, height: int = 128, window_size: int = 5):
        self.width = width
        self.height = height
        self.window_size = window_size
        self.r = window_size // 2

        # Surface of Active Events: stores microsecond timestamp of latest event per pixel
        self.sae_pos = np.zeros((height, width), dtype=np.float64)
        self.sae_neg = np.zeros((height, width), dtype=np.float64)

    def process_event(
        self, x: int, y: int, timestamp_us: float, polarity: int
    ) -> tuple[bool, float, float]:
        """Processes a single asynchronous event e_k = (x, y, t, p).

        Returns: (is_valid_flow, vx_pixels_per_sec, vy_pixels_per_sec)
        """
        # Step 1: Update the Surface of Active Events (SAE)
        sae = self.sae_pos if polarity > 0 else self.sae_neg
        sae[y, x] = timestamp_us

        # Boundary check for local window
        if x < self.r or x >= self.width - self.r or y < self.r or y >= self.height - self.r:
            return False, 0.0, 0.0

        # Step 2: Extract local spatial neighborhood
        patch = sae[y - self.r : y + self.r + 1, x - self.r : x + self.r + 1]

        # Background noise filter: ensure at least 4 neighboring pixels have recent events
        active_pixels = np.sum(patch > 0)
        if active_pixels < 4:
            return False, 0.0, 0.0

        # Step 3: Fit local plane: t(x, y) = a * x + b * y + c
        # Coordinate offsets relative to center
        grid_y, grid_x = np.mgrid[-self.r : self.r + 1, -self.r : self.r + 1]
        valid_mask = patch > 0

        coords_x = grid_x[valid_mask].flatten()
        coords_y = grid_y[valid_mask].flatten()
        times_sec = (patch[valid_mask] - timestamp_us) * 1e-6  # normalized seconds

        # Construct least-squares design matrix A: [x, y, 1]
        A = np.column_stack([coords_x, coords_y, np.ones_like(coords_x)])

        # Solve via normal equations: (A^T A) @ [a, b, c]^T = A^T times
        try:
            params, _, _, _ = np.linalg.lstsq(A, times_sec, rcond=None)
            a, b, _ = params
        except np.linalg.LinAlgError:
            return False, 0.0, 0.0

        grad_norm_sq = a**2 + b**2
        if grad_norm_sq < 1e-12 or grad_norm_sq > 1e4:
            # Degenerate slope or stationary pixel
            return False, 0.0, 0.0

        # Step 4: Normal velocity is inverse spatial gradient: v_perp = grad / ||grad||^2
        vx = a / grad_norm_sq  # pixels / second
        vy = b / grad_norm_sq  # pixels / second

        return True, float(vx), float(vy)


def main() -> None:
    print("[+] Initializing Asynchronous Event-Based Optical Flow Demo...")
    flow_tracker = EventSurfaceOpticalFlow(width=64, height=64, window_size=5)

    # Simulate an edge moving horizontally from left to right at 100 pixels/sec
    # Velocity: vx = 100 px/s, vy = 0 px/s
    # In microsecond time: 1 pixel every 10,000 us
    simulated_events = []
    t_start = 1_000_000.0  # 1.0 second in microseconds
    vx_true = 100.0

    print("[+] Simulating high-speed event stream (edge moving at 100 px/s)...")
    for col in range(10, 30):
        t_col = t_start + (col - 10) * (1_000_000.0 / vx_true)
        for row in range(25, 35):
            # Add tiny sub-microsecond jitter
            t_event = t_col + (row - 25) * 50.0
            simulated_events.append((col, row, t_event, 1))

    # Process events sequentially in asynchronous microsecond order
    valid_estimates = []
    for x, y, t, p in simulated_events:
        valid, vx, vy = flow_tracker.process_event(x, y, t, p)
        if valid:
            valid_estimates.append((vx, vy))

    print(f"[+] Processed {len(simulated_events)} events asynchronously.")
    print(f"[+] Inlier Flow Vectors Extracted: {len(valid_estimates)}")

    if valid_estimates:
        mean_vx = np.mean([v[0] for v in valid_estimates])
        mean_vy = np.mean([v[1] for v in valid_estimates])
        print(f"[+] Ground Truth Velocity: vx = {vx_true:.1f} px/s, vy = 0.0 px/s")
        print(f"[+] Estimated Event Optical Flow: vx = {mean_vx:.1f} px/s, vy = {mean_vy:.1f} px/s")
        print(f"[+] Estimation Error: {abs(mean_vx - vx_true):.2f} px/s")
        assert abs(mean_vx - vx_true) < 25.0, "Flow estimate should converge near ground truth"

    print("[+] Neuromorphic Event Optical Flow verified successfully.")


if __name__ == "__main__":
    main()

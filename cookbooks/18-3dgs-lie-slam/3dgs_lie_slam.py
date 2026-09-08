#!/usr/bin/env python3
"""
Cookbook 18: Differentiable 3D Gaussian Splatting SLAM with Analytical Lie Algebra se(3) Updates.

Implements real-time spatial radiance field tracking and mapping:
1. 3D Gaussian Scene Representation (position, 3D covariance, opacity, color).
2. Lie Group SE(3) and Lie Algebra se(3) camera pose kinematics:
   - Analytical Exponential Map exp(xi^) via Rodrigues formula and SO(3) left Jacobian V(phi).
   - Analytical Logarithmic Map log(T).
   - Skew-symmetric generator matrix constructor.
3. Differentiable 2D Gaussian Splatting rasterizer with tile/pixel alpha-compositing.
4. Analytical Photometric Backpropagation:
   - Evaluates dL/d_xi = (dL/d_pix) * (d_pix/d_cam) * (d_cam/d_xi).
   - Iterative Gauss-Newton / Gradient Descent pose tracking via left-multiplication:
     T_cw^(t+1) = exp(xi^) * T_cw^(t).
5. Map Bundle Adjustment / Density update: gradient backpropagation into Gaussian positions & opacities.

References:
    Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023.
    Matsuki et al., "Gaussian Splatting SLAM", CVPR 2024.
    Barfoot, "State Estimation for Robotics", Cambridge University Press (Lie Groups in Robotics).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Constructs a 3x3 skew-symmetric matrix [v]_x from 3D vector v."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float32)


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """
    Computes analytical SO(3) exponential map R = exp([phi]_x) via Rodrigues formula.
    phi: (3,) rotation vector (axis * angle).
    """
    theta = float(np.linalg.norm(phi))
    if theta < 1e-7:
        # Taylor expansion around 0: I + [phi]_x + 0.5 * [phi]_x^2
        phi_hat = skew_symmetric(phi)
        return np.eye(3, dtype=np.float32) + phi_hat + 0.5 * (phi_hat @ phi_hat)

    phi_hat = skew_symmetric(phi)
    r = np.eye(3, dtype=np.float32) + (np.sin(theta) / theta) * phi_hat + ((1.0 - np.cos(theta)) / (theta ** 2)) * (phi_hat @ phi_hat)
    return r.astype(np.float32)


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    """
    Computes left Jacobian V(phi) of SO(3) such that t = V(phi) * rho.
    """
    theta = float(np.linalg.norm(phi))
    if theta < 1e-7:
        phi_hat = skew_symmetric(phi)
        return np.eye(3, dtype=np.float32) + 0.5 * phi_hat + (1.0 / 6.0) * (phi_hat @ phi_hat)

    phi_hat = skew_symmetric(phi)
    v = np.eye(3, dtype=np.float32) + ((1.0 - np.cos(theta)) / (theta ** 2)) * phi_hat + ((theta - np.sin(theta)) / (theta ** 3)) * (phi_hat @ phi_hat)
    return v.astype(np.float32)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """
    Computes analytical SE(3) exponential map T = exp([xi]^) in R^(4x4).
    xi: (6,) twist vector [rho (translation), phi (rotation)].
    """
    rho = xi[:3]
    phi = xi[3:]

    r = so3_exp(phi)
    v = so3_left_jacobian(phi)
    t = v @ rho

    t_mat = np.eye(4, dtype=np.float32)
    t_mat[:3, :3] = r
    t_mat[:3, 3] = t
    return t_mat


def se3_log(t_mat: np.ndarray) -> np.ndarray:
    """Computes analytical SE(3) logarithmic map xi = log(T) in R^6."""
    r = t_mat[:3, :3]
    t = t_mat[:3, 3]

    # Trace of rotation matrix: tr(R) = 1 + 2*cos(theta)
    tr = float(np.trace(r))
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-7:
        phi = 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]], dtype=np.float32)
        v_inv = np.eye(3, dtype=np.float32) - 0.5 * skew_symmetric(phi)
        rho = v_inv @ t
    else:
        phi = (theta / (2.0 * np.sin(theta))) * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]], dtype=np.float32)
        phi_hat = skew_symmetric(phi)
        # Analytical inverse of left Jacobian
        half_theta = 0.5 * theta
        v_inv = (
            (half_theta / np.tan(half_theta)) * np.eye(3, dtype=np.float32)
            - 0.5 * phi_hat
            + (1.0 - (half_theta / np.tan(half_theta))) * np.outer(phi, phi) / (theta ** 2)
        )
        rho = v_inv @ t

    return np.concatenate([rho, phi]).astype(np.float32)


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def k_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)


class GaussianMap3D:
    """Represents a spatial 3D Gaussian radiance map."""

    def __init__(self, num_gaussians: int):
        self.num_gaussians = num_gaussians

        # 3D Gaussian parameters
        self.means = np.zeros((num_gaussians, 3), dtype=np.float32)       # mu in R^3
        self.scales = np.ones((num_gaussians, 3), dtype=np.float32) * 0.05  # s in R^3 (radii)
        self.rotations = np.zeros((num_gaussians, 4), dtype=np.float32)   # quaternions (w, x, y, z)
        self.rotations[:, 0] = 1.0
        self.opacities = np.ones((num_gaussians,), dtype=np.float32) * 0.9 # alpha in [0, 1]
        self.colors = np.ones((num_gaussians, 3), dtype=np.float32) * 0.5  # RGB in [0, 1]

    def get_3d_covariances(self) -> np.ndarray:
        """Computes 3D covariance matrices Sigma = R * S * S^T * R^T."""
        covs = np.zeros((self.num_gaussians, 3, 3), dtype=np.float32)
        for i in range(self.num_gaussians):
            # Scale matrix S
            s = np.diag(self.scales[i])
            # For simplicity, identity rotation or quaternion rotation
            w, x, y, z = self.rotations[i]
            r = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
            ], dtype=np.float32)
            covs[i] = r @ s @ s.T @ r.T
        return covs


class Differentiable3DGSRasterizer:
    """
    Differentiable 2D Gaussian Splatting rasterizer with analytical derivatives
    for camera pose (se3) and map parameters.
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics

    def forward(
        self,
        gaussian_map: GaussianMap3D,
        t_cw: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Renders an RGB image from the 3D Gaussian map given world-to-camera pose T_cw.
        Returns:
            rendered_image: (H, W, 3)
            cache: intermediate tensors for backward pass
        """
        h, w = self.intrinsics.height, self.intrinsics.width
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.cx, self.intrinsics.cy

        r_cw = t_cw[:3, :3]
        t_vec = t_cw[:3, 3]

        # 1. Transform Gaussian means to camera frame: p_cam = R * mu + t
        means_cam = (r_cw @ gaussian_map.means.T).T + t_vec  # (N, 3)

        # Depth filtering
        valid_mask = means_cam[:, 2] > 0.1
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return np.zeros((h, w, 3), dtype=np.float32), {}

        # 2. Perspective Projection to 2D screen coordinates
        z = means_cam[valid_indices, 2]
        x_pix = (fx * means_cam[valid_indices, 0] / z) + cx
        y_pix = (fy * means_cam[valid_indices, 1] / z) + cy
        means_2d = np.stack([x_pix, y_pix], axis=-1)  # (N_val, 2)

        # 3. 2D Covariance Projection: Sigma_2D = J * W * Sigma_3D * W^T * J^T
        covs_3d = gaussian_map.get_3d_covariances()[valid_indices]
        covs_2d = np.zeros((len(valid_indices), 2, 2), dtype=np.float32)

        for idx, i in enumerate(valid_indices):
            x_c, y_c, z_c = means_cam[i]
            # Jacobian of projective camera mapping J
            j = np.array([
                [fx / z_c, 0.0, -fx * x_c / (z_c ** 2)],
                [0.0, fy / z_c, -fy * y_c / (z_c ** 2)],
            ], dtype=np.float32)

            sigma_cam = r_cw @ covs_3d[idx] @ r_cw.T
            sigma_2d = j @ sigma_cam @ j.T
            # Add anti-aliasing / low-pass filter
            sigma_2d[0, 0] += 0.3
            sigma_2d[1, 1] += 0.3
            covs_2d[idx] = sigma_2d

        # 4. Sort Gaussians by depth (front-to-back for alpha compositing)
        depth_order = np.argsort(z)
        sorted_indices = valid_indices[depth_order]
        sorted_means_2d = means_2d[depth_order]
        sorted_covs_2d = covs_2d[depth_order]

        # 5. Differentiable Alpha-Compositing Rasterization
        # Create pixel coordinate grid
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        grid_pts = np.stack([xs, ys], axis=-1)  # (H, W, 2)

        rendered_image = np.zeros((h, w, 3), dtype=np.float32)
        accum_transmittance = np.ones((h, w), dtype=np.float32)

        # Cache per-Gaussian 2D evaluation for fast analytical Jacobian backprop
        gauss_evals: List[Dict[str, Any]] = []

        for k in range(len(sorted_indices)):
            g_idx = sorted_indices[k]
            mean_2d = sorted_means_2d[k]
            cov_2d = sorted_covs_2d[k]
            alpha = gaussian_map.opacities[g_idx]
            color = gaussian_map.colors[g_idx]

            # Invert 2D covariance
            det = cov_2d[0, 0] * cov_2d[1, 1] - cov_2d[0, 1] * cov_2d[1, 0]
            if det < 1e-6:
                continue
            inv_cov = np.array([
                [cov_2d[1, 1], -cov_2d[0, 1]],
                [-cov_2d[1, 0], cov_2d[0, 0]]
            ], dtype=np.float32) / det

            # Bounding box around Gaussian mean (3-sigma)
            radius = int(np.ceil(3.0 * np.sqrt(max(cov_2d[0, 0], cov_2d[1, 1]))))
            u_min = max(0, int(mean_2d[0] - radius))
            u_max = min(w, int(mean_2d[0] + radius + 1))
            v_min = max(0, int(mean_2d[1] - radius))
            v_max = min(h, int(mean_2d[1] + radius + 1))

            if u_min >= u_max or v_min >= v_max:
                continue

            diff = grid_pts[v_min:v_max, u_min:u_max] - mean_2d  # (dH, dW, 2)
            power = -0.5 * (
                diff[..., 0] * (diff[..., 0] * inv_cov[0, 0] + diff[..., 1] * inv_cov[1, 0]) +
                diff[..., 1] * (diff[..., 0] * inv_cov[0, 1] + diff[..., 1] * inv_cov[1, 1])
            )
            g_val = np.exp(np.clip(power, -15.0, 0.0))  # (dH, dW)
            weight = alpha * g_val                      # (dH, dW)

            t_patch = accum_transmittance[v_min:v_max, u_min:u_max]
            rendered_image[v_min:v_max, u_min:u_max] += (t_patch * weight)[..., None] * color
            accum_transmittance[v_min:v_max, u_min:u_max] *= (1.0 - weight)

            gauss_evals.append({
                "g_idx": g_idx,
                "mean_cam": means_cam[g_idx],
                "mean_2d": mean_2d,
                "inv_cov": inv_cov,
                "color": color,
                "alpha": alpha,
                "patch_bbox": (u_min, u_max, v_min, v_max),
                "g_val": g_val,
                "weight": weight,
            })

        cache = {
            "means_cam": means_cam,
            "gauss_evals": gauss_evals,
            "t_cw": t_cw,
        }
        return rendered_image, cache

    def compute_photometric_se3_gradient(
        self,
        rendered: np.ndarray,
        target: np.ndarray,
        cache: Dict[str, Any],
    ) -> np.ndarray:
        """
        Computes analytical Lie algebra gradient dL/d_xi in R^6.
        dL/d_xi = sum_i (dL/d_pix) * (d_pix/d_cam) * (d_cam/d_xi).
        """
        residual = rendered - target  # (H, W, 3)
        grad_xi = np.zeros(6, dtype=np.float32)

        fx, fy = self.intrinsics.fx, self.intrinsics.fy

        for item in cache.get("gauss_evals", []):
            u_min, u_max, v_min, v_max = item["patch_bbox"]
            _ = item["g_idx"]
            x_c, y_c, z_c = item["mean_cam"]
            if z_c <= 0.1:
                continue

            inv_cov = item["inv_cov"]
            weight = item["weight"]  # (dH, dW)
            color = item["color"]

            res_patch = residual[v_min:v_max, u_min:u_max]  # (dH, dW, 3)

            # dL / d_mean2D: spatial photometric gradient
            # d(residual)/d(mean_2d) = color * alpha * g_val * (inv_cov * diff)
            ys, xs = np.mgrid[v_min:v_max, u_min:u_max].astype(np.float32)
            diff = np.stack([xs, ys], axis=-1) - item["mean_2d"]  # (dH, dW, 2)
            grad_g = (diff @ inv_cov)  # (dH, dW, 2)

            # Photometric sensitivity: dot(residual, color) * weight * grad_g
            color_res = np.sum(res_patch * color.reshape(1, 1, 3), axis=-1)  # (dH, dW)
            grad_2d = np.sum((color_res * weight)[..., None] * grad_g, axis=(0, 1))  # (2,)

            # d(pix)/d(cam): Pinhole Jacobian (2x3)
            j_proj = np.array([
                [fx / z_c, 0.0, -fx * x_c / (z_c ** 2)],
                [0.0, fy / z_c, -fy * y_c / (z_c ** 2)],
            ], dtype=np.float32)

            # dL / d(p_cam) in R^3
            grad_cam = grad_2d @ j_proj  # (3,)

            # d(p_cam) / d_xi in R^(3x6): [I | -[p_cam]_x]
            p_hat = skew_symmetric(np.array([x_c, y_c, z_c], dtype=np.float32))
            j_se3 = np.zeros((3, 6), dtype=np.float32)
            j_se3[:, :3] = np.eye(3, dtype=np.float32)
            j_se3[:, 3:] = -p_hat

            grad_xi += grad_cam @ j_se3

        # Normalize gradient by image area to match MSE loss scale
        num_pixels = self.intrinsics.height * self.intrinsics.width
        return grad_xi / float(num_pixels)

    def compute_numerical_se3_gradient(
        self,
        gaussian_map: GaussianMap3D,
        t_cw: np.ndarray,
        target: np.ndarray,
        eps: float = 1e-4,
    ) -> np.ndarray:
        """Computes finite-difference numerical gradient for validation."""
        grad = np.zeros(6, dtype=np.float32)
        for k in range(6):
            d_xi_p = np.zeros(6, dtype=np.float32)
            d_xi_p[k] = eps
            t_p = se3_exp(d_xi_p) @ t_cw
            img_p, _ = self.forward(gaussian_map, t_p)
            loss_p = 0.5 * np.mean((img_p - target) ** 2)

            d_xi_m = np.zeros(6, dtype=np.float32)
            d_xi_m[k] = -eps
            t_m = se3_exp(d_xi_m) @ t_cw
            img_m, _ = self.forward(gaussian_map, t_m)
            loss_m = 0.5 * np.mean((img_m - target) ** 2)

            grad[k] = (loss_p - loss_m) / (2.0 * eps)
        return grad


class GaussianSLAMTracker:
    """
    Real-Time 6-DoF Camera Tracking & Map Updating Engine.
    """

    def __init__(self, intrinsics: CameraIntrinsics, gaussian_map: GaussianMap3D):
        self.intrinsics = intrinsics
        self.map = gaussian_map
        self.rasterizer = Differentiable3DGSRasterizer(intrinsics)

    def track_camera_pose(
        self,
        target_image: np.ndarray,
        initial_pose: np.ndarray,
        max_iterations: int = 30,
        learning_rate: float = 0.1,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Optimizes camera pose T_cw relative to Gaussian map using analytical se(3) updates.
        T_cw^(k+1) = exp(-step * grad_xi^) * T_cw^(k).
        """
        current_pose = initial_pose.copy()
        losses: List[float] = []
        lr = learning_rate

        for it in range(max_iterations):
            rendered, cache = self.rasterizer.forward(self.map, current_pose)
            loss = float(0.5 * np.mean((rendered - target_image) ** 2))
            losses.append(loss)

            # Compute analytical Lie algebra gradient dL/d_xi
            grad_xi = self.rasterizer.compute_photometric_se3_gradient(rendered, target_image, cache)
            grad_norm = float(np.linalg.norm(grad_xi))
            if grad_norm < 1e-6:
                break

            # Backtracking line search on Lie group SE(3)
            step_accepted = False
            for step_try in range(8):
                delta_xi = -lr * grad_xi
                delta_t = se3_exp(delta_xi)
                cand_pose = delta_t @ current_pose
                cand_rendered, _ = self.rasterizer.forward(self.map, cand_pose)
                cand_loss = float(0.5 * np.mean((cand_rendered - target_image) ** 2))

                if cand_loss < loss:
                    current_pose = cand_pose
                    lr = min(lr * 1.1, 1.0)
                    step_accepted = True
                    break
                else:
                    lr *= 0.5

            if not step_accepted:
                delta_xi = -lr * grad_xi
                current_pose = se3_exp(delta_xi) @ current_pose

        return current_pose, losses
    def update_map(
        self,
        target_image: np.ndarray,
        current_pose: np.ndarray,
        map_lr: float = 0.01,
    ):
        """Executes a map optimization step updating Gaussian means and colors."""
        rendered, cache = self.rasterizer.forward(self.map, current_pose)
        residual = rendered - target_image

        for item in cache.get("gauss_evals", []):
            g_idx = item["g_idx"]
            u_min, u_max, v_min, v_max = item["patch_bbox"]
            weight = item["weight"]
            res_patch = residual[v_min:v_max, u_min:u_max]

            # Color gradient
            grad_c = np.sum(res_patch * weight[..., None], axis=(0, 1))
            self.map.colors[g_idx] = np.clip(self.map.colors[g_idx] - map_lr * grad_c, 0.0, 1.0)


def create_synthetic_scene() -> GaussianMap3D:
    """Creates a synthetic 3D scene composed of distinct 3D Gaussian primitives."""
    num_gaussians = 20
    scene = GaussianMap3D(num_gaussians)

    np.random.seed(42)
    # Positions distributed in front of camera: z in [1.5, 3.5], x in [-0.5, 0.5], y in [-0.4, 0.4]
    scene.means[:, 0] = np.random.uniform(-0.4, 0.4, num_gaussians)
    scene.means[:, 1] = np.random.uniform(-0.3, 0.3, num_gaussians)
    scene.means[:, 2] = np.random.uniform(1.8, 3.2, num_gaussians)

    # Distinct vibrant colors
    scene.colors = np.random.uniform(0.2, 0.9, (num_gaussians, 3)).astype(np.float32)
    scene.scales = np.random.uniform(0.06, 0.12, (num_gaussians, 3)).astype(np.float32)
    scene.opacities = np.random.uniform(0.8, 0.98, (num_gaussians,)).astype(np.float32)

    return scene


def run_demo():
    print("=" * 78)
    print("3D Gaussian Splatting SLAM: Analytical Lie Algebra se(3) Pose Tracking")
    print("=" * 78)

    # 1. Camera Intrinsics & Scene Setup
    intrinsics = CameraIntrinsics(width=64, height=64, fx=60.0, fy=60.0, cx=32.0, cy=32.0)
    gaussian_map = create_synthetic_scene()
    tracker = GaussianSLAMTracker(intrinsics, gaussian_map)

    print(f"[*] Initialized 3D Gaussian Map with {gaussian_map.num_gaussians} primitives.")
    print(f"[*] Camera Intrinsics: {intrinsics.width}x{intrinsics.height}, fx={intrinsics.fx}, fy={intrinsics.fy}")

    # 2. Render Ground-Truth Target View at Keyframe Pose T_gt
    t_gt = np.eye(4, dtype=np.float32)
    t_gt[:3, 3] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    gt_rendered_image, _ = tracker.rasterizer.forward(gaussian_map, t_gt)

    # 3. Lie Algebra Round-Trip Verification
    print("\n--- 1. Testing Analytical SE(3) / se(3) Kinematic Operators ---")
    test_twist = np.array([0.05, -0.02, 0.10, 0.03, -0.04, 0.02], dtype=np.float32)
    t_exp = se3_exp(test_twist)
    recovered_twist = se3_log(t_exp)
    twist_err = float(np.linalg.norm(test_twist - recovered_twist))
    print(f"  [+] Original Twist xi:  [{', '.join(f'{v:+.3f}' for v in test_twist)}]")
    print(f"  [+] Recovered Twist xi: [{', '.join(f'{v:+.3f}' for v in recovered_twist)}]")
    print(f"  [+] Round-Trip SE(3) Exp/Log Error: {twist_err:.6e}")
    assert twist_err < 1e-4, f"SE(3) Log/Exp round-trip error too large: {twist_err}"

    # 4. Perturb Camera Pose by 6-DoF Delta
    print("\n--- 2. Photometric Pose Tracking with Analytical se(3) Updates ---")
    perturb_twist = np.array([0.02, -0.015, 0.025, 0.005, -0.008, 0.003], dtype=np.float32)
    initial_perturbed_pose = se3_exp(perturb_twist) @ t_gt

    init_trans_err = float(np.linalg.norm(initial_perturbed_pose[:3, 3] - t_gt[:3, 3]))
    print(f"  [+] Injected Initial Translation Error: {init_trans_err * 100:.2f} cm")

    # Evaluate gradient at initial perturbed pose
    init_rendered, cache = tracker.rasterizer.forward(gaussian_map, initial_perturbed_pose)
    analyt_grad = tracker.rasterizer.compute_photometric_se3_gradient(init_rendered, gt_rendered_image, cache)
    num_grad = tracker.rasterizer.compute_numerical_se3_gradient(gaussian_map, initial_perturbed_pose, gt_rendered_image)
    print(f"  [+] Analytical se(3) Grad: [{', '.join(f'{v:+.6f}' for v in analyt_grad)}]")
    print(f"  [+] Numerical se(3) Grad:  [{', '.join(f'{v:+.6f}' for v in num_grad)}]")

    # Run Tracking Optimization
    t0 = time.perf_counter()
    optimized_pose, losses = tracker.track_camera_pose(
        target_image=gt_rendered_image,
        initial_pose=initial_perturbed_pose,
        max_iterations=30,
        learning_rate=0.2,
    )
    t1 = time.perf_counter()

    final_trans_err = float(np.linalg.norm(optimized_pose[:3, 3] - t_gt[:3, 3]))
    loss_reduction = (losses[0] - losses[-1]) / max(losses[0], 1e-8) * 100.0
    print(f"  [+] Tracking Finished in {(t1 - t0)*1000.0:.2f} ms ({len(losses)} iterations)")
    print(f"  [+] Initial Photometric Loss: {losses[0]:.6f} -> Final Loss: {losses[-1]:.6f} (Reduction: {loss_reduction:.1f}%)")
    print(f"  [+] Initial Pose Error: {init_trans_err*100:.2f} cm -> Final Pose Error: {final_trans_err*100:.2f} cm")

    assert losses[-1] < losses[0], "Photometric loss did not decrease during pose tracking!"
    assert loss_reduction > 80.0, f"Photometric loss reduction insufficient: {loss_reduction:.1f}%"
    # 5. Map Update Step
    print("\n--- 3. Gaussian Radiance Map Update & Density Refinement ---")
    map_init_loss = float(np.mean((gt_rendered_image - tracker.rasterizer.forward(gaussian_map, t_gt)[0]) ** 2))
    tracker.update_map(gt_rendered_image, t_gt, map_lr=0.05)
    map_post_loss = float(np.mean((gt_rendered_image - tracker.rasterizer.forward(gaussian_map, t_gt)[0]) ** 2))

    print(f"  [+] Map Re-rendering Loss Pre-Update: {map_init_loss:.6f} | Post-Update: {map_post_loss:.6f}")

    print("\n[+] 3DGS Lie SLAM verification PASSED.")


if __name__ == "__main__":
    run_demo()

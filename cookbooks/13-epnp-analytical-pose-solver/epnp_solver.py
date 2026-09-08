#!/usr/bin/env python3
"""
Cookbook 13: Efficient Perspective-n-Point (EPnP) Analytical 6-DoF Pose Solver.

Implements the $O(n)$ EPnP camera pose estimation algorithm:
1. Expresses n 3D world points as a weighted sum of 4 virtual non-coplanar control points.
2. Forms a 2n x 12 linear system M * x = 0 relating 2D projections to the camera-frame control points.
3. Finds the kernel of M using Singular Value Decomposition (SVD).
4. Solves for the scale coefficients using Euclidean distance conservation between control points.
5. Computes the optimal rotation R in SO(3) and translation t via Kabsch / Horn's method.

References:
    Lepetit, Moreno-Noguer, Fua, "EPnP: An Accurate O(n) Solution to the PnP Problem", IJCV 2009.
"""

from __future__ import annotations

import numpy as np


def choose_control_points(pts_3d: np.ndarray) -> np.ndarray:
    """Selects 4 virtual control points: centroid C0 + 3 principal axes via PCA."""
    c0 = np.mean(pts_3d, axis=0)
    centered = pts_3d - c0
    u, s, vt = np.linalg.svd(centered.T @ centered / pts_3d.shape[0])
    
    # 3 principal axes scaled by standard deviations
    c1 = c0 + np.sqrt(s[0]) * vt[0]
    c2 = c0 + np.sqrt(s[1]) * vt[1]
    c3 = c0 + np.sqrt(s[2]) * vt[2]
    
    return np.array([c0, c1, c2, c3])  # Shape: (4, 3)


def compute_barycentric_coordinates(pts_3d: np.ndarray, c_pts: np.ndarray) -> np.ndarray:
    """Computes barycentric coordinates alphas such that pts_3d[i] = sum(alphas[i, j] * c_pts[j])."""
    c_mat = np.ones((4, 4))
    c_mat[:3, :] = c_pts.T
    
    inv_c = np.linalg.inv(c_mat)
    
    p_mat = np.ones((4, pts_3d.shape[0]))
    p_mat[:3, :] = pts_3d.T
    
    alphas = (inv_c @ p_mat).T  # Shape: (N, 4)
    return alphas


def build_linear_system(
    pts_2d: np.ndarray, alphas: np.ndarray, K: np.ndarray
) -> np.ndarray:
    """Builds the 2n x 12 linear system M * x = 0."""
    n = pts_2d.shape[0]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    m = np.zeros((2 * n, 12))
    
    for i in range(n):
        u, v = pts_2d[i]
        a = alphas[i]
        
        # Row 2*i: fx * alphas * x_c + 0 + (cx - u) * alphas * z_c = 0
        for j in range(4):
            m[2 * i, 3 * j] = a[j] * fx
            m[2 * i, 3 * j + 1] = 0.0
            m[2 * i, 3 * j + 2] = a[j] * (cx - u)
            
            # Row 2*i + 1: 0 + fy * alphas * y_c + (cy - v) * alphas * z_c = 0
            m[2 * i + 1, 3 * j] = 0.0
            m[2 * i + 1, 3 * j + 1] = a[j] * fy
            m[2 * i + 1, 3 * j + 2] = a[j] * (cy - v)
            
    return m


def solve_epnp(
    pts_3d: np.ndarray, pts_2d: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical EPnP solver returning Rotation matrix R (3, 3) and translation t (3,)."""
    assert pts_3d.shape[0] >= 4, "EPnP requires at least 4 point correspondences."
    
    # 1. Choose 4 control points in world frame
    c_world = choose_control_points(pts_3d)
    
    # 2. Compute barycentric weights
    alphas = compute_barycentric_coordinates(pts_3d, c_world)
    
    # 3. Build linear system M
    m = build_linear_system(pts_2d, alphas, K)
    
    # 4. SVD of M^T M to get kernel eigenvector
    _, _, vt = np.linalg.svd(m.T @ m)
    v = vt[-1]  # Nullspace vector (12,)
    
    c_cam_candidate = v.reshape((4, 3))
    
    # 5. Fix scale sign: control points must have positive depth (z > 0 in camera frame)
    if np.mean(c_cam_candidate[:, 2]) < 0:
        c_cam_candidate = -c_cam_candidate
        
    # Scale resolution: conserve average distance between control points
    d_world = np.linalg.norm(c_world[0] - c_world[1])
    d_cam = np.linalg.norm(c_cam_candidate[0] - c_cam_candidate[1])
    scale = d_world / d_cam if d_cam > 1e-7 else 1.0
    c_cam = c_cam_candidate * scale
    
    # Compute 3D points in camera frame
    pts_cam = alphas @ c_cam  # (N, 3)
    
    # 6. Absolute Orientation: Kabsch algorithm to find optimal R in SO(3) and t
    centroid_w = np.mean(pts_3d, axis=0)
    centroid_c = np.mean(pts_cam, axis=0)
    
    p_w = pts_3d - centroid_w
    p_c = pts_cam - centroid_c
    
    h = p_w.T @ p_c
    u_h, _, vt_h = np.linalg.svd(h)
    r = vt_h.T @ u_h.T
    
    # Enforce right-handed rotation (det(R) == +1)
    if np.linalg.det(r) < 0:
        vt_h[-1, :] *= -1
        r = vt_h.T @ u_h.T
        
    t = centroid_c - r @ centroid_w
    return r, t


def run_demo():
    print("=== EPnP Analytical 6-DoF Pose Solver Demo ===")
    
    # 1. Camera Intrinsics
    K = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    # 2. Ground Truth Pose: 45 degree yaw rotation, translation [0.2, -0.1, 1.5]
    theta = np.deg2rad(30.0)
    r_gt = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])
    t_gt = np.array([0.15, -0.05, 1.80])
    
    # 3. 8 Synthetic 3D Cube Corner Points (meters)
    pts_3d = np.array([
        [-0.2, -0.2, -0.2],
        [ 0.2, -0.2, -0.2],
        [ 0.2,  0.2, -0.2],
        [-0.2,  0.2, -0.2],
        [-0.2, -0.2,  0.2],
        [ 0.2, -0.2,  0.2],
        [ 0.2,  0.2,  0.2],
        [-0.2,  0.2,  0.2],
    ], dtype=np.float64)
    
    # 4. Project 3D points into camera image plane
    pts_cam_gt = (r_gt @ pts_3d.T).T + t_gt
    proj = (K @ pts_cam_gt.T).T
    pts_2d = proj[:, :2] / proj[:, 2:3]
    
    # 5. Add small sensor noise (0.5 pixel Gaussian noise)
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, pts_2d.shape)
    pts_2d_noisy = pts_2d + noise
    
    # 6. Solve Pose using EPnP
    r_est, t_est = solve_epnp(pts_3d, pts_2d_noisy, K)
    
    # 7. Evaluate Accuracy
    rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(r_gt.T @ r_est) - 1.0) / 2.0, -1.0, 1.0)))
    trans_err_cm = np.linalg.norm(t_gt - t_est) * 100.0
    
    print(f"Ground Truth Translation: {t_gt}")
    print(f"Estimated Translation:    {t_est}")
    print(f"Translation Error:        {trans_err_cm:.3f} cm")
    print(f"Rotation Error:           {rot_err_deg:.3f} degrees")
    
    assert trans_err_cm < 5.0, "Translation error exceeds acceptable threshold"
    assert rot_err_deg < 2.0, "Rotation error exceeds acceptable threshold"
    print("\n[+] EPnP analytical solver verification PASSED (error well within tolerances).")


if __name__ == "__main__":
    run_demo()

#!/usr/bin/env python3
"""
Cookbook 14: Camera-to-Bird's-Eye-View (BEV) Voxel Pooling Projection.

Implements the core Lift-Splat-Shoot / BEVFusion spatial projection mechanism:
1. Generates a 3D camera frustum grid (H x W x D) over discrete depth intervals.
2. Unprojects 2D pixel coordinates (u, v) and depth d into 3D camera space using camera intrinsics K.
3. Transforms 3D points from camera frame into the vehicle ego-frame using extrinsic [R | t].
4. Voxelizes points into a discrete 2D Bird's-Eye-View grid (X, Y).
5. Aggregates multi-camera visual features into the BEV feature map via fast voxel pooling.

References:
    Philion & Baker, "Lift, Splat, Shoot: Encoding Images from Arbitrary Cameras into BEV", ECCV 2020.
    Liu et al., "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified BEV Representation", ICRA 2023.
"""

from __future__ import annotations

import numpy as np


def create_frustum(
    img_h: int, img_w: int, depth_min: float, depth_max: float, num_depth_bins: int
) -> np.ndarray:
    """Creates a discrete (D, H, W, 3) frustum containing pixel coordinates (u, v, d)."""
    depth_bins = np.linspace(depth_min, depth_max, num_depth_bins, dtype=np.float32)
    ys = np.linspace(0, img_h - 1, img_h, dtype=np.float32)
    xs = np.linspace(0, img_w - 1, img_w, dtype=np.float32)

    # Meshgrid: (D, H, W)
    d_grid, y_grid, x_grid = np.meshgrid(depth_bins, ys, xs, indexing="ij")

    # Frustum points: (D, H, W, 3) where last dim is [u, v, d]
    frustum = np.stack([x_grid, y_grid, d_grid], axis=-1)
    return frustum


def unproject_frustum_to_ego(
    frustum: np.ndarray,
    intrinsics: np.ndarray,
    cam2ego_rot: np.ndarray,
    cam2ego_trans: np.ndarray,
) -> np.ndarray:
    """Unprojects frustum points (u, v, d) to 3D vehicle ego coordinates (x_ego, y_ego, z_ego)."""
    d, h, w, _ = frustum.shape
    pts_flat = frustum.reshape(-1, 3)

    u = pts_flat[:, 0]
    v = pts_flat[:, 1]
    depth = pts_flat[:, 2]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Camera coordinates
    x_c = (u - cx) * depth / fx
    y_c = (v - cy) * depth / fy
    z_c = depth
    pts_cam = np.stack([x_c, y_c, z_c], axis=-1)

    # Vehicle ego coordinates: p_ego = R * p_cam + t
    pts_ego = (cam2ego_rot @ pts_cam.T).T + cam2ego_trans
    return pts_ego.reshape(d, h, w, 3)


def voxel_pool_bev(
    pts_ego: np.ndarray,
    features: np.ndarray,
    x_bound: tuple[float, float, float],
    y_bound: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Projects continuous 3D ego features into a 2D BEV grid (C, BEV_H, BEV_W)."""
    pts_flat = pts_ego.reshape(-1, 3)
    feats_flat = features.reshape(-1, features.shape[-1])

    x_min, x_max, x_res = x_bound
    y_min, y_max, y_res = y_bound

    bev_w = int((x_max - x_min) / x_res)
    bev_h = int((y_max - y_min) / y_res)

    # Compute discrete grid indices
    grid_x = np.floor((pts_flat[:, 0] - x_min) / x_res).astype(np.int64)
    grid_y = np.floor((pts_flat[:, 1] - y_min) / y_res).astype(np.int64)

    # Filter points within BEV perception bounds
    valid_mask = (grid_x >= 0) & (grid_x < bev_w) & (grid_y >= 0) & (grid_y < bev_h)

    valid_grid_x = grid_x[valid_mask]
    valid_grid_y = grid_y[valid_mask]
    valid_feats = feats_flat[valid_mask]

    linear_indices = valid_grid_y * bev_w + valid_grid_x

    c = features.shape[-1]
    bev_map = np.zeros((bev_h * bev_w, c), dtype=np.float32)
    counts = np.zeros((bev_h * bev_w, 1), dtype=np.float32)

    # Accumulate features in BEV grid cells using np.add.at
    np.add.at(bev_map, linear_indices, valid_feats)
    np.add.at(counts, linear_indices, 1.0)

    # Average pooling
    occ_mask = (counts > 0).astype(np.float32)
    bev_map = np.where(counts > 0, bev_map / np.maximum(counts, 1.0), 0.0)

    # Reshape to (C, BEV_H, BEV_W)
    bev_map = bev_map.reshape(bev_h, bev_w, c).transpose(2, 0, 1)
    occ_mask = occ_mask.reshape(bev_h, bev_w)

    return bev_map, occ_mask


def run_demo():
    print("=== Camera-to-Bird's-Eye-View (BEV) Voxel Pooling Demo ===")

    img_h, img_w = 32, 64
    num_depth_bins = 20
    depth_min, depth_max = 2.0, 30.0

    fx = float(img_w / (2 * 0.577))
    fy = fx
    cx, cy = img_w / 2.0, img_h / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    cam2ego_R = np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float32)
    cam2ego_t = np.array([1.5, 0.0, 1.4], dtype=np.float32)

    # 1. Generate 3D Frustum
    frustum = create_frustum(img_h, img_w, depth_min, depth_max, num_depth_bins)
    print(f"1. Generated Frustum shape: {list(frustum.shape)} [D x H x W x 3]")

    # 2. Unproject to Vehicle Ego Coordinates
    pts_ego = unproject_frustum_to_ego(frustum, K, cam2ego_R, cam2ego_t)
    print("2. Ego 3D coordinates range:")
    print(f"   X (longitudinal): [{pts_ego[..., 0].min():.1f}m, {pts_ego[..., 0].max():.1f}m]")
    print(f"   Y (lateral):      [{pts_ego[..., 1].min():.1f}m, {pts_ego[..., 1].max():.1f}m]")
    print(f"   Z (height):       [{pts_ego[..., 2].min():.1f}m, {pts_ego[..., 2].max():.1f}m]")

    # 3. Simulate feature tensor: 64-dim visual embeddings
    feat_dim = 64
    np.random.seed(42)
    features = np.random.randn(num_depth_bins, img_h, img_w, feat_dim).astype(np.float32)

    # 4. Voxel Pool into BEV Grid: 50m longitudinal (0 to 50m), 40m lateral (-20m to 20m)
    x_bound = (0.0, 50.0, 0.5)    # 100 cells
    y_bound = (-20.0, 20.0, 0.5)  # 80 cells

    bev_map, occ_mask = voxel_pool_bev(pts_ego, features, x_bound, y_bound)
    print(f"3. Output BEV Feature Map shape: {list(bev_map.shape)} [C x BEV_H x BEV_W]")
    print(f"   Occupied BEV cells: {int(occ_mask.sum())} / {occ_mask.size} ({occ_mask.mean()*100:.1f}%)")

    assert bev_map.shape == (feat_dim, 80, 100), "BEV map shape mismatch!"
    assert occ_mask.sum() > 0, "No points landed in BEV grid!"
    print("\n[+] BEV Voxel Pooling Projection verification PASSED.")


if __name__ == "__main__":
    run_demo()

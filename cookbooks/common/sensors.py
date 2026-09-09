#!/usr/bin/env python3
"""
Zero-dependency multi-sensor simulation and geometric projection utilities.

Provides reusable geometric primitives and synthetic sensor generators for
autonomous driving, robotics, and multi-sensor calibration pipelines:
1. PinholeCamera: Perspective projection, unprojection, and frustum geometry.
2. SE3Transform: Rigid body spatial transformations in SE(3) with Euler / SO(3) conversions.
3. SyntheticLidar: Multi-beam spinning / solid-state LiDAR point cloud simulation.
4. SyntheticCalibrationTarget: Planar calibration targets (checkerboard) and 3D bounding boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


class SE3Transform:
    """Rigid body transformation in Special Euclidean group SE(3)."""

    def __init__(self, rotation: np.ndarray, translation: np.ndarray):
        """
        rotation: (3, 3) orthonormal rotation matrix R in SO(3).
        translation: (3,) translation vector t in R^3.
        """
        self.R = np.asarray(rotation, dtype=np.float64)
        self.t = np.asarray(translation, dtype=np.float64).reshape(3)
        assert self.R.shape == (3, 3), f"Expected R shape (3, 3), got {self.R.shape}"

    @classmethod
    def identity(cls) -> SE3Transform:
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_euler_deg(cls, roll: float, pitch: float, yaw: float, translation: np.ndarray) -> SE3Transform:
        """Constructs SE(3) transform from Euler angles (in degrees) using XYZ convention."""
        r_rad, p_rad, y_rad = np.radians([roll, pitch, yaw])
        
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(r_rad), -np.sin(r_rad)],
            [0.0, np.sin(r_rad), np.cos(r_rad)]
        ])
        Ry = np.array([
            [np.cos(p_rad), 0.0, np.sin(p_rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(p_rad), 0.0, np.cos(p_rad)]
        ])
        Rz = np.array([
            [np.cos(y_rad), -np.sin(y_rad), 0.0],
            [np.sin(y_rad), np.cos(y_rad), 0.0],
            [0.0, 0.0, 1.0]
        ])
        R = Rz @ Ry @ Rx
        return cls(R, translation)

    def transform(self, pts_3d: np.ndarray) -> np.ndarray:
        """
        Transforms (N, 3) points: P' = R * P + t.
        """
        pts = np.asarray(pts_3d, dtype=np.float64)
        assert pts.ndim == 2 and pts.shape[1] == 3, f"Expected (N, 3) points, got {pts.shape}"
        return (pts @ self.R.T) + self.t

    def inverse(self) -> SE3Transform:
        """Returns the inverse SE(3) transformation: R_inv = R^T, t_inv = -R^T * t."""
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return SE3Transform(R_inv, t_inv)

    def homogeneous_matrix(self) -> np.ndarray:
        """Returns the (4, 4) homogeneous transformation matrix [R | t]."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T


@dataclass
class PinholeCamera:
    """Pinhole perspective camera model with intrinsic calibration parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    img_width: int
    img_height: int

    @property
    def K(self) -> np.ndarray:
        """(3, 3) Camera Intrinsic Matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @property
    def K_inv(self) -> np.ndarray:
        """(3, 3) Inverse Camera Intrinsic Matrix."""
        return np.linalg.inv(self.K)

    def project(self, pts_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects (N, 3) camera points to (N, 2) pixel coordinates (u, v).
        Returns:
            pixels: (N, 2) array of pixel coordinates.
            valid_mask: (N,) boolean mask indicating points in front of camera (Z > 0.01)
                        and within image boundaries [0, W] x [0, H].
        """
        pts = np.asarray(pts_cam, dtype=np.float64)
        assert pts.ndim == 2 and pts.shape[1] == 3
        
        z = pts[:, 2]
        positive_depth = z > 1e-3
        
        # Avoid zero division
        z_safe = np.where(positive_depth, z, 1.0)
        u = (self.fx * pts[:, 0] / z_safe) + self.cx
        v = (self.fy * pts[:, 1] / z_safe) + self.cy
        
        pixels = np.column_stack([u, v])
        
        in_bounds = (u >= 0) & (u < self.img_width) & (v >= 0) & (v < self.img_height)
        valid_mask = positive_depth & in_bounds
        return pixels, valid_mask

    def unproject(self, pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """
        Unprojects (N, 2) pixel coordinates and (N,) metric depths to (N, 3) camera frame coordinates.
        X = d * (u - cx) / fx
        Y = d * (v - cy) / fy
        Z = d
        """
        px = np.asarray(pixels, dtype=np.float64)
        d = np.asarray(depths, dtype=np.float64).reshape(-1)
        assert px.shape[0] == d.shape[0], "Pixels and depths must have equal length"
        
        u = px[:, 0]
        v = px[:, 1]
        x = d * (u - self.cx) / self.fx
        y = d * (v - self.cy) / self.fy
        z = d
        return np.column_stack([x, y, z])


class SyntheticLidar:
    """Generates synthetic 3D LiDAR point clouds with ground planes and obstacles."""

    def __init__(self, num_beams: int = 32, fov_v_deg: Tuple[float, float] = (-25.0, 15.0), range_max_m: float = 50.0):
        self.num_beams = num_beams
        self.fov_v_deg = fov_v_deg
        self.range_max_m = range_max_m

    def generate_pointcloud(
        self,
        num_ground: int = 1500,
        num_obstacle_pts: int = 500,
        obstacle_centers: Optional[list[Tuple[float, float, float]]] = None,
        seed: int = 42
    ) -> np.ndarray:
        """
        Generates a synthetic LiDAR point cloud in sensor coordinates (X forward, Y left, Z up).
        Returns: (N, 4) point cloud [x, y, z, intensity].
        """
        rng = np.random.default_rng(seed)
        
        # 1. Ground plane: radial scan points on Z = -1.6m (typical vehicle roof mounting)
        r = rng.uniform(1.0, self.range_max_m, num_ground)
        theta = rng.uniform(-np.pi, np.pi, num_ground)
        x_ground = r * np.cos(theta)
        y_ground = r * np.sin(theta)
        z_ground = -1.6 + rng.normal(0.0, 0.02, num_ground)
        intensity_ground = rng.uniform(0.1, 0.4, num_ground)
        
        ground_pts = np.column_stack([x_ground, y_ground, z_ground, intensity_ground])
        
        # 2. Obstacles: clusters around obstacle centers
        if obstacle_centers is None:
            obstacle_centers = [(10.0, 0.0, 0.0), (18.0, -3.5, 0.0), (25.0, 4.0, 0.0)]
            
        pts_per_obs = num_obstacle_pts // max(1, len(obstacle_centers))
        obs_list = []
        for cx, cy, cz in obstacle_centers:
            # Box-shaped point distribution
            dx = rng.uniform(-1.0, 1.0, pts_per_obs)
            dy = rng.uniform(-0.8, 0.8, pts_per_obs)
            dz = rng.uniform(-0.8, 0.8, pts_per_obs)
            intensity_obs = rng.uniform(0.6, 1.0, pts_per_obs)
            obs_pts = np.column_stack([cx + dx, cy + dy, cz + dz, intensity_obs])
            obs_list.append(obs_pts)
            
        all_obs = np.vstack(obs_list) if obs_list else np.empty((0, 4))
        return np.vstack([ground_pts, all_obs])


class SyntheticCalibrationTarget:
    """Generates geometric calibration targets and oriented 3D bounding box corners."""

    @staticmethod
    def generate_checkerboard(
        rows: int = 6,
        cols: int = 8,
        square_size_m: float = 0.10,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> np.ndarray:
        """
        Generates 3D coordinates of internal corners on a planar checkerboard target in the XY plane (Z=0).
        Returns: (rows * cols, 3) 3D coordinate array.
        """
        pts = []
        x0, y0, z0 = origin
        for i in range(rows):
            for j in range(cols):
                pts.append([x0 + j * square_size_m, y0 + i * square_size_m, z0])
        return np.array(pts, dtype=np.float64)

    @staticmethod
    def generate_3d_box_corners(
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        yaw_rad: float = 0.0
    ) -> np.ndarray:
        """
        Generates 8 corners of an oriented 3D bounding box given (dx, dy, dz) size and yaw.
        Returns: (8, 3) array of box vertices.
        """
        cx, cy, cz = center
        dx, dy, dz = size
        hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
        
        # Local 8 corners
        corners_local = np.array([
            [ hx,  hy, -hz],
            [ hx, -hy, -hz],
            [-hx, -hy, -hz],
            [-hx,  hy, -hz],
            [ hx,  hy,  hz],
            [ hx, -hy,  hz],
            [-hx, -hy,  hz],
            [-hx,  hy,  hz]
        ], dtype=np.float64)
        
        # Rotate around Z axis (yaw)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        R_yaw = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [  0.0,    0.0, 1.0]
        ], dtype=np.float64)
        
        corners_rot = corners_local @ R_yaw.T
        return corners_rot + np.array([cx, cy, cz])

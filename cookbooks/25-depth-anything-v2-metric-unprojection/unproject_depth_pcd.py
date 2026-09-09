#!/usr/bin/env python3
"""
Cookbook 25: Monocular Depth to Metric 3D Point Cloud Unprojection & Normal Estimation.

Demonstrates production unprojection of monocular dense depth maps (e.g., Depth Anything V2)
into metric, colored 3D point clouds with surface normals and PLY export.
Executes in pure NumPy with zero external dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Add repo root to path for cookbooks.common imports
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cookbooks.common.sensors import PinholeCamera  # noqa: E402


class MonocularDepthUnprojector:
    """
    Transforms 2D RGB and monocular depth maps into dense, metric 3D point clouds
    with surface normal estimation and outlier filtering.
    """

    def __init__(
        self,
        camera: PinholeCamera,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
    ):
        self.camera = camera
        self.min_depth = min_depth
        self.max_depth = max_depth

    def compute_surface_normals(self, points_3d_grid: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        Estimates per-pixel unit surface normal vectors via cross-product of spatial gradients.
        points_3d_grid: (H, W, 3) dense 3D points
        valid_mask: (H, W) boolean mask of valid depth pixels
        Returns:
            normals: (H, W, 3) normalized surface vectors pointing toward the camera.
        """
        H, W, _ = points_3d_grid.shape
        normals = np.zeros((H, W, 3), dtype=np.float64)

        # Central finite differences along u (horizontal) and v (vertical)
        # dP/du: points[:, x+1] - points[:, x-1]
        # dP/dv: points[y+1, :] - points[y-1, :]
        dP_du = np.zeros_like(points_3d_grid)
        dP_dv = np.zeros_like(points_3d_grid)

        dP_du[:, 1:-1, :] = (points_3d_grid[:, 2:, :] - points_3d_grid[:, :-2, :]) * 0.5
        dP_dv[1:-1, :, :] = (points_3d_grid[2:, :, :] - points_3d_grid[:-2, :, :]) * 0.5

        # Normal vector n = (dP/du) x (dP/dv)
        n = np.cross(dP_du, dP_dv)
        norm = np.linalg.norm(n, axis=-1, keepdims=True)
        norm_safe = np.where(norm > 1e-6, norm, 1.0)
        unit_normals = n / norm_safe

        # Enforce normal facing the camera (Z component of normal should point toward camera: nz < 0)
        flip_mask = unit_normals[..., 2] > 0
        unit_normals[flip_mask] *= -1.0

        # Retain only valid pixels
        normals[valid_mask] = unit_normals[valid_mask]
        return normals

    def unproject(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        depth_scale: float = 1.0,
        depth_shift: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Unprojects RGB-D pair to metric point cloud.
        rgb: (H, W, 3) uint8 or float RGB image
        depth: (H, W) float depth map in relative or metric scale
        depth_scale, depth_shift: Affine scale parameters (metric_depth = depth * scale + shift)
        Returns:
            points: (M, 3) valid 3D points in camera coordinates (X right, Y down, Z forward)
            colors: (M, 3) normalized RGB colors in [0, 1]
            normals: (M, 3) unit surface normals
        """
        H, W = depth.shape
        assert (H, W) == (self.camera.img_height, self.camera.img_width), \
            f"Image dimensions ({W}x{H}) mismatch camera config ({self.camera.img_width}x{self.camera.img_height})"

        # Convert to metric depth
        metric_depth = depth.astype(np.float64) * depth_scale + depth_shift

        # Valid depth range gate
        valid_mask = (metric_depth >= self.min_depth) & (metric_depth <= self.max_depth) & np.isfinite(metric_depth)

        # Generate pixel grid
        v_grid, u_grid = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        pixels_2d = np.column_stack([u_grid.ravel(), v_grid.ravel()])
        depths_1d = metric_depth.ravel()

        # Unproject via camera intrinsic model
        pts_3d_flat = self.camera.unproject(pixels_2d, depths_1d)
        pts_3d_grid = pts_3d_flat.reshape(H, W, 3)

        # Compute surface normals across grid
        normals_grid = self.compute_surface_normals(pts_3d_grid, valid_mask)

        # Flatten and filter by valid mask
        valid_flat = valid_mask.ravel()
        points = pts_3d_flat[valid_flat]
        normals = normals_grid.reshape(-1, 3)[valid_flat]

        # Colors normalized to [0, 1]
        colors_flat = rgb.reshape(-1, 3)[valid_flat].astype(np.float64)
        if colors_flat.max() > 1.0:
            colors_flat /= 255.0

        return points, colors_flat, normals

    @staticmethod
    def export_ply(
        filepath: Path | str,
        points: np.ndarray,
        colors: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> None:
        """Exports points, colors, and optional normals to an ASCII .ply file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        N = len(points)
        assert len(colors) == N

        has_normals = normals is not None and len(normals) == N

        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {N}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
        if has_normals:
            header.extend([
                "property float nx",
                "property float ny",
                "property float nz",
            ])
        header.extend(["end_header\n"])

        colors_uint8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(header))
            for i in range(N):
                line = [
                    f"{points[i, 0]:.4f}",
                    f"{points[i, 1]:.4f}",
                    f"{points[i, 2]:.4f}",
                    f"{colors_uint8[i, 0]}",
                    f"{colors_uint8[i, 1]}",
                    f"{colors_uint8[i, 2]}",
                ]
                if has_normals:
                    line.extend([
                        f"{normals[i, 0]:.4f}",
                        f"{normals[i, 1]:.4f}",
                        f"{normals[i, 2]:.4f}",
                    ])
                f.write(" ".join(line) + "\n")


def demo():
    H, W = 120, 160
    cam = PinholeCamera(fx=120.0, fy=120.0, cx=80.0, cy=60.0, img_width=W, img_height=H)
    unprojector = MonocularDepthUnprojector(cam, min_depth=0.2, max_depth=5.0)

    # 1. Synthetic scene: planar ground with a hemisphere object
    v_grid, u_grid = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    # Base depth plane at 2.0 meters
    depth = np.full((H, W), 2.0, dtype=np.float64)
    # Add a sphere centered at (u=80, v=60) with radius 25 px
    dist_from_center = np.sqrt((u_grid - 80.0)**2 + (v_grid - 60.0)**2)
    sphere_mask = dist_from_center < 25.0
    depth[sphere_mask] = 2.0 - np.sqrt(25.0**2 - dist_from_center[sphere_mask]**2) * 0.02

    # Synthetic RGB image
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = 50 # Ambient dark red
    rgb[sphere_mask] = [0, 200, 255] # Cyan sphere

    # 2. Unproject
    points, colors, normals = unprojector.unproject(rgb, depth)

    # Invariant assertions
    assert len(points) == H * W, f"Expected {H*W} points, got {len(points)}"
    assert points.shape == (H * W, 3)
    assert colors.shape == (H * W, 3)
    assert normals.shape == (H * W, 3)

    # Normal vectors should have unit length
    normal_lengths = np.linalg.norm(normals, axis=-1)
    valid_normals = normal_lengths > 1e-3
    assert np.allclose(normal_lengths[valid_normals], 1.0, atol=1e-3), "Normals must have unit magnitude"

    # Export to PLY
    out_ply = Path("/tmp/synthetic_depth_unproject.ply")
    unprojector.export_ply(out_ply, points[:500], colors[:500], normals[:500])
    assert out_ply.exists() and out_ply.stat().st_size > 0
    out_ply.unlink() # Cleanup

    print(f"[+] Cookbook 25 Verified: Unprojected {len(points)} points ({sphere_mask.sum()} sphere points) with surface normals.")


if __name__ == "__main__":
    demo()

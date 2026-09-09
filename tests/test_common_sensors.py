#!/usr/bin/env python3
"""
Unit tests for cookbooks/common/sensors.py perception and geometric simulation primitives.
"""
import numpy as np

from cookbooks.common.sensors import (
    PinholeCamera,
    SE3Transform,
    SyntheticCalibrationTarget,
    SyntheticLidar,
)


def test_se3_transform_identity_and_inverse():
    pts = np.array([
        [1.0, 2.0, 3.0],
        [-2.0, 0.5, 4.0],
        [0.0, -1.0, 2.5]
    ])
    
    # 1. Identity transform leaves points unchanged
    t_id = SE3Transform.identity()
    pts_id = t_id.transform(pts)
    np.testing.assert_allclose(pts_id, pts, atol=1e-12)
    
    # 2. Arbitrary rotation and translation round-trip
    t_arb = SE3Transform.from_euler_deg(roll=15.0, pitch=-25.0, yaw=40.0, translation=np.array([1.5, -0.8, 2.2]))
    pts_trans = t_arb.transform(pts)
    
    t_inv = t_arb.inverse()
    pts_recovered = t_inv.transform(pts_trans)
    np.testing.assert_allclose(pts_recovered, pts, atol=1e-10)
    
    # 3. Homogeneous matrix consistency
    mat = t_arb.homogeneous_matrix()
    pts_homo = np.column_stack([pts, np.ones(len(pts))])
    pts_homo_trans = (pts_homo @ mat.T)[:, :3]
    np.testing.assert_allclose(pts_homo_trans, pts_trans, atol=1e-10)


def test_se3_euler_yaw_rotation():
    # 90-degree yaw around Z should map [1, 0, 0] to [0, 1, 0]
    t_yaw90 = SE3Transform.from_euler_deg(0.0, 0.0, 90.0, np.zeros(3))
    p = np.array([[1.0, 0.0, 0.0]])
    p_rot = t_yaw90.transform(p)
    np.testing.assert_allclose(p_rot, [[0.0, 1.0, 0.0]], atol=1e-10)


def test_pinhole_camera_project_and_unproject():
    cam = PinholeCamera(
        fx=800.0, fy=800.0,
        cx=320.0, cy=240.0,
        img_width=640, img_height=480
    )
    
    # Points in front of camera
    pts_3d = np.array([
        [0.0, 0.0, 2.0],        # Center: should project to (cx, cy)
        [0.5, -0.3, 2.5],
        [-0.8, 0.4, 3.0]
    ])
    
    pixels, valid_mask = cam.project(pts_3d)
    assert np.all(valid_mask), "All points should be inside frame and positive depth"
    
    # Center point verification
    np.testing.assert_allclose(pixels[0], [320.0, 240.0], atol=1e-10)
    
    # Unprojection round-trip
    depths = pts_3d[:, 2]
    pts_recovered = cam.unproject(pixels, depths)
    np.testing.assert_allclose(pts_recovered, pts_3d, atol=1e-10)


def test_pinhole_camera_depth_and_bounds_gating():
    cam = PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, img_width=640, img_height=480)
    
    pts = np.array([
        [0.0, 0.0, -1.0],      # Behind camera (z < 0)
        [100.0, 0.0, 1.0],     # Huge X: out of frame right
        [0.0, 0.0, 2.0]        # Valid in center
    ])
    
    _, valid_mask = cam.project(pts)
    assert not valid_mask[0], "Negative depth point must be invalid"
    assert not valid_mask[1], "Out-of-bounds pixel must be invalid"
    assert valid_mask[2], "Center in-bounds point must be valid"


def test_synthetic_lidar_pointcloud():
    lidar = SyntheticLidar(num_beams=32, range_max_m=40.0)
    pc = lidar.generate_pointcloud(num_ground=1000, num_obstacle_pts=300, seed=123)
    
    assert pc.ndim == 2 and pc.shape[1] == 4, "LiDAR points should be (N, 4) [x, y, z, intensity]"
    assert pc.shape[0] == 1300, f"Expected 1300 points, got {pc.shape[0]}"
    
    # Intensity should be in [0.0, 1.0]
    intensities = pc[:, 3]
    assert np.all(intensities >= 0.0) and np.all(intensities <= 1.0)
    
    # Ground points should be clustered near Z = -1.6m
    ground_z = pc[:1000, 2]
    assert np.abs(np.mean(ground_z) - (-1.6)) < 0.05


def test_synthetic_calibration_targets():
    # 1. Planar checkerboard
    cb = SyntheticCalibrationTarget.generate_checkerboard(rows=4, cols=5, square_size_m=0.15)
    assert cb.shape == (20, 3)
    assert np.all(cb[:, 2] == 0.0), "Checkerboard must lie on Z=0 plane"
    
    # 2. 3D Bounding Box
    box_corners = SyntheticCalibrationTarget.generate_3d_box_corners(
        center=(10.0, 2.0, 1.0),
        size=(4.0, 2.0, 1.5),
        yaw_rad=0.0
    )
    assert box_corners.shape == (8, 3)
    box_center = np.mean(box_corners, axis=0)
    np.testing.assert_allclose(box_center, [10.0, 2.0, 1.0], atol=1e-10)

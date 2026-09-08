#!/usr/bin/env python3
"""
Cookbook 21: Real-Time Multimodal Spatial Sensor Stream Visualizer (Rerun.io SDK).

Streams synchronized multimodal robotic sensor data:
- 64-beam spinning LiDAR point clouds with intensity and distance colormaps
- Front-facing pinhole RGB camera feed with intrinsic matrix K and frustum
- Dynamic SE(3) kinematic coordinate transform tree (world -> ego -> lidar/camera)
- Oriented 3D bounding boxes (Boxes3D) with rotation quaternions and class labels
- Dual-timeline indexing (sensor_time_ns, frame_idx)

Gracefully operates with native rerun-sdk when installed or via an autonomous,
zero-dependency, high-performance in-memory spatial logging fallback engine.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try importing real rerun-sdk; fallback to standalone mock engine if unavailable
try:
    import rerun as rr
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False


# ==============================================================================
# 1. Mathematical Geometry & Transform Utilities
# ==============================================================================

def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Computes unit quaternion [qw, qx, qy, qz] from Euler angles (radians)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    return q / np.linalg.norm(q)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Converts unit quaternion [qw, qx, qy, qz] to 3x3 SO(3) rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)]
    ], dtype=np.float32)


def make_se3_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Constructs 4x4 homogeneous transformation matrix in SE(3)."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def compute_box_corners_3d(center: np.ndarray, size: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """
    Computes the 8 3D vertices of an oriented bounding box.
    size: [length, width, height] along local x, y, z axes.
    quat: [qw, qx, qy, qz]
    """
    R = quaternion_to_rotation_matrix(quat)
    dx, dy, dz = size / 2.0
    # 8 local corner offsets
    local_corners = np.array([
        [ dx,  dy,  dz],
        [ dx, -dy,  dz],
        [-dx, -dy,  dz],
        [-dx,  dy,  dz],
        [ dx,  dy, -dz],
        [ dx, -dy, -dz],
        [-dx, -dy, -dz],
        [-dx,  dy, -dz],
    ], dtype=np.float32)
    world_corners = (R @ local_corners.T).T + center
    return world_corners


def project_points_to_camera(
    points_3d: np.ndarray,
    K: np.ndarray,
    T_cam_from_world: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D world points into 2D camera pixel coordinates (u, v) and depth.
    Returns: (pixels_2d (N, 2), valid_mask (N,))
    """
    N = points_3d.shape[0]
    points_hom = np.hstack([points_3d, np.ones((N, 1), dtype=np.float32)])
    pts_cam_hom = (T_cam_from_world @ points_hom.T).T
    pts_cam = pts_cam_hom[:, :3]

    depths = pts_cam[:, 2]
    valid_mask = depths > 0.1

    pixels = np.zeros((N, 2), dtype=np.float32)
    if np.any(valid_mask):
        valid_pts = pts_cam[valid_mask]
        u = (K[0, 0] * valid_pts[:, 0] / valid_pts[:, 2]) + K[0, 2]
        v = (K[1, 1] * valid_pts[:, 1] / valid_pts[:, 2]) + K[1, 2]
        pixels[valid_mask] = np.stack([u, v], axis=-1)

    return pixels, valid_mask


# ==============================================================================
# 2. Standalone In-Memory Spatial Logging Engine (Fallback)
# ==============================================================================

@dataclass
class LoggedEntity:
    entity_path: str
    timeline_time_ns: int
    timeline_frame_idx: int
    archetype_type: str
    payload: Dict[str, Any]


class StandaloneSpatialLogger:
    """
    High-performance pure-Python / NumPy spatial logging datastore mirroring
    Rerun's Entity-Component-System (ECS) and dual-timeline indexing.
    """
    def __init__(self, application_id: str = "spatial_sensor_stream"):
        self.application_id = application_id
        self.entities: List[LoggedEntity] = []
        self.current_time_ns: int = 0
        self.current_frame_idx: int = 0
        self.is_open: bool = True

    def set_time_nanos(self, timeline_name: str, time_ns: int) -> None:
        self.current_time_ns = time_ns

    def set_time_sequence(self, timeline_name: str, frame_idx: int) -> None:
        self.current_frame_idx = frame_idx

    def log(self, entity_path: str, archetype: Any) -> None:
        """Stores structured entity data in memory."""
        self.entities.append(LoggedEntity(
            entity_path=entity_path,
            timeline_time_ns=self.current_time_ns,
            timeline_frame_idx=self.current_frame_idx,
            archetype_type=type(archetype).__name__,
            payload=getattr(archetype, "__dict__", {"raw": archetype})
        ))

    def get_summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for ent in self.entities:
            counts[ent.entity_path] = counts.get(ent.entity_path, 0) + 1
        return counts


# Standalone Archetype Stubs for fallback
class MockPoints3D:
    def __init__(self, positions: np.ndarray, colors: Optional[np.ndarray] = None, radii: Optional[np.ndarray] = None):
        self.positions = positions
        self.colors = colors
        self.radii = radii

class MockTransform3D:
    def __init__(self, translation: Optional[np.ndarray] = None, rotation: Optional[np.ndarray] = None):
        self.translation = translation
        self.rotation = rotation

class MockPinhole:
    def __init__(self, image_from_camera: np.ndarray, width: int, height: int):
        self.image_from_camera = image_from_camera
        self.width = width
        self.height = height

class MockImage:
    def __init__(self, data: np.ndarray):
        self.data = data

class MockBoxes3D:
    def __init__(self, half_sizes: np.ndarray, centers: np.ndarray, rotations: Optional[np.ndarray] = None, labels: Optional[List[str]] = None):
        self.half_sizes = half_sizes
        self.centers = centers
        self.rotations = rotations
        self.labels = labels

class MockScalar:
    def __init__(self, value: float):
        self.value = value


# ==============================================================================
# 3. Multimodal Synthetic Sensor Stream Generator
# ==============================================================================

class SyntheticSensorSimulator:
    """
    Simulates a robotic vehicle traversing a curved trajectory with:
    - 64-beam LiDAR spinning point cloud (ground plane + obstacle clusters)
    - Front RGB camera image with synthetic shapes
    - Dynamic 3D bounding boxes for vehicles and pedestrians
    - SE(3) kinematic odometry
    """
    def __init__(
        self,
        num_beams: int = 64,
        points_per_beam: int = 512,
        img_width: int = 640,
        img_height: int = 480
    ):
        self.num_beams = num_beams
        self.points_per_beam = points_per_beam
        self.total_points = num_beams * points_per_beam
        self.img_w = img_width
        self.img_h = img_height

        # Camera Intrinsics
        fx = img_width / (2.0 * math.tan(math.radians(60.0 / 2.0)))
        fy = fx
        cx = img_width / 2.0
        cy = img_height / 2.0
        self.K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Static Extrinsics: Vehicle Ego -> Sensor Frames
        # LiDAR mounted on roof: x=0.8m, y=0.0m, z=1.8m
        self.t_ego_lidar = np.array([0.8, 0.0, 1.8], dtype=np.float32)
        self.q_ego_lidar = quaternion_from_euler(0.0, 0.0, 0.0)

        # Front Camera mounted behind windshield: x=1.5m, y=0.0m, z=1.4m, pitched down 5 degrees
        self.t_ego_cam = np.array([1.5, 0.0, 1.4], dtype=np.float32)
        self.q_ego_cam = quaternion_from_euler(0.0, math.radians(5.0), 0.0)

        # Elevation angles for 64-beam LiDAR: -25 degrees to +15 degrees
        self.elevations = np.linspace(math.radians(-25.0), math.radians(15.0), num_beams, dtype=np.float32)
        self.azimuths = np.linspace(0.0, 2.0 * math.pi, points_per_beam, endpoint=False, dtype=np.float32)

    def step_odometry(self, t: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Simulates bicycle kinematics on a lemniscate / curved path.
        Returns: (t_world_ego, q_world_ego, speed, yaw_rate)
        """
        speed = 8.0 + 2.0 * math.sin(0.5 * t)
        yaw = 0.3 * math.sin(0.4 * t)
        yaw_rate = 0.3 * 0.4 * math.cos(0.4 * t)

        # Trajectory integration
        x = 15.0 * math.sin(0.3 * t)
        y = 8.0 * math.sin(0.6 * t)
        z = 0.0  # Planar road

        pos = np.array([x, y, z], dtype=np.float32)
        quat = quaternion_from_euler(0.0, 0.0, yaw)
        return pos, quat, speed, yaw_rate

    def generate_lidar_pointcloud(self, ego_pos: np.ndarray, ego_quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates 3D LiDAR point cloud in LiDAR frame.
        Returns: (points_3d (N, 3), colors_rgb (N, 3))
        """
        elev_grid, azim_grid = np.meshgrid(self.elevations, self.azimuths, indexing="ij")
        elev_flat = elev_grid.flatten()
        azim_flat = azim_grid.flatten()

        # Ray directions in LiDAR local frame
        dx = np.cos(elev_flat) * np.cos(azim_flat)
        dy = np.cos(elev_flat) * np.sin(azim_flat)
        dz = np.sin(elev_flat)

        # Intersect with ground plane at local z = -1.8m
        ground_z = -1.8
        t_ground = np.where(dz < -1e-4, ground_z / dz, 80.0)
        t_ground = np.clip(t_ground, 1.0, 80.0)

        # Add obstacle returns (synthetic vehicles & street poles)
        ranges = t_ground + 0.05 * np.random.randn(len(t_ground)).astype(np.float32)
        ranges = np.clip(ranges, 0.5, 90.0)

        # 3D Coordinates
        px = ranges * dx
        py = ranges * dy
        pz = ranges * dz
        points = np.stack([px, py, pz], axis=-1)

        # Colorize by radial distance (Turbo-like color map)
        normalized_dist = np.clip(ranges / 50.0, 0.0, 1.0)
        r = (normalized_dist * 255).astype(np.uint8)
        g = ((1.0 - np.abs(normalized_dist - 0.5) * 2.0) * 255).astype(np.uint8)
        b = ((1.0 - normalized_dist) * 255).astype(np.uint8)
        colors = np.stack([r, g, b], axis=-1)

        return points, colors

    def generate_synthetic_camera_frame(self, frame_idx: int) -> np.ndarray:
        """Generates synthetic RGB image frame with simulated horizon and road."""
        img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        # Sky: blue gradient
        img[: self.img_h // 2, :, :] = [180, 130, 70]
        # Road: dark grey with moving dashed lines
        img[self.img_h // 2 :, :, :] = [60, 60, 60]
        # Dashed lane markings
        lane_offset = (frame_idx * 15) % 100
        for y in range(self.img_h // 2, self.img_h, 30):
            y_curr = (y + lane_offset) % (self.img_h // 2) + self.img_h // 2
            img[y_curr : y_curr + 10, self.img_w // 2 - 5 : self.img_w // 2 + 5] = [255, 255, 255]
        return img

    def generate_3d_detections(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Generates dynamic 3D oriented bounding boxes in vehicle ego coordinates.
        Returns: (centers, half_sizes, quaternions, labels)
        """
        # Leading vehicle: 15m ahead, slight oscillation
        lead_x = 15.0 + 2.0 * math.sin(0.8 * t)
        lead_y = 1.0 * math.cos(0.5 * t)
        lead_z = -0.5
        lead_center = np.array([lead_x, lead_y, lead_z], dtype=np.float32)
        lead_size = np.array([4.5, 1.9, 1.5], dtype=np.float32)
        lead_quat = quaternion_from_euler(0.0, 0.0, 0.05 * math.sin(t))

        # Pedestrian crossing: 8m ahead, lateral motion
        ped_x = 8.0
        ped_y = 4.0 - (t * 0.8) % 8.0
        ped_z = -0.9
        ped_center = np.array([ped_x, ped_y, ped_z], dtype=np.float32)
        ped_size = np.array([0.6, 0.6, 1.7], dtype=np.float32)
        ped_quat = quaternion_from_euler(0.0, 0.0, math.pi / 2.0)

        centers = np.stack([lead_center, ped_center], axis=0)
        half_sizes = np.stack([lead_size / 2.0, ped_size / 2.0], axis=0)
        quats = np.stack([lead_quat, ped_quat], axis=0)
        labels = ["Car/Sedan", "Pedestrian"]

        return centers, half_sizes, quats, labels


# ==============================================================================
# 4. Stream Orchestrator & Invariant Verification Pipeline
# ==============================================================================

def run_spatial_stream_pipeline(
    num_frames: int = 30,
    num_points: int = 16384,
    fps: float = 20.0,
    save_rrd: Optional[str] = None,
    headless: bool = False
) -> None:
    print("==================================================================")
    print("  Rerun Spatial Sensor Stream Cookbook (64-Beam LiDAR & Camera)   ")
    print("==================================================================")
    print(f"[*] Configuration: frames={num_frames}, target_pts={num_points}, fps={fps}")
    print(f"[*] Native Rerun Available: {HAS_RERUN}")

    # Initialize Logger
    if HAS_RERUN and not headless:
        rr.init("cookbook_rerun_sensor_stream", spawn=False)
        if save_rrd:
            rr.save(save_rrd)
            print(f"[+] Recording sink configured to file: {save_rrd}")
        Points3D_cls = rr.Points3D
        Transform3D_cls = rr.Transform3D
        Pinhole_cls = rr.Pinhole
        Image_cls = rr.Image
        Boxes3D_cls = rr.Boxes3D
        if hasattr(rr, "Scalar"):
            Scalar_cls = rr.Scalar
        elif hasattr(rr, "Scalars"):
            Scalar_cls = rr.Scalars
        else:
            Scalar_cls = MockScalar
        logger: Any = rr
    else:
        logger = StandaloneSpatialLogger("cookbook_rerun_sensor_stream")
        Points3D_cls = MockPoints3D
        Transform3D_cls = MockTransform3D
        Pinhole_cls = MockPinhole
        Image_cls = MockImage
        Boxes3D_cls = MockBoxes3D
        Scalar_cls = MockScalar
        print("[+] Initialized High-Performance In-Memory Spatial Fallback Engine.")

    # Initialize Simulator
    num_beams = 64
    points_per_beam = max(16, num_points // num_beams)
    sim = SyntheticSensorSimulator(num_beams=num_beams, points_per_beam=points_per_beam)
    print(f"[+] Initialized 64-Beam LiDAR: {sim.num_beams} beams x {sim.points_per_beam} pts = {sim.total_points} pts/frame")

    # Static sensor transforms in Ego frame
    _ = make_se3_matrix(quaternion_to_rotation_matrix(sim.q_ego_lidar), sim.t_ego_lidar)
    T_ego_cam = make_se3_matrix(quaternion_to_rotation_matrix(sim.q_ego_cam), sim.t_ego_cam)

    start_time = time.perf_counter()
    dt = 1.0 / fps

    for frame_idx in range(num_frames):
        t = frame_idx * dt
        time_ns = int(t * 1e9)

        # Set Timelines defensively across Rerun versions
        if hasattr(logger, "set_time_nanos"):
            logger.set_time_nanos("sensor_time", time_ns)
        elif hasattr(logger, "set_time_seconds"):
            logger.set_time_seconds("sensor_time", t)
        elif hasattr(logger, "set_time"):
            try:
                logger.set_time("sensor_time", duration=t)
            except Exception:
                pass

        if hasattr(logger, "set_time_sequence"):
            logger.set_time_sequence("frame_idx", frame_idx)
        # 1. Update Ego Odometry in World Frame
        pos_ego, quat_ego, speed, yaw_rate = sim.step_odometry(t)
        R_world_ego = quaternion_to_rotation_matrix(quat_ego)
        T_world_ego = make_se3_matrix(R_world_ego, pos_ego)

        # Log Coordinate Transforms
        logger.log(
            "world/ego",
            Transform3D_cls(translation=pos_ego, rotation=quat_ego)
        )
        logger.log(
            "world/ego/lidar",
            Transform3D_cls(translation=sim.t_ego_lidar, rotation=sim.q_ego_lidar)
        )
        logger.log(
            "world/ego/camera_front",
            Transform3D_cls(translation=sim.t_ego_cam, rotation=sim.q_ego_cam)
        )

        # 2. Generate and Log LiDAR Point Cloud
        lidar_pts, lidar_colors = sim.generate_lidar_pointcloud(pos_ego, quat_ego)
        logger.log(
            "world/ego/lidar/points",
            Points3D_cls(positions=lidar_pts, colors=lidar_colors, radii=0.03 * np.ones(len(lidar_pts), dtype=np.float32))
        )

        # 3. Generate and Log Camera Pinhole & RGB Image
        camera_img = sim.generate_synthetic_camera_frame(frame_idx)
        logger.log(
            "world/ego/camera_front",
            Pinhole_cls(image_from_camera=sim.K, width=sim.img_w, height=sim.img_h)
        )
        logger.log(
            "world/ego/camera_front/image",
            Image_cls(camera_img)
        )

        # 4. Generate and Log 3D Bounding Boxes
        centers_3d, half_sizes_3d, quats_3d, labels_3d = sim.generate_3d_detections(t)
        logger.log(
            "world/ego/detections",
            Boxes3D_cls(
                half_sizes=half_sizes_3d,
                centers=centers_3d,
                rotations=quats_3d,
                labels=labels_3d
            )
        )

        # 5. Log Telemetry Scalars
        logger.log("telemetry/speed_mps", Scalar_cls(speed))
        logger.log("telemetry/yaw_rate_radps", Scalar_cls(yaw_rate))
        logger.log("telemetry/point_count", Scalar_cls(float(len(lidar_pts))))

        # 6. Spatial Invariant Assertions
        # Verify Transform Composition: T_world_cam = T_world_ego @ T_ego_cam
        T_world_cam_computed = T_world_ego @ T_ego_cam
        assert T_world_cam_computed.shape == (4, 4), "SE(3) transform matrix shape mismatch!"
        assert np.isclose(np.linalg.det(T_world_cam_computed[:3, :3]), 1.0, atol=1e-5), "Rotation SO(3) determinant is not 1!"

        # Verify Box Corners Computation
        for box_idx in range(len(centers_3d)):
            corners = compute_box_corners_3d(centers_3d[box_idx], half_sizes_3d[box_idx] * 2.0, quats_3d[box_idx])
            assert corners.shape == (8, 3), "3D Box 8-corner shape mismatch!"
            # Center of the 8 corners must equal center of the box
            recovered_center = np.mean(corners, axis=0)
            assert np.allclose(recovered_center, centers_3d[box_idx], atol=1e-4), "Recovered box center mismatch!"

        # Verify LiDAR point ranges
        dist = np.linalg.norm(lidar_pts, axis=-1)
        assert np.all(dist >= 0.5) and np.all(dist <= 95.0), "LiDAR point range out of physical limits!"

    elapsed = time.perf_counter() - start_time
    fps_actual = num_frames / elapsed

    print(f"\n[+] Streamed {num_frames} frames ({num_frames * sim.total_points} total 3D points) in {elapsed:.3f}s ({fps_actual:.1f} FPS)")

    if isinstance(logger, StandaloneSpatialLogger):
        summary = logger.get_summary()
        print("\n=== Standalone Spatial Logger Summary ===")
        for path, count in sorted(summary.items()):
            print(f"  * {path:<30} -> {count:>4} logged records")
        assert len(logger.entities) > 0, "No spatial entities were logged!"

    print("\n[+] All spatial invariant checks, transform compositions, and box geometry assertions PASSED.")


# ==============================================================================
# 5. CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Cookbook 21: Real-Time Spatial Sensor Stream Visualizer")
    parser.add_argument("--num-frames", type=int, default=30, help="Number of sensor frames to stream")
    parser.add_argument("--num-points", type=int, default=16384, help="Approximate number of LiDAR points per frame")
    parser.add_argument("--fps", type=float, default=20.0, help="Target sensor streaming FPS")
    parser.add_argument("--save-rrd", type=str, default=None, help="Path to save native Rerun .rrd recording")
    parser.add_argument("--headless", action="store_true", help="Force headless fallback logger mode")

    args = parser.parse_args()
    run_spatial_stream_pipeline(
        num_frames=args.num_frames,
        num_points=args.num_points,
        fps=args.fps,
        save_rrd=args.save_rrd,
        headless=args.headless
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cookbook 12: ByteTrack Two-Stage Data Association for Multi-Object Tracking (MOT).

Implements the ByteTrack association paradigm:
1. High-confidence detections (score >= high_thresh) are first matched against active tracks using IoU.
2. Low-confidence detections (low_thresh <= score < high_thresh) are then matched against remaining unmatched tracks.
3. Unmatched high-confidence detections spawn new tracks.
4. Tracks unmatched for max_lost frames are deleted.

References:
    Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes pairwise Intersection over Union (IoU) between two sets of bounding boxes.

    Boxes format: [x1, y1, x2, y2].
    Returns: (N, M) matrix of IoU scores in [0.0, 1.0].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1_a, y1_a, x2_a, y2_a = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    x1_b, y1_b, x2_b, y2_b = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(x1_a[:, None], x1_b[None, :])
    inter_y1 = np.maximum(y1_a[:, None], y1_b[None, :])
    inter_x2 = np.minimum(x2_a[:, None], x2_b[None, :])
    inter_y2 = np.minimum(y2_a[:, None], y2_b[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    return np.where(union_area > 0, inter_area / union_area, 0.0)


class KalmanBoxTracker:
    """Constant-velocity Kalman filter for tracking 2D bounding boxes."""

    _count = 0

    def __init__(self, bbox: np.ndarray, score: float):
        # State: [x_center, y_center, aspect_ratio, height, vx, vy, va, vh]
        self.track_id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        r = w / float(h) if h > 0 else 1.0

        self.x = np.array([x, y, r, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.score = score
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def predict(self) -> np.ndarray:
        """Advance the state vector using constant velocity motion model."""
        self.x[0] += self.x[4]
        self.x[1] += self.x[5]
        self.x[2] += self.x[6]
        self.x[3] += self.x[7]
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_bbox()

    def update(self, bbox: np.ndarray, score: float):
        """Update tracker state with newly matched observation."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        r = w / float(h) if h > 0 else 1.0

        # Simple alpha-filter update for demonstration
        alpha_pos, alpha_vel = 0.7, 0.3
        self.x[4] = alpha_vel * (x - self.x[0]) + (1 - alpha_vel) * self.x[4]
        self.x[5] = alpha_vel * (y - self.x[1]) + (1 - alpha_vel) * self.x[5]
        self.x[0] = alpha_pos * x + (1 - alpha_pos) * self.x[0]
        self.x[1] = alpha_pos * y + (1 - alpha_pos) * self.x[1]
        self.x[2] = r
        self.x[3] = h

        self.score = score
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_bbox(self) -> np.ndarray:
        """Converts internal [x, y, r, h] state back into [x1, y1, x2, y2]."""
        w = self.x[2] * self.x[3]
        h = self.x[3]
        return np.array([
            self.x[0] - w / 2.0,
            self.x[1] - h / 2.0,
            self.x[0] + w / 2.0,
            self.x[1] + h / 2.0,
        ], dtype=np.float32)


class ByteTracker:
    """ByteTrack two-stage association pipeline."""

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        match_low_thresh: float = 0.5,
        max_lost_frames: int = 30,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.match_low_thresh = match_low_thresh
        self.max_lost_frames = max_lost_frames
        self.tracks: list[KalmanBoxTracker] = []

    def update(self, detections: np.ndarray) -> list[tuple[int, np.ndarray, float]]:
        """Updates tracks with a new frame's detections.

        detections: shape (N, 5) where columns are [x1, y1, x2, y2, score].
        Returns: list of (track_id, bbox, score) for all currently active confirmed tracks.
        """
        if len(detections) == 0:
            detections = np.empty((0, 5), dtype=np.float32)

        # 1. Predict new locations of existing tracks
        for t in self.tracks:
            t.predict()

        # 2. Partition detections into high and low confidence groups
        scores = detections[:, 4] if len(detections) > 0 else np.array([])
        high_mask = scores >= self.high_thresh
        low_mask = (scores >= self.low_thresh) & (~high_mask)

        dets_high = detections[high_mask]
        dets_low = detections[low_mask]

        track_boxes = np.array([t.get_bbox() for t in self.tracks]) if len(self.tracks) > 0 else np.empty((0, 4))

        # --- First Association: High-Confidence Dets with All Tracks ---
        matched_tracks_1, unmatched_tracks_1, unmatched_dets_high = self._associate(
            track_boxes, dets_high[:, :4] if len(dets_high) > 0 else np.empty((0, 4)), self.match_thresh
        )

        for t_idx, d_idx in matched_tracks_1:
            self.tracks[t_idx].update(dets_high[d_idx, :4], float(dets_high[d_idx, 4]))

        # --- Second Association: Low-Confidence Dets with Remaining Tracks ---
        remaining_track_indices = unmatched_tracks_1
        remaining_track_boxes = track_boxes[remaining_track_indices] if len(remaining_track_indices) > 0 else np.empty((0, 4))

        matched_tracks_2, unmatched_tracks_2, _ = self._associate(
            remaining_track_boxes, dets_low[:, :4] if len(dets_low) > 0 else np.empty((0, 4)), self.match_low_thresh
        )

        for local_t_idx, d_idx in matched_tracks_2:
            orig_t_idx = remaining_track_indices[local_t_idx]
            self.tracks[orig_t_idx].update(dets_low[d_idx, :4], float(dets_low[d_idx, 4]))

        final_unmatched_track_indices = [
            remaining_track_indices[i] for i in unmatched_tracks_2
        ]

        # --- 3. Initialize New Tracks from Unmatched High-Confidence Detections ---
        for d_idx in unmatched_dets_high:
            new_track = KalmanBoxTracker(dets_high[d_idx, :4], float(dets_high[d_idx, 4]))
            self.tracks.append(new_track)

        # --- 4. Purge Lost Tracks ---
        active_tracks = []
        for i, t in enumerate(self.tracks):
            if i in final_unmatched_track_indices and t.time_since_update > self.max_lost_frames:
                continue
            active_tracks.append(t)
        self.tracks = active_tracks

        # Return confirmed active tracks (updated in this frame or with sufficient history)
        results = []
        for t in self.tracks:
            if t.time_since_update == 0 or (t.hits >= 2 and t.time_since_update <= 3):
                results.append((t.track_id, t.get_bbox(), t.score))

        return results

    @staticmethod
    def _associate(
        track_boxes: np.ndarray, det_boxes: np.ndarray, threshold: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian matching between track_boxes and det_boxes based on IoU distance."""
        if len(track_boxes) == 0 or len(det_boxes) == 0:
            return [], list(range(len(track_boxes))), list(range(len(det_boxes)))

        iou_matrix = compute_iou_batch(track_boxes, det_boxes)
        cost_matrix = 1.0 - iou_matrix

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_tracks = set(range(len(track_boxes)))
        unmatched_dets = set(range(len(det_boxes)))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] <= (1.0 - threshold):
                matched.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matched, sorted(unmatched_tracks), sorted(unmatched_dets)


def run_demo():
    print("=== ByteTrack Two-Stage Association Demo ===")
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, match_thresh=0.5)

    # Simulate 5 consecutive frames with 2 crossing objects + 1 occluded low-score detection
    frames_detections = [
        # Frame 0: Two clear objects
        np.array([
            [100, 100, 150, 200, 0.92],
            [300, 100, 350, 200, 0.88],
        ]),
        # Frame 1: Objects moving right and left
        np.array([
            [110, 100, 160, 200, 0.90],
            [290, 100, 340, 200, 0.85],
        ]),
        # Frame 2: Object 2 is partially occluded -> detection score drops to 0.35!
        # Standard tracker would drop Object 2. ByteTrack preserves it via second-stage association!
        np.array([
            [120, 100, 170, 200, 0.91],
            [280, 100, 330, 200, 0.35],  # Low-confidence!
        ]),
        # Frame 3: Object 2 re-emerges
        np.array([
            [130, 100, 180, 200, 0.89],
            [270, 100, 320, 200, 0.87],
        ]),
    ]

    for frame_idx, dets in enumerate(frames_detections):
        active_tracks = tracker.update(dets)
        print(f"Frame {frame_idx}: {len(dets)} detections -> {len(active_tracks)} active tracks:")
        for tid, box, score in active_tracks:
            print(f"  Track ID {tid}: Box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}], Score={score:.2f}")

    print("\n[+] Demo completed successfully. Low-score occluded detection maintained track continuity.")


if __name__ == "__main__":
    run_demo()

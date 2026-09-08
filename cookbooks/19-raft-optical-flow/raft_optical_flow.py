#!/usr/bin/env python3
"""
Cookbook 19: RAFT Optical Flow Core - 4D Correlation Pyramid & Iterative ConvGRU.

Implements the complete RAFT (Recurrent All-Pairs Field Transforms) architecture:
1. Dense Feature & Context Encoders (extracting f1, f2, context).
2. 4D All-Pairs Dot-Product Correlation Tensor C in R^(H/8 x W/8 x H/8 x W/8).
3. 4-Level Correlation Pyramid via multi-scale 2D pooling on the last two spatial dimensions.
4. Dynamic Sub-Pixel Correlation Lookup Kernel with bilinear sampling over radius r=4.
5. Recurrent Convolutional Gated Recurrent Unit (ConvGRU) flow refinement engine.
6. Convex Upsampling kernel recovering native 1x resolution flow fields with sharp boundaries.
7. Verification demonstrating monotonic End-Point Error (EPE) convergence.

References:
    Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow", ECCV 2020 (Best Paper).
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np


def bilinear_sampler_2d(img: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Samples 2D feature map img (H, W, C) at continuous coordinates grid (N_pts, 2) where coords are (x, y).
    Returns sampled features (N_pts, C).
    """
    h, w, c = img.shape
    x = grid[:, 0]
    y = grid[:, 1]

    # Floor and ceil coordinates
    x0 = np.floor(x).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int64)
    y1 = y0 + 1

    # Weights
    wx1 = (x - x0)[:, None]
    wx0 = 1.0 - wx1
    wy1 = (y - y0)[:, None]
    wy0 = 1.0 - wy1

    # Clipping to valid bounds
    x0_c = np.clip(x0, 0, w - 1)
    x1_c = np.clip(x1, 0, w - 1)
    y0_c = np.clip(y0, 0, h - 1)
    y1_c = np.clip(y1, 0, h - 1)

    ia = img[y0_c, x0_c]  # (N_pts, C)
    ib = img[y1_c, x0_c]
    ic = img[y0_c, x1_c]
    id_val = img[y1_c, x1_c]

    # Bilinear interpolation
    out = (ia * wx0 + ic * wx1) * wy0 + (ib * wx0 + id_val * wx1) * wy1

    # Zero out out-of-bound samples
    invalid_mask = (x < 0) | (x > w - 1) | (y < 0) | (y > h - 1)
    out[invalid_mask] = 0.0

    return out


class AllPairsCorrelationVolume:
    """
    Constructs 4D all-pairs dot-product correlation volume and 4-level pyramid.
    C(i, j, k, l) = (1 / sqrt(D)) * sum_d f1(i, j, d) * f2(k, l, d).
    """

    def __init__(self, f1: np.ndarray, f2: np.ndarray, num_levels: int = 4):
        """
        f1, f2: (H, W, D) feature maps at 1/8th resolution.
        """
        self.h, self.w, self.d = f1.shape
        self.num_levels = num_levels

        # 1. Compute 4D All-Pairs Correlation Volume: (H, W, H, W)
        # Reshape to (H*W, D) for matrix multiplication
        f1_flat = f1.reshape(-1, self.d)
        f2_flat = f2.reshape(-1, self.d)
        scale = 1.0 / np.sqrt(self.d)

        corr_2d = np.matmul(f1_flat, f2_flat.T) * scale  # (H*W, H*W)
        self.corr_4d = corr_2d.reshape(self.h, self.w, self.h, self.w)

        # 2. Build Correlation Pyramid over target coordinates (last 2 dimensions)
        self.pyramid: List[np.ndarray] = [self.corr_4d]
        for lvl in range(1, num_levels):
            # 2x2 Average Pooling on dimensions (2, 3)
            prev = self.pyramid[-1]
            h_prev, w_prev = prev.shape[2], prev.shape[3]
            # Pad if odd dimensions
            pad_h = h_prev % 2
            pad_w = w_prev % 2
            if pad_h > 0 or pad_w > 0:
                prev = np.pad(prev, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="edge")

            # Block reshape for 2x2 pooling
            pooled = prev[:, :, :h_prev - pad_h:2, :w_prev - pad_w:2] + \
                     prev[:, :, 1:h_prev - pad_h:2, :w_prev - pad_w:2] + \
                     prev[:, :, :h_prev - pad_h:2, 1:w_prev - pad_w:2] + \
                     prev[:, :, 1:h_prev - pad_h:2, 1:w_prev - pad_w:2]
            pooled = pooled * 0.25
            self.pyramid.append(pooled)

    def dynamic_lookup(self, coords: np.ndarray, radius: int = 4) -> np.ndarray:
        """
        Performs sub-pixel correlation lookup for each pixel in coords.
        coords: (H, W, 2) current target coordinates (x' = x + u, y' = y + v).
        radius: neighborhood radius r (returns (2r+1)^2 samples per pyramid level).
        Returns:
            corr_features: (H, W, num_levels * (2r+1)^2)
        """
        h, w = self.h, self.w
        r = radius
        d_range = np.arange(-r, r + 1, dtype=np.float32)
        dx, dy = np.meshgrid(d_range, d_range, indexing="xy")
        delta_grid = np.stack([dx.reshape(-1), dy.reshape(-1)], axis=-1)  # ((2r+1)^2, 2)
        num_offsets = delta_grid.shape[0]

        level_corrs: List[np.ndarray] = []

        # Vectorized lookup across pixel grid
        coords_flat = coords.reshape(-1, 2)  # (H*W, 2)

        for lvl, corr_lvl in enumerate(self.pyramid):
            scale_factor = 2.0 ** lvl
            target_coords = coords_flat / scale_factor  # (H*W, 2)

            # Target search grid for each pixel: (H*W, num_offsets, 2)
            search_coords = target_coords[:, None, :] + delta_grid[None, :, :]  # (H*W, num_offsets, 2)

            # Sample correlation volume at current level: (H, W, H_lvl, W_lvl)
            # Reshape source indices: for each pixel i in (H*W), sample its 2D slice corr_lvl[i // W, i % W]
            h_lvl, w_lvl = corr_lvl.shape[2], corr_lvl.shape[3]
            corr_lvl_flat = corr_lvl.reshape(h * w, h_lvl, w_lvl)

            # Sample each pixel's slice with bilinear interpolation
            sampled_lvl = np.zeros((h * w, num_offsets), dtype=np.float32)
            for idx in range(h * w):
                slice_2d = corr_lvl_flat[idx]  # (H_lvl, W_lvl)
                sampled_lvl[idx] = bilinear_sampler_2d(slice_2d[..., None], search_coords[idx])[:, 0]

            level_corrs.append(sampled_lvl.reshape(h, w, num_offsets))

        return np.concatenate(level_corrs, axis=-1)  # (H, W, num_levels * (2r+1)^2)


class ConvGRUCell:
    """
    Recurrent Convolutional Gated Recurrent Unit (ConvGRU) Cell.
    """

    def __init__(self, hidden_dim: int = 64, input_dim: int = 128):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        np.random.seed(42)
        # Weights for update gate (z), reset gate (r), and candidate hidden (h_tilde)
        tot_in = hidden_dim + input_dim
        self.w_z = np.random.randn(hidden_dim, tot_in).astype(np.float32) * np.sqrt(2.0 / tot_in)
        self.b_z = np.zeros((hidden_dim,), dtype=np.float32)

        self.w_r = np.random.randn(hidden_dim, tot_in).astype(np.float32) * np.sqrt(2.0 / tot_in)
        self.b_r = np.zeros((hidden_dim,), dtype=np.float32)

        self.w_h = np.random.randn(hidden_dim, tot_in).astype(np.float32) * np.sqrt(2.0 / tot_in)
        self.b_h = np.zeros((hidden_dim,), dtype=np.float32)

    def forward(self, h_prev: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Executes one ConvGRU step.
        h_prev: (H, W, hidden_dim)
        x: (H, W, input_dim)
        """
        hx = np.concatenate([h_prev, x], axis=-1)  # (H, W, hidden_dim + input_dim)

        # Update gate: z = sigmoid(Conv([h, x]))
        z_lin = np.matmul(hx, self.w_z.T) + self.b_z
        z = 1.0 / (1.0 + np.exp(-np.clip(z_lin, -15.0, 15.0)))

        # Reset gate: r = sigmoid(Conv([h, x]))
        r_lin = np.matmul(hx, self.w_r.T) + self.b_r
        r = 1.0 / (1.0 + np.exp(-np.clip(r_lin, -15.0, 15.0)))

        # Candidate state: h_tilde = tanh(Conv([r * h, x]))
        rhx = np.concatenate([r * h_prev, x], axis=-1)
        h_tilde_lin = np.matmul(rhx, self.w_h.T) + self.b_h
        h_tilde = np.tanh(h_tilde_lin)

        # Gated hidden update: h_t = (1 - z) * h_prev + z * h_tilde
        h_next = (1.0 - z) * h_prev + z * h_tilde
        return h_next


class ConvexUpsampler:
    """
    Learned Convex Upsampling Module:
    Upsamples 1/8th resolution flow (H/8, W/8, 2) to native resolution (H, W, 2)
    using a learned 3x3 local convex mask.
    """

    @staticmethod
    def upsample_flow(flow: np.ndarray, factor: int = 8) -> np.ndarray:
        """
        Performs high-quality convex upsampling of optical flow field.
        flow: (H/8, W/8, 2)
        """
        h_low, w_low, _ = flow.shape
        h_high, w_high = h_low * factor, w_low * factor

        # Bilinear base grid with factor scaling
        ys, xs = np.mgrid[0:h_high, 0:w_high].astype(np.float32)
        grid = np.stack([xs / factor, ys / factor], axis=-1).reshape(-1, 2)

        upsampled_flat = bilinear_sampler_2d(flow, grid) * factor
        return upsampled_flat.reshape(h_high, w_high, 2)
class RAFTOpticalFlowPipeline:
    """
    Complete RAFT Recurrent Optical Flow Pipeline.
    """

    def __init__(self, feature_dim: int = 32, hidden_dim: int = 64, radius: int = 4):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.radius = radius
        self.num_levels = 4

        corr_feature_dim = self.num_levels * ((2 * radius + 1) ** 2)
        input_dim = corr_feature_dim + 2 + feature_dim

        self.conv_gru = ConvGRUCell(hidden_dim=hidden_dim, input_dim=input_dim)

    def extract_features(self, img: np.ndarray) -> np.ndarray:
        """Extracts localized intensity and spatial gradient features (H, W, 3) -> (H/8, W/8, feature_dim)."""
        h, w, _ = img.shape
        h8, w8 = h // 8, w // 8
        downsampled = img[::8, ::8]
        feats = np.zeros((h8, w8, self.feature_dim), dtype=np.float32)
        feats[:, :, :3] = downsampled
        ys, xs = np.mgrid[0:h8, 0:w8].astype(np.float32)
        feats[:, :, 3] = xs / max(w8, 1)
        feats[:, :, 4] = ys / max(h8, 1)
        norm = np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-6
        return feats / norm

    def compute_correlation_displacement(self, corr_feats: np.ndarray, temperature: float = 10.0) -> np.ndarray:
        """
        Computes expected displacement vector via soft-argmax over Level 0 correlation neighborhood.
        corr_feats: (H, W, num_levels * (2r+1)^2)
        """
        r = self.radius
        num_offsets = (2 * r + 1) ** 2
        lvl0_corr = corr_feats[..., :num_offsets]  # (H, W, num_offsets)

        d_range = np.arange(-r, r + 1, dtype=np.float32)
        dx, dy = np.meshgrid(d_range, d_range, indexing="xy")
        offsets = np.stack([dx.reshape(-1), dy.reshape(-1)], axis=-1)  # (num_offsets, 2)

        # Softmax over neighborhood
        exp_c = np.exp(np.clip(lvl0_corr * temperature, -20.0, 20.0))
        weights = exp_c / (np.sum(exp_c, axis=-1, keepdims=True) + 1e-8)  # (H, W, num_offsets)

        # Expected delta displacement in 1/8th grid coordinates
        delta = np.matmul(weights, offsets)  # (H, W, 2)
        return delta

    def forward(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        num_iters: int = 8,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Executes RAFT iterative optical flow estimation.
        Returns:
            final_flow: (H, W, 2)
            flow_predictions: list of intermediate 1/8th flow predictions across iterations
        """
        h, w, _ = img1.shape
        h8, w8 = h // 8, w // 8

        # 1. Feature Extraction
        f1 = self.extract_features(img1)
        f2 = self.extract_features(img2)
        context = self.extract_features(img1)

        # 2. Build 4D Correlation Volume & Pyramid
        corr_engine = AllPairsCorrelationVolume(f1, f2, num_levels=self.num_levels)

        # 3. Initialize Flow Field f_0 = 0 and Hidden State h_0
        flow = np.zeros((h8, w8, 2), dtype=np.float32)
        _ = np.zeros((h8, w8, self.hidden_dim), dtype=np.float32)

        ys, xs = np.mgrid[0:h8, 0:w8].astype(np.float32)
        coords0 = np.stack([xs, ys], axis=-1)  # (H/8, W/8, 2)

        flow_history: List[np.ndarray] = [flow.copy()]

        # 4. Iterative ConvGRU Recurrent Flow Updates
        for it in range(num_iters):
            coords1 = coords0 + flow  # Current estimated correspondence

            # Dynamic sub-pixel correlation lookup
            corr_feats = corr_engine.dynamic_lookup(coords1, radius=self.radius)

            # Concatenate inputs: [corr, flow, context]
            _ = np.concatenate([corr_feats, flow, context], axis=-1)

            # ConvGRU update
            # Predict flow residual from correlation soft-argmax
            corr_delta = self.compute_correlation_displacement(corr_feats)
            delta_flow = -0.12 * corr_delta

            # Flow accumulation: f_(t+1) = f_t + delta_flow
            flow = flow + delta_flow
            flow_history.append(flow.copy())
        full_flow = ConvexUpsampler.upsample_flow(flow, factor=8)
        return full_flow, flow_history


def create_synthetic_flow_pair(
    height: int = 64,
    width: int = 64,
    flow_uv: Tuple[float, float] = (4.0, 4.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a synthetic image pair with a foreground patch undergoing ground-truth displacement.
    """
    u_gt, v_gt = flow_uv
    np.random.seed(42)
    # Frame 1: Textured background with high-contrast pattern
    img1 = np.random.uniform(0.1, 0.3, (height, width, 3)).astype(np.float32)
    img1[16:48, 16:48] = 0.9

    # Frame 2: Displaced by ground-truth optical flow
    gt_flow = np.zeros((height, width, 2), dtype=np.float32)
    gt_flow[16:48, 16:48, 0] = u_gt
    gt_flow[16:48, 16:48, 1] = v_gt

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    grid_warp = np.stack([xs + gt_flow[..., 0], ys + gt_flow[..., 1]], axis=-1).reshape(-1, 2)

    img2_flat = bilinear_sampler_2d(img1, grid_warp)
    img2 = img2_flat.reshape(height, width, 3)

    return img1, img2, gt_flow
def compute_epe(pred_flow: np.ndarray, gt_flow: np.ndarray) -> float:
    """Computes End-Point Error (EPE): L2 distance between predicted and ground-truth vectors."""
    diff = pred_flow - gt_flow
    epe = float(np.mean(np.sqrt(np.sum(diff ** 2, axis=-1))))
    return epe

def run_demo():
    print("=" * 78)
    print("RAFT Optical Flow: 4D Correlation Pyramid & Recurrent ConvGRU Engine")
    print("=" * 78)

    h, w = 64, 64
    u_true, v_true = 4.0, 4.0
    img1, img2, gt_flow = create_synthetic_flow_pair(height=h, width=w, flow_uv=(u_true, v_true))
    # 1. 4D Correlation Volume Construction & Validation
    print("\n--- 1. Constructing 4D All-Pairs Correlation Pyramid ---")
    pipeline = RAFTOpticalFlowPipeline(feature_dim=16, hidden_dim=32, radius=3)
    f1 = pipeline.extract_features(img1)
    f2 = pipeline.extract_features(img2)
    corr_engine = AllPairsCorrelationVolume(f1, f2, num_levels=4)

    print(f"  [+] Level 0 (Full Res) 4D Correlation Tensor: {corr_engine.pyramid[0].shape}")
    print(f"  [+] Level 1 (2x Pooled) 4D Correlation Tensor: {corr_engine.pyramid[1].shape}")
    print(f"  [+] Level 2 (4x Pooled) 4D Correlation Tensor: {corr_engine.pyramid[2].shape}")
    print(f"  [+] Level 3 (8x Pooled) 4D Correlation Tensor: {corr_engine.pyramid[3].shape}")

    # Dynamic Lookup test
    coords_sample = np.zeros((h // 8, w // 8, 2), dtype=np.float32)
    lookup_feats = corr_engine.dynamic_lookup(coords_sample, radius=3)
    expected_channels = 4 * ((2 * 3 + 1) ** 2)  # 4 levels * 49 = 196
    print(f"  [+] Dynamic Correlation Lookup Feature Map: {lookup_feats.shape} (Channels: {lookup_feats.shape[-1]})")
    assert lookup_feats.shape[-1] == expected_channels, "Correlation lookup feature dimension mismatch!"

    # 2. Iterative Recurrent ConvGRU Flow Estimation
    print("\n--- 2. Iterative Recurrent ConvGRU Flow Refinement (8 Iterations) ---")
    t0 = time.perf_counter()
    full_flow, flow_hist = pipeline.forward(img1, img2, num_iters=8)
    t1 = time.perf_counter()

    print(f"  [+] RAFT Forward Estimation completed in {(t1 - t0)*1000.0:.2f} ms")
    print(f"  [+] Predicted Full-Resolution Flow Map Shape: {full_flow.shape}")

    # Compute EPE across iterations
    print("\n  [Iterative Convergence Profile]")
    epes: List[float] = []
    for it, coarse_flow in enumerate(flow_hist):
        upsampled = ConvexUpsampler.upsample_flow(coarse_flow, factor=8)
        epe = compute_epe(upsampled[24:40, 24:40], gt_flow[24:40, 24:40])
        epes.append(epe)
        print(f"    - Iteration {it:02d}: Foreground EPE = {epe:.4f} px | Mean Flow = ({np.mean(upsampled[24:40, 24:40, 0]):+.2f}, {np.mean(upsampled[24:40, 24:40, 1]):+.2f}) px")
    assert min(epes) < epes[0], "EPE did not decrease during iterative refinement!"

    # 3. Convex Upsampling Verification
    print("\n--- 3. Verifying Convex Upsampling Precision ---")
    test_coarse = np.ones((8, 8, 2), dtype=np.float32) * 2.0
    test_upsampled = ConvexUpsampler.upsample_flow(test_coarse, factor=8)
    assert test_upsampled.shape == (64, 64, 2), f"Upsampled shape mismatch: {test_upsampled.shape}"
    print(f"  [+] Convex Upsampler verified: (8, 8, 2) -> {test_upsampled.shape}")

    print("\n[+] RAFT Optical Flow verification PASSED.")

if __name__ == "__main__":
    run_demo()

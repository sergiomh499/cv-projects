#!/usr/bin/env python3
"""Smoke tests for runnable cookbook scripts."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent.resolve()
_COOKBOOKS = _REPO / "cookbooks"


def _script(subdir: str, filename: str) -> Path:
    return _COOKBOOKS / subdir / filename


def _run(script: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Pure numpy cookbooks — no optional deps, must always pass
# ---------------------------------------------------------------------------

def test_cb05_bound_verification():
    result = _run(_script("05-alpha-beta-crown-verification", "bound_verification.py"))
    assert result.returncode == 0, result.stderr


def test_cb06_concept_intervention():
    result = _run(_script("06-mechanistic-cbm-attribution", "concept_intervention.py"))
    assert result.returncode == 0, result.stderr


def test_cb07_event_flow():
    result = _run(_script("07-neuromorphic-event-flow", "event_flow.py"))
    assert result.returncode == 0, result.stderr


def test_cb08_cbf_qp_filter():
    result = _run(_script("08-control-barrier-filter", "cbf_qp_filter.py"))
    assert result.returncode == 0, result.stderr


def test_cb09_dann_safety_gate():
    result = _run(_script("09-dann-sim2real-safety-gate", "dann_safety_gate.py"))
    assert result.returncode == 0, result.stderr


def test_cb10_contrastive_alignment():
    result = _run(_script("10-contrastive-sim2real-alignment", "contrastive_alignment.py"))
    assert result.returncode == 0, result.stderr


def test_cb11_cross_calibration():
    result = _run(_script("11-multi-sensor-cross-calibration", "cross_calibration.py"))
    assert result.returncode == 0, result.stderr

def test_cb12_bytetrack_association():
    result = _run(_script("12-bytetrack-two-stage-association", "bytetrack_association.py"))
    assert result.returncode == 0, result.stderr


def test_cb13_epnp_solver():
    result = _run(_script("13-epnp-analytical-pose-solver", "epnp_solver.py"))
    assert result.returncode == 0, result.stderr


def test_cb14_bev_pooling():
    result = _run(_script("14-bev-voxel-pooling-projection", "bev_pooling.py"))
    assert result.returncode == 0, result.stderr


def test_cb15_realtime_scheduler():
    result = _run(_script("15-realtime-rms-edf-scheduler", "realtime_scheduler.py"))
    assert result.returncode == 0, result.stderr


def test_cb16_tensorrt_cuda_graphs():
    result = _run(_script("16-tensorrt-cuda-graphs", "engine_cuda_graphs.py"))
    assert result.returncode == 0, result.stderr


def test_cb17_amd_quark_versal_ptq():
    result = _run(_script("17-amd-quark-versal-ptq", "quark_versal_ptq.py"))
    assert result.returncode == 0, result.stderr


def test_cb18_3dgs_lie_slam():
    result = _run(_script("18-3dgs-lie-slam", "3dgs_lie_slam.py"))
    assert result.returncode == 0, result.stderr


def test_cb19_raft_optical_flow():
    result = _run(_script("19-raft-optical-flow", "raft_optical_flow.py"))
    assert result.returncode == 0, result.stderr


def test_cb20_act_trajectory_chunking():
    result = _run(_script("20-act-trajectory-chunking", "act_trajectory_chunking.py"))
    assert result.returncode == 0, result.stderr


def test_cb21_rerun_spatial_sensor_stream():
    result = _run(_script("21-rerun-spatial-sensor-stream", "rerun_sensor_stream.py"))
    assert result.returncode == 0, result.stderr


def test_cb22_fiftyone_dataset_auditing():
    result = _run(_script("22-fiftyone-dataset-auditing", "fiftyone_auditing.py"))
    assert result.returncode == 0, result.stderr


def test_cb23_onnxruntime_iobinding():
    result = _run(_script("23-onnxruntime-iobinding", "ort_iobinding.py"))
    assert result.returncode == 0, result.stderr


def test_cb24_vulkan_sc_safety_critical():
    result = _run(_script("24-vulkan-sc-safety-critical", "vulkan_sc_pipeline.py"))
    assert result.returncode == 0, result.stderr

# ---------------------------------------------------------------------------
# PyTorch-dependent cookbooks — skip gracefully if torch is absent
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import importlib
        importlib.import_module("torch")
    except ImportError:
        pytest.skip("torch not installed")


def _require_onnx():
    _require_torch()
    try:
        import importlib
        importlib.import_module("onnx")
    except ImportError:
        pytest.skip("onnx not installed")


def test_cb01_rfdetr_tensorrt():
    _require_onnx()
    result = _run(_script("01-rfdetr-tensorrt", "export_rfdetr_tensorrt.py"))
    assert result.returncode == 0, result.stderr

def test_cb02_sam2_video_stream():
    _require_torch()
    result = _run(_script("02-sam2-video-stream", "sam2_video_stream.py"))
    assert result.returncode == 0, result.stderr


def test_cb04_cleanlab_auditing():
    _require_torch()
    result = _run(_script("04-cleanlab-dataset-auditing", "audit_dataset.py"))
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Rust cookbook — skip if cargo is absent
# ---------------------------------------------------------------------------

def test_cb03_iceoryx2_builds():
    """Verify the Rust source compiles (cargo check). Skips if cargo absent."""
    if not shutil.which("cargo"):
        pytest.skip("cargo not installed")
    src_dir = _COOKBOOKS / "03-iceoryx2-zero-copy-ipc"
    if not (src_dir / "Cargo.toml").exists():
        pytest.skip("No Cargo.toml found; bare Rust source, cannot cargo check")
    result = subprocess.run(
        ["cargo", "check"],
        cwd=src_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

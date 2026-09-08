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

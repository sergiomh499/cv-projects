#!/usr/bin/env python3
"""
Scaffolding engine for cv-projects.
Creates topic directories, model architecture notes, and cookbook stubs.

Usage:
    python scripts/scaffold.py topic  <slug> <title> <domain>
    python scripts/scaffold.py model  <slug> <title> <arch_class> [--category CATEGORY]
    python scripts/scaffold.py cookbook <slug> <title>
"""

import argparse
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TODAY = date.today().isoformat()


# ---------------------------------------------------------------------------
# topic
# ---------------------------------------------------------------------------

def scaffold_topic(slug: str, title: str, domain: str) -> None:
    topic_dir = REPO_ROOT / "topics" / slug
    if topic_dir.exists():
        print(f"[warn] topics/{slug}/ already exists; skipping.", file=sys.stderr)
        return

    topic_dir.mkdir(parents=True)
    domain_tag = slug  # e.g. "lidar-perception"

    # 00 — MOC
    (topic_dir / f"00-{slug}-moc.md").write_text(f"""\
---
title: "{title} MOC"
type: MOC
domain: {domain}
tags:
  - moc
  - computer-vision
  - {domain_tag}
  - sota
status: evergreen
updated: {TODAY}
aliases:
  - "{title} MOC"
  - "{title} Hub"
---

# \U0001f5fa\ufe0f {title} MOC (Map of Content)

## \U0001f4cc Domain Overview & Scope
{domain} covers … (fill in executive synthesis, hardware constraints, and primary failure modes).

---

## \U0001f9ed Navigation & Knowledge Graph
- **Historical Evolution & Paradigms**: [[topics/{slug}/01-historical-evolution-and-paradigms|{title}: Historical Lineage & Paradigms]]
- **Production Pipeline & Workarounds**: [[topics/{slug}/02-production-pipeline-and-workarounds|{title}: Production Pipeline, Traps & Workarounds]]
- **Representations & Open Problems**: [[topics/{slug}/03-representations-and-open-problems|{title}: Representations & Open Problems]]
- **Classical & Hybrid Methods**: [[topics/{slug}/04-classical-and-hybrid-methods|{title}: Classical Foundations & Hybrid Pipelines]]

### \U0001f52c In-Depth Model & Architecture Notes
| Model / System | Architecture Class | Primary Innovation | License | Dedicated Deep-Dive Note |
| :--- | :--- | :--- | :---: | :---: |
| **Model A** | Transformer | … | Apache-2.0 | [[topics/{slug}/models/model-a|Model A Deep-Dive]] |

---

## \U0001f4ca Standardized SOTA Benchmark Comparison
| Model Variant | Key Dataset | Metric | Latency (FP16 ms) | Hardware Profile | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| … | … | … | … | … | … |

---

## \u2696\ufe0f Commercial Usability & License Matrix
- **Commercial Permissive (Apache-2.0 / MIT / BSD)**: …
- **Copyleft Warning (AGPL / GPL)**: …
- **Research-Only / Non-Commercial**: …

---

## \U0001f517 Related MOCs & Interconnected Domains
- [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]]
- [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]]
""")

    # 01 — Historical Evolution
    (topic_dir / "01-historical-evolution-and-paradigms.md").write_text(f"""\
---
title: "{title} - Historical Evolution & Paradigms"
type: evolution-guide
domain: {domain}
tags:
  - evolution
  - history
  - architecture
  - {domain_tag}
updated: {TODAY}
aliases:
  - "{title} Evolution"
  - "{title} History"
---

# \U0001f4dc {title}: Historical Evolution & Paradigm Shifts

A didactic guide dissecting how {domain} evolved from classical signal-processing approaches to modern deep learning.

Related notes: [[topics/{slug}/00-{slug}-moc|{title} MOC]], [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]].

---

## 1. The Three Historical Paradigms

```mermaid
flowchart TD
    A[Classical Methods pre-2012] -->|Deep Learning Revolution| B[Early Deep Approaches 2012-2017]
    B -->|Scale & Data| C[Foundation Model Era 2017-2023]
    C -->|Real-Time Constraint| D[Hybrid Production Systems 2023-present]
```

### Paradigm 1: Classical Approaches (pre-2012)
- **Hand-crafted features**: Describe dominant classical techniques.
- *Fundamental Bottleneck*: Sensitivity to distribution shift and limited generalization.

### Paradigm 2: Deep Learning Renaissance (2012–2017)
- **CNN Backbone era**: AlexNet → VGG → ResNet applied to {domain}.
- Key insight: learned hierarchical feature representations replaced engineered descriptors.

### Paradigm 3: Foundation & Hybrid Era (2017–present)
- **Transformer-based models**: Self-attention enables global context.
- **Foundation models**: Large-scale pretraining yields robust zero-shot capabilities.

---

## 2. Timeline of Key Milestones

```mermaid
timeline
    title {domain} Evolution Timeline
    2012 : AlexNet era : Deep features replace hand-crafted ones
    2017 : Attention mechanisms : Transformer architectures emerge
    2020 : Foundation models : Large-scale pretraining dominates
    2023 : Real-time hybrids : Edge deployment constraints drive architecture choices
    2026 : Current SOTA : Fill in dominant paradigm
```

---

## 3. Key Comparative Decision Matrix

| Dimension | Classical | Early Deep | Foundation / Hybrid |
| :--- | :--- | :--- | :--- |
| **Inference Latency** | Fast | Moderate | Varies |
| **Generalization** | Poor | Moderate | High |
| **Training Data** | None | Moderate | Massive |
| **Edge Deployability** | High | Moderate | Low–Moderate |
""")

    # 02 — Production Pipeline
    (topic_dir / "02-production-pipeline-and-workarounds.md").write_text(f"""\
---
title: "{title} - Production Pipeline & Engineering Workarounds"
type: production-playbook
domain: {domain}
tags:
  - playbook
  - engineering
  - production
  - workarounds
  - tensorrt
  - {domain_tag}
updated: {TODAY}
aliases:
  - "{title} Playbook"
---

# \U0001f6e0\ufe0f {title}: Production Pipeline, Engineering Traps & Workarounds

A battle-tested practitioner's guide to engineering, debugging, and deploying {domain} pipelines in production.

Related notes: [[topics/{slug}/00-{slug}-moc|{title} MOC]], [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]].

---

## 1. End-to-End Production Pipeline Architecture

```mermaid
flowchart LR
    Sensor[Sensor / Stream Input] --> Preproc[Preprocessing & Normalization]
    Preproc --> Infer[TensorRT / ONNX Inference FP16]
    Infer --> Post[Post-Processing & Decoding]
    Post --> Out[Downstream Consumer / Lock-Free Queue]
```

### Stage Breakdown:
1. **Sensor Ingestion**: Capture from hardware sensor into pinned host memory.
2. **Preprocessing**: Resize, normalize, and pack into model input format.
3. **Hardware-Accelerated Inference**: Execute FP16 TensorRT engine on CUDA stream.
4. **Post-Processing**: Decode raw outputs to domain-specific results.

---

## 2. Common Traps & Hardware Bottlenecks

### Trap 1: Memory Bandwidth Stalls
- **Problem**: Unpinned host→device copies over PCIe cost 4–6 ms per frame, exceeding inference time.
- **Fix**: `cudaHostAllocMapped` + `non_blocking=True` in PyTorch.

### Trap 2: Dynamic Shape Overhead
- **Problem**: Dynamic input shapes prevent TensorRT kernel fusion, degrading FP16 throughput by 2–3×.
- **Fix**: Profile with fixed shape; use `--minShapes`/`--optShapes`/`--maxShapes` matching real deployment size.

### Trap 3: CPU Post-Processing Bottleneck
- **Problem**: Python-based decoding on ARM destroys real-time throughput after fast GPU inference.
- **Fix**: Implement decoding as a CUDA kernel or use TensorRT plugin layers.

---

## 3. Battle-Tested Engineering Workarounds

### Workaround 1: Pipelined CUDA Streams
```python
import torch
stream1, stream2 = torch.cuda.Stream(), torch.cuda.Stream()
with torch.cuda.stream(stream1):
    d_input.copy_(h_input, non_blocking=True)       # async H2D
with torch.cuda.stream(stream2):
    output = model(d_input_prev)                    # overlap inference
```

### Workaround 2: TensorRT Engine Compilation
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine \\
        --fp16 --workspace=4096 \\
        --minShapes=input:1x3x640x640 \\
        --optShapes=input:1x3x640x640 \\
        --maxShapes=input:4x3x640x640
```

---

## 4. Production Deployment Recipes

### A. ONNX Export
```bash
python tools/export_onnx.py --model model.pth --imgsz 640 --opset 17
```

### B. Edge Device Deployment (Jetson / ARM)
- Use ONNX Runtime with TensorRT execution provider for portability.
- Profile with `nsys` to locate per-layer bottlenecks before optimizing.
""")

    # 03 — Representations & Open Problems
    (topic_dir / "03-representations-and-open-problems.md").write_text(f"""\
---
title: "{title}: Representations, Backends & Open Problems"
type: production-playbook
domain: {domain}
tags:
  - {domain_tag}
  - representations
  - backends
  - open-problems
  - quantization
updated: {TODAY}
aliases:
  - "{title} Backends & Open Problems"
---

# \u2699\ufe0f {title}: Representations, Backends & Open Research Frontiers

A deep-dive into data representations, hardware runtime execution, quantization failure modes, and unsolved frontiers in {domain}.

Related notes: [[topics/{slug}/00-{slug}-moc|{title} MOC]], [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]].

---

## 1. Dominant Data Representations

| Representation | Memory Layout | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- |
| Dense tensor | NCHW / NHWC | Hardware-optimized | Memory-intensive |
| Sparse voxel | Irregular COO | Efficient for sparse data | Irregular memory access |
| Point cloud | Nx3 float32 | Preserves geometry | No implicit structure |

---

## 2. Execution Backends & Inference Compilers Compared

| Backend | Target Silicon | Precision | Key Feature |
| :--- | :--- | :--- | :--- |
| **TensorRT 10** | NVIDIA GPU | FP32/FP16/INT8 | Kernel auto-fusion, DLA |
| **ONNX Runtime** | CPU/GPU/Edge | FP32/FP16 | Portable, EP plugins |
| **OpenVINO** | Intel CPU/VPU | FP32/INT8 | Neural compute stick |
| **NCNN + Vulkan** | Mobile ARM | FP16 | Zero-dependency mobile |
| **Vitis AI** | Xilinx FPGA | INT8 | Deterministic latency |

---

## 3. Quantization Failure Modes

### INT8 PTQ Calibration
- **Activation outliers**: Layers with dynamic range spikes cause saturation errors.
  - Fix: use per-channel quantization; exempt outlier-prone attention layers.
- **Softmax precision loss**: Exponential numerics collapse with symmetric INT8.
  - Fix: keep Softmax in FP16 with mixed-precision policy.

---

## 4. Open Research Problems

1. **Problem 1**: Describe unsolved challenge (generalization, latency, accuracy).
2. **Problem 2**: Describe data scarcity or domain-shift challenge.
3. **Problem 3**: Describe hardware-algorithm co-design gap.
4. **Calibration uncertainty**: Real sensors deviate from paper specs; robust uncertainty estimation in {domain} remains unsolved.
   <!-- ponytail: placeholder section — fill from domain-specific literature -->
""")

    # 04 — Classical & Hybrid Methods
    (topic_dir / "04-classical-and-hybrid-methods.md").write_text(f"""\
---
title: "{title}: Classical Foundations & Hybrid Pipelines"
type: production-playbook
domain: {domain}
tags:
  - {domain_tag}
  - classical-cv
  - hybrid-pipelines
  - mathematical-foundations
updated: {TODAY}
aliases:
  - "{title} Classical & Hybrid Methods"
---

# \U0001f4d0 {title}: Classical Foundations & Hybrid Pipelines

A technical analysis of classical signal-processing algorithms, mathematical derivations, and modern hybrid deep-classical {domain} pipelines.

Related notes: [[topics/{slug}/00-{slug}-moc|{title} MOC]], [[topics/{slug}/03-representations-and-open-problems|{title} Backends & Open Problems]].

---

## 1. Classical Algorithms vs. Modern Deep Methods

```mermaid
flowchart TD
    Input[Input Data] --> Branch{{Approach}}
    Branch -->|Classical| Feat[Hand-crafted Feature Extraction]
    Branch -->|Deep Learning| CNN[Backbone: CNN / Transformer]
    Branch -->|Hybrid| Hybrid[Classical Preproc + Deep Backbone]
    Feat --> Out[Classical Decoder / Classifier]
    CNN --> Out2[Learned Decoder / Head]
    Hybrid --> Out3[Best-of-Both Pipeline]
```

---

## 2. Mathematical Derivation

### Core Mathematical Foundation
The fundamental optimization objective in {domain}:

$$\\mathcal{{L}} = \\mathcal{{L}}_{{\\text{{task}}}} + \\lambda \\mathcal{{R}}(\\theta)$$

where $\\mathcal{{L}}_{{\\text{{task}}}}$ is the task-specific loss and $\\mathcal{{R}}(\\theta)$ is the regularization term.

### Classical Feature Representation
Classical descriptors operate on local signal statistics:
$$\\phi(x) = \\text{{descriptor}}(\\nabla I(x))$$

Provide domain-specific mathematical formulation here.

---

## 3. Hybrid Pipeline Architecture

Modern production systems combine classical signal processing with deep learning:

1. **Classical preprocessing**: Filter noise, normalize signal, extract robust features.
2. **Deep backbone inference**: Run learned feature extractor on preprocessed input.
3. **Classical post-processing**: Apply geometric constraints, filtering, or fusion rules.

### Hybrid Synergy Benefits
- **Reliability**: Classical components provide deterministic behavior under distribution shift.
- **Efficiency**: Classical priors reduce the search space for the learned component.
- **Interpretability**: Classical stages are inspectable and debuggable.

---

## 4. When to Use Classical vs. Deep vs. Hybrid

| Criterion | Classical | Deep | Hybrid |
| :--- | :--- | :--- | :--- |
| **Data availability** | None needed | Large dataset | Small dataset |
| **Latency budget** | < 1 ms | 2–50 ms | 1–10 ms |
| **Distribution shift robustness** | High | Low | Moderate–High |
| **Interpretability required** | Full | None | Partial |
""")

    # README
    (topic_dir / "README.md").write_text(f"""\
---
title: {title} Master Index & Playbook
tags:
  - computer-vision
  - {domain_tag}
  - index
  - sota
  - models
  - playbooks
updated: {TODAY}
aliases:
  - "{title} Playbook"
  - "{title} Index"
---

# {title}: Master Index & Domain Guide

> **Obsidian Users**: Access the unified Map of Content at [[topics/{slug}/00-{slug}-moc|{title} MOC]].

## Overview
{domain} … (fill in executive brief: scope, key challenges, production constraints).

---

## SOTA & Research
Current state-of-the-art models and benchmark results for {domain}.

| Model | Metric | Latency | License |
| :--- | :--- | :--- | :--- |
| Model A | … | … ms | Apache-2.0 |
| Model B | … | … ms | MIT |

---

## Architecture Alternatives & Trade-offs

| Architecture Class | Best For | Trade-off |
| :--- | :--- | :--- |
| CNN-based | Low latency, edge | Lower accuracy ceiling |
| Transformer-based | High accuracy | Higher compute cost |
| Hybrid | Production balance | Implementation complexity |

---

## Popular Repos & Integrations

- [Repo A](https://github.com/org/repo-a) — description (License)
- [Repo B](https://github.com/org/repo-b) — description (License)

---

## End-to-End Pipeline & Workarounds

See [[topics/{slug}/02-production-pipeline-and-workarounds|Production Pipeline & Workarounds]] for detailed pipeline breakdown.

Key workarounds:
1. Pinned memory for zero-copy sensor ingestion.
2. TensorRT static shapes for kernel fusion.
3. GPU post-processing to avoid CPU bottleneck.

---

## Deployment & Real-time Notes

- **Jetson Orin**: Target < 10 ms end-to-end; use TensorRT FP16 + DLA offload.
- **RTX 4090 / A100**: TensorRT FP16 with CUDA graph capture for deterministic latency.
- **FPGA (Vitis AI)**: INT8 fixed-point; deterministic worst-case latency; no dynamic dispatch.
- **Mobile (NCNN + Vulkan)**: Vulkan compute shaders; zero runtime dependency; profiler: `ncnn-profiler`.

---

## Topic Organization

```text
topics/{slug}/
├── 00-{slug}-moc.md                          # Map of Content & Cross-Domain Hub
├── 01-historical-evolution-and-paradigms.md   # Historical Evolution & Paradigm Shifts
├── 02-production-pipeline-and-workarounds.md  # Production Pipeline, Traps & Workarounds
├── 03-representations-and-open-problems.md    # Representations, Backends & Open Problems
└── 04-classical-and-hybrid-methods.md         # Classical Foundations & Hybrid Pipelines
```
""")

    print(f"[+] Scaffolded topic: topics/{slug}/  (6 files)")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def scaffold_model(
    slug: str,
    title: str,
    arch_class: str,
    category: str = "real-time-unified",
) -> None:
    model_dir = REPO_ROOT / "architectures" / category
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / f"{slug}.md"
    if dest.exists():
        print(f"[warn] architectures/{category}/{slug}.md already exists; skipping.", file=sys.stderr)
        return

    dest.write_text(f"""\
---
title: "{title}"
type: model-deep-dive
architecture_class: {arch_class}
primary_license: Apache-2.0
updated: {TODAY}
tags:
  - model
  - {category}
  - {slug}
aliases:
  - "{title}"
---

# \u26a1 {title}

{title} is a {arch_class} architecture designed for … (fill in primary application and key innovation).

Related topics: *(add relevant MOC wikilinks here)*

---

## 1. Core Architectural Design

```mermaid
flowchart LR
    Input[Input Tensor] --> Backbone[{arch_class} Backbone]
    Backbone --> Neck[Feature Pyramid Neck]
    Neck --> Head[Task-Specific Head]
    Head --> Output[Predictions]
```

### Key Innovations
1. **Innovation 1**: Mathematical formulation or architectural choice.
2. **Innovation 2**: Describe training objective or loss formulation:
   $$\\mathcal{{L}} = \\mathcal{{L}}_{{\\text{{cls}}}} + \\lambda_1 \\mathcal{{L}}_{{\\text{{reg}}}} + \\lambda_2 \\mathcal{{L}}_{{\\text{{aux}}}}$$
3. **Innovation 3**: Describe efficiency mechanism.

---

## 2. Performance Benchmark Summary

| Variant | Parameters | FLOPs | Primary Metric | TensorRT FP16 Latency | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **{slug}-S** | … M | … G | … | … ms | Apache-2.0 |
| **{slug}-L** | … M | … G | … | … ms | Apache-2.0 |

---

## 3. Deployment Constraints & Hardware Notes

- **Dynamic Shape**: Describe any shape constraints for TensorRT compilation.
- **INT8 Quantization**: Identify layers sensitive to quantization (attention, norms).
- **Memory Footprint**: VRAM usage for batch size 1 at FP16.
- **Minimum CUDA Compute**: `sm_80` (Ampere) for Flash Attention; `sm_75` for base ops.

### TensorRT Engine Compilation
```bash
trtexec --onnx={slug}.onnx --saveEngine={slug}.engine \\
        --fp16 \\
        --minShapes=input:1x3x640x640 \\
        --optShapes=input:1x3x640x640 \\
        --maxShapes=input:4x3x640x640
```

---

## 4. Official Resources
- **Repository**: [org/{slug}](https://github.com/org/{slug})
- **Paper**: *{title}* (Year) — [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
""")

    print(f"[+] Scaffolded model: architectures/{category}/{slug}.md")


# ---------------------------------------------------------------------------
# cookbook
# ---------------------------------------------------------------------------

def scaffold_cookbook(slug: str, title: str) -> None:
    cookbooks_dir = REPO_ROOT / "cookbooks"
    cookbooks_dir.mkdir(exist_ok=True)

    # Determine next number: find max existing prefix
    existing = [
        d.name for d in cookbooks_dir.iterdir()
        if d.is_dir() and d.name[:2].isdigit()
    ]
    next_num = max((int(d.split("-")[0]) for d in existing), default=0) + 1
    prefix = f"{next_num:02d}-{slug}"
    cookbook_dir = cookbooks_dir / prefix
    if cookbook_dir.exists():
        print(f"[warn] cookbooks/{prefix}/ already exists; skipping.", file=sys.stderr)
        return

    cookbook_dir.mkdir()
    script = cookbook_dir / f"{slug}.py"
    script.write_text(f"""\
#!/usr/bin/env python3
\"\"\"
Cookbook {next_num:02d}: {title}

Self-contained demonstration with assert-based verification.
No external dependencies beyond the Python standard library (and numpy if available).

Run:
    python cookbooks/{prefix}/{slug}.py
\"\"\"

import math
import sys


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def demo_{slug.replace("-", "_")}() -> None:
    \"\"\"
    Demonstrate: {title}

    Fill in the algorithmic body here. Use only stdlib (math, random, itertools, etc.)
    or numpy if heavy numerics are required.
    \"\"\"
    print(f"[+] {title} — starting demo...")

    # TODO: implement the demonstration
    # Example: compute something verifiable
    result = math.sqrt(2.0)
    expected = 1.4142135623730951

    print(f"    sqrt(2) = {{result:.10f}}")
    assert abs(result - expected) < 1e-9, f"Expected {{expected}}, got {{result}}"

    print(f"[+] {title} — verified OK.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_{slug.replace("-", "_")}()
""")

    print(f"[+] Scaffolded cookbook: cookbooks/{prefix}/{slug}.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scaffold",
        description="Scaffold new topics, model notes, or cookbook scripts for cv-projects.",
    )
    sub = p.add_subparsers(dest="kind", required=True)

    t = sub.add_parser("topic", help="Scaffold a new topic directory with all required notes.")
    t.add_argument("slug", help="URL-safe slug, e.g. lidar-perception")
    t.add_argument("title", help='Human-readable title, e.g. "LiDAR Perception"')
    t.add_argument("domain", help='Domain name, e.g. "LiDAR Perception"')

    m = sub.add_parser("model", help="Scaffold a new model architecture note.")
    m.add_argument("slug", help="URL-safe slug, e.g. yolov12")
    m.add_argument("title", help='Human-readable title, e.g. "YOLOv12"')
    m.add_argument("arch_class", help='Architecture class, e.g. "Attention-Centric Real-Time Detector"')
    m.add_argument("--category", default="real-time-unified",
                   help="Architecture category subfolder (default: real-time-unified)")

    c = sub.add_parser("cookbook", help="Scaffold a new numbered cookbook script.")
    c.add_argument("slug", help="URL-safe slug, e.g. stereo-depth-estimation")
    c.add_argument("title", help='Human-readable title, e.g. "Stereo Depth Estimation"')

    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.kind == "topic":
        scaffold_topic(args.slug, args.title, args.domain)
    elif args.kind == "model":
        scaffold_model(args.slug, args.title, args.arch_class, args.category)
    elif args.kind == "cookbook":
        scaffold_cookbook(args.slug, args.title)
    return 0


if __name__ == "__main__":
    sys.exit(main())

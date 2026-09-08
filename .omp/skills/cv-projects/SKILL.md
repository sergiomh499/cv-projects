# cv-projects — Agent Operating Manual

## Purpose

This repository is a **Curated Knowledge & Cookbook Hub** for Computer Vision,
Machine Learning, AI Research, Sensor Fusion, and Hardware Deployment (CUDA,
TensorRT, Vulkan SC, Vitis AI, FPGA/NPU). It is structured as an Obsidian
vault with Dataview metadata, executable cookbooks, CLI tooling, and a pytest
test suite. Agents work here to add topics, write cookbooks, update
architectures, run validation, and maintain structural consistency.

---

## Repository Layout

```
cv-projects/
├── topics/                     # 20 domain topic directories
├── architectures/              # Cross-cutting architecture notes
│   ├── foundation-models/
│   ├── real-time-unified/
│   └── transformer-detectors/
├── cookbooks/                  # Numbered executable demonstrations
├── scripts/                    # CLI tooling
├── templates/                  # Obsidian note templates
├── docs/                       # Long-form reference documents
├── tests/                      # pytest suite
└── pyproject.toml
```

---

## Domain Topics Taxonomy

There are **20 topic domains** under `topics/`. Each directory is a
self-contained knowledge module with a fixed 6-file structure:

| Filename | Role |
|---|---|
| `00-<slug>-moc.md` | Map of Content — index and entry point for the domain |
| `01-historical-evolution-and-paradigms.md` | Chronological survey, key papers, paradigm shifts |
| `02-production-pipeline-and-workarounds.md` | Engineering reality: deployment patterns, known pitfalls, workarounds |
| `03-<domain-specific-title>.md` | Deep-dive on the domain's core technical axis (e.g. backends, solvers, representations) |
| `04-classical-and-hybrid-methods.md` | Classical baselines, hybrid classical+learned approaches |
| `README.md` | Brief human-readable summary for the domain directory |

**Current domains:**

1. `object-detection/`
2. `object-classification/`
3. `object-segmentation/`
4. `video-tracking/`
5. `lidar-perception/`
6. `sensor-fusion/`
7. `6dof-pose-estimation/`
8. `slam-and-spatial-perception/`
9. `optical-and-scene-flow-perception/`
10. `visual-guidance-and-robotics/`
11. `vla-and-physical-ai-robotics/`
12. `real-time-systems/`
13. `gpu-deployment/`
14. `fpga-deployment/`
15. `event-based-neuromorphic-vision/`
16. `active-3d-sensing-and-structured-light/`
17. `thermal-and-hyperspectral-vision/`
18. `explainability-and-interpretability/`
19. `safety-verification-and-robustness/`
20. `data-quality-and-verification/`

When adding a new topic, create all 6 files. Use `python scripts/cli.py scaffold` to generate the skeleton.

---

## Architectures Vault

`architectures/` holds cross-cutting notes on specific models and systems that
span multiple domains. Sub-directories by category:

- `foundation-models/` — Large vision/language models (Florence-2, Qwen2-VL, etc.)
- `real-time-unified/` — Production real-time stacks (BEVFusion, YOLOv12, TensorRT+Vulkan, etc.)
- `transformer-detectors/` — Transformer-based detection architectures

Each note is a standalone `.md` file named `<model-or-system-slug>.md`.
Frontmatter must include `tags`, `domain`, and `status` fields for Dataview queries.

---

## Cookbooks

`cookbooks/` contains numbered, self-contained executable demonstrations:

```
cookbooks/<NN>-<slug>/<script>.py
```

- `NN` is a zero-padded two-digit sequence number.
- `<slug>` is a short kebab-case description.
- Each script must be **self-contained**: runnable with `python <script>.py`
  with no external state.
- Each script must include an **assert-based verification block** that runs on
  `__main__` to confirm correctness of the demonstrated technique.
- Cookbooks are smoke-tested automatically by `tests/test_cookbooks_smoke.py`.

Example: `cookbooks/01-rfdetr-tensorrt/export_rfdetr_tensorrt.py`

---

## CLI Tooling

All tooling is invoked via the unified CLI:

```
python scripts/cli.py <command>
```

| Command | Action |
|---|---|
| `validate` | Check structural integrity of the vault (topic files, frontmatter, cookbook structure) |
| `sync` | Verify that doc indexes and MOC files are consistent with directory contents |
| `scaffold` | Generate the 6-file skeleton for a new topic domain |
| `bench` | Run hardware benchmark scripts under `scripts/` |
| `test` | Alias for running the pytest suite |

The CI pipeline runs `validate`, `sync`, and `pytest` on every push and PR.

---

## Frontmatter Schema

All topic notes and architecture files use YAML frontmatter compatible with
Obsidian and Dataview. Minimum required fields:

```yaml
---
tags: [<domain-tag>, <type-tag>]
domain: <domain-slug>
status: draft | active | stable
updated: YYYY-MM-DD
---
```

MOC files additionally include:
```yaml
type: moc
```

Cookbook scripts do **not** use frontmatter; they are Python, not Markdown.

---

## Verification Rules

Before yielding any work in this repository, an agent MUST confirm:

1. **Vault integrity passes:**
   ```
   python scripts/cli.py validate
   ```
   This checks all 20 topic directories have the required 6 files, frontmatter
   is well-formed, and cookbook directories contain at least one `.py` file.

2. **Pytest suite passes:**
   ```
   uv run pytest -v tests/
   ```
   Key test files:
   - `tests/test_vault_integrity.py` — structural checks mirroring `validate`
   - `tests/test_cookbooks_smoke.py` — imports and runs each cookbook's
     assertion block

3. **No ruff lint errors** for any Python added or modified:
   ```
   uv run ruff check .
   ```

CI enforces all three on every push to `main` and every pull request.

---

## Development Environment

- Python: **3.12** (managed by uv)
- Dependency management: `uv sync --extra dev`
- `pyproject.toml` extras:
  - `dev`: pytest, ruff, pre-commit, commitizen
  - `deployment`: onnx, onnxruntime-gpu

Install:
```
uv python install 3.12
uv sync --extra dev
```

---

## Agent Conventions

- **Never** add a new topic without the full 6-file structure; `validate` will fail.
- **Never** add a cookbook without an assert-based `__main__` verification block.
- **Always** use the existing file structure; check `glob` and `grep` before creating files.
- Prefer editing existing notes over creating new ones when updating a topic.
- When scaffolding, run `python scripts/cli.py scaffold` rather than creating files manually.
- Architecture notes go under `architectures/<category>/`, not under `topics/`.
- Long-form reference docs go under `docs/`, not under `topics/` or `architectures/`.

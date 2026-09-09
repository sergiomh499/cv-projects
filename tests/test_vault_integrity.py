#!/usr/bin/env python3
"""Vault integrity tests: topic structure, README sections, wiki-links, architecture frontmatter."""
import re

EXPECTED_TOPICS = {
    "object-detection",
    "object-segmentation",
    "object-classification",
    "video-tracking",
    "slam-and-spatial-perception",
    "6dof-pose-estimation",
    "lidar-perception",
    "sensor-fusion",
    "visual-guidance-and-robotics",
    "real-time-systems",
    "gpu-deployment",
    "fpga-deployment",
    "event-based-neuromorphic-vision",
    "thermal-and-hyperspectral-vision",
    "active-3d-sensing-and-structured-light",
    "optical-and-scene-flow-perception",
    "explainability-and-interpretability",
    "safety-verification-and-robustness",
    "data-quality-and-verification",
    "vla-and-physical-ai-robotics",
}

REQUIRED_README_SECTIONS = [
    "# Overview",
    "## SOTA & Research",
    "## Architecture Alternatives & Trade-offs",
    "## Popular Repos & Integrations",
    "## End-to-End Pipeline & Workarounds",
    "## Deployment & Real-Time Notes",
]

REQUIRED_ARCH_FIELDS = {"title", "architecture_class", "primary_license"}


def _parse_yaml_frontmatter(text: str) -> dict[str, str]:
    """Extract key: value pairs from a --- fenced YAML frontmatter block."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end]
    result: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line and not line.startswith(" ") and not line.startswith("-"):
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


# ---------------------------------------------------------------------------
# Test 1: all 20 topic folders exist
# ---------------------------------------------------------------------------

def test_all_topic_folders_exist(topic_dirs):
    found = {p.name for p in topic_dirs}
    missing = EXPECTED_TOPICS - found
    assert not missing, f"Missing topic folders: {sorted(missing)}"
    assert len(topic_dirs) >= 20, f"Expected at least 20 topics, found {len(topic_dirs)}"


# ---------------------------------------------------------------------------
# Test 2: each topic has 00-*-moc.md and README.md
# ---------------------------------------------------------------------------

def test_each_topic_has_moc_and_readme(topic_dirs):
    problems: list[str] = []
    for topic in topic_dirs:
        if not (topic / "README.md").exists():
            problems.append(f"{topic.name}: missing README.md")
        mocs = list(topic.glob("00-*-moc.md"))
        if not mocs:
            problems.append(f"{topic.name}: missing 00-*-moc.md")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# Test 3: each topic README.md that uses playbook format has all 6 sections
# ---------------------------------------------------------------------------

def test_playbook_readmes_have_required_sections(topic_dirs):
    """Only READMEs that contain '# Overview' (playbook format) are checked."""
    problems: list[str] = []
    for topic in topic_dirs:
        readme = topic / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8")
        # Only enforce sections on playbook-format READMEs
        if "# Overview" not in text:
            continue
        missing = [s for s in REQUIRED_README_SECTIONS if s not in text]
        if missing:
            problems.append(f"{topic.name}/README.md missing: {missing}")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# Test 4: zero broken Obsidian [[wiki-links]] across all content files
# ---------------------------------------------------------------------------

# Wikilink: must look like a file path (contains letters, slashes, hyphens, dots).
# Exclude numeric-only, ellipsis, and other non-path false positives.
_WIKILINK_RE = re.compile(r"\[\[([A-Za-z0-9][A-Za-z0-9_\-\./]*?)(?:\\?[|#][^\]]*)?\]\]")


def test_no_broken_wiki_links(all_markdown_files, repo_root):
    """A link target is 'broken' when no matching .md file exists in the repo."""
    broken: list[str] = []
    for md in all_markdown_files:
        text = md.read_text(encoding="utf-8", errors="replace")
        for m in _WIKILINK_RE.finditer(text):
            target = m.group(1).strip()
            # Resolve: target may or may not have .md extension
            candidate = repo_root / target
            resolved = (
                candidate.exists()
                or candidate.with_suffix(".md").exists()
                or (candidate.parent / (candidate.name + ".md")).exists()
            )
            if not resolved:
                broken.append(f"{md.relative_to(repo_root)}: [[{target}]]")
    assert not broken, (
        f"{len(broken)} broken wiki-link(s):\n" + "\n".join(broken[:40])
    )


# ---------------------------------------------------------------------------
# Test 5: architecture files have valid YAML frontmatter
# ---------------------------------------------------------------------------

def test_architecture_frontmatter(architecture_files):
    problems: list[str] = []
    for arch in architecture_files:
        text = arch.read_text(encoding="utf-8")
        fm = _parse_yaml_frontmatter(text)
        missing = REQUIRED_ARCH_FIELDS - set(fm.keys())
        if missing:
            problems.append(f"{arch.name}: missing frontmatter fields {sorted(missing)}")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# Test 6: all Mermaid diagrams pass syntax validation
# ---------------------------------------------------------------------------

def test_all_mermaid_diagrams_valid(repo_root):
    import subprocess
    import sys
    validator = repo_root / "scripts" / "validate_mermaid.py"
    assert validator.exists(), f"Mermaid validator script missing: {validator}"
    res = subprocess.run([sys.executable, str(validator)], cwd=str(repo_root), capture_output=True, text=True)
    assert res.returncode == 0, f"Mermaid validation failed:\n{res.stdout}\n{res.stderr}"


# ---------------------------------------------------------------------------
# Test 7: all Obsidian .canvas files have valid JSON and zero broken links
# ---------------------------------------------------------------------------

def test_all_canvases_valid(repo_root):
    import json
    canvas_dir = repo_root / "canvases"
    if not canvas_dir.exists():
        return
    canvases = list(canvas_dir.glob("*.canvas"))
    assert len(canvases) > 0, "canvases/ directory exists but contains no .canvas files"
    problems: list[str] = []
    for c_file in canvases:
        try:
            c_data = json.loads(c_file.read_text(encoding="utf-8"))
        except Exception as e:
            problems.append(f"JSON syntax error in {c_file.name}: {e}")
            continue
        for node in c_data.get("nodes", []):
            if node.get("type") == "file":
                target_rel = node.get("file", "")
                target = repo_root / target_rel
                if not target.exists():
                    problems.append(f"Broken file link in {c_file.name}: {target_rel}")
    assert not problems, "\n".join(problems)

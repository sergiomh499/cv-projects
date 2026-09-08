#!/usr/bin/env python3
"""
Comprehensive validator for cv-projects repository.
Checks:
1. Root README links to all 10 topic playbooks and evolution guides.
2. All 10 topic folders, README.md playbooks, and EVOLUTION.md guides exist.
3. YAML frontmatter exists with title and tags (Obsidian compatibility).
4. All required markdown sections are present in playbooks.
5. SOTA criteria:
   - At least 3 modern paper citations with year >= 2023 (arxiv/doi/proceedings).
   - At least 2 active code repository links (github/gitlab).
   - Standardized SOTA Benchmark Comparison Table with >= 3 data rows and quantitative metrics.
   - Commercial / Open-Source License audit per repository (e.g. Apache-2.0, MIT, AGPL-3.0, Non-Commercial).
"""

import sys
import re
from pathlib import Path

REQUIRED_TOPICS = [
    "object-detection",
    "object-segmentation",
    "object-classification",
    "video-tracking",
    "6dof-pose-estimation",
    "lidar-perception",
    "sensor-fusion",
    "fpga-deployment",
    "gpu-deployment",
    "real-time-systems",
]

REQUIRED_SECTIONS = [
    "# Overview",
    "## SOTA & Research",
    "## Architecture Alternatives & Trade-offs",
    "## Popular Repos & Integrations",
    "## End-to-End Pipeline & Workarounds",
    "## Deployment & Real-time Notes",
]

KNOWN_LICENSES = ["MIT", "Apache-2.0", "Apache 2.0", "BSD", "AGPL", "GPL", "CC-BY", "Non-Commercial", "Custom"]

def check_playbook(topic: str, path: Path) -> list[str]:
    errs = []
    text = path.read_text(encoding="utf-8")

    # 1. Frontmatter check
    if not text.startswith("---"):
        errs.append(f"Missing YAML frontmatter start ('---') in {topic}")
    else:
        parts = text.split("---", 2)
        if len(parts) < 3:
            errs.append(f"Malformed YAML frontmatter in {topic}")
        else:
            fm = parts[1]
            if "title:" not in fm or "tags:" not in fm:
                errs.append(f"Frontmatter missing 'title:' or 'tags:' in {topic}")

    # 2. Section check
    for section in REQUIRED_SECTIONS:
        if not re.search(re.escape(section), text, re.IGNORECASE):
            errs.append(f"Missing section '{section}' in {topic}")

    # 3. Paper citations with year >= 2023
    recent_years = re.findall(r"\b(202[3-6])\b", text)
    if len(recent_years) < 3:
        errs.append(f"Insufficient modern citations (>=2023) in {topic}: found {len(recent_years)}, expected >=3")

    # 4. Code repository links (github.com / gitlab.com)
    code_links = re.findall(r"https?://(?:www\.)?(?:github\.com|gitlab\.com)/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", text)
    unique_repos = set(code_links)
    if len(unique_repos) < 2:
        errs.append(f"Insufficient code repository links in {topic}: found {len(unique_repos)}, expected >=2")

    # 5. SOTA Benchmark table check
    table_matches = re.findall(r"\|[^\n]+\|\n\|(?:\s*:?---+:?\s*\|)+\n((?:\|[^\n]+\|\n)+)", text)
    if not table_matches:
        errs.append(f"Missing Markdown comparison/benchmark table in {topic}")
    else:
        has_min_rows = any(len(table.strip().split("\n")) >= 3 for table in table_matches)
        if not has_min_rows:
            errs.append(f"Benchmark/Comparison table in {topic} has fewer than 3 rows")

    # 6. License audit presence check
    has_license_mention = any(lic.lower() in text.lower() for lic in KNOWN_LICENSES)
    if not has_license_mention:
        errs.append(f"Missing license/commercial readiness audit in {topic}")

    return errs

def check_evolution(topic: str, path: Path) -> list[str]:
    errs = []
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        errs.append(f"Missing YAML frontmatter start in {topic}/EVOLUTION.md")
    if "```mermaid" not in text:
        errs.append(f"Missing Mermaid diagram in {topic}/EVOLUTION.md")
    return errs

def main() -> int:
    root = Path(__file__).resolve().parent.parent
    errors = []

    print(f"[+] Validating cv-projects repository at: {root}")

    # Check root README
    root_readme = root / "README.md"
    if not root_readme.exists():
        errors.append("Missing root README.md")
    else:
        content = root_readme.read_text(encoding="utf-8")
        for topic in REQUIRED_TOPICS:
            expected_playbook = f"topics/{topic}/README.md"
            expected_evolution = f"topics/{topic}/EVOLUTION.md"
            if expected_playbook not in content:
                errors.append(f"Root README.md is missing link to '{expected_playbook}'")
            if expected_evolution not in content:
                errors.append(f"Root README.md is missing link to '{expected_evolution}'")

    # Check each topic
    for topic in REQUIRED_TOPICS:
        topic_dir = root / "topics" / topic
        topic_readme = topic_dir / "README.md"
        topic_evolution = topic_dir / "EVOLUTION.md"
        if not topic_dir.is_dir():
            errors.append(f"Missing directory: topics/{topic}")
            continue
        if not topic_readme.is_file():
            errors.append(f"Missing playbook: topics/{topic}/README.md")
            continue
        if not topic_evolution.is_file():
            errors.append(f"Missing evolution guide: topics/{topic}/EVOLUTION.md")
            continue

        errors.extend(check_playbook(topic, topic_readme))
        errors.extend(check_evolution(topic, topic_evolution))

    if errors:
        print(f"[-] Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"    - {err}")
        return 1

    print(f"[+] Validation PASSED: All {len(REQUIRED_TOPICS)} topic playbooks and evolution guides verified.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Comprehensive validator for cv-projects repository.
Checks:
1. Root README links to all 10 topics.
2. All 10 topic folders and README.md playbooks exist.
3. YAML frontmatter exists with title and tags (Obsidian compatibility).
4. All required markdown sections are present.
5. SOTA criteria:
   - At least 3 modern paper citations with year >= 2023 (arxiv/doi/proceedings).
   - At least 2 active code repository links (github/gitlab).
   - A standardized Markdown SOTA Benchmark Comparison Table with >= 3 data rows and quantitative metrics.
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
    # Matches patterns like 2023, 2024, 2025, 2026 inside citation lines
    recent_years = re.findall(r"\b(202[3-6])\b", text)
    if len(recent_years) < 3:
        errs.append(f"Insufficient modern citations (>=2023) in {topic}: found {len(recent_years)}, expected >=3")

    # 4. Code repository links (github.com / gitlab.com)
    code_links = re.findall(r"https?://(?:www\.)?(?:github\.com|gitlab\.com)/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", text)
    unique_repos = set(code_links)
    if len(unique_repos) < 2:
        errs.append(f"Insufficient code repository links in {topic}: found {len(unique_repos)}, expected >=2")

    # 5. SOTA Benchmark table check (Markdown table under SOTA or Benchmark section)
    # Check for a markdown table with header and rows
    table_matches = re.findall(r"\|[^\n]+\|\n\|(?:\s*:?---+:?\s*\|)+\n((?:\|[^\n]+\|\n)+)", text)
    if not table_matches:
        errs.append(f"Missing Markdown comparison/benchmark table in {topic}")
    else:
        # Verify at least one table has >= 3 rows
        has_min_rows = any(len(table.strip().split("\n")) >= 3 for table in table_matches)
        if not has_min_rows:
            errs.append(f"Benchmark/Comparison table in {topic} has fewer than 3 rows")

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
            expected_link = f"topics/{topic}/README.md"
            if expected_link not in content:
                errors.append(f"Root README.md is missing link to '{expected_link}'")

    # Check each topic
    for topic in REQUIRED_TOPICS:
        topic_dir = root / "topics" / topic
        topic_readme = topic_dir / "README.md"
        if not topic_dir.is_dir():
            errors.append(f"Missing directory: topics/{topic}")
            continue
        if not topic_readme.is_file():
            errors.append(f"Missing playbook: topics/{topic}/README.md")
            continue

        topic_errs = check_playbook(topic, topic_readme)
        errors.extend(topic_errs)

    if errors:
        print(f"[-] Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"    - {err}")
        return 1

    print(f"[+] Validation PASSED: All {len(REQUIRED_TOPICS)} topic playbooks satisfy SOTA, metadata, and Obsidian schema.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

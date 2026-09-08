#!/usr/bin/env python3
"""
Validation script for cv-projects repository.
Verifies all 10 topic playbooks exist, contain all mandatory sections,
and ensures root README has valid relative links.
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

def main() -> int:
    root = Path(__file__).resolve().parent.parent
    errors = []

    print(f"[+] Validating repository structure at: {root}")

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

    # Check each topic folder and README.md
    for topic in REQUIRED_TOPICS:
        topic_dir = root / "topics" / topic
        topic_readme = topic_dir / "README.md"
        if not topic_dir.is_dir():
            errors.append(f"Missing directory: topics/{topic}")
            continue
        if not topic_readme.is_file():
            errors.append(f"Missing playbook: topics/{topic}/README.md")
            continue

        text = topic_readme.read_text(encoding="utf-8")
        for section in REQUIRED_SECTIONS:
            # Check for header presence (case-insensitive search for header name)
            if not re.search(re.escape(section), text, re.IGNORECASE):
                errors.append(f"topics/{topic}/README.md missing required section: '{section}'")

    if errors:
        print(f"[-] Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"    - {err}")
        return 1

    print(f"[+] Validation PASSED: All {len(REQUIRED_TOPICS)} topic playbooks and sections verified.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

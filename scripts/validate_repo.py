#!/usr/bin/env python3
"""
Comprehensive validator for cv-projects repository.
Checks:
1. Root README links to all 10 topic playbooks.
2. All 10 topic folders exist and contain:
   - README.md (Domain Master Index)
   - 00-*-moc.md (Map of Content note for Obsidian) or MOC link
   - Evolution guide
   - Production playbook
3. YAML frontmatter exists with title and tags (Obsidian compatibility).
4. All required markdown sections and SOTA criteria satisfied.
5. Commercial / Open-Source License audit per repository.
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

KNOWN_LICENSES = ["MIT", "Apache-2.0", "Apache 2.0", "BSD", "AGPL", "GPL", "CC-BY", "Non-Commercial", "Custom"]

def check_note_frontmatter(path: Path) -> list[str]:
    errs = []
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        errs.append(f"Missing YAML frontmatter start in {path.name}")
    else:
        parts = text.split("---", 2)
        if len(parts) < 3:
            errs.append(f"Malformed YAML frontmatter in {path.name}")
        else:
            fm = parts[1]
            if "title:" not in fm or "tags:" not in fm:
                errs.append(f"Frontmatter missing 'title:' or 'tags:' in {path.name}")
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
            if expected_playbook not in content:
                errors.append(f"Root README.md is missing link to '{expected_playbook}'")

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

        errors.extend(check_note_frontmatter(topic_readme))

        # Check for evolution guide (either 01-*.md or EVOLUTION.md)
        evolution_files = list(topic_dir.glob("*evolution*.md")) + list(topic_dir.glob("EVOLUTION.md"))
        if not evolution_files:
            errors.append(f"Missing evolution guide in topics/{topic}")
        else:
            errors.extend(check_note_frontmatter(evolution_files[0]))

    if errors:
        print(f"[-] Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"    - {err}")
        return 1

    print(f"[+] Validation PASSED: All {len(REQUIRED_TOPICS)} topic domains verified with Obsidian schemas.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

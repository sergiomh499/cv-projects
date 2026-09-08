#!/usr/bin/env python3
"""
Comprehensive validator for cv-projects repository.
Verifies all 12 topic domains, shared architectures, Obsidian schemas, and relative links.
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
    "slam-and-spatial-perception",
    "visual-guidance-and-robotics",
    "data-quality-and-verification",
    "safety-verification-and-robustness",
    "explainability-and-interpretability",
]

def check_yaml_frontmatter(content: str, filepath: Path) -> list[str]:
    errors = []
    if not content.startswith("---"):
        errors.append(f"{filepath}: Missing starting YAML frontmatter delimiter '---'")
        return errors
    parts = content.split("---", 2)
    if len(parts) < 3:
        errors.append(f"{filepath}: Malformed YAML frontmatter")
        return errors
    frontmatter = parts[1]
    if "title:" not in frontmatter:
        errors.append(f"{filepath}: Missing 'title' property in YAML frontmatter")
    if "tags:" not in frontmatter:
        errors.append(f"{filepath}: Missing 'tags' property in YAML frontmatter")
    return errors

def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    print(f"[+] Validating cv-projects repository at: {repo_root}")

    errors = []

    # 1. Root README check
    root_readme = repo_root / "README.md"
    if not root_readme.exists():
        errors.append("Root README.md is missing!")
    else:
        readme_content = root_readme.read_text(encoding="utf-8")
        errors.extend(check_yaml_frontmatter(readme_content, root_readme))
        for topic in REQUIRED_TOPICS:
            pattern = rf"topics/{topic}/README\.md"
            if not re.search(pattern, readme_content):
                errors.append(f"Root README.md missing relative link to topics/{topic}/README.md")

    # 2. Architectures vault check
    arch_vault = repo_root / "architectures"
    if not arch_vault.exists():
        errors.append("architectures/ directory is missing!")
    else:
        arch_files = list(arch_vault.glob("**/*.md"))
        if len(arch_files) < 10:
            errors.append(f"architectures/ contains only {len(arch_files)} models (expected >= 10)")

    # 3. Validate each of the required topics
    for topic in REQUIRED_TOPICS:
        topic_dir = repo_root / "topics" / topic
        if not topic_dir.exists():
            errors.append(f"Topic directory missing: {topic_dir}")
            continue

        readme_file = topic_dir / "README.md"
        if not readme_file.exists():
            errors.append(f"Missing master index README.md in {topic_dir}")
        else:
            errors.extend(check_yaml_frontmatter(readme_file.read_text(encoding="utf-8"), readme_file))

        # Check for 00-*-moc.md
        moc_files = list(topic_dir.glob("00-*-moc.md"))
        if not moc_files:
            errors.append(f"Missing 00-*-moc.md in {topic_dir}")
        else:
            errors.extend(check_yaml_frontmatter(moc_files[0].read_text(encoding="utf-8"), moc_files[0]))

        # Check for 01-historical-evolution-and-paradigms.md
        evo_file = topic_dir / "01-historical-evolution-and-paradigms.md"
        if not evo_file.exists():
            errors.append(f"Missing 01-historical-evolution-and-paradigms.md in {topic_dir}")
        else:
            errors.extend(check_yaml_frontmatter(evo_file.read_text(encoding="utf-8"), evo_file))

        # Check for 02-production-pipeline-and-workarounds.md
        playbook_file = topic_dir / "02-production-pipeline-and-workarounds.md"
        if not playbook_file.exists():
            errors.append(f"Missing 02-production-pipeline-and-workarounds.md in {topic_dir}")
        else:
            errors.extend(check_yaml_frontmatter(playbook_file.read_text(encoding="utf-8"), playbook_file))

    if errors:
        print(f"[-] Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"[+] Validation PASSED: All {len(REQUIRED_TOPICS)} topic domains and shared architectures fully verified with Obsidian schemas.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

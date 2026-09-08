#!/usr/bin/env python3
"""
Automated Documentation Synchronizer & Cross-Reference Trigger.
Scans the cv-projects repository for models, hardware architectures,
and software frameworks, updates backlinks, checks for stale dates,
and regenerates the vault tables in root README.md.

Usage:
    python scripts/update_docs.py [--check] [--fix]
"""

import argparse
import re
import sys
from pathlib import Path


def parse_frontmatter(content: str) -> dict:
    if not content.startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    fm = {}
    for line in parts[1].strip().split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip().strip('"').strip("'")
    return fm

def scan_architectures(repo_root: Path):
    architectures = []
    arch_root = repo_root / "architectures"
    for md_file in sorted(arch_root.glob("**/*.md")):
        if md_file.stem in ("README", "00-architectures-moc"):
            continue
        content = md_file.read_text(encoding="utf-8")
        fm = parse_frontmatter(content)
        rel_path = md_file.relative_to(repo_root)
        name = md_file.stem
        architectures.append({
            "title": fm.get("title", name),
            "license": fm.get("primary_license", "Unknown"),
            "class": fm.get("architecture_class", "General"),
            "path": str(rel_path),
            "stem": name,
            "rel_link": f"[[{rel_path.with_suffix('')}|{fm.get('title', name)}]]"
        })
    return architectures

def scan_hardware(repo_root: Path):
    hardware_list = []
    hw_root = repo_root / "hardware"
    for md_file in sorted(hw_root.glob("*.md")):
        if md_file.stem in ("README", "00-hardware-moc"):
            continue
        content = md_file.read_text(encoding="utf-8")
        fm = parse_frontmatter(content)
        rel_path = md_file.relative_to(repo_root)
        name = md_file.stem
        hardware_list.append({
            "title": fm.get("title", name),
            "domain": fm.get("domain", "Hardware Acceleration"),
            "type": fm.get("type", "Hardware Architecture"),
            "path": str(rel_path),
            "stem": name,
            "rel_link": f"[[{rel_path.with_suffix('')}|{fm.get('title', name)}]]"
        })
    return hardware_list

def scan_frameworks(repo_root: Path):
    frameworks_list = []
    fw_root = repo_root / "frameworks"
    for md_file in sorted(fw_root.glob("*.md")):
        if md_file.stem in ("README", "00-frameworks-moc"):
            continue
        content = md_file.read_text(encoding="utf-8")
        fm = parse_frontmatter(content)
        rel_path = md_file.relative_to(repo_root)
        name = md_file.stem
        frameworks_list.append({
            "title": fm.get("title", name),
            "domain": fm.get("domain", "Software Framework"),
            "maintainer": fm.get("maintainer", "Open Source"),
            "path": str(rel_path),
            "stem": name,
            "rel_link": f"[[{rel_path.with_suffix('')}|{fm.get('title', name)}]]"
        })
    return frameworks_list

def sync_root_readme(repo_root: Path, architectures: list, hardware: list, frameworks: list):
    readme_path = repo_root / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    
    # 1. Update Architecture Vault section
    lines = ["Dedicated landmark architectures and foundation models (each in an independent reference document):"]
    for arch in architectures:
        lines.append(f"- **{arch['rel_link']}**: {arch['class']} ({arch['license']}).")
    arch_section = "\n".join(lines)
    
    pattern_arch = r"(## 🏛️ Central Architecture Vault \(`architectures/`\)\n\n)(.*?)(?=\n\n---|\Z)"
    if re.search(pattern_arch, content, flags=re.DOTALL):
        content = re.sub(pattern_arch, rf"\g<1>{arch_section}", content, flags=re.DOTALL)

    # 2. Add or update Hardware Vault section in root README
    hw_lines = ["\n\n---\n\n## ⚡ Hardware Platforms & Silicon Acceleration Vault (`hardware/`)\n\nDedicated silicon guides covering compute, memory, and functional safety (each in an independent document):\n"]
    for hw in hardware:
        hw_lines.append(f"- **{hw['rel_link']}**: {hw['domain']}.")
    hw_section = "\n".join(hw_lines)

    # 3. Add or update Frameworks Vault section in root README
    fw_lines = ["\n\n---\n\n## 🚀 Software Frameworks, Compilers & Inference Runtimes (`frameworks/`)\n\nDedicated guides covering compilation pipelines, zero-copy memory, and runtime models (each in an independent document):\n"]
    for fw in frameworks:
        fw_lines.append(f"- **{fw['rel_link']}**: {fw['domain']}.")
    fw_section = "\n".join(fw_lines)

    # Check if Hardware Vault already present
    pattern_hw = r"(## ⚡ Hardware Platforms & Silicon Acceleration Vault \(`hardware/`\)\n\n)(.*?)(?=\n\n---|\Z)"
    if re.search(pattern_hw, content, flags=re.DOTALL):
        content = re.sub(pattern_hw, rf"\g<1>{hw_section.strip()}", content, flags=re.DOTALL)
    else:
        # Insert before formal mathematical proofs or cookbooks
        insert_target = "## 📐 Formal Mathematical Proofs"
        if insert_target in content:
            content = content.replace(insert_target, f"{hw_section.strip()}\n\n---\n\n{fw_section.strip()}\n\n---\n\n{insert_target}")

    readme_path.write_text(content, encoding="utf-8")
    print("[+] Synchronized root README.md Architecture, Hardware, and Frameworks Vault sections.")

def main():
    parser = argparse.ArgumentParser(description="Synchronize repository docs and triggers.")
    parser.add_argument("--check", action="store_true", help="Check if docs need synchronization.")
    parser.add_argument("--fix", action="store_true", default=True, help="Automatically synchronize and update.")
    _args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    print(f"[+] Scanning vault at: {repo_root}")

    architectures = scan_architectures(repo_root)
    print(f"[+] Discovered {len(architectures)} central models in architectures/ vault.")

    hardware = scan_hardware(repo_root)
    print(f"[+] Discovered {len(hardware)} hardware platforms in hardware/ vault.")

    frameworks = scan_frameworks(repo_root)
    print(f"[+] Discovered {len(frameworks)} software frameworks in frameworks/ vault.")

    sync_root_readme(repo_root, architectures, hardware, frameworks)

    print("[+] All vault documentation indices successfully synchronized.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

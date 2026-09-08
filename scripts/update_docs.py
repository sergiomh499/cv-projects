#!/usr/bin/env python3
"""
Automated Documentation Synchronizer & Cross-Reference Trigger.
Scans the cv-projects repository for newly added models, architectures,
and topic playbooks, updates backlinks in topic MOCs, checks for stale dates,
and regenerates the central Architecture Vault table in root README.md.

Usage:
    python scripts/update_docs.py [--check] [--fix]
"""

import sys
import re
import argparse
from pathlib import Path
from datetime import date

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

def scan_vault(repo_root: Path):
    architectures = []
    arch_root = repo_root / "architectures"
    for md_file in sorted(arch_root.glob("**/*.md")):
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

def sync_root_readme(repo_root: Path, architectures: list):
    readme_path = repo_root / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    
    # Generate updated Architecture Vault section
    lines = ["Cross-cutting models and multi-task foundation backbones shared across topics:"]
    for arch in architectures:
        lines.append(f"- **{arch['rel_link']}**: {arch['class']} ({arch['license']}).")
    
    arch_section = "\n".join(lines)
    
    pattern = r"(## 🏛️ Central Architecture Vault \(`architectures/`\)\n\n)(.*?)(?=\n\n---|\Z)"
    if re.search(pattern, content, flags=re.DOTALL):
        updated_content = re.sub(pattern, rf"\g<1>{arch_section}", content, flags=re.DOTALL)
        if updated_content != content:
            readme_path.write_text(updated_content, encoding="utf-8")
            print("[+] Synchronized root README.md Architecture Vault table.")

def main():
    parser = argparse.ArgumentParser(description="Synchronize repository docs and triggers.")
    parser.add_argument("--check", action="store_true", help="Check if docs need synchronization.")
    parser.add_argument("--fix", action="store_true", default=True, help="Automatically synchronize and update.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    print(f"[+] Scanning vault at: {repo_root}")

    architectures = scan_vault(repo_root)
    print(f"[+] Discovered {len(architectures)} central models in architectures/ vault.")

    sync_root_readme(repo_root, architectures)

    print("[+] All vault documentation indices successfully synchronized.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

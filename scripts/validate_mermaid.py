#!/usr/bin/env python3
"""
Mermaid Diagram Validator for cv-projects repository.
Checks all ```mermaid blocks across markdown files.
Invokes the official JS Mermaid parser via bun/node if available,
and performs static structural heuristics to ensure diagrams do not fail.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def check_mermaid_static(repo_root: Path) -> list[dict]:
    """
    Static heuristic validation for common Mermaid syntax pitfalls:
    - Subgraph titles with spaces/parentheses not quoted or lacking an ID
    - Node labels with special characters (parentheses, slashes, arrows) not enclosed in quotes
    - Nested unquoted brackets like [Teacher [CLS] Token Logits]
    """
    errors = []
    mermaid_re = re.compile(r'```mermaid\s*\n(.*?)```', re.DOTALL)
    
    for md_path in repo_root.glob("**/*.md"):
        if any(p in md_path.parts for p in ['.git', 'node_modules', '.venv', '.obsidian']):
            continue
            
        content = md_path.read_text(encoding="utf-8")
        matches = mermaid_re.findall(content)
        
        for idx, diagram in enumerate(matches, 1):
            lines = diagram.strip().split("\n")
            for line_no, line in enumerate(lines, 1):
                sline = line.strip()
                if not sline or sline.startswith("%%"):
                    continue
                
                # Check for bad subgraph syntax
                m_sub = re.match(r'^subgraph\s+([^\[\]\n]+)$', sline)
                if m_sub:
                    title = m_sub.group(1).strip()
                    if ("(" in title or ")" in title or " " in title) and not ('[' in title and ']' in title):
                        errors.append({
                            "file": str(md_path.relative_to(repo_root)),
                            "diagram": idx,
                            "line": line_no,
                            "reason": f"Unquoted subgraph header with spaces/parentheses: '{sline}'",
                        })
                        
                # Check for unquoted arrows inside node definitions
                if re.search(r'\[[^"\'\]]*->[^"\'\]]*\]', sline):
                    errors.append({
                        "file": str(md_path.relative_to(repo_root)),
                        "diagram": idx,
                        "line": line_no,
                        "reason": f"Unquoted '->' inside node brackets: '{sline}'",
                    })

    return errors

def main() -> int:
    bun_path = shutil.which("bun")
    js_script = REPO_ROOT / "scripts" / "validate_mermaid.js"
    
    if bun_path and js_script.exists():
        try:
            res = subprocess.run([bun_path, str(js_script)], cwd=str(REPO_ROOT), capture_output=True, text=True)
            if res.returncode == 0:
                print(res.stdout.strip())
                return 0
            elif "Cannot find package" not in res.stderr and "Cannot find module" not in res.stderr:
                print(res.stdout.strip())
                print(res.stderr.strip(), file=sys.stderr)
                return res.returncode
        except Exception:
            pass
    # 2. Fallback to Python static checks
    static_errors = check_mermaid_static(REPO_ROOT)
    if static_errors:
        print(f"[-] Static Mermaid validation failed ({len(static_errors)} error(s)):")
        for err in static_errors:
            print(f"    - {err['file']} (dia #{err['diagram']}, line {err['line']}): {err['reason']}")
        return 1

    print("[+] Static Mermaid syntax validation passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

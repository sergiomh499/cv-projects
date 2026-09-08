#!/usr/bin/env python3
"""
Unified CLI entry point for cv-projects developer tooling.

Subcommands:
  validate  — run validate_repo.py checks
  sync      — run update_docs.py sync
  scaffold  — scaffold topic, model, or cookbook (delegates to scaffold.py)
  bench     — run benchmark_hardware_real.py
  test      — invoke pytest on tests/

Usage:
    python scripts/cli.py --help
    python scripts/cli.py validate
    python scripts/cli.py sync [--check] [--fix]
    python scripts/cli.py scaffold topic  <slug> <title> <domain>
    python scripts/cli.py scaffold model  <slug> <title> <arch_class> [--category CAT]
    python scripts/cli.py scaffold cookbook <slug> <title>
    python scripts/cli.py bench
    python scripts/cli.py test [pytest-args ...]
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def _run(cmd: list[str]) -> int:
    """Run a subprocess and return its exit code."""
    result = subprocess.run(cmd)
    return result.returncode


def cmd_validate(_args: argparse.Namespace) -> int:
    return _run([sys.executable, str(SCRIPTS / "validate_repo.py")])


def cmd_sync(args: argparse.Namespace) -> int:
    cmd = [sys.executable, str(SCRIPTS / "update_docs.py")]
    if getattr(args, "check", False):
        cmd.append("--check")
    if getattr(args, "fix", False):
        cmd.append("--fix")
    return _run(cmd)


def cmd_scaffold(args: argparse.Namespace) -> int:
    # Reconstruct argv for scaffold.py from parsed namespace
    cmd = [sys.executable, str(SCRIPTS / "scaffold.py"), args.scaffold_kind]
    if args.scaffold_kind == "topic":
        cmd += [args.slug, args.title, args.domain]
    elif args.scaffold_kind == "model":
        cmd += [args.slug, args.title, args.arch_class, "--category", args.category]
    elif args.scaffold_kind == "cookbook":
        cmd += [args.slug, args.title]
    return _run(cmd)


def cmd_bench(_args: argparse.Namespace) -> int:
    return _run([sys.executable, str(SCRIPTS / "benchmark_hardware_real.py")])


def cmd_test(args: argparse.Namespace) -> int:
    extra = getattr(args, "pytest_args", []) or []
    return _run([sys.executable, "-m", "pytest", str(REPO_ROOT / "tests")] + extra)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cv",
        description="cv-projects developer CLI: validate, sync, scaffold, bench, test.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # validate
    sub.add_parser("validate", help="Validate repository integrity (frontmatter, links, structure).")

    # sync
    sy = sub.add_parser("sync", help="Sync documentation and regenerate cross-references.")
    sy.add_argument("--check", action="store_true", help="Report issues without writing changes.")
    sy.add_argument("--fix", action="store_true", help="Automatically fix stale dates and missing links.")

    # scaffold
    sc = sub.add_parser("scaffold", help="Scaffold new topic, model, or cookbook.")
    sc_sub = sc.add_subparsers(dest="scaffold_kind", required=True)

    sc_topic = sc_sub.add_parser("topic", help="Create a full topic directory with all required notes.")
    sc_topic.add_argument("slug", help="URL-safe slug, e.g. lidar-perception")
    sc_topic.add_argument("title", help='Human-readable title, e.g. "LiDAR Perception"')
    sc_topic.add_argument("domain", help='Domain label, e.g. "LiDAR Perception"')

    sc_model = sc_sub.add_parser("model", help="Create a model architecture deep-dive note.")
    sc_model.add_argument("slug", help="URL-safe slug, e.g. yolov12")
    sc_model.add_argument("title", help='Human-readable title, e.g. "YOLOv12"')
    sc_model.add_argument("arch_class", help='Architecture class, e.g. "Attention-Centric Real-Time Detector"')
    sc_model.add_argument("--category", default="real-time-unified",
                          help="Architecture category subfolder (default: real-time-unified)")

    sc_cb = sc_sub.add_parser("cookbook", help="Create a numbered cookbook script.")
    sc_cb.add_argument("slug", help="URL-safe slug, e.g. stereo-depth-estimation")
    sc_cb.add_argument("title", help='Human-readable title, e.g. "Stereo Depth Estimation"')

    # bench
    sub.add_parser("bench", help="Run hardware benchmark suite.")

    # test
    ts = sub.add_parser("test", help="Run pytest on tests/ directory.")
    ts.add_argument("pytest_args", nargs=argparse.REMAINDER,
                    help="Additional arguments forwarded to pytest (e.g. -v, -k pattern).")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "validate": cmd_validate,
        "sync": cmd_sync,
        "scaffold": cmd_scaffold,
        "bench": cmd_bench,
        "test": cmd_test,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

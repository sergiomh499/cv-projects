#!/usr/bin/env python3
"""Shared pytest fixtures for cv-projects test suite."""
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Path to the repository workspace root."""
    return Path(__file__).parent.parent.resolve()


@pytest.fixture(scope="session")
def topic_dirs(repo_root) -> list[Path]:
    """All topic subdirectory paths under topics/."""
    return sorted(p for p in (repo_root / "topics").iterdir() if p.is_dir())


@pytest.fixture(scope="session")
def all_markdown_files(repo_root) -> list[Path]:
    """All .md files excluding .obsidian/ and templates/."""
    return sorted(
        p for p in repo_root.rglob("*.md")
        if ".obsidian" not in p.parts and "templates" not in p.parts
    )


@pytest.fixture(scope="session")
def architecture_files(repo_root) -> list[Path]:
    """All .md files under architectures/."""
    return sorted((repo_root / "architectures").rglob("*.md"))

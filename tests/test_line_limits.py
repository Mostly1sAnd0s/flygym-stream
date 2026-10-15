"""Meta-test: no project file may reach 500 lines (PLAN.md §2.3).

Covers .py, .js, .html, .css, .toml files anywhere in the project,
skipping third-party / build directories. Per-file line budgets are
listed in PLAN.md §4; this test enforces the hard 500-line ceiling.
"""

from __future__ import annotations

import pathlib

EXTENSIONS = {".py", ".js", ".cjs", ".html", ".css", ".toml"}
SKIP_DIRS = {".venv", ".git", "__pycache__", "node_modules"}
MAX_LINES = 500  # hard ceiling; files must stay strictly below this


def project_root() -> pathlib.Path:
    """Return the project root (parent of this tests/ directory)."""
    return pathlib.Path(__file__).resolve().parents[1]


def project_files() -> list[pathlib.Path]:
    """Return every checked source file in the project."""
    root = project_root()
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def test_every_file_stays_below_500_lines() -> None:
    """Fail with a report of every offending file (path + line count)."""
    root = project_root()
    offenders = []
    for path in project_files():
        n_lines = len(path.read_text(encoding="utf-8").splitlines())
        if n_lines >= MAX_LINES:
            offenders.append(f"  {path.relative_to(root)}: {n_lines} lines")
    assert not offenders, (
        f"Files at or above {MAX_LINES} lines (PLAN.md §2.3):\n" + "\n".join(offenders)
    )

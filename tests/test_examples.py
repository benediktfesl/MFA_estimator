"""Smoke tests for example scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SCRIPT = REPO_ROOT / "examples" / "mfa_estimator_example.py"


def test_example_script_exists() -> None:
    assert EXAMPLE_SCRIPT.is_file()


def test_example_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--nr" in result.stdout


def test_example_script_imports_successfully() -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_SCRIPT), "--nr", "0"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
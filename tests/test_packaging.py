"""Tests for the build configuration in setup.py."""

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SETUP_PY = REPO_ROOT / "setup.py"


def _load_ext_modules(monkeypatch):
    """Run setup.py with setup() stubbed out and return its ext_modules."""
    pytest.importorskip("pybind11")
    import setuptools

    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    namespace = runpy.run_path(str(SETUP_PY))
    return namespace["ext_modules"]


def test_cpp_extension_is_optional(monkeypatch):
    ext_modules = _load_ext_modules(monkeypatch)

    assert [ext.name for ext in ext_modules] == ["src.cpp._lob_cpp"]
    assert all(ext.optional for ext in ext_modules)


def test_build_succeeds_without_a_compiler(tmp_path):
    """With CC and CXX pointing at a missing compiler, build_ext warns and skips."""
    pytest.importorskip("pybind11")
    missing = str(tmp_path / "no-such-compiler")
    env = dict(os.environ, CC=missing, CXX=missing)

    result = subprocess.run(
        [
            sys.executable,
            str(SETUP_PY),
            "build_ext",
            "--build-lib",
            str(tmp_path / "lib"),
            "--build-temp",
            str(tmp_path / "temp"),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not list((tmp_path / "lib").rglob("_lob_cpp*"))

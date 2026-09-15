"""GeoSWE refuses the GPU backend when two CuPy builds are installed.

Every CuPy wheel (cupy-cuda12x, cupy-cuda13x, ...) installs the same `cupy` package,
so a second one overwrites the first. That state came from installing the default
CUDA 12 build and then adding the CUDA 13 one, and it surfaced much later as
"Failed to find CUDA headers". CuPy only warns; GeoSWE stops with the fix.
"""
from types import SimpleNamespace

import pytest

from geoswe import backend


def _fake(*names):
    return lambda: [SimpleNamespace(metadata={"Name": n}) for n in names]


def test_two_cupy_builds_are_refused(monkeypatch):
    monkeypatch.delenv("GEOSWE_ALLOW_MULTIPLE_CUPY", raising=False)
    monkeypatch.setattr("importlib.metadata.distributions",
                        _fake("numpy", "cupy-cuda12x", "cupy_cuda13x", "cupyx-tools"))
    with pytest.raises(RuntimeError, match=r"cupy-cuda12x, cupy-cuda13x.*pip uninstall -y"):
        backend._check_single_cupy()


def test_one_build_passes_and_other_packages_are_ignored(monkeypatch):
    monkeypatch.delenv("GEOSWE_ALLOW_MULTIPLE_CUPY", raising=False)
    monkeypatch.setattr("importlib.metadata.distributions", _fake("cupy-cuda13x", "cupyx-tools", "numpy"))
    backend._check_single_cupy()
    assert backend._installed_cupy_builds() == ["cupy-cuda13x"]


def test_the_check_can_be_bypassed(monkeypatch):
    monkeypatch.setenv("GEOSWE_ALLOW_MULTIPLE_CUPY", "1")
    monkeypatch.setattr("importlib.metadata.distributions", _fake("cupy", "cupy-cuda12x"))
    backend._check_single_cupy()

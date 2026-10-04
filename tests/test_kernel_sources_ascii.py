"""CUDA kernel sources must be ASCII.

CuPy writes the source to a file with the process encoding before calling NVRTC
(``cupy/cuda/compiler.py``), and importing the CUDA stack can leave the process in
the C locale. A Greek sigma in a kernel comment then aborts every kernel build with
``UnicodeEncodeError``, which is how this was found: CuPy 13.6 on a CUDA 13 host.
The check reads the shipped sources, so it needs neither a GPU nor CuPy.
"""
import ast
from pathlib import Path

import geoswe

CUDA_MARKERS = ("__global__", 'extern "C"', "__device__", "#define", "threadIdx", "blockIdx")


def _kernel_sources():
    for path in sorted(Path(geoswe.__file__).parent.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and any(marker in node.value for marker in CUDA_MARKERS)):
                yield path, node.lineno, node.value


def test_cuda_kernel_sources_are_ascii():
    offenders = [(p.name, line, sorted({c for c in src if ord(c) > 127}))
                 for p, line, src in _kernel_sources() if not src.isascii()]
    assert not offenders, (
        "non-ASCII characters in CUDA kernel source break the build when the process "
        f"encoding is not UTF-8: {offenders}")


def test_the_check_sees_the_kernel_sources():
    """Guard the guard: a typo in the markers would make the test vacuous."""
    assert sum(1 for _ in _kernel_sources()) > 5

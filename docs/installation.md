# Installation

GeoSWE needs only **NumPy** to run on the CPU. GPU acceleration, multi-GPU runs,
and GeoTIFF I/O are opt-in extras. Python 3.10 or newer is required; the test
suite runs on 3.10, 3.12 and 3.13, on Linux and on macOS.

## pip

An unpinned install gives the newest release. For work whose numbers you intend to
publish, pin the version you ran instead (`pip install geoswe==1.1.0`) and record it
beside the results: releases change what the solver computes, and
[citing](citing.md) explains what to put in the paper.

```bash
# CPU only (NumPy backend): the solver and the examples
pip install geoswe

# NVIDIA GPU (CUDA 12 or CUDA 13 drivers)
pip install "geoswe[gpu]"

# AMD GPU (ROCm 7.x)
pip install "geoswe[gpu-rocm]"

# everything a run needs: GPU + MPI + GeoTIFF I/O + CSV forcings + the examples' plots
pip install "geoswe[all]"
```

```{note}
The distribution, repository, and import name are all **`geoswe`**. The
development version installs straight from the repository:
`pip install "geoswe[gpu] @ git+https://github.com/GeoSWE/geoswe.git"`.
```

From a source checkout, which also gives you the examples, the bundled terrain
and the tests:

```bash
git clone https://github.com/GeoSWE/geoswe.git
cd geoswe
pip install -e ".[all]"
```

### Extras

| Extra | Pulls in | Needed for |
|---|---|---|
| `gpu` | `cupy-cuda12x[ctk]`, `scipy` | GPU acceleration (dense **and** compressed solvers) on CUDA 12 and CUDA 13 drivers |
| `gpu-cuda13` | `cupy-cuda13x[ctk]`, `scipy` | the same with the CUDA 13 build of CuPy, if you prefer it |
| `gpu-rocm` | `cupy-rocm-7-0`, `scipy` | GPU acceleration on AMD GPUs with ROCm 7; see [AMD GPUs](amd_gpus.md) |
| `mpi` | `mpi4py` | multi-GPU / distributed runs (halo exchange) |
| `io` | `rasterio`, `pyproj` | reading DEMs and writing flood GeoTIFFs |
| `forcings` | `pandas`, `scipy` | CSV rainfall/tide ingestion, case conditioning, the high-level runner |
| `examples` | `matplotlib` | the plots the examples and the benchmark scripts draw |
| `docs` | `sphinx`, `myst-parser`, … | building this documentation |
| `test` | `pytest`, `scipy` | running the test suite |

`all` is `gpu`, `mpi`, `io`, `forcings` and `examples` together; `docs` and `test`
are not part of it. Without `examples` the examples still run and print their
numbers, and say that they are skipping their plot.

```{note}
Install **one** CuPy build per environment. The `gpu` extra ships the CUDA
headers CuPy compiles against, so it works on CUDA 12 and CUDA 13 machines alike,
with or without a system CUDA toolkit; a CUDA 13 machine does not need the CUDA 13
build. To switch builds anyway, remove the old one first
(`pip uninstall -y cupy-cuda12x cupy-cuda13x`), then install the extra you want.
GeoSWE refuses the GPU backend when two CuPy builds are installed, because they
overwrite each other; that includes a ROCm build next to a CUDA one. On **CUDA 11**,
install GeoSWE without the `gpu` extra and add `cupy-cuda11x` yourself.
```

```{note}
The `all` extra installs the NVIDIA build. On an AMD GPU, name the extras:
`pip install "geoswe[gpu-rocm,mpi,io,forcings]"`. CuPy's ROCm build compiles
kernels with the machine's own ROCm installation, so ROCm must be installed and on
the path at run time. [AMD GPUs](amd_gpus.md) has the details and the settings for
OLCF Frontier.
```

## conda

A development environment (GPU build) is provided:

```bash
conda env create -f environment.yml
conda activate geoswe
pip install -e ".[all]"
```

## Choosing the backend

GeoSWE picks its array backend at import time from the `GEOSWE_BACKEND`
environment variable:

| `GEOSWE_BACKEND` | Behaviour |
|---|---|
| unset | the GPU (CuPy) when CuPy and a GPU, NVIDIA or AMD, are present, otherwise NumPy on the CPU |
| `cupy` | the GPU; an error if no device is visible, and a warning with the NumPy backend if CuPy is not installed |
| `numpy` | the NumPy CPU backend |

Precision follows the backend unless `Config(dtype=...)` says otherwise:
`float32` on the GPU, `float64` on the CPU.

```bash
GEOSWE_BACKEND=numpy python my_script.py     # force CPU
```

The choice is made once, when `geoswe` is first imported, because several
modules specialize for the backend at import time. From Python, set the variable
before the import:

```python
import os
os.environ["GEOSWE_BACKEND"] = "numpy"       # or "cupy"; before importing geoswe
import geoswe
print(geoswe.get_backend(), geoswe.USING_CUPY)
print(geoswe.gpu_platform())                 # "cuda" (NVIDIA), "hip" (AMD ROCm), or None on the CPU
```

`geoswe.set_backend(...)` only works before any solver module has been
imported; calling it later raises a `RuntimeError` pointing you back to
`GEOSWE_BACKEND`.

## Verifying the install

```bash
python -c "import geoswe; print(geoswe.__version__, geoswe.get_backend())"
pip install ".[test]" && pytest             # from a source checkout; GPU tests skip unless a GPU is free
```

Some tests skip without an optional tool, and `pytest -rs` lists each skip with its
reason. `pip install ".[test,io,forcings]"` adds the GeoTIFF and CSV ingestion
tests, and the check that compiles every CUDA kernel source runs only where `nvcc`
is on `PATH` or under `CUDA_PATH` or `CUDA_HOME` (it needs the compiler, not a
device).

```{tip}
`Failed to find CUDA headers` means CuPy can run on the device but cannot compile
kernels: CuPy was installed without its headers (for example `pip install
cupy-cuda12x` directly, or CuPy 13) and the machine has no CUDA toolkit. It does
not mean you need a different CUDA build. Reinstall through the extra
(`gpu`), add the headers (`pip install "cupy-cuda12x[ctk]"`),
or point `CUDA_PATH` at a CUDA toolkit installation.
```

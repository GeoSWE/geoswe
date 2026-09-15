# Installation

GeoSWE needs only **NumPy** to run on the CPU. GPU acceleration, multi-GPU runs,
and GeoTIFF I/O are opt-in extras.

## pip

```bash
# CPU-only (NumPy backend) — enough for the examples, tests, and docs
pip install geoswe

# GPU (CUDA 12.x)
pip install "geoswe[gpu]"

# everything: GPU + MPI + GeoTIFF I/O + CSV forcings
pip install "geoswe[gpu,mpi,io,forcings]"
```

```{note}
The distribution, repository, and import name are all **`geoswe`** — one token
everywhere.
```

From a source checkout:

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
| `mpi` | `mpi4py` | multi-GPU / distributed runs (halo exchange) |
| `io` | `rasterio`, `pyproj` | reading DEMs and writing flood GeoTIFFs |
| `forcings` | `pandas`, `scipy` | CSV rainfall/tide ingestion, case conditioning, the high-level runner |
| `docs` | `sphinx`, `myst-parser`, … | building this documentation |
| `test` | `pytest` | running the test suite |

```{note}
Install **one** CuPy build per environment. The `gpu` extra ships the CUDA
headers CuPy compiles against, so it works on CUDA 12 and CUDA 13 machines alike,
with or without a system CUDA toolkit; a CUDA 13 machine does not need the CUDA 13
build. To switch builds anyway, remove the old one first
(`pip uninstall -y cupy-cuda12x cupy-cuda13x`), then install the extra you want.
GeoSWE refuses the GPU backend when two CuPy builds are installed, because they
overwrite each other. On **CUDA 11**, install `cupy-cuda11x` manually:
`pip install geoswe && pip install cupy-cuda11x`.
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
| unset / `cupy` | use CuPy (GPU); **fall back to NumPy** if CuPy can't be imported |
| `numpy` | force the NumPy CPU backend |

```bash
GEOSWE_BACKEND=numpy python my_script.py     # force CPU
```

The environment variable is the **reliable** mechanism: several modules
specialise for the backend at import time, so the choice must be made before
`import geoswe`. From Python, set it before the import:

```python
import os
os.environ["GEOSWE_BACKEND"] = "numpy"       # or "cupy" — BEFORE importing geoswe
import geoswe
print(geoswe.get_backend(), geoswe.USING_CUPY)
```

`geoswe.set_backend(...)` only works before any solver module has been
imported; calling it later raises a `RuntimeError` pointing you back to
`GEOSWE_BACKEND`.

## Verifying the install

```bash
GEOSWE_BACKEND=numpy python -c "import geoswe; print(geoswe.__version__)"
pip install ".[test]" && pytest -q          # CPU tests pass; GPU tests skip unless a GPU is free
```

```{tip}
`Failed to find CUDA headers` means CuPy can run on the device but cannot compile
kernels: CuPy was installed without its headers (for example `pip install
cupy-cuda12x` directly, or CuPy 13) and the machine has no CUDA toolkit. It does
not mean you need a different CUDA build. Reinstall through the extra
(`pip install "geoswe[gpu]"`), add the headers (`pip install "cupy-cuda12x[ctk]"`),
or point `CUDA_PATH` at a CUDA toolkit installation.
```

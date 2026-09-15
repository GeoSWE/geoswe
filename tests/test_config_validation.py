"""Config enum validation matrix: every enum field x one typo -> ValueError.

Guards against silent misconfiguration: wb_method / pde / sigma_bc / dtype
must all pass Config's _enum_ok validation.
"""
import numpy as np
import pytest
from geoswe import Mesh2D, Config, Solver2D

# Valid values mirror the _enum_ok table in solver.py (flux is currently
# {'lf','hllc'} there — use two valid + one invalid per the contract).
VALID = {
    "time": ["euler", "ssprk3"],
    "flux": ["lf", "hllc"],
    "recon": ["first", "muscl", "linear2", "linear3", "linear5", "weno5"],
    "wb_method": ["audusse", "srm"],
    "pde": ["igr", "baseline"],
    "sigma_bc": ["neumann", "periodic"],
    "dtype": ["float32", "float64"],
    "rk_storage": ["low_storage", "high_storage"],
    "sigma_stages": ["all", "first"],
}

TYPOS = {
    "time": "rk3",
    "flux": "hllcc",
    "recon": "weno7",
    "wb_method": "audusse_srm",
    "pde": "igr2",
    "sigma_bc": "dirichlet",
    "dtype": "float16",
    "rk_storage": "lowstorage",
    "sigma_stages": "second",
}


@pytest.mark.parametrize("field", sorted(VALID))
def test_enum_typo_raises_valueerror(field):
    with pytest.raises(ValueError):
        Config(**{field: TYPOS[field]})


@pytest.mark.parametrize(
    "field,value",
    [(f, v) for f in sorted(VALID) for v in VALID[f]],
)
def test_valid_enum_value_constructs(field, value):
    cfg = Config(**{field: value})
    assert getattr(cfg, field) == value


def test_solver2d_with_typo_config_raises():
    # The contract is that a typo cannot survive to Solver construction:
    # Config raises in __post_init__, so Solver2D(...) with a typo'd kwarg
    # config can never be built.
    mesh = Mesh2D(nx=8, ny=8, dx=1.0, dy=1.0, ngh=4)
    with pytest.raises(ValueError):
        cfg = Config(flux="roe")  # invalid -> raises here
        q0 = np.zeros((3, 8, 8)); q0[0] = 1.0
        Solver2D(mesh, cfg, q0, np.zeros((8, 8)))

# API reference

Auto-generated from the package docstrings. The most-used classes are
{py:class}`~geoswe.Mesh2D`, {py:class}`~geoswe.Config`, and
{py:class}`~geoswe.Solver2D`.

A name re-exported at the top level is documented in that form, because that is
how you import it and how every page writes it: {py:class}`~geoswe.Solver2D`,
not `geoswe.solver.Solver2D`. What is not re-exported keeps its module, so
`run_cached` is {py:func}`geoswe.compressed_solver.run_cached`.

```{toctree}
:maxdepth: 2
:caption: The solver

solver
mesh
forcing
compressed
runlib
```

The three modules below prepare the inputs of a run and collect its output at
gauge points. They ship in the wheel and are supported API as of 1.1.0, under the
same compatibility promise as the solver, but they are not the solver: a case you
already hold as NumPy arrays needs none of them. Two of them want extras, `io`
and `forcings`; each page says which, and `gauges` needs neither.

```{toctree}
:maxdepth: 2
:caption: Preparing inputs

data_prep
io
gauges
```

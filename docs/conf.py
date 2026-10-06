"""Sphinx configuration for the GeoSWE documentation."""
import os
import sys

# Force the NumPy backend so importing `geoswe` during autodoc never touches CuPy.
os.environ.setdefault("GEOSWE_BACKEND", "numpy")

# Make the package importable when building from a source checkout without install.
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information ------------------------------------------------------
project = "GeoSWE"
author = "Peng Chen"
copyright = "2026, Peng Chen and the GeoSWE contributors"
release = "1.1.1"
version = "1.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",              # Markdown source
    "sphinx.ext.autodoc",       # pull docstrings from the package
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # NumPy/Google-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",       # render the governing-equations math
    "sphinx_copybutton",
    "sphinx_design",
]

# Document the GPU/MPI/GIS modules without those stacks installed (e.g. on RTD).
# compressed_solver hard-imports cupy at module top; mpi_halo needs mpi4py; etc.
autodoc_mock_imports = ["cupy", "mpi4py", "rasterio", "pyproj", "pandas", "scipy"]
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Unresolved Python cross-references are silent by default, which is how every
# `{py:class}`/`{py:meth}` reference in this tree came to render as plain monospace
# instead of a link. nitpicky makes each one a warning and -W makes it a failure.
# Keep the list below to targets that cannot resolve, and say why: an entry added to
# quiet a build hides the next real broken link.
nitpicky = True
nitpick_ignore = [
    # solver.py:1623 reads `mask : ndarray of bool, or None to clear.`, so napoleon
    # takes the whole description as the type and hands the Python domain two targets
    # that no object has. Writing that one line as `ndarray of bool or None`, with
    # "None clears the mask" in the description below it, retires both entries.
    ("py:class", "ndarray"),
    ("py:class", "None to clear."),
    # `np.ndarray` in the Solver1D/Solver2D constructor annotations: solver.py imports
    # the backend array module as `np`, so the annotation is rendered as written and
    # there is no object of that name to point at. numpy's own `numpy.ndarray` resolves.
    ("py:class", "np.ndarray"),
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "dev"]   # docs/dev is internal

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = "GeoSWE"
html_static_path = []
html_baseurl = "https://geoswe.readthedocs.io/en/latest/"


# -- Unreleased Config fields ----------------------------------------------------
# The Config fields of an unreleased PDE option are kept out of the rendered API
# (signature and parameter list here; the member list via :exclude-members: in
# api/solver.rst) until that option is documented. Delete this block to show them.
_HIDDEN_CONFIG_FIELDS = {"alpha", "sigma_max_iter", "sigma_tol", "sigma_bc",
                         "sigma_h_min", "sigma_stages", "sigma_halo_every"}
# Both names Config can be documented under: the re-exported `geoswe.Config`, which
# api/solver.rst uses, and the defining `geoswe.solver.Config`. The hook is keyed on
# the name autodoc reports, so a page that moves the class between the two silently
# publishes the hidden fields unless both are listed here.
_CONFIG_NAMES = {"geoswe.Config", "geoswe.solver.Config"}


def _hide_config_fields(app, what, name, obj, options, signature, return_annotation):
    if name not in _CONFIG_NAMES:
        return None
    import inspect
    stores = []
    temp = getattr(app.env, "temp_data", None)
    if isinstance(temp, dict):
        stores.append(temp.get("annotations", {}))
    doc = getattr(app.env, "current_document", None)
    if doc is not None:
        stores.append(getattr(doc, "autodoc_annotations", {}))
    for store in stores:
        for key in _HIDDEN_CONFIG_FIELDS:
            store.get(name, {}).pop(key, None)
    sig = inspect.signature(obj)
    params = [p.replace(annotation=inspect.Parameter.empty)
              for n, p in sig.parameters.items() if n not in _HIDDEN_CONFIG_FIELDS]
    return str(sig.replace(parameters=params, return_annotation=inspect.Signature.empty)), return_annotation


def setup(app):
    # priority 600 runs after sphinx.ext.autodoc.typehints records the annotations
    app.connect("autodoc-process-signature", _hide_config_fields, priority=600)

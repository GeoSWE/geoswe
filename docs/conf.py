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
release = "1.0.0"
version = "1.0"

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

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "dev"]   # docs/dev is internal

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = "GeoSWE"
html_static_path = []
html_baseurl = "https://geoswe.github.io/geoswe/"


# -- Unreleased Config fields ----------------------------------------------------
# The Config fields of an unreleased PDE option are kept out of the rendered API
# (signature and parameter list here; the member list via :exclude-members: in
# api/solver.rst) until that option is documented. Delete this block to show them.
_HIDDEN_CONFIG_FIELDS = {"alpha", "sigma_max_iter", "sigma_tol", "sigma_bc",
                         "sigma_h_min", "sigma_stages", "sigma_halo_every"}


def _hide_config_fields(app, what, name, obj, options, signature, return_annotation):
    if name != "geoswe.solver.Config":
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

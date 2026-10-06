High-level runner (``runlib``)
==============================

``geoswe.runlib`` is the coastal surge and rain runner behind the paper's
county-to-continental applications. It loads a prepared case (bed, Manning,
active mask, ring boundary), wires the forcings, and drives the dense or
compressed solver. It is a library, not a command: no module in it has a
``__main__`` guard, and the scripts under ``benchmark/pinellas_3m/``
(``run_pinellas_mpi.py``) show how a case runner calls it. It assumes a coastal
case near sea level (the bed is clipped to -10..50 m, a coastal ring and sponge
are applied, Green-Ampt infiltration is on in fp32 unless ``GEOSWE_GA=0``), so
for other problems build the solvers directly, as in the flood tutorial.

``import geoswe.runlib`` works on a NumPy-only install. ``case``, ``cli`` and
``replay`` import their optional dependencies inside the functions that need
them; ``driver``, which needs CuPy, mpi4py and pandas at import, is the one
submodule the package does not import for you, so ask for it explicitly with
``from geoswe.runlib import driver``.

Case loading
------------

.. module:: geoswe.runlib.case

.. autofunction:: load_case

Options of the case runners
---------------------------

.. module:: geoswe.runlib.cli

.. autofunction:: build_parser
.. autofunction:: parse

Driver and cache replay
-----------------------

.. module:: geoswe.runlib.driver

.. autofunction:: main

.. module:: geoswe.runlib.replay

``build_cached_parser`` is where the replay entry point's own options are
defined, the checkpoint and wall-clock ones among them
(``--checkpoint-every-h``, ``--ckpt-dir``, ``--resume``, ``--max-wall-min``,
``--stop-at-epoch``, ``--stop-buffer-min``).

.. autofunction:: main
.. autofunction:: build_cached_parser

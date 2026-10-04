High-level runner (``runlib``)
==============================

``geoswe.runlib`` is the coastal surge and rain runner behind the paper's
county-to-continental applications. It loads a prepared case (bed, Manning,
active mask, ring boundary), wires the forcings, and drives the dense or
compressed solver. It is a library, not a command: the scripts under
``benchmark/pinellas_3m/`` (``run_pinellas_mpi.py``) show how a case runner calls
it. It assumes a coastal case near sea level (the bed is clipped to -10..50 m, a
coastal ring and sponge are applied, Green-Ampt infiltration is on by default),
so for other problems build the solvers directly, as in the flood tutorial.

Case loading
------------

.. currentmodule:: geoswe.runlib.case

.. autofunction:: load_case

Options of the case runners
---------------------------

.. currentmodule:: geoswe.runlib.cli

.. autofunction:: build_parser
.. autofunction:: parse

Driver and cache replay
-----------------------

.. currentmodule:: geoswe.runlib.driver

.. autofunction:: main

.. currentmodule:: geoswe.runlib.replay

.. autofunction:: main

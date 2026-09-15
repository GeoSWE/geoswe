High-level runner (``runlib``)
==============================

``geoswe.runlib`` is the operational coastal-surge + rain runner used by the
county-to-continental applications. It loads a prepared case (bed, Manning,
active mask, ring boundary), wires the forcings, and drives the dense or
compressed solver. Most users build solvers directly; ``runlib`` is for full
georeferenced case runs.

Case loading
------------

.. currentmodule:: geoswe.runlib.case

.. autofunction:: load_case

Command-line options
--------------------

.. currentmodule:: geoswe.runlib.cli

.. autofunction:: build_parser
.. autofunction:: parse

Driver and cache replay
-----------------------

.. currentmodule:: geoswe.runlib.driver

.. autofunction:: main

.. currentmodule:: geoswe.runlib.replay

.. autofunction:: main

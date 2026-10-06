Compressed active-cell solver
=============================

GPU-only. See :doc:`../compressed_mesh` for the concepts.

.. currentmodule:: geoswe

``geoswe.CompressedSolver`` is resolved lazily, because ``geoswe.compressed_solver``
imports CuPy at module load: the name exists on a CPU-only install and raises
``ImportError`` naming the ``gpu`` extra only when you touch it. For the same
reason it is deliberately left out of ``geoswe.__all__``, so that
``from geoswe import *`` keeps working without a GPU.

.. autoclass:: CompressedSolver
   :members:
   :member-order: bysource

Cache replay
------------

.. py:module:: geoswe.compressed_solver
.. py:currentmodule:: geoswe

..
   The pair above registers the module as a link target and then puts the
   reference context back to ``geoswe``, the module the classes on this page are
   documented under. ``run_cached``'s docstring writes ``CompressedSolver``
   unqualified, which resolves only while that is the current module, so the
   directives below name their own module in full instead of switching it.

``run_cached`` runs a cache written by :meth:`CompressedSolver.save_cache` without
rebuilding the mesh. It is not re-exported at the top level: import it as
``from geoswe.compressed_solver import run_cached``.

.. autofunction:: geoswe.compressed_solver.run_cached

The compressed mesh
-------------------

.. py:module:: geoswe.compressed_mesh
.. py:currentmodule:: geoswe

The flat active-cell layout the solver is packed into. Reading it is a
measurement, not part of running a case: ``memory_summary`` sizes the
indirection tables, and ``pack``/``unpack`` move a padded ``(nx + 2*ngh, ny +
2*ngh)`` array in and out of the flat ``(N_active,)`` one.

.. autoclass:: geoswe.compressed_mesh.CompressedMesh2D
   :members:

Solvers and configuration
==========================

.. currentmodule:: geoswe

The three classes a run is built from. All three are defined in
``geoswe.solver`` and re-exported at the top level, which is the form this page
documents them in.

.. autoclass:: Config
   :members:
   :exclude-members: alpha, sigma_max_iter, sigma_tol, sigma_bc, sigma_h_min, sigma_stages, sigma_halo_every

.. autoclass:: Solver2D
   :members:
   :member-order: bysource

.. autoclass:: Solver1D
   :members:
   :member-order: bysource

Gauge time series
=================

.. module:: geoswe.gauges

``geoswe.gauges`` records depth, velocity and water-surface elevation at fixed
cells and writes one ``gauge_<name>.csv`` per gauge, which is how a run is
compared against a tide or stream gauge. It is an output helper and not part of
the solver: nothing in ``geoswe.solver`` or the run driver calls it, so you drive
it yourself, from the step callback. NumPy is all it needs.

Everything here indexes the solver's **padded** arrays. ``GaugeRecorder.i`` and
``.j`` are interior indices plus ``mesh.ngh``, so the state and bed to sample are
``solver.q`` and ``solver.b``, not ``solver.q_interior``, ``solver.depth()`` and
the interior bed you handed ``Solver2D``. ``gauges_from_coords`` checks the bed
it is handed and stamps the shape on each recorder, which then checks every
sample, because the interior bed is the plausible mistake: it used to place the
gauge ``ngh`` cells away and report that cell's water, silently unless the gauge
sat within ``ngh`` of the far edge.

.. code-block:: python

   import numpy as np
   from geoswe import Config, Mesh2D, Solver2D
   from geoswe.gauges import GaugeBank, gauges_from_coords

   mesh = Mesh2D(nx=20, ny=20, dx=10.0, dy=10.0)
   bed = np.zeros((20, 20))
   q0 = np.zeros((3, 20, 20)); q0[0] = 1.0
   solver = Solver2D(mesh, Config(), q0, bed)

   # x0, y0 are the outer edge of cell (0, 0) in the bed's own CRS, and
   # solver.b is the padded bed, so the returned (i, j) are padded too.
   bank = GaugeBank(gauges_from_coords([("G1", 105.0, 105.0)], mesh, solver.b,
                                       x0=0.0, y0=0.0, dx=mesh.dx),
                    every_s=60.0)
   solver.run(t_end=600.0, callback=lambda s, step: bank.step(s.t, s.q, s.b))
   print(bank.write_all("gauges_out"))

Under MPI, give each rank its own rank-local mesh and interior origin:
``gauges_from_coords`` has no way to tell a global origin from a local one, and
with a global one every rank places the gauge at the same local cell and then
overwrites the same CSV.

.. autofunction:: gauges_from_coords

.. autoclass:: GaugeBank
   :members:

.. autoclass:: GaugeRecorder
   :members:

Forcings
========

.. currentmodule:: geoswe

Rainfall and a prescribed water level, the two forcings this module carries.
Both bisect the ``time_s`` array they are handed, so both refuse one that steps
backwards; the CSV constructors sort instead and warn, on the grounds that the
file is not the caller's to fix. A river hydrograph is not here: it enters
through :meth:`~geoswe.Solver2D.add_inflow`. See :doc:`../userguide/forcings`
for the physics and the units.

.. autoclass:: RainfallForcing
   :members:

.. autoclass:: StageBoundary
   :members:

Fetching a tide-gauge record
----------------------------

``download_noaa_tide_csv`` is not re-exported at the top level: import it as
``from geoswe.forcing import download_noaa_tide_csv``. It needs the ``forcings``
extra, for the two CO-OPS date strings it formats with pandas.

.. autofunction:: geoswe.forcing.download_noaa_tide_csv

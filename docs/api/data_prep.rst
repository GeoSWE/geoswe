Case preparation from raw data
==============================

.. module:: geoswe.data_prep

``geoswe.data_prep`` turns the files an agency publishes into the arrays a
``Solver2D`` run takes: USGS 3DEP topography and NOAA CUDEM bathymetry into one
projected bed, NLCD land cover into a Manning field, a NOAA CO-OPS CSV into a
stage series, MRMS GRIB2 files into a rainfall series. Each function below hands
back plain arrays; ``CaseData`` is a container to collect them in, which nothing
in the package returns, so fill it yourself or pass the arrays to ``Solver2D``
directly.

Needs the ``io`` extra (rasterio, for the raster paths) and the ``forcings``
extra (pandas for the CSV and GRIB stamps, scipy for ``clean_dem``)::

    pip install "geoswe[io,forcings]"

Every one of those imports is function-local, so ``import geoswe.data_prep``
works without them and the function you call names what it needs.
``mrms_to_uniform_timeseries`` needs more than either extra carries: xarray with
the cfgrib engine and cfgrib's own eccodes library, installed by hand. It raises
rather than handing back a series a run cannot tell from a dry forecast.

Both of the DEM filters report what they moved, because a bed that was reshaped
in silence is a run that succeeds on the wrong terrain. ``clip_range`` defaults
to ``None`` and clips nothing; when you do pass a range,
``merge_dems_to_grid``'s ``meta`` carries ``n_clipped`` and ``clipped_frac``
beside ``n_denormal`` and ``valid_frac``, and ``clean_dem`` warns with the same
numbers and the DEM's own elevation range.

The grid
--------

.. autoclass:: CaseData
   :members:

.. autofunction:: merge_dems_to_grid
.. autofunction:: clean_dem
.. autofunction:: landcover_to_manning_on_grid

The forcings and the coast
--------------------------

.. autofunction:: load_noaa_tide_csv
.. autofunction:: detect_coastline_cells
.. autofunction:: mrms_to_uniform_timeseries

GeoTIFF input and output
========================

.. module:: geoswe.io_geotiff

``geoswe.io_geotiff`` reads the rasters a case is built from and writes the
rasters a run produces, so that QGIS or ArcGIS can open them. The solver itself
never calls it; the run drivers do, for the depth frames and the end-of-run
rasters. A case whose bed comes from a ``.npy`` file never needs it at all.

Needs the ``io`` extra, which installs rasterio::

    pip install "geoswe[io]"

``read_geotiff`` imports rasterio unguarded, so a missing one arrives as
``ModuleNotFoundError: No module named 'rasterio'``. ``write_geotiff`` catches
it and raises ``writing GeoTIFFs needs rasterio (the io extra): pip install
rasterio`` instead. Install the extra before a long compressed run, not after:
the compressed step loop writes ``max_depth.tif`` and ``final_depth.tif``
through ``write_geotiff`` once the loop is over, so a missing rasterio surfaces
at the end of the run rather than the start.

Reading and writing
-------------------

.. autoclass:: GeoArray
   :members:

.. autofunction:: read_geotiff
.. autofunction:: write_geotiff

Reprojection and roughness
--------------------------

``reproject_to_utm`` defaults to bilinear resampling, which is right for a DEM
and wrong for a land-cover raster: an average of two class codes is not a code.
Pass ``resampling="nearest"`` for anything categorical, and read
``nlcd_to_manning``'s warning if you are not sure which you did.

.. autofunction:: reproject_to_utm
.. autofunction:: nlcd_to_manning

.. py:data:: NLCD_TO_MANNING

   Manning's n for each of the 20 NLCD land-cover classes this table covers, as
   a plain dict you can read or copy. The values sit at the rough end of the
   published ranges, which is the operational choice for flood extent, and they
   are not the table the Florida and continental cases used; the comment above
   the table in the source gives the provenance and the scope of each.

.. py:data:: DEFAULT_MANNING
   :value: 0.035

   Manning's n given to a cell whose code matches no class, and to NLCD NoData
   (code 0) and NaN cells.

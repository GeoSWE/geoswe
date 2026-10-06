Meshes and backend
==================

.. currentmodule:: geoswe

.. autoclass:: Mesh2D
   :members:

.. autoclass:: Mesh1D
   :members:

Backend control
---------------

These live in ``geoswe.backend`` and are re-exported at the top level, which is
how every page and example calls them: ``geoswe.get_backend()``.
``geoswe.set_backend`` only takes effect before any solver module has been
imported; see :doc:`../installation`.

.. autofunction:: set_backend
.. autofunction:: get_backend
.. autofunction:: gpu_platform
.. autofunction:: to_host
.. autofunction:: to_device
.. autofunction:: sync

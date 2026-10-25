Partition
=========

.. module:: geoswe.partition

Where a 1xN strip decomposition cuts the split axis: equal shares of the work, or
shares proportional to the measured speed of each rank's device. See
:ref:`devices-of-unequal-speed` for why the second can be faster.

.. autofunction:: strip_bounds

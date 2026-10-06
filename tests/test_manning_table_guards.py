"""What ``Solver2D.set_manning_table`` refuses, and the one thing it only warns about.

The class-table path is the roughness the large runs use, and its four guards (the padded
shape, the uint8 dtype, C-contiguity and the class-id range) sat in the code with nothing
exercising them. Each one stands in front of a kernel that checks nothing itself: the
fused friction kernel indexes the class array with the padded stride and declares
``(const unsigned char*, const float*)``, so a wrong shape, dtype or class id is read as
plausible-but-wrong roughness instead of raising.
"""
import numpy as np
import pytest

from geoswe import Config, Mesh2D, Solver2D

NX, NY, NGH = 12, 10, 2
TABLE = np.array([0.03, 0.05], np.float32)


def solver(friction="manning"):
    mesh = Mesh2D(nx=NX, ny=NY, dx=10.0, dy=10.0, ngh=NGH)
    return Solver2D(mesh, Config(friction=friction), np.zeros((3, NX, NY)), np.zeros((NX, NY)))


def padded(fill=0, dtype=np.uint8):
    return np.full((NX + 2 * NGH, NY + 2 * NGH), fill, dtype)


def test_padded_class_array_is_installed():
    s = solver()
    s.set_manning_table(padded(fill=1), TABLE)
    assert s._manning_cls is not None and int(s._manning_tab.size) == 2


def test_interior_shaped_class_array_is_refused():
    """The kernel strides the class array by the PADDED row length, so an interior-shaped
    one is read as scrambled, partition-dependent roughness."""
    with pytest.raises(ValueError, match="PADDED"):
        solver().set_manning_table(np.zeros((NX, NY), np.uint8), TABLE)


def test_non_uint8_class_array_is_refused():
    with pytest.raises(ValueError, match="uint8"):
        solver().set_manning_table(padded(dtype=np.int32), TABLE)


def test_non_contiguous_class_array_is_refused():
    strided = np.zeros((NX + 2 * NGH, 2 * (NY + 2 * NGH)), np.uint8)[:, ::2]
    assert strided.shape == padded().shape and not strided.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        solver().set_manning_table(strided, TABLE)


def test_class_id_past_the_table_is_refused():
    """Class 2 against a two-entry table is an out-of-bounds device read."""
    with pytest.raises(ValueError, match="out of range"):
        solver().set_manning_table(padded(fill=2), TABLE)


def test_a_table_set_with_friction_off_warns():
    """set_manning switches friction on; set_manning_table does not, so the run would be
    silently frictionless with the roughness the large runs rely on in hand."""
    with pytest.warns(UserWarning, match="friction"):
        solver(friction=None).set_manning_table(padded(), TABLE)

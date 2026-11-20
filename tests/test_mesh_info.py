"""CPU-side logic of MeshData/FunctionData used by the GUI (no browser)."""

import numpy as np
import pytest
from ngsolve import *
from netgen.occ import *

from ngsolve_webgpu import MeshData, FunctionData, RegionState


@pytest.fixture(autouse=True)
def _configure_dirs():
    # override the conftest fixture that opens a browser
    pass


@pytest.fixture(scope="module")
def two_boxes():
    left = Box((0, 0, 0), (1, 1, 1)).mat("left")
    right = Box((1, 0, 0), (2, 1, 1)).mat("right")
    left.faces.Min(X).name = "inlet"
    right.faces.Max(X).name = "outlet"
    return Mesh(OCCGeometry(Glue([left, right])).GenerateMesh(maxh=0.5))


def test_region_state(two_boxes):
    st = RegionState(MeshData(two_boxes))
    alpha = lambda name: st.surf_alphas()[st.boundaries.index(name)]
    interface = next(i for i, d in enumerate(st.fd_doms) if min(d) > 0)
    st.hidden = {"left"}
    assert alpha("inlet") == 0 and alpha("outlet") == 1
    assert st.surf_alphas()[interface] == 1
    st.hidden = {"left", "right"}
    assert st.surf_alphas()[interface] == 0
    st.overrides = {"inlet": True}
    assert alpha("inlet") == 1


def test_compute_range(two_boxes):
    f = FunctionData(MeshData(two_boxes), x, order=1)
    # sampled at integration points: extremes lie inside the elements
    lo, hi = f.compute_range(materials=["left"])
    assert 0 <= lo[1] < 0.3 and 0.7 < hi[1] <= 1
    lo, hi = f.compute_range(boundaries=["outlet"])
    assert lo[1] == pytest.approx(2) and hi[1] == pytest.approx(2)


def test_undeformed():
    md = MeshData(Mesh(unit_square.GenerateMesh(maxh=0.5)))
    md.deformation_data = FunctionData(md, CF((0.1 * x, 0, 0)), order=1)
    P = np.array([0.4, 0.3, 0.0])
    np.testing.assert_allclose(md.undeformed(P * [1.1, 1, 1]), P, atol=1e-8)

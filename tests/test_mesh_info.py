"""CPU-side region logic of MeshData used by the GUI (no browser)."""

import pytest
from ngsolve import *
from netgen.occ import *

from ngsolve_webgpu import MeshData, RegionState


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


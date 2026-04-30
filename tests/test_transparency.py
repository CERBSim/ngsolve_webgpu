"""Visual regression tests for transparent rendering."""

import netgen.occ as occ


def _make_transparent_cube():
    """Create a unit cube with 2 transparent and 4 opaque colored faces."""
    box = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(1, 1, 1))
    colors = [
        (0.7, 0.2, 0.2, 0.4),
        (0.2, 0.7, 0.2, 0.4),
        (0.2, 0.2, 0.7, 1.0),
        (0.7, 0.7, 0.2, 1.0),
        (0.7, 0.2, 0.7, 1.0),
        (0.2, 0.7, 0.7, 1.0),
    ]
    for face, col in zip(box.faces, colors):
        face.col = col
    return occ.OCCGeometry(box)


def _make_uniform_transparent_cube():
    """Create a unit cube with all faces the same transparent color."""
    box = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(1, 1, 1))
    for face in box.faces:
        face.col = (0.2, 0.2, 0.7, 0.4)
    return occ.OCCGeometry(box)


class TestTransparency:
    """Tests for transparent geometry and mesh faces."""

    def test_geometry_transparent(self, webgpu_env):
        """Geometry with some faces set to semi-transparent."""
        import webgpu.jupyter as wj
        from ngsolve_webgpu.geometry import GeometryRenderer

        webgpu_env.ensure_canvas(600, 600)
        geo = _make_transparent_cube()
        renderer = GeometryRenderer(geo)
        renderer.edges.active = False
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "transparent_cube.png")

    def test_mesh_transparent(self, webgpu_env):
        """Mesh without wireframe, same geometry — should match."""
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData, MeshElements2d

        webgpu_env.ensure_canvas(600, 600)
        geo = _make_transparent_cube()
        mesh = ngs.Mesh(geo.GenerateMesh(maxh=10))
        mesh_data = MeshData(mesh)
        elements = MeshElements2d(mesh_data)
        scene = wj.Draw([elements], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "transparent_cube.png", threshold=0.05)

    def test_uniform_transparent_geo(self, webgpu_env):
        """Uniform transparent cube rendered as geometry (baseline)."""
        import webgpu.jupyter as wj
        from ngsolve_webgpu.geometry import GeometryRenderer

        webgpu_env.ensure_canvas(600, 600)
        geo = _make_uniform_transparent_cube()
        renderer = GeometryRenderer(geo)
        renderer.edges.active = False
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "transparent_uniform.png")

    def test_uniform_transparent_mesh(self, webgpu_env):
        """Uniform transparent cube rendered as mesh — must match geometry."""
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData, MeshElements2d

        webgpu_env.ensure_canvas(600, 600)
        geo = _make_uniform_transparent_cube()
        mesh = ngs.Mesh(geo.GenerateMesh(maxh=10))
        mesh_data = MeshData(mesh)
        elements = MeshElements2d(mesh_data)
        scene = wj.Draw([elements], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "transparent_uniform.png", threshold=0.02)

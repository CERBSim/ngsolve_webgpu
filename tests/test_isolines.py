"""Visual regression tests for isoline rendering.

Tests cover:
- IsolineRenderer with show_field=True (color + isolines on top)
- Isolines of one function overlaid on another function's color field
- ClippingIsolineRenderer on 3D clipping plane
- Draw() with isolines kwarg (end-to-end)
"""


class TestIsolineRenderer:
    """IsolineRenderer (2D surface) tests."""

    def test_isolines_show_field(self, webgpu_env):
        """Colored field with isolines drawn on top (show_field=True)."""
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData
        from ngsolve_webgpu.cf import FunctionData
        from ngsolve_webgpu.isolines import IsolineRenderer

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2))
        mesh_data = MeshData(mesh)
        func_data = FunctionData(mesh_data, ngs.sin(3 * ngs.x) * ngs.cos(3 * ngs.y), order=2)

        renderer = IsolineRenderer(func_data, n_lines=10, show_field=True)
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "isolines_show_field.png")

    def test_isolines_overlay_different_function(self, webgpu_env):
        """Isolines of function B overlaid on color field of function A."""
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from webgpu.colormap import Colormap
        from ngsolve_webgpu.mesh import MeshData
        from ngsolve_webgpu.cf import FunctionData, CFRenderer
        from ngsolve_webgpu.isolines import IsolineRenderer

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.15))
        mesh_data = MeshData(mesh)

        # Function A: color field
        cf_color = ngs.x * ngs.y
        func_data_color = FunctionData(mesh_data, cf_color, order=2)
        colormap = Colormap()
        r_color = CFRenderer(func_data_color, colormap=colormap)

        # Function B: isolines overlay
        cf_iso = ngs.sin(5 * ngs.x) * ngs.cos(5 * ngs.y)
        func_data_iso = FunctionData(mesh_data, cf_iso, order=2)
        r_iso = IsolineRenderer(func_data_iso, n_lines=12, show_field=False,
                                color=(1.0, 0.0, 0.0, 1.0))

        scene = wj.Draw([r_color, r_iso], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "isolines_overlay_different_func.png")

    def test_clipping_isolines_show_field(self, webgpu_env):
        """Clipping plane with colored field and isolines."""
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from webgpu.clipping import Clipping
        from ngsolve_webgpu.mesh import MeshData
        from ngsolve_webgpu.cf import FunctionData
        from ngsolve_webgpu.isolines import ClippingIsolineRenderer

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.3))
        mesh_data = MeshData(mesh)
        func_data = FunctionData(mesh_data, ngs.sin(3 * ngs.x) * ngs.y, order=2)

        clipping = Clipping()
        clipping.mode = clipping.Mode.PLANE
        clipping.center = [0.5, 0.5, 0.5]

        renderer = ClippingIsolineRenderer(func_data, clipping=clipping,
                                           n_lines=10, show_field=True)
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "clipping_isolines_show_field.png")

    def test_draw_isolines_3d(self, webgpu_env):
        """Draw() with isolines=8 on a 3D mesh (surface + clipping)."""
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.4))
        scene = Draw(ngs.x * ngs.y, mesh, width=600, height=600,
                     isolines=8, clipping=True)

        webgpu_env.assert_matches_baseline(scene, "draw_isolines_3d.png")

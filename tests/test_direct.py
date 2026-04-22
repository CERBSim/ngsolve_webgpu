"""Visual regression tests for direct renderer construction."""


class TestDirectRenderers:
    """Tests constructing renderers directly (low-level API)."""

    def test_cf_renderer_direct(self, webgpu_env):
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData
        from ngsolve_webgpu.cf import FunctionData, CFRenderer

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        mesh_data = MeshData(mesh)
        function_data = FunctionData(mesh_data, ngs.x * ngs.y, order=2)
        renderer = CFRenderer(function_data)
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "cf_renderer_direct.png")

    def test_clipping_cf_direct(self, webgpu_env):
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData
        from ngsolve_webgpu.cf import FunctionData
        from ngsolve_webgpu.clipping import ClippingCF
        from webgpu.clipping import Clipping

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5))
        mesh_data = MeshData(mesh)
        function_data = FunctionData(mesh_data, ngs.x * ngs.y * ngs.z, order=2)
        clipping = Clipping()
        clipping.mode = clipping.Mode.PLANE
        clipping.center = [0.5, 0.5, 0.5]
        renderer = ClippingCF(function_data, clipping=clipping)
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "clipping_cf_direct.png")

    def test_symmetry_mirror_x(self, webgpu_env):
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData
        from ngsolve_webgpu.cf import FunctionData, CFRenderer
        from ngsolve_webgpu.symmetry import Symmetry

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        mesh_data = MeshData(mesh)
        function_data = FunctionData(mesh_data, ngs.x * ngs.y, order=2)
        sym = Symmetry()
        sym.mirror_x()
        renderer = CFRenderer(function_data, symmetry=sym)
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "symmetry_mirror_x.png")

    def test_isosurface(self, webgpu_env):
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData
        from ngsolve_webgpu.cf import FunctionData
        from ngsolve_webgpu.isosurface import IsoSurfaceRenderer

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.1))
        mesh_data = MeshData(mesh)
        func_data = FunctionData(mesh_data, ngs.x * ngs.y * ngs.z, order=2)
        levelset_data = FunctionData(mesh_data, (ngs.x-0.5)**2 + (ngs.y-0.5)**2 + (ngs.z-0.5)**2 - 0.3**2, order=2)
        renderer = IsoSurfaceRenderer(func_data, levelset_data)
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "isosurface.png")

    def test_draw_geometry(self, webgpu_env):
        import netgen.occ
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        geo = netgen.occ.unit_cube
        scene = Draw(geo, width=600, height=600)

        webgpu_env.assert_matches_baseline(scene, "geometry.png")
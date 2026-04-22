"""Visual regression tests for vector field rendering."""


class TestVectors:
    """Vector field rendering tests."""

    def test_draw_vectors_2d(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        cf = ngs.CF((ngs.x, ngs.y))
        scene = Draw(cf, mesh, width=600, height=600, vectors=True)

        webgpu_env.assert_matches_baseline(scene, "vectors_2d.png")

    def test_draw_vectors_grid_size(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        cf = ngs.CF((ngs.x, ngs.y))
        scene = Draw(cf, mesh, width=600, height=600, vectors={"grid_size": 10})

        webgpu_env.assert_matches_baseline(scene, "vectors_grid_size.png")

    def test_surface_vectors_direct(self, webgpu_env):
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData
        from ngsolve_webgpu.cf import FunctionData
        from ngsolve_webgpu.vectors import SurfaceVectors

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5))
        mesh_data = MeshData(mesh)
        cf = ngs.CF((ngs.x, ngs.y, ngs.z))
        function_data = FunctionData(mesh_data, cf, order=2)
        renderer = SurfaceVectors(function_data, grid_size=20)
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "surface_vectors_direct.png")

    def test_clipping_vectors_direct(self, webgpu_env):
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData
        from ngsolve_webgpu.cf import FunctionData
        from ngsolve_webgpu.vectors import ClippingVectors
        from webgpu.clipping import Clipping

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5))
        mesh_data = MeshData(mesh)
        cf = ngs.CF((ngs.x, ngs.y, ngs.z))
        function_data = FunctionData(mesh_data, cf, order=2)
        clipping = Clipping()
        clipping.mode = clipping.Mode.PLANE
        clipping.center = [0.5, 0.5, 0.5]
        renderer = ClippingVectors(function_data, grid_size=20, clipping=clipping)
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "clipping_vectors_direct.png")

    def test_fieldlines(self, webgpu_env):
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.cf import FieldLines

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5))
        cf = ngs.CF((ngs.x, ngs.y, ngs.z))
        renderer = FieldLines(cf, mesh, num_lines=20)
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "fieldlines.png", threshold=0.05)
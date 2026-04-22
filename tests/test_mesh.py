"""Visual regression tests for mesh rendering."""


class TestMesh:
    """Mesh rendering tests (2D and 3D)."""

    def test_draw_mesh_2d(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        scene = Draw(mesh, width=600, height=600)

        assert scene.bounding_box is not None
        webgpu_env.assert_matches_baseline(scene, "mesh_2d.png")

    def test_draw_mesh_2d_region(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.3))
        scene = Draw(mesh.Boundaries("top|right"), width=600, height=600)

        webgpu_env.assert_matches_baseline(scene, "mesh_2d_region.png")

    def test_draw_mesh_3d(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5))
        scene = Draw(mesh, width=600, height=600)

        webgpu_env.assert_matches_baseline(scene, "mesh_3d.png")

    def test_draw_mesh_3d_shrink(self, webgpu_env):
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData, MeshElements3d

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5))
        mesh_data = MeshData(mesh)
        renderer = MeshElements3d(mesh_data)
        renderer._shrink = 0.8
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "mesh_3d_shrink.png")

    def test_draw_mesh_3d_clipping(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5))
        scene = Draw(mesh, width=600, height=600, clipping=True)

        webgpu_env.assert_matches_baseline(scene, "mesh_3d_clipping.png")

    def test_draw_subdivision(self, webgpu_env):
        import ngsolve as ngs
        from netgen.occ import Circle
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        circle = Circle((0, 0), 1).Face()
        mesh = circle.GenerateMesh(maxh=0.5)
        mesh.Curve(3)
        scene = Draw(mesh, width=600, height=600, subdivision=3)

        webgpu_env.assert_matches_baseline(scene, "subdivision.png")
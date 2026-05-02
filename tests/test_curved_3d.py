"""Visual regression tests for curved 3D element rendering.

These tests verify that curved tet faces (M5) render correctly:
- Curved faces follow the actual geometry (sphere surface)
- Subdivision produces smooth curved triangles
- Normals are correct (proper lighting)
- Shrink mode shows individual curved element faces
"""


class TestCurved3dMesh:
    """Curved 3D mesh element rendering tests."""

    def test_curved_3d_sphere_shrink(self, webgpu_env):
        """Sphere mesh with shrink: curved tet faces should be visible and smooth."""
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from netgen.occ import Sphere, OCCGeometry
        from ngsolve_webgpu.mesh import MeshData, MeshElements3d

        webgpu_env.ensure_canvas(600, 600)
        sphere = Sphere((0, 0, 0), 1)
        ngmesh = OCCGeometry(sphere).GenerateMesh(maxh=0.8)
        mesh = ngs.Mesh(ngmesh)
        mesh.Curve(3)

        mesh_data = MeshData(mesh)
        renderer = MeshElements3d(mesh_data)
        renderer._shrink = 0.8
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "curved_3d_sphere_shrink.png")

    def test_curved_3d_sphere_full(self, webgpu_env):
        """Sphere mesh without shrink: closed curved surface should look like a sphere."""
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from netgen.occ import Sphere, OCCGeometry
        from ngsolve_webgpu.mesh import MeshData, MeshElements3d

        webgpu_env.ensure_canvas(600, 600)
        sphere = Sphere((0, 0, 0), 1)
        ngmesh = OCCGeometry(sphere).GenerateMesh(maxh=0.8)
        mesh = ngs.Mesh(ngmesh)
        mesh.Curve(3)

        mesh_data = MeshData(mesh)
        renderer = MeshElements3d(mesh_data)
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "curved_3d_sphere_full.png")

    def test_curved_3d_sphere_with_hole(self, webgpu_env):
        """Sphere with hole: mix of curved and straight tets, shrunk."""
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from netgen.occ import Sphere, Box, Pnt, OCCGeometry
        from ngsolve_webgpu.mesh import MeshData, MeshElements3d

        webgpu_env.ensure_canvas(600, 600)
        shape = Sphere((0, 0, 0), 1) - Box(Pnt(-0.3, -0.3, -0.3), Pnt(0.3, 0.3, 0.3))
        ngmesh = OCCGeometry(shape).GenerateMesh(maxh=0.4)
        mesh = ngs.Mesh(ngmesh)
        mesh.Curve(3)

        mesh_data = MeshData(mesh)
        renderer = MeshElements3d(mesh_data)
        renderer._shrink = 0.85
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "curved_3d_sphere_hole_shrink.png")

    def test_curved_3d_surface(self, webgpu_env):
        """Sphere boundary surface: curved 2D faces on a 3D mesh via Draw(mesh)."""
        import ngsolve as ngs
        from netgen.occ import Sphere, OCCGeometry
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        sphere = Sphere((0, 0, 0), 1)
        ngmesh = OCCGeometry(sphere).GenerateMesh(maxh=0.8)
        mesh = ngs.Mesh(ngmesh)
        mesh.Curve(3)

        scene = Draw(mesh, width=600, height=600)

        webgpu_env.assert_matches_baseline(scene, "curved_3d_surface.png")

    def test_curved_3d_high_order(self, webgpu_env):
        """Order-4 curved sphere: higher order should give smoother result."""
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from netgen.occ import Sphere, OCCGeometry
        from ngsolve_webgpu.mesh import MeshData, MeshElements3d

        webgpu_env.ensure_canvas(600, 600)
        sphere = Sphere((0, 0, 0), 1)
        ngmesh = OCCGeometry(sphere).GenerateMesh(maxh=1.0)
        mesh = ngs.Mesh(ngmesh)
        mesh.Curve(4)

        mesh_data = MeshData(mesh)
        renderer = MeshElements3d(mesh_data)
        renderer._shrink = 0.8
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "curved_3d_high_order.png")

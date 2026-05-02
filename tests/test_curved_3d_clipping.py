"""Visual regression tests for curved 3D clipping plane rendering (M6).

These tests verify that the clipping plane follows curved geometry:
- Clipping surface is subdivided and curved (not piecewise flat)
- Function values are evaluated correctly on the curved clipping surface
- Works for both curved-only and mixed curved/straight meshes
"""


class TestCurved3dClipping:
    """Curved 3D clipping plane rendering tests."""

    def test_clipping_sphere_function(self, webgpu_env):
        """Clipping plane through curved sphere with sin function should follow sphere curvature."""
        import ngsolve as ngs
        from netgen.occ import Sphere, OCCGeometry
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        sphere = Sphere((0, 0, 0), 1)
        mesh = ngs.Mesh(OCCGeometry(sphere).GenerateMesh(maxh=0.5))
        mesh.Curve(3)

        scene = Draw(ngs.sin(5 * ngs.x), mesh, width=600, height=600, clipping=True)

        webgpu_env.assert_matches_baseline(scene, "curved_3d_clipping_sphere_func.png")

    def test_clipping_sphere_offset(self, webgpu_env):
        """Clipping at an offset: curved boundary of clipping surface should be circular."""
        import ngsolve as ngs
        from netgen.occ import Sphere, OCCGeometry
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        sphere = Sphere((0, 0, 0), 1)
        mesh = ngs.Mesh(OCCGeometry(sphere).GenerateMesh(maxh=0.5))
        mesh.Curve(3)

        scene = Draw(
            ngs.x * ngs.y * ngs.z, mesh, width=600, height=600,
            clipping={"y": 0.3},
        )

        webgpu_env.assert_matches_baseline(scene, "curved_3d_clipping_sphere_offset.png")

    def test_clipping_sphere_high_order(self, webgpu_env):
        """Higher curve order (5) on coarser mesh: subdivision should produce smooth clipping surface."""
        import ngsolve as ngs
        from netgen.occ import Sphere, OCCGeometry
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        sphere = Sphere((0, 0, 0), 1)
        mesh = ngs.Mesh(OCCGeometry(sphere).GenerateMesh(maxh=0.8))
        mesh.Curve(5)

        scene = Draw(ngs.x, mesh, width=600, height=600, clipping=True)

        webgpu_env.assert_matches_baseline(scene, "curved_3d_clipping_high_order.png")

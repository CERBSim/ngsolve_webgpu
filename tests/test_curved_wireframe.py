"""Visual regression tests for curved quad wireframe rendering.

These tests verify that curved quad wireframe has no diagonal stub lines.
The fix reorders the lam computation in the wireframe shader so the diagonal
edge of quads-split-into-triangles comes last and can be collapsed.
"""


def _make_hex_sphere_shell(maxh=0.5, curve_order=3):
    """Create a curved hex sphere shell mesh with quad boundary elements."""
    from netgen.occ import Sphere, Pnt, Glue, IdentificationType, gp_Trsf

    sphere = Sphere(Pnt(0, 0, 0), 1)
    sphere2 = Sphere(Pnt(0, 0, 0), 1.1)
    sphere.faces[0].Identify(
        sphere2.faces[0],
        "cs",
        IdentificationType.CLOSESURFACES,
        trafo=gp_Trsf.Scale((0, 0, 0), 1.1),
    )
    geo = Glue([sphere2 - sphere])
    geo.faces.quad_dominated = True
    mesh = geo.GenerateMesh(maxh=maxh)
    mesh.Curve(curve_order)
    return mesh


class TestCurvedQuadWireframe:
    """Curved quad wireframe tests — no diagonal stub lines."""

    def test_curved_quad_wireframe(self, webgpu_env):
        """Hex sphere shell: wireframe should show clean quad edges, no stubs."""
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = _make_hex_sphere_shell()
        scene = Draw(mesh, width=600, height=600)

        webgpu_env.assert_matches_baseline(scene, "curved_quad_wireframe.png")

    def test_curved_quad_wireframe_subdivision(self, webgpu_env):
        """Hex sphere shell with subdivision=3: smooth curved quad wireframe."""
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = _make_hex_sphere_shell()
        scene = Draw(mesh, width=600, height=600, subdivision=3)

        webgpu_env.assert_matches_baseline(
            scene, "curved_quad_wireframe_subdivision.png"
        )

    def test_curved_quad_wireframe_function(self, webgpu_env):
        """CF on hex sphere shell: wireframe overlay should have no stubs."""
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = _make_hex_sphere_shell()
        scene = Draw(ngs.sin(3 * ngs.x), mesh, width=600, height=600)

        webgpu_env.assert_matches_baseline(
            scene, "curved_quad_wireframe_function.png"
        )

    def test_curved_quad_wireframe_high_order(self, webgpu_env):
        """Order-5 curved hex sphere shell with subdivision=3."""
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = _make_hex_sphere_shell(curve_order=5)
        scene = Draw(mesh, width=600, height=600, subdivision=3)

        webgpu_env.assert_matches_baseline(
            scene, "curved_quad_wireframe_high_order.png"
        )

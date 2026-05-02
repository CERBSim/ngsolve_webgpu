"""Visual regression tests for curved hex and prism element rendering (M7).

These tests verify that curved hexahedral and prismatic elements render correctly:
- Curved hex faces from quad-dominated sphere shell meshes
- Curved prism faces from mixed tet+prism sphere meshes
- Shrink mode, boundary surface, function evaluation, and clipping
"""


def _make_hex_mesh():
    """Sphere shell mesh with hex+prism elements (quad-dominated)."""
    from netgen.occ import Sphere, Pnt, Glue, IdentificationType, gp_Trsf

    sphere = Sphere(Pnt(0, 0, 0), 1)
    sphere2 = Sphere(Pnt(0, 0, 0), 1.1)
    sphere.faces[0].Identify(
        sphere2.faces[0], "cs", IdentificationType.CLOSESURFACES,
        trafo=gp_Trsf.Scale((0, 0, 0), 1.1),
    )
    geo = Glue([sphere2 - sphere])
    geo.faces.quad_dominated = True
    mesh = geo.GenerateMesh(maxh=0.5)
    mesh.Curve(3)
    return mesh


def _make_prism_mesh():
    """Sphere shell + inner sphere mesh with tet+prism elements."""
    from netgen.occ import Sphere, Pnt, Glue, IdentificationType, gp_Trsf

    sphere = Sphere(Pnt(0, 0, 0), 1)
    sphere2 = Sphere(Pnt(0, 0, 0), 1.1)
    sphere.faces[0].Identify(
        sphere2.faces[0], "cs", IdentificationType.CLOSESURFACES,
        trafo=gp_Trsf.Scale((0, 0, 0), 1.1),
    )
    geo = Glue([sphere2 - sphere, sphere])
    mesh = geo.GenerateMesh(maxh=0.5)
    mesh.Curve(3)
    return mesh


class TestCurvedHex:
    """Curved hexahedral element rendering tests."""

    def test_curved_hex_shrink(self, webgpu_env):
        """Hex sphere shell with shrink: curved hex faces should be visible."""
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData, MeshElements3d

        webgpu_env.ensure_canvas(600, 600)
        mesh = _make_hex_mesh()

        mesh_data = MeshData(mesh)
        renderer = MeshElements3d(mesh_data)
        renderer._shrink = 0.8
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "curved_hex_shrink.png")

    def test_curved_hex_surface(self, webgpu_env):
        """Hex sphere shell boundary surface via Draw(mesh)."""
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = _make_hex_mesh()

        scene = Draw(mesh, width=600, height=600)

        webgpu_env.assert_matches_baseline(scene, "curved_hex_surface.png")

    def test_curved_hex_function(self, webgpu_env):
        """Draw sin(3*x) on hex mesh with clipping."""
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = _make_hex_mesh()

        scene = Draw(ngs.sin(3 * ngs.x), mesh, width=600, height=600, clipping=True)

        webgpu_env.assert_matches_baseline(scene, "curved_hex_function.png")

    def test_curved_hex_clipping(self, webgpu_env):
        """Draw x*y*z on hex mesh with clipping at y=0.3 offset."""
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = _make_hex_mesh()

        scene = Draw(
            ngs.x * ngs.y * ngs.z, mesh, width=600, height=600,
            clipping={"y": 0.3},
        )

        webgpu_env.assert_matches_baseline(scene, "curved_hex_clipping.png")


class TestCurvedPrism:
    """Curved prismatic element rendering tests."""

    def test_curved_prism_shrink(self, webgpu_env):
        """Prism+tet mesh with shrink: curved prism faces should be visible."""
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData, MeshElements3d

        webgpu_env.ensure_canvas(600, 600)
        mesh = _make_prism_mesh()

        mesh_data = MeshData(mesh)
        renderer = MeshElements3d(mesh_data)
        renderer._shrink = 0.8
        scene = wj.Draw([renderer], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "curved_prism_shrink.png")

    def test_curved_prism_surface(self, webgpu_env):
        """Prism+tet mesh boundary surface via Draw(mesh)."""
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = _make_prism_mesh()

        scene = Draw(mesh, width=600, height=600)

        webgpu_env.assert_matches_baseline(scene, "curved_prism_surface.png")

    def test_curved_prism_function(self, webgpu_env):
        """Draw sin(3*x) on prism+tet mesh with clipping."""
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = _make_prism_mesh()

        scene = Draw(ngs.sin(3 * ngs.x), mesh, width=600, height=600, clipping=True)

        webgpu_env.assert_matches_baseline(scene, "curved_prism_function.png")

    def test_curved_prism_clipping(self, webgpu_env):
        """Draw x*y*z on prism+tet mesh with clipping at y=0.3 offset."""
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = _make_prism_mesh()

        scene = Draw(
            ngs.x * ngs.y * ngs.z, mesh, width=600, height=600,
            clipping={"y": 0.3},
        )

        webgpu_env.assert_matches_baseline(scene, "curved_prism_clipping.png")

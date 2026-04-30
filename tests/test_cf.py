"""Visual regression tests for CoefficientFunction rendering."""


class TestCoefficientFunction:
    """CoefficientFunction rendering tests."""

    def test_draw_coefficient_function(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        scene = Draw(ngs.x * ngs.y, mesh, width=600, height=600)

        webgpu_env.assert_matches_baseline(scene, "cf_xy.png")

    def test_draw_cf_3d(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5))
        scene = Draw(ngs.x * ngs.y * ngs.z, mesh, width=600, height=600, clipping=True)

        webgpu_env.assert_matches_baseline(scene, "cf_3d.png")

    def test_draw_cf_order1(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        scene = Draw(ngs.sin(10*ngs.x), mesh, width=600, height=600, order=1)

        webgpu_env.assert_matches_baseline(scene, "cf_order1.png")

    def test_draw_cf_order4(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        scene = Draw(ngs.sin(10*ngs.x), mesh, width=600, height=600, order=4)

        webgpu_env.assert_matches_baseline(scene, "cf_order4.png")

    def test_draw_cf_vector(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        cf = ngs.CF((ngs.x, ngs.y))
        scene = Draw(cf, mesh, width=600, height=600)

        webgpu_env.assert_matches_baseline(scene, "cf_vector.png")

    def test_draw_cf_vector_component(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw
        from ngsolve_webgpu.cf import CFRenderer

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        cf = ngs.CF((ngs.x, ngs.y))
        scene = Draw(cf, mesh, width=600, height=600)
        webgpu_env.page.wait_for_timeout(500)

        for ro in scene.render_objects:
            if isinstance(ro, CFRenderer):
                ro.set_component(0)
                break

        scene.render()

        webgpu_env.assert_matches_baseline(scene, "cf_vector_component.png")

    def test_draw_deformation_2d(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        cf = ngs.x * ngs.y
        deformation = ngs.CF((0, ngs.sin(5 * ngs.x), 0))
        scene = Draw(cf, mesh, width=600, height=600, deformation=deformation)

        webgpu_env.assert_matches_baseline(scene, "deformation_2d.png")

    def test_draw_colormap_range(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw
        from ngsolve_webgpu.cf import CFRenderer

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        scene = Draw(ngs.x * ngs.y, mesh, width=600, height=600)
        webgpu_env.page.wait_for_timeout(500)

        for ro in scene.render_objects:
            if isinstance(ro, CFRenderer):
                ro.colormap.autoscale = False
                ro.colormap.set_min_max(0.0, 0.5)
                break

        scene.render()

        webgpu_env.assert_matches_baseline(scene, "colormap_range.png")

    def test_draw_clipping_dict(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5))
        scene = Draw(ngs.x * ngs.y * ngs.z, mesh, width=600, height=600,
                     clipping={"nx": 1, "ny": 0, "nz": 0})

        webgpu_env.assert_matches_baseline(scene, "clipping_dict.png", threshold=0.25)


class TestComplexFields:
    """Regression tests for complex field visualization."""

    def test_complex_scalar_real(self, webgpu_env):
        """Complex scalar field defaults to real part visualization."""
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        fes = ngs.H1(mesh, order=3, complex=True)
        gf = ngs.GridFunction(fes)
        gf.Set(ngs.sin(3 * ngs.x) + 1j * ngs.cos(3 * ngs.y))
        scene = Draw(gf, mesh, width=600, height=600, order=3)

        webgpu_env.assert_matches_baseline(scene, "complex_scalar_real.png")

    def test_complex_scalar_abs(self, webgpu_env):
        """Complex scalar field in absolute value mode."""
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw
        from ngsolve_webgpu.cf import CFRenderer

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        fes = ngs.H1(mesh, order=3, complex=True)
        gf = ngs.GridFunction(fes)
        gf.Set(ngs.sin(3 * ngs.x) + 1j * ngs.cos(3 * ngs.y))
        scene = Draw(gf, mesh, width=600, height=600, order=3)
        webgpu_env.page.wait_for_timeout(500)

        for ro in scene.render_objects:
            if isinstance(ro, CFRenderer):
                ro.set_complex_mode("abs")
                break
        scene.render()

        webgpu_env.assert_matches_baseline(scene, "complex_scalar_abs.png")

    def test_complex_scalar_imag(self, webgpu_env):
        """Complex scalar field in imaginary part mode."""
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw
        from ngsolve_webgpu.cf import CFRenderer

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        fes = ngs.H1(mesh, order=3, complex=True)
        gf = ngs.GridFunction(fes)
        gf.Set(ngs.sin(3 * ngs.x) + 1j * ngs.cos(3 * ngs.y))
        scene = Draw(gf, mesh, width=600, height=600, order=3)
        webgpu_env.page.wait_for_timeout(500)

        for ro in scene.render_objects:
            if isinstance(ro, CFRenderer):
                ro.set_complex_mode("imag")
                break
        scene.render()

        webgpu_env.assert_matches_baseline(scene, "complex_scalar_imag.png")

    def test_complex_scalar_animate(self, webgpu_env):
        """Complex scalar field with a fixed phase rotation."""
        import math
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw
        from ngsolve_webgpu.cf import CFRenderer

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        fes = ngs.H1(mesh, order=3, complex=True)
        gf = ngs.GridFunction(fes)
        gf.Set(ngs.sin(3 * ngs.x) + 1j * ngs.cos(3 * ngs.y))
        scene = Draw(gf, mesh, width=600, height=600, order=3)
        webgpu_env.page.wait_for_timeout(500)

        for ro in scene.render_objects:
            if isinstance(ro, CFRenderer):
                ro.set_phase(math.pi / 4)
                break
        scene.render()

        webgpu_env.assert_matches_baseline(scene, "complex_scalar_animate.png")

    def test_complex_vector(self, webgpu_env):
        """Complex vector field rendering (norm by default)."""
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        fes = ngs.VectorH1(mesh, order=2, complex=True)
        gf = ngs.GridFunction(fes)
        gf.Set(ngs.CF((ngs.x + 1j * ngs.y, 2 * ngs.x - 1j * ngs.y)))
        scene = Draw(gf, mesh, width=600, height=600, order=2)

        webgpu_env.assert_matches_baseline(scene, "complex_vector.png")

    def test_complex_vector_component_imag(self, webgpu_env):
        """Complex vector field: component 1, imaginary part."""
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw
        from ngsolve_webgpu.cf import CFRenderer

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        fes = ngs.VectorH1(mesh, order=2, complex=True)
        gf = ngs.GridFunction(fes)
        gf.Set(ngs.CF((ngs.x + 1j * ngs.y, 2 * ngs.x - 1j * ngs.y)))
        scene = Draw(gf, mesh, width=600, height=600, order=2)
        webgpu_env.page.wait_for_timeout(500)

        for ro in scene.render_objects:
            if isinstance(ro, CFRenderer):
                ro.set_component(1)
                ro.set_complex_mode("imag")
                break
        scene.render()

        webgpu_env.assert_matches_baseline(scene, "complex_vector_comp1_imag.png")
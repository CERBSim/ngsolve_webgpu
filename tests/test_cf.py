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
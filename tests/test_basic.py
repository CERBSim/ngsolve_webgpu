"""Visual regression tests for ngsolve_webgpu rendering."""

import numpy as np

class TestRendering:
    """Full pipeline: NGSolve -> webgpu rendering -> screenshot."""

    def test_draw_mesh_2d(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        canvas_id = webgpu_env.ensure_canvas(600, 600)

        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        scene = Draw(mesh, width=600, height=600)
        webgpu_env.page.wait_for_timeout(500)

        assert scene is not None
        assert len(scene.render_objects) > 0, "Scene has no render objects"
        assert scene.bounding_box is not None, "No bounding box"

        path = webgpu_env.output_dir / "mesh_2d.png"
        webgpu_env.output_dir.mkdir(parents=True, exist_ok=True)
        webgpu_env.readback_texture(scene, path)
        assert path.exists(), "Screenshot file not created"
        webgpu_env.assert_matches_baseline(path, "mesh_2d.png")

    def test_draw_coefficient_function(self, webgpu_env):
        import ngsolve as ngs
        from ngsolve_webgpu.jupyter import Draw

        canvas_id = webgpu_env.ensure_canvas(600, 600)

        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        cf = ngs.x * ngs.y
        scene = Draw(cf, mesh, width=600, height=600)
        webgpu_env.page.wait_for_timeout(500)

        assert scene is not None
        assert len(scene.render_objects) > 0
        assert scene.bounding_box is not None

        path = webgpu_env.output_dir / "cf_xy.png"
        webgpu_env.output_dir.mkdir(parents=True, exist_ok=True)
        webgpu_env.readback_texture(scene, path)
        assert path.exists(), "Screenshot file not created"
        webgpu_env.assert_matches_baseline(path, "cf_xy.png")

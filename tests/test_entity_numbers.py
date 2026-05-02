"""Visual regression tests for EntityNumbers renderer."""


class TestEntityNumbers:
    """Tests for mesh entity number rendering."""

    def test_vertex_numbers_2d(self, webgpu_env):
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData, MeshWireframe2d
        from ngsolve_webgpu.entity_numbers import EntityNumbers

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        mesh_data = MeshData(mesh)
        wireframe = MeshWireframe2d(mesh_data)
        numbers = EntityNumbers(mesh_data, entity="vertices", font_size=15)
        scene = wj.Draw([wireframe, numbers], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "entity_numbers_vertices_2d.png")

    def test_edge_numbers_2d(self, webgpu_env):
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData, MeshWireframe2d
        from ngsolve_webgpu.entity_numbers import EntityNumbers

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        mesh_data = MeshData(mesh)
        wireframe = MeshWireframe2d(mesh_data)
        numbers = EntityNumbers(mesh_data, entity="edges", font_size=12)
        scene = wj.Draw([wireframe, numbers], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "entity_numbers_edges_2d.png")

    def test_surface_element_numbers_2d(self, webgpu_env):
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData, MeshElements2d, MeshWireframe2d
        from ngsolve_webgpu.entity_numbers import EntityNumbers

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        mesh_data = MeshData(mesh)
        surface = MeshElements2d(mesh_data)
        wireframe = MeshWireframe2d(mesh_data)
        numbers = EntityNumbers(mesh_data, entity="surface_elements", font_size=12)
        scene = wj.Draw([surface, wireframe, numbers], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "entity_numbers_surface_elements_2d.png")

    def test_facet_numbers_3d(self, webgpu_env):
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from webgpu.clipping import Clipping
        from ngsolve_webgpu.mesh import MeshData, MeshElements2d, MeshWireframe2d
        from ngsolve_webgpu.entity_numbers import EntityNumbers

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5))
        mesh_data = MeshData(mesh)
        clipping = Clipping()
        clipping.mode = clipping.Mode.PLANE
        clipping.center = [0.5, 0.5, 0.5]
        surface = MeshElements2d(mesh_data, clipping=clipping)
        wireframe = MeshWireframe2d(mesh_data, clipping=clipping)
        numbers = EntityNumbers(mesh_data, entity="facets", font_size=10, clipping=clipping)
        scene = wj.Draw([surface, wireframe, numbers], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "entity_numbers_facets_3d.png")

    def test_volume_element_numbers(self, webgpu_env):
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from webgpu.clipping import Clipping
        from ngsolve_webgpu.mesh import MeshData, MeshElements3d
        from ngsolve_webgpu.entity_numbers import EntityNumbers

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5))
        mesh_data = MeshData(mesh)
        clipping = Clipping()
        clipping.mode = clipping.Mode.PLANE
        clipping.center = [0.5, 0.5, 0.5]
        vol = MeshElements3d(mesh_data, clipping=clipping)
        vol.shrink = 0.1
        numbers = EntityNumbers(mesh_data, entity="volume_elements", font_size=12, clipping=clipping)
        scene = wj.Draw([vol, numbers], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "entity_numbers_volume_elements.png")

    def test_point_numbers_backward_compat(self, webgpu_env):
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData, MeshWireframe2d
        from ngsolve_webgpu.entity_numbers import PointNumbers

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        mesh_data = MeshData(mesh)
        wireframe = MeshWireframe2d(mesh_data)
        numbers = PointNumbers(mesh_data, font_size=15)
        scene = wj.Draw([wireframe, numbers], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "entity_numbers_point_compat.png")

    def test_multiple_entity_types(self, webgpu_env):
        import ngsolve as ngs
        import webgpu.jupyter as wj
        from ngsolve_webgpu.mesh import MeshData, MeshWireframe2d
        from ngsolve_webgpu.entity_numbers import EntityNumbers

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.5))
        mesh_data = MeshData(mesh)
        wireframe = MeshWireframe2d(mesh_data)
        vertex_nums = EntityNumbers(mesh_data, entity="vertices", font_size=18)
        edge_nums = EntityNumbers(mesh_data, entity="edges", font_size=12)
        scene = wj.Draw([wireframe, vertex_nums, edge_nums], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "entity_numbers_multiple.png")

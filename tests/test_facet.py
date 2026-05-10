"""Visual regression tests for facet (element-boundary) CF rendering."""


class TestFacetRendering:
    def test_facet_cf_x(self, webgpu_env):
        """Facet rendering of CF(x) on unit square with wireframe."""
        import ngsolve as ngs
        from ngsolve_webgpu.mesh import MeshData, MeshWireframe2d
        from ngsolve_webgpu.facet_cf import FacetFunctionData, FacetCFRenderer
        from webgpu.colormap import Colormap
        import webgpu.jupyter as wj

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        mdata = MeshData(mesh)
        facet_data = FacetFunctionData(mdata, ngs.x, order=2)
        renderer = FacetCFRenderer(facet_data, colormap=Colormap(minval=0, maxval=1))
        wf = MeshWireframe2d(mdata)
        scene = wj.Draw([renderer, wf], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "facet_cf_x.png")

    def test_facet_cf_sin(self, webgpu_env):
        """Facet rendering of sin(5*x)*y — more colorful, tests higher variation."""
        import ngsolve as ngs
        from ngsolve_webgpu.mesh import MeshData, MeshWireframe2d
        from ngsolve_webgpu.facet_cf import FacetFunctionData, FacetCFRenderer
        from webgpu.colormap import Colormap
        import webgpu.jupyter as wj

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        mdata = MeshData(mesh)
        cf = ngs.sin(5 * ngs.x) * ngs.y
        facet_data = FacetFunctionData(mdata, cf, order=3)
        renderer = FacetCFRenderer(facet_data, colormap=Colormap(minval=-1, maxval=1))
        wf = MeshWireframe2d(mdata)
        scene = wj.Draw([renderer, wf], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "facet_cf_sin.png")

    def test_facet_l2_jumps(self, webgpu_env):
        """Facet rendering of an L2 GridFunction — should show inter-element jumps."""
        import ngsolve as ngs
        from ngsolve_webgpu.mesh import MeshData, MeshWireframe2d
        from ngsolve_webgpu.facet_cf import FacetFunctionData, FacetCFRenderer
        from webgpu.colormap import Colormap
        import webgpu.jupyter as wj

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        fes = ngs.L2(mesh, order=0)
        gf = ngs.GridFunction(fes)
        gf.vec[:].FV().NumPy()[:] = [i/gf.vec.size for i in range(gf.vec.size)]
        mdata = MeshData(mesh)
        facet_data = FacetFunctionData(mdata, gf, order=2)
        renderer = FacetCFRenderer(facet_data, colormap=Colormap(minval=0, maxval=1))
        wf = MeshWireframe2d(mdata)
        scene = wj.Draw([renderer, wf], 600, 600)

        webgpu_env.assert_matches_baseline(scene, "facet_l2_jumps.png")

    def test_facet_data_extraction(self):
        """FacetFunctionData correctly extracts Bernstein coefficients for CF(x)."""
        import ngsolve as ngs
        from ngsolve_webgpu.mesh import MeshData
        from ngsolve_webgpu.facet_cf import FacetFunctionData

        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
        mdata = MeshData(mesh)
        fd = FacetFunctionData(mdata, ngs.x, order=2)

        assert fd.n_edges > 0
        assert fd.minval < fd.maxval
        assert fd.function_buffer[0] == 1.0  # ncomp
        assert fd.function_buffer[1] == 2.0  # order

    def test_facet_3d_data_extraction(self):
        """3D facet data: Bernstein coefficients reproduce the CF on each MtAE face."""
        import ngsolve as ngs
        import numpy as np
        from ngsolve_webgpu.mesh import MeshData
        from ngsolve_webgpu.facet_cf import FacetCFRenderer3D

        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5))
        mdata = MeshData(mesh)
        cf = ngs.CF(ngs.x + 2 * ngs.y + 3 * ngs.z)
        renderer = FacetCFRenderer3D(mdata, cf, order=2)
        renderer._extract_facet_data()

        ndof = 6  # order 2

        def eval_bernstein(coeffs, u, v):
            c = coeffs.copy()
            b = np.array([u, v, 1 - u - v])
            dy = 3
            for n in range(2, 0, -1):
                i0 = 0
                for iy in range(n):
                    for ix in range(n - iy):
                        c[i0+ix] = b[0]*c[i0+ix] + b[1]*c[i0+ix+1] + b[2]*c[i0+ix+dy-iy]
                    i0 += dy - iy
            return c[0]

        # Data is in MtAE order: face g = opposite local vertex g.
        # MtAE bary (xi, eta, 1-xi-eta) -> face vertices sorted by ascending global index.
        max_err = 0.0
        for el_idx in range(mesh.ne):
            el = mesh[ngs.ElementId(ngs.VOL, el_idx)]
            gv = [v.nr for v in el.vertices]
            vp = np.array([list(mesh[v].point) for v in el.vertices])
            for g in range(4):
                coeffs = renderer._facet_data[3 + (el_idx*4+g)*ndof : 3 + (el_idx*4+g+1)*ndof]
                # Face g: vertices are all except local vertex g, sorted by global index
                face_locals = [i for i in range(4) if i != g]
                face_locals.sort(key=lambda i: gv[i])
                p0, p1, p2 = vp[face_locals[0]], vp[face_locals[1]], vp[face_locals[2]]
                for xi, eta in [(0.3, 0.2), (0.5, 0.1), (0.0, 0.5)]:
                    phys = xi * p0 + eta * p1 + (1-xi-eta) * p2
                    expected = phys[0] + 2*phys[1] + 3*phys[2]
                    got = eval_bernstein(coeffs, xi, eta)
                    max_err = max(max_err, abs(got - expected))

        assert max_err < 1e-5, f"max error {max_err} too large"

    def test_facet_3d_visual(self, webgpu_env):
        """3D facet rendering: CF(x) and FacetFESpace GridFunction, with clipping."""
        import ngsolve as ngs
        from ngsolve_webgpu.facet_cf import FacetCFRenderer3D
        from ngsolve_webgpu.mesh import MeshData
        from webgpu.colormap import Colormap
        from webgpu.clipping import Clipping
        import webgpu.jupyter as wj

        webgpu_env.ensure_canvas(600, 600)
        mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5))
        mdata = MeshData(mesh)

        clipping = Clipping()
        clipping.mode = clipping.Mode.PLANE
        clipping.center = [0.5, 0.5, 0.5]
        clipping.normal = [1, 0, 0]

        # CF renderer
        cf_renderer = FacetCFRenderer3D(
            mdata, ngs.x, order=2,
            colormap=Colormap(minval=0, maxval=1),
            clipping=clipping,
        )

        # FacetFESpace GridFunction renderer
        fes = ngs.FacetFESpace(mesh, order=2)
        gf = ngs.GridFunction(fes)
        gf.Set(ngs.x, definedon=mesh.Boundaries(".*"))
        gf_renderer = FacetCFRenderer3D(
            mdata, gf, order=2,
            colormap=Colormap(minval=0, maxval=1),
            clipping=clipping,
        )
        gf_renderer.active = False

        scene = wj.Draw([cf_renderer, gf_renderer], 600, 600)

        # 1) Plain CF
        webgpu_env.assert_matches_baseline(scene, "facet_cf_3d_x.png")

        # 2) FacetFESpace GridFunction
        cf_renderer.active = False
        gf_renderer.active = True
        webgpu_env.assert_matches_baseline(scene, "facet_cf_3d_facetfe.png")

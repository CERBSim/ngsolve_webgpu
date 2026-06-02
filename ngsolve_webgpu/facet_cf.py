"""Facet (element-boundary) CF rendering data extraction and rendering."""

import numpy as np
from webgpu.clipping import Clipping
from webgpu.colormap import Colormap
from webgpu.renderer import Renderer, RenderOptions
from webgpu.utils import (
    BufferBinding,
    UniformBinding,
    buffer_from_array,
    read_shader_file,
    uniform_from_array,
)
from webgpu.webgpu_api import PrimitiveTopology

from .cf import vandermonde_1d, vandermonde_trig
from .mesh import MeshData, MeshElements3d


class FacetFunctionData:
    """Extract and pack CF values on element-boundary facets for GPU rendering."""

    def __init__(self, mesh_data: MeshData, cf, order: int, deformation_cf=None):
        import ngsolve as ngs

        self.mesh_data = mesh_data
        self.order = order

        mesh = mesh_data.ngs_mesh
        ndof = order + 1

        # Integration rule on segments, evaluated on element boundaries
        seg_rule = ngs.IntegrationRule(
            [(r / order,) for r in range(ndof)], [0] * ndof
        )

        with ngs.TaskManager():
            mpts = mesh.MapToAllElements(
                {ngs.ET.SEGM: seg_rule}, ngs.VOL, element_boundary=ngs.BND
            )
            vals = cf(mpts)
            coords = ngs.CF((ngs.x, ngs.y, ngs.z))(mpts)

        n_edges = len(mpts) // ndof
        coords = np.array(coords).reshape(n_edges, ndof, 3)

        comps = cf.dim
        is_complex = cf.is_complex
        if comps > 1:
            vals = np.array(vals).reshape(n_edges, ndof, comps)
        else:
            vals = np.array(vals).reshape(n_edges, ndof, 1)

        # Convert to Bernstein coefficients
        ibmat = vandermonde_1d(order)
        geom_bernstein = np.einsum("ij,ejd->eid", ibmat, coords)
        func_bernstein = np.einsum("ij,ejc->eic", ibmat, vals).astype(np.float32)

        # Compute opposite vertices
        opposite = self._compute_opposite_vertices(mesh, n_edges)

        # Optional deformation
        deform_bernstein = None
        if deformation_cf is not None:
            with ngs.TaskManager():
                deform_coords = deformation_cf(mpts)
            deform_coords = np.array(deform_coords).reshape(n_edges, ndof, 3)
            deform_bernstein = np.einsum("ij,ejd->eid", ibmat, deform_coords).astype(np.float32)

        # Pack buffers
        self.geometry_buffer = self._pack_geometry_buffer(
            n_edges, order, geom_bernstein, opposite, deform_bernstein
        )
        self.function_buffer = self._pack_function_buffer(
            n_edges, order, comps, is_complex, func_bernstein
        )

        self.minval = float(np.min(vals))
        self.maxval = float(np.max(vals))
        self.n_edges = n_edges

    def _compute_opposite_vertices(self, mesh, n_edges):
        """Compute opposite vertex position for each element-boundary edge."""
        import ngsolve as ngs

        opposite = np.zeros((n_edges, 3), dtype=np.float32)

        # Use ngsolve element vertex ordering (matches MapToAllElements)
        edge_idx = 0
        for el in mesh.Elements(ngs.VOL):
            el_verts = el.vertices
            pts = [np.array(mesh[v].point) for v in el_verts]
            # Pad to 3D
            pts = [np.append(p, [0.0] * (3 - len(p))) for p in pts]
            n_facets = len(el_verts)

            if n_facets == 3:
                # element_boundary=BND facet ordering: opp is local vertex [1, 0, 2]
                opp_local = [1, 0, 2]
            else:
                opp_local = None

            for f in range(n_facets):
                if opp_local is not None:
                    opposite[edge_idx] = pts[opp_local[f]]
                else:
                    opposite[edge_idx] = np.mean(pts, axis=0)
                edge_idx += 1

        return opposite

    def _pack_geometry_buffer(
        self, n_edges, order, geom_bernstein, opposite, deform_bernstein
    ):
        """Pack geometry buffer: [header(5)] [geom] [opposite] [deform?]"""
        ndof = order + 1
        header_size = 5

        offset_geometry = header_size
        offset_opposite = offset_geometry + n_edges * ndof * 3
        total_size = offset_opposite + n_edges * 3

        if deform_bernstein is not None:
            offset_deformation = total_size
            total_size += n_edges * ndof * 3
        else:
            offset_deformation = 0

        buf = np.zeros(total_size, dtype=np.float32)

        # Header as float32 values (evalSegVec3 reads with u32() conversion)
        buf[0:5] = np.array(
            [n_edges, order, offset_geometry, offset_opposite, offset_deformation],
            dtype=np.float32,
        )

        buf[offset_geometry : offset_geometry + n_edges * ndof * 3] = (
            geom_bernstein.astype(np.float32).reshape(-1)
        )
        buf[offset_opposite : offset_opposite + n_edges * 3] = opposite.reshape(-1)

        if deform_bernstein is not None:
            buf[offset_deformation : offset_deformation + n_edges * ndof * 3] = (
                deform_bernstein.reshape(-1)
            )

        return buf

    def _pack_function_buffer(self, n_edges, order, comps, is_complex, func_bernstein):
        """Pack function buffer: [ncomp, order, is_complex] [values...]"""
        header = np.array(
            [comps, order, 1.0 if is_complex else 0.0], dtype=np.float32
        )
        values = func_bernstein.reshape(-1)
        return np.concatenate([header, values])


class FacetCFRenderer(Renderer):
    """Render CF values on element-boundary facets as colored thick lines."""

    topology: PrimitiveTopology = PrimitiveTopology.triangle_strip
    depthBias: int = 1
    depthBiasSlopeScale: int = 1

    def __init__(
        self,
        facet_data: FacetFunctionData,
        colormap: Colormap = None,
        clipping: Clipping = None,
        subdivision: int = 4,
        thickness: float = 0.008,
        label: str = "FacetCFRenderer",
    ):
        super().__init__(label=label)
        self.facet_data = facet_data
        self.clipping = clipping or Clipping()
        self.gpu_objects.colormap = colormap or Colormap()
        self.subdivision = subdivision
        self.thickness = thickness
        self._buffers = {}

    @property
    def colormap(self):
        return self.gpu_objects.colormap

    @colormap.setter
    def colormap(self, value: Colormap):
        self.gpu_objects.colormap = value

    def update(self, options: RenderOptions):
        self.clipping.update(options)

        data = self.facet_data
        self.n_instances = data.n_edges
        self.n_vertices = 2 * (self.subdivision + 1)

        if self.needs_update or "geometry" not in self._buffers:
            self._buffers["geometry"] = buffer_from_array(
                data.geometry_buffer, label="facet_geometry",
                reuse=self._buffers.get("geometry"),
            )
            self._buffers["values"] = buffer_from_array(
                data.function_buffer, label="facet_values",
                reuse=self._buffers.get("values"),
            )
        self._buffers["subdivision"] = uniform_from_array(
            np.array([self.subdivision], dtype=np.uint32),
            label="facet_subdivision",
            reuse=self._buffers.get("subdivision"),
        )
        self._buffers["thickness"] = uniform_from_array(
            np.array([self.thickness], dtype=np.float32),
            label="facet_thickness",
            reuse=self._buffers.get("thickness"),
        )
        deform_scale = getattr(data.mesh_data, "deformation_scale", 1.0)
        self._buffers["deformation_scale"] = uniform_from_array(
            np.array([deform_scale], dtype=np.float32),
            label="facet_deformation_scale",
            reuse=self._buffers.get("deformation_scale"),
        )

        if self.colormap.autoscale:
            self.colormap.widen_range(data.minval, data.maxval, timestamp=options.timestamp)

        self.shader_defines = {"MAX_EVAL_ORDER": data.order}

    def get_bounding_box(self):
        return self.facet_data.mesh_data.get_bounding_box()

    def get_shader_code(self):
        return read_shader_file("ngsolve/facet_edge.wgsl")

    def get_bindings(self):
        return [
            *self.clipping.get_bindings(),
            *self.gpu_objects.colormap.get_bindings(),
            BufferBinding(80, self._buffers["geometry"]),
            BufferBinding(81, self._buffers["values"]),
            UniformBinding(82, self._buffers["subdivision"]),
            UniformBinding(83, self._buffers["thickness"]),
            UniformBinding(84, self._buffers["deformation_scale"]),
        ]


class FacetCFRenderer3D(MeshElements3d):
    """Render CF on 3D element faces with slight shrink to reveal inter-element jumps."""
    fragment_entry_point: str = "fragment_facet_cf"

    def __init__(self, mesh_data, cf, order, colormap=None, clipping=None, shrink=0.999, component=None):
        from .cf import FunctionSettings, ComplexSettings
        super().__init__(data=mesh_data, clipping=clipping)
        self.cf = cf
        self._order = order
        self.gpu_objects.colormap = colormap or Colormap()
        self._shrink = shrink
        if component is None:
            component = -1 if cf.dim > 1 else 0
        self.gpu_objects.settings = FunctionSettings(component=component)
        self.gpu_objects.complex_settings = ComplexSettings()
        self._facet_gpu = None
        self._facet_data = None
        self._minval = 0.0
        self._maxval = 1.0

    def _extract_facet_data(self):
        """Extract per-face CF values on element-boundary trig faces, convert to Bernstein.

        Face reordering and barycentric permutation are handled in the shader.
        Data stays in MtAE order: face g of element e = opposite local vertex g.
        """
        import ngsolve as ngs

        mesh = self.data.ngs_mesh
        order = self._order

        # Build trig integration rule matching vandermonde_trig point ordering
        trig_pts = []
        for j in range(order + 1):
            for k in range(order + 1 - j):
                x = (order - j - k) / order if order > 0 else 0.0
                y = k / order if order > 0 else 0.0
                trig_pts.append((x, y))
        ndof_trig = len(trig_pts)

        trig_rule = ngs.IntegrationRule(trig_pts, [0] * ndof_trig)

        with ngs.TaskManager():
            mpts = mesh.MapToAllElements(
                {ngs.ET.TRIG: trig_rule}, ngs.VOL, element_boundary=ngs.BND
            )
            vals = self.cf(mpts)

        n_faces = len(mpts) // ndof_trig
        comps = self.cf.dim
        is_complex = self.cf.is_complex

        if comps > 1:
            vals = np.array(vals).reshape(n_faces, ndof_trig, comps)
        else:
            vals = np.array(vals).reshape(n_faces, ndof_trig, 1)

        # Convert to Bernstein coefficients
        ibmat = vandermonde_trig(order)
        func_bernstein = np.einsum("ij,ejc->eic", ibmat, vals).astype(np.float32)

        # Min/max
        self._minval = float(np.min(vals))
        self._maxval = float(np.max(vals))

        # Pack function buffer: [ncomp, order, is_complex] [values...]
        header = np.array(
            [comps, order, 1.0 if is_complex else 0.0], dtype=np.float32
        )
        values = func_bernstein.reshape(-1)
        self._facet_data = np.concatenate([header, values])

    def update(self, options: RenderOptions):
        super().update(options)

        if self._facet_data is None:
            try:
                self._extract_facet_data()
            except Exception:
                self.active = False
                return

        if self.needs_update or self._facet_gpu is None:
            self._facet_gpu = buffer_from_array(
                self._facet_data, label="facet_values_3d", reuse=self._facet_gpu
            )

        self.shader_defines["FACET_CF"] = "1"
        self.shader_defines["MAX_EVAL_ORDER"] = max(
            self.shader_defines.get("MAX_EVAL_ORDER", 1), self._order
        )

        if self.colormap.autoscale:
            self.colormap.widen_range(
                self._minval, self._maxval, timestamp=options.timestamp
            )

    def get_bindings(self):
        return [
            *super().get_bindings(),
            BufferBinding(81, self._facet_gpu),
            *self.gpu_objects.settings.get_bindings(),
            *self.gpu_objects.complex_settings.get_bindings(),
        ]

    def get_export_interactions(self, options, buffer_registry):
        from .cf import _complex_phase_export_interactions, _component_export_interactions
        out = list(super().get_export_interactions(options, buffer_registry))
        out += _component_export_interactions(
            self.gpu_objects.settings.uniform, self.cf.dim, buffer_registry,
        )
        if self.cf.is_complex:
            out += _complex_phase_export_interactions(
                [(self.gpu_objects.complex_settings.uniform, True)],
                buffer_registry,
            )
        return out

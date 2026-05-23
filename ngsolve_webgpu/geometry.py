import numpy as np
import webgpu
from webgpu.clipping import Clipping
from webgpu.renderer import MultipleRenderer, Renderer, RenderOptions
from webgpu.utils import (
    buffer_from_array,
    create_bind_group,
    read_buffer,
    read_shader_file,
    uniform_from_array,
)
from webgpu.webgpu_api import *


class Binding:
    VERTICES = 90
    NORMALS = 91
    INDICES = 92
    COLORS = 93


class BaseGeometryRenderer(Renderer):
    clipping: Clipping | None = None
    select_entry_point: str = "fragmentQueryIndex"
    vis_data: dict
    _select_active: bool = True

    def __init__(self, clipping, *args, **kwargs):
        self.clipping = clipping
        self._selection = None
        from .pick import HighlightUniforms
        self._highlight_uniforms = HighlightUniforms()
        super().__init__(*args, **kwargs)

    def select(self, options, x, y):
        if not self._select_active:
            return
        super().select(options, x, y)

    def set_selection(self, indices):
        """Set which region indices are selected (list/set of ints)."""
        if self._selection is None:
            return
        self._selection[:] = 0
        for idx in indices:
            if 0 <= idx < len(self._selection):
                self._selection[idx] = 1
        self.device.queue.writeBuffer(
            self._buffers["selection"], 0, self._selection.tobytes()
        )


class GeometryFaceRenderer(BaseGeometryRenderer):
    n_vertices: int = 3

    def __init__(self, geo, clipping, symmetry=None):
        super().__init__(clipping, label="GeometryFaces")
        self.symmetry = symmetry
        self.geo = geo
        self.colors = None
        self.active = True
        self._buffers = {}

    def get_bounding_box(self):
        bbox = self.bounding_box
        if self.symmetry:
            bbox = self.symmetry.expand_bbox(bbox)
        return bbox

    def set_colors(self, colors):
        """colors is numpy float32 array with 4 times number indices entries"""
        self.colors = colors
        if "colors" in self._buffers:
            self.device.queue.writeBuffer(self._buffers["colors"], 0, self.colors.tobytes())

    def update(self, options):
        if self._buffers:
            return
        vis_data = self.vis_data
        self.bounding_box = (vis_data["min"], vis_data["max"])
        verts = vis_data["vertices"]
        self.n_instances = len(verts) // 9
        self._original_n_instances = self.n_instances
        if self.symmetry:
            self.n_instances *= self.symmetry.n_copies
            self.shader_defines["SYMMETRY"] = "1"
        normals = vis_data["normals"]
        indices = vis_data["indices"]
        if self.colors is None:
            self.colors = vis_data["face_colors"]
        alphas = self.colors[3::4]
        self.transparent = bool(np.any((alphas > 0) & (alphas < 1.0)))
        self._buffers = {}
        for key, data in zip(
            ("vertices", "normals", "indices", "colors"),
            (verts, normals, indices, self.colors),
        ):
            self._buffers[key] = buffer_from_array(data)
        self._solid_ids = self._build_solid_ids()
        self._buffers["solid_ids"] = buffer_from_array(self._solid_ids)
        n_regions = max(len(self.colors) // 4, 1)
        self._selection = np.zeros(n_regions, dtype=np.uint32)
        self._buffers["selection"] = buffer_from_array(self._selection)
        self.shader_defines["HAS_SELECTION"] = "1"

    def _build_solid_ids(self):
        n_faces = len(self.geo.faces)
        solid_ids = np.full(max(n_faces, 1), 0xFFFFFFFF, dtype=np.uint32)
        try:
            solids = list(self.geo.shape.solids)
            if not solids:
                return solid_ids
            for solid_idx, solid in enumerate(solids):
                centers = {tuple(round(c, 8) for c in f.center) for f in solid.faces}
                for fi in range(n_faces):
                    gc = tuple(round(c, 8) for c in self.geo.faces[fi].center)
                    if gc in centers and solid_ids[fi] == 0xFFFFFFFF:
                        solid_ids[fi] = solid_idx
        except Exception:
            pass
        return solid_ids

    def get_bindings(self):
        bindings = [
            *self.clipping.get_bindings(),
            webgpu.BufferBinding(Binding.VERTICES, self._buffers["vertices"]),
            webgpu.BufferBinding(Binding.NORMALS, self._buffers["normals"]),
            webgpu.BufferBinding(Binding.INDICES, self._buffers["indices"]),
            webgpu.BufferBinding(Binding.COLORS, self._buffers["colors"]),
            webgpu.BufferBinding(94, self._buffers["solid_ids"]),
            webgpu.BufferBinding(58, self._buffers["selection"]),
            *self._highlight_uniforms.get_bindings(),
        ]
        if self.symmetry:
            bindings += self.symmetry.get_bindings(self._original_n_instances)
        return bindings

    def get_shader_code(self):
        return read_shader_file("ngsolve/geo_face.wgsl")


class GeometryEdgeRenderer(BaseGeometryRenderer):
    n_vertices: int = 4
    topology: PrimitiveTopology = PrimitiveTopology.triangle_strip

    # make sure that edges are rendered on top of faces
    depthBias: int = -5
    depthBiasSlopeScale: int = -5

    def __init__(self, geo, clipping, symmetry=None):
        self.geo = geo
        super().__init__(clipping, label="GeometryEdges")
        self.symmetry = symmetry
        self.active = True
        self.thickness = 0.005
        self._buffers = {}

    def set_colors(self, colors):
        """colors is numpy float32 array with 4 times number indices entries"""
        self.colors = colors
        if "colors" in self._buffers:
            self.device.queue.writeBuffer(self._buffers["colors"], 0, self.colors.tobytes())

    def update(self, options):
        if self._buffers:
            return
        vis_data = self.vis_data
        verts = vis_data["edges"]
        self.colors = vis_data["edge_colors"]
        self.n_instances = len(verts) // 6
        self._original_n_instances = self.n_instances
        if self.symmetry:
            self.n_instances *= self.symmetry.n_copies
            self.shader_defines["SYMMETRY"] = "1"
        self.thickness_uniform = uniform_from_array(np.array([self.thickness], dtype=np.float32))
        self._buffers = {}
        self._buffers["vertices"] = buffer_from_array(verts)
        self._buffers["colors"] = buffer_from_array(self.colors)
        self._buffers["index"] = buffer_from_array(vis_data["edge_indices"])
        n_regions = max(len(self.colors) // 4, 1)
        self._selection = np.zeros(n_regions, dtype=np.uint32)
        self._buffers["selection"] = buffer_from_array(self._selection)
        self.shader_defines["HAS_SELECTION"] = "1"

    def get_shader_code(self):
        return read_shader_file("ngsolve/geo_edge.wgsl")

    def get_bindings(self):
        bindings = [
            *self.clipping.get_bindings(),
            webgpu.BufferBinding(90, self._buffers["vertices"]),
            webgpu.BufferBinding(91, self._buffers["colors"]),
            webgpu.UniformBinding(92, self.thickness_uniform),
            webgpu.BufferBinding(93, self._buffers["index"]),
            webgpu.BufferBinding(58, self._buffers["selection"]),
            *self._highlight_uniforms.get_bindings(),
        ]
        if self.symmetry:
            bindings += self.symmetry.get_bindings(self._original_n_instances)
        return bindings


class GeometryVertexRenderer(BaseGeometryRenderer):
    n_vertices: int = 4
    topology: PrimitiveTopology = PrimitiveTopology.triangle_strip
    depthBias: int = -10
    depthBiasSlopeScale: int = -10

    def __init__(self, geo, clipping):
        self.geo = geo
        super().__init__(clipping, label="GeometryVertices")
        self.active = True
        self.thickness = 0.05
        self._buffers = {}

    def set_colors(self, colors):
        """colors is numpy float32 array with 4 times number indices entries"""
        self.colors = colors
        if "colors" in self._buffers:
            self.device.queue.writeBuffer(self._buffers["colors"], 0, self.colors.tobytes())

    def get_shader_code(self):
        return read_shader_file("ngsolve/geo_vertex.wgsl")

    def update(self, options):
        if self._buffers:
            return
        verts = set(self.geo.shape.vertices)
        self.colors = np.array(
            [v.col if v.col is not None else [0.3, 0.3, 0.3, 1.0] for v in verts],
            dtype=np.float32,
        ).flatten()
        self.n_instances = len(verts)
        vert_values = np.array([[pi for pi in v.p] for v in verts], dtype=np.float32).flatten()
        self._buffers = {}
        self._buffers["vertices"] = buffer_from_array(vert_values)
        self._buffers["colors"] = buffer_from_array(self.colors)
        self.thickness_uniform = uniform_from_array(np.array([self.thickness], dtype=np.float32))
        n_regions = max(self.n_instances, 1)
        self._selection = np.zeros(n_regions, dtype=np.uint32)
        self._buffers["selection"] = buffer_from_array(self._selection)
        self.shader_defines["HAS_SELECTION"] = "1"

    def get_bindings(self):
        return [
            *self.clipping.get_bindings(),
            webgpu.BufferBinding(90, self._buffers["vertices"]),
            webgpu.BufferBinding(91, self._buffers["colors"]),
            webgpu.UniformBinding(92, self.thickness_uniform),
            webgpu.BufferBinding(58, self._buffers["selection"]),
            *self._highlight_uniforms.get_bindings(),
        ]


class GeometryRenderer(MultipleRenderer):
    def __init__(self, geo, label="Geometry", clipping=None, symmetry=None):
        self.geo = geo
        self.clipping = clipping or Clipping()
        self.faces = GeometryFaceRenderer(geo, self.clipping, symmetry=symmetry)
        self.edges = GeometryEdgeRenderer(geo, self.clipping, symmetry=symmetry)
        self.vertices = GeometryVertexRenderer(geo, self.clipping)
        self.faces.clipping = self.clipping
        self.edges.clipping = self.clipping
        self.vertices.clipping = self.clipping
        self.vertices.active = False
        self._vis_data = None
        super().__init__([self.vertices, self.edges, self.faces])

    def update(self, options: RenderOptions):
        if self._vis_data is None:
            self._vis_data = self.geo._visualizationData()
        self.clipping.update(options)
        for ro in self.render_objects:
            ro.vis_data = self._vis_data
            ro.update(options)

        self.canvas = options.canvas

    def get_bounding_box(self):
        pmin, pmax = self.geo.shape.bounding_box
        pmin = [pi + 1e-7 for pi in pmin] # correct for added eps in occ
        pmax = [pi - 1e-7 for pi in pmax]
        return ([pmin[0], pmin[1], pmin[2]], [pmax[0], pmax[1], pmax[2]])

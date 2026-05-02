import numpy as np

from webgpu.clipping import Clipping
from webgpu.font import Font
from webgpu.renderer import Renderer, RenderOptions
from webgpu.utils import BufferBinding, UniformBinding, buffer_from_array, uniform_from_array, read_shader_file

from .mesh import MeshData, Binding, ElType


class EntityNumbers(Renderer):
    """Render numbers for mesh entities (vertices, edges, facets, surface/volume elements).

    Positions are derived on the GPU from the mesh buffer. For edges and facets,
    a small connectivity buffer is uploaded on demand.
    """

    fragment_entry_point: str = "fragmentFont"
    select_entry_point: str = ""
    depthBias: int = -1
    n_digits: int = 7

    _ENTITY_CONFIG = {
        "vertices": "vertexVertexNumber",
        "edges": "vertexEdgeNumber",
        "facets": "vertexFacetNumber",
        "surface_elements": "vertexSurfaceElementNumber",
        "volume_elements": "vertexVolumeElementNumber",
    }

    def __init__(self, data: MeshData, entity: str = "vertices", font_size=20, clipping=None, zero_based=True):
        if entity not in self._ENTITY_CONFIG:
            raise ValueError(f"Unknown entity type: {entity!r}. Must be one of {list(self._ENTITY_CONFIG)}")
        super().__init__(label=f"{entity} numbers")
        self.data = data
        self.entity = entity
        self.zero_based = zero_based
        self.vertex_entry_point = self._ENTITY_CONFIG[entity]
        self.font_size = font_size
        self.clipping = clipping or Clipping()
        self.n_vertices = self.n_digits * 6
        self._edge_buffer = None
        self._facet_buffer = None
        self._dummy_uniform = None
        self._dummy_buffer = None
        self._offset_buffer = None
        if entity == "volume_elements":
            data.need_3d = True

    def update(self, options: RenderOptions):
        self.clipping.update(options)
        self.data.update(options)
        self.font = Font(options.canvas, self.font_size)
        self._mesh_buffers = self.data.get_buffers()

        offset = 0 if self.zero_based else 1
        self._offset_buffer = uniform_from_array(
            np.array([offset], dtype=np.uint32), label="number_offset", reuse=self._offset_buffer
        )

        if self.entity == "vertices":
            self.n_instances = self.data.num_elements["vertices"]
        elif self.entity == "edges":
            self._update_edges()
        elif self.entity == "facets":
            self._update_facets()
        elif self.entity == "surface_elements":
            self.n_instances = self.data.num_elements[ElType.TRIG]
        elif self.entity == "volume_elements":
            n = 0
            for et in (ElType.TET, ElType.HEX, ElType.PRISM, ElType.PYRAMID):
                n += self.data.num_elements.get(et, 0)
            self.n_instances = n

    def _update_edges(self):
        """Build edge connectivity buffer from NGSolve mesh."""
        import ngsolve
        ngs_mesh = self.data.ngs_mesh
        if not isinstance(ngs_mesh, ngsolve.Mesh):
            ngs_mesh = ngsolve.Mesh(self.data.mesh)
        edges = np.array(
            [(e.vertices[0].nr, e.vertices[1].nr) for e in ngs_mesh.edges],
            dtype=np.uint32,
        )
        self.n_instances = len(edges)
        self._edge_buffer = buffer_from_array(edges, label="edge_connectivity", reuse=self._edge_buffer)

    def _update_facets(self):
        """Build facet connectivity buffer from NGSolve mesh."""
        import ngsolve
        ngs_mesh = self.data.ngs_mesh
        if not isinstance(ngs_mesh, ngsolve.Mesh):
            ngs_mesh = ngsolve.Mesh(self.data.mesh)
        facets = []
        for f in ngs_mesh.faces:
            verts = [v.nr for v in f.vertices]
            nv = len(verts)
            if nv == 3:
                facets.append((verts[0], verts[1], verts[2], 3))
            else:
                facets.append((verts[0], verts[1], verts[2], 4))
        facet_data = np.array(facets, dtype=np.uint32)
        self.n_instances = len(facet_data)
        self._facet_buffer = buffer_from_array(facet_data, label="facet_connectivity", reuse=self._facet_buffer)

    def get_shader_code(self):
        return read_shader_file("ngsolve/entity_numbers.wgsl")

    def get_bounding_box(self):
        return self.data.get_bounding_box()

    def get_bindings(self):
        bindings = [
            *self.clipping.get_bindings(),
            *self.font.get_bindings(),
            *self.data.get_bindings(),
        ]
        # mesh/utils.wgsl declares u_mesh at binding 20 — bind a dummy uniform
        if self._dummy_uniform is None:
            self._dummy_uniform = uniform_from_array(
                np.zeros(4, dtype=np.float32), label="dummy_mesh_uniform"
            )
        bindings.append(UniformBinding(Binding.MESH, self._dummy_uniform))
        # Edge/facet buffers — bind a dummy if not needed (shader still declares them)
        if self._dummy_buffer is None:
            self._dummy_buffer = buffer_from_array(np.array([0], dtype=np.uint32), label="dummy_entity")
        edge_buf = self._edge_buffer if self._edge_buffer is not None else self._dummy_buffer
        facet_buf = self._facet_buffer if self._facet_buffer is not None else self._dummy_buffer
        bindings.append(BufferBinding(12, edge_buf))
        bindings.append(BufferBinding(13, facet_buf))
        bindings.append(UniformBinding(14, self._offset_buffer))
        return bindings


# Backward compatibility
class PointNumbers(EntityNumbers):
    def __init__(self, data, font_size=20, label=None, clipping=None):
        super().__init__(data, entity="vertices", font_size=font_size, clipping=clipping, zero_based=False)

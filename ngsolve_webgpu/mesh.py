from enum import Enum

import netgen.meshing
import numpy as np
from webgpu.clipping import Clipping
from webgpu.font import Font
from webgpu.renderer import Renderer, RenderOptions, check_timestamp

# from webgpu.uniforms import Binding
from webgpu.uniforms import UniformBase, ct
from webgpu.utils import (
    BufferBinding,
    UniformBinding,
    buffer_from_array,
    get_device,
    read_shader_file,
    uniform_from_array,
)
from webgpu.webgpu_api import *


class Binding:
    """Binding numbers for uniforms in shader code in uniforms.wgsl"""

    EDGES = 8
    TRIGS = 9
    SEG_FUNCTION_VALUES = 11
    VERTICES = 12
    TRIGS_INDEX = 13
    CURVATURE_VALUES_2D = 14
    SUBDIVISION = 15
    DEFORMATION_VALUES = 16
    DEFORMATION_SCALE = 17

    MESH = 20
    EDGE = 21
    SEG = 22
    TRIG = 23
    QUAD = 24
    TET = 25
    PYRAMID = 26
    PRISM = 27
    HEX = 28

    LINE_INTEGRAL_CONVOLUTION = 40
    LINE_INTEGRAL_CONVOLUTION_INPUT_TEXTURE = 41
    LINE_INTEGRAL_CONVOLUTION_OUTPUT_TEXTURE = 42


class _eltype:
    dim: int
    primitive_topology: PrimitiveTopology
    num_vertices_per_primitive: int

    def __init__(self, dim, primitive_topology, num_vertices_per_primitive):
        self.dim = dim
        self.primitive_topology = primitive_topology
        self.num_vertices_per_primitive = num_vertices_per_primitive


class ElType(Enum):
    POINT = _eltype(0, PrimitiveTopology.point_list, 1)
    SEG = _eltype(1, PrimitiveTopology.line_list, 2)
    TRIG = _eltype(2, PrimitiveTopology.triangle_list, 3)
    QUAD = _eltype(2, PrimitiveTopology.triangle_list, 2 * 3)
    TET = _eltype(3, PrimitiveTopology.triangle_list, 4 * 3)
    HEX = _eltype(3, PrimitiveTopology.triangle_list, 6 * 2 * 3)
    PRISM = _eltype(3, PrimitiveTopology.triangle_list, 2 * 3 + 3 * 2 * 3)
    PYRAMID = _eltype(3, PrimitiveTopology.triangle_list, 4 + 2 * 3)

    @staticmethod
    def from_dim_np(dim: int, np: int):
        if dim == 2:
            if np == 3:
                return ElType.TRIG
            if np == 4:
                return ElType.QUAD
        if dim == 3:
            if np == 4:
                return ElType.TET
            if np == 8:
                return ElType.HEX
            if np == 6:
                return ElType.PRISM
            if np == 5:
                return ElType.PYRAMID
        raise ValueError(f"Unsupported element type dim={dim} np={np}")


ElTypes2D = [ElType.TRIG, ElType.QUAD]
ElTypes3D = [ElType.TET, ElType.HEX, ElType.PRISM, ElType.PYRAMID]


class MeshData:
    # only for drawing the mesh, not needed for function values
    num_elements: dict[str | ElType, int]
    elements: dict[str | ElType, np.ndarray]
    gpu_elements: dict[str | ElType, Buffer]
    subdivision: int

    mesh: netgen.meshing.Mesh
    curvature_data = None
    deformation_data = None
    _ngs_mesh = None
    _last_mesh_timestamp: int = -1
    _timestamp: float = -1
    _needs_update: bool = True

    def __init__(self, mesh):
        self.on_region = False
        self.need_3d = False
        if isinstance(mesh, netgen.meshing.Mesh):
            self.mesh = mesh
        else:
            self._ngs_mesh = mesh
            import ngsolve as ngs

            if isinstance(mesh, ngs.Region):
                self.on_region = True
                mesh = mesh.mesh
            self.mesh = mesh.ngmesh
        self.num_elements = {}
        self.elements = {}
        self.gpu_elements = {}
        self.subdivision = None
        self._deformation_scale = 1

    @property
    def deformation_scale(self):
        return self._deformation_scale

    @deformation_scale.setter
    def deformation_scale(self, value):
        self._deformation_scale = value
        if self.gpu_elements and "deformation_scale" in self.gpu_elements:
            get_device().queue.writeBuffer(
                self.gpu_elements["deformation_scale"],
                0,
                np.array([self._deformation_scale], dtype=np.float32).tobytes(),
            )

    @property
    def ngs_mesh(self):
        import ngsolve

        if self._ngs_mesh is None:
            self._ngs_mesh = ngsolve.Mesh(self.mesh)
        return self._ngs_mesh

    def set_needs_update(self):
        """Update GPU data on next render call"""
        self._last_mesh_timestamp = -1
        self._timestamp = -1

    @check_timestamp
    def update(self, options: RenderOptions):
        if self._last_mesh_timestamp != self.mesh._timestamp:
            self._create_data()
        if "curvature_2d" in self.gpu_elements:
            self.gpu_elements.pop("curvature_2d")
        if "deformation_2d" in self.gpu_elements:
            self.gpu_elements.pop("deformation_2d")
        # prevent recursion
        self._timestamp = options.timestamp
        if self.curvature_data:
            self.curvature_data.update(options)
            self.elements["curvature_2d"] = self.curvature_data.data_2d
        else:
            self.elements["curvature_2d"] = np.array([0], dtype=np.float32)

        if self.deformation_data:
            self.deformation_data.update(options)
            self.elements["deformation_2d"] = self.deformation_data.data_2d
        else:
            self.elements["deformation_2d"] = np.array([-1], dtype=np.float32)

    def _create_data(self):
        # TODO: implement other element types than triangles
        # TODO: handle region correctly to draw only part of the mesh
        mesh = self.mesh
        self.num_elements = {eltype: 0 for eltype in ElType}
        self.elements = {}
        self.gpu_elements = {}

        # Vertices
        nv = len(mesh.Points())
        self.num_elements["vertices"] = nv
        vertices = np.array(mesh.Coordinates(), dtype=np.float32)
        if vertices.shape[1] == 2:
            vertices = np.hstack((vertices, np.zeros((nv, 1), dtype=np.float32)))

        self.pmin = np.min(vertices, axis=0)
        self.pmax = np.max(vertices, axis=0)
        self.elements["vertices"] = vertices

        # Trigs TODO: Quads
        trigs = mesh.Elements2D().NumPy()
        if self.on_region:
            region = self.ngs_mesh
            import ngsolve as ngs

            if region.VB() == ngs.VOL and region.mesh.dim == 3:
                region = region.Boundaries()
            indices = np.flatnonzero(region.Mask()) + 1
            trigs = trigs[np.isin(trigs["index"], indices)]
        self.num_elements[ElType.TRIG] = len(trigs)
        trigs_data = np.zeros((len(trigs), 4), dtype=np.uint32)
        trigs_data[:, :3] = trigs["nodes"][:, :3] - 1
        trigs_data[:, 3] = trigs["index"]
        self.elements[ElType.TRIG] = trigs_data

        # 3d Elements
        if self.need_3d:
            els = mesh.Elements3D().NumPy()
            for num_pts in (4, 5, 6, 8):
                eltype = ElType.from_dim_np(3, num_pts)
                filtered = els[els["np"] == num_pts]
                nels = len(filtered)
                if nels == 0:
                    continue
                u32array = np.empty((nels, num_pts + 2), dtype=np.uint32)
                u32array[:, :num_pts] = filtered["nodes"][:, :num_pts] - 1
                u32array[:, num_pts] = filtered["index"]
                self.elements[eltype] = u32array
                self.num_elements[eltype] = len(filtered)

        curve_order = mesh.GetCurveOrder()
        if self.deformation_data is not None:
            curve_order = max(curve_order, self.deformation_data.order)
        if curve_order > 1:
            import ngsolve as ngs

            from .cf import FunctionData

            cf = ngs.CF((ngs.x, ngs.y, ngs.z))
            self.curvature_data = FunctionData(self, cf, curve_order)
        else:
            if self.subdivision is None:
                self.subdivision = 1
        if self.subdivision is None:
            deformation_order = 1
            if self.deformation_data:
                deformation_order = self.deformation_data.order
            order = max(curve_order, deformation_order)
            if order > 3:
                subdiv = (order + 2) // 3 + 1
            elif order > 1:
                subdiv = 3
            else:
                subdiv = 1
            self.subdivision = subdiv

        self._last_mesh_timestamp = mesh._timestamp

    def get_bounding_box(self):
        pmin, pmax = self.mesh.bounding_box
        return ([pmin[0], pmin[1], pmin[2]],
                [pmax[0], pmax[1], pmax[2]])

    def get_buffers(self):
        for eltype in self.elements:
            if eltype not in self.gpu_elements:
                self.gpu_elements[eltype] = buffer_from_array(
                    self.elements[eltype], label="mesh_" + str(eltype)
                )
        if "subdivision" not in self.gpu_elements:
            self.gpu_elements["subdivision"] = uniform_from_array(
                np.array([self.subdivision], dtype=np.uint32),
                label="subdivision",
            )
        if "deformation_scale" not in self.gpu_elements:
            self.gpu_elements["deformation_scale"] = uniform_from_array(
                np.array([self.deformation_scale], dtype=np.float32), label="deformation_scale"
            )

        return self.gpu_elements


class MeshElements2d(Renderer):
    depthBias: int = 1
    depthBiasSlopeScale: float = 1.0
    vertex_entry_point: str = "vertexTrigP1Indexed"
    fragment_entry_point: str = "fragment2dElement"
    color = (0, 1, 0, 1)

    def __init__(self, data: MeshData, label="MeshElements2d", clipping=None):
        super().__init__(label=label)
        self.data = data
        self.clipping = clipping or Clipping()

    def update(self, options: RenderOptions):
        self.clipping.update(options)
        self.data.update(options)
        self.subdivision = self.data.subdivision
        self.n_vertices = 3 * self.subdivision**2

        self._buffers = self.data.get_buffers()
        self.n_instances = self.data.num_elements[ElType.TRIG]
        self.color_uniform = buffer_from_array(
            np.array(self.color, dtype=np.float32), label="color_uniform"
        )

        order = 1
        if self.data.curvature_data:
            order = max(order, self.data.curvature_data.order)
        if self.data.deformation_data:
            order = max(order, self.data.deformation_data.order)

        self.shader_defines = {"MAX_EVAL_ORDER": 1, "MAX_EVAL_ORDER_VEC3": order + 1}

    def get_bounding_box(self):
        return self.data.get_bounding_box()

    def get_bindings(self):
        bindings = [
            *self.clipping.get_bindings(),
            BufferBinding(Binding.VERTICES, self._buffers["vertices"]),
            BufferBinding(Binding.TRIGS_INDEX, self._buffers[ElType.TRIG]),
            BufferBinding(Binding.CURVATURE_VALUES_2D, self._buffers["curvature_2d"]),
            BufferBinding(Binding.DEFORMATION_VALUES, self._buffers["deformation_2d"]),
            UniformBinding(Binding.DEFORMATION_SCALE, self._buffers["deformation_scale"]),
            UniformBinding(Binding.SUBDIVISION, self._buffers["subdivision"]),
        ]
        if hasattr(self, "color_uniform"):
            bindings.append(BufferBinding(54, self.color_uniform))
        return bindings

    def get_shader_code(self):
        return read_shader_file("ngsolve/mesh.wgsl")


class MeshWireframe2d(MeshElements2d):
    depthBias: int = 0
    depthBiasSlopeScale: float = 0.
    topology: PrimitiveTopology = PrimitiveTopology.line_strip
    color = (0, 0, 0, 1)
    fragment_entry_point: str = "fragmentWireframe2d"
    vertex_entry_point: str = "vertexWireframe2d"

    def update(self, options: RenderOptions):
        super().update(options)
        self.n_vertices = 3 * self.subdivision + 1


class El3dUniform(UniformBase):
    _binding = Binding.MESH
    _fields_ = [
        ("subdivision", ct.c_uint32),
        ("shrink", ct.c_float),
        ("padding", ct.c_float * 2),
    ]

    def __init__(self, subdivision=0, shrink=1.0):
        super().__init__(subdivision=subdivision, shrink=shrink)


class MeshElements3d(Renderer):
    n_vertices: int = 3 * 4

    def __init__(self, data: MeshData, clipping=None):
        super().__init__(label="MeshElements3d")
        data.need_3d = True
        self.data = data
        self.clipping = clipping or Clipping()
        self._shrink = 1.0
        self.uniforms = None

    @property
    def shrink(self):
        return self._shrink

    @shrink.setter
    def shrink(self, value):
        self._shrink = value
        if self.uniforms is not None:
            self.uniforms.shrink = value

    def get_bounding_box(self) -> tuple[list[float], list[float]] | None:
        return self.data.get_bounding_box()

    def update(self, options: RenderOptions):
        if self.uniforms is None:
            self.uniforms = El3dUniform()
            self.uniforms.shrink = self._shrink
        self.data.update(options)
        self.clipping.update(options)
        self._buffers = self.data.get_buffers()
        self.uniforms.update_buffer()
        self.n_instances = self.data.num_elements[ElType.TET]

    def add_options_to_gui(self, gui):
        if gui is None:
            return

        def set_shrink(value):
            self.uniforms.shrink = value
            self.uniforms.update_buffer()

        gui.slider(
            label="Shrink", value=self.uniforms.shrink, min=0.0, max=1.0, step=0.01, func=set_shrink
        )

    def get_bindings(self):
        return [
            *self.clipping.get_bindings(),
            BufferBinding(Binding.VERTICES, self._buffers["vertices"]),
            BufferBinding(Binding.TET, self._buffers[ElType.TET]),
            *self.uniforms.get_bindings(),
        ]

    def get_shader_code(self):
        return read_shader_file("ngsolve/elements3d.wgsl")


class PointNumbers(Renderer):
    """Render a point numbers of a mesh"""

    _buffers: dict

    def __init__(self, data, font_size=20, label=None, clipping=None):
        super().__init__(label=label)
        self.n_digits = 6
        self.data = data
        self.depthBias = -1
        self.vertex_entry_point = "vertexPointNumber"
        self.fragment_entry_point = "fragmentFont"
        self.n_vertices = self.n_digits * 6
        self.font_size = font_size
        self.clipping = clipping or Clipping()

    def update(self, options: RenderOptions):
        self.clipping.update(options)
        self.font = Font(options.canvas, self.font_size)
        self._buffers = self.data.get_buffers()
        self.n_instances = self.data.num_elements["vertices"]

    def get_shader_code(self):
        return read_shader_file("ngsolve/numbers.wgsl")

    def get_bounding_box(self):
        return self.data.get_bounding_box()

    def get_bindings(self):
        return [
            *self.clipping.get_bindings(),
            *self.font.get_bindings(),
            BufferBinding(Binding.VERTICES, self._buffers["vertices"]),
        ]

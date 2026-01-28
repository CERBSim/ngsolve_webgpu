from enum import Enum

import netgen.meshing
import typing
import numpy as np
import threading

from webgpu.clipping import Clipping
from webgpu.font import Font
from webgpu.renderer import Renderer, RenderOptions, check_timestamp
from webgpu.colormap import Colormap

if typing.TYPE_CHECKING:
    import netgen.meshing
    import ngsolve


# from webgpu.uniforms import Binding
from webgpu.uniforms import UniformBase, ct
from webgpu.utils import (
    BufferBinding,
    UniformBinding,
    buffer_from_array,
    get_device,
    read_shader_file,
    uniform_from_array,
    Lock,
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
    DEFORMATION_3D_VALUES = 18

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
            if np == 10:
                return ElType.TET
        raise ValueError(f"Unsupported element type dim={dim} np={np}")


ElTypes2D = [ElType.TRIG, ElType.QUAD]
ElTypes3D = [ElType.TET, ElType.HEX, ElType.PRISM, ElType.PYRAMID]


class MeshData:
    # only for drawing the mesh, not needed for function values
    num_elements: dict[str | ElType, int]
    elements: dict[str | ElType, np.ndarray]
    gpu_elements: dict[str | ElType, Buffer]
    subdivision: int

    el2d_bitarray = None
    el3d_bitarray = None
    mesh: "netgen.meshing.Mesh | ngsolve.Mesh"
    curvature_data = None
    deformation_data = None
    _ngs_mesh = None
    _last_mesh_timestamp: int = -1
    _timestamp: float = -1
    _need_3d: bool = False
    _update_lock: Lock

    def __init__(self, mesh, el2d_bitarray=None, el3d_bitarray=None):
        import netgen.meshing

        self.on_region = False
        self.el2d_bitarray = el2d_bitarray
        self.el3d_bitarray = el3d_bitarray
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
        self._update_lock = Lock()

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

    @property
    def need_3d(self):
        return self._need_3d

    @need_3d.setter
    def need_3d(self, value: bool):
        if value == self._need_3d:
            return
        self.set_needs_update()
        self._need_3d = value

    def set_needs_update(self):
        """Update GPU data on next render call"""
        self._last_mesh_timestamp = -1
        self._timestamp = -1

    @property
    def needs_update(self):
        return self._timestamp < 0 or self._last_mesh_timestamp != self.mesh._timestamp

    @check_timestamp
    def update(self, options: RenderOptions):
        # prevent recursion
        self._timestamp = options.timestamp
        with self._update_lock:
            if self._last_mesh_timestamp != self.mesh._timestamp:
                self._create_data()

                if self.curvature_data:
                    self.curvature_data.update(options)
                    self.elements["curvature_2d"] = self.curvature_data.data_2d
                else:
                    self.elements["curvature_2d"] = np.array([0], dtype=np.float32)
                self.gpu_elements.pop("curvature_2d", None)

            if self.deformation_data:
                self.deformation_data.update(options)

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
        if self.el2d_bitarray is not None:
            trigs = trigs[np.array(self.el2d_bitarray, dtype=bool)]
        if self.on_region:
            region = self.ngs_mesh
            import ngsolve as ngs

            if region.VB() == ngs.VOL and region.mesh.dim == 3:
                region = region.Boundaries()
            indices = np.flatnonzero(region.Mask()) + 1
            trigs = trigs[np.isin(trigs["index"], indices)]
        self.num_elements[ElType.TRIG] = len(trigs)

        trigs_data = np.zeros((len(trigs), 4), dtype=np.int32)

        trigs_data[:, :3] = trigs["nodes"][:, :3] - 1
        trigs_data[:, 3] = trigs["index"] - 1
        # trigs_data = trigs_data.flatten()

        # append fourth point for quads

        # triangle data
        # pi0, pi1, pi2, index, pi0, pi1, pi2, pi3, index, ....
        # quad in "triangle" data
        # pi0, pi1, pi2, -offset
        # at offset:
        # pi3, index

        quads_data = []

        el_numers = np.array(range(len(trigs)), dtype=np.int32)
        quads_index = trigs["np"] == 4

        quad_numbers = el_numers[quads_index]
        print("quad numbers", quad_numbers)

        num_quads = np.sum(quads_index)

        for i in range(len(trigs)):
            if trigs["np"][i] == 4:
                pi3 = trigs["nodes"][i][3] - 1
                idx = trigs["index"][i] - 1
                offset = 2 + num_quads + len(trigs)*4 + len(quads_data)
                quads_data.append(pi3)
                quads_data.append(idx)
                trigs_data[i][3] = -offset

        print("trigs data", trigs_data)
        print("quads data", quads_data)

        metadata = np.array([len(trigs), num_quads], dtype=np.int32)
        all_data = np.concatenate( (metadata, trigs_data.flatten(), quad_numbers, np.array(quads_data, dtype=np.int32)))
        self.elements[ElType.TRIG] = all_data
        print("num_quads", num_quads, type(num_quads))
        self.num_elements[ElType.TRIG] += num_quads
        print("num_elements trig", self.num_elements[ElType.TRIG])


        # todo:
        # build a 3d data array similar to the 2d one above, and store it in self.elements["el3d"]
        self.elements["el3d"] = np.array([0], dtype=np.uint32)

        # 3d Elements
        if self.need_3d:
            els = mesh.Elements3D().NumPy()
            if self.el3d_bitarray is not None:
                els = els[np.array(self.el3d_bitarray, dtype=bool)]
            for num_pts in (4, 5, 6, 8, 10):
                eltype = ElType.from_dim_np(3, num_pts)
                filtered = els[els["np"] == num_pts]
                nels = len(filtered)
                if nels == 0:
                    continue
                lo_num_pts = 4 if num_pts == 10 else num_pts
                u32array = np.empty((nels, lo_num_pts + 2), dtype=np.uint32)
                u32array[:, :lo_num_pts] = filtered["nodes"][:, :lo_num_pts] - 1
                u32array[:, lo_num_pts] = filtered["index"] - 1
                self.elements[eltype] = u32array
                self.num_elements[eltype] = len(filtered)

            els_data = np.zeros((len(els), 4), dtype=np.int32)

            els_data[:, :3] = els["nodes"][:, :3] - 1
            els_data[:, 3] = els["index"] - 1

            rest_data = []

            els_numbers = np.array(range(len(els)), dtype=np.int32)
            rest_index = (els["np"] == 5) or (els["np"] == 6) or (els["np"] == 8) or (els["np"] == 10) 

            rest_numbers = els_numbers[rest_index]
            print("rest numbers", rest_numbers)

            num_rests = np.sum(rest_index)

            for i in range(len(els)):
                if els["np"][i] == 5: #pyramid
                    pi5 = els["nodes"][i][5] - 1
                    idx = els["index"][i] - 1
                    offset = 3 + num_rests + len(els)*5 + len(rest_data)
                    rest_data.append(5)                            
                    rest_data.append(pi5)
                    rest_data.append(idx)
                    els_data[i][5] = -offset
                elif els["np"][i] == 6: #prism
                    pi5, pi6 = [els["nodes"][i][j] - 1 for j in range(5, 7)]
                    idx = els["index"][i] - 1
                    offset = 3 + num_rests + len(els)*5 + len(rest_data)
                    rest_data.extend([6, pi5, pi6, idx])
                    els_data[i][5] = -offset
                elif els["np"][i] == 8: #HEX
                    pi5, pi6, pi7, pi8 = [els["nodes"][i][j] - 1 for j in range(5, 9)]
                    idx = els["index"][i] - 1
                    offset = 3 + num_rests + len(els)*5 + len(rest_data)
                    rest_data.extend([8, pi5, pi6, pi7, pi8, idx])
                    els_data[i][5] = -offset

            print("els data", els_data)
            print("els data", rest_data)

            metadata = np.array([len(els), num_rests], dtype=np.int32)
            all_data = np.concatenate( (metadata, els_data.flatten(), rest_numbers, np.array(rest_data, type=np.int32)))
            self.elements[ElType.TET ] = all_data
            
            self.elements[ElType.TET] = all_data
            print("num_rests", num_rests, type(num_rests))
            self.num_elements[ElType.TET] += num_rests
            print("num_elements els", self.num_elements[ElType.TRIG])

        try:
            curve_order = mesh.GetCurveOrder()
        except:
            curve_order = 1
            print("Mesh has no curve order, using 1, update NGSolve/Netgen to detect curved meshes")
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

        for key in self.num_elements:
            self.num_elements[key] = int(self.num_elements[key])

        self._last_mesh_timestamp = mesh._timestamp


    def get_bounding_box(self):
        pmin, pmax = self.mesh.bounding_box
        return ([pmin[0], pmin[1], pmin[2]], [pmax[0], pmax[1], pmax[2]])

    def get_buffers(self, force_update=False):
        for eltype in self.elements:
            if eltype not in self.gpu_elements or force_update:
                self.gpu_elements[eltype] = buffer_from_array(
                    self.elements[eltype],
                    label="mesh_" + str(eltype),
                    reuse=self.gpu_elements.get(eltype, None),
                )
        if "subdivision" not in self.gpu_elements or force_update:
            self.gpu_elements["subdivision"] = uniform_from_array(
                np.array([self.subdivision], dtype=np.uint32),
                label="subdivision",
                reuse=self.gpu_elements.get("subdivision", None),
            )
        if "deformation_scale" not in self.gpu_elements or force_update:
            self.gpu_elements["deformation_scale"] = uniform_from_array(
                np.array([self.deformation_scale], dtype=np.float32),
                label="deformation_scale",
                reuse=self.gpu_elements.get("deformation_scale", None),
            )

        result = self.gpu_elements.copy()
        self._dummy_buffer = buffer_from_array(
            np.array([-1], dtype=np.float32),
            label="dummy_deformation",
            reuse=getattr(self, "_dummy_buffer", None)
        )

        result["deformation_2d"] = self._dummy_buffer
        result["deformation_3d"] = self._dummy_buffer
        if self.deformation_data:
            deform_buffers = self.deformation_data.get_buffers(include_mesh_data=False)
            result["deformation_2d"] = deform_buffers["data_2d"]
            if "data_3d" in deform_buffers:
                result["deformation_3d"] = deform_buffers["data_3d"]

        return result

class BaseMeshElements2d(Renderer):
    depthBias: int = 1
    depthBiasSlopeScale: float = 1.0
    vertex_entry_point: str = "vertexTrigP1Indexed"
    fragment_entry_point: str = "fragment2dElement"
    color = (0, 1, 0, 1)

    def __init__(self, data: MeshData, label="MeshElements2d", clipping=None):
        super().__init__(label=label)
        self.data = data
        self.clipping = clipping or Clipping()
        self.color_uniform = None

    def update(self, options: RenderOptions):
        self.clipping.update(options)
        self.data.update(options)
        self.subdivision = self.data.subdivision
        self.n_vertices = 3 * self.subdivision**2

        self._buffers = self.data.get_buffers()
        self.n_instances = self.data.num_elements[ElType.TRIG]
        print("n_instances", self.n_instances)
        self.color_uniform = buffer_from_array(
            np.array(self.color, dtype=np.float32), label="color_uniform", reuse=self.color_uniform
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
            BufferBinding(Binding.DEFORMATION_3D_VALUES, self._buffers["deformation_3d"]),
            UniformBinding(Binding.DEFORMATION_SCALE, self._buffers["deformation_scale"]),
            UniformBinding(Binding.SUBDIVISION, self._buffers["subdivision"]),
        ]
        if hasattr(self, "color_uniform"):
            bindings.append(BufferBinding(54, self.color_uniform))
        return bindings

    def get_shader_code(self):
        return read_shader_file("ngsolve/mesh.wgsl")


class MeshElements2d(BaseMeshElements2d):
    fragment_entry_point = "fragment2dElement"
    select_entry_point: str = "select2dElement"

    def __init__(
        self, data: MeshData, clipping=None, colors: list | None = None, label="MeshElements2d"
    ):
        super().__init__(data, label=label, clipping=clipping)
        if colors is None:
            mesh = data.mesh
            colors = [[int(ci * 255) for ci in fd.color] for fd in mesh.FaceDescriptors()]
        self.gpu_objects.colormap = Colormap(colormap=colors, minval=-0.5, maxval=len(colors) - 0.5)
        self.gpu_objects.colormap.discrete = 0
        self.gpu_objects.colormap.n_colors = 4 * len(colors)

    @property
    def colormap(self):
        return self.gpu_objects.colormap

    @colormap.setter
    def colormap(self, value: Colormap):
        self.gpu_objects.colormap = value

    def get_bindings(self):
        return super().get_bindings() + self.gpu_objects.colormap.get_bindings()


class MeshWireframe2d(BaseMeshElements2d):
    depthBias: int = 0
    depthBiasSlopeScale: float = 0.0
    topology: PrimitiveTopology = PrimitiveTopology.line_strip
    color = (0, 0, 0, 1)
    fragment_entry_point: str = "fragmentWireframe2d"
    vertex_entry_point: str = "vertexWireframe2d"

    def update(self, options: RenderOptions):
        super().update(options)
        self.n_vertices = 3 * self.subdivision + 1


class MeshSegments(Renderer):
    """Render 1D mesh elements (segments) as thick screen-space lines."""

    n_vertices: int = 4
    topology: PrimitiveTopology = PrimitiveTopology.triangle_strip

    # render on top of faces similar to GeometryEdgeRenderer
    depthBias: int = -5
    depthBiasSlopeScale: int = -5

    def __init__(self, data: MeshData, clipping=None, label: str = "MeshSegments"):
        super().__init__(label=label)
        self.data = data
        self.clipping = clipping or Clipping()
        self.thickness = 0.005
        self._buffers: dict[str, Buffer] = {}
        self._thickness_uniform: Buffer | None = None

    def get_bounding_box(self):
        return self.data.get_bounding_box()

    def update(self, options: RenderOptions):
        self.clipping.update(options)
        self.data.update(options)

        mesh = self.data.mesh
        vertices = self.data.elements["vertices"]

        try:
            segs = mesh.Elements1D().NumPy()
        except Exception:
            segs = []

        if len(segs) == 0:
            self.n_instances = 0
            self._buffers = {}
            return

        # build segment endpoint coordinates
        node_ids = segs["nodes"][:, :2] - 1
        seg_coords = vertices[node_ids].reshape(-1, 6).astype(np.float32)

        # indices and colors: one color entry per segment index
        indices = segs["index"].astype(np.uint32)
        edge_indices = indices - 1
        max_index = int(edge_indices.max()) if edge_indices.size > 0 else -1
        # TODO: Colors not yet available in NGSolve
        colors = np.zeros((max_index + 1, 4), dtype=np.float32)
        colors[:, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


        self.n_instances = seg_coords.shape[0]

        self._buffers = {}
        self._buffers["vertices"] = buffer_from_array(
            seg_coords.flatten(), label="mesh_segments_vertices"
        )
        self._buffers["colors"] = buffer_from_array(
            colors.flatten(), label="mesh_segments_colors"
        )
        self._buffers["index"] = buffer_from_array(
            edge_indices.astype(np.uint32), label="mesh_segments_indices"
        )

        self._thickness_uniform = uniform_from_array(
            np.array([self.thickness], dtype=np.float32),
            label="mesh_segments_thickness",
            reuse=self._thickness_uniform,
        )

    def get_shader_code(self):
        # reuse the thick-line implementation from geometry edges
        return read_shader_file("ngsolve/geo_edge.wgsl")

    def get_bindings(self):
        return [
            *self.clipping.get_bindings(),
            BufferBinding(90, self._buffers["vertices"]),
            BufferBinding(91, self._buffers["colors"]),
            UniformBinding(92, self._thickness_uniform),
            BufferBinding(93, self._buffers["index"]),
        ]


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

    def __init__(self, data: MeshData, clipping=None, colors: list | None = None):
        super().__init__(label="MeshElements3d")
        data.need_3d = True
        self.data = data
        self.clipping = clipping or Clipping()
        self._shrink = 1.0
        self.uniforms = None
        if colors is None:
            colors = [[255, 0, 0, 255] for _ in range(len(data.ngs_mesh.GetMaterials()))]
        self.gpu_objects.colormap = Colormap(colormap=colors, minval=-0.5, maxval=len(colors) - 0.5)
        self.gpu_objects.colormap.discrete = 0
        self.gpu_objects.colormap.n_colors = 4 * len(colors)

    @property
    def colormap(self):
        return self.gpu_objects.colormap

    @colormap.setter
    def colormap(self, value: Colormap):
        self.gpu_objects.colormap = value

    @property
    def shrink(self):
        return self._shrink

    @shrink.setter
    def shrink(self, value):
        self._shrink = value
        if self.uniforms is not None:
            self.uniforms.shrink = value
            self.uniforms.update_buffer()

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
            *self.gpu_objects.colormap.get_bindings(),
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

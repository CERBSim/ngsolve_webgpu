import math
from enum import Enum

import netgen.meshing
import ngsolve as ngs
import ngsolve.webgui
import numpy as np
from webgpu.font import Font
from webgpu.render_object import (
    BaseRenderObject,
    DataObject,
    RenderObject,
    _add_render_object,
)

# from webgpu.uniforms import Binding
from webgpu.utils import (
    BufferBinding,
    ShaderStage,
    TextureBinding,
    create_bind_group,
    decode_bytes,
    encode_bytes,
    read_shader_file,
)
from webgpu.webgpu_api import *
from webgpu.colormap import Colormap


class Binding:
    """Binding numbers for uniforms in shader code in uniforms.wgsl"""

    EDGES = 8
    TRIGS = 9
    TRIG_FUNCTION_VALUES = 10
    SEG_FUNCTION_VALUES = 11
    VERTICES = 12
    TRIGS_INDEX = 13
    GBUFFERLAM = 14

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


class MeshData(DataObject):
    vertices: bytes = b""
    trigs: bytes = b""
    edges: bytes = b""

    num_trigs: int = 0
    num_verts: int = 0
    num_edges: int = 0
    num_tets: int = 0

    # only for drawing the mesh, not needed for function values
    num_elements: dict[str, int]
    elements: dict[str, bytes]

    mesh: netgen.meshing.Mesh
    _last_mesh_timestamp: int = -1

    __BUFFER_NAMES = [
        "vertices",
        "trigs",
        "edges",
    ]
    __INT_NAMES = ["num_trigs", "num_verts", "num_edges"]

    def __init__(self, mesh: netgen.meshing.Mesh):
        _add_render_object(self)
        self.mesh = mesh
        self._buffers = {}

    def redraw(self, timestamp: float | None = None):
        super().redraw(mesh=self.mesh)

    def update(self, mesh: netgen.meshing.Mesh = None):
        if mesh:
            self.mesh = mesh

    def _create_data(self):
        # TODO: implement other element types than triangles
        # TODO: handle region correctly to draw only part of the mesh

        mesh = self.mesh
        for name in self.__BUFFER_NAMES:
            setattr(self, name, b"")
        for name in self.__INT_NAMES:
            setattr(self, name, 0)

        # Vertices
        self.num_verts = len(mesh.Points())
        vertices = np.zeros((self.num_verts, 3), dtype=np.float32)
        for i, p in enumerate(mesh.Points()):
            vertices[i] = list(p.p)

        self.pmin = np.min(vertices, axis=0)
        self.pmax = np.max(vertices, axis=0)
        self.vertices = vertices.tobytes()

        # Trigs TODO: Quads
        self.num_trigs = len(mesh.Elements2D())
        trigs = mesh.Elements2D().NumPy()
        trigs_data = np.zeros((self.num_trigs, 4), dtype=np.uint32)
        trigs_data[:, :3] = trigs["nodes"][:, :3] - 1
        trigs_data[:, 3] = trigs["index"]
        self.trigs = trigs_data.tobytes()

        # 3d Elements
        self.num_els = {eltype.name: 0 for eltype in ElType}
        elements = {eltype.name: [] for eltype in ElType}

        for i, el in enumerate(mesh.Elements3D()):
            eltype = ElType.from_dim_np(3, len(el.vertices))
            data = [p.nr - 1 for p in el.vertices]
            data.append(el.index)
            data.append(i)
            elements[eltype.name].append(data)
            self.num_els[eltype.name] += 1

        self.elements = {}
        for eltype in elements:
            self.elements[eltype] = np.array(
                elements[eltype], dtype=np.uint32
            ).tobytes()

        self._last_mesh_timestamp = mesh._timestamp

    def needs_update(self):
        return self._last_mesh_timestamp != self.mesh._timestamp

    def get_bounding_box(self):
        return (self.pmin, self.pmax)

    def _create_buffers(self, device: Device):
        data = {}
        for name in self.__BUFFER_NAMES:
            b = getattr(self, name)
            if b:
                data[name] = b

        for eltype in self.elements:
            data[eltype] = self.elements[eltype]

        buffers = {}
        for key in data:
            d = data[key]
            buffer = device.createBuffer(
                size=len(d), usage=BufferUsage.STORAGE | BufferUsage.COPY_DST
            )
            device.queue.writeBuffer(buffer, 0, d)
            buffers[key] = buffer

        self._buffers = buffers
        return self._buffers


class FunctionData(DataObject):
    mesh_data: MeshData
    function_data: bytes = b""
    cf: ngs.CoefficientFunction
    order: int
    _timestamp: float = -1

    def __init__(self, mesh_data: MeshData, cf: ngs.CoefficientFunction, order: int):
        _add_render_object(self)
        self.mesh_data = mesh_data
        self.cf = cf
        self.order = order

    def redraw(self, timestamp: float | None = None):
        self.mesh_data.redraw(timestamp)
        super().redraw(timestamp, cf=self.cf, order=self.order)

    def update(
        self, cf: ngs.CoefficientFunction | None = None, order: int | None = None
    ):
        if cf is not None:
            self.cf = cf
            self.function_data = b""
        if order is not None:
            self.order = order
            self.function_data = b""

    def _create_data(self):
        self.function_data = evaluate_cf(
            self.cf, ngs.Mesh(self.mesh_data.mesh), self.order
        )

    def _create_buffers(self, device: Device):
        buffer = device.createBuffer(
            size=len(self.function_data),
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST,
        )
        device.queue.writeBuffer(buffer, 0, self.function_data)
        return self.mesh_data.get_buffers(device) | {"function": buffer}

    def get_bounding_box(self):
        return self.mesh_data.get_bounding_box()


class CoefficientFunctionRenderObject(RenderObject):
    """Use "vertices", "index" and "trig_function_values" buffers to render a mesh"""

    def __init__(self, data: FunctionData, label=None):
        super().__init__(label=label)
        self.data = data
        self.n_vertices = 3

        # shift trigs behind to ensure that edges are rendered properly
        self.depthBias = 1
        self.depthBiasSlopeScale = 1.0
        self.vertex_entry_point = "vertexTrigP1Indexed"
        self.fragment_entry_point = "fragmentTrig"

    def redraw(self, timestamp: float | None = None):
        timestamp = self.data.redraw(timestamp)
        super().redraw(timestamp)

    def update(self):
        self.colormap = Colormap(self.device)
        self._buffers = self.data.get_buffers(self.device)
        self.n_instances = self.data.mesh_data.num_trigs
        self.create_render_pipeline()

    def get_bounding_box(self):
        return self.data.get_bounding_box()

    def get_shader_code(self):
        shader_code = ""

        for file_name in [
            "clipping.wgsl",
            "eval.wgsl",
            "mesh.wgsl",
            "shader.wgsl",
            "uniforms.wgsl",
        ]:
            shader_code += read_shader_file(file_name, __file__)

        shader_code += self.colormap.get_shader_code()
        shader_code += self.options.camera.get_shader_code()
        shader_code += self.options.light.get_shader_code()
        return shader_code

    def get_bindings(self):
        return [
            *self.options.get_bindings(),
            *self.colormap.get_bindings(),
            BufferBinding(Binding.TRIG_FUNCTION_VALUES, self._buffers["function"]),
            BufferBinding(Binding.VERTICES, self._buffers["vertices"]),
            BufferBinding(Binding.TRIGS_INDEX, self._buffers["trigs"]),
        ]


class Mesh3dElementsRenderObject(RenderObject):
    # TODO: currently not working
    def get_bindings(self):
        bindings = [
            *self.options.get_bindings(),
            *self.colormap.get_bindings(),
            BufferBinding(Binding.VERTICES, self._buffers["vertices"]),
        ]

        for eltype in ElType:
            if self.data.num_els[eltype.name]:
                bindings.append(
                    BufferBinding(
                        getattr(Binding, eltype.name), self._buffers[eltype.name]
                    )
                )

        return bindings

    def _create_pipelines(self):
        bind_layout, self._bind_group = self.create_bind_group()
        shader_module = self.gpu.shader_module

        self._pipelines = {}
        for eltype in ElType:

            if self.data.num_els[eltype.name] == 0:
                continue
            el_name = eltype.name.capitalize()

            self._pipelines[el_name.upper()] = self.device.createRenderPipeline(
                label=f"{self.label}:{el_name}",
                layout=self.device.createPipelineLayout([bind_layout]),
                vertex=VertexState(
                    module=shader_module,
                    entryPoint=f"vertexMesh{el_name}",
                ),
                fragment=FragmentState(
                    module=shader_module,
                    entryPoint="fragmentMesh",
                    targets=[ColorTargetState(format=self.gpu.format)],
                ),
                primitive=PrimitiveState(
                    topology=eltype.value.primitive_topology,
                ),
                depthStencil=DepthStencilState(
                    **self.gpu.depth_stencil,
                    # shift trigs behind to ensure that edges are rendered properly
                    depthBias=1.0,
                    depthBiasSlopeScale=1,
                ),
                multisample=self.gpu.multisample,
            )

    def render(self, encoder):
        render_pass = self.gpu.begin_render_pass(encoder)
        render_pass.setBindGroup(0, self._bind_group)
        for name, pipeline in self._pipelines.items():
            eltype = ElType[name].value
            render_pass.setPipeline(pipeline)
            render_pass.draw(eltype.num_vertices_per_primitive, self.data.num_els[name])
        render_pass.end()


def _get_bernstein_matrix_trig(n, intrule):
    """Create inverse vandermonde matrix for the Bernstein basis functions on a triangle of degree n and given integration points"""
    ndtrig = int((n + 1) * (n + 2) / 2)

    mat = ngs.Matrix(ndtrig, ndtrig)
    fac_n = math.factorial(n)
    for row, ip in enumerate(intrule):
        col = 0
        x = 1.0 - ip.point[0] - ip.point[1]
        y = ip.point[1]
        z = 1.0 - x - y
        for i in range(n + 1):
            factor = fac_n / math.factorial(i) * x**i
            for j in range(n + 1 - i):
                k = n - i - j
                factor2 = 1.0 / (math.factorial(j) * math.factorial(k))
                mat[row, col] = factor * factor2 * y**j * z**k
                col += 1
    return mat


def evaluate_cf(cf, mesh, order):
    """Evaluate a coefficient function on a mesh and returns the values as a flat array, ready to copy to the GPU as storage buffer.
    The first two entries are the function dimension and the polynomial order of the stored values.
    """
    comps = cf.dim
    int_points = ngsolve.webgui._make_trig(order)
    intrule = ngs.IntegrationRule(
        int_points,
        [
            0,
        ]
        * len(int_points),
    )
    ibmat = _get_bernstein_matrix_trig(order, intrule).I

    ndof = ibmat.h

    pts = mesh.MapToAllElements({ngs.ET.TRIG: intrule, ngs.ET.QUAD: intrule}, ngs.VOL)
    pmat = cf(pts)
    pmat = pmat.reshape(-1, ndof, comps)

    values = np.zeros((ndof, pmat.shape[0], comps), dtype=np.float32)
    for i in range(comps):
        ngsmat = ngs.Matrix(pmat[:, :, i].transpose())
        values[:, :, i] = ibmat * ngsmat

    values = values.transpose((1, 0, 2)).flatten()
    ret = np.concatenate(([np.float32(cf.dim), np.float32(order)], values))
    return ret.tobytes()


class PointNumbersRenderObject(RenderObject):
    """Render a point numbers of a mesh"""

    _buffers: dict

    def __init__(self, data, font_size=20, label=None):
        super().__init__(label=label)
        self.n_digits = 6
        self.data = data
        self.vertex_entry_point = "vertexPointNumber"
        self.fragment_entry_point = "fragmentText"
        self.n_vertices = self.n_digits * 6
        self.font_size = font_size

    def update(self):
        self.font = Font(self.canvas, self.font_size)
        self._buffers = self.data.get_buffers(self.device)
        self.n_instances = self.data.num_verts
        self.create_render_pipeline()

    def get_shader_code(self):
        shader_code = read_shader_file("clipping.wgsl", __file__)
        shader_code += read_shader_file("numbers.wgsl", __file__)
        shader_code += read_shader_file("uniforms.wgsl", __file__)
        shader_code += self.font.get_shader_code()
        shader_code += self.options.camera.get_shader_code()
        return shader_code

    def get_bounding_box(self):
        return self.data.get_bounding_box()

    def get_bindings(self):
        return [
            *self.options.get_bindings(),
            *self.font.get_bindings(),
            BufferBinding(Binding.VERTICES, self._buffers["vertices"]),
        ]

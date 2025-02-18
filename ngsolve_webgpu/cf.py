import math
import numpy as np

import ngsolve.webgui
import ngsolve as ngs

from webgpu.render_object import (
    DataObject,
    RenderObject,
    _add_render_object,
)
from webgpu.vectors import BaseVectorRenderObject, VectorRenderer
from webgpu.colormap import Colormap
from webgpu.utils import (
    BufferBinding,
    read_shader_file,
    buffer_from_array
)
from webgpu.webgpu_api import Device, BufferUsage

from .mesh import MeshData
from .mesh import Binding as MeshBinding

class Binding:
    TRIG_FUNCTION_VALUES = 10
    COMPONENT = 55

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

    vb = mesh.Materials(".*")
    if mesh.dim == 3:
        vb = mesh.Boundaries(".*")
    pts = mesh.MapToAllElements({ngs.ET.TRIG: intrule, ngs.ET.QUAD: intrule}, vb)
    pmat = cf(pts)
    minval, maxval = (
        (min(pmat.reshape(-1)), max(pmat.reshape(-1))) if len(pmat) else (0, 1)
    )
    pmat = pmat.reshape(-1, ndof, comps)

    values = np.zeros((ndof, pmat.shape[0], comps), dtype=np.float32)
    for i in range(comps):
        ngsmat = ngs.Matrix(pmat[:, :, i].transpose())
        values[:, :, i] = ibmat * ngsmat

    values = values.transpose((1, 0, 2)).flatten()
    ret = np.concatenate(([np.float32(cf.dim), np.float32(order)], values))
    return ret.tobytes(), minval, maxval


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
        self.function_data, self.minval, self.maxval = evaluate_cf(
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

def _change_cf_dim(me, value):
    me.component = value
    me.redraw()

class CoefficientFunctionRenderObject(RenderObject):
    """Use "vertices", "index" and "trig_function_values" buffers to render a mesh"""

    def __init__(self, data: FunctionData, component = 0,
                 label=None):
        super().__init__(label=label)
        self.data = data
        self.n_vertices = 3
        self.colormap = Colormap()

        # shift trigs behind to ensure that edges are rendered properly
        self.depthBias = 1
        self.depthBiasSlopeScale = 1.0
        self.vertex_entry_point = "vertexTrigP1Indexed"
        self.fragment_entry_point = "fragmentTrig"
        self.component = component
        self.colormap = Colormap()

    def redraw(self, timestamp: float | None = None):
        timestamp = self.data.redraw(timestamp)
        super().redraw(timestamp, component=self.component)

    def update(self, component=None):
        if component is not None:
            self.component = component
        self.colormap.options = self.options
        self._buffers = self.data.get_buffers(self.device)
        self.colormap.options = self.options
        if self.colormap.autoupdate:
            self.colormap.set_min_max(self.data.minval, self.data.maxval)
        self.colormap.update()
        self.n_instances = self.data.mesh_data.num_trigs
        self.component_buffer = buffer_from_array(np.array([self.component],
                                                           np.uint32))
        self.create_render_pipeline()

    def get_bounding_box(self):
        return self.data.get_bounding_box()

    def add_options_to_gui(self, gui):
        if self.data.cf.dim > 1:
            options = { "Norm" : 0 }
            for d in range(self.data.cf.dim):
                options[str(d)] = d+1
            gui.dropdown(func=_change_cf_dim,
                         objects=self, label="Component",
                         values=options)

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
            BufferBinding(MeshBinding.VERTICES, self._buffers["vertices"]),
            BufferBinding(MeshBinding.TRIGS_INDEX, self._buffers["trigs"]),
            BufferBinding(Binding.COMPONENT, self.component_buffer)
        ]


class VectorCFRenderer(VectorRenderer):
    def __init__(self, cf: ngs.CoefficientFunction,
                 mesh: ngs.Mesh, grid_size=20, size=None):
        # calling super-super class to not create points and vectors
        BaseVectorRenderObject.__init__(self)
        self.cf = cf
        self.mesh = mesh
        # this somehow segfaults in pyodide?
        self.grid_size = grid_size
        self.size = size

    def redraw(self, timestamp=None):
        super().redraw(timestamp=timestamp, cf=self.cf, mesh=self.mesh, grid_size=self.grid_size)

    def update(self, cf=None, mesh=None, grid_size=None, size=None):
        if cf is not None:
            self.cf = cf
        if mesh is not None:
            self.mesh = mesh
        if grid_size is not None:
            self.gridsize = grid_size
        if size is not None:
            self.size = size
        bb = self.mesh.ngmesh.bounding_box
        self.bounding_box = np.array([[bb[0][0], bb[0][1], bb[0][2]],
                                      [bb[1][0], bb[1][1], bb[1][2]]])
        vs = np.linspace(self.bounding_box[0][0], self.bounding_box[1][0], self.grid_size+1, endpoint=False)[1:]
        points = np.meshgrid(vs, vs)
        xvals = points[0].flatten()
        yvals = points[1].flatten()
        self.size = self.size or 1 / 60 * np.linalg.norm(
            self.bounding_box[1] - self.bounding_box[0]
        )
        mpts_ = self.mesh(xvals, yvals, 0.)
        pts, mpts = [], []
        for i in range(len(xvals)):
            if mpts_[i]["nr"] != -1:
                mpts.append(mpts_[i])
                pts.append([xvals[i], yvals[i], 0.])
        self.points = np.array(pts, dtype=np.float32).reshape(-1)
        values = self.cf(mpts)
        self.vectors = np.array([values[:,0], values[:,1], np.zeros_like(values[:,0])], dtype=np.float32).T.reshape(-1)
        super().update()
        

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
from webgpu.clipping import Clipping
from webgpu.utils import BufferBinding, read_shader_file, buffer_from_array
from webgpu.webgpu_api import Device, BufferUsage

from .mesh import MeshData
from .mesh import Binding as MeshBinding


class Binding:
    TRIG_FUNCTION_VALUES = 10
    COMPONENT = 55

_intrules_3d = {}
def get_3d_intrules(order):
    if order in _intrules_3d:
        return _intrules_3d[order]
    if order > 2:
        raise RuntimeError("only order 1 and 2 supported in 3D")
    p1_tets = {}
    p1_tets[ngs.ET.TET]   = [[(1,0,0), (0,1,0), (0,0,1), (0,0,0)]]
    p1_tets[ngs.ET.PYRAMID]=[[(1,0,0), (0,1,0), (0,0,1), (0,0,0)],
                             [(1,0,0), (0,1,0), (0,0,1), (1,1,0)]]
    p1_tets[ngs.ET.PRISM] = [[(1,0,0), (0,1,0), (0,0,1), (0,0,0)],
                             [(0,0,1), (0,1,0), (0,1,1), (1,0,0)],
                             [(1,0,1), (0,1,1), (1,0,0), (0,0,1)]]
    p1_tets[ngs.ET.HEX]   = [[(1,0,0), (0,1,0), (0,0,1), (0,0,0)],
                             [(0,1,1), (1,1,1), (1,1,0), (1,0,1)],
                             [(1,0,1), (0,1,1), (1,0,0), (0,0,1)],
                             [(0,1,1), (1,1,0), (0,1,0), (1,0,0)],
                             [(0,0,1), (0,1,0), (0,1,1), (1,0,0)],
                             [(1,0,1), (1,1,0), (0,1,1), (1,0,0)]]

    def makeP2Tets( p1_tets ):
        midpoint = lambda p0, p1: tuple((0.5*(p0[i]+p1[i]) for i in range(3)))
        p2_tets = []
        for tet in p1_tets:
            tet.append( midpoint(tet[0], tet[3]) )
            tet.append( midpoint(tet[1], tet[3]) )
            tet.append( midpoint(tet[2], tet[3]) )
            tet.append( midpoint(tet[0], tet[1]) )
            tet.append( midpoint(tet[0], tet[2]) )
            tet.append( midpoint(tet[1], tet[2]) )
            p2_tets.append(tet)
        return p2_tets
    rules = {}
    for eltype in p1_tets:
        points = p1_tets[eltype]
        if order == 2:
            points = makeP2Tets( points )
        rules[eltype] = ngs.IntegrationRule( sum(points, []) )
    _intrules_3d[order] = rules
    return rules

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

    if isinstance(mesh, ngs.Region):
        if mesh.VB() == ngs.VOL and mesh.mesh.dim == 3:
            region = mesh.Boundaries()
        else:
            region = mesh
    else:
        region = mesh.Materials(".*")
        if mesh.dim == 3:
            region = mesh.Boundaries(".*")
    pts = region.mesh.MapToAllElements(
        {ngs.ET.TRIG: intrule, ngs.ET.QUAD: intrule}, region
    )
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
    ret = np.concatenate(([np.float32(cf.dim), np.float32(order)], values.reshape(-1)))
    # print("ret = ", ret)
    return ret, minval, maxval

class FunctionData(DataObject):
    mesh_data: MeshData
    function_data: np.ndarray
    cf: ngs.CoefficientFunction
    order: int
    _timestamp: float = -1

    def __init__(self, mesh_data: MeshData, cf: ngs.CoefficientFunction, order: int):
        _add_render_object(self)
        self.mesh_data = mesh_data
        self.cf = cf
        self.order = order
        self.need_3d = False
        self._needs_update = True

    def redraw(self, timestamp: float | None = None):
        self.mesh_data.redraw(timestamp)
        self._needs_update = True
        super().redraw(timestamp, cf=self.cf, order=self.order)
        self._needs_update = False

    def update(
        self, cf: ngs.CoefficientFunction | None = None, order: int | None = None
    ):
        if cf is not None:
            self.cf = cf
            self.function_data = np.array([], dtype=np.float32)
            self._needs_update = True
        if order is not None:
            self.order = order
            self.function_data = np.array([], dtype=np.float32)
            self._needs_update = True

    def needs_update(self):
        if self._needs_update == True:
            self._needs_update = False
            return True
        return self._needs_update

    def _create_data(self):
        self.function_data, self.minval, self.maxval = evaluate_cf(
            self.cf, self.mesh_data.ngs_mesh, self.order
        )
        if self.need_3d:
            self.data_3d, minval, maxval = self.evaluate_3d(self.cf, self.mesh_data.ngs_mesh, self.order)
            self.minval = min(self.minval, minval)
            self.maxval = max(self.maxval, maxval)

    def _create_buffers(self, device: Device):
        self.mesh_data.need_3d = self.need_3d
        buffers = self.mesh_data.get_buffers(device)
        buffers["function"] = buffer_from_array(self.function_data)
        if self.need_3d:
            buffers["data_3d"] = buffer_from_array(self.data_3d)
        return buffers

    def get_bounding_box(self):
        return self.mesh_data.get_bounding_box()

    def evaluate_3d(self, cf, region, order):
        intrules = get_3d_intrules(order)
        if not isinstance(region, ngs.Region):
            region = region.Materials(".*")
        pts = region.mesh.MapToAllElements(intrules, region)
        vals = cf(pts)
        vmin, vmax = vals.min(), vals.max()
        ret = np.concatenate(([np.float32(cf.dim), np.float32(order)], vals.reshape(-1)), dtype=np.float32)
        return ret, vmin, vmax

def _change_cf_dim(me, value):
    me.component = value
    me.redraw()


class CoefficientFunctionRenderObject(RenderObject):
    """Use "vertices", "index" and "trig_function_values" buffers to render a mesh"""

    def __init__(self, data: FunctionData, component=0, label=None):
        super().__init__(label=label)
        self.data = data
        self.n_vertices = 3
        self.colormap = Colormap()
        self.clipping = Clipping()

        # shift trigs behind to ensure that edges are rendered properly
        self.depthBias = 1
        self.depthBiasSlopeScale = 1.0
        self.vertex_entry_point = "vertexTrigP1Indexed"
        self.fragment_entry_point = "fragmentTrig"
        self.component = component

    def redraw(self, timestamp: float | None = None):
        timestamp = self.data.redraw(timestamp)
        super().redraw(timestamp, component=self.component)

    def update(self, component=None):
        if component is not None:
            self.component = component
        self._buffers = self.data.get_buffers(self.device)
        self.colormap.options = self.options
        if self.colormap.autoupdate:
            self.colormap.set_min_max(
                self.data.minval, self.data.maxval, set_autoupdate=False
            )
        self.colormap.update()
        self.clipping.update()
        self.n_instances = self.data.mesh_data.num_trigs
        self.component_buffer = buffer_from_array(np.array([self.component], np.uint32))
        self.create_render_pipeline()

    def get_bounding_box(self):
        return self.data.get_bounding_box()

    def add_options_to_gui(self, gui):
        if self.data.cf.dim > 1:
            options = {"Norm": 0}
            for d in range(self.data.cf.dim):
                options[str(d)] = d + 1
            gui.dropdown(
                func=_change_cf_dim, objects=self, label="Component", values=options
            )

    def get_shader_code(self):
        shader_code = ""

        for file_name in [
            "eval.wgsl",
            "mesh.wgsl",
            "shader.wgsl",
            "uniforms.wgsl",
        ]:
            shader_code += read_shader_file(file_name, __file__)

        shader_code += self.colormap.get_shader_code()
        shader_code += self.clipping.get_shader_code()
        shader_code += self.options.camera.get_shader_code()
        shader_code += self.options.light.get_shader_code()
        return shader_code

    def get_bindings(self):
        return [
            *self.options.get_bindings(),
            *self.colormap.get_bindings(),
            *self.clipping.get_bindings(),
            BufferBinding(Binding.TRIG_FUNCTION_VALUES, self._buffers["function"]),
            BufferBinding(MeshBinding.VERTICES, self._buffers["vertices"]),
            BufferBinding(MeshBinding.TRIGS_INDEX, self._buffers["trigs"]),
            BufferBinding(Binding.COMPONENT, self.component_buffer),
        ]


class VectorCFRenderer(VectorRenderer):
    def __init__(
        self, cf: ngs.CoefficientFunction, mesh: ngs.Mesh, grid_size=20, size=None
    ):
        # calling super-super class to not create points and vectors
        BaseVectorRenderObject.__init__(self)
        self.cf = cf
        self.mesh = mesh
        # this somehow segfaults in pyodide?
        self.grid_size = grid_size
        self.size = size

    def redraw(self, timestamp=None):
        super().redraw(
            timestamp=timestamp, cf=self.cf, mesh=self.mesh, grid_size=self.grid_size
        )

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
        self.bounding_box = np.array(
            [[bb[0][0], bb[0][1], bb[0][2]], [bb[1][0], bb[1][1], bb[1][2]]]
        )
        vs = np.linspace(
            self.bounding_box[0][0],
            self.bounding_box[1][0],
            self.grid_size + 1,
            endpoint=False,
        )[1:]
        points = np.meshgrid(vs, vs)
        xvals = points[0].flatten()
        yvals = points[1].flatten()
        self.size = self.size or 1 / 60 * np.linalg.norm(
            self.bounding_box[1] - self.bounding_box[0]
        )
        mpts_ = self.mesh(xvals, yvals, 0.0)
        pts, mpts = [], []
        for i in range(len(xvals)):
            if mpts_[i]["nr"] != -1:
                mpts.append(mpts_[i])
                pts.append([xvals[i], yvals[i], 0.0])
        self.points = np.array(pts, dtype=np.float32).reshape(-1)
        values = self.cf(mpts)
        self.vectors = np.array(
            [values[:, 0], values[:, 1], np.zeros_like(values[:, 0])], dtype=np.float32
        ).T.reshape(-1)
        super().update()

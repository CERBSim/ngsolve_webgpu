import math

import ngsolve as ngs
import ngsolve.webgui
import numpy as np
from webgpu.clipping import Clipping
from webgpu.colormap import Colormap
from webgpu.renderer import Renderer, RenderOptions, check_timestamp
from webgpu.shapes import ShapeRenderer, generate_cylinder
from webgpu.utils import (
    BufferBinding,
    UniformBinding,
    buffer_from_array,
)
from webgpu.vectors import BaseVectorRenderer, VectorRenderer
from webgpu.webgpu_api import Buffer

from .mesh import Binding as MeshBinding, MeshElements2d
from .mesh import ElType, MeshData


class Binding:
    FUNCTION_VALUES_2D = 10
    COMPONENT = 55


_intrules_3d = {}


def get_3d_intrules(order):
    if order in _intrules_3d:
        return _intrules_3d[order]
    ref_pts = [
        [(order - i - j - k) / order, k / order, j / order]
        for i in range(order + 1)
        for j in range(order + 1 - i)
        for k in range(order + 1 - i - j)
    ]
    p1_tets = {ngs.ET.TET: [[(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]]}
    p1_tets[ngs.ET.PYRAMID] = [
        [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)],
        [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)],
    ]
    p1_tets[ngs.ET.PRISM] = [
        [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)],
        [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0)],
        [(1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 0, 1)],
    ]
    p1_tets[ngs.ET.HEX] = [
        [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)],
        [(0, 1, 1), (1, 1, 1), (1, 1, 0), (1, 0, 1)],
        [(1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 0, 1)],
        [(0, 1, 1), (1, 1, 0), (0, 1, 0), (1, 0, 0)],
        [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0)],
        [(1, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 0)],
    ]
    rules = {}
    if order > 1:
        ho_tets = {}
        for eltype in p1_tets:
            for tet in p1_tets[eltype]:
                ho_tets[eltype] = []
                for lam in ref_pts:
                    lami = [*lam, 1 - sum(lam)]
                    ho_tets[eltype].append(
                        [sum([lami[j] * tet[j][i] for j in range(4)]) for i in range(3)]
                    )
            rules[eltype] = ngs.IntegrationRule(ho_tets[eltype])
    else:
        for eltype in p1_tets:
            rules[eltype] = ngs.IntegrationRule(sum(p1_tets[eltype], []))
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
    pts = region.mesh.MapToAllElements({ngs.ET.TRIG: intrule, ngs.ET.QUAD: intrule}, region)
    pmat = cf(pts)
    minval, maxval = (min(pmat.reshape(-1)), max(pmat.reshape(-1))) if len(pmat) else (0, 1)
    pmat = pmat.reshape(-1, ndof, comps)

    values = np.zeros((ndof, pmat.shape[0], comps), dtype=np.float32)
    for i in range(comps):
        ngsmat = ngs.Matrix(pmat[:, :, i].transpose())
        values[:, :, i] = ibmat * ngsmat

    values = values.transpose((1, 0, 2)).flatten()
    ret = np.concatenate(([np.float32(cf.dim), np.float32(order)], values.reshape(-1)))
    # print("ret = ", ret)
    return ret, minval, maxval


class FunctionData:
    mesh_data: MeshData
    data_2d: np.ndarray | None = None
    data_3d: np.ndarray | None = None
    gpu_2d: Buffer | None = None
    gpu_3d: Buffer | None = None
    cf: ngs.CoefficientFunction
    order: int
    order_3d: int
    _timestamp: float = -1
    _needs_update: bool = True
    minval: float = 1e99
    maxval: float = -1e99

    def __init__(
        self,
        mesh_data: MeshData,
        cf: ngs.CoefficientFunction,
        order: int,
        order3d: int = -1,
    ):
        self.mesh_data = mesh_data
        self.cf = cf
        self.order = order
        self.order_3d = order if order3d == -1 else order3d
        self.need_3d = False

    @check_timestamp
    def update(self, options: RenderOptions):
        if self.need_3d:
            self.mesh_data.need_3d = True
        self.mesh_data.update(options)
        self._create_data()

    def set_needs_update(self):
        """Set this data to be updated on the next render call"""
        self._timestamp = -1

    @property
    def num_elements(self):
        return self.mesh_data.num_elements

    @property
    def subdivision(self):
        return self.mesh_data.subdivision

    @property
    def curvature_data(self):
        return self.mesh_data.curvature_data

    @property
    def deformation_data(self):
        return self.mesh_data.deformation_data

    def _create_data(self):
        self.gpu_2d = None
        self.gpu_3d = None
        self.data_2d, self.minval, self.maxval = evaluate_cf(
            self.cf, self.mesh_data.ngs_mesh, self.order
        )
        if self.need_3d:
            self.data_3d, minval, maxval = self.evaluate_3d(
                self.cf, self.mesh_data.ngs_mesh, self.order_3d
            )
            self.minval = min(self.minval, minval)
            self.maxval = max(self.maxval, maxval)

    def get_buffers(self):
        buffers = self.mesh_data.get_buffers().copy()
        if self.gpu_2d is None:
            self.gpu_2d = buffer_from_array(self.data_2d, label="function_data_2d")
            if self.data_3d is not None:
                self.gpu_3d = buffer_from_array(self.data_3d, label="function_data_3d")
        buffers["data_2d"] = self.gpu_2d
        if self.gpu_3d is not None:
            buffers["data_3d"] = self.gpu_3d
        self.data_2d = None
        self.data_3d = None
        return buffers

    def get_bounding_box(self):
        return self.mesh_data.get_bounding_box()

    def evaluate_3d(self, cf, region, order):
        intrules = get_3d_intrules(order)
        if not isinstance(region, ngs.Region):
            region = region.Materials(".*")
        pts = region.mesh.MapToAllElements(intrules, region)
        V_inv = vandermonde_3d(order).T
        vals = cf(pts).reshape(-1, len(intrules[ngs.ET.TET])).dot(V_inv)
        vmin, vmax = vals.min(), vals.max()
        ret = np.concatenate(
            ([np.float32(cf.dim), np.float32(order)], vals.reshape(-1)),
            dtype=np.float32,
        )
        return ret, vmin, vmax


_vandermonde_mats = {}


def vandermonde_3d(order):
    if order in _vandermonde_mats:
        return _vandermonde_mats[order]
    basis_indices = [
        (order - i - j - k, k, j, i)
        for i in range(order + 1)
        for j in range(order + 1 - i)
        for k in range(order + 1 - i - j)
    ]
    n = len(basis_indices)
    V = np.zeros((n, n))
    for r, (i, j, k, l) in enumerate(basis_indices):
        for c, (a, b, c2, d) in enumerate(basis_indices):
            multinom_coef = math.factorial(order) / (
                math.factorial(a) * math.factorial(b) * math.factorial(c2) * math.factorial(d)
            )
            V[r, c] = (
                multinom_coef
                * (i / order) ** a
                * (j / order) ** b
                * (k / order) ** c2
                * (l / order) ** d
            )
    _vandermonde_mats[order] = np.linalg.inv(V)
    return _vandermonde_mats[order]


class CFRenderer(MeshElements2d):
    """Use "vertices", "index" and "trig_function_values" buffers to render a mesh"""

    fragment_entry_point = "fragmentTrig"

    def __init__(
        self,
        data: FunctionData,
        component=-1,
        label="CFRenderer",
        clipping: Clipping = None,
        colormap: Colormap = None,
    ):
        super().__init__(data=data.mesh_data, label=label, clipping=clipping)
        self.data = data
        self.colormap = colormap or Colormap()
        self.component = component if self.data.cf.dim > 1 else 0

    def update(self, options: RenderOptions):
        self.data.update(options)
        super().update(options)
        if self.colormap.autoscale:
            self.colormap.set_min_max(self.data.minval, self.data.maxval, set_autoscale=False)
        self.colormap._update_and_create_render_pipeline(options)
        self.component_buffer = buffer_from_array(np.array([self.component], np.int32))
        self.shader_defines["MAX_EVAL_ORDER"] = self.data.order

    def get_bounding_box(self):
        return self.data.get_bounding_box()

    def add_options_to_gui(self, gui):
        if gui is None:
            return
        if self.data.cf.dim > 1:
            options = {"Norm": -1}
            for d in range(self.data.cf.dim):
                options[str(d)] = d
            gui.dropdown(func=self.change_cf_dim, label="Component", values=options)

    def change_cf_dim(self, value):
        self.component = value
        self.component_buffer = buffer_from_array(np.array([self.component], np.int32))
        self.set_needs_update()

    def get_bindings(self):
        return [
            *super().get_bindings(),
            *self.colormap.get_bindings(),
            BufferBinding(Binding.FUNCTION_VALUES_2D, self._buffers["data_2d"]),
            BufferBinding(Binding.COMPONENT, self.component_buffer),
        ]

    def set_needs_update(self):
        self.data._timestamp = -1
        super().set_needs_update()


class VectorCFRenderer(VectorRenderer):
    def __init__(
        self,
        cf: ngs.CoefficientFunction,
        mesh: ngs.Mesh,
        grid_size=20,
        size=None,
    ):
        # calling super-super class to not create points and vectors
        BaseVectorRenderer.__init__(self)
        self.cf = cf
        self.mesh = mesh
        # this somehow segfaults in pyodide?
        self.grid_size = grid_size
        self.size = size

    def redraw(self, timestamp=None):
        super().redraw(timestamp=timestamp, cf=self.cf, mesh=self.mesh, grid_size=self.grid_size)

    def update(self, options: RenderOptions):
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
        super().update(options)


class FieldLines(ShapeRenderer):
    def __init__(
        self,
        cf,
        start_region: ngs.Region | ngs.Mesh,
        num_lines: int = 100,
        length: float = 0.5,
        max_points_per_line: float = 500,
        thickness: float = 0.0015,
        tolerance: float = 0.0005,
        direction: int = 0,
    ):
        self.fieldline_options = {
            "thickness": thickness,
            "num_lines": num_lines,
            "length": length,
            "max_points_per_line": max_points_per_line,
            "tolerance": tolerance,
            "direction": direction,
        }
        self.cf = cf
        if isinstance(start_region, ngs.Mesh):
            self.mesh = start_region
            self.start_region = start_region.Materials(".*")
        else:
            self.start_region = start_region
            self.mesh = start_region.mesh

        bbox = self.mesh.ngmesh.bounding_box
        thickness = (bbox[1] - bbox[0]).Norm() * thickness

        cyl = generate_cylinder(8, thickness, 1.0, top_face=False, bottom_face=False)

        super().__init__(cyl, None, None)
        self.scale_mode = ShapeRenderer.SCALE_Z

    def get_bounding_box(self):
        pmin, pmax = self.mesh.ngmesh.bounding_box
        return ([pmin[0], pmin[1], pmin[2]], [pmax[0], pmax[1], pmax[2]])

    def update(self, options):
        from ngsolve.webgui import FieldLines

        data = FieldLines(self.cf, self.start_region, **self.fieldline_options)
        self.positions = data["pstart"]
        self.directions = data["pend"]
        self.directions = self.directions - self.positions
        self.values = data["value"]
        super().update(options)

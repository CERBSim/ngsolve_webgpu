import math
import time
import threading
from typing_extensions import deprecated
import typing

import numpy as np
from webgpu.clipping import Clipping
from webgpu.colormap import Colormap
from webgpu.renderer import Renderer, RenderOptions, check_timestamp
from webgpu.shapes import ShapeRenderer, generate_cylinder
from webgpu.utils import BufferBinding, UniformBinding, buffer_from_array, write_array_to_buffer
from webgpu.renderer import BaseRenderer, RenderOptions, check_timestamp
from webgpu.uniforms import UniformBase, ct
from webgpu.utils import (
    BufferBinding,
    UniformBinding,
    buffer_from_array,
)
from webgpu.renderer import RenderOptions, check_timestamp
from webgpu.utils import (
    BufferBinding,
    buffer_from_array,
)
from webgpu.vectors import BaseVectorRenderer, VectorRenderer
from webgpu.webgpu_api import Buffer

from .mesh import Binding as MeshBinding, BaseMeshElements2d
from .mesh import ElType, MeshData
from .mesh import Binding as MeshBinding, MeshElements2d
from .mesh import ElType, MeshData
from .mesh import MeshElements2d
from .mesh import MeshData

if typing.TYPE_CHECKING:
    import ngsolve as ngs


class Binding:
    FUNCTION_VALUES_2D = 10
    FUNCTION_VALUES_3D = 13
    FUNCTION_SETTINGS = 55
    COMPLEX_SETTINGS = 56


_intrules_3d = {}


def get_3d_intrules(order):
    import ngsolve as ngs

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
            ho_tets[eltype] = []
            for tet in p1_tets[eltype]:
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
    import ngsolve as ngs
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
    The first three entries are the function dimension, the polynomial order, and the is_complex flag.
    """
    import ngsolve as ngs
    import ngsolve.webgui
    comps = cf.dim
    is_complex = cf.is_complex
    int_points = ngsolve.webgui._make_trig(order)
    ndof = len(int_points)
    
    make_rule = lambda pts: ngs.IntegrationRule(pts, [0] * len(pts))
    
    trig_rule = make_rule(int_points)
    
    ibmat = _get_bernstein_matrix_trig(order, trig_rule).I

    quad_points = ngsolve.webgui._make_quad(order)
    quad_rule1 = make_rule(quad_points[:ndof])
    quad_rule2 = make_rule(quad_points[ndof:])
    
    if isinstance(mesh, ngs.Region):
        if mesh.VB() == ngs.VOL and mesh.mesh.dim == 3:
            region = mesh.Boundaries()
        else:
            region = mesh
    else:
        region = mesh.Materials(".*")
        if mesh.dim == 3:
            region = mesh.Boundaries(".*")
    with ngs.TaskManager():
        pts = region.mesh.MapToAllElements({ngs.ET.TRIG: trig_rule, ngs.ET.QUAD: quad_rule1}, region)
        pmat = cf(pts)
        
    with ngs.TaskManager():
        pts2 = region.mesh.MapToAllElements({ngs.ET.QUAD: quad_rule2}, region)
        pmat2 = cf(pts2)
    
    pmat = np.concatenate((pmat, pmat2), axis=0)
    pmat = pmat.reshape(-1, ndof, comps)

    if is_complex:
        # For complex: compute min/max from absolute values
        pmat_abs = np.abs(pmat)  # element-wise absolute value
        minval = np.min(pmat_abs, axis=(0, 1))
        maxval = np.max(pmat_abs, axis=(0, 1))
        norm = np.linalg.norm(pmat_abs, axis=2)
        minval = [float(np.min(norm))] + [float(v) for v in minval]
        maxval = [float(np.max(norm))] + [float(v) for v in maxval]

        # Bernstein transform: split into real/imag, transform separately
        # Output layout: interleaved [Re(comp0), Im(comp0), Re(comp1), Im(comp1), ...]
        # stride per DOF = 2 * comps
        pmat_re = pmat.real
        pmat_im = pmat.imag

        values = np.zeros((ndof, pmat.shape[0], comps * 2), dtype=np.float32)
        with ngs.TaskManager():
            for i in range(comps):
                ngsmat_re = ngs.Matrix(pmat_re[:, :, i].transpose().copy())
                ngsmat_im = ngs.Matrix(pmat_im[:, :, i].transpose().copy())
                values[:, :, i * 2] = ibmat * ngsmat_re
                values[:, :, i * 2 + 1] = ibmat * ngsmat_im

        values = values.transpose((1, 0, 2)).flatten()
        ret = np.concatenate(([np.float32(comps), np.float32(order), np.float32(1.0)], values.reshape(-1)))
    else:
        minval = np.min(pmat.real, axis=(0, 1))
        maxval = np.max(pmat.real, axis=(0, 1))
        norm = np.linalg.norm(pmat.real, axis=2)
        minval = [float(np.min(norm))] + [float(v) for v in minval]
        maxval = [float(np.max(norm))] + [float(v) for v in maxval]

        values = np.zeros((ndof, pmat.shape[0], comps), dtype=np.float32)
        with ngs.TaskManager():
            for i in range(comps):
                ngsmat = ngs.Matrix(pmat[:, :, i].transpose())
                values[:, :, i] = ibmat * ngsmat

        values = values.transpose((1, 0, 2)).flatten()
        ret = np.concatenate(([np.float32(comps), np.float32(order), np.float32(0.0)], values.reshape(-1)))

    return ret, minval, maxval


class FunctionData:
    mesh_data: MeshData
    data_2d: np.ndarray | None = None
    data_3d: np.ndarray | None = None
    gpu_2d: Buffer | None = None
    gpu_3d: Buffer | None = None
    cf: "ngs.CoefficientFunction"
    order: int
    order_3d: int
    _timestamp: float = -1
    minval: list[float]
    maxval: list[float]

    def __init__(
        self,
        mesh_data: MeshData,
        cf: "ngs.CoefficientFunction",
        order: int,
        order3d: int = -1,
    ):
        self.mesh_data = mesh_data
        self.cf = cf
        self.order = order
        self.order_3d = order if order3d == -1 else order3d
        self._need_3d = mesh_data.need_3d

    @check_timestamp
    def update(self, options: RenderOptions):
        self.mesh_data.update(options)
        self._create_data()

    @property
    def needs_update(self) -> bool:
        return self.mesh_data.needs_update or self._timestamp < 0

    def set_needs_update(self):
        """Set this data to be updated on the next render call"""
        self._timestamp = -1

    @property
    def need_3d(self):
        return self._need_3d

    @need_3d.setter
    def need_3d(self, value: bool):
        if self._need_3d == value:
            return
        self.mesh_data.need_3d = value
        self._need_3d = value
        self.set_needs_update()

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
        try:
            self.data_2d, self.minval, self.maxval = evaluate_cf(
                self.cf, self.mesh_data.ngs_mesh, self.order
            )
        except Exception:
            self.data_2d = None
            self.minval = [1e99] * (self.cf.dim + 1)
            self.maxval = [-1e99] * (self.cf.dim + 1)
        if self.need_3d:
            try:
                self.data_3d, minval, maxval = self.evaluate_3d(
                self.cf, self.mesh_data.ngs_mesh, self.order_3d
                )
                self.minval = [min(v1, v2) for v1, v2 in zip(self.minval, minval)]
                self.maxval = [max(v1, v2) for v1, v2 in zip(self.maxval, maxval)]
            except Exception:
                self.data_3d = None

    def get_buffers(self, include_mesh_data=True):
        if include_mesh_data:
            buffers = self.mesh_data.get_buffers().copy()
        else:
            buffers = {}

        if self.data_2d is not None:
            self.gpu_2d = buffer_from_array(
                self.data_2d, label="function_data_2d", reuse=self.gpu_2d
            )
            if self.data_3d is not None:
                self.gpu_3d = buffer_from_array(
                    self.data_3d, label="function_data_3d", reuse=self.gpu_3d
                )
            buffers["data_2d"] = self.gpu_2d

        if self.data_3d is not None:
            buffers["data_3d"] = self.gpu_3d
            self._need_gpu_update = False

        return buffers

    def get_bounding_box(self):
        return self.mesh_data.get_bounding_box()

    def evaluate_3d_old(self, cf, region, order):
        import ngsolve as ngs
        if isinstance(region, ngs.Mesh):
            region = region.Materials(".*")
        if region.mesh.dim != 3 or region.VB() != ngs.VOL:
            return np.array([]), [1e99, 1e99], [-1e99, -1e99]
        import ngsolve as ngs

        intrules = get_3d_intrules(order)
        ndof = len(intrules[ngs.ET.TET])

        """
        if not isinstance(region, ngs.Region):
            region = region.Materials(".*")
        """

        with ngs.TaskManager():
            pts = region.mesh.MapToAllElements(intrules, region)
            V_inv = vandermonde_3d(order).T
            pmat = cf(pts).reshape(-1, len(intrules[ngs.ET.TET]))
            comps = cf.dim
            comp_vals = pmat.reshape(-1, ndof, comps)

            I, K, L = comp_vals.shape[0], comp_vals.shape[2], V_inv.shape[1]
            vals = np.zeros((I, L, K))

            ibmat = ngs.Matrix(V_inv.T)  # note the transpose for matching dimensions

            for k in range(K):
                ngsmat = ngs.Matrix(comp_vals[:, :, k].T)
                result = ibmat * ngsmat
                vals[:, :, k] = np.array(result).T

        minval = np.min(comp_vals, axis=(0, 1))
        maxval = np.max(comp_vals, axis=(0, 1))
        if comps > 1:
            norm = np.linalg.norm(comp_vals, axis=2)
        else:
            norm = np.abs(comp_vals)
        vmin = [float(np.min(norm))] + [float(v) for v in minval]
        vmax = [float(np.max(norm))] + [float(v) for v in maxval]

        ret = np.concatenate(
            ([np.float32(cf.dim), np.float32(order), np.float32(0.0)], vals.reshape(-1)),
            dtype=np.float32,
        )
        return ret, vmin, vmax
    
    def evaluate_3d(self, cf, region, order):
        import ngsolve as ngs
        import ngsolve.webgui
        
        comps = cf.dim
        intrules = get_3d_intrules(order)

        # Defining the integration rules
        tet_rule = intrules[ngs.ET.TET]
        ndof = len(intrules[ngs.ET.TET])

        def split_rule(intrule) : 
            rule1 = [intrule[i].point for i in range(ndof)]  
            rule2 = [intrule[i].point for i in range(ndof, len(intrule))]

            rule1 = ngs.IntegrationRule(rule1, [0]*len(rule1)) 
            rule2 = ngs.IntegrationRule(rule2, [0]*len(rule2))
             
            return rule1, rule2
        
        pyra_rule1, pyra_rule2 = split_rule(intrules[ngs.ET.PYRAMID])
        prism_rule1, prism_rule2 = split_rule(intrules[ngs.ET.PRISM])
        hex_rule1, hex_rule2 = split_rule(intrules[ngs.ET.HEX])

        if isinstance(region, ngs.Mesh):
            region = region.Materials(".*")
        if region.mesh.dim != 3 or region.VB() != ngs.VOL:
            return np.array([]), [1e99, 1e99], [-1e99, -1e99]
        
        # TET
        with ngs.TaskManager():
            pts = region.mesh.MapToAllElements({ngs.ET.TET: tet_rule, 
                                                ngs.ET.PYRAMID: pyra_rule1,
                                                ngs.ET.PRISM: prism_rule1,
                                                ngs.ET.HEX: hex_rule1,}
                                                , region)
            pmat = cf(pts)

        # PYRA
        with ngs.TaskManager():
            pts_pyra = region.mesh.MapToAllElements({ngs.ET.PYRAMID: pyra_rule2}, region)
            pmat_pyra = cf(pts_pyra)
        
        # PRISM
        with ngs.TaskManager():
            pts_prism = region.mesh.MapToAllElements({ngs.ET.PRISM: prism_rule2}, region)
            pmat_prism = cf(pts_prism)

        # HEX
        with ngs.TaskManager():
            pts_hex = region.mesh.MapToAllElements({ngs.ET.HEX: hex_rule2}, region)
            pmat_hex = cf(pts_hex)

        pmat = np.concatenate((pmat, pmat_pyra, pmat_prism, pmat_hex))
        pmat = pmat.reshape(-1, ndof, comps)

        is_complex = cf.is_complex

        with ngs.TaskManager():
            V_inv = vandermonde_3d(order).T
            I, K, L = pmat.shape[0], pmat.shape[2], V_inv.shape[1]

            ibmat = ngs.Matrix(V_inv.T)  # note the transpose for matching dimensions

            if is_complex:
                vals = np.zeros((I, L, K * 2), dtype=np.float32)
                for k in range(K):
                    ngsmat_re = ngs.Matrix(pmat[:, :, k].real.T.copy())
                    ngsmat_im = ngs.Matrix(pmat[:, :, k].imag.T.copy())
                    vals[:, :, k * 2] = np.array(ibmat * ngsmat_re).T
                    vals[:, :, k * 2 + 1] = np.array(ibmat * ngsmat_im).T
            else:
                vals = np.zeros((I, L, K))
                for k in range(K):
                    ngsmat = ngs.Matrix(pmat[:, :, k].T)
                    result = ibmat * ngsmat
                    vals[:, :, k] = np.array(result).T

        if is_complex:
            pmat_abs = np.abs(pmat)
            minval = np.min(pmat_abs, axis=(0, 1))
            maxval = np.max(pmat_abs, axis=(0, 1))
            norm = np.linalg.norm(pmat_abs, axis=2)
        else:
            minval = np.min(pmat, axis=(0, 1))
            maxval = np.max(pmat, axis=(0, 1))
            if comps > 1:
                norm = np.linalg.norm(pmat, axis=2)
            else:
                norm = np.abs(pmat)
        vmin = [float(np.min(norm))] + [float(v) for v in minval]
        vmax = [float(np.max(norm))] + [float(v) for v in maxval]

        ret = np.concatenate(
            ([np.float32(cf.dim), np.float32(order), np.float32(1.0 if is_complex else 0.0)], vals.reshape(-1)),
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


class FunctionUniform(UniformBase):
    _binding = Binding.FUNCTION_SETTINGS
    _fields_ = [
        ("component", ct.c_int32),
        ("padding", ct.c_uint32*3),
    ]
    
class FunctionSettings(BaseRenderer):
    def __init__(self, component):
        super().__init__()
        self._component = component
        self.uniform = None

    def update(self, options: RenderOptions):
        if self.uniform is None:
            self.uniform = FunctionUniform(component=self._component)
            self.uniform.update_buffer()

    def get_bindings(self):
        return self.uniform.get_bindings()
        
    @property
    def component(self):
        return self._component
        
    @component.setter
    def component(self, value):
        self._component = value
        if self.uniform is not None:
            self.uniform.component = value
            self.uniform.update_buffer()


class ComplexUniform(UniformBase):
    _binding = Binding.COMPLEX_SETTINGS
    _fields_ = [
        ("mode", ct.c_uint32),      # 0=PhaseRotate, 1=Abs, 2=Arg
        ("phase", ct.c_float),      # phase angle for mode 0
        ("padding", ct.c_uint32*2),
    ]


class ComplexSettings(BaseRenderer):
    PHASE_ROTATE = 0
    ABS = 1
    ARG = 2

    def __init__(self, mode=0, phase=0.0):
        super().__init__()
        self._mode = mode
        self._phase = phase
        self.uniform = None

    def update(self, options: RenderOptions):
        if self.uniform is None:
            self.uniform = ComplexUniform(mode=self._mode, phase=self._phase)
            self.uniform.update_buffer()

    def get_bindings(self):
        return self.uniform.get_bindings()

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
        if self.uniform is not None:
            self.uniform.mode = value
            self.uniform.update_buffer()

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value
        if self.uniform is not None:
            self.uniform.phase = value
            self.uniform.update_buffer()


class PhaseAnimation:
    """Sweeps the complex phase uniform and re-renders the scene."""

    def __init__(self, complex_settings, scene, speed=1.0, fps=60):
        self.complex_settings = complex_settings
        self.scene = scene
        self.speed = speed
        self._fps = fps
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self.complex_settings.mode = ComplexSettings.PHASE_ROTATE
        self._running = True
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread = None

    @property
    def running(self):
        return self._running

    def _loop(self):
        while self._running:
            t = time.time() - self._t0
            phase = (t * self.speed * 2 * math.pi) % (2 * math.pi)
            self.complex_settings.phase = phase
            self.scene.render()
            time.sleep(1 / self._fps)


class CFRenderer(BaseMeshElements2d):
    """Use "vertices", "index" and "trig_function_values" buffers to render a mesh"""

    fragment_entry_point = "fragmentTrig"

    def __init__(
        self,
        data: FunctionData,
        component=None,
        label="CFRenderer",
        clipping: Clipping = None,
        colormap: Colormap = None,
        symmetry=None,
    ):
        super().__init__(data=data.mesh_data, label=label, clipping=clipping, symmetry=symmetry)
        self.data = data
        self.gpu_objects.colormap = colormap or Colormap()
        self._on_component_change = []
        if component is None:
            component = -1 if self.data.cf.dim > 1 else 0
        self.gpu_objects.settings = FunctionSettings(component=component)
        self.gpu_objects.complex_settings = ComplexSettings()
        self._phase_animation = None
        self._scene = None
        self._anim_speed = 1.0
        
    @property
    def colormap(self):
        return self.gpu_objects.colormap

    @colormap.setter
    def colormap(self, value: Colormap):
        self.gpu_objects.colormap = value
        
    @property
    def component(self):
        return self.gpu_objects.settings.component

    def update(self, options: RenderOptions):
        self.data.update(options)
        if self.data.data_2d is None:
            self.active = False
            return
        super().update(options)
        if self.gpu_objects.colormap.autoscale:
            self.gpu_objects.colormap.set_min_max(
                self.data.minval[self.component + 1],
                self.data.maxval[self.component + 1],
                set_autoscale=False,
            )
        self.shader_defines["MAX_EVAL_ORDER"] = self.data.order

    def on_component_change(self, callback):
        self._on_component_change.append(callback)

    def get_bounding_box(self):
        return self.data.get_bounding_box()

    def add_options_to_gui(self, gui):
        if gui is None:
            return
        if self.data.cf.dim > 1:
            options = {"Norm": -1}
            for d in range(self.data.cf.dim):
                options[str(d)] = d
            gui.dropdown(func=self.set_component, label="Component", values=options)
        if self.data.cf.is_complex:
            f = gui.folder("Complex")
            complex_options = {"Real": "real", "Imag": "imag", "Abs": "abs", "Arg": "arg"}
            f.dropdown(func=self.set_complex_mode, label="Mode", values=complex_options)
            f.slider(0.0, func=self._set_phase_from_gui, min=0.0, max=6.283, label="Phase")
            f.checkbox(func=self._toggle_animation, label="Animate", value=False)
            f.slider(1.0, func=self._set_speed_from_gui, min=0.1, max=5.0, label="Speed")

    @deprecated("Use set_component instead")
    def change_cf_dim(self, value):
        self.set_component(value)
        
    def set_component(self, component: int):
        self.gpu_objects.settings.component = component
        for cb in self._on_component_change:
            cb(component)
        self.set_needs_update()

    def set_complex_mode(self, mode):
        """Set complex visualization mode: 'real', 'imag', 'abs', 'arg'"""
        import math
        mode_map = {
            "real": (ComplexSettings.PHASE_ROTATE, 0.0),
            "imag": (ComplexSettings.PHASE_ROTATE, -math.pi / 2),
            "abs": (ComplexSettings.ABS, None),
            "arg": (ComplexSettings.ARG, None),
        }
        if isinstance(mode, str):
            shader_mode, phase = mode_map[mode.lower()]
        else:
            shader_mode, phase = mode, None
        self.gpu_objects.complex_settings.mode = shader_mode
        if phase is not None:
            self.gpu_objects.complex_settings.phase = phase

    def set_phase(self, phase: float):
        """Set the phase angle for complex animate mode"""
        self.gpu_objects.complex_settings.phase = phase

    def animate_phase(self, scene=None, speed=1.0, fps=60):
        """Start phase-sweep animation."""
        scene = scene or self._scene
        if scene is None:
            raise ValueError("No scene available. Pass scene or call from a Draw()-created renderer.")
        self.stop_animation()
        self._phase_animation = PhaseAnimation(
            self.gpu_objects.complex_settings, scene, speed=speed, fps=fps
        )
        self._phase_animation.start()

    def stop_animation(self):
        """Stop phase-sweep animation."""
        if self._phase_animation is not None:
            self._phase_animation.stop()
            self._phase_animation = None

    def _set_phase_from_gui(self, value):
        self.gpu_objects.complex_settings.mode = ComplexSettings.PHASE_ROTATE
        self.set_phase(value)
        if self._scene:
            self._scene.render()

    def _set_speed_from_gui(self, value):
        self._anim_speed = value
        if self._phase_animation is not None:
            self._phase_animation.speed = value

    def _toggle_animation(self, value):
        if value:
            self.animate_phase(speed=self._anim_speed)
        else:
            self.stop_animation()
            self.set_complex_mode("real")
            if self._scene:
                self._scene.render()

    def get_bindings(self):
        return [
            *super().get_bindings(),
            *self.gpu_objects.colormap.get_bindings(),
            *self.gpu_objects.settings.get_bindings(),
            BufferBinding(Binding.FUNCTION_VALUES_2D, self._buffers["data_2d"]),
        ]

    def set_needs_update(self):
        self.data._timestamp = -1
        super().set_needs_update()


class VectorCFRenderer(VectorRenderer):
    def __init__(
        self,
        cf: "ngs.CoefficientFunction",
        mesh: "ngs.Mesh",
        grid_size=20,
        size=None,
    ):
        # calling super-super class to not create points and vectors
        BaseVectorRenderer.__init__(self)
        self.scale_with_vector_length = False
        self.cf = cf
        self.mesh = mesh
        # this somehow segfaults in pyodide?
        self.grid_size = grid_size
        self.size = size

    def get_bounding_box(self):
        bb = self.mesh.ngmesh.bounding_box
        pmin = [bb[0][0], bb[0][1], bb[0][2]]
        pmax = [bb[1][0], bb[1][1], bb[1][2]]
        return (pmin, pmax)

    def redraw(self, timestamp=None):
        super().redraw(timestamp=timestamp, cf=self.cf, mesh=self.mesh, grid_size=self.grid_size)

    def update(self, options: RenderOptions):
        self.options = options
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
        start_region: "ngs.Region | ngs.Mesh",
        num_lines: int = 100,
        length: float = 0.5,
        max_points_per_line: float = 500,
        thickness: float = 0.0015,
        tolerance: float = 0.0005,
        direction: int = 0,
        colormap=None,
        clipping=None,
    ):
        import ngsolve as ngs
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

        super().__init__(cyl, None, None, colormap=colormap, clipping=clipping)
        self.scale_mode = ShapeRenderer.SCALE_Z

    def get_bounding_box(self):
        pmin, pmax = self.mesh.ngmesh.bounding_box
        return ([pmin[0], pmin[1], pmin[2]], [pmax[0], pmax[1], pmax[2]])

    def update(self, options):
        from ngsolve.webgui import FieldLines

        data = FieldLines(self.cf, self.start_region, **self.fieldline_options)
        bbox = self.mesh.ngmesh.bounding_box
        thickness = (bbox[1] - bbox[0]).Norm() * self.fieldline_options["thickness"]
        self.shape_data = generate_cylinder(8, thickness, 1.0, top_face=False, bottom_face=False)

        self.positions = data["pstart"]
        self.directions = data["pend"]
        self.directions = self.directions - self.positions
        self.values = data["value"]
        super().update(options)

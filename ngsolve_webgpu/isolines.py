"""Isoline rendering settings, uniform management, and renderer classes."""

import ctypes as ct

from webgpu.renderer import BaseRenderer, RenderOptions
from webgpu.uniforms import UniformBase
from webgpu.utils import read_shader_file

ISOLINE_BINDING = 60


class IsolineUniform(UniformBase):
    _binding = ISOLINE_BINDING
    _fields_ = [
        ("n_lines", ct.c_uint32),
        ("thickness", ct.c_float),
        ("show_field", ct.c_uint32),
        ("padding", ct.c_uint32),
        ("color_r", ct.c_float),
        ("color_g", ct.c_float),
        ("color_b", ct.c_float),
        ("color_a", ct.c_float),
    ]


class IsolineSettings(BaseRenderer):
    """Manages isoline uniform state and GPU buffer.

    Parameters
    ----------
    n_lines : int
        Number of evenly-spaced isoline levels between colormap min and max.
        Set to 0 to disable.
    thickness : float
        Line thickness in approximate pixels (default 1.5).
    color : tuple
        RGBA color for the isolines (default black, fully opaque).
    show_field : bool
        If True, render the colored field with isolines on top.
        If False, render only the isolines (discard everything else).
    """

    def __init__(self, n_lines=10, thickness=1.5, color=(0.0, 0.0, 0.0, 1.0), show_field=True):
        super().__init__()
        self._n_lines = n_lines
        self._thickness = thickness
        self._color = tuple(color)
        self._show_field = show_field
        self.uniform = None

    def update(self, options: RenderOptions):
        if self.uniform is None:
            self.uniform = IsolineUniform(
                n_lines=self._n_lines,
                thickness=self._thickness,
                show_field=1 if self._show_field else 0,
                color_r=self._color[0],
                color_g=self._color[1],
                color_b=self._color[2],
                color_a=self._color[3],
            )
            self.uniform.update_buffer()

    def get_bindings(self):
        return self.uniform.get_bindings()

    def _sync_uniform(self):
        if self.uniform is not None:
            self.uniform.n_lines = self._n_lines
            self.uniform.thickness = self._thickness
            self.uniform.show_field = 1 if self._show_field else 0
            self.uniform.color_r = self._color[0]
            self.uniform.color_g = self._color[1]
            self.uniform.color_b = self._color[2]
            self.uniform.color_a = self._color[3]
            self.uniform.update_buffer()

    @property
    def n_lines(self):
        return self._n_lines

    @n_lines.setter
    def n_lines(self, value):
        self._n_lines = int(value)
        self._sync_uniform()

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        self._thickness = float(value)
        self._sync_uniform()

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = tuple(value)
        self._sync_uniform()

    @property
    def show_field(self):
        return self._show_field

    @show_field.setter
    def show_field(self, value):
        self._show_field = bool(value)
        self._sync_uniform()


def _isoline_export_interactions(uniform, buffer_registry, label="Isolines"):
    """Build export GUI interactions for isoline controls."""
    from webgpu.export.gui import gui_interaction, Checkbox, Slider, Write, Target

    if uniform is None:
        return []
    buf = getattr(uniform, "_buffer", None)
    if buf is None:
        return []
    key = id(buf)
    if key not in buffer_registry._buffers:
        return []
    buf_id = buffer_registry._buffers[key][0]

    # IsolineUniform layout: n_lines (u32 @0), thickness (f32 @4), show_field (u32 @8)
    return [gui_interaction(
        label,
        [
            Checkbox(var="enabled", name="Enabled", default=uniform.n_lines > 0),
            Slider(var="count", name="Count", default=float(uniform.n_lines),
                   min=0.0, max=50.0, step=1.0),
            Slider(var="thickness", name="Thickness", default=float(uniform.thickness),
                   min=0.5, max=5.0, step=0.1),
        ],
        [
            Write(targets=[Target(buf_id, offset=0, dtype="u32")],
                  expr="enabled ? Math.round(count) : 0",
                  trigger="enabled"),
            Write(targets=[Target(buf_id, offset=0, dtype="u32")],
                  expr="enabled ? Math.round(count) : 0",
                  trigger="count"),
            Write(targets=[Target(buf_id, offset=4, dtype="f32")],
                  expr="thickness", trigger="thickness"),
        ],
    )]


# ---------------------------------------------------------------------------
# Renderer classes
# ---------------------------------------------------------------------------

from .cf import CFRenderer


class IsolineRenderer(CFRenderer):
    """Renderer that draws only isolines (no filled color).

    Use this to overlay isolines of one function over a different visualization.
    The renderer discards all non-isoline pixels, so the underlying rendering
    shows through.

    Parameters
    ----------
    data : FunctionData
        The function data whose isolines to draw.
    n_lines : int
        Number of evenly-spaced isoline levels (default 10).
    thickness : float
        Line thickness in approximate pixels (default 1.5).
    color : tuple
        RGBA color for the isolines (default black).
    clipping : Clipping, optional
        Clipping plane settings.
    colormap : Colormap, optional
        Colormap that defines the value range for isoline spacing.
        If None, a new one is created and autoscaled to the function range.
    symmetry : Symmetry, optional
        Symmetry settings.

    Examples
    --------
    Draw the field of function A, with isolines of function B overlaid::

        from ngsolve_webgpu import *
        mesh_data = MeshData(mesh)
        func_A = FunctionData(mesh_data, cf_A, order=2)
        func_B = FunctionData(mesh_data, cf_B, order=2)

        r_color = CFRenderer(func_A, colormap=colormap)
        r_iso = IsolineRenderer(func_B, n_lines=15, color=(0, 0, 0, 1))

        scene = Draw([r_color, r_iso])
    """

    fragment_entry_point = "fragmentIsolines"

    def __init__(
        self,
        data,
        n_lines=10,
        thickness=1.5,
        color=(0.0, 0.0, 0.0, 1.0),
        show_field=False,
        clipping=None,
        colormap=None,
        symmetry=None,
    ):
        self._isolines = IsolineSettings(
            n_lines=n_lines, thickness=thickness, color=color, show_field=show_field
        )
        super().__init__(
            data,
            label="IsolineRenderer",
            clipping=clipping,
            colormap=colormap,
            symmetry=symmetry,
        )
        self.gpu_objects.isolines = self._isolines
        # Overlay mode: render on top of other geometry
        if not show_field:
            self.depthBias = -2
            self.depthBiasSlopeScale = -2

    @property
    def isolines(self):
        return self._isolines

    def get_shader_code(self):
        return read_shader_file("ngsolve/isolines_render.wgsl")

    def get_bindings(self):
        return [
            *super().get_bindings(),
            *self.gpu_objects.isolines.get_bindings(),
        ]

    def get_export_interactions(self, options, buffer_registry):
        out = list(super().get_export_interactions(options, buffer_registry))
        out += _isoline_export_interactions(
            self.gpu_objects.isolines.uniform, buffer_registry
        )
        return out

from .clipping import ClippingCF


class ClippingIsolineRenderer(ClippingCF):
    """Clipping-plane renderer that draws only isolines.

    Same as IsolineRenderer but for the clipping plane cross-section in 3D.

    Parameters
    ----------
    data : FunctionData
        The function data whose isolines to draw on the clip plane.
    clipping : Clipping, optional
        Clipping plane settings.
    n_lines : int
        Number of isoline levels (default 10).
    thickness : float
        Line thickness in approximate pixels (default 1.5).
    color : tuple
        RGBA color for the isolines (default black).
    colormap : Colormap, optional
        Colormap defining the value range.
    symmetry : Symmetry, optional
        Symmetry settings.
    """

    fragment_entry_point = "fragmentClippingIsolines"

    def __init__(
        self,
        data,
        clipping=None,
        n_lines=10,
        thickness=1.5,
        color=(0.0, 0.0, 0.0, 1.0),
        show_field=False,
        colormap=None,
        symmetry=None,
    ):
        self._isolines = IsolineSettings(
            n_lines=n_lines, thickness=thickness, color=color, show_field=show_field
        )
        super().__init__(
            data,
            clipping=clipping,
            colormap=colormap,
            symmetry=symmetry,
        )
        self.gpu_objects.isolines = self._isolines

    @property
    def isolines(self):
        return self._isolines

    def get_shader_code(self):
        return read_shader_file("ngsolve/isolines_clipping.wgsl")

    def get_bindings(self, compute=False, count=False):
        bindings = super().get_bindings(compute, count)
        if not compute:
            bindings += self.gpu_objects.isolines.get_bindings()
        return bindings

    def get_export_interactions(self, options, buffer_registry):
        out = list(super().get_export_interactions(options, buffer_registry))
        out += _isoline_export_interactions(
            self.gpu_objects.isolines.uniform, buffer_registry
        )
        return out

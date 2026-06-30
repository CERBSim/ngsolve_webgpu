import ctypes as ct

from webgpu import (
    BufferBinding,
    Clipping,
    Colormap,
    read_shader_file,
)
from webgpu.renderer import RenderOptions
from webgpu.uniforms import UniformBase

from .cf import CFRenderer
from .clipping import ClippingCF

ISOSURFACE_BINDING = 27


class IsosurfaceUniform(UniformBase):
    """Mirror of the ``IsosurfaceUniforms`` struct in
    ``shaders/isosurface/common.wgsl`` (binding 27).

    - ``subdivision``: number of compute-shader sub-tet refinement levels
      (only used by the isosurface compute shader).
    - ``value``: the levelset value defining the isosurface / clipping cutoff.
    """

    _binding = ISOSURFACE_BINDING
    _fields_ = [
        ("subdivision", ct.c_uint32),
        ("value", ct.c_float),
        ("padding", ct.c_float),
        ("padding1", ct.c_float),
    ]


class IsosurfaceSettings:
    """Shared isosurface state: a single ``u_iso`` uniform (binding 27).

    Pass one instance to several renderers (isosurface + negative surface +
    negative clipping) so they share a single ``value`` and produce a single
    GUI control. The ``subdivision`` field is only meaningful for the
    isosurface compute shader, so only the IsoSurfaceRenderer writes it.

    The first renderer that registers via :meth:`claim_gui_owner` becomes the
    sole emitter of the value slider, so a shared instance yields one control.
    """

    def __init__(self, value: float = 0.0, label: str = "Isosurface"):
        self._value = float(value)
        self._subdivision = 0
        self.label = label
        self.uniform = None
        self.gui_owner = None

    def _ensure_uniform(self):
        if self.uniform is None:
            self.uniform = IsosurfaceUniform(
                subdivision=self._subdivision, value=self._value
            )
        return self.uniform

    def update(self):
        """Make sure the GPU buffer exists and is in sync."""
        self._ensure_uniform().update_buffer()

    def set_subdivision(self, subdivision):
        self._subdivision = int(subdivision)
        self._ensure_uniform().subdivision = self._subdivision

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = float(v)
        if self.uniform is not None:
            self.uniform.value = self._value
            self.uniform.update_buffer()

    def get_bindings(self):
        return self._ensure_uniform().get_bindings()

    def claim_gui_owner(self, renderer):
        """The first registering renderer owns the (single) GUI control."""
        if self.gui_owner is None:
            self.gui_owner = renderer


def _levelset_value_range(levelset):
    """Slider bounds for the isosurface value, from the levelset data range.

    ``minval``/``maxval`` index 0 is the magnitude; index 1 (component 0) holds
    the signed scalar range we want for a levelset slider.
    """
    try:
        if len(levelset.minval) > 1:
            lo, hi = float(levelset.minval[1]), float(levelset.maxval[1])
        else:
            lo, hi = float(levelset.minval[0]), float(levelset.maxval[0])
    except Exception:
        lo, hi = -1.0, 1.0
    if not (hi > lo):
        lo, hi = -1.0, 1.0
    return lo, hi


def _isosurface_value_export_interactions(settings, levelset, buffer_registry):
    """Build an export GUI slider that writes the isosurface ``value`` field."""
    from webgpu.export.gui import gui_interaction, Slider, Write, Target

    uniform = settings.uniform
    if uniform is None:
        return []
    buf = getattr(uniform, "_buffer", None)
    if buf is None:
        return []
    # Register if not already reached via render/compute bindings, so the slider
    # works regardless of the order in which the exporter gathers things.
    buf_id = buffer_registry.register_buffer(buf, "uniform")

    lo, hi = _levelset_value_range(levelset)
    # IsosurfaceUniform layout: subdivision (u32 @0), value (f32 @4)
    return [gui_interaction(
        settings.label,
        [Slider(var="value", name="Level", default=float(uniform.value),
                min=lo, max=hi, step=(hi - lo) / 100.0)],
        [Write(targets=[Target(buf_id, offset=4, dtype="f32")],
               expr="value", trigger="value")],
    )]


class IsoSurfaceRenderer(ClippingCF):
    compute_shader = "ngsolve/isosurface/compute.wgsl"
    vertex_entry_point = "vertex_isosurface"
    fragment_entry_point = "fragment_isosurface"

    def __init__(
        self,
        func_data,
        levelset_data,
        clipping: Clipping | None = None,
        colormap: Colormap | None = None,
        symmetry=None,
        value: float = 0.0,
        settings: IsosurfaceSettings | None = None,
    ):
        super().__init__(func_data, clipping, colormap, symmetry=symmetry)
        self.levelset = levelset_data
        self.levelset.need_3d = True
        self.subdivision = 0
        self._iso_subdivision = None  # None = auto-computed from levelset order
        self.iso_settings = settings or IsosurfaceSettings(value=value)
        self.iso_settings.claim_gui_owner(self)

    def get_shader_code(self):
        return read_shader_file("ngsolve/isosurface/render.wgsl")

    def _get_iso_subdivision(self):
        """Compute subdivision level for the isosurface compute shader.

        Uses all-edge-midpoint subdivision (8 sub-tets per level).
        Each level halves all edges, so after k levels each edge
        has 2^k segments.
        """
        if self._iso_subdivision is not None:
            return self._iso_subdivision
        order = self.levelset.order_3d
        if order > 3:
            k = 0
            while (1 << k) < order:
                k += 1
            return k
        elif order > 1:
            return order - 1
        else:
            return 0

    def update(self, options: RenderOptions):
        self.iso_settings.set_subdivision(self._get_iso_subdivision())
        self.iso_settings.update()
        self.levelset.update(options)
        self.levelset_buffer = self.levelset.get_buffers()["data_3d"]
        super().update(options)

        # No render-side subdivision for isosurfaces: the compute shader finds
        # correct zero-crossing positions on sub-tet edges. Render-side subdivision
        # would interpolate between those vertices, going off the surface.
        self.shader_defines["CLIPPING_SUBDIVISION"] = 1
        self.n_vertices = 3

    def get_bindings(self, compute=False):
        bindings = super().get_bindings(compute)
        if compute:
            bindings += self.iso_settings.get_bindings()
        bindings += [
            BufferBinding(26, self.levelset_buffer),
        ]
        return bindings

    def get_export_interactions(self, options, buffer_registry):
        out = list(super().get_export_interactions(options, buffer_registry))
        if self.iso_settings.gui_owner is self:
            out += _isosurface_value_export_interactions(
                self.iso_settings, self.levelset, buffer_registry
            )
        return out

    def get_export_compute_passes(self, options, buffer_registry):
        passes = super().get_export_compute_passes(options, buffer_registry)
        # Re-extract the isosurface when the level value (u_iso) changes, the
        # same way the clipping-plane buffer already re-triggers the compute.
        iso_buf = self.iso_settings.uniform._buffer
        try:
            iso_id = buffer_registry.get_id(iso_buf)
        except KeyError:
            iso_id = buffer_registry.register_buffer(iso_buf, "uniform")
        for p in passes:
            if iso_id not in p.triggers:
                p.triggers.append(iso_id)
        return passes


class NegativeSurfaceRenderer(CFRenderer):
    def __init__(
        self, functiondata, levelsetdata, clipping: Clipping = None, colormap: Colormap = None,
        symmetry=None, value: float = 0.0, settings: IsosurfaceSettings | None = None,
    ):
        super().__init__(
            functiondata, label="NegativeSurfaceRenderer", clipping=clipping, colormap=colormap, symmetry=symmetry
        )
        self.fragment_entry_point = "fragmentCheckLevelset"
        self.levelset = levelsetdata
        self.iso_settings = settings or IsosurfaceSettings(value=value, label="Negative Surface")
        self.iso_settings.claim_gui_owner(self)

    def update(self, options: RenderOptions):
        self.iso_settings.update()
        self.levelset.update(options)
        buffers = self.levelset.get_buffers()
        self.levelset_buffer = buffers["data_2d"]
        super().update(options)

    def get_bindings(self):
        return super().get_bindings() + [
            BufferBinding(80, self.levelset_buffer),
            *self.iso_settings.get_bindings(),
        ]

    def get_shader_code(self):
        return read_shader_file("ngsolve/isosurface/negative_surface.wgsl")

    def get_export_interactions(self, options, buffer_registry):
        out = list(super().get_export_interactions(options, buffer_registry))
        if self.iso_settings.gui_owner is self:
            out += _isosurface_value_export_interactions(
                self.iso_settings, self.levelset, buffer_registry
            )
        return out


class NegativeClippingRenderer(ClippingCF):
    fragment_entry_point = "fragment_neg_clip"

    def __init__(self, data, levelsetdata, clipping: Clipping = None, colormap: Colormap = None,
                 symmetry=None, value: float = 0.0, settings: IsosurfaceSettings | None = None):
        super().__init__(data, clipping, colormap, symmetry=symmetry)
        self.levelset = levelsetdata
        self.levelset.need_3d = True
        self.iso_settings = settings or IsosurfaceSettings(value=value, label="Negative Clipping")
        self.iso_settings.claim_gui_owner(self)

    def update(self, options):
        self.iso_settings.update()
        self.levelset.update(options)
        buffers = self.levelset.get_buffers()
        self.levelset_buffer = buffers["data_3d"]
        super().update(options)

        # Override subdivision based on function order for smooth rendering
        # on the clipping plane (positions stay correct since the plane is flat).
        order = max(self.data.order_3d, self.levelset.order_3d)
        if order > 3:
            func_subdiv = (order + 2) // 3 + 1
        elif order > 1:
            func_subdiv = 3
        else:
            func_subdiv = 1
        clip_subdiv = self.shader_defines.get("CLIPPING_SUBDIVISION", 1)
        subdiv = max(func_subdiv, clip_subdiv)
        self.shader_defines["CLIPPING_SUBDIVISION"] = subdiv
        self.n_vertices = 3 * subdiv * subdiv

    def get_bindings(self, compute=False):
        bindings = super().get_bindings(compute)
        if not compute:
            bindings += [
                BufferBinding(80, self.levelset_buffer),
                *self.iso_settings.get_bindings(),
            ]
        return bindings

    def get_shader_code(self):
        return read_shader_file("ngsolve/isosurface/negative_clipping.wgsl")

    def get_export_interactions(self, options, buffer_registry):
        out = list(super().get_export_interactions(options, buffer_registry))
        if self.iso_settings.gui_owner is self:
            out += _isosurface_value_export_interactions(
                self.iso_settings, self.levelset, buffer_registry
            )
        return out


def make_isosurface_renderers(
    func_data,
    levelset_data,
    clipping: Clipping | None = None,
    colormap: Colormap | None = None,
    symmetry=None,
    value: float = 0.0,
    label: str = "Isosurface",
    isosurface: bool = True,
    negative_surface: bool = True,
    negative_clipping: bool = True,
):
    """Build the isosurface renderer trio wired to one shared IsosurfaceSettings.

    Using this helper guarantees all renderers share a single ``value`` uniform
    and expose a single GUI control, instead of one per renderer.

    Parameters
    ----------
    func_data, levelset_data, clipping, colormap, symmetry
        Passed through to the individual renderers.
    value : float
        Initial isosurface level.
    label : str
        Label of the shared GUI control.
    isosurface, negative_surface, negative_clipping : bool
        Toggle which of the three renderers to create.

    Returns
    -------
    (renderers, settings) : (list, IsosurfaceSettings)
        ``renderers`` is the list of created renderers in draw order; the first
        one owns the single GUI control. ``settings`` is the shared state; set
        ``settings.value = ...`` to drive all of them at once.
    """
    settings = IsosurfaceSettings(value=value, label=label)
    renderers = []
    if isosurface:
        renderers.append(IsoSurfaceRenderer(
            func_data, levelset_data, clipping, colormap, symmetry=symmetry, settings=settings
        ))
    if negative_surface:
        renderers.append(NegativeSurfaceRenderer(
            func_data, levelset_data, clipping=clipping, colormap=colormap,
            symmetry=symmetry, settings=settings,
        ))
    if negative_clipping:
        renderers.append(NegativeClippingRenderer(
            func_data, levelset_data, clipping, colormap, symmetry=symmetry, settings=settings
        ))
    return renderers, settings

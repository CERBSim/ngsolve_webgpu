"""Line Integral Convolution (LIC) of a vector coefficient function on a 3D
clipping plane.

The renderer reuses ``ClippingCF`` for everything that draws the cut plane
(curvature-aware clip geometry, deformation, symmetry, colormap, component
selection) and adds three compute passes that produce a streamline ("LIC")
texture which the clipping fragment shader modulates the colour with:

  1. ``clear_field``        (lic/evaluate.wgsl) - reset the field texture.
  2. ``evaluate_lic_field`` (lic/evaluate.wgsl) - iterate the volume elements,
     clip each against the plane with the *same* ``clipTet`` ``ClippingCF`` uses
     and software-rasterise every cut triangle into a field texture storing the
     in-plane flow direction, the scalar colour value and a coverage mask.
  3. ``compute_lic`` (lic/line_integral_convolution.wgsl) - march along the
     normalised flow streaming white noise into a grayscale LIC image.

The render pass (clipping/render.wgsl, ``#ifdef LIC`` branch) samples that image
in plane-parameter space and modulates the colormapped value by it.

See ``LIC_IMPLEMENTATION_PLAN.md`` for the design rationale (in particular why
this is a compute-shader rasterisation rather than an offscreen render pass, and
why the static-HTML export path is intentionally degraded for v1).
"""

import ctypes as ct
import os

import numpy as np

from webgpu.renderer import Renderer, RenderOptions
from webgpu.uniforms import UniformBase
from webgpu.utils import (
    StorageTextureBinding,
    TextureBinding,
    read_shader_file,
    run_compute_shader,
)
from webgpu.webgpu_api import ShaderStage, TextureFormat, TextureUsage

from .clipping import ClippingCF
from .cf import FunctionData
from webgpu.clipping import Clipping
from webgpu.colormap import Colormap


class LicUniforms(UniformBase):
    """Plane parameterisation + LIC parameters, shared by all three stages and
    the render pass at ``@group(0) @binding(40)``.

    Byte layout MUST match ``shaders/lic/common.wgsl`` (struct ``LicUniforms``,
    80 bytes = 5 * 16). The pre-existing ``webgpu.uniforms``
    ``LineIntegralConvolutionUniforms`` (also binding 40) lacks the plane-basis
    fields, so it is deliberately *not* reused.
    """

    _binding = 40
    _fields_ = [
        ("origin", ct.c_float * 4),     # xyz: plane origin (bbox centre)
        ("tangent1", ct.c_float * 4),   # xyz: first  in-plane unit vector (u)
        ("tangent2", ct.c_float * 4),   # xyz: second in-plane unit vector (v)
        ("width", ct.c_uint32),
        ("height", ct.c_uint32),
        ("kernel_length", ct.c_uint32),
        ("oriented", ct.c_uint32),
        ("thickness", ct.c_uint32),
        ("inv_extent", ct.c_float),     # world -> [0,1] scale (= 1 / (2 R))
        ("contrast", ct.c_float),       # LIC modulation strength in [0, 1]
        ("pad", ct.c_uint32),
    ]

    def __init__(
        self,
        *,
        width=1024,
        height=1024,
        kernel_length=30,
        oriented=0,
        thickness=10,
        contrast=1.0,
        **kwargs,
    ):
        super().__init__(
            width=width,
            height=height,
            kernel_length=kernel_length,
            oriented=oriented,
            thickness=thickness,
            contrast=contrast,
            **kwargs,
        )


class ClippingLIC(ClippingCF):
    """Clipping-plane renderer that overlays a Line Integral Convolution of the
    (vector) coefficient function onto the cut plane."""

    def __init__(
        self,
        data: FunctionData,
        clipping: Clipping = None,
        colormap: Colormap = None,
        component=None,
        symmetry=None,
        *,
        kernel_length: int = 30,
        oriented: bool = False,
        thickness: int = 10,
        contrast: float = 1.0,
        resolution: int = 256,
    ):
        super().__init__(
            data, clipping=clipping, colormap=colormap, component=component, symmetry=symmetry
        )
        self.kernel_length = int(kernel_length)
        self.oriented = bool(oriented)
        self.thickness = int(thickness)
        self.contrast = float(contrast)
        self.resolution = int(resolution)

        self._lic_uniforms = LicUniforms(
            width=self.resolution,
            height=self.resolution,
            kernel_length=self.kernel_length,
            oriented=int(self.oriented),
            thickness=self.thickness,
            contrast=self.contrast,
        )
        self._field_tex = None   # rgba32float: (dir_u, dir_v, value, mask)
        self._lic_tex = None     # rgba8unorm: (lic_value, mask, 0, 0)
        self._lic_enabled = False

        # Drag-refresh: once captured, the JS engine owns the clip compute and
        # re-renders a plane move GPU-side WITHOUT calling Python update(), so the
        # Python-dispatched LIC textures would go stale on a drag. Re-mark dirty
        # on a plane change so update() re-runs and re-dispatches the LIC passes.
        # We intentionally use the lighter Renderer.set_needs_update (via
        # _refresh) rather than ClippingCF.set_needs_update: a plane move does not
        # change the field values, only the cut, so a full volume-CF
        # re-evaluation per drag tick is unnecessary.
        self._clipping.callbacks.append(self._refresh)

    # ------------------------------------------------------------------ update
    def update(self, options: RenderOptions):
        super().update(options)
        self._lic_enabled = False
        if not self.active:
            return
        if os.environ.get("WEBGPU_EXPORTING"):
            # Static-HTML export serialises texture *pixel data*, so it can't
            # carry the GPU-computed LIC texture; show the plain clipping plane
            # there (graceful degrade). The live JS engine, by contrast, binds
            # the LIC output texture by handle, so LIC works there too — the LIC
            # compute still runs from Python in this update() before the engine
            # renders. (lic_output is rgba8unorm so its filterable "float" sample
            # type matches the engine's auto bind-group layout.)
            return

        self._lic_enabled = True
        # Compile the LIC branch of clipping/render.wgsl. shader_defines is reset
        # from the mesh defines inside super().update(), so set this afterwards.
        self.shader_defines["LIC"] = 1

        # FunctionSettings (binding 55, u_function_component) is a gpu_objects
        # child whose uniform buffer is created lazily in its own update(). The
        # render loop runs that child update AFTER this parent update() (see
        # renderer.py _update_and_create_render_pipeline: self.update() first,
        # then the gpu_objects loop), but _dispatch_lic() binds 55 right now, so
        # initialise it here. update() is idempotent (creates the buffer once).
        self.gpu_objects.settings.update(options)

        self._update_lic_uniforms()
        self._ensure_lic_textures()
        self._dispatch_lic()

    def _update_lic_uniforms(self):
        """Recompute the in-plane orthonormal basis and fill the LIC uniform."""
        n = np.asarray(self._clipping.normal, dtype=np.float64)
        nrm = np.linalg.norm(n)
        n = n / nrm if nrm > 1e-12 else np.array([0.0, 0.0, 1.0])
        # An in-plane basis: cross the normal with whichever axis it is least
        # aligned with, then complete the right-handed frame.
        helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(helper, n)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)  # already unit: n, t1 orthonormal

        (pmin, pmax) = self.data.get_bounding_box()
        pmin = np.asarray(pmin, dtype=np.float64)
        pmax = np.asarray(pmax, dtype=np.float64)
        centre = 0.5 * (pmin + pmax)
        radius = 0.5 * float(np.linalg.norm(pmax - pmin))
        if radius <= 1e-12:
            radius = 1.0

        u = self._lic_uniforms
        u.origin[:] = [float(centre[0]), float(centre[1]), float(centre[2]), 0.0]
        u.tangent1[:] = [float(t1[0]), float(t1[1]), float(t1[2]), 0.0]
        u.tangent2[:] = [float(t2[0]), float(t2[1]), float(t2[2]), 0.0]
        u.width = int(self.resolution)
        u.height = int(self.resolution)
        u.kernel_length = int(self.kernel_length)
        u.oriented = 1 if self.oriented else 0
        u.thickness = int(self.thickness)
        u.inv_extent = float(1.0 / (2.0 * radius))
        u.contrast = float(self.contrast)
        u.update_buffer()

    def _ensure_lic_textures(self):
        """(Re)allocate the field and LIC textures at the current resolution.

        Both are written by compute (storage) and sampled afterwards (texture),
        so both usages are required. COPY_SRC is added when exporting to mirror
        colormap.py, even though the export path is degraded for v1."""
        res = int(self.resolution)
        extra = TextureUsage.COPY_SRC if os.environ.get("WEBGPU_EXPORTING") else 0
        usage = TextureUsage.STORAGE_BINDING | TextureUsage.TEXTURE_BINDING | extra
        if self._field_tex is None or self._field_tex.width != res or self._field_tex.height != res:
            self._field_tex = self.device.createTexture(
                size=[res, res, 1],
                usage=usage,
                format=TextureFormat.rgba32float,
                dimension="2d",
                label="lic_field",
            )
        if self._lic_tex is None or self._lic_tex.width != res or self._lic_tex.height != res:
            # rgba8unorm (not rg32float): the value+mask are both in [0,1], and a
            # filterable format means its "float" sample type matches the bind-
            # group layout the JS engine infers for the render pass (an
            # unfilterable rg32float there is rejected -> device lost).
            self._lic_tex = self.device.createTexture(
                size=[res, res, 1],
                usage=usage,
                format=TextureFormat.rgba8unorm,
                dimension="2d",
                label="lic_output",
            )

    def _lic_compute_defines(self):
        d = {
            "MAX_EVAL_ORDER": self.shader_defines.get("MAX_EVAL_ORDER", 1),
            "MAX_EVAL_ORDER_VEC3": self.shader_defines.get("MAX_EVAL_ORDER_VEC3", 1),
            "CLIPPING_SUBDIVISION": 1,
        }
        if self.data.cf.is_complex:
            d["IS_COMPLEX"] = 1
        return d

    def _dispatch_lic(self):
        """Run the three LIC compute passes. Each run_compute_shader call submits
        its own command encoder, so they execute sequentially on the queue and
        stage N reliably sees stage N-1's output (no manual barriers needed)."""
        res = int(self.resolution)
        gx = (res + 15) // 16
        gy = (res + 15) // 16
        defines = self._lic_compute_defines()
        eval_code = read_shader_file("ngsolve/lic/evaluate.wgsl")
        lic_code = read_shader_file("ngsolve/lic/line_integral_convolution.wgsl")

        # Stage 1a: clear the field texture (only touches binding 40 + 41).
        run_compute_shader(
            eval_code,
            self._lic_uniforms.get_bindings()
            + [StorageTextureBinding(41, self._field_tex, access="write-only", dim=2)],
            [gx, gy, 1],
            entry_point="clear_field",
            defines=defines,
            label="lic_clear_field",
        )

        # Stage 1b: rasterise the clipped vector field into the field texture.
        #
        # run_compute_shader builds an *explicit* WebGPU pipeline layout from
        # exactly the bindings passed, so we hand it a MINIMAL set — precisely the
        # bindings evaluate_lic_field statically references — rather than a
        # superset, which (per the validation notes in the implementation plan)
        # can be rejected as "binding not used".
        #
        # evaluate_lic_field shares clipping/compute.wgsl's clipTet/getTetPoints
        # closure, so it touches the same geometry bindings — mesh(110),
        # deformation-scale(17), deformation-3d(18), clipping(1) — plus complex(56)
        # (reached statically via getTetPoints -> evalTetComplex) and, unlike the
        # clip-fill shader, function-values-3d(13) and component(55) for evalTet on
        # the function itself. It does NOT use deformation-2d(16), the tet counter
        # (21), the cut-trig output (24) or u_ntets(22), so those are excluded.
        EVAL_REFERENCED = {110, 17, 18, 13, 1, 56}
        eval_bindings = [b for b in self.get_bindings(compute=True) if b.nr in EVAL_REFERENCED]
        eval_bindings += self.gpu_objects.settings.get_bindings()       # 55 u_function_component
        eval_bindings += self._lic_uniforms.get_bindings()              # 40 u_lic
        eval_bindings += [StorageTextureBinding(41, self._field_tex, access="write-only", dim=2)]

        n_tets = int(self.data.mesh_data.num_elements.get("tets", 0))
        n_wg = min(n_tets // 256 + 1, 1024)
        run_compute_shader(
            eval_code,
            eval_bindings,
            [n_wg, 1, 1],
            entry_point="evaluate_lic_field",
            defines=defines,
            label="lic_evaluate_field",
        )

        # Stage 2: line integral convolution. Reads the field texture as a sampled
        # (unfilterable-float) texture, writes the grayscale LIC image (storage).
        run_compute_shader(
            lic_code,
            self._lic_uniforms.get_bindings()
            + [
                TextureBinding(
                    41,
                    self._field_tex,
                    sample_type="unfilterable-float",
                    dim=2,
                    visibility=ShaderStage.COMPUTE,
                ),
                StorageTextureBinding(42, self._lic_tex, access="write-only", dim=2),
            ],
            [gx, gy, 1],
            entry_point="compute_lic",
            defines=defines,
            label="lic_convolve",
        )

    # ---------------------------------------------------------------- bindings
    def get_bindings(self, compute=False):
        bindings = super().get_bindings(compute=compute)
        if (not compute) and self._lic_enabled and self._lic_tex is not None:
            # Render pass: add the LIC uniform (40) and the LIC output texture
            # (42) the #ifdef LIC branch of clipping/render.wgsl samples. The
            # field texture (41) is NOT added here: render never reads it.
            bindings = bindings + self._lic_uniforms.get_bindings() + [
                TextureBinding(
                    42,
                    self._lic_tex,
                    sample_type="float",  # rgba8unorm is filterable
                    dim=2,
                    visibility=ShaderStage.FRAGMENT,
                ),
            ]
        return bindings

    # ----------------------------------------------------------------- helpers
    def _refresh(self):
        """Mark dirty so the next render re-runs update() (and re-dispatches the
        LIC passes) WITHOUT invalidating the volume CF data. Skips
        ClippingCF.set_needs_update's data.set_needs_update() to avoid a costly
        per-drag CF re-evaluation: neither a plane move nor a LIC-parameter change
        affects the field values, only how they are convolved/rendered."""
        super(ClippingCF, self).set_needs_update()

    def set_kernel_length(self, value: int):
        self.kernel_length = int(value)
        self._refresh()

    def set_oriented(self, value: bool):
        self.oriented = bool(value)
        self._refresh()

    def set_thickness(self, value: int):
        self.thickness = int(value)
        self._refresh()

    def set_contrast(self, value: float):
        self.contrast = float(value)
        self._refresh()

    def set_resolution(self, value: int):
        self.resolution = int(value)
        self._refresh()


# Backwards-compatible / descriptive alias.
LineIntegralConvolution = ClippingLIC

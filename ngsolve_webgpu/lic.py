"""Line Integral Convolution (LIC) of a vector coefficient function on a 3D
clipping plane, computed in SCREEN space.

The renderer reuses ``ClippingCF`` for everything that draws the cut plane
(curvature-aware clip geometry, deformation, symmetry, colormap, component
selection) and adds three compute passes that produce a streamline ("LIC")
texture which the clipping fragment shader modulates the colour with. Unlike the
original plane-parameter (orthographic) version, the LIC is built in the rendered
image itself under the live perspective camera, so the textures are the size of
the canvas (times an optional ``supersample`` SSAA factor) and are recomputed
whenever the camera (or clip plane) moves:

  1. ``clear_field``        (lic/evaluate.wgsl) - reset the field texture.
  2. ``evaluate_lic_field`` (lic/evaluate.wgsl) - iterate the volume elements,
     clip each against the plane with the *same* ``clipTet`` ``ClippingCF`` uses
     and software-rasterise every cut triangle into a *screen-space* field texture
     storing the camera-projected flow direction, the scalar colour value and a
     coverage mask. Points where the field has no value (NaN) drop out of the
     coverage mask so the streamline kernel stops there.
  3. ``compute_lic`` (lic/line_integral_convolution.wgsl) - march along the
     normalised screen-space flow streaming white noise into a grayscale LIC
     image (also canvas-sized).

The render pass (clipping/render.wgsl, ``#ifdef LIC`` branch) samples that image
at each fragment's framebuffer pixel and modulates the colormapped value by it.

Dispatch model: the three passes are dispatched from Python in ``update()`` (they
read the camera uniform at binding 0), so the LIC tracks the camera in the live
Python render path (and the ``WEBGPU_TESTING`` harness) - recomputing every frame
makes rotation laggy by design, in exchange for crisp, perspective-correct
streamlines at any zoom. NOTE: the live JS export engine's compute DAG is
buffer-only (it cannot bind storage textures), so it does not re-run these passes
GPU-side; there the plane simply shows the last Python-dispatched LIC. Static-HTML
export stays degraded (no LIC), as before.

See ``LIC_IMPLEMENTATION_PLAN.md`` for the original design rationale.
"""

import ctypes as ct
import os

import numpy as np

from webgpu.renderer import RenderOptions
from webgpu.scene import debounce
from webgpu.uniforms import UniformBase
from webgpu.utils import (
    StorageTextureBinding,
    TextureBinding,
    BufferBinding,
    read_shader_file,
    run_compute_shader,
)
from webgpu.webgpu_api import ShaderStage, TextureFormat, TextureUsage

from .clipping import ClippingCF
from .cf import FunctionData, CFRenderer, Binding
from webgpu.clipping import Clipping
from webgpu.colormap import Colormap


class LicUniforms(UniformBase):
    """Plane basis + LIC parameters, shared by all three stages and the render
    pass at ``@group(0) @binding(40)``.

    Byte layout MUST match ``shaders/lic/common.wgsl`` (struct ``LicUniforms``,
    144 bytes = 5 * 16 scalars + a 64-byte ``lic_mvp`` mat4x4 at offset 80). The
    ``mat4x4`` is 16-byte aligned and offset 80 is already a multiple of 16, so
    no padding is needed before it. The pre-existing ``webgpu.uniforms``
    ``LineIntegralConvolutionUniforms`` (also binding 40) lacks the plane-basis
    fields, so it is deliberately *not* reused.

    ``width``/``height`` are the LIC texture size, i.e. ``supersample`` times the
    canvas (framebuffer) device-pixel size, since the LIC is computed in screen
    space and supersampled; ``tangent1``/``tangent2`` give the in-plane flow basis
    and ``inv_extent`` sets the world step for the camera Jacobian. ``origin`` is
    retained for the basis but no longer indexes a texture.
    """

    _binding = 40
    _fields_ = [
        ("origin", ct.c_float * 4),     # xyz: plane origin (bbox centre)
        ("tangent1", ct.c_float * 4),   # xyz: first  in-plane unit vector (u)
        ("tangent2", ct.c_float * 4),   # xyz: second in-plane unit vector (v)
        ("width", ct.c_uint32),         # LIC texture width  = supersample * canvas px
        ("height", ct.c_uint32),        # LIC texture height = supersample * canvas px
        ("kernel_length", ct.c_uint32),
        ("oriented", ct.c_uint32),
        ("thickness", ct.c_uint32),
        ("inv_extent", ct.c_float),     # world -> [0,1] scale (= 1 / (2 R))
        ("contrast", ct.c_float),       # LIC modulation strength in [0, 1]
        ("supersample", ct.c_uint32),   # LIC texels per canvas pixel (SSAA factor)
        ("lic_mvp", ct.c_float * 16),   # camera MVP the LIC field was built with
                                        # (staleness check in the render pass)
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
        supersample=1,
        **kwargs,
    ):
        super().__init__(
            width=width,
            height=height,
            kernel_length=kernel_length,
            oriented=oriented,
            thickness=thickness,
            contrast=contrast,
            supersample=supersample,
            **kwargs,
        )


def _store_lic_mvp(lic_uniforms: "LicUniforms", options: RenderOptions):
    """Record the camera MVP the LIC field is about to be built with into the LIC
    uniform, so the render pass can detect a stale LIC (camera moved since this
    dispatch) and skip modulation until Python re-dispatches.

    ``model_view_projection`` holds the exact 16 floats bound at camera binding 0,
    so the stored layout matches ``u_camera.model_view_projection`` in the shader
    byte-for-byte. Must be called AFTER ``options.update_buffers()`` so the camera
    uniform reflects the current view. No-op before the camera uniform exists."""
    cam_u = getattr(options, "_camera_uniforms", None)
    if cam_u is None:
        return
    lic_uniforms.lic_mvp[:] = cam_u.model_view_projection
    lic_uniforms.update_buffer()


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
        supersample: int = 2,
    ):
        super().__init__(
            data, clipping=clipping, colormap=colormap, component=component, symmetry=symmetry
        )
        self.kernel_length = int(kernel_length)
        self.oriented = bool(oriented)
        self.thickness = int(thickness)
        self.contrast = float(contrast)
        # Supersampling factor (SSAA): the LIC is computed at supersample x the
        # canvas resolution and box-downsampled in the render pass, which
        # antialiases thin streaks and gives smooth, effectively fractional line
        # widths (notably for the oriented/droplet mode). kernel_length and
        # thickness keep their canvas-pixel meaning (scaled internally). Cost
        # scales ~supersample**2; 1 disables it.
        self.supersample = max(1, int(supersample))
        # ``resolution`` is kept for backwards compatibility but no longer sets the
        # LIC texture size: in the screen-space path the textures follow the canvas
        # (see _ensure_lic_textures). It only seeds the uniform before the first
        # update(), where width/height are overwritten with the real canvas size.
        self.resolution = int(resolution)

        self._lic_uniforms = LicUniforms(
            width=self.resolution,
            height=self.resolution,
            kernel_length=self.kernel_length,
            oriented=int(self.oriented),
            thickness=self.thickness,
            contrast=self.contrast,
            supersample=self.supersample,
        )
        self._field_tex = None   # rgba32float: (screen_dir_x, screen_dir_y, value, mask)
        self._lic_tex = None     # rgba8unorm: (lic_value, mask, 0, 0)
        self._lic_enabled = False
        self._cam_observer_registered = False

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

        self._register_camera_observer(options)
        self._update_lic_uniforms(options)
        self._ensure_lic_textures(options)
        self._dispatch_lic(options)

    def _register_camera_observer(self, options: RenderOptions):
        """Mark dirty whenever the camera moves, so the screen-space LIC is
        re-dispatched for the new view. In the Python render path this drives the
        (intentionally laggy) per-rotation recompute; it is a no-op where the host
        does not observe the camera or does not re-render from Python."""
        if self._cam_observer_registered:
            return
        cam = getattr(options, "camera", None)
        register = getattr(cam, "register_observer", None)
        if register is not None:
            register(self._on_camera_moved)
            self._cam_observer_registered = True

    def _update_lic_uniforms(self, options: RenderOptions):
        """Recompute the in-plane orthonormal basis and fill the LIC uniform.

        ``width``/``height`` are set to the canvas size: the field/LIC textures and
        the screen-space projection in the shaders are all in framebuffer pixels."""
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

        f = max(1, int(self.supersample))
        u = self._lic_uniforms
        u.origin[:] = [float(centre[0]), float(centre[1]), float(centre[2]), 0.0]
        u.tangent1[:] = [float(t1[0]), float(t1[1]), float(t1[2]), 0.0]
        u.tangent2[:] = [float(t2[0]), float(t2[1]), float(t2[2]), 0.0]
        # The LIC is computed at f texels per canvas pixel; kernel_length and
        # thickness are given in canvas pixels, so scale them into LIC texels so
        # the apparent streak length/width is independent of the SSAA factor.
        u.width = int(options.canvas.width) * f
        u.height = int(options.canvas.height) * f
        u.kernel_length = max(1, int(self.kernel_length) * f)
        u.oriented = 1 if self.oriented else 0
        u.thickness = max(1, int(self.thickness) * f)
        u.inv_extent = float(1.0 / (2.0 * radius))
        u.contrast = float(self.contrast)
        u.supersample = f
        u.update_buffer()

    def _ensure_lic_textures(self, options: RenderOptions):
        """(Re)allocate the field and LIC textures at the canvas resolution.

        Screen-space LIC means both textures are framebuffer-sized and are
        reallocated when the canvas is resized. Both are written by compute
        (storage) and sampled afterwards (texture), so both usages are required.
        COPY_SRC is added when exporting to mirror colormap.py, even though the
        export path is degraded for v1."""
        f = max(1, int(self.supersample))
        w = int(options.canvas.width) * f
        h = int(options.canvas.height) * f
        extra = TextureUsage.COPY_SRC if os.environ.get("WEBGPU_EXPORTING") else 0
        usage = TextureUsage.STORAGE_BINDING | TextureUsage.TEXTURE_BINDING | extra
        if self._field_tex is None or self._field_tex.width != w or self._field_tex.height != h:
            self._field_tex = self.device.createTexture(
                size=[w, h, 1],
                usage=usage,
                format=TextureFormat.rgba32float,
                dimension="2d",
                label="lic_field",
            )
        if self._lic_tex is None or self._lic_tex.width != w or self._lic_tex.height != h:
            # rgba8unorm (not rg32float): the value+mask are both in [0,1], and a
            # filterable format means its "float" sample type matches the bind-
            # group layout the JS engine infers for the render pass (an
            # unfilterable rg32float there is rejected -> device lost).
            self._lic_tex = self.device.createTexture(
                size=[w, h, 1],
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
        if self.region_visibility is not None:
            d["REGION_VISIBILITY"] = 1
        return d

    def _dispatch_lic(self, options: RenderOptions):
        """Run the three LIC compute passes. Each run_compute_shader call submits
        its own command encoder, so they execute sequentially on the queue and
        stage N reliably sees stage N-1's output (no manual barriers needed)."""
        # The evaluate pass reads the camera uniform (binding 0) to project the
        # flow into screen space, so make sure that buffer reflects the current
        # camera before we dispatch. The legacy render path renders objects
        # without re-writing it each frame; this keeps the streamlines aligned
        # with the cut plane the render pass draws (which reads the same buffer).
        # In JS-engine mode skip_camera_buffer_write is set, so this is a no-op
        # for the camera buffer (the engine owns it).
        options.update_buffers()
        _store_lic_mvp(self._lic_uniforms, options)

        f = max(1, int(self.supersample))
        w = int(options.canvas.width) * f
        h = int(options.canvas.height) * f
        gx = (w + 15) // 16
        gy = (h + 15) // 16
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
        # the function itself. The screen-space version additionally references the
        # camera uniform (0) for cameraMapPoint. It does NOT use deformation-2d(16),
        # the tet counter (21), the cut-trig output (24) or u_ntets(22), so those
        # are excluded.
        EVAL_REFERENCED = {110, 17, 18, 13, 1, 56, 34}
        eval_bindings = [b for b in self.get_bindings(compute=True) if b.nr in EVAL_REFERENCED]
        eval_bindings += self.gpu_objects.settings.get_bindings()       # 55 u_function_component
        eval_bindings += self._lic_uniforms.get_bindings()              # 40 u_lic
        eval_bindings += options._camera_uniforms.get_bindings()        # 0  u_camera
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
    @debounce(rate_hz=2)
    def _refresh(self):
        """Mark dirty so the next render re-runs update() (and re-dispatches the
        LIC passes) WITHOUT invalidating the volume CF data. Skips
        ClippingCF.set_needs_update's data.set_needs_update() to avoid a costly
        per-drag CF re-evaluation: neither a plane move nor a LIC-parameter change
        affects the field values, only how they are convolved/rendered."""
        super(ClippingCF, self).set_needs_update()

    def _on_camera_moved(self):
        """Camera-move observer. Unlike the debounced _refresh, this marks dirty immediately"""
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

    def set_supersample(self, value: int):
        # SSAA factor: higher = smoother/finer streaks, ~value**2 more compute.
        self.supersample = max(1, int(value))
        self._refresh()


class SurfaceLIC(CFRenderer):
    """2D surface renderer (``cf.dim == mesh.dim == 2``) that REPLACES the flat
    field colouring with a Line Integral Convolution of the (vector) coefficient
    function, computed in SCREEN space — the 2D analogue of :class:`ClippingLIC`.

    There is no cutting plane here: instead of clipping volume elements, the
    field-evaluation compute pass
    (``lic/evaluate.wgsl :: evaluate_lic_field_surface``) software-rasterises the
    mesh's surface triangles directly into a screen-space field texture under the
    live camera. The convolution pass (``line_integral_convolution.wgsl``) and the
    render-side modulation (``mesh/render.wgsl`` ``#ifdef LIC`` branch) are shared
    verbatim with the clipping path, so the streamlines track the camera the same
    way (recomputed per frame; degraded on static-HTML export).

    Because this IS the surface renderer (a ``CFRenderer``), callers should hide
    the plain surface field while it is active to avoid z-fighting — mirroring how
    the 3D LIC replaces the clip-plane field.
    """

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
        supersample: int = 2,
    ):
        super().__init__(
            data, clipping=clipping, colormap=colormap, component=component, symmetry=symmetry
        )
        self.kernel_length = int(kernel_length)
        self.oriented = bool(oriented)
        self.thickness = int(thickness)
        self.contrast = float(contrast)
        # SSAA factor: LIC computed at supersample x the canvas, box-downsampled in
        # the render pass; see ClippingLIC for the rationale. Cost ~supersample**2.
        self.supersample = max(1, int(supersample))
        # Kept for API symmetry with ClippingLIC; the screen-space textures follow
        # the canvas, so this does not set the rendered resolution.
        self.resolution = int(resolution)

        self._lic_uniforms = LicUniforms(
            width=self.resolution,
            height=self.resolution,
            kernel_length=self.kernel_length,
            oriented=int(self.oriented),
            thickness=self.thickness,
            contrast=self.contrast,
            supersample=self.supersample,
        )
        self._field_tex = None   # rgba32float: (screen_dir_x, screen_dir_y, value, mask)
        self._lic_tex = None     # rgba8unorm: (lic_value, mask, 0, 0)
        self._lic_enabled = False
        self._cam_observer_registered = False
        # A clip plane can still cut a 2D surface; re-dispatch when it moves.
        self.clipping.callbacks.append(self._refresh)

    # ------------------------------------------------------------------ update
    def update(self, options: RenderOptions):
        super().update(options)
        self._lic_enabled = False
        if not self.active:
            return
        if os.environ.get("WEBGPU_EXPORTING"):
            # Static-HTML export can't carry the GPU-computed texture; show the
            # plain surface field there (graceful degrade), as ClippingLIC does.
            return

        self._lic_enabled = True
        # Compile the LIC branch of mesh/render.wgsl. shader_defines is reset from
        # the mesh defines inside super().update(), so set this afterwards.
        self.shader_defines["LIC"] = 1

        self._register_camera_observer(options)
        self._update_lic_uniforms(options)
        self._ensure_lic_textures(options)
        self._dispatch_lic(options)

    def _register_camera_observer(self, options: RenderOptions):
        """Mark dirty whenever the camera moves so the screen-space LIC is
        re-dispatched for the new view (see ClippingLIC._register_camera_observer)."""
        if self._cam_observer_registered:
            return
        cam = getattr(options, "camera", None)
        register = getattr(cam, "register_observer", None)
        if register is not None:
            register(self._on_camera_moved)
            self._cam_observer_registered = True

    def _update_lic_uniforms(self, options: RenderOptions):
        """Fill the LIC uniform. ``width``/``height`` are the canvas size times the
        SSAA factor; there is no in-plane basis (the flow is the surface field
        itself, projected straight to screen), so tangent1/tangent2 are left as the
        world axes and are unused by the surface field pass."""
        (pmin, pmax) = self.data.get_bounding_box()
        pmin = np.asarray(pmin, dtype=np.float64)
        pmax = np.asarray(pmax, dtype=np.float64)
        centre = 0.5 * (pmin + pmax)
        radius = 0.5 * float(np.linalg.norm(pmax - pmin))
        if radius <= 1e-12:
            radius = 1.0

        f = max(1, int(self.supersample))
        u = self._lic_uniforms
        u.origin[:] = [float(centre[0]), float(centre[1]), float(centre[2]), 0.0]
        u.tangent1[:] = [1.0, 0.0, 0.0, 0.0]
        u.tangent2[:] = [0.0, 1.0, 0.0, 0.0]
        u.width = int(options.canvas.width) * f
        u.height = int(options.canvas.height) * f
        u.kernel_length = max(1, int(self.kernel_length) * f)
        u.oriented = 1 if self.oriented else 0
        u.thickness = max(1, int(self.thickness) * f)
        u.inv_extent = float(1.0 / (2.0 * radius))
        u.contrast = float(self.contrast)
        u.supersample = f
        u.update_buffer()

    def _ensure_lic_textures(self, options: RenderOptions):
        """(Re)allocate the field and LIC textures at the canvas resolution; see
        ClippingLIC._ensure_lic_textures."""
        f = max(1, int(self.supersample))
        w = int(options.canvas.width) * f
        h = int(options.canvas.height) * f
        extra = TextureUsage.COPY_SRC if os.environ.get("WEBGPU_EXPORTING") else 0
        usage = TextureUsage.STORAGE_BINDING | TextureUsage.TEXTURE_BINDING | extra
        if self._field_tex is None or self._field_tex.width != w or self._field_tex.height != h:
            self._field_tex = self.device.createTexture(
                size=[w, h, 1],
                usage=usage,
                format=TextureFormat.rgba32float,
                dimension="2d",
                label="surface_lic_field",
            )
        if self._lic_tex is None or self._lic_tex.width != w or self._lic_tex.height != h:
            self._lic_tex = self.device.createTexture(
                size=[w, h, 1],
                usage=usage,
                format=TextureFormat.rgba8unorm,
                dimension="2d",
                label="surface_lic_output",
            )

    def _lic_compute_defines(self):
        # The surface field pass evaluates the vector field with evalTrigVec3ReIm
        # (and the curvature with evalTrigVec3), both sized by MAX_EVAL_ORDER_VEC3,
        # so it must cover the field order AND the mesh curvature order. The
        # imported scalar evalTrig also references MAX_EVAL_ORDER at compile time.
        field_order = int(self.data.order)
        curv = self.data.curvature_data.order if self.data.curvature_data else 0
        d = {
            "MAX_EVAL_ORDER": max(field_order, 1),
            "MAX_EVAL_ORDER_VEC3": max(field_order, curv, 1),
        }
        if self.region_visibility is not None:
            d["REGION_VISIBILITY"] = 1
        return d

    def _dispatch_lic(self, options: RenderOptions):
        """Run the three LIC compute passes (clear, surface field, convolve). Each
        run_compute_shader submits its own encoder, so they execute sequentially."""
        # The surface field pass reads the camera uniform (binding 0) to project
        # the flow to screen; make sure it reflects the current camera first.
        options.update_buffers()
        _store_lic_mvp(self._lic_uniforms, options)

        f = max(1, int(self.supersample))
        w = int(options.canvas.width) * f
        h = int(options.canvas.height) * f
        gx = (w + 15) // 16
        gy = (h + 15) // 16
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
            label="surface_lic_clear_field",
        )

        # Stage 1b: rasterise the surface vector field into the field texture.
        # evaluate_lic_field_surface statically references only mesh(110),
        # function-values-2d(10), the LIC uniform(40), the camera(0) and the field
        # output texture(41) — hand run_compute_shader exactly that minimal set
        # (a superset can be rejected as "binding not used"). Build it explicitly
        # rather than filtering self.get_bindings(): that would touch the lazily-
        # created FunctionSettings/ComplexSettings uniforms, which the render loop
        # only initialises AFTER this update() runs.
        eval_bindings = [
            b for b in self.data.mesh_data.get_bindings()
            if getattr(b, "nr", None) == 110                            # 110 mesh
        ]
        eval_bindings += [BufferBinding(Binding.FUNCTION_VALUES_2D, self._buffers["data_2d"])]
        eval_bindings += self._lic_uniforms.get_bindings()              # 40 u_lic
        eval_bindings += options._camera_uniforms.get_bindings()        # 0  u_camera
        eval_bindings += [StorageTextureBinding(41, self._field_tex, access="write-only", dim=2)]
        if self.region_visibility is not None:
            eval_bindings += self.region_visibility.get_bindings()

        n_trigs = int(self.n_instances)
        n_wg = min(n_trigs // 256 + 1, 1024)
        run_compute_shader(
            eval_code,
            eval_bindings,
            [n_wg, 1, 1],
            entry_point="evaluate_lic_field_surface",
            defines=defines,
            label="surface_lic_evaluate_field",
        )

        # Stage 2: line integral convolution (shared, geometry-agnostic shader).
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
            label="surface_lic_convolve",
        )

    # ---------------------------------------------------------------- bindings
    def get_bindings(self):
        bindings = super().get_bindings()
        if self._lic_enabled and self._lic_tex is not None:
            # Render pass: add the LIC uniform (40) and the LIC output texture (42)
            # the #ifdef LIC branch of mesh/render.wgsl samples.
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
    @debounce(rate_hz=2)
    def _refresh(self):
        """Mark dirty so the next render re-dispatches the LIC passes WITHOUT
        re-evaluating the surface CF data (CFRenderer.set_needs_update would, which
        is wasteful per camera/clip tick — neither changes the field values)."""
        super(CFRenderer, self).set_needs_update()

    def _on_camera_moved(self):
        """Immediate camera-move observer; see ClippingLIC._on_camera_moved for
        why this is NOT debounced (the post-drag settle must re-dispatch)."""
        super(CFRenderer, self).set_needs_update()

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

    def set_supersample(self, value: int):
        self.supersample = max(1, int(value))
        self._refresh()

    def set_resolution(self, value: int):
        # Deprecated for the screen-space path (textures follow the canvas); kept
        # for API compatibility with ClippingLIC.
        self.resolution = int(value)
        self._refresh()


# Backwards-compatible / descriptive alias.
LineIntegralConvolution = ClippingLIC

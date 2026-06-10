from webgpu import create_bind_group, read_shader_file
from webgpu.utils import buffer_from_array, uniform_from_array, write_array_to_buffer
from webgpu.clipping import Clipping
from webgpu.colormap import Colormap
from webgpu.renderer import Renderer, RenderOptions
from webgpu.utils import BufferBinding, UniformBinding, ReadBuffer, run_compute_shader, read_buffer
import ctypes

from webgpu.webgpu_api import *

import numpy as np

from .cf import FunctionData, FunctionSettings, ComplexSettings, PhaseAnimation, _complex_phase_export_interactions
from .cf import Binding as CFBinding
from .cf import _bind_component_param

from .mesh import MeshElements3d, ElType
from .mesh import Binding as MeshBinding

from time import time

t0 = time()


class VolumeCF(MeshElements3d):
    fragment_entry_point: str = "cf_fragment_main"

    def __init__(self, data: FunctionData):
        super().__init__(data=data.mesh_data)
        self.data = data
        self.data.need_3d = True
        self.gpu_objects.colormap = Colormap()

    def get_bindings(self):
        return super().get_bindings() + [
            BufferBinding(10, self._buffers["data_3d"]),
            *self.gpu_objects.colormap.get_bindings(),
        ]

    def get_shader_code(self):
        return read_shader_file("ngsolve/mesh/render.wgsl")


class ClippingCF(Renderer):
    vertex_entry_point = "vertex_clipping"
    fragment_entry_point = "fragment_clipping"
    compute_shader = "ngsolve/clipping/compute.wgsl"
    select_entry_point = "select_clipping"
    n_vertices = 3
    subdivision = 0

    def __init__(
        self, data: FunctionData, clipping: Clipping = None, colormap: Colormap = None, component=None, symmetry=None
    ):
        super().__init__()
        from .pick import HighlightUniforms
        self._clipping = clipping or Clipping()
        self.clipping = Clipping()
        self.gpu_objects.colormap = colormap or Colormap()
        self.data = data
        self.data.need_3d = True
        if self.data.mesh_data.deformation_data is not None:
            self.data.mesh_data.deformation_data.need_3d = True
        self.options = None
        self.cut_trigs = None
        self.trig_counter = None

        self.n_tets = None
        if component is None:
            component = -1 if self.data.cf.dim > 1 else 0
        self.gpu_objects.settings = FunctionSettings(component=component)
        self.gpu_objects.complex_settings = ComplexSettings()
        self._highlight_uniforms = HighlightUniforms()
        self._phase_animation = None
        self._scene = None
        self._anim_speed = 1.0
        self.symmetry = symmetry
        self._gui_params = []
        self._bind_component_param()

    def _bind_component_param(self):
        """Auto-bind to FunctionData's shared component_param."""
        _bind_component_param(self)

    @property
    def colormap(self):
        return self.gpu_objects.colormap

    @colormap.setter
    def colormap(self, value: Colormap):
        self.gpu_objects.colormap = value

    def set_needs_update(self):
        """Invalidate field data so the next update re-evaluates it."""
        self.data.set_needs_update()
        super().set_needs_update()

    def update(self, options: RenderOptions):
        self.data.update(options)
        if self.data.data_3d is None:
            self.active = False
            return
        self.shader_defines = self.data.mesh_data.get_shader_defines()
        clip_subdiv = self.data.mesh_data.subdivision or 1
        if self.data.mesh_data.curvature_3d_data is None:
            clip_subdiv = 1
        self.shader_defines["CLIPPING_SUBDIVISION"] = clip_subdiv
        self.n_vertices = 3 * clip_subdiv * clip_subdiv
        self._clipping.update(options)

        if not hasattr(self.clipping, "uniforms"):
            self.clipping.update(options)

        ctypes.memmove(
            ctypes.addressof(self.clipping.uniforms),
            ctypes.addressof(self._clipping.uniforms),
            ctypes.sizeof(self.clipping.uniforms),
        )
        self.clipping.uniforms.update_buffer()

        self._buffers = self.data.get_buffers()
        if self.gpu_objects.colormap.autoscale:
            component = self.gpu_objects.settings.component
            self.gpu_objects.colormap.widen_range(
                self.data.minval[component + 1],
                self.data.maxval[component + 1],
                timestamp=options.timestamp,
            )
        self.gpu_objects.complex_settings.update(options)
        self._ensure_compute_buffers()
        if self.symmetry:
            self.n_instances = self._original_n_instances * self.symmetry.n_copies
            self.shader_defines["SYMMETRY"] = "1"

    def get_bounding_box(self):
        bbox = self.data.get_bounding_box()
        if self.symmetry:
            bbox = self.symmetry.expand_bbox(bbox)
        return bbox

    def get_shader_code(self):
        return read_shader_file("ngsolve/clipping/render.wgsl")

    def set_component(self, component: int):
        self.gpu_objects.settings.component = component

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

    def _toggle_animation(self, value):
        if value:
            self.animate_phase(speed=self._anim_speed)
        else:
            self.stop_animation()
            self.set_complex_mode("real")
            if self._scene:
                self._scene.render()

    def _set_phase_from_gui(self, value):
        self.gpu_objects.complex_settings.mode = ComplexSettings.PHASE_ROTATE
        self.set_phase(value)
        if self._scene:
            self._scene.render()

    def _set_speed_from_gui(self, value):
        self._anim_speed = value
        if self._phase_animation is not None:
            self._phase_animation.speed = value

    def get_export_interactions(self, options, buffer_registry):
        from webgpu.export.gui import gui_interaction, Checkbox, Write, Target
        out = list(super().get_export_interactions(options, buffer_registry))
        if self.data.cf.is_complex:
            out += _complex_phase_export_interactions(
                [(self.gpu_objects.complex_settings.uniform, True)],
                buffer_registry,
            )
        # Toggle the clipping function render pass on/off.
        out.append(gui_interaction(
            "Clipping Function",
            [Checkbox(var="enabled", name="Enabled", default=True)],
            [Write(
                targets=[Target(f"render_{self._id}", dtype="pass_enable")],
                expr="enabled ? 1 : 0", trigger="enabled",
            )],
        ))
        return out

    def get_bindings(self, compute=False):
        bindings = [
            *self.data.mesh_data.get_bindings(),
            UniformBinding(22, self.n_tets),

            # BufferBinding(MeshBinding.TET, self._buffers[ElType.TET]),
            BufferBinding(13, self._buffers["data_3d"]),
            *self.clipping.get_bindings(),
        ]
        if compute:
            bindings += [
                BufferBinding(
                    21,
                    self.trig_counter,
                    read_only=False,
                    visibility=ShaderStage.COMPUTE,
                ),
                BufferBinding(24, self.cut_trigs, read_only=False),
                *self.gpu_objects.complex_settings.get_bindings(),
            ]
        else:
            bindings += [
                *self.gpu_objects.colormap.get_bindings(),
                *self.gpu_objects.settings.get_bindings(),
                *self.gpu_objects.complex_settings.get_bindings(),
                BufferBinding(24, self.cut_trigs),
                *self._highlight_uniforms.get_bindings(),
            ]
            if self.symmetry:
                bindings += self.symmetry.get_bindings(self._original_n_instances)
        return bindings

    def _ensure_compute_buffers(self):
        """Allocate the GPU buffers the clip-plane compute pass binds.

        The actual count+fill compute is owned by the JS engine via the
        countThenFill protocol (see get_export_compute_passes): it sizes the
        output buffer (cut_trigs) and the indirect draw count itself, reading
        the atomic counter GPU-side. Python therefore never dispatches the
        compute or reads the count back — it only needs:

          * trig_counter / n_tets / cut_trigs to exist so get_bindings() and the
            Python render-pipeline bind-group layout are valid, and
          * n_instances > 0 so capture_scene_live() captures this renderer
            (the real per-frame instance count comes from the GPU indirect
            buffer at draw time, not from this value).

        cut_trigs is a minimal placeholder; the JS engine creates and resizes
        its own owned buffer in initPipelines()/processReadbacks().
        """
        ntets = self.data.mesh_data.num_elements[ElType.TET] * 4**self.subdivision
        self.n_tets = uniform_from_array(
            np.array([ntets], dtype=np.uint32), label="n_tets", reuse=self.n_tets
        )
        if self.trig_counter is None:
            self.trig_counter = buffer_from_array(
                np.array([0], dtype=np.uint32),
                usage=BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,
                label="trig_counter",
            )
        if self.cut_trigs is None:
            self.cut_trigs = self.device.createBuffer(
                size=64, usage=BufferUsage.STORAGE, label="cut_trigs"
            )
        self.n_instances = 1
        self._original_n_instances = 1
        if self.symmetry:
            self.n_instances = self._original_n_instances * self.symmetry.n_copies
            self.shader_defines["SYMMETRY"] = "1"

    def get_export_descriptor(self, options, buffer_registry):
        # Use _clipping (user-facing) buffer so render shader sees GUI updates
        saved_clipping = self.clipping
        self.clipping = self._clipping
        result = super().get_export_descriptor(options, buffer_registry)
        self.clipping = saved_clipping
        # Use drawIndirect if compute passes set up the indirect buffer
        if hasattr(self, '_export_indirect_buf_id'):
            result.draw_indirect = self._export_indirect_buf_id
        return result

    def get_export_compute_passes(self, options, buffer_registry):
        from webgpu.export.format import ExportComputePass
        from webgpu.utils import preprocess_shader_code

        shader = preprocess_shader_code(
            read_shader_file(self.compute_shader),
            defines=self.shader_defines,
        )

        # For export, use _clipping (user-facing) buffer in compute bindings
        saved_clipping = self.clipping
        self.clipping = self._clipping

        # The JS engine owns the output buffer (cut_trigs) and indirect buffer
        # entirely via the countThenFill protocol.  Python only needs minimal
        # placeholders so the buffer registry can assign stable IDs.
        min_size = 64
        if self.cut_trigs is None or self.cut_trigs.size < min_size:
            self.cut_trigs = self.device.createBuffer(
                size=min_size, usage=BufferUsage.STORAGE, label="cut_trigs"
            )

        fill_bindings = self.get_bindings(compute=True)
        fill_binding_map = buffer_registry.register_bindings(fill_bindings)

        self.clipping = saved_clipping

        clipping_buf_id = buffer_registry.get_id(self._clipping.uniforms._buffer)
        trig_counter_id = buffer_registry.get_id(self.trig_counter)
        cut_trigs_id = buffer_registry.get_id(self.cut_trigs)

        # Indirect buffer: also JS-owned.  Create a minimal placeholder for
        # the registry ID; the JS engine creates the real one in initPipelines.
        if not hasattr(self, '_indirect_buffer') or self._indirect_buffer is None:
            self._indirect_buffer = self.device.createBuffer(
                size=16,
                usage=BufferUsage.INDIRECT | BufferUsage.STORAGE | BufferUsage.COPY_DST,
                label="clip_indirect",
            )
        indirect_buf_id = buffer_registry.register_buffer(self._indirect_buffer, "indirect")
        self._export_indirect_buf_id = indirect_buf_id

        fill_pass_id = f"clip_fill_{self._id}"

        return [
            ExportComputePass(
                id=fill_pass_id,
                shader=shader,
                bindings=fill_binding_map,
                workgroups=[1024, 1, 1],
                triggers=[clipping_buf_id],
                reset_buffers=[trig_counter_id],
                count_then_fill={
                    "counter_id": trig_counter_id,
                    "output_id": cut_trigs_id,
                    "element_size": 64,  # sizeof(SubTrig)
                    "indirect_id": indirect_buf_id,
                    "vertex_count": self.n_vertices,
                },
            ),
        ]

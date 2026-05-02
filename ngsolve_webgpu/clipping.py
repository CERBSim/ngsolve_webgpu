from webgpu import create_bind_group, read_shader_file
from webgpu.utils import buffer_from_array, uniform_from_array, write_array_to_buffer
from webgpu.clipping import Clipping
from webgpu.colormap import Colormap
from webgpu.renderer import Renderer, RenderOptions
from webgpu.utils import BufferBinding, UniformBinding, ReadBuffer, run_compute_shader, read_buffer
import ctypes

from webgpu.webgpu_api import *

import numpy as np

from .cf import FunctionData, FunctionSettings, ComplexSettings, PhaseAnimation
from .cf import Binding as CFBinding

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
    select_entry_point = "fragment_select_no_clipping"
    n_vertices = 3
    subdivision = 0

    def __init__(
        self, data: FunctionData, clipping: Clipping = None, colormap: Colormap = None, component=-1, symmetry=None
    ):
        super().__init__()
        self._clipping = clipping or Clipping()
        self.clipping = Clipping()
        self.gpu_objects.colormap = colormap or Colormap()
        self._clipping.callbacks.append(self.set_needs_update)
        self.data = data
        self.data.need_3d = True
        if self.data.mesh_data.deformation_data is not None:
            self.data.mesh_data.deformation_data.need_3d = True
        self.options = None
        self.cut_trigs_counter = None
        self.cut_trigs = None
        self.trig_counter = None
        self.only_count = None
        self.n_tets = None
        if component is None:
            component = -1 if self.data.cf.dim > 1 else 0
        self.gpu_objects.settings = FunctionSettings(component=component)
        self.gpu_objects.complex_settings = ComplexSettings()
        self._phase_animation = None
        self._scene = None
        self._anim_speed = 1.0
        self.symmetry = symmetry

    @property
    def colormap(self):
        return self.gpu_objects.colormap

    @colormap.setter
    def colormap(self, value: Colormap):
        self.gpu_objects.colormap = value

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
            self.gpu_objects.colormap.set_min_max(
                self.data.minval[component + 1],
                self.data.maxval[component + 1],
                set_autoscale=False,
            )
        self.gpu_objects.complex_settings.update(options)
        self.build_clip_plane()
        if self.symmetry:
            self.n_instances = self._original_n_instances * self.symmetry.n_copies
            self.shader_defines["SYMMETRY"] = "1"

    def get_bounding_box(self):
        return self.data.get_bounding_box()

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

    def add_options_to_gui(self, gui):
        if gui is None:
            return
        def set_enabled(value):
            self.active = value
            self.set_needs_update()
        folder = gui.folder("Clipping", closed=True)
        folder.checkbox("function", self.active, set_enabled)
        if self.data.cf.is_complex:
            f = folder.folder("Complex")
            complex_options = {"Real": "real", "Imag": "imag", "Abs": "abs", "Arg": "arg"}
            f.dropdown(func=self.set_complex_mode, label="Mode", values=complex_options)
            f.slider(0.0, func=self._set_phase_from_gui, min=0.0, max=6.283, label="Phase")
            f.checkbox(func=self._toggle_animation, label="Animate", value=False)
            f.slider(1.0, func=self._set_speed_from_gui, min=0.1, max=5.0, label="Speed")

    def get_bindings(self, compute=False, count: bool = False):
        bindings = [
            *self.data.mesh_data.get_bindings(),
            UniformBinding(22, self.n_tets),
            UniformBinding(23, self.only_count),
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
                BufferBinding(
                    24, self.cut_trigs_counter if count else self.cut_trigs, read_only=False
                ),
                *self.gpu_objects.complex_settings.get_bindings(),
            ]
        else:
            bindings += [
                *self.gpu_objects.colormap.get_bindings(),
                *self.gpu_objects.settings.get_bindings(),
                *self.gpu_objects.complex_settings.get_bindings(),
                BufferBinding(24, self.cut_trigs),
            ]
            if self.symmetry:
                bindings += self.symmetry.get_bindings(self._original_n_instances)
        return bindings

    def build_clip_plane(self):
        for count in [True, False]:
            ntets = self.data.mesh_data.num_elements[ElType.TET] * 4**self.subdivision
            self.trig_counter = buffer_from_array(
                np.array([0], dtype=np.uint32),
                usage=BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,
                label="trig_counter",
                reuse=self.trig_counter,
            )
            self.n_tets = uniform_from_array(
                np.array([ntets], dtype=np.uint32), label="n_tets", reuse=self.n_tets
            )
            self.only_count = uniform_from_array(
                np.array([count], dtype=np.uint32), label="only_count", reuse=self.only_count
            )
            if count:
                self.cut_trigs_counter = buffer_from_array(
                    np.array([0.0] * 64, dtype=np.float32),
                    label="cut_trigs_counter",
                    reuse=self.cut_trigs_counter,
                )
            else:
                buffer_size = max(64, 64 * self.n_instances)
                if self.cut_trigs is None or self.cut_trigs.size < buffer_size:
                    if self.cut_trigs is not None:
                        self.cut_trigs.destroy()
                    self.cut_trigs = self.device.createBuffer(
                        size=buffer_size, usage=BufferUsage.STORAGE, label="cut_trigs"
                    )

            shader_code = read_shader_file(self.compute_shader)
            run_compute_shader(
                shader_code, self.get_bindings(compute=True, count=count), 1024, "build_clip_plane", defines=self.data.mesh_data.get_shader_defines()
            )
            if count:
                self.n_instances = int(read_buffer(self.trig_counter, np.uint32)[0])
                self._original_n_instances = self.n_instances

    def render(self, options: RenderOptions):
        if bytes(self._clipping.uniforms) != bytes(self.clipping.uniforms):
            self.set_needs_update()
        super().render(options)

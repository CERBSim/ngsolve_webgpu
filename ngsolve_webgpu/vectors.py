import ctypes
import numpy as np
import ngsolve as ngs

from webgpu.colormap import Colormap
from webgpu.clipping import Clipping
from webgpu.shapes import ShapeRenderer, generate_cone, generate_cylinder
from webgpu.utils import (
    create_buffer,
    BufferUsage,
    BufferBinding,
    ReadBuffer,
    UniformBinding,
    read_buffer,
    read_shader_file,
    run_compute_shader,
    uniform_from_array,
    buffer_from_array,
    write_array_to_buffer,
)

from .cf import FunctionData, MeshData, ComplexSettings, PhaseAnimation, Binding as FunctionBinding
from .mesh import Binding as MeshBinding
from .mesh import ElType

import math
import time
import threading


class _VectorPhaseAnimation:
    """Phase animation that updates both ngsolve ComplexSettings and ShapeRenderer's complex uniform."""

    def __init__(self, renderer, scene, speed=1.0, fps=60):
        self.renderer = renderer
        self.scene = scene
        self.speed = speed
        self._fps = fps
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread = None

    def _loop(self):
        while self._running:
            t = time.time() - self._t0
            phase = (t * self.speed * 2 * math.pi) % (2 * math.pi)
            self.renderer.set_phase(phase)
            self.scene.render()
            time.sleep(1 / self._fps)

class VectorRenderer(ShapeRenderer):
    def __init__(
        self,
        function_data: FunctionData,
        grid_size: float = 20,
        clipping: Clipping = None,
        colormap: Colormap = None,
        symmetry=None,
        vector_symmetry="polar",
        scale_by_value: bool = False,
    ):
        self.u_nvectors = None
        self.clipping = clipping or Clipping()
        self.function_data = function_data
        self.scale_by_value = scale_by_value
        mesh = function_data.mesh_data
        bbox = mesh.get_bounding_box()
        self.box_size = np.linalg.norm(np.array(bbox[1]) - np.array(bbox[0]))
        self.set_grid_size(grid_size)
        self.__buffers = {}
        self.symmetry = symmetry
        self.vector_symmetry = vector_symmetry
        self._expanded_buffers = {}
        self._complex_settings = ComplexSettings()
        self._phase_animation = None
        self._scene = None
        self._anim_speed = 1.0
        self.user_scale = 1.0
        super().__init__(self.generate_shape(), None, None, colormap=colormap)

    def set_grid_size(self, grid_size: float):
        self.grid_spacing = 1/grid_size * self.box_size
        if hasattr(self, "u_grid_spacing") and self.u_grid_spacing is not None:
            write_array_to_buffer(
                self.u_grid_spacing, np.array([self.grid_spacing], dtype=np.float32)
            )
        
    def get_bounding_box(self):
        return self.function_data.mesh_data.get_bounding_box()
        
    def generate_shape(self):
        cyl = generate_cylinder(8, 0.05, 0.5, bottom_face=True)
        cone = generate_cone(8, 0.2, 0.5, bottom_face=True)
        arrow = cyl + cone.move((0, 0, 0.5))
        return arrow

    def get_compute_bindings(self):
        self.u_grid_spacing = uniform_from_array(
            np.array([self.grid_spacing], dtype=np.float32), label="grid_spacing",
            reuse=self.u_grid_spacing if hasattr(self, "u_grid_spacing") else None
        )
        buffers = self.function_data.get_buffers()
        mesh_data = self.function_data.mesh_data
        self._u_subdivision = uniform_from_array(
            np.array([mesh_data.subdivision or 1], dtype=np.uint32), label="subdivision",
            reuse=getattr(self, '_u_subdivision', None)
        )
        self._u_component = uniform_from_array(
            np.array([-1], dtype=np.int32), label="component",
            reuse=getattr(self, '_u_component', None)
        )
        # MeshUniforms: {subdivision: u32, shrink: f32, padding0: f32, padding1: f32}
        mesh_uniforms = np.zeros(4, dtype=np.float32)
        mesh_uniforms[0] = np.frombuffer(np.array([mesh_data.subdivision or 1], dtype=np.uint32).tobytes(), dtype=np.float32)[0]
        mesh_uniforms[1] = 1.0  # shrink
        self._u_mesh_uniforms = uniform_from_array(
            mesh_uniforms, label="mesh_uniforms",
            reuse=getattr(self, '_u_mesh_uniforms', None)
        )
        bindings = [
            # Only include storage buffers that are actually used by the shader.
            # WebGPU maxStorageBuffersPerShaderStage default is 8; we must stay under it.
            # mesh data (storage, ro) - used by loadTriangle, mesh.num_trigs
            BufferBinding(MeshBinding.MESH_DATA, mesh_data.gpu_data),
            # uniforms from eval/common and mesh/utils (these don't count toward storage limit)
            *self._complex_settings.get_bindings(),
            UniformBinding(MeshBinding.DEFORMATION_SCALE, mesh_data.gpu_elements['deformation_scale']),
            UniformBinding(15, self._u_subdivision),  # u_subdivision
            UniformBinding(20, self._u_mesh_uniforms),  # u_mesh (MeshUniforms)
            UniformBinding(55, self._u_component),  # u_function_component
            # compute-specific bindings
            BufferBinding(21, self.u_nvectors, read_only=False),
            BufferBinding(22, self.__buffers["positions"], read_only=False),
            BufferBinding(23, self.__buffers["directions"], read_only=False),
            BufferBinding(29, self.__buffers["values"], read_only=False),
            UniformBinding(31, self.u_grid_spacing),
        ]
        if self.function_data.cf.is_complex:
            bindings.append(BufferBinding(25, self.__buffers["directions_imag"], read_only=False))
        return bindings

    def allocate_buffers(self):
        is_complex = self.function_data.cf.is_complex
        names = ["positions", "directions", "values"]
        if is_complex:
            names.append("directions_imag")
        for name in names:
            size = 4 * self.n_vectors if name == "values" else 3 * 4 * self.n_vectors
            self.__buffers[name] = create_buffer(
                size=size,
                usage=BufferUsage.VERTEX | BufferUsage.STORAGE,
                label=name,
                reuse=self.__buffers.get(name, None),
            )
        if not is_complex:
            self.__buffers.pop("directions_imag", None)
            
    def compute_vectors(self):
        self.u_nvectors = buffer_from_array(
            np.array([0], dtype=np.uint32),
            label="n_vectors",
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,
            reuse=self.u_nvectors,
        )
        self.n_vectors = 1
        self.allocate_buffers()
        assert hasattr(self, "n_search_els") and self.n_search_els > 0, "n_search_els should be set in the renderer"
        assert hasattr(self, "compute_shader_file") and hasattr(self, "compute_entry_point"), "compute_shader_file and compute_entry_point should be set in the renderer"
        self.u_grid_spacing = uniform_from_array(
            np.array([self.grid_spacing], dtype=np.float32), label="grid_spacing",
            reuse=self.u_grid_spacing if hasattr(self, "u_grid_spacing") else None
        )
        n_work_groups = min(self.n_search_els // 256 + 1, 1024)
        count_shader = self.compute_shader_file.replace('.wgsl', '_count.wgsl')
        run_compute_shader(
            read_shader_file(count_shader),
            [BufferBinding(110, self.function_data.mesh_data.gpu_data),
             BufferBinding(21, self.u_nvectors, read_only=False),
             UniformBinding(31, self.u_grid_spacing)],
            n_work_groups,
            entry_point=self.compute_entry_point,
        )
        self.n_vectors = int(read_buffer(self.u_nvectors, np.uint32)[0])
        write_array_to_buffer(self.u_nvectors, np.array([0], dtype=np.uint32))
        self.allocate_buffers()
        defines = {
                "MAX_EVAL_ORDER": self.function_data.order,
                "MAX_EVAL_ORDER_VEC3": self.function_data.order,
            }
        if self.function_data.cf.is_complex:
            defines["IS_COMPLEX"] = 1
        if self.scale_by_value:
            defines["SCALE_BY_VALUE"] = 1
        run_compute_shader(
            read_shader_file(self.compute_shader_file),
            self.get_compute_bindings(),
            n_work_groups,
            entry_point=self.compute_entry_point,
            defines=defines,
        )
        self.positions_buffer = self.__buffers["positions"]
        self.directions_buffer = self.__buffers["directions"]
        self.directions_imag_buffer = self.__buffers.get("directions_imag", None)
        self.values_buffer = self.__buffers["values"]
        if self.symmetry and self.symmetry.n_copies > 1:
            self._expand_vectors_for_symmetry()
        
    def _expand_vectors_for_symmetry(self):
        """Run a GPU compute shader to copy+transform vectors for all symmetry copies."""
        n = self.n_vectors
        nc = self.symmetry.n_copies
        total = n * nc

        # Allocate expanded buffers
        for name in ["positions", "directions", "directions_imag", "values"]:
            size = 4 * total if name == "values" else 3 * 4 * total
            self._expanded_buffers[name] = create_buffer(
                size=size,
                usage=BufferUsage.VERTEX | BufferUsage.STORAGE,
                label=f"sym_{name}",
                reuse=self._expanded_buffers.get(name, None),
            )

        sym_bindings = self.symmetry.get_bindings(n)

        bindings = [
            BufferBinding(0, self.__buffers["positions"]),
            BufferBinding(1, self.__buffers["directions"]),
            BufferBinding(2, self.__buffers["values"]),
            BufferBinding(3, self._expanded_buffers["positions"], read_only=False),
            BufferBinding(4, self._expanded_buffers["directions"], read_only=False),
            BufferBinding(5, self._expanded_buffers["values"], read_only=False),
            *sym_bindings,
        ]
        defines = {}
        if self.vector_symmetry == "axial":
            defines["AXIAL_VECTORS"] = "1"

        n_work_groups = min(total // 256 + 1, 1024)
        run_compute_shader(
            read_shader_file("ngsolve/symmetry_expand_vectors.wgsl"),
            bindings,
            n_work_groups,
            entry_point="expand_vectors",
            defines=defines,
        )

        self.positions_buffer = self._expanded_buffers["positions"]
        self.directions_buffer = self._expanded_buffers["directions"]
        self.directions_imag_buffer = self._expanded_buffers["directions_imag"]
        self.values_buffer = self._expanded_buffers["values"]

        # Expand imag directions too
        bindings_imag = [
            BufferBinding(0, self.__buffers["positions"]),
            BufferBinding(1, self.__buffers["directions_imag"]),
            BufferBinding(2, self.__buffers["values"]),
            BufferBinding(3, self._expanded_buffers["positions"], read_only=False),
            BufferBinding(4, self._expanded_buffers["directions_imag"], read_only=False),
            BufferBinding(5, self._expanded_buffers["values"], read_only=False),
            *sym_bindings,
        ]
        run_compute_shader(
            read_shader_file("ngsolve/symmetry_expand_vectors.wgsl"),
            bindings_imag,
            n_work_groups,
            entry_point="expand_vectors",
            defines=defines,
        )

    def update(self, options):
        self.function_data.update(options)
        self._complex_settings.update(options)
        self.compute_vectors()
        is_complex = self.function_data.cf.is_complex
        max_val = self.function_data.maxval[0]
        if self.scale_by_value:
            # max arrow ~ 4 * grid_spacing
            self._scale = self.grid_spacing * 4.0 / max(max_val, 1e-10) * self.user_scale
            self._scale_mode = 0
        else:
            # fixed size ~ 5% of bounding box diagonal
            self._scale = self.box_size * 0.01 * self.user_scale
            self._scale_mode = 2
        if self.gpu_objects.colormap.autoscale:
            self.gpu_objects.colormap.set_min_max(
                self.function_data.minval[0],
                self.function_data.maxval[0],
                set_autoscale=False,
            )
        if self.active:
            super().update(options)
            if self._complex_uniforms is not None:
                self._complex_uniforms.is_complex = 1 if is_complex else 0
                self._complex_uniforms.color_override = 1 if (is_complex and self.scale_by_value) else 0
                self._complex_uniforms.update_buffer()

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
        self._complex_settings.mode = shader_mode
        if phase is not None:
            self._complex_settings.phase = phase

    def set_phase(self, phase: float):
        """Set the phase angle for complex animate mode"""
        self._complex_settings.phase = phase
        # Also update the ShapeRenderer's complex uniform (binding 11)
        if self._complex_uniforms is not None:
            self._complex_uniforms.phase = phase
            self._complex_uniforms.update_buffer()

    def animate_phase(self, scene=None, speed=1.0, fps=60):
        """Start phase-sweep animation."""
        scene = scene or self._scene
        if scene is None:
            raise ValueError("No scene available.")
        self.stop_animation()
        # Use a wrapper that updates both uniforms
        self._phase_animation = _VectorPhaseAnimation(
            self, scene, speed=speed, fps=fps
        )
        self._phase_animation.start()

    def stop_animation(self):
        """Stop phase-sweep animation."""
        if self._phase_animation is not None:
            self._phase_animation.stop()
            self._phase_animation = None

    def add_options_to_gui(self, gui):
        if gui is None:
            return
        super().add_options_to_gui(gui)
        if self.function_data.cf.is_complex:
            f = gui.folder("Complex")
            complex_options = {"Real": "real", "Imag": "imag", "Abs": "abs"}
            f.dropdown(func=self.set_complex_mode, label="Mode", values=complex_options)
            f.slider(0.0, func=self._set_phase_from_gui, min=0.0, max=6.283, label="Phase")
            f.checkbox(func=self._toggle_animation, label="Animate", value=False)
            f.slider(1.0, func=self._set_speed_from_gui, min=0.1, max=5.0, label="Speed")

    def _set_phase_from_gui(self, value):
        self._complex_settings.mode = ComplexSettings.PHASE_ROTATE
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

class SurfaceVectors(VectorRenderer):
    def __init__(self, function_data: FunctionData, grid_size: float = 20, clipping: Clipping = None, colormap: Colormap = None, symmetry=None, vector_symmetry="polar", scale_by_value: bool = False):
        self.compute_shader_file = "ngsolve/surface_vectors.wgsl"
        self.compute_entry_point = "compute_surface_vectors"
        self.u_ntrigs = None
        super().__init__(function_data=function_data, grid_size=grid_size, clipping=clipping, colormap=colormap, symmetry=symmetry, vector_symmetry=vector_symmetry, scale_by_value=scale_by_value)
        
    def update(self, options):
        self.n_search_els = self.function_data.mesh_data.ngs_mesh.GetNE(ngs.BND)
        super().update(options)
        
    def compute_vectors(self):
        if "data_2d" not in self.function_data.get_buffers():
            self.active = False
            return
        return super().compute_vectors()

    def get_compute_bindings(self):
        bindings = super().get_compute_bindings()
        buffers = self.function_data.get_buffers()
        return bindings + [
            BufferBinding(FunctionBinding.FUNCTION_VALUES_2D, buffers["data_2d"]),
        ]

class ClippingVectors(VectorRenderer):
    def __init__(
        self,
        function_data: FunctionData,
        grid_size: float = 20,
        clipping: Clipping = None,
        colormap: Colormap = None,
        symmetry=None,
        vector_symmetry="polar",
        scale_by_value: bool = False,
    ):
        self.compute_shader_file = "ngsolve/clipping_vectors.wgsl"
        self.compute_entry_point = "compute_clipping_vectors"
        self.u_ntets = None
        function_data.need_3d = True
        super().__init__(
            function_data=function_data, grid_size=grid_size, clipping=clipping,
            colormap=colormap, symmetry=symmetry, vector_symmetry=vector_symmetry,
            scale_by_value=scale_by_value)
        self.__clipping = Clipping()
        self.clipping.callbacks.append(self.set_needs_update)
        
    def get_compute_bindings(self):
        bindings = super().get_compute_bindings()
        buffers = self.function_data.get_buffers()
        mesh_data = self.function_data.mesh_data
        dummy = mesh_data._dummy_buffer
        self.u_ntets = uniform_from_array(
            np.array([self.n_search_els], dtype=np.uint32), label="n_tets",
            reuse=self.u_ntets)
        deform_3d = (mesh_data.deformation_data.gpu_3d
                     if mesh_data.deformation_data and mesh_data.deformation_data.gpu_3d
                     else dummy)
        return bindings + [
            *self.__clipping.get_bindings(),
            BufferBinding(FunctionBinding.FUNCTION_VALUES_3D, buffers["data_3d"]),
            BufferBinding(MeshBinding.DEFORMATION_3D_VALUES, deform_3d),
            UniformBinding(24, self.u_ntets),
        ]

    def compute_vectors(self):
        if "data_3d" not in self.function_data.get_buffers():
            self.active = False
            return
        self.u_nvectors = buffer_from_array(
            np.array([0], dtype=np.uint32),
            label="n_vectors",
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,
            reuse=self.u_nvectors,
        )
        self.n_vectors = 1
        self.allocate_buffers()
        assert self.n_search_els > 0
        self.u_grid_spacing = uniform_from_array(
            np.array([self.grid_spacing], dtype=np.float32), label="grid_spacing",
            reuse=self.u_grid_spacing if hasattr(self, "u_grid_spacing") else None
        )
        self.u_ntets = uniform_from_array(
            np.array([self.n_search_els], dtype=np.uint32), label="n_tets",
            reuse=self.u_ntets)
        mesh_data = self.function_data.mesh_data
        dummy = mesh_data._dummy_buffer
        n_work_groups = min(self.n_search_els // 256 + 1, 1024)

        # Count pass: MODE=0 - only needs minimal storage buffers
        deform_3d = (mesh_data.deformation_data.gpu_3d
                     if mesh_data.deformation_data and mesh_data.deformation_data.gpu_3d
                     else dummy)
        count_bindings = [
            BufferBinding(MeshBinding.MESH_DATA, mesh_data.gpu_data),
            BufferBinding(MeshBinding.DEFORMATION_3D_VALUES, deform_3d),
            BufferBinding(21, self.u_nvectors, read_only=False),
            *self.__clipping.get_bindings(),
            *self._complex_settings.get_bindings(),
            UniformBinding(MeshBinding.DEFORMATION_SCALE, mesh_data.gpu_elements['deformation_scale']),
            UniformBinding(15, uniform_from_array(
                np.array([mesh_data.subdivision or 1], dtype=np.uint32),
                reuse=getattr(self, '_u_subdivision', None))),
            UniformBinding(24, self.u_ntets),
            UniformBinding(31, self.u_grid_spacing),
            UniformBinding(55, uniform_from_array(
                np.array([-1], dtype=np.int32),
                reuse=getattr(self, '_u_component', None))),
        ]
        run_compute_shader(
            read_shader_file(self.compute_shader_file),
            count_bindings,
            n_work_groups,
            entry_point=self.compute_entry_point,
            defines={"MODE": 0, "MAX_EVAL_ORDER": self.function_data.order, "MAX_EVAL_ORDER_VEC3": self.function_data.order},
        )
        self.n_vectors = int(read_buffer(self.u_nvectors, np.uint32)[0])
        write_array_to_buffer(self.u_nvectors, np.array([0], dtype=np.uint32))
        self.allocate_buffers()

        # Eval pass: MODE=1, NEED_EVAL - needs function data + output buffers
        eval_defines = {"MODE": 1, "NEED_EVAL": 1, "MAX_EVAL_ORDER": self.function_data.order, "MAX_EVAL_ORDER_VEC3": self.function_data.order}
        if self.function_data.cf.is_complex:
            eval_defines["IS_COMPLEX"] = 1
        if self.scale_by_value:
            eval_defines["SCALE_BY_VALUE"] = 1
        run_compute_shader(
            read_shader_file(self.compute_shader_file),
            self.get_compute_bindings(),
            n_work_groups,
            entry_point=self.compute_entry_point,
            defines=eval_defines,
        )
        self.positions_buffer = self._VectorRenderer__buffers["positions"]
        self.directions_buffer = self._VectorRenderer__buffers["directions"]
        self.directions_imag_buffer = self._VectorRenderer__buffers.get("directions_imag", None)
        self.values_buffer = self._VectorRenderer__buffers["values"]
        if self.symmetry and self.symmetry.n_copies > 1:
            self._expand_vectors_for_symmetry()

    def update(self, options):
        self.function_data.update(options)
        self.n_search_els = self.function_data.mesh_data.num_elements["tets"]
        self.clipping.update(options)
        if not hasattr(self.__clipping, "uniforms"):
            self.__clipping.update(options)

        ctypes.memmove(
            ctypes.addressof(self.__clipping.uniforms),
            ctypes.addressof(self.clipping.uniforms),
            ctypes.sizeof(self.clipping.uniforms),
        )
        self.__clipping.uniforms.update_buffer()
        super().update(options)

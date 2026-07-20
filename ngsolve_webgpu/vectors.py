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

from .cf import FunctionData, MeshData, ComplexSettings, PhaseAnimation, Binding as FunctionBinding, _complex_phase_export_interactions
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
    _vector_gui_label = "Vectors"

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
        self._last_data_ts = None
        self.clipping = clipping or Clipping()
        self.function_data = function_data
        self.scale_by_value = scale_by_value
        mesh = function_data.mesh_data
        bbox = mesh.get_bounding_box()
        # Use the longest side of the bounding box (not its diagonal) so that
        # thin slabs and cubes of similar lateral extent get a similar sampling
        # density. The diagonal would couple the spacing to the aspect ratio.
        sides = np.array(bbox[1]) - np.array(bbox[0])
        self.box_size = float(np.max(sides))
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
        self._directions_per_instance = True
        self.region_visibility = None
        super().__init__(self.generate_shape(), None, None, colormap=colormap)

    def set_grid_size(self, grid_size: float):
        self.grid_spacing = 1/grid_size * self.box_size
        if hasattr(self, "u_grid_spacing") and self.u_grid_spacing is not None:
            write_array_to_buffer(
                self.u_grid_spacing, np.array([self.grid_spacing], dtype=np.float32)
            )
        
    def set_needs_update(self):
        self.function_data.set_needs_update()
        super().set_needs_update()

    def get_bounding_box(self):
        bbox = self.function_data.mesh_data.get_bounding_box()
        if self.symmetry:
            bbox = self.symmetry.expand_bbox(bbox)
        return bbox
        
    def generate_shape(self):
        cyl = generate_cylinder(8, 0.05, 0.5, bottom_face=True)
        cone = generate_cone(8, 0.15, 0.5, bottom_face=True)
        arrow = cyl + cone.move((0, 0, 0.5))
        return arrow.move((0, 0, -0.5))

    def _eval_order_defines(self):
        """Shader eval-order defines for the surface/clipping vector compute.

        The shader evaluates BOTH the function values (function order) and the
        CURVED geometry positions (mesh curve order) through the same
        N_DOFS_TRIG_VEC3-sized array. On a curved mesh the curve order can
        exceed the function order (e.g. mesh.Curve(5) with an order-2 field), so
        the array must be sized for the larger of the two — otherwise the curved
        de Casteljau overflows the array, the triangle corners come out
        degenerate, and the grid sampler counts zero arrows (nothing draws).
        """
        order = self.function_data.order
        md = getattr(self.function_data, "mesh_data", None)
        if md is not None:
            md_defines = md.get_shader_defines()
            order = max(order, md_defines.get("MAX_EVAL_ORDER_VEC3", 1),
                        md_defines.get("MAX_EVAL_ORDER", 1))
        defines = {"MAX_EVAL_ORDER": order, "MAX_EVAL_ORDER_VEC3": order}
        if self.function_data.cf.is_complex:
            defines["IS_COMPLEX"] = 1
        if self.scale_by_value:
            defines["SCALE_BY_VALUE"] = 1
        if self._region_visibility_enabled():
            defines["REGION_VISIBILITY"] = 1
        return defines

    def _region_visibility_enabled(self):
        return self.region_visibility is not None

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
        if self._region_visibility_enabled():
            bindings += self.region_visibility.get_bindings()
        return bindings

    def allocate_buffers(self):
        import os
        export_extra = BufferUsage.COPY_SRC if os.environ.get("WEBGPU_EXPORTING") else 0
        is_complex = self.function_data.cf.is_complex
        names = ["positions", "directions", "values"]
        if is_complex:
            names.append("directions_imag")
        for name in names:
            size = 4 * self.n_vectors if name == "values" else 3 * 4 * self.n_vectors
            self.__buffers[name] = create_buffer(
                size=size,
                usage=BufferUsage.VERTEX | BufferUsage.STORAGE | export_extra,
                label=name,
                reuse=self.__buffers.get(name, None),
            )
        if not is_complex:
            self.__buffers.pop("directions_imag", None)
            
    def _allocate_js_export_buffers(self):
        self.u_nvectors = buffer_from_array(
            np.array([0], dtype=np.uint32),
            label="n_vectors",
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,
            reuse=self.u_nvectors,
            use_cache=False,
        )
        self.n_vectors = 1
        self.allocate_buffers()
        self.positions_buffer = self.__buffers["positions"]
        self.directions_buffer = self.__buffers["directions"]
        self.directions_imag_buffer = self.__buffers.get("directions_imag", None)
        self.values_buffer = self.__buffers["values"]

    def compute_vectors(self):
        if not (self.symmetry and self.symmetry.n_copies > 1):
            if self.positions_buffer is None:
                self._allocate_js_export_buffers()
            return
        self.u_nvectors = buffer_from_array(
            np.array([0], dtype=np.uint32),
            label="n_vectors",
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,
            reuse=self.u_nvectors,
            use_cache=False,
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
        count_bindings = [
            BufferBinding(110, self.function_data.mesh_data.gpu_data),
            BufferBinding(21, self.u_nvectors, read_only=False),
            UniformBinding(31, self.u_grid_spacing),
        ]
        count_defines = {}
        if self._region_visibility_enabled():
            count_bindings += self.region_visibility.get_bindings()
            count_defines["REGION_VISIBILITY"] = 1
        run_compute_shader(
            read_shader_file(count_shader),
            count_bindings,
            n_work_groups,
            entry_point=self.compute_entry_point,
            defines=count_defines,
        )
        self.n_vectors = int(read_buffer(self.u_nvectors, np.uint32)[0])
        write_array_to_buffer(self.u_nvectors, np.array([0], dtype=np.uint32))
        self.allocate_buffers()
        defines = self._eval_order_defines()
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
        import os
        export_extra = BufferUsage.COPY_SRC if os.environ.get("WEBGPU_EXPORTING") else 0
        for name in ["positions", "directions", "directions_imag", "values"]:
            size = 4 * total if name == "values" else 3 * 4 * total
            self._expanded_buffers[name] = create_buffer(
                size=size,
                usage=BufferUsage.VERTEX | BufferUsage.STORAGE | export_extra,
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
        dirty = self.needs_update or self._last_data_ts != self.function_data._timestamp
        if not dirty:
            return
        self._last_data_ts = self.function_data._timestamp
        self.compute_vectors()
        is_complex = self.function_data.cf.is_complex
        max_val = self.function_data.maxval[0]
        if self.scale_by_value:
            # max arrow fills grid cell
            self._scale = 1.6 * self.grid_spacing / max(max_val, 1e-10) * self.user_scale
            self._scale_mode = 0
        else:
            # fixed size (directions are unit vectors)
            self._scale = 1.6 * self.grid_spacing * self.user_scale
            self._scale_mode = 2
        if self.gpu_objects.colormap.autoscale:
            self.gpu_objects.colormap.widen_range(
                self.function_data.minval[0],
                self.function_data.maxval[0],
                timestamp=options.timestamp,
            )
        if self.active:
            super().update(options)
            self.n_instances = self.n_vectors
            if self.symmetry and self.symmetry.n_copies > 1:
                self.n_instances *= self.symmetry.n_copies
            if self._complex_uniforms is not None:
                self._complex_uniforms.is_complex = 1 if is_complex else 0
                self._complex_uniforms.color_override = 1 if (is_complex and self.scale_by_value) else 0
                self._sync_complex_uniforms()

    def _sync_complex_uniforms(self):
        """Mirror the source-of-truth complex mode/phase (ComplexSettings,
        binding 56) onto the arrow render uniform (ShapeComplexUniform, binding
        11) the vertex shader reads to combine the real/imag directions."""
        if self._complex_uniforms is not None:
            self._complex_uniforms.mode = int(self._complex_settings.mode)
            self._complex_uniforms.phase = float(self._complex_settings.phase)
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
        self._sync_complex_uniforms()

    def set_phase(self, phase: float):
        """Set the phase angle for complex animate mode"""
        self._complex_settings.phase = phase
        self._sync_complex_uniforms()

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

    def _grid_size_export_interaction(self, buffer_registry):
        """Expose the arrow grid density as a slider in the JS-engine GUI.

        Writes ``box_size / grid_size`` to the compute pass's grid-spacing
        uniform (matching :meth:`set_grid_size`); that buffer is the recompute
        trigger, so dragging the slider regenerates the arrows live.
        """
        from webgpu.export.gui import gui_interaction, Slider, Write, Target

        u = getattr(self, "u_grid_spacing", None)
        if u is None:
            return []
        try:
            buf_id = buffer_registry.get_id(u)
        except KeyError:
            return []
        box = float(self.box_size)
        current = box / self.grid_spacing if self.grid_spacing else 20.0
        return [gui_interaction(
            self._vector_gui_label,
            [Slider(var="grid_size", name="Grid size",
                    default=float(round(current)), min=2.0, max=80.0, step=1.0)],
            [Write(targets=[Target(buf_id, offset=0, dtype="f32")],
                   expr=f"{box!r} / Math.max(grid_size, 1.0)", trigger="grid_size")],
        )]

    def get_export_interactions(self, options, buffer_registry):
        out = list(super().get_export_interactions(options, buffer_registry))
        out += self._grid_size_export_interaction(buffer_registry)
        if self.function_data.cf.is_complex:
            out += _complex_phase_export_interactions(
                [
                    (self._complex_settings.uniform, True),
                    (self._complex_uniforms, False),
                ],
                buffer_registry,
            )
        return out

class SurfaceVectors(VectorRenderer):
    _vector_gui_label = "Surface Vectors"

    def __init__(self, function_data: FunctionData, grid_size: float = 20, clipping: Clipping = None, colormap: Colormap = None, symmetry=None, vector_symmetry="polar", scale_by_value: bool = False):
        self.compute_shader_file = "ngsolve/surface_vectors.wgsl"
        self.compute_entry_point = "compute_surface_vectors"
        self.u_ntrigs = None
        super().__init__(function_data=function_data, grid_size=grid_size, clipping=clipping, colormap=colormap, symmetry=symmetry, vector_symmetry=vector_symmetry, scale_by_value=scale_by_value)
        self.gpu_objects.clipping = self.clipping
        self._vec_js = False
        self._vec_indirect = None
        self._vec_indirect_id = None

    def update(self, options):
        self.n_search_els = self.function_data.mesh_data.num_elements[ElType.TRIG]
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

    def get_export_compute_passes(self, options, buffer_registry):
        """Emit a JS-engine count-then-fill pass that recomputes the surface
        arrows GPU-side.
        """
        from webgpu.export.format import ExportComputePass
        from webgpu.utils import preprocess_shader_code

        if (self.symmetry and self.symmetry.n_copies > 1) or self.positions_buffer is None:
            self._vec_js = False
            return []

        defines = self._eval_order_defines()
        shader = preprocess_shader_code(
            read_shader_file(self.compute_shader_file), defines=defines
        )

        bindings = self.get_compute_bindings()
        binding_map = buffer_registry.register_bindings(bindings)

        counter_id = buffer_registry.get_id(self.u_nvectors)
        pos_id = buffer_registry.get_id(self.positions_buffer)
        siblings = [
            {"id": buffer_registry.get_id(self.directions_buffer), "element_size": 12},
            {"id": buffer_registry.get_id(self.values_buffer), "element_size": 4},
        ]
        if self.function_data.cf.is_complex and self.directions_imag_buffer is not None:
            siblings.append(
                {"id": buffer_registry.get_id(self.directions_imag_buffer), "element_size": 12}
            )

        if self._vec_indirect is None:
            self._vec_indirect = self.device.createBuffer(
                size=20,
                usage=BufferUsage.INDIRECT | BufferUsage.STORAGE | BufferUsage.COPY_DST,
                label="surf_vec_indirect",
            )
        indirect_id = buffer_registry.register_buffer(self._vec_indirect, "indirect")
        self._vec_indirect_id = indirect_id

        # Trigger this pass on the renderer's own grid-spacing uniform — the
        # buffer that actually changes when the arrow density changes (already
        # registered via the compute bindings, binding 31). It must NOT be the
        # clipping buffer: surface-vector positions are computed over the whole
        # surface and are independent of the clip plane (the plane only clips the
        # rendered arrows), so dragging the clip plane must not recompute them.
        # Mesh/field/deformation changes recompute through the host dirty path
        # (set_needs_update → notifyDirty), and the engine re-marks this trigger
        # itself after a count-then-fill resize.
        triggers = [buffer_registry.get_id(self.u_grid_spacing)]
        if self._region_visibility_enabled():
            triggers.append(buffer_registry.get_id(self.region_visibility.buffer))

        n_work_groups = min(self.n_search_els // 256 + 1, 1024)
        self._vec_js = True
        return [
            ExportComputePass(
                id=f"surf_vectors_{self._id}",
                shader=shader,
                bindings=binding_map,
                workgroups=[n_work_groups, 1, 1],
                triggers=triggers,
                reset_buffers=[counter_id],
                entry_point=self.compute_entry_point,
                count_then_fill={
                    "counter_id": counter_id,
                    "output_id": pos_id,
                    "element_size": 12,            # vec3<f32> position
                    "indirect_id": indirect_id,
                    "vertex_count": self.n_vertices,  # arrow index count per instance
                    "indexed": True,
                    "siblings": siblings,
                    # Pre-size the engine-owned arrow buffers to the count the
                    # Python pass already computed, so the engine's count-then-
                    # fill fills every arrow on the first frame instead of
                    # depending on an async-readback resize to converge. The
                    # surface arrows are static (toggled once, not dragged like
                    # the clipping plane), so they may only get one render frame
                    # — without this they can stick at the 1-element bootstrap
                    # size and draw a single arrow. The resize logic still runs
                    # as a safety net if the GPU count exceeds this hint.
                    "initial_count": int(getattr(self, "n_vectors", 0) or 0),
                },
            ),
        ]

    def get_export_descriptor(self, options, buffer_registry):
        result = super().get_export_descriptor(options, buffer_registry)
        if self._vec_js and self._vec_indirect_id is not None:
            # Arrow instance count comes from the JS-computed indirect buffer.
            result.draw_indirect = self._vec_indirect_id
        return result

class ClippingVectors(VectorRenderer):
    _vector_gui_label = "Clipping Vectors"

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
        # JS engine owns the clip-vector compute (countThenFill) once captured;
        # set in get_export_compute_passes. While owned, a clip-plane move needs
        # no Python recompute or pipeline rebuild (see _on_clipping_changed).
        self._vec_js = False
        self._vec_indirect = None
        self._vec_indirect_id = None
        self.clipping.callbacks.append(self._on_clipping_changed)

    def _region_visibility_enabled(self):
        return (
            self.region_visibility is not None
            and not self.function_data.cf.is_complex
        )

    def _on_clipping_changed(self):
        """Clip plane moved/rotated. In JS-compute mode the engine re-runs the
        vector compute from the shared clipping buffer's trigger on the next
        notifyDirty()+render() (issued by the host's scene.render()), so no
        pipeline invalidation is needed — that would force a full engine
        reinstall + shader recompile every drag tick. When Python still owns the
        compute (e.g. symmetry expansion), fall back to recomputing."""
        if self._vec_js:
            return
        self.set_needs_update()

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
        if not (self.symmetry and self.symmetry.n_copies > 1):
            if self.positions_buffer is None:
                self._allocate_js_export_buffers()
            return
        self.u_nvectors = buffer_from_array(
            np.array([0], dtype=np.uint32),
            label="n_vectors",
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,
            reuse=self.u_nvectors,
            use_cache=False,
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
        count_defines = {"MODE": 0, "MAX_EVAL_ORDER": self.function_data.order, "MAX_EVAL_ORDER_VEC3": 1}
        if self._region_visibility_enabled():
            count_bindings += self.region_visibility.get_bindings()
            count_defines["REGION_VISIBILITY"] = 1
        vec3_order = mesh_data.get_shader_defines()["MAX_EVAL_ORDER_VEC3"]
        run_compute_shader(
            read_shader_file(self.compute_shader_file),
            count_bindings,
            n_work_groups,
            entry_point=self.compute_entry_point,
            defines=count_defines,
        )
        count_approx = int(read_buffer(self.u_nvectors, np.uint32)[0])
        self.n_vectors = count_approx * 2 + 100
        write_array_to_buffer(self.u_nvectors, np.array([0], dtype=np.uint32))
        self.allocate_buffers()
        n_allocated = self.n_vectors

        # Eval pass: MODE=1, NEED_EVAL - needs function data + output buffers
        eval_defines = {"MODE": 1, "NEED_EVAL": 1, "MAX_EVAL_ORDER": self.function_data.order, "MAX_EVAL_ORDER_VEC3": vec3_order}
        if self.function_data.cf.is_complex:
            eval_defines["IS_COMPLEX"] = 1
        if self.scale_by_value:
            eval_defines["SCALE_BY_VALUE"] = 1
        if self._region_visibility_enabled():
            eval_defines["REGION_VISIBILITY"] = 1
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
        self.n_vectors = min(int(read_buffer(self.u_nvectors, np.uint32)[0]), n_allocated)
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

    def get_export_compute_passes(self, options, buffer_registry):
        """Emit a JS-engine countThenFill pass that recomputes the clip-plane
        arrows GPU-side whenever the shared clipping uniform changes.

        The single shader (compiled with NEED_EVAL) both counts vectors via an
        atomic and writes them, gated by arrayLength(&positions); the engine
        sizes positions and its sibling buffers (directions/values/imag) in
        lockstep from the counter, and draws the arrow glyphs with
        drawIndexedIndirect using the GPU-computed instance count.
        """
        from webgpu.export.format import ExportComputePass
        from webgpu.utils import preprocess_shader_code

        # Symmetry expansion is a second Python compute pass over the result;
        # keep that case on the Python path until it too is ported.
        if (self.symmetry and self.symmetry.n_copies > 1) or self.positions_buffer is None:
            self._vec_js = False
            return []

        mesh_data = self.function_data.mesh_data
        defines = {
            "MODE": 1,
            "NEED_EVAL": 1,
            "MAX_EVAL_ORDER": self.function_data.order,
            "MAX_EVAL_ORDER_VEC3": mesh_data.get_shader_defines()["MAX_EVAL_ORDER_VEC3"],
        }
        if self.function_data.cf.is_complex:
            defines["IS_COMPLEX"] = 1
        if self.scale_by_value:
            defines["SCALE_BY_VALUE"] = 1
        if self._region_visibility_enabled():
            defines["REGION_VISIBILITY"] = 1
        shader = preprocess_shader_code(
            read_shader_file(self.compute_shader_file), defines=defines
        )

        # Bind the *shared* clipping uniform (the buffer the GUI writes) so the
        # JS pass re-runs from its trigger on a plane move. get_compute_bindings
        # otherwise binds the per-renderer copy (__clipping), which the GUI never
        # touches directly.
        saved = self._ClippingVectors__clipping
        self._ClippingVectors__clipping = self.clipping
        try:
            bindings = self.get_compute_bindings()
            binding_map = buffer_registry.register_bindings(bindings)
        finally:
            self._ClippingVectors__clipping = saved

        counter_id = buffer_registry.get_id(self.u_nvectors)
        clip_id = buffer_registry.get_id(self.clipping.uniforms._buffer)
        pos_id = buffer_registry.get_id(self.positions_buffer)
        siblings = [
            {"id": buffer_registry.get_id(self.directions_buffer), "element_size": 12},
            {"id": buffer_registry.get_id(self.values_buffer), "element_size": 4},
        ]
        if self.function_data.cf.is_complex and self.directions_imag_buffer is not None:
            siblings.append(
                {"id": buffer_registry.get_id(self.directions_imag_buffer), "element_size": 12}
            )

        # Indirect draw buffer is JS-owned at runtime; create a Python
        # placeholder so the registry can assign a stable id.
        if self._vec_indirect is None:
            self._vec_indirect = self.device.createBuffer(
                size=20,
                usage=BufferUsage.INDIRECT | BufferUsage.STORAGE | BufferUsage.COPY_DST,
                label="vec_indirect",
            )
        indirect_id = buffer_registry.register_buffer(self._vec_indirect, "indirect")
        self._vec_indirect_id = indirect_id

        n_work_groups = min(self.n_search_els // 256 + 1, 1024)
        self._vec_js = True
        triggers = [clip_id, buffer_registry.get_id(self.u_grid_spacing)]
        if self._region_visibility_enabled():
            triggers.append(buffer_registry.get_id(self.region_visibility.buffer))
        return [
            ExportComputePass(
                id=f"clip_vectors_{self._id}",
                shader=shader,
                bindings=binding_map,
                workgroups=[n_work_groups, 1, 1],
                triggers=triggers,
                reset_buffers=[counter_id],
                entry_point=self.compute_entry_point,
                count_then_fill={
                    "counter_id": counter_id,
                    "output_id": pos_id,
                    "element_size": 12,            # vec3<f32> position
                    "indirect_id": indirect_id,
                    "vertex_count": self.n_vertices,  # arrow index count per instance
                    "indexed": True,
                    "siblings": siblings,
                },
            ),
        ]

    def get_export_descriptor(self, options, buffer_registry):
        result = super().get_export_descriptor(options, buffer_registry)
        if self._vec_js and self._vec_indirect_id is not None:
            # Arrow instance count comes from the JS-computed indirect buffer.
            result.draw_indirect = self._vec_indirect_id
        return result

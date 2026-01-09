import ngsolve as ngs
import numpy as np
import ctypes
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

from .cf import FunctionData, MeshData, Binding as FunctionBinding
from .mesh import Binding as MeshBinding
from .mesh import ElType


class SurfaceVectors(ShapeRenderer):
    def __init__(
        self,
        function_data: FunctionData,
        grid_size: float = 20,
        clipping: Clipping = None,
        colormap: Colormap = None,
    ):
        self.clipping = clipping or Clipping()

        self.function_data = function_data
        self.mesh = function_data.mesh_data

        bbox = self.mesh.get_bounding_box()
        self.box_size = np.linalg.norm(np.array(bbox[1]) - np.array(bbox[0]))
        self.set_grid_size(grid_size)

        cyl = generate_cylinder(8, 0.05, 0.5, bottom_face=True)
        cone = generate_cone(8, 0.2, 0.5, bottom_face=True)
        arrow = cyl + cone.move((0, 0, 0.5))

        super().__init__(arrow, None, None, colormap=colormap)
        # self.scale_mode = ShapeRenderer.SCALE_Z

    def get_bounding_box(self):
        return self.mesh.get_bounding_box()

    def get_compute_bindings(self):
        return []

    def set_grid_size(self, grid_size: float):
        self.grid_spacing = 1/grid_size * self.box_size
        if hasattr(self, "u_grid_spacing") and self.u_grid_spacing is not None:
            write_array_to_buffer(
                self.u_grid_spacing, np.array([self.grid_spacing], dtype=np.float32)
            )

    def compute_vectors(self):
        self.u_nvectors = buffer_from_array(
            np.array([0], dtype=np.uint32),
            label="n_vectors",
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,
        )

        mesh_buffers = self.mesh.get_buffers()
        func_buffers = self.function_data.get_buffers()
        n_trigs = self.mesh.num_elements[ElType.TRIG]
        self.u_ntrigs = uniform_from_array(np.array([n_trigs], dtype=np.uint32), label="n_trigs")

        positions = buffer_from_array(np.array([0], dtype=np.float32), label="positions")
        directions = buffer_from_array(np.array([0], dtype=np.float32), label="positions")
        values = buffer_from_array(np.array([0], dtype=np.float32), label="positions")

        bindings = [
            *self.gpu_objects.colormap.get_bindings(),
            BufferBinding(MeshBinding.VERTICES, mesh_buffers["vertices"]),
            BufferBinding(MeshBinding.TRIGS_INDEX, mesh_buffers[ElType.TRIG]),
            BufferBinding(22, positions, read_only=False),
            BufferBinding(23, directions, read_only=False),
            BufferBinding(25, values, read_only=False),
            BufferBinding(21, self.u_nvectors, read_only=False),
            UniformBinding(24, self.u_ntrigs),
            UniformBinding(31, self.u_grid_spacing),
            BufferBinding(MeshBinding.CURVATURE_VALUES_2D, mesh_buffers["curvature_2d"]),
            # BufferBinding(MeshBinding.DEFORMATION_VALUES, mesh_buffers["deformation_2d"]),
            UniformBinding(MeshBinding.DEFORMATION_SCALE, mesh_buffers["deformation_scale"]),
            BufferBinding(FunctionBinding.FUNCTION_VALUES_2D, func_buffers["data_2d"]),
        ]
        run_compute_shader(
            read_shader_file("ngsolve/surface_vectors.wgsl"),
            bindings,
            min(n_trigs // 256 + 1, 1024),
            entry_point="compute_surface_vectors",
            defines={
                "MODE": 0,
                "MAX_EVAL_ORDER": self.function_data.order,
                "MAX_EVAL_ORDER_VEC3": self.function_data.order,
            },
        )

        self.n_vectors = int(read_buffer(self.u_nvectors, np.uint32)[0])
        write_array_to_buffer(self.u_nvectors, np.array([0], dtype=np.uint32))
        buffers = {}
        for name in ["positions", "directions"]:
            buffers[name] = self.device.createBuffer(
                size=3 * 4 * self.n_vectors,
                usage=BufferUsage.STORAGE | BufferUsage.COPY_SRC,
                label=name,
            )
        buffers["values"] = self.device.createBuffer(
            size=4 * self.n_vectors,
            usage=BufferUsage.STORAGE | BufferUsage.COPY_SRC,
            label="values",
        )

        bindings = [
            *self.gpu_objects.colormap.get_bindings(),
            BufferBinding(MeshBinding.VERTICES, mesh_buffers["vertices"]),
            BufferBinding(MeshBinding.TRIGS_INDEX, mesh_buffers[ElType.TRIG]),
            BufferBinding(22, buffers["positions"], read_only=False),
            BufferBinding(23, buffers["directions"], read_only=False),
            BufferBinding(25, buffers["values"], read_only=False),
            BufferBinding(21, self.u_nvectors, read_only=False),
            UniformBinding(31, self.u_grid_spacing),
            BufferBinding(MeshBinding.CURVATURE_VALUES_2D, mesh_buffers["curvature_2d"]),
            BufferBinding(FunctionBinding.FUNCTION_VALUES_2D, func_buffers["data_2d"]),
            # BufferBinding(MeshBinding.DEFORMATION_VALUES, mesh_buffers["deformation_2d"]),
            UniformBinding(MeshBinding.DEFORMATION_SCALE, mesh_buffers["deformation_scale"]),
            UniformBinding(24, self.u_ntrigs),
        ]

        run_compute_shader(
            read_shader_file("ngsolve/surface_vectors.wgsl"),
            bindings,
            min(n_trigs // 256 + 1, 1024),
            entry_point="compute_surface_vectors",
            defines={
                "MODE": 1,
                "MAX_EVAL_ORDER": self.function_data.order,
                "MAX_EVAL_ORDER_VEC3": self.function_data.order,
            },
        )

        self.positions = read_buffer(buffers["positions"], np.float32).reshape(-1)
        self.values = read_buffer(buffers["values"], np.float32).reshape(-1)
        self.directions = read_buffer(buffers["directions"], np.float32).reshape(-1)

    def update(self, options):
        self.mesh.update(options)
        self.function_data.update(options)
        self.clipping.update(options)
        self.compute_vectors()
        super().update(options)
        return


class ClippingVectors(SurfaceVectors):
    def __init__(
        self,
        function_data: FunctionData,
        grid_size: float = 20,
        clipping: Clipping = None,
        colormap: Colormap = None,
    ):
        super().__init__(
            function_data=function_data, grid_size=grid_size, clipping=clipping
        , colormap=colormap)
        function_data.need_3d = True

        self.u_nvectors = None
        self.u_ntets = None

        self.__buffers = {}

        self.clipping.callbacks.append(self.set_needs_update)

        self.__clipping = Clipping()

    def get_compute_bindings(self, count=False):
        self.u_ntets = uniform_from_array(
            np.array([self.function_data.mesh_data.num_elements[ElType.TET]], dtype=np.uint32),
            label="n_tets",
            reuse=self.u_ntets,
        )
        if not hasattr(self, "u_grid_spacing") or self.u_grid_spacing is None:
            self.u_grid_spacing = uniform_from_array(
                np.array([self.grid_spacing], dtype=np.float32), label="grid_spacing"
            )

        buffers = self.function_data.get_buffers()

        bindings = [
            *self.gpu_objects.colormap.get_bindings(),
            *self.__clipping.get_bindings(),
            BufferBinding(MeshBinding.VERTICES, buffers["vertices"]),
            BufferBinding(MeshBinding.TET, buffers[ElType.TET]),
            BufferBinding(22, self.__buffers["positions"], read_only=False),
            BufferBinding(23, self.__buffers["directions"], read_only=False),
            BufferBinding(29, self.__buffers["values"], read_only=False),
            BufferBinding(21, self.u_nvectors, read_only=False),
            UniformBinding(24, self.u_ntets),
            UniformBinding(31, self.u_grid_spacing),
            BufferBinding(MeshBinding.DEFORMATION_3D_VALUES, buffers["deformation_3d"]),
            UniformBinding(MeshBinding.DEFORMATION_SCALE, buffers["deformation_scale"]),
            BufferBinding(FunctionBinding.FUNCTION_VALUES_3D, buffers["data_3d"]),
        ]

        return bindings

    def allocate_buffers(self):
        for name in ["positions", "directions", "values"]:
            size = 4 * self.n_vectors if name == "values" else 3 * 4 * self.n_vectors
            self.__buffers[name] = create_buffer(
                size=size,
                usage=BufferUsage.VERTEX | BufferUsage.STORAGE,
                label=name,
                reuse=self.__buffers.get(name, None),
            )

    def compute_vectors(self):
        self.u_nvectors = buffer_from_array(
            np.array([0], dtype=np.uint32),
            label="n_vectors",
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,
            reuse=self.u_nvectors,
        )

        self.n_vectors = 1
        self.allocate_buffers()

        n_tets = self.mesh.num_elements[ElType.TET]

        bindings = self.get_compute_bindings()

        n_work_groups = min(n_tets // 256 + 1, 1024)

        run_compute_shader(
            read_shader_file("ngsolve/clipping_vectors.wgsl"),
            bindings,
            n_work_groups,
            entry_point="compute_clipping_vectors",
            defines={
                "MODE": 0,
            },
        )

        self.n_vectors = int(read_buffer(self.u_nvectors, np.uint32)[0])
        write_array_to_buffer(self.u_nvectors, np.array([0], dtype=np.uint32))

        self.allocate_buffers()
        bindings = self.get_compute_bindings()

        run_compute_shader(
            read_shader_file("ngsolve/clipping_vectors.wgsl"),
            bindings,
            n_work_groups,
            entry_point="compute_clipping_vectors",
            defines={
                "MODE": 1,
            },
        )

        self.n_vectors = int(read_buffer(self.u_nvectors, np.uint32)[0])

        self.positions_buffer = self.__buffers["positions"]
        self.directions_buffer = self.__buffers["directions"]
        self.values_buffer = self.__buffers["values"]

    def update(self, options):
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

import ngsolve as ngs
import numpy as np
from webgpu.shapes import ShapeRenderer, generate_cone, generate_cylinder
from webgpu.utils import (
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
        mesh: MeshData,
        grid_size: float = 0.02,
    ):
        self.function_data = function_data
        self.mesh = mesh

        bbox = mesh.get_bounding_box()
        grid_size = np.linalg.norm(np.array(bbox[1]) - np.array(bbox[0])) * grid_size

        cyl = generate_cylinder(8, 0.05, 0.5, bottom_face=True)
        cone = generate_cone(8, 0.2, 0.5, bottom_face=True)
        arrow = cyl + cone.move((0, 0, 0.5))

        super().__init__(arrow, None, None)
        # self.scale_mode = ShapeRenderer.SCALE_Z

    def get_bounding_box(self):
        return self.mesh.get_bounding_box()

    def get_compute_bindings(self):
        return []

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
            *self.colormap.get_bindings(),
            BufferBinding(MeshBinding.VERTICES, mesh_buffers["vertices"]),
            BufferBinding(MeshBinding.TRIGS_INDEX, mesh_buffers[ElType.TRIG]),
            BufferBinding(22, positions, read_only=False),
            BufferBinding(23, directions, read_only=False),
            BufferBinding(25, values, read_only=False),
            BufferBinding(21, self.u_nvectors, read_only=False),
            UniformBinding(24, self.u_ntrigs),
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
            *self.colormap.get_bindings(),
            BufferBinding(MeshBinding.VERTICES, mesh_buffers["vertices"]),
            BufferBinding(MeshBinding.TRIGS_INDEX, mesh_buffers[ElType.TRIG]),
            BufferBinding(22, buffers["positions"], read_only=False),
            BufferBinding(23, buffers["directions"], read_only=False),
            BufferBinding(25, buffers["values"], read_only=False),
            BufferBinding(21, self.u_nvectors, read_only=False),
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
        self.colormap.update(options)
        self.compute_vectors()
        super().update(options)
        return

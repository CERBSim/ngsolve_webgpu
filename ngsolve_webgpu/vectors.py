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

from .cf import FunctionData, MeshData, Binding as FunctionBinding
from .mesh import Binding as MeshBinding
from .mesh import ElType

class VectorRenderer(ShapeRenderer):
    def __init__(
        self,
        function_data: FunctionData,
        grid_size: float = 20,
        clipping: Clipping = None,
        colormap: Colormap = None,
    ):
        self.u_nvectors = None
        self.clipping = clipping or Clipping()
        self.function_data = function_data
        mesh = function_data.mesh_data
        bbox = mesh.get_bounding_box()
        self.box_size = np.linalg.norm(np.array(bbox[1]) - np.array(bbox[0]))
        self.set_grid_size(grid_size)
        self.__buffers = {}
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
        self.u_nsearch = uniform_from_array(
            np.array([self.n_search_els], dtype=np.uint32), label="n_search_els",
            reuse=self.u_nsearch if hasattr(self, "u_nsearch") else None)
        return [
            BufferBinding(MeshBinding.MESH_DATA, buffers["mesh"]),
            BufferBinding(21, self.u_nvectors, read_only=False),
            BufferBinding(22, self.__buffers["positions"], read_only=False),
            BufferBinding(23, self.__buffers["directions"], read_only=False),
            UniformBinding(24, self.u_nsearch),
            BufferBinding(29, self.__buffers["values"], read_only=False),
            UniformBinding(31, self.u_grid_spacing)]

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
        assert hasattr(self, "n_search_els") and self.n_search_els > 0, "n_search_els should be set in the renderer"
        assert hasattr(self, "compute_shader_file") and hasattr(self, "compute_entry_point"), "compute_shader_file and compute_entry_point should be set in the renderer"
        n_work_groups = min(self.n_search_els // 256 + 1, 1024)
        run_compute_shader(
            read_shader_file(self.compute_shader_file),
            self.get_compute_bindings(),
            n_work_groups,
            entry_point=self.compute_entry_point,
            defines={
                "MODE": 0,
                "MAX_EVAL_ORDER": self.function_data.order,
                "MAX_EVAL_ORDER_VEC3": self.function_data.order,
            }
        )
        self.n_vectors = int(read_buffer(self.u_nvectors, np.uint32)[0])
        write_array_to_buffer(self.u_nvectors, np.array([0], dtype=np.uint32))
        self.allocate_buffers()
        run_compute_shader(
            read_shader_file(self.compute_shader_file),
            self.get_compute_bindings(),
            n_work_groups,
            entry_point=self.compute_entry_point,
            defines={
                "MODE": 1,
                "MAX_EVAL_ORDER": self.function_data.order,
                "MAX_EVAL_ORDER_VEC3": self.function_data.order,
            }
        )
        self.positions_buffer = self.__buffers["positions"]
        self.directions_buffer = self.__buffers["directions"]
        self.values_buffer = self.__buffers["values"]
        
    def update(self, options):
        self.function_data.update(options)
        self.compute_vectors()
        super().update(options)

class SurfaceVectors(VectorRenderer):
    def __init__(self, function_data: FunctionData, grid_size: float = 20, clipping: Clipping = None, colormap: Colormap = None):
        self.compute_shader_file = "ngsolve/surface_vectors.wgsl"
        self.compute_entry_point = "compute_surface_vectors"
        self.u_ntrigs = None
        super().__init__(function_data=function_data, grid_size=grid_size, clipping=clipping, colormap=colormap)
        
    def update(self, options):
        self.n_search_els = self.function_data.mesh_data.ngs_mesh.GetNE(ngs.BND)
        super().update(options)
        
    def get_compute_bindings(self):
        bindings = super().get_compute_bindings()
        buffers = self.function_data.get_buffers()
        return bindings + [
            BufferBinding(MeshBinding.DEFORMATION_VALUES, buffers["deformation_2d"]),
            UniformBinding(MeshBinding.DEFORMATION_SCALE, buffers["deformation_scale"]),
            BufferBinding(MeshBinding.CURVATURE_VALUES_2D, buffers["curvature_2d"]),
            BufferBinding(FunctionBinding.FUNCTION_VALUES_2D, buffers["data_2d"]),
        ]

class ClippingVectors(VectorRenderer):
    def __init__(
        self,
        function_data: FunctionData,
        grid_size: float = 20,
        clipping: Clipping = None,
        colormap: Colormap = None,
    ):
        self.compute_shader_file = "ngsolve/clipping_vectors.wgsl"
        self.compute_entry_point = "compute_clipping_vectors"
        self.u_ntets = None
        function_data.need_3d = True
        super().__init__(
            function_data=function_data, grid_size=grid_size, clipping=clipping,
            colormap=colormap)
        self.__clipping = Clipping()
        self.clipping.callbacks.append(self.set_needs_update)
        
    def get_compute_bindings(self):
        bindings = super().get_compute_bindings()
        buffers = self.function_data.get_buffers()
        return bindings + [
            *self.__clipping.get_bindings(),
            BufferBinding(MeshBinding.TET, buffers[ElType.TET]),
            BufferBinding(MeshBinding.DEFORMATION_3D_VALUES, buffers["deformation_3d"]),
            UniformBinding(MeshBinding.DEFORMATION_SCALE, buffers["deformation_scale"]),
            BufferBinding(FunctionBinding.FUNCTION_VALUES_3D, buffers["data_3d"]),
        ]

    def update(self, options):
        self.n_search_els = self.function_data.mesh_data.ngs_mesh.GetNE(ngs.VOL)
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

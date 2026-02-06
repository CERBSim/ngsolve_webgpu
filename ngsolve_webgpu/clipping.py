from webgpu import create_bind_group, read_shader_file
from webgpu.utils import buffer_from_array, uniform_from_array, write_array_to_buffer
from webgpu.clipping import Clipping
from webgpu.colormap import Colormap
from webgpu.renderer import Renderer, RenderOptions
from webgpu.utils import BufferBinding, UniformBinding, ReadBuffer, run_compute_shader, read_buffer
import ctypes

from webgpu.webgpu_api import *

import numpy as np

from .cf import FunctionData
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
        return read_shader_file("ngsolve/mesh.wgsl")


class ClippingCF(Renderer):
    vertex_entry_point = "vertex_clipping"
    fragment_entry_point = "fragment_clipping"
    compute_shader = "ngsolve/clipping/compute.wgsl"
    select_entry_point = "fragment_select_no_clipping"
    n_vertices = 3
    subdivision = 0

    def __init__(
        self, data: FunctionData, clipping: Clipping = None, colormap: Colormap = None, component=-1
    ):
        super().__init__()
        self._clipping = clipping or Clipping()
        self.clipping = Clipping()
        self.gpu_objects.colormap = colormap or Colormap()
        self._clipping.callbacks.append(self.set_needs_update)
        self.data = data
        self.component = component if data.cf.dim > 1 else 0
        self.data.need_3d = True
        if self.data.mesh_data.deformation_data is not None:
            self.data.mesh_data.deformation_data.need_3d = True
        self.options = None
        self.component_buffer = None
        self.cut_trigs_counter = None
        self.cut_trigs = None
        self.trig_counter = None
        self.only_count = None
        self.n_tets = None

    @property
    def colormap(self):
        return self.gpu_objects.colormap

    @colormap.setter
    def colormap(self, value: Colormap):
        self.gpu_objects.colormap = value

    def update(self, options: RenderOptions):
        self.component_buffer = uniform_from_array(
            np.array([self.component], np.int32),
            label="component_buffer",
            reuse=self.component_buffer,
        )
        self.data.update(options)
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
        self.build_clip_plane()

    def get_bounding_box(self):
        return self.data.get_bounding_box()

    def get_shader_code(self):
        return read_shader_file("ngsolve/clipping/render.wgsl")

    def set_component(self, component: int):
        self.component = component
        self.component_buffer = uniform_from_array(np.array([self.component], np.int32))
        self.set_needs_update()

    def get_bindings(self, compute=False, count: bool = False):
        bindings = [
            BufferBinding(MeshBinding.VERTICES, self._buffers["vertices"]),
            UniformBinding(22, self.n_tets),
            UniformBinding(23, self.only_count),
            BufferBinding(MeshBinding.TET, self._buffers[ElType.TET]),
            BufferBinding(13, self._buffers["data_3d"]),
            UniformBinding(17, self._buffers["deformation_scale"]),
            BufferBinding(18, self._buffers["deformation_3d"]),
            UniformBinding(CFBinding.COMPONENT, self.component_buffer),
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
            ]
        else:
            bindings += [
                *self.gpu_objects.colormap.get_bindings(),
                BufferBinding(24, self.cut_trigs),
            ]
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
                shader_code, self.get_bindings(compute=True, count=count), 1024, "build_clip_plane"
            )
            if count:
                self.n_instances = int(read_buffer(self.trig_counter, np.uint32)[0])

    def render(self, options: RenderOptions):
        if bytes(self._clipping.uniforms) != bytes(self.clipping.uniforms):
            self.set_needs_update()
        super().render(options)

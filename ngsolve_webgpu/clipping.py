from webgpu import create_bind_group, read_shader_file
from webgpu.utils import buffer_from_array, uniform_from_array
from webgpu.clipping import Clipping
from webgpu.colormap import Colormap
from webgpu.render_object import RenderObject
from webgpu.utils import BufferBinding, UniformBinding, ReadBuffer, run_compute_shader, read_buffer

from webgpu.webgpu_api import *

import numpy as np

from .cf import FunctionData

from .mesh import Mesh3dElementsRenderObject, ElType
from .mesh import Binding as MeshBinding


class VolumeCF(Mesh3dElementsRenderObject):
    fragment_entry_point: str = "cf_fragment_main"

    def __init__(self, data: FunctionData):
        super().__init__(data=data.mesh_data)
        self.data = data
        self.data.need_3d = True
        self.colormap = Colormap()

    def update(self, timestamp):
        if self._timestamp == timestamp:
            return
        self.colormap.options = self.options
        self.colormap.update(timestamp)
        super().update(timestamp)

    def get_bindings(self):
        return super().get_bindings() + [
            BufferBinding(10, self._buffers["data_3d"]),
            *self.colormap.get_bindings(),
        ]

    def get_shader_code(self):
        return read_shader_file("ngsolve/mesh.wgsl")


class ClippingCF(RenderObject):
    compute_shader = "ngsolve/clipping/compute.wgsl"
    n_vertices = 3
    subdivision = 0

    def __init__(self, data: FunctionData):
        super().__init__()
        self.clipping = Clipping()
        self.colormap = Colormap()
        self.clipping.callbacks.append(self.build_clip_plane)
        self.data = data
        self.data.need_3d = True

    def update(self, timestamp):
        if timestamp == self._timestamp:
            return
        self._timestamp = timestamp
        self.data.update(timestamp)
        self.clipping.update(timestamp)
        self.colormap.options = self.options
        self.colormap.update(timestamp)
        self._buffers = self.data.get_buffers()
        self.build_clip_plane()

    def get_bounding_box(self):
        return self.data.get_bounding_box()

    def get_shader_code(self):
        return read_shader_file("ngsolve/clipping/render.wgsl")

    def get_bindings(self, compute=False):
        bindings = [
            *self.options.camera.get_bindings(),
            BufferBinding(MeshBinding.VERTICES, self._buffers["vertices"]),
            UniformBinding(22, self.n_tets),
            UniformBinding(23, self.only_count),
            BufferBinding(MeshBinding.TET, self._buffers[ElType.TET]),
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
            ]
        else:
            bindings += [
                *self.colormap.get_bindings(),
                *self.options.light.get_bindings(),
                BufferBinding(24, self.cut_trigs),
            ]
        return bindings

    def build_clip_plane(self):
        for count in [True, False]:
            ntets = self.data.mesh_data.num_elements[ElType.TET] * 4**self.subdivision
            self.trig_counter = buffer_from_array(
                np.array([0], dtype=np.uint32),
                usage=BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,
            )
            self.n_tets = uniform_from_array(np.array([ntets], dtype=np.uint32))
            self.only_count = uniform_from_array(np.array([count], dtype=np.uint32))
            if count:
                self.cut_trigs = buffer_from_array(np.array([0.0] * 64, dtype=np.float32))
            else:
                self.cut_trigs = self.device.createBuffer(
                    size=64 * self.n_instances, usage=BufferUsage.STORAGE
                )

            shader_code = read_shader_file(self.compute_shader)
            run_compute_shader(
                shader_code, self.get_bindings(compute=True), 1024, "build_clip_plane"
            )
            if count:
                self.n_instances = int(read_buffer(self.trig_counter, np.uint32)[0])
        self.create_render_pipeline()

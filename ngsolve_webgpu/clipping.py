

from webgpu import create_bind_group, read_shader_file
from webgpu.utils import buffer_from_array, uniform_from_array
from webgpu.clipping import Clipping
from webgpu.colormap import Colormap
from webgpu.render_object import RenderObject
from webgpu.utils import BufferBinding, UniformBinding, ReadBuffer

from webgpu.webgpu_api import *

import numpy as np

from .cf import FunctionData

from .mesh import Mesh3dElementsRenderObject, ElType
from .mesh import Binding as MeshBinding

class VolumeCF(Mesh3dElementsRenderObject):
    fragment_entry_point: str = "cf_fragment_main"
    def __init__(self, data: FunctionData):
        super().__init__(data = data.mesh_data)
        self.data = data
        self.data.need_3d = True
        self.colormap = Colormap()

    def update(self, data: FunctionData | None = None):
        self.colormap.options = self.options
        self.colormap.update()
        super().update(data)

    def get_bindings(self):
        return super().get_bindings() + [
            BufferBinding(10, self._buffers["data_3d"]),
            *self.colormap.get_bindings()]
            

    def get_shader_code(self):
        eval_code = read_shader_file("eval.wgsl", __file__)
        return super().get_shader_code() + self.colormap.get_shader_code() + eval_code

class ClippingCF(RenderObject):
    n_vertices = 3
    def __init__(self, data: FunctionData, clipping=None, colormap=None):
        super().__init__(label="ClippingCF")
        self.clipping = clipping or Clipping()
        self.colormap = colormap or Colormap()
        self.clipping.callbacks.append(self.update)
        self.data = data
        self.data.need_3d = True

    def redraw(self, timestamp: float | None = None):
        timestamp = self.data.redraw(timestamp)
        super().redraw(timestamp)

    def update(self, data: FunctionData | None = None):
        if data is not None:
            self.data = data
        self.clipping.update()
        self.colormap.options = self.options
        self.colormap.update()
        self._buffers = self.data.get_buffers(self.device)
        self.build_clip_plane()
        self.create_render_pipeline()

    def get_shader_code(self, compute=False):
        shader_code = ""
        shader_code += self.clipping.get_shader_code()
        shader_code += self.options.camera.get_shader_code()
        shader_code += read_shader_file("clipping/common.wgsl", __file__)
        if compute:
            shader_code += read_shader_file("clipping/compute.wgsl", __file__)
        else:
            shader_code += read_shader_file("clipping/render.wgsl", __file__)
            shader_code += read_shader_file("eval/common.wgsl", __file__)
            shader_code += read_shader_file("eval/tet.wgsl", __file__)
            shader_code += self.colormap.get_shader_code()
            shader_code += self.options.light.get_shader_code()
        return shader_code

    def get_bindings(self, compute=False):
        bindings = [
            *self.options.camera.get_bindings(),
            BufferBinding(MeshBinding.VERTICES, self._buffers["vertices"]),
            UniformBinding(22, self.n_tets),
            UniformBinding(23, self.only_count),
            BufferBinding(MeshBinding.TET, self._buffers[ElType.TET.name]),
            BufferBinding(13, self._buffers["data_3d"]),
            *self.clipping.get_bindings()]
        if compute:
            bindings += [BufferBinding(21, self.trig_counter,
                                       read_only=False,
                                       visibility=ShaderStage.COMPUTE),
                         BufferBinding(24, self.cut_trigs, read_only=False),
                         ]
        else:
            bindings += [*self.colormap.get_bindings(),
                         *self.options.light.get_bindings(),
                         BufferBinding(24, self.cut_trigs)
                         ]
        return bindings

    def build_clip_plane(self):
        for count in [True, False]:
            encoder = self.device.createCommandEncoder("build_clip_plane")
            ntets = self.data.mesh_data.num_els[ElType.TET.name]
            self.trig_counter = buffer_from_array(
                np.array([0], dtype=np.uint32),
                usage=BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC)
            self.n_tets = uniform_from_array(np.array([ntets], dtype=np.uint32))
            self.only_count = uniform_from_array(np.array([count], dtype=np.uint32))
            if count:
                self.cut_trigs = buffer_from_array(np.array([0.] * 64, dtype=np.float32))
            else:
                self.cut_trigs = self.device.createBuffer(
                    size=64 * self.n_instances,
                    usage=BufferUsage.STORAGE)
            layout, group = create_bind_group(
                self.device,
                self.get_bindings(compute=True),
                label="create_clip_plane")
            shader_module = self.device.createShaderModule(
                code=self.get_shader_code(compute=True))
            pipeline = self.device.createComputePipeline(
                self.device.createPipelineLayout([layout]),
                label="create_clip_plane",
                compute=ComputeState(
                    module=shader_module,
                    entryPoint="main"))
            compute_pass = encoder.beginComputePass(label="build_clip_plane")
            compute_pass.setPipeline(pipeline)
            compute_pass.setBindGroup(0, group)
            compute_pass.dispatchWorkgroups(1024)
            compute_pass.end()
            if count:
                read = ReadBuffer(self.trig_counter, encoder)
            self.device.queue.submit([encoder.finish()])
            if count:
                array = read.get_array(dtype=np.uint32)
                self.n_instances = int(array[0])

            

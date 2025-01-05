
import webgpu
from webgpu.gpu import RenderObject
from webgpu.utils import (
    BufferBinding,
    ShaderStage,
    TextureBinding,
    create_bind_group,
    decode_bytes,
    encode_bytes,
    read_shader_file
)

from webgpu.webgpu_api import (
    CommandEncoder,
    BufferUsage,
    VertexState,
    FragmentState,
    ColorTargetState,
    PrimitiveState,
    PrimitiveTopology,
    DepthStencilState,
    CompareFunction,
    TextureFormat,
    RenderPassColorAttachment,
    Color,
    RenderPassDepthStencilAttachment,
    ComputeState,
    MapMode,
    BindGroupLayoutEntry,
    BindGroupEntry,
)

import numpy as np


class Binding:
    VERTICES = 90
    NORMALS = 91
    INDICES = 92

class IsoSurfaceRenderObject(RenderObject):
    def __init__(self, gpu, cf, mesh, label):
        super().__init__(gpu, label=label)
        self.cf = cf
        self.mesh = mesh
        self.update()

    def count_cut_trigs(self):
        encoder = self.gpu.device.createCommandEncoder(
            label="count_iso_trigs")
        # binding -> counter i32
        print("mesh tets = ", self.mesh.ne)
        counter_buffer = self.device.createBuffer(
            size=4,
            usage=BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST,
        )
        result_buffer = self.device.createBuffer(
            size=4,
            usage=BufferUsage.MAP_READ | BufferUsage.COPY_DST,
        )
        self.device.queue.writeBuffer(counter_buffer,
                                      0, np.array([0], dtype=np.int32).tobytes(), 0, 4)
        binding = BufferBinding(80,
                                counter_buffer,
                                read_only=False,
                                visibility=ShaderStage.COMPUTE)
        layout = self.device.createBindGroupLayout(
            entries=[BindGroupLayoutEntry(**binding.layout)
            ], label="count_iso_trigs")
        group = self.device.createBindGroup(label="count_iso_trigs",
                                            layout=layout,
                                            entries=[BindGroupEntry(**binding.binding)])
        print("layout = ", layout)
        shader_code = ""
        shader_code += read_shader_file("isosurface.wgsl", __file__)
        shader_module = self.device.createShaderModule(shader_code)
        pipeline = self.device.createComputePipeline(
            self.device.createPipelineLayout([layout], self.label),
            label="count_iso_trigs",
            compute=ComputeState(
                module=shader_module,
                entryPoint="count_iso_triangles")
            )
        # self.gpu.begin_render_pass(encoder, label="count_iso_trigs")
        compute_pass = encoder.beginComputePass(label="count_iso_trigs")
        compute_pass.setBindGroup(0, group)
        compute_pass.setPipeline(pipeline)
        compute_pass.dispatchWorkgroups(self.mesh.ne)
        compute_pass.end()

        counter_buffer.handle = counter_buffer
        result_buffer.handle = result_buffer
        encoder.copyBufferToBuffer(counter_buffer, 0, result_buffer, 0, 4)
        self.device.queue.submit([encoder.finish()])
        result = result_buffer.mapAsync(MapMode.READ)
        def print_result(r):
            data = result_buffer.getMappedRange(0, 4)
            # b = np.frombuffer(data.to_bytes(), dtype=np.int32)
            b = np.frombuffer(memoryview(data.to_py()), dtype=np.int32)
            print("count = ", b)
            result_buffer.unmap()
        result.then(print_result)

        
        
        

    def update(self):
        self.count_cut_trigs()

    def render(self, encoder):
        pass


def render_isosurface(cf, mesh, name="isosurface"):
    from webgpu.jupyter import gpu
    import js
    import pyodide.ffi
    iso = IsoSurfaceRenderObject(gpu, cf, mesh, name)
    def render_function(t):
        gpu.update_uniforms()
        encoder = gpu.device.createCommandEncoder()
        iso.render(encoder)
        gpu.device.queue.submit([encoder.finish()])
    render_function = pyodide.ffi.create_proxy(render_function)
    gpu.input_handler.render_function = render_function
    js.requestAnimationFrame(render_function)

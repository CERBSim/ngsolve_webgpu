import webgpu
from webgpu.gpu import RenderObject
from webgpu.utils import (
    BufferBinding,
    ShaderStage,
    TextureBinding,
    create_bind_group,
    decode_bytes,
    encode_bytes,
    read_shader_file,
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
        self.n_cut_trigs = None
        self.cut_trigs_set = False
        self.update()

    def count_cut_trigs(self):
        encoder = self.gpu.device.createCommandEncoder(label="count_iso_trigs")
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
        self.device.queue.writeBuffer(
            counter_buffer, 0, np.array([0], dtype=np.int32).tobytes(), 0, 4
        )
        binding = BufferBinding(
            80, counter_buffer, read_only=False, visibility=ShaderStage.COMPUTE
        )
        layout = self.device.createBindGroupLayout(
            entries=[BindGroupLayoutEntry(**binding.layout)], label="count_iso_trigs"
        )
        group = self.device.createBindGroup(
            label="count_iso_trigs",
            layout=layout,
            entries=[BindGroupEntry(**binding.binding)],
        )
        print("layout = ", layout)
        shader_code = ""
        # shader_code += read_shader_file("mesh.wgsl", __file__)
        shader_code += read_shader_file("isosurface.wgsl", __file__)
        shader_module = self.device.createShaderModule(shader_code)
        pipeline = self.device.createComputePipeline(
            self.device.createPipelineLayout([layout], self.label),
            label="count_iso_trigs",
            compute=ComputeState(
                module=shader_module, entryPoint="count_iso_triangles"
            ),
        )
        # self.gpu.begin_render_pass(encoder, label="count_iso_trigs")
        compute_pass = encoder.beginComputePass(label="count_iso_trigs")
        compute_pass.setBindGroup(0, group)
        compute_pass.setPipeline(pipeline)
        compute_pass.dispatchWorkgroups(self.mesh.ne)
        compute_pass.end()

        encoder.copyBufferToBuffer(counter_buffer, 0, result_buffer, 0, 4)
        self.device.queue.submit([encoder.finish()])
        import pyodide

        task = pyodide.webloop.asyncio.get_running_loop().run_until_complete(
            result_buffer.mapAsync(MapMode.READ, 0, 4)
        )

        def read_buffer(task):
            print("in read buffer")
            data = result_buffer.getMappedRange(0, 4)
            print("data = ", data)
            b = np.frombuffer(memoryview(data.to_py()), dtype=np.int32)
            print("b = ", b)
            self.n_cut_trigs = b[0]
            result_buffer.unmap()
            self.write_cut_trigs()

        task.then(read_buffer)

    def write_cut_trigs(self):
        pass

    def update(self):
        self.count_cut_trigs()

    def render(self, encoder):
        if self.cut_trigs_set is False:
            return


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

import webgpu
from webgpu.gpu import RenderObject
from webgpu.utils import (
    BufferBinding,
    UniformBinding,
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

    def create_cut_trigs(self, count=None):
        import ngsolve as ngs
        compute_encoder = self.gpu.device.createCommandEncoder(label="count_iso_trigs")
        # binding -> counter i32
        print("mesh tets = ", self.mesh.ne)
        print("create cut trigs count=", count)
        counter_buffer = self.device.createBuffer(
            size=4,
            usage=BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST,
            label="counter"
        )
        mesh_pts = self.mesh.MapToAllElements(ngs.IntegrationRule([(0,0,0),
                                                                   (1,0,0),
                                                                   (0,1,0),
                                                                   (0,0,1)]),
                                              self.mesh.Materials(".*"))
        func_values = self.cf(mesh_pts).flatten()
        function_value_buffer = self.device.createBuffer(
            size=len(func_values) * func_values.itemsize,
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST,
            label="function"
        )
        self.device.queue.writeBuffer(
            function_value_buffer, 0, func_values.tobytes())
        vertices = ngs.CF((ngs.x,ngs.y,ngs.z))(mesh_pts).flatten()
        vertex_buffer = self.device.createBuffer(
            size=len(vertices) * vertices.itemsize,
            label="vertex",
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST)
        self.device.queue.writeBuffer(
            vertex_buffer, 0, vertices.tobytes())
        result_buffer = self.device.createBuffer(
            size=4, label="result",
            usage=BufferUsage.MAP_READ | BufferUsage.COPY_DST,
        )
        only_count = self.device.createBuffer(
            label="only_count",
            size=4, usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST)
        self.device.queue.writeBuffer(only_count,
                                      0, np.array([1 if count is None else 0], dtype=np.uint32).tobytes(), 0, 4)
        self.device.queue.writeBuffer(
            counter_buffer, 0, np.array([0], dtype=np.uint32).tobytes(), 0, 4
        )
        bindings = [*self.gpu.u_view.get_bindings(),
                    *self.gpu.u_font.get_bindings(),
                    *self.gpu.u_mesh.get_bindings(),
                    BufferBinding(80, counter_buffer,
                          read_only=False,
                          visibility=ShaderStage.COMPUTE),
            BufferBinding(83, function_value_buffer,
                          visibility=ShaderStage.ALL),
            BufferBinding(12, vertex_buffer,
                          visibility=ShaderStage.ALL),
            UniformBinding(81, only_count)]
        cut_buffer_size = 64 * (count if count is not None else 1)
        print("cut_buffer_size = ", cut_buffer_size)
        cut_trigs_buffer = self.device.createBuffer(
            label="cut trigs",
            size=cut_buffer_size,
            usage=BufferUsage.STORAGE | BufferUsage.COPY_SRC)
        cut_trigs_binding = BufferBinding(82, cut_trigs_buffer,
                                          read_only=False,
                                          visibility=ShaderStage.COMPUTE)

        layout, group = create_bind_group(
            self.device,
            bindings + [cut_trigs_binding])
        shader_code = ""
        compute_shader_code = read_shader_file("compute_isosurface.wgsl", __file__)
        # shader_code += read_shader_file("mesh.wgsl", __file__)
        shader_code += read_shader_file("isosurface.wgsl", __file__)
        shader_module = self.device.createShaderModule(compute_shader_code + shader_code)
        pipeline = self.device.createComputePipeline(
            self.device.createPipelineLayout([layout], self.label),
            label="count_iso_trigs",
            compute=ComputeState(
                module=shader_module, entryPoint="create_iso_triangles"
            ),
        )
        # self.gpu.begin_render_pass(encoder, label="count_iso_trigs")
        compute_pass = compute_encoder.beginComputePass(label="count_iso_trigs")
        compute_pass.setBindGroup(0, group)
        compute_pass.setPipeline(pipeline)
        compute_pass.dispatchWorkgroups(self.mesh.ne)
        compute_pass.end()

        compute_encoder.copyBufferToBuffer(counter_buffer, 0, result_buffer, 0, 4)
        self.device.queue.submit([compute_encoder.finish()])
        import pyodide

        task = pyodide.webloop.asyncio.get_running_loop().run_until_complete(
            result_buffer.mapAsync(MapMode.READ, 0, 4)
        )

        def read_buffer(task):
            data = result_buffer.getMappedRange(0, 4)
            b = np.frombuffer(memoryview(data.to_py()), dtype=np.int32)
            print("b = ", b)
            self.n_cut_trigs = b[0]
            result_buffer.unmap()
            self.create_cut_trigs(b[0])

        if self.n_cut_trigs is None:
            task.then(read_buffer)
        else:
            print("Create render pipeline with cut trigs set = ", self.n_cut_trigs)
            # encoder = self.gpu.device.createCommandEncoder()
            render_cut_trigs_binding = BufferBinding(82, cut_trigs_buffer,
                                                     visibility=ShaderStage.VERTEX)
            render_layout, self._bind_group = create_bind_group(
                self.device,
                bindings + [render_cut_trigs_binding])
            self.n_cut_trigs = count
            render_shader_code = read_shader_file("render_isosurface.wgsl", __file__)
            render_shader_module = self.device.createShaderModule(render_shader_code + shader_code)
            self._pipeline = self.device.createRenderPipeline(
                self.device.createPipelineLayout([render_layout], self.label),
                vertex=VertexState(module=render_shader_module,
                                   entryPoint="vertexIsoSurface"),
                fragment=FragmentState(module=render_shader_module,
                                       entryPoint="fragmentIsoSurface",
                                       targets=[
                                           ColorTargetState(format=self.gpu.format)]),
                primitive=PrimitiveState(
                    topology=PrimitiveTopology.triangle_list),
                depthStencil = DepthStencilState(
                    format=TextureFormat.depth24plus,
                    depthWriteEnabled=True,
                    depthCompare=CompareFunction.less,
                    # shift trigs behind to ensure that edges are rendered properly
                    depthBias=1,
                    depthBiasSlopeScale=1.0))

            self.cut_trigs_set = True
            # self.render(encoder)
            # self.gpu.update_uniforms()
            # self.gpu.device.queue.submit([encoder.finish()])

    def update(self):
        self.create_cut_trigs()

    def render(self, encoder):
        if self.cut_trigs_set is False:
            return
        print("render with cut trigs set = ", self.n_cut_trigs)
        render_pass = self.gpu.begin_render_pass(encoder, label=self.label)
        render_pass.setBindGroup(0, self._bind_group)
        render_pass.setPipeline(self._pipeline)
        # render_pass.draw(3, self.n_cut_trigs)
        render_pass.draw(3, 1)
        render_pass.end()



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
    # js.requestAnimationFrame(render_function)

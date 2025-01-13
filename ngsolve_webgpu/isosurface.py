import numpy as np
import webgpu
from webgpu.gpu import RenderObject
from webgpu.utils import (
    BufferBinding,
    ShaderStage,
    TextureBinding,
    UniformBinding,
    create_bind_group,
    decode_bytes,
    encode_bytes,
    read_shader_file,
)
from webgpu.colormap import Colormap
from webgpu.webgpu_api import (
    BindGroupEntry,
    BindGroupLayoutEntry,
    BufferUsage,
    Color,
    ColorTargetState,
    CommandEncoder,
    CompareFunction,
    ComputeState,
    DepthStencilState,
    FragmentState,
    MapMode,
    PrimitiveState,
    PrimitiveTopology,
    RenderPassColorAttachment,
    RenderPassDepthStencilAttachment,
    TextureFormat,
    VertexState,
)


class Binding:
    VERTICES = 90
    NORMALS = 91
    INDICES = 92


class IsoSurfaceRenderObject(RenderObject):
    def __init__(self, gpu, levelset, function, mesh, label):
        super().__init__(gpu, label=label)
        self.levelset = levelset
        self.function = function
        self.mesh = mesh
        self.n_cut_trigs = None
        self.cut_trigs_set = False
        self.task = None
        self.update()

    def count_cut_trigs(self):
        count = None
        import ngsolve as ngs

        compute_encoder = self.gpu.device.createCommandEncoder(label="count_iso_trigs")
        # binding -> counter i32
        print("mesh tets = ", self.mesh.ne)
        self.counter_buffer = self.device.createBuffer(
            size=4,
            usage=BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST,
            label="counter",
        )
        self.mesh_pts = self.mesh.MapToAllElements(
            ngs.IntegrationRule([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),
            self.mesh.Materials(".*"),
        )
        func_values = np.array(self.levelset(self.mesh_pts).flatten(), dtype=np.float32)
        self.function_value_buffer = self.device.createBuffer(
            size=len(func_values) * func_values.itemsize,
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST,
            label="function",
        )
        self.device.queue.writeBuffer(self.function_value_buffer, 0, func_values.tobytes())
        func_values = np.array(self.levelset(self.mesh_pts).flatten(),
                               dtype=np.float32)
        vertices = np.array(
            ngs.CF((ngs.x, ngs.y, ngs.z))(self.mesh_pts), dtype=np.float32
        )
        self.pmin = np.min(vertices, axis=0)
        self.pmax = np.max(vertices, axis=0)
        vertices = vertices.flatten()
        self.vertex_buffer = self.device.createBuffer(
            size=len(vertices) * vertices.itemsize,
            label="vertex",
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST,
        )
        # just a dummy here, needed in create
        cut_trigs_buffer = self.device.createBuffer(
            label="cut trigs " + str(self.n_cut_trigs),
            size=64,
            usage=BufferUsage.STORAGE,
        )
        cut_trigs_binding = BufferBinding(
            82, cut_trigs_buffer, read_only=False, visibility=ShaderStage.COMPUTE
        )
        self.device.queue.writeBuffer(self.vertex_buffer, 0, vertices.tobytes())
        self.result_buffer = self.device.createBuffer(
            size=4,
            label="result",
            usage=BufferUsage.MAP_READ | BufferUsage.COPY_DST,
        )
        self.only_count = self.device.createBuffer(
            label="only_count", size=4, usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST
        )
        self.device.queue.writeBuffer(
            self.only_count, 0, np.uint32(1).tobytes(), 0, 4,
        )
        self.device.queue.writeBuffer(
            self.counter_buffer, 0, np.array([0], dtype=np.uint32).tobytes(), 0, 4
        )
        bindings = [
            *self.gpu.camera.get_bindings(),
            *self.gpu.u_mesh.get_bindings(),
            BufferBinding(
                80, self.counter_buffer, read_only=False, visibility=ShaderStage.COMPUTE
            ),
            BufferBinding(83, self.function_value_buffer,
                          visibility=ShaderStage.ALL),
            BufferBinding(12, self.vertex_buffer, visibility=ShaderStage.ALL),
            UniformBinding(81, self.only_count),
            cut_trigs_binding
        ]

        layout, group = create_bind_group(
            self.device, bindings, "count_cut_trigs"
        )
        shader_code = ""
        compute_shader_code = read_shader_file("compute_isosurface.wgsl", __file__)
        shader_code += read_shader_file("isosurface.wgsl", __file__)
        shader_module = self.device.createShaderModule(
            compute_shader_code + shader_code
        )
        pipeline = self.device.createComputePipeline(
            self.device.createPipelineLayout([layout], self.label + " count"),
            label="count_iso_trigs",
            compute=ComputeState(
                module=shader_module,
                entryPoint="create_iso_triangles"
            ),
        )
        compute_pass = compute_encoder.beginComputePass(label="count_iso_trigs")
        compute_pass.setBindGroup(0, group)
        compute_pass.setPipeline(pipeline)
        compute_pass.dispatchWorkgroups(self.mesh.ne)
        compute_pass.end()

        compute_encoder.copyBufferToBuffer(self.counter_buffer, 0,
                                           self.result_buffer, 0, 4)
        self.device.queue.submit([compute_encoder.finish()])
        import pyodide

        def read_buffer(task):
            data = self.result_buffer.getMappedRange(0, 4)
            b = np.frombuffer(memoryview(data.to_py()), dtype=np.int32)
            # conversion to int imporant, type is np.uint32 which fucks things up...
            self.n_cut_trigs = int(b[0])
            self.result_buffer.unmap()
            self.create_cut_trigs()

        task = pyodide.webloop.asyncio.get_running_loop().run_until_complete(
            self.result_buffer.mapAsync(MapMode.READ, 0, 4)
        )
        task.then(read_buffer)
        def error(e):
            print("error", e)
            raise e
        task.catch(error)

    def create_cut_trigs(self):
        import ngsolve as ngs
        compute_encoder = self.gpu.device.createCommandEncoder(label="create_iso_trigs")
        # binding -> counter i32
        self.device.queue.writeBuffer(self.only_count, 0,
                                      np.uint32(0).tobytes(), 0, 4)
        self.device.queue.writeBuffer(
            self.counter_buffer, 0, np.array([0], dtype=np.uint32).tobytes(), 0, 4
        )
        cut_buffer_size = 64 * self.n_cut_trigs
        self.cut_trigs_buffer = self.device.createBuffer(
            label="cut trigs " + str(self.n_cut_trigs),
            size=cut_buffer_size,
            usage=BufferUsage.STORAGE,
        )
        cut_trigs_binding = BufferBinding(
            82, self.cut_trigs_buffer, read_only=False, visibility=ShaderStage.COMPUTE
        )
        bindings = [
            *self.gpu.camera.get_bindings(),
            *self.gpu.u_mesh.get_bindings(),
            BufferBinding(
                80, self.counter_buffer,
                read_only=False, visibility=ShaderStage.COMPUTE
            ),
            BufferBinding(83, self.function_value_buffer,
                          visibility=ShaderStage.ALL),
            BufferBinding(12, self.vertex_buffer, visibility=ShaderStage.ALL),
            UniformBinding(81, self.only_count),
        ]

        layout, group = create_bind_group(
            self.device, bindings + [cut_trigs_binding], "create_cut_trigs"
        )
        shader_code = ""
        compute_shader_code = read_shader_file("compute_isosurface.wgsl", __file__)
        shader_code += read_shader_file("isosurface.wgsl", __file__)
        shader_module = self.device.createShaderModule(
            compute_shader_code + shader_code
        )
        pipeline = self.device.createComputePipeline(
            self.device.createPipelineLayout([layout], self.label + " create"),
            label="count_iso_trigs",
            compute=ComputeState(
                module=shader_module,
                entryPoint="create_iso_triangles"
            ),
        )
        # self.gpu.begin_render_pass(encoder, label="count_iso_trigs")
        compute_pass = compute_encoder.beginComputePass(label="count_iso_trigs")
        compute_pass.setBindGroup(0, group)
        compute_pass.setPipeline(pipeline)
        compute_pass.dispatchWorkgroups(self.mesh.ne)
        compute_pass.end()

        self.device.queue.submit([compute_encoder.finish()])
        print("Create render pipeline with cut trigs set = ", self.n_cut_trigs)
        encoder = self.gpu.device.createCommandEncoder()
        draw_func_values = np.array(self.function(self.mesh_pts).flatten(), dtype=np.float32)
        self.colormap = Colormap(self.gpu.device, min(draw_func_values), max(draw_func_values))
        self.draw_func_value_buffer = self.device.createBuffer(
            size=len(draw_func_values) * draw_func_values.itemsize,
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST,
            label="draw function",
        )
        self.device.queue.writeBuffer(
            self.draw_func_value_buffer, 0, draw_func_values.tobytes()
        )
        draw_func_binding = BufferBinding(84, self.draw_func_value_buffer,
                                          visibility=ShaderStage.VERTEX)
        render_cut_trigs_binding = BufferBinding(82, self.cut_trigs_buffer,
                                                 visibility=ShaderStage.VERTEX)
        render_layout, self._bind_group = create_bind_group(
            self.device, bindings + [render_cut_trigs_binding, draw_func_binding,
                                     *self.colormap.get_bindings()]
        )
        render_shader_code = read_shader_file("render_isosurface.wgsl", __file__)
        render_shader_module = self.device.createShaderModule(
            render_shader_code + shader_code + self.colormap.get_shader_code()
        )
        self._pipeline = self.device.createRenderPipeline(
            self.device.createPipelineLayout([render_layout], self.label),
            vertex=VertexState(
                module=render_shader_module, entryPoint="vertexIsoSurface"
            ),
            fragment=FragmentState(
                module=render_shader_module,
                entryPoint="fragmentIsoSurface",
                targets=[ColorTargetState(format=self.gpu.format)],
            ),
            primitive=PrimitiveState(topology=PrimitiveTopology.triangle_list),
            depthStencil=DepthStencilState(
                format=TextureFormat.depth24plus,
                depthWriteEnabled=True,
                depthCompare=CompareFunction.less,
                # shift trigs behind to ensure that edges are rendered properly
                depthBias=1,
                depthBiasSlopeScale=1.0,
            ),
            multisample=self.gpu.multisample,
        )

        self.cut_trigs_set = True
        self.render(encoder)
        self.gpu.update_uniforms()
        self.gpu.device.queue.submit([encoder.finish()])

    def update(self):
        self.count_cut_trigs()

    def set_render_camera(self, input_handler):
        import numpy as np
        input_handler.transform._center = 0.5 * (self.pmin + self.pmax)
        input_handler.transform._scale = 2/np.linalg.norm(self.pmax - self.pmin)
        input_handler.transform.rotate(30, -20)
        input_handler._update_uniforms()


    def render(self, encoder):
        if self.cut_trigs_set is False:
            return
        render_pass = self.gpu.begin_render_pass(encoder, label=self.label)
        render_pass.setBindGroup(0, self._bind_group)
        render_pass.setPipeline(self._pipeline)
        render_pass.draw(3, self.n_cut_trigs)
        render_pass.end()


def render_isosurface(levelset, function, mesh, name="isosurface"):
    import js
    import pyodide.ffi
    from webgpu.jupyter import gpu

    iso = IsoSurfaceRenderObject(gpu, levelset, function, mesh, name)
    iso.set_render_camera(gpu.input_handler)

    def render_function(t):
        gpu.update_uniforms()
        encoder = gpu.device.createCommandEncoder()
        iso.render(encoder)
        gpu.device.queue.submit([encoder.finish()])

    render_function = pyodide.ffi.create_proxy(render_function)
    gpu.input_handler.render_function = render_function
    # js.requestAnimationFrame(render_function)

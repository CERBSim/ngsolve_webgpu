import numpy as np
from webgpu import (RenderObject, BufferBinding, read_shader_file, Colormap,
                    UniformBinding, create_bind_group)
from webgpu.webgpu_api import (
    BufferUsage,
    ComputeState,
    ShaderStage,
    MapMode,
)


class Binding:
    COUNTER = 80
    COUNT_FLAG = 81
    CUT_TRIANGLES = 82
    FUNCTION_VALUES = 83
    DRAW_FUNCTION_VALUES = 84
    VERTICES = 12

class IsoSurfaceRenderObject(RenderObject):
    n_vertices: int = 3
    vertex_entry_point: str = "vertexIsoSurface"
    fragment_entry_point: str = "fragmentIsoSurface"

    def __init__(self, gpu, levelset, function, mesh, label):
        super().__init__(gpu, label=label)
        self.levelset = levelset
        self.function = function
        self.mesh = mesh
        self.cut_trigs_set = False
        self.task = None

    def count_cut_trigs(self):
        import ngsolve as ngs

        compute_encoder = self.gpu.device.createCommandEncoder(label="count_iso_trigs")
        # binding -> counter i32
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
        self.device.queue.writeBuffer(
            self.function_value_buffer, 0, func_values.tobytes()
        )
        func_values = np.array(self.levelset(self.mesh_pts).flatten(), dtype=np.float32)
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
            label="cut trigs " + str(self.n_instances),
            size=64,
            usage=BufferUsage.STORAGE,
        )
        cut_trigs_binding = BufferBinding(
            Binding.CUT_TRIANGLES,
            cut_trigs_buffer,
            read_only=False,
            visibility=ShaderStage.COMPUTE,
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
            self.only_count,
            0,
            np.uint32(1).tobytes(),
            0,
            4,
        )
        self.device.queue.writeBuffer(
            self.counter_buffer, 0, np.array([0], dtype=np.uint32).tobytes(), 0, 4
        )
        bindings = [
            *self.gpu.camera.get_bindings(),
            *self.gpu.u_mesh.get_bindings(),
            BufferBinding(
                Binding.COUNTER,
                self.counter_buffer,
                read_only=False,
                visibility=ShaderStage.COMPUTE,
            ),
            BufferBinding(
                Binding.FUNCTION_VALUES,
                self.function_value_buffer,
                visibility=ShaderStage.ALL,
            ),
            BufferBinding(
                Binding.VERTICES, self.vertex_buffer, visibility=ShaderStage.ALL
            ),
            UniformBinding(Binding.COUNT_FLAG, self.only_count),
            cut_trigs_binding,
        ]

        layout, group = create_bind_group(self.device, bindings, "count_cut_trigs")
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
                module=shader_module, entryPoint="create_iso_triangles"
            ),
        )
        compute_pass = compute_encoder.beginComputePass(label="count_iso_trigs")
        compute_pass.setBindGroup(0, group)
        compute_pass.setPipeline(pipeline)
        compute_pass.dispatchWorkgroups(self.mesh.ne)
        compute_pass.end()

        compute_encoder.copyBufferToBuffer(
            self.counter_buffer, 0, self.result_buffer, 0, 4
        )
        self.device.queue.submit([compute_encoder.finish()])
        import pyodide

        def read_buffer(task):
            try:
                data = self.result_buffer.getMappedRange(0, 4)
                b = np.frombuffer(memoryview(data.to_py()), dtype=np.int32)
                # conversion to int imporant, type is np.uint32 which fucks things up...
                self.n_instances = int(b[0])
                self.result_buffer.unmap()
                self.create_cut_trigs()
            except Exception as e:
                print(e)

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
        self.device.queue.writeBuffer(self.only_count, 0, np.uint32(0).tobytes(), 0, 4)
        self.device.queue.writeBuffer(
            self.counter_buffer, 0, np.array([0], dtype=np.uint32).tobytes(), 0, 4
        )
        cut_buffer_size = 64 * self.n_instances
        self.cut_trigs_buffer = self.device.createBuffer(
            label="cut trigs ",
            size=cut_buffer_size,
            usage=BufferUsage.STORAGE,
        )
        cut_trigs_binding = BufferBinding(
            Binding.CUT_TRIANGLES,
            self.cut_trigs_buffer,
            read_only=False,
            visibility=ShaderStage.COMPUTE,
        )
        bindings = [
            *self.gpu.camera.get_bindings(),
            *self.gpu.u_mesh.get_bindings(),
            BufferBinding(
                Binding.COUNTER,
                self.counter_buffer,
                read_only=False,
                visibility=ShaderStage.COMPUTE,
            ),
            BufferBinding(
                Binding.FUNCTION_VALUES,
                self.function_value_buffer,
                visibility=ShaderStage.ALL,
            ),
            BufferBinding(
                Binding.VERTICES, self.vertex_buffer, visibility=ShaderStage.ALL
            ),
            UniformBinding(Binding.COUNT_FLAG, self.only_count),
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
        self.shader_code = shader_code
        pipeline = self.device.createComputePipeline(
            self.device.createPipelineLayout([layout], self.label + " create"),
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

        self.device.queue.submit([compute_encoder.finish()])
        print("Create render pipeline with cut trigs set = ", self.n_instances)
        draw_func_values = np.array(
            self.function(self.mesh_pts).flatten(), dtype=np.float32
        )
        self.colormap = Colormap(
            self.gpu.device, min(draw_func_values), max(draw_func_values)
        )
        self.draw_func_value_buffer = self.device.createBuffer(
            size=len(draw_func_values) * draw_func_values.itemsize,
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST,
            label="draw function",
        )
        self.device.queue.writeBuffer(
            self.draw_func_value_buffer, 0, draw_func_values.tobytes()
        )
        self.create_render_pipeline()
        self.cut_trigs_set = True
        encoder = self.gpu.device.createCommandEncoder()
        self.gpu.update_uniforms()
        self.render(encoder)
        self.gpu.device.queue.submit([encoder.finish()])

    def get_bindings(self):
        return [
            *self.gpu.camera.get_bindings(),
            *self.gpu.u_mesh.get_bindings(),
            *self.colormap.get_bindings(),
            BufferBinding(
                Binding.FUNCTION_VALUES,
                self.function_value_buffer,
            ),
            BufferBinding(
                Binding.VERTICES, self.vertex_buffer,
            ),
            UniformBinding(Binding.COUNT_FLAG, self.only_count),
            BufferBinding(
                Binding.CUT_TRIANGLES, self.cut_trigs_buffer, visibility=ShaderStage.VERTEX
                ),
            BufferBinding(
                Binding.DRAW_FUNCTION_VALUES,
                self.draw_func_value_buffer,
                visibility=ShaderStage.VERTEX),
            ]
        

    def get_shader_code(self):
        render_shader_code = read_shader_file("render_isosurface.wgsl", __file__)
        return render_shader_code + self.shader_code + self.colormap.get_shader_code()

    def update(self):
        self.count_cut_trigs()

    def get_bounding_box(self):
        return (self.pmin, self.pmax)

    def render(self, encoder):
        if self.cut_trigs_set is False:
            return
        super().render(encoder)

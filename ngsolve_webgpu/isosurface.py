import numpy as np
from webgpu import (
    BufferBinding,
    Colormap,
    Clipping,
    RenderObject,
    UniformBinding,
    create_bind_group,
    read_shader_file,
)
from webgpu.webgpu_api import BufferUsage, ComputeState, MapMode, ShaderStage

from webgpu.utils import uniform_from_array, buffer_from_array, ReadBuffer

from .cf import CoefficientFunctionRenderObject


class Binding:
    COUNTER = 80
    COUNT_FLAG = 81
    CUT_TRIANGLES = 82
    FUNCTION_VALUES = 83
    DRAW_FUNCTION_VALUES = 84
    N_TETS = 85
    VERTICES = 12


class IsoSurfaceRenderObject(RenderObject):
    n_vertices: int = 3
    vertex_entry_point: str = "vertexIsoSurface"
    fragment_entry_point: str = "fragmentIsoSurface"

    def __init__(self, levelset, function, mesh, label):
        super().__init__(label=label)
        self.levelset = levelset
        self.function = function
        self.mesh = mesh
        self.cut_trigs_set = False
        self.task = None
        self.colormap = Colormap()
        self.clipping = Clipping()

    def update(self, levelset=None, function=None, mesh=None):
        if levelset is not None:
            self.levelset = levelset
        if function is not None:
            self.function = function
        if mesh is not None:
            self.mesh = mesh
        self.cut_trigs_set = False
        self.count_cut_trigs()

    def redraw(self, timestamp: float | None = None, **kwargs):
        super().redraw(levelset=self.levelset, function=self.function, mesh=self.mesh)

    def count_cut_trigs(self):
        import ngsolve as ngs

        self.clipping.update()
        compute_encoder = self.device.createCommandEncoder(label="count_iso_trigs")
        # binding -> counter i32
        self.counter_buffer = buffer_from_array(
            np.array([0], dtype=np.uint32),
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,
        )

        self.ntets_buffer = uniform_from_array(
            np.array([self.mesh.ne], dtype=np.uint32)
        )

        self.mesh_pts = self.mesh.MapToAllElements(
            ngs.IntegrationRule([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),
            self.mesh.Materials(".*"),
        )
        func_values = np.array(self.levelset(self.mesh_pts).flatten(), dtype=np.float32)

        self.function_value_buffer = buffer_from_array(func_values)

        func_values = np.array(self.levelset(self.mesh_pts).flatten(), dtype=np.float32)
        vertices = np.array(
            ngs.CF((ngs.x, ngs.y, ngs.z))(self.mesh_pts), dtype=np.float32
        )
        self.pmin = np.min(vertices, axis=0)
        self.pmax = np.max(vertices, axis=0)
        vertices = vertices.flatten()
        self.vertex_buffer = buffer_from_array(vertices)

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
        self.result_buffer = self.device.createBuffer(
            size=4,
            label="result",
            usage=BufferUsage.MAP_READ | BufferUsage.COPY_DST,
        )
        self.only_count = uniform_from_array(np.array([1], dtype=np.uint32))
        bindings = [
            *self.options.camera.get_bindings(),
            *self.clipping.get_bindings(),
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
            UniformBinding(Binding.N_TETS, self.ntets_buffer),
            UniformBinding(Binding.COUNT_FLAG, self.only_count),
            cut_trigs_binding,
        ]

        layout, group = create_bind_group(self.device, bindings, "count_cut_trigs")
        shader_code = ""
        compute_shader_code = read_shader_file("compute_isosurface.wgsl", __file__)
        shader_code += read_shader_file("isosurface.wgsl", __file__)
        shader_code += self.clipping.get_shader_code()
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
        compute_pass.dispatchWorkgroups(min(1024, self.mesh.ne))
        compute_pass.end()

        compute_encoder.copyBufferToBuffer(
            self.counter_buffer, 0, self.result_buffer, 0, 4
        )
        read = ReadBuffer(self.result_buffer, compute_encoder)
        self.device.queue.submit([compute_encoder.finish()])

        array = read.get_array(np.uint32)
        print("array", array)
        self.n_instances = int(array[0])
        print("n_instances", self.n_instances)
        self.create_cut_trigs()

    def create_cut_trigs(self):
        if self.n_instances == 0:
            return
        compute_encoder = self.device.createCommandEncoder(label="create_iso_trigs")
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
            *self.options.camera.get_bindings(),
            *self.clipping.get_bindings(),
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
            UniformBinding(Binding.N_TETS, self.ntets_buffer),
            UniformBinding(Binding.COUNT_FLAG, self.only_count),
        ]

        layout, group = create_bind_group(
            self.device, bindings + [cut_trigs_binding], "create_cut_trigs"
        )
        shader_code = ""
        compute_shader_code = read_shader_file("compute_isosurface.wgsl", __file__)
        shader_code += read_shader_file("isosurface.wgsl", __file__)
        shader_code += self.clipping.get_shader_code()
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
        compute_pass = compute_encoder.beginComputePass(label="count_iso_trigs")
        compute_pass.setBindGroup(0, group)
        compute_pass.setPipeline(pipeline)
        compute_pass.dispatchWorkgroups(min(1024, self.mesh.ne))
        compute_pass.end()

        self.device.queue.submit([compute_encoder.finish()])
        draw_func_values = np.array(
            self.function(self.mesh_pts).flatten(), dtype=np.float32
        )
        self.colormap.options = self.options
        if self.colormap.autoupdate:
            self.colormap.set_min_max(
                min(draw_func_values), max(draw_func_values), set_autoupdate=False
            )
        self.colormap.update()
        self.clipping.update()
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
        self.options.render_function(0.0)

    def get_bindings(self):
        return [
            *self.options.camera.get_bindings(),
            *self.clipping.get_bindings(),
            *self.colormap.get_bindings(),
            BufferBinding(
                Binding.FUNCTION_VALUES,
                self.function_value_buffer,
            ),
            BufferBinding(
                Binding.VERTICES,
                self.vertex_buffer,
            ),
            UniformBinding(Binding.COUNT_FLAG, self.only_count),
            BufferBinding(
                Binding.CUT_TRIANGLES,
                self.cut_trigs_buffer,
                visibility=ShaderStage.VERTEX,
            ),
            BufferBinding(
                Binding.DRAW_FUNCTION_VALUES,
                self.draw_func_value_buffer,
                visibility=ShaderStage.VERTEX,
            ),
        ]

    def get_shader_code(self):
        render_shader_code = read_shader_file("render_isosurface.wgsl", __file__)
        return render_shader_code + self.shader_code + self.colormap.get_shader_code()

    def get_bounding_box(self):
        return (self.pmin, self.pmax)

    def render(self, encoder):
        if self.cut_trigs_set is False:
            return
        super().render(encoder)


class NegativeSurfaceRenderer(CoefficientFunctionRenderObject):
    def __init__(self, functiondata, levelsetdata):
        super().__init__(functiondata, label="NegativeSurfaceRenderer")
        self.fragment_entry_point = "fragmentCheckLevelset"
        self.levelset = levelsetdata

    def redraw(self, timestamp: float | None = None, **kwargs):
        self.levelset.redraw(timestamp=timestamp)
        super().redraw(timestamp=timestamp, **kwargs)

    def update(self, **kwargs):
        buffers = self.levelset.get_buffers(self.device)
        self.levelset_buffer = buffers["function"]
        super().update(**kwargs)

    def get_bindings(self):
        return super().get_bindings() + [BufferBinding(80, self.levelset_buffer)]

    def get_shader_code(self):
        return super().get_shader_code() + read_shader_file(
            "negative_surface.wgsl", __file__
        )

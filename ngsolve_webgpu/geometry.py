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
    BufferUsage,
    ColorTargetState,
    CommandEncoder,
    CompareFunction,
    DepthStencilState,
    FragmentState,
    PrimitiveState,
    PrimitiveTopology,
    TextureFormat,
    VertexState,
)


class Binding:
    VERTICES = 90
    NORMALS = 91
    INDICES = 92


class GeometryRenderObject(RenderObject):
    def __init__(self, gpu, geo, label):
        super().__init__(gpu, label=label)
        self.geo = geo
        self.update()

    def update(self):
        import numpy as np

        bindings = [
            *self.gpu.camera.get_bindings(),
            *self.gpu.u_mesh.get_bindings(),
        ]
        vis_data = self.geo._visualizationData()
        verts = vis_data["vertices"].flatten()
        self.num_trigs = len(verts) // 9
        normals = vis_data["normals"].flatten()
        center = 0.5 * (vis_data["max"] + vis_data["min"])
        diam = np.linalg.norm(vis_data["max"] - vis_data["min"])
        indices = np.array(vis_data["triangles"][3::4], dtype=np.uint32).flatten()
        self._buffers = {}
        for key, data, binding in zip(
            ("vertices", "normals", "indices"),
            (verts, normals, indices),
            (Binding.VERTICES, Binding.NORMALS, Binding.INDICES),
        ):
            b = self.device.createBuffer(
                size=len(data) * data.itemsize,
                usage=BufferUsage.STORAGE | BufferUsage.COPY_DST,
            )
            self.device.queue.writeBuffer(b, 0, data.tobytes())
            self._buffers[key] = b
            bindings.append(BufferBinding(binding, b))
        bind_layout, self._bind_group = create_bind_group(
            self.device, bindings, self.label
        )
        shader_code = ""
        shader_code += read_shader_file("colormap.wgsl", webgpu.__file__)
        for shader in ["uniforms", "shader", "geometry", "eval", "clipping"]:
            shader_code += read_shader_file(f"{shader}.wgsl", __file__)

        shader_code += self.gpu.camera.get_shader_code()
        shader_code += self.gpu.light.get_shader_code()

        shader_module = self.device.createShaderModule(shader_code)

        self._pipeline = self.device.createRenderPipeline(
            self.device.createPipelineLayout([bind_layout], self.label),
            vertex=VertexState(module=shader_module, entryPoint="vertexGeo"),
            fragment=FragmentState(
                module=shader_module,
                entryPoint="fragmentGeo",
                targets=[ColorTargetState(format=self.gpu.format)],
            ),
            primitive=PrimitiveState(topology=PrimitiveTopology.triangle_list),
            depthStencil=DepthStencilState(
                format=TextureFormat.depth24plus,
                depthWriteEnabled=True,
                depthCompare=CompareFunction.less,
                # shift trigs behind to ensure that edges are rendered properly
                depthBias=1,
                depthBiasSlopeScale=0.0,
            ),
            multisample=self.gpu.multisample,
        )

        self.gpu.update_uniforms()

    def render(self, encoder: CommandEncoder):
        render_pass = self.gpu.begin_render_pass(encoder, label=self.label)
        render_pass.setBindGroup(0, self._bind_group)
        render_pass.setPipeline(self._pipeline)
        render_pass.draw(3, self.num_trigs)
        render_pass.end()


def render_geometry(geo, name="Geometry"):
    import js
    import pyodide.ffi
    from webgpu.jupyter import gpu

    render_object = GeometryRenderObject(gpu, geo, name)

    def render_function(t):
        gpu.update_uniforms()
        encoder = gpu.device.createCommandEncoder()
        render_object.render(encoder)
        gpu.device.queue.submit([encoder.finish()])

    render_function = pyodide.ffi.create_proxy(render_function)
    gpu.input_handler.render_function = render_function
    js.requestAnimationFrame(render_function)

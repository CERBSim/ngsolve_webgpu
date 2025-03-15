import webgpu
from webgpu.utils import buffer_from_array, create_bind_group, ReadBuffer
from webgpu.webgpu_api import *

import numpy as np

class Binding:
    VERTICES = 90
    NORMALS = 91
    INDICES = 92
    COLORS = 93


class GeometryRenderObject(webgpu.RenderObject):
    vertex_entry_point: str = "vertexGeo"
    fragment_entry_point: str = "fragmentGeo"
    n_vertices: int = 3

    def __init__(self, geo, label="Geometry"):
        self.geo = geo
        super().__init__(label=label)
        self.colors = None

    def get_bounding_box(self):
        return self.bounding_box

    def set_colors(self, colors):
        """ colors is numpy float32 array with 4 times number indices entries"""
        self.colors = colors
        if "colors" in self._buffers:
            self.device.queue.writeBuffer(self._buffers["colors"],
                                          0, self.colors.tobytes())

    def update(self):
        vis_data = self.geo._visualizationData()
        self.bounding_box = (vis_data["min"], vis_data["max"])
        verts = vis_data["vertices"]
        self.n_instances = len(verts) // 9
        normals = vis_data["normals"]
        indices = vis_data["indices"]
        if self.colors is None:
            self.colors = vis_data["face_colors"]
        self._buffers = {}
        for key, data in zip(
            ("vertices", "normals", "indices", "colors"),
            (verts, normals, indices, self.colors),
        ):
            self._buffers[key] = buffer_from_array(data)
        self.create_render_pipeline()

    def get_bindings(self):
        return [
            *self.options.camera.get_bindings(),
            webgpu.BufferBinding(Binding.VERTICES, self._buffers["vertices"]),
            webgpu.BufferBinding(Binding.NORMALS, self._buffers["normals"]),
            webgpu.BufferBinding(Binding.INDICES, self._buffers["indices"]),
            webgpu.BufferBinding(Binding.COLORS, self._buffers["colors"]),
        ]

    def get_shader_code(self):
        shader_code = ""
        for shader in ["geometry", "clipping"]:
            shader_code += webgpu.read_shader_file(f"{shader}.wgsl", __file__)

        shader_code += self.options.camera.get_shader_code()
        shader_code += self.options.light.get_shader_code()
        return shader_code

    def pick_index(self, mouseX, mouseY):
        rect = self.canvas.canvas.getBoundingClientRect()
        mouseX -= rect.x
        mouseY -= int(rect.y)
        texture_format = TextureFormat.r32uint
        texture = self.device.createTexture(
            size=[rect.width, rect.height, 1],
            sampleCount=1,
            format=texture_format,
            usage=TextureUsage.COPY_SRC | TextureUsage.RENDER_ATTACHMENT
        )
        target = ColorTargetState(format=texture_format)
        shader_module = self.device.createShaderModule(self.get_shader_code())
        layout, group = create_bind_group(self.device, self.get_bindings())
        playout = self.device.createPipelineLayout([layout])
        depth_texture = self.device.createTexture(
            size=[rect.width, rect.height, 1],
            format=self.canvas.depth_format,
            usage=TextureUsage.RENDER_ATTACHMENT,
            sampleCount=1,
        )
        pipeline = self.device.createRenderPipeline(
            layout=playout,
            vertex=VertexState(
                module=shader_module, entryPoint=self.vertex_entry_point
            ),
            fragment=FragmentState(
                module=shader_module,
                entryPoint="fragmentQueryFaceIndex",
                targets=[target],
            ),
            primitive=PrimitiveState(topology=self.topology),
            depthStencil=DepthStencilState(
                format=self.options.canvas.depth_format,
                depthWriteEnabled=True,
                depthCompare=CompareFunction.less,
                depthBias=self.depthBias,
            ),
            multisample=MultisampleState(count=1),
        )
        read_buffer = self.device.createBuffer(4, BufferUsage.MAP_READ | BufferUsage.COPY_DST)
        encoder = self.device.createCommandEncoder()
        render_pass = encoder.beginRenderPass(
            colorAttachments=[RenderPassColorAttachment(
                view=texture.createView(),
                clearValue=Color(1),
                loadOp=LoadOp.clear
            )],
            depthStencilAttachment=RenderPassDepthStencilAttachment(
                view=depth_texture.createView(),
                depthClearValue=1.0,
                depthLoadOp=LoadOp.clear
            )
        )
        render_pass.setPipeline(pipeline)
        render_pass.setBindGroup(0, group)
        render_pass.draw(self.n_vertices, self.n_instances)
        render_pass.end()
        encoder.copyTextureToBuffer(TexelCopyTextureInfo(texture,
                                                         origin=Origin3d(mouseX, mouseY, 0)),
                                    TexelCopyBufferInfo(TexelCopyBufferLayout(1),
                                                        read_buffer),
                                    [1, 1, 1])
        self.device.queue.submit([encoder.finish()])
        read_buffer.handle.mapAsync(MapMode.READ, 0, 4)
        result = np.frombuffer(read_buffer.handle.getMappedRange(0, 4),
                               dtype=np.uint32)
        return result[0]
            

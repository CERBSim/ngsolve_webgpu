import webgpu
from webgpu.render_object import MultipleRenderObject, RenderObject
from webgpu.utils import (
    buffer_from_array,
    create_bind_group,
    ReadBuffer,
    uniform_from_array,
)
from webgpu.webgpu_api import *
from webgpu.clipping import Clipping

import numpy as np


class Binding:
    VERTICES = 90
    NORMALS = 91
    INDICES = 92
    COLORS = 93


class GeometryFaceRenderer(RenderObject):
    n_vertices: int = 3
    depthBias: int = 1
    clipping: Clipping | None = None

    def __init__(self, geo):
        super().__init__(label="GeometryFaces")
        self.geo = geo
        self.colors = None
        self.active = True

    def get_bounding_box(self):
        return self.bounding_box

    def set_colors(self, colors):
        """colors is numpy float32 array with 4 times number indices entries"""
        self.colors = colors
        if "colors" in self._buffers:
            self.device.queue.writeBuffer(
                self._buffers["colors"], 0, self.colors.tobytes()
            )

    def update(self, vis_data):
        self.bounding_box = (vis_data["min"], vis_data["max"])
        verts = vis_data["vertices"]
        self.n_instances = len(verts) // 6
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
        shader_code += webgpu.read_shader_file(f"geo_face.wgsl", __file__)
        shader_code += self.options.camera.get_shader_code()
        shader_code += self.options.light.get_shader_code()
        shader_code += self.clipping.get_shader_code()
        return shader_code

    def pick_index_render(self, encoder, texture, depth_texture, load_op):
        texture_format = TextureFormat.rg32uint
        target = ColorTargetState(format=texture_format)
        shader_module = self.device.createShaderModule(self.get_shader_code())
        layout, group = create_bind_group(self.device, self.get_bindings())
        playout = self.device.createPipelineLayout([layout])
        pipeline = self.device.createRenderPipeline(
            layout=playout,
            vertex=VertexState(
                module=shader_module, entryPoint=self.vertex_entry_point
            ),
            fragment=FragmentState(
                module=shader_module,
                entryPoint="fragmentQueryIndex",
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
        render_pass = encoder.beginRenderPass(
            colorAttachments=[
                RenderPassColorAttachment(
                    view=texture.createView(), clearValue=Color(0, 3), loadOp=load_op
                )
            ],
            depthStencilAttachment=RenderPassDepthStencilAttachment(
                view=depth_texture.createView(),
                depthClearValue=1.0,
                depthLoadOp=load_op,
            ),
        )
        render_pass.setPipeline(pipeline)
        render_pass.setBindGroup(0, group)
        render_pass.draw(self.n_vertices, self.n_instances)
        render_pass.end()


class GeometryEdgeRenderer(RenderObject):
    n_vertices: int = 4
    topology: PrimitiveTopology = PrimitiveTopology.triangle_strip
    clipping: Clipping | None = None

    def __init__(self, geo):
        self.geo = geo
        super().__init__(label="GeometryEdges")
        self.active = True
        self.thickness = 0.02

    def set_colors(self, colors):
        """colors is numpy float32 array with 4 times number indices entries"""
        self.colors = colors
        if "colors" in self._buffers:
            self.device.queue.writeBuffer(
                self._buffers["colors"], 0, self.colors.tobytes()
            )

    def update(self, vis_data):
        verts = vis_data["edges"]
        self.colors = vis_data["edge_colors"]
        self.n_instances = len(verts) // 6
        self.thickness_uniform = uniform_from_array(
            np.array([self.thickness], dtype=np.float32)
        )
        self._buffers = {}
        self._buffers["vertices"] = buffer_from_array(verts)
        self._buffers["colors"] = buffer_from_array(self.colors)
        self._buffers["index"] = buffer_from_array(vis_data["edge_indices"])
        self.create_render_pipeline()

    def get_shader_code(self):
        shader_code = ""
        shader_code += webgpu.read_shader_file(f"geo_edge.wgsl", __file__)
        shader_code += self.options.camera.get_shader_code()
        shader_code += self.clipping.get_shader_code()
        return shader_code

    def get_bindings(self):
        return [
            *self.options.camera.get_bindings(),
            webgpu.BufferBinding(90, self._buffers["vertices"]),
            webgpu.BufferBinding(91, self._buffers["colors"]),
            webgpu.UniformBinding(92, self.thickness_uniform),
            webgpu.BufferBinding(93, self._buffers["index"]),
        ]

    def pick_index_render(self, encoder, texture, depth_texture, load_op):
        texture_format = TextureFormat.rg32uint
        target = ColorTargetState(format=texture_format)
        shader_module = self.device.createShaderModule(self.get_shader_code())
        layout, group = create_bind_group(self.device, self.get_bindings())
        playout = self.device.createPipelineLayout([layout])
        pipeline = self.device.createRenderPipeline(
            layout=playout,
            vertex=VertexState(
                module=shader_module, entryPoint=self.vertex_entry_point
            ),
            fragment=FragmentState(
                module=shader_module,
                entryPoint="fragmentQueryIndex",
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
        render_pass = encoder.beginRenderPass(
            colorAttachments=[
                RenderPassColorAttachment(
                    view=texture.createView(), clearValue=Color(0, 3), loadOp=load_op
                )
            ],
            depthStencilAttachment=RenderPassDepthStencilAttachment(
                view=depth_texture.createView(),
                depthClearValue=1.0,
                depthLoadOp=load_op,
            ),
        )
        render_pass.setPipeline(pipeline)
        render_pass.setBindGroup(0, group)
        render_pass.draw(self.n_vertices, self.n_instances)
        render_pass.end()


class GeometryVertexRenderer(RenderObject):
    n_vertices: int = 4
    topology: PrimitiveTopology = PrimitiveTopology.triangle_strip
    depthBias: int = 0
    clipping: Clipping | None = None

    def __init__(self, geo):
        self.geo = geo
        super().__init__(label="GeometryVertices")
        self.active = True
        self.thickness = 0.05

    def set_colors(self, colors):
        """colors is numpy float32 array with 4 times number indices entries"""
        self.colors = colors
        if "colors" in self._buffers:
            self.device.queue.writeBuffer(
                self._buffers["colors"], 0, self.colors.tobytes()
            )

    def get_shader_code(self):
        shader_code = ""
        shader_code += webgpu.read_shader_file(f"geo_vertex.wgsl", __file__)
        shader_code += self.clipping.get_shader_code()
        shader_code += self.options.camera.get_shader_code()
        return shader_code

    def update(self, vis_data):
        verts = set(self.geo.shape.vertices)
        self.colors = np.array(
            [v.col if v.col is not None else [0.3, 0.3, 0.3, 1.0] for v in verts],
            dtype=np.float32,
        ).flatten()
        self.n_instances = len(verts)
        vert_values = np.array(
            [[pi for pi in v.p] for v in verts], dtype=np.float32
        ).flatten()
        self._buffers = {}
        self._buffers["vertices"] = buffer_from_array(vert_values)
        self._buffers["colors"] = buffer_from_array(self.colors)
        self.thickness_uniform = uniform_from_array(
            np.array([self.thickness], dtype=np.float32)
        )
        self.create_render_pipeline()

    def get_bindings(self):
        return [
            *self.options.camera.get_bindings(),
            webgpu.BufferBinding(90, self._buffers["vertices"]),
            webgpu.BufferBinding(91, self._buffers["colors"]),
            webgpu.UniformBinding(92, self.thickness_uniform),
        ]

    def pick_index_render(self, encoder, texture, depth_texture, load_op):
        texture_format = TextureFormat.rg32uint
        target = ColorTargetState(format=texture_format)
        shader_module = self.device.createShaderModule(self.get_shader_code())
        layout, group = create_bind_group(self.device, self.get_bindings())
        playout = self.device.createPipelineLayout([layout])
        pipeline = self.device.createRenderPipeline(
            layout=playout,
            vertex=VertexState(
                module=shader_module, entryPoint=self.vertex_entry_point
            ),
            fragment=FragmentState(
                module=shader_module,
                entryPoint="fragmentQueryIndex",
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
        render_pass = encoder.beginRenderPass(
            colorAttachments=[
                RenderPassColorAttachment(
                    view=texture.createView(), clearValue=Color(0, 3), loadOp=load_op
                )
            ],
            depthStencilAttachment=RenderPassDepthStencilAttachment(
                view=depth_texture.createView(),
                depthClearValue=1.0,
                depthLoadOp=load_op,
            ),
        )
        render_pass.setPipeline(pipeline)
        render_pass.setBindGroup(0, group)
        render_pass.draw(self.n_vertices, self.n_instances)
        render_pass.end()


class GeometryRenderObject(MultipleRenderObject):
    def __init__(self, geo, label="Geometry"):
        self.geo = geo
        self.faces = GeometryFaceRenderer(geo)
        self.edges = GeometryEdgeRenderer(geo)
        self.vertices = GeometryVertexRenderer(geo)
        self.clipping = Clipping()
        self.faces.clipping = self.clipping
        self.edges.clipping = self.clipping
        self.vertices.clipping = self.clipping
        super().__init__([self.vertices, self.edges, self.faces])

    def update(self, timestamp):
        if timestamp == self._timestamp:
            return
        self._timestamp = timestamp
        vis_data = self.geo._visualizationData()
        self.bounding_box = (vis_data["min"] + 1e-7, vis_data["max"] - 1e-7)

        for ro in self.render_objects:
            ro.options = self.options
            ro.update(vis_data)

    def get_bounding_box(self):
        return self.bounding_box

    def render(self, encoder):
        for r in self.render_objects:
            if r.active:
                r.render(encoder)

    def pick_index(self, mouseX, mouseY):
        rect = self.canvas.canvas.getBoundingClientRect()
        mouseX -= rect.x
        mouseY -= int(rect.y)
        texture_format = TextureFormat.rg32uint
        read_size = 8
        texture = self.device.createTexture(
            size=[rect.width, rect.height, 1],
            sampleCount=1,
            format=texture_format,
            usage=TextureUsage.COPY_SRC | TextureUsage.RENDER_ATTACHMENT,
        )
        depth_texture = self.device.createTexture(
            size=[rect.width, rect.height, 1],
            format=self.canvas.depth_format,
            usage=TextureUsage.RENDER_ATTACHMENT,
            sampleCount=1,
        )
        read_buffer = self.device.createBuffer(
            read_size, BufferUsage.MAP_READ | BufferUsage.COPY_DST
        )
        encoder = self.device.createCommandEncoder()

        load_op = LoadOp.clear
        for ro in self.render_objects:
            if ro.active:
                ro.pick_index_render(encoder, texture, depth_texture, load_op)
                load_op = LoadOp.load
        encoder.copyTextureToBuffer(
            TexelCopyTextureInfo(texture, origin=Origin3d(mouseX, mouseY, 0)),
            TexelCopyBufferInfo(TexelCopyBufferLayout(1), read_buffer),
            [1, 1, 1],
        )
        self.device.queue.submit([encoder.finish()])
        read_buffer.handle.mapAsync(MapMode.READ, 0, read_size)
        result = np.frombuffer(
            read_buffer.handle.getMappedRange(0, read_size), dtype=np.uint32
        )
        return result

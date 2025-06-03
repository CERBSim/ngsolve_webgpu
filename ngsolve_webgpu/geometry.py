import numpy as np
import webgpu
from webgpu.clipping import Clipping
from webgpu.renderer import MultipleRenderer, Renderer, RenderOptions
from webgpu.utils import (
    buffer_from_array,
    create_bind_group,
    read_buffer,
    read_shader_file,
    uniform_from_array,
)
from webgpu.webgpu_api import *


class Binding:
    VERTICES = 90
    NORMALS = 91
    INDICES = 92
    COLORS = 93


class BaseGeometryRenderer(Renderer):
    def create_render_pipeline(self, options):
        super().create_render_pipeline(options)
        self.create_pick_pipeline(options)

    def create_pick_pipeline(self, options):
        texture_format = TextureFormat.rg32uint
        target = ColorTargetState(format=texture_format)
        shader_module = self.device.createShaderModule(self._get_preprocessed_shader_code())
        layout, self.pick_group = create_bind_group(
            self.device, self.get_bindings() + options.get_bindings()
        )
        playout = self.device.createPipelineLayout([layout])
        self.pick_pipeline = self.device.createRenderPipeline(
            layout=playout,
            vertex=VertexState(module=shader_module, entryPoint=self.vertex_entry_point),
            fragment=FragmentState(
                module=shader_module,
                entryPoint="fragmentQueryIndex",
                targets=[target],
            ),
            primitive=PrimitiveState(topology=self.topology),
            depthStencil=DepthStencilState(
                format=options.canvas.depth_format,
                depthWriteEnabled=True,
                depthCompare=CompareFunction.less,
                depthBias=self.depthBias,
            ),
            multisample=MultisampleState(count=1),
        )

    def pick_index_render(self, encoder, texture, depth_texture, load_op):
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
        render_pass.setPipeline(self.pick_pipeline)
        render_pass.setBindGroup(0, self.pick_group)
        render_pass.draw(self.n_vertices, self.n_instances)
        render_pass.end()


class GeometryFaceRenderer(BaseGeometryRenderer):
    n_vertices: int = 3
    clipping: Clipping | None = None

    def __init__(self, geo):
        super().__init__(label="GeometryFaces")
        self.geo = geo
        self.colors = None
        self.active = True
        self._buffers = {}

    def get_bounding_box(self):
        return self.bounding_box

    def set_colors(self, colors):
        """colors is numpy float32 array with 4 times number indices entries"""
        self.colors = colors
        if "colors" in self._buffers:
            self.device.queue.writeBuffer(self._buffers["colors"], 0, self.colors.tobytes())

    def update(self, options, vis_data):
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

    def get_bindings(self):
        return [
            webgpu.BufferBinding(Binding.VERTICES, self._buffers["vertices"]),
            webgpu.BufferBinding(Binding.NORMALS, self._buffers["normals"]),
            webgpu.BufferBinding(Binding.INDICES, self._buffers["indices"]),
            webgpu.BufferBinding(Binding.COLORS, self._buffers["colors"]),
        ]

    def get_shader_code(self):
        return read_shader_file("ngsolve/geo_face.wgsl")


class GeometryEdgeRenderer(BaseGeometryRenderer):
    n_vertices: int = 4
    topology: PrimitiveTopology = PrimitiveTopology.triangle_strip
    clipping: Clipping | None = None

    # make sure that edges are rendered on top of faces
    depthBias: int = -5
    depthBiasSlopeScale: int = -5

    def __init__(self, geo):
        self.geo = geo
        super().__init__(label="GeometryEdges")
        self.active = True
        self.thickness = 0.015
        self._buffers = {}

    def set_colors(self, colors):
        """colors is numpy float32 array with 4 times number indices entries"""
        self.colors = colors
        if "colors" in self._buffers:
            self.device.queue.writeBuffer(self._buffers["colors"], 0, self.colors.tobytes())

    def update(self, options, vis_data):
        verts = vis_data["edges"]
        self.colors = vis_data["edge_colors"]
        self.n_instances = len(verts) // 6
        self.thickness_uniform = uniform_from_array(np.array([self.thickness], dtype=np.float32))
        self._buffers = {}
        self._buffers["vertices"] = buffer_from_array(verts)
        self._buffers["colors"] = buffer_from_array(self.colors)
        self._buffers["index"] = buffer_from_array(vis_data["edge_indices"])

    def get_shader_code(self):
        return read_shader_file("ngsolve/geo_edge.wgsl")

    def get_bindings(self):
        return [
            webgpu.BufferBinding(90, self._buffers["vertices"]),
            webgpu.BufferBinding(91, self._buffers["colors"]),
            webgpu.UniformBinding(92, self.thickness_uniform),
            webgpu.BufferBinding(93, self._buffers["index"]),
        ]


class GeometryVertexRenderer(BaseGeometryRenderer):
    n_vertices: int = 4
    topology: PrimitiveTopology = PrimitiveTopology.triangle_strip
    clipping: Clipping | None = None

    def __init__(self, geo):
        self.geo = geo
        super().__init__(label="GeometryVertices")
        self.active = True
        self.thickness = 0.05
        self._buffers = {}

    def set_colors(self, colors):
        """colors is numpy float32 array with 4 times number indices entries"""
        self.colors = colors
        if "colors" in self._buffers:
            self.device.queue.writeBuffer(self._buffers["colors"], 0, self.colors.tobytes())

    def get_shader_code(self):
        return read_shader_file("ngsolve/geo_vertex.wgsl")

    def update(self, options, vis_data):
        verts = set(self.geo.shape.vertices)
        self.colors = np.array(
            [v.col if v.col is not None else [0.3, 0.3, 0.3, 1.0] for v in verts],
            dtype=np.float32,
        ).flatten()
        self.n_instances = len(verts)
        vert_values = np.array([[pi for pi in v.p] for v in verts], dtype=np.float32).flatten()
        self._buffers = {}
        self._buffers["vertices"] = buffer_from_array(vert_values)
        self._buffers["colors"] = buffer_from_array(self.colors)
        self.thickness_uniform = uniform_from_array(np.array([self.thickness], dtype=np.float32))

    def get_bindings(self):
        return [
            webgpu.BufferBinding(90, self._buffers["vertices"]),
            webgpu.BufferBinding(91, self._buffers["colors"]),
            webgpu.UniformBinding(92, self.thickness_uniform),
        ]


class GeometryRenderer(MultipleRenderer):
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

    def update(self, options: RenderOptions):
        vis_data = self.geo._visualizationData()

        for ro in self.render_objects:
            ro.update(options, vis_data)

        self.canvas = options.canvas

    def get_bounding_box(self):
        pmin, pmax = self.geo.shape.bounding_box
        return ([pmin[0], pmin[1], pmin[2]],
                [pmax[0], pmax[1], pmax[2]])

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
        rbuffer = self.device.createBuffer(read_size, BufferUsage.COPY_DST | BufferUsage.MAP_READ)
        encoder = self.device.createCommandEncoder()
        load_op = LoadOp.clear
        for ro in self.render_objects:
            if ro.active:
                ro.pick_index_render(encoder, texture, depth_texture, load_op)
                load_op = LoadOp.load
        encoder.copyTextureToBuffer(
            TexelCopyTextureInfo(texture, origin=Origin3d(mouseX, mouseY, 0)),
            TexelCopyBufferInfo(rbuffer),
            [1, 1, 1],
        )
        self.device.queue.submit([encoder.finish()])
        result = read_buffer(rbuffer, np.uint32, size=read_size)
        return result

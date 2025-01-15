
import webgpu
from webgpu.webgpu_api import BufferUsage

class Binding:
    VERTICES = 90
    NORMALS = 91
    INDICES = 92

class GeometryRenderObject(webgpu.RenderObject):
    vertex_entry_point: str = "vertexGeo"
    fragment_entry_point: str = "fragmentGeo"
    n_vertices: int = 3

    def __init__(self, gpu, geo, label="Geometry"):
        self.geo = geo
        super().__init__(gpu, label=label)

    def get_bounding_box(self):
        return self.bounding_box

    def get_bindings(self):
        import numpy as np
        bindings = [
            *self.gpu.camera.get_bindings(),
            *self.gpu.u_mesh.get_bindings(),
        ]
        vis_data = self.geo._visualizationData()
        self.bounding_box = (vis_data["min"], vis_data["max"])
        verts = vis_data["vertices"].flatten()
        self.n_instances = len(verts) // 9
        normals = vis_data["normals"].flatten()
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
            bindings.append(webgpu.BufferBinding(binding, b))
        return bindings

    def get_shader_code(self):
        shader_code = webgpu.read_shader_file("colormap.wgsl", webgpu.__file__)
        for shader in ["uniforms", "shader", "geometry", "eval", "clipping"]:
            shader_code += webgpu.read_shader_file(f"{shader}.wgsl", __file__)

        shader_code += self.gpu.camera.get_shader_code()
        shader_code += self.gpu.light.get_shader_code()
        return shader_code

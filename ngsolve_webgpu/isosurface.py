import numpy as np
from webgpu import (
    BufferBinding,
    Colormap,
    Clipping,
    read_shader_file,
)
from webgpu.utils import UniformBinding, uniform_from_array
from .clipping import ClippingCF
from .cf import CFRenderer


class IsoSurfaceRenderObject(ClippingCF):
    compute_shader = "isosurface/compute.wgsl"
    vertex_entry_point = "vertex_isosurface"
    fragment_entry_point = "fragment_isosurface"

    def __init__(self, func_data, levelset_data):
        super().__init__(func_data)
        self.levelset = levelset_data
        self.levelset.need_3d = True
        self.colormap = Colormap()
        self.clipping = Clipping()
        self.subdivision = 0

    def get_shader_code(self, compute=False):
        code = super().get_shader_code(compute=compute)
        if not compute:
            code += read_shader_file("isosurface/render.wgsl", __file__)
        return code

    def update(self, timestamp):
        if timestamp == self._timestamp:
            return
        self.uniform_subdiv = uniform_from_array(
            np.array([self.subdivision], dtype=np.uint32)
        )
        self.levelset.update(timestamp)
        self.levelset_buffer = self.levelset.get_buffers()["data_3d"]
        super().update(timestamp)

    def get_bindings(self, compute=False):
        bindings = super().get_bindings(compute)
        if compute:
            bindings.append(UniformBinding(27, self.uniform_subdiv))
        bindings += [
            BufferBinding(26, self.levelset_buffer),
        ]
        return bindings


class NegativeSurfaceRenderer(CFRenderer):
    def __init__(self, functiondata, levelsetdata):
        super().__init__(functiondata, label="NegativeSurfaceRenderer")
        self.fragment_entry_point = "fragmentCheckLevelset"
        self.levelset = levelsetdata

    def update(self, timestamp):
        if timestamp == self._timestamp:
            return
        self.levelset.update(timestamp)
        buffers = self.levelset.get_buffers()
        self.levelset_buffer = buffers["data_2d"]
        super().update(timestamp)

    def get_bindings(self):
        return super().get_bindings() + [BufferBinding(80, self.levelset_buffer)]

    def get_shader_code(self):
        return super().get_shader_code() + read_shader_file(
            "isosurface/negative_surface.wgsl", __file__
        )


class NegativeClippingRenderer(ClippingCF):
    fragment_entry_point = "fragment_neg_clip"

    def __init__(self, data, levelsetdata):
        super().__init__(data)
        self.levelset = levelsetdata
        self.levelset.need_3d = True

    def update(self, timestamp):
        if self._timestamp == timestamp:
            return
        self.levelset.update(timestamp)
        buffers = self.levelset.get_buffers()
        self.levelset_buffer = buffers["data_3d"]
        super().update(timestamp)

    def get_bindings(self, compute=False):
        bindings = super().get_bindings(compute)
        if not compute:
            bindings += [BufferBinding(80, self.levelset_buffer)]
        return bindings

    def get_shader_code(self, compute=False):
        code = super().get_shader_code(compute)
        if not compute:
            code += read_shader_file("isosurface/negative_clipping.wgsl", __file__)
        return code

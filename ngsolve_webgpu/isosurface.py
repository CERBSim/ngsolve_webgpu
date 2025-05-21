import numpy as np
from webgpu import (
    BufferBinding,
    Clipping,
    Colormap,
    read_shader_file,
)
from webgpu.renderer import RenderOptions
from webgpu.utils import UniformBinding, uniform_from_array

from .cf import CFRenderer
from .clipping import ClippingCF


class IsoSurfaceRenderer(ClippingCF):
    compute_shader = "ngsolve/isosurface/compute.wgsl"
    vertex_entry_point = "vertex_isosurface"
    fragment_entry_point = "fragment_isosurface"

    def __init__(
        self,
        func_data,
        levelset_data,
        clipping: Clipping | None = None,
        colormap: Colormap | None = None,
    ):
        super().__init__(func_data, clipping, colormap)
        self.levelset = levelset_data
        self.levelset.need_3d = True
        self.subdivision = 0

    def get_shader_code(self):
        return read_shader_file("ngsolve/isosurface/render.wgsl")

    def update(self, options: RenderOptions):
        self.uniform_subdiv = uniform_from_array(np.array([self.subdivision], dtype=np.uint32))
        self.levelset.update(options)
        self.levelset_buffer = self.levelset.get_buffers()["data_3d"]
        super().update(options)

    def get_bindings(self, compute=False):
        bindings = super().get_bindings(compute)
        if compute:
            bindings.append(UniformBinding(27, self.uniform_subdiv))
        bindings += [
            BufferBinding(26, self.levelset_buffer),
        ]
        return bindings


class NegativeSurfaceRenderer(CFRenderer):
    def __init__(
        self, functiondata, levelsetdata, clipping: Clipping = None, colormap: Colormap = None
    ):
        super().__init__(
            functiondata, label="NegativeSurfaceRenderer", clipping=clipping, colormap=colormap
        )
        self.fragment_entry_point = "fragmentCheckLevelset"
        self.levelset = levelsetdata

    def update(self, options: RenderOptions):
        self.levelset.update(options)
        buffers = self.levelset.get_buffers()
        self.levelset_buffer = buffers["data_2d"]
        super().update(options)

    def get_bindings(self):
        return super().get_bindings() + [BufferBinding(80, self.levelset_buffer)]

    def get_shader_code(self):
        return read_shader_file("ngsolve/isosurface/negative_surface.wgsl")


class NegativeClippingRenderer(ClippingCF):
    fragment_entry_point = "fragment_neg_clip"

    def __init__(self, data, levelsetdata, clipping: Clipping = None, colormap: Colormap = None):
        super().__init__(data, clipping, colormap)
        self.levelset = levelsetdata
        self.levelset.need_3d = True

    def update(self, options):
        self.levelset.update(options)
        buffers = self.levelset.get_buffers()
        self.levelset_buffer = buffers["data_3d"]
        super().update(options)

    def get_bindings(self, compute=False):
        bindings = super().get_bindings(compute)
        if not compute:
            bindings += [BufferBinding(80, self.levelset_buffer)]
        return bindings

    def get_shader_code(self):
        return read_shader_file("ngsolve/isosurface/negative_clipping.wgsl")

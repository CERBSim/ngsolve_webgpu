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
        symmetry=None,
    ):
        super().__init__(func_data, clipping, colormap, symmetry=symmetry)
        self.levelset = levelset_data
        self.levelset.need_3d = True
        self.subdivision = 0
        self._iso_subdivision = None  # None = auto-computed from levelset order

    def get_shader_code(self):
        return read_shader_file("ngsolve/isosurface/render.wgsl")

    def _get_iso_subdivision(self):
        """Compute subdivision level for the isosurface compute shader.

        Uses all-edge-midpoint subdivision (8 sub-tets per level).
        Each level halves all edges, so after k levels each edge
        has 2^k segments.
        """
        if self._iso_subdivision is not None:
            return self._iso_subdivision
        order = self.levelset.order_3d
        if order > 3:
            k = 0
            while (1 << k) < order:
                k += 1
            return k
        elif order > 1:
            return order - 1
        else:
            return 0

    def update(self, options: RenderOptions):
        iso_subdiv = self._get_iso_subdivision()
        self.uniform_subdiv = uniform_from_array(np.array([iso_subdiv], dtype=np.uint32))
        self.levelset.update(options)
        self.levelset_buffer = self.levelset.get_buffers()["data_3d"]
        super().update(options)

        # No render-side subdivision for isosurfaces: the compute shader finds
        # correct zero-crossing positions on sub-tet edges. Render-side subdivision
        # would interpolate between those vertices, going off the surface.
        self.shader_defines["CLIPPING_SUBDIVISION"] = 1
        self.n_vertices = 3

    def get_bindings(self, compute=False, count=False):
        bindings = super().get_bindings(compute, count)
        if compute:
            bindings.append(UniformBinding(27, self.uniform_subdiv))
        bindings += [
            BufferBinding(26, self.levelset_buffer),
        ]
        return bindings


class NegativeSurfaceRenderer(CFRenderer):
    def __init__(
        self, functiondata, levelsetdata, clipping: Clipping = None, colormap: Colormap = None, symmetry=None
    ):
        super().__init__(
            functiondata, label="NegativeSurfaceRenderer", clipping=clipping, colormap=colormap, symmetry=symmetry
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

    def __init__(self, data, levelsetdata, clipping: Clipping = None, colormap: Colormap = None, symmetry=None):
        super().__init__(data, clipping, colormap, symmetry=symmetry)
        self.levelset = levelsetdata
        self.levelset.need_3d = True

    def update(self, options):
        self.levelset.update(options)
        buffers = self.levelset.get_buffers()
        self.levelset_buffer = buffers["data_3d"]
        super().update(options)

        # Override subdivision based on function order for smooth rendering
        # on the clipping plane (positions stay correct since the plane is flat).
        order = max(self.data.order_3d, self.levelset.order_3d)
        if order > 3:
            func_subdiv = (order + 2) // 3 + 1
        elif order > 1:
            func_subdiv = 3
        else:
            func_subdiv = 1
        clip_subdiv = self.shader_defines.get("CLIPPING_SUBDIVISION", 1)
        subdiv = max(func_subdiv, clip_subdiv)
        self.shader_defines["CLIPPING_SUBDIVISION"] = subdiv
        self.n_vertices = 3 * subdiv * subdiv

    def get_bindings(self, compute=False, count=False):
        bindings = super().get_bindings(compute, count)
        if not compute:
            bindings += [BufferBinding(80, self.levelset_buffer)]
        return bindings

    def get_shader_code(self):
        return read_shader_file("ngsolve/isosurface/negative_clipping.wgsl")

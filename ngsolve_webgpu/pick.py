"""Interpret WebGPU pick events for NGSolve mesh renderers."""

import ctypes as ct
import numpy as np
from webgpu.uniforms import UniformBase


class HighlightUniforms(UniformBase):
    _binding = 57
    _fields_ = [
        ("renderer_id", ct.c_uint32),
        ("element_id", ct.c_uint32),
        ("region_index", ct.c_uint32),
        ("_pad", ct.c_uint32),
    ]

    def __init__(self, **kwargs):
        super().__init__(renderer_id=0, element_id=0xFFFFFFFF, region_index=0xFFFFFFFF, **kwargs)


class MeshPickResult:
    """Rich pick result for mesh-based renderers.

    Decodes the raw SelectEvent user_data (2 x uint32) written by
    select2dElement / select3dElement / select_clipping shaders:
      channel 2 = element instance id
      channel 3 = region index (0-based)
    """

    def __init__(self, event, mesh, camera, kind="surface"):
        self.event = event
        self.kind = kind  # "surface", "volume", or "clipping"
        self.element_nr = int(event.uint32[0])
        self.region_index = int(event.uint32[1])
        self.world_pos = event.calculate_position(camera)

        # Derive region name: surface uses boundaries, volume/clipping uses materials
        try:
            if kind == "surface":
                names = mesh.GetBoundaries()
            else:
                names = mesh.GetMaterials()
            self.region_name = names[self.region_index] if self.region_index < len(names) else f"region {self.region_index}"
        except Exception:
            self.region_name = f"region {self.region_index}"

    def evaluate(self, cf, mesh):
        """Evaluate a CoefficientFunction at the picked world position."""
        try:
            mip = mesh(*self.world_pos)
            return np.array(cf(mip))
        except Exception:
            return None

    @property
    def kind_label(self):
        """Short label for display."""
        return {"surface": "Surf", "volume": "Vol", "clipping": "Clip"}.get(self.kind, self.kind)

    def __repr__(self):
        pos = self.world_pos
        return (
            f"MeshPickResult(kind={self.kind}, el={self.element_nr}, "
            f"region={self.region_name}, "
            f"pos=({pos[0]:.4g}, {pos[1]:.4g}, {pos[2]:.4g}))"
        )

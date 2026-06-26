from .mesh import (
    MeshData,
    MeshWireframe2d,
    MeshElements2d,
    MeshElements3d,
    MeshSegments,
    MeshIdentifications,
)
from .entity_numbers import EntityNumbers
from .cf import FunctionData, CFRenderer
from .facet_cf import FacetFunctionData, FacetCFRenderer, FacetCFRenderer3D
from .pick import MeshPickResult, HighlightUniforms, GeoPickResult
from .clipping import ClippingCF
from .lic import ClippingLIC, SurfaceLIC, LineIntegralConvolution
from .isolines import IsolineSettings, IsolineRenderer, ClippingIsolineRenderer
from webgpu.colormap import Colorbar, Colormap
from webgpu.clipping import Clipping
from .geometry import GeometryRenderer
from .vectors import SurfaceVectors, ClippingVectors
from .symmetry import Symmetry


from webgpu.utils import register_shader_directory as _register_shader_directory
from pathlib import Path as _Path

_register_shader_directory("ngsolve", _Path(__file__).parent / "shaders")

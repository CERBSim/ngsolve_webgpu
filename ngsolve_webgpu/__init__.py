
from .mesh import MeshData, MeshWireframe2d, MeshElements2d, MeshElements3d
from .cf import FunctionData, CFRenderer
from webgpu.colormap import Colorbar


from webgpu.utils import register_shader_directory as _register_shader_directory
from pathlib import Path as _Path

_register_shader_directory("ngsolve", _Path(__file__).parent / "shaders")

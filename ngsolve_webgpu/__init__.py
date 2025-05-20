from webgpu.utils import register_shader_directory
from pathlib import Path

register_shader_directory("ngsolve", Path(__file__).parent / "shaders")

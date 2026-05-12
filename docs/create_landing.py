"""Generate the documentation landing-page scene.

Solves a small linear-elasticity problem on a Π-shaped frame, fixes the
bottom face and applies a downward load on the two top faces. Visualizes
the deformed shape coloured by von Mises stress and overlays force
arrows on the loaded faces.

Run once to (re)generate ``docs/_static/landing.html``::

    WEBGPU_EXPORTING=1 python docs/create_landing.py

The generated HTML embeds the JS engine plus the serialized scene blob,
so it works as a standalone interactive widget.
"""

import base64
import os

os.environ["WEBGPU_EXPORTING"] = "1"

from netgen.occ import Box, Pnt, Glue, OCCGeometry, Axis, X, Y, Z
from ngsolve import (
    Mesh, H1, VectorH1, GridFunction, BilinearForm, LinearForm,
    InnerProduct, Sym, Grad, Trace, Id, CF, IfPos, x, y, z, ds, dx,
)

from ngsolve_webgpu import MeshData, MeshWireframe2d
from ngsolve_webgpu.cf import FunctionData, CFRenderer
from ngsolve_webgpu.vectors import SurfaceVectors
from webgpu.colormap import Colorbar, Colormap
from webgpu.jupyter import Draw
from webgpu.engine import engine_js

# --------------------------------------------------------------------- #
# Π-frame geometry: horizontal base + two vertical posts.
#
#   z=5  ┌──┐    ┌──┐         ← two top faces (loaded)
#        │  │    │  │
#        │  │    │  │
#   z=1  ├──┼────┼──┤
#        │   base   │
#   z=0  └──────────┘         ← single bottom face (fixed)
# --------------------------------------------------------------------- #

base   = Box(Pnt(0, 0, 0), Pnt(5, 1, 1))
left   = Box(Pnt(0, 0, 1), Pnt(1, 1, 5))
right  = Box(Pnt(4, 0, 1), Pnt(5, 1, 5))

shape = Glue([base, left, right])

# Tag faces by location.
for f in shape.faces:
    cz = f.center.z
    cx = f.center.x
    if abs(cz - 0.0) < 1e-3:
        f.name = "fix"
    elif abs(cz - 5.0) < 1e-3 and (cx < 1.0 or cx > 4.0):
        f.name = "load"

mesh = Mesh(OCCGeometry(shape).GenerateMesh(maxh=0.5))

# --------------------------------------------------------------------- #
# Linear elasticity (small-strain, isotropic, plane-3D).
# --------------------------------------------------------------------- #

E, nu = 200.0, 0.3
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))

def strain(u): return Sym(Grad(u))
def stress(u): return 2 * mu * strain(u) + lam * Trace(strain(u)) * Id(3)

fes = VectorH1(mesh, order=2, dirichlet="fix")
u, v = fes.TnT()
gfu = GridFunction(fes)

a = BilinearForm(InnerProduct(stress(u), strain(v)) * dx).Assemble()

# Downward load on the two top faces (force per unit area).
load = CF((0, 0, -2.0))
f = LinearForm(load * v * ds(definedon=mesh.Boundaries("load"))).Assemble()

gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

# Von Mises stress: sqrt(3/2 * dev(sigma) : dev(sigma))
sig = stress(gfu)
dev = sig - (Trace(sig) / 3) * Id(3)
von_mises = (1.5 * InnerProduct(dev, dev)) ** 0.5

# Amplify deformation so it's visible in the scene.
deformation = 5.0 * gfu

# Force arrows: nonzero only on the loaded surface.
arrow_cf = mesh.BoundaryCF({"load": CF((0, 0, -1))}, default=CF((0, 0, 0)))

# --------------------------------------------------------------------- #
# Build the scene
# --------------------------------------------------------------------- #

mesh_data = MeshData(mesh)
mesh_data.deformation_data = FunctionData(mesh_data, deformation, order=2)

vm_data = FunctionData(mesh_data, von_mises, order=2)
colormap = Colormap()
cfr = CFRenderer(vm_data, colormap=colormap)

wf = MeshWireframe2d(mesh_data)

arrow_data = FunctionData(mesh_data, arrow_cf, order=1)
arrows = SurfaceVectors(arrow_data, grid_size=14, scale_by_value=True)

scene = Draw([wf, cfr, arrows, Colorbar(colormap)], width=900, height=480)

# --------------------------------------------------------------------- #
# Emit standalone HTML snippet
# --------------------------------------------------------------------- #

blob_b64 = base64.b64encode(scene.export()).decode()

canvas_id = "landing_canvas"
html = f"""\
<div style="width:100%; max-width:900px; height:480px; margin:0 auto 1em auto; position:relative;">
<canvas id="{canvas_id}" width="900" height="480" style="width:100%; height:100%; border-radius:8px; display:block; background:#f5f5f7;"></canvas>
<script>
{engine_js}
RenderEngine.create("{canvas_id}", "{blob_b64}");
</script>
</div>
<p style="text-align:center; font-size:0.9em; color:#666; margin-top:0.25em;">
Linear elasticity on a Π-frame — fixed at the base, loaded on the two
top faces. Colour = von Mises stress, arrows = applied traction.
Drag to rotate, scroll to zoom.
</p>
"""

out = os.path.join(os.path.dirname(__file__), "_static", "landing.html")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    fh.write(html)
print(f"Written to {out}")

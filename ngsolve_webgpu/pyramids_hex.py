from netgen.occ import *
from ngsolve import *
ngsglobals.msg_level = 0

box = unit_cube.shape
box2 = unit_cube.shape.Move((0, 0, 1.1))

box.faces.Max(Z).Identify(box.faces.Min(Z), "boxZ", 3)
box.faces.Max(X).Identify(box.faces.Min(X), "boxX", 3)
box2.faces.Max(Z).Identify(box2.faces.Min(Z), "box2Z", 3)

box_around = Box((-0.1, -0.1, -0.1), (1.1, 1.1, 2.2)) - (box + box2)

# all element types
#shape = Glue([box, box2, box_around])

# only hexes
#shape = box

# only pyramids + tets
#shape = box_around

# only prisms
#shape = box2

mesh = shape.GenerateMesh(maxh=.2)

Draw(mesh, draw_vol=True)
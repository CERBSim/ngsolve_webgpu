
from webgpu.shapes import ShapeRenderer, generate_cylinder

class ContactPairs(ShapeRenderer):
    def __init__(self, mesh, contact, thickness=0.001,
                 color=[0.5, 0.0, 1.0, 1.0], **kwargs):
        self.mesh = mesh
        self.contact = contact
        self.thickness = thickness
        cyl = generate_cylinder(8, thickness, 1.,
                                top_face=True,
                                bottom_face=False)
        super().__init__(cyl, colors=color, **kwargs)
        self.scale_mode = ShapeRenderer.SCALE_Z

    def update(self, options):
        data = self.contact._GetWebguiData()["position"]
        # data is startpoint1, endpoint1, startpoint2, endpoint2, ...
        # points always have length 3
        self.positions = [data[i:i+3] for i in range(0, len(data), 6)]
        self.directions = [[data[i+3+j] - data[i+j] for
                            j in range(3)] for i in range(0, len(data), 6)]
        # Ensure minimum length for visibility
        for i in range(len(self.directions)//3):
            length = sum(self.directions[i*3+j]**2 for j in range(3))**0.5
            if length < self.thickness:
                self.directions[i*3+2] = self.thickness

        
        self.values = [0] * (len(self.positions)//3)
        super().update(options)

    def get_bounding_box(self):
        pmin, pmax = self.mesh.ngmesh.bounding_box
        return ([pmin[0], pmin[1], pmin[2]], [pmax[0], pmax[1], pmax[2]])
        
    

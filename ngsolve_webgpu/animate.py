from webgpu.render_object import RenderObject
import ngsolve as ngs


class Animation(RenderObject):
    def __init__(self, child):
        super().__init__()
        self.child = child
        self.data = child.data
        self.time_index = -1
        self.max_time = -1
        self.gfs = set()
        self.parameters = dict()
        f = self.data.cf
        self.crawl_function(f)
        # initial solution
        self.add_time(initial=True)
        self.store = True

    def update(self, timestamp):
        self.child.options = self.options
        self.child.update(timestamp)

    def get_bounding_box(self):
        return self.child.get_bounding_box()

    def crawl_function(self, f):
        if f is None:
            return
        if isinstance(f, ngs.GridFunction):
            self.gfs.add(f)
        elif isinstance(f, ngs.Parameter) or isinstance(f, ngs.ParameterC):
            self.parameters[f] = []
        else:
            for c in f.data["childs"]:
                self.crawl_function(c)

    def add_time(self, initial=False):
        self.max_time += 1
        self.time_index = self.max_time
        for gf in self.gfs:
            gf.AddMultiDimComponent(gf.vec)
        for par, vals in self.parameters.items():
            vals.append(par.Get())
        if not initial:
            self.slider.max(self.max_time)
            # set value triggers set_time_index
            self.slider.setValue(self.time_index)

    def redraw(self, timestamp: float | None = None):
        if self.store:
            self.add_time()
        else:
            self.child.redraw(timestamp)

    def render(self, encoder):
        self.child.render(encoder)

    def add_options_to_gui(self, gui):
        self.slider = gui.slider(
            0,
            self.set_time_index,
            min=0,
            max=0,
            step=1,
            label="animate",
        )
        self.child.add_options_to_gui(gui)

    def set_time_index(self, time_index):
        self.time_index = time_index
        for gf in self.gfs:
            gf.vec.data = gf.vecs[time_index + 1]
        for p, vals in self.parameters.items():
            p.Set(vals[time_index])
        self.child.redraw()

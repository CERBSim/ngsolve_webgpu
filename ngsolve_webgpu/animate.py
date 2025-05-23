from webgpu.renderer import BaseRenderer
import ngsolve as ngs


class Animation(BaseRenderer):
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
        self.store = True
        self._last_rendered_time_index = -1

    def update(self, options):
        self.child.update(options)

    def create_render_pipeline(self, options):
        self.child.create_render_pipeline(options)

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

    def add_time(self):
        self.max_time += 1
        self.time_index = self.max_time
        for gf in self.gfs:
            gf.AddMultiDimComponent(gf.vec)
        for par, vals in self.parameters.items():
            vals.append(par.Get())
        if hasattr(self, "slider"):
            self.slider.max(self.max_time)
            # set value triggers set_time_index
            # self.slider.setValue(self.time_index)

    def render(self, options):
        time_index = self.time_index
        if self._last_rendered_time_index != time_index:
            for gf in self.gfs:
                gf.vec.data = gf.vecs[time_index + 1]
            for p, vals in self.parameters.items():
                p.Set(vals[time_index])
            self.child.data._timestamp = -1
            self.child.set_needs_update()
            self.child._update_and_create_render_pipeline(options)
        self.child.render(options)
        self._last_rendered_time_index = time_index

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
        # how to trigger update and re-render here?

    @property
    def needs_update(self):
        return self.child.needs_update() or super().needs_update()

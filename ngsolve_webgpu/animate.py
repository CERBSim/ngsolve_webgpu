from webgpu.render_object import RenderObject
import ngsolve as ngs
import webgpu.proxy as proxy


class Animation(RenderObject):
    def __init__(self, child):
        super().__init__()
        self.child = child
        self.data = child.data
        self.time_index = 0
        self.functions = []
        self.function_data = []
        self.parameter_data = []
        self.gfs = set()
        self.parameters = set()
        f = self.data.cf
        self.add_function(f, setup=True)
        self.store = True

    def update(self):
        self.child.options = self.options
        self.child.update()

    def get_bounding_box(self):
        return self.child.get_bounding_box()

    def add_function(self, function, setup):
        self.functions.append(function)
        gf_data = []
        pdata = []

        def crawl_children(f):
            if f is None:
                return
            if isinstance(f, ngs.GridFunction):
                if setup:
                    self.gfs.add(f)
                gf_data.append(f.vec.Copy())
            elif isinstance(f, ngs.Parameter) or isinstance(f, ngs.ParameterC):
                if setup:
                    self.parameters.add(f)
                pdata.append(f.Get())
            else:
                for c in f.data["childs"]:
                    crawl_children(c)

        self.function_data.append(gf_data)
        self.parameter_data.append(pdata)
        crawl_children(function)

    def redraw(self, timestamp: float | None = None):
        if self.store:
            self.time_index += 1
            self.add_function(self.data.cf, setup=False)
            self.slider.max(self.time_index)
            # set value triggers set_time_index
            self.slider.setValue(self.time_index)
        else:
            self.child.redraw(timestamp)

    def render(self, encoder):
        self.child.render(encoder)

    def add_options_to_gui(self, gui):
        self.slider = gui.slider(
            0,
            self.set_time_index,
            min=0,
            max=len(self.functions) - 1,
            step=1,
            label="animate",
        )
        self.child.add_options_to_gui(gui)

    def set_time_index(self, time_index):
        self.time_index = time_index
        self.data.cf = self.functions[time_index]
        for gf, data in zip(self.gfs, self.function_data[time_index]):
            gf.vec.data = data
        for p, pdata in zip(self.parameters, self.parameter_data[time_index]):
            p.Set(pdata)
        self.child.redraw()
        self.options.render_function()

from webgpu.renderer import MultipleRenderer


class Animation(MultipleRenderer):
    """Wrap a renderer and let users scrub through stored timesteps.

    The wrapper crawls the underlying CoefficientFunction tree for
    GridFunction and Parameter objects. Calling ``add_time()`` snapshots
    the current state. A slider in the GUI selects which frame to render.

    During HTML/JS export (``WEBGPU_EXPORTING=1``) the wrapper additionally
    captures the per-frame contents of the child renderer's data buffers
    and emits a ``time_animation`` interaction so the exported page also
    has a working slider — without any Python at runtime.
    """

    def __init__(self, child):
        super().__init__([child])
        self.child = child
        self.data = child.data
        self.time_index = 0
        self.max_time = -1
        self.gfs = set()
        self.parameters = dict()
        self._last_rendered_time_index = -1
        self.crawl_function(self.data.cf)

    # ------------------------------------------------------------------ #
    # Pipeline plumbing
    # ------------------------------------------------------------------ #

    def get_bounding_box(self):
        return self.child.get_bounding_box()

    def _update_and_create_render_pipeline(self, options):
        # Fully delegate so the child's gpu_objects (colormap, settings, ...)
        # are initialized through the standard recursion path.
        self.child._update_and_create_render_pipeline(options)

    def render(self, options):
        time_index = self.time_index
        if self._last_rendered_time_index != time_index:
            self._apply_time(time_index)
            self.child.data._timestamp = -1
            self.child.set_needs_update()
            self.child._update_and_create_render_pipeline(options)
            self._last_rendered_time_index = time_index
        self.child.render(options)

    def render_opaque(self, options):
        # MultipleRenderer.render() splits opaque/transparent — keep our
        # custom render() logic by overriding both halves to delegate.
        if self.child.transparent:
            return
        self.render(options)

    def render_transparent(self, options):
        if not self.child.transparent:
            return
        self.render(options)

    @property
    def transparent(self):
        return self.child.transparent

    def select(self, options, x, y):
        self.child.select(options, x, y)

    # ------------------------------------------------------------------ #
    # Time control
    # ------------------------------------------------------------------ #

    def crawl_function(self, f):
        if f is None:
            return
        import ngsolve as ngs
        if isinstance(f, ngs.GridFunction):
            self.gfs.add(f)
        elif isinstance(f, (ngs.Parameter, ngs.ParameterC)):
            self.parameters[f] = []
        else:
            for c in f.data["childs"]:
                self.crawl_function(c)

    def add_time(self):
        """Snapshot the current GridFunction values and parameter values."""
        self.max_time += 1
        self.time_index = self.max_time
        for gf in self.gfs:
            gf.AddMultiDimComponent(gf.vec)
        for par, vals in self.parameters.items():
            vals.append(par.Get())
        if hasattr(self, "slider"):
            self.slider.max(self.max_time)

    def _apply_time(self, time_index):
        for gf in self.gfs:
            gf.vec.data = gf.vecs[time_index + 1]
        for p, vals in self.parameters.items():
            p.Set(vals[time_index])

    def set_time_index(self, time_index):
        self.time_index = int(time_index)

    def add_options_to_gui(self, gui):
        self.slider = gui.slider(
            0,
            self.set_time_index,
            min=0,
            max=max(self.max_time, 0),
            step=1,
            label="animate",
        )
        self.child.add_options_to_gui(gui)

    # ------------------------------------------------------------------ #
    # Export integration (WEBGPU_EXPORTING=1)
    # ------------------------------------------------------------------ #

    # Names of buffers on the child renderer that hold per-frame data.
    # CFRenderer caches its function-value buffers in `_buffers` under
    # these keys. Add more here as new renderers gain animation support.
    _ANIMATED_BUFFER_KEYS = ("data_2d", "data_3d")

    def get_export_interactions(self, options, buffer_registry):
        """Capture per-frame buffer contents and emit a time_animation interaction.

        Re-evaluates the child for every stored frame, reads back the
        affected GPU buffers, and stores them as raw frame buffers in the
        registry. The JS engine's ``time_animation`` handler uploads the
        right snapshot to the live buffer when the slider moves.
        """
        from webgpu.utils import read_buffer
        from webgpu.export.format import ExportInteraction

        if self.max_time < 0:
            return []

        child_buffers = getattr(self.child, "_buffers", None)
        if not child_buffers:
            return []

        # Identify animated buffers that the child actually uses and that
        # the registry already knows about (i.e. they're bound somewhere).
        targets = []
        for key in self._ANIMATED_BUFFER_KEYS:
            buf = child_buffers.get(key)
            if buf is None:
                continue
            try:
                buf_id = buffer_registry.get_id(buf)
            except KeyError:
                continue
            targets.append({"key": key, "proxy": buf, "buffer_id": buf_id, "frames": []})

        if not targets:
            return []

        # Remember current state so we can restore it after capture.
        saved_index = self.time_index

        for i in range(self.max_time + 1):
            self._apply_time(i)
            self.child.data._timestamp = -1
            self.child.set_needs_update()
            self.child._update_and_create_render_pipeline(options)
            for t in targets:
                data = bytes(read_buffer(t["proxy"]))
                frame_id = buffer_registry.add_raw_buffer("frame", data)
                t["frames"].append(frame_id)

        # Restore live state.
        self._apply_time(saved_index)
        self.child.data._timestamp = -1
        self.child.set_needs_update()
        self.child._update_and_create_render_pipeline(options)
        self._last_rendered_time_index = saved_index

        return [
            ExportInteraction(
                type="time_animation",
                buffer_id=targets[0]["buffer_id"],  # primary (also in `targets`)
                config={
                    "label": "Animation",
                    "targets": [
                        {"buffer_id": t["buffer_id"], "frames": t["frames"]}
                        for t in targets
                    ],
                },
            )
        ]

"""Per-region visibility (draw alpha) shared between renderers.

Used like ``Clipping``/``Colormap``: one instance per scene/tab, assigned to the
renderers that should respect it (``renderer.region_visibility = rv``). Holds a
single storage buffer of f32 alphas with the layout::

    [bitcast(n_vol), bitcast(n_surf), vol_alpha[n_vol], surf_alpha[n_surf]]

``vol`` alphas are indexed by the 3D element region index (netgen material
number - 1), ``surf`` alphas by the 2D element index (face descriptor
number - 1 on 3D meshes, material number - 1 on 2D meshes). Out-of-range
indices count as visible, so partial arrays are safe.

Renderers only compile the visibility check into their shaders when an
instance is assigned (``REGION_VISIBILITY`` define); without one nothing
changes — no extra binding, no pipeline permutation. An alpha of exactly 0
culls the element in the vertex/compute stage (before rasterization, i.e.
cheaper than a fragment ``discard``); fractional alphas are reserved for
future ghost rendering.

The GPU buffer is allocated once with the sizes of the first ``set_alphas``
call and updated in place afterwards, so live/export engines can keep a
stable reference to it (it acts as a compute-pass trigger there).
"""

import numpy as np

from webgpu.utils import BufferBinding, buffer_from_array, write_array_to_buffer


class Binding:
    REGION_ALPHA = 34


class RegionVisibility:
    """Shared per-region alpha buffer; see module docstring for the layout."""

    def __init__(self, vol_alphas=None, surf_alphas=None):
        self._vol = np.asarray(
            [] if vol_alphas is None else vol_alphas, dtype=np.float32
        )
        self._surf = np.asarray(
            [] if surf_alphas is None else surf_alphas, dtype=np.float32
        )
        self._buffer = None
        # Called after every alpha change; the owning scene hooks re-render /
        # recompute here.
        self.callbacks = []

    @property
    def vol_alphas(self):
        return self._vol

    @property
    def surf_alphas(self):
        return self._surf

    @property
    def buffer(self):
        """The GPU buffer (created on first access)."""
        return self._ensure_buffer()

    def _data(self):
        header = np.array([len(self._vol), len(self._surf)], dtype=np.uint32)
        return np.concatenate(
            (header.view(np.float32), self._vol, self._surf)
        )

    def _ensure_buffer(self):
        if self._buffer is None:
            self._buffer = buffer_from_array(
                self._data(), label="region_alpha", use_cache=False
            )
        return self._buffer

    def set_alphas(self, vol=None, surf=None):
        """Set per-region alphas (None keeps the current array).

        The array sizes are fixed by the first call that creates the GPU
        buffer; later calls must pass arrays of the same length so the buffer
        can be updated in place (live engines hold a stable reference to it).
        """
        if vol is not None:
            vol = np.asarray(vol, dtype=np.float32)
            if self._buffer is not None and len(vol) != len(self._vol):
                raise ValueError(
                    f"vol alpha count changed ({len(self._vol)} -> {len(vol)}) "
                    "after the GPU buffer was created"
                )
            self._vol = vol
        if surf is not None:
            surf = np.asarray(surf, dtype=np.float32)
            if self._buffer is not None and len(surf) != len(self._surf):
                raise ValueError(
                    f"surf alpha count changed ({len(self._surf)} -> {len(surf)}) "
                    "after the GPU buffer was created"
                )
            self._surf = surf
        if self._buffer is not None:
            write_array_to_buffer(self._buffer, self._data())
        for cb in self.callbacks:
            cb()

    def get_bindings(self):
        return [BufferBinding(Binding.REGION_ALPHA, self._ensure_buffer())]

    @staticmethod
    def apply(renderer, defines: dict):
        """Add the REGION_VISIBILITY define if ``renderer`` has an instance
        assigned. Helper for the renderers' ``update`` methods."""
        if getattr(renderer, "region_visibility", None) is not None:
            defines["REGION_VISIBILITY"] = 1

    @staticmethod
    def bindings(renderer):
        """Bindings contributed by ``renderer.region_visibility`` (empty when
        none is assigned). Helper for the renderers' ``get_bindings``."""
        rv = getattr(renderer, "region_visibility", None)
        return rv.get_bindings() if rv is not None else []


class RegionState:
    """Region visibility intent and the alphas derived from it.

    Holds the hidden codim-0 region names and per-boundary overrides
    (``True`` = force show, ``False`` = force hide, absent = auto) and derives
    the alpha arrays for :class:`RegionVisibility`. A boundary face is drawn
    iff at least one adjacent material is visible; free-standing surfaces
    stay visible. Regions are addressed by name, repeated names toggle
    together. Indexing as in :class:`RegionVisibility`.

    *mesh* is a ``MeshData`` (or anything with ``dim``, ``material_names``,
    ``boundary_names``, ``boundary_adjacency``); an ngsolve mesh is accepted
    and wrapped.
    """

    def __init__(self, mesh):
        if not hasattr(mesh, "material_names"):
            from .mesh import MeshBuffers

            mesh = MeshBuffers.get_shared(mesh)
        self.dim = mesh.dim
        self.hidden = set()
        self.overrides = {}

        self.materials = list(mesh.material_names)
        if self.dim == 3:
            self.boundaries = list(mesh.boundary_names)
            self.fd_doms = list(mesh.boundary_adjacency)
            # one boundary region per face descriptor (el2d index == fd number)
            n = min(len(self.boundaries), len(self.fd_doms))
            self.boundaries = self.boundaries[:n]
            self.fd_doms = self.fd_doms[:n]
        else:
            self.boundaries = []
            self.fd_doms = []
        self._cache_key = None

    @property
    def unique_materials(self):
        return list(dict.fromkeys(self.materials))

    @property
    def unique_boundaries(self):
        return list(dict.fromkeys(self.boundaries))

    def material_name(self, index):
        """Region name for a 0-based material index (None if out of range)."""
        if 0 <= index < len(self.materials):
            return self.materials[index]
        return None

    def vol_alphas(self):
        return np.array(
            [0.0 if m in self.hidden else 1.0 for m in self.materials],
            dtype=np.float32,
        )

    def surf_alphas(self):
        # cached: a regions panel queries this once per boundary row
        key = (frozenset(self.hidden), frozenset(self.overrides.items()))
        if key != self._cache_key:
            self._alphas = self._compute_surf_alphas()
            self._effective = {n for n, a in zip(self.boundaries, self._alphas) if a > 0}
            self._cache_key = key
        return self._alphas.copy()

    def _compute_surf_alphas(self):
        if self.dim != 3:
            # 2D: the drawn surface elements are the materials
            return self.vol_alphas()
        vol = self.vol_alphas()
        nmat = len(vol)
        alphas = np.empty(len(self.fd_doms), dtype=np.float32)
        for i, (name, doms) in enumerate(zip(self.boundaries, self.fd_doms)):
            ov = self.overrides.get(name)
            if ov is not None:
                alphas[i] = 1.0 if ov else 0.0
                continue
            doms = [d for d in doms if 1 <= d <= nmat]
            alphas[i] = 1.0 if (not doms or any(vol[d - 1] > 0 for d in doms)) else 0.0
        return alphas

    def material_visible(self, name):
        return name not in self.hidden

    def boundary_effective(self, name):
        """Whether any boundary region with this name is currently drawn."""
        self.surf_alphas()
        return name in self._effective

    def visible_boundary_names(self):
        alphas = self.surf_alphas()
        return list(dict.fromkeys(
            n for i, n in enumerate(self.boundaries) if alphas[i] > 0
        ))

    def any_hidden(self):
        """True if the state hides anything relative to the default."""
        if self.hidden & set(self.materials):
            return True
        return any(self.overrides.get(n) is False for n in self.boundaries)

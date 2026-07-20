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

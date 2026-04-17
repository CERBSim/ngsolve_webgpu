"""Symmetry rendering support via GPU instance multiplication.

Renders symmetric copies of a mesh/field by multiplying draw instances
and applying transform matrices in the vertex shader.
"""

import numpy as np
from webgpu.utils import buffer_from_array, uniform_from_array, write_array_to_buffer, BufferBinding, UniformBinding


class Symmetry:
    """Defines geometric symmetry operations for rendering full solutions
    computed on a reduced domain.

    Usage:
        sym = Symmetry()
        sym.mirror_x()   # 2 copies
        sym.mirror_y()   # 4 copies (group closure)
        renderer = CFRenderer(data, symmetry=sym)

    For fields with antisymmetric behavior under mirrors, specify parity:
        sym_odd = sym.with_parity([-1, 1])  # odd under mirror_x, even under mirror_y
    """

    BINDING_TRANSFORMS = 120
    BINDING_INFO = 121

    def __init__(self):
        self._generators = []  # list of (matrix_4x4, label)
        self._transforms = None  # cached: list of (matrix_4x4, det_sign, generator_mask)
        self._gpu_buffer = None
        self._info_buffer = None
        self._parity = None  # per-generator parity, default all +1

    def mirror_x(self):
        """Add mirror symmetry in the YZ plane (x -> -x)."""
        self._add_generator(np.diag([-1.0, 1.0, 1.0, 1.0]))
        return self

    def mirror_y(self):
        """Add mirror symmetry in the XZ plane (y -> -y)."""
        self._add_generator(np.diag([1.0, -1.0, 1.0, 1.0]))
        return self

    def mirror_z(self):
        """Add mirror symmetry in the XY plane (z -> -z)."""
        self._add_generator(np.diag([1.0, 1.0, -1.0, 1.0]))
        return self

    def with_parity(self, parity):
        """Return a new Symmetry with per-generator parity for scalar fields.

        Args:
            parity: list of +1/-1, one per generator (in order of mirror_x/y/z calls).
                    -1 means the scalar field is antisymmetric under that mirror.

        Returns:
            A new Symmetry object with the specified parity.
        """
        assert len(parity) == len(self._generators), \
            f"parity length {len(parity)} != number of generators {len(self._generators)}"
        s = Symmetry()
        s._generators = self._generators.copy()
        s._transforms = None
        s._gpu_buffer = None
        s._info_buffer = None
        s._parity = list(parity)
        return s

    @property
    def n_copies(self):
        """Number of symmetry copies (including original)."""
        return len(self._get_transforms())

    def _add_generator(self, matrix):
        self._generators.append(matrix)
        self._transforms = None  # invalidate cache
        self._gpu_buffer = None

    def _get_transforms(self):
        """Build group closure from generators. Returns list of (matrix, det_sign, gen_mask)."""
        if self._transforms is not None:
            return self._transforms

        # Each transform is tagged with a bitmask of which generators compose it
        # (matrix, det_sign, generator_bitmask)
        result = [(np.eye(4), 1.0, 0)]

        for gi, gen in enumerate(self._generators):
            new = []
            gen_det = np.sign(np.linalg.det(gen[:3, :3]))
            for mat, det, mask in result:
                combined = gen @ mat
                combined_det = gen_det * det
                combined_mask = mask | (1 << gi)
                # Check if this transform already exists
                if not any(np.allclose(combined, r[0]) for r in result + new):
                    new.append((combined, combined_det, combined_mask))
            result.extend(new)

        self._transforms = result
        return result

    def _get_gpu_data(self):
        """Pack transforms into GPU buffer format.

        Buffer layout per transform (80 bytes):
            mat4x4<f32>  transform    (64 bytes, column-major)
            f32          det_sign     (4 bytes)
            f32          value_sign   (4 bytes)
            f32          _pad1        (4 bytes)
            f32          _pad2        (4 bytes)
        """
        transforms = self._get_transforms()
        parity = self._parity or [1] * len(self._generators)

        data = []
        for mat, det_sign, gen_mask in transforms:
            # Column-major layout for WGSL mat4x4
            for col in range(4):
                for row in range(4):
                    data.append(mat[row, col])
            # det_sign
            data.append(det_sign)
            # value_sign: product of parities for active generators
            value_sign = 1.0
            for gi in range(len(self._generators)):
                if gen_mask & (1 << gi):
                    value_sign *= parity[gi]
            data.append(value_sign)
            # padding
            data.append(0.0)
            data.append(0.0)

        return np.array(data, dtype=np.float32)

    def get_bindings(self, n_elements):
        """Get GPU bindings for symmetry data.

        Args:
            n_elements: original instance count (before multiplication)

        Returns:
            List of bindings for slots 120 (transforms) and 121 (info).
        """
        gpu_data = self._get_gpu_data()
        self._gpu_buffer = buffer_from_array(gpu_data, label="symmetry_transforms", reuse=self._gpu_buffer)

        info = np.array([self.n_copies, n_elements, 0, 0], dtype=np.uint32)
        self._info_buffer = uniform_from_array(info, label="symmetry_info", reuse=self._info_buffer)

        return [
            BufferBinding(self.BINDING_TRANSFORMS, self._gpu_buffer),
            UniformBinding(self.BINDING_INFO, self._info_buffer),
        ]

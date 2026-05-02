import numpy as np
import numpy.testing as npt
import pytest

from ngsolve_webgpu.cf import (
    vandermonde_1d,
    vandermonde_trig,
    vandermonde_hex,
    vandermonde_prism,
    get_hex_intrule,
    get_prism_intrule,
)

ORDERS = [1, 2, 3, 4]


class TestVandermonde:
    # --- DOF counts ---

    @pytest.mark.parametrize("order", ORDERS)
    def test_1d_shape(self, order):
        V = vandermonde_1d(order)
        n = order + 1
        assert V.shape == (n, n)

    @pytest.mark.parametrize("order", ORDERS)
    def test_trig_shape(self, order):
        V = vandermonde_trig(order)
        n = (order + 1) * (order + 2) // 2
        assert V.shape == (n, n)

    @pytest.mark.parametrize("order", ORDERS)
    def test_hex_shape(self, order):
        V = vandermonde_hex(order)
        n = (order + 1) ** 3
        assert V.shape == (n, n)

    @pytest.mark.parametrize("order", ORDERS)
    def test_prism_shape(self, order):
        V = vandermonde_prism(order)
        n = (order + 1) * (order + 2) // 2 * (order + 1)
        assert V.shape == (n, n)

    # --- Roundtrip: sample poly at nodes, inv-Vandermonde to coeffs, Vandermonde back to values ---

    @pytest.mark.parametrize("order", ORDERS)
    def test_1d_roundtrip(self, order):
        Vinv = vandermonde_1d(order)
        V = np.linalg.inv(Vinv)
        n = order
        nodes = np.array([i / n for i in range(n + 1)])
        # polynomial: f(t) = 1 + 2t + 3t^2 (truncated to fit order)
        f = sum(c * nodes**k for k, c in enumerate([1, 2, 3]) if k <= order)
        coeffs = Vinv @ f
        npt.assert_allclose(V @ coeffs, f, atol=1e-12)

    @pytest.mark.parametrize("order", ORDERS)
    def test_trig_roundtrip(self, order):
        Vinv = vandermonde_trig(order)
        V = np.linalg.inv(Vinv)
        n = order
        # build node coords matching basis_indices ordering
        nodes = []
        for j in range(n + 1):
            for k in range(n + 1 - j):
                x = (n - j - k) / n
                y = k / n
                nodes.append((x, y))
        nodes = np.array(nodes)
        x, y = nodes[:, 0], nodes[:, 1]
        f = 1 + 2 * x + 3 * y
        coeffs = Vinv @ f
        npt.assert_allclose(V @ coeffs, f, atol=1e-12)

    @pytest.mark.parametrize("order", ORDERS)
    def test_hex_roundtrip(self, order):
        Vinv = vandermonde_hex(order)
        V = np.linalg.inv(Vinv)
        n = order
        nodes = []
        for iz in range(n + 1):
            for iy in range(n + 1):
                for ix in range(n + 1):
                    nodes.append((ix / n, iy / n, iz / n))
        nodes = np.array(nodes)
        x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
        f = 1 + x + 2 * y + 3 * z
        coeffs = Vinv @ f
        npt.assert_allclose(V @ coeffs, f, atol=1e-12)

    @pytest.mark.parametrize("order", ORDERS)
    def test_prism_roundtrip(self, order):
        Vinv = vandermonde_prism(order)
        V = np.linalg.inv(Vinv)
        n = order
        nodes = []
        for l in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1 - j):
                    a = n - j - k
                    nodes.append((a / n, k / n, l / n))
        nodes = np.array(nodes)
        x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
        f = 1 + x + 2 * y + 3 * z
        coeffs = Vinv @ f
        npt.assert_allclose(V @ coeffs, f, atol=1e-12)

    # --- Invertibility / condition number ---

    @pytest.mark.parametrize("order", ORDERS)
    def test_1d_condition(self, order):
        V = vandermonde_1d(order)
        assert np.linalg.cond(V) < 1e6

    @pytest.mark.parametrize("order", ORDERS)
    def test_trig_condition(self, order):
        V = vandermonde_trig(order)
        assert np.linalg.cond(V) < 1e6

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_hex_condition(self, order):
        V = vandermonde_hex(order)
        assert np.linalg.cond(V) < 1e8

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_prism_condition(self, order):
        V = vandermonde_prism(order)
        assert np.linalg.cond(V) < 1e8

    # --- Integration rule point counts ---

    @pytest.mark.parametrize("order", ORDERS)
    def test_hex_intrule_npoints(self, order):
        ir = get_hex_intrule(order)
        assert len(ir) == (order + 1) ** 3

    @pytest.mark.parametrize("order", ORDERS)
    def test_prism_intrule_npoints(self, order):
        ir = get_prism_intrule(order)
        assert len(ir) == (order + 1) * (order + 2) // 2 * (order + 1)

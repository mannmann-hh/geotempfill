"""Tests for baseline interpolation methods."""

from __future__ import annotations

import numpy as np
import pytest

from geotempfill.baselines import (
    haversine_km,
    idw_fill,
    mean_fill,
    temporal_mean_fill,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tensor_with_known_means(seed=0):
    """Tensor where each (variable, station) has a clearly distinct mean.

    Used so that filling unobserved entries with the per-(v, s) mean
    is exactly recoverable when the mean-bearing rows are all observed.
    """
    rng = np.random.default_rng(seed)
    n_vars, n_times, n_stations = 2, 6, 4
    base = np.zeros((n_vars, n_times, n_stations))
    for v in range(n_vars):
        for s in range(n_stations):
            base[v, :, s] = (v + 1) * 10 + s + rng.standard_normal(n_times) * 0.01
    return base


# ---------------------------------------------------------------------------
# mean_fill
# ---------------------------------------------------------------------------

class TestMeanFill:
    def test_preserves_observed_entries(self):
        T = _tensor_with_known_means()
        rng = np.random.default_rng(0)
        mask = rng.random(T.shape) > 0.4
        # Ensure every (v, s) has at least one observation
        mask[:, 0, :] = True
        out = mean_fill(T, mask)
        np.testing.assert_array_equal(out[mask], T[mask])

    def test_fills_with_per_station_mean(self):
        T = _tensor_with_known_means()
        mask = np.ones_like(T, dtype=bool)
        # Hide a single entry; it should come back as the column mean.
        mask[0, 3, 2] = False
        out = mean_fill(T, mask)
        expected = T[0, mask[0, :, 2], 2].mean()
        assert abs(out[0, 3, 2] - expected) < 1e-10

    def test_falls_back_to_global_when_station_has_no_data(self):
        T = _tensor_with_known_means()
        mask = np.ones_like(T, dtype=bool)
        # Station 0 is completely missing for variable 0.
        mask[0, :, 0] = False
        out = mean_fill(T, mask)
        # Filler must equal the global mean of the *observed* variable-0 data
        # (i.e. across stations 1..3).
        expected = T[0, :, 1:].mean()
        np.testing.assert_allclose(out[0, :, 0], expected)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mean_fill(np.zeros((2, 3, 4)), np.ones((2, 3), dtype=bool))


# ---------------------------------------------------------------------------
# temporal_mean_fill
# ---------------------------------------------------------------------------

class TestTemporalMeanFill:
    def test_fills_with_cross_section_mean(self):
        T = _tensor_with_known_means()
        mask = np.ones_like(T, dtype=bool)
        mask[1, 2, 3] = False  # hide one entry
        out = temporal_mean_fill(T, mask)
        expected = T[1, 2, mask[1, 2, :]].mean()
        assert abs(out[1, 2, 3] - expected) < 1e-10

    def test_preserves_observed_entries(self):
        T = _tensor_with_known_means()
        rng = np.random.default_rng(1)
        mask = rng.random(T.shape) > 0.3
        mask[:, :, 0] = True  # at least one obs per time step
        out = temporal_mean_fill(T, mask)
        np.testing.assert_array_equal(out[mask], T[mask])


# ---------------------------------------------------------------------------
# IDW
# ---------------------------------------------------------------------------

class TestIDW:
    def test_haversine_known_distance(self):
        # Roughly 111 km per degree of latitude near the equator.
        d = haversine_km(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        assert 110.0 < float(d) < 112.0

    def test_haversine_zero_for_same_point(self):
        d = haversine_km(np.array([45.0, 7.0]), np.array([45.0, 7.0]))
        assert float(d) == 0.0

    def test_idw_recovers_constant_field(self):
        # If every observed value is the same constant, IDW should
        # also fill missing entries with that constant.
        T = np.full((1, 3, 5), 7.0)
        mask = np.ones_like(T, dtype=bool)
        mask[0, 1, 2] = False  # hide one cell
        coords = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]]
        )
        out = idw_fill(T, mask, coords=coords, power=2.0)
        assert abs(out[0, 1, 2] - 7.0) < 1e-10

    def test_idw_closer_station_dominates(self):
        # Two donor stations: a "near" one with value 0, a "far" one with
        # value 100. The IDW estimate should be close to 0.
        T = np.zeros((1, 1, 3))
        T[0, 0, 0] = 0.0     # donor near
        T[0, 0, 1] = 100.0   # donor far
        T[0, 0, 2] = 999.0   # target (will be hidden)
        mask = np.array([[[True, True, False]]])
        coords = np.array(
            [[0.0, 0.0], [0.0, 10.0], [0.0, 0.001]]
            # target is at (0, 0.001), almost on top of donor 0.
        )
        out = idw_fill(T, mask, coords=coords, power=2.0)
        assert abs(out[0, 0, 2]) < 1.0   # nowhere near 100

    def test_idw_preserves_observed_entries(self):
        rng = np.random.default_rng(0)
        T = rng.standard_normal((2, 4, 5))
        mask = rng.random(T.shape) > 0.3
        mask[:, :, 0] = True  # at least one observed donor per (v, t)
        coords = rng.uniform(0, 50, size=(5, 2))
        out = idw_fill(T, mask, coords=coords)
        np.testing.assert_allclose(out[mask], T[mask])

    def test_invalid_power_raises(self):
        T = np.zeros((1, 1, 2))
        mask = np.ones_like(T, dtype=bool)
        coords = np.zeros((2, 2))
        with pytest.raises(ValueError):
            idw_fill(T, mask, coords=coords, power=0.0)

    def test_shape_check(self):
        T = np.zeros((1, 1, 3))
        mask = np.ones_like(T, dtype=bool)
        with pytest.raises(ValueError):
            idw_fill(T, mask, coords=np.zeros((4, 2)))   # wrong S

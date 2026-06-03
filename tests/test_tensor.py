"""Tests for the long-table -> 3D tensor builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geotempfill.tensor import WeatherTensor, build_tensor


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

def _make_long_df(n_stations=3, n_days=5, missing_rate=0.2, seed=0):
    """Generate a long-format DataFrame with random gaps."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for s in range(n_stations):
        for d in dates:
            if rng.random() < missing_rate:
                continue
            rows.append({
                "station": f"USW{s:05d}",
                "date": d,
                "TMAX": float(rng.uniform(-5, 30)),
                "TMIN": float(rng.uniform(-15, 20)),
            })
    return pd.DataFrame(rows)


def _make_station_meta(n_stations=3):
    return pd.DataFrame(
        {
            "latitude": [40.0 + 0.1 * i for i in range(n_stations)],
            "longitude": [-100.0 + 0.1 * i for i in range(n_stations)],
            "name": [f"Station {i}" for i in range(n_stations)],
        },
        index=[f"USW{s:05d}" for s in range(n_stations)],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildTensor:
    def test_basic_shape_and_axes(self):
        df = _make_long_df(n_stations=3, n_days=4, missing_rate=0.0)
        t = build_tensor(df, variables=["TMAX", "TMIN"], freq="D")
        assert isinstance(t, WeatherTensor)
        assert t.shape == (2, 4, 3)
        assert t.variables == ["TMAX", "TMIN"]
        assert len(t.times) == 4
        assert len(t.stations) == 3

    def test_complete_data_has_no_missing(self):
        df = _make_long_df(n_stations=4, n_days=10, missing_rate=0.0)
        t = build_tensor(df, variables=["TMAX", "TMIN"], freq="D")
        assert t.missing_rate == 0.0
        assert t.mask.all()
        assert np.isfinite(t.data).all()

    def test_missing_rate_matches_input(self):
        df = _make_long_df(n_stations=4, n_days=20, missing_rate=0.5, seed=7)
        t = build_tensor(df, variables=["TMAX", "TMIN"], freq="D")
        # The mask should reflect the truly observed entries; allow a bit
        # of slack since the random missing rate is per-row, not per-cell.
        assert 0.3 < t.missing_rate < 0.7

    def test_values_round_trip_through_pivot(self):
        """A non-missing value in the long table must reappear at the
        right (variable, time, station) position in the tensor."""
        df = pd.DataFrame({
            "station": ["A", "A", "B"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "TMAX": [10.0, 11.0, 5.0],
            "TMIN": [0.0, 1.0, -2.0],
        })
        t = build_tensor(
            df, variables=["TMAX", "TMIN"], freq="D",
        )
        # Stations are sorted alphabetically -> A is index 0, B is index 1.
        assert t.stations == ["A", "B"]
        assert t.data[0, 0, 0] == 10.0     # TMAX, day 0, station A
        assert t.data[0, 1, 0] == 11.0     # TMAX, day 1, station A
        assert t.data[0, 0, 1] == 5.0      # TMAX, day 0, station B
        assert np.isnan(t.data[0, 1, 1])   # B has no day-1 record
        assert t.mask[0, 0, 0]
        assert not t.mask[0, 1, 1]

    def test_station_coords_are_carried_through(self):
        df = _make_long_df(n_stations=3, n_days=3, missing_rate=0.0)
        meta = _make_station_meta(3)
        t = build_tensor(
            df, variables=["TMAX"], freq="D", station_coords=meta,
        )
        assert t.station_coords is not None
        assert list(t.station_coords.index) == t.stations
        assert "latitude" in t.station_coords.columns

    def test_invalid_variable_raises(self):
        df = _make_long_df(n_stations=2, n_days=2, missing_rate=0.0)
        with pytest.raises(ValueError):
            build_tensor(df, variables=["NOT_A_COLUMN"])

    def test_aggregates_duplicates_with_mean(self):
        df = pd.DataFrame({
            "station": ["A", "A"],
            "date": ["2024-01-01", "2024-01-01"],
            "TMAX": [10.0, 12.0],
        })
        t = build_tensor(df, variables=["TMAX"], freq="D", aggfunc="mean")
        assert t.data[0, 0, 0] == 11.0

    def test_unknown_aggfunc_raises(self):
        df = _make_long_df(n_stations=2, n_days=2, missing_rate=0.0)
        with pytest.raises(ValueError):
            build_tensor(df, variables=["TMAX"], freq="D", aggfunc="weird")

    def test_explicit_time_axis(self):
        df = _make_long_df(n_stations=2, n_days=3, missing_rate=0.0)
        explicit_times = pd.date_range("2024-01-01", periods=10, freq="D")
        t = build_tensor(
            df, variables=["TMAX"], freq="D", times=explicit_times,
        )
        assert len(t.times) == 10
        # Days 3..9 should all be missing for every station.
        assert (~t.mask[0, 3:, :]).all()

    def test_fill_for_algorithm_replaces_nans_with_zeros(self):
        df = _make_long_df(n_stations=3, n_days=5, missing_rate=0.5, seed=1)
        t = build_tensor(df, variables=["TMAX", "TMIN"], freq="D")
        out = t.fill_for_algorithm()
        assert np.isfinite(out).all()
        # Where mask is False, value must now be exactly zero.
        np.testing.assert_array_equal(out[~t.mask], 0.0)
        # Where mask is True, original (non-NaN) values must be preserved.
        np.testing.assert_array_equal(out[t.mask], t.data[t.mask])

    def test_daily_to_monthly_aggregation(self):
        """Daily input + freq='MS' must produce a real monthly mean per
        (var, station, month), not just pick the value from the 1st of
        the month.

        Regression test for the silent ``build_tensor`` bug where non-daily
        frequencies skipped the snap-to-grid step and only kept rows whose
        timestamp happened to match a grid point exactly.
        """
        dates = pd.date_range("2020-01-01", "2020-03-31", freq="D")
        df = pd.DataFrame({
            "station": ["A"] * len(dates),
            "date": dates,
            "TMAX": np.arange(len(dates), dtype=float),
        })
        t = build_tensor(df, variables=["TMAX"], freq="MS")

        assert t.shape == (1, 3, 1)
        # Jan: 31 days valued 0..30  -> mean 15.0
        assert abs(t.data[0, 0, 0] - 15.0) < 1e-9
        # Feb (2020 is leap): 29 days valued 31..59  -> mean 45.0
        assert abs(t.data[0, 1, 0] - 45.0) < 1e-9
        # Mar: 31 days valued 60..90  -> mean 75.0
        assert abs(t.data[0, 2, 0] - 75.0) < 1e-9

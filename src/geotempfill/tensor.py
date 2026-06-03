"""
Tensor construction utilities.

The HaLRTC algorithm operates on dense N-dimensional NumPy arrays with a
boolean observation mask. Geospatial weather observations, however, almost
always arrive as a *long* table (one row per station/time/variable
measurement). This module bridges the two: it pivots a long DataFrame into
a (variable, time, station) third-order tensor and keeps track of which
entries are real observations and which need to be filled.

The 3-mode design mirrors the construction in Liao et al. (2024), where a
``(virus, quarter, city)`` tensor is assembled from sparse surveillance
records. For weather we replace ``virus`` with measurement variable
(temperature, dew point, precipitation, ...), ``quarter`` with month or
day, and ``city`` with weather station.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["WeatherTensor", "NDTensor", "build_tensor", "build_nd_tensor"]


@dataclass
class NDTensor:
    """An N-dimensional tensor with named axes and an observation mask.

    This generic container is useful outside the weather-station setting. For
    example, syndromic surveillance data can be represented as
    ``(syndrome, quarter, city)`` or ``(syndrome, week, city, age_group)``.
    """

    data: np.ndarray
    mask: np.ndarray
    dims: list[str]
    coordinates: dict[str, list]

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def missing_rate(self) -> float:
        return float(1.0 - self.mask.mean())

    def fill_for_algorithm(self) -> np.ndarray:
        out = np.where(self.mask, self.data, 0.0)
        return out.astype(float)


@dataclass
class WeatherTensor:
    """A (variable, time, station) tensor with a boolean observation mask.

    Attributes
    ----------
    data : np.ndarray
        Float array of shape ``(n_vars, n_times, n_stations)`` holding the
        observed values; missing entries are stored as ``NaN`` so that the
        raw object always has a self-describing data array.
    mask : np.ndarray of bool
        Same shape as ``data``. ``True`` for observed entries.
    variables : list of str
        Names of the variables along axis 0.
    times : pd.DatetimeIndex
        Time coordinates along axis 1.
    stations : list of str
        Station identifiers along axis 2.
    station_coords : pd.DataFrame, optional
        Optional table with station metadata (latitude, longitude, name,
        elevation, ...). Index must equal ``stations``. It is carried
        along so the visualization module can map stations geographically
        without a second lookup.
    """

    data: np.ndarray
    mask: np.ndarray
    variables: list
    times: pd.DatetimeIndex
    stations: list
    station_coords: Optional[pd.DataFrame] = None

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def missing_rate(self) -> float:
        """Fraction of entries that are missing (in [0, 1])."""
        return float(1.0 - self.mask.mean())

    def fill_for_algorithm(self) -> np.ndarray:
        """Return ``data`` with NaNs replaced by zeros.

        ``halrtc`` does not look at unmasked entries, but having NaNs in
        the array can poison NumPy reductions; replacing them with zeros
        is the safest input.
        """
        out = np.where(self.mask, self.data, 0.0)
        return out.astype(float)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"WeatherTensor(shape={self.shape}, "
            f"missing={self.missing_rate:.1%}, "
            f"variables={self.variables})"
        )


def build_tensor(
    df: pd.DataFrame,
    *,
    variables: Sequence[str],
    time_col: str = "date",
    station_col: str = "station",
    freq: str = "D",
    aggfunc: str = "mean",
    station_coords: Optional[pd.DataFrame] = None,
    times: Optional[Iterable] = None,
    stations: Optional[Iterable] = None,
) -> WeatherTensor:
    """Pivot a long-format weather DataFrame into a 3-D tensor.

    Parameters
    ----------
    df : pd.DataFrame
        Long table with at least one row per (station, time) observation.
        Must contain ``time_col``, ``station_col``, and one column per
        entry in ``variables``.
    variables : sequence of str
        Names of the variable columns to stack along axis 0. Order is
        preserved.
    time_col : str, default "date"
        Column holding observation timestamps. Will be parsed by
        :func:`pandas.to_datetime` if not already datetime.
    station_col : str, default "station"
        Column holding station identifiers (string or int).
    freq : str, default "D"
        Pandas frequency string used to build the regular time axis when
        ``times`` is not provided. ``"D"`` for daily, ``"MS"`` for month
        start, ``"QS"`` for quarter start, etc.
    aggfunc : str, default "mean"
        Aggregator passed to :meth:`pandas.DataFrame.pivot_table` for
        cells where multiple records share the same (variable, time,
        station) coordinates. Use ``"mean"`` for safety with raw
        observations or ``"first"`` if duplicates are not expected.
    station_coords : pd.DataFrame, optional
        Optional metadata, indexed by station id. Stored on the result
        for downstream mapping.
    times : iterable of timestamps, optional
        Explicit time axis. If given it overrides ``freq``.
    stations : iterable, optional
        Explicit station axis. If given it overrides the auto-discovered
        station list.

    Returns
    -------
    WeatherTensor
    """
    if not variables:
        raise ValueError("`variables` must contain at least one column name")
    missing_cols = set(variables) - set(df.columns)
    if missing_cols:
        raise ValueError(f"missing variable columns in df: {sorted(missing_cols)}")
    if time_col not in df.columns:
        raise ValueError(f"time column '{time_col}' not found in df")
    if station_col not in df.columns:
        raise ValueError(f"station column '{station_col}' not found in df")

    # --- normalise inputs --------------------------------------------------
    work = df.loc[:, [time_col, station_col] + list(variables)].copy()
    work[time_col] = pd.to_datetime(work[time_col])

    if stations is None:
        station_list = sorted(work[station_col].astype(str).unique().tolist())
    else:
        station_list = [str(s) for s in stations]

    if times is None:
        # Build a regular axis covering the data range
        time_index = pd.date_range(
            work[time_col].min(),
            work[time_col].max(),
            freq=freq,
        )
    else:
        time_index = pd.DatetimeIndex(times)

    n_vars = len(variables)
    n_times = len(time_index)
    n_stations = len(station_list)

    # Map station ids to a fixed integer axis
    station_to_ix = {sid: i for i, sid in enumerate(station_list)}

    data = np.full((n_vars, n_times, n_stations), np.nan, dtype=float)

    # Snap each observation to the nearest position on the regular axis;
    # observations that fall outside the chosen time range or to unknown
    # stations are dropped.
    work["__sid_ix"] = work[station_col].astype(str).map(station_to_ix)
    work = work.dropna(subset=["__sid_ix"])
    work["__sid_ix"] = work["__sid_ix"].astype(int)

# Snap each observation to its corresponding grid point. For daily
    # freq we floor to "D"; for any other freq (MS, QS, W, ...) we use
    # searchsorted so that an observation falling between two grid points
    # is assigned to the most recent grid point that precedes it. This
    # makes the downstream groupby.mean() act as a true bucket
    # aggregation (e.g. daily -> monthly mean).
    if freq.upper().startswith("D"):
        work["__t"] = work[time_col].dt.floor("D")
    else:
        ts = pd.DatetimeIndex(work[time_col])
        grid_pos = time_index.searchsorted(ts, side="right") - 1
        in_range = grid_pos >= 0
        work = work.loc[in_range].copy()
        grid_pos = grid_pos[in_range]
        work["__t"] = time_index[grid_pos]

    t_to_ix = {t: i for i, t in enumerate(time_index)}
    work["__t_ix"] = work["__t"].map(t_to_ix)
    work = work.dropna(subset=["__t_ix"])
    work["__t_ix"] = work["__t_ix"].astype(int)

    # Aggregate duplicates and assign into the dense tensor
    for v_ix, v_name in enumerate(variables):
        sub = work[["__t_ix", "__sid_ix", v_name]].dropna(subset=[v_name])
        if sub.empty:
            continue
        if aggfunc == "mean":
            grouped = sub.groupby(["__t_ix", "__sid_ix"], as_index=False)[v_name].mean()
        elif aggfunc == "first":
            grouped = sub.groupby(["__t_ix", "__sid_ix"], as_index=False)[v_name].first()
        elif aggfunc == "median":
            grouped = sub.groupby(["__t_ix", "__sid_ix"], as_index=False)[v_name].median()
        else:
            raise ValueError(
                f"aggfunc must be one of 'mean', 'first', 'median'; got {aggfunc!r}"
            )
        data[v_ix, grouped["__t_ix"].to_numpy(), grouped["__sid_ix"].to_numpy()] = (
            grouped[v_name].to_numpy()
        )

    mask = ~np.isnan(data)

    # Align the optional station metadata to our station axis
    coords = None
    if station_coords is not None:
        coords = station_coords.reindex([str(s) for s in station_list])

    return WeatherTensor(
        data=data,
        mask=mask,
        variables=list(variables),
        times=time_index,
        stations=station_list,
        station_coords=coords,
    )


def build_nd_tensor(
    df: pd.DataFrame,
    *,
    index_cols: Sequence[str],
    value_col: str,
    coordinates: Optional[dict[str, Iterable]] = None,
    aggfunc: str = "mean",
) -> NDTensor:
    """Build a generic N-dimensional tensor from a long-format table.

    Parameters
    ----------
    df:
        Long table containing one row per observed tensor cell.
    index_cols:
        Columns that define the tensor axes, in axis order.
    value_col:
        Numeric column holding the observed value.
    coordinates:
        Optional explicit coordinate values for one or more axes. Missing axes
        are inferred from the data and sorted.
    aggfunc:
        Aggregation used when duplicate coordinates appear. One of
        ``"mean"``, ``"first"``, or ``"median"``.
    """
    if not index_cols:
        raise ValueError("index_cols must contain at least one column")
    missing_cols = set(index_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"missing index columns in df: {sorted(missing_cols)}")
    if value_col not in df.columns:
        raise ValueError(f"value column '{value_col}' not found in df")
    if aggfunc not in {"mean", "first", "median"}:
        raise ValueError("aggfunc must be one of 'mean', 'first', or 'median'")

    coordinates = coordinates or {}
    work = df.loc[:, list(index_cols) + [value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])

    axis_values: dict[str, list] = {}
    axis_maps: dict[str, dict] = {}

    for col in index_cols:
        if col in coordinates:
            values = list(coordinates[col])
        else:
            values = sorted(work[col].dropna().unique().tolist())

        if not values:
            raise ValueError(f"axis '{col}' has no coordinate values")

        axis_values[col] = values
        axis_maps[col] = {value: i for i, value in enumerate(values)}

    shape = tuple(len(axis_values[col]) for col in index_cols)
    data = np.full(shape, np.nan, dtype=float)

    ix_cols = []
    for col in index_cols:
        ix_col = f"__{col}_ix"
        work[ix_col] = work[col].map(axis_maps[col])
        ix_cols.append(ix_col)

    work = work.dropna(subset=ix_cols)
    for ix_col in ix_cols:
        work[ix_col] = work[ix_col].astype(int)

    if aggfunc == "mean":
        grouped = work.groupby(ix_cols, as_index=False)[value_col].mean()
    elif aggfunc == "first":
        grouped = work.groupby(ix_cols, as_index=False)[value_col].first()
    else:
        grouped = work.groupby(ix_cols, as_index=False)[value_col].median()

    index = tuple(grouped[ix_col].to_numpy() for ix_col in ix_cols)
    data[index] = grouped[value_col].to_numpy()
    mask = ~np.isnan(data)

    return NDTensor(
        data=data,
        mask=mask,
        dims=list(index_cols),
        coordinates=axis_values,
    )

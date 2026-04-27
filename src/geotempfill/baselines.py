"""
Baseline interpolation methods.

The reference paper (Liao et al., 2024) compares HaLRTC against three
classical methods: Cokriging, hierarchical Bayesian, and a sandwich
spatial estimator. Reimplementing all three rigorously is beyond the
scope of a course project, so this module instead provides three
*lightweight* baselines that capture the same intuition and let us run
side-by-side accuracy comparisons:

- :func:`mean_fill` -- per-(variable, station) climatology. The simplest
  reasonable baseline: replace each missing entry with the long-term mean
  for that station and variable. It captures the *station* signal but is
  blind to time.

- :func:`temporal_mean_fill` -- per-(variable, time) cross-section mean.
  Replaces a missing value with the mean of all observed stations at the
  same time and variable. Captures the *time* signal but is blind to
  station identity.

- :func:`idw_fill` -- inverse-distance weighting over space. For each
  missing (variable, time, station) cell, look at the values measured at
  the same (variable, time) cross-section in nearby stations, weight
  them by ``1 / d^p`` and average. Genuine spatial interpolation, the
  closest analogue to (co)kriging that we can implement without GP
  fitting.

Each function takes a tensor + mask + (optional) coordinates and returns
a fully filled NumPy array of the same shape as the input. They are used
in :mod:`geotempfill.evaluation` to benchmark HaLRTC.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["mean_fill", "temporal_mean_fill", "idw_fill"]


def _safe_mean(values: np.ndarray, mask: np.ndarray, axis) -> np.ndarray:
    """Mean of ``values`` over masked entries along ``axis``.

    Returns NaN where every entry along ``axis`` is masked out.
    """
    masked = np.where(mask, values, 0.0)
    counts = mask.sum(axis=axis)
    sums = masked.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return means


def mean_fill(tensor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill missing entries with the per-(variable, station) mean.

    For each ``(v, s)`` pair, compute the mean of ``tensor[v, :, s]``
    over the times where ``mask[v, :, s]`` is True; broadcast that mean
    across the time axis to fill the missing entries. If a station has
    no observation for a variable, the global mean of that variable is
    used.
    """
    if tensor.shape != mask.shape:
        raise ValueError("tensor and mask must have the same shape")
    if tensor.ndim != 3:
        raise ValueError("mean_fill expects a (var, time, station) tensor")

    # (n_vars, n_stations) per-station climatology
    per_station = _safe_mean(tensor, mask, axis=1)

    # Per-variable global mean as a fallback for stations with no data
    per_var = _safe_mean(tensor, mask, axis=(1, 2))

    # Where per_station is NaN, fall back to per_var
    fallback = np.broadcast_to(per_var[:, None], per_station.shape)
    per_station = np.where(np.isnan(per_station), fallback, per_station)

    # Broadcast across the time axis
    filler = per_station[:, None, :]
    out = np.where(mask, tensor, filler)
    return out


def temporal_mean_fill(tensor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill missing entries with the per-(variable, time) cross-section mean.

    For each ``(v, t)`` pair, compute the mean of all observed stations
    at that variable and time. This is the "what was happening today
    across the network" baseline.
    """
    if tensor.shape != mask.shape:
        raise ValueError("tensor and mask must have the same shape")
    if tensor.ndim != 3:
        raise ValueError("temporal_mean_fill expects a (var, time, station) tensor")

    # (n_vars, n_times) cross-section mean over stations
    cross = _safe_mean(tensor, mask, axis=2)

    # Per-variable mean as fallback for time slices with no obs
    per_var = _safe_mean(tensor, mask, axis=(1, 2))
    cross = np.where(np.isnan(cross), per_var[:, None], cross)

    filler = cross[:, :, None]
    out = np.where(mask, tensor, filler)
    return out


def idw_fill(
    tensor: np.ndarray,
    mask: np.ndarray,
    coords: np.ndarray,
    *,
    power: float = 2.0,
    k: Optional[int] = None,
) -> np.ndarray:
    """Inverse-distance-weighted fill over the station axis.

    For each missing entry ``(v, t, s)``, consider the stations ``s'``
    where ``mask[v, t, s']`` is True and combine their values with weights
    ``w(s, s') = 1 / d(s, s')^power``, where ``d`` is the great-circle
    distance computed via :func:`haversine_km`.

    Parameters
    ----------
    tensor : np.ndarray, shape (V, T, S)
    mask : np.ndarray of bool, same shape
    coords : np.ndarray, shape (S, 2)
        Station coordinates as (latitude, longitude) in degrees. Must be
        ordered to match the station axis of ``tensor``.
    power : float, default 2.0
        IDW exponent. 1.0 = inverse distance, 2.0 = inverse squared.
    k : int, optional
        If set, only the ``k`` nearest observed stations participate in
        each cell's weighted average. ``None`` (default) uses all
        observed stations.

    Returns
    -------
    np.ndarray
        Filled tensor of the same shape as ``tensor``.
    """
    if tensor.shape != mask.shape:
        raise ValueError("tensor and mask must have the same shape")
    if tensor.ndim != 3:
        raise ValueError("idw_fill expects a (var, time, station) tensor")
    if power <= 0:
        raise ValueError("power must be strictly positive")

    n_vars, n_times, n_stations = tensor.shape
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (n_stations, 2):
        raise ValueError(
            f"coords must have shape ({n_stations}, 2), got {coords.shape}"
        )

    # (S, S) station-to-station distance matrix
    dist = haversine_km(coords[:, None, :], coords[None, :, :])
    # Avoid division by zero on the diagonal: a station has zero distance
    # to itself, but it can never be a *donor* if it's missing, so put a
    # tiny floor on the off-diagonal too.
    np.fill_diagonal(dist, np.inf)
    eps = 1e-9
    dist = np.where(dist <= 0, eps, dist)

    # Pre-compute IDW weights once.
    weights_full = 1.0 / np.power(dist, power)  # shape (S, S)

    out = tensor.copy().astype(float)

    for v in range(n_vars):
        for t in range(n_times):
            obs_row = mask[v, t]            # (S,) boolean
            if not obs_row.any():
                continue
            values_row = tensor[v, t]       # (S,)

            # For each target station s, restrict to observed donors
            # weights[s, :] -> only columns where obs_row is True.
            donor_w = weights_full[:, obs_row]  # (S, n_obs)
            donor_v = values_row[obs_row]       # (n_obs,)

            if k is not None and donor_w.shape[1] > k:
                # Keep the top-k weights per row
                top_idx = np.argpartition(-donor_w, kth=k - 1, axis=1)[:, :k]
                rows = np.arange(donor_w.shape[0])[:, None]
                donor_w_k = donor_w[rows, top_idx]
                donor_v_k = donor_v[top_idx]
                num = (donor_w_k * donor_v_k).sum(axis=1)
                den = donor_w_k.sum(axis=1)
            else:
                num = donor_w @ donor_v
                den = donor_w.sum(axis=1)

            with np.errstate(invalid="ignore", divide="ignore"):
                est = np.where(den > 0, num / den, np.nan)

            target = ~obs_row
            out[v, t, target] = est[target]

    # Stations with absolutely no donors at a given (v, t) will have NaN
    # left over -> fall back to the per-variable mean as a last resort.
    if np.isnan(out).any():
        per_var = _safe_mean(tensor, mask, axis=(1, 2))
        for v in range(n_vars):
            sl = out[v]
            sl[np.isnan(sl)] = per_var[v]
            out[v] = sl

    return out


def haversine_km(
    a: np.ndarray, b: np.ndarray, *, radius_km: float = 6371.0088
) -> np.ndarray:
    """Great-circle distance between two arrays of (lat, lon) points.

    Inputs may be arbitrarily broadcastable. Coordinates are in degrees.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lat1 = np.deg2rad(a[..., 0])
    lon1 = np.deg2rad(a[..., 1])
    lat2 = np.deg2rad(b[..., 0])
    lon2 = np.deg2rad(b[..., 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))

"""
Spatial interpolation methods for station-indexed weather tensors.

The routines here are intended as optional comparison methods for HaLRTC.
They operate on the same ``(variable, time, station)`` tensors used by the
rest of the package and preserve observed entries exactly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .baselines import idw_fill

__all__ = ["kriging_fill", "cokriging_fill"]


def _haversine_km(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_stations, 2)")

    lat = np.deg2rad(coords[:, 0])
    lon = np.deg2rad(coords[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    h = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def _covariance(distance_km: np.ndarray, *, sill: float, range_km: float, nugget: float):
    return sill * np.exp(-distance_km / range_km) + nugget * (distance_km == 0.0)


def _default_range_km(dist: np.ndarray) -> float:
    positive = dist[np.isfinite(dist) & (dist > 0.0)]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def kriging_fill(
    tensor: np.ndarray,
    mask: np.ndarray,
    coords: np.ndarray,
    *,
    range_km: Optional[float] = None,
    nugget: float = 1e-6,
    min_points: int = 3,
    fallback: str = "idw",
    idw_power: float = 2.0,
) -> np.ndarray:
    """
    Fill missing entries with ordinary kriging over the station axis.

    This is a lightweight ordinary-kriging baseline. For every
    ``(variable, time)`` slice, observed stations are used as donors and
    missing stations are predicted from an exponential covariance model.
    The covariance sill is estimated from the observed values in that slice;
    the range defaults to the median station distance unless supplied.

    Parameters
    ----------
    tensor, mask:
        Arrays with shape ``(variable, time, station)``.
    coords:
        Station coordinates as ``(latitude, longitude)`` in the same order as
        the tensor station axis.
    range_km:
        Optional covariance range. If omitted, the median station distance is
        used.
    nugget:
        Small diagonal stabilizer for the kriging system.
    min_points:
        Minimum observed stations required for kriging in a time slice.
    fallback:
        ``"idw"`` uses IDW for slices that cannot be kriged. ``"mean"`` uses
        the per-variable observed mean.
    idw_power:
        Power parameter passed to IDW fallback.
    """
    tensor = np.asarray(tensor, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    coords = np.asarray(coords, dtype=float)

    if tensor.shape != mask.shape:
        raise ValueError("tensor and mask must have the same shape")
    if tensor.ndim != 3:
        raise ValueError("kriging_fill expects a (var, time, station) tensor")
    if coords.shape != (tensor.shape[2], 2):
        raise ValueError(f"coords must have shape ({tensor.shape[2]}, 2)")
    if nugget < 0:
        raise ValueError("nugget must be non-negative")
    if min_points < 2:
        raise ValueError("min_points must be at least 2")
    if fallback not in {"idw", "mean"}:
        raise ValueError("fallback must be either 'idw' or 'mean'")

    n_vars, n_times, _ = tensor.shape
    dist = _haversine_km(coords)
    model_range = float(range_km) if range_km is not None else _default_range_km(dist)
    if model_range <= 0:
        raise ValueError("range_km must be positive")

    if fallback == "idw":
        out = idw_fill(tensor, mask, coords=coords, power=idw_power)
    else:
        out = tensor.copy()
        for v in range(n_vars):
            vals = tensor[v][mask[v]]
            fill_value = float(vals.mean()) if vals.size else np.nan
            out[v][~mask[v]] = fill_value

    for v in range(n_vars):
        for t in range(n_times):
            observed = mask[v, t]
            missing = ~observed
            n_obs = int(observed.sum())

            if n_obs < min_points or not missing.any():
                continue

            y = tensor[v, t, observed]
            if not np.isfinite(y).all():
                continue

            sill = float(np.var(y))
            if sill <= 0.0 or not np.isfinite(sill):
                out[v, t, missing] = float(y.mean())
                continue

            donor_dist = dist[np.ix_(observed, observed)]
            target_dist = dist[np.ix_(missing, observed)]

            cov = _covariance(
                donor_dist,
                sill=sill,
                range_km=model_range,
                nugget=nugget,
            )

            system = np.empty((n_obs + 1, n_obs + 1), dtype=float)
            system[:n_obs, :n_obs] = cov
            system[:n_obs, n_obs] = 1.0
            system[n_obs, :n_obs] = 1.0
            system[n_obs, n_obs] = 0.0

            rhs = np.empty((n_obs + 1, target_dist.shape[0]), dtype=float)
            rhs[:n_obs, :] = _covariance(
                target_dist.T,
                sill=sill,
                range_km=model_range,
                nugget=0.0,
            )
            rhs[n_obs, :] = 1.0

            try:
                weights = np.linalg.solve(system, rhs)[:n_obs, :]
            except np.linalg.LinAlgError:
                weights = np.linalg.lstsq(system, rhs, rcond=None)[0][:n_obs, :]

            out[v, t, missing] = y @ weights

    return np.where(mask, tensor, out)


def _variable_standardization(
    tensor: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.zeros(tensor.shape[0], dtype=float)
    stds = np.ones(tensor.shape[0], dtype=float)
    z = tensor.copy().astype(float)

    for v in range(tensor.shape[0]):
        values = tensor[v][mask[v]]
        if values.size == 0:
            means[v] = 0.0
            stds[v] = 1.0
            z[v] = 0.0
            continue

        means[v] = float(values.mean())
        std = float(values.std())
        stds[v] = std if np.isfinite(std) and std > 0.0 else 1.0
        z[v] = (tensor[v] - means[v]) / stds[v]

    return z, means, stds


def _variable_correlation(tensor_z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n_vars = tensor_z.shape[0]
    corr = np.eye(n_vars, dtype=float)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            valid = mask[i] & mask[j]
            if valid.sum() < 2:
                value = 0.0
            else:
                xi = tensor_z[i][valid]
                xj = tensor_z[j][valid]
                if np.std(xi) == 0.0 or np.std(xj) == 0.0:
                    value = 0.0
                else:
                    value = float(np.corrcoef(xi, xj)[0, 1])
                    if not np.isfinite(value):
                        value = 0.0
            corr[i, j] = value
            corr[j, i] = value

    # Numerical guard: clip to a positive semidefinite correlation matrix.
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-6, None)
    corr_psd = (eigvecs * eigvals) @ eigvecs.T
    scale = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(scale, scale)
    return corr_psd


def cokriging_fill(
    tensor: np.ndarray,
    mask: np.ndarray,
    coords: np.ndarray,
    *,
    range_km: Optional[float] = None,
    nugget: float = 1e-6,
    min_points: int = 5,
    max_points: Optional[int] = 120,
    fallback: str = "kriging",
    idw_power: float = 2.0,
) -> np.ndarray:
    """
    Fill missing entries with an experimental simple cokriging baseline.

    This implementation uses a separable covariance model:

        cov((var_i, station_a), (var_j, station_b))
            = corr(var_i, var_j) * exp(-distance(a, b) / range)

    Variables are standardized before fitting the correlation model and
    predictions are transformed back to the original units. The method is
    useful as a lightweight multivariable spatial baseline, but it is not a
    full linear model of coregionalization or a strict reproduction of a
    geostatistical cokriging workflow.
    """
    tensor = np.asarray(tensor, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    coords = np.asarray(coords, dtype=float)

    if tensor.shape != mask.shape:
        raise ValueError("tensor and mask must have the same shape")
    if tensor.ndim != 3:
        raise ValueError("cokriging_fill expects a (var, time, station) tensor")
    if coords.shape != (tensor.shape[2], 2):
        raise ValueError(f"coords must have shape ({tensor.shape[2]}, 2)")
    if nugget < 0:
        raise ValueError("nugget must be non-negative")
    if min_points < 2:
        raise ValueError("min_points must be at least 2")
    if max_points is not None and max_points < min_points:
        raise ValueError("max_points must be at least min_points")
    if fallback not in {"kriging", "idw"}:
        raise ValueError("fallback must be either 'kriging' or 'idw'")

    n_vars, n_times, n_stations = tensor.shape
    dist = _haversine_km(coords)
    model_range = float(range_km) if range_km is not None else _default_range_km(dist)
    if model_range <= 0:
        raise ValueError("range_km must be positive")

    if fallback == "kriging":
        out = kriging_fill(
            tensor,
            mask,
            coords,
            range_km=model_range,
            nugget=nugget,
            min_points=max(3, min_points),
            idw_power=idw_power,
        )
    else:
        out = idw_fill(tensor, mask, coords=coords, power=idw_power)

    tensor_z, means, stds = _variable_standardization(tensor, mask)
    var_corr = _variable_correlation(tensor_z, mask)
    spatial_cov = np.exp(-dist / model_range)

    for target_var in range(n_vars):
        for t in range(n_times):
            missing_stations = np.flatnonzero(~mask[target_var, t])
            if missing_stations.size == 0:
                continue

            donor_vars, donor_stations = np.nonzero(mask[:, t, :])
            n_donors = donor_vars.size
            if n_donors < min_points:
                continue

            donor_values = tensor_z[donor_vars, t, donor_stations]
            finite = np.isfinite(donor_values)
            donor_vars = donor_vars[finite]
            donor_stations = donor_stations[finite]
            donor_values = donor_values[finite]
            n_donors = donor_vars.size

            if n_donors < min_points:
                continue

            for target_station in missing_stations:
                chosen = np.arange(n_donors)
                if max_points is not None and n_donors > max_points:
                    score = (
                        np.abs(var_corr[target_var, donor_vars])
                        * spatial_cov[target_station, donor_stations]
                    )
                    chosen = np.argpartition(-score, max_points - 1)[:max_points]

                dv = donor_vars[chosen]
                ds = donor_stations[chosen]
                y = donor_values[chosen]

                cov = var_corr[np.ix_(dv, dv)] * spatial_cov[np.ix_(ds, ds)]
                cov = cov + np.eye(cov.shape[0]) * nugget
                rhs = var_corr[target_var, dv] * spatial_cov[target_station, ds]

                try:
                    weights = np.linalg.solve(cov, rhs)
                except np.linalg.LinAlgError:
                    weights = np.linalg.lstsq(cov, rhs, rcond=None)[0]

                pred_z = float(weights @ y)
                out[target_var, t, target_station] = (
                    pred_z * stds[target_var] + means[target_var]
                )

    return np.where(mask, tensor, out)

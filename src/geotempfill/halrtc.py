"""
High-accuracy Low-Rank Tensor Completion (HaLRTC).

Implementation of the HaLRTC algorithm for filling missing values in
multidimensional arrays, following the formulation of Liu et al. (2013)
and the application to surveillance/observational data described by
Liao et al. (2024).

The algorithm assumes an N-th order tensor M with a known set of observed
entries (mask Omega). It seeks a tensor X that:

    minimizes   sum_i alpha_i * || X_(i) ||_*           (sum of nuclear
                                                         norms of the
                                                         mode-i unfoldings)
    subject to  X[Omega] = M[Omega]                      (interpolation
                                                         constraint)

This convex relaxation of the low-Tucker-rank problem is solved with
the Alternating Direction Method of Multipliers (ADMM), as in
Liu et al. (2013), Algorithm 4 ("HaLRTC").

References
----------
Liu, J., Musialski, P., Wonka, P., & Ye, J. (2013). Tensor completion
    for estimating missing values in visual data. IEEE TPAMI, 35(1),
    208-220.
Liao, Y., Shi, Y., Fan, Z., et al. (2024). A new disease mapping method
    for improving data completeness of syndromic surveillance with high
    missing rates. Transactions in GIS, 28, 1869-1882.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "HaLRTCResult",
    "halrtc",
    "svt",
    "unfold",
    "fold",
    "apply_elevation_temperature_correction",
    "inverse_elevation_temperature_correction",
]


# ---------------------------------------------------------------------------
# Tensor unfolding / folding
# ---------------------------------------------------------------------------

def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Mode-n unfolding of a tensor into a matrix."""
    if mode < 0 or mode >= tensor.ndim:
        raise ValueError(f"mode must be in [0, {tensor.ndim - 1}], got {mode}")

    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def fold(matrix: np.ndarray, mode: int, shape: tuple[int, ...]) -> np.ndarray:
    """Fold a mode-n unfolded matrix back into a tensor."""
    full_shape = [shape[mode]] + [
        shape[k] for k in range(len(shape)) if k != mode
    ]
    return np.moveaxis(matrix.reshape(full_shape), 0, mode)


# ---------------------------------------------------------------------------
# Singular Value Thresholding
# ---------------------------------------------------------------------------

def svt(matrix: np.ndarray, tau: float) -> np.ndarray:
    """Singular value thresholding."""
    if tau < 0:
        raise ValueError("tau must be non-negative")

    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)

    keep = s_thresh > 0

    if not np.any(keep):
        return np.zeros_like(matrix)

    return (u[:, keep] * s_thresh[keep]) @ vh[keep, :]


# ---------------------------------------------------------------------------
# Physical helper functions
# ---------------------------------------------------------------------------

def apply_elevation_temperature_correction(
    data: np.ndarray,
    elevation_m: np.ndarray,
    variable_names: list[str],
    variables_to_correct: tuple[str, ...] = ("TMAX", "TMIN"),
    lapse_rate_c_per_km: float = 6.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply elevation-based correction to temperature variables.

    For selected temperature variables:

        T_corrected = T + lapse_rate * elevation_km

    This creates an intermediate reference-elevation representation.
    The returned correction tensor must be saved so that the result can be
    transformed back later.

    Parameters
    ----------
    data:
        Tensor with shape (variable, time, station).

    elevation_m:
        Station elevations in meters, shape (station,).

    variable_names:
        Variable names corresponding to data axis 0.

    variables_to_correct:
        Variables to correct, normally ("TMAX", "TMIN").

    lapse_rate_c_per_km:
        Temperature lapse rate in °C/km.

    Returns
    -------
    data_corr:
        Corrected tensor.

    correction:
        Correction tensor with same shape as data.
    """
    data = np.asarray(data, dtype=float)
    elevation_m = np.asarray(elevation_m, dtype=float)

    if data.ndim != 3:
        raise ValueError("data must have shape (variable, time, station)")

    n_stations = data.shape[2]

    if elevation_m.shape != (n_stations,):
        raise ValueError(
            f"elevation_m must have shape ({n_stations},), "
            f"got {elevation_m.shape}"
        )

    data_corr = data.copy()
    correction = np.zeros_like(data_corr)

    elevation_km = elevation_m / 1000.0

    for var_name in variables_to_correct:
        if var_name not in variable_names:
            continue

        v = variable_names.index(var_name)
        correction[v, :, :] = lapse_rate_c_per_km * elevation_km[None, :]
        data_corr[v, :, :] = data_corr[v, :, :] + correction[v, :, :]

    return data_corr, correction


def inverse_elevation_temperature_correction(
    data_corr: np.ndarray,
    correction: np.ndarray,
) -> np.ndarray:
    """
    Transform elevation-corrected temperature values back to original units.

    This simply performs:

        data_original = data_corr - correction
    """
    if data_corr.shape != correction.shape:
        raise ValueError("data_corr and correction must have the same shape")

    return data_corr - correction


# ---------------------------------------------------------------------------
# Spatial helper functions
# ---------------------------------------------------------------------------

def _haversine_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute great-circle distance matrix between stations.

    coords shape:
        (n_stations, 2)

    coords columns:
        latitude, longitude

    return:
        distance matrix in kilometers
    """
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_stations, 2)")

    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat[:, None])
        * np.cos(lat[None, :])
        * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    return 6371.0 * c


def _inverse_distance_weights(
    coords: np.ndarray,
    power: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Build inverse-distance weight matrix.

    W[i, j] = contribution of station j when estimating station i.
    The diagonal is zero, so a station does not use itself.
    """
    if power <= 0:
        raise ValueError("spatial_power must be positive")

    dist = _haversine_distance_matrix(coords)

    with np.errstate(divide="ignore"):
        weights = 1.0 / np.maximum(dist, eps) ** power

    np.fill_diagonal(weights, 0.0)

    row_sum = weights.sum(axis=1, keepdims=True)

    weights = np.divide(
        weights,
        row_sum,
        out=np.zeros_like(weights),
        where=row_sum > 0,
    )

    return weights


def _apply_spatial_smoothing(
    X: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    *,
    spatial_weight: float,
    station_mode: int,
) -> np.ndarray:
    """
    Apply inverse-distance smoothing only to missing entries.

    For each missing value:

        X_new = (1 - spatial_weight) * X_HaLRTC
                + spatial_weight * X_spatial

    Observed entries are never changed.
    """
    if spatial_weight <= 0:
        return X

    if not (0.0 <= spatial_weight <= 1.0):
        raise ValueError("spatial_weight must be between 0 and 1")

    ndim = X.ndim

    if station_mode < 0:
        station_mode += ndim

    if station_mode < 0 or station_mode >= ndim:
        raise ValueError(f"station_mode must be in [0, {ndim - 1}]")

    n_stations = X.shape[station_mode]

    if weights.shape != (n_stations, n_stations):
        raise ValueError(
            f"weights must have shape ({n_stations}, {n_stations}), "
            f"got {weights.shape}"
        )

    X_last = np.moveaxis(X, station_mode, -1)
    mask_last = np.moveaxis(mask, station_mode, -1)

    original_shape = X_last.shape

    flat_X = X_last.reshape(-1, n_stations)
    flat_mask = mask_last.reshape(-1, n_stations)

    flat_out = flat_X.copy()

    for row_idx in range(flat_X.shape[0]):
        values = flat_X[row_idx]
        observed = flat_mask[row_idx]

        if observed.sum() < 2:
            continue

        W = weights.copy()

        # Only observed training stations are allowed to provide evidence.
        W[:, ~observed] = 0.0

        row_sum = W.sum(axis=1, keepdims=True)

        W = np.divide(
            W,
            row_sum,
            out=np.zeros_like(W),
            where=row_sum > 0,
        )

        spatial_estimate = W @ values

        missing = ~observed

        flat_out[row_idx, missing] = (
            (1.0 - spatial_weight) * values[missing]
            + spatial_weight * spatial_estimate[missing]
        )

    X_out = flat_out.reshape(original_shape)
    X_out = np.moveaxis(X_out, -1, station_mode)

    # Hard constraint: observed entries must remain fixed.
    return np.where(mask, X, X_out)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HaLRTCResult:
    """Output of HaLRTC."""

    completed: np.ndarray
    n_iter: int
    converged: bool
    history: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main HaLRTC algorithm
# ---------------------------------------------------------------------------

def halrtc(
    tensor: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: Optional[np.ndarray] = None,
    rho: float = 1e-3,
    max_iter: int = 500,
    tol: float = 1e-5,
    verbose: bool = False,
    coords: Optional[np.ndarray] = None,
    spatial_weight: float = 0.0,
    spatial_power: float = 2.0,
    station_mode: int = -1,
) -> HaLRTCResult:
    """
    Fill missing entries in a tensor using HaLRTC.

    If coords is provided and spatial_weight > 0, a location-aware
    inverse-distance smoothing step is applied after each HaLRTC update.
    """
    tensor = np.asarray(tensor, dtype=float)

    if tensor.shape != mask.shape:
        raise ValueError("tensor and mask must have the same shape")

    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    if not mask.any():
        raise ValueError("mask must mark at least one observed entry")

    if rho <= 0:
        raise ValueError("rho must be strictly positive")

    if not (0.0 <= spatial_weight <= 1.0):
        raise ValueError("spatial_weight must be between 0 and 1")

    ndim = tensor.ndim

    if station_mode < 0:
        station_mode += ndim

    if station_mode < 0 or station_mode >= ndim:
        raise ValueError(f"station_mode must be in [0, {ndim - 1}]")

    if alpha is None:
        alpha = np.full(ndim, 1.0 / ndim)
    else:
        alpha = np.asarray(alpha, dtype=float)

        if alpha.shape != (ndim,):
            raise ValueError(f"alpha must have length {ndim}")

        if np.any(alpha < 0):
            raise ValueError("alpha entries must be non-negative")

        if not np.isclose(alpha.sum(), 1.0):
            alpha = alpha / alpha.sum()

    spatial_weights = None

    if coords is not None and spatial_weight > 0:
        coords = np.asarray(coords, dtype=float)

        expected_stations = tensor.shape[station_mode]

        if coords.shape != (expected_stations, 2):
            raise ValueError(
                f"coords must have shape ({expected_stations}, 2), "
                f"got {coords.shape}"
            )

        spatial_weights = _inverse_distance_weights(
            coords,
            power=spatial_power,
        )

    obs_mean = float(tensor[mask].mean())
    X = np.where(mask, tensor, obs_mean).astype(float)

    M = [X.copy() for _ in range(ndim)]
    Y = [np.zeros_like(X) for _ in range(ndim)]

    history: list[float] = []
    converged = False
    last_iter = 0

    shape = tensor.shape
    inv_ndim = 1.0 / ndim

    for it in range(1, max_iter + 1):
        last_iter = it
        X_prev = X.copy()

        # 1. Update auxiliary tensors M_n.
        for n in range(ndim):
            tau = alpha[n] / rho
            unfolded = unfold(X + Y[n] / rho, n)
            M[n] = fold(svt(unfolded, tau), n, shape)

        # 2. Update X.
        avg = np.zeros_like(X)

        for n in range(ndim):
            avg += M[n] - Y[n] / rho

        avg *= inv_ndim

        X = np.where(mask, tensor, avg)

        # 3. Optional location-aware smoothing.
        if spatial_weights is not None:
            X = _apply_spatial_smoothing(
                X,
                mask,
                spatial_weights,
                spatial_weight=spatial_weight,
                station_mode=station_mode,
            )

            X = np.where(mask, tensor, X)

        # 4. Dual update.
        for n in range(ndim):
            Y[n] = Y[n] - rho * (M[n] - X)

        # 5. Convergence check.
        denom = np.linalg.norm(X_prev)

        if denom > 0:
            rel_change = np.linalg.norm(X - X_prev) / denom
        else:
            rel_change = np.linalg.norm(X - X_prev)

        history.append(float(rel_change))

        if verbose and (it == 1 or it % 25 == 0):
            print(f"[HaLRTC] iter={it:4d}  rel_change={rel_change:.3e}")

        if rel_change < tol:
            converged = True

            if verbose:
                print(
                    f"[HaLRTC] converged at iter {it} "
                    f"(rel_change={rel_change:.3e})"
                )

            break

    return HaLRTCResult(
        completed=X,
        n_iter=last_iter,
        converged=converged,
        history=history,
    )
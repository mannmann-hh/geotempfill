"""
Lightweight hierarchical Bayesian-style gap filling.

This module intentionally avoids heavy probabilistic-programming dependencies.
The method below is an empirical-Bayes additive baseline for 3-D
spatiotemporal tensors. It estimates variable-specific temporal and station
effects from observed entries and optionally smooths those effects over time
and space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["EmpiricalBayesResult", "empirical_bayes_fill"]


@dataclass
class EmpiricalBayesResult:
    """Output from the empirical-Bayes spatiotemporal filler."""

    completed: np.ndarray
    variable_mean: np.ndarray
    variable_std: np.ndarray
    time_effect: np.ndarray
    station_effect: np.ndarray
    n_iter: int
    history: list[float] = field(default_factory=list)


def _standardize_by_variable(
    tensor: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.zeros(tensor.shape[0], dtype=float)
    stds = np.ones(tensor.shape[0], dtype=float)
    z = tensor.copy().astype(float)

    for v in range(tensor.shape[0]):
        values = tensor[v][mask[v]]
        if values.size == 0:
            z[v] = 0.0
            continue

        means[v] = float(values.mean())
        std = float(values.std())
        stds[v] = std if np.isfinite(std) and std > 0.0 else 1.0
        z[v] = (tensor[v] - means[v]) / stds[v]

    return z, means, stds


def _regularized_mean(
    values: np.ndarray,
    mask: np.ndarray,
    axis: int,
    shrinkage: float,
) -> np.ndarray:
    sums = np.where(mask, values, 0.0).sum(axis=axis)
    counts = mask.sum(axis=axis)
    return sums / np.maximum(counts + shrinkage, 1.0)


def _smooth_time_effect(effect: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0 or effect.shape[1] < 3:
        return effect

    if not 0.0 <= strength <= 1.0:
        raise ValueError("temporal_smoothing must be between 0 and 1")

    smoothed = effect.copy()
    smoothed[:, 1:-1] = (
        0.25 * effect[:, :-2]
        + 0.50 * effect[:, 1:-1]
        + 0.25 * effect[:, 2:]
    )
    return (1.0 - strength) * effect + strength * smoothed


def _distance_weights(
    coords: np.ndarray,
    power: float = 2.0,
    spatial_neighbors: Optional[int] = None,
) -> np.ndarray:
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
    dist = 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))

    with np.errstate(divide="ignore"):
        weights = 1.0 / np.maximum(dist, 1e-9) ** power
    np.fill_diagonal(weights, 0.0)

    if spatial_neighbors is not None:
        if spatial_neighbors < 1:
            raise ValueError("spatial_neighbors must be at least 1")
        if spatial_neighbors < weights.shape[1] - 1:
            keep = np.argpartition(-weights, spatial_neighbors - 1, axis=1)[
                :, :spatial_neighbors
            ]
            sparse = np.zeros_like(weights)
            rows = np.arange(weights.shape[0])[:, None]
            sparse[rows, keep] = weights[rows, keep]
            weights = sparse

    row_sum = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, row_sum, out=np.zeros_like(weights), where=row_sum > 0)


def _smooth_station_effect(
    effect: np.ndarray,
    coords: Optional[np.ndarray],
    strength: float,
    spatial_neighbors: Optional[int],
) -> np.ndarray:
    if strength <= 0.0:
        return effect

    if not 0.0 <= strength <= 1.0:
        raise ValueError("spatial_smoothing must be between 0 and 1")
    if coords is None:
        return effect

    weights = _distance_weights(coords, spatial_neighbors=spatial_neighbors)
    smoothed = effect @ weights.T
    return (1.0 - strength) * effect + strength * smoothed


def empirical_bayes_fill(
    tensor: np.ndarray,
    mask: np.ndarray,
    *,
    coords: Optional[np.ndarray] = None,
    shrinkage: float = 5.0,
    temporal_smoothing: float = 0.20,
    spatial_smoothing: float = 0.10,
    spatial_neighbors: Optional[int] = 8,
    max_iter: int = 100,
    tol: float = 1e-6,
    return_result: bool = False,
) -> np.ndarray | EmpiricalBayesResult:
    """
    Fill missing entries with a variable-specific spatiotemporal additive model.

    The fitted structure is estimated in per-variable standardized units:

        z[v, t, s] = time_effect[v, t] + station_effect[v, s]

    This is a deterministic empirical-Bayes-style baseline. It borrows the
    hierarchical shrinkage idea but does not run MCMC or return posterior
    samples. Observed entries are preserved exactly.
    """
    tensor = np.asarray(tensor, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if tensor.shape != mask.shape:
        raise ValueError("tensor and mask must have the same shape")
    if tensor.ndim != 3:
        raise ValueError("empirical_bayes_fill expects a (var, time, station) tensor")
    if not mask.any():
        raise ValueError("mask must mark at least one observed entry")
    if shrinkage < 0:
        raise ValueError("shrinkage must be non-negative")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    z, variable_mean, variable_std = _standardize_by_variable(tensor, mask)
    n_vars, n_times, n_stations = tensor.shape
    time_eff = np.zeros((n_vars, n_times), dtype=float)
    station_eff = np.zeros((n_vars, n_stations), dtype=float)

    if coords is not None and np.asarray(coords).shape != (n_stations, 2):
        raise ValueError(f"coords must have shape ({n_stations}, 2)")

    history: list[float] = []

    for it in range(1, max_iter + 1):
        prev_time = time_eff.copy()
        prev_station = station_eff.copy()

        residual = z - station_eff[:, None, :]
        time_eff = _regularized_mean(residual, mask, axis=2, shrinkage=shrinkage)
        time_eff = time_eff - time_eff.mean(axis=1, keepdims=True)
        time_eff = _smooth_time_effect(time_eff, temporal_smoothing)

        residual = z - time_eff[:, :, None]
        station_eff = _regularized_mean(residual, mask, axis=1, shrinkage=shrinkage)
        station_eff = station_eff - station_eff.mean(axis=1, keepdims=True)
        station_eff = _smooth_station_effect(
            station_eff,
            coords,
            spatial_smoothing,
            spatial_neighbors,
        )

        delta = np.linalg.norm(time_eff - prev_time) + np.linalg.norm(
            station_eff - prev_station
        )
        scale = np.linalg.norm(prev_time) + np.linalg.norm(prev_station) + 1.0
        rel_change = float(delta / scale)
        history.append(rel_change)

        if rel_change < tol:
            break

    fitted_z = time_eff[:, :, None] + station_eff[:, None, :]
    fitted = fitted_z * variable_std[:, None, None] + variable_mean[:, None, None]
    completed = np.where(mask, tensor, fitted)

    result = EmpiricalBayesResult(
        completed=completed,
        variable_mean=variable_mean,
        variable_std=variable_std,
        time_effect=time_eff,
        station_effect=station_eff,
        n_iter=it,
        history=history,
    )

    if return_result:
        return result
    return completed

"""
Method registry for benchmark-style gap filling.

The registry keeps CLI/application code from growing a long chain of
method-specific ``if`` blocks. Each registered method accepts the same context
dictionary and returns a completed tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .baselines import idw_fill, mean_fill, temporal_mean_fill
from .bayesian import empirical_bayes_fill
from .halrtc import halrtc
from .spatial import cokriging_fill, kriging_fill

__all__ = ["FillMethod", "METHODS", "DEFAULT_METHODS", "run_fill_method"]


@dataclass(frozen=True)
class FillMethod:
    name: str
    label: str
    runner: Callable[..., np.ndarray]


def _run_halrtc(**ctx) -> np.ndarray:
    result = halrtc(
        ctx["data"],
        ctx["mask"],
        rho=ctx.get("rho", 5e-3),
        max_iter=ctx.get("max_iter", 500),
        tol=ctx.get("tol", 1e-5),
        verbose=ctx.get("verbose", False),
    )
    return result.completed


def _run_mean(**ctx) -> np.ndarray:
    return mean_fill(ctx["data"], ctx["mask"])


def _run_temporal_mean(**ctx) -> np.ndarray:
    return temporal_mean_fill(ctx["data"], ctx["mask"])


def _run_idw(**ctx) -> np.ndarray:
    return idw_fill(
        ctx["data"],
        ctx["mask"],
        coords=ctx["coords"],
        power=ctx.get("idw_power", 2.0),
    )


def _run_kriging(**ctx) -> np.ndarray:
    return kriging_fill(
        ctx["data"],
        ctx["mask"],
        coords=ctx["coords"],
        range_km=ctx.get("kriging_range_km"),
        nugget=ctx.get("kriging_nugget", 1e-6),
        min_points=ctx.get("kriging_min_points", 3),
        idw_power=ctx.get("idw_power", 2.0),
    )


def _run_cokriging(**ctx) -> np.ndarray:
    return cokriging_fill(
        ctx["data"],
        ctx["mask"],
        coords=ctx["coords"],
        range_km=ctx.get("kriging_range_km"),
        nugget=ctx.get("kriging_nugget", 1e-6),
        min_points=ctx.get("cokriging_min_points", 5),
        max_points=ctx.get("cokriging_max_points", 120),
        idw_power=ctx.get("idw_power", 2.0),
    )


def _run_empirical_bayes(**ctx) -> np.ndarray:
    return empirical_bayes_fill(
        ctx["data"],
        ctx["mask"],
        coords=ctx.get("coords"),
        shrinkage=ctx.get("bayes_shrinkage", 5.0),
        temporal_smoothing=ctx.get("bayes_temporal_smoothing", 0.20),
        spatial_smoothing=ctx.get("bayes_spatial_smoothing", 0.10),
        spatial_neighbors=ctx.get("bayes_spatial_neighbors", 8),
    )


METHODS = {
    "halrtc": FillMethod("halrtc", "HaLRTC", _run_halrtc),
    "mean": FillMethod("mean", "MeanFill", _run_mean),
    "temporal": FillMethod("temporal", "TempMean", _run_temporal_mean),
    "idw": FillMethod("idw", "IDW", _run_idw),
    "kriging": FillMethod("kriging", "Kriging", _run_kriging),
    "cokriging": FillMethod("cokriging", "Cokriging", _run_cokriging),
    "empirical_bayes": FillMethod(
        "empirical_bayes",
        "EmpBayes",
        _run_empirical_bayes,
    ),
}

DEFAULT_METHODS = ("halrtc", "mean", "temporal", "idw")


def run_fill_method(name: str, **ctx) -> tuple[str, np.ndarray]:
    key = name.lower()
    if key not in METHODS:
        valid = ", ".join(sorted(METHODS))
        raise ValueError(f"unknown method {name!r}; choose from: {valid}")
    method = METHODS[key]
    return method.label, method.runner(**ctx)

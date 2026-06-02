"""
geotempfill - tensor-completion-based gap filling for sparse weather data.

This package implements a NumPy-only version of the High-accuracy Low-Rank
Tensor Completion (HaLRTC) algorithm and applies it to NOAA GHCN-Daily
station observations. It mirrors the methodology of Liao et al. (2024,
*Transactions in GIS*), where HaLRTC is used to recover heavily missing
syndromic surveillance data, but the target domain here is meteorological
observations from a US state.

Quick start
-----------
>>> import geotempfill as gtf
>>> obs, stations = gtf.fetch_state_data("VT", years=[2020, 2021])
>>> tensor = gtf.build_tensor(
...     obs, variables=["TMAX", "TMIN"], freq="MS", station_coords=stations,
... )
>>> result = gtf.halrtc(tensor.fill_for_algorithm(), tensor.mask)
>>> result.completed.shape
(2, 24, 38)

See the example notebook ``examples/usage_example.ipynb`` for a full,
end-to-end walkthrough including baseline comparisons and figures.
"""

from .baselines import idw_fill, mean_fill, temporal_mean_fill
from .bayesian import EmpiricalBayesResult, empirical_bayes_fill
from .data import (
    GhcnStation,
    fetch_state_data,
    fetch_station_data,
    list_stations,
)
from .evaluation import Metrics, hide_random, score
from .halrtc import (
    HaLRTCResult,
    fold,
    halrtc,
    svt,
    unfold,
    apply_elevation_temperature_correction,
    inverse_elevation_temperature_correction,
)
from .spatial import cokriging_fill, kriging_fill
from .tensor import WeatherTensor, build_tensor
from .visualize import (
    plot_convergence,
    plot_method_comparison,
    plot_missing_heatmap,
    plot_station_map,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # algorithm
    "halrtc",
    "HaLRTCResult",
    "svt",
    "unfold",
    "fold",
    # tensor construction
    "build_tensor",
    "WeatherTensor",
    # data
    "GhcnStation",
    "list_stations",
    "fetch_station_data",
    "fetch_state_data",
    # baselines
    "mean_fill",
    "temporal_mean_fill",
    "idw_fill",
    "kriging_fill",
    "cokriging_fill",
    "empirical_bayes_fill",
    "EmpiricalBayesResult",
    # evaluation
    "Metrics",
    "hide_random",
    "score",
    # visualization
    "plot_station_map",
    "plot_missing_heatmap",
    "plot_convergence",
    "plot_method_comparison",
]

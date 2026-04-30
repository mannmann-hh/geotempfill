from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_station_map(stations: pd.DataFrame):
    """Plot station locations using longitude and latitude."""
    required = {"latitude", "longitude"}
    if not required.issubset(stations.columns):
        raise ValueError("stations must contain latitude and longitude columns")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(stations["longitude"], stations["latitude"], s=18)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("GHCN-Daily station locations")
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_missing_heatmap(tensor, variable: int = 0):
    """Plot observed/missing pattern for one variable."""
    missing = ~tensor.mask[variable]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.imshow(missing, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Station")
    ax.set_ylabel("Time index")
    ax.set_title(f"Missing pattern: {tensor.variables[variable]}")
    return fig, ax


def plot_convergence(history):
    """Plot HaLRTC relative change over iterations."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(history) + 1), history)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative change")
    ax.set_title("HaLRTC convergence")
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_method_comparison(results: Mapping[str, object], metric: str = "rmse"):
    """Plot a bar chart comparing methods by one metric."""
    names = list(results.keys())
    values = []

    for name in names:
        item = results[name]
        if isinstance(item, dict):
            values.append(item[metric])
        else:
            values.append(getattr(item, metric))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, values)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Method comparison: {metric.upper()}")
    ax.grid(True, axis="y", alpha=0.3)
    return fig, ax

def plot_station_error_map(
    stations: pd.DataFrame,
    station_errors: pd.Series | pd.DataFrame,
    error_col: str = "error",
):
    """
    Plot station-level prediction error on a longitude-latitude map.

    stations:
        DataFrame with latitude and longitude.
        Index should be station_id, or it should contain station_id column.

    station_errors:
        Series indexed by station_id, or DataFrame with station_id and error_col.
    """
    required = {"latitude", "longitude"}
    if not required.issubset(stations.columns):
        raise ValueError("stations must contain latitude and longitude columns")

    stations_plot = stations.copy()

    if "station_id" in stations_plot.columns:
        stations_plot = stations_plot.set_index("station_id")

    if isinstance(station_errors, pd.Series):
        errors = station_errors.rename(error_col)
    else:
        if "station_id" in station_errors.columns:
            errors = station_errors.set_index("station_id")[error_col]
        else:
            errors = station_errors[error_col]

    stations_plot = stations_plot.join(errors, how="left")

    fig, ax = plt.subplots(figsize=(7, 5))

    sc = ax.scatter(
        stations_plot["longitude"],
        stations_plot["latitude"],
        c=stations_plot[error_col],
        s=45,
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Station-level prediction error")
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(error_col)

    return fig, ax
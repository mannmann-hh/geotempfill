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
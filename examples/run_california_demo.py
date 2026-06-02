"""
End-to-end demo for geotempfill using California GHCN-Daily data.

This script intentionally uses a small subset of stations and years so that
it is suitable for a course-project demo.

Run from the project root:

    python examples/run_california_demo.py

Or, after installing the package:

    python -m examples.run_california_demo

Outputs:
    data/raw/observations_CA.csv
    data/raw/stations_CA.csv
    results/reports/california_demo_metrics.json
    results/figures/california_station_map.png
    results/figures/california_missing_heatmap.png
    results/figures/california_halrtc_convergence.png
    results/figures/california_method_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import geotempfill as gtf
from geotempfill.visualize import plot_station_error_map



def standardize_by_variable(data: np.ndarray, mask: np.ndarray):
    """
    Standardize each variable separately using observed entries only.
    """
    data_std = data.copy().astype(float)
    means = np.zeros(data.shape[0])
    stds = np.ones(data.shape[0])

    for v in range(data.shape[0]):
        observed_values = data[v][mask[v]]
        mean = observed_values.mean()
        std = observed_values.std()

        if std == 0 or np.isnan(std):
            std = 1.0

        means[v] = mean
        stds[v] = std
        data_std[v] = (data[v] - mean) / std

    return data_std, means, stds


def inverse_standardize_by_variable(
    data_std: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
):
    """Convert standardized tensor back to original scale."""
    data = data_std.copy().astype(float)

    for v in range(data.shape[0]):
        data[v] = data_std[v] * stds[v] + means[v]

    return data


def metric_to_dict(m):
    return {
        "rmse": m.rmse,
        "mae": m.mae,
        "r2": m.r2,
        "r": m.r,
        "n": m.n,
    }


def format_fraction_label(fraction: float) -> str:
    """Format a fraction as a filesystem-safe percentage label."""
    pct = fraction * 100.0
    text = f"{pct:.2f}".rstrip("0").rstrip(".")
    text = text.replace(".", "p")
    return f"missing_{text}pct"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the California geotempfill demo with configurable holdout rate."
    )
    parser.add_argument(
        "--hide-fraction",
        type=float,
        default=0.10,
        help="Fraction of originally observed entries to hide for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when selecting held-out test entries.",
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        default=30,
        help="Maximum number of stations to use when downloading fresh data.",
    )
    parser.add_argument(
        "--spatial-weight",
        type=float,
        default=0.10,
        help="Weight for location-aware smoothing inside HaLRTC.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum number of HaLRTC iterations.",
    )
    parser.add_argument(
        "--skip-kriging",
        action="store_true",
        help="Do not run the ordinary kriging baseline.",
    )
    parser.add_argument(
        "--skip-cokriging",
        action="store_true",
        help="Do not run the experimental simple cokriging baseline.",
    )
    parser.add_argument(
        "--kriging-range-km",
        type=float,
        default=None,
        help="Kriging/cokriging covariance range; defaults to median station distance.",
    )
    parser.add_argument(
        "--kriging-nugget",
        type=float,
        default=1e-6,
        help="Small diagonal stabilizer for kriging/cokriging systems.",
    )
    parser.add_argument(
        "--cokriging-max-points",
        type=int,
        default=120,
        help="Maximum donor variable-station observations per cokriging solve.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not 0.0 < args.hide_fraction < 1.0:
        raise ValueError("--hide-fraction must be between 0 and 1")
    if not 0.0 <= args.spatial_weight <= 1.0:
        raise ValueError("--spatial-weight must be between 0 and 1")

    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "raw"
    cache_dir = project_root / "data" / "cache"

    obs_path = raw_dir / "observations_CA.csv"
    stations_path = raw_dir / "stations_CA.csv"

    state = "CA"
    years = [2020, 2021]
    variables = ["TMAX", "TMIN", "ADPT", "ASLP", "AWBT"]
    max_stations = args.max_stations

    hide_fraction = args.hide_fraction
    seed = args.seed

    rho = 5e-3
    max_iter = args.max_iter
    tol = 1e-5

    spatial_weight = args.spatial_weight
    spatial_power = 2.0

    output_label = f"{format_fraction_label(hide_fraction)}_seed{seed}"
    figures_dir = project_root / "results" / "figures" / output_label
    reports_dir = project_root / "results" / "reports" / output_label

    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("geotempfill California demo")
    print("============================================================")
    print(f"State:          {state}")
    print(f"Years:          {years}")
    print(f"Variables:      {variables}")
    print(f"Max stations:   {max_stations}")
    print(f"Hide fraction:  {hide_fraction}")
    print(f"Spatial weight: {spatial_weight}")
    print(f"Kriging:        {not args.skip_kriging}")
    print(f"Cokriging:      {not args.skip_cokriging}")
    print(f"Output group:   {output_label}")
    print("============================================================")

    if obs_path.exists() and stations_path.exists():
        print("\n[1/6] Loading cached CSV files...")
        obs = pd.read_csv(obs_path, parse_dates=["date"])
        stations = pd.read_csv(stations_path).set_index("station_id")
        stations.index = stations.index.astype(str)
    else:
        print("\n[1/6] Downloading NOAA GHCN-Daily data...")
        obs, stations = gtf.fetch_state_data(
            state,
            variables=variables,
            years=years,
            max_stations=max_stations,
            cache_dir=cache_dir,
            progress=True,
        )
        obs.to_csv(obs_path, index=False)
        stations.to_csv(stations_path)

        print(f"Saved observations -> {obs_path}")
        print(f"Saved stations     -> {stations_path}")

    print(f"Observation rows: {len(obs):,}")
    print(f"Station rows:     {len(stations):,}")

    if obs.empty:
        raise RuntimeError("No observations were downloaded.")

    print("\n[2/6] Building monthly tensor...")

    tensor = gtf.build_tensor(
        obs,
        variables=variables,
        time_col="date",
        station_col="station",
        freq="MS",
        station_coords=stations,
    )

    print(f"Tensor shape:  {tensor.shape}")
    print(f"Missing rate:  {tensor.missing_rate:.2%}")
    print(f"Variables:     {tensor.variables}")
    print(f"Time steps:    {len(tensor.times)}")
    print(f"Stations:      {len(tensor.stations)}")

    if tensor.mask.sum() == 0:
        raise RuntimeError("The tensor contains no observed entries.")

    data = tensor.fill_for_algorithm()

    # --------------------------------------------------------------
    # Physical correction before standardization
    # --------------------------------------------------------------
    elevation = tensor.station_coords["elevation"].to_numpy()
    
    data_phys, physical_correction = gtf.apply_elevation_temperature_correction(
        data,
        elevation,
        list(tensor.variables),
        )

    # --------------------------------------------------------------
    # Standardize each variable separately after physical correction
    # --------------------------------------------------------------
    data_std, var_means, var_stds = standardize_by_variable(
        data_phys,
        tensor.mask,
    )

    print("\n[3/6] Creating held-out test entries...")

    rng = np.random.default_rng(seed)
    train_mask, holdout = gtf.hide_random(
        tensor.mask,
        hide_fraction,
        rng=rng,
    )

    n_observed = int(tensor.mask.sum())
    n_holdout = int(len(holdout[0]))

    print(f"Observed entries: {n_observed:,}")
    print(f"Held-out entries: {n_holdout:,}")

    coords = tensor.station_coords[["latitude", "longitude"]].to_numpy()

    print("\n[4/6] Running physically corrected, standardized, location-aware HaLRTC...")

    halrtc_result_std = gtf.halrtc(
        data_std,
        train_mask,
        rho=rho,
        max_iter=max_iter,
        tol=tol,
        verbose=True,
        coords=coords,
        spatial_weight=spatial_weight,
        spatial_power=spatial_power,
        station_mode=2,
    )

    # --------------------------------------------------------------
    # Inverse transform:
    # standardized space -> physical-corrected space -> original space
    # --------------------------------------------------------------
    completed_phys = inverse_standardize_by_variable(
        halrtc_result_std.completed,
        var_means,
        var_stds,
    )
    
    completed_original = completed_phys - physical_correction

    # Keep originally observed values exactly unchanged.
    completed_original = np.where(train_mask, data, completed_original)

    print(f"HaLRTC iterations: {halrtc_result_std.n_iter}")
    print(f"HaLRTC converged:  {halrtc_result_std.converged}")

    print("\n[5/6] Running baselines...")

    pred_mean = gtf.mean_fill(data, train_mask)
    pred_temporal = gtf.temporal_mean_fill(data, train_mask)
    pred_idw = gtf.idw_fill(data, train_mask, coords=coords, power=2.0)

    predictions = {
        "PhysicalLocationHaLRTC": completed_original,
        "MeanFill": pred_mean,
        "TemporalMean": pred_temporal,
        "IDW": pred_idw,
    }

    if not args.skip_kriging:
        print("Running Kriging...")
        predictions["Kriging"] = gtf.kriging_fill(
            data,
            train_mask,
            coords=coords,
            range_km=args.kriging_range_km,
            nugget=args.kriging_nugget,
            idw_power=2.0,
        )

    if not args.skip_cokriging:
        print("Running Cokriging...")
        predictions["Cokriging"] = gtf.cokriging_fill(
            data,
            train_mask,
            coords=coords,
            range_km=args.kriging_range_km,
            nugget=args.kriging_nugget,
            max_points=args.cokriging_max_points,
            idw_power=2.0,
        )

    # --------------------------------------------------------------
    # Per-variable metrics
    # --------------------------------------------------------------
    per_variable_metrics = {}

    v_idx, t_idx, s_idx = holdout

    for method_name, pred in predictions.items():
        per_variable_metrics[method_name] = {}

        for var_i, var_name in enumerate(tensor.variables):
            selected = v_idx == var_i

            if selected.sum() == 0:
                continue

            var_holdout = (
                v_idx[selected],
                t_idx[selected],
                s_idx[selected],
            )

            per_variable_metrics[method_name][var_name] = gtf.score(
                data,
                pred,
                var_holdout,
            )

    print("\nPer-variable evaluation on held-out entries")
    print("============================================================")

    for method_name, var_results in per_variable_metrics.items():
        print(f"\n{method_name}")
        print(f"{'Variable':<10} {'RMSE':>10} {'MAE':>10} {'R^2':>10} {'r':>10} {'n':>8}")
        print("------------------------------------------------------------")

        for var_name, m in var_results.items():
            print(
                f"{var_name:<10} "
                f"{m.rmse:10.4f} "
                f"{m.mae:10.4f} "
                f"{m.r2:10.4f} "
                f"{m.r:10.4f} "
                f"{m.n:8d}"
            )

    report = {
        "config": {
            "state": state,
            "years": years,
            "variables": variables,
            "max_stations": max_stations,
            "hide_fraction": hide_fraction,
            "output_group": output_label,
            "seed": seed,
            "rho": rho,
            "max_iter": max_iter,
            "tol": tol,
            "spatial_weight": spatial_weight,
            "spatial_power": spatial_power,
            "run_kriging": not args.skip_kriging,
            "run_cokriging": not args.skip_cokriging,
            "kriging_range_km": args.kriging_range_km,
            "kriging_nugget": args.kriging_nugget,
            "cokriging_max_points": args.cokriging_max_points,
            "standardized_by_variable": True,
            "physical_correction": {
                "enabled": True,
                "temperature_lapse_rate_c_per_km": 6.5,
                "corrected_variables": ["TMAX", "TMIN"],
            },
        },
        "tensor": {
            "shape": list(tensor.shape),
            "missing_rate": tensor.missing_rate,
            "observed_entries": n_observed,
            "heldout_entries": n_holdout,
        },
        "halrtc": {
            "iterations": halrtc_result_std.n_iter,
            "converged": halrtc_result_std.converged,
            "final_relative_change": (
                float(halrtc_result_std.history[-1])
                if halrtc_result_std.history
                else None
            ),
        },

        "per_variable_metrics": {
            method_name: {
                var_name: metric_to_dict(m)
                for var_name, m in var_results.items()
            }
            for method_name, var_results in per_variable_metrics.items()
        },
    }

    report_path = reports_dir / "california_demo_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report -> {report_path}")

    print("\n[6/6] Saving figures...")

    fig, _ = gtf.plot_station_map(stations)
    fig.savefig(
        figures_dir / "california_station_map.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, _ = gtf.plot_missing_heatmap(tensor, variable=0)
    fig.savefig(
        figures_dir / "california_missing_heatmap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, _ = gtf.plot_convergence(halrtc_result_std.history)
    fig.savefig(
        figures_dir / "california_halrtc_convergence.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    abs_errors = np.abs(
        completed_original[holdout] - data[holdout]
    )

    station_error_df = pd.DataFrame(
        {
            "station_id": np.asarray(tensor.stations)[s_idx],
            "error": abs_errors,
        }
    )

    station_errors = (
        station_error_df
        .groupby("station_id")["error"]
        .mean()
    )

    fig, _ = plot_station_error_map(
        tensor.station_coords,
        station_errors,
        error_col="error",
    )

    fig.savefig(
        figures_dir / "california_station_error_map.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved figures -> {figures_dir}")
    print("\nDemo completed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

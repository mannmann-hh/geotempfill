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

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import geotempfill as gtf


def main() -> int:
    # ------------------------------------------------------------------
    # 1. Project paths
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "raw"
    cache_dir = project_root / "data" / "cache"
    figures_dir = project_root / "results" / "figures"
    reports_dir = project_root / "results" / "reports"

    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    obs_path = raw_dir / "observations_CA.csv"
    stations_path = raw_dir / "stations_CA.csv"

    # ------------------------------------------------------------------
    # 2. Demo configuration
    # ------------------------------------------------------------------
    state = "CA"
    years = [2020, 2021]
    variables = ["TMAX", "TMIN"]
    max_stations = 30

    hide_fraction = 0.10
    seed = 0

    rho = 5e-3
    max_iter = 300
    tol = 1e-5

    print("============================================================")
    print("geotempfill California demo")
    print("============================================================")
    print(f"State:          {state}")
    print(f"Years:          {years}")
    print(f"Variables:      {variables}")
    print(f"Max stations:   {max_stations}")
    print(f"Hide fraction:  {hide_fraction}")
    print("============================================================")

    # ------------------------------------------------------------------
    # 3. Download or load data
    # ------------------------------------------------------------------
    if obs_path.exists() and stations_path.exists():
        print("\n[1/6] Loading cached CSV files...")
        import pandas as pd

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
        raise RuntimeError(
            "No observations were downloaded. Try increasing max_stations "
            "or changing the year range."
        )

    # ------------------------------------------------------------------
    # 4. Build monthly weather tensor
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5. Create held-out evaluation mask
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 6. Run HaLRTC
    # ------------------------------------------------------------------
    print("\n[4/6] Running HaLRTC...")

    halrtc_result = gtf.halrtc(
        data,
        train_mask,
        rho=rho,
        max_iter=max_iter,
        tol=tol,
        verbose=True,
    )

    print(f"HaLRTC iterations: {halrtc_result.n_iter}")
    print(f"HaLRTC converged:  {halrtc_result.converged}")

    # ------------------------------------------------------------------
    # 7. Run baselines
    # ------------------------------------------------------------------
    print("\n[5/6] Running baselines...")

    coords = tensor.station_coords[["latitude", "longitude"]].to_numpy()

    pred_mean = gtf.mean_fill(data, train_mask)
    pred_temporal = gtf.temporal_mean_fill(data, train_mask)
    pred_idw = gtf.idw_fill(data, train_mask, coords=coords, power=2.0)

    predictions = {
        "HaLRTC": halrtc_result.completed,
        "MeanFill": pred_mean,
        "TemporalMean": pred_temporal,
        "IDW": pred_idw,
    }

    metrics = {
        name: gtf.score(data, pred, holdout)
        for name, pred in predictions.items()
    }

    print("\nMethod comparison on held-out entries")
    print("------------------------------------------------------------")
    print(f"{'Method':<15} {'RMSE':>10} {'MAE':>10} {'R^2':>10} {'r':>10} {'n':>8}")
    print("------------------------------------------------------------")
    for name, m in metrics.items():
        print(
            f"{name:<15} "
            f"{m.rmse:10.4f} "
            f"{m.mae:10.4f} "
            f"{m.r2:10.4f} "
            f"{m.r:10.4f} "
            f"{m.n:8d}"
        )
    print("------------------------------------------------------------")

    # ------------------------------------------------------------------
    # 8. Save JSON report
    # ------------------------------------------------------------------
    report = {
        "config": {
            "state": state,
            "years": years,
            "variables": variables,
            "max_stations": max_stations,
            "hide_fraction": hide_fraction,
            "seed": seed,
            "rho": rho,
            "max_iter": max_iter,
            "tol": tol,
        },
        "tensor": {
            "shape": list(tensor.shape),
            "missing_rate": tensor.missing_rate,
            "observed_entries": n_observed,
            "heldout_entries": n_holdout,
        },
        "halrtc": {
            "iterations": halrtc_result.n_iter,
            "converged": halrtc_result.converged,
            "final_relative_change": (
                float(halrtc_result.history[-1])
                if halrtc_result.history
                else None
            ),
        },
        "metrics": {
            name: {
                "rmse": m.rmse,
                "mae": m.mae,
                "r2": m.r2,
                "r": m.r,
                "n": m.n,
            }
            for name, m in metrics.items()
        },
    }

    report_path = reports_dir / "california_demo_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report -> {report_path}")

    # ------------------------------------------------------------------
    # 9. Save figures
    # ------------------------------------------------------------------
    print("\n[6/6] Saving figures...")

    fig, _ = gtf.plot_station_map(stations)
    fig.savefig(figures_dir / "california_station_map.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, _ = gtf.plot_missing_heatmap(tensor, variable=0)
    fig.savefig(figures_dir / "california_missing_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, _ = gtf.plot_convergence(halrtc_result.history)
    fig.savefig(figures_dir / "california_halrtc_convergence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics_dict = {
        name: {
            "rmse": m.rmse,
            "mae": m.mae,
            "r2": m.r2,
            "r": m.r,
            "n": m.n,
        }
        for name, m in metrics.items()
    }

    fig, _ = gtf.plot_method_comparison(metrics_dict, metric="rmse")
    fig.savefig(figures_dir / "california_method_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figures -> {figures_dir}")

    print("\nDemo completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
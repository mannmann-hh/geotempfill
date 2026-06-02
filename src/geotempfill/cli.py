"""
Command-line interface for geotempfill.

Run ``python -m geotempfill --help`` to see the available commands.

Two subcommands are exposed:

- ``download``  -- pull NOAA GHCN-Daily data for a US state and save the
  long-format observations and station metadata as CSV files. Useful for
  preparing datasets without writing any Python code.

- ``benchmark`` -- given a previously downloaded CSV, build the
  ``(variable, time, station)`` tensor, hide a configurable fraction of
  the observations, run HaLRTC and the baselines, and print a small
  summary table of RMSE / R^2 / r per method.

The CLI is meant as a *thin wrapper*: it imports the same functions as
the example notebook and the tests do, with no extra logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from . import __version__
from .baselines import idw_fill, mean_fill, temporal_mean_fill
from .bayesian import empirical_bayes_fill
from .data import fetch_state_data
from .evaluation import hide_random, score
from .halrtc import halrtc
from .spatial import cokriging_fill, kriging_fill
from .tensor import build_tensor


# ---------------------------------------------------------------------------
# `download` subcommand
# ---------------------------------------------------------------------------

def _cmd_download(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    years = list(range(args.start_year, args.end_year + 1))
    print(
        f"Downloading GHCN-Daily for state={args.state}, years={years}, "
        f"variables={args.variables}, max_stations={args.max_stations}"
    )
    obs, stations = fetch_state_data(
        args.state,
        variables=args.variables,
        years=years,
        max_stations=args.max_stations,
        cache_dir=cache_dir,
    )
    obs_path = out_dir / f"observations_{args.state}.csv"
    sta_path = out_dir / f"stations_{args.state}.csv"
    obs.to_csv(obs_path, index=False)
    stations.to_csv(sta_path)
    print(f"Wrote {len(obs):,} obs rows  -> {obs_path}")
    print(f"Wrote {len(stations):,} stations -> {sta_path}")
    return 0


# ---------------------------------------------------------------------------
# `benchmark` subcommand
# ---------------------------------------------------------------------------

def _cmd_benchmark(args: argparse.Namespace) -> int:
    obs = pd.read_csv(args.obs)
    stations = pd.read_csv(args.stations).set_index("station_id")
    stations.index = stations.index.astype(str)

    tensor = build_tensor(
        obs,
        variables=args.variables,
        time_col="date",
        station_col="station",
        freq=args.freq,
        station_coords=stations,
    )
    print(f"Built tensor: shape={tensor.shape}, missing={tensor.missing_rate:.1%}")

    # Replace NaNs (which only sit at masked positions anyway) with zeros
    # before passing to the algorithms.
    data = tensor.fill_for_algorithm()

    # Hide a random fraction of *observed* entries for evaluation.
    rng = np.random.default_rng(args.seed)
    train_mask, holdout = hide_random(tensor.mask, args.hide_fraction, rng=rng)
    print(
        f"Held out {holdout.size:,} entries "
        f"({holdout.size / tensor.mask.sum():.1%} of observed)"
    )

    truth = data.copy()
    coords = tensor.station_coords[["latitude", "longitude"]].to_numpy()

    print("\nRunning HaLRTC...")
    halrtc_res = halrtc(
        data, train_mask,
        rho=args.rho, max_iter=args.max_iter, tol=args.tol, verbose=args.verbose,
    )
    print(f"  iterations={halrtc_res.n_iter}, converged={halrtc_res.converged}")

    print("Running baselines...")
    pred_mean = mean_fill(data, train_mask)
    pred_temp = temporal_mean_fill(data, train_mask)
    pred_idw = idw_fill(data, train_mask, coords=coords, power=args.idw_power)

    methods = {
        "HaLRTC": halrtc_res.completed,
        "MeanFill": pred_mean,
        "TempMean": pred_temp,
        "IDW": pred_idw,
    }

    if args.include_kriging:
        print("Running kriging...")
        methods["Kriging"] = kriging_fill(
            data,
            train_mask,
            coords=coords,
            range_km=args.kriging_range_km,
            nugget=args.kriging_nugget,
            min_points=args.kriging_min_points,
            idw_power=args.idw_power,
        )

    if args.include_cokriging:
        print("Running cokriging...")
        methods["Cokriging"] = cokriging_fill(
            data,
            train_mask,
            coords=coords,
            range_km=args.kriging_range_km,
            nugget=args.kriging_nugget,
            min_points=args.cokriging_min_points,
            max_points=args.cokriging_max_points,
            idw_power=args.idw_power,
        )

    if args.include_bayes:
        print("Running empirical Bayes...")
        methods["EmpBayes"] = empirical_bayes_fill(
            data,
            train_mask,
            coords=coords,
            shrinkage=args.bayes_shrinkage,
            temporal_smoothing=args.bayes_temporal_smoothing,
            spatial_smoothing=args.bayes_spatial_smoothing,
        )

    rows = []
    for name, pred in methods.items():
        m = score(truth, pred, holdout)
        rows.append((name, m.rmse, m.r2, m.r, m.n))

    print("\n  Method      RMSE       R^2        r          n")
    print("  ----------  ---------  ---------  ---------  --------")
    for name, rmse, r2, r, n in rows:
        print(f"  {name:<10}  {rmse:9.4f}  {r2:9.4f}  {r:9.4f}  {n:8d}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {k: v for k, v in vars(args).items() if k != "func"},
            "tensor_shape": list(tensor.shape),
            "missing_rate": tensor.missing_rate,
            "results": {
                name: {"rmse": rmse, "r2": r2, "r": r, "n": n}
                for name, rmse, r2, r, n in rows
            },
        }
        report_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote results -> {report_path}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geotempfill",
        description=(
            "Tensor-completion-based gap filling for NOAA GHCN-Daily "
            "weather observations."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- download ----
    p_dl = sub.add_parser("download", help="download GHCN-Daily for a US state")
    p_dl.add_argument("--state", required=True, help="two-letter US state code, e.g. CA")
    p_dl.add_argument("--start-year", type=int, default=2020)
    p_dl.add_argument("--end-year", type=int, default=2022)
    p_dl.add_argument(
        "--variables", nargs="+", default=["TMAX", "TMIN", "PRCP"],
        help="GHCN-Daily element codes",
    )
    p_dl.add_argument(
        "--max-stations", type=int, default=None,
        help="limit how many stations to fetch (useful for tests)",
    )
    p_dl.add_argument("--out-dir", default="data/raw")
    p_dl.add_argument("--cache-dir", default=None)
    p_dl.set_defaults(func=_cmd_download)

    # ---- benchmark ----
    p_bm = sub.add_parser(
        "benchmark", help="run HaLRTC + baselines and print metrics"
    )
    p_bm.add_argument("--obs", required=True, help="path to observations CSV")
    p_bm.add_argument("--stations", required=True, help="path to stations CSV")
    p_bm.add_argument(
        "--variables", nargs="+", default=["TMAX", "TMIN", "PRCP"],
    )
    p_bm.add_argument(
        "--freq", default="MS",
        help="pandas frequency string, e.g. 'D', 'MS' (month start), 'QS'",
    )
    p_bm.add_argument("--hide-fraction", type=float, default=0.05)
    p_bm.add_argument("--rho", type=float, default=5e-3)
    p_bm.add_argument("--max-iter", type=int, default=500)
    p_bm.add_argument("--tol", type=float, default=1e-5)
    p_bm.add_argument("--idw-power", type=float, default=2.0)
    p_bm.add_argument(
        "--include-kriging",
        action="store_true",
        help="also run ordinary kriging as a spatial baseline",
    )
    p_bm.add_argument(
        "--include-cokriging",
        action="store_true",
        help="also run experimental simple cokriging as a multivariable spatial baseline",
    )
    p_bm.add_argument(
        "--kriging-range-km",
        type=float,
        default=None,
        help="ordinary kriging covariance range; defaults to median station distance",
    )
    p_bm.add_argument("--kriging-nugget", type=float, default=1e-6)
    p_bm.add_argument("--kriging-min-points", type=int, default=3)
    p_bm.add_argument("--cokriging-min-points", type=int, default=5)
    p_bm.add_argument(
        "--cokriging-max-points",
        type=int,
        default=120,
        help="maximum donor variable-station observations per cokriging solve",
    )
    p_bm.add_argument(
        "--include-bayes",
        action="store_true",
        help="also run a lightweight empirical-Bayes additive baseline",
    )
    p_bm.add_argument("--bayes-shrinkage", type=float, default=5.0)
    p_bm.add_argument("--bayes-temporal-smoothing", type=float, default=0.20)
    p_bm.add_argument("--bayes-spatial-smoothing", type=float, default=0.10)
    p_bm.add_argument("--seed", type=int, default=0)
    p_bm.add_argument("--verbose", action="store_true")
    p_bm.add_argument(
        "--report", default=None,
        help="optional path to write a JSON report",
    )
    p_bm.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

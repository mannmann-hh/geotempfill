"""Smoke tests for the CLI entry point.

These tests cover the gap that allowed a tuple-vs-ndarray bug to ship
unnoticed: the unit tests for ``hide_random`` use ``len(holdout[0])`` to
count held-out cells, but ``cli._cmd_benchmark`` originally used
``holdout.size`` (which crashes because ``holdout`` is a tuple).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _make_fake_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Write a tiny synthetic observations + stations CSV pair into tmp_path."""
    n_stations = 6
    rng = np.random.default_rng(0)

    stations = pd.DataFrame({
        "station_id": [f"USW{i:05d}" for i in range(n_stations)],
        "latitude":  [35.0 + 0.5 * i for i in range(n_stations)],
        "longitude": [-120.0 + 0.5 * i for i in range(n_stations)],
        "elevation": [10.0 + 50.0 * i for i in range(n_stations)],
        "state": ["CA"] * n_stations,
        "name":  [f"Stn {i}" for i in range(n_stations)],
    })

    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    rows = []
    for sid in stations["station_id"]:
        for d in dates:
            if rng.random() < 0.15:
                continue
            rows.append({
                "station": sid,
                "date": d,
                "TMAX": 20.0 + 5.0 * rng.standard_normal(),
                "TMIN": 10.0 + 5.0 * rng.standard_normal(),
            })

    obs = pd.DataFrame(rows)

    obs_path = tmp_path / "obs.csv"
    sta_path = tmp_path / "stations.csv"
    obs.to_csv(obs_path, index=False)
    stations.to_csv(sta_path, index=False)
    return obs_path, sta_path


def test_benchmark_runs_end_to_end(tmp_path, capsys):
    """The benchmark subcommand must run without crashing and produce a report."""
    from geotempfill.cli import main

    obs_path, sta_path = _make_fake_dataset(tmp_path)
    report_path = tmp_path / "report.json"

    rc = main([
        "benchmark",
        "--obs", str(obs_path),
        "--stations", str(sta_path),
        "--variables", "TMAX", "TMIN",
        "--freq", "MS",
        "--hide-fraction", "0.1",
        "--max-iter", "20",
        "--report", str(report_path),
    ])

    assert rc == 0

    captured = capsys.readouterr().out
    assert "Held out" in captured
    assert "Running methods" in captured

    payload = json.loads(report_path.read_text())
    assert "config" in payload
    assert "results" in payload
    # At least the four default methods must produce a result row.
    assert len(payload["results"]) >= 4
    for method_name, metrics in payload["results"].items():
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "r" in metrics
        assert "n" in metrics


def test_benchmark_reports_correct_holdout_count(tmp_path, capsys):
    """Regression test for the ``holdout.size`` AttributeError.

    Catches the bug where ``cli._cmd_benchmark`` tried to call ``.size`` on
    the tuple returned by ``hide_random`` instead of on one of its
    component arrays.
    """
    from geotempfill.cli import main

    obs_path, sta_path = _make_fake_dataset(tmp_path)

    rc = main([
        "benchmark",
        "--obs", str(obs_path),
        "--stations", str(sta_path),
        "--variables", "TMAX", "TMIN",
        "--freq", "MS",
        "--hide-fraction", "0.1",
        "--max-iter", "5",
    ])

    assert rc == 0
    out = capsys.readouterr().out
    # The "Held out N entries" line must print without crashing.
    assert "Held out" in out
    assert "of observed" in out

"""
NOAA GHCN-Daily data downloader.

This module downloads station metadata and daily observations from NOAA
GHCN-Daily public HTTPS endpoints. It returns data in a format suitable for
building a (variable, time, station) tensor.
"""

from __future__ import annotations

import gzip
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

__all__ = [
    "GhcnStation",
    "list_stations",
    "fetch_station_data",
    "fetch_state_data",
]

logger = logging.getLogger(__name__)

STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
STATION_DATA_URL = (
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily"
    "/access/{station_id}.csv"
)

_HTTP_HEADERS = {
    "User-Agent": "geotempfill/0.1 (educational project; +https://github.com/)"
}

_UNIT_DIVISOR = {
    "TMAX": 10.0,
    "TMIN": 10.0,
    "TAVG": 10.0,
    "PRCP": 10.0,
    "SNOW": 1.0,
    "SNWD": 1.0,
    "AWND": 10.0,
}


@dataclass(frozen=True)
class GhcnStation:
    """Metadata for a single GHCN-Daily station."""

    station_id: str
    latitude: float
    longitude: float
    elevation: float
    state: str
    name: str

    def to_dict(self) -> dict:
        return {
            "station_id": self.station_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
            "state": self.state,
            "name": self.name,
        }


def _http_get(url: str, *, timeout: float = 30.0, retries: int = 3) -> bytes:
    """Fetch a URL and return raw bytes, with simple retry logic."""
    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        req = Request(url, headers=_HTTP_HEADERS)
        try:
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_err = exc
            logger.warning(
                "GET %s failed (attempt %d/%d): %s",
                url,
                attempt,
                retries,
                exc,
            )

    assert last_err is not None
    raise last_err


def _maybe_decompress(payload: bytes) -> bytes:
    """Gunzip a payload if it is gzip encoded."""
    if len(payload) >= 2 and payload[:2] == b"\x1f\x8b":
        return gzip.decompress(payload)
    return payload


_STATION_COLSPECS = [
    (0, 11),
    (12, 20),
    (21, 30),
    (31, 37),
    (38, 40),
    (41, 71),
]

_STATION_COLNAMES = [
    "station_id",
    "latitude",
    "longitude",
    "elevation",
    "state",
    "name",
]


def list_stations(
    *,
    state: Optional[str] = None,
    country: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> List[GhcnStation]:
    """Return GHCN-Daily stations, optionally filtered by US state/country."""
    raw: Optional[bytes] = None
    cache_path: Optional[Path] = None

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "ghcnd-stations.txt"

        if cache_path.exists():
            raw = cache_path.read_bytes()

    if raw is None:
        raw = _maybe_decompress(_http_get(STATIONS_URL))
        if cache_path is not None:
            cache_path.write_bytes(raw)

    df = pd.read_fwf(
        io.BytesIO(raw),
        colspecs=_STATION_COLSPECS,
        names=_STATION_COLNAMES,
        dtype=str,
    )

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["elevation"] = pd.to_numeric(df["elevation"], errors="coerce")
    df["state"] = df["state"].fillna("").str.strip()
    df["name"] = df["name"].fillna("").str.strip()

    if country:
        df = df[df["station_id"].str.startswith(country.upper())]

    if state:
        df = df[df["state"] == state.upper()]

    df = df.dropna(subset=["latitude", "longitude"])

    return [GhcnStation(**row) for row in df.to_dict(orient="records")]


def fetch_station_data(
    station_id: str,
    *,
    variables: Sequence[str] = ("TMAX", "TMIN", "PRCP"),
    years: Optional[Iterable[int]] = None,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Download the daily record for one station.

    The returned DataFrame always contains:

        station, date, <requested variables...>

    If a station does not report a requested variable, that variable is
    kept as an all-NaN column.
    """
    cache_path: Optional[Path] = None
    raw: Optional[bytes] = None

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{station_id}.csv"

        if cache_path.exists():
            raw = cache_path.read_bytes()

    if raw is None:
        url = STATION_DATA_URL.format(station_id=station_id)
        raw = _http_get(url)

        if cache_path is not None:
            cache_path.write_bytes(raw)

    df = pd.read_csv(io.BytesIO(raw), low_memory=False)

    if "ELEMENT" in df.columns and "VALUE" in df.columns:
        df = (
            df.pivot_table(
                index=["STATION", "DATE"],
                columns="ELEMENT",
                values="VALUE",
                aggfunc="first",
            )
            .reset_index()
        )
        df.columns.name = None

    df = df.rename(columns={"STATION": "station", "DATE": "date"})

    if "station" not in df.columns:
        df["station"] = station_id

    if "date" not in df.columns:
        raise ValueError(f"station file for {station_id} does not contain DATE/date")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    for v in variables:
        if v not in df.columns:
            df[v] = np.nan

    keep_cols = ["station", "date", *variables]
    df = df.loc[:, keep_cols].copy()

    for v in variables:
        df[v] = pd.to_numeric(df[v], errors="coerce") / _UNIT_DIVISOR.get(v, 1.0)

    if years is not None:
        years_set = {int(y) for y in years}
        df = df[df["date"].dt.year.isin(years_set)]

    return df.reset_index(drop=True)


def fetch_state_data(
    state: str,
    *,
    variables: Sequence[str] = ("TMAX", "TMIN", "PRCP"),
    years: Iterable[int] = (2020, 2021, 2022),
    max_stations: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download usable observations for stations in a US state.

    This function avoids selecting precipitation-only volunteer stations first.
    It prioritizes official weather/climate stations and only keeps stations
    that actually contain at least one non-missing value for the requested
    variables in the requested years.
    """
    years_list = [int(y) for y in years]

    def station_priority(st: GhcnStation) -> tuple[int, str]:
        if st.station_id.startswith("USW"):
            group = 0
        elif st.station_id.startswith("USC"):
            group = 1
        elif st.station_id.startswith("USR"):
            group = 2
        elif st.station_id.startswith("US1"):
            group = 9
        else:
            group = 5

        return group, st.station_id

    stations_all = [
        s
        for s in list_stations(state=state, country="US", cache_dir=cache_dir)
        if s.station_id.startswith(("USW", "USC", "USR"))
    ]

    stations_all = sorted(stations_all, key=station_priority)

    if not stations_all:
        raise ValueError(f"no GHCN stations found for state {state!r}")

    rows: List[pd.DataFrame] = []
    selected_stations: List[GhcnStation] = []
    total = len(stations_all)

    for i, st in enumerate(stations_all, 1):
        if max_stations is not None and len(selected_stations) >= max_stations:
            break

        if progress:
            target = max_stations if max_stations is not None else "all"
            print(
                f"  [{i:4d}/{total}] {st.station_id} {st.name} "
                f"(kept {len(selected_stations)}/{target})"
            )

        try:
            sub = fetch_station_data(
                st.station_id,
                variables=variables,
                years=years_list,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            logger.warning("skipping %s: %s", st.station_id, exc)
            continue

        if sub.empty:
            continue

        has_requested_data = any(sub[v].notna().any() for v in variables)

        if not has_requested_data:
            continue

        rows.append(sub)
        selected_stations.append(st)

    if not rows:
        raise ValueError(
            f"no usable observations found for state={state!r}, "
            f"variables={list(variables)!r}, years={years_list!r}"
        )

    observations = pd.concat(rows, ignore_index=True)

    stations_df = pd.DataFrame([s.to_dict() for s in selected_stations])
    stations_df = stations_df.set_index("station_id")

    return observations, stations_df
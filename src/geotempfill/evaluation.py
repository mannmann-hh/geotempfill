from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Metrics:
    """Evaluation metrics computed on held-out tensor entries."""

    rmse: float
    mae: float
    r2: float
    r: float
    n: int


def hide_random(
    mask: np.ndarray,
    fraction: float,
    *,
    rng: Optional[np.random.Generator | int] = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Randomly hide a fraction of observed entries.

    Parameters
    ----------
    mask:
        Boolean tensor where True means observed.
    fraction:
        Fraction of observed entries to hide.
    rng:
        Random generator or integer seed.

    Returns
    -------
    train_mask:
        Copy of mask with selected observed entries changed to False.
    holdout:
        Tuple of index arrays pointing to the hidden entries.
    """
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")

    observed = np.argwhere(mask)
    if observed.size == 0:
        raise ValueError("mask has no observed entries")

    generator = np.random.default_rng(rng)
    n_hide = max(1, int(round(len(observed) * fraction)))

    chosen = generator.choice(len(observed), size=n_hide, replace=False)
    hidden_coords = observed[chosen]

    train_mask = mask.copy()
    holdout = tuple(hidden_coords[:, i] for i in range(hidden_coords.shape[1]))
    train_mask[holdout] = False

    return train_mask, holdout


def score(
    truth: np.ndarray,
    pred: np.ndarray,
    holdout: tuple[np.ndarray, ...],
) -> Metrics:
    """Compute RMSE, MAE, R² and Pearson r on held-out entries."""
    if truth.shape != pred.shape:
        raise ValueError("truth and pred must have the same shape")

    y_true = np.asarray(truth[holdout], dtype=float)
    y_pred = np.asarray(pred[holdout], dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    n = int(y_true.size)
    if n == 0:
        return Metrics(rmse=np.nan, mae=np.nan, r2=np.nan, r=np.nan, n=0)

    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    if n > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = np.nan

    return Metrics(rmse=rmse, mae=mae, r2=r2, r=r, n=n)
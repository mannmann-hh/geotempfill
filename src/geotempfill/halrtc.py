"""
High-accuracy Low-Rank Tensor Completion (HaLRTC).

Implementation of the HaLRTC algorithm for filling missing values in
multidimensional arrays, following the formulation of Liu et al. (2013)
and the application to surveillance/observational data described by
Liao et al. (2024).

The algorithm assumes an N-th order tensor M with a known set of observed
entries (mask Omega). It seeks a tensor X that:

    minimizes   sum_i alpha_i * || X_(i) ||_*           (sum of nuclear
                                                         norms of the
                                                         mode-i unfoldings)
    subject to  X[Omega] = M[Omega]                      (interpolation
                                                         constraint)

This convex relaxation of the low-Tucker-rank problem is solved with
the Alternating Direction Method of Multipliers (ADMM), as in
Liu et al. (2013), Algorithm 4 ("HaLRTC").

References
----------
Liu, J., Musialski, P., Wonka, P., & Ye, J. (2013). Tensor completion
    for estimating missing values in visual data. IEEE TPAMI, 35(1),
    208-220.
Liao, Y., Shi, Y., Fan, Z., et al. (2024). A new disease mapping method
    for improving data completeness of syndromic surveillance with high
    missing rates. Transactions in GIS, 28, 1869-1882.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["HaLRTCResult", "halrtc", "svt", "unfold", "fold"]


# ---------------------------------------------------------------------------
# Helpers: tensor unfolding / folding
# ---------------------------------------------------------------------------

def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Mode-`mode` unfolding of a tensor into a matrix.

    Mode-n unfolding rearranges a tensor of shape (I_0, ..., I_{N-1}) into
    a matrix of shape (I_mode, prod(I_k for k != mode)). This follows the
    convention used in Kolda & Bader (2009).

    Parameters
    ----------
    tensor : np.ndarray
        Input N-dimensional array.
    mode : int
        Mode along which to unfold (0-indexed).

    Returns
    -------
    np.ndarray
        2-D matrix unfolding.
    """
    if mode < 0 or mode >= tensor.ndim:
        raise ValueError(f"mode must be in [0, {tensor.ndim - 1}], got {mode}")
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def fold(matrix: np.ndarray, mode: int, shape: tuple) -> np.ndarray:
    """Inverse of `unfold`: refold a matrix back into a tensor.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix of shape (shape[mode], prod(other dims)).
    mode : int
        Mode used in the original unfolding.
    shape : tuple of int
        Target tensor shape.

    Returns
    -------
    np.ndarray
        N-dimensional tensor.
    """
    full_shape = [shape[mode]] + [shape[k] for k in range(len(shape)) if k != mode]
    return np.moveaxis(matrix.reshape(full_shape), 0, mode)


# ---------------------------------------------------------------------------
# Singular Value Thresholding (proximal operator of the nuclear norm)
# ---------------------------------------------------------------------------

def svt(matrix: np.ndarray, tau: float) -> np.ndarray:
    """Singular value thresholding operator D_tau.

    Computes  argmin_X  tau * ||X||_* + 0.5 * ||X - matrix||_F^2

    by soft-thresholding the singular values of `matrix` by `tau`.

    Parameters
    ----------
    matrix : np.ndarray
        2-D input matrix.
    tau : float
        Threshold (must be non-negative).

    Returns
    -------
    np.ndarray
        Same shape as `matrix`.
    """
    if tau < 0:
        raise ValueError("tau must be non-negative")
    # `full_matrices=False` keeps the SVD economy-size, much faster
    # for tall/thin or fat/short matrices (typical of mode unfoldings).
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    nonzero = s_thresh > 0
    if not np.any(nonzero):
        return np.zeros_like(matrix)
    return (u[:, nonzero] * s_thresh[nonzero]) @ vh[nonzero, :]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HaLRTCResult:
    """Output of the HaLRTC algorithm."""

    completed: np.ndarray
    """Tensor with missing entries filled in."""

    n_iter: int
    """Number of ADMM iterations actually performed."""

    converged: bool
    """Whether the stopping criterion was met before max_iter."""

    history: list = field(default_factory=list)
    """Per-iteration relative change ||X^{k+1} - X^k||_F / ||X^k||_F.
    Useful for diagnostics and plotting convergence."""


# ---------------------------------------------------------------------------
# Main HaLRTC routine
# ---------------------------------------------------------------------------

def halrtc(
    tensor: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: Optional[np.ndarray] = None,
    rho: float = 1e-3,
    max_iter: int = 500,
    tol: float = 1e-5,
    verbose: bool = False,
) -> HaLRTCResult:
    """Fill missing entries in `tensor` with the HaLRTC algorithm.

    Parameters
    ----------
    tensor : np.ndarray
        N-dimensional array containing observations. Values at positions
        where `mask` is False are ignored (any value, NaN included, is OK).
    mask : np.ndarray of bool
        Same shape as `tensor`. ``True`` marks observed entries to be kept
        fixed; ``False`` marks entries to be estimated.
    alpha : np.ndarray, optional
        Non-negative weights summing to 1, one per mode. They control the
        relative importance of low-rankness across modes. By default each
        mode gets equal weight 1/N.
    rho : float, default 1e-3
        ADMM penalty parameter. Liao et al. (2024) report sensible values
        in [1e-4, 1e-2]; the examples here use 5e-3 (the value selected
        on the IFV dataset of the paper).
    max_iter : int, default 500
        Maximum number of ADMM iterations.
    tol : float, default 1e-5
        Convergence tolerance on the relative Frobenius change of X.
    verbose : bool, default False
        If True, print iteration progress.

    Returns
    -------
    HaLRTCResult
        Object holding the completed tensor and convergence diagnostics.

    Notes
    -----
    The implementation follows Liu et al. (2013), Algorithm 4. At each
    iteration:

        for n in 0..N-1:
            M_n = fold_n( SVT_{alpha_n / rho}(  X_(n) + Y_n_(n) / rho  ) )
        X = (1 - Omega) * mean_n( M_n - Y_n / rho ) + Omega * tensor
        for n in 0..N-1:
            Y_n = Y_n - rho * (M_n - X)

    where Omega is the boolean mask of observed entries.
    """
    if tensor.shape != mask.shape:
        raise ValueError("tensor and mask must have the same shape")
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if not mask.any():
        raise ValueError("mask must mark at least one observed entry")

    ndim = tensor.ndim
    if alpha is None:
        alpha = np.full(ndim, 1.0 / ndim)
    else:
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (ndim,):
            raise ValueError(f"alpha must have length {ndim}")
        if np.any(alpha < 0):
            raise ValueError("alpha entries must be non-negative")
        if not np.isclose(alpha.sum(), 1.0):
            alpha = alpha / alpha.sum()

    if rho <= 0:
        raise ValueError("rho must be strictly positive")

    # Initialise X with the observed entries; missing entries get the mean
    # of the observed values (a neutral starting point).
    obs_mean = float(np.asarray(tensor)[mask].mean())
    X = np.where(mask, tensor, obs_mean).astype(float)

    # Auxiliary tensors M_n and dual variables Y_n, one per mode.
    M = [X.copy() for _ in range(ndim)]
    Y = [np.zeros_like(X) for _ in range(ndim)]

    history: list = []
    converged = False
    last_iter = 0

    shape = tensor.shape
    inv_ndim = 1.0 / ndim

    for it in range(1, max_iter + 1):
        last_iter = it
        X_prev = X

        # --- (1) Update each M_n via singular value thresholding ----------
        for n in range(ndim):
            tau = alpha[n] / rho
            tmp = unfold(X + Y[n] / rho, n)
            M[n] = fold(svt(tmp, tau), n, shape)

        # --- (2) Update X: average of (M_n - Y_n/rho) on the missing set,
        #         keep observed entries unchanged.
        avg = np.zeros_like(X)
        for n in range(ndim):
            avg += M[n] - Y[n] / rho
        avg *= inv_ndim
        X = np.where(mask, tensor, avg)

        # --- (3) Dual update -------------------------------------------
        for n in range(ndim):
            Y[n] = Y[n] - rho * (M[n] - X)

        # --- Convergence check -----------------------------------------
        denom = np.linalg.norm(X_prev)
        rel_change = (
            np.linalg.norm(X - X_prev) / denom
            if denom > 0
            else np.linalg.norm(X - X_prev)
        )
        history.append(rel_change)

        if verbose and (it % 25 == 0 or it == 1):
            print(f"[HaLRTC] iter={it:4d}  rel_change={rel_change:.3e}")

        if rel_change < tol:
            converged = True
            if verbose:
                print(f"[HaLRTC] converged at iter {it} (rel_change={rel_change:.3e})")
            break

    return HaLRTCResult(
        completed=X,
        n_iter=last_iter,
        converged=converged,
        history=history,
    )

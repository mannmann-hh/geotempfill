"""Unit tests for the HaLRTC core algorithm and its building blocks."""

from __future__ import annotations

import numpy as np
import pytest

from geotempfill.halrtc import HaLRTCResult, fold, halrtc, svt, unfold


# ---------------------------------------------------------------------------
# unfold / fold
# ---------------------------------------------------------------------------

class TestUnfoldFold:
    def test_unfold_shape(self):
        T = np.arange(24).reshape(2, 3, 4)
        for mode in range(3):
            U = unfold(T, mode)
            expected_cols = T.size // T.shape[mode]
            assert U.shape == (T.shape[mode], expected_cols)

    def test_fold_inverse_of_unfold(self):
        rng = np.random.default_rng(42)
        T = rng.standard_normal((4, 5, 6))
        for mode in range(3):
            U = unfold(T, mode)
            T_back = fold(U, mode, T.shape)
            np.testing.assert_allclose(T_back, T)

    def test_unfold_invalid_mode(self):
        T = np.zeros((2, 3, 4))
        with pytest.raises(ValueError):
            unfold(T, 5)
        with pytest.raises(ValueError):
            unfold(T, -1)


# ---------------------------------------------------------------------------
# Singular value thresholding
# ---------------------------------------------------------------------------

class TestSVT:
    def test_svt_zero_threshold_is_identity(self):
        rng = np.random.default_rng(0)
        M = rng.standard_normal((5, 7))
        np.testing.assert_allclose(svt(M, 0.0), M, atol=1e-10)

    def test_svt_large_threshold_zeros_matrix(self):
        rng = np.random.default_rng(0)
        M = rng.standard_normal((5, 7))
        s = np.linalg.svd(M, compute_uv=False)
        out = svt(M, s.max() + 1.0)
        np.testing.assert_allclose(out, np.zeros_like(M), atol=1e-10)

    def test_svt_reduces_nuclear_norm(self):
        rng = np.random.default_rng(1)
        M = rng.standard_normal((6, 8))
        nn_before = np.linalg.svd(M, compute_uv=False).sum()
        out = svt(M, 0.5)
        nn_after = np.linalg.svd(out, compute_uv=False).sum()
        assert nn_after < nn_before

    def test_svt_negative_tau_raises(self):
        with pytest.raises(ValueError):
            svt(np.zeros((3, 3)), -0.1)


# ---------------------------------------------------------------------------
# HaLRTC end-to-end
# ---------------------------------------------------------------------------

def _make_low_rank_tensor(shape=(4, 12, 8), rank=2, seed=0):
    """Build a tensor with low Tucker rank along every mode."""
    rng = np.random.default_rng(seed)
    factors = [rng.standard_normal((s, rank)) for s in shape]
    core = rng.standard_normal((rank, rank, rank))
    return np.einsum("ir,jt,ks,rts->ijk", factors[0], factors[1], factors[2], core)


class TestHaLRTC:
    def test_recovers_low_rank_tensor_from_partial_obs(self):
        rng = np.random.default_rng(0)

        # Build a smooth low-rank-ish tensor:
        # X[v, t, s] = A[v] + B[t] + C[s]
        n_vars, n_times, n_stations = 2, 8, 6

        var_effect = np.array([10.0, 15.0])[:, None, None]
        time_effect = np.linspace(-2.0, 2.0, n_times)[None, :, None]
        station_effect = np.linspace(-1.0, 1.0, n_stations)[None, None, :]

        truth = var_effect + time_effect + station_effect

        # Hide 30% of entries
        mask = rng.random(truth.shape) > 0.30
        observed = np.where(mask, truth, 0.0)

        result = halrtc(
            observed,
            mask,
            rho=1.0,
            max_iter=500,
            tol=1e-6,
        )

        missing = ~mask
        rmse = np.sqrt(np.mean((result.completed[missing] - truth[missing]) ** 2))

        assert rmse < 1.0, f"RMSE too large: {rmse}"

    def test_observed_entries_are_preserved(self):
        rng = np.random.default_rng(1)
        truth = rng.normal(size=(2, 5, 4))
        mask = rng.random(truth.shape) > 0.4
        observed = np.where(mask, truth, 0.0)

        result = halrtc(
            observed,
            mask,
            rho=1.0,
            max_iter=50,
            tol=1e-6,
        )

        assert np.allclose(result.completed[mask], truth[mask])
    def test_history_is_decreasing_eventually(self):
        truth = _make_low_rank_tensor(seed=2)
        rng = np.random.default_rng(2)
        mask = rng.random(truth.shape) > 0.4
        mask[0, 0, 0] = True
        result = halrtc(
            np.where(mask, truth, 0.0), mask, rho=5e-3, max_iter=100, tol=1e-7,
        )
        assert result.history[-1] < result.history[0]

    def test_alpha_normalisation(self):
        truth = _make_low_rank_tensor(seed=3)
        rng = np.random.default_rng(3)
        mask = rng.random(truth.shape) > 0.4
        mask[0, 0, 0] = True
        result = halrtc(
            np.where(mask, truth, 0.0), mask,
            alpha=np.array([2.0, 2.0, 2.0]),  # not summing to 1
            rho=5e-3, max_iter=20, tol=1e-7,
        )
        assert np.isfinite(result.completed).all()

    def test_invalid_inputs_raise(self):
        T = np.zeros((3, 3, 3))
        with pytest.raises(ValueError):
            halrtc(T, np.zeros((3, 3, 3), dtype=bool))  # all-False mask
        with pytest.raises(ValueError):
            halrtc(T, np.ones((2, 2, 2), dtype=bool))  # shape mismatch
        with pytest.raises(ValueError):
            halrtc(T, np.ones_like(T, dtype=bool), rho=0.0)
        with pytest.raises(ValueError):
            halrtc(
                T, np.ones_like(T, dtype=bool),
                alpha=np.array([0.5, 0.5]),  # wrong length
            )
        with pytest.raises(ValueError):
            halrtc(
                T, np.ones_like(T, dtype=bool),
                alpha=np.array([0.5, 0.6, -0.1]),  # negative
            )

    def test_max_iter_one_returns_result(self):
        truth = _make_low_rank_tensor()
        mask = np.ones_like(truth, dtype=bool)
        mask[0, 0, 0] = False
        result = halrtc(
            np.where(mask, truth, 0.0), mask, rho=5e-3, max_iter=1,
        )
        assert result.n_iter == 1
        assert len(result.history) == 1

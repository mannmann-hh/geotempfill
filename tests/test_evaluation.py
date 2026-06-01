import numpy as np
import pytest

from geotempfill.evaluation import Metrics, hide_random, score


def test_hide_random_returns_train_mask_and_holdout():
    mask = np.ones((2, 5, 4), dtype=bool)

    train_mask, holdout = hide_random(mask, fraction=0.25, rng=0)

    assert train_mask.shape == mask.shape
    assert train_mask.dtype == bool

    # 2 * 5 * 4 = 40 observed entries, 25% should be hidden
    assert len(holdout) == 3
    assert len(holdout[0]) == 10

    # Hidden entries should be False in the train mask
    assert np.all(train_mask[holdout] == False)

    # Non-hidden observed entries should remain True
    assert train_mask.sum() == 30


def test_hide_random_only_hides_observed_entries():
    mask = np.zeros((2, 4, 3), dtype=bool)
    mask[0, :, :] = True  # only first variable is observed

    train_mask, holdout = hide_random(mask, fraction=0.5, rng=42)

    # All hidden entries must come from observed entries
    assert np.all(mask[holdout] == True)

    # The second variable was never observed, so it should remain all False
    assert np.all(train_mask[1, :, :] == False)


def test_hide_random_is_reproducible_with_seed():
    mask = np.ones((3, 4, 5), dtype=bool)

    train_mask_1, holdout_1 = hide_random(mask, fraction=0.2, rng=123)
    train_mask_2, holdout_2 = hide_random(mask, fraction=0.2, rng=123)

    assert np.array_equal(train_mask_1, train_mask_2)

    for a, b in zip(holdout_1, holdout_2):
        assert np.array_equal(a, b)


def test_hide_random_rejects_invalid_fraction():
    mask = np.ones((2, 3, 4), dtype=bool)

    with pytest.raises(ValueError):
        hide_random(mask, fraction=0.0)

    with pytest.raises(ValueError):
        hide_random(mask, fraction=1.0)

    with pytest.raises(ValueError):
        hide_random(mask, fraction=-0.1)

    with pytest.raises(ValueError):
        hide_random(mask, fraction=1.5)


def test_hide_random_rejects_empty_observed_mask():
    mask = np.zeros((2, 3, 4), dtype=bool)

    with pytest.raises(ValueError):
        hide_random(mask, fraction=0.2)


def test_score_perfect_prediction():
    truth = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    pred = truth.copy()

    holdout = (
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 0, 1]),
    )

    # score(truth, pred, holdout)
    metrics = score(truth, pred, holdout)

    assert isinstance(metrics, Metrics)
    assert metrics.n == 4
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.mae == pytest.approx(0.0)
    assert metrics.r2 == pytest.approx(1.0)
    assert metrics.r == pytest.approx(1.0)


def test_score_nonperfect_prediction():
    truth = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    pred = np.array([[[1.0, 2.5, 2.0, 5.0]]])

    holdout = (
        np.array([0, 0, 0, 0]),
        np.array([0, 0, 0, 0]),
        np.array([0, 1, 2, 3]),
    )

    metrics = score(truth, pred, holdout)

    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.5, 2.0, 5.0])
    err = y_pred - y_true

    expected_rmse = np.sqrt(np.mean(err**2))
    expected_mae = np.mean(np.abs(err))
    expected_r2 = 1.0 - np.sum(err**2) / np.sum((y_true - y_true.mean()) ** 2)
    expected_r = np.corrcoef(y_true, y_pred)[0, 1]

    assert metrics.n == 4
    assert metrics.rmse == pytest.approx(expected_rmse)
    assert metrics.mae == pytest.approx(expected_mae)
    assert metrics.r2 == pytest.approx(expected_r2)
    assert metrics.r == pytest.approx(expected_r)


def test_score_ignores_nan_values():
    truth = np.array([[[1.0, 2.0, np.nan, 4.0]]])
    pred = np.array([[[1.0, 2.5, 3.0, np.nan]]])

    holdout = (
        np.array([0, 0, 0, 0]),
        np.array([0, 0, 0, 0]),
        np.array([0, 1, 2, 3]),
    )

    metrics = score(truth, pred, holdout)

    # Only first two entries are valid:
    # truth = [1.0, 2.0], pred = [1.0, 2.5]
    assert metrics.n == 2
    assert metrics.rmse == pytest.approx(np.sqrt((0.0**2 + 0.5**2) / 2))
    assert metrics.mae == pytest.approx(0.25)


def test_score_returns_nan_when_no_valid_entries():
    truth = np.array([[[np.nan, np.nan]]])
    pred = np.array([[[1.0, 2.0]]])

    holdout = (
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([0, 1]),
    )

    metrics = score(truth, pred, holdout)

    assert metrics.n == 0
    assert np.isnan(metrics.rmse)
    assert np.isnan(metrics.mae)
    assert np.isnan(metrics.r2)
    assert np.isnan(metrics.r)


def test_score_rejects_shape_mismatch():
    truth = np.ones((2, 3, 4))
    pred = np.ones((2, 3, 5))

    holdout = (
        np.array([0]),
        np.array([0]),
        np.array([0]),
    )

    with pytest.raises(ValueError):
        score(truth, pred, holdout)
from __future__ import annotations

import numpy as np
import pytest

from geotempfill.bayesian import EmpiricalBayesResult, empirical_bayes_fill


def test_empirical_bayes_preserves_observed_entries():
    rng = np.random.default_rng(0)
    tensor = rng.normal(size=(2, 5, 4))
    mask = rng.random(tensor.shape) > 0.3
    mask[:, 0, :] = True

    out = empirical_bayes_fill(tensor, mask)

    np.testing.assert_allclose(out[mask], tensor[mask])


def test_empirical_bayes_recovers_constant_field():
    tensor = np.full((2, 5, 4), 12.0)
    mask = np.ones_like(tensor, dtype=bool)
    mask[0, 2, 1] = False
    mask[1, 4, 3] = False

    out = empirical_bayes_fill(tensor, mask)

    np.testing.assert_allclose(out[~mask], 12.0)


def test_empirical_bayes_can_return_result_object():
    tensor = np.ones((1, 3, 2))
    mask = np.ones_like(tensor, dtype=bool)
    mask[0, 1, 1] = False

    result = empirical_bayes_fill(tensor, mask, return_result=True)

    assert isinstance(result, EmpiricalBayesResult)
    assert result.completed.shape == tensor.shape
    assert result.variable_mean.shape == (1,)
    assert result.time_effect.shape == (1, 3)
    assert result.station_effect.shape == (1, 2)
    assert result.n_iter >= 1


def test_empirical_bayes_rejects_invalid_inputs():
    tensor = np.zeros((1, 2, 3))

    with pytest.raises(ValueError):
        empirical_bayes_fill(tensor, np.ones((1, 2), dtype=bool))

    with pytest.raises(ValueError):
        empirical_bayes_fill(tensor, np.zeros_like(tensor, dtype=bool))

    with pytest.raises(ValueError):
        empirical_bayes_fill(tensor, np.ones_like(tensor, dtype=bool), shrinkage=-1.0)

    with pytest.raises(ValueError):
        empirical_bayes_fill(
            tensor,
            np.ones_like(tensor, dtype=bool),
            coords=np.zeros((2, 2)),
        )

from __future__ import annotations

import numpy as np
import pytest

from geotempfill.methods import DEFAULT_METHODS, METHODS, run_fill_method


def test_method_registry_contains_default_methods():
    for name in DEFAULT_METHODS:
        assert name in METHODS


def test_run_fill_method_returns_label_and_completed_tensor():
    tensor = np.array([[[1.0, 2.0], [3.0, 0.0]]])
    mask = np.array([[[True, True], [True, False]]])
    coords = np.array([[0.0, 0.0], [0.0, 1.0]])

    label, completed = run_fill_method(
        "mean",
        data=tensor,
        mask=mask,
        coords=coords,
    )

    assert label == "MeanFill"
    assert completed.shape == tensor.shape
    np.testing.assert_allclose(completed[mask], tensor[mask])


def test_run_fill_method_rejects_unknown_method():
    with pytest.raises(ValueError):
        run_fill_method("not_a_method", data=np.zeros((1, 1, 1)), mask=np.ones((1, 1, 1), dtype=bool))

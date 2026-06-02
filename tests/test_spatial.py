from __future__ import annotations

import numpy as np
import pytest

from geotempfill.spatial import cokriging_fill, kriging_fill


def test_kriging_recovers_constant_field():
    tensor = np.full((1, 2, 4), 12.0)
    mask = np.ones_like(tensor, dtype=bool)
    mask[0, 0, 2] = False

    coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    out = kriging_fill(tensor, mask, coords=coords)

    assert out[0, 0, 2] == pytest.approx(12.0)


def test_kriging_preserves_observed_entries():
    rng = np.random.default_rng(0)
    tensor = rng.normal(size=(2, 4, 5))
    mask = rng.random(tensor.shape) > 0.3
    mask[:, :, 0] = True

    coords = np.array(
        [
            [34.0, -118.0],
            [35.0, -118.2],
            [36.0, -119.0],
            [37.0, -120.0],
            [38.0, -121.0],
        ]
    )

    out = kriging_fill(tensor, mask, coords=coords)

    np.testing.assert_allclose(out[mask], tensor[mask])


def test_kriging_rejects_bad_coords_shape():
    tensor = np.zeros((1, 2, 3))
    mask = np.ones_like(tensor, dtype=bool)

    with pytest.raises(ValueError):
        kriging_fill(tensor, mask, coords=np.zeros((2, 2)))


def test_cokriging_recovers_constant_field():
    tensor = np.full((2, 2, 4), 12.0)
    mask = np.ones_like(tensor, dtype=bool)
    mask[0, 0, 2] = False

    coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    out = cokriging_fill(tensor, mask, coords=coords)

    assert out[0, 0, 2] == pytest.approx(12.0)


def test_cokriging_preserves_observed_entries():
    rng = np.random.default_rng(1)
    tensor = rng.normal(size=(3, 4, 5))
    mask = rng.random(tensor.shape) > 0.35
    mask[:, :, 0] = True

    coords = np.array(
        [
            [34.0, -118.0],
            [35.0, -118.2],
            [36.0, -119.0],
            [37.0, -120.0],
            [38.0, -121.0],
        ]
    )

    out = cokriging_fill(tensor, mask, coords=coords)

    np.testing.assert_allclose(out[mask], tensor[mask])

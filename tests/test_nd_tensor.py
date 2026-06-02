from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geotempfill.tensor import NDTensor, build_nd_tensor


def test_build_nd_tensor_basic_shape_and_values():
    df = pd.DataFrame(
        {
            "syndrome": ["flu", "flu", "gi", "gi"],
            "quarter": ["Q1", "Q2", "Q1", "Q2"],
            "city": ["A", "A", "B", "B"],
            "count": [10, 12, 3, 5],
        }
    )

    tensor = build_nd_tensor(
        df,
        index_cols=["syndrome", "quarter", "city"],
        value_col="count",
    )

    assert isinstance(tensor, NDTensor)
    assert tensor.dims == ["syndrome", "quarter", "city"]
    assert tensor.shape == (2, 2, 2)
    assert tensor.mask.sum() == 4

    flu = tensor.coordinates["syndrome"].index("flu")
    q1 = tensor.coordinates["quarter"].index("Q1")
    city_a = tensor.coordinates["city"].index("A")

    assert tensor.data[flu, q1, city_a] == 10.0


def test_build_nd_tensor_explicit_coordinates_keep_missing_cells():
    df = pd.DataFrame(
        {
            "virus": ["a"],
            "week": [1],
            "region": ["north"],
            "rate": [2.5],
        }
    )

    tensor = build_nd_tensor(
        df,
        index_cols=["virus", "week", "region"],
        value_col="rate",
        coordinates={
            "virus": ["a", "b"],
            "week": [1, 2],
            "region": ["north", "south"],
        },
    )

    assert tensor.shape == (2, 2, 2)
    assert tensor.missing_rate == pytest.approx(7 / 8)
    assert np.isfinite(tensor.fill_for_algorithm()).all()

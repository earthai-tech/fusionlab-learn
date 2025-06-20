"""
Unit tests for `make_dict_to_tuple_fn`.

Run with::

    pytest tests/test_make_dict_to_tuple_fn.py
or simply::

    python tests/test_make_dict_to_tuple_fn.py
"""

from typing import Dict

import numpy as np
import pytest
import tensorflow as tf

from fusionlab.nn.utils import make_dict_to_tuple_fn  


# Helper – quick tensor factory (content is irrelevant for shape tests)
def t(shape):
    """Return a float32 tensor of zeros with the requested shape."""
    return tf.constant(np.zeros(shape), dtype=tf.float32)


# FIXTURES
@pytest.fixture
def full_feature_dict() -> Dict[str, tf.Tensor]:
    """All feature keys present."""
    return {
        "coords":           t((2, 6, 3)),
        "dynamic_features": t((2, 6, 4)),
        "static_features":  t((2, 5)),
        "future_features":  t((2, 3, 1)),
        "subsidence":       t((2, 3, 1)),
        "gwl":              t((2, 3, 1)),
    }


@pytest.fixture
def partial_feature_dict(full_feature_dict):
    """Missing 'future_features' key."""
    d = full_feature_dict.copy()
    d.pop("future_features")
    return d


# TESTS

#pass-through target behaviour
def test_pass_through_targets(full_feature_dict):
    mapper = make_dict_to_tuple_fn(
        ["coords", "dynamic_features", "static_features", "future_features"]
    )
    targets_in = t((2, 3, 1))
    (coords, dyn, stat, fut), targets_out = mapper(full_feature_dict, targets_in)

    assert coords.shape[-1] == 3
    assert dyn.shape[-1]    == 4
    assert stat.shape[-1]   == 5
    assert fut.shape[-1]    == 1
    tf.debugging.assert_equal(targets_in, targets_out)


#extract multiple keys as dict
def test_extract_target_dict(full_feature_dict):
    mapper = make_dict_to_tuple_fn(
        ["coords", "dynamic_features"],   # feature tuple
        target_keys=["subsidence", "gwl"]
    )
    (coords, dyn), targets = mapper(full_feature_dict)

    assert isinstance(targets, dict)
    assert set(targets.keys()) == {"subsidence", "gwl"}


#missing optional feature tolerated
def test_allow_missing_optional(partial_feature_dict):
    mapper = make_dict_to_tuple_fn(
        ["coords", "dynamic_features", "static_features", "future_features"]
    )
    (coords, dyn, stat, fut), _ = mapper(partial_feature_dict)

    assert fut is None                                    # placeholder
    assert coords is partial_feature_dict["coords"]


# missing optional feature raises when disabled
def test_missing_feature_raises(partial_feature_dict):
    mapper = make_dict_to_tuple_fn(
        ["coords", "dynamic_features", "static_features", "future_features"],
        allow_missing_optional=False
    )
    with pytest.raises(KeyError):
        mapper(partial_feature_dict)


# ------------------------------------------------------------------
# PYTHON ENTRY-POINT – lets you run this file directly
# ------------------------------------------------------------------
if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__]))

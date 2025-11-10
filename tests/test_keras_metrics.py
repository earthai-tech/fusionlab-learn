
import numpy as np
import pytest
import tensorflow as tf

from fusionlab.nn.keras_metrics import (
    _split_interval,
    CentralCoverage,
    IntervalSharpness,
)

# ---------------------------
# Unit tests for _split_interval
# ---------------------------

def test_split_interval_tuple():
    lo = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
    hi = tf.constant([2.0, 3.0, 4.0], dtype=tf.float32)
    a, b = _split_interval((lo, hi))
    assert a is lo and b is hi
    np.testing.assert_allclose(a.numpy(), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(b.numpy(), [2.0, 3.0, 4.0])


def test_split_interval_lastdim2():
    # shape (..., 2) packs (lo, hi)
    packed = tf.constant([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]], dtype=tf.float32)
    lo, hi = _split_interval(packed)
    np.testing.assert_allclose(lo.numpy(), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(hi.numpy(), [2.0, 3.0, 4.0])


def test_split_interval_quantile_axis():
    # Quantile stack [0.1, 0.5, 0.9] along axis=1
    yq = tf.constant(
        [[0.0, 1.0, 2.0],   # sample 1: (q10, q50, q90)
         [1.0, 2.0, 3.0],   # sample 2
         [2.0, 3.0, 4.0]],  # sample 3
        dtype=tf.float32
    )
    lo, hi = _split_interval(y_pred_interval=yq, q_axis=1, lo_index=0, hi_index=-1)
    np.testing.assert_allclose(lo.numpy(), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(hi.numpy(), [2.0, 3.0, 4.0])


# ---------------------------
# Unit tests for metrics (standalone)
# ---------------------------

def test_central_coverage_unweighted_with_tuple():
    # y ∈ R^5, central interval (lo, hi) as tuple
    y = tf.constant([0.0, 1.0, 2.0,  5.0, -1.0], dtype=tf.float32)
    lo = tf.constant([-0.5, 0.0, 1.5, 4.5, -2.0], dtype=tf.float32)
    hi = tf.constant([ 0.5, 2.0, 2.5, 4.9, -1.5], dtype=tf.float32)
    # hits at indices: 0 (0 in [-0.5,0.5]), 1, 2; miss at 3 (5.0 not <=4.9), 4 (-1.0 not >= -1.5? actually yes, -1.0 ∈ [-2.0,-1.5]? no, -1.0 > -1.5, but hi=-1.5 < -1.0 so miss)
    # total hits = 3/5 = 0.6
    m = CentralCoverage()
    m.update_state(y_true=y, y_pred_interval=(lo, hi))
    cov = float(m.result().numpy())
    assert np.isclose(cov, 0.6, atol=1e-6)


def test_central_coverage_weighted():
    y = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
    lo = tf.constant([-1.0, 1.1, 1.9], dtype=tf.float32)
    hi = tf.constant([ 1.0, 1.9, 2.1], dtype=tf.float32)
    # hits: [True, False, True] with weights [1, 2, 3] => (1*1 + 0*2 + 1*3)/(1+2+3)=4/6=2/3
    w = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    m = CentralCoverage()
    m.update_state(y_true=y, y_pred_interval=(lo, hi), sample_weight=w)
    cov = float(m.result().numpy())
    assert np.isclose(cov, 2/3, atol=1e-6)


def test_interval_sharpness_unweighted_with_tuple():
    lo = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
    hi = tf.constant([1.0, 2.5, 2.3], dtype=tf.float32)
    # widths: [1.0, 1.5, 0.3] => mean = (1+1.5+0.3)/3 = 2.8/3
    m = IntervalSharpness()
    m.update_state(y_true=tf.zeros_like(lo), y_pred_interval=(lo, hi))
    shp = float(m.result().numpy())
    assert np.isclose(shp, (1.0 + 1.5 + 0.3) / 3.0, atol=1e-6)


def test_interval_sharpness_weighted():
    lo = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
    hi = tf.constant([1.0, 3.0, 2.5], dtype=tf.float32)
    w  = tf.constant([1.0, 2.0, 1.0], dtype=tf.float32)
    # widths: [1.0, 2.0, 0.5]; weighted mean = (1*1 + 2*2 + 1*0.5) / (1+2+1) = (1 + 4 + 0.5)/4 = 5.5/4 = 1.375
    m = IntervalSharpness()
    m.update_state(y_true=tf.zeros_like(lo), y_pred_interval=(lo, hi), sample_weight=w)
    shp = float(m.result().numpy())
    assert np.isclose(shp, 1.375, atol=1e-6)


def test_metrics_reset_state():
    y = tf.constant([0.0, 0.0], dtype=tf.float32)
    lo = tf.constant([-1.0, -1.0], dtype=tf.float32)
    hi = tf.constant([ 1.0,  1.0], dtype=tf.float32)

    cov = CentralCoverage()
    cov.update_state(y, (lo, hi))
    assert float(cov.result().numpy()) == 1.0
    cov.reset_state()
    # after reset, result should be NaN-safe zero (0/0 -> 0 via divide_no_nan)
    assert float(cov.result().numpy()) == 0.0

    shp = IntervalSharpness()
    shp.update_state(y, (lo, hi))
    assert float(shp.result().numpy()) == 2.0  # width=2.0, averaged over 2 → 2.0
    shp.reset_state()
    assert float(shp.result().numpy()) == 0.0


# ---------------------------
# Keras integration test (model.evaluate)
# ---------------------------

def test_model_evaluate_with_interval_metrics():
    """
    Build a tiny model that emits (lo, hi) in the last dimension.
    Verify that Keras evaluation returns expected coverage and sharpness.
    """
    # Simple dataset: x ~ N(0,1), y_true = x (inside interval [x-1, x+1])
    # Coverage should be 1.0; sharpness (mean width) should be exactly 2.0.
    n = 32
    rng = np.random.RandomState(0)
    x = rng.randn(n, 1).astype("float32")
    y = x.copy().astype("float32")

    inputs = tf.keras.Input(shape=(1,), name="x")
    lo = inputs - 1.0
    hi = inputs + 1.0
    y_pred = tf.concat([lo, hi], axis=-1)   # shape (None, 2) → interpreted as (lo, hi)

    model = tf.keras.Model(inputs=inputs, outputs=y_pred)
    # Use a dummy loss; metrics read intervals from y_pred directly
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[CentralCoverage(name="cov"), IntervalSharpness(name="shp")],
    )

    # Keras will pass y_true to metrics (unused for sharpness; used for coverage)
    out = model.evaluate(x, y, verbose=0, return_dict=True)

    # Allow tiny numeric tolerance
    assert np.isclose(out["cov"], 1.0, atol=1e-6)
    assert np.isclose(out["shp"], 2.0, atol=1e-6)


# ---------------------------
# Optional: test using quantile stacks (0.1, 0.5, 0.9)
# ---------------------------

def test_metrics_with_quantile_stack_pre_sliced():
    """
    Simulate a (B, Q) quantile stack with Q=3 for [0.1, 0.5, 0.9].
    Pre-slice lo=q[:,0], hi=q[:,-1] and feed as (lo, hi) tuple.
    """
    # Ground truth
    y = tf.constant([0.0,  2.0,  5.0], dtype=tf.float32)
    # Quantiles stacked along axis=1: (q10, q50, q90)
    q = tf.constant(
        [[-1.0,  0.0,  1.0],   # y=0 inside [-1, 1]
         [ 1.5,  2.0,  2.5],   # y=2 inside [1.5, 2.5]
         [ 3.0,  4.0,  4.9]],  # y=5 outside [3.0, 4.9]
        dtype=tf.float32
    )
    lo = q[:, 0]       # q10
    hi = q[:, -1]      # q90

    cov = CentralCoverage()
    cov.update_state(y_true=y, y_pred_interval=(lo, hi))
    # Two of three are inside => 2/3
    assert np.isclose(float(cov.result().numpy()), 2/3, atol=1e-6)

    shp = IntervalSharpness()
    shp.update_state(y_true=y, y_pred_interval=(lo, hi))
    # widths: [2.0, 1.0, 1.9] => mean = (2 + 1 + 1.9)/3 = 4.9/3
    assert np.isclose(float(shp.result().numpy()), 4.9/3.0, atol=1e-6)

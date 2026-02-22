# tests/test_keras_metrics_shapes.py
import numpy as np
import pytest

# tf = pytest.importorskip("tensorflow")
import tensorflow as tf 

from fusionlab.nn.keras_metrics import (
    _infer_quantile_axis,
    coverage80_fn,
    sharpness80_fn,
    mae_q50_fn,
    mse_q50_fn,
    make_coverage80,
)


def _as_tf(x):
    return tf.convert_to_tensor(x, dtype=tf.float32)


def _assert_scalar_close(t, expected, tol=1e-6):
    assert isinstance(t, tf.Tensor)
    assert t.shape.rank == 0, f"Expected scalar tensor, got shape {t.shape}"
    v = float(t.numpy())
    assert np.isfinite(v), f"Non-finite metric value: {v}"
    assert v == pytest.approx(float(expected), abs=tol)


@pytest.mark.parametrize("B,H,O", [(2, 3, 1), (3, 2, 2)])
def test_metrics_quantiles_rank4_BHQO(B, H, O):
    """
    y_true: (B,H,O)
    y_pred: (B,H,Q,O) with Q=3 = (q10,q50,q90)
    """
    y = np.arange(B * H * O, dtype=np.float32).reshape(B, H, O)

    q10 = y - 1.0
    q50 = y
    q90 = y + 1.0
    yp = np.stack([q10, q50, q90], axis=2)  # (B,H,3,O)

    y_t = _as_tf(y)
    yp_t = _as_tf(yp)

    _assert_scalar_close(coverage80_fn(y_t, yp_t), 1.0)
    _assert_scalar_close(sharpness80_fn(y_t, yp_t), 2.0)
    _assert_scalar_close(mae_q50_fn(y_t, yp_t), 0.0)
    _assert_scalar_close(mse_q50_fn(y_t, yp_t), 0.0)


@pytest.mark.parametrize("B,H", [(2, 3), (4, 1)])
def test_metrics_quantiles_rank3_BHQ_ytrue_BH(B, H):
    """
    y_true: (B,H)
    y_pred: (B,H,Q) with Q=3
    """
    y = np.arange(B * H, dtype=np.float32).reshape(B, H)

    q10 = y - 1.0
    q50 = y
    q90 = y + 1.0
    yp = np.stack([q10, q50, q90], axis=2)  # (B,H,3)

    y_t = _as_tf(y)
    yp_t = _as_tf(yp)

    _assert_scalar_close(coverage80_fn(y_t, yp_t), 1.0)
    _assert_scalar_close(sharpness80_fn(y_t, yp_t), 2.0)
    _assert_scalar_close(mae_q50_fn(y_t, yp_t), 0.0)
    _assert_scalar_close(mse_q50_fn(y_t, yp_t), 0.0)


@pytest.mark.parametrize("B,H", [(2, 3), (3, 2)])
def test_metrics_quantiles_rank3_BHQ_ytrue_BH1(B, H):
    """
    y_true: (B,H,1)
    y_pred: (B,H,Q) -> requires the metric's (rank mismatch) expand_dims path
    """
    y2 = np.arange(B * H, dtype=np.float32).reshape(B, H)
    y = y2[..., None]  # (B,H,1)

    q10 = y2 - 1.0
    q50 = y2
    q90 = y2 + 1.0
    yp = np.stack([q10, q50, q90], axis=2)  # (B,H,3)

    y_t = _as_tf(y)
    yp_t = _as_tf(yp)

    _assert_scalar_close(coverage80_fn(y_t, yp_t), 1.0)
    _assert_scalar_close(sharpness80_fn(y_t, yp_t), 2.0)
    _assert_scalar_close(mae_q50_fn(y_t, yp_t), 0.0)
    _assert_scalar_close(mse_q50_fn(y_t, yp_t), 0.0)


@pytest.mark.parametrize("B,H", [(2, 3), (4, 2)])
def test_metrics_packed_interval_rank3_BH2(B, H):
    """
    y_true: (B,H)
    y_pred: (B,H,2) packed (lo,hi)
    """
    y = np.arange(B * H, dtype=np.float32).reshape(B, H)
    lo = y - 1.0
    hi = y + 1.0
    yp = np.stack([lo, hi], axis=-1)  # (B,H,2)

    y_t = _as_tf(y)
    yp_t = _as_tf(yp)

    _assert_scalar_close(coverage80_fn(y_t, yp_t), 1.0)
    _assert_scalar_close(sharpness80_fn(y_t, yp_t), 2.0)
    _assert_scalar_close(mae_q50_fn(y_t, yp_t), 0.0)  # midpoint == y
    _assert_scalar_close(mse_q50_fn(y_t, yp_t), 0.0)


def test_make_coverage80_fixed_axis_matches_coverage80_fn():
    """
    make_coverage80(q_axis=2) should match coverage80_fn on (B,H,Q,O).
    """
    B, H, O = 2, 3, 1
    y = np.arange(B * H * O, dtype=np.float32).reshape(B, H, O)

    q10 = y - 1.0
    q50 = y
    q90 = y + 1.0
    yp = np.stack([q10, q50, q90], axis=2)  # (B,H,3,O)

    y_t = _as_tf(y)
    yp_t = _as_tf(yp)

    cov_fixed = make_coverage80(q_axis=2)
    _assert_scalar_close(cov_fixed(y_t, yp_t), 1.0)
    _assert_scalar_close(coverage80_fn(y_t, yp_t), 1.0)


# --------------------------------------------------------------------
# Axis inference tests (shape-only)
# --------------------------------------------------------------------

def test_infer_quantile_axis_strong_convention_rank4_BHQO():
    x = tf.zeros((2, 3, 3, 1), dtype=tf.float32)
    assert _infer_quantile_axis(x, n_q=3) == 2


def test_infer_quantile_axis_strong_convention_rank3_BHQ():
    x = tf.zeros((2, 3, 3), dtype=tf.float32)
    assert _infer_quantile_axis(x, n_q=3) == 2

def test_infer_quantile_axis_rank4_quantiles_last_dim():
    x = tf.zeros((2, 3, 1, 3), dtype=tf.float32)  # (B,H,O,Q)
    assert _infer_quantile_axis(x, n_q=3) == 3


def test_infer_quantile_axis_packed_interval_rank4_BHO2_returns_none():
    x = tf.zeros((2, 3, 1, 2), dtype=tf.float32)  # (B,H,O,2) packed
    assert _infer_quantile_axis(x, n_q=3) is None


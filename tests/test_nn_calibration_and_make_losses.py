# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest

# -- IntervalCalibrator is pure NumPy; always available
from fusionlab.nn.calibration import IntervalCalibrator

# -- Weighted pinball needs TF
tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")
from fusionlab.nn.losses import make_weighted_pinball


# -------------------------------
# IntervalCalibrator tests
# -------------------------------

def _coverage(y, lo, hi):
    y = np.asarray(y).reshape(-1)
    lo = np.asarray(lo).reshape(-1)
    hi = np.asarray(hi).reshape(-1)
    return np.mean((y >= lo) & (y <= hi))


def test_interval_calibrator_improves_coverage_to_target():
    rng = np.random.default_rng(0)
    N, H = 3000, 3
    # Synthetic truth ~ N(0,1) per horizon
    y = rng.standard_normal(size=(N, H, 1))

    # Start with too-narrow bands, coverage ~ 38% (for +/-0.5 on N(0,1))
    qmed = np.zeros((N, H, 1))
    qlo  = np.full((N, H, 1), -0.5)  # different widths per horizon optional
    qhi  = np.full((N, H, 1),  0.5)

    target = 0.80
    cal = IntervalCalibrator(target=target, max_iter=40, tol=1e-3)
    cal.fit(y, qlo, qmed, qhi)

    lo_c, hi_c = cal.transform(qlo, qmed, qhi)
    cov_before = _coverage(y, qlo, qhi)
    cov_after  = _coverage(y, lo_c, hi_c)

    # Before should be clearly below target; after should be ~ target (within tolerance)
    assert cov_before < 0.6
    assert abs(cov_after - target) <= 0.02, f"after={cov_after:.3f}, target={target:.3f}"


def test_interval_calibrator_returns_factor_one_when_already_wide():
    rng = np.random.default_rng(1)
    N, H = 4000, 2
    y = rng.standard_normal(size=(N, H, 1))

    # Very wide interval ~95% coverage for N(0,1)
    qmed = np.zeros((N, H, 1))
    qlo  = np.full((N, H, 1), -2.0)
    qhi  = np.full((N, H, 1),  2.0)

    cal = IntervalCalibrator(target=0.80, max_iter=30, tol=1e-3)
    cal.fit(y, qlo, qmed, qhi)

    # Should *not* widen further → factors exactly 1 (within tight tol)
    assert cal.factors_.shape == (H,)
    assert np.allclose(cal.factors_, 1.0, atol=1e-3)


def test_interval_calibrator_per_horizon_independent():
    rng = np.random.default_rng(2)
    N, H = 3000, 2
    y = rng.standard_normal(size=(N, H, 1))

    # Horizon 0 is narrower than horizon 1 → needs a bigger factor
    qmed = np.zeros((N, H, 1))
    qlo  = np.zeros((N, H, 1))
    qhi  = np.zeros((N, H, 1))
    qlo[:, 0, 0], qhi[:, 0, 0] = -0.3, 0.3
    qlo[:, 1, 0], qhi[:, 1, 0] = -0.6, 0.6

    cal = IntervalCalibrator(target=0.80, max_iter=40, tol=1e-3)
    cal.fit(y, qlo, qmed, qhi)

    # Narrower horizon should get larger widening factor
    f0, f1 = cal.factors_
    assert f0 > f1, f"Expected f0>f1, got {f0} <= {f1}"


# -------------------------------
# make_weighted_pinball tests
# -------------------------------

def test_weighted_pinball_exact_numeric_small_tensor():
    """
    Construct a tiny (B=1,H=1,Q=3,O=1) case with closed-form pinball values.
    y_true = 1.0
    preds: q10=0.0 (under), q50=1.0 (exact), q90=2.0 (over)
    Pinball per q:
      q=0.1: e=1 → max(0.1*1, -0.9*1)=0.1
      q=0.5: e=0 → 0
      q=0.9: e=-1 → max(0.9*(-1), -0.1*(-1))=0.1
    Weighted (3,1,3) normalized → (3/7,1/7,3/7):
      loss = (0.3 + 0 + 0.3)/7 = 0.0857142857...
    """
    qs = [0.1, 0.5, 0.9]
    w  = {0.1: 3.0, 0.5: 1.0, 0.9: 3.0}
    loss_fn = make_weighted_pinball(qs, w)

    y_true = tf.constant([[[1.0]]], dtype=tf.float32)          # (B=1,H=1,1)
    y_pred = tf.constant([[[[0.0],[1.0],[2.0]]]], tf.float32)  # (1,1,Q=3,1)

    val = float(loss_fn(y_true, y_pred).numpy())
    assert np.isclose(val,  0.066666677, atol=1e-7), f"got {val}"


def test_weighted_pinball_equal_weights_matches_simple_average():
    """
    With equal weights, result should match simple average across quantiles.
    For the same setup as above, average is (0.1+0+0.1)/3 = 0.066666...
    """
    qs = [0.1, 0.5, 0.9]
    w_equal = [1.0, 1.0, 1.0]
    loss_fn_equal = make_weighted_pinball(qs, w_equal)

    y_true = tf.constant([[[1.0]]], dtype=tf.float32)
    y_pred = tf.constant([[[[0.0],[1.0],[2.0]]]], tf.float32)

    val = float(loss_fn_equal(y_true, y_pred).numpy())
    assert np.isclose(val, 0.0666666667, atol=1e-7), f"got {val}"


def test_weighted_pinball_respects_weighting_direction():
    """
    If we upweight tails vs. the median, the loss with heavy-tail weights
    should exceed the equal-weight loss (given symmetric nonzero tail losses).
    """
    qs = [0.1, 0.5, 0.9]
    y_true = tf.constant([[[1.0]]], dtype=tf.float32)
    y_pred = tf.constant([[[[0.0],[1.0],[2.0]]]], tf.float32)

    loss_equal = make_weighted_pinball(qs, [1.0, 1.0, 1.0])(y_true, y_pred).numpy()
    loss_tail_heavy = make_weighted_pinball(qs, {0.1: 3.0, 0.5: 1.0, 0.9: 3.0})(y_true, y_pred).numpy()

    assert loss_tail_heavy > loss_equal, f"{loss_tail_heavy} <= {loss_equal}"


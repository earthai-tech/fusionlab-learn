import numpy as np
import pytest

from fusionlab.metrics import (
    coverage_score,
    crps_score,
    weighted_interval_score,
    prediction_stability_score,
    time_weighted_mean_absolute_error,
    quantile_calibration_error,
    mean_interval_width_score,
    theils_u_score,
)


def test_coverage_score_basic():
    """Coverage: 2 of 3 within intervals → 2/3."""
    y_true  = np.array([1, 2, 3])
    y_lower = np.array([0, 2, 4])
    y_upper = np.array([2, 3, 5])
    assert coverage_score(y_true, y_lower, y_upper) == pytest.approx(2/3)


def test_coverage_score_nan_policy():
    y_true  = np.array([1, np.nan])
    y_lower = np.array([0, 0])
    y_upper = np.array([2, 4])

    # raise on NaN
    with pytest.raises(ValueError):
        coverage_score(y_true, y_lower, y_upper, nan_policy='raise')
    # omit NaN → only first sample → covered → 1.0
    assert coverage_score(y_true, y_lower, y_upper, nan_policy='omit') == 1.0
    # propagate → result should be NaN
    assert np.isnan(
        coverage_score(y_true, y_lower, y_upper, nan_policy='propagate')
    )


def test_crps_score_perfect_and_nan():
    # Perfect ensemble = zero CRPS
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert crps_score(y_true, y_pred) == pytest.approx(0.0)

    # NaN handling
    y_true2 = np.array([0.0, np.nan])
    y_pred2 = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError):
        crps_score(y_true2, y_pred2, nan_policy='raise')
    assert crps_score(y_true2, y_pred2, nan_policy='omit') == pytest.approx(0.0)
    assert np.isnan(crps_score(y_true2, y_pred2, nan_policy='propagate'))


def test_weighted_interval_score_simple():
    y   = np.array([1.0, 2.0])
    lows = np.array([[0.0], [1.0]])
    ups  = np.array([[2.0], [3.0]])
    med = np.array([1.0, 2.0])
    alpha = [0.5]
    # interval score = 0.25, abs_err=0, WIS = 0.25/(1+1)=0.125
    wis = weighted_interval_score(y, lows, ups, med, alpha)
    assert wis == pytest.approx(0.125)


def test_prediction_stability_score():
    # Constant forecasts → zero jumps → PSS = 0
    y_pred = np.array([[1, 1, 1], [2, 2, 2]])
    assert prediction_stability_score(y_pred) == pytest.approx(0.0)

    # Single-sample, step of 1 → mean jump = 1
    y_pred2 = np.array([[1, 2, 3]])
    assert prediction_stability_score(y_pred2) == pytest.approx(1.0)


def test_time_weighted_mae_default_weights():
    y_true = np.array([[1.0, 2.0]])
    y_pred = np.array([[2.0, 2.0]])
    # diffs [1,0], inverse-time w=[2/3,1/3] → weighted MAE = 2/3
    twmae = time_weighted_mean_absolute_error(y_true, y_pred)
    assert twmae == pytest.approx(2/3)


def test_quantile_calibration_error():
    y_true = np.array([1, 2, 3, 4])
    # q=0.5 always under/at → prop=1 → err=|1-0.5|=0.5
    # q=1.0 always under/at → prop=1 → err=|1-1|=0
    q_levels = [0.5, 1.0]
    y_pred = np.vstack([
        [1, 1, 4, 4],  # for q=0.5
        [1, 2, 3, 4],  # for q=1.0
    ]).T
    qce = quantile_calibration_error(y_true, y_pred, q_levels)
    assert qce == pytest.approx(0.25)


def test_mean_interval_width_score():
    lows = np.array([1.0, 2.0, np.nan])
    ups  = np.array([2.0, 3.0, 5.0])
    # omit → widths=[1,1] → mean=1
    assert mean_interval_width_score(lows, ups, nan_policy='omit') == pytest.approx(1.0)
    # propagate → NaN in input → result NaN
    assert np.isnan(
        mean_interval_width_score(lows, ups, nan_policy='propagate')
    )


def test_theils_u_score():
    y_true = np.array([[1, 2, 3], [2, 2, 2]])
    # perfect forecast → U=0
    assert theils_u_score(y_true, y_true) == pytest.approx(0.0)
    # persistence baseline → U=1
    y_pred = np.column_stack([y_true[:, 0], y_true[:, :-1]])
    assert theils_u_score(y_true, y_pred, nan_policy='omit') == pytest.approx(1.0)
    # propagate → NaN in true → result NaN
    y_true2 = y_true.copy()
    y_true2[0, 2] = np.nan
    assert np.isnan(
        theils_u_score(y_true2, y_pred, nan_policy='propagate')
    )

if __name__=='__main__': 
    pytest.main([__file__])

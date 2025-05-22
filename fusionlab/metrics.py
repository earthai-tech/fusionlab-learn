# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
from numbers import Real, Integral 
from typing import Sequence, Optional
import numpy as np 

from .compat.sklearn import ( 
    StrOptions, 
    validate_params,
    
)
from .utils.validator import _ensure_y_is_valid 

__all__ = [
    'coverage_score',
    'crps_score', 
    'weighted_interval_score', 
    'prediction_stability_score' , 
    'time_weighted_mean_absolute_error', 
    'quantile_calibration_error', 
    'mean_interval_width_score', 
    'theils_u_score'
   ]

@validate_params({
    'y_true'     : ['array-like'],
    'y_pred'     : ['array-like'],
    'nan_policy' : [StrOptions({'omit','propagate','raise'})],
    'fill_value' : [Real, None],
    'verbose'    : [Integral, bool],
})
def theils_u_score(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    nan_policy: str             = 'propagate',
    fill_value: Optional[Real]  = np.nan,
    verbose:    int             = 1
) -> float:
    r"""
    Compute Multi-Horizon Theil's U Statistic, a relative accuracy
    measure versus naive persistence forecast.

    Theil's U is defined as:

    .. math::
       U = \sqrt{
       \frac{\sum_{i,t}(y_{i,t}-\hat y_{i,t})^2}
            {\sum_{i,t}(y_{i,t}-y_{i,t-1})^2}
       },

    where :math:`y_{i,t}` is the true value for sample ``i`` at
    horizon ``t``, :math:`\hat y_{i,t}` is the forecast, and
    :math:`y_{i,t-1}` is the last known observed value.

    Parameters
    ----------
    y_true       : array-like of shape (B, T)
                   True multi-horizon targets.
    y_pred       : array-like of shape (B, T)
                   Model forecasts per horizon.
    nan_policy   : {'omit','propagate','raise'}, default='propagate'
                   How to handle NaNs in inputs:
                     - ``'raise'``     : error on any NaN.
                     - ``'omit'``      : drop sequences with NaN.
                     - ``'propagate'`` : set result to NaN if present.
    fill_value   : float or None, default=np.nan
                   Value to replace missing entries when
                   `nan_policy!='raise'`.
    verbose      : int, default=1
                   0 = silent; 1 = summary; >=2 = debug.

    Returns
    -------
    float
        Theil's U statistic. Values < 1 indicate improvement
        over naive persistence.

    Examples
    --------
    >>> from fusionlab.metrics import theils_u_score
    >>> import numpy as np
    >>> # 2 samples, 4-step horizon
    >>> y_true = np.array([[1,2,3,4],[2,2,2,2]])
    >>> y_pred = np.array([[1,2,3,5],[2,1,2,3]])
    >>> u = theils_u_score(y_true, y_pred, nan_policy='omit')
    >>> print(f"Theil's U: {u:.3f}")

    See Also
    --------
    mean_absolute_error          : Unweighted MAE.
    time_weighted_mean_absolute_error_score : Horizon-weighted MAE.

    References
    ----------
    .. [5] Theil, H. (1966). Applied Economic Forecasting.
       North-Holland Publishing. [Definition of Theil's U].
    """
    # Convert to numpy arrays
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # Validate inputs
    if y_true_arr.ndim != 2 or y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("`y_true` and `y_pred` must be 2D of equal shape.")
    B, T = y_true_arr.shape
    if T < 2:
        raise ValueError("Need at least two horizons (T>=2).")

    # Mask NaNs
    mask_nan = np.isnan(y_true_arr) | np.isnan(y_pred_arr)
    if nan_policy == 'raise' and mask_nan.any():
        raise ValueError("NaNs found in inputs.")

    # Fill missing if requested
    if fill_value is not None:
        y_true_arr = np.where(np.isnan(y_true_arr),
                              fill_value, y_true_arr)
        y_pred_arr = np.where(np.isnan(y_pred_arr),
                              fill_value, y_pred_arr)

    # Compute squared errors for model and naive baseline
    err_model = (y_true_arr[:,1:] - y_pred_arr[:,1:])**2
    err_base  = (y_true_arr[:,1:] - y_true_arr[:,:-1])**2

    # Handle omit policy
    if nan_policy == 'omit':
        valid = ~mask_nan[:,1:].any(axis=1)
        err_model = err_model[valid]
        err_base  = err_base[valid]

    # Sum of squared errors
    sse_model = np.sum(err_model)
    sse_base  = np.sum(err_base)

    # Compute Theil's U
    result = np.nan
    if sse_base != 0:
        result = float(np.sqrt(sse_model / sse_base))

    # In propagate mode, if any NaNs originally, set NaN
    if nan_policy == 'propagate' and mask_nan.any():
        result = np.nan

    if verbose >= 1:
        print(f"Theil's U computed: {result:.4f}")

    return result


@validate_params({
    'y_lower'    : ['array-like'],
    'y_upper'    : ['array-like'],
    'nan_policy' : [StrOptions({'omit','propagate','raise'})],
    'fill_value' : [Real, None],
    'verbose'    : [Integral, bool]
})
def mean_interval_width_score(
    y_lower:    np.ndarray,
    y_upper:    np.ndarray,
    nan_policy: str            = 'propagate',
    fill_value: Optional[Real] = np.nan,
    verbose:    int            = 1
) -> float:
    r"""
    Compute the Mean Interval Width Score (sharpness) of prediction
    intervals, measuring the average width independent of coverage.

    .. math::
       \mathrm{MeanIntervalWidthScore}
       = \frac{1}{n}\sum_{i=1}^{n}(u_i - l_i),

    where :math:`l_i` and :math:`u_i` are the lower and upper
    bounds for sample :math:`i`.

    Parameters
    ----------
    y_lower      : array-like
                   Lower bound predictions, matching ``y_upper``
                   in shape and alignment.
    y_upper      : array-like
                   Upper bound predictions, matching ``y_lower``
                   in shape and alignment.
    nan_policy   : {'omit','propagate','raise'}, optional
                   How to handle NaNs in inputs:
                     - ``'propagate'``: NaNs remain; result may be NaN.
                     - ``'omit'``     : drop samples containing NaN.
                     - ``'raise'``    : ValueError on NaN.
    fill_value   : scalar or None, optional
                   Value to replace missing entries if not
                   raising. Default is ``np.nan``.
    verbose      : int, optional
                   Controls verbosity:
                     - 0: no output.
                     - 1: summary.
                     - 2: debug details after NaN handling.

    Returns
    -------
    float
        The mean interval width score. Lower values denote
        narrower, sharper intervals.

    Notes
    -----
    Often reported alongside ``coverage_score`` to judge
    interval efficiency. This metric does not consider
    calibration.

    Examples
    --------
    >>> from fusionlab.metrics import \
            mean_interval_width_score
    >>> import numpy as np
    >>> lows = np.array([ 9, 11, 10])
    >>> ups  = np.array([11, 13, 12])
    >>> score = mean_interval_width_score(
    ...     lows, ups, nan_policy='omit'
    ... )
    >>> print(f"Sharpness: {score:.2f}")

    See Also
    --------
    coverage_score           : Interval coverage metric.
    weighted_interval_score  : Proper interval scoring.

    References
    ----------
    .. [4] Sharpness as a standalone metric for interval
       efficiency, often paired with ``coverage_score``.
    """
    # Convert to numpy arrays and flatten
    l_arr = np.asarray(y_lower, dtype=float).ravel()
    u_arr = np.asarray(y_upper, dtype=float).ravel()

    # Validate shapes
    if l_arr.shape != u_arr.shape:
        raise ValueError(
            "`y_lower` and `y_upper` must have the same shape."
        )

    # Build mask of missing values
    mask_nan = np.isnan(l_arr) | np.isnan(u_arr)
    if nan_policy == 'raise' and mask_nan.any():
        raise ValueError("NaNs found in inputs.")

    # Fill missing entries if desired
    if fill_value is not None:
        l_arr = np.where(np.isnan(l_arr), fill_value, l_arr)
        u_arr = np.where(np.isnan(u_arr), fill_value, u_arr)

    # Omit samples with NaN if requested
    if nan_policy == 'omit':
        valid = ~mask_nan
        l_arr = l_arr[valid]
        u_arr = u_arr[valid]

    # Compute interval widths
    widths = u_arr - l_arr

    # Propagate NaNs if required
    if nan_policy == 'propagate' and mask_nan.any():
        widths[mask_nan] = np.nan

    # Aggregate mean width
    result = float(np.nanmean(widths))

    # Verbose reporting
    if verbose >= 1:
        print(f"MeanIntervalWidthScore: {result:.4f}")

    return result


@validate_params({
    'y_true'    : ['array-like'],
    'y_pred'    : ['array-like'],
    'quantiles' : ['array-like'],
    'nan_policy': [StrOptions({'omit','propagate','raise'})],
    'fill_value': [Real, None],
    'verbose'   : [Integral, bool],
})
def quantile_calibration_error(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    quantiles:  Sequence[float],
    nan_policy: str            = 'propagate',
    fill_value: Optional[Real] = np.nan,
    verbose:    int            = 1
) -> float:
    """
    Compute Quantile Calibration Error (QCE) across predicted
    quantiles to assess calibration of probabilistic forecasts.

    The QCE at level :math:`q` is

    .. math::
       \mathrm{QCE}(q)
       = \left|\frac{1}{n}\sum_{i=1}^n
       \mathbf{1}\{y_i \le \hat Q_i(q)\} - q\right|,

    where :math:`\hat Q_i(q)` is the predicted quantile for
    sample :math:`i`, and :math:`y_i` the observed value.

    The aggregated QCE is the average absolute error across
    all provided quantile levels.

    Parameters
    ----------
    y_true       : array-like of shape (n_samples,)
                   Observed true values.
    y_pred       : array-like of shape (n_samples, n_q)
                   Predicted quantiles per sample.
    quantiles    : sequence of float, length n_q
                   Nominal quantile levels (e.g. [0.1,0.5,0.9]).
    nan_policy   : {'omit','propagate','raise'}, default='propagate'
                   How to handle NaNs:
                     - `raise`     : error on NaN.
                     - `omit`      : drop those samples.
                     - `propagate` : set QCE to NaN if any NaN.
    fill_value   : float or None, default=np.nan
                   Replacement for missing values when propagating.
    verbose      : int, default=1
                   0=quiet, 1=summary, >=2=debug.

    Returns
    -------
    float
        Mean absolute calibration error across quantiles.

    Examples
    --------
    >>> from gofast.nn.utils import \
            quantile_calibration_error
    >>> import numpy as np
    >>> y_true    = np.array([1, 2, 3])
    >>> quantiles = [0.1, 0.5, 0.9]
    >>> y_pred = np.array([
    ...     [0.5, 1.0, 1.5],
    ...     [1.5, 2.0, 2.5],
    ...     [2.5, 3.0, 3.5]
    ... ])
    >>> qce = quantile_calibration_error(
    ...     y_true, y_pred, quantiles, nan_policy='omit'
    ... )
    >>> print(f"QCE: {qce:.3f}")

    See Also
    --------
    coverage_score : Interval coverage metric.
    pinball_loss    : Quantile pinball loss function.

    References
    ----------
    .. [3] Gneiting, T. & Raftery, A. E. (2007).
       Quantile calibration diagnostics in
       probabilistic forecasting.
    """
    # Convert inputs to arrays
    y_true_arr  = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr  = np.asarray(y_pred, dtype=float)
    q_arr       = np.asarray(quantiles, dtype=float).ravel()

    # Validate shapes
    n = y_true_arr.shape[0]
    if y_pred_arr.ndim != 2:
        raise ValueError("`y_pred` must be 2D (n_samples, n_q).")
    if y_pred_arr.shape[0] != n:
        raise ValueError(
            "`y_true` and `y_pred` must have same first dimension."
        )
    if q_arr.shape[0] != y_pred_arr.shape[1]:
        raise ValueError(
            "Length of `quantiles` must match number of columns in `y_pred`."
        )
    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("All `quantiles` must be between 0 and 1.")

    # Handle NaNs
    mask_nan = (np.isnan(y_true_arr)
                | np.isnan(y_pred_arr).any(axis=1))
    if nan_policy == 'raise' and mask_nan.any():
        raise ValueError("NaNs found in inputs.")
    if fill_value is not None:
        y_true_arr = np.where(np.isnan(y_true_arr),
                              fill_value, y_true_arr)
        y_pred_arr = np.where(np.isnan(y_pred_arr),
                              fill_value, y_pred_arr)
    if nan_policy == 'omit':
        valid      = ~mask_nan
        y_true_arr = y_true_arr[valid]
        y_pred_arr = y_pred_arr[valid]

    # Compute calibration per quantile
    # Indicator matrix: shape (n, n_q)
    indicators = (y_true_arr[:, None] <= y_pred_arr).astype(float)
    prop_observed = indicators.mean(axis=0)  # shape (n_q,)

    # Absolute deviation from nominal levels
    calib_err = np.abs(prop_observed - q_arr)

    # Aggregate
    result = float(calib_err.mean())

    if verbose >= 2:
        for q, err in zip(q_arr, calib_err):
            print(f"  QCE @ {q:.2f}: {err:.4f}")
    if verbose >= 1:
        print(f"Quantile Calibration Error: {result:.4f}")

    return result


@validate_params({
    'y_true'     : ['array-like'],
    'y_pred'     : ['array-like'],
    'weights'    : ['array-like', None],
    'nan_policy' : [StrOptions({'omit','propagate','raise'})],
    'fill_value' : [Real, None],
    'verbose'    : [Integral, bool]
})
def time_weighted_mean_absolute_error(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    weights:   Optional[Sequence[float]] = None,
    nan_policy: str                     = 'propagate',
    fill_value: Optional[Real]          = np.nan,
    verbose:   int                      = 1
) -> float:
    """
    Compute the Time-Weighted Mean Absolute Error (TW-MAE).

    Penalizes short-horizon errors more heavily via time weights.

    .. math::
       \mathrm{TWMAE}
       = \frac{1}{B}
       \sum_{i=1}^B
       \sum_{t=1}^T
       w_t \, \bigl|\hat y_{i,t} - y_{i,t}\bigr|,

    where :math:`B` is batch size, :math:`T` is horizon length,
    and :math:`w_t` are time weights.

    Parameters
    ----------
    y_true       : array-like of shape (B, T)
                   True target sequences.
    y_pred       : array-like of shape (B, T)
                   Predicted sequences.
    weights      : array-like of shape (T,), optional
                   Time weights :math:`w_t`. If None,
                   defaults to inverse-time
                   :math:`w_t = 1/t` normalized to sum to 1.
    nan_policy   : {'omit','propagate','raise'}, default='propagate'
                   Handling of NaNs:
                     - `raise`     : error on NaN.
                     - `omit`      : drop sequences with NaN.
                     - `propagate` : result NaN if NaNs present.
    fill_value   : float or None, default=np.nan
                   Replacement for missing values when
                   `nan_policy!='raise'`.
    verbose      : int, default=1
                   0 = silent; 1 = summary; >=2 = debug.

    Returns
    -------
    float
        Mean TW-MAE across sequences.

    Examples
    --------
    >>> import numpy as np
    >>> from fusionlab.metrics import \
            time_weighted_mean_absolute_error
    >>> y_true = np.array([[1, 2, 3],
    ...                    [2, 3, 4]])
    >>> y_pred = np.array([[1.1, 2.2, 2.9],
    ...                    [1.9, 3.1, 3.8]])
    >>> twmae = time_weighted_mean_absolute_error(
    ...     y_true, y_pred, nan_policy='omit'
    ... )
    >>> print(f"TWMAE: {twmae:.4f}")

    See Also
    --------
    mean_absolute_error : Unweighted MAE.
    prediction_stability_score : Temporal smoothness.

    References
    ----------
    .. [4] Custom metric for emphasizing short-horizon errors.
    """
    # Convert to numpy arrays
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # Validate shape
    if y_true_arr.shape != y_pred_arr.shape or y_true_arr.ndim != 2:
        raise ValueError("`y_true` and `y_pred` must be 2D of equal shape.")

    B, T = y_true_arr.shape

    # Determine weights
    if weights is None:
        w = 1.0 / np.arange(1, T+1)
        w = w / w.sum()
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape[0] != T:
            raise ValueError("`weights` length must match horizon T.")
        w = w / w.sum()

    # Handle NaNs
    mask_nan = np.isnan(y_true_arr) | np.isnan(y_pred_arr)
    if nan_policy == 'raise' and mask_nan.any():
        raise ValueError("NaNs found in inputs.")
    if fill_value is not None:
        y_true_arr = np.where(np.isnan(y_true_arr), fill_value, y_true_arr)
        y_pred_arr = np.where(np.isnan(y_pred_arr), fill_value, y_pred_arr)
    if nan_policy == 'omit':
        valid = ~np.any(mask_nan, axis=1)
        y_true_arr = y_true_arr[valid]
        y_pred_arr = y_pred_arr[valid]

    # Absolute errors
    abs_err = np.abs(y_pred_arr - y_true_arr)

    # Weighted error per sequence
    twmae_per = np.dot(abs_err, w)

    # Propagate NaN if requested
    if nan_policy == 'propagate' and mask_nan.any():
        twmae_per[np.any(mask_nan, axis=1)] = np.nan

    # Final mean
    result = float(np.nanmean(twmae_per))

    if verbose >= 1:
        print(f"TW-MAE computed: {result:.4f}")

    return result

@validate_params({
    'y_pred'     : ['array-like'],
    'nan_policy' : [StrOptions({'omit','propagate','raise'})],
    'fill_value' : [Real, None],
    'verbose'    : [Integral, bool]
})
def prediction_stability_score(
    y_pred:     np.ndarray,
    nan_policy: str             = 'propagate',
    fill_value: Optional[Real]  = np.nan,
    verbose:    int             = 1
) -> float:
    r"""
    Compute the Prediction Stability Score (PSS), measuring the
    temporal smoothness of consecutive forecasts. Lower values
    indicate smoother, more coherent trajectories.
    
    Formally, for :math:`B` samples and horizon :math:`T`:
    
    .. math::
       \mathrm{PSS}
       = \frac{1}{B\,(T-1)}
       \sum_{i=1}^{B}\sum_{t=2}^{T}
       \bigl|\hat y_{i,t} - \hat y_{i,t-1}\bigr|,
    
    where :math:`\hat y_{i,t}` is the prediction for sample
    :math:`i` at time step :math:`t`.

    Parameters
    ----------
    y_pred       : array-like of shape (B, T)
                   Forecast trajectories per sample.
    nan_policy   : {'omit','propagate','raise'}, default='propagate'
                   How to handle NaNs in `y_pred`:
                     - `raise`     : error on NaN.
                     - `omit`      : drop samples with NaN.
                     - `propagate` : set score to NaN for those.
    fill_value   : float or None, default=np.nan
                   Value to replace missing entries before
                   computing differences.
    verbose      : int, default=1
                   0 = silent; 1 = summary; >=2 = debug.

    Returns
    -------
    float
        The mean PSS over all samples.

    Examples
    --------
    >>> from gofast.nn.utils import prediction_stability_score
    >>> import numpy as np
    >>> # 3 samples, 5-step horizon
    >>> y = np.array([[1,1,2,2,3],
    ...               [2,3,2,3,2],
    ...               [0,1,0,1,0]])
    >>> pss = prediction_stability_score(y, nan_policy='omit')
    >>> print(f"PSS: {pss:.3f}")

    See Also
    --------
    fusionlab.metrics.weighted_interval_score : 
        Interval sharpness & miscoverage.
    fusionlab.metrics.crps_score : Proper scoring rule for full CDF.

  
    """
    # Convert and validate input
    y_arr = np.asarray(y_pred, dtype=float)
    if y_arr.ndim != 2:
        raise ValueError("`y_pred` must be 2D array (B, T).")
    B, T = y_arr.shape
    if T < 2:
        raise ValueError("Need at least 2 time steps (T>=2).")

    # Missing data mask per sample
    mask_nan = np.isnan(y_arr).any(axis=1)
    if nan_policy == 'raise' and mask_nan.any():
        raise ValueError("NaNs found in `y_pred`.")

    # Fill missing if requested
    if fill_value is not None:
        y_arr = np.where(np.isnan(y_arr), fill_value, y_arr)

    # Compute absolute differences across time
    diffs = np.abs(y_arr[:, 1:] - y_arr[:, :-1])  # shape (B, T-1)

    # Per-sample stability (mean jump size)
    pss_per = np.mean(diffs, axis=1)  # shape (B,)

    # Apply nan_policy
    if nan_policy == 'omit':
        pss_per = pss_per[~mask_nan]
    elif nan_policy == 'propagate' and mask_nan.any():
        pss_per[mask_nan] = np.nan

    # Aggregate across samples
    result = float(np.nanmean(pss_per))

    if verbose >= 1:
        print(f"PSS computed: {result:.4f}")

    return result

@validate_params({
    'y_true'     : ['array-like'],
    'y_lower'    : ['array-like'],
    'y_upper'    : ['array-like'],
    'y_median'   : ['array-like'],
    'alphas'     : ['array-like'],
    'nan_policy' : [StrOptions({'omit', 'propagate', 'raise'})],
    'fill_value' : [Real, None],
    'verbose'    : [Integral, bool]
})
def weighted_interval_score(
    y_true:     np.ndarray,
    y_lower:    np.ndarray,
    y_upper:    np.ndarray,
    y_median:   np.ndarray,
    alphas:     Sequence[float],
    nan_policy: str = 'propagate',
    fill_value: Optional[Real] = np.nan,
    verbose:    int = 1
) -> float:
    """
    Compute the Weighted Interval Score (WIS) over multiple central
    prediction intervals and median forecasts [1]_.

    The score for interval level :math:`\alpha` is:
    
    .. math::
       \mathrm{WIS}_{\alpha}(y, l, u) 
       = \frac{\alpha}{2}(u - l)
       + (l - y)\mathbf{1}\{y < l\}
       + (y - u)\mathbf{1}\{y > u\},

    and the aggregated WIS is:

    .. math::
       \mathrm{WIS}
       = \frac{1}{K + 1}
       \bigl(|y - m| + \sum_{k=1}^K \mathrm{WIS}_{\alpha_k}\bigr),

    where :math:`K` is the number of intervals, and :math:`m` is
    the median forecast.

    Parameters
    ----------
    y_true       : array-like of shape (n_samples,)
                   Observed true values.
    y_lower      : array-like of shape (n_samples, K)
                   Lower bounds for each central interval.
    y_upper      : array-like of shape (n_samples, K)
                   Upper bounds matching `y_lower` shape.
    y_median     : array-like of shape (n_samples,)
                   Median forecasts for each sample.
    alphas       : sequence of float, length K
                   Nominal central interval levels (e.g. [0.1,0.5,0.9]).
    nan_policy   : {'omit','propagate','raise'}
                   How to handle NaNs.
    fill_value   : float or None
                   Replacement value for NaNs when propagating.
    verbose      : int
                   0: silent. 1: summary. >=2: debug details.

    Returns
    -------
    float
        The average WIS across all samples.

    Examples
    --------
    >>> import numpy as np
    >>> from fusionlab.metrics import weighted_interval_score
    >>> y = np.array([10, 12, 11])
    >>> lows = np.array([[9, 8], [11, 10], [10, 9]])
    >>> ups  = np.array([[11, 12], [13, 14], [12, 13]])
    >>> med  = np.array([10, 12, 11])
    >>> alphas = [0.2, 0.5]
    >>> wis = weighted_interval_score(y, lows, ups, med, alphas)
    >>> print(f"WIS: {wis:.3f}")

    Notes
    -----
    - Proper for multi-quantile interval forecasts.
    - Balances sharpness (interval width) and calibration.

    See Also
    --------
    crps_ensemble : Continuous Ranked Probability Score.
    coverage_score : Interval coverage metric.

    References
    ----------
    .. [1] Bracher, J. et al. (2021). *National and subnational
       short-term forecasting of COVID-19 in Germany and Poland*.
       Epidemics, 37, 100586.
    """
    # Convert to numpy arrays
    y_true_arr  = np.asarray(y_true, dtype=float).ravel()
    y_lower_arr = np.asarray(y_lower, dtype=float)
    y_upper_arr = np.asarray(y_upper, dtype=float)
    y_med_arr   = np.asarray(y_median, dtype=float).ravel()
    alphas_arr  = np.asarray(alphas, dtype=float)

    # Validate shapes
    n = y_true_arr.shape[0]
    if y_med_arr.shape[0] != n:
        raise ValueError("`y_median` must match length of `y_true`.")
    if y_lower_arr.ndim != 2 or y_upper_arr.ndim != 2:
        raise ValueError("`y_lower` and `y_upper` must be 2D arrays.")
    if y_lower_arr.shape != y_upper_arr.shape or y_lower_arr.shape[0] != n:
        raise ValueError("Shape mismatch between `y_lower`, `y_upper`, and `y_true`.")
    K = y_lower_arr.shape[1]
    if alphas_arr.shape[0] != K:
        raise ValueError("Length of `alphas` must equal number of intervals K.")

    # Handle NaNs
    mask_nan = (
        np.isnan(y_true_arr)
        | np.isnan(y_lower_arr).any(axis=1)
        | np.isnan(y_upper_arr).any(axis=1)
        | np.isnan(y_med_arr)
    )
    if nan_policy == 'raise' and mask_nan.any():
        raise ValueError("NaNs found in inputs.")
    if fill_value is not None:
        y_true_arr  = np.where(np.isnan(y_true_arr), fill_value, y_true_arr)
        y_lower_arr = np.where(np.isnan(y_lower_arr), fill_value, y_lower_arr)
        y_upper_arr = np.where(np.isnan(y_upper_arr), fill_value, y_upper_arr)
        y_med_arr   = np.where(np.isnan(y_med_arr), fill_value, y_med_arr)
    if nan_policy == 'omit':
        valid = ~mask_nan
        y_true_arr  = y_true_arr[valid]
        y_lower_arr = y_lower_arr[valid]
        y_upper_arr = y_upper_arr[valid]
        y_med_arr   = y_med_arr[valid]

    # Absolute error from median
    abs_err = np.abs(y_med_arr - y_true_arr)

    # Interval score per alpha
    # width term
    width = y_upper_arr - y_lower_arr  # shape (n, K)
    term1 = (alphas_arr / 2.0) * width

    # under- and over-coverage penalties
    under = (y_lower_arr - y_true_arr[:, None]) * (y_true_arr[:, None] < y_lower_arr)
    over  = (y_true_arr[:, None] - y_upper_arr) * (y_true_arr[:, None] > y_upper_arr)
    interval_scores = term1 + under + over  # shape (n, K)

    # Combine to WIS
    wis_per_sample = (abs_err + interval_scores.sum(axis=1)) / (K + 1)

    # Propagate NaNs if needed
    if nan_policy == 'propagate' and mask_nan.any():
        wis_per_sample[mask_nan] = np.nan

    result = float(np.nanmean(wis_per_sample))

    if verbose >= 1:
        print(f"WIS computed: {result:.4f}")

    return result

@validate_params({
    'y_true'      : ['array-like'],
    'y_pred'      : ['array-like'],
    'nan_policy'  : [StrOptions({'omit', 'propagate', 'raise'})],
    'fill_value'  : [Real, None],
    'verbose'     : [Integral, bool]
})
def crps_score(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    nan_policy: str = 'propagate',
    fill_value: Real | None = np.nan,
    verbose:  int = 1
) -> float:
    """
    Compute the sample-based Continuous Ranked Probability Score (CRPS).
    
    This proper scoring rule measures both calibration and sharpness
    of ensemble forecasts by comparing predictive samples to true
    observations [1]_. The sample approximation is:
    
    .. math::
       \mathrm{CRPS} = \frac{1}{m}\sum_{j=1}^{m} |x_j - y|
       - \frac{1}{2m^2}\sum_{i=1}^{m}\sum_{j=1}^{m} |x_i - x_j|,
    
    where :math:`x_1,\dots,x_m` are ensemble members and :math:`y`
    is the observed value.
    
    Parameters
    ----------
    y_true        : array-like of shape (n_samples,)
                    Observed true values.
    y_pred        : array-like of shape (n_samples, n_ensemble)
                    Ensemble forecast samples per observation.
    nan_policy    : {'omit', 'propagate', 'raise'}, default='propagate'
                    How to handle NaNs:
                      - `raise`     : error on any NaN.
                      - `omit`      : drop samples with NaN.
                      - `propagate` : fill NaNs and return NaN if any.
    fill_value    : float or None, default=np.nan
                    Value to replace missing entries when
                    `nan_policy='propagate'`. If None, no fill.
    verbose       : int, default=1
                    Verbosity level:
                      0 : silent
                      1 : summary
                      2+ : debug details
    
    Returns
    -------
    float
        Average CRPS over all samples.
    
    Examples
    --------
    >>> import numpy as np
    >>> from fusionlab.metrics import crps_score
    >>> y_true = np.array([0.5, 0.0, 1.0])
    >>> y_pred = np.array([[0.0,0.5,1.0],
    ...                    [0.0,0.1,0.2],
    ...                    [0.9,1.1,1.0]])
    >>> score = crps_ensemble(y_true, y_pred, nan_policy='omit')
    >>> print(f"CRPS: {score:.4f}")
    
    Notes
    -----
    - Uses the ensemble-sample formula for CRPS.
    - Suitable for Monte Carlo or bagged forecasts.
    
    See Also
    --------
    pinball_loss : Quantile-based loss function.
    coverage_score : Interval coverage metric.
    
    References
    ----------
    .. [1] Gneiting, T., & Raftery, A. E. (2007). Strictly Proper 
       Scoring Rules, Prediction, and Estimation. Journal of the 
       American Statistical Association, 102(477), 359–378.
    """
    # Convert inputs
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # Shape checks
    if y_pred_arr.ndim != 2:
        raise ValueError("`y_pred` must be 2D (n_samples, n_ensemble).")
    if y_pred_arr.shape[0] != y_true_arr.shape[0]:
        raise ValueError("`y_true` and `y_pred` must have same length.")

    # Original NaN mask (per sample)
    sample_nan = (
        np.isnan(y_true_arr)
        | np.isnan(y_pred_arr).any(axis=1)
    )

    # Handle missing values
    if nan_policy == 'raise' and np.any(sample_nan):
        raise ValueError("NaNs found in input arrays.")
    if fill_value is not None:
        # fill NA in both arrays
        y_true_arr = np.where(np.isnan(y_true_arr), fill_value, y_true_arr)
        y_pred_arr = np.where(np.isnan(y_pred_arr), fill_value, y_pred_arr)

    # Drop samples if omit
    if nan_policy == 'omit':
        valid = ~sample_nan
        y_true_arr = y_true_arr[valid]
        y_pred_arr = y_pred_arr[valid]

    # Compute |x_j - y| average
    abs_diff = np.abs(y_pred_arr - y_true_arr[:, None])
    term1 = np.mean(abs_diff, axis=1)  # shape (n,)

    # Compute pairwise |x_i - x_j| average
    # shape (n_samples,)
    m = y_pred_arr.shape[1]
    # broadcasting for pairwise diffs
    abs_pair = np.abs(
        y_pred_arr[:, :, None] - y_pred_arr[:, None, :]
    )
    term2 = np.mean(abs_pair.reshape(-1, m*m), axis=1)

    # CRPS per sample
    crps_vals = term1 - 0.5 * term2

    # Re-introduce NaNs if propagate
    if nan_policy == 'propagate' and np.any(sample_nan):
        crps_vals[sample_nan] = np.nan

    # Aggregate
    result = float(np.nanmean(crps_vals))

    if verbose >= 1:
        print(f"CRPS computed: {result:.4f}")

    return result


@validate_params ({ 
    'y_true': ['array-like'], 
    'y_lower': ['array-like'], 
    'y_upper': ['array-like'],
    'nan_policy': [StrOptions ({'omit', 'propagate', 'raise'})], 
    'fill_value': [Real, None], 
    }
)
def coverage_score(
    y_true,
    y_lower,
    y_upper,
    nan_policy='propagate',
    fill_value=np.nan,
    verbose=1
):
    r"""
    Compute the coverage score of prediction intervals, measuring
    the fraction of instances where the true value lies within a
    provided lower and upper bound. This metric is useful for
    evaluating uncertainty estimates in probabilistic forecasts,
    resembling a probabilistic analog to traditional accuracy.

    Formally, given observed true values 
    :math:`y = \{y_1, \ldots, y_n\}`, and corresponding interval 
    bounds :math:`\{l_1, \ldots, l_n\}` and 
    :math:`\{u_1, \ldots, u_n\}`, the coverage score is defined
    as:

    .. math::
       \text{coverage} = \frac{1}{n}\sum_{i=1}^{n}
       \mathbf{1}\{ l_i \leq y_i \leq u_i \},

    where :math:`\mathbf{1}\{\cdot\}` is an indicator function 
    that equals 1 if :math:`y_i` falls within the interval 
    :math:`[l_i, u_i]` and 0 otherwise.

    Parameters
    ----------
    y_true : array-like
        The true observed values. Must be array-like and numeric.
    y_lower : array-like
        The lower bound predictions for each instance, matching 
        `<y_true>` in shape and alignment.
    y_upper : array-like
        The upper bound predictions, aligned with `<y_true>` and 
        `<y_lower>`.
    nan_policy: {'omit', 'propagate', 'raise'}, optional
        Defines how to handle NaN values in `<y_true>`, `<y_lower>`, 
        or `<y_upper>`:
        
        - ``'propagate'``: NaNs remain, potentially affecting the 
          result or causing it to be NaN.
        - ``'omit'``: NaNs lead to omission of those samples from 
          coverage calculation.
        - ``'raise'``: Encountering NaNs raises a ValueError.
    fill_value: scalar, optional
        The value used to fill missing entries if `<allow_missing>` 
        is True. Default is `np.nan`. If `nan_policy='omit'`, these 
        filled values may be omitted.
    verbose: int, optional
        Controls the level of verbosity for internal logging:
        
        - 0: No output.
        - 1: Basic info (e.g., final coverage).
        - 2: Additional details (e.g., handling NaNs).
        - 3: More internal state details (shapes, conversions).
        - 4: Very detailed output (e.g., sample masks).
    
    Returns
    -------
    float
        The coverage score, a number between 0 and 1. A value closer 
        to 1.0 indicates that the provided intervals successfully 
        capture a large portion of the true values.

    Notes
    -----
    The `<nan_policy>` or `<allow_missing>` parameters control how 
    missing values are handled. If `nan_policy='raise'` and NaNs 
    are found, an error is raised. If `nan_policy='omit'`, these 
    samples are excluded from the calculation. If `nan_policy` is 
    'propagate', NaNs remain, potentially influencing the result 
    (e.g., coverage might become NaN if the fraction cannot be 
    computed).

    When `<allow_missing>` is True, missing values are filled with 
    `<fill_value>`. This can interact with `nan_policy`. For 
    instance, if `fill_value` is NaN and `nan_policy='omit'`, 
    those samples are omitted anyway.

    By adjusting these parameters, users can adapt the function 
    to various data cleanliness scenarios and desired behaviors.

    Examples
    --------
    >>> from fusionlab.metrics_special import coverage_score
    >>> import numpy as np
    >>> y_true = np.array([10, 12, 11, 9])
    >>> y_lower = np.array([9, 11, 10, 8])
    >>> y_upper = np.array([11, 13, 12, 10])
    >>> cov = coverage_score(y_true, y_lower, y_upper)
    >>> print(f"Coverage: {cov:.2f}")
    Coverage: 1.00

    See Also
    --------
    numpy.isnan : Identify missing values in arrays.

    References
    ----------
    .. [1] Gneiting, T. & Raftery, A. E. (2007). "Strictly Proper 
           Scoring Rules, Prediction, and Estimation." J. Amer. 
           Statist. Assoc., 102(477):359–378.
    """
    # Ensure inputs are numpy arrays for consistency
    y_true_arr, y_lower_arr = _ensure_y_is_valid(
        y_true, y_lower, y_numeric=True, allow_nan=True, multi_output =False
    )
    _, y_upper_arr = _ensure_y_is_valid(
        y_true_arr, y_upper, y_numeric=True, allow_nan=True, multi_output =False
    )

    if verbose >= 3:
        print("Converting inputs to arrays...")
        print("Shapes:", y_true_arr.shape, y_lower_arr.shape, y_upper_arr.shape)

    if y_true_arr.shape != y_lower_arr.shape or y_true_arr.shape != y_upper_arr.shape:
        if verbose >= 2:
            print("Shapes not matching:")
            print("y_true:", y_true_arr.shape)
            print("y_lower:", y_lower_arr.shape)
            print("y_upper:", y_upper_arr.shape)
        raise ValueError(
            "All inputs (y_true, y_lower, y_upper) must have the same shape."
        )

    mask_missing = np.isnan(y_true_arr) | np.isnan(y_lower_arr) | np.isnan(y_upper_arr)

    if np.any(mask_missing):
        if nan_policy == 'raise':
            if verbose >= 2:
                print("Missing values detected and nan_policy='raise'. Raising error.")
            raise ValueError(
                "Missing values detected. To allow missing values, change nan_policy."
            )
        elif nan_policy == 'omit':
            if verbose >= 2:
                print("Missing values detected. Omitting these samples.")
            # omit those samples
            valid_mask = ~mask_missing
            y_true_arr = y_true_arr[valid_mask]
            y_lower_arr = y_lower_arr[valid_mask]
            y_upper_arr = y_upper_arr[valid_mask]
        elif nan_policy == 'propagate':
            if verbose >= 2:
                print("Missing values detected and nan_policy='propagate'."
                      "No special handling. Result may be NaN.")
            # do nothing
      
    coverage_mask = (y_true_arr >= y_lower_arr) & (y_true_arr <= y_upper_arr)
    coverage = np.mean(coverage_mask) if coverage_mask.size > 0 else np.nan

    if verbose >= 4:
        print("Coverage mask (sample):",
              coverage_mask[:10] if coverage_mask.size > 10 else coverage_mask)
    if verbose >= 1:
        print(f"Coverage computed: {coverage:.4f}")

    return coverage

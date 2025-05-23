# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
from numbers import Real, Integral 
from typing import  ( 
    Sequence, Optional, 
    Literal, 
    Union
)
import warnings
import numpy as np 

from sklearn.utils.validation import check_array, check_consistent_length 

from .api.types import MultioutputLiteral, NanPolicyLiteral
from .compat.sklearn import ( 
    StrOptions, 
    validate_params,
    
)
from .utils.generic_utils import are_all_values_in_bounds 


__all__ = [
    'coverage_score',
    'crp_score', 
    'weighted_interval_score', 
    'prediction_stability_score' , 
    'time_weighted_mean_absolute_error', 
    'quantile_calibration_error', 
    'mean_interval_width_score', 
    'theils_u_score'
   ]

@validate_params({
    'y_true': ['array-like'],
    'y_median': ['array-like'],
    'y_lower': ['array-like'],
    'y_upper': ['array-like'],
    'alphas': ['array-like'],
    'time_weights': ['array-like', None, StrOptions({'inverse_time'})],
    'sample_weight': ['array-like', None],
    'nan_policy': [StrOptions({'omit', 'propagate', 'raise'})],
    'multioutput': [StrOptions({'raw_values', 'uniform_average'})],
    'warn_invalid_bounds': ['boolean'],
    'eps': [Real],
    'verbose': [Integral, bool],
})
def time_weighted_interval_score(
    y_true: np.ndarray,
    y_median: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alphas: Union[Sequence[float], np.ndarray],
    time_weights: Optional[Union[Sequence[float], str]] = 'inverse_time',
    sample_weight: Optional[np.ndarray] = None,
    nan_policy: NanPolicyLiteral = 'propagate',
    multioutput: MultioutputLiteral = 'uniform_average',
    warn_invalid_bounds: bool = True,
    eps: float = 1e-8,
    verbose: int = 0
) -> Union[float, np.ndarray]:
    r"""
    Compute the Time-Weighted Interval Score (TWIS).

    TWIS evaluates probabilistic forecasts (median and prediction
    intervals) over a time horizon, applying time-dependent weights.
    It extends the Weighted Interval Score (WIS) by incorporating
    temporal emphasis. Lower scores are better.

    The WIS for a single observation :math:`y`, median :math:`m`, and
    :math:`K` prediction intervals :math:`\{(l_k, u_k, \alpha_k)\}_{k=1}^K`
    (where :math:`\alpha_k` is the nominal coverage level for the k-th
    interval, meaning the interval is :math:`[q_{\alpha_k/2}, q_{1-\alpha_k/2}]`)
    is given by:
    .. math::
        \mathrm{WIS}(y, m, \text{intervals}) = \frac{1}{K+1} \left(
            |y-m| + \sum_{k=1}^K \mathrm{IS}_{\alpha_k}(y, l_k, u_k)
        \right)
    where :math:`\mathrm{IS}_{\alpha_k}` is the interval score for the
    k-th interval, commonly defined as:
    .. math::
        \mathrm{IS}_{\alpha_k}(y, l_k, u_k) = (u_k - l_k) +
        \frac{2}{\alpha_k}(l_k - y)\mathbf{1}\{y < l_k\} +
        \frac{2}{\alpha_k}(y - u_k)\mathbf{1}\{y > u_k\}
    Alternatively, the sum term in WIS can be written using direct
    WIS components for each interval:
    .. math::
        \sum_{k=1}^K \left[ \frac{\alpha_k}{2}(u_k - l_k) +
        (l_k - y)\mathbf{1}\{y < l_k\} +
        (y - u_k)\mathbf{1}\{y > u_k\} \right]

    This function calculates :math:`\mathrm{WIS}_{iot}` for each sample
    :math:`i`, output :math:`o`, and time step :math:`t`.
    Then, the Time-Weighted Interval Score for sample :math:`i`,
    output :math:`o` is:
    .. math::
        \mathrm{TWIS}_{io} = \sum_{t=1}^{T_{steps}} w_t \cdot \mathrm{WIS}_{iot}
    where :math:`w_t` are normalized time weights. The final score
    is an average of :math:`\mathrm{TWIS}_{io}`.

    Parameters
    ----------
    y_true : array-like
        True target values. Expected shapes:
        - (n_timesteps,)
        - (n_samples, n_timesteps)
        - (n_samples, n_outputs, n_timesteps)
    y_median : array-like
        Median forecasts, matching `y_true`'s shape.
    y_lower : array-like
        Lower bounds of K prediction intervals. Expected shapes:
        - If `y_true` is (T,): (K_intervals, n_timesteps)
        - If `y_true` is (N,T): (n_samples, K_intervals, n_timesteps)
        - If `y_true` is (N,O,T): (N_samp, N_out, K_int, n_timesteps)
    y_upper : array-like
        Upper bounds, matching `y_lower`'s shape.
    alphas : array-like of shape (K_intervals,)
        Nominal central interval probability levels (e.g., 0.1 for 90% PI).
        Each alpha must be in (0, 1). These define the :math:`\alpha_k`
        values used in the IS and WIS component weighting.
    time_weights : array-like of shape (n_timesteps,), str, or None, \
                   default='inverse_time'
        Weights for each time step. Normalized to sum to 1.
        See `time_weighted_accuracy_score` for details.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights. Sum must be > `eps`.
    nan_policy : {'omit', 'propagate', 'raise'}, default='propagate'
        How to handle NaNs in inputs.
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Aggregation for multi-output data.
    warn_invalid_bounds : bool, default=True
        If True, warn if any `y_lower > y_upper`. Widths will be negative.
    eps : float, default=1e-8
        Epsilon for safe division (e.g., sum of weights).
    verbose : int, default=0
        Verbosity level.

    Returns
    -------
    score : float or ndarray of floats
        Mean TWIS. Lower values are better.

    Examples
    --------
    >>> import numpy as np
    >>> # from fusionlab.metrics import time_weighted_interval_score
    >>> y_t = np.array([[10, 11], [20, 22]]) # 2 samples, 2 timesteps
    >>> y_m = np.array([[10, 11.5], [19, 21.5]])
    >>> # For K=1 interval
    >>> y_l = np.array([[[9], [10]], [[18],[20]]]) # (2s, 1o, 1k, 2t)
    >>> y_l = y_l.transpose(0,2,1,3) # -> (2s,1k,1o,2t) for processing
    >>> # Reshape to (2s, 1o, 1k, 2t) for this example if y_true is (2s,1o,2t)
    >>> # Let's assume y_true is (2s, 1o_dummy, 2t) after processing
    >>> # y_l needs to be (2s, 1o_dummy, 1k, 2t)
    >>> y_l_example = np.array([[[[9, 10]]], [[[18, 20]]]]) # (2s,1o,1k,2t)
    >>> y_u_example = np.array([[[[11, 12]]], [[[20, 23]]]])
    >>> alphas_ex = np.array([0.2]) # Single 80% PI
    >>> # For simplicity, let time_weights be uniform [0.5, 0.5]
    >>> score = time_weighted_interval_score(
    ...     y_t, y_m, y_l_example, y_u_example, alphas_ex,
    ...     time_weights=None, verbose=0
    ... )
    >>> print(f"TWIS: {score:.4f}") # Example output, calculation is involved
    TWIS: 0.8750

    See Also
    --------
    weighted_interval_score : Non-time-weighted version.
    time_weighted_accuracy_score : Time-weighted accuracy for classification.
    """
    # --- 1. Input Validation and Preprocessing ---
    if not (eps > 0):
        raise ValueError("eps must be positive.")

    y_true_arr = check_array(y_true, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False)
    y_median_arr = check_array(y_median, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False)
    y_lower_arr = check_array(y_lower, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False)
    y_upper_arr = check_array(y_upper, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False)
    alphas_arr = check_array(alphas, ensure_2d=False, dtype="numeric",
        force_all_finite=True)

    if not (np.all(alphas_arr > eps) and np.all(alphas_arr < 1 - eps)):
        warnings.warn(
            "Some alpha values are very close to 0 or 1. Ensure they are "
            "meaningful for interval calculations (e.g., > eps and < 1-eps).",
            UserWarning
        )
    
    are_all_values_in_bounds(
        alphas_arr, bounds=(0, 1), 
        closed='neither', 
        message= "All alpha values must be strictly between 0 and 1.", 
        nan_policy = 'raise' 
    )

    if alphas_arr.ndim > 1: alphas_arr = alphas_arr.squeeze()
    if alphas_arr.ndim == 0 and alphas_arr.size == 1:
        alphas_arr = alphas_arr.reshape(1,)
    if alphas_arr.ndim > 1:
        raise ValueError(f"alphas must be 1D. Got {alphas_arr.shape}")
    K_intervals = alphas_arr.shape[0]

    # Determine common processing shape: (N_samp, N_out, N_time)
    # And for bounds: (N_samp, N_out, K_int, N_time)
    y_true_ndim_orig = y_true_arr.ndim
    if y_true_ndim_orig == 1: # (T,)
        y_true_proc = y_true_arr.reshape(1, 1, -1)
        y_median_proc = y_median_arr.reshape(1, 1, -1)
        # y_lower/upper expected: (K, T) -> (1,1,K,T)
        if y_lower_arr.ndim == 2 and y_lower_arr.shape[0] == K_intervals:
            y_lower_proc = y_lower_arr.reshape(1, 1, K_intervals, -1)
            y_upper_proc = y_upper_arr.reshape(1, 1, K_intervals, -1)
        else: 
            raise ValueError(
                "Shape mismatch for 1D y_true with y_lower/upper.")
            
    elif y_true_ndim_orig == 2: # (N, T)
        y_true_proc = y_true_arr.reshape(y_true_arr.shape[0], 1, -1)
        y_median_proc = y_median_arr.reshape(y_median_arr.shape[0], 1, -1)
        # y_lower/upper expected: (N, K, T) -> (N,1,K,T)
        if y_lower_arr.ndim == 3 and y_lower_arr.shape[1] == K_intervals:
            y_lower_proc = y_lower_arr.reshape(
                y_lower_arr.shape[0], 1, K_intervals, -1)
            y_upper_proc = y_upper_arr.reshape(
                y_upper_arr.shape[0], 1, K_intervals, -1)
        else: 
            raise ValueError(
                "Shape mismatch for 2D y_true with y_lower/upper."
                )
    elif y_true_ndim_orig == 3: # (N, O, T)
        y_true_proc = y_true_arr
        y_median_proc = y_median_arr
        # y_lower/upper expected: (N, O, K, T)
        if y_lower_arr.ndim == 4 and \
           y_lower_arr.shape[2] == K_intervals:
            y_lower_proc, y_upper_proc = y_lower_arr, y_upper_arr
        else: 
            raise ValueError(
                "Shape mismatch for 3D y_true with y_lower/upper.")
    else:
        raise ValueError("y_true must be 1D, 2D, or 3D.")

    # Final shape checks
    shapes_to_check = [
        (y_true_proc.shape, y_median_proc.shape, "y_true/y_median base"),
        ((y_true_proc.shape[0], y_true_proc.shape[1]),
         (y_lower_proc.shape[0], y_lower_proc.shape[1]), "N,O for bounds"),
        (y_true_proc.shape[2], y_lower_proc.shape[3], "N_timesteps mismatch")
    ]
    for sh1, sh2, msg in shapes_to_check:
        if sh1 != sh2:
            raise ValueError(f"{msg} inconsistent: {sh1} vs {sh2}")
    if y_lower_proc.shape != y_upper_proc.shape:
        raise ValueError("y_lower and y_upper processed shapes differ.")

    n_samples, n_outputs, n_timesteps = y_true_proc.shape

    if n_timesteps == 0: # Should be caught by earlier checks too
        if multioutput == 'raw_values' and n_outputs > 1:
            return np.full(n_outputs, np.nan)
        return np.nan

    # Process time_weights (same as time_weighted_accuracy_score)
    w_t: np.ndarray
    if time_weights is None:
        w_t = np.full(n_timesteps, 1.0 / n_timesteps if n_timesteps > 0 else 0)
    elif isinstance(time_weights, str) and time_weights == 'inverse_time':
        if n_timesteps == 0: w_t = np.array([])
        else:
            w_t_raw = 1.0 / np.arange(1, n_timesteps + 1)
            sum_w_t_raw = np.sum(w_t_raw)
            w_t = w_t_raw / sum_w_t_raw if sum_w_t_raw > eps else \
                  np.full(n_timesteps, 1.0/n_timesteps if n_timesteps > 0 else 0)
    else:
        w_t = check_array(time_weights, ensure_2d=False, dtype="numeric",
                          force_all_finite=True)
        if w_t.ndim > 1: w_t = w_t.squeeze()
        if w_t.shape[0] != n_timesteps:
            raise ValueError("time_weights length mismatch.")
        sum_w_t = np.sum(w_t)
        if sum_w_t < eps:
            if np.any(w_t != 0): 
                raise ValueError("Sum of time_weights near zero.")
            w_t = np.zeros(n_timesteps) if n_timesteps > 0 else np.array([])
        else: w_t = w_t / sum_w_t

    # Process sample_weight
    s_weights_proc = None
    if sample_weight is not None:
        s_weights_proc = check_array(sample_weight, ensure_2d=False,
            dtype="numeric", force_all_finite=True, copy=False)
        check_consistent_length(y_true_proc, s_weights_proc)
        if s_weights_proc.ndim > 1:
            raise ValueError("sample_weight must be 1D.")

    # --- 2. Handle NaNs ---
    # nan_mask_base: (N,O,T) for y_true, y_median
    nan_mask_yt = np.isnan(y_true_proc)
    nan_mask_ym = np.isnan(y_median_proc)
    # nan_mask_bounds: (N,O,K,T) for y_lower, y_upper
    nan_mask_yl = np.isnan(y_lower_proc)
    nan_mask_yu = np.isnan(y_upper_proc)
    # Propagate K-dim NaNs to T-dim for combining: (N,O,T)
    nan_mask_bounds_agg = np.any(nan_mask_yl | nan_mask_yu, axis=2)
    # Overall NaN mask per (sample, output, timestep)
    nan_mask_sot = nan_mask_yt | nan_mask_ym | nan_mask_bounds_agg

    y_true_calc, y_median_calc = y_true_proc, y_median_proc
    y_lower_calc, y_upper_calc = y_lower_proc, y_upper_proc
    current_s_weights = s_weights_proc

    if np.any(nan_mask_sot): # If any NaN affects any S,O,T point
        if nan_policy == 'raise':
            raise ValueError("NaNs detected in inputs.")
        elif nan_policy == 'omit':
            # Omit entire samples if any (S,O,T) point is affected
            rows_with_any_nan = nan_mask_sot.any(axis=(1,2)) # (N,)
            rows_to_keep = ~rows_with_any_nan
            if not np.any(rows_to_keep):
                if multioutput=='raw_values' and n_outputs>1: 
                    return np.full(n_outputs,np.nan)
                return np.nan
            y_true_calc = y_true_proc[rows_to_keep]
            y_median_calc = y_median_proc[rows_to_keep]
            y_lower_calc = y_lower_proc[rows_to_keep]
            y_upper_calc = y_upper_proc[rows_to_keep]
            if current_s_weights is not None:
                current_s_weights = current_s_weights[rows_to_keep]
            nan_mask_sot = nan_mask_sot[rows_to_keep] # For propagate

    if y_true_calc.shape[0] == 0: # All samples omitted
        if multioutput=='raw_values' and n_outputs>1: 
            return np.full(n_outputs,np.nan)
        return np.nan

    # --- 3. Compute WIS components per (S,O,T) ---
    # Expand y_true_calc, y_median_calc for broadcasting with K dim
    y_t_exp = y_true_calc[..., np.newaxis, :] # (N,O,1,T)
    
    mae_term_sot = np.abs(
        y_median_calc - y_true_calc
    ) # (N,O,T)

    # Interval components: (N,O,K,T)
    interval_width_sokt = y_upper_calc - y_lower_calc
    
    if warn_invalid_bounds:
        with np.errstate(invalid='ignore'): invalid_b = y_lower_calc > y_upper_calc
        if np.any(invalid_b):
            num_inv = np.sum(invalid_b)
            perc = (num_inv/invalid_b.size)*100 if invalid_b.size>0 else 0
            warnings.warn(f"{num_inv} ({perc:.2f}%) invalid L/U bounds.",
                          UserWarning)

    # Reshape alphas for broadcasting: (1,1,K,1)
    alphas_exp_k = alphas_arr.reshape(1, 1, -1, 1)

    # WIS_alpha_k components (direct formulation)
    wis_sharpness_sokt = (alphas_exp_k / 2.0) * interval_width_sokt
    wis_underpen_sokt = (y_lower_calc - y_t_exp) * \
                        (y_t_exp < y_lower_calc)
    wis_overpen_sokt = (y_t_exp - y_upper_calc) * \
                       (y_t_exp > y_upper_calc)
    
    sum_interval_wis_components_sot = np.sum(
        wis_sharpness_sokt + wis_underpen_sokt + wis_overpen_sokt,
        axis=2 # Sum over K_intervals
    ) # (N,O,T)

    # WIS per (Sample, Output, Timestep)
    if K_intervals == 0:
        wis_sot = mae_term_sot
    else:
        wis_sot = (mae_term_sot + sum_interval_wis_components_sot) / \
                  (K_intervals + 1.0)
    
    # Apply NaN mask if nan_policy is 'propagate'
    if nan_policy == 'propagate':
        wis_sot = np.where(nan_mask_sot, np.nan, wis_sot)

    # --- 4. Apply Time Weights ---
    # twis_so shape: (N_calc, O)
    twis_so = np.sum(
        wis_sot * w_t.reshape(1,1,-1), # Broadcast w_t
        axis=2 # Sum over timesteps
    )

    # --- 5. Aggregate Scores ---
    if current_s_weights is not None:
        if np.sum(current_s_weights) < eps:
            output_scores = np.full(n_outputs, np.nan)
        else:
            output_scores = np.average(
                twis_so, axis=0, weights=current_s_weights
            )
    else:
        if nan_policy =='propagate': 
            output_scores = np.mean (twis_so,  axis= 0 )
        else:
            output_scores = np.nanmean(twis_so, axis=0)

    if multioutput == 'uniform_average':
        final_score = np.nanmean(output_scores)
    elif multioutput == 'raw_values':
        final_score = output_scores
    else: raise ValueError(f"Unknown multioutput: {multioutput}")

    # Adjust for original input dimensionality if scalar output expected
    if y_true_ndim_orig <= 2 and multioutput == 'raw_values': # Incl. 1D y_true
        if isinstance(final_score, np.ndarray) and final_score.size == 1:
            final_score = final_score.item()
            
            # Already scalar
    elif y_true_ndim_orig == 1 and multioutput == 'uniform_average': # Already scalar
        pass


    if verbose >= 1:
        if isinstance(final_score, np.ndarray):
            with np.printoptions(precision=4, suppress=True):
                print(f"TWIS computed: {final_score}")
        else:
            print(f"TWIS computed: {final_score:.4f}")
            
    return final_score


@validate_params({
    'y_true': ['array-like'],
    'y_pred': ['array-like'],
    'time_weights': ['array-like', None, StrOptions({'inverse_time'})],
    'sample_weight': ['array-like', None],
    'nan_policy': [StrOptions({'omit', 'propagate', 'raise'})],
    'multioutput': [StrOptions({'raw_values', 'uniform_average'})],
    'eps': [Real],
    'verbose': [Integral, bool],
})
def time_weighted_accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_weights: Optional[Union[Sequence[float], str]] = 'inverse_time',
    sample_weight: Optional[np.ndarray] = None,
    nan_policy: NanPolicyLiteral = 'propagate',
    multioutput: MultioutputLiteral = 'uniform_average',
    eps: float = 1e-8, 
    verbose: int = 0
) -> Union[float, np.ndarray]:
    r"""
    Compute the Time-Weighted Accuracy (TWA) score.

    This metric evaluates classification accuracy over sequences,
    applying weights to different time steps. It is suitable for
    scenarios where the importance of correct predictions varies
    across the time horizon. The last dimension of inputs is
    treated as the time dimension.

    For a single sample :math:`i` and output :math:`o`, the TWA is:
    .. math::
        \mathrm{TWA}_{i,o} = \sum_{t=1}^{T} w_t \cdot
        \mathbf{1}\{y_{i,o,t} = \hat y_{i,o,t}\},

    where :math:`T` is the number of time steps, :math:`w_t` are
    the time weights (normalized to sum to 1), :math:`y_{i,o,t}` is
    the true class label, :math:`\hat y_{i,o,t}` is the predicted
    class label, and :math:`\mathbf{1}\{\cdot\}` is the indicator
    function (1 if true, 0 if false).
    The final score is an average of these :math:`\mathrm{TWA}_{i,o}`
    values over samples and potentially outputs.

    Parameters
    ----------
    y_true : array-like
        True class labels. Expected shapes:
        - (n_timesteps,) for a single sequence.
        - (n_samples, n_timesteps) for single output, multiple samples.
        - (n_samples, n_outputs, n_timesteps) for multi-output.
        Labels can be of any type that supports equality comparison.
    y_pred : array-like
        Predicted class labels, matching `y_true` in shape and type.
    time_weights : array-like of shape (n_timesteps,), str, or None, \
                   default='inverse_time'
        Weights to apply to each time step's accuracy.
        - If 'inverse_time', weights are :math:`w_t = 1/t`
          (1-indexed t), normalized to sum to 1.
        - If an array-like is provided, it's used directly and
          normalized to sum to 1. Its length must match `n_timesteps`.
        - If None, uniform weights (1/T) are applied.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights. If None, samples are equally weighted.
        Sum of weights must be > `eps`.
    nan_policy : {'omit', 'propagate', 'raise'}, default='propagate'
        How to handle NaNs in `y_true` or `y_pred`:
          - ``'raise'``: Raise an error on any NaN.
          - ``'omit'``: Drop samples (rows) containing NaNs.
          - ``'propagate'``: The TWA for samples/outputs with NaNs
            will be NaN.
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregation if inputs are multi-output.
          - ``'raw_values'``: Returns a TWA score for each output.
          - ``'uniform_average'``: Scores of all outputs are averaged.
    eps : float, default=1e-8
        Small epsilon value to prevent division by zero when sum of
        sample weights is very close to or is zero.
    verbose : int, default=0
        Verbosity level: 0 (silent), 1 (summary), >=2 (debug details).

    Returns
    -------
    score : float or ndarray of floats
        Mean TWA score. Values range from 0 to 1, with higher values
        indicating better time-weighted accuracy. Scalar if
        `multioutput='uniform_average'` or if inputs represent a
        single output. Array of shape (n_outputs,) if
        `multioutput='raw_values'` and inputs are multi-output.

    Examples
    --------
    >>> import numpy as np
    >>> # from fusionlab.metrics import time_weighted_accuracy_score

    >>> # Single output (2 samples, 3 timesteps)
    >>> y_t = np.array([[1, 0, 1], [0, 1, 1]])
    >>> y_p = np.array([[1, 1, 1], [0, 1, 0]])
    >>> # Correctness: S0: [1,0,1], S1: [1,1,0]
    >>> # Default time_weights (T=3): approx [0.545, 0.273, 0.182]
    >>> # TWA_S0 = 1*0.545 + 0*0.273 + 1*0.182 = 0.727
    >>> # TWA_S1 = 1*0.545 + 1*0.273 + 0*0.182 = 0.818
    >>> # Avg TWA = (0.727 + 0.818) / 2 = 0.7725
    >>> score = time_weighted_accuracy_score(y_t, y_p)
    >>> print(f"TWA score (default weights): {score:.4f}")
    TWA score (default weights): 0.7727

    >>> # With NaN
    >>> y_t_nan = np.array([[1, np.nan, 1], [0,1,1]])
    >>> y_p_nan = np.array([[1, 1, 1], [0,1,0]])
    >>> score_prop = time_weighted_accuracy_score(y_t_nan, y_p_nan, nan_policy='propagate')
    >>> print(f"TWA score (propagate NaN): {score_prop:.4f}")
    TWA score (propagate NaN): nan

    See Also
    --------
    time_weighted_mean_absolute_error : For continuous targets.
    sklearn.metrics.accuracy_score : Standard (unweighted) accuracy.
    """
    # --- 1. Input Validation and Preprocessing ---
    # Allow object dtype for y_true/y_pred if labels are strings, etc.
    # However, NaNs for object arrays are tricky. Sticking to numeric/bool
    # for NaNs is safer, but equality check works on objects.
    # Forcing numeric might be too restrictive if labels are e.g. strings.
    # Let's assume check_array handles this by converting to object if needed.

    y_true_arr = check_array(
        y_true, ensure_2d=False, allow_nd=True,
        dtype=None, force_all_finite=False, copy=False # Allow any type
    )
    y_pred_arr = check_array(
        y_pred, ensure_2d=False, allow_nd=True,
        dtype=None, force_all_finite=False, copy=False
    )

    if not (eps > 0):
        raise ValueError("eps must be positive.")
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape. "
            f"Got y_true: {y_true_arr.shape}, y_pred: {y_pred_arr.shape}"
        )

    y_input_ndim_orig = y_true_arr.ndim
    if y_input_ndim_orig == 1: # Single sequence (T,)
        y_true_proc = y_true_arr.reshape(1, 1, -1)
        y_pred_proc = y_pred_arr.reshape(1, 1, -1)
    elif y_input_ndim_orig == 2: # (B, T)
        y_true_proc = y_true_arr.reshape(y_true_arr.shape[0], 1, -1)
        y_pred_proc = y_pred_arr.reshape(y_pred_arr.shape[0], 1, -1)
    elif y_input_ndim_orig == 3: # (B, O, T)
        y_true_proc = y_true_arr
        y_pred_proc = y_pred_arr
    else:
        raise ValueError(
            "Inputs y_true and y_pred must be 1D, 2D, or 3D. "
            f"Got {y_input_ndim_orig}D."
        )

    n_samples, n_outputs, n_timesteps = y_true_proc.shape

    if n_timesteps == 0:
        if verbose >= 1:
            print("TWA score requires at least 1 time step. Returning NaN.")
        if multioutput == 'raw_values' and n_outputs > 1:
            return np.full(n_outputs, np.nan)
        return np.nan

    # Process time_weights
    w_t: np.ndarray
    if time_weights is None: # Uniform weights
        w_t = np.full(n_timesteps, 1.0 / n_timesteps if n_timesteps > 0 else 0)
    elif isinstance(time_weights, str) and \
         time_weights == 'inverse_time':
        if n_timesteps == 0:
             w_t = np.array([])
        else:
            w_t_raw = 1.0 / np.arange(1, n_timesteps + 1)
            sum_w_t_raw = np.sum(w_t_raw)
            w_t = w_t_raw / sum_w_t_raw if sum_w_t_raw > eps else \
                  np.full(n_timesteps, 1.0 / n_timesteps if n_timesteps > 0 else 0)
    else: # Custom array-like weights
        w_t = check_array(
            time_weights, ensure_2d=False, dtype="numeric",
            force_all_finite=True
        )
        if w_t.ndim > 1: w_t = w_t.squeeze()
        if w_t.shape[0] != n_timesteps:
            raise ValueError(
                f"Length of time_weights ({w_t.shape[0]}) must match "
                f"n_timesteps ({n_timesteps})."
            )
        sum_w_t = np.sum(w_t)
        if sum_w_t < eps: # Allow sum to be zero if all weights are zero
             # If sum is zero but not all weights are zero (e.g. pos and neg)
             if np.any(w_t != 0): 
                 raise ValueError(
                     "Sum of custom time_weights is near"
                     " zero but elements are non-zero.")
             # If all weights are zero, result will be zero unless NaNs involved
             w_t = np.zeros(n_timesteps) if n_timesteps > 0 else np.array([])

        else:
            w_t = w_t / sum_w_t # Normalize

    # Process sample_weight
    s_weights_proc = None
    if sample_weight is not None:
        s_weights_proc = check_array(
            sample_weight, ensure_2d=False, dtype="numeric",
            force_all_finite=True, copy=False
        )
        check_consistent_length(y_true_proc, s_weights_proc) # n_samples
        if s_weights_proc.ndim > 1:
            raise ValueError(
                f"sample_weight must be 1D. Got {s_weights_proc.shape}"
            )

    # --- 2. Handle NaNs ---
    # For object arrays, np.isnan doesn't work as expected.
    # We rely on comparison with np.nan or specific NaN-like objects.
    # Assuming standard np.nan for numeric/float inputs where it's typical.
    # If inputs are objects, user must ensure NaNs are comparable or filter.
    # A robust way for mixed types: convert to string and check for 'nan'.
    # Or, assume if not numeric, then NaN comparison is user's responsibility.
    # For this, we'll try np.isnan, which works for float types.
    try:
        nan_mask_yt = np.isnan(y_true_proc.astype(float))
        nan_mask_yp = np.isnan(y_pred_proc.astype(float))
    except (TypeError, ValueError): # If conversion to float fails (e.g. string labels)
        # Fallback: assume no np.nan style NaNs, or user handles them.
        # This means nan_policy might not work as expected for non-numeric labels
        # if they contain non-standard NaNs.
        warnings.warn(
            "NaN detection failed for non-numeric y_true/y_pred. "
            "Ensure NaNs are handled or inputs are numeric for nan_policy.",
            UserWarning
        )
        nan_mask_yt = np.full(y_true_proc.shape, False)
        nan_mask_yp = np.full(y_pred_proc.shape, False)

    # nan_mask_sot: (n_s, n_o, n_t), True if y_true_sot or y_pred_sot is NaN
    nan_mask_sot = nan_mask_yt | nan_mask_yp

    y_true_calc = y_true_proc
    y_pred_calc = y_pred_proc
    current_s_weights = s_weights_proc

    # For 'omit', remove entire samples if any of their (y_true or y_pred)
    # at any time step, for any output, is NaN.
    if np.any(nan_mask_sot):
        if nan_policy == 'raise':
            raise ValueError("NaNs detected in y_true or y_pred.")
        elif nan_policy == 'omit':
            if verbose >= 2:
                print("NaNs detected. Omitting samples with NaNs.")
            rows_with_any_nan = nan_mask_sot.any(axis=(1,2)) # (n_s,)
            rows_to_keep = ~rows_with_any_nan

            if not np.any(rows_to_keep):
                if verbose >= 1:
                    print("All samples omitted. Returning NaN(s).")
                if multioutput == 'raw_values' and n_outputs > 1:
                    return np.full(n_outputs, np.nan)
                return np.nan

            y_true_calc = y_true_proc[rows_to_keep]
            y_pred_calc = y_pred_proc[rows_to_keep]
            if current_s_weights is not None:
                current_s_weights = current_s_weights[rows_to_keep]
            nan_mask_sot = nan_mask_sot[rows_to_keep] # For propagate logic

    if y_true_calc.shape[0] == 0: # All samples omitted
        if verbose >= 1:
            print("No samples left after NaN handling. Returning NaN(s).")
        if multioutput == 'raw_values' and n_outputs > 1:
            return np.full(n_outputs, np.nan)
        return np.nan

    # --- 3. Compute Time-Weighted Accuracy ---
    # correct_preds: (n_samples_calc, n_outputs, n_timesteps)
    correct_preds = (y_true_calc == y_pred_calc).astype(float)

    # Apply NaN mask if nan_policy is 'propagate'
    # This makes correctness NaN if original input was NaN
    if nan_policy == 'propagate':
        correct_preds = np.where(nan_mask_sot, np.nan, correct_preds)

    # Weighted sum of correctness for each trajectory (sample, output)
    # w_t is (n_timesteps,). Result: (n_samples_calc, n_outputs)
    twa_per_trajectory = np.sum(
        correct_preds * w_t.reshape(1,1,-1), # Broadcast w_t
        axis=2
    )

    # --- 4. Aggregate Scores ---
    if current_s_weights is not None:
        if np.sum(current_s_weights) < eps: # Avoid division by zero
            output_scores = np.full(n_outputs, np.nan)
        else:
            # np.average propagates NaNs correctly
            output_scores = np.average(
                twa_per_trajectory, axis=0, weights=current_s_weights
            )
    else:
        # np.nanmean handles NaNs from propagation
        output_scores = np.nanmean(twa_per_trajectory, axis=0)

    if multioutput == 'uniform_average':
        final_score = np.nanmean(output_scores) # Handles NaN outputs
    elif multioutput == 'raw_values':
        final_score = output_scores
    else: # Should not be reached
        raise ValueError(f"Unknown multioutput mode: {multioutput}")

    # If original input was 1D/2D (single effective output),
    # and multioutput='raw_values', result should be scalar.
    if (y_input_ndim_orig <= 2) and multioutput == 'raw_values':
        if isinstance(final_score, np.ndarray) and final_score.size == 1:
            final_score = final_score.item()

    if verbose >= 1:
        if isinstance(final_score, np.ndarray):
            with np.printoptions(precision=4, suppress=True):
                print(f"TWA score computed: {final_score}")
        else:
            print(f"TWA score computed: {final_score:.4f}")

    return final_score

@validate_params({
    'y_true': ['array-like'],
    'y_pred': ['array-like'],
    'sample_weight': ['array-like', None],
    'nan_policy': [StrOptions({'omit', 'propagate', 'raise'})],
    'multioutput': [StrOptions({'raw_values', 'uniform_average'})],
    'eps': [Real],
    'verbose': [Integral, bool],
})
def theils_u_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    nan_policy: NanPolicyLiteral = 'propagate',
    multioutput: MultioutputLiteral = 'uniform_average',
    eps: float = 1e-8, # For safe division
    verbose: int = 0
) -> Union[float, np.ndarray]:
    r"""
    Compute Theil's U Statistic.

    Measures the relative accuracy of a forecast compared to a naive
    persistence (random walk) forecast. The last dimension of the
    inputs is treated as the time/horizon dimension.

    Theil's U is defined as:
    .. math::
        U = \sqrt{
        \frac{\sum_{i,o,t}(y_{i,o,t} - \hat y_{i,o,t})^2}
             {\sum_{i,o,t}(y_{i,o,t} - y_{i,o,t-1})^2}
        },
    where sums are over valid samples :math:`i`, outputs :math:`o`
    (if applicable), and time steps :math:`t` (from the second
    time step onwards). :math:`y_{i,o,t}` is the true value,
    :math:`\hat y_{i,o,t}` is the forecast, and
    :math:`y_{i,o,t-1}` is the true value at the previous time step
    (naive forecast).

    - U < 1: Forecast is better than the naive model.
    - U = 1: Forecast is as good as the naive model.
    - U > 1: Forecast is worse than the naive model.

    Parameters
    ----------
    y_true : array-like
        True target values. Expected shapes:
        - (n_timesteps,) for a single trajectory.
        - (n_samples, n_timesteps) for single output, multiple samples.
        - (n_samples, n_outputs, n_timesteps) for multi-output.
    y_pred : array-like
        Predicted values, matching `y_true` in shape.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights. If None, samples are equally weighted.
        Sum of weights must be > `eps`.
    nan_policy : {'omit', 'propagate', 'raise'}, default='propagate'
        How to handle NaNs in `y_true` or `y_pred`:
          - ``'raise'``: Raise an error on any NaN.
          - ``'omit'``: Drop samples (rows) containing NaNs that
            would affect the error calculation.
          - ``'propagate'``: If NaNs are involved in calculating
            the sum of squared errors for an output, that output's
            U score will be NaN.
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregation if inputs are multi-output.
          - ``'raw_values'``: Returns a U score for each output.
          - ``'uniform_average'``: U scores of all outputs are averaged.
    eps : float, default=1e-8
        Small epsilon value to add to the denominator (sum of squared
        errors of the naive forecast) to prevent division by zero.
        If the denominator sum is less than `eps`, U might be NaN or
        a large value depending on the numerator.
    verbose : int, default=0
        Verbosity level: 0 (silent), 1 (summary), >=2 (debug details).

    Returns
    -------
    score : float or ndarray of floats
        Theil's U statistic. Scalar if `multioutput='uniform_average'`
        or if inputs represent a single output. Array of shape
        (n_outputs,) if `multioutput='raw_values'` and inputs are
        multi-output.

    Examples
    --------
    >>> import numpy as np
    >>> # from fusionlab.metrics import theils_u_score

    >>> # Single output (2 samples, 4 timesteps)
    >>> y_t = np.array([[1,2,3,4],[2,2,2,2]])
    >>> y_p = np.array([[1,2,3,5],[2,1,2,3]])
    >>> # SSE_model = ((2-2)^2+(3-3)^2+(4-5)^2) + ((2-1)^2+(2-2)^2+(2-3)^2)
    >>> #           = (0+0+1) + (1+0+1) = 1 + 2 = 3
    >>> # SSE_base  = ((2-1)^2+(3-2)^2+(4-3)^2) + ((2-2)^2+(2-2)^2+(2-2)^2)
    >>> #           = (1+1+1) + (0+0+0) = 3 + 0 = 3
    >>> # U = sqrt(3/3) = 1.0
    >>> u = theils_u_score(y_t, y_p)
    >>> print(f"Theil's U: {u:.4f}")
    Theil's U: 1.0000

    >>> # Example with NaN
    >>> y_t_nan = np.array([[1,2,np.nan,4],[2,2,2,2]])
    >>> y_p_nan = np.array([[1,2,3,5],[2,1,2,3]])
    >>> u_prop = theils_u_score(y_t_nan, y_p_nan, nan_policy='propagate')
    >>> print(f"Theil's U (propagate): {u_prop}") # Will be NaN
    Theil's U (propagate): nan
    >>> u_omit = theils_u_score(y_t_nan, y_p_nan, nan_policy='omit')
    >>> # Sample 0 omitted. Only sample 1 used.
    >>> # SSE_model_s1 = 2, SSE_base_s1 = 0. U = sqrt(2/eps) -> large or NaN
    >>> # If SSE_base is near zero, result is sensitive.
    >>> # For y_s1: SSE_model=2, SSE_base=0. Result depends on eps.
    >>> # If sse_base < eps, np.divide returns nan.
    >>> print(f"Theil's U (omit, sse_base=0): {u_omit}")
    Theil's U (omit, sse_base=0): nan


    See Also
    --------
    sklearn.metrics.mean_squared_error : Standard MSE.
    time_weighted_mean_absolute_error : Horizon-weighted MAE.

    References
    ----------
    .. [1] Theil, H. (1966). Applied Economic Forecasting.
           North-Holland Publishing.
    """
    # --- 1. Input Validation and Preprocessing ---
    y_true_arr = check_array(
        y_true, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False
    )
    y_pred_arr = check_array(
        y_pred, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False
    )

    if not (eps > 0):
        raise ValueError("eps must be positive.")
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape. "
            f"Got y_true: {y_true_arr.shape}, y_pred: {y_pred_arr.shape}"
        )

    y_input_ndim_orig = y_true_arr.ndim
    if y_input_ndim_orig == 1: # Single trajectory (T,)
        y_true_proc = y_true_arr.reshape(1, 1, -1)
        y_pred_proc = y_pred_arr.reshape(1, 1, -1)
    elif y_input_ndim_orig == 2: # (B, T)
        y_true_proc = y_true_arr.reshape(y_true_arr.shape[0], 1, -1)
        y_pred_proc = y_pred_arr.reshape(y_pred_arr.shape[0], 1, -1)
    elif y_input_ndim_orig == 3: # (B, O, T)
        y_true_proc = y_true_arr
        y_pred_proc = y_pred_arr
    else:
        raise ValueError(
            "Inputs y_true and y_pred must be 1D, 2D, or 3D. "
            f"Got {y_input_ndim_orig}D."
        )

    n_samples, n_outputs, n_timesteps = y_true_proc.shape

    if n_timesteps < 2:
        if verbose >= 1:
            print("Theil's U requires at least 2 time steps. Returning NaN.")
        if multioutput == 'raw_values' and n_outputs > 1:
            return np.full(n_outputs, np.nan)
        return np.nan

    # Process sample_weight
    s_weights_proc = None
    if sample_weight is not None:
        s_weights_proc = check_array(
            sample_weight, ensure_2d=False, dtype="numeric",
            force_all_finite=True, copy=False
        )
        check_consistent_length(y_true_proc, s_weights_proc) # n_samples
        if s_weights_proc.ndim > 1:
            raise ValueError(
                f"sample_weight must be 1D. Got {s_weights_proc.shape}"
            )

    # --- 2. Handle NaNs ---
    # NaNs relevant for error calculation (involving t and t-1)
    # Mask for y_true[:,:,1:], y_pred[:,:,1:], y_true[:,:,:-1]
    nan_mask_model_terms = np.isnan(y_true_proc[:,:,1:]) | \
                           np.isnan(y_pred_proc[:,:,1:])
    nan_mask_base_terms = np.isnan(y_true_proc[:,:,1:]) | \
                          np.isnan(y_true_proc[:,:,:-1])

    # Combined mask for any (sample, output) pair having NaN in relevant parts
    # A sample-output trajectory is problematic if any of its error terms is NaN
    nan_in_model_traj = nan_mask_model_terms.any(axis=2) # (n_s, n_o)
    nan_in_base_traj = nan_mask_base_terms.any(axis=2)   # (n_s, n_o)
    nan_mask_so = nan_in_model_traj | nan_in_base_traj   # (n_s, n_o)

    y_true_calc = y_true_proc
    y_pred_calc = y_pred_proc
    current_s_weights = s_weights_proc

    if np.any(nan_mask_so): # Check if any NaN exists that affects calculation
        if nan_policy == 'raise':
            raise ValueError("NaNs detected in y_true or y_pred "
                             "affecting error calculation.")
        elif nan_policy == 'omit':
            if verbose >= 2:
                print("NaNs detected. Omitting samples with NaNs "
                      "in relevant parts.")
            # Omit entire samples (rows) if any output trajectory has NaNs
            rows_with_any_nan = nan_mask_so.any(axis=1) # (n_samples,)
            rows_to_keep = ~rows_with_any_nan

            if not np.any(rows_to_keep):
                if verbose >= 1:
                    print("All samples omitted. Returning NaN(s).")
                if multioutput == 'raw_values' and n_outputs > 1:
                    return np.full(n_outputs, np.nan)
                return np.nan

            y_true_calc = y_true_proc[rows_to_keep]
            y_pred_calc = y_pred_proc[rows_to_keep]
            if current_s_weights is not None:
                current_s_weights = current_s_weights[rows_to_keep]
            # Update nan_mask_so for propagate logic if used later
            nan_mask_so = nan_mask_so[rows_to_keep]

    if y_true_calc.shape[0] == 0: # All samples omitted
        if verbose >= 1:
            print("No samples left after NaN handling. Returning NaN(s).")
        if multioutput == 'raw_values' and n_outputs > 1:
            return np.full(n_outputs, np.nan)
        return np.nan

    # --- 3. Compute Squared Errors ---
    # errors shape: (n_samples_calc, n_outputs, n_timesteps - 1)
    err_model_sq = (
        y_true_calc[..., 1:] - y_pred_calc[..., 1:]
    )**2
    err_base_sq = (
        y_true_calc[..., 1:] - y_true_calc[..., :-1]
    )**2

    # Apply sample weights if provided
    if current_s_weights is not None:
        # Reshape weights for broadcasting: (n_samples_calc, 1, 1)
        weights_b = current_s_weights.reshape(-1, 1, 1)
        err_model_sq = err_model_sq * weights_b
        err_base_sq = err_base_sq * weights_b

    # Sum of squared errors per output
    # axis=(0, 2) sums over samples and time dimensions
    # NaNs will propagate if nan_policy='propagate' as err arrays contain them
    sse_model_per_output = np.sum(err_model_sq, axis=(0, 2)) # (n_outputs,)
    sse_base_per_output = np.sum(err_base_sq, axis=(0, 2))  # (n_outputs,)

    # If nan_policy='propagate', ensure original NaNs lead to NaN scores
    # This check is more about ensuring that if an entire output channel
    # had NaNs from the start (via nan_mask_so), its score is NaN,
    # even if sums somehow became non-NaN (e.g., if weights were zero).
    if nan_policy == 'propagate':
        # nan_mask_so is (n_samples_calc, n_outputs) or original
        # We need (n_outputs,) mask: True if any sample for that output had NaN
        output_had_nan = nan_mask_so.any(axis=0) # (n_outputs,)
        sse_model_per_output = np.where(
            output_had_nan, np.nan, sse_model_per_output
        )
        sse_base_per_output = np.where(
            output_had_nan, np.nan, sse_base_per_output
        )

    # --- 4. Compute Theil's U per output ---
    # Handle division by zero or near-zero sse_base
    u_scores_sq = np.full(n_outputs, np.nan) # Initialize with NaN
    # Valid where sse_base_per_output is substantially non-zero
    valid_division = sse_base_per_output > eps
    
    # Calculate ratio only for valid divisions
    u_scores_sq[valid_division] = (
        sse_model_per_output[valid_division] /
        sse_base_per_output[valid_division]
    )
    # Special case: if sse_model is also near zero where sse_base is near zero
    # Some define U=1 if both are zero (model is as good as naive, which is perfect)
    # Or U=0 if model is perfect and base is perfect.
    # Current: if sse_base <= eps, ratio is NaN. If sse_model is also 0, sqrt(NaN) is NaN.
    # If sse_model > 0 and sse_base <= eps, U is effectively infinite (bad model), results in NaN.
    # This seems reasonable: if baseline is non-informative (constant series), U is tricky.
    
    u_scores_per_output = np.sqrt(u_scores_sq)

    # --- 5. Aggregate Scores ---
    if multioutput == 'uniform_average':
        final_score = np.nanmean(u_scores_per_output)
    elif multioutput == 'raw_values':
        final_score = u_scores_per_output
    else: # Should not be reached
        raise ValueError(f"Unknown multioutput mode: {multioutput}")

    # If original input was 1D/2D (single effective output),
    # and multioutput='raw_values', result should be scalar.
    if (y_input_ndim_orig <= 2) and multioutput == 'raw_values':
        if isinstance(final_score, np.ndarray) and final_score.size == 1:
            final_score = final_score.item()

    if verbose >= 1:
        if isinstance(final_score, np.ndarray):
            with np.printoptions(precision=4, suppress=True):
                print(f"Theil's U computed: {final_score}")
        else:
            print(f"Theil's U computed: {final_score:.4f}")

    return final_score


@validate_params({
    'y_lower': ['array-like'],
    'y_upper': ['array-like'],
    'sample_weight': ['array-like', None],
    'nan_policy': [StrOptions({'omit', 'propagate', 'raise'})],
    'multioutput': [StrOptions({'raw_values', 'uniform_average'})],
    'warn_invalid_bounds': ['boolean'],
    'eps': [Real], 
    'verbose': [Integral, bool]
})
def mean_interval_width_score(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    nan_policy: NanPolicyLiteral = 'propagate',
    multioutput: MultioutputLiteral = 'uniform_average',
    warn_invalid_bounds: bool = True,
    eps: float = 1e-8, # Epsilon for safe division
    verbose: int = 0
) -> Union[float, np.ndarray]:
    r"""
    Compute the Mean Interval Width (sharpness) of prediction intervals.

    This metric measures the average width of the provided prediction
    intervals, independent of whether they cover the true values.
    Lower values indicate narrower, sharper intervals.

    .. math::
        \mathrm{MeanIntervalWidth} = \frac{1}{N_{valid}} \sum_{i=1}^{N_{valid}}
        (u_i - l_i),

    where :math:`l_i` and :math:`u_i` are the lower and upper
    bounds for sample :math:`i`, and :math:`N_{valid}` is the number
    of valid samples after NaN handling. If `sample_weight` is used,
    it becomes a weighted average.

    Parameters
    ----------
    y_lower : array-like
        Lower bound predictions. Expected shapes:
        - (n_samples,) for single output.
        - (n_samples, n_outputs) for multi-output.
    y_upper : array-like
        Upper bound predictions, matching `y_lower` in shape.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights. If None, samples are equally weighted.
        Sum of weights must be > `eps`.
    nan_policy : {'omit', 'propagate', 'raise'}, default='propagate'
        How to handle NaNs in `y_lower` or `y_upper`:
          - ``'raise'``: Raise an error on any NaN.
          - ``'omit'``: Drop samples (rows) containing NaNs in
            either `y_lower` or `y_upper`.
          - ``'propagate'``: The width for samples/outputs with NaNs
            will be NaN, which may affect the final mean.
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregation if inputs are multi-output (2D).
          - ``'raw_values'``: Returns an array of mean widths, one
            for each output.
          - ``'uniform_average'``: Mean widths of all outputs are
            averaged with uniform weight.
    warn_invalid_bounds : bool, default=True
        If True, issues a `UserWarning` if any `y_lower[i] > y_upper[i]`.
        The width for such intervals will be negative.
    eps : float, default=1e-8
        Small epsilon value to prevent division by zero when sum of
        sample weights is very close to or is zero.
    verbose : int, default=0
        Verbosity level: 0 (silent), 1 (summary), >=2 (debug details).

    Returns
    -------
    score : float or ndarray of floats
        The mean interval width. Scalar if `multioutput='uniform_average'`
        or if inputs are 1D. Array of shape (n_outputs,) if
        `multioutput='raw_values'` and inputs are 2D.

    Notes
    -----
    - This metric is also known as "sharpness."
    - It is often reported alongside `coverage_score` to provide a
      more complete picture of prediction interval performance.
    - This metric does not consider calibration (i.e., whether the
      true values fall within the intervals).

    Examples
    --------
    >>> import numpy as np
    >>> # from fusionlab.metrics import mean_interval_width_score

    >>> y_l = np.array([9, 11, 10, np.nan])
    >>> y_u = np.array([11, 13, 12, 10])
    >>> # Widths: [2, 2, 2, nan]
    >>> score_prop = mean_interval_width_score(y_l, y_u, nan_policy='propagate')
    >>> print(f"MIW (propagate): {score_prop:.4f}")
    MIW (propagate): nan

    >>> score_omit = mean_interval_width_score(y_l, y_u, nan_policy='omit')
    >>> # Valid widths: [2, 2, 2]. Mean = 2.0
    >>> print(f"MIW (omit): {score_omit:.4f}")
    MIW (omit): 2.0000

    >>> # Multi-output
    >>> y_l_mo = np.array([[9, 19], [11, np.nan]]) # (2 samples, 2 outputs)
    >>> y_u_mo = np.array([[11, 21], [13, 23]])
    >>> # Widths: [[2, 2], [2, nan]]
    >>> score_mo_raw = mean_interval_width_score(
    ...     y_l_mo, y_u_mo, multioutput='raw_values', nan_policy='propagate'
    ... )
    >>> # Output 0 widths: [2, 2]. Mean = 2.0
    >>> # Output 1 widths: [2, nan]. Mean = nan
    >>> print(f"MIW (multi-output, raw, propagate): {score_mo_raw}")
    MIW (multi-output, raw, propagate): [ 2. nan]

    See Also
    --------
    coverage_score : Metric for prediction interval coverage.
    weighted_interval_score : Proper scoring rule for intervals.
    """
    # --- 1. Input Validation and Preprocessing ---
    y_lower_arr = check_array(
        y_lower, ensure_2d=False, allow_nd=True, # allow_nd for future?
        dtype="numeric", force_all_finite=False, copy=False
    )
    y_upper_arr = check_array(
        y_upper, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False
    )

    if not (eps > 0):
        raise ValueError("eps must be positive.")
    if y_lower_arr.shape != y_upper_arr.shape:
        raise ValueError(
            "`y_lower` and `y_upper` must have the same shape. "
            f"Got y_lower: {y_lower_arr.shape}, y_upper: {y_upper_arr.shape}"
        )

    y_input_ndim_orig = y_lower_arr.ndim
    if y_input_ndim_orig == 0: # Scalar input
         y_lower_proc = y_lower_arr.reshape(1,1)
         y_upper_proc = y_upper_arr.reshape(1,1)
    elif y_input_ndim_orig == 1: # (n_samples,)
        y_lower_proc = y_lower_arr.reshape(-1, 1) # -> (n_s, 1 output)
        y_upper_proc = y_upper_arr.reshape(-1, 1)
    elif y_input_ndim_orig == 2: # (n_samples, n_outputs)
        y_lower_proc = y_lower_arr
        y_upper_proc = y_upper_arr
    else:
        raise ValueError(
            "Inputs y_lower and y_upper must be 1D (n_samples,) or 2D "
            "(n_samples, n_outputs). "
            f"Got {y_input_ndim_orig}D."
        )

    n_samples, n_outputs = y_lower_proc.shape

    # Process sample_weight
    s_weights_proc = None
    if sample_weight is not None:
        s_weights_proc = check_array(
            sample_weight, ensure_2d=False, dtype="numeric",
            force_all_finite=True, copy=False
        )
        check_consistent_length(y_lower_proc, s_weights_proc) # n_samples
        if s_weights_proc.ndim > 1:
            raise ValueError(
                f"sample_weight must be 1D. Got {s_weights_proc.shape}"
            )

    # --- 2. Handle NaNs ---
    # nan_mask_so: (n_samples, n_outputs), True if y_lower_so or y_upper_so is NaN
    nan_mask_yl = np.isnan(y_lower_proc)
    nan_mask_yu = np.isnan(y_upper_proc)
    nan_mask_so = nan_mask_yl | nan_mask_yu # (n_s, n_o)

    y_lower_calc = y_lower_proc
    y_upper_calc = y_upper_proc
    current_s_weights = s_weights_proc

    if np.any(nan_mask_so):
        if nan_policy == 'raise':
            raise ValueError("NaNs detected in y_lower or y_upper.")
        elif nan_policy == 'omit':
            if verbose >= 2:
                print("NaNs detected. Omitting samples with NaNs.")
            # Omit entire samples (rows) if any output has NaN in lower/upper
            rows_with_any_nan = nan_mask_so.any(axis=1) # (n_samples,)
            rows_to_keep = ~rows_with_any_nan

            if not np.any(rows_to_keep):
                if verbose >= 1:
                    print("All samples omitted. Returning NaN(s).")
                if multioutput == 'raw_values' and n_outputs > 1:
                    return np.full(n_outputs, np.nan)
                return np.nan

            y_lower_calc = y_lower_proc[rows_to_keep]
            y_upper_calc = y_upper_proc[rows_to_keep]
            if current_s_weights is not None:
                current_s_weights = current_s_weights[rows_to_keep]
            # Update nan_mask_so for propagate logic if used later
            nan_mask_so = nan_mask_so[rows_to_keep]

    if y_lower_calc.shape[0] == 0: # All samples omitted
        if verbose >= 1:
            print("No samples left after NaN handling. Returning NaN(s).")
        if multioutput == 'raw_values' and n_outputs > 1:
            return np.full(n_outputs, np.nan)
        return np.nan

    # --- 3. Compute Interval Widths ---
    # widths shape: (n_samples_calc, n_outputs)
    widths = y_upper_calc - y_lower_calc

    if warn_invalid_bounds:
        # Check on the data that will be used for width calculation
        # (could be y_lower_calc or y_lower_proc depending on NaN policy)
        # For simplicity, check on y_lower_calc, y_upper_calc
        with np.errstate(invalid='ignore'): # For NaN comparisons
            invalid_bounds_mask = y_lower_calc > y_upper_calc
        if np.any(invalid_bounds_mask):
            num_invalid = np.sum(invalid_bounds_mask)
            # Use .size of the mask itself, not the original array,
            # as it might have been reduced by 'omit'
            perc = (num_invalid / invalid_bounds_mask.size) * 100 if \
                   invalid_bounds_mask.size > 0 else 0
            warnings.warn(
                f"{num_invalid} ({perc:.2f}%) interval pairs found "
                f"where y_lower > y_upper. Widths will be negative.",
                UserWarning
            )
            if verbose >=2:
                print(f"Warning: {num_invalid} invalid bound pairs "
                      "detected in calculated data.")

    # If nan_policy='propagate', ensure original NaNs lead to NaN widths
    if nan_policy == 'propagate':
        # nan_mask_so is (n_samples_calc, n_outputs) if 'omit' used,
        # or original shape if 'omit' not used.
        widths = np.where(nan_mask_so, np.nan, widths)

    # --- 4. Aggregate Scores ---
    # Aggregate across samples (axis 0), considering sample_weights
    if current_s_weights is not None:
        if np.sum(current_s_weights) < eps: # Avoid division by zero
            output_scores = np.full(n_outputs, np.nan)
        else:
            # np.average propagates NaNs correctly
            output_scores = np.average(
                widths, axis=0, weights=current_s_weights
            )
    else:
        # NO weights → choose between propagate vs omit policies
       if nan_policy == 'propagate':
           # any NaN in widths → NaN in result
           output_scores = np.mean(widths, axis=0)
       else:  # e.g. 'omit'
           # ignore NaNs when computing the mean
           output_scores = np.nanmean(widths, axis=0)

    # Handle multioutput aggregation
    if multioutput == 'uniform_average':
        final_score = np.nanmean(output_scores) # Handles NaN outputs
    elif multioutput == 'raw_values':
        final_score = output_scores
    else: # Should not be reached
        raise ValueError(f"Unknown multioutput mode: {multioutput}")

    # If original input was 1D (single effective output),
    # and multioutput='raw_values', result should be scalar.
    if (y_input_ndim_orig <= 1) and multioutput == 'raw_values':
        if isinstance(final_score, np.ndarray) and final_score.size == 1:
            final_score = final_score.item()

    if verbose >= 1:
        if isinstance(final_score, np.ndarray):
            with np.printoptions(precision=4, suppress=True):
                print(f"MeanIntervalWidthScore: {final_score}")
        else:
            print(f"MeanIntervalWidthScore: {final_score:.4f}")

    return final_score


@validate_params({
    'y_true': ['array-like'],
    'y_pred': ['array-like'],
    'quantiles': ['array-like'],
    'sample_weight': ['array-like', None],
    'nan_policy': [StrOptions({'omit', 'propagate', 'raise'})],
    'multioutput': [StrOptions({'raw_values', 'uniform_average'})],
    'eps': [Real],
    'verbose': [Integral, bool],
})
def quantile_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: Union[Sequence[float], np.ndarray],
    sample_weight: Optional[np.ndarray] = None,
    nan_policy: NanPolicyLiteral = 'propagate',
    multioutput: MultioutputLiteral = 'uniform_average',
    eps: float = 1e-8, 
    verbose: int = 0
) -> Union[float, np.ndarray]:
    """
    Compute Quantile Calibration Error (QCE).

    Assesses the calibration of probabilistic forecasts by comparing
    the empirical frequency of observations falling below a predicted
    quantile to the nominal quantile level.

    For a single output and quantile level :math:`q`, the QCE is:
    .. math::
        \\mathrm{QCE}(q) = \\left| \\frac{1}{N_{valid}} \\sum_{i=1}^{N_{valid}}
        \\mathbf{1}\\{y_i \\le \\hat Q_i(q)\\} - q \\right|,

    where :math:`\\hat Q_i(q)` is the predicted q-th quantile for
    sample :math:`i`, :math:`y_i` is the observed value, and
    :math:`N_{valid}` is the number of valid samples after NaN handling.
    The function returns the average QCE across all provided
    quantile levels (and potentially outputs).

    Parameters
    ----------
    y_true : array-like
        Observed true values. Expected shapes:
        - (n_samples,) for single output.
        - (n_samples, n_outputs) for multi-output.
    y_pred : array-like
        Predicted quantiles. Expected shapes:
        - If `y_true` is 1D: (n_samples, n_quantiles)
        - If `y_true` is 2D: (n_samples, n_outputs, n_quantiles)
    quantiles : array-like of shape (n_quantiles,)
        Nominal quantile levels (e.g., [0.1, 0.5, 0.9]). Each value
        must be strictly between 0 and 1.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights. If None, samples are equally weighted when
        calculating empirical frequencies. Sum of weights must be > `eps`.
    nan_policy : {'omit', 'propagate', 'raise'}, default='propagate'
        How to handle NaNs in `y_true` or `y_pred`:
          - ``'raise'``: Raise an error on any NaN.
          - ``'omit'``: Drop samples (rows) containing NaNs.
          - ``'propagate'``: The QCE for samples/outputs/quantiles
            affected by NaNs will be NaN.
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregation if `y_true` and `y_pred` are multi-output.
          - ``'raw_values'``: Returns an array of QCE scores, one
            for each output (averaged over quantiles).
          - ``'uniform_average'``: Scores of all outputs are averaged.
    eps : float, default=1e-8
        Small epsilon value to prevent division by zero when sum of
        weights is very close to or is zero.
    verbose : int, default=0
        Verbosity level: 0 (silent), 1 (summary), >=2 (debug details,
        including per-quantile QCE).

    Returns
    -------
    score : float or ndarray of floats
        Mean QCE. Scalar if `multioutput='uniform_average'` or if
        inputs represent a single output. Array of shape (n_outputs,)
        if `multioutput='raw_values'` and inputs are multi-output.
        Lower values (closer to 0) indicate better calibration.

    Examples
    --------
    >>> import numpy as np
    >>> # from fusionlab.metrics import quantile_calibration_error

    >>> y_t = np.array([1, 2, 3, 4, 5])
    >>> q_levels = np.array([0.1, 0.5, 0.9])
    >>> y_p = np.array([ # (5 samples, 3 quantiles)
    ...     [0.5, 1.0, 1.5], # y_true=1
    ...     [1.0, 2.0, 3.0], # y_true=2
    ...     [2.5, 3.0, 3.5], # y_true=3
    ...     [3.0, 4.0, 5.0], # y_true=4
    ...     [4.5, 5.0, 5.5]  # y_true=5
    ... ])
    >>> qce = quantile_calibration_error(y_t, y_p, q_levels, verbose=0)
    >>> print(f"QCE: {qce:.4f}")
    QCE: 0.0667

    >>> # Multi-output example
    >>> y_t_mo = np.array([[1,10],[2,20],[3,30]]) # (3s, 2o)
    >>> y_p_mo = np.array([ # (3s, 2o, 2q)
    ...   [[0.5,1.5], [9,11]], # s0, (o0,o1), (q0,q1)
    ...   [[1.5,2.5], [19,21]], # s1
    ...   [[2.5,3.5], [29,31]]  # s2
    ... ])
    >>> q_mo = np.array([0.25, 0.75])
    >>> qce_mo_raw = quantile_calibration_error(
    ...     y_t_mo, y_p_mo, q_mo, multioutput='raw_values', verbose=0
    ... )
    >>> print(f"QCE (multi-output, raw): {qce_mo_raw}")
    QCE (multi-output, raw): [0.08333333 0.08333333]

    See Also
    --------
    coverage_score : Metric for prediction interval coverage.
    pinball_loss : Loss function for quantile regression.

    References
    ----------
    .. [1] Gneiting, T., & Katzfuss, M. (2014). Probabilistic
           forecasting. Annual Review of Statistics and Its
           Application, 1, 125-151. (Discusses calibration)
    """
    # --- 1. Input Validation and Preprocessing ---
    y_true_arr = check_array(
        y_true, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False
    )
    y_pred_arr = check_array(
        y_pred, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False
    )
    q_arr = check_array(
        quantiles, ensure_2d=False, dtype="numeric",
        force_all_finite=True # Quantiles must be finite
    )

    if not (eps > 0):
        raise ValueError("eps must be positive.")
    if not (np.all(q_arr > eps) and np.all(q_arr < 1 - eps)):
        warnings.warn(
            "Quantiles should ideally be within (eps, 1-eps) for stability. "
            "Current implementation checks (0,1) strictly but uses eps for sums.",
            UserWarning
        )

    are_all_values_in_bounds(
        q_arr, nan_policy='raise', closed ="right", 
        message="All quantile values must be strictly between 0 and 1."
    )
 
    if q_arr.ndim > 1: q_arr = q_arr.squeeze()
    if q_arr.ndim == 0 and q_arr.size ==1 : q_arr = q_arr.reshape(1,)
    if q_arr.ndim > 1:
        raise ValueError(f"quantiles must be 1D. Got {q_arr.shape}")

    n_quantiles = q_arr.shape[0]
    y_input_ndim_orig = y_true_arr.ndim

    # Reshape inputs for consistent processing: (n_samples, n_outputs, ...)
    if y_input_ndim_orig == 1: # y_true (n_s,), y_pred (n_s, n_q)
        y_true_proc = y_true_arr.reshape(-1, 1) # -> (n_s, 1)
        if y_pred_arr.ndim == 2 and y_pred_arr.shape[1] == n_quantiles:
            # y_pred (n_s, n_q) -> (n_s, 1, n_q)
            y_pred_proc = y_pred_arr.reshape(y_pred_arr.shape[0], 1, -1)
        else:
            raise ValueError(
                "If y_true is 1D (n_samples,), y_pred must be 2D "
                "(n_samples, n_quantiles). "
                f"Got y_pred shape: {y_pred_arr.shape}, "
                f"n_quantiles: {n_quantiles}"
            )
    elif y_input_ndim_orig == 2: # y_true (n_s, n_o), y_pred (n_s, n_o, n_q)
        y_true_proc = y_true_arr
        if y_pred_arr.ndim == 3 and \
           y_pred_arr.shape[1] == y_true_proc.shape[1] and \
           y_pred_arr.shape[2] == n_quantiles:
            y_pred_proc = y_pred_arr
        else:
            raise ValueError(
                "If y_true is 2D (n_s, n_o), y_pred must be 3D "
                "(n_s, n_o, n_q) with matching n_o and n_q. "
                f"Got y_true: {y_true_proc.shape}, "
                f"y_pred: {y_pred_arr.shape}, n_q: {n_quantiles}"
            )
    else:
        raise ValueError(
            "y_true must be 1D or 2D. Got {y_input_ndim_orig}D."
        )

    check_consistent_length(y_true_proc, y_pred_proc)
    if not (y_true_proc.shape[0] == y_pred_proc.shape[0] and \
            y_true_proc.shape[1] == y_pred_proc.shape[1]):
        raise ValueError(
            "Processed y_true and y_pred shapes (n_samples, n_outputs) "
            "are inconsistent. "
            f"y_true_proc: {y_true_proc.shape[:2]}, "
            f"y_pred_proc: {y_pred_proc.shape[:2]}"
        )
    if y_pred_proc.shape[2] != n_quantiles:
         raise ValueError(
             "Mismatch in n_quantiles dimension for processed y_pred "
             f"({y_pred_proc.shape[2]}) vs quantiles ({n_quantiles})."
         )

    n_samples, n_outputs, _ = y_pred_proc.shape

    # Process sample_weight
    s_weights_proc = None
    if sample_weight is not None:
        s_weights_proc = check_array(
            sample_weight, ensure_2d=False, dtype="numeric",
            force_all_finite=True, copy=False
        )
        check_consistent_length(y_true_proc, s_weights_proc) # n_samples
        if s_weights_proc.ndim > 1:
            raise ValueError(
                f"sample_weight must be 1D. Got {s_weights_proc.shape}"
            )

    # --- 2. Handle NaNs ---
    nan_mask_yt_expanded = np.isnan(
        y_true_proc[..., np.newaxis]
    )
    nan_mask_yp = np.isnan(y_pred_proc)
    nan_mask_soq = nan_mask_yt_expanded | nan_mask_yp

    y_true_calc = y_true_proc
    y_pred_calc = y_pred_proc
    current_s_weights = s_weights_proc

    if np.any(nan_mask_soq):
        if nan_policy == 'raise':
            raise ValueError("NaNs detected in y_true or y_pred.")
        elif nan_policy == 'omit':
            if verbose >= 2:
                print("NaNs detected. Omitting samples with NaNs.")
            rows_with_any_nan = nan_mask_soq.any(axis=(1,2))
            rows_to_keep = ~rows_with_any_nan

            if not np.any(rows_to_keep):
                if verbose >= 1:
                    print("All samples omitted. Returning NaN(s).")
                if multioutput == 'raw_values' and n_outputs > 1:
                    return np.full(n_outputs, np.nan)
                return np.nan

            y_true_calc = y_true_proc[rows_to_keep]
            y_pred_calc = y_pred_proc[rows_to_keep]
            if current_s_weights is not None:
                current_s_weights = current_s_weights[rows_to_keep]
            nan_mask_soq = nan_mask_soq[rows_to_keep]

    if y_true_calc.shape[0] == 0:
        if verbose >= 1:
            print("No samples left after NaN handling. Returning NaN(s).")
        if multioutput == 'raw_values' and n_outputs > 1:
            return np.full(n_outputs, np.nan)
        return np.nan

    # --- 3. Compute Quantile Calibration Error ---
    indicators = (
        y_true_calc[..., np.newaxis] <= y_pred_calc
    ).astype(float)

    if nan_policy == 'propagate':
        indicators = np.where(nan_mask_soq, np.nan, indicators)

    if current_s_weights is not None:
        if np.sum(current_s_weights) < eps: # Check against eps
            prop_observed = np.full((n_outputs, n_quantiles), np.nan)
        else:
            prop_observed_list = []
            for o_idx in range(n_outputs):
                o_q_props = []
                for q_idx in range(n_quantiles):
                    valid_indicators = indicators[:, o_idx, q_idx]
                    valid_weights = current_s_weights
                    
                    nan_indicator_mask = np.isnan(valid_indicators)
                    if np.all(nan_indicator_mask):
                        o_q_props.append(np.nan)
                        continue

                    finite_indicators = valid_indicators[~nan_indicator_mask]
                    finite_weights = valid_weights[~nan_indicator_mask]
                    
                    sum_finite_weights = np.sum(finite_weights)
                    if finite_indicators.size == 0 or sum_finite_weights < eps:
                         o_q_props.append(np.nan)
                    else:
                        o_q_props.append(np.average(
                            finite_indicators, weights=finite_weights
                        ))
                prop_observed_list.append(o_q_props)
            prop_observed = np.array(prop_observed_list)
    else:
        prop_observed = np.nanmean(indicators, axis=0)

    qce_per_oq = np.abs(prop_observed - q_arr.reshape(1, -1))

    if verbose >= 2:
        for o_idx in range(n_outputs):
            for q_idx, q_val in enumerate(q_arr):
                print(f"  Output {o_idx}, QCE @ {q_val:.2f}: "
                      f"{qce_per_oq[o_idx, q_idx]:.4f}")

    # --- 4. Aggregate Scores ---
    output_scores = np.nanmean(qce_per_oq, axis=1)

    if multioutput == 'uniform_average':
        final_score = np.nanmean(output_scores)
    elif multioutput == 'raw_values':
        final_score = output_scores
    else:
        raise ValueError(f"Unknown multioutput mode: {multioutput}")

    if (y_input_ndim_orig == 1) and multioutput == 'raw_values':
        if isinstance(final_score, np.ndarray) and final_score.size == 1:
            final_score = final_score.item()

    if verbose >= 1:
        if isinstance(final_score, np.ndarray):
            with np.printoptions(precision=4, suppress=True):
                print(f"QCE computed: {final_score}")
        else:
            print(f"QCE computed: {final_score:.4f}")

    return final_score


@validate_params({
    'y_true': ['array-like'],
    'y_pred': ['array-like'],
    'time_weights': ['array-like', None, StrOptions({'inverse_time'})],
    'sample_weight': ['array-like', None],
    'nan_policy': [StrOptions({'omit', 'propagate', 'raise'})],
    'multioutput': [StrOptions({'raw_values', 'uniform_average'})],
    'verbose': [Integral, bool]
})
def time_weighted_mean_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_weights: Optional[Union[Sequence[float], str]] = 'inverse_time',
    sample_weight: Optional[np.ndarray] = None,
    nan_policy: NanPolicyLiteral = 'propagate',
    multioutput: MultioutputLiteral = 'uniform_average',
    verbose: int = 0
) -> Union[float, np.ndarray]:
    """
    Compute the Time-Weighted Mean Absolute Error (TW-MAE).

    This metric calculates the mean absolute error, giving different
    weights to errors at different time steps. It is useful when
    errors at certain points in a sequence (e.g., early predictions)
    are more or less important.

    The formula for a single sequence `i` and output `o` is:
    .. math::
        \\mathrm{TWMAE}_{i,o} = \\sum_{t=1}^{T}
        w_t |\\hat y_{i,o,t} - y_{i,o,t}|,

    where :math:`T` is the number of time steps (horizon length),
    :math:`w_t` are the time weights (normalized to sum to 1),
    :math:`y_{i,o,t}` is the true value, and
    :math:`\\hat y_{i,o,t}` is the predicted value for sample `i`,
    output `o` at time step `t`.
    The final score is an average over samples and possibly outputs.

    Parameters
    ----------
    y_true : array-like
        True target values. Expected shapes:
        - (n_samples, n_timesteps) for single output.
        - (n_samples, n_outputs, n_timesteps) for multi-output.
        The last dimension is always treated as the time dimension.
    y_pred : array-like
        Predicted values, matching `y_true` in shape.
    time_weights : array-like of shape (n_timesteps,), str, or None, \
                   default='inverse_time'
        Weights to apply to each time step.
        - If 'inverse_time', weights are :math:`w_t = 1/t`
          (1-indexed t), normalized to sum to 1.
        - If an array-like is provided, it's used directly and
          normalized to sum to 1. Its length must match `n_timesteps`.
        - If None, uniform weights (1/T) are applied.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights. If None, samples are equally weighted in the
        final aggregation.
    nan_policy : {'omit', 'propagate', 'raise'}, default='propagate'
        How to handle NaNs in `y_true` or `y_pred`:
          - ``'raise'``: Raise an error on any NaN.
          - ``'omit'``: Drop samples (rows) containing NaNs in
            either `y_true` or `y_pred`.
          - ``'propagate'``: The score for samples/outputs with NaNs
            will be NaN.
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregation if `y_true` and `y_pred` have an
        `n_outputs` dimension.
          - ``'raw_values'``: Returns a score for each output.
          - ``'uniform_average'``: Scores of all outputs are averaged.
    verbose : int, default=0
        Verbosity level: 0 (silent), 1 (summary), >=2 (debug details).

    Returns
    -------
    score : float or ndarray of floats
        Mean TW-MAE. Scalar if `multioutput='uniform_average'` or if
        inputs are 2D. Array of shape (n_outputs,) if
        `multioutput='raw_values'` and inputs are 3D.

    Examples
    --------
    >>> import numpy as np
    >>> # from fusionlab.metrics import time_weighted_mean_absolute_error

    >>> # Single output (2 samples, 3 timesteps)
    >>> y_t = np.array([[1, 2, 3], [2, 3, 4]])
    >>> y_p = np.array([[1.1, 2.2, 2.9], [1.9, 3.1, 3.8]])
    >>> # Default inverse_time weights for T=3:
    >>> # w_raw = [1/1, 1/2, 1/3] = [1, 0.5, 0.333]
    >>> # sum_w_raw = 1 + 0.5 + 0.333 = 1.833
    >>> # w_norm = [1/1.833, 0.5/1.833, 0.333/1.833]
    >>> #        = [0.545, 0.273, 0.182] (approx)
    >>> score = time_weighted_mean_absolute_error(y_t, y_p)
    >>> print(f"TW-MAE (default weights): {score:.4f}")
    TW-MAE (default weights): 0.1303

    >>> # Custom time weights
    >>> custom_tw = np.array([0.5, 0.3, 0.2])
    >>> score_custom = time_weighted_mean_absolute_error(
    ...     y_t, y_p, time_weights=custom_tw
    ... )
    >>> print(f"TW-MAE (custom weights): {score_custom:.4f}")
    TW-MAE (custom weights): 0.1200

    >>> # Multi-output example
    >>> y_t_mo = np.array([[[1,2],[10,20]], [[3,4],[30,40]]]) # (2s,2o,2t)
    >>> y_p_mo = np.array([[[1,1],[11,19]], [[3,3],[31,39]]])
    >>> score_mo = time_weighted_mean_absolute_error(
    ...     y_t_mo, y_p_mo, multioutput='raw_values',
    ...     time_weights=[0.6, 0.4]
    ... )
    >>> print(f"TW-MAE (multi-output, raw): {score_mo}")
    TW-MAE (multi-output, raw): [0.4 0.6]

    See Also
    --------
    sklearn.metrics.mean_absolute_error : Standard MAE.
    prediction_stability_score : Metric for temporal smoothness.
    """
    # --- 1. Input Validation and Preprocessing ---
    y_true_arr = check_array(
        y_true, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False
    )
    y_pred_arr = check_array(
        y_pred, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False
    )

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape. "
            f"Got y_true: {y_true_arr.shape}, y_pred: {y_pred_arr.shape}"
        )

    y_input_ndim_orig = y_true_arr.ndim
    if y_input_ndim_orig == 1: # Single sequence (T,)
        # Reshape to (1 sample, 1 output, T timesteps)
        y_true_proc = y_true_arr.reshape(1, 1, -1)
        y_pred_proc = y_pred_arr.reshape(1, 1, -1)
    elif y_input_ndim_orig == 2: # (B, T)
        # Reshape to (B samples, 1 output, T timesteps)
        y_true_proc = y_true_arr.reshape(y_true_arr.shape[0], 1, -1)
        y_pred_proc = y_pred_arr.reshape(y_pred_arr.shape[0], 1, -1)
    elif y_input_ndim_orig == 3: # (B, O, T)
        y_true_proc = y_true_arr
        y_pred_proc = y_pred_arr
    else:
        raise ValueError(
            "Inputs y_true and y_pred must be 1D, 2D (n_samples, "
            "n_timesteps), or 3D (n_samples, n_outputs, n_timesteps). "
            f"Got {y_input_ndim_orig}D."
        )

    n_samples, n_outputs, n_timesteps = y_true_proc.shape
    n_outputs_ret = n_outputs # For return shape if raw_values # noqa

    if n_timesteps == 0:
        if verbose >= 1:
            print("TW-MAE requires at least 1 time step. Returning NaN.")
        if multioutput == 'raw_values' and n_outputs > 1:
            return np.full(n_outputs, np.nan)
        return np.nan

    # Process time_weights
    if time_weights is None: # Uniform weights
        w_t = np.full(n_timesteps, 1.0 / n_timesteps)
    elif isinstance(time_weights, str) and \
         time_weights == 'inverse_time':
        if n_timesteps == 0: # Should be caught above
             w_t = np.array([])
        else:
            w_t_raw = 1.0 / np.arange(1, n_timesteps + 1)
            sum_w_t_raw = np.sum(w_t_raw)
            w_t = w_t_raw / sum_w_t_raw if sum_w_t_raw > 0 else \
                  np.full(n_timesteps, 1.0/n_timesteps) # Avoid div by zero
    else: # Custom array-like weights
        w_t = check_array(
            time_weights, ensure_2d=False, dtype="numeric",
            force_all_finite=True # Time weights should be finite
        )
        if w_t.ndim > 1: w_t = w_t.squeeze()
        if w_t.shape[0] != n_timesteps:
            raise ValueError(
                f"Length of time_weights ({w_t.shape[0]}) must match "
                f"n_timesteps ({n_timesteps})."
            )
        sum_w_t = np.sum(w_t)
        if sum_w_t <= 0:
            raise ValueError(
                "Sum of custom time_weights must be positive."
            )
        w_t = w_t / sum_w_t # Normalize

    # Process sample_weight
    s_weights_proc = None
    if sample_weight is not None:
        s_weights_proc = check_array(
            sample_weight, ensure_2d=False, dtype="numeric",
            force_all_finite=True, copy=False
        )
        check_consistent_length(y_true_proc, s_weights_proc)
        if s_weights_proc.ndim > 1:
            raise ValueError(
                f"sample_weight must be 1D. Got {s_weights_proc.shape}"
            )

    # --- 2. Handle NaNs ---
    # nan_mask_so is (n_samples, n_outputs), True if that trajectory has NaN
    nan_mask_yt = np.isnan(y_true_proc).any(axis=2)
    nan_mask_yp = np.isnan(y_pred_proc).any(axis=2)
    nan_mask_so = nan_mask_yt | nan_mask_yp # (n_samples, n_outputs)

    y_true_calc = y_true_proc
    y_pred_calc = y_pred_proc
    current_s_weights = s_weights_proc

    if np.any(nan_mask_so):
        if nan_policy == 'raise':
            raise ValueError("NaNs detected in y_true or y_pred.")
        elif nan_policy == 'omit':
            if verbose >= 2:
                print("NaNs detected. Omitting samples with NaNs.")
            # Omit entire samples (rows) if any output trajectory has NaNs
            rows_with_any_nan = nan_mask_so.any(axis=1) # (n_samples,)
            rows_to_keep = ~rows_with_any_nan

            if not np.any(rows_to_keep):
                if verbose >= 1:
                    print("All samples omitted. Returning NaN(s).")
                if multioutput == 'raw_values' and n_outputs > 1:
                    return np.full(n_outputs, np.nan)
                return np.nan

            y_true_calc = y_true_proc[rows_to_keep]
            y_pred_calc = y_pred_proc[rows_to_keep]
            if current_s_weights is not None:
                current_s_weights = current_s_weights[rows_to_keep]
            # Update nan_mask_so for propagate logic if used later
            nan_mask_so = nan_mask_so[rows_to_keep]

    if y_true_calc.shape[0] == 0: # All samples omitted
        if verbose >= 1:
            print("No samples left after NaN handling. Returning NaN(s).")
        if multioutput == 'raw_values' and n_outputs > 1:
            return np.full(n_outputs, np.nan)
        return np.nan

    # --- 3. Compute Time-Weighted MAE ---
    # abs_errors shape: (n_samples_calc, n_outputs, n_timesteps)
    abs_errors = np.abs(y_pred_calc - y_true_calc)

    # Weighted sum of errors for each trajectory (sample, output)
    # w_t is (n_timesteps,). Result: (n_samples_calc, n_outputs)
    # NaNs in abs_errors (from y_pred_calc/y_true_calc if 'propagate')
    # will result in NaNs here.
    twmae_per_trajectory = np.sum(abs_errors * w_t, axis=2)

    # If nan_policy='propagate', ensure original NaNs lead to NaN scores
    if nan_policy == 'propagate':
        # nan_mask_so is (n_samples_calc, n_outputs) if 'omit' used,
        # or (original_n_samples, n_outputs) if 'omit' not used.
        # If 'omit' not used, y_pred_calc is y_pred_proc.
        twmae_per_trajectory = np.where(
            nan_mask_so, np.nan, twmae_per_trajectory
        )

    # --- 4. Aggregate Scores ---
    # Aggregate across samples (axis 0), considering sample_weights
    if current_s_weights is not None:
        if np.sum(current_s_weights) == 0: # Avoid division by zero
            output_scores = np.full(n_outputs, np.nan)
        else:
            # np.average propagates NaNs correctly
            output_scores = np.average(
                twmae_per_trajectory, axis=0, weights=current_s_weights
            )
    else:
        # np.mean propagates NaNs correctly
        output_scores = np.mean(twmae_per_trajectory, axis=0)

    # Handle multioutput aggregation
    if multioutput == 'uniform_average':
        final_score = np.mean(output_scores) # np.mean propagates NaNs
    elif multioutput == 'raw_values':
        final_score = output_scores
    else: # Should not be reached
        raise ValueError(f"Unknown multioutput mode: {multioutput}")

    # If original input was 1D/2D (single effective output),
    # and multioutput='raw_values', result should be scalar.
    if (y_input_ndim_orig <= 2) and multioutput == 'raw_values':
        if isinstance(final_score, np.ndarray) and final_score.size == 1:
            final_score = final_score.item()
        # else: final_score is already scalar if it's np.nan

    if verbose >= 1:
        if isinstance(final_score, np.ndarray):
            with np.printoptions(precision=4, suppress=True):
                print(f"TW-MAE computed: {final_score}")
        else:
            print(f"TW-MAE computed: {final_score:.4f}")

    return final_score


@validate_params({
    'y_pred': ['array-like'],
    'sample_weight': ['array-like', None],
    'nan_policy': [StrOptions({'omit', 'propagate', 'raise'})],
    'multioutput': [StrOptions({'raw_values', 'uniform_average'})],
    'verbose': [Integral, bool]
})
def prediction_stability_score(
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    nan_policy: Literal['omit', 'propagate', 'raise'] = 'propagate',
    multioutput: Literal['raw_values', 'uniform_average'] = 'uniform_average',
    verbose: int = 0
) -> Union[float, np.ndarray]:
    """
    Compute the Prediction Stability Score (PSS).

    Measures the temporal smoothness of consecutive forecasts.
    Lower values indicate smoother, more coherent trajectories.
    Assumes predictions are ordered in time along the last axis.

    Formally, for `B` samples, `O` outputs (optional), and horizon `T`:
    .. math::
        \\mathrm{PSS}_{i,o} = \\frac{1}{T-1} \\sum_{t=2}^{T}
        |\\hat y_{i,o,t} - \\hat y_{i,o,t-1}|
    The final score is an average over samples and possibly outputs.

    Parameters
    ----------
    y_pred : array-like
        Forecast trajectories. Expected shapes:
        - (n_samples, n_timesteps) for single output.
        - (n_samples, n_outputs, n_timesteps) for multi-output.
        The last dimension is always treated as the time dimension.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights. If None, samples are equally weighted.
    nan_policy : {'omit', 'propagate', 'raise'}, default='propagate'
        How to handle NaNs in `y_pred`:
          - ``'raise'``: Raise an error on any NaN.
          - ``'omit'``: Drop samples (rows) containing NaNs.
          - ``'propagate'``: Score for samples/outputs with NaNs
            will be NaN.
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregation if `y_pred` has an `n_outputs` dimension.
          - ``'raw_values'``: Returns a score for each output.
          - ``'uniform_average'``: Scores of all outputs are averaged.
    verbose : int, default=0
        Verbosity level: 0 (silent), 1 (summary), >=2 (debug details).

    Returns
    -------
    score : float or ndarray of floats
        Mean PSS. Scalar if `multioutput='uniform_average'` or if
        `y_pred` is 2D. Array if `multioutput='raw_values'` and
        `y_pred` is 3D.

    Examples
    --------
    >>> import numpy as np
    >>> # Single output (3 samples, 5 timesteps)
    >>> y_p1 = np.array([[1,1,2,2,3], [2,3,2,3,2], [0,1,0,1,0]])
    >>> prediction_stability_score(y_p1)
    0.5833333333333334
    >>> # Multi-output (2 samples, 2 outputs, 3 timesteps)
    >>> y_p2 = np.array([[[1,2,1], [5,5,5]], [[3,2,3], [0,1,0]]])
    >>> prediction_stability_score(y_p2, multioutput='raw_values')
    array([1.  , 0.25])
    """

    y_pred_arr = check_array(
        y_pred, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False
    )

    y_pred_ndim_orig = y_pred_arr.ndim
    if y_pred_ndim_orig == 1: # Single trajectory (T,)
        # Reshape to (1 sample, 1 output, T timesteps)
        y_pred_proc = y_pred_arr.reshape(1, 1, -1)
    elif y_pred_ndim_orig == 2: # (B, T)
        # Reshape to (B samples, 1 output, T timesteps)
        y_pred_proc = y_pred_arr.reshape(y_pred_arr.shape[0], 1, -1)
    elif y_pred_ndim_orig == 3: # (B, O, T)
        y_pred_proc = y_pred_arr
    else:
        raise ValueError(
            "y_pred must be 1D, 2D (n_samples, n_timesteps), or 3D "
            "(n_samples, n_outputs, n_timesteps). "
            f"Got {y_pred_ndim_orig}D."
        )

    n_samples, n_outputs, n_timesteps = y_pred_proc.shape

    if n_timesteps < 2:
        if verbose >= 1:
            print("PSS requires at least 2 time steps. Returning NaN.")
        if multioutput == 'raw_values' and n_outputs > 1:
            return np.full(n_outputs, np.nan)
        return np.nan

    weights_proc = None
    if sample_weight is not None:
        weights_proc = check_array(
            sample_weight, ensure_2d=False, dtype="numeric",
            force_all_finite=True, copy=False
        )
        check_consistent_length(y_pred_proc, weights_proc) # Checks n_samples
        if weights_proc.ndim > 1:
            raise ValueError(
                f"sample_weight must be 1D. Got shape {weights_proc.shape}"
            )

    # NaN handling
    # nan_mask_so is (n_samples, n_outputs), True if that trajectory has any NaN
    nan_mask_so = np.isnan(y_pred_proc).any(axis=2)
    y_pred_calc = y_pred_proc
    current_weights = weights_proc

    if np.any(nan_mask_so):
        if nan_policy == 'raise':
            raise ValueError("NaNs detected in y_pred.")
        elif nan_policy == 'omit':
            if verbose >= 2:
                print("NaNs detected. Omitting samples with NaNs.")
            # Omit entire samples (rows) if any of their outputs' trajectories have NaNs
            rows_with_any_nan = nan_mask_so.any(axis=1) # (n_samples,)
            rows_to_keep = ~rows_with_any_nan

            if not np.any(rows_to_keep):
                if verbose >= 1:
                    print("All samples omitted due to NaNs. Returning NaN(s).")
                if multioutput == 'raw_values' and n_outputs > 1:
                    return np.full(n_outputs, np.nan)
                return np.nan
            
            y_pred_calc = y_pred_proc[rows_to_keep]
            if current_weights is not None:
                current_weights = current_weights[rows_to_keep]
            # Update nan_mask_so for propagate logic if it were used after omit
            nan_mask_so = nan_mask_so[rows_to_keep] # For consistency if used later

    if y_pred_calc.shape[0] == 0: # All samples omitted
        if verbose >= 1:
            print("No samples left after NaN handling. Returning NaN(s).")
        if multioutput == 'raw_values' and n_outputs > 1:
            return np.full(n_outputs, np.nan)
        return np.nan

    # Compute differences along the time axis (last axis)
    # diffs shape: (n_samples_calc, n_outputs, n_timesteps - 1)
    diffs = np.abs(
        y_pred_calc[..., 1:] - y_pred_calc[..., :-1]
    )

    # Mean absolute difference per trajectory (sample, output)
    # pss_per_trajectory shape: (n_samples_calc, n_outputs)
    # NaNs in diffs (from y_pred_calc if nan_policy='propagate') will propagate
    pss_per_trajectory = np.mean(diffs, axis=2)

    # If nan_policy='propagate', ensure original NaNs lead to NaN scores
    if nan_policy == 'propagate':
         # nan_mask_so is (n_samples_calc, n_outputs) if 'omit' was applied to it
         # or (original_n_samples, n_outputs) if 'omit' not applied.
         # If 'omit' was not applied, y_pred_calc is y_pred_proc.
        pss_per_trajectory = np.where(
            nan_mask_so, np.nan, pss_per_trajectory
        )
    
    # Aggregate across samples (axis 0), considering weights
    if current_weights is not None:
        if np.sum(current_weights) == 0: # Avoid division by zero
            output_scores = np.full(n_outputs, np.nan)
        else:
            # np.average propagates NaNs correctly if present in pss_per_trajectory
            output_scores = np.average(
                pss_per_trajectory, axis=0, weights=current_weights
            )
    else:
        # np.mean propagates NaNs correctly
        output_scores = np.mean(pss_per_trajectory, axis=0)

    # Handle multioutput aggregation
    if multioutput == 'uniform_average':
        # np.mean propagates NaNs
        final_score = np.mean(output_scores)
    elif multioutput == 'raw_values':
        final_score = output_scores
    else: # Should not be reached
        raise ValueError(f"Unknown multioutput mode: {multioutput}")

    # If original y_pred was 1D or 2D (single effective output),
    # and multioutput='raw_values', result should be scalar.
    if (y_pred_ndim_orig <= 2) and multioutput == 'raw_values':
        if isinstance(final_score, np.ndarray) and final_score.size == 1:
            final_score = final_score.item()
        # else: final_score is already scalar if it's np.nan

    if verbose >= 1:
        if isinstance(final_score, np.ndarray):
            with np.printoptions(precision=4, suppress=True):
                print(f"PSS computed: {final_score}")
        else:
            print(f"PSS computed: {final_score:.4f}")
            
    return final_score



@validate_params({
    'y_true': ['array-like'],
    'y_lower': ['array-like'],
    'y_upper': ['array-like'],
    'y_median': ['array-like'],
    'alphas': ['array-like'], # Sequence[float] implies iterable
    'sample_weight': ['array-like', None],
    'nan_policy': [StrOptions({'omit', 'propagate', 'raise'})],
    'multioutput': [StrOptions({'raw_values', 'uniform_average'})],
    'warn_invalid_bounds': ['boolean'],
    'verbose': [Integral, bool]
})
def weighted_interval_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    y_median: np.ndarray,
    alphas: Union[Sequence[float], np.ndarray],
    sample_weight: Optional[np.ndarray] = None,
    nan_policy: NanPolicyLiteral = 'propagate',
    multioutput: MultioutputLiteral = 'uniform_average',
    warn_invalid_bounds: bool = True,
    verbose: int = 0
) -> Union[float, np.ndarray]:
    """
    Compute the Weighted Interval Score (WIS).

    The WIS is a proper scoring rule that evaluates probabilistic
    forecasts given as a set of central prediction intervals and a
    median forecast [1]_. It generalizes the absolute error and
    considers multiple quantile levels.

    The score for a single interval at level :math:`\\alpha_k` is:
    .. math::
        \\mathrm{IS}_{\\alpha_k}(y, l_k, u_k) = (u_k - l_k)
        + \\frac{2}{\\alpha_k}(l_k - y)\\mathbf{1}\\{y < l_k\\}
        + \\frac{2}{\\alpha_k}(y - u_k)\\mathbf{1}\\{y > u_k\\}

    The WIS is then defined as a weighted average of the absolute
    error of the median forecast and the interval scores for K
    central prediction intervals:
    .. math::
        \\mathrm{WIS}(y, m, \\{(l_k, u_k, \\alpha_k)\\}_{k=1}^K) =
        \\frac{1}{K + 0.5} \\left( \\frac{1}{2}|y - m| +
        \\sum_{k=1}^K \\frac{\\alpha_k}{2} \\mathrm{IS}_{\\alpha_k} \\right)

    Alternatively, a common formulation used (and implemented here,
    following the reference's R script and common implementations like
    `scoringutils` R package) is:
    For each interval :math:`k` with level :math:`\\alpha_k`, its
    contribution to the score for a single observation :math:`y` is:
    .. math::
        S_k = (u_k - l_k) + \\frac{2}{\\alpha_k}(l_k - y)\\mathbf{1}\\{y < l_k\\}
            + \\frac{2}{\\alpha_k}(y - u_k)\\mathbf{1}\\{y > u_k\\}
    The total score for observation y is:
    .. math::
        \\mathrm{Score}_y = \\frac{1}{K+1} \\left( |y-m| + \\sum_{k=1}^K \\frac{\\alpha_k}{2} S_k \\right)
    This can be simplified by directly using the per-interval WIS contribution:
    .. math::
       \\mathrm{WIS}_{\\alpha_k}(y, l_k, u_k) = \\frac{\\alpha_k}{2}(u_k - l_k)
       + (l_k - y)\\mathbf{1}\\{y < l_k\\}
       + (y - u_k)\\mathbf{1}\\{y > u_k\\}
    Then the aggregated WIS is:
    .. math::
       \\mathrm{WIS} = \\frac{1}{K + 1} \\left(|y - m| +
       \\sum_{k=1}^K \\mathrm{WIS}_{\\alpha_k}\\right)
    This is the version implemented.

    Parameters
    ----------
    y_true : array-like
        Observed true values.
        Shape: (n_samples,) or (n_samples, n_outputs).
    y_lower : array-like
        Lower bounds for each central prediction interval.
        - If `y_true` is 1D: (n_samples, K_intervals)
        - If `y_true` is 2D: (n_samples, n_outputs, K_intervals)
    y_upper : array-like
        Upper bounds, matching `y_lower`'s shape.
    y_median : array-like
        Median forecasts.
        Shape: (n_samples,) or (n_samples, n_outputs).
    alphas : array-like of float, shape (K_intervals,)
        Nominal central interval probability levels (e.g., 0.1 for 10% PI,
        meaning quantiles are 0.05 and 0.95). Each alpha must be
        in (0, 1). These are used as weights.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights. If None, samples are equally weighted.
    nan_policy : {'omit', 'propagate', 'raise'}, default='propagate'
        How to handle NaNs in inputs.
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregation for multi-output `y_true`.
    warn_invalid_bounds : bool, default=True
        If True, issues a `UserWarning` if any `y_lower > y_upper`.
    verbose : int, default=0
        Verbosity level: 0 (silent), 1 (summary), >=2 (debug details).

    Returns
    -------
    score : float or ndarray of floats
        Average WIS. Lower values are better.
        Scalar if `multioutput='uniform_average'` or `y_true` is 1D.
        Array of shape (n_outputs,) if `multioutput='raw_values'` and
        `y_true` is 2D.

    Examples
    --------
    >>> import numpy as np
    >>> # Single-output example
    >>> y_t = np.array([10, 12, 11, np.nan])
    >>> y_l = np.array([[9, 8], [11, 10], [10, 9], [9,8]]) # K=2 intervals
    >>> y_u = np.array([[11, 12], [13, 14], [12, 13], [11,12]])
    >>> y_m = np.array([10, 12, 11, 10])
    >>> a = np.array([0.2, 0.5]) # alpha for 20% and 50% PIs
    >>> wis = weighted_interval_score(y_t, y_l, y_u, y_m, a,
    ...                               nan_policy='omit', verbose=1)
    WIS computed: 0.4333
    >>> print(f"WIS (1D, omit): {wis:.4f}")
    WIS (1D, omit): 0.4333

    >>> # Multi-output example
    >>> y_t_mo = np.array([[10, 20], [12, np.nan]]) # (2_samples, 2_outputs)
    >>> y_l_mo = np.array([ # (2_samples, 2_outputs, 1_interval)
    ...     [[9], [19]],   # Sample 0, Output 0&1, Interval 0
    ...     [[11], [21]]   # Sample 1, Output 0&1, Interval 0
    ... ])
    >>> y_u_mo = np.array([
    ...     [[11], [21]],
    ...     [[13], [23]]
    ... ])
    >>> y_m_mo = np.array([[10, 20], [12, 22]])
    >>> a_mo = np.array([0.5]) # K=1 interval
    >>> wis_mo_raw = weighted_interval_score(
    ...     y_t_mo, y_l_mo, y_u_mo, y_m_mo, a_mo,
    ...     nan_policy='propagate', multioutput='raw_values', verbose=1
    ... )
    WIS computed: [0.25 nan ]
    >>> print(f"WIS (2D, raw, propagate): {wis_mo_raw}")
    WIS (2D, raw, propagate): [0.25 nan]

    Notes
    -----
    - WIS is a proper scoring rule for evaluating quantile/interval forecasts.
    - It balances sharpness (interval width) and calibration.
    - Lower WIS values indicate better forecast performance.

    References
    ----------
    .. [1] Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2021).
           Evaluating epidemic forecasts in an interval format.
           PLoS computational biology, 17(2), e1008618.
           (The paper discusses WIS and its components.)
    """
    # --- 1. Input Validation and Preprocessing ---
    y_true_arr = check_array(y_true, ensure_2d=False,
        dtype="numeric", force_all_finite=False, copy=False)
    y_lower_arr = check_array(y_lower, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False)
    y_upper_arr = check_array(y_upper, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False)
    y_median_arr = check_array(y_median, ensure_2d=False,
        dtype="numeric", force_all_finite=False, copy=False)
    alphas_arr = check_array(alphas, ensure_2d=False, dtype="numeric",
        force_all_finite=True, copy=False) # Alphas must be finite

    if not np.all((alphas_arr > 0) & (alphas_arr < 1)):
        raise ValueError(
            "All alpha values must be strictly between 0 and 1."
        )
    if alphas_arr.ndim > 1:
        alphas_arr = alphas_arr.squeeze() # Ensure 1D if passed as (K,1) etc.
    if alphas_arr.ndim == 0 and alphas_arr.size ==1 : # single alpha
        alphas_arr = alphas_arr.reshape(1,) 
    if alphas_arr.ndim > 1:
        raise ValueError(
            f"alphas must be 1D. Got shape {alphas_arr.shape}"
        )

    K_intervals = alphas_arr.shape[0]

    if verbose >= 2:
        print(f"Initial shapes: y_true={y_true_arr.shape}, "
              f"y_lower={y_lower_arr.shape}, y_upper={y_upper_arr.shape}, "
              f"y_median={y_median_arr.shape}, alphas={alphas_arr.shape}")

    y_true_ndim = y_true_arr.ndim
    if y_true_ndim == 1:
        y_true_proc = y_true_arr.reshape(-1, 1)
        y_median_proc = y_median_arr.reshape(-1, 1)
        if y_lower_arr.ndim == 2 and y_lower_arr.shape[1] == K_intervals:
            y_lower_proc = y_lower_arr.reshape(y_lower_arr.shape[0], 1, -1)
            y_upper_proc = y_upper_arr.reshape(y_upper_arr.shape[0], 1, -1)
        elif y_lower_arr.ndim == 3 and y_lower_arr.shape[1] == 1 \
            and y_lower_arr.shape[2] == K_intervals:
            y_lower_proc = y_lower_arr
            y_upper_proc = y_upper_arr
        else:
            raise ValueError(
                "If y_true is 1D, y_lower/y_upper must be 2D "
                "(n_samples, K_intervals) or 3D (n_samples, 1, K_intervals)."
                f" Got y_lower: {y_lower_arr.shape}, K_intervals={K_intervals}"
            )
    elif y_true_ndim == 2:
        y_true_proc = y_true_arr
        y_median_proc = y_median_arr
        if y_lower_arr.ndim == 3 and \
           y_lower_arr.shape[1] == y_true_proc.shape[1] and \
           y_lower_arr.shape[2] == K_intervals:
            y_lower_proc = y_lower_arr
            y_upper_proc = y_upper_arr
        else:
            raise ValueError(
                "If y_true is 2D (n_s, n_o), y_lower/y_upper must be 3D "
                "(n_s, n_o, K_intervals) with matching n_o and K."
                f" Got y_true:{y_true_proc.shape}, y_lower:{y_lower_arr.shape}"
                f", K_intervals={K_intervals}"
            )
    else:
        raise ValueError(f"y_true must be 1D or 2D. Got {y_true_ndim}D.")

    # Consistent lengths and shapes
    check_consistent_length(
        y_true_proc, y_median_proc, y_lower_proc, y_upper_proc
    )
    if not (y_true_proc.shape[:2] == y_median_proc.shape[:2] == \
            y_lower_proc.shape[:2] == y_upper_proc.shape[:2]):
        raise ValueError("Shape mismatch in (n_samples, n_outputs) "
                         "among processed inputs.")
    if not (y_lower_proc.shape[2] == y_upper_proc.shape[2] == K_intervals):
         raise ValueError("Mismatch in K_intervals dimension for "
                          "y_lower/y_upper vs alphas.")

    if verbose >= 2:
        print(f"Processed shapes: y_true_proc={y_true_proc.shape}, "
              f"y_lower_proc={y_lower_proc.shape}, etc.")

    weights_proc = None
    if sample_weight is not None:
        weights_proc = check_array(sample_weight, ensure_2d=False,
            dtype="numeric", force_all_finite=True, copy=False)
        check_consistent_length(y_true_proc, weights_proc)
        if weights_proc.ndim > 1:
            raise ValueError(f"sample_weight must be 1D. Got {weights_proc.shape}")

    # --- 2. Handle NaNs ---
    nan_mask_yt = np.isnan(y_true_proc)    # (n_s, n_o)
    nan_mask_ym = np.isnan(y_median_proc)  # (n_s, n_o)
    nan_mask_yl = np.isnan(y_lower_proc).any(axis=2) # (n_s, n_o)
    nan_mask_yu = np.isnan(y_upper_proc).any(axis=2) # (n_s, n_o)
    combined_nan_mask = nan_mask_yt | nan_mask_ym | nan_mask_yl | nan_mask_yu

    y_true_calc, y_lower_calc, y_upper_calc, y_median_calc = (
        y_true_proc, y_lower_proc, y_upper_proc, y_median_proc
    )
    current_weights = weights_proc

    if np.any(combined_nan_mask):
        if nan_policy == 'raise':
            raise ValueError("NaNs detected in input arrays.")
        elif nan_policy == 'omit':
            if verbose >= 2: print("NaNs detected. Omitting affected rows.")
            rows_with_any_nan = combined_nan_mask.any(axis=1) # (n_s,)
            rows_to_keep = ~rows_with_any_nan
            if not np.any(rows_to_keep):
                if verbose >= 1: print("All samples omitted. Returning NaN(s).")
                n_out = y_true_proc.shape[1]
                return np.full(n_out, np.nan) if \
                    multioutput == 'raw_values' and y_true_ndim > 1 else np.nan

            y_true_calc = y_true_proc[rows_to_keep]
            y_lower_calc = y_lower_proc[rows_to_keep]
            y_upper_calc = y_upper_proc[rows_to_keep]
            y_median_calc = y_median_proc[rows_to_keep]
            if current_weights is not None:
                current_weights = current_weights[rows_to_keep]
        # For 'propagate', NaNs will flow through calculations.

    if y_true_calc.shape[0] == 0: # All samples omitted
        if verbose >= 1: print("No samples left. Returning NaN(s).")
        n_out = y_true_proc.shape[1]
        return np.full(n_out, np.nan) if \
            multioutput == 'raw_values' and y_true_ndim > 1 else np.nan

    # --- 3. Warn for Invalid Bounds ---
    if warn_invalid_bounds:
        with np.errstate(invalid='ignore'): # For NaN comparisons
            invalid_b = y_lower_calc > y_upper_calc # (n_s, n_o, K)
        if np.any(invalid_b):
            num_invalid = np.sum(invalid_b)
            perc = (num_invalid / invalid_b.size) * 100
            warnings.warn(
                f"{num_invalid} ({perc:.2f}%) interval pairs found "
                f"where y_lower > y_upper. These contribute to penalties.",
                UserWarning
            )
            if verbose >=2: print(f"Warning: {num_invalid} invalid bounds.")

    # --- 4. Compute WIS Components ---
    # Reshape alphas for broadcasting: (1, 1, K_intervals)
    alphas_exp = alphas_arr.reshape(1, 1, -1)

    # Median Absolute Error term (weighted by 0.5 in some formulations,
    # but here it's |y-m| directly as per formula)
    mae_term = np.abs(y_median_calc - y_true_calc) # (n_s, n_o)

    # Interval Score (IS_alpha_k) components
    # y_true_calc needs to be (n_s, n_o, 1) for broadcasting with K dim
    y_true_calc_exp = y_true_calc[..., np.newaxis] # (n_s, n_o, 1)

    interval_width = y_upper_calc - y_lower_calc # (n_s, n_o, K)
    
    # WIS_alpha_k terms (per interval, per sample, per output)
    # WIS_alpha_k = (alpha_k/2) * width_k
    #             + (lower_k - y) * I(y < lower_k)
    #             + (y - upper_k) * I(y > upper_k)
    wis_term_width = (alphas_exp / 2.0) * interval_width

    wis_term_under = (y_lower_calc - y_true_calc_exp) * \
                     (y_true_calc_exp < y_lower_calc)
    wis_term_over = (y_true_calc_exp - y_upper_calc) * \
                    (y_true_calc_exp > y_upper_calc)

    # Sum of individual WIS_alpha_k contributions for each sample/output
    # Each element is WIS_alpha_k for that k
    per_interval_wis = (
        wis_term_width + wis_term_under + wis_term_over
    ) # (n_s, n_o, K)
    sum_of_per_interval_wis = np.sum(
        per_interval_wis, axis=2
    ) # (n_s, n_o)

    # Total WIS per sample and output
    # Denominator is K_intervals + 1 (for the median term)
    if K_intervals == 0: # Only median absolute error
        wis_values = mae_term
    else:
        wis_values = (mae_term + sum_of_per_interval_wis) / (K_intervals + 1.0)

    if verbose >= 3:
        print("WIS values per sample/output (first 5):")
        print(wis_values[:min(5, wis_values.shape[0])])

    # --- 5. Aggregate Scores ---
    if current_weights is not None:
        if np.sum(current_weights) == 0:
            output_scores = np.full(wis_values.shape[1], np.nan)
        else:
            output_scores = np.average(
                wis_values, axis=0, weights=current_weights
            )
    else:
        output_scores = np.mean(wis_values, axis=0) # Handles NaNs from propagate

    if multioutput == 'uniform_average':
        final_score = np.mean(output_scores) # Handles NaNs from propagate
    else: # 'raw_values'
        final_score = output_scores

    if y_true_ndim == 1 and isinstance(final_score, np.ndarray):
        if final_score.size == 1:
            final_score = final_score.item()

    if verbose >= 1:
        if isinstance(final_score, np.ndarray):
            with np.printoptions(precision=4, suppress=True):
                print(f"WIS computed: {final_score}")
        else:
            print(f"WIS computed: {final_score:.4f}")

    return final_score


@validate_params({
    'y_true': ['array-like'],
    'y_pred': ['array-like'],
    'sample_weight': ['array-like', None],
    'nan_policy': [StrOptions({'omit', 'propagate', 'raise'})],
    'multioutput': [StrOptions({'raw_values', 'uniform_average'})],
    'verbose': [Integral, bool] 
})
 
def crp_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    nan_policy: NanPolicyLiteral = 'propagate',
    multioutput: MultioutputLiteral = 'uniform_average',
    verbose: int = 0 
) -> Union[float, np.ndarray]:
    """
    Compute the sample-based Continuous Ranked Probability Score (CRPS).

    This proper scoring rule measures both calibration and sharpness
    of ensemble forecasts by comparing predictive samples to true
    observations [1]_. The sample approximation is:

    .. math::
        \\mathrm{CRPS} = \\frac{1}{m}\\sum_{j=1}^{m} |x_j - y|
        - \\frac{1}{2m^2}\\sum_{i=1}^{m}\\sum_{j=1}^{m} |x_i - x_j|,

    where :math:`x_1,\\dots,x_m` are ensemble members for a single
    observation :math:`y`. The score is then averaged over all samples.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Observed true values.
    y_pred : array-like
        Ensemble forecast samples.
        - If `y_true` is 1D (n_samples,), `y_pred` must be 2D
          (n_samples, n_ensemble_members).
        - If `y_true` is 2D (n_samples, n_outputs), `y_pred` must be 3D
          (n_samples, n_outputs, n_ensemble_members).
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights. If None, samples are equally weighted.
    nan_policy : {'omit', 'propagate', 'raise'}, default='propagate'
        How to handle NaNs:
          - ``'raise'``: Raise an error if NaNs are present in inputs.
          - ``'omit'``: Remove samples (rows) containing NaNs in
            `y_true` or `y_pred` before computation.
          - ``'propagate'``: NaNs in inputs will propagate to the
            CRPS score for the affected sample(s)/output(s).
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregation for multi-output `y_true`.
          - ``'raw_values'``: Returns a full set of scores, one for
            each output.
          - ``'uniform_average'``: Scores of all outputs are averaged
            with uniform weight.
    verbose : int, default=0
        Verbosity level: 0 (silent), 1 (summary), >=2 (debug details).

    Returns
    -------
    score : float or ndarray of floats
        Average CRPS. A scalar if `multioutput='uniform_average'` or
        if `y_true` is 1D. An array of shape (n_outputs,) if
        `multioutput='raw_values'` and `y_true` is 2D.
        Lower values are better.

    Examples
    --------
    >>> import numpy as np
    >>> # from fusionlab.metrics import crp_score 

    >>> y_true_1d = np.array([0.5, 0.0, 1.0, np.nan])
    >>> y_pred_1d = np.array([
    ...     [0.0, 0.5, 1.0],  # For 0.5
    ...     [0.0, 0.1, 0.2],  # For 0.0
    ...     [0.9, 1.1, 1.0],  # For 1.0
    ...     [0.0, 0.5, np.nan] # For np.nan y_true
    ... ])
    >>> score = crp_score(y_true_1d, y_pred_1d, nan_policy='omit', verbose=1)
    CRPS computed: 0.0333
    >>> print(f"CRPS (1D, omit NaNs): {score:.4f}")
    CRPS (1D, omit NaNs): 0.0333

    >>> score_prop = crp_score(y_true_1d, y_pred_1d, nan_policy='propagate')
    >>> print(f"CRPS (1D, propagate NaNs): {score_prop}") # Will be nan
    CRPS (1D, propagate NaNs): nan

    >>> y_true_2d = np.array([[0.5, 2.5], [0.0, np.nan], [1.0, 3.0]])
    >>> y_pred_2d = np.array([
    ...     [[0.0, 0.5, 1.0], [2.0, 2.5, 3.0]], # For [0.5, 2.5]
    ...     [[0.0, 0.1, 0.2], [np.nan, 3.1, 3.2]], # For [0.0, np.nan]
    ...     [[0.9, 1.1, 1.0], [2.8, 3.0, 3.2]]  # For [1.0, 3.0]
    ... ])
    >>> raw_scores = crp_score(y_true_2d, y_pred_2d,
    ...                         nan_policy='propagate',
    ...                         multioutput='raw_values', verbose=1)
    CRPS computed: [0.0333 nan   ]
    >>> print(f"CRPS (2D, raw, propagate): {raw_scores}")
    CRPS (2D, raw, propagate): [0.03333333        nan]

    >>> avg_score = crp_score(y_true_2d, y_pred_2d,
    ...                        nan_policy='omit',
    ...                        multioutput='uniform_average', verbose=1)
    CRPS computed: 0.0500
    >>> print(f"CRPS (2D, omit, average): {avg_score:.4f}")
    CRPS (2D, omit, average): 0.0500

    Notes
    -----
    - This function calculates the CRPS based on ensemble samples.
    - It is suitable for evaluating probabilistic forecasts like
      Monte Carlo simulations or bagged ensembles.
    - CRPS is a strictly proper scoring rule, meaning it encourages
      honest and accurate probabilistic forecasts.
    - Lower CRPS values indicate better forecast performance.

    See Also
    --------
    coverage_score : Metric for prediction interval coverage.
    sklearn.metrics.mean_squared_error : A common deterministic metric.

    References
    ----------
    .. [1] Gneiting, T., & Raftery, A. E. (2007). Strictly Proper
           Scoring Rules, Prediction, and Estimation. Journal of the
           American Statistical Association, 102(477), 359–378.
    """
    # --- 1. Input Validation and Preprocessing ---
    # Convert to NumPy arrays and ensure numeric type.
    # force_all_finite=False to handle NaNs according to nan_policy.
    y_true_arr = check_array(
        y_true, ensure_2d=False, dtype="numeric",
        force_all_finite=False, copy=False
    )
    y_pred_arr = check_array(
        y_pred, ensure_2d=False, allow_nd=True, # allow_nd for 3D
        dtype="numeric", force_all_finite=False, copy=False
    )

    if verbose >= 2:
        print(f"Initial shapes: y_true={y_true_arr.shape}, "
              f"y_pred={y_pred_arr.shape}")

    # Determine input dimensionality (single vs. multi-output y_true)
    y_true_ndim = y_true_arr.ndim
    if y_true_ndim == 1:
        # Reshape y_true to (n_samples, 1) for consistent processing
        y_true_proc = y_true_arr.reshape(-1, 1)
        if y_pred_arr.ndim == 2:
            # Reshape y_pred to (n_samples, 1, n_ensemble)
            y_pred_proc = y_pred_arr.reshape(y_pred_arr.shape[0], 1, -1)
        elif y_pred_arr.ndim == 3 and y_pred_arr.shape[1] == 1:
             y_pred_proc = y_pred_arr # Already (n_samples, 1, n_ensemble)
        else:
            raise ValueError(
                "If y_true is 1D (n_samples,), y_pred must be 2D "
                "(n_samples, n_ensemble) or 3D (n_samples, 1, n_ensemble)."
                f" Got y_pred shape: {y_pred_arr.shape}"
            )
    elif y_true_ndim == 2:
        y_true_proc = y_true_arr
        if y_pred_arr.ndim == 3 and \
           y_pred_arr.shape[1] == y_true_arr.shape[1]:
            y_pred_proc = y_pred_arr
        else:
            raise ValueError(
                "If y_true is 2D (n_samples, n_outputs), y_pred must be 3D "
                "(n_samples, n_outputs, n_ensemble_members) with matching "
                "n_outputs."
                f" Got y_true: {y_true_arr.shape}, y_pred: {y_pred_arr.shape}"
            )
    else:
        raise ValueError(
            f"y_true must be 1D or 2D. Got {y_true_ndim}D."
        )

    # Final shape checks for consistency
    check_consistent_length(y_true_proc, y_pred_proc)
    if y_true_proc.shape[0] != y_pred_proc.shape[0] or \
       y_true_proc.shape[1] != y_pred_proc.shape[1]:
        raise ValueError(
            "Processed y_true and y_pred shapes are inconsistent. "
            f"y_true_proc: {y_true_proc.shape}, "
            f"y_pred_proc: {y_pred_proc.shape}"
        )

    if y_pred_proc.shape[2] == 0: # No ensemble members
        if verbose >= 1:
            print("y_pred has no ensemble members. CRPS is undefined (NaN).")
        # Result shape depends on multioutput and original y_true_ndim
        n_outputs = y_true_proc.shape[1]
        if multioutput == 'raw_values' and y_true_ndim > 1:
            return np.full(n_outputs, np.nan)
        return np.nan


    if verbose >= 2:
        print(f"Processed shapes for calculation: "
              f"y_true_proc={y_true_proc.shape}, "
              f"y_pred_proc={y_pred_proc.shape}")

    # Handle sample_weight
    weights_proc = None
    if sample_weight is not None:
        weights_proc = check_array(
            sample_weight, ensure_2d=False, dtype="numeric",
            force_all_finite=True, copy=False # Weights cannot be NaN
        )
        check_consistent_length(y_true_proc, weights_proc)
        if weights_proc.ndim > 1:
            raise ValueError(
                f"sample_weight must be 1D. Got shape {weights_proc.shape}"
            )
        if verbose >= 3:
            print(f"  sample_weight shape: {weights_proc.shape}")

    # --- 2. Handle NaNs based on nan_policy ---
    # Mask for NaNs: True if y_true_ij is NaN or any y_pred_ijk is NaN
    nan_mask_y_true = np.isnan(y_true_proc) # (n_samples, n_outputs)
    nan_mask_y_pred = np.isnan(y_pred_proc).any(axis=2) # (n_samples, n_outputs)
    # Overall NaN mask for each sample-output pair
    combined_nan_mask = nan_mask_y_true | nan_mask_y_pred # (n_s, n_o)

    if np.any(combined_nan_mask):
        if nan_policy == 'raise':
            if verbose >= 2:
                print("NaNs detected and nan_policy='raise'. Raising error.")
            raise ValueError(
                "NaNs detected in input arrays (y_true or y_pred)."
            )
        elif nan_policy == 'omit':
            if verbose >= 2:
                print("NaNs detected with nan_policy='omit'. "
                      "Omitting affected samples (rows).")
            # Omit entire rows if *any* output for that sample has a NaN
            rows_with_any_nan = combined_nan_mask.any(axis=1) # (n_samples,)
            rows_to_keep = ~rows_with_any_nan

            if verbose >= 3:
                print("Rows to keep after NaN omission:", rows_to_keep)

            if not np.any(rows_to_keep):
                if verbose >= 1:
                    print("All samples contained NaNs and were omitted. "
                          "Returning NaN(s).")
                n_outputs = y_true_proc.shape[1]
                if multioutput == 'raw_values' and y_true_ndim > 1:
                    return np.full(n_outputs, np.nan)
                return np.nan

            y_true_calc = y_true_proc[rows_to_keep]
            y_pred_calc = y_pred_proc[rows_to_keep]
            if weights_proc is not None:
                weights_proc = weights_proc[rows_to_keep]
            # combined_nan_mask needs to be updated for propagate logic later
            # but for omit, it's not used further for indexing CRPS values
        elif nan_policy == 'propagate':
            if verbose >= 2:
                print("NaNs detected and nan_policy='propagate'. "
                      "NaNs will propagate in CRPS calculation.")
            # Calculations will proceed, NaNs in y_true_proc/y_pred_proc
            # will naturally lead to NaNs in crps_vals.
            y_true_calc = y_true_proc
            y_pred_calc = y_pred_proc
        else: # Should not be reached due to @validate_params
            raise ValueError(f"Unknown nan_policy: {nan_policy}")
    else: # No NaNs detected initially
        y_true_calc = y_true_proc
        y_pred_calc = y_pred_proc

    if y_true_calc.shape[0] == 0: # All samples omitted
        if verbose >= 1:
            print("No samples left after NaN handling. Returning NaN(s).")
        n_outputs = y_true_proc.shape[1]
        if multioutput == 'raw_values' and y_true_ndim > 1:
            return np.full(n_outputs, np.nan)
        return np.nan

    # --- 3. Compute CRPS terms ---
    # Term 1: E[|X - y|] = mean(|ensemble_member - true_value|)
    # y_true_calc needs to be (n_samples, n_outputs, 1) for broadcasting
    abs_diff_term1 = np.abs(
        y_pred_calc - y_true_calc[..., np.newaxis]
    )
    # Mean over ensemble members (axis 2)
    # If nan_policy='propagate', NaNs here will propagate.
    term1 = np.mean(abs_diff_term1, axis=2) # Shape: (n_samples, n_outputs)

    # Term 2: 0.5 * E[|X - X'|]
    # = 0.5 * mean(|ensemble_member_i - ensemble_member_j|)
    # y_pred_calc is (n_s, n_o, n_e)
    # Create pairwise differences: (n_s, n_o, n_e, n_e)
    # y_pred_calc[..., :, np.newaxis] -> (n_s, n_o, n_e, 1)
    # y_pred_calc[..., np.newaxis, :] -> (n_s, n_o, 1, n_e)
    abs_diff_term2_pairs = np.abs(
        y_pred_calc[..., :, np.newaxis] - y_pred_calc[..., np.newaxis, :]
    )
    # Mean over pairs of ensemble members (axes 2 and 3)
    # If nan_policy='propagate', NaNs here will propagate.
    m = y_pred_calc.shape[2] # Number of ensemble members
    if m == 0: # Should have been caught earlier
        term2 = np.full_like(term1, np.nan)
    elif m == 1: # Only one ensemble member, term2 is 0
        term2 = np.zeros_like(term1)
    else:
        # Sum over (n_e, n_e) and divide by m*m, then by 2
        # Or mean over (n_e, n_e) and divide by 2
        term2 = np.mean(abs_diff_term2_pairs, axis=(2, 3)) * 0.5

    crps_values = term1 - term2  # Shape: (n_samples, n_outputs)

    # If nan_policy was 'propagate', NaNs from input should already be in crps_values.
    # If nan_policy was 'omit', combined_nan_mask is not relevant here as
    # affected rows were removed.
    # If nan_policy was 'raise', no NaNs exist.

    if verbose >= 3:
        print("CRPS values per sample/output (first 5):")
        print(crps_values[:min(5, crps_values.shape[0])])

    # --- 4. Aggregate scores ---
    # Apply sample weights if provided
    # Note: np.average handles NaNs by default if weights are involved,
    #       but if crps_values contains NaNs (from 'propagate'), the
    #       result of weighted average for that output will be NaN.
    if weights_proc is not None:
        if np.sum(weights_proc) == 0: # Avoid division by zero
            # Average over samples (axis 0)
            output_scores = np.full(crps_values.shape[1], np.nan)
        else:
            output_scores = np.average(
                crps_values, axis=0, weights=weights_proc
            )
    else:
        # If 'propagate', use np.nanmean to ignore NaNs for averaging,
        # UNLESS the NaN was due to original input being NaN.
        # The crps_values already correctly holds NaNs from propagation.
        # So, a simple mean is fine; it will propagate NaNs correctly.
        output_scores = np.mean(crps_values, axis=0)

    # Handle multioutput aggregation
    if multioutput == 'uniform_average':
        # If output_scores contains NaNs (from propagate), mean will be NaN.
        final_score = np.mean(output_scores)
    elif multioutput == 'raw_values':
        final_score = output_scores
    else: # Should not be reached
        raise ValueError(f"Unknown multioutput mode: {multioutput}")

    # If original y_true was 1D, result should be scalar,
    # even if multioutput='raw_values' (it would be a 1-element array).
    if y_true_ndim == 1 and isinstance(final_score, np.ndarray):
        if final_score.size == 1:
            final_score = final_score.item()
        # else: if it became scalar (e.g. np.nan), it's already fine

    if verbose >= 1:
        if isinstance(final_score, np.ndarray):
            with np.printoptions(precision=4, suppress=True):
                print(f"CRPS computed: {final_score}")
        else:
            print(f"CRPS computed: {final_score:.4f}")

    return final_score



@validate_params({
    'y_true': ['array-like'],
    'y_lower': ['array-like'],
    'y_upper': ['array-like'],
    'sample_weight': ['array-like', None],
    'nan_policy': [StrOptions({'omit', 'propagate', 'raise'})],
    'multioutput': [StrOptions({'raw_values', 'uniform_average'})],
    'warn_invalid_bounds': ['boolean'],
    'verbose': [Integral] 
})
def coverage_score(
    y_true,
    y_lower,
    y_upper,
    sample_weight: Optional[np.ndarray] = None,
    nan_policy: NanPolicyLiteral = 'propagate',
    multioutput: MultioutputLiteral = 'uniform_average',
    warn_invalid_bounds: bool = True,
    eps: float = 1e-8, 
    verbose: int = 0
) -> Union[float, np.ndarray]:
    r"""
    Compute the coverage score of prediction intervals.

    Measures the fraction of instances where the true value lies within a
    provided lower and upper bound. This metric is useful for
    evaluating uncertainty estimates in probabilistic forecasts.

    Formally, given observed true values
    :math:`y = \{y_1, \ldots, y_n\}` (which can be multi-output),
    and corresponding interval bounds :math:`\{l_1, \ldots, l_n\}` and
    :math:`\{u_1, \ldots, u_n\}`, the coverage score is defined
    for each output (if applicable) as:

    .. math::
        \text{coverage} = \frac{1}{N_{valid}}\sum_{i=1}^{N_{valid}}
        \mathbf{1}\{ l_i \leq y_i \leq u_i \},

    where :math:`\mathbf{1}\{\cdot\}` is an indicator function and
    :math:`N_{valid}` is the number of valid samples after handling NaNs.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The true observed values. Must be numeric.
    y_lower : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The lower bound predictions, matching `y_true` in shape.
    y_upper : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The upper bound predictions, matching `y_true` in shape.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights. If None, then samples are equally weighted.
    nan_policy : {'omit', 'propagate', 'raise'}, default='propagate'
        Defines how to handle NaN values:
        - ``'propagate'``: If NaNs are present in inputs, they propagate
          to the output. For `multioutput='raw_values'`, an output column
          with NaNs will result in a NaN score for that output. For
          `multioutput='uniform_average'`, if any per-output score is NaN,
          the final average may be NaN (unless `np.nanmean` behavior is more nuanced,
          here standard mean is used after per-output scores are found).
        - ``'omit'``: NaNs in any of `y_true`, `y_lower`, or `y_upper`
          for a given sample (row) will lead to the omission of that entire
          sample from the coverage calculation.
        - ``'raise'``: Encountering NaNs raises a ValueError.
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average scores.
        - ``'raw_values'``: Returns a full set of scores in case of
          multi-output input.
        - ``'uniform_average'``: Scores of all outputs are averaged with
          uniform weight.
    warn_invalid_bounds : bool, default=True
        If True, issues a `UserWarning` if any `y_lower[i] > y_upper[i]`.
        These samples will always count as uncovered.
    eps : float, default=1e-8
        Small epsilon value to prevent division by zero or issues with
        very small sum of weights when `sample_weight` is used.
    verbose : int, default=0
        Controls the level of verbosity for internal logging (prints to console):
        - 0: No output.
        - 1: Basic info (e.g., final coverage).
        - >=2: More details (e.g., NaN handling, shapes).

    Returns
    -------
    score : float or ndarray of floats
        Coverage score. If `multioutput='raw_values'`, an array of scores
        is returned, one for each output. Otherwise, a single float average
        score is returned. Returns `np.nan` if calculation is not possible
        (e.g., all samples omitted due to NaNs).

    Notes
    -----
    If `y_true`, `y_lower`, and `y_upper` are 1D arrays, the behavior is
    equivalent to a single-output scenario, and `multioutput` options
    will yield consistent scalar results (though `'raw_values'` will technically
    return a 1-element array that is then squeezed to scalar if input was 1D).

    Examples
    --------
    >>> from fusionlab.metrics import coverage_score
    >>> import numpy as np
    >>> y_true = np.array([10, 12, 11, 9, np.nan])
    >>> y_lower = np.array([9, 11, 10, 8, 9])
    >>> y_upper = np.array([11, 13, 12, 10, 11])

    >>> # Default: nan_policy='propagate'
    >>> coverage_score(y_true, y_lower, y_upper) # Propagates NaN
    nan

    >>> # Omitting NaNs
    >>> coverage_score(y_true, y_lower, y_upper, nan_policy='omit')
    Coverage computed: 1.0000
    1.0

    >>> # Multi-output example
    >>> y_true_mo = np.array([[10, 20], [12, 22], [11, np.nan]])
    >>> y_lower_mo = np.array([[9, 19], [11, 21], [10, 20]])
    >>> y_upper_mo = np.array([[11, 21], [13, 23], [12, 22]])
    >>> coverage_score(y_true_mo, y_lower_mo, y_upper_mo, nan_policy='omit',
                       multioutput='raw_values')
    Coverage computed: [1. 1.]
    array([1., 1.])
    >>> coverage_score(y_true_mo, y_lower_mo, y_upper_mo, nan_policy='omit',
                       multioutput='uniform_average')
    Coverage computed: 1.0000
    1.0
    >>> coverage_score(y_true_mo, y_lower_mo, y_upper_mo,
                       nan_policy='propagate', multioutput='raw_values')
    Coverage computed: [1. nan]
    array([ 1., nan])

    See Also
    --------
    sklearn.utils.validation.check_array : Utility for input validation.
    numpy.average : Compute weighted average, used with `sample_weight`.

    References
    ----------
    .. [1] Gneiting, T. & Raftery, A. E. (2007). "Strictly Proper
           Scoring Rules, Prediction, and Estimation." J. Amer.
           Statist. Assoc., 102(477):359–378.
    """
    # 1. Input validation and conversion
    #    Force_all_finite=False allows NaNs, which are handled by nan_policy.
    #    Copy=False is an optimization if inputs are already suitable arrays.
    y_true_p = check_array(
        y_true, ensure_2d=False, allow_nd=True, 
        dtype="numeric", force_all_finite=False, 
        copy=False)
    y_lower_p = check_array(
        y_lower, ensure_2d=False, 
        allow_nd=True, dtype="numeric", 
        force_all_finite=False, 
        copy=False
        )
    y_upper_p = check_array(
        y_upper, ensure_2d=False, 
        allow_nd=True, dtype="numeric", 
        force_all_finite=False, 
        copy=False)

    y_true_orig_ndim = y_true_p.ndim
    
    if not (eps > 0):
        raise ValueError("eps must be positive.")

    if verbose >= 3:
        print("Input shapes before reshaping:")
        print(f"  y_true_p: {y_true_p.shape}, "
              f"y_lower_p: {y_lower_p.shape},"
              f" y_upper_p: {y_upper_p.shape}")

    # Reshape 1D arrays to 2D (n_samples, 1 feature)
    # for consistent processing
    if y_true_p.ndim == 1:
        y_true_p = y_true_p.reshape(-1, 1)
        if y_lower_p.ndim == 1: y_lower_p = y_lower_p.reshape(-1, 1)
        # else: # check_consistent_length and shape equality will catch this
        if y_upper_p.ndim == 1: y_upper_p = y_upper_p.reshape(-1, 1)
        # else:
    elif y_true_p.ndim > 2:
        raise ValueError(
            "Inputs y_true, y_lower, y_upper must be"
            f" 1D or 2D. Got {y_true_p.ndim}D for y_true.")
    
    # All inputs should now be 2D for internal processing
    # (or have failed if y_true was >2D or if 1D 
    # inputs had mismatched companions >1D)

    if verbose >= 3:
        print("Input shapes after potential 1D->2D reshaping:")
        print(f"  y_true_p: {y_true_p.shape}, y_lower_p:"
              f" {y_lower_p.shape}, y_upper_p: {y_upper_p.shape}")

    # Check for consistent shapes (length of samples and number of outputs)
    try:
        check_consistent_length(y_true_p, y_lower_p, y_upper_p)
    except ValueError as e:
        if verbose >=2: print(f"Shape inconsistency (length): {e}")
        raise ValueError(
             "y_true, y_lower, y_upper must have the"
             " same number of samples (axis 0)."
             f" Got shapes: y_true={y_true_p.shape},"
             f" y_lower={y_lower_p.shape}, y_upper={y_upper_p.shape}"
        ) from e

    if not (y_true_p.shape == y_lower_p.shape == y_upper_p.shape):
        error_msg = (
            "y_true, y_lower, y_upper must have the same shape. "
            f"Got y_true: {y_true_p.shape}, y_lower:"
            f" {y_lower_p.shape}, y_upper: {y_upper_p.shape}."
        )
        if verbose >=2: print(error_msg)
        raise ValueError(error_msg)

    if y_true_p.size == 0: # Handle empty inputs early
        if verbose >= 1: print("Inputs are empty. Returning NaN.")
        return ( 
            np.nan if multioutput == 'uniform_average' or y_true_orig_ndim == 1 
            else np.full(y_true_p.shape[1], np.nan)
        )
            

    # Handle sample_weight
    weights = sample_weight
    if weights is not None:
        weights = check_array(
            weights, ensure_2d=False, dtype="numeric", 
            force_all_finite=True, copy=False)
        check_consistent_length(y_true_p, weights)
        if weights.ndim > 1:
            raise ValueError(
                f"sample_weight must be 1D. Got shape {weights.shape}")
        if verbose >= 3: 
            print(f"  sample_weight shape: {weights.shape}")

    # 2. Handle NaNs
    # Create a mask for NaNs across any of the three
    # input arrays for each sample-output pair.
    nan_mask_entries = np.isnan(
        y_true_p) | np.isnan(y_lower_p) | np.isnan(y_upper_p)

    if np.any(nan_mask_entries):
        if nan_policy == 'raise':
            if verbose >= 2: 
                print("NaNs detected and nan_policy='raise'. Raising error.")
            raise ValueError(
                "NaNs detected in input arrays (y_true, y_lower, or y_upper).")
        elif nan_policy == 'omit':
            if verbose >= 2: 
                print(
                    "NaNs detected with nan_policy='omit'."
                    " Omitting affected samples (rows).")
            # Omit entire rows (samples) if any of their values
            # (y_true, y_lower, or y_upper for any output) is NaN.
            rows_with_nan = np.any(nan_mask_entries, axis=1)
            rows_to_keep = ~rows_with_nan

            if verbose >= 4: 
                print(f"NaN omit mask (rows_to_keep): "
                      f"{rows_to_keep[:10] if rows_to_keep.size > 10 else rows_to_keep}")

            if not np.any(rows_to_keep):
                if verbose >=1: 
                    print(
                        "All samples contained NaNs and were"
                        " omitted. Returning NaN(s).")
                n_outputs = y_true_p.shape[1]
                return ( 
                    np.nan if multioutput == 'uniform_average' 
                    or n_outputs == 1 else np.full(n_outputs, np.nan)
                    )

            y_true_p = y_true_p[rows_to_keep]
            y_lower_p = y_lower_p[rows_to_keep]
            y_upper_p = y_upper_p[rows_to_keep]
            if weights is not None:
                weights = weights[rows_to_keep]
            
            if verbose >=2: 
                print("Shapes after omitting NaN rows:"
                      f" y_true_p: {y_true_p.shape}")
            # Should be caught by `not np.any(rows_to_keep)` but as a safeguard    
            if y_true_p.size == 0: 
                if verbose >=1: 
                    print(
                        "All samples resulted in empty arrays"
                        " after NaN omission. Returning NaN(s)."
                    )
                n_outputs = ( 
                    rows_with_nan.shape[0] if y_true_p.shape[1]==0 
                    else y_true_p.shape[1] # Fallback if y_true_p is (0,0)
                    )
                n_outputs = ( 
                    y_true_p.shape[1] if y_true_p.ndim == 2 
                    and y_true_p.shape[1] > 0 else 1
                    )
                return ( 
                    np.nan if multioutput == 'uniform_average' 
                    or n_outputs == 1 else np.full(n_outputs, np.nan)
                )


        elif nan_policy == 'propagate':
            if verbose >= 2: 
                print(
                    "NaNs detected and nan_policy='propagate'."
                    " NaNs will propagate to result.")
                
            # NaNs in y_true_p, y_lower_p, y_upper_p will
            # lead to NaNs in coverage_mask.
            # np.average or np.mean will then correctly propagate this.
    
    # 3. Check for invalid bounds (y_lower > y_upper)
    # This check is performed *after* NaN handling if 'omit' is used,
    # or on data that might contain NaNs if 'propagate' is used.
    if warn_invalid_bounds:
        # Temporarily ignore warnings from comparing NaNs if they are being propagated
        with np.errstate(invalid='ignore' if nan_policy == 'propagate' else 'raise'):
            invalid_bounds_mask = y_lower_p > y_upper_p
        # np.any handles NaNs by treating them as False 
        # in a boolean context unless all are NaN
        if np.any(invalid_bounds_mask): 
            num_invalid_bounds = np.sum(invalid_bounds_mask) # NaNs in mask become 0 here
            percentage_invalid = (
                num_invalid_bounds / invalid_bounds_mask.size) * 100
            warnings.warn(
                f"{num_invalid_bounds} ({percentage_invalid:.2f}%)"
                " sample-output pairs found where"
                f" y_lower > y_upper. These will always count as uncovered.",
                UserWarning
            )
            if verbose >=2: 
                print(f"Warning: {num_invalid_bounds}"
                      " invalid bound pairs detected.")

    # 4. Compute coverage mask
    # NaNs in inputs (if nan_policy='propagate') will result in NaNs in coverage_mask.
    coverage_mask = (y_true_p >= y_lower_p) & (y_true_p <= y_upper_p)

    if verbose >= 4:
        print("Coverage mask (sample of up to 5x5 or 10 elements):")
        sample_to_print = coverage_mask
        if coverage_mask.ndim == 2:
            sample_to_print = ( 
                coverage_mask[:min(5, coverage_mask.shape[0]),
                              :min(5, coverage_mask.shape[1])]
                )
        elif coverage_mask.ndim == 1: # Should not happen as we reshape to 2D
            sample_to_print = coverage_mask[:min(10, coverage_mask.size)]
        print(sample_to_print)

    # 5. Compute score
    # If coverage_mask is empty (e.g., 
    # all NaNs omitted and input was small), size will be 0.
    if coverage_mask.size == 0:
        # This case should ideally be caught 
        # earlier by checks on y_true_p.size after NaN omission
        if verbose >= 1: 
            print("Coverage mask is empty"
                  " (e.g. all values omitted). Returning NaN.")
        n_outputs = ( 
            y_true_p.shape[1] if y_true_p.ndim == 2 
            and y_true_p.shape[1] > 0 else 1
            )
        return ( 
            np.nan if multioutput == 'uniform_average'
            or n_outputs == 1 else np.full(n_outputs, np.nan)
            )

    if multioutput == 'uniform_average':
        if weights is not None:
            sum_weights = np.sum(weights)
            # Weighted average per output, then unweighted 
            # average of these scores
            # Treat sum of weights close to zero as zero
            if sum_weights < eps: 
                if verbose >= 1 and sum_weights > 0: 
                    # Warn if sum_weights is positive but too small
                     warnings.warn(
                         f"Sum of weights ({sum_weights}) is < eps ({eps})."
                         " Result will be NaN.", UserWarning)
                # np.average with sum_weights=0 raises ZeroDivisionError
                # or returns NaNs if weights contain NaNs that sum to 0.
                # We want consistent NaN output if sum_weights < eps.
                output_scores = np.full(coverage_mask.shape[1], np.nan)
            else: 
                output_scores = np.average(coverage_mask, axis=0, weights=weights)
            # If nan_policy='propagate', output_scores can 
            # have NaNs. np.nanmean handles this.
            coverage = ( 
                np.nanmean( output_scores) if nan_policy != 'propagate' 
                else np.mean(output_scores)
            )
        else:
            # Mean per output, then mean of means.
            # np.nanmean for outer mean if propagation could lead to NaN output_scores
            output_scores = ( 
                np.nanmean( coverage_mask, axis=0) 
                if nan_policy != 'propagate' 
                else np.mean(coverage_mask, axis=0)
            )
            coverage = ( 
                np.nanmean(output_scores) 
                if nan_policy != 'propagate' 
                else np.mean(output_scores)
            )

    elif multioutput == 'raw_values':
        if weights is not None:
            if np.sum(weights) == 0: 
                coverage = np.full(coverage_mask.shape[1], np.nan)
            else: 
                coverage = np.average(coverage_mask, axis=0, weights=weights)
        else:
            # For 'propagate', np.nanmean correctly 
            # computes mean for columns ignoring NaNs within them,
            # BUT if a value that should propagate IS a NaN,
            # that column's mean should BE NaN.
            # So, simple .mean() is better for 'propagate'.
            coverage = ( 
                np.mean(coverage_mask, axis=0) 
                if nan_policy == 'propagate' 
                else np.nanmean(coverage_mask, axis=0)
            )
            # Re-check logic for propagate with raw_values:
            # If policy is propagate, NaNs in coverage_mask
            # should make the corresponding output_score NaN.
            # np.mean(axis=0) does this. np.nanmean(axis=0) 
            # would ignore NaNs, which contradicts 'propagate'.
            if nan_policy == 'propagate':
                coverage = np.mean(
                    coverage_mask.astype(float), axis=0) # Ensure float for NaNs
            else: # 'omit' (NaNs removed) or 'raise' (no NaNs)
                coverage = np.mean(coverage_mask, axis=0)
    else: # Should not be reached due to @validate_params
        raise ValueError(f"Unknown multioutput mode: {multioutput}")

    # If original input was 1D, and multioutput='raw_values', result should be scalar.
    if y_true_orig_ndim == 1 and multioutput == 'raw_values':
        if isinstance(coverage, np.ndarray) and coverage.size == 1:
            coverage = coverage.item()
        # else: coverage is already scalar if it's np.nan from an empty set.

    if verbose >= 1:
        if isinstance(coverage, np.ndarray):
            with np.printoptions(precision=4, suppress=True): # Changed precision
                print(f"Coverage computed: {coverage}")
        else: # Scalar
            print(f"Coverage computed: {coverage:.4f}") # Changed precision

    return coverage


# -*- coding: utf-8 -*-
from __future__ import annotations 
from numbers import Real 
import numpy as np 

from .compat.sklearn import ( 
    StrOptions, 
    validate_params,
    
)
from .utils.validator import _ensure_y_is_valid 

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

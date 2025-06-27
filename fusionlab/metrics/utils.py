# fusionlab/metrics/utils.py 

# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio  <etanoyau@gmail.com>

import os
from typing import Union, Tuple, Optional, Any, Dict, Sequence 

import numpy as np
import pandas as pd 
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from ..core.checks import are_all_frames_valid 
from ..core.diagnose_q import validate_quantiles 

from ._metrics import (
    coverage_score,
    prediction_stability_score,
    quantile_calibration_error,
)
def compute_quantile_diagnostics(
    *dfs: Union[ pd.DataFrame, Sequence[pd.DataFrame]],
    base_name: str,
    quantiles: Sequence[float],
    coverage_quantile_indices: Tuple[int, int] = (0, -1),
    savefile: Optional[str] = None,
    savepath: Optional[str] = None,
    filename: str = "diagnostics.json",
    name : Optional [str] =None, 
    verbose: int = 0,
    logger: Any = None,
) -> Dict[str, float]:
    """
    Compute coverage, prediction stability, and calibration error
    for a set of quantile forecasts, then optionally save results.

    Parameters
    ----------
    *dfs   pd.DataFrame or list of DataFrames, 
        Sequence of DataFrames to concat horizontally. Must contain
        actuals and quantile columns.
    base_name
        Base column name for this forecast (e.g. "temperature").
    quantiles
        Iterable of quantile levels (e.g. [0.1, 0.5, 0.9]).
    coverage_quantile_indices
        Tuple (L, U) giving indices into the sorted quantile list for
        coverage test; default is (0, -1).
    savefile
        If provided, its directory will be used as savepath.
    savepath
        Directory in which to write `filename`. If None and `savefile`
        is None, current working directory is used.
    filename
        Name of the JSON file to write the diagnostics into.
    name: str or optional 
        Name of data to compute quantile diagnostics 
    verbose
        Verbosity level for logging.
    logger
        Logger instance passed to `vlog`.
     
    Returns
    -------
    results : dict
        Dictionary with keys `"coverage"`, `"pss"`, and `"qce"`.
    
    Raises
    ------
    ValueError
        If required columns are missing or inputs have mismatched shapes.
    """
    
    from ..utils.generic_utils import vlog, insert_affix_in 
    from ..utils.io_utils import to_txt 
    
    dfs = are_all_frames_valid (*dfs, ops ='validate')
    # 1. Prepare the DataFrame
    df = pd.concat(list(dfs), axis=1)
    # 2. Validate & sort quantiles
    quantiles_sorted = sorted(
        validate_quantiles(quantiles, dtype=np.float64)
    )
    l_idx, u_idx = coverage_quantile_indices
    try:
        lower_q = quantiles_sorted[l_idx]
        upper_q = quantiles_sorted[u_idx]
    except IndexError:
        raise ValueError(
            f"coverage_quantile_indices {coverage_quantile_indices} "
            f"out of range for {quantiles_sorted}"
        )

    # 3. Build column names
    lower_q_col  = f"{base_name}_q{int(lower_q * 100)}"
    upper_q_col  = f"{base_name}_q{int(upper_q * 100)}"
    actual_col   = f"{base_name}_actual"
    
    # 3b. Determine which quantile to treat as “median”
    n_q = len(quantiles_sorted)
    if n_q == 1:
        # No quantiles or single quantile → use actuals
        median_q_col = actual_col
    elif n_q % 2 == 1:
        # Odd count (3,5,7,…) → pick the true middle
        mid_idx = n_q // 2
        m_q = quantiles_sorted[mid_idx]
        median_q_col = f"{base_name}_q{int(m_q * 100)}"
    else:
        # Even count (2,4,6,…) → average the two central quantiles
        i1, i2 = n_q // 2 - 1, n_q // 2
        q1, q2 = quantiles_sorted[i1], quantiles_sorted[i2]
        # build a synthetic “median” name like q25_75 for 0.25 & 0.75
        median_q_col = f"{base_name}_q{int(q1*100)}_" \
                       f"{int(q2*100)}"

    # 4. Check columns exist
    for col in (lower_q_col, upper_q_col, actual_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # 5. Compute metrics
    coverage = coverage_score(
        df[actual_col], df[lower_q_col], df[upper_q_col]
    )
    y_pred = stack_quantile_predictions(
        q_lower=df[lower_q_col],
        q_median=df[median_q_col],
        q_upper=df[upper_q_col],
    )
    pss = prediction_stability_score(y_pred)
    qce = quantile_calibration_error(
        y_true=df[actual_col],
        y_pred=df[[lower_q_col, median_q_col, upper_q_col]],
        quantiles=quantiles_sorted
    )
    
    # compute standard regression metrics on the median forecast
    y_true = df[actual_col].values
    y_med  = df[median_q_col].values
    r2   = r2_score(y_true, y_med)
    mse  = mean_squared_error(y_true, y_med)
    mae  = mean_absolute_error(y_true, y_med)
    mape = mean_absolute_percentage_error(y_true, y_med)

    # 6. Log coverage
    vlog(
        f"Coverage for {base_name} "
        f"({lower_q:.2f}-{upper_q:.2f}): {coverage:.4f}",
        level=3, verbose=verbose, logger=logger
    )

    results = {
        "r2":       float(r2),
        "mse":      float(mse),
        "mae":      float(mae),
        "mape":     float(mape),
        "coverage": float(coverage),
        "pss":      float(pss),
        "qce":      float(qce),
    }

    # 7. Determine savepath
    if savefile is not None:
        savepath = os.path.dirname(savefile)
    if savepath is None:
        savepath = os.getcwd()
    os.makedirs(savepath, exist_ok=True)

    # 8. Save to JSON
    try:
        vlog(
            f"Saving diagnostics to {os.path.join(savepath, filename)}",
            level=2, verbose=verbose, logger=logger
        )
        to_txt(
            results,
            format='json',
            indent=4,
            filename=insert_affix_in(filename, affix=name, separator='_'),
            savepath=savepath,
        )
    except Exception as e:
        vlog(
            f"Error saving diagnostics: {e}",
            level=1, verbose=verbose, logger=logger
        )

    return results

def stack_quantile_predictions(
    q_lower:   Union[np.ndarray, Sequence],
    q_median:  Union[np.ndarray, Sequence],
    q_upper:   Union[np.ndarray, Sequence],
) -> np.ndarray:
    """
    Stack three quantile trajectories into a single y_pred array
    of shape (n_samples, 3, n_timesteps), ready for PSS.

    Parameters
    ----------
    q_lower, q_median, q_upper : array-like
        Each is either
        - 1D: (n_timesteps,) → interpreted as a single sample, or
        - 2D: (n_samples, n_timesteps)

    Returns
    -------
    y_pred : np.ndarray, shape (n_samples, 3, n_timesteps)
        Where axis=1 indexes [lower, median, upper].

    Raises
    ------
    ValueError
        If the three inputs (after promotion) do not share the same shape.
    """
    def _ensure_2d(arr):
        a = np.asarray(arr)
        if a.ndim == 1:
            return a.reshape(1, -1)
        if a.ndim == 2:
            return a
        raise ValueError(
            f"Each quantile array must be 1D or 2D, got shape {a.shape}")

    lower = _ensure_2d(q_lower)
    median = _ensure_2d(q_median)
    upper = _ensure_2d(q_upper)

    if not (lower.shape == median.shape == upper.shape):
        raise ValueError(
            "All three quantile arrays must have the same shape "
            f"after promotion, got {lower.shape}, {median.shape}, {upper.shape}"
        )

    # Stack along new axis=1 → (n_samples, 3, n_timesteps)
    y_pred = np.stack([lower, median, upper], axis=1)
    return y_pred
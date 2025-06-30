# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Provides high-level utilities for computing and saving diagnostic
metrics for quantile forecasts.
"""
from __future__ import annotations

from pathlib import Path 
from typing import (
    Union, Tuple, Optional, Any, Dict, Sequence, List, Callable
)

import numpy as np
import pandas as pd 
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from ..core.checks import are_all_frames_valid 
from ..core.handlers import _get_valid_kwargs 
from ..core.diagnose_q import validate_quantiles 

from ._metrics import (
    coverage_score,
    prediction_stability_score,
    quantile_calibration_error,
)
def compute_quantile_diagnostics(
    *dfs: Union[ pd.DataFrame, Sequence[pd.DataFrame]],
    target_name: str,
    quantiles: Sequence[float],
    coverage_quantile_indices: Tuple[int, int] = (0, -1),
    savefile: Optional[str] = None,
    filename: str = "diagnostics.json",
    name : Optional [str] =None, 
    metrics: Optional [Union [ List[str], List[callable],  Dict[str, callable]]] =None,   
    default_metrics_set: Optional[str] = 'regression',
    verbose: int = 0,
    logger: Any = None,
    **kwargs # future extension 
) -> Dict[str, float]:
    """Compute and save a suite of diagnostic metrics for quantile forecasts.

    This function calculates a core set of probabilistic forecast
    diagnostics (coverage, PSS, QCE) and can be extended with standard
    regression metrics (e.g., MAE, MSE) applied to the median forecast.

    Parameters
    ----------
    *dfs : pd.DataFrame or Sequence[pd.DataFrame]
        One or more pandas DataFrames containing the forecast results.
        They will be concatenated horizontally. Must contain columns for
        actual values and predicted quantiles.
    target_name : str
        The base name of the target variable used in the column names,
        e.g., "subsidence". The function expects columns like
        "subsidence_actual", "subsidence_q50", etc.
    quantiles : Sequence[float]
        An iterable of the quantile levels that were predicted,
        e.g., ``[0.1, 0.5, 0.9]``.
    coverage_quantile_indices : tuple of (int, int), default=(0, -1)
        A tuple `(L, U)` specifying the indices of the sorted `quantiles`
        list to use for the lower and upper bounds of the coverage score
        calculation. The default `(0, -1)` uses the outermost quantiles.
    savefile : str or os.PathLike, optional
        The path to save the resulting JSON file. If a directory is
        provided, a default filename will be used. If None, the results
        are not saved to disk.
    name : str, optional
        An optional name or affix to include in the output filename to
        distinguish between different diagnostic runs, e.g., 'validation'.
    metrics : list, dict, or None, optional
        A list or dictionary specifying additional regression metrics to
        compute on the median forecast.
        - If a list, can contain strings (e.g., 'mae', 'r2') or callable
          functions.
        - If a dictionary, maps metric names to callable functions.
        - If None, only the metrics from `default_metrics_set` are used.
    default_metrics_set : {'regression', None}, default='regression'
        Specifies a default set of metrics to compute.
        - 'regression': Includes 'r2', 'mse', 'mae', and 'mape'.
        - None: No default metrics are computed; only those specified in
          the `metrics` parameter will be used.
    verbose : int, default=0
        The verbosity level for logging messages.
    logger : logging.Logger, optional
        A logger instance for outputting messages. Defaults to `print`.
    **kwargs
        Additional keyword arguments to be passed to the metric functions.

    Returns
    -------
    dict
        A dictionary containing the computed diagnostic scores.

    Notes
    -----
    The function automatically identifies the median quantile for
    computing standard regression metrics. If an even number of
    quantiles are provided, the `_actual` column is used as the
    point forecast for these metrics.

    Examples
    --------
    >>> from fusionlab.metrics.utils import compute_quantile_diagnostics
    >>> import pandas as pd
    >>> # Create a sample forecast DataFrame
    >>> data = {
    ...     'subsidence_actual': [10, 12, 15, 11],
    ...     'subsidence_q10': [8, 10, 13, 9],
    ...     'subsidence_q50': [10.5, 11.5, 15.2, 10.8],
    ...     'subsidence_q90': [12, 13, 17, 12.5],
    ... }
    >>> df = pd.DataFrame(data)
    >>> quantiles = [0.1, 0.5, 0.9]

    >>> # 1. Compute default diagnostics (core + regression metrics)
    >>> results = compute_quantile_diagnostics(df, base_name='subsidence',
    ...                                        quantiles=quantiles)
    >>> print(sorted(results.keys()))
    ['coverage', 'mae', 'mape', 'mse', 'pss', 'qce', 'r2']

    >>> # 2. Compute only the core quantile diagnostics, no defaults
    >>> results_core = compute_quantile_diagnostics(df, base_name='subsidence',
    ...                                             quantiles=quantiles,
    ...                                             default_metrics_set=None)
    >>> print(sorted(results_core.keys()))
    ['coverage', 'pss', 'qce']

    >>> # 3. Add a custom metric to the defaults
    >>> from sklearn.metrics import mean_squared_log_error
    >>> results_custom = compute_quantile_diagnostics(
    ...     df, base_name='subsidence', quantiles=quantiles,
    ...     metrics=[mean_squared_log_error]
    ... )
    >>> 'mean_squared_log_error' in results_custom
    True
    """
    
    from ..utils.generic_utils import  ( 
        ExistenceChecker,  vlog, insert_affix_in 
    )
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
    lower_q_col  = f"{target_name}_q{int(lower_q * 100)}"
    upper_q_col  = f"{target_name}_q{int(upper_q * 100)}"
    actual_col   = f"{target_name}_actual"
    
    # 3b. Determine which quantile to treat as “median”
    n_q = len(quantiles_sorted)
    if n_q == 1:
        # No quantiles or single quantile → use actuals
        median_q_col = actual_col
    elif n_q % 2 == 1:
        # Odd count (3,5,7,…) → pick the true middle
        mid_idx = n_q // 2
        m_q = quantiles_sorted[mid_idx]
        median_q_col = f"{target_name}_q{int(m_q * 100)}"
    else:
        # Even count (2,4,6,…) → average the two central quantiles
        i1, i2 = n_q // 2 - 1, n_q // 2
        q1, q2 = quantiles_sorted[i1], quantiles_sorted[i2]
        # build a synthetic “median” name like q25_75 for 0.25 & 0.75
        median_q_col = f"{target_name}_q{int(q1*100)}_" \
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
    results = {
        "coverage": float(coverage),
        "pss":      float(pss),
        "qce":      float(qce),
    }
    
    # 6. Log coverage
    vlog(
        f"Coverage for {target_name} "
        f"({lower_q:.2f}-{upper_q:.2f}): {coverage:.4f}",
        level=3, verbose=verbose, logger=logger
    )
    # 5. Compute the core quantile diagnostic metrics
    y_true = df[actual_col].values
    resolved_metrics = _resolve_metrics(metrics, default_set=default_metrics_set)
    y_med = df[median_q_col].values
    
    for metric_name, metric_func in resolved_metrics.items():
        try:
            kwargs = _get_valid_kwargs(metric_func, kwargs)
            score = metric_func(y_true, y_med, **kwargs)
            results[metric_name] = float(score)
        except Exception as e:
            vlog(f"Could not compute metric '{metric_name}': {e}",
                 level=1, verbose=verbose, logger=logger)
    
    # 7. Save results to JSON file if requested
    if savefile:
        save_path = Path(savefile)
        
        # Determine if the provided path is a directory or a full file path
        if save_path.suffix and save_path.name: # It's a file path
            output_dir = save_path.parent
            output_filename = save_path.name
        else: # It's intended as a directory
            output_dir = save_path
            output_filename = "diagnostics_results.json"
        
        # Ensure the output directory exists
        ExistenceChecker.ensure_directory(output_dir)
        
        try:
            # Add the optional name affix to the filename
            full_filename = insert_affix_in(
                output_filename, affix=name, separator='.')
            
            vlog(
                f"Saving diagnostics to {output_dir / full_filename}",
                level=2, verbose=verbose, logger=logger
            )
            
            to_txt(
                results,
                format='json',
                indent=4,
                filename=full_filename,
                savepath=str(output_dir),
                verbose=verbose,
                logger=logger
            )
        except Exception as e:
            vlog(
                f"Error saving diagnostics: {e}",
                level=1, verbose=verbose, logger=logger
            )

    return results


def _resolve_metrics(
    metrics: Optional[Union[List[str], List[Callable], Dict[str, Callable]]],
    default_set: Optional[str] = None
) -> Dict[str, Callable]:
    """
    Parses a flexible list/dict of metrics into a standardized
    dictionary of name -> callable function, with optional default sets.
    """
    # --- Predefined default metric sets ---
    REGRESSION_METRICS = {
        'r2': r2_score,
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        # 'mape': mean_absolute_percentage_error,
    }
    # Future sets will add CLASSIFICATION_METRICS
    # by adding a new dictionary and handling it below.

    resolved_metrics = {}

    # 1. Load the default set if requested by the user
    if default_set:
        if default_set == 'regression':
            resolved_metrics.update(REGRESSION_METRICS)
        elif default_set == 'classification':
            # This is a placeholder for future extension.
            raise NotImplementedError(
                "The 'classification' default metric set is not yet "
                "implemented. Please provide a custom list or dictionary "
                "of classification metrics."
            )
        else:
            # Handle any other unknown default sets gracefully.
            raise ValueError(f"Unknown default_set: '{default_set}'. "
                             "Supported sets are: 'regression'.")

    # 2. Process and merge/override with user-provided metrics
    if metrics is None:
        # If user provides no metrics, return the loaded default set
        return resolved_metrics

    # Determine the set of available metric names for validation
    available_metrics = REGRESSION_METRICS.keys() # Extend this for other sets

    if isinstance(metrics, dict):
        # User-provided dictionary overrides any defaults
        resolved_metrics.update(metrics)
    elif isinstance(metrics, list):
        # User provided a list of strings or callables
        for metric in metrics:
            if isinstance(metric, str):
                if metric in available_metrics:
                    resolved_metrics[metric] = REGRESSION_METRICS[metric]
                else:
                    raise ValueError(f"Unknown metric string: '{metric}'.")
            elif callable(metric):
                resolved_metrics[metric.__name__] = metric
            else:
                raise TypeError(
                    f"Metric must be a string or callable, got {type(metric)}."
                )
    else:
        raise TypeError(
            f"`metrics` must be a list or dict, got {type(metrics)}."
        )

    return resolved_metrics

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
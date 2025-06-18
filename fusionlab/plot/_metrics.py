# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from numbers import Real
import warnings
from typing import ( 
    Sequence, Optional,
    Union, Tuple, List, 
    Literal, Any, Dict
)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import to_rgba

import numpy as np
import pandas as pd 
from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from ..api.docs import DocstringComponents, _shared_metric_plot_params
from ..api.types import ( 
    MetricFunctionType,
    PlotKind, 
    MetricType, 
    PlotKindWIS, 
    PlotKindTheilU
)
from ..core.handlers import columns_manager
from ..core.io import _get_valid_kwargs 
from ..core.checks import exist_features
from ..core.diagnose_q import validate_quantiles
from ..utils.generic_utils import are_all_values_in_bounds 

__all__= [
     'plot_coverage',
     'plot_crps',
     'plot_mean_interval_width',
     'plot_prediction_stability',
     'plot_quantile_calibration',
     'plot_theils_u_score',
     'plot_time_weighted_metric',
     'plot_weighted_interval_score', 
     'plot_qce_donut', 
     'plot_radar_scores'
 ]

_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_shared_metric_plot_params), 
)

def plot_theils_u_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_values: Optional[Union[float, np.ndarray]] = None,
    metric_kws: Optional[Dict[str, Any]] = None,
    kind: PlotKindTheilU = 'summary_bar', 
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = "Theil's U Statistic",
    ylabel: Optional[str] = None,
    bar_color: Union[str, List[str]] = 'chocolate',
    bar_width: float = 0.8,
    score_annotation_format: str = "{:.4f}",
    reference_line_at_1: bool = True, 
    reference_line_props: Optional[Dict[str, Any]] = None,
    show_grid: bool = True,
    grid_props: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    verbose: int = 0,
    **kwargs: Any 
) -> plt.Axes:
    
    # *********************************************************
    from ..metrics._registry import get_metric
    theils_u_score = get_metric("theils_u_score")
    # *********************************************************
    # --- 1. Input Validation and Preparation ---
    # y_true, y_pred: (T,), (N,T), or (N,O,T)
    # Metric function handles detailed shape validation.
    # Here, we primarily pass them through.
    y_true_arr = check_array(
        y_true, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False
    )
    y_pred_arr = check_array(
        y_pred, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=False
    )

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    # Determine n_outputs for labeling if raw_values used
    # Based on the shape convention of theils_u_score's y_true_proc
    n_outputs = 1
    if y_true_arr.ndim == 3: # (N,O,T)
        n_outputs = y_true_arr.shape[1] # noqa
    elif y_true_arr.ndim == 1 and y_true_arr.shape[0] < 2 : # (T,) with T<2
         pass # Will be caught by metric
    elif y_true_arr.ndim == 2 and y_true_arr.shape[1] < 2: # (N,T) with T<2
         pass # will be caught by metric
         
    if y_true_arr.size == 0 or \
       (y_true_arr.ndim > 0 and y_true_arr.shape[-1] < 2):
        # Metric itself will raise error for T<2, but good to catch empty early
        warnings.warn(
            "Input data is empty or has fewer than 2 time steps. "
            "Cannot generate Theil's U plot."
        )
        if ax is None: _, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or "Theil's U Plot (No/Invalid Data)")
        if show_grid: ax.grid(**(grid_props or {}))
        return ax

    # --- Plotting Setup ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize) # type: ignore
    plot_title_str = title

    # --- Metric Calculation Handling ---
    current_metric_kws = metric_kws or {}
    default_kws_for_metric = {
        'nan_policy': 'propagate',
        'multioutput': 'uniform_average',
        'eps': 1e-8,
        'verbose': 0 # Metric's internal verbose
    }

    # --- Plotting Logic ---
    if kind == 'summary_bar':
        scores_to_plot: Union[float, np.ndarray]
        if metric_values is not None:
            scores_to_plot = metric_values
            if verbose > 0:
                print(f"Using pre-computed Theil's U: {scores_to_plot}")
        else:
            # For summary bar, respect user's multioutput choice
            summary_bar_default_kws = default_kws_for_metric.copy()
            if 'multioutput' not in current_metric_kws:
                 summary_bar_default_kws['multioutput'] = 'uniform_average'
            
            effective_kws = {**summary_bar_default_kws, **current_metric_kws}
            cleaned_kws = _get_valid_kwargs(
                theils_u_score, effective_kws
            )
            
            scores_to_plot = theils_u_score(
                y_true_arr, y_pred_arr, **cleaned_kws
            )
            if verbose > 0:
                print(f"Computed Theil's U for summary: {scores_to_plot}")
        
        scores_arr_bar: np.ndarray
        x_labels_bar: List[str]
        multioutput_used = (current_metric_kws or {}).get(
            'multioutput', default_kws_for_metric['multioutput'])

        if np.isscalar(scores_to_plot) or \
           (isinstance(scores_to_plot, np.ndarray) and \
            scores_to_plot.ndim == 0):
            scores_arr_bar = np.array([scores_to_plot])
            x_labels_bar = ["Theil's U"]
        elif isinstance(scores_to_plot, np.ndarray) and \
             scores_to_plot.ndim == 1:
            scores_arr_bar = scores_to_plot
            x_labels_bar = [f'Output {i}' for i in range(len(scores_arr_bar))]
            if plot_title_str and multioutput_used == 'raw_values':
                plot_title_str += " (Per Output)"
        else:
            raise TypeError(
                f"Unexpected type/shape for Theil's U scores: "
                f"{type(scores_to_plot)}"
            )

        bars = ax.bar(x_labels_bar, scores_arr_bar, color=bar_color,
                      width=bar_width, **kwargs.get('bar_kwargs', {}))
        ax.set_ylabel(ylabel or "Theil's U Statistic")
        
        # Reference line at U=1
        if reference_line_at_1:
            ref_line_defaults = {
                'color': 'black', 'linestyle': '--', 'linewidth': 1
            }
            ref_props = {**ref_line_defaults, **(reference_line_props or {})}
            ax.axhline(1, **ref_props) # type: ignore

        # Auto-adjust y-limits
        if scores_arr_bar.size > 0 and not np.all(np.isnan(scores_arr_bar)):
            min_val = np.nanmin(scores_arr_bar)
            max_val = np.nanmax(scores_arr_bar)
            # Ensure y=1 is visible if reference line is plotted
            plot_min_y = min(min_val, 0.8 if reference_line_at_1 else min_val)
            plot_max_y = max(max_val, 1.2 if reference_line_at_1 else max_val)
            
            padding = 0.1 * abs(plot_max_y - plot_min_y) \
                      if abs(plot_max_y - plot_min_y) > 1e-6 else 0.1
            ax.set_ylim(plot_min_y - padding, plot_max_y + padding + 0.05)


        for bar_obj in bars:
            yval = bar_obj.get_height()
            if not np.isnan(yval):
                x = bar_obj.get_x() + bar_obj.get_width() / 2.0
                dy = 3 if yval >= 0 else -10
                va = 'bottom' if yval >= 0 else 'top'
                ax.annotate(
                    score_annotation_format.format(yval),
                    xy=(x, yval),
                    xytext=(0, dy),
                    textcoords="offset points",
                    ha="center",
                    va=va,
                )

    else:
        # Currently, only 'summary_bar' is implemented for Theil's U.
        # Other kinds like distribution of squared errors could be added.
        raise ValueError(
            f"Unknown plot kind: '{kind}'. "
            "Currently only 'summary_bar' is supported for Theil's U."
        )

    if plot_title_str:
        ax.set_title(plot_title_str)
    
    if show_grid:
        current_grid_props = grid_props if grid_props is not None \
            else {'linestyle': ':', 'alpha': 0.7}
        ax.grid(**current_grid_props)
    else:
        ax.grid(False)

    return ax

plot_theils_u_score.__doc__=r""" 
Visualise Theil’s U statistic.
    
A single‑bar (or multi‑bar) summary plot that benchmarks a model’s
error against a naïve “last‑value’’ forecast.  
*U < 1* implies the model improves upon the naïve baseline;  
*U = 1* indicates parity;  
*U > 1* denotes under‑performance.

Parameters
----------
{params.base.y_true}
{params.base.y_pred}

metric_values : float or ndarray, optional  
    Pre‑computed Theil’s U statistic(s).  When supplied the helper
    skips internal evaluation and plots the given number(s) verbatim.
metric_kws : dict, optional  
    Additional keyword arguments forwarded to
    :func:`fusionlab.metrics.theils_u_score`
    (e.g. ``multioutput='raw_values'``).

kind : {{'summary_bar'}}, default ``'summary_bar'``  
    Currently only a bar‑chart summary is available.  Additional kinds
    may be added in future releases.

reference_line_at_1 : bool, default ``True``  
    Draw a horizontal reference line at *U = 1* to highlight the
    naïve‑benchmark threshold.
reference_line_props : dict, optional  
    Matplotlib style overrides for the reference line
    (colour, linestyle, linewidth …).

{params.base.figsize}
{params.base.title}
{params.base.ylabel}
{params.base.bar_color}
{params.base.bar_width}
{params.base.score_annotation_format}
{params.base.show_grid}
{params.base.grid_props}
{params.base.ax}
{params.base.verbose}
{params.base.kwargs}

Returns
-------
matplotlib.axes.Axes  
    The axes object with the rendered plot.

Notes
-----
For a univariate series the statistic is

.. math::

   U = \sqrt{{\frac{{\sum_{{t=2}}^{{T}} \bigl(y_t - \hat y_t\bigr)^2}}
                     {{\sum_{{t=2}}^{{T}} \bigl(y_t - y_{{t-1}}\bigr)^2}}}}

where :math:`y_{{t-1}}` is the naïve forecast.  
The helper calls :func:`fusionlab.metrics.theils_u_score` for the
computation.

Examples
--------
>>> import numpy as np, matplotlib.pyplot as plt
>>> from fusionlab.plot.evaluation import plot_theils_u_score
>>> rng = np.random.default_rng(0)
>>> y_true = rng.normal(size=100)
>>> y_pred = y_true + rng.normal(scale=.2, size=100)
>>> plot_theils_u_score(y_true, y_pred,
...                     bar_color='steelblue',
...                     figsize=(6, 4))
>>> plt.show()

See Also
--------
fusionlab.metrics.theils_u_score  
    Metric implementation.
fusionlab.plot.evaluation.plot_crps  
    Continuous Ranked Probability Score visualiser.
fusionlab.plot.evaluation.plot_wis  
    Weighted Interval Score plot.

References
----------
.. [1] H. Theil, *Applied Economic Forecasting*, North‑Holland, 1966.  
.. [2] Makridakis, Wheelwright & Hyndman,
       *Forecasting: Methods and Applications*, 3rd ed., 1998.
""".format(params=_param_docs)

def _calculate_per_sample_output_wis(
    y_true_so: np.ndarray, # Shape (N, O)
    y_median_so: np.ndarray, # Shape (N, O)
    y_lower_sok: np.ndarray, # Shape (N, O, K)
    y_upper_sok: np.ndarray, # Shape (N, O, K)
    alphas_k: np.ndarray, # Shape (K,)
    nan_policy: Literal['omit', 'propagate', 'raise'] = 'propagate',
    warn_invalid_bounds: bool = True,
    # Note: sample_weight is applied *after* these per-sample scores
    # when aggregating for the final metric value. Here we want raw per-sample.
) -> np.ndarray:
    """
    Calculate per-sample, per-output WIS (non-time-weighted).
    Returns array of shape (N, O).
    NaNs in inputs are handled based on nan_policy.
    """
    n_samples, n_outputs = y_true_so.shape
    K_intervals = alphas_k.shape[0]
    
    # Expand y_true_so, y_median_so for broadcasting with K: (N,O,1)
    y_t_exp_sok = y_true_so[..., np.newaxis]
    # y_m_exp_sok = y_median_so[..., np.newaxis] # Not directly used in IS part

    # Base NaN mask from y_true, y_median (N,O)
    nan_mask_base_so = np.isnan(y_true_so) | np.isnan(y_median_so)
    # NaN mask from bounds (N,O), True if any K for that S,O is NaN
    nan_mask_bounds_so = np.any(
        np.isnan(y_lower_sok) | np.isnan(y_upper_sok), axis=2
    )
    combined_nan_mask_so = nan_mask_base_so | nan_mask_bounds_so # (N,O)

    # Initialize WIS values per sample-output
    wis_values_so = np.full((n_samples, n_outputs), np.nan)

    # Calculate for valid (non-NaN according to policy) entries
    if nan_policy == 'raise' and np.any(combined_nan_mask_so):
        raise ValueError("NaNs found in inputs for per-sample WIS.")

    # For 'omit', we'd filter rows. For 'propagate', NaNs will flow.
    # This helper focuses on calculation; omit filtering happens before calling.
    # If called with data already filtered by 'omit', combined_nan_mask_so
    # for the passed data should be all False.
    mae_term_so = np.abs(y_median_so - y_true_so) # (N,O)

    if K_intervals > 0:
        interval_width_sok = y_upper_sok - y_lower_sok # (N,O,K)
        if warn_invalid_bounds and np.any(y_lower_sok > y_upper_sok):
            warnings.warn(
                "y_lower > y_upper found in inputs for per-sample WIS. "
                "Widths will be negative, affecting score.", UserWarning
            )
        
        alphas_exp_k = alphas_k.reshape(1, 1, -1) # (1,1,K)

        wis_sharp_sok = (alphas_exp_k / 2.0) * interval_width_sok
        wis_under_sok = (y_lower_sok - y_t_exp_sok) * \
                        (y_t_exp_sok < y_lower_sok)
        wis_over_sok = (y_t_exp_sok - y_upper_sok) * \
                       (y_t_exp_sok > y_upper_sok)
        
        sum_interval_comps_so = np.sum( # Sum over K
            wis_sharp_sok + wis_under_sok + wis_over_sok, axis=2
        ) # (N,O)
        wis_values_so = (mae_term_so + sum_interval_comps_so) / \
                        (K_intervals + 1.0)
    else: # K_intervals is 0, WIS is just MAE of median
        wis_values_so = mae_term_so

    if nan_policy == 'propagate':
        wis_values_so = np.where(
            combined_nan_mask_so, np.nan, wis_values_so
        )
    # If nan_policy='omit', NaNs should have been filtered before this helper.
    # If nan_policy='raise', error would have been raised.
    
    return wis_values_so

def plot_weighted_interval_score(
    y_true: np.ndarray,
    y_median: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alphas: np.ndarray,
    metric_values: Optional[Union[float, np.ndarray]] = None,
    metric_kws: Optional[Dict[str, Any]] = None,
    kind: PlotKindWIS = 'summary_bar',
    output_idx: Optional[int] = None,
    hist_bins: Union[int, Sequence[Real], str] = 'auto',
    hist_color: str = 'mediumseagreen',
    hist_edgecolor: str = 'black',
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = "Weighted Interval Score (WIS)",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    bar_color: Union[str, List[str]] = 'mediumseagreen',
    bar_width: float = 0.8,
    score_annotation_format: str = "{:.4f}",
    show_score_on_title: bool = True, 
    show_grid: bool = True,
    grid_props: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    verbose: int = 0,
    **kwargs: Any
) -> plt.Axes:

    # ****************************************************************
    from ..metrics._registry import get_metric
    weighted_interval_score = get_metric("weighted_interval_score")
    # ****************************************************************
    
    # --- 1. Input Validation and Preparation ---
    y_true_arr = check_array(
        y_true, ensure_2d=False, dtype="numeric",
        force_all_finite=False, copy=True)
    y_median_arr = check_array(
        y_median, ensure_2d=False, dtype="numeric",
        force_all_finite=False, copy=True)
    y_lower_arr = check_array(
        y_lower, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=True)
    y_upper_arr = check_array(
        y_upper, ensure_2d=False, allow_nd=True,
         dtype="numeric", force_all_finite=False, copy=True)
    alphas_arr = check_array(
        alphas, ensure_2d=False, dtype="numeric",
        force_all_finite=True)

    are_all_values_in_bounds(
        alphas_arr , bounds= (0, 1), nan_policy='raise', 
        message = "All alpha values must be in (0,1)."
    )
    
    if alphas_arr.ndim > 1: alphas_arr = alphas_arr.squeeze()
    if alphas_arr.ndim == 0: alphas_arr = alphas_arr.reshape(1,)
    K_intervals = alphas_arr.shape[0]

    # Reshape inputs for consistent processing:
    # y_true_proc, y_median_proc: (N, O)
    # y_lower_proc, y_upper_proc: (N, O, K)
    y_true_ndim_orig = y_true_arr.ndim
    if y_true_ndim_orig == 1: # (N,)
        y_true_proc = y_true_arr.reshape(-1, 1)
        y_median_proc = y_median_arr.reshape(-1, 1)
        if y_lower_arr.ndim == 2 and \
           y_lower_arr.shape[1] == K_intervals: # (N,K)
            y_lower_proc = y_lower_arr.reshape(y_lower_arr.shape[0], 1, -1)
            y_upper_proc = y_upper_arr.reshape(y_upper_arr.shape[0], 1, -1)
        else:
            raise ValueError("Shape mismatch for 1D y_true with bounds.")
    elif y_true_ndim_orig == 2: # (N,O)
        y_true_proc = y_true_arr
        y_median_proc = y_median_arr
        if y_lower_arr.ndim == 3 and \
           y_lower_arr.shape[1] == y_true_proc.shape[1] and \
           y_lower_arr.shape[2] == K_intervals: # (N,O,K)
            y_lower_proc, y_upper_proc = y_lower_arr, y_upper_arr
        else:
            raise ValueError("Shape mismatch for 2D y_true with bounds.")
    else:
        raise ValueError("y_true must be 1D or 2D.")

    # Check consistency for all processed shapes
    shapes_to_match = (y_true_proc.shape[0], y_true_proc.shape[1]) # (N,O)
    if not (y_median_proc.shape == shapes_to_match and \
            y_lower_proc.shape[:2] == shapes_to_match and \
            y_upper_proc.shape[:2] == shapes_to_match and \
            y_lower_proc.shape[2] == K_intervals and \
            y_upper_proc.shape[2] == K_intervals):
        raise ValueError("Processed input shapes are inconsistent.")

    n_samples, n_outputs = y_true_proc.shape

    if n_samples == 0:
        warnings.warn("Input arrays are empty. Cannot generate plot.")
        if ax is None: _, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or "WIS Plot (No Data)")
        if show_grid: ax.grid(**(grid_props or {}))
        return ax
    
    # --- Plotting Setup ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize) # type: ignore
    plot_title_str = title

    # --- Metric Calculation Handling ---
    current_metric_kws = metric_kws or {}
    default_kws_for_metric = {
        'nan_policy': 'propagate',
        'multioutput': 'uniform_average',
        'warn_invalid_bounds': True,
        'eps': 1e-8,
        'verbose': 0
    }
    
    # --- Plotting Logic ---
    if kind == 'scores_histogram':
        # Calculate per-sample, per-output WIS values for histogram
        # We need to handle NaNs before calling _calculate_per_sample_output_wis
        # if nan_policy is 'omit'.
        
        nan_policy_hist = current_metric_kws.get(
            'nan_policy', default_kws_for_metric['nan_policy']
        )
        warn_bounds_hist = current_metric_kws.get(
            'warn_invalid_bounds', default_kws_for_metric['warn_invalid_bounds']
        )

        y_t_hist, y_m_hist = y_true_proc, y_median_proc
        y_l_hist, y_u_hist = y_lower_proc, y_upper_proc
        s_weights_hist = (current_metric_kws or {}).get('sample_weight', None)

        # NaN mask for inputs to _calculate_per_sample_output_wis
        nan_mask_base_so_hist = np.isnan(y_t_hist) | np.isnan(y_m_hist)
        nan_mask_bounds_so_hist = np.any(
            np.isnan(y_l_hist) | np.isnan(y_u_hist), axis=2
        )
        combined_nan_mask_so_hist = nan_mask_base_so_hist | nan_mask_bounds_so_hist

        if np.any(combined_nan_mask_so_hist):
            if nan_policy_hist == 'raise':
                raise ValueError("NaNs found in inputs for histogram.")
            elif nan_policy_hist == 'omit':
                rows_with_nan = combined_nan_mask_so_hist.any(axis=1)
                rows_to_keep = ~rows_with_nan
                if not np.any(rows_to_keep):
                    ax.text(0.5,0.5,"All samples omitted due to NaNs.",
                            ha='center',va='center',transform=ax.transAxes)
                    if show_grid: ax.grid(**(grid_props or {}))
                    ax.set_title(plot_title_str or "WIS Scores (No Data)")
                    return ax
                y_t_hist = y_t_hist[rows_to_keep]
                y_m_hist = y_m_hist[rows_to_keep]
                y_l_hist = y_l_hist[rows_to_keep]
                y_u_hist = y_u_hist[rows_to_keep]
                if s_weights_hist is not None:
                    s_weights_hist = s_weights_hist[rows_to_keep]
            # If 'propagate', _calculate_per_sample_output_wis will handle it.
        
        if y_t_hist.shape[0] == 0: # All samples omitted
             ax.text(0.5,0.5,"No valid samples for WIS histogram.",
                     ha='center',va='center',transform=ax.transAxes)
             if show_grid: ax.grid(**(grid_props or {}))
             ax.set_title(plot_title_str or "WIS Scores (No Data)")
             return ax

        # Calculate raw per-sample-output WIS scores
        # Pass the nan_policy for _calculate to use internally for propagation
        per_so_wis_scores = _calculate_per_sample_output_wis(
            y_t_hist, y_m_hist, y_l_hist, y_u_hist, alphas_arr,
            nan_policy=nan_policy_hist, # This ensures NaNs propagate if needed
            warn_invalid_bounds=warn_bounds_hist
        ) # Shape (N_calc, O)

        # Select output for histogram
        scores_to_plot_hist: np.ndarray
        current_output_label = ""
        if n_outputs > 1:
            if output_idx is None:
                raise ValueError(
                    "For multi-output data and kind='scores_histogram', "
                    "'output_idx' must be specified."
                )
            if not (0 <= output_idx < n_outputs):
                raise ValueError(f"output_idx {output_idx} out of bounds.")
            scores_to_plot_hist = per_so_wis_scores[:, output_idx]
            current_output_label = f" (Output {output_idx})"
        else: # Single output
            scores_to_plot_hist = per_so_wis_scores.ravel()

        valid_scores_for_hist = scores_to_plot_hist[
            ~np.isnan(scores_to_plot_hist)
        ]

        if valid_scores_for_hist.size > 0:
            ax.hist(valid_scores_for_hist, bins=hist_bins,
                    color=hist_color, edgecolor=hist_edgecolor,
                    **kwargs.get('hist_kwargs', {}))
            
            if show_score_on_title: # Show mean of the *plotted* scores
                mean_of_plotted_wis = np.mean(valid_scores_for_hist)
                score_text = f"Mean WIS: {mean_of_plotted_wis:.4f}"
                current_title = plot_title_str or \
                                "Distribution of WIS Values"
                plot_title_str = (
                    f"{current_title}{current_output_label}\n({score_text})"
                )
        else:
            ax.text(0.5,0.5, "No valid WIS values for histogram.",
                    ha='center', va='center', transform=ax.transAxes)
            current_title = plot_title_str or \
                            "Distribution of WIS Values"
            plot_title_str = f"{current_title}{current_output_label} (No Data)"

        ax.set_xlabel(xlabel or 'WIS per Sample')
        ax.set_ylabel(ylabel or 'Frequency')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    elif kind == 'summary_bar':
        scores_to_plot_bar: Union[float, np.ndarray]
        if metric_values is not None:
            scores_to_plot_bar = metric_values
            if verbose > 0:
                print(f"Using pre-computed WIS values: {scores_to_plot_bar}")
        else:
            summary_bar_default_kws = default_kws_for_metric.copy()
            if 'multioutput' not in current_metric_kws:
                 summary_bar_default_kws['multioutput'] = 'uniform_average'
            
            effective_kws = {**summary_bar_default_kws, **current_metric_kws}
            cleaned_kws = _get_valid_kwargs(
                weighted_interval_score, effective_kws
            )
            
            scores_to_plot_bar = weighted_interval_score(
                y_true_arr, y_lower_arr, y_upper_arr, y_median_arr, alphas_arr,
                **cleaned_kws
            )
            if verbose > 0:
                print(f"Computed WIS for summary: {scores_to_plot_bar}")
        
        scores_arr_bar: np.ndarray
        x_labels_bar: List[str]
        multioutput_used = (current_metric_kws or {}).get(
            'multioutput', default_kws_for_metric['multioutput'])

        if np.isscalar(scores_to_plot_bar) or \
           (isinstance(scores_to_plot_bar, np.ndarray) and \
            scores_to_plot_bar.ndim == 0):
            scores_arr_bar = np.array([scores_to_plot_bar])
            x_labels_bar = ['Mean WIS']
        elif isinstance(scores_to_plot_bar, np.ndarray) and \
             scores_to_plot_bar.ndim == 1:
            scores_arr_bar = scores_to_plot_bar
            x_labels_bar = [f'Output {i}' for i in range(len(scores_arr_bar))]
            if plot_title_str and multioutput_used == 'raw_values':
                plot_title_str += " (Per Output)"
        else:
            raise TypeError(
                f"Unexpected type/shape for WIS scores: {type(scores_to_plot_bar)}"
            )

        bars = ax.bar(x_labels_bar, scores_arr_bar, color=bar_color,
                      width=bar_width, **kwargs.get('bar_kwargs', {}))
        ax.set_ylabel(ylabel or 'Weighted Interval Score (WIS)')
        
        if scores_arr_bar.size > 0 and not np.all(np.isnan(scores_arr_bar)):
            min_val = np.nanmin(scores_arr_bar)
            max_val = np.nanmax(scores_arr_bar)
            padding = 0.1 * abs(max_val - min_val) if abs(max_val-min_val)>1e-6 else 0.1
            ax.set_ylim(min(0, min_val - padding), max_val + padding + 0.05)

        for bar_obj in bars:
            yval = bar_obj.get_height()
            if not np.isnan(yval):
                x = bar_obj.get_x() + bar_obj.get_width() / 2.0
                # choose offset and vertical alignment based on sign
                offset = 3 if yval >= 0 else -10
                va = 'bottom' if yval >= 0 else 'top'
        
                ax.annotate(
                    score_annotation_format.format(yval),
                    xy=(x, yval),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va=va,
                )

    else:
        raise ValueError(
            f"Unknown plot kind: '{kind}'. Choose 'scores_histogram' "
            "or 'summary_bar'."
        )

    if plot_title_str:
        ax.set_title(plot_title_str)
    
    if show_grid:
        current_grid_props = grid_props if grid_props is not None \
            else {'linestyle': ':', 'alpha': 0.7}
        ax.grid(**current_grid_props)
    else:
        ax.grid(False)

    return ax

plot_weighted_interval_score.__doc__ =r"""
Visualise Weighted Interval Score (WIS).

WIS aggregates interval widths and coverage penalties across a set of
central prediction intervals, producing a proper scoring rule that
simultaneously rewards *sharpness* and *calibration* of probabilistic
forecasts [1]_.

The helper provides two complementary views:

* **'summary_bar'** – one bar per output (or a single bar for the
  uniform average).  
* **'scores_histogram'** – the distribution of per‑sample WIS values
  for a selected output.

Parameters
----------
{params.base.y_true}
y_median : ndarray
    Median (50 % quantile) forecast, shape compatible with
    ``y_true``.
{params.base.y_lower}
{params.base.y_upper}

alphas : ndarray of shape (K,)
    Alpha levels that define the nominal coverage of each prediction
    interval: :math:`\alpha_k = 1 - (q_{{k+1}} - q_k)`.  Must satisfy
    ``0 < α < 1`` and be strictly increasing.

metric_values : float or ndarray, optional
    Pre‑computed WIS value(s).  When supplied, plotting is performed
    without recalculating the metric.
metric_kws : dict, optional
    Extra keyword arguments forwarded to
    :func:`fusionlab.metrics.weighted_interval_score`
    (e.g. ``multioutput='raw_values'``).

kind : {{'summary_bar', 'scores_histogram'}}, default ``'summary_bar'``
    Style of visualisation.

output_idx : int, optional
    Index of the target variable to visualise when
    ``kind='scores_histogram'`` on multi‑output data.

hist_bins : int | sequence | str, default ``'auto'``
    Binning strategy for the histogram (passed to
    :func:`matplotlib.pyplot.hist`).

hist_color : str, default ``'mediumseagreen'``  
hist_edgecolor : str, default ``'black'``  
    Bar‑face and edge colours for the histogram.

{params.base.figsize}
title : str, optional
    Custom figure title.  If *None*, a context‑aware title is generated.
{params.base.xlabel}
{params.base.ylabel}
{params.base.bar_color}
{params.base.bar_width}
{params.base.score_annotation_format}

show_score_on_title : bool, default ``True``
    Append the mean WIS to the title when
    ``kind='scores_histogram'``.

{params.base.show_grid}
{params.base.grid_props}
{params.base.ax}
{params.base.verbose}
{params.base.kwargs}

Returns
-------
matplotlib.axes.Axes  
    The axes object containing the plot.

Notes
-----
The weighted interval score for a single observation and :math:`K`
central prediction intervals is

.. math::

   \mathrm{{WIS}} \;=\;
   \frac{{1}}{{K + 0.5}}\;\Bigl[
   \lvert y - \hat{{y}}_{{0.5}}\rvert\;+\;
   \sum_{{k=1}}^{{K}} \alpha_k
     \bigl\{{\, (y < l_k)\,(l_k - y)
              + (y > u_k)\,(y - u_k)
              + (u_k - l_k) \bigr\}}
   \Bigr],

where :math:`[l_k, u_k]` is the :math:`(1-\alpha_k)` central interval.
Lower WIS indicates a sharper, better‑calibrated forecast.

Examples
--------
>>> import numpy as np, matplotlib.pyplot as plt
>>> from fusionlab.plot.evaluation import plot_weighted_interval_score
>>> rng = np.random.default_rng(0)
>>> y_true   = rng.normal(size=100)
>>> y_med    = y_true + rng.normal(scale=.1, size=100)
>>> y_lower  = y_med - 1.0
>>> y_upper  = y_med + 1.0
>>> alphas   = np.array([0.2])
>>> plot_weighted_interval_score(y_true, y_med,
...                              y_lower, y_upper, alphas,
...                              kind='summary_bar',
...                              bar_color='slateblue')
>>> plt.show()

See Also
--------
fusionlab.metrics.weighted_interval_score  
    Numerical implementation of WIS.
fusionlab.plot.evaluation.plot_crps  
    Continuous Ranked Probability Score visualiser.
fusionlab.plot.evaluation.plot_theils_u_score  
    Deterministic relative‑skill bar plot.

References
----------
.. [1] Bracher, J. et al.  *Evaluating Probabilistic Forecasts with
       Scoring Rules.*  *arXiv preprint* arXiv:2101.05552, 2021.
""".format(params=_param_docs)

def _get_metric_function(
    metric_type: MetricType
) -> MetricFunctionType:
    """Helper to retrieve the appropriate metric function."""
    from ..metrics._registry import get_metric
    
    if metric_type == 'mae':
        return get_metric("time_weighted_mean_absolute_error")
    elif metric_type == 'accuracy':
        return get_metric("time_weighted_accuracy_score")
    elif metric_type == 'interval_score':
        return get_metric("time_weighted_interval_score")
    else:
        # This case should ideally be caught by Literal type hinting
        # or earlier validation.
        raise ValueError(f"Unknown metric_type: {metric_type}")

def _calculate_per_timestep_values(
    metric_type: MetricType,
    y_true_sot: np.ndarray, # Shape (N, O, T)
    y_pred_sot: Optional[np.ndarray] = None, # Shape (N, O, T)
    y_median_sot: Optional[np.ndarray] = None, # Shape (N, O, T)
    y_lower_sokt: Optional[np.ndarray] = None, # Shape (N, O, K, T)
    y_upper_sokt: Optional[np.ndarray] = None, # Shape (N, O, K, T)
    alphas_k: Optional[np.ndarray] = None, # Shape (K,)
    nan_policy: Literal['omit', 'propagate', 'raise'] = 'propagate', 
    verbose : int =0, 
) -> np.ndarray:
    """
    Calculate per-timestep, un-time-weighted metric values.
    Returns array of shape (N, O, T).
    NaNs in inputs are handled based on nan_policy.
    """
    n_samples, n_outputs, n_timesteps = y_true_sot.shape
    per_timestep_vals = np.full((n_samples, n_outputs, n_timesteps), np.nan)

    # Base NaN mask from y_true (N,O,T)
    nan_mask_base = np.isnan(y_true_sot)

    if metric_type == 'mae':
        if y_pred_sot is None:
            raise ValueError("y_pred is required for MAE.")
        nan_mask_pred = np.isnan(y_pred_sot)
        combined_nan_mask = nan_mask_base | nan_mask_pred
        
        abs_errors = np.abs(y_pred_sot - y_true_sot)
        if nan_policy == 'propagate':
            per_timestep_vals = np.where(combined_nan_mask, np.nan, abs_errors)
        elif nan_policy == 'omit': # Omit NaNs per S,O,T point for this calculation
            per_timestep_vals = np.where(combined_nan_mask, np.nan, abs_errors)
        elif nan_policy == 'raise' and np.any(combined_nan_mask):
            raise ValueError("NaNs found in inputs for per-timestep MAE.")
        else: # No NaNs or policy handled
            per_timestep_vals = abs_errors

    elif metric_type == 'accuracy':
        if y_pred_sot is None:
            raise ValueError("y_pred is required for accuracy.")
        nan_mask_pred = np.isnan(y_pred_sot)
        combined_nan_mask = nan_mask_base | nan_mask_pred

        correct_preds = (y_true_sot == y_pred_sot).astype(float)
        if nan_policy == 'propagate':
            per_timestep_vals = np.where(combined_nan_mask, np.nan, correct_preds)
        elif nan_policy == 'omit':
            per_timestep_vals = np.where(combined_nan_mask, np.nan, correct_preds)
        elif nan_policy == 'raise' and np.any(combined_nan_mask):
            raise ValueError("NaNs found for per-timestep accuracy.")
        else:
            per_timestep_vals = correct_preds

    elif metric_type == 'interval_score':
        if not all(v is not None for v in [
            y_median_sot, y_lower_sokt, y_upper_sokt, alphas_k
        ]):
            raise ValueError(
                "y_median, y_lower, y_upper, alphas are required for "
                "interval_score."
            )
        # Assert K_intervals > 0
        K_intervals = alphas_k.shape[0] # type: ignore
        if K_intervals == 0 and verbose > 0: # type: ignore
             warnings.warn("TWIS with K=0 intervals; effectively Time-Weighted MAE.")
             
        nan_mask_median = np.isnan(y_median_sot) # type: ignore
        nan_mask_bounds = np.any( # True if any K for that S,O,T is NaN
            np.isnan(y_lower_sokt) | np.isnan(y_upper_sokt), axis=2 # type: ignore
        ) # (N,O,T)
        combined_nan_mask = nan_mask_base | nan_mask_median | nan_mask_bounds

        # Calculate WIS_sot (non-time-weighted)
        mae_term_sot = np.abs(y_median_sot - y_true_sot) # (N,O,T) type: ignore

        # Expand y_true_sot for broadcasting with K: (N,O,1,T)
        y_t_exp_sokt = y_true_sot[..., np.newaxis, :]
        y_t_exp_sokt = np.swapaxes(y_t_exp_sokt, 2, 3) # (N,O,T,1)
        
        # Reshape alphas for broadcasting: (1,1,K,1)
        alphas_exp_k = alphas_k.reshape(1, 1, -1, 1) # type: ignore

        # Interval components: (N,O,K,T)
        interval_width_sokt = y_upper_sokt - y_lower_sokt # type: ignore
        
        wis_sharp_sokt = (alphas_exp_k / 2.0) * interval_width_sokt
        wis_under_sokt = (y_lower_sokt - y_t_exp_sokt) * \
                         (y_t_exp_sokt < y_lower_sokt) # type: ignore
        wis_over_sokt = (y_t_exp_sokt - y_upper_sokt) * \
                        (y_t_exp_sokt > y_upper_sokt) # type: ignore
        
        sum_interval_wis_comps_sot = np.sum( # Sum over K
            wis_sharp_sokt + wis_under_sokt + wis_over_sokt, axis=2
        ) # (N,O,T)

        wis_sot = (mae_term_sot + sum_interval_wis_comps_sot) / \
                  (K_intervals + 1.0) if K_intervals > 0 else mae_term_sot

        if nan_policy == 'propagate':
            per_timestep_vals = np.where(combined_nan_mask, np.nan, wis_sot)
        elif nan_policy == 'omit':
            per_timestep_vals = np.where(combined_nan_mask, np.nan, wis_sot)
        elif nan_policy == 'raise' and np.any(combined_nan_mask):
            raise ValueError("NaNs found for per-timestep interval score.")
        else:
            per_timestep_vals = wis_sot
    else:
        raise ValueError(f"Unsupported metric_type for per-timestep: {metric_type}")

    return per_timestep_vals


def plot_time_weighted_metric(
    metric_type: MetricType,
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    y_median: Optional[np.ndarray] = None,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    alphas: Optional[np.ndarray] = None,
    time_weights: Optional[Union[Sequence[float], str]] = 'inverse_time',
    metric_values: Optional[Union[float, np.ndarray]] = None,
    metric_kws: Optional[Dict[str, Any]] = None,
    kind: PlotKind = 'summary_bar',
    output_idx: Optional[int] = None,
    sample_idx: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    profile_line_color: str = 'royalblue',
    profile_line_style: str = '-',
    profile_marker: Optional[str] = 'o',
    time_weights_color: str = 'gray',
    show_time_weights_on_profile: bool = False,
    bar_color: Union[str, List[str]] = 'royalblue',
    bar_width: float = 0.8,
    score_annotation_format: str = "{:.4f}",
    show_score_on_title: bool = True,
    show_grid: bool = True,
    grid_props: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    verbose: int = 0,
    **kwargs: Any
) -> plt.Axes:
    
    # --- 1. Input Validation and Metric Function Selection ---
    metric_func = _get_metric_function(metric_type)
    
    # Basic validation for y_true
    y_true_arr = check_array(y_true, ensure_2d=False, allow_nd=True,
                             dtype="numeric" if metric_type != 'accuracy' else None,
                             force_all_finite=False, copy=True)

    # Prepare metric_kws, ensuring plot's time_weights is used
    current_metric_kws = (metric_kws or {}).copy()
    current_metric_kws['time_weights'] = time_weights # Override/set
    
    default_overall_metric_kws = {
        'nan_policy': 'propagate',
        'multioutput': 'uniform_average',
        'eps': 1e-8,
        'verbose': 0 # Metric's internal verbose
    }
    # For overall score calculation if metric_values is None
    overall_score_kws = {
        **default_overall_metric_kws,
        **current_metric_kws
    }
    # For per-timestep value calculation (nan_policy and eps are relevant)
    per_timestep_nan_policy = overall_score_kws.get(
        'nan_policy', 'propagate'
    ) # type: ignore
    # per_timestep_eps = overall_score_kws.get('eps', 1e-8) # type: ignore

    # --- 2. Reshape Inputs to Standard (N, O, T) or (N, O, K, T) ---
    # This part needs to be robust for different y_true_arr.ndim
    # and corresponding prediction shapes.
    y_true_ndim_orig = y_true_arr.ndim
    if y_true_ndim_orig == 1: # (T,) -> (1,1,T)
        y_true_proc = y_true_arr.reshape(1,1,-1)
        n_samples, n_outputs, n_timesteps = 1, 1, y_true_arr.shape[0]
    elif y_true_ndim_orig == 2: # (N,T) -> (N,1,T)
        y_true_proc = y_true_arr.reshape(y_true_arr.shape[0], 1, -1)
        n_samples, n_outputs, n_timesteps =  ( 
            y_true_arr.shape[0], 1, y_true_arr.shape[1]
            )
    elif y_true_ndim_orig == 3: # (N,O,T)
        y_true_proc = y_true_arr
        n_samples, n_outputs, n_timesteps = y_true_proc.shape
    else:
        raise ValueError("y_true must be 1D, 2D, or 3D.")

    # Process prediction arrays based on metric_type
    y_pred_proc = y_median_proc = y_lower_proc = y_upper_proc = None
    alphas_proc = None

    if metric_type in ['mae', 'accuracy']:
        if y_pred is None: 
            raise ValueError("y_pred is required for MAE/Accuracy.")
            
        y_p = check_array(
            y_pred, ensure_2d=False, allow_nd=True,
            dtype=y_true_proc.dtype, 
            force_all_finite=False
            )
        if y_p.shape != y_true_arr.shape: 
            raise ValueError("Shape mismatch: y_true vs y_pred.")
        y_pred_proc = y_p.reshape(n_samples, n_outputs, n_timesteps)

    elif metric_type == 'interval_score':
        if not all(
                v is not None for v in [y_median, y_lower, y_upper, alphas]):
            raise ValueError(
                "y_median, y_lower, y_upper,"
                " alphas required for interval_score.")
        
        y_m = check_array(
            y_median, ensure_2d=False, 
            allow_nd=True, dtype="numeric", 
            force_all_finite=False) # type: ignore
        y_l = check_array(y_lower, 
                          ensure_2d=False, allow_nd=True, 
                          dtype="numeric", 
                          force_all_finite=False) # type: ignore
        y_u = check_array(
            y_upper, ensure_2d=False, 
            allow_nd=True, 
            dtype="numeric", force_all_finite=False
            ) # type: ignore
        alphas_proc = check_array(
            alphas, ensure_2d=False, 
            dtype="numeric", 
            force_all_finite=True
            ) # type: ignore
        if alphas_proc.ndim > 1:
            alphas_proc = alphas_proc.squeeze()
        if alphas_proc.ndim == 0: 
            alphas_proc = alphas_proc.reshape(1,)
        K_intervals = alphas_proc.shape[0]

        if y_m.shape != y_true_arr.shape:
            raise ValueError("Shape mismatch: y_true vs y_median.")
        y_median_proc = y_m.reshape(
            n_samples, n_outputs, n_timesteps)
        
        # Expected y_lower/upper: (N,O,K,T) or compatible
        expected_bounds_shape_prefix = (
            n_samples, n_outputs, K_intervals)
        if ( 
                y_l.ndim == 2 and K_intervals==y_l.shape[0]
                and n_timesteps==y_l.shape[1] and n_samples==1 
                and n_outputs==1
            ): #(K,T)
            y_lower_proc = y_l.reshape(1,1,K_intervals,n_timesteps)
            y_upper_proc = y_u.reshape(1,1,K_intervals,n_timesteps)
        elif ( 
                y_l.ndim == 3 and y_l.shape[:1]==(n_samples,) 
                and y_l.shape[1]==K_intervals 
                and n_outputs==1
            ): #(N,K,T)
            y_lower_proc = y_l.reshape(n_samples,1,K_intervals,n_timesteps)
            y_upper_proc = y_u.reshape(n_samples,1,K_intervals,n_timesteps)
        elif y_l.ndim == 4 and y_l.shape[:3] == expected_bounds_shape_prefix : # (N,O,K,T)
            y_lower_proc, y_upper_proc = y_l, y_u
        else: 
            raise ValueError(
                f"y_lower/y_upper shape incompatible. Expected"
                " compatible with (N,O,K,T)="
                f"{(n_samples,n_outputs,K_intervals,n_timesteps)},"
                f" got {y_l.shape}")
        if ( 
                y_lower_proc.shape[3] != n_timesteps 
                or y_upper_proc.shape[3] != n_timesteps 
            ):
            raise ValueError("Timestep dimension mismatch in bounds.")


    if n_samples == 0 or n_timesteps == 0: # Handled after reshaping for clarity
        warnings.warn("Effective data is empty. Cannot generate plot.")
        if ax is None: _, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or f"{metric_type.upper()} Plot (No Data)")
        if show_grid: ax.grid(**(grid_props or {}))
        return ax

    # --- Plotting Setup ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize) # type: ignore
    plot_title_str = title if title is not None else \
        f"Time-Weighted {metric_type.replace('_',' ').title()}"
    
    # --- Plotting Logic ---
    if kind == 'time_profile':
        if n_timesteps < 1 and metric_type != 'interval_score': # IS can have K=0
             warnings.warn("Need at least 1 timestep for time_profile plot.")
             # Fallback to empty plot
             ax.set_title(plot_title_str + " (Not Enough Data)")
             if show_grid: ax.grid(**(grid_props or {}))
             return ax

        per_timestep_sot = _calculate_per_timestep_values(
            metric_type, y_true_proc, y_pred_proc, y_median_proc,
            y_lower_proc, y_upper_proc, alphas_proc,
            nan_policy=per_timestep_nan_policy
        ) # (N,O,T)

        # Aggregate over samples if sample_idx is None
        profile_data_ot: np.ndarray # Shape (O,T)
        sample_weights_for_avg = (current_metric_kws or {}).get('sample_weight', None)

        if sample_idx is not None:
            if not (0 <= sample_idx < n_samples):
                raise ValueError(f"sample_idx {sample_idx} out of bounds.")
            profile_data_ot = per_timestep_sot[sample_idx, :, :] # (O,T)
            plot_title_str += f" (Sample {sample_idx})"
        else: # Average over samples
            if sample_weights_for_avg is not None:
                s_w = check_array(sample_weights_for_avg, ensure_2d=False,
                                  dtype="numeric", force_all_finite=True)
                check_consistent_length(per_timestep_sot, s_w)
                if np.sum(s_w) < default_overall_metric_kws['eps']: # type: ignore
                    profile_data_ot = np.full((n_outputs, n_timesteps), np.nan)
                else: # Weighted average, careful with NaNs in per_timestep_sot
                    profile_data_ot = np.ma.average(
                        np.ma.masked_invalid(per_timestep_sot), # type: ignore
                        axis=0, weights=s_w
                    )
                    if isinstance(profile_data_ot, np.ma.MaskedArray):
                        profile_data_ot = profile_data_ot.filled(np.nan)
            else:
                profile_data_ot = np.nanmean(per_timestep_sot, axis=0)
        
        # Select output
        profile_to_plot_t: np.ndarray # Shape (T,)
        current_output_label = ""
        output_idx_to_use = 0
        if n_outputs > 1:
            if output_idx is None:
                warnings.warn(
                    "Multi-output data for time_profile without specified "
                    "'output_idx'. Plotting first output or average if applicable."
                )
                # Default to first output or average if appropriate for metric
                # For now, let's default to first output for profile plot
                profile_to_plot_t = profile_data_ot[0, :]
                current_output_label = " (Output 0)"
            elif not (0 <= output_idx < n_outputs):
                raise ValueError(f"output_idx {output_idx} out of bounds.")
            else:
                profile_to_plot_t = profile_data_ot[output_idx, :]
                current_output_label = f" (Output {output_idx})"
                output_idx_to_use = output_idx
        else: # Single output
            profile_to_plot_t = profile_data_ot.ravel() # Should be (1,T) -> (T,)

        time_steps_x = np.arange(n_timesteps)
        ax.plot(time_steps_x, profile_to_plot_t,
                color=profile_line_color, linestyle=profile_line_style,
                marker=profile_marker if profile_marker else '',
                label=f"{metric_type.upper()} Profile",
                **kwargs.get('plot_kwargs', {}))

        ax.set_xlabel(xlabel or "Time Step")
        ax.set_ylabel(ylabel or f"Per-Timestep {metric_type.upper()}")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if show_time_weights_on_profile and n_timesteps > 0:
            # Process actual time_weights for plotting
            w_t_plot: np.ndarray
            if time_weights is None:
                w_t_plot = np.full(n_timesteps, 1.0/n_timesteps if n_timesteps > 0 else 0)
            elif isinstance(time_weights, str) and time_weights == 'inverse_time':
                 if n_timesteps == 0: w_t_plot = np.array([])
                 else:
                    w_raw_plot = 1./np.arange(1,n_timesteps+1)
                    sum_w_plot = np.sum(w_raw_plot)
                    w_t_plot = w_raw_plot/(
                        sum_w_plot 
                        if sum_w_plot > default_overall_metric_kws['eps'] 
                        else 1) # type: ignore
            else:
                w_t_plot = check_array(
                    time_weights,ensure_2d=False,
                    dtype="numeric",
                    force_all_finite=True
                    ) # type: ignore
                if w_t_plot.shape[0]!=n_timesteps: raise ValueError(
                        "time_weights length mismatch for plot.")
                sum_w_t_plot = np.sum(w_t_plot)
                if sum_w_t_plot < default_overall_metric_kws['eps']: # type: ignore
                     w_t_plot = np.zeros(
                         n_timesteps) if not np.any(
                             w_t_plot!=0) else w_t_plot
                else: w_t_plot = w_t_plot / sum_w_t_plot

            if w_t_plot.size == n_timesteps : # Ensure it was processed correctly
                ax2 = ax.twinx()
                ax2.bar(time_steps_x, w_t_plot, alpha=0.3, width=0.8,
                        color=time_weights_color, label='Time Weights')
                ax2.set_ylabel('Time Weight', color=time_weights_color)
                ax2.tick_params(axis='y', labelcolor=time_weights_color)
                # Ensure legend includes items from both axes
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='best')


        if show_score_on_title:
            score_for_title: Optional[Union[float, np.ndarray]] = None
            if metric_values is not None: # Use pre-computed
                 # If metric_values is array (raw multioutput), select the one for title
                if isinstance(metric_values, np.ndarray) and n_outputs > 1:
                    score_for_title =( 
                        metric_values[output_idx_to_use] 
                        if output_idx is not None 
                        and output_idx_to_use < len(metric_values) 
                        else np.nanmean(metric_values)
                    )
                else: score_for_title = metric_values

            else: # Calculate overall score for the plotted output/average
                title_score_kws = {**overall_score_kws}
                # For title, if specific output/sample plotted, score that.
                # If averaged profile, then overall score.
                # This can get complex. Simplest: show overall score from metric_kws.
                if n_outputs > 1 and output_idx is not None:
                    title_score_kws['multioutput'] = 'raw_values'
                else: # Single output or averaged profile
                    title_score_kws['multioutput'] = 'uniform_average'
                
                cleaned_title_kws = _get_valid_kwargs(metric_func, title_score_kws)
                
                # Prepare inputs for metric_func based on metric_type
                metric_inputs = {'y_true': y_true_arr}
                if metric_type in ['mae', 'accuracy']: metric_inputs['y_pred'] = y_pred # type: ignore
                elif metric_type == 'interval_score':
                    metric_inputs.update({
                        'y_median': y_median, 'y_lower': y_lower, # type: ignore
                        'y_upper': y_upper, 'alphas': alphas # type: ignore
                    })
                
                try:
                    calculated_scores = metric_func(**metric_inputs, **cleaned_title_kws)
                    if ( 
                            isinstance(calculated_scores, np.ndarray)
                            and n_outputs > 1 
                            and output_idx is not None
                        ):
                        score_for_title = calculated_scores[output_idx_to_use]
                    else:
                        score_for_title = float(np.ravel(calculated_scores)[0]) # type: ignore
                except Exception as e:
                    warnings.warn(f"Could not calculate score for title: {e}")
            
            if ( 
                    score_for_title is not None 
                    and not np.isnan(score_for_title)
                ): # type: ignore
                score_text = f"Overall Score: {score_for_title:.4f}" 
                plot_title_str = ( 
                    f"{plot_title_str}{current_output_label}\n({score_text})"
                    )


    elif kind == 'summary_bar':
        scores_to_plot: Union[float, np.ndarray]
        if metric_values is not None:
            scores_to_plot = metric_values
            if verbose > 0: print(f"Using pre-computed scores: {scores_to_plot}")
        else:
            summary_bar_kws = {**overall_score_kws, **current_metric_kws}
            # Respect user's multioutput for summary bar
            if 'multioutput' not in current_metric_kws:
                 summary_bar_kws['multioutput'] = 'uniform_average'
            
            cleaned_kws = _get_valid_kwargs(metric_func, summary_bar_kws)
            
            metric_inputs = {'y_true': y_true_arr}
            if metric_type == 'mae': metric_inputs['y_pred'] = y_pred # type: ignore
            elif metric_type == 'accuracy': metric_inputs['y_pred'] = y_pred # type: ignore
            elif metric_type == 'interval_score':
                metric_inputs.update({
                    'y_median': y_median, 'y_lower': y_lower, # type: ignore
                    'y_upper': y_upper, 'alphas': alphas # type: ignore
                })
            scores_to_plot = metric_func(**metric_inputs, **cleaned_kws)
            if verbose > 0: print(f"Computed scores for summary: {scores_to_plot}")
        
        scores_arr_bar: np.ndarray
        x_labels_bar: List[str]
        multioutput_used = (current_metric_kws or {}).get(
            'multioutput', default_overall_metric_kws['multioutput'])

        if np.isscalar(scores_to_plot) or \
           (isinstance(scores_to_plot, np.ndarray) and scores_to_plot.ndim == 0):
            scores_arr_bar = np.array([scores_to_plot])
            x_labels_bar = [f"Overall {metric_type.upper()}"]
        elif isinstance(scores_to_plot, np.ndarray) and scores_to_plot.ndim == 1:
            scores_arr_bar = scores_to_plot
            x_labels_bar = [f'Output {i}' for i in range(len(scores_arr_bar))]
            if plot_title_str and multioutput_used == 'raw_values':
                plot_title_str += " (Per Output)"
        else:
            raise TypeError(f"Unexpected scores type/shape: {type(scores_to_plot)}")

        bars = ax.bar(x_labels_bar, scores_arr_bar, color=bar_color,
                      width=bar_width, **kwargs.get('bar_kwargs', {}))
        ax.set_ylabel(ylabel or f"{metric_type.upper()} Score")
        
        if scores_arr_bar.size > 0 and not np.all(np.isnan(scores_arr_bar)):
            min_val = np.nanmin(scores_arr_bar)
            max_val = np.nanmax(scores_arr_bar)
            padding = 0.1 * abs(max_val - min_val) if abs(max_val - min_val) > 1e-6 else 0.1
            # Adjust y_lim based on metric type (accuracy vs error)
            if metric_type == 'accuracy':
                ax.set_ylim(max(0, min_val - padding), min(1, max_val + padding + 0.05))
            else: # MAE, Interval Score (errors, can be >1 or <0 for IS)
                ax.set_ylim(min_val - padding if min_val < 0 else 0,
                            max_val + padding + 0.05)


        for bar_obj in bars:
            yval = bar_obj.get_height()
            if not np.isnan(yval):
                x = bar_obj.get_x() + bar_obj.get_width() / 2.0
                offset_y = 3 if yval >= 0 else -10
                va = 'bottom' if yval >= 0 else 'top'
                ax.annotate(
                    score_annotation_format.format(yval),
                    xy=(x, yval),
                    xytext=(0, offset_y),
                    textcoords="offset points",
                    ha="center",
                    va=va,
                )
    else:
        raise ValueError(f"Unknown plot kind: '{kind}'.")

    if plot_title_str:
        ax.set_title(plot_title_str)
    
    if show_grid:
        current_grid_props = grid_props if grid_props is not None \
            else {'linestyle': ':', 'alpha': 0.7}
        ax.grid(**current_grid_props)
    else:
        ax.grid(False)

    return ax

plot_time_weighted_metric.__doc__=r"""
Visualise time‑weighted error / accuracy metrics (MAE, classification
accuracy, or interval‑based scores) as either

* a **summary bar** of the overall time‑weighted score, or one bar per
  output dimension; or
* a **time‑profile** curve that shows how the metric evolves over the
  forecasting horizon, optionally overlaid with the weight
  distribution.

The helper delegates numeric computation to the corresponding
metric in :pymod:`fusionlab.metrics` and applies the chosen *time
weights* before visualisation.

Parameters
----------
metric_type : {{'mae', 'accuracy', 'interval_score'}}
    Which metric to compute and plot.
    * ``'mae'`` – Mean Absolute Error.  
    * ``'accuracy'`` – Classification accuracy.  
    * ``'interval_score'`` – Weighted interval score
      (requires median, bounds, and ``alphas``).

{params.base.y_true}
{params.base.y_pred}
{params.base.y_median}
{params.base.y_lower}
{params.base.y_upper}
{params.base.alphas}

time_weights : 1‑D sequence, ``'inverse_time'`` or ``None``,\
    default ``'inverse_time'``
    * **array‑like** – explicit non‑negative weights for each
      timestep *(length T)*.  They are automatically normalised to sum
      to 1.  
    * ``'inverse_time'`` – use
      :math:`w_t \propto 1 / (t + 1)` (early timesteps matter more).  
    * ``None`` – uniform weights *(1/T)*.

metric_values : float or ndarray, optional
    Pre‑computed time‑weighted score(s) to plot, bypassing internal
    metric evaluation.
{params.base.metric_kws}

kind : {{'summary_bar', 'time_profile'}}, default ``'summary_bar'``
    * **summary_bar** – bar plot of the overall score.  
    * **time_profile** – line plot of the per‑timestep metric
      (averaged over samples), optionally with the weight profile.

output_idx : int, optional
    Output dimension to plot when the data are multi‑output and
    ``kind='time_profile'``.
{params.base.sample_idx}

figsize, title, xlabel, ylabel
    {{see shared parameters below}}

Time‑profile styling
^^^^^^^^^^^^^^^^^^^^
profile_line_color : str, default ``'royalblue'``  
profile_line_style : str, default ``'-'``  
profile_marker : str or ``None``, default ``'o'``  
    Matplotlib properties for the metric curve.

time_weights_color : str, default ``'gray'``  
show_time_weights_on_profile : bool, default ``False``  
    If *True*, draws a semi‑transparent bar chart of the
    normalised weights on a secondary y‑axis.

Summary‑bar styling
^^^^^^^^^^^^^^^^^^^
{params.base.bar_color}
{params.base.bar_width}
{params.base.score_annotation_format}

Common plot controls
^^^^^^^^^^^^^^^^^^^^
{params.base.figsize}
{params.base.show_score_on_title}
{params.base.show_grid}
{params.base.grid_props}
{params.base.ax}
{params.base.verbose}
{params.base.kwargs}

Returns
-------
matplotlib.axes.Axes  
    The axes object with the rendered figure.

Notes
-----
Let :math:`w_t` be the *normalised* time weight for horizon *t*.
For MAE the time‑weighted score is

.. math::

   \text{{TW‑MAE}} \;=\; \sum_{{t=1}}^{{T}} w_t \,
   \lvert y_t - \hat y_t\rvert .

Analogous definitions apply for accuracy (with the 0‑1 loss) and
for the weighted interval score (using the per‑timestep WIS).

If *``kind='time_profile'``* the helper first computes the unweighted
metric value for each timestep, then applies the ``time_weights`` when
plotting or when aggregating to a single title score.

Examples
--------
>>> import numpy as np, matplotlib.pyplot as plt
>>> from fusionlab.plot.evaluation import plot_time_weighted_metric
>>> T = 24
>>> y_true = np.sin(np.linspace(0, 3*np.pi, T))
>>> y_pred = y_true + np.random.normal(0, 0.1, T)
>>> ax = plot_time_weighted_metric(
...     metric_type='mae',
...     y_true=y_true,
...     y_pred=y_pred,
...     kind='time_profile',
...     show_time_weights_on_profile=True,
...     figsize=(8, 4))
>>> plt.show()

See Also
--------
fusionlab.metrics.time_weighted_mae  
fusionlab.metrics.time_weighted_accuracy  
fusionlab.metrics.time_weighted_interval_score  
fusionlab.plot.evaluation.plot_weighted_interval_score

References
----------
.. [1] Tay, F.E.H., *et al.* “Application of Weighted Metrics in Time‑
       Series Forecast Evaluation,” *International Journal of Forecasting*,
       vol 35, 2019.
""".format(params=_param_docs)

def plot_quantile_calibration(
    y_true: np.ndarray,
    y_pred_quantiles: np.ndarray,
    quantiles: np.ndarray,
    qce_values: Optional[Union[float, np.ndarray]] = None,
    metric_kws: Optional[Dict[str, Any]] = None,
    kind: Literal['reliability_diagram', 'summary_bar'] = 'reliability_diagram',
    output_idx: Optional[int] = None, 
    perfect_calib_color: str = 'red',
    observed_prop_color: str = 'blue',
    observed_prop_marker: str = 'o',
    figsize: Tuple[float, float] = (8, 8), 
    title: Optional[str] = "Quantile Calibration Error (QCE)",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    bar_color: Union[str, List[str]] = 'darkcyan',
    bar_width: float = 0.8,
    score_annotation_format: str = "{:.4f}",
    show_score: bool = True,
    show_grid: bool = True,
    grid_props: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    verbose: int = 0,
    **kwargs: Any 
) -> plt.Axes:
    
    # *************************************************************************
    from ..metrics._registry import get_metric
    quantile_calibration_error = get_metric("quantile_calibration_error")
    # **************************************************************************
    
    # --- Input Validation and Preparation ---
    # y_true: (N,), (N,O)
    # y_pred_quantiles: (N,Q) or (N,O,Q)
    # quantiles: (Q,)
    y_true_arr = check_array(
        y_true, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=True
    )
    y_pred_q_arr = check_array(
        y_pred_quantiles, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=True
    )
    q_arr = check_array(
        quantiles, ensure_2d=False, dtype="numeric",
        force_all_finite=True # Quantiles must be finite and in (0,1)
    )
    are_all_values_in_bounds(
        q_arr , bounds= (0, 1), nan_policy='raise', 
        message = "All quantile values must be in (0,1)."
    )
    if q_arr.ndim > 1: q_arr = q_arr.squeeze()
    if q_arr.ndim == 0: q_arr = q_arr.reshape(1,)
    n_quantiles = q_arr.shape[0]

    # Reshape inputs for consistent processing:
    # y_true_proc: (N, O)
    # y_pred_proc: (N, O, Q)
    y_true_ndim_orig = y_true_arr.ndim
    if y_true_ndim_orig == 1: # (N,)
        y_true_proc = y_true_arr.reshape(-1, 1) # (N,1)
        if y_pred_q_arr.ndim == 2 and \
           y_pred_q_arr.shape[1] == n_quantiles: # (N,Q)
            y_pred_proc = y_pred_q_arr.reshape(
                y_pred_q_arr.shape[0], 1, -1 # (N,1,Q)
            )
        else:
            raise ValueError(
                "If y_true is 1D, y_pred_quantiles must be 2D (N,Q)."
            )
    elif y_true_ndim_orig == 2: # (N,O)
        y_true_proc = y_true_arr
        if y_pred_q_arr.ndim == 3 and \
           y_pred_q_arr.shape[1] == y_true_proc.shape[1] and \
           y_pred_q_arr.shape[2] == n_quantiles: # (N,O,Q)
            y_pred_proc = y_pred_q_arr
        else:
            raise ValueError(
                "If y_true is 2D (N,O), y_pred_quantiles must be 3D (N,O,Q)."
            )
    else:
        raise ValueError("y_true must be 1D or 2D.")

    if y_true_proc.shape[0] != y_pred_proc.shape[0]: # Samples mismatch
        raise ValueError("y_true and y_pred_quantiles n_samples mismatch.")

    n_samples, n_outputs, _ = y_pred_proc.shape

    if n_samples == 0:
        warnings.warn("Input arrays are empty. Cannot generate plot.")
        if ax is None: _, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or "QCE Plot (No Data)")
        if show_grid: ax.grid(**(grid_props or {}))
        return ax
    if n_quantiles == 0 and kind != 'summary_bar':
        warnings.warn("No quantiles provided. Cannot generate plot.")
        if ax is None: _, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or f"QCE {kind} (No Quantiles)")
        if show_grid: ax.grid(**(grid_props or {}))
        return ax

    # --- Plotting Setup ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize) # type: ignore

    plot_title_str = title # Use a mutable string

    # --- Metric Calculation Handling ---
    current_metric_kws = metric_kws or {}
    default_kws_for_metric = {
        'nan_policy': 'propagate',
        'multioutput': 'uniform_average',
        'eps': 1e-8,
        'verbose': 0
    }

    # --- Plotting Logic ---
    if kind == 'reliability_diagram':
        if n_outputs > 1 and output_idx is None:
            raise ValueError(
                "For multi-output data and kind='reliability_diagram', "
                "'output_idx' must be specified."
            )
        if output_idx is not None and not (0 <= output_idx < n_outputs):
            raise ValueError(f"output_idx {output_idx} out of bounds.")
        
        # Select data for the specific output
        current_output_to_plot = output_idx if output_idx is not None else 0
        
        y_t_plot = y_true_proc[:, current_output_to_plot]
        y_p_plot = y_pred_proc[:, current_output_to_plot, :]

        # Calculate observed proportions
        # Handle NaNs based on metric_kws for calculating proportions
        nan_policy_plot = current_metric_kws.get(
            'nan_policy', default_kws_for_metric['nan_policy']
        )
        sample_weight_plot = current_metric_kws.get('sample_weight', None)
        eps_plot = current_metric_kws.get('eps', default_kws_for_metric['eps'])

        # Create indicators: (N_samples, N_quantiles)
        indicators = (
            y_t_plot[:, np.newaxis] <= y_p_plot
        ).astype(float)

        nan_mask_yt_exp = np.isnan(y_t_plot[:, np.newaxis])
        nan_mask_yp = np.isnan(y_p_plot)
        nan_mask_sq = nan_mask_yt_exp | nan_mask_yp # (N_samples, N_quantiles)

        if np.any(nan_mask_sq):
            if nan_policy_plot == 'raise':
                raise ValueError("NaNs found in data for reliability diagram.")
            elif nan_policy_plot == 'omit':
                # Omit samples if *any* of their quantiles or true value is NaN
                rows_with_nan = nan_mask_sq.any(axis=1) # (N_samples,)
                rows_to_keep = ~rows_with_nan
                if not np.any(rows_to_keep):
                    ax.text(0.5,0.5,"All samples omitted due to NaNs.",
                            ha='center',va='center',transform=ax.transAxes)
                    if show_grid: ax.grid(**(grid_props or {}))
                    ax.set_title(plot_title_str or "Reliability Diagram (No Data)")
                    return ax
                indicators = indicators[rows_to_keep]
                if sample_weight_plot is not None:
                    sample_weight_plot = sample_weight_plot[rows_to_keep]
                nan_mask_sq = nan_mask_sq[rows_to_keep] # For propagate consistency
            # If 'propagate', NaNs in indicators are handled by nanmean/average
        
        if indicators.shape[0] == 0: # All samples omitted
             ax.text(0.5,0.5,"No valid samples for reliability diagram.",
                     ha='center',va='center',transform=ax.transAxes)
             if show_grid: ax.grid(**(grid_props or {}))
             ax.set_title(plot_title_str or "Reliability Diagram (No Data)")
             return ax

        if nan_policy_plot == 'propagate':
            indicators = np.where(nan_mask_sq, np.nan, indicators)
        
        observed_proportions: np.ndarray
        if sample_weight_plot is not None:
            sum_sw = np.sum(sample_weight_plot)
            if sum_sw < eps_plot:
                observed_proportions = np.full(n_quantiles, np.nan)
            else:
                # Weighted average, careful with NaNs in indicators
                temp_props = []
                for q_idx in range(n_quantiles):
                    valid_inds_q = indicators[:, q_idx]
                    finite_mask_q = ~np.isnan(valid_inds_q)
                    if np.any(finite_mask_q):
                        sum_finite_weights = np.sum(sample_weight_plot[finite_mask_q])
                        if sum_finite_weights >= eps_plot:
                            prop = np.sum(
                                valid_inds_q[finite_mask_q] * \
                                sample_weight_plot[finite_mask_q]
                            ) / sum_finite_weights
                            temp_props.append(prop)
                        else: temp_props.append(np.nan)
                    else: temp_props.append(np.nan)
                observed_proportions = np.array(temp_props)
        else:
            observed_proportions = np.nanmean(indicators, axis=0)

        # Plotting reliability diagram
        ax.plot([0, 1], [0, 1], linestyle='--', color=perfect_calib_color,
                label='Perfect Calibration')
        ax.plot(q_arr, observed_proportions, marker=observed_prop_marker,
                linestyle='-', color=observed_prop_color,
                label='Observed Proportion')
        
        if show_score:
            score_for_title: Optional[float] = None
            if qce_values is not None:
                if n_outputs > 1 and output_idx is not None:
                    if isinstance(qce_values, np.ndarray) and \
                       qce_values.ndim == 1 and output_idx < len(qce_values):
                        score_for_title = qce_values[output_idx]
                elif np.isscalar(qce_values) or \
                     (isinstance(qce_values, np.ndarray) and qce_values.size==1):
                    score_for_title = float(np.ravel(qce_values)[0])
            else: # Calculate score
                title_kws = {**default_kws_for_metric, **current_metric_kws}
                if n_outputs > 1 and output_idx is not None:
                    # Score for specific output
                    title_kws['multioutput'] = 'raw_values'
                    cleaned_kws = _get_valid_kwargs(quantile_calibration_error, title_kws)
                    try:
                        all_output_scores = quantile_calibration_error(
                            y_true_arr, y_pred_q_arr, q_arr, **cleaned_kws
                        )
                        score_for_title = all_output_scores[output_idx]
                    except Exception as e:
                        warnings.warn(f"Could not calculate QCE for title: {e}")
                else: # Overall score
                    title_kws['multioutput'] = 'uniform_average'
                    cleaned_kws = _get_valid_kwargs(quantile_calibration_error, title_kws)
                    try:
                        score_for_title = quantile_calibration_error(
                            y_true_arr, y_pred_q_arr, q_arr, **cleaned_kws
                        )
                    except Exception as e:
                        warnings.warn(f"Could not calculate QCE for title: {e}")
            
            if score_for_title is not None and not np.isnan(score_for_title):
                score_text = f"Avg. QCE: {score_for_title:.4f}"
                current_title = plot_title_str or "Quantile Reliability Diagram"
                output_label_title = f" (Output {output_idx})" \
                                     if n_outputs > 1 and output_idx is not None else ""
                plot_title_str = f"{current_title}{output_label_title}\n({score_text})"

        ax.set_xlabel(xlabel or "Nominal Quantile Level (q)")
        ax.set_ylabel(ylabel or "Observed Proportion (y <= Q_pred(q))")
        ax.legend(loc='best')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

    elif kind == 'summary_bar':
        scores_to_plot: Union[float, np.ndarray]
        if qce_values is not None:
            scores_to_plot = qce_values
            if verbose > 0: print(f"Using pre-computed QCE: {scores_to_plot}")
        else:
            summary_bar_default_kws = default_kws_for_metric.copy()
            if 'multioutput' not in current_metric_kws:
                 summary_bar_default_kws['multioutput'] = 'uniform_average'
            
            effective_kws = {**summary_bar_default_kws, **current_metric_kws}
            cleaned_kws = _get_valid_kwargs(quantile_calibration_error, effective_kws)
            
            scores_to_plot = quantile_calibration_error(
                y_true_arr, y_pred_q_arr, q_arr, **cleaned_kws
            )
            if verbose > 0: print(f"Computed QCE for summary: {scores_to_plot}")
        
        scores_arr_bar: np.ndarray
        x_labels_bar: List[str]
        multioutput_used = (current_metric_kws or {}).get(
            'multioutput', default_kws_for_metric['multioutput'])

        if np.isscalar(scores_to_plot) or \
           (isinstance(scores_to_plot, np.ndarray) and scores_to_plot.ndim == 0):
            scores_arr_bar = np.array([scores_to_plot])
            x_labels_bar = ['Mean QCE']
        elif isinstance(scores_to_plot, np.ndarray) and scores_to_plot.ndim == 1:
            scores_arr_bar = scores_to_plot
            x_labels_bar = [f'Output {i}' for i in range(len(scores_arr_bar))]
            if plot_title_str and multioutput_used == 'raw_values':
                plot_title_str += " (Per Output)"
        else:
            raise TypeError(f"Unexpected QCE scores type/shape: {type(scores_to_plot)}")

        bars = ax.bar(x_labels_bar, scores_arr_bar, color=bar_color,
                      width=bar_width, **kwargs.get('bar_kwargs', {}))
        ax.set_ylabel(ylabel or 'Quantile Calibration Error (QCE)')
        
        if scores_arr_bar.size > 0 and not np.all(np.isnan(scores_arr_bar)):
            max_val = np.nanmax(scores_arr_bar)
            ax.set_ylim(0, max(0.1, max_val + 0.1 * max_val + 0.05))


        for bar_obj in bars:
            yval = bar_obj.get_height()
            if not np.isnan(yval):
                x = bar_obj.get_x() + bar_obj.get_width() / 2.0
                ax.annotate(
                    score_annotation_format.format(yval),
                    xy=(x, yval),
                    xytext=(0, 3),                # offset in points
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

    else:
        raise ValueError(
            f"Unknown plot kind: '{kind}'. Choose 'reliability_diagram' "
            "or 'summary_bar'."
        )

    if plot_title_str:
        ax.set_title(plot_title_str)
    
    if show_grid:
        current_grid_props = grid_props if grid_props is not None \
            else {'linestyle': ':', 'alpha': 0.7}
        ax.grid(**current_grid_props)
    else:
        ax.grid(False)

    return ax

plot_quantile_calibration.__doc__=r"""
Visualise Quantile Calibration Error (QCE).

Two complementary views are supported:

* **'reliability_diagram'** – plots the observed proportion  
  :math:`\Pr(y \le \hat q)` against the nominal quantile level  
  *q*.  Perfect calibration lies on the diagonal.
* **'summary_bar'** – one bar per output (or an overall bar) showing
  the time‑weighted QCE score.

Parameters
----------
{params.base.y_true}
{params.base.y_pred_quantiles}
{params.base.quantiles}

qce_values : float or ndarray, optional  
    Pre‑computed QCE value(s).  If supplied, the helper skips internal
    metric evaluation.
metric_kws : dict, optional  
    Extra keyword arguments passed to
    :func:`fusionlab.metrics.quantile_calibration_error`.

kind : {{'reliability_diagram', 'summary_bar'}},  
    default ``'reliability_diagram'``  
    Choose the visualisation style.
output_idx : int, optional  
    Output dimension to plot when the data contain multiple outputs
    and ``kind='reliability_diagram'``.

perfect_calib_color : str, default ``'red'``  
    Line colour for the 45‑degree “perfect calibration’’ reference.
observed_prop_color : str, default ``'blue'``  
observed_prop_marker : str, default ``'o'``  
    Style for the observed‑proportion curve.

{params.base.figsize}
{params.base.title}
{params.base.xlabel}
{params.base.ylabel}
{params.base.bar_color}
{params.base.bar_width}
{params.base.score_annotation_format}
show_score : bool, default ``True``  
    Display the average QCE on the plot or title.
{params.base.show_grid}
{params.base.grid_props}
{params.base.ax}
{params.base.verbose}
{params.base.kwargs}

Returns
-------
matplotlib.axes.Axes  
    Axes containing the calibration plot.

Notes
-----
The *quantile calibration error* for one output is

.. math::

   \mathrm{{QCE}} \;=\;
   \frac{{1}}{{Q}}\sum_{{k=1}}^{{Q}}
   \bigl|\,
   \hat F_y(q_k) \;-\; q_k
   \bigr|,

where :math:`\hat F_y(q_k)` is the empirical cdf evaluated at the
predicted quantile :math:`\hat q_k`.

Examples
--------
>>> import numpy as np, matplotlib.pyplot as plt
>>> from fusionlab.plot.evaluation import plot_quantile_calibration
>>> rng = np.random.default_rng(0)
>>> y_true = rng.normal(size=500)
>>> qs = np.array([0.1, 0.5, 0.9])
>>> y_pred_q = np.quantile(
...     y_true[:, None] + rng.normal(scale=.1, size=(500, 3)),
...     qs, axis=1).T
>>> plot_quantile_calibration(
...     y_true, y_pred_q, qs, kind='reliability_diagram')
>>> plt.show()

See Also
--------
fusionlab.metrics.quantile_calibration_error  
    Numeric implementation of QCE.
fusionlab.plot.evaluation.plot_weighted_interval_score  
    Visualises interval‑based probabilistic scores.
fusionlab.plot.evaluation.plot_time_weighted_metric  
    Time‑weighted MAE / accuracy / interval‑score plots.

References
----------
.. [1] Gneiting, T. & Katzfuss, M.  *Probabilistic Forecasting,*  
       *Annu. Rev. Stat. Appl.*, 2014.
""".format(params=_param_docs) 

def plot_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    coverage_values: Optional[Union[float, np.ndarray]] = None,
    metric_kws: Optional[Dict[str, Any]] = None,
    sample_indices: Optional[np.ndarray] = None,
    output_index: Optional[int] = None,
    kind: Literal['intervals', 'summary_bar'] = 'intervals',
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = "Prediction Interval Coverage",
    xlabel: str = 'Sample Index',
    ylabel: str = 'Value',
    covered_color: str = 'mediumseagreen',
    uncovered_color: str = 'salmon',
    line_color: Optional[str] = 'dimgray', 
    line_style: str = '--',              
    line_width: float = 0.8,             
    marker: str = 'o',                   
    marker_size: int = 30,               
    interval_color: str = 'skyblue',
    interval_alpha: float = 0.5,
    legend: bool = True,
    show_score: bool = True,             
    bar_color: Union[str, List[str]] = 'cornflowerblue',
    bar_width: float = 0.8,
    score_annotation_format: str = "{:.2%}", 
    show_grid: bool = True,
    grid_props: Optional[Dict[str, Any]] = None, 
    ax: Optional[plt.Axes] = None,
    verbose: int = 0,
    **kwargs: Any 
) -> plt.Axes:

    # ************************************************
    from ..metrics._registry import get_metric
    coverage_score = get_metric("coverage_score")
    # ************************************************
    
    # --- Input Validation and Preparation ---
    y_true_arr = check_array(
        y_true, ensure_2d=False, force_all_finite=False,
         dtype="numeric", copy=True)
    y_lower_arr = check_array(
        y_lower, ensure_2d=False, force_all_finite=False,
        dtype="numeric", copy=True)
    y_upper_arr = check_array(
        y_upper, ensure_2d=False, force_all_finite=False,
        dtype="numeric", copy=True
    )

    if not (y_true_arr.shape == y_lower_arr.shape == y_upper_arr.shape):
        raise ValueError(
            "y_true, y_lower, and y_upper must have the same shape."
        )
    if y_true_arr.ndim > 2:
        raise ValueError(
            "Inputs y_true, y_lower, y_upper must be 1D or 2D."
        )
    
    n_samples = y_true_arr.shape[0]
    n_outputs = y_true_arr.shape[1] if y_true_arr.ndim == 2 else 1

    if n_samples == 0:
        warnings.warn("Input arrays are empty. Cannot generate plot.")
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or "Coverage Plot (No Data)")
        if show_grid: 
            ax.grid(**(grid_props or {})) # Apply grid even for empty
        return ax

    # --- Plotting Setup ---
    if ax is None:
        # Create new figure and axes if none provided
        fig, ax = plt.subplots(figsize=figsize) # type: ignore

    plot_title_str = title # Use a mutable string for title

    # --- Metric Calculation Handling ---
    # Consolidate metric_kws for internal calls
    current_metric_kws = metric_kws or {}
    # Define default kws for coverage_score if not provided by user
    # These are used if coverage_values is None.
    default_kws_for_metric = {
        'nan_policy': 'propagate',
        'multioutput': 'uniform_average', # Default for single title score
        'eps': 1e-8,
        'verbose': 0 # Metric's internal verbose, not plot's
    }

    if kind == 'intervals':
        y_t_plot, y_l_plot, y_u_plot = y_true_arr, y_lower_arr, y_upper_arr
        current_output_label = ""
        output_idx_to_use = 0 # Default for 1D or (N,1) case

        if y_true_arr.ndim == 2: # Multi-output or (N,1)
            if n_outputs > 1: # Truly multi-output
                if output_index is None:
                    raise ValueError(
                        "For 2D y_true with >1 output and kind='intervals', "
                        "'output_index' must be specified."
                    )
                if not (0 <= output_index < n_outputs):
                    raise ValueError(
                        f"output_index {output_index} is out of bounds for "
                        f"{n_outputs} outputs."
                    )
                output_idx_to_use = output_index
            # else: n_outputs is 1, output_idx_to_use remains 0
            
            y_t_plot = y_true_arr[:, output_idx_to_use]
            y_l_plot = y_lower_arr[:, output_idx_to_use]
            y_u_plot = y_upper_arr[:, output_idx_to_use]
            if n_outputs > 1: # Add label only if truly multi-output
                current_output_label = f" (Output {output_idx_to_use})"
        
        if sample_indices is None:
            x_indices = np.arange(n_samples)
        else:
            x_indices = check_array(sample_indices, ensure_2d=False, copy=False)
            check_consistent_length(x_indices, y_t_plot)

        nan_mask_plot = np.isnan(y_t_plot) | \
                        np.isnan(y_l_plot) | \
                        np.isnan(y_u_plot)
        valid_indices = ~nan_mask_plot
        
        x_plot = x_indices[valid_indices]
        y_t_plot_valid = y_t_plot[valid_indices]
        y_l_plot_valid = y_l_plot[valid_indices]
        y_u_plot_valid = y_u_plot[valid_indices]

        covered_mask = (y_t_plot_valid >= y_l_plot_valid) & \
                       (y_t_plot_valid <= y_u_plot_valid)

        ax.fill_between(
            x_plot, y_l_plot_valid, y_u_plot_valid,
            color=interval_color, alpha=interval_alpha,
            label='Prediction Interval', **kwargs.get('fill_between_kwargs', {})
        )
        if line_color: # Renamed from true_line_color
            ax.plot(
                x_plot, y_t_plot_valid, color=line_color,
                linestyle=line_style, linewidth=line_width, # Renamed
                label='True Values (line)', **kwargs.get('plot_kwargs', {})
            )
        ax.scatter(
            x_plot[covered_mask], y_t_plot_valid[covered_mask],
            color=covered_color, marker=marker, s=marker_size, # Renamed
            label='Covered True Value', zorder=3,
            **kwargs.get('scatter_kwargs', {})
        )
        ax.scatter(
            x_plot[~covered_mask], y_t_plot_valid[~covered_mask],
            color=uncovered_color, marker=marker, s=marker_size, # Renamed
            label='Uncovered True Value', zorder=3,
            **kwargs.get('scatter_kwargs', {})
        )

        if show_score: # Renamed from show_score_on_title
            score_for_title: Optional[float] = None
            if coverage_values is not None:
                if y_true_arr.ndim == 2 and n_outputs > 1:
                    if isinstance(coverage_values, np.ndarray) and \
                       coverage_values.ndim == 1 and \
                       output_idx_to_use < len(coverage_values):
                        score_for_title = coverage_values[output_idx_to_use]
                    elif np.isscalar(coverage_values): # If user passed overall avg
                        score_for_title = float(coverage_values)
                elif np.isscalar(coverage_values) or \
                     (isinstance(coverage_values, np.ndarray) and coverage_values.size ==1):
                    score_for_title = float(np.ravel(coverage_values)[0])
            else: # Calculate score for the title
                # For title, always calculate a single score for the plotted output
                title_kws = {**default_kws_for_metric, **current_metric_kws}
                # Ensure multioutput is 'uniform_average' for single score title
                title_kws['multioutput'] = 'uniform_average' 
                cleaned_title_kws = _get_valid_kwargs(coverage_score, title_kws)
                
                score_data_y_true = ( 
                    y_true_arr[:, output_idx_to_use] 
                    if y_true_arr.ndim == 2 else y_true_arr
                    )
                score_data_y_lower = ( 
                    y_lower_arr[:, output_idx_to_use] 
                    if y_lower_arr.ndim == 2 else y_lower_arr
                    )
                score_data_y_upper = ( 
                    y_upper_arr[:, output_idx_to_use] 
                    if y_upper_arr.ndim == 2 else y_upper_arr
                    )
                
                try:
                    score_for_title = coverage_score(
                        score_data_y_true, score_data_y_lower,
                        score_data_y_upper,
                        **cleaned_title_kws
                    )
                    if verbose > 0:
                        print(f"Coverage score (for title, output "
                              f"{output_idx_to_use if y_true_arr.ndim==2 and n_outputs > 1 else ''})"
                              f": {score_for_title:.4f}")
                except Exception as e:
                    warnings.warn(f"Could not calculate score for title: {e}")
            
            if score_for_title is not None and not np.isnan(score_for_title):
                score_text = f"Coverage: {score_for_title:.2%}"
                if plot_title_str:
                    plot_title_str = f"{plot_title_str}{current_output_label}\n({score_text})"
                else:
                    plot_title_str = f"Coverage{current_output_label} ({score_text})"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend()

    elif kind == 'summary_bar':
        scores_to_plot: Union[float, np.ndarray]
        if coverage_values is not None:
            scores_to_plot = coverage_values
            if verbose > 0:
                print(f"Using pre-computed coverage values: {scores_to_plot}")
        else:
            # For summary bar, respect user's multioutput choice in metric_kws
            # Default to 'uniform_average' if not specified
            summary_bar_default_kws = default_kws_for_metric.copy()
            if 'multioutput' not in current_metric_kws: # if user did not specify
                 summary_bar_default_kws['multioutput'] = 'uniform_average'

            effective_kws = {**summary_bar_default_kws, **current_metric_kws}
            cleaned_kws = _get_valid_kwargs(coverage_score, effective_kws)
            
            scores_to_plot = coverage_score(
                y_true_arr, y_lower_arr, y_upper_arr,
                **cleaned_kws
            )
            if verbose > 0:
                print(f"Computed coverage score(s) for summary: {scores_to_plot}")
        
        scores_arr: np.ndarray
        x_labels: List[str]
        multioutput_used_for_score = (current_metric_kws or {}).get(
            'multioutput', default_kws_for_metric['multioutput'])

        if np.isscalar(scores_to_plot) or \
           (isinstance(scores_to_plot, np.ndarray) and scores_to_plot.ndim == 0):
            scores_arr = np.array([scores_to_plot])
            x_labels = ['Overall Coverage']
        elif isinstance(scores_to_plot, np.ndarray) and scores_to_plot.ndim == 1:
            scores_arr = scores_to_plot
            x_labels = [f'Output {i}' for i in range(len(scores_arr))]
            if plot_title_str and multioutput_used_for_score == 'raw_values':
                plot_title_str += " (Per Output)"
        else:
            raise TypeError(
                f"Unexpected type or shape for scores: {type(scores_to_plot)}"
            )

        bars = ax.bar(x_labels, scores_arr, color=bar_color, width=bar_width,
                      **kwargs.get('bar_kwargs', {}))
        ax.set_ylabel('Coverage Score')
        ax.set_ylim(0, max(1.1, np.nanmax(
            scores_arr) + 0.1 if scores_arr.size > 0 and not np.all(
                np.isnan(scores_arr)) else 1.1) )


        for bar_val in bars:
            yval = bar_val.get_height()
            if not np.isnan(yval):
                ax.text(bar_val.get_x() + bar_val.get_width()/2.0, yval + 0.02,
                        score_annotation_format.format(yval),
                        ha='center', va='bottom')
    else:
        raise ValueError(
            f"Unknown plot kind: '{kind}'. "
            "Choose 'intervals' or 'summary_bar'."
        )

    if plot_title_str:
        ax.set_title(plot_title_str)
    
    if show_grid:
        current_grid_props = grid_props if grid_props is not None \
            else {'linestyle': ':', 'alpha': 0.7}
        ax.grid(**current_grid_props)
    else:
        ax.grid(False)

    return ax

plot_coverage.__doc__ =r"""
Visualise prediction‑interval coverage in two ways:

* **'intervals'** –  true values overlaid on their prediction
  intervals, coloured by whether each point is covered.
* **'summary_bar'** – bar chart of the empirical coverage rate
  (overall or per output).

Parameters
------------

{params.base.y_true}
{params.base.y_lower}
{params.base.y_upper}

coverage_values : float or ndarray, optional  
    Pre‑computed coverage score(s).  If supplied, the helper skips the
    internal call to :func:`fusionlab.metrics.coverage_score`.
metric_kws : dict, optional  
    Extra keyword arguments forwarded to
    :func:`fusionlab.metrics.coverage_score` when
    ``coverage_values`` is *None*.

sample_indices : ndarray, optional  
    Custom x‑axis locations for the **'intervals'** plot.  Must match
    the first dimension of ``y_true``.
output_index : int, optional  
    Output dimension to visualise when the data contain multiple
    outputs and ``kind='intervals'``.

kind : {{'intervals', 'summary_bar'}}, default ``'intervals'``  
    Select the visualisation style.

{params.base.figsize}
title : str, optional  
    Figure title.  If *None*, a context‑aware default is generated.
xlabel : str, default ``'Sample Index'``  
ylabel : str, default ``'Value'``  

Interval‑plot styling
^^^^^^^^^^^^^^^^^^^^^
covered_color   : str, default ``'mediumseagreen'``  
uncovered_color : str, default ``'salmon'``  
line_color      : str or None, default ``'dimgray'``  
line_style      : str, default ``'--'``  
line_width      : float, default 0.8  
marker          : str, default ``'o'``  
marker_size     : int, default 30  
interval_color  : str, default ``'skyblue'``  
interval_alpha  : float, default 0.5  
legend          : bool, default ``True``  
show_score      : bool, default ``True``  
    Append the empirical coverage (as a percentage) to the title of
    the *intervals* plot.

Summary‑bar styling
^^^^^^^^^^^^^^^^^^^
{params.base.bar_color}
{params.base.bar_width}
{params.base.score_annotation_format}

{params.base.show_grid}
{params.base.grid_props}
{params.base.ax}
{params.base.verbose}
{params.base.kwargs}

Returns
-------
matplotlib.axes.Axes  
    Axes containing the coverage visualisation.

Notes
-----
The empirical coverage for one output is

.. math::

   \widehat C \;=\;
   \frac{{1}}{{N}} \sum_{{i=1}}^{{N}}
   \mathbb{{1}}\{{\,y_i \in [\ell_i, u_i]\,\}},

where :math:`[\ell_i, u_i]` is the prediction interval for sample *i*.
The helper colours covered points with *covered_color* and uncovered
points with *uncovered_color*.

Examples
--------
>>> import numpy as np, matplotlib.pyplot as plt
>>> from fusionlab.plot.evaluation import plot_coverage
>>> rng = np.random.default_rng(1)
>>> y_true  = rng.normal(size=50)
>>> y_lower = y_true - 1.0
>>> y_upper = y_true + 1.0
>>> plot_coverage(y_true, y_lower, y_upper, kind='intervals',
...               figsize=(8, 4))
>>> plt.show()

See Also
--------
fusionlab.metrics.coverage_score  
    Numerical implementation of empirical coverage.
fusionlab.plot.evaluation.plot_weighted_interval_score  
    Visualises interval sharpness and calibration jointly.
fusionlab.plot.evaluation.plot_quantile_calibration  
    Reliability diagrams for quantile forecasts.

References
----------
.. [1] Gneiting, T. & Raftery, A.E. (2007).  *Strictly Proper Scoring
       Rules, Prediction, and Estimation*.  *JASA* 102(477), 359‑378.
""".format(params=_param_docs)

def plot_crps(
    y_true: np.ndarray,
    y_pred_ensemble: np.ndarray,
    crps_values: Optional[Union[float, np.ndarray]] = None,
    metric_kws: Optional[Dict[str, Any]] = None,
    kind: Literal['ensemble_ecdf', 'scores_histogram', 'summary_bar'] = 'summary_bar',
    sample_idx: int = 0,
    output_idx: int = 0, 
    ecdf_color: str = 'dodgerblue',
    true_value_color: str = 'red',
    ensemble_marker_color: str = 'gray',
    hist_bins: Union[int, Sequence[Real], str] = 'auto',
    hist_color: str = 'skyblue',
    hist_edgecolor: str = 'black',
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = "Continuous Ranked Probability Score (CRPS)",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    bar_color: Union[str, List[str]] = 'cornflowerblue',
    bar_width: float = 0.8,
    score_annotation_format: str = "{:.4f}", 
    show_score: bool = True,
    show_grid: bool = True,
    grid_props: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    verbose: int = 0,
    **kwargs: Any 
) -> plt.Axes:
    # *************************************************
    from ..metrics._registry import get_metric
    continuous_ranked_probability_score = get_metric(
        "continuous_ranked_probability_score")
    # *************************************************
    
    # --- Input Validation and Preparation ---
    # y_true: (N,), (N,O)
    # y_pred_ensemble: (N,M), (N,O,M)
    y_true_arr = check_array(
        y_true, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=True
    )
    y_pred_ensemble_arr = check_array(
        y_pred_ensemble, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=True
    )
    # Reshape y_true and y_pred_ensemble for consistent processing
    # Target shape for y_true_proc: (N, O)
    # Target shape for y_pred_proc: (N, O, M)
    # y_true_ndim_orig = y_true_arr.ndim
    
    if y_true_arr.ndim == 1: # (N,) implies single output
        y_true_proc = y_true_arr.reshape(-1, 1) # (N, 1)
        if y_pred_ensemble_arr.ndim == 2: # (N, M)
            y_pred_proc = y_pred_ensemble_arr.reshape(
                y_pred_ensemble_arr.shape[0], 1, -1 # (N, 1, M)
            )
        elif y_pred_ensemble_arr.ndim == 3 and \
             y_pred_ensemble_arr.shape[1] == 1: # (N, 1, M)
            y_pred_proc = y_pred_ensemble_arr
        else:
            raise ValueError(
                "If y_true is 1D, y_pred_ensemble must be 2D (N,M) "
                "or 3D (N,1,M)."
            )
    elif y_true_arr.ndim == 2: # (N, O)
        y_true_proc = y_true_arr
        if y_pred_ensemble_arr.ndim == 3 and \
           y_pred_ensemble_arr.shape[1] == y_true_proc.shape[1]: # (N,O,M)
            y_pred_proc = y_pred_ensemble_arr
        else:
            raise ValueError(
                "If y_true is 2D (N,O), y_pred_ensemble must be 3D (N,O,M) "
                "with matching number of outputs."
            )
    else:
        raise ValueError("y_true must be 1D or 2D.")

    if y_true_proc.shape[:2] != y_pred_proc.shape[:2]:
        raise ValueError(
            "Mismatch in n_samples or n_outputs between y_true and "
            "y_pred_ensemble after processing."
        )

    n_samples, n_outputs, n_members = y_pred_proc.shape

    if n_samples == 0:
        warnings.warn("Input arrays are empty. Cannot generate plot.")
        if ax is None: _, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or "CRPS Plot (No Data)")
        if show_grid: ax.grid(**(grid_props or {}))
        return ax
    if n_members == 0 and kind != 'summary_bar': # summary_bar might use precomputed
        warnings.warn(
            "No ensemble members in y_pred_ensemble. "
            f"Cannot generate '{kind}' plot."
        )
        if ax is None: _, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or f"CRPS {kind} (No Ensemble Members)")
        if show_grid: ax.grid(**(grid_props or {}))
        return ax

    # --- Plotting Setup ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize) # type: ignore

    plot_title_str = title # Use a mutable string

    # --- Metric Calculation Handling ---
    current_metric_kws = metric_kws or {}
    default_kws_for_metric = {
        'nan_policy': 'propagate',
        'multioutput': 'uniform_average', # Default for overall score
        'verbose': 0 # Metric's internal verbose
    }
    # For 'scores_histogram', we need per-sample CRPS values.
    # The crps_score function, as refactored, returns per-sample, per-output
    # scores before the final aggregation if 'multioutput' is 'raw_values'
    # and then averages over samples.
    # We need to ensure we get raw per-sample scores if kind is histogram.
    
    # --- Plotting Logic ---
    if kind == 'ensemble_ecdf':
        if not (0 <= sample_idx < n_samples):
            raise ValueError(f"sample_idx {sample_idx} out of bounds.")
        if not (0 <= output_idx < n_outputs):
            raise ValueError(f"output_idx {output_idx} out of bounds.")

        sample_ensemble = y_pred_proc[sample_idx, output_idx, :]
        sample_true_value = y_true_proc[sample_idx, output_idx]

        if np.isnan(sample_true_value) or np.all(np.isnan(sample_ensemble)):
            ax.text(0.5, 0.5, "Data for ECDF contains NaNs",
                    ha='center', va='center', transform=ax.transAxes)
        else:
            # Remove NaNs from ensemble for ECDF calculation
            sample_ensemble_valid = sample_ensemble[~np.isnan(sample_ensemble)]
            if sample_ensemble_valid.size == 0:
                 ax.text(0.5, 0.5, "No valid ensemble members for ECDF",
                    ha='center', va='center', transform=ax.transAxes)
            else:
                sorted_ensemble = np.sort(sample_ensemble_valid)
                ecdf_y = np.arange(1, len(sorted_ensemble) + 1) / len(sorted_ensemble)
                ax.step(sorted_ensemble, ecdf_y, where='post',
                        color=ecdf_color, label='Ensemble ECDF')
                # Plot ensemble members as rug plot or faint points
                ax.plot(sorted_ensemble, np.zeros_like(sorted_ensemble) - 0.05,
                        '|', color=ensemble_marker_color, markersize=10,
                        alpha=0.5, label='Ensemble Members')
                ax.axvline(sample_true_value, color=true_value_color,
                           linestyle='--', label=f'True Value: {sample_true_value:.2f}')
                
                if show_score:
                    instance_crps = None
                    if crps_values is not None: # User provided per-instance scores
                        # Assuming crps_values could be (N,O) or (N,)
                        if crps_values.ndim == 2 and \
                           sample_idx < crps_values.shape[0] and \
                           output_idx < crps_values.shape[1]:
                            instance_crps = crps_values[sample_idx, output_idx]
                        elif crps_values.ndim == 1 and n_outputs == 1 and \
                             sample_idx < crps_values.shape[0]:
                             instance_crps = crps_values[sample_idx]
                    else: # Calculate for this instance
                        instance_kws = {**default_kws_for_metric, **current_metric_kws}
                        instance_kws['multioutput'] = 'raw_values' # Get raw for this one
                        cleaned_kws = _get_valid_kwargs(
                            continuous_ranked_probability_score, instance_kws)
                        try:
                            # Need to pass single sample/output to crps_score
                            # crps_score expects (N,O) for y_true, (N,O,M) for y_pred
                            temp_y_true = y_true_proc[sample_idx:sample_idx+1, output_idx:output_idx+1]
                            temp_y_pred = y_pred_proc[sample_idx:sample_idx+1, output_idx:output_idx+1, :]
                            
                            calculated_scores = continuous_ranked_probability_score(
                                temp_y_true, temp_y_pred, **cleaned_kws
                            ) # Should be scalar or (1,) array
                            instance_crps = float(np.ravel(calculated_scores)[0])
                        except Exception as e:
                            warnings.warn(f"Could not calculate CRPS for ECDF title: {e}")

                    if instance_crps is not None and not np.isnan(instance_crps):
                        score_text = f"CRPS: {instance_crps:.4f}"
                        current_title = plot_title_str or "Ensemble ECDF vs True Value"
                        plot_title_str = f"{current_title}\n({score_text})"
                        
        ax.set_xlabel(xlabel or 'Value')
        ax.set_ylabel(ylabel or 'Cumulative Probability')
        ax.legend(loc='best')
        ax.set_ylim(-0.1, 1.1)

    elif kind == 'scores_histogram':
        # This requires per-sample (and potentially per-output) CRPS values
        per_sample_output_crps: Optional[np.ndarray] = None
        if crps_values is not None:
            if crps_values.ndim == 1 and len(crps_values) == n_samples and n_outputs == 1:
                per_sample_output_crps = crps_values # Assumed (N,) for single output
            elif crps_values.ndim == 2 and crps_values.shape[0] == n_samples \
                                       and crps_values.shape[1] == n_outputs:
                per_sample_output_crps = crps_values # (N,O)
            else:
                warnings.warn("Provided `crps_values` shape incompatible for histogram.")
        else:
            hist_kws = {**default_kws_for_metric, **current_metric_kws} # noqa
            # XXX TODO:
            # To get per-sample, per-output scores, need to call 
            # crps_score differently
            # The current continuous_ranked_probability_score refactor 
            # returns (N_calc, O) before final aggregation
            # This is not directly exposed. We need to call it in a loop
            # or modify continuous_ranked_probability_score
            # For now, let's assume we can get per-sample scores for a chosen 
            # output_idx Or average over outputs if output_idx is None.
            # Let's compute CRPS for each sample, for a specific 
            # output_idx or averaged
            all_sample_crps_list = []
            for s_idx in range(n_samples):
                temp_y_true = y_true_proc[s_idx:s_idx+1, ...] # (1,O)
                temp_y_pred = y_pred_proc[s_idx:s_idx+1, ...] # (1,O,M)
                
                # Kws for single sample CRPS calculation
                single_sample_kws = {**default_kws_for_metric, **current_metric_kws}
                single_sample_kws['multioutput'] = 'raw_values' # get (O,) scores
                cleaned_kws_ss = _get_valid_kwargs(
                    continuous_ranked_probability_score, single_sample_kws)

                try:
                    s_crps = continuous_ranked_probability_score(
                        temp_y_true, temp_y_pred, **cleaned_kws_ss) # (O,)
                    if n_outputs > 1 and output_idx is not None:
                        if 0 <= output_idx < n_outputs:
                            all_sample_crps_list.append(s_crps[output_idx])
                        else: # Should not happen if validated
                            all_sample_crps_list.append(np.nan)
                    else: # Single output or average over outputs for this sample
                        all_sample_crps_list.append(np.nanmean(s_crps))
                except Exception:
                    all_sample_crps_list.append(np.nan)
            per_sample_output_crps = np.array(all_sample_crps_list)

        if per_sample_output_crps is not None:
            valid_crps_for_hist = per_sample_output_crps[
                ~np.isnan(per_sample_output_crps)]
            if valid_crps_for_hist.size > 0:
                ax.hist(valid_crps_for_hist, bins=hist_bins,
                        color=hist_color, edgecolor=hist_edgecolor,
                        **kwargs.get('hist_kwargs', {}))
                avg_crps_for_title = np.nanmean(valid_crps_for_hist)
                if show_score and not np.isnan(avg_crps_for_title):
                     current_title = plot_title_str or "Distribution of CRPS Values"
                     plot_title_str = f"{current_title}\n(Mean CRPS: {avg_crps_for_title:.4f})"
            else:
                ax.text(0.5,0.5, "No valid CRPS scores for histogram",
                        ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5,0.5, "CRPS scores not available for histogram",
                    ha='center', va='center', transform=ax.transAxes)

        ax.set_xlabel(xlabel or 'CRPS per Sample')
        ax.set_ylabel(ylabel or 'Frequency')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


    elif kind == 'summary_bar':
        scores_to_plot: Union[float, np.ndarray]
        if crps_values is not None:
            scores_to_plot = crps_values
            if verbose > 0: print(f"Using pre-computed CRPS: {scores_to_plot}")
        else:
            summary_bar_default_kws = default_kws_for_metric.copy()
            # Respect user's multioutput for summary bar
            if 'multioutput' not in current_metric_kws:
                 summary_bar_default_kws['multioutput'] = 'uniform_average'
            
            effective_kws = {**summary_bar_default_kws, **current_metric_kws}
            cleaned_kws = _get_valid_kwargs(
                continuous_ranked_probability_score, effective_kws)
            
            scores_to_plot = continuous_ranked_probability_score(
                y_true_proc, y_pred_proc, **cleaned_kws
            )
            if verbose > 0: print(f"Computed CRPS for summary: {scores_to_plot}")
        
        scores_arr: np.ndarray
        x_labels: List[str]
        multioutput_used = (current_metric_kws or {}).get(
            'multioutput', default_kws_for_metric['multioutput'])

        if np.isscalar(scores_to_plot) or \
           (isinstance(scores_to_plot, np.ndarray) and scores_to_plot.ndim == 0):
            scores_arr = np.array([scores_to_plot])
            x_labels = ['Overall CRPS']
        elif isinstance(scores_to_plot, np.ndarray) and scores_to_plot.ndim == 1:
            scores_arr = scores_to_plot
            x_labels = [f'Output {i}' for i in range(len(scores_arr))]
            if plot_title_str and multioutput_used == 'raw_values':
                plot_title_str += " (Per Output)"
        else:
            raise TypeError(f"Unexpected scores type/shape: {type(scores_to_plot)}")

        bars = ax.bar(x_labels, scores_arr, color=bar_color, width=bar_width,
                      **kwargs.get('bar_kwargs', {}))
        ax.set_ylabel(ylabel or 'CRPS')
        
        min_score = np.nanmin(scores_arr) if scores_arr.size > 0 else 0
        max_score = np.nanmax(scores_arr) if scores_arr.size > 0 else 0.1
        ax.set_ylim(min(0, min_score - 0.1 * abs(min_score)),
                    max_score + 0.1 * abs(max_score) + 0.05)


        for bar in bars:
            yval = bar.get_height()
            if not np.isnan(yval):
                x = bar.get_x() + bar.get_width() / 2.0
                label = score_annotation_format.format(yval)
                ax.annotate(
                    label, 
                    xy=(x, yval),
                    xytext=(0, 3 if yval >= 0 else -10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if yval >= 0 else "top",
                )

    else:
        raise ValueError(f"Unknown plot kind: '{kind}'.")

    if plot_title_str:
        ax.set_title(plot_title_str)
    
    if show_grid:
        current_grid_props = grid_props if grid_props is not None \
            else {'linestyle': ':', 'alpha': 0.7}
        ax.grid(**current_grid_props)
    else:
        ax.grid(False)

    return ax

plot_crps.__doc__=r"""
Visualise the Continuous Ranked Probability Score (CRPS) for
ensemble forecasts.

Three complementary views are available:

* **'ensemble_ecdf'** – ECDF of a single ensemble, the true value,  
  and the per‑instance CRPS.
* **'scores_histogram'** – distribution of per‑sample CRPS values.
* **'summary_bar'** – bar chart of the overall CRPS (or one bar per
  output).

Parameters
----------
{params.base.y_true}

y_pred_ensemble : ndarray  
    Ensemble predictions.  Shape *(N, M)* for a single output or
    *(N, O, M)* for multiple outputs, where *M* is the number of
    ensemble members.

crps_values : float or ndarray, optional  
    Pre‑computed CRPS value(s).  If supplied, the helper skips internal
    calls to :func:`fusionlab.metrics.continuous_ranked_probability_score`.
{params.base.metric_kws}

kind : {{'ensemble_ecdf', 'scores_histogram', 'summary_bar'}},  
    default ``'summary_bar'``  
    Style of plot to generate.

sample_idx : int, default 0  
    Index of the sample to display when ``kind='ensemble_ecdf'``.
output_idx : int, default 0  
    Output dimension to display when ``kind='ensemble_ecdf'``.

ecdf_color : str, default ``'dodgerblue'``  
true_value_color : str, default ``'red'``  
ensemble_marker_color : str, default ``'gray'``  
    Styling parameters for the ECDF plot.

hist_bins : int | sequence | str, default ``'auto'``  
hist_color : str, default ``'skyblue'``  
hist_edgecolor : str, default ``'black'``  
    Histogram styling parameters.

{params.base.figsize}
{params.base.title}
{params.base.xlabel}
{params.base.ylabel}
{params.base.bar_color}
{params.base.bar_width}
{params.base.score_annotation_format}
show_score : bool, default ``True``  
    Append the numeric CRPS to the title (where applicable).
{params.base.show_grid}
{params.base.grid_props}
{params.base.ax}
{params.base.verbose}
{params.base.kwargs}

Returns
-------
matplotlib.axes.Axes  
    Axes containing the CRPS visualisation.

Notes
-----
For one observation with ensemble members
:math:`x_1,\dots,x_M` and true value :math:`y`, the
sample‑based CRPS is

.. math::

   \operatorname{{CRPS}} \;=\;
     \frac{{1}}{{M}}\sum_{{j=1}}^M |x_j - y|
     \;-\;\frac{{1}}{{2M^2}}\sum_{{i=1}}^M\sum_{{j=1}}^M |x_i - x_j|.

Lower scores indicate sharper and better‑calibrated
probabilistic forecasts.

Examples
--------
>>> import numpy as np, matplotlib.pyplot as plt
>>> from fusionlab.plot.evaluation import plot_crps
>>> rng = np.random.default_rng(0)
>>> y_true = rng.normal(size=500)
>>> ens     = y_true[:, None] + rng.normal(scale=.5, size=(500, 20))
>>> plot_crps(y_true, ens, kind='scores_histogram')
>>> plt.show()

See Also
--------
fusionlab.metrics.continuous_ranked_probability_score  
    Numeric computation of sample‑based CRPS.
fusionlab.plot.evaluation.plot_quantile_calibration  
    Reliability diagrams for quantile forecasts.
fusionlab.plot.evaluation.plot_weighted_interval_score  
    Interval‑based sharpness and calibration plot.

References
----------
.. [1] Hersbach, H. (2000). *Decomposition of the Continuous Ranked  
       Probability Score for Ensemble Prediction Systems*.  
       *Weather and Forecasting*, 15(5), 559‑570.
""".format(params=_param_docs)

def plot_mean_interval_width(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    miw_values: Optional[Union[float, np.ndarray]] = None,
    metric_kws: Optional[Dict[str, Any]] = None,
    kind: Literal['widths_histogram', 'summary_bar'] = 'summary_bar',
    output_idx: Optional[int] = None, 
    hist_bins: Union[int, Sequence[Real], str] = 'auto',
    hist_color: str = 'mediumpurple',
    hist_edgecolor: str = 'black',
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = "Mean Interval Width (Sharpness)",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    bar_color: Union[str, List[str]] = 'mediumpurple',
    bar_width: float = 0.8,
    score_annotation_format: str = "{:.4f}",
    show_score: bool = True, 
    show_grid: bool = True,
    grid_props: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    verbose: int = 0,
    **kwargs: Any 
) -> plt.Axes:

    # *************************************************
    from ..metrics._registry import get_metric
    mean_interval_width_score = get_metric(
        "mean_interval_width_score")
    # *************************************************
    
    # --- Input Validation and Preparation ---
    # y_lower, y_upper: (N,), (N,O)
    y_lower_arr = check_array(
        y_lower, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=True
    )
    y_upper_arr = check_array(
        y_upper, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=True
    )

    if y_lower_arr.shape != y_upper_arr.shape:
        raise ValueError(
            "y_lower and y_upper must have the same shape."
        )
    if y_lower_arr.ndim > 2:
        raise ValueError("Inputs y_lower/y_upper must be 1D or 2D.")

    # Reshape for consistent processing: (N, O)
    y_lower_proc = y_lower_arr.reshape(
        -1, 1) if y_lower_arr.ndim == 1 else y_lower_arr
    y_upper_proc = y_upper_arr.reshape(
        -1, 1) if y_upper_arr.ndim == 1 else y_upper_arr

    n_samples, n_outputs = y_lower_proc.shape

    if n_samples == 0:
        warnings.warn("Input arrays are empty. Cannot generate plot.")
        if ax is None: _, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or "Mean Interval Width (No Data)")
        if show_grid: ax.grid(**(grid_props or {}))
        return ax

    # --- Plotting Setup ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize) # type: ignore

    plot_title_str = title # Use a mutable string

    # --- Metric Calculation Handling ---
    current_metric_kws = metric_kws or {}
    default_kws_for_metric = {
        'nan_policy': 'propagate',
        'multioutput': 'uniform_average', # Default for overall score
        'warn_invalid_bounds': True,
        'eps': 1e-8,
        'verbose': 0 # Metric's internal verbose
    }

    # --- Plotting Logic ---
    if kind == 'widths_histogram':
        # For histogram, we need individual widths, calculated from inputs.
        # `miw_values` (if mean scores) is not used for this kind.
        # Handle NaNs based on metric_kws for calculating widths
        # This logic needs to align with how mean_interval_width_score handles it
        nan_policy_hist = current_metric_kws.get(
            'nan_policy', default_kws_for_metric['nan_policy']
        )
        temp_y_lower, temp_y_upper = y_lower_proc, y_upper_proc
        nan_mask_inputs = np.isnan(temp_y_lower) | np.isnan(temp_y_upper)

        if np.any(nan_mask_inputs):
            if nan_policy_hist == 'raise':
                raise ValueError(
                    "NaNs found in y_lower/y_upper for histogram."
                )
            elif nan_policy_hist == 'omit':
                # Omit rows where *any* output has NaN for this sample
                rows_with_nan = nan_mask_inputs.any(axis=1)
                rows_to_keep = ~rows_with_nan
                if not np.any(rows_to_keep):
                    ax.text(0.5,0.5,"All samples omitted due to NaNs.",
                            ha='center', va='center', transform=ax.transAxes)
                    if show_grid: ax.grid(**(grid_props or {}))
                    ax.set_title(plot_title_str or "Interval Widths (No Data)")
                    return ax
                temp_y_lower = temp_y_lower[rows_to_keep]
                temp_y_upper = temp_y_upper[rows_to_keep]
                # Update nan_mask_inputs for propagate logic if it were used
                nan_mask_inputs = nan_mask_inputs[rows_to_keep]

        individual_widths = temp_y_upper - temp_y_lower # (N_calc, O)
        if nan_policy_hist == 'propagate':
            # nan_mask_inputs is (N_calc, O) if omit was applied to it
            individual_widths = np.where(
                nan_mask_inputs, np.nan, individual_widths
            )

        # Select output for histogram if multi-output
        widths_to_plot: np.ndarray
        current_output_label = ""
        if n_outputs > 1:
            if output_idx is None:
                raise ValueError(
                    "For multi-output data and kind='widths_histogram', "
                    "'output_idx' must be specified."
                )
            if not (0 <= output_idx < n_outputs):
                raise ValueError(
                    f"output_idx {output_idx} out of bounds for "
                    f"{n_outputs} outputs."
                )
            widths_to_plot = individual_widths[:, output_idx]
            current_output_label = f" (Output {output_idx})"
        else: # Single output
            widths_to_plot = individual_widths.ravel()

        valid_widths_for_hist = widths_to_plot[
            ~np.isnan(widths_to_plot)
        ]

        if valid_widths_for_hist.size > 0:
            ax.hist(valid_widths_for_hist, bins=hist_bins,
                    color=hist_color, edgecolor=hist_edgecolor,
                    **kwargs.get('hist_kwargs', {}))
            
            if show_score:
                mean_of_plotted_widths = np.mean(valid_widths_for_hist)
                score_text = f"Mean Width: {mean_of_plotted_widths:.4f}"
                current_title = plot_title_str or \
                                "Distribution of Interval Widths"
                plot_title_str = (
                    f"{current_title}{current_output_label}\n({score_text})"
                )
        else:
            ax.text(0.5,0.5, "No valid interval widths for histogram.",
                    ha='center', va='center', transform=ax.transAxes)
            current_title = plot_title_str or \
                            "Distribution of Interval Widths"
            plot_title_str = f"{current_title}{current_output_label} (No Data)"


        ax.set_xlabel(xlabel or 'Interval Width')
        ax.set_ylabel(ylabel or 'Frequency')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    elif kind == 'summary_bar':
        scores_to_plot: Union[float, np.ndarray]
        if miw_values is not None:
            scores_to_plot = miw_values
            if verbose > 0:
                print(f"Using pre-computed MIW values: {scores_to_plot}")
        else:
            # For summary bar, respect user's multioutput choice in metric_kws
            summary_bar_default_kws = default_kws_for_metric.copy()
            if 'multioutput' not in current_metric_kws:
                 summary_bar_default_kws['multioutput'] = 'uniform_average'
            
            effective_kws = {**summary_bar_default_kws, **current_metric_kws}
            cleaned_kws = _get_valid_kwargs(
                mean_interval_width_score, effective_kws
            )
            
            scores_to_plot = mean_interval_width_score(
                y_lower_arr, # Use original full arrays for score calculation
                y_upper_arr,
                **cleaned_kws
            )
            if verbose > 0:
                print(f"Computed MIW score(s) for summary: {scores_to_plot}")
        
        scores_arr: np.ndarray
        x_labels: List[str]
        multioutput_used_for_score = (current_metric_kws or {}).get(
            'multioutput', default_kws_for_metric['multioutput']
        )

        if np.isscalar(scores_to_plot) or \
           (isinstance(scores_to_plot, np.ndarray) and scores_to_plot.ndim == 0):
            scores_arr = np.array([scores_to_plot])
            x_labels = ['Mean Interval Width']
        elif isinstance(scores_to_plot, np.ndarray) and scores_to_plot.ndim == 1:
            scores_arr = scores_to_plot
            x_labels = [f'Output {i}' for i in range(len(scores_arr))]
            if plot_title_str and multioutput_used_for_score == 'raw_values':
                plot_title_str += " (Per Output)"
        else:
            raise TypeError(
                f"Unexpected type or shape for MIW scores: {type(scores_to_plot)}"
            )

        bars = ax.bar(x_labels, scores_arr, color=bar_color,
                      width=bar_width, **kwargs.get('bar_kwargs', {}))
        ax.set_ylabel(ylabel or 'Mean Interval Width')
        
        # Auto-adjust y-limits for better visualization
        if scores_arr.size > 0 and not np.all(np.isnan(scores_arr)):
            min_val = np.nanmin(scores_arr)
            max_val = np.nanmax(scores_arr)
            padding = 0.1 * (max_val - min_val) if (
                max_val - min_val) > 1e-6 else 0.1
            ax.set_ylim(min(
                0, min_val - padding), max_val + padding + 0.05) 

        for bar_obj in bars:
            yval = bar_obj.get_height()
            if not np.isnan(yval):
                x = bar_obj.get_x() + bar_obj.get_width() / 2.0
                label = score_annotation_format.format(yval)
                ax.annotate(
                    label,
                    xy=(x, yval),
                    xytext=(0, 3 if yval >= 0 else -10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if yval >= 0 else "top",
                )

    else:
        raise ValueError(
            f"Unknown plot kind: '{kind}'. Choose 'widths_histogram' "
            "or 'summary_bar'."
        )

    if plot_title_str:
        ax.set_title(plot_title_str)
    
    if show_grid:
        current_grid_props = grid_props if grid_props is not None \
            else {'linestyle': ':', 'alpha': 0.7}
        ax.grid(**current_grid_props)
    else:
        ax.grid(False)

    return ax

plot_mean_interval_width.__doc__ =r"""
Visualise Mean Interval Width (MIW) – a simple sharpness measure
equal to the average distance between lower and upper prediction‐
interval bounds.

Two complementary views are implemented:

* **'widths_histogram'** – distribution of individual interval widths
  for a chosen output.
* **'summary_bar'** – bar chart of the averaged width (overall or one
  bar per output).

Parameters
----------
{params.base.y_lower}
{params.base.y_upper}

miw_values : float or ndarray, optional  
    Pre‑computed MIW score(s).  If supplied the helper skips the
    internal call to
    :func:`fusionlab.metrics.mean_interval_width_score`.
metric_kws : dict, optional  
    Extra keyword arguments forwarded to
    :func:`fusionlab.metrics.mean_interval_width_score`.

kind : {{'widths_histogram', 'summary_bar'}},  
    default ``'summary_bar'``  
    Select the visualisation style.

output_idx : int, optional  
    Output dimension to plot when ``kind='widths_histogram'`` on
    multi‑output data.

hist_bins : int | sequence | str, default ``'auto'``  
hist_color : str, default ``'mediumpurple'``  
hist_edgecolor : str, default ``'black'``  
    Styling options for the histogram.

{params.base.figsize}
{params.base.title}
{params.base.xlabel}
{params.base.ylabel}
{params.base.bar_color}
{params.base.bar_width}
{params.base.score_annotation_format}
show_score : bool, default ``True``  
    Display the mean width on the histogram title.
{params.base.show_grid}
{params.base.grid_props}
{params.base.ax}
{params.base.verbose}
{params.base.kwargs}

Returns
-------
matplotlib.axes.Axes  
    Axes containing the MIW visualisation.

Notes
-----
For a single observation the interval width is simply

.. math::

   w_i \;=\; u_i \;-\; \ell_i ,

where :math:`u_i` and :math:`\ell_i` are the upper and lower bounds.
The mean interval width over *N* samples is

.. math::

   \text{{MIW}} \;=\; \frac{{1}}{{N}}\sum_{{i=1}}^{{N}} w_i.

Lower MIW indicates a *sharper* forecast, but should always be
interpreted together with coverage diagnostics.

Examples
--------
>>> import numpy as np, matplotlib.pyplot as plt
>>> from fusionlab.plot.evaluation import plot_mean_interval_width
>>> rng = np.random.default_rng(1)
>>> y_l = rng.normal(loc=-1.0, scale=.5, size=200)
>>> y_u = y_l + rng.uniform(1.5, 2.5, size=200)
>>> plot_mean_interval_width(
...     y_lower=y_l, y_upper=y_u, kind='widths_histogram',
...     figsize=(8, 4))
>>> plt.show()

See Also
--------
fusionlab.metrics.mean_interval_width_score  
    Numeric computation of MIW.
fusionlab.plot.evaluation.plot_coverage  
    Shows how many observations fall inside the intervals.
fusionlab.plot.evaluation.plot_weighted_interval_score  
    Combines width with calibration penalties.

References
----------
.. [1] Gneiting, T. & Katzfuss, M. (2014). *Probabilistic Forecasting*.
       *Ann. Rev. Stat. Appl.*, 1, 125‑151 — section 4.1, “Sharpness”.
""".format(params=_param_docs)

def plot_prediction_stability(
    y_pred: np.ndarray,
    pss_values: Optional[Union[float, np.ndarray]] = None,
    metric_kws: Optional[Dict[str, Any]] = None,
    kind: Literal['scores_histogram', 'summary_bar'] = 'summary_bar',
    output_idx: Optional[int] = None, 
    hist_bins: Union[int, Sequence[Real], str] = 'auto',
    hist_color: str = 'teal',
    hist_edgecolor: str = 'black',
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = "Prediction Stability Score (PSS)",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    bar_color: Union[str, List[str]] = 'teal',
    bar_width: float = 0.8,
    score_annotation_format: str = "{:.4f}",
    show_score: bool = True, 
    show_grid: bool = True,
    grid_props: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    verbose: int = 0,
    **kwargs: Any 
) -> plt.Axes:

    # **********************************************************************
    from ..metrics._registry import get_metric
    prediction_stability_score = get_metric("prediction_stability_score")
    # **********************************************************************
    
    # --- Input Validation and Preparation ---
    # y_pred: (T,), (N,T), or (N,O,T)
    y_pred_arr = check_array(
        y_pred, ensure_2d=False, allow_nd=True,
        dtype="numeric", force_all_finite=False, copy=True
    )

    # Reshape for consistent processing: (N, O, T)
    y_pred_ndim_orig = y_pred_arr.ndim
    if y_pred_ndim_orig == 1: # (T,)
        y_pred_proc = y_pred_arr.reshape(1, 1, -1)
    elif y_pred_ndim_orig == 2: # (N, T)
        y_pred_proc = y_pred_arr.reshape(y_pred_arr.shape[0], 1, -1)
    elif y_pred_ndim_orig == 3: # (N, O, T)
        y_pred_proc = y_pred_arr
    else:
        raise ValueError(
            "y_pred must be 1D, 2D (n_samples, n_timesteps), or 3D "
            "(n_samples, n_outputs, n_timesteps)."
        )

    n_samples, n_outputs, n_timesteps = y_pred_proc.shape

    if n_samples == 0:
        warnings.warn("Input y_pred is empty. Cannot generate plot.")
        if ax is None: _, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or "PSS Plot (No Data)")
        if show_grid: ax.grid(**(grid_props or {}))
        return ax
    
    if n_timesteps < 2 and kind != 'summary_bar':
        # summary_bar might use precomputed values
        warnings.warn(
            "PSS requires at least 2 time steps for histogram. "
            "Plot may be empty or misleading."
        )
        if ax is None: _, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or f"PSS {kind} (Not Enough Timesteps)")
        if show_grid: ax.grid(**(grid_props or {}))
        return ax


    # --- Plotting Setup ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize) # type: ignore

    plot_title_str = title # Use a mutable string

    # --- Metric Calculation Handling ---
    current_metric_kws = metric_kws or {}
    default_kws_for_metric = {
        'nan_policy': 'propagate',
        'multioutput': 'uniform_average', # Default for overall score
        'verbose': 0 # Metric's internal verbose
    }

    # --- Plotting Logic ---
    if kind == 'scores_histogram':
        # For histogram, we need PSS per trajectory (per sample, per output).
        # The prediction_stability_score function's internal `pss_per_trajectory`
        # (before sample averaging) is what we need.
        
        # To get these, we call prediction_stability_score in a way that
        # allows us to capture these intermediate values if not directly exposed.
        # However, the refactored `prediction_stability_score` calculates
        # `pss_per_trajectory` of shape (N_calc, O). This is suitable.

        # We need to calculate these raw per-trajectory scores.
        # `pss_values` if provided should be these raw scores.
        per_trajectory_scores: Optional[np.ndarray] = None
        if pss_values is not None:
            # Assume pss_values, if provided for histogram, are already
            # per-trajectory (N,O) or (N,) if single output.
            if pss_values.ndim == 1 and n_outputs == 1 and \
               len(pss_values) == n_samples :
                per_trajectory_scores = pss_values.reshape(-1,1) # Ensure (N,1)
            elif pss_values.ndim == 2 and \
                 pss_values.shape[0] == n_samples and \
                 pss_values.shape[1] == n_outputs:
                per_trajectory_scores = pss_values
            else:
                warnings.warn(
                    "Provided `pss_values` shape incompatible for histogram. "
                    "Recalculating per-trajectory scores."
                )
        
        if per_trajectory_scores is None:
            # Calculate per-trajectory scores.
            # This requires a bit of care as the main metric returns aggregated scores.
            # We can simulate by calling for each sample if needed, or use internals.
            # The current `prediction_stability_score` structure:
            # y_pred_proc (N,O,T) -> diffs (N,O,T-1) -> pss_per_trajectory (N,O)
            # This `pss_per_trajectory` is exactly what we need.
            # So, we can effectively run parts of the metric logic here.
            
            nan_policy_hist = current_metric_kws.get(
                'nan_policy', default_kws_for_metric['nan_policy']
            )
            temp_y_pred = y_pred_proc.copy() # Use copy for local NaN handling
            
            nan_mask_sot = np.isnan(temp_y_pred)
            nan_mask_so_hist = nan_mask_sot.any(axis=2) # (N,O)

            if np.any(nan_mask_so_hist):
                if nan_policy_hist == 'raise':
                    raise ValueError(
                        "NaNs found in y_pred for histogram."
                    )
                elif nan_policy_hist == 'omit':
                    rows_with_nan = nan_mask_so_hist.any(axis=1) # (N,)
                    rows_to_keep = ~rows_with_nan
                    if not np.any(rows_to_keep):
                        ax.text(0.5,0.5,"All samples omitted due to NaNs.",
                                ha='center',va='center',transform=ax.transAxes)
                        if show_grid: ax.grid(**(grid_props or {}))
                        ax.set_title(plot_title_str or "PSS Scores (No Data)")
                        return ax
                    temp_y_pred = temp_y_pred[rows_to_keep]
                    nan_mask_so_hist = nan_mask_so_hist[rows_to_keep]
            
            if temp_y_pred.shape[0] == 0 or temp_y_pred.shape[2] < 2:
                 ax.text(0.5,0.5,"Not enough data for PSS histogram.",
                         ha='center',va='center',transform=ax.transAxes)
                 if show_grid: ax.grid(**(grid_props or {}))
                 ax.set_title(plot_title_str or "PSS Scores (No Data)")
                 return ax

            diffs_hist = np.abs(
                temp_y_pred[..., 1:] - temp_y_pred[..., :-1]
            )
            per_trajectory_scores = np.mean(diffs_hist, axis=2) # (N_calc, O)

            if nan_policy_hist == 'propagate':
                per_trajectory_scores = np.where(
                    nan_mask_so_hist, np.nan, per_trajectory_scores
                )
        
        # Select output for histogram
        scores_to_plot_hist: np.ndarray
        current_output_label = ""
        if n_outputs > 1:
            if output_idx is None:
                raise ValueError(
                    "For multi-output data and kind='scores_histogram', "
                    "'output_idx' must be specified."
                )
            if not (0 <= output_idx < n_outputs):
                raise ValueError(
                    f"output_idx {output_idx} out of bounds for "
                    f"{n_outputs} outputs."
                )
            scores_to_plot_hist = per_trajectory_scores[:, output_idx]
            current_output_label = f" (Output {output_idx})"
        else: # Single output
            scores_to_plot_hist = per_trajectory_scores.ravel()

        valid_scores_for_hist = scores_to_plot_hist[
            ~np.isnan(scores_to_plot_hist)
        ]

        if valid_scores_for_hist.size > 0:
            ax.hist(valid_scores_for_hist, bins=hist_bins,
                    color=hist_color, edgecolor=hist_edgecolor,
                    **kwargs.get('hist_kwargs', {}))
            
            if show_score:
                mean_of_plotted_pss = np.mean(valid_scores_for_hist)
                score_text = f"Mean PSS: {mean_of_plotted_pss:.4f}"
                current_title = plot_title_str or \
                                "Distribution of PSS Values"
                plot_title_str = (
                    f"{current_title}{current_output_label}\n({score_text})"
                )
        else:
            ax.text(0.5,0.5, "No valid PSS values for histogram.",
                    ha='center', va='center', transform=ax.transAxes)
            current_title = plot_title_str or \
                            "Distribution of PSS Values"
            plot_title_str = f"{current_title}{current_output_label} (No Data)"

        ax.set_xlabel(xlabel or 'PSS per Trajectory')
        ax.set_ylabel(ylabel or 'Frequency')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    elif kind == 'summary_bar':
        scores_to_plot_bar: Union[float, np.ndarray]
        if pss_values is not None:
            scores_to_plot_bar = pss_values
            if verbose > 0:
                print(f"Using pre-computed PSS values: {scores_to_plot_bar}")
        else:
            summary_bar_default_kws = default_kws_for_metric.copy()
            if 'multioutput' not in current_metric_kws:
                 summary_bar_default_kws['multioutput'] = 'uniform_average'
            
            effective_kws = {**summary_bar_default_kws, **current_metric_kws}
            cleaned_kws = _get_valid_kwargs(
                prediction_stability_score, effective_kws
            )
            
            scores_to_plot_bar = prediction_stability_score(
                y_pred_arr, # Use original y_pred_arr for metric call
                **cleaned_kws
            )
            if verbose > 0:
                print(f"Computed PSS for summary: {scores_to_plot_bar}")
        
        scores_arr_bar: np.ndarray
        x_labels_bar: List[str]
        multioutput_used = (current_metric_kws or {}).get(
            'multioutput', default_kws_for_metric['multioutput'])

        if np.isscalar(scores_to_plot_bar) or \
           (isinstance(scores_to_plot_bar, np.ndarray) and \
            scores_to_plot_bar.ndim == 0):
            scores_arr_bar = np.array([scores_to_plot_bar])
            x_labels_bar = ['Mean PSS']
        elif isinstance(scores_to_plot_bar, np.ndarray) and \
             scores_to_plot_bar.ndim == 1:
            scores_arr_bar = scores_to_plot_bar
            x_labels_bar = [f'Output {i}' for i in range(len(scores_arr_bar))]
            if plot_title_str and multioutput_used == 'raw_values':
                plot_title_str += " (Per Output)"
        else:
            raise TypeError(
                f"Unexpected type/shape for PSS scores: {type(scores_to_plot_bar)}"
            )

        bars = ax.bar(x_labels_bar, scores_arr_bar, color=bar_color,
                      width=bar_width, **kwargs.get('bar_kwargs', {}))
        ax.set_ylabel(ylabel or 'Prediction Stability Score (PSS)')
        
        if scores_arr_bar.size > 0 and not np.all(np.isnan(scores_arr_bar)):
            min_val = np.nanmin(scores_arr_bar)
            max_val = np.nanmax(scores_arr_bar)
            padding = 0.1 * (max_val - min_val) if (max_val-min_val)>1e-6 else 0.1
            ax.set_ylim(min(0, min_val - padding), max_val + padding + 0.05)

        for bar_obj in bars:
            yval = bar_obj.get_height()
            if not np.isnan(yval):
                x = bar_obj.get_x() + bar_obj.get_width() / 2.0
                dy = 3 if yval >= 0 else -10
                va = 'bottom' if yval >= 0 else 'top'
                ax.annotate(
                    score_annotation_format.format(yval),
                    xy=(x, yval),
                    xytext=(0, dy),
                    textcoords="offset points",
                    ha="center",
                    va=va,
                )

    else:
        raise ValueError(
            f"Unknown plot kind: '{kind}'. Choose 'scores_histogram' "
            "or 'summary_bar'."
        )

    if plot_title_str:
        ax.set_title(plot_title_str)
    
    if show_grid:
        current_grid_props = grid_props if grid_props is not None \
            else {'linestyle': ':', 'alpha': 0.7}
        ax.grid(**current_grid_props)
    else:
        ax.grid(False)

    return ax

plot_prediction_stability.__doc__=r"""
Visualise the Prediction Stability Score (PSS) — the average
absolute change between successive time steps in a forecast
trajectory.  Lower PSS ⇒ smoother (more stable) predictions.

Two complementary views are provided:

* **'scores_histogram'** – distribution of per‑trajectory PSS values
  for a chosen output.
* **'summary_bar'** – bar chart of the mean PSS (overall or one bar
  per output).

Parameters
----------
y_pred : ndarray  
    Model predictions.  Accepts  
    * 1‑D ``(T,)``              – single trajectory, one output;  
    * 2‑D ``(N, T)``            – *N* trajectories, one output;  
    * 3‑D ``(N, O, T)``         – *N* trajectories, *O* outputs.  
    The final dimension is the temporal axis (*T ≥ 2* for PSS).

pss_values : float or ndarray, optional  
    Pre‑computed PSS value(s).  If supplied the helper skips internal
    calls to
    :func:`fusionlab.metrics.prediction_stability_score`.
metric_kws : dict, optional  
    Extra keyword arguments forwarded to the metric function.

kind : {{'scores_histogram', 'summary_bar'}},  
    default ``'summary_bar'``  
    Select the visualisation style.

output_idx : int, optional  
    Output dimension to plot when ``kind='scores_histogram'`` on
    multi‑output data.

hist_bins : int | sequence | str, default ``'auto'``  
hist_color : str, default ``'teal'``  
hist_edgecolor : str, default ``'black'``  
    Styling options for the histogram.

{params.base.figsize}
{params.base.title}
{params.base.xlabel}
{params.base.ylabel}
{params.base.bar_color}
{params.base.bar_width}
{params.base.score_annotation_format}
show_score : bool, default ``True``  
    Display the mean PSS on the histogram title.
{params.base.show_grid}
{params.base.grid_props}
{params.base.ax}
{params.base.verbose}
{params.base.kwargs}

Returns
-------
matplotlib.axes.Axes  
    Axes containing the stability visualisation.

Notes
-----
For one trajectory :math:`(\hat y_{{1}},\dots,\hat y_{{T}})` the stability
score is

.. math::

   \operatorname{{PSS}}
   \;=\;
   \frac{{1}}{{T-1}}\sum_{{t=2}}^{{T}}
   \bigl|\hat y_{{t}} - \hat y_{{t-1}}\bigr|.

The helper first reshapes ``y_pred`` to *(N, O, T)*, computes the
per‑trajectory scores, and then aggregates or plots them according to
``kind``.

Examples
--------
>>> import numpy as np, matplotlib.pyplot as plt
>>> from fusionlab.plot.evaluation import plot_prediction_stability
>>> rng = np.random.default_rng(0)
>>> preds = rng.normal(size=(200, 30))      # 200 series, 30 time steps
>>> plot_prediction_stability(
...     preds, kind='scores_histogram', figsize=(8, 4))
>>> plt.show()

See Also
--------
fusionlab.metrics.prediction_stability_score  
    Numeric implementation of PSS.
fusionlab.plot.evaluation.plot_time_weighted_metric  
    Time‑weighted MAE, accuracy, and interval‑score plots.

References
----------
.. [1] Hyndman, R.J. & Athanasopoulos, G. *Forecasting: Principles and
       Practice*, 3rd ed., OTexts, 2021 — section 2.6, “Stability”.
""".format(params=_param_docs)


def plot_qce_donut(
    df: pd.DataFrame,
    actual_col: str,
    quantile_cols: List[str],
    quantile_levels: List[float],
    metric_kws: Optional[Dict[str, Any]] = None,
    figsize: Tuple[float, float] = (8, 8),
    title: Optional[str] = "Quantile Calibration Error Contributions",
    colors: Optional[List[str]] = None,
    center_text_format: str = "Avg QCE:\n{:.4f}",
    segment_label_format: str = "{name}\n({value:.2f})", # {name}, {value}, {percent}
    startangle: float = 90,
    counterclock: bool = False,
    wedgeprops: Optional[Dict[str, Any]] = None,
    donut_width: float = 0.4, # Width of the donut ring
    value_annotations: bool=True, 
    # Legend and labels
    show_legend: bool = True,
    legend_title: Optional[str] = "Quantiles",
    legend_loc: str = "center left",
    legend_bbox_to_anchor: Tuple[float, float] = (0.95, 0.5),
    # Common params
    # Grid and labels
    show_grid: bool = True,
    grid_props: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    verbose: int = 0,
    **kwargs: Any # For future matplotlib extensions
) -> plt.Axes:
    """
    Visualizes Quantile Calibration Error (QCE) components as a donut chart.
    (Full docstring to be added later for detailed parameter explanation)
    """
    
    df, quantile_levels = _validate_qce_plot_inputs(
        df, actual_col=actual_col, 
        quantile_cols= quantile_cols, 
        quantile_levels= quantile_levels, 
        error_policy= 'raise'
    )

    y_true_np = df[actual_col].to_numpy(dtype=float)
    y_pred_quantiles_np = df[quantile_cols].to_numpy(dtype=float) # (N, Q)
    quantile_levels_np = np.array(quantile_levels, dtype=float) # (Q,)

    n_samples, n_quantiles = y_pred_quantiles_np.shape
    if n_samples != len(y_true_np):
        raise ValueError("Length mismatch: actual_col vs quantile_cols.")

    # --- 2. Handle NaNs and Sample Weights (from metric_kws) ---
    current_metric_kws = metric_kws or {}
    nan_policy = current_metric_kws.get('nan_policy', 'propagate')
    sample_weight = current_metric_kws.get('sample_weight', None)
    eps = current_metric_kws.get('eps', 1e-8)

    # Create a mask for NaNs across y_true and all relevant y_pred_quantiles
    # nan_mask_samples is 1D (n_samples,)
    nan_mask_true = np.isnan(y_true_np)
    nan_mask_preds = np.isnan(y_pred_quantiles_np).any(axis=1)
    combined_nan_mask_samples = nan_mask_true | nan_mask_preds

    y_t_calc = y_true_np
    y_p_calc = y_pred_quantiles_np
    s_weights_calc = sample_weight

    if np.any(combined_nan_mask_samples):
        if nan_policy == 'raise':
            raise ValueError("NaNs found in input data.")
        elif nan_policy == 'propagate':
            # If any NaN leads to an overall NaN result for donut chart
            warnings.warn(
                "NaNs found with nan_policy='propagate'. Donut chart "
                "may not be meaningful if miscalibrations become NaN."
            )
            # Calculations below will propagate NaNs
        elif nan_policy == 'omit':
            if verbose > 0:
                print("NaNs detected. Omitting samples with NaNs.")
            rows_to_keep = ~combined_nan_mask_samples
            if not np.any(rows_to_keep):
                if verbose > 0: warnings.warn("All samples omitted due to NaNs.")
                # Create an empty plot or return early
                if ax is None: _, ax = plt.subplots(figsize=figsize)
                ax.set_title(title or "QCE Donut (No Data)")
                return ax # type: ignore
            
            y_t_calc = y_true_np[rows_to_keep]
            y_p_calc = y_pred_quantiles_np[rows_to_keep]
            if s_weights_calc is not None:
                s_weights_calc = check_array(s_weights_calc, ensure_2d=False,
                                             dtype="numeric", force_all_finite=True)
                check_consistent_length(y_true_np, s_weights_calc)
                s_weights_calc = s_weights_calc[rows_to_keep]
    
    if y_t_calc.shape[0] == 0: # All samples omitted or original empty
        if verbose > 0: warnings.warn("No valid samples for QCE calculation.")
        if ax is None: _, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or "QCE Donut (No Valid Data)")
        return ax # type: ignore

    # --- 3. Calculate Per-Quantile Miscalibration ---
    indicators = (y_t_calc[:, np.newaxis] <= y_p_calc).astype(float)
    
    observed_proportions_q: np.ndarray
    if s_weights_calc is not None:
        sum_sw = np.sum(s_weights_calc)
        if sum_sw < eps:
            warnings.warn(f"Sum of sample_weight ({sum_sw}) < eps ({eps}). "
                          "Observed proportions may be unstable or NaN.")
            observed_proportions_q = np.full(n_quantiles, np.nan)
        else:
            # Weighted average, careful with NaNs in indicators if propagate was used
            # and some y_t_calc or y_p_calc were NaN
            # Assuming NaNs in indicators will propagate with np.average if weights are numbers
            observed_proportions_q = np.average(
                indicators, axis=0, weights=s_weights_calc
            )
    else:
        observed_proportions_q = np.nanmean(indicators, axis=0)

    miscalibrations_q = np.abs(
        observed_proportions_q - quantile_levels_np
    ) # Shape (Q,)

    # Handle cases where all miscalibrations are NaN or zero
    valid_miscal_mask = ~np.isnan(miscalibrations_q)
    if not np.any(valid_miscal_mask) or \
       np.sum(miscalibrations_q[valid_miscal_mask]) < eps :
        # If all are NaN or sum is effectively zero, donut chart is not meaningful
        warnings.warn(
            "All per-quantile miscalibrations are NaN or sum to near zero. "
            "Donut chart cannot be generated meaningfully."
        )
        if ax is None: _, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or "QCE Donut (No Miscalibration / Data Issues)")
        # Add text to center indicating the situation
        center_text = "No Miscalibration\nor Data Issues"
        if np.any(np.isnan(miscalibrations_q)):
             center_text = "Data Issues\n(NaNs)"
        elif np.sum(miscalibrations_q[valid_miscal_mask]) < eps :
             center_text = "Perfect Calibration\n(Avg QCE ~ 0)"
             
        ax.text(0.5, 0.5, center_text, ha='center', va='center',
                transform=ax.transAxes, fontsize='large')
        ax.set_aspect('equal') # Ensure circle
        ax.set_xticks([])
        ax.set_yticks([])
        return ax # type: ignore

    # Use quantile_cols for labels if available and match length
    segment_names = quantile_cols if len(quantile_cols) == n_quantiles \
                    else [f"q={q:.2f}" for q in quantile_levels_np]

    # Filter out NaN miscalibrations for plotting pie
    plot_miscalibrations = miscalibrations_q[valid_miscal_mask]
    plot_segment_names = [
        name for i, name in enumerate(segment_names) if valid_miscal_mask[i]
    ]
    plot_colors = None
    if colors:
        plot_colors = [
            c for i,c in enumerate(colors) if valid_miscal_mask[i]
        ] if len(colors) == n_quantiles else colors


    # --- 4. Plotting ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize) # type: ignore
    
    ax.set_title(title or "QCE Donut")

    wedges, texts, autotexts = ax.pie(
        plot_miscalibrations,
        labels=None, # Labels handled by legend or custom placement
        autopct=lambda pct: segment_label_format.format(
            name="", # Name will be in legend
            value= (pct/100.)*np.sum(plot_miscalibrations), # Value this segment represents
            percent=pct
        ) if value_annotations else None,
        startangle=startangle,
        counterclock=counterclock,
        colors=plot_colors,
        wedgeprops=wedgeprops or dict(width=donut_width, edgecolor='w')
    )

    # Center circle to make it a donut
    center_circle = plt.Circle((0,0), 1-donut_width, fc='white')
    ax.add_artist(center_circle)

    # Text in the center
    avg_qce = np.nanmean(miscalibrations_q) # Mean of non-NaN miscalibrations
    if not np.isnan(avg_qce):
        center_label = center_text_format.format(avg_qce)
        ax.text(0, 0, center_label, ha='center', va='center',
                fontsize='large', fontweight='bold')

    ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

    if show_legend:
        # Create legend with proper labels (quantile levels or names)
        # Use plot_segment_names which correspond to plot_miscalibrations
        legend_handles = []
        for i, w in enumerate(wedges):
            # Ensure color is RGBA for legend patch
            face_color = w.get_facecolor()
            if isinstance(face_color, tuple) and len(face_color) == 4:
                patch_color = face_color
            else: # Convert if it's a named color string or RGB
                patch_color = to_rgba(face_color) # type: ignore
            
            legend_handles.append(
                plt.Rectangle((0,0),1,1, facecolor=patch_color)
            )
        ax.legend(
            legend_handles,
            plot_segment_names,
            title=legend_title,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor
        )

    # Grid is not typically used for pie/donut charts
    if show_grid:
        if verbose > 0:
            warnings.warn("'show_grid' is True, but grids are not "
                          "standard for donut charts.")
        # ax.grid(**(grid_props or {})) # Usually off for pie
    
    # Remove default ticks and labels for pie chart axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax

def _validate_qce_plot_inputs(
    df: pd.DataFrame,
    actual_col: str,
    quantile_cols: List[str],
    quantile_levels: List[Real],
    error_policy: Literal['raise', 'warn', 'ignore'] = 'raise'
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Validates inputs for QCE plotting functions.

    Checks DataFrame type, column existence, and quantile properties.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing actual and predicted quantile values.
    actual_col : str
        Name of the column containing true observed values.
    quantile_cols : List[str]
        List of column names corresponding to predicted quantiles.
    quantile_levels : List[Real]
        List of nominal quantile levels (e.g., [0.1, 0.5, 0.9]).
    error_policy : {'raise', 'warn', 'ignore'}, default='raise'
        Policy for handling feature existence errors from `exist_features`.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        The validated input DataFrame and a NumPy array of validated
        and processed quantile levels.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame, or if `quantile_cols`
        or `quantile_levels` are not lists of the correct types.
    ValueError
        If columns are missing (and `error_policy='raise'`), if
        lengths of `quantile_cols` and `quantile_levels` mismatch,
        or if `quantile_levels` are not strictly between 0 and 1.
    """
    if not isinstance(df, pd.DataFrame):
        # For non-existence errors, _report_condition is better,
        # but for fundamental type errors, direct raise is common.
        # Match this with how _report_condition is used elsewhere.
        # For now, direct raise for type errors on df.
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    # Validate existence of actual_col
    exist_features(df, features=actual_col, error=error_policy)

    # Validate quantile_cols type and existence
    if not (isinstance(quantile_cols, list) and
            all(isinstance(qc, str) for qc in quantile_cols)):
        raise TypeError(
            "'quantile_cols' must be a list of strings."
        )
    if not quantile_cols: # Check if the list is empty
        raise ValueError("'quantile_cols' cannot be empty.")
        
    exist_features(df, features=quantile_cols, error=error_policy)

    # Validate and process quantile_levels
    # We expect quantile_levels to be numeric and in (0,1) for QCE.
    # validate_quantiles with mode='strict' ensures they are in [0,1].
    # An additional check for strict (0,1) is needed.
    try:
        # Use asarray=True to get a NumPy array for easier processing
        # Pass round_digits and dtype if they are relevant from a higher context
        # or use defaults within validate_quantiles.
        # For QCE, high precision is good, so float64 might be better if available.
        validated_levels_np = validate_quantiles(
            quantile_levels,
            asarray=True,
            mode="strict", # Ensures values are in [0,1] and numeric
            # Default round_digits and dtype from validate_quantiles used
        )
    except (TypeError, ValueError) as e:
        # Catch errors from validate_quantiles (e.g., non-numeric, out of [0,1])
        # Re-raise as ValueError for consistency, or let original error propagate
        raise ValueError(
            f"Validation of 'quantile_levels' failed: {e}"
        ) from e

    
    # After validate_quantiles(mode='strict'), levels are in [0,1].
    # Check for strict (0,1) as QCE is typically not for 0 or 1.
    
    are_all_values_in_bounds(
        validated_levels_np, bounds =(0, 1), closed='neither',
             message =(
            "All 'quantile_levels' must be strictly between 0 and 1 "
            "(exclusive of 0 and 1)."
        )
    )
    # Check for length consistency
    if len(quantile_cols) != len(validated_levels_np):
        raise ValueError(
            "Length of 'quantile_cols' ({}) must match the length of "
            "validated 'quantile_levels' ({}).".format(
                len(quantile_cols), len(validated_levels_np)
            )
        )
    
    # df is returned as is, as it's not modified by these checks,
    # but its columns are confirmed to exist.
    # validated_levels_np is returned as it's processed.
    return df, validated_levels_np


def plot_radar_scores(
    data_values: Optional[Union[List[Real], Dict[str, Real], Real]] = None,
    category_names: Optional[List[str]] = None,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    metric_functions: Optional[Union[
        MetricFunctionType, List[MetricFunctionType]]] = None,
    metric_kwargs_list: Optional[Union[
        Dict[str, Any], List[Dict[str, Any]]]] = None,
    normalize_values: bool = False,
    plot_target_type: Literal['metric'] = 'metric', # For future expansion
    # Plotting customizations
    figsize: Tuple[float, float] = (8, 8),
    title: Optional[str] = "Metric Scores Radar Plot",
    value_annotations: bool = True,
    annotation_format: str = "{:.2f}",
    fill_radar: bool = True,
    fill_alpha: float = 0.25,
    line_color: Optional[str] = None, # Auto-cycles if multiple lines
    line_width: float = 2,
    marker: Optional[str] = 'o',
    # Radial axis customization
    r_min: Optional[Real] = None,
    r_max: Optional[Real] = None,
    r_ticks_count: int = 5,
    # Grid and labels
    show_grid: bool = True,
    grid_props: Optional[Dict[str, Any]] = None,
    category_label_props: Optional[Dict[str, Any]] = None,
    value_label_props: Optional[Dict[str, Any]] = None,
    legend_label: Optional[str] = None, # For when plotting single entity
    ax: Optional[plt.Axes] = None,
    verbose: int = 0,
    **kwargs: Any # For future matplotlib extensions
) -> plt.Axes:
    """
    Generates a radar plot to visualize multiple scores or attributes.
    Primarily designed for comparing metric scores.
    """
    # --- 1. Input Processing and Score Calculation ---
    final_values_to_plot: np.ndarray
    final_category_names: List[str]

    if data_values is not None:
        if isinstance(data_values, dict):
            final_category_names = list(data_values.keys())
            final_values_to_plot = np.array(list(data_values.values()),
                                            dtype=float)
        else: # List, scalar, or array-like
            # Use column_manager to handle scalar or list-like
            managed_values = columns_manager(
                data_values, # force_array=True, to_list=False
            )
            if hasattr(managed_values, '__iter__'): 
                managed_values = np.array ( managed_values)
                
            if managed_values is None or managed_values.size == 0:
                raise ValueError(
                    "If 'data_values' is not a dict, it must not be empty "
                    "after processing."
                )
            final_values_to_plot = managed_values.astype(float)

            if category_names is not None:
                if len(category_names) < len(final_values_to_plot):
                    if verbose > 0:
                        warnings.warn(
                            "Length of 'category_names' is less than "
                            "'data_values'. Auto-generating remaining names."
                        )
                    base_name = f"{plot_target_type}_"
                    num_missing = len(final_values_to_plot) - len(category_names)
                    auto_names = [f"{base_name}{i+len(category_names)+1}"
                                  for i in range(num_missing)]
                    final_category_names = category_names + auto_names
                elif len(category_names) > len(final_values_to_plot):
                     if verbose > 0:
                        warnings.warn(
                            "Length of 'category_names' is greater than "
                            "'data_values'. Truncating 'category_names'."
                        )
                     final_category_names = category_names[:len(final_values_to_plot)]
                else:
                    final_category_names = category_names
            else: # Auto-generate all names
                base_name = f"{plot_target_type}_"
                final_category_names = [f"{base_name}{i+1}" for i in
                                        range(len(final_values_to_plot))]
    elif plot_target_type == 'metric':
        if y_true is None or y_pred is None:
            raise ValueError(
                "If 'data_values' is None and 'plot_target_type' is 'metric',"
                " 'y_true' and 'y_pred' must be provided."
            )
        y_t = check_array(y_true, ensure_2d=False, force_all_finite=True,
                          dtype="numeric")
        y_p = check_array(y_pred, ensure_2d=False, force_all_finite=True,
                          dtype="numeric")
        check_consistent_length(y_t, y_p)

        # Default metric functions if none provided
        if metric_functions is None:
            metric_funcs_to_use: List[MetricFunctionType] = [
                mean_absolute_error,
                lambda yt, yp, **kws: np.sqrt(mean_squared_error(yt, yp, **kws)), # RMSE
                mean_absolute_percentage_error
            ]
            default_names = ["MAE", "RMSE", "MAPE"]
            # Ensure default kwargs for RMSE's mean_squared_error if any
            default_metric_kws_list: List[Dict] = [{}, {'squared': False}, {}]
            # User can override these defaults via metric_kwargs_list
            if metric_kwargs_list is None:
                metric_kws_list_proc = default_metric_kws_list
            elif isinstance(metric_kwargs_list, dict): # Apply to all defaults
                metric_kws_list_proc = [
                    {**dkw, **metric_kwargs_list} for dkw in default_metric_kws_list
                ]
            elif isinstance(metric_kwargs_list, list) and \
                 len(metric_kwargs_list) == len(metric_funcs_to_use):
                metric_kws_list_proc = [
                    {**dkw, **ukw} for dkw, ukw in zip(
                        default_metric_kws_list, metric_kwargs_list) # type: ignore
                ]
            else:
                raise ValueError(
                    "Invalid 'metric_kwargs_list' for default functions.")

            final_category_names = category_names if category_names and \
                len(category_names)==len(default_names) else default_names

        elif callable(metric_functions): # Single function
            metric_funcs_to_use = [metric_functions]
            metric_kws_list_proc = [metric_kwargs_list or {}] \
                if isinstance(metric_kwargs_list, dict) or metric_kwargs_list is None \
                else metric_kwargs_list # Should be list of one dict
            
            if category_names:
                final_category_names = category_names
            else:
                func_name = getattr(metric_functions, '__name__', 'metric_1')
                final_category_names = [func_name]
        
        elif isinstance(metric_functions, list): # List of functions
            metric_funcs_to_use = metric_functions
            if metric_kwargs_list is None:
                metric_kws_list_proc = [{} for _ in metric_funcs_to_use]
            elif isinstance(metric_kwargs_list, list) and \
                 len(metric_kwargs_list) == len(metric_funcs_to_use):
                metric_kws_list_proc = metric_kwargs_list
            else:
                raise ValueError(
                    "'metric_kwargs_list' must be a list of dicts matching "
                    "'metric_functions' length, or None."
                )
            if category_names and len(category_names) == len(metric_funcs_to_use):
                final_category_names = category_names
            elif category_names is None:
                final_category_names = [
                    getattr(f, '__name__', f"{plot_target_type}_{i+1}")
                    for i, f in enumerate(metric_funcs_to_use)
                ]
            else:
                raise ValueError(
                    "Length of 'category_names' must match 'metric_functions'."
                )
        else:
            raise TypeError(
                "'metric_functions' must be None, a callable, or list of callables.")

        # Compute scores
        computed_scores = []
        for func, kws_for_func in zip(metric_funcs_to_use, metric_kws_list_proc):
            valid_kws = _get_valid_kwargs(func, kws_for_func)
            try:
                score = func(y_t, y_p, **valid_kws)
                computed_scores.append(score)
            except Exception as e:
                warnings.warn(
                    "Error computing metric "
                    f"{getattr(func,'__name__','unknown')}:"
                    f" {e}. Skipping."
                )
                computed_scores.append(np.nan)
        final_values_to_plot = np.array(
            computed_scores, dtype=float
            )

    else:
        raise ValueError(
            f"Unsupported 'plot_target_type': {plot_target_type}. "
            "Currently only 'metric' is supported."
        )

    if final_values_to_plot.ndim > 1:
        # This implies multiple sets of values (e.g., from multi-output metrics)
        # The current design plots one radar line.
        # For now, require scalar values per category.
        raise ValueError(
            "Each category for the radar plot must correspond to a single "
            "scalar value. Received multi-dimensional values."
        )
    if len(final_values_to_plot) != len(final_category_names):
        raise ValueError(
            "Mismatch between number of values to plot and category names. "
            f"Got {len(final_values_to_plot)} values and "
            f"{len(final_category_names)} names."
        )
    
    num_vars = len(final_category_names)
    if num_vars < 3:
        warnings.warn(
            "Radar plots are typically used for 3 or more categories. "
            "Consider a bar chart for fewer categories."
        )
        # Fallback or proceed? For now, proceed if user insists.
        if num_vars == 0:
            if ax is None: _, ax = plt.subplots(figsize=figsize)
            ax.set_title(title or "Radar Plot (No Data)")
            if show_grid: ax.grid(**(grid_props or {}))
            return ax


    # --- 2. Normalization (if requested) ---
    plot_values = final_values_to_plot.copy()
    original_values_for_annotation = final_values_to_plot.copy()
    
    # Handle NaNs before normalization or plotting
    nan_mask = np.isnan(plot_values)
    if np.all(nan_mask): # All values are NaN
        warnings.warn("All values for radar plot are NaN. Plot will be empty.")
        # Proceed to draw empty radar axes
    
    # Replace NaNs with a value for plotting structure, but they won't be "seen"
    # if we are careful with plotting (e.g., don't connect across NaNs).
    # Or, for simplicity in radar, they might be plotted at 0 or min.
    # For now, let's make them 0 for structure, but annotations will show NaN.
    plot_values_for_structure = np.nan_to_num(plot_values, nan=0.0)


    if normalize_values:
        # Min-max scale to [0, 1] for better shape comparison
        # Only use non-NaN values for finding min/max
        valid_plot_vals = plot_values[~nan_mask]
        if valid_plot_vals.size > 0:
            min_val = np.min(valid_plot_vals)
            max_val = np.max(valid_plot_vals)
            if max_val - min_val > 1e-8: # Avoid division by zero if all same
                # Apply scaling to non-NaN original values
                scaled_non_nan = (valid_plot_vals - min_val) / (max_val - min_val)
                # Put scaled values back, keep NaNs as NaNs in plot_values
                # plot_values will be used for plotting line/fill
                # original_values_for_annotation keeps original scale for text
                temp_scaled_values = np.full_like(plot_values, np.nan)
                temp_scaled_values[~nan_mask] = scaled_non_nan
                plot_values = temp_scaled_values
                plot_values_for_structure = np.nan_to_num(plot_values, nan=0.0)

                if verbose > 0:
                    print(f"Values normalized. Original range: [{min_val:.2f}, {max_val:.2f}].")
            elif verbose > 0: # All valid values are the same
                warnings.warn(
                    "All valid values are identical after NaN handling; "
                    "normalization to [0,1] range results in all zeros or ones. "
                    "Radar shape might not be informative."
                )
                # Set all to 0.5 to show a regular polygon if all same
                plot_values[~nan_mask] = 0.5 
                plot_values_for_structure = np.nan_to_num(plot_values, nan=0.0)

        else: # All values were NaN
            if verbose > 0: warnings.warn("All values are NaN, cannot normalize.")
        
        # If normalized, radial ticks should be 0 to 1
        if r_min is None: r_min = 0
        if r_max is None: r_max = 1
        
    # --- 3. Radar Plot Creation ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True)) # type: ignore
    elif not hasattr(ax, 'plot'): # Check if it's a Matplotlib Axes
         raise TypeError("`ax` must be a Matplotlib Axes object.")
    # If ax is provided, assume it's already polar if needed, or make it so.
    # This is tricky. Standard practice is to create polar on a figure.
    # For simplicity, if ax is passed, we'll try to use it as is.
    # User should ensure it's a polar Axes if passing one.

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Make the plot close by appending the first value and angle
    plot_data_closed = np.concatenate(
        (plot_values_for_structure, [plot_values_for_structure[0]]))
    angles_closed = angles + [angles[0]]
    original_values_closed = np.concatenate( # noqa
        (original_values_for_annotation, [original_values_for_annotation[0]])
    ) 
    nan_mask_closed = np.concatenate((nan_mask, [nan_mask[0]]))


    # Plotting the data line
    # To handle NaNs gracefully (not connecting across them):
    # Plot segments between non-NaN points.
    segments = np.ma.masked_where( # noqa
        nan_mask_closed, plot_data_closed).tolist()
    
    # For a single radar line, line_color can be a string
    # If comparing multiple entities on same radar later, this would need cycling
    current_line_color = line_color if line_color is not None else \
                         next(ax._get_lines.prop_cycler)['color'] # type: ignore

    ax.plot(angles_closed, plot_data_closed, color=current_line_color,
            linewidth=line_width, marker=marker if marker else '',
            label=legend_label if legend_label else plot_target_type.title())
    
    if fill_radar:
        ax.fill(angles_closed, plot_data_closed, color=current_line_color,
                alpha=fill_alpha)

    # Category labels
    ax.set_xticks(angles)
    category_label_defaults = {'size': 'medium'}
    category_label_final_props = {
        **category_label_defaults,
        **(category_label_props or {})
    }
    ax.set_xticklabels(final_category_names, **category_label_final_props)

    # Radial axis (y-axis in polar)
    if r_min is not None or r_max is not None:
        ax.set_ylim(r_min, r_max)
    
    # Set radial ticks. MaxNLocator helps get "nice" ticks.
    # Get current ylim if not set by user, to inform tick locator
    # from matplotlib.ticker import MaxNLocator

    current_r_lim = ax.get_ylim()
    
    # Try your preferred prune option, otherwise fall back
    try:
        locator = MaxNLocator(nbins=r_ticks_count, prune='min')
    except ValueError:
        # 'min' isn’t supported; fall back to no pruning (or use 'lower'/'upper' as you see fit)
        locator = MaxNLocator(nbins=r_ticks_count)
    
    # Now generate your tick locations
    r_tick_locs = locator.tick_values(current_r_lim[0], current_r_lim[1])


    # current_r_lim = ax.get_ylim()
    # r_tick_locs = MaxNLocator(
    #     nbins=r_ticks_count, prune='min' # 'min' to ensure 0 is often a tick
    # ).tick_values(current_r_lim[0], current_r_lim[1])
    
    # Filter out ticks outside the explicit r_min/r_max if they were set
    if r_min is not None: 
        r_tick_locs = r_tick_locs[r_tick_locs >= r_min]
    if r_max is not None: 
        r_tick_locs = r_tick_locs[r_tick_locs <= r_max]
    # Ensure 0 is a tick if within range, and if r_min is not set above 0
    if (r_min is None or r_min <=0) and 0 not in r_tick_locs and current_r_lim[0] <=0:
        r_tick_locs = np.unique(np.sort(np.concatenate(([0], r_tick_locs))))

    ax.set_yticks(r_tick_locs)

    value_label_defaults = {'size': 'small'}
    value_label_final_props = {
        **value_label_defaults,
        **(value_label_props or {})
    }
    ax.set_yticklabels(
        [annotation_format.format(tick) for tick in r_tick_locs],
        **value_label_final_props
    )


    # Value annotations on the radar points
    if value_annotations:
        for i, (angle, val_plot, val_orig) in enumerate(zip(
            angles, plot_values, original_values_for_annotation
            )):
            if not np.isnan(val_plot) and not np.isnan(val_orig):
                # Use original value for annotation text
                annotation_text = annotation_format.format(val_orig)
                ax.text(angle, val_plot, annotation_text,
                        ha='center', va='bottom' if val_plot >=0 else 'top',
                        fontsize=category_label_final_props.get('size', 'small'), # type: ignore
                        bbox=dict(facecolor='white', alpha=0.5,
                                  edgecolor='none', pad=1.0)
                       ) # Add background for readability

    if title:
        ax.set_title(title, va='bottom',
                     fontdict={'fontsize': plt.rcParams['axes.titlesize'], # type: ignore
                               'fontweight': plt.rcParams['axes.titleweight']}) # type: ignore
    
    if show_grid:
        grid_final_props = grid_props if grid_props is not None \
            else {'linestyle': '--', 'alpha': 0.7, 'linewidth':0.5}
        ax.grid(**grid_final_props)
    else:
        ax.grid(False)
        
    if legend_label: # If a single entity is plotted, legend might be useful
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    return ax

def _calculate_qce_miscalibrations(
    y_true_period: np.ndarray,      # (N_period,)
    y_pred_q_period: np.ndarray,  # (N_period, Q)
    quantile_levels: np.ndarray,  # (Q,)
    sample_weight_period: Optional[np.ndarray], # (N_period,)
    eps: float
) -> np.ndarray:
    """Calculates per-quantile miscalibrations for a given period."""
    indicators = (
        y_true_period[:, np.newaxis] <= y_pred_q_period
    ).astype(float) # (N_period, Q)

    observed_proportions_q: np.ndarray
    if sample_weight_period is not None:
        sum_sw = np.sum(sample_weight_period)
        if sum_sw < eps:
            return np.full(quantile_levels.shape[0], np.nan)
        
        # Weighted average, handling NaNs from inputs if they exist
        # NaNs in indicators should result from NaNs in y_true or y_pred_q
        temp_props_list = []
        for q_idx in range(quantile_levels.shape[0]):
            valid_inds_q = indicators[:, q_idx]
            finite_mask_q = ~np.isnan(valid_inds_q)
            if np.any(finite_mask_q):
                current_weights = sample_weight_period[finite_mask_q]
                sum_finite_weights = np.sum(current_weights)
                if sum_finite_weights >= eps:
                    prop = np.sum(
                        valid_inds_q[finite_mask_q] * current_weights
                    ) / sum_finite_weights
                    temp_props_list.append(prop)
                else:
                    temp_props_list.append(np.nan)
            else:
                temp_props_list.append(np.nan)
        observed_proportions_q = np.array(temp_props_list)
    else:
        observed_proportions_q = np.nanmean(indicators, axis=0)

    miscalibrations_q = np.abs(
        observed_proportions_q - quantile_levels
    )
    return miscalibrations_q

# --- Main Plotting Function ---
# make actual_col to be optional set to None,
# because some metrics works with the prediction only. 
# also if metric is not provided and actual_col is None, 
# the you plot the uncertainty distribution of quantile every years ( like average) 
# and plot rather than qce instead. 
# so revise 

# --- Main Plotting Function ---
def plot_nested_quantiles(
    df: pd.DataFrame,
    quantile_cols: List[str],
    quantile_levels: List[Real],
    actual_col: Optional[str] = None, # Now optional
    dt_col: Optional[str] = None,
    periods: Optional[List[Any]] = None, # Renamed
    metric_func: Optional[MetricFunctionType] = None,
    metric_kws: Optional[Dict[str, Any]] = None,
    # Plotting customizations
    figsize: Tuple[float, float] = (10, 10),
    title: Optional[str] = None, # Default title set based on mode
    colors: Optional[List[str]] = None,
    show_center_text: bool = True,
    center_text_format: str = "Avg:\n{value:.3f}",
    segment_label_format: str = "{name}\n{value:.2f}",
    show_segment_labels: bool = True,
    startangle: float = 90,
    counterclock: bool = False,
    wedgeprops: Optional[Dict[str, Any]] = None, # User can set gap here
    donut_width: float = 0.3,
    donut_ring_spacing: float = 0.05,
    donut_base_radius: float = 0.3, 
    segment_explode: Optional[Union[float, List[float]]] = None,
    # Legend
    show_overall_legend: bool = True,
    legend_title: Optional[str] = "Quantiles",
    legend_loc: str = "center left",
    legend_bbox_to_anchor: Tuple[float, float] = (1.05, 0.5),
    # Common params
    show_grid: bool=False, # for API consistency 
    grid_props: dict =None, # for API consistency 
    ax: Optional[plt.Axes] = None,
    verbose: int = 0,
    **kwargs: Any
) -> plt.Axes:
    """
    Visualizes quantile-based metrics or average quantile values
    over periods as nested donut charts.
    (Full docstring to be expanded later)
    """
    # --- 1. Determine Plotting Mode ---
    plot_mode: Literal['qce', 'custom_metric', 'avg_quantiles']
    if actual_col is None and metric_func is None:
        plot_mode = 'avg_quantiles'
        default_title = "" #"Average Predicted Quantile Values by Period"
    elif metric_func is not None:
        plot_mode = 'custom_metric'
        func_name = getattr(metric_func, '__name__', 'Custom Metric')
        default_title = f"{func_name} Evolution Over Periods"
    else: # actual_col is provided and metric_func is None
        plot_mode = 'qce'
        default_title = "Quantile Calibration Error by Period"

    final_title = title if title is not None else default_title

    # --- 2. Input Validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    
    cols_to_check_existence = quantile_cols[:]
    if actual_col: # Only check if provided
        cols_to_check_existence.append(actual_col)
    if dt_col:
        cols_to_check_existence.append(dt_col)
    exist_features(df, features=cols_to_check_existence, error='raise')

    if not (isinstance(quantile_cols, list) and
            all(isinstance(qc, str) for qc in quantile_cols)):
        raise TypeError("'quantile_cols' must be a list of strings.")
    if not quantile_cols:
        raise ValueError("'quantile_cols' cannot be empty.")
        
    try:
        q_levels_np = validate_quantiles(
            quantile_levels, asarray=True, mode="strict"
        )
        # For QCE and custom metrics expecting strict (0,1) quantiles
        if plot_mode != 'avg_quantiles': # Avg quantiles don't impose this
            if not np.all((q_levels_np > 0) & (q_levels_np < 1)):
                raise ValueError(
                    "All 'quantile_levels' must be strictly in (0,1) "
                    "for metric calculation."
                )
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Validation of 'quantile_levels' failed: {e}"
        ) from e

    if len(quantile_cols) != len(q_levels_np):
        raise ValueError(
            "Length of 'quantile_cols' must match 'quantile_levels'."
        )

    # --- 3. Data Preparation ---
    metric_kws = metric_kws or {}
    eps = metric_kws.get('eps', 1e-8)
    sample_weight_col = metric_kws.get('sample_weight_col', None)
    if sample_weight_col:
        exist_features(df, features=sample_weight_col, error='raise')

    if dt_col:
        unique_periods = sorted(df[dt_col].unique())
        periods_to_use = periods # Use renamed parameter
        if periods_to_use is not None:
            periods_val = [p for p in periods_to_use if p in unique_periods]
            if not periods_val:
                raise ValueError(
                    "None of the specified 'periods' found in data."
                )
        else:
            periods_val = unique_periods
    else: 
        periods_val = ["Overall"] 

    num_periods = len(periods_val)
    if num_periods == 0:
        warnings.warn("No periods to plot.")
        if ax is None: _, ax = plt.subplots(figsize=figsize)
        ax.set_title(final_title + " (No Data)")
        return ax # type: ignore

    # --- 4. Calculate Values per Period and Quantile ---
    period_values_dict: Dict[Any, np.ndarray] = {} 
    
    for period_name in periods_val:
        period_df_slice = df if dt_col is None else \
                          df[df[dt_col] == period_name]
        if period_df_slice.empty:
            period_values_dict[period_name] = np.full(len(q_levels_np), np.nan)
            if verbose > 0:
                warnings.warn(
                    f"No data for period '{period_name}'. Values set to NaN."
                )
            continue

        y_p_q_period = period_df_slice[quantile_cols].to_numpy(dtype=float)
        s_weights_period = period_df_slice[sample_weight_col].to_numpy(dtype=float) \
                           if sample_weight_col else None
        
        nan_policy_metric = metric_kws.get('nan_policy', 'propagate')
        
        # NaN handling for y_p_q_period and potentially y_t_period
        # This needs to be done carefully based on the plot_mode
        
        current_period_values: np.ndarray

        if plot_mode == 'avg_quantiles':
            # NaNs in y_p_q_period for this mode
            nan_mask_preds_period = np.isnan(y_p_q_period) # (N_period, Q)
            if np.any(nan_mask_preds_period):
                if nan_policy_metric == 'raise':
                    raise ValueError(
                        f"NaNs found in quantile_cols for period '{period_name}'."
                    )
                elif nan_policy_metric == 'omit':
                    # Omit rows if ANY quantile in that row is NaN
                    rows_with_nan_in_preds = nan_mask_preds_period.any(axis=1)
                    keep_rows = ~rows_with_nan_in_preds
                    if not np.any(keep_rows):
                        current_period_values = np.full(len(q_levels_np), np.nan)
                    else:
                        y_p_q_period_clean = y_p_q_period[keep_rows]
                        s_weights_period_clean = s_weights_period[keep_rows] \
                            if s_weights_period is not None else None
                        if s_weights_period_clean is not None and \
                           np.sum(s_weights_period_clean) < eps:
                            current_period_values = np.full(len(q_levels_np), np.nan)
                        else:
                            current_period_values = np.average(
                                y_p_q_period_clean, axis=0,
                                weights=s_weights_period_clean
                            )
                else: # propagate
                    current_period_values = np.nanmean(y_p_q_period, axis=0)
            else: # No NaNs in y_p_q_period
                if s_weights_period is not None and np.sum(s_weights_period) < eps:
                     current_period_values = np.full(len(q_levels_np), np.nan)
                else:
                    current_period_values = np.average(
                        y_p_q_period, axis=0, weights=s_weights_period
                    )
        else: # 'qce' or 'custom_metric'
            y_t_period = period_df_slice[actual_col].to_numpy(dtype=float) # type: ignore
            nan_mask_true_period = np.isnan(y_t_period)
            nan_mask_preds_period_any = np.isnan(y_p_q_period).any(axis=1)
            combined_nan_samples = nan_mask_true_period | nan_mask_preds_period_any

            if np.any(combined_nan_samples):
                if nan_policy_metric == 'raise':
                    raise ValueError(
                        f"NaNs found in data for period '{period_name}'."
                    )
                elif nan_policy_metric == 'omit':
                    keep_rows = ~combined_nan_samples
                    if not np.any(keep_rows):
                        current_period_values = np.full(len(q_levels_np), np.nan)
                    else:
                        y_t_period = y_t_period[keep_rows]
                        y_p_q_period = y_p_q_period[keep_rows]
                        if s_weights_period is not None:
                            s_weights_period = s_weights_period[keep_rows]
                # If 'propagate', handled by metric func or helper
            
            if y_t_period.shape[0] == 0:
                 current_period_values = np.full(len(q_levels_np), np.nan)
            elif plot_mode == 'qce':
                current_period_values = _calculate_qce_miscalibrations(
                    y_t_period, y_p_q_period, q_levels_np,
                    s_weights_period, eps
                )
            else: # 'custom_metric'
                try:
                    # Custom metric func signature might vary.
                    # Assuming it can handle these inputs or uses _get_valid_kwargs.
                    # For simplicity, pass all relevant, let metric handle.
                    metric_args = {
                        'y_true_period': y_t_period,
                        'y_pred_q_period': y_p_q_period,
                        'quantile_levels': q_levels_np,
                        'sample_weight_period': s_weights_period,
                        'eps': eps
                    }
                    # Allow metric_kws to override these if needed
                    final_metric_args = {**metric_args, **(metric_kws or {})}
                    cleaned_metric_args = _get_valid_kwargs(
                        metric_func, final_metric_args) # type: ignore
                    current_period_values = metric_func(
                        **cleaned_metric_args) # type: ignore
                except Exception as e:
                    warnings.warn(
                        f"Error calling custom metric_func for period "
                        f"'{period_name}': {e}. Values set to NaN."
                    )
                    current_period_values = np.full(len(q_levels_np), np.nan)
        
        period_values_dict[period_name] = current_period_values

    # --- 5. Plotting Setup ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize) # type: ignore
    
    ax.set_title(final_title)
    ax.axis('equal') 
    ax.set_xticks([])
    ax.set_yticks([])

    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colors = prop_cycle.by_key()['color']
        plot_colors = [default_colors[i % len(default_colors)]
                       for i in range(len(q_levels_np))]
    elif len(colors) < len(q_levels_np):
        warnings.warn("Not enough colors for quantiles. Colors will cycle.")
        plot_colors = [colors[i % len(colors)] for i in range(len(q_levels_np))]
    else:
        plot_colors = colors[:len(q_levels_np)]

    if segment_explode is None:
        explodes = [0] * len(q_levels_np)
    elif isinstance(segment_explode, float):
        explodes = [segment_explode] * len(q_levels_np)
    elif isinstance(segment_explode, list) and \
         len(segment_explode) == len(q_levels_np):
        explodes = segment_explode
    else:
        warnings.warn("Invalid 'segment_explode'. Using no explosion.")
        explodes = [0] * len(q_levels_np)

    # Default wedge properties for gap
    base_wedgeprops = {'edgecolor': 'white', 'linewidth': 1.5}
    if wedgeprops: # User can override/add to this
        base_wedgeprops.update(wedgeprops)

    # --- 6. Plotting Nested Donuts ---
    current_outer_radius = donut_base_radius + \
                           num_periods * donut_width + \
                           max(0, num_periods - 1) * donut_ring_spacing
    
    legend_items_overall = [] 

    for i, period_name in enumerate(periods_val): # Use validated periods_val
        values_for_period = period_values_dict[period_name]
        valid_scores_mask = ~np.isnan(values_for_period)

        if not np.any(valid_scores_mask):
            if verbose > 0:
                warnings.warn(
                    f"All values for period '{period_name}' are NaN. "
                    "Skipping donut for this period."
                )
            current_outer_radius -= (donut_width + donut_ring_spacing)
            continue

        plot_data_period = values_for_period[valid_scores_mask]
        # Use original quantile_cols for names if plot_mode is 'avg_quantiles'
        # and lengths match, otherwise generate from q_levels_np
        if plot_mode == 'avg_quantiles' and \
           len(quantile_cols) == len(q_levels_np):
            period_segment_names_all = quantile_cols
        else:
            period_segment_names_all = [f"q={q:.2f}" for q in q_levels_np]
             
        period_segment_names = [ # noqa
            period_segment_names_all[j] for j, keep in enumerate(valid_scores_mask) if keep
        ]
        period_colors = [
            plot_colors[j] for j, keep in enumerate(valid_scores_mask) if keep
        ]
        period_explodes = [
            explodes[j] for j, keep in enumerate(valid_scores_mask) if keep
        ]

        sum_plot_data = np.sum(plot_data_period)
        # For pie chart, values should be positive. If metric can be negative,
        # this visualization might not be ideal or needs transformation.
        # Assuming metric scores (like QCE) or avg quantiles are non-negative.
        if sum_plot_data < eps and not np.all(plot_data_period < eps):
             if verbose > 0:
                warnings.warn(
                    f"Sum of values for period '{period_name}' is near zero, "
                    "but individual values are not all zero. "
                    "Donut segments may not be visible or meaningful."
                )
        
        # If all plot_data_period are zero or negative, pie chart is problematic
        if np.all(plot_data_period <= eps) and plot_data_period.size > 0 :
            if verbose > 0:
                warnings.warn(
                    f"All values for period '{period_name}' are zero or negative. "
                    "Drawing empty/minimal donut ring."
                )
            # Draw a simple ring to indicate the period without segments
            ring = plt.Circle((0,0), current_outer_radius - donut_width/2,
                              width=donut_width, color='lightgray', fill=False,
                              linestyle='--', ec='gray')
            ax.add_artist(ring)
            # Add period label to the ring
            angle_for_label = startangle + \
                (counterclock * 2 -1) * (i * 360/num_periods) # Distribute labels
            text_x = (current_outer_radius - donut_width/2) * \
                     np.cos(np.deg2rad(angle_for_label))
            text_y = (current_outer_radius - donut_width/2) * \
                     np.sin(np.deg2rad(angle_for_label))
            ax.text(text_x, text_y, str(period_name), ha='center', va='center',
                    fontsize='small', color='dimGray')

        elif plot_data_period.size > 0 : # Ensure there's data to plot
            wedges, *texts_autotexts = ax.pie(
                plot_data_period,
                radius=current_outer_radius,
                labels=None, 
                autopct=(lambda pct: segment_label_format.format(
                            name="", 
                            value=(pct/100.)*sum_plot_data if sum_plot_data > eps else 0,
                            percent=pct)
                        ) if show_segment_labels and sum_plot_data > eps else None,
                startangle=startangle,
                counterclock=counterclock,
                colors=period_colors,
                explode=period_explodes,
                wedgeprops={**base_wedgeprops, 'width': donut_width} # type: ignore
            )
            
        
        if i == 0 and show_overall_legend and plot_data_period.size > 0:
            # Create legend items based on the *original* full set of quantiles
            # and their assigned colors, to ensure consistency.
            for q_idx, q_level_val in enumerate(q_levels_np):
                # Use original quantile_cols if available and matches, else q=...
                leg_name = quantile_cols[q_idx] if \
                    len(quantile_cols) == len(q_levels_np) else f"q={q_level_val:.2f}"
                legend_items_overall.append(
                    (plt.Rectangle((0,0),1,1, facecolor=plot_colors[q_idx]), leg_name)
                )
        
        if show_center_text and plot_data_period.size > 0:
            # For the innermost donut, show overall average for that period
            avg_value_period = np.nanmean(values_for_period) 
            if not np.isnan(avg_value_period):
                if i == num_periods -1 : 
                     center_text_val = center_text_format.format(value=avg_value_period)
                     if plot_mode == 'avg_quantiles' and dt_col is None: # Single overall donut
                         center_text_val = "Avg. Quantile\nValues"
                     elif plot_mode == 'avg_quantiles':
                          center_text_val = f"{period_name}\nAvg. Values"

                     ax.text(0, 0, center_text_val,
                             ha='center', va='center', fontsize='medium',
                             fontweight='bold',
                             path_effects=kwargs.get('center_text_path_effects', None))
                elif verbose > 1 and dt_col is not None: 
                    print(
                        f"Period '{period_name}' Avg Score/Value: {avg_value_period:.3f}"
                    )
        current_outer_radius -= (donut_width + donut_ring_spacing)

    # --- 7. Final Touches (Legend, Grid) ---
    if show_overall_legend and legend_items_overall:
        # Remove duplicate legend items if colors/names were cycled
        unique_legend_items = []
        seen_labels = set()
        for handle, label in legend_items_overall:
            if label not in seen_labels:
                unique_legend_items.append((handle, label))
                seen_labels.add(label)
        
        if unique_legend_items:
            handles, labels = zip(*unique_legend_items)
            ax.legend(handles, labels, title=legend_title,
                      loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)

    if show_grid:
        if verbose > 0:
            warnings.warn(
                "'show_grid' for donut chart might not be conventional."
            )
        ax.grid(**(grid_props or {'linestyle': ':', 'alpha': 0.3}))
    else:
        ax.grid(False)
        
    return ax

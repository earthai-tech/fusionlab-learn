# -*- coding: utf‑8 -*-
# License: BSD‑3‑Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Convenience helpers for **quick‑look** visual inspection of model
forecasts.
"""

from __future__ import annotations

import os
import re
import logging 
import warnings
from typing import Any, List, Optional, Tuple, Union, Dict, Callable 

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

from ..core.checks import ( 
    check_non_emptiness, 
    check_spatial_columns, 
    is_in_if
)
from ..core.handlers import columns_manager
from ..utils.forecast_utils import ( 
    format_forecast_dataframe, 
    get_value_prefixes, 
    get_value_prefixes_in, 
    detect_forecast_type, 
    get_step_names 
)
from ..utils.generic_utils import ( 
    _coerce_dt_kw, 
    get_actual_column_name,
    vlog
)
from ..utils.validator import ( 
    assert_xy_in, is_frame, 
    validate_positive_integer 
)

__all__= [
    "forecast_view",
    "plot_forecast_by_step", 
    "plot_forecasts", 
    "visualize_forecasts", 
 ]

def plot_forecast_by_step(
    df: pd.DataFrame,
    value_prefixes: Optional[List[str]] = None,
    kind: str = "dual",
    steps: Optional[int] =None, # step to plot 
    step_names: Optional[Union[Dict[int, str], List[str]]] = None,
    spatial_cols: Optional[Tuple[str, str]] = None,
    max_cols: Union[int, str] = "auto",
    cmap: str = "viridis",
    cbar: str = "uniform",
    axis_off: bool = False,
    show_grid: bool = True,
    grid_props: Optional[Dict] = None,
    figsize: Optional[Tuple[float, float]] = None,
    savefig: Optional[str] = None,
    save_fmts: Union[str, List[str]] = ".png",
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    verbose: int = 1,
):
    """Plots forecast data, organizing subplots by forecast step.

    This function creates a grid of plots to visualize forecast
    data from a long-format DataFrame. Each row in the grid
    corresponds to a unique `forecast_step`.

    It robustly handles both spatial scatter plots (if `spatial_cols`
    are provided and found) and temporal line plots (as a fallback),
    making it versatile for different types of forecast data.

    Parameters
    ----------
    df : pd.DataFrame
        Input long-format DataFrame. Must contain a 'forecast_step'
        column and value columns (e.g., 'subsidence_q50').
    value_prefixes : list of str, optional
        The base names of metrics to plot (e.g., ``['subsidence']``).
        If ``None``, prefixes are auto-detected from column names.
        A separate figure is generated for each prefix.
    kind : {'dual', 'pred_only'}, default 'dual'
        Determines what to plot:
        - ``'dual'``: Plots both actual values and predictions
          side-by-side for comparison.
        - ``'pred_only'``: Plots only the predicted values.
    steps : int or list of int, optional
        A specific forecast step or list of steps to visualize. If
        ``None``, all unique steps found in the DataFrame are plotted.
    step_names : dict or list, optional
        Custom labels for the forecast steps in plot titles.
        - If a dict, maps step numbers to names (e.g., ``{1: 'Year 1'}``).
        - If a list, maps step index to names (e.g., ``['Y1', 'Y2']``,
          where index 0 corresponds to step 1).
    spatial_cols : tuple of str, optional
        Tuple of column names for spatial coordinates, e.g.,
        ``('coord_x', 'coord_y')``. If provided and found, spatial
        scatter plots are created. If ``None`` or columns are not found,
        the function falls back to temporal line plots.
    max_cols : int or 'auto', default 'auto'
        Controls the number of subplots per row. If ``'auto'``, it's
        determined by the number of metrics (e.g., actual + quantiles)
        being plotted for each prefix.
    cmap : str, default 'viridis'
        The Matplotlib colormap for spatial plots.
    cbar : {'uniform', 'individual'}, default 'uniform'
        Controls color bar scaling for spatial plots.
        - ``'uniform'``: All subplots share a single color scale.
        - ``'individual'``: Each subplot has its own color scale.
    axis_off : bool, default False
        If ``True``, turns off plot axes, ticks, and labels.
    show_grid : bool, default True
        If ``True`` and `axis_off` is ``False``, displays a grid on
        the subplots.
    grid_props : dict, optional
        Properties for the grid, e.g., ``{'linestyle': '--'}``.
    figsize : tuple of (float, float), optional
        The size of the figure for each `value_prefix`. If ``None``,
        the size is automatically calculated based on the layout.
    savefig : str, optional
        Path to save the figure(s). The prefix name and '_by_step'
        will be appended (e.g., 'my_plot_subsidence_by_step.png').
    save_fmts : str or list of str, default '.png'
        Format(s) to save the figure, e.g., ``['.png', '.pdf']``.
    verbose : int, default 1
        Controls the verbosity of logging messages.

    Returns
    -------
    None
        This function displays and/or saves Matplotlib figures and
        does not return any value.

    See Also
    --------
    forecast_view : A related function that plots by year/time
                    instead of by step index.

    Notes
    -----
    - The function expects a long-format DataFrame where each row is a
      unique combination of a sample and a forecast step.
    - If `spatial_cols` are not found, temporal line plots are
      generated by plotting values against the DataFrame's index. This
      is most meaningful when the DataFrame is sorted by a relevant
      variable (e.g., a spatial coordinate) for each step.

    Examples
    --------
    >>> import pandas as pd
    >>> # from fusionlab.plot.forecast import plot_forecast_by_step
    >>> data = {
    ...     'sample_idx':      list(range(4)) * 2,
    ...     'forecast_step':   [1] * 4 + [2] * 4,
    ...     'coord_x':         np.random.rand(8),
    ...     'coord_y':         np.random.rand(8),
    ...     'subsidence_q50':  np.random.randn(8),
    ...     'subsidence_actual': np.random.randn(8),
    ... }
    >>> df_step_example = pd.DataFrame(data)
    >>> # plot_forecast_by_step(
    ... #     df=df_step_example,
    ... #     value_prefixes=['subsidence'],
    ... #     spatial_cols=('coord_x', 'coord_y'),
    ... #     step_names={1: 'Year 1 Forecast', 2: 'Year 2 Forecast'},
    ... #     kind='dual'
    ... # )

    """
    # 1. Detect prefixes if not provided.
    if value_prefixes is None:
        vlog("Auto-detecting value prefixes…",
             level=1, verbose=verbose)
        
        # Define default columns to exclude from prefix detection.
        exclude_from_prefix = ['sample_idx', 'forecast_step']
        if spatial_cols:
            exclude_from_prefix.extend(spatial_cols)
        
        value_prefixes = get_value_prefixes_in(df, exclude_cols=exclude_from_prefix)
        if not value_prefixes:
            raise ValueError("Could not auto-detect value prefixes.")
        vlog(f"Detected prefixes: {value_prefixes}",
             level=2, verbose=verbose)

    # Ensure forecast_step column exists.
    if "forecast_step" not in df.columns:
        raise ValueError("DataFrame must contain 'forecast_step'.")

    # Collect metric suffixes found in columns.
    metrics = _get_metrics_from_cols(df.columns, value_prefixes)
    actual_metrics = sorted([m for m in metrics if "actual" in m])
    pred_metrics = sorted([m for m in metrics if "actual" not in m])
    
    # Determine if actuals will be plotted.
    plot_actuals = kind == "dual" and bool(actual_metrics)
    # Check for spatial plotting capability.
    has_spatial_coords = spatial_cols and all(
        c in df.columns for c in spatial_cols)
    if kind == 'spatial' and not has_spatial_coords:
        warnings.warn(
            f"kind='spatial' but columns {spatial_cols} not found. "
            "Falling back to temporal line plots."
        )
        kind = 'temporal' # Fallback kind
    elif kind == 'spatial':
        vlog("Plotting spatial data.", level=2, verbose=verbose)
    else: # kind is 'temporal'
        vlog("Plotting temporal data.", level=2, verbose=verbose)


    # Unique steps to visualize.
    steps_to_plot = sorted(df["forecast_step"].unique())
    if steps is not None: 
        steps = columns_manager( steps, empty_as_none= False)
        steps = [validate_positive_integer(v, f"Step {v}") 
                 for v in steps ]
        steps_to_plot = sorted (is_in_if(
            steps_to_plot, steps, 
            return_intersect= True,
            )
        )
    n_rows = len(steps_to_plot)
    if n_rows == 0:
        vlog("No forecast steps to plot.", level=0, verbose=verbose)
        return

    # 2. Compute global color limits if cbar is 'uniform'.
    vmin, vmax = None, None
    if cbar == "uniform":
        cols_for_cbar = [
            f"{p}_{m}" for p in value_prefixes for m in metrics
            if f"{p}_{m}" in df.columns
        ]
        if cols_for_cbar:
            vmin = df[cols_for_cbar].dropna().min().min()
            vmax = df[cols_for_cbar].dropna().max().max()

    # Shared kwargs for plotting helpers.
    plot_kwargs = dict(
        cmap=cmap, vmin=vmin, vmax=vmax, axis_off=axis_off,
        show_grid=show_grid,
        grid_props=grid_props or {"linestyle": ":", "alpha": 0.7},
        verbose=verbose,
    )

    # 3. Loop over each prefix and produce a grid of sub-plots.
    for prefix in value_prefixes:
        # Select relevant metrics for this prefix.
        prefix_preds = [
            m for m in pred_metrics if f"{prefix}_{m}" in df.columns
        ]
        prefix_actual = next(
            (m for m in actual_metrics if f"{prefix}_{m}" in df.columns),
            None,
        )

        # Determine number of columns for the grid.
        if max_cols == "auto":
            n_cols = len(prefix_preds) + (
                1 if plot_actuals and prefix_actual else 0
            )
        else:
            n_cols = int(max_cols)

        if n_cols == 0:
            continue

        # Create figure and axes grid.
        fig_size = figsize or (n_cols * 4, n_rows * 3.5)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=fig_size, squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(
            f"Forecast for '{prefix.upper()}' by Step",
            fontsize=16, weight='bold'
        )
        
        # Iterate over each forecast step (row).
        for r_idx, step in enumerate(steps_to_plot):
            step_df = df[df["forecast_step"] == step]
            step_lbl = f"Step {step}"
            # Custom label overrides.
            if isinstance(step_names, dict):
                step_lbl = step_names.get(step, step_lbl)
            elif isinstance(step_names, list) and step - 1 < len(step_names):
                step_lbl = f"Step {step}: {step_names[step - 1]}"

            c_idx = 0
            # Plot actuals if requested and available.
            if plot_actuals and prefix_actual:
                ax = axes[r_idx][c_idx]
                plot_kwargs["title"] = f"Actual ({step_lbl})"
                plot_col = f"{prefix}_{prefix_actual}"
                if kind == 'spatial':
                    _plot_spatial_subplot(
                        ax, step_df, *spatial_cols, plot_col, **plot_kwargs
                    )
                else:
                    _plot_temporal_subplot(
                        ax, step_df, plot_col, **plot_kwargs
                    )
                c_idx += 1

            # Plot prediction metrics.
            for metric in prefix_preds:
                if c_idx >= n_cols: break
                ax = axes[r_idx][c_idx]
                plot_kwargs["title"] = (
                    f"{metric.replace('_', ' ').title()} ({step_lbl})"
                )
                plot_col = f"{prefix}_{metric}"
                if kind == 'spatial':
                    _plot_spatial_subplot(
                        ax, step_df, *spatial_cols, plot_col, **plot_kwargs
                    )
                else:
                    _plot_temporal_subplot(
                        ax, step_df, plot_col, **plot_kwargs
                    )
                c_idx += 1

            # Blank unused axes in this row.
            for j in range(c_idx, n_cols):
                axes[r_idx][j].axis("off")

        # Add a single color-bar if using uniform scaling.
        if cbar == "uniform" and vmin is not None and vmax is not None:
            fig.colorbar(
                ScalarMappable(
                    norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
                    cmap=cmap
                ),
                ax=axes.ravel().tolist(), orientation="vertical",
                shrink=0.8, pad=0.04
            )

        # 4. Save figure to disk if requested.
        if savefig:
            fmts = [save_fmts] if isinstance(save_fmts, str) else save_fmts
            base, _ = os.path.splitext(savefig)
            for fmt in fmts:
                fname = (
                    f"{base}_{prefix}_by_step"
                    f"{fmt if fmt.startswith('.') else '.' + fmt}"
                )
                out_dir = os.path.dirname(fname)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                vlog(f"Saving figure to {fname}", level=1, verbose=verbose)
                fig.savefig(fname, dpi=300, bbox_inches="tight")

        plt.show()
        
    
@check_non_emptiness 
def forecast_view_in(
    forecast_df: pd.DataFrame,
    value_prefixes: Optional[List[str]]=None, 
    kind: str = 'dual',
    view_quantiles: Optional[List[Union[str, float]]] = None,
    view_years: Optional[List[Union[str, int]]] = None,
    spatial_cols: Tuple[str, str] = ('coord_x', 'coord_y'),
    time_col='coord_t', 
    max_cols: Union[int, str] = 'auto',
    cmap: str = 'viridis',
    cbar: str = 'uniform',
    axis_off: bool = False,
    show_grid: bool = True,
    grid_props: Optional[Dict] = None,
    figsize: Optional[Tuple[float, float]] = None,
    savefig: Optional[str] = None,
    save_fmts: Union[str, List[str]] = '.png',
    verbose: int = 1
):
    """Generates and displays spatial forecast visualizations.

    This function creates a grid of scatter plots to visualize
    spatio-temporal forecast data. It can handle both long-format
    and wide-format DataFrames, automatically arranging plots by
    year and metric (e.g., different quantiles).

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Input DataFrame containing forecast data. The function
        auto-detects if the format is long or wide.

    value_prefixes : list of str, optional
        The base names of the metrics to plot (e.g.,
        ``['subsidence', 'GWL']``). If ``None``, the function
        will attempt to automatically infer these from the
        DataFrame's column names. A separate figure is generated
        for each prefix.

    kind : {'dual', 'pred_only'}, default 'dual'
        Determines what to plot:
        - ``'dual'``: Plots both actual values and predictions
          side-by-side for comparison, if actuals are available.
        - ``'pred_only'``: Plots only the predicted values.

    view_quantiles : list of str or float, optional
        A list to filter which quantiles are displayed. Values can be
        floats (e.g., ``[0.1, 0.5]``) or strings (e.g.,
        ``['q10', 'q50']``). If ``None``, all detected quantiles
        are plotted.

    view_years : list of str or int, optional
        A list of years to display. If ``None``, all detected years
        in the forecast are plotted.

    spatial_cols : tuple of str, default ('coord_x', 'coord_y')
        A tuple containing the names of the columns to be used for
        the x and y axes of the scatter plots.

    time_col : str, default 'coord_t'
        The name of the column representing the time dimension in a
        long-format DataFrame. Used by the internal format detector.

    max_cols : int or 'auto', default 'auto'
        Controls the number of subplots per row.
        - If ``'auto'``, the number of columns is automatically set
          to the number of quantiles being plotted (plus one if
          `kind='dual'`).
        - If an integer, sets a fixed number of columns for the
          prediction plots. If `kind='dual'`, an additional column
          for the actuals is added, potentially exceeding this number.

    cmap : str, default 'viridis'
        The Matplotlib colormap to use for the scatter plots.

    cbar : {'uniform', 'individual'}, default 'uniform'
        Controls the color bar scaling:
        - ``'uniform'``: All subplots in a figure share a single,
          uniform color scale, determined by the global min/max of
          all plotted data. A single color bar is displayed for the
          entire figure.
        - ``'individual'``: Each subplot has its own color bar scaled
          to its own data range. (Note: Current implementation
          defaults to uniform; individual color bars would be a
          future enhancement).

    axis_off : bool, default False
        If ``True``, turns off the axes (ticks, labels, spines) for
        all subplots.

    show_grid : bool, default True
        If ``True`` and `axis_off` is ``False``, a grid is displayed
        on the subplots.

    grid_props : dict, optional
        A dictionary of properties to pass to `ax.grid()` (e.g.,
        ``{'linestyle': ':', 'alpha': 0.7}``).

    figsize : tuple of (float, float), optional
        The size of the figure for each `value_prefix`. If ``None``,
        the size is automatically calculated based on the number of
        rows and columns.

    savefig : str, optional
        If a file path is provided (e.g., 'my_forecast.png'), the
        figure(s) will be saved. The prefix name will be appended
        to the filename (e.g., 'my_forecast_subsidence.png').

    save_fmts : str or list of str, default '.png'
        The format(s) to save the figure in (e.g., ``['.png', '.pdf']``).

    verbose : int, default 1
        Controls the verbosity of logging messages. `0` is silent,
        `1` provides basic info, and higher values provide more detail.

    Returns
    -------
    None
        This function does not return any value. It displays and/or
        saves Matplotlib figures directly.

    See Also
    --------
    format_forecast_dataframe : The utility used to detect and pivot
                                the input DataFrame.
    get_value_prefixes : The utility used to auto-detect metric prefixes.

    Notes
    -----
    - The function creates a grid where each row corresponds to a year
      from `view_years`, and columns correspond to the actuals (if
      `kind='dual'`) and each of the selected quantiles.
    - If an actual value for a specific year is not available, the
      function will attempt to fill that plot using the most recent
      known 'actual' value to facilitate comparison across the row.
    - Currently, a separate figure is generated for each prefix in
      `value_prefixes`.
    """
    # Use the format utility to ensure we have a wide DataFrame
    # Auto-detect value prefixes if not provided
    if value_prefixes is None:
        vlog("`value_prefixes` not provided. Auto-detecting...",
             level=1, verbose=verbose)
        value_prefixes = get_value_prefixes(forecast_df)
        if not value_prefixes:
            raise ValueError(
                "Could not auto-detect any value prefixes. Please "
                "provide them explicitly."
            )
        vlog(f"Detected prefixes: {value_prefixes}",
             level=2, verbose=verbose)
        
    value_prefixes = columns_manager(value_prefixes) 
    
    df_wide = format_forecast_dataframe(
        df= forecast_df,
        to_wide=True,
        value_prefixes=value_prefixes,
        # Pass relevant pivot args if df is long
        id_vars=[
            c for c in ['sample_idx', *spatial_cols] 
            if c in forecast_df.columns
        ],
        time_col=time_col,
        static_actuals_cols=[
            c for c in forecast_df.columns 
            if c.endswith('_actual') and c.count('_') == 1
        ],
        verbose=verbose
    )

    vlog("Parsing wide DataFrame columns to build plot structure.",
         level=1, verbose=verbose)
    plot_structure = _parse_wide_df_columns(df_wide, value_prefixes)

    # --- Filter data based on user preferences ---
    all_years = sorted(list(
        set(y for p_data in plot_structure.values() 
            for y in p_data if y.isdigit())
    ))
    years_to_plot = [str(y) for y in view_years] if view_years else all_years

    # Identify available and requested quantiles
    all_quantiles = sorted(list(set(
        suffix for p_data in plot_structure.values()
        for y_data in p_data.values() if isinstance(y_data, dict)
        for suffix in y_data if suffix.startswith('q')
    )))
    
    if view_quantiles:
        # Normalize requested quantiles to 'qXX' format
        req_q_norm = []
        for q in view_quantiles:
            if isinstance(q, float):
                req_q_norm.append(f"q{int(q*100)}")
            else: # is str
                req_q_norm.append(q if q.startswith('q') else f"q{q}")
        quantiles_to_plot = [q for q in all_quantiles if q in req_q_norm]
    else:
        quantiles_to_plot = all_quantiles

    # Handle deterministic case (no quantiles found)
    is_deterministic = not bool(quantiles_to_plot)
    if is_deterministic:
        # Look for a default prediction column, e.g., 'subsidence_2022_pred'
        quantiles_to_plot = ['pred']
        vlog("No quantiles found. Assuming deterministic forecast.",
             level=1, verbose=verbose)
    
    # --- Setup Plot Layout ---
    plot_actuals = (kind == 'dual')
    n_rows = len(years_to_plot)
    if n_rows == 0:
        vlog("No years to plot after filtering.", level=0, 
             verbose=verbose)
        return
        
    # --- Prepare for plotting ---
    x_col, y_col = spatial_cols
    grid_props = grid_props or {'linestyle': ':', 'alpha': 0.7}
    
    # Determine uniform color range if needed
    vmin, vmax = None, None
    if cbar == 'uniform':
        all_plot_cols = []
        # Collect every column-reference that actually exists in df_wide
        def _collect_cols(node):
            """Recursively collect column names from plot_structure nodes."""
            if isinstance(node, str):
                all_plot_cols.append(node)
            elif isinstance(node, dict):
                for sub in node.values():
                    _collect_cols(sub)
    
        for p_data in plot_structure.values():
            _collect_cols(p_data)

        valid_plot_cols = [c for c in all_plot_cols if c in df_wide.columns]
        if valid_plot_cols:
            min_vals = [df_wide[c].dropna().min() for c in valid_plot_cols]
            max_vals = [df_wide[c].dropna().max() for c in valid_plot_cols]
            
            if any(pd.notna(v) for v in min_vals): 
                vmin = min(v for v in min_vals if pd.notna(v))
            if any(pd.notna(v) for v in max_vals): 
               vmax = max(v for v in max_vals if pd.notna(v))
           
            vmin = min(min_vals)
            vmax = max(max_vals)
    
    
    for prefix in value_prefixes:
        if max_cols == 'auto':
            n_cols_prefix = len(quantiles_to_plot)
            if plot_actuals: n_cols_prefix += 1
        else:
            n_cols_prefix = int(max_cols)

        if n_cols_prefix == 0: 
            continue

        figsize_prefix = figsize or (n_cols_prefix * 3.5, n_rows * 3)
        fig, axes = plt.subplots(
            n_rows, n_cols_prefix, figsize=figsize_prefix, 
            squeeze=False, constrained_layout=True
        )
        fig.suptitle(
            f"Forecast Visualization for '{prefix.upper()}'", 
            fontsize=14, weight='bold'
        )

        last_known_actual_col = plot_structure[prefix].get("static_actual")
        for row_idx, year in enumerate(years_to_plot):
            ax_row = axes[row_idx]
            col_idx = 0
            
            if plot_actuals:
                actual_col = plot_structure[prefix].get(
                    year, {}).get('actual', last_known_actual_col)
                if actual_col: last_known_actual_col = actual_col
                
                title = f"Actual ({year})"
                _plot_single_scatter(
                    ax_row[col_idx], df_wide, x_col, y_col, 
                    actual_col, cmap, vmin, vmax, title,
                    axis_off, show_grid, grid_props
                )
                col_idx += 1
            
            for q_suffix in quantiles_to_plot:
                if col_idx >= n_cols_prefix: continue
                pred_col = plot_structure[prefix].get(year, {}).get(q_suffix)
                title = f"{q_suffix.replace('q', 'Q').capitalize()} ({year})"
                if is_deterministic: title = f"Prediction ({year})"
                _plot_single_scatter(
                    ax_row[col_idx], df_wide, x_col, y_col, pred_col,
                    cmap, vmin, vmax, title, axis_off, show_grid, grid_props
                )
                col_idx += 1
            
            for i in range(col_idx, n_cols_prefix):
                ax_row[i].axis('off')

        if cbar == 'uniform' and vmin is not None and vmax is not None:
             fig.colorbar(
                ScalarMappable(
                    norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                ax=axes, orientation='vertical', fraction=0.02, pad=0.04
             )

        if savefig:
            fmts = [save_fmts] if isinstance(save_fmts, str) else save_fmts
            base, _ = os.path.splitext(savefig)
        
            for fmt in fmts:
                fname = f"{base}_{prefix}{fmt if fmt.startswith('.') else '.' + fmt}"

                # --- ensure output directory exists --
                out_dir = os.path.dirname(fname)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                vlog(f"Saving figure to {fname}", level=1, verbose=verbose)
                fig.savefig(fname, dpi=300, bbox_inches="tight")


        plt.show()


@check_non_emptiness 
def forecast_view(
    forecast_df: pd.DataFrame,
    value_prefixes: Optional[List[str]] = None,
    kind: str = 'dual',
    view_quantiles: Optional[List[Union[str, float]]] = None,
    view_years: Optional[List[Union[str, int]]] = None,
    spatial_cols: Tuple[str, str] = None,
    time_col: str = 'coord_t',
    max_cols: Union[int, str] = 'auto',
    cmap: str = 'viridis',
    cbar: str = 'uniform',
    axis_off: bool = False,
    show_grid: bool = True,
    grid_props: Optional[Dict] = None,
    figsize: Optional[Tuple[float, float]] = None,
    savefig: Optional[str] = None,
    save_fmts: Union[str, List[str]] = '.png',
    dt_col: Optional[str] = None,
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    verbose: int = 1,
    
    **kws
):
    """Generates and displays spatial forecast visualizations.

    This function creates a grid of scatter plots to visualize
    spatio-temporal forecast data. It can handle both long-format
    and wide-format DataFrames, automatically arranging plots by
    year and metric (e.g., different quantiles).

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Input DataFrame containing forecast data. The function
        auto-detects if the format is long or wide.

    value_prefixes : list of str, optional
        The base names of the metrics to plot (e.g.,
        ``['subsidence', 'GWL']``). If ``None``, the function
        will attempt to automatically infer these from the
        DataFrame's column names. A separate figure is generated
        for each prefix.

    kind : {'dual', 'pred_only'}, default 'dual'
        Determines what to plot:
        - ``'dual'``: Plots both actual values and predictions
          side-by-side for comparison, if actuals are available.
        - ``'pred_only'``: Plots only the predicted values.

    view_quantiles : list of str or float, optional
        A list to filter which quantiles are displayed. Values can be
        floats (e.g., ``[0.1, 0.5]``) or strings (e.g.,
        ``['q10', 'q50']``). If ``None``, all detected quantiles
        are plotted.

    view_years : list of str or int, optional
        A list of years to display. If ``None``, all detected years
        in the forecast are plotted.

    spatial_cols : tuple of str, default ('coord_x', 'coord_y')
        A tuple containing the names of the columns to be used for
        the x and y axes of the scatter plots.

    time_col : str, default 'coord_t'
        The name of the column representing the time dimension in a
        long-format DataFrame. Used by the internal format detector.
        
        ..note::
            'time_col' and 'dt_col' can be used interchangeability. 

    max_cols : int or 'auto', default 'auto'
        Controls the number of subplots per row.
        - If ``'auto'``, the number of columns is automatically set
          to the number of quantiles being plotted (plus one if
          `kind='dual'`).
        - If an integer, sets a fixed number of columns for the
          prediction plots. If `kind='dual'`, an additional column
          for the actuals is added, potentially exceeding this number.

    cmap : str, default 'viridis'
        The Matplotlib colormap to use for the scatter plots.

    cbar : {'uniform', 'individual'}, default 'uniform'
        Controls the color bar scaling:
        - ``'uniform'``: All subplots in a figure share a single,
          uniform color scale, determined by the global min/max of
          all plotted data. A single color bar is displayed for the
          entire figure.
        - ``'individual'``: Each subplot has its own color bar scaled
          to its own data range. (Note: Current implementation
          defaults to uniform; individual color bars would be a
          future enhancement).

    axis_off : bool, default False
        If ``True``, turns off the axes (ticks, labels, spines) for
        all subplots.

    show_grid : bool, default True
        If ``True`` and `axis_off` is ``False``, a grid is displayed
        on the subplots.

    grid_props : dict, optional
        A dictionary of properties to pass to `ax.grid()` (e.g.,
        ``{'linestyle': ':', 'alpha': 0.7}``).

    figsize : tuple of (float, float), optional
        The size of the figure for each `value_prefix`. If ``None``,
        the size is automatically calculated based on the number of
        rows and columns.

    savefig : str, optional
        If a file path is provided (e.g., 'my_forecast.png'), the
        figure(s) will be saved. The prefix name will be appended
        to the filename (e.g., 'my_forecast_subsidence.png').

    save_fmts : str or list of str, default '.png'
        The format(s) to save the figure in (e.g., ``['.png', '.pdf']``).

    verbose : int, default 1
        Controls the verbosity of logging messages. `0` is silent,
        `1` provides basic info, and higher values provide more detail.
        
    kws: dict, 
       Keywords arguments for feature extensions. 
       
    Returns
    -------
    None
        This function does not return any value. It displays and/or
        saves Matplotlib figures directly.

    See Also
    --------
    format_forecast_dataframe : The utility used to detect and pivot
                                the input DataFrame.
    get_value_prefixes : The utility used to auto-detect metric prefixes.

    Notes
    -----
    - The function creates a grid where each row corresponds to a year
      from `view_years`, and columns correspond to the actuals (if
      `kind='dual'`) and each of the selected quantiles.
    - If an actual value for a specific year is not available, the
      function will attempt to fill that plot using the most recent
      known 'actual' value to facilitate comparison across the row.
    - Currently, a separate figure is generated for each prefix in
      `value_prefixes`.
    """
    # canonicalise column name
    kw = _coerce_dt_kw(
        dt_col=dt_col, time_col=time_col, _time_default=time_col)
    time_col = kw.get("dt_col", time_col)        # adopt canonical name
        
    _spatial_cols = spatial_cols or []
    
    is_q = detect_forecast_type(forecast_df, value_prefixes=value_prefixes)
    
    if is_q =='deterministic': 
        warnings.warn (
            "Deterministic prediction is detected. It is recommended" 
            " to use 'fusionlab.plot.forecast.plot_forecast_by_step'"
            " instead. Use at your own risk."
        )
        
    if value_prefixes is None:
        vlog("`value_prefixes` not provided. Auto-detecting...",
             level=1, verbose=verbose)
        value_prefixes = get_value_prefixes(
            forecast_df,
            exclude_cols=['sample_idx', *_spatial_cols,
                          time_col, 'forecast_step']
        )
        if not value_prefixes:
            raise ValueError(
                "Could not auto-detect any value prefixes. "
                "Please provide them explicitly."
            )
        vlog(f"Detected prefixes: {value_prefixes}",
             level=2, verbose=verbose)
             
    df_wide = format_forecast_dataframe(
        forecast_df, to_wide=True, value_prefixes=value_prefixes,
        id_vars=[c for c in ['sample_idx', *_spatial_cols]
                 if c in forecast_df.columns],
        time_col=time_col,
        static_actuals_cols=[
            c for c in forecast_df.columns
            if c.endswith('_actual') and c.count('_') == 1
        ],
        spatial_cols = _spatial_cols, 
        verbose=verbose
    )

    plot_structure = _parse_wide_df_columns(df_wide, value_prefixes)

    all_years = sorted([
        y for p_data in plot_structure.values() for y in p_data
        if y.isdigit()
    ])
    years_to_plot = [str(y) for y in view_years] if view_years else all_years
    
    all_quantiles = sorted(list(set(
        s for p in plot_structure.values()
        for y in p.values() if isinstance(y, dict)
        for s in y if s.startswith('q')
    )))
    
    if view_quantiles:
        req_q_norm = [
            f"q{int(q*100)}" if isinstance(q, float) 
            else (q if q.startswith('q') else f"q{q}")
            for q in view_quantiles
        ]
        quantiles_to_plot = [q for q in all_quantiles if q in req_q_norm]
    else:
        quantiles_to_plot = all_quantiles

    if not quantiles_to_plot:
        quantiles_to_plot = ['pred']
        vlog("No quantiles found. Assuming deterministic forecast.",
             level=1, verbose=verbose)
    
    n_rows = len(years_to_plot)
    if n_rows == 0:
        vlog("No years to plot after filtering.", level=0, verbose=verbose)
        return

    vmin, vmax = None, None
    if cbar == 'uniform':
        all_plot_cols = [
            c for p in plot_structure.values()
            for y in p.values()
            for c in (y.values() if isinstance(y, dict) else [y])
            if c in df_wide
        ]
        if all_plot_cols:
            min_vals = [df_wide[c].dropna().min() for c in all_plot_cols]
            max_vals = [df_wide[c].dropna().max() for c in all_plot_cols]
            if any(pd.notna(v) for v in min_vals):
                vmin = min(v for v in min_vals if pd.notna(v))
            if any(pd.notna(v) for v in max_vals):
                vmax = max(v for v in max_vals if pd.notna(v))
    
    plot_kwargs = dict(
        cmap=cmap, vmin=vmin, vmax=vmax, axis_off=axis_off,
        show_grid=show_grid,
        grid_props=(grid_props or {'linestyle': ':', 'alpha': 0.7}),
        verbose=verbose
    )

    for prefix in value_prefixes:
        num_pred_cols = len(quantiles_to_plot)
        num_actual_cols = 1 if kind == 'dual' else 0
        if max_cols == 'auto':
            n_cols = num_pred_cols + num_actual_cols
        else:
            n_cols = int(max_cols)

        if n_cols == 0: 
            continue

        fig_size = figsize or (n_cols * 3.5, n_rows * 3)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=fig_size, squeeze=False,
            constrained_layout=True
        )
        fig.suptitle(
            f"Forecast Visualization for '{prefix.upper()}'",
            fontsize=14, weight='bold'
        )

        _plot_forecast_grid(
            fig, axes, df_wide, plot_structure, years_to_plot,
            quantiles_to_plot, prefix, kind, spatial_cols, 
            s = kws.pop('s', 10), plot_kwargs=plot_kwargs,
            _logger=_logger, 
        )
        
        if cbar == 'uniform' and vmin is not None and vmax is not None:
             fig.colorbar(
                ScalarMappable(
                    norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                ax=axes.ravel().tolist(), orientation='vertical',
                shrink=0.8, pad=0.04
             )

        if savefig:
            fmts = [save_fmts] if isinstance(save_fmts, str) else save_fmts
            base, _ = os.path.splitext(savefig)
            for fmt in fmts:
                fname = f"{base}_{prefix}{fmt if fmt.startswith('.') else '.' + fmt}"
                out_dir = os.path.dirname(fname)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                vlog(f"Saving figure to {fname}", level=1,
                     verbose=verbose, logger=_logger
                     )
                fig.savefig(fname, dpi=300, bbox_inches="tight")

        plt.show()

@check_non_emptiness 
def plot_forecasts(
    forecast_df: pd.DataFrame,
    target_name: str = "target",
    quantiles: Optional[List[float]] = None,
    output_dim: int = 1,
    kind: str = "temporal",
    actual_data: Optional[pd.DataFrame] = None,
    dt_col: Optional[str] = None,
    actual_target_name: Optional[str] = None,
    sample_ids: Optional[Union[int, List[int], str]] = "first_n",
    num_samples: int = 3,
    horizon_steps: Optional[Union[int, List[int]]] = 1,
    spatial_cols: Optional[List[str]] = None,
    max_cols: int = 2,
    figsize: Tuple[float, float] = (8, 4.5), 
    scaler: Optional[Any] = None,
    scaler_feature_names: Optional[List[str]] = None,
    target_idx_in_scaler: Optional[int] = None,
    titles: Optional[List[str]] = None,
    cbar: Optional[str] = None, #new
    step_names : Dict[int, Any] =None, # new
    show_grid: bool = True, # new
    grid_props: Optional[Dict] = None, # new
    verbose: int = 0,
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    **plot_kwargs: Any
    ) -> None:
    """
    Visualizes model forecasts from a structured DataFrame.
    
    This function generates plots to visualize time series forecasts,
    supporting temporal line plots for individual samples/items and
    spatial scatter plots for specific forecast horizons. It can
    handle point forecasts and quantile (probabilistic) forecasts,
    optionally overlaying actual values for comparison. Predictions
    and actuals can be inverse-transformed if a scaler is provided.
    
    The input `forecast_df` is expected to be in a long format, typically
    generated by :func:`~fusionlab.nn.utils.format_predictions_to_dataframe`,
    containing 'sample_idx' and 'forecast_step' columns, along with
    prediction columns (e.g., '{target_name}_pred',
    '{target_name}_q50') and optionally actual value columns
    (e.g., '{target_name}_actual').
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        A pandas DataFrame containing the forecast data in long format.
        Must include 'sample_idx' and 'forecast_step' columns,
        along with prediction columns (and optionally actuals and
        spatial coordinate columns).
    target_name : str, default "target"
        The base name of the target variable. This is used to
        construct column names for predictions and actuals (e.g.,
        "target_pred", "target_q50", "target_actual").
    quantiles : List[float], optional
        A list of quantiles that were predicted (e.g., `[0.1, 0.5, 0.9]`).
        If provided, the function will plot the median quantile as a line
        and the range between the first and last quantile as a shaded
        uncertainty interval (for temporal plots). If ``None``, a point
        forecast is assumed. Default is ``None``.
    output_dim : int, default 1
        The number of target variables (output dimensions) predicted
        at each time step. If > 1, separate subplots or plot groups
        may be generated for each output dimension.
    kind : {'temporal', 'spatial'}, default "temporal"
        The type of plot to generate:
        - ``'temporal'``: Plots forecasts over the horizon for selected
          samples (time series line plots).
        - ``'spatial'``: Creates scatter plots of forecast values over
          spatial coordinates for selected horizon steps. Requires
          `spatial_cols` to be specified.
    actual_data : pd.DataFrame, optional
        An optional DataFrame containing the true actual values for
        comparison. If `forecast_df` already contains actual value
        columns (e.g., '{target_name}_actual'), this may not be needed
        or can be used to supplement. If provided, `dt_col` and
        `actual_target_name` might be used for alignment or direct plotting.
        *Note: Current implementation primarily uses actuals from `forecast_df`.*
    dt_col : str, optional
        Name of the datetime column in `actual_data` or `forecast_df`
        if needed for x-axis labeling in temporal plots. If `forecast_df`
        uses 'forecast_step', this might be less critical unless aligning
        with a true date axis from `actual_data`.
    actual_target_name : str, optional
        The name of the target column in `actual_data` if it differs
        from `target_name`. If ``None``, `target_name` is assumed.
    sample_ids : int, List[int], str, default "first_n"
        Specifies which samples (based on 'sample_idx' in `forecast_df`)
        to plot for `kind='temporal'`.
        - If `int`: Plots the sample at that specific index.
        - If `List[int]`: Plots all samples with these indices.
        - If ``"first_n"``: Plots the first `num_samples`.
        - If ``"all"``: Plots all unique samples (can be many plots).
    num_samples : int, default 3
        Number of samples to plot if `sample_ids="first_n"`.
    horizon_steps : int, List[int], str, default 1
        Specifies which forecast horizon step(s) to plot for
        `kind='spatial'`.
        - If `int`: Plots the specified single step (1-indexed).
        - If `List[int]`: Plots all specified steps.
        - If ``"all"`` or ``None``: Plots all unique forecast steps available.
    spatial_cols : List[str], optional
        Required if `kind='spatial'`. A list of two column names
        from `forecast_df` representing the x and y spatial
        coordinates (e.g., `['longitude', 'latitude']`).
        Default is ``None``.
    max_cols : int, default 2
        Maximum number of subplots to arrange per row in the figure.
    figsize : Tuple[float, float], default (8, 4.5)
        The size `(width, height)` in inches for **each subplot**.
        The total figure size will be inferred based on the number
        of rows and columns of subplots.
    scaler : Any, optional
        A fitted scikit-learn-like scaler object (must have an
        `inverse_transform` method). If provided along with
        `scaler_feature_names` and `target_idx_in_scaler`,
        prediction and actual columns related to `target_name`
        will be inverse-transformed before plotting. Default is ``None``.
    scaler_feature_names : List[str], optional
        A list of all feature names (in order) that the `scaler` was
        originally fit on. Required for targeted inverse transform if
        `scaler` is provided. Default is ``None``.
    target_idx_in_scaler : int, optional
        The index of the `target_name` (or the specific output
        dimension being plotted) within the `scaler_feature_names`
        list. Required for targeted inverse transform if `scaler` is
        provided. Default is ``None``.
    titles : List[str], optional
        A list of custom titles for the subplots. If provided, its
        length should match the number of subplots generated.
        Default is ``None`` (titles are auto-generated).
    cbar : {'uniform', None}, default=None  
        Controls colour‑bar behaviour for *spatial* plots.  Use
        ``'uniform'`` to compute a single global ``vmin`` / ``vmax`` and
        attach one shared colour‑bar to the figure.  Any other value (or
        *None*) lets each subplot auto‑scale its own colour‑bar.
    step_names : dict[int, Any] or None, default=None  
        Optional mapping that renames forecast‑step integers when building
        subplot titles—e.g. ``{1: "2021", 2: "2022"}``.  Keys are coerced to
        ``int``; values are converted to ``str``.  Missing steps fall back
        to *default_name* (``""`` is the step number itself).
    show_grid : bool, default=True  
        Toggle the display of ``ax.grid`` in both temporal and spatial
        plots.  Set to *False* for a cleaner look when overlaying graphics
        on a map.
    grid_props : dict or None, default=None  
        Keyword arguments forwarded to :pyfunc:`matplotlib.axes.Axes.grid`
        (e.g. ``{"linestyle": ":", "alpha": 0.6}``).  Ignored when
        *show_grid* is *False*.
    verbose : int, default 0
        Verbosity level for logging during plot generation.
        - ``0``: Silent.
        - ``1`` or higher: Print informational messages.
    **plot_kwargs : Any
        Additional keyword arguments to pass to the underlying
        Matplotlib plotting functions (e.g., `ax.plot`, `ax.scatter`,
        `ax.fill_between`). Can be used to customize line styles,
        colors, marker sizes, etc. For example,
        `median_plot_kwargs={'color': 'blue'}`,
        `fill_between_kwargs={'color': 'lightblue'}`.
    
    Returns
    -------
    None
        This function directly generates and shows/saves plots using
        Matplotlib and does not return any value.
    
    Raises
    ------
    TypeError
        If `forecast_df` is not a pandas DataFrame.
    ValueError
        If essential columns like 'sample_idx' or 'forecast_step'
        are missing from `forecast_df`.
        If `kind='spatial'` is chosen but `spatial_cols` are not
        provided or not found in `forecast_df`.
        If an unsupported `kind` is specified.
    
    See Also
    --------
    fusionlab.nn.utils.format_predictions_to_dataframe :
        Utility to generate the `forecast_df` expected by this function.
    matplotlib.pyplot.plot : Underlying plotting function for temporal forecasts.
    matplotlib.pyplot.scatter : Underlying plotting function for spatial forecasts.
    
    Examples
    --------
    >>> from fusionlab.nn.utils import format_predictions_to_dataframe
    >>> from fusionlab.plot.forecast import plot_forecasts
    >>> import pandas as pd
    >>> import numpy as np
    
    >>> # Assume preds_point (B,H,O) and preds_quant (B,H,Q) are available
    >>> B, H, O, Q_len = 4, 3, 1, 3
    >>> preds_point = np.random.rand(B, H, O)
    >>> preds_quant = np.random.rand(B, H, Q_len)
    >>> y_true_seq = np.random.rand(B, H, O)
    >>> quantiles_list = [0.1, 0.5, 0.9]
    
    >>> # Create DataFrames using format_predictions_to_dataframe
    >>> df_point = format_predictions_to_dataframe(
    ...     predictions=preds_point, y_true_sequences=y_true_seq,
    ...     target_name="value", forecast_horizon=H, output_dim=O
    ... )
    >>> df_quant = format_predictions_to_dataframe(
    ...     predictions=preds_quant, y_true_sequences=y_true_seq,
    ...     target_name="value", quantiles=quantiles_list,
    ...     forecast_horizon=H, output_dim=O
    ... )
    
    >>> # Example 1: Plot temporal point forecast for first sample
    >>> # plot_forecasts(df_point, target_name="value", sample_ids=0)
    
    >>> # Example 2: Plot temporal quantile forecast for first 2 samples
    >>> # plot_forecasts(df_quant, target_name="value",
    ... #                quantiles=quantiles_list, sample_ids="first_n",
    ... #                num_samples=2, max_cols=1)
    
    >>> # Example 3: Spatial plot (requires spatial_cols in df_quant)
    >>> # Assume df_quant has 'lon' and 'lat' columns
    >>> # df_quant['lon'] = np.random.rand(len(df_quant)) * 10
    >>> # df_quant['lat'] = np.random.rand(len(df_quant)) * 5
    >>> # plot_forecasts(df_quant, target_name="value",
    ... #                quantiles=quantiles_list, kind='spatial',
    ... #                horizon_steps=1, spatial_cols=['lon', 'lat'])
    """

    vlog(f"Starting forecast visualization (kind='{kind}')...",
         level=3, verbose=verbose, logger=_logger)

    if not isinstance(forecast_df, pd.DataFrame):
        raise TypeError("`forecast_df` must be a pandas DataFrame, typically "
                        "the output of `format_predictions_to_dataframe`.")

    if 'sample_idx' not in forecast_df.columns or \
       'forecast_step' not in forecast_df.columns:
        raise ValueError(
            "`forecast_df` must contain 'sample_idx' and 'forecast_step' "
            "columns."
        )

    # --- Data Preparation & Inverse Transform ---
    # Work on a copy to avoid modifying the original DataFrame
    df_to_plot = forecast_df.copy()
    actual_col_names_in_df = [] # To store names of actual columns in df_to_plot
    pred_col_names_in_df = []   # To store names of prediction columns

    # Identify prediction and actual columns based on target_name and quantiles
    base_pred_name = target_name
    base_actual_name = actual_target_name if actual_target_name \
        else target_name

    if quantiles:
        # Sort quantiles to ensure consistent order for plotting (e.g., low, mid, high)
        sorted_quantiles = sorted([float(q) for q in quantiles])
        for o_idx in range(output_dim):
            for q_val in sorted_quantiles:
                q_suffix = f"_q{int(q_val*100)}"
                col_name = f"{base_pred_name}"
                if output_dim > 1: col_name += f"_{o_idx}"
                col_name += q_suffix
                if col_name in df_to_plot.columns:
                    pred_col_names_in_df.append(col_name)
            # Actual column for this output dimension
            actual_col = f"{base_actual_name}"
            if output_dim > 1: actual_col += f"_{o_idx}"
            actual_col += "_actual"
            if actual_col in df_to_plot.columns:
                actual_col_names_in_df.append(actual_col)
    else: # Point forecast
        for o_idx in range(output_dim):
            col_name = f"{base_pred_name}"
            if output_dim > 1: col_name += f"_{o_idx}"
            col_name += "_pred"
            if col_name in df_to_plot.columns:
                pred_col_names_in_df.append(col_name)
            actual_col = f"{base_actual_name}"
            if output_dim > 1: actual_col += f"_{o_idx}"
            actual_col += "_actual"
            if actual_col in df_to_plot.columns:
                actual_col_names_in_df.append(actual_col)

    if not pred_col_names_in_df:
        warnings.warn("No prediction columns found in `forecast_df` "
                      "based on `target_name` and `quantiles`.")
        # Attempt to find any column ending with _pred or _qXX
        pred_cols_found = [c for c in df_to_plot.columns if "_pred" in c or "_q" in c]
        if pred_cols_found:
            pred_col_names_in_df = pred_cols_found
            vlog(f"  Auto-detected prediction columns: {pred_col_names_in_df}",
                 level=2, verbose=verbose,logger=_logger)
        else:
            vlog("  No prediction columns could be auto-detected. "
                 "Plotting may be limited.", level=2, verbose=verbose, 
                 logger=_logger)

    # Apply inverse transform if scaler is provided
    if scaler is not None:
        if scaler_feature_names is None or target_idx_in_scaler is None:
            warnings.warn(
                "Scaler provided, but `scaler_feature_names` or "
                "`target_idx_in_scaler` is missing. Inverse transform "
                "cannot be applied correctly to specific target columns.",
                UserWarning
            )
        else:
            vlog("  Applying inverse transformation using provided scaler...",
                 level=4, verbose=verbose, logger=_logger)
            cols_to_inverse_transform = pred_col_names_in_df + actual_col_names_in_df
            dummy_array_shape = (len(df_to_plot), len(scaler_feature_names))

            for col_to_inv in cols_to_inverse_transform:
                if col_to_inv in df_to_plot.columns:
                    try:
                        dummy = np.zeros(dummy_array_shape)
                        dummy[:, target_idx_in_scaler] = df_to_plot[col_to_inv]
                        df_to_plot[col_to_inv] = scaler.inverse_transform(
                            dummy
                            )[:, target_idx_in_scaler]
                    except Exception as e:
                        warnings.warn(
                            f"Failed to inverse transform column '{col_to_inv}'. "
                            f"Plotting scaled value. Error: {e}"
                            )
            vlog("    Inverse transformation applied.", 
                 level=5, verbose=verbose, logger=_logger
                 )

    # --- Select Samples/Items to Plot ---
    unique_sample_ids = df_to_plot['sample_idx'].unique()
    selected_ids_for_plot = []
    if isinstance(sample_ids, str):
        if sample_ids.lower() == "all":
            selected_ids_for_plot = unique_sample_ids
        elif sample_ids.lower() == "first_n":
            selected_ids_for_plot = unique_sample_ids[:num_samples]
        else:
            warnings.warn(f"Unknown string for `sample_ids`: "
                          f"'{sample_ids}'. Plotting first sample.")
            selected_ids_for_plot = unique_sample_ids[:1]
    elif isinstance(sample_ids, int):
        if sample_ids < len(unique_sample_ids):
            selected_ids_for_plot = [unique_sample_ids[sample_ids]]
        else:
            warnings.warn(f"sample_idx {sample_ids} out of range. "
                          "Plotting first sample.")
            selected_ids_for_plot = unique_sample_ids[:1]
    elif isinstance(sample_ids, list):
        selected_ids_for_plot = [
            sid for sid in sample_ids if sid in unique_sample_ids
            ]
    if len(selected_ids_for_plot) ==0:
        vlog("No valid sample_idx found to plot. Defaulting to first available.",
             level=2, verbose=verbose, logger=_logger)
        selected_ids_for_plot = unique_sample_ids[:1]
    if len(selected_ids_for_plot) ==0 : # check if is empty
        vlog ("No sample to plot. Returning ...", level =1, verbose=verbose)
        return

    vlog(f"  Plotting for sample_idx: {selected_ids_for_plot}",
         level=4, verbose=verbose, logger=_logger)

    # --- Plotting Logic ---
    if kind == "temporal":
        num_plots = len(selected_ids_for_plot) * output_dim
        if num_plots == 0:
            vlog("No data to plot for temporal type.", level=2, 
                 verbose=verbose, logger=_logger)
            return

        n_cols_plot = min(max_cols, num_plots)
        n_rows_plot = (num_plots + n_cols_plot - 1) // n_cols_plot

        fig, axes = plt.subplots(
            n_rows_plot, n_cols_plot,
            figsize=(n_cols_plot * figsize[0],
                     n_rows_plot * figsize[1]),
            squeeze=False # Always return 2D array for axes
        )
        axes_flat = axes.flatten()
        plot_idx = 0

        for s_idx in selected_ids_for_plot:
            sample_df = df_to_plot[df_to_plot['sample_idx'] == s_idx]
            if sample_df.empty:
                continue

            for o_idx in range(output_dim):
                if plot_idx >= len(axes_flat): break # Should not happen
                ax = axes_flat[plot_idx]

                title = f"Sample ID: {s_idx}"
                if output_dim > 1:
                    title += f", Target Dim: {o_idx}"
                if titles and plot_idx < len(titles):
                    title = titles[plot_idx]
                ax.set_title(title)

                # Plot actuals if available
                actual_col = f"{base_actual_name}"
                if output_dim > 1: actual_col += f"_{o_idx}"
                actual_col += "_actual"
                if actual_col in sample_df.columns:
                    ax.plot(
                        sample_df['forecast_step'], sample_df[actual_col],
                        label=f'Actual {target_name}'
                              f'{f"_{o_idx}" if output_dim > 1 else ""}',
                        marker='o', linestyle='--'
                    )

                # Plot predictions
                if quantiles:
                    median_q_val = 0.5
                    if 0.5 not in quantiles: # Find closest if 0.5 not present
                        median_q_val = quantiles[len(quantiles) // 2]

                    median_col = f"{base_pred_name}"
                    if output_dim > 1: median_col += f"_{o_idx}"
                    median_col += f"_q{int(median_q_val*100)}"

                    lower_q_col = f"{base_pred_name}"
                    if output_dim > 1: lower_q_col += f"_{o_idx}"
                    lower_q_col += f"_q{int(sorted_quantiles[0]*100)}"

                    upper_q_col = f"{base_pred_name}"
                    if output_dim > 1: upper_q_col += f"_{o_idx}"
                    upper_q_col += f"_q{int(sorted_quantiles[-1]*100)}"

                    if median_col in sample_df.columns:
                        ax.plot(
                            sample_df['forecast_step'], sample_df[median_col],
                            label=f'Median (q{int(median_q_val*100)})',
                            marker='x',
                            **plot_kwargs.get("median_plot_kwargs", {})
                        )
                    if lower_q_col in sample_df.columns and \
                       upper_q_col in sample_df.columns:
                        ax.fill_between(
                            sample_df['forecast_step'],
                            sample_df[lower_q_col],
                            sample_df[upper_q_col],
                            color='gray', alpha=0.3,
                            label=f'Interval (q{int(sorted_quantiles[0]*100)}'
                                  f'-q{int(sorted_quantiles[-1]*100)})',
                            **plot_kwargs.get("fill_between_kwargs", {})
                        )
                else: # Point forecast
                    pred_col = f"{base_pred_name}"
                    if output_dim > 1: pred_col += f"_{o_idx}"
                    pred_col += "_pred"
                    if pred_col in sample_df.columns:
                        ax.plot(
                            sample_df['forecast_step'], sample_df[pred_col],
                            label=f'Predicted {target_name}'
                                  f'{f"_{o_idx}" if output_dim > 1 else ""}',
                            marker='x',
                            **plot_kwargs.get("point_plot_kwargs", {})
                        )
                ax.set_xlabel("Forecast Step into Horizon")
                ax.set_ylabel(
                    f"{target_name}{f' (Dim {o_idx})' if output_dim > 1 else ''}"
                    )
                ax.legend()
                ax.grid(True)
                plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes_flat)):
            axes_flat[i].set_visible(False)
        fig.tight_layout()
        plt.show()

    elif kind == "spatial":
        
        spatial_x_col = None 
        spatial_y_col =None 
        
        if spatial_cols is not None:
            spatial_cols = columns_manager (spatial_cols, empty_as_none=False)
            if len(spatial_cols )!=2: 
                raise ValueError(
                    "Spatial_cols need exactly two columns (e.g., longitude,"
                    f" latitude ). Got {spatial_cols}")
                
            spatial_x_col , spatial_y_col = spatial_cols 
    
        if spatial_x_col is None or spatial_y_col is None:
            raise ValueError(
                "`spatial_x_col` and `spatial_y_col` must be provided "
                "for `kind='spatial'`."
            )
        if spatial_x_col not in df_to_plot.columns or \
           spatial_y_col not in df_to_plot.columns:
            raise ValueError(
                f"Spatial columns '{spatial_x_col}' or '{spatial_y_col}' "
                "not found in forecast_df."
            )

        steps_to_plot = []
        if isinstance(horizon_steps, int):
            steps_to_plot = [horizon_steps]
        elif isinstance(horizon_steps, list):
            steps_to_plot = horizon_steps
        elif horizon_steps is None or \
             str(horizon_steps).lower() == "all":
            # Plot all unique forecast steps present in the data
            steps_to_plot = sorted(df_to_plot['forecast_step'].unique())
        else:
            raise ValueError("Invalid `horizon_steps`.")

        num_plots = len(steps_to_plot) * output_dim
        if num_plots == 0:
            vlog("No data/steps to plot for spatial type.",
                 level=2, verbose=verbose, logger=_logger)
            return
        
        vmin, vmax = None, None
        
        if cbar=="uniform": 
            vlog("Calculating uniform color scale for all subplots...",
                 level=2, verbose=verbose, logger=_logger)
                 
            #  Filter DataFrame to only the steps being plotted 
            df_for_scaling = df_to_plot[
                df_to_plot['forecast_step'].isin(steps_to_plot)
            ]
            
            # Identify all columns that will be used for coloring
            cols_to_scan = []
            if quantiles:
                median_q = 0.5
                if 0.5 not in quantiles:
                    median_q = sorted(quantiles)[len(quantiles) // 2]
                for o_idx in range(output_dim):
                    col_name = f"{base_pred_name}"
                    if output_dim > 1: col_name += f"_{o_idx}"
                    col_name += f"_q{int(median_q*100)}"
                    cols_to_scan.append(col_name)
            else: # Point forecasts
                for o_idx in range(output_dim):
                    col_name = f"{base_pred_name}"
                    if output_dim > 1: col_name += f"_{o_idx}"
                    col_name += "_pred"
                    cols_to_scan.append(col_name)

            existing_color_cols = [
                c for c in cols_to_scan if c in df_for_scaling.columns
            ]
            
            if existing_color_cols:
                # Calculate min/max on the filtered DataFrame 
                min_vals = [
                    df_for_scaling[c].dropna().min() 
                    for c in existing_color_cols]
                max_vals = [df_for_scaling[c].dropna().max() 
                            for c in existing_color_cols]
                
                if any(pd.notna(v) for v in min_vals):
                    vmin = min(v for v in min_vals if pd.notna(v))
                if any(pd.notna(v) for v in max_vals):
                    vmax = max(v for v in max_vals if pd.notna(v))
                
                if vmin is not None and vmax is not None:
                    vlog(f"Uniform color range set: vmin={vmin:.2f}, vmax={vmax:.2f}",
                         level=3, verbose=verbose, logger=_logger)

            
        n_cols_plot = min(max_cols, num_plots)
        n_rows_plot = (num_plots + n_cols_plot - 1) // n_cols_plot

        fig, axes = plt.subplots(
            n_rows_plot, n_cols_plot,
            figsize=(n_cols_plot * figsize[0],
                     n_rows_plot * figsize[1]),
            squeeze=False
        )
        axes_flat = axes.flatten()
        plot_idx = 0

        step_names = get_step_names(
            steps_to_plot, step_names =step_names, 
            default_name= "{}"
        )
        grid_props = grid_props or {'linestyle': ':', 'alpha': 0.7}
        for step in steps_to_plot:
            step_df = df_to_plot[df_to_plot['forecast_step'] == step]
            if step_df.empty:
                continue
            for o_idx in range(output_dim):
                if plot_idx >= len(axes_flat): break
                ax = axes_flat[plot_idx]

                # Determine column to use for color intensity
                color_col = None
                plot_title_suffix = ""
                if quantiles: # Use median for color
                    median_q_val = 0.5
                    if 0.5 not in quantiles:
                        median_q_val = quantiles[len(quantiles) // 2]
                    color_col = f"{base_pred_name}"
                    if output_dim > 1: color_col += f"_{o_idx}"
                    color_col += f"_q{int(median_q_val*100)}"
                    plot_title_suffix = f" (Median q{int(median_q_val*100)})"
                else: # Point forecast
                    color_col = f"{base_pred_name}"
                    if output_dim > 1: color_col += f"_{o_idx}"
                    color_col += "_pred"

                if color_col not in step_df.columns:
                    warnings.warn(f"Color column '{color_col}' not found "
                                  "for spatial plot. Skipping subplot.")
                    plot_idx +=1 # Increment to avoid reusing subplot
                    continue

                scatter_data = step_df.dropna(
                    subset=[spatial_x_col, spatial_y_col, color_col]
                    )
                if scatter_data.empty:
                    warnings.warn(f"No valid data to plot for step {step}, "
                                  f"output_dim {o_idx} after dropna. Skipping.")
                    plot_idx += 1
                    continue

                # Normalize color data for better visualization
                if cbar !='uniform': 
                    norm = mcolors.Normalize(
                        vmin=scatter_data[color_col].min(),
                        vmax=scatter_data[color_col].max()
                        )
                else: 
                    norm=None # norm will be applied for uniform scale
                
                cmap = plot_kwargs.get('cmap', 'viridis')
                step_name = step_names.get(step, '{}') # for consistency

                sc = ax.scatter(
                    scatter_data[spatial_x_col], 
                    scatter_data[spatial_y_col],
                    c=scatter_data[color_col], 
                    cmap=cmap, 
                    norm=norm,
                    s=plot_kwargs.get('s', 50), # Marker size
                    alpha=plot_kwargs.get('alpha', 0.7)
                )
                if cbar !='uniform': # use default behavior 
                    fig.colorbar(sc, ax=ax, label=f"{target_name}{plot_title_suffix}")
                
                title = step_name.format(f"Forecast Step: {step}")
                
                if output_dim > 1: 
                    title += f", Target Dim: {o_idx}"
                    
                if titles and plot_idx < len(titles):
                    title = titles[plot_idx]
                    
                ax.set_title(title)
                ax.set_xlabel(spatial_x_col)
                ax.set_ylabel(spatial_y_col)
                
                if show_grid: 
                    ax.grid(True, **grid_props)
                else: 
                    ax.grid(False) 
                
                plot_idx += 1
        
        for i in range(plot_idx, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        fig.tight_layout()
        
        if cbar=='uniform':
            # This creates one colorbar for the whole
            # figure if uniform scale was used.
            
            # Adjust the main plot area to make space for the colorbar
            # This leaves 10% space on the right for the cbar
            fig.subplots_adjust(right=0.90) 
            
            # Create a new axis for the colorbar
            # [left, bottom, width, height] in figure coordinates
            cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            mappable = ScalarMappable(norm=norm, cmap=cmap)
        
            # Add colorbar to the figure
            fig.colorbar(
                mappable, 
                cax=cax,
               # ax=axes.ravel().tolist(), # Associate with all axes
                label=f"{target_name} ({plot_title_suffix.strip()})",
                orientation='vertical',
                shrink=0.8, 
                pad=0.04
            )
    
        
        plt.show()

    else:
        raise ValueError(
            f"Unsupported `kind`: '{kind}'. "
            "Choose 'temporal' or 'spatial'."
            )
    vlog("Forecast visualization complete.", level=3, 
         verbose=verbose, logger=_logger)
    

@check_non_emptiness
def visualize_forecasts(
    forecast_df,
    dt_col,
    tname,
    test_data=None,
    eval_periods=None,      
    mode="quantile",       
    kind="spatial", 
    actual_name=None,        
    x=None,                
    y=None,                
    cmap="coolwarm",       
    max_cols=3, 
    axis="on",  
    s=2, 
    show_grid=True,
    grid_props=None,           
    verbose=1,
    **kw
):
    r"""
    Visualize forecast results and actual test data for one or more
    evaluation periods.

    The function plots a grid of scatter plots comparing actual values
    with forecasted predictions. Each evaluation period yields two plots:
    one for actual values and one for predicted values. If multiple
    evaluation periods are provided, the grid layout wraps after
    ``max_cols`` columns.

    .. math::

       \hat{y}_{t+i} = f\Bigl(
       X_{\text{static}},\;X_{\text{dynamic}},\;
       X_{\text{future}}\Bigr)

    for :math:`i = 1, \dots, N`, where :math:`N` is the forecast horizon.

    Parameters
    ----------
    forecast_df : pandas.DataFrame
        DataFrame containing forecast results with a time column,
        spatial coordinates, and prediction columns.
    dt_col      : str
        Name of the time column used to filter forecast results (e.g.
        ``"year"``).
    tname : str
        Target variable name used to construct forecast columns (e.g.
        ``"subsidence"``). This argument is required.
        
        
    eval_periods : scalar or list, optional
        Evaluation period(s) used to select forecast results. If set to
        ``None``, the function selects up to three unique periods from
        ``test_data[dt_col]``.
    mode        : str, optional
        Forecast mode. Must be either ``"quantile"`` or ``"point"``.
        Default is ``"quantile"``.
    kind        : str, optional
        Type of visualization. If ``"spatial"``, spatial columns are
        required; otherwise, the provided `x` and `y` columns are used.
    x : str, optional
        Column name for the x-axis. For non-spatial plots, this must be
        provided or will be inferred via ``assert_xy_in``.
    y : str, optional
        Column name for the y-axis. For non-spatial plots, this must be
        provided or will be inferred via ``assert_xy_in``.
    cmap : str, optional
        Colormap used for scatter plots. Default is ``"coolwarm"``.
    max_cols : int, optional
        Maximum number of evaluation periods to plot per row. If the
        number of periods exceeds ``max_cols``, a new row is started.
    axis: str, optional, 
       Wether to keep the axis of set it to False. 
    show_grid: bool, default=True, 
       Visualize the grid 
    grid_props: dict, optional 
       Grid properties for visualizations. If none the properties is 
       infered as ``{"linestyle":":", 'alpha':0.7}``.
    verbose : int, optional
        Verbosity level. Controls the amount of output printed.

    Returns
    -------
    None
        The function displays the visualization plot.

    Examples
    --------
    Example 1: **Spatial Visualization**
    
    In this example, we visualize the forecasted and actual values of the
    **subsidence** target variable, using **longitude** and **latitude**
    for the spatial coordinates. We visualize the results for two
    evaluation periods (2023 and 2024), using **quantile** mode for the forecast.

    >>> from fusionlab.plot.forecast import visualize_forecasts
    >>> forecast_results = pd.DataFrame({
    >>>     'longitude': [-103.808151, -103.808151, -103.808151],
    >>>     'latitude': [0.473152, 0.473152, 0.473152],
    >>>     'subsidence_q50': [0.3, 0.4, 0.5],
    >>>     'subsidence': [0.35, 0.42, 0.49],
    >>>     'date': ['2023-01-01', '2023-01-02', '2023-01-03']
    >>> })
    >>> test_data = pd.DataFrame({
    >>>     'longitude': [-103.808151, -103.808151, -103.808151],
    >>>     'latitude': [0.473152, 0.473152, 0.473152],
    >>>     'subsidence': [0.35, 0.41, 0.49],
    >>>     'date': ['2023-01-01', '2023-01-02', '2023-01-03']
    >>> })
    >>> visualize_forecasts(
    >>>     forecast_df=forecast_results,
    >>>     test_data=test_data,
    >>>     dt_col="date",
    >>>     tname="subsidence",
    >>>     eval_periods=[2023, 2024],
    >>>     mode="quantile",
    >>>     kind="spatial",
    >>>     cmap="coolwarm",
    >>>     max_cols=2,
    >>>     verbose=1
    >>> )
    
    Example 2: **Non-Spatial Visualization**
    
    In this example, we visualize the forecasted and actual values of the
    **subsidence** target variable in a **non-spatial** context. The columns
    `longitude` and `latitude` are still provided but used for non-spatial
    x and y axes. Evaluation is for 2023.

    >>> from fusionlab.plot.forecast import visualize_forecasts
    >>> forecast_results = pd.DataFrame({
    >>>     'longitude': [-103.808151, -103.808151, -103.808151],
    >>>     'latitude': [0.473152, 0.473152, 0.473152],
    >>>     'subsidence_pred': [0.35, 0.41, 0.48],
    >>>     'subsidence': [0.36, 0.43, 0.49],
    >>>     'date': ['2023-01-01', '2023-01-02', '2023-01-03']
    >>> })
    >>> test_data = pd.DataFrame({
    >>>     'longitude': [-103.808151, -103.808151, -103.808151],
    >>>     'latitude': [0.473152, 0.473152, 0.473152],
    >>>     'subsidence': [0.36, 0.42, 0.50],
    >>>     'date': ['2023-01-01', '2023-01-02', '2023-01-03']
    >>> })
    >>> forecast_df_point = visualize_forecasts(
    >>>     forecast_df=forecast_results,
    >>>     test_data=test_data,
    >>>     dt_col="date",
    >>>     tname="subsidence",
    >>>     eval_periods=[2023],
    >>>     mode="point",
    >>>     kind="non-spatial",
    >>>     x="longitude",
    >>>     y="latitude",
    >>>     cmap="viridis",
    >>>     max_cols=1,
    >>>     axis="off",
    >>>     show_grid=True,
    >>>     grid_props={"linestyle": "--", "alpha": 0.5},
    >>>     verbose=2
    >>> )

    Notes
    -----
    - In ``quantile`` mode, the function uses the column
      ``<tname>_q50`` for visualization.
    - In ``point`` mode, the column ``<tname>_pred`` is used.
    - For spatial visualizations, if ``x`` and ``y`` are not provided,
      they default to ``"longitude"`` and ``"latitude"``.
    - The evaluation period(s) are determined by filtering
      ``forecast_df[dt_col] == <eval_period>``.
    - Use ``assert_xy_in`` to validate that the x and y columns exist in
      the provided DataFrames.

    See Also
    --------
    generate_forecast : Function to generate forecast results.
    coverage_score   : Function to compute the coverage score.

    References
    ----------
    .. [1] Kouadio L. et al., "Gofast Forecasting Model", Journal of
       Advanced Forecasting, 2025. (In review)
    """
    # ******************************************************
    from ..utils.ts_utils import filter_by_period
    # *********************************************************

    # Check that forecast_df is a valid DataFrame
    is_frame (
        forecast_df, 
        df_only=True,
        objname="Forecast data", 
        error="raise"
    )
    if eval_periods is None:
        unique_periods = sorted(forecast_df[dt_col].unique())
        if verbose:
            print("No eval_period provided; using up to three unique " +
                  "periods from forecast data.")
        eval_periods = unique_periods[:3]
    
    eval_periods = columns_manager(eval_periods, to_string=True )
    # Check if test_data is provided, else set it to None
    if test_data is not None: 
        is_frame (
            test_data, 
            df_only=True,
            objname="Test data", 
            error="raise"
        )
        # filterby periods ensure Ensure dt_col is in Pandas datetime format
        test_data =filter_by_period (test_data, eval_periods, dt_col)

    forecast_df =filter_by_period (forecast_df, eval_periods, dt_col)
 
    # Convert eval_periods to Pandas datetime64[ns] format
    # # Ensure dtype match before filtering
    eval_periods= forecast_df[dt_col].astype(str).unique()

    # Determine x and y columns for spatial or non-spatial visualization
    if kind == "spatial":
        if x is None and y is None:
            x, y = "longitude", "latitude"
        check_spatial_columns(forecast_df, spatial_cols=(x, y ))
        x, y = assert_xy_in(x, y, data=forecast_df, asarray=False)
    else:
        if x is None or y is None:
            raise ValueError("For non-spatial kind, both x and y must be provided.")
        x, y = assert_xy_in(x, y, data=forecast_df, asarray=False)

    # Set prediction column based on forecast mode
    if mode == "quantile":
        pred_col   = f"{tname}_q50"
        pred_label = f"Predicted {tname} (q50)"
    elif mode == "point":
        pred_col   = f"{tname}_pred"
        pred_label = f"Predicted {tname}"
    else:
        raise ValueError("Mode must be either 'quantile' or 'point'.")

    # XXX  # restore back to origin_dtype before 

    # Loop over evaluation periods and plot
    df_actual = test_data if test_data is not None else forecast_df 
    
    actual_name = get_actual_column_name (
        df_actual, tname, actual_name=actual_name , 
        default_to='tname', 
    )
    
    # Compute global min-max for color scale
    # for all plot.
    vmin = forecast_df[pred_col].min()
    vmax = forecast_df[pred_col].max()
    
    if test_data is not None and actual_name in test_data.columns:
        vmin = min(vmin, test_data[actual_name].min())
        vmax = max(vmax, test_data[actual_name].max())

    # Determine common periods in both forecast_df and test_data (if available)
    if test_data is not None: 
        available_periods = is_in_if (
            forecast_df[dt_col].astype(str).unique(), 
            test_data[dt_col].astype(str).unique(), 
            return_intersect=True, 
            )
        # available_periods = sorted(set(forecast_df[dt_col]) & set(test_data[dt_col]))
    else:
        # sorted(forecast_df[dt_col].astype(str).unique())
        available_periods = eval_periods 
    
    # Ensure the eval_periods only contain periods available in the data
    eval_periods = [p for p in eval_periods if p in available_periods]
    
    if len(eval_periods) == 0:
        raise ValueError(
            "[ERROR] No valid evaluation periods found in forecast or test data.")
    
    # Compute subplot grid dimensions
    n_periods = len(eval_periods)
    n_cols = min(n_periods, max_cols)
    n_rows = int(np.ceil(n_periods / max_cols))
    
    # Two rows per evaluation period if 
    # test_data is passed or is not empty 
    total_rows = n_rows * 2  if test_data is not None else n_rows 
    
    # Create subplot grid
    fig, axes = plt.subplots(
        total_rows, n_cols, figsize=(5 * n_cols, 4 * total_rows)
    )
    
    # Ensure `axes` is a 2D array for consistent indexing
    if total_rows == 1 and n_cols == 1:
        pass # axes = np.array([[axes]])
    elif total_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(total_rows, 1)

    for idx, period in enumerate(eval_periods):
        # Try to reconvert 
        col_idx = idx % n_cols
        row_idx = (idx // n_cols) * 2 if test_data is not None else (idx // n_cols)

        # Filter data for the current period using 'isin' for robustness
        forecast_subset = forecast_df[forecast_df[dt_col].isin([period])]
        
        if test_data is not None:
            test_subset = test_data[test_data[dt_col].isin([period])]
            # test_subset =filter_by_period (test_data, period, dt_col)
        else:
            test_subset = forecast_subset  # If no test_data, use forecast_df itself
        
        if forecast_subset.empty or test_subset.empty:
            if verbose:
                print(f"[WARNING] No data for period {period}; skipping.")
            continue
        
        # Plot actual values
        if test_data is not None: 
            ax_actual = axes[row_idx, col_idx]
            sc_actual = ax_actual.scatter(
                test_subset[x.name],
                test_subset[y.name],
                c=test_subset[actual_name],
                cmap=cmap,
                alpha=0.7,
                edgecolors='k', 
                s=s, 
                vmin=vmin,  
                vmax=vmax,  
                **kw
            )
            ax_actual.set_title(f"Actual {tname.capitalize()} ({period})")
            ax_actual.set_xlabel(x.name.capitalize())
            ax_actual.set_ylabel(y.name.capitalize())
            if axis == "off": 
                ax_actual.set_axis_off()
            else: 
                ax_actual.set_axis_on()
    
            fig.colorbar(sc_actual, ax=ax_actual, label=tname.capitalize())
            if show_grid: 
                if grid_props is None: 
                    grid_props = {"linestyle": ":", 'alpha': 0.7}
                ax_actual.grid(True, **grid_props)

        # Plot predicted values
        ax_pred = axes[row_idx + 1, col_idx
                       ] if test_data is not None else axes[row_idx, col_idx]
        # ax_pred = axes[row_idx + 1, col_idx]
        sc_pred = ax_pred.scatter(
            forecast_subset[x.name],
            forecast_subset[y.name],
            c=forecast_subset[pred_col],
            cmap=cmap,
            alpha=0.7,
            edgecolors='k',
            s=s, 
            vmin=vmin,  # Apply global min
            vmax=vmax,  # Apply global max
            **kw
        )
        ax_pred.set_title(f"{pred_label} ({period})")
        ax_pred.set_xlabel(x.name.capitalize())
        ax_pred.set_ylabel(y.name.capitalize())
        if axis == "off": 
            ax_pred.set_axis_off()
        else: 
            ax_pred.set_axis_on()

        if show_grid: 
            if grid_props is None: 
                grid_props = {"linestyle": ":", 'alpha': 0.7}
            ax_pred.grid(True, **grid_props)

        fig.colorbar(sc_pred, ax=ax_pred, label=pred_label)

    plt.tight_layout()
    plt.show()


def _get_metrics_from_cols(
    columns: List[str], prefixes: List[str]
) -> List[str]:
    """Extracts metric suffixes (e.g., q10, actual) from column names."""
    metrics = set()
    for col in columns:
        for p in prefixes:
            if col.startswith(p + '_'):
                metrics.add(col[len(p)+1:])
    return sorted(list(metrics))

def _parse_wide_df_columns(
    df_wide: pd.DataFrame,
    value_prefixes: List[str]
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Parses wide-format columns into a structured dictionary."""
    plot_structure = {prefix: {} for prefix in value_prefixes}
    
    # Regex for columns with years, e.g., GWL_2022_q10 or GWL_2022_actual
    pattern_year = re.compile(
        r'(' + '|'.join(re.escape(p) for p in value_prefixes) +
        r')_(\d{4})_?(.*)'
    )
    # Regex for columns without years, e.g., GWL_q10 or GWL_pred
    pattern_no_year = re.compile(
        r'(' + '|'.join(re.escape(p) for p in value_prefixes) +
        r')_([a-zA-Z].*)'
    )
    
    for col in df_wide.columns:
        match_year = pattern_year.match(col)
        match_no_year = pattern_no_year.match(col)

        if match_year:
            prefix, year, suffix = match_year.groups()
            # If suffix is empty, it's a point prediction
            suffix = suffix or 'pred' 
            if year not in plot_structure[prefix]:
                plot_structure[prefix][year] = {}
            plot_structure[prefix][year][suffix] = col
        elif match_no_year: 
            # Handles columns like 'subsidence_q10',
            # 'subsidence_actual'
            prefix, suffix = match_no_year.groups()
            if "static" not in plot_structure[prefix]:
                 plot_structure[prefix]["static"] = {}
            plot_structure[prefix]["static"][suffix] = col
        elif any(col == p for p in value_prefixes): 
            # Handles base prefix column e.g. 'subsidence'
             if "static" not in plot_structure[col]:
                 plot_structure[col]["static"] ={}
             plot_structure[col]['static']['pred'] = col
             
    return plot_structure

def _plot_spatial_subplot(ax, df, x_col, y_col, c_col, s= 10,  **kwargs):
    """Helper to create a single scatter subplot."""
    if c_col is None or c_col not in df.columns:
        ax.set_title(f"{kwargs.get('title', '')}\n(Data not found)",
                     fontsize=10, color='red')
        ax.axis('off')
        return None
        
    plot_df = df[[x_col, y_col, c_col]].dropna()
    if plot_df.empty:
        ax.set_title(f"{kwargs.get('title', '')}\n(No valid data)", 
                     fontsize=10)
        ax.axis('off')
        return None
        
    scatter = ax.scatter(
        plot_df[x_col], plot_df[y_col], c=plot_df[c_col],
        cmap=kwargs.get('cmap'), vmin=kwargs.get('vmin'),
        vmax=kwargs.get('vmax'), s=s,
        edgecolors='k', linewidths=0.1, alpha=0.8
    )
    ax.set_title(kwargs.get('title', ''), fontsize=10)
    
    if kwargs.get('axis_off'):
        ax.axis('off')
    else:
        ax.tick_params(axis='both', which='major', labelsize=8)
        if kwargs.get('show_grid'):
            ax.grid(**kwargs.get('grid_props'))
    return scatter

def _plot_temporal_subplot(ax, df, value_col, title, **kwargs):
    """Helper to create a single line plot (fallback)."""
    if value_col is None or value_col not in df.columns:
        ax.set_title(f"{title}\n(Data not found)", fontsize=10, color='red')
        ax.axis('off')
        return
        
    plot_df = df[[value_col]].dropna()
    if plot_df.empty:
        ax.set_title(f"{title}\n(No valid data)", fontsize=10)
        ax.axis('off')
        return

    ax.plot(plot_df.index, plot_df[value_col])
    ax.set_title(title, fontsize=10)
    if not kwargs.get('axis_off'):
        ax.tick_params(axis='both', which='major', labelsize=8)
        if kwargs.get('show_grid'):
            ax.grid(**kwargs.get('grid_props'))

def _plot_forecast_grid(
    fig, axes, df_wide, plot_structure, years_to_plot,
    quantiles_to_plot, prefix, kind, spatial_cols, s, plot_kwargs, 
    _logger
):
    # Check if spatial coordinates are available for plotting.
    has_spatial_coords = all(
        c in df_wide.columns for c in spatial_cols
    )
    # Log a fallback message if spatial coordinates are not found.
    if not has_spatial_coords:
        vlog(
            f"Spatial columns {spatial_cols} not found. "
            "Falling back to temporal line plots.",
            level=1, verbose=plot_kwargs.get('verbose', 0), 
            logger=_logger
        )

    # Get the last known static actual column for fallback.
    last_known_actual_col = plot_structure[prefix].get(
        "static", {}).get("actual")

    # Iterate through each year to create a row of plots.
    for row_idx, year in enumerate(years_to_plot):
        ax_row = axes[row_idx]
        col_idx = 0
        
        # Plot actual data if 'dual' mode is enabled.
        if kind == 'dual':
            actual_col = plot_structure[prefix].get(
                year, {}).get('actual', last_known_actual_col)
            
            if actual_col:
                last_known_actual_col = actual_col
            
            title = f"Actual ({year})"
            plot_kwargs['title'] = title
            
            # Choose plotting function based on coordinate availability.
            if has_spatial_coords:
                _plot_spatial_subplot(
                    ax_row[col_idx], df_wide, *spatial_cols,
                    actual_col, s =s, **plot_kwargs
                )
            else:
                _plot_temporal_subplot(
                    ax_row[col_idx], df_wide, actual_col, **plot_kwargs
                )
            col_idx += 1
        
        # Plot each requested prediction/quantile.
        for q_suffix in quantiles_to_plot:
            if col_idx >= len(ax_row): continue
            
            pred_col = plot_structure[prefix].get(year, {}).get(q_suffix)
            
            # Format the title for the subplot.
            title_suffix = (
                q_suffix.replace('q', 'Q').capitalize() 
                if 'q' in q_suffix else "Prediction"
            )
            title = f"{title_suffix} ({year})"
            plot_kwargs['title'] = title
            
            # Choose plotting function based on coordinate availability.
            if has_spatial_coords:
                _plot_spatial_subplot(
                    ax_row[col_idx], df_wide, *spatial_cols,
                    pred_col, s=s,  **plot_kwargs
                )
            else:
                _plot_temporal_subplot(
                    ax_row[col_idx], df_wide, pred_col, **plot_kwargs
                )
            col_idx += 1
        
        # Turn off any remaining unused axes in the current row.
        for i in range(col_idx, len(ax_row)):
            ax_row[i].axis('off')

def _plot_single_scatter(
    ax, df, x_col, y_col, c_col, cmap, vmin, vmax, title, 
    axis_off, show_grid, grid_props
):
    """Helper to create a single scatter subplot."""
    if c_col not in df.columns:
        ax.set_title(f"{title}\n(Data not found)",
                     fontsize=10, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        return None
        
    # Drop NaNs for this specific plot to avoid plotting empty points
    plot_df = df[[x_col, y_col, c_col]].dropna()
    
    if plot_df.empty:
        ax.set_title(f"{title}\n(No valid data)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return None
        
    scatter = ax.scatter(
        plot_df[x_col], plot_df[y_col], c=plot_df[c_col],
        cmap=cmap, vmin=vmin, vmax=vmax, s=10
    )
    ax.set_title(title, fontsize=10)
    
    if axis_off:
        ax.axis('off')
    else:
        ax.tick_params(axis='both', which='major', labelsize=8)
        if show_grid:
            ax.grid(**grid_props)
    return scatter
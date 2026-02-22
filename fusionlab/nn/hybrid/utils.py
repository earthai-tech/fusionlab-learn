# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Net Model utilities.
"""
from __future__ import annotations

import warnings
import os
import numpy as np 

from typing import List, Optional, Union, Dict
import matplotlib.pyplot as plt

from ...core.handlers import _get_valid_kwargs 
from .. import KERAS_DEPS, KERAS_BACKEND

if KERAS_BACKEND:
    History =KERAS_DEPS.History 
else:
    class History:
        def __init__(self):
            self.history = {}


__all__ = ['select_mode', 'plot_history_in', 'plot_history']

def select_mode(
    mode: Union[str, None] = None,
    default: str = "pihal_like",
) -> str:
    r"""
    Resolve a user‑supplied *mode* string to the canonical value
    ``'pihal'`` or ``'tft'``.

    The helper is used throughout
    ``fusionlab.nn.models.utils`` to decide whether the model should
    follow the PIHALNet or the Temporal Fusion Transformer (TFT)
    convention for handling :pydata:`future_features`.

    Parameters
    ----------
    mode : Union[str, None], optional
        Case‑insensitive keyword.  Accepted values are

        * ``'pihal'``        or ``'pihal_like'``
        * ``'tft'``          or ``'tft_like'``
        * *None*             – fall back to *default*.
    default : {'pihal', 'tft'}, default ``'pihal'``
        Canonical value returned when *mode* is *None*.

    Returns
    -------
    str
        ``'pihal_like'`` or ``'tft_like'``.

    Raises
    ------
    ValueError
        If *mode* is not *None* and does not match any accepted
        keyword.

    Notes
    -----
    * ``'..._like'`` aliases are provided for backward compatibility
      with earlier API versions.
    * The function strips whitespace and converts *mode* to lower
      case before matching.

    Examples
    --------
    >>> select_mode('TFT_like')
    'tft'
    >>> select_mode(None, default='tft')
    'tft'
    >>> select_mode('invalid')
    Traceback (most recent call last):
        ...
    ValueError: Invalid mode 'invalid'. Choose one of: pihal, ...

    See Also
    --------
    fusionlab.nn.pinn.PIHALNet.call
        Uses the resolved mode to slice *future_features*.
    fusionlab.nn.pinn.HLNet
        High‑level model wrapper that exposes the *mode* argument.

    References
    ----------
    * Lim, B. et al. *Temporal Fusion Transformers for
      Interpretable Multi‑horizon Time Series Forecasting.*
      NeurIPS 2021.
    * Kouadio, L. K. et al. *Physics‑Informed Heterogeneous Attention
      Learning for Spatio‑Temporal Subsidence Prediction.*
      IEEE T‑PAMI 2025 (in press).
    """

    canonical = {"pihal": "pihal_like", "pihal_like": "pihal_like",
                 "tft": "tft_like", "tft_like": "tft_like", 
                 "tft-like": "tft_like","pihal-like": "pihal_like",}

    if mode is None:
        return canonical[default]

    try:
        return canonical[str(mode).lower().strip()]
    except KeyError:  # unknown keyword
        valid = ", ".join(sorted(canonical.keys()))
        raise ValueError(
            f"Invalid mode '{mode}'. Choose one of: {valid} or None."
        ) from None


def plot_history_in(
    history: Union[History, Dict],
    metrics: Optional[Dict[str, List[str]]] = None,
    layout: str = 'subplots',
    title: str = "Model Training History",
    figsize: Optional[tuple] = None,
    style: str = 'default',
    savefig: Optional[str] = None,
    max_cols: Union[int, str] = 'auto',
    show_grid: bool = True,
    grid_props: Optional[Dict] = None,
    yscale_settings: Optional[Dict[str, str]] = None,
    log_fn =None,  
    **plot_kwargs
) -> None:
    r"""Visualizes the training and validation history of a Keras model.

    This function creates plots for loss and other specified metrics
    from a Keras History object or a dictionary, allowing for easy
    comparison of training and validation performance over epochs.

    Parameters
    ----------
    history : :class:`keras.callbacks.History` or dict
        The History object returned by the ``fit`` method of a Keras
        model, or a dictionary with the same structure, e.g.,
        ``{'loss': [0.1, 0.05], 'val_loss': [0.12, 0.08]}``.

    metrics : dict, optional
        A dictionary specifying which metrics to plot. The keys
        become the subplot titles, and the values are lists of
        metric keys from the history dictionary. If ``None``, the
        function auto-detects all available metrics and groups
        them into sensible subplots.

        .. code-block:: python

           # Example for PIHALNet:
           metrics_to_plot = {
               "Loss Components": ["total_loss", "data_loss", "physics_loss"],
               "Subsidence MAE": ["subs_pred_mae"],
           }

    layout : {'single', 'subplots'}, default 'subplots'
        Determines the plot layout:
            
        - ``'single'``: All specified metrics are plotted on a single
          graph. This is useful for comparing the trends of different
          losses together.
        - ``'subplots'``: Each key in the `metrics` dictionary gets
          its own dedicated subplot in a grid.

    title : str, default "Model Training History"
        The main title for the entire figure, displayed at the top.

    figsize : tuple of (float, float), optional
        The size of the figure in inches (width, height). If ``None``,
        a suitable size is automatically calculated based on the
        number of subplots and the `layout`.

    style : str, default 'default'
        The Matplotlib style to use for the plot. For a list of
        available styles, use ``plt.style.available``. Using styles
        like 'seaborn-v0_8-whitegrid' or 'ggplot' can enhance
        readability.

    savefig : str, optional
        If a file path is provided (e.g., ``'output/training.png'``),
        the figure will be saved to that location. The directory will
        be created if it does not exist.

    max_cols : int or 'auto', default 'auto'
        The maximum number of subplots per row when `layout` is
        'subplots'. If ``'auto'``, it defaults to 2 for a balanced
        layout.

    show_grid : bool, default True
        If ``True``, a grid is displayed on each subplot, which can
        aid in reading metric values.

    grid_props : dict, optional
        A dictionary of properties to pass to ``ax.grid()`` for
        customizing the grid appearance. For example:
        ``{'linestyle': '--', 'alpha': 0.6}``.
        
    yscale_settings : dict, optional
        A dictionary mapping subplot titles (keys from `metrics` dict)
        to their desired y-axis scale (e.g., 'linear' or 'log').
        This allows for per-subplot scale control.

        .. code-block:: python
        
           yscale_settings = {
               "Loss Components": "log",
               "Component Losses": "log",
               "Subsidence MAE": "linear" 
           }

    **plot_kwargs : dict
        Additional keyword arguments to pass directly to the
        Matplotlib ``ax.plot()`` function. This allows for detailed
        customization of the plotted lines, such as ``linewidth``,
        ``marker``, or ``color``.

    Returns
    -------
    None
        This function does not return any value. It displays and/or
        saves the Matplotlib figure directly.
        
    See Also
    --------
    fusionlab.plot.forecast.forecast_view : For visualizing predictions.

    Notes
    -----
    - The function automatically detects corresponding validation
      metrics (e.g., ``val_loss`` for ``loss``) and plots them on
      the same axes with a dashed line for easy comparison between
      training and validation performance.
    - If `metrics` is not provided, the function will attempt to
      intelligently group all available metrics from the history
      object, with all loss-related keys being plotted together.

    Examples
    --------
    >>> from tensorflow.keras.callbacks import History
    >>> import numpy as np
    >>> # from fusionlab.nn.models.utils import plot_history_in

    >>> # --- Example 1: Standard Model History on Subplots ---
    >>> history_standard = History()
    >>> history_standard.history = {
    ...     'loss': np.linspace(1.0, 0.2, 20),
    ...     'val_loss': np.linspace(1.1, 0.3, 20),
    ...     'mae': np.linspace(0.8, 0.15, 20),
    ...     'val_mae': np.linspace(0.85, 0.25, 20),
    ... }
    >>> plot_history_in(
    ...      history_standard,
    ...      layout='subplots',
    ...      title='Standard Model Training History',
    ...      max_cols=2
    ...  )

    >>> # --- Example 2: PIHALNet Loss Components on a Single Plot ---
    >>> history_pinn = {
    ...     'total_loss': np.exp(-np.arange(0, 2, 0.1)),
    ...     'val_total_loss': np.exp(-np.arange(0, 2, 0.1)) * 1.1,
    ...     'data_loss': np.exp(-np.arange(0, 2, 0.1)) * 0.6,
    ...     'physics_loss': np.exp(-np.arange(0, 2, 0.1)) * 0.4,
    ... }
    >>> pinn_metrics = {"Loss Components": ["total_loss", "data_loss", "physics_loss"]}
    >>> plot_history_in(
    ...      history_pinn,
    ...      metrics=pinn_metrics,
    ...      layout='single',
    ...      title='PIHALNet Loss Breakdown'
    ...  )
    >>> # --- Optional Post-Training Utilities ---
    >>> subsmodel_metrics = {
    ...    "Total Loss": ["loss", "val_loss"],
    ...    "Physics Loss": ["physics_loss", "val_physics_loss"],
    ...    "Data Loss": ["data_loss", "val_data_loss"],
    ...    "Component Losses": [
    ...        "consolidation_loss", "val_consolidation_loss",
    ...        "gw_flow_loss", "val_gw_flow_loss",
    ...        "prior_loss", "val_prior_loss",
    ...        "smooth_loss", "val_smooth_loss"
    ...    ],
    ...    "Subsidence MAE": ["subs_pred_mae", "val_subs_pred_mae"],
    ...    "GWL MAE": ["gwl_pred_mae", "val_gwl_pred_mae"],
    ... }
    >>> # --- NEW: Define the scale for each plot ---
    >>> plot_scales = {
    ...    "Total Loss": "log",
    ...    "Physics Loss": "log",
    ...    "Data Loss": "log",
    ...    "Component Losses": "log",
    ...    "Subsidence MAE": "linear", # Keep MAE linear to see it approach 0
    ...    "GWL MAE": "linear"        # Keep MAE linear
    ... }
    >>> print("\n--- SubsModel Training History on Separate Subplots ---")
    >>> plot_history_in(
    ...    history.history,
    ...    metrics=subsmodel_metrics,
    ...    layout='single',
    ...    title=f"{MODEL_NAME} Training History",
    ...    yscale_settings=plot_scales, # <--- PASS THE NEW PARAMETER HERE
    ...    savefig=os.path.join(
    ...        RUN_OUTPUT_PATH,
    ...        f"{CITY_NAME}_{MODEL_NAME.lower()}_training_history_plot",
    ...    ),
    ...)
    """
    log = log_fn if log_fn is not None else print 

    if isinstance(history, History):
        history_dict = history.history
    elif isinstance(history, dict):
        history_dict = history
    else:
        raise TypeError(
            "`history` must be a Keras History object or a dictionary."
        )

    if not history_dict:
        warnings.warn("The history dictionary is empty. Nothing to plot.")
        return

    try:
        plt.style.use(style)
    except OSError:
        warnings.warn(
            f"Style '{style}' not found. See `plt.style.available` "
            "for options. Falling back to 'default' style."
        )
        plt.style.use('default')

    if metrics is None:
        metrics = {}
        for key in history_dict.keys():
            if not key.startswith('val_'):
                base_metric_name = key.replace('_', ' ').title()
                group_name = "Losses" if "loss" in key.lower() else base_metric_name
                if group_name not in metrics:
                    metrics[group_name] = []
                metrics[group_name].append(key)
        
        if "Losses" in metrics:
             metrics["Losses"].sort(
                 key=lambda x: (x != 'loss' and x != 'total_loss', x)
             )

    n_plots = len(metrics)
    if n_plots == 0:
        warnings.warn("No valid metrics found to plot.")
        return

    # --- Plotting Setup ---
    if layout == 'subplots':
        n_cols_resolved = 2 if max_cols == 'auto' else int(max_cols)
        n_cols = min(n_cols_resolved, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        if figsize is None:
            figsize = (n_cols * 6, n_rows * 5)
    else: # layout == 'single'
        n_rows, n_cols = 1, 1
        if figsize is None:
            figsize = (10, 6)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    fig.suptitle(title, fontsize=16, weight='bold')
    
    grid_properties = grid_props or {'linestyle': ':', 'alpha': 0.7}
    
    # --- New yscale_settings logic ---
    if yscale_settings is None:
        yscale_settings = {}
    
    # --- Revised Plotting Logic ---
    if layout == 'single':
        ax = axes_flat[0]
        # Collect all metric keys from all groups to plot on one axis
        all_metric_keys = [
            key for group in metrics.values() for key in group
        ]
        for metric in all_metric_keys:
            if metric not in history_dict:
                warnings.warn(f"Metric '{metric}' not found. Skipping.")
                continue
            
            epochs = range(1, len(history_dict[metric]) + 1)
            plot_kwargs = _get_valid_kwargs (ax.plot, plot_kwargs)
            ax.plot(
                epochs, history_dict[metric],
                label=f'Train {metric.replace("_", " ").title()}',
                **plot_kwargs
            )
            val_metric = f'val_{metric}'
            if val_metric in history_dict:
                ax.plot(
                    epochs, history_dict[val_metric], linestyle='--',
                    label=f'Val {metric.replace("_", " ").title()}',
                    **plot_kwargs
                )
        
        subplot_title = next(iter(metrics.keys())) if n_plots == 1 else "All Metrics"
        ax.set_title(subplot_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend()
        if show_grid:
            ax.grid(**grid_properties)
        
        # --- Apply yscale to single plot ---
        # Use the scale specified for the first metric group, or default to linear
        first_group_scale = yscale_settings.get(subplot_title, 'linear')
        ax.set_yscale(first_group_scale)
        if first_group_scale == 'log':
            ax.set_ylabel("Value (log scale)")
            
    else: # layout == 'subplots'
        plot_idx = 0
        for subplot_title, metric_keys in metrics.items():
            if plot_idx >= len(axes_flat): break
            ax = axes_flat[plot_idx]
            
            for metric in metric_keys:
                if metric not in history_dict:
                    warnings.warn(f"Metric '{metric}' not found. Skipping.")
                    continue

                epochs = range(1, len(history_dict[metric]) + 1)
                plot_kwargs = _get_valid_kwargs (ax.plot, plot_kwargs)
                ax.plot(
                    epochs, history_dict[metric],
                    label=f'Train {metric.replace("_", " ").title()}',
                    **plot_kwargs
                )
                val_metric = f'val_{metric}'
                if val_metric in history_dict:
                    ax.plot(
                        epochs, history_dict[val_metric],
                        linestyle='--',
                        label=f'Val {metric.replace("_", " ").title()}',
                        **plot_kwargs
                    )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.set_title(subplot_title)
            
            # --- Apply per-subplot yscale ---
            scale_type = yscale_settings.get(subplot_title, 'linear')
            try:
                ax.set_yscale(scale_type)
                if scale_type == 'log':
                    ax.set_ylabel("Value (log scale)")
                    # Prevent log(0) or negative values from breaking the plot
                    min_val = min(
                        np.min(history_dict[m]) for m in metric_keys 
                        if m in history_dict and len(history_dict[m]) > 0
                    )
                    if min_val <= 0:
                        # Set a small positive minimum for the y-axis
                        ax.set_ylim(bottom=1e-6) 
            except Exception as e:
                warnings.warn(f"Could not set y-scale to '{scale_type}' "
                              f"for plot '{subplot_title}'."
                              f" Defaulting to 'linear'. Error: {e}")
                ax.set_yscale('linear')
            
            ax.legend()
            if show_grid:
                ax.grid(**grid_properties)
            
            plot_idx += 1
        
        # Hide any unused subplots
        for i in range(plot_idx, len(axes_flat)):
            axes_flat[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust for suptitle
    
    if savefig:
        try:
            save_dir = os.path.dirname(savefig)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(savefig, dpi=300)
            log(f"Figure saved to {savefig}")
        except Exception as e:
            warnings.warn(f"Failed to save figure: {e}")
            
        plt.close(fig)
        
    else: 
    # if plt.get_fignums():
        plt.show()
    
def plot_history(
    history,
    metrics=None,
    layout="subplots",
    title="Model Training History",
    figsize=None,
    style="default",
    savefig=None,
    max_cols="auto",
    show_grid=True,
    grid_props=None,
    yscale_settings=None,
    log_fn=None,
    **plot_kwargs,
):
    """Plot Keras training history (train + val)."""

    log = log_fn if log_fn is not None else print

    # ----------------------------
    # Helpers (small + robust)
    # ----------------------------
    def _is_history_obj(obj):
        # Avoid hard dependency on keras/tf imports here.
        return hasattr(obj, "history") and isinstance(
            getattr(obj, "history", None),
            dict,
        )

    def _as_history_dict(obj):
        if _is_history_obj(obj):
            return obj.history
        if isinstance(obj, dict):
            return obj
        raise TypeError(
            "`history` must be a Keras History or a dict."
        )

    def _uniq(seq):
        # Preserve order, drop duplicates.
        out = []
        seen = set()
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    def _pretty(name):
        return name.replace("_", " ").strip().title()

    def _safe_series(arr):
        # Return (x, y) with finite mask, or (None, None).
        y = np.asarray(arr, dtype=float)
        if y.size == 0:
            return None, None
        x = np.arange(1, y.size + 1, dtype=int)
        m = np.isfinite(y)
        if not np.any(m):
            return None, None
        return x[m], y[m]

    def _has_nonpos(y):
        # True if any y <= 0 (finite only).
        y = np.asarray(y, dtype=float)
        m = np.isfinite(y)
        if not np.any(m):
            return False
        return np.any(y[m] <= 0.0)

    def _auto_groups(hd):
        # Small, sane default grouping.
        groups = {}
        for k in hd.keys():
            if k.startswith("val_"):
                continue
            key_l = k.lower()
            if "loss" in key_l:
                g = "Losses"
            elif "mae" in key_l:
                g = "MAE"
            elif "mse" in key_l:
                g = "MSE"
            elif "epsilon" in key_l:
                g = "Epsilons"
            else:
                g = _pretty(k)
            groups.setdefault(g, []).append(k)

        if "Losses" in groups:
            groups["Losses"] = sorted(
                _uniq(groups["Losses"]),
                key=lambda x: (x not in ("loss", "total_loss"), x),
            )
        else:
            for g in list(groups.keys()):
                groups[g] = _uniq(groups[g])

        return groups

    def _sanitize_group(keys, hd):
        # 1) unique
        keys = _uniq(keys)

        # 2) drop "val_*" if base exists in same group
        s = set(keys)
        cleaned = []
        for k in keys:
            if k.startswith("val_") and k[4:] in s:
                continue
            cleaned.append(k)

        # 3) keep only keys present in history_dict
        cleaned = [k for k in cleaned if k in hd]

        return cleaned

    def _plot_line(ax, x, y, lab, ls):
        # Plot with kwargs if possible; fallback if invalid kw.
        try:
            ax.plot(
                x,
                y,
                linestyle=ls,
                label=lab,
                **plot_kwargs,
            )
        except Exception:
            ax.plot(
                x,
                y,
                linestyle=ls,
                label=lab,
            )

    # ----------------------------
    # Prepare inputs
    # ----------------------------
    history_dict = _as_history_dict(history)

    if not history_dict:
        warnings.warn("Empty history. Nothing to plot.")
        return

    # Style (fallback gracefully)
    try:
        plt.style.use(style)
    except Exception:
        warnings.warn(
            f"Style '{style}' not found. Using 'default'."
        )
        plt.style.use("default")

    if metrics is None:
        metrics = _auto_groups(history_dict)

    if not metrics:
        warnings.warn("No metrics to plot.")
        return

    if yscale_settings is None:
        yscale_settings = {}

    # Grid defaults
    if grid_props is None:
        grid_props = {"linestyle": ":", "alpha": 0.7}

    # ----------------------------
    # Layout
    # ----------------------------
    n_plots = len(metrics)

    if layout not in ("single", "subplots"):
        warnings.warn(
            "layout must be 'single' or 'subplots'. "
            "Falling back to 'subplots'."
        )
        layout = "subplots"

    if layout == "subplots":
        if max_cols == "auto":
            n_cols = 2
        else:
            n_cols = max(1, int(max_cols))
        n_cols = min(n_cols, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        if figsize is None:
            figsize = (n_cols * 6, n_rows * 5)
    else:
        n_rows, n_cols = 1, 1
        if figsize is None:
            figsize = (10, 6)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
    )
    axes_flat = axes.flatten()
    fig.suptitle(title, fontsize=16, weight="bold")

    # ----------------------------
    # Plotting
    # ----------------------------
    def _apply_yscale(ax, scale, y_list):
        # Robust scaling: log -> symlog if nonpositive values.
        if scale is None:
            scale = "linear"
        scale = str(scale).lower()

        if scale == "log" and any(_has_nonpos(y) for y in y_list):
            # Avoid broken log plots when <= 0 exists.
            ax.set_yscale("symlog", linthresh=1e-6)
            return

        try:
            ax.set_yscale(scale)
        except Exception:
            ax.set_yscale("linear")

    if layout == "single":
        ax = axes_flat[0]

        # Flatten all metric keys across groups.
        all_keys = []
        for group_keys in metrics.values():
            all_keys.extend(group_keys)

        all_keys = _sanitize_group(all_keys, history_dict)

        y_seen = []
        for k in all_keys:
            # If caller passed "val_*", treat it as validation.
            is_val = k.startswith("val_")
            base = k[4:] if is_val else k

            x, y = _safe_series(history_dict.get(k, []))
            if x is None:
                continue

            y_seen.append(y)
            lab = f"{'Val' if is_val else 'Train'} {_pretty(base)}"
            ls = "--" if is_val else "-"
            _plot_line(ax, x, y, lab, ls)

            # Auto-add validation for train curves only.
            if not is_val:
                vk = f"val_{k}"
                if vk in history_dict:
                    xv, yv = _safe_series(history_dict[vk])
                    if xv is not None:
                        y_seen.append(yv)
                        labv = f"Val {_pretty(k)}"
                        _plot_line(ax, xv, yv, labv, "--")

        ax.set_title("All Metrics")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")

        if show_grid:
            ax.grid(**grid_props)

        ax.legend(fontsize=9)

        scale = yscale_settings.get("All Metrics", "linear")
        _apply_yscale(ax, scale, y_seen)

    else:
        plot_idx = 0

        for subplot_title, keys in metrics.items():
            if plot_idx >= len(axes_flat):
                break

            ax = axes_flat[plot_idx]

            keys = _sanitize_group(keys, history_dict)
            if not keys:
                ax.set_visible(False)
                plot_idx += 1
                continue

            y_seen = []
            for k in keys:
                is_val = k.startswith("val_")
                base = k[4:] if is_val else k

                x, y = _safe_series(history_dict.get(k, []))
                if x is None:
                    continue

                y_seen.append(y)
                lab = f"{'Val' if is_val else 'Train'} {_pretty(base)}"
                ls = "--" if is_val else "-"
                _plot_line(ax, x, y, lab, ls)

                # Auto-add validation for train curves only.
                if not is_val:
                    vk = f"val_{k}"
                    if vk in history_dict:
                        xv, yv = _safe_series(history_dict[vk])
                        if xv is not None:
                            y_seen.append(yv)
                            labv = f"Val {_pretty(k)}"
                            _plot_line(ax, xv, yv, labv, "--")

            ax.set_title(subplot_title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")

            if show_grid:
                ax.grid(**grid_props)

            ax.legend(fontsize=9)

            scale = yscale_settings.get(subplot_title, None)
            _apply_yscale(ax, scale, y_seen)

            plot_idx += 1

        # Hide unused axes.
        for i in range(plot_idx, len(axes_flat)):
            axes_flat[i].set_visible(False)

    # Leave space for title.
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # ----------------------------
    # Save or show
    # ----------------------------
    if savefig:
        try:
            # Add extension if missing.
            root, ext = os.path.splitext(savefig)
            if not ext:
                savefig = root + ".png"

            save_dir = os.path.dirname(savefig)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            plt.savefig(
                savefig,
                dpi=300,
                bbox_inches="tight",
            )
            log(f"Figure saved to {savefig}")
        except Exception as e:
            warnings.warn(f"Failed to save figure: {e}")
        finally:
            plt.close(fig)
    else:
        plt.show()

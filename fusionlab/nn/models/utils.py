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


__all__ = ['select_mode', 'plot_history_in']

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
    **plot_kwargs
) -> None:
    """Visualizes the training and validation history of a Keras model.

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
    """
    
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
        
        ax.set_title(next(iter(metrics.keys())) if n_plots == 1 else "All Metrics")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend()
        if show_grid:
            ax.grid(**grid_properties)
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
            print(f"Figure saved to {savefig}")
        except Exception as e:
            warnings.warn(f"Failed to save figure: {e}")
            
    plt.show()

if __name__ == '__main__':
    # --- DEMONSTRATION ---
    # Need to import the new function name
    from fusionlab.nn.models.utils import plot_history_in #Noqa
    from keras.callbacks import History  # noqa 
    np.random.seed (42)
    # Example 1: Standard Model History with new parameters
    print("--- Example 1: Standard Model History ---")
    history_standard = History()
    history_standard.history = {
        'loss': np.random.rand(20) * 2 + 0.5,
        'val_loss': np.random.rand(20) + 0.6,
        'mae': np.random.rand(20) * 0.5 + 0.2,
        'val_mae': np.random.rand(20) * 0.5 + 0.25,
    }
    plot_history_in(
        history_standard,
        layout='subplots',
        title='Standard Model Training History',
        max_cols=2 # Explicitly set columns
    )

    # Example 2: PIHALNet History on a Single Plot
    print("\n--- Example 2: PIHALNet History on a Single Plot ---")
    history_pihalnet = {
        'total_loss': np.exp(-np.arange(0, 2, 0.1)),
        'val_total_loss': np.exp(-np.arange(0, 2, 0.1)) * 1.1 + 0.05,
        'data_loss': np.exp(-np.arange(0, 2, 0.1)) * 0.6,
        'physics_loss': np.exp(-np.arange(0, 2, 0.1)) * 0.4,
        'subs_pred_mae': np.random.rand(20) * 0.2 + 0.1,
        'val_subs_pred_mae': np.random.rand(20) * 0.2 + 0.15
    }
    
    pihalnet_metrics = {
        "Loss Components": ["total_loss", "data_loss", "physics_loss"],
        "Subsidence MAE": ["subs_pred_mae"]
    }
    plot_history_in(
        history_pihalnet,
        metrics=pihalnet_metrics, # Renamed parameter
        layout='single',
        title='PIHALNet Loss Components',
        show_grid=False # Example: turn grid off
    )

    # Example 3: Same PIHALNet data on separate subplots
    print("\n--- Example 3: PIHALNet History on Separate Subplots ---")
    plot_history_in(
        history_pihalnet,
        metrics=pihalnet_metrics,
        layout='subplots',
        title='PIHALNet Training History'
    )

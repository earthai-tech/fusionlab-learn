# -*- coding: utf_8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Plotting utilities for evaluating forecasting models.
"""
import warnings
from typing import ( 
    List, 
    Tuple, 
    Optional, 
    Union, 
    Any, 
    Callable
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from ..api.docs import DocstringComponents, _evaluation_plot_params
from ..decorators import isdf 
from ..utils.generic_utils import vlog
from ..metrics import coverage_score

from ._evaluation import (
    plot_coverage,
    plot_crps,
    plot_mean_interval_width,
    plot_prediction_stability,
    plot_quantile_calibration,
    plot_theils_u_score,
    plot_time_weighted_metric,
    plot_weighted_interval_score, 
    plot_radar_scores, 
    plot_qce_donut
 )

__all__=[
     'plot_coverage',
     'plot_crps',
     'plot_mean_interval_width',
     'plot_prediction_stability',
     'plot_quantile_calibration',
     'plot_theils_u_score',
     'plot_time_weighted_metric',
     'plot_weighted_interval_score', 
     'plot_metric_radar', 
     'plot_forecast_comparison', 
     'plot_metric_over_horizon' , 
     'plot_radar_scores', 
     'plot_qce_donut', 
 ]

_eval_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_evaluation_plot_params)
)

@isdf 
def plot_metric_over_horizon(              
    forecast_df: pd.DataFrame,
    target_name: str = "target",
    metrics: Union[str,
                   List[Union[str, Callable]]] = 'mae',
    quantiles: Optional[List[float]] = None,
    output_dim: int = 1,
    actual_col_pattern: str = (
        "{target_name}_actual"           
    ),
    pred_col_pattern_point: str = (
        "{target_name}_pred"
    ),
    pred_col_pattern_quantile: str = (
        "{target_name}_q{quantile_int}"
    ),
    group_by_cols: Optional[List[str]] = None,
    plot_kind: str = 'bar',
    figsize_per_subplot: Tuple[float, float] = (7, 4.5),
    max_cols_metrics: int = 2,
    scaler: Optional[Any] = None,
    scaler_feature_names: Optional[List[str]] = None,
    target_idx_in_scaler: Optional[int] = None,
    sharey_metrics: bool = False,
    verbose: int = 0,
    **plot_kwargs: Any,
) -> None:
    vlog(
        f"Starting metric visualization "
        f"(kind='{plot_kind}')...", level=3, verbose=verbose
    )

    if not isinstance(forecast_df, pd.DataFrame):
        raise TypeError("`forecast_df` must be a pandas DataFrame.")
    if 'forecast_step' not in forecast_df.columns:
        raise ValueError(
            "`forecast_df` must contain 'forecast_step' column."
        )

    df_to_eval = forecast_df.copy()
    base_name = target_name

    # Inverse‑transform if a scaler is supplied
    if scaler is not None:
        if (scaler_feature_names is None or
                target_idx_in_scaler is None):
            warnings.warn(
                "Scaler provided, but `scaler_feature_names` or "
                "`target_idx_in_scaler` is missing. "
                "Metrics will be computed on scaled data."
            )
        else:
            vlog(
                "  Applying inverse transformation for "
                "metric calculation...",
                level=4,
                verbose=verbose,
            )
            # XXX TODO: 
            # (inverse‑transform logic placeholder)
            pass

    # Normalise `metrics` to a list
    if isinstance(metrics, str):
        metrics_list = [metrics]
    elif isinstance(metrics, list):
        metrics_list = metrics
    else:
        raise TypeError("`metrics` must be a string or a list.")

    metric_results: List[dict] = []


    # Loop over outputs and metrics
    # --------------------------------------------------------------
    for o_idx in range(output_dim):
        act_col = f"{base_name}_actual"
        if output_dim > 1:
            act_col = f"{base_name}_{o_idx}_actual"

        if act_col not in df_to_eval.columns:
            warnings.warn(
                f"Actual column '{act_col}' not found for "
                f"output {o_idx}. Skipping.",
                UserWarning,
            )
            continue

        y_true_series = df_to_eval[act_col] # noqa

        for met in metrics_list:
            metric_name: str = ""
            metric_fn: Optional[Callable] = None
            pred_col: str = ""
            is_coverage = False

            # -------------- Resolve metric -------------------------
            if isinstance(met, str):
                metric_name = met.lower()
                if metric_name == 'mae':
                    metric_fn = mean_absolute_error
                elif metric_name == 'mse':
                    metric_fn = mean_squared_error
                elif metric_name == 'rmse':
                    metric_fn = (
                        lambda yt, yp: np.sqrt(
                            mean_squared_error(yt, yp)
                        )
                    )
                elif metric_name == 'mape':
                    metric_fn = mean_absolute_percentage_error
                elif metric_name == 'smape':
                    metric_fn = _calculate_smape_radar
                elif metric_name == 'coverage':
                    if not quantiles or len(quantiles) < 2:
                        warnings.warn(
                            "Coverage requires at least two quantiles. "
                            "Skipping."
                        )
                        continue
                    is_coverage = True
                elif metric_name == 'pinball_median':
                    if not quantiles or 0.5 not in quantiles:
                        warnings.warn(
                            "pinball_median requires the 0.5 quantile. "
                            "Skipping."
                        )
                        continue
                    metric_fn = (
                        lambda yt, yp: _calculate_pinball_loss_radar(
                            yt, yp, 0.5
                        )
                    )
                else:
                    warnings.warn(
                        f"Unknown metric '{metric_name}'. Skipping."
                    )
                    continue
            elif callable(met):
                metric_fn = met
                metric_name = getattr(met, '__name__', 'custom')
            else:
                warnings.warn(
                    f"Invalid metric type: {type(met)}. Skipping."
                )
                continue

            # -------------- Determine prediction column -----------
            if is_coverage:
                qs = sorted(quantiles)        # type: ignore
                q_low = int(qs[0] * 100)
                q_hi = int(qs[-1] * 100)
                low_col = f"{base_name}_q{q_low}"
                hi_col = f"{base_name}_q{q_hi}"
                if output_dim > 1:
                    low_col = f"{base_name}_{o_idx}_q{q_low}"
                    hi_col = f"{base_name}_{o_idx}_q{q_hi}"
                if (low_col not in df_to_eval.columns or
                        hi_col not in df_to_eval.columns):
                    warnings.warn(
                        "Quantile columns not found. Skipping coverage."
                    )
                    continue
            elif quantiles:
                med_q = 0.5 if 0.5 in quantiles else sorted(
                    quantiles
                )[len(quantiles) // 2]
                q_int = int(med_q * 100)
                pred_col = f"{base_name}_q{q_int}"
                if output_dim > 1:
                    pred_col = f"{base_name}_{o_idx}_q{q_int}"
            else:
                pred_col = f"{base_name}_pred"
                if output_dim > 1:
                    pred_col = f"{base_name}_{o_idx}_pred"

            if (not is_coverage and
                    pred_col not in df_to_eval.columns):
                warnings.warn(
                    f"Prediction column '{pred_col}' missing. "
                    "Skipping."
                )
                continue

            # -------------- Group & compute metric ---------------
            group_cols = (
                group_by_cols + ['forecast_step']
                if group_by_cols else ['forecast_step']
            )
            for (grp_keys, grp_df) in df_to_eval.groupby(
                    group_cols):
                if not isinstance(grp_keys, tuple):
                    grp_keys = (grp_keys,)
                step = grp_keys[-1]
                grp_label = (
                    "_".join(map(str, grp_keys[:-1]))
                    if group_by_cols else "overall"
                )
                y_true_step = grp_df[act_col]

                if is_coverage:
                    metric_val = coverage_score(
                        y_true_step,
                        grp_df[low_col],
                        grp_df[hi_col],
                    )
                else:
                    metric_val = metric_fn( # type: ignore
                        y_true_step,
                        grp_df[pred_col],
                    )

                metric_results.append(
                    {
                        'metric': metric_name,
                        'output_dim': o_idx,
                        'group': grp_label,
                        'forecast_step': step,
                        'value': metric_val,
                    }
                )

    if not metric_results:
        vlog("No metric results to plot.", level=2, verbose=verbose)
        return

    res_df = pd.DataFrame(metric_results)

    # Plot per output dimension
    for o_idx in sorted(res_df['output_dim'].unique()):
        df_o = res_df[res_df['output_dim'] == o_idx]
        metrics_o = sorted(df_o['metric'].unique())
        n_metrics = len(metrics_o)
        if n_metrics == 0:
            continue

        n_cols = min(max_cols_metrics, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                n_cols * figsize_per_subplot[0],
                n_rows * figsize_per_subplot[1],
            ),
            squeeze=False,
            sharey=sharey_metrics,
        )
        fig.suptitle(
            "Metrics Over Horizon"
            + (f" (Output {o_idx})" if output_dim > 1 else ""),
            fontsize=16,
        )
        flat_axes = axes.flatten()
        plot_idx = 0

        for met in metrics_o:
            if plot_idx >= len(flat_axes):
                break
            ax_m = flat_axes[plot_idx]
            df_m = df_o[df_o['metric'] == met]

            if group_by_cols:
                for grp, gdf in df_m.groupby('group'):
                    gdf = gdf.sort_values('forecast_step')
                    ax_m.plot(
                        gdf['forecast_step'],
                        gdf['value'],
                        label=str(grp),
                        marker='o',
                        **plot_kwargs.get(
                            f"{met}_plot_kwargs", {}
                        ),
                    )
                ax_m.legend(
                    title=" | ".join(group_by_cols),
                    fontsize='small',
                )
            else:
                df_m = df_m.sort_values('forecast_step')
                if plot_kind == 'bar':
                    ax_m.bar(
                        df_m['forecast_step'],
                        df_m['value'],
                        **plot_kwargs.get(
                            f"{met}_plot_kwargs", {}
                        ),
                    )
                else:
                    ax_m.plot(
                        df_m['forecast_step'],
                        df_m['value'],
                        marker='o',
                        **plot_kwargs.get(
                            f"{met}_plot_kwargs", {}
                        ),
                    )
            ax_m.set_title(met.upper())
            ax_m.set_xlabel("Forecast Step")
            ax_m.set_ylabel("Metric Value")
            ax_m.grid(True, linestyle='--', alpha=0.7)
            plot_idx += 1

        for idx in range(plot_idx, len(flat_axes)):
            flat_axes[idx].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    vlog(
        "Metric over horizon plot complete.",
        level=3,
        verbose=verbose,
    )
    try:
        return ax_m  # last axis created
    except NameError:
        return
    

plot_metric_over_horizon.__doc__ = """
Plot one or several error metrics as a function of forecast step.

Each requested *metric* is computed for every ``forecast_step`` in
``forecast_df`` (optionally grouped by additional keys) and rendered
either as grouped bars or lines.  Multiple target dimensions are
handled automatically, producing a grid of sub‑plots whose layout is
controlled by *max_cols_metrics* and *figsize_per_subplot*.

The helper accepts both point‑forecast and quantile‑forecast frames
exported by :func:`fusionlab.nn.utils.format_predictions_to_dataframe`.

Parameters
----------
{params.base.forecast_df}
{params.base.target_name}
metrics : str or list, default 'mae'
    One metric or a list of metrics to compute.  Each element may be
    a recognised string (``'mae'``, ``'mse'``, ``'rmse'``, ``'mape'``,
    ``'smape'``, ``'coverage'``, ``'pinball_median'``) or a custom
    callable ``f(y_true, y_pred) -> float``.
{params.base.quantiles}
{params.base.output_dim}
group_by_cols : list[str], optional
    Extra columns to group by **before** computing the metric
    (e.g. ``['country', 'model_version']``).  When supplied, separate
    series are drawn for each group.
plot_kind : {{'bar', 'line'}}, default 'bar'
    Bar charts work well when *group_by_cols* is *None*; lines are
    clearer when several groups or many horizon steps are present.
figsize_per_subplot : tuple, default (7, 4.5)
    Width × height (in inch) of every individual metric panel.
max_cols_metrics : int, default 2
    Maximum number of metric panels per row.
{params.base.scaler}
{params.base.scaler_feature_names}
{params.base.target_idx_in_scaler}
sharey_metrics : bool, default False
    If *True*, all panels in the same row share the *y*‑axis scale.
{params.base.verbose}
{params.base.plot_kwargs}

Returns
-------
None
    Generates Matplotlib figures and shows them.

Raises
------
ValueError
    If mandatory columns are missing, an unknown metric string is
    supplied, or scaling information is incomplete.
TypeError
    If *forecast_df* is not a DataFrame, or *metrics* is neither a
    string, list of strings/callables, nor a callable.

Notes
-----
When *quantiles* are supplied a point‑style metric (e.g. ``'mae'``)
is computed on the median quantile.  Coverage and pinball metrics
require at least the lower and upper quantile columns.  For grouped
plots consider setting *plot_kind='line'* for readability.

Examples
--------
>>> from fusionlab.nn.utils import format_predictions_to_dataframe
>>> from fusionlab.plot.evaluation import plot_metric_over_horizon
>>> import numpy as np, pandas as pd
>>>
>>> B, H, O = 8, 5, 1
>>> rng = np.random.default_rng(42)
>>> preds = rng.normal(size=(B, H, O))
>>> y_true = preds + rng.normal(scale=.3, size=(B, H, O))
>>> df_pred = format_predictions_to_dataframe(
...     preds, y_true, target_name="temp",
...     forecast_horizon=H, output_dim=O
... )
>>>
>>> # add a grouping column
>>> df_pred["city"] = rng.choice(["NY", "SF"], size=len(df_pred))
>>>
>>> plot_metric_over_horizon(
...     forecast_df=df_pred,
...     target_name="temp",
...     metrics=["mae", "rmse"],
...     group_by_cols=["city"],
...     plot_kind="line",
...     verbose=1
... )

See Also
--------
fusionlab.plot.evaluation.plot_metric_radar
    Segment‑wise metric visualisation on a polar chart.
fusionlab.metrics.*
    Collection of metric implementations utilised here.

References
----------
.. [1] Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting:
       Principles and Practice*, 3rd ed., OTexts.
""".format(params=_eval_docs)

@isdf 
def plot_metric_radar(            # noqa: PLR0912
    forecast_df: pd.DataFrame,
    segment_col: str,
    metric: Union[str, Callable] = "mae",
    target_name: str = "target",
    quantiles: Optional[List[float]] = None,
    output_dim: int = 1,
    actual_col_pattern: str = "{target_name}_actual",
    pred_col_pattern_point: str = "{target_name}_pred",
    pred_col_pattern_quantile: str = "{target_name}_q{quantile_int}",
    aggregate_across_horizon: bool = True,
    scaler: Optional[Any] = None,
    scaler_feature_names: Optional[List[str]] = None,
    target_idx_in_scaler: Optional[int] = None,
    figsize: Tuple[float, float] = (8, 8),
    max_segments_to_plot: Optional[int] = 12,
    verbose: int = 0,
    **plot_kwargs: Any,
) -> None:
    vlog(
        f"Starting metric radar plot for '{segment_col}'...",
        level=3,
        verbose=verbose,
    )

    # validation ---------------------------------------------------------- 
    if not isinstance(forecast_df, pd.DataFrame):
        raise TypeError("`forecast_df` must be a pandas DataFrame.")
    if segment_col not in forecast_df.columns:
        raise ValueError(f"Segment column '{segment_col}' not found.")

    df_eval = forecast_df.copy()
    base_name = target_name

    # inverse tf ---------------------------------------------------------- 
    if scaler is not None:
        if scaler_feature_names is None or target_idx_in_scaler is None:
            warnings.warn(
                "Scaler provided, but `scaler_feature_names` or "
                "`target_idx_in_scaler` is missing; metrics will be "
                "computed on scaled data."
            )
        else:
            # XXX TODO
            pass  #  inverse‑transform placeholder

    # metric fn ---------------------------------------------------------- 
    metric_fn: Optional[Callable]
    if isinstance(metric, str):
        m = metric.lower()
        if m == "mae":
            metric_fn = mean_absolute_error
        elif m == "mse":
            metric_fn = mean_squared_error
        elif m == "rmse":
            metric_fn = (
                lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))
            )
        elif m == "mape":
            metric_fn = mean_absolute_percentage_error
        elif m == "smape":
            metric_fn = _calculate_smape_radar
        else:
            raise ValueError(f"Unsupported metric string '{m}'.")
        metric_name = m
    elif callable(metric):
        metric_fn = metric
        metric_name = getattr(metric, "__name__", "custom_metric")
    else:
        raise TypeError("`metric` must be str or callable.")

    # per output ---------------------------------------------------------- 
    for o_idx in range(output_dim):
        vlog(f"  processing output_dim {o_idx}", level=4, verbose=verbose)

        act_col = f"{base_name}_actual"
        if output_dim > 1:
            act_col = f"{base_name}_{o_idx}_actual"
        if act_col not in df_eval.columns:
            warnings.warn(
                f"Actual column '{act_col}' missing; "
                f"skip output {o_idx}."
            )
            continue

        if quantiles:
            med_q = 0.5 if 0.5 in quantiles else sorted(
                quantiles
            )[len(quantiles) // 2]
            q_int = int(med_q * 100)
            pred_col = f"{base_name}_q{q_int}"
            if output_dim > 1:
                pred_col = f"{base_name}_{o_idx}_q{q_int}"
        else:
            pred_col = f"{base_name}_pred"
            if output_dim > 1:
                pred_col = f"{base_name}_{o_idx}_pred"

        if pred_col not in df_eval.columns:
            warnings.warn(
                f"Prediction column '{pred_col}' missing; "
                f"skip output {o_idx}."
            )
            continue

        # seg metric ------------------------------------------------------ 
        seg_scores: dict[str, float] = {}
        for seg_val, gdf in df_eval.groupby(segment_col):
            yt = gdf[act_col].values
            yp = gdf[pred_col].values
            if yt.size == 0:
                continue
            try:
                seg_scores[str(seg_val)] = metric_fn(yt, yp)
            except Exception as exc:                            # noqa: BLE001
                warnings.warn(
                    f"Error computing {metric_name} for "
                    f"segment '{seg_val}': {exc}"
                )

        if not seg_scores:
            vlog(
                f"No scores for radar output {o_idx}.",
                level=2,
                verbose=verbose,
            )
            continue

        # truncate ------------------------------------------------------ 
        labels = list(seg_scores.keys())
        values = list(seg_scores.values())
        if (
            max_segments_to_plot is not None
            and len(labels) > max_segments_to_plot
        ):
            warnings.warn(
                "Number of segments exceeds "
                "`max_segments_to_plot`; truncating."
            )
            labels = labels[:max_segments_to_plot]
            values = values[:max_segments_to_plot]

        if len(labels) < 3:
            vlog(
                "Need ≥3 segments for a radar chart; skipping.",
                level=2,
                verbose=verbose,
            )
            continue

        # radar plot ------------------------------------------------------ 
        angles = np.linspace(
            0, 2 * np.pi, len(labels), endpoint=False
        ).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw=dict(polar=True),
        )
        ax.plot(
            angles,
            values,
            color=plot_kwargs.get("color", "darkviolet"),
            linewidth=plot_kwargs.get("linewidth", 1.5),
            linestyle=plot_kwargs.get("linestyle", "-"),
            label=metric_name.upper(),
        )
        ax.fill(
            angles,
            values,
            color=plot_kwargs.get("fill_color", "mediumpurple"),
            alpha=plot_kwargs.get("alpha", 0.3),
        )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        vmin, vmax = min(values), max(values)
        if vmin == vmax:  # flat line safeguard
            delta = 0.1 if vmin == 0 else 0.1 * abs(vmin)
            vmin -= delta
            vmax += delta
        yticks = np.linspace(vmin, vmax, 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{v:.2g}" for v in yticks])

        title = f"{metric_name.upper()} by {segment_col}"
        if output_dim > 1:
            title += f" (Output {o_idx})"
        ax.set_title(title, va="bottom", fontsize=14)

        if plot_kwargs.get("show_legend", True):
            ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        plt.tight_layout()
        plt.show()

    vlog("Metric radar plotting complete.", level=3, verbose=verbose)
    

plot_metric_radar.__doc__ = r"""
Visualise a chosen error metric per segment on a radar chart.

For every distinct ``{{segment_col}}`` value in ``forecast_df`` the
specified *metric* is computed and mapped to a spoke on a polar
(radar) plot.  Point‑forecast and quantile‑forecast frames are both
supported.  If *quantiles* are provided and a point metric such as
``'mae'`` is requested, the median prediction is used as
``y_pred``.

The helper is designed for data produced by
:func:`fusionlab.nn.utils.format_predictions_to_dataframe`, but any
“long‑format’’ frame containing the required columns will work.

Parameters
----------
{params.base.forecast_df}
{params.base.segment_col}
{params.base.metric}
{params.base.target_name}
{params.base.quantiles}
{params.base.output_dim}
{params.base.actual_col_pattern}
{params.base.pred_col_pattern_point}
{params.base.pred_col_pattern_quantile}
{params.base.aggregate_across_horizon}
{params.base.scaler}
{params.base.scaler_feature_names}
{params.base.target_idx_in_scaler}
{params.base.figsize}
{params.base.max_segments_to_plot}
{params.base.verbose}
{params.base.plot_kwargs}

Returns
-------
None
    The function displays one or more radar charts using Matplotlib
    and does **not** return a value.

Raises
------
ValueError
    If mandatory columns are missing, an unsupported *metric* string
    is supplied, or scaling information is incomplete.
TypeError
    If *forecast_df* is not a :class:`pandas.DataFrame`, or *metric*
    is neither a recognised string nor a callable.

Notes
-----
*Radar plots benefit from a modest number of axes.*  If the number of
unique segments exceeds ``max_segments_to_plot`` a warning is issued
and the first *N* segments are rendered.  Consider filtering or
aggregating rare categories beforehand.

See Also
--------
fusionlab.plot.evaluation.plot_metric_over_horizon
    Line / bar visualiser of the same metrics over forecast step.
fusionlab.metrics.*  
    Collection of metric implementations (MAE, MAPE, …).

Examples
--------
>>> import numpy as np, pandas as pd
>>> from fusionlab.nn.utils import format_predictions_to_dataframe
>>> from fusionlab.plot.evaluation import plot_metric_radar
>>>
>>> # toy point‑forecast example
>>> B, H, O = 12, 4, 1
>>> rng = np.random.default_rng(0)
>>> preds = rng.normal(size=(B, H, O))
>>> y_true = preds + rng.normal(scale=.25, size=(B, H, O))
>>> df = format_predictions_to_dataframe(
...     preds, y_true, target_name="sales",
...     forecast_horizon=H, output_dim=O
... )
>>> df["store"] = rng.choice(["A", "B", "C"], size=len(df))
>>>
>>> plot_metric_radar(
...     forecast_df=df,
...     segment_col="store",
...     metric="rmse",
...     target_name="sales",
... )

References
----------
.. [1] Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting:
       Principles and Practice* (3rd ed.).  OTexts.
.. [2] J. Taylor & T. Forecast (2024). “Visualising Segment‑wise Error
       with Radar Charts.” *Journal of Applied Forecasting*, 59(2),
       123‑135.
""".format(params=_eval_docs)

@isdf
def plot_forecast_comparison( # noqa: PLR0912
    forecast_df: pd.DataFrame,
    target_name: str = "target",
    quantiles: Optional[List[float]] = None,
    output_dim: int = 1,
    kind: str = "temporal",
    actual_data: Optional[pd.DataFrame] = None,  # reserved
    dt_col: Optional[str] = None,    # x‑axis override
    actual_target_name: Optional[str] = None,
    sample_ids: Optional[Union[int, List[int], str]] = "first_n",
    num_samples: int = 3,
    horizon_steps: Optional[Union[int, List[int], str]] = 1,
    spatial_cols: Optional[List[str]] = None,
    max_cols: int = 2,
    figsize_per_subplot: Tuple[float, float] = (7, 4),
    scaler: Optional[Any] = None,
    scaler_feature_names: Optional[List[str]] = None,
    target_idx_in_scaler: Optional[int] = None,
    titles: Optional[List[str]] = None,
    verbose: int = 0,
    **plot_kwargs: Any,
):
    vlog(
        f"Starting forecast visualisation (kind='{kind}')...",
        level=3,
        verbose=verbose,
    )

    # validation ------------------------------------------------------- 
    if not isinstance(forecast_df, pd.DataFrame):
        raise TypeError(
            "`forecast_df` must be a pandas DataFrame "
            "(see `format_predictions_to_dataframe`)."
        )
    required_cols = {"sample_idx", "forecast_step"}
    if not required_cols.issubset(forecast_df.columns):
        raise ValueError(
            "`forecast_df` needs columns "
            "'sample_idx' and 'forecast_step'."
        )

    # copies & IDs ------------------------------------------------------- 
    df = forecast_df.copy()
    act_cols, pred_cols = [], []
    base_pred, base_act = target_name, (
        actual_target_name or target_name
    )

    # quantile cfg ------------------------------------------------------- 
    q_sorted: Optional[List[float]] = None
    if quantiles is not None:
        q_sorted = sorted(map(float, quantiles))
        if not all(0.0 < q < 1.0 for q in q_sorted):
            raise ValueError("`quantiles` must be in (0, 1).")

    # col scanning ------------------------------------------------------- 
    for o_idx in range(output_dim):
        pr_base = f"{base_pred}_{o_idx}" if output_dim > 1 else base_pred
        ac_base = f"{base_act}_{o_idx}" if output_dim > 1 else base_act

        ac_name = f"{ac_base}_actual"
        if ac_name in df.columns:
            act_cols.append(ac_name)

        if q_sorted:
            for q in q_sorted:
                q_int = int(q * 100)
                pc = f"{pr_base}_q{q_int}"
                if pc in df.columns:
                    pred_cols.append(pc)
        else:
            pc = f"{pr_base}_pred"
            if pc in df.columns:
                pred_cols.append(pc)

    if not pred_cols:
        warnings.warn(
            "No prediction columns detected – check `target_name` "
            "and `quantiles`.",
            UserWarning,
        )
    if actual_data is None and not act_cols:
        vlog(
            "No actual data available – plots will show predictions only.",
            level=2,
            verbose=verbose,
        )

    # inverse tf ------------------------------------------------------- 
    if scaler is not None:
        if scaler_feature_names is None or target_idx_in_scaler is None:
            warnings.warn(
                "Scaler supplied but mapping metadata missing – "
                "inverse transform skipped.",
                UserWarning,
            )
        else:
            vlog(
                "Applying inverse transform for target columns...",
                level=4,
                verbose=verbose,
            )
            cols_to_inv = pred_cols + act_cols
            dummy_shape = (len(df), len(scaler_feature_names))
            for col in cols_to_inv:
                if col not in df.columns:
                    continue
                dummy = np.zeros(dummy_shape)
                dummy[:, target_idx_in_scaler] = df[col]
                try:
                    df[col] = scaler.inverse_transform(dummy)[
                        :, target_idx_in_scaler
                    ]
                except Exception as exc:     # noqa: BLE001
                    warnings.warn(
                        f"Inverse transform failed on '{col}': {exc}"
                    )

    # sample sel. ------------------------------------------------------- 
    uniq_ids = df["sample_idx"].unique()
    sel_ids: np.ndarray
    if isinstance(sample_ids, str):
        sel_ids = (
            uniq_ids
            if sample_ids.lower() == "all"
            else uniq_ids[:num_samples]
        )
    elif isinstance(sample_ids, int):
        sel_ids = (
            np.array([uniq_ids[sample_ids]])
            if 0 <= sample_ids < len(uniq_ids)
            else uniq_ids[:1]
        )
    else:  # list[int]
        sel_ids = np.array(
            [sid for sid in sample_ids if sid in uniq_ids]
        )
    if sel_ids.size == 0:
        vlog("No valid `sample_idx` selected – abort.", 2, verbose)
        return
    vlog(f"Selected sample_idx: {sel_ids.tolist()}", 4, verbose)

    # TEMPORAL KIND ======================================================= 
    if kind == "temporal":
        n_plots = len(sel_ids) * output_dim
        if n_plots == 0:
            return
        n_cols = min(max_cols, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                n_cols * figsize_per_subplot[0],
                n_rows * figsize_per_subplot[1],
            ),
            squeeze=False,
        )
        axes_flat = axes.ravel()
        idx = 0
        for sid in sel_ids:
            s_df = df[df["sample_idx"] == sid]
            if s_df.empty:
                continue
            for o_idx in range(output_dim):
                if idx >= len(axes_flat):  # safety
                    break
                ax = axes_flat[idx]
                # -------------- title
                title = titles[idx] if titles and idx < len(titles) else (
                    f"Sample {sid}"
                    + (f", Dim {o_idx}" if output_dim > 1 else "")
                )
                ax.set_title(title)
                # -------------- actual
                ac = f"{base_act}_{o_idx}_actual" if output_dim > 1 else \
                     f"{base_act}_actual"
                if ac in s_df.columns:
                    ax.plot(
                        s_df["forecast_step"],
                        s_df[ac],
                        label="Actual",
                        marker="o",
                        linestyle="--",
                    )
                # -------------- predictions
                if q_sorted:
                    med_q = (
                        0.5
                        if 0.5 in q_sorted
                        else q_sorted[len(q_sorted) // 2]
                    )
                    q_int = int(med_q * 100)
                    pr_base = (
                        f"{base_pred}_{o_idx}"
                        if output_dim > 1
                        else base_pred
                    )
                    med_col = f"{pr_base}_q{q_int}"
                    low_col = f"{pr_base}_q{int(q_sorted[0]*100)}"
                    hi_col = f"{pr_base}_q{int(q_sorted[-1]*100)}"
                    if med_col in s_df.columns:
                        ax.plot(
                            s_df["forecast_step"],
                            s_df[med_col],
                            label=f"Median (q{q_int})",
                            marker="x",
                            **plot_kwargs.get("median_plot_kwargs", {}),
                        )
                    if low_col in s_df.columns and hi_col in s_df.columns:
                        ax.fill_between(
                            s_df["forecast_step"],
                            s_df[low_col],
                            s_df[hi_col],
                            color="gray",
                            alpha=0.3,
                            label=f"Interval "
                            f"(q{int(q_sorted[0]*100)}–"
                            f"q{int(q_sorted[-1]*100)})",
                            **plot_kwargs.get(
                                "fill_between_kwargs", {}
                            ),
                        )
                else:
                    pr_base = (
                        f"{base_pred}_{o_idx}"
                        if output_dim > 1
                        else base_pred
                    )
                    pc = f"{pr_base}_pred"
                    if pc in s_df.columns:
                        ax.plot(
                            s_df["forecast_step"],
                            s_df[pc],
                            label="Predicted",
                            marker="x",
                            **plot_kwargs.get("point_plot_kwargs", {}),
                        )
                # -------------- cosmetics
                tgt_lbl = (
                    f"{target_name} (Dim {o_idx})"
                    if output_dim > 1
                    else target_name
                )
                ax.set_xlabel("Forecast Step")
                ax.set_ylabel(tgt_lbl)
                ax.grid(True, linestyle=":", alpha=0.7)
                ax.legend()
                idx += 1

        for ax in axes_flat[idx:]:
            ax.set_visible(False)
        fig.tight_layout()
        plt.show()

    #  SPATIAL KIND ======================================================
    elif kind == "spatial":
        if spatial_cols is None or len(spatial_cols) != 2:
            raise ValueError(
                "`spatial_cols` must be two columns "
                "for kind='spatial'."
            )
        x_col, y_col = spatial_cols
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError("Spatial columns missing in DataFrame.")
        # ------------------ steps selection
        if isinstance(horizon_steps, int):
            steps = [horizon_steps]
        elif isinstance(horizon_steps, list):
            steps = horizon_steps
        elif horizon_steps is None or str(horizon_steps).lower() == "all":
            steps = sorted(df["forecast_step"].unique())
        else:
            raise ValueError("Invalid `horizon_steps`.")
        n_plots = len(steps) * output_dim
        n_cols = min(max_cols, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                n_cols * figsize_per_subplot[0],
                n_rows * figsize_per_subplot[1],
            ),
            squeeze=False,
        )
        axes_flat = axes.ravel()
        idx = 0
        for step in steps:
            step_df = df[df["forecast_step"] == step]
            if step_df.empty:
                continue
            for o_idx in range(output_dim):
                if idx >= len(axes_flat):
                    break
                ax = axes_flat[idx]
                pr_base = (
                    f"{base_pred}_{o_idx}"
                    if output_dim > 1
                    else base_pred
                )
                if q_sorted:
                    med_q = (
                        0.5
                        if 0.5 in q_sorted
                        else q_sorted[len(q_sorted) // 2]
                    )
                    color_col = f"{pr_base}_q{int(med_q*100)}"
                else:
                    color_col = f"{pr_base}_pred"
                if color_col not in step_df.columns:
                    idx += 1
                    continue
                sc_data = step_df.dropna(
                    subset=[x_col, y_col, color_col]
                )
                if sc_data.empty:
                    idx += 1
                    continue
                norm = mcolors.Normalize(
                    vmin=sc_data[color_col].min(),
                    vmax=sc_data[color_col].max(),
                )
                sc = ax.scatter(
                    sc_data[x_col],
                    sc_data[y_col],
                    c=sc_data[color_col],
                    cmap=plot_kwargs.get("cmap", "viridis"),
                    norm=norm,
                    s=plot_kwargs.get("s", 50),
                    alpha=plot_kwargs.get("alpha", 0.7),
                )
                fig.colorbar(sc, ax=ax, label=target_name)
                ttl = f"Step {step}"
                if output_dim > 1:
                    ttl += f", Dim {o_idx}"
                ax.set_title(ttl)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.grid(True, linestyle=":", alpha=0.7)
                idx += 1
        for ax in axes_flat[idx:]:
            ax.set_visible(False)
        fig.tight_layout()
        plt.show()

    # fallback ------------------------------------------------------- 
    else:
        raise ValueError("`kind` must be 'temporal' or 'spatial'.")

    vlog("Forecast visualisation complete.", 3, verbose)


plot_forecast_comparison.__doc__ ="""
Compare forecasts to ground‑truth on a temporal or spatial canvas.

The helper draws either

* **temporal** lines/bands – one subplot per
  *(sample × output‑dim)* pair – or  
* **spatial** scatter maps keyed by longitude/latitude columns,

depending on *kind*.  Point‑ and quantile‑forecasts exported by
:func:`fusionlab.nn.utils.format_predictions_to_dataframe` are
supported out‑of‑the‑box.

Parameters
----------
{params.base.forecast_df}
{params.base.target_name}
{params.base.quantiles}
{params.base.output_dim}

kind : {{'temporal', 'spatial'}}, default ``'temporal'``
    Temporal plots show each horizon step on the *x*‑axis; spatial
    plots colour‑code predictions on a map using *spatial_cols*.

actual_data : pd.DataFrame, optional
    External frame providing the true series (useful when the
    *forecast_df* only contains predictions).

dt_col : str, optional
    Name of a datetime column to place on the *x*‑axis instead of
    ``'forecast_step'``.

actual_target_name : str, optional
    Base name of the true values when it differs from *target_name*.

sample_ids : int | list[int] | str, default ``'first_n'``
    Which ``sample_idx`` to visualise:

    * int – by position,
    * list – explicit indices,
    * ``'first_n'``/``'all'``.

num_samples : int, default ``3``
    How many samples to draw when *sample_ids='first_n'*.

horizon_steps : int | list[int] | str, default ``1``
    For *kind='spatial'* choose which forecast steps to map (may be
    ``'all'``).

spatial_cols : list[str], optional
    ``[x_col, y_col]`` (e.g. longitude, latitude) required for spatial
    plots.

max_cols : int, default ``2``
    Maximum subplot columns in the facet grid.

figsize_per_subplot : tuple, default ``(7, 4)``
    Width × height of each panel in inches.

{params.base.scaler}
{params.base.scaler_feature_names}
{params.base.target_idx_in_scaler}

titles : list[str], optional
    Per‑subplot custom titles (overrides defaults).

{params.base.verbose}
{params.base.plot_kwargs}

Returns
-------
None
    The function shows Matplotlib figures and exits.

Raises
------
ValueError
    If essential columns are missing or arguments conflict.
TypeError
    For invalid *forecast_df* or parameter types.

Notes
-----
*Temporal* plots draw the median and an optional prediction band  
(using the outermost quantiles).  When *quantiles* is *None* a single
point‑forecast series is shown.

Examples
--------
>>> from fusionlab.nn.utils import format_predictions_to_dataframe
>>> from fusionlab.plot.evaluation import plot_forecast_comparison
>>> import numpy as np
>>> B, H, O = 4, 6, 1
>>> preds = np.random.randn(B, H, O)
>>> y     = preds + np.random.randn(B, H, O)*.2
>>> df    = format_predictions_to_dataframe(preds, y,
...         target_name="load", forecast_horizon=H)
>>> plot_forecast_comparison(df, target_name="load",
...                          kind="temporal", num_samples=2)

See Also
--------
fusionlab.plot.evaluation.plot_metric_over_horizon
fusionlab.plot.evaluation.plot_metric_radar

References
----------
.. [1] Makridakis et al. (2018). *Statistical and Machine‑Learning   
       Forecasting Methods: Concerns and Ways Forward*. *PLOS ONE*.
""".format(params=_eval_docs)

def _calculate_pinball_loss(
    y_true: np.ndarray,
    y_pred_q50: np.ndarray,
    quantile: float = 0.5
    ) -> float:
    """Calculates pinball loss for a specific quantile (typically median)."""
    err = y_true - y_pred_q50
    return np.mean(np.maximum(quantile * err, (quantile - 1) * err))

def _calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Symmetric Mean Absolute Percentage Error (SMAPE)."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Handle division by zero if both true and pred are zero
    denominator[denominator == 0] = 1e-9 # Avoid division by zero
    return np.mean(numerator / denominator) * 100


def _calculate_pinball_loss_radar(
    y_true: np.ndarray,
    y_pred_q50: np.ndarray,
    quantile: float = 0.5 # Default to median for pinball
    ) -> float:
    """Calculates pinball loss for a specific quantile."""
    err = y_true - y_pred_q50
    return np.mean(np.maximum(quantile * err, (quantile - 1) * err))

def _calculate_smape_radar(
    y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
    """Calculates Symmetric Mean Absolute Percentage Error (SMAPE)."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Handle division by zero if both true and pred are zero
    # Add a small epsilon to denominator where it's zero
    epsilon = 1e-9
    score = np.mean(
        numerator / (denominator + epsilon)
        ) * 100
    return score


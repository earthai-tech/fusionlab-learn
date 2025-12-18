# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

"""
Forecast utilities.
"""
from __future__ import annotations
import os 
import re
import logging 
from collections.abc import Mapping, Sequence
from pathlib import Path
import json 
import inspect

from typing import ( 
    Dict, 
    Iterable, 
    List, 
    Union, 
    Any, 
    Optional,
    Tuple , 
    Callable,
    Literal, 
    MutableMapping,
)
import warnings
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib.style as style

import numpy as np 
import pandas as pd

from sklearn.preprocessing import MinMaxScaler 
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .._fusionlog import fusionlog 

from ..core.handlers import columns_manager 
from ..core.checks import check_spatial_columns, check_empty  
from ..core.io import is_data_readable, SaveFile
from ..decorators import isdf 
from ..metrics._registry import get_metric
from .generic_utils import vlog, normalize_model_inputs, ensure_directory_exists  
from .validator import is_frame 

logger = fusionlog().get_fusionlab_logger(__name__)

__all__= [ 
     'detect_forecast_type',
     'format_forecast_dataframe',
     'get_value_prefixes',
     'get_value_prefixes_in',
     'pivot_forecast_dataframe', 
     'get_step_names', 
     'stack_quantile_predictions', 
     'adjust_time_predictions', 
     'add_forecast_times', 
     'pivot_forecast', 
     'calibrate_forecasts', 
     'plot_reliability_diagram', 
     'format_and_forecast', 
     'evaluate_forecast'
     
]

_DIGIT_RE = re.compile(r"\d+")



def evaluate_forecast(
    eval_data: Union[str, os.PathLike, pd.DataFrame],
    *,
    target_name: str = "subsidence",
    column_map: Optional[Mapping[str, Any]] = None,
    quantile_interval: Tuple[float, float] = (0.1, 0.9),
    per_horizon: bool = False,
    extra_metrics: Optional[
        Union[Sequence[str], Mapping[str, Callable[..., Any]]]
    ] = None,
    extra_metric_kwargs: Optional[
        Mapping[str, Mapping[str, Any]]
    ] = None,
    savefile: Optional[Union[str, os.PathLike, bool]] = None,
    save_format: str = ".json",
    time_as_str: bool = True,
    verbose: int = 1,
    logger: Optional[logging.Logger] = None,
) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Evaluate forecast diagnostics from an evaluation DataFrame.

    This helper consumes the ``df_eval`` output from
    :func:`format_and_forecast` (or a compatible DataFrame) and
    computes aggregate metrics such as MAE, MSE, :math:`R^2`,
    coverage, and sharpness. It can also optionally evaluate
    metrics per forecast horizon and apply additional user-defined
    metrics.

    By default it expects the following columns:

    - ``'sample_idx'``
    - ``'forecast_step'``
    - ``'coord_t'``  (time)
    - Quantile or point-prediction columns for the target, e.g.:

      - Quantile mode:
        ``f'{target_name}_q10'``,
        ``f'{target_name}_q50'``,
        ``f'{target_name}_q90'``,
        ...
      - Point mode:
        ``f'{target_name}_pred'``.

    - Actual column:
      ``f'{target_name}_actual'``.

    A flexible ``column_map`` allows remapping these logical
    roles to arbitrary column names, e.g.::

        column_map = {
            'coord_t': 'date',
            'actual': 'true_subs',
            'pred': 'subs_predicted',
        }

    or, for quantile columns::

        column_map = {
            'coord_t': 'date',
            'quantiles': {
                0.1: 'subs_q10',
                0.5: 'subs_q50',
                0.9: 'subs_q90',
            },
        }

    Parameters
    ----------
    eval_data : str, path-like, or pandas.DataFrame
        Either a path to a CSV file containing the evaluation
        DataFrame (as saved by :func:`format_and_forecast`) or an
        in-memory DataFrame.

    target_name : str, default='subsidence'
        Base name for the target columns. Used to infer default
        column names such as ``f'{target_name}_q10'``,
        ``f'{target_name}_pred'``, and
        ``f'{target_name}_actual'``.

    column_map : dict, optional
        Optional mapping to override default column names. The
        following keys are recognized:

        - ``'sample_idx'`` : sample index column name
          (default ``'sample_idx'``).
        - ``'forecast_step'`` : horizon index column name
          (default ``'forecast_step'``).
        - ``'coord_t'`` : time coordinate column
          (default ``'coord_t'``).
        - ``'actual'`` : name or list of names for the actual
          target column(s). Currently a single column is supported;
          default ``f'{target_name}_actual'``.
        - ``'pred'`` : point prediction column for non-quantile
          mode, default ``f'{target_name}_pred'``.
        - ``'quantiles'`` :

          * If a mapping: ``{q: col_name}`` for quantile levels,
            where ``q`` is a float in (0, 1).
          * If a sequence of column names, the quantile value will
            be inferred from suffix patterns like
            ``f'{target_name}_q{int(q*100):d}'``.

    quantile_interval : tuple of float, default=(0.1, 0.9)
        Interval (lower, upper) used for coverage and sharpness
        metrics, typically corresponding to an 80% interval
        between Q10 and Q90.

    per_horizon : bool, default=False
        If ``True``, compute per-horizon MAE/MSE/R² grouped by
        the ``forecast_step`` column.

    extra_metrics : sequence of str or mapping, optional
        Optional additional metrics to compute.

        - If a sequence of strings (e.g. ``['pss', 'pit']``),
          each name is resolved via
          :func:`fusionlab.metrics._registry.get_metric`.
          If the name is not present in the registry, an error
          is raised, prompting the user to pass a callable instead.
        - If a mapping ``{name: func}``, each ``func`` is called as::

              func(y_true, y_pred, **extra_metric_kwargs.get(name, {}))

          where ``y_pred`` is the median (Q50) or point forecast.

        For more complex metrics that require full quantile
        structure or temporal sequences, pass a suitable wrapper
        function that internally uses the DataFrame as needed.

    extra_metric_kwargs : mapping, optional
        Optional mapping of per-metric keyword arguments. Keys must
        match the names in ``extra_metrics``. Each value is a
        dict of kwargs forwarded to the corresponding metric
        function.

    savefile : str, path-like, or bool, optional
        If provided, metrics are saved to disk.

        - If ``True``: a filename is auto-generated near
          ``eval_data`` (if it is a path) or in the current working
          directory.
        - If a string/path without extension: the extension is
          taken from ``save_format``.
        - If a string/path with extension: that extension takes
          precedence over ``save_format``.

    save_format : {'.json', 'json', '.csv', 'csv'}, default='.json'
        Output format when ``savefile`` is truthy. JSON preserves
        nested structure; CSV is flattened into a tall table.

        - For JSON, the function returns the metrics dictionary.
        - For CSV, the function returns the metrics DataFrame.

    time_as_str : bool, default=True
        If ``True``, time keys in the result dictionary are
        converted to strings (useful for JSON serialization). If
        there is only a single time value, the result is flattened
        and the time key is omitted.

    verbose : int, default=1
        Verbosity level passed to :func:`vlog`.

    logger : logging.Logger, optional
        Optional logger instance used by :func:`vlog`.

    Returns
    -------
    results : dict or pandas.DataFrame
        If ``save_format`` is JSON (default), returns a dict:

        - Single time value:

          .. code-block:: python

             {
                 "overall_mae": ...,
                 "overall_mse": ...,
                 "overall_r2": ...,
                 "coverage80": ...,
                 "sharpness80": ...,
                 "per_horizon_mae": {1: ..., 2: ..., ...},
                 ...
             }

        - Multiple time values:

          .. code-block:: python

             {
                 "2021": { ...metrics... },
                 "2022": { ...metrics... },
             }

        If ``save_format`` is CSV, returns a DataFrame with
        flattened rows:

        - Columns include: ``coord_t``, ``metric``, ``horizon``,
          and ``value``.

    Notes
    -----
    - Default metrics in *quantile* mode:

      - ``overall_mae``, ``overall_mse``, ``overall_r2``
      - ``coverage80`` and ``sharpness80`` (using the requested
        interval, e.g., Q10–Q90)

      If ``per_horizon=True``, also:

      - ``per_horizon_mae``, ``per_horizon_mse``,
        ``per_horizon_r2`` (each a mapping from horizon index to
        score).

    - Default metrics in *point* mode (no quantiles):

      - ``mae``, ``mse``, ``r2``

      And optionally, if ``per_horizon=True``:

      - ``per_horizon_mae``, ``per_horizon_mse``,
        ``per_horizon_r2``.
    """
    vlog(
        "[evaluate_forecast] Starting metrics computation.",
        verbose=verbose,
        level=1,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 1. Load DataFrame
    # ------------------------------------------------------------------
    source_path: Optional[Path] = None
    if isinstance(eval_data, (str, os.PathLike)):
        source_path = Path(eval_data)
        df = pd.read_csv(source_path)
        vlog(
            f"[evaluate_forecast] Loaded eval_data from '{source_path}'.",
            verbose=verbose,
            level=2,
            logger=logger,
        )
    elif isinstance(eval_data, pd.DataFrame):
        df = eval_data.copy()
        vlog(
            "[evaluate_forecast] Using in-memory eval DataFrame.",
            verbose=verbose,
            level=2,
            logger=logger,
        )
    else:
        raise TypeError(
            "eval_data must be a path or a pandas.DataFrame. "
            f"Got type={type(eval_data).__name__!r}."
        )

    column_map = dict(column_map or {})

    # Logical column names (with defaults)
    # sample_col = column_map.get("sample_idx", "sample_idx")
    step_col = column_map.get("forecast_step", "forecast_step")
    time_col = column_map.get("coord_t", "coord_t")

    # Actual column (required)
    actual_spec = column_map.get(
        "actual", f"{target_name}_actual"
    )
    if isinstance(actual_spec, str):
        actual_cols = [actual_spec]
    else:
        actual_cols = list(actual_spec)

    if len(actual_cols) != 1:
        raise ValueError(
            "evaluate_forecast currently supports exactly one "
            "actual target column. Got "
            f"{len(actual_cols)}: {actual_cols!r}."
        )
    actual_col = actual_cols[0]
    if actual_col not in df.columns:
        raise KeyError(
            f"Actual column '{actual_col}' is not present in "
            f"DataFrame columns={list(df.columns)!r}."
        )

    # ------------------------------------------------------------------
    # 2. Detect quantile vs point mode and build quantile map
    # ------------------------------------------------------------------
    quantiles_spec = column_map.get("quantiles", None)
    quantile_map: Dict[float, str] = {}

    if quantiles_spec is not None:
        # User explicitly provided quantile columns
        if isinstance(quantiles_spec, Mapping):
            for q, col in quantiles_spec.items():
                q_float = float(q)
                if col not in df.columns:
                    raise KeyError(
                        f"Quantile column '{col}' for q={q_float} "
                        "not present in DataFrame."
                    )
                quantile_map[q_float] = col
        else:
            # Sequence of column names; try to infer q from suffix
            from re import match, escape

            for col in quantiles_spec:
                if col not in df.columns:
                    raise KeyError(
                        f"Quantile column '{col}' not present in "
                        "DataFrame."
                    )
                # Pattern: f'{target_name}_q{int(q*100)}'
                pat = rf"^{escape(target_name)}_q(\d+)$"
                m = match(pat, col)
                if not m:
                    raise ValueError(
                        f"Cannot infer quantile level from column "
                        f"name '{col}'. Expected pattern "
                        f"'{target_name}_qXX'."
                    )
                q_int = int(m.group(1))
                quantile_map[q_int / 100.0] = col
    else:
        # Auto-detect quantiles from column names
        from re import match, escape

        pat = rf"^{escape(target_name)}_q(\d+)$"
        for col in df.columns:
            m = match(pat, col)
            if m:
                q_int = int(m.group(1))
                quantile_map[q_int / 100.0] = col

    quantile_mode = len(quantile_map) > 0

    # Prediction (median / point) column
    if quantile_mode:
        # Choose q closest to 0.5 as "median"
        median_q = min(quantile_map.keys(), key=lambda q: abs(q - 0.5))
        median_col = quantile_map[median_q]
        vlog(
            f"[evaluate_forecast] Detected quantile mode with "
            f"{len(quantile_map)} quantiles; median q~{median_q:.2f} "
            f"via column '{median_col}'.",
            verbose=verbose,
            level=2,
            logger=logger,
        )
    else:
        pred_col = column_map.get(
            "pred", f"{target_name}_pred"
        )
        if pred_col not in df.columns:
            raise KeyError(
                "Point-prediction mode inferred, but prediction "
                f"column '{pred_col}' is missing."
            )
        median_col = pred_col
        vlog(
            f"[evaluate_forecast] Detected point mode using "
            f"column '{median_col}' for predictions.",
            verbose=verbose,
            level=2,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # 3. Group by time (coord_t) if available
    # ------------------------------------------------------------------
    has_time = time_col in df.columns
    if has_time:
        groups = df.groupby(time_col)
        vlog(
            f"[evaluate_forecast] Grouping by time column "
            f"'{time_col}'. Found {groups.ngroups} group(s).",
            verbose=verbose,
            level=2,
            logger=logger,
        )
    else:
        # Single pseudo-group with key None
        groups = [(None, df)]
        vlog(
            "[evaluate_forecast] No time column present; "
            "treating entire DataFrame as one group.",
            verbose=verbose,
            level=2,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # 4. Prepare extra metrics (names -> callables)
    # ------------------------------------------------------------------
    metric_fns: Dict[str, Callable[..., Any]] = {}
    if extra_metrics is not None:
        if isinstance(extra_metrics, Mapping):
            metric_fns = dict(extra_metrics)
        else:
            # Sequence of names: resolve via get_metric
            for name in extra_metrics:
                try:
                    metric_fns[name] = get_metric(name)
                except KeyError as exc:
                    raise ValueError(
                        f"Metric name '{name}' not found in "
                        "fusionlab.metrics registry. "
                        "Please either:\n"
                        "  - register it in METRIC_REGISTRY, or\n"
                        "  - pass an explicit callable via "
                        "`extra_metrics={'name': fn}`."
                    ) from exc

    extra_metric_kwargs = dict(extra_metric_kwargs or {})

    # ------------------------------------------------------------------
    # 5. Core metric computation loop
    # ------------------------------------------------------------------
    results_by_time: MutableMapping[Any, Dict[str, Any]] = {}

    # Pre-resolve coverage & sharpness metrics, if quantile mode
    if quantile_mode:
        try:
            coverage_fn = get_metric("coverage_score")
        except KeyError:
            coverage_fn = None

        try:
            miw_fn = get_metric("mean_interval_width_score")
        except KeyError:
            miw_fn = None
    else:
        coverage_fn = None
        miw_fn = None

    q_lower_target, q_upper_target = quantile_interval

    for t_val, g in (groups if has_time else groups):
        # When has_time is False, groups is a list of tuples
        # (None, df). When True, groups is a pandas GroupBy and
        # iteration yields (t_val, g) pairs directly.
        if not has_time:
            t_key = None
        else:
            t_key = t_val

        y_true = g[actual_col].to_numpy()
        y_pred = g[median_col].to_numpy()

        metrics_here: Dict[str, Any] = {}

        # ---- 5a. Default deterministic metrics ------------------------
        if quantile_mode:
            metrics_here["overall_mae"] = float(
                mean_absolute_error(y_true, y_pred)
            )
            metrics_here["overall_mse"] = float(
                mean_squared_error(y_true, y_pred)
            )
            # R² requires at least 2 samples
            if y_true.size >= 2:
                metrics_here["overall_r2"] = float(
                    r2_score(y_true, y_pred)
                )
            else:
                metrics_here["overall_r2"] = np.nan
        else:
            metrics_here["mae"] = float(
                mean_absolute_error(y_true, y_pred)
            )
            metrics_here["mse"] = float(
                mean_squared_error(y_true, y_pred)
            )
            if y_true.size >= 2:
                metrics_here["r2"] = float(
                    r2_score(y_true, y_pred)
                )
            else:
                metrics_here["r2"] = np.nan

        # ---- 5b. Coverage and sharpness (quantile mode only) ----------
        if quantile_mode and coverage_fn is not None and miw_fn is not None:
            # Choose quantiles closest to requested interval
            qs_sorted = sorted(quantile_map.keys())
            q_low = min(
                qs_sorted, key=lambda q: abs(q - q_lower_target)
            )
            q_high = min(
                qs_sorted, key=lambda q: abs(q - q_upper_target)
            )

            y_lower = g[quantile_map[q_low]].to_numpy()
            y_upper = g[quantile_map[q_high]].to_numpy()

            # coverage80
            try:
                cov_val = coverage_fn(
                    y_true,
                    y_lower,
                    y_upper,
                    nan_policy="omit",
                    verbose=0,
                )
            except TypeError:
                cov_val = coverage_fn(y_true, y_lower, y_upper)
            metrics_here["coverage80"] = float(cov_val)

            # sharpness80 (mean interval width)
            try:
                miw_val = miw_fn(
                    y_lower,
                    y_upper,
                    nan_policy="omit",
                    verbose=0,
                )
            except TypeError:
                miw_val = miw_fn(y_lower, y_upper)
            metrics_here["sharpness80"] = float(miw_val)

        # ---- 5c. Per-horizon deterministic metrics --------------------
        if per_horizon and step_col in g.columns:
            per_mae: Dict[int, float] = {}
            per_mse: Dict[int, float] = {}
            per_r2: Dict[int, float] = {}

            for h, g_h in g.groupby(step_col):
                yt_h = g_h[actual_col].to_numpy()
                yp_h = g_h[median_col].to_numpy()

                h_int = int(h)
                per_mae[h_int] = float(
                    mean_absolute_error(yt_h, yp_h)
                )
                per_mse[h_int] = float(
                    mean_squared_error(yt_h, yp_h)
                )
                if yt_h.size >= 2:
                    per_r2[h_int] = float(
                        r2_score(yt_h, yp_h)
                    )
                else:
                    per_r2[h_int] = np.nan

            metrics_here["per_horizon_mae"] = per_mae
            metrics_here["per_horizon_mse"] = per_mse
            metrics_here["per_horizon_r2"] = per_r2

        # ---- 5d. Extra metrics ----------------------------------------
        for m_name, fn in metric_fns.items():
            kwargs = dict(extra_metric_kwargs.get(m_name, {}) or {})
        
            sig = inspect.signature(fn)
            param_names = list(sig.parameters.keys())
        
            # Case 1: standard supervised metric: metric(y_true, y_pred, ...)
            if ("y_true" in param_names) and ("y_pred" in param_names):
                # Prefer explicit kw usage if supported
                try:
                    m_val = fn(y_true=y_true, y_pred=y_pred, **kwargs)
                except TypeError:
                    # Fall back to positional (y_true, y_pred)
                    m_val = fn(y_true, y_pred, **kwargs)
        
            # Case 2: prediction-only metric: metric(y_pred, sample_weight=..., ...)
            elif param_names and param_names[0] in ("y_pred", "y_forecast", "y_hat"):
                # e.g. prediction_stability_score(y_pred, sample_weight=None, ...)
                m_val = fn(y_pred, **kwargs)
        
            else:
                # Fallback heuristic: try (y_true, y_pred) then (y_pred,)
                try:
                    m_val = fn(y_true, y_pred, **kwargs)
                except TypeError:
                    m_val = fn(y_pred, **kwargs)
        
            # Try to convert scalar arrays to Python floats
            if isinstance(m_val, np.ndarray) and m_val.size == 1:
                m_val = float(m_val.reshape(()))
        
            metrics_here[m_name] = m_val

        results_by_time[t_key] = metrics_here

    # ------------------------------------------------------------------
    # 6. Flatten for single vs multi time
    # ------------------------------------------------------------------
    if len(results_by_time) == 1:
        # Single time or no time column → return metrics only
        final_result: Union[Dict[str, Any], Dict[Any, Any]]
        final_result = next(iter(results_by_time.values()))
    else:
        # Multiple time values → dict keyed by time
        final_result = {}
        for k, v in results_by_time.items():
            if time_as_str and k is not None:
                final_result[str(k)] = v
            else:
                final_result[k] = v

    # ------------------------------------------------------------------
    # 7. Optionally save to disk
    # ------------------------------------------------------------------
    if savefile:
        # Normalize save_format
        fmt = save_format.lower()
        if not fmt.startswith("."):
            fmt = f".{fmt}"

        # Determine output path
        if isinstance(savefile, bool):
            # Auto-generate filename
            base_dir = (
                source_path.parent
                if source_path is not None
                else Path.cwd()
            )
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            if fmt in (".csv",):
                fname = f"eval_{target_name}_diagnostics_{ts}.csv"
            else:
                fname = f"eval_{target_name}_diagnostics_{ts}.json"
            out_path = base_dir / fname
        else:
            out_path = Path(savefile)
            if out_path.suffix:
                # Explicit extension takes precedence
                fmt = out_path.suffix.lower()
            else:
                out_path = out_path.with_suffix(fmt)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt in (".json", "json"):
            # Ensure JSON-serializable (convert numpy scalars)
            def _to_jsonable(obj: Any) -> Any:
                if isinstance(obj, (np.generic,)):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            json_ready = {}

            if len(results_by_time) == 1:
                json_ready = {
                    k: _to_jsonable(v)
                    for k, v in final_result.items()  # type: ignore[arg-type]
                }
            else:
                for t_key, metrics_here in final_result.items():  # type: ignore[union-attr]
                    json_ready[t_key] = {
                        k: _to_jsonable(v)
                        for k, v in metrics_here.items()
                    }

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    json_ready,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            vlog(
                f"[evaluate_forecast] Metrics saved to JSON at "
                f"'{out_path}'.",
                verbose=verbose,
                level=1,
                logger=logger,
            )
            return final_result  # dict

        elif fmt in (".csv", "csv"):
            # Flatten to long-form DataFrame
            rows = []
            for t_key, metrics_here in results_by_time.items():
                t_val = t_key
                if time_as_str and t_val is not None:
                    t_val = str(t_val)
                for m_name, m_val in metrics_here.items():
                    if isinstance(m_val, Mapping):
                        # e.g. per_horizon metrics: dict[horizon -> value]
                        for h, v in m_val.items():
                            rows.append(
                                {
                                    "coord_t": t_val,
                                    "metric": m_name,
                                    "horizon": h,
                                    "value": v,
                                }
                            )
                    else:
                        rows.append(
                            {
                                "coord_t": t_val,
                                "metric": m_name,
                                "horizon": None,
                                "value": m_val,
                            }
                        )
            df_out = pd.DataFrame(rows)
            df_out.to_csv(out_path, index=False)
            vlog(
                f"[evaluate_forecast] Metrics saved to CSV at "
                f"'{out_path}'.",
                verbose=verbose,
                level=1,
                logger=logger,
            )
            return df_out

        else:
            raise ValueError(
                f"Unsupported save_format/extension '{fmt}'. "
                "Use '.json' or '.csv'."
            )

    # No saving requested: just return in-memory result
    return final_result


def _find_scaler_block(scaler_info, keys=("targets", "target", "y")):
    """Best-effort helper to locate a scaler block inside scaler_info."""
    if not isinstance(scaler_info, dict):
        return None
    for k in keys:
        if k in scaler_info:
            block = scaler_info.get(k)
            if isinstance(block, dict):
                return block
    return None

def _inverse_with_block(values, block, col_name):
    """Inverse-transform a 1D array using a scaler block.

    Supports both:
    - old schema  : block["columns"] / block["cols"]
    - new PINN schema: block["all_features"] + block["idx"]
    """
    if block is None or not isinstance(block, dict):
        return values

    scaler = block.get("scaler")
    if scaler is None or not hasattr(scaler, "inverse_transform"):
        return values

    # ---- Old style: columns/cols present ---------------------------------
    cols = block.get("columns") or block.get("cols")
    if cols:
        if col_name not in cols:
            return values
        idx = cols.index(col_name)
        n_features = len(cols)
    else:
        # ---- New PINN style: use all_features + idx -----------------------
        idx = block.get("idx")
        if idx is None:
            return values
        all_features = block.get("all_features")
        if all_features:
            n_features = len(all_features)
        else:
            # Fallback: trust the scaler
            n_features = getattr(scaler, "n_features_in_", idx + 1)

    values = np.asarray(values).reshape(-1)
    tmp = np.zeros((values.shape[0], n_features), dtype=np.float32)
    tmp[:, idx] = values

    try:
        inv = scaler.inverse_transform(tmp)[:, idx]
    except Exception:
        return values
    return inv

def _auto_output_target_name(
    target_name: str,
    output_target_name: str | None,
    *,
    suffixes: tuple[str, ...] = ("_cum", "_cumulative"),
) -> str:
    """
    If output_target_name is not provided, derive a clean output prefix.

    Examples
    --------
    target_name='subsidence_cum'   -> 'subsidence'
    target_name='subsidence'       -> 'subsidence'
    target_name='subsidence_cumulative' -> 'subsidence'
    """
    if output_target_name is not None and str(output_target_name).strip():
        return str(output_target_name)

    name = str(target_name)
    for sfx in suffixes:
        if name.endswith(sfx):
            return name[: -len(sfx)]
    return name

def format_and_forecast(
    y_pred: dict,
    y_true: dict | None,
    *,
    coords: np.ndarray | None = None,
    quantiles: list[float] | None = None,
    
    target_name: str = "subsidence",
    output_target_name: str | None = None,   
    scaler_target_name: str | None = None,   

    target_key_pred: str = "subs_pred",
    component_index: int = 0,
    scaler_info: dict | None = None,
    coord_scaler: object | None = None,
    coord_columns: tuple[str, str, str] = ("coord_t", "coord_x", "coord_y"),
    #
    # Temporal config
    train_end_time: float | int | str | None = None,
    forecast_start_time: float | int | str | None = None,
    forecast_horizon: int | None = None,
    future_time_grid: np.ndarray | None = None,
    eval_forecast_step: int | None = None,
    #
    eval_export: object = "all",
    value_mode: str = "rate",   
    input_value_mode: str = "rate",    # what MODEL produced (NEW)
    rate_first: str = "cum_over_dtref",# for cumulative->rate (NEW)
    absolute_baseline: float | Mapping[int, float] | None = None,  
    # Output / I/O
    sample_index_offset: int = 0,
    city_name: str | None = None,
    model_name: str | None = None,
    dataset_name: str | None = None,
    csv_eval_path: str | None = None,
    csv_future_path: str | None = None,
    #
    # Time dtype control
    time_as_datetime: bool = False,
    time_format: str | None = None,
    #
    # Evaluation options (metrics, optional)
    eval_metrics: bool = False,
    metrics_column_map: Mapping[str, Any] | None = None,
    metrics_quantile_interval: tuple[float, float] = (0.1, 0.9),
    metrics_per_horizon: bool = False,
    metrics_extra: (
        Sequence[str]
        | Mapping[str, Callable[..., Any]]
        | None
    ) = None,
    metrics_extra_kwargs: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
    metrics_savefile: str | os.PathLike | bool | None = None,
    metrics_save_format: str = ".json",
    metrics_time_as_str: bool = True,
    # Logging
    verbose: int = 1,
    logger: logging.Logger | None = None,
    **kws
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Format PINN forecasts into evaluation and future DataFrames.

    This helper takes the raw model outputs (already split into
    ``y_pred['subs_pred']`` / ``y_pred['gwl_pred']``), the matching
    ground-truth dictionary (``y_true``), and optional coordinate and
    scaler information, and returns two DataFrames:

    - ``df_eval``: predictions + actuals for an evaluation year
      (typically the last training year, e.g. 2022).
    - ``df_future``: predictions for the future horizon
      (e.g. 2023–2025), without actuals.

    Parameters
    ----------
    y_pred : dict
        Dictionary of model predictions, as returned by
        ``GeoPriorSubsNet.predict`` post-processed into
        ``{'subs_pred': ..., 'gwl_pred': ...}``.

        For subsidence, the expected shapes are:

        - Quantile mode: ``(B, H, Q, O)`` where:
          ``B`` = batch size, ``H`` = horizon steps,
          ``Q`` = number of quantiles, ``O`` = output dim.
        - Point mode: ``(B, H, O)``.

    y_true : dict or None
        Dictionary of true targets, typically

        ``{'subsidence': ..., 'gwl': ...}`` or
        ``{'subs_pred': ..., 'gwl_pred': ...}``.

        If ``None``, evaluation DataFrame is still created but
        without the ``'<target_name>_actual'`` column.

    coords : ndarray, optional
        Optional coordinates array aligned with predictions.
        Commonly shaped ``(B, H, 3)`` with columns
        ``[t_scaled, x_scaled, y_scaled]``.  Only ``x`` and ``y``
        are used when inverse-transforming spatial coordinates;
        time is overwritten by the provided temporal config if
        given.

    quantiles : list of float or None, optional
        List of quantiles (e.g. ``[0.1, 0.5, 0.9]``) if the model
        was trained in probabilistic mode.  If ``None``, a single
        column ``'<target_name>_pred'`` is emitted instead.

    target_name : str, default='subsidence'
        Logical target identifier used as the default key for locating the
        target scaler in ``scaler_info`` and as a fallback for resolving
        truth arrays in ``y_true``.
    
        Column naming is controlled by ``output_target_name`` (or the
        auto-derived output prefix when it is ``None``).

    output_target_name : str or None, optional
        Output prefix used when creating DataFrame columns for predictions
        and actuals.
    
        This controls the *column naming* only (e.g. the function will
        emit ``f"{output_target_name}_q10"``, ``f"{output_target_name}_pred"``,
        and ``f"{output_target_name}_actual"``).
    
        If ``None`` (default), the function derives the output prefix from
        ``target_name`` and applies a small convenience rule:
    
        * if ``target_name`` ends with ``"_cum"`` (or ``"_cumulative"``),
          the suffix is stripped for output naming.
    
        This keeps downstream tooling consistent (many plotting and metrics
        utilities expect names like ``subsidence_q10`` rather than
        ``subsidence_cum_q10``), while still allowing the scaler lookup to
        use the true target key::
    
            >>> target_name = "subsidence_cum"
            >>> output_target_name = None
            # Output columns become: subsidence_q10, subsidence_q50, subsidence_actual
        
            >>> target_name = "subsidence_cum"
            >>> output_target_name = "subsidence_cum"
            # Output columns keep the suffix: subsidence_cum_q10, ...
    
    scaler_target_name : str or None, optional
        Name used to locate the target scaling block inside ``scaler_info``
        and to perform inverse-transform for predictions and actuals.
    
        This controls the *scaler key and inverse scaling*, not the output
        column naming.
    
        If ``None`` (default), the scaler key is assumed to be
        ``target_name``. This is important when you want clean output
        columns but the scaler was fitted/stored under the original target
        name.
    
        Notes:
        
        A common pattern is to keep ``target_name="subsidence_cum"`` so the
        scaler lookup matches the Stage-1 schema, while letting
        ``output_target_name=None`` produce clean output columns:
    
        * inverse-transform uses ``scaler_target_name="subsidence_cum"``
          (implicitly via ``target_name``),
        * output columns are named ``subsidence_*`` due to the auto-strip
          rule.

    target_key_pred : str, default='subs_pred'
        Key inside ``y_pred`` that holds the subsidence forecasts.

    component_index : int, default=0
        Index along the output dimension O to use when
        ``output_subsidence_dim > 1``.  For scalar subsidence this
        is 0.

    scaler_info : dict, optional
        Optional Stage-1 ``scaler_info`` mapping containing a
        target scaler under keys such as ``'targets'`` or
        ``'target'``.  The block is expected to have fields:

        - ``'scaler'``: an sklearn-like transformer.
        - ``'columns'`` or ``'cols'``: list of column names.

        If present and consistent, subsidence values (predicted and
        actual) are inverse-transformed for ``target_name``.

    coord_scaler : object, optional
        Optional scaler used for coordinates.  If provided, it is
        only used to inverse-transform ``coord_x`` and ``coord_y``
        when ``coords`` is given and ``coord_columns`` can be
        matched.  Time is *not* taken from the inverse transform; it
        is controlled by the temporal config.

    coord_columns : tuple of str, default=('coord_t','coord_x','coord_y')
        Logical names of the time, x, and y coordinate columns.
        These are used for DataFrame column naming and for mapping
        into ``coord_scaler`` if its block carries column names.

    train_end_time : scalar or str or datetime, optional
        Physical time associated with the evaluation year (e.g.
        2022).  If ``eval_forecast_step`` is not given, the last
        horizon step is assumed to correspond to this time.

    forecast_start_time : scalar or str or datetime, optional
        First time in the future forecast horizon (e.g. 2023).

    forecast_horizon : int, optional
        Number of forecast steps in the future horizon (e.g. 3).
        If ``future_time_grid`` is not given, this is used together
        with ``forecast_start_time`` to build a regular grid.

    future_time_grid : array-like, optional
        Explicit physical times for each forecast step, length H.
        For yearly data this might be ``[2023, 2024, 2025]``.
        If provided, it overrides any automatic construction from
        ``forecast_start_time`` and ``forecast_horizon``.

    eval_forecast_step : int or None, optional
        Horizon step index (1-based) to use for evaluation.  If
        ``None``, defaults to the last horizon step H.
        
    eval_export : {"all", "last"} or str or int or sequence, optional
        Controls which evaluation rows are exported in ``df_eval`` and
        written to ``csv_eval_path``. By default (``"all"``), the
        function exports the multi-horizon evaluation DataFrame
        (``df_eval_all``), which contains one row per sample and
        forecast step (e.g. years 2020, 2021, 2022 for ``H=3``).

        Accepted values are:

        * ``"all"`` or ``"full"`` or ``"horizons"`` :
          export all horizons from ``df_eval_all``.
        * ``"last"`` or ``"single"`` or ``"default"`` :
          export only the single evaluation step specified by
          ``eval_forecast_step`` (backwards-compatible behaviour).
        * Other ``str`` (e.g. ``"2022"``) :
          interpreted as a time value for ``coord_t``; only rows of
          ``df_eval_all`` whose time column matches this value are
          exported.
        * ``int`` or scalar non-string :
          interpreted as a single time value (e.g. ``2022``).
        * sequence of values (e.g. ``[2021, 2022]``) :
          interpreted as a set of time values; only rows whose
          ``coord_t`` belongs to this set are exported.

        If ``time_as_datetime=True``, the selection values are
        converted with ``pandas.to_datetime`` using ``time_format``
        before filtering. If ``df_eval_all`` is not available
        (e.g. no ground truth was provided), the function falls back
        to exporting the single-step ``df_eval`` regardless of
        ``eval_export``.

    value_mode : {"rate", "cumulative", "absolute_cumulative"}, optional
        Controls how forecast values are interpreted along the temporal
        horizon for each sample. The default is ``"rate"``, which
        treats each forecast step as an incremental rate (e.g.
        annual subsidence rate) and leaves predictions unchanged.

        Supported modes are:

        * ``"rate"`` :
          keep per-step predictions as provided by the model
          (current behaviour).
        * ``"cumulative"`` or ``"cum"`` :
          convert per-step rates into *relative cumulative* values
          by applying a cumulative sum over ``forecast_step`` for
          each ``sample_idx``. For example, for years 2023–2025,
          the value at 2024 is the sum of the 2023 and 2024 rates.
        * ``"absolute_cumulative"`` or ``"abs_cum"`` or ``"absolute"`` :
          same as ``"cumulative"``, then add an absolute baseline
          provided by ``absolute_baseline`` (e.g. cumulative
          subsidence at the end of the training period), yielding
          absolute cumulative trajectories.

        Cumulative transforms are applied consistently to:

        * the future forecast DataFrame (``df_future``),
        * the multi-horizon evaluation DataFrame (``df_eval_all``),
        * and the single-step evaluation DataFrame (``df_eval``,
          which is regenerated from ``df_eval_all`` after the
          transformation).

        When an unsupported string is given, the function logs a
        warning and falls back to ``"rate"``.

    absolute_baseline : float or Mapping[int, float], optional
        Baseline value to use when ``value_mode`` requests absolute
        cumulative outputs (``"absolute_cumulative"``, ``"abs_cum"``,
        ``"absolute"``). This baseline is interpreted as the
        pre-forecast cumulative level for each sample, for example,
        cumulative subsidence at ``train_end_time`` (e.g. end of
        2022), and is added *after* applying the cumulative sum over
        the forecast horizon.

        If a scalar ``float`` is provided, the same baseline value is
        added to all samples. If a mapping is provided, it must map
        ``sample_idx`` (integers) to baseline values, allowing
        per-sample baselines:

        * ``absolute_baseline = {sample_idx: baseline_value, ...}``

        Only prediction columns for ``target_name`` are shifted (e.g.
        ``"subsidence_q10"``, ``"subsidence_q50"``, ``"subsidence_q90"``
        or ``"subsidence_pred"``). When ``df_eval_all`` is present,
        the corresponding ``"<target_name>_actual"`` column is shifted
        as well, so evaluation metrics operate on absolute cumulative
        values.

        If ``value_mode`` is an absolute cumulative variant but
        ``absolute_baseline`` is ``None``, the function logs a warning
        and degrades gracefully to relative cumulative mode (i.e. no
        baseline shift is applied).

    sample_index_offset : int, default=0
        Offset added to ``sample_idx`` (useful when concatenating
        multiple tiles).

    city_name, model_name, dataset_name : str, optional
        Optional metadata used only for logging.

    csv_eval_path : str, optional
        If provided, ``df_eval`` is written to this path
        (directories are created if needed).

    csv_future_path : str, optional
        If provided, ``df_future`` is written to this path.

    time_as_datetime : bool, default=False
        If ``True``, time values are converted using
        :func:`pandas.to_datetime` with the provided
        ``time_format`` (if any).

    time_format : str or None, optional
        Optional format string passed to :func:`pandas.to_datetime`
        when ``time_as_datetime=True``.

    eval_metrics : bool, default=False
        If ``True``, automatically call
        :func:`evaluate_forecast` on the resulting
        ``df_eval`` to compute diagnostics.  Metrics are
        not returned by this function; they are either
        written to disk (if ``metrics_savefile`` is
        provided) or discarded.  For programmatic access
        to the metrics dictionary, call
        :func:`evaluate_forecast` directly.

    metrics_column_map : mapping, optional
        Optional column mapping forwarded to
        :func:`evaluate_forecast` (see its documentation
        for details).  If ``None``, default column names
        such as ``'coord_t'``, ``'forecast_step'``,
        ``f'{target_name}_q10'``, and
        ``f'{target_name}_actual'`` are assumed.

    metrics_quantile_interval : tuple of float, default=(0.1, 0.9)
        Interval used for coverage and sharpness
        diagnostics in quantile mode, forwarded to
        :func:`evaluate_forecast`.

    metrics_per_horizon : bool, default=False
        If ``True``, per-horizon MAE/MSE/R² are computed
        by :func:`evaluate_forecast` and included in the
        diagnostics.

    metrics_extra : sequence or mapping, optional
        Optional additional metrics to compute, forwarded
        to :func:`evaluate_forecast`.  Can be:

        - A sequence of metric names (resolved via
          ``fusionlab.metrics._registry.get_metric``).
        - A mapping ``{name: func}`` where ``func`` is a
          callable taking ``(y_true, y_pred, **kwargs)``.

    metrics_extra_kwargs : mapping, optional
        Optional per-metric keyword arguments, forwarded
        to :func:`evaluate_forecast`.  Keys must match
        metric names in ``metrics_extra``.

    metrics_savefile : str, path-like, bool, or None
        If truthy, diagnostics from
        :func:`evaluate_forecast` are written to disk.
        Behavior matches the ``savefile`` argument of
        :func:`evaluate_forecast`.  When ``True``, a
        filename is auto-generated near the evaluation
        CSV (if any) or in the current working
        directory.

    metrics_save_format : {'.json', 'json', '.csv', 'csv'}, default='.json'
        Output format for diagnostics written by
        :func:`evaluate_forecast`.  JSON preserves the
        nested metric structure; CSV flattens it into a
        tall table.

    metrics_time_as_str : bool, default=True
        If ``True``, time keys in the diagnostics written
        by :func:`evaluate_forecast` are converted to
        strings (useful for JSON serialization).

    verbose : int, default=1
        Verbosity level passed to :func:`vlog`.

    logger : logging.Logger, optional
        Logger instance; if ``None``, a module-level ``LOG`` is
        used.

    Returns
    -------
    df_eval_to_write : pandas.DataFrame
        DataFrame containing predictions and actuals for the
        evaluation time.  Columns include:

        - ``'sample_idx'``
        - ``'forecast_step'``
        - quantile columns (e.g. ``subsidence_q10``) or
          ``subsidence_pred``
        - ``'subsidence_actual'`` (if y_true given)
        - ``coord_t``, ``coord_x``, ``coord_y`` (names from
          ``coord_columns``).

    df_future : pandas.DataFrame
        DataFrame containing predictions for the future horizon,
        without actuals.  Same structure as ``df_eval`` but without
        the ``'<target_name>_actual'`` column.
    
    Notes
    -----
    This function separates *scaler lookup* (``scaler_target_name``) from
    *output column naming* (``output_target_name``). 
    This is useful when the stored scaler key contains suffixes like 
    ``"_cum"`` but downstream tools expect canonical names such 
    as ``subsidence_*``.


    """

    t_col, x_col, y_col = coord_columns
    scale_name = scaler_target_name or target_name
    out_name = _auto_output_target_name(target_name, output_target_name)

    vlog(
        "[format_and_forecast] Starting formatting of predictions "
        f"for city={city_name}, model={model_name}, "
        f"dataset={dataset_name}",
        verbose=verbose,
        level=1,
        logger = logger
    )

    if target_key_pred not in y_pred:
        raise KeyError(
            f"y_pred must contain key '{target_key_pred}'. "
            f"Got keys={list(y_pred.keys())}"
        )

    subs_pred = np.asarray(y_pred[target_key_pred])
    if subs_pred.ndim not in (3, 4):
        raise ValueError(
            f"{target_key_pred} must have ndim 3 or 4. "
            f"Got shape={subs_pred.shape!r}"
        )

    B = subs_pred.shape[0]
    H = subs_pred.shape[1]

    # Choose evaluation horizon step (1-based → 0-based index)
    if eval_forecast_step is None or not (1 <= eval_forecast_step <= H):
        eval_forecast_step = H  # last step by default
    s_idx = eval_forecast_step - 1

    vlog(
        f"[format_and_forecast] B={B}, H={H}, eval_step={eval_forecast_step}",
        verbose=verbose,
        level=2,
        logger = logger
    )

    # ------------------------------------------------------------------
    # Split predictions into eval (one horizon step) and future (all H)
    # ------------------------------------------------------------------
    if quantiles is not None:
        # Expected shape: (B, H, Q, O)
        if subs_pred.ndim != 4:
            raise ValueError(
                "Quantile mode requires subs_pred.ndim == 4. "
                f"Got shape={subs_pred.shape!r}"
            )
        Q = subs_pred.shape[2]
        if len(quantiles) != Q:
            vlog(
                "[format_and_forecast] Warning: len(quantiles) does not "
                f"match Q={Q}; using first {min(len(quantiles), Q)}.",
                verbose=verbose,
                level=1,
                logger = logger
            )
            Q = min(len(quantiles), Q)

        # (B, Q) eval, (B, H, Q) future
        subs_eval_q = subs_pred[:, s_idx, :Q, component_index]
        subs_future_q = subs_pred[:, :, :Q, component_index]
    else:
        # Point forecasts: expected shape (B, H, O)
        if subs_pred.ndim != 3:
            raise ValueError(
                "Point mode requires subs_pred.ndim == 3. "
                f"Got shape={subs_pred.shape!r}"
            )
        subs_eval_q = subs_pred[:, s_idx, component_index]    # (B,)
        subs_future_q = subs_pred[:, :, component_index]      # (B, H)

    # ------------------------------------------------------------------
    # Ground truth for evaluation (last step + all horizons)
    # ------------------------------------------------------------------
    subs_true_eval = None          # (B,)
    subs_true_all_flat = None      # (B*H,)
    true_arr = None

    true_key_candidates = (out_name, scale_name, target_key_pred, target_name)
    
    true_arr = None
    if y_true is not None:
        for k in true_key_candidates:
            if k in y_true:
                true_arr = np.asarray(y_true[k])
                break
            
    if true_arr is not None:
        if true_arr.ndim != 3:
            vlog(
                "[format_and_forecast] y_true has unexpected ndim; "
                "skipping actual inverse transform.",
                verbose=verbose,
                level=1,
                logger=logger,
            )
        else:
            # (B, H, O) → pick component_index
            subs_true_eval = true_arr[:, s_idx, component_index]          # (B,)
            subs_true_all_flat = true_arr[:, :, component_index].reshape(-1)  # (B*H,)

    # ------------------------------------------------------------------
    # Coords (only used for x/y; time will come from temporal config)
    # ------------------------------------------------------------------
    coord_x_eval = coord_y_eval = None
    coord_x_future = coord_y_future = None
    
    if coords is not None:
        coords = np.asarray(coords)
        if coords.ndim == 3 and coords.shape[-1] >= 3:
            B_coords, H_coords, D_coords = coords.shape
    
            # Optional sanity check: coords horizon should match predictions
            if H_coords != H:
                vlog(
                    "[format_and_forecast] Warning: coords.shape[1] "
                    f"({H_coords}) != H ({H}); continuing but shapes "
                    "may be misaligned.",
                    verbose=verbose,
                    level=1,
                    logger=logger,
                )
            # ---- NEW: inverse-transform coords if we have a coord_scaler ----
            if coord_scaler is not None and hasattr(
                    coord_scaler, "inverse_transform"):
                try:
                    flat = coords.reshape(-1, D_coords)        # (B*H, D)
                    flat_inv = coord_scaler.inverse_transform(flat)
                    coords = flat_inv.reshape(B_coords, H_coords, D_coords)
                except Exception:
                    vlog(
                        "[format_and_forecast] coord_scaler inverse_transform "
                        "failed; falling back to scaled coords.",
                        verbose=verbose,
                        level=1,
                        logger=logger,
                    )
    
            # Now coords are in physical units; extract x/y for eval + future
            coord_x_eval = coords[:, s_idx, 1]
            coord_y_eval = coords[:, s_idx, 2]
    
            coord_x_future = coords[:, :, 1].reshape(-1)
            coord_y_future = coords[:, :, 2].reshape(-1)
        else:
            vlog(
                "[format_and_forecast] coords has unexpected shape; "
                "spatial coords will not be included.",
                verbose=verbose,
                level=1,
                logger=logger,
            )

    # ------------------------------------------------------------------
    # Inverse-transform subsidence (pred + actual) using scaler_info
    # ------------------------------------------------------------------
    if isinstance(scaler_info, dict) and scale_name  in scaler_info:
        # New Stage-1 schema: one entry per target (e.g. "subsidence")
        target_block = scaler_info[scale_name]
    else:
        # Backwards compatibility with old "targets"/"target"/"y" schema
        target_block = _find_scaler_block(scaler_info)

    
    # Eval
    if quantiles is not None:
        eval_cols = []
        for j in range(subs_eval_q.shape[1]):
            vals = subs_eval_q[:, j].reshape(-1)
            vals_inv = _inverse_with_block(vals, target_block, scale_name)
            eval_cols.append(vals_inv)
        subs_eval_q = np.stack(eval_cols, axis=1)  # (B, Q)
    else:
        vals = subs_eval_q.reshape(-1)
        subs_eval_q = _inverse_with_block(vals, target_block, scale_name)

    # Inverse-transform ground truth (last step + all horizons)
    if subs_true_eval is not None:
        subs_true_eval = _inverse_with_block(
            subs_true_eval.reshape(-1), target_block, scale_name
        )

    if subs_true_all_flat is not None:
        subs_true_all_flat = _inverse_with_block(
            subs_true_all_flat.reshape(-1), target_block, scale_name
        )

    # Future
    if quantiles is not None:
        future_cols = []
        for j in range(subs_future_q.shape[2]):
            vals = subs_future_q[:, :, j].reshape(-1)
            vals_inv = _inverse_with_block(vals, target_block, scale_name)
            future_cols.append(vals_inv)
        subs_future_flat = np.stack(future_cols, axis=1)  # (B*H, Q)
    else:
        vals = subs_future_q.reshape(-1)
        subs_future_flat = _inverse_with_block(
            vals, target_block, scale_name
        ).reshape(-1)

    # ------------------------------------------------------------------
    # Build temporal grid for future horizon (if not given)
    # ------------------------------------------------------------------
    if future_time_grid is None and forecast_start_time is not None:
        if forecast_horizon is None:
            forecast_horizon = H
        # For generality, we don't assume "years"; we simply treat
        # forecast_start_time as the value for step=1 and increment
        # by 1 for each subsequent step.  If the user needs monthly
        # or arbitrary spacing, they can pass an explicit
        # ``future_time_grid``.
        base = forecast_start_time
        try:
            # Try numeric
            base_val = float(base)
            future_time_grid = np.array(
                [base_val + i for i in range(forecast_horizon)],
                dtype=float,
            )
        except Exception:
            # Leave as raw objects (e.g., strings or datetimes)
            future_time_grid = np.array(
                [base for _ in range(forecast_horizon)], dtype=object
            )

    if future_time_grid is None:
        # Fallback to a simple range [1..H]
        future_time_grid = np.arange(1, H + 1)

    future_time_grid = np.asarray(future_time_grid)
    if future_time_grid.shape[0] != H:
        raise ValueError(
            "future_time_grid must have length H. "
            f"Got len={future_time_grid.shape[0]}, H={H}"
        )

    # ------------------------------------------------------------------
    # Convert times to datetime if requested
    # ------------------------------------------------------------------
    def _to_time(values):
        if not time_as_datetime:
            return values
        return pd.to_datetime(values, format=time_format, errors="coerce")
    
    # ------------------------------------------------------------------
    # Build evaluation time grid for ALL horizons first (length H)
    # ------------------------------------------------------------------
    if train_end_time is not None:
        # Numeric arithmetic: end-(H-1), ..., end
        try:
            end_val = float(train_end_time)
            eval_time_grid_full = np.array(
                [end_val - (H - 1) + i for i in range(H)],
                dtype=float,
            )
        except Exception:
            # Fallback: same label repeated
            eval_time_grid_full = np.array(
                [train_end_time for _ in range(H)], dtype=object
            )
    else:
        # If no train_end_time provided, use future grid if it matches H,
        # otherwise fall back to 1..H.
        if future_time_grid is not None and len(future_time_grid) == H:
            eval_time_grid_full = np.asarray(future_time_grid)
        else:
            eval_time_grid_full = np.arange(1, H + 1)
    
    # ------------------------------------------------------------------
    # eval time must correspond to the chosen eval_forecast_step (s_idx)
    # ------------------------------------------------------------------
    eval_time_value = eval_time_grid_full[s_idx]
    eval_time_series = _to_time(np.full(B, eval_time_value, dtype=object))
    
    # Future times: repeat grid for each sample
    future_time_series = _to_time(np.tile(np.asarray(future_time_grid), B))
    
    # Repeat per sample for (B*H)
    eval_all_time_series = _to_time(np.tile(eval_time_grid_full, B))

    # ------------------------------------------------------------------
    # Build DataFrames
    # ------------------------------------------------------------------
    # Evaluation DataFrame (one row per sample)
    sample_idx_eval = (
        np.arange(sample_index_offset, sample_index_offset + B)
        .astype(int)
    )
    forecast_step_eval = np.full(B, eval_forecast_step, dtype=int)

    df_eval = pd.DataFrame(
        {
            "sample_idx": sample_idx_eval,
            "forecast_step": forecast_step_eval,
        }
    )

    if quantiles is not None:
        for j, q in enumerate(quantiles[: subs_eval_q.shape[1]]):
            q_name = int(round(q * 100))
            df_eval[f"{out_name}_q{q_name:d}"] = subs_eval_q[:, j]
    else:
        df_eval[f"{out_name}_pred"] = subs_eval_q

    if subs_true_eval is not None:
        df_eval[f"{out_name}_actual"] = subs_true_eval

    df_eval[t_col] = eval_time_series

    if coord_x_eval is not None:
        df_eval[x_col] = coord_x_eval
    if coord_y_eval is not None:
        df_eval[y_col] = coord_y_eval

    # Future DataFrame (one row per sample × horizon)
    N = B * H
    sample_idx_future = np.repeat(
        np.arange(sample_index_offset, sample_index_offset + B), H
    ).astype(int)
    forecast_step_future = np.tile(np.arange(1, H + 1), B).astype(int)

    df_future = pd.DataFrame(
        {
            "sample_idx": sample_idx_future,
            "forecast_step": forecast_step_future,
        }
    )

    if quantiles is not None:
        for j, q in enumerate(quantiles[: subs_future_flat.shape[1]]):
            q_name = int(round(q * 100))
            df_future[f"{out_name}_q{q_name:d}"] = subs_future_flat[:, j]
    else:
        df_future[f"{out_name}_pred"] = subs_future_flat

    df_future[t_col] = future_time_series

    if coord_x_future is not None and coord_x_future.shape[0] == N:
        df_future[x_col] = coord_x_future
    if coord_y_future is not None and coord_y_future.shape[0] == N:
        df_future[y_col] = coord_y_future

    # ------------------------------------------------------------------
    # Build "all horizons" evaluation DF (for per-year metrics)
    # ------------------------------------------------------------------
    df_eval_all = None
    if subs_true_all_flat is not None:
        # Base index columns: same structure as df_future
        df_eval_all = pd.DataFrame(
            {
                "sample_idx": sample_idx_future,      # length B*H
                "forecast_step": forecast_step_future,
                t_col: eval_all_time_series,          # 2020,2021,2022
                f"{out_name}_actual": subs_true_all_flat,
            }
        )

        # Add predictions
        if quantiles is not None:
            for j, q in enumerate(quantiles[: subs_future_flat.shape[1]]):
                q_name = int(round(q * 100))
                df_eval_all[f"{out_name}_q{q_name:d}"] = subs_future_flat[:, j]
        else:
            df_eval_all[f"{out_name}_pred"] = subs_future_flat

        # Add coords if available
        if coord_x_future is not None and coord_x_future.shape[0] == N:
            df_eval_all[x_col] = coord_x_future
        if coord_y_future is not None and coord_y_future.shape[0] == N:
            df_eval_all[y_col] = coord_y_future

    # ------------------------------------------------------------------
    # Optional: transform values between (rate <-> cumulative)
    # ------------------------------------------------------------------
    out_mode = (value_mode or "rate").lower()
    in_mode  = (input_value_mode or "rate").lower()
    
    # prediction columns to transform
    pred_cols = [
        c for c in df_future.columns
        if c.startswith(f"{out_name}_q") or c == f"{out_name}_pred"
    ]
    
    # normalize mode aliases
    CUM_MODES = {
        "cum", "cumulative", "relative_cumulative",
        "absolute_cumulative", "abs_cum", "absolute"
    }
    RATE_MODES = {"rate", "increment", "diff", "delta"}
    
    if out_mode not in (CUM_MODES | RATE_MODES):
        vlog(
            f"[format_and_forecast] Unknown value_mode={value_mode!r}; "
            "falling back to 'rate'.",
            verbose=verbose, level=1, logger=logger,
        )
        out_mode = "rate"
    
    if in_mode not in (CUM_MODES | RATE_MODES):
        vlog(
            f"[format_and_forecast] Unknown input_value_mode={input_value_mode!r}; "
            "falling back to 'rate'.",
            verbose=verbose, level=1, logger=logger,
        )
        in_mode = "rate"
    
    def _rate_by_sample(
        df: pd.DataFrame,
        cols: list[str],
        *,
        include_actual: bool,
        time_col: str,
        group_col: str = "sample_idx",
        first: str = "cum_over_dtref",
    ) -> pd.DataFrame:
        out = df.copy()
        use_cols = list(cols)
        if include_actual:
            act = f"{out_name}_actual"
            if act in out.columns:
                use_cols.append(act)
    
        # group per sample (each sample has H rows)
        for _, gidx in out.groupby(group_col, sort=False).groups.items():
            g = out.loc[gidx].sort_values("forecast_step")
            t = pd.to_numeric(g[time_col], errors="coerce").to_numpy(float)
    
            if t.size <= 1:
                continue
    
            dt = np.diff(t)
            good = np.isfinite(dt) & (dt != 0)
            dt_ref = float(np.median(dt[good])) if np.any(good) else 1.0
            if not np.isfinite(dt_ref) or dt_ref == 0:
                dt_ref = 1.0
    
            dt_i = np.diff(t)
            dt_i = np.where((~np.isfinite(dt_i)) | (dt_i == 0), dt_ref, dt_i)
    
            for ccol in use_cols:
                c = pd.to_numeric(g[ccol], errors="coerce").to_numpy(float)
                r = np.zeros_like(c, dtype=float)
    
                if first == "nan":
                    r[0] = np.nan
                elif first == "cum_over_dtref":
                    r[0] = c[0] / dt_ref
                else:
                    raise ValueError("rate_first must be 'nan' or 'cum_over_dtref'.")
    
                r[1:] = np.diff(c) / dt_i
                out.loc[g.index, ccol] = r
    
        return out
    
    # --- transform path -------------------------------------------------
    want_cum = out_mode in CUM_MODES
    have_cum = in_mode in CUM_MODES
    
    if want_cum and not have_cum:
        # rate -> cumulative (legacy behavior)
        df_future = _cum_by_sample(
            df_future, pred_cols, include_actual=False,
            target_name=target_name, verbose=verbose
        )
        if df_eval_all is not None:
            df_eval_all = _cum_by_sample(
                df_eval_all, pred_cols, include_actual=True,
                target_name=target_name, verbose=verbose
            )
            df_eval = df_eval_all[df_eval_all["forecast_step"] == eval_forecast_step].copy()
    
    elif (not want_cum) and have_cum:
        # cumulative -> rate (NEW)
        df_future = _rate_by_sample(
            df_future, pred_cols, include_actual=False,
            time_col=t_col, first=rate_first
        )
        if df_eval_all is not None:
            df_eval_all = _rate_by_sample(
                df_eval_all, pred_cols, include_actual=True,
                time_col=t_col, first=rate_first
            )
            df_eval = df_eval_all[df_eval_all["forecast_step"] == eval_forecast_step].copy()
    
    # absolute baseline shift ONLY makes sense when output is cumulative
    if want_cum and out_mode in ("absolute_cumulative", "abs_cum", "absolute"):
        if absolute_baseline is None:
            vlog(
                "[format_and_forecast] absolute cumulative requested but "
                "absolute_baseline is None; keeping relative cumulative.",
                verbose=verbose, level=1, logger=logger,
            )
        else:
            df_future = _add_baseline(
                df_future, pred_cols, include_actual=False,
                target_name=out_name, absolute_baseline=absolute_baseline
            )
            if df_eval_all is not None:
                df_eval_all = _add_baseline(
                    df_eval_all, pred_cols, include_actual=True,
                    target_name=out_name, absolute_baseline=absolute_baseline
                )
                df_eval = df_eval_all[df_eval_all["forecast_step"] == eval_forecast_step].copy()

    # ------------------------------------------------------------------
    # Drop actuals from future DF and save CSVs if requested
    # ------------------------------------------------------------------
    actual_col = f"{out_name}_actual"
    if actual_col in df_future.columns:
        df_future = df_future.drop(columns=[actual_col])

    # Decide which evaluation DataFrame to export
    df_eval_to_write = df_eval  # default fallback

    if eval_export is not None and df_eval_all is not None:
        # Case 1: string control ("all", "last", "2022", ...)
        if isinstance(eval_export, str):
            key = eval_export.lower()

            if key in ("all", "full", "horizons"):
                # Use the full multi-horizon DF (2020,2021,2022,...)
                df_eval_to_write = df_eval_all

            elif key in ("last", "single", "default"):
                # Old behaviour: single eval_forecast_step only
                df_eval_to_write = df_eval

            else:
                # Treat as a single time value for t_col, e.g. "2022"
                vals = [eval_export]
                if time_as_datetime:
                    vals = pd.to_datetime(vals, format=time_format, errors="coerce")
                df_eval_to_write = df_eval_all[df_eval_all[t_col].isin(vals)]

        else:
            # Non-string: scalar or sequence of time values (2021, [2021, 2022], ...)
            if isinstance(eval_export, Sequence):
                vals = list(eval_export)
            else:
                vals = [eval_export]

            if time_as_datetime:
                vals = pd.to_datetime(vals, format=time_format, errors="coerce")

            df_eval_to_write = df_eval_all[df_eval_all[t_col].isin(vals)]

    if csv_eval_path:
        ensure_directory_exists(os.path.dirname(csv_eval_path))
        df_eval_to_write.to_csv(csv_eval_path, index=False)
        vlog(
            f"[format_and_forecast] df_eval written to {csv_eval_path} "
            f"(rows={len(df_eval_to_write)})",
            verbose=verbose,
            level=1,
            logger=logger,
        )

    if csv_future_path:
        ensure_directory_exists(os.path.dirname(csv_future_path))
        df_future.to_csv(csv_future_path, index=False)
        vlog(
            f"[format_and_forecast] df_future written to "
            f"{csv_future_path}",
            verbose=verbose,
            level=1,
            logger = logger
        )
    # ------------------------------------------------------------------
    # Optional metrics evaluation (using df_eval)
    # ------------------------------------------------------------------
    if eval_metrics:
        # Prefer full-horizon eval DF (per-year metrics),
        # fall back to single-step df_eval if needed.
        df_for_metrics = df_eval_all if df_eval_all is not None else df_eval

        vlog(
            "[format_and_forecast] Running evaluate_forecast on "
            f"{'df_eval_all' if df_eval_all is not None else 'df_eval'}.",
            verbose=verbose,
            level=1,
            logger=logger,
        )
        
        # We pass df_eval directly; if metrics_savefile is True
        # without a path, evaluate_forecast will auto-generate
        # a filename in the current working directory.  If you
        # want it near csv_eval_path, pass an explicit path.
        
        evaluate_forecast(
            df_for_metrics,
            target_name=out_name,
            column_map=metrics_column_map,
            quantile_interval=metrics_quantile_interval,
            per_horizon=metrics_per_horizon,
            extra_metrics=metrics_extra,
            extra_metric_kwargs=metrics_extra_kwargs,
            savefile=metrics_savefile,
            save_format=metrics_save_format,
            time_as_str=metrics_time_as_str,
            verbose=verbose,
            logger=logger,
        )

    vlog(
        "[format_and_forecast] Done.",
        verbose=verbose,
        level=1,
        logger = logger
    )

    return df_eval_to_write, df_future

# Helper to cumulative-sum per sample over forecast_step
def _cum_by_sample(
    df: pd.DataFrame,
    cols: list[str],
    include_actual: bool = False, 
    target_name ="subsidence", 
    verbose: int = 2 
    ) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not {"sample_idx", "forecast_step"}.issubset(df.columns):
        vlog(
            "[format_and_forecast] Cannot apply cumulative mode: "
            "missing 'sample_idx' or 'forecast_step'.",
            verbose=verbose,
            level=1,
            logger=logger,
        )
        return df

    df_sorted = df.sort_values(
        ["sample_idx", "forecast_step"], kind="mergesort"
    ).copy()

    g = df_sorted.groupby("sample_idx", sort=False)
    df_sorted[cols] = g[cols].cumsum()
    if include_actual and f"{target_name}_actual" in df_sorted.columns:
        df_sorted[f"{target_name}_actual"] = g[
            f"{target_name}_actual"
        ].cumsum()

    return df_sorted

# Helper to add an absolute baseline per sample
def _add_baseline(
    df: pd.DataFrame,
    cols: list[str],
    target_name: str, 
    absolute_baseline: Mapping, 
    include_actual: bool = False,
    
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if absolute_baseline is None:
        return df

    if isinstance(absolute_baseline, Mapping):
        base_series = df["sample_idx"].map(absolute_baseline)
    else:
        # Scalar baseline
        base_series = pd.Series(
            float(absolute_baseline),
            index=df.index,
            dtype=float,
        )

    base_series = base_series.fillna(0.0).astype(float)
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(float) + base_series.to_numpy()
    if include_actual:
        a_col = f"{target_name}_actual"
        if a_col in df.columns:
            df[a_col] = df[a_col].astype(float) + base_series.to_numpy()
    return df

@check_empty(['y_val', 'forecast_val'])
def calibrate_forecasts(
    y_val: pd.Series,
    forecasts_val: pd.DataFrame,
    *forecasts_to_calibrate: Union[
        pd.DataFrame,
        Mapping[str, pd.DataFrame]
    ],
    quantiles: Tuple[int, ...] = (
        10, 20, 30, 40, 50, 60, 70, 80, 90
    ),
    prefix: str = 'subsidence',
    verbose: Optional[int] = None,
    _logger=None
):
    vlog(
        "Starting forecast calibration...",
        verbose,
        level=3,
        logger=_logger
    )
    calibrators = {}
    # Train a calibration model for each quantile
    for q in quantiles:
        col = f"{prefix}_q{q}"
        if col not in forecasts_val.columns:
            vlog(
                f"Skip quantile {q}: '{col}' missing",  # noqa
                verbose,
                level=2,
                logger=_logger
            )
            continue
        y_bin = (
            y_val < forecasts_val[col]
        ).astype(int)
        iso = IsotonicRegression(
            y_min=0,
            y_max=1,
            out_of_bounds="clip"
        )
        calibrators[col] = iso.fit(
            forecasts_val[col],
            y_bin
        )
    # Normalize input forecasts into dict format
    inputs = normalize_model_inputs(
        *forecasts_to_calibrate
    )
    results = {}
    vlog(
        f"Applying calibration to {len(inputs)} model(s)...",
        verbose,
        level=3,
        logger=_logger
    )
    # Apply calibrators to each DataFrame
    for name, df in inputs.items():
        df_cal = df.copy()
        for q in quantiles:
            col = f"{prefix}_q{q}"
            if col in calibrators:
                out_col = f"{col}_cal_prob"
                df_cal[out_col] = (
                    calibrators[col].predict(
                        df[col]
                    )
                )
        results[name] = df_cal
    return results, calibrators

calibrate_forecasts.__doc__ = r"""
Train and apply isotonic calibration models on forecast data.

Parameters
----------
y_val : pandas.Series
    Observed target values for calibration.
forecasts_val : pandas.DataFrame
    Validation forecasts with columns named
    `<prefix>_q{quantile}` for each quantile.
*forecasts_to_calibrate : DataFrame or dict
    One or more DataFrames (or a mapping) of new forecasts
    to calibrate using trained models.
quantiles : tuple of int, optional
    List of quantile levels (e.g., 10, 50, 90) to calibrate.
prefix : str, default 'subsidence'
    Column name prefix for quantile forecasts.
verbose : int, optional
    Verbosity level passed to vlog for logging.
_logger : Logger or callable, optional
    Sink for log messages (print or logging.Logger).

Returns
-------
results : dict
    Mapping of model names to calibrated DataFrames,
    each containing new `<prefix>_q{q}_cal_prob`
    probability columns.
calibrators : dict
    Fitted IsotonicRegression models per quantile.

Notes
-----
This function fits an isotonic regression for each
quantile on validation data and then predicts
calibrated probabilities for new forecasts. Use
`normalize_model_inputs` to handle varied input formats.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from fusionlab.utils.forecast_utils import calibrate_forecasts
>>> # Create dummy validation series
>>> idx = pd.date_range('2021-01-01', periods=50)
>>> y_val = pd.Series(
...     np.random.randn(50), index=idx
... )
>>> # Simulate validation forecasts
>>> val_df = pd.DataFrame({
...     'subsidence_q10': y_val - 0.2,
...     'subsidence_q90': y_val + 0.2
... }, index=idx)
>>> # Simulate two new forecast sets
>>> newA = val_df + 0.1
>>> newB = val_df - 0.1
>>> # Calibrate forecasts
>>> results, models = calibrate_forecasts(
...     y_val,
...     val_df,
...     newA,
...     newB,
...     quantiles=(10, 90),
...     verbose=2
... )
>>> # Access calibrated probabilities
>>> results['ModelA']['subsidence_q10_cal_prob']
>>> # Inspect fitted calibrator for q90
>>> models['subsidence_q90'].threshold_
"""

@check_empty(['models_data'])
def plot_reliability_diagram(
    models_data: Dict,
    y_true: pd.Series = None,
    prefix: str = 'subsidence',
    figsize: Tuple[int, int] = (8, 8),
    title: str = "Reliability Diagram",
    plot_style: str = 'seaborn-whitegrid',
    verbose: Optional[int] = None,
    _logger=None
):
    # Initialize logging
    vlog(
        "Starting reliability diagram plot...",
        verbose,
        level=3,
        logger=_logger
    )

    # Apply plotting style and set figure size
    style.use(plot_style)
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot perfect calibration baseline
    ax.plot(
        [0, 1], [0, 1],
        'k--',
        label='Perfect Calibration',
        lw=2.5,
        zorder=1
    )

    # Default colors and markers
    colors = [
        '#E41A1C',
        '#377EB8',
        '#4DAF4A',
        '#984EA3',
        '#FF7F00'
    ]
    markers = ['o', 's', '^', 'D', 'P']

    # Wrap simple inputs into nested structure
    first = next(iter(models_data.values()))
    if isinstance(first, pd.DataFrame):
        models_data = {
            name: {'forecasts': df}
            for name, df in models_data.items()
        }

    # Define probability intervals
    intervals = {
        0.8: (10, 90), # 80% PI uses q10 and q90
        0.6: (20, 80), # 60% PI uses q20 and q80
        0.4: (30, 70), # 40% PI uses q30 and q70
        0.2: (40, 60), # 20% PI uses q40 and q60
    }

    # Loop through each model
    for i, (model_name, data) in enumerate(models_data.items()):
        vlog(
            f"Processing model: {model_name}",
            verbose,
            level=3,
            logger=_logger
        )

        # Use precomputed points if available
        if 'observed_probs' in data and \
           'nominal_probs' in data:
            nominal = data['nominal_probs']
            observed = data['observed_probs']
        else:
            # Require forecasts key
            if 'forecasts' not in data:
                vlog(
                    f"Skip {model_name}: missing forecasts",  # noqa
                    verbose,
                    level=2,
                    logger=_logger
                )
                continue

            forecasts = data['forecasts']
            style_line = data.get(
                'style', '-'
            )
            marker = data.get(
                'marker', markers[i]
            )
            color = data.get(
                'color', colors[i]
            )

            # Compute reliability points
            nominal = sorted(intervals.keys(), reverse=True)
            observed = []

            for prob in nominal:
                low_q, up_q = intervals[prob]
                low_c = f"{prefix}_q{low_q}"
                up_c = f"{prefix}_q{up_q}"

                if low_c not in forecasts.columns or \
                   up_c not in forecasts.columns:
                    continue

                covered = (
                    (y_true >= forecasts[low_c]) &
                    (y_true <= forecasts[up_c])
                )
                observed.append(np.mean(covered))

            # Trim nominal to match observed length
            nominal = nominal[:len(observed)]

        # Plot the curve
        if nominal:
            ax.plot(
                nominal,
                observed,
                marker=marker,
                linestyle=style_line,
                color=color,
                label=model_name,
                lw=2,
                markersize=7,
                zorder=2
            )

    # Finalize labels and layout
    ax.set_xlabel(
        'Forecast Probability (Nominal)',
        fontsize=14
    )
    ax.set_ylabel(
        'Observed Frequency (Empirical)',
        fontsize=14
    )
    ax.set_title(
        title,
        fontsize=18,
        fontweight='bold'
    )
    ax.legend(
        fontsize=12,
        loc='upper left'
    )
    ax.grid(
        True,
        which='both',
        linestyle=':',
        linewidth=0.7
    )
    ax.set_aspect(
        'equal',
        adjustable='box'
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


plot_reliability_diagram.__doc__ = r"""
Plot a reliability diagram for one or multiple models.

Parameters
----------
models_data : dict
    Mapping of model names to forecast data. Each value
    can be a pandas.DataFrame or a nested dict with keys
    'forecasts', 'color', 'marker', and 'style'.
y_true : pandas.Series, optional
    Observed values for empirical coverage calculations.
    Required when forecasts need processing.
prefix : str, default 'subsidence'
    Column prefix for quantile forecast fields.
figsize : tuple of int, default (8, 8)
    Figure size (width, height) in inches.
title : str, default 'Reliability Diagram'
    Text title displayed at the top of the plot.
plot_style : str, default 'seaborn-whitegrid'
    Matplotlib style sheet name to apply.
verbose : int, optional
    Verbosity level passed to fusionlab.utils.generic_utils.vlog.
_logger : Logger or callable, optional
    Function or logger instance for internal messages.

Returns
-------
None
    Displays the calibration plot and returns nothing.

Notes
-----
This function draws a diagonal baseline (perfect
calibration) and computes empirical coverage for
probabilistic intervals using specified quantiles.
It wraps simple DataFrame inputs into the required
nested format and uses `vlog` for conditional logging.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from fusionlab.utils.forecast_utils import plot_reliability_diagram
>>> # Create dummy true time series
>>> dates = pd.date_range('2020-01-01', periods=100)
>>> y_true = pd.Series(
...     np.random.randn(100), index=dates
... )
>>> # Create forecasts for ModelA
>>> dfA = pd.DataFrame({
...     'subsidence_q10': y_true - 0.5,
...     'subsidence_q90': y_true + 0.5
... }, index=dates)
>>> # Simple usage with one model
>>> plot_reliability_diagram(
...     models_data={'ModelA': dfA},
...     y_true=y_true,
...     verbose=2
... )
>>> # Create forecasts for ModelB
... # with custom styling
>>> dfB = pd.DataFrame({
...     'subsidence_q10': y_true - 1.0,
...     'subsidence_q90': y_true + 1.0
... }, index=dates)
>>> custom_logger = print
>>> # Custom styling and logger
>>> plot_reliability_diagram(
...     models_data={
...         'ModelA': {
...             'forecasts': dfA,
...             'color': 'C0',
...             'marker': 'x'
...         },
...         'ModelB': {
...             'forecasts': dfB,
...             'color': 'C1',
...             'marker': 'o'
...         }
...     },
...     y_true=y_true,
...     verbose=4,
...     _logger=custom_logger
... )
"""

    
def _plot_reliability_diagram(
    models_data,
    y_true,
    figsize=(8, 8),
    title="Reliability Diagram",
    show_grid=True,
    plot_style='seaborn-whitegrid', 
    verbose=0, 
   ):
    """
    Plots a reliability diagram for one or multiple models.
    """
    style.use(plot_style)
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot the perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration',
            lw=2.5, zorder=1)

    # Default properties for iterating through models
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']
    markers = ['o', 's', '^', 'D', 'P']
    
    for i, (model_name, data) in enumerate(models_data.items()):
        forecasts = data['forecasts']
        line_style = data.get('style', '-')
        marker = data.get('marker', markers[i % len(markers)])
        color = data.get('color', colors[i % len(colors)])
        
        nominal_probs = np.arange(0.1, 1.0, 0.1)
        observed_probs = []

        for prob in nominal_probs:
            lower_q = int(50 - (prob * 100 / 2))
            upper_q = int(50 + (prob * 100 / 2))
            
            lower_bound_col = f'q{lower_q}'
            upper_bound_col = f'q{upper_q}'

            if ( 
                    lower_bound_col not in forecasts.columns 
                    or upper_bound_col not in forecasts.columns
                ):
                if verbose: 
                    vlog (f"Warning: Columns for {prob*100:.0f}% PI"
                          f" not found in '{model_name}'. Skipping.", level=1, 
                          verbose= verbose  
                          )
                continue

            lower_bound = forecasts[lower_bound_col]
            upper_bound = forecasts[upper_bound_col]
            
            is_covered = (y_true >= lower_bound) & (y_true <= upper_bound)
            observed_probs.append(np.mean(is_covered))
        
        # Filter nominal_probs to match available data
        plottable_nominal = nominal_probs[:len(observed_probs)]
        
        ax.plot(
            plottable_nominal,
            observed_probs,
            marker=marker,
            linestyle=line_style,
            color=color,
            label=model_name,
            lw=2,
            markersize=7,
            zorder=2
        )

    ax.set_xlabel('Forecast Probability (Nominal)', fontsize=14)
    ax.set_ylabel('Observed Frequency (Empirical)', fontsize=14)
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    
    if show_grid:
        ax.grid(True, which='both', linestyle=':', linewidth=0.7)
        
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()
    
    
@SaveFile
@check_empty(["df"])
@isdf
def calibrate_probability_forecast(
    df: pd.DataFrame,
    prob_col: str,
    actual_col: str,
    method: str = 'isotonic',
    out_col: Optional[str] = None,
    clip: bool = True,
    savefile: Optional[str]=None, 
) -> pd.DataFrame:
    """
    Calibrate a probability forecast using either isotonic regression
    or logistic (Platt scaling).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw probability forecasts and actuals.
    prob_col : str
        Column name of raw probability forecasts (values in [0,1]).
    actual_col : str
        Column name of binary outcomes (0 or 1).
    method : {'isotonic','logistic'}, default 'isotonic'
        Calibration method:
        - 'isotonic': non-parametric isotonic regression.
        - 'logistic': parametric Platt scaling with logistic regression.
    out_col : str, optional
        Name for calibrated probability column. If None, defaults to
        f"{prob_col}_calib".
    clip : bool, default True
        If True, clip output to [0,1].

    Returns
    -------
    pd.DataFrame
        A copy of df with an added calibrated probability column.
    """
    df = df.copy()
    if out_col is None:
        out_col = f"{prob_col}_calib"
    y_pred = df[prob_col].values
    y_true = df[actual_col].values

    if method == 'isotonic':
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(y_pred, y_true)
        y_cal = iso.transform(y_pred)
    elif method == 'logistic':
        lr = LogisticRegression(solver='lbfgs')
        lr.fit(y_pred.reshape(-1, 1), y_true)
        y_cal = lr.predict_proba(y_pred.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown method '{method}'")

    if clip:
        y_cal = np.clip(y_cal, 0, 1)

    df[out_col] = y_cal
    return df

@SaveFile 
@check_empty(["df"])
@isdf
def calibrate_quantile_forecasts_(
    df: pd.DataFrame,
    quantiles: Sequence[float],
    q_prefix: str,
    actual_col: str,
    method: Literal["isotonic", "logistic"] = "isotonic",
    out_prefix: str = "calib",
    grid_mode: Literal["unit", "range"] = "unit",
    grid_size: int = 1001,
    group_by: Optional[str] = None, 
    savefile:Optional[str] =None, # will handle by decorator so once the function 
    # return dataframe, decorator take it and explort to csv format instead . 
) -> pd.DataFrame:
    """
    Calibrate quantile forecasts by fitting a monotonic classifier
    at each quantile level, then inverting the CDF at the nominal q.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw quantile forecasts and actuals.
    quantiles : sequence of float
        Nominal quantile levels, e.g. [0.1,0.5,0.9].
    q_prefix : str
        Prefix for quantile columns (e.g. 'subsidence' for
        columns subsidence_q10, subsidence_q50, …).
    actual_col : str
        Column name of the true continuous outcome.
    method : {'isotonic','logistic'}, default 'isotonic'
        Calibration method to fit P(actual <= threshold).
    out_prefix : str, default 'calib'
        Prefix for the calibrated columns (e.g. 'calib_subsidence_q10').
    grid_mode : {'unit','range'}, default 'unit'
        Which domain to build the inversion grid over:
          - 'unit' → np.linspace(0,1,grid_size)
          - 'range' → np.linspace(min(raw), max(raw), grid_size)
    grid_size : int, default 1001
        Number of points in the inversion grid.
    group_by : str, optional
        If provided, calibrate separately per `df[group_by]`
        (e.g. 'forecast_step').

    Returns
    -------
    pd.DataFrame
        A copy of `df` with new columns
        `{out_prefix}_{q_prefix}_qXX` appended.
    """

    def _calibrate_block(block: pd.DataFrame) -> pd.DataFrame:
        out = block.copy()
        for q in quantiles:
            raw_col = f"{q_prefix}_q{int(q*100)}"
            cal_col = f"{out_prefix}_{raw_col}"

            # Build grid
            if grid_mode == "unit":
                grid = np.linspace(0.0, 1.0, grid_size)
            else:  # 'range'
                vals = out[raw_col].to_numpy()
                grid = np.linspace(vals.min(), vals.max(), grid_size)

            # Binary target: actual ≤ raw_pred
            y_thr = (out[actual_col] <= out[raw_col]).astype(int).values
            x_thr = out[raw_col].values

            # Fit calibrator
            if method == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(x_thr, y_thr)
                cdf_vals = iso.predict(grid)
            else:  # 'logistic'
                lr = LogisticRegression(solver="lbfgs")
                lr.fit(x_thr.reshape(-1, 1), y_thr)
                cdf_vals = lr.predict_proba(grid.reshape(-1, 1))[:, 1]

            # Invert at q
            idx = np.searchsorted(cdf_vals, q, side="left")
            idx = np.clip(idx, 0, grid_size - 1)
            out[cal_col] = grid[idx]

        return out

    # Apply either globally or per-group
    if group_by and group_by in df.columns:
        pieces = []
        for _, grp in df.groupby(group_by, sort=False):
            pieces.append(_calibrate_block(grp))
        df_out = pd.concat(pieces).sort_index()
    else:
        df_out = _calibrate_block(df)

    return df_out

@check_empty(["df"])
@isdf
def calibrate_quantile_forecasts(
    df: pd.DataFrame,
    quantiles: Sequence[float],
    q_prefix: str,
    actual_col: str,
    method: Literal["isotonic", "logistic"] = "isotonic",
    out_prefix: str = "calib",
    grid_mode: Literal["unit", "range", "unit_group", "range_group"] = "unit",
    grid_size: int = 1001,
    group_by: Optional[str] = None,
    savefile: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calibrate quantile forecasts by fitting a monotonic classifier
    at each quantile level, then inverting the CDF at the nominal q.

    grid_mode controls both domain and grouping:
      - "unit": global grid [0,1]
      - "range": global grid over [min(raw), max(raw)]
      - "unit_group": per-group grids in [0,1], groups via group_by or 'forecast_step'
      - "range_group": per-group grids over each group's raw range

    Parameters
    ----------
    df : pd.DataFrame
        Raw quantile forecasts (q_prefix_qXX) and true values.
    quantiles : sequence of float
        E.g. [0.1,0.5,0.9].
    q_prefix : str
        Prefix (e.g. "subsidence") of quantile columns.
    actual_col : str
        Column name of true continuous outcome.
    method : {'isotonic','logistic'}
    out_prefix : str
        Prefix for calibrated columns.
    grid_mode : {'unit','range','unit_group','range_group'}
    grid_size : int
    group_by : str, optional
        Column to group by if using group modes.
    savefile : str, optional
        If provided, decorator will save output.

    Returns
    -------
    pd.DataFrame
    """
    # infer grouping for _group modes
    if grid_mode.endswith("_group"):
        grp = group_by or "forecast_step"
    else:
        grp = None

    def _calibrate_block(block: pd.DataFrame) -> pd.DataFrame:
        out = block.copy()
        for q in quantiles:
            raw_col = f"{q_prefix}_q{int(q*100)}"
            cal_col = f"{out_prefix}_{raw_col}"

            # build grid domain
            if grid_mode.startswith("unit"):
                grid = np.linspace(0.0, 1.0, grid_size)
            else:  # 'range' modes
                vals = out[raw_col].to_numpy()
                grid = np.linspace(vals.min(), vals.max(), grid_size)

            # binary target
            y_thr = (out[actual_col] <= out[raw_col]).astype(int).values
            x_thr = out[raw_col].values

            # fit calibrator
            if method == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(x_thr, y_thr)
                cdf = iso.predict(grid)
            else:
                lr = LogisticRegression(solver="lbfgs")
                lr.fit(x_thr.reshape(-1,1), y_thr)
                cdf = lr.predict_proba(grid.reshape(-1,1))[:,1]

            # invert at q
            idx = np.searchsorted(cdf, q, side="left")
            idx = np.clip(idx, 0, grid_size-1)
            out[cal_col] = grid[idx]

        return out

    # dispatch global vs grouped
    if grp and grp in df.columns:
        pieces = []
        for _, g in df.groupby(grp, sort=False):
            pieces.append(_calibrate_block(g))
        df_out = pd.concat(pieces).sort_index()
    else:
        df_out = _calibrate_block(df)

    return df_out

@check_empty(["df"])
@isdf
def compute_quantile_coverage(
    df: pd.DataFrame,
    quantiles: Sequence[float],
    q_prefix: str,
    actual_col: str
) -> pd.DataFrame:
    """
    For each nominal quantile q in `quantiles`, compute the fraction
    of samples where actual <= predicted q‑quantile.
    
    Returns a DataFrame with columns ['nominal', 'empirical'].
    """
    records = []
    for q in quantiles:
        col = f"{q_prefix}_q{int(q*100)}"
        if col not in df:
            raise KeyError(f"Column '{col}' not found")
        # fraction of times actual ≤ prediction
        emp = np.mean(df[actual_col] <= df[col])
        records.append({"nominal": q, "empirical": emp})
    return pd.DataFrame.from_records(records)


@check_empty(["df"])
@isdf
def plot_reliability_diagram_in(
    coverage_df: pd.DataFrame,
    ax: plt.Axes = None,
    figsize=(6,6),
    title: str = "Reliability Diagram",
    diag_kwargs: dict = None,
    pts_kwargs: dict = None
) -> plt.Axes:
    """
    Plot nominal vs empirical probabilities.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    diag_kwargs  = diag_kwargs or {"linestyle":"--", "color":"gray"}
    pts_kwargs   = pts_kwargs  or {"marker":"o", "linewidth":1.5}
    
    # diagonal
    ax.plot([0,1],[0,1], **diag_kwargs)
    # points
    ax.plot(
        coverage_df["nominal"],
        coverage_df["empirical"],
        **pts_kwargs
    )
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel("Nominal probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(title)
    ax.grid(True)
    return ax


@check_empty(["df"])
@isdf
def pivot_forecast(
    df: pd.DataFrame,
    *,
    index_col: str = "sample_idx",
    pivot_col: Optional[str] = None,
    step_col: str = "forecast_step",
    time_col: str = "coord_t",
    value_cols: Optional[Union[str, Sequence[str]]] = None,
    spatial_cols: Optional[Sequence[str]] = None,
    aggfunc: Union[str, Callable] = "first",
    fill_value: Any = np.nan,
    sep: str = "_",
    time_formatter: Callable[[Any], str] = lambda t: pd.to_datetime(
        t).strftime("%Y-%m-%d"),
    inplace: bool = False,
    savefile: Optional[str] = None,
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Pivot a long-format forecast DataFrame into a wide one.

    This will take rows identified by `index_col` + a
    `step_col` (or datetime `time_col`) and spread each
    forecast step/time into its own set of columns for each
    value in `value_cols`, then re-attach the `spatial_cols`.

    Parameters
    ----------
    df
        Long-format forecasts. Must include `index_col` and
        at least one of `step_col` or `time_col`.
    index_col
        Column that identifies each sample (e.g. "sample_idx").
    pivot_col
        If provided, pivot on this column instead of
        auto-detecting. Must be either `step_col` or `time_col`.
    step_col
        Name of the integer 1‑based forecast step column.
    time_col
        Name of the datetime column (e.g. "coord_t").
    value_cols
        Which forecast columns to pivot (e.g. "subsidence_q50"
        or ["subsidence_q10","subsidence_q50","subsidence_q90"]).
        If None, will auto-pick all numeric columns except
        index/pivot/spatial.
    spatial_cols
        List of columns holding static spatial info
        (e.g. ["longitude","latitude"]) to join back once pivoted.
    aggfunc
        Aggregation function for pivot (default "first").
    fill_value
        What to put where a sample/step is missing (default NaN).
    sep
        Separator between value name and step/time in the new
        column names (default "_").
    time_formatter
        How to turn a datetime/timestamp into a string
        for column names (default "%Y-%m-%d").
    inplace
        If True, modifies `df` instead of copying.
    savefile
        If given, writes the resulting wide DataFrame
        to CSV at this path.
    verbose
        Passed to `vlog` for logging.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with one row per `index_col`
        and columns like `<value><sep><step>` or
        `<value><sep><formatted time>`.
    Example 
    --------
    >>> from fusionlab.utils.forecast_utils import pivot_forecast
    
    >>> dff = pivot_forecast(
    ...    df_,
    ...    index_col="sample_idx",
    ...    pivot_col="coord_t",                # ← force pivot on the datetime
    ...    value_cols=["subsidence_q10","subsidence_q50","subsidence_q90"],
    ...    spatial_cols=["longitude","latitude"],
    ...    sep="_",                            # you’ll get subsidence_q50_2022 etc.
    ...    time_formatter=lambda t: f"{t.year}",
    ...    verbose=1
    ... )
    [INFO] Pivoting on 'coord_t' for values ['subsidence_q10', 'subsidence_q50', 'subsidence_q90']
    [INFO] Joining back spatial cols ['longitude', 'latitude']
    
    >>> dff.columns
    Out[37]: 
    Index(['sample_idx', 'subsidence_q10_2022', 'subsidence_q10_2023',
           'subsidence_q10_2024', 'subsidence_q50_2022', 'subsidence_q50_2023',
           'subsidence_q50_2024', 'subsidence_q90_2022', 'subsidence_q90_2023',
           'subsidence_q90_2024', 'longitude', 'latitude'],
          dtype='object')
    """

    if not inplace:
        df = df.copy()

    if time_formatter is not (lambda t: t) and pivot_col is None:
        pivot_col = time_col

    # 1) Decide pivot column
    if pivot_col is None:
        if step_col in df.columns:
            pivot_col = step_col
        elif time_col in df.columns:
            pivot_col = time_col
        else:
            raise KeyError(
                f"Neither '{step_col}' nor '{time_col}' found; "
                "cannot auto-detect pivot column."
            )
    elif pivot_col not in df.columns:
        raise KeyError(f"pivot_col='{pivot_col}' not in DataFrame")

    # 2) Determine value columns
    if value_cols is None:
        excluded = {index_col, step_col, time_col, *(spatial_cols or ())}
        value_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in excluded
        ]
    elif isinstance(value_cols, str):
        value_cols = [value_cols]

    # 3) Pivot
    vlog(f"Pivoting on '{pivot_col}' for values {value_cols}",
         level=3, verbose=verbose)
    wide = df.pivot_table(
        index=index_col,
        columns=pivot_col,
        values=value_cols,
        aggfunc=aggfunc,
        fill_value=fill_value
    )

    # 4) Flatten column MultiIndex → single level
    new_cols = []
    for val_name, pivot_val in wide.columns:
        if pivot_col == time_col:
            pv_str = time_formatter(pivot_val)
        else:
            pv_str = str(pivot_val)
        new_cols.append(f"{val_name}{sep}{pv_str}")
    wide.columns = new_cols

    # 5) Re-attach spatial columns
    if spatial_cols:
        vlog(f"Joining back spatial cols {spatial_cols}",
             level=3, verbose=verbose)
        sp = (
            df[[index_col, *spatial_cols]]
            .drop_duplicates(index_col)
            .set_index(index_col)
        )
        wide = wide.join(sp, how="left")

    # 6) Reset index so index_col is a column again
    result = wide.reset_index()

    # 7) Optionally save to CSV
    if savefile:
        os.makedirs(os.path.dirname(savefile) or ".", exist_ok=True)
        result.to_csv(savefile, index=False)
        vlog(f"Pivoted forecasts saved to: {savefile}", 
             level=1, verbose=verbose 
             )

    return result

@check_empty(["df"])
@isdf
def add_forecast_times(
    df: pd.DataFrame,
    *,
    forecast_times: Optional[
        Sequence[Union[int, str, date, datetime, pd.Timestamp]]
    ] = None,
    start: Union[int, str, date, datetime, pd.Timestamp, None] = None,
    freq: str = "YS",
    step_col: str = "forecast_step",
    time_col: str = "coord_t",
    error: Literal["raise", "warn", "ignore"] = "raise",
    inplace: bool = False,
    savefile: Optional[str] =None, 
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Map each 1‑based forecast_step into an explicit calendar time.

    You may either:
      1. Pass `forecast_times` of length H (one per step), **or**
      2. Pass a single `start` plus a pandas‐style `freq` to generate H dates.

    If any entry in `forecast_times` is an integer of exactly 4 digits,
    it will be interpreted as January 1 of that year.
    
    Parameters
    ----------
    df : pd.DataFrame
        Long‐format forecast table. Must contain an integer
        column `step_col` with values 1..H.
    forecast_times : sequence, optional
        Explicit sequence of length H specifying the target times.
        Each entry may be:
        - int (interpreted as January 1 of that year)
        - str/`pd.Timestamp`/`datetime.date`
    start : int or str or date or Timestamp, optional
        Only used if `forecast_times` is None. The first time in the
        sequence; subsequent times will be generated via `pd.date_range`.
        If int, treated as a year at Jan 1.
    freq : str, default "YS"
        Pandas offset alias for frequency (e.g. "YS"=year start,
        "MS"=month start, "D"=day, etc.). Only used when `start` is set.
    step_col : str, default "forecast_step"
        Name of the 1‑based step index in `df`.
    time_col : str, default "coord_t"
        Name of the new column to create with mapped times.
    error : {'raise','warn','ignore'}, default 'raise'
        Policy if `df[step_col].max()` > number of provided times:
        - 'raise': throw ValueError
        - 'warn': issue warning, then still map what you can (truncate)
        - 'ignore': silently truncate to available times
    inplace : bool, default False
        If True, modify `df` in place; otherwise return a new DataFrame.
        
    savefile : str, optional
        If provided, path to CSV where the resulting DataFrame
        will be saved.
        
    verbose : int, default 0
        Passed to `vlog` for debug logging.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added column `time_col` of dtype datetime64.

    Raises
    ------
    ValueError
        If neither `forecast_times` nor `start` is provided,
        or if `error='raise'` and there aren’t enough times.

    Examples
    --------
    >>> from fusionlab.utils.forecast_utils import add_forecast_times
    >>> df = pd.DataFrame({
    ...     "sample_idx": [0]*3 + [1]*3,
    ...     "forecast_step": [1,2,3]*2
    ... })
    >>> add_forecast_times(df,
    ...     forecast_times=[2022,2023,2024])
       sample_idx  forecast_step     coord_t
    0           0              1  2022-01-01
    1           0              2  2023-01-01
    2           0              3  2024-01-01
    3           1              1  2022-01-01
    4           1              2  2023-01-01
    5           1              3  2024-01-01

    >>> # Or generate from a start + yearly freq:
    >>> add_forecast_times(df, start="2022-06-15", freq="YS")
    """
    if not inplace:
        df = df.copy()

    if step_col not in df.columns:
        raise KeyError(f"'{step_col}' not found in DataFrame")

    # 1. Build raw list of time candidates
    if forecast_times is not None:
        raw_times = list(forecast_times)
    elif start is not None:
        # interpret 4-digit int or parse otherwise
        if isinstance(start, int) and 1000 <= start <= 9999:
            start = pd.Timestamp(year=start, month=1, day=1)
        raw_times = pd.date_range(
            start=start, periods=int(df[step_col].max()), freq=freq
        ).tolist()
    else:
        raise ValueError("Must provide either `forecast_times` or `start`")

    # 2. Normalize each entry to pd.Timestamp
    times: list[pd.Timestamp] = []
    for t in raw_times:
        if isinstance(t, int) and 1000 <= t <= 9999:
            # 4‑digit → Jan 1 of that year
            times.append(pd.Timestamp(year=t, month=1, day=1))
        else:
            # let pandas parse strings/dates/timestamps
            times.append(pd.to_datetime(t))

    H = len(times)
    max_step = int(df[step_col].max())
    if max_step > H:
        msg = (
            f"Found {step_col} up to {max_step}, "
            f"but only {H} forecast times provided"
        )
        if error == "raise":
            raise ValueError(msg)
        elif error == "warn":
            warnings.warn(msg, UserWarning)
        # truncate
        H = min(H, max_step)

    vlog(f"Mapping steps 1..{H} -> times {times[:H]}", level=3, 
         verbose=verbose)
    arr = np.array(times[:H], dtype="datetime64[ns]")
    # clip to [1,H], subtract 1 for zero-based
    idx = df[step_col].to_numpy().clip(1, H).astype(int) - 1
    df[time_col] = arr[idx]

    # 3. Optionally save to CSV
    if savefile:
        os.makedirs(os.path.dirname(savefile) or ".", exist_ok=True)
        df.to_csv(savefile, index=False)
        vlog(f"Forecast times DataFrame saved to: {savefile}", 
             level=1, verbose=verbose )

    return df

@check_empty(['df']) 
def _normalize_for_pinn(
    df: pd.DataFrame,
    time_col: str,
    coord_x: str,
    coord_y: str,
    cols_to_scale: Union[List[str], str, None] = "auto",
    scale_coords: bool = True,
    verbose: int = 1, 
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    **kws
) -> Tuple[pd.DataFrame, Optional[MinMaxScaler], Optional[MinMaxScaler]]:
    r"""
    Apply Min-Max normalization to spatial–temporal coordinates and
    optionally to other numeric columns. If `cols_to_scale == "auto"`,
    automatically select numeric columns excluding categorical and
    one-hot features.

    By default, this function scales the time, longitude, and latitude
    columns (if `scale_coords=True`). Then, it either scales explicitly
    provided columns in `cols_to_scale` or automatically infers numeric
    columns (excluding coordinates if `scale_coords` is False, and
    excluding one-hot/boolean columns).

    The Min-Max scaling for a feature \(x\) is:

    .. math::
       x' = \frac{x - \min(x)}{\max(x) - \min(x)}

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least `time_col`, `lon_col`, `lat_col`.
    time_col : str
        Name of the numeric time column (e.g., year as numeric or datetime).
    coord_x : str
        Name of the longitude column.
    coord_y : str
        Name of the latitude column.
    cols_to_scale : list of str or "auto" or None, default "auto"
        - If a list of column names: scale exactly those columns.
        - If "auto": select all numeric columns, then:
          * Exclude `time_col`, `lon_col`, `lat_col` if `scale_coords=False`.
          * Exclude any columns whose values are only \{0,1\} (assumed one-hot).
        - If None: no extra columns are scaled.
    scale_coords : bool, default True
        If True, Min-Max scale `[time_col, lon_col, lat_col]`. Otherwise,
        leave these columns unchanged.
    verbose : int, default 1
        Verbosity level via `vlog` (≥2 for detailed debug info).

    Returns
    -------
    df_scaled : pd.DataFrame
        A new DataFrame with specified columns normalized.
    coord_scaler : MinMaxScaler or None
        The fitted scaler for `[time_col, lon_col, lat_col]` if
        `scale_coords=True`, else None.
    other_scaler : MinMaxScaler or None
        The fitted scaler for `cols_to_scale` (after auto-selection),
        or None if no other columns were scaled.

    Raises
    ------
    TypeError
        If `df` is not a DataFrame, or `cols_to_scale` is neither a list
        nor "auto" nor None, or if any explicitly provided column is not
        a string.
    ValueError
        If required columns (`time_col`, `lon_col`, `lat_col`) or any
        of `cols_to_scale` do not exist in `df`, or cannot be converted
        to numeric.

    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.nn.pinn.utils import normalize_for_pinn
    >>> data = {
    ...     "year_num": [0.0, 1.0, 2.0],
    ...     "lon": [100.0, 101.0, 102.0],
    ...     "lat": [30.0, 31.0, 32.0],
    ...     "feat1": [10.0, 20.0, 30.0],
    ...     "one_hot_A": [0, 1, 0]
    ... }
    >>> df = pd.DataFrame(data)
    >>> df_scaled, coord_scl, feat_scl = normalize_for_pinn(
    ...     df,
    ...     time_col="year_num",
    ...     coord_x="lon",
    ...     coord_y="lat",
    ...     cols_to_scale="auto",
    ...     scale_coords=True,
    ...     verbose=2
    ... )
    >>> # 'year_num','lon','lat','feat1' get scaled; 'one_hot_A' excluded
    >>> df_scaled["year_num"].tolist()
    [0.0, 0.5, 1.0]
    >>> df_scaled["feat1"].tolist()
    [0.0, 0.5, 1.0]

    Notes
    -----
    - When `cols_to_scale="auto"`, numeric columns with only {0,1}
      values are assumed one-hot and excluded from scaling.
    - If `scale_coords=False`, coordinate columns remain unchanged,
      and auto-selection (if used) will exclude them.
    - Returned `coord_scaler` is None if `scale_coords=False`.
      Returned `other_scaler` is None if `cols_to_scale` is None or
      results in an empty set after filtering.

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : Scales features to [0,1].
    """
    # --- Validate df ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` must be a pandas DataFrame, got "
                        f"{type(df).__name__}")

    # --- Validate core column names ---
    for name in (time_col, coord_x, coord_y):
        if not isinstance(name, str):
            raise TypeError(f"Column names must be strings, got {name}")
        if name not in df.columns:
            raise ValueError(f"Column '{name}' not found in DataFrame")

    # --- Validate cols_to_scale type ---
    if cols_to_scale is not None and cols_to_scale != "auto":
        if not isinstance(cols_to_scale, list) or not all(
            isinstance(c, str) for c in cols_to_scale
        ):
            raise TypeError("`cols_to_scale` must be a list of strings, "
                            "'auto', or None")

    # Make a copy to avoid side effects
    df_scaled = df.copy(deep=True)
    coord_scaler: Optional[MinMaxScaler] = None
    other_scaler: Optional[MinMaxScaler] = None

    # --- 1. Scale coordinates if requested ---

    if scale_coords:
        vlog("Scaling time, lon, lat columns...",
             verbose=verbose, level=2, logger=_logger
             )
        coord_cols = [time_col, coord_x, coord_y]
        for col in coord_cols:
            if not pd.api.types.is_numeric_dtype(df_scaled[col]):
                try:
                    df_scaled[col] = pd.to_numeric(df_scaled[col])
                    vlog(f"Converted '{col}' to numeric.", 
                         verbose=verbose, level=3, logger=_logger)
                except Exception as e:
                    raise ValueError(
                        f"Cannot convert '{col}' to numeric: {e}"
                    )
        coord_scaler = MinMaxScaler()
        df_scaled[coord_cols] = coord_scaler.fit_transform(
            df_scaled[coord_cols]
        )
        if verbose >= 3:
            logger.debug(
                f" coord_scaler.data_min_: {coord_scaler.data_min_}"
            )
            logger.debug(
                f" coord_scaler.data_max_: {coord_scaler.data_max_}"
            )

    # --- 2. Determine `other_cols_to_scale` ---
    if cols_to_scale == "auto":
        vlog("Auto-selecting numeric columns to scale...", 
             verbose=verbose, level=2, logger=_logger)
        # Start with all numeric columns
        numeric_cols = df_scaled.select_dtypes(
            include=[np.number]).columns.tolist()

        # Exclude coordinate columns if not scaling them 
        # if not scale_coords :
        for c in (time_col, coord_x, coord_y):
            if c in numeric_cols:
                numeric_cols.remove(c)

        # Exclude one-hot columns: numeric columns whose unique values ⊆ {0,1}
        auto_cols = []
        for c in numeric_cols:
            # uniq = pd.unique(df_scaled[c])
            vals = pd.Series(df_scaled[c]).dropna().unique()
            # if set(np.unique(uniq)) <= {0, 1}:
            if len(vals) and set(np.unique(vals)) <= {0, 1}:
                vlog(f"Excluding one-hot/boolean column '{c}' from auto-scaling.", 
                     verbose=verbose, level=3, logger=_logger)
                continue
            auto_cols.append(c)

        other_cols_to_scale = auto_cols
        vlog(f"Auto-selected columns: {other_cols_to_scale}", 
             verbose=verbose, level=2, logger=_logger)
    elif isinstance(cols_to_scale, list):
        other_cols_to_scale = cols_to_scale.copy()
    else:  # cols_to_scale is None
        other_cols_to_scale = []

    # --- 3. Scale `other_cols_to_scale` if any ---
    if other_cols_to_scale:
        vlog(f"Scaling additional columns: {other_cols_to_scale}", 
             verbose=verbose, level=2, logger=_logger)
        # Verify existence and numeric type
        valid_cols = []
        for col in other_cols_to_scale:
            if col not in df_scaled.columns:
                raise ValueError(f"Column '{col}' not found for scaling.")
            if not pd.api.types.is_numeric_dtype(df_scaled[col]):
                try:
                    df_scaled[col] = pd.to_numeric(df_scaled[col])
                    vlog(f"Converted '{col}' to numeric.", 
                         verbose=verbose, level=3, logger=_logger)
                except Exception as e:
                    raise ValueError(
                        f"Cannot convert '{col}' to numeric: {e}"
                    )
            valid_cols.append(col)

        if valid_cols:
            other_scaler = MinMaxScaler()
            df_scaled[valid_cols] = other_scaler.fit_transform(
                df_scaled[valid_cols]
            )
            if verbose >= 3:
                logger.debug(
                    f" other_scaler.data_min_: {other_scaler.data_min_}"
                )
                logger.debug(
                    f" other_scaler.data_max_: {other_scaler.data_max_}"
                )

    return df_scaled, coord_scaler, other_scaler

@check_empty(['df']) 
def normalize_for_pinn(
    df: pd.DataFrame,
    time_col: str,
    coord_x: str,
    coord_y: str,
    cols_to_scale: Union[List[str], str, None] = "auto",
    scale_coords: bool = True,
    verbose: int = 1, 
    forecast_horizon: Optional[int] = None,  
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    coord_scaler: Optional[MinMaxScaler] = None,
    fit_coord_scaler: bool = True,
    other_scaler: Optional[MinMaxScaler] = None,
    fit_other_scaler: bool = True,
    **kws
) -> Tuple[pd.DataFrame, Optional[MinMaxScaler], Optional[MinMaxScaler]]:
    """
    Apply Min-Max normalization to spatial–temporal coordinates and 
    optionally to other numeric columns. If `cols_to_scale == "auto"`, 
    automatically select numeric columns excluding categorical and one-hot 
    features.

    By default, this function scales the time, longitude, and latitude 
    columns (if `scale_coords=True`). Then, it either scales explicitly 
    provided columns in `cols_to_scale` or automatically infers numeric 
    columns (excluding coordinates if `scale_coords` is False, and excluding 
    one-hot/boolean columns).

    The Min-Max scaling for a feature \(x\) is:

    .. math::
       x' = \frac{x - \min(x)}{\max(x) - \min(x)}

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing at least `time_col`, `lon_col`, 
        and `lat_col` columns. The DataFrame should contain temporal 
        and spatial information to be scaled.
        
    time_col : str
        The name of the numeric time column (e.g., year as numeric 
        or datetime). This column will be used to adjust and scale 
        the temporal data.
        
    coord_x : str
        The name of the longitude column in the DataFrame. This column 
        will be scaled along with the latitude and time columns.
        
    coord_y : str
        The name of the latitude column in the DataFrame. This column 
        will be scaled along with the longitude and time columns.
        
    cols_to_scale : list of str or "auto" or None, default "auto"
        If a list of column names, scales exactly those columns.
        If "auto", selects all numeric columns, excluding `time_col`, 
        `lon_col`, `lat_col` if `scale_coords=False`, and excluding 
        one-hot encoded columns (values only \{0,1\}).
        If None, no extra columns are scaled.

    scale_coords : bool, default True
        If True, scales the `[time_col, lon_col, lat_col]` columns.
        If False, these columns remain unchanged.
        
    verbose : int, default 1
        Verbosity level for logging. Values higher than 1 provide 
        more detailed logging information.

    forecast_horizon : Optional[int], default None
        The number of time steps to shift the time column by. This is 
        added to the time values before scaling if provided.

    _logger : Optional[Union[logging.Logger, Callable[[str], None]]], default None
        Logger or function to handle logging messages. If None, the 
        default logging mechanism is used.

    **kws : Additional keyword arguments
        These will be passed on to any other internal function used in 
        the data processing or scaling steps.

    Returns
    -------
    df_scaled : pd.DataFrame
        A new DataFrame with the specified columns normalized.
        
    coord_scaler : MinMaxScaler or None
        The fitted scaler for the `[time_col, lon_col, lat_col]` columns 
        if `scale_coords=True`, else None.
        
    other_scaler : MinMaxScaler or None
        The fitted scaler for any additional columns that were scaled 
        (either explicitly provided or auto-selected). None if no 
        columns were scaled beyond the coordinates.

    Raises
    ------
    TypeError
        If `df` is not a DataFrame, or `cols_to_scale` is neither a list 
        nor "auto" nor None, or if any explicitly provided column is not 
        a string.
        
    ValueError
        If required columns (`time_col`, `lon_col`, `lat_col`) or any 
        of `cols_to_scale` do not exist in `df`, or cannot be converted 
        to numeric.

    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.nn.pinn.utils import normalize_for_pinn
    >>> data = {
    ...     "year_num": [0.0, 1.0, 2.0],
    ...     "lon": [100.0, 101.0, 102.0],
    ...     "lat": [30.0, 31.0, 32.0],
    ...     "feat1": [10.0, 20.0, 30.0],
    ...     "one_hot_A": [0, 1, 0]
    ... }
    >>> df = pd.DataFrame(data)
    >>> df_scaled, coord_scl, feat_scl = normalize_for_pinn(
    ...     df,
    ...     time_col="year_num",
    ...     coord_x="lon",
    ...     coord_y="lat",
    ...     cols_to_scale="auto",
    ...     scale_coords=True,
    ...     verbose=2
    ... )
    >>> # 'year_num','lon','lat','feat1' get scaled; 'one_hot_A' excluded
    >>> df_scaled["year_num"].tolist()
    [0.0, 0.5, 1.0]
    >>> df_scaled["feat1"].tolist()
    [0.0, 0.5, 1.0]

    Notes
    -----
    - When `cols_to_scale="auto"`, numeric columns with only {0,1} 
      values are assumed one-hot and excluded from scaling.
    - If `scale_coords=False`, coordinate columns remain unchanged, 
      and auto-selection (if used) will exclude them.
    - Returned `coord_scaler` is None if `scale_coords=False`. 
      Returned `other_scaler` is None if `cols_to_scale` is None or 
      results in an empty set after filtering.

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : Scales features to [0,1].
    """

    # --- Validate df ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` must be a pandas DataFrame, got "
                        f"{type(df).__name__}")

    # --- Validate core column names ---
    for name in (time_col, coord_x, coord_y):
        if not isinstance(name, str):
            raise TypeError(f"Column names must be strings, got {name}")
        if name not in df.columns:
            raise ValueError(f"Column '{name}' not found in DataFrame")

    # --- Validate cols_to_scale type ---
    if cols_to_scale is not None and cols_to_scale != "auto":
        if not isinstance(cols_to_scale, list) or not all(
            isinstance(c, str) for c in cols_to_scale
        ):
            raise TypeError("`cols_to_scale` must be a list of strings, "
                            "'auto', or None")

    # Make a copy to avoid side effects
    df_scaled = df.copy(deep=True)
    coord_scaler: Optional[MinMaxScaler] = None
    other_scaler: Optional[MinMaxScaler] = None

    # # --- 1. Adjust time before scaling ---
    if forecast_horizon is not None:
        if  pd.api.types.is_numeric_dtype(
                df_scaled[time_col]):
            # Only do this if you are NOT scaling coords, otherwise it cancels out anyway.
            if not scale_coords:
                df_scaled[time_col] = df_scaled[time_col] + float(forecast_horizon)
                vlog(f"Time column adjusted with forecast horizon: {forecast_horizon}",
                      verbose=verbose, level=4, logger=_logger)
        # # Check if time_col is integer (year)
        # if pd.api.types.is_integer_dtype(df_scaled[time_col]):
        #     # If it's an integer (year), we can simply add the forecast_horizon
        #     df_scaled[time_col] = df_scaled[time_col] + forecast_horizon
        #     vlog(f"Time column adjusted with forecast horizon: {forecast_horizon}",
        #           verbose=verbose, level=4, logger=_logger)
        elif pd.api.types.is_datetime64_any_dtype(df_scaled[time_col]):
            # If time_col is datetime, use the helper function to increment dates
            df_scaled = increment_dates_by_horizon(
                df_scaled, time_col, forecast_horizon
            )
            vlog(f"Time column adjusted with forecast horizon: {forecast_horizon}",
                  verbose=verbose, level=4, logger=_logger)

    # --- 2. Scale coordinates if requested ---
    if scale_coords:
        vlog("Scaling time, lon, lat columns...",
             verbose=verbose, level=2, logger=_logger
             )
        coord_cols = [time_col, coord_x, coord_y]
        for col in coord_cols:
            if not pd.api.types.is_numeric_dtype(df_scaled[col]):
                try:
                    df_scaled[col] = pd.to_numeric(df_scaled[col])
                    vlog(f"Converted '{col}' to numeric.", 
                         verbose=verbose, level=3, logger=_logger)
                except Exception as e:
                    raise ValueError(
                        f"Cannot convert '{col}' to numeric: {e}"
                    )
        if coord_scaler is None:
            coord_scaler = MinMaxScaler()

        # df_scaled[coord_cols] = coord_scaler.fit_transform(
        #     df_scaled[coord_cols]
        # )
        if fit_coord_scaler:
            df_scaled[coord_cols] = coord_scaler.fit_transform(
                df_scaled[coord_cols])
        else:
            if not hasattr(coord_scaler, "min_"):
                raise ValueError(
                    "fit_coord_scaler=False but `coord_scaler` is not fitted."
                )
            df_scaled[coord_cols] = coord_scaler.transform(
                df_scaled[coord_cols])
            
    # --- 3. Determine `other_cols_to_scale` ---
    if cols_to_scale == "auto":
        vlog("Auto-selecting numeric columns to scale...", 
             verbose=verbose, level=2, logger=_logger)
        # Start with all numeric columns
        numeric_cols = df_scaled.select_dtypes(
            include=[np.number]).columns.tolist()

        # Exclude coordinate columns if not scaling them 
        for c in (time_col, coord_x, coord_y):
            if c in numeric_cols:
                numeric_cols.remove(c)

        # Exclude one-hot columns: numeric columns whose unique values ⊆ {0,1}
        auto_cols = []
        for c in numeric_cols:
            uniq = pd.unique(df_scaled[c])
            if set(np.unique(uniq)) <= {0, 1}:
                vlog(f"Excluding one-hot/boolean column '{c}' from auto-scaling.", 
                     verbose=verbose, level=3, logger=_logger)
                continue
            auto_cols.append(c)

        other_cols_to_scale = auto_cols
        vlog(f"Auto-selected columns: {other_cols_to_scale}", 
             verbose=verbose, level=2, logger=_logger)
    elif isinstance(cols_to_scale, list):
        other_cols_to_scale = cols_to_scale.copy()
    else:  # cols_to_scale is None
        other_cols_to_scale = []

    # --- 4. Scale `other_cols_to_scale` if any ---
    if other_cols_to_scale:
        vlog(f"Scaling additional columns: {other_cols_to_scale}", 
             verbose=verbose, level=2, logger=_logger)
        # Verify existence and numeric type
        valid_cols = []
        for col in other_cols_to_scale:
            if col not in df_scaled.columns:
                raise ValueError(f"Column '{col}' not found for scaling.")
            if not pd.api.types.is_numeric_dtype(df_scaled[col]):
                try:
                    df_scaled[col] = pd.to_numeric(df_scaled[col])
                    vlog(f"Converted '{col}' to numeric.", 
                         verbose=verbose, level=3, logger=_logger)
                except Exception as e:
                    raise ValueError(
                        f"Cannot convert '{col}' to numeric: {e}"
                    )
            valid_cols.append(col)

        if valid_cols:
            
            if other_scaler is None:
                other_scaler = MinMaxScaler()

            # df_scaled[valid_cols] = other_scaler.fit_transform(
            #     df_scaled[valid_cols]
            # )
            if fit_other_scaler:
                df_scaled[valid_cols] = other_scaler.fit_transform(
                    df_scaled[valid_cols])
            else:
                if not hasattr(coord_scaler, "min_"):
                    raise ValueError(
                        "fit_coord_scaler=False but"
                        " `other_scaler` is not fitted."
                    )
                df_scaled[valid_cols] = other_scaler.transform(
                    df_scaled[valid_cols])
            
            vlog ( f" other_scaler.data_min_: {other_scaler.data_min_}", 
                  verbose=verbose, level =3, logger =_logger ) 
            vlog(
                f" other_scaler.data_max_: {other_scaler.data_max_}", 
                verbose=verbose, level =3 , logger =_logger, 
            )

    return df_scaled, coord_scaler, other_scaler

def increment_dates_by_horizon(
        df: pd.DataFrame, time_col: str, 
        forecast_horizon: int) -> pd.DataFrame:
    """
    Increments the values in a datetime column by the forecast horizon.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time column to be adjusted.
    time_col : str
        The name of the datetime column in the DataFrame.
    forecast_horizon : int
        The forecast horizon (number of years or time steps to add).

    Returns
    -------
    pd.DataFrame
        The DataFrame with the adjusted time column.
    """
    # Convert the time column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # Add the forecast horizon (years)
    df[time_col] = df[time_col] + pd.DateOffset(years=forecast_horizon)
    
    return df

def adjust_time_predictions(
    df: pd.DataFrame, 
    time_col: str, 
    forecast_horizon: int, 
    coord_scaler: Optional[MinMaxScaler] = None, 
    inverse_transformed: bool = False,  
    verbose: int = 1
) -> pd.DataFrame:
    """
    Adjusts time predictions by adding the forecast horizon to inverse
    normalized time. If the time column has already been inverse-transformed,
    skip the inverse transformation.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time predictions (inverse scaled).
        The time column specified by `time_col` should contain the time
        values that need to be adjusted.
        
    time_col : str
        The name of the time column in the DataFrame. This column will
        be adjusted by adding the forecast horizon.
        
    forecast_horizon : int
        The forecast horizon (e.g., number of years or time steps)
        that will be added to the time predictions. This value shifts the
        time predictions forward.
        
    coord_scaler : MinMaxScaler, optional
        The scaler that was used for the coordinates. It is necessary to
        reverse the scaling for the time column if it was previously normalized.
        If not provided, the time column should already be inverse-transformed.
        
    inverse_transformed : bool, default False
        If `True`, skips the inverse transformation of the time column and
        directly adds the forecast horizon. This is useful when the time column
        has already been inverse-transformed, and you only need to adjust the
        time by the forecast horizon.
        
    verbose : int, default 1
        Verbosity level for logging. Higher values (e.g., `verbose=2`) provide
        more detailed information about the operation.

    Returns
    -------
    pd.DataFrame
        The adjusted DataFrame with the time column updated to reflect the
        forecast horizon. The time predictions are adjusted by adding the
        `forecast_horizon` to each entry in the time column.
        
    Raises
    ------
    ValueError
        If the time column is not found in the DataFrame or if the scaler is
        not available when necessary.
        
    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> # Sample data for illustration
    >>> df = pd.DataFrame({
    >>>     'year': [0.0, 0.5, 1.0],
    >>>     'subsidence': [0.1, 0.2, 0.3]
    >>> })
    >>> scaler = MinMaxScaler()
    >>> df_scaled = df.copy()
    >>> df_scaled['year'] = scaler.fit_transform(df_scaled[['year']])
    >>> adjusted_df = adjust_time_predictions(
    >>>     df_scaled,
    >>>     time_col='year',
    >>>     forecast_horizon=4,
    >>>     coord_scaler=scaler,
    >>>     inverse_transformed=False,
    >>>     verbose=2
    >>> )
    >>> adjusted_df['year']
    [0.0, 0.5, 1.0] -> After adjustment, will be shifted to the future.
    
    Notes
    -----
    - The time column must be in a normalized scale if not already
      inverse-transformed.
    - If `inverse_transformed=True`, the time values will directly be adjusted
      by the `forecast_horizon` without applying the inverse transformation.
    - The forecast horizon is added directly to the time values after the
      necessary inverse transformation (if applicable).
    
    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : Scales features to [0,1].
    """
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame.")

    if coord_scaler is None and not inverse_transformed:
        raise ValueError("coord_scaler is required unless `inverse_transformed` is True.")

    # Apply inverse scaling to the time predictions if they haven't been inverse-transformed yet
    time_predictions = df[time_col].values

    if not inverse_transformed:
        # Revert the time predictions from the normalized scale to the original scale
        time_predictions = coord_scaler.inverse_transform(
            time_predictions.reshape(-1, 1)).flatten()

    # Add the forecast horizon to the time predictions
    adjusted_time_predictions = time_predictions + forecast_horizon

    # Update the DataFrame with the adjusted time predictions
    df[time_col] = adjusted_time_predictions

    if verbose >= 1:
        logger.info(f"Time predictions adjusted by {forecast_horizon} years.")

    return df


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

def get_step_names(
    forecast_steps: Iterable[int],
    step_names:Optional[
        Union[ Mapping[Any, str], Sequence[str], None]] = None,
    default_name: str = "",
) -> Dict[int, str]:
    
    r"""
    Build a *step → label* mapping for multi‑horizon plots.

    The helper reconciles an integer list ``forecast_steps`` with an
    optional *alias* container (*dict* or *sequence*) and returns a
    dictionary whose keys are the integer steps and whose values are
    human‑readable labels.

    Matching is **case‑insensitive** and tolerant to common
    delimiters—e.g. ``"Step 1"``, ``"step‑1"``, or ``"forecast step 1"``
    will all map to integer step ``1``.

    Parameters
    ----------
    forecast_steps : Iterable[int]
        Ordered steps, e.g. ``[1, 2, 3]``.
    step_names : dict | list | tuple | None, default=None
        Custom labels.  Accepted forms

        * **dict** – keys may be ``int`` or *any* string
          representation of the step.
        * **sequence** – positional, where the *k*‑th element labels
          step ``k+1``.
        * **None** – no custom mapping.
    default_name : str, default=""
        Fallback label for steps missing from *step_names*.  If
        empty, the step number itself is used (as a string).

    Returns
    -------
    dict[int, str]
        Mapping ``{step : label}`` for every element of
        *forecast_steps*.

    Notes
    -----
    * Dictionary keys are normalised with
      ``int(re.sub(r"[^0-9]", "", str(key)))`` before matching.
    * Duplicate keys in *step_names* are resolved by **last‐one wins**
      semantics.

    Examples
    --------
    >>> from fusionlab.utils.forecast_utils import get_step_names
    >>> get_step_names(
    ...     forecast_steps=[1, 2, 3],
    ...     step_names={"1": "Year 2021", 2: "2022", "step 3": "2023"},
    ... )
    {1: 'Year 2021', 2: '2022', 3: '2023'}

    >>> get_step_names(
    ...     forecast_steps=[1, 2, 3, 4],
    ...     step_names={"1": "2021", "2": "2022"},
    ... )
    {1: '2021', 2: '2022', 3: '3', 4: '4'}

    >>> get_step_names(
    ...     [1, 2, 3, 4],
    ...     step_names=None,
    ...     default_name="step with no name",
    ... )
    {1: 'step with no name', 2: 'step with no name',
     3: 'step with no name', 4: 'step with no name'}

    See Also
    --------
    fusionlab.utils.data_utils.widen_temporal_columns :
        Converts long format to wide; often used with
        *forecast_steps* when plotting.
    """
    # Ensure we have a concrete list to preserve order and allow
    # multiple passes.
    forecast_steps = columns_manager( forecast_steps , empty_as_none= False)
    steps: List[int] = [int(s) for s in forecast_steps]
    lookup: Dict[int, str] = {}

    if step_names is None:
        pass # remain empty
    elif isinstance(step_names, Mapping):
        for k, v in step_names.items():
            idx = _to_int_key(k) 
            # Skip keys that cannot be coerced to int (e.g. None, dict)
            if idx is not None:
                lookup[idx] = str(v)
    elif isinstance(step_names, Sequence) and not isinstance(
            step_names, (str, bytes)):
        for idx, v in enumerate(step_names, start=1):
            lookup[idx] = str(v)
    else:
        raise TypeError(
            "`step_names` must be a mapping, a sequence, or None "
            f"(got {type(step_names).__name__})."
        )

    # Build the final mapping, applying defaults where necessary.
    result: Dict[int, str] = {}
    for step in steps:
        if step in lookup:
            result[step] = lookup[step]
        elif default_name:
            result[step] = default_name
        else:
            result[step] = str(step)
    return result

def _to_int_key(key: Any) -> int | None:
    """Try to coerce a mapping key to int by stripping non‑digits."""
    if isinstance(key, int):
        return key
    digits = "".join(_DIGIT_RE.findall(str(key)))
    return int(digits) if digits else None


@isdf
def format_forecast_dataframe(
    df: pd.DataFrame,
    to_wide: bool = True,
    time_col: str = 'coord_t',
    spatial_cols: Tuple[str]=('coord_x', 'coord_y'), 
    value_prefixes: Optional[List[str]] = None,
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    **pivot_kwargs
) -> Union[pd.DataFrame, str]:
    """Auto-detects DataFrame format and conditionally pivots to wide format.

    This function serves as a smart wrapper. It first determines if the
    input DataFrame is in a 'long' or 'wide' forecast format based on
    its column structure. If `to_wide` is True and the format is
    'long', it calls :func:`pivot_forecast_dataframe` to perform the
    transformation.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to check and potentially transform.
    to_wide : bool, default True
        - If ``True``, the function's goal is to return a wide-format
          DataFrame. It will pivot a long-format frame or return a
          wide-format frame as is.
        - If ``False``, the function only performs detection and
          returns a string ('wide', 'long', or 'unknown').
    time_col : str, default 'coord_t'
        The name of the column that indicates the time step. Its
        presence is a primary indicator of a long-format DataFrame.
    value_prefixes : list of str, optional
        A list of prefixes for the value columns (e.g., ['subsidence',
        'GWL']). If ``None``, the function will attempt to infer them
        from column names that do not match common ID columns.
    **pivot_kwargs
        Additional keyword arguments to pass down to the
        :func:`pivot_forecast_dataframe` function if it is called.
        Common arguments include `id_vars`, `static_actuals_cols`,
        `verbose`, etc.

    Returns
    -------
    pd.DataFrame or str
        - If `to_wide` is ``True``, returns the (potentially pivoted)
          wide-format ``pd.DataFrame``.
        - If `to_wide` is ``False``, returns a string: 'wide', 'long',
          or 'unknown'.

    See Also
    --------
    pivot_forecast_dataframe : The underlying function that performs
                               the pivot operation.

    Examples
    --------
    >>> # df_long is a typical long-format forecast output
    >>> df_long.columns
    Index(['sample_idx', 'forecast_step', 'coord_t', 'coord_x', ...])
    >>> # Detect format
    >>> format_str = format_forecast_dataframe(df_long, to_wide=False)
    >>> print(format_str)
    'long'
    >>>
    >>> # Convert to wide format
    >>> df_wide = format_forecast_dataframe(
    ...     df_long,
    ...     to_wide=True,
    ...     id_vars=['sample_idx', 'coord_x', 'coord_y'],
    ...     value_prefixes=['subsidence', 'GWL'],
    ...     static_actuals_cols=['subsidence_actual']
    ... )
    >>> # print(df_wide.columns)
    # Index(['sample_idx', 'coord_x', 'coord_y', 'subsidence_actual',
    #        'GWL_2018_q50', ...], dtype='object')
    """
    # --- Format Detection Logic ---
    # Heuristic 1: If time_col exists, it's very likely 'long' format.
    is_long_format = time_col in df.columns

    # Heuristic 2: Check for wide-format columns like 'prefix_YYYY_suffix'
    # Use a regex to look for (prefix)_(4-digit year)_...
    _spatial_cols = columns_manager(spatial_cols, empty_as_none= False)
    
    if spatial_cols: 
        spatial_cols = columns_manager (spatial_cols)
        check_spatial_columns(df, spatial_cols=spatial_cols)
        # coord_x, coord_y = spatial_cols 
        
    if value_prefixes is None:
        # Auto-infer prefixes if not provided
        # Exclude common ID columns
        non_value_cols = {
            'sample_idx', *_spatial_cols,  'forecast_step', time_col
            }
        value_prefixes = sorted(list(set(
            [c.split('_')[0] for c in df.columns
             if c not in non_value_cols]
        )))

    wide_col_pattern = re.compile(
        r'(' + '|'.join(re.escape(p) for p in value_prefixes) + 
        r')_(\d{4})_?.*'
    )
    has_wide_columns = any(wide_col_pattern.match(col) for col in df.columns)

    detected_format = 'unknown'
    if is_long_format:
        detected_format = 'long'
    elif has_wide_columns:
        detected_format = 'wide'
    
    verbose = pivot_kwargs.get('verbose', 0)
    vlog(f"Auto-detected DataFrame format: '{detected_format}'",
         level=1, verbose=verbose, logger = _logger 
         )

    # --- Action based on mode ---
    if to_wide:
        if detected_format == 'long':
            vlog("`to_wide` is True and format is 'long'. "
                 "Pivoting DataFrame...", level=1, verbose=verbose, 
                 logger = _logger )
            # Pass necessary args to the pivot function
            pivot_args = {
                'time_col': time_col,
                'value_prefixes': value_prefixes,
                **pivot_kwargs # Pass through other args
            }
            if 'id_vars' not in pivot_args:
                # Provide a sensible default for id_vars if not given
                pivot_args['id_vars'] = [
                    c for c in ['sample_idx', *_spatial_cols]
                    if c in df.columns
                ]
                vlog(f"Using default id_vars: {pivot_args['id_vars']}",
                     level=2, verbose=verbose, logger = _logger 
                     )

            return pivot_forecast_dataframe(
                df.copy(), _logger =_logger,  
                **pivot_args)
        
        elif detected_format == 'wide':
            vlog("`to_wide` is True but DataFrame is already in wide "
                 "format. Returning as is.", level=1, verbose=verbose, 
                 logger = _logger 
                 )
            return df
        else: # 'unknown'
            vlog("Warning: DataFrame format is 'unknown'. "
                 "Cannot pivot. Returning original DataFrame.",
                 level=1, verbose=verbose, logger = _logger 
                 )
            return df
    else: # to_wide is False
        return detected_format

@isdf
def get_value_prefixes(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None, 
    spatial_cols: Tuple[str, str] = ('coord_x', 'coord_y'), 
    time_col: str ='coord_t'
) -> List[str]:
    """
    Automatically detects the prefixes of value columns from a DataFrame.

    This utility inspects the column names to infer the base names of
    the metrics being forecasted (e.g., 'subsidence', 'GWL'),
    excluding common ID and coordinate columns. It works with both
    long and wide format forecast DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which to detect value prefixes.
    exclude_cols : list of str, optional
        A list of columns to explicitly ignore during detection. If
        None, a default list of common ID/coordinate columns is
        used (e.g., 'sample_idx', 'coord_x', 'coord_t', etc.).

    Returns
    -------
    list of str
        A sorted list of unique prefixes found in the column names.

    Examples
    --------
    >>> from fusionlab.utils.data_utils import get_values_prefixes
    >>> # For a long-format DataFrame
    >>> long_cols = ['sample_idx', 'coord_t', 'subsidence_q50', 'GWL_q50']
    >>> df_long = pd.DataFrame(columns=long_cols)
    >>> get_value_prefixes(df_long)
    ['GWL', 'subsidence']

    >>> # For a wide-format DataFrame
    >>> wide_cols = ['sample_idx', 'coord_x', 'subsidence_2022_q90', 'GWL_2022_q50']
    >>> df_wide = pd.DataFrame(columns=wide_cols)
    >>> get_value_prefixes(df_wide)
    ['GWL', 'subsidence']
    """
    if exclude_cols is None:
        # Default set of columns that are not value columns
        exclude_cols = {
            'sample_idx', 'forecast_step', time_col,
            *spatial_cols
        }
    else:
        exclude_cols = set(exclude_cols)

    prefixes = set()
    for col in df.columns:
        if col in exclude_cols:
            continue
        # The prefix is assumed to be the part before the first underscore
        prefix = col.split('_')[0]
        prefixes.add(prefix)
    
    return sorted(list(prefixes))

def get_test_data_from(
    df: pd.DataFrame, 
    time_col: str,
    time_steps: int,  
    train_end_year: Optional[int] = None,  
    forecast_horizon: Optional[int] = 1,  
    strategy: str = 'onwards',  
    objective: str = "pure" , # for forecasting
    verbose: int = 1, 
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
) -> pd.DataFrame:
    r"""
    Prepares the test data for forecasting by ensuring there is enough 
    future data.
    It adjusts the start and end years based on the forecast horizon
    and strategy provided.
    
    Parameters
    ----------
    df : pd.DataFrame
        The scaled dataframe containing the time series data.
    time_col : str
        The column name for time in the dataframe (e.g., year).
    time_steps : int
        The number of steps to look back for generating sequences.
    train_end_year : int, optional
        The last year used for training. If not provided, defaults to the last year 
        in the dataset.
    forecast_horizon : int, optional
        The forecast horizon, i.e., the number of years to predict. Default is 1.
    strategy : {'onwards', 'lookback'}, default 'onwards'
        - 'onwards': Start the test data from `train_end_year + time_steps`.
        - 'lookback': Use the available data until `train_end_year`.
    verbose : int, optional
        Verbosity level for logging (default 1).
    _logger : logging.Logger or Callable, optional
        Logger or callable for logging the messages.

    Returns
    -------
    pd.DataFrame
        The prepared test data, based on the forecast horizon
        and selected strategy.
    
    Notes
    -----
    - If `train_end_year` is not provided, it is determined automatically 
      from the latest year in the `time_col`.
    - The `strategy` parameter defines how the test data is selected:
      * `onwards`: Select data starting from `train_end_year + time_steps`.
      * `lookback`: Select data until `train_end_year`.
    - If there isn't enough data for the `forecast_horizon`, the function will 
      adjust and either fetch earlier data or use the `lookback` strategy.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    >>>     'year': [2015, 2016, 2017, 2018, 2019, 2020],
    >>>     'subsidence': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    >>> })
    >>> test_data = get_test_data_from(
    >>>     df=df, 
    >>>     time_col='year', 
    >>>     time_steps=3, 
    >>>     forecast_horizon=2, 
    >>>     strategy='onwards', 
    >>>     verbose=2
    >>> )
    >>> print(test_data)
    """
    def _v(msg, lvl): 
        vlog(msg, verbose=verbose, level=lvl, logger=_logger)
    
    # Determine the last available year if not explicitly given
    if train_end_year is None:
        if pd.api.types.is_numeric_dtype(df[time_col]):
            train_end_year = df[time_col].max()  # Use the last numeric year
        elif pd.api.types.is_datetime64_any_dtype(df[time_col]):
            # Handle various datetime formats
            if df[time_col].dt.is_year_end.any():
                train_end_year = df[time_col].dt.year.max()  # Year format
            elif df[time_col].dt.is_month_end.any():
                train_end_year = df[time_col].dt.to_period('M').max().year  # Month format
            elif df[time_col].dt.week.any():
                train_end_year = df[time_col].dt.to_period('W').max().year  # Week format
            else:
                # Default to full datetime handling
                train_end_year = df[time_col].dt.year.max()

        if verbose >= 1:
            _v("Train end year not specified."
               f" Using last available year: {train_end_year}",
               lvl=1)
    
    # Adjust forecast end year based on the datetime frequency
    forecast_end_year = train_end_year + forecast_horizon
    forecast_start_year = train_end_year - forecast_horizon
    if objective =="forecasting": 
        forecast_end_year= train_end_year + time_steps
        forecast_start_year = train_end_year - time_steps
        
    if verbose >= 2:
        _v(f"Forecast start year: {forecast_start_year},"
           f" Forecast end year: {forecast_end_year}", lvl=2)
    
    # Handle data based on the specified strategy
    if strategy == 'onwards':
        # Get the test data starting from the forecast start year
        # (train_end_year + time_steps)
        test_data = df[df[time_col] >= forecast_end_year]
    elif strategy == 'lookback':
        # If 'lookback', select data that ends at `train_end_year`
        #(train_end_year - time_steps)
        if objective =='forecasting':
            test_data = df[df[time_col] >= forecast_start_year]
        else:
            test_data = df[df[time_col] > forecast_start_year]
    else:
        raise ValueError(f"Invalid strategy '{strategy}'."
                         " Choose either 'onwards' or 'lookback'.")
    
    # Check if enough data is available for the forecast horizon
    if test_data.empty or len(test_data[time_col].unique()) < forecast_horizon:
        if verbose >= 1:
            _v(f"Not enough data for the forecast horizon"
               f" ({forecast_horizon} years). Adjusting test data.", 
               lvl=1)
        
        # Fall back to using a lookback strategy if there
        # isn't enough data for the forecast horizon
        # (train_end_year - time_steps)
        if objective =='forecasting': 
            test_data = df[df[time_col] >=forecast_start_year]
        else:
            test_data = df[df[time_col] > forecast_start_year]
        
        if verbose >= 1:
            _v(f"Using data from {forecast_start_year} onwards.", lvl=1)
    
    # Log the years included in the test data
    if verbose >= 1:
        _v(f"Test data years: {test_data[time_col].unique()}", lvl=2)
    
    # Return the test data
    return test_data


@check_empty(['data']) 
@is_data_readable 
def pivot_forecast_dataframe(
    data: pd.DataFrame,
    id_vars: List[str],
    time_col: str,
    value_prefixes: List[str],
    static_actuals_cols: Optional[List[str]] = None,
    time_col_is_float_year: Union[bool, str] = 'auto',
    round_time_col: bool = False,
    verbose: int = 0,
    savefile: Optional[str] = None, 
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    **kws
) -> pd.DataFrame:
    """Transforms a long-format forecast DataFrame to a wide format.
    
    This utility reshapes time series prediction data from a "long"
    format, where each row represents a single time step for a given
    sample, to a "wide" format, where each row represents a single
    sample and columns correspond to values at different time steps.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input long-format DataFrame. It must contain the columns
        specified in `id_vars` and `time_col`, as well as value
        columns that start with the strings in `value_prefixes`.
    
    id_vars : list of str
        A list of column names that uniquely identify each sample
        or group. These columns will be preserved in the wide-format
        output. For example: ``['sample_idx', 'coord_x', 'coord_y']``.
    
    time_col : str
        The name of the column that represents the time step or year
        of the forecast (e.g., 'coord_t' or 'forecast_step').
        This column's values will become part of the new column names.
    
    value_prefixes : list of str
        A list of prefixes for the value columns that need to be
        pivoted. The function identifies columns starting with
        these prefixes. For instance, ``['subsidence', 'GWL']``
        would match 'subsidence_q10', 'GWL_q50', etc.
    
    static_actuals_cols : list of str, optional
        A list of columns containing static "actual" or ground truth
        values for each sample. These values are assumed to be
        constant for each unique `sample_idx` and are merged back
        into the wide DataFrame after pivoting.
        Example: ``['subsidence_actual']``.
    
    time_col_is_float_year : bool or 'auto', default 'auto'
        Controls how the `time_col` values are formatted into new
        column names.
        - If ``'auto'``, automatically detects if `time_col` has a
          float dtype.
        - If ``True``, treats `time_col` values (e.g., 2018.0) as
          years and converts them to integer strings ('2018').
        - If ``False``, uses the string representation of the value
          as is.
    
    round_time_col : bool, default False
        If ``True`` and `time_col` is a float type, its values will
        be rounded to the nearest integer before being used in
        column names. This is useful for cleaning up float years
        (e.g., 2018.0001 -> 2018).
    
    verbose : int, default 0
        Controls the verbosity of logging messages. `0` is silent.
        Higher values print more details about the process.
    
    savefile : str, optional
        If a file path is provided, the final wide-format DataFrame
        will be saved as a CSV file to that location.
    
    Returns
    -------
    pd.DataFrame
        A wide-format DataFrame with one row per unique combination
        of `id_vars`. New columns are created in the format
        `{prefix}_{time_str}{_suffix}` (e.g., 'subsidence_2018_q10').
    
    See Also
    --------
    pandas.pivot_table : The core function used for reshaping data.
    pandas.merge : Used to re-join static columns after pivoting.
    
    Notes
    -----
    - The combination of columns in `id_vars` and `time_col` must
      uniquely identify each row in `df_long` for the pivot to
      succeed without data loss.
    - If using `static_actuals_cols`, the `id_vars` list must
      contain 'sample_idx' to correctly merge the static data back.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.utils.data_utils import pivot_forecast_dataframe
    >>> data = {
    ...     'sample_idx':      [0, 0, 1, 1],
    ...     'coord_t':         [2018.0, 2019.0, 2018.0, 2019.0],
    ...     'coord_x':         [0.1, 0.1, 0.5, 0.5],
    ...     'coord_y':         [0.2, 0.2, 0.6, 0.6],
    ...     'subsidence_q50':  [-8, -9, -13, -14],
    ...     'subsidence_actual': [-8.5, -8.5, -13.2, -13.2],
    ...     'GWL_q50':         [1.2, 1.3, 2.2, 2.3],
    ... }
    >>> df_long_example = pd.DataFrame(data)
    >>> df_wide = pivot_forecast_dataframe(
    ...     data=df_long_example,
    ...     id_vars=['sample_idx', 'coord_x', 'coord_y'],
    ...     time_col='coord_t',
    ...     value_prefixes=['subsidence', 'GWL'],
    ...     static_actuals_cols=['subsidence_actual'],
    ...     verbose=0
    ... )
    >>> print(df_wide.columns)
    Index(['sample_idx', 'coord_x', 'coord_y', 'subsidence_actual',
           'GWL_2018_q50', 'GWL_2019_q50', 'subsidence_2018_q50',
           'subsidence_2019_q50'],
          dtype='object')
    """
    is_frame(data, df_only= True) 
    
    df_processed = data.copy()
    
    vlog(f"Starting pivot operation. Initial shape: {df_processed.shape}",
         level=2, verbose=verbose, logger = _logger )

    if not isinstance(df_processed, pd.DataFrame):
        raise TypeError(
            "`df_long` must be a pandas DataFrame."
        )

    value_cols_to_pivot = [
        col for col in df_processed.columns
        if any(col.startswith(prefix) for prefix in value_prefixes)
    ]
    
    required_cols = id_vars + [time_col] + value_cols_to_pivot
    missing_cols = [
        col for col in required_cols if col not in df_processed.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in DataFrame: {missing_cols}"
        )

    # Determine if the time column should be treated as a float year
    is_float_year = False
    if time_col_is_float_year == 'auto':
        if pd.api.types.is_float_dtype(df_processed[time_col]):
            is_float_year = True
            vlog(f"'{time_col}' auto-detected as float year.",
                 level=2, verbose=verbose, logger = _logger 
                 )
    elif time_col_is_float_year is True:
        is_float_year = True
    
    # Round the time column before pivoting if requested
    if round_time_col and is_float_year:
        vlog(f"Rounding time column '{time_col}'.",
             level=1, verbose=verbose, logger = _logger 
             )
        df_processed[time_col] = df_processed[time_col].round().astype(int)
        # After rounding, it's no longer a float year
        is_float_year = False
    elif round_time_col and not is_float_year:
        vlog(f"Warning: `round_time_col` is True but '{time_col}' "
             "is not a float year. Skipping rounding.",
             level=1, verbose=verbose, logger = _logger  )

    static_df = None
    if static_actuals_cols:
        if 'sample_idx' not in df_processed.columns:
            raise ValueError(
                "'sample_idx' must be in df_long to handle "
                "static_actuals_cols."
            )
        vlog(f"Extracting static columns: {static_actuals_cols}",
             level=2, verbose=verbose, logger = _logger )
        static_df = df_processed[
            ['sample_idx'] + static_actuals_cols
        ].drop_duplicates(
            subset=['sample_idx']
        ).set_index('sample_idx')

    pivot_index = list(set(id_vars) & set(df_processed.columns))
    pivot_columns = [time_col]
    pivot_values = value_cols_to_pivot

    vlog(f"Pivoting data with index={pivot_index}, "
         f"columns='{time_col}'.", level=2, verbose=verbose, 
         logger = _logger  )
    try:
        df_pivoted = df_processed.pivot_table(
            index=pivot_index,
            columns=pivot_columns,
            values=pivot_values,
            aggfunc='first'
        )
    except Exception as e:
        raise RuntimeError(
            "Pandas pivot_table failed. Check if `id_vars` and "
            f"`time_col` uniquely identify rows. Error: {e}"
        )

    vlog("Flattening pivoted column names.", level=2, verbose=verbose, 
         logger = _logger  )
    new_columns = []
    for value_col, time_val in df_pivoted.columns:
        parts = value_col.split('_', 1)
        prefix = parts[0]
        suffix = f"_{parts[1]}" if len(parts) > 1 else ""
        
        time_str = str(time_val)
        if is_float_year:
            try:
                time_str = str(int(time_val))
            except (ValueError, TypeError):
                pass 
        
        new_col_name = f"{prefix}_{time_str}{suffix}"
        new_columns.append(new_col_name)

    df_pivoted.columns = new_columns
    df_pivoted = df_pivoted.reset_index()

    if static_df is not None:
        vlog("Merging static columns back into the wide DataFrame.",
             level=2, verbose=verbose, logger = _logger  )
        df_wide = pd.merge(
            df_pivoted, static_df, on='sample_idx', how='left'
        )
        cols_order = id_vars + static_actuals_cols + [
            c for c in df_wide.columns 
            if c not in id_vars + static_actuals_cols
        ]
        df_wide = df_wide[cols_order]
    else:
        df_wide = df_pivoted
    
    vlog(f"Pivot complete. Final shape: {df_wide.shape}",
         level=1, verbose=verbose, logger = _logger 
         )

    if savefile:
        try:
            vlog(f"Saving DataFrame to '{savefile}'.",
                 level=1, verbose=verbose, logger = _logger )
            save_dir = os.path.dirname(savefile)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            df_wide.to_csv(savefile, index=False)
            vlog("Save successful.", level=2, verbose=verbose, 
                 logger = _logger 
                 )
        except Exception as e:
            logger.error(f"Failed to save file to '{savefile}': {e}")

    return df_wide

@isdf
def get_value_prefixes_in(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None
) -> List[str]:
    """
    Automatically detects the prefixes of value columns from a DataFrame.
    (This is a dependency for the function below)
    """
    if exclude_cols is None:
        exclude_cols = {
            'sample_idx', 'forecast_step', 'coord_t',
            'coord_x', 'coord_y'
        }
    else:
        exclude_cols = set(exclude_cols)

    prefixes = set()
    for col in df.columns:
        if col in exclude_cols:
            continue
        # The prefix is assumed to be the
        # part before the first underscore
        prefix = col.split('_')[0]
        prefixes.add(prefix)
    
    return sorted(list(prefixes))

@isdf 
def detect_forecast_type(
    df: pd.DataFrame,
    value_prefixes: Optional[List[str]] = None, 
    
) -> str:
    """
    Auto-detects whether a DataFrame contains deterministic or
    quantile forecasts, supporting both long and wide formats.

    This utility inspects column names to determine the nature of the
    predictions.

    - It identifies a 'quantile' forecast if it finds columns
      containing a ``_qXX`` pattern (e.g., 'subsidence_q10',
      'GWL_2022_q50').
    - It identifies a 'deterministic' forecast if no quantile columns
      are found, but columns ending in ``_pred``, `_actual`, or
      matching a base prefix exist (e.g., 'subsidence_pred',
      'subsidence_2022_actual', 'GWL').

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    value_prefixes : list of str, optional
        A list of value prefixes (e.g., ['subsidence', 'GWL']) to
        focus the search on. If ``None``, prefixes are inferred
        from column names.

    Returns
    -------
    str
        One of 'quantile', 'deterministic', or 'unknown'.

    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.utils.forecast_utils import detect_forecast_type
    >>> # Long format quantile
    >>> df_quant_long = pd.DataFrame(columns=['subsidence_q50', 'GWL_q90'])
    >>> detect_forecast_type(df_quant_long)
    'quantile'
    
    >>> # Wide format quantile
    >>> df_quant_wide = pd.DataFrame(columns=['subsidence_2022_q50'])
    >>> detect_forecast_type(df_quant_wide)
    'quantile'

    >>> # Deterministic forecast
    >>> df_determ = pd.DataFrame(columns=['subsidence_pred', 'GWL'])
    >>> detect_forecast_type(df_determ)
    'deterministic'
    """
    if value_prefixes is None:
        # Auto-detect prefixes if not provided.
        value_prefixes = get_value_prefixes(df)

    if not value_prefixes:
        return 'unknown'
    
    # Updated regex to handle both long (_q50) and wide (_YYYY_q50) formats
    # It looks for '_q' followed by one or more digits anywhere in the string.
    quantile_pattern = re.compile(r'_q\d+')
    # Regex to find deterministic suffixes like _pred or _actual at the end
    pred_pattern = re.compile(r'_(pred|actual)$')
    
    has_quantile = False
    has_pred_or_actual = False
    has_bare_prefix = False

    for col in df.columns:
        if quantile_pattern.search(col):
            # Check if it belongs to one of our prefixes
            for prefix in value_prefixes:
                if col.startswith(prefix):
                    has_quantile = True
                    break
        if has_quantile:
            break
    
    if has_quantile:
        return 'quantile'

    # If no quantiles, check for deterministic patterns
    for col in df.columns:
        for prefix in value_prefixes:
            # Check for exact prefix match (e.g., 'subsidence')
            if col == prefix:
                has_bare_prefix = True
            # Check for patterns like 'subsidence_pred', 'GWL_2022_actual'
            elif col.startswith(prefix) and pred_pattern.search(col):
                has_pred_or_actual = True
        
        if has_pred_or_actual or has_bare_prefix:
            break
            
    if has_pred_or_actual or has_bare_prefix:
        return 'deterministic'

    # If neither format is detected, return 'unknown'.
    return 'unknown'



# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

"""
Forecast utilities.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import json
import logging

from typing import ( 
    Dict, 
    Union, 
    Any, 
    Optional,
    Tuple , 
    Literal, 
)
import numpy as np 
import pandas as pd 

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..core.checks import check_empty 
from ..core.io import SaveFile
from ..decorators import isdf 
from .generic_utils import vlog, normalize_model_inputs


def _infer_quantile_map(
    df: pd.DataFrame,
    *,
    target_name: str,
    column_map: Optional[Mapping[str, Any]] = None,
) -> Dict[float, str]:
    column_map = dict(column_map or {})
    q_spec = column_map.get("quantiles", None)
    q_map: Dict[float, str] = {}

    if q_spec is not None:
        if isinstance(q_spec, Mapping):
            for q, col in q_spec.items():
                qf = float(q)
                if col not in df.columns:
                    raise KeyError(
                        f"Quantile col '{col}' (q={qf}) missing."
                    )
                q_map[qf] = col
        else:
            for col in q_spec:
                if col not in df.columns:
                    raise KeyError(f"Quantile col '{col}' missing.")
                # user provided list => infer q from suffix _qXX
                import re

                m = re.match(
                    rf"^{target_name}_q(\d+)$", str(col)
                )
                if not m:
                    raise ValueError(
                        f"Cannot infer q from '{col}'. "
                        f"Expected '{target_name}_qXX'."
                    )
                q_map[int(m.group(1)) / 100.0] = col
        return q_map

    # auto-detect
    import re

    pat = re.compile(rf"^{target_name}_q(\d+)$")
    for col in df.columns:
        m = pat.match(str(col))
        if m:
            q_map[int(m.group(1)) / 100.0] = str(col)

    return q_map


def _infer_actual_col(
    df: pd.DataFrame,
    *,
    target_name: str,
    column_map: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    column_map = dict(column_map or {})
    spec = column_map.get("actual", f"{target_name}_actual")
    if spec is None:
        return None
    if isinstance(spec, str):
        return spec if spec in df.columns else None
    cols = list(spec)
    if len(cols) != 1:
        raise ValueError("Only 1 actual col supported.")
    return cols[0] if cols[0] in df.columns else None


def _pick_median_q(
    q_map: Mapping[float, str],
    *,
    median_q: float = 0.5,
) -> float:
    if not q_map:
        raise ValueError("No quantiles detected.")
    return min(q_map.keys(), key=lambda q: abs(q - median_q))


def _pick_interval_qs(
    q_map: Mapping[float, str],
    *,
    interval: Tuple[float, float],
) -> Tuple[float, float]:
    lo_t, hi_t = interval
    qs = sorted(q_map.keys())
    q_lo = min(qs, key=lambda q: abs(q - lo_t))
    q_hi = min(qs, key=lambda q: abs(q - hi_t))
    if q_lo >= q_hi:
        raise ValueError(
            f"Bad interval selection: q_lo={q_lo}, q_hi={q_hi}."
        )
    return q_lo, q_hi


def _coverage_and_width(
    y_true: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
) -> Tuple[float, float]:
    m = np.isfinite(y_true) & np.isfinite(y_lo) & np.isfinite(y_hi)
    if not np.any(m):
        return float("nan"), float("nan")
    yt = y_true[m]
    lo = y_lo[m]
    hi = y_hi[m]
    cov = float(np.mean((yt >= lo) & (yt <= hi)))
    wid = float(np.mean(hi - lo))
    return cov, wid


def _fit_factor_bisect(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_md: np.ndarray,
    q_hi: np.ndarray,
    *,
    target: float = 0.8,
    tol: float = 1e-3,
    f_max: float = 5.0,
    max_iter: int = 32,
) -> float:
    y_true = np.asarray(y_true, float)
    q_lo = np.asarray(q_lo, float)
    q_md = np.asarray(q_md, float)
    q_hi = np.asarray(q_hi, float)

    def _cov(f: float) -> float:
        lo = q_md - f * (q_md - q_lo)
        hi = q_md + f * (q_hi - q_md)
        cov, _ = _coverage_and_width(y_true, lo, hi)
        return cov

    c1 = _cov(1.0)
    if np.isnan(c1):
        return 1.0
    if c1 >= target - tol:
        return 1.0

    cmax = _cov(f_max)
    if np.isnan(cmax):
        return 1.0
    if cmax < target - tol:
        return float(f_max)

    lo, hi = 1.0, float(f_max)
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        cm = _cov(mid)
        if cm >= target:
            hi = mid
        else:
            lo = mid
        if (hi - lo) <= 1e-6:
            break

    return float(hi)


def fit_interval_factors_df(
    df_eval: pd.DataFrame,
    *,
    target_name: str = "subsidence",
    column_map: Optional[Mapping[str, Any]] = None,
    step_col: str = "forecast_step",
    interval: Tuple[float, float] = (0.1, 0.9),
    target_coverage: float = 0.8,
    median_q: float = 0.5,
    tol: float = 1e-3,
    f_max: float = 5.0,
    max_iter: int = 32,
    verbose: int = 1,
    logger: Optional[logging.Logger] = None,
) -> Dict[int, float]:
    q_map = _infer_quantile_map(
        df_eval, target_name=target_name, column_map=column_map
    )
    act_col = _infer_actual_col(
        df_eval, target_name=target_name, column_map=column_map
    )
    if act_col is None:
        raise ValueError(
            "Cannot fit calibration factors: missing actual col."
        )

    q_md = _pick_median_q(q_map, median_q=median_q)
    q_lo, q_hi = _pick_interval_qs(q_map, interval=interval)

    if step_col not in df_eval.columns:
        groups = [(1, df_eval)]
    else:
        groups = df_eval.groupby(step_col)

    factors: Dict[int, float] = {}
    for h, g in groups:
        h_int = int(h)
        yt = g[act_col].to_numpy()
        lo = g[q_map[q_lo]].to_numpy()
        md = g[q_map[q_md]].to_numpy()
        hi = g[q_map[q_hi]].to_numpy()

        f = _fit_factor_bisect(
            yt,
            lo,
            md,
            hi,
            target=target_coverage,
            tol=tol,
            f_max=f_max,
            max_iter=max_iter,
        )
        factors[h_int] = float(f)

        vlog(
            f"[calib] fitted factor H{h_int}={f:.6g}",
            verbose=verbose,
            level=2,
            logger=logger,
        )

    return factors


def _enforce_monotonic_q(
    q_vals: np.ndarray,
    *,
    mode: str = "cummax",
) -> np.ndarray:
    if mode in (None, "", "none"):
        return q_vals
    if mode == "sort":
        return np.sort(q_vals, axis=1)
    if mode == "cummax":
        return np.maximum.accumulate(q_vals, axis=1)
    raise ValueError(
        "enforce_monotonic must be one of "
        "{'none','cummax','sort'}."
    )


def apply_interval_factors_df(
    df: pd.DataFrame,
    factors: Mapping[int, float] | float,
    *,
    target_name: str = "subsidence",
    column_map: Optional[Mapping[str, Any]] = None,
    step_col: str = "forecast_step",
    median_q: float = 0.5,
    keep_original: bool = False,
    factor_col: str = "calibration_factor",
    calibrated_col: str = "is_calibrated",
    enforce_monotonic: str = "cummax",
    verbose: int = 1,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    df2 = df.copy()
    q_map = _infer_quantile_map(
        df2, target_name=target_name, column_map=column_map
    )
    if not q_map:
        raise ValueError("No quantile columns to calibrate.")

    q_md = _pick_median_q(q_map, median_q=median_q)
    qs = sorted(q_map.keys())
    cols = [q_map[q] for q in qs]

    if keep_original:
        for c in cols:
            df2[f"{c}_raw"] = df2[c]

    # Resolve factors
    if isinstance(factors, (int, float)):
        f_map = None
        f_scalar = float(factors)
    else:
        f_map = {int(k): float(v) for k, v in factors.items()}
        f_scalar = None

    if step_col not in df2.columns:
        df2[step_col] = 1

    # Apply per horizon
    med_col = q_map[q_md]
    for h, g_idx in df2.groupby(step_col).groups.items():
        h_int = int(h)
        f_h = (
            float(f_map.get(h_int, 1.0))
            if f_map is not None
            else float(f_scalar)
        )

        idx = np.asarray(list(g_idx))
        med = df2.loc[idx, med_col].to_numpy(dtype=float)

        q_vals = df2.loc[idx, cols].to_numpy(dtype=float)

        # scale each quantile around median
        for j, q in enumerate(qs):
            if q < q_md:
                q_vals[:, j] = med - f_h * (med - q_vals[:, j])
            elif q > q_md:
                q_vals[:, j] = med + f_h * (q_vals[:, j] - med)
            else:
                q_vals[:, j] = med

        q_vals = _enforce_monotonic_q(
            q_vals, mode=enforce_monotonic
        )

        df2.loc[idx, cols] = q_vals
        df2.loc[idx, factor_col] = f_h

    df2[calibrated_col] = True
    return df2


def _looks_calibrated(
    df_eval: pd.DataFrame,
    *,
    target_name: str,
    column_map: Optional[Mapping[str, Any]],
    step_col: str,
    interval: Tuple[float, float],
    target: float,
    tol: float,
    calibrated_col: str,
    verbose: int,
    logger: Optional[logging.Logger],
) -> bool:
    if calibrated_col in df_eval.columns:
        v = df_eval[calibrated_col]
        if bool(np.all(v.astype(bool).to_numpy())):
            vlog(
                "[calib] detected calibrated_col=True.",
                verbose=verbose,
                level=2,
                logger=logger,
            )
            return True

    act = _infer_actual_col(
        df_eval, target_name=target_name, column_map=column_map
    )
    if act is None:
        return False

    q_map = _infer_quantile_map(
        df_eval, target_name=target_name, column_map=column_map
    )
    if not q_map:
        return False

    q_lo, q_hi = _pick_interval_qs(q_map, interval=interval)
    lo_col = q_map[q_lo]
    hi_col = q_map[q_hi]

    yt = df_eval[act].to_numpy(dtype=float)
    lo = df_eval[lo_col].to_numpy(dtype=float)
    hi = df_eval[hi_col].to_numpy(dtype=float)
    cov, _ = _coverage_and_width(yt, lo, hi)

    if np.isnan(cov):
        return False

    ok = abs(cov - target) <= tol
    vlog(
        f"[calib] auto-check coverage={cov:.4f} "
        f"target={target:.4f} tol={tol:.4f} -> {ok}",
        verbose=verbose,
        level=2,
        logger=logger,
    )
    return bool(ok)

def _normalize_factors(
    factors: Union[float, Mapping[Any, Any]]
) -> Union[float, Dict[int, float]]:
    if isinstance(factors, (int, float)):
        return float(factors)

    out: Dict[int, float] = {}
    for k, v in dict(factors).items():
        out[int(k)] = float(v)
    return out

def calibrate_quantile_forecasts(
    *,
    df_eval: Optional[pd.DataFrame] = None,
    df_future: Optional[pd.DataFrame] = None,
    target_name: str = "subsidence",
    column_map: Optional[Mapping[str, Any]] = None,
    step_col: str = "forecast_step",
    interval: Tuple[float, float] = (0.1, 0.9),
    target_coverage: float = 0.8,
    median_q: float = 0.5,
    use: str | bool = "auto",
    tol: float = 0.02,
    f_max: float = 5.0,
    max_iter: int = 32,
    keep_original: bool = False,
    enforce_monotonic: str = "cummax",
    overall_key: Optional[str] = "__overall__",
    calibrated_col: str = "is_calibrated",
    factor_col: str = "calibration_factor",
    factors: Optional[
        Union[float, Mapping[int, float], Mapping[str, float]]
    ] = None,
    save_eval: Optional[str | Path] = None,
    save_future: Optional[str | Path] = None,
    save_stats: Optional[str | Path] = None,
    verbose: int = 1,
    logger: Optional[logging.Logger] = None,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Dict[str, Any],
]:
    use_s: Any = use
    if isinstance(use, str):
        use_s = use.strip().lower()

    def _save_df(dfX: Optional[pd.DataFrame], p: Any) -> None:
        if not p or dfX is None:
            return
        pp = Path(p)
        pp.parent.mkdir(parents=True, exist_ok=True)
        dfX.to_csv(pp, index=False)

    def _save_stats(statsX: Dict[str, Any]) -> None:
        if not save_stats:
            return
        pp = Path(save_stats)
        pp.parent.mkdir(parents=True, exist_ok=True)
        with open(pp, "w", encoding="utf-8") as f:
            json.dump(statsX, f, indent=2)

    def _jsonable_factors(fx: Any) -> Any:
        if isinstance(fx, (int, float)):
            return float(fx)
        return {str(int(k)): float(v) for k, v in dict(fx).items()}

    stats: Dict[str, Any] = {
        "target": float(target_coverage),
        "interval": [float(interval[0]), float(interval[1])],
        "f_max": float(f_max),
        "tol": float(tol),
        "overall_key": overall_key,
    }

    # ----------------------------
    # 0) Hard disable
    # ----------------------------
    if use is False or use_s == "false":
        stats["skipped"] = True
        stats["reason"] = "use_false"
        _save_df(df_eval, save_eval)
        _save_df(df_future, save_future)
        _save_stats(stats)
        return df_eval, df_future, stats

    # ----------------------------
    # 1) Auto skip if calibrated
    #    (safe default, even if
    #     user passes factors)
    #    -> to force, set use=True
    # ----------------------------
    if use_s == "auto" and df_eval is not None:
        if _looks_calibrated(
            df_eval,
            target_name=target_name,
            column_map=column_map,
            step_col=step_col,
            interval=interval,
            target=target_coverage,
            tol=tol,
            calibrated_col=calibrated_col,
            verbose=verbose,
            logger=logger,
        ):
            stats["skipped"] = True
            stats["reason"] = "already_calibrated"
            _save_df(df_eval, save_eval)
            _save_df(df_future, save_future)
            _save_stats(stats)
            return df_eval, df_future, stats

    # ----------------------------
    # 2) Choose factors:
    #    - user factors win
    #    - else fit from eval
    # ----------------------------
    f_apply: Any = None

    if factors is not None:
        f_apply = _normalize_factors(factors)
        stats["factors_source"] = "user"
        stats["factors"] = _jsonable_factors(f_apply)

    elif df_eval is not None:
        f_apply = fit_interval_factors_df(
            df_eval,
            target_name=target_name,
            column_map=column_map,
            step_col=step_col,
            interval=interval,
            target_coverage=target_coverage,
            median_q=median_q,
            tol=tol * 0.1,
            f_max=f_max,
            max_iter=max_iter,
            verbose=verbose,
            logger=logger,
        )
        stats["factors_source"] = "fit"
        stats["factors"] = _jsonable_factors(f_apply)

    else:
        stats["skipped"] = True
        stats["reason"] = "no_eval_no_factors"
        _save_df(df_eval, save_eval)
        _save_df(df_future, save_future)
        _save_stats(stats)
        return df_eval, df_future, stats

    # If still None (shouldn't happen), skip safely.
    if f_apply is None:
        stats["skipped"] = True
        stats["reason"] = "no_factors"
        _save_df(df_eval, save_eval)
        _save_df(df_future, save_future)
        _save_stats(stats)
        return df_eval, df_future, stats

    # ----------------------------
    # 3) Apply
    # ----------------------------
    df_eval_cal = None
    df_future_cal = None

    if df_eval is not None:
        df_eval_cal = apply_interval_factors_df(
            df_eval,
            f_apply,
            target_name=target_name,
            column_map=column_map,
            step_col=step_col,
            median_q=median_q,
            keep_original=keep_original,
            factor_col=factor_col,
            calibrated_col=calibrated_col,
            enforce_monotonic=enforce_monotonic,
            verbose=verbose,
            logger=logger,
        )

    if df_future is not None:
        df_future_cal = apply_interval_factors_df(
            df_future,
            f_apply,
            target_name=target_name,
            column_map=column_map,
            step_col=step_col,
            median_q=median_q,
            keep_original=keep_original,
            factor_col=factor_col,
            calibrated_col=calibrated_col,
            enforce_monotonic=enforce_monotonic,
            verbose=verbose,
            logger=logger,
        )

    # Optional: compute before/after stats (eval only)
    if df_eval is not None:
        act = _infer_actual_col(
            df_eval, target_name=target_name, column_map=column_map
        )
        q_map = _infer_quantile_map(
            df_eval, target_name=target_name, column_map=column_map
        )
        if act is not None and q_map:
            q_lo, q_hi = _pick_interval_qs(q_map, interval=interval)
            lo_col = q_map[q_lo]
            hi_col = q_map[q_hi]

            def _summ(dfX: pd.DataFrame) -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                yt = dfX[act].to_numpy(float)
                lo = dfX[lo_col].to_numpy(float)
                hi = dfX[hi_col].to_numpy(float)
                cov, wid = _coverage_and_width(yt, lo, hi)
                out["coverage"] = cov
                out["sharpness"] = wid

                if step_col in dfX.columns:
                    per: Dict[str, Any] = {}
                    for h, g in dfX.groupby(step_col):
                        yt_h = g[act].to_numpy(float)
                        lo_h = g[lo_col].to_numpy(float)
                        hi_h = g[hi_col].to_numpy(float)
                        c_h, w_h = _coverage_and_width(yt_h, lo_h, hi_h)
                        per[str(int(h))] = {
                            "coverage": c_h,
                            "sharpness": w_h,
                        }
                    out["per_horizon"] = per
                return out

            stats["eval_before"] = _summ(df_eval)
            stats["eval_after"] = _summ(df_eval_cal)
            
    # Save outputs
    _save_df(df_eval_cal, save_eval)
    _save_df(df_future_cal, save_future)
    _save_stats(stats)

    return df_eval_cal, df_future_cal, stats


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
def calibrate_qforecasts(
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

# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from  __future__ import annotations 

import os
import logging 
import warnings 
from typing import ( 
    Optional, 
    List, 
    Tuple, 
    Dict, 
    Union, 
    Callable, 
    Literal 
)
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from ..core.checks import ( 
    exist_features, 
    check_datetime, 
)

from ..decorators import isdf 
from .generic_utils import vlog
from fusionlab._optdeps import HAS_TQDM, with_progress 
 

__all__ = [
    'check_sequence_feasibility', 'get_sequence_counts',
    'generate_pinn_sequences', 'generate_ts_sequences', 
    'build_future_sequences_npz'
 ]

@isdf
def build_future_sequences_npz(
    df_scaled: pd.DataFrame,
    *,
    time_col: str,
    time_col_num: str | None,
    lon_col: str,
    lat_col: str,
    time_steps: int,
    train_end_time: object | None = None,
    forecast_start_time: object | None = None,
    forecast_horizon: int | None = None,
    subs_col: str | None = None,
    gwl_col: str | None = None,
    h_field_col: str | None = None,
    static_features: list[str] | None = None,
    dynamic_features: list[str] | None = None,
    future_features: list[str] | None = None,
    group_id_cols: list[str] | None = None,
    mode: str | None = None,
    model_name: str | None = None,
    artifacts_dir: str | None = None,
    prefix: str = "future",
    future_mode: str ="auto", 
    verbose: int = 1,
    logger=None,
    stop_check: Callable[[], bool] = None, 
    progress_hook: Optional[Callable[[float], None]] = None,
    **kws,  
) -> dict:
    """
    Build history–future sequences and save them as compressed NPZ files.

    This helper constructs, for each spatial group, a sliding window of
    `time_steps` "history" points followed by a multi–step forecast
    horizon and exports the resulting NumPy arrays to disk.  It is
    time-agnostic: the `time_col` can be numeric (e.g. year, index),
    year-like floats, datetimes, or strings, as long as equality on
    that column is meaningful.

    If `train_end_time`, `forecast_start_time`, or `forecast_horizon`
    are not provided, they are inferred from the sorted unique values
    in ``df_scaled[time_col]``:

    * ``train_end_time``: by default the second-to-last unique time,
      leaving at least one future step.
    * ``forecast_start_time``: by default the first time strictly
      after ``train_end_time``.
    * ``forecast_horizon``: by default one time step ahead, clipped to
      the number of available future points.

    For each valid group, the function builds:

    * **History dynamic features**: shape
      ``(time_steps, n_dynamic)``.
    * **Future features**:
      - If ``mode`` starts with ``"tft"`` (e.g. ``"tft_like"``),
        history and future rows are concatenated, giving shape
        ``(time_steps + H, n_future)``.
      - Otherwise, only the forecast horizon is used, giving shape
        ``(H, n_future)``.
    * **Static features**: one vector per group,
      shape ``(n_static,)``.
    * **Coordinates** over the horizon: shape ``(H, 3)`` where the
      three columns are ``[time_num, lon, lat]``.
    * **H_field** over the horizon: shape ``(H, 1)``.
    * **Targets** (if present) over the horizon for subsidence and
      groundwater level: shapes ``(H, 1)`` each.

    All per-group arrays are stacked along a new batch dimension and
    written as two NPZ files:

    * ``<prefix>_inputs.npz``: coordinates, dynamic, static, future
      features and H field.
    * ``<prefix>_targets.npz``: subsidence and groundwater targets.

    Parameters
    ----------
    df_scaled : pandas.DataFrame
        Pre-processed (typically scaled) dataframe containing all
        required columns: time, spatial coordinates, static/dynamic/
        future features and optional targets.
    time_col : str
        Name of the column encoding the temporal index (e.g.
        ``"year"``, ``"date"``, ``"t_index"``).  May be numeric,
        datetime, or string.
    time_col_num : str or None
        Optional numeric time column used as a tie-breaker when
        multiple rows share the same ``time_col`` value.  If provided
        and present in a group, the last row sorted by this column is
        selected for that time.
    lon_col : str
        Name of the longitude (or x-coordinate) column.
    lat_col : str
        Name of the latitude (or y-coordinate) column.
    time_steps : int
        Length of the history window (number of time steps in the
        past). Must be strictly positive.
    train_end_time : object, optional
        Effective end of the training period.  If ``None``, it is
        inferred as the second-to-last unique value in
        ``df_scaled[time_col]`` (after sorting).
    forecast_start_time : object, optional
        First time step of the forecast horizon.  If ``None``, it is
        inferred as the first unique time strictly greater than
        ``train_end_time``.
    forecast_horizon : int, optional
        Number of future time steps to include.  If ``None``, a
        default horizon of ``1`` is used and clipped to the maximum
        number of available future time points.
    subs_col : str, optional
        Name of the subsidence target column.  If ``None`` or missing
        from a group, subsidence targets are filled with ``NaN``.
    gwl_col : str, optional
        Name of the groundwater-level target column.  If ``None`` or
        missing from a group, groundwater targets are filled with
        ``NaN``.
    h_field_col : str, optional
        Name of the hydraulic-head field column used as an additional
        horizon-level input (``H_field``).  If ``None`` or missing,
        a zero field is used.
    static_features : list of str, optional
        Names of static (time-invariant) feature columns.  Any names
        not present in the dataframe are silently ignored.
    dynamic_features : list of str, optional
        Names of dynamic (history) feature columns used to build the
        ``(time_steps, n_dynamic)`` sequence.  Missing columns are
        ignored.
    future_features : list of str, optional
        Names of future covariate columns used to build the
        history+future or future-only sequence, depending on
        ``mode``.  Missing columns are ignored.
    group_id_cols : list of str, optional
        Columns used to define spatial (or logical) groups, typically
        something like ``["lon", "lat"]`` or a station identifier.
        If ``None`` or empty, the entire dataframe is treated as a
        single global group.
    mode : str, optional
        Controls how future features are constructed.  If the
        lower-cased value starts with ``"tft"`` (e.g. ``"tft_like"``),
        future features are built on top of both history and future
        rows.  Otherwise, only the forecast horizon rows are used.
    model_name : str, optional
        Optional model identifier used only in logging messages.
    artifacts_dir : str, optional
        Directory where NPZ files are written.  If ``None`` or empty,
        the current working directory is used.
    prefix : str, default="future"
        Prefix for the output NPZ filenames:
        ``"<prefix>_inputs.npz"`` and ``"<prefix>_targets.npz"``.
    future_mode : {'auto', 'pure-inference', 'pure-data-driven'}, \
default 'auto'
        Strategy used to construct the future (forecast) portion of
        the sequences.

        - ``'pure-data-driven'``:
          Use only time points that actually exist in ``df_scaled``
          strictly after the history window. All future time indices
          must be present in the data; otherwise a ``ValueError`` is
          raised. This corresponds to the original, strictly
          data-driven behaviour.

        - ``'pure-inference'``:
          Always synthesize future time points from the last history
          time, using the median positive time step (or ``1.0`` as a
          fallback). Future inputs are built by re-using the last
          available history row (for ``future_features``, ``H_field``,
          etc.), and future targets (e.g. subsidence, GWL) are filled
          with ``NaN`` since the true future is unknown. This mode
          does not require any rows beyond ``train_end_time``.

        - ``'auto'``:
          Try data-driven mode first. If there are enough actual
          future time points after ``train_end_time`` to cover the
          requested ``forecast_horizon``, behave like
          ``'pure-data-driven'``. If not, automatically fall back to
          the synthetic ``'pure-inference'`` behaviour described
          above and emit an informational log message via ``vlog``.

    verbose : int, default=1
        Verbosity level forwarded to :func:`fusionlab.utils.vlog`.  A
        value ``>= 3`` provides detailed progress logs (temporal
        inference, per-group status, dropped groups, etc.).
    logger : logging.Logger or callable, optional
        Optional logger or logging function used by
        :func:`fusionlab.utils.vlog`.  If ``None``, messages are
        printed to standard output.
    **kws
        Reserved for future extensions.  Currently ignored.

    Returns
    -------
    dict
        A small dictionary with the absolute paths to the written NPZ
        files:

        ``{"future_inputs_npz": <path>, "future_targets_npz": <path>}``.

    Raises
    ------
    ValueError
        If there are not enough history points before
        ``train_end_time`` to satisfy ``time_steps``, if no future
        points are available after ``forecast_start_time``, or if all
        groups are dropped due to incomplete history/horizon windows.

    Notes
    -----
    Groups that do not contain all required history and future times
    are silently dropped, but the number of dropped groups is reported
    via :func:`fusionlab.utils.vlog` when ``verbose > 0``.

    Examples
    --------
    >>> from fusionlab.nn.pinn.sequences import (
    ...     build_future_sequences_npz,
    ... )
    >>> result = build_future_sequences_npz(
    ...     df_scaled=df_scaled,
    ...     time_col="year",
    ...     time_col_num="t_index",
    ...     lon_col="lon",
    ...     lat_col="lat",
    ...     time_steps=5,
    ...     # Let the function infer times/horizon:
    ...     train_end_time=None,
    ...     forecast_start_time=None,
    ...     forecast_horizon=None,
    ...     subs_col="subsidence",
    ...     gwl_col="gwl",
    ...     h_field_col="H_field",
    ...     static_features=["lithology_class"],
    ...     dynamic_features=["rainfall_mm", "GWL_depth_bgs_z"],
    ...     future_features=["normalized_urban_load_proxy"],
    ...     group_id_cols=["lon", "lat"],
    ...     mode="tft_like",
    ...     model_name="GeoPriorSubsNet",
    ...     artifacts_dir="results/zhongshan/future_npz",
    ...     prefix="zhongshan_future",
    ...     verbose=2,
    ... )
    >>> result["future_inputs_npz"]
    'results/zhongshan/future_npz/zhongshan_future_inputs.npz'
    >>> result["future_targets_npz"]
    'results/zhongshan/future_npz/zhongshan_future_targets.npz'
    """
    def _p(frac: float) -> None:
        if progress_hook is not None:
            # clamp, avoid crashing
            try:
                f = max(0.0, min(1.0, float(frac)))
                progress_hook(f)
            except Exception:
                pass

    _p(0.0)  # start
    
    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _save_npz(path: str, arrays: dict) -> str:
        np.savez_compressed(path, **arrays)
        return path

    def _to_time_index(series: pd.Series) -> np.ndarray:
        """Convert a time column to a numeric index (year-like)."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return series.dt.year.to_numpy(dtype=float)
        try:
            return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        except Exception:
            vals, _ = pd.factorize(series)
            return vals.astype(float)

    future_mode_norm = (future_mode or "auto").lower()
    if future_mode_norm not in ("auto", "pure-inference", "pure-data-driven"):
        raise ValueError(
            "build_future_sequences_npz: 'future_mode' must be one of "
            "{'auto', 'pure-inference', 'pure-data-driven'}; "
            f"got {future_mode!r}."
        )

    if artifacts_dir is None or not str(artifacts_dir).strip():
        artifacts_dir = os.getcwd()

    # ------------------------------------------------------------------
    # Step 1: infer temporal configuration (history + future windows)
    # ------------------------------------------------------------------

    exist_features(
        df_scaled, features=time_col, 
        message="Time col{time_col} column is missing."
    )
    
    vlog("Validating time-series dataset...", 
         level=1, verbose=verbose)
    
    check_datetime(
        df_scaled,
        dt_cols= time_col, 
        ops="check_only",
        consider_dt_as="numeric",
        accept_dt=True, 
        allow_int=True, 
    )
    
    t_series = df_scaled[time_col]
    t_idx_all = _to_time_index(t_series)
    mask_finite = np.isfinite(t_idx_all)
    if not mask_finite.any():
        raise ValueError(
            "build_future_sequences_npz: time column contains no finite "
            "values after conversion."
        )

    unique_times = np.unique(t_idx_all[mask_finite])
    unique_times.sort()

    T = int(time_steps)
    if T <= 0:
        raise ValueError(
            "build_future_sequences_npz: 'time_steps' must be > 0; "
            f"got {time_steps!r}."
        )

    # Resolve train_end_time
    if train_end_time is None:
        train_end_idx = float(unique_times.max())
    else:
        try:
            train_end_idx = float(train_end_time)
        except Exception as e:
            raise ValueError(
                "build_future_sequences_npz: could not convert "
                f"train_end_time={train_end_time!r} to numeric index."
            ) from e

    # History times: last T distinct times <= train_end_idx
    hist_candidates = unique_times[unique_times <= train_end_idx]
    if hist_candidates.size < T:
        raise ValueError(
            "build_future_sequences_npz: not enough past time points "
            f"<= train_end_time to build history of length {T}. "
            f"Available={hist_candidates.size}."
        )
    hist_times = hist_candidates[-T:]
    last_hist = hist_times[-1]

    # Step (for synthetic future): median positive diff or 1.0
    if hist_times.size >= 2:
        diffs = np.diff(hist_times)
        step = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
    else:
        step = 1.0

    # All future times *in data* (strictly after last_hist)
    data_future_all = unique_times[unique_times > last_hist]

    # Horizon H
    if forecast_horizon is None:
        H = int(data_future_all.size) if data_future_all.size > 0 else 1
    else:
        H = int(forecast_horizon)
    if H <= 0:
        raise ValueError(
            "build_future_sequences_npz: 'forecast_horizon' must be > 0; "
            f"got {forecast_horizon!r}."
        )

    # Resolve forecast_start_time index
    if forecast_start_time is None:
        if data_future_all.size > 0:
            f_start_idx = float(data_future_all[0])
        else:
            f_start_idx = float(last_hist + step)
    else:
        try:
            f_start_idx = float(forecast_start_time)
        except Exception as e:
            raise ValueError(
                "build_future_sequences_npz: could not convert "
                f"forecast_start_time={forecast_start_time!r} to numeric index."
            ) from e

    # Candidate data-driven future times (>= f_start_idx, after last_hist)
    data_future_needed = data_future_all[data_future_all >= f_start_idx][:H]
    has_enough_future_data = data_future_needed.size == H
    
    _p(0.1)
    
    # Decide effective mode + future times
    using_synthetic_future = False
    if future_mode_norm == "pure-data-driven":
        if not has_enough_future_data:
            raise ValueError(
                "build_future_sequences_npz: pure-data-driven mode "
                f"requested but only {data_future_needed.size} future "
                f"time point(s) found starting from {f_start_idx}; "
                f"need {H}."
            )
        fut_times = data_future_needed
        eff_future_mode = "pure-data-driven"

    elif future_mode_norm == "pure-inference":
        using_synthetic_future = True
        eff_future_mode = "pure-inference"
        start = max(float(last_hist), float(f_start_idx))
        fut_times = np.array(
            [start + step * (i + 1) for i in range(H)],
            dtype=float,
        )

    else:  # auto
        if has_enough_future_data:
            fut_times = data_future_needed
            eff_future_mode = "pure-data-driven"
        else:
            using_synthetic_future = True
            eff_future_mode = "pure-inference"
            start = max(float(last_hist), float(f_start_idx))
            fut_times = np.array(
                [start + step * (i + 1) for i in range(H)],
                dtype=float,
            )

    # Window actually needed in df_scaled
    if using_synthetic_future:
        all_times_needed = hist_times
    else:
        all_times_needed = np.concatenate([hist_times, fut_times])

    if verbose:
        vlog(
            "[Future] Step 1/3: temporal config resolved.\n"
            f"  hist_times       = {hist_times.tolist()}\n"
            f"  fut_times        = {fut_times.tolist()}\n"
            f"  history length   = {T}\n"
            f"  horizon length   = {H}\n"
            f"  effective mode   = {eff_future_mode}\n"
            f"  train_end_time   = {train_end_time!r}\n"
            f"  forecast_start   = {forecast_start_time!r}",
            verbose=verbose,
            level=1,
            logger=logger,
        )
        if using_synthetic_future:
            vlog(
                "[Future] No usable future time points after train_end_time; "
                "falling back to synthetic 'pure-inference' horizon "
                "(targets will be NaN).",
                verbose=verbose,
                level=2,
                logger=logger,
            )

    # ------------------------------------------------------------------
    # Step 2: subset df_scaled to the required window
    # ------------------------------------------------------------------
    mask_window = np.isin(t_idx_all, all_times_needed)
    df_win = df_scaled.loc[mask_window].copy()
    if df_win.empty:
        raise ValueError(
            "build_future_sequences_npz: no rows in df_scaled for the "
            f"required time window {all_times_needed.tolist()}."
        )

    # Normalize lists
    static_features = list(static_features) if static_features is not None else []
    dynamic_features = list(dynamic_features) if dynamic_features is not None else []
    future_features = list(future_features) if future_features is not None else []
    group_id_cols = list(group_id_cols) if group_id_cols is not None else []

    static_cols = [c for c in static_features if c in df_win.columns]
    dyn_cols = [c for c in dynamic_features if c in df_win.columns]
    fut_cols = [c for c in future_features if c in df_win.columns]

    mode_norm = (mode or "").lower()
    is_tft_like = mode_norm.startswith("tft")

    if verbose:
        vlog(
            "[Future] Step 2/3: feature and grouping config.\n"
            f"  static_cols   = {static_cols}\n"
            f"  dyn_cols      = {dyn_cols}\n"
            f"  fut_cols      = {fut_cols}\n"
            f"  group_id_cols = {group_id_cols or ['<global>']}\n"
            f"  mode          = {mode_norm or 'None'}",
            verbose=verbose,
            level=1,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Step 3: per-group construction
    # ------------------------------------------------------------------

    def _rows_for_times(
          required: np.ndarray, 
          g_df: pd.DataFrame
        ) -> list | None:
        """Pick one row per required time from a group."""
        g_time_idx = _to_time_index(g_df[time_col])
        order = np.argsort(g_time_idx)
        g_sorted = g_df.iloc[order].reset_index(drop=True)
        g_time_sorted = g_time_idx[order]

        rows: list[pd.Series] = []
        for t_req in required:
            m = (g_time_sorted == t_req)
            if not np.any(m):
                return None
            sub = g_sorted.loc[m]
            if time_col_num is not None and time_col_num in sub.columns:
                sub = sub.sort_values(time_col_num)
            rows.append(sub.iloc[-1])
        return rows

    def _build_synthetic_future_rows(
        hist_rows: list[pd.Series],
        fut_times_arr: np.ndarray,
    ) -> list[pd.Series]:
        """Replicate last history row and overwrite time columns."""
        if not hist_rows:
            return []
        last_row = hist_rows[-1]
        out = []
        for t_val in fut_times_arr:
            row = last_row.copy()
            row[time_col] = t_val
            if time_col_num is not None and time_col_num in row.index:
                try:
                    row[time_col_num] = float(t_val)
                except Exception:
                    # keep original numeric coord
                    pass
            out.append(row)
        return out

    coords_list, dyn_list, fut_list = [], [], []
    static_list, H_list = [], []
    subs_target_list, gwl_target_list = [], []

    # # # Grouping
    # # Setup grouping (allow no grouping => single global group)
    # if group_id_cols:
    #     grouped = df_win.groupby(group_id_cols)
    #     n_groups = grouped.ngroups
    #     group_iter = grouped
    # else:
    #     n_groups = 1
    #     group_iter = [(None, df_win)]
    
    # # Progress logging frequency for non-tqdm fallback
    # log_every = max(1, n_groups // 10)
    
    # # Optionally wrap with tqdm if available and verbosity suggests user cares
    # use_tqdm = HAS_TQDM  and verbose >= 1 and n_groups > 1
    # if use_tqdm:
    #     iter_groups = _TQDM(
    #         group_iter,
    #         total=n_groups,
    #         desc=f"[Future] groups ({model_name or ''})".strip(),
    #         leave=False,
    #     )
    # else:
    #     iter_groups = group_iter

        # Grouping
    # Setup grouping (allow no grouping => single global group)
    
    if group_id_cols:
        grouped = df_win.groupby(group_id_cols)
        n_groups = grouped.ngroups
        group_iter = grouped
    else:
        n_groups = 1
        group_iter = [(None, df_win)]

    # Progress logging frequency for non-tqdm fallback
    log_every = max(1, n_groups // 10)

    # Small adapter so tqdm output goes into your logger instead of terminal
    def _log_progress_line(msg: str) -> None:
        # Use vlog so it respects verbose + user logger
        vlog(
            msg,
            verbose=verbose,
            level=2,
            logger=logger,
        )

    # Optionally wrap with tqdm, but redirect its ASCII bar to the log
    use_tqdm = HAS_TQDM and verbose >= 1 and n_groups > 1
    if use_tqdm:
        iter_groups = with_progress(
            group_iter,
            total=n_groups,
            desc=f"[Future] groups ({model_name or ''})".strip(),
            leave=False,
            ascii=True,
            log_fn=_log_progress_line,  
            mininterval=1.0,             # optional: don't spam too often
        )
    else:
        iter_groups = group_iter

    dropped_groups = 0
    
    for gi, (gid, g) in enumerate(iter_groups, start=1):
        
        # after finishing this group:
        _p(0.1 + 0.8 * (gi + 1) / n_groups)
        
        if stop_check and stop_check():
            raise InterruptedError("Sequence generation aborted.")
        # If tqdm is present, keep vlog quieter; if not, use your previous pattern
        if not use_tqdm:
            if verbose and (gi == 1 or gi % log_every == 0 or gi == n_groups):
                vlog(
                    f"[Future] Processing group {gi}/{n_groups} gid={gid!r}",
                    verbose=verbose,
                    level=2,
                    logger=logger,
                )
            else:
                vlog(
                    f"[Future] Proc.subset group {gi}/{n_groups} gid={gid!r}",
                    verbose=verbose,
                    level=6,
                    logger=logger,
                )
        # 1) History rows (must exist in data)
        hist_rows = _rows_for_times(hist_times, g)
        if hist_rows is None:
            dropped_groups += 1
            if verbose >= 2:
                vlog(
                    f"[Future] Dropping group {gid!r}: incomplete history "
                    f"for times={hist_times.tolist()}.",
                    verbose=verbose,
                    level=2,
                    logger=logger,
                )
            continue

        # 2) Future rows: data-driven or synthetic
        if using_synthetic_future:
            fut_rows = _build_synthetic_future_rows(hist_rows, fut_times)
        else:
            fut_rows = _rows_for_times(fut_times, g)
            if fut_rows is None:
                dropped_groups += 1
                if verbose >= 2:
                    vlog(
                        f"[Future] Dropping group {gid!r}: incomplete future "
                        f"for times={fut_times.tolist()} in data-driven mode.",
                        verbose=verbose,
                        level=2,
                        logger=logger,
                    )
                continue

        # Dynamic features: history only
        if dyn_cols:
            dyn_seq = np.stack(
                [row[dyn_cols].to_numpy(dtype=np.float32) for row in hist_rows],
                axis=0,
            )
        else:
            dyn_seq = np.zeros((len(hist_rows), 0), dtype=np.float32)

        # Future features
        if fut_cols:
            if is_tft_like:
                rows_for_future = hist_rows + fut_rows
            else:
                rows_for_future = fut_rows
            fut_seq = np.stack(
                [row[fut_cols].to_numpy(dtype=np.float32) for row in rows_for_future],
                axis=0,
            )
        else:
            T_fut = len(hist_rows) + len(fut_rows) if is_tft_like else len(fut_rows)
            fut_seq = np.zeros((T_fut, 0), dtype=np.float32)

        # Coords for horizon (t, lon, lat)
        t_vals = []
        for row in fut_rows:
            if time_col_num is not None and time_col_num in row.index:
                t_vals.append(row[time_col_num])
            else:
                t_vals.append(row[time_col])
        t_vals = np.array(t_vals, dtype=np.float32)
        lon_vals = np.array([row[lon_col] for row in fut_rows], dtype=np.float32)
        lat_vals = np.array([row[lat_col] for row in fut_rows], dtype=np.float32)
        coords = np.stack([t_vals, lon_vals, lat_vals], axis=-1)

        # H_field over horizon
        if h_field_col is not None and h_field_col in g.columns:
            h_vals = np.array(
                [row[h_field_col] for row in fut_rows],
                dtype=np.float32,
            )
        else:
            h_vals = np.zeros(len(fut_rows), dtype=np.float32)
        H_seq = h_vals.reshape(-1, 1)

        # Static features (one vector per group)
        if static_cols:
            static_vec = hist_rows[0][static_cols].to_numpy(dtype=np.float32)
        else:
            static_vec = np.zeros(0, dtype=np.float32)

        # Targets over horizon
        if using_synthetic_future:
            subs_vals = np.full(len(fut_rows), np.nan, dtype=np.float32)
            gwl_vals = np.full(len(fut_rows), np.nan, dtype=np.float32)
        else:
            if subs_col is not None and subs_col in g.columns:
                subs_vals = np.array(
                    [row[subs_col] for row in fut_rows],
                    dtype=np.float32,
                )
            else:
                subs_vals = np.full(len(fut_rows), np.nan, dtype=np.float32)

            if gwl_col is not None and gwl_col in g.columns:
                gwl_vals = np.array(
                    [row[gwl_col] for row in fut_rows],
                    dtype=np.float32,
                )
            else:
                gwl_vals = np.full(len(fut_rows), np.nan, dtype=np.float32)

        subs_seq = subs_vals.reshape(-1, 1)
        gwl_seq = gwl_vals.reshape(-1, 1)

        coords_list.append(coords)
        dyn_list.append(dyn_seq)
        fut_list.append(fut_seq)
        static_list.append(static_vec)
        H_list.append(H_seq)
        subs_target_list.append(subs_seq)
        gwl_target_list.append(gwl_seq)

    if not coords_list:
        raise ValueError(
            "build_future_sequences_npz: no valid groups with complete "
            "history window (and future, if required)."
        )

    if dropped_groups and verbose:
        vlog(
            f"[Future] Dropped {dropped_groups} group(s) due to incomplete "
            "history/future windows.",
            verbose=verbose,
            level=1,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Stack and save
    # ------------------------------------------------------------------
    coords_arr = np.stack(coords_list, axis=0)
    dyn_arr = np.stack(dyn_list, axis=0)
    fut_arr = np.stack(fut_list, axis=0)
    static_arr = np.stack(static_list, axis=0)
    H_arr = np.stack(H_list, axis=0)
    subs_targets_arr = np.stack(subs_target_list, axis=0)
    gwl_targets_arr = np.stack(gwl_target_list, axis=0)

    future_inputs_np = {
        "coords": coords_arr,
        "dynamic_features": dyn_arr,
        "static_features": static_arr,
        "future_features": fut_arr,
        "H_field": H_arr,
    }
    future_targets_np = {
        "subsidence": subs_targets_arr,
        "gwl": gwl_targets_arr,
    }

    os.makedirs(artifacts_dir, exist_ok=True)
    future_inputs_npz = os.path.join(artifacts_dir, f"{prefix}_inputs.npz")
    future_targets_npz = os.path.join(artifacts_dir, f"{prefix}_targets.npz")

    _save_npz(future_inputs_npz, future_inputs_np)
    _save_npz(future_targets_npz, future_targets_np)

    if verbose:
        vlog(
            "[Future] Step 3/3: NPZs saved.\n"
            f"  inputs : {future_inputs_npz}\n"
            f"  targets: {future_targets_npz}\n"
            f"  mode   : {eff_future_mode}"
            + (" (synthetic future targets=NaN)"
               if using_synthetic_future else ""),
            verbose=verbose,
            level=1,
            logger=logger,
        )
    # After saving NPZs:
    _p(1.0)
    
    return {
        "future_inputs_npz": future_inputs_npz,
        "future_targets_npz": future_targets_npz,
    }


@isdf 
def check_sequence_feasibility(
    df: pd.DataFrame,
    *,
    time_col: str,
    group_id_cols: Optional[List[str]] = None,
    time_steps: int = 12,
    forecast_horizon: int = 3,
    engine: Literal["vectorized", "native", "pyarrow"] = "vectorized",
    mode: Optional[str] = None,  
    logger: Callable[[str], None] = print,
    verbose: int = 0,
    error: Literal["raise", "warn", "ignore"] = "warn",
) -> Tuple[bool, Dict[Union[str, Tuple], int]]:
    """
    Quick pre-flight feasibility check for sliding-window sequence
    generation

    Checks whether the input table is *long enough*—per group—to yield at
    least one `(look-back + horizon)` sliding window, **without** allocating
    large NumPy tensors.  It is typically called immediately before
    :pyfunc:`prepare_pinn_data_sequences` or similar generators to “fail
    fast’’ on data shortages.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Tidy time-series table in **long** format.  Every row represents one
        observation timestamp (and optionally one entity when
        *group_id_cols* is given).  The function never mutates *df*.
    time_col : str
        Column that defines temporal order inside each trajectory.  Must be
        sortable; no other assumptions (numeric, datetime, …) are made.
    group_id_cols : list of str or None, default None
        Column names that jointly identify independent trajectories
        (e.g. ``["well_id"]`` or ``["site", "layer_id"]``).  When *None* the
        whole DataFrame is treated as a single group.
    time_steps : int, default 12
        Look-back window :math:`T_\text{past}` consumed by the encoder.
    forecast_horizon : int, default 3
        Prediction horizon :math:`H` produced by the decoder.
    engine : {'vectorized', 'loop', 'pyarrow'}, default 'vectorized'
        * **'vectorized'** – fastest; single :pymeth:`DataFrame.groupby.size`
          call (C-level) plus NumPy math.
        * **'native'** – reproduces the original Python loop for
          debuggability.
        * **'pyarrow'** – forces pandas’ Arrow backend, then runs the same
          vectorised logic; ~20 % faster on very wide frames when
          *pyarrow* ≥ 14 is installed.
    mode : {'pihal_like', 'tft_like'} or None, optional
        Present only for API symmetry.  **Ignored** – feasibility depends
        *solely* on ``time_steps + forecast_horizon``.
    logger : callable, default :pyfunc:`print`
        Sink for human-readable log messages.  Must accept a single `str`.
    verbose : int, default 0
        Verbosity level:
        0 → silent, 1 → summary lines, 2 → per-group detail.
    error : {'raise', 'warn', 'ignore'}, default 'warn'
        Action when *no* group is long enough.
    
        * ``'raise'`` – raise :class:`SequenceGeneratorError`.
        * ``'warn'``  – emit :class:`UserWarning`, return ``False``.
        * ``'ignore'`` – stay silent, return ``False``.
    
    Returns
    -------
    feasible : bool
        ``True`` iff *at least one* sequence can be produced,
        otherwise ``False``.
    counts : dict
        Mapping **group key → # sequences**.
        The key is a tuple of the group values—or *None* when
        *group_id_cols* is *None*.
    
    Raises
    ------
    SequenceGeneratorError
        Raised only when ``error='raise'`` *and* all groups fail the length
        check.
    
    Notes
    -----
    A group passes the check iff
    
    .. math::
    
       \\text{len(group)} \\;\\ge\\; T_\\text{past} + H
    
    No validation of time-gaps, duplicates, or NaNs is performed; those are
    deferred to the full preparation routine.
    
    The **Arrow backend** (``engine='pyarrow'``) can accelerate very wide
    frames because each column is represented as a contiguous Arrow array
    with cheap zero-copy slicing.
    
    Examples
    --------
    * Minimal usage

    >>> from fusionlab.utils.sequence_utils import check_sequence_feasibility
    >>> ok, counts = check_sequence_feasibility(
    ...     df,
    ...     time_col="date",
    ...     group_id_cols=["site"],
    ...     time_steps=6,
    ...     forecast_horizon=3,
    ... )
    >>> ok
    True
    >>> counts            # doctest: +ELLIPSIS
    {'A': 9, 'B': 9}
    
    * Fail-fast behaviour

    >>> check_sequence_feasibility(
    ...     df_small,
    ...     time_col="t",
    ...     time_steps=10,
    ...     forecast_horizon=5,
    ...     error="raise",
    ... )
    Traceback (most recent call last):
    ...
    SequenceGeneratorError: No group is long enough ...
    
    * Switching engines

    >>> _ , _ = check_sequence_feasibility(
    ...     df,
    ...     time_col="ts",
    ...     group_id_cols=None,
    ...     engine="pyarrow",   # requires pandas 2.1+, pyarrow installed
    ...     verbose=1,
    ... )
    ✅ Feasible: 1 234 567 sequences possible.
    
    References
    ----------
    * McKinney, W. *pandas 2.0 User Guide*, sec. “GroupBy: split-apply-combine’’.
    * Arrow Project. (2025). *Arrow Columnar Memory Format v2*.

    """
    # --- tiny inline logger 
    def _v(msg: str, *, lvl: int = 1) -> None:
        vlog(msg, verbose=verbose, level=lvl, logger=logger)

    min_len = time_steps + forecast_horizon
    _v(f"Required length per group: {min_len}", lvl=2)

    # deterministic ordering
    # or just df; sorting not needed for counts
    
    # sort_cols = (group_id_cols or []) + [time_col]
    # df_sorted = df.sort_values(sort_cols)
    
    # inside your feasibility function
    total_sequences, counts, sizes = get_sequence_counts(
        df,
        group_id_cols=group_id_cols,
        min_len=min_len,
        engine=engine,        
        verbose=verbose,
        logger=logger,    
    )
    
    if total_sequences == 0:
        longest = int(sizes.max()) if not sizes.empty else 0
        msg = (
            "No group is long enough to create any sequence.\n"
            f"Each trajectory needs >= {min_len} consecutive records "
            f"(time_steps={time_steps}, horizon={forecast_horizon}), "
            f"but the longest has only {longest}.\n"
            "-> Reduce `time_steps` / `forecast_horizon`, "
            "or supply more data."
        )
        # _v("❌ " + msg.splitlines()[0], lvl=1)
        _v("" + msg.splitlines()[0], lvl=1)

        if error == "raise":
            raise SequenceGeneratorError(msg)
        if error == "warn":
            warnings.warn(msg, UserWarning, stacklevel=2)
        return False, counts

    # _v(rf"✅ Feasible: {total_sequences} sequences possible.", lvl=1)
    _v(f" Feasible: {total_sequences} sequences possible.", lvl=1)
    return True, counts

def _sequence_counts_fast(
    df: pd.DataFrame,
    group_id_cols: Optional[List[str]],
    min_len: int,
) -> Tuple[int, Dict[Union[str, Tuple], int], pd.Series]:
    """Vectorised: one C call → group sizes, then NumPy math."""
    if group_id_cols:
        sizes = df.groupby(group_id_cols, sort=False).size()
    else:
        sizes = pd.Series([len(df)], index=[None])

    n_seq_series = np.maximum(sizes - min_len + 1, 0)
    return int(n_seq_series.sum()), n_seq_series.to_dict(), sizes

def _sequence_counts_loop(
    df: pd.DataFrame,
    group_id_cols: Optional[List[str]],
    min_len: int,
) -> Tuple[int, Dict[Union[str, Tuple], int], pd.Series]:
    """Original Python loop – slower but easy to single-step."""
    if group_id_cols:
        iterator = df.groupby(group_id_cols)
    else:
        iterator = [(None, df)]

    counts: Dict[Union[str, Tuple], int] = {}
    sizes_dict: Dict[Union[str, Tuple], int] = {}
    total_sequences = 0

    for g_key, g_df in iterator:
        n_pts = len(g_df)
        n_seq = max(n_pts - min_len + 1, 0)
        counts[g_key] = n_seq
        sizes_dict[g_key] = n_pts
        total_sequences += n_seq

    sizes = pd.Series(sizes_dict)
    return total_sequences, counts, sizes

@isdf
def get_sequence_counts(
    df: pd.DataFrame,
    *,
    group_id_cols: Optional[List[str]],
    min_len: int,
    engine: Literal["vectorized", "native", "pyarrow"] = "vectorized",
    verbose: int = 0,
    logger=print,
) -> Tuple[int, Dict[Union[str, Tuple], int], pd.Series]:
    """
    Return the **total** number of feasible sliding-window sequences and
    a mapping *group → count* using the requested execution *engine*.

    Parameters
    ----------
    engine : {'vectorized', 'native', 'pyarrow'}, default 'vectorized'
        Execution backend.

        * **'vectorized'** – fast C-level :pymeth:`DataFrame.groupby.size`
          (recommended).
        * **'native'** – original Python loop (easier to debug, slower).
        * **'pyarrow'** – forces pandas’ Arrow backend *if available*,
          then runs the vectorised path.  Falls back silently to
          ``'vectorized'`` when *pyarrow* is not installed.
    """
    def _v(msg: str, lvl: int = 1) -> None:
        vlog(msg, verbose=verbose, level=lvl, logger=logger)

    if engine == "pyarrow":
        try:
            import pyarrow  # noqa: F401
        except ImportError:  # ⇢ graceful fallback
            # _v("⚠  pyarrow not installed — reverting to 'vectorized'.", lvl=1)
            _v("  pyarrow not installed — reverting to 'vectorized'.", lvl=1)
            engine = "vectorized"

    if engine == "pyarrow":
        old_backend = pd.options.mode.dtype_backend
        pd.options.mode.dtype_backend = "pyarrow"
        try:
            total, counts, sizes = _sequence_counts_fast(
                df, group_id_cols, min_len)
        finally:
            pd.options.mode.dtype_backend = old_backend

    elif engine == "vectorized":
        total, counts, sizes = _sequence_counts_fast(
            df, group_id_cols, min_len)

    elif engine == "native":
        total, counts, sizes = _sequence_counts_loop(
            df, group_id_cols, min_len)

    else:  # pragma: no cover
        raise ValueError(
            f"Unknown engine='{engine}'. "
            "Choose 'vectorized', 'loop', or 'pyarrow'."
        )

    if verbose >= 2:
        for g_key, n_seq in counts.items():
            size_str = sizes[g_key]
            _v(
                f"Group {g_key if g_key is not None else '<whole>'}: "
                f"{size_str} pts -> {n_seq} seq.",
                lvl=2,
            )

    return total, counts, sizes

@isdf 
def generate_pinn_sequences(
    df: pd.DataFrame,
    time_col: str,
    subsidence_col: str,
    gwl_col: str,
    dynamic_cols: List[str],
    static_cols: Optional[List[str]] = None,
    future_cols: Optional[List[str]] = None,
    spatial_cols: Optional[Tuple[str, str]] = None,
    group_id_cols: Optional[List[str]] = None,
    time_steps: int = 12,
    forecast_horizon: int = 3,
    output_subsidence_dim: int = 1,
    output_gwl_dim: int = 1,
    mode: str = 'pihal_like',
    normalize_coords: bool = True,
    cols_to_scale: Union[List[str], str, None] = None,
    method: str = 'rolling',
    stride: int = 1,
    random_samples: Optional[int] = None,
    expand_step: int = 1,
    n_bootstrap: int = 0,
    progress_hook: Optional[Callable[[float], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    verbose: int = 1,
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    **kwargs
) -> Tuple[Dict[str, np.ndarray],
           Dict[str, np.ndarray],
           Optional[MinMaxScaler]]:
    """
    Generate input/target arrays for PINN models using various sampling
    methods (rolling, strided, random, expanding, bootstrap).

    Parameters
    ----------
    df : pd.DataFrame
        Full time-series data.
    time_col : str
        Name of the time coordinate column.
    subsidence_col : str
        Name of the subsidence target column.
    gwl_col : str
        Name of the groundwater level target column.
    dynamic_cols : list[str]
        Names of past-covariate columns.
    static_cols : list[str], optional
        Names of static feature columns.
    future_cols : list[str], optional
        Names of known-future feature columns.
    spatial_cols : (str, str), optional
        Tuple of (lon_col, lat_col) for spatial coords.
    group_id_cols : list[str], optional
        Column(s) identifying independent time-series groups.
    time_steps : int, default 12
        Look-back window length T.
    forecast_horizon : int, default 3
        Prediction horizon H.
    output_subsidence_dim : int, default 1
        Last-dim of subsidence target.
    output_gwl_dim : int, default 1
        Last-dim of GWL target.
    mode : {'pihal_like','tft_like'}, default 'pihal_like'
        Shapes the “future” window length for TFT vs. PIHALNet.
    normalize_coords : bool, default True
        Apply MinMax scaling to (t,x,y) across all sequences.
    cols_to_scale : list[str] or 'auto' or None
        Additional columns to scale via MinMax.
    method : {'rolling','strided','random','expanding','bootstrap'}
        Sequence-generation strategy.
    stride : int, default 1
        Step size for 'strided' sampling.
    random_samples : int, optional
        Number of random start indices for 'random' sampling.
    expand_step : int, default 1
        Increment size for 'expanding' sampling.
    n_bootstrap : int, default 0
        Number of blocks for 'bootstrap' sampling.
    progress_hook : callable, optional
        Called with float in [0,1] to report overall progress.
    stop_check : callable, optional
        If returns True, aborts sequence generation early.
    verbose : int, default 1
        Verbosity level (higher = more logs).
    _logger : logging.Logger or callable, optional
        Logger or print‐style function for vlog().
    **kwargs
        Passed to helper.

    Returns
    -------
    inputs : dict[str, np.ndarray]
        Contains 'coords', 'dynamic_features', optionally
        'static_features' and 'future_features'.
    targets : dict[str, np.ndarray]
        Contains 'subsidence' and 'gwl' arrays.
    coord_scaler : MinMaxScaler or None
        Fitted scaler for coords, if normalization was applied.
    """
    def _v(msg, lvl):
        vlog(msg, verbose=verbose, level=lvl, logger=_logger)

    # Optionally allow early abort
    if stop_check and stop_check():
        _v("Sequence generation aborted before start.", 1)
        return {}, {}, None

    # Split into groups
    groups = [g for _, g in df.groupby(group_id_cols)] \
             if group_id_cols else [df]

    sequences: List[Tuple[pd.DataFrame,int]] = []
    L = time_steps + forecast_horizon

    total_groups = len(groups)
    for gi, gdf in enumerate(groups):
        if stop_check and stop_check():
            _v("Sequence generation aborted.", 1)
            break

        length = len(gdf)
        max_start = length - L
        if max_start < 0:
            _v(f"Group {gi} too short (len={length}); skipping.", 2)
            continue

        # Determine start indices by method
        if method == 'rolling':
            starts = range(0, max_start + 1)
        elif method == 'strided':
            starts = range(0, max_start + 1, stride)
        elif method == 'random':
            all_starts = list(range(0, max_start + 1))
            if random_samples is None or random_samples > len(all_starts):
                starts = all_starts
            else:
                starts = np.random.choice(
                    all_starts,random_samples, replace=False
                    )
        elif method == 'expanding':
            starts = list(range(0, max_start + 1, expand_step))
        elif method == 'bootstrap':
            block_size = L
            blocks = list(range(0, length - block_size + 1, block_size))
            starts = np.random.choice(blocks, n_bootstrap, replace=True)
        else:
            raise ValueError(f"Unknown method '{method}'")

        for i in starts:
            sequences.append((gdf, int(i)))

        # Report group‐level progress
        if progress_hook:
            progress_hook((gi+1)/total_groups * 0.5)

    # Build arrays from these starts
    inputs, targets, coord_scaler = _build_from_starts(
        sequences,
        time_col, time_steps, forecast_horizon,
        subsidence_col, gwl_col,
        dynamic_cols, static_cols or [],
        future_cols or [], spatial_cols,
        mode, normalize_coords, cols_to_scale,
        output_subsidence_dim, output_gwl_dim,
        verbose, _logger
    )

    # Final progress
    if progress_hook:
        progress_hook(1.0)

    return inputs, targets, coord_scaler


def _build_from_starts(
    seqs: List[Tuple[pd.DataFrame,int]],
    time_col: str,
    T: int,
    H: int,
    subs_col: str,
    gwl_col: str,
    dyn_cols: List[str],
    stat_cols: List[str],
    fut_cols: List[str],
    spatial_cols: Optional[Tuple[str,str]],
    mode: str,
    norm_coords: bool,
    cols_to_scale: Union[List[str],str,None],
    out_sub_dim: int,
    out_gwl_dim: int,
    verbose: int = 1,
    _logger=None
) -> Tuple[Dict[str,np.ndarray],
           Dict[str,np.ndarray],
           Optional[MinMaxScaler]]:
    def _v(msg, lvl):
        vlog(msg, verbose=verbose, level=lvl, logger=_logger)

    N = len(seqs)
    _v(f"Building {N} sequences (T={T}, H={H})", 1)

    # Allocate arrays
    coords = np.zeros((N, H, 3), dtype=np.float32)
    dyn    = np.zeros((N, T, len(dyn_cols)), dtype=np.float32)
    stat   = np.zeros((N, len(stat_cols)), dtype=np.float32)\
             if stat_cols else None
    fut_len = T+H if mode=='tft_like' else H
    fut    = np.zeros((N, fut_len, len(fut_cols)),
                      dtype=np.float32) if fut_cols else None
    subs   = np.zeros((N, H, out_sub_dim), dtype=np.float32)
    gwl_a  = np.zeros((N, H, out_gwl_dim), dtype=np.float32)

    # Fit coordinate scaler if needed
    if norm_coords and spatial_cols:
        all_blocks = []
        for gdf, i in seqs:
            window = gdf.iloc[i:i+T+H]
            block = np.stack([
                window[time_col].values[:H],
                window[spatial_cols[0]].values[:H],
                window[spatial_cols[1]].values[:H]
            ], axis=1)
            all_blocks.append(block)
        flat = np.vstack(all_blocks)
        coord_scl = MinMaxScaler().fit(flat)
    else:
        coord_scl = None

    # Fill arrays
    for idx, (gdf, i) in enumerate(seqs):
        window = gdf.iloc[i:i+T+H]
        dyn[idx] = window.iloc[:T][dyn_cols].values
        if stat is not None:
            stat[idx] = gdf.iloc[0][stat_cols].values
        if fut is not None:
            if mode == 'tft_like':
                fut[idx] = window.iloc[:T+H][fut_cols].values
            else:
                fut[idx] = window.iloc[T:T+H][fut_cols].values

        # Coordinates
        block = np.stack([
            window[time_col].values[:H],
            (window[spatial_cols[0]].values[:H]
             if spatial_cols else np.zeros(H)),
            (window[spatial_cols[1]].values[:H]
             if spatial_cols else np.zeros(H))
        ], axis=1)
        coords[idx] = coord_scl.transform(block) if coord_scl else block

        # Targets
        subs[idx] = window.iloc[T:T+H][subs_col].values\
                    .reshape(H, out_sub_dim)
        gwl_a[idx] = window.iloc[T:T+H][gwl_col].values\
                     .reshape(H, out_gwl_dim)

        if verbose >= 3 and idx % 1000 == 0:
            _v(f"   → Processed sequence {idx+1}/{N}", 2)

    inputs = {'coords': coords, 'dynamic_features': dyn}
    if stat is not None:  inputs['static_features'] = stat
    if fut is not None:   inputs['future_features'] = fut
    targets = {'subsidence': subs, 'gwl': gwl_a}

    _v("Sequence building complete.", 1)
    return inputs, targets, coord_scl

@isdf 
def generate_ts_sequences(
    df: pd.DataFrame,
    time_col: str,
    dynamic_cols: List[str],
    static_cols: Optional[List[str]] = None,
    future_cols: Optional[List[str]] = None,
    spatial_cols: Optional[Tuple[str,str]] = None,
    group_id_cols: Optional[List[str]] = None,
    time_steps: int = 12,
    forecast_horizon: int = 1,
    normalize_coords: bool = True,
    cols_to_scale: Union[List[str], str, None] = None,
    method: str = 'rolling',
    stride: int = 1,
    random_samples: Optional[int] = None,
    expand_step: int = 1,
    n_bootstrap: int = 0,
    progress_hook: Optional[Callable[[float], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    verbose: int = 1,
    _logger: Optional[Callable[[str], None]] = None,
    **kwargs
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Optional[MinMaxScaler]
]:
    """
    Generate time-series windows for encoder/decoder and covariates.
    Supports rolling, strided, random, expanding, and bootstrap.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input frame with time and feature columns.
    time_col : str
        Name of the time coordinate column.
    dynamic_cols : list[str]
        Past-covariate columns for encoder inputs.
    static_cols : list[str] or None
        Static covariate columns, repeated per window.
    future_cols : list[str] or None
        Known-future covariates for decoder inputs.
    spatial_cols : tuple(str,str) or None
        (lon, lat) column names for spatial coords.
    group_id_cols : list[str] or None
        Columns to group by for independent series.
    time_steps : int
        Number of past steps (T) per window.
    forecast_horizon : int
        Number of future steps (H) per window.
    normalize_coords : bool
        If True, MinMax-scale spatial coords.
    cols_to_scale : list[str] or 'auto' or None
        Other columns to MinMax-scale.
    method : str
        'rolling','strided','random','expanding','bootstrap'.
    stride : int
        Step size for 'strided' windows.
    random_samples : int or None
        Number of random windows if method='random'.
    expand_step : int
        Increment for 'expanding' windows.
    n_bootstrap : int
        Number of bootstrap samples if method='bootstrap'.
    progress_hook : callable or None
        Receives float [0,1] as work progresses.
    stop_check : callable or None
        If returns True, aborts generation.
    verbose : int
        Verbosity level. >0 logs progress.
    _logger : callable or None
        Logger to use for messages.

    Returns
    -------
    inputs : dict of np.ndarray
        'encoder_inputs','static','future','coords'.
    targets : dict of np.ndarray
        'decoder_targets'.
    coord_scaler : MinMaxScaler or None
        Fitted scaler for coords, if normalized.

    Raises
    ------
    SequenceGeneratorError
        If no valid windows could be generated.
    """
    def _v(msg, lvl):
        vlog(msg, verbose=verbose, level=lvl, logger=_logger)

    # split into groups
    if group_id_cols:
        groups = [g for _, g in df.groupby(group_id_cols)]
    else:
        groups = [df]

    all_enc, all_dec = [], []
    all_stat, all_fut, all_coord = [], [], []
    L = time_steps + forecast_horizon
    total = 0

    for gdf in groups:
        if stop_check and stop_check():
            _v("Generation aborted by stop_check()", 1)
            break
        M = len(gdf)
        if M < L:
            _v(f"Group too small (len={M}), skip", 2)
            continue

        dyn = gdf[dynamic_cols].values
        win = sliding_window_view(dyn, window_shape=L, axis=0)
        idx = np.arange(win.shape[0])
        if method == 'strided':
            idx = idx[::stride]
        elif method == 'random':
            if random_samples and random_samples < len(idx):
                idx = np.random.choice(idx, random_samples,
                                       replace=False)
        elif method == 'expanding':
            idx = idx[::expand_step]
        elif method == 'bootstrap':
            idx = np.random.randint(0, len(idx), size=n_bootstrap)
        elif method != 'rolling':
            raise ValueError(f"Unknown method {method}")

        if idx.size == 0:
            continue

        enc = win[idx, :time_steps]
        dec = win[idx, time_steps:]
        all_enc.append(enc); all_dec.append(dec)

        if static_cols:
            st = gdf.iloc[0][static_cols].values.astype(np.float32)
            all_stat.append(np.repeat(st[None,:], len(idx), 0))

        if future_cols:
            fut = gdf[future_cols].values
            fw = sliding_window_view(fut, window_shape=forecast_horizon,
                                     axis=0)
            fw = fw[time_steps:time_steps+len(idx)]
            all_fut.append(fw)

        if spatial_cols:
            t,x,y = (gdf[c].values for c in 
                     (time_col,)+spatial_cols)
            coord = np.stack([
                sliding_window_view(t, L,0)[idx,time_steps:],
                sliding_window_view(x, L,0)[idx,time_steps:],
                sliding_window_view(y, L,0)[idx,time_steps:]
            ],1)
            all_coord.append(coord)

        total += len(idx)
        if progress_hook:
            progress_hook(min(1.0, total/ (len(df)//L + 1)))

    if not all_enc:
        raise SequenceGeneratorError(
            "No sequences generated (series too short)"
        )

    Xe = np.concatenate(all_enc,0)
    Xd = np.concatenate(all_dec,0)
    inputs = {'encoder_inputs': Xe}
    targets = {'decoder_targets': Xd}
    if static_cols:
        inputs['static'] = np.concatenate(all_stat,0)
    if future_cols:
        inputs['future'] = np.concatenate(all_fut,0)
    if spatial_cols:
        coords = np.concatenate(all_coord,0)
        if normalize_coords:
            flat = coords.reshape(-1,3)
            coord_scl = MinMaxScaler().fit(flat)
            inputs['coords'] = coord_scl.transform(
                flat).reshape(coords.shape)
        else:
            coord_scl = None
            inputs['coords'] = coords
    else:
        coord_scl = None

    _v(f"Generated {Xe.shape[0]} windows",1)
    return inputs, targets, coord_scl


def _generate_ts_sequences(
    series: Union[np.ndarray, pd.Series],
    time_steps: int = 12,
    forecast_horizon: int = 1,
    method: str = 'rolling',        # 'rolling','strided','random','expanding','bootstrap'
    stride: int = 1,                # for 'strided'
    random_samples: Optional[int] = None,  # for 'random'
    expand_step: int = 1,           # for 'expanding'
    n_bootstrap: int = 0,           # for 'bootstrap'
    shuffle: bool = False           # whether to shuffle final arrays
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (X, y) arrays from a 1D series.

    X has shape (N, time_steps),
    y has shape (N, forecast_horizon).

    Parameters
    ----------
    series
        1D array or pandas Series of length M.
    time_steps
        Number of past steps (T) for each input window.
    forecast_horizon
        Number of future steps (H) for each target window.
    method
        Sampling strategy:
        - 'rolling': every possible window,
        - 'strided': every `stride` windows,
        - 'random': random subset of starts,
        - 'expanding': windows starting at 0,expand by `expand_step`,
        - 'bootstrap': `n_bootstrap` random blocks of size T+H.
    stride
        Step size for 'strided'.
    random_samples
        Number of random windows if method='random'.
    expand_step
        Increment for 'expanding'.
    n_bootstrap
        Number of bootstrap samples if method='bootstrap'.
    shuffle
        Shuffle output windows.

    Returns
    -------
    X : ndarray, shape (N, time_steps)
    y : ndarray, shape (N, forecast_horizon)

    Raises
    ------
    ValueError
        If `method` is unknown or if the series is too short.
    """
    # ensure numpy array
    arr = series.values if isinstance(series, pd.Series) else np.asarray(series)
    M = arr.shape[0]
    L = time_steps + forecast_horizon
    max_start = M - L
    if max_start < 0:
        # not enough data for even one window
        return np.empty((0, time_steps)), np.empty((0, forecast_horizon))

    # rolling windows via stride_tricks
    windows = sliding_window_view(arr, window_shape=L)

    if method == 'rolling':
        idx = np.arange(max_start+1)
    elif method == 'strided':
        idx = np.arange(0, max_start+1, stride)
    elif method == 'random':
        all_idx = np.arange(max_start+1)
        if random_samples is None or random_samples >= len(all_idx):
            idx = all_idx
        else:
            idx = np.random.choice(all_idx, random_samples, replace=False)
    elif method == 'expanding':
        idx = np.arange(0, max_start+1, expand_step)
    elif method == 'bootstrap':
        # pick random blocks of length L
        idx = np.random.randint(0, max_start+1, size=n_bootstrap)
    else:
        raise ValueError(f"Unknown method: {method}")

    # slice out X and y
    X = windows[idx, :time_steps]
    y = windows[idx, time_steps:time_steps+forecast_horizon]

    if shuffle:
        p = np.random.permutation(len(X))
        X, y = X[p], y[p]

    return X, y

class SequenceGeneratorError(RuntimeError):
    """Raised when no sequence can be generated with the given settings."""

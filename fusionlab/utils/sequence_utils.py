# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

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

from ..decorators import isdf 
from .generic_utils import vlog

__all__ = [
    'check_sequence_feasibility', 'get_sequence_counts',
    'generate_pinn_sequences', 'generate_ts_sequences'
 ]

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
            f"Each trajectory needs ≥ {min_len} consecutive records "
            f"(time_steps={time_steps}, horizon={forecast_horizon}), "
            f"but the longest has only {longest}.\n"
            "→ Reduce `time_steps` / `forecast_horizon`, "
            "or supply more data."
        )
        _v("❌ " + msg.splitlines()[0], lvl=1)

        if error == "raise":
            raise SequenceGeneratorError(msg)
        if error == "warn":
            warnings.warn(msg, UserWarning, stacklevel=2)
        return False, counts

    _v(f"✅ Feasible: {total_sequences} sequences possible.", lvl=1)
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
            _v("⚠️  pyarrow not installed — reverting to 'vectorized'.", lvl=1)
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
                f"{size_str} pts → {n_seq} seq.",
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

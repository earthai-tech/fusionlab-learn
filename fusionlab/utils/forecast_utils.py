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
from typing import ( 
    Dict, 
    Iterable, 
    List, 
    Union, 
    Any, 
    Optional,
    Tuple , 
    Callable
)
import pandas as pd

from .._fusionlog import fusionlog 
from ..core.handlers import columns_manager 
from ..core.checks import check_spatial_columns, check_empty  
from ..core.io import is_data_readable  
from ..decorators import isdf 

from .generic_utils import vlog 
from .validator import is_frame 

logger = fusionlog().get_fusionlab_logger(__name__)

__all__= [ 
     'detect_forecast_type',
     'format_forecast_dataframe',
     'get_value_prefixes',
     'get_value_prefixes_in',
     'pivot_forecast_dataframe', 
     'get_step_names'
     ]

_DIGIT_RE = re.compile(r"\d+")


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
         level=1, verbose=verbose)

    if savefile:
        try:
            vlog(f"Saving DataFrame to '{savefile}'.",
                 level=1, verbose=verbose)
            save_dir = os.path.dirname(savefile)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            df_wide.to_csv(savefile, index=False)
            vlog("Save successful.", level=2, verbose=verbose)
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




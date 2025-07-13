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
    Callable, 
)
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 

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
     'get_step_names', 
     'stack_quantile_predictions', 
     'adjust_time_predictions', 
     
     ]

_DIGIT_RE = re.compile(r"\d+")


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

    # --- 1. Adjust time before scaling ---
    if forecast_horizon is not None:
        # Check if time_col is integer (year)
        if pd.api.types.is_integer_dtype(df_scaled[time_col]):
            # If it's an integer (year), we can simply add the forecast_horizon
            df_scaled[time_col] = df_scaled[time_col] + forecast_horizon
            vlog(f"Time column adjusted with forecast horizon: {forecast_horizon}",
                 verbose=verbose, level=4, logger=_logger)
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
        coord_scaler = MinMaxScaler()
        df_scaled[coord_cols] = coord_scaler.fit_transform(
            df_scaled[coord_cols]
        )

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



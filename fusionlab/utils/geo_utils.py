# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import os
from typing import List, Optional, Dict, Any
import warnings 

import pandas as pd
import numpy as np

from .._fusionlog import fusionlog 
from ..core.checks import ( 
    exist_features, 
    check_spatial_columns 
    )
from ..core.handlers import columns_manager
from ..core.io import SaveFile 

logger = fusionlog().get_fusionlab_logger(__name__)

__all__ = [
    "resolve_spatial_columns", "interpolate_temporal_gaps", 
    "augment_spatiotemporal_data"]


def resolve_spatial_columns(
    df,
    spatial_cols=None,
    lon_col=None,
    lat_col=None
):
    """
    Helper to validate and resolve spatial columns.

    Accepts either explicit lon/lat columns or a
    list of spatial_cols. Returns (lon_col, lat_col).

    - If lon_col and lat_col are both provided, they
      take precedence (warn if spatial_cols also set).
    - Else if spatial_cols is provided, it must yield
      exactly two column names.
    - Otherwise, error is raised.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame for feature checks.
    spatial_cols : list[str] or None
        Two-element list of [lon_col, lat_col].
    lon_col : str or None
        Name of longitude column.
    lat_col : str or None
        Name of latitude column.

    Returns
    -------
    (lon_col, lat_col) : tuple of str
        Validated column names for longitude and
        latitude.

    Raises
    ------
    ValueError
        If neither lon/lat nor valid spatial_cols is
        provided, or if spatial_cols len != 2.
    """
    # Case 1: explicit lon/lat
    if lon_col is not None and lat_col is not None:
        if spatial_cols:
            warnings.warn(
                "Both lon_col/lat_col and spatial_cols set;"
                " spatial_cols will be ignored.",
                UserWarning
            )
        exist_features(
            df,
            features=[lon_col, lat_col],
            name="Longitude/Latitude"
        )
        return lon_col, lat_col

    # Case 2: spatial_cols provided
    if spatial_cols:
        spatial_cols = columns_manager(
            spatial_cols,
            empty_as_none=False
        )
        check_spatial_columns(
            df,
            spatial_cols=spatial_cols
        )
        exist_features(
            df,
            features=spatial_cols,
            name="Spatial columns"
        )
        if len(spatial_cols) != 2:
            raise ValueError(
                "spatial_cols must contain exactly two"
                " column names"
            )
        lon, lat = spatial_cols
        return lon, lat

    # Neither provided
    raise ValueError(
        "Either lon_col & lat_col, or spatial_cols"
        " must be provided."
    )

@SaveFile 
def interpolate_temporal_gaps(
    series_df: pd.DataFrame,
    time_col: str,
    value_cols: List[str],
    freq: Optional[str] = None,
    method: str = 'linear',
    order: Optional[int] = None,
    fill_limit: Optional[int] = None,
    fill_limit_direction: str = 'forward', 
    savefile: Optional[str] =None, 
) -> pd.DataFrame:
    r"""
    Interpolates missing values in specified columns of a time series
    DataFrame.

    This function is designed to work on a DataFrame representing a time
    series for a single spatial group (e.g., one monitoring location),
    sorted by time. If :code:`freq` is provided, the DataFrame’s index is
    first reindexed to that frequency, which can create NaN values for
    missing time steps. These NaNs, along with any pre-existing NaNs in
    :code:`value_cols`, are then interpolated.

    Let :math:`t_1 < t_2 < \dots < t_n` be the original timestamps. If
    :code:`freq` yields a new index :math:`\{t_i'\}` that includes times
    not in the original, NaNs appear at those :math:`t_i'`. Then for each
    column :math:`v` in :math:`\{\text{value\_cols}\}`, we perform:

    .. math::
       v(t) \;=\; 
       \begin{cases}
         \text{interpolate}(v,\;t;\;\text{method},\;\dots)
         & \text{for } t \in \{t_i'\}\,,\\
         v(t) & \text{if } t \in \{t_1,\dots,t_n\}\text{ and not NaN.}
       \end{cases}

    Parameters
    ----------
    series_df : pd.DataFrame
        Input DataFrame for a single time series, ideally sorted by
        :code:`time_col`. The :code:`time_col` should be convertible to
        datetime.
    time_col : str
        Name of the column containing datetime information.
    value_cols : List[str]
        List of column names whose missing values (NaNs) should be
        interpolated.
    freq : str or None, default None
        The desired frequency for the time series (e.g., 'D' for daily,
        'MS' for month start, 'AS' for year start). If provided, the
        DataFrame is reindexed to this frequency before interpolation.
        This helps identify and fill gaps where entire time steps are
        missing.
    method : str, default 'linear'
        Interpolation method to use. Passed to
        `pandas.DataFrame.interpolate()`. Common methods: 'linear',
        'time', 'polynomial', 'spline'. If 'polynomial' or 'spline',
        :code:`order` must be specified.
    order : int or None, default None
        Order for polynomial or spline interpolation. Required if
        :code:`method` is 'polynomial' or 'spline'.
    fill_limit : int or None, default None
        Maximum number of consecutive NaNs to fill. Passed to
        `pandas.DataFrame.interpolate()`.
    fill_limit_direction : str, default 'forward'
        Direction for :code:`fill_limit` ('forward', 'backward',
        'both'). Passed to `pandas.DataFrame.interpolate()`.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns interpolated. If :code:`freq`
        was used, the DataFrame will have a DatetimeIndex. Other columns
        not in :code:`value_cols` will be forward-filled after reindexing
        if :code:`freq` is set, to propagate their last known values into
        new empty rows.

    Raises
    ------
    TypeError
        If :code:`series_df` is not a DataFrame or if
        :code:`value_cols` is not a list of strings.
        Also if :code:`time_col` is missing from the DataFrame.
    ValueError
        If :code:`order` is required but not provided for 'polynomial'
        or 'spline'.

    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.utils.geo_utils import interpolate_temporal_gaps
    >>> # Sample time series with missing dates
    >>> df = pd.DataFrame({
    ...     'date': ['2020-01-01', '2020-01-03', '2020-01-06'],
    ...     'value': [1.0, None, 4.0]
    ... })
    >>> df
             date  value
    0 2020-01-01    1.0
    1 2020-01-03    NaN
    2 2020-01-06    4.0
    >>> result = interpolate_temporal_gaps(
    ...     df, time_col='date', value_cols=['value'], freq='D'
    ... )
    >>> result.head()
             date  value
    0 2020-01-01    1.0
    1 2020-01-02    2.0
    2 2020-01-03    3.0
    3 2020-01-04    3.0
    4 2020-01-05    3.5

    Notes
    -----
    - Ensure :code:`series_df` pertains to a single spatial group and is
      sorted by time for meaningful interpolation.
    - The 'time' method for interpolation requires the index to be a
      DatetimeIndex.
    - Polynomial or spline methods require :code:`order` to be specified.

    See Also
    --------
    pandas.DataFrame.interpolate : Core interpolation method.
    pandas.DataFrame.asfreq : Reindex DataFrame to fixed frequency.
    """
    if not isinstance(series_df, pd.DataFrame):
        raise TypeError("Input 'series_df' must be a pandas DataFrame.")
    if (
        not isinstance(value_cols, list)
        or not all(isinstance(col, str) for col in value_cols)
    ):
        raise TypeError("'value_cols' must be a list of strings.")
    if time_col not in series_df.columns:
        raise ValueError(
            f"Time column '{time_col}' not found in DataFrame."
        )

    df_interpolated = series_df.copy()

    # Convert time column to datetime and set as index for interpolation
    try:
        df_interpolated[time_col] = pd.to_datetime(
            df_interpolated[time_col]
        )
    except Exception as e:
        logger.error(
            f"Could not convert time column '{time_col}' to datetime: {e}"
        )
        raise

    df_interpolated = df_interpolated.sort_values(by=time_col)
    # Store original index name if it exists, to restore it later if needed.
    original_index_name = df_interpolated.index.name
    df_interpolated = df_interpolated.set_index(time_col, drop=False)

    original_columns = series_df.columns.tolist()  # Keep track of original columns

    if freq:
        try:
            # Reindex to the specified frequency. This may create new rows
            # with NaNs.
            df_interpolated = df_interpolated.asfreq(freq)

            # Forward fill other columns that are not being interpolated
            # to propagate their values into the new empty rows created by
            # asfreq.
            other_cols = [
                col
                for col in original_columns
                if col not in value_cols and col != time_col
            ]
            if other_cols:
                df_interpolated[other_cols] = df_interpolated[
                    other_cols
                ].ffill()

            # The time_col itself might become NaN in new rows if it was
            # dropped when setting index and then re-added. Ensure it's
            # filled from the new index.
            if time_col not in df_interpolated.columns:
                # if it was dropped
                df_interpolated[time_col] = df_interpolated.index
            else:
                # if it was kept, ensure it is also ffilled from the new
                # index
                df_interpolated[time_col] = df_interpolated.index

        except ValueError as e:
            logger.error(
                f"Invalid frequency string: '{freq}'. Error: {e}. "
                "Skipping reindexing."
            )
            # If reindexing fails, return a copy of the original to avoid
            # partial modification
            return series_df.copy()
        except Exception as e:
            logger.error(
                f"Error during reindexing with frequency '{freq}': {e}"
            )
            return series_df.copy()

    # Perform interpolation
    # Check if value_cols exist after potential reindexing
    missing_value_cols = [
        vc
        for vc in value_cols
        if vc not in df_interpolated.columns
    ]
    if missing_value_cols:
        logger.warning(
            f"Value columns not found in DataFrame after potential "
            f"reindexing: {missing_value_cols}. Skipping interpolation for "
            f"these."
        )
        value_cols = [
            vc for vc in value_cols if vc in df_interpolated.columns
        ]

    if not value_cols:
        logger.warning(
            "No valid value_cols left to interpolate. Returning processed "
            "DataFrame."
        )
        # If freq was applied, df_interpolated is reindexed and possibly
        # ffilled
        # If not, it's a copy of series_df with sorted datetime index
        df_to_return = df_interpolated.reset_index(drop=True)
        if original_index_name:  # Restore original index name if it existed
            df_to_return.index.name = original_index_name
        return df_to_return

    interpolate_kwargs = {
        'method': method,
        'limit': fill_limit,
        'limit_direction': fill_limit_direction
    }
    if method in ['polynomial', 'spline']:
        if order is None:
            raise ValueError(
                f"Order must be specified for '{method}' interpolation."
            )
        interpolate_kwargs['order'] = order
    if method == 'time':
        if not isinstance(df_interpolated.index, pd.DatetimeIndex):
            logger.warning(
                "Interpolation method 'time' requires a DatetimeIndex. "
                "Consider setting `freq` or ensuring `time_col` is a "
                "DatetimeIndex."
            )
            # Fallback or let pandas handle it/error out

    try:
        df_interpolated[value_cols] = df_interpolated[
            value_cols
        ].interpolate(**interpolate_kwargs)
    except Exception as e:
        logger.error(
            f"Error during interpolation of columns {value_cols}: {e}"
        )
        # Return the DataFrame as it is before the erroring interpolation
        # attempt
        df_to_return = df_interpolated.reset_index(drop=True)
        if original_index_name:
            df_to_return.index.name = original_index_name
        return df_to_return

    # Reset index to restore time_col as a column and get default integer index
    df_interpolated = df_interpolated.reset_index(drop=True)
    if original_index_name:  # Restore original index name if it existed
        df_interpolated.index.name = original_index_name

    # Ensure original column order if possible, and all original columns
    # are present
    # This is important if asfreq added/removed rows and columns were
    # just ffilled
    final_cols_ordered = []
    for col in original_columns:
        if col in df_interpolated.columns:
            final_cols_ordered.append(col)
    # Add any new columns that might have been created (e.g. if time_col
    # was index and then reset)
    # though current logic tries to preserve original_columns.
    for col in df_interpolated.columns:
        if col not in final_cols_ordered:
            final_cols_ordered.append(col)

    return df_interpolated[final_cols_ordered]


@SaveFile 
def augment_series_features(
    series_df: pd.DataFrame,
    feature_cols: List[str],
    noise_level: float = 0.01,
    noise_type: str = 'gaussian',
    random_seed: Optional[int] = None, 
    savefile: Optional[str] =None,
) -> pd.DataFrame:
    r"""
    Augments specified feature columns in a time series DataFrame by adding noise.

    This function is typically applied to sequences that are already long
    enough, to create more training examples and improve model robustness.

    Suppose :math:`x_i` are original feature values for one column. Then:

    - For Gaussian noise:
      .. math::
         \text{noise}_i \sim \mathcal{N}\bigl(0,\; \sigma_x \times
         \text{noise\_level}\bigr),
         \quad \sigma_x = \mathrm{std}(x_i).
      The augmented values are :math:`x_i + \text{noise}_i`.

    - For Uniform noise:
      .. math::
         \text{range}_x = \max(x_i) - \min(x_i), \quad
         \text{noise}_i \sim \mathcal{U}\Bigl(
         -\tfrac{\text{range}_x \times \text{noise\_level}}{2},\;
         \tfrac{\text{range}_x \times \text{noise\_level}}{2}\Bigr).
      The augmented values are :math:`x_i + \text{noise}_i`.

    Parameters
    ----------
    series_df : pd.DataFrame
        Input DataFrame representing one or more time series.
    feature_cols : List[str]
        List of column names (features) to which noise will be added.
    noise_level : float, default 0.01
        Magnitude of the noise.
        - For 'gaussian': standard deviation of the noise relative to
          feature's std.
        - For 'uniform': half-width of the uniform distribution relative
          to feature's range.
    noise_type : str, default 'gaussian'
        Type of noise to add. Options: 'gaussian', 'uniform'.
    random_seed : int or None, default None
        Seed for the random number generator for reproducible results.

    Returns
    -------
    pd.DataFrame
        DataFrame with noise added to the specified feature columns.

    Raises
    ------
    ValueError
        If `feature_cols` are not in `series_df` or `noise_type` is invalid.
    TypeError
        If inputs are not of the expected type.

    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.utils.geo_utils import augment_series_features
    >>> df = pd.DataFrame({
    ...     'x': [10.0, 12.0, 15.0, 13.0],
    ...     'y': [100, 110, 105, 115]
    ... })
    >>> # Add Gaussian noise at 5% level to column 'x'
    >>> df_aug = augment_series_features(df, ['x'], noise_level=0.05,
    ...                                  noise_type='gaussian',
    ...                                  random_seed=42)
    >>> df_aug['x']
    0    10.248357
    1    11.930868
    2    15.323844
    3    12.761515
    Name: x, dtype: float64

    Notes
    -----
    - If a feature column has zero variance or NaN range, no noise is added
      and a debug log is emitted.
    - Non-numeric columns are skipped with a warning.

    See Also
    --------
    pandas.DataFrame.sample : Random sampling methods.
    sklearn.utils.resample : Resampling utilities for data augmentation.
    """
    if not isinstance(series_df, pd.DataFrame):
        raise TypeError("Input 'series_df' must be a pandas DataFrame.")
    if (
        not isinstance(feature_cols, list)
        or not all(isinstance(col, str) for col in feature_cols)
    ):
        raise TypeError("'feature_cols' must be a list of strings.")

    missing_cols = [col for col in feature_cols if col not in series_df.columns]
    if missing_cols:
        raise ValueError(
            f"Feature columns not found in DataFrame: {missing_cols}"
        )

    if noise_type not in ['gaussian', 'uniform']:
        raise ValueError(
            f"Invalid noise_type: '{noise_type}'. "
            "Choose 'gaussian' or 'uniform'."
        )

    if random_seed is not None:
        np.random.seed(random_seed)

    df_augmented = series_df.copy()

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df_augmented[col]):
            logger.warning(
                f"Column '{col}' is not numeric. "
                "Skipping noise augmentation for this column."
            )
            continue

        feature_values = df_augmented[col].values
        if noise_type == 'gaussian':
            # Scale noise by the standard deviation of the feature
            col_std = np.std(feature_values)
            if col_std == 0 or np.isnan(col_std):  # Avoid division by zero
                noise = 0
                if np.isnan(col_std):
                    logger.debug(
                        f"Std of column '{col}' is NaN. Adding zero noise."
                    )
            else:
                noise = np.random.normal(
                    0, col_std * noise_level, size=feature_values.shape
                )
        elif noise_type == 'uniform':
            # Scale noise by the range of the feature
            col_min = np.min(feature_values)
            col_max = np.max(feature_values)
            col_range = col_max - col_min
            if col_range == 0 or np.isnan(col_range):  # Avoid zero range
                noise = 0
                if np.isnan(col_range):
                    logger.debug(
                        f"Range of column '{col}' is NaN. Adding zero noise."
                    )
            else:
                noise = np.random.uniform(
                    -col_range * noise_level / 2.0,
                    col_range * noise_level / 2.0,
                    size=feature_values.shape
                )

        df_augmented[col] = (
            feature_values + noise.astype(feature_values.dtype)
        )  # Ensure dtype consistency

    return df_augmented

@SaveFile 
def augment_spatiotemporal_data(
    df: pd.DataFrame,
    mode: str,
    group_by_cols: Optional[List[str]] = None,
    time_col: Optional[str] = None,
    value_cols_interpolate: Optional[List[str]] = None,
    feature_cols_augment: Optional[List[str]] = None,
    interpolation_kwargs: Optional[Dict[str, Any]] = None,
    augmentation_kwargs: Optional[Dict[str, Any]] = None,
    savefile: Optional[str] =None,
    verbose: bool = False
) -> pd.DataFrame:
    r"""
    Applies temporal interpolation and/or feature augmentation to a
    spatiotemporal DataFrame.

    This function can perform one of three operations on each group of
    the DataFrame:

    1. :math:`\text{interpolate}` only: fill temporal gaps via
       `interpolate_temporal_gaps`.
    2. :math:`\text{augment\_features}` only: add noise to features via
       `augment_series_features`.
    3. :math:`\text{both}`: first interpolate, then augment features.

    Let :math:`G` be the set of groups defined by
    :code:`group_by_cols`. For each group :math:`g \in G`, if mode
    includes interpolation, we compute:

    .. math::
       \text{interpolated\_df}_g = 
       \text{interpolate\_temporal\_gaps}(
         \text{series\_df}_g,\;\dots
       )

    Then if mode includes augmentation, we compute:

    .. math::
       \text{augmented\_df}_g = 
       \text{augment\_series\_features}(
         \text{interpolated\_df}_g,\;\dots
       )

    Finally, all processed groups are concatenated:

    .. math::
       \text{result} = \bigcup_{g \in G} \text{processed\_df}_g.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame (e.g., Zhongshan data).
    mode : str
        The augmentation mode. Options:
        - 'interpolate': Applies only
          :func:`interpolate_temporal_gaps`.
        - 'augment_features': Applies only
          :func:`augment_series_features`.
        - 'both': Applies `interpolate_temporal_gaps` first, then
          `augment_series_features`.
    group_by_cols : list of str or None, default None
        Columns to group by for temporal interpolation (e.g.,
        ['longitude', 'latitude']). Required if mode includes
        interpolation.
    time_col : str or None, default None
        Name of the time column. Required if mode includes
        interpolation.
    value_cols_interpolate : list of str or None, default None
        Columns to interpolate. Required if mode includes
        interpolation.
    feature_cols_augment : list of str or None, default None
        Columns for noise augmentation. Required if mode includes
        augmentation.
    interpolation_kwargs : dict or None, default None
        Keyword arguments passed to
        :func:`interpolate_temporal_gaps` (e.g., {'freq': 'AS'}).
    augmentation_kwargs : dict or None, default None
        Keyword arguments passed to
        :func:`augment_series_features` (e.g., {'noise_level': 0.02}).
    savefile: str, optional, 
        Save the dataframe into the csv format by default. 
    verbose : bool, default False
        If True, prints progress messages (via print). Otherwise,
        relies on logger.

    Returns
    -------
    pd.DataFrame
        The processed DataFrame. Groups are reassembled in original
        order of grouping and then concatenated.

    Raises
    ------
    ValueError
        If `mode` is invalid or required parameters for the selected
        mode are missing.

    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.utils.geo_utils import augment_spatiotemporal_data
    >>> df = pd.DataFrame({
    ...     'lon': [0, 0, 1, 1],
    ...     'lat': [0, 0, 1, 1],
    ...     'date': ['2020-01-01', '2020-01-03',
    ...              '2020-01-01', '2020-01-04'],
    ...     'value': [1.0, None, 2.0, None]
    ... })
    >>> result = augment_spatiotemporal_data(
    ...     df,
    ...     mode='both',
    ...     group_by_cols=['lon', 'lat'],
    ...     time_col='date',
    ...     value_cols_interpolate=['value'],
    ...     feature_cols_augment=['value'],
    ...     interpolation_kwargs={'freq': 'D'},
    ...     augmentation_kwargs={'noise_level': 0.05,
    ...                          'noise_type': 'gaussian',
    ...                          'random_seed': 0}
    ... )
    >>> 'value' in result.columns
    True

    Notes
    -----
    - Ensure `df` contains all columns in `group_by_cols` if mode
      includes interpolation.
    - Date column must be convertible to datetime.
    - Groups are processed independently, then concatenated.

    See Also
    --------
    interpolate_temporal_gaps : Fill temporal gaps per group.
    augment_series_features : Add noise to feature columns.
    """
    if mode not in ['interpolate', 'augment_features', 'both']:
        raise ValueError(
            "Invalid mode. Choose from 'interpolate', "
            "'augment_features', or 'both'."
        )

    processed_df = df.copy()
    interpolation_kwargs = interpolation_kwargs or {}
    augmentation_kwargs = augmentation_kwargs or {}

    if mode in ['interpolate', 'both']:
        if not all([group_by_cols, time_col, value_cols_interpolate]):
            raise ValueError(
                "For 'interpolate' or 'both' mode, 'group_by_cols', "
                "'time_col', and 'value_cols_interpolate' must be provided."
            )

        logger.info(
            f"Starting temporal interpolation for groups: "
            f"{group_by_cols}..."
        )
        interpolated_groups = []
        grouped = processed_df.groupby(
            group_by_cols, group_keys=True
        )  # group_keys=True for safety

        num_groups = len(grouped)
        for i, (name, group_df) in enumerate(grouped):
            if verbose:  # Simple print for progress, logger can be used too
                print(
                    f"  Interpolating group {i+1}/{num_groups}: {name}",
                    end='\r'
                )

            interpolated_group = interpolate_temporal_gaps(
                series_df=group_df,
                time_col=time_col,
                value_cols=value_cols_interpolate,
                **interpolation_kwargs
            )
            interpolated_groups.append(interpolated_group)

        if verbose:
            print("\nInterpolation of all groups complete.")

        if interpolated_groups:
            processed_df = pd.concat(
                interpolated_groups, ignore_index=True
            )
            logger.info("Temporal interpolation applied to all groups.")
        else:
            logger.warning(
                "No groups found for interpolation or all groups were empty."
            )

    if mode in ['augment_features', 'both']:
        if not feature_cols_augment:
            raise ValueError(
                "For 'augment_features' or 'both' mode, "
                "'feature_cols_augment' must be provided."
            )
        logger.info(
            f"Starting feature augmentation for columns: "
            f"{feature_cols_augment}..."
        )
        processed_df = augment_series_features(
            series_df=processed_df,
            feature_cols=feature_cols_augment,
            **augmentation_kwargs
        )
        logger.info("Feature augmentation applied.")

    return processed_df

    
def augment_city_spatiotemporal_data(
    df: pd.DataFrame,
    city: str,
    mode: str = 'interpolate',
    group_by_cols: Optional[List[str]] = None,  
    time_col: Optional[str] = None,            
    value_cols_interpolate: Optional[List[str]] = None,
    feature_cols_augment: Optional[List[str]] = None,  
    interpolation_config: Optional[Dict[str, Any]] = None,
    augmentation_config: Optional[Dict[str, Any]] = None,
    target_name: Optional[str] = None,
    interpolate_target: bool = False,
    verbose: bool = False,  # Changed default to False for less noisy library use
    coordinate_precision: Optional[int] = None,
    savefile: Optional[str] = None,
) -> pd.DataFrame:
    r"""
    A robust and versatile function to augment spatiotemporal data for
    Nansha or Zhongshan, with control over coordinate precision and
    augmentation parameters.

    This wrapper calls `augment_spatiotemporal_data` to apply temporal
    interpolation and/or feature augmentation. Let :math:`G` be groups
    defined by :code:`group_by_cols` and let each group :math:`g \in G`
    have its own sequence :math:`\{(t_i, \mathbf{x}_i)\}`. Then:

    1. If mode includes 'interpolate', for each group :math:`g`:
       .. math::
          \text{interp}_g = \texttt{interpolate\_temporal\_gaps}\bigl(
            \text{series\_df}_g,\;\dots\bigr).

    2. If mode includes 'augment_features', for each group :math:`g`:
       .. math::
          \text{aug}_g = \texttt{augment\_series\_features}\bigl(
            \text{interp}_g\ {\text{or}}\ \text{series\_df}_g,\;\dots\bigr).

    Finally, all processed groups are concatenated:
    .. math::
       \text{result} = \bigcup_{g \in G} \text{processed\_df}_g.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must include 'longitude', 'latitude', 'year',
        and relevant value/feature columns.
    city : str
        City identifier: 'nansha' or 'zhongshan'. Used for potential
        city-specific default logic in the future, currently for validation.
    mode : str, default 'interpolate'
        Augmentation mode. Options:
        - 'interpolate': Applies only `interpolate_temporal_gaps`.
        - 'augment_features': Applies only `augment_series_features`.
        - 'both': Applies interpolation first, then feature augmentation.
    group_by_cols : list of str or None, default ['longitude', 'latitude']
        Columns to group by for interpolation.
    time_col : str or None, default 'year'
        Time column name for interpolation.
    value_cols_interpolate : list of str or None, default None
        Columns to interpolate. If None, defaults are determined based on
        numeric columns, excluding IDs, target (unless
        :code:`interpolate_target` is True), and common non-driver
        categorical-like columns.
    feature_cols_augment : list of str or None, default None
        Columns to augment with noise. If None, defaults are determined
        based on numeric columns, excluding IDs and the target.
    interpolation_config : dict or None, default {'freq': 'AS',
        'method': 'linear'}
        Keyword arguments for `interpolate_temporal_gaps`.
    augmentation_config : dict or None, default {'noise_level': 0.01,
        'noise_type': 'gaussian'}
        Keyword arguments for `augment_series_features`.
    target_name : str or None, default None
        Name of the target variable column (e.g., 'subsidence'). Used to
        conditionally include/exclude it from default interpolation/
        augmentation.
    interpolate_target : bool, default False
        If True and :code:`target_name` is provided, the target variable
        will be included in the default list for
        :code:`value_cols_interpolate`.
    verbose : bool, default False
        If True, prints progress messages from
        `augment_spatiotemporal_data`.
    coordinate_precision : int or None, default None
        Number of decimal places to round longitude/latitude before
        grouping. If None, no rounding.
    savefile : str or None, default None
        If a path is provided, the augmented DataFrame is saved as a CSV.

    Returns
    -------
    pd.DataFrame
        The augmented DataFrame.

    Raises
    ------
    ValueError
        If :code:`city` is invalid, :code:`mode` is invalid, or required
        parameters for the selected :code:`mode` are missing.
    TypeError
        If :code:`df` is not a DataFrame or other inputs have incorrect
        types.

    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.utils.geo_utils import augment_city_spatiotemporal_data
    >>> df_zh = pd.DataFrame({
    ...     'longitude': [113.38, 113.38],
    ...     'latitude': [22.77, 22.77],
    ...     'year': [2018, 2020],
    ...     'GWL': [5.5, 39.4],
    ...     'rainfall_mm': [311, 177],
    ...     'subsidence': [12.5, 12.8]
    ... })
    >>> augmented = augment_city_spatiotemporal_data(
    ...     df=df_zh,
    ...     city='zhongshan',
    ...     mode='both',
    ...     target_name='subsidence',
    ...     interpolate_target=False,
    ...     interpolation_config={'freq': 'AS', 'method': 'linear'},
    ...     augmentation_config={'noise_level': 0.01,
    ...                          'noise_type': 'gaussian',
    ...                          'random_seed': 42},
    ...     coordinate_precision=4,
    ...     verbose=True,
    ...     savefile=None
    ... )
    >>> augmented.shape
    (2, 6)

    Notes
    -----
    - Ensure :code:`df` contains 'longitude', 'latitude', and
      :code:`time_col` columns.
    - If :code:`coordinate_precision` is set, longitude/latitude are
      rounded to avoid overly granular groups.
    - Default interpolation columns exclude categorical or target
      columns unless specified by :code:`interpolate_target`.
    - Default augmentation columns exclude the target column.

    See Also
    --------
    interpolate_temporal_gaps : Fill temporal gaps per group.
    augment_spatiotemporal_data : Core function for group-level
        spatiotemporal augmentation.
    augment_series_features : Add noise to numeric features.
    """
    # Validate DataFrame type
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    # Validate city parameter
    city_lower = city.strip().lower()
    if city_lower not in ['nansha', 'zhongshan']:
        raise ValueError("`city` must be 'nansha' or 'zhongshan'.")
    logger.info(f"Processing data for city: {city_lower.capitalize()}.")

    # Copy input to avoid modifying original
    df_city = df.copy()

    # Ensure 'year' (or specified time_col) is datetime
    # Default time_col for processing, can be overridden by user
    _time_col = time_col or 'year'
    if _time_col not in df_city.columns:
        raise ValueError(f"Time column '{_time_col}' not found in DataFrame.")
    try:
        # Attempt to convert to datetime if not already.
        # If 'year' is int like 2015, format='%Y' makes it YYYY-01-01
        first_val = df_city[_time_col].iloc[0]
        if pd.api.types.is_integer(first_val) and 1000 < first_val < 3000:
            df_city[_time_col] = pd.to_datetime(df_city[_time_col], format='%Y')
        else:
            df_city[_time_col] = pd.to_datetime(df_city[_time_col])
        logger.debug(f"Ensured '{_time_col}' is datetime.")
    except Exception as e:
        logger.warning(
            f"Could not convert time column '{_time_col}' to datetime: {e}. "
            "Proceeding, but interpolation might behave unexpectedly."
        )

    # --- Define default parameters ---
    _group_by_cols = group_by_cols or ['longitude', 'latitude']

    # Default columns to exclude from auto-selection
    default_exclude_cols = set(_group_by_cols + [_time_col])
    # Add known categorical or ID-like columns (expand as needed)
    known_non_numeric_or_id_cols = {
        'geology', 'density_tier', 'subsidence_intensity',
        'density_concentration', 'rainfall_category',
        'building_concentration', 'soil_thickness'
    }
    default_exclude_cols.update(known_non_numeric_or_id_cols)

    if target_name and target_name in df_city.columns:
        if not interpolate_target:
            # Exclude target from interpolation by default
            default_exclude_cols.add(target_name)

    numeric_cols = df_city.select_dtypes(include=np.number).columns.tolist()

    _value_cols_interpolate = value_cols_interpolate
    if _value_cols_interpolate is None:
        _value_cols_interpolate = [
            col for col in numeric_cols if col not in default_exclude_cols
        ]
        if interpolate_target and target_name and target_name in numeric_cols:
            _value_cols_interpolate.append(target_name)
        logger.info(
            f"Defaulting 'value_cols_interpolate' to: "
            f"{_value_cols_interpolate}"
        )

    _feature_cols_augment = feature_cols_augment
    if _feature_cols_augment is None:
        augment_exclude_cols = default_exclude_cols.copy()
        if target_name:
            augment_exclude_cols.add(target_name)
        _feature_cols_augment = [
            col for col in numeric_cols if col not in augment_exclude_cols
        ]
        logger.info(
            f"Defaulting 'feature_cols_augment' to: "
            f"{_feature_cols_augment}"
        )

    _interpolation_config = interpolation_config or {
        'freq': 'AS',
        'method': 'linear'
    }
    _augmentation_config = augmentation_config or {
        'noise_level': 0.01,
        'noise_type': 'gaussian',
        'random_seed': None
    }

    # --- Coordinate Precision ---
    if coordinate_precision is not None:
        if not (isinstance(coordinate_precision, int) and
                coordinate_precision >= 0):
            raise ValueError("'coordinate_precision' must be a "
                             "non-negative integer.")
        if 'longitude' in df_city.columns and 'latitude' in df_city.columns:
            df_city['longitude'] = df_city['longitude'].round(
                coordinate_precision
            )
            df_city['latitude'] = df_city['latitude'].round(
                coordinate_precision
            )
            logger.info(
                f"Coordinates rounded to {coordinate_precision} "
                "decimal places."
            )
        else:
            logger.warning(
                "Longitude/Latitude columns not found for rounding, "
                "but coordinate_precision was set."
            )

    logger.info(
        f"Original DataFrame shape for augmentation: {df_city.shape}"
    )

    # --- Call the core augmentation function ---
    try:
        df_augmented = augment_spatiotemporal_data(
            df=df_city,
            mode=mode,
            group_by_cols=_group_by_cols,
            time_col=_time_col,
            value_cols_interpolate=_value_cols_interpolate,
            feature_cols_augment=_feature_cols_augment,
            interpolation_kwargs=_interpolation_config,
            augmentation_kwargs=_augmentation_config,
            verbose=verbose
        )
        logger.info(
            f"Augmented DataFrame shape: {df_augmented.shape}"
        )

        if savefile:
            try:
                # Ensure directory exists for savefile
                save_dir = os.path.dirname(savefile)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                df_augmented.to_csv(savefile, index=False)
                logger.info(
                    f"Augmented DataFrame saved to: {savefile}"
                )
            except Exception as e_save:
                logger.error(
                    f"Failed to save augmented DataFrame to '{savefile}': "
                    f"{e_save}"
                )

        return df_augmented

    except ValueError as ve:
        logger.error(
            f"ValueError during core augmentation process: {ve}"
        )
        raise
    except TypeError as te:
        logger.error(
            f"TypeError during core augmentation process: {te}"
        )
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during core augmentation: {e}"
        )
        raise

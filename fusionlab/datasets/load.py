# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

"""
Functions to load sample datasets included with the ``fusionlab``
package, suitable for demonstrating and testing forecasting models
and utilities. Datasets are returned as pandas DataFrames or structured
Bunch objects.
"""

from __future__ import annotations

import os
import textwrap
import warnings
import joblib
import pandas as pd
import numpy as np

from typing import (
    Optional,
    Union,
    Tuple,
    List,
    Dict,
    Any
)

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder
)

from .._fusionlog import fusionlog
from ..api.bunch import XBunch
from ._property import (
    get_data,
    download_file_if,
    RemoteMetadata,
    FLAB_DMODULE,
    FLAB_REMOTE_DATA_URL
)
from ._config import CITY_CONFIGS 

try:
    from ..utils.geo_utils import augment_city_spatiotemporal_data
    from ..utils.geo_utils import generate_dummy_pinn_data
    from ..utils.io_utils import fetch_joblib_data
    from ..nn.utils import reshape_xtft_data
    from ..utils.data_utils import nan_ops
    from ..utils.spatial_utils import spatial_sampling
except ImportError:
    warnings.warn(
        "Could not import augment_city_spatiotemporal_data from "
        "fusionlab.utils.geo_utils. Data augmentation will not be "
        "available."
    )
    warnings.warn(
        "Could not import spatial_sampling from fusionlab.utils. "
        "Sampling functionality will be limited."
    )
    def spatial_sampling(*args, **kwargs):
        warnings.warn(
            "spatial_sampling is unavailable. Returning full data."
        )
        return args[0] if args else None

    def augment_city_spatiotemporal_data(df, **kwargs):
        warnings.warn(
            "augment_city_spatiotemporal_data is unavailable. "
            "Skipping augmentation."
        )
        return df

    def nan_ops(df, **kwargs):
        warnings.warn("nan_ops function not found. Skipping NaN handling.")
        return df

    def reshape_xtft_data(*args, **kwargs):
        warnings.warn(
            "reshape_xtft_data currently operates with tensorflow as "
            "backend. Make sure to have tensorflow installed."
        )
        raise ModuleNotFoundError

    def fetch_joblib_data(*args, **kwargs):
        warnings.warn("fetch_joblib_data not found. Caching disabled.")
        raise FileNotFoundError  # Mimic cache miss


logger = fusionlog().get_fusionlab_logger(__name__)

# --- Metadata Definition ---
_ZHONGSHAN_METADATA = RemoteMetadata(
    file='zhongshan_2000.csv',  # The already sampled file
    url=FLAB_REMOTE_DATA_URL,
    checksum=None,  # TODO: Add checksum
    descr_module=None,
    data_module=FLAB_DMODULE
)

_NANSHA_METADATA = RemoteMetadata(
    file='nansha_2000.csv',
    url=FLAB_REMOTE_DATA_URL,
    checksum=None,  # TODO: Add checksum
    descr_module=None,
    data_module=FLAB_DMODULE
)

CITY_CONFIGS["zhongshan"]['metadata'] = _ZHONGSHAN_METADATA
CITY_CONFIGS["nansha"]['metadata'] = _NANSHA_METADATA

__all__ = [
    "fetch_zhongshan_data",
    "fetch_nansha_data",
    "load_processed_subsidence_data",
    "load_subsidence_pinn_data"
]

def load_subsidence_pinn_data(
    data_name: str = 'zhongshan',
    strategy: str = "load",
    n_samples: Optional[int] = None,
    include_coords: bool = True,
    include_target: bool = True,
    encode_categoricals: bool = True,
    scale_numericals: bool = True,
    scaler_type: str = "minmax",
    return_dataframe: bool = False,
    data_home: Optional[str] = None,
    use_cache: bool = True,
    save_cache: bool = False,
    cache_suffix: str = "",
    augment_data: bool = False,
    augment_mode: str = 'both',
    group_by_cols: Optional[List[str]] = None,
    time_col: Optional[str] = None,
    value_cols_interpolate: Optional[List[str]] = None,
    feature_cols_augment: Optional[List[str]] = None,
    interpolation_config: Optional[Dict[str, Any]] = None,
    feature_config: Optional[Dict[str, Any]] = None,
    target_name: Optional[str] = None,  # Default will be subsidence_col
    interpolate_target: bool = False,
    coordinate_precision: Optional[int] = 4,  # Default precision,
    year_range: Union [Tuple [float, float] ,  None] = None,
    coords_range: Union [
        Tuple[Tuple[float, float], Tuple[float, float]],
        None] = None,

    vars_range = None,
    verbose: int = 1
) -> Union[pd.DataFrame, XBunch]:

    if data_name.lower() not in CITY_CONFIGS:
        raise ValueError(
            f"Unknown data_name: '{data_name}'. "
            f"Choose from {list(CITY_CONFIGS.keys())}."
        )

    cfg = CITY_CONFIGS[data_name.lower()]
    metadata = cfg["metadata"]

    # Resolve local file path, downloading if necessary
    # get_data typically returns a path within the package or data_home
    # download_file_if_missing handles the download logic
    data_path_resolved = os.path.join(
        get_data(data_home),
        metadata.file
    )
    try:
        # Ensure the file is downloaded if it's from a remote URL and
        # doesn't exist
        if metadata.url and not os.path.exists(data_path_resolved):
            if verbose >= 1:
                logger.info(
                    f"Attempting to download {data_name} data"
                    f" from {metadata.url} to "
                    f"{data_path_resolved}..."
                )
            download_file_if(
                data_path_resolved,
                metadata.url,
                checksum=metadata.checksum
            )
    except Exception as e:
        if strategy == "load":
            raise FileNotFoundError(
                f"Failed to download or find required data file "
                f"'{metadata.file}' for {data_name}: {e}"
            ) from e

        logger.warning(
            f"Could not download data for {data_name}: {e}. "
            "Will proceed based on strategy."
        )
        # If strategy is fallback or generate, df will be None and
        # handled later

    # --- Caching Logic ---
    cache_dir = get_data(data_home)  # Specific cache subdir
    # Include data_name in cache filename
    base_name = f"{data_name}_{os.path.splitext(metadata.file)[0]}"
    proc_fname = f"{base_name}_processed{cache_suffix}.joblib"
    proc_fpath = os.path.join(cache_dir, proc_fname)

    df = None
    encoder_info: Dict[str, Any] = {}
    scaler_info: Dict[str, Any] = {}

    if use_cache:
        try:
            cached = joblib.load(proc_fpath)
            if isinstance(cached, dict) and "data" in cached:
                df = cached["data"]
                encoder_info = cached.get("encoder_info", {})
                scaler_info = cached.get("scaler_info", {})
                if verbose >= 1:
                    logger.info(f"Loaded cached data from: {proc_fpath}")
            else:  # Cache file is invalid
                if verbose >= 1:
                    logger.info(
                        f"Cache file {proc_fpath} is invalid or malformed."
                    )
                df = None  # Force reload/generation
        except FileNotFoundError:
            if verbose >= 1:
                logger.info(f"No cache file found at: {proc_fpath}")
        except Exception as e:
            warnings.warn(
                f"Error loading cache ({proc_fpath}: {e}); will reprocess."
            )
            df = None  # Force reload/generation

    # --- Data Loading / Generation ---
    if df is None:  # If not loaded from cache
        if strategy == "generate":
            df = None  # Force dummy generation

        elif strategy == "load":
            if not os.path.exists(data_path_resolved):
                raise FileNotFoundError(
                    f"Data file '{data_path_resolved}' not found and "
                    "strategy is 'load'."
                )
            if verbose >= 1:
                logger.info(
                    f"Loading raw data from: {data_path_resolved}"
                )
            df = pd.read_csv(data_path_resolved)

        elif strategy == "fallback":
            try:
                if verbose >= 1:
                    logger.info(
                        f"Attempting to load from: {data_path_resolved}"
                    )
                df = pd.read_csv(data_path_resolved)
            except Exception:
                if verbose >= 1:
                    logger.info(
                        f"Failed to load '{data_path_resolved}', "
                        "generating dummy data."
                    )
                df = None  # Trigger dummy generation

        else:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Choose 'load', "
                "'generate', or 'fallback'."
            )

        if df is None:  # Generate dummy data
            n_samples = n_samples or 500_000
            if verbose >= 1:
                logger.info(
                    f"Generating {n_samples} dummy samples "
                    f"for {data_name}..."
                )

            dummy_data_dict = generate_dummy_pinn_data(
                n_samples=n_samples,
                year_range=year_range,
                coords_range=coords_range,
                vars_range=vars_range,
            )

            # Add city-specific categorical and numerical features
            for cat_col in cfg["categorical_cols"]:
                # Simplified dummy categories
                dummy_data_dict[cat_col] = np.random.choice(
                    [f"{cat_col}_A", f"{cat_col}_B", f"{cat_col}_C"],
                    size=n_samples
                )
            for num_col in cfg["numerical_main"]:
                if num_col not in dummy_data_dict:
                    dummy_data_dict[num_col] = np.random.rand(n_samples)
            if data_name == 'nansha' and 'soil_thickness' in cfg[
                "numerical_main"
            ]:
                dummy_data_dict['soil_thickness'] = np.random.uniform(
                    1, 10, n_samples
                )

            df = pd.DataFrame(dummy_data_dict)

        if verbose >= 1:
            logger.info(f"Data loaded/generated. Initial shape: {df.shape}")

        # --- Preprocessing ---
        # Ensure essential columns exist
        essential_for_core_processing = [
            cfg["time_col"],
            cfg["lon_col"],
            cfg["lat_col"],
            cfg["subsidence_col"],
            cfg["gwl_col"]
        ]
        missing_essentials = [
            c for c in essential_for_core_processing if c not in df.columns
        ]
        if missing_essentials:
            raise ValueError(
                f"DataFrame for {data_name} is missing essential columns: "
                f"{missing_essentials}"
            )

        df = df.dropna(subset=essential_for_core_processing).copy()

        # Datetime conversion
        try:
            time_column_to_convert = cfg["time_col"]
            if pd.api.types.is_numeric_dtype(df[time_column_to_convert]) and \
               all(df[time_column_to_convert].apply(
                   lambda x: 1900 < x < 2100
               )):  # Heuristic for year int
                df[cfg["dt_col_name"]] = pd.to_datetime(
                    df[time_column_to_convert], format="%Y"
                )
            else:
                df[cfg["dt_col_name"]] = pd.to_datetime(
                    df[time_column_to_convert]
                )
        except Exception as e:
            raise ValueError(
                f"Error parsing '{cfg['time_col']}' to datetime: {e}"
            )
        df = df.dropna(subset=[cfg["dt_col_name"]])
        if verbose >= 2:
            logger.debug(
                f"Shape after datetime conversion & NA drop: {df.shape}"
            )

        # One-Hot Encode
        current_encoder_info = {
            "columns": {}, "names": {}, "encoder_instance": None
        }
        if encode_categoricals:
            cats_to_encode = [
                c for c in cfg["categorical_cols"] if c in df.columns
            ]
            if cats_to_encode:
                if verbose >= 1:
                    logger.info(f"One-Hot encoding: {cats_to_encode}")
                encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown='ignore',
                    dtype=np.float32
                )
                enc_data = encoder.fit_transform(df[cats_to_encode])
                ohe_cols = encoder.get_feature_names_out(cats_to_encode)
                enc_df = pd.DataFrame(
                    enc_data,
                    columns=ohe_cols,
                    index=df.index
                )
                df = df.drop(columns=cats_to_encode)
                df = pd.concat([df, enc_df], axis=1)
                for i, col_cat in enumerate(cats_to_encode):
                    current_encoder_info["columns"][col_cat] = [
                        name for name in ohe_cols
                        if name.startswith(col_cat + "_")
                    ]
                    current_encoder_info["names"][col_cat] = list(
                        encoder.categories_[i]
                    )
                current_encoder_info["encoder_instance"] = encoder
                encoder_info = current_encoder_info
            elif verbose >= 2:
                logger.debug("No categorical columns found for encoding.")
        elif verbose >= 1:
            logger.info("Skipping categorical encoding.")

        # Numerical Time Coordinate (must be done AFTER all rows are fixed)
        df[cfg["time_col"] + "_numeric"] = (
            df[cfg["dt_col_name"]].dt.year
            + (df[cfg["dt_col_name"]].dt.dayofyear - 1)
            / (365 + df[cfg["dt_col_name"]].dt.is_leap_year.astype(int))
        )

        # Scale Numerical Features
        current_scaler_info = {"columns": [], "scaler_instance": None}
        if scale_numericals:
            # Scale only the main numerical features, not coords, time, or targets.
            cols_to_scale = [
                c for c in cfg["numerical_main"] if c in df.columns
            ]
            if cols_to_scale:
                if verbose >= 1:
                    logger.info(
                        f"Scaling numerical features: {cols_to_scale} "
                        f"with {scaler_type} scaler."
                    )
                if scaler_type == "minmax":
                    scaler = MinMaxScaler()
                elif scaler_type == "standard":
                    scaler = StandardScaler()
                else:
                    raise ValueError(f"Unknown scaler_type: {scaler_type}")

                df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                current_scaler_info["columns"] = cols_to_scale
                current_scaler_info["scaler_instance"] = scaler
                scaler_info = current_scaler_info
            elif verbose >= 2:
                logger.debug(
                    "No numerical features found/specified for scaling."
                )
        elif verbose >= 1:
            logger.info("Skipping numerical scaling.")

        # Save to cache if enabled
        if save_cache:
            os.makedirs(cache_dir, exist_ok=True)
            save_obj = {
                "data": df,
                "encoder_info": encoder_info,
                "scaler_info": scaler_info
            }
            try:
                joblib.dump(save_obj, proc_fpath)
                if verbose >= 1:
                    logger.info(
                        f"Saved processed cache to: {proc_fpath}"
                    )
            except Exception as e:
                warnings.warn(
                    f"Failed to save cache to {proc_fpath}: {e}"
                )

    # --- Data Augmentation Step ---
    if augment_data:
        if verbose >= 1:
            logger.info(
                f"Applying data augmentation (mode: {augment_mode})..."
            )

        # Set defaults for augmentation parameters if not provided by user
        aug_group_cols = group_by_cols or [
            cfg["lon_col"], cfg["lat_col"]
        ]
        aug_time_col = time_col or cfg["time_col"]  # Original time col

        aug_val_cols_interp = value_cols_interpolate
        if aug_val_cols_interp is None:
            aug_val_cols_interp = cfg["default_value_cols_interpolate"]
        # Conditionally add target to interpolation
        _aug_target_name = target_name or cfg["subsidence_col"]
        if interpolate_target and _aug_target_name not in aug_val_cols_interp:
            aug_val_cols_interp.append(_aug_target_name)

        aug_feat_cols_noise = feature_cols_augment
        if aug_feat_cols_noise is None:
            aug_feat_cols_noise = cfg["default_feature_cols_augment"]
            # Ensure target is not in noise augmentation list by default
            if _aug_target_name in aug_feat_cols_noise:
                aug_feat_cols_noise = [
                    c for c in aug_feat_cols_noise
                    if c != _aug_target_name
                ]

        interp_conf = interpolation_config or {
            'freq': 'AS', 'method': 'linear'
        }
        aug_conf = feature_config or {
            'noise_level': 0.01, 'noise_type': 'gaussian'
        }

        df = augment_city_spatiotemporal_data(  # Call the geo_utils function
            df=df,
            city=data_name,  # Pass city to the underlying augmentation
            mode=augment_mode,
            group_by_cols=aug_group_cols,
            time_col=aug_time_col,  # Original time col for datetime ops
            value_cols_interpolate=aug_val_cols_interp,
            feature_cols_augment=aug_feat_cols_noise,
            interpolation_config=interp_conf,
            augmentation_config=aug_conf,
            target_name=_aug_target_name,
            interpolate_target=interpolate_target,
            verbose=verbose > 1,  # Pass down a boolean verbose
            coordinate_precision=coordinate_precision,
            # savefile parameter is not used here, augmentation in-memory
        )
        if verbose >= 1:
            logger.info(f"Data augmentation complete. New shape: {df.shape}")

    if return_dataframe:
        return df.copy()

    # --- Build XBunch for output ---
    feature_cols = [
        c for c in df.columns
        if c not in [
            cfg["dt_col_name"],
            cfg["subsidence_col"],
            cfg["gwl_col"]
        ]
    ]
    if not include_coords:
        feature_cols = [
            c for c in feature_cols
            if c not in [cfg["lon_col"], cfg["lat_col"]]
        ]

    target_cols_bunch: list[str] = []
    if include_target:
        target_cols_bunch = [cfg["subsidence_col"], cfg["gwl_col"]]
        # Ensure target columns exist before trying to access them
        missing_targets = [
            tc for tc in target_cols_bunch if tc not in df.columns
        ]
        if missing_targets:
            logger.warning(
                f"Target columns {missing_targets} not found in DataFrame "
                "for XBunch. Target array will be None."
            )
            target_array = None
            target_cols_bunch = [
                tc for tc in target_cols_bunch
                if tc in df.columns
            ]
        else:
            target_array = df[target_cols_bunch].values
    else:
        target_array = None

    data_array = (
        df[feature_cols].values
        if feature_cols
        else np.array([[] for _ in range(len(df))])
    )

    descr = (
        f"Processed {data_name.capitalize()} PINN data.\n"
        f"Load Strategy: {strategy}.\n"
        f"Cache Used: {'Yes' if use_cache else 'No'}, "
        f"Cache Path: {proc_fpath if use_cache else 'N/A'}.\n"
        f"Categorical Encoding: {'Applied' if encode_categoricals else 'Skipped'}.\n"
        f"Numerical Scaling: {scaler_type if scale_numericals else 'Skipped'}.\n"
        f"Augmentation: {'Applied' if augment_data else 'None'} "
        f"(Mode: {augment_mode if augment_data else 'N/A'}).\n"
        f"Rows: {len(df)}, Features: {len(feature_cols)} "
        f"(in 'data' array).\n"
        f"Targets: {target_cols_bunch if include_target else 'None'}.\n"
        f"Coordinate Precision: {coordinate_precision} decimal places.\n"
        f"Time Column (numeric): {cfg['time_col'] + '_numeric'}."
    )

    bunch_dict: Dict[str, Any] = {
        "frame": df.copy(),  # Return a copy
        "data": data_array,
        "feature_names": feature_cols,
        "target_names": target_cols_bunch,
        "target": target_array,
        "DESCR": descr,
        "encoder_info": encoder_info,  # From cache or fresh processing
        "scaler_info": scaler_info     # From cache or fresh processing
    }
    if include_coords:
        if cfg["lon_col"] in df.columns:
            bunch_dict["longitude"] = df[cfg["lon_col"]].values
        if cfg["lat_col"] in df.columns:
            bunch_dict["latitude"] = df[cfg["lat_col"]].values

    return XBunch(**bunch_dict)

load_subsidence_pinn_data.__doc__="""\
Load and preprocess subsidence‐focused PINN data for Zhongshan or Nansha.

This function handles data retrieval (from local CSV or remote),
optional dummy‐data generation, caching, preprocessing (datetime
conversion, one‐hot encoding, numeric scaling), and optional
spatio‐temporal augmentation.  When `return_dataframe=False`, it
returns an XBunch object suitable for downstream modeling.

Parameters
----------
data_name : str, default='zhongshan'
    Which city dataset to load.  Supported values: 'zhongshan',
    'nansha'.  Case‐insensitive.  Used to select city‐specific
    metadata (file names, feature lists, etc.).
strategy : {'load', 'generate', 'fallback'}, default='load'
    - 'load': strictly attempt to read the CSV from a local or
      downloaded location; raise an error if not found.
    - 'generate': skip CSV loading and always generate randomized
      “dummy” data matching the schema.
    - 'fallback': attempt CSV load; if loading fails (file missing
      or corrupted), generate dummy data instead.
n_samples : int or None, default=None
    Number of dummy rows to generate when `strategy` is 'generate'
    or when generation is triggered under 'fallback'.  If None,
    defaults to 500 000 samples.
include_coords : bool, default=True
    If True, include `'longitude'` and `'latitude'` arrays in the
    returned XBunch (under keys `"longitude"` and `"latitude"`).
    If False, drop those coordinate columns from the feature set.
include_target : bool, default=True
    If True, include subsidence and GWL as targets in the XBunch
    (`"target_names"` and `"target"`).  If False, return an XBunch
    with no `"target"` array.
encode_categoricals : bool, default=True
    If True, One‐Hot Encode any city‐specific categorical columns
    (e.g., `'geology'`, `'density_tier'`).  Otherwise, skip encoding.
scale_numericals : bool, default=True
    If True, apply MinMaxScaler or StandardScaler (see `scaler_type`)
    to the city’s “main” numeric features (e.g., rainfall, density,
    seismic risk).  Coordinates and targets are not scaled here.
scaler_type : {'minmax', 'standard'}, default='minmax'
    Which scaler to apply when `scale_numericals=True`.  - 'minmax'
    uses `sklearn.preprocessing.MinMaxScaler`.  - 'standard' uses
    `sklearn.preprocessing.StandardScaler`.
return_dataframe : bool, default=False
    If True, return the processed `pd.DataFrame` directly.  Otherwise,
    pack results into an `XBunch` with fields:
      - `"frame"`: the processed DataFrame
      - `"data"`: feature matrix (numpy array)
      - `"feature_names"`: list of column names in `"data"`
      - `"target_names"`: list of target column names (or `[]`)
      - `"target"`: target array (or `None`)
      - `"DESCR"`: textual description
      - `"longitude"`, `"latitude"` (if `include_coords=True`)
      - `"encoder_info"`: dict of one‐hot encoder details
      - `"scaler_info"`: dict of scaler details
data_home : str or None, default=None
    Root directory for caching and for locating local data files.
    Passed to `fusionlab.datasets.get_data()`.  If None, uses the
    package’s default data directory.
use_cache : bool, default=True
    If True, attempt to load a previously processed `.joblib` cache
    (filename includes `data_name` and `cache_suffix`).  If the cache
    exists and is valid, skip reprocessing and return cached results.
save_cache : bool, default=False
    If True, after successful processing, save the processed DataFrame
    and encoder/scaler info to a `.joblib` file under `data_home`.
    Subsequent calls with `use_cache=True` will load from this cache.
cache_suffix : str, default=''
    Suffix to append to the cache filename (before `.joblib`).  Useful
    to distinguish different processing parameters or versions.
augment_data : bool, default=False
    If True, apply spatio‐temporal augmentation via
    `augment_city_spatiotemporal_data`.  This can interpolate missing
    values, add noise to features, and upsample the time dimension.
augment_mode : str, default='both'
    Passed to the augmentation routine.  Typical options include
    `'both'` (interpolate + noise), `'interpolate_only'`,
    `'noise_only'`.  See `fusionlab.utils.geo_utils.augment_city_spatiotemporal_data`
    for full details.
group_by_cols : list[str] or None, default=None
    Which columns to group by during augmentation (e.g., coordinates).
    If None, defaults to the city’s spatial columns (`lon_col`, `lat_col`).
time_col : str or None, default=None
    Time column name (string) to pass to augmentation.  If None,
    uses the city’s configured `"time_col"`.
value_cols_interpolate : list[str] or None, default=None
    Which numeric columns to interpolate during augmentation (e.g.,
    `"GWL"`, `"rainfall_mm"`).  If None, uses the city’s
    `"default_value_cols_interpolate"` list.
feature_cols_augment : list[str] or None, default=None
    Which columns to add noise to during augmentation.  If None,
    uses the city’s `"default_feature_cols_augment"` list.  Note that
    the target column is never noise‐augmented by default.
interpolation_config : dict or None, default=None
    Configuration passed to the interpolation step of augmentation
    (e.g., `{'freq': 'AS', 'method': 'linear'}`).  If None, defaults
    to `{'freq': 'AS', 'method': 'linear'}`.
feature_config : dict or None, default=None
    Configuration for adding noise, e.g., `{'noise_level': 0.01,
    'noise_type': 'gaussian'}`.  If None, sensible defaults are used.
target_name : str or None, default=None
    If `interpolate_target=True`, this names the column to interpolate
    (default is the city’s `"subsidence_col"`).  If not provided,
    the function uses the configured subsidence column.
interpolate_target : bool, default=False
    If True, include the target column itself in the interpolation
    pass.  (Useful when filling gaps in observed subsidence values.)
coordinate_precision : int or None, default=4
    Number of decimal places to round latitude/longitude to.  After
    rounding, spatial grouping (e.g., augmentation) will treat points
    at the same rounded coordinate as identical.  Set to None to skip
    coordinate rounding.
year_range : tuple[int, int] or None, default=None
    If dummy data generation is used, the `(min_year, max_year)` range
    for uniformly sampling integer years.  If None, defaults to `(2000, 2025)`.
coords_range : tuple[tuple[float, float], tuple[float, float]] or None, default=None
    If dummy generation is used, spatial bounds as
    `((lon_min, lon_max), (lat_min, lat_max))`.  If None, defaults to
    `((113.0, 113.8), (22.3, 22.8))` for Zhongshan and a similar range
    for Nansha.
vars_range : dict or None, default=None
    If dummy generation is used, a dictionary specifying ranges for
    other variables.  For example:
    `{"rainfall_mm": (500, 2500), "GWL": (1.0, 4.0)}`.  If a variable
    is omitted, its default distribution is used.
verbose : {0, 1, 2}, default=1
    Controls verbosity of console output:
    - 0: silent (except for exceptions).
    - 1: high‐level info messages.
    - 2: debug‐level messages (detailed shape/log prints).

Returns
-------
Union[pd.DataFrame, XBunch]
    If `return_dataframe=True`, returns the processed DataFrame.
    Otherwise, returns an XBunch with the following keys:
      - frame: pandas DataFrame of processed data
      - data: numpy array (rows × features) ready for modeling
      - feature_names: list of column names corresponding to `data`
      - target_names: list of target column names (or empty list)
      - target: numpy array of target values (or None if `include_target=False`)
      - DESCR: a multi‐line description string summarizing processing steps
      - longitude, latitude: numpy arrays if `include_coords=True`
      - encoder_info: dict containing one‐hot encoder metadata
      - scaler_info: dict containing scaler metadata

Notes
-----
1. **Caching**: When `use_cache=True`, the function looks for a
   `.joblib` file named `{data_name}_{basename}_processed{cache_suffix}.joblib`
   under `data_home`.  If found and valid, this file is loaded to
   skip reprocessing.  If `save_cache=True`, the final processed
   DataFrame is saved to the same path for future reuse.
2. **Dummy Generation**: When `strategy='generate'` or when
   fallback generation is triggered under `'fallback'`, the function
   calls `generate_dummy_pinn_data(...)` to produce a synthetic
   dataset.  Users can override `year_range`, `coords_range`, and
   `vars_range` to control the random distributions.  See the
   `fusionlab.utils.geo_utils.generate_dummy_pinn_data` docstring
   for details on default behavior.
3. **Augmentation**: When `augment_data=True`, the function invokes
   `augment_city_spatiotemporal_data(...)` with parameters:
     - `group_by_cols`: columns used to group points (spatially)
     - `time_col`: column used for temporal interpolation
     - `value_cols_interpolate`: numeric columns to interpolate
     - `feature_cols_augment`: columns to which noise is added
     - `interpolation_config`: interpolation parameters (freq/method)
     - `feature_config`: noise configuration (level/type)
     - `coordinate_precision`: precision used to round coords before grouping
   Ensure that `fusionlab.utils.geo_utils.augment_city_spatiotemporal_data`
   is available in your installation if using augmentation.
4. **One‐Hot Encoding**: Only the configured categorical columns
   (e.g., `'geology'`, `'density_tier'`) are encoded.  All other
   string columns remain unchanged.
5. **Numeric Scaling**: Only the city’s `numerical_main` features
   (e.g., rainfall, density, seismic risk) are passed through the
   chosen scaler.  Coordinates, time numeric columns, and targets
   are not scaled here; downstream models or sequence preprocessors
   may handle those separately.

Examples
--------
**1. Simple load of processed Zhongshan data (no caching, no augmentation):**

>>> from fusionlab.datasets.load import load_subsidence_pinn_data
>>> zbunch = load_subsidence_pinn_data(
...     data_name='zhongshan',
...     strategy='load',
...     use_cache=False,
...     encode_categoricals=True,
...     scale_numericals=True,
...     scaler_type='minmax',
...     return_dataframe=False,
...     verbose=1
... )
>>> print(zbunch.frame.head())
     year  longitude  latitude  GWL  rainfall_mm  density_tier_...  ...
     2000   113.05     22.35    3.2  1200.0     ...

**2. Force generation of 100 000 dummy Nansha samples, skip encoding:**

>>> nbunch = load_subsidence_pinn_data(
...     data_name='nansha',
...     strategy='generate',
...     n_samples=100000,
...     encode_categoricals=False,
...     scale_numericals=True,
...     scaler_type='standard',
...     save_cache=True,
...     cache_suffix='_v1',
...     verbose=2
... )
>>> print(nbunch.data.shape)
(100000, 5)  # e.g., 5 numeric features

**3. Load Zhongshan data, but fallback to dummy if file missing, then
apply augmentation with yearly interpolation and Gaussian noise:**

>>> zbunch_aug = load_subsidence_pinn_data(
...     data_name='zhongshan',
...     strategy='fallback',
...     use_cache=False,
...     augment_data=True,
...     augment_mode='both',
...     interpolation_config={'freq':'YS','method':'linear'},
...     feature_config={'noise_level':0.02,'noise_type':'gaussian'},
...     verbose=1
... )
>>> print(zbunch_aug.frame.shape)
(e.g., 550000, 12)  # Augmented rows added after interpolation

"""


def fetch_zhongshan_data(
    *,
    n_samples: Optional[Union[int, str]] = None, 
    as_frame: bool = False,
    include_coords: bool = True,
    include_target: bool = True,
    data_home: Optional[str] = None,
    download_if_missing: bool = True,
    force_download: bool = False,
    random_state: Optional[int] = None, 
    verbose: bool = True
) -> Union[XBunch, pd.DataFrame]:
    r"""Fetch the Zhongshan land subsidence dataset (sampled 2000 points).

    Loads the `zhongshan_2000.csv` file, which contains features
    related to land subsidence spatially sampled down to ~2000 points
    from a larger dataset [Liu24]_. Includes coordinates, year,
    hydrogeological factors, geological properties, risk scores, and
    measured subsidence (target).

    Optionally allows further sub-sampling using the `n_samples`
    parameter via :func:`~fusionlab.utils.spatial_utils.spatial_sampling`.

    Column details: 'longitude', 'latitude', 'year', 'GWL',
    'seismic_risk_score', 'rainfall_mm', 'subsidence',
    'geological_category', 'normalized_density', 'density_tier',
    'subsidence_intensity', 'density_concentration',
    'normalized_seismic_risk_score', 'rainfall_category'.

    Parameters
    ----------
    n_samples : int, str or None, default=None
        Number of samples to load.
        - If ``None`` or ``'*'``: Load the full sampled dataset (~2000 rows).
        - If `int`: Sub-sample the specified number using spatial
          stratification via
          :func:`~fusionlab.utils.spatial_utils.spatial_sampling`.
          Must be <= number of rows in the full file.
          Requires `spatial_sampling` to be available.

    as_frame : bool, default=False
        Return type: ``False`` for Bunch object, ``True`` for DataFrame.

    include_coords : bool, default=True
        Include 'longitude' and 'latitude' columns.

    include_target : bool, default=True
        Include the 'subsidence' column.

    data_home : str, optional
        Path to cache directory. Defaults to ``~/fusionlab_data``.

    download_if_missing : bool, default=True
        Attempt download if file is not found locally.

    force_download : bool, default=False
        Force download attempt even if file exists locally.

    random_state : int, optional
        Seed for the random number generator used during sub-sampling
        if `n_samples` is an integer. Ensures reproducibility.

    verbose : bool, default=True
        Print status messages during file fetching and sampling.

    Returns
    -------
    data : :class:`~fusionlab.api.bunch.Bunch` or pandas.DataFrame
        Loaded or sampled data. Bunch object includes `frame`, `data`,
        `feature_names`, `target_names`, `target`, coords, and `DESCR`.

    Raises
    ------
    ValueError
        If `n_samples` is invalid (e.g., non-integer, negative, or larger
        than available rows when sampling).
    FileNotFoundError, RuntimeError
        If the dataset file cannot be found or downloaded.
    OSError
        If there is an error reading the dataset file.

    References
    ----------
    .. [Liu24] Liu, J., et al. (2024). Machine learning-based techniques...
               *Journal of Environmental Management*, 352, 120078.
    """
    # --- Step 1: Obtain filepath using helper ---
    filepath_to_load = download_file_if(
        metadata=_ZHONGSHAN_METADATA, data_home=data_home,
        download_if_missing=download_if_missing,
        force_download=force_download, error='raise', 
        verbose=verbose
    )

    # --- Step 2: Load data ---
    try:
        df = pd.read_csv(filepath_to_load)
        if verbose:
            print(f"Successfully loaded full data ({len(df)} rows)"
                  f" from: {filepath_to_load}")
    except Exception as e:
        raise OSError(
            f"Error reading dataset file at {filepath_to_load}: {e}"
            ) from e

    # --- Step 3: Optional Sub-sampling ---
    if n_samples is not None and n_samples != '*':
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(f"`n_samples` must be a positive integer"
                             f" or '*' or None. Got {n_samples}.")

        total_rows = len(df)
        if n_samples > total_rows:
             warnings.warn(
                 f"Requested n_samples ({n_samples}) is larger than "
                 f"available rows ({total_rows}). Returning full dataset."
             )
        elif 'longitude' not in df.columns or 'latitude' not in df.columns:
             warnings.warn(
                 "Coordinate columns ('longitude', 'latitude') not found. "
                 "Using simple random sampling instead of spatial sampling."
             )
             df = df.sample(n=n_samples, random_state=random_state)
             if verbose:
                  print(f"Performed simple random sampling: {len(df)} rows.")
        else:
            # Use spatial sampling
            if verbose:
                print(f"Performing spatial sampling for {n_samples} rows...")
            # Use verbose level 1 for spatial_sampling basic info
            sample_verbose = 1 if verbose else 0
            df = spatial_sampling(
                df,
                sample_size=n_samples,
                x_coord='longitude', # Assuming standard names
                y_coord='latitude',
                random_state=random_state,
                verbose=sample_verbose
            )
            if verbose:
                print(f"Spatial sampling complete: {len(df)} rows selected.")
    elif verbose:
        print("Loading full dataset (n_samples is None or '*').")

    # --- Step 4: Column Selection ---
    coord_cols = ['longitude', 'latitude']
    target_col = 'subsidence'
    feature_cols = [
        col for col in df.columns
        if col not in coord_cols + [target_col]
        ]
    cols_to_keep = []
    if include_coords:
        cols_to_keep.extend([c for c in coord_cols if c in df.columns])
    cols_to_keep.extend(feature_cols)
    if include_target:
        if target_col in df.columns:
            cols_to_keep.append(target_col)
        else: warnings.warn(f"Target column '{target_col}' not found.")

    final_cols = [c for c in cols_to_keep if c in df.columns]
    df_subset = df[final_cols].copy()

    # --- Step 5: Return DataFrame or Bunch ---
    if as_frame:
        df_subset.sort_values('year', inplace =True)
        return df_subset
    else:
        # Assemble Bunch object (descriptions need updating)
        target_names = ([target_col] if include_target and
                        target_col in df_subset else [])
        target_array = df_subset[target_names].values.ravel(
            ) if target_names else None
        bunch_feature_names = [
            c for c in df_subset.columns
            if c not in coord_cols + target_names
            ]
        try:
             data_array = df_subset[bunch_feature_names].select_dtypes(
                 include=np.number).values
        except Exception:
             data_array = None
             warnings.warn("Could not extract numerical data for Bunch.data")

        # Update description based on actual loaded/sampled size
        descr = textwrap.dedent(f"""\
        Zhongshan Land Subsidence Dataset (Raw Features)

        **Origin:**
        Spatially stratified sample from the dataset used in [Liu24]_,
        focused on Zhongshan, China. Contains raw features potentially
        influencing land subsidence. This function loads the pre-sampled
        'zhongshan_2000.csv' file and optionally sub-samples it further.

        **Data Characteristics (Loaded/Sampled):**
        - Samples: {len(df_subset)}
        - Total Columns Loaded: {len(df_subset.columns)}
        - Feature Columns (in Bunch): {len(bunch_feature_names)}
        - Target Column ('subsidence'): {'Present' if target_names else 'Not Loaded'}

        **Available Columns in Frame:** {', '.join(df_subset.columns)}
        """) # Removed full Bunch contents for brevity

        bunch_dict = {
            "frame": df_subset, "data": data_array,
            "feature_names": bunch_feature_names,
            "target_names": target_names, "target": target_array,
            "DESCR": descr,
        }
        if include_coords:
            if 'longitude' in df_subset:
                bunch_dict['longitude'] = df_subset['longitude'].values
            if 'latitude' in df_subset:
                bunch_dict['latitude'] = df_subset['latitude'].values

        return XBunch(**bunch_dict)


def fetch_nansha_data(
    *,
    n_samples: Optional[Union[int, str]] = None,
    as_frame: bool = False,
    include_coords: bool = True,
    include_target: bool = True,
    data_home: Optional[str] = None,
    download_if_missing: bool = True,
    force_download: bool = False,
    random_state: Optional[int] = None, 
    verbose: bool = True
) -> Union[XBunch, pd.DataFrame]:
    r"""Fetch the sampled Nansha land subsidence dataset (2000 points).

    Loads the `nansha_2000.csv` file, which contains features related
    to land subsidence in Nansha, China, spatially sampled down to 2000
    representative data points. It includes geographical coordinates,
    temporal information (year), geological factors, hydrogeological
    factors (GWL, rainfall), building concentration, risk scores, soil
    thickness, and the measured land subsidence (target).

    Optionally allows further sub-sampling using the `n_samples`
    parameter via :func:`~fusionlab.utils.spatial_utils.spatial_sampling`.

    Column details: 'longitude', 'latitude', 'year',
    'building_concentration', 'geology', 'GWL', 'rainfall_mm',
    'normalized_seismic_risk_score', 'soil_thickness', 'subsidence'.

    The function searches for the data file (`nansha_2000.csv`)
    using the logic in :func:`~fusionlab.datasets._property.download_file_if`
    (Cache > Package > Download).

    Parameters
    ----------
    n_samples : int, str or None, default=None
        Number of samples to load.
        - If ``None`` or ``'*'``: Load the full sampled dataset (~2000 rows).
        - If `int`: Sub-sample the specified number using spatial
          stratification via
          :func:`~fusionlab.utils.spatial_utils.spatial_sampling`.
          Must be <= number of rows in the full file.
          Requires `spatial_sampling` to be available.

    as_frame : bool, default=False
        Return type: ``False`` for Bunch object, ``True`` for DataFrame.

    include_coords : bool, default=True
        Include 'longitude' and 'latitude' columns.

    include_target : bool, default=True
        Include the 'subsidence' column.

    data_home : str, optional
        Path to cache directory. Defaults to ``~/fusionlab_data``.

    download_if_missing : bool, default=True
        Attempt download if file is not found locally.

    force_download : bool, default=False
        Force download attempt even if file exists locally.

    random_state : int, optional
        Seed for the random number generator used during sub-sampling.

    verbose : bool, default=True
        Print status messages during file fetching and sampling.

    Returns
    -------
    data : :class:`~fusionlab.api.bunch.Bunch` or pandas.DataFrame
        Loaded or sampled data. Bunch object includes `frame`, `data`,
        `feature_names`, `target_names`, `target`, coords, and `DESCR`.

    Raises
    ------
    ValueError
        If `n_samples` is invalid.
    FileNotFoundError, RuntimeError
        If the dataset file cannot be found or downloaded.
    OSError
        If there is an error reading the dataset file.
    """
    # --- Step 1: Obtain filepath using helper ---
    filepath_to_load = download_file_if(
        metadata=_NANSHA_METADATA, # Use Nansha metadata
        data_home=data_home,
        download_if_missing=download_if_missing,
        force_download=force_download,
        error='raise', # Raise error if not found/downloaded
        verbose=verbose
    )

    # --- Step 2: Load data ---
    try:
        df = pd.read_csv(filepath_to_load)
        if verbose:
            print(f"Successfully loaded full data ({len(df)} rows)"
                  f" from: {filepath_to_load}")
    except Exception as e:
        raise OSError(
            f"Error reading dataset file at {filepath_to_load}: {e}"
            ) from e

    # --- Step 3: Optional Sub-sampling ---
    if n_samples is not None and n_samples != '*':
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(f"`n_samples` must be a positive integer"
                             f" or '*' or None. Got {n_samples}.")

        total_rows = len(df)
        if n_samples > total_rows:
             warnings.warn(
                 f"Requested n_samples ({n_samples}) is larger than "
                 f"available rows ({total_rows}). Returning full dataset."
             )
        elif 'longitude' not in df.columns or 'latitude' not in df.columns:
             warnings.warn(
                 "Coordinate columns ('longitude', 'latitude') not found. "
                 "Using simple random sampling."
             )
             df = df.sample(n=n_samples, random_state=random_state)
             if verbose:
                  print(f"Performed simple random sampling: {len(df)} rows.")
        else:
            # Use spatial sampling
            if verbose:
                print(f"Performing spatial sampling for {n_samples} rows...")
            sample_verbose = 1 if verbose else 0
            df = spatial_sampling(
                df,
                sample_size=n_samples,
                x_coord='longitude',
                y_coord='latitude',
                random_state=random_state,
                verbose=sample_verbose
            )
            if verbose:
                print(f"Spatial sampling complete: {len(df)} rows selected.")
    elif verbose:
        print("Loading full dataset (n_samples is None or '*').")


    # --- Step 4: Column Selection ---
    coord_cols = ['longitude', 'latitude']
    target_col = 'subsidence' # Assuming same target name
    # Identify feature columns for Nansha data
    feature_cols = [
        col for col in df.columns
        if col not in coord_cols + [target_col]
        ]

    cols_to_keep = []
    if include_coords:
        cols_to_keep.extend([c for c in coord_cols if c in df.columns])
    cols_to_keep.extend(feature_cols)
    if include_target:
        if target_col in df.columns:
            cols_to_keep.append(target_col)
        else: warnings.warn(f"Target column '{target_col}' not found.")

    final_cols = [c for c in cols_to_keep if c in df.columns]
    df_subset = df[final_cols].copy()

    # --- Step 5: Return DataFrame or Bunch ---
    if as_frame:
        return df_subset
    else:
        # Assemble Bunch object
        target_names = ([target_col] if include_target and
                        target_col in df_subset else [])
        target_array = df_subset[target_names].values.ravel(
            ) if target_names else None
        bunch_feature_names = [
            c for c in df_subset.columns
            if c not in coord_cols + target_names
            ]
        try:
             data_array = df_subset[bunch_feature_names].select_dtypes(
                 include=np.number).values
        except Exception:
             data_array = None
             warnings.warn("Could not extract numerical data for Bunch.data")

        # Create description for Nansha
        descr = textwrap.dedent(f"""\
        Sampled Nansha Land Subsidence Dataset (Raw Features)

        **Origin:**
        Spatially stratified sample (n={len(df_subset)}) focused on
        Nansha, China. Contains raw features potentially influencing
        land subsidence, including geological info, building concentration,
        hydrogeology, etc.

        **Data Characteristics (Loaded/Sampled):**
        - Samples: {len(df_subset)}
        - Total Columns Loaded: {len(df_subset.columns)}
        - Feature Columns (in Bunch): {len(bunch_feature_names)}
        - Target Column ('subsidence'): {'Present' if target_names else 'Not Loaded'}

        **Available Columns in Frame:** {', '.join(df_subset.columns)}
        """) # Simplified Bunch contents description

        bunch_dict = {
            "frame": df_subset, "data": data_array,
            "feature_names": bunch_feature_names,
            "target_names": target_names, "target": target_array,
            "DESCR": descr,
        }
        if include_coords:
            if 'longitude' in df_subset:
                bunch_dict['longitude'] = df_subset['longitude'].values
            if 'latitude' in df_subset:
                bunch_dict['latitude'] = df_subset['latitude'].values

        return XBunch(**bunch_dict)

def load_processed_subsidence_data(
    dataset_name: str = 'zhongshan',
    *,
    # --- Keep all parameters as defined previously ---
    n_samples: Optional[Union[int, str]] = None,
    as_frame: bool = False,
    include_coords: bool = True, 
    include_target: bool = True,
    data_home: Optional[str] = None,
    download_if_missing: bool = True,
    force_download_raw: bool = False,
    random_state: Optional[int] = None,
    apply_feature_select: bool = True,
    apply_nan_ops: bool = True,
    encode_categoricals: bool = True,
    scale_numericals: bool = True,
    scaler_type: str = 'minmax',
    return_sequences: bool = False,
    time_steps: int = 4,
    forecast_horizons: int = 4,
    target_col: str = 'subsidence',
    scale_target: bool=False, 
    use_processed_cache: bool = True,
    use_sequence_cache: bool = True,
    save_processed_frame: bool = False,
    save_sequences: bool = False,
    cache_suffix: str = "",
    nan_handling_method: Optional[str] = 'fill',
    verbose: bool = True
) -> Union[XBunch, pd.DataFrame, Tuple[np.ndarray, ...]]:
    r"""Loads, preprocesses, and optionally sequences landslide datasets.

    This function provides a complete pipeline to prepare the Zhongshan
    or Nansha landslide datasets for use with forecasting models like
    TFT/XTFT. It performs the following steps:

    1. Loads the raw sampled data ('zhongshan_2000.csv' or
       'nansha_2000.csv') using fetch functions (:func:`Workspace_zhongshan_data`
       or :func:`Workspace_nansha_data`), optionally sub-sampling using
       spatial stratification if `n_samples` is specified.
    2. Optionally applies a predefined preprocessing sequence, mirroring
       steps often used in research (e.g., based on [Liu24]_):
       - Feature Selection (selecting a subset of columns).
       - NaN Handling (e.g., filling missing values).
       - Categorical Encoding (using One-Hot Encoding).
       - Numerical Scaling (using MinMaxScaler or StandardScaler).
    3. Optionally reshapes the fully processed data into sequences
       suitable for TFT/XTFT models using
       :func:`~fusionlab.utils.ts_utils.reshape_xtft_data`.
    4. Optionally leverages caching by loading/saving the processed
       DataFrame or the final sequence arrays to/from disk (`.joblib`)
       to accelerate repeated executions with the same parameters.

    Parameters
    ----------
    dataset_name : {'zhongshan', 'nansha'}, default='zhongshan'
        Which dataset to load and process ('zhongshan' or 'nansha').
    n_samples : int, str, or None, default=None
        Number of samples to load from the raw dataset file.
        - If ``None`` or ``'*'``: Loads the full dataset (~2000 rows).
        - If `int`: Sub-samples the specified number using spatial
          stratification via
          :func:`~fusionlab.utils.spatial_utils.spatial_sampling`.
          Must be a positive integer less than or equal to the total
          available samples.
    as_frame : bool, default=False
        Determines the return type *only if* ``return_sequences`` is ``False``.
        - If ``False``: Returns a Bunch object containing the processed
          DataFrame and metadata.
        - If ``True``: Returns only the processed pandas DataFrame.
    include_coords : bool, default=True
        If ``True``, include 'longitude' and 'latitude' columns in the
        output ``frame`` (and Bunch attributes).
    include_target : bool, default=True
        If ``True``, include the target column ('subsidence') in the
        output ``frame`` (and Bunch attributes).
    data_home : str, optional
        Specify a directory path to cache raw datasets and processed
        files. If ``None``, uses the path determined by
        :func:`~fusionlab.datasets._property.get_data`
        (typically ``~/fusionlab_data``). Default is ``None``.
    download_if_missing : bool, default=True
        If ``True``, attempt to download the raw dataset file from the
        remote repository if it's not found locally.
    force_download_raw : bool, default=False
        If ``True``, forces download of the raw dataset file, ignoring
        any local cache or packaged version.
    random_state : int, optional
        Seed for the random number generator used during sub-sampling
        when ``n_samples`` is an integer. Ensures reproducibility.
    apply_feature_select : bool, default=True
        If ``True``, selects only the subset of features typically used
        in reference examples for the specified `dataset_name`. If
        ``False``, attempts to use all columns found (after excluding
        coords/target).
    apply_nan_ops : bool, default=True
        If ``True``, apply NaN handling using the internal :func:`nan_ops`
        utility with the strategy specified by ``nan_handling_method``.
    encode_categoricals : bool, default=True
        If ``True``, apply Scikit-learn's OneHotEncoder to predefined
        categorical columns ('geology', 'density_tier' for Zhongshan;
        'geology' for Nansha). Adds new columns for encoded features
        and removes the original categorical columns.
    scale_numericals : bool, default=True
        If ``True``, apply feature scaling to predefined numerical columns
        (excluding coordinates, year, target, and encoded categoricals)
        using the scaler specified by ``scaler_type``. Target column is
        also scaled.
    scaler_type : {'minmax', 'standard'}, default='minmax'
        Type of scaler to use if `scale_numericals` is True.
    return_sequences : bool, default=False
        Controls the final output format.
        - If ``True``: Performs sequence generation using
          :func:`~fusionlab.utils.ts_utils.reshape_xtft_data` and
          returns the sequence arrays.
        - If ``False``: Skips sequence generation and returns the
          processed DataFrame or Bunch object (controlled by `as_frame`).
    time_steps : int, default=4
        Lookback window size (number of past time steps) for sequence
        generation. Only used if ``return_sequences=True``.
    forecast_horizons : int, default=4
        Prediction horizon (number of future steps) for sequence
        generation. Only used if ``return_sequences=True``.
    target_col : str, default='subsidence'
        Name of the target variable column used for sequence generation.
    scale_target: bool, default=False 
        Whether to scale the target or not. 
    use_processed_cache : bool, default=True
        If ``True`` and ``return_sequences=False``, attempts to load a
        previously saved processed DataFrame (and scaler/encoder info)
        from the cache directory before running the preprocessing steps.
    use_sequence_cache : bool, default=True
        If ``True`` and ``return_sequences=True``, attempts to load
        previously saved sequence arrays from the cache directory before
        running preprocessing and sequence generation.
    save_processed_frame : bool, default=False
        If ``True`` and preprocessing is performed (cache miss or
        ``use_processed_cache=False``), saves the resulting processed
        DataFrame, scaler info, and encoder info to a joblib file in
        the cache directory. Ignored if ``return_sequences=True``.
    save_sequences : bool, default=False
        If ``True`` and sequence generation is performed (cache miss or
        ``use_sequence_cache=False``), saves the resulting sequence
        arrays (`static_data`, `dynamic_data`, `future_data`,
        `target_data`) to a joblib file in the cache directory. Only used
        if ``return_sequences=True``.
    cache_suffix : str, default=""
        Optional suffix appended to cache filenames (before '.joblib')
        to allow caching results from different processing variations
        (e.g., different `n_samples` or preprocessing flags).
    nan_handling_method : str, default='fill'
        Method used by :func:`nan_ops` if ``apply_nan_ops=True``.
        Typically 'fill' (forward fill then backward fill).
    verbose : bool, default=True
        If ``True``, print status messages during file fetching,
        processing, caching, and sequence generation.

    Returns
    -------
    Processed Data : Union[Bunch, pd.DataFrame, Tuple[np.ndarray, ...]]
        The type depends on `return_sequences` and `as_frame`:
        - If `return_sequences=True`: Returns a tuple containing the
          sequence arrays required by TFT/XTFT:
          ``(static_data, dynamic_data, future_data, target_data)``
        - If `return_sequences=False` and `as_frame=True`: Returns the
          fully processed pandas DataFrame (after selection, NaN handling,
          encoding, scaling).
        - If `return_sequences=False` and `as_frame=False`: Returns a
          :class:`~fusionlab.api.bunch.Bunch` object containing the
          processed DataFrame (`frame`), extracted numerical features
          (`data`), feature names (`feature_names`), target info
          (`target_names`, `target`), coordinates (`longitude`,
          `latitude`), and a description (`DESCR`).

    Raises
    ------
    ValueError
        If `dataset_name` is invalid, `n_samples` is invalid, or required
        columns are missing for selected processing steps.
    FileNotFoundError, RuntimeError, OSError
        If underlying raw data loading fails (fetching from cache,
        package, or download).

    References
    ----------
    .. [Liu24] Liu, J., et al. (2024). Machine learning-based techniques...
               *Journal of Environmental Management*, 352, 120078.

    Examples
    --------
    >>> from fusionlab.datasets import load_processed_subsidence_data
    >>> # Load processed Zhongshan data as a Bunch object
    >>> data_bunch = load_processed_subsidence_data(dataset_name='zhongshan',
    ...                                             as_frame=False,
    ...                                             return_sequences=False)
    >>> print(data_bunch.frame.head())
    >>> print(data_bunch.feature_names)

    >>> # Load Nansha data, preprocess, and return sequences
    >>> static, dynamic, future, target = load_processed_subsidence_data(
    ...     dataset_name='nansha',
    ...     return_sequences=True,
    ...     time_steps=6,
    ...     forecast_horizons=3,
    ...     scale_numericals=True,
    ...     scaler_type='standard',
    ...     verbose=False
    ... )
    >>> print(f"Nansha sequences shapes: S={static.shape}, D={dynamic.shape},"
    ...       f" F={future.shape}, y={target.shape}")

    >>> # Load a small sample and save processed frame
    >>> df_proc_sample = load_processed_subsidence_data(
    ...     dataset_name='zhongshan',
    ...     n_samples=100,
    ...     random_state=42,
    ...     as_frame=True,
    ...     return_sequences=False,
    ...     save_processed_frame=True,
    ...     cache_suffix="_sample100"
    ... )
    >>> print(f"Loaded and processed sample shape: {df_proc_sample.shape}")

    """
    # --- Configuration based on dataset name ---
    if dataset_name == 'zhongshan':
        fetch_func = fetch_zhongshan_data
        default_features = [ # Features used in paper example
            'longitude', 'latitude', 'year', 'GWL', 'rainfall_mm',
            'geology', 'normalized_density', 'density_tier',
            'normalized_seismic_risk_score', 'subsidence'
            ]
        categorical_cols = ['geology', 'density_tier']
        # Numerical cols excluding coords, year, target, categoricals
        numerical_cols = [
            'GWL', 'rainfall_mm', 'normalized_density',
            'normalized_seismic_risk_score'
            ]
        spatial_cols = ['longitude', 'latitude']
        dt_col = 'year' # Time column for reshaping
    elif dataset_name == 'nansha':
        fetch_func = fetch_nansha_data
        default_features = [ # Features listed for Nansha
             'longitude', 'latitude', 'year', 'building_concentration',
             'geology', 'GWL', 'rainfall_mm',
             'normalized_seismic_risk_score', 'soil_thickness', 'subsidence'
             ]
        categorical_cols = ['geology', 'building_concentration'] # Example, adjust as needed
        numerical_cols = [
             'GWL', 'rainfall_mm',
             'normalized_seismic_risk_score', 'soil_thickness'
             ]
        spatial_cols = ['longitude', 'latitude']
        dt_col = 'year'
    else:
        raise ValueError(f"Unknown dataset_name: '{dataset_name}'."
                         " Choose 'zhongshan' or 'nansha'.")
    # --- Define Cache Filenames ---
    # ... (Keep cache filename logic) ...
    data_dir = get_data(data_home)
    processed_fname = f"{dataset_name}_processed{cache_suffix}.joblib"
    processed_fpath = os.path.join(data_dir, processed_fname)
    seq_fname = (f"{dataset_name}_sequences_T{time_steps}_H{forecast_horizons}"
                 f"{cache_suffix}.joblib")
    seq_fpath = os.path.join(data_dir, seq_fname)

    # --- Try Loading Cached Sequences ---
    if return_sequences and use_sequence_cache:
        try:
            sequences_data = fetch_joblib_data(
                seq_fpath, 'static_data', 'dynamic_data',
                'future_data', 'target_data', verbose=verbose > 1,
                error_mode='raise'
            )
            if verbose: print(f"Loaded cached sequences from: {seq_fpath}")
            return sequences_data
        
        except FileNotFoundError:
             if verbose > 0: 
                 print(f"Sequence cache not found: {seq_fpath}")
        except Exception as e:
             warnings.warn(
                 f"Error loading cached sequences: {e}. Reprocessing.")

    # --- Try Loading Cached Processed DataFrame ---
    df_processed = None
    # Initialize scaler/encoder info to be loaded from cache or created
    scaler_info = {'columns': [], 'scaler': None}
    encoder_info = {'columns': {}, 'names': {}}
    if use_processed_cache:
        try:
            cached_proc = fetch_joblib_data(
                processed_fpath, 'data', 'scaler_info', 'encoder_info',
                 verbose=verbose > 1, error_mode='ignore'
            )
            if cached_proc and isinstance(cached_proc, dict):
                 df_processed = cached_proc.get('data')
                 scaler_info = cached_proc.get('scaler_info', scaler_info)
                 encoder_info = cached_proc.get('encoder_info', encoder_info)
                 if df_processed is not None and verbose:
                      print(
                          f"Loaded cached processed DataFrame: {processed_fpath}")
            elif verbose > 0:
                 print(
                     f"Processed cache invalid/not found: {processed_fpath}")
        except FileNotFoundError:
             if verbose > 0:
                 print(
                     f"Processed cache not found: {processed_fpath}")
        except Exception as e:
             warnings.warn(
                 f"Error loading cached processed data: {e}. Reprocessing.")
             df_processed = None

    # --- Perform Processing if Cache Miss ---
    if df_processed is None:
        if verbose: print("Processing data from raw file...")
        # 1. Load Raw Data
        df_raw = fetch_func(
            as_frame=True, 
            n_samples=n_samples, 
            data_home=data_home,
            download_if_missing=download_if_missing,
            force_download=force_download_raw, 
            random_state=random_state,
            verbose=verbose > 1, 
            include_coords=True, 
            include_target=True
        )
        df_processed = df_raw.copy()
        df_processed.sort_values('year', inplace=True)
        # 2. Feature Selection
        if apply_feature_select:
            # Ensure all required features for the steps exist
            required_cols = default_features + [target_col]
            missing_fs = [f for f in required_cols if f not in df_processed.columns]
            if missing_fs:
                 raise ValueError(f"Required features missing from raw data:"
                                  f" {missing_fs}")
            # Select only the columns needed for this workflow
            df_processed = df_processed[default_features].copy()
            if verbose:
                 print(f"  Applied feature selection. Kept: {default_features}")

        # 3. NaN Handling
        if apply_nan_ops:
             original_len = len(df_processed)
             df_processed = nan_ops(
                 df_processed, ops='sanitize', 
                 action=nan_handling_method,
                 process="do_anyway", 
                 verbose=verbose > 1
                 )
             if verbose:
                 print(f"  Applied NaN handling ('{nan_handling_method}')."
                       f" Rows removed: {original_len - len(df_processed)}")

        # 4. Categorical Encoding
        encoder_info = {'columns': {}, 'names': {}} # Reset encoder info
        if encode_categoricals:
             if verbose: print("  Encoding categorical features...")
             # Keep non-categorical columns
             cols_to_keep_temp = df_processed.columns.difference(
                 categorical_cols).tolist()
             df_encoded_list = [df_processed[cols_to_keep_temp]]

             for col in categorical_cols:
                 if col in df_processed.columns:
                      encoder = OneHotEncoder(
                          sparse_output=False,
                        handle_unknown='ignore',
                        dtype=np.float32
                        ) # Ensure float output
                      encoded_data = encoder.fit_transform(df_processed[[col]])
                      # Use categories_ for naming to handle unseen values if needed
                      new_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                      encoded_df = pd.DataFrame(encoded_data,
                                                columns=new_cols,
                                                index=df_processed.index)
                      df_encoded_list.append(encoded_df)
                      encoder_info['columns'][col] = new_cols
                      encoder_info['names'][col] = encoder.categories_[0]
                      if verbose > 1: print(f"    Encoded '{col}' -> {len(new_cols)} cols")
                 else:
                      warnings.warn(f"Categorical column '{col}' not found.")
             # Combine original non-categorical with new encoded columns
             df_processed = pd.concat(df_encoded_list, axis=1)
        else:
             if verbose: 
                 print("  Skipped categorical encoding.")

        # 5. Numerical Scaling
        scaler_info = {'columns': [], 'scaler': None} # Reset scaler info
        if scale_numericals:
            # Identify numerical columns to scale from the *current* dataframe
            # Exclude coordinates, target (scaled separately or not at all),
            # and already encoded categoricals.
            current_num_cols = df_processed.select_dtypes(include=np.number).columns
            encoded_flat_list = [
                item for sublist in encoder_info['columns'].values()
                for item in sublist
                ]
            cols_to_scale = list(
                set(numerical_cols) & set(current_num_cols) - set(
                    encoded_flat_list))
            # Also scale the target column if present
            if scale_target: 
                if target_col in df_processed.columns and target_col not in cols_to_scale:
                    cols_to_scale.append(target_col)
            else: 
                # for consisteny drop target if exist in cols_to_scale 
                if target_col in cols_to_scale: 
                    try:
                        cols_to_scale.remove(target_col)
                    except: 
                        cols_to_scale = [
                            col for col in cols_to_scale if col !=target_col
                        ]
                    
            if cols_to_scale:
                if verbose: 
                    print(f"  Scaling numerical features: {cols_to_scale}...")
                if scaler_type == 'minmax':
                    scaler = MinMaxScaler()
                elif scaler_type == 'standard': 
                    scaler = StandardScaler()
                else: 
                    raise ValueError(f"Unknown scaler_type: {scaler_type}")

                df_processed[cols_to_scale] = scaler.fit_transform(
                    df_processed[cols_to_scale])
                scaler_info['columns'] = cols_to_scale
                scaler_info['scaler'] = scaler # Store fitted scaler
            else:
                 if verbose: print("  No numerical columns found/left to scale.")
        else:
             if verbose: print("  Skipped numerical scaling.")

        # 6. Save Processed DataFrame if requested
        if save_processed_frame:
            # Include data and potentially scalers/encoders
            save_data = {
                'data': df_processed,
                'scaler_info': scaler_info,
                'encoder_info': encoder_info
            }
            try:
                joblib.dump(save_data, processed_fpath)
                if verbose: print(f"Saved processed DataFrame to: {processed_fpath}")
            except Exception as e:
                warnings.warn(f"Failed to save processed data: {e}")

    # --- Return Processed DataFrame or Bunch if Sequences Not Requested ---
    if not return_sequences:
        if as_frame:
            return df_processed
        else:
            # Create Bunch from processed data
            # ... (Keep Bunch creation logic as before, using df_processed) ...
            target_names = ([target_col] if include_target and
                            target_col in df_processed else [])
            target_array = df_processed[target_names].values.ravel(
                ) if target_names else None
            bunch_feature_names = [
                c for c in df_processed.columns
                if c not in spatial_cols + target_names # Use spatial_cols
                ]
            try:
                data_array = df_processed[bunch_feature_names].select_dtypes(
                    include=np.number).values
            except Exception: data_array = None

            descr = textwrap.dedent(f"""\
            Processed {dataset_name.capitalize()} Landslide Dataset
            (Processing based on [Liu24] Zhongshan Example)

            **Origin:** See fetch_{dataset_name}_data docstring.
            **Processing Applied:** Feature Selection={apply_feature_select},
            NaN Handling='{nan_handling_method if apply_nan_ops else 'None'}',
            Categorical Encoding={'OneHot' if encode_categoricals else 'None'},
            Numerical Scaling={'None' if not scale_numericals else scaler_type}.

            **Data Characteristics:**
            - Samples: {len(df_processed)}
            - Columns: {len(df_processed.columns)}
            """)

            bunch_dict = {
                "frame": df_processed, "data": data_array,
                "feature_names": bunch_feature_names,
                "target_names": target_names, "target": target_array,
                "DESCR": descr,
            }
            # Add coords only if requested AND present
            if include_coords:
                if 'longitude' in df_processed: 
                    bunch_dict['longitude'] = df_processed['longitude'].values
                if 'latitude' in df_processed:
                    bunch_dict['latitude'] = df_processed['latitude'].values

            return XBunch(**bunch_dict)

    # --- Generate Sequences if Requested ---
    if verbose:
        print("\nReshaping processed data into sequences...")

    # === Step 6 Revision: Define final feature sets for reshape_xtft_data ===
    # Get list of one-hot encoded columns generated previously
    encoded_cat_cols = []
    if encode_categoricals and encoder_info.get('columns'):
        for cols in encoder_info['columns'].values():
            encoded_cat_cols.extend(cols)

    # Define Static Features for the model sequences
    # Paper example used: coords + encoded geology + encoded density_tier
    final_static_cols = list(spatial_cols) # Start with coordinates
    if dataset_name == 'zhongshan':
         # Add encoded columns if they exist in df_processed
         final_static_cols.extend([c for c in encoded_cat_cols if c.startswith(
             'geology_') or c.startswith('density_tier_')])
    elif dataset_name == 'nansha':
         # Add encoded columns for nansha if applicable
         final_static_cols.extend([c for c in encoded_cat_cols if c.startswith(
             'geology_') or c.startswith('building_concentration_')])
    # Ensure no duplicates and columns exist
    final_static_cols = sorted(list(set(
        c for c in final_static_cols if c in df_processed.columns)))
    if verbose > 1: print(f"  Final Static Cols: {final_static_cols}")

    # Define Dynamic Features for the model sequences
    # Paper example used: 'GWL', 'rainfall_mm', 'normalized_seismic_risk_score',
    #  'normalized_density'
    # These should correspond to columns in numerical_cols (already scaled)
    final_dynamic_cols = sorted(list(set(
        c for c in numerical_cols if c in df_processed.columns and c != target_col
    )))
    if verbose > 1: 
        print(f"  Final Dynamic Cols: {final_dynamic_cols}")

    # Define Future Features for the model sequences
    # Paper example used: 'rainfall_mm'
    final_future_cols = ['rainfall_mm'] # As per paper example
    # Ensure it exists
    final_future_cols = [c for c in final_future_cols if c in df_processed.columns]
    if not final_future_cols and dataset_name=='zhongshan': 
        # Check if required feature missing
         warnings.warn("'rainfall_mm' required for future features based on"
                       " example, but not found in processed data.")
    if verbose > 1:
        print(f"  Final Future Cols: {final_future_cols}")

    # --- End Step 6 Revision ---

    # Check if required columns exist before calling reshape
    required_for_reshape = (
         [dt_col, target_col] + final_static_cols + final_dynamic_cols
         + final_future_cols + spatial_cols
         )
    # Remove potential duplicates before checking
    required_unique = sorted(list(set(required_for_reshape)))
    missing_in_processed = [
         c for c in required_unique if c not in df_processed.columns
         ]
    if missing_in_processed:
        raise ValueError(f"Columns missing for reshape_xtft_data after"
                         f" processing: {missing_in_processed}. Available:"
                         f" {df_processed.columns.tolist()}")

    # Call reshape_xtft_data on the fully processed DataFrame
    static_data, dynamic_data, future_data, target_data = reshape_xtft_data(
        df=df_processed,
        dt_col=dt_col, # Use 'year' as the time index column
        target_col=target_col,
        static_cols=final_static_cols,
        dynamic_cols=final_dynamic_cols,
        future_cols=final_future_cols,
        spatial_cols=spatial_cols,
        time_steps=time_steps,
        forecast_horizons=forecast_horizons,
        verbose=verbose > 0
    )

    # Save sequences if requested
    if save_sequences:
         sequence_data_to_save = {
             'static_data': static_data, 'dynamic_data': dynamic_data,
             'future_data': future_data, 'target_data': target_data
         }
         try:
              joblib.dump(sequence_data_to_save, seq_fpath)
              if verbose: print(f"Saved sequences to: {seq_fpath}")
         except Exception as e:
              warnings.warn(f"Failed to save sequences: {e}")

    return static_data, dynamic_data, future_data, target_data

__all__.extend([ 
    "load_processed_subsidence_data",
    ])

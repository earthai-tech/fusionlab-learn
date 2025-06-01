# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com> 

"""
Physics-Informed Neural Network (PINN) Utility functions.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Union#, Any
import warnings # noqa 

from ..._fusionlog import fusionlog
from ...api.util import get_table_size 
from ...core.checks import exist_features, check_datetime
from ...core.handlers import columns_manager
from ...utils.validator import validate_positive_integer #, is_frame
from ...decorators import isdf 
from ...utils.generic_utils import print_box, vlog
from ...utils.geo_utils import resolve_spatial_columns 
from ...utils.io_utils import save_job 
from .. import KERAS_BACKEND, KERAS_DEPS

if KERAS_BACKEND:
    Tensor = KERAS_DEPS.Tensor
else:
    Tensor = np.ndarray # Fallback for type hinting

logger = fusionlog().get_fusionlab_logger(__name__)
_TW = get_table_size()

@isdf 
def prepare_pinn_data_sequences(
    df: pd.DataFrame,
    time_col: str,
    subsidence_col: str,
    gwl_col: str,
    dynamic_cols: List[str],
    static_cols: Optional[List[str]] = None,
    future_cols: Optional[List[str]] = None,
    spatial_cols: Optional[Tuple[str, str]]=None, 
    lon_col: Optional[str]=None,
    lat_col: Optional[str]=None,
    group_id_cols: Optional[List[str]] = None,
    time_steps: int = 12,
    forecast_horizon: int = 3,
    output_subsidence_dim: int = 1, # Typically 1 for subsidence value
    output_gwl_dim: int = 1,       # Typically 1 for GWL value
    datetime_format: Optional[str] = None,
    normalize_coords: bool = True, # Option to normalize t,x,y
    savefile: Optional[str] = None,
    verbose: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Reshapes and prepares time-series data into sequences for PINN models.

    This function transforms a Pandas DataFrame into structured NumPy arrays
    suitable for training Physics-Informed Neural Networks like PIHALNet.
    It creates sequences of dynamic features, future known features,
    static features, target variables (subsidence and GWL), and importantly,
    spatio-temporal coordinates corresponding to the forecast horizon.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing all necessary time-series data.
    time_col : str
        Name of the column representing time (e.g., 'year', 'date').
        This will be converted to a numerical representation for PINN.
    lon_col : str
        Name of the column for longitude.
    lat_col : str
        Name of the column for latitude.
    subsidence_col : str
        Name of the target column for subsidence.
    gwl_col : str
        Name of the target column for Groundwater Level (GWL).
    dynamic_cols : List[str]
        List of column names for dynamic (past observed) features.
    static_cols : List[str], optional
        List of column names for static (time-invariant) features.
        If None, no static features are used. Default is None.
    future_cols : List[str], optional
        List of column names for known future features.
        If None, no future features are used. Default is None.
    group_id_cols : List[str], optional
        List of column names to group the data by (e.g., site ID,
        borehole ID). Sequences are generated independently for each group.
        If None, the entire DataFrame is treated as a single group.
        Default is None.
    time_steps : int, default 12
        Number of past time steps to use for dynamic features (lookback).
    forecast_horizon : int, default 3
        Number of future time steps to predict.
    output_subsidence_dim : int, default 1
        The feature dimension of the subsidence target. Typically 1.
    output_gwl_dim : int, default 1
        The feature dimension of the GWL target. Typically 1.
    datetime_format : str, optional
        Format string for parsing the `time_col` if it's not already
        in datetime format. If None, attempts automatic parsing.
        Default is None.
    normalize_coords : bool, default True
        If True, normalizes the 't', 'x', 'y' coordinate values
        (derived from `time_col`, `lon_col`, `lat_col`) to the [0, 1]
        range based on the min/max within each group. This is often
        beneficial for neural network training.
    verbose : int, default 0
        Verbosity level for logging (0-10).
        - 0: Silent.
        - 1: Basic info.
        - 2: More details on shapes and processing.
        - 5: Per-group processing info.
        - 7: Per-sequence sample data (use with caution for large data).

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        A tuple containing two dictionaries:
        1. `inputs_dict`:
           - 'coords': np.ndarray of shape `(N, H, 3)` for `[t, x, y]`
             coordinates over the forecast horizon.
           - 'static_features': np.ndarray of shape `(N, D_s)`.
           - 'dynamic_features': np.ndarray of shape `(N, T_past, D_d)`.
           - 'future_features': np.ndarray of shape `(N, H, D_f)`.
        2. `targets_dict`:
           - 'subsidence': np.ndarray of shape `(N, H, O_subs)`.
           - 'gwl': np.ndarray of shape `(N, H, O_gwl)`.
        Where N is the total number of sequences, H is `forecast_horizon`,
        T_past is `time_steps`, D_s/d/f are feature dimensions, and
        O_subs/gwl are output dimensions.

    Raises
    ------
    ValueError
        If required columns are missing, or if data is insufficient
        to create any sequences.
    TypeError
        If `df` is not a Pandas DataFrame.
    """
    # Entry log
    vlog("Starting PINN data sequence preparation...",
         verbose=verbose, level=1)
    
    if verbose >= 1:
        logger.info("Starting PINN data sequence preparation...")
    df_proc = df.copy()

    # --- 1. Validate Input Parameters and Columns ---
    if verbose >= 2:
        logger.debug("Validating input parameters and columns.")

    # Validate essential columns
    vlog("Validating essential columns...",
         verbose=verbose, level=2)
    
    lon_col, lat_col = resolve_spatial_columns(
        df_proc, spatial_cols =spatial_cols, 
        lon_col=lon_col, 
        lat_col= lat_col
    )
    essential_cols = [time_col, lon_col, lat_col, subsidence_col, gwl_col]
    exist_features(
        df_proc, features=essential_cols, 
        message="Essential column(s) missing."
        )
    
    # check datetime since we are refering to time series data 
    # Check dt_col data type
    vlog("Validating time-series dataset...",
         verbose=verbose, level=2)
    
    check_datetime(
        df_proc,
        dt_cols= time_col, 
        ops="check_only",
        consider_dt_as="numeric",
        accept_dt=True, 
        allow_int=True, 
    )
    
    vlog("Managing feature column lists...",
         verbose=verbose, level=3)
    
    dynamic_cols = columns_manager(
        dynamic_cols, empty_as_none=False
        )
    exist_features(
        df_proc, features=dynamic_cols, 
        name="Dynamic feature column(s)"
    )
    
    static_cols = columns_manager(static_cols) # Allows None
    if static_cols:
        exist_features(
            df_proc, features=static_cols,
            name="Static feature column(s)")
    
    future_cols = columns_manager(future_cols) # Allows None
    if future_cols:
        exist_features(
            df_proc, features=future_cols, 
            name="Future feature column(s)")

    group_id_cols = columns_manager(group_id_cols) # Allows None
    if group_id_cols:
        exist_features(
            df_proc, features=group_id_cols, 
            name="Group ID column(s)"
            )

    vlog("Validating time_steps and forecast_horizon...",
         verbose=verbose, level=3)
    
    time_steps = validate_positive_integer(
        time_steps, "time_steps")
    forecast_horizon = validate_positive_integer(
        forecast_horizon, "forecast_horizon")
    
    # Convert time column to numerical representation
    # (e.g., days since epoch, or normalized year)
    # This is crucial for PINN as 't' is an input for differentiation.
    vlog("Converting time column to numeric values...",
         verbose=verbose, level=4)
    
    try:
        df_proc[time_col] = pd.to_datetime(
            df_proc[time_col], format=datetime_format)
        # Convert to a numerical representation, e.g., year with fraction
        df_proc[f"{time_col}_numeric"] = (
            df_proc[time_col].dt.year + 
            (df_proc[time_col].dt.dayofyear -1) / 
            (365 + df_proc[time_col].dt.is_leap_year)
        )
        numerical_time_col = f"{time_col}_numeric"
        if verbose >=2:
            logger.debug(
                f"Converted datetime column '{time_col}'"
                f" to numerical '{numerical_time_col}'.")
        vlog(f"Time column converted to '{numerical_time_col}'",
             verbose=verbose, level=5)
        
    except Exception as e:
        raise ValueError(
            f"Failed to convert or process time column '{time_col}'. "
            f"Ensure it's datetime-like or specify `datetime_format`. Error: {e}"
        )

    # --- 2. Group and Sort Data ---
    vlog("Grouping and sorting data...",
        verbose=verbose, level=2)
    
    if verbose >= 2:
        logger.debug(
            f"Grouping and sorting data. Group IDs:"
            f" {group_id_cols or 'None (single group)'}."
        )
    
    # Sort by group IDs (if any), then by time within each group
    sort_by_cols = [numerical_time_col]
    if group_id_cols:
        sort_by_cols = group_id_cols + sort_by_cols
    df_proc = df_proc.sort_values(by=sort_by_cols).reset_index(drop=True)

    if group_id_cols:
        grouped_data = df_proc.groupby(group_id_cols)
        group_keys = list(grouped_data.groups.keys())
        if verbose >= 1:
            logger.info(
                f"Data grouped by {group_id_cols} into {len(group_keys)} groups."
            )
    else:
        # Treat entire DataFrame as a single group
        grouped_data = [(None, df_proc)] # List of one tuple (key, group_df)
        group_keys = [None]
        if verbose >= 1:
            logger.info("Processing entire DataFrame as a single group.")

     
    # --- 3. First Pass: Calculate Total Number of Sequences ---
    
    total_sequences = 0
    min_len_per_group = time_steps + forecast_horizon
    valid_group_dfs = [] # Store dataframes of groups that are long enough

    vlog("Counting valid sequences in groups...",
        verbose=verbose, level=4)
     
    if verbose >= 2:
        logger.debug(
            "First pass: Calculating total sequences. Min"
            f" length per group: {min_len_per_group}."
        )

    for group_key in group_keys:
        group_df = grouped_data.get_group(group_key) if group_id_cols else df_proc
        
        key_str = group_key if group_key is not None else "<Full Dataset>"
        if len(group_df) < min_len_per_group:
            if verbose >= 5: # More detailed per-group logging
                logger.info(
                    f"Group '{key_str}' has {len(group_df)} points, less than "
                    f"min required ({min_len_per_group}). Skipping."
                )
            vlog(f"Group {key_str} too small:"
                 f" {len(group_df)} < {min_len_per_group}",
                 verbose=verbose, level=6)
            
            continue
        
        num_seq_in_group = len(group_df) - min_len_per_group + 1
        total_sequences += num_seq_in_group
        valid_group_dfs.append(group_df)
        if verbose >= 5:
            logger.debug(
                f"Group '{group_key if group_key else '<Full Dataset>'}' "
                f"will yield {num_seq_in_group} sequences."
            )
        vlog(
            f"Group {key_str} yields {total_sequences} seqs.",
             verbose=verbose, 
             level=6
            )
        
        
    if total_sequences == 0:
        raise ValueError(
            "No group has enough data points to create sequences with "
            f"time_steps={time_steps} and forecast_horizon={forecast_horizon}."
        )
    if verbose >= 1:
        logger.info(
            "Total valid sequences to be"
            f" generated: {total_sequences}.")

    vlog(f"Total sequences: {total_sequences}",
        verbose=verbose, level=1)
     
    # --- 4. Pre-allocate NumPy Arrays ---
    vlog("Pre-allocating arrays...",
         verbose=verbose, level=2)
    
    # Feature dimensions
    num_dynamic_feats = len(dynamic_cols)
    num_static_feats = len(static_cols) if static_cols else 0
    num_future_feats = len(future_cols) if future_cols else 0

    # Initialize arrays
    # Coords for the forecast horizon: (N, H, 3) for [t, x, y]
    coords_horizon_arr = np.zeros(
        (total_sequences, forecast_horizon, 3), dtype=np.float32
    )
    static_features_arr = np.zeros(
        (total_sequences, num_static_feats), dtype=np.float32
    )
    dynamic_features_arr = np.zeros(
        (total_sequences, time_steps, num_dynamic_feats),
        dtype=np.float32
    )
    future_features_arr = np.zeros(
        (total_sequences, forecast_horizon, num_future_feats),
        dtype=np.float32
    )
    target_subsidence_arr = np.zeros(
        (total_sequences, forecast_horizon, output_subsidence_dim),
        dtype=np.float32
    )
    target_gwl_arr = np.zeros(
        (total_sequences, forecast_horizon, output_gwl_dim), 
        dtype=np.float32
    )
    vlog("Arrays shapes set.",
         verbose=verbose, level=5)
    
    if verbose >= 2:
        logger.debug("Pre-allocated NumPy arrays for sequence data:")
        logger.debug(f"  Coords Horizon: {coords_horizon_arr.shape}")
        logger.debug(f"  Static Features: {static_features_arr.shape}")
        logger.debug(f"  Dynamic Features: {dynamic_features_arr.shape}")
        logger.debug(f"  Future Features: {future_features_arr.shape}")
        logger.debug(f"  Target Subsidence: {target_subsidence_arr.shape}")
        logger.debug(f"  Target GWL: {target_gwl_arr.shape}")

    # --- 5. Second Pass: Populate Arrays with Rolling Windows ---
    current_seq_idx = 0
    if verbose >= 2:
        logger.debug("Second pass: Populating sequence arrays...")
    
    vlog("Populating arrays with data...",
        verbose=verbose, level=2)
    
    for group_df in valid_group_dfs: # Iterate over pre-filtered valid groups
        group_t_coords = group_df[numerical_time_col].values
        group_x_coords = group_df[lon_col].values
        group_y_coords = group_df[lat_col].values

        # Normalization parameters for coordinates (per group if grouped)
        t_min, t_max = group_t_coords.min(), group_t_coords.max()
        x_min, x_max = group_x_coords.min(), group_x_coords.max()
        y_min, y_max = group_y_coords.min(), group_y_coords.max()
        
        if verbose >= 3:
            print()
            print_box(
                f"Group window: t {t_min}-{t_max}",
                width=_TW,
                align='center',
                border_char='+',
                horizontal_char='-',
                vertical_char='|',
                padding=1
            )

        group_static_vals = None
        if static_cols and num_static_feats > 0:
            # For static features, take the first row of the group
            # (assuming they are constant within a group)
            group_static_vals = group_df.iloc[0][
                static_cols].values.astype(np.float32)

        num_seq_in_this_group = len(group_df) - min_len_per_group + 1
        for i in range(num_seq_in_this_group):
            # --- Static Features ---
            if static_cols and num_static_feats > 0:
                static_features_arr[current_seq_idx] = group_static_vals

            # --- Dynamic Features (lookback window) ---
            dynamic_start_idx = i
            dynamic_end_idx = i + time_steps
            dynamic_features_arr[current_seq_idx] = group_df.iloc[
                dynamic_start_idx:dynamic_end_idx
            ][dynamic_cols].values.astype(np.float32)

            # --- Future Features & Target Coordinates (forecast horizon) ---
            horizon_start_idx = i + time_steps
            horizon_end_idx = i + time_steps + forecast_horizon
            
            if future_cols and num_future_feats > 0:
                future_features_arr[current_seq_idx] = group_df.iloc[
                    horizon_start_idx:horizon_end_idx
                ][future_cols].values.astype(np.float32)

            # Coordinates for the forecast horizon
            t_horizon = group_t_coords[horizon_start_idx:horizon_end_idx]
            x_horizon = group_x_coords[horizon_start_idx:horizon_end_idx]
            y_horizon = group_y_coords[horizon_start_idx:horizon_end_idx]

            if normalize_coords:
                # Avoid division by zero if min == max 
                # (e.g., single point in time/space for horizon)
                t_horizon = (t_horizon - t_min) / (t_max - t_min + 1e-9)
                x_horizon = (x_horizon - x_min) / (x_max - x_min + 1e-9)
                y_horizon = (y_horizon - y_min) / (y_max - y_min + 1e-9)
            
            coords_horizon_arr[current_seq_idx, :, 0] = t_horizon
            coords_horizon_arr[current_seq_idx, :, 1] = x_horizon
            coords_horizon_arr[current_seq_idx, :, 2] = y_horizon

            # --- Target Variables (forecast horizon) ---
            target_subsidence_arr[current_seq_idx] = group_df.iloc[
                horizon_start_idx:horizon_end_idx
            ][subsidence_col].values.reshape(
                forecast_horizon, output_subsidence_dim).astype(np.float32)
            
            target_gwl_arr[current_seq_idx] = group_df.iloc[
                horizon_start_idx:horizon_end_idx
            ][gwl_col].values.reshape(
                forecast_horizon, output_gwl_dim).astype(np.float32)

            
            
            if verbose >= 7: # Very detailed trace
                logger.debug(f"  Sequence {current_seq_idx}:")
                logger.debug("    Dynamic window:"
                             f" {dynamic_start_idx}-{dynamic_end_idx-1}")
                logger.debug(
                    "    Horizon window:"
                    f" {horizon_start_idx}-{horizon_end_idx-1}")
                logger.debug(
                    f"    Coords (first step):"
                    f" {coords_horizon_arr[current_seq_idx, 0, :]}")

            vlog(
                f"Seq {current_seq_idx}:"
                f" dyn {dynamic_start_idx}-{dynamic_end_idx-1},"
                f"hzn {horizon_start_idx}-{horizon_end_idx-1}",
                verbose=verbose, level=7
            )
            
            current_seq_idx += 1
    
    if verbose >= 1:
        logger.info("Successfully populated sequence arrays.")
    
    vlog("Data population complete.",
         verbose=verbose, level=1)
    
    inputs_dict = {
        'coords': coords_horizon_arr,
        'static_features': static_features_arr if num_static_feats > 0 else None,
        'dynamic_features': dynamic_features_arr,
        'future_features': future_features_arr if num_future_feats > 0 else None,
    }
    # Filter out None entries from inputs_dict if model call expects only present keys
    inputs_dict = {k: v for k, v in inputs_dict.items() if v is not None}


    targets_dict = {
        'subsidence': target_subsidence_arr,
        'gwl': target_gwl_arr
    }

    # --- Save to File (Optional) ---
    if savefile:
        vlog(f"\nPreparing to save sequence data to '{savefile}'...", 
             verbose=verbose, level=3
             )
        job_dict = {
            'static_data': static_features_arr,
            'dynamic_data': dynamic_features_arr,
            'future_data': future_features_arr,
            'subsidence': target_subsidence_arr,
            'gwl': target_gwl_arr, 
            'static_features': static_cols,
            'dynamic_features': dynamic_cols,
            'future_features': future_cols,
            'inputs_dict': inputs_dict, 
            'targets_dict': targets_dict, 
            'subsidence_col': subsidence_col,
            'spatial_features': spatial_cols,
            'lon_col': lon_col, 
            'lat_col': lat_col, 
            'time_col': time_col,
            'time_steps': time_steps,
            'forecast_horizon': forecast_horizon,
        }
        # Add version information if available
        try:
            job_dict.update(get_versions())
        except NameError: # If get_versions is not defined/imported
            vlog("\n  `get_versions` not found, version info not saved.", 
                 verbose=verbose, level =1 )

        try:
            # save_job to be a wrapper around joblib.dump
            save_job(job_dict, savefile, append_versions=False)
      
            if verbose >= 1:
                vlog(f"Sequence data dictionary successfully "
                      f"saved to '{savefile}'.", 
                      verbose=verbose, level=1
            )
        except Exception as e:
            vlog(f"Failed to save job dictionary to "
                  f"'{savefile}': {e}", verbose=verbose, level=1)
                
    if verbose >= 1:
        logger.info("PINN data sequence preparation completed.")
        if verbose >= 3:
            for key, arr in inputs_dict.items():
                logger.debug("  Final input '{key}' shape:"
                             f" {arr.shape if arr is not None else 'None'}")
            for key, arr in targets_dict.items():
                logger.debug(
                    f"  Final target '{key}' shape: {arr.shape}")
    
    vlog("PINN data sequence preparation successfully completed.",
         verbose=verbose, level=3)
    
    return inputs_dict, targets_dict

def process_pde_modes(
    pde_mode: Union[str, list, None], 
    enforce_consolidation: bool = False,
    pde_mode_config: Union[str, list, None] = None, 
    solo_return: bool=False, 
) -> list:
    """
    Process and validate the `pde_mode` argument to determine the active PDE modes.

    This function handles `pde_mode` inputs and processes them according to
    the following rules:
    - If the input is 'none', only the mode 'none' will be active.
    - If the input is 'both', it will set both 'consolidation' and 'gw_flow'.
    - If the input is not 'consolidation' and `enforce_consolidation` is True,
      it will issue a warning and fallback to using only 'consolidation'.
    - If `pde_mode_config` is provided, it overrides any other mode setting.

    Parameters
    ----------
    pde_mode : str, list of str, or None
        The desired PDE modes. Can be:
        - A string (e.g., 'consolidation', 'gw_flow', etc.)
        - A list of strings (e.g., ['consolidation', 'gw_flow'])
        - None (to set no active modes)
    enforce_consolidation : bool, default=False
        If True, the function ensures that 'consolidation' is the only
        mode active and issues a warning if another mode is passed.
    pde_mode_config : str, list of str, or None, optional
        If provided, overrides the `pde_mode` argument.

    Returns
    -------
    list of str
        A list of active PDE modes. The list will contain the modes in lowercase.
        If 'none' or 'both' is specified, these will be processed according to the logic.

    Raises
    ------
    TypeError
        If `pde_mode` is neither a string, a list of strings, nor None.

    Warnings
    --------
    If `enforce_consolidation` is True and a mode other than 'consolidation'
    is passed, a warning will be issued, and 'consolidation' will be used
    as the active mode instead.
    """
    # If pde_mode_config is provided, use it, otherwise fall back to pde_mode
    if pde_mode_config:
        pde_mode = pde_mode_config

    if isinstance(pde_mode, str):
        pde_modes_active = [pde_mode.lower()]
    elif isinstance(pde_mode, list):
        pde_modes_active = [p_type.lower() for p_type in pde_mode]
    elif pde_mode is None:
        pde_modes_active = ['none']  # Explicitly 'none' if None is provided
    else:
        raise TypeError("`pde_mode` must be a string, list of strings, or None.")

    # Handle special cases for "none" and "both"
    if "none" in pde_modes_active:  # If 'none' is present, override others
        pde_modes_active = ['none']
    if "both" in pde_modes_active:  # If 'both' is present, use both modes
        pde_modes_active = ['consolidation', 'gw_flow']

    # Enforce consolidation mode if specified
    if enforce_consolidation and 'consolidation' not in pde_modes_active:
        warnings.warn(
            "You have passed a mode other than 'consolidation'. "
            "Falling back to 'consolidation' as the active mode.",
            UserWarning
        )
        pde_modes_active = ['consolidation']

    # Ensure 'consolidation' is the only active mode if it was defaulted,
    # or if user explicitly selected only unsupported modes.
    if any(unsupported in pde_modes_active 
           for unsupported in ['gw_flow', 'both', 'none']) \
       and 'consolidation' not in pde_modes_active:
        # This case means decorator didn't force 'consolidation', so we do it here
        logger.info(
            f"Unsupported pde_mode '{pde_mode}' "
            "selected without 'consolidation'. "
            "PIHALNet will use 'consolidation' mode."
        )
        pde_modes_active = ['consolidation']

    if solo_return: 
        pde_modes_active = pde_modes_active[0]
        
    # Return the processed pde_modes_active
    return pde_modes_active

def check_and_rename_keys(inputs, y): # ranem to check_input_keys 
    """
    Helper function to check and rename keys in the inputs
    and target dictionaries.

    This function ensures that the necessary keys are present in both the 
    `inputs` and `y` dictionaries. If the keys for 'subsidence' or 'gwl' 
    are not found, it attempts to rename them from possible alternatives 
    like 'subs_pred' or 'gwl_pred'.

    Parameters
    ----------
    inputs : dict
        A dictionary containing the input data. The keys 'coords' and 
        'dynamic_features' are expected.
        
    y : dict
        A dictionary containing the target values. The keys 'subsidence' 
        and 'gwl' are expected, but they could also appear as 'subs_pred' 
        or 'gwl_pred'.

    Raises
    ------
    ValueError
        If required keys are missing in `inputs` or `y`, or if renaming 
        does not result in valid keys for 'subsidence' and 'gwl'.
    """
    
    # Check if 'coords' and 'dynamic_features' are in inputs
    if 'coords' not in inputs or inputs['coords'] is None:
        raise ValueError("Input 'coords' is missing or None.")
    if 'dynamic_features' not in inputs or inputs['dynamic_features'] is None:
        raise ValueError("Input 'dynamic_features' is missing or None.")
    
    # Check for 'subsidence' in y, allow renaming from 'subs_pred'
    # just check whether subsidence or subs_pred in y 
    
    if 'subsidence' not in y and 'subs_pred' in y:
        y['subsidence'] = y.pop('subs_pred')
    if 'subsidence' not in y:
        # here by explicit to hel the 
        raise ValueError("Target 'subsidence' is missing or None.") 
        # use that user should provide subsidene or subs_pred 
    
    # Check for 'gwl' in y, allow renaming from 'gwl_pred'
    # just check whether gwl or gwl_pred one,its in the y 
    # 
    if 'gwl_pred' not in y and 'gwl' in y:
        # no need to rename yet, later this will handle si just check only 
        y['gwl_pred'] = y.pop('gwl')  
    if 'gwl' not in y:
        raise ValueError("Target 'gwl' is missing or None.")
    
    return inputs, y

def check_required_input_keys(inputs, y=None, message=None ):
    """
    Helper function to check and rename keys in the inputs
    and target dictionaries.

    This function ensures that the necessary keys are present in both the 
    `inputs` and `y` dictionaries. If the keys for 'subsidence' or 'gwl' 
    are not found, it attempts to rename them from possible alternatives 
    like 'subs_pred' or 'gwl_pred'.

    Parameters
    ----------
    inputs : dict
        A dictionary containing the input data. The keys 'coords' and 
        'dynamic_features' are expected.
        
    y : dict
        A dictionary containing the target values. The keys 'subsidence' 
        and 'gwl' are expected, but they could also appear as 'subs_pred' 
        or 'gwl_pred'.
        
    message : str, optional 
       Message to raise error when inputs/y are not dictionnary. 
       
    Raises
    ------
    ValueError
        If required keys are missing in `inputs` or `y`, or if renaming 
        does not result in valid keys for 'subsidence' and 'gwl'.
    """
    if inputs is not None: 
        if not isinstance (inputs, dict): 
            message = message or (
                "Inputs must be a dictionnary containing"
                " 'coords' and 'dynamic_features'."
                f" Got {type(inputs).__name__!r}")
            raise TypeError (message )
        
        # Check if 'coords' and 'dynamic_features' are in inputs
        if 'coords' not in inputs or inputs['coords'] is None:
            raise ValueError("Input 'coords' is missing or None.")
        if 'dynamic_features' not in inputs or inputs['dynamic_features'] is None:
            raise ValueError("Input 'dynamic_features' is missing or None.")
    
    if y is not None: 
        if not isinstance (y, dict): 
            message = message or (
                "Target `y` must be a dictionnary containing"
                " 'subs_pred/subsidence' and 'gwl/gwl_red'."
                f" Got {type(y).__name__!r}"
            )
            raise TypeError (message)
            
        # Check for 'subsidence' in y, allow renaming from 'subs_pred'
        if 'subsidence' not in y and 'subs_pred' not in y :
            raise ValueError(
                "Target 'subsidence' is missing or None."
                " Please provide 'subsidence' or 'subs_pred'.")
        
        # Check for 'gwl' in y, allow renaming from 'gwl_pred'
        if 'gwl' not in y and 'gwl_pred' not in y:
            raise ValueError(
                "Target 'gwl' is missing or None."
                " Please provide 'gwl' or 'gwl_pred'.")
    
    return inputs, y

# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com> 

"""
Physics-Informed Neural Network (PINN) Utility functions.
"""
import os
import logging 
from typing import List, Tuple, Optional, Union, Dict, Any
from typing import Sequence,  Callable
import warnings # noqa 
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable  

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from ..._fusionlog import fusionlog
from ...api.util import get_table_size 
from ...core.checks import ( 
    exist_features, 
    check_datetime, 
)
from ...core.io import SaveFile 
from ...core.handlers import columns_manager
from ...utils.validator import validate_positive_integer 
from ...metrics.utils import compute_quantile_diagnostics 
from ...decorators import isdf
from ...utils.data_utils import mask_by_reference  
from ...utils.deps_utils import ensure_pkg 
from ...utils.forecast_utils import normalize_for_pinn
from ...utils.generic_utils import ( 
    print_box, vlog, select_mode, rename_dict_keys
)
from ...utils.geo_utils import resolve_spatial_columns 
from ...utils.io_utils import save_job 
from ...utils.sequence_utils import check_sequence_feasibility

from .. import KERAS_BACKEND, KERAS_DEPS

Model = KERAS_DEPS.Model
Tensor = KERAS_DEPS.Tensor

tf_shape = KERAS_DEPS.shape
tf_convert_to_tensor =KERAS_DEPS.convert_to_tensor 
tf_float32 = KERAS_DEPS.float32 
tf_cast =KERAS_DEPS.cast
tf_concat =KERAS_DEPS.concat
tf_expand_dims =KERAS_DEPS.expand_dims
tf_debugging =KERAS_DEPS.debugging
tf_fill = KERAS_DEPS.fill 
tf_reshape = KERAS_DEPS.reshape 

logger = fusionlog().get_fusionlab_logger(__name__)
_TW = get_table_size()

all__= [
        'format_pihalnet_predictions',
        'prepare_pinn_data_sequences',
        'format_pinn_predictions', 
        'extract_txy_in', 
        'extract_txy', 
        'plot_hydraulic_head',
]

@SaveFile 
def format_pinn_predictions(
    predictions: Optional[Dict[str, 'Tensor']] = None,
    model: Optional['Model'] = None,
    model_inputs: Optional[Dict[str, 'Tensor']] = None,
    y_true_dict: Optional[Dict[str, Union[np.ndarray, 'Tensor']]] = None,
    target_mapping: Optional[Dict[str, str]] = None,
    include_gwl: bool = True,
    include_coords: bool = True,
    quantiles: Optional[List[float]] = None,
    forecast_horizon: Optional[int] = None,
    output_dims: Optional[Dict[str, int]] = None,
    ids_data_array: Optional[Union[np.ndarray, 'pd.DataFrame']] = None,
    ids_cols: Optional[List[str]] = None,
    ids_cols_indices: Optional[List[int]] = None,
    scaler_info: Optional[Dict[str, Dict[str, Any]]] = None,
    coord_scaler: Optional[Any] = None,
    evaluate_coverage: bool = False,
    coverage_quantile_indices: Tuple[int, int] = (0, -1),
    savefile: Optional[str] = None,
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    name: Optional[str]=None, 
    model_name: Optional[str] = None,
    stop_check: Callable[[], bool] = None, 
    verbose: int = 0,
    **kwargs
) -> 'pd.DataFrame':
    r"""Formats PINN model predictions into a structured pandas DataFrame.

    This is a general-purpose utility for transforming raw model outputs
    (from models like PIHALNet or TransFlowSubsNet) into a long-format
    DataFrame suitable for analysis, visualization, or export.

    This is a powerful, general-purpose utility for transforming raw
    model outputs into a long-format DataFrame suitable for analysis,
    visualization, or export. It handles multi-target outputs (e.g.,
    subsidence and GWL), point or quantile forecasts, and can
    optionally include true values, coordinate information, and other
    metadata. It also supports inverse-scaling of predictions and
    evaluation of quantile coverage.

    Parameters
    ----------
    predictions : dict of Tensors, optional
        The dictionary of prediction tensors, typically returned by a
        model's ``.predict()`` method. Keys should match the model's
        output layer names (e.g., ``'subs_pred'``, ``'gwl_pred'``).
        If ``None``, predictions are generated internally using the
        `model` and `model_inputs` arguments. Default is ``None``.
    model : keras.Model, optional
        A compiled Keras model instance used to generate predictions
        if the `predictions` dictionary is not provided.
        Default is ``None``.
    model_inputs : dict of Tensors, optional
        A dictionary of input tensors matching the model's signature,
        required only if `predictions` is ``None``. Default is ``None``.
    y_true_dict : dict, optional
        A dictionary containing the ground-truth target arrays, keyed
        by their base names (e.g., ``'subsidence'``, ``'gwl'``).
        If provided, an ``<target>_actual`` column will be added to
        the output DataFrame for comparison. Default is ``None``.
    target_mapping : dict, optional
        A custom mapping from model output keys to desired base names in
        the DataFrame columns. For example:
        ``{'subs_pred': 'subsidence_mm', 'gwl_pred': 'head_m'}``.
        Default is ``None``.
    include_gwl : bool, default=True
        Toggles the inclusion of groundwater level (GWL) predictions
        in the final DataFrame.
    include_coords : bool, default=True
        Toggles the inclusion of the spatio-temporal coordinate columns
        (``coord_t``, ``coord_x``, ``coord_y``) in the final DataFrame.
    quantiles : list of float, optional
        The list of quantile levels (e.g., ``[0.1, 0.5, 0.9]``) that
        the model predicted. This is crucial for correctly parsing
        probabilistic forecasts. Default is ``None``.
    forecast_horizon : int, optional
        The length of the forecast horizon. If ``None``, it is inferred
        from the shape of the prediction tensors. Default is ``None``.
    output_dims : dict of str, optional
        A dictionary specifying the feature dimension of each target,
        e.g., ``{'subs_pred': 1, 'gwl_pred': 1}``. If ``None``, it's
        inferred from the tensor shapes. Default is ``None``.
    ids_data_array : np.ndarray or pd.DataFrame, optional
        An array or DataFrame containing static identifiers (e.g., well
        IDs, site categories) for each sample. Its length must match
        the number of samples in the prediction. Default is ``None``.
    ids_cols : list of str, optional
        A list of column names for the `ids_data_array`. Required if
        `ids_data_array` is a NumPy array. Default is ``None``.
    ids_cols_indices : list of int, optional
        A list of column indices to select from `ids_data_array` if it
        is a NumPy array. Default is ``None``.
    scaler_info : dict, optional
        A dictionary providing the necessary information to perform
        inverse scaling on a per-target basis. Each key should be a
        target name (e.g., 'subsidence') and its value a dictionary
        containing ``{'scaler': obj, 'all_features': list, 'idx': int}``.
        Default is ``None``.
    coord_scaler : object, optional
        A fitted scikit-learn-like scaler object used to perform an
        inverse transform on the coordinate columns. Default is ``None``.
    evaluate_coverage : bool, default=False
        If ``True`` and quantile predictions are present, calculates the
        unconditional coverage of the prediction interval.
    coverage_quantile_indices : tuple of (int, int), default=(0, -1)
        The indices of the lower and upper quantiles in the sorted
        `quantiles` list to use for the coverage calculation. Default
        is ``(0, -1)``, which corresponds to the full range.
    savefile : str, optional
        If a file path is provided, the final DataFrame is saved to a
        CSV file at this location. Default is ``None``.
    name: str or None 
        Name of the prediction. Name is used to format the output of 
        the data and coverage result if applicable. 
    model_name: str, None, 
        Name of the model. 
        
    verbose : int, default=0
        The verbosity level, from 0 (silent) to 5 (trace every step).
    **kwargs: dict,  
        Additional keyword arguments for future extensions. 
        
    Returns
    -------
    pd.DataFrame
        A long-format DataFrame where each row represents a single
        forecast step for a single sample. Columns include sample and
        step identifiers, coordinates, predictions, and optionally
        actuals and metadata.

    Notes
    -----
    - The function returns a column-aligned DataFrame, which simplifies
      subsequent analysis and plotting.
    - For quantile forecasts, prediction columns are named using the
      pattern ``<target_name>_q<quantile*100>``, e.g., ``subsidence_q5``,
      ``subsidence_q50``, ``subsidence_q95``.
    - For point forecasts, the column is named ``<target_name>_pred``.

    See Also
    --------
    fusionlab.plot.forecast.plot_forecasts : A powerful utility for
        visualizing the DataFrame produced by this function.
    """
    #
    # This acts as a backward-compatible wrapper.
    return format_pihalnet_predictions(
        pihalnet_outputs=predictions,
        model=model,
        model_inputs=model_inputs,
        y_true_dict=y_true_dict,
        target_mapping=target_mapping,
        include_gwl=include_gwl,
        include_coords=include_coords,
        quantiles=quantiles,
        forecast_horizon=forecast_horizon,
        output_dims=output_dims,
        ids_data_array=ids_data_array,
        ids_cols=ids_cols,
        ids_cols_indices=ids_cols_indices,
        scaler_info=scaler_info,
        coord_scaler=coord_scaler,
        evaluate_coverage=evaluate_coverage,
        coverage_quantile_indices=coverage_quantile_indices,
        savefile=savefile,
        name =name, 
        model_name=model_name,
        verbose=verbose,
        _logger = _logger, 
        stop_check = stop_check, 
        **kwargs 
    )

@SaveFile
def format_pihalnet_predictions(
    pihalnet_outputs: Optional[Dict[str, "Tensor"]] = None,
    model: Optional["Model"] = None,
    model_inputs: Optional[Dict[str, "Tensor"]] = None,
    y_true_dict: Optional[Dict[str, Union[np.ndarray, "Tensor"]]] = None,
    target_mapping: Optional[Dict[str, str]] = None,
    include_gwl: bool = True,
    include_coords: bool = True,
    quantiles: Optional[List[float]] = None,
    forecast_horizon: Optional[int] = None,
    output_dims: Optional[Dict[str, int]] = None,
    ids_data_array: Optional[Union[np.ndarray, 'pd.DataFrame']] = None,
    ids_cols: Optional[List[str]] = None,
    ids_cols_indices: Optional[List[int]] = None,
    scaler_info: Optional[Dict[str, Dict[str, Any]]] = None,
    coord_scaler: Optional[Any] = None,
    evaluate_coverage: bool = False,
    coverage_quantile_indices: Tuple[int, int] = (0, -1),
    savefile: Optional[str] = None,
    name: Optional[str]=None, 
    model_name: Optional[str] = None, 
    apply_mask: bool=False,
    mask_values: Optional [Union[float, int]]=None,
    mask_fill_value: Optional[float]=None,
    verbose: int = 0, 
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None, 
    stop_check : Callable [[], bool] =None, 
    **kwargs
) -> pd.DataFrame:
    r"""
    Formats PIHALNet/GeoPriorSubsNet predictions into a structured 
    pandas DataFrame, handling inversion, quantiles, and coordinates.

    This function is the core formatter. It:
    1. Gets model outputs (or uses provided ones).
    2. Unpacks 'data_final' if `model_name` is 'geoprior'.
    3. Inverse-transforms all prediction and actual arrays using `scaler_info`.
    4. Builds a long-format DataFrame with sample_idx and forecast_step.
    5. Appends inverted quantile/point predictions.
    6. Appends inverted actual values.
    7. Appends inverted coordinates.
    8. Appends static/ID columns.
    9. Evaluates coverage on the inverted data.
    
    Parameters
    ----------
    pihalnet_outputs : dict, optional
        Raw output from `model.predict()`. If None, `model` and 
        `model_inputs` must be provided.
    model : tf.keras.Model, optional
        Trained model instance (if `pihalnet_outputs` is None).
    model_inputs : dict, optional
        Inputs for the model to generate predictions 
        (if `pihalnet_outputs` is None).
    y_true_dict : dict, optional
        Dictionary of true target arrays (e.g., {'subs_pred': y_true_s}).
        Required for including actuals and evaluating coverage.
    target_mapping : dict, optional
        Maps prediction keys to base names for DataFrame columns.
        Default: {'subs_pred': 'subsidence', 'gwl_pred': 'gwl'}.
    include_gwl : bool, default True
        Whether to include 'gwl_pred' in the final DataFrame.
    include_coords : bool, default True
        Whether to include 'coord_t', 'coord_x', 'coord_y' columns.
    quantiles : list[float], optional
        List of quantiles (e.g., [0.1, 0.5, 0.9]). If provided,
        quantile columns (e.g., 'subsidence_q10') are created.
    forecast_horizon : int, optional
        The forecast horizon length (H). If not provided, it's
        inferred from the prediction array's shape.
    output_dims : dict, optional
        Maps prediction keys to their output dimension (O).
        E.g., {'subs_pred': 1, 'gwl_pred': 1}. Crucial for
        correctly splitting GeoPrior outputs and reshaping.
    ids_data_array : np.ndarray or pd.DataFrame, optional
        Static/ID data (e.g., original coordinates) to merge.
        Must have the same number of samples (B) as predictions.
    ids_cols : list[str], optional
        Column names if `ids_data_array` is a DataFrame.
    ids_cols_indices : list[int], optional
        Column indices if `ids_data_array` is a NumPy array.
    scaler_info : dict, optional
        Dictionary for inverse scaling, structured as:
        { 'subsidence': {'scaler': scaler_obj, 'idx': 0, 'all_features': [...]},
          'gwl': {'scaler': scaler_obj, 'idx': 1, 'all_features': [...]} }
    coord_scaler : sklearn.preprocessing.Scaler, optional
        A *fitted* scaler object for inverse transforming the 'coords' tensor.
    evaluate_coverage : bool, default False
        If True, calculates coverage percentage for quantiles.
    coverage_quantile_indices : tuple[int, int], default (0, -1)
        Indices of the low and high quantiles in the `quantiles` list
        to use for coverage (e.g., 0 and -1 for 10th and 90th).
    savefile : str, optional
        If provided, saves the final DataFrame to this path.
    model_name : str, optional
        Specifies the model type. If 'geoprior' or 'geopriorsubsnet',
        triggers unpacking of the 'data_final' output.
    apply_mask : bool, default False
        If True, masks predictions based on `mask_values` in the
        first target's `_actual` column.
    mask_values : float or int, optional
        The value in the `_actual` column to trigger masking.
    mask_fill_value : float, optional
        The value to replace masked predictions with (e.g., np.nan).
    verbose : int, default 0
        Logging verbosity.
    _logger : logging.Logger or callable, optional
        Logger object.
    stop_check : callable, optional
        Function to check for early stopping.
        
    Returns
    -------
    pd.DataFrame
        A long-format DataFrame with predictions, actuals, and coordinates.
    """
    vlog(f"Starting model prediction formatting (verbose={verbose}).",
         level=3, verbose=verbose, logger=_logger)

    # --- 1. Obtain Model Predictions (if not provided) ---
    if pihalnet_outputs is None:
        if model is None or model_inputs is None:
            raise ValueError(
                "If 'pihalnet_outputs' is None, both 'model' and "
                "'model_inputs' must be provided."
            )
        vlog("  Predictions not provided, generating from model...",
             level=4, verbose=verbose, logger=_logger)
        try:
            pihalnet_outputs = model.predict(model_inputs, verbose=0)
            if not isinstance(pihalnet_outputs, dict):
                 raise ValueError(
                     "Model output is not a dictionary as expected.")
            vlog("  Model predictions generated.", level=5,
                 verbose=verbose, logger=_logger)
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate predictions from model: {e}"
            ) from e
        
        if stop_check and stop_check():
            raise InterruptedError("Model prediction aborted.")
            
    # --- 2. Unpack GeoPrior Model Output (if needed) ---
    is_geoprior = str(model_name).lower().strip() in ('geoprior', 'geopriorsubsnet')
    if is_geoprior:
        vlog(f"GeoPrior model detected (model_name='{model_name}'). "
             "Unpacking 'data_final' tensor.",
             level=3, verbose=verbose, logger=_logger)
        
        if output_dims is None:
            output_dims = {'subs_pred': 1, 'gwl_pred': 1}
            vlog("  `output_dims` not provided, defaulting to "
                 "{'subs_pred': 1, 'gwl_pred': 1}.",
                 level=4, verbose=verbose, logger=_logger)
        
        pihalnet_outputs = _unpack_geoprior_outputs(
            geoprior_outputs=pihalnet_outputs,
            output_dims=output_dims,
            quantiles=quantiles,
            verbose=verbose,
            _logger=_logger
        )

    # --- 3. Ensure all data is NumPy & INVERSE TRANSFORM ---
    if target_mapping is None:
        target_mapping = {'subs_pred': 'subsidence', 'gwl_pred': 'gwl'}
        vlog("  `target_mapping` not provided, using defaults.",
             level=4, verbose=verbose, logger=_logger)
             
    if not include_gwl:
        target_mapping.pop('gwl_pred', None)

    # Process predictions
    processed_preds = {}
    if pihalnet_outputs:
        vlog("  Converting and inverse-transforming predictions...",
             level=4, verbose=verbose, logger=_logger)
        for pred_key, base_name in target_mapping.items():
            if pred_key in pihalnet_outputs:
                val_tensor = pihalnet_outputs[pred_key]
                pred_array = val_tensor.numpy() if hasattr(
                    val_tensor, 'numpy') else np.asarray(val_tensor)
                
                inverted_array = _inverse_transform_array(
                    pred_array, base_name, scaler_info, verbose, _logger
                )
                processed_preds[pred_key] = inverted_array
            else:
                 vlog(f"  [WARN] Prediction key '{pred_key}' not found "
                      "in model outputs.",
                      level=2, verbose=verbose, logger=_logger)
    
    if not processed_preds:
         vlog("  No valid prediction keys found in model output. "
              "Returning empty DF.",
              level=1, verbose=verbose, logger=_logger)
         return pd.DataFrame()

    # Process actuals (y_true)
    processed_actuals = {}
    if y_true_dict:
        vlog("  Converting and inverse-transforming actuals (y_true)...",
             level=4, verbose=verbose, logger=_logger)
        for true_key, base_name in target_mapping.items():
            
            # --- START FIX ---
            # Try to get the tensor using the pred_key ('subs_pred') first
            val_tensor = y_true_dict.get(true_key)
            if val_tensor is None:
                # If not found, try getting it with the base_name ('subsidence')
                val_tensor = y_true_dict.get(base_name)
            
            if val_tensor is not None:
            # --- END FIX ---
                true_array = val_tensor.numpy() if hasattr(
                    val_tensor, 'numpy') else np.asarray(val_tensor)
                
                inverted_array = _inverse_transform_array(
                    true_array, base_name, scaler_info, verbose, _logger
                )
                # Use true_key ('subs_pred') to align with processed_preds
                processed_actuals[true_key] = inverted_array
            else:
                vlog(f"  [WARN] True data key '{true_key}' (or '{base_name}') "
                     "not found in y_true_dict.",
                     level=2, verbose=verbose, logger=_logger)

    # --- 4. Initialize DataFrame (Base) ---
    first_pred_key = list(processed_preds.keys())[0]
    first_pred_array = processed_preds[first_pred_key]
    
    num_samples = first_pred_array.shape[0]
    H_inferred = forecast_horizon or first_pred_array.shape[1]
    
    if num_samples == 0 or H_inferred == 0:
        vlog("  No samples or zero forecast horizon. Returning empty DF.",
             level=1, verbose=verbose, logger=_logger)
        return pd.DataFrame()

    vlog(f"  Formatting for {num_samples} samples, Horizon={H_inferred}.",
         level=4, verbose=verbose, logger=_logger)

    sample_indices = np.repeat(np.arange(num_samples), H_inferred)
    forecast_steps = np.tile(np.arange(1, H_inferred + 1), num_samples)
    
    all_data_dfs = [
        pd.DataFrame({
            'sample_idx': sample_indices,
            'forecast_step': forecast_steps
        })
    ]

    # --- 5. Populate DataFrame with INVERTED Data ---
    if output_dims is None:
        vlog("  [WARN] `output_dims` is None. Inferring from arrays. "
             "This may be unreliable for multi-output models.",
             level=2, verbose=verbose, logger=_logger)
        output_dims = {}
        for k, v in processed_preds.items():
            output_dims[k] = v.shape[-1] if quantiles is None else v.shape[-1]
        
    if quantiles is not None:
        df_part = _add_quantiles_to_df(
            processed_preds, target_mapping, quantiles,
            output_dims, num_samples, H_inferred, verbose, _logger
        )
        all_data_dfs.append(df_part)
    else:
        df_part = _add_point_preds_to_df(
            processed_preds, target_mapping, output_dims,
            num_samples, H_inferred, verbose, _logger
        )
        all_data_dfs.append(df_part)

    if processed_actuals:
        df_part = _add_actuals_to_df(
            processed_actuals, target_mapping, output_dims,
            num_samples, H_inferred, verbose, _logger
        )
        all_data_dfs.append(df_part)

    # # --- 6. Add Coordinates (inverted inside helper) ---
    df_coord_cols = []
    if include_coords and model_inputs:
        df_part, df_coord_cols = _add_coords_to_df(
            model_inputs=model_inputs,
            coord_scaler=coord_scaler,
            num_samples=num_samples,
            H_inferred=H_inferred,
            verbose=verbose,
            _logger=_logger,
            force_future_from=kwargs.get("forecast_start_year"),
        )
        all_data_dfs.append(df_part)

    # --- 7. Add IDs (no inversion needed) ---
    df_part = _add_ids_to_df(
        ids_data_array, ids_cols, ids_cols_indices,
        num_samples, H_inferred, verbose, _logger
    )
    all_data_dfs.append(df_part)
    
    if stop_check and stop_check():
        raise InterruptedError("DataFrame population aborted.")

    # --- 8. Concatenate all DataFrames ---
    final_df = pd.concat(all_data_dfs, axis=1)

    # --- 10. Optional Masking ---
    if apply_mask:
        vlog("  Applying mask to prediction columns...",
             level=4, verbose=verbose, logger=_logger)
        if mask_values is None or mask_fill_value is None:
            raise ValueError(
                "When apply_mask=True, both mask_values and "
                "mask_fill_value must be provided."
            )
        
        # we assume you only ever want to mask against your first target:
        first_base = list(target_mapping.values())[0]
        ref_col    = f"{first_base}_actual"
        if ref_col not in final_df.columns:
            vlog(f"  [WARN] Masking reference column '{ref_col}' not in "
                 "DataFrame. Skipping masking.",
                 level=2, verbose=verbose, logger=_logger)
        else:
            # collect all the forecast columns you produced
            mask_cols = [
                c for c in final_df.columns if
                any(c.startswith(f"{base}_q") for base in target_mapping.values()) or
                any(c.startswith(f"{base}_pred") for base in target_mapping.values())
            ]
            
            if not mask_cols:
                vlog("  [WARN] No maskable forecast columns found in final_df. "
                     "Skipping masking.",
                     level=2, verbose=verbose, logger=_logger)
            else:
                try:
                    # Use the imported mask_by_reference function
                    final_df = mask_by_reference(
                        data=final_df,
                        ref_col=ref_col,
                        values=mask_values,
                        find_closest=False,
                        fill_value=mask_fill_value,
                        mask_columns=mask_cols,
                        error="ignore",
                        inplace=False
                    )
                    vlog(f"  Successfully applied mask to {len(mask_cols)} "
                         f"columns based on '{ref_col}'.",
                         level=4, verbose=verbose, logger=_logger)
                except Exception as e:
                    vlog(f"  [WARN] Failed to apply mask: {e}",
                         level=2, verbose=verbose, logger=_logger)

    # --- 9. Evaluate Coverage (now compares inverted vs. inverted) ---
    if evaluate_coverage and quantiles and processed_actuals:
        _evaluate_coverage(
            df=final_df,
            target_mapping=target_mapping,
            q_indices=coverage_quantile_indices,
            savefile=savefile,
            name=name,
            verbose=verbose,
            _logger=_logger
        )


    if stop_check and stop_check():
        raise InterruptedError("Metric/Masking aborted.")

    vlog("Model prediction formatting to DataFrame complete.",
         level=3, verbose=verbose, logger=_logger)
    
    return final_df


def _format_target_predictions(
    predictions_np: np.ndarray,
    num_samples: int,
    H: int, # Horizon
    O: int, # Output dim for this specific target
    base_target_name: str,
    quantiles: Optional[List[float]],
    verbose: int = 0
) -> Tuple[List[str], pd.DataFrame]:
    """Helper to format predictions for a single target variable."""
    pred_cols_names = []
    
    # Expected input shapes to this helper:
    # Point: (N, H, O)
    # Quantile: (N, H, Q, O) OR (N, H, Q) if O=1 was pre-squeezed
    
    if quantiles:
        num_q = len(quantiles)
        # Ensure predictions_np is (N, H, Q, O)
        if ( 
                predictions_np.ndim == 3  
                and predictions_np.shape[-1] == num_q 
                and O == 1
            ):
            # Case: (N, H, Q), implies O=1
            preds_to_process = np.expand_dims(predictions_np, axis=-1) # (N,H,Q,1)
        elif predictions_np.ndim == 3 and predictions_np.shape[-1] == num_q * O:
            # Case: (N, H, Q*O)
            preds_to_process = predictions_np.reshape((num_samples, H, num_q, O))
        elif ( 
                predictions_np.ndim == 4 
                and predictions_np.shape[2] == num_q 
                and predictions_np.shape[3] == O
            ):
            # Case: (N, H, Q, O) - already correct
            preds_to_process = predictions_np
        else:
            raise ValueError(
                f"Unexpected quantile prediction shape for {base_target_name}: "
                f"{predictions_np.shape}. Expected compatible with N={num_samples}, "
                f"H={H}, Q={num_q}, O={O}")

        # Now preds_to_process is (N, H, Q, O)
        df_data_for_concat = []
        for o_idx in range(O):
            for q_idx, q_val in enumerate(quantiles):
                col_name = f"{base_target_name}"
                if O > 1: col_name += f"_{o_idx}"
                col_name += f"_q{int(q_val*100)}"
                pred_cols_names.append(col_name)
                df_data_for_concat.append(
                    preds_to_process[:, :, q_idx, o_idx].reshape(-1)
                )
        pred_df_part = pd.DataFrame(
            dict(zip(pred_cols_names, df_data_for_concat))
        )

    else: # Point forecast
        # predictions_np should be (N, H, O)
        if predictions_np.ndim !=3 or predictions_np.shape[-1] != O:
            raise ValueError(
                f"Unexpected point prediction shape for {base_target_name}: "
               f"{predictions_np.shape}. Expected (N,H,O) with O={O}")
            
        df_data_for_concat = []
        for o_idx in range(O):
            col_name = f"{base_target_name}"
            if O > 1: col_name += f"_{o_idx}"
            col_name += "_pred"
            pred_cols_names.append(col_name)
            df_data_for_concat.append(
                predictions_np[:, :, o_idx].reshape(-1)
            )
        pred_df_part = pd.DataFrame(
            dict(zip(pred_cols_names, df_data_for_concat))
        )
        
    return pred_cols_names, pred_df_part

def _unpack_geoprior_outputs(
    geoprior_outputs: Dict[str, Tensor],
    output_dims: Dict[str, int],
    quantiles: Optional[List[float]],
    verbose: int = 0,
    _logger: Any = None
) -> Dict[str, Tensor]:
    """
    Unpacks the nested 'data_final' tensor from GeoPriorSubsNet
    into a flat dictionary expected by the formatter.
    """
    if 'data_final' not in geoprior_outputs:
        vlog("  [WARN] 'data_final' key not found in GeoPrior model output. "
             "Assuming outputs are already flat.",
             level=2, verbose=verbose, logger=_logger)
        return geoprior_outputs  # Return as-is

    data_final_tensor = geoprior_outputs['data_final']
    
    # Get the split index for subsidence
    s_dim = output_dims.get('subs_pred')
    if s_dim is None:
        vlog("  [WARN] 'output_dims' did not contain 'subs_pred'. "
             "Assuming subsidence output_dim = 1.",
             level=2, verbose=verbose, logger=_logger)
        s_dim = 1  # Default to 1 as a fallback

    # Split the tensor based on whether quantiles are present
    if quantiles is not None:
        # Shape is (B, H, Q, O_total)
        vlog(f"  Splitting 4D quantile tensor at final axis index {s_dim}.",
             level=4, verbose=verbose, logger=_logger)
        s_pred_tensor = data_final_tensor[..., :s_dim]
        h_pred_tensor = data_final_tensor[..., s_dim:]
    else:
        # Shape is (B, H, O_total)
        vlog(f"  Splitting 3D point-forecast tensor at final axis index {s_dim}.",
             level=4, verbose=verbose, logger=_logger)
        s_pred_tensor = data_final_tensor[..., :s_dim]
        h_pred_tensor = data_final_tensor[..., s_dim:]

    # Create the new flat dictionary
    unpacked_outputs = {
        'subs_pred': s_pred_tensor,
        'gwl_pred': h_pred_tensor
    }
    
    # Pass through any other keys that aren't the standard nested ones
    for k, v in geoprior_outputs.items():
        if k not in ['data_final', 'data_mean', 'phys_mean_raw']:
            unpacked_outputs[k] = v
            
    return unpacked_outputs

def _inverse_transform_array(
    array: np.ndarray,
    base_name: str,
    scaler_info: Optional[Dict[str, Dict[str, Any]]],
    verbose: int = 0,
    _logger: Any = None
) -> np.ndarray:
    """
    Helper to inverse-transform a single numpy array using scaler_info.
    
    This function finds the correct scaler and column index from
    scaler_info and applies the inverse min-max scaling.
    """
    if scaler_info is None or base_name not in scaler_info:
        # No scaler info for this target, return original array
        vlog(f"    - No scaler_info found for '{base_name}'. "
             "Returning original (scaled) array.",
             level=4, verbose=verbose, logger=_logger)
        return array

    try:
        info = scaler_info[base_name]
        scaler = info['scaler']
        col_index = info['idx']

        # Get the specific min and scale for this column from the scaler
        # scaler.min_ is the 'intercept'
        # scaler.scale_ is the 'coefficient'
        min_val = scaler.min_[col_index]
        scale_val = scaler.scale_[col_index] 

        # --- THIS IS THE FIX ---
        # The inverse of (X * scale_) + min_ is (X - min_) / scale_
        if scale_val == 0:
            # Handle case where scale is zero (all original values were the same)
            # The inverse is the original constant value, which we get from data_min_
            try:
                original_min = scaler.data_min_[col_index]
            except AttributeError:
                # Fallback if data_min_ is not available (should be on a fitted scaler)
                original_min = 0 # This is a guess, but division by zero is worse
                vlog(f"  [WARN] Scaler for '{base_name}' has scale_=0 but "
                     "no 'data_min_' attribute. Inverse may be incorrect.",
                     level=2, verbose=verbose, logger=_logger)
            inverted_array = np.full_like(array, original_min)
        else:
            inverted_array = (array - min_val) / scale_val
        
        vlog(f"    - Applied inverse transform to '{base_name}' array "
             f"(min_attr: {min_val:.4f}, scale_attr: {scale_val:.4f})",
             level=5, verbose=verbose, logger=_logger)
        
        return inverted_array
    
    except AttributeError:
        vlog(f"  [WARN] Scaler for '{base_name}' is missing 'min_' or 'scale_'"
             " attributes. Returning original array.",
             level=2, verbose=verbose, logger=_logger)
        return array
    except IndexError:
        vlog(f"  [WARN] Column index '{col_index}' out of bounds for "
             f"scaler of '{base_name}'. Returning original array.",
             level=2, verbose=verbose, logger=_logger)
        return array
    except Exception as e:
        vlog(f"  [WARN] Failed to inverse transform '{base_name}': {e}"
             ". Returning original array.",
             level=2, verbose=verbose, logger=_logger)
        return array

def _add_quantiles_to_df(
    predictions: Dict[str, np.ndarray],
    target_mapping: Dict[str, str],
    quantiles: List[float],
    output_dims: Dict[str, int],
    num_samples: int,
    H_inferred: int,
    verbose: int = 0,
    _logger: Any = None
) -> pd.DataFrame:
    """Formats INVERTED quantile predictions into a DataFrame."""
    df_parts = []
    for pred_key, base_name in target_mapping.items():
        if pred_key not in predictions:
            continue
        
        pred_array = predictions[pred_key] # Should be (B, H, Q, O)
        
        O_target = output_dims.get(pred_key)
        if O_target is None:
             O_target = pred_array.shape[-1]
             vlog(f"  [WARN] output_dim for '{pred_key}' not specified. "
                  f"Inferred {O_target} from array shape.",
                  level=2, verbose=verbose, logger=_logger)
        
        if pred_array.shape != (num_samples, H_inferred, len(quantiles), O_target):
            vlog(f"  [WARN] Quantile array for '{pred_key}' has unexpected shape "
                 f"{pred_array.shape}. Expected {(num_samples, H_inferred, len(quantiles), O_target)}. "
                 "Skipping.", level=2, verbose=verbose, logger=_logger)
            continue

        # Reshape: (B, H, Q, O) -> (B*H, Q*O)
        reshaped_preds = pred_array.reshape(num_samples * H_inferred, -1)
        
        # Create column names
        col_names = []
        for o_idx in range(O_target):
            for q in quantiles:
                # Format quantile string (e.g., q05, q50, q95)
                q_str = f"q{int(q*100):02d}"
                if q_str == 'q50': 
                    q_str = 'q50' # Standardize
                elif q_str == 'q10':
                    q_str = 'q10'
                elif q_str == 'q90': 
                    q_str = 'q90'
                elif q_str == 'q05':
                    q_str = 'q05'
                elif q_str == 'q95':
                    q_str = 'q95'
                
                col_name = f"{base_name}_{q_str}"
                if O_target > 1:
                    col_name += f"_{o_idx}"
                col_names.append(col_name)
        
        if reshaped_preds.shape[1] != len(col_names):
             vlog(f"  [WARN] Mismatch in quantile columns for '{base_name}'. "
                  f"Expected {len(col_names)} columns, found {reshaped_preds.shape[1]}. "
                  "Skipping.", level=2, verbose=verbose, logger=_logger)
             continue

        df_parts.append(pd.DataFrame(reshaped_preds, columns=col_names))
        vlog(f"    - Added quantile columns for '{base_name}': {col_names}",
             level=5, verbose=verbose, logger=_logger)
             
    return pd.concat(df_parts, axis=1) if df_parts else pd.DataFrame()

def _add_point_preds_to_df(
    predictions: Dict[str, np.ndarray],
    target_mapping: Dict[str, str],
    output_dims: Dict[str, int],
    num_samples: int,
    H_inferred: int,
    verbose: int = 0,
    _logger: Any = None
) -> pd.DataFrame:
    """Formats INVERTED point predictions into a DataFrame."""
    df_parts = []
    for pred_key, base_name in target_mapping.items():
        if pred_key not in predictions:
            continue
        
        pred_array = predictions[pred_key] # (B, H, O)
        O_target = output_dims.get(pred_key)
        if O_target is None:
             O_target = pred_array.shape[-1]
             vlog(f"  [WARN] output_dim for '{pred_key}' not specified. "
                  f"Inferred {O_target} from array shape.",
                  level=2, verbose=verbose, logger=_logger)
        
        if pred_array.shape != (num_samples, H_inferred, O_target):
            vlog(f"  [WARN] Point pred array for '{pred_key}' has unexpected shape "
                 f"{pred_array.shape}. Expected {(num_samples, H_inferred, O_target)}. "
                 "Skipping.", level=2, verbose=verbose, logger=_logger)
            continue
        
        reshaped_preds = pred_array.reshape(num_samples * H_inferred, O_target)
        
        col_names = []
        for o_idx in range(O_target):
            col_name = f"{base_name}_pred"
            if O_target > 1:
                col_name += f"_{o_idx}"
            col_names.append(col_name)

        df_parts.append(pd.DataFrame(reshaped_preds, columns=col_names))
        vlog(f"    - Added point pred columns for '{base_name}': {col_names}",
             level=5, verbose=verbose, logger=_logger)
    
    return pd.concat(df_parts, axis=1) if df_parts else pd.DataFrame()

def _add_actuals_to_df(
    y_true_dict: Dict[str, np.ndarray],
    target_mapping: Dict[str, str],
    output_dims: Dict[str, int],
    num_samples: int,
    H_inferred: int,
    verbose: int = 0,
    _logger: Any = None
) -> pd.DataFrame:
    """Formats INVERTED actuals into a DataFrame."""
    df_parts = []
    for pred_key, base_name in target_mapping.items():
        if pred_key not in y_true_dict:
            continue
            
        true_array = y_true_dict[pred_key] # (B, H, O)
        O_target = output_dims.get(pred_key)
        if O_target is None:
             O_target = true_array.shape[-1]
             vlog(f"  [WARN] output_dim for '{pred_key}' not specified. "
                  f"Inferred {O_target} from array shape.",
                  level=2, verbose=verbose, logger=_logger)
        
        if true_array.shape != (num_samples, H_inferred, O_target):
            vlog(f"  [WARN] Actuals array for '{pred_key}' has unexpected shape "
                 f"{true_array.shape}. Expected {(num_samples, H_inferred, O_target)}. "
                 "Skipping.", level=2, verbose=verbose, logger=_logger)
            continue

        reshaped_true = true_array.reshape(num_samples * H_inferred, O_target)
        
        col_names = []
        for o_idx in range(O_target):
            col_name = f"{base_name}_actual"
            if O_target > 1:
                col_name += f"_{o_idx}"
            col_names.append(col_name)
        
        df_parts.append(pd.DataFrame(reshaped_true, columns=col_names))
        vlog(f"    - Added actuals columns for '{base_name}': {col_names}",
             level=5, verbose=verbose, logger=_logger)
    
    return pd.concat(df_parts, axis=1) if df_parts else pd.DataFrame()

def _add_coords_to_df(
    model_inputs: Dict[str, "Tensor"],
    coord_scaler: Any,
    num_samples: int,
    H_inferred: int,
    verbose: int = 0,
    _logger: Any = None,
    force_future_from: Optional[float] = None,   
) -> Tuple[pd.DataFrame, List[str]]:
    """Formats and inverse-transforms coordinates."""
    
    coord_names = ['coord_t', 'coord_x', 'coord_y']
    if 'coords' not in model_inputs or model_inputs['coords'] is None:
        vlog("  'coords' not in model_inputs. Skipping coordinate columns.",
             level=2, verbose=verbose, logger=_logger)
        return pd.DataFrame(), []
    
    coords_arr = model_inputs['coords']
    if hasattr(coords_arr, 'numpy'):
        coords_arr = coords_arr.numpy()
        
    if coords_arr.shape != (num_samples, H_inferred, 3):
        vlog(f"  'coords' shape mismatch ({coords_arr.shape}). Expected "
             f"{(num_samples, H_inferred, 3)}. Skipping coordinate columns.",
             level=2, verbose=verbose, logger=_logger)
        return pd.DataFrame(), []

    coords_reshaped = coords_arr.reshape(num_samples * H_inferred, 3)
    
    if coord_scaler is not None:
        vlog("  Applying inverse transform to t,x,y coordinates...",
             level=4, verbose=verbose, logger=_logger)
        try:
            coords_reshaped = coord_scaler.inverse_transform(coords_reshaped)
        except Exception as e:
            vlog(f"  [WARN] Could not inverse transform coordinates: {e}. "
                 "Using normalized coordinates.",
                 level=2, verbose=verbose, logger=_logger)

    # ---  override time with forecast years, if requested ---
    if force_future_from is not None:
        # years_for_one_sample: e.g. [2023, 2024, 2025] for H_inferred=3
        years = np.arange(
            float(force_future_from),
            float(force_future_from) + float(H_inferred),
            dtype=coords_reshaped.dtype,
        )
        # Repeat for all samples and flatten to match (num_samples*H, )
        years_tiled = np.tile(years, num_samples)
        coords_reshaped[:, 0] = years_tiled
        vlog(f"  Overriding coord_t with forecast years starting at "
             f"{force_future_from} for horizon={H_inferred}.",
             level=4, verbose=verbose, logger=_logger)

    vlog(f"  Added coordinate columns: {coord_names}",
         level=4, verbose=verbose, logger=_logger)
    return pd.DataFrame(coords_reshaped, columns=coord_names), coord_names

def _add_ids_to_df(
    ids_data_array: Any,
    ids_cols: Optional[List[str]],
    ids_cols_indices: Optional[List[int]],
    num_samples: int,
    H_inferred: int,
    verbose: int = 0,
    _logger: Any = None
) -> pd.DataFrame:
    """Extracts and repeats static/ID columns."""
    if ids_data_array is None:
        vlog("  No `ids_data_array` provided, skipping static/ID columns.",
             level=4, verbose=verbose, logger=_logger)
        return pd.DataFrame()
    
    vlog("  Processing additional static/ID columns...",
         level=4, verbose=verbose, logger=_logger)
    
    ids_np_array = None
    ids_cols_to_use = []
    
    if isinstance(ids_data_array, pd.DataFrame):
        ids_cols_to_use = list(ids_cols) if ids_cols else list(ids_data_array.columns)
        try:
            ids_np_array = ids_data_array[ids_cols_to_use].values
        except KeyError as e:
            vlog(f"  [WARN] Columns not in ids_data_array: {e}. Skipping IDs.",
                 level=2, verbose=verbose, logger=_logger)
            return pd.DataFrame()
    
    elif isinstance(ids_data_array, np.ndarray):
        ids_np_array = ids_data_array
        if ids_cols_indices:
            ids_np_array = ids_np_array[:, ids_cols_indices]
        
        if ids_cols and len(ids_cols) == ids_np_array.shape[1]:
            ids_cols_to_use = list(ids_cols)
        else:
            ids_cols_to_use = [f"id_{k}" for k in range(ids_np_array.shape[1])]
    
    elif isinstance(ids_data_array, Tensor): # Is a Tensor
        ids_np_array = ids_data_array.numpy()
        if ids_cols_indices:
            ids_np_array = ids_np_array[:, ids_cols_indices]
        if ids_cols and len(ids_cols) == ids_np_array.shape[1]:
            ids_cols_to_use = list(ids_cols)
        else:
            ids_cols_to_use = [f"id_{k}" for k in range(ids_np_array.shape[1])]
    else:
        vlog(f"  [WARN] Unsupported type for `ids_data_array`: "
             f"{type(ids_data_array)}. Skipping IDs.",
             level=2, verbose=verbose, logger=_logger)
        return pd.DataFrame()

    # Now, repeat the data
    if ids_np_array is not None and ids_np_array.shape[0] == num_samples:
        # Use np.repeat to tile each sample H_inferred times
        expanded_ids_data = np.repeat(ids_np_array, H_inferred, axis=0)
        vlog(f"    - Added static/ID columns: {ids_cols_to_use}",
             level=5, verbose=verbose, logger=_logger)
        return pd.DataFrame(expanded_ids_data, columns=ids_cols_to_use)
    else:
        vlog(f"  [WARN] `ids_data_array` sample size"
             f" ({ids_np_array.shape[0] if ids_np_array is not None else 'None'}) "
             f"does not match predictions ({num_samples}). Skipping IDs.",
             level=2, verbose=verbose, logger=_logger)
        return pd.DataFrame()

def _evaluate_coverage(
    df: pd.DataFrame,
    target_mapping: Dict[str, str],
    q_indices: Tuple[int, int],
    savefile: Optional[str] = None,
    name: Optional[str] = None,
    verbose: int = 0,
    _logger: Any = None
):
    """Calculates and logs quantile coverage from an INVERTED DataFrame."""
    vlog("  Evaluating forecast coverage...",
         level=4, verbose=verbose, logger=_logger)
         
    for base_name in target_mapping.values():
        actual_col = f"{base_name}_actual"
        
        # Find quantile columns
        q_cols = sorted([c for c in df.columns if c.startswith(base_name) and "_q" in c])
        if not q_cols or actual_col not in df.columns:
            vlog(f"  Skipping coverage for '{base_name}': "
                 "Missing actuals or quantile columns.",
                 level=3, verbose=verbose, logger=_logger)
            continue
        
        try:
            q_low_col = q_cols[q_indices[0]]
            q_high_col = q_cols[q_indices[1]]
        except IndexError:
            vlog(f"  [WARN] Coverage q_indices {q_indices} are out of bounds "
                 f"for detected quantile columns: {q_cols}. Skipping coverage.",
                 level=2, verbose=verbose, logger=_logger)
            continue

        q_low = df[q_low_col]
        q_high = df[q_high_col]
        actual = df[actual_col]
        
        is_covered = (actual >= q_low) & (actual <= q_high)
        coverage_percent = is_covered.mean() * 100
        
        vlog(f"  Forecast Coverage ({base_name}, {q_low_col} to {q_high_col}): "
             f"{coverage_percent:.2f}%",
             level=1, verbose=verbose, logger=_logger)
        
        # Store in df attributes
        df.attrs[f'{base_name}_coverage'] = coverage_percent

        # Use compute_quantile_diagnostics
        try:
            cov_savefile = None
            if savefile:
                # If savefile has an extension, treat it as a file path.
                # Example: ".../subs_eval.csv"
                #   -> ".../subs_eval_diagnostics_results.json"
                root, ext = os.path.splitext(savefile)
                if ext:
                    cov_savefile = f"{root}_diagnostics_results.json"
                else:
                    # No extension: treat `savefile` as a directory-like path
                    # Example: "results/nansha_eval"
                    #   -> "results/nansha_eval/diagnostics_results.json"
                    cov_savefile = os.path.join(
                        savefile, "diagnostics_results.json"
                    )

            compute_quantile_diagnostics(
                df,
                target_name=base_name,
                quantiles=[float(c.split("_q")[-1]) / 100.0 for c in q_cols],
                coverage_quantile_indices=q_indices,
                savefile=cov_savefile,
                name=name,
                verbose=verbose,
                logger=_logger,
            )

        except Exception as e:
            vlog(f"  [WARN] Failed to compute quantile diagnostics: {e}",
                 level=2, verbose=verbose, logger=_logger)
            
def _get_model_predictions(
    pihalnet_outputs: Optional[Dict[str, Tensor]],
    model: Optional[Model],
    model_inputs: Optional[Dict[str, Tensor]],
    verbose: int,
    stop_check: Callable[[], bool],
    _logger: Any,
) -> Dict[str, Tensor]:
    """
    Obtain model outputs dict, running predict if needed.
    """
    if pihalnet_outputs is not None:
        return pihalnet_outputs
    if model is None or model_inputs is None:
        raise ValueError(
            "If 'pihalnet_outputs' is None, both 'model' and "
            "'model_inputs' must be provided."
        )
    vlog("  Predictions not provided, generating from model...",
         level=4, verbose=verbose, logger=_logger)
    try:
        outputs = model.predict(model_inputs, verbose=0)
        if not isinstance(outputs, dict):
            raise ValueError(
                "Model output is not a dict as expected from PINN models"
            )
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate predictions: {e}"
        ) from e
    vlog("  Model predictions generated.", level=5,
                 verbose=verbose, logger=_logger)
    
    if stop_check and stop_check():
        raise InterruptedError("Model prediction aborted.")
    return outputs


def _convert_outputs_to_numpy(
    pihalnet_outputs: Dict[str, Any], 
    verbose: int, _logger, 
) -> Dict[str, np.ndarray]:
    """
    Ensure prediction tensors are NumPy arrays.
    """
    proc: Dict[str, np.ndarray] = {}
    for key, val in pihalnet_outputs.items():
        if key in ('subs_pred', 'gwl_pred'):
            if hasattr(val, 'numpy'):
                proc[key] = val.numpy()
            elif isinstance(val, np.ndarray):
                proc[key] = val
            else:
                try:
                    proc[key] = np.array(val)
                except Exception as e: 
                    raise TypeError(
                        f"Could not convert output '{key}' to NumPy. "
                        f"Type: {type(val)}. Error: {e}"
                    ) from e
                    
        elif key == 'pde_residual':
            # PDE residual can be handled differently if needed
            pass # Not typically added to this output DataFrame
       
    if not proc:
        vlog("  No 'subs_pred' or 'gwl_pred'"
             " found in outputs. Returning empty DF.",
             level=1, verbose=verbose, logger=_logger)
        return pd.DataFrame()
        
    return proc


def _define_targets(
    processed: Dict[str, np.ndarray],
    include_gwl: bool,
    target_mapping: Optional[Dict[str, str]] = None,
    stop_check: Callable[[], bool]=None,
) -> Dict[str, str]:
    """
    Map processed outputs to base target names.
    """
    if target_mapping is None:
        target_mapping = {
            'subs_pred': 'subsidence',
            'gwl_pred': 'gwl'
        }
    targets: Dict[str, str] = {}
    if 'subs_pred' in processed:
        targets['subs_pred'] = target_mapping['subs_pred']
    if include_gwl and 'gwl_pred' in processed:
        targets['gwl_pred'] = target_mapping['gwl_pred']
        
    if stop_check and stop_check():
        raise InterruptedError("Target confifuration aborted.")
        
    return targets


def _infer_dimensions(
    first_pred: np.ndarray,
    forecast_horizon: Optional[int], 
    verbose : int =0, 
    _logger: Any = print, 
) -> Tuple[int, int]:
    """
    Infer num_samples and horizon (H) from the array.
    """
    N = first_pred.shape[0]
    H = forecast_horizon or first_pred.shape[1]
    if N == 0 or H == 0:
        raise ValueError("No samples or zero forecast horizon.")
        
    vlog(f"  Formatting for {N} samples,"
         f" Horizon={H}.",
         level=4, verbose=verbose, logger=_logger)

    return N, H


def _build_sample_df(
    num_samples: int,
    H: int
) -> pd.DataFrame:
    """
    Build DataFrame of sample_idx and forecast_step.
    """
    idx = np.repeat(np.arange(num_samples), H)
    steps = np.tile(np.arange(1, H+1), num_samples)
    return pd.DataFrame({
        'sample_idx': idx,
        'forecast_step': steps
    })


def _add_coordinate_columns(
    base_df: pd.DataFrame,
    model_inputs: Optional[Dict[str, Any]],
    coord_scaler: Any,
    include_coords: bool,
    stop_check: Callable[[], bool],
    _logger: Any,
    verbose: int
) -> pd.DataFrame:
    """
    Extract and inverse-transform coords if requested.
    """
    if not include_coords:
        return pd.DataFrame()
    coords_arr = None
    if model_inputs and 'coords' in model_inputs:
        coords_arr = model_inputs['coords']
        if hasattr(coords_arr, 'numpy'):
            coords_arr = coords_arr.numpy()
    if coords_arr is None:
        return pd.DataFrame()
    N = base_df['sample_idx'].nunique()
    H = base_df['forecast_step'].max()
    
    if coords_arr.shape != (N, H, 3):
        vlog("  'coords' shape mismatch or not found."
                " Skipping coordinate columns.",
                level=2, verbose=verbose, logger=_logger)
        return pd.DataFrame()
    
    resh = coords_arr.reshape(N*H, 3)
    if coord_scaler:
        vlog("  Applying inverse transform to t,x,y coordinates...",
              level=4, verbose=verbose, logger=_logger)
        try:
            resh = coord_scaler.inverse_transform(resh)
        except Exception:
            pass
        
    if stop_check and stop_check():
        raise InterruptedError("Coordinates transformation aborted.")
            
    return pd.DataFrame(resh, columns=['coord_t','coord_x','coord_y'])


def _add_id_columns(
    base_df: pd.DataFrame,
    ids_data_array: Any,
    ids_cols: Optional[List[str]],
    ids_cols_indices: Optional[List[int]],
    stop_check: Callable[[], bool],
    _logger: Any,
    verbose: int
) -> pd.DataFrame:
    """
    Process and repeat static ID columns per forecast step.

    Converts DataFrame, Series, NumPy array, or Tensor to
    NumPy, selects requested columns or indices, then uses
    `sample_idx` from base_df to expand each ID row
    across all forecast steps.
    """
    # Skip if no IDs provided
    if ids_data_array is None:
        vlog(
            "  No `ids_data_array` provided, skipping ID columns.",
            level=4, verbose=verbose, logger=_logger
        )
        return pd.DataFrame()

    vlog(
        "  Processing additional static/ID columns...",
        level=4, verbose=verbose, logger=_logger
    )

    # Convert to NumPy array
    try:
        if isinstance(ids_data_array, pd.DataFrame):
            arr = ids_data_array.values
            col_names = ids_cols or list(ids_data_array.columns)
        elif isinstance(ids_data_array, pd.Series):
            arr = ids_data_array.to_frame().values
            col_names = ids_cols or [ids_data_array.name or 'id_0']
            if arr.ndim == 2 and len(col_names) != arr.shape[1]:
                vlog(
                    "    Series IDs length mismatch, using first col.",
                    level=5, verbose=verbose, logger=_logger
                )
                col_names = col_names[:1]
        elif hasattr(ids_data_array, 'numpy'):
            arr = ids_data_array.numpy()
            col_names = ids_cols or [f'id_{i}' for i in range(arr.shape[1])]
        elif isinstance(ids_data_array, np.ndarray):
            arr = ids_data_array
            if ids_cols_indices is not None:
                arr = arr[:, ids_cols_indices]
            col_names = ids_cols or [f'id_{i}' for i in range(arr.shape[1])]
        else:
            _logger.warning(
                f"Unsupported IDs type {type(ids_data_array)}."
                " Skipping ID columns."
            )
            return pd.DataFrame()
    except Exception as e:
        _logger.warning(
            f"Error converting IDs to array: {e}."
            " Skipping ID columns."
        )
        return pd.DataFrame()

    vlog(
        f"    Converted IDs to NumPy with shape {arr.shape}",
        level=5, verbose=verbose, logger=_logger
    )
    vlog(
        f"    Selected ID column names: {col_names}",
        level=5, verbose=verbose, logger=_logger
    )

    # Expand rows according to sample_idx
    sample_indices = base_df['sample_idx'].values
    if arr.shape[0] != sample_indices.max() + 1:
        _logger.warning(
            f"IDs rows ({arr.shape[0]}) != samples "
            f"({sample_indices.max()+1}). Skipping."
        )
        return pd.DataFrame()

    expanded = arr[sample_indices]
    id_df = pd.DataFrame(expanded, columns=col_names)
    vlog(
        f"    Expanded ID DataFrame shape: {id_df.shape}",
        level=5, verbose=verbose, logger=_logger
    )

    # User cancellation check
    if stop_check and stop_check():
        raise InterruptedError("ID columns processing aborted.")

    return id_df

    # # Logic to convert ids_data_array to 2D NumPy
    # # and repeat according to base_df indices
    # # ... (Detailed implementation)
    # return pd.DataFrame()  # stub for brevity

def _process_target_variable(
    preds: np.ndarray,
    pred_key: str,
    base_name: str,
    num_samples: int,
    H: int,
    quantiles: Optional[List[float]],
    y_true_dict: Optional[Dict[str, Any]],
    scaler_info: Optional[Dict[str, Any]],
    verbose: int,
    stop_check: Callable[[], bool],
    _logger: Any
) -> pd.DataFrame:
    """
    Format predictions, attach actuals, and prepare scaling.

    - Calls _format_target_predictions for quantiles
      or point forecasts.
    - Reshapes and adds true values if provided.
    - Records inverse-scaling actions to apply later.
    """
    # 1. Format predictions
    cols_pred, df_pred = _format_target_predictions(
        preds,
        num_samples,
        H,
        O=(preds.shape[-1] if preds.ndim == 3 else 1),
        base_target_name=base_name,
        quantiles=quantiles,
        verbose=verbose
    )
    dfs: List[pd.DataFrame] = [df_pred]

    # 2. Add actual values
    if y_true_dict:
        y_true = y_true_dict.get(pred_key)
        if y_true is None:
            y_true = y_true_dict.get(base_name)
        if hasattr(y_true, 'numpy'):
            y_true = y_true.numpy()
        if y_true is not None:
            arr = y_true.reshape(num_samples * H, -1)
            cols_act = []
            for i in range(arr.shape[1]):
                col = base_name
                if arr.shape[1] > 1:
                    col += f"_{i}"
                cols_act.append(f"{col}_actual")
            dfs.append(pd.DataFrame(arr, columns=cols_act))

    # 3. Prepare inverse-scaling info
    if scaler_info and base_name in scaler_info:
        info = scaler_info[base_name]
        to_transform = cols_pred + cols_act
        info['_cols_to_inv'] = to_transform

    if stop_check and stop_check():
        raise InterruptedError(
            "Target variable processing aborted."
        )

    return pd.concat(dfs, axis=1)

def _apply_masking(
    df: pd.DataFrame,
    targets: Dict[str, str],
    apply_mask: bool,
    mask_values: Any,
    mask_fill_value: float,
    quantiles: Optional[List[float]]
) -> pd.DataFrame:
    """
    Mask forecasts based on reference column values.
    """
    if not apply_mask:
        return df
    first_base = list(targets.values())[0]
    ref = f"{first_base}_actual"
    if ref not in df:
        raise KeyError(f"Reference col '{ref}' missing")
    cols = []
    if quantiles:
        for base in targets.values():
            for q in quantiles:
                cols.append(f"{base}_q{int(q*100)}")
    else:
        cols = [c for c in df if c.endswith("_pred")]
    return mask_by_reference(
        data=df,
        ref_col=ref,
        values=mask_values,
        find_closest=False,
        fill_value=mask_fill_value,
        mask_columns=cols,
        error="ignore",
        inplace=False
    )


def _compute_coverage(
    df: pd.DataFrame,
    targets: Dict[str, str],
    quantiles: List[float],
    y_true_dict: Dict[str, Any],
    coverage_quantile_indices: Tuple[int, int],
    savefile: Optional[str],
    name: Optional[str],
    verbose: int,
    _logger: Any
) -> None:
    """
    Compute quantile diagnostics per target.
    """
    for base in targets.values():
        try:
            compute_quantile_diagnostics(
                df,
                target_name=base,
                quantiles=quantiles,
                coverage_quantile_indices=
                    coverage_quantile_indices,
                savefile=(os.path.join(
                    os.path.dirname(savefile),
                    'diagnostics_results.json'
                ) if savefile else None),
                name=name,
                verbose=verbose,
                logger=_logger
            )
        except Exception as e:
            vlog(f"Skipping coverage due to {e}",
                 level=2, verbose=verbose,
                 logger=_logger)

#XXX : TODO 
# revise format outputs and apply
#  mask by reference 
def format_preds(
    pihalnet_outputs: Optional[Dict[str, Tensor]] = None,
    model: Optional[Model] = None,
    model_inputs: Optional[Dict[str, Tensor]] = None,
    y_true_dict: Optional[Dict[str, Any]] = None,
    target_mapping: Optional[Dict[str, str]] = None,
    include_gwl: bool = True,
    include_coords: bool = True,
    quantiles: Optional[List[float]] = None,
    forecast_horizon: Optional[int] = None,
    output_dims: Optional[Dict[str, int]] = None,
    ids_data_array: Optional[Any] = None,
    ids_cols: Optional[List[str]] = None,
    ids_cols_indices: Optional[List[int]] = None,
    scaler_info: Optional[Dict[str, Any]] = None,
    coord_scaler: Optional[Any] = None,
    evaluate_coverage: bool = False,
    coverage_quantile_indices: Tuple[int, int] = (0, -1),
    savefile: Optional[str] = None,
    name: Optional[str] = None,
    apply_mask: bool = False,
    mask_values: Optional[Any] = None,
    mask_fill_value: Optional[float] = None,
    verbose: int = 0,
    _logger: Any = None,
    stop_check: Callable[[], bool] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Main function orchestrating all helper steps.
    """
    vlog("Starting formatting", level=3,
         verbose=verbose, logger=_logger)
    # Step 1: obtain raw outputs
    raw_out = _get_model_predictions(
        pihalnet_outputs, model, model_inputs,
        verbose, stop_check, _logger
    )
    # Step 2: convert to NumPy
    proc = _convert_outputs_to_numpy(
        raw_out, verbose, _logger)
    
    # Step 3: define targets
    targets = _define_targets(
        proc, include_gwl, target_mapping, 
        stop_check,  
    )
    if not targets:
        vlog("  No valid targets to process after"
             " filtering. Returning empty DF.",
             level=1, verbose=verbose, logger=_logger)
        return pd.DataFrame()
    
    # Step 4: infer dims
    num_samp, H = _infer_dimensions(
        proc[next(iter(targets))], forecast_horizon, 
        verbose , _logger, 
    )
    # Step 5: build base df
    base_df = _build_sample_df(num_samp, H)
    # Step 6: add coords
    coords_df = _add_coordinate_columns(
        base_df, model_inputs, coord_scaler,
        include_coords, stop_check, _logger, verbose
    )
    # Step 7: add IDs
    ids_df = _add_id_columns(
        base_df, ids_data_array, ids_cols,
        ids_cols_indices, stop_check, _logger,
        verbose
    )
    # Step 8: process targets
    dfs = []
    for key, base in targets.items():
        dfs.append(_process_target_variable(
            proc[key], key, base,
            num_samp, H, quantiles,
            y_true_dict, scaler_info,
            verbose, stop_check, _logger
        ))
    # Step 9: concat all
    final_df = pd.concat([base_df, coords_df,
                          ids_df] + dfs, axis=1)
    # Step 10: masking
    final_df = _apply_masking(
        final_df, targets, apply_mask,
        mask_values, mask_fill_value, quantiles
    )
    # Step 11: coverage
    if evaluate_coverage and quantiles:
        _compute_coverage(
            final_df, targets, quantiles,
            y_true_dict, coverage_quantile_indices,
            savefile, name, verbose, _logger
        )
    vlog("Formatting complete", level=3,
         verbose=verbose, logger=_logger)
    return final_df


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
    h_field_col: Optional[str]=None,  
    lon_col: Optional[str]=None,
    lat_col: Optional[str]=None,
    group_id_cols: Optional[List[str]] = None,
    time_steps: int = 12,
    forecast_horizon: int = 3,
    output_subsidence_dim: int = 1, 
    output_gwl_dim: int = 1,       
    datetime_format: Optional[str] = None,
    normalize_coords: bool = True, 
    cols_to_scale: Union[List[str], str, None] = None,
    return_coord_scaler: bool =False, 
    coord_scaler: Optional[MinMaxScaler] = None,
    fit_coord_scaler: bool = True,
    mode: Optional[str] =None, 
    model: Optional[str] =None,
    savefile: Optional[str] = None,
    progress_hook: Optional[Callable[[float], None]] = None,
    stop_check: Callable[[], bool] = None, 
    verbose: int = 0,
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    **kws
) -> Union[
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[MinMaxScaler]]
]:
    # -------------------------------------------------------------------------
    _to_range = lambda f, lo, hi: lo + (hi - lo) * f      # noqa: E731
    
    vlog("Starting PINN data sequence preparation...",
         verbose=verbose, level=1, logger=_logger)

    df_proc = df.copy()

    # --- 1. Validate Input Parameters and Columns ---
    if verbose >= 2:
        logger.debug("Validating input parameters and columns.")

    vlog("Validating essential columns...",
         verbose=verbose, level=2, logger=_logger)
    
    lon_col, lat_col = resolve_spatial_columns(
        df_proc, spatial_cols =spatial_cols, 
        lon_col=lon_col, 
        lat_col= lat_col
    )
    essential_cols = [time_col, lon_col, lat_col, subsidence_col, gwl_col]
    
    # --- MODIFICATION 1: Conditional Validation ---
    is_geoprior = str(model).lower().strip() in ('geoprior', 'geopriorsubsnet')
    
    if is_geoprior:
        if h_field_col is None:
            # Check for default names
            if 'H_field' in df_proc.columns:
                h_field_col = 'H_field'
            elif 'soil_thickness' in df_proc.columns:
                h_field_col = 'soil_thickness'
            else:
                raise ValueError(
                    "`model` is 'geoprior' but `h_field_col` was not "
                    "provided and default names 'H_field' or 'soil_thickness' "
                    "were not found in the DataFrame."
                )
        
        vlog(f"GeoPrior model detected. Using '{h_field_col}' as H_field.",
             verbose=verbose, level=3, logger=_logger)
        
        essential_cols.append(h_field_col)
        vlog(f"Validated '{h_field_col}' for GeoPriorSubsNet.",
              verbose=verbose, level=3, logger=_logger)
        
    # --- End Modification 1 ---

        
    exist_features(
        df_proc, features=essential_cols, 
        message="Essential column(s) missing."
    )
    
    vlog("Validating time-series dataset...",
         verbose=verbose, level=2, logger=_logger)
    
    check_datetime(
        df_proc,
        dt_cols= time_col, 
        ops="check_only",
        consider_dt_as="numeric",
        accept_dt=True, 
        allow_int=True, 
    )
    
    vlog("Managing feature column lists...",
         verbose=verbose, level=3, logger=_logger)
    
    dynamic_cols = columns_manager(
        dynamic_cols, empty_as_none=False
    )
    exist_features(
        df_proc, features=dynamic_cols, 
        name="Dynamic feature column(s)"
    )
    
    static_cols = columns_manager(static_cols) 
    if static_cols:
        exist_features(
            df_proc, features=static_cols,
            name="Static feature column(s)")
    
    future_cols = columns_manager(future_cols) 
    if future_cols:
        exist_features(
            df_proc, features=future_cols, 
            name="Future feature column(s)")

    group_id_cols = columns_manager(group_id_cols) 
    if group_id_cols:
        exist_features(
            df_proc, features=group_id_cols, 
            name="Group ID column(s)"
            )
        

    vlog("Validating time_steps and forecast_horizon...",
         verbose=verbose, level=3, logger=_logger)
    
    time_steps = validate_positive_integer(
        time_steps, "time_steps")
    forecast_horizon = validate_positive_integer(
        forecast_horizon, "forecast_horizon", )
    

    vlog(
        "Pre-flight: assessing sliding-window feasibility ...",
        verbose=verbose,
        level=1,
        logger=_logger,
    )
    
    ok, _ = check_sequence_feasibility(
        df_proc.copy(),
        time_col=time_col,
        group_id_cols=group_id_cols,
        time_steps=time_steps,
        forecast_horizon=forecast_horizon,
        verbose=verbose,
        error="raise",  # fail fast on impossibility,
        logger= _logger, 
    )
    
    vlog(
        "Feasibility check passed — generating sequences...",
        verbose=verbose,
        level=1,
        logger=_logger,
    )
    
    vlog("Starting PINN data sequence generation...",
         verbose=verbose, level=1, logger=_logger)

    vlog("Converting time column to numeric values...",
         verbose=verbose, level=4, logger=_logger)
    if pd.api.types.is_numeric_dtype(df_proc[time_col]):
        numerical_time_col = time_col
        if verbose >= 2:
            logger.debug(
                f"Time column '{time_col}' is already numeric. Using it directly."
            )
    else:
        try:
            df_proc[time_col] = pd.to_datetime(
                df_proc[time_col], format=datetime_format)
            df_proc[f"{time_col}_numeric"] = (
                df_proc[time_col].dt.year +
                (df_proc[time_col].dt.dayofyear - 1) /
                (365 + df_proc[time_col].dt.is_leap_year.astype(int))
            )
            numerical_time_col = f"{time_col}_numeric"
            if verbose >= 2:
                logger.debug(
                    f"Converted datetime column '{time_col}'"
                    f" to numerical '{numerical_time_col}'."
                )
            vlog(f"Time column converted to '{numerical_time_col}'",
                 verbose=verbose, level=5, logger=_logger)
        except Exception as e:
            raise ValueError(
                f"Failed to convert or process time column '{time_col}'. "
                f"Ensure it's datetime-like or specify `datetime_format`."
                f" Error: {e}"
            )
   
    mode = select_mode(mode, default='pihal_like')
    vlog(f"Operating in '{mode}' data preparation mode.",
         level=1, verbose=verbose, logger=_logger)

    df_proc, coord_scaler, cols_scaler = normalize_for_pinn(
        df=df_proc,
        time_col=numerical_time_col,
        coord_x=lon_col,
        coord_y=lat_col,
        scale_coords=normalize_coords,
        cols_to_scale=cols_to_scale,
        forecast_horizon=forecast_horizon,
        verbose=verbose,
        _logger=_logger,
        coord_scaler=coord_scaler,
        fit_coord_scaler=fit_coord_scaler,
    )

    # --- 2. Group and Sort Data ---
    vlog("Grouping and sorting data...",
        verbose=verbose, level=2, logger=_logger)
    
    if verbose >= 2:
        logger.debug(
            f"Grouping and sorting data. Group IDs:"
            f" {group_id_cols or 'None (single group)'}."
        )
    
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
        grouped_data = [(None, df_proc)] 
        group_keys = [None]
        if verbose >= 1:
            logger.info("Processing entire DataFrame as a single group.")

     
    # --- 3. First Pass: Calculate Total Number of Sequences ---
    total_sequences = 0
    min_len_per_group = time_steps + forecast_horizon
    valid_group_dfs = [] 

    vlog("Counting valid sequences in groups...",
        verbose=verbose, level=4, logger=_logger)
     
    if verbose >= 2:
        logger.debug(
            "First pass: Calculating total sequences. Min"
            f" length per group: {min_len_per_group}."
        )

    n_groups = len(group_keys) or 1
    for g_idx, group_key in enumerate(group_keys):
        group_df = grouped_data.get_group(group_key) if group_id_cols else df_proc
        
        if stop_check and stop_check():
            raise InterruptedError("Sequence generation aborted.")
        
        key_str = group_key if group_key is not None else "<Full Dataset>"
        if progress_hook is not None:
            progress_hook(_to_range((g_idx + 1) / n_groups, 0.0, 0.50))
                          
        if len(group_df) < min_len_per_group:
            if verbose >= 5: 
                logger.info(
                    f"Group '{key_str}' has {len(group_df)} points, less than "
                    f"min required ({min_len_per_group}). Skipping."
                )
            vlog(f"Group {key_str} too small:"
                 f" {len(group_df)} < {min_len_per_group}",
                 verbose=verbose, level=6, logger=_logger)
            
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
             level=6, logger=_logger
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
        verbose=verbose, level=1, logger=_logger)
     
    # --- 4. Pre-allocate NumPy Arrays ---
    vlog("Pre-allocating arrays...",
         verbose=verbose, level=2, logger=_logger)
    
    num_dynamic_feats = len(dynamic_cols)
    num_static_feats = len(static_cols) if static_cols else 0
    num_future_feats = len(future_cols) if future_cols else 0

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
    
    if mode == 'tft_like':
        future_window_len = time_steps + forecast_horizon
        vlog(f"Allocating future features array for 'tft_like' mode "
             f"with time dimension: {future_window_len}",
             level=2, verbose=verbose, logger=_logger
            )
    else: 
        future_window_len = forecast_horizon
    
    future_features_arr = np.zeros(
        (total_sequences, future_window_len, num_future_feats),
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
    
    # --- MODIFICATION 2: Conditional Allocation ---
    H_field_arr = None # Initialize as None
    if is_geoprior:
        H_field_arr = np.zeros(
            (total_sequences, forecast_horizon, 1), dtype=np.float32
        )
        vlog(f"Allocated 'H_field' array with shape {H_field_arr.shape}",
             verbose=verbose, level=4, logger=_logger)
    # --- End Modification 2 ---
        
    vlog("Arrays shapes set.",
         verbose=verbose, level=5, logger=_logger)
    
    if verbose >= 2:
        logger.debug("Pre-allocated NumPy arrays for sequence data:")
        logger.debug(f"  Coords Horizon: {coords_horizon_arr.shape}")
        logger.debug(f"  Static Features: {static_features_arr.shape}")
        logger.debug(f"  Dynamic Features: {dynamic_features_arr.shape}")
        logger.debug(f"  Future Features: {future_features_arr.shape}")
        logger.debug(f"  Target Subsidence: {target_subsidence_arr.shape}")
        logger.debug(f"  Target GWL: {target_gwl_arr.shape}")
        # --- MODIFICATION 2b: Conditional Logging ---
        if H_field_arr is not None:
            logger.debug(f"  H_field ({h_field_col}): {H_field_arr.shape}")
            
        # --- End Modification 2b ---

    # --- 5. Second Pass: Populate Arrays with Rolling Windows ---
    current_seq_idx = 0
    if verbose >= 2:
        logger.debug("Second pass: Populating sequence arrays...")
    
    vlog("Populating arrays with data...",
        verbose=verbose, level=2, logger=_logger)
    
    total_seq = sum(len(gdf) - min_len_per_group + 1
                for gdf in valid_group_dfs) or 1
    done_seq  = 0   
    
    for group_df in valid_group_dfs: 
        group_t_coords = group_df[numerical_time_col].values
        group_x_coords = group_df[lon_col].values
        group_y_coords = group_df[lat_col].values

        t_min_group, t_max_group = group_t_coords.min(), group_t_coords.max() 

        if verbose >= 3:
            print()
            time_scale_info = f"{t_min_group:.4f}-{t_max_group:.4f}"
            if normalize_coords and coord_scaler: 
                time_scale_info += " (normalized)"
            else:
                time_scale_info += " (original scale)"
            
            print_box(
                f"Group window t: {time_scale_info}",
                width=_TW,
                align='center',
                border_char='+',
                horizontal_char='-',
                vertical_char='|',
                padding=1
            )

        group_static_vals = None
        if static_cols and num_static_feats > 0:
            group_static_vals = group_df.iloc[0][
                static_cols].values.astype(np.float32)
            
        # --- MODIFICATION 3: Get static H_val for the group ---
        group_H_val = None
        if is_geoprior:
            # Get the single static soil thickness value for this group
            group_H_val = group_df.iloc[0][h_field_col]
        # --- End Modification 3 ---

        num_seq_in_this_group = len(group_df) - min_len_per_group + 1
        for i in range(num_seq_in_this_group):
            
            done_seq += 1
            if progress_hook is not None:
                progress_hook(_to_range(done_seq / total_seq, 0.5, 1.00))
                
            if stop_check and stop_check():
                raise InterruptedError("Sequence generation aborted by user")
                
            if static_cols and num_static_feats > 0:
                static_features_arr[current_seq_idx] = group_static_vals

            dynamic_start_idx = i
            dynamic_end_idx = i + time_steps
            dynamic_features_arr[current_seq_idx] = group_df.iloc[
                dynamic_start_idx:dynamic_end_idx
            ][dynamic_cols].values.astype(np.float32)

            horizon_start_idx = i + time_steps
            horizon_end_idx = i + time_steps + forecast_horizon
 
            if future_cols and num_future_feats > 0:
                if mode == 'tft_like':
                    future_start_idx = i
                    future_end_idx = i + time_steps + forecast_horizon
                else: # 'pihal_like' mode
                    future_start_idx = horizon_start_idx
                    future_end_idx = horizon_end_idx
             
                future_features_arr[current_seq_idx] = group_df.iloc[
                    future_start_idx:future_end_idx
                ][future_cols].values.astype(np.float32)

            t_horizon = group_t_coords[horizon_start_idx:horizon_end_idx]
            x_horizon = group_x_coords[horizon_start_idx:horizon_end_idx]
            y_horizon = group_y_coords[horizon_start_idx:horizon_end_idx]
            
            coords_horizon_arr[current_seq_idx, :, 0] = t_horizon
            coords_horizon_arr[current_seq_idx, :, 1] = x_horizon
            coords_horizon_arr[current_seq_idx, :, 2] = y_horizon

            target_subsidence_arr[current_seq_idx] = group_df.iloc[
                horizon_start_idx:horizon_end_idx
            ][subsidence_col].values.reshape(
                forecast_horizon, output_subsidence_dim).astype(np.float32)
            
            target_gwl_arr[current_seq_idx] = group_df.iloc[
                horizon_start_idx:horizon_end_idx
            ][gwl_col].values.reshape(
                forecast_horizon, output_gwl_dim).astype(np.float32)

            # --- MODIFICATION 4: Populate H_field array ---
            if H_field_arr is not None:
                # Tile the static H_val across the forecast horizon
                H_field_arr[current_seq_idx, :, 0] = group_H_val
            # --- End Modification 4 ---
                
            if verbose >= 7: 
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
                verbose=verbose, level=7, logger=_logger
            )
            
            current_seq_idx += 1

        
    if verbose >= 1:
        logger.info("Successfully populated sequence arrays.")
    
    vlog("Data population complete.",
         verbose=verbose, level=1, logger=_logger)
    
    inputs_dict = {
        'coords': coords_horizon_arr, 
        'static_features': static_features_arr if num_static_feats > 0 else None,
        'dynamic_features': dynamic_features_arr,
        'future_features': future_features_arr if num_future_feats > 0 else None,
    }
    
    # --- MODIFICATION 5: Conditionally add H_field to inputs_dict ---
    if H_field_arr is not None:
        inputs_dict['H_field'] = H_field_arr
    # --- End Modification 5 ---
        
    inputs_dict = {k: v for k, v in inputs_dict.items() if v is not None}

    targets_dict = {
        'subsidence': target_subsidence_arr,
        'gwl': target_gwl_arr
    }

    if savefile:
        vlog(f"\nPreparing to save sequence data to '{savefile}'...", 
             verbose=verbose, level=3, logger=_logger
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
            'cols_scaler': cols_scaler, 
            'coord_scaler': coord_scaler, 
            'normalize_coords_flag': normalize_coords, 
            'saved_coord_scaler_flag':coord_scaler is not None, 
            # --- MODIFICATION 6: Conditionally add to savefile ---
            'model_type': model,
            'h_field_col': h_field_col,
            'H_field': H_field_arr if H_field_arr is not None else None,
            # --- End Modification 6 ---
        }
        try:
            job_dict.update(get_versions())
        except NameError: 
            vlog("\n  `get_versions` not found, version info not saved.", 
                 verbose=verbose, level =1, logger=_logger)

        try:
            save_job(job_dict, savefile, append_versions=False)
      
            if verbose >= 1:
                vlog(f"Sequence data dictionary successfully "
                      f"saved to '{savefile}'.", 
                      verbose=verbose, level=1, logger=_logger
            )
        except Exception as e:
            vlog(f"Failed to save job dictionary to "
                  f"'{savefile}': {e}", verbose=verbose, level=1, 
                  logger=_logger)
                
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
         verbose=verbose, level=3, logger=_logger)
    
    if return_coord_scaler: 
        return inputs_dict, targets_dict, coord_scaler 
    
    return inputs_dict, targets_dict


def process_pde_modes(
    pde_mode: Union[str, list, None], 
    enforce_consolidation: bool = False,
    pde_mode_config: Union[str, list, None] = None, 
    solo_return: bool=False, 
) -> list:
    r"""
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
        pde_modes_active = [str(p_type).lower() for p_type in pde_mode]
    elif pde_mode is None:
        pde_modes_active = ['none']  # Explicitly 'none' if None is provided
    else:
        raise TypeError("`pde_mode` must be a string, list of strings, or None.")

    # Handle special cases for "none" and "both"
    if "none" in pde_modes_active or "off" in pde_modes_active:  # If 'none' is present, override others
        pde_modes_active = ['none']
    if "both" in pde_modes_active or "on" in pde_modes_active:  # If 'both' is present, use both modes
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
            "Model will use 'consolidation' mode."
        )
        # pde_modes_active = ['consolidation']

    if solo_return: 
        pde_modes_active = pde_modes_active[0]
        
    # Return the processed pde_modes_active
    return pde_modes_active

def check_and_rename_keys(inputs, y): # ranem to check_input_keys 
    r"""
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

def _check_required_input_keys(
        inputs, y=None, message=None, 
        
        ):
    r"""
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



def check_required_input_keys(
    inputs: Optional[Dict[str, Any]],
    y: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
    model_name: Optional[str] = None,
    do_rename: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Validate presence of required keys in `inputs` and `y`.
    Optionally canonicalize keys via reverse alias mapping.
    
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

    # ---- inputs checks -------------------------------------------------
    if inputs is not None:
        if not isinstance(inputs, dict):
            msg = message or (
                "Inputs must be a dict with 'coords' and "
                "'dynamic_features'. Got "
                f"{type(inputs).__name__!r}."
            )
            raise TypeError(msg)

        # Canonicalize GeoPrior-specific inputs (H field aliases)
        if do_rename and model_name:
            name = str(model_name).lower()
            if name in ("geoprior", "geopriorsubsnet",
                        "geopriorsubsnet"):
                inputs = rename_dict_keys(
                    inputs,
                    {
                        "H_field": (
                            "H_field",
                            "soil_thickness",
                            "soil thickness",
                            "h_field",
                        )
                    },
                    order="reverse",
                )

        # Mandatory base inputs
        if "coords" not in inputs or inputs.get("coords") is None:
            raise ValueError("Input 'coords' is missing or None.")
        if ("dynamic_features" not in inputs or
                inputs.get("dynamic_features") is None):
            raise ValueError(
                "Input 'dynamic_features' is missing or None."
            )

        # GeoPrior requires H_field (after alias unification)
        if model_name:
            name = str(model_name).lower()
            if name in ("geoprior", "geopriorsubsnet",
                        "geopriorsubsnet"):
                if "H_field" not in inputs or inputs.get("H_field") is None:
                    raise ValueError(
                        "GeoPrior requires 'H_field' in inputs "
                        "(aliases accepted: 'soil_thickness', "
                        "'soil thickness', 'h_field')."
                    )

    # ---- target checks -------------------------------------------------
    if y is not None:
        if not isinstance(y, dict):
            msg = message or (
                "Target `y` must be a dict containing "
                "'subs_pred/subsidence' and 'gwl_pred/gwl'. Got "
                f"{type(y).__name__!r}."
            )
            raise TypeError(msg)

        # Canonicalize targets to subs_pred / gwl_pred
        if do_rename:
            y = rename_dict_keys(
                y,
                {
                    "subs_pred": ("subs_pred", "subsidence"),
                    "gwl_pred": ("gwl_pred", "gwl"),
                },
                order="reverse",
            )

        # Accept either canonical or legacy if not renaming
        has_subs = ("subs_pred" in y) or ("subsidence" in y)
        has_gwl = ("gwl_pred" in y) or ("gwl" in y)

        if not has_subs:
            raise ValueError(
                "Target missing subsidence. Provide 'subs_pred' or "
                "'subsidence'."
            )
        if not has_gwl:
            raise ValueError(
                "Target missing gwl. Provide 'gwl_pred' or 'gwl'."
            )

    return inputs, y
       
def _extract_txy_in(
    inputs: Union[Tensor, np.ndarray, Dict[str, Union[Tensor, np.ndarray]]],
    coord_slice_map: Optional[Dict[str, int]] = None, 
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Extracts t, x, y tensors from `inputs`, which may be:
      - A single 3D tensor of shape (batch, time_steps, 3)
      - A dict containing a key 'coords' with such a tensor
      - A dict containing separate keys 't', 'x', and 'y'
    
    Parameters
    ----------
    inputs : tf.Tensor or np.ndarray or dict
        If tensor/array: expected shape is (batch, time_steps, 3).
        If dict:
          - If 'coords' in dict: dict['coords'] must be (batch, time_steps, 3).
          - Otherwise, dict must have keys 't', 'x', 'y' each of shape
            (batch, time_steps, 1) or (batch, time_steps).
    coord_slice_map : dict, optional
        Mapping from 't', 'x', 'y' to their index in the last dimension of
        the coords tensor. Defaults to {'t': 0, 'x': 1, 'y': 2}.
    
    Returns
    -------
    t : tf.Tensor
        Tensor of shape (batch, time_steps, 1) corresponding to time coordinate.
    x : tf.Tensor
        Tensor of shape (batch, time_steps, 1) corresponding to x coordinate.
    y : tf.Tensor
        Tensor of shape (batch, time_steps, 1) corresponding to y coordinate.
    
    Raises
    ------
    ValueError
        If `inputs` is not in one of the supported formats, or dimensions
        are inconsistent.
    """
    
    # Default slice map
    if coord_slice_map is None:
        coord_slice_map = {'t': 0, 'x': 1, 'y': 2}

    # Helper to ensure output is a tf.Tensor with a final singleton dim
    def _ensure_tensor_with_last_dim(inp: Union[Tensor, np.ndarray]) -> Tensor:
        if isinstance(inp, np.ndarray):
            inp = tf_convert_to_tensor(inp)
        if not isinstance(inp, Tensor):
            raise ValueError(f"Expected tf.Tensor or np.ndarray, got {type(inp)}")
        # If shape is (batch, time_steps), add last dimension
        if inp.ndim == 2:
            inp = tf_expand_dims(inp, axis=-1)
        # Now expect (batch, time_steps, 1)
        if inp.ndim != 3 or inp.shape[-1] != 1:
            raise ValueError(
                "Coordinate array must have shape "
                f"(batch, time_steps, 1); got {inp.shape}")
        return inp

    # Case 1: inputs is a dict
    if isinstance(inputs, dict):
        # If 'coords' key is present
        if 'coords' in inputs:
            coords_tensor = inputs['coords']
            if isinstance(coords_tensor, np.ndarray):
                coords_tensor = tf_convert_to_tensor(coords_tensor)
            if not isinstance(coords_tensor, Tensor):
                raise ValueError(f"Expected tensor/array for 'coords';"
                                 f" got {type(coords_tensor)}")
            # Expect shape (batch, time_steps, 3)
            if coords_tensor.ndim != 3 or coords_tensor.shape[-1] < 3:
                raise ValueError(
                    f"'coords' must have shape (batch, time_steps, ≥3);"
                    f" got {coords_tensor.shape}"
                )
            # Slice out t, x, y
            t = coords_tensor[..., coord_slice_map['t']:coord_slice_map['t'] + 1]
            x = coords_tensor[..., coord_slice_map['x']:coord_slice_map['x'] + 1]
            y = coords_tensor[..., coord_slice_map['y']:coord_slice_map['y'] + 1]
            return tf_cast(
                t, tf_float32), tf_cast(x, tf_float32), tf_cast(y, tf_float32)

        # If keys 't','x','y' exist separately
        if all(k in inputs for k in ('t', 'x', 'y')):
            t = _ensure_tensor_with_last_dim(inputs['t'])
            x = _ensure_tensor_with_last_dim(inputs['x'])
            y = _ensure_tensor_with_last_dim(inputs['y'])
            return ( 
                tf_cast(t, tf_float32), tf_cast(x, tf_float32),
                tf_cast(y, tf_float32)
                )
        
        raise ValueError(
            "Dict `inputs` must contain either key 'coords' or keys 't', 'x', 'y'."
        )

    # Case 2: inputs is a single tensor/array
    if isinstance(inputs, (Tensor, np.ndarray)):
        coords_tensor = inputs
        if isinstance(coords_tensor, np.ndarray):
            coords_tensor = tf_convert_to_tensor(coords_tensor)
        # Expect shape (batch, time_steps, 3)
        if coords_tensor.ndim != 3 or coords_tensor.shape[-1] < 3:
            raise ValueError(
                f"Tensor `inputs` must have shape (batch, time_steps, 3);"
                f" got {coords_tensor.shape}"
            )
        t = coords_tensor[..., coord_slice_map['t']:coord_slice_map['t'] + 1]
        x = coords_tensor[..., coord_slice_map['x']:coord_slice_map['x'] + 1]
        y = coords_tensor[..., coord_slice_map['y']:coord_slice_map['y'] + 1]
        return ( 
            tf_cast(t, tf_float32), tf_cast(x, tf_float32),
            tf_cast(y, tf_float32)
            )

    raise ValueError(
        f"`inputs` must be a tensor/array or dict; got {type(inputs)}"
    )

def _get_coords(
    inputs: Union[Dict[str, Any], Sequence[Any], Tensor],
    *,
    check_shape: bool = False,
    is_time_dependent: bool = True,
) -> Tensor:
    r"""
    Extract the **coords** tensor from any input layout.

    Parameters
    ----------
    inputs : dict, sequence or tensor
        * **Dict** – must contain a ``"coords"`` key.  
        * **Sequence** – coords are expected at index 0.  
        * **Tensor** – interpreted as coords directly.
    check_shape : bool, default ``False``
        If ``True``, validate that the last dimension equals 3 and that
        the rank matches *is_time_dependent*.
    is_time_dependent : bool, default ``True``
        * ``True``  → expect a 3-D shape ``(B, T, 3)``  
        * ``False`` → allow a 2-D shape ``(B, 3)``; a 3-D shape is also
          accepted (useful when the model ignores time).

    Returns
    -------
    tf.Tensor
        The extracted coords tensor.

    Raises
    ------
    KeyError
        If a dict is missing the ``"coords"`` key.
    TypeError
        If *inputs* is not dict, sequence or tensor.
    ValueError
        If ``check_shape`` is ``True`` and the tensor shape is invalid.

    Notes
    -----
    The function is lightweight and executes outside any
    ``tf.function`` context; use it freely inside the model’s
    ``train_step`` or data-pipe helpers.
    """
    # ── 1. locate the tensor 
    if isinstance(inputs, Tensor):
        coords = inputs
    elif isinstance(inputs, (tuple, list)):
        coords = inputs[0]
    elif isinstance(inputs, dict):
        try:
            coords = inputs["coords"]
        except KeyError as err:
            raise KeyError("Input dict lacks a 'coords' entry.") from err
    else:
        raise TypeError(
            "inputs must be a dict, sequence, or Tensor; "
            f"got {type(inputs).__name__}"
        )

    # ── 2. validate shape if requested 
    if check_shape:
        shape = coords.shape           # static if known, else dynamic
        # last dim must be 3
        if shape.rank is not None:
            if shape[-1] != 3:
                raise ValueError(
                    "coords[..., -1] must equal 3 (t, x, y); "
                    f"found {shape[-1]}"
                )
            # rank check when static
            if is_time_dependent and shape.rank != 3:
                raise ValueError(
                    "Expected coords rank 3 (B, T, 3) for time-dependent "
                    "data; received rank "
                    f"{shape.rank}"
                )
            if not is_time_dependent and shape.rank not in (2, 3):
                raise ValueError(
                    "Expected coords rank 2 or 3 for static data; "
                    f"received rank {shape.rank}"
                )
        else:  # dynamic shapes (inside tf.function)
            tf_debugging.assert_equal(
                tf_shape(coords)[-1],
                3,
                message="coords[..., -1] must equal 3 (t, x, y)",
            )
            if is_time_dependent:
                tf_debugging.assert_rank(coords, 3)
            else:
                tf_debugging.assert_rank_in(coords, (2, 3))

    return coords

@ensure_pkg(
    KERAS_BACKEND or "tensorflow",
    extra="TensorFlow is required for this function."
   )
def extract_txy_in(
    inputs: Union[Tensor, np.ndarray, Dict[str, Union[Tensor, np.ndarray]]],
    coord_slice_map: Optional[Dict[str, int]] = None,
    expect_dim: Optional[str] = None,
    verbose: int = 0,
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    **kws
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Extracts t, x, y tensors from various input formats.

    This utility standardizes coordinate inputs, accepting a single
    tensor or a dictionary, and handling both 2D (spatial/static)
    and 3D (spatio-temporal) data. It ensures a consistent 3D
    output format for robust downstream processing.

    Parameters
    ----------
    inputs : tf.Tensor, np.ndarray, or dict
        The input data containing coordinates.
        - If a single tensor/array: Can be 2D `(batch, 3)` or 3D
          `(batch, time_steps, 3)`.
        - If a dict: Can contain a 'coords' key with the tensor, or
          separate 't', 'x', 'y' keys.

    coord_slice_map : dict, optional
        Mapping for 't', 'x', 'y' to their index in the last
        dimension of a coordinate tensor.
        Defaults to `{'t': 0, 'x': 1, 'y': 2}`.

    expect_dim : {'2d', '3d'}, optional
        If provided, enforces that the input resolves to the
        specified dimension.
        - '2d': Input must be `(batch, 3)` or dict of `(batch, 1)`.
        - '3d': Input must be `(batch, time, 3)` or dict of `(batch, time, 1)`.
        If None (default), both are accepted and 2D is expanded to 3D.

    verbose : int, default 0
        Controls the verbosity of logging messages. `0` is silent,
        `1` provides basic info, and higher values provide more detail.

    Returns
    -------
    t, x, y : Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        The extracted t, x, and y coordinate tensors, each reshaped
        to be 3D with a singleton last dimension, e.g.,
        `(batch, time_steps, 1)`.

    Raises
    ------
    ValueError
        If input format is unsupported, dimensions are inconsistent,
        or `expect_dim` constraint is violated.
    """
    vlog("Extracting (t, x, y) coordinates from inputs...",
         level=2, verbose=verbose, logger=_logger)
         
    if expect_dim and expect_dim not in ['2d', '3d']:
        raise ValueError("`expect_dim` must be None, '2d', or '3d'.")

    if coord_slice_map is None:
        coord_slice_map = {'t': 0, 'x': 1, 'y': 2}

    def _ensure_3d_and_validate(tensor, name):
        """Helper to convert to tensor, ensure 3D, and validate."""
        if not isinstance(tensor, Tensor):
            tensor = tf_convert_to_tensor(tensor, dtype=tf_float32)
        
        if expect_dim == '2d' and tensor.ndim != 2:
             raise ValueError(
                f"Input '{name}' must be 2D (expect_dim='2d'), "
                f"but got rank {tensor.ndim}."
            )
        elif expect_dim == '3d' and tensor.ndim != 3:
             raise ValueError(
                f"Input '{name}' must be 3D (expect_dim='3d'), "
                f"but got rank {tensor.ndim}."
            )

        # For consistency, always expand 2D tensors to 3D.
        if tensor.ndim == 2:
            vlog(f"Expanding 2D input '{name}' to 3D for consistency.",
                 level=3, verbose=verbose, logger=_logger)
            return tf_expand_dims(tensor, axis=1)
        elif tensor.ndim == 3:
            return tensor
        else:
            raise ValueError(
                f"Input '{name}' must be a 2D or 3D tensor, but got "
                f"rank {tensor.ndim} with shape {tensor.shape}."
            )

    # --- Main Logic ---
    if isinstance(inputs, dict):
        if 'coords' in inputs:
            coords_tensor = _ensure_3d_and_validate(
                inputs['coords'], 'coords')
        elif all(k in inputs for k in ('t', 'x', 'y')):
            t = _ensure_3d_and_validate(inputs['t'], 't')
            x = _ensure_3d_and_validate(inputs['x'], 'x')
            y = _ensure_3d_and_validate(inputs['y'], 'y')
            # The shapes are now guaranteed to be 3D.
            return tf_cast(t, tf_float32), tf_cast(x, tf_float32), tf_cast(y, tf_float32)
        else:
            raise ValueError(
                "Dict `inputs` must contain either 'coords' key "
                "or all of 't', 'x', 'y' keys."
            )
    elif isinstance(inputs, (Tensor, np.ndarray)):
        coords_tensor = _ensure_3d_and_validate(inputs, 'inputs')
    else:
        raise TypeError(
            f"`inputs` must be a tensor/array or dict; got {type(inputs)}"
        )

    # Slice the now-guaranteed 3D coords_tensor
    tf_debugging.assert_greater_equal(
       tf_shape(coords_tensor)[-1],
       3,
       message=(
           "Coordinate tensor must carry at least three features "
           "(t, x, y) in the last dimension,"
           f" but got shape {coords_tensor.shape}"
       )
   )
    
    # if tf_shape(coords_tensor)[-1] < 3:
    #     raise ValueError(
    #         "Coordinate tensor must have at least 3 features (t,x,y) "
    #         f"in the last dimension, but got shape {coords_tensor.shape}"
    #     )
        
    t = coords_tensor[..., coord_slice_map['t']:coord_slice_map['t'] + 1]
    x = coords_tensor[..., coord_slice_map['x']:coord_slice_map['x'] + 1]
    y = coords_tensor[..., coord_slice_map['y']:coord_slice_map['y'] + 1]

    vlog(f"Successfully extracted t:{t.shape}, x:{x.shape}, "
         f"y:{y.shape}", level=2, verbose=verbose, logger=_logger)
         
    return tf_cast(t, tf_float32), tf_cast(x, tf_float32), tf_cast(y, tf_float32)

@ensure_pkg(
    KERAS_BACKEND or "tensorflow",
    extra="TensorFlow is required for this function."
   )
def extract_txy(
    inputs: Union[Tensor, np.ndarray, Dict[str, Union[Tensor, np.ndarray]]],
    coord_slice_map: Optional[Dict[str, int]] = None,
    expect_dim: Optional[str] = None,
    verbose: int = 0,
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    **kws
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Extracts t, x, y tensors from various input formats.

    This utility standardizes coordinate inputs, accepting a single
    tensor or a dictionary, and handling both 2D (spatial/static)
    and 3D (spatio-temporal) data with flexible dimension validation.

    Parameters
    ----------
    inputs : tf.Tensor, np.ndarray, or dict
        The input data containing coordinates. Can be a single tensor
        or a dictionary with 'coords' or 't', 'x', 'y' keys.

    coord_slice_map : dict, optional
        Mapping for 't', 'x', 'y' to their index in the last
        dimension of a coordinate tensor.
        Defaults to `{'t': 0, 'x': 1, 'y': 2}`.

    expect_dim : {'2d', '3d', '3d_only'}, optional
        Enforces a constraint on the input's dimension.
        - ``'2d'``: Input must resolve to a 2D shape like `(batch, 3)`.
        - ``'3d'``: Accepts 3D input. If 2D is given, it's expanded
          to 3D with a time dimension of 1.
        - ``'3d_only'``: Input must be 3D; raises an error for 2D input.
        - ``None`` (default): Accepts both 2D and 3D inputs without
          changing their rank.

    verbose : int, default 0
        Controls logging verbosity.

    Returns
    -------
    t, x, y : Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        The extracted t, x, and y coordinate tensors. Their rank (2D
        or 3D) depends on the input and the `expect_dim` mode.

    Raises
    ------
    ValueError
        If input format is unsupported, dimensions are inconsistent,
        or `expect_dim` constraint is violated.
    """
    vlog("Extracting (t, x, y) coordinates from inputs...",
         level=2, verbose=verbose, logger=_logger)
         
    if expect_dim and expect_dim not in ['2d', '3d', '3d_only']:
        raise ValueError(
            "`expect_dim` must be None, '2d', '3d', or '3d_only'."
        )

    if coord_slice_map is None:
        coord_slice_map = {'t': 0, 'x': 1, 'y': 2}

    def _process_tensor(tensor, name):
        """Helper to convert to tensor and validate dimension."""
        if not isinstance(tensor, Tensor):
            tensor = tf_convert_to_tensor(tensor, dtype=tf_float32)

        input_ndim = tensor.ndim
        
        # Validate against expect_dim
        if expect_dim == '2d' and input_ndim != 2:
            raise ValueError(
                f"Input '{name}' must be 2D for expect_dim='2d', "
                f"but got rank {input_ndim}."
            )
        elif expect_dim == '3d_only' and input_ndim != 3:
             raise ValueError(
                f"Input '{name}' must be 3D for expect_dim='3d_only', "
                f"but got rank {input_ndim}."
            )
        
        # Handle expansion for '3d' mode
        if expect_dim == '3d' and input_ndim == 2:
            vlog(f"Expanding 2D input '{name}' to 3D for expect_dim='3d'.",
                 level=3, verbose=verbose, logger=_logger)
            return tf_expand_dims(tensor, axis=1)

        # For all other cases, including `expect_dim=None`, return as is
        # after basic rank validation.
        if input_ndim not in [2, 3]:
            raise ValueError(
                f"Input '{name}' must be a 2D or 3D tensor, but got "
                f"rank {input_ndim} with shape {tensor.shape}."
            )
        return tensor

    # --- Main Logic ---
    if isinstance(inputs, dict):
        if 'coords' in inputs:
            coords_tensor = _process_tensor(inputs['coords'], 'coords')
        elif all(k in inputs for k in ('t', 'x', 'y')):
            # When t,x,y are separate, they are typically 1D or 2D.
            # We process them and then concatenate.
            t_p = _process_tensor(inputs['t'], 't')
            x_p = _process_tensor(inputs['x'], 'x')
            y_p = _process_tensor(inputs['y'], 'y')
            
            return tf_cast(
                t_p, tf_float32), tf_cast(
                    x_p, tf_float32), tf_cast(y_p, tf_float32)
                    
            # The individual tensors might be 2D or 3D. Concat will work if
            # their ranks match, which is handled by _process_tensor.
            # coords_tensor = tf_concat([t_p, x_p, y_p], axis=-1)
        else:
            raise ValueError(
                "Dict `inputs` must contain either 'coords' key "
                "or all of 't', 'x', 'y' keys."
            )
    elif isinstance(inputs, (Tensor, np.ndarray)):
        coords_tensor = _process_tensor(inputs, 'inputs')
    else:
        raise TypeError(
            f"`inputs` must be a tensor/array or dict; got {type(inputs)}"
        )

    # Slice the processed coords_tensor
    if tf_shape(coords_tensor)[-1] < 3:
        raise ValueError(
            "Coordinate tensor must have at least 3 features (t,x,y) "
            f"in the last dimension, but got shape {coords_tensor.shape}"
        )
        
    # Slicing keeps the original number of dimensions
    t = coords_tensor[..., coord_slice_map['t']:coord_slice_map['t'] + 1]
    x = coords_tensor[..., coord_slice_map['x']:coord_slice_map['x'] + 1]
    y = coords_tensor[..., coord_slice_map['y']:coord_slice_map['y'] + 1]

    vlog(f"Successfully extracted t:{t.shape}, x:{x.shape}, "
         f"y:{y.shape}", level=2, verbose=verbose, logger=_logger)
         
    return tf_cast(t, tf_float32), tf_cast(x, tf_float32), tf_cast(y, tf_float32)


@ensure_pkg(
    KERAS_BACKEND or "tensorflow",
    extra="TensorFlow is required for this function."
   )
def plot_hydraulic_head(
    model: Model,
    t_slice: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    resolution: int = 100,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = 'viridis',
    colorbar_label: str = 'Hydraulic Head (h)',
    save_path: Optional[str] = None,
    show_plot: bool = True,
    **contourf_kwargs: Any
) -> Tuple[plt.Axes, ScalarMappable]:
    r"""Generate and plot a 2D contour map of a hydraulic head solution.

    This utility visualizes the output of a Physics-Informed Neural
    Network (PINN) that solves for the hydraulic head
    :math:`h(t, x, y)`. It automates the process of creating a
    spatial grid, running model predictions, and generating a
    publication-quality contour plot for a specific slice in time.

    Parameters
    ----------
    model : tf.keras.Model
        The trained PINN model. It is expected to have a ``.predict()``
        method that accepts a dictionary of tensors with keys
        ``{'t', 'x', 'y'}``.
    t_slice : float
        The specific point in time :math:`t` for which to plot the
        2D spatial solution.
    x_bounds : tuple of float
        A tuple ``(x_min, x_max)`` defining the spatial domain for
        the x-axis.
    y_bounds : tuple of float
        A tuple ``(y_min, y_max)`` defining the spatial domain for
        the y-axis.
    resolution : int, optional
        The number of points to sample along each spatial axis,
        creating a grid of ``resolution x resolution`` points for
        prediction. Higher values result in a smoother plot.
        Default is 100.
    ax : matplotlib.axes.Axes, optional
        A pre-existing Matplotlib Axes object to plot on. If ``None``,
        a new figure and axes are created internally. This is useful
        for embedding this plot within a larger figure arrangement.
        Default is ``None``.
    title : str, optional
        A custom title for the plot. If ``None``, a default title
        is generated using the value of `t_slice`. Default is ``None``.
    cmap : str, optional
        The name of the Matplotlib colormap to use for the contour
        plot. Default is ``'viridis'``.
    colorbar_label : str, optional
        The text label for the color bar. Default is
        ``'Hydraulic Head (h)'``.
    save_path : str, optional
        If provided, the path (including filename and extension)
        where the generated plot will be saved. This is only active
        when the function creates its own figure (i.e., when `ax`
        is ``None``). Default is ``None``.
    show_plot : bool, optional
        If ``True``, calls ``plt.show()`` to display the plot. This
        is only active when the function creates its own figure.
        Default is ``True``.
    **contourf_kwargs : any
        Additional keyword arguments that are passed directly to the
        ``matplotlib.pyplot.contourf`` function. This allows for
        advanced customization (e.g., ``levels=20``, ``extend='both'``).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Matplotlib Axes object on which the contour plot was drawn.
    contour : matplotlib.cm.ScalarMappable
        The contour plot object, which can be used for further
        customizations, such as modifying the color bar.

    See Also
    --------
    fusionlab.nn.pinn.PiTGWFlow : The PINN model this function is
                                 designed to visualize.

    Notes
    -----
    The core mechanism of this function involves creating a 2D
    meshgrid of :math:`(x, y)` coordinates. These grid points are then
    "flattened" into a long list of points, as the PINN model expects
    a batch of individual coordinates for prediction, not a grid.

    The prediction process is as follows:

    1.  A grid of shape ``(resolution, resolution)`` is created for
        :math:`x` and :math:`y`.
    2.  These grids are reshaped into column vectors of shape
        ``(resolution*resolution, 1)``.
    3.  A time vector of the same shape, filled with `t_slice`, is
        created.
    4.  The model's ``.predict()`` method is called on these flat
        tensors.
    5.  The resulting flat prediction vector is reshaped back to the
        original ``(resolution, resolution)`` grid shape for plotting.

    If a custom `ax` is provided, the user is responsible for calling
    ``plt.show()`` or saving the parent figure.

    Examples
    --------
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> import matplotlib.pyplot as plt
    >>> # This is a mock model for demonstration purposes.
    >>> # In practice, you would use a trained PiTGWFlow model.
    >>> class MockPINN(tf.keras.Model):
    ...     def call(self, inputs):
    ...         # A simple analytical function for demonstration
    ...         t, x, y = inputs['t'], inputs['x'], inputs['y']
    ...         return tf.sin(np.pi * x) * tf.cos(np.pi * y) * tf.exp(-t)
    ...
    >>> mock_model = MockPINN()

    **1. Simple Plotting Example**

    This example creates a single plot and saves it to a file.

    >>> ax, contour = plot_hydraulic_head(
    ...     model=mock_model,
    ...     t_slice=0.5,
    ...     x_bounds=(-1, 1),
    ...     y_bounds=(-1, 1),
    ...     resolution=50,
    ...     save_path="hydraulic_head_t0.5.png",
    ...     show_plot=False  # Do not display interactively
    ... )
    Plot saved to hydraulic_head_t0.5.png

    **2. Advanced Example with Subplots**

    This example shows how to use the `ax` parameter to draw the
    solution at two different times side-by-side in one figure.

    >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    >>> fig.suptitle('Hydraulic Head at Different Times', fontsize=16)
    ...
    >>> # Plot solution at t = 0.1
    >>> plot_hydraulic_head(
    ...     model=mock_model, t_slice=0.1, x_bounds=(-1, 1),
    ...     y_bounds=(-1, 1), ax=ax1, show_plot=False
    ... )
    ...
    >>> # Plot solution at t = 1.0
    >>> plot_hydraulic_head(
    ...     model=mock_model, t_slice=1.0, x_bounds=(-1, 1),
    ...     y_bounds=(-1, 1), ax=ax2, show_plot=False
    ... )
    ...
    >>> plt.tight_layout(rect=[0, 0, 1, 0.96])
    >>> plt.show()
    """
    # ... function implementation follows ...
    # --- 1. Handle Matplotlib Figure and Axes ---
    if ax is None:
        # Create a new figure only if no axes are provided
        fig, current_ax = plt.subplots(figsize=(9, 7))
        # Flag to indicate we have control over the figure object
        fig_created = True
    else:
        current_ax = ax
        fig = current_ax.figure
        fig_created = False

    # --- 2. Create Prediction Grid ---
    x_range = np.linspace(x_bounds[0], x_bounds[1], resolution)
    y_range = np.linspace(y_bounds[0], y_bounds[1], resolution)
    X, Y = np.meshgrid(x_range, y_range)

    # Flatten the grid to a list of points for model prediction
    x_flat = tf_convert_to_tensor(X.ravel(), dtype=tf_float32)
    y_flat = tf_convert_to_tensor(Y.ravel(), dtype=tf_float32)
    t_flat = tf_fill(x_flat.shape, t_slice)

    # Reshape to column vectors (N, 1) as expected by the model
    grid_coords = {
        't': tf_reshape(t_flat, (-1, 1)),
        'x': tf_reshape(x_flat, (-1, 1)),
        'y': tf_reshape(y_flat, (-1, 1))
    }

    # --- 3. Run Model Prediction ---
    h_pred_flat = model.predict(grid_coords)
    # Reshape the flat predictions back to the grid shape for plotting
    h_pred_grid = tf_reshape(h_pred_flat, X.shape)

    # --- 4. Plotting ---
    # Set default contour levels if not provided
    if 'levels' not in contourf_kwargs:
        contourf_kwargs['levels'] = 100

    contour = current_ax.contourf(
        X, Y, h_pred_grid, cmap=cmap, **contourf_kwargs
    )

    # Add color bar
    fig.colorbar(contour, ax=current_ax, label=colorbar_label)

    # Set plot labels and title
    if title is None:
        title = f'Learned Hydraulic Head Solution at t = {t_slice}'
    current_ax.set_title(title, fontsize=14)
    current_ax.set_xlabel('x-coordinate')
    current_ax.set_ylabel('y-coordinate')
    current_ax.set_aspect('equal')

    # --- 5. Save and Show ---
    if fig_created:
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            # If not showing, close the figure to free up memory
            plt.close(fig)

    return current_ax, contour
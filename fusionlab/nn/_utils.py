# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Utility functions for neural networks models.

This module provides utility functions to preprocess data for Temporal Fusion 
Transformer (TFT) models, including splitting sequences into static and dynamic 
inputs and creating input sequences with corresponding targets for time series 
forecasting.

"""
import os
import json
import time
import datetime 
from numbers import Integral, Real 
from textwrap import dedent 
import warnings
import logging
from typing import ( 
    List, Tuple, Optional,
    Union, Dict, Callable, 
    Any, Sequence
)
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import r2_score

from .._fusionlog import fusionlog
from ..api.util import get_table_size
from ..core.checks import (
    ParamsValidator, 
    are_all_frames_valid, 
    exist_features, 
    check_params, 
    check_spatial_columns,
    check_empty,
    assert_ratio,
    check_datetime,
    )
from ..core.diagnose_q import ( 
    check_forecast_mode, 
    validate_quantiles, 
    validate_consistency_q
)
from ..core.handlers import TypeEnforcer, columns_manager 
from ..core.io import is_data_readable 
from ..compat.sklearn import ( 
    StrOptions, 
    Interval, 
    HasMethods, 
    Hidden, 
    validate_params
)
from ..decorators import Deprecated, Appender, isdf 
from ..exceptions import NotEnoughDataError 
from ..utils.data_utils import ( 
    mask_by_reference
)
from ..utils.deps_utils import ensure_pkg, get_versions 
from ..utils.generic_utils import ( 
    print_box, 
    vlog
)
from ..utils.io_utils import save_job
from ..utils.sys_utils import BatchDataFrameBuilder, build_large_df 

from ..utils.validator import ( 
    validate_sequences, 
    check_consistent_length, 
    parameter_validator,
    is_frame, 
    validate_positive_integer, 
)
from . import KERAS_DEPS, KERAS_BACKEND, dependency_message
from .keras_validator import validate_keras_model, check_keras_model_status 

if KERAS_BACKEND:
    from ..compat.tf import optional_tf_function # Noqa E:404
    
Callback=KERAS_DEPS.Callback
Tensor =KERAS_DEPS.Tensor
Model =KERAS_DEPS.Model
Dataset =KERAS_DEPS.Dataset 
History =KERAS_DEPS.History 

tf_convert_to_tensor =KERAS_DEPS.convert_to_tensor

tf_float32 = KERAS_DEPS.float32
tf_int32 = KERAS_DEPS.int32
tf_rank = KERAS_DEPS.rank 
tf_cast = KERAS_DEPS.cast
tf_squeeze =KERAS_DEPS.squeeze
tf_zeros = KERAS_DEPS.zeros 
tf_shape =KERAS_DEPS.shape 
tf_concat = KERAS_DEPS.concat
tf_constant =KERAS_DEPS.constant
tf_Assert = KERAS_DEPS.Assert 
tf_debugging = KERAS_DEPS.debugging 
tf_equal = KERAS_DEPS.equal 
tf_control_dependencies= KERAS_DEPS.control_dependencies
tf_identity =KERAS_DEPS.identity
tf_greater_equal =KERAS_DEPS.greater_equal
tf_logical_and =KERAS_DEPS.logical_and 
tf_cond = KERAS_DEPS.cond 
tf_stack =KERAS_DEPS.stack 
tf_no_op = KERAS_DEPS.no_op
tf_print = KERAS_DEPS.print 

   
DEP_MSG = dependency_message('nn.utils') 
_TW = get_table_size()

logger = fusionlog().get_fusionlab_logger(__name__)


__all__ = [
    "split_static_dynamic", 
    "create_sequences",
    "compute_forecast_horizon", 
    "prepare_spatial_future_data", 
    "compute_anomaly_scores",
    "reshape_xtft_data", 
    "reshape_xtft_data_in", 
    "generate_forecast", 
    "generate_forecast_with", 
    "forecast_multi_step", 
    "forecast_single_step", 
    "step_to_long", 
    "prepare_model_inputs" , 
    "extract_batches_from_dataset", 
    "format_predictions", 
    "export_keras_losses"
   ]

def export_keras_losses(
    history: History,
    keys: Optional[List[str]] = None,
    savefile: Optional[str] = None,
    verbose: int = 0,
    formats: Tuple[str, ...] = ("json", "csv"),
) -> dict:
    """
    Export loss(es) (and any other metric) from a Keras History object.

    Parameters
    ----------
    history
        The History object returned by model.fit().
    keys
        List of history.history keys to export, e.g.
        ["loss", "val_loss", "physics_loss"]. If None, defaults to
        all keys ending with "loss".
    savefile
        Path (optionally with extension) where to write the output.
        If extension is:
         - .json → writes only JSON
         - .csv  → writes only CSV
        If no extension given, will write all formats in `formats`
        using `savefile` as the base name.
    verbose
        If >0, prints status messages.
    formats
        File formats to write when `savefile` has no extension.
        Valid entries are "json" and "csv".

    Returns
    -------
    result : dict
        {
            "epochs_run": int,
            "<key1>": [ ... ],
            "<key2>": [ ... ],
            ...
        }
    """
    if isinstance(history, History):
        hist = history.history
    elif isinstance(history, dict):
        hist = history
    else:
        raise TypeError(
            "`history` must be a Keras History object or a dictionary."
        )
  
    # 1) Determine which keys to export
    if keys is None:
        keys = [k for k in hist if k.endswith("loss")]
    missing = [k for k in keys if k not in hist]
    if missing:
        raise ValueError(f"Requested keys not in history: {missing!r}")

    # 2) Build the results dict
    epochs = len(hist[keys[0]])
    result = {"epochs_run": epochs}
    for k in keys:
        result[k] = hist[k]

    # 3) If savefile provided, write out
    if savefile:
        base, ext = os.path.splitext(savefile)
        # Determine which extensions to write
        if ext:
            exts = [ext.lstrip(".").lower()]
        else:
            exts = list(formats)

        for e in exts:
            out_path = (
                savefile if ext else f"{base}.{e}"
            )
            if e == "json":
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=4)
                if verbose:
                    print(
                        f"[export_keras_losses] Wrote JSON to {out_path}")
            elif e == "csv":
                df = pd.DataFrame(result)
                df['epochs_run'] = range (1, epochs + 1 ) 
                df.to_csv(out_path, index=False)
                if verbose:
                    print(
                        f"[export_keras_losses] Wrote CSV to {out_path}")
            else:
                if verbose:
                    print(
                        f"[export_keras_losses] Skipping unsupported format: {e!r}")

    return result

def extract_batches_from_dataset(
    dataset: Dataset,
    num_batches_to_extract: Union[int, str] = 1,
    agg: bool = False, 
    errors: str = 'warn'
) -> Union[List[Tuple[Any, ...]], Optional[Tuple[Any, ...]]]:
    """
    Extracts a specified number of batches from a tf.data.Dataset.
    Optionally aggregates the extracted batches.

    Parameters:
    ----------
    dataset : tf.data.Dataset
        The TensorFlow dataset to extract batches from.
    num_batches_to_extract : Union[int, str], default 1
        Number of batches: int, or 'all', '*', 'auto'.
    agg : bool, default False
        If True, attempts to aggregate the extracted batches into a single
        tuple structure by concatenating corresponding tensors/arrays or
        aggregating dictionaries.
    errors : str, default 'warn'
        Error handling: 'raise', 'warn', 'ignore'.

    Returns:
    -------
    Union[List[Tuple[Any, ...]], Optional[Tuple[Any, ...]]]
        - If `agg` is False: A list of batch tuples.
        - If `agg` is True: A single tuple representing the aggregated batch data,
          or `None` if no batches were extracted.
        Returns an empty list (if `agg=False`) or `None` (if `agg=True`)
        if 0 batches are requested or the dataset is empty.

    Raises:
    ------
    TypeError
        If `dataset` is not a `tf.data.Dataset` or `num_batches_to_extract`
        is invalid (and `errors='raise'`).
    ValueError
        If `num_batches_to_extract` is negative, or fewer batches are available
        than requested (and `errors='raise'` and not taking all).
    RuntimeError
        For unexpected errors during dataset iteration (and `errors='raise'`).
    """
    # Validate 'dataset' type
    if not isinstance(dataset, Dataset):
        msg = "Input 'dataset' must be a tf.data.Dataset instance."
        if errors == 'raise':
            raise TypeError(msg)
        elif errors == 'warn':
            logger.warning(msg + " Returning empty list/None.")
        return None if agg else []

    # Validate 'errors' parameter
    if errors not in ['raise', 'warn', 'ignore']:
        warnings.warn(
            f"Invalid 'errors' value: '{errors}'. Defaulting to 'warn'.",
            UserWarning
        )
        errors = 'warn'

    # Process 'num_batches_to_extract'
    actual_num_to_extract: int
    take_all: bool = False

    if isinstance(num_batches_to_extract, str) and \
       num_batches_to_extract.lower() in ['all', '*', 'auto']:
        take_all = True
        actual_num_to_extract = -1 # Placeholder
    elif isinstance(num_batches_to_extract, int):
        if num_batches_to_extract < 0:
            msg = (f"num_batches_to_extract cannot be negative "
                   f"({num_batches_to_extract}).")
            if errors == 'raise':
                raise ValueError(msg)
            elif errors == 'warn':
                warnings.warn(
                    msg + " Returning 0 batches (empty list/None).",
                    UserWarning)
            return None if agg else []
        elif num_batches_to_extract == 0:
            return None if agg else []
        else:
            actual_num_to_extract = num_batches_to_extract
    else:
        msg = (f"Invalid type for num_batches_to_extract: "
               f"{type(num_batches_to_extract)}."
               " Must be int or str ('all', '*', 'auto').")
        if errors == 'raise':
            raise TypeError(msg)
        elif errors == 'warn':
            warnings.warn(
                msg + " Defaulting to extracting 1 batch.",
                UserWarning)
            actual_num_to_extract = 1
        else: # errors == 'ignore'
            logger.debug(msg + " Defaulting to extracting 1 batch.")
            actual_num_to_extract = 1

    collected_batches: List[Tuple[Any, ...]] = []
    dataset_iterator = iter(dataset)
    batches_extracted_count = 0

    if take_all:
        try:
            while True:
                batch = next(dataset_iterator)
                collected_batches.append(batch)
                batches_extracted_count += 1
        except StopIteration:
            logger.debug(f"Finished extracting all {batches_extracted_count} "
                          "batches from the dataset.")
        except Exception as e:
            msg = f"An unexpected error occurred while extracting all batches: {e}"
            if errors == 'raise':
                raise RuntimeError(msg) from e
            elif errors == 'warn':
                logger.error(msg)
                warnings.warn(
                    "Error during 'all' batch extraction. "
                    "Returning partially collected batches or None if aggregating.",
                    RuntimeWarning)
            # Return what was collected before the error, then aggregate if needed
            if agg: # Aggregation happens after collection or partial collection
                if not collected_batches: return None
                # (Aggregation logic will be applied below)
            else:
                return collected_batches # Return list of batches
            
    else: # Specific number of batches
        try:
            for _ in range(actual_num_to_extract):
                batch = next(dataset_iterator)
                collected_batches.append(batch)
                batches_extracted_count += 1
        except StopIteration:
            if batches_extracted_count < actual_num_to_extract:
                msg = (f"Dataset exhausted. Requested"
                       f" {actual_num_to_extract} batches, "
                       f"but only {batches_extracted_count} were available.")
                if errors == 'raise':
                    raise ValueError(msg)
                elif errors == 'warn':
                    warnings.warn(
                        msg + " Returning available batches"
                        " (or aggregated if agg=True).", UserWarning)
        except Exception as e:
            msg = (f"An unexpected error occurred while extracting "
                   f"{actual_num_to_extract} batches: {e}")
            if errors == 'raise':
                raise RuntimeError(msg) from e
            elif errors == 'warn':
                logger.error(msg)
                warnings.warn(
                    f"Error during batch extraction"
                    f" (requested {actual_num_to_extract}). "
                    "Returning partially collected batches or"
                    " None if aggregating.",
                    RuntimeWarning)
            if agg:
                if not collected_batches: return None
            else:
                return collected_batches

    # Handle case where dataset was empty and specific number > 0 was requested
    if not take_all and actual_num_to_extract > 0 and batches_extracted_count == 0:
        msg = (f"Dataset appears to be empty. Could not extract the "
               f"requested {actual_num_to_extract} batch(es).")
        if errors == 'warn':
            warnings.warn(msg + " Returning empty list/None.", UserWarning)
        elif errors == 'ignore':
            logger.debug(msg + " Returning empty list/None.")
        return None if agg else []

    # --- Aggregation Step ---
    if agg:
        if not collected_batches:
            return None

        try:
            # Assuming all batches have the same tuple structure
            num_elements_in_batch_tuple = len(collected_batches[0])
            aggregated_result_parts = []

            for i in range(num_elements_in_batch_tuple):
                component_parts = [batch[i] for batch in collected_batches]
                aggregated_component = _aggregate_component(component_parts)
                aggregated_result_parts.append(aggregated_component)
            
            return tuple(aggregated_result_parts)
        except Exception as e:
            msg = f"Error during aggregation of batches: {e}"
            if errors == 'raise':
                raise RuntimeError(msg) from e
            elif errors == 'warn':
                logger.error(msg)
                warnings.warn("Aggregation failed. Returning None.",
                              RuntimeWarning)
            return None # Failed aggregation
    else:
        return collected_batches


def _aggregate_component(parts: List[Any]) -> Any:
    """
    Helper function to aggregate a list of components from batches.
    Tries to concatenate if components are TF tensors or NumPy arrays.
    If components are dicts, aggregates their values recursively.
    """
    if not parts:
        return None

    first_part = parts[0]

    if isinstance(first_part, dict):
        aggregated_dict: Dict[str, Any] = {}
        # Assume all dicts in 'parts' have similar keys as the first one
        # More robustly, one might collect all unique keys first.
        for key in first_part.keys():
            # Collect items for this key from all dicts in 'parts'
            items_for_key = []
            for p in parts:
                if isinstance(p, dict) and key in p:
                    items_for_key.append(p[key])
                else: # Handle cases where a key might be missing in some dicts
                    logger.debug(
                        f"Key '{key}' not found in a batch"
                        " component during aggregation.")
            
            if not items_for_key:
                aggregated_dict[key] = None # Or an empty list: []
                continue

            # Try to concatenate if all items for this key are tensors/arrays
            if all(isinstance(item, Tensor) for item in items_for_key):
                try:
                    aggregated_dict[key] = tf_concat(items_for_key, axis=0)
                except Exception as e:
                    logger.debug(
                        f"TF concat failed for dict key '{key}':"
                        f" {e}. Falling back to list.")
                    aggregated_dict[key] = items_for_key
            elif all(isinstance(item, np.ndarray) for item in items_for_key):
                try:
                    aggregated_dict[key] = np.concatenate(items_for_key, axis=0)
                except Exception as e:
                    logger.debug(
                        f"NumPy concat failed for dict key '{key}':"
                        f" {e}. Falling back to list.")
                    aggregated_dict[key] = items_for_key
            else:
                # If types are mixed or not concat-able, store as a list
                aggregated_dict[key] = items_for_key
        return aggregated_dict
    
    elif all(isinstance(p, Tensor) for p in parts):
        try:
            return tf_concat(parts, axis=0)
        except Exception as e:
            logger.debug(
                f"TF concat failed for tensor list: {e}."
                " Falling back to list.")
            return parts
    elif all(isinstance(p, np.ndarray) for p in parts):
        try:
            return np.concatenate(parts, axis=0)
        except Exception as e:
            logger.debug("NumPy concat failed for ndarray list:"
                         f" {e}. Falling back to list.")
            return parts
    else:
        # If components are not dicts and not all tensors/arrays,
        # return the list of parts
        return parts
    
    
def format_predictions(
    predictions: Optional[Union[np.ndarray, Tensor]] = None,
    model: Optional[Model] = None,
    inputs: Optional[List[Optional[Union[np.ndarray, Tensor]]]] = None,
    y_true_sequences: Optional[Union[np.ndarray, Tensor]] = None,
    target_name: Optional[str] = "target",
    quantiles: Optional[List[float]] = None,
    forecast_horizon: Optional[int] = None,
    output_dim: Optional[int] = None,
    spatial_data_array: Optional[Union[np.ndarray, Tensor]] = None,
    spatial_cols: Optional[List[str]] = None,
    spatial_cols_indices: Optional[List[int]] = None,
    evaluate_coverage: bool = False,
    scaler: Optional[Any] = None, 
    scaler_feature_names: Optional[List[str]] = None,
    target_idx_in_scaler: Optional[int] = None,
    verbose: int = 0,
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    **kwargs: Any
) -> pd.DataFrame:

    """Formats model predictions into a structured pandas DataFrame.

    This utility function takes raw model predictions (either directly
    as an array/tensor or generated by a provided model and its inputs)
    and transforms them into a long-format pandas DataFrame. It can
    handle point forecasts, quantile forecasts, single or multi-output
    predictions, and optionally include actual target values, spatial
    identifiers, and perform coverage score evaluation for quantile
    forecasts.
    
    The output DataFrame is structured with 'sample_idx' (identifying
    the original input sequence) and 'forecast_step' (from 1 to H,
    where H is the forecast horizon).
    
    Parameters
    ----------
    predictions : np.ndarray or tf.Tensor, optional
        The raw prediction tensor or array.
        - For point forecasts, expected shapes:
            - (num_samples, forecast_horizon, output_dim)
            - (num_samples, forecast_horizon) if output_dim=1 (will be reshaped)
            - (num_samples, output_dim) if forecast_horizon=1 (will be reshaped)
        - For quantile forecasts, expected shapes:
            - (num_samples, forecast_horizon, num_quantiles * output_dim)
            - (num_samples, forecast_horizon, num_quantiles, output_dim)
            - (num_samples, num_quantiles * output_dim) if forecast_horizon=1
        If ``None``, `model` and `inputs` must be provided.
        Default is ``None``.
    model : tf.keras.Model, optional
        A trained Keras model to generate predictions if `predictions`
        is not provided. Used in conjunction with `inputs`.
        Default is ``None``.
    inputs : List[Optional[Union[np.ndarray, tf.Tensor]]], optional
        A list of input tensors (e.g., [static, dynamic, future])
        required by the `model` to generate predictions. Required if
        `predictions` is ``None`` and `model` is provided.
        Default is ``None``.
    y_true_sequences : np.ndarray or tf.Tensor, optional
        The true target values corresponding to the predictions, used
        for including actuals in the output DataFrame and for
        evaluation. Expected shape:
        `(num_samples, forecast_horizon, output_dim)`.
        Default is ``None``.
    target_name : str, optional
        Base name for the target variable. Used to prefix prediction
        and actual column names (e.g., "sales_pred", "sales_q50",
        "sales_actual"). Default is "target".
    quantiles : List[float], optional
        A list of quantiles that were predicted by the model (e.g.,
        `[0.1, 0.5, 0.9]`). Required if the predictions are quantile
        forecasts. If provided, prediction columns will be named like
        `{target_name}_q10`, `{target_name}_q50`, etc.
        Default is ``None`` (for point forecasts).
    forecast_horizon : int, optional
        The number of time steps into the future that the model
        predicts. If not provided, it's inferred from
        `predictions.shape[1]` or `y_true_sequences.shape[1]`.
        Default is ``None``.
    output_dim : int, optional
        The number of target variables predicted at each time step
        (e.g., 1 for univariate, >1 for multivariate target).
        If not provided, it's inferred from the shape of
        `predictions` or `y_true_sequences`. Default is ``None``.
    spatial_data_array : np.ndarray or tf.Tensor or pd.DataFrame or pd.Series, optional
        An array or DataFrame containing static spatial/identifier
        features for each of the `num_samples` sequences.
        - If NumPy/Tensor: Expected shape `(num_samples, num_spatial_features)`.
          `spatial_cols_indices` must be provided.
        - If DataFrame/Series: Must have `num_samples` rows.
          `spatial_cols` must be provided.
        These features will be repeated for each forecast step in the
        output DataFrame. Default is ``None``.
    spatial_cols : List[str], optional
        List of column names to select from `spatial_data_array` if it's
        a DataFrame/Series, or names to assign to columns if
        `spatial_data_array` is NumPy/Tensor and `spatial_cols_indices`
        are provided. Default is ``None``.
    spatial_cols_indices : List[int], optional
        List of column indices to select from `spatial_data_array` if
        it's a NumPy/Tensor. Length must match `spatial_cols` if
        provided. Default is ``None``.
    evaluate_coverage : bool, default False
        If ``True``, `quantiles` are provided (at least two), and
        `y_true_sequences` is available, calculates the coverage score
        using the first and last quantiles as interval bounds.
        Requires `fusionlab.metrics.coverage_score`.
    scaler : Any, optional
        A fitted scikit-learn-like scaler object (must have an
        `inverse_transform` method) used to scale the target variable
        and potentially other features. If provided along with
        `scaler_feature_names` and `target_idx_in_scaler`, predictions
        and actuals for the target will be inverse-transformed.
        Default is ``None``.
    scaler_feature_names : List[str], optional
        A list of all feature names (in order) that the `scaler` was
        originally fit on. Required if `scaler` is provided and
        targeted inverse transformation is needed. Default is ``None``.
    target_idx_in_scaler : int, optional
        The index of the `target_name` within the
        `scaler_feature_names` list. Required if `scaler` is
        provided and targeted inverse transformation is needed.
        Default is ``None``.
    verbose : int, default 0
        Verbosity level for logging during processing.
        - ``0``: Silent.
        - ``1``: Basic info.
        - ``3``: More detailed steps.
        - ``5``: Very detailed shape information.
    **kwargs : Any
        Additional keyword arguments (currently not used but included
        for future extensibility).
    
    Returns
    -------
    pandas.DataFrame
        A long-format DataFrame with columns including:
        - `sample_idx`: Identifier for the original input sequence.
        - `forecast_step`: Time step within the forecast horizon (1 to H).
        - Spatial columns (if `spatial_data_array` provided).
        - Prediction columns:
            - Point forecast: `{target_name}_pred` (or
              `{target_name}_{output_idx}_pred` for multi-output).
            - Quantile forecast: `{target_name}_qXX` (or
              `{target_name}_{output_idx}_qXX` for multi-output).
        - Actual value columns (if `y_true_sequences` provided):
            - `{target_name}_actual` (or
              `{target_name}_{output_idx}_actual`).
        Values in prediction and actual columns will be inverse-
        transformed if a valid `scaler` and related parameters are given.
    
    Raises
    ------
    ValueError
        If `predictions` is None and `model` or `inputs` is also None.
        If `predictions` shape is invalid (not 2D, 3D, or 4D).
        If `quantiles` are provided but prediction shape is incompatible
        for inferring `output_dim`.
        If `spatial_data_array` is provided without necessary name/index
        parameters.
    TypeError
        If `predictions` or other inputs cannot be converted to the
        expected tensor/array types.
    
    See Also
    --------
    fusionlab.nn.utils.forecast_multi_step : Higher-level forecasting utility.
    fusionlab.metrics.coverage_score : For evaluating quantile forecast intervals.
    
    Examples
    --------
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> from fusionlab.nn.utils import format_predictions_to_dataframe
    
    >>> B, H, O = 4, 3, 1 # Batch, Horizon, OutputDim
    >>> Q = [0.1, 0.5, 0.9]
    >>> preds_point = tf.random.normal((B, H, O))
    >>> preds_quant = tf.random.normal((B, H, len(Q))) # For O=1
    >>> y_true = tf.random.normal((B, H, O))
    
    >>> # Point forecast
    >>> df_point = format_predictions_to_dataframe(
    ...     predictions=preds_point, y_true_sequences=y_true,
    ...     target_name="value", forecast_horizon=H, output_dim=O
    ... )
    >>> print(df_point.head(H)) # Show first sample's horizon
       sample_idx  forecast_step  value_pred  value_actual
    0           0              1   -0.576731     -0.647362
    1           0              2    0.183931      1.198977
    2           0              3   -0.766871      0.534040
    
    
    >>> # Quantile forecast
    >>> df_quant = format_predictions_to_dataframe(
    ...     predictions=preds_quant, y_true_sequences=y_true,
    ...     target_name="value", quantiles=Q,
    ...     forecast_horizon=H, output_dim=O
    ... )
    >>> print(df_quant.head(H))
       sample_idx  forecast_step  value_q10  value_q50  value_q90  value_actual
    0           0              1  -0.209947   0.263107  -0.308929     -0.647362
    1           0              2   0.303091   0.594701  -0.225007      1.198977
    2           0              3   0.136699  -1.237739   0.002834      0.534040
    
    >>> # With spatial data (NumPy array)
    >>> spatial_np = np.array([[101, 201], [102, 202], [103, 203], [104, 204]])
    >>> df_spatial = format_predictions_to_dataframe(
    ...     predictions=preds_point,
    ...     spatial_data_array=spatial_np,
    ...     spatial_cols=['store_id', 'region_id'],
    ...     spatial_cols_indices=[0, 1]
    ... )
    >>> print(df_spatial[['sample_idx', 'forecast_step', 'store_id']].head(H))
       sample_idx  forecast_step  store_id
    0           0              1     101.0
    1           0              2     101.0
    2           0              3     101.0
    """

    from ..metrics import coverage_score
    
    vlog("Starting prediction formatting to DataFrame.",
         level=3, verbose=verbose, logger = _logger)

    # --- 1. Validate and Obtain Predictions ---
    if predictions is None:
        if model is None or inputs is None:
            raise ValueError(
                "If 'predictions' is None, both 'model' and "
                "'inputs' must be provided to generate predictions."
            )
        vlog("  Predictions not provided, generating from model...",
             level=4, verbose=verbose, logger = _logger)
        try:
            # Ensure inputs is a list for Keras predict
            # The prepare_inputs util should have done this.
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            predictions_raw = model.predict(inputs, verbose=0)
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate predictions from model: {e}"
            ) from e
    else:
        predictions_raw = predictions

    # Convert predictions to NumPy array if it's a TensorFlow Tensor
    if hasattr(predictions_raw, 'numpy'):
        predictions_np = predictions_raw.numpy()
    elif isinstance(predictions_raw, np.ndarray):
        predictions_np = predictions_raw
    else:
        try:
            predictions_np = np.array(predictions_raw)
        except Exception as e:
            raise TypeError(
                f"Could not convert 'predictions' to NumPy array. "
                f"Type: {type(predictions_raw)}. Error: {e}"
            ) from e

    if predictions_np.ndim not in [2, 3, 4]: # (S,F), (S,H,F), (S,H,Q,F)
        raise ValueError(
            f"Predictions must be 2D, 3D or 4D. "
            f"Got shape: {predictions_np.shape}"
        )
    vlog(f"  Raw predictions shape: {predictions_np.shape}",
         level=5, verbose=verbose, logger=_logger)

    # --- 2. Infer Shapes and Parameters ---
    num_samples = predictions_np.shape[0]

    # Infer Forecast Horizon (H)
    if forecast_horizon is not None:
        H = forecast_horizon
    elif predictions_np.ndim >= 2: # Must be at least (S,H) or (S,F)
        H = predictions_np.shape[1]
    else: # Should not happen due to ndim check
        raise ValueError("Cannot infer forecast_horizon from predictions.")

    # Infer Output Dimension (O) and Number of Quantiles (Q)
    num_quantiles_actual = 1
    if quantiles:
        # Sort and validate quantiles for consistency
        quantiles = sorted([float(q) for q in quantiles])
        if not all(0 < q < 1 for q in quantiles):
            raise ValueError("Quantiles must be between 0 and 1.")
        num_quantiles_actual = len(quantiles)

    if output_dim is not None:
        O = output_dim
    elif predictions_np.ndim == 4: # (S, H, Q, O)
        O = predictions_np.shape[3]
    elif predictions_np.ndim == 3: # (S, H, F_out)
        # F_out could be Q*O or O
        if quantiles:
            if predictions_np.shape[2] % num_quantiles_actual == 0:
                O = predictions_np.shape[2] // num_quantiles_actual
            else:
                raise ValueError(
                    f"Prediction's last dim ({predictions_np.shape[2]}) "
                    f"not divisible by num_quantiles ({num_quantiles_actual})."
                    " Cannot infer output_dim."
                )
        else: # Point forecast
            O = predictions_np.shape[2]
    elif predictions_np.ndim == 2: # (S, F_out) -> H=1 assumed by Keras
        # This case implies H=1 was squeezed by model or is (Samples, Features)
        # If H was > 1, this indicates a shape mismatch.
        # For now, assume if 2D, H=1 was intended for this structure.
        if H > 1 and num_samples * H == predictions_np.shape[0]:
             # Predictions might already be flattened (N*H, F_out)
             # Reshape to (N, H, F_out)
            try:
                predictions_np = predictions_np.reshape(
                    (num_samples, H, -1)) # Infer last dim
                O = predictions_np.shape[2] // num_quantiles_actual \
                    if quantiles else predictions_np.shape[2]
            except ValueError as e:
                 raise ValueError(
                    f"Cannot reshape 2D predictions of shape "
                    f"{predictions_np.shape} to 3D using H={H}. {e}"
                 ) from e
        elif quantiles:
            if predictions_np.shape[1] % num_quantiles_actual == 0:
                O = predictions_np.shape[1] // num_quantiles_actual
            else:
                raise ValueError(
                    f"2D Prediction's last dim ({predictions_np.shape[1]}) "
                    f"not divisible by num_quantiles ({num_quantiles_actual})."
                )
        else: # Point forecast, 2D
            O = predictions_np.shape[1]
        # If predictions were 2D (N, F_out), reshape to (N, 1, F_out) if H=1
        if H == 1 and predictions_np.ndim == 2:
            predictions_np = predictions_np.reshape((num_samples, 1, -1))

    if predictions_np.ndim == 3 and quantiles and O > 1 and \
       predictions_np.shape[-1] == num_quantiles_actual * O:
        # Reshape (B, H, Q*O) to (B, H, Q, O) for easier processing
        try:
            predictions_np = predictions_np.reshape(
                num_samples, H, num_quantiles_actual, O
            )
            vlog(f"  Reshaped predictions for multi-output quantiles to "
                 f"{predictions_np.shape}", level=5, verbose=verbose, 
                 logger=_logger )
        except ValueError as e:
            raise ValueError(
                f"Could not reshape predictions from (B,H,Q*O) to "
                f"(B,H,Q,O). Shape: {predictions_np.shape}, Q={num_quantiles_actual}, "
                f"O={O}. Error: {e}"
            ) from e

    vlog(f"  Inferred/Validated: Samples={num_samples}, Horizon={H}, "
         f"OutputDim={O}, NumQuantiles={num_quantiles_actual}",
         level=4, verbose=verbose, logger=_logger)

    # --- 3. Prepare Base DataFrame (Long Format) ---
    # Index for each sample and each forecast step
    sample_indices = np.repeat(np.arange(num_samples), H)
    forecast_steps = np.tile(np.arange(1, H + 1), num_samples)
    df_list = [
        pd.DataFrame({
            'sample_idx': sample_indices,
            'forecast_step': forecast_steps
        })
    ]

    # --- 4. Add Spatial Columns ---
    if spatial_data_array is not None:
        if spatial_cols is None and spatial_cols_indices is None:
            warnings.warn(
                "spatial_data_array provided, but neither "
                "spatial_cols (for DataFrame) nor "
                "spatial_cols_indices (for NumPy array) were given. "
                "Skipping spatial columns.", UserWarning
            )
        elif isinstance(spatial_data_array, (pd.DataFrame, pd.Series)):
            if spatial_cols is None:
                warnings.warn(
                    "spatial_data_array is DataFrame/Series but "
                    "spatial_cols is missing. Skipping.", UserWarning
                    )
            else:
                cols = [spatial_cols] if \
                    isinstance(spatial_cols, str) \
                    else list(spatial_cols)
                try:
                    spatial_df_part = pd.DataFrame(
                        spatial_data_array[cols].iloc[sample_indices].values,
                        columns=cols
                        )
                    df_list.append(spatial_df_part)
                    vlog(f"  Added spatial columns: {cols}",
                         level=4, verbose=verbose, logger=_logger)
                except KeyError as e:
                    warnings.warn(f"Spatial columns not found in "
                                  f"spatial_data_array: {e}. Skipping.")
        elif isinstance(spatial_data_array, (np.ndarray, Tensor)):
            if spatial_cols_indices is None:
                warnings.warn(
                    "spatial_data_array is NumPy/Tensor but "
                    "spatial_cols_indices is missing. Skipping.", UserWarning
                    )
            else:
                s_arr = spatial_data_array.numpy() \
                    if hasattr(spatial_data_array, 'numpy') \
                    else np.array(spatial_data_array)
                indices = [spatial_cols_indices] if \
                    isinstance(spatial_cols_indices, int) \
                    else list(spatial_cols_indices)
                # Ensure names are provided if indices are used
                names = spatial_cols if spatial_cols else \
                    [f"spatial_{k}" for k in range(len(indices))]
                if len(names) != len(indices):
                    warnings.warn(
                        "Length of spatial_cols does not match "
                        "spatial_cols_indices. Using default names."
                        )
                    names = [f"spatial_{k}" for k in range(len(indices))]
                try:
                    spatial_df_part = pd.DataFrame(
                        s_arr[sample_indices][:, indices], columns=names
                        )
                    df_list.append(spatial_df_part)
                    vlog(f"  Added spatial columns from array: {names}",
                         level=4, verbose=verbose, logger=_logger)
                except IndexError as e:
                    warnings.warn(
                        f"Spatial indices out of bounds for "
                        f"spatial_data_array: {e}. Skipping."
                        )
        else:
            warnings.warn(
                f"Unsupported type for spatial_data_array: "
                f"{type(spatial_data_array)}. Skipping."
                )

    # --- 5. Reshape and Add Predictions ---
    # Reshape predictions to (num_samples * H, FeaturesOut)
    # FeaturesOut is O for point, Q*O or (Q,O) for quantile
    pred_reshaped_for_df = predictions_np.reshape(num_samples * H, -1)
    pred_cols_names = []

    base_target_name = target_name if target_name else "pred"

    if quantiles: # Quantile forecast
        if O == 1: # Univariate target, multivariate quantiles
            # pred_reshaped_for_df shape: (N*H, Q)
            for i, q_val in enumerate(quantiles):
                col_name = f"{base_target_name}_q{int(q_val*100)}"
                pred_cols_names.append(col_name)
            pred_df_part = pd.DataFrame(
                pred_reshaped_for_df, columns=pred_cols_names
                )
        else: # Multi-output quantiles
            # predictions_np original shape (N, H, Q, O)
            # pred_reshaped_for_df should be (N*H, Q*O) if not further reshaped
            # Or, if predictions_np was (N,H,Q,O), then reshape to (N*H, Q, O)
            # then iterate through O and Q
            if predictions_np.ndim == 4: # (N, H, Q, O)
                temp_pred_df_parts = []
                for o_idx in range(O):
                    for q_idx, q_val in enumerate(quantiles):
                        col_name = (f"{base_target_name}_{o_idx}_"
                                    f"q{int(q_val*100)}")
                        pred_cols_names.append(col_name)
                        # Extract (N*H, 1) for this target_dim & quantile
                        series_data = predictions_np[:, :, q_idx, o_idx].reshape(-1, 1)
                        temp_pred_df_parts.append(pd.DataFrame(
                            series_data, columns=[col_name]))
                pred_df_part = pd.concat(temp_pred_df_parts, axis=1)
            else: # Should be (N*H, Q*O)
                # This case requires careful unstacking based on Q and O
                # Assuming Q is the faster moving index in the last dim
                for o_idx in range(O):
                    for q_idx, q_val in enumerate(quantiles):
                        col_name = (f"{base_target_name}_{o_idx}_"
                                    f"q{int(q_val*100)}")
                        pred_cols_names.append(col_name)
                pred_df_part = pd.DataFrame(
                    pred_reshaped_for_df, columns=pred_cols_names
                    )
    else: # Point forecast
        if O == 1:
            col_name = f"{base_target_name}_pred"
            pred_cols_names.append(col_name)
        else: # Multi-output point
            for o_idx in range(O):
                col_name = f"{base_target_name}_{o_idx}_pred"
                pred_cols_names.append(col_name)
        pred_df_part = pd.DataFrame(
            pred_reshaped_for_df, columns=pred_cols_names
            )
    df_list.append(pred_df_part)
    vlog(f"  Added prediction columns: {pred_cols_names}",
         level=4, verbose=verbose, logger=_logger)

    # --- 6. Inverse Transform Predictions (if scaler provided) ---
    
    # XXX TODO: 
        
    # This step should happen *after* predictions are in the DataFrame
    # to allow easy column selection for multi-output scalers.
    # For simplicity now, assume inverse transform happens on `predictions_np`
    # before adding to DataFrame if scaler is complex.
    # If scaler is simple (for target only), it can be done on the column.

    # --- 7. Add Actual Values (if y_true_sequences provided) ---
    actual_cols_names = []
    if y_true_sequences is not None:
        if hasattr(y_true_sequences, 'numpy'):
            y_true_np = y_true_sequences.numpy()
        else:
            y_true_np = np.array(y_true_sequences)

        if y_true_np.shape[0] != num_samples or y_true_np.shape[1] != H:
            warnings.warn(
                f"Shape of y_true_sequences {y_true_np.shape} "
                f"is incompatible with predictions "
                f"({num_samples}, {H}, ...). Skipping actuals.", UserWarning
            )
        else:
            # y_true_np shape: (N, H, O)
            y_true_reshaped = y_true_np.reshape(num_samples * H, O)
            if O == 1:
                col_name = f"{base_target_name}_actual"
                actual_cols_names.append(col_name)
            else:
                for o_idx in range(O):
                    col_name = f"{base_target_name}_{o_idx}_actual"
                    actual_cols_names.append(col_name)
            actual_df_part = pd.DataFrame(
                y_true_reshaped, columns=actual_cols_names
                )
            df_list.append(actual_df_part)
            vlog(f"  Added actual value columns: {actual_cols_names}",
                 level=4, verbose=verbose, logger=_logger)

    # --- Concatenate all parts ---
    final_df = pd.concat(df_list, axis=1)

    # --- 8. Inverse Transform (if scaler provided) ---
    # Now apply inverse transform on the DataFrame columns
    if scaler is not None:
        if scaler_feature_names is None or target_idx_in_scaler is None:
            warnings.warn(
                "Scaler provided, but `scaler_feature_names` or "
                "`target_idx_in_scaler` is missing. Cannot perform "
                "targeted inverse transform. Attempting on all pred/actual cols.",
                UserWarning
            )
            # Try to inverse transform all prediction and actual columns
            # This assumes they were all scaled together, which might be wrong.
            cols_to_inv = pred_cols_names + actual_cols_names
            for col in cols_to_inv:
                if col in final_df:
                    try:
                        # This simple inverse assumes scaler was fit on single col
                        final_df[col] = scaler.inverse_transform(
                            final_df[[col]]
                            )
                    except Exception as e:
                        warnings.warn(f"Failed to simple inverse transform "
                                      f"column '{col}': {e}")
        else: # Targeted inverse transform
            vlog("  Applying inverse transformation using scaler...",
                 level=4, verbose=verbose, logger=_logger)
            # Create dummy array for inverse transform
            dummy_array_shape = (len(final_df), len(scaler_feature_names))

            for col_group_idx in range(O): # For each output dimension
                # Inverse transform predictions
                if quantiles:
                    for q_idx, q_val in enumerate(quantiles):
                        pred_col_name = f"{base_target_name}"
                        if O > 1: pred_col_name += f"_{col_group_idx}"
                        pred_col_name += f"_q{int(q_val*100)}"

                        if pred_col_name in final_df:
                            dummy = np.zeros(dummy_array_shape)
                            dummy[:, target_idx_in_scaler] = final_df[pred_col_name]
                            final_df[pred_col_name] = scaler.inverse_transform(
                                dummy)[:, target_idx_in_scaler]
                else: # Point forecast
                    pred_col_name = f"{base_target_name}"
                    if O > 1: pred_col_name += f"_{col_group_idx}"
                    pred_col_name += "_pred"
                    if pred_col_name in final_df:
                        dummy = np.zeros(dummy_array_shape)
                        dummy[:, target_idx_in_scaler] = final_df[pred_col_name]
                        final_df[pred_col_name] = scaler.inverse_transform(
                            dummy)[:, target_idx_in_scaler]

                # Inverse transform actuals
                actual_col_name = f"{base_target_name}"
                if O > 1: actual_col_name += f"_{col_group_idx}"
                actual_col_name += "_actual"
                if actual_col_name in final_df:
                    dummy = np.zeros(dummy_array_shape)
                    dummy[:, target_idx_in_scaler] = final_df[actual_col_name]
                    final_df[actual_col_name] = scaler.inverse_transform(
                        dummy)[:, target_idx_in_scaler]
            vlog("    Inverse transformation applied.", level=5, 
                 verbose=verbose, logger=_logger)

    # --- 9. Evaluate Coverage (if requested for quantiles) ---
    if ( 
            evaluate_coverage 
            and quantiles and 
            y_true_sequences is not None
        ):
        if len(quantiles) < 2:
            warnings.warn(
                "Coverage score requires at least two quantiles "
                "(lower and upper bound). Skipping evaluation.", UserWarning
            )
        elif O > 1:
            warnings.warn(
                "Coverage score evaluation for multi-output (>1 output_dim) "
                "quantile forecasts is not directly implemented in this "
                "example. Skipping.", UserWarning
            )
        else: 
            #  XXX TODO:  Univariate quantile forecast
            # Assume quantiles are sorted, first is lower, last is upper
            # for a symmetric interval around median (if present).
            # More robustly, user should specify which quantiles form the interval.
            # For this example, assume first and last define the interval.
            lower_q_col = f"{base_target_name}_q{int(quantiles[0]*100)}"
            upper_q_col = f"{base_target_name}_q{int(quantiles[-1]*100)}"
            actual_col = f"{base_target_name}_actual"

            if lower_q_col in final_df and upper_q_col in final_df \
                    and actual_col in final_df:
                coverage = coverage_score(
                    final_df[actual_col],
                    final_df[lower_q_col],
                    final_df[upper_q_col]
                )
                # Add as a global attribute or print
                # For DataFrame, it's harder to add a single score. Print for now.
                vlog(f"  Coverage Score ({quantiles[0]}-{quantiles[-1]}): "
                     f"{coverage:.4f}", level=3, verbose=verbose, 
                     logger=_logger)
                # Could add to df.attrs if desired: final_df.attrs['coverage_score'] = coverage
            else:
                warnings.warn(
                    "Required columns for coverage score not found in DataFrame. "
                    "Skipping evaluation.", UserWarning
                )
    elif evaluate_coverage:
         warnings.warn("`fusionlab.metrics.coverage_score` not found. "
                       "Skipping coverage evaluation.", ImportWarning)

    vlog("Prediction formatting to DataFrame complete.",
         level=3, verbose=verbose, logger=_logger)
    
    return final_df


@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)   
@validate_params ({
    'forecast_horizon': [Interval(Integral, 0, None, closed='left'), None], 
    'model_type': [StrOptions({'strict', 'flexible'})]
    })
def prepare_model_inputs(
    dynamic_input: Union[np.ndarray, Tensor],
    static_input: Optional[Union[np.ndarray, Tensor]] = None,
    future_input: Optional[Union[np.ndarray, Tensor]] = None,
    model_type: str = 'strict',
    forecast_horizon: Optional[int] = None,
    verbose: int = 0,
    **kwargs 
) -> List[Optional[Tensor]]:
    """Prepares a list of input tensors for a model's call method.

    This function standardizes the creation of the input list
    `[static, dynamic, future]` expected by many models in
    ``fusionlab``. It handles cases where static or future inputs
    might be ``None``, creating appropriate dummy tensors with zero
    features if the `model_type` is 'strict'.

    Parameters
    ----------
    dynamic_input : np.ndarray or tf.Tensor
        The dynamic (past observed) features. This input is
        always required and must be a valid tensor or array.
        Expected shape: `(batch_size, past_time_steps, num_dynamic_features)`.
    static_input : np.ndarray or tf.Tensor, optional
        The static (time-invariant) features.
        Expected shape: `(batch_size, num_static_features)`.
        If ``None`` and `model_type` is 'strict', a dummy tensor
        with 0 static features will be created. Default is ``None``.
    future_input : np.ndarray or tf.Tensor, optional
        The known future features.
        Expected shape: `(batch_size, future_time_span, num_future_features)`.
        If ``None`` and `model_type` is 'strict', a dummy tensor
        with 0 future features will be created. The time span for
        this dummy future tensor will be `past_time_steps` (from
        `dynamic_input`) plus `forecast_horizon` if provided,
        otherwise just `past_time_steps`. Default is ``None``.
    model_type : {'strict', 'flexible'}, default 'strict'
        Determines how ``None`` inputs for static and future features
        are handled:
        - ``'strict'``: If `static_input` or `future_input` is
          ``None``, a dummy tensor with a feature dimension of 0
          will be created and included in the output list. This is
          for models that expect a 3-element list of tensors, even
          if some paths are unused.
        - ``'flexible'``: If `static_input` or `future_input` is
          ``None``, ``None`` itself will be placed in the
          corresponding position in the output list. This is for
          models that can internally handle ``None`` inputs for
          optional feature types.
    forecast_horizon : int, optional
        The forecast horizon. Used only if `model_type='strict'`
        and `future_input` is ``None``, to determine the time
        dimension of the dummy future tensor
        (as `past_time_steps + forecast_horizon`). If not provided
        in this scenario, the dummy future tensor's time dimension
        will match `dynamic_input`'s `past_time_steps`.
        Default is ``None``.
    verbose : int, default 0
        Verbosity level. If > 0, prints information about dummy
        tensor creation.
        - ``0``: Silent.
        - ``1``: Basic info on dummy creation.
        - ``2``: More details on shapes.

    Returns
    -------
    List[Optional[tf.Tensor]]
        A list containing three elements in the order:
        `[processed_static_input, processed_dynamic_input, processed_future_input]`.
        Elements can be TensorFlow tensors or ``None`` (if
        `model_type='flexible'` and original input was ``None``).
        All returned tensors are cast to `tf.float32`.

    Raises
    ------
    ValueError
        If `dynamic_input` is ``None``.
        If `dynamic_input` is not at least 2D (needs batch dimension).
        If `static_input` (when provided) is not 2D.
        If `future_input` (when provided) is not 3D.
    TypeError
        If inputs cannot be converted to TensorFlow tensors.

    Examples
    --------
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> from fusionlab.nn.utils import prepare_model_inputs 
    >>> B, T, H = 2, 10, 3
    >>> D_s, D_d, D_f = 2, 4, 1
    >>> dyn_in = tf.random.normal((B, T, D_d))
    >>> stat_in = tf.random.normal((B, D_s))
    >>> fut_in = tf.random.normal((B, T + H, D_f))

    >>> # Strict mode, all inputs provided
    >>> s, d, f = prepare_model_inputs(dyn_in, stat_in, fut_in, model_type='strict')
    >>> print(f"S: {s.shape}, D: {d.shape}, F: {f.shape}")
    S: (2, 2), D: (2, 10, 4), F: (2, 13, 1)

    >>> # Strict mode, static is None
    >>> s, d, f = prepare_model_inputs(dyn_in, static_input=None, future_input=fut_in,
    ...                                model_type='strict', forecast_horizon=H)
    >>> print(f"S: {s.shape}, D: {d.shape}, F: {f.shape}")
    S: (2, 0), D: (2, 10, 4), F: (2, 13, 1)

    >>> # Flexible mode, static and future are None
    >>> s, d, f = prepare_model_inputs(dyn_in, static_input=None, future_input=None,
    ...                                model_type='flexible')
    >>> print(f"S: {s is None}, D: {d.shape}, F: {f is None}")
    S: True, D: (2, 10, 4), F: True
    """
    
    # --- Input Validation and Type Conversion ---
    if dynamic_input is None:
        raise ValueError("`dynamic_input` is required and cannot be None.")

    def _to_tensor_float32(
        data: Optional[Union[np.ndarray, Tensor]], name: str
        ) -> Optional[Tensor]:
        # Helper to convert to tf.Tensor and tf.float32 if not None.
        if data is None:
            return None
        try:
            tensor_data = tf_convert_to_tensor(data)
            if tensor_data.dtype != tf_float32:
                tensor_data = tf_cast(tensor_data, tf_float32)
            return tensor_data
        except Exception as e:
            raise TypeError(
                f"Failed to convert '{name}' to TensorFlow tensor "
                f"of type float32. Error: {e}"
            ) from e

    processed_dynamic_input = _to_tensor_float32(
        dynamic_input, "dynamic_input"
        )
    processed_static_input = _to_tensor_float32(
        static_input, "static_input"
        )
    processed_future_input = _to_tensor_float32(
        future_input, "future_input"
        )

    # --- Shape Checks for Provided Inputs ---
    # Dynamic input must be at least 2D (Batch, Features) or 3D (B, T, F)
    if tf_rank(processed_dynamic_input) < 2:
        raise ValueError(
            f"dynamic_input must be at least 2D. "
            f"Got rank {tf_rank(processed_dynamic_input)}."
            )
    # If 2D, assume it's (Batch, Features) and needs expansion for time
    if tf_rank(processed_dynamic_input) == 2:
        # This case implies dynamic input is treated as static-like
        # or time dimension is 1. For TFTs, dynamic is usually 3D.
        # This utility assumes dynamic_input is already time-distributed.
        warnings.warn(
            "dynamic_input was 2D. For sequence models, it's "
            "typically expected to be 3D (Batch, TimeSteps, Features). "
            "Proceeding, but ensure this is intended.", UserWarning
        )

    if processed_static_input is not None and \
       tf_rank(processed_static_input) != 2:
        raise ValueError(
            f"static_input, if provided, must be 2D (Batch, Features)."
            f" Got rank {tf_rank(processed_static_input)}."
            )

    if processed_future_input is not None and \
       tf_rank(processed_future_input) != 3:
        raise ValueError(
            f"future_input, if provided, must be 3D "
            f"(Batch, TimeSteps, Features). "
            f"Got rank {tf_rank(processed_future_input)}."
            )

    # --- Determine Batch Size and Time Steps from Dynamic Input ---
    # These are used for creating dummy tensors if needed.
    # Use tf.shape for graph compatibility.
    shape_dynamic = tf_shape(processed_dynamic_input)
    batch_size = shape_dynamic[0]
    # Dynamic input might be 2D (B, D_dyn) or 3D (B, T_dyn, D_dyn)
    # Default to 1 time step if dynamic input is 2D
    past_time_steps = shape_dynamic[1] if \
        tf_rank(processed_dynamic_input) == 3 else tf_constant( 
            1, dtype=tf_int32)
    

    if verbose >= 2:
        print(f"  prepare_model_inputs: Derived batch_size={batch_size}, "
              f"past_time_steps={past_time_steps}")

    # --- Handle based on model_type ---
    s_to_pass: Optional[Tensor] = processed_static_input
    d_to_pass: Tensor = processed_dynamic_input # Always present
    f_to_pass: Optional[Tensor] = processed_future_input

    if model_type == 'strict':
        if s_to_pass is None:
            # Create dummy 2D static tensor: (Batch, 0 Features)
            # s_to_pass = tf_zeros((batch_size, 0), dtype=tf_float32)
            s_to_pass = tf_zeros(
                    (tf_cast(batch_size, tf_int32), # Ensure int32
                     tf_constant(0, dtype=tf_int32)),     # Num features
                    dtype=tf_float32
                )
            
            if verbose >= 1:
                print(f"  prepare_model_inputs (strict): Created dummy "
                      f"static input with shape {s_to_pass.shape}")
        elif tf_rank(s_to_pass) == 2 and tf_shape(s_to_pass)[-1] == 0:
             if verbose >= 1:
                print(f"  prepare_model_inputs (strict): Static input "
                      f"provided with 0 features, shape {s_to_pass.shape}")


        if f_to_pass is None:
            # Determine time dimension for dummy future tensor
            num_future_timesteps = tf_cast(past_time_steps, tf_int32)
            if forecast_horizon is not None and forecast_horizon > 0:
                # Ensure forecast_horizon is a tensor for addition
                fh_tensor = tf_cast(forecast_horizon, tf_int32)
                num_future_timesteps += fh_tensor
            # Create dummy 3D future tensor: (Batch, Time, 0 Features)
            f_to_pass = tf_zeros(
                (tf_cast(batch_size, tf_int32),      
                 tf_cast(num_future_timesteps, tf_int32), 
                 tf_constant(0, dtype=tf_int32)),             
                dtype=tf_float32
                )
            
            if verbose >= 1:
                print(f"  prepare_model_inputs (strict): Created dummy "
                      f"future input with shape {f_to_pass.shape}")
        elif tf_rank(f_to_pass) == 3 and tf_shape(f_to_pass)[-1] == 0:
            if verbose >= 1:
                print(f"  prepare_model_inputs (strict): Future input "
                      f"provided with 0 features, shape {f_to_pass.shape}")

    elif model_type == 'flexible':
        # For flexible models, None inputs are passed through as None.
        # The processed_static_input, etc., are already correct.
        if verbose >= 1:
            print(f"  prepare_model_inputs (flexible): Passing inputs "
                  f"as is (Static: {type(s_to_pass)}, "
                  f"Dynamic: {type(d_to_pass)}, Future: {type(f_to_pass)})")
    else:
        # This wont reach since @validate_parameters handle it. 
        raise ValueError(
            f"Invalid `model_type`: '{model_type}'. "
            "Must be 'strict' or 'flexible'."
            )
    # Return in the standard order [Static, Dynamic, Future]
    return [s_to_pass, d_to_pass, f_to_pass]


@Appender (dedent( 
    prepare_model_inputs.__doc__.replace (
        'prepare_model_inputs', 'prepare_model_inputs_in'),), 
    join='\n', 
)
@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)   
@validate_params ({
    'forecast_horizon': [Interval(Integral, 0, None, closed='left'), None], 
    'model_type': [StrOptions({'strict', 'flexible'})]
    })
# @optional_tf_function 
def prepare_model_inputs_in(
    dynamic_input: Union[np.ndarray, Tensor],
    static_input: Optional[Union[np.ndarray, Tensor]] = None,
    future_input: Optional[Union[np.ndarray, Tensor]] = None,
    model_type: str = 'strict',
    forecast_horizon: Optional[int] = None,
    verbose: int = 0,
    # **kwargs # Removed if not used
) -> List[Optional[Tensor]]:
    """Prepares a list of input tensors for a model's call method in graph 
    compatible mode.
    """
    
    if not KERAS_BACKEND:
        raise RuntimeError("prepare_model_inputs requires a Keras backend.")

    if dynamic_input is None:
        raise ValueError(
            "`dynamic_input` is required and cannot be None."
            )

    def _to_tensor_float32(
        data: Optional[Union[np.ndarray, Tensor]], name: str
        ) -> Optional[Tensor]:
        if data is None:
            return None
        try:
            # Ensure it's a tensor first
            tensor_data = tf_convert_to_tensor(data, dtype=tf_float32)
            # Explicitly cast if it's not float32 
            # (though convert_to_tensor with dtype should handle it)
            if tensor_data.dtype != tf_float32:
                 tensor_data = tf_cast(tensor_data, tf_float32)
            return tensor_data
        except Exception as e:
            raise TypeError(
                f"Failed to convert '{name}' to TensorFlow tensor "
                f"of type float32. Error: {e}"
            ) from e

    processed_dynamic_input = _to_tensor_float32(
        dynamic_input, "dynamic_input"
        )
    processed_static_input = _to_tensor_float32(
        static_input, "static_input"
        )
    processed_future_input = _to_tensor_float32(
        future_input, "future_input"
        )

    # --- Shape Assertions (Graph-Compatible) ---
    rank_dynamic = tf_rank(processed_dynamic_input)
    # Use tf.control_dependencies for assertions if they don't return the tensor
    # However, tf.Assert ops are executed when the tensor they depend on is computed.
    # Forcing dependency by including the tensor itself in the assertion data.
    assert_op_dyn_rank = tf_Assert(
        tf_greater_equal(rank_dynamic, tf_constant(2, dtype=rank_dynamic.dtype)),
        ["dynamic_input must be at least 2D. Got rank:", rank_dynamic, 
         "Shape:", tf_shape(processed_dynamic_input)]
    )
    # Make sure assertion is executed
    with tf_control_dependencies([assert_op_dyn_rank]):
        processed_dynamic_input = tf_identity(processed_dynamic_input)

    # if tf_rank(processed_dynamic_input) == 2 and verbose > 0:
    #     logger.warning(
    #         "dynamic_input was 2D. For sequence models, it's "
    #         "typically expected to be 3D (Batch, TimeSteps, Features). "
    #         "Proceeding, but ensure this is intended."
    #     )
    
    # static: None or an int
    static_rank = processed_dynamic_input.shape.rank
    
    if static_rank == 2 and verbose > 0:
        logger.warning(
            "dynamic_input was 2D. For sequence models, it's "
            "typically expected to be 3D (Batch, TimeSteps, Features). "
            "Proceeding, but ensure this is intended."
        )
    
    if processed_static_input is not None:
        rank_static = tf_rank(processed_static_input)
        assert_op_stat_rank = tf_Assert(
            tf_equal(rank_static, tf_constant(2, dtype=rank_static.dtype)),
            ["static_input, if provided, must be 2D. Got rank:", rank_static,
             "Shape:", tf_shape(processed_static_input)]
        )
        with tf_control_dependencies([assert_op_stat_rank]):
            processed_static_input = tf_identity(processed_static_input)
            
    if processed_future_input is not None:
        rank_future = tf_rank(processed_future_input)
        assert_op_fut_rank = tf_Assert(
            tf_equal(rank_future, tf_constant(3, dtype=rank_future.dtype)),
            ["future_input, if provided, must be 3D. Got rank:", rank_future,
             "Shape:", tf_shape(processed_future_input)]
        )
        with tf_control_dependencies([assert_op_fut_rank]):
            processed_future_input = tf_identity(processed_future_input)


    # XXX TO OPTIMIZE with tf.function : Graph compatible 
    # ------------------ START: GRAPH-COMPATIBLE OPTIMIZATION --------------------
    
    d_to_pass: Tensor = processed_dynamic_input
    s_to_pass: Optional[Tensor] = processed_static_input
    f_to_pass: Optional[Tensor] = processed_future_input

    # Use tf.cond for conditional logic on tensors
    past_time_steps = tf_cond(
        tf_equal(tf_rank(d_to_pass), 3),
        lambda: tf_shape(d_to_pass)[1],
        lambda: tf_constant(1, dtype=tf_int32)
    )

    if verbose >= 2:
        # tf.print is the graph-compatible way to print tensor values
        tf_print(
            "prepare_model_inputs: Derived batch_size=", tf_shape(d_to_pass)[0],
            " past_time_steps=", past_time_steps
        )

    # Logic for handling 'strict' mode where all inputs must be tensors
    if model_type == 'strict':
        
        # --- Handle Static Input ---
        if s_to_pass is None:
            # Create a dummy static tensor if one is not provided
            # FIX: Construct the shape as a single tensor using tf.stack
            batch_size = tf_shape(d_to_pass)[0]
            # shape_tensor = tf_stack([batch_size, 0]) # Creates a tensor like [B, 0]
            # s_to_pass = tf_zeros(shape_tensor, dtype=tf_float32)
           
            # batch_size = tf_shape(d_to_pass)[0]
            # s_to_pass = tf_zeros((batch_size, 0), dtype=tf_float32)
            s_to_pass = tf_zeros(
                        (batch_size, tf_constant(0, dtype=tf_int32)),
                        dtype=tf_float32
                    )
            if verbose >= 1:
                tf_print(
                    "prepare_model_inputs (strict): Created dummy"
                    " static input with shape", tf_shape(s_to_pass))
        
        elif verbose >=1:
             # Use tf.cond to conditionally 
             # print a message if the input has 0 features
            tf_cond(
                tf_equal(tf_shape(s_to_pass)[-1], 0),
                lambda: tf_print(
                    "prepare_model_inputs (strict): Static input"
                    " provided with 0 features, shape", tf_shape(s_to_pass)),
                lambda: tf_no_op()
            )

        # --- Handle Future Input ---
        if f_to_pass is None:
            # batch_size = tf_shape(d_to_pass)[0]
            shape_dynamic = tf_shape(d_to_pass)
            batch_size = shape_dynamic[0]
            past_time_steps = tf_cond(
                tf_equal(tf_rank(d_to_pass), 3),
                lambda: tf_shape(d_to_pass)[1],
                lambda: tf_constant(1, dtype=tf_int32)
            )
            
            # Conditionally add forecast_horizon to the sequence length
            num_future_timesteps = past_time_steps
            if forecast_horizon is not None and forecast_horizon > 0:
                num_future_timesteps = past_time_steps + tf_cast(
                    forecast_horizon, dtype=tf_int32)
    
            f_to_pass = tf_zeros(
                (batch_size, num_future_timesteps,  
                 tf_constant(0, dtype=tf_int32)),
                dtype=tf_float32
            )

            if verbose >= 1:
                tf_print(
                    "prepare_model_inputs (strict): Created dummy"
                    " future input with shape", tf_shape(f_to_pass))

        elif verbose >= 1:
            # Conditionally print a message if the input has 0 features
            tf_cond(
                tf_equal(tf_shape(f_to_pass)[-1], 0),
                lambda: tf_print(
                    "prepare_model_inputs (strict): Future input provided"
                    " with 0 features, shape", tf_shape(f_to_pass)),
                lambda: tf_no_op()
            )
            
    # --- Logic for 'flexible' mode ---
    elif model_type == 'flexible' and verbose >= 1:
         tf_print(
             "prepare_model_inputs (flexible): Passing inputs as is."
      )

    # ------------------ END: GRAPH-COMPATIBLE OPTIMIZATION --------------------
    
    return [s_to_pass, d_to_pass, f_to_pass]


@check_params({'domain_func': Optional[Callable]}, coerce=False)
@ParamsValidator( 
    { 
        'y_true': ['array-like:np:transf'], 
        'y_pred': ['array-like:np:transf', None], 
        'method': [StrOptions(
            {'statistical', 'domain', 'isolation_forest','residual', 'stats'})
            ], 
        'threshold': [Interval(Real, 1, None, closed ='left')], 
        'contamination': [Interval(Real, 0, None, closed='left')], 
        'epsilon': [Hidden(Interval(Real, 0, 1 , closed ='neither'))], 
        'estimator':[HasMethods(['fit', 'predict']), None], 
        'random_state': ['random_state', None], 
        'residual_metric': [StrOptions({'mse', 'mae','rmse'})], 
        'objective': [str]
    }
)
def compute_anomaly_scores(
    y_true,
    y_pred=None,
    method='statistical',
    threshold=3.0,
    domain_func=None,
    contamination=0.05,
    epsilon=1e-6,
    estimator=None,
    random_state=None,
    residual_metric='mse', 
    objective='ts',
    verbose=1
):
    r"""
    Compute anomaly scores for given true targets using various methods.
    
    This utility function, ``anomaly_scores``, provides a flexible approach
    to compute anomaly scores outside the XTFT model [1]_. Anomaly scores
    serve as indicators of how unusual certain observations are, guiding
    the model towards more robust and stable forecasts. By detecting and
    quantifying anomalies, practitioners can adjust forecasting strategies,
    improve predictive performance, and handle irregular patterns more
    effectively.
    
    Parameters
    ----------
    y_true : np.ndarray
        The ground truth target values with shape ``(B, H, O)``, where:
        - `B`: batch size
        - `H`: number of forecast horizons (time steps ahead)
        - `O`: output dimension (e.g., number of target variables).
        
        Typically, `y_true` corresponds to the same array passed as the
        forecast target to the model. All computations of anomalies
        are relative to these true values or, if provided, their
        predicted counterparts `y_pred`.
    
    y_pred : np.ndarray, optional
        The predicted values with shape ``(B, H, O)``. If provided and
        the `method` is set to `'residual'`, the anomaly scores are derived
        from the residuals between `y_true` and `y_pred`. In this scenario,
        anomalies reflect discrepancies indicating unusual conditions
        or model underperformance.
    
    method : str, optional
        The method used to compute anomaly scores. Supported options:
        - ``"statistical"`` or ``"stats"``:  
          Uses mean and standard deviation of `y_true` to measure deviation
          from the mean. Points far from the mean by a certain factor
          (controlled by `threshold`) yield higher anomaly scores.
          
          Formally, let :math:`\mu` be the mean of `y_true` and 
          :math:`\sigma` its standard deviation. The anomaly score for 
          a point :math:`y` is:  
          .. math::  
             (\frac{y - \mu}{\sigma + \varepsilon})^2
          
          where :math:`\varepsilon` is a small constant for numerical
          stability.
        
        - ``"domain"``:  
          Uses a domain-specific heuristic (provided by `domain_func`)
          to compute scores. If no `domain_func` is provided, a default
          heuristic marks negative values as anomalies.
    
        - ``"isolation_forest"`` or ``"if"``:  
          Employs the IsolationForest algorithm to detect outliers. 
          The model learns a structure to isolate anomalies more quickly
          than normal points. Higher contamination rates allow more
          points to be considered anomalous.
    
        - ``"residual"``:  
          If `y_pred` is provided, anomalies are derived from residuals:
          the difference `(y_true - y_pred)`. By default, mean squared 
          error (`mse`) is used. Other metrics include `mae` and `rmse`,
          offering flexibility in quantifying deviations:
          .. math::
             \text{MSE: }(y_{true} - y_{pred})^2
    
        Default is ``"statistical"``.
    
    threshold : float, optional
        Threshold factor for the `statistical` method. Defines how far
        beyond mean ± (threshold * std) is considered anomalous. Though
        not directly applied as a mask here, it can guide interpretation
        of scores. Default is ``3.0``.
    
    domain_func : callable, optional
        A user-defined function for `domain` method. It takes `y_true`
        as input and returns an array of anomaly scores with the same
        shape. If none is provided, the default heuristic:
        .. math::
           \text{anomaly}(y) = \begin{cases}
           |y| \times 10 & \text{if } y < 0 \\
           0 & \text{otherwise}
           \end{cases}
    
    contamination : float, optional
        Used in the `isolation_forest` method. Specifies the proportion
        of outliers in the dataset. Default is ``0.05``.
    
    epsilon : float, optional
        A small constant :math:`\varepsilon` for numerical stability
        in calculations, especially during statistical normalization.
        Default is ``1e-6``.
    
    estimator : object, optional
        A pre-fitted IsolationForest estimator for the `isolation_forest`
        method. If not provided, a new estimator will be created and fitted
        to `y_true`.
    
    random_state : int, optional
        Sets a random state for reproducibility in the `isolation_forest`
        method.
    
    residual_metric : str, optional
        The metric used to compute anomalies from residuals if `method`
        is set to `'residual'`. Supported metrics:
        - ``"mse"``: mean squared error per point `(residuals**2)`
        - ``"mae"``: mean absolute error per point `|residuals|`
        - ``"rmse"``: root mean squared error `sqrt((residuals**2))`
        
        Default is ``"mse"``.
    
    objective : str, optional
        Specifies the type of objective, for future extensibility.
        Default is ``"ts"`` indicating time series. Could be extended
        for other tasks in the future.
    
    verbose : int, optional
        Controls verbosity. If `verbose=1`, some messages or warnings
        may be printed. Higher values might produce more detailed logs.
    
    Returns
    -------
    anomaly_scores : np.ndarray
        An array of anomaly scores with the same shape as `y_true`.
        Higher values indicate more unusual or anomalous points.
    
    Notes
    -----
    Choosing an appropriate method depends on the data characteristics,
    domain requirements, and model complexity. Statistical methods
    are quick and interpretable but may oversimplify anomalies. Domain
    heuristics leverage expert knowledge, while isolation forest applies
    a more robust, data-driven approach. Residual-based anomalies help
    assess model performance and highlight periods where the model 
    struggles.
    
    Examples
    --------
    >>> from fusionlab.nn.losses import compute_anomaly_scores
    >>> import numpy as np
    
    >>> # Statistical method example
    >>> y_true = np.random.randn(32, 20, 1)  # (B,H,O)
    >>> scores = compute_anomaly_scores(y_true, method='statistical', threshold=3)
    >>> scores.shape
    (32, 20, 1)
    
    >>> # Domain-specific example
    >>> def my_heuristic(y):
    ...     return np.where(y < -1, np.abs(y)*5, 0.0)
    >>> scores = compute_anomaly_scores(y_true, method='domain',
                                        domain_func=my_heuristic)
    
    >>> # Isolation Forest example
    >>> scores = compute_anomaly_scores(y_true, method='isolation_forest',
                                        contamination=0.1)
    
    >>> # Residual-based example
    >>> y_pred = y_true + np.random.normal(0, 1, y_true.shape)  # Introduce noise
    >>> scores = compute_anomaly_scores(y_true, y_pred=y_pred, method='residual',
                                        residual_metric='mae')
    
    See Also
    --------
    fusionlab.nn.losses.compute_quantile_loss` : 
        For computing quantile losses.
    fusionlab.nn.losses.objective_loss :
        For integrating anomaly scores into a multi-objective loss.
    
    References
    ----------
    .. [1] Wang, X., et al. "Enhanced Temporal Fusion Transformer for Time
           Series Forecasting." International Journal of Forecasting, 37(3),
           2021.
    """
    if objective == 'ts': 
        if y_true.ndim != 3:
            raise ValueError(
                "`y_true` must be a 3-dimensional array with the shape (B, H, O):\n"
                "  - B: Batch size (number of samples per batch).\n"
                "  - H: Number of horizons (time steps ahead).\n"
                "  - O: Output dimension (e.g., number of target variables).\n"
                "Please ensure the input conforms to the specified shape."
            )
    elif y_true.ndim not in [1, 2]:
        raise ValueError(
            "`y_true` must be a 1D or 2D array for non-time-series objectives.\n"
            f"Received shape: {y_true.shape}."
        )
    
    if y_pred is not None:
        check_consistent_length(y_true, y_pred)

    # Normalize method aliases
    method = method.lower()
    method_map = {
        'stats': 'statistical',
        'statistical': 'statistical',
        'if': 'isolation_forest',
        'isolation_forest': 'isolation_forest',
    }
    method = method_map.get(method, method)

    if verbose >= 3:
        print(f"[INFO] Using method: {method}")

    if method == 'statistical':
        mean = y_true.mean()
        std = y_true.std()
        anomaly_scores = ((y_true - mean) / (std + epsilon))**2
        if verbose >= 5:
            print(f"[DEBUG] Mean: {mean}, Std: {std}")

    elif method == 'domain':
        if domain_func is None:
            # Default heuristic: negative values are anomalies
            anomaly_scores = np.where(y_true < 0, np.abs(y_true) * 10, 0.0)
        else:
            anomaly_scores = domain_func(y_true)
            if not isinstance(anomaly_scores, np.ndarray):
                raise ValueError("`domain_func` must return a numpy array.")
            if anomaly_scores.shape != y_true.shape:
                raise ValueError(
                    "`domain_func` output shape must match `y_true` shape.")
        if verbose >= 5:
            print("[DEBUG] Domain-based anomaly scores calculated.")

    elif method == 'isolation_forest':
        flat_data = y_true.reshape(-1, 1)
        if estimator is None:
            iso = IsolationForest(
                contamination=contamination,
                random_state=random_state
            )
            iso.fit(flat_data)
        else:
            iso = estimator

        raw_scores = iso.score_samples(flat_data)
        anomaly_scores = -raw_scores.reshape(y_true.shape)
        if verbose >= 5:
            print("[DEBUG] Isolation Forest raw scores computed.")

    elif method == 'residual':
        if y_pred is None:
            raise ValueError("`y_pred` must be provided if method='residual'.")
        if not isinstance(y_pred, np.ndarray):
            raise ValueError("`y_pred` must be a numpy array.")
        if y_pred.shape != y_true.shape:
            raise ValueError(
                "`y_pred` shape must match `y_true` shape for residual method.")

        residuals = y_true - y_pred
        residual_metric = residual_metric.lower()

        if residual_metric == 'mse':
            # Mean Squared Error per point is just squared residual
            anomaly_scores = residuals**2
        elif residual_metric == 'mae':
            # Mean Absolute Error per point is absolute residual
            anomaly_scores = np.abs(residuals)
        elif residual_metric == 'rmse':
            # Root Mean Squared Error per point: sqrt(residual^2)
            # Essentially absolute value, but we'll respect definition
            anomaly_scores = np.sqrt((residuals**2) + epsilon)
        if verbose >= 5:
            print(f"[DEBUG] Residuals calculated using {residual_metric}.")

    if verbose >= 1:
        print("[INFO] Anomaly scores computed successfully.")
    
    return anomaly_scores


@TypeEnforcer({"0": 'array-like', "1": 'array-like'})
@validate_params({ 
    "sequences": ['array-like'], 
    "static_indices":[Integral], 
    "dynamic_indices": [Integral], 
    "static_time_step": [Interval( Integral, 0, None, closed="left")], 
    "static_reshape_shape": [tuple, None], 
    "dynamic_reshape_shape": [tuple, None],
    }
)
def split_static_dynamic(
    sequences: np.ndarray, 
    static_indices: List[int], 
    dynamic_indices: List[int],
    static_time_step: int = 0,
    reshape_static: bool = True,
    reshape_dynamic: bool = True,
    static_reshape_shape: Optional[Tuple[int, ...]] = None,
    dynamic_reshape_shape: Optional[Tuple[int, ...]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split sequences into static and dynamic inputs for the model.

    The `split_static_dynamic` function divides input sequences into static and 
    dynamic components based on specified feature indices. Static features are 
    typically location-specific and do not change over time, while dynamic features 
    vary across different time steps.

    .. math::
        \text{Static Inputs} = \mathbf{S} = \mathbf{X}_{t, static\_indices} \\
        \text{Dynamic Inputs} = \mathbf{D} = \mathbf{X}_{:, dynamic\_indices}

    Parameters
    ----------
    sequences : `numpy.ndarray`
        Array of input sequences with shape 
        (batch_size, sequence_length, num_features).

    static_indices : `List[int]`
        Indices of static features within the feature dimension.

    dynamic_indices : `List[int]`
        Indices of dynamic features within the feature dimension.

    static_time_step : `int`, default=`0`
        Time step from which to extract static features (default is the first 
        time step).

    reshape_static : `bool`, default=`True`
        Whether to reshape static inputs. If `False`, returns without reshaping.

    reshape_dynamic : `bool`, default=`True`
        Whether to reshape dynamic inputs. If `False`, returns without reshaping.

    static_reshape_shape : `Optional[Tuple[int, ...]]`, default=`None`
        Desired shape for static inputs after reshaping. If `None`, defaults to 
        (batch_size, num_static_vars, 1).

    dynamic_reshape_shape : `Optional[Tuple[int, ...]]`, default=`None`
        Desired shape for dynamic inputs after reshaping. If `None`, defaults to 
        (batch_size, sequence_length, num_dynamic_vars, 1).

    Returns
    -------
    Tuple[`numpy.ndarray`, `numpy.ndarray`]
        A tuple containing:
        - Static inputs with shape as specified.
        - Dynamic inputs with shape as specified.

    Raises
    ------
    ValueError
        If `static_time_step` is out of range for the given sequence length.

    Examples
    --------
    >>> import numpy as np
    >>> from fusionlab.nn.utils import split_static_dynamic
    >>> 
    >>> # Create a dummy sequence array
    >>> sequences = np.random.rand(100, 10, 5)  # (
    ...   batch_size=100, sequence_length=10, num_features=5)
    >>> 
    >>> # Define static and dynamic feature indices
    >>> static_indices = [0, 1]  # e.g., longitude and latitude
    >>> dynamic_indices = [2, 3, 4]  # e.g., year, GWL, density
    >>> 
    >>> # Split the sequences
    >>> static_inputs, dynamic_inputs = split_static_dynamic(
    ...     sequences, 
    ...     static_indices=static_indices, 
    ...     dynamic_indices=dynamic_indices,
    ...     static_time_step=0
    ... )
    >>> 
    >>> print(static_inputs.shape)
    (100, 2, 1)
    >>> print(dynamic_inputs.shape)
    (100, 10, 3, 1)

    Notes
    -----
    - **Static Features:** These are typically location-specific features such as 
      geographical coordinates or categorical attributes that remain constant 
      over time.

    - **Dynamic Features:** These features vary over different time steps and are 
      essential for capturing temporal dependencies in the data.

    - **Reshaping:** The function provides flexibility in reshaping the static and 
      dynamic inputs to match the input requirements of various models, including 
      Temporal Fusion Transformers.

    See Also
    --------
    `create_sequences` : Function to create input sequences and targets for 
    time series forecasting.

    References
    ----------
    .. [1] Qin, Y., Song, D., Chen, H., Cheng, W., Jiang, G., & Cottrell, G. (2017). 
       Temporal fusion transformers for interpretable multi-horizon time series forecasting. 
       *arXiv preprint arXiv:1912.09363*.
    """
    # Validate static_time_step
    if not (0 <= static_time_step < sequences.shape[1]):
        raise ValueError(
            f"static_time_step {static_time_step} is out of range"
            f" for sequence_length {sequences.shape[1]}"
        )
    sequences = validate_sequences(sequences, check_shape= True )
    # Extract static inputs
    static_inputs = sequences[:, static_time_step, static_indices]
    if reshape_static:
        if static_reshape_shape:
            static_inputs = static_inputs.reshape(*static_reshape_shape)
        else:
            static_inputs = static_inputs.reshape(-1, len(static_indices), 1)
    
    # Extract dynamic inputs
    dynamic_inputs = sequences[:, :, dynamic_indices]
    if reshape_dynamic:
        if dynamic_reshape_shape:
            dynamic_inputs = dynamic_inputs.reshape(*dynamic_reshape_shape)
        else:
            dynamic_inputs = dynamic_inputs.reshape(
                -1, sequences.shape[1], len(dynamic_indices), 1)
    
    return static_inputs, dynamic_inputs


@isdf
@validate_params({ 
    'df': ['array-like'], 
    'sequence_length': [Interval(Integral, 2, None, closed ='left')], 
    'target_col': [str], 
    'step': [Interval(Integral, 1, None, closed ='left')], 
    'forecast_horizon': [Interval(Integral, 0, None, closed ='left'), None], 
    }
)
def create_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    target_col: str,
    step: int = 1,
    include_overlap: bool = True,
    drop_last: bool = True,
    forecast_horizon: Optional[int] = None,
    verbose: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and corresponding targets for time series
    forecasting.

    The `create_sequences` function generates sequences of features
    and their corresponding targets from a time series dataset. This
    is essential for training sequence models like Temporal Fusion
    Transformers, LSTMs, and others that rely on temporal dependencies.

    See more in :ref:`User Guide <user_guide>`. 

    Parameters
    ----------
    df : `pandas.DataFrame`
        The processed DataFrame containing features and the target
        variable.
    sequence_length : `int`
        The number of past time steps to include in each input
        sequence.
    target_col : `str`
        The name of the target column.
    step : `int`, default=`1`
        The step size between the starts of consecutive sequences.
    include_overlap : `bool`, default=`True`
        Whether to include overlapping sequences based on the step
        size.
    drop_last : `bool`, default=`True`
        Whether to drop the last sequence if it does not have enough
        data points.
    forecast_horizon : `int`, optional, default=`None`
        The number of future time steps to predict. If set to `None`,
        the function will create targets for a single future time step.
        If provided, targets will consist of the next
        `forecast_horizon` time steps.
    verbose : `int`, default=`3`
        Controls the verbosity of logging. Ranges from `0` (no logs)
        to `7` (maximal logs).

    Returns
    -------
    Tuple[`numpy.ndarray`, `numpy.ndarray`]
        A tuple containing:
          - `sequences`: Array of input sequences with shape
            (num_sequences, sequence_length, num_features).
          - `targets`:
              - If `forecast_horizon` is `None`: Array of target
                values with shape (num_sequences,).
              - If `forecast_horizon` is an integer: Array of target
                sequences with shape (num_sequences,
                forecast_horizon).

    Raises
    ------
    ValueError
        If the DataFrame `df` does not contain the `target_col`.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from fusionlab.nn.utils import create_sequences

    >>> # Create a dummy DataFrame
    >>> data = pd.DataFrame({
    ...     'feature1': np.random.rand(100),
    ...     'feature2': np.random.rand(100),
    ...     'feature3': np.random.rand(100),
    ...     'target': np.random.rand(100)
    ... })

    >>> # Create sequences for single-step forecasting
    >>> sequence_length = 4
    >>> sequences, targets = create_sequences(
    ...     df=data,
    ...     sequence_length=sequence_length,
    ...     target_col='target',
    ...     step=1,
    ...     include_overlap=True,
    ...     drop_last=True,
    ...     forecast_horizon=None
    ... )
    >>> print(sequences.shape)
    (95, 4, 4)
    >>> print(targets.shape)
    (95,)

    >>> # Create sequences for multi-step forecasting (e.g., 3 steps)
    >>> forecast_horizon = 3
    >>> sequences, targets = create_sequences(
    ...     df=data,
    ...     sequence_length=4,
    ...     target_col='target',
    ...     step=1,
    ...     include_overlap=True,
    ...     drop_last=True,
    ...     forecast_horizon=3
    ... )
    >>> print(sequences.shape)
    (92, 4, 4)
    >>> print(targets.shape)
    (92, 3)

    Notes
    -----
    - **Sequence Creation:** The function slides a window of size
      `sequence_length` across the DataFrame to create input sequences.
      Each sequence is associated with a target value or sequence of
      values that immediately follow the input sequence.

    - **Forecast Horizon:** 
        - If `forecast_horizon` is `None`, the function creates
          targets for a single future time step.
        - If `forecast_horizon` is an integer `H`, the function
          creates targets consisting of the next `H` time steps.

    - **Step Size:** The `step` parameter controls the stride of the
      sliding window. A `step` of `1` results in overlapping sequences,
      while a larger `step` reduces overlap.

    - **Handling Incomplete Sequences:** If `drop_last` is set to
      `False`, the function includes the last sequence even if it
      doesn't have enough data points to form a complete sequence or
      target.

    - **Data Validation:** The function utilizes
      `are_all_frames_valid` from `fusionlab.core.checks` to ensure the
      integrity of input DataFrame before processing and
      `exist_features` to verify the presence of the target column.
     
    The sequences generation can be expressed as: 
        
    .. math::
        \\text{For each sequence } i, \\\\
        \\mathbf{X}^{(i)} = \\left[ \\mathbf{x}_{i}, \\mathbf{x}_{i+1}, \\\\
        \\dots, \\mathbf{x}_{i+T-1} \\right] \\\\
        y^{(i)} =
        \\begin{cases}
            \\mathbf{x}_{i+T} & \\text{if } \\text{forecast\\_horizon} =
            \\text{None} \\\\
            \\left[ \\mathbf{x}_{i+T}, \\mathbf{x}_{i+T+1}, \\dots, \\\\
            \\mathbf{x}_{i+T+H-1} \\right] & \\text{if }
            \\text{forecast\\_horizon} = H
        \\end{cases}

    Where:
      - :math:`\\mathbf{X}^{(i)}` is the input sequence of length
        :math:`T`.
      - :math:`y^{(i)}` is the target value(s) following the sequence.
      

    See Also
    --------
    fusionlab.nn.utils.split_static_dynamic :
        Function to split sequences into static and dynamic inputs.

    References
    ----------
    .. [1] Brownlee, J. (2018). Time Series Forecasting with Python:
           Create accurate models in Python to forecast the future and
           gain insight from your time series data. Machine Learning
           Mastery.
    .. [2] Qin, Y., Song, D., Chen, H., Cheng, W., Jiang, G., &
           Cottrell, G. (2017). Temporal fusion transformers for
           interpretable multi-horizon time series forecasting.
           *arXiv preprint arXiv:1912.09363*.
    """
    if verbose >= 1:
        print("[INFO] Starting sequence generation ...")

    # Validate all frames
    are_all_frames_valid(df, df_only=True, error_msg=(
        "DataFrame contains invalid or missing data."
    ))

    # Validate that target_col exists in the DataFrame
    exist_features(
        df, features=target_col,
        name=f"Target column '{target_col}'"
    )

    if verbose >= 2:
        print(
            f"  [DEBUG] sequence_length={sequence_length}, step={step},\n "
            f"drop_last={drop_last}, forecast_horizon={forecast_horizon}"
        )
        print(f"  [DEBUG] DataFrame length={len(df)}")

    sequences = []
    targets = []
    total_length = len(df)

    # Determine the maximum required steps based on forecast_horizon
    max_horizon = forecast_horizon if forecast_horizon is not None else 1

    if verbose >= 2:
        print(f"  [DEBUG] max_horizon set to {max_horizon}")

    # Main loop to generate sequences
    for i in range(0, total_length - sequence_length - max_horizon + 1, step):
        seq = df.iloc[i : i + sequence_length]
        if forecast_horizon is None:
            target = df.iloc[i + sequence_length][target_col]
        else:
            target = df.iloc[
                i + sequence_length : i + sequence_length + forecast_horizon
            ][target_col]

        sequences.append(seq.values)
        # If multi-step, store array; else store scalar
        if forecast_horizon is not None:
            targets.append(target.values)
        else:
            targets.append(target)

        if verbose >= 3:
            print(
                f"    [TRACE] Created sequence index {len(sequences) - 1} "
                f"from row {i} to {i + sequence_length - 1}, "
                f"target index from {i + sequence_length} to "
                f"{i + sequence_length + max_horizon - 1}"
            )

    # Handle any remaining data if drop_last is False
    if not drop_last:
        if forecast_horizon is not None:
            remaining = (total_length - sequence_length) % step
            # We must also ensure we have enough data for the forecast
            # horizon if we are taking the last chunk
            if remaining != 0 and (total_length - sequence_length
                                   ) >= forecast_horizon:
                seq = df.iloc[-sequence_length:]
                target = df.iloc[-forecast_horizon:][target_col]
                sequences.append(seq.values)
                targets.append(target.values)
                if verbose >= 2:
                    print(
                        "  [DEBUG] Appended the last incomplete sequence "
                        "and targets with forecast horizon."
                    )
        else:
            if (total_length - sequence_length) % step != 0:
                seq = df.iloc[-sequence_length:]
                target = df.iloc[-1][target_col]
                sequences.append(seq.values)
                targets.append(target)
                if verbose >= 2:
                    print(
                        "  [DEBUG] Appended the last incomplete sequence "
                        "and single-step target."
                    )

    sequences = np.array(sequences)
    targets = np.array(targets)

    if verbose >= 1:
        print(
            "[INFO] Sequence generation completed. "
            f"Created {sequences.shape[0]} sequences of length "
            f"{sequence_length} with target dimension {targets.shape}."
        )

    return sequences, targets

@is_data_readable(data_to_read ='data')
def compute_forecast_horizon(
    data=None,
    dt_col=None,
    start_pred=None,
    end_pred=None,
    error='raise',
    verbose=1
):
    """
    Compute the forecast horizon for time series forecasting models.

    This function calculates the number of future time steps (`forecast_horizon`)
    a model should predict based on the provided data or specified prediction
    dates. It intelligently infers the frequency of the data and computes the
    horizon accordingly. The function accommodates various datetime formats and
    handles different input scenarios robustly.

    Parameters
    ----------
    data : pandas.DataFrame, pandas.Series, list, or numpy.ndarray, optional
        The dataset containing datetime information. If a `pandas.DataFrame` is
        provided, the `dt_col` parameter must be specified to indicate which
        column contains the datetime data. For `pandas.Series`, `list`, or
        `numpy.ndarray`, the function attempts to infer the frequency directly.

    dt_col : str, optional
        The name of the column in `data` that contains datetime information.
        This parameter is **required** if `data` is a `pandas.DataFrame`.
        Example:
        ``dt_col='timestamp'``

    start_pred : str, int, or datetime-like
        The starting point for forecasting. This can be a date string
        (e.g., `'2023-04-10'`), a `datetime` object, or an integer representing
        a year (e.g., `2024`). If an integer is provided, it is interpreted as a
        year, and a warning is issued to inform the user of this interpretation.

    end_pred : str, int, or datetime-like
        The ending point for forecasting. Similar to `start_pred`, this can
        be a date string, a `datetime` object, or an integer representing a year.
        The function calculates the forecast horizon based on the difference
        between `start_pred` and `end_pred.

    error : {'raise', 'warn', 'ignore'}, default='raise'
        Defines the error handling behavior when encountering issues such as
        invalid input types, missing date columns, or unparseable dates.

        - `'raise'`: Raises a `ValueError` when an error is encountered.
        - `'warn'`: Emits a warning and attempts to proceed with default behavior.

    verbose : int, default=1
        Controls the level of verbosity for debug information.

        - `0`: No output.
        - `1`: Minimal output (e.g., starting message).
        - `2`: Intermediate output (e.g., detected dates, computed horizons).
        - `3`: Detailed output (e.g., types of predictions, inferred frequencies).

    Returns
    -------
    int or None
        The computed `forecast_horizon` representing the number of steps ahead
        the model should predict. Returns `None` if an error occurs and `error`
        is set to `'warn'`.

    Raises
    ------
    ValueError
        If invalid parameters are provided and `error` is set to `'raise'`.

    Examples
    --------
    >>> from fusionlab.nn.utils import compute_forecast_horizon
    >>> import pandas as pd
    >>> import numpy as np
    >>> from datetime import datetime, timedelta
    >>> 
    >>> # Example 1: Using a DataFrame with a Date Column
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    ...     'value': np.random.randn(100)
    ... })
    >>> horizon = compute_forecast_horizon(
    ...     data=df,
    ...     dt_col='date',
    ...     start_pred='2023-04-10',
    ...     end_pred='2023-04-20',
    ...     error='raise',
    ...     verbose=3
    ... )
    >>> print(f"Forecast Horizon: {horizon}")
    Forecast Horizon: 11

    >>> # Example 2: Using a List of Datetimes
    >>> dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    >>> horizon = compute_forecast_horizon(
    ...     data=dates,
    ...     start_pred='2023-04-10',
    ...     end_pred='2023-04-20',
    ...     error='warn',
    ...     verbose=2
    ... )
    >>> print(f"Forecast Horizon: {horizon}")
    Forecast Horizon: 11

    >>> # Example 3: Handling Integer Years
    >>> horizon = compute_forecast_horizon(
    ...     start_pred=2024,
    ...     end_pred=2030,
    ...     error='raise',
    ...     verbose=1
    ... )
    Forecast Horizon: 7

    >>> # Example 4: Without Providing Data (Assuming Frequency Based on Prediction Dates)
    >>> horizon = compute_forecast_horizon(
    ...     start_pred='2023-04-10',
    ...     end_pred='2023-04-20',
    ...     error='raise',
    ...     verbose=1
    ... )
    Forecast Horizon: 11

    Notes
    -----
    - When `data` is not provided, the function relies solely on the difference
      between `start_pred` and `end_pred` to compute the forecast
      horizon. In such cases, if the frequency cannot be inferred, the horizon
      is calculated based on the largest possible time unit (years, months,
      weeks, days).
    
    - If `start_pred` is after `end_pred`, the function returns `0`
      and issues a warning or raises an error based on the `error` parameter.

    - The function attempts to infer the frequency of the data using `pandas`
      utilities. If the frequency cannot be inferred, it defaults to calculating
      the horizon based on the time difference in the most significant unit.

    See Also
    --------
    pandas.date_range : Generates a fixed frequency DatetimeIndex.
    pandas.infer_freq : Infers the frequency of a DatetimeIndex.

    References
    ----------
    .. [1] McKinney, Wes. *Python for Data Analysis: Data Wrangling with Pandas,
       NumPy, and IPython*. O'Reilly Media, 2017.
    .. [2] Harris, C.R., Millman, K.J., van der Walt, S.J., et al. 
       *Array programming with NumPy*. Nature, 585(7825), pp.357-362, 2020.
    """

    if verbose >= 1:
        print("Starting compute_forecast_horizon...")
    
    # Validate verbosity level
    if not isinstance(verbose, int) or not (0 <= verbose <= 3):
        raise ValueError("`verbose` must be an integer between 0 and 3.")
    
    frequency = None
    dates = None

    # Process data if provided
    if data is not None:
        if isinstance(data, pd.DataFrame):
            if dt_col is None:
                message = "`dt_col` must be specified when data is a DataFrame."
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
                    return None
            if dt_col not in data.columns:
                message = f"`dt_col` '{dt_col}' not found in DataFrame."
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
                    return None
            dates = pd.to_datetime(data[dt_col], errors='coerce')
            if dates.isnull().any():
                message = "Some dates could not be converted. Check `dt_col` format."
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
            if verbose >= 2:
                print(f"Detected dates from DataFrame:\n {dates.head()}")
            frequency = pd.infer_freq(dates)
            if frequency is None:
                message = "Could not infer frequency from the date column."
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
            else:
                if verbose >= 2:
                    print(f"Inferred frequency: {frequency}")
        
        elif isinstance(data, (pd.Series, list, np.ndarray)):
            dates = pd.to_datetime(data, errors='coerce')
            if pd.isnull(dates).any():
                message = "Some dates could not be converted. Check the datetime format."
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
            frequency = pd.infer_freq(dates)
            if frequency is None:
                message = "Could not infer frequency from the datetime data."
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
            else:
                if verbose >= 2:
                    print(f"Inferred frequency: {frequency}")
        
        else:
            message = ( 
                "Unsupported data type. Data should"
                " be a DataFrame, Series, list, or numpy array."
                f" Got {type(data).__name__!r}"
                )
            if error == 'raise':
                raise ValueError(message)
            elif error == 'warn':
                warnings.warn(message)
                return None
    
    # Convert start and end prediction to datetime or year
    def convert_prediction(pred):
        if isinstance(pred, int):
            # Treat as year
            converted = pd.to_datetime(f"{pred}", format="%Y", errors='coerce')
            if pd.isnull(converted):
                return None, 'year'
            return converted, 'year'
        else:
            # Attempt to convert to datetime
            converted = pd.to_datetime(pred, errors='coerce')
            if pd.isnull(converted):
                return None, None
            return converted, 'datetime'
    
    start_pred, start_type = convert_prediction(start_pred)
    end_pred, end_type = convert_prediction(end_pred)
    
    if start_pred is None or end_pred is None:
        message = ( 
            "Could not convert `start_prediction`"
            " or `end_prediction` to datetime."
            )
        if error == 'raise':
            raise ValueError(message)
        elif error == 'warn':
            warnings.warn(message)
            return None
    
    if verbose >= 2:
        print(f"Start Prediction: {start_pred} ({start_type})")
        print(f"End Prediction: {end_pred} ({end_type})")
    
    # Handle integer years with warnings
    if start_type == 'year' or end_type == 'year':
        message = (
            "Detected integer inputs for `start_prediction` or `end_prediction`. "
            "Interpreted as years."
        )
        if error == 'warn':
            warnings.warn(message)
        elif error == 'raise':
            emsg ="Set `error='ignore'` to interpret input integers as years."
            raise ValueError(
                message.replace("Interpreted as years.", emsg)
            )
    
    # Compute forecast horizon based on frequency
    if frequency is not None:
        try:
            pred_range = pd.date_range(
                start=start_pred, 
                end=end_pred, 
                freq=frequency
            )
            forecast_horizon = len(pred_range)  # Include both start and end
            if forecast_horizon <= 0:
                message = "`end_prediction` is before `start_prediction`."
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
                    return 0
            if verbose >= 2:
                print(
                    f"Computed forecast horizon based on frequency '{frequency}': "
                    f"{forecast_horizon}"
                )
            if verbose > 0: 
                print(f"Forecast Horizon: {forecast_horizon}")
            return forecast_horizon
        
        except Exception as e:
            message = f"Error computing date range: {e}"
            if error == 'raise':
                raise ValueError(message)
            elif error == 'warn':
                warnings.warn(message)
                return None
    else:
        # If frequency is unknown, compute difference in years, months, days, etc.
        delta = end_pred - start_pred
        if delta.days >= 365:
            forecast_horizon = delta.days // 365 + 1
            unit = "years"
        elif delta.days >= 30:
            forecast_horizon = delta.days // 30 + 1
            unit = "months"
        elif delta.days >= 7:
            forecast_horizon = delta.days // 7 + 1
            unit = "weeks"
        else:
            forecast_horizon = delta.days + 1
            unit = "days"
        
        if forecast_horizon <= 0:
            message = "`end_prediction` is before `start_prediction`."
            if error == 'raise':
                raise ValueError(message)
            elif error == 'warn':
                warnings.warn(message)
                return 0
        
        if verbose >= 2:
            print(
                f"Frequency unknown. Computed forecast horizon based on {unit}: "
                f"{forecast_horizon}"
            )
        if verbose >= 1: 
            print(f"Forecast Horizon: {forecast_horizon}")
            
        return forecast_horizon
    
@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def extract_callbacks_from(fit_params, return_fit_params=False):
    r"""
    Extract Keras callbacks from a dictionary of fit parameters. The
    function scans the provided ``fit_params`` dictionary, looks for keys 
    associated with callback instances, and removes them from 
    ``fit_params`` returning a list of callbacks. Optionally, it can 
    return the updated dictionary without these callbacks.

    This function is particularly useful when working with scikit-learn-
    style estimators that pass parameters through ``**fit_params``. By 
    extracting callbacks directly, the user can seamlessly integrate 
    TensorFlow/Keras callbacks such as `EarlyStopping` or 
    `ModelCheckpoint` into their training pipelines.

    The function can handle two scenarios:
    1. A parameter called ``'callbacks'`` containing a list of callbacks.
    2. Individual callback instances passed as keyword arguments.

    After extraction, if ``return_fit_params=True``, it returns a tuple 
    `(callbacks, fit_params)` where `callbacks` is the extracted list and 
    `fit_params` is the remaining dictionary. Otherwise, it returns only 
    `callbacks`.

    .. math::
        n_{\mathrm{callbacks}} = n_{\mathrm{list\_callbacks}} \;+\;
        n_{\mathrm{kwarg\_callbacks}}

    Here, :math:`n_{\mathrm{callbacks}}` represents the total number of 
    extracted callbacks, :math:`n_{\mathrm{list\_callbacks}}` is the 
    number of callbacks initially found in the ``'callbacks'`` parameter, 
    and :math:`n_{\mathrm{kwarg\_callbacks}}` is the number of callbacks 
    discovered among the other keyword arguments.

    Parameters
    ----------
    fit_params : dict
        The dictionary of parameters to be passed to a model's 
        training method. May contain one of the following:
        
        * ``'callbacks'``: a list of callback instances.
        * Arbitrary keyword arguments that are callback instances.

    return_fit_params : bool, optional
        If ``True``, returns a tuple of `(callbacks, fit_params)` where 
        `fit_params` no longer contains the extracted callbacks. If 
        ``False``, returns only the `callbacks` list. Default is ``False``.

    Returns
    -------
    callbacks : list of tf.keras.callbacks.Callback
        A list of extracted callback instances.

    (callbacks, fit_params) : tuple (only if `return_fit_params=True`)
        A tuple where the first element is a list of extracted callbacks 
        and the second is the updated `fit_params` dictionary after 
        removing the callbacks.

    Examples
    --------
    >>> from fusionlab.nn.utils import extract_callbacks_from
    >>> from tensorflow.keras.callbacks import EarlyStopping
    >>> fit_params = {
    ...     'epochs': 100,
    ...     'batch_size': 64,
    ...     'verbose': 1,
    ...     'early_stopping': EarlyStopping(patience=5)
    ... }
    >>> callbacks, updated_params = extract_callbacks_from(fit_params, 
    ...                                                    return_fit_params=True)
    >>> print(callbacks)
    [<keras.src.callbacks.EarlyStopping object at 0x...>]
    >>> print(updated_params)
    {'epochs': 100, 'batch_size': 64, 'verbose': 1}

    Notes
    -----
    Consider using this function when integrating Keras callbacks within 
    a pipeline or estimator that follows scikit-learn conventions, where 
    parameters are passed as `fit_params`. This approach enables a clean 
    and modular integration of callbacks into your training loops.

    See Also
    --------
    tf.keras.callbacks.Callback :
        Base class used to build new callbacks.

    References
    ----------
    .. [1] François Chollet, et al. Keras Documentation. 
           https://keras.io/api/callbacks/
    """

    callbacks = []

    # If user provides a callbacks list directly
    if 'callbacks' in fit_params and isinstance(fit_params['callbacks'], list):
        cb_list = fit_params.pop('callbacks')
        for c in cb_list:
            if isinstance(c, Callback):
                callbacks.append(c)

    # If user provides individual callbacks as keyword arguments
    keys_to_remove = []
    for k, v in fit_params.items():
        if isinstance(v, Callback):
            callbacks.append(v)
            keys_to_remove.append(k)

    # Remove callback entries from fit_params
    for k in keys_to_remove:
        fit_params.pop(k, None)

    if return_fit_params:
        return callbacks, fit_params
    return callbacks


@TypeEnforcer({"0": 'array-like', "1": 'array-like'})
@ParamsValidator({ 
    "final_processed_data": ['array-like:df'], 
    "feature_columns":['array-like'], 
    "dynamic_feature_indices": ['array-like'], 
    "sequence_length": [Interval( Integral, 1, None, closed="left")], 
    "dt_col": [str], 
    "static_feature_names": ['array-like', None], 
    "forecast_horizon": [Interval( Integral, 1, None, closed="left"), None],
    "future_years":['array-like', Interval( Integral, 1, None, closed="left"), None],
    "encoded_cat_columns": ['array-like', None], 
    "scaling_params": [dict, None]
    }
)
def prepare_spatial_future_data(
    final_processed_data: pd.DataFrame,
    feature_columns: List[str],
    dynamic_feature_indices: List[int],
    sequence_length: int = 1,
    dt_col: str = 'date',
    static_feature_names: Optional[List[str]] = None,
    forecast_horizon: Optional[int] = None,
    future_years: Optional[List[int]] = None,
    encoded_cat_columns: Optional[List[str]] = None,
    scaling_params: Optional[Dict[str, Dict[str, float]]] = None,
    spatial_cols: Tuple [str, str]=None, 
    squeeze_last: bool=False, 
    verbosity: int = 0,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[int],
    List[int],
    List[float],
    List[float]
]:
    """
    Prepare future static and dynamic inputs for making predictions.

    This function prepares the necessary static and dynamic inputs required for
    forecasting future values in time series data. It processes the provided
    dataset by grouping it by ``location_id``, extracting the last sequence of
    data points based on the specified ``sequence_length``, and generating future
    inputs for prediction over the defined ``forecast_horizon``.

    The function handles both integer and datetime representations of the
    ``dt_col``, extracting the year from datetime columns when necessary. It
    also allows for flexibility in specifying static features and encoded
    categorical variables.

    .. math::
        \text{scaled\_time} = \frac{\text{future\_time} - \mu}{\sigma}

    Parameters
    ----------
    final_processed_data : pandas.DataFrame
        The processed DataFrame containing all features and targets. Must include
        the ``location_id`` column and the specified ``dt_col``.
    feature_columns : List[str]
        List of feature column names to be used for dynamic input preparation.
    dynamic_feature_indices : List[int]
        Indices of dynamic features in ``feature_columns``. These features are
        considered time-dependent and are used to prepare dynamic inputs.
    sequence_length : int, optional
        The number of past time steps to include in each input sequence.
        Default is ``1``.
    dt_col : str, optional
        The name of the time-related column in ``final_processed_data``.
        Defaults to ``'date'``.
    static_feature_names : List[str], optional
        List of static feature column names. If not provided, defaults to
        ``['longitude', 'latitude']`` plus any ``encoded_cat_columns``.
    forecast_horizon : int, optional
        The number of future time steps to predict. If set to ``None``,
        the function defaults to predicting the next immediate time step.
    future_years : List[int], optional
        List of future years to predict. Must match the length of
        ``forecast_horizon`` if ``forecast_horizon`` is provided.
    encoded_cat_columns : List[str], optional
        List of encoded categorical column names to be treated as static features.
    scaling_params : Dict[str, Dict[str, float]], optional
        Dictionary containing scaling parameters (mean and standard deviation)
        for features. Example: ``{'year': {'mean': 2000, 'std': 10}}``.
        If not provided, the function computes the mean and std for the
        ``dt_col``.
    squeeze_last: bool, default=True, 
       Squeeze the last axis which correspond to the output dimension ``y`` 
       if equal to ``1``. 
       
    verbosity : int, optional
        Verbosity level from ``0`` to ``7`` for debugging and understanding
        the process. Higher values produce more detailed logs.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[int], List[int], List[float], List[float]]
        A tuple containing:

        - ``future_static_inputs`` : numpy.ndarray
            Array of future static inputs with shape ``(num_samples, num_static_vars, 1)``.
        - ``future_dynamic_inputs`` : numpy.ndarray
            Array of future dynamic inputs with shape ``(num_samples, sequence_length, num_dynamic_vars, 1)``.
        - ``future_years_list`` : List[int]
            List of future time values corresponding to each sample.
        - ``location_ids_list`` : List[int]
            List of location IDs corresponding to each sample.
        - ``longitudes`` : List[float]
            List of longitude values corresponding to each sample.
        - ``latitudes`` : List[float]
            List of latitude values corresponding to each sample.

    Examples
    --------
    >>> from fusionlab.nn.utils import prepare_spatial_future_data
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'location_id': [1, 1, 1, 2, 2, 2],
    ...     'year': [2018, 2019, 2020, 2018, 2019, 2020],
    ...     'longitude': [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
    ...     'latitude': [50.0, 50.0, 50.0, 60.0, 60.0, 60.0],
    ...     'temperature': [15, 16, 15.5, 20, 21, 20.5],
    ...     'rainfall': [100, 110, 105, 200, 210, 205],
    ...     'encoded_cat': [1, 1, 1, 2, 2, 2]
    ... })
    >>> feature_cols = ['year', 'temperature', 'rainfall', 'encoded_cat']
    >>> dynamic_indices = [0, 1, 2]
    >>> future_static, future_dynamic, future_years, loc_ids, longs,\
        lats = prepare_spatial_future_data(
    ...     final_processed_data=data,
    ...     feature_columns=feature_cols,
    ...     dynamic_feature_indices=dynamic_indices,
    ...     sequence_length=2,
    ...     forecast_horizon=1,
    ...     future_years=[2021],
    ...     encoded_cat_columns=['encoded_cat'],
    ...     verbosity=5,
    ...     dt_col='year'
    ... )
    >>> print(future_static.shape)
    (2, 3, 1)
    >>> print(future_dynamic.shape)
    (2, 2, 3, 1)

    Notes
    -----
    - The function handles both integer and datetime representations of the
      ``dt_col``. If ``dt_col`` is a datetime type, the year is extracted for
      scaling purposes.
    - If ``forecast_horizon`` is set to ``None``, the function defaults to
      generating data for the next immediate time step based on the last entry
      in the time column.
    - Ensure that the length of ``future_years`` matches ``forecast_horizon``
      if ``forecast_horizon`` is provided.
    - The ``static_feature_names`` parameter allows for flexibility in specifying
      which static features to include. If not provided, it defaults to
      ``['longitude', 'latitude']`` plus any ``encoded_cat_columns``.

    See Also
    --------
    prepare_future_data : Main function for preparing future data inputs.

    References
    ----------
    .. [1] Smith, J., & Doe, A. (2020). *Time Series Forecasting Methods*. Journal of
       Data Science, 15(3), 123-145.
    .. [2] Johnson, L. (2019). *Advanced Neural Networks for Time Series Prediction*.
       Machine Learning Review, 22(4), 567-589.
    """
    future_years = columns_manager(future_years, empty_as_none= False )
    static_feature_names = static_feature_names or []
    # Initialize verbosity levels
    def log(message: str, level: int):
        if verbosity >= level:
            print(message)

    log(
        "Starting prepare_future_data with verbosity level "
        f"{verbosity}", 
        1
    ) 
    if spatial_cols is None: 
        spatial_cols=['longitude', 'latitude']
    check_spatial_columns(final_processed_data, spatial_cols=spatial_cols)
    spatial_cols = columns_manager(spatial_cols)
    # Lists to store future inputs and related data
    future_static_inputs_list = []
    future_dynamic_inputs_list = []
    future_years_list = []
    location_ids_list = []
    longitudes = []
    latitudes = []

    # Handle scaling parameters
    scaling_params = scaling_params or {}
    log("Scaling parameters set.", 3)

    # Determine time column
    detected_dt_col = _determine_dt_column(final_processed_data, dt_col, log)
    
    # Handle scaling for time column
    time_mean, time_std = _handle_time_scaling(
        final_processed_data,
        detected_dt_col,
        scaling_params,
        log
    )

    # Ensure encoded_cat_columns is a list
    encoded_cat_columns = encoded_cat_columns or []
    if not encoded_cat_columns:
        log("No encoded categorical columns provided.", 5)
    else:
        log(f"Encoded categorical columns: {encoded_cat_columns}", 5)

    # Determine static feature names
    static_features = _determine_static_features(
        static_feature_names, encoded_cat_columns, log, 
        spatial_cols = spatial_cols
    )

    # Group data by 'location_id'
    grouped = _group_by_location(final_processed_data, log)

    # Iterate over each location
    for name, group in grouped:
        group = _sort_group_by_time(group, detected_dt_col, log, name)

        # Ensure there is enough data to create a sequence
        if len(group) >= sequence_length:
            last_sequence = group.iloc[-sequence_length:]
            last_sequence_features = last_sequence[feature_columns]
            log(
                f"Extracted last {sequence_length} records for "
                f"location_id {name}.", 
                4
            )

            # Prepare static inputs
            static_inputs, static_values = _prepare_static_inputs(
                last_sequence_features,
                static_features,
                log,
                name
            )
            # Prepare dynamic inputs
            dynamic_inputs = _prepare_dynamic_inputs(
                last_sequence_features,
                dynamic_feature_indices,
                sequence_length,
                log,
                name
            )

            # Determine future steps based on forecast_horizon
            future_steps = forecast_horizon if forecast_horizon is not None else 1

            # Generate future inputs for each future step
            for step in range(future_steps):
                future_time = _determine_future_time(
                    forecast_horizon,
                    step,
                    future_years,
                    group,
                    detected_dt_col,
                    log,
                    name
                )

                log(
                    f"Generating future data for step {step + 1} "
                    f"time {future_time} at location_id {name}.", 
                    6
                )

                # Copy dynamic inputs
                future_dynamic_inputs = dynamic_inputs.copy()

                # Update the time feature for the next time step
                _update_time_feature(
                    future_dynamic_inputs,
                    feature_columns,
                    dynamic_feature_indices,
                    detected_dt_col,
                    future_time,
                    time_mean,
                    time_std,
                    log,
                    name
                )

                # Append to lists
                _append_to_lists(
                    future_static_inputs_list,
                    future_dynamic_inputs_list,
                    future_years_list,
                    location_ids_list,
                    longitudes,
                    latitudes,
                    static_inputs,
                    static_values,
                    static_features,
                    future_dynamic_inputs,
                    future_time,
                    name
                )

        else:
            log(
                f"Insufficient data for location_id {name}: "
                f"required={sequence_length}, available={len(group)}.", 
                4
            )

    # Convert lists to numpy arrays for model input
    log("Converting lists to numpy arrays.", 3)
    future_static_inputs, future_dynamic_inputs = _convert_to_numpy(
        future_static_inputs_list,
        future_dynamic_inputs_list,
        log
    )
    if squeeze_last: 
        future_dynamic_inputs= future_dynamic_inputs.squeeze(-1)
        future_static_inputs= future_static_inputs.squeeze(-1)
    log(
        f"Final shapes - static: {future_static_inputs.shape}, "
        f"dynamic: {future_dynamic_inputs.shape}", 
        4
    )

    log("prepare_future_data completed successfully.", 1)

    return (
        future_static_inputs,
        future_dynamic_inputs,
        future_years_list,
        location_ids_list,
        longitudes,
        latitudes
    )


# Helper Functions
def _determine_dt_column(
        final_processed_data: pd.DataFrame,
        dt_col: str,
        log_func
    ) -> str:
    """Determine and validate the time column."""
    if dt_col in final_processed_data.columns:
        log_func(f"Using time column: {dt_col}", 4)
        return dt_col
    else:
        raise ValueError(
            f"final_processed_data must contain the '{dt_col}' column."
        )


def _handle_time_scaling(
        final_processed_data: pd.DataFrame,
        detected_dt_col: str,
        scaling_params: Dict[str, Dict[str, float]],
        log_func
    ) -> Tuple[float, float]:
    """Handle scaling for the time column."""
    if detected_dt_col not in scaling_params:
        if detected_dt_col == 'year':
            time_mean = final_processed_data[detected_dt_col].mean()
            time_std = final_processed_data[detected_dt_col].std()
        else:
            # Handle 'date' or other time columns
            if pd.api.types.is_datetime64_any_dtype(
                final_processed_data[detected_dt_col]
            ):
                final_processed_data['year_extracted'] = (
                    final_processed_data[detected_dt_col].dt.year
                )
                dt_col_extracted = 'year_extracted'
                time_mean = final_processed_data[dt_col_extracted].mean()
                time_std = final_processed_data[dt_col_extracted].std()
                log_func(
                    f"Extracted year from '{detected_dt_col}' column as "
                    f"'{dt_col_extracted}'",
                    5
                )
                detected_dt_col = dt_col_extracted
            else:
                # Assume 'date' column contains integer year
                time_mean = final_processed_data[detected_dt_col].mean()
                time_std = final_processed_data[detected_dt_col].std()
        scaling_params[detected_dt_col] = {'mean': time_mean, 'std': time_std}
        log_func(
            f"Computed scaling for '{detected_dt_col}': mean={time_mean}, "
            f"std={time_std}", 
            4
        )
    else:
        time_mean = scaling_params[detected_dt_col]['mean']
        time_std = scaling_params[detected_dt_col]['std']
        log_func(
            f"Using provided scaling for '{detected_dt_col}': mean="
            f"{time_mean}, std={time_std}", 
            4
        )
    return time_mean, time_std

def _determine_static_features(
        static_feature_names: Optional[List[str]],
        encoded_cat_columns: List[str],
        log_func, 
        spatial_cols, 
    ) -> List[str]:
    """Determine the static feature names."""
    
    if static_feature_names is not None or len(static_feature_names) !=0:
        log_func(
            f"Using provided static feature names: {static_feature_names}", 
            5
        )
        return static_feature_names + encoded_cat_columns
    else:
        static_features = list(spatial_cols) + encoded_cat_columns
        log_func(
            f"Using default static feature names: {static_features}", 
            5
        )
        return static_features


def _group_by_location(
        final_processed_data: pd.DataFrame,
        log_func
    ) -> pd.core.groupby.DataFrameGroupBy:
    """Group the data by 'location_id'."""
    if 'location_id' not in final_processed_data.columns:
        raise ValueError(
            "final_processed_data must contain 'location_id' column."
        )
    grouped = final_processed_data.groupby('location_id')
    log_func(
        "Grouped data by 'location_id'. Number of groups: "
        f"{len(grouped)}", 
        2
    )
    return grouped


def _sort_group_by_time(
        group: pd.DataFrame,
        detected_dt_col: str,
        log_func,
        location_id: Union[int, str]
    ) -> pd.DataFrame:
    """Sort the group by the time column."""
    group = group.sort_values(detected_dt_col).reset_index(drop=True)
    log_func(f"Processing location_id: {location_id}", 3)
    return group


def _prepare_static_inputs(
        last_sequence_features: pd.DataFrame,
        static_features: List[str],
        log_func,
        location_id: Union[int, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare static inputs from the last sequence."""
    if not all(
        col in last_sequence_features.columns 
        for col in static_features
    ):
        raise ValueError(
            f"One or more static feature columns missing in data "
            f"for location_id {location_id}."
        )
    static_values = last_sequence_features.iloc[0][
        static_features
    ].values
    static_inputs = static_values.reshape(
        1, len(static_values)
    ).astype(np.float32)  # Shape: (1, num_static_vars)
    log_func(
        f"Prepared static inputs for location_id {location_id}.", 
        5
    )
    return static_inputs, static_values

def _prepare_dynamic_inputs(
        last_sequence_features: pd.DataFrame,
        dynamic_feature_indices: List[int],
        sequence_length: int,
        log_func,
        location_id: Union[int, str]
    ) -> np.ndarray:
    """Prepare dynamic inputs from the last sequence."""
    dynamic_features = last_sequence_features.iloc[
        :, dynamic_feature_indices
    ].values
    dynamic_inputs = dynamic_features.reshape(
        sequence_length,
        len(dynamic_feature_indices),
        1
    ).astype(np.float32)  # Shape: (sequence_length, num_dynamic_vars, 1)
    log_func(
        f"Prepared dynamic inputs for location_id {location_id}.", 
        5
    )
    return dynamic_inputs


def _determine_future_time(
        forecast_horizon: Optional[int],
        step: int,
        future_years: List[int],
        group: pd.DataFrame,
        detected_dt_col: str,
        log_func,
        location_id: Union[int, str]
    ) -> int:
    """Determine the future time value."""
    if forecast_horizon is not None:
        if step < len(future_years):
            future_time = future_years[step]
        else:
            raise ValueError(
                "Length of future_years must be equal to forecast_horizon."
            )
    else:
        # If forecast_horizon is None, use the next time step
        if detected_dt_col == 'year_extracted':
            future_time = int(group[detected_dt_col].iloc[-1]) + 1
        elif detected_dt_col == 'year':
            future_time = int(group[detected_dt_col].iloc[-1]) + 1
        else:
            # Assume 'date' column has been processed
            if pd.api.types.is_datetime64_any_dtype(
                group[detected_dt_col]
            ):
                last_date = group[detected_dt_col].iloc[-1]
                future_time = last_date.year + 1
            else:
                future_time = int(group[detected_dt_col].iloc[-1]) + 1
    return future_time

def _update_time_feature(
        future_dynamic_inputs: np.ndarray,
        feature_columns: List[str],
        dynamic_feature_indices: List[int],
        detected_dt_col: str,
        future_time: int,
        time_mean: float,
        time_std: float,
        log_func,
        location_id: Union[int, str]
    ):
    """Update the time feature in the dynamic inputs."""
    if detected_dt_col in feature_columns:
        time_idx_in_feature = feature_columns.index(detected_dt_col)
        if time_idx_in_feature in dynamic_feature_indices:
            dynamic_time_index = (
                dynamic_feature_indices.index(time_idx_in_feature)
            )
            scaled_time = (future_time - time_mean) / time_std
            future_dynamic_inputs[
                -1, dynamic_time_index, 0
            ] = scaled_time
            log_func(
                f"Scaled '{detected_dt_col}' for future input: "
                f"original={future_time}, scaled={scaled_time}", 
                7
            )
        else:
            log_func(
                f"'{detected_dt_col}' is not in dynamic_feature_indices "
                f"for location_id {location_id}.", 
                6
            )
    else:
        log_func(
            f"'{detected_dt_col}' column not found in "
            f"feature_columns for location_id {location_id}.", 
            6
        )


def _append_to_lists(
        future_static_inputs_list: List[np.ndarray],
        future_dynamic_inputs_list: List[np.ndarray],
        future_years_list: List[int],
        location_ids_list: List[Union[int, str]],
        longitudes: List[float],
        latitudes: List[float],
        static_inputs: np.ndarray,
        static_values: np.ndarray,
        static_features: List[str],
        future_dynamic_inputs: np.ndarray,
        future_time: int,
        name: Union[int, str]
    ):
    """Append the prepared inputs and related data to their respective lists."""
    future_static_inputs_list.append(static_inputs)
    future_dynamic_inputs_list.append(future_dynamic_inputs)
    future_years_list.append(future_time)
    location_ids_list.append(name)
    longitudes.append(static_values[static_features.index('longitude')] 
                      if 'longitude' in static_features else np.nan)
    latitudes.append(static_values[static_features.index('latitude')] 
                     if 'latitude' in static_features else np.nan)


def _convert_to_numpy(
        future_static_inputs_list: List[np.ndarray],
        future_dynamic_inputs_list: List[np.ndarray],
        log_func
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the lists to numpy arrays with appropriate shapes."""
    future_static_inputs = np.array(
        future_static_inputs_list
    )  # Shape: (num_samples, num_static_vars)
    future_dynamic_inputs = np.array(
        future_dynamic_inputs_list
    )  # Shape: (num_samples, sequence_length, num_dynamic_vars, 1)

    # Reshape static inputs to ensure the correct shape
    future_static_inputs = future_static_inputs.reshape(
        future_static_inputs.shape[0],
        future_static_inputs.shape[1],
        1
    )
    return future_static_inputs, future_dynamic_inputs

def set_default_params(
    quantiles: Union[str, List[float], None] = None,
    scales: Union[str, List[int], None] = None,
    multi_scale_agg: Union[str, None] = None
) -> Tuple[List[float], List[int], bool]:
    """
    Sets and validates default values for quantiles, scales, and
    return_sequences parameters.

    Parameters
    ----------
    quantiles : str, list of float, or None, optional
        Specifies the quantiles to be used for probabilistic forecasting.
        
        - If set to `'auto'`, it defaults to `[0.1, 0.5, 0.9]`.
        - If a list is provided, each element must be a float between 0 and 1 (exclusive).
        - If `None`, it remains as `None` (can be used for deterministic forecasting).

    scales : str, list of int, or None, optional
        Specifies the scaling factors to be used in multi-scale processing.
        
        - If set to `'auto'` or `None`, it defaults to `[1]`.
        - If a list is provided, each element must be a positive integer.

    multi_scale_agg : str or None, optional
        Determines the aggregation method for multi-scale features.
        
        - If set to `None`, `return_sequences` is set to `False`.
        - Otherwise, `return_sequences` is set to `True`.
        - Expected aggregation methods could include `'average'`, `'concat'`,
          ``'sum'``, ``'last'``, ``'auto'`` (fallack to 'last'),etc., 
        depending on model requirements.

    Returns
    -------
    Tuple[List[float], List[int], bool]
        A tuple containing:
        
        - `quantiles`: A list of validated quantile floats.
        - `scales`: A list of validated scale integers.
        - `return_sequences`: A boolean indicating whether to return 
        sequences based on `multi_scale_agg`.

    Raises
    ------
    ValueError
        If `quantiles` is neither `'auto'` nor a list of valid floats.
        If `scales` is neither `'auto'` nor a list of valid positive integers.
        If `multi_scale_agg` is provided but not a recognized aggregation method.

    Examples
    --------
    >>> # Example 1: Using default 'auto' settings
    >>> quantiles, scales, return_sequences = set_default_parameters(
        quantiles='auto', scales='auto', multi_scale_agg='auto')
    >>> print(quantiles)
    [0.1, 0.5, 0.9]
    >>> print(scales)
    [1]
    >>> print(return_sequences)
    True

    >>> # Example 2: Providing custom quantiles and scales
    >>> quantiles, scales, return_sequences = set_default_parameters(
    ...     quantiles=[0.05, 0.5, 0.95],
    ...     scales=[1, 2, 4],
    ...     multi_scale_agg='concat'
    ... )
    >>> print(quantiles)
    [0.05, 0.5, 0.95]
    >>> print(scales)
    [1, 2, 4]
    >>> print(return_sequences)
    True

    >>> # Example 3: Invalid quantiles input
    >>> set_default_parameters(quantiles=[-0.1, 1.2])
    Traceback (most recent call last):
    ...
    ValueError: Each quantile must be a float between 0 and 1 (exclusive). 
    Invalid quantiles: [-0.1, 1.2]

    >>> # Example 4: Invalid scales input
    >>> set_default_parameters(scales=[0, -2])
    Traceback (most recent call last):
    ...
    ValueError: Each scale must be a positive integer. Invalid scales: [0, -2]
    """

    # Set default quantiles if 'auto'
    if quantiles == 'auto':
        quantiles = [0.1, 0.5, 0.9]
    elif quantiles is not None:
        if not isinstance(quantiles, list):
            raise ValueError(
                "'quantiles' must be a list of floats or"
               f" 'auto', but got type {type(quantiles).__name__}.")
        # Validate each quantile
        invalid_quantiles = [q for q in quantiles if not isinstance(
            q, float) or not (0 < q < 1)]
        if invalid_quantiles:
            raise ValueError(
                f"Each quantile must be a float between 0 and 1 (exclusive). "
                f"Invalid quantiles: {invalid_quantiles}"
            )
    else:
        # quantiles remains None
        pass

    # Set default scales if 'auto' or None
    if scales is None or scales == 'auto':
        scales = [1]
    elif isinstance(scales, list):
        # Validate each scale
        invalid_scales = [s for s in scales if not isinstance(s, int) or s <= 0]
        if invalid_scales:
            raise ValueError(
                f"Each scale must be a positive integer. Invalid scales: {invalid_scales}"
            )
    else:
        raise ValueError(
            "'scales' must be a list of positive integers,"
            f" 'auto', or None, but got type {type(scales).__name__}.")

    # Set return_sequences based on multi_scale_agg
    if multi_scale_agg is None:
        return_sequences = False
    else:
        # Optionally, you can validate multi_scale_agg against allowed methods
        allowed_aggregations = {
            'sum', 'concat', 'average', 'flatten', 'last', 'auto', 
            }  # Example allowed methods
        if not isinstance(multi_scale_agg, str):
            raise ValueError(
                f"'multi_scale_agg' must be a string indicating"
                " the aggregation method, 'auto', or None, "
                f"but got type {type(multi_scale_agg).__name__}."
            )
        if multi_scale_agg.lower() not in allowed_aggregations:
            raise ValueError(
                f"'multi_scale_agg' must be one of {allowed_aggregations}, "
                f"'auto', or None, but got '{multi_scale_agg}'."
            )
        return_sequences = True

    return quantiles, scales, return_sequences


@isdf
def reshape_xtft_data(
    df: pd.DataFrame,
    dt_col: str,
    target_col: str,
    dynamic_cols: List[str],
    static_cols: Optional[List[str]] = None,
    future_cols: Optional[List[str]] = None,
    spatial_cols: Optional[Union[str, List[str]]] = None,
    time_steps: int = 4,
    forecast_horizon: int = 1,
    to_datetime: Optional[str] = None,
    model: str = "xtft",
    error: str = "raise", 
    savefile: Optional[str] = None,
    forecast_horizons: Optional[int] = None,  
    verbose: int = 1, 
    **kw
) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Reshapes time series data into rolling sequences for models like
    Temporal Fusion Transformer (TFT) and Extreme Temporal Fusion
    Transformer (XTFT).

    This function transforms a Pandas DataFrame into a set of aligned
    sequences suitable for training or evaluating sequence-to-sequence
    models. It handles static, dynamic (past observed), and future
    known covariates, and can process data grouped by spatial or
    other identifiers. The core process involves validating inputs,
    optionally grouping data, sorting, calculating the total number
    of sequences, pre-allocating NumPy arrays for efficiency, and then
    populating these arrays by creating rolling windows. Future features
    are extracted for a combined period covering both the lookback
    (`time_steps`) and prediction (`forecast_horizons`) windows.

    """
    # *************************************************************
    from ..utils.ts_utils import ts_validator
    # *************************************************************
    # Backward‐compatibility: accept `forecast_horizons` (deprecated)
    if forecast_horizons is not None:
       warnings.warn(
           "`forecast_horizons` is deprecated and will be removed in a "
           "future release; please use `forecast_horizon` instead.",
           DeprecationWarning,
           stacklevel=2
       )
       # override the new param with the old value
       forecast_horizon = forecast_horizons
        
    # --- Initial Validations ---
    if verbose >=1: 
        print("[INFO] Reshaping time‑series"
              " data into rolling sequences...")
        
    # Validate and convert datetime column
    # verbose > 1 maps to True for ts_validator's own verbose flag
    df = ts_validator(
        df, dt_col=dt_col, 
        to_datetime=to_datetime,
        as_index=False, 
        verbose=verbose > 1
    )

    # Ensure target column exists
    exist_features(
        df, features=target_col, 
        name='Target column',
    )

    # Manage and validate feature column lists
    # Convert single string to list and check existence
    dynamic_cols = columns_manager(dynamic_cols, empty_as_none=False)
    exist_features(
        df, features=dynamic_cols, 
        name="Dynamic columns",
        error=error
    )

    if static_cols:
        static_cols = columns_manager(static_cols)
        exist_features(
            df, features=static_cols,
            name="Static columns", error=error
        )
    if future_cols:
        future_cols = columns_manager(future_cols)
        exist_features(
            df, features=future_cols,
            name="Future columns", error=error
            )

    spatial_cols = columns_manager(
        spatial_cols, empty_as_none=True
        )
    if spatial_cols:
        exist_features(
            df, features=spatial_cols, 
            name="Spatial columns",
            error=error
        )

    # Validate positive integers for sequence lengths
    time_steps = validate_positive_integer(time_steps, 'time_steps')
    forecast_horizon = validate_positive_integer(
        forecast_horizon, 'forecast_horizon'
        )
    # Minimum length a group must have to produce at least one sequence
    min_len_per_group = time_steps + forecast_horizon

    if verbose >= 2:
        print(f"  [DEBUG] Validated parameters: time_steps={time_steps},"
              f" forecast_horizon={forecast_horizon}")

    # --- Grouping and Sorting ---
    if spatial_cols:
        # Sort by spatial groups first, then by time within each group
        # to ensure correct sequence generation.
        df = df.sort_values(
            by=spatial_cols + [dt_col]).reset_index(drop=True)
        grouped = df.groupby(spatial_cols)
        # Store group keys for the second pass
        group_keys = list(grouped.groups.keys())
        if verbose >= 1:
            print(f"\n[INFO] Data grouped by {spatial_cols} into "
                  f"{len(group_keys)} groups.")
    else:
        # If no spatial columns, sort the entire DataFrame by time
        df = df.sort_values(by=dt_col).reset_index(drop=True)
        # Treat entire DataFrame as a single group for iteration
        grouped = [(None, df)] # List of one tuple
        group_keys = [None] # Single key for the one group
        if verbose >= 1:
            print("[INFO] Processing entire DataFrame as a single group.")

    # --- First Pass: Calculate total number of sequences ---
    total_sequences = 0
    # Store indices of groups that are long enough
    valid_group_indices: List[int] = []
    # Store group DataFrames to avoid re-grouping in the second pass
    temp_grouped_list: List[pd.DataFrame] = []

    if verbose >= 2:
        print("[INFO] Starting first pass to calculate total sequences...")
    for group_idx, group_key in enumerate(group_keys):
        # Retrieve the group DataFrame
        group_df = grouped.get_group(group_key) if spatial_cols else df
        temp_grouped_list.append(group_df)

        if len(group_df) < min_len_per_group:
            if verbose >= 1:
                key_str = group_key if group_key is not None \
                    else "<Full Dataset>"
                print(
                    "   \n[DEBUG] Warning:\n"
                    f"       Group '{key_str}'\n"
                    f"       has {len(group_df)} points, less"
                    f" than min required ({min_len_per_group}). Skipping."
                )
            continue # Skip this group if too short
        # Add number of possible sequences from this group
        total_sequences += (len(group_df) - min_len_per_group + 1)
        valid_group_indices.append(group_idx)
        
    if total_sequences == 0:
        raise ValueError(
            "Not enough data points in any group to create sequences "
            f"with time_steps={time_steps} and "
            f"forecast_horizon={forecast_horizon}."
        )
    if verbose >= 1:
        print(f"\n[INFO] Total valid sequences to be generated: {total_sequences}")

    # --- Pre-allocate NumPy Arrays for efficiency ---
    # Get feature dimensions
    num_dynamic_features = len(dynamic_cols)
    num_static_features = len(static_cols) if static_cols else 0
    num_future_features = len(future_cols) if future_cols else 0

    # Initialize arrays with zeros, or None if features not provided
    static_data_arr = np.zeros(
        (total_sequences, num_static_features), dtype=np.float32
        ) # if static_cols else None

    dynamic_data_arr = np.zeros(
        (total_sequences, time_steps, num_dynamic_features),
        dtype=np.float32
        )

    future_data_arr = np.zeros(
        (total_sequences, time_steps + forecast_horizon,
         num_future_features), dtype=np.float32
        ) # if future_cols else None

    target_data_arr = np.zeros(
        (total_sequences, forecast_horizon, 1), dtype=np.float32
        ) # Assuming target_col is univariate

    if verbose >= 2:
        print("\n[INFO] Pre-allocated NumPy arrays for sequence data.")
        print(f"  [DEBUG] Static array shape: {static_data_arr.shape}")
        print(f"  [DEBUG] Dynamic array shape: {dynamic_data_arr.shape}")
        print(f"  [DEBUG] Future array shape: {future_data_arr.shape}")
        print(f"  [DEBUG] Target array shape: {target_data_arr.shape}")
        
    # --- Second Pass: Populate Pre-allocated Arrays ---
    current_seq_idx = 0 # Index for filling the pre-allocated arrays
    if verbose >= 2:
        print("\n[INFO] Starting second pass to populate sequence arrays...")


    for group_idx in valid_group_indices:
        group_df = temp_grouped_list[group_idx]
        group_key_for_log = group_keys[group_idx] if spatial_cols \
            else "<Full Dataset>"
        if verbose >= 3:
            print()
            print_box(
                f"Processing group: {group_key_for_log}",
                width=_TW,
                align='center',
                border_char='+',
                horizontal_char='-',
                vertical_char='|',
                padding=1
            )
        group_static_values = None
        if static_cols and num_static_features > 0: # Check num_static_features
            if spatial_cols:
                group_static_values = group_df.iloc[0][static_cols].values
        
        num_seq_in_group = len(group_df) - min_len_per_group + 1
        for i in range(num_seq_in_group):
            if static_cols and num_static_features > 0:
                if spatial_cols:
                    static_data_arr[current_seq_idx] = group_static_values
                else:
                    static_data_arr[current_seq_idx] = group_df.iloc[i][static_cols].values
            # If num_static_features is 0, static_data_arr[current_seq_idx] is an empty slice.

            dynamic_data_arr[current_seq_idx] = group_df.iloc[
                i : i + time_steps][dynamic_cols].values

            if future_cols and num_future_features > 0:
                future_data_arr[current_seq_idx] = group_df.iloc[
                    i : i + time_steps + forecast_horizon][future_cols].values
            # If num_future_features is 0, future_data_arr[current_seq_idx] is an empty slice.

            target_data_arr[current_seq_idx] = group_df.iloc[
                i + time_steps : i + time_steps + forecast_horizon
                ][target_col].values.reshape(forecast_horizon, 1)
            
            if verbose >= 7: # Very detailed logging for debugging
                print(f"\n    [TRACE] Seq {current_seq_idx} (Group {group_key_for_log}, "
                      f"WinStartIdx {i}):")
                if num_static_features > 0:
                    print("      Static sample: "
                          f"{list(static_data_arr[current_seq_idx][:2])}") # First 2
                print("      - Dynamic sample (first step): "
                      f"{list(dynamic_data_arr[current_seq_idx][0])}")
                if num_future_features > 0:
                    print("      - Future sample (first step):"
                          f" {list(future_data_arr[current_seq_idx][0])}")
                print("      - Target sample (first step):"
                      f" {list(target_data_arr[current_seq_idx][0])}")
                
            current_seq_idx += 1
            
            if verbose >= 2:
                # print(f"\nProcessed sequences for location: {key}")
                print(f"\n    * Total sequences so far: {len(target_data_arr)}")

    if verbose >= 1:
        print("\n[INFO] Final data shapes after reshaping:")
        shape_log = (
            "  [DEBUG] Static Data : "
            f"{static_data_arr.shape if static_data_arr is not None else 'None'}\n"
            "  [DEBUG] Dynamic Data: "
            f"{dynamic_data_arr.shape if dynamic_data_arr is not None else 'None'}\n"
            "  [DEBUG] Future Data : "
            f"{future_data_arr.shape if future_data_arr is not None else 'None'}\n"
            "  [DEBUG] Target Data : "
            f"{target_data_arr.shape if target_data_arr is not None else 'None'}"
        )
        print(shape_log)

    # --- Save to File (Optional) ---
    if savefile:
        if verbose >= 1:
            print(f"\nPreparing to save sequence data to '{savefile}'...")
        job_dict = {
            'static_data': static_data_arr,
            'dynamic_data': dynamic_data_arr,
            'future_data': future_data_arr,
            'target_data': target_data_arr,
            'static_features': static_cols,
            'dynamic_features': dynamic_cols,
            'future_features': future_cols,
            'target_feature': target_col,
            'spatial_features': spatial_cols,
            'dt_col': dt_col,
            'time_steps': time_steps,
            'forecast_horizon': forecast_horizon,
        }
        # Add version information if available
        try:
            job_dict.update(get_versions())
        except NameError: # If get_versions is not defined/imported
            if verbose >=2:
                print("\n  `get_versions` not found, version info not saved.")

        try:
            # save_job to be a wrapper around joblib.dump
            save_job(job_dict, savefile, append_versions=False)
            if verbose >= 1:
                print(f"[INFO] Sequence data dictionary successfully "
                      f"saved to '{savefile}'.")
        except Exception as e:
            if verbose >=1:
                print(f"[ERROR] Failed to save job dictionary to "
                      f"'{savefile}': {e}")

    if verbose>=1: 
        print("\n[INFO] Time‑series data successfully"
              " reshaped into rolling sequences. ")
        
    return static_data_arr, dynamic_data_arr, future_data_arr, target_data_arr

@Deprecated(
    reason=(
        "`reshape_xtft_data_in` is **deprecated** and will be removed in a "
        "future release. Switch to `reshape_xtft_data`, which offers a more "
        "robust, consistent API (supports explicit `forecast_horizon`, "
        "two‑pass pre‑allocation, richer validation, and unified saving)."
    ),
    version="0.2.0",         
    category=DeprecationWarning, 
    alternative="reshape_xtft_data",
)

@isdf
def reshape_xtft_data_in(
    df,
    dt_col,
    target_col,
    dynamic_cols,
    static_cols= None,
    future_cols= None,
    spatial_cols= None, 
    time_steps = 4,
    forecast_horizon= 1, 
    to_datetime = None,
    model="xtft", 
    error="raise", 
    savefile=None, 
    forecast_horizons= None,
    verbose= 3
):
    """
    Reshape data for sequence models (XTFT/TFT) by generating rolling 
    sequences.
    
    This function transforms the input DataFrame into rolling sequences
    suitable for sequence-to-sequence models such as XTFT and TFT. It 
    generates aligned sequences of dynamic, static, and future features 
    along with target values. If spatial columns are provided, the DataFrame 
    is grouped accordingly; otherwise, the entire DataFrame is treated as a 
    single group.
    
    Parameters
    ----------
    df            : `pandas.DataFrame`
        The input DataFrame containing time series data. It must include a 
        datetime column specified by the ``dt_col``.
    dt_col        : `str`
        The name of the datetime column in `df`. This column is processed 
        using the inline method ``ts_validator`` to ensure proper datetime 
        formatting.
    target_col    : `str`
        The column in `df` holding the target values for forecasting.
    dynamic_cols  : `list` or `str`
        A list (or a single string) of column names representing dynamic 
        features that vary over time. These columns are managed by the inline 
        method ``columns_manager``.
    static_cols   : `list` or `str`, optional
        A list (or a single string) of column names representing static 
        features that remain constant within a group. Processed via 
        ``columns_manager``. If omitted, static data is set to ``None``.
    future_cols   : `list` or `str`, optional
        A list (or a single string) of column names representing future 
        covariates used for forecasting. Managed via ``columns_manager``. If 
        not provided, future data is set to ``None``.
    spatial_cols  : `list` or `str`, optional
        Column names used to group the DataFrame by spatial location 
        (e.g., ``"longitude"`` and ``"latitude"``). If omitted, the entire 
        DataFrame is treated as a single group.
    time_steps    : `int`, optional
        The number of consecutive time steps to include in each rolling 
        sequence.
    forecast_horizons : `int`, optional
        The number of future time steps to forecast for each input sequence.
    to_datetime   : `str`, optional
        Specifies the conversion rule for the datetime column. Acceptable 
        values include ``"auto"``, ``"Y"``, ``"M"``, ``"W"``, ``"D"``, 
        ``"H"``, ``"min"``, and ``"s"``.
    model : `str`, optional
        The model type for which data is being reshaped. Supported values 
        include ``"xtft"`` and ``"tft"``. For these models, both static and 
        future features are required; otherwise, consider models such as 
        ``"lstm"``.
    error         : `str`, optional
        Determines the behavior when required features are missing. Options 
        are ``"raise"``, ``"warn"``, or ``"ignore"``.
    savefile: str, or path-like object, optional 
         name of file to store the model.
         The *file* argument must have a write() method that accepts a
         single bytes argument. It can thus be a file object opened for
         binary writing, an io.BytesIO instance, or any other custom
         object that meets this interface.
    verbose       : `int`, optional
        The verbosity level for logging. Levels are:
          - ``1``: Basic information.
          - ``2``: Detailed processing steps.
          - ``3``: Debug-level information including internal shapes.
    
    Returns
    -------
    tuple of `numpy.ndarray` or `None`
        A tuple containing four arrays:
          - ``static_data``  : Array of static feature sequences or 
            ``None`` if static features are not provided.
          - ``dynamic_data`` : Array of dynamic feature sequences.
          - ``future_data``  : Array of future covariate sequences or 
            ``None`` if future features are not provided.
          - ``target_data``  : Array of target value sequences.
    
    Examples
    --------
    >>> from fusionlab.nn.utils import reshape_xtft_data
    >>> static_data, dynamic_data, future_data, target_data = 
    ...     reshape_xtft_data(
    ...         df,
    ...         dt_col           = "date",
    ...         target_col       = "target",
    ...         dynamic_cols     = ["feat1", "feat2"],
    ...         static_cols      = ["static1"],
    ...         future_cols      = ["future1"],
    ...         spatial_cols     = ["longitude", "latitude"],
    ...         time_steps       = 4,
    ...         forecast_horizons= 1,
    ...         to_datetime      = "auto",
    ...         model            = "xtft",
    ...         error            = "raise",
    ...         verbose          = 3
    ...     )
    
    Notes
    -----
    - The function uses inline methods such as ``ts_validator``, 
      ``columns_manager``, and ``exist_features`` for data validation 
      and processing.
    - If ``spatial_cols`` are not provided, the entire DataFrame is treated 
      as one group, and the group key is labeled as 
      ``<Undefined location>``.
    - The rolling sequence generation is based on a sliding window approach, 
      where each input sequence of ``time_steps`` is paired with a target 
      sequence of the subsequent ``forecast_horizons`` rows.
    - For models like XTFT and TFT, both static and future features are 
      critical for optimal performance [1]_.
      
      The function constructs rolling windows for input sequences. For a 
      given time series :math:`\mathbf{X} = [\mathbf{x}_1, \ldots, 
      \mathbf{x}_N]` and target series :math:`\mathbf{Y} = [y_1, \ldots, 
      y_N]`, a rolling window at index :math:`i` is defined as:
      
      .. math::
      
         \mathbf{X}^{(i)} =
         \begin{bmatrix}
           \mathbf{x}_{i} \\
           \mathbf{x}_{i+1} \\
           \vdots \\
           \mathbf{x}_{i+T-1}
         \end{bmatrix}
      
      and the corresponding target is given by:
      
      .. math::
      
         \mathbf{Y}^{(i)} =
         \begin{bmatrix}
           y_{i+T} \\
           y_{i+T+1} \\
           \vdots \\
           y_{i+T+H-1}
         \end{bmatrix}
      
      where :math:`T` is the ``time_steps`` and :math:`H` is the 
      ``forecast_horizons``.
    
    See Also
    --------
    fusionlab.utils.ts_utils.ts_validator :
        Validates and converts the datetime column.
    fusionlab.core.handlers.columns_manager` :
        Formats and validates lists of column names.
    fusionlab.core.checks.exist_features` : 
        Checks for the existence of specified columns in a DataFrame.

    """
    # *************************************************************
    from ..utils.ts_utils import ts_validator
    # *************************************************************
    
    # Backward‐compatibility: accept `forecast_horizons` (deprecated)
    if forecast_horizons is not None:
       warnings.warn(
           "`forecast_horizons` is deprecated and will be removed in a "
           "future release; please use `forecast_horizon` instead.",
           DeprecationWarning,
           stacklevel=2
       )
       # override the new param with the old value
       forecast_horizon = forecast_horizons
       
    # -- Model-specific requirements --
    model= parameter_validator(
        "model", target_strs={"xtft", "tft", "any","lstm", None}
        )(model)
    
    if model in ["xtft", "tft"]:
        msg = (
            f"Using {model.upper()} requires both static_cols"
            " and future_cols to be provided. If these columns"
            " are missing, consider using an alternative model"
            " (e.g., LSTM) that does not depend on these features,"
            " or set ``model='any'``"
        )
        if static_cols is None or future_cols is None:
            if error == "raise":
                raise ValueError(msg)
            elif error == "warn":
                warnings.warn(msg)
            else:
                if verbose >= 1:
                    print("Warning:", msg)
            
    # Validate and convert the datetime column.
    df = ts_validator(
        df,
        dt_col     = dt_col,
        to_datetime= to_datetime,
        as_index   = False,
        verbose    = verbose
    )
    exist_features(df, target_col, name='Target column')
    # Manage spatial columns.
    spatial_cols = columns_manager(
        spatial_cols,
        empty_as_none = False
    )
    if spatial_cols:
        spatial_cols = list(spatial_cols)  # ensure list type
        exist_features(
            df,
            features = spatial_cols,
            name = "Spatial columns"
        )
        group_by_cols = spatial_cols
    else:
        # No spatial columns provided;
        # process entire DataFrame as one group.
        group_by_cols = None

    # Sort the DataFrame.
    if group_by_cols:
        sorted_cols = [dt_col] + group_by_cols
        df = df.sort_values(sorted_cols).reset_index(drop=True)
        grouped = df.groupby(group_by_cols)
    else:
        df = df.sort_values(dt_col).reset_index(drop=True)
        grouped = [(None, df)]

    # Ensure required columns are in list format.
    dynamic_cols = columns_manager(dynamic_cols)
    exist_features(
        df,
        features = dynamic_cols,
        name = "Dynamic columns"
    )
    if static_cols:
        static_cols = columns_manager(static_cols)
        exist_features(
            df,
            features = static_cols,
            name = "Static columns"
        )
    if future_cols:
        future_cols = columns_manager(future_cols)
        exist_features(
            df,
            features = future_cols,
            name = "Future columns"
        )

    # Initialize lists for sequence data.
    static_data  = []
    dynamic_data = []
    future_data  = []
    target_data  = []

    # print("grouped=", grouped)
    # Process each group (location or entire DataFrame).
    for key, group in grouped:
        if not spatial_cols:
            key = "<Undefined location>"  # default key for no spatial info

        # Sort group by datetime.
        group = group.sort_values(dt_col)
        
        min_len = time_steps + forecast_horizon
        if len(group) < min_len:
            if verbose >= 1:
                print(
                    f"\nGroup Warning:\n" 
                    f"     Group '{key}' has {len(group)} data points, "
                    f"which is less than:\n      time_steps ({time_steps}) + "
                    f"forecast_horizon ({forecast_horizon}).\n "
                    f"     Skipping sequence generation for this group."
            )
            continue  # Skip to the next group
                
        # Extract static features if provided.        
        if spatial_cols:
            if static_cols:
                static_values = group.iloc[0][static_cols].values
            else:
                static_values = None

        # Generate rolling sequences.
        for i in range(len(group) - time_steps - forecast_horizon + 1):
            sequence_data = group.iloc[i : i + time_steps]
            dynamic_seq   = sequence_data[dynamic_cols].values

            # Extract static features for the current sequence
            if not spatial_cols:
                if static_cols:
                    static_values = sequence_data.iloc[0][static_cols].values
                else:
                    static_values = None
                
            # Handle future features if provided.
            if future_cols:
                future_seq = np.repeat(
                    sequence_data[future_cols].iloc[0].values.reshape(1, -1),
                    time_steps,
                    axis = 0,
                )
            else:
                future_seq = None

            # Extract target values for forecast horizon.
            target_seq = group.iloc[
                i + time_steps : i + time_steps + forecast_horizon
            ][target_col].values

            if verbose >= 3:
                print(f"\nLocation: {key}")
                if static_cols:
                    print(f"  Static shape           : "
                          f"{static_values.shape}")
                else:
                    print("  Static features not provided")
                print(f"  Dynamic shape          : "
                      f"{dynamic_seq.shape}")
                if future_cols:
                    print(f"  Future shape           : "
                          f"{future_seq.shape}")
                else:
                    print("  Future features not provided")
                print(f"  Target (before reshape): "
                      f"{target_seq.shape}")

            # Reshape target to (forecast_horizons, 1).
            target_seq = target_seq.reshape(forecast_horizon, 1)
            if verbose >= 3:
                print(f"  Target (after reshape) : "
                      f"{target_seq.shape}")

            # Append sequences to lists.
            if static_cols:
                static_data.append(static_values)
            dynamic_data.append(dynamic_seq)
            if future_cols:
                future_data.append(future_seq)
            target_data.append(target_seq)

        if verbose >= 2:
            print(f"\nProcessed sequences for location: {key}")
            print(f"  Total sequences so far: {len(target_data)}")

    # Convert lists to NumPy arrays (or set to None if empty).
    static_data  = np.array(static_data) if static_data else None
    dynamic_data = np.array(dynamic_data) if dynamic_data else None
    future_data  = np.array(future_data) if future_data else None
    target_data  = np.array(target_data) if target_data else None

    if verbose >= 1:
        print("\nFinal data shapes:")
        print(f"  Static Data : {static_data.shape}" if static_data is not None 
              else "  Static Data : None")
        print(f"  Dynamic Data: {dynamic_data.shape}" if dynamic_data is not None 
              else "  Dynamic Data: None")
        print(f"  Future Data : {future_data.shape}" if future_data is not None 
              else "  Future Data : None")
        print(f"  Target Data : {target_data.shape}" if target_data is not None 
              else "  Target Data : None")
    
    if savefile:
        # Prepare the dictionary for saving
        job_dict = {
            'static_data'      : static_data,
            'dynamic_data'     : dynamic_data,
            'future_data'      : future_data,
            'target_data'      : target_data,
            'dynamic_features' : dynamic_cols,
            'static_features'  : static_cols,
            'future_features'  : future_cols,
        }
    
        # Include spatial columns if provided
        if spatial_cols:
            job_dict['spatial_features'] = spatial_cols
    
        # Verbose message before saving
        if verbose >= 1:
            print(
                f"[INFO] Preparing to save job dictionary "
                f"to '{savefile}' ..."
            )
        # Append version to jobdict 
        job_dict.update(get_versions())

        try: 
            # Save job dictionary to the specified file
            save_job(job_dict, savefile, append_versions=False)
        except Exception as e : 
            if verbose >= 1:
                print(
                    f"[ERROR] Failed to save job dictionary "
                    f"to '{savefile}': {str(e)}"
                )
        else: 
            # Verbose message after saving
            if verbose >= 1:
                print(
                    f"[INFO] Job dictionary successfully saved "
                    f"to '{savefile}'."
                )
                
    return static_data, dynamic_data, future_data, target_data

@check_empty(
    params=['train_data', "dynamic_features"],
    none_as_empty=True, 
    allow_none=False
 )
def generate_forecast(
    xtft_model,
    train_data,
    dt_col,           
    dynamic_features,
    future_features=None,
    static_features=None,
    test_data=None,   
    mode="quantile",  
    spatial_cols=None,
    forecast_horizon=4,
    time_steps=3,
    q=None,           
    tname=None,       
    forecast_dt=None, 
    savefile=None,     
    verbose=3, 
    **kw
):
    """
    Generate forecast using the XTFT model.
    
    This function uses a pre-trained Keras model to forecast future 
    values based on provided historical data. The model receives three 
    inputs: `X_static`, `X_dynamic`, and `X_future` re-built from 
    `train_data`, and outputs predictions over a specified
    forecast horizon.
    
    See more in :ref:`User Guide <user_guide>`. 
    
    Parameters
    ----------
    xtft_model : object
        A validated Keras model instance. It is processed by the 
        ``validate_keras_model`` method [1]_.
    train_data : pandas.DataFrame
        The training data containing historical records. Must include 
        the `dt_col` and all required feature columns.
    dt_col : str
        Name of the column representing time. It may be a datetime or 
        numeric column (e.g. ``"year"``).
    dynamic_features : list of str
        List of dynamic feature column names. They are formatted via 
        ``columns_manager``.
    future_features  : list of str, optional
        List of future feature names. These columns are tiled over the 
        forecast horizon.
    static_features  : list of str, optional
        List of static feature names. If not provided, a dummy input is 
        used.
    test_data : pandas.DataFrame, optional
        DataFrame containing actual values used for evaluation. If 
        provided, it is used to compute the R² and coverage score for 
        ``mode='quantile'``.
    mode : str, optional
        Forecast mode. Must be either ``"quantile"`` or ``"point"``. In 
        ``quantile`` mode, predictions for multiple quantiles (default: 
        [0.1, 0.5, 0.9]) are computed.
    spatial_cols : list of str, optional
        List of spatial column names for grouping the data. When provided,
        forecasts are computed per location; otherwise, a global 
        forecast is performed.
    forecast_horizon : int, optional
        Number of future periods to forecast. Default is 4.
    time_steps   : int, optional
        Number of past time steps to use as input for the model. Default 
        is 3.
    q : list of float, optional
        List of quantiles for use in ``quantile`` mode. Default is 
        [0.1, 0.5, 0.9]. Each quantile is validated by the 
        ``assert_ratio`` function.
    tname : str, optional
        Target variable name used for constructing forecast result 
        columns. Defaults to ``"target"``.
    forecast_dt  : list or str, optional
        List of forecast dates or ``"auto"`` to derive dates from `dt_col`. 
        In auto mode, if `dt_col` is datetime, frequency is inferred using 
        ``pd.infer_freq``.
    savefile     : str, optional
        Path to the CSV file where forecast results will be saved. If not 
        provided, a default filename is generated.
    verbose      : int, optional
        Verbosity level (0-7). Controls the amount of execution output.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the forecast results. In ``quantile`` mode, 
        each forecast period includes columns for each quantile; in 
        ``point`` mode, a single prediction column is provided.
    
    Examples
    --------
    (1) Example refering to Train data only 
    
    >>> import os 
    >>> import pandas as pd
    >>> import numpy as np
    >>> from fusionlab.nn.transformers import XTFT
    >>> from fusionlab.nn.losses import combined_quantile_loss
    >>> from fusionlab.nn.utils import generate_forecast
    >>> 
    >>> # Create a dummy training DataFrame with a date column,
    >>> # dynamic features "feat1", "feat2", static feature "stat1",
    >>> # and target "price".
    >>> date_rng = pd.date_range(start="2020-01-01", periods=50, freq="D")
    >>> train_df = pd.DataFrame({
    ...     "date": date_rng,
    ...     "feat1": np.random.rand(50),
    ...     "feat2": np.random.rand(50),
    ...     "stat1": np.random.rand(50),
    ...     "price": np.random.rand(50)
    ... })
    >>> 
    >>> # Prepare a dummy XTFT model with example parameters.
    >>> # Note: The model expects the following input shapes:
    >>> # - X_static: (n_samples, static_input_dim)
    >>> # - X_dynamic: (n_samples, time_steps, dynamic_input_dim)
    >>> # - X_future:  (n_samples, time_steps, future_input_dim)
    >>> my_model = XTFT(
    ...     static_input_dim=1,           # "stat1"
    ...     dynamic_input_dim=2,          # "feat1" and "feat2"
    ...     future_input_dim=1,           # features provided for dim1
    ...     forecast_horizon=5,           # Forecasting 5 periods ahead
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     embed_dim=16,
    ...     max_window_size=3,
    ...     memory_size=50,
    ...     num_heads=2,
    ...     dropout_rate=0.1,
    ...     lstm_units=32,
    ...     attention_units=32,
    ...     hidden_units=16
    ... )
    >>> my_model.compile(optimizer="adam")
    >>> 
    >>> # Create dummy input arrays for model fitting.
    >>> # For simplicity, assume time_steps = 3 and use random data.
    >>> X_static = train_df[["stat1"]].values      # shape: (50, 1)
    >>> # Create a dummy dynamic input array of shape (50, 3, 2)
    >>> X_dynamic = np.random.rand(50, 3, 2)
    >>> # Create a dummy features 
    >>> X_future = np.random.rand(50, 3, 1)
    >>> # Create dummy target output from "price"
    >>> y_array = train_df["price"].values.reshape(50, 1, 1)
    >>> 
    >>> # Fit the model on the dummy data.
    >>> my_model.fit(
    ...     x=[X_static, X_dynamic, X_future],
    ...     y=y_array,
    ...     epochs=1,
    ...     batch_size=8
    ... )
    >>> 
    >>> # Generate forecast using the generate_forecast function.
    >>> forecast = generate_forecast(
    ...     xtft_model=my_model,
    ...     train_data=train_df,
    ...     dt_col="date",
    ...     dynamic_features=["feat1", "feat2"],
    ...     static_features=["stat1"],
    ...     forecast_horizon=5,
    ...     time_steps=3,
    ...     tname="price",
    ...     mode="quantile",
    ...     verbose=3
    ... )
    >>> print(forecast.head())
    
    (2) Example refering to Test data included.
    
    >>> # Create a dummy DataFrame with a date column,
    >>> # two dynamic features ("feat1", "feat2"), one static feature ("stat1"),
    >>> # and target "price".
    >>> date_rng = pd.date_range(start="2020-01-01", periods=60, freq="D")
    >>> data = {
    ...     "date": date_rng,
    ...     "feat1": np.random.rand(60),
    ...     "feat2": np.random.rand(60),
    ...     "stat1": np.random.rand(60),
    ...     "price": np.random.rand(60)
    ... }
    >>> df = pd.DataFrame(data)
    >>> 
    >>> # Split the DataFrame into training and test sets.
    >>> # Training data: dates before 2020-02-01
    >>> # Test data: dates from 2020-02-01 onward.
    >>> train_df = df[df["date"] < "2020-02-01"].copy()
    >>> test_df  = df[df["date"] >= "2020-02-01"].copy()
    >>> 
    >>> # Create dummy input arrays for model fitting.
    >>> # Assume time_steps = 3.
    >>> X_static = train_df[["stat1"]].values      # Shape: (n_train, 1)
    >>> X_dynamic = np.random.rand(len(train_df), 3, 2)
    >>> X_future  = np.random.rand(len(train_df), 3, 1)
    >>> # Create dummy target output from "price".
    >>> y_array   = train_df["price"].values.reshape(len(train_df), 1, 1)
    >>> 
    >>> # Instantiate a dummy XTFT model.
    >>> my_model = XTFT(
    ...     static_input_dim=1,           # "stat1"
    ...     dynamic_input_dim=2,          # "feat1" and "feat2"
    ...     future_input_dim=1,           # For the provided future feature
    ...     forecast_horizon=5,           # Forecasting 5 periods ahead
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     embed_dim=16,
    ...     max_window_size=3,
    ...     memory_size=50,
    ...     num_heads=2,
    ...     dropout_rate=0.1,
    ...     lstm_units=32,
    ...     attention_units=32,
    ...     hidden_units=16
    ... )
    >>> loss_fn = combined_quantile_loss(my_model.quantiles) 
    >>> my_model.compile(optimizer="adam", loss=loss_fn)
    >>> 
    >>> # Fit the model on the training data.
    >>> my_model.fit(
    ...     x=[X_static, X_dynamic, X_future],
    ...     y=y_array,
    ...     epochs=1,
    ...     batch_size=8, 
    ...     callbacks = [early_stopping, model_checkpoint]
    ... )
    >>> 
    >>> # Generate forecast using the generate_forecast function.
    >>> # This example uses test_df for evaluation, which will compute 
    >>> # metrics like R² Score and Coverage Score.
    >>> forecast = generate_forecast(
    ...     xtft_model=my_model,
    ...     train_data=train_df,
    ...     dt_col="date",
    ...     dynamic_features=["feat1", "feat2"],
    ...     static_features=["stat1"],
    ...     test_data=test_df.iloc[:5, :], # to fit the first horizon forecasting.
    ...     forecast_horizon=5,
    ...     time_steps=3,
    ...     tname="price",
    ...     mode="quantile",
    ...     verbose=3
    ... )
    >>> print(forecast.head())

    (3) Example of Point forecasting 

    >>> # Create a dummy training DataFrame with a date column,
    >>> # two dynamic features ("feat1", "feat2"), one static feature ("stat1"),
    >>> # and target "price".
    >>> date_rng = pd.date_range(start="2020-01-01", periods=50, freq="D")
    >>> train_df = pd.DataFrame({
    ...     "date": date_rng,
    ...     "feat1": np.random.rand(50),
    ...     "feat2": np.random.rand(50),
    ...     "stat1": np.random.rand(50),
    ...     "price": np.random.rand(50)
    ... })
    >>> 
    >>> # Create dummy input arrays for model fitting.
    >>> # X_static is derived from the static feature "stat1".
    >>> X_static = train_df[["stat1"]].values      # shape: (50, 1)
    >>> 
    >>> # X_dynamic is a dummy dynamic array for "feat1" and "feat2".
    >>> # For time_steps = 3, its shape is (50, 3, 2).
    >>> X_dynamic = np.random.rand(50, 3, 2)
    >>> 
    >>> # X_future is a dummy array for future features.
    >>> # Here, we assume a single future feature with shape (50, 3, 1).
    >>> X_future = np.random.rand(50, 3, 1)
    >>> 
    >>> # Create dummy target output from "price".
    >>> y_array = train_df["price"].values.reshape(50, 1, 1)
    >>> 
    >>> # Instantiate a dummy XTFT model.
    >>> my_model = XTFT(
    ...     static_input_dim=1,           # "stat1"
    ...     dynamic_input_dim=2,          # "feat1" and "feat2"
    ...     future_input_dim=1,           # Provided future feature
    ...     forecast_horizon=5,           # Forecast 5 periods ahead
    ...     quantiles=None,    # [0.1, 0.5, 0.9] Not used in point mode
    ...     embed_dim=16,
    ...     max_window_size=3,
    ...     memory_size=50,
    ...     num_heads=2,
    ...     dropout_rate=0.1,
    ...     lstm_units=32,
    ...     attention_units=32,
    ...     hidden_units=16
    ... )
    >>> my_model.compile(optimizer="adam")
    >>> 
    >>> # Fit the model on the dummy data.
    >>> my_model.fit(
    ...     x=[X_static, X_dynamic, X_future],
    ...     y=y_array,
    ...     epochs=1,
    ...     batch_size=8
    ... )
    >>> 
    >>> # Generate forecast using the generate_forecast function in point mode.
    >>> forecast = generate_forecast(
    ...     xtft_model=my_model,
    ...     train_data=train_df,
    ...     dt_col="date",
    ...     dynamic_features=["feat1", "feat2"],
    ...     static_features=["stat1"],
    ...     forecast_horizon=5,
    ...     time_steps=3,
    ...     tname="price",
    ...     mode="point",
    ...     verbose=3
    ... )
    >>> print(forecast.head())

    Notes
    -----
    The function groups data by `spatial_cols` if provided, and 
    formats features via ``columns_manager``. It validates the time 
    column using ``check_datetime`` and uses dummy inputs for missing 
    static or future features. The forecast is produced by invoking 
    ``xtft_model.predict`` on a list containing static, dynamic, and 
    future inputs. The predictions are generated as follows:
    
    .. math::
    
       \hat{y}_{t+i} = f\Bigl(X_{\text{static}},\;
       X_{\text{dynamic}},\; X_{\text{future}}\Bigr)
    
    where :math:`i` denotes the forecast period.
    
    See Also
    --------
    fusionlab.nn.utils.reshape_xtft_data:
        Function to reshape data for XTFT models.
    fusionlab.utils.validator.validate_keras_model:
        Function to validate Keras model compatibility.
    fusionlab.core.handlers.columns_manager: 
        Utility to manage and format column names.
    fusionlab.core.checks.check_datetime: 
        Function to check and validate datetime columns.
    fusionlab.core.checks.check_spatial_columns: 
        Function to validate spatial columns in data.
    fusionlab.core.checks.assert_ratio: 
        Function to validate and assert ratio values.
    fusionlab.metrics_special.coverage_score: 
        Function to compute coverage score for quantile predictions.

    References
    ----------
    .. [1] Kouadio et al., "Gofast Forecasting Model", Journal of 
       Advanced Forecasting, 2025. (In Review)
    """
    # **********************************************************
    from ..metrics import coverage_score
    # **********************************************************
    # Validate the model
    xtft_model = validate_keras_model(
        xtft_model,
        deep_check=True
    )
    xtft_model = check_keras_model_status(
        xtft_model, 
        ops="validate", 
        mode="fit"
    )
    
    if verbose >= 1:
        print(
            "\nGenerating {} forecast for {} periods..."
            .format(mode, forecast_horizon)
        )

    # Format features
    dynamic_features = columns_manager(
        dynamic_features, empty_as_none=False
    )
    static_features = (
        columns_manager(static_features)
        if static_features is not None else None
    )
    future_features = (
        columns_manager(future_features)
        if future_features is not None else None
    )

    if spatial_cols:
        spatial_cols = columns_manager(spatial_cols)
        check_spatial_columns(train_data, spatial_cols)

    # Set default quantiles if not provided
    q=check_forecast_mode(mode, q, error="warn", ops="validate")
    
    if q is None:
        q = [0.1, 0.5, 0.9]
    
    q = columns_manager(q)
    q = [assert_ratio(
        r,bounds=(0, 1),exclude_values=[0, 1], 
        name=f"quantile '{r}'")for r in q
    ]

    # Set default target name
    if tname is None:
        tname = "target"

    # Check dt_col data type
    check_datetime(
        train_data,
        dt_cols= dt_col, 
        ops="check_only",
        consider_dt_as="numeric",
        accept_dt=True, 
        allow_int=True, 
    )

    # Determine forecast dates; if None, set to "auto"
    if forecast_dt is None:
        forecast_dt = "auto"
    if forecast_dt == "auto":
        last_date = train_data[dt_col].max()
        if pd.api.types.is_datetime64_any_dtype(
                train_data[dt_col]):
            inferred_freq = pd.infer_freq(
                train_data[dt_col].sort_values()
            )
            if inferred_freq is None:
                inferred_freq = 'D'
                if verbose >= 2:
                    print(
                        "Could not infer frequency; defaulting "
                        "to daily."
                    )
            forecast_dt = pd.date_range(
                start=last_date +
                pd.Timedelta(1, unit=inferred_freq[0]),
                periods=forecast_horizon
            ).tolist()
        else:
            forecast_dt = [
                last_date + i
                for i in range(1, forecast_horizon + 1)
            ]
    if verbose >= 2:
        print(
            "Forecasting for dates/periods: {}"
            .format(forecast_dt)
        )

    # Handle situations where forecast_dt
    # length differs from forecast_horizon
    if len(forecast_dt) < forecast_horizon:
        # Case 1: forecast_dt has fewer dates than forecast_horizon.
        missing_dates = forecast_horizon - len(forecast_dt)
        
        # Assuming the last date is the starting point for new dates.
        last_date = forecast_dt[-1]
        
        # Append missing dates based on inferred frequency or context.
        if pd.api.types.is_datetime64_any_dtype(train_data[dt_col]):
            inferred_freq = pd.infer_freq(
                train_data[dt_col].sort_values()
            )
            
            if inferred_freq:
                new_dates = [
                    last_date + pd.Timedelta(days=i)
                    for i in range(1, missing_dates + 1)
                ]
            else:
                new_dates = [
                    last_date + pd.Timedelta(days=i)
                    for i in range(1, missing_dates + 1)
                ]
                
            forecast_dt.extend(new_dates)
            
            # Issue a warning that the user might 
            # want to double-check the forecast dates.
            warnings.warn(
                f"The provided forecast_dt is shorter than the"
                f" forecast_horizon ({len(forecast_dt)} dates"
                f" instead of {forecast_horizon}). "
                f"The missing dates have been filled with sequential"
                f" dates starting from {last_date}.",
                category=UserWarning
            )
    
    elif len(forecast_dt) > forecast_horizon:
        # Case 2: forecast_dt has more dates than forecast_horizon.
        # Trim the forecast_dt list
        forecast_dt = forecast_dt[:forecast_horizon]  
        # Issue a verbose message to indicate
        # the truncation of forecast dates.
        if verbose >= 1:
            print(
                f"forecast_dt contained more dates than"
                " forecast_horizon. Only the first"
                f" {forecast_horizon} dates have been used."
            )

    # Determine iteration base: group by spatial_cols or use global
    if spatial_cols:
        unique_locations = (
            train_data[spatial_cols]
            .drop_duplicates()
            .reset_index(drop=True)
        )
    else:
        unique_locations = pd.DataFrame({"global": [0]})

    forecast_results = []
    chunk_size = 100_000  # Adjust based on available memory
    start_time = time.time()
    # Iterate over each location or global forecast
    for idx, loc in unique_locations.iterrows():
        # Print a status message if verbosity is at least 1
        if verbose >= 1:
            # Number of locations processed so far
            completed = idx + 1
            total = len(unique_locations)
            remaining = total - completed
    
            # Calculate elapsed time
            elapsed = time.time() - start_time
    
            # If we have processed at least one location,
            # estimate average time per location
            if completed > 0:
                avg_time_per_loc = elapsed / completed
                eta_seconds = avg_time_per_loc * remaining
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            else:
                # Before any iteration is complete, no ETA
                eta_str = "N/A"
    
            print(
                f"[INFO] Processing location {completed}/{total} - "
                f"({remaining} remaining). ETA: {eta_str}"
            )
            
        if spatial_cols:
            condition = np.ones(len(train_data), dtype=bool)
            for col in spatial_cols:
                condition &= (train_data[col] == loc[col])
            location_data = (
                train_data[condition]
                .sort_values(dt_col)
                .iloc[-time_steps:]
            )
        else:
            location_data = (
                train_data.sort_values(dt_col)
                .iloc[-time_steps:]
            )
            loc = {}  # dummy global location

        if len(location_data) < time_steps:
            try:
                loc_str = (tuple(loc.values())
                           if spatial_cols else "global")
            except: 
                loc_str = (tuple(loc.values)
                           if spatial_cols else "global")
            if verbose >= 2:
                print(
                    "Skipping {} - Insufficient data (requires {} "
                    "steps).".format(loc_str, time_steps)
                )
            continue

        # Prepare model inputs
        model_inputs = {}

        # Static features: use last row or dummy array
        dummy=[]
        if static_features:
            X_static_forecast = (
                location_data.iloc[-1][static_features]
                .values.reshape(1, -1)
            )
        else:
            X_static_forecast = np.zeros((1, 1))
            dummy.append("static")
            
        model_inputs['static'] = X_static_forecast

        # Dynamic features: use last time_steps rows
        X_dynamic_forecast = (
            location_data[dynamic_features]
            .values.reshape(1, time_steps, -1)
        )
        model_inputs['dynamic'] = X_dynamic_forecast

        # Future features: tile first row or dummy input
        if future_features:
            X_future_forecast = (
                np.tile(
                    location_data[future_features].iloc[0].values,
                    (time_steps, 1)
                ).reshape(1, time_steps, -1)
            )
        else:
            X_future_forecast = np.zeros((1, time_steps, 1))
            dummy.append("future")
            
        model_inputs['future'] = X_future_forecast
        
        # warn at once: idx==0 is for consistency. 
        if len(dummy) != 0 and idx==0:
            warnings.warn(
                "Expected three inputs for the transformer model."
                " Proceeding with the following dummy input(s) '{}'"
                " may result in suboptimal performance or"
                " unexpected behavior. Use at your own risk."
                .format(", ".join(dummy))
            )

        # Run prediction
        try:
            y_pred_forecast = xtft_model.predict(
                [
                    np.asarray(model_inputs['static'], dtype=np.float32),
                    np.asarray(model_inputs['dynamic'], dtype=np.float32),
                    np.asarray(model_inputs['future'], dtype=np.float32)
                ]
            )
        except Exception as e:
            loc_str = (tuple(loc.values())
                       if spatial_cols else "global")
            print(
                "[ERROR] Error predicting for {}: {}"
                .format(loc_str, str(e))
            )
            continue
        else:
            loc_str = (tuple(loc.values)
                       if spatial_cols else "global")
            if verbose >= 3:
                print(
                    "Predicted for {} - Shape: {}"
                    .format(loc_str, y_pred_forecast.shape)
                )

        # Build forecast entries for each forecast period
        for i, period in enumerate(forecast_dt):
            if verbose >= 2:
                print( "[DEBUG] Building forecast entry"
                      f" for period {period} (index {i})."
                     )
            forecast_entry = {}
            if spatial_cols:
                for col in spatial_cols:
                    forecast_entry[col] = loc[col]
                    if verbose >= 5:
                        print(f"  [TRACE] Setting '{col}' to {loc[col]}.")
                        
            forecast_entry[dt_col] = period
            
            if verbose >= 3:
                print(f"  [TRACE] Setting '{dt_col}' to {period}.")
                
            if mode == "quantile":
                for qi, quantile in enumerate(q):
                    col_name = "{}_q{}".format(
                        tname, int(round(quantile * 100))
                    )
                    forecast_entry[col_name] = y_pred_forecast[0, i, qi].item()
                    
                    if verbose >= 5:
                        print(f"  [TRACE] Setting '{col_name}'"
                              f" to {forecast_entry[col_name]}.")
            else:
                col_name = "{}_pred".format(tname)
                forecast_entry[col_name] = y_pred_forecast[0, i, 0].item()
                
                if verbose >= 5:
                    print(f"  [TRACE] Setting '{col_name}'"
                          f" to {forecast_entry[col_name]}.")

            forecast_results.append(forecast_entry)
            
        if verbose >= 3:
            print("[DEBUG] Completed entry for index"
                  f" {i}: {forecast_entry}")
            
 
    # Optionally, provide a status message before building
    if verbose >= 1:
        if len(forecast_results) >= chunk_size:
            print(
                "[INFO] Constructing a large DataFrame"
                f" (~{len(forecast_results):,} rows). "
                 "This may take a while..."
            )
        else: 
            print(
                f"[INFO] Constructing DataFrame from "
                f"{len(forecast_results):,} rows..."
            )
    
    start_time = time.time()  # Begin timing the DataFrame construction
    
    if len(forecast_results) < chunk_size:
        if verbose >= 3:
            print(
                "  [DEBUG] Row count is below the specified chunk_size."
                " Using standard DataFrame construction."
            )
        forecast_df = pd.DataFrame(forecast_results)
    else:
        if verbose >= 3:
            print(
                "  [DEBUG] Attempting chunk-based DataFrame"
                " construction via 'build_large_df'..."
            )
        try:
            forecast_df = build_large_df(
                forecast_results=forecast_results,
                dt_col=dt_col,
                tname=tname,
                chunk_size=chunk_size,
                spatial_cols=spatial_cols,
                verbose=verbose
            )
        except Exception as e:
            warnings.warn(
                "Chunk-based DataFrame construction failed. "
                "Falling back to direct construction:\n"
                f"{type(e).__name__}: {str(e)}",
                category=RuntimeWarning
            )
            forecast_df = pd.DataFrame(forecast_results)
    
    build_time = time.time() - start_time  # Calculate elapsed time
    
    if verbose >= 1:
        if len(forecast_results) >= chunk_size:
            print(
                "\n[INFO] Successfully created a large DataFrame with"
                f" {len(forecast_df):,} rows in {build_time:.2f}"
                " seconds. If you need further processing, plan"
                " for the additional time accordingly."
            )
        else:
            print("\n[INFO] DataFrame construction completed"
                  f" in {build_time:.2f} seconds.")
            
        print("\nForecasting completed. Sample results:")
        print(forecast_df.head())

    if savefile is None:
        savefile = "{}_forecast_{}_results.csv"\
                   .format(mode, tname)

    if forecast_df.empty: 
        e_msg=(
            f"Insufficient data for time equals to {time_steps} steps."
            " Consider reducing the time steps or provide enough data."
            )
        raise NotEnoughDataError(e_msg)
   
    
    forecast_df.to_csv(savefile, index=False)
    if verbose >= 1:
        print(
            "Forecast results saved to: {}"
            .format(savefile)
        )
     
    # Evaluation if test_data is provided
    if test_data is not None:
        # Obtain unique evaluation dates from test_data 
        # and forecast_df,then sort them.
        eval_dates_test = np.sort(test_data[dt_col].unique())
        eval_dates_forecast = np.sort(
            forecast_df[dt_col].unique()
        )
        
        # Compute the intersection of dates available in both
        # DataFrames.
        eval_dates = np.intersect1d(
            eval_dates_test, eval_dates_forecast
        )
        if verbose >=3:
            print("  [DEBUG] Common evaluation dates:\n", eval_dates)
        
        # If the number of common dates exceeds forecast_horizon,
        # warn the user and select only the first forecast_horizon
        # dates.
        if len(eval_dates) > forecast_horizon:
            warnings.warn(
                f"The number of unique evaluation dates "
                f"({len(eval_dates)}) in test_data exceeds the "
                f"forecast_horizon ({forecast_horizon}). Only"
                f" the first {forecast_horizon} dates will"
                " be used for evaluation."
            )
            eval_dates = eval_dates[:forecast_horizon]
        # If fewer common dates than forecast_horizon are found,
        # warn the user that evaluation will proceed with available dates.
        elif len(eval_dates) < forecast_horizon:
            warnings.warn(
                f"Only {len(eval_dates)} unique evaluation"
                " dates were found, which is less than the"
                f" forecast_horizon ({forecast_horizon})."
                " Evaluation will be performed on the"
                " available dates."
            )
        
        # Filter forecast_df and test_data to only include rows with
        # the common evaluation dates.
        forecast_eval = forecast_df[
            forecast_df[dt_col].isin(eval_dates)
        ]
        # forecast_eval = forecast_df[
        #     forecast_df[dt_col] == eval_dates
        # ]
        test_data_filtered = test_data[
            test_data[dt_col].isin(eval_dates)
        ]

        if spatial_cols:
            exist_features(test_data, spatial_cols)
            test_data_sorted = (
                test_data_filtered.sort_values(by=spatial_cols)
                .reset_index(drop=True)
            )
            forecast_eval_sorted = (
                forecast_eval.sort_values(by=spatial_cols)
                .reset_index(drop=True)
            )
        else:
            test_data_sorted = (
                test_data_filtered.sort_values(by=dt_col)
                .reset_index(drop=True)
            )
            forecast_eval_sorted = (
                forecast_eval.sort_values(by=dt_col)
                .reset_index(drop=True)
            )

        actual = test_data_sorted[tname].values
        pred_col = (f"{tname}_q50"
                    if mode == "quantile"
                    else f"{tname}_pred")
        predicted = forecast_eval_sorted[pred_col].values
   
        try:
            r2 = r2_score(actual, predicted)
            print(
                "[INFO] XTFT Model R² Score: {:.4f}"
                .format(r2)
            )
        except Exception as e:
            print(
                "[ERROR] Error computing R² Score: {}"
                .format(str(e))
            )

        if mode == "quantile":
            try:
                lower_col = "{}_q{}".format(
                    tname, int(round(q[0] * 100))
                )
                upper_col = "{}_q{}".format(
                    tname, int(round(q[-1] * 100))
                )
                cov = coverage_score(
                    y_true=actual,
                    y_lower=forecast_eval_sorted[lower_col],
                    y_upper=forecast_eval_sorted[upper_col]
                )
                print(
                    "[INFO] Coverage Score: {:.4f}"
                    .format(cov)
                )
            except Exception as e:
                print(
                    "[ERROR] Error computing Coverage Score: {}"
                    .format(str(e))
                )

    return forecast_df

def forecast_single_step(
    xtft_model,
    inputs, 
    y=None,
    dt_col=None,
    mode="quantile",
    spatial_cols=None,
    q=None,
    tname=None,
    forecast_dt=None,
    apply_mask=False,
    mask_values=None,
    mask_fill_value=None,
    savefile=None,
    verbose=3, 
    **kws
):
    """
    Generate a single-step forecast using the XTFT model.
    
    This function generates a forecast for a single future time step
    using a pre-trained XTFT deep learning model. The model takes three
    inputs: `X_static`, `X_dynamic`, and `X_future`, and produces a
    prediction according to the formulation:
    
    .. math::
    
        \hat{y}_{t+1} = f\Bigl( X_{\text{static}},\;
        X_{\text{dynamic}},\; X_{\text{future}} \Bigr)
    
    where :math:`f` is the trained XTFT model. The predictions can be
    either quantile-based or point-based, as determined by the `mode`
    parameter.
    
    Parameters
    ----------
    xtft_model : object
        A validated Keras model instance. The model is expected to be
        verified via ``validate_keras_model``.
    inputs : list or tuple of numpy.ndarray
        A list containing three elements: ``X_static``, ``X_dynamic``,
        and ``X_future``. If ``spatial_cols`` is provided, it is assumed
        that the first column of ``X_static`` corresponds to the first
        spatial coordinate and the second column to the second spatial
        coordinate of the original training data.
    y : numpy.ndarray, optional
        Actual target values. If provided, evaluation metrics such as
        R² Score and (in quantile mode) the coverage score are computed.
    dt_col : str, optional
        Name of the time column (e.g. ``"year"``). If provided, a column
        with this name is added to the output DataFrame. The actual time
        values must be supplied externally.
    mode : str, optional
        Forecast mode. Must be either ``"quantile"`` or ``"point"``.
        In quantile mode, predictions are generated for multiple
        quantiles (default: ``0.1``, ``0.5``, and ``0.9``).
    spatial_cols : list of str, optional
        List of spatial column names. If provided, it must contain at least
        two elements and correspond to the first and second columns of the
        original training data's ``X_static``.
    q : list of float, optional
        List of quantiles for quantile forecasting. Default is
        ``[0.1, 0.5, 0.9]`` when `mode` is ``"quantile"``.
    tname : str, optional
        Target variable name for predictions. This name is used to
        construct output column names (e.g. ``"subsidence"``). Default is
        ``"target"``.
    forecast_dt : any, optional
        Forecast datetime information. Not used in this function but may be
        provided for compatibility.
    apply_mask : bool, optional
        If True, applies a masking function (``mask_by_reference``) to
        replace predictions in non-subsiding areas. Requires that both
        ``mask_values`` and ``mask_fill_value`` are provided.
    mask_values : scalar, optional
        Reference value(s) used for masking. Must be provided if
        ``apply_mask`` is True.
    mask_fill_value : scalar, optional
        Value used to fill masked predictions. Must be provided if
        ``apply_mask`` is True.
    savefile : str, optional
        Path to a CSV file where the forecast results will be saved.
        If not provided, a default filename is generated.
    verbose : int, optional
        Verbosity level controlling printed output. Higher values result in
        more detailed output.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the forecast results. In quantile mode,
        the output includes columns for each quantile (e.g.
        ``<tname>_q10``, ``<tname>_q50``, ``<tname>_q90``). In point mode,
        a single prediction column (``<tname>_pred``) is provided. If `y`
        is provided, an additional column with the actual target values
        (``<tname>_actual``) is included.
    
    Examples
    --------

    >>> from fusionlab.nn.transformers import XTFT
    >>> from fusionlab.nn.utils import forecast_single_step
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create a dummy training DataFrame with a date column,
    >>> # two dynamic features ("feat1", "feat2"), a static feature ("stat1"),
    >>> # and dummy spatial features ("longitude", "latitude"), as well as the
    >>> # target variable "subsidence".
    >>> date_rng = pd.date_range(start="2020-01-01", periods=50, freq="D")
    >>> train_df = pd.DataFrame({
    ...     "date": date_rng,
    ...     "longitude": np.random.uniform(-180, 180, 50),
    ...     "latitude": np.random.uniform(-90, 90, 50),
    ...     "feat1": np.random.rand(50),
    ...     "feat2": np.random.rand(50),
    ...     "stat1": np.random.rand(50),
    ...     "subsidence": np.random.rand(50)
    ... })
    >>> 
    >>> # Prepare dummy inputs for the model.
    >>> # For the static input, combine the spatial feature "longitude" and the
    >>> # static feature "stat1". The forecast_single_step function expects that,
    >>> # if spatial_cols is provided, the first two columns of X_static correspond
    >>> # to the spatial coordinates.
    >>> X_static = train_df[["longitude", "stat1"]].values   # shape: (50, 2)
    >>> 
    >>> # Create a dummy dynamic input array for "feat1" and "feat2".
    >>> # Assume time_steps = 3, so the shape is (50, 3, 2).
    >>> X_dynamic = np.random.rand(50, 3, 2)
    >>> 
    >>> # Create a dummy future input array.
    >>> # For this example, assume one future feature with shape (50, 3, 1).
    >>> X_future = np.random.rand(50, 3, 1)
    >>> 
    >>> # Create dummy target output from "subsidence", reshaped to (50, 1, 1)
    >>> y_array = train_df["subsidence"].values.reshape(50, 1, 1)
    >>> 
    >>> # Instantiate a dummy XTFT model.
    >>> # The model expects:
    >>> #   - X_static with shape (n_samples, static_input_dim)
    >>> #   - X_dynamic with shape (n_samples, time_steps, dynamic_input_dim)
    >>> #   - X_future with shape (n_samples, time_steps, future_input_dim)
    >>> my_model = XTFT(
    ...     static_input_dim=2,         # "longitude" and "stat1"
    ...     dynamic_input_dim=2,        # "feat1" and "feat2"
    ...     future_input_dim=1,         # One future feature
    ...     forecast_horizon=1,         # Single-step forecast
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     embed_dim=16,
    ...     max_window_size=3,
    ...     memory_size=50,
    ...     num_heads=2,
    ...     dropout_rate=0.1,
    ...     lstm_units=32,
    ...     attention_units=32,
    ...     hidden_units=16
    ... )
    >>> my_model.compile(optimizer="adam")
    >>> 
    >>> # Fit the model on the dummy data.
    >>> my_model.fit(
    ...     x=[X_static, X_dynamic, X_future],
    ...     y=y_array,
    ...     epochs=1,
    ...     batch_size=8
    ... )
    >>> 
    >>> # Package the inputs as expected by forecast_single_step.
    >>> inputs = [X_static, X_dynamic, X_future]
    >>> 
    >>> # Generate a single-step quantile forecast.
    >>> forecast_df = forecast_single_step(
    ...     xtft_model=my_model,
    ...     inputs=inputs,
    ...     y=y_array,
    ...     dt_col="date",                # The time column name in the output
    ...     mode="quantile",              # Can be "quantile" or "point"
    ...     spatial_cols=["longitude", "latitude"],
    ...     q=[0.1, 0.5, 0.9],
    ...     tname="subsidence",
    ...     apply_mask=True,
    ...     mask_values=0,
    ...     mask_fill_value=0,
    ...     verbose=3
    ... )
    >>> print(forecast_df.head())

    
    Notes
    -----
    - In quantile mode, the function computes predictions for multiple
      quantiles and uses the median (``0.5``) for evaluation.
    - If ``spatial_cols`` is provided, it must be the first and second
      columns of the original training data's ``X_static``.
    - The function internally utilizes ``validate_keras_model`` for model
      validation, ``assert_ratio`` for quantile verification, and
      ``mask_by_reference`` for masking operations.
    - Evaluation metrics such as R² Score and Coverage Score are computed
      if actual target values (`y`) are provided.
    - The prediction output is expected to have the shape
      :math:`(n, 1, m)`, where :math:`m` is the number of outputs (e.g., the
      number of quantiles in quantile mode, or 1 in point mode) [1]_.
    
    See Also
    --------
    generate_forecast_multi_step : Function for multi-step forecasts.
    coverage_score            : Function to compute the coverage score.
    validate_keras_model      : Function to validate a Keras model.
    assert_ratio              : Function to validate quantile ratios.
    
    """
    # **********************************************************
    from ..metrics import coverage_score
    # **********************************************************
    
    # Validate inputs: expect a list/tuple of three elements.
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
        raise ValueError(
            "inputs must be a list or tuple containing "
            "[X_static, X_dynamic, X_future]."
        )
    X_static, X_dynamic, X_future = inputs

    # Validate model
    xtft_model = validate_keras_model(xtft_model, deep_check=True)
    if verbose >= 1:
        print("\nGenerating single-step forecast in mode:",
              mode)

    # Set default quantiles for quantile mode.
    q=check_forecast_mode(mode, q, error="warn", ops="validate")
    if mode == "quantile":
        if q is None:
            q = [0.1, 0.5, 0.9]
        q = [assert_ratio(r, bounds=(0, 1), exclude_values=[0, 1],
                          name=f"quantile '{r}'")
             for r in q]

    # Set default target name if not provided.
    if tname is None:
        tname = "target"

    # Generate predictions.
    xtft_model = check_keras_model_status(
        xtft_model, 
        ops="validate", 
        mode="fit"
    )
    y_pred = xtft_model.predict(
        [
            np.asarray(X_static, dtype=np.float32),
            np.asarray(X_dynamic, dtype=np.float32),
            np.asarray(X_future, dtype=np.float32)
        ]
        # [X_static, X_dynamic, X_future
        #  ]
    )

    pred_df = pd.DataFrame()

    if spatial_cols is not None and len(spatial_cols) >= 2:
        pred_df[spatial_cols[0]] = X_static[:, 0]
        pred_df[spatial_cols[1]] = X_static[:, 1]
    # Otherwise, if no spatial_cols, do nothing.

    # Optionally add dt_col if provided 
    # (dt values must be supplied externally).
    if dt_col is not None:
        # User is responsible for adding dt values.
        pred_df[dt_col] = None

    # If y is provided, add actual target values.
    if y is not None:
        pred_df[f"{tname}_actual"] = y.flatten()

    # Assign prediction columns based on mode.
    if mode == "quantile":
        for i, quantile in enumerate(q):
            col_name = f"{tname}_q{int(round(quantile * 100))}"
            pred_df[col_name] = y_pred[:, 0, i]
        eval_pred = pred_df[f"{tname}_q50"].values
    elif mode == "point":
        col_name = f"{tname}_pred"
        pred_df[col_name] = y_pred[:, 0, 0]
        eval_pred = pred_df[col_name].values
    else:
        raise ValueError("Mode must be either 'quantile' or 'point'.")

    # Optionally apply masking.
    if apply_mask:
        if mask_values is None or mask_fill_value is None:
            raise ValueError(
                "mask_values and mask_fill_value must be provided "
                "when apply_mask is True."
            )
        if y is None:
            raise ValueError("y must be provided to apply masking.")
        if mode == "quantile":
            mask_cols=[] 
            for quantile in q:
                mask_cols.append (f"{tname}_q{int(round(quantile * 100))}")
  
        else:
            mask_cols = [f"{tname}_pred"]
            
        # Check if the provided mask_values exists in the
        # reference column. If not, warn the user and"
        # skip applying mask_by_reference.
        ref_col = f"{tname}_actual"
        unique_vals = pred_df[ref_col].unique()
        if mask_values not in unique_vals:
            warnings.warn(
                f"[mask_by_reference] No matching value"
                f" found for '{mask_values}' in {tname}"
                f" column. Masking will be skipped."
                f" Please ensure that the '{mask_values}'"
                f" exists in the '{tname}' column before"
                " applying the mask."
            )
        else:
            pred_df = mask_by_reference(
                data=pred_df,
                ref_col=ref_col,
                values=mask_values,
                fill_value=mask_fill_value,
                mask_columns=mask_cols
            )
            
    # If y is provided, compute evaluation metrics.
    if y is not None:
        r2 = r2_score(y.flatten(), eval_pred)
        if verbose >= 1:
            print(f"XTFT Model R² Score: {r2:.4f}")
        if mode == "quantile":
            lower_col = f"{tname}_q{int(round(q[0]*100))}"
            upper_col = f"{tname}_q{int(round(q[-1]*100))}"
            cov = coverage_score(
                y_true=y.flatten(),
                y_lower=pred_df[lower_col],
                y_upper=pred_df[upper_col]
            )
            if verbose >= 1:
                print(f"XTFT Model Coverage Score: {cov:.4f}")

    # Save results if requested.
    if savefile is None:
        savefile = f"{mode}_forecast_{tname}_single_step.csv"
    pred_df.to_csv(savefile, index=False)
    if verbose >= 1:
        print(f"Forecast results saved to: {savefile}")

    return pred_df

def forecast_multi_step(
    xtft_model,
    inputs,
    forecast_horizon,
    y=None,
    dt_col=None,
    mode="quantile",
    spatial_cols=None,
    q=None,
    tname=None,
    forecast_dt=None,
    apply_mask=False,
    mask_values=None,
    mask_fill_value=None,
    savefile=None,
    verbose=3, 
    **kws
    ):
    """
    Generate a multi-step forecast using the XTFT model.
    
    This function generates forecasts for multiple future time steps 
    using a pre-trained XTFT deep learning model. The model takes 
    three inputs: `X_static`, `X_dynamic`, and `X_future`, and produces 
    predictions according to the formulation:
    
    .. math::
    
        \hat{y}_{t+i} = f\Bigl( X_{\text{static}},\; 
        X_{\text{dynamic}},\; X_{\text{future}} \Bigr)
    
    for :math:`i = 1, \dots, forecast_horizon`, where :math:`f` is the 
    trained XTFT model.
    
    Parameters
    ----------
    xtft_model : object
        A validated Keras model instance. The model is expected to be 
        verified via ``validate_keras_model``.
    inputs : list or tuple of numpy.ndarray
        A list containing three elements: ``X_static``, ``X_dynamic``, and 
        ``X_future``. If ``spatial_cols`` is provided, it is assumed that 
        the first two columns of ``X_static`` correspond to the first and 
        second spatial coordinates of the original training data.
    forecast_horizon : int
        The number of future time steps to forecast. For example, if 
        ``forecast_horizon`` is 4, the model will generate predictions for 
        4 steps ahead.
    y : numpy.ndarray, optional
        Actual target values. If provided, evaluation metrics such as R² 
        Score and, in quantile mode, the coverage score are computed.
    dt_col : str, optional
        Name of the time column (e.g. ``"year"``). If provided, a column 
        with this name is added to the output DataFrame. The actual time 
        values must be supplied externally.
    mode : str, optional
        Forecast mode. Must be either ``"quantile"`` or ``"point"``. In 
        quantile mode, predictions are generated for multiple quantiles 
        (default: ``[0.1, 0.5, 0.9]``); in point mode, a single prediction 
        is generated.
    spatial_cols : list of str, optional
        A list of spatial column names. If provided, it must contain at least 
        two elements corresponding to the first and second columns of the 
        original training data's ``X_static``.
    time_steps : int, optional
        The number of historical time steps used as input. Default is 
        ``3``.
    q : list of float, optional
        List of quantile values for quantile forecasting. The default is 
        ``[0.1, 0.5, 0.9]`` when ``mode`` is ``"quantile"``.
    tname : str, optional
        Target variable name used to construct output column names. For 
        instance, if ``tname`` is ``"subsidence"``, then output columns may 
        be named ``"subsidence_q10_step1"``, ``"subsidence_q50_step2"``, etc. 
        Default is ``"target"``.
    forecast_dt : any, optional
        Forecast datetime information. If provided and its length matches 
        ``forecast_horizon``, its values are added to the output DataFrame.
    apply_mask : bool, optional
        If True, applies masking via ``mask_by_reference`` to replace 
        predictions in non-subsiding areas. Requires that both 
        ``mask_values`` and ``mask_fill_value`` are provided.
    mask_values : scalar, optional
        The reference value(s) used for masking. Must be provided if 
        ``apply_mask`` is True.
    mask_fill_value : scalar, optional
        The value used to fill masked predictions. Must be provided if 
        ``apply_mask`` is True.
    savefile : str, optional
        File path to save the forecast results as a CSV file. If not 
        provided, a default filename is generated.
    verbose : int, optional
        Verbosity level controlling printed output. Higher values produce 
        more detailed messages.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the multi-step forecast results. In 
        quantile mode, the DataFrame includes columns for each quantile 
        and each forecast step (e.g. ``<tname>_q10_step1``, 
        ``<tname>_q50_step2``, etc.); in point mode, it contains a single 
        prediction column per forecast step (e.g. ``<tname>_pred_step1``). 
        If ``y`` is provided, an additional column (``<tname>_actual``) is 
        included.
    
    Examples
    --------

    >>> from fusionlab.nn.transformers import XTFT
    >>> from fusionlab.nn.utils import forecast_multi_step
    >>> from fusionlab.nn.losses import combined_quantile_loss 
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create a dummy training DataFrame with a date column,
    >>> # spatial features ("longitude", "latitude"), two dynamic 
    >>> # features ("feat1", "feat2"), a static feature ("stat1"), and 
    >>> # the target variable "subsidence".
    >>> date_rng = pd.date_range(start="2020-01-01", periods=60, 
    ...                          freq="D")
    >>> train_df = pd.DataFrame({
    ...     "date": date_rng,
    ...     "longitude": np.random.uniform(-180, 180, 60),
    ...     "latitude": np.random.uniform(-90, 90, 60),
    ...     "feat1": np.random.rand(60),
    ...     "feat2": np.random.rand(60),
    ...     "stat1": np.random.rand(60),
    ...     "subsidence": np.random.rand(60)
    ... })
    >>> 
    >>> # Prepare dummy input arrays for model training.
    >>> # X_static is constructed using "longitude" and "stat1".
    >>> X_static = train_df[["longitude", "stat1"]].values  
    >>> # X_dynamic for "feat1" and "feat2" with time_steps = 3.
    >>> X_dynamic = np.random.rand(60, 3, 2)
    >>> # X_future is a dummy future feature array with shape (60, 3, 1).
    >>> X_future = np.random.rand(60, 3, 1)
    >>> # Target output from "subsidence" reshaped to 
    >>> # (60, 1, 1). For multi-step forecast, forecast_horizon is 4.
    >>> forecast_horizon = 4
    >>> y_array = train_df["subsidence"].values.reshape(60, 1, 1)
    >>> 
    >>> # Instantiate a dummy XTFT model.
    >>> my_model = XTFT(
    ...     static_input_dim=2,    # "longitude" and "stat1"
    ...     dynamic_input_dim=2,   # "feat1" and "feat2"
    ...     future_input_dim=1,    # One future feature
    ...     forecast_horizon=forecast_horizon,
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     embed_dim=16,
    ...     max_window_size=3,
    ...     memory_size=50,
    ...     num_heads=2,
    ...     dropout_rate=0.1,
    ...     lstm_units=32,
    ...     attention_units=32,
    ...     hidden_units=16
    ... )
    >>> my_model.compile(
    ...    optimizer="adam", 
    ...    loss=combined_quantile_loss(my_model.quantiles)
    ...    )
    >>> 
    >>> # Fit the model on the dummy data for demonstration.
    >>> my_model.fit(
    ...     x=[X_static, X_dynamic, X_future],
    ...     y=y_array,
    ...     epochs=1,
    ...     batch_size=8
    ... )
    >>> 
    >>> # Generate forecast datetime values for the forecast horizon.
    >>> forecast_dates = pd.date_range(start="2020-02-01", 
    ...                                periods=forecast_horizon, freq="D")
    >>> 
    >>> # Package inputs as expected by forecast_multi_step.
    >>> inputs = [X_static, X_dynamic, X_future]
    >>> 
    >>> # Generate a multi-step forecast in quantile mode.
    >>> forecast_df_quantile = forecast_multi_step(
    ...     xtft_model=my_model,
    ...     inputs=inputs,
    ...     forecast_horizon=forecast_horizon,
    ...     y=y_array,
    ...     dt_col="date",
    ...     mode="quantile",
    ...     spatial_cols=["longitude", "latitude"],
    ...     q=[0.1, 0.5, 0.9],
    ...     tname="subsidence",
    ...     forecast_dt=forecast_dates,
    ...     apply_mask=False,
    ...     verbose=3
    ... )
    >>> print("Quantile Forecast:")
    >>> print(forecast_df_quantile.head())
    >>> 
    
    (2) For point forecast
    
    >>> # Instantiate a dummy XTFT model.
    >>> my_model = XTFT(
    ...     static_input_dim=2,    # "longitude" and "stat1"
    ...     dynamic_input_dim=2,   # "feat1" and "feat2"
    ...     future_input_dim=1,    # One future feature
    ...     forecast_horizon=forecast_horizon,
    ...     quantiles=None, # set quantiles to None
    ...     embed_dim=16,
    ...     max_window_size=3,
    ...     memory_size=50,
    ...     num_heads=2,
    ...     dropout_rate=0.1,
    ...     lstm_units=32,
    ...     attention_units=32,
    ...     hidden_units=16
    ... )
    >>> my_model.compile(
    ...    optimizer="adam", loss="mse", 
    ...    )
    >>> 
    >>> # Fit the model on the dummy data for demonstration.
    >>> my_model.fit(
    ...     x=[X_static, X_dynamic, X_future],
    ...     y=y_array,
    ...     epochs=1,
    ...     batch_size=8
    ... )
    
    >>> # Generate a multi-step forecast in point mode.
    >>> forecast_df_point = forecast_multi_step(
    ...     xtft_model=my_model,
    ...     inputs=inputs,
    ...     forecast_horizon=forecast_horizon,
    ...     y=y_array,
    ...     dt_col="date",
    ...     mode="point",
    ...     spatial_cols=["longitude", "latitude"],
    ...     tname="subsidence",
    ...     forecast_dt=forecast_dates,
    ...     apply_mask=False,
    ...     verbose=3
    ... )
    >>> print("Point Forecast:")
    >>> print(forecast_df_point.head())

    
    Notes
    -----
    - In quantile mode, predictions are generated for each specified 
      quantile for every forecast step, and the median (``0.5``) is used 
      for evaluation.
    - In point mode, a single prediction is generated per forecast step.
    - The output prediction array is expected to have the shape 
      :math:`(n, forecast\_horizon, m)`, where :math:`n` is the number of 
      samples and :math:`m` is the number of outputs per step (e.g., number 
      of quantiles in quantile mode or 1 in point mode).
    - The provided ``spatial_cols`` must correspond to the first two 
      columns of the original training data's ``X_static``.
    - Evaluation metrics such as R² Score and Coverage Score (in 
      quantile mode) are computed if actual target values (``y``) are provided.
    - The DataFrame is constructed by iterating over each sample and 
      each forecast step.
    
    See Also
    --------
    forecast_single_step : Function for single-step forecasts.
    coverage_score       : Function to compute the coverage score.
    validate_keras_model : Function to validate a Keras model.
    assert_ratio         : Function to verify quantile ratios.

    """
    # **********************************************************
    from ..metrics import coverage_score
    # **********************************************************
    
    def vprint(msg, level=1):
        """
        Prints `msg` if the current verbosity setting is >= `level`.
        """
        if verbose >= level:
            print(msg)
            
    # Validate inputs: expect a list/tuple of three elements.
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
        raise ValueError(
            "inputs must be a list or tuple containing "
            "[X_static, X_dynamic, X_future]."
        )
    X_static, X_dynamic, X_future = inputs

    # Validate model.
    xtft_model = validate_keras_model(xtft_model, deep_check=True)
    if verbose >= 1:
        print("\nGenerating multi-step forecast in mode:", mode)

    # Set default quantiles for quantile mode.
    if q is None:
        q = [0.1, 0.5, 0.9]
        
    q=check_forecast_mode(
        mode, q, error="warn", ops="validate", 
        round_digits=2, 
        dtype = np.float64 
    )
    if mode == "quantile":
        q = [assert_ratio(r, bounds=(0, 1), exclude_values=[0, 1],
                          name=f"quantile '{r}'")
             for r in q]

    # Set default target name if not provided.
    if tname is None:
        tname = "target"

    xtft_model= check_keras_model_status(
        xtft_model, 
        ops="validate", 
        mode="fit",
        error="warn"
    )
    # Generate predictions using the XTFT model.
    y_pred = xtft_model.predict(
        [
            np.asarray(X_static, dtype=np.float32),
            np.asarray(X_dynamic, dtype=np.float32),
            np.asarray(X_future, dtype=np.float32)
        ]
    )#.squeeze(-1)
 
    # Determine available forecast steps based on y.
    available_steps = forecast_horizon
    if y is not None:
        if len(y.shape) == 3:
            if y.shape[1] < forecast_horizon:
                warnings.warn(
                    f"Provided y has forecast steps "
                    f"({y.shape[1]}) which is less than the "
                    f"specified forecast_horizon ({forecast_horizon}). "
                    "Evaluation will be performed on the"
                    " available steps."
                )
            available_steps = min(forecast_horizon, y.shape[1])
        else:
            # For non-3D y, use its length along axis 0.
            available_steps = forecast_horizon
    
    vprint(
        f"[INFO] Forecast horizon: {forecast_horizon}, "
        f"Available forecast steps: {available_steps}.",
        level=1
    )

    # Use the BatchDataFrameBuilder context manager
    # for memory efficient management to deal with large dataframe.
    row_count = 0
    n_samples = X_static.shape[0]
    
    with BatchDataFrameBuilder(
            chunk_size=50_000,  processor='auto', verbose=verbose
            ) as builder:
        # Loop over each sample and each forecast step.
        for j in range(n_samples):
            vprint(f"[INFO] Processing sample index {j}.", level=2)
            for i in range(available_steps):
                
                vprint(f"  [DEBUG] Forecast step {i + 1}/{available_steps}.", 
                       level=3)
                
                row = {}
                # Add spatial columns if provided.
                if spatial_cols is not None and len(spatial_cols) >= 2:
                    row[spatial_cols[0]] = X_static[j, 0]
                    row[spatial_cols[1]] = X_static[j, 1]
                    
                    vprint(
                        f"    [DEBUG] Spatial columns set: "
                        f"{spatial_cols[0]}={row[spatial_cols[0]]}, "
                        f"{spatial_cols[1]}={row[spatial_cols[1]]}",
                        level=3
                    )
    
                # Add dt_col if provided.
                if dt_col is not None:
                    if (forecast_dt is not None and
                        len(forecast_dt) == forecast_horizon):
                        row[dt_col] = forecast_dt[i]
                    else:
                        row[dt_col] = None
                    
                    vprint(f"    [DEBUG] {dt_col}={row[dt_col]}", level=3)
                    
                # Add actual target value if y is provided.
                if y is not None:
                    if len(y.shape) == 3:
                        row[f"{tname}_actual"] = y[j, i, 0]
                    else:
                        row[f"{tname}_actual"] = y[j, i]
                        
                    vprint(
                        f"    [DEBUG] Actual: {tname}_actual={row[f'{tname}_actual']}",
                        level=3
                    )
                # Assign prediction columns based on mode.
                if mode == "quantile":
                    for iq, quantile in enumerate(q):
                        col_name = (
                            f"{tname}_q{int(quantile * 100)}_step{i+1}"
                        )
                        row[col_name] = y_pred[j, i, iq]
                        
                        vprint(
                            f"    [DEBUG] Assigned {col_name}={row[col_name]}",
                            level=3
                        )
                elif mode == "point":
                    col_name = f"{tname}_pred_step{i+1}"
                    row[col_name] = y_pred[j, i]
                    
                    vprint(
                        f"    [DEBUG] Assigned {col_name}={row[col_name]}",
                        level=3
                    )
                else:
                    raise ValueError(
                        "Mode must be either 'quantile' or 'point'."
                    )
                # Add this row to the builder
                builder.add_row(row)
                # count rows...
                row_count += 1

    vprint(
        f"[INFO] Finished generating predictions for {row_count:,} rows.", 
        level=1
    )
    pred_df = step_to_long( 
        df= builder.final_df, # pd.DataFrame(rows)
        tname =tname, 
        dt_col=dt_col, 
        spatial_cols=spatial_cols, 
        mode=mode, 
        verbose=verbose
    )
    # Optionally apply masking.
    if apply_mask:
        if mask_values is None or mask_fill_value is None:
            raise ValueError(
                "mask_values and mask_fill_value must be provided "
                "when apply_mask is True."
            )
        if y is None:
            raise ValueError("y must be provided to apply masking.")
        
        if mode == "quantile":
            mask_cols = [
                f"{tname}_q{int(quant*100)}"
                for quant in q # [q[0], q[-1]]
            ]
        else:
            mask_cols = [
                f"{tname}_pred_step{i+1}"
                for i in range(forecast_horizon)
            ]
        
        # Check if the provided mask_values exists in the reference column.
        ref_col = f"{tname}_actual"
        unique_vals = pred_df[ref_col].unique()
        if mask_values not in unique_vals:
            warnings.warn(
                "[mask_by_reference] No matching value"
                f" found for '{mask_values}' in {tname}"
                " column. Masking will be skipped. Please"
                " ensure the value exists before applying "
                "the mask."
            )
        else:
            pred_df = mask_by_reference(
                data=pred_df,
                ref_col=ref_col,
                values=mask_values,
                fill_value=mask_fill_value,
                mask_columns=mask_cols
            )

    # If y is provided, compute evaluation metrics.
    if y is not None:
        # Determine the number of available steps from y.
        available_steps = forecast_horizon
        if len(y.shape) == 3:
            if y.shape[1] < forecast_horizon:
                warnings.warn(
                    f"Provided y has forecast steps "
                    f"({y.shape[1]}) which is less than the "
                    f"specified forecast_horizon ({forecast_horizon}). "
                    "Evaluation will use available steps."
                )
            available_steps = min(forecast_horizon, y.shape[1])
        else:
            available_steps = forecast_horizon
    
        # Compute evaluation predictions using available_steps.
        if mode == "quantile":
            median_preds = []
            for j in range(n_samples):
                for i in range(available_steps):
                    median_preds.append(y_pred[j, i, 1])
            eval_pred = np.array(median_preds)
        else:
            eval_pred = y_pred[:, :available_steps].flatten()
    
        # Adjust y to only include available forecast steps.
        if len(y.shape) == 3:
            y_eval = y[:, :available_steps, 0].flatten()
        else:
            y_eval = y.flatten()
    
        # Compute R² Score.
        r2 = r2_score(y_eval, eval_pred)
        if verbose >= 1:
            print(f"XTFT Model R² Score: {r2:.4f}")
    
        # For quantile mode, compute Coverage Score.
        if mode == "quantile":
            lower = []
            upper = []
            for j in range(n_samples):
                for i in range(available_steps):
                    lower.append(y_pred[j, i, 0])
                    upper.append(y_pred[j, i, -1])
            lower = np.array(lower)
            upper = np.array(upper)
            cov = coverage_score(y_eval, lower, upper)
            if verbose >= 1:
                print(f"XTFT Model Coverage Score: {cov:.4f}")
 
    # Save results if requested.
    if savefile is None:
        savefile = f"{mode}_forecast_{tname}_multi_step.csv"
    pred_df.to_csv(savefile, index=False)
    if verbose >= 1:
        print(f"Forecast results saved to: {savefile}")
    
    return pred_df

def _step_to_long_q(
    df, 
    tname: str = None,         # Can be None (auto-detect if possible)
    spatial_cols: list = None,  # Defaults to ['longitude', 'latitude'] if present
    dt_col: str = None,         # Time column, if any
    sort: bool = False,         # Sort by dt_col if provided
    verbose: int = 3            # Verbosity level (0 to 7)
):
    # Helper logging function based on verbosity.
    def log(message, level=3):
        if verbose >= level:
            print(message)

    init_size = len(df)
    log(
        "[INFO] Initiating consistency check for "
        f"{init_size:,} samples...", 
        level=7
    )

    # Auto-detect tname if not provided: use the unique prefix before 
    # the first '_' in columns that contain "q" and "step".
    if tname is None:
        possible_tnames = {col.split("_")[0] 
                           for col in df.columns 
                           if "q" in col and "step" in col}
        if len(possible_tnames) == 1:
            tname = possible_tnames.pop()
            log(f"[INFO] Auto-detected tname: {tname}", level=4)
        else:
            raise ValueError(
                f"Could not infer `tname`. Candidates: {possible_tnames}. "
                "Please provide `tname` explicitly."
            )

    # Identify quantile step columns dynamically.
    quantile_columns = [
        col 
        for col in df.columns 
        if f"{tname}_q" in col or ("q" in col and "step" in col)
    ]
    log(
        f"[INFO] Identified {len(quantile_columns)} quantile step columns.", 
        level=5
    )

    # Extract unique quantile levels (e.g., 'q10', 'q50', 'q89')
    quantile_levels = sorted(
        set(col.split("_")[1] for col in quantile_columns if "step" in col)
    )
    log(f"[INFO] Quantile levels extracted: {quantile_levels}", level=5)

    # Set default spatial columns if not provided.
    if spatial_cols is None:
        spatial_cols = [col for col in ["longitude", "latitude"] 
                        if col in df.columns]
    if spatial_cols:
        log(f"[INFO] Using spatial columns: {spatial_cols}", level=4)
        spatial_cols = list(spatial_cols)
    # Identify non-quantile columns (excluding spatial columns).
    non_quantile_columns = [
        col for col in df.columns 
        if col not in quantile_columns and col not in spatial_cols 
        and col !=dt_col
    ] 
    
    # Convert `dt_col` if it is a Timestamp column
    restore_dt_col = False
    if dt_col and np.issubdtype(df[dt_col].dtype, np.datetime64):
        safe_dt_values= df[dt_col].copy() 
        df[dt_col] = df[dt_col].view(np.int64)  # Convert to int64 timestamps
        restore_dt_col = True
        
        log(f"[INFO] Converted datetime column '{dt_col}' to int64 timestamps.",
            level=6)
        
    # Define base and final column names.
    base_columns = spatial_cols  + ([dt_col] if dt_col else []) + non_quantile_columns
    
    final_columns = base_columns + [f"{tname}_{q}" for q in quantile_levels]
    log(f"[INFO] Final columns determined: {len(final_columns)}", level=6)

    # Convert DataFrame to a NumPy array for fast operations.
    data_array   = df.to_numpy()
    column_index = {col: i for i, col in enumerate(df.columns)}
    
    # Initialize output array filled with NaNs.
    output_array = np.empty((data_array.shape[0], len(final_columns)), dtype=np.float64)
    output_array.fill(np.nan)
    log("[INFO] Re-Initialized output array with NaNs.", level=6)

    # Copy over carry-over columns (e.g., dt_col, spatial, actual).
    for col in base_columns:
        if col in column_index:
            output_array[:, final_columns.index(col)] = data_array[:, column_index[col]]
            log(f"  [DEBUG] Imputed column: {col}", level=7)

    # Process quantile forecast columns:
    for q in quantile_levels:
        log(f"  [DEBUG] Processing quantile level: {q}", level=4)
        # Get all columns that contain this quantile level and 'step'.
        step_cols = [col for col in df.columns if f"{q}_step" in col]
        step_indices = [column_index[col] for col in step_cols]
        final_col_idx = final_columns.index(f"{tname}_{q}")

        for step_idx in step_indices:
            # Create a boolean mask for non-NaN entries.
            mask = ~np.isnan(data_array[:, step_idx])
            output_array[mask, final_col_idx] = data_array[mask, step_idx]
            log(
                f"   [TRACE] Assigned values for {tname}_{q} using "
                f"{df.columns[step_idx]}", 
                level=7
            )

    # Convert the output array back to a DataFrame.
    final_df = pd.DataFrame(output_array, columns=final_columns)
    # Restore original data types for carry-over columns.
    for col in base_columns:
        if col in df.columns:
            final_df[col] = final_df[col].astype(df[col].dtype)
            log(f"  [TRACE] Restored dtype for column: {col}", level=6)

    # Restore `dt_col` if it was a Timestamp
    if restore_dt_col:
        try:
            final_df[dt_col]= safe_dt_values 
        except: # rather than reconvert. 
           final_df[dt_col] = pd.to_datetime(
               final_df[dt_col].astype(np.int64))
        log(f"[INFO] Restored datetime column '{dt_col}' to Timestamp format.",
            level=6)
        
    # Sort by dt_col if requested and present.
    if sort and dt_col and dt_col in final_df.columns:
        final_df = final_df.sort_values(by=dt_col, ignore_index=True)
        log(f"[INFO] Sorted DataFrame by {dt_col}.", level=4)

    log(f"[INFO] Processing completed. Total new rows: {len(final_df):,}", level=3)
    return final_df

def _step_to_long_pred(
    df, 
    tname: str = None,        # Auto-detect target name if None
    spatial_cols: list = None,  # Defaults to ['longitude', 'latitude'] if present
    dt_col: str = None,         # Time column, if any
    sort: bool = False,         # Sorts by dt_col if provided
    verbose: int = 3            # Verbosity level (0 to 7)
):
    # Helper logging function based on verbosity level.
    def log(message, level=3):
        """Logs messages with indentation based on depth."""
        if verbose >= level:
            print(message)

    init_size = len(df)
    log(
        f"[INFO] Initiating processing for {init_size:,} samples...",
        level=7
    )

    # Auto-detect 'tname' if not provided.
    if tname is None:
        possible_tnames = {
            col.split("_")[0] for col in df.columns 
            if "pred_step" in col
        }
        if len(possible_tnames) == 1:
            tname = possible_tnames.pop()
            log(f"[INFO] Auto-detected tname: {tname}", level=4)
        else:
            raise ValueError(
                f"Could not infer `tname`. Candidates: {possible_tnames}. "
                "Please provide `tname` explicitly."
            )

    # Identify step-based prediction columns dynamically.
    pred_columns = [
        col for col in df.columns 
        if f"{tname}_pred_step" in col or "pred_step" in col
    ]
    log(
        f"[INFO] Identified {len(pred_columns)} step-based prediction "
        f"columns.", level=5
    )

    # Default spatial columns if not provided.
    if spatial_cols is None:
        spatial_cols = [
            col for col in ["longitude", "latitude"] 
            if col in df.columns
        ]
    if spatial_cols:
        log(f"[INFO] Using spatial columns: {spatial_cols}", level=4)
        spatial_cols = list(spatial_cols)
    # Identify non-prediction columns (excluding spatial columns).
    non_pred_columns = [
        col for col in df.columns 
        if col not in pred_columns and col not in spatial_cols
        and col !=dt_col
    ]
    
    # Convert `dt_col` if it is a Timestamp column
    restore_dt_col = False
    if dt_col and np.issubdtype(df[dt_col].dtype, np.datetime64):
        safe_dt_values= df[dt_col].copy() 
        df[dt_col] = df[dt_col].view(np.int64)  # Convert to int64 timestamps
        restore_dt_col = True
        
        log(f"[INFO] Converted datetime column '{dt_col}' to int64 timestamps.",
            level=6)
        
    # Define base and final column names.
    base_columns  = spatial_cols + ([dt_col] if dt_col else []) + non_pred_columns
    final_columns = base_columns + [f"{tname}_pred"]
    log(
        f"[INFO] Final columns determined: {final_columns}", 
        level=6
    )

    # Convert the DataFrame to a NumPy array for fast processing.
    data_array   = df.to_numpy().astype(float) # XXX RECHECK .astype(float)
    column_index = {col: i for i, col in enumerate(df.columns)}

    # Initialize an output array filled with NaNs.
    output_array = np.empty(
        (data_array.shape[0], len(final_columns)), 
        dtype=np.float64
    )
    output_array.fill(np.nan)
    log("[INFO] Re-initialized output array with NaNs.", level=6)

    # Copy spatial, date, and all non-prediction columns.
    for col in base_columns:
        if col in column_index:
            idx_final = final_columns.index(col)
            idx_source = column_index[col]
            output_array[:, idx_final] = data_array[:, idx_source]
            log(f"  [DEBUG] Copied column: {col}", level=7)

    # Process prediction step columns.
    log("  [DEBUG] Processing step-based predictions...", level=4)
    final_pred_idx = final_columns.index(f"{tname}_pred")
    for step_col in pred_columns:
        step_idx = column_index[step_col]
        mask     = ~np.isnan(data_array[:, step_idx])  # Valid (non-NaN) values.
        output_array[mask, final_pred_idx] = data_array[mask, step_idx]
        log(
            f"   [TRACE] Assigned values from {step_col} to "
            f"{tname}_pred", level=7
        )

    # Convert the output array back to a DataFrame.
    final_df = pd.DataFrame(output_array, columns=final_columns)
    log(
        "[INFO] Converted output array back to DataFrame.", 
        level=5
    )

    # Restore original data types for carry-over columns.
    for col in base_columns:
        if col in df.columns:
            final_df[col] = final_df[col].astype(df[col].dtype)
            log(
                f"  [TRACE] Restored dtype for column: {col}", 
                level=6
            )
    # Restore `dt_col` if it was a Timestamp
    if restore_dt_col:
        try:
            final_df[dt_col]= safe_dt_values 
        except: # rather than reconvert. 
           final_df[dt_col] = pd.to_datetime(
               final_df[dt_col].astype(np.int64))
        log(f"[INFO] Restored datetime column '{dt_col}' to Timestamp format.",
            level=6)
        
    # Sort by dt_col if requested and available.
    if sort and dt_col and dt_col in final_df.columns:
        final_df = final_df.sort_values(
            by=dt_col, 
            ignore_index=True
        )
        log(f"[INFO] Sorted DataFrame by {dt_col}.", level=4)

    log(
        f"[INFO] Processing completed. Total new rows: {len(final_df):,}", 
        level=3
    )
    return final_df

def step_to_long(
    df: pd.DataFrame,
    tname: str = None,
    dt_col: str = None,
    spatial_cols: list = None,
    mode: str = "quantile",
    quantiles: list = None,
    verbose: int= 3,
    sort: bool=True, 
) -> pd.DataFrame:
     
    is_frame(
        df, df_only=True, 
        objname="Step Data", 
        error="raise"
    )
    quantiles = check_forecast_mode(
        mode, q=quantiles, 
        ops="validate", 
        error="warn",
        dtype=np.float64, 
        round_digits=2
        )
    # detect_digit to check whether ther is quantile
    if mode=="quantile": 
        quantiles = validate_consistency_q(
            quantiles, df.columns, 
            error="raise", 
            default_to="auto_q"
        )
    quantiles = validate_quantiles (
        quantiles, round_digits=2, 
        dtype=np.float64, 
    )
    
    if mode=="quantile": 
        final_df = _step_to_long_q( 
            df = df, 
            tname=tname, 
            spatial_cols= spatial_cols, 
            dt_col=dt_col, 
            verbose=verbose, 
            sort=sort, 
      )
    else : # point" 
         final_df = _step_to_long_pred( 
             df=df, 
             tname=tname, 
             spatial_cols= spatial_cols, 
             dt_col=dt_col, 
             sort=sort, 
             verbose=verbose 
       )
    
    return final_df

def generate_forecast_with(
        xtft_model,
        inputs,
        forecast_horizon,
        y=None,
        dt_col=None,
        mode="quantile",
        spatial_cols=None,
        q=None,
        tname=None,
        forecast_dt=None,
        apply_mask=False,
        mask_values=None,
        mask_fill_value=None,
        savefile=None,
        verbose=3, 
        **kw
    ):
    
    forecast_horizon = validate_positive_integer(
        forecast_horizon, "forecast_horizon", 
        msg = f"forecast_horizon must be at least 1. Got '{forecast_horizon}'"
    )
    if forecast_horizon == 1:
        return forecast_single_step(
            xtft_model=xtft_model,
            inputs=inputs,
            y=y,
            dt_col=dt_col,
            mode=mode,
            spatial_cols=spatial_cols,
            q=q,
            tname=tname,
            forecast_dt=forecast_dt,
            apply_mask=apply_mask,
            mask_values=mask_values,
            mask_fill_value=mask_fill_value,
            savefile=savefile,
            verbose=verbose
        )
    elif forecast_horizon > 1:
        return forecast_multi_step(
            xtft_model=xtft_model,
            inputs=inputs,
            forecast_horizon=forecast_horizon,
            y=y,
            dt_col=dt_col,
            mode=mode,
            spatial_cols=spatial_cols,
            q=q,
            tname=tname,
            forecast_dt=forecast_dt,
            apply_mask=apply_mask,
            mask_values=mask_values,
            mask_fill_value=mask_fill_value,
            savefile=savefile,
            verbose=verbose
        )


def squeeze_last_dim_if(tensors, output_dims):
    """
    Squeeze the last dimension of tensor(s) if it equals 1 based on `output_dims`.

    `output_dims` can be:
      * a single int: apply that rule to every tensor
      * a list of ints: must match length of `tensors` list, each applied
        to corresponding tensor
      * a dict mapping keys to ints: each key must appear in `tensors` dict;
        unmatched keys in `output_dims` will warn but be ignored

    Parameters
    ----------
    tensors : tf.Tensor or list of tf.Tensor or dict of tf.Tensor
        - If a single `tf.Tensor`, processed with `output_dims` if int.
        - If a `list`, each element is processed with its corresponding
          entry in `output_dims` if list, or with the single int if int.
        - If a `dict`, each value is processed with the matching int in
          `output_dims` dict, or with the single int if `output_dims` is int.

    output_dims : int or list of int or dict of int
        - If int: all tensors use that output dimension rule.
        - If list: length must equal `len(tensors)` when `tensors` is list.
        - If dict: keys map to ints; for any key in `tensors` missing from
          `output_dims`, no squeeze is applied; keys in `output_dims` absent
          in `tensors` produce a warning.

    Returns
    -------
    tf.Tensor or list or dict
        Same structure as `tensors`, but with trailing size-1 axes removed
        where `output_dim == 1`. If `output_dim != 1`, that tensor is unchanged.

    Raises
    ------
    TypeError
        If `tensors` is not a `tf.Tensor`, list, or dict; or if `output_dims`
        type is invalid; or if list lengths do not match.
    ValueError
        If list lengths mismatch.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from fusionlab.utils.generic_utils import squeeze_last_dim_if
    >>>
    >>> # Single tensor, single int → squeeze if last dim=1
    >>> t = tf.zeros((8, 5, 1))
    >>> out = squeeze_last_dim_if(t, output_dims=1)
    >>> out.shape
    TensorShape([8, 5])
    >>>
    >>> # List of tensors, list of ints
    >>> t_list = [tf.zeros((4, 1)), tf.zeros((4, 2, 1))]
    >>> out_list = squeeze_last_dim_if(t_list, output_dims=[1, 2])
    >>> [x.shape for x in out_list]
    [TensorShape([4]), TensorShape([4, 2, 1])]
    >>>
    >>> # Dict of tensors, dict of ints
    >>> t_dict = {
    ...     "a": tf.zeros((3, 1, 1)),
    ...     "b": tf.zeros((3, 1, 2))
    ... }
    >>> out_dict = squeeze_last_dim_if(t_dict, output_dims={"a": 1, "b": 2})
    >>> {k: v.shape for k, v in out_dict.items()}
    {'a': TensorShape([3, 1]), 'b': TensorShape([3, 1, 2])}

    Notes
    -----
    - This function does a shallow traversal:
      * If `tensors` is list, each element must be `tf.Tensor`.
      * If `tensors` is dict, each value must be `tf.Tensor`.
    - For dict mode: missing keys in `output_dims` → no change;
      extra keys in `output_dims` → warning.

    See Also
    --------
    tf.squeeze : Remove dimensions of size 1 from the shape of a tensor.
    tf.reshape : Manually reshape a tensor if more complex edits are needed.
    """

    # Validate output_dims type
    if not isinstance(output_dims, (int, list, dict)):
        raise TypeError(
            f"`output_dims` must be int, list, or dict, got "
            f"{type(output_dims).__name__}"
        )

    def _process_tensor(tensor, od):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Expected tf.Tensor, got {type(tensor).__name__}")
        # Only squeeze when od == 1 and last axis size is 1
        if od == 1 and tensor.shape.rank is not None and tensor.shape[-1] == 1:
            return tf_squeeze(tensor, axis=-1)
        return tensor

    # Single tensor case
    if isinstance(tensors, Tensor):
        if isinstance(output_dims, int):
            return _process_tensor(tensors, output_dims)
        else:
            raise TypeError(
                "`output_dims` must be int when `tensors` is a single Tensor"
            )

    # List of tensors case
    if isinstance(tensors, list):
        if isinstance(output_dims, int):
            return [
                _process_tensor(t, output_dims) for t in tensors
            ]
        if isinstance(output_dims, list):
            if len(output_dims) != len(tensors):
                raise ValueError(
                    "Length of `output_dims` list must match length of `tensors` list"
                )
            return [
                _process_tensor(t, od) for t, od in zip(tensors, output_dims)
            ]
        # dict not allowed if tensors is list
        raise TypeError(
            "`output_dims` must be int or list when `tensors` is a list"
        )

    # Dict of tensors case
    if isinstance(tensors, dict):
        result = {}
        if isinstance(output_dims, int):
            for key, tensor in tensors.items():
                result[key] = _process_tensor(tensor, output_dims)
            return result

        if isinstance(output_dims, dict):
            # Warn about extra keys in output_dims not in tensors
            for k in output_dims:
                if k not in tensors:
                    warnings.warn(
                        f"Key '{k}' in `output_dims` not found in `tensors`. Ignored."
                    )
            # Process only matching keys
            for key, tensor in tensors.items():
                od = output_dims.get(key, None)
                if od is None:
                    # No output_dim specified for this key → no change
                    result[key] = tensor
                else:
                    result[key] = _process_tensor(tensor, od)
            return result

        # list not allowed if tensors is dict
        raise TypeError(
            "`output_dims` must be int or dict when `tensors` is a dict"
        )

    # Invalid type for tensors
    raise TypeError(
        f"`tensors` must be a tf.Tensor, list, or dict, got {type(tensors).__name__}"
    )

@Deprecated(
    reason=(
        "`format_predictions_to_dataframe` is **deprecated** and will be removed "
        "in a future release. Switch to `format_predictions`, which offers a more "
        "robust, consistent API (supports explicit `forecast_horizon`, enriched "
        "validation, and unified outputs)."
    ),
    version="0.2.3",
    category=DeprecationWarning,
    alternative="format_predictions",
)
def format_predictions_to_dataframe(
    predictions: Optional[Union[np.ndarray, Tensor]] = None,
    model: Optional[Model] = None,
    inputs: Optional[List[Optional[Union[np.ndarray, Tensor]]]] = None,
    y_true_sequences: Optional[Union[np.ndarray, Tensor]] = None,
    target_name: Optional[str] = "target",
    quantiles: Optional[List[float]] = None,
    forecast_horizon: Optional[int] = None,
    output_dim: Optional[int] = None,
    spatial_data_array: Optional[Union[np.ndarray, Tensor]] = None,
    spatial_cols: Optional[List[str]] = None,
    spatial_cols_indices: Optional[List[int]] = None,
    evaluate_coverage: bool = False,
    scaler: Optional[Any] = None,
    scaler_feature_names: Optional[List[str]] = None,
    target_idx_in_scaler: Optional[int] = None,
    verbose: int = 0,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Deprecated alias for `format_predictions`. See `format_predictions`
    for the updated, recommended API. All original parameters are forwarded
    to `format_predictions`.

    Returns
    -------
    pd.DataFrame
        The formatted prediction DataFrame from `format_predictions`.
    """
    return format_predictions(
        predictions=predictions,
        model=model,
        inputs=inputs,
        y_true_sequences=y_true_sequences,
        target_name=target_name,
        quantiles=quantiles,
        forecast_horizon=forecast_horizon,
        output_dim=output_dim,
        spatial_data_array=spatial_data_array,
        spatial_cols=spatial_cols,
        spatial_cols_indices=spatial_cols_indices,
        evaluate_coverage=evaluate_coverage,
        scaler=scaler,
        scaler_feature_names=scaler_feature_names,
        target_idx_in_scaler=target_idx_in_scaler,
        verbose=verbose,
        **kwargs
    )

def make_dict_to_tuple_fn(
    feature_keys: Sequence[str],
    target_keys: Optional[Sequence[str]] = None,
    *,
    allow_missing_optional: bool = True,
) -> Callable[
    [Dict[str, Tensor], Union[Tensor, Dict[str, Tensor], None]],
    Tuple[Tuple[Tensor, ...], Union[Tensor, Dict[str, Tensor], None]],
]:
    """
    Create a `tf.data.Dataset.map` function that converts a *feature
    dictionary* into a **positional tuple** expected by a sub-classed
    Keras model.

    Parameters
    ----------
    feature_keys : Sequence[str]
        Keys – **in the exact positional order required by the model** –
        that will populate the tuple.  For your PINN:

        >>> feature_keys = [
        ...     "coords",           # (B, T, 3)
        ...     "static_features",  # (B, S)
        ...     "dynamic_features", # (B, T, D)
        ...     "future_features",  # (B, H, F)
        ... ]

    target_keys : Sequence[str] or None, default ``None``
        * ``None`` – pass through the dataset’s existing target element.
        * sequence – extract those keys (from *features* **or** *targets*
          if it is a dict) and return them as:
            * a single tensor  if `len(target_keys) == 1`;
            * a dict `{key: tensor}` otherwise.

    allow_missing_optional : bool, default ``True``
        Whether to substitute ``None`` for missing *optional* feature or
        target keys.  If ``False``, a missing key raises ``KeyError``.

    Returns
    -------
    Callable
        A map-function you can plug into a `tf.data.Dataset`:

        >>> mapper = make_dict_to_tuple_fn(feature_keys, ["subsidence", "gwl"])
        >>> train_ds = train_ds.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)

    Notes
    -----
    * Duplicate names in `feature_keys` are rejected so the positional
      tuple cannot be ambiguous.
    * The function is fully Autograph-compatible; TensorFlow will stage
      it as part of the dataset pipeline.
    """

    # Sanity checks
    feature_keys = list(feature_keys)
    if len(feature_keys) != len(set(feature_keys)):
        raise ValueError(
            "`feature_keys` contain duplicates"
            " – tuple order would be ambiguous.")

    target_keys = list(target_keys) if target_keys is not None else None

    # Map-function returned to the caller
    def _dict_to_tuple(
        features: Dict[str, Tensor],
        targets: Union[Tensor, Dict[str, Tensor], None] = None,
    ):
        # Build positional feature tuple 
        positional: List[Optional[Tensor]] = []
        for k in feature_keys:
            if k in features:
                positional.append(features[k])
            elif allow_missing_optional:
                positional.append(None)
            else:
                raise KeyError(f"Missing required feature key '{k}'")
        inputs_tuple: Tuple[Tensor, ...] = tuple(positional)

        # Derive targets_out 
        if target_keys is None:
            targets_out = targets  # pass through
        else:
            def _grab(src, key):
                if isinstance(src, dict) and key in src:
                    return src[key]
                if allow_missing_optional:
                    return None
                raise KeyError(f"Target key '{key}' not found.")

            if isinstance(targets, dict):
                collected = {
                    k: _grab(features, k) or _grab(
                        targets, k) for k in target_keys}
            else:
                collected = {
                    k: _grab(features, k) for k in target_keys}

            targets_out = next(
                iter(collected.values())) if len(
                    collected) == 1 else collected

        return inputs_tuple, targets_out

    return _dict_to_tuple


step_to_long.__doc__=r"""
Convert a multi-step forecast DataFrame from wide to long format.

This function transforms a DataFrame containing multi-step forecast
predictions into a long-format DataFrame. In quantile mode, forecast
columns such as ``subsidence_q10_step1``, ``subsidence_q50_step1``, etc. 
are consolidated into unified columns (e.g. ``subsidence_q10``, 
``subsidence_q50``, etc.), while in point mode, a single prediction 
column (``subsidence_pred``) is generated. The transformation also 
carries over additional columns (e.g. spatial coordinates and time) 
from the original DataFrame.

Parameters
----------
df           : pandas.DataFrame
    The multi-step forecast DataFrame. Expected to contain forecast 
    prediction columns (e.g. columns with ``_q`` or ``_pred_step`` in their
    names) along with other identifiers.
tname        : str, optional
    The base name of the target variable (e.g. ``"subsidence"``). If ``None``,
    the function attempts to auto-detect the target name from the column names.
dt_col       : str, optional
    The name of the time column to include in the final DataFrame. If not
    provided, time sorting is not performed.
spatial_cols : list of str, optional
    A list of spatial coordinate columns (e.g. ``["longitude", "latitude"]``)
    to be retained in the final output.
mode         : {"quantile", "point"}, default="quantile"
    The forecast mode. In ``"quantile"`` mode, multiple quantile forecast 
    columns are merged into unified columns. In ``"point"`` mode, a single
    prediction column is produced.
quantiles    : list of float, optional
    The quantile values for quantile mode (e.g. ``[0.1, 0.5, 0.9]``). If 
    not provided, defaults are used.
sort         : bool, optional
    If True, sorts the final DataFrame by the column specified in ``dt_col``
    (if present). Default is True.

verbose      : int, optional
    Verbosity level for logging output. Higher values (e.g. 5 to 7) provide
    more detailed debug information.

Returns
-------
pandas.DataFrame
    A long-format DataFrame with the following columns:
      - Spatial columns (if provided)
      - The time column (``dt_col``), if provided
      - Forecast prediction columns:
        - In quantile mode: unified columns (e.g. 
          ``subsidence_q10``, ``subsidence_q50``, etc.)
        - In point mode: a single column (``subsidence_pred``)

Examples
--------
>>> from fusionlab.nn.utils import step_to_long
>>> # Given a DataFrame `forecast_df` with columns like:
>>> # ['longitude', 'latitude', 'year', 'subsidence_actual',
>>> #  'subsidence_q10_step1', 'subsidence_q50_step1', 'subsidence_q89_step1',
>>> #  'subsidence_q10_step2', ...]
>>> long_df = step_to_long(
...     df=forecast_df,
...     tname="subsidence",
...     dt_col="year",
...     spatial_cols=["longitude", "latitude"],
...     mode="quantile",
...     quantiles=[0.1, 0.5, 0.9],
...     verbose=3,
...     sort=True
... )
>>> print(long_df.head())

Notes
-----
Internally, this function calls:

- :func:`check_forecast_mode` to validate the user-specified 
  quantiles.
- :func:`validate_consistency_q` and :func:`validate_quantiles` to ensure 
  that the quantile values provided by the user match those auto-detected 
  from the DataFrame.
- Depending on the ``mode``, it then calls either 
  :func:`_step_to_long_q` (for quantile mode) or 
  :func:`_step_to_long_pred` (for point mode) to perform the conversion.
  
Mathematically, let :math:`X \in \mathbb{R}^{n \times m}` represent the 
wide-format DataFrame, where each row corresponds to one sample and each 
forecast step is stored in separate columns. The function reshapes :math:`X` 
into a long-format DataFrame :math:`Y \in \mathbb{R}^{(n \cdot s) \times p}`, 
where :math:`s` is the forecast horizon and :math:`p` is the number of output 
columns after merging forecast step values.
 
  
See Also
--------
_step_to_long_q : Converts multi-step quantile forecasts to long format.
_step_to_long_pred: Converts multi-step point forecasts to long format.
detect_digits   : Extracts numeric values from strings for quantile detection.

References
----------
.. [1] Wickham, H. (2014). "Tidy Data". Journal of Statistical Software,
       59(10), 1-23.
.. [2] McKinney, W. (2010). "Data Structures for Statistical Computing in 
       Python". Proceedings of the 9th Python in Science Conference.
"""

generate_forecast_with.__doc__="""\
Generate forecasts using a pre-trained XTFT model based on the forecast
horizon.

There are two approaches to generating forecasts with an XTFT model:

1. A monolithic function (e.g. ``generate_forecast``) that handles both
   single-step and multi-step forecasts within a single implementation.
   This approach results in a single, large function that internally
   branches its logic based on the value of the forecast horizon.
   
2. A modular design where the single-step and multi-step forecasting
   functionalities are separated into two distinct functions (e.g.
   ``forecast_single_step`` and ``forecast_multi_step``), with a thin
   wrapper (e.g. ``generate_xtft_forecast``) that dispatches to the
   appropriate function based on the forecast horizon.

The modular approach (2) is generally preferred because it separates
concerns and improves code readability, maintainability, and unit testing.
Each function becomes responsible for a well-defined task: one for
single-step forecasts and one for multi-step forecasts. The wrapper
function, which we propose to name ``generate_xtft_forecast``, simply
selects the correct method based on the forecast horizon. Use this
approach when your application may need to handle both short- and long-
term forecasts, as it keeps the codebase modular and easier to debug.

Below is an implementation of the wrapper function 
``generate_xtft_forecast`` that calls 
``forecast_single_step`` when ``forecast_horizon`` equals 1 and 
``forecast_multi_step`` when ``forecast_horizon`` is greater than 1.

Parameters
----------
xtft_model : object
    A validated Keras model instance. The model is expected to be
    verified via ``validate_keras_model``.
inputs : list or tuple of numpy.ndarray
    A list containing three elements: ``X_static``, ``X_dynamic``, and
    ``X_future``. If ``spatial_cols`` is provided, it is assumed that
    the first two columns of ``X_static`` correspond to the first and
    second spatial coordinates of the original training data.
forecast_horizon : int
    The number of future time steps to forecast. A value of 1 triggers a
    single-step forecast; values greater than 1 trigger a multi-step forecast.
y : numpy.ndarray, optional
    Actual target values for evaluation. If provided, evaluation metrics
    (e.g., R² Score, and in quantile mode, the coverage score) are computed.
dt_col : str, optional
    Name of the time column (e.g. ``"year"``). If provided, the output
    DataFrame includes a column with these values.
mode : str, optional
    Forecast mode, either ``"quantile"`` or ``"point"``. In quantile mode,
    predictions are generated for multiple quantiles (default: ``[0.1, 0.5,
    0.9]``); in point mode, a single prediction is generated.
spatial_cols : list of str, optional
    List of spatial column names. If provided, it must contain at least two
    elements corresponding to the first and second columns of the original
    training data's ``X_static``.
time_steps : int, optional
    The number of historical time steps used as input.
q : list of float, optional
    List of quantile values for quantile forecasting. Default is
    ``[0.1, 0.5, 0.9]`` when ``mode`` is ``"quantile"``.
tname : str, optional
    Target variable name used to construct output column names (e.g.,
    ``"subsidence"``). Default is ``"target"``.
forecast_dt : any, optional
    Forecast datetime information. If provided and its length matches
    ``forecast_horizon``, its values are added to the output DataFrame.
apply_mask : bool, optional
    If True, applies masking (via ``mask_by_reference``) to adjust
    predictions in non-subsiding areas. Requires that both ``mask_values``
    and ``mask_fill_value`` are provided.
mask_values : scalar, optional
    The reference value(s) used for masking. Must be provided if
    ``apply_mask`` is True.
mask_fill_value : scalar, optional
    The value used to fill masked predictions. Must be provided if
    ``apply_mask`` is True.
savefile : str, optional
    File path to save the forecast results as a CSV file. If not provided,
    a default filename is generated.
verbose : int, optional
    Verbosity level controlling printed output.
**kw: dict, optional 
   Does nothing; here for future extension. 
Returns
-------
pandas.DataFrame
    A DataFrame containing the forecast results. In quantile mode, the
    output includes columns for each quantile and forecast step (e.g.
    ``<tname>_q10_step1``, ``<tname>_q50_step2``, etc.); in point mode,
    it contains a single prediction column per forecast step (e.g.
    ``<tname>_pred_step1``). If ``y`` is provided, an additional column
    (``<tname>_actual``) is included.

Examples
--------

>>> from fusionlab.nn.transformers import XTFT
>>> from fusionlab.nn.utils import generate_forecast_with
>>> import numpy as np
>>> 
>>> # Prepare a dummy XTFT model with example parameters.
>>> my_model = XTFT(
...     static_input_dim=10,
...     dynamic_input_dim=5,
...     future_input_dim=3,
...     forecast_horizon=1,          # This parameter will be updated in the
...                                  # wrapper function based on forecast_horizon.
...     quantiles=[0.1, 0.5, 0.9],
...     embed_dim=32,
...     max_window_size=3,
...     memory_size=100,
...     num_heads=4,
...     dropout_rate=0.1,
...     lstm_units=64,
...     attention_units=64,
...     hidden_units=32
... )
>>> my_model.compile(optimizer='adam')
>>> 
>>> # Create dummy input data.
>>> X_static = np.random.rand(100, 10)
>>> X_dynamic = np.random.rand(100, 3, 5)
>>> X_future  = np.random.rand(100, 3, 3)
>>> y_array   = np.random.rand(100, 1, 1)  # For single-step target output.
>>> inputs    = [X_static, X_dynamic, X_future]
>>> 
>>> # Fit the model with dummy data.
>>> my_model.fit(
...     x=[X_static, X_dynamic, X_future],
...     y=y_array,
...     epochs=1,
...     batch_size=32
... )
>>> 
>>> # Example for a single-step forecast:
>>> forecast_df = generate_forecast_with(
...     xtft_model=my_model,
...     inputs=inputs,
...     forecast_horizon=1,
...     y=y_array,
...     dt_col="year",
...     mode="quantile",
...     spatial_cols=["longitude", "latitude"],
...     tname="subsidence",
...     verbose=3
... )
>>> print(forecast_df.head())
>>> 
>>> # Example for a multi-step forecast:
>>> forecast_dates = ["2023", "2024", "2025", "2026"]
>>> forecast_df = generate_forecast_with(
...     xtft_model=my_model,
...     inputs=inputs,
...     forecast_horizon=4,
...     y=y_array,
...     dt_col="year",
...     mode="point",
...     spatial_cols=["longitude", "latitude"],
...     tname="subsidence",
...     forecast_dt=forecast_dates,
...     verbose=3
... )
>>> print(forecast_df.head())

See Also
--------
forecast_single_step : Generates a single-step forecast.
forecast_multi_step  : Generates a multi-step forecast.
validate_keras_model : Validates a Keras model.
coverage_score       : Computes the coverage score.

References
----------
.. [1] Kouadio et al., "Gofast Forecasting Model", Journal of Advanced
   Forecasting, 2025. (In Review)
"""


reshape_xtft_data.__doc__+=r"""\
The core process involves:
1.  Validating input DataFrame and specified columns.
2.  Optionally grouping the data by `spatial_cols`. If not provided,
    the entire DataFrame is treated as a single group.
3.  Sorting data within each group by the datetime column (`dt_col`).
4.  A two-pass system for efficiency:
    a.  **First Pass:** Iterates through groups to determine the total
        number of valid sequences that can be generated based on
        `time_steps` and `forecast_horizon`. Groups too short to
        form even one complete input-target pair are skipped.
    b.  **Pre-allocation:** Empty NumPy arrays are created with the
        final determined dimensions to store static, dynamic, future,
        and target sequences.
    c.  **Second Pass:** Iterates through valid groups again, creating
        rolling windows and populating the pre-allocated arrays.
5.  For each window:
    * **Static features** (`static_cols`): Extracted once per group if
        `spatial_cols` are used, or from the first time step of the
        current input window if no spatial grouping.
    * **Dynamic features** (`dynamic_cols`): Extracted for the lookback
        period of `time_steps`.
    * **Future features** (`future_cols`): Extracted for a combined
        period covering both the `time_steps` (lookback) and the
        `forecast_horizon`. This provides the model with known future
        information relevant to both the input window and the
        prediction window.
    * **Target features** (`target_col`): Extracted for the
        `forecast_horizon` immediately following the input window.
6.  Optionally saves the generated sequence arrays and metadata to a
    `.joblib` file.

Parameters
----------
df : pandas.DataFrame
    The input DataFrame containing time series data. It must include
    a datetime column specified by `dt_col` and the `target_col`.
dt_col : str
    The name of the datetime column in `df`. This column is
    processed to ensure proper datetime formatting.
target_col : str
    The column in `df` holding the target values for forecasting.
dynamic_cols : List[str]
    A list of column names representing dynamic features that vary
    over time (e.g., past observed values, exogenous variables).
static_cols : List[str], optional
    A list of column names representing static features that are
    time-invariant for each group (if `spatial_cols` are used) or
    per sequence (if no `spatial_cols`). If ``None``, no static
    data is generated. Default is ``None``.
future_cols : List[str], optional
    A list of column names representing known future covariates.
    If ``None``, no future data is generated. Default is ``None``.
spatial_cols : str or List[str], optional
    Column name(s) used to group the DataFrame, typically by a
    spatial identifier (e.g., 'location_id', ['longitude', 'latitude']).
    If ``None``, the entire DataFrame is treated as a single group.
    Default is ``None``.
time_steps : int, default=4
    The number of past time steps to include in each input
    sequence (lookback window).
forecast_horizon : int, default=1
    The number of future time steps to predict for each input
    sequence.
to_datetime : str, optional
    Specifies a conversion rule for `dt_col` if it's not already
    in datetime format (e.g., "auto", "Y", "M", "D"). Passed to
    :func:`~fusionlab.utils.ts_utils.ts_validator`. Default is ``None``.
model : str, default="xtft"
    Indicates the target model type. Currently, this parameter is
    primarily for forward compatibility or specific internal checks
    and does not alter the core reshaping logic for numerical data.
    Supported: {"xtft", "tft", "any", "lstm", None}.
error : str, default='raise'
    Error handling strategy if required columns are missing.
    Options: {'raise', 'warn', 'ignore'}.
savefile : str, optional
    If provided, path to save the generated sequence arrays and
    metadata as a ``.joblib`` file. Default is ``None``.
verbose : int, default=1
    Verbosity level for logging and status messages:
    - ``0``: Silent.
    - ``1``: Basic information (e.g., grouping, total sequences,
      final shapes, save messages).
    - ``2``: More detailed processing steps (e.g., per-group sequence
      counts if `spatial_cols` used).
    - ``3`` (or higher): Debug-level information including internal
      shapes during sequence generation (not typically used).
**kw  
    Placeholder for forward‑compat extensions.
    
Returns
-------
static_data_arr : numpy.ndarray or None
    Array of static feature sequences. Shape:
    :math:`(\text{TotalSequences}, \text{NumStaticFeatures})`.
    Returns ``None`` if `static_cols` is not provided.
dynamic_data_arr : numpy.ndarray
    Array of dynamic feature sequences. Shape:
    :math:`(\text{TotalSequences}, \text{time_steps}, \text{NumDynamicFeatures})`.
future_data_arr : numpy.ndarray or None
    Array of future covariate sequences. Shape:
    :math:`(\text{TotalSequences}, \text{time_steps} + \text{forecast_horizons}, \text{NumFutureFeatures})`.
    Returns ``None`` if `future_cols` is not provided.
target_data_arr : numpy.ndarray
    Array of target value sequences. Shape:
    :math:`(\text{TotalSequences}, \text{forecast_horizons}, 1)`.

Raises
------
ValueError
    If `time_steps` or `forecast_horizons` are not positive integers.
    If not enough data points exist in any group to create sequences.
    If required columns specified in `*_cols` arguments are missing
    and `error='raise'`.
KeyError
    If `target_col` or columns in `dynamic_cols` (or other provided
    `*_cols`) do not exist in the DataFrame and `error='raise'`.

Notes
-----
- The function sorts data by `spatial_cols` (if any) and then by
  `dt_col` before generating sequences. Ensure `dt_col` represents
  a sortable time progression.
- For non-spatial data (`spatial_cols=None`), static features are
  extracted from the first time step of each generated input window.
- The `future_data_arr` is constructed to span both the input
  lookback window (`time_steps`) and the prediction window
  (`forecast_horizons`), providing the model with all known future
  information relevant to the current input and target sequences.

Rolling Window Concept
------------------------
The function constructs rolling windows. For an input sequence
starting at index :math:`j` within a group:
- Dynamic Input :math:`\mathbf{X}^{(j)}_{dyn} = [\mathbf{x}_{j}, ..., \mathbf{x}_{j+T-1}]`
- Future Input :math:`\mathbf{X}^{(j)}_{fut} = [\mathbf{z}_{j}, ..., \mathbf{z}_{j+T+H-1}]`
- Static Input :math:`\mathbf{X}^{(j)}_{stat}` (constant for the sequence)
- Target :math:`\mathbf{Y}^{(j)} = [y_{j+T}, ..., y_{j+T+H-1}]^T`
where :math:`T` is `time_steps` and :math:`H` is `forecast_horizons`.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from fusionlab.nn.utils import reshape_xtft_data

>>> # Example 1: Basic usage without spatial grouping
>>> n_points = 50
>>> df1 = pd.DataFrame({
...     'Date': pd.to_datetime(pd.date_range('2023-01-01', periods=n_points)),
...     'Target': np.arange(n_points),
...     'Dynamic1': np.random.rand(n_points),
...     'Static1_val': np.random.rand(n_points) * 10, # Will be sequence-static
...     'Future1': np.random.rand(n_points) + 5
... })
>>> s, d, f, t = reshape_xtft_data(
...     df1, dt_col='Date', target_col='Target',
...     dynamic_cols=['Dynamic1'], static_cols=['Static1_val'],
...     future_cols=['Future1'], time_steps=5, forecast_horizons=3,
...     verbose=0
... )
>>> print(f"Example 1 Shapes: S={s.shape}, D={d.shape}, F={f.shape}, T={t.shape}")
Example 1 Shapes: S=(43, 1), D=(43, 5, 1), F=(43, 8, 1), T=(43, 3, 1)

>>> # Example 2: With spatial grouping
>>> df_list = []
>>> for group_id in ['A', 'B']:
...     group_df = pd.DataFrame({
...         'Date': pd.to_datetime(pd.date_range('2023-01-01', periods=30)),
...         'Target': np.random.rand(30) + (10 if group_id == 'A' else 20),
...         'Dynamic1': np.random.rand(30),
...         'Static_Group': 100 if group_id == 'A' else 200, # Truly static per group
...         'Future1': np.random.rand(30) + 5,
...         'GroupID': group_id
...     })
...     df_list.append(group_df)
>>> df2 = pd.concat(df_list)
>>> s, d, f, t = reshape_xtft_data(
...     df2, dt_col='Date', target_col='Target',
...     dynamic_cols=['Dynamic1'], static_cols=['Static_Group'],
...     future_cols=['Future1'], spatial_cols=['GroupID'],
...     time_steps=6, forecast_horizons=4, verbose=0
... )
>>> print(f"\nExample 2 Shapes (Spatial): S={s.shape}, D={d.shape}, F={f.shape}, T={t.shape}")
Example 2 Shapes (Spatial): S=(42, 1), D=(42, 6, 1), F=(42, 10, 1), T=(42, 4, 1)

See Also
--------
fusionlab.utils.ts_utils.ts_validator : Validates and converts datetime columns.
fusionlab.core.handlers.columns_manager : Formats and validates column lists.
fusionlab.core.checks.exist_features : Checks for column existence.
fusionlab.utils.io_utils.save_job : Utility for saving processed data.
fusionlab.nn.utils.create_sequences : Simpler sequence creation for basic models.

References
----------
.. [1] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).
   Temporal Fusion Transformers for interpretable multi-horizon
   time series forecasting. *International Journal of Forecasting*,
   37(4), 1748-1764. (Illustrates the type of data structure TFTs expect).

"""
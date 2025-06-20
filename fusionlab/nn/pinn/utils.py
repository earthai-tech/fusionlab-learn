# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com> 

"""
Physics-Informed Neural Network (PINN) Utility functions.
"""
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
    check_empty
)
from ...core.io import SaveFile 
from ...core.diagnose_q import validate_quantiles
from ...core.handlers import columns_manager
from ...utils.validator import validate_positive_integer 
from ...decorators import isdf 
from ...utils.deps_utils import ensure_pkg 
from ...utils.generic_utils import print_box, vlog, select_mode 
from ...utils.geo_utils import resolve_spatial_columns 
from ...utils.io_utils import save_job  

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
        'normalize_for_pinn', 
        'prepare_pinn_data_sequences',
        'format_pinn_predictions', 
        'extract_txy_in', 
        'extract_txy', 
        'plot_hydraulic_head', 
]

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
    verbose: int = 0,
    
    **kwargs
) -> 'pd.DataFrame':
    """Formats PINN model predictions into a structured pandas DataFrame.

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
        verbose=verbose,
        _logger = _logger, 
        **kwargs 
    )

@SaveFile 
def format_pihalnet_predictions(
    pihalnet_outputs: Optional[Dict[str, Tensor]] = None,
    model: Optional[Model] = None,
    model_inputs: Optional[Dict[str, Tensor]] = None,
    y_true_dict: Optional[Dict[str, Union[np.ndarray, Tensor]]] = None,
    target_mapping: Optional[Dict[str, str]] = None,
    include_gwl: bool = True,
    include_coords: bool = True,
    quantiles: Optional[List[float]] = None,
    forecast_horizon: Optional[int] = None,
    output_dims: Optional[Dict[str, int]] = None,
    ids_data_array: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ids_cols: Optional[List[str]] = None,
    ids_cols_indices: Optional[List[int]] = None,
    scaler_info: Optional[Dict[str, Dict[str, Any]]] = None,
    # e.g., {'subsidence': {'scaler': scaler_obj, 
    #                       'all_features': ['f1', 'subs', 'f3'], 'idx': 1}}
    coord_scaler: Optional[Any] = None, 
    evaluate_coverage: bool = False,
    coverage_quantile_indices: Tuple[int, int] = (0, -1),
    savefile: str = None, 
    verbose: int = 0, 
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Formats PIHALNet predictions into a structured pandas DataFrame.

    This function processes the output dictionary from PIHALNet (or
    generates predictions if a model and inputs are provided) and
    transforms them into a long-format DataFrame. It handles dual
    targets (subsidence, GWL), point or quantile forecasts, and can
    optionally include true target values, coordinate information,
    additional identifiers, perform inverse scaling, and evaluate
    quantile coverage.
    
    Structure PIHALNet predictions, targets, coordinates, and user
    metadata into a **long‑format** :pyclass:`pandas.DataFrame`
    ready for post‑processing, plotting, or export.

    The routine accepts either a pre‑computed output dictionary
    from :class:`~fusionlab.nn.pinn.PIHALNet` **or** a compiled model
    and its inputs.  It supports single‑point as well as quantile
    forecasts, dual‑target learning (subsidence / ground‑water level),
    optional inverse scaling, coordinate recovery, quantile‑coverage
    diagnostics, and the injection of arbitrary static ID columns.

    Parameters
    ----------
    pihalnet_outputs : dict[str, Tensor] or None, default=None
        Dictionary returned by ``PIHALNet.predict``.  Keys should
        include ``'subs_pred'`` and/or ``'gwl_pred'``; a
        ``'pde_residual'`` key is silently ignored.  If *None*,
        predictions are generated from *model* and *model_inputs*.
    model : keras.Model or None, default=None
        A compiled PIHALNet instance used to generate predictions
        when *pihalnet_outputs* is *None*.
    model_inputs : dict[str, Tensor] or None, default=None
        Input dictionary matching the signature of
        :pymeth:`PIHALNet.call`.  Required with *model*.
    y_true_dict : dict[str, ndarray | Tensor] or None, default=None
        Ground‑truth arrays keyed by ``'subsidence'`` / ``'gwl'`` or
        by their prediction keys.  Shapes must match
        ``(N, H, O)``.
    target_mapping : dict[str, str] or None, default=None
        Custom mapping between network output keys and column base
        names.  Example: ``{'subs_pred': 'subs', 'gwl_pred': 'gwl'}``.
    include_gwl, include_coords_in_df : bool, default=True
        Toggles the export of GWL predictions and coordinate columns.
    quantiles : list[float] or None, default=None
        Sorted quantile levels *q* such that
        :math:`0 < q_1 < \dots < q_Q < 1`.
    forecast_horizon : int or None, default=None
        Overrides the horizon inferred from prediction shape.
    output_dims : dict[str, int] or None, default=None
        Explicit feature dimension per target.  If *None*, each
        target’s last axis defines :math:`O`.
    ids_data_array : ndarray | DataFrame | Tensor or None, default=None
        Static identifiers (e.g. well ID, lithology) of shape
        ``(N, P)``.
    ids_cols : list[str] or None, default=None
        Column names for *ids_data_array* (DataFrame or ndarray).
    ids_cols_indices : list[int] or None, default=None
        Subset of columns to retain from an *ids_data_array* ndarray.
    scaler_info : dict[str, dict] or None, default=None
        Per‑target inverse‑scaling metadata.  Each sub‑dict must
        contain ``'scaler'``, ``'all_features'``, and ``'idx'``.
    coord_scaler : object or None, default=None
        Scikit‑learn‑compatible scaler used to denormalize the
        coordinate triplet *(t, x, y)*.
    evaluate_coverage : bool, default=False
        Compute unconditional coverage of the central quantile band
        defined by *coverage_quantile_indices*.
    coverage_quantile_indices : tuple[int, int], default=(0, -1)
        Indices ``(L, U)`` pointing to the lower and upper
        quantile columns used in the coverage test.

        .. math::

           \text{Coverage} \;=\;
           \frac{1}{n}\sum_{i=1}^{n}
           \mathbb{1}\!\bigl[
               y_i \in \bigl[\hat{y}^{(L)}_i,
                              \hat{y}^{(U)}_i\bigr]
           \bigr]
    savefile : str or None, default=None
        If given, ``final_df.to_csv(savefile, index=False)`` is called.
    verbose : int, default=0
        Verbosity from *0* (silent) to *5* (trace every step).

    Returns
    -------
    pandas.DataFrame
        Long‑format frame with one row per sample–step pair and
        columns

        * ``sample_idx`` – original sample index :math:`n`.
        * ``forecast_step`` – lead time :math:`h\;(\ge 1)`.
        * ``coord_t``, ``coord_x``, ``coord_y`` – coordinates
          (optional).
        * Prediction columns ``<base>[_<k>]`` or
          ``<base>_q<q%>``.
        * ``<base>_actual`` – ground‑truth targets (optional).
        * Extra ID columns supplied via *ids_data_array*.

    Raises
    ------
    ValueError
        Missing mandatory inputs or shape mismatches.
    RuntimeError
        Model prediction failures.
    TypeError
        Unsupported tensor type in *pihalnet_outputs*.
    Warning
        Non‑fatal issues are logged when *verbose* > 0.

    Notes
    -----
    * The exported DataFrame is **column‑aligned** across targets,
      enabling straightforward melt/merge operations.
    * If *quantiles* is provided and predictions are shape
      ``(N, H, Q, O)``, columns are named
      ``<base>_q{int(q*100)}``.
    * Inverse transforms are applied **after concatenation** to avoid
      duplicate scaler calls.

    Examples
    --------
    >>> from fusionlab.nn.pinn.utils import format_pihalnet_predictions
    >>> df = format_pihalnet_predictions(
    ...     model=pihalnet,
    ...     model_inputs=batch,
    ...     y_true_dict={'subsidence': y_true},
    ...     quantiles=[0.05, 0.50, 0.95],
    ...     ids_data_array=meta_df[['well_id', 'region']],
    ...     ids_cols=['well_id', 'region'],
    ...     verbose=3,
    ... )
    >>> df.head()
       sample_idx  forecast_step  subs_q5  subs_q50  subs_q95  subs_actual
    0           0              1    -2.11     -1.34     -0.54        -1.20
    1           0              2    -2.35     -1.62     -0.80        -1.58
    2           0              3    -2.58     -1.91     -1.01        -1.85
    3           1              1    -3.05     -2.40     -1.72        -2.55
    4           1              2    -3.29     -2.67     -1.99        -2.81

    See Also
    --------
    fusionlab.metrics.coverage_score
    fusionlab.nn.pinn.PIHALNet.predict
    fusionlab.nn.pinn.utils._format_target_predictions

    References
    ----------
    .. [1] Kouadio M. K. *et al.* “Physics‑Informed Heterogeneous
       Attention Learning for Spatio‑Temporal Subsidence Prediction,”
       *IEEE T‑PAMI*, 2025 (in press).
    """
    # **********************************************************
    from ...metrics import coverage_score
    # **********************************************************
    
    vlog(f"Starting PIHALNet prediction formatting (verbose={verbose}).",
         level=3, verbose=verbose, logger=_logger)

    # --- 1. Obtain Model Predictions if not provided ---
    if pihalnet_outputs is None:
        if model is None or model_inputs is None:
            raise ValueError(
                "If 'pihalnet_outputs' is None, both 'model' and "
                "'model_inputs' must be provided."
            )
        vlog("  Predictions not provided, generating from model...",
             level=4, verbose=verbose, logger=_logger)
        try:
            # model.predict expects a format that its call method understands
            # For PIHALNet, it's a dictionary.
            pihalnet_outputs = model.predict(model_inputs, verbose=0)
            if not isinstance(pihalnet_outputs, dict):
                 # If model.predict doesn't return dict (e.g. if it's not PIHALNet)
                 # this indicates a mismatch. Forcing it for PIHALNet's structure.
                 raise ValueError(
                     "Model output is not a dictionary"
                     " as expected from PIHALNet.")
            vlog("  Model predictions generated.", level=5,
                 verbose=verbose, logger=_logger)
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate predictions from model: {e}"
            ) from e
    
    # Ensure predictions are NumPy arrays
    processed_outputs = {}
    for key, val_tensor in pihalnet_outputs.items():
        if key in ['subs_pred', 'gwl_pred']: # Only process expected pred keys
            if hasattr(val_tensor, 'numpy'):
                processed_outputs[key] = val_tensor.numpy()
            elif isinstance(val_tensor, np.ndarray):
                processed_outputs[key] = val_tensor
            else:
                try:
                    processed_outputs[key] = np.array(val_tensor)
                except Exception as e:
                    raise TypeError(
                        f"Could not convert output '{key}' to NumPy. "
                        f"Type: {type(val_tensor)}. Error: {e}"
                    ) from e
        elif key == 'pde_residual':
            # PDE residual can be handled differently if needed
            pass # Not typically added to this output DataFrame

    if not processed_outputs:
        vlog("  No 'subs_pred' or 'gwl_pred'"
             " found in outputs. Returning empty DF.",
             level=1, verbose=verbose, logger=_logger)
        return pd.DataFrame()

    # --- 2. Define Targets to Process and Their Names ---
    if target_mapping is None:
        target_mapping = {'subs_pred': 'subsidence', 'gwl_pred': 'gwl'}

    targets_to_process = {}
    if 'subs_pred' in processed_outputs:
        targets_to_process['subs_pred'] = target_mapping.get(
            'subs_pred', 'subsidence'
            )
    if include_gwl and 'gwl_pred' in processed_outputs:
        targets_to_process['gwl_pred'] = target_mapping.get(
            'gwl_pred', 'gwl'
            )

    if not targets_to_process:
        vlog("  No valid targets to process after"
             " filtering. Returning empty DF.",
             level=1, verbose=verbose, logger=_logger)
        return pd.DataFrame()

    # --- 3. Prepare Base DataFrame (Sample Index and Forecast Step) ---
    # Infer num_samples and horizon from the first available target
    first_pred_key = list(targets_to_process.keys())[0]
    first_pred_array = processed_outputs[first_pred_key]
    
    num_samples = first_pred_array.shape[0]
    H_inferred = forecast_horizon or first_pred_array.shape[1]
    
    if num_samples == 0 or H_inferred == 0:
        vlog("  No samples or zero forecast horizon."
             " Returning empty DF.",
              level=1, verbose=verbose, logger=_logger)
        return pd.DataFrame()

    vlog(f"  Formatting for {num_samples} samples,"
         f" Horizon={H_inferred}.",
         level=4, verbose=verbose, logger=_logger)

    sample_indices = np.repeat(np.arange(num_samples), H_inferred)
    forecast_steps = np.tile(np.arange(1, H_inferred + 1), num_samples)
    
    # List to hold all column DataFrames
    all_data_dfs = [
        pd.DataFrame({
            'sample_idx': sample_indices,
            'forecast_step': forecast_steps
        })
    ]
    
    # --- 4. Add Coordinates if Requested ---
    if include_coords:
        if model_inputs and 'coords' in model_inputs and \
           model_inputs['coords'] is not None:
            coords_arr = model_inputs['coords']
            if hasattr(coords_arr, 'numpy'): 
                coords_arr = coords_arr.numpy()
            
            if coords_arr.shape[0] == num_samples and \
               coords_arr.shape[1] == H_inferred and coords_arr.shape[2] == 3:
                coords_reshaped = coords_arr.reshape(num_samples * H_inferred, 3)
                # --- INSERT THE INVERSE TRANSFORM LOGIC HERE ---
                if coord_scaler is not None:
                    vlog("  Applying inverse transform to t,x,y coordinates...",
                         level=4, verbose=verbose, logger=_logger)
                    try:
                        # Overwrite the reshaped coordinates with their original scale
                        coords_reshaped = coord_scaler.inverse_transform(coords_reshaped)
                    except Exception as e:
                        vlog(f"  WARNING: Could not inverse transform coordinates: {e}."
                             " Using normalized coordinates instead.",
                             level=1, verbose=verbose, logger=_logger)
                # --- END OF NEW LOGIC ---
                coord_names = ['coord_t', 'coord_x', 'coord_y']
                all_data_dfs.append(
                    pd.DataFrame(coords_reshaped, columns=coord_names)
                )
                vlog(f"  Added coordinate columns: {coord_names}",
                     level=4, verbose=verbose, logger=_logger)
            else:
                vlog("  'coords' shape mismatch or not found."
                     " Skipping coordinate columns.",
                     level=2, verbose=verbose, logger=_logger)
        else:
            vlog("  `model_inputs['coords']` not available."
                 " Skipping coordinate columns.",
                 level=2, verbose=verbose, logger=_logger)

    # --- 5. Add Additional Static/ID Columns ---
    # These are static per original sample, so they need to be
    # repeated H_inferred times.
    # `sample_indices` (created earlier as np.repeat(np.arange(num_samples), H_inferred))
    # already provides the correct mapping to expand these static IDs.
    
    if ids_data_array is not None:
        vlog("  Processing additional static/ID columns...",
             level=4, verbose=verbose, logger=_logger)
        
        # Ensure ids_data_array is a NumPy array for consistent indexing
        ids_np_array: np.ndarray
        if isinstance(ids_data_array, pd.DataFrame):
            if ids_cols is None:
                logger.warning(
                    "ids_data_array is a DataFrame but `ids_cols` is "
                    "missing. Using all columns from ids_data_array."
                )
                ids_cols_to_use = list(ids_data_array.columns)
                ids_np_array = ids_data_array.values
            else:
                ids_cols_to_use = list(ids_cols) # Ensure it's a list
                try:
                    ids_np_array = ids_data_array[ids_cols_to_use].values
                except KeyError as e:
                    logger.warning(
                        f"One or more columns in `ids_cols` not found "
                        f"in ids_data_array: {e}. Skipping IDs."
                    )
                    ids_np_array = None # Flag to skip
        elif isinstance(ids_data_array, pd.Series):
            ids_np_array = ids_data_array.to_frame().values # Convert to 2D array
            ids_cols_to_use = [ids_data_array.name or 'id_0'] \
                if ids_cols is None else list(ids_cols)
            if len(ids_cols_to_use) != 1:
                 logger.warning(
                    "ids_data_array is a Series, but `ids_cols` "
                    "suggests multiple columns. Using Series name or 'id_0'."
                )
                 ids_cols_to_use = [ids_cols_to_use[0]]

        elif isinstance(ids_data_array, np.ndarray):
            ids_np_array = ids_data_array
            if ids_cols_indices is not None:
                try:
                    ids_np_array = ids_np_array[:, ids_cols_indices]
                except IndexError as e:
                    logger.warning(
                        f"One or more `ids_cols_indices` out of bounds "
                        f"for ids_data_array: {e}. Skipping IDs."
                    )
                    ids_np_array = None # Flag to skip
            
            # Determine column names for NumPy array
            if ids_cols:
                ids_cols_to_use = list(ids_cols)
                if ids_np_array is not None and \
                   len(ids_cols_to_use) != ids_np_array.shape[1]:
                    logger.warning(
                        f"Length of `ids_cols` ({len(ids_cols_to_use)}) "
                        f"does not match number of selected columns "
                        f"({ids_np_array.shape[1]}) from ids_data_array. "
                        "Using default names 'id_N'."
                    )
                    ids_cols_to_use = [
                        f"id_{k}" for k in range(ids_np_array.shape[1])
                    ]
            elif ids_np_array is not None: # No ids_cols provided for NumPy array
                ids_cols_to_use = [
                    f"id_{k}" for k in range(ids_np_array.shape[1])
                ]
            else: # ids_np_array became None due to error
                ids_cols_to_use = []

        elif hasattr(ids_data_array, 'numpy'): # Check for TensorFlow Tensor
            ids_np_array = ids_data_array.numpy()
            # Recurse with NumPy array logic (similar to above)
            if ids_cols_indices is not None:
                try:
                    ids_np_array = ids_np_array[:, ids_cols_indices]
                except IndexError as e:
                    logger.warning(
                        f"One or more `ids_cols_indices` out of bounds "
                        f"for tensor ids_data_array: {e}. Skipping IDs."
                    )
                    ids_np_array = None
            if ids_cols:
                ids_cols_to_use = list(ids_cols)
                if ids_np_array is not None and \
                    len(ids_cols_to_use) != ids_np_array.shape[1]:
                    logger.warning(
                        f"Length of `ids_cols` ({len(ids_cols_to_use)}) "
                        f"does not match number of selected columns "
                        f"({ids_np_array.shape[1]}) from tensor ids_data_array. "
                        "Using default names 'id_N'."
                    )
                    ids_cols_to_use = [
                        f"id_{k}" for k in range(ids_np_array.shape[1])
                    ]
            elif ids_np_array is not None:
                ids_cols_to_use = [
                    f"id_{k}" for k in range(ids_np_array.shape[1])
                ]
            else:
                ids_cols_to_use = []
        else:
            logger.warning(
                f"Unsupported type for `ids_data_array`: "
                f"{type(ids_data_array)}. Skipping additional ID columns."
            )
            ids_np_array = None

        if ids_np_array is not None:
            num_id_samples = ids_np_array.shape[0]
            if num_id_samples == num_samples:
                # Repeat each static ID row H_inferred times using sample_indices
                # sample_indices has length num_samples * H_inferred
                # and contains values from 0 to num_samples-1, repeated.
                # Example: if num_samples=2, H=3 -> [0,0,0,1,1,1]
                
                # If ids_np_array is 1D (e.g. a single ID column from Series)
                if ids_np_array.ndim == 1:
                    ids_np_array = ids_np_array.reshape(-1, 1)
                    if len(ids_cols_to_use) == 0 and ids_np_array.shape[1]==1:
                        ids_cols_to_use = ['id_0'] # Default name for single col

                if len(ids_cols_to_use) == ids_np_array.shape[1]:
                    expanded_ids_data = ids_np_array[sample_indices // H_inferred]
                    ids_df_part = pd.DataFrame(
                        expanded_ids_data, columns=ids_cols_to_use
                    )
                    all_data_dfs.append(ids_df_part)
                    vlog(f"    Added additional static/ID columns: {ids_cols_to_use}",
                         level=5, verbose=verbose, logger=_logger)
                else:
                    logger.warning(
                        "Mismatch between number of resolved ID column names "
                        f"({len(ids_cols_to_use)}) and columns in processed "
                        f"ID data ({ids_np_array.shape[1]}). Skipping IDs."
                    )
            else:
                vlog(
                    f"  `ids_data_array` has {num_id_samples} samples, but "
                    f"predictions have {num_samples} samples. Skipping ID columns.",
                    level=2, verbose=verbose, logger=_logger
                )
    elif verbose >= 4:
        vlog("  No `ids_data_array` provided, skipping additional static/ID columns.",
             level=4, verbose=verbose, logger=_logger)

    # --- 6. Process Each Target Variable (Subsidence, GWL) ---
    for pred_key, base_name in targets_to_process.items():
        preds_np_target = processed_outputs[pred_key]
        y_true_target = None
        if y_true_dict:
            y_true_target = y_true_dict.get(
                pred_key) or y_true_dict.get(base_name) # Allow both keys
            if y_true_target is not None and hasattr(
                    y_true_target, 'numpy'):
                y_true_target = y_true_target.numpy()

        # Infer output dimension for this specific target
        O_target = (output_dims.get(pred_key) if output_dims 
                    else preds_np_target.shape[-1])
        if quantiles and preds_np_target.ndim == 4: # B, H, Q, O
             O_target= preds_np_target.shape[-1]
        elif quantiles and preds_np_target.ndim == 3 and \
             preds_np_target.shape[-1] % len(quantiles) == 0: # B, H, Q*O or B,H,Q if O=1
             if preds_np_target.shape[-1] == len(quantiles): O_target =1
             else: O_target = preds_np_target.shape[-1] // len(quantiles)

        # --- 6a. Reshape and Add Predictions ---
        pred_cols_target, pred_df_target = _format_target_predictions(
            preds_np_target, num_samples, H_inferred, O_target, 
            base_name, quantiles, verbose
        )
        all_data_dfs.append(pred_df_target)

        # --- 6b. Add Actuals ---
        actual_cols_target = []
        if y_true_target is not None:
            # y_true shape (N, H, O_target)
            if y_true_target.shape == (num_samples, H_inferred, O_target):
                y_true_reshaped = y_true_target.reshape(
                    num_samples * H_inferred, O_target)
                for o_idx in range(O_target):
                    col_name = f"{base_name}"
                    if O_target > 1: col_name += f"_{o_idx}"
                    col_name += "_actual"
                    actual_cols_target.append(col_name)
                all_data_dfs.append(pd.DataFrame(
                    y_true_reshaped, columns=actual_cols_target))
                vlog(f"    Added actuals for {base_name}: {actual_cols_target}",
                     level=5, verbose=verbose, logger=_logger)
            else:
                vlog(f"    y_true shape for {base_name} ({y_true_target.shape}) "
                     f"mismatched expected ({num_samples},"
                     f"{H_inferred},{O_target}). Skipping actuals.",
                     level=2, verbose=verbose, logger=_logger)
        
        # --- 6c. Inverse Scaling (applied per target) ---
        if scaler_info and base_name in scaler_info:
            s_info = scaler_info[base_name]
            scaler = s_info.get('scaler')
            all_feat = s_info.get('all_features') # List of names scaler was fit on
            target_idx = s_info.get('idx')        # Index of this target in all_feat

            if scaler and all_feat and target_idx is not None:
                vlog(f"  Applying inverse transform for {base_name}...",
                     level=4, verbose=verbose, logger=_logger)
                # Create combined list of columns to transform for this target
                cols_to_transform_for_target = pred_cols_target + actual_cols_target
                
                for col_to_inv in cols_to_transform_for_target:
                    if col_to_inv in pd.concat(all_data_dfs, axis=1).columns: 
                        # Check if column exists before access
                        dummy_for_inv = np.zeros((len(sample_indices), len(all_feat)))
                        dummy_for_inv[:, target_idx] = pd.concat(
                            all_data_dfs, axis=1)[col_to_inv].values # Ensure it's a DataFrame
                        
                        inversed_col = scaler.inverse_transform(dummy_for_inv)[:, target_idx]
                        
                        # Update in the correct sub-dataframe (this is tricky)
                        # Better to do this on the final merged df. Store names for now.
                        # For now, let's assume we update after final_df is formed.
                        # This part needs to be done on the final DataFrame.
                        # Storing for later:
                        if not hasattr(scaler_info[base_name], '_cols_to_inv_transform'):
                            scaler_info[base_name]['_cols_to_inv_transform'] = []
                        scaler_info[base_name]['_cols_to_inv_transform'].append(
                            (col_to_inv, inversed_col)
                        )
            else:
                vlog(f"  Scaler info incomplete for {base_name}."
                     " Skipping inverse transform.",
                     level=3, verbose=verbose, logger=_logger)

        # --- 6d. Coverage Score (applied per target) ---
        if ( 
                evaluate_coverage 
                and quantiles 
                and y_true_target is not None 
                and len(quantiles) >= 2 
                and O_target == 1
            ):
            # Assume quantiles_sorted is available if quantiles is not None   
            quantiles_sorted = sorted (
                validate_quantiles(quantiles, dtype=np.float64)
            )
            l_idx, u_idx = coverage_quantile_indices
            lower_q_col = f"{base_name}_q{int(quantiles_sorted[l_idx]*100)}"
            upper_q_col = f"{base_name}_q{int(quantiles_sorted[u_idx]*100)}"
            actual_col = f"{base_name}_actual" # Assumes O_target=1 for actual

            # Access from the currently forming DataFrame parts
            temp_df_for_coverage = pd.concat(all_data_dfs, axis=1)

            if ( 
                    lower_q_col in temp_df_for_coverage and 
                    upper_q_col in temp_df_for_coverage and 
                    actual_col in temp_df_for_coverage
               ):
                
                coverage = coverage_score(
                    temp_df_for_coverage[actual_col],
                    temp_df_for_coverage[lower_q_col],
                    temp_df_for_coverage[upper_q_col]
                )
                vlog(f"  Coverage Score for {base_name} "
                     f"({quantiles_sorted[l_idx]}-"
                     f"{quantiles_sorted[u_idx]}): {coverage:.4f}",
                     level=3, verbose=verbose, logger=_logger)
                # Store it if needed: e.g. 
                # final_df.attrs[f'{base_name}_coverage'] = coverage
            else:
                 vlog(
                     "  Required columns for coverage for"
                     f" {base_name} not found. Skipping.",
                      level=2, verbose=verbose, logger=_logger
                     )

    # --- 7. Concatenate all DataFrames ---
    final_df = pd.concat(all_data_dfs, axis=1)

    # --- Re-apply Inverse Scaling (if stored) ---
    if scaler_info:
        for base_name in targets_to_process.values():
            if ( 
                    base_name in scaler_info 
                    and '_cols_to_inv_transform' 
                    in scaler_info[base_name]
                ):
                for col_name, inversed_values in scaler_info[
                        base_name]['_cols_to_inv_transform']:
                    if col_name in final_df:
                        final_df[col_name] = inversed_values
                vlog(f"  Final inverse transform applied for {base_name}.",
                     level=4, verbose=verbose, logger=_logger)


    vlog("PIHALNet prediction formatting to DataFrame complete.",
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
    output_subsidence_dim: int = 1, 
    output_gwl_dim: int = 1,       
    datetime_format: Optional[str] = None,
    normalize_coords: bool = True, 
    cols_to_scale: Union[List[str], str, None] = None,
    return_coord_scaler: bool =False, 
    mode: Optional[str] =None, 
    savefile: Optional[str] = None,
    verbose: int = 0,
    _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
    **kws
) -> Union[
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[MinMaxScaler]]
]:
    """
    Reshapes and prepares time-series data into sequences for PINN models.

    This function transforms a Pandas DataFrame into structured NumPy arrays
    suitable for training Physics-Informed Neural Networks like PIHALNet.
    It creates sequences of dynamic features, future known features,
    static features, target variables (subsidence and GWL), and importantly,
    spatio-temporal coordinates corresponding to the forecast horizon.
    
    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Tidy time‑series table in **long** format.  Each row must
        correspond to *exactly one* time stamp and (optionally) one
        spatial/entity identifier.  All feature and target columns listed
        below must already exist in *df*.
    time_col : str
        Column holding the observation time.  May be numeric
        (year‑fraction, ordinal day, seconds since epoch, …) or any
        pandas‑recognised datetime dtype; non‑numeric values are converted
        to a numeric “year + fraction‑of‑year” scale.
    subsidence_col, gwl_col : str
        Column names of the **target variables**: vertical land movement
        (subsidence) and ground‑water level (GWL).  Values are sliced to
        produce the horizon targets with shapes  
        ``(N, forecast_horizon, output_✱_dim)``.
    dynamic_cols : list[str]
        Names of **past‑covariate** columns sampled over the look‑back
        window of length :pydata:`time_steps`.  All columns are stacked in
        the order supplied to form  
        ``dynamic_features`` → shape  
        ``(N, time_steps, len(dynamic_cols))``.
    static_cols : list[str] or None, default None
        Columns that are *constant within a group* (e.g. soil type,
        aquifer class).  Only the first value in each group is stored,
        resulting in ``static_features`` of shape  
        ``(N, len(static_cols))``.  Pass *None* if the data set has no
        static covariates.
    future_cols : list[str] or None, default None
        Known‑future covariates such as weather forecasts.  Their temporal
        span depends on :pydata:`mode`:
    
        * ``'pihal_like'`` → length = ``forecast_horizon``  
          (decoder only)
        * ``'tft_like'`` → length = ``time_steps + forecast_horizon``  
          (encoder + decoder)
    
        When *None* an empty ``future_features`` tensor is returned. 
    spatial_cols : (str, str) or None, default None
        Tuple ``(lon, lat)`` if longitude and latitude are already named
        explicitly.  Overrides *lon_col* / *lat_col*.  Required unless the
        spatial columns are supplied separately.
    lon_col, lat_col : str or None, default None
        Fallback column names for longitude and latitude when
        *spatial_cols* is *None*.  Both must be provided together.
    group_id_cols : list[str] or None, default None
        Keys that identify independent trajectories (e.g.
        ``['site_id']`` or ``['site', 'layer']``).  Each group is treated
        as an isolated time‑series for rolling‑window extraction.  If
        *None* the entire DataFrame is processed as one group.
    time_steps : int, default 12
        Length of the look‑back window :math:`T_\text{past}` supplied to
        the encoder.  Must be ≥ 1.
    forecast_horizon : int, default 3
        Prediction length :math:`H`.  Determines the time dimension of the
        decoder outputs and of ``coords``.
    output_subsidence_dim, output_gwl_dim : int, default 1
        Final dimension of each target tensor.  Use > 1 when predicting
        several layers simultaneously.
    datetime_format : str or None, default None
        Custom parsing string forwarded to :pyfunc:`pandas.to_datetime`
        when *time_col* is a string column.
    normalize_coords : bool, default True
        If True, normalizes the 't', 'x', 'y' coordinate values
        (derived from `time_col`, `lon_col`, `lat_col`) to the [0, 1]
        range based on the min/max within each group. This is often
        beneficial for neural network training.
    cols_to_scale : list of str or "auto" or None, default "auto"
        - If a list of column names: scale exactly those columns.
        - If "auto": select all numeric columns, then:
          * Exclude `time_col`, `lon_col`, `lat_col` if `scale_coords=False`.
          * Exclude any columns whose values are only \{0,1\} (assumed one-hot).
        - If None: no extra columns are scaled.
    return_coord_scaler : bool, default False
        When enabled the function returns a three‑tuple
        ``(inputs_dict, targets_dict, coord_scaler)`` where
        *coord_scaler* is the fitted scaler instance or *None* when
        scaling was disabled.  
    mode : {'pihal_like', 'tft_like'} or None, default None  
        Selects the temporal schema used to construct the *future_features*
        window and consequently how the downstream model will consume it.
    
        * **'pihal_like'** – builds future tensors with length equal to the
          forecast horizon :math:`H` only.  These tensors are intended for
          models that feed known‑future covariates exclusively to the
          decoder (PIHALNet convention).
    
        * **'tft_like'** – builds future tensors whose time dimension is
          :math:`T_\text{past} + H`, i.e. the look‑back window
          (*time_steps*) followed by the horizon.  This matches the
          Temporal Fusion Transformer, where the encoder ingests the
          overlapping first segment and the decoder receives the horizon
          segment.
    
        * **None** – treated as ``'pihal_like'`` for backward
          compatibility.
    
        The setting affects the allocated size of
        ``future_features_arr`` and the slice indices used during rolling
        window extraction; make sure it matches the *mode* expected by the
        target model.
    savefile : str or None, default None
        Path to a ``.joblib`` file.  When provided, all produced arrays and
        meta‑data are serialised for reproducibility and faster reloads.
    verbose : int, default 0
        Verbosity level for logging (0-10).
        - 0: Silent.
        - 1: Basic info.
        - 2: More details on shapes and processing.
        - 5: Per-group processing info.
        - 7: Per-sequence sample data (use with caution for large data).
    Returns
    -------
    Union[
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[MinMaxScaler]]
    ]
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
        
        
        If `return_coord_scaler` is False (default):
            A tuple containing two dictionaries: `inputs_dict` and `targets_dict`.
        If `return_coord_scaler` is True:
            A tuple containing three elements: `inputs_dict`, `targets_dict`,
            and `coord_scaler`. `coord_scaler` is the `MinMaxScaler` instance
            used if `normalize_coords` was True (and thus coordinates were
            normalized), otherwise it's `None`.
    Raises
    ------
    ValueError
        If required columns are missing, or if data is insufficient
        to create any sequences.
    TypeError
        If `df` is not a Pandas DataFrame.
        
    Notes
    -----
    * A sequence is generated only if a group contains at least  
    
      .. math::  
         T_\text{past} + H
    
      consecutive observations without gaps.
    * The function is **stateless**; call it each time the configuration
      changes.
    
    Examples
    --------
    Generate rolling windows for two sites with a 6‑step look‑back
    (:pydata:`time_steps=6`) and a 3‑step horizon
    (:pydata:`forecast_horizon=3`).  
    Future covariates are prepared *TFT‑style* (length = ``6 + 3``).
    
    >>> import numpy as np, pandas as pd
    >>> from fusionlab.nn.utils import prepare_pinn_data_sequences
    >>>
    >>> rng = np.random.default_rng(42)
    >>> dates = pd.date_range("2022‑01‑31", periods=18, freq="M")
    >>>
    >>> df = pd.DataFrame({
    ...     "date":        np.tile(dates, 2),
    ...     "site":        ["A"] * 18 + ["B"] * 18,
    ...     "lon":         np.repeat([113.10, 113.25], 18),
    ...     "lat":         np.repeat([22.60,  22.75], 18),
    ...     "subs":        rng.normal(size=36),
    ...     "gwl":         rng.normal(size=36),
    ...     "rain":        rng.random(36),
    ...     "evap":        rng.random(36),
    ...     "forecast_rain": rng.random(36),
    ...     "soil_type":   ["sand"] * 18 + ["clay"] * 18,     # static covariate
    ... })
    >>>
    >>> inputs, targets = prepare_pinn_data_sequences(
    ...     df,
    ...     time_col="date",
    ...     subsidence_col="subs",
    ...     gwl_col="gwl",
    ...     dynamic_cols=["rain", "evap"],
    ...     static_cols=["soil_type"],
    ...     future_cols=["forecast_rain"],
    ...     spatial_cols=("lon", "lat"),
    ...     group_id_cols=["site"],
    ...     time_steps=6,
    ...     forecast_horizon=3,
    ...     mode="tft_like",
    ...     verbose=0,
    ... )
    >>>
    >>> # Inspect a few key tensors
    >>> inputs["dynamic_features"].shape       # N × 6 × 2
    (24, 6, 2)
    >>> inputs["future_features"].shape        # N × (6+3) × 1
    (24, 9, 1)
    >>> targets["subsidence"].shape            # N × 3 × 1
    (24, 3, 1)
    >>> len(inputs["coords"]) == len(targets["gwl"])
    True

    See Also
    --------
    fusionlab.nn.utils.reshape_xtft_data  
        Reshapes arrays for TFT‑style input pipelines.
    fusionlab.nn.utils.create_sequences  
        Generic sliding‑window generator used internally.
    
    """
    
    # Entry log
    vlog("Starting PINN data sequence preparation...",
         verbose=verbose, level=1, logger=_logger)
    
    if verbose >= 1:
        logger.info("Starting PINN data sequence preparation...")
    df_proc = df.copy()

    # --- 1. Validate Input Parameters and Columns ---
    if verbose >= 2:
        logger.debug("Validating input parameters and columns.")

    # Validate essential columns
    vlog("Validating essential columns...",
         verbose=verbose, level=2, logger=_logger)
    
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
    
    # Convert time column to numerical representation
    # (e.g., days since epoch, or normalized year)
    # This is crucial for PINN as 't' is an input for differentiation.
    vlog("Converting time column to numeric values...",
         verbose=verbose, level=4, logger=_logger)
    # Check if the provided time column is already numeric
    if pd.api.types.is_numeric_dtype(df_proc[time_col]):
        numerical_time_col = time_col
        if verbose >= 2:
            logger.debug(
                f"Time column '{time_col}' is already numeric. Using it directly."
            )
    else:
        # If not numeric, then perform the conversion
        try:
            df_proc[time_col] = pd.to_datetime(
                df_proc[time_col], format=datetime_format)
            # Convert to a numerical representation, e.g., year with fraction
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

    # Initialize coord_scaler
    # --- Apply Global Coordinate Normalization (if enabled) ---
    df_proc, coord_scaler, cols_scaler = normalize_for_pinn(
        df=df_proc, 
        time_col= numerical_time_col, 
        coord_x=lon_col, 
        coord_y=lat_col, 
        scale_coords= normalize_coords, 
        cols_to_scale =cols_to_scale, 
        verbose =verbose 
    )

    # --- 2. Group and Sort Data ---
    vlog("Grouping and sorting data...",
        verbose=verbose, level=2, logger=_logger)
    
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
        verbose=verbose, level=4, logger=_logger)
     
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
    # Determine the required length of the future features sequence
    if mode == 'tft_like':
        # For TFT style, we need future values for both the lookback
        # and forecast periods.
        future_window_len = time_steps + forecast_horizon
        vlog(f"Allocating future features array for 'tft_like' mode "
             f"with time dimension: {future_window_len}",
             level=2, verbose=verbose, logger=_logger
            )
    else: # 'pihal_like' mode
        # For PIHALNet's encoder-decoder, we only need future values
        # for the forecast horizon.
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

    # --- 5. Second Pass: Populate Arrays with Rolling Windows ---
    current_seq_idx = 0
    if verbose >= 2:
        logger.debug("Second pass: Populating sequence arrays...")
    
    vlog("Populating arrays with data...",
        verbose=verbose, level=2, logger=_logger)
    
    for group_df in valid_group_dfs: # Iterate over pre-filtered valid groups
        group_t_coords = group_df[numerical_time_col].values
        group_x_coords = group_df[lon_col].values
        group_y_coords = group_df[lat_col].values

        # # Normalization parameters for coordinates (per group if grouped)
        # t_min, t_max = group_t_coords.min(), group_t_coords.max()
        # x_min, x_max = group_x_coords.min(), group_x_coords.max()
        # y_min, y_max = group_y_coords.min(), group_y_coords.max()
        
        # Normalization parameters for coordinates (per group if grouped)
        # These are now only for reference if normalize_coords was False,
        # or for potential debugging. The actual normalization happens globally if enabled.
        t_min_group, t_max_group = group_t_coords.min(), group_t_coords.max() 

        if verbose >= 3:
            print()
            time_scale_info = f"{t_min_group:.4f}-{t_max_group:.4f}"
            if normalize_coords and coord_scaler: # Check if scaler exists
                # Attempt to show original scale for 't' if possible
                # This requires careful handling as group_t_coords might be a slice
                # For simplicity, we'll just note if it's normalized.
                time_scale_info += " (normalized)"
            else:
                time_scale_info += " (original scale)"
            
            print_box(
               # f"Group window: t {t_min}-{t_max}",
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
                if future_cols and num_future_feats > 0:
                    if mode == 'tft_like':
                        # For TFT mode, we need the full window:
                        # from the start of the dynamic window to the end of the horizon.
                        future_start_idx = i
                        future_end_idx = i + time_steps + forecast_horizon
                    else: # 'pihal_like' mode
                        # For PIHAL mode, we only need the horizon window.
                        future_start_idx = horizon_start_idx
                        future_end_idx = horizon_end_idx
                
                future_features_arr[current_seq_idx] = group_df.iloc[
                    future_start_idx:future_end_idx
                ][future_cols].values.astype(np.float32)

            # Coordinates for the forecast horizon
            t_horizon = group_t_coords[horizon_start_idx:horizon_end_idx]
            x_horizon = group_x_coords[horizon_start_idx:horizon_end_idx]
            y_horizon = group_y_coords[horizon_start_idx:horizon_end_idx]

            # if normalize_coords:
            #     # Avoid division by zero if min == max 
            #     # (e.g., single point in time/space for horizon)
            #     t_horizon = (t_horizon - t_min) / (t_max - t_min + 1e-9)
            #     x_horizon = (x_horizon - x_min) / (x_max - x_min + 1e-9)
            #     y_horizon = (y_horizon - y_min) / (y_max - y_min + 1e-9)
            
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
                verbose=verbose, level=7, logger=_logger
            )
            
            current_seq_idx += 1
    
    if verbose >= 1:
        logger.info("Successfully populated sequence arrays.")
    
    vlog("Data population complete.",
         verbose=verbose, level=1, logger=_logger)
    
    inputs_dict = {
        'coords': coords_horizon_arr, # Shape: (N, forecast_horizon, 3)
        'static_features': static_features_arr if num_static_feats > 0 else None,
        # Shape: (N, time_steps, ...)
        'dynamic_features': dynamic_features_arr,
        # Shape: (N, forecast_horizon, ...)
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

        }
        # Add version information if available
        try:
            job_dict.update(get_versions())
        except NameError: # If get_versions is not defined/imported
            vlog("\n  `get_versions` not found, version info not saved.", 
                 verbose=verbose, level =1, logger=_logger)

        try:
            # save_job to be a wrapper around joblib.dump
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

@check_empty(['df']) 
def normalize_for_pinn(
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

def _extract_txy_in(
    inputs: Union[Tensor, np.ndarray, Dict[str, Union[Tensor, np.ndarray]]],
    coord_slice_map: Optional[Dict[str, int]] = None, 
) -> Tuple[Tensor, Tensor, Tensor]:
    """
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
    """
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
    """
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
    """
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
    """Generate and plot a 2D contour map of a hydraulic head solution.

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
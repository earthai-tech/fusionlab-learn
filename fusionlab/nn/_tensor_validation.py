# -*- coding: utf_8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import warnings
from typing import List, Tuple, Optional, Union, Dict, Any
from ..utils.deps_utils import ensure_pkg 
from ..compat.tf import ( 
    optional_tf_function, 
    suppress_tf_warnings, 
    tf_debugging_assert_equal
)
from ..compat.tf import  HAS_TF, TFConfig 
from . import KERAS_DEPS, KERAS_BACKEND

import numpy as np 

if KERAS_BACKEND:
    Tensor=KERAS_DEPS.Tensor

    tf_shape = KERAS_DEPS.shape
    tf_float32=KERAS_DEPS.float32
    tf_int32=KERAS_DEPS.int32
    tf_convert_to_tensor =KERAS_DEPS.convert_to_tensor 
    tf_cast=KERAS_DEPS.cast 
    tf_reduce_all=KERAS_DEPS.reduce_all
    tf_equal=KERAS_DEPS.equal 
    tf_debugging= KERAS_DEPS.debugging 
    tf_pad = KERAS_DEPS.pad 
    tf_rank= KERAS_DEPS.rank 
    tf_less =KERAS_DEPS.less
    tf_constant =KERAS_DEPS.constant
    tf_assert_equal=KERAS_DEPS.assert_equal
    tf_autograph=KERAS_DEPS.autograph
    tf_concat = KERAS_DEPS.concat
    register_keras_serializable=KERAS_DEPS.register_keras_serializable
    tf_expand_dims=KERAS_DEPS.expand_dims
    tf_control_dependencies=KERAS_DEPS.control_dependencies
    tf_get_static_value = KERAS_DEPS.get_static_value
    
    if hasattr(tf_autograph, 'set_verbosity'):
        tf_autograph.set_verbosity(0) 
    
else: 
   # Warn the user that TensorFlow
   # is required for this module
    warnings.warn(
        "TensorFlow is not installed. Please install"
        " TensorFlow to use this module.",
        ImportWarning
    )

    class Tensor: pass 
    def tf_shape(t): return np.array(t.shape)
    def tf_concat(t, axis): return np.concatenate(t, axis=axis)
    def tf_expand_dims(t, axis): return np.expand_dims(t, axis=axis)
    def tf_pad(tensor, paddings, mode="CONSTANT", constant_values=0):
        return np.pad(tensor, paddings, mode=mode, constant_values=constant_values)
    class tf_debugging:
        @staticmethod
        def assert_greater_equal(a,b,message): assert a >= b, message

    tf_float32 = np.float32
    tf_int32 = np.int32
    def tf_convert_to_tensor(x, dtype=None): return np.array(x, dtype=dtype)
    def tf_cast(x, dtype): return x.astype(dtype)
    def tf_equal(x, y): return np.equal(x, y)

    def tf_rank(x): return np.ndim(x)
    def tf_less(x,y): return np.less(x,y)


if HAS_TF:
    config = TFConfig()
#     # Enable compatibility mode for ndim
#     config.compat_ndim_enabled = True 

# --------------------------- tensor validation -------------------------------

def set_anomaly_config(
        anomaly_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Processes the anomaly_config dictionary to ensure it contains
    'anomaly_scores' and 'anomaly_loss_weight' keys.

    Parameters:
    - anomaly_config (Optional[Dict[str, Any]]): 
        A dictionary that may contain:
            - 'anomaly_scores': Precomputed anomaly scores tensor.
            - 'anomaly_loss_weight': Weight for anomaly loss.

    Returns:
    - Dict[str, Any]: 
        A dictionary with keys 'anomaly_scores' and 'anomaly_loss_weight',
        setting them to None if they were not provided.
    """
    if anomaly_config is None:
        return {'anomaly_loss_weight': None, 'anomaly_scores': None}
    
    # Create a copy to avoid mutating the original dictionary
    config = anomaly_config.copy()

    # Ensure 'anomaly_scores' key exists
    if 'anomaly_scores' not in config:
        config['anomaly_scores'] = None

    # Ensure 'anomaly_loss_weight' key exists
    if 'anomaly_loss_weight' not in config:
        config['anomaly_loss_weight'] = None

    return config

@ensure_pkg(
    'tensorflow',
    extra="Need 'tensorflow' for this function to proceed."
)
def validate_anomaly_scores_in(
        scores_tensor: Tensor
    ) -> Tensor:
    """
    Validate and format anomaly scores tensor to ensure proper
    shape and type.

    Parameters
    ----------
    scores_tensor : Tensor
        Input anomaly scores tensor of any shape. Will be converted to:
        - dtype: tf_float32
        - shape: (batch_size, features) with at least 2 dimensions

    Returns
    -------
    Tensor
        Validated anomaly scores tensor with:
        - dtype: tf_float32
        - shape: (batch_size, features) where features >= 1

    Raises
    ------
    ValueError
        If input cannot be converted to TensorFlow tensor
    TypeError
        If input contains invalid non-numeric types

    Notes
    -----
    1. Automatically adds feature dimension if missing
    2. Ensures float32 precision for numerical stability
    3. Designed for internal use with anomaly detection workflows

    Examples
    --------
    >>> valid_scores = validate_anomaly_scores_in([0.1, 0.5, 0.3])
    >>> valid_scores.shape
    TensorShape([3, 1])

    >>> valid_scores = validate_anomaly_scores_in([[0.2], [0.4], [0.9]])
    >>> valid_scores.shape
    TensorShape([3, 1])

    See Also
    --------
    validate_anomaly_scores : Full validation with config handling
    CombinedTotalLoss : Usage of validated scores in loss calculation
    """

    # Check and convert tensor type
    if not isinstance(scores_tensor, Tensor):
        try:
            scores_tensor = tf_convert_to_tensor(
                scores_tensor, dtype=tf_float32
            )
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid anomaly scores input: {e}\n"
                "Expected array-like or TensorFlow tensor."
            ) from e

    # Ensure float32 precision
    scores_tensor = tf_cast(scores_tensor, tf_float32)

    # Add feature dimension if needed
    if len(scores_tensor.shape) != 2:
        scores_tensor = tf_expand_dims(scores_tensor, -1)

    return scores_tensor

@ensure_pkg(
    'tensorflow',
    extra="Requires TensorFlow for anomaly score validation"
)
def validate_anomaly_config(
    anomaly_config: Optional[Dict[str, Any]],
    forecast_horizon: int=1,
    default_anomaly_loss_weight: float = 1.0,
    strategy: Optional[str] = None, 
    return_loss_weight: bool=False, 
) -> Tuple[Dict[str, Any], Optional[str], float]:
    """
    Validates and processes anomaly detection configuration with strategy-aware checks.

    Parameters
    ----------
    anomaly_config : Optional[Dict[str, Any]]
        Configuration dictionary containing:
        - anomaly_scores: Tensor of shape (batch_size, forecast_horizon)
        - anomaly_loss_weight: Float weight for loss component
    forecast_horizon : int
        Expected number of forecasting steps
    default_anomaly_loss_weight : float, default=1.0
        Default weight if not specified in config
    strategy : Optional[str], optional
        Anomaly detection strategy to validate against

    Returns
    -------
    Tuple[Dict[str, Any], Optional[str], float]
        1. Validated configuration dictionary
        2. Active strategy (None if invalid)
        3. Final anomaly loss weight

    Raises
    ------
    ValueError
        For invalid tensor shapes in 'from_config' strategy
    TypeError
        For non-numeric anomaly loss weights
    """
    # Initialize with default-safe configuration
    config = set_anomaly_config(anomaly_config or {})
    active_strategy = strategy
    # Update the weight with the default in dict if None 
    loss_weight = config.get(
        'anomaly_loss_weight') or  default_anomaly_loss_weight
    # keep updated update the config dict 
    config.update({'anomaly_loss_weight': loss_weight})
    
    # Strategy-specific validation
    if active_strategy == 'from_config':
        try:
            scores = validate_anomaly_scores(
                config, 
                forecast_horizon=forecast_horizon,
                mode='strict'
            )
            config['anomaly_scores'] = scores
        except (ValueError, TypeError) as e:
            warnings.warn(
                f"Disabled anomaly detection: {e}",
                UserWarning
            )
            active_strategy = None
            config['anomaly_scores'] = None

    # Weight validation with type safety
    if (weight := config.get('anomaly_loss_weight')) is not None:
        if isinstance(weight, (int, float)):
            loss_weight = float(weight)
        else:
            warnings.warn(
                f"Ignoring invalid weight type {type(weight).__name__}, "
                f"using default {default_anomaly_loss_weight}",
                UserWarning
            )
    # Update the weight with the default in dict if None 
    config.update ({ 
        'anomaly_loss_weight': loss_weight
        })
    
    if return_loss_weight : 
        return config, active_strategy, loss_weight 
    
    return config, active_strategy

@ensure_pkg(
    'tensorflow',
    extra="Need 'tensorflow' for this function to proceed."
)
def validate_anomaly_scores(
    anomaly_config: Optional[Dict[str, Any]],
    forecast_horizon: Optional[int]=None,
    mode: str= 'strict', 
) -> Optional[Tensor]:
    """
    Validates and processes the ``anomaly_scores`` in the provided 
    `anomaly_config` dictionary.

    Parameters:
    - ``anomaly_config`` (Optional[`Dict[str, Any]`]): 
        Dictionary that may contain:
            - 'anomaly_scores': Precomputed anomaly scores tensor.
            - 'anomaly_loss_weight': Weight for anomaly loss.
    - ``forecast_horizon`` (int): 
        The expected number of forecast horizons (second dimension 
        of `anomaly_scores`).
    - ``mode`` (str) : 
        The mode for checking the anomaly score. In ``strict`` mode, 
        anomaly score should exclusively be 2D tensor. In 'soft' mode
        can expand dimensions to fit the 2D dimensons. 

    Returns:
    - Optional[`Tensor`]: 
        Validated `anomaly_scores` tensor of shape 
        (batch_size, forecast_horizons), cast to float32.
        Returns None if `anomaly_scores` is not provided.

    Raises:
    - ValueError: 
        If `anomaly_scores` is provided but is not a 2D tensor or the 
        second dimension does not match `forecast_horizons`.
        
    See Also: 
        validate_anomaly_scores_in: 
            Anomaly scores validated in ``'soft'`` mode
    """

    if anomaly_config is None:
        # If `anomaly_config` is None, no `anomaly_scores` or 
        # `anomaly_loss_weight` are set
        return None

    if isinstance(anomaly_config, dict):
        # Ensure 'anomaly_scores' key exists in the dictionary
        if 'anomaly_scores' not in anomaly_config:
            anomaly_config['anomaly_scores'] = None

        anomaly_scores = anomaly_config.get('anomaly_scores')
    else:
        # Assume `anomaly_scores` is passed directly as `anomaly_config`
        anomaly_scores = anomaly_config
        anomaly_config = {}

    if anomaly_scores is not None:
        # Convert to tensor if not already a TensorFlow tensor
        if not isinstance(anomaly_scores, Tensor):
            try:
                anomaly_scores = tf_convert_to_tensor(
                    anomaly_scores,
                    dtype=tf_float32
                )
            except (ValueError, TypeError) as e:
                raise ValueError(
                    "Failed to convert `anomaly_scores`"
                    f" to a TensorFlow tensor: {e}"
                )
        else:
            # Cast to float32 if it's already a tensor
            anomaly_scores = tf_cast(anomaly_scores, tf_float32)

        if mode !='strict': # in soft" mode, expand dim. 
            return validate_anomaly_scores_in(anomaly_scores) 
        
        
        # Validate that `anomaly_scores` is a 2D tensor
        if len(anomaly_scores.shape) != 2:
            raise ValueError(
                f"`anomaly_scores` must be a 2D tensor with shape "
                f"(batch_size, forecast_horizon), but got "
                f"{len(anomaly_scores.shape)}D tensor."
            )

        # Validate that the second dimension matches `forecast_horizons`
        if anomaly_scores.shape[1] != forecast_horizon:
            raise ValueError(
                f"`anomaly_scores` second dimension must be "
                f"{forecast_horizon}, but got "
                f"{anomaly_scores.shape[1]}."
            )

        # Update the `anomaly_config` with the processed 
        # `anomaly_scores` tensor
        anomaly_config['anomaly_scores'] = anomaly_scores
        return anomaly_scores

    else:
        # If `anomaly_scores` is not provided, ensure it's set to None
        anomaly_config['anomaly_scores'] = None
        return anomaly_scores

@optional_tf_function
def validate_tft_inputs(
    inputs : Union[List[Any], Tuple[Any, ...]],
    dynamic_input_dim : int,
    static_input_dim : Optional[int] = None,
    future_covariate_dim : Optional[int] = None,
    error: str = 'raise'
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """
    Validate and process the input tensors for TFT (Temporal Fusion
    Transformer) models in a consistent manner.

    The function enforces that ``dynamic_input_dim`` (past inputs)
    is always provided, while ``static_input_dim`` and 
    ``future_covariate_dim`` can be `None`. Depending on how many 
    items are in `inputs`, this function decides which item 
    corresponds to which tensor (past, static, or future). It also 
    converts each valid item to a :math:`\\text{tf_float32}` tensor, 
    verifying shapes and optionally raising or warning if invalid 
    conditions occur.

    Parameters
    ----------
    inputs : 
        list or tuple of input items. 
        - If length is 1, interpret as only dynamic inputs.
        - If length is 2, interpret second item either as static 
          or future inputs, depending on whether 
          ``static_input_dim`` or ``future_covariate_dim`` is set.
        - If length is 3, interpret them as 
          (past_inputs, future_inputs, static_inputs) in order.
    dynamic_input_dim : int
        Dimensionality of the dynamic (past) inputs. This is 
        mandatory for the TFT model.
    static_input_dim : int, optional
        Dimensionality of static inputs. If not `None`, expects 
        a second or third item in ``inputs`` to be assigned 
        as static inputs.
    future_covariate_dim : int, optional
        Dimensionality of future covariates. If not `None`, 
        expects a second or third item in ``inputs`` to be 
        assigned as future inputs.
    error : str, default='raise'
        Error-handling strategy if invalid conditions arise.
        - `'raise'` : Raise a `ValueError` upon invalid usage.
        - `'warn'`  : Issue a warning and proceed.
        - `'ignore'`: Silence the issue and proceed (not 
          recommended).

    Returns
    -------
    tuple of Tensor
        Returns a three-element tuple 
        (past_inputs, future_inputs, static_inputs). 
        - `past_inputs` is always present.
        - `future_inputs` or `static_inputs` may be `None` if 
          not provided or `None` in shape.

    Notes
    -----
    If the length of `inputs` is three but one of 
    ``static_input_dim`` or ``future_covariate_dim`` is `None`, 
    then based on ``error`` parameter, a `ValueError` is raised, 
    a warning is issued, or the issue is silently ignored.
    
    .. math::
        \\text{past\\_inputs} \\in 
            \\mathbb{R}^{B \\times T \\times \\text{dynamic\\_input\\_dim}}
        \\quad
        \\text{future\\_inputs} \\in 
            \\mathbb{R}^{B \\times T' \\times \\text{future\\_covariate\\_dim}}
        \\quad
        \\text{static\\_inputs} \\in 
            \\mathbb{R}^{B \\times \\text{static\\_input\\_dim}}
            
    Examples
    --------
    >>> from fusionlab.nn._tensor_validation import validate_tft_inputs
    >>> import tensorflow as tf
    >>> # Example with only past (dynamic) inputs
    >>> single_input = tf_random.normal([8, 24, 10])  # batch=8, time=24
    >>> past, fut, stat = validate_tft_inputs(
    ...     [single_input], dynamic_input_dim=10
    ... )
    >>> print(past.shape)
    (8, 24, 10)
    >>> print(fut, stat)  # None, None

    >>> # Example with two inputs: dynamic past and static
    >>> dynamic_in = tf_random.normal([8, 24, 20])
    >>> static_in  = tf_random.normal([8, 5])
    >>> past, fut, stat = validate_tf_inputs(
    ...     [dynamic_in, static_in],
    ...     dynamic_input_dim=20,
    ...     static_input_dim=5
    ... )
    >>> print(past.shape, stat.shape)
    (8, 24, 20) (8, 5)
    >>> print(fut)  # None

    See Also
    --------
    Other internal functions that manipulate or validate 
    TFT inputs.

    References
    ----------
    .. [1] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2019).
           Temporal Fusion Transformers for Interpretable
           Multi-horizon Time Series Forecasting.
    """

    # 1) Basic checks and shape verifications.
    if not isinstance(inputs, (list, tuple)):
        inputs= [inputs] # When single input is provided
        msg = ("`inputs` must be a list or tuple, got "
               f"{type(inputs)} instead.")
        if error == 'raise':
            raise ValueError(msg)
        elif error == 'warn':
            warnings.warn(msg)
        # if error=='ignore', do nothing

    num_inputs = len(inputs)

    # 2) Convert each item to tf_float32 and gather shapes
    def to_float32_tensor(x: Any) -> Tensor:
        """Convert x to tf_float32 tensor."""
        tensor = tf_convert_to_tensor(x)
        if tensor.dtype != tf_float32:
            tensor = tf_cast(tensor, tf_float32)
        return tensor

    # Initialize placeholders
    past_inputs: Optional[Tensor] = None
    future_inputs : Optional[Tensor] = None
    static_inputs : Optional[Tensor] = None

    # 3) Assign based on how many items are in `inputs`
    if num_inputs == 1:
        # Only dynamic/past inputs
        past_inputs = to_float32_tensor(inputs[0])

    elif num_inputs == 2:
        # We have past + either static or future
        # Decide based on static_input_dim / future_covariate_dim
        past_inputs = to_float32_tensor(inputs[0])
        second_data = to_float32_tensor(inputs[1])

        if static_input_dim is not None and future_covariate_dim is None:
            # second_data is static
            static_inputs = second_data
        elif static_input_dim is None and future_covariate_dim is not None:
            # second_data is future
            future_inputs = second_data
        else:
            # ambiguous or invalid
            msg = ("With two inputs, must have either "
                   "`static_input_dim` or `future_covariate_dim` "
                   "set, but not both or neither.")
            if error == 'raise':
                raise ValueError(msg)
            elif error == 'warn':
                warnings.warn(msg)
            # if error == 'ignore', do nothing

    elif num_inputs == 3:
        # We have past + future + static
        # Check if both static_input_dim and future_covariate_dim
        # are defined
        if (static_input_dim is None or future_covariate_dim is None):
            msg = ("Expect three inputs for past, future, "
                   "and static. But one of `static_input_dim` "
                   "or `future_covariate_dim` is None.")
            if error == 'raise':
                raise ValueError(msg)
            elif error == 'warn':
                warnings.warn(msg)
            # if error == 'ignore', do nothing

        past_inputs   = to_float32_tensor(inputs[0])
        future_inputs = to_float32_tensor(inputs[1])
        static_inputs = to_float32_tensor(inputs[2])

    else:
        # Invalid length
        msg = (f"`inputs` has length {num_inputs}, but only 1, 2, or 3 "
               "items are supported.")
        if error == 'raise':
            raise ValueError(msg)
        elif error == 'warn':
            warnings.warn(msg)
        # if error == 'ignore', do nothing

    # 4) Additional shape checks (e.g., batch size consistency).
    non_null_tensors = [
        x for x in [past_inputs, future_inputs, static_inputs] 
        if x is not None
    ]

    # If we have at least one non-None tensor, let's define a reference
    # batch size from the first. We'll do a static shape check if 
    # possible. If shape[0] is None, we do a dynamic check with tf_shape().
    if non_null_tensors:
        # For simplicity, let's define a function to get batch size.
        # If static shape is None, we fallback to tf_shape(x)[0].
        def get_batch_size(t: Tensor) -> Union[int, Tensor]:
            """Return the first-dim batch size, static if available."""
            if t.shape.rank and t.shape[0] is not None:
                return t.shape[0]  # static shape
            return tf_shape(t)[0]  # fallback to dynamic

        # Reference batch size
        ref_batch_size = get_batch_size(non_null_tensors[0])

        # Check all other non-null items
        for t in non_null_tensors[1:]:
            batch_size = get_batch_size(t)
            # We compare them in a consistent manner. If either
            # is a Tensor, we rely on tf_equal or a python check 
            # if both are python ints. We'll do a python approach 
            # if they're both int, else a tf_cond approach if needed.
            if (isinstance(ref_batch_size, int) and 
                isinstance(batch_size, int)):
                # Both are static
                if ref_batch_size != batch_size:
                    msg = (f"Inconsistent batch sizes among inputs. "
                           f"Got {ref_batch_size} vs {batch_size}.")
                    if error == 'raise':
                        raise ValueError(msg)
                    elif error == 'warn':
                        warnings.warn(msg)
                    # if error=='ignore', do nothing
            else:
                # At least one is dynamic. We'll do a tf_level check.
                # In eager mode, we can still evaluate it directly. 
                # Let's do so carefully.
                are_equal = tf_reduce_all(
                    tf_equal(ref_batch_size, batch_size)
                )
                if not bool(are_equal.numpy()): # are_equal.numpy()
                    msg = ("Inconsistent batch sizes among inputs. "
                           "Got a mismatch in dynamic shapes.")
                    if error == 'raise':
                        raise ValueError(msg)
                    elif error == 'warn':
                        warnings.warn(msg)
                    # if error=='ignore', do nothing

    # 5) Return the triple (past_inputs, future_inputs, static_inputs)
    return past_inputs, future_inputs, static_inputs



@optional_tf_function
def validate_xtft_inputs(
    inputs: List[Optional[Union[np.ndarray, Tensor]]],
    static_input_dim: Optional[int] = None,
    dynamic_input_dim: Optional[int] = None,
    future_covariate_dim: Optional[int] = None,
    forecast_horizon: Optional[int] = None, # For future span check
    error: str = "raise",
    verbose: int = 0
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Validates and standardizes inputs for XTFT-like models.
    Expects `inputs` list as [static, dynamic, future].
    Returns (static_processed, dynamic_processed, future_processed).
    (Full docstring omitted for brevity as requested)
    """
    # Function entry log, controlled by verbosity.
    if verbose >= 2: # Level 2 for major function entries
        print(
            f"Enter `validate_xtft_inputs` (verbose={verbose})"
            )

    # --- 1. Basic Input Structure Validation ---
    # Ensure `inputs` is a list/tuple of exactly 3 elements.
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
        msg = (
            "`inputs` must be a list or tuple of 3 elements: "
            "[static_input, dynamic_input, future_input]."
            f" Received {len(inputs)} elements of type {type(inputs)}."
            )
        if error == 'raise':
            raise ValueError(msg)
        # If not raising, warn and attempt to proceed if possible,
        # though this state is likely an error from the caller.
        warnings.warn(msg, UserWarning)
        # Fallback or further error handling might be needed if
        # `inputs` structure is fundamentally wrong. For now, assume
        # the caller (e.g., XTFT.call) ensures a 3-element list.

    # Unpack inputs based on the expected order:
    # [Static, Dynamic, Future]
    static_raw, dynamic_raw, future_raw = inputs

    if verbose >= 3: # Level 3 for initial state logging
        s_shape = getattr(static_raw, 'shape', "None")
        d_shape = getattr(dynamic_raw, 'shape', "None")
        f_shape = getattr(future_raw, 'shape', "None")
        print(
            f"  Raw input shapes: Static={s_shape}, "
            f"Dynamic={d_shape}, Future={f_shape}"
        )

    # --- 2. Type Conversion and Rank/Dimension Checks ---
    processed_tensors: List[Optional[Tensor]] = []
    # Define properties for each input type in the order:
    # static, dynamic, future (matching the `inputs` list order).
    input_properties = [
        {"name": "Static", "data": static_raw,
         "feat_dim": static_input_dim, "expected_rank": 2},
        {"name": "Dynamic", "data": dynamic_raw,
         "feat_dim": dynamic_input_dim, "expected_rank": 3},
        {"name": "Future", "data": future_raw,
         "feat_dim": future_covariate_dim, "expected_rank": 3},
    ]

    for prop_idx, prop in enumerate(input_properties):
        name = prop["name"]
        data_input = prop["data"]
        expected_feat_dim = prop["feat_dim"]
        expected_rank_int = prop["expected_rank"] # Python int

        if verbose >= 4: # Level 4 for per-input validation step
            print(f"    Validating {name} input...")

        if data_input is not None:
            # Convert to TensorFlow tensor and ensure float32 type.
            if not isinstance(data_input, Tensor): # tf.Tensor
                try:
                    data_input = tf_convert_to_tensor(
                        data_input, dtype=tf_float32
                        )
                except Exception as e:
                    # Catch conversion errors.
                    raise TypeError(
                        f"Failed to convert {name} input to tensor: {e}"
                        ) from e
            elif data_input.dtype != tf_float32:
                data_input = tf_cast(data_input, tf_float32)

            # Check rank using TensorFlow operations for graph safety.
            current_rank_tensor = tf_rank(data_input)
            expected_rank_tensor = tf_constant(
                expected_rank_int, dtype=current_rank_tensor.dtype
                )
            # Graph-compatible assertion for rank.
            tf_debugging.assert_equal(
                current_rank_tensor, expected_rank_tensor,
                message=(
                        f"{name} input must be {expected_rank_int}D. "
                        f"Got rank {current_rank_tensor}. "
                        f"Input shape: {tf_shape(data_input)}."
                    ),
                # data=[current_rank_tensor, tf_shape(data_input)],
                summarize=3 # Show more shape details in error.
            )

            # Check feature dimension (last dimension of the tensor).
            if expected_feat_dim is not None:
                actual_feat_dim_tensor = tf_shape(data_input)[-1]
                expected_feat_dim_tensor = tf_constant(
                    expected_feat_dim,
                    dtype=actual_feat_dim_tensor.dtype
                    )
                tf_debugging.assert_equal(
                    actual_feat_dim_tensor, expected_feat_dim_tensor,
                    # message=(
                    #     f"{name} input last dimension mismatch. "
                    #     f"Expected {expected_feat_dim}, got feature dim "
                    #     f"for input shape."
                    #     ),
                    message=(
                        f"{name} input last dimension mismatch. "
                        f"Expected {expected_feat_dim}, got feature dim"
                        f" {actual_feat_dim_tensor}. "
                        f"Input shape: {tf_shape(data_input)}."
                    ),
                    # data=[actual_feat_dim_tensor,
                    #       tf_shape(data_input)],
                    summarize=3
                )
            processed_tensors.append(data_input)
        else:
            # Handle None inputs: if input_dim was specified, it's an error.
            if expected_feat_dim is not None and error == "raise":
                raise ValueError(
                    f"{name} input is None but its dimension "
                    f"({expected_feat_dim}) was specified as required "
                    "during model initialization."
                    )
            processed_tensors.append(None) # Keep None if optional
        if verbose >= 5: # Level 5 for detailed per-input result
            shape_str = processed_tensors[-1].shape if \
                processed_tensors[-1] is not None else "None"
            print(f"      {name} validated. Shape: {shape_str}")

    # Unpack processed tensors in the order [static, dynamic, future]
    static_p, dynamic_p, future_p = processed_tensors

    # --- 3. Batch Size Consistency Check ---
    # Collect all non-None tensors for batch size comparison.
    non_null_for_batch_check = [
        t for t in processed_tensors if t is not None
        ]

    if len(non_null_for_batch_check) > 1:
        if verbose >= 3:
            print("  Checking batch size consistency across inputs...")
        # Get batch size of the first non-None tensor as reference.
        ref_batch_size_tensor = _get_batch_size_for_validation(
            non_null_for_batch_check[0], verbose=verbose
            )

        for t_idx, t_current in enumerate(
            non_null_for_batch_check[1:], start=1
            ):
            current_batch_size_tensor = _get_batch_size_for_validation(
                t_current, verbose=verbose
                )
            # Graph-compatible assertion for batch size.
            tf_debugging.assert_equal(
                ref_batch_size_tensor, # This was from previous version, should be:
                              # ref_batch_size_tensor
                current_batch_size_tensor, # This was from previous version, should be:
                                  # current_batch_size_tensor
                message=(
                   "Inconsistent batch sizes among provided inputs. "
                   f"Reference batch size: {ref_batch_size_tensor}, "
                   f"Current batch size: {current_batch_size_tensor}."
                ),
                # data=[ref_batch_size_tensor, current_batch_size_tensor],
                summarize=10
            )
    
        if verbose >= 3:
            print("    Batch sizes are consistent.")

    # --- 4. Time Dimension Consistency (Dynamic vs. Future) ---
    if dynamic_p is not None and future_p is not None:
        if verbose >= 3:
            print("  Checking time dim consistency (dynamic vs future)...")

        # dynamic_p shape: (B, T_past, N_d)
        # future_p shape: (B, T_future_span, N_f)
        t_past_dyn = tf_shape(dynamic_p)[1]
        t_span_fut = tf_shape(future_p)[1]

        # Future input time span must be >= dynamic input's past time span.
        tf_debugging.assert_greater_equal(
            t_span_fut, t_past_dyn,
            message=(
                "Future input time span must be >= dynamic input time span. "
                f"Future time span: {t_span_fut}, Past dynamic"
                f" time span: {t_past_dyn}."
            ),
            #data=[t_span_fut, t_past_dyn], # For dynamic error message
            summarize=10
        )

        # Optional: Check if future_p spans enough for forecast_horizon.
        # This is a warning as models might use future inputs differently.
        if forecast_horizon is not None:
            fh_tensor = tf_cast(
                forecast_horizon, dtype=t_span_fut.dtype
                )
            # Required span for future if used in decoder: T_past + H
            required_future_span_for_decode = t_past_dyn + fh_tensor
            # Use tf.less for graph-compatible comparison
            if tf_less(t_span_fut, required_future_span_for_decode):
                # Python warning is fine here as it's informational.
                warnings.warn(
                    f"Future input time span ({t_span_fut}) is less "
                    f"than dynamic lookback ({t_past_dyn}) + "
                    f"forecast_horizon ({forecast_horizon}). This "
                    f"might be insufficient for models using future "
                    "inputs in a decoder stage.",
                    UserWarning
                )
        if verbose >= 3:
            print("    Time dimensions (dynamic vs future) are compatible.")

    if verbose >= 2:
        final_s_shape = static_p.shape if static_p is not None else 'None'
        final_d_shape = dynamic_p.shape if dynamic_p is not None else 'None'
        final_f_shape = future_p.shape if future_p is not None else 'None'
        print(f"Exit `validate_xtft_inputs`. Processed shapes: "
              f"S={final_s_shape}, D={final_d_shape}, F={final_f_shape}")

    # Return in the order: static, dynamic, future
    # This matches the original unpacking order from `inputs` list.
    return static_p, dynamic_p, future_p


def _get_batch_size(
    t: Union[np.ndarray, Tensor]
    ) -> Union[int, Tensor]:
    """
    Return the first-dimension batch size, static if available,
    otherwise dynamic.
    """
    # Check if static shape for batch dimension is known
    if hasattr(t, 'shape') and t.shape.rank is not None \
            and t.shape.dims is not None and len(t.shape.dims) > 0 \
            and t.shape.dims[0].value is not None:
        return t.shape.dims[0].value  # Static batch size
    # Fallback to dynamic shape retrieval using tf.shape
    return tf_shape(t)[0]


@optional_tf_function
def _validate_tft_inputs(
    inputs: List[Optional[Union[np.ndarray, Tensor]]],
    static_input_dim: Optional[int] = None,
    dynamic_input_dim: Optional[int] = None,
    future_covariate_dim: Optional[int] = None,
    forecast_horizon: Optional[int] = None,
    error: str = "raise",
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Validates and standardizes inputs for TFT-like models.
    (Full docstring omitted for brevity as requested)
    """
    # --- 1. Basic Input Structure Validation ---
    if not isinstance(inputs, (list, tuple)):
        # If a single tensor is passed, wrap it in a list.
        # This might occur if only dynamic_input is used.
        # However, the function expects a specific order if multiple
        # inputs are present.
        inputs = [inputs]
        # It's better to enforce the list structure from the caller.
        # For now, we'll issue a warning if it's not a list/tuple.
        msg = (
            "`inputs` should ideally be a list or tuple. "
            f"Received {type(inputs)}. Assuming it's dynamic_input."
        )
        warnings.warn(msg, UserWarning)

    # Expected order: [dynamic, future, static]
    # Handle cases where some inputs might be None
    if len(inputs) > 3:
        raise ValueError(
            "Too many inputs. Expected up to 3: "
            "[dynamic_inputs, future_inputs, static_inputs]."
        )

    # Pad with None if fewer than 3 inputs are provided to simplify unpacking
    # This assumes the order is [dynamic, future, static] if all present,
    # or [dynamic, future] if static is None, etc.
    # The caller (e.g., TFT.call) is responsible for correct ordering.
    dynamic_input_raw = inputs[0] if len(inputs) > 0 else None
    future_input_raw = inputs[1] if len(inputs) > 1 else None
    static_input_raw = inputs[2] if len(inputs) > 2 else None

    # --- 2. Type Conversion and Rank/Dimension Checks ---
    processed_tensors: List[Optional[Tensor]] = []
    input_names = ["Dynamic", "Future", "Static"]
    # Expected feature dimensions for each input type
    expected_feature_dims = [
        dynamic_input_dim, future_covariate_dim, static_input_dim
        ]
    # Expected number of dimensions (rank) for each input type
    expected_ndims = [3, 3, 2] # Dynamic=3D, Future=3D, Static=2D
    raw_inputs_ordered = [
        dynamic_input_raw, future_input_raw, static_input_raw
        ]

    for i, data_input in enumerate(raw_inputs_ordered):
        name = input_names[i]
        expected_feat_dim = expected_feature_dims[i]
        expected_rank = expected_ndims[i]

        if data_input is not None:
            # Convert to TensorFlow tensor and ensure float32
            if not isinstance(data_input, Tensor): # Check against tf.Tensor
                try:
                    data_input = tf_convert_to_tensor(
                        data_input, dtype=tf_float32
                        )
                except Exception as e:
                    raise TypeError(
                        f"Failed to convert {name} input to tensor:"
                        f" {e}"
                        ) from e
            elif data_input.dtype != tf_float32:
                data_input = tf_cast(data_input, tf_float32)

            # Check rank
            current_rank = tf_rank(data_input)
            # Compare ranks using TensorFlow operations
            rank_matches = tf_equal(
                current_rank,
                tf_cast(expected_rank, dtype=current_rank.dtype)
                )
            # Use tf.debugging.Assert for graph-mode error
            tf_debugging.assert_equal(
                rank_matches, tf_cast(True, dtype=rank_matches.dtype),
                message=f"{name} input must be {expected_rank}D. "
                        f"Got rank {current_rank} for shape "
                        f"{tf_shape(data_input)}."
            )

            # Check feature dimension (last dimension)
            if expected_feat_dim is not None:
                actual_feat_dim = tf_shape(data_input)[-1]
                dim_matches = tf_equal(
                    actual_feat_dim,
                    tf_cast(expected_feat_dim, dtype=actual_feat_dim.dtype)
                    )
                tf_debugging.assert_equal(
                    dim_matches, tf_cast(True, dtype=dim_matches.dtype),
                    message=f"{name} input last dimension mismatch. "
                            f"Expected {expected_feat_dim}, got "
                            f"{actual_feat_dim} for shape "
                            f"{tf_shape(data_input)}."
                )
            processed_tensors.append(data_input)
        else:
            # If input is None, check if corresponding dim was expected
            if expected_feat_dim is not None and error == "raise":
                raise ValueError(
                    f"{name} input is None but {name}_input_dim "
                    f"({expected_feat_dim}) was specified."
                    )
            processed_tensors.append(None)

    dynamic_p, future_p, static_p = processed_tensors

    # --- 3. Batch Size Consistency Check ---
    non_null_tensors = [t for t in processed_tensors if t is not None]
    if len(non_null_tensors) > 1:
        ref_batch_size = _get_batch_size(non_null_tensors[0])
        for t_idx, t_current in enumerate(non_null_tensors[1:], start=1):
            current_batch_size = _get_batch_size(t_current)
            # Ensure both are tensors for tf.equal
            ref_b_tensor = tf_convert_to_tensor(
                ref_batch_size, dtype=tf_int32
                )
            current_b_tensor = tf_convert_to_tensor(
                current_batch_size, dtype=tf_int32
                )
            # Graph-compatible assertion for batch size
            tf_debugging.assert_equal(
                ref_b_tensor, current_b_tensor,
                message=(
                    "Inconsistent batch sizes among provided inputs."
                    # More detailed message can be added if needed
                ),
                summarize=10 # Summarize tensor values in error message
            )

    # --- 4. Time Dimension Consistency (Dynamic vs. Future) ---
    if dynamic_p is not None and future_p is not None:
        # dynamic_p shape: (B, T_past, N_d)
        # future_p shape: (B, T_future_span, N_f)
        t_past_dyn = tf_shape(dynamic_p)[1]
        t_span_fut = tf_shape(future_p)[1]

        # Future input time span must be at least as long as
        # dynamic input's past time span.
        tf_debugging.assert_greater_equal(
            t_span_fut, t_past_dyn,
            message=(
                "Future input time span must be >= dynamic input "
                "time span. Got Dynamic T={dynamic_T}, "
                "Future T={future_T}."
            ),
            # Pass actual tensor values for dynamic message formatting
            # data=[t_past_dyn, t_span_fut],
            summarize=10
        )

        # Optional: Check if future_p spans enough for forecast_horizon
        # This is a warning as some models might only use future inputs
        # aligned with the encoder part (dynamic_p).
        if forecast_horizon is not None:
            # Convert forecast_horizon to tensor for comparison
            fh_tensor = tf_cast(forecast_horizon, dtype=t_span_fut.dtype)
            required_future_span = t_past_dyn + fh_tensor
            if tf_less(t_span_fut, required_future_span):
                # This warning will execute during graph tracing if
                # tf_less evaluates to True then.
                # For a runtime warning, tf.cond with tf.print is needed.
                warnings.warn(
                    f"Future input time span ({t_span_fut}) is less "
                    f"than dynamic lookback ({t_past_dyn}) + "
                    f"forecast_horizon ({forecast_horizon}). This "
                    f"might be insufficient for decoder stages.",
                    UserWarning
                )

    # Return in the order: dynamic, future, static
    return dynamic_p, future_p, static_p

@optional_tf_function
def validate_xtft_inputs_in(
    inputs: Union[List[Any], Tuple[Any, ...]],
    dynamic_input_dim: int,
    static_input_dim: int,
    future_covariate_dim: Optional[int] = None, 
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Validates and processes the ``inputs`` for the XTFT model.
    
    Parameters:
    - ``inputs`` (Union[List[Any], Tuple[Any, ...]]): 
        A list or tuple containing the inputs to the model in the following 
        order: [static_input, dynamic_input, future_covariate_input].
        
        - `static_input`: TensorFlow tensor or array-like object 
          representing static features.
        - `dynamic_input`: TensorFlow tensor or array-like object 
          representing dynamic features.
        - `future_covariate_input`: (Optional) TensorFlow tensor or 
          array-like object representing future covariates.
          Can be `None` if not used.
    - ``static_input_dim`` (int): 
        The expected dimensionality of the static input features 
        (i.e., number of static features).
    - ``dynamic_input_dim`` (int): 
        The expected dimensionality of the dynamic input features 
        (i.e., number of dynamic features).
    - ``future_covariate_dim`` (Optional[int], optional): 
        The expected dimensionality of the future covariate features 
        (i.e., number of future covariate features).
        If `None`, the function expects `future_covariate_input` to be 
        `None`.
    
    Returns:
    - ``static_input`` (`Tensor`): 
        Validated static input tensor of shape 
        `(batch_size, static_input_dim)` and dtype `float32`.
    - ``dynamic_input`` (`Tensor`): 
        Validated dynamic input tensor of shape 
        `(batch_size, time_steps, dynamic_input_dim)` and dtype `float32`.
    - ``future_covariate_input`` (`Tensor` or `None`): 
        Validated future covariate input tensor of shape 
        `(batch_size, time_steps, future_covariate_dim)` and dtype `float32`.
        Returns `None` if `future_covariate_dim` is `None` or if the input 
        was `None`.
    
    Raises:
    - ValueError: 
        If ``inputs`` is not a list or tuple with the required number of 
        elements.
        If ``future_covariate_dim`` is specified but 
        ``future_covariate_input`` is `None`.
        If the provided inputs do not match the expected dimensionalities.
        If the inputs contain incompatible batch sizes.
    
    Examples:
    ---------
    >>> # Example without future covariates
    >>> import tensorflow as tf
    >>> from fusionlab.nn._tensor_validation import validate_xtft_inputs 
    >>> static_input = tf.random.normal((32, 10))
    >>> dynamic_input = tf.random.normal((32, 20, 45))
    >>> inputs = [static_input, dynamic_input, None]
    >>> validated_static, validated_dynamic, validated_future = validate_xtft_inputs(
    ...     inputs,
    ...     static_input_dim=10,
    ...     dynamic_input_dim=45,
    ...     future_covariate_dim=None
    ... )
    >>> print(validated_static.shape, validated_dynamic.shape, validated_future)
    (32, 10) (32, 20, 45) None
    
    >>> # Example with future covariates
    >>> future_covariate_input = tf.random.normal((32, 20, 5))
    >>> inputs = [static_input, dynamic_input, future_covariate_input]
    >>> validated_static, validated_dynamic, validated_future = validate_xtft_inputs(
    ...     inputs,
    ...     static_input_dim=10,
    ...     dynamic_input_dim=45,
    ...     future_covariate_dim=5
    ... )
    >>> print(validated_static.shape, validated_dynamic.shape, validated_future.shape)
    (32, 10) (32, 20, 45) (32, 20, 5)
    """

    # Step 1: Validate the type and length of inputs
    if not isinstance(inputs, (list, tuple)):
        raise ValueError(
            f"'inputs' must be a list or tuple, but got type {type(inputs).__name__}."
        )
    
    expected_length = 3
    if len(inputs) != expected_length:
        raise ValueError(
            f"'inputs' must contain exactly {expected_length} elements: "
            f"[static_input, dynamic_input, future_covariate_input]. "
            f"Received {len(inputs)} elements."
        )
    
    # Unpack inputs
    static_input, dynamic_input, future_covariate_input = inputs

    # Step 2: Validate static_input
    if static_input is None:
        raise ValueError("``static_input`` cannot be None.")
    
    # Convert to tensor if not already
    if not isinstance(static_input, Tensor):
        try:
            static_input = tf_convert_to_tensor(
                static_input,
                dtype=tf_float32
            )
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to convert ``static_input`` to a TensorFlow tensor: {e}"
            )
    else:
        # Ensure dtype is float32
        static_input = tf_cast(static_input, tf_float32)
    
    # Check static_input dimensions
    if len(static_input.shape) != 2:
        raise ValueError(
            f"``static_input`` must be a 2D tensor with shape "
            f"(batch_size, static_input_dim), but got {len(static_input.shape)}D tensor."
        )
    
    # Check static_input_dim
    if static_input.shape[1] is not None and static_input.shape[1] != static_input_dim:
        raise ValueError(
            f"``static_input`` has incorrect feature dimension. Expected "
            f"{static_input_dim}, but got {static_input.shape[1]}."
        )
    elif static_input.shape[1] is None:
        # Dynamic dimension, cannot validate now
        pass

    # Step 3: Validate dynamic_input
    if dynamic_input is None:
        raise ValueError("``dynamic_input`` cannot be None.")
    
    # Convert to tensor if not already
    if not isinstance(dynamic_input, Tensor):
        try:
            dynamic_input = tf_convert_to_tensor(
                dynamic_input,
                dtype=tf_float32
            )
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to convert ``dynamic_input`` to a TensorFlow tensor: {e}"
            )
    else:
        # Ensure dtype is float32
        dynamic_input = tf_cast(dynamic_input, tf_float32)
    
    # Check dynamic_input dimensions
    if len(dynamic_input.shape) != 3:
        raise ValueError(
            f"``dynamic_input`` must be a 3D tensor with shape "
            f"(batch_size, time_steps, dynamic_input_dim), but got "
            f"{len(dynamic_input.shape)}D tensor."
        )
    
    # Check dynamic_input_dim
    if dynamic_input.shape[2] is not None and dynamic_input.shape[2] != dynamic_input_dim:
        raise ValueError(
            f"``dynamic_input`` has incorrect feature dimension. Expected "
            f"{dynamic_input_dim}, but got {dynamic_input.shape[2]}."
        )
    elif dynamic_input.shape[2] is None:
        # Dynamic dimension, cannot validate now
        pass

    # Step 4: Validate future_covariate_input
    if future_covariate_dim is not None:
        if future_covariate_input is None:
            raise ValueError(
                "``future_covariate_dim`` is specified, but "
                "``future_covariate_input`` is None."
            )
        
        # Convert to tensor if not already
        if not isinstance(future_covariate_input, Tensor):
            try:
                future_covariate_input = tf_convert_to_tensor(
                    future_covariate_input,
                    dtype=tf_float32
                )
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Failed to convert ``future_covariate_input`` to a TensorFlow tensor: {e}"
                )
        else:
            # Ensure dtype is float32
            future_covariate_input = tf_cast(future_covariate_input, tf_float32)
        
        # Check future_covariate_input dimensions
        if len(future_covariate_input.shape) != 3:
            raise ValueError(
                f"``future_covariate_input`` must be a 3D tensor with shape "
                f"(batch_size, time_steps, future_covariate_dim), but got "
                f"{len(future_covariate_input.shape)}D tensor."
            )
        
        # Check future_covariate_dim
        if (future_covariate_input.shape[2] is not None and 
            future_covariate_input.shape[2] != future_covariate_dim):
            raise ValueError(
                f"``future_covariate_input`` has incorrect feature dimension. "
                f"Expected {future_covariate_dim}, but got "
                f"{future_covariate_input.shape[2]}."
            )
        elif future_covariate_input.shape[2] is None:
            # Dynamic dimension, cannot validate now
            pass
    else:
        if future_covariate_input is not None:
            raise ValueError(
                "``future_covariate_dim`` is None, but "
                "``future_covariate_input`` is provided."
            )
    
    # Step 5: Validate batch sizes across inputs
    static_batch_size = tf_shape(static_input)[0]
    dynamic_batch_size = tf_shape(dynamic_input)[0]
    
    with suppress_tf_warnings():
        if future_covariate_dim is not None:
            future_batch_size = tf_shape(future_covariate_input)[0]
            # Check if all batch sizes are equal
            batch_size_cond = tf_reduce_all([
                tf_equal(static_batch_size, dynamic_batch_size),
                tf_equal(static_batch_size, future_batch_size)
            ])
        else:
            # Check only static and dynamic batch sizes
            batch_size_cond = tf_equal(static_batch_size, dynamic_batch_size)
        
        # Ensure batch sizes match
        tf_debugging.assert_equal(
            batch_size_cond, True,
            message=(
                "Batch sizes do not match across inputs: "
                f"``static_input`` batch_size={static_batch_size}, "
                f"``dynamic_input`` batch_size={dynamic_batch_size}" +
                (f", ``future_covariate_input`` batch_size={future_batch_size}" 
                 if future_covariate_dim is not None else "")
            )
        )

    return static_input, dynamic_input, future_covariate_input

@optional_tf_function
def validate_batch_sizes(
    static_batch_size: Tensor,
    dynamic_batch_size: Tensor,
    future_batch_size: Optional[Tensor] = None
) -> None:
    """
    Validates that the batch sizes of static, dynamic, and future 
    covariate inputs match.
    
    Parameters:
    - ``static_batch_size`` (`Tensor`): 
        Batch size of the static input.
    - ``dynamic_batch_size`` (`Tensor`): 
        Batch size of the dynamic input.
    - ``future_batch_size`` (`Optional[Tensor]`, optional): 
        Batch size of the future covariate input.
        Defaults to `None`.
    
    Raises:
    - tf_errors.InvalidArgumentError: 
        If the batch sizes do not match.
    """
    tf_debugging.assert_equal(
        static_batch_size, dynamic_batch_size,
        message=(
            "Batch sizes do not match across inputs: "
            f"``static_input`` batch_size={static_batch_size.numpy()}, "
            f"``dynamic_input`` batch_size={dynamic_batch_size.numpy()}" +
            (f", ``future_covariate_input`` batch_size={future_batch_size.numpy()}" 
             if future_batch_size is not None else "")
        )
    )
    if future_batch_size is not None:
        tf_debugging.assert_equal(
            static_batch_size, future_batch_size,
            message=(
                "Batch sizes do not match between static and future covariate inputs: "
                f"``static_input`` batch_size={static_batch_size.numpy()}, "
                f"``future_covariate_input`` batch_size={future_batch_size.numpy()}."
            )
        )

@optional_tf_function
def check_batch_sizes(
    static_batch_size: Tensor,
    dynamic_batch_size: Tensor,
    future_batch_size: Optional[Tensor] = None
) -> None:
    """
    Checks that the batch sizes of static, dynamic, and future covariate 
    inputs are equal.
    
    Parameters:
    - ``static_batch_size`` (`Tensor`): 
        Batch size of the static input.
    - ``dynamic_batch_size`` (`Tensor`): 
        Batch size of the dynamic input.
    - ``future_batch_size`` (`Optional[Tensor]`, optional): 
        Batch size of the future covariate input.
        Defaults to `None`.
    
    Raises:
    - tf_errors.InvalidArgumentError: 
        If the batch sizes do not match.
    """
    tf_assert_equal(
        static_batch_size, dynamic_batch_size,
        message=(
            "Batch sizes do not match across inputs: "
            f"``static_input`` batch_size={static_batch_size.numpy()}, "
            f"``dynamic_input`` batch_size={dynamic_batch_size.numpy()}" +
            (f", ``future_covariate_input`` batch_size={future_batch_size.numpy()}" 
             if future_batch_size is not None else "")
        )
    )
    if future_batch_size is not None:
        tf_assert_equal(
            static_batch_size, future_batch_size,
            message=(
                "Batch sizes do not match between static and future covariate inputs: "
                f"``static_input`` batch_size={static_batch_size.numpy()}, "
                f"``future_covariate_input`` batch_size={future_batch_size.numpy()}."
            )
        )


def validate_batch_sizes_eager(
    static_batch_size: int,
    dynamic_batch_size: int,
    future_batch_size: Optional[int] = None
) -> None:
    """
    Validates that the batch sizes of static, dynamic, and future covariate 
    inputs match in eager execution mode.
    
    Parameters:
    - ``static_batch_size`` (int): 
        Batch size of the static input.
    - ``dynamic_batch_size`` (int): 
        Batch size of the dynamic input.
    - ``future_batch_size`` (`Optional[int]`, optional): 
        Batch size of the future covariate input.
        Defaults to `None`.
    
    Raises:
    - AssertionError: 
        If the batch sizes do not match.
    """
    assert static_batch_size == dynamic_batch_size, (
        "Batch sizes do not match across inputs: "
        f"``static_input`` batch_size={static_batch_size}, "
        f"``dynamic_input`` batch_size={dynamic_batch_size}" +
        (f", ``future_covariate_input`` batch_size={future_batch_size}" 
         if future_batch_size is not None else "")
    )
    if future_batch_size is not None:
        assert static_batch_size == future_batch_size, (
            "Batch sizes do not match between static and future covariate inputs: "
            f"``static_input`` batch_size={static_batch_size}, "
            f"``future_covariate_input`` batch_size={future_batch_size}."
        )


def align_temporal_dimensions(
    tensor_ref: Tensor,
    tensor_to_align: Tensor,
    ref_time_dim_index: int = 1,
    align_time_dim_index: int = 1,
    mode: str = 'slice_to_ref',
    # allow_broadcast_shorter_ref: bool = False, # Removed for clarity
    name: str = "tensor_to_align",
    padding_value: int = 0
) -> Tuple[Tensor, Tensor]:
    r"""Aligns the time dimension of `tensor_to_align` to `tensor_ref`.

    This function is typically used to ensure that two temporal tensors
    (e.g., dynamic past features and known future features) have a
    compatible time dimension before operations like concatenation or
    element-wise addition, especially for input to encoder stages.

    Parameters
    ----------
    tensor_ref : Tensor or np.ndarray
        The reference tensor, typically 3D (Batch, Time_Ref, Features_Ref).
        The time dimension of this tensor is used as the target length.
    tensor_to_align : Tensor or np.ndarray
        The tensor whose time dimension needs to be aligned, typically 3D
        (Batch, Time_Align, Features_Align).
    ref_time_dim_index : int, default=1
        The index of the time dimension in `tensor_ref`.
    align_time_dim_index : int, default=1
        The index of the time dimension in `tensor_to_align`.
    mode : {'slice_to_ref', 'pad_to_ref', 'truncate_ref_if_shorter'}, default='slice_to_ref'
        Strategy for alignment:
        - ``'slice_to_ref'``: Slices `tensor_to_align` if it is longer
          than `tensor_ref` along the time dimension. Raises an error
          if `tensor_to_align` is shorter. `tensor_ref` is unchanged.
        - ``'pad_to_ref'``: If `tensor_to_align` is shorter than
          `tensor_ref` in time, it's padded with `padding_value` (default 0)
          at the end of its time dimension. If longer, it's sliced.
          `tensor_ref` is unchanged.
        - ``'truncate_ref_if_shorter'``: If `tensor_to_align`'s time
          dimension is shorter than `tensor_ref`'s, then `tensor_ref`
          itself is truncated to match `tensor_to_align`'s time length.
          `tensor_to_align` is returned as is. If `tensor_to_align` is
          not shorter, both are returned as is.
    name : str, default="tensor_to_align"
        Name for the tensor being aligned, used in error messages.
    padding_value : int, default=0
        Value to use for padding if `mode='pad_to_ref'` and
        `tensor_to_align` is shorter.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple containing:
        - ``output_tensor_ref``: The reference tensor, potentially modified
          if `mode='truncate_ref_if_shorter'`.
        - ``output_tensor_to_align``: The tensor_to_align, potentially
          modified by slicing or padding.

    Raises
    ------
    ValueError
        If input tensors have unsupported ranks, invalid time dimension
        indices, or if time dimensions are incompatible based on the
        chosen mode (e.g., for 'slice_to_ref' if align is shorter).
    NotImplementedError
        If an unsupported mode is requested.
    """
    # Validate ranks
    rank_ref = len(tensor_ref.shape)
    rank_align = len(tensor_to_align.shape)

    if rank_ref < 2 or rank_align < 2:
        raise ValueError(
            "Both tensor_ref and tensor_to_align must be at least 2D. "
            f"Got ranks: ref={rank_ref}, align={rank_align}"
        )
    # Ensure time dimension indices are valid for the given ranks
    if rank_ref > 1 and not (0 <= ref_time_dim_index < rank_ref):
        raise ValueError(f"Invalid ref_time_dim_index {ref_time_dim_index} "
                         f"for tensor_ref with rank {rank_ref}")
    if rank_align > 1 and not (0 <= align_time_dim_index < rank_align):
        raise ValueError(f"Invalid align_time_dim_index {align_time_dim_index} "
                         f"for tensor_to_align with rank {rank_align}")

    # Get dynamic shapes using tf_shape for graph compatibility
    shape_ref = tf_shape(tensor_ref)
    shape_align = tf_shape(tensor_to_align)

    # Determine target time steps from tensor_ref's specified time dimension
    # If rank is 1 (e.g. a vector considered as having 1 time step), handle gracefully.
    target_time_steps = shape_ref[ref_time_dim_index] if rank_ref > 1 else 1
    current_align_time_steps = shape_align[align_time_dim_index] if rank_align > 1 else 1

    # Initialize outputs
    output_tensor_ref = tensor_ref
    output_tensor_to_align = tensor_to_align

    if mode == 'slice_to_ref':
        tf_debugging.assert_greater_equal(
            current_align_time_steps, target_time_steps,
            message=(
                f"{name} time steps ({current_align_time_steps}) must be >= "
                f"reference tensor time steps ({target_time_steps}) "
                f"for mode 'slice_to_ref'."
            )
        )
        slicers = [slice(None)] * rank_align
        slicers[align_time_dim_index] = slice(None, target_time_steps)
        output_tensor_to_align = tensor_to_align[tuple(slicers)]

    elif mode == 'pad_to_ref':
        if current_align_time_steps < target_time_steps:
            # Calculate padding needed for the time dimension
            padding_needed = target_time_steps - current_align_time_steps
            # Construct paddings argument for tf_pad
            # It's a list of pairs for each dimension: [[before, after], ...]
            paddings = [[0, 0]] * rank_align
            paddings[align_time_dim_index] = [0, padding_needed]
            output_tensor_to_align = tf_pad(
                tensor_to_align, paddings, mode="CONSTANT",
                constant_values=padding_value
            )
        elif current_align_time_steps > target_time_steps:
            # Slice if tensor_to_align is longer
            slicers = [slice(None)] * rank_align
            slicers[align_time_dim_index] = slice(None, target_time_steps)
            output_tensor_to_align = tensor_to_align[tuple(slicers)]
        # If equal, no change needed for output_tensor_to_align

    elif mode == 'truncate_ref_if_shorter':
        # "truncate tensor_ref if tensor_to_align is shorter"
        if current_align_time_steps < target_time_steps:
            # Truncate tensor_ref to match tensor_to_align's time length
            slicers_ref = [slice(None)] * rank_ref
            slicers_ref[ref_time_dim_index] = slice(None, current_align_time_steps)
            output_tensor_ref = tensor_ref[tuple(slicers_ref)]
            # output_tensor_to_align remains as is
        # If tensor_to_align is not shorter, both are returned as is in this mode.

    else:
        raise ValueError(f"Unsupported alignment mode: '{mode}'. "
                         "Choose 'slice_to_ref', 'pad_to_ref', or "
                         "'truncate_ref_if_shorter'.")

    return output_tensor_ref, output_tensor_to_align

def validate_minimal_inputs(
    X_static: Union[np.ndarray, Tensor],
    X_dynamic: Union[np.ndarray, Tensor],
    X_future: Union[np.ndarray, Tensor],
    y: Optional[Union[np.ndarray, Tensor]] = None,
    forecast_horizon: Optional[int] = None, # Model's output horizon
    deep_check: bool = True
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
    r"""
    Validate minimal inputs for forecasting models like TFT/XTFT.

    This function verifies that the provided input arrays
    (``X_static``, ``X_dynamic``, ``X_future`` and, optionally, ``y``)
    have the expected dimensionality and consistent shapes. It
    converts inputs to ``float32`` and ensures shapes meet requirements:

    .. math::
        X_{\text{static}} \in \mathbb{R}^{B \times N_s} \\
        X_{\text{dynamic}} \in \mathbb{R}^{B \times T_{past} \times N_d} \\
        X_{\text{future}} \in \mathbb{R}^{B \times T_{future\_span} \times N_f}

    and, if provided,

    .. math::
        y \in \mathbb{R}^{B \times H \times O}

    where :math:`B` is batch size, :math:`T_{past}` is the lookback
    period for dynamic inputs, :math:`T_{future\_span}` is the total
    time steps for future inputs (must be :math:`\ge T_{past}`),
    :math:`H` is the model's output forecast horizon, and
    :math:`N_s, N_d, N_f, O` are feature/output dimensions.

    Parameters
    ----------
    X_static : np.ndarray or Tensor
        Static features, shape (Batch, NumStaticFeatures).
    X_dynamic : np.ndarray or Tensor
        Dynamic past features, shape (Batch, PastTimeSteps, NumDynamicFeatures).
    X_future : np.ndarray or Tensor
        Known future features, shape (Batch, FutureTimeSpan, NumFutureFeatures).
        ``FutureTimeSpan`` must be >= ``PastTimeSteps``.
    y : np.ndarray or Tensor, optional
        Target values, shape (Batch, ForecastHorizon, OutputDim).
    forecast_horizon : int, optional
        The expected output forecast horizon of the model. Used to
        validate ``y.shape[1]`` if ``y`` is provided.
    deep_check : bool, default=True
        If True, perform full consistency checks on batch sizes and
        relevant time dimensions.

    Returns
    -------
    tuple
        Validated input tensors (X_static, X_dynamic, X_future) or
        (X_static, X_dynamic, X_future, y), converted to float32.

    Raises
    ------
    ValueError
        If inputs have unexpected dimensions or inconsistent shapes.
    TypeError
        If inputs are not np.ndarray or Tensor.
    """
    def _check_tensor_shape_(
        arr: Union[np.ndarray, Tensor],
        expected_ndim_int: int, # Expected rank as Python int
        name: str
    ):
        """Helper to check tensor dimensionality using tf_rank."""
        # origin_dim = arr.ndim 
        if not isinstance(arr, (np.ndarray, Tensor)): # Check against Tensor
            raise TypeError(
                f"{name} must be a NumPy array or TensorFlow Tensor. "
                f"Got {type(arr)}."
            )
        # Use tf_rank for TensorFlow tensors, arr.ndim for NumPy arrays
        current_rank = tf_rank(arr) if isinstance(arr, Tensor) else arr.ndim 

        # Compare ranks (tf_rank returns a 0-D Tensor)
        if not tf_equal(current_rank, tf_constant(
                expected_ndim_int, dtype=current_rank.dtype)):
            # Construct error message parts
            expected_descriptions = {
                "static": ("Expected shape is (B, Ns) [2D]", 2),
                "dynamic": ("Expected shape is (B, T_past, Nd) [3D]", 3),
                "future": ("Expected shape is (B, T_future_span, Nf) [3D]", 3),
                "target": ("Expected shape is (B, H, O) [3D]", 3)
            }
            desc_key = None
            for key_prefix in expected_descriptions.keys():
                if key_prefix in name.lower():
                    desc_key = key_prefix
                    break
            
            expected_msg = f"Expected {expected_ndim_int}D."
            if desc_key:
                expected_msg = expected_descriptions[desc_key][0]

            # Try to get concrete rank for error message if possible
            try:
                # This might work in eager, fail in graph if rank is symbolic
                current_rank_val = current_rank.numpy() if hasattr(
                    current_rank, 'numpy') else current_rank
            except: # pylint: disable=bare-except
                current_rank_val = "<unknown_in_graph>"

            raise ValueError(
                f"{name} must be {expected_ndim_int}D.\n"
                f"{expected_msg}\n"
                f"Got array with rank {current_rank_val} and shape {arr.shape}."
            )
        return arr
    
    def _check_tensor_shape(
        arr: Union[np.ndarray, Tensor],
        expected_ndim: int,
        name: str
    ):
        """Helper to check tensor dimensionality."""
        origin_dim = arr.ndim 
        if not isinstance(arr, (np.ndarray, Tensor)):
            raise TypeError(
                f"{name} must be a NumPy array or TensorFlow Tensor. "
                f"Got {type(arr)}."
            )
        if origin_dim != expected_ndim:
            raise ValueError(
                f"{name} must be {expected_ndim}D. Got {origin_dim}D "
                f"with shape {arr.shape}."
            )
        return arr

    def _ensure_float32(data: Union[np.ndarray, Tensor], name: str):
        """Ensure data is float32."""
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        elif hasattr(data, "dtype"): # Check for Tensor-like objects
            # For TensorFlow tensors, use tf_cast
            if KERAS_BACKEND and data.dtype != tf_float32:
                return tf_cast(data, tf_float32)
            # For other tensor types or if already float32, return as is
            return data
        else:
            raise TypeError(
                f"Unsupported data type for {name}: {type(data)}. "
                "Must be np.ndarray or Tensor."
            )

    # Ensure inputs are tensors and float32
    X_static = _ensure_float32(X_static, "X_static")
    X_dynamic = _ensure_float32(X_dynamic, "X_dynamic")
    X_future = _ensure_float32(X_future, "X_future")

    # Initial dimension checks
    X_static = _check_tensor_shape(X_static, 2, "X_static")
    X_dynamic = _check_tensor_shape(X_dynamic, 3, "X_dynamic")
    X_future = _check_tensor_shape(X_future, 3, "X_future")

    if y is not None:
        y = _ensure_float32(y, "y")
        y = _check_tensor_shape(y, 3, "y (target)")

    if not deep_check:
        return (X_static, X_dynamic, X_future) if y is None \
            else (X_static, X_dynamic, X_future, y)

    # --- Deeper Consistency Checks ---
    # Get shapes (Batch, NumStaticFeatures)
    B_sta, Ns = X_static.shape
    # (Batch, PastTimeSteps, NumDynamicFeatures)
    B_dyn, T_past_dyn, Nd = X_dynamic.shape
    # (Batch, FutureTimeSpan, NumFutureFeatures)
    B_fut, T_span_fut, Nf = X_future.shape

    # 1. Validate batch sizes match
    if not (B_sta == B_dyn == B_fut):
        raise ValueError(
            f"Batch sizes do not match: X_static ({B_sta}), "
            f"X_dynamic ({B_dyn}), X_future ({B_fut})."
        )

    # 2. Validate future time span covers at least dynamic past time span
    if T_span_fut < T_past_dyn:
        raise ValueError(
            f"Future input time span ({T_span_fut}) must be at least "
            f"as long as dynamic input time span ({T_past_dyn})."
        )

    # 3. Validate y if provided
    if y is not None:
        B_y, H_y, O_y = y.shape # H_y is horizon from y data

        if B_y != B_sta: # Check against a common batch size
            raise ValueError(
                f"Batch size of y ({B_y}) does not match "
                f"input data batch size ({B_sta})."
            )

        # If forecast_horizon parameter is given, it should match y's horizon
        if forecast_horizon is not None and forecast_horizon != H_y:
            warnings.warn(
                f"Provided 'forecast_horizon' parameter ({forecast_horizon}) "
                f"differs from y.shape[1] ({H_y}). "
                f"Using horizon from y data ({H_y}) for validation.",
                UserWarning
            )
            effective_horizon = H_y
        elif forecast_horizon is None and y is not None:
            effective_horizon = H_y # Infer from y
        elif forecast_horizon is not None: # and matches y.shape[1]
            effective_horizon = forecast_horizon
        else: # y is None, forecast_horizon might be None or int
            effective_horizon = forecast_horizon # Can be None

        # If model output horizon (effective_horizon) is known,
        # check if future input span is sufficient.
        # Future inputs should cover T_past_dyn (for encoder) + effective_horizon (for decoder).
        if effective_horizon is not None and T_span_fut < (T_past_dyn + effective_horizon):
            warnings.warn(
                f"Future input time span ({T_span_fut}) is less than "
                f"dynamic lookback ({T_past_dyn}) + output horizon ({effective_horizon}). "
                f"This might be insufficient for some model architectures "
                f"that use future inputs during decoding.",
                UserWarning
            )
        return X_static, X_dynamic, X_future, y

    return X_static, X_dynamic, X_future

def validate_minimal_inputs_in(
    X_static, X_dynamic, 
    X_future, y=None, 
    forecast_horizon=None, 
    deep_check=True
):
    r"""
    Validate minimal inputs for forecasting models.
    
    This function verifies that the provided input arrays 
    (``X_static``, ``X_dynamic``, ``X_future`` and, optionally, ``y``)
    have the expected dimensionality and consistent shapes for use in
    forecasting models. It converts the inputs to ``float32`` for 
    numerical stability and ensures that the shapes match the following 
    requirements:
    
    .. math::
       X_{\text{static}} \in \mathbb{R}^{B \times N_s}, \quad
       X_{\text{dynamic}} \in \mathbb{R}^{B \times F \times N_d}, \quad
       X_{\text{future}} \in \mathbb{R}^{B \times F \times N_f}
    
    and, if provided,
    
    .. math::
       y \in \mathbb{R}^{B \times F \times O},
    
    where :math:`B` is the batch size, :math:`F` is the forecast horizon, 
    :math:`N_s` is the number of static features, :math:`N_d` is the number 
    of dynamic features, :math:`N_f` is the number of future features, and 
    :math:`O` is the output dimension.
    
    The function uses an internal helper, :func:`check_shape`, to validate 
    that each input has the expected number of dimensions. For example:
    
    - ``X_static`` should be 2D with shape (``B``, ``N_s``)
    - ``X_dynamic`` and ``X_future`` should be 3D with shape 
      (``B``, ``F``, ``N_d``) or (``B``, ``F``, ``N_f``) respectively.
    - If provided, ``y`` should be 3D with shape (``B``, ``F``, ``O``).
    
    In addition, the function verifies that:
    
      - The batch sizes (``B``) are identical across all inputs.
      - The forecast horizon (``F``) is consistent between dynamic and 
        future inputs.
      - If a specific ``forecast_horizon`` is provided and it differs 
        from the input, a warning is issued and the forecast horizon from 
        the data is used.
    
    Parameters
    ----------
    X_static       : np.ndarray or Tensor
        The static feature input, expected to have shape (``B``, ``N_s``).
    X_dynamic      : np.ndarray or Tensor
        The dynamic feature input, expected to have shape (``B``, ``F``, 
        ``N_d``).
    X_future       : np.ndarray or Tensor
        The future feature input, expected to have shape (``B``, ``F``, 
        ``N_f``).
    y              : np.ndarray or Tensor, optional
        The target output, expected to have shape (``B``, ``F``, ``O``).
    forecast_horizon: int, optional
        The expected forecast horizon (``F``). If provided and it differs 
        from the input data, a warning is issued and the input forecast 
        horizon is used.
    deep_check     : bool, optional
        If True, perform full consistency checks on batch sizes and forecast 
        horizons. Default is True.
    
    Returns
    -------
    tuple
        If ``y`` is provided, returns a tuple:
        
        ``(X_static, X_dynamic, X_future, y)``
        
        Otherwise, returns:
        
        ``(X_static, X_dynamic, X_future)``
    
    Raises
    ------
    ValueError
        If any input does not have the expected dimensions, or if the batch 
        sizes or forecast horizons are inconsistent.
    TypeError
        If an input is not an instance of np.ndarray or Tensor.
    
    Examples
    --------
    >>> from fusionlab.nn._tensor_validation import validate_minimal_inputs
    >>> import numpy as np
    >>> X_static0  = np.random.rand(100, 5)
    >>> X_dynamic0 = np.random.rand(100, 10, 3)
    >>> X_future0  = np.random.rand(100, 10, 2)
    >>> y0         = np.random.rand(100, 10, 1)
    >>> validated_1 = validate_minimal_inputs(X_static0, X_dynamic0, 
    ...                                      X_future0, forecast_horizon=10)
    >>> X_static_v , X_dynamic_v, X_future_v = validated_1
    >>> X_static_v.shape , X_dynamic_v.shape, X_future_v.shape 
    ((100, 5), (100, 10, 3), (100, 10, 2))
    >>> 
    >>> validated_2 = validate_minimal_inputs(X_static0, X_dynamic0, 
    ...                                      X_future0, y0,  forecast_horizon=10)
    >>> X_static_v2 , X_dynamic_v2, X_future_v2, y_v2 = validated_2
    >>> X_static_v2.shape , X_dynamic_v2.shape, X_future_v2.shape, y_v2.shape 
    ((100, 5), (100, 10, 3), (100, 10, 2), (100, 10, 1))

    Notes
    -----
    This function is essential to ensure that the inputs for forecasting 
    models are correctly shaped. The helper function :func:`check_shape` is 
    used internally to provide detailed error messages based on the expected 
    shapes for different types of data:
    
    - For static data: (``B``, ``N_s``)
    - For dynamic data: (``B``, ``F``, ``N_d``)
    - For future data: (``B``, ``F``, ``N_f``)
    - For target data: (``B``, ``F``, ``O``)
    
    See Also
    --------
    np.ndarray.astype, tf_cast
        For data type conversion methods.
    
    References
    ----------
    .. [1] McKinney, W. (2010). "Data Structures for Statistical Computing 
           in Python". Proceedings of the 9th Python in Science Conference.
    .. [2] Van der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). "The 
           NumPy Array: A Structure for Efficient Numerical Computation". 
           Computing in Science & Engineering, 13(2), 22-30.
    """

    def check_shape(
        arr, 
        expect_dim: str = "2d", 
        name: str = "Static data 'X_static'"
    ):
        # Get the number of dimensions of the input array.
        origin_dim = arr.ndim
    
        # Define expected shape descriptions for different types.
        expected_descriptions = {
            "static":  ("Expected shape is (B, Ns):\n"
                        "  - B: Batch size\n"
                        "  - Ns: Number of static features."),
            "dynamic": ("Expected shape is (B, F, Nd):\n"
                        "  - B: Batch size\n"
                        "  - F: Forecast horizon\n"
                        "  - Nd: Number of dynamic features."),
            "future":  ("Expected shape is (B, F, Nf):\n"
                        "  - B: Batch size\n"
                        "  - F: Forecast horizon\n"
                        "  - Nf: Number of future features."),
            "target":  ("Expected shape is (B, F, O):\n"
                        "  - B: Batch size\n"
                        "  - F: Forecast horizon\n"
                        "  - O: Output dimension for target.")
        }
    
        # Determine which expected description to use based on `name`.
        keyword = None
        for key in expected_descriptions.keys():
            if key in name.lower():
                keyword = key
                break
    
        if keyword is not None:
            expected_msg = expected_descriptions[keyword]
        else:
            expected_msg = f"Expected {expect_dim} dimensions."
    
        # Check if the input array has the expected dimensions.
        if (expect_dim == "2d" and origin_dim != 2) or \
           (expect_dim == "3d" and origin_dim != 3):
            raise ValueError(
                f"{name} must have {expect_dim}.\n"
                f"{expected_msg}\n"
                f"Got array with {origin_dim} dimensions."
            )
    
        return arr

    # Convert inputs to float32 for numerical stability.
    def ensure_float32(data):
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        elif hasattr(data, "dtype") and data.dtype.kind in "fiu":
            return tf_cast(data, tf_float32)
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                "Must be np.ndarray or Tensor."
            )

    X_static  = ensure_float32(X_static)
    X_dynamic = ensure_float32(X_dynamic)
    X_future  = ensure_float32(X_future)
    
    X_static = check_shape(
        X_static, '2d', 
    )
    X_dynamic = check_shape(
        X_dynamic, '3d', 
        name ="Dynamic data 'X_dynamic'"
    )
    X_future =check_shape(
        X_future, '3d',
        name="Future data 'X_future'"
    )
    
    if y is not None:
        y = ensure_float32(y)
        X_future =check_shape(
            X_future, '3d', 
            name="Target data 'y'"
        )
        
    if not deep_check: 
        return (X_static, X_dynamic, X_future ) if y is None else ( 
            X_static, X_dynamic, X_future, y 
    )
   # Now if deep check is True , going deeper as below 
   # and control hroizon
   
    # Ensure correct dimensions:
    #   X_static must be 2D, X_dynamic and X_future must be 3D.
    B_sta, Ns    = X_static.shape
    B_dyn, F_dyn, Nd = X_dynamic.shape
    B_fut, F_fut, Nf = X_future.shape

    # Validate that batch sizes match.
    if not (B_sta == B_dyn == B_fut):
        raise ValueError(
            f"Batch sizes do not match: X_static ({B_sta}), "
            f"X_dynamic ({B_dyn}), X_future ({B_fut}). "
            "Ensure data is correctly shaped using "
            "`fusionlab.nn.utils.reshape_xft_data`."
        )

    # Validate forecast horizon consistency.
    if F_dyn != F_fut:
        raise ValueError(
            f"Forecast horizons do not match: X_dynamic ({F_dyn}), "
            f"X_future ({F_fut}). Ensure data is correctly shaped."
        )

    # If a forecast_horizon is provided, warn if it differs from input.
    if forecast_horizon is not None and forecast_horizon != F_dyn:
        
        warnings.warn(
            f"Provided forecast_horizon={forecast_horizon} differs from "
            f"input forecast horizon F_dyn={F_dyn}. Using F_dyn from input.",
            UserWarning
        )

    # Validate y if provided: y must be 3D and match batch and horizon.
    if y is not None:
        B_y, F_y, O = y.shape
        if B_y != B_sta:
            raise ValueError(
                f"Batch size of y ({B_y}) does not match X_static ({B_sta}). "
                "Ensure data is correctly shaped."
            )
        if F_y != F_dyn:
            raise ValueError(
                f"Forecast horizon of y ({F_y}) does not match "
                f"X_dynamic/X_future ({F_dyn}). Ensure data is correctly shaped."
            )
        return X_static, X_dynamic, X_future, y

    return X_static, X_dynamic, X_future
 

def combine_temporal_inputs_for_lstm(
    dynamic_selected: "Tensor",
    future_selected: "Tensor",
    mode: str = 'strict' 
    ) -> "Tensor":
    """Combines selected dynamic (past) and future features for LSTM input.

    Handles potential shape mismatches based on the selected mode.

    Args:
        dynamic_selected: Tensor containing processed dynamic features,
            ideally shape (Batch, T_past, HiddenUnits).
        future_selected: Tensor containing processed known future features,
            ideally shape (Batch, T_future_total, HiddenUnits) where
            T_future_total >= T_past.
        mode (str): Handling mode for shape/dimension issues.
            - 'strict' (default): Enforces 3D inputs and that
              T_future_total >= T_past, raising errors otherwise.
            - 'soft': Attempts to handle 2D inputs by adding a time
              dimension (with warning). Still requires T_future_total >= T_past.

    Returns:
        Tensor: Combined features ready for LSTM input, with shape
                (Batch, T_past, CombinedFeatures). Feature dimension
                depends on concatenation axis used.

    Raises:
        ValueError: If inputs have unsupported ranks or incompatible
                    time dimensions in 'strict' mode or if basic
                    shape requirements aren't met in 'soft' mode.
    """
    # --- Validate Mode ---
    if mode not in ['strict', 'soft']:
        raise ValueError(f"Invalid mode: '{mode}'. Choose 'strict' or 'soft'.")

    # --- Input Tensor Validation and Processing ---
    processed_dynamic = dynamic_selected
    processed_future = future_selected

    # Check ranks and potentially reshape in 'soft' mode
    dynamic_rank = len(processed_dynamic.shape)
    future_rank = len(processed_future.shape)

    if mode == 'soft':
        # Attempt to add time dimension if inputs are 2D
        if dynamic_rank == 2:
            warnings.warn(
                "Soft mode: Received 2D dynamic_selected input."
                " Assuming TimeSteps=1 and adding dimension.", UserWarning
                )
            processed_dynamic = tf_expand_dims(processed_dynamic, axis=1)
            dynamic_rank = 3
        if future_rank == 2:
            warnings.warn(
                "Soft mode: Received 2D future_selected input."
                " Assuming TimeSteps=1 and adding dimension.", UserWarning
                )
            processed_future = tf_expand_dims(processed_future, axis=1)
            future_rank = 3

    # --- Strict Shape Checks (Applied in both modes after potential reshape) ---
    if dynamic_rank != 3 or future_rank != 3:
        raise ValueError(
            f"Inputs must be 3D (Batch, Time, Features) after "
            f"processing. Got shapes: dynamic={processed_dynamic.shape}, "
            f"future={processed_future.shape}"
        )

    # Get dynamic time steps
    # Use tf_shape for compatibility with graph mode
    dynamic_shape = tf_shape(processed_dynamic)
    future_shape = tf_shape(processed_future)
    # This might indicate future_selected includes horizon steps
    # For LSTM input, we only need the past portion here.
    # This assumes T_past = self.dynamic_input_time_steps (if defined)
    # Let's assume dynamic_selected has T_past length
    
    num_dynamic_steps = dynamic_shape[1] # T_past
    # Take only the first T_past steps from future_selected
    num_future_steps = future_shape[1]   # T_future_total
    # Warning: This might discard future info if not handled later
    
    # Check T_future_total >= T_past (critical for slicing)
    # Use tf_debugging.assert for graph-mode check
    tf_assert = tf_debugging.assert_greater_equal(
        num_future_steps, num_dynamic_steps,
        message=( # Use tuple for message args in TF assert
            f"Future input time steps ({num_future_steps}) must be >= "
            f"Dynamic input time steps ({num_dynamic_steps}) for LSTM input prep."
        )
    )
    # Ensure the assertion is part of the graph execution
    with tf_control_dependencies([tf_assert]):
        # Slice future features to match the lookback period (T_past)
        future_selected_for_lstm = processed_future[:, :num_dynamic_steps, :]
        # Shape: (Batch, T_past, HiddenUnits)

    # --- Concatenate Features ---
    # Concatenate along the feature dimension (last axis)
    combined_lstm_input = tf_concat(
        [processed_dynamic, future_selected_for_lstm], axis=-1
        )
    # Shape: (Batch, T_past, DynamicFeatures + FutureFeatures)
    # Note: Assuming dynamic/future selected features have same HiddenUnits dim
    # Comment: Combined dynamic past and known future features for LSTM window.

    return combined_lstm_input


def _get_batch_size_for_val( 
    t: Union[np.ndarray, Tensor],
    verbose: int = 0
    ) -> Tensor:
    """Return batch size as a TF tensor, preferring static."""
    if not isinstance(t, Tensor):
        t_tensor = tf_convert_to_tensor(t)
    else:
        t_tensor = t
    rank = tf_rank(t_tensor)
    # Assert tensor is at least 1D to have a batch dimension.
    tf_debugging.assert_greater_equal(
        rank, tf_constant(1, dtype=rank.dtype),
        message=(
        "Input to _get_batch_size_for_val must have rank ≥ 1: "
        f"got rank {rank}."
        ),
        summarize=1
    )
    batch_s = tf_shape(t_tensor)[0]
    if verbose >= 7: # Very detailed debug
        print(f"    _get_batch_size_for_val: input "
              f"{getattr(t_tensor, 'shape', 'N/A')}, got {batch_s}")
    return batch_s

@optional_tf_function
def _validate_tensor_basic(
    data_input: Optional[Union[np.ndarray, Tensor]],
    name: str,
    expected_rank_int: int,
    expected_feat_dim: Optional[int],
    mode: str, # 'strict' or 'soft'
    error: str,
    verbose: int
) -> Optional[Tensor]:
    """
    Basic validation: type, rank, and optionally feature dimension.
    Returns processed tensor or None.
    """
    if data_input is None:
        # If input is None, check if its dimension was specified as required.
        if expected_feat_dim is not None and mode == 'strict' \
                and error == "raise":
            # This implies a mismatch between model config and provided data.
            raise ValueError(
                f"{name} input is None but its dimension "
                f"({expected_feat_dim}) was specified as required "
                "for the model in 'strict' mode."
            )
        if verbose >= 5: print(f"      {name} is None, skipping checks.")
        return None # Keep None if optional and not provided

    # Convert to TensorFlow tensor and ensure float32 type.
    if not isinstance(data_input, Tensor): # tf.Tensor or KerasTensor
        try:
            data_input = tf_convert_to_tensor(
                data_input, dtype=tf_float32
                )
        except Exception as e:
            raise TypeError(
                f"Failed to convert {name} input to tensor: {e}"
                ) from e
    elif data_input.dtype != tf_float32:
        data_input = tf_cast(data_input, tf_float32)

    # Check rank using TensorFlow operations for graph safety.
    current_rank_tensor = tf_rank(data_input)
    expected_rank_tensor = tf_constant(
        expected_rank_int, dtype=current_rank_tensor.dtype
        )
    # Graph-compatible assertion for rank.
    # Graph-compatible assertion for rank.
    tf_debugging_assert_equal(
        current_rank_tensor,
        expected_rank_tensor,
        "%s input must be %dD, but got rank %d for input shape %s.",
        name,
        expected_rank_int,
        current_rank_tensor,
        tf_shape(data_input),
        summarize=3  # Show more shape details in the error.
    )
    
    # Check feature dimension if in 'strict' mode or if
    # expected_feat_dim was explicitly provided.
    if (
        mode == 'strict'
        or expected_feat_dim is not None
    ) and expected_feat_dim is not None:
    
        # Extract the actual feature dimension
        actual_feat_dim_tensor = tf_shape(data_input)[-1]
        expected_feat_dim_tensor = tf_constant(
            expected_feat_dim,
            dtype=actual_feat_dim_tensor.dtype
        )
    
        tf_debugging_assert_equal(
            actual_feat_dim_tensor,
            expected_feat_dim_tensor,
            "%s input last dimension mismatch: expected %d, got %d "
            "for input shape %s.",
            name,
            expected_feat_dim,
            actual_feat_dim_tensor,
            tf_shape(data_input),
            summarize=3  # Show more tensor details in the error.
        )
    
    if verbose >= 5:
        print(
            f"      {name} validated. "
            f"Shape: {data_input.shape}"
        )


    return data_input


def _get_batch_size_for_validation(
    t: Union[np.ndarray, Tensor],
    verbose: int = 0  # Verbosity for this helper
    ) -> Tensor: # Always returns a 0-D Tensor
    """
    Return the first-dimension batch size as a TensorFlow tensor.
    Uses tf.shape for graph compatibility.
    """
    # Ensure input is a tensor for tf.shape.
    if not isinstance(t, Tensor): # Check against base tf.Tensor
        t_tensor = tf_convert_to_tensor(t)
    else:
        t_tensor = t

    # Check rank before accessing shape element.
    rank = tf_rank(t_tensor)
    # tf.Assert is graph-compatible.
    tf_debugging.assert_greater_equal(
        rank, tf_constant(1, dtype=rank.dtype), # Must be at least 1D
        # message="Input tensor to _get_batch_size must be at least 1D.",
        # data=[rank], # Pass tensor for dynamic message formatting
        message=(
           "Input tensor to _get_batch_size must be at least 1D. "
           f"Input tensor rank: {rank}."
         ),
        summarize=1  # Summarize tensor value in error
        )

    batch_s = tf_shape(t_tensor)[0]
    if verbose >= 7: # Very detailed debug for this helper
        print(f"    _get_batch_size: input shape "
              f"{getattr(t_tensor, 'shape', 'N/A')}, "
              f"determined batch size: {batch_s}")
    return batch_s

# @optional_tf_function
def _validate_tft_flexible_inputs_soft_mode(
    inputs_raw: Union[Tensor, np.ndarray, List[Optional[Union[Tensor, np.ndarray]]]],
    verbose: int = 0
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Helper to infer static, dynamic, and future inputs for
    TFTFlexible in 'soft' mode.
    """
    if verbose >= 4:
        print("  Enter `_validate_tft_flexible_inputs_soft_mode`...")

    static_p, dynamic_p, future_p = None, None, None

    # Ensure inputs_raw is a list for consistent processing.
    if not isinstance(inputs_raw, (list, tuple)):
        # Single tensor provided.
        inputs_list = [inputs_raw]
    else:
        inputs_list = list(inputs_raw)

    num_provided_inputs = len(inputs_list)

    if verbose >= 5:
        print(f"    Flexible helper received {num_provided_inputs} input(s).")

    if num_provided_inputs == 1:
        # Single input: Assume it's dynamic.
        # Static and future will be None.
        # Basic rank check: dynamic should be 3D.
        inp0 = inputs_list[0]
        if inp0 is not None:
            rank0 = tf_rank(tf_convert_to_tensor(inp0))
            # Graph-compatible assertion for single input dimension in soft mode.
            tf_debugging.assert_equal(
                rank0,
                tf_constant(3, dtype=rank0.dtype),
                message=(
                    "Single input to tft_flex (soft mode)"
                    " must be 3D (dynamic): expected rank 3,"
                    f" got {rank0} for input shape {tf_shape(inp0)}."
                ),
                # Removed unsupported `data` kwarg; details are now in the message.
                summarize=3  # Show tensor details in the error.
            )
            dynamic_p = inp0
        if verbose >= 5:
            print("    Mode: Single input -> Dynamic.")

    elif num_provided_inputs == 2:
        # Two inputs: [dynamic, static] or [dynamic, future].
        # Infer based on ranks (Static=2D, Dynamic/Future=3D).
        inp0, inp1 = inputs_list[0], inputs_list[1]
        rank0 = tf_rank(tf_convert_to_tensor(inp0)) if inp0 is not None else -1
        rank1 = tf_rank(tf_convert_to_tensor(inp1)) if inp1 is not None else -1

        # Convert ranks to Python int for easier logic here,
        # as these are structural checks before deep validation.
        # This is safe as tf.rank on a defined tensor is a 0-D tensor.
        try:
            py_rank0 = rank0.numpy() if hasattr(rank0, 'numpy') else int(rank0)
            py_rank1 = rank1.numpy() if hasattr(rank1, 'numpy') else int(rank1)
        except: # Fallback if .numpy() fails in some context
            py_rank0 = -1 if rank0 is -1 else 3 # Assume 3D if tensor
            py_rank1 = -1 if rank1 is -1 else 3 # Assume 3D if tensor

        if py_rank0 == 3 and py_rank1 == 2:
            # [Dynamic (3D), Static (2D)]
            dynamic_p, static_p = inp0, inp1
            if verbose >= 5: print("    Mode: Two inputs -> Dynamic, Static.")
        elif py_rank0 == 3 and py_rank1 == 3:
            # [Dynamic (3D), Future (3D)]
            dynamic_p, future_p = inp0, inp1
            if verbose >= 5: print("    Mode: Two inputs -> Dynamic, Future.")
        elif py_rank0 == 2 and py_rank1 == 3:
            # [Static (2D), Dynamic (3D)] - User might pass in this order.
            static_p, dynamic_p = inp0, inp1
            if verbose >= 5: print("    Mode: Two inputs -> Static, Dynamic.")
        else:
            # Ambiguous or unsupported combination for 2 inputs.
            raise ValueError(
                "With two inputs for tft_flex (soft mode), expect one "
                "2D (static) and one 3D (dynamic), or two 3D "
                "(dynamic, future). Got ranks: "
                f"{py_rank0 if inp0 is not None else 'None'}, "
                f"{py_rank1 if inp1 is not None else 'None'}."
            )

    elif num_provided_inputs == 3:
        # Three inputs: Assume [static, dynamic, future] order.
        # This is the standard order for `validate_model_inputs`.
        static_p, dynamic_p, future_p = (
            inputs_list[0], inputs_list[1], inputs_list[2]
            )
        if verbose >= 5:
            print("    Mode: Three inputs -> Static, Dynamic, Future.")
        # Basic rank checks for this assumed order.
        if static_p is not None:
            rank_s = tf_rank(tf_convert_to_tensor(static_p))
            tf_debugging.assert_equal(
                rank_s,
                tf_constant(2, dtype=rank_s.dtype),
                message=(
                    "Static input (first of 3) must be 2D: "
                    f"expected rank 2, got {rank_s} for"
                    f" input shape {tf_shape(static_p)}."
                ), 
                summarize=3
            )
        if dynamic_p is not None:
            rank_d = tf_rank(tf_convert_to_tensor(dynamic_p))
            tf_debugging.assert_equal(
                rank_d,
                tf_constant(3, dtype=rank_d.dtype),
                message=(
                    "Dynamic input (second of 3) must be 3D: "
                    f"expected rank 3, got {rank_d} for input"
                    f" shape {tf_shape(dynamic_p)}."
                ), 
                summarize=3
            )
        if future_p is not None:
            rank_f = tf_rank(tf_convert_to_tensor(future_p))
            tf_debugging.assert_equal(
                rank_f,
                tf_constant(3, dtype=rank_f.dtype),
                message=(
                    "Future input (third of 3) must be 3D: "
                    f"expected rank 3, got {rank_f} for"
                    f" input shape {tf_shape(future_p)}."
                ),
                summarize=3
            )

    elif num_provided_inputs == 0:
        # No inputs provided, all remain None.
        if verbose >=5: print("    Mode: Zero inputs provided.")
    else:
        # More than 3 inputs, which is not standard for TFT types.
        raise ValueError(
            f"Received {num_provided_inputs} inputs. "
            "validate_model_inputs expects a list of 1 to 3 tensors "
            "([static], [dynamic], [future])."
        )

    if verbose >= 4:
        s_s = getattr(static_p, 'shape', "None")
        d_s = getattr(dynamic_p, 'shape', "None")
        f_s = getattr(future_p, 'shape', "None")
        print(f"  Exit `_validate_tft_flexible_inputs_soft_mode`. "
              f"Inferred shapes: S={s_s}, D={d_s}, F={f_s}")
    
    # Ensure the dynamic input is provided
    if dynamic_p is None:
        raise ValueError(
            "Parameter 'dynamic_p' is required and cannot be None. "
            "Please provide a valid dynamic input."
        )

    return static_p, dynamic_p, future_p

def validate_model_inputs(
    inputs: Union[Tensor, np.ndarray,
                 List[Optional[Union[np.ndarray, Tensor]]]],
    static_input_dim: Optional[int] = None,
    dynamic_input_dim: Optional[int] = None,
    future_covariate_dim: Optional[int] = None,
    forecast_horizon: Optional[int] = None,
    error: str = "raise",
    mode: str = "strict",
    deep_check: Optional[bool] = None,
    model_name: Optional[str] = None,
    verbose: int = 0,
    **kwargs
) -> Tuple[Optional[Tensor], Optional[Tensor],
           Optional[Tensor]]:
    r"""
    Validate and homogenise the triplet of tensors that acts as
    input to Temporal‑Fusion‑Transformer‑type models.
    (e.g. :pyfunc:`~fusionlab.nn._xtft.XTFT.call`).

    The helper inspects the list/tuple *inputs* and verifies
    fundamental structural constraints:

    * rank‐2 *static* tensors have shape
      :math:`(B,\;F_\text{static})`,
    * rank‐3 *dynamic* tensors have shape
      :math:`(B,\;T_\text{past},\;F_\text{dyn})`,
    * rank‐3 *future* tensors (known covariates) have shape
      :math:`(B,\;T_\text{future},\;F_\text{fut})` with
      :math:`T_\text{future}\ge T_\text{past}` and, if
      *forecast_horizon* is supplied, additionally

      .. math::

         T_\text{future}\;\ge\;
         T_\text{past}\;+\;\text{forecast\_horizon}.

    When ``model_name == 'tft_flex'`` and *mode* is ``'soft'`` the
    function can infer the rôle of each tensor (static / dynamic /
    future) by inspecting rank and feature dimension, thereby
    allowing more concise user code.

    The routine raises or logs informative diagnostics depending on
    *error* and *verbose* settings.

    Notes
    -----
    The helper calls two internal utilities:

    * ``_validate_tft_flexible_inputs_soft_mode`` – recognises the
      rôle of each tensor in *soft* mode.
    * ``_validate_tensor_basic`` – validates rank, dtype and
      feature‑dimension.
    * ``_get_batch_size_for_val`` – extracts batch dimension for
      consistency checks.

    Although these helpers are prefixed with “_”, they are listed
    here for completeness because they encapsulate most of the
    heavy lifting.

    Parameters
    ----------
    inputs : Union[Tensor, np.ndarray,
                  List[Optional[Union[np.ndarray, Tensor]]]]
        Triplet ``[static, dynamic, future]`` or, when
        *model_name* is ``'tft_flex'`` and *mode* is ``'soft'``,
        the tensors in any order.  See *Examples*.
    static_input_dim : int or None, optional
        Expected feature dimension of the static block.
    dynamic_input_dim : int or None, optional
        Expected feature dimension of the dynamic past block.
    future_covariate_dim : int or None, optional
        Expected feature dimension of the known‑future covariates.
    forecast_horizon : int or None, optional
        If given, checks that the future span is large enough for
        the decoder horizon (see equation above).
    error : {'raise', 'warn', 'ignore'}, default ``'raise'``
        Behaviour when a validation test fails.
    mode : {'strict', 'soft'}, default ``'strict'``
        *strict* enforces every rule; *soft* relaxes feature‑dim
        checks and allows automatic rôle inference for
        ``'tft_flex'``.
    deep_check : bool or None, deprecated
        Legacy switch superseded by *mode*.  Will be removed in a
        future release.
    model_name : str or None, optional
        Name of the caller model.  Currently recognised value is
        ``'tft_flex'``.
    verbose : int, default ``0``
        0 = silent, 1 = warnings, 2+ = verbose tracing.
    **kwargs
        Reserved for future extensions.

    Returns
    -------
    static : Tensor or None
        Sanitised static tensor, shape
        :math:`(B,\;F_\text{static})`.
    dynamic : Tensor or None
        Sanitised dynamic tensor, shape
        :math:`(B,\;T_\text{past},\;F_\text{dyn})`.
    future : Tensor or None
        Sanitised future tensor, shape
        :math:`(B,\;T_\text{future},\;F_\text{fut})`.

    Raises
    ------
    ValueError
        If a critical inconsistency is detected and
        ``error == 'raise'``.
    UserWarning
        When ``error == 'warn'`` and an inconsistency is found.

    Examples
    --------
    >>> import numpy as np
    >>> from fusionlab.nn._tensor_validation import validate_model_inputs
    >>> B, Tpast, Tfuture = 16, 12, 18
    >>> static  = np.random.rand(B, 5).astype("float32")
    >>> dynamic = np.random.rand(B, Tpast, 7).astype("float32")
    >>> future  = np.random.rand(B, Tfuture, 3).astype("float32")
    >>> s, d, f = validate_model_inputs(
    ...     [static, dynamic, future],
    ...     static_input_dim=5,
    ...     dynamic_input_dim=7,
    ...     future_covariate_dim=3,
    ...     forecast_horizon=6,
    ...     verbose=1
    ... )
    >>> s.shape, d.shape, f.shape
    ((16, 5), (16, 12, 7), (16, 18, 3))

    References
    ----------
    .. [1] Lim B., Zohren S., "Temporal Fusion Transformers for
           Interpretable Multi‑horizon Time Series Forecasting",
           *International Journal of Forecasting*, 2021.
    """

    # --- 0. Handle deep_check deprecation and mode setting ---
    if deep_check is not None:
        warnings.warn(
            "'deep_check' is deprecated and will be removed. "
            "Use 'mode' ('strict' or 'soft') instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if mode == 'strict' and not deep_check:
            mode = 'soft' # User explicitly set mode, override deep_check
        elif deep_check:
            mode = 'strict'
        else: # deep_check is False
            mode = 'soft'

    if mode not in ['strict', 'soft']:
        raise ValueError("`mode` must be 'strict' or 'soft'.")

    if verbose >= 2:
        print(f"Enter `validate_model_inputs` (mode='{mode}', "
              f"model_name='{model_name}', verbose={verbose})")

    # --- 1. Input Interpretation (Smart Handling for tft_flex) ---
    static_raw, dynamic_raw, future_raw = None, None, None

    if model_name == 'tft_flex' and mode == 'soft':
        if verbose >= 3:
            print("  Running 'tft_flex' in 'soft' mode: "
                  "inferring input roles.")
        # Helper infers roles based on number/rank of inputs.
        static_raw, dynamic_raw, future_raw = \
            _validate_tft_flexible_inputs_soft_mode(
                inputs, verbose=verbose
                )
    else:
        # Standard path: expect a list of 3 (some can be None).
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
            raise ValueError(
                f"`inputs` must be a list/tuple of 3 elements for "
                f"model '{model_name}' in '{mode}' mode: "
                "[static, dynamic, future]. "
                f"Received {len(inputs) if isinstance(inputs, (list,tuple)) else 1} "
                f"element(s) of type {type(inputs)}."
            )
        static_raw, dynamic_raw, future_raw = inputs

    if verbose >= 3:
        s_s = getattr(static_raw, 'shape', "None")
        d_s = getattr(dynamic_raw, 'shape', "None")
        f_s = getattr(future_raw, 'shape', "None")
        print(f"  Inputs after role assignment/initial unpack: "
              f"S={s_s}, D={d_s}, F={f_s}")

    # --- 2. Individual Tensor Validation (Type, Rank, Features) ---
    # Define properties for each input type in the order:
    # static, dynamic, future.
    input_properties = [
        {"name": "Static", "data": static_raw,
         "feat_dim": static_input_dim, "expected_rank": 2},
        {"name": "Dynamic", "data": dynamic_raw,
         "feat_dim": dynamic_input_dim, "expected_rank": 3},
        {"name": "Future", "data": future_raw,
         "feat_dim": future_covariate_dim, "expected_rank": 3},
    ]
    processed_tensors: List[Optional[Tensor]] = []

    for prop in input_properties:
        # For 'tft_flex' in 'soft' mode, expected_feat_dim might
        # be None if the corresponding *_input_dim was not passed.
        # _validate_tensor_basic handles this.
        current_expected_feat_dim = prop["feat_dim"]
        if model_name == 'tft_flex' and mode == 'soft' \
                and prop["data"] is not None:
            # In soft mode for tft_flex, don't enforce feat_dim
            # if it wasn't explicitly provided to the validator.
            # The model's __init__ will handle defaults.
            pass # feat_dim check will be skipped if None

        validated_tensor = _validate_tensor_basic(
            prop["data"], prop["name"], prop["expected_rank"],
            current_expected_feat_dim, # Pass potentially None dim
            mode, error, verbose
        )
        processed_tensors.append(validated_tensor)

    static_p, dynamic_p, future_p = processed_tensors

    # --- 3. Batch Size Consistency Check ---
    # (Keep existing batch size check logic using _get_batch_size_for_val
    #  and tf.debugging.assert_equal - this part was already robust)
    non_null_for_batch_check = [
        t for t in processed_tensors if t is not None
        ]
    if len(non_null_for_batch_check) > 1:
        if verbose >= 3:
            print("  Checking batch size consistency across inputs...")
        ref_batch_tensor = _get_batch_size_for_val( # Use renamed helper
            non_null_for_batch_check[0], verbose=verbose
            )
        for t_current in non_null_for_batch_check[1:]:
            current_batch_tensor = _get_batch_size_for_val( # Use renamed helper
                t_current, verbose=verbose
                )
            tf_debugging.assert_equal(
                ref_batch_tensor, current_batch_tensor,
               message=(
                    "Inconsistent batch sizes among inputs: "
                    f"reference batch size = {ref_batch_tensor},"
                    f" current batch size = {current_batch_tensor}."
                ), 
                # data=[ref_batch_tensor, current_batch_tensor],
                summarize=10
            )
        if verbose >= 3: print("    Batch sizes are consistent.")


    # --- 4. Time Dimension Consistency (Dynamic vs. Future) ---
    # (Keep existing time dimension check logic - also robust)
    if dynamic_p is not None and future_p is not None:
        if verbose >= 3:
            print("  Checking time dim (dynamic vs future)...")
        t_past_dyn = tf_shape(dynamic_p)[1]
        t_span_fut = tf_shape(future_p)[1]
        
        tf_debugging.assert_greater_equal(
            t_span_fut, t_past_dyn,
            message=(
               "Future input time span must be >= dynamic input time span: "
               f"future span = {t_span_fut}, past dynamic span = {t_past_dyn}."
           ),
            # data=[t_span_fut, t_past_dyn], 
            summarize=10
        )
        # if forecast_horizon is not None:
            # fh_tensor = tf_cast(forecast_horizon, dtype=t_span_fut.dtype)
            # req_fut_span = t_past_dyn + fh_tensor
            # if tf_less(t_span_fut, req_fut_span) and verbose >= 1:
            #     t_s_val = tf_get_static_value(t_span_fut, partial=True)
            #     t_p_val = tf_get_static_value(t_past_dyn, partial=True)
            #     warnings.warn(
            #         f"Future input time span ({t_s_val}) is less "
            #         f"than dynamic lookback ({t_p_val}) + "
            #         f"forecast_horizon ({forecast_horizon}). May be "
            #         "insufficient for decoder.", UserWarning
            #     )
        if forecast_horizon is not None and verbose >= 1:
            # try to get concrete Python ints
            t_s_val = tf_get_static_value(t_span_fut, partial=True)
            t_p_val = tf_get_static_value(t_past_dyn, partial=True)
        
            # only warn if both spans are statically known
            if (
                t_s_val is not None
                and t_p_val is not None
                and t_s_val < (t_p_val + forecast_horizon)
            ):
                warnings.warn(
                    f"Future input time span ({t_s_val}) is less than "
                    f"dynamic lookback ({t_p_val}) + forecast_horizon "
                    f"({forecast_horizon}). May be insufficient for decoder.",
                    UserWarning
                )


        if verbose >= 3:
            print("    Time dimensions (dynamic vs future) compatible.")

    if verbose >= 2:
        s_s = static_p.shape if static_p is not None else 'None'
        d_s = dynamic_p.shape if dynamic_p is not None else 'None'
        f_s = future_p.shape if future_p is not None else 'None'
        print(f"Exit `validate_model_inputs`. Processed shapes: "
              f"S={s_s}, D={d_s}, F={f_s}")
    
    # XXX TODO: Use 'prepare model so  model can handle None if passe. 
    #     None should systematically converted to a zeros tensors. 
    
    # try: 
    #     from .utils import prepare_model_inputs 
    # except: 
    #    # For safety try to reconvert the tensor to zeros tensor. 
    #    static_p, dynamic_p, future_p = prepare_model_inputs(
    #        static_input= static_p, 
    #        dynamic_input= dynamic_p, 
    #        future_input= future_p,
    #        # forecast_horizon= forecast_horizon # mute since it has been done. 
    #        model_type="strict",# for convertir None to zeros tensor.
    #       )
       
    # Return in the order: static, dynamic, future
    return static_p, dynamic_p, future_p


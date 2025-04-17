# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import warnings
from typing import List, Tuple, Optional, Union, Dict, Any
from ..utils.deps_utils import ensure_pkg 
from ..compat.tf import optional_tf_function, suppress_tf_warnings
from ..compat.tf import TFConfig, HAS_TF  
from . import KERAS_DEPS, KERAS_BACKEND

import numpy as np 

if KERAS_BACKEND:
    Tensor=KERAS_DEPS.Tensor

    tf_shape = KERAS_DEPS.shape
    tf_float32=KERAS_DEPS.float32
    tf_convert_to_tensor =KERAS_DEPS.convert_to_tensor 
    tf_cast=KERAS_DEPS.cast 
    tf_reduce_all=KERAS_DEPS.reduce_all
    tf_equal=KERAS_DEPS.equal 
    tf_debugging= KERAS_DEPS.debugging 
    tf_assert_equal=KERAS_DEPS.assert_equal
    tf_autograph=KERAS_DEPS.autograph
    register_keras_serializable=KERAS_DEPS.register_keras_serializable
    tf_expand_dims=KERAS_DEPS.expand_dims
    tf_autograph.set_verbosity(0)
    
else: 
   # Warn the user that TensorFlow
   # is required for this module
    warnings.warn(
        "TensorFlow is not installed. Please install"
        " TensorFlow to use this module.",
        ImportWarning
    )
    
if HAS_TF:
    config = TFConfig()
    # Enable compatibility mode for ndim
    config.compat_ndim_enabled = True 

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
        - dtype: tf.float32
        - shape: (batch_size, features) with at least 2 dimensions

    Returns
    -------
    Tensor
        Validated anomaly scores tensor with:
        - dtype: tf.float32
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
    >>> from gofast.nn._tensor_validation import validate_tft_inputs
    >>> import tensorflow as tf
    >>> # Example with only past (dynamic) inputs
    >>> single_input = tf.random.normal([8, 24, 10])  # batch=8, time=24
    >>> past, fut, stat = validate_tft_inputs(
    ...     [single_input], dynamic_input_dim=10
    ... )
    >>> print(past.shape)
    (8, 24, 10)
    >>> print(fut, stat)  # None, None

    >>> # Example with two inputs: dynamic past and static
    >>> dynamic_in = tf.random.normal([8, 24, 20])
    >>> static_in  = tf.random.normal([8, 5])
    >>> past, fut, stat = validate_tft_inputs(
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
            # if they're both int, else a tf.cond approach if needed.
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
                # At least one is dynamic. We'll do a tf-level check.
                # In eager mode, we can still evaluate it directly. 
                # Let's do so carefully.
                are_equal = tf_reduce_all(
                    tf_equal(ref_batch_size, batch_size)
                )
                if not bool(are_equal.numpy()):
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
    >>> from gofast.nn._tensor_validation import validate_xtft_inputs 
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
    - tf.errors.InvalidArgumentError: 
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
    - tf.errors.InvalidArgumentError: 
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

def validate_minimal_inputs(
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
    `X_static`       : np.ndarray or tf.Tensor
        The static feature input, expected to have shape (``B``, ``N_s``).
    `X_dynamic`      : np.ndarray or tf.Tensor
        The dynamic feature input, expected to have shape (``B``, ``F``, 
        ``N_d``).
    `X_future`       : np.ndarray or tf.Tensor
        The future feature input, expected to have shape (``B``, ``F``, 
        ``N_f``).
    `y`              : np.ndarray or tf.Tensor, optional
        The target output, expected to have shape (``B``, ``F``, ``O``).
    `forecast_horizon`: int, optional
        The expected forecast horizon (``F``). If provided and it differs 
        from the input data, a warning is issued and the input forecast 
        horizon is used.
    `deep_check`     : bool, optional
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
        If an input is not an instance of np.ndarray or tf.Tensor.
    
    Examples
    --------
    >>> from gofast.nn._tensor_validation import validate_minimal_inputs
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
    np.ndarray.astype, tf.cast
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
                "Must be np.ndarray or tf.Tensor."
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
            "`gofast.nn.utils.reshape_xft_data`."
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
 






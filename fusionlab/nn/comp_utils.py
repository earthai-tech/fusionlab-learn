# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

"""
Component helper utilities.
"""
from __future__ import annotations
from typing import Union, Dict, List, Optional, Tuple, Any
from ..params import LearnableC, FixedC, DisabledC
from ..params import LearnableK, LearnableSs, LearnableQ 
from . import KERAS_BACKEND, KERAS_DEPS

if KERAS_BACKEND:
    Tensor = KERAS_DEPS.Tensor
    tf_rank = KERAS_DEPS.rank
else:
    Tensor = object

def resolve_gw_coeffs(
    gw_flow_coeffs: Optional[Dict[str, Any]] = None,
    K: Optional[Union[Any, 'LearnableK']] = 1e-4,
    Ss: Optional[Union[Any, 'LearnableSs']] = 1e-5,
    Q: Optional[Union[Any, 'LearnableQ']] = 0.0,
    param_status: Optional[str] = None  # 'learnable', 'fixed', or 'auto'
) -> Tuple[Any, Any, Any]:
    """
    Resolves hydraulic conductivity (K), specific storage (Ss), and 
    source/sink term (Q) from a dictionary, with fallbacks to default values.

    This function checks for the presence of keys `'K'`, `'Ss'`, and `'Q'` 
    in the `gw_flow_coeffs` dictionary. If a key is present, its value is 
    used. Otherwise, the corresponding default value is applied. The function 
    also resolves whether the parameters are learnable or fixed, depending 
    on the `param_status`.

    Parameters
    ----------
    gw_flow_coeffs : dict, optional
        A dictionary containing the coefficients for hydraulic conductivity (`K`), 
        specific storage (`Ss`), and the source/sink term (`Q`). If provided, 
        these values will override the default ones.
        
    K : Any, optional
        The default value for hydraulic conductivity (K), if not provided in 
        `gw_flow_coeffs`. Default is :math:`1e-4`.
    
    Ss : Any, optional
        The default value for specific storage (Ss), if not provided in 
        `gw_flow_coeffs`. Default is :math:`1e-5`.
    
    Q : Any, optional
        The default value for the source/sink term (Q), if not provided in 
        `gw_flow_coeffs`. Default is :math:`0.0`.

    param_status : str, optional
        Defines how the parameters are handled:
        - `'learnable'`: Treat all parameters as learnable variables.
        - `'fixed'`: Keep parameters fixed (as constants).
        - `'auto'`: Automatically detect whether parameters are fixed or 
          learnable. If `gw_flow_coeffs` contains instances of learnable 
          parameters (e.g., `LearnableK`), they will be treated as learnable.

    Returns
    -------
    Tuple[Any, Any, Any]
        A tuple containing the resolved values for `(K, Ss, Q)`. These can 
        either be fixed values (e.g., `float`) or learnable instances 
        (e.g., `LearnableK`, `LearnableSs`, `LearnableQ`).

    Raises
    ------
    ValueError
        If `param_status` is invalid or if the conversion of string values 
        to float fails.

    Notes
    -----
    - If `param_status` is `'learnable'`, all parameters are treated as 
      learnable.
    - If `param_status` is `'fixed'`, the parameters are treated as fixed 
      constants.
    - If `param_status` is `'auto'`, the function checks if the parameters 
      are instances of `LearnableK`, `LearnableSs`, or `LearnableQ`, and 
      treats them as learnable if so.
    
    Examples
    --------
    >>> resolve_gw_coeffs(None, K=1e-4, Ss=1e-5, Q=0.0)
    (1e-4, 1e-5, 0.0)
    
    >>> coeffs = {'K': 99, 'Ss': 88, 'Q': 77}
    >>> resolve_gw_coeffs(gw_flow_coeffs=coeffs)
    (99, 88, 77)
    
    >>> coeffs_partial = {'K': 99}
    >>> resolve_gw_coeffs(gw_flow_coeffs=coeffs_partial, Ss=1e-5, Q=0.0)
    (99, 1e-5, 0.0)
    
    >>> resolve_gw_coeffs(K='1e-4', Ss='1e-5', Q='0.0')
    (0.0001, 1e-05, 0.0)
    
    .. math::
        K = \text{{hydraulic conductivity}} \quad \text{{(e.g., }} K = 1e-4\text{{)}}
        
        Ss = \text{{specific storage}} \quad \text{{(e.g., }} Ss = 1e-5\text{{)}}
        
        Q = \text{{source/sink term}} \quad \text{{(e.g., }} Q = 0.0\text{{)}}

    References
    ----------
    - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., 
      Gomez, A., Kaiser, Ł., Polosukhin, I. (2017). Attention is all you 
      need. *NeurIPS 2017*, 30, 6000-6010.
    - Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine 
      Translation by Jointly Learning to Align and Translate. *ICLR 2015*.
    """
    
    def _resolve_param(param, default_value, param_name):
        """Helper function to handle parameter resolution logic."""
        if isinstance(param, str):
            if param_status == 'learnable':
                # Treat it as learnable if specified
                param = default_value
            try:
                # Attempt to convert to float
                param = float(param)
            except ValueError:
                raise ValueError(
                    f"Invalid value for {param_name}:"
                    f" cannot convert '{param}' to float.")
        
        elif isinstance(param, (int, float)):
            # Fixed value, convert to float
            param = float(param)
        
        elif isinstance(param, (LearnableK, LearnableSs, LearnableQ)):
            # If param is already a learnable instance, keep as is
            pass
        
        else:
            raise ValueError(
                f"Invalid type for {param_name}: {type(param).__name__}")
        
        return param

    # Default Learnable Parameters if not provided
    default_K = LearnableK(
        initial_value=K, name="param_K") if isinstance(K, (int, float)) else K
    default_Ss = LearnableSs(
        initial_value=Ss, name="param_Ss") if isinstance(Ss, (int, float)) else Ss
    default_Q = LearnableQ(
        initial_value=Q, name="param_Q") if isinstance(Q, (int, float)) else Q

    # Resolve the parameters (K, Ss, Q) using the provided dictionary or default values
    if isinstance(gw_flow_coeffs, dict):
        K = gw_flow_coeffs.get('K', default_K)
        Ss = gw_flow_coeffs.get('Ss', default_Ss)
        Q = gw_flow_coeffs.get('Q', default_Q)

    # Apply the learned parameter logic based on param_status
    K = _resolve_param(K, default_value=default_K, param_name='K')
    Ss = _resolve_param(Ss, default_value=default_Ss, param_name='Ss')
    Q = _resolve_param(Q, default_value=default_Q, param_name='Q')

    return K, Ss, Q

def split_decoder_outputs(
    predictions_combined: Tensor,
    decoded_outputs_for_mean: Tensor,
    output_dims: Dict[str, int],
    quantiles: Optional[List[float]] = None
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Splits a combined output tensor into individual target streams.

    This utility takes the stacked output from a multi-target model
    and separates it into predictions for each target variable. It
    correctly handles both deterministic (point) and probabilistic
    (quantile) forecasts.

    Parameters
    ----------
    predictions_combined : tf.Tensor
        The final output tensor from the model's last layer (e.g.,
        QuantileDistributionModeling). Shape can be
        `(B, H, Sum_of_O)` for point forecasts or
        `(B, H, Q, Sum_of_O)` for quantile forecasts.

    decoded_outputs_for_mean : tf.Tensor
        The output tensor from the decoder *before* the quantile
        layer. This is used to extract the mean predictions for
        physics loss calculations. Shape is `(B, H, Sum_of_O)`.

    output_dims : Dict[str, int]
        A dictionary mapping the name of each target to its feature
        dimension. The order of keys determines the splitting order.
        Example: `{'subs_pred': 1, 'gwl_pred': 1}`.

    quantiles : List[float], optional
        A list of quantiles used in the forecast. The presence of
        this list indicates that `predictions_combined` has a
        quantile dimension.

    Returns
    -------
    final_preds_dict : Dict[str, tf.Tensor]
        A dictionary mapping each target name to its final prediction
        tensor (for data loss).

    mean_preds_dict : Dict[str, tf.Tensor]
        A dictionary mapping each target name to its mean prediction
        tensor (for physics loss).
    """
    final_preds_dict = {}
    mean_preds_dict = {}
    start_idx = 0

    # First, split the mean predictions (always 3D)
    for target_name, dim in output_dims.items():
        end_idx = start_idx + dim
        mean_preds_dict[target_name] = decoded_outputs_for_mean[
            ..., start_idx:end_idx
        ]
        start_idx = end_idx

    # Reset start index for the final predictions
    start_idx = 0
    # Determine the axis where features are concatenated
    # For quantiles (B, H, Q, O), it's axis -1
    # For point (B, H, O), it's also axis -1
    feature_axis = -1 # noqa
    
    # Keras sometimes returns a different rank in graph vs eager.
    # A robust check on rank is necessary.
    is_quantile_mode = quantiles is not None
    
    if is_quantile_mode:
        # Expected rank is 4: (B, H, Q, O_combined)
        # We need to handle slicing on the last dimension
        for target_name, dim in output_dims.items():
            end_idx = start_idx + dim
            final_preds_dict[target_name] = predictions_combined[
                ..., start_idx:end_idx
            ]
            start_idx = end_idx
    else:
        # Expected rank is 3: (B, H, O_combined)
        for target_name, dim in output_dims.items():
            end_idx = start_idx + dim
            final_preds_dict[target_name] = predictions_combined[
                ..., start_idx:end_idx
            ]
            start_idx = end_idx
            
    return final_preds_dict, mean_preds_dict


def normalize_C_descriptor(
    raw: Union[LearnableC, FixedC, DisabledC, str, float, None]
):
    """
    Internal helper: turn the user‐passed `raw` into exactly one of
    our three classes (LearnableC, FixedC, DisabledC).

    Raises
    ------
    ValueError
        If `raw` is an unrecognized string or negative float.
    TypeError
        If `raw` is not one of the expected types.
    """
    # 1) Already one of our classes?
    if isinstance(raw, (LearnableC, FixedC, DisabledC)):
        return raw

    # 2) If user passed a bare float or int: treat as FixedC(value=raw)
    if isinstance(raw, (float, int)):
        if raw < 0:
            raise ValueError(
                "Numeric pinn_coefficient_C must"
                f" be non‐negative, got {raw}"
            )
        # Nonzero means 'fixed to that value'
        return FixedC(value=float(raw))

    # 3) If user passed a string, allow the legacy
    # values "learnable" or "fixed"
    if isinstance(raw, str):
        low = raw.strip().lower()
        if low == "learnable":
            # Default initial value = 0.01 is built into LearnableC
            return LearnableC(initial_value=0.01)
        if low == "fixed":
            # Default fixed value = 1.0
            return FixedC(value=1.0)
        if low in ("none", "disabled", "off"):
            return DisabledC()

        raise ValueError(
            f"Unrecognized pinn_coefficient_C string: '{raw}'. "
            "Expected 'learnable', 'fixed', 'none', or use"
            " a LearnableC/FixedC/DisabledC instance."
        )

    # 4) If user passed None, treat as DisabledC()
    if raw is None:
        return DisabledC()

    raise TypeError(
        f"pinn_coefficient_C must be LearnableC, FixedC, DisabledC, "
        f"str, float, or None; got {type(raw).__name__}."
    )
    
def resolve_attention_levels(
    att_levels: Union[str, List[str], int, None]
) -> List[str]:
    """
    Resolves the attention levels based on the input parameter 
    `att_levels`. This function checks and returns the appropriate 
    attention mechanisms to be used in the model.

    Parameters
    ----------
    att_levels : str, list of str, int, or None
        - If None or 'use_all' or '*', it will use all attention 
          mechanisms: ['cross', 'hierarchical', 'memory'].
        - If 'hier_att' or 'hierarchical_attention', only hierarchical 
          attention is used.
        - If 'memo_aug_att' or 'memory_augmented_attention', only memory- 
          augmented attention is used.
        - If a list of strings, it will use those specific attention 
          types in the provided order.
        - If an integer (1, 2, 3), maps it to cross attention (1), 
          hierarchical attention (2), or memory-augmented attention (3).
    
    Returns
    -------
    List[str]
        A list of attention mechanisms to be applied in the specified 
        order.
    
    Notes
    -----
    - The function handles various types of inputs, including strings, 
      lists, integers, and `None`. The order of the list is important 
      since the attention mechanisms will be applied accordingly.
    - If an invalid type or attention level is provided, an error will 
      be raised.
    
    References
    ----------
    - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., 
      Gomez, A., Kaiser, Ł., Polosukhin, I. (2017). Attention is all 
      you need. *NeurIPS 2017*, 30, 6000-6010.
    """
    
    # Define the mapping for attention types
    attention_map = {
        'cross': 'cross',
        'cross_att': 'cross',
        'cross_attention': 'cross',
        'hier': 'hierarchical',
        'hierarchical': 'hierarchical', 
        'hier_att': 'hierarchical',
        'hierarchical_attention': 'hierarchical',
        'memory': 'memory',
        'memo_aug': 'memory',
        'memo_aug_att': 'memory',
        'memory_augmented_attention': 'memory',
    }
    
    # If att_levels is None or 'use_all' or '*', use all attentions
    if att_levels is None or att_levels in ['use_all', '*']:
        return ['cross', 'hierarchical', 'memory']
    
    # If att_levels is a single string, check for valid attention types
    elif isinstance(att_levels, str):
        # If it's a valid attention type, return it in a list
        if att_levels in attention_map:
            return [attention_map[att_levels]]
        # Handle the case where it's a number as a string (e.g., '1' or '2')
        elif att_levels == '1':
            return ['cross']
        elif att_levels == '2':
            return ['hierarchical']
        elif att_levels == '3':
            return ['memory']
        else:
            raise ValueError(f"Invalid attention type: {att_levels}")
    
    # If att_levels is a list of strings, process each one
    elif isinstance(att_levels, list):
        # Validate that all entries are valid attention types
        valid_attentions = []
        for level in att_levels:
            if level in attention_map:
                valid_attentions.append(attention_map[level])
            elif level == '1':
                valid_attentions.append('cross')
            elif level == '2':
                valid_attentions.append('hierarchical')
            elif level == '3':
                valid_attentions.append('memory')
            else:
                raise ValueError(
                    f"Invalid attention type: {level}"
                    )
        return valid_attentions
    
    # If att_levels is an integer (1, 2, 3), map it to the corresponding 
    # attention mechanism
    elif isinstance(att_levels, int):
        if att_levels == 1:
            return ['cross']
        elif att_levels == 2:
            return ['hierarchical']
        elif att_levels == 3:
            return ['memory']
        else:
            raise ValueError("Invalid integer for attention type. Use 1, 2, or 3.")
    
    # If none of the above cases match, raise an error
    else:
        raise TypeError(f"Invalid type for att_levels: {type(att_levels)}. "
                          "Expected str, list, int, or None.")


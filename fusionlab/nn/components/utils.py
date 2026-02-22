

import warnings
from typing import Optional, Dict, Any, Union, List 

from ..._fusionlog import fusionlog 
from ...utils.generic_utils import select_mode

DEFAULT_ARCHITECTURE = {
    'encoder_type': 'hybrid',
    'decoder_attention_stack': ['cross', 'hierarchical', 'memory'],
    'feature_processing': 'vsn',
}

logger = fusionlog().get_fusionlab_logger(__name__)

__all__ =['resolve_attn_levels','configure_architecture']

def resolve_attn_levels(
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

def configure_architecture(
    objective: Optional[str] = None,
    use_vsn: bool = True,
    attention_levels: Optional[Union[str, List[str]]] = None,
    architecture_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Initializes and validates the model's architectural configuration.

    This function centralizes the logic for setting up the model's internal
    architecture, merging default settings with user-provided arguments.

    The order of precedence is:
    1. Default architecture settings.
    2. Explicit keyword arguments (`objective`, `use_vsn`, etc.).
    3. User-provided `architecture_config` dictionary (overrides others).

    Parameters
    ----------
    objective : str, optional
        The high-level objective ('hybrid' or 'transformer').
    use_vsn : bool, default=True
        Whether to use Variable Selection Networks (VSN) for feature processing.
    attention_levels : Union[str, List[str]], optional
        The desired attention layers (e.g., ['cross', 'hierarchical', 'memory']).
    architecture_config : dict, optional
        A dictionary of specific architectural settings provided by the user.

    Returns
    -------
    dict
        The finalized architecture configuration.
    """
    # Step 1: Start with the default architecture
    final_config = DEFAULT_ARCHITECTURE.copy()

    # Step 2: Apply settings from explicit keyword arguments.
    # 2.1: Handle the `objective` argument (sets `encoder_type`)
    final_config['encoder_type'] = select_mode(
        objective, default='hybrid', canonical=['hybrid', 'transformer']
    )

    # 2.2: Set the `feature_processing` method based on `use_vsn`
    if not use_vsn:
        # Use dense processing if VSN is not enabled
        final_config['feature_processing'] = 'dense'  

    # 2.3: Resolve and apply the `attention_levels` argument to `decoder_attention_stack`
    final_config['decoder_attention_stack'] = resolve_attn_levels(
        attention_levels
    )

    # Step 3: Merge and override with the user-provided `architecture_config`
    if architecture_config:
        user_config = architecture_config.copy()

        # Handle deprecated 'objective' key in the user config for backward compatibility
        if 'objective' in user_config:
            warnings.warn(
                "The 'objective' key in `architecture_config` is deprecated. "
                "Please use 'encoder_type' instead.",
                FutureWarning
            )
            user_config['encoder_type'] = user_config.pop('objective')

        # Update the final config with the user-provided configuration
        final_config.update(user_config)

    # Step 4: Final validation and reconciliation.
    # Ensure `feature_processing` is consistent with `use_vsn`
    if not use_vsn and final_config.get('feature_processing') == 'vsn':
        logger.info(
            "`use_vsn=False` was passed, but `architecture_config` specified"
            " `feature_processing='vsn'`. Reverting to 'dense'."
        )
        final_config['feature_processing'] = 'dense'

    # Return the final architecture configuration
    return final_config

def resolve_fusion_mode(fusion_mode: Optional[str]=None) -> str:
    """
    Resolve the fusion mode to ensure a valid and consistent value.
    
    Parameters
    ----------
    fusion_mode : str or None
        The fusion mode selected by the user. Can be one of:
        - 'integrated' (default if None)
        - 'disjoint', 'independent', or 'isolated' (fallback to 'disjoint')
    
    Returns
    -------
    str
        The resolved fusion mode: 'integrated' or 'disjoint'
    """
    if fusion_mode is None:
        # Default to 'integrated' if no value is provided
        return 'integrated'
    
    fusion_mode = fusion_mode.lower()
    
    if fusion_mode in ['integrated']:
        return 'integrated'
    elif fusion_mode in ['independent', 'isolated', 'disjoint']:
        return 'disjoint'
    else:
        # In case of an invalid value, fallback to 'integrated'
        logger.warning(f"Invalid fusion_mode value '{fusion_mode}' provided. "
                       "Falling back to 'integrated'.")
        return 'integrated'


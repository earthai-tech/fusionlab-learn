# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""fusionlab.nn.components

Flat public API re-export for all component blocks.
"""

from .attention import (              
    TemporalAttentionLayer,
    CrossAttention,
    MemoryAugmentedAttention,
    HierarchicalAttention,
    ExplainableAttention,
    MultiResolutionAttentionFusion,
)

from ._attention_utils import (        
    create_causal_mask,
    combine_masks,
)

from .masks import (                   
    pad_mask_from_lengths,
    sequence_mask_3d,
)

from .gating_norm import (          
    GatedResidualNetwork,
    VariableSelectionNetwork,
    LearnedNormalization,
    StaticEnrichmentLayer,
)
from .temporal import (                
    MultiScaleLSTM,
    DynamicTimeWindow, 
)  
from .temporal_utils import (  
    aggregate_multiscale,
    aggregate_multiscale_on_3d,
    aggregate_time_window_output,
)
)

from .misc import (                    
    MultiModalEmbedding,
    PositionalEncoding,
    TSPositionalEncoding,
    Activation,
)
from .losses import (            
    AdaptiveQuantileLoss,
    MultiObjectiveLoss,
    AnomalyLoss,
    CRPSLoss
    
)
from .heads import (            
    QuantileDistributionModeling,
    CombinedHeadLoss,       
    QuantileHead, 
    PointForecastHead, 
    GaussianHead 
    MixtureDensityHead 

)
from .encoder_decoder import (           
    MultiDecoder,
    TransformerEncoderBlock,   
    TransformerDecoderBlock, 
)

__all__ = [
        # attention.py
        "TransformerEncoderLayer",
        "TransformerDecoderLayer",
        "TemporalAttentionLayer",
        "CrossAttention",
        "MemoryAugmentedAttention",
        "HierarchicalAttention",
        "ExplainableAttention",
        "MultiResolutionAttentionFusion",
        # _attention_utils.py
        "create_causal_mask",
        "combine_masks",
        # masks.py
        "pad_mask_from_lengths",
        "sequence_mask_3d",
        # gating_norm.py
        "GatedResidualNetwork",
        "VariableSelectionNetwork",
        "LearnedNormalization",
        "StaticEnrichmentLayer",
        # temporal.py
        "MultiScaleLSTM",
        "aggregate_multiscale",
        "aggregate_multiscale_on_3d",
        # temporal_utils.py
        "DynamicTimeWindow",
        "aggregate_time_window_output",
        # misc.py
        "MultiModalEmbedding",
        "PositionalEncoding",
        "TSPositionalEncoding",
        "Activation",
        # _loss_utils.py
        "AdaptiveQuantileLoss",
        "MultiObjectiveLoss",
        "QuantileDistributionModeling",
        "CRPSLoss", 
        "AnomalyLoss",
        # heads_losses.py
        "CombinedHeadLoss",
        "QuantileHead",
        "PointForecastHead",
        "MixtureDensityHead", 
        
        # utils_layers.py
        "MultiDecoder",
        "TransformerEncoderBlock",
        "TransformerDecoderBlock",
    ]


# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Provides a collection of specialized Keras-compatible 
layers and components for constructing advanced time series 
forecasting and anomaly detection models. It includes building 
blocks such as attention mechanisms, multi-scale LSTMs, gating 
and normalization layers, and multi-objective loss functions.
"""
from ._attention_utils import (        
    create_causal_mask,
    combine_masks,
)
from ._loss_utils import ( 
    MeanSquaredErrorLoss,
    QuantileLoss,
    HuberLoss,
    WeightedLoss,
    compute_loss_with_reduction, 
    compute_quantile_loss 
)

from ._temporal_utils import (  
    aggregate_multiscale,
    aggregate_multiscale_on_3d,
    aggregate_time_window_output,
)
from .attention import (              
    TemporalAttentionLayer,
    CrossAttention,
    MemoryAugmentedAttention,
    HierarchicalAttention,
    ExplainableAttention,
    MultiResolutionAttentionFusion,
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
    GaussianHead, 
    MixtureDensityHead 
)
from .encoder_decoder import (           
    MultiDecoder,
    TransformerEncoderBlock,   
    TransformerDecoderBlock, 
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    
)

from .layer_utils import ( 
    maybe_expand_time,
    broadcast_like,
    ensure_rank_at_least,
    apply_residual,
    drop_path,

    ResidualAdd,
    LayerScale,
    StochasticDepth,
    SqueezeExcite1D,
    Gate,
    )

__all__ = [

        "TransformerEncoderLayer",
        "TransformerDecoderLayer",
        "TemporalAttentionLayer",
        "CrossAttention",
        "MemoryAugmentedAttention",
        "HierarchicalAttention",
        "ExplainableAttention",
        "MultiResolutionAttentionFusion",
        "ResidualAdd",
        "LayerScale",
        "StochasticDepth",
        "SqueezeExcite1D",
        "Gate",
        "GatedResidualNetwork",
        "VariableSelectionNetwork",
        "LearnedNormalization",
        "StaticEnrichmentLayer",
        "MultiScaleLSTM",
        "DynamicTimeWindow",
        "MultiModalEmbedding",
        "PositionalEncoding",
        "TSPositionalEncoding",
        "Activation",
        "AdaptiveQuantileLoss",
        "MultiObjectiveLoss",
        "QuantileDistributionModeling",
        "CRPSLoss", 
        "AnomalyLoss",
        "CombinedHeadLoss",
        "QuantileHead",
        "PointForecastHead",
        "MixtureDensityHead", 
        "GaussianHead", 
        "MultiDecoder",
        "TransformerEncoderBlock",
        "TransformerDecoderBlock",
        "MeanSquaredErrorLoss",
        "QuantileLoss",
        "HuberLoss",
        "WeightedLoss",
        
        "compute_loss_with_reduction", 
        "compute_quantile_loss", 
        "maybe_expand_time",
        "broadcast_like",
        "ensure_rank_at_least",
        "apply_residual",
        "drop_path",
        "create_causal_mask",
        "combine_masks",
        "pad_mask_from_lengths",
        "sequence_mask_3d",
        "aggregate_multiscale",
        "aggregate_multiscale_on_3d",
        "aggregate_time_window_output",
        
]
    


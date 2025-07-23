# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Encoder/Decoder building blocks (Transformer-style + generic decoders).
"""

from __future__ import annotations

from typing import Optional

from ...api.property import NNLearner
from ...utils.deps_utils import ensure_pkg
from ._config import KERAS_BACKEND, DEP_MSG 
from ._config import (                             
    Layer, 
    Dense,
    Dropout,
    LayerNormalization, 
    Sequential,
    MultiHeadAttention,
    Tensor, 
    
    register_keras_serializable,
    tf_stack,
    tf_autograph, 
    # tf_shape,
    # tf_unstack, 
    # tf_abs, 
    # tf_constant,
    # tf_float32, 
    # tf_reshape, 
    # tf_maximum, 
    # tf_reduce_mean, 
    # tf_square, 
    # tf_sqrt, 
    # tf_erf
)

__all__ = [
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "MultiDecoder",
    "TransformerEncoderBlock", 
    "TransformerDecoderBlock"
    
]

_EPSILON= 1E-6

@register_keras_serializable(
        "fusionlab.nn.components", 
        name="TransformerEncoderBlock"
   )
class TransformerEncoderBlock(Layer):
    """
    Transformer Encoder Block:
    Consists of multi-head self-attention and position-wise
    feed-forward network.
    
    Args:
    - embed_dim (int): The dimensionality of the embedding (and
      output of the attention).
    - num_heads (int): The number of attention heads.
    - ffn_dim (int): The dimensionality of the feed-forward
      network.
    - dropout_rate (float): Dropout rate to be applied after each
      layer.
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)    
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)

        # Multi-Head Self-Attention Layer
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim,
                                      dropout=dropout_rate)

        # Position-wise Feed-Forward Network
        self.ffn = Sequential([
            Dense(ffn_dim, activation='relu'),
            Dense(embed_dim)
        ])

        # Layer Normalization and Dropout
        self.layernorm1 = LayerNormalization(epsilon=_EPSILON)
        self.layernorm2 = LayerNormalization(epsilon=_EPSILON)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs: Tensor, training: bool = False,
             mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the encoder block.

        Args:
        - inputs: Tensor of shape (batch_size, seq_len, embed_dim)
        - training: Boolean flag to indicate if training mode is active
        - mask: Optional mask to apply on attention

        Returns:
        - output: Tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Multi-Head Self Attention
        attn_output = self.mha(inputs, inputs, inputs,
                               attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)

        # Add & Norm
        out1 = self.layernorm1(inputs + attn_output)  # Residual connection

        # Feed-Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        # Add & Norm
        output = self.layernorm2(out1 + ffn_output)  # Residual connection

        return output

    def get_config(self):
        """
        Returns the configuration dictionary for serialization.

        Returns
        -------
        dict
            Configuration including `embed_dim`, `num_heads`, 
            `ffn_dim`, and `dropout_rate`.
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: dict):
        """
        Reconstructs the TransformerEncoderBlock from a config dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary with keys: `embed_dim`, `num_heads`,
            `ffn_dim`, `dropout_rate`.

        Returns
        -------
        TransformerEncoderBlock
            A new instance of the TransformerEncoderBlock.
        """
        return cls(**config)

@register_keras_serializable(
    "fusionlab.nn.components", 
    name="TransformerDecoderBlock"
    )
class TransformerDecoderBlock(Layer):
    """
    Transformer Decoder Block:
    Consists of masked multi-head self-attention, multi-head
    cross-attention (from encoder), and position-wise
    feed-forward network.

    Args:
    - embed_dim (int): The dimensionality of the embedding.
    - num_heads (int): The number of attention heads.
    - ffn_dim (int): The dimensionality of the feed-forward
      network.
    - dropout_rate (float): Dropout rate to be applied after each
      layer.
    """
    
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)    
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)

        # Masked Multi-Head Self-Attention
        self.mha1 = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim,
            dropout=dropout_rate
            )

        # Cross-Attention (Decoder attends to Encoder)
        self.mha2 = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim,
            dropout=dropout_rate
            )

        # Feed-Forward Network
        self.ffn = Sequential([
            Dense(ffn_dim, activation='relu'),
            Dense(embed_dim)
        ])

        # Layer Normalization and Dropout
        self.layernorm1 = LayerNormalization(
            epsilon=_EPSILON)
        self.layernorm2 = LayerNormalization(
            epsilon=_EPSILON)
        self.layernorm3 = LayerNormalization(
            epsilon=_EPSILON)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs: Tensor, enc_output: Tensor, 
             training: bool = False,
             look_ahead_mask: Optional[Tensor] = None,
             padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the decoder block.

        Args:
        - inputs: Tensor of shape (batch_size, seq_len, embed_dim) for decoder input
        - enc_output: Tensor of shape (batch_size, seq_len, embed_dim) from the encoder
        - training: Boolean flag to indicate if training mode is active
        - look_ahead_mask: Mask for the self-attention to prevent looking ahead
        - padding_mask: Mask for padding tokens in the encoder

        Returns:
        - output: Tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Masked Multi-Head Self-Attention
        attn1_output = self.mha1(inputs, inputs, inputs,
                                 attention_mask=look_ahead_mask,
                                 training=training)
        attn1_output = self.dropout1(attn1_output, training=training)

        # Add & Norm
        out1 = self.layernorm1(inputs + attn1_output)  # Residual connection

        # Cross-Attention (Decoder attends to Encoder)
        attn2_output = self.mha2(
            out1, enc_output, enc_output,
            attention_mask=padding_mask, training=training
            )
        attn2_output = self.dropout2(attn2_output, training=training)

        # Add & Norm
        out2 = self.layernorm2(out1 + attn2_output)  # Residual connection

        # Feed-Forward Network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)

        # Add & Norm
        output = self.layernorm3(out2 + ffn_output)  # Residual connection

        return output

    def get_config(self):
        """
        Returns the configuration dictionary for serialization.

        Returns
        -------
        dict
            Configuration including `embed_dim`, `num_heads`, 
            `ffn_dim`, and `dropout_rate`.
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: dict):
        """
        Reconstructs the TransformerDecoderBlock from a config dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary with keys: `embed_dim`, `num_heads`,
            `ffn_dim`, `dropout_rate`.

        Returns
        -------
        TransformerDecoderBlock
            A new instance of the TransformerDecoderBlock.
        """
        return cls(**config)

@register_keras_serializable(
    'fusionlab.nn.transformers', 
    name="TransformerEncoderLayer"
)
class TransformerEncoderLayer(Layer, NNLearner):
    """
    A single layer of the Transformer Encoder.

    Args:
        embed_dim (int): Dimensionality of the input and output.
        num_heads (int): Number of attention heads.
        ffn_dim (int): Hidden dimensionality of the feed-forward network.
        dropout_rate (float): Dropout rate.
        ffn_activation (str): Activation function for the FFN.
        layer_norm_epsilon (float): Epsilon for LayerNormalization.
    """
    
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ffn_dim: int, 
        dropout_rate: float = 0.1,
        ffn_activation: str = 'relu',
        layer_norm_epsilon: float = 1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate
        self.ffn_activation = ffn_activation
        self.layer_norm_epsilon = layer_norm_epsilon

        self.mha = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        self.ffn = Sequential([
            Dense(ffn_dim, activation=ffn_activation),
            Dense(embed_dim)
        ], name="encoder_ffn")
        self.layernorm1 = LayerNormalization(epsilon=layer_norm_epsilon)
        self.layernorm2 = LayerNormalization(epsilon=layer_norm_epsilon)
        self.dropout1 = Dropout(dropout_rate) # MHA output dropout is in MHA layer
        self.dropout_ffn = Dropout(dropout_rate)

    @tf_autograph.experimental.do_not_convert
    def call(
            self, x: Tensor, training: bool = False, 
            attention_mask: Optional[Tensor] = None) -> Tensor:
        attn_output = self.mha(
            query=x, value=x, key=x,
            attention_mask=attention_mask, training=training
        )
        # Dropout after MHA is already handled by MHA layer's dropout param.
        # self.dropout1 is if we want additional dropout on the residual sum.
        out1 = self.layernorm1(x + attn_output) # Post-norm

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output) # Post-norm
        return out2
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout_rate": self.dropout_rate,
            "ffn_activation": self.ffn_activation,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return config

@register_keras_serializable(
    'fusionlab.nn.transformers', name="TransformerDecoderLayer")
class TransformerDecoderLayer(Layer, NNLearner):
    """
    A single layer of the Transformer Decoder.
    (Arguments similar to TransformerEncoderLayer)
    """
    
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ffn_dim: int, 
        dropout_rate: float = 0.1,
        ffn_activation: str = 'relu',
        layer_norm_epsilon: float = 1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate
        self.ffn_activation = ffn_activation
        self.layer_norm_epsilon = layer_norm_epsilon

        self.mha1_self_attn = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, 
            dropout=dropout_rate
        )
        self.mha2_cross_attn = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim,
            dropout=dropout_rate
        )
        self.ffn = Sequential([
            Dense(ffn_dim, activation=ffn_activation),
            Dense(embed_dim)
        ], name="decoder_ffn")
        
        self.layernorm1 = LayerNormalization(epsilon=layer_norm_epsilon)
        self.layernorm2 = LayerNormalization(epsilon=layer_norm_epsilon)
        self.layernorm3 = LayerNormalization(epsilon=layer_norm_epsilon)
        
        # Dropout layers if needed beyond MHA's internal dropout
        self.dropout_ffn = Dropout(dropout_rate)

    @tf_autograph.experimental.do_not_convert
    def call(
        self, 
        x: Tensor, 
        enc_output: Tensor, 
        training: bool = False, 
        look_ahead_mask: Optional[Tensor] = None, 
        # For encoder output in cross-attention
        padding_mask: Optional[Tensor] = None, 
    ) -> Tensor:
        
        # Masked Multi-Head Self-Attention (for decoder inputs)
        attn1_output = self.mha1_self_attn(
            query=x, value=x, key=x, 
            attention_mask=look_ahead_mask, 
            training=training
        )
        out1 = self.layernorm1(x + attn1_output)

        # Multi-Head Cross-Attention (Query=Decoder, Key/Value=Encoder)
        attn2_output = self.mha2_cross_attn(
            query=out1, value=enc_output, key=enc_output,
            attention_mask=padding_mask, training=training
        )
        out2 = self.layernorm2(out1 + attn2_output) 

        # Feed-Forward Network
        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout_rate": self.dropout_rate,
            "ffn_activation": self.ffn_activation,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return config
 

@register_keras_serializable(
    'fusionlab.nn.components', 
    name="MultiDecoder"
 )
class MultiDecoder(Layer, NNLearner):
    r"""
    MultiDecoder for multi-horizon forecasting [1]_.

    This layer takes a single feature vector per example
    of shape :math:`(B, F)` and produces a separate
    output for each horizon step, resulting in
    :math:`(B, H, O)`.

    .. math::
        \mathbf{Y}_h = \text{Dense}_h(\mathbf{x}),\,
        h \in [1..H]

    Each horizon has its own decoder layer.

    Parameters
    ----------
    output_dim : int
        Number of output features for each horizon.
    num_horizons : int
        Number of forecast horizons.

    Notes
    -----
    This layer is particularly useful when you want
    separate parameters for each horizon, instead
    of a single shared head.

    Methods
    -------
    call(`x`, training=False)
        Forward pass that produces
        horizon-specific outputs.
    get_config()
        Returns configuration for serialization.
    from_config(`config`)
        Builds a new instance from config.

    Examples
    --------
    >>> from fusionlab.nn.components import MultiDecoder
    >>> import tensorflow as tf
    >>> # Input of shape (batch_size, feature_dim)
    >>> x = tf.random.normal((32, 128))
    >>> # Instantiate multi-horizon decoder
    >>> decoder = MultiDecoder(output_dim=1, num_horizons=3)
    >>> # Output shape => (32, 3, 1)
    >>> y = decoder(x)

    See Also
    --------
    MultiModalEmbedding
        Provides feature embeddings that can be
        fed into MultiDecoder.
    QuantileDistributionModeling
        Projects deterministic outputs into multiple
        quantiles per horizon.

    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2021). "Time-series
           forecasting with deep learning: a survey."
           *Philosophical Transactions of the Royal
           Society A*, 379(2194), 20200209.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, output_dim: int, num_horizons: int):
        r"""
        Initialize the MultiDecoder.

        Parameters
        ----------
        output_dim : int
            Number of features each horizon
            decoder should output.
        num_horizons : int
            Number of horizons to predict, each
            with its own Dense layer.
        """
        super().__init__()
        self.output_dim = output_dim
        self.num_horizons = num_horizons
        # Create a Dense decoder for each horizon
        self.decoders = [
            Dense(output_dim)
            for _ in range(num_horizons)
        ]

    @tf_autograph.experimental.do_not_convert
    def call(self, x, training=False):
        r"""
        Forward pass: each horizon has a separate
        Dense layer.

        Parameters
        ----------
        ``x`` : tf.Tensor
            A 2D tensor (B, F).
        training : bool, optional
            Unused in this layer. Defaults to
            ``False``.

        Returns
        -------
        tf.Tensor
            A 3D tensor of shape (B, H, O).
        """
        outputs = [
            decoder(x) for decoder in self.decoders
        ]
        return tf_stack(outputs, axis=1)

    def get_config(self):
        r"""
        Returns layer configuration for
        serialization.

        Returns
        -------
        dict
            Dictionary containing 'output_dim'
            and 'num_horizons'.
        """
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'num_horizons': self.num_horizons
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Create a new MultiDecoder from the config.

        Parameters
        ----------
        ``config`` : dict
            Contains 'output_dim', 'num_horizons'.

        Returns
        -------
        MultiDecoder
            A new instance.
        """
        return cls(**config)

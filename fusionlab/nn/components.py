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
from __future__ import annotations 
import warnings  
from numbers import Real, Integral  
from typing import Optional, Union, List, Dict, Tuple, Callable

import numpy as np 

from .._fusionlog import fusionlog 
from ..api.property import  NNLearner 
from ..core.checks import validate_nested_param
from ..core.handlers import param_deprecated_message
from ..compat.sklearn import validate_params, Interval, StrOptions
from ..utils.deps_utils import ensure_pkg

from . import KERAS_DEPS, KERAS_BACKEND, dependency_message
from ..compat.tf import standalone_keras

if KERAS_BACKEND:

    LSTM = KERAS_DEPS.LSTM
    LayerNormalization = KERAS_DEPS.LayerNormalization 
    TimeDistributed = KERAS_DEPS.TimeDistributed
    MultiHeadAttention = KERAS_DEPS.MultiHeadAttention
    Model = KERAS_DEPS.Model 
    BatchNormalization = KERAS_DEPS.BatchNormalization
    Input = KERAS_DEPS.Input
    Softmax = KERAS_DEPS.Softmax
    Flatten = KERAS_DEPS.Flatten
    Dropout = KERAS_DEPS.Dropout 
    Dense = KERAS_DEPS.Dense
    Embedding =KERAS_DEPS.Embedding 
    Concatenate=KERAS_DEPS.Concatenate 
    Layer = KERAS_DEPS.Layer 
    Loss=KERAS_DEPS.Loss
    register_keras_serializable=KERAS_DEPS.register_keras_serializable
    Tensor=KERAS_DEPS.Tensor
    Sequential =KERAS_DEPS.Sequential

    tf_Assert= KERAS_DEPS.Assert
    tf_TensorShape= KERAS_DEPS.TensorShape
    tf_concat = KERAS_DEPS.concat
    tf_shape = KERAS_DEPS.shape
    tf_reshape=KERAS_DEPS.reshape
    tf_repeat =KERAS_DEPS.repeat
    tf_add = KERAS_DEPS.add
    tf_cast=KERAS_DEPS.cast
    tf_maximum = KERAS_DEPS.maximum
    tf_reduce_mean = KERAS_DEPS.reduce_mean
    tf_add_n = KERAS_DEPS.add_n
    tf_float32=KERAS_DEPS.float32
    tf_constant=KERAS_DEPS.constant 
    tf_square=KERAS_DEPS.square 
    tf_transpose=KERAS_DEPS.transpose 
    tf_logical_and=KERAS_DEPS.logical_and 
    tf_logical_not = KERAS_DEPS.logical_not 
    tf_logical_or = KERAS_DEPS.logical_or
    tf_get_static_value =KERAS_DEPS.get_static_value
    tf_reduce_sum = KERAS_DEPS.reduce_sum
    tf_stack = KERAS_DEPS.stack
    tf_expand_dims = KERAS_DEPS.expand_dims
    tf_tile = KERAS_DEPS.tile
    tf_range=KERAS_DEPS.range 
    tf_rank=KERAS_DEPS.rank
    tf_split = KERAS_DEPS.split
    tf_multiply=KERAS_DEPS.multiply
    tf_cond=KERAS_DEPS.cond
    tf_constant =KERAS_DEPS.constant 
    tf_equal =KERAS_DEPS.equal 
    tf_int32=KERAS_DEPS.int32 
    tf_debugging =KERAS_DEPS.debugging 
    tf_autograph=KERAS_DEPS.autograph
    
    tf_newaxis = KERAS_DEPS.newaxis 
    tf_pow = KERAS_DEPS.pow
    tf_sin = KERAS_DEPS.sin
    tf_cos = KERAS_DEPS.cos
    tf_ones = KERAS_DEPS.ones 
    tf_linalg = KERAS_DEPS.linalg
    tf_floordiv = KERAS_DEPS.floordiv
    tf_greater =KERAS_DEPS.greater 
    

    try:
        # Equivalent to: from tensorflow.keras import activations
        activations = KERAS_DEPS.activations  
    except (ImportError, AttributeError) as e: 
        try: 
            activations = standalone_keras('activations')
        except: 
            raise ImportError (str(e))
    except: 
        raise ImportError(
                "Module 'activations' could not be"
                " imported from either tensorflow.keras"
                " or standalone keras. Ensure that TensorFlow "
                "or standalone Keras is installed and the"
                " module exists."
        )
    

_logger = fusionlog().get_fusionlab_logger(__name__)

DEP_MSG = dependency_message('components') 

__all__ = [
     'AdaptiveQuantileLoss',
     'AnomalyLoss',
     'CrossAttention',
     'DynamicTimeWindow',
     'ExplainableAttention',
     'GatedResidualNetwork',
     'HierarchicalAttention',
     'LearnedNormalization',
     'MemoryAugmentedAttention',
     'MultiDecoder',
     'MultiModalEmbedding',
     'MultiObjectiveLoss',
     'MultiResolutionAttentionFusion',
     'MultiScaleLSTM',
     'PositionalEncoding',
     'QuantileDistributionModeling',
     'StaticEnrichmentLayer',
     'TemporalAttentionLayer',
     'VariableSelectionNetwork',
     'Activation', 
     'TransformerEncoderLayer', 
     'TransformerDecoderLayer', 
     'PositionalEncodingTF', 
     'aggregate_multiscale', 
     'aggregate_time_window_output', 
     'create_causal_mask'
    ]


@register_keras_serializable(
    'fusionlab.nn.components', name="Activation"
  )
class Activation(Layer, NNLearner):
    r"""
    Flexible activation layer that transparently delegates to any
    built‑in or user‑defined activation function.

    Parameters
    ----------
    activation : str or Callable or None, default ``'relu'``
        Identifier of the desired activation.

        * If *str*, it must be recognised by
          :pymeth:`keras.activations.get`.
        * If *Callable*, it must follow the signature
          ``f(tensor) -> tensor``.
        * If *None*, the layer acts as the identity mapping
          :math:`f(x)=x`.

    **kwargs
        Additional keyword arguments forwarded to
        :class:`keras.layers.Layer` (e.g. ``name`` or ``dtype``).

    Notes
    -----
    Let :math:`\mathbf{x}\in\mathbb{R}^{n}` be the input tensor and
    :math:`\phi` the resolved activation function.  The layer performs

    .. math::

        \mathbf{y} = \phi(\mathbf{x}).

    Because :pyclass:`Activation` inherits from
    :class:`keras.layers.Layer`, it can be freely composed inside a
    ``tf.keras.Sequential`` or functional graph.

    Methods
    -------
    call(inputs, training=False)
        Apply the resolved activation to *inputs*.

    get_config()
        Return a JSON‑serialisable configuration dictionary.

    __repr__()
        Nicely formatted string representation—helpful in interactive
        sessions.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from fusionlab.nn.components import Activation
    >>> x  = tf.constant([‑2., 0., 1.5])
    >>> act = Activation('swish')
    >>> act(x).numpy()
    array([‑0.238, 0.   , 1.273], dtype=float32)

    Custom callable:

    >>> def leaky_relu(x, alpha=0.1):
    ...     return tf.where(x > 0, x, alpha * x)
    ...
    >>> act = Activation(leaky_relu)
    >>> act(x).numpy()
    array([‑0.2, 0. , 1.5], dtype=float32)

    See Also
    --------
    keras.activations.get
        Canonical resolver used under the hood.
    keras.layers.Activation
        Native Keras counterpart with fewer conveniences.

    References
    ----------
    .. [1] Ramachandran, Prajit, et al. *Searching for Activation
       Functions*. arXiv preprint arXiv:1710.05941 (2017).
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self,
                 activation: Union[str, Callable, None] = 'relu',
                 **kwargs):
        super().__init__(**kwargs)

        # Store original user input for debugging / introspection
        self.activation_original = activation

        # Resolve activation into (callable, canonical string)
        if activation is None:
            self.activation_fn  = activations.get(None)
            self.activation_str = 'linear'

        elif isinstance(activation, str):
            # Try to get a standard name via serialize,
            # fallback to object name
            try:
                self.activation_fn  = activations.get(activation)
                self.activation_str = activation
            except ValueError as err:
                raise ValueError(
                    f"Unknown activation '{activation}'."
                ) from err

        elif callable(activation):
            self.activation_fn = activation
            try:                                       # Try serialising
                ser = activations.serialize(activation)
                # Fallback if serialize doesn't give simple string
                self.activation_str = (
                    ser if isinstance(ser, str)
                    else getattr(activation, '__name__',
                                  activation.__class__.__name__)
                )
            except ValueError: 
                # Fallback if serialize doesn't give simple string
                self.activation_str = getattr(
                    activation, '__name__',
                    activation.__class__.__name__
                )
        else:
            raise TypeError(
                "Parameter 'activation' must be *str*, Callable, or "
                "*None*. Received type "
                f"{type(activation).__name__!r}."
            )

        if not callable(self.activation_fn):
            raise TypeError(
                f"Resolved activation '{self.activation_str}' is not "
                "callable."
            )

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training: bool = False):
        """
        Apply the stored activation to `inputs`.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of arbitrary shape.
        training : bool, default ``False``
            Present for API compatibility; ignored because most
            activations do not behave differently at training time.

        Returns
        -------
        tf.Tensor
            Tensor with identical shape to *inputs* but transformed
            element‑wise by the activation.
        """
        # A single line keeps Autograph happy 
        # and maximises performance
        return self.activation_fn(inputs)


    def get_config(self) -> dict:
        """
        Configuration dictionary for model serialization.

        Returns
        -------
        dict
            JSON‑friendly mapping that allows
            :pyfunc:`keras.layers.deserialize` to recreate the layer.
        """
        config = super().get_config()
        # Save the CANONICAL STRING NAME for serialization
        config.update({
            'activation': self.activation_str
        })
        return config

    # String representation
    def __repr__(self) -> str:                         # noqa: D401
        """
        Return *repr(self)*.

        The canonical activation string is included for clarity.
        """
        return (f"{self.__class__.__name__}("
                f"activation={self.activation_str!r})")
    
# -------------------- Pure Transformers components ------------------------------

@register_keras_serializable(
    'fusionlab.nn.transformers', 
    name="TransformerEncoderLayer"
)
class TransformerEncoderLayer(Layer):
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
class TransformerDecoderLayer(Layer):
    """
    A single layer of the Transformer Decoder.
    (Arguments similar to TransformerEncoderLayer)
    """
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
    'fusionlab.nn.components', name="PositionalEncodingTF")
class PositionalEncodingTF(Layer):
    """
    Standard Transformer Positional Encoding using sine and cosine functions.
    Adds positional information to input embeddings.

    Args:
        max_position (int): Maximum sequence length that this layer can handle.
        embed_dim (int): The dimensionality of the embeddings (and the
                         positional encoding).
    """
    def __init__(self, max_position: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.max_position = max_position
        self.embed_dim = embed_dim
        # self.pos_encoding is created once and stored.
        self.pos_encoding = self._build_positional_encoding(max_position, embed_dim)

    def _build_positional_encoding(
            self, position: int, d_model: int) -> Tensor:
        """Builds the positional encoding matrix using NumPy 
        then converts to Tensor."""
    
        # 1. Calculate angles in NumPy
        # 'pos' is for positions (sequence length), 'i' is for dimension
        pos_np = np.arange(position)[:, np.newaxis]
        i_np = np.arange(d_model)[np.newaxis, :]
        
        angle_rates_np = 1 / np.power(
            10000, (2 * (i_np // 2)) / np.float32(d_model)
        )
        angle_rads_np = pos_np * angle_rates_np
    
        # 2. Apply sin to even indices in the array; 2i
        angle_rads_np[:, 0::2] = np.sin(angle_rads_np[:, 0::2])
    
        # 3. Apply cos to odd indices in the array; 2i+1
        angle_rads_np[:, 1::2] = np.cos(angle_rads_np[:, 1::2])
    
        # 4. Add a new axis for batch dimension and cast to TensorFlow tensor
        # The self.pos_encoding expects (1, max_position, embed_dim)
        pos_encoding_tensor = tf_cast(
            angle_rads_np[np.newaxis, ...], dtype=tf_float32
        )
        
        return pos_encoding_tensor

    def _tf_build_positional_encoding(self, position, d_model):
        """Builds the positional encoding matrix."""
        angle_rads = self._get_angles(
            # Use np.arange for non-Tensor context 
            # if KERAS_DEPS.arange isn't suitable
            tf_range(position)[:, tf_newaxis],
            tf_range(d_model)[tf_newaxis, :],
            d_model
            )
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = tf_sin(angle_rads[:, 0::2])
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = tf_cos(angle_rads[:, 1::2])

        pos_encoding_np = angle_rads[tf_newaxis, ...]
        
        return tf_cast(pos_encoding_np, dtype=tf_float32)
        
    def _get_angles(self, pos, i, d_model):
        """Calculates the angle rates for positional encoding."""
        # Use np.power for non-Tensor context
        angle_rates = 1 / np.power(10000, (
            2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    def _tf_get_angles(self, pos, i, d_model):
        """Calculates the angle rates for positional encoding."""
        # cast d_model to float32
        d_model_f = tf_cast(d_model, tf_float32)
        # compute floor(i/2) as an integer tensor
        half_i = tf_floordiv(i, 2)
        # build the numerator 2 * (i//2), then cast to float32
        numer = tf_cast(2 * half_i, tf_float32)
        # now both numer and d_model_f are float32
        exponent = numer / d_model_f
        # compute the rates with float constants
        angle_rates = 1.0 / tf_pow(10000.0, exponent)
        # and finally apply to pos (cast pos to float32 if needed)
        return tf_cast(pos, tf_float32) * angle_rates

    def call(self, x):
        """Adds positional encoding to the input tensor `x`."""
        if not KERAS_BACKEND:
            raise RuntimeError(
                "PositionalEncodingTF layer requires "
                "a Keras backend (TensorFlow)."
            )
        input_seq_len = tf_shape(x)[1]
        # Add positional encoding up to the length of the input sequence.
        return x + self.pos_encoding[:, :input_seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_position": self.max_position,
            "embed_dim": self.embed_dim,
        })
        return config


# -------------------- TFT components ----------------------------------------


@register_keras_serializable('fusionlab.nn.components', name='PositionalEncoding')
class PositionalEncoding(Layer, NNLearner):
    r"""
    Positional Encoding layer that incorporates temporal 
    positions into an input sequence by adding positional 
    information to each time step. This helps models, 
    especially those based on attention mechanisms, to 
    capture the order of time steps [1]_.

    .. math::
        \mathbf{Z} = \mathbf{X} + \text{PositionEncoding}

    where :math:`\mathbf{X}` is the original input and 
    :math:`\mathbf{Z}` is the output with positional 
    encodings added.

    Parameters
    ----------
    None 
        This layer does not define additional 
        constructor parameters beyond the standard 
        Keras ``Layer``.

    Notes
    -----
    - This class adds a positional index to each feature 
      across time steps, effectively encoding the temporal 
      position.
    - Because attention-based models do not inherently 
      encode sequence ordering, positional encoding 
      is crucial for sequence awareness.

    Methods
    -------
    call(`inputs`)
        Perform the forward pass, adding positional 
        encoding to the input tensor.

    get_config()
        Return the configuration of this layer for 
        serialization.

    Examples
    --------
    >>> from fusionlab.nn.components import PositionalEncoding
    >>> import tensorflow as tf
    >>> # Create random input of shape
    ... # (batch_size, time_steps, feature_dim)
    >>> inputs = tf.random.normal((32, 10, 64))
    >>> # Instantiate the positional encoding layer
    >>> pe = PositionalEncoding()
    >>> # Forward pass
    >>> outputs = pe(inputs)

    See Also
    --------
    TemporalFusionTransformer 
        Combines positional encoding in dynamic 
        features for time series.

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N., 
           Uszkoreit, J., Jones, L., Gomez, A. N., 
           Kaiser, Ł., & Polosukhin, I. (2017). 
           "Attention is all you need." In *Advances 
           in Neural Information Processing Systems* 
           (pp. 5998-6008).
    """
    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        r"""
        Forward pass that adds positional encoding to 
        ``inputs``.

        Parameters
        ----------
        inputs : tf.Tensor
            A 3D tensor of shape 
            :math:`(B, T, D)`, where ``B`` is 
            batch size, ``T`` is time steps, and 
            ``D`` is feature dimension.
        training : bool, optional
            Boolean flag indicating whether the layer is 
            in training mode.
            Not used in this layer but included for 
            Keras API compatibility.
        
        Returns
        -------
        tf.Tensor
            A 3D tensor of the same shape 
            :math:`(B, T, D)`, where each time step 
            has been augmented with its position index.

        Notes
        -----
        1. Construct position indices
           :math:`p = [0, 1, 2, \dots, T - 1]`.
        2. Tile and broadcast across features.
        3. Add positional index to inputs.
        """
        # Extract shapes dynamically
        batch_size = tf_shape(inputs)[0]
        seq_len = tf_shape(inputs)[1]
        feature_dim = tf_shape(inputs)[2]

        # Create position indices
        position_indices = tf_range(
            0,
            seq_len,
            dtype='float32'
        )
        position_indices = tf_expand_dims(
            position_indices,
            axis=0
        )
        position_indices = tf_expand_dims(
            position_indices,
            axis=-1
        )

        # Tile to match input shape
        position_encoding = tf_tile(
            position_indices,
            [batch_size, 1, feature_dim]
        )

        # Return input plus positional encoding
        return inputs + position_encoding

    def get_config(self):
        r"""
        Return the configuration of this layer
        for serialization.

        Returns
        -------
        dict
            Dictionary of layer configuration.
        """
        config = super().get_config().copy()
        return config


@register_keras_serializable(
    'fusionlab.nn.components', name="GatedResidualNetwork"
)
@param_deprecated_message(
    conditions_params_mappings=[
        {
            'param': 'use_time_distributed',
            'condition': lambda v: v is not None and v is not False,
            'message': (
                "The 'use_time_distributed' parameter in GatedResidualNetwork "
                "is deprecated and has no effect.\n"
                "The layer automatically handles time dimensions based on "
                "input rank.\n"
                "If using within VariableSelectionNetwork, control time "
                "distribution via the VSN's own 'use_time_distributed' parameter."
            ),
        }
    ],
    warning_category=DeprecationWarning 
)
class GatedResidualNetwork(Layer):
    """Gated Residual Network applying transformations with optional context."""

    _COMMON_ACTIVATIONS = {
        "relu", "tanh", "sigmoid", "elu", "selu", "gelu", "linear", None
    }

    @validate_params({
        "units": [Interval(Integral, 0, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "use_batch_norm": [bool],
        "activation": [StrOptions(_COMMON_ACTIVATIONS)],
        "output_activation": [StrOptions(_COMMON_ACTIVATIONS), None],
        "use_time_distributed": [bool, None],
    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        units: int,
        dropout_rate: float = 0.0,
        activation: str = 'elu',
        output_activation: Optional[str] = None,
        use_batch_norm: bool = False,
        use_time_distributed: Optional[bool] = None, 
        **kwargs
    ):
        """Initializes the GatedResidualNetwork layer."""
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation_str = activation
        self.output_activation_str = output_activation
        # The use_time_distributed parameter is stored only to allow
        # the decorator to check its value. It is NOT used in the
        # layer's logic anymore.
        self._deprecated_use_td = use_time_distributed 

        # --- Convert activation strings to callable functions ---
        try:
            self.activation_fn = activations.get(activation)
            self.output_activation_fn = activations.get(output_activation) \
                if output_activation is not None else None
        except Exception as e:
             # Catch potential errors during activation lookup
             raise ValueError(
                 f"Failed to get activation function '{activation}' or "
                 f"'{output_activation}'. Error: {e}"
                 ) from e

        # --- Define Internal Layers ---
        # Dense layer processing input (x + optional context)
        # Activation is applied *after* this layer manually
        self.input_dense = Dense(self.units, activation=None,
                                 name="input_dense")

        # Dense layer projecting context (if provided)
        # No bias as per original paper often; no activation needed here
        self.context_dense = Dense(
            self.units, use_bias=False,
            name="context_dense"
        )

        # Optional Batch Normalization (applied after main activation)
        self.batch_norm = BatchNormalization(
            name="batch_norm"
            ) if self.use_batch_norm else None

        # Dropout Layer (applied after activation/norm)
        self.dropout = Dropout(
            self.dropout_rate, name="grn_dropout"
            )

        # Dense layer for main transformation path (after dropout)
        self.output_dense = Dense(
            self.units, activation=None,
            name="output_dense"
        )

        # Dense layer for gating mechanism applied to input projection
        self.gate_dense = Dense(
            self.units, activation='sigmoid',
            name="gate_dense"
        )

        # Final Layer Normalization (standard in GRN)
        self.layer_norm = LayerNormalization(
            name="output_layer_norm"
        )

        # Projection layer for residual 
        # connection (created in build)
        self.projection = None

    def build(self, input_shape):
        """Builds the residual projection layer if needed."""
        # Use TensorShape object directly if available
        if not isinstance(input_shape, tf_TensorShape):
            # Attempt conversion, handles tuples, lists, TensorShape
            try:
                input_shape = tf_TensorShape(input_shape)
            except TypeError:
                 raise ValueError(
                    f"Could not convert input_shape to TensorShape:"
                    f" {input_shape}"
                    )

        # Check rank using the TensorShape object property
        input_rank = input_shape.rank # This returns None if rank is unknown

        # Check minimum rank requirement only if rank is known
        if input_rank is not None and input_rank < 2:
            raise ValueError(
                "Input shape must have at least 2 dimensions "
                f"(Batch, Features). Received rank: {input_rank}"
                f", shape: {input_shape}"
            )

        input_dim = None
        # Only try to get last dimension if rank is known
        if input_rank is not None:
            input_dim = input_shape[-1]
            # Further check if last dimension itself is known (is an integer)
            if not isinstance(input_dim, int) or input_dim <= 0 :
                 # Last dimension is unknown or invalid
                 warnings.warn(
                     f"Input shape {input_shape} has unknown or invalid "
                     "last dimension in GRN build. Cannot check "
                     "if projection layer is needed.", RuntimeWarning
                     )
                 input_dim = None # Treat as unknown if not valid int

        # Create projection layer only if dimensions are known and differ
        if (input_dim is not None) and (input_dim != self.units):
            if self.projection is None: # Avoid recreating
                self.projection = Dense(self.units, name="residual_projection")
                # Build projection layer using the full input shape object
                self.projection.build(input_shape)
                # Comment: Residual projection created and built.
        elif input_dim == self.units:
             # Set projection to None explicitly if dims match
             self.projection = None

        # context_dense builds lazily on first call
        # Call the build method of the parent class
        super().build(input_shape)
        
    def call(self, x, context=None, training=False):
        """Forward pass implementing GRN with optional context."""
        # Input x shape (B, ..., F_in)
        # Context shape (if provided) (B, ..., Units) after projection
        """Forward pass implementing GRN with optional context."""
        _logger.debug(
            f"DEBUG_GRN: Entering call. x shape: {tf_shape(x)},"
            f" context provided: {context is not None}") # DEBUG
        # --- 1. Residual Connection Setup ---
        shortcut = x
        if self.projection is not None:
            _logger.debug("DEBUG_GRN: Applying projection.") # DEBUG
            shortcut = self.projection(shortcut) # Shape (B, ..., Units)

        # --- 2. Process Input and Context ---
        # Project input features to 'units' dimension
        _logger.debug(
            f"DEBUG_GRN: Applying input_dense to x shape: {tf_shape(x)}") # DEBUG
        projected_input = self.input_dense(x) # Shape (B, ..., Units)
        input_plus_context = projected_input # No context added; Default 
        
        # Add processed context if provided
        if context is not None:
            _logger.debug("DEBUG_GRN: Applying context_dense"
                  f" to context shape: {tf_shape(context)}") # DEBUG
            context_proj = self.context_dense(context) # Shape (B, ..., Units)

            # Ensure context can be added (handle broadcasting)
            # x_rank = tf_rank(projected_input)
            
             # Use standard Python len() on shapes now,
            # Use standard Python len() on shapes now, 
            x_rank = len(projected_input.shape)
            ctx_rank = len(context_proj.shape)
            
            # x_rank = projected_input.shape.rank 
            # #ctx_rank = tf_rank(context_proj)
            # ctx_rank = context_proj.shape.rank 
            _logger.debug(
                f"DEBUG_GRN: x_rank={x_rank}, ctx_rank={ctx_rank}") # DEBUG
            if x_rank == 3 and ctx_rank == 2:# e.g., x=(B,T,U), ctx=(B,U)
                # Add time dimension for broadcasting: (B,U) -> (B,1,U)
                context_proj_expanded = tf_expand_dims(context_proj, axis=1)
                # Now shapes should be broadcast-compatible
                _logger.debug("DEBUG_GRN: Adding context.") # DEBUG
                input_plus_context = tf_add(projected_input, context_proj_expanded)
            elif x_rank == ctx_rank:
                # Ranks match, add directly
                _logger.debug(
                    "DEBUG_GRN: Ranks match,  Adding context directly."
                    ) # DEBUG
                input_plus_context = tf_add(projected_input, context_proj)
                
            else:
                # Raise error for incompatible ranks
                raise ValueError(
                    f"Incompatible ranks GRN input ({x_rank})"
                    f" and context ({ctx_rank}). Cannot broadcast/add."
                )

        # --- 3. Apply Activation and Regularization ---
        _logger.debug("Applying activation_fn.") # DEBUG
        activated_features = self.activation_fn(input_plus_context)
        if self.batch_norm is not None:
            # Apply BN after activation
            activated_features = self.batch_norm(activated_features,
                                                 training=training)
        _logger.debug("Applying dropout.") # DEBUG
        regularized_features = self.dropout(activated_features,
                                            training=training)

        # --- 4. Main Transformation Path ---
        _logger.debug("Applying output_dense.") # DEBUG
        transformed_output = self.output_dense(regularized_features)

        # --- 5. Gating Path ---
        _logger.debug("Applying gate_dense.") # DEBUG
        # Gate depends on input+context projection *before* main activation
        gate_values = self.gate_dense(input_plus_context)

        # --- 6. Apply Gate ---
        _logger.debug("Applying gate multiplication.") # DEBUG
        gated_output = tf_multiply(transformed_output, gate_values)

        # --- 7. Add Residual ---
        _logger.debug("Adding residual connection.") # DEBUG
        residual_output = tf_add(shortcut, gated_output)

        # --- 8. Final Normalization & Optional Activation ---
        _logger.debug("Applying layer_norm.") # DEBUG
        normalized_output = self.layer_norm(residual_output)
        final_output = normalized_output
        if self.output_activation_fn is not None:
            _logger.debug("Applying output_activation_fn.") # DEBUG
            final_output = self.output_activation_fn(normalized_output)
            #  Applied final output activation.
        _logger.debug("Exiting call successfully.") # DEBUG
        return final_output
    
    def get_config(self):
        """Returns the layer configuration."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            # 'use_time_distributed' removed from config
            'activation': self.activation_str, # Use original string
            'output_activation': self.output_activation_str, # Use original string
            'use_batch_norm': self.use_batch_norm,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates layer from its config."""
        return cls(**config)

@register_keras_serializable(
    'fusionlab.nn.components',
    name="VariableSelectionNetwork"
)
class VariableSelectionNetwork(Layer, NNLearner): 
    """Applies GRN to each variable and learns importance weights."""

    @validate_params({
        "num_inputs": [Interval(Integral, 0, None, closed='left')],
        "units": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "use_time_distributed": [bool],
        "use_batch_norm": [bool],
        "activation": [StrOptions(
            {"elu", "relu", "tanh", "sigmoid", "linear", "gelu", None}
            )]
    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        num_inputs: int,
        units: int,
        dropout_rate: float = 0.0,
        use_time_distributed: bool = False,
        activation: str = 'elu',
        use_batch_norm: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_inputs = num_inputs
        self.units = units
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed
        self.use_batch_norm = use_batch_norm
        
        # Store original activation string for config
        _Activation = Activation(activation) 
        self.activation_str = _Activation.activation_str 
        self.activation_fn = _Activation.activation_fn 

        # --- Layers ---
        # 1. GRN for each individual input variable
        #    GRN's __init__ should handle converting activation string
        self.single_variable_grns = [
            GatedResidualNetwork(
                units=units, dropout_rate=dropout_rate,
                activation=self.activation_str, # Pass string
                use_batch_norm=use_batch_norm,
                name=f"single_var_grn_{i}"
            ) for i in range(num_inputs)
        ]

        # 2. Dense layer to compute variable importances (applied later)
        #    Output units = 1 per variable for the original weighting method
        self.variable_importance_dense = Dense(
            1, name="variable_importance_dense"
        )

        # 3. Softmax for normalizing weights across variables (N dimension)
        #    Axis -2 assumes stacked_outputs shape (B, [T,] N, units)
        self.softmax = Softmax(axis=-2, name="variable_weights_softmax")

        # 4. Optional context projection layer (created in build)
        #    Projects external context to 'units' for GRNs
        self.context_projection = None

        # Attribute to store weights
        self.variable_importances_ = None

    @tf_autograph.experimental.do_not_convert
    def build(self, input_shape):
        """Builds internal GRNs and projection layers 
        with explicit shapes."""
        # Use TensorShape object for robust handling
        if not isinstance(input_shape, tf_TensorShape):
            input_shape = tf_TensorShape(input_shape)

        input_rank = input_shape.rank
        expected_min_rank = 3 if self.use_time_distributed else 2

        # Check if rank is known and sufficient
        if input_rank is None or input_rank < expected_min_rank:
            # If rank unknown or too low at build time,
            # we cannot proceed reliably.
            # This indicates an issue upstream or 
            # requires dynamic shapes throughout.
            raise ValueError(
                f"VSN build requires input rank >= {expected_min_rank}"
                f" with known rank. Received shape: {input_shape}"
            )

        # Determine shape of input slices passed to single_variable_grns
        # Add feature dim F=1 if missing
        # Add feature dimension if missing
        inferred_input_shape = tf_cond(
             tf_equal(input_rank, expected_min_rank),
             lambda: input_shape.as_list() + [1],
             lambda: input_shape.as_list()
         )
        # Shape: (B, N, F=1) or (B, T, N, F=1)

        # Ensure dimensions (except batch) are 
        # known for building sub-layers
        if any(d is None for d in inferred_input_shape[1:]):
             # This should ideally not happen if 
             # input comes from previous layers
             # but handle defensively.
             raise ValueError(
                 f"VSN build received unknown non-batch dimensions in shape "
                 f"{inferred_input_shape}. Cannot reliably build sub-layers."
             )

        # Calculate the expected shape for a single variable slice
        if self.use_time_distributed:
            # Input (B, T, N, F) -> Slice is (B, T, F)
            single_var_input_shape = tf_TensorShape(
                [inferred_input_shape[0], # Batch (can be None)
                 inferred_input_shape[1], # Time (should be known)
                 inferred_input_shape[3]] # Features (should be known)
                )
        else:
            # Input (B, N, F) -> Slice is (B, F)
             single_var_input_shape = tf_TensorShape(
                 [inferred_input_shape[0], # Batch (can be None)
                  inferred_input_shape[2]] # Features (should be known)
                 )

        # --- Explicitly build each single_variable_grn ---
        # Use the calculated slice shape
        for grn in self.single_variable_grns:
            if not grn.built:
                try:
                    grn.build(single_var_input_shape)
                    # Comment: Built internal GRN with calculated shape.
                except Exception as e:
                     # Add more context if GRN build fails
                     raise RuntimeError(
                         f"Failed to build internal GRN {grn.name} with shape "
                         f"{single_var_input_shape} derived from VSN input "
                         f"{input_shape}. Original error: {e}"
                         ) from e

        # Build context projection layer lazily (or here if context shape known)
        if self.context_projection is None:
             self.context_projection = Dense(
                  self.units, name="context_projection",
                  # Pass string, Dense handles activation resolution
                  activation=self.activation_str
                  )
             # Let Keras build context_projection on first call with context

        # Build other internal layers like weighting_grn if needed here
        super().build(input_shape) # Call parent build last
        

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, context=None, training=False):
        """Execute the forward pass with optional context."""
        _logger.debug(f"VSN '{self.name}': Entering call method.")
        _logger.debug(
            f"  Initial input shape: {getattr(inputs, 'shape', 'N/A')}")
        _logger.debug(f"  Context provided: {context is not None}")
        _logger.debug(f"  Training mode: {training}")

        # --- Input Validation and Reshaping ---
        # Use Python len() on shape - works reliably with decorator
        try:
            actual_rank = len(inputs.shape)
        except Exception as e:
             _logger.error(f"VSN '{self.name}': Failed to get input rank."
                          f" Input type: {type(inputs)}. Error: {e}")
             raise TypeError(f"Could not determine rank of input with shape"
                             f" {getattr(inputs, 'shape', 'N/A')}") from e

        expected_min_rank = 3 if self.use_time_distributed else 2
        _logger.debug(f"  Input rank: actual={actual_rank}, expected_min="
                     f"{expected_min_rank}")

        if actual_rank < expected_min_rank:
            # Raise error if rank is insufficient
            raise ValueError(
                f"VSN '{self.name}': Input rank must be >= "
                f"{expected_min_rank}. Got rank {actual_rank} for "
                f"shape {inputs.shape}."
            )

        # Add feature dimension if missing (e.g., B,N -> B,N,1 or B,T,N -> B,T,N,1)
        if actual_rank == expected_min_rank:
            _logger.debug(
                f"  Input rank matches minimum expected ({actual_rank})."
                " Expanding feature dimension."
                )
            inputs = tf_expand_dims(inputs, axis=-1)
            _logger.debug(
                f"  Input shape after expansion: {inputs.shape}"
                )
        # Input shape is now (B, N, F) or (B, T, N, F)

        # --- Context Processing ---
        processed_context = None
        if context is not None:
            _logger.debug(
                f"  Processing provided context. Shape: {context.shape}"
                )
            # Ensure context projection layer is created (lazily if needed)
            if self.context_projection is None:
                 _logger.warning(
                     f"VSN '{self.name}': Context projection layer"
                     " not built in build method. Building lazily."
                )
                 self.context_projection = Dense(
                      self.units, name="context_projection",
                      activation=self.activation_str # Use string
                      )
            processed_context = self.context_projection(context)
            _logger.debug(
                f"  Processed context shape: {processed_context.shape}")
            # Note: GRN's call method handles broadcasting this context
        else:
            _logger.debug("  No context provided.")

        # --- Apply GRN to each variable ---
        var_outputs = []
        _logger.debug(
            f"  Applying single_variable_grns to {self.num_inputs}"
            " inputs..."
            )
        # Python loop - should execute as Python code due to decorator
        for i in range(self.num_inputs):
            _logger.debug(
                f"    Processing variable index {i}")
            # Slice input for the i-th variable
            if self.use_time_distributed:
                # Slice variable i: (B, T, N, F) -> (B, T, F)
                var_input = inputs[:, :, i, :]
                _logger.debug(
                    "      Sliced var_input shape (TD):"
                    f" {var_input.shape}")
            else:
                # Slice variable i: (B, N, F) -> (B, F)
                var_input = inputs[:, i, :]
                _logger.debug(
                    "      Sliced var_input shape (non-TD):"
                    f" {var_input.shape}")

            # Apply the i-th GRN, passing the (potentially None) context
            # GRN's call method should also have @do_not_convert if needed
            grn_output = self.single_variable_grns[i](
                var_input,
                context=processed_context, # Pass processed context
                training=training
            )
            var_outputs.append(grn_output)
            _logger.debug(
                "      GRN output shape for var {i}:"
                f" {grn_output.shape}")
            # Output shape: (B, T, units) or (B, units)

        # --- Stack GRN outputs along variable dimension (N) ---
        # axis=-2 places N before the 'units' dimension
        stacked_outputs = tf_stack(var_outputs, axis=-2)
        _logger.debug(
            f"  Stacked GRN outputs shape: {stacked_outputs.shape}")
        # Shape: (B, T, N, units) or (B, N, units)

        # --- Calculate Variable Importance Weights (Original Simple Logic) ---
        # 1. Apply Dense layer (output units = 1) to stacked outputs
        #    Acts on the last dimension ('units')
        _logger.debug("  Calculating importance logits...")
        importance_logits = self.variable_importance_dense(stacked_outputs)
        _logger.debug(
            f"  Importance logits shape: {importance_logits.shape}"
            )
        # Shape: (B, [T,] N, 1)

        # 2. Apply Softmax across the variable dimension (N, axis=-2)
        _logger.debug("  Calculating importance weights (softmax)...")
        weights = self.softmax(importance_logits)
        _logger.debug(f"  Importance weights shape: {weights.shape}")
        # Shape: (B, [T,] N, 1)
        self.variable_importances_ = weights # Store weights

        # --- Weighted Combination ---
        # Multiply stacked GRN outputs by weights and sum across N
        _logger.debug("  Performing weighted sum...")
        weighted_sum = tf_reduce_sum(
            tf_multiply(stacked_outputs, weights),
            axis=-2 # Sum across the variable dimension (N)
        )
        _logger.debug(
            f"  Final weighted sum output shape: {weighted_sum.shape}"
            )
        # Final output shape: (B, T, units) or (B, units)

        _logger.debug(f"VSN '{self.name}': Exiting call method.")
        
        return weighted_sum
    
    def get_config(self):
        """Returns the layer configuration."""
        config = super().get_config()
        config.update({
            'num_inputs': self.num_inputs,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'use_time_distributed': self.use_time_distributed,
            'activation': self.activation_str, 
            'use_batch_norm': self.use_batch_norm,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates layer from its config."""
        return cls(**config)
        

@register_keras_serializable(
    'fusionlab.nn.components',
    name="TemporalAttentionLayer"
)
class TemporalAttentionLayer(Layer):
    """Temporal Attention Layer conditioning query with context."""

    @validate_params({
         "units": [Interval(Integral, 0, None, 
                            closed='left')],
         "num_heads": [Interval(Integral, 0, None,
                                closed='left')],
         "dropout_rate": [Interval(Real, 0, 1,
                                   closed="both")],
         "use_batch_norm": [bool],
     })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        units: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        activation: str = 'elu',
        use_batch_norm: bool = False,
        **kwargs
    ):
        """Initializes the TemporalAttentionLayer."""
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation_str = Activation(activation).activation_str 

        # --- Define Internal Layers ---
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units,
            dropout=dropout_rate,
            name="mha"
        )
        self.dropout = Dropout(dropout_rate, name="attn_dropout")
        self.layer_norm1 = LayerNormalization(name="layer_norm_1")

        # GRN to process the input context_vector
        # Ensure this is a single instance, passing the activation string
        self.context_grn = GatedResidualNetwork(
            units=units, # Output matches main path 'units'
            dropout_rate=dropout_rate,
            activation=self.activation_str,
            use_batch_norm=self.use_batch_norm,
            name="context_grn"
            # Note: GRN's internal activation handling should be fixed
        )

        # Final GRN (position-wise feedforward)
        # Ensure this is also a single instance
        self.output_grn = GatedResidualNetwork(
            units=units,
            dropout_rate=dropout_rate,
            activation=self.activation_str,
            use_batch_norm=self.use_batch_norm,
            name="output_grn"
        )
        
    def build(self, input_shape):
        """Builds internal layers, especially GRNs."""
        # input_shape corresponds to the main 'inputs' tensor (B, T, U)
        if not isinstance(input_shape, (list, tuple)):
             # If only main input shape is passed (common)
             main_input_shape = tuple(input_shape)
        elif len(input_shape) == 2: 
            #  [inputs_shape, context_shape] rarelly happended
             main_input_shape = tuple(input_shape[0])
             # Optionally build context_grn if context_shape is known
             context_shape = tuple(input_shape[1])
             if not self.context_grn.built:
                  self.context_grn.build(context_shape)
        else:
             raise ValueError(
                 "Unexpected input_shape format for build.")
 
        if len(main_input_shape) < 3:
            raise ValueError(
                "TemporalAttentionLayer expects input rank >= 3")

        # Define expected input shape for output_grn
        # It receives output from layer_norm1, which has same shape as input
        output_grn_input_shape = main_input_shape

        # Explicitly build the output GRN if not already built
        if not self.output_grn.built:
            self.output_grn.build(output_grn_input_shape)
            # Developer comment: Explicitly built output_grn.

        # Build context_grn lazily during call or here
        # Call the parent build method AFTER building sub-layers
        super().build(input_shape)
        # Developer comment: Layer built status should now be True.

    def call(self, inputs, context_vector=None, training=False):
        """Forward pass of the temporal attention layer."""
        # Input shapes: inputs=(B, T, U), context_vector=(B, U_ctx)

        query = inputs # Default query
        processed_context = None

        # --- Process Context Vector (if provided) ---
        if context_vector is not None:
            # Pass context_vector as the main input 'x' to context_grn
            processed_context = self.context_grn(
                x=context_vector,
                context=None, # No nested context for the context_grn itself
                training=training
            )
            # Output shape: (B, units)

            # Expand context across time: (B, units) -> (B, 1, units)
            context_expanded = tf_expand_dims(processed_context, axis=1)
            # Add to inputs (broadcasting handles time dimension)
            query = tf_add(inputs, context_expanded)
            # Comment: Query now incorporates static context.

        # --- Multi-Head Self-Attention ---
        attn_output = self.multi_head_attention(
            query=query, value=inputs, key=inputs, training=training
        ) # Shape: (B, T, units)

        # --- Add & Norm (First Residual Connection) ---
        attn_output_dropout = self.dropout(attn_output, training=training)
        # Residual connection uses original 'inputs'
        x_attn = self.layer_norm1(tf_add(inputs, attn_output_dropout))
        # Shape: (B, T, units)

        # --- Position-wise Feedforward (Final GRN) ---
        # This GRN takes the output of the attention block as input 'x'
        # It does not receive the external 'context_vector' here.
        # --- DEBUG lines ---
        _logger.debug("\nDEBUG>> About to call self.output_grn")
        _logger.debug(
            "DEBUG>> Type of self.output_grn:"
            f" {type(self.output_grn)}")
        _logger.debug(
            "DEBUG>> Is self.output_grn callable:"
            f" {callable(self.output_grn)}")
        try:
            # Try accessing an attribute expected on a Keras layer
            _logger.debug(
                "DEBUG>> self.output_grn name:"
                f" {self.output_grn.name}")
            _logger.debug(
                "DEBUG>> self.output_grn built status:"
                f" {self.output_grn.built}")
        except AttributeError as ae:
             _logger.debug(
                 "DEBUG>> Failed to access attributes"
                 f" of self.output_grn: {ae}")
        _logger.debug(
            f"DEBUG>> Input x_attn shape: {tf_shape(x_attn)}\n")
        
        # --- End DEBUG lines ---
        output = self.output_grn(
            x=x_attn,
            context=None, # No external context for the final GRN
            training=training
        )
        # Shape: (B, T, units)
        return output

    def get_config(self):
        """Returns the layer configuration."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_str, 
            'use_batch_norm': self.use_batch_norm,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates layer from its config."""
        return cls(**config)

@register_keras_serializable(
    'fusionlab.nn.components', 
    name="StaticEnrichmentLayer"
 )
class StaticEnrichmentLayer(Layer, NNLearner):
    r"""
    Static Enrichment Layer for combining static
    and temporal features [1]_.

    This layer enriches temporal features with static
    context, enabling the model to modulate temporal
    dynamics based on static information. It concatenates
    a tiled static context vector to temporal features
    and processes them through a
    :class:`GatedResidualNetwork`, yielding an
    enriched feature map that combines both static and
    temporal information.

    .. math::
        \mathbf{Z} = \text{GRN}\big([\mathbf{C}, 
        \mathbf{X}]\big)

    where :math:`\mathbf{C}` is a static context vector
    tiled over the time dimension, and :math:`\mathbf{X}`
    are the temporal features.

    Parameters
    ----------
    units : int
        Number of hidden units within the
        internally used `GatedResidualNetwork`.
    activation : str, optional
        Activation function used in the
        GRN. Must be one of 
        {'elu', 'relu', 'tanh', 'sigmoid', 'linear'}.
        Defaults to ``'elu'``.
    use_batch_norm : bool, optional
        Whether to apply batch normalization
        within the GRN. Defaults to ``False``.
    **kwargs :
        Additional arguments passed to
        the parent Keras ``Layer``.

    Notes
    -----
    This layer performs the following:
    1. Expand static context from shape
       :math:`(B, U)` to :math:`(B, T, U)`.
    2. Concatenate with temporal features 
       :math:`(B, T, D)` along the last dimension.
    3. Pass the combined tensor through a 
       `GatedResidualNetwork`.

    Methods
    -------
    call(`static_context_vector`, `temporal_features`,
         training=False)
        Forward pass of the static enrichment layer.

    get_config()
        Returns the configuration dictionary
        for serialization.

    from_config(`config`)
        Instantiates the layer from a
        configuration dictionary.

    Examples
    --------
    >>> from fusionlab.nn.components import StaticEnrichmentLayer
    >>> import tensorflow as tf
    >>> # Define static context of shape (batch_size, units)
    ... # and temporal features of shape
    ... # (batch_size, time_steps, units)
    >>> static_context_vector = tf.random.normal((32, 64))
    >>> temporal_features = tf.random.normal((32, 10, 64))
    >>> # Instantiate the static enrichment layer
    >>> sel = StaticEnrichmentLayer(
    ...     units=64,
    ...     activation='relu',
    ...     use_batch_norm=True
    ... )
    >>> # Forward pass
    >>> outputs = sel(
    ...     static_context_vector,
    ...     temporal_features,
    ...     training=True
    ... )

    See Also
    --------
    GatedResidualNetwork
        Used within the static enrichment layer to
        combine static and temporal features.
    TemporalFusionTransformer
        Incorporates the static enrichment mechanism.

    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2021). "Time-series
           forecasting with deep learning: a survey."
           *Philosophical Transactions of the Royal
           Society A*, 379(2194), 20200209.
    """

    @validate_params({
        "units": [Interval(Integral, 1, None, 
                           closed='left')],
        "use_batch_norm": [bool],
    })
    @ensure_pkg(KERAS_BACKEND or "keras", 
                extra=DEP_MSG)
    def __init__(
            self,
            units,
            activation='elu',
            use_batch_norm=False,
            **kwargs
    ):
        r"""
        Initialize the StaticEnrichmentLayer.

        Parameters
        ----------
        units : int
            Number of hidden units in the internal
            :class:`GatedResidualNetwork`.
        activation : str, optional
            Activation function for the GRN.
            Defaults to ``'elu'``.
        use_batch_norm : bool, optional
            Whether to apply batch normalization
            in the GRN. Defaults to ``False``.
        **kwargs :
            Additional arguments passed to
            the parent Keras ``Layer``.
        """
        super().__init__(**kwargs)
        self.units = units
        self.use_batch_norm = use_batch_norm

        # Create the activation object
        self.activation = activation

        # GatedResidualNetwork instance
        self.grn = GatedResidualNetwork(
            units=units,
            activation=self.activation,
            use_batch_norm=use_batch_norm
        )
    @tf_autograph.experimental.do_not_convert
    def call(
        self,
        temporal_features,
        context_vector,
        training=False
    ):
        r"""
        Forward pass of the static enrichment layer.

        Parameters
        ----------
        ``static_context_vector`` : tf.Tensor
            Static context of shape 
            :math:`(B, U)`.
        ``temporal_features`` : tf.Tensor
            Temporal features of shape
            :math:`(B, T, D)`.
        training : bool, optional
            Whether the layer is in training mode.
            Defaults to ``False``.

        Returns
        -------
        tf.Tensor
            Enriched temporal features of shape
            :math:`(B, T, U)`, assuming 
            ``units = U``.

        Notes
        -----
        1. Expand and tile `static_context_vector`
           over time steps.
        2. Concatenate with `temporal_features`.
        3. Pass through internal GRN for final
           transformation.
        """
        # Expand the static context to align
        # with temporal features along T
        static_context_expanded = tf_expand_dims(
            context_vector,
            axis=1
        )

        # Tile across the time dimension
        static_context_expanded = tf_tile(
            static_context_expanded,
            [
                1,
                tf_shape(temporal_features)[1],
                1
            ]
        )

        # Concatenate static context
        # with temporal features
        combined = tf_concat(
            [static_context_expanded, temporal_features],
            axis=-1
        )

        # Transform with GRN
        output = self.grn(combined, training=training)
        return output

    def get_config(self):
        r"""
        Return the layer configuration for
        serialization.

        Returns
        -------
        dict
            Configuration dictionary containing
            initialization parameters.
        """
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm,
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Create a new instance from a config
        dictionary.

        Parameters
        ----------
        ``config`` : dict
            Configuration as returned by
            ``get_config``.

        Returns
        -------
        StaticEnrichmentLayer
            Instantiated layer object.
        """
        return cls(**config)


# -------------------- XTFT components ----------------------------------------

@register_keras_serializable(
    'fusionlab.nn.components', name="LearnedNormalization"
)
class LearnedNormalization(Layer, NNLearner):
    r"""
    Learned Normalization layer that learns mean and
    standard deviation parameters for normalizing
    input features. This layer can be used to replace
    or augment standard data preprocessing steps by
    allowing the model to learn the optimal scaling
    dynamically.

    Parameters
    ----------
    None
        This layer does not define additional
        initialization parameters besides standard
        Keras `Layer`.

    Notes
    -----
    This layer maintains two trainable weights:
    1) mean: shape :math:`(D,)`
    2) stddev: shape :math:`(D,)`
    where ``D`` is the last dimension of the input
    (feature dimension).

    Methods
    -------
    call(`inputs`, training=False)
        Forward pass. Normalizes the input by subtracting
        the learned mean and dividing by the learned
        standard deviation plus a small epsilon.

    get_config()
        Returns the configuration dictionary for
        serialization.

    from_config(`config`)
        Instantiates the layer from a config dictionary.

    Examples
    --------
    >>> from fusionlab.nn.components import LearnedNormalization
    >>> import tensorflow as tf
    >>> # Create input of shape (batch_size, features)
    >>> x = tf.random.normal((32, 10))
    >>> # Instantiate the learned normalization layer
    >>> norm_layer = LearnedNormalization()
    >>> # Forward pass
    >>> x_norm = norm_layer(x)

    See Also
    --------
    MultiModalEmbedding
        An embedding layer that can be used alongside
        learned normalization in a pipeline.
    HierarchicalAttention
        Another specialized layer for attention
        mechanisms.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, **kws):
        super().__init__(**kws)

    def build(self, input_shape):
        r"""
        Build method that creates trainable weights
        for mean and stddev according to the last
        dimension of the input.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input, typically
            (batch_size, ..., feature_dim).
        """
        self.mean = self.add_weight(
            "mean",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True
        )
        self.stddev = self.add_weight(
            "stddev",
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True
        )
        super().build(input_shape)

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        r"""
        Forward pass of the LearnedNormalization layer.

        Subtracts the learned `mean` from ``inputs`` and
        divides by ``stddev + 1e-6`` to avoid division by zero.

        Parameters
        ----------
        ``inputs`` : tf.Tensor
            Input tensor of shape 
            :math:`(B, ..., D)`.
        training : bool, optional
            Flag indicating if the layer is in
            training mode. Defaults to ``False``.

        Returns
        -------
        tf.Tensor
            Normalized tensor of the same shape
            as ``inputs``.
        """
        return (inputs - self.mean) / (self.stddev + 1e-6)

    def get_config(self):
        r"""
        Returns the configuration dictionary for
        this layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        config = super().get_config().copy()
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Instantiates the layer from a config
        dictionary.

        Parameters
        ----------
        ``config`` : dict
            Configuration dictionary.

        Returns
        -------
        LearnedNormalization
            A new instance of this layer.
        """
        return cls(**config)


@register_keras_serializable(
    'fusionlab.nn.components', name="MultiModalEmbedding"
)
class MultiModalEmbedding(Layer, NNLearner):
    r"""
    MultiModalEmbedding layer for embedding multiple
    input modalities into a common feature space and
    concatenating them along the last dimension.

    This layer takes a list of tensors, each representing
    a different modality with the same batch and time
    dimensions. It applies a dense projection (with
    activation) to each modality, converting them to
    the same dimensionality before concatenation.

    .. math::
        \mathbf{H}_{out} = \text{Concat}\big(
        \text{Dense}(\mathbf{M_1}),\,
        \text{Dense}(\mathbf{M_2}),\,\dots\big)

    where each :math:`\mathbf{M_i}` is a tensor for a
    specific modality.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the output embedding for
        each modality.

    Notes
    -----
    This layer expects each input modality tensor to
    have the same batch and time dimensions,
    but potentially different feature dimensions.

    Methods
    -------
    call(`inputs`, training=False)
        Forward pass that projects each modality
        separately, then concatenates.

    get_config()
        Returns a configuration dictionary for
        serialization.

    from_config(`config`)
        Recreates the layer from a config dict.

    Examples
    --------
    >>> from fusionlab.nn.components import MultiModalEmbedding
    >>> import tensorflow as tf
    >>> # Suppose we have two modalities:
    ... #   dynamic_modality  : (batch, time, dyn_dim)
    ... #   future_modality   : (batch, time, fut_dim)
    >>> dyn_input = tf.random.normal((32, 10, 16))
    >>> fut_input = tf.random.normal((32, 10, 8))
    >>> # Instantiate the layer
    >>> mm_embed = MultiModalEmbedding(embed_dim=32)
    >>> # Forward pass with both modalities
    >>> outputs = mm_embed([dyn_input, fut_input])

    See Also
    --------
    LearnedNormalization
        Normalizes input features before embedding.
    HierarchicalAttention
        Another specialized layer that can be used
        after embeddings are computed.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Will hold a separate Dense layer
        # for each modality
        self.dense_layers = []

    def build(self, input_shape):
        r"""
        Build method that creates a Dense layer
        for each modality based on input_shape.

        Parameters
        ----------
        input_shape : list of tuples
            Each tuple corresponds to a modality's
            shape, typically (batch_size, time_steps,
            feature_dim).
        """
        for modality_shape in input_shape:
            if modality_shape is not None:
                self.dense_layers.append(
                    Dense(
                        self.embed_dim,
                        activation='relu'
                    )
                )
            else:
                raise ValueError(
                    "Unsupported modality type."
                )
        super().build(input_shape)

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        r"""
        Forward pass: project each modality
        into `embed_dim` and concatenate.

        Parameters
        ----------
        ``inputs`` : list of tf.Tensor
            Each tensor has shape
            :math:`(B, T, D_i)` where `D_i` can
            vary by modality.
        training : bool, optional
            Indicates if the layer is in training
            mode. Defaults to ``False``.

        Returns
        -------
        tf.Tensor
            A concatenated embedding of shape
            :math:`(B, T, \sum_{i}(\text{embed_dim}))`.
        """
        embeddings = []
        for idx, modality in enumerate(inputs):
            if isinstance(modality, Tensor):
                modality_embed = (
                    self.dense_layers[idx](
                        modality
                    )
                )
            else:
                raise ValueError(
                    "Unsupported modality type."
                )
            embeddings.append(modality_embed)

        return tf_concat(embeddings, axis=-1)

    def get_config(self):
        r"""
        Returns the configuration dictionary
        of this layer.

        Returns
        -------
        dict
            Configuration including `embed_dim`.
        """
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Recreates a MultiModalEmbedding layer from
        a config dictionary.

        Parameters
        ----------
        ``config`` : dict
            Configuration as produced by
            ``get_config``.

        Returns
        -------
        MultiModalEmbedding
            A new instance of this layer.
        """
        return cls(**config)


@register_keras_serializable(
    'fusionlab.nn.components', 
    name="HierarchicalAttention"
)
class HierarchicalAttention(Layer, NNLearner):
    r"""
    Hierarchical Attention layer that processes
    short-term and long-term sequences separately
    using multi-head attention, then combines
    their outputs [1]_.

    This allows the model to focus on different
    aspects of the data in short-term and long-term
    contexts and aggregate the attention outputs
    for a more comprehensive representation.

    .. math::
        \mathbf{Z} = \text{MHA}(\mathbf{X}_{s})
                     + \text{MHA}(\mathbf{X}_{l})

    where :math:`\mathbf{X}_{s}` and
    :math:`\mathbf{X}_{l}` are the short- and
    long-term sequences, respectively.

    Parameters
    ----------
    units : int
        Dimensionality of the projection for the
        attention keys, queries, and values.
    num_heads : int
        Number of attention heads to use in each
        multi-head attention sub-layer.

    Notes
    -----
    The output shape depends on the last
    dimension in the short and long sequences,
    projected to `units`. The final output is
    the sum of the short-term attention output
    and the long-term attention output.

    Methods
    -------
    call(`inputs`, training=False)
        Forward pass. Expects a list `[short_term,
        long_term]` with shapes
        (B, T, D_s) and (B, T, D_l).

    get_config()
        Returns configuration dictionary for
        serialization.

    from_config(`config`)
        Recreates the layer from a config dict.

    Examples
    --------
    >>> from fusionlab.nn.components import HierarchicalAttention
    >>> import tensorflow as tf
    >>> # Suppose short_term and long_term have
    ... # shape (batch_size, time_steps, features).
    >>> short_term = tf.random.normal((32, 10, 64))
    >>> long_term  = tf.random.normal((32, 10, 64))
    >>> # Instantiate hierarchical attention
    >>> ha = HierarchicalAttention(units=64, num_heads=4)
    >>> # Forward pass
    >>> outputs = ha([short_term, long_term])

    See Also
    --------
    MultiModalEmbedding
        Can precede attention by embedding
        multiple sources of input.
    LearnedNormalization
        Can be applied to short_term and
        long_term sequences prior to attention.

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N.,
           Uszkoreit, J., Jones, L., Gomez, A. N.,
           Kaiser, L., & Polosukhin, I. (2017).
           "Attention is all you need."
           In *Advances in Neural Information
           Processing Systems* (pp. 5998-6008).
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, units: int, num_heads: int):
        super().__init__()
        self.units = units

        # Dense layers for short/long sequences
        self.short_term_dense = Dense(units)
        self.long_term_dense = Dense(units)

        # Multi-head attention for short/long
        self.short_term_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units
        )
        self.long_term_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units
        )

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        r"""
        Forward pass of the HierarchicalAttention.

        Parameters
        ----------
        ``inputs`` : list of tf.Tensor
            A list `[short_term, long_term]`.
            Each tensor should have shape
            :math:`(B, T, D)`.
        training : bool, optional
            Indicates whether the layer is
            in training mode. Defaults to
            ``False``.

        Returns
        -------
        tf.Tensor
            A tensor of shape :math:`(B, T, U)`,
            where `U = units`, representing the
            combined attention outputs.
        """
        short_term, long_term = inputs

        # Linear projections to unify
        # dimensionality
        short_term = self.short_term_dense(
            short_term
        )
        long_term = self.long_term_dense(
            long_term
        )

        # Multi-head attention on short_term
        short_term_attention = (
            self.short_term_attention(
                short_term,
                short_term
            )
        )

        # Multi-head attention on long_term
        long_term_attention = (
            self.long_term_attention(
                long_term,
                long_term
            )
        )

        # Combine
        return short_term_attention + long_term_attention

    def get_config(self):
        r"""
        Returns a dictionary of config
        parameters for serialization.

        Returns
        -------
        dict
            Dictionary with 'units',
            'short_term_dense' config,
            and 'long_term_dense' config.
        """
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'short_term_dense': self.short_term_dense.get_config(),
            'long_term_dense': self.long_term_dense.get_config()
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Recreates the HierarchicalAttention
        layer from a config dictionary.

        Parameters
        ----------
        ``config`` : dict
            Configuration dictionary.

        Returns
        -------
        HierarchicalAttention
            A new instance with the
            specified configuration.
        """
        return cls(**config)

@register_keras_serializable(
    'fusionlab.nn.components',
    name="CrossAttention"
)
class CrossAttention(Layer, NNLearner):
    r"""
    CrossAttention layer that attends one source
    sequence to another [1]_.

    This layer transforms two input sources,
    ``source1`` and ``source2``, into a shared
    dimensionality via separate dense layers,
    then applies multi-head attention using
    ``source1`` as the query and ``source2`` as
    both key and value. The output shape depends
    on the specified ``units``.

    .. math::
        \mathbf{H}_{\text{out}} = \text{MHA}(
            \mathbf{W}_{1}\,\mathbf{S}_1,\,
            \mathbf{W}_{2}\,\mathbf{S}_2,\,
            \mathbf{W}_{2}\,\mathbf{S}_2
        )

    where :math:`\mathbf{S}_1` and :math:`\mathbf{S}_2`
    are the two source sequences.

    Parameters
    ----------
    units : int
        Dimensionality for the internal projections
        of the query/key/value in multi-head attention.
    num_heads : int
        Number of attention heads.

    Notes
    -----
    Cross attention is particularly useful when
    focusing on how one sequence (the query) relates
    to another (the key/value). For example, in
    multi-modal time series settings, one might
    attend dynamic covariates to static ones or
    vice versa.

    Methods
    -------
    call(`inputs`, training=False)
        Forward pass of the cross-attention layer.
    get_config()
        Returns the configuration dictionary for
        serialization.
    from_config(`config`)
        Creates a new layer from the given config.

    Examples
    --------
    >>> from fusionlab.nn.components import CrossAttention
    >>> import tensorflow as tf
    >>> # Two sequences of shape (batch_size, time_steps, features)
    >>> source1 = tf.random.normal((32, 10, 64))
    >>> source2 = tf.random.normal((32, 10, 64))
    >>> # Instantiate the CrossAttention layer
    >>> cross_attn = CrossAttention(units=64, num_heads=4)
    >>> # Forward pass
    >>> outputs = cross_attn([source1, source2])

    See Also
    --------
    HierarchicalAttention
        Another attention-based layer focusing on
        short/long-term sequences.
    MemoryAugmentedAttention
        Uses a learned memory matrix to enhance
        representations.

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N.,
           Uszkoreit, J., Jones, L., Gomez, A. N.,
           Kaiser, L., & Polosukhin, I. (2017).
           "Attention is all you need." In
           *Advances in Neural Information
           Processing Systems* (pp. 5998-6008).
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, units: int, num_heads: int):
        r"""
        Initialize the CrossAttention layer.

        Parameters
        ----------
        units : int
            Number of output units for the
            internal Dense projections and
            multi-head attention dimension.
        num_heads : int
            Number of attention heads to use
            in the multi-head attention module.
        """
        super().__init__()
        self.units = units
        # Dense layers to project each source
        self.source1_dense = Dense(units)
        self.source2_dense = Dense(units)
        # Multi-head attention
        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units
        )

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        r"""
        Forward pass of CrossAttention.

        Parameters
        ----------
        ``inputs`` : list of tf.Tensor
            A list [source1, source2], each of shape
            (batch_size, time_steps, features).
        training : bool, optional
            Indicates if the layer is in training
            mode (for dropout, if any).
            Defaults to ``False``.

        Returns
        -------
        tf.Tensor
            A tensor of shape (batch_size, time_steps,
            units) representing cross-attended features.
        """
        source1, source2 = inputs
        # Project each source
        source1 = self.source1_dense(source1)
        source2 = self.source2_dense(source2)
        # Apply cross attention
        return self.cross_attention(
            query=source1,
            value=source2,
            key=source2
        )

    def get_config(self):
        r"""
        Returns configuration dictionary for this
        layer.

        Returns
        -------
        dict
            Configuration dictionary, including
            'units'.
        """
        config = super().get_config().copy()
        config.update({'units': self.units})
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Create a new CrossAttention layer from
        the given config dictionary.

        Parameters
        ----------
        ``config`` : dict
            Configuration as returned by
            ``get_config``.

        Returns
        -------
        CrossAttention
            A new instance of CrossAttention.
        """
        return cls(**config)


@register_keras_serializable(
    'fusionlab.nn.components', 
    name="MemoryAugmentedAttention"
)
class MemoryAugmentedAttention(Layer, NNLearner):
    r"""
    Memory-Augmented Attention layer that uses a
    learned memory matrix to enhance temporal
    representation [1]_.

    This layer maintains a trainable memory of
    shape :math:`(\text{memory_size}, \text{units})`
    and attends over it with the input serving
    as the query. The resulting context is added
    back to the input as a residual connection,
    giving a memory-augmented feature.

    .. math::
        \mathbf{Z} = \mathbf{X} +
        \text{MHA}(\mathbf{X}, \mathbf{M}, \mathbf{M})

    where :math:`\mathbf{M}` is the learned memory.

    Parameters
    ----------
    units : int
        Dimensionality for the memory and the
        multi-head attention projections.
    memory_size : int
        Number of slots in the learned memory
        matrix.
    num_heads : int
        Number of attention heads in the
        multi-head attention.

    Notes
    -----
    The learned memory is a trainable parameter
    of shape (memory_size, units). It is expanded
    at each forward pass to match the batch size.

    Methods
    -------
    call(`inputs`, training=False)
        Forward pass of the memory-augmented
        attention layer.
    get_config()
        Returns the configuration for
        serialization.
    from_config(`config`)
        Instantiates the layer from the given
        config dictionary.

    Examples
    --------
    >>> from fusionlab.nn.components import MemoryAugmentedAttention
    >>> import tensorflow as tf
    >>> # Suppose we have an input of shape (batch_size, time_steps, units)
    >>> x = tf.random.normal((32, 10, 64))
    >>> # Instantiate with a memory size of 20
    >>> maa = MemoryAugmentedAttention(
    ...     units=64,
    ...     memory_size=20,
    ...     num_heads=4
    ... )
    >>> # Forward pass
    >>> outputs = maa(x)

    See Also
    --------
    CrossAttention
        Another specialized attention mechanism
        focusing on cross-sequence interactions.
    HierarchicalAttention
        Combines short/long-term sequences with
        attention.

    References
    ----------
    .. [1] Graves, A., Wayne, G., & Danihelka, I.
           (2014). Neural Turing Machines. *arXiv
           preprint arXiv:1410.5401*.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        units: int,
        memory_size: int,
        num_heads: int
    ):
        super().__init__()
        self.units = units
        self.memory_size = memory_size
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units
        )

    def build(self, input_shape):
        r"""
        Build method that creates the trainable
        memory matrix of shape
        (memory_size, units).

        Parameters
        ----------
        input_shape : tuple
            Shape of the input, e.g. 
            (batch_size, time_steps, units).
        """
        self.memory = self.add_weight(
            "memory",
            shape=(self.memory_size, self.units),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        r"""
        Forward pass of MemoryAugmentedAttention.

        Parameters
        ----------
        ``inputs`` : tf.Tensor
            A 3D tensor of shape (batch_size,
            time_steps, units).
        training : bool, optional
            Indicates whether the layer is in
            training mode. Defaults to ``False``.

        Returns
        -------
        tf.Tensor
            A tensor of the same shape as inputs:
            (batch_size, time_steps, units), 
            augmented by the learned memory.
        """
        # Expand memory to match batch dimension
        batch_size = tf_shape(inputs)[0]
        memory_expanded = tf_expand_dims(self.memory, axis=0)
        memory_expanded = tf_tile(
            memory_expanded,
            [batch_size, 1, 1]
        )

        # Attend memory with inputs as query
        memory_attended = self.attention(
            query=inputs,
            value=memory_expanded,
            key=memory_expanded
        )
        # Residual connection
        return memory_attended + inputs

    def get_config(self):
        r"""
        Returns configuration of this layer.

        Returns
        -------
        dict
            Dictionary including 'units' and
            'memory_size'.
        """
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'memory_size': self.memory_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Creates a new instance from a given
        config dictionary.

        Parameters
        ----------
        ``config`` : dict
            Configuration dictionary as returned
            by ``get_config``.

        Returns
        -------
        MemoryAugmentedAttention
            A new instance of this layer.
        """
        return cls(**config)


@register_keras_serializable(
    'fusionlab.nn.components', 
    name="AdaptiveQuantileLoss"
)
class AdaptiveQuantileLoss(Loss, NNLearner):
    r"""
    Adaptive Quantile Loss layer that computes
    quantile loss for given quantiles [1]_.

    The layer expects ``y_true`` of shape
    :math:`(B, H, O)`, where ``B`` is batch size,
    ``H`` is horizon, and ``O`` is output dimension,
    and ``y_pred`` of shape
    :math:`(B, H, Q, O)`, where ``Q`` is the
    number of quantiles if they are specified.

    .. math::
        \text{QuantileLoss}(\hat{y}, y) =
        \max(q \cdot (y - \hat{y}),\,
        (q - 1) \cdot (y - \hat{y}))

    The final loss is the mean across batch, time,
    quantiles, and output dimension.

    Parameters
    ----------
    quantiles : list of float, optional
        A list of quantiles used to compute
        quantile loss. If set to ``'auto'``,
        defaults to [0.1, 0.5, 0.9]. If ``None``,
        the loss returns 0.0 (no quantile loss).

    Notes
    -----
    For quantile regression, each quantile
    penalizes under- and over-estimates
    differently, encouraging a robust modeling
    of the distribution of possible outcomes.

    Methods
    -------
    call(`y_true`, `y_pred`, training=False)
        Compute the quantile loss.
    get_config()
        Returns configuration for serialization.
    from_config(`config`)
        Creates a new instance from config dict.

    Examples
    --------
    >>> from fusionlab.nn.components import AdaptiveQuantileLoss
    >>> import tensorflow as tf
    >>> # Suppose y_true is (B, H, O)
    ... # y_pred is (B, H, Q, O)
    >>> y_true = tf.random.normal((32, 10, 1))
    >>> y_pred = tf.random.normal((32, 10, 3, 1))
    >>> # Instantiate with custom quantiles
    >>> aq_loss = AdaptiveQuantileLoss([0.2, 0.5, 0.8])
    >>> # Forward pass (loss calculation)
    >>> loss_value = aq_loss(y_true, y_pred)

    See Also
    --------
    MultiObjectiveLoss
        Can combine this quantile loss with an
        anomaly loss.
    AnomalyLoss
        Computes anomaly-based loss, complementary
        to quantile loss.

    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2021). "Time-series
           forecasting with deep learning: a survey."
           *Philosophical Transactions of the Royal
           Society A*, 379(2194), 20200209.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, quantiles: Optional[List[float]], 
                 name="AdaptiveQuantileLoss"):
        super().__init__(name=name)
        if quantiles == 'auto':
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles

    @tf_autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        r"""
        Compute quantile loss.

        Parameters
        ----------
        ``y_true`` : tf.Tensor
            Ground truth of shape (B, H, O).
        ``y_pred`` : tf.Tensor
            Predicted values of shape (B, H, Q, O)
            if quantiles is not None.
        training : bool, optional
            Unused parameter, included for
            consistency. Defaults to ``False``.

        Returns
        -------
        tf.Tensor
            A scalar representing the mean quantile
            loss. 0.0 if ``quantiles`` is None.
        """
        if self.quantiles is None:
            return 0.0
        # Expand y_true to match y_pred's quantile
        # dimension
        y_true_expanded = tf_expand_dims(
            y_true,
            axis=2
        )  # => (B, H, 1, O)
        error = y_true_expanded - y_pred
        quantiles = tf_constant(
            self.quantiles,
            dtype=tf_float32
        )
        quantiles = tf_reshape(
            quantiles,
            [1, 1, len(self.quantiles), 1]
        )
        # quantile loss
        quantile_loss = tf_maximum(
            quantiles * error,
            (quantiles - 1) * error
        )
        return tf_reduce_mean(quantile_loss)

    def get_config(self):
        r"""
        Configuration for serialization.

        Returns
        -------
        dict
            Dictionary with 'quantiles'.
        """
        config = super().get_config().copy()
        config.update({'quantiles': self.quantiles})
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Creates a new instance from a config dict.

        Parameters
        ----------
        ``config`` : dict
            Configuration dictionary.

        Returns
        -------
        AdaptiveQuantileLoss
            A new instance of the layer.
        """
        return cls(**config)


@register_keras_serializable(
    'fusionlab.nn.components',
    name="AnomalyLoss"
)
class AnomalyLoss(Loss, NNLearner):
    r"""
    Anomaly Loss layer computing mean squared
    anomaly scores.

    This layer expects anomaly scores of shape
    :math:`(B, H, D)` and multiplies their
    mean squared value by a weight factor.

    .. math::
        \text{AnomalyLoss}(\mathbf{a}) =
        w \cdot \frac{1}{BHD} \sum (\mathbf{a})^2

    where :math:`\mathbf{a}` is the anomaly
    score, and :math:`w` is the weight.

    Parameters
    ----------
    weight : float, optional
        Scalar multiplier for the computed
        mean squared anomaly scores.
        Defaults to 1.0.

    Notes
    -----
    Anomaly loss is often combined with other
    losses in a multi-task setting where
    predictive performance and anomaly detection
    performance are both important.

    Methods
    -------
    call(`anomaly_scores`)
        Compute mean squared anomaly loss.
    get_config()
        Return configuration for serialization.
    from_config(`config`)
        Instantiates a new instance from config.

    Examples
    --------
    >>> from fusionlab.nn.components import AnomalyLoss
    >>> import tensorflow as tf
    >>> # Suppose anomaly_scores is (B, H, D)
    >>> anomaly_scores = tf.random.normal((32, 10, 8))
    >>> # Instantiate anomaly loss
    >>> anomaly_loss_fn = AnomalyLoss(weight=2.0)
    >>> # Compute anomaly loss
    >>> loss_value = anomaly_loss_fn(anomaly_scores)

    See Also
    --------
    AdaptiveQuantileLoss
        Another specialized loss that can be
        combined for multi-objective optimization.
    MultiObjectiveLoss
        Demonstrates how anomaly and quantile loss
        can be merged.

    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2021). "Time-series
           forecasting with deep learning: a survey."
           *Philosophical Transactions of the Royal
           Society A*, 379(2194), 20200209.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, weight: float = 1.0, name="AnomalyLoss"):

        super().__init__(name=name)
        self.weight = weight

    @tf_autograph.experimental.do_not_convert
    def call(self, anomaly_scores: Tensor, y_pred=None 
             ): 
        r"""
        Forward pass that computes the mean squared
        anomaly score multiplied by `weight`.

        Parameters
        ----------
        ``anomaly_scores`` : tf.Tensor
            Tensor of shape (B, H, D) representing
            anomaly scores.
        ``y_pred``: Optional 
           Does nothing, just for API consistency.

        Returns
        -------
        tf.Tensor
            A scalar loss value representing the
            weighted mean squared anomaly.
        """
        return self.weight * tf_reduce_mean(
            tf_square(anomaly_scores)
        )

    def get_config(self):
        r"""
        Return configuration dictionary for
        this layer.

        Returns
        -------
        dict
            Includes 'weight'.
        """
        config = super().get_config().copy()
        config.update({'weight': self.weight})
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Recreates an AnomalyLoss layer from a config.

        Parameters
        ----------
        ``config`` : dict
            Configuration containing 'weight'.

        Returns
        -------
        AnomalyLoss
            A new instance of this layer.
        """
        return cls(**config)


@register_keras_serializable(
    'fusionlab.nn.components', 
    name="MultiObjectiveLoss"
)
class MultiObjectiveLoss(Loss, NNLearner):
    r"""
    Multi-Objective Loss layer combining quantile
    loss and anomaly loss [1]_.

    This layer expects:
    1. ``y_true``: :math:`(B, H, O)`
    2. ``y_pred``: :math:`(B, H, Q, O)`, if
       quantiles are used (or (B, H, 1, O)
       for a single quantile).
    3. ``anomaly_scores``: :math:`(B, H, D)`,
       optional.

    .. math::
        \text{Loss} = \text{QuantileLoss} +
                      \text{AnomalyLoss}

    If ``anomaly_scores`` is None, only
    quantile loss is returned.

    Parameters
    ----------
    quantile_loss_fn : Layer
        A callable implementing quantile loss, e.g.
        :class:`AdaptiveQuantileLoss`.
    anomaly_loss_fn : Layer
        A  callable implementing anomaly loss, e.g.
        :class:`AnomalyLoss`.

    Notes
    -----
    This layer allows multi-task learning by
    combining two objectives: forecasting
    accuracy (quantile loss) and anomaly
    detection (anomaly loss).

    Methods
    -------
    call(`y_true`, `y_pred`, `anomaly_scores`=None,
         training=False)
        Compute the combined loss.
    get_config()
        Returns configuration for serialization.
    from_config(`config`)
        Rebuilds the layer from a config dict.

    Examples
    --------
    >>> from fusionlab.nn.components import (
    ...     MultiObjectiveLoss,
    ...     AdaptiveQuantileLoss,
    ...     AnomalyLoss
    ... )
    >>> import tensorflow as tf
    >>> # Suppose y_true is (B, H, O),
    ... # and y_pred is (B, H, Q, O).
    >>> y_true = tf.random.normal((32, 10, 1))
    >>> y_pred = tf.random.normal((32, 10, 3, 1))
    >>> anomaly_scores = tf.random.normal((32, 10, 8))
    >>> # Instantiate loss components
    >>> q_loss_fn = AdaptiveQuantileLoss([0.2, 0.5, 0.8])
    >>> a_loss_fn = AnomalyLoss(weight=2.0)
    >>> # Combine them
    >>> mo_loss = MultiObjectiveLoss(q_loss_fn, a_loss_fn)
    >>> # Compute the combined loss
    >>> total_loss = mo_loss(y_true, y_pred, anomaly_scores)

    See Also
    --------
    AdaptiveQuantileLoss
        Implements quantile loss to handle
        uncertainty in predictions.
    AnomalyLoss
        Computes anomaly-based MSE for
        anomaly detection tasks.

    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2021).
           "Time-series forecasting with deep
           learning: a survey." *Philosophical
           Transactions of the Royal Society A*,
           379(2194), 20200209.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        quantile_loss_fn,
        anomaly_loss_fn, 
        name="MultiObjectiveLoss"
    ):
        super().__init__(name=name)
        self.quantile_loss_fn = quantile_loss_fn
        self.anomaly_loss_fn = anomaly_loss_fn

    @tf_autograph.experimental.do_not_convert
    def call(self, y_true, y_pred, anomaly_scores=None):
             
        r"""
        Compute combined quantile and anomaly loss.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth of shape (B, H, O).
        y_pred : tf.Tensor
            Predictions of shape (B, H, Q, O)
            (or (B, H, 1, O) if Q=1).
        anomaly_scores : tf.Tensor or None, optional
            Tensor of shape (B, H, D).
            If None, anomaly loss is omitted.
        training : bool, optional
            Indicates training mode. Defaults to
            ``False``.

        Returns
        -------
        tf.Tensor
            A scalar representing the sum of
            quantile loss and anomaly loss (if
            anomaly_scores is provided).
        """
        quantile_loss = self.quantile_loss_fn(
            y_true,
            y_pred
        )
        if anomaly_scores is not None:
            anomaly_loss = self.anomaly_loss_fn(
                anomaly_scores
            )
            return quantile_loss + anomaly_loss
        return quantile_loss

    def get_config(self):
        r"""
        Returns configuration dictionary, including
        configs of the sub-layers.

        Returns
        -------
        dict
            Contains serialized configs of
            quantile_loss_fn and anomaly_loss_fn.
        """
        config = super().get_config().copy()
        config.update({
            'quantile_loss_fn': self.quantile_loss_fn.get_config(),
            'anomaly_loss_fn': self.anomaly_loss_fn.get_config()
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Creates a new MultiObjectiveLoss from
        the config dictionary.

        Parameters
        ----------
        ``config`` : dict
            Configuration dictionary with sub-layer
            configs.

        Returns
        -------
        MultiObjectiveLoss
            A new instance combining quantile
            and anomaly losses.
        """
        # Rebuild sub-layers from their configs
        quantile_loss_fn = AdaptiveQuantileLoss.from_config(
            config['quantile_loss_fn']
        )
        anomaly_loss_fn = AnomalyLoss.from_config(
            config['anomaly_loss_fn']
        )
        return cls(
            quantile_loss_fn=quantile_loss_fn,
            anomaly_loss_fn=anomaly_loss_fn
        )

@register_keras_serializable(
    'fusionlab.nn.components', 
    name="ExplainableAttention"
)
class ExplainableAttention(Layer, NNLearner):
    r"""
    ExplainableAttention layer that returns attention
    scores from multi-head attention [1]_.

    This layer is useful for interpretability,
    providing insight into how the attention
    mechanism focuses on different time steps.

    .. math::
        \mathbf{A} = \text{MHA}(\mathbf{X},\,\mathbf{X})
        \rightarrow \text{attention\_scores}

    Here, :math:`\mathbf{X}` is an input tensor,
    and ``attention_scores`` is the matrix
    capturing attention weights.

    Parameters
    ----------
    num_heads : int
        Number of heads for multi-head attention.
    key_dim : int
        Dimensionality of the query/key projections.

    Notes
    -----
    Unlike standard layers that return the
    transformation output, this layer specifically
    returns the attention score matrix for
    interpretability.

    Methods
    -------
    call(`inputs`, training=False)
        Forward pass that outputs only the
        attention scores.
    get_config()
        Returns the configuration for serialization.
    from_config(`config`)
        Creates a new instance from the given config.

    Examples
    --------
    >>> from fusionlab.nn.components import ExplainableAttention
    >>> import tensorflow as tf
    >>> # Suppose we have input of shape (batch_size, time_steps, features)
    >>> x = tf.random.normal((32, 10, 64))
    >>> # Instantiate explainable attention
    >>> ea = ExplainableAttention(num_heads=4, key_dim=64)
    >>> # Forward pass returns attention scores: (B, num_heads, T, T)
    >>> scores = ea(x)

    See Also
    --------
    CrossAttention
        Another attention variant for cross-sequence
        contexts.
    MultiResolutionAttentionFusion
        For fusing features via multi-head attention.

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N.,
           Uszkoreit, J., Jones, L., Gomez, A. N.,
           Kaiser, L., & Polosukhin, I. (2017).
           "Attention is all you need." In
           *Advances in Neural Information
           Processing Systems* (pp. 5998-6008).
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, num_heads: int, key_dim: int):
        r"""
        Initialize the ExplainableAttention layer.

        Parameters
        ----------
        num_heads : int
            Number of attention heads.
        key_dim : int
            Dimensionality of query/key projections
            in multi-head attention.
        """
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        # MultiHeadAttention, focusing on returning
        # the attention scores
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim
        )

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        r"""
        Forward pass that returns only the
        attention scores.

        Parameters
        ----------
        ``inputs`` : tf.Tensor
            Tensor of shape (B, T, D).
        training : bool, optional
            Indicates training mode; not used in
            this layer. Defaults to ``False``.

        Returns
        -------
        tf.Tensor
            Attention scores of shape
            (B, num_heads, T, T).
        """
        _, attention_scores = self.attention(
            inputs,
            inputs,
            return_attention_scores=True
        )
        return attention_scores

    def get_config(self):
        r"""
        Returns the layer configuration.

        Returns
        -------
        dict
            Dictionary containing 'num_heads'
            and 'key_dim'.
        """
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Creates a new instance from the config
        dictionary.

        Parameters
        ----------
        ``config`` : dict
            Configuration dictionary.

        Returns
        -------
        ExplainableAttention
            A new instance of this layer.
        """
        return cls(**config)


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


@register_keras_serializable(
    'fusionlab.nn.components', 
    name="MultiResolutionAttentionFusion"
)
class MultiResolutionAttentionFusion(Layer, NNLearner):
    r"""
    MultiResolutionAttentionFusion layer applying
    multi-head attention fusion over features [1]_.

    This layer merges or fuses features at different
    resolutions or sources via multi-head attention.
    The input is projected to shape `(B, T, D)`,
    and the output shares the same shape.

    .. math::
        \mathbf{Z} = \text{MHA}(\mathbf{X}, \mathbf{X})

    Parameters
    ----------
    units : int
        Dimension of the key, query, and value
        projections.
    num_heads : int
        Number of attention heads.

    Notes
    -----
    Typically used in multi-resolution contexts
    where time steps or multiple feature sets
    are merged.

    Methods
    -------
    call(`inputs`, training=False)
        Forward pass of the multi-head attention
        layer.
    get_config()
        Returns config for serialization.
    from_config(`config`)
        Reconstructs the layer from a config.

    Examples
    --------
    >>> from fusionlab.nn.components import MultiResolutionAttentionFusion
    >>> import tensorflow as tf
    >>> x = tf.random.normal((32, 10, 64))
    >>> # Instantiate multi-resolution attention
    >>> mraf = MultiResolutionAttentionFusion(
    ...     units=64,
    ...     num_heads=4
    ... )
    >>> # Forward pass => (32, 10, 64)
    >>> y = mraf(x)

    See Also
    --------
    HierarchicalAttention
        Combines short and long-term sequences
        with attention.
    ExplainableAttention
        Another attention layer returning
        attention scores.

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N.,
           Uszkoreit, J., Jones, L., Gomez, A. N.,
           Kaiser, L., & Polosukhin, I. (2017).
           "Attention is all you need." In
           *Advances in Neural Information
           Processing Systems* (pp. 5998-6008).
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, units: int, num_heads: int):
        r"""
        Initialize the MultiResolutionAttentionFusion
        layer.

        Parameters
        ----------
        units : int
            Dimensionality for the attention
            projections.
        num_heads : int
            Number of heads for multi-head
            attention.
        """
        super().__init__()
        self.units = units
        self.num_heads = num_heads
        # MultiHeadAttention instance
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units
        )

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        r"""
        Forward pass applying multi-head attention
        to fuse features.

        Parameters
        ----------
        ``inputs`` : tf.Tensor
            Tensor of shape (B, T, D).
        training : bool, optional
            Indicates training mode. Defaults to
            ``False``.

        Returns
        -------
        tf.Tensor
            Tensor of shape (B, T, D),
            representing fused features.
        """
        return self.attention(inputs, inputs)

    def get_config(self):
        r"""
        Returns configuration dictionary with
        'units' and 'num_heads'.

        Returns
        -------
        dict
            Configuration for serialization.
        """
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Instantiate a new 
        MultiResolutionAttentionFusion layer from
        config.

        Parameters
        ----------
        ``config`` : dict
            Configuration dictionary.

        Returns
        -------
        MultiResolutionAttentionFusion
            A new instance of this layer.
        """
        return cls(**config)


@register_keras_serializable(
    'fusionlab.nn.components',
    name="DynamicTimeWindow"
)
class DynamicTimeWindow(Layer, NNLearner):
    r"""
    DynamicTimeWindow layer that slices the last
    `max_window_size` steps from the input sequence.

    This helps in focusing on the most recent time
    steps if the sequence is longer than
    `max_window_size`.

    .. math::
        \mathbf{Z} = \mathbf{X}[:, -W:, :]

    where `W` = `max_window_size`.

    Parameters
    ----------
    max_window_size : int
        Number of time steps to keep from
        the end of the sequence.

    Notes
    -----
    This can be used for models that only need
    the last few time steps instead of the entire
    sequence.

    Methods
    -------
    call(`inputs`, training=False)
        Slice the last `max_window_size` steps.
    get_config()
        Returns configuration dictionary.
    from_config(`config`)
        Recreates the layer from config.

    Examples
    --------
    >>> from fusionlab.nn.components import DynamicTimeWindow
    >>> import tensorflow as tf
    >>> x = tf.random.normal((32, 50, 64))
    >>> # Keep last 10 time steps
    >>> dtw = DynamicTimeWindow(max_window_size=10)
    >>> y = dtw(x)
    >>> y.shape
    TensorShape([32, 10, 64])

    See Also
    --------
    MultiResolutionAttentionFusion
        Another layer that can be used after
        slicing to fuse temporal features.

    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2021). 
           "Time-series forecasting with deep
           learning: a survey." 
           *Philosophical Transactions of
           the Royal Society A*, 379(2194),
           20200209.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, max_window_size: int):
        r"""
        Initialize the DynamicTimeWindow layer.

        Parameters
        ----------
        max_window_size : int
            Number of steps to slice from the end
            of the sequence.
        """
        super().__init__()
        self.max_window_size = max_window_size

    def call(self, inputs, training=False):
        r"""
        Forward pass that slices the last
        `max_window_size` steps.

        Parameters
        ----------
        ``inputs`` : tf.Tensor
            Tensor of shape :math:`(B, T, D)`.
        training : bool, optional
            Unused. Defaults to ``False``.

        Returns
        -------
        tf.Tensor
            A sliced tensor of shape 
            :math:`(B, W, D)` where W = 
            `max_window_size`.
        """
        return inputs[:, -self.max_window_size:, :]

    def get_config(self):
        r"""
        Returns configuration dictionary.

        Returns
        -------
        dict
            Contains 'max_window_size'.
        """
        config = super().get_config().copy()
        config.update({
            'max_window_size': self.max_window_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Creates a new DynamicTimeWindow layer
        from config.

        Parameters
        ----------
        ``config`` : dict
            Must include 'max_window_size'.

        Returns
        -------
        DynamicTimeWindow
            A new instance of this layer.
        """
        return cls(**config)


@register_keras_serializable(
    'fusionlab.nn.components', 
    name="QuantileDistributionModeling"
)
class QuantileDistributionModeling(Layer, NNLearner):
    r"""
    QuantileDistributionModeling layer projects
    deterministic outputs into quantile
    predictions [1]_.

    Depending on whether `quantiles` is specified,
    this layer:
      - Returns (B, H, O) if `quantiles` is None.
      - Returns (B, H, Q, O) otherwise, where Q
        is the number of quantiles.

    .. math::
        \mathbf{Y}_q = \text{Dense}_q(\mathbf{X}),
        \forall q \in \text{quantiles}

    Parameters
    ----------
    quantiles : list of float or str or None
        List of quantiles. If `'auto'`, defaults
        to [0.1, 0.5, 0.9]. If ``None``, no extra
        quantile dimension is added.
    output_dim : int
        Output dimension per quantile or in the
        deterministic case.

    Notes
    -----
    This layer is often used after a decoder
    to provide probabilistic forecasts via
    quantile outputs.

    Methods
    -------
    call(`inputs`, training=False)
        Projects inputs into desired quantile
        shape.
    get_config()
        Returns configuration dictionary.
    from_config(`config`)
        Instantiates from config.

    Examples
    --------
    >>> from fusionlab.nn.components import QuantileDistributionModeling
    >>> import tensorflow as tf
    >>> x = tf.random.normal((32, 10, 64))  # (B, H, O)
    >>> # Instantiate with quantiles
    >>> qdm = QuantileDistributionModeling([0.25, 0.5, 0.75], output_dim=1)
    >>> # Forward pass => (B, H, Q, O) => (32, 10, 3, 1)
    >>> y = qdm(x)

    See Also
    --------
    MultiDecoder
        Outputs multi-horizon predictions that
        can be further turned into quantiles.
    AdaptiveQuantileLoss
        Computes quantile losses for outputs
        generated by this layer.

    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2021).
           "Time-series forecasting with deep
           learning: a survey." *Philosophical
           Transactions of the Royal Society A*,
           379(2194), 20200209.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        quantiles: Optional[Union[str, List[float]]],
        output_dim: int
    ):
        r"""
        Initialize the QuantileDistributionModeling
        layer.

        Parameters
        ----------
        quantiles : list of float or str or None
            If `'auto'`, defaults to [0.1, 0.5, 0.9].
            If None, returns deterministic output.
        output_dim : int
            Output dimension for each quantile or
            the deterministic case.
        """
        super().__init__()
        if quantiles == 'auto':
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles
        self.output_dim = output_dim

        # Create Dense layers if quantiles specified
        if self.quantiles is not None:
            self.output_layers = [
                Dense(output_dim) for _ in self.quantiles
            ]
        else:
            self.output_layer = Dense(output_dim)

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        r"""
        Forward pass projecting to quantile outputs
        or deterministic outputs.

        Parameters
        ----------
        ``inputs`` : tf.Tensor
            A 3D tensor of shape (B, H, O).
        training : bool, optional
            Unused in this layer. Defaults to
            ``False``.

        Returns
        -------
        tf.Tensor
            - If `quantiles` is None:
              (B, H, O)
            - Else: (B, H, Q, O)
        """
        # No quantiles => deterministic
        if self.quantiles is None:
            return self.output_layer(inputs)

        # Quantile predictions => (B, H, Q, O)
        outputs = []
        for output_layer in self.output_layers:
            quantile_output = output_layer(inputs)
            outputs.append(quantile_output)
        return tf_stack(outputs, axis=2)

    def get_config(self):
        r"""
        Configuration dictionary for layer
        serialization.

        Returns
        -------
        dict
            Contains 'quantiles' and 'output_dim'.
        """
        config = super().get_config().copy()
        config.update({
            'quantiles': self.quantiles,
            'output_dim': self.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Creates a new instance from the given
        config dict.

        Parameters
        ----------
        ``config`` : dict
            Configuration dictionary with
            'quantiles' and 'output_dim'.

        Returns
        -------
        QuantileDistributionModeling
            A new instance.
        """
        return cls(**config)


@register_keras_serializable(
    'fusionlab.nn.components', 
    name='MultiScaleLSTM'
)
class MultiScaleLSTM(Layer, NNLearner):
    r"""
    MultiScaleLSTM layer applying multiple LSTMs
    at different sampling scales and concatenating
    their outputs [1]_.

    Each LSTM can either return the full sequence
    or only the last hidden state, controlled by
    `return_sequences`. The user specifies `scales`
    to sub-sample the time dimension. For example,
    a scale of 2 processes every 2nd time step.

    Parameters
    ----------
    lstm_units : int
        Number of units in each LSTM.
    scales : list of int or str or None, optional
        List of scale factors. If `'auto'` or None,
        defaults to `[1]` (no sub-sampling).
    return_sequences : bool, optional
        If True, each LSTM returns the entire
        sequence. Otherwise, it returns only the
        last hidden state. Defaults to False.
    **kwargs
        Additional arguments passed to the parent
        Keras `Layer`.

    Notes
    -----
    - If `return_sequences=False`, the output is
      concatenated along features:
      :math:`(B, \text{units} \times \text{num\_scales})`.
    - If `return_sequences=True`, a list of
      sequence outputs is returned. Each may have
      a different time dimension if scales differ.

    Methods
    -------
    call(`inputs`, training=False)
        Forward pass, applying each LSTM at the
        specified scale.
    get_config()
        Returns the layer's configuration dict.
    from_config(`config`)
        Builds the layer from the config dict.

    Examples
    --------
    >>> from fusionlab.nn.components import MultiScaleLSTM
    >>> import tensorflow as tf
    >>> x = tf.random.normal((32, 20, 16))  # (B, T, D)
    >>> # Instantiating a multi-scale LSTM
    >>> mslstm = MultiScaleLSTM(lstm_units=32,
    ...     scales=[1, 2], return_sequences=False)
    >>> y = mslstm(x)  # shape => (32, 64)
    >>> # because scale=1 and scale=2 each produce 32 units,
    ... # which are concatenated => 64

    See Also
    --------
    DynamicTimeWindow
        For slicing sequences before applying
        multi-scale LSTMs.
    TemporalFusionTransformer
        A complex model that can incorporate
        multi-scale modules.

    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2021).
           "Time-series forecasting with deep
           learning: a survey." *Philosophical
           Transactions of the Royal Society A*,
           379(2194), 20200209.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        lstm_units: int,
        scales: Union[str, List[int], None] = None,
        return_sequences: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if scales is None or scales == 'auto':
            scales = [1]
        # Validate that scales is a list of int
        scales = validate_nested_param(
            scales,
            List[int],
            'scales'
        )

        self.lstm_units = lstm_units
        self.scales = scales
        self.return_sequences = return_sequences

        # Create an LSTM for each scale
        self.lstm_layers = [
            LSTM(
                lstm_units,
                return_sequences=return_sequences
            )
            for _ in scales
        ]

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        r"""
        Forward pass that processes the input
        at multiple scales.

        Parameters
        ----------
        ``inputs`` : tf.Tensor
            Shape (B, T, D).
        training : bool, optional
            Training mode. Defaults to ``False``.

        Returns
        -------
        tf.Tensor or list of tf.Tensor
            - If `return_sequences=False`, returns
              a single 2D tensor of shape
              (B, lstm_units * len(scales)).
            - If `return_sequences=True`, returns
              a list of 3D tensors, each with shape
              (B, T', lstm_units), where T' depends
              on the scale sub-sampling.
        """
        outputs = []
        for scale, lstm in zip(self.scales, self.lstm_layers):
            scaled_input = inputs[:, ::scale, :]
            lstm_output = lstm(
                scaled_input,
                training=training
            )
            outputs.append(lstm_output)

        # If return_sequences=False:
        #   => (B, units) from each sub-lstm
        #      -> concat => (B, units*len(scales))
        if not self.return_sequences:
            return tf_concat(outputs, axis=-1)
        else:
            # return a list of sequences
            return outputs

    def get_config(self):
        r"""
        Returns a config dictionary containing
        'lstm_units', 'scales', and
        'return_sequences'.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        config = super().get_config().copy()
        config.update({
            'lstm_units': self.lstm_units,
            'scales': self.scales,
            'return_sequences': self.return_sequences
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Builds MultiScaleLSTM from the given
        config dictionary.

        Parameters
        ----------
        ``config`` : dict
            Must include 'lstm_units', 'scales',
            'return_sequences'.

        Returns
        -------
        MultiScaleLSTM
            A new instance of this layer.
        """
        return cls(**config)


@register_keras_serializable(
    'fusionlab.nn.components', name='CategoricalEmbeddingProcessor'
)
class CategoricalEmbeddingProcessor(Layer, NNLearner):
    """Embeds multiple categorical features and concatenates them."""
    def __init__(
        self,
        categorical_embedding_info: Dict[int, Tuple[int, int]],
        # Dict mapping feature index to (vocab_size, embedding_dim)
        **kwargs
    ):
        super().__init__(**kwargs)
        self.categorical_embedding_info = categorical_embedding_info
        self.embedding_layers = {}

        # Create Embedding layers based on info provided
        for index, (vocab_size, embed_dim) in \
                self.categorical_embedding_info.items():
            self.embedding_layers[index] = Embedding(
                input_dim=vocab_size,
                output_dim=embed_dim,
                name=f"cat_embed_idx_{index}"
            )

    def call(self, inputs):
        """Applies embedding to specified indices and concatenates."""
        ## XXX TODO
        # Input shape: (Batch, [TimeSteps,] NumCatFeatures)
        # Note: Assumes input `inputs` ONLY contains the categorical features
        #       in the correct order corresponding to the indices in the info dict.
        #       A more robust version might take the full feature tensor and indices.

        embeddings = []
        # Check rank to handle static (2D) vs dynamic/future (3D)
        input_rank = len(inputs.shape)
        num_cat_features_provided = inputs.shape[-1]

        # Validate number of features matches expected keys
        if num_cat_features_provided != len(self.categorical_embedding_info):
             raise ValueError(
                f"Number of input categorical features ({num_cat_features_provided})"
                f" does not match number of embedding layers"
                f" ({len(self.categorical_embedding_info)})."
                 )

        feature_index_counter = 0
        for original_index in sorted(self.embedding_layers.keys()):
            # Assume the input tensor columns are ordered corresponding
            # to the sorted original indices.
            if input_rank == 2: # Static: (Batch, NumCatFeatures)
                feature_tensor = inputs[:, feature_index_counter]
            elif input_rank == 3: # Dynamic/Future: (Batch, Time, NumCatFeatures)
                feature_tensor = inputs[:, :, feature_index_counter]
            else:
                raise ValueError(f"Unsupported input rank: {input_rank}")

            # Apply embedding layer corresponding to the original index
            embed_layer = self.embedding_layers[original_index]
            embeddings.append(embed_layer(feature_tensor))
            feature_index_counter += 1

        # Concatenate embeddings along the last dimension
        if not embeddings:
            return None # Or handle appropriately if no categoricals
        return tf_concat(embeddings, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "categorical_embedding_info": self.categorical_embedding_info,
        })
        return config

    @classmethod
    def from_config(cls, config):
         # Keras serialization handles nested layers like Embedding
        return cls(**config)
    
    
# -----functions --------------------------------------------------------------

@register_keras_serializable(
    'fusionlab.nn.components', 
    name='aggregate_multiscale'
)
def aggregate_multiscale(lstm_output, mode="auto"):
    r"""Aggregate multi-scale LSTM outputs using 
    specified temporal fusion strategy.

    This function implements multiple strategies for combining outputs from
    multi-scale LSTMs operating at different temporal resolutions. Supports
    six aggregation modes: ``average``, ``sum``, ``flatten``, ``concat``,
    ``last`` (default fallback), and ``auto``[1]_.
    Designed for compatibility with ``MultiScaleLSTM`` layer outputs.
    
    See more in :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    lstm_output : list of tf.Tensor or tf.Tensor
        Input features from multi-scale processing:
        - List of 3D tensors [(B, T', U), ...] when ``mode`` != 'auto'
        - Single 2D tensor (B, U*S) when ``mode=None``
        where:
          B = Batch size
          T' = Variable time dimension (scale-dependent)
          U = LSTM units per scale
          S = Number of scales (len(scales))
    mode : {'auto', 'sum', 'average', 'flatten', 'concat', 'last'}, optional
        Aggregation strategy:
        - ``auto`` : (Default) Concatenate last timesteps from each scale
        - ``sum`` : Temporal summation per scale + feature concatenation
        - ``average`` : Temporal mean per scale + feature concatenation
        - ``flatten`` : Flatten all time-feature dimensions (requires equal T')
        - ``concat`` : Feature concatenation + last global timestep
        - ``last`` : Alias for ``auto`` (backward compatibility)

    Returns
    -------
    tf.Tensor
        Aggregated features with shape:
        - (B, U*S) for modes: ``average``, ``sum``, ``last``
        - (B, T'*U*S) for ``flatten`` mode
        - (B, U*S) for ``concat`` mode (last timestep only)
        - (B, U*S) for ``auto`` mode
        
        In sum: 
        - (B, U*S) for ``auto``/``last``, ``sum``, ``average``, ``concat``
        - (B, T'*U*S) for ``flatten`` mode.

    Notes
    -----
    
    * Mode Comparison Table:

    +------------+---------------------+---------------------+-------------------+
    | Mode       | Temporal Handling   | Requirements        | Typical Use Case  |
    +============+=====================+=====================+===================+
    | ``auto``   | Last step per scale | None                | Default choice    |
    | (last)     |                     |                     | for variable T'   |
    +------------+---------------------+---------------------+-------------------+
    | ``sum``    | Full sequence sum   | None                | Emphasize temporal|
    |            | per scale           |                     | accumulation      |
    +------------+---------------------+---------------------+-------------------+
    | ``average``| Full sequence mean  | None                | Smooth temporal   |
    |            | per scale           |                     | patterns          |
    +------------+---------------------+---------------------+-------------------+
    | ``flatten``| Preserve all time   | Equal T' across     | Fixed-length      |
    |            | steps               | scales              | sequence models   |
    +------------+---------------------+---------------------+-------------------+
    | ``concat`` | Last global step    | Equal T' across     | Specialized       |
    |            | of concatenated     | scales              | architectures     |
    |            | features            |                     | with aligned T'   |
    +------------+---------------------+---------------------+-------------------+

    Mathematical Formulation:

    For S scales with outputs :math:`\{\mathbf{X}_s \in \mathbb{R}^{B \times T'_s 
    \times U}\}_{s=1}^S`:

    .. math::
        \text{auto} &: \bigoplus_{s=1}^S \mathbf{X}_s^{(:, T'_s, :)} 
        \quad \text{(Last step concatenation)}
        
        \text{sum} &: \bigoplus_{s=1}^S \sum_{t=1}^{T'_s} \mathbf{X}_s^{(:, t, :)}
        
        \text{average} &: \bigoplus_{s=1}^S \frac{1}{T'_s} \sum_{t=1}^{T'_s} 
        \mathbf{X}_s^{(:, t, :)}
        
        \text{flatten} &: \text{vec}\left( \bigoplus_{s=1}^S \mathbf{X}_s \right)
        
        \text{concat} &: \left( \bigoplus_{s=1}^S \mathbf{X}_s \right)^{(:, T', :)}

    where :math:`\bigoplus` = feature concatenation, :math:`\text{vec}` = flatten.

    * Critical differences between key modes ``'concat'`` and ``'last'``:

    +------------------+---------------------+-----------------------+
    | Aspect           | ``concat``          | ``last`` (default)    |
    +==================+=====================+=======================+
    | Time alignment   | Requires equal T'   | Handles variable T'   |
    +------------------+---------------------+-----------------------+
    | Feature mixing   | Cross-scale mixing  | Scale-independent     |
    +------------------+---------------------+-----------------------+
    | Scale validity   | Only valid when     | Robust to arbitrary   |
    |                  | scales=[1,1,...]    | scale configurations  |
    +------------------+---------------------+-----------------------+
    
    Examples
    --------
    >>> from fusionlab.nn.components import aggregate_multiscale
    >>> import tensorflow as tf
    
    # Three scales with different time dimensions
    >>> outputs = [
    ...     tf.random.normal((32, 10, 64)),  # Scale 1: T'=10
    ...     tf.random.normal((32, 5, 64)),   # Scale 2: T'=5
    ...     tf.random.normal((32, 2, 64))    # Scale 3: T'=2
    ... ]
    
    # Default auto mode (last timesteps)
    >>> agg_auto = aggregate_multiscale(outputs, mode='auto')
    >>> agg_auto.shape
    (32, 192)  # 64 units * 3 scales

    # Last timestep aggregation (default)
    >>> agg_last = aggregate_multiscale(outputs, mode='last')
    >>> print(agg_last.shape)
    (32, 192)
    
    # Flatten mode (requires manual padding for equal T')
    >>> padded_outputs = [tf.pad(o, [[0,0],[0,3],[0,0]]) for o in outputs[:2]] 
    >>> padded_outputs.append(outputs[2])
    >>> agg_flat = aggregate_multiscale(padded_outputs, mode='flatten')
    >>> agg_flat.shape
    (32, 1280)  # (10+3)*64*3 = 13*192 = 2496? Wait need to check dimensions

    See Also
    --------
    MultiScaleLSTM : Base layer producing multi-scale LSTM outputs
    TemporalFusionTransformer : Advanced temporal fusion architecture
    HierarchicalAttention : Alternative temporal aggregation approach

    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2021). Time-series forecasting with deep
       learning: a survey. Philosophical Transactions of the Royal Society A,
       379(2194), 20200209. https://doi.org/10.1098/rsta.2020.0209
    """
    # "auto", use the last LastStep-First Approach
    if mode is None: 
        # No additional aggregation needed
        lstm_features = lstm_output  # (B, units * len(scales))

    # Apply chosen aggregation to full sequences
    elif mode == "average":
        # Average over time dimension for each scale and then concatenate
        averaged_outputs = [
            tf_reduce_mean(o, axis=1) 
            for o in lstm_output
        ]  # Each is (B, units)
        lstm_features = tf_concat(
            averaged_outputs,
            axis=-1
        )  # (B, units * len(scales))

    elif mode== "flatten":
        # Flatten time and feature dimensions for all scales
        # Assume equal time lengths for all scales
        concatenated = tf_concat(
            lstm_output, 
            axis=-1
        )  # (B, T', units*len(scales))
        shape = tf_shape(concatenated)
        (batch_size,
         time_dim,
         feat_dim) = shape[0], shape[1], shape[2]
        lstm_features = tf_reshape(
            concatenated,
            [batch_size, time_dim * feat_dim]
        )
    elif mode =='sum': 
        # Sum over time dimension for each scale and concatenate
        summed_outputs = [
            tf_reduce_sum(o, axis=1) 
            for o in lstm_output
            ]
        lstm_features = tf_concat(
            summed_outputs, axis=-1)
        
    elif mode=="concat": 
        # Concatenate along the feature dimension for each
        # time step and take the last time step
        concatenated = tf_concat(
            lstm_output, axis=-1)  # (B, T', units * len(scales))
        last_output = concatenated[:, -1, :]  # (B, units * len(scales))
        lstm_features = last_output
        
    else: # "last" or "auto"
        # Default fallback: take the last time step from each scale
        # and concatenate
        last_outputs = [
            o[:, -1, :] 
            for o in lstm_output
        ]  # (B, units)
        lstm_features = tf_concat(
            last_outputs,
            axis=-1
        )  # (B, units * len(scales))
    
    return lstm_features 

@register_keras_serializable(
    'fusionlab.nn.components', 
    name='aggregate_time_window_output'
)
def aggregate_time_window_output(
        time_window_output:Tensor,
        mode: Optional[str]=None
    ):
    """
    Aggregates time window output features based on the specified
    aggregation method.

    This function performs the final aggregation on a 3D tensor
    representing temporal features. The aggregation can be done by
    selecting the last time step, computing the average across time,
    or flattening the temporal and feature dimensions into a single
    vector per sample.

    The aggregation methods are defined as follows:

    .. math::
       \text{last: } F = T[:, -1, :]

    .. math::
       \text{average: } F = \frac{1}{T_{dim}} \sum_{i=1}^{T_{dim}}
       T[:, i, :]

    .. math::
       \text{flatten: } F = \text{reshape}(T, (batch\_size,
       time\_dim \times feat\_dim))

    where :math:`T` is the input tensor with shape
    :math:`(batch\_size, time\_dim, feat\_dim)` and :math:`F` is the
    aggregated output.

    Parameters
    ----------
    time_window_output : tf.Tensor
        A 3D tensor of shape :math:`(batch\_size, time\_dim,
        feat\_dim)` representing the output features over time.
    mode : str, optional
        Aggregation method to apply. Supported values are:

        - ``"last"``: Selects the features from the last time step.
        - ``"average"``: Computes the mean of features across
          the time dimension.
        - ``"flatten"``: Flattens the time and feature dimensions
          into a single vector per sample.

        If ``mode`` is `None`, the function falls back to the
        ``flatten`` aggregation method.

    Returns
    -------
    tf.Tensor
        The aggregated features tensor after applying the specified
        aggregation method.

    Raises
    ------
    ValueError
        If an unsupported aggregation method is provided in the
        ``mode`` argument.

    Examples
    --------
    >>> from fusionlab.nn.components import aggregate_time_window_output
    >>> import tensorflow as tf
    >>> # Create a dummy tensor with shape (2, 3, 4)
    >>> dummy = tf.random.uniform((2, 3, 4))
    >>> # Apply average aggregation
    >>> result = aggregate_time_window_output(dummy,
    ...                                      mode="average")

    Notes
    -----
    - The function uses TensorFlow operations to ensure compatibility
      with TensorFlow's computation graph.
    - It is recommended to use this function as part of a larger neural
      network pipeline [1]_.

    See Also
    --------
    tf.reduce_mean
        TensorFlow operation to compute mean along axes.

    References
    ----------
    .. [1] Author Name, "Title of the reference", Journal/Conference,
       Year.

    """
    mode = mode or 'flatten' 
    if mode == "last":
        # Select the features corresponding to the last time step for
        # each sample.
        final_features = time_window_output[:, -1, :]

    elif mode == "average":
        # Compute the mean of the features across the time dimension.
        final_features = tf_reduce_mean(time_window_output, axis=1)

    elif mode == "flatten":
        # Retrieve the dynamic shape of the input tensor.
        shape = tf_shape(time_window_output)
        batch_size, time_dim, feat_dim = (
            shape[0],
            shape[1],
            shape[2]
        )
        # Flatten the time and feature dimensions into a single vector
        # per sample.
        final_features = tf_reshape(
            time_window_output,
            [batch_size, time_dim * feat_dim]
        )

    else:
        # Raise an error if an unsupported aggregation method is provided.
        raise ValueError(
            f"Unsupported mode value: '{mode}'. Supported values are "
            f"'last', 'average', or 'flatten'."
        )

    return final_features

def create_causal_mask(size: Union[int,Tensor]) -> Tensor:
    """
    Creates a causal attention mask of shape [1,1,seq_len,seq_len]
    where mask[0,0,i,j] = 1.0 if j > i else 0.0.
    """
    
    # Make sure size is a 0-D int32 Tensor
    size = tf_cast(size, tf_int32)

    # Build a vector [0,1,2,...,size-1]
    idxs = tf_range(size)                  # shape: [size]

    # Compare row < col for every pair (i,j)
    #   row_idxs: [size,1], col_idxs: [1,size]
    row_idxs = tf_expand_dims(idxs, 1)     # [size,1]
    col_idxs = tf_expand_dims(idxs, 0)     # [1,size]

    # mask2d[i,j] = True if j > i, else False
    mask2d = tf_greater(col_idxs, row_idxs)  # [size,size], dtype=bool

    # Cast to float (1.0 for masked positions, 0.0 elsewhere)
    mask2d = tf_cast(mask2d, tf_float32)     # [size,size]

    # Expand to [1,1,size,size] so it broadcasts over (batch, heads)
    mask = tf_expand_dims(tf_expand_dims(mask2d, 0), 1)

    return mask

# def create_causal_mask(size: Union[int, Tensor]) -> Tensor:
#     """Creates a causal attention mask of shape [1,1,seq_len,seq_len]."""
#     # ensure `size` is an int32 Tensor
#     size = tf_cast(size, tf_int32)

#     # build shape as a Tensor so we don't capture Python tuples of Tensors
#     shape = tf_stack([size, size])
#     ones = tf_ones(shape, dtype=tf_float32)

#     # make the [seq_len, seq_len] causal matrix
#     mask2d = 1.0 - tf_linalg.band_part(ones, -1, 0)

#     # now expand batch dim then head dim → [1,1,seq_len,seq_len]
#     mask = tf_expand_dims(mask2d, 0)  # → [1, seq_len, seq_len]
#     mask = tf_expand_dims(mask, 1)    # → [1, 1, seq_len, seq_len]

#     return mask

def _create_causal_mask(size: Union[int, Tensor]) -> Tensor:
    """Creates a causal attention mask for the decoder."""
    mask = 1 - tf_linalg.band_part(tf_ones((size, size)), -1, 0)
    # Add batch and head dimensions for broadcasting
    return mask[tf_expand_dims(tf_range(size), 0), :] # (1, 1, seq_len, seq_len) -> Keras MHA expects (B, T, T) or (B, N_heads, T, T)
                                                      # TF MHA expects (B, N_heads, T, T)
                                                      # Let's make it (1,1,T,T) for TF MHA layer, it will broadcast
    # # Keras MHA expects mask shape (batch_size, num_heads, query_length, key_length)
    # # or (batch_size, query_length, key_length)
    # # For causal, query_length == key_length == size
    # return tf_expand_dims(tf_expand_dims(
    #     1 - tf_linalg.band_part(tf_ones((size, size)), -1, 0), axis=0), axis=0)


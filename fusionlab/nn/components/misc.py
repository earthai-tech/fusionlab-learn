# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Misc / utility layers & helpers
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np 

from ._config import (
    Layer, Dense, Dropout,
    Tensor, TensorShape,
    activations,
    tf_autograph, 
    tf_cast, 
    tf_float32, 
    tf_range, 
    tf_newaxis,
    tf_exp, tf_log, 
    tf_sin, tf_cos,
    tf_stack, 
    tf_reshape,
    tf_shape,
    tf_concat, 
    tf_pow, 
    tf_floordiv,
    register_keras_serializable,
    DEP_MSG, KERAS_BACKEND,
)

from ..api.property import NNLearner
from ..utils.deps_utils import ensure_pkg

__all__ = [
    "Activation",
    "PositionwiseFeedForward",
    "PositionalEncoding",
    "TSPositionalEncoding",
    "MultiModalEmbedding",
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

@register_keras_serializable(
    "fusionlab.nn.components", 
    name='PositionwiseFeedForward')
class PositionwiseFeedForward(Layer, NNLearner):
    """Implements the Position-wise Feed-Forward Network (FFN) layer.

    This layer is a core component of a standard Transformer block,
    typically applied after the multi-head attention sub-layer. Its
    purpose is to process the context-rich output from the attention
    mechanism at each position independently, adding non-linearity
    and transformative capacity to the model.

    The network consists of two fully-connected (Dense) layers with a
    non-linear activation function in between. The first layer expands
    the input dimensionality, and the second layer projects it back down.

    Parameters
    ----------
    embed_dim : int
        The input and output dimensionality of the layer. This must match
        the embedding dimension of the Transformer, often denoted as
        :math:`d_{model}`.
    ffn_dim : int
        The dimensionality of the inner, expanded hidden layer. It is
        common practice in Transformer architectures to set this to four
        times the `embed_dim`.
    activation : str, optional
        The activation function to use in the inner layer. Any valid
        Keras activation string is accepted. Defaults to ``"relu"``.
    dropout_rate : float, optional
        The dropout rate applied for regularization, typically after the
        first activation function. Defaults to ``0.1``.
    **kwargs
        Standard keyword arguments for a Keras ``Layer``.

    Notes
    -----
    The "position-wise" nature of this layer is its defining
    characteristic. The same instance of this layer, with the exact
    same set of learned weights (:math:`W_1, b_1, W_2, b_2`), is applied
    to the feature vector at every single position (e.g., time step)
    in the input sequence. It does not mix information between positions;
    that task is handled by the preceding self-attention layer.

    The mathematical operation for a single position vector :math:`x` is:
    
    .. math::
       \text{FFN}(x) = \text{Linear}_2(\text{activation}(\text{Linear}_1(x)))

    The residual connection (:math:`x + \text{Dropout}(\text{FFN}(x))`)
    is typically applied outside this layer, within the main
    Transformer block.

    See Also
    --------
    fusionlab.nn.components.TransformerEncoderLayer : A typical consumer of this layer.
    tf.keras.layers.Dense : The core building block of the FFN.

    References
    ----------
    .. [1] Vaswani, A., et al. "Attention Is All You Need." *NeurIPS 2017*.

    Examples
    --------
    >>> import tensorflow as tf
    >>> # Create a dummy input tensor (batch, sequence_length, embed_dim)
    >>> input_tensor = tf.random.normal((32, 50, 128))
    ...
    >>> # Instantiate the FFN layer
    >>> ffn_layer = PositionwiseFeedForward(embed_dim=128, ffn_dim=512)
    ...
    >>> # Pass the input through the layer
    >>> output_tensor = ffn_layer(input_tensor, training=True)
    ...
    >>> # The output shape remains the same as the input shape
    >>> print(f"Input Shape: {input_tensor.shape}")
    >>> print(f"Output Shape: {output_tensor.shape}")
    Input Shape: (32, 50, 128)
    Output Shape: (32, 50, 128)
    """
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Store configuration for serialization
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.activation_str = activation
        self.dropout_rate = dropout_rate

        # Define the internal layers once in the constructor
        self.dense_1 = Dense(
            units=ffn_dim,
            name="ffn_dense_1"
        )
        self.activation = Activation(activation).activation_fn
        self.dense_2 = Dense(
            units=embed_dim,
            name="ffn_dense_2"
        )
        self.dropout = Dropout(rate=dropout_rate)

    def call(self, x: Tensor, training: bool = False) -> Tensor:
        """Defines the forward pass for the FFN layer."""
        # Project to the intermediate dimension
        x = self.dense_1(x)
        # Apply the non-linear activation function
        x = self.activation(x)
        # Apply dropout for regularization
        x = self.dropout(x, training=training)
        # Project back to the original embedding dimension
        x = self.dense_2(x)
        return x

    def get_config(self):
        """Returns the configuration of the layer for serialization."""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "ffn_dim": self.ffn_dim,
            "activation": self.activation_str,
            "dropout_rate": self.dropout_rate,
        })
        return config
    
@register_keras_serializable(
    'fusionlab.nn.components', name='PositionalEncoding')
class PositionalEncoding(Layer, NNLearner):
    r"""Injects positional information into an input tensor.

    This layer adds a positional encoding to the input, allowing models
    like Transformers to understand the order of the sequence. It uses
    the standard sinusoidal encoding from the "Attention Is All You
    Need" paper [1]_.

    The positional encoding :math:`PE` is defined as:

    .. math::
        PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)

    .. math::
        PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)

    where :math:`pos` is the position in the sequence, :math:`i` is the
    dimension index, and :math:`d_{\text{model}}` is the feature dimension.

    Parameters
    ----------
    max_length : int, default 2048
        The maximum possible sequence length. The encoding matrix will be
        pre-calculated up to this length.
    **kwargs
        Standard Keras Layer keyword arguments.

    Examples 
    --------
    >>> import tensorflow as tf 
    >>> from fusionlab.nn.components import PositionalEncoding
    >>> batch_size = 4
    >>> sequence_length = 50
    >>> feature_dimension = 128

    >>> # Create dummy input tensor
    >>> input_tensor = tf.random.normal(
    ...    (batch_size, sequence_length, feature_dimension)
    ... )

    >>> # Instantiate and apply the layer
    >>> pos_encoding_layer = PositionalEncoding(max_length=5000)
    >>> output_tensor = pos_encoding_layer(input_tensor)

    >>> print("Input Tensor Shape:", input_tensor.shape)
    >>> print("Output Tensor Shape:", output_tensor.shape)
    >>> # The shape should be unchanged.
    >>> assert input_tensor.shape == output_tensor.shape

    >>> # You can visualize the encoding if you wish
    >>> import matplotlib.pyplot as plt
    >>> pe_matrix = pos_encoding_layer.positional_encoding[0, :, :].numpy()
    >>> plt.figure(figsize=(10, 5))
    >>> cax = plt.matshow(pe_matrix, fignum=1, aspect='auto', cmap='viridis')
    >>> plt.gcf().colorbar(cax)
    >>> plt.title("Sinusoidal Positional Encoding Matrix")
    >>> plt.xlabel("Feature Dimension")
    >>> plt.ylabel("Position in Sequence")
    >>> plt.show()
    
    References
    ----------
    .. [1] Vaswani, A., et al. (2017). "Attention is all you need."
           *Advances in Neural Information Processing Systems*, 30.
    """
    def __init__(self, max_length: int = 2048, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.positional_encoding = None

    def build(self, input_shape: TensorShape):
        """Pre-calculates the positional encoding matrix."""
        # The input shape is (batch, sequence_length, feature_dim)
        _, _, feature_dim = input_shape
        
        if self.positional_encoding is None:
            # The calculation is done once and stored.
            # Ensure feature_dim is a concrete value for matrix creation.
            if feature_dim is None:
                raise ValueError(
                    "The feature dimension of the input to "
                    "PositionalEncoding cannot be `None`. Please "
                    "ensure the input has a defined feature dimension."
                )

            # Cast to float for calculations
            d_model = tf_cast(feature_dim, tf_float32)

            # Create a matrix of positions (max_length, 1)
            positions = tf_range(
                self.max_length, dtype=tf_float32)[:, tf_newaxis]

            # Create the division term for the sine/cosine functions
            # Shape: (feature_dim / 2)
            div_term = tf_exp(
                tf_range(0, feature_dim, 2, dtype=tf_float32) * \
                (-tf_log(10000.0) / d_model)
            )

            # Calculate sinusoidal values for even and odd indices
            # Shape of each: (max_length, feature_dim / 2)
            pe_sin = tf_sin(positions * div_term)
            pe_cos = tf_cos(positions * div_term)

            # Interleave sin and cos values to get final encoding
            # Resulting shape: (max_length, feature_dim)
            pe_interleaved = tf_reshape(
                tf_stack([pe_sin, pe_cos], axis=-1),
                shape=[self.max_length, feature_dim]
            )

            # Add an extra dimension for broadcasting across the batch
            # Shape: (1, max_length, feature_dim)
            self.positional_encoding = pe_interleaved[tf_newaxis, :, :]

        super().build(input_shape)

    def call(self, inputs: Tensor, training=False ) -> Tensor:
        r"""Adds positional encoding to the input tensor.
        
        The 'training' argument is accepted but not used.
        This ensures API compatibility with Keras.
        
        Parameters
        ----------
        inputs : tf.Tensor
            A 3D tensor of shape :math:`(B, T, D)`, where ``B`` is
            the batch size, ``T`` is the sequence length, and ``D``
            is the feature dimension.

        Returns
        -------
        tf.Tensor
            The input tensor with positional encodings added.
            Shape: :math:`(B, T, D)`.
        Notes 
        ------
        The Positional encoding does not depends on training. 
        The sinusoidal PositionalEncoding layer performs a deterministic 
        mathematical operation. It calculates a fixed matrix of sine and 
        cosine values based on position and feature dimension and simply 
        adds it to the input. This calculation is the same whether you are 
        training the model or running it for inference. Unlike layers such 
        as Dropout or BatchNormalization, PositionalEncoding has no different
        behavior during training.
        
        """
        # Get the sequence length of the current input batch.
        seq_len = tf_shape(inputs)[1]
        
        # Slice the pre-calculated encoding matrix to match the input
        # sequence length and add it to the input tensor.
        # The broadcasting mechanism will handle the batch dimension.
        return inputs + self.positional_encoding[:, :seq_len, :]

    def get_config(self) -> dict:
        """Returns the configuration of the layer."""
        config = super().get_config()
        config.update({
            'max_length': self.max_length,
        })
        return config

    
@register_keras_serializable(
    'fusionlab.nn.components', name="TSPositionalEncoding")
class TSPositionalEncoding(Layer, NNLearner):
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
        self.pos_encoding = self._build_positional_encoding(
            max_position, embed_dim)

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

    def call(self, x, training=False):
        """Adds positional encoding to the input tensor `x`.
        The 'training' argument is accepted but not used.
        This ensures API compatibility with Keras.
        """
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
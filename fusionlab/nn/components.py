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
from numbers import Real, Integral  
from typing import Optional, Union, List 

from ..api.docstring import DocstringComponents
from ..api.property import  NNLearner 
from ..core.checks import validate_nested_param
from ..compat.sklearn import validate_params, Interval
from ..utils.deps_utils import ensure_pkg

from . import KERAS_DEPS, KERAS_BACKEND, dependency_message
from ._nn_docs import _shared_nn_params
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

    tf_concat = KERAS_DEPS.concat
    tf_shape = KERAS_DEPS.shape
    tf_reshape=KERAS_DEPS.reshape
    tf_add = KERAS_DEPS.add
    tf_maximum = KERAS_DEPS.maximum
    tf_reduce_mean = KERAS_DEPS.reduce_mean
    tf_add_n = KERAS_DEPS.add_n
    tf_float32=KERAS_DEPS.float32
    tf_constant=KERAS_DEPS.constant 
    tf_square=KERAS_DEPS.square 
    tf_autograph=KERAS_DEPS.autograph
    tf_reduce_sum = KERAS_DEPS.reduce_sum
    tf_stack = KERAS_DEPS.stack
    tf_expand_dims = KERAS_DEPS.expand_dims
    tf_tile = KERAS_DEPS.tile
    tf_range=KERAS_DEPS.range 
    tf_concat = KERAS_DEPS.concat
    tf_shape = KERAS_DEPS.shape
    tf_rank=KERAS_DEPS.rank
    
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
     'aggregate_multiscale', 
     'aggregate_time_window_output'
    ]

_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_shared_nn_params), 
)

@register_keras_serializable('gofast.nn.components', name="Activation")
class Activation(Layer):
    """
    Custom Activation layer that wraps a Keras activation function
    and captures its name.
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, activation='relu', **kw):
        super().__init__(**kw)
        # Get the activation function; Keras
        # will raise an error if invalid
        self.activation= activations.get(activation)
        # self.activation = activation  
        # Store the original activation parameter

        # Assign activation name
        if isinstance(activation, str):
            self.activation_name = activation
        elif callable(activation):
            # Try to get the name from the activation function
            self.activation_name = getattr(
                activation, '__name__', 
                activation.__class__.__name__
            )
        else:
            # Fallback to string representation
            self.activation_name = str(activation)
            
    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        r"""
        Forward pass of the Activation layer.
    
        Applies the stored activation function to the input tensor.
        The ``training`` parameter is accepted for API compatibility but
        is not used in this implementation.
    
        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor on which the activation is applied.
        training : bool, optional
            Boolean flag indicating whether the layer is in training mode.
            This argument is ignored. Default is ``False``.
    
        Returns
        -------
        tf.Tensor
            The output tensor after applying the activation function.
        """
        return self.activation(inputs)


    def get_config(self):
        config = super(Activation, self).get_config()
        # Serialize the activation function properly
        config.update({
            'activation': activations.serialize(self.activation)
        })
        return config

    def __repr__(self):
        return ( 
            f"{self.__class__.__name__}("
            f"activation={self.activation_name!r})"
        )

# -------------------- TFT components ----------------------------------------

@register_keras_serializable('gofast.nn.components', name='PositionalEncoding')
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
    >>> from gofast.nn.components import PositionalEncoding
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
    'gofast.nn.components', name="GatedResidualNetwork"
)
class GatedResidualNetwork(Layer, NNLearner):
    r"""
    Gated Residual Network (GRN) for deep feature 
    transformation with gating and residual 
    connections [1]_.

    This layer captures complex nonlinear relationships 
    by applying two linear transformations with a 
    specified `activation`, followed by gating and a 
    residual skip connection. An optional batch 
    normalization can be included. The shape of the 
    output can match the input if a projection is 
    applied.

    .. math::
        \mathbf{h} = \text{LayerNorm}\Big(
            \mathbf{x} + \big(\mathbf{W}_2
            \,\phi(\mathbf{W}_1\,\mathbf{x} + 
            \mathbf{b}_1)\,\mathbf{g}\big)
        \Big)

    where :math:`\mathbf{g}` is the output of the gate.

    Parameters
    ----------
    units : int
        Number of hidden units in the GRN.
    dropout_rate : float, optional
        Dropout rate used after the second linear 
        transformation. Defaults to 0.0.
    use_time_distributed : bool, optional
        Whether to wrap this layer with 
        ``TimeDistributed`` for temporal data. 
        Defaults to False.
    activation : str, optional
        Activation function to use. Must be one 
        of {'elu', 'relu', 'tanh', 'sigmoid', 
        'linear'}. Defaults to 'elu'.
    use_batch_norm : bool, optional
        Whether to apply batch normalization 
        after the first linear transformation. 
        Defaults to False.
    **kwargs : 
        Additional arguments passed to the 
        parent Keras ``Layer``.

    Notes
    -----
    - The gating mechanism is used to control 
      the contribution of the transformed 
      features to the output.
    - The residual connection helps in training 
      deeper networks by mitigating vanishing 
      gradient issues.

    Methods
    -------
    call(`x`, training=False)
        Forward pass of the GRN. Accepts an 
        input tensor ``x`` of shape 
        (batch_size, ..., input_dim).

    get_config()
        Returns the configuration dictionary 
        for serialization.

    from_config(`config`)
        Creates a new GRN from a given 
        configuration dictionary.

    Examples
    --------
    >>> from gofast.nn.components import GatedResidualNetwork
    >>> import tensorflow as tf
    >>> # Create a random input of shape
    ... # (batch_size, time_steps, input_dim)
    >>> inputs = tf.random.normal((32, 10, 64))
    >>> # Instantiate GRN
    >>> grn = GatedResidualNetwork(
    ...     units=64,
    ...     dropout_rate=0.1,
    ...     use_time_distributed=True,
    ...     activation='relu',
    ...     use_batch_norm=True
    ... )
    >>> # Forward pass
    >>> outputs = grn(inputs)

    See Also
    --------
    VariableSelectionNetwork 
        Utilizes GRN to process multiple 
        input variables.
    PositionalEncoding 
        Provides positional encoding for 
        sequence inputs.

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
        "dropout_rate": [Interval(Real, 0, 1, 
                                  closed="both")],
        "use_time_distributed": [bool],
        "use_batch_norm": [bool],
    })
    @ensure_pkg(KERAS_BACKEND or "keras", 
                extra=DEP_MSG)
    def __init__(
        self,
        units,
        dropout_rate=0.0,
        use_time_distributed=False,
        activation='elu',
        use_batch_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed
        self.use_batch_norm = use_batch_norm

        # Store activation as an object
        self.activation= activation

        # First linear transform
        self.linear = Dense(
            units, 
            activation=self.activation
        )
        # Second linear transform
        self.linear2 = Dense(
            units, 
            activation= self.activation 
        )

        # Optionally apply batch normalization
        if self.use_batch_norm:
            self.batch_norm = BatchNormalization()

        # Dropout layer
        self.dropout = Dropout(
            dropout_rate
        )

        # Layer normalization for the output
        self.layer_norm = LayerNormalization()

        # Gate for controlling feature flow
        self.gate = Dense(
            units,
            activation='sigmoid'
        )

        # Projection for matching dimensions if needed
        self.projection = None

    def build(self, input_shape):
        r"""
        Build method that creates the projection 
        layer if the input dimension does not 
        match `units`.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor,
            typically (batch_size, ..., input_dim).
        """
        input_dim = input_shape[-1]

        # Create projection only if input_dim != units
        if input_dim != self.units:
            self.projection = Dense(
                self.units
            )
        super().build(input_shape)
        
    @tf_autograph.experimental.do_not_convert
    def call(self, x, training=False):
        r"""
        Forward pass of the GRN, which applies:
        1) Two linear transformations with a 
           specified activation,
        2) An optional batch normalization, 
        3) A gating mechanism, 
        4) A residual skip connection, 
        5) Layer normalization.

        Parameters
        ----------
        ``x`` : tf.Tensor
            Input tensor of shape 
            :math:`(B, ..., \text{input_dim})`.
        training : bool, optional
            Indicates whether the layer is 
            in training mode for dropout 
            and batch normalization.

        Returns
        -------
        tf.Tensor
            Output tensor of shape 
            :math:`(B, ..., \text{units})`.
        """
        # Save reference for residual
        shortcut = x

        # First linear transform
        x = self.linear(x)

        # # Activation
        # x = self.activation(x)

        # Batch normalization if enabled
        if self.use_batch_norm:
            x = self.batch_norm(
                x,
                training=training
            )

        # Second linear transform
        x = self.linear2(x)

        # Dropout
        x = self.dropout(
            x,
            training=training
        )

        # Gate
        gate_output = self.gate(x)

        # Multiply by gate output
        x = x * gate_output

        # If dimensions differ, apply projection
        if self.projection is not None:
            shortcut = self.projection(shortcut)

        # Residual connection
        x = x + shortcut

        # Layer normalization
        x = self.layer_norm(x)
        return x

    def get_config(self):
        r"""
        Return the configuration dictionary 
        of the GRN.

        Returns
        -------
        dict
            Configuration dictionary containing 
            parameters that define this layer.
        """
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'use_time_distributed': (
                self.use_time_distributed
            ),
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm,
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Create a new instance of this layer 
        from a given config dictionary.

        Parameters
        ----------
        ``config`` : dict
            Configuration dictionary as 
            returned by ``get_config``.

        Returns
        -------
        GatedResidualNetwork
            A new instance of 
            GatedResidualNetwork.
        """
        return cls(**config)


@register_keras_serializable(
    'gofast.nn.components', 
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
    >>> from gofast.nn.components import StaticEnrichmentLayer
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
            static_context_vector,
            temporal_features,
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
            static_context_vector,
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


@register_keras_serializable(
    'gofast.nn.components', 
    name="TemporalAttentionLayer"
)
class TemporalAttentionLayer(Layer, NNLearner):
    r"""
    Temporal Attention Layer for focusing on
    important time steps [1]_.

    This layer uses a multi-head attention
    mechanism on temporal sequences to learn
    which time steps are most relevant for
    prediction. It also enriches queries
    with a static context via a 
    :class:`GatedResidualNetwork`, then applies
    multi-head attention followed by a second
    GRN to refine the attended sequence.

    .. math::
        \mathbf{Z} = \text{GRN}\Big(
            \mathbf{X} + \text{MHA}\big(\mathbf{Q},
            \mathbf{X}, \mathbf{X}\big)\Big)

    where :math:`\mathbf{Q}` is the sum of
    :math:`\mathbf{X}` and the tiled static
    context.

    Parameters
    ----------
    units : int
        Dimensionality of the query/key/value
        projections within multi-head attention,
        as well as hidden units in the GRN.
    num_heads : int
        Number of attention heads.
    dropout_rate : float, optional
        Dropout rate applied in multi-head
        attention and in the GRNs. Defaults
        to 0.0.
    activation : str, optional
        Activation function used in the GRNs.
        Must be one of {'elu', 'relu', 'tanh',
        'sigmoid', 'linear'}. Defaults to
        ``'elu'``.
    use_batch_norm : bool, optional
        Whether to apply batch normalization
        in the GRNs. Defaults to ``False``.
    **kwargs :
        Additional arguments passed to the
        parent Keras ``Layer``.

    Notes
    -----
    1. The static context is first transformed
       by a GRN, expanded, and added to the
       input sequence to form the query.
    2. Multi-head attention is performed
       between the query, key, and value
       (all set to `inputs`).
    3. The result is normalized and passed
       through another GRN for final 
       transformation.

    Methods
    -------
    call(`inputs`, `context_vector`, training=False)
        Forward pass of the temporal attention
        layer.

    get_config()
        Returns configuration dictionary for
        serialization.

    from_config(`config`)
        Creates a new instance from a config
        dictionary.

    Examples
    --------
    >>> from gofast.nn.components import TemporalAttentionLayer
    >>> import tensorflow as tf
    >>> # Create random inputs
    ... # shape (batch_size, time_steps, units)
    >>> inputs = tf.random.normal((32, 10, 64))
    >>> # Create static context vector
    ... # shape (batch_size, units)
    >>> context_vector = tf.random.normal((32, 64))
    >>> # Instantiate the layer
    >>> tal = TemporalAttentionLayer(
    ...     units=64,
    ...     num_heads=4,
    ...     dropout_rate=0.1,
    ...     activation='relu',
    ...     use_batch_norm=True
    ... )
    >>> # Forward pass
    >>> outputs = tal(
    ...     inputs,
    ...     context_vector,
    ...     training=True
    ... )

    See Also
    --------
    GatedResidualNetwork
        Provides transformation for the
        query and the final output.
    TemporalFusionTransformer
        A composite model utilizing temporal 
        attention for time-series forecasting.

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
        "num_heads": [Interval(Integral, 1, None,
                               closed='left')],
        "dropout_rate": [Interval(Real, 0, 1,
                                  closed="both")],
        "use_batch_norm": [bool],
    })
    @ensure_pkg(KERAS_BACKEND or "keras",
                extra=DEP_MSG)
    def __init__(
            self,
            units,
            num_heads,
            dropout_rate=0.0,
            activation='elu',
            use_batch_norm=False,
            **kwargs
    ):
        r"""
        Initialize the TemporalAttentionLayer.

        Parameters
        ----------
        units : int
            Dimensionality for query/key/value
            projections and hidden units in 
            the GRNs.
        num_heads : int
            Number of attention heads in
            multi-head attention.
        dropout_rate : float, optional
            Dropout rate for both multi-head
            attention and GRNs. Defaults to 0.0.
        activation : str, optional
            Activation used in the internal GRNs.
            Defaults to ``'elu'``.
        use_batch_norm : bool, optional
            Whether to apply batch normalization
            in the GRNs. Defaults to ``False``.
        **kwargs :
            Additional arguments passed to
            the parent Keras ``Layer``.
        """
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # Create activation object
        self.activation = activation

        self.use_batch_norm = use_batch_norm

        # Multi-head attention
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units,
            dropout=dropout_rate
        )

        # Dropout
        self.dropout = Dropout(dropout_rate)

        # Layer normalization for residual block
        self.layer_norm = LayerNormalization()

        # GRN to transform input prior 
        # to multi-head attention
        self.grn = GatedResidualNetwork(
            units,
            dropout_rate,
            use_time_distributed=True,
            activation=self.activation,
            use_batch_norm=use_batch_norm
        )

        # GRN to transform context vector
        self.context_grn = GatedResidualNetwork(
            units,
            dropout_rate,
            activation=self.activation,
            use_batch_norm=use_batch_norm
        )
    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, context_vector, training=False):
        r"""
        Forward pass of the temporal attention layer.

        Parameters
        ----------
        ``inputs`` : tf.Tensor
            A 3D tensor of shape 
            :math:`(B, T, U)`, where ``B`` is the
            batch size, ``T`` is the time dimension,
            and ``U`` is the feature dimension
            (matching `units`).
        ``context_vector`` : tf.Tensor
            A 2D tensor of shape 
            :math:`(B, U)`, representing 
            static context to enrich the 
            query.
        training : bool, optional
            Whether the layer is in training mode
            (e.g. for dropout). Defaults to ``False``.

        Returns
        -------
        tf.Tensor
            A 3D tensor of shape :math:`(B, T, U)`,
            representing the output after temporal
            attention and the final GRN.

        Notes
        -----
        1. The context vector is transformed by
           a GRN and expanded to shape (B, T, U).
        2. This expanded context is added to
           `inputs` to form the attention query.
        3. Multi-head attention is applied with
           query, key, and value all set to 
           `inputs`.
        4. The result is normalized and then
           passed through another GRN for the
           final output.
        """
        # Transform context vector
        context_vector = self.context_grn(
            context_vector,
            training=training
        )

        # Expand and tile context
        context_expanded = tf_expand_dims(
            context_vector,
            axis=1
        )
        context_expanded = tf_tile(
            context_expanded,
            [1, tf_shape(inputs)[1], 1]
        )

        # Combine with inputs to form query
        query = inputs + context_expanded

        # Apply multi-head attention
        attn_output = self.multi_head_attention(
            query=query,
            value=inputs,
            key=inputs,
            training=training
        )

        # Dropout on attention output
        attn_output = self.dropout(
            attn_output,
            training=training
        )

        # Residual connection + layer norm
        x = self.layer_norm(
            inputs + attn_output
        )

        # Final GRN
        output = self.grn(
            x,
            training=training
        )
        return output

    def get_config(self):
        r"""
        Return the configuration dictionary for
        serialization.

        Returns
        -------
        dict
            Configuration parameters for
            this layer.
        """
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm,
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Create a new instance of 
        TemporalAttentionLayer from a
        given config dictionary.

        Parameters
        ----------
        ``config`` : dict
            Configuration dictionary, as 
            returned by ``get_config``.

        Returns
        -------
        TemporalAttentionLayer
            A new instance of 
            TemporalAttentionLayer.
        """
        return cls(**config)
    
    
# -------------------- XTFT components ----------------------------------------

@register_keras_serializable(
    'gofast.nn.components', name="LearnedNormalization"
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
    >>> from gofast.nn.components import LearnedNormalization
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
    def __init__(self):
        super().__init__()

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
    'gofast.nn.components', name="MultiModalEmbedding"
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
    >>> from gofast.nn.components import MultiModalEmbedding
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
    'gofast.nn.components', name="HierarchicalAttention"
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
    >>> from gofast.nn.components import HierarchicalAttention
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
    'gofast.nn.components', name="CrossAttention"
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
    >>> from gofast.nn.components import CrossAttention
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
    'gofast.nn.components', 
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
    >>> from gofast.nn.components import MemoryAugmentedAttention
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
    'gofast.nn.components', 
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
    >>> from gofast.nn.components import AdaptiveQuantileLoss
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
    'gofast.nn.components',
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
    >>> from gofast.nn.components import AnomalyLoss
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
    'gofast.nn.components', 
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
    >>> from gofast.nn.components import (
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
    'gofast.nn.components', 
    name="VariableSelectionNetwork"
)
class VariableSelectionNetwork(Layer, NNLearner):
    r"""
    VariableSelectionNetwork is designed to handle multiple
    input variables (static, dynamic, or future covariates)
    by passing each variable through its own
    GatedResidualNetwork (GRN). Then, a learned
    importance weighting is applied to combine these
    variable-specific embeddings into a single output
    representation.

    The user can choose whether the network should treat
    inputs as time-distributed (e.g., for dynamic
    or future inputs with time steps) by setting 
    `use_time_distributed`. In time-distributed mode,
    the layer expects a rank-4 input
    :math:`(B, T, N, F)`, or rank-3 input
    :math:`(B, T, N)` which is expanded to
    :math:`(B, T, N, 1)`. In non-time-distributed mode,
    the layer expects a rank-3 input
    :math:`(B, N, F)`, or rank-2 input :math:`(B, N)`
    which is expanded to :math:`(B, N, 1)`.

    Mathematically, let :math:`h_i` be the GRN output
    of the i-th variable, and let :math:`\alpha_i`
    be its learned importance weight. The final output
    :math:`o` can be written as:

    .. math::
        o = \sum_{i=1}^{N} \alpha_i h_i

    where

    .. math::
        \alpha_i = 
          \frac{\exp(\mathbf{w}^\top h_i)}
               {\sum_{j=1}^{N}\exp(\mathbf{w}^\top h_j)}

    Here, :math:`\mathbf{w}` is trained via a 
    ``variable_importance_dense`` layer of shape (1).

    Parameters
    ----------
    num_inputs : int
        Number of distinct input variables (N). Each
        variable is processed separately within its own GRN.
    units : int
        Number of hidden units in each GatedResidualNetwork (GRN).
    dropout_rate : float, optional
        Dropout rate used in each GRN. Defaults to 0.0.
    use_time_distributed : bool, optional
        If `use_time_distributed` is True, the input is
        interpreted as (batch, time_steps, num_inputs, features).
        Otherwise, the input is interpreted as
        (batch, num_inputs, features). Defaults to False.
    activation : str, optional
        Activation function to use inside each GRN.
        One of {'elu', 'relu', 'tanh', 'sigmoid', 'linear'}.
        Defaults to 'elu'.
    use_batch_norm : bool, optional
        Whether to apply batch normalization within each GRN.
        Defaults to False.
    **kwargs : 
        Additional keyword arguments passed to the parent
        Keras ``Layer``.

    Notes
    -----
    - This layer is often used within TFT/XTFT-based models
      to learn variable-specific representations and their
      relative importance.
    - Each GRN is defined in the class
      `GatedResidualNetwork` which applies a nonlinear
      transformation followed by a gating mechanism
      for flexible feature extraction.

    Examples
    --------
    >>> from gofast.nn.components import VariableSelectionNetwork
    >>> vsn = VariableSelectionNetwork(
    ...     num_inputs=5,
    ...     units=32,
    ...     dropout_rate=0.1,
    ...     use_time_distributed=False,
    ...     activation='relu',
    ...     use_batch_norm=True
    ... )

    See Also
    --------
    GatedResidualNetwork
        Implements the internal GRN used for variable
        transformation.
    SuperXTFT
        Enhanced version of XTFT using variable 
        selection networks (VSNs) for input features.

    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2020). "Time Series
           Forecasting With Deep Learning: A Survey."
           Phil. Trans. R. Soc. A, 379(2194),
           20200209.
    """
    @validate_params({
        "num_inputs": [Interval(Integral, 1, None, 
                                closed='left')],
        "units": [Interval(Integral, 1, None, 
                           closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, 
                                  closed="both")],
        "use_time_distributed": [bool],
        "use_batch_norm": [bool],
    })
    @ensure_pkg(KERAS_BACKEND or "keras", 
                extra=DEP_MSG)
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

        # Create the activation object from its name
        self.activation = activation

        # Build one GRN for each variable
        self.single_variable_grns = [
            GatedResidualNetwork(
                units=units,
                dropout_rate=dropout_rate,
                use_time_distributed=False,
                activation=self.activation,
                use_batch_norm=use_batch_norm
            )
            for _ in range(num_inputs)
        ]
        
        # Dense layer to compute variable importances
        self.variable_importance_dense = Dense(
            1,
            name="variable_importance"
        )

        # Softmax for normalizing variable weights
        self.softmax = Softmax(axis=-2)

    def call(self, inputs, training=False):
        r"""
        Execute the forward pass.

        The method processes each variable in `inputs`
        with its own GRN and then applies a learned
        importance weighting to combine them. The shape
        of `inputs` is determined by `use_time_distributed`
        and the rank of `inputs`.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor representing either static or 
            dynamic/future time-distributed input(s).
            - If `use_time_distributed` is True, 
              expects rank-4 data 
              (B, T, N, F) or rank-3 data 
              (B, T, N) expanded to 
              (B, T, N, 1).
            - If `use_time_distributed` is False,
              expects rank-3 data 
              (B, N, F) or rank-2 data (B, N) 
              expanded to (B, N, 1).
        training : bool, optional
            Indicates if layer should behave in 
            training mode (e.g., for dropout).
            Defaults to False.

        Returns
        -------
        outputs : tf.Tensor
            A tensor of shape 
            :math:`(B, T, units)` if 
            `use_time_distributed` is True,
            otherwise :math:`(B, units)` for 
            the non-time-distributed case. 
            Each variable-specific embedding 
            is weighted by the variable-level
            importance scores.
        """
        # rank = tf_rank(inputs)
        actual_rank=inputs.shape.rank 
        
        var_outputs = []

        # Case 1: time-distributed
        if self.use_time_distributed:
            if actual_rank == 3:
                # Expand last dim if necessary
                # (B, T, N) -> (B, T, N, 1)
                inputs = tf_expand_dims(inputs, axis=-1)

            # Now we assume rank == 4 => (B, T, N, F)
            for i in range(self.num_inputs):
                var_input = inputs[:, :, i, :]
                grn_output = self.single_variable_grns[i](
                    var_input,
                    training=training
                )
                var_outputs.append(grn_output)
        else:
            # Case 2: non-time-distributed
            if actual_rank == 2:
                # Expand if shape => (B, N)
                # becomes => (B, N, 1)
                inputs = tf_expand_dims(inputs, axis=-1)

            # Now we assume rank == 3 => (B, N, F)
            for i in range(self.num_inputs):
                var_input = inputs[:, i, :]
                grn_output = self.single_variable_grns[i](
                    var_input,
                    training=training
                )
                var_outputs.append(grn_output)

        # Stack variable outputs => 
        # shape (B, T?, N, units)
        stacked_outputs = tf_stack(
            var_outputs,
            axis=-2
        )

        # Compute importances => shape (B, T?, N, 1)
        self.variable_importances_ = (
            self.variable_importance_dense(
                stacked_outputs
            )
        )

        # Normalize across the variable dimension => -2
        weights = self.softmax(
            self.variable_importances_
        )

        # Weighted sum => shape (B, T?, units)
        outputs = tf_reduce_sum(
            stacked_outputs * weights,
            axis=-2
        )
        return outputs

    def get_config(self):
        r"""
        Get the configuration of this layer.

        Returns
        -------
        config : dict
            Dictionary containing the initialization
            parameters of this layer.
        """
        config = super().get_config()
        config.update({
            'num_inputs': self.num_inputs,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'use_time_distributed': self.use_time_distributed,
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm,
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Instantiate a new layer from its config.

        This method creates a `VariableSelectionNetwork`
        layer using the dictionary returned by
        ``get_config``.

        Parameters
        ----------
        config : dict
            Configuration dictionary typically produced
            by ``get_config``.

        Returns
        -------
        VariableSelectionNetwork
            A new instance of this class with the
            specified configuration.
        """
        return cls(**config)

@register_keras_serializable(
    'gofast.nn.components', 
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
    >>> from gofast.nn.components import ExplainableAttention
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
    'gofast.nn.components', 
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
    >>> from gofast.nn.components import MultiDecoder
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
    'gofast.nn.components', 
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
    >>> from gofast.nn.components import MultiResolutionAttentionFusion
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
    'gofast.nn.components',
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
    >>> from gofast.nn.components import DynamicTimeWindow
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
    'gofast.nn.components', 
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
    >>> from gofast.nn.components import QuantileDistributionModeling
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
    'gofast.nn.components', 
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
    >>> from gofast.nn.components import MultiScaleLSTM
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

# -----functions --------------------------------------------------------------

@register_keras_serializable(
    'gofast.nn.components', 
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
    >>> from gofast.nn.components import aggregate_multiscale
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
    'gofast.nn.components', 
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
    >>> from gofast.nn.components import aggregate_time_window_output
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

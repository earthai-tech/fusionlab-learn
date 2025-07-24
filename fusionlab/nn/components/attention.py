# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Attention-centric layers for FusionLab.
"""

from __future__ import annotations
from typing import Optional
from numbers import Real, Integral

from ...api.property import NNLearner
from ...compat.sklearn import validate_params, Interval
from ...utils.deps_utils import ensure_pkg
from ._config import (                                  
    KERAS_BACKEND, 
    DEP_MSG, 
    _logger,
    LayerNormalization,
    MultiHeadAttention, 
    Dropout, 
    Dense, 
    Layer, 
    Tensor, 
    register_keras_serializable,
    tf_shape, 
    tf_expand_dims, 
    tf_tile, 
    tf_bool, 
    tf_add, 
    tf_cast,
    tf_logical_and, 
    tf_ones_like, 
    tf_ones, 
    tf_autograph 
)
from .gating_norm import GatedResidualNetwork
from .misc import Activation


__all__ = [
    "TemporalAttentionLayer",
    "CrossAttention",
    "MemoryAugmentedAttention",
    "HierarchicalAttention",
    "ExplainableAttention",
    "MultiResolutionAttentionFusion",
]

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
    name="CrossAttention_"
)
class CrossAttention_(Layer, NNLearner):
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
    'fusionlab.nn.components', name="CrossAttention"
)
class CrossAttention(Layer, NNLearner):
    r"""
    CrossAttention that attends ``source1`` (query) to ``source2``
    (key/value) with optional masks.
   
    attention_mask : Tensor, optional
        Bool / 0‑1 mask broadcastable to (B, Tq, Tv). Passed
        directly to Keras ``MultiHeadAttention``.
    query_mask, value_mask : Tensor, optional
        1D/2D masks (B, Tq) or (B, Tv). If provided and
        ``attention_mask`` is None, they are combined to form
        (B, Tq, Tv).
    use_causal_mask : bool
        Forwarded to MHA. Default False.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, units: int, num_heads: int):
        super().__init__()
        self.units = units
        self.source1_dense = Dense(units)
        self.source2_dense = Dense(units)
        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=units
        )

    @tf_autograph.experimental.do_not_convert
    def call(
        self,
        inputs,
        training: bool = False,
        *,
        attention_mask: Optional[Tensor] = None,
        query_mask: Optional[Tensor] = None,
        value_mask: Optional[Tensor] = None,
        use_causal_mask: bool = False,
        **kwargs,
    ):
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
        
        attention_mask : Tensor, optional
            Bool / 0‑1 mask broadcastable to (B, Tq, Tv). Passed
            directly to Keras ``MultiHeadAttention``.
        query_mask, value_mask : Tensor, optional
            1D/2D masks (B, Tq) or (B, Tv). If provided and
            ``attention_mask`` is None, they are combined to form
            (B, Tq, Tv).
        use_causal_mask : bool
            Forwarded to MHA. Default False.

        Returns
        -------
        tf.Tensor
            A tensor of shape (batch_size, time_steps,
            units) representing cross-attended features.
        """
        
        source1, source2 = inputs  # shapes: (B, Tq, Fq), (B, Tv, Fv)

        # Project to common dim
        q = self.source1_dense(source1)
        kv = self.source2_dense(source2)

        # Build attention_mask if needed
        if attention_mask is None and (query_mask is not None
                                       or value_mask is not None):
            # default to all True if one side is None
            if query_mask is None:
                query_mask = tf_ones_like(
                    source1[..., 0], dtype=tf_bool)
            if value_mask is None:
                value_mask = tf_ones_like(
                    source2[..., 0], dtype=tf_bool)

            qm = tf_expand_dims(tf_cast(query_mask, tf_bool), axis=-1)
            vm = tf_expand_dims(tf_cast(value_mask, tf_bool), axis=1)
            # (B, Tq, 1) & (B, 1, Tv) -> (B, Tq, Tv)
            attention_mask = tf_logical_and(qm, vm)

        return self.cross_attention(
            query=q,
            key=kv,
            value=kv,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            training=training,
        )

    def get_config(self):
        cfg = super().get_config().copy()
        cfg.update({'units': self.units})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(
    'fusionlab.nn.components', 
    name="MemoryAugmentedAttention_"
)
class MemoryAugmentedAttention_(Layer, NNLearner):
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
    'fusionlab.nn.components', name="MemoryAugmentedAttention"
)
class MemoryAugmentedAttention(Layer, NNLearner):
    r"""Memory-augmented attention with optional masking."""

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, units: int, memory_size: int, num_heads: int):
        super().__init__()
        self.units = units
        self.memory_size = memory_size
        self.attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=units
        )

    def build(self, input_shape):
        self.memory = self.add_weight(
            "memory",
            shape=(self.memory_size, self.units),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    @tf_autograph.experimental.do_not_convert
    def call(
        self,
        inputs,
        training: bool = False,
        *,
        attention_mask: Optional[Tensor] = None,
        query_mask: Optional[Tensor] = None,
        value_mask: Optional[Tensor] = None,
        use_causal_mask: bool = False,
        **kwargs,
    ):
        # inputs: (B, T, U)
        batch_size = tf_shape(inputs)[0]

        mem = tf_expand_dims(self.memory, 0)          # (1, M, U)
        mem = tf_tile(mem, [batch_size, 1, 1])        # (B, M, U)

        # Build attention_mask if only per-sequence masks given
        if attention_mask is None and (query_mask is not None
                                       or value_mask is not None):
            if query_mask is None:
                query_mask = tf_ones_like(inputs[..., 0], dtype=tf_bool)
            if value_mask is None:
                value_mask = tf_ones(
                    (batch_size, self.memory_size), dtype=tf_bool
                )
            qm = tf_expand_dims(tf_cast(query_mask, tf_bool), -1)  # (B,T,1)
            vm = tf_expand_dims(tf_cast(value_mask, tf_bool), 1)   # (B,1,M)
            attention_mask = tf_logical_and(qm, vm)                # (B,T,M)

        mem_att = self.attention(
            query=inputs,
            key=mem,
            value=mem,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            training=training,
        )
        return mem_att + inputs

    def get_config(self):
        cfg = super().get_config().copy()
        cfg.update({'units': self.units,
                    'memory_size': self.memory_size})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable(
    'fusionlab.nn.components', 
    name="HierarchicalAttention_"
)
class HierarchicalAttention_(Layer, NNLearner):
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
    'fusionlab.nn.components', name="HierarchicalAttention"
)
class HierarchicalAttention(Layer, NNLearner):
    r"""Short/long-term MHA with optional masks."""

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, units: int, num_heads: int):
        super().__init__()
        self.units = units
        self.short_term_dense = Dense(units)
        self.long_term_dense = Dense(units)
        self.short_term_attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=units
        )
        self.long_term_attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=units
        )

    @tf_autograph.experimental.do_not_convert
    def call(
        self,
        inputs,
        training: bool = False,
        *,
        short_mask: Optional[Tensor] = None,
        long_mask: Optional[Tensor] = None,
        use_causal_mask: bool = False,
        **kwargs,
    ):
        # inputs: [short_term, long_term]
        short_term, long_term = inputs

        s = self.short_term_dense(short_term)
        l = self.long_term_dense(long_term)

        # Build masks to (B, T, T) if provided as (B,T)
        def _expand_mask(m):
            if m is None:
                return None
            m = tf_cast(m, tf_bool)
            qm = tf_expand_dims(m, 1)  # (B,1,T)
            vm = tf_expand_dims(m, 2)  # (B,T,1)
            return tf_logical_and(vm, qm)  # (B,T,T)

        s_mask = _expand_mask(short_mask)
        l_mask = _expand_mask(long_mask)

        s_att = self.short_term_attention(
            query=s,
            key=s,
            value=s,
            attention_mask=s_mask,
            use_causal_mask=use_causal_mask,
            training=training,
        )
        l_att = self.long_term_attention(
            query=l,
            key=l,
            value=l,
            attention_mask=l_mask,
            use_causal_mask=use_causal_mask,
            training=training,
        )
        return s_att + l_att

    def get_config(self):
        cfg = super().get_config().copy()
        cfg.update({'units': self.units,
                    'short_term_dense': self.short_term_dense.get_config(),
                    'long_term_dense': self.long_term_dense.get_config()})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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
    
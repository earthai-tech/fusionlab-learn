# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Gating / normalization building blocks:
"""

from __future__ import annotations

from ._config import (    
    _logger, 
    DEP_MSG,
    KERAS_BACKEND, 
                  
    activations,
    Layer, Dense, Dropout, 
    BatchNormalization, 
    LayerNormalization,
    Softmax, 

    tf_shape, 
    tf_expand_dims, tf_tile,
    tf_add, 
    tf_multiply,
    tf_stack, tf_reduce_sum,
    tf_equal, tf_cond,
    tf_TensorShape,  
    register_keras_serializable,
    tf_autograph, 
    tf_concat, 
)

from ..api.property import NNLearner
from ..core.handlers import param_deprecated_message
from ..compat.sklearn import validate_params, Interval, StrOptions
from ..utils.deps_utils import ensure_pkg

from .misc import Activation

# In case you need warnings
import warnings
from numbers import Real, Integral
from typing import Optional

__all__ = [
    "GatedResidualNetwork",
    "VariableSelectionNetwork",
    "LearnedNormalization",
    "StaticEnrichmentLayer",
]

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

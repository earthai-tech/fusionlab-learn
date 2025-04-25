# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Neural network components for anomaly detection in time series.
"""
from numbers import Real, Integral
from typing import Optional, Union, List
import numpy as np 
import warnings 

from ..api.property import NNLearner 
from ..compat.sklearn import validate_params, Interval, StrOptions
from ..core.checks import is_iterable 
from ..utils.deps_utils import ensure_pkg
from ..utils.validator import validate_positive_integer

from . import KERAS_DEPS, KERAS_BACKEND, dependency_message

if KERAS_BACKEND:
    Layer = KERAS_DEPS.Layer
    LSTM = KERAS_DEPS.LSTM
    Dense = KERAS_DEPS.Dense
    Dropout = KERAS_DEPS.Dropout
    RepeatVector = KERAS_DEPS.RepeatVector
    TimeDistributed = KERAS_DEPS.TimeDistributed
    Dense = KERAS_DEPS.Dense
    Dropout = KERAS_DEPS.Dropout
    Bidirectional = KERAS_DEPS.Bidirectional
    LayerNormalization = KERAS_DEPS.LayerNormalization
    BatchNormalization = KERAS_DEPS.BatchNormalization 
    regularizers = KERAS_DEPS.regularizers
    
    tf_reduce_mean = KERAS_DEPS.reduce_mean
    tf_square = KERAS_DEPS.square
    tf_subtract = KERAS_DEPS.subtract
    tf_concat = KERAS_DEPS.concat 
    tf_reduce_max = KERAS_DEPS.reduce_max 
    tf_abs = KERAS_DEPS.abs 
    tf_expand_dims=KERAS_DEPS.expand_dims 
    tf_shape = KERAS_DEPS.shape 
    tf_minimum =KERAS_DEPS.minimum
    register_keras_serializable = KERAS_DEPS.register_keras_serializable
    Tensor = KERAS_DEPS.Tensor 


DEP_MSG = dependency_message('nn.anomaly_detection')

__all__ = [
    "LSTMAutoencoderAnomaly","SequenceAnomalyScoreLayer", 
    "PredictionErrorAnomalyScore", 
]

@register_keras_serializable(
    'fusionlab.nn.anomaly_detection', name='LSTMAutoencoderAnomaly'
)
class LSTMAutoencoderAnomaly(Layer, NNLearner): 
    """LSTM Autoencoder for time series reconstruction-based anomaly
       detection."""

    @validate_params({
        "latent_dim": [Interval(Integral, 1, None, closed="left")],
        "lstm_units": [Interval(Integral, 1, None, closed="left")],
        "num_encoder_layers": [Interval(Integral, 1, None, closed="left")],
        "num_decoder_layers": [Interval(Integral, 1, None, closed="left")],
        "n_features": [Interval(Integral, 1, None, closed="left"), None],
        "n_repeats": [Interval(Integral, 1, None, closed="left"), None], # Added
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "recurrent_dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "use_bidirectional_encoder": [bool],
        "use_bottleneck_dense": [bool],
    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        latent_dim: int,
        lstm_units: int,
        n_features: Optional[int] = None,
        n_repeats: Optional[int] = None, 
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        activation: str = 'tanh',
        intermediate_activation: str = 'relu',
        dropout_rate: float = 0.0,
        recurrent_dropout_rate: float = 0.0,
        use_bidirectional_encoder: bool = False,
        use_bottleneck_dense: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Store all hyperparameters
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.n_features = n_features
        self.n_repeats = n_repeats 
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.activation = activation
        self.intermediate_activation = intermediate_activation
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.use_bidirectional_encoder = use_bidirectional_encoder
        self.use_bottleneck_dense = use_bottleneck_dense

        # --- Define Encoder Layers ---
        self.encoder_layers = []
        for i in range(self.num_encoder_layers):
            is_last_encoder = (i == self.num_encoder_layers - 1)
            layer_name = f"encoder_lstm_{i+1}"
            lstm_layer = LSTM(
                self.lstm_units, return_sequences=not is_last_encoder,
                return_state=is_last_encoder, dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout_rate, name=layer_name
            )
            if self.use_bidirectional_encoder:
                self.encoder_layers.append(
                    Bidirectional(lstm_layer, name=f"bi_{layer_name}")
                )
            else:
                self.encoder_layers.append(lstm_layer)

        # --- Optional Bottleneck Layer ---
        if self.use_bottleneck_dense:
            self.bottleneck_dense_h = Dense(
                self.latent_dim, activation=self.intermediate_activation,
                name="bottleneck_h"
            )
            self.bottleneck_dense_c = Dense(
                self.latent_dim, activation=self.intermediate_activation,
                name="bottleneck_c"
            )

        # --- Define Decoder LSTM Layers ---
        self.decoder_layers = []
        for i in range(self.num_decoder_layers):
            # ... (decoder LSTM creation logic as before) ...
             layer_name = f"decoder_lstm_{i+1}"
             self.decoder_layers.append(LSTM(
                 self.lstm_units, return_sequences=True,
                 dropout=self.dropout_rate,
                 recurrent_dropout=self.recurrent_dropout_rate,
                 name=layer_name
             ))

        # --- Conditionally Define Final Dense Layer ---
        self.decoder_dense = None 
        if self.n_features is not None:
            self.decoder_dense = TimeDistributed(
                Dense(self.n_features, activation=self.activation),
                name="decoder_dense"
            )
    
        # --- Initialize Repeater placeholder ---
        self.repeater = None 

    def build(self, input_shape):
        """Configure layers whose dimensions depend on input shape."""
        if len(input_shape) != 3:
            raise ValueError(
                "Input should be 3D (Batch, TimeSteps, Features)."
                f" Received shape: {input_shape}"
            )
        _batch_size, time_steps, features = input_shape

        # --- Determine number of repeats ---
        # Use specified n_repeats if provided, otherwise use input time_steps
        num_repeats = self.n_repeats if self.n_repeats is not None else time_steps
        # Setting number of repetitions for decoder input.

        # --- Create Repeater Vector ---
        if self.repeater is None:
            # Determine the dimension feeding into the repeater
            if self.use_bottleneck_dense:
                repeater_input_dim = self.latent_dim
            elif self.use_bidirectional_encoder:
                repeater_input_dim = self.lstm_units * 2
            else:
                repeater_input_dim = self.lstm_units

            self.repeater = RepeatVector(
                num_repeats, # Use determined number of repeats
                # Provide input_shape hint for clarity (optional but good)
                input_shape=(_batch_size, repeater_input_dim),
                name="repeater"
            )
            # Repeater created/configured in build.

        # --- Create or Verify Final Dense Layer ---
        if self.decoder_dense is None:
            # If n_features was NOT provided at init, create layer now
            self.decoder_dense = TimeDistributed(
                Dense(features, activation=self.activation),
                # Hint input shape based on decoder LSTM output
                input_shape=(_batch_size, num_repeats, self.lstm_units),
                name="decoder_dense"
            )
            # decoder_dense created in build.
        elif self.n_features != features:
            # If layer exists (n_features was given), verify match
            raise ValueError(
                f"Input feature dimension ({features}) does not match "
                f"n_features ({self.n_features}) provided during "
                "initialization."
            )

        # Ensure super().build is called
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Forward pass: Encode -> [Bottleneck] -> Repeat -> Decode."""
        # --- Encoding ---
        
        encoded = inputs
        encoder_states = None
        for i, layer in enumerate(self.encoder_layers):
            
             is_last_encoder = (i == self.num_encoder_layers - 1)
             if is_last_encoder:
                 if self.use_bidirectional_encoder:
                     encoded, fh, fc, bh, bc = layer(encoded, training=training)
                     state_h = tf_concat([fh, bh], axis=-1)
                     state_c = tf_concat([fc, bc], axis=-1)
                 else:
                     encoded, state_h, state_c = layer(encoded, training=training)
                 encoder_states = [state_h, state_c]
             else:
                 encoded = layer(encoded, training=training)

        # --- Optional Bottleneck ---
        
        latent_vector = encoder_states[0]
        decoder_initial_state = encoder_states
        if self.use_bottleneck_dense:
            # ... (apply bottleneck_dense_h/c) ...
             bottleneck_h = self.bottleneck_dense_h(encoder_states[0])
             bottleneck_c = self.bottleneck_dense_c(encoder_states[1])
             latent_vector = bottleneck_h
             decoder_initial_state = [bottleneck_h, bottleneck_c]

        # --- Decoding ---
        # Repeater layer (created in build) must exist now
        if self.repeater is None:
             raise RuntimeError("Repeater layer was not built.")
             # Shape (B, num_repeats, latent_dim_eff)
        repeated_vector = self.repeater(latent_vector)

        decoded = repeated_vector
        initial_state_for_decoder = decoder_initial_state
        for i, layer in enumerate(self.decoder_layers):
             # Pass initial state only to the first decoder layer
            if i == 0:
                decoded = layer(
                    decoded, initial_state=initial_state_for_decoder,
                    training=training
                )
            else:
                decoded = layer(decoded, training=training)
        # Output shape: (B, num_repeats, lstm_units)

        # --- Final Reconstruction ---
        # Decoder dense layer (created in init or build) must exist
        if self.decoder_dense is None:
             raise RuntimeError("Decoder dense layer was not built.")

        reconstructions = self.decoder_dense(
            decoded, training=training
        ) # Output shape: (B, num_repeats, features)

        return reconstructions

    def compute_reconstruction_error(
        self,
        inputs: Union[np.ndarray, "Tensor"],
        reconstructions: Optional[Union[np.ndarray, "Tensor"]] = None
        ) -> "Tensor":
        """Computes Mean Squared Error per sample."""
        if reconstructions is None:
            reconstructions = self(inputs, training=False)

        # Ensure shapes match for error calculation, considering n_repeats
        # If n_repeats != time_steps, comparison needs adjustment
        input_time_steps = tf_shape(inputs)[1]
        recon_time_steps = tf_shape(reconstructions)[1]

        if input_time_steps != recon_time_steps:
            # If lengths differ, only compare the overlapping part or handle error
            # Here, we might compare only the first 'min(T, n_repeats)' steps
            min_steps = tf_minimum(input_time_steps, recon_time_steps)
            error = tf_subtract(inputs[:, :min_steps, :],
                                reconstructions[:, :min_steps, :])
            warnings.warn(f"Input time steps ({input_time_steps}) != "
                          f"reconstruction steps ({recon_time_steps}) due to "
                          f"n_repeats. Error calculated over first {min_steps} steps.")
        else:
            error = tf_subtract(inputs, reconstructions)

        squared_error = tf_square(error)
        # Average over time and feature dimensions (axis 1 and 2)
        mse_per_sample = tf_reduce_mean(squared_error, axis=[1, 2])
        return mse_per_sample

    def get_config(self):
        """Returns the layer configuration."""
        config = super().get_config()
        # Update with all parameters from __init__
        config.update({
            "latent_dim": self.latent_dim,
            "lstm_units": self.lstm_units,
            "n_features": self.n_features,
            "n_repeats": self.n_repeats, # Add new parameter
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "activation": self.activation,
            "intermediate_activation": self.intermediate_activation,
            "dropout_rate": self.dropout_rate,
            "recurrent_dropout_rate": self.recurrent_dropout_rate,
            "use_bidirectional_encoder": self.use_bidirectional_encoder,
            "use_bottleneck_dense": self.use_bottleneck_dense,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates layer from its config."""
        return cls(**config)

LSTMAutoencoderAnomaly.__doc__+=r"""\
This layer implements a configurable LSTM autoencoder architecture.
It encodes an input sequence into a lower-dimensional latent
representation and then decodes this representation back into a
sequence, attempting to reconstruct the original input. Training
typically involves minimizing the reconstruction error on normal data.

The core idea is that anomalous sequences, deviating from patterns
learned on normal data, will result in higher reconstruction errors,
which can serve as anomaly scores. This layer offers flexibility
in the number of encoder/decoder layers, bidirectionality,
bottleneck configuration, output feature dimension specification,
and the length of the reconstructed sequence.

Parameters
----------
latent_dim : int
    Dimensionality of the latent space (bottleneck). This controls
    the degree of information compression. If `use_bottleneck_dense`
    is True, this defines the output size of the bottleneck Dense
    layer applied to the final encoder hidden state. If False, this
    parameter might not be directly used (effective latent dim
    depends on `lstm_units` and `use_bidirectional_encoder`).
lstm_units : int
    Number of hidden units in each LSTM layer for both the encoder
    and decoder. Determines the capacity of the LSTMs.
n_features : int, optional, default=None
    Allows pre-specifying the number of output features (last
    dimension) for the reconstructed sequence.
    * If an integer is provided, the final `TimeDistributed(Dense)`
      layer is created during initialization with this many units.
      An error will be raised during the `build` step if the actual
      input feature dimension doesn't match this value.
    * If ``None`` (default), the number of output features is
      inferred from the input data's feature dimension during the
      `build` step.
n_repeats : int, optional, default=None
    Specifies a fixed number of time steps for the output sequence
    generated by the decoder.
    * If an integer is provided, the latent vector from the encoder
      is repeated `n_repeats` times before being fed into the
      decoder LSTM stack. The output reconstruction will have this
      many time steps, regardless of the input sequence length.
    * If ``None`` (default), the latent vector is repeated a number
      of times equal to the number of time steps in the *input*
      sequence, aiming to reconstruct the input fully.
num_encoder_layers : int, default=1
    Number of LSTM layers stacked in the encoder. Must be >= 1.
num_decoder_layers : int, default=1
    Number of LSTM layers stacked in the decoder. Must be >= 1.
activation : str, default='tanh'
    Activation function applied to the final TimeDistributed Dense
    output layer of the decoder, reconstructing the features.
    Examples: 'tanh', 'sigmoid', 'linear'. Choose based on the
    expected range or normalization of the input data.
intermediate_activation : str, default='relu'
    Activation function used in the optional bottleneck Dense
    layers (if `use_bottleneck_dense=True`).
dropout_rate : float, default=0.0
    Dropout rate applied to the non-recurrent connections (inputs
    and outputs) of the LSTM layers. Value between 0 and 1.
recurrent_dropout_rate : float, default=0.0
    Dropout rate applied to the recurrent connections within the
    LSTM layers. Value between 0 and 1. Note: Using recurrent
    dropout may require disabling GPU acceleration (CuDNN) for LSTMs.
use_bidirectional_encoder : bool, default=False
    If True, wraps the encoder LSTM layers with a Bidirectional
    wrapper, processing the input sequence in both forward and
    backward directions. The final hidden states are typically
    concatenated.
use_bottleneck_dense : bool, default=False
    If True, adds Dense layers after the final encoder LSTM layer
    to explicitly project the final hidden state (`state_h`) and
    cell state (`state_c`) to the specified `latent_dim`. If False,
    the final encoder states are used directly.
**kwargs
    Additional keyword arguments passed to the parent Keras `Layer`.

Notes
-----
This layer expects input data with the shape
`(Batch, TimeSteps, Features)`. The output shape will be
`(Batch, OutputTimeSteps, OutputFeatures)`, where `OutputTimeSteps`
is determined by `n_repeats` (or input `TimeSteps` if `n_repeats` is
None) and `OutputFeatures` is determined by `n_features` (or input
`Features` if `n_features` is None).

**Use Case and Importance**

This component is primarily used for *unsupervised* anomaly
detection in sequential data. By training the autoencoder primarily
on normal data, it learns the underlying patterns and structure
inherent in that normal behavior. When presented with new data,
sequences conforming to these learned patterns will be reconstructed
accurately (low error), while sequences containing anomalies or novel
patterns will result in poor reconstructions (high error). This
reconstruction error serves as a valuable, data-driven anomaly score,
particularly useful when labeled anomaly data is scarce or unavailable.
The added flexibility via `n_features` and `n_repeats` allows for
potential sequence-to-sequence tasks beyond pure reconstruction or
handling cases where output dimensions differ from input.

**Mathematical Formulation**

The enhanced LSTM autoencoder involves:

1.  **Encoder:** A stack of `num_encoder_layers` LSTMs
    (optionally bidirectional) processes the input sequence
    :math:`\mathbf{X} \in \mathbb{R}^{T \times F}`. The final layer
    outputs the last hidden state :math:`h_T` and cell state :math:`c_T`.

    .. math::
       [h_T, c_T] = \text{Encoder}_{LSTM\_Stack}(\mathbf{X})

2.  **Bottleneck (Optional):** If `use_bottleneck_dense=True`,
    the final states are projected to `latent_dim`:
    :math:`h'_T = \text{Dense}_{h}(h_T)`,
    :math:`c'_T = \text{Dense}_{c}(c_T)`. The latent vector used
    for decoding is :math:`\mathbf{z} = h'_T`. The decoder initial
    state is :math:`[h'_T, c'_T]`. If False, :math:`\mathbf{z} = h_T`
    and the initial state is :math:`[h_T, c_T]`.

3.  **Decoder Input Repetition:** The latent vector :math:`\mathbf{z}`
    is repeated $T'$ times using :class:`~tf.keras.layers.RepeatVector`,
    where $T' = \text{n\_repeats}$ if specified, otherwise $T' = T$
    (input time steps).

    .. math::
       \mathbf{Z}_{repeated} = \text{Repeat}(\mathbf{z})\\
           \in \mathbb{R}^{T' \times \text{dim}(\mathbf{z})}

4.  **Decoder:** A stack of `num_decoder_layers` LSTMs processes
    :math:`\mathbf{Z}_{repeated}`, initialized with the final
    (potentially bottlenecked) state from the encoder.

    .. math::
       \mathbf{H}_{dec} = \text{Decoder}_{LSTM\_Stack}\\
           (\mathbf{Z}_{repeated}, \text{initial_state}) \in\\
               \mathbb{R}^{T' \times \text{lstm\_units}}

5.  **Reconstruction:** A :class:`~tf.keras.layers.TimeDistributed`
    Dense layer maps the decoder's output sequence :math:`\mathbf{H}_{dec}`
    to the target feature dimension $F'$ (where $F' = \text{n\_features}$
    if specified, otherwise $F'=F$).

    .. math::
       \mathbf{\hat{X}} = \text{TimeDistributed}(\text{Dense}(\mathbf{H}_{dec}))\\
           \in \mathbb{R}^{T' \times F'}

The anomaly score is typically the reconstruction error, e.g.,
:math:`Error = ||\mathbf{X}_{[:T'',:F'']} - \mathbf{\hat{X}}_{[:T'',:F']}||^2`,
where comparison might be limited to overlapping dimensions if $T' \neq T$
or $F' \neq F$. The `compute_reconstruction_error` method handles
comparison over potentially differing time steps.

Methods
-------
call(inputs, training=False)
    Performs the forward pass (encoding and decoding). Output shape
    depends on `n_repeats` and `n_features`.
compute_reconstruction_error(inputs, reconstructions=None)
    Calculates the mean squared error per sample, potentially only
    over overlapping time steps if input/output lengths differ due
    to `n_repeats`.

Examples
--------
>>> from fusionlab.nn.anomaly_detection import LSTMAutoencoderAnomaly
>>> import tensorflow as tf
>>> B, T, F = 32, 20, 5 # Batch, TimeSteps, Features
>>> inputs = tf.random.normal((B, T, F))
>>> # Instantiate with specific output features and repeats
>>> lstm_ae = LSTMAutoencoderAnomaly(
...     latent_dim=8,
...     lstm_units=16,
...     n_features=F,  # Explicitly state output features
...     n_repeats=T,   # Explicitly state output time steps
...     num_encoder_layers=2,
...     num_decoder_layers=2,
... )
>>> # Get reconstructions
>>> reconstructions = lstm_ae(inputs)
>>> print(f"Reconstruction shape: {reconstructions.shape}") # Should be (32, 20, 5)
TensorShape([32, 20, 5])
>>> # Compute error
>>> error = lstm_ae.compute_reconstruction_error(inputs)
>>> print(f"Error shape: {error.shape}") # Should be (32,)
TensorShape([32])

See Also
--------
tensorflow.keras.layers.Layer : Base class for Keras layers.
tensorflow.keras.layers.LSTM : The recurrent layer used internally.
tensorflow.keras.layers.RepeatVector : Used to feed decoder.
tensorflow.keras.layers.TimeDistributed : Wraps the final Dense layer.
tensorflow.keras.layers.Bidirectional : Wrapper for bidirectional RNNs.
fusionlab.nn.transformers.XTFT : Can potentially incorporate anomaly
    scores derived from reconstruction errors.
fusionlab.nn.losses.anomaly_loss : Can be used with anomaly scores
    derived from this layer's error.
SequenceAnomalyScoreLayer : Alternative anomaly detection component.

References
----------
.. [1] Malhotra, P., Vig, L., Shroff, G., & Agarwal, P. (2015).
       Long Short Term Memory Networks for Anomaly Detection in
       Time Series. *Proc. European Symposium on Artificial Neural
       Networks (ESANN)*, 480-485.
"""

@register_keras_serializable(
    'fusionlab.nn.anomaly_detection', name='SequenceAnomalyScoreLayer'
)
class SequenceAnomalyScoreLayer(Layer, NNLearner):
    """Computes an anomaly score from input features using a Multi-Layer
    Perceptron (MLP)."""
    _COMMON_ACTIVATIONS = {
        "relu", "tanh", "sigmoid", "elu", "selu", "gelu", "linear", 
    }

    @validate_params({
        "hidden_units": [
            Interval(Integral, 1, None, closed="left"),
            'array-like', 
        ],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "use_norm": [bool, StrOptions({'layer', 'batch'})],
        "activation": [StrOptions(_COMMON_ACTIVATIONS), None],
        "final_activation": [StrOptions(_COMMON_ACTIVATIONS), None],

    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        hidden_units: Union[int, List[int]],
        activation: str = 'relu',
        dropout_rate: float = 0.1,
        use_norm: Union[bool, str] = False, 
        final_activation: str = 'linear',
        kernel_regularizer=None, 
        bias_regularizer=None,   
        **kwargs
    ):
        super().__init__(**kwargs)

        # Store parameters
        # Ensure hidden_units is a list for iteration
        hidden_units = is_iterable(
            hidden_units, transform=True, exclude_string=True 
            )
        self.hidden_units = [
            validate_positive_integer(v, f"hidden_unit {v}")
            for v in hidden_units ]
        
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.final_activation = final_activation
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # --- Define Internal Layers ---
        self.hidden_layers = []
        self.norm_layers = []
        self.dropout_layers = [] # Use separate dropout instances

        # Create hidden layers based on hidden_units list
        for i, units in enumerate(self.hidden_units):
            # Add Dense layer
            self.hidden_layers.append(Dense(
                units,
                activation=self.activation,
                kernel_regularizer=self.kernel_regularizer, 
                bias_regularizer=self.bias_regularizer,  
                name=f"hidden_dense_{i+1}"
            ))
            # Add Normalization layer if specified
            if self.use_norm:
                norm_layer = None
                norm_type = 'layer' if isinstance(self.use_norm, bool) else self.use_norm
                if norm_type == 'layer':
                    norm_layer = LayerNormalization(name=f"layer_norm_{i+1}")
                elif norm_type == 'batch':
                    norm_layer = BatchNormalization(name=f"batch_norm_{i+1}")
                # Store None if use_norm is False, to keep lists aligned
                self.norm_layers.append(norm_layer)
            else:
                 self.norm_layers.append(None) # Keep list aligned

            # Add Dropout layer (applied after norm if used)
            self.dropout_layers.append(Dropout(
                self.dropout_rate,
                name=f"score_dropout_{i+1}"
            ))

        # Final output layer (1 unit for the score)
        self.score_dense = Dense(
            1,
            activation=self.final_activation,
            name="score_output"
        )

    def call(self, inputs, training=False):
        """
        Forward pass: (Dense -> [Norm] -> Dropout) * N -> Dense Output.
        Expects inputs of shape (Batch, Features).
        """
        x = inputs

        # Process through hidden layers
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            if self.use_norm and self.norm_layers[i] is not None:
                # Pass training flag to BatchNormalization
                if isinstance(self.norm_layers[i], BatchNormalization):
                     x = self.norm_layers[i](x, training=training)
                else: # LayerNormalization doesn't always use training flag
                     x = self.norm_layers[i](x)
            x = self.dropout_layers[i](x, training=training)

        # Final score prediction
        scores = self.score_dense(x)
        return scores

    def get_config(self):
        """Returns the layer configuration."""
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "use_norm": self.use_norm,
            "final_activation": self.final_activation,
            # Serialize regularizers if added
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates layer from its config."""
        # Deserialize regularizers if added
        config['kernel_regularizer'] = regularizers.deserialize(
            config.get('kernel_regularizer'))
        config['bias_regularizer'] = regularizers.deserialize(
            config.get('bias_regularizer'))
        return cls(**config)

SequenceAnomalyScoreLayer.__doc__+=r"""\
This layer processes input features, typically representing learned
embeddings or aggregated sequence information from upstream layers,
through a configurable MLP to produce a scalar anomaly score for
each input sample.

It provides flexibility in defining the depth and width of the MLP,
activation functions, normalization, and dropout for regularization.
The output score reflects the model's learned assessment of how
anomalous the input features are.

Parameters
----------
hidden_units : int or list of int
    Specifies the structure of the hidden layers in the MLP.
    * If `int`: A single hidden layer with that many units is used.
    * If `list[int]`: Creates multiple hidden layers, where each
      integer in the list defines the number of units for the
      corresponding layer.
activation : str, default='relu'
    Activation function applied after each hidden dense layer (but
    before normalization or dropout). Common choices include 'relu',
    'elu', 'gelu', 'tanh'.
dropout_rate : float, default=0.1
    Dropout rate applied after activation (and normalization, if used)
    in each hidden layer. Value between 0 and 1.
use_norm : bool or str, default=False
    Specifies whether to apply normalization after the activation
    in hidden layers.
    * `False`: No normalization.
    * `True` or `'layer'`: Use Layer Normalization.
    * `'batch'`: Use Batch Normalization. Note that Batch Normalization
      behaves differently during training and inference.
final_activation : str, default='linear'
    Activation function applied to the final output neuron that
    produces the scalar anomaly score.
    * 'linear': Produces an unbounded score.
    * 'sigmoid': Produces a score between 0 and 1, interpretable
      as a probability or normalized score.
    * Other activations like 'softplus' can also be used to ensure
      non-negative scores.
**kwargs
    Additional keyword arguments passed to the parent Keras `Layer`.

Notes
-----
This layer typically expects input features with shape
`(Batch, Features)`. If your input is sequential
`(Batch, TimeSteps, Features)`, you might need to flatten or pool it
before feeding it to this layer.

**Use Case and Importance**

This layer is designed to be a *part* of a larger model, acting as a
dedicated "scoring head" that learns to map complex internal features
to an anomaly score. It's useful when you want the model to learn
what constitutes an anomaly based on learned representations, rather
than relying solely on reconstruction error or predefined rules. This
approach aligns well with the concept of feature-based anomaly detection
within models like XTFT. Training this layer effectively requires
integrating it into a larger network and defining a suitable loss
function that utilizes its output score, potentially combining it with
the primary task's loss (e.g., forecasting loss) or using anomaly labels
if available (supervised training).

**Mathematical Formulation**

The layer implements a standard Multi-Layer Perceptron (MLP). For an
input feature vector :math:`\mathbf{h}` and $L$ hidden layers:

Let :math:`\mathbf{h}^{(0)} = \mathbf{h}`.
For each hidden layer $i = 1 \dots L$:

.. math::
   \mathbf{a}^{(i)} = \text{Dense}_i(\mathbf{h}^{(i-1)}) \\
   \mathbf{n}^{(i)} = \text{Activation}(\mathbf{a}^{(i)}) \\
   \mathbf{o}^{(i)} = \text{Normalization}(\mathbf{n}^{(i)}) \quad (\text{if use_norm=True}) \\
   \mathbf{h}^{(i)} = \text{Dropout}(\mathbf{o}^{(i)} \text{ or } \mathbf{n}^{(i)})

The final score is computed from the last hidden layer's output
:math:`\mathbf{h}^{(L)}`:

.. math::
   \text{Score} = \text{FinalActivation}(\text{Dense}_{out}(\mathbf{h}^{(L)}))

where `Dense` includes weights, biases, and the specified activation
or normalization steps.

Methods
-------
call(inputs, training=False)
    Performs the forward pass to compute anomaly scores.

Examples
--------
>>> from fusionlab.nn.anomaly_detection import SequenceAnomalyScoreLayer
>>> import tensorflow as tf
>>> B, F = 32, 64 # Batch, Features
>>> # Assume 'features' are output from another layer
>>> features = tf.random.normal((B, F))
>>> # Instantiate with multiple hidden layers and LayerNorm
>>> anomaly_scorer = SequenceAnomalyScoreLayer(
...     hidden_units=[64, 32], # Two hidden layers
...     activation='relu',
...     dropout_rate=0.2,
...     use_norm='layer', # Use Layer Normalization
...     final_activation='sigmoid' # Output score between 0 and 1
... )
>>> # Get anomaly scores
>>> scores = anomaly_scorer(features, training=True) # Pass training flag
>>> scores.shape
TensorShape([32, 1])

See Also
--------
tensorflow.keras.layers.Layer : Base class for Keras layers.
tensorflow.keras.layers.Dense : Fully-connected layer used internally.
tensorflow.keras.layers.Dropout : Dropout regularization layer.
tensorflow.keras.layers.LayerNormalization : Normalization layer option.
tensorflow.keras.layers.BatchNormalization : Normalization layer option.
fusionlab.nn.transformers.XTFT : Can incorporate feature-based anomaly
    detection potentially using layers like this.
fusionlab.nn.losses.AnomalyLoss : Loss component for anomaly scores.
LSTMAutoencoderAnomaly : Alternative reconstruction-based component.

References
----------
.. [1] Chalapathy, R., & Chawla, S. (2019). Deep learning for
       anomaly detection: A survey. *arXiv preprint arXiv:1901.03407*.
"""

@register_keras_serializable(
    'fusionlab.nn.anomaly_detection', name='PredictionErrorAnomalyScore'
)
class PredictionErrorAnomalyScore(Layer, NNLearner):
    """ Calculates an anomaly score based on prediction error between
    true and predicted sequences."""
    @validate_params({
        "error_metric": [StrOptions({"mae", "mse"})],
        "aggregation": [StrOptions({"mean", "max"})],
    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        error_metric: str = 'mae',
        aggregation: str = 'mean',
        **kwargs
    ):
        """
        Initialize layer.

        Args:
            error_metric (str): Metric for step-wise error
                ('mae' or 'mse'). Default is 'mae'.
            aggregation (str): How to aggregate step-wise errors
                ('mean' or 'max'). Default is 'mean'.
        """
        super().__init__(**kwargs)
        self.error_metric = error_metric.lower()
        self.aggregation = aggregation.lower()

    def call(self, inputs, training=False):
        """
        Calculate anomaly score from prediction error.

        Args:
            inputs (list[Tensor]): List containing [y_true, y_pred].
                Both tensors should have shape (Batch, TimeSteps, Features).
            training (bool): Ignored.

        Returns:
            Tensor: Anomaly scores, shape (Batch, 1).
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                "Input must be a list or tuple: [y_true, y_pred]."
            )
        y_true, y_pred = inputs

        # Ensure shapes match (basic check)
        if y_true.shape != y_pred.shape:
             warnings.warn(
                 f"Shapes of y_true {y_true.shape} and y_pred"
                 f" {y_pred.shape} do not match. Ensure they are"
                 " compatible for element-wise operations."
             )
             # Attempt to proceed if broadcasting might work,
             # otherwise TF will raise an error later.

        # Calculate element-wise error
        error = tf_subtract(y_true, y_pred)

        # Calculate step-wise error score based on metric
        if self.error_metric == 'mae':
            step_error = tf_abs(error)
        elif self.error_metric == 'mse':
            step_error = tf_square(error)
        else:
            # Should not happen due to validation, but belt-and-suspenders
            raise ValueError("Invalid error_metric specified.")

        # Average error across features dimension first
        # Shape becomes (Batch, TimeSteps)
        error_per_step = tf_reduce_mean(step_error, axis=-1)

        # Aggregate errors across time dimension
        if self.aggregation == 'mean':
            # Shape becomes (Batch,)
            score = tf_reduce_mean(error_per_step, axis=-1)
        elif self.aggregation == 'max':
            # Shape becomes (Batch,)
            score = tf_reduce_max(error_per_step, axis=-1)
        else:
            # Should not happen due to validation
            raise ValueError("Invalid aggregation specified.")

        # Reshape score to (Batch, 1) for consistency
        score = tf_expand_dims(score, axis=-1)
        return score

    def get_config(self):
        """Returns the layer configuration."""
        config = super().get_config()
        config.update({
            "error_metric": self.error_metric,
            "aggregation": self.aggregation,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates layer from its config."""
        return cls(**config)

PredictionErrorAnomalyScore.__doc__+=r"""\
This layer quantifies the discrepancy between ground truth (`y_true`)
and model predictions (`y_pred`) for time series, aggregating the
error across time and features to produce a single anomaly score per
sequence.

It provides a direct way to measure how well a model's predictions
match the actual outcomes, with larger errors typically indicating
more anomalous or unexpected behavior.

Parameters
----------
error_metric : {'mae', 'mse'}, default='mae'
    The metric used to calculate the element-wise error between
    `y_true` and `y_pred` at each time step and feature.
    * ``'mae'``: Mean Absolute Error, $|y_{true} - y_{pred}|$. Less
      sensitive to large outliers.
    * ``'mse'``: Mean Squared Error, $(y_{true} - y_{pred})^2$.
      Penalizes larger errors more heavily.
aggregation : {'mean', 'max'}, default='mean'
    The method used to aggregate the per-step errors (which are
    already averaged across features) into a single score for the
    entire sequence.
    * ``'mean'``: Computes the average error across all time steps.
    * ``'max'``: Takes the maximum error encountered across all time
      steps. More sensitive to single large deviations.
**kwargs
    Additional keyword arguments passed to the parent Keras `Layer`.

Notes
-----
This layer expects input as a list or tuple containing two tensors:
`[y_true, y_pred]`, both with the shape
`(Batch, TimeSteps, Features)`.

**Use Case and Importance**

This component directly implements the core logic behind
*prediction-based* anomaly detection. It assumes that anomalies
manifest as poor predictions by a model trained on normal patterns.
It's particularly useful when integrated into a multi-task learning
setup where a forecasting model generates `y_pred`. The output score
from this layer can then be fed into a loss function (like
:class:`~fusionlab.nn.components.AnomalyLoss` or used within
:func:`~fusionlab.nn.losses.prediction_based_loss`) to penalize the
model for large prediction errors, implicitly guiding it to recognize
or adapt to anomalous points. This approach links anomaly detection
directly to the model's predictive performance.

**Mathematical Formulation**

1.  **Element-wise Error:** Calculate the error term :math:`e_{t,f}`
    at each time step :math:`t` and feature :math:`f`.

    .. math::
       e_{t,f} = y_{true; t,f} - y_{pred; t,f}

2.  **Step Error Score:** Apply the chosen metric (`mae` or `mse`)
    and average across features ($F$) to get a score for each time
    step :math:`t`.

    .. math::
       \text{Error}_t = \frac{1}{F} \sum_{f=1}^F \text{metric}(e_{t,f})

    where :math:`\text{metric}(e) = |e|` for MAE, and
    :math:`\text{metric}(e) = e^2` for MSE.

3.  **Sequence Aggregation:** Aggregate the step errors
    :math:`\{\text{Error}_t\}_{t=1}^T` across time ($T$) using the
    chosen aggregation method (`mean` or `max`).

    .. math::
       \text{Score}_{seq} = \text{Aggregation}_{t=1}^T (\text{Error}_t)

Methods
-------
call(inputs, training=False)
    Calculates the anomaly score based on input `[y_true, y_pred]`.

Examples
--------
>>> from fusionlab.nn.anomaly_detection import PredictionErrorAnomalyScore
>>> import tensorflow as tf
>>> B, T, F = 32, 20, 3 # Batch, TimeSteps, Features
>>> # Assume y_true and y_pred come from your model/data
>>> y_true = tf.random.normal((B, T, F))
>>> y_pred = y_true + tf.random.normal((B, T, F), stddev=0.5) # Add noise
>>> # Instantiate the layer using Mean Absolute Error and Max aggregation
>>> error_scorer = PredictionErrorAnomalyScore(
...     error_metric='mae',
...     aggregation='max'
... )
>>> # Calculate scores
>>> anomaly_scores = error_scorer([y_true, y_pred])
>>> anomaly_scores.shape
TensorShape([32, 1])

See Also
--------
tensorflow.keras.layers.Layer : Base class for Keras layers.
fusionlab.nn.losses.prediction_based_loss : Loss function factory using
    a similar error-based anomaly concept.
fusionlab.nn.components.AnomalyLoss : Loss component that can take
    scores from this or other layers.
LSTMAutoencoderAnomaly : Reconstruction-based anomaly detection.
SequenceAnomalyScoreLayer : Feature-based anomaly scoring layer.

References
----------
.. [1] Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly
       detection: A survey. *ACM computing surveys (CSUR)*, 41(3),
       1-58. (General survey covering deviation-based methods).
"""
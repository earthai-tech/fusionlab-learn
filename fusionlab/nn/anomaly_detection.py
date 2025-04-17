# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
# File: fusionlab/nn/anomaly_detection.py

"""
Neural network components for anomaly detection in time series.
"""
from numbers import Real, Integral
from typing import Optional, Union
import numpy as np 

from ..api.property import NNLearner 
from ..compat.sklearn import validate_params, Interval
from ..utils.deps_utils import ensure_pkg

from . import KERAS_DEPS, KERAS_BACKEND, dependency_message

if KERAS_BACKEND:
    Layer = KERAS_DEPS.Layer
    LSTM = KERAS_DEPS.LSTM
    Dense = KERAS_DEPS.Dense
    Dropout = KERAS_DEPS.Dropout
    RepeatVector = KERAS_DEPS.RepeatVector
    TimeDistributed = KERAS_DEPS.TimeDistributed
    Dense = KERAS_DEPS.Dense

    tf_reduce_mean = KERAS_DEPS.reduce_mean
    tf_square = KERAS_DEPS.square
    tf_subtract = KERAS_DEPS.subtract
    register_keras_serializable = KERAS_DEPS.register_keras_serializable
    Tensor = KERAS_DEPS.Tensor 
else:
    # Define dummy classes or raise ImportError if TF/Keras is mandatory
    Layer = object
    Dense = object
    Dropout = object

DEP_MSG = dependency_message('nn.anomaly_detection')

__all__ = [
    "LSTMAutoencoderAnomaly",
]

@register_keras_serializable(
    'fusionlab.nn.anomaly_detection', name='LSTMAutoencoderAnomaly'
)
class LSTMAutoencoderAnomaly(Layer, NNLearner): 
    """LSTM Autoencoder for reconstruction-based anomaly detection."""

    @validate_params({
        "latent_dim": [Interval(Integral, 1, None, closed="left")],
        "lstm_units": [Interval(Integral, 1, None, closed="left")],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        latent_dim: int,
        lstm_units: int,
        activation: str = 'tanh',
        dropout_rate: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.activation = activation
        self.dropout_rate = dropout_rate

        # Define Encoder layer(s)
        # Using return_state=True to potentially use state later
        self.encoder_lstm = LSTM(
            self.lstm_units,
            return_sequences=False, # Get only the last output/state
            return_state=True,
            dropout=self.dropout_rate,
            name="encoder_lstm"
        )
        # Optional: Dense layer for bottleneck if latent_dim != lstm_units
        # self.bottleneck_dense = Dense(self.latent_dim, ...)

        # Define Decoder layer(s)
        self.repeater = RepeatVector(
            -1, # Time steps placeholder, set in build
            name="repeater"
        )
        self.decoder_lstm = LSTM(
            self.lstm_units,
            return_sequences=True, # Need output for each step
            dropout=self.dropout_rate,
            name="decoder_lstm"
        )
        # Output layer to reconstruct original features
        self.decoder_dense = TimeDistributed(
            Dense(
                -1, # Feature dim placeholder, set in build
                activation=self.activation
                ),
            name="decoder_dense"
        )

    def build(self, input_shape):
        """Configure layer dimensions based on input shape."""
        if len(input_shape) != 3:
            raise ValueError(
                "Input should be 3D (Batch, TimeSteps, Features)."
                f" Received shape: {input_shape}"
            )
        _batch_size, time_steps, features = input_shape

        self.repeater.n = time_steps
        self.decoder_dense.layer.units = features

        # Let Keras build internal layers implicitly on first call
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Forward pass: Encode -> Repeat -> Decode."""
        # Encode
        # Use final hidden state (state_h) as latent representation
        _encoder_outputs, state_h, _state_c = self.encoder_lstm(
            inputs, training=training
        )
        latent_vector = state_h # Shape: (Batch, lstm_units)
        # Add bottleneck dense layer here if self.latent_dim != self.lstm_units
        # latent_vector = self.bottleneck_dense(latent_vector)

        # Decode
        repeated_vector = self.repeater(latent_vector)
        decoder_output = self.decoder_lstm(
            repeated_vector, training=training
        )
        reconstructions = self.decoder_dense(
            decoder_output, training=training
        )
        return reconstructions

    def compute_reconstruction_error(
        self,
        inputs: Union[np.ndarray, "Tensor"],
        reconstructions: Optional[Union[np.ndarray, "Tensor"]] = None
        ) -> "Tensor":
        """Computes Mean Squared Error per sample."""
        if reconstructions is None:
            reconstructions = self(inputs, training=False)

        error = tf_subtract(inputs, reconstructions)
        squared_error = tf_square(error)
        mse_per_sample = tf_reduce_mean(squared_error, axis=[1, 2])
        return mse_per_sample

    def get_config(self):
        """Returns the layer configuration."""
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "lstm_units": self.lstm_units,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates layer from its config."""
        return cls(**config)


@register_keras_serializable(
    'fusionlab.nn.anomaly_detection', name='SequenceAnomalyScoreLayer'
)
class SequenceAnomalyScoreLayer(Layer, NNLearner):
    """Computes an anomaly score from input sequence features."""

    @validate_params({
        "output_units": [Interval(Integral, 1, None, closed="left")],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        # Add validation for activation strings if needed
    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        output_units: int,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        final_activation: str = 'linear',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_units = output_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.final_activation = final_activation

        # Define internal layers
        self.hidden_dense = Dense(
            self.output_units,
            activation=self.activation,
            name="hidden_dense"
        )
        self.dropout = Dropout(
            self.dropout_rate,
            name="score_dropout"
        )
        # Final layer outputs a single score per sample
        self.score_dense = Dense(
            1,
            activation=self.final_activation,
            name="score_output"
        )

    def call(self, inputs, training=False):
        """Forward pass: Dense -> Dropout -> Dense Output."""
        x = self.hidden_dense(inputs)
        x = self.dropout(x, training=training)
        scores = self.score_dense(x)
        return scores

    def get_config(self):
        """Returns the layer configuration."""
        config = super().get_config()
        config.update({
            "output_units": self.output_units,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "final_activation": self.final_activation,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates layer from its config."""
        return cls(**config)

# test_anomaly_detection.py

import pytest
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Union

# --- Attempt to import anomaly detection classes and dependencies ---
try:
    from fusionlab.nn.anomaly_detection import (
        LSTMAutoencoderAnomaly,
        SequenceAnomalyScoreLayer,
        PredictionErrorAnomalyScore
    )
    from fusionlab.nn import KERAS_BACKEND
    if KERAS_BACKEND:
        Tensor = tf.Tensor # Use tf.Tensor directly if tf is the backend
    else: # Fallback for type hinting if Keras not fully loaded
        class Tensor: pass
    FUSIONLAB_INSTALLED = True
except ImportError as e:
    print(f"Skipping anomaly_detection tests due to import error: {e}")
    FUSIONLAB_INSTALLED = False
    class Tensor: pass
    class LSTMAutoencoderAnomaly: pass
    class SequenceAnomalyScoreLayer: pass
    class PredictionErrorAnomalyScore: pass

pytestmark = pytest.mark.skipif(
    not FUSIONLAB_INSTALLED,
    reason="fusionlab.nn.anomaly_detection or dependencies not found"
)

# --- Constants for Test Data ---
BATCH_SIZE = 4
SEQUENCE_LENGTH = 20
N_FEATURES = 1 # Primarily for LSTMAutoencoder
LATENT_DIM = 8
LSTM_UNITS = 16
HIDDEN_UNITS_SCORE = 10 # For SequenceAnomalyScoreLayer

# --- Fixtures ---
@pytest.fixture
def dummy_sequence_data() -> tf.Tensor:
    """Generates dummy 3D sequence data (Batch, TimeSteps, Features)."""
    return tf.random.normal(
        (BATCH_SIZE, SEQUENCE_LENGTH, N_FEATURES), dtype=tf.float32
        )

@pytest.fixture
def dummy_feature_data() -> tf.Tensor:
    """Generates dummy 2D feature data (Batch, Features)."""
    return tf.random.normal(
        (BATCH_SIZE, LSTM_UNITS), dtype=tf.float32 # Example feature dim
        )

# === Tests for LSTMAutoencoderAnomaly ===

def test_lstm_autoencoder_instantiation():
    """Test LSTMAutoencoderAnomaly instantiation with minimal args."""
    try:
        model = LSTMAutoencoderAnomaly(
            latent_dim=LATENT_DIM,
            lstm_units=LSTM_UNITS,
            n_features=N_FEATURES, # Required if decoder_dense built in init
            n_repeats=SEQUENCE_LENGTH # Required for RepeatVector
        )
        assert isinstance(model, tf.keras.Model)
        assert model.latent_dim == LATENT_DIM
        assert model.n_repeats == SEQUENCE_LENGTH
    except Exception as e:
        pytest.fail(f"LSTMAutoencoderAnomaly instantiation failed: {e}")
    print("LSTMAutoencoderAnomaly instantiation: OK")

def test_lstm_autoencoder_call_and_output_shape(dummy_sequence_data):
    """Test call method and output shape of LSTMAutoencoderAnomaly."""
    model = LSTMAutoencoderAnomaly(
        latent_dim=LATENT_DIM, lstm_units=LSTM_UNITS,
        n_features=N_FEATURES, n_repeats=SEQUENCE_LENGTH
    )
    # Build the model by calling it, or Keras will do it in fit/predict
    # To ensure layers like RepeatVector are built correctly before predict
    # if build method relies on input_shape from call.
    try:
        # Explicit build if needed, or ensure first call builds it.
        model.build(input_shape=dummy_sequence_data.shape)
        reconstructions = model(dummy_sequence_data, training=False)
    except Exception as e:
        pytest.fail(f"LSTMAutoencoderAnomaly call failed: {e}")

    expected_shape = (BATCH_SIZE, SEQUENCE_LENGTH, N_FEATURES)
    assert reconstructions.shape == expected_shape, \
        (f"Output shape mismatch. Expected {expected_shape}, "
         f"got {reconstructions.shape}")
    print("LSTMAutoencoderAnomaly call and output shape: OK")

def test_lstm_autoencoder_reconstruction_error(dummy_sequence_data):
    """Test compute_reconstruction_error method."""
    model = LSTMAutoencoderAnomaly(
        latent_dim=LATENT_DIM, lstm_units=LSTM_UNITS,
        n_features=N_FEATURES, n_repeats=SEQUENCE_LENGTH
    )
    # Get reconstructions (model needs to be built)
    reconstructions = model(dummy_sequence_data, training=False)
    try:
        errors = model.compute_reconstruction_error(
            dummy_sequence_data, reconstructions
        )
    except Exception as e:
        pytest.fail(f"compute_reconstruction_error failed: {e}")

    assert errors.shape == (BATCH_SIZE,), \
        f"Reconstruction error shape mismatch. Expected ({BATCH_SIZE},), got {errors.shape}"
    assert errors.dtype == tf.float32
    print("LSTMAutoencoderAnomaly compute_reconstruction_error: OK")

def test_lstm_autoencoder_fit_predict(dummy_sequence_data):
    """Test a minimal fit and predict cycle."""
    model = LSTMAutoencoderAnomaly(
        latent_dim=LATENT_DIM, lstm_units=LSTM_UNITS,
        n_features=N_FEATURES, n_repeats=SEQUENCE_LENGTH
    )
    model.compile(optimizer='adam', loss='mse')
    try:
        model.fit(dummy_sequence_data, dummy_sequence_data, # X and y are same
                  epochs=1, batch_size=2, verbose=0)
        _ = model.predict(dummy_sequence_data, verbose=0)
    except Exception as e:
        pytest.fail(f"LSTMAutoencoderAnomaly fit/predict cycle failed: {e}")
    print("LSTMAutoencoderAnomaly fit/predict cycle: OK")

def test_lstm_autoencoder_serialization(dummy_sequence_data):
    """Test get_config and from_config for LSTMAutoencoderAnomaly."""
    model1 = LSTMAutoencoderAnomaly(
        latent_dim=LATENT_DIM, lstm_units=LSTM_UNITS,
        n_features=N_FEATURES, n_repeats=SEQUENCE_LENGTH,
        dropout_rate=0.1, use_bidirectional_encoder=True
    )
    # Call model to build it
    _ = model1(dummy_sequence_data)
    config = model1.get_config()

    try:
        model2 = LSTMAutoencoderAnomaly.from_config(config)
    except Exception as e:
        pytest.fail(f"LSTMAutoencoderAnomaly.from_config failed: {e}")

    assert model2.latent_dim == model1.latent_dim
    assert model2.use_bidirectional_encoder == model1.use_bidirectional_encoder
    # Check if a prediction runs
    try:
        _ = model2.predict(dummy_sequence_data, verbose=0)
    except Exception as e:
        pytest.fail(f"Prediction failed on model from_config: {e}")
    print("LSTMAutoencoderAnomaly serialization: OK")


# === Tests for SequenceAnomalyScoreLayer ===

def test_sequence_score_layer_instantiation():
    """Test SequenceAnomalyScoreLayer instantiation."""
    try:
        layer = SequenceAnomalyScoreLayer(hidden_units=HIDDEN_UNITS_SCORE)
        assert isinstance(layer, tf.keras.layers.Layer)
        assert layer.hidden_units == [HIDDEN_UNITS_SCORE] # is_iterable makes it a list
    except Exception as e:
        pytest.fail(f"SequenceAnomalyScoreLayer instantiation failed: {e}")

    # Test with list of hidden units
    try:
        layer_multi = SequenceAnomalyScoreLayer(hidden_units=[32, 16])
        assert layer_multi.hidden_units == [32, 16]
    except Exception as e:
        pytest.fail(f"SequenceAnomalyScoreLayer with list hidden_units failed: {e}")
    print("SequenceAnomalyScoreLayer instantiation: OK")

def test_sequence_score_layer_call_and_output_shape(dummy_feature_data):
    """Test call method and output shape of SequenceAnomalyScoreLayer."""
    layer = SequenceAnomalyScoreLayer(
        hidden_units=HIDDEN_UNITS_SCORE,
        activation='relu',
        dropout_rate=0.1
        )
    try:
        scores = layer(dummy_feature_data, training=False)
    except Exception as e:
        pytest.fail(f"SequenceAnomalyScoreLayer call failed: {e}")

    expected_shape = (BATCH_SIZE, 1) # Output is a single score per sample
    assert scores.shape == expected_shape, \
        (f"Output shape mismatch. Expected {expected_shape}, "
         f"got {scores.shape}")
    assert scores.dtype == tf.float32
    print("SequenceAnomalyScoreLayer call and output shape: OK")

def test_sequence_score_layer_serialization(dummy_feature_data):
    """Test get_config and from_config for SequenceAnomalyScoreLayer."""
    layer1 = SequenceAnomalyScoreLayer(
        hidden_units=[HIDDEN_UNITS_SCORE, HIDDEN_UNITS_SCORE // 2],
        activation='tanh', final_activation='sigmoid'
        )
    # Call to build
    _ = layer1(dummy_feature_data)
    config = layer1.get_config()

    try:
        layer2 = SequenceAnomalyScoreLayer.from_config(config)
    except Exception as e:
        pytest.fail(f"SequenceAnomalyScoreLayer.from_config failed: {e}")

    assert layer2.hidden_units == layer1.hidden_units
    assert layer2.final_activation == layer1.final_activation
    try:
        _ = layer2.predict(dummy_feature_data, verbose=0) # Use predict if it's a Model
    except AttributeError: # If it's just a Layer, call it directly
        _ = layer2(dummy_feature_data, training=False)
    except Exception as e:
        pytest.fail(f"Prediction/call failed on layer from_config: {e}")
    print("SequenceAnomalyScoreLayer serialization: OK")

# === Tests for PredictionErrorAnomalyScore ===

@pytest.mark.parametrize("metric", ['mae', 'mse'])
@pytest.mark.parametrize("agg", ['mean', 'max'])
def test_prediction_error_score_instantiation(metric, agg):
    """Test PredictionErrorAnomalyScore instantiation."""
    try:
        layer = PredictionErrorAnomalyScore(
            error_metric=metric, aggregation=agg
            )
        assert isinstance(layer, tf.keras.layers.Layer) # Or Model if it inherits
        assert layer.error_metric == metric
        assert layer.aggregation == agg
    except Exception as e:
        pytest.fail(f"PredictionErrorAnomalyScore instantiation failed "
                    f"(metric={metric}, agg={agg}): {e}")
    print(f"PredictionErrorAnomalyScore instantiation (m={metric}, a={agg}): OK")

def test_prediction_error_score_call_and_output_shape(dummy_sequence_data):
    """Test call method and output shape of PredictionErrorAnomalyScore."""
    y_true = dummy_sequence_data
    # Create dummy predictions with some error
    y_pred = y_true + tf.random.normal(tf.shape(y_true), stddev=0.5)

    layer = PredictionErrorAnomalyScore(error_metric='mse', aggregation='max')
    try:
        scores = layer([y_true, y_pred], training=False)
    except Exception as e:
        pytest.fail(f"PredictionErrorAnomalyScore call failed: {e}")

    expected_shape = (BATCH_SIZE, 1) # Single score per sample
    assert scores.shape == expected_shape, \
        (f"Output shape mismatch. Expected {expected_shape}, "
         f"got {scores.shape}")
    assert scores.dtype == tf.float32
    print("PredictionErrorAnomalyScore call and output shape: OK")

def test_prediction_error_score_call_invalid_input(dummy_sequence_data):
    """Test PredictionErrorAnomalyScore call with invalid input list."""
    layer = PredictionErrorAnomalyScore()
    with pytest.raises(ValueError, match="Input must be a list or tuple:"):
        layer(dummy_sequence_data, training=False) # Not a list
    with pytest.raises(ValueError, match="Input must be a list or tuple:"):
        layer([dummy_sequence_data], training=False) # List of one
    print("PredictionErrorAnomalyScore invalid input call: OK")


def test_prediction_error_score_serialization(dummy_sequence_data):
    """Test get_config and from_config for PredictionErrorAnomalyScore."""
    y_true = dummy_sequence_data
    y_pred = y_true + 0.1

    layer1 = PredictionErrorAnomalyScore(error_metric='mse', aggregation='max')
    # Call to build (if necessary, depends on Model vs Layer inheritance)
    _ = layer1([y_true, y_pred])
    config = layer1.get_config()

    try:
        layer2 = PredictionErrorAnomalyScore.from_config(config)
    except Exception as e:
        pytest.fail(f"PredictionErrorAnomalyScore.from_config failed: {e}")

    assert layer2.error_metric == layer1.error_metric
    assert layer2.aggregation == layer1.aggregation
    try:
        scores1 = layer1([y_true, y_pred])
        scores2 = layer2([y_true, y_pred])
        assert np.allclose(scores1.numpy(), scores2.numpy())
    except Exception as e:
        pytest.fail(f"Output mismatch after from_config: {e}")
    print("PredictionErrorAnomalyScore serialization: OK")


# Allows running the tests directly if needed
if __name__=='__main__':
     pytest.main([__file__])

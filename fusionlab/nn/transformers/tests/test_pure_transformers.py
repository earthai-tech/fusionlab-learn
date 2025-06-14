# file: fusionlab/nn/transformers/tests/test_pure_transformers.py

import pytest
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Optional

try:
    from fusionlab.nn.transformers import TimeSeriesTransformer
    KERAS_BACKEND = True
except ImportError:
    KERAS_BACKEND = False

# Skip all tests if TensorFlow/Keras backend is not available
pytestmark = pytest.mark.skipif(
    not KERAS_BACKEND, reason="TensorFlow/Keras backend not available"
)

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def default_model_params() -> Dict[str, Any]:
    """Provides a standard set of initialization parameters."""
    return {
        "static_input_dim": 5,
        "dynamic_input_dim": 6,
        "future_input_dim": 4,
        "embed_dim": 32,
        "num_heads": 4,
        "ffn_dim": 64,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "forecast_horizon": 12,
        "output_dim": 1,
        "dropout_rate": 0.1,
        "input_dropout_rate": 0.1,
        "max_seq_len_encoder": 50,
        "max_seq_len_decoder": 20,
    }

@pytest.fixture
def dummy_inputs() -> Dict[str, tf.Tensor]:
    """Provides a standard dictionary of dummy input tensors."""
    batch_size = 8
    past_steps = 24
    horizon = 12
    return {
        "static": tf.random.normal((batch_size, 5)),
        "dynamic": tf.random.normal((batch_size, past_steps, 6)),
        "future": tf.random.normal((batch_size, horizon, 4)),
        "targets": tf.random.normal((batch_size, horizon, 1)),
    }

# --- Test Cases ---

def test_instantiation(default_model_params):
    """Tests basic model instantiation."""
    model = TimeSeriesTransformer(**default_model_params)
    assert isinstance(model, TimeSeriesTransformer)
    assert len(model.encoder_layers) == default_model_params["num_encoder_layers"]
    assert len(model.decoder_layers) == default_model_params["num_decoder_layers"]
    assert model.quantile_modeling is None

def test_instantiation_with_quantiles(default_model_params):
    """Tests instantiation with quantile forecasting enabled."""
    params = default_model_params.copy()
    params["quantiles"] = [0.1, 0.5, 0.9]
    model = TimeSeriesTransformer(**params)
    assert model.quantile_modeling is not None
    assert len(model.quantiles) == 3

@pytest.mark.parametrize("use_grn", [True, False])
def test_static_processor_creation(default_model_params, use_grn):
    """Tests the creation of the static feature processor."""
    params = default_model_params.copy()
    params["use_grn_for_static"] = use_grn
    model = TimeSeriesTransformer(**params)
    if use_grn:
        assert "GatedResidualNetwork" in model.static_processor.__class__.__name__
    else:
        assert "Dense" in model.static_processor.__class__.__name__

def test_call_with_all_inputs(default_model_params, dummy_inputs):
    """Tests the forward pass with all standard inputs."""
    model = TimeSeriesTransformer(**default_model_params)
    inputs = [dummy_inputs["static"], dummy_inputs["dynamic"], dummy_inputs["future"]]
    predictions = model(inputs, training=False)
    
    expected_shape = (
        dummy_inputs["dynamic"].shape[0],
        default_model_params["forecast_horizon"],
        default_model_params["output_dim"],
    )
    assert predictions.shape == expected_shape

def test_call_with_quantiles(default_model_params, dummy_inputs):
    """Tests the forward pass with quantile output."""
    params = default_model_params.copy()
    params["quantiles"] = [0.1, 0.5, 0.9]
    model = TimeSeriesTransformer(**params)
    inputs = [dummy_inputs["static"], dummy_inputs["dynamic"], dummy_inputs["future"]]
    predictions = model(inputs, training=False)
    
    # Shape should be (batch, horizon, num_quantiles) since output_dim=1
    expected_shape = (
        dummy_inputs["dynamic"].shape[0],
        params["forecast_horizon"],
        len(params["quantiles"]),
    )
    assert predictions.shape == expected_shape

def test_call_without_static_features(default_model_params, dummy_inputs):
    """Tests the forward pass when no static features are provided."""
    params = default_model_params.copy()
    params["static_input_dim"] = 0
    model = TimeSeriesTransformer(**params)
    # The first element in the list is None
    inputs = [None, dummy_inputs["dynamic"], dummy_inputs["future"]]
    predictions = model(inputs, training=False)
    
    expected_shape = (
        dummy_inputs["dynamic"].shape[0],
        params["forecast_horizon"],
        params["output_dim"],
    )
    assert predictions.shape == expected_shape

def test_call_without_future_features(default_model_params, dummy_inputs):
    """Tests the forward pass when no future features are provided."""
    params = default_model_params.copy()
    params["future_input_dim"] = 0
    model = TimeSeriesTransformer(**params)
    # The last element in the list is None
    inputs = [dummy_inputs["static"], dummy_inputs["dynamic"], None]
    predictions = model(inputs, training=False)
    
    expected_shape = (
        dummy_inputs["dynamic"].shape[0],
        params["forecast_horizon"],
        params["output_dim"],
    )
    assert predictions.shape == expected_shape

@pytest.mark.parametrize("mode", ['add_to_encoder_input', 'add_to_decoder_input', 'none'])
def test_static_integration_modes(default_model_params, dummy_inputs, mode):
    """Tests all static feature integration modes."""
    params = default_model_params.copy()
    params["static_integration_mode"] = mode
    model = TimeSeriesTransformer(**params)
    inputs = [dummy_inputs["static"], dummy_inputs["dynamic"], dummy_inputs["future"]]
    
    try:
        predictions = model(inputs, training=False)
        assert predictions is not None
    except Exception as e:
        pytest.fail(f"Model call failed for static_integration_mode='{mode}' with error: {e}")

def test_fit_and_evaluate(default_model_params, dummy_inputs):
    """Tests if the model can be compiled, trained, and evaluated."""
    model = TimeSeriesTransformer(**default_model_params)
    model.compile(optimizer="adam", loss="mse")
    
    inputs = [dummy_inputs["static"], dummy_inputs["dynamic"], dummy_inputs["future"]]
    targets = dummy_inputs["targets"]

    try:
        history = model.fit(inputs, targets, epochs=1, batch_size=4, verbose=0)
        assert "loss" in history.history
        
        eval_results = model.evaluate(inputs, targets, verbose=0)
        assert isinstance(eval_results, float)
    except Exception as e:
        pytest.fail(f"model.fit() or .evaluate() failed with an exception: {e}")

def test_serialization(default_model_params, dummy_inputs):
    """Tests model serialization and deserialization."""
    model1 = TimeSeriesTransformer(**default_model_params)
    # Build the model
    inputs = [dummy_inputs["static"], dummy_inputs["dynamic"], dummy_inputs["future"]]
    _ = model1(inputs)
    
    config = model1.get_config()
    assert isinstance(config, dict)
    
    # Recreate model from config
    model2 = TimeSeriesTransformer.from_config(config)
    assert model2.static_input_dim == model1.static_input_dim
    assert model2.num_encoder_layers == model1.num_encoder_layers
    
    # Check that the reloaded model can make a prediction
    try:
        predictions = model2(inputs)
        assert predictions is not None
    except Exception as e:
        pytest.fail(f"Prediction with reloaded model failed: {e}")

if __name__ == '__main__':
    pytest.main([__file__, '-vv'])
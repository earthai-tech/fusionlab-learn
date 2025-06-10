# test_halnet.py
import pytest
import numpy as np

import os
import tempfile
from typing import List, Dict, Tuple, Optional, Any

KERAS_BACKEND = False

try: 
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import MeanSquaredError
    
    # Adjust the import path based on your project structure
    from fusionlab.nn.models._halnet import HALNet
    from fusionlab.nn.losses import combined_quantile_loss 
    KERAS_BACKEND =True 
except: 
   pass 
# --- End Imports ---

# Skip all tests if TF/Keras backend is not available
pytestmark = pytest.mark.skipif(
    not KERAS_BACKEND, reason="TensorFlow/Keras backend not available"
)

# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def default_model_params() -> Dict[str, Any]:
    """Provides a default set of parameters for HALNet."""
    return {
        "static_input_dim": 3,
        "dynamic_input_dim": 5,
        "future_input_dim": 2,
        "output_dim": 1,
        "forecast_horizon": 4,
        "max_window_size": 8,
        "embed_dim": 16,
        "hidden_units": 16,
        "lstm_units": 16,
        "attention_units": 8,
        "num_heads": 2,
        "dropout_rate": 0.0,
        "vsn_units": 8,
        "mode": 'tft_like', # use future_span (time_steps + forecast_hirizon)
    }

@pytest.fixture
def dummy_input_data(default_model_params: Dict[str, Any]) -> Tuple[List[tf.Tensor], int]:
    """Generates dummy input data for HALNet."""
    batch_size = 4
    time_steps = default_model_params["max_window_size"]
    
    if default_model_params['mode'] =='pihal_like': 
        future_span = default_model_params["forecast_horizon"]
    else: 
        # Note: For HALNet, future input spans the full range needed by the model
        # in tft_like
        future_span = time_steps + default_model_params["forecast_horizon"]

    static_features = tf.random.normal(
        (batch_size, default_model_params["static_input_dim"]))
    dynamic_features = tf.random.normal(
        (batch_size, time_steps, default_model_params["dynamic_input_dim"]))
    future_features = tf.random.normal(
        (batch_size, future_span, default_model_params["future_input_dim"]))
    
    # HALNet expects a list of [static, dynamic, future]
    return [static_features, dynamic_features, future_features], batch_size

# --- Test Functions ---

def test_halnet_instantiation(default_model_params: Dict[str, Any]):
    """Tests if HALNet can be instantiated with default parameters."""
    try:
        model = HALNet(**default_model_params)
        assert isinstance(model, HALNet)
        assert model.name == "HALNet"
    except Exception as e:
        pytest.fail(f"HALNet instantiation failed with default params: {e}")

@pytest.mark.parametrize("use_vsn", [True, False])
@pytest.mark.parametrize("static_dim", [0, 3])
@pytest.mark.parametrize("future_dim", [0, 2])
def test_halnet_instantiation_variations(
    default_model_params: Dict[str, Any], use_vsn: bool, static_dim: int, future_dim: int
):
    """Tests model instantiation with varying VSN and input dimension settings."""
    params = default_model_params.copy()
    params["use_vsn"] = use_vsn
    params["static_input_dim"] = static_dim
    params["future_input_dim"] = future_dim

    # HALNet's internal logic requires at least one dynamic feature
    assert params["dynamic_input_dim"] > 0
    
    model = HALNet(**params)
    assert isinstance(model, HALNet)
    assert model.use_vsn == use_vsn

def test_call_shapes_point_prediction(
    default_model_params: Dict[str, Any], dummy_input_data: Tuple[List[tf.Tensor], int]
):
    """Tests the output shape for a non-quantile (point) forecast."""
    params = default_model_params.copy()
    params["quantiles"] = None
    model = HALNet(**params)
    
    inputs, batch_size = dummy_input_data
    outputs = model(inputs, training=False)
    
    expected_shape = (
        batch_size,
        params["forecast_horizon"],
        params["output_dim"]
    )
    assert outputs.shape == expected_shape, \
        f"Expected point prediction shape {expected_shape}, but got {outputs.shape}"

def test_call_shapes_quantile_prediction(
    default_model_params: Dict[str, Any], dummy_input_data: Tuple[List[tf.Tensor], int]
):
    """Tests the output shape for a quantile forecast."""
    params = default_model_params.copy()
    my_quantiles = [0.1, 0.5, 0.9]
    params["quantiles"] = my_quantiles
    model = HALNet(**params)
    
    inputs, batch_size = dummy_input_data
    outputs = model(inputs, training=False)
    
    expected_shape = (
        batch_size,
        params["forecast_horizon"],
        len(my_quantiles), 
        params["output_dim"],
    )
    assert outputs.shape == expected_shape, \
        f"Expected quantile prediction shape {expected_shape}, but got {outputs.shape}"

def test_halnet_fit_and_serialization(
    default_model_params: Dict[str, Any], dummy_input_data: Tuple[List[tf.Tensor], int]
):
    """
    Tests model training, saving, and reloading to ensure the full
    workflow is functional.
    """
    params = default_model_params.copy()
    params["quantiles"] = None # Use simple MSE for this test
    model = HALNet(**params)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

    inputs, batch_size = dummy_input_data
    # Create dummy targets matching the model's output shape
    targets = tf.random.normal((
        batch_size, params["forecast_horizon"], params["output_dim"]
    ))

    # Test initial fit
    history = model.fit(inputs, targets, epochs=2, batch_size=2, verbose=0)
    assert "loss" in history.history
    assert not np.isnan(history.history["loss"][-1])

    # Test saving and loading
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "halnet_test_model.keras")
        model.save(model_path)
        
        # Load with custom objects dictionary
        reloaded_model = load_model(
            model_path,
            custom_objects={"HALNet": HALNet}
        )
        assert isinstance(reloaded_model, HALNet)

        # Check if reloaded model can continue training
        reloaded_history = reloaded_model.fit(
            inputs, targets, epochs=1, batch_size=2, verbose=0
        )
        assert "loss" in reloaded_history.history
        assert not np.isnan(reloaded_history.history["loss"][0])

def test_get_config_from_config(
    default_model_params: Dict[str, Any], dummy_input_data: Tuple[List[tf.Tensor], int]
):
    """Verifies that model configuration can be correctly serialized and deserialized."""
    model1 = HALNet(**default_model_params)
    inputs, _ = dummy_input_data
    # Build the model by calling it
    _ = model1(inputs)
    
    config = model1.get_config()
    assert isinstance(config, dict)
    
    # Recreate model from config
    model2 = HALNet.from_config(config.copy())
    config2 = model2.get_config()

    # Compare relevant config parameters
    for key in default_model_params:
        if key in config and key in config2:
            val1 = config[key]
            val2 = config2[key]
            if isinstance(val1, (list, tuple)):
                assert list(val1 or []) == list(val2 or []), \
                    f"Config mismatch for list key: {key}"
            else:
                assert val1 == val2, f"Config mismatch for key: {key}"


if __name__=="__main__": 
    pytest.main([__file__, '-vv'])
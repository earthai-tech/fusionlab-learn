# File: fusionlab/nn/pinn/tests/test_geos.py
import pytest
import numpy as np
import tempfile
import os
from typing import Dict, Any, List, Tuple

try: 
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    # Adjust import paths based on your project structure
    from fusionlab.nn.pinn._geos import PITGWFlow
    from fusionlab.params import LearnableK, LearnableSs, LearnableQ
    from fusionlab.nn import KERAS_BACKEND 
except Exception as e : 
    print(f"Skipping combine_temporal_inputs_for_lstm tests due to"
          f" import error: {e}")
    KERAS_BACKEND = False
# --- End Imports ---

# Skip all tests in this file if TensorFlow/Keras backend is not available
pytestmark = pytest.mark.skipif(
    not KERAS_BACKEND, reason="TensorFlow/Keras backend not available"
)
# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def default_model_params() -> Dict[str, Any]:
    """Provides a standard set of initialization parameters for the model."""
    return {
        "hidden_units": [20, 20, 20], # Use a list of 3 for consistency
        "activation": "tanh",
        "learning_rate": 1e-3,
        "K": 1.5,
        "Ss": 5e-5,
        "Q": 0.0,
    }

@pytest.fixture
def collocation_points_dict() -> Dict[str, tf.Tensor]:
    """Provides a standard dictionary of coordinate tensors."""
    n_points = 16
    tf.random.set_seed(42)
    return {
        "t": tf.random.uniform((n_points, 1), dtype=tf.float32),
        "x": tf.random.uniform((n_points, 1), dtype=tf.float32),
        "y": tf.random.uniform((n_points, 1), dtype=tf.float32),
    }

# --- Test Cases ---

def test_pitgwflow_instantiation(default_model_params):
    """Tests basic model instantiation with default float parameters."""
    model = PITGWFlow(**default_model_params)
    assert isinstance(model, PITGWFlow)
    assert model.net is not None
    # InputLayer is not counted in .layers, so 3 hidden + 1 output = 4
    assert len(model.net.layers) == 4

@pytest.mark.parametrize("hidden_units_config", [[10, 10, 10], [20], 16])
def test_pitgwflow_hidden_units_flexibility(default_model_params, hidden_units_config):
    """Tests that hidden_units can be an int or a list."""
    params = default_model_params.copy()
    params["hidden_units"] = hidden_units_config
    model = PITGWFlow(**params)
    
    # The logic in __init__ ensures there are always 3 hidden layers.
    # Total layers = 3 Hidden + 1 Output = 4
    assert len(model.net.layers) == 4

def test_pitgwflow_learnable_params(default_model_params):
    """Tests instantiation with learnable physical parameters."""
    params = default_model_params.copy()
    params["K"] = LearnableK(initial_value=0.5)
    params["Ss"] = LearnableSs(initial_value=1e-4)
    model = PITGWFlow(**params)
    
    trainable_vars_names = [v.name for v in model.trainable_variables]
    assert any("param_K" in name for name in trainable_vars_names)
    assert any("param_Ss" in name for name in trainable_vars_names)

def test_call_with_dict_input(default_model_params, collocation_points_dict):
    """Tests the forward pass with a dictionary of coordinate tensors."""
    model = PITGWFlow(**default_model_params)
    outputs = model(collocation_points_dict) # # sqeeeze last
    
    # The call method returns a 3D tensor (batch, time, features)
    # where time=1 for this input format.
    expected_shape = (collocation_points_dict['t'].shape[0], 1)  
    assert outputs.shape == expected_shape

def test_compute_residual(default_model_params, collocation_points_dict):
    """Tests the PDE residual computation."""
    model = PITGWFlow(**default_model_params)
    
    # For compute_residual, inputs must be tf.Variable watched by a tape
    t = tf.Variable(collocation_points_dict['t'])
    x = tf.Variable(collocation_points_dict['x'])
    y = tf.Variable(collocation_points_dict['y'])
    
    residual = model.compute_residual(coords={"t": t, "x": x, "y": y})
    assert residual is not None
    assert residual.shape == (t.shape[0], 1)
    assert not tf.reduce_all(tf.math.is_nan(residual))

def test_train_step_and_fit(default_model_params, collocation_points_dict):
    """
    Tests the custom train_step by running model.fit() for one epoch.
    This is a key integration test.
    """
    params = default_model_params.copy()
    params["K"] = LearnableK(0.5) # Ensure there's something to learn
    model = PITGWFlow(**params)

    # Re-compile to include metrics for history tracking
    model.compile(optimizer=model.optimizer, loss="mse", metrics=["mae"])
    
    # Create a tf.data.Dataset
    dummy_targets = tf.zeros_like(collocation_points_dict['t'])
    dataset = tf.data.Dataset.from_tensor_slices(
        (collocation_points_dict, dummy_targets)
    ).batch(4)

    try:
        history = model.fit(dataset, epochs=1, verbose=0)
        assert history is not None
        assert "pde_loss" in history.history
        assert not np.isnan(history.history["pde_loss"][0])
    except Exception as e:
        pytest.fail(f"model.fit() failed with an exception: {e}")

def test_serialization_and_reloading(default_model_params, collocation_points_dict):
    """Tests that the model can be saved, reloaded, and still used."""
    params = default_model_params.copy()
    params["K"] = LearnableK(0.8)
    params["Ss"] = 1.2e-4
    
    model1 = PITGWFlow(**params)
    # Build the model by calling it on data
    _ = model1(collocation_points_dict)
    
    config = model1.get_config()
    assert isinstance(config, dict)
    assert isinstance (config['K'], LearnableK) is True
    assert isinstance (config['Ss'], LearnableSs)  is False

    # Recreate model from config
    model2 = PITGWFlow.from_config(config.copy())
    
    assert model2.hidden_units == model1.hidden_units
    assert isinstance(model2.K_config, LearnableK)
    assert isinstance(model2.Ss_config, float)
    assert np.isclose(model2.K_config.initial_value, 0.8)

    # Check that the reloaded model can make a prediction
    try:
        predictions = model2.predict(collocation_points_dict, verbose=0)
        expected_shape = (collocation_points_dict['t'].shape[0], 1)
        assert predictions.shape == expected_shape
    except Exception as e:
        pytest.fail(f"Prediction with reloaded model failed: {e}")

        
if __name__=='__main__': 
    pytest.main( [__file__, '-vv'])
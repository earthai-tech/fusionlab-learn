
import pytest
import numpy as np
import tempfile
import os
from typing import Dict, Any

try:
    import tensorflow as tf
    from fusionlab.nn.pinn._gw_models import PiTGWFlow
    from fusionlab.params import LearnableK, LearnableSs, LearnableQ
    from fusionlab.nn import KERAS_BACKEND
except ImportError as e:
    KERAS_BACKEND = False

    # Skip all tests in this file if TensorFlow/Keras backend is not available
    pytestmark = pytest.mark.skipif(
        not KERAS_BACKEND, reason=f"Skipping tests due to import error: {e}"
    )

# --- Pytest Fixtures ---
@pytest.fixture(scope="module")
def default_model_params() -> Dict[str, Any]:
    """Provides a standard set of initialization parameters for the model."""
    return {
        "hidden_units": [20, 20, 20],
        "activation": "tanh",
        "learning_rate": 1e-3,
        "K": 1.5,
        "Ss": 5e-5,
        "Q": 0.0,
    }

@pytest.fixture
def collocation_points_dataset() -> tf.data.Dataset:
    """Provides a tf.data.Dataset of coordinate tensors for fitting and evaluating."""
    n_points = 16
    tf.random.set_seed(42)
    coords = {
        "t": tf.random.uniform((n_points, 1), dtype=tf.float32),
        "x": tf.random.uniform((n_points, 1), dtype=tf.float32),
        "y": tf.random.uniform((n_points, 1), dtype=tf.float32),
    }
    # Dummy targets are required for Keras fit/evaluate, but unused by our PINN
    dummy_targets = tf.zeros_like(coords['t'])
    return tf.data.Dataset.from_tensor_slices((coords, dummy_targets)).batch(8)

@pytest.fixture
def collocation_points_dict(collocation_points_dataset) -> Dict[str, tf.Tensor]:
    """Provides a standard dictionary of coordinate tensors from the dataset."""
    for data in collocation_points_dataset.unbatch().batch(16).take(1):
        return data[0]

# --- Test Cases ---

def test_pitgwflow_instantiation(default_model_params):
    """Tests basic model instantiation with default float parameters."""
    model = PiTGWFlow(**default_model_params)
    assert isinstance(model, PiTGWFlow)
    assert model.coord_mlp is not None
    # 3 hidden layers + 1 output layer = 4 layers in the Sequential MLP
    assert len(model.coord_mlp.layers) == 4

@pytest.mark.parametrize("hidden_units_config", [[10, 10, 10], [20], 16])
def test_pitgwflow_hidden_units_flexibility(default_model_params, hidden_units_config):
    """Tests that hidden_units can be an int or a list of varying lengths."""
    params = default_model_params.copy()
    params["hidden_units"] = hidden_units_config
    model = PiTGWFlow(**params)
    # The __init__ logic ensures there are always 3 hidden layers + 1 output
    assert len(model.coord_mlp.layers) == 4

def test_pitgwflow_learnable_params(default_model_params):
    """Tests instantiation with learnable physical parameters."""
    params = default_model_params.copy()
    params["K"] = LearnableK(initial_value=0.5)
    params["Ss"] = LearnableSs(initial_value=1e-4)
    model = PiTGWFlow(**params)

    trainable_vars_names = [v.name for v in model.trainable_variables]
    assert any("param_K" in name for name in trainable_vars_names)
    assert any("param_Ss" in name for name in trainable_vars_names)

def test_call_with_dict_input(default_model_params, collocation_points_dict):
    """Tests the forward pass with a dictionary of coordinate tensors."""
    model = PiTGWFlow(**default_model_params)
    outputs = model(collocation_points_dict)

    # Input shape for 't' is (16, 1), so batch=16, time=1.
    # Output shape should be (batch, time, 1) -> (16, 1, 1)
    expected_shape = (collocation_points_dict['t'].shape[0], 1, 1)
    assert outputs.shape == expected_shape

def test_train_step_and_fit(default_model_params, collocation_points_dataset):
    """Tests the custom train_step by running model.fit() for one epoch."""
    params = default_model_params.copy()
    params["K"] = LearnableK(0.5)  # Ensure there's a trainable variable
    model = PiTGWFlow(**params)

    # Keras requires compilation before fitting. The loss/metrics are for API
    # compatibility; our train_step uses its own pde_loss.
    model.compile(optimizer=model.optimizer)

    try:
        history = model.fit(collocation_points_dataset, epochs=1, verbose=0)
        assert history is not None
        assert "pde_loss" in history.history
        assert not np.isnan(history.history["pde_loss"][0])
    except Exception as e:
        pytest.fail(f"model.fit() failed with an exception: {e}")

def test_test_step_and_evaluate(default_model_params, collocation_points_dataset):
    """Tests the custom test_step by running model.evaluate()."""
    model = PiTGWFlow(**default_model_params)
    model.compile(optimizer=model.optimizer) # Compile is necessary for evaluate

    try:
        results = model.evaluate(collocation_points_dataset, verbose=0, return_dict=True)
        assert "pde_loss" in results
        assert not np.isnan(results["pde_loss"])
    except Exception as e:
        pytest.fail(f"model.evaluate() failed with an exception: {e}")

def test_serialization_and_reloading(default_model_params, collocation_points_dict):
    """Tests that the model can be configured, reloaded, and still used."""
    params = default_model_params.copy()
    params["K"] = LearnableK(0.8)
    params["Ss"] = 1.2e-4

    model1 = PiTGWFlow(**params)
    _ = model1(collocation_points_dict) # Build model

    config = model1.get_config()
    assert isinstance(config, dict)
    assert isinstance(config['K'], float)
    assert isinstance(config['Ss'], float)

    # Recreate model from config
    model2 = PiTGWFlow.from_config(config.copy())

    assert model2.activation == model1.activation
    assert isinstance(model2.K_config, float)
    assert np.isclose(model2.K.initial_value, 0.8)
    assert np.isclose(model2.Ss_config, 1.2e-4)

    # Check that the reloaded model can make a prediction
    try:
        predictions = model2(collocation_points_dict)
        # Expected shape (batch, time, 1) -> (16, 1, 1)
        expected_shape = (collocation_points_dict['t'].shape[0], 1, 1)
        assert predictions.shape == expected_shape
    except Exception as e:
        pytest.fail(f"Prediction with reloaded model failed: {e}")

if __name__ == '__main__':
    pytest.main([__file__, '-vv'])
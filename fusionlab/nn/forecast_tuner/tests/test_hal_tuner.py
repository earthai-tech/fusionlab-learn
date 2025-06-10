
import pytest
import numpy as np

import tempfile
from typing import Dict, Any, List, Tuple

# Keras Tuner might be an optional dependency for the library,
# but it's a requirement for running the tuner tests.
HAS_KT=False 
try: 
    import keras_tuner as kt
    import tensorflow as tf
    # Adjust import paths based on your project structure
    from fusionlab.nn.forecast_tuner.hal_tuner import HALTuner
    from fusionlab.nn.models import HALNet
    from tensorflow.keras.losses import MeanSquaredError
    HAS_KT=True 
    FUSIONLAB_INSTALLED =True 
except: 
    FUSIONLAB_INSTALLED=False 


pytestmark = pytest.mark.skipif(
    not (FUSIONLAB_INSTALLED and HAS_KT),
    reason="Keras Tuner or fusionlab components not found"
)
# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def default_fixed_params() -> Dict[str, Any]:
    """Provides a default set of fixed parameters for HALNet."""
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
    }

def generate_dummy_data(
    params: Dict[str, Any], batch_size: int = 4
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Helper to generate dummy data based on HALNet mode."""
    time_steps = params["max_window_size"]
    horizon = params["forecast_horizon"]
    
    # Determine the required length of the future_features tensor
    if params.get('mode', 'tft_like') == 'tft_like':
        future_span = time_steps + horizon
    else: # pihal_like
        future_span = horizon
        
    static_data = np.random.rand(
        batch_size, params["static_input_dim"]).astype(np.float32)
    dynamic_data = np.random.rand(
        batch_size, time_steps, params["dynamic_input_dim"]).astype(np.float32)
    future_data = np.random.rand(
        batch_size, future_span, params["future_input_dim"]).astype(np.float32)
    
    targets = np.random.rand(
        batch_size, horizon, params["output_dim"]).astype(np.float32)
        
    return [static_data, dynamic_data, future_data], targets


# --- Test Functions ---

@pytest.mark.skipif(kt is None, reason="KerasTuner is not installed.")
def test_haltuner_initialization(default_fixed_params):
    """Tests successful instantiation and validation of fixed params."""
    # Test successful creation
    tuner = HALTuner(
        fixed_model_params=default_fixed_params,
        project_name="test_init_success"
    )
    assert isinstance(tuner, HALTuner)
    assert tuner.project_name == "test_init_success"

    # Test failure when a required parameter is missing
    incomplete_params = default_fixed_params.copy()
    del incomplete_params["dynamic_input_dim"]
    with pytest.raises(ValueError, match="Missing required key"):
        HALTuner(fixed_model_params=incomplete_params)

@pytest.mark.skipif(kt is None, reason="KerasTuner is not installed.")
def test_haltuner_create_method(default_fixed_params):
    """Tests the .create() classmethod for correct dimension inference."""
    params = default_fixed_params.copy()
    inputs_data, targets_data = generate_dummy_data(params)
    
    tuner = HALTuner.create(
        inputs_data=inputs_data,
        targets_data=targets_data,
        project_name="test_create"
    )
    
    # Check if inferred dimensions match the source data
    assert tuner.fixed_model_params["static_input_dim"] == inputs_data[0].shape[-1]
    assert tuner.fixed_model_params["dynamic_input_dim"] == inputs_data[1].shape[-1]
    # Note: HALNet create method doesn't know about `tft_like` slicing,
    # so future_input_dim matches the full provided tensor.
    assert tuner.fixed_model_params["future_input_dim"] == inputs_data[2].shape[-1]
    assert tuner.fixed_model_params["output_dim"] == targets_data.shape[-1]
    assert tuner.fixed_model_params["forecast_horizon"] == targets_data.shape[1]
    assert tuner.fixed_model_params["max_window_size"] == inputs_data[1].shape[1]

@pytest.mark.skipif(kt is None, reason="KerasTuner is not installed.")
def test_haltuner_build_model_compiles(default_fixed_params):
    """
    Tests that the build method constructs a Keras model that can be
    compiled without errors.
    """
    tuner = HALTuner(
        fixed_model_params=default_fixed_params,
        project_name="test_build"
    )
    
    # Create a dummy HyperParameters object
    hp = kt.HyperParameters()
    
    try:
        model = tuner.build(hp)
        assert isinstance(model, tf.keras.Model)
        assert model.optimizer is not None
        assert model.loss is not None
    except Exception as e:
        pytest.fail(f"HALTuner.build(hp) failed with an exception: {e}")

@pytest.mark.parametrize("mode", ["pihal_like", "tft_like"])
@pytest.mark.skipif(kt is None, reason="KerasTuner is not installed.")
def test_haltuner_run_integration(default_fixed_params, mode):
    """
    An integration test to verify the end-to-end `run` workflow for
    both 'pihal_like' and 'tft_like' modes.
    """
    params = default_fixed_params.copy()
    params['mode'] = mode
    
    # Generate data with the correct shapes for the specified mode
    inputs_data, targets_data = generate_dummy_data(params)
    
    # Use a temporary directory for tuner results to avoid clutter
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = HALTuner(
            fixed_model_params=params,
            max_trials=1,
            executions_per_trial=1,
            directory=tmpdir,
            project_name=f"test_run_{mode}",
            overwrite=True
        )
        
        # Split data for validation
        train_inputs = [arr[:-1] for arr in inputs_data]
        val_inputs = [arr[-1:] for arr in inputs_data]
        train_targets, val_targets = targets_data[:-1], targets_data[-1:]
        
        try:
            best_model, best_hps, _ = tuner.run(
                inputs=train_inputs,
                y=train_targets,
                validation_data=(val_inputs, val_targets),
                epochs=1, # Just one epoch for a quick test
                batch_size=2
            )
            
            assert isinstance(best_model, tf.keras.Model)
            assert isinstance(best_hps, kt.HyperParameters)

        except Exception as e:
            pytest.fail(
                f"HALTuner.run() failed for mode='{mode}' with "
                f"exception: {e}"
            )

if __name__=="__main__": 
    pytest.main([__file__, '-vv'])
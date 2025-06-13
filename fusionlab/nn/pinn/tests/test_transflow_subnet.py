
import pytest
import numpy as np

import tempfile
import os
from typing import Dict, Any,  Tuple

try: 
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    
    from fusionlab.nn.pinn._transflow_subnet import TransFlowSubsNet
    from fusionlab.params import LearnableK, LearnableSs, LearnableQ, LearnableC
    from tensorflow.keras.optimizers import Adam

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
    """Provides a default set of initialization parameters for the model."""
    return {
        "static_input_dim": 2,
        "dynamic_input_dim": 3,
        "future_input_dim": 1,
        "output_subsidence_dim": 1,
        "output_gwl_dim": 1,
        "forecast_horizon": 6,
        "max_window_size": 5,  # time_steps
        "embed_dim": 16,
        "hidden_units": 16,
        "lstm_units": 16,
        "attention_units": 8,
        "num_heads": 2,
        "use_vsn": False,
    }

@pytest.fixture
def dummy_data_generator():
    """
    A factory fixture to generate correctly shaped dummy data for both
    'pihal_like' and 'tft_like' modes.
    """
    def _generate(
        params: Dict[str, Any], batch_size: int = 2
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        
        time_steps = params["max_window_size"]
        horizon = params["forecast_horizon"]
        mode = params.get('mode', 'pihal_like') # Use 'mode' as per __init__

        # Determine the required length of the future_features tensor
        if 'tft' in mode:
            future_span = time_steps + horizon
        else: # pihal_like
            future_span = horizon

        # --- Generate Data ---
        static_features = np.random.rand(
            batch_size, params["static_input_dim"]).astype("float32")
        dynamic_features = np.random.rand(
            batch_size, time_steps, params["dynamic_input_dim"]).astype("float32")
        
        # Coords for the FORECAST window
        forecast_coords = np.random.rand(
            batch_size, horizon, 3).astype("float32")
        
        future_features = np.random.rand(
            batch_size, future_span, params["future_input_dim"]).astype("float32")
        
        subs_targets = np.random.rand(
            batch_size, horizon, params["output_subsidence_dim"]).astype("float32")
        gwl_targets = np.random.rand(
            batch_size, horizon, params["output_gwl_dim"]).astype("float32")

        inputs = {
            "coords": forecast_coords,
            "static_features": static_features,
            "dynamic_features": dynamic_features,
            "future_features": future_features,
        }
        targets = {"subs_pred": subs_targets, "gwl_pred": gwl_targets}
        return inputs, targets

    return _generate

# --- Test Cases ---

@pytest.mark.parametrize("mode", ["pihal_like", "tft_like"])
def test_transflowsubsnet_instantiation(default_model_params, mode):
    """Tests if the model instantiates correctly in both modes."""
    params = default_model_params.copy()
    params['mode'] = mode
    
    try:
        model = TransFlowSubsNet(**params)
        assert isinstance(model, TransFlowSubsNet)
        assert model._mode == mode
    except Exception as e:
        pytest.fail(f"Instantiation failed for mode='{mode}' with error: {e}")

def test_transflowsubsnet_learnable_params(default_model_params):
    """Tests instantiation with learnable physical parameters."""
    params = default_model_params.copy()
    params["K"] = LearnableK(initial_value=0.5)
    params["pinn_coefficient_C"] = LearnableC(initial_value=0.01)
    
    model = TransFlowSubsNet(**params)
    
    trainable_vars_names = [v.name for v in model.trainable_variables]
    assert any("param_K" in name for name in trainable_vars_names)
    assert any("log_pinn_coefficient_C" in name for name in trainable_vars_names)

@pytest.mark.parametrize("mode", ["pihal_like", "tft_like"])
def test_transflowsubsnet_call_forward_pass(
    default_model_params, dummy_data_generator, mode
):
    """Tests the forward pass (call method) for both modes."""
    params = default_model_params.copy()
    params['mode'] = mode
    
    inputs_dict, _ = dummy_data_generator(params)
    
    model = TransFlowSubsNet(**params)
    
    try:
        outputs = model(inputs_dict, training=False)
        assert isinstance(outputs, dict)
        assert "subs_pred" in outputs
        assert "gwl_pred" in outputs
        assert "gwl_pred_mean" in outputs
        assert "subs_pred_mean" in outputs
        
        # Check output shapes
        expected_shape = (
            inputs_dict['coords'].shape[0],
            params["forecast_horizon"],
            params["output_subsidence_dim"]
        )
        assert outputs['subs_pred'].shape == expected_shape
        
    except Exception as e:
        pytest.fail(f"Model call failed for mode='{mode}' with error: {e}")

@pytest.mark.parametrize("mode", ["pihal_like", "tft_like"])
def test_transflowsubsnet_fit_integration(
    default_model_params, dummy_data_generator, mode
):
    """Tests the end-to-end training process with model.fit()."""
    params = default_model_params.copy()
    params['mode'] = mode
    
    inputs_dict, targets_dict = dummy_data_generator(params, batch_size=4)
    
    model = TransFlowSubsNet(**params)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={"subs_pred": "mse", "gwl_pred": "mse"},
        lambda_cons=0.1,
        lambda_gw=0.1
    )
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (inputs_dict, targets_dict)).batch(2)

    try:
        history = model.fit(dataset, epochs=1, verbose=0)
        assert history is not None
        assert "total_loss" in history.history
        assert not np.isnan(history.history["total_loss"][0])
    except Exception as e:
        pytest.fail(f"model.fit() failed for mode='{mode}' with error: {e}")

def test_serialization(default_model_params, dummy_data_generator):
    """Tests that the model can be saved, reloaded, and still used."""
    params = default_model_params.copy()
    params["K"] = LearnableK(0.8) # Mix learnable and fixed
    params["Ss"] = 1e-7 
    model1 = TransFlowSubsNet(**params)
    inputs_dict, _ = dummy_data_generator(params)
    _ = model1(inputs_dict) # Build model

    config = model1.get_config()
    assert isinstance(config, dict)
    assert isinstance(config['K'], LearnableK)
    assert isinstance(config['Ss'], float)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "tgwdsnet_model.keras")
        model1.save(model_path)
        
        reloaded_model = load_model(
            model_path,
            custom_objects={"TransFlowSubsNet": TransFlowSubsNet}
        )
        assert isinstance(reloaded_model, TransFlowSubsNet)
        
        # Check a parameter
        assert isinstance(reloaded_model.K_config, LearnableK)
        assert np.isclose(reloaded_model.K_config.initial_value, 0.8)

        # Check prediction
        predictions = reloaded_model.predict(inputs_dict, verbose=0)
        assert "subs_pred" in predictions
        assert predictions['subs_pred'].shape[0] == inputs_dict['coords'].shape[0]

def test_serialization_2(default_model_params, dummy_data_generator):
    """Tests that the model can be saved, reloaded, and still used."""
    params = default_model_params.copy()
    params["K"] = LearnableK(0.8) # Learnable only
    params["Ss"] = LearnableSs(0.8)
    params["Q"] = LearnableQ(0.8)
    params["pinn_coefficient_C"]="learnable"
    
    model1 = TransFlowSubsNet(**params)
    inputs_dict, _ = dummy_data_generator(params)
    _ = model1(inputs_dict) # Build model

    config = model1.get_config()
    assert isinstance(config, dict)
    assert isinstance(config['K'], LearnableK)
    assert isinstance(config['Ss'], LearnableSs)
    assert isinstance(config['Q'], LearnableQ)
    assert isinstance(config["pinn_coefficient_C"], str) # learnable 

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "tgwdsnet_model.keras")
        model1.save(model_path)
        
        reloaded_model = load_model(
            model_path,
            custom_objects={
                "TransFlowSubsNet": TransFlowSubsNet, 
                "LearnableK":       LearnableK,
                "LearnableSs":      LearnableSs,
                "LearnableQ":       LearnableQ,
                }
        )
        assert isinstance(reloaded_model, TransFlowSubsNet)
        
        # Check a parameter
        assert isinstance(reloaded_model.K_config, LearnableK)
        assert np.isclose(reloaded_model.K_config.initial_value, 0.8)

        # Check prediction
        predictions = reloaded_model.predict(inputs_dict, verbose=0)
        assert "subs_pred" in predictions
        assert predictions['subs_pred'].shape[0] == inputs_dict['coords'].shape[0]
        
if __name__ =='__main__': # pragma : no cover 
    pytest.main( [__file__,  "--maxfail=1 ", "--disable-warnings",  "-q"])
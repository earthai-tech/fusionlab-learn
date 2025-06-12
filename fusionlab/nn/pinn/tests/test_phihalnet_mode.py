
import pytest
import numpy as np

from typing import Dict, Any,Tuple

try:
    # import tensorflow as tf
    from fusionlab.nn.pinn._pihal import PIHALNet
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.optimizers import Adam
    from tensorflow.data import Dataset
    from fusionlab.nn import KERAS_BACKEND
except ImportError as e:
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
    """Provides a default set of fixed parameters for PIHALNet."""
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
        "use_vsn": False, # Simpler to test non-VSN path first
    }

@pytest.fixture
def dummy_data_generator():
    """
    A factory fixture to generate dummy data based on model parameters,
    correctly handling shapes for both 'pihal_like' and 'tft_like' modes.
    """
    def _generate(
        params: Dict[str, Any], batch_size: int = 2
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        
        time_steps = params["max_window_size"]
        horizon = params["forecast_horizon"]
        mode = params.get('mode', 'pihal_like') # Default to pihal_like

        # Determine the required length of the future_features tensor
        if mode == 'tft_like':
            future_span = time_steps + horizon
        else: # pihal_like
            future_span = horizon

        # --- Generate Data ---
        static_features = np.random.rand(
            batch_size, params["static_input_dim"]
        ).astype("float32")
        dynamic_features = np.random.rand(
            batch_size, time_steps, params["dynamic_input_dim"]
        ).astype("float32")
        
        # This is the coordinates tensor for the FORECAST window
        forecast_coords = np.random.rand(
            batch_size, horizon, 3
        ).astype("float32")
        
        # This is the future known features tensor
        future_features = np.random.rand(
            batch_size, future_span, params["future_input_dim"]
        ).astype("float32")
        
        # Targets match the forecast horizon
        subs_targets = np.random.rand(
            batch_size, horizon, params["output_subsidence_dim"]
        ).astype("float32")
        gwl_targets = np.random.rand(
            batch_size, horizon, params["output_gwl_dim"]
        ).astype("float32")

        # Package inputs and targets into dictionaries
        inputs = {
            "coords": forecast_coords,
            "static_features": static_features,
            "dynamic_features": dynamic_features,
            "future_features": future_features,
        }
        targets = {
            "subs_pred": subs_targets,
            "gwl_pred": gwl_targets,
        }
        return inputs, targets

    return _generate

# --- Test Cases ---

@pytest.mark.parametrize("mode", ["pihal_like", "tft_like"])
def test_pihalnet_instantiation(default_model_params, mode):
    """
    Tests if PIHALNet instantiates correctly in both modes.
    """
    params = default_model_params.copy()
    params['mode'] = mode
    
    try:
        model = PIHALNet(**params)
        assert isinstance(model, PIHALNet)
        assert model.mode == mode
    except Exception as e:
        pytest.fail(
            f"PIHALNet instantiation failed for mode='{mode}' with error: {e}"
        )

@pytest.mark.parametrize("mode", ["pihal_like", "tft_like"])
def test_pihalnet_call_forward_pass(
    default_model_params, dummy_data_generator, mode
):
    """
    Tests the forward pass (call method) for both modes to ensure
    internal shape compatibility.
    """
    params = default_model_params.copy()
    params['mode'] = mode
    
    # Generate data with the correct shapes for the specified mode
    inputs_dict, _ = dummy_data_generator(params)
    batch_size = inputs_dict['coords'].shape[0]
    
    model = PIHALNet(**params)
    
    try:
        outputs = model(inputs_dict, training=False)
        assert isinstance(outputs, dict)
        assert "subs_pred" in outputs
        assert "gwl_pred" in outputs
        assert "pde_residual" in outputs
        
        # Check output shapes
        expected_shape = (
            batch_size,
            params["forecast_horizon"],
            params["output_subsidence_dim"]
        )
        assert outputs['subs_pred'].shape == expected_shape
        
    except Exception as e:
        pytest.fail(
            f"Model call failed for mode='{mode}' with error: {e}"
        )

@pytest.mark.parametrize("mode", ["pihal_like", "tft_like"])
def test_pihalnet_fit_integration(
    default_model_params, dummy_data_generator, mode
):
    """
    Tests the end-to-end training process with model.fit() for one epoch
    for both operational modes.
    """
    params = default_model_params.copy()
    params['mode'] = mode
    params['pde_mode'] = 'consolidation' # Ensure PDE is active for loss calc

    # Generate correctly shaped data for the mode
    inputs_dict, targets_dict = dummy_data_generator(params, batch_size=4)
    
    model = PIHALNet(**params)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={
            "subs_pred": MeanSquaredError(),
            "gwl_pred": MeanSquaredError()
        },
        lambda_pde=0.1
    )
    
    # Create a tf.data.Dataset
    dataset = Dataset.from_tensor_slices(
        (inputs_dict, targets_dict)
    ).batch(2)

    # Test the fit method
    try:
        history = model.fit(dataset, epochs=1, verbose=0)
        assert history is not None
        assert "total_loss" in history.history
        assert not np.isnan(history.history["total_loss"][0])
    except Exception as e:
        pytest.fail(
            f"model.fit() failed for mode='{mode}' with error: {e}"
        )

if __name__=="__main__": # pragma: no cover 
   pytest.main([__file__, '-v'])
# -*- coding: utf-8 -*-
# test_revised_tft.py (Place in your tests directory, e.g., fusionlab/nn/tests/)

import pytest
import numpy as np
import tensorflow as tf

# --- Attempt to import revised TFT and dependencies ---
try:
    # Assuming the revised TFT class is accessible via this import path
    from fusionlab.nn.transformers import TFT # Or the specific module path
    from fusionlab.nn.losses import combined_quantile_loss
    from fusionlab.nn import KERAS_BACKEND # Check backend availability
except ImportError as e:
    print(f"Skipping revised TFT tests due to import error: {e}")
    KERAS_BACKEND = False
# --- End Imports ---

# Skip all tests if TF/Keras backend is not available
pytestmark = pytest.mark.skipif(
    not KERAS_BACKEND, reason="TensorFlow/Keras backend not available"
)

# --- Test Fixtures ---

@pytest.fixture(scope="module") # Reuse fixture across tests in this module
def tft_dummy_data():
    """Provides consistent dummy data with static, dynamic, and future."""
    B, T_past, H = 4, 12, 6  # Batch, PastTimeSteps, Horizon
    D_dyn, D_stat, D_fut = 5, 3, 2 # Feature dimensions

    # Note: Future inputs shape needs careful consideration based on how
    # they are combined with past inputs before the LSTM in your TFT impl.
    # Assuming here future covers only the horizon for simplicity in testing call.
    # Adjust if your model expects future inputs spanning T_past + H.
    T_future = H

    X_static = np.random.rand(B, D_stat).astype(np.float32)
    X_dynamic = np.random.rand(B, T_past, D_dyn).astype(np.float32)
    X_future = np.random.rand(B, T_future, D_fut).astype(np.float32)
    # Target shape matches batch and horizon, with 1 output dim
    y = np.random.rand(B, H, 1).astype(np.float32)
    return {
        "B": B, "T_past": T_past, "H": H, "T_future": T_future,
        "D_dyn": D_dyn, "D_stat": D_stat, "D_fut": D_fut,
        "X_static": X_static, "X_dynamic": X_dynamic,
        "X_future": X_future, "y": y
    }

# --- Test Functions ---

def test_tft_revised_instantiation(tft_dummy_data):
    """Test revised TFT instantiation with required args."""
    data = tft_dummy_data
    model = TFT(
        dynamic_input_dim=data["D_dyn"],
        static_input_dim=data["D_stat"], # Required
        future_input_dim=data["D_fut"], # Required
        forecast_horizon=data["H"],
        hidden_units=16, # Non-default
        num_heads=2      # Non-default
    )
    assert isinstance(model, tf.keras.Model)
    assert model.static_input_dim == data["D_stat"]
    assert model.future_input_dim == data["D_fut"]
    assert model.quantiles is None # Default is point forecast
    print("Revised TFT Instantiation OK")

def test_tft_revised_instantiation_quantiles(tft_dummy_data):
    """Test revised TFT instantiation with quantiles."""
    data = tft_dummy_data
    quantiles = [0.1, 0.5, 0.9]
    model = TFT(
        dynamic_input_dim=data["D_dyn"],
        static_input_dim=data["D_stat"],
        future_input_dim=data["D_fut"],
        forecast_horizon=data["H"],
        quantiles=quantiles,
        hidden_units=8,
        num_heads=1
    )
    assert isinstance(model, tf.keras.Model)
    assert np.around(model.quantiles, 1).tolist() == quantiles
    assert model.num_quantiles == len(quantiles)
    print("Revised TFT Instantiation with Quantiles OK")

@pytest.mark.parametrize("use_quantiles", [False, True])
def test_tft_revised_call_and_output_shape(tft_dummy_data, use_quantiles):
    """Test model call with required inputs and check output shape."""
    data = tft_dummy_data
    B, H = data["B"], data["H"]
    quantiles = [0.25, 0.5, 0.75] if use_quantiles else None
    num_outputs = len(quantiles) if quantiles else 1 # Output dim is 1

    model = TFT(
        dynamic_input_dim=data["D_dyn"],
        static_input_dim=data["D_stat"],
        future_input_dim=data["D_fut"],
        forecast_horizon=H,
        quantiles=quantiles,
        hidden_units=8,
        num_heads=1,
        output_dim=1 
    )

    # Prepare inputs in the REQUIRED order: [static, dynamic, future]
    inputs = [
        data["X_static"],
        data["X_dynamic"],
        data["X_future"]
    ]

    # Perform forward pass
    try:
        # Build the model first if needed (call might do this)
        # model.build(input_shape=[i.shape for i in inputs]) # Optional explicit build
        predictions = model(inputs, training=False)
    except Exception as e:
        pytest.fail(f"Model call failed. Error: {e}")

    # Check output shape
    expected_shape = (B, H, num_outputs)
    assert predictions.shape == expected_shape, \
        f"Output shape mismatch. Expected {expected_shape}, got {predictions.shape}"

    # Check if VSN importance attributes exist (optional check)
    assert hasattr(model, 'static_vsn') and hasattr(model.static_vsn, 'variable_importances_')
    assert hasattr(model, 'dynamic_vsn') and hasattr(model.dynamic_vsn, 'variable_importances_')
    assert hasattr(model, 'future_vsn') and hasattr(model.future_vsn, 'variable_importances_')

    print(f"Revised TFT Call OK: quantiles={use_quantiles}, Output Shape={predictions.shape}")

def test_tft_revised_call_wrong_input_number(tft_dummy_data):
    """Test that call raises ValueError if not exactly 3 inputs."""
    data = tft_dummy_data
    model = TFT( # Instantiate with required dims
        dynamic_input_dim=data["D_dyn"], static_input_dim=data["D_stat"],
        future_input_dim=data["D_fut"], forecast_horizon=data["H"]
    )
    # Try calling with only 2 inputs
    bad_inputs = [data["X_static"], data["X_dynamic"]]
    with pytest.raises(ValueError, match=r"expects inputs as a list/tuple of 3 elements"):
        _ = model(bad_inputs, training=False)
    print("Revised TFT Call Input Number Check OK")


@pytest.mark.parametrize("use_quantiles", [False, True])
def test_tft_revised_compile_and_fit(tft_dummy_data, use_quantiles):
    """Test compile and a single training step."""
    data = tft_dummy_data
    quantiles = [0.2, 0.5, 0.8] if use_quantiles else None
    # Loss is handled by the custom compile method based on quantiles
    loss_fn = None

    model = TFT(
        dynamic_input_dim=data["D_dyn"],
        static_input_dim=data["D_stat"],
        future_input_dim=data["D_fut"],
        forecast_horizon=data["H"],
        quantiles=quantiles,
        hidden_units=8,
        num_heads=1,
        output_dim=1
    )

    # Prepare inputs and targets
    inputs = [
        data["X_static"],
        data["X_dynamic"],
        data["X_future"]
    ]
    y = data["y"] # Shape (B, H, 1)

    try:
        # Use the model's compile method
        model.compile(optimizer='adam', loss=loss_fn) # Pass None for loss
        history = model.fit(
            inputs, y, epochs=1, batch_size=2, verbose=0
        )
    except Exception as e:
        pytest.fail(f"Revised TFT compile/fit failed (quantiles={use_quantiles}). Error: {e}")

    assert history is not None
    assert 'loss' in history.history
    print(f"Revised TFT Compile/Fit OK: quantiles={use_quantiles}")

def test_tft_revised_serialization(tft_dummy_data):
    """Test model get_config and from_config for the revised TFT."""
    data = tft_dummy_data
    quantiles = [0.1, 0.5, 0.9]
    original_params = dict(
        dynamic_input_dim=data["D_dyn"],
        static_input_dim=data["D_stat"],
        future_input_dim=data["D_fut"],
        forecast_horizon=data["H"],
        quantiles=quantiles,
        hidden_units=16,
        num_heads=2,
        dropout_rate=0.05,
        recurrent_dropout_rate=0.02,
        activation='gelu',
        use_batch_norm=True,
        num_lstm_layers=2,
        lstm_units=[20, 10], # Example list
        output_dim=1
    )
    model = TFT(**original_params)

    # Prepare inputs to build the model before get_config
    inputs = [data["X_static"], data["X_dynamic"], data["X_future"]]
    _ = model(inputs) # Run forward pass to build

    try:
        config = model.get_config()
        # Ensure all necessary params are in config
        for key in original_params:
             # Special handling for quantiles/lstm_units if needed based on get_config impl.
             if key in ['lstm_units']: # Check original spec if stored differently
                 assert config.get(key) == model._lstm_units, f"Config mismatch for {key}"
             elif key in config:
                 assert config.get(key) == original_params[key], f"Config mismatch for {key}"
             else:
                 # Check if param is implicitly handled by base Model config
                 pass # Or add specific checks if needed

        rebuilt_model = TFT.from_config(config)
    except Exception as e:
        pytest.fail(f"Revised TFT serialization/deserialization failed. Error: {e}")

    assert isinstance(rebuilt_model, TFT)
    # Check a few key config parameters match after rebuild
    assert rebuilt_model.static_input_dim == original_params["static_input_dim"]
    assert rebuilt_model.forecast_horizon == original_params["forecast_horizon"]
    assert rebuilt_model.quantiles == original_params["quantiles"]
    assert rebuilt_model.num_heads == original_params["num_heads"]
    assert rebuilt_model.recurrent_dropout_rate == original_params["recurrent_dropout_rate"]

    # Optional: Check output shape consistency
    try:
        preds_original = model(inputs)
        preds_rebuilt = rebuilt_model(inputs)
        assert preds_original.shape == preds_rebuilt.shape
    except Exception as e:
         pytest.fail(f"Prediction shape mismatch after from_config. Error: {e}")

    print("Revised TFT Serialization (get_config/from_config) OK")
    
if __name__ =='__main__': 
    pytest.main([__file__])
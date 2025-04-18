# test_tft.py (or similar location like fusionlab/nn/tests/test_tft.py)

import pytest
import numpy as np
import tensorflow as tf

# Attempt to import necessary components
try:
    from fusionlab.nn import TemporalFusionTransformer
    # Import loss if testing quantile mode compilation/training
    from fusionlab.nn.losses import combined_quantile_loss
    # Check if TF backend is available (using the flag from your nn package)
    from fusionlab.nn import KERAS_BACKEND
except ImportError as e:
    print(f"Skipping TFT tests due to import error: {e}")
    # Set flag to False if import fails, pytest marker handles skipping
    KERAS_BACKEND = False

# Skip all tests in this file if TensorFlow/Keras backend is not available
pytestmark = pytest.mark.skipif(
    not KERAS_BACKEND, reason="TensorFlow/Keras backend not available"
)

# --- Test Fixtures ---

@pytest.fixture
def dummy_data():
    """Provides consistent dummy data for tests."""
    B, T, H = 4, 10, 3  # Batch, TimeSteps, Horizon
    D_dyn, D_stat, D_fut = 5, 2, 3 # Feature dimensions
    # Note: Future features often span T + H steps in preprocessing
    # but the model might only need the part corresponding to T
    # Adjust X_future shape based on model's internal handling/validation
    X_static = np.random.rand(B, D_stat).astype(np.float32)
    X_dynamic = np.random.rand(B, T, D_dyn).astype(np.float32)
    # Assuming model's internal validation takes T steps for future input context
    X_future = np.random.rand(B, T, D_fut).astype(np.float32)
    # Target shape matches batch and horizon, with 1 output dim
    y = np.random.rand(B, H, 1).astype(np.float32)
    return {
        "B": B, "T": T, "H": H,
        "D_dyn": D_dyn, "D_stat": D_stat, "D_fut": D_fut,
        "X_static": X_static, "X_dynamic": X_dynamic,
        "X_future": X_future, "y": y
    }

# --- Test Functions ---

def test_tft_instantiation(dummy_data):
    """Test basic model instantiation."""
    model = TemporalFusionTransformer(
        dynamic_input_dim=dummy_data["D_dyn"],
        forecast_horizon=dummy_data["H"]
    )
    assert isinstance(model, tf.keras.Model)
    print("Basic Instantiation OK")

    # Test instantiation with all dims
    model_full = TemporalFusionTransformer(
        dynamic_input_dim=dummy_data["D_dyn"],
        static_input_dim=dummy_data["D_stat"],
        future_input_dim=dummy_data["D_fut"],
        forecast_horizon=dummy_data["H"],
        hidden_units=16,
        num_heads=2,
        dropout_rate=0.0,
        quantiles=[0.2, 0.5, 0.8],
        activation='relu',
        use_batch_norm=True,
        num_lstm_layers=2,
        lstm_units=[10, 8]
    )
    assert isinstance(model_full, tf.keras.Model)
    assert model_full.num_quantiles == 3
    print("Full Instantiation OK")

@pytest.mark.parametrize(
    "use_static, use_future, quantiles",
    [
        (False, False, None),         # Dynamic only, point forecast
        (True, False, None),          # Dynamic + Static, point forecast
        (False, True, None),          # Dynamic + Future, point forecast
        (True, True, None),           # All inputs, point forecast
        (True, True, [0.1, 0.5, 0.9]) # All inputs, quantile forecast
    ]
)
def test_tft_call_and_output_shape(dummy_data, use_static, use_future, quantiles):
    """Test model call with different input combinations and check output shape."""
    B, H = dummy_data["B"], dummy_data["H"]
    D_dyn = dummy_data["D_dyn"]
    D_stat = dummy_data["D_stat"] if use_static else None
    D_fut = dummy_data["D_fut"] if use_future else None
    num_quantiles = len(quantiles) if quantiles else 1

    model = TemporalFusionTransformer(
        dynamic_input_dim=D_dyn,
        static_input_dim=D_stat,
        future_input_dim=D_fut,
        forecast_horizon=H,
        quantiles=quantiles,
        hidden_units=8, # Keep small for testing
        num_heads=1
    )

    # Prepare inputs based on flags
    inputs = []
    inputs.append(dummy_data["X_dynamic"])
    if use_future:
        inputs.append(dummy_data["X_future"])
    if use_static:
        inputs.append(dummy_data["X_static"])

    # Perform forward pass
    # Need to build model first or run predict once
    try:
        predictions = model(inputs, training=False)
    except Exception as e:
        pytest.fail(f"Model call failed with static={use_static}, "
                    f"future={use_future}, quantiles={quantiles}. Error: {e}")

    # Check output shape
    expected_shape = (B, H, num_quantiles)
    assert predictions.shape == expected_shape, \
        f"Output shape mismatch. Expected {expected_shape}, got {predictions.shape}"

    # Check interpretability attributes existence
    if use_static:
        assert hasattr(model, 'static_variable_importances_')
        assert model.static_variable_importances_ is not None
    if use_future:
         assert hasattr(model, 'future_variable_importances_')
         assert model.future_variable_importances_ is not None
    assert hasattr(model, 'dynamic_variable_importances_')
    assert model.dynamic_variable_importances_ is not None

    print(f"Call OK: static={use_static}, future={use_future},"
          f" quantiles={quantiles}, Output Shape={predictions.shape}")

@pytest.mark.parametrize("use_quantiles", [False, True])
def test_tft_compile_and_fit(dummy_data, use_quantiles):
    """Test if the model compiles and runs a single training step."""
    quantiles = [0.2, 0.5, 0.8] if use_quantiles else None
    loss_fn = combined_quantile_loss(quantiles) if use_quantiles else 'mse'
    # num_quantiles = len(quantiles) if quantiles else 1

    model = TemporalFusionTransformer(
        dynamic_input_dim=dummy_data["D_dyn"],
        static_input_dim=dummy_data["D_stat"],
        future_input_dim=dummy_data["D_fut"],
        forecast_horizon=dummy_data["H"],
        quantiles=quantiles,
        hidden_units=8,
        num_heads=1
    )

   # --- FIX: Prepare inputs IN THE CORRECT ORDER ---
    # Expected order by validate_tft_inputs for 3 inputs:
    # (Dynamic, Future, Static)
    inputs = [
        dummy_data["X_dynamic"], # Index 0
        dummy_data["X_future"],  # Index 1
        dummy_data["X_static"]   # Index 2
    ]
    # --------------------------------------------

    #y = dummy_data["y"] # Shape (B, H, 1)
    # Target y shape needs to be (B, H, 1) even for point forecast loss like MSE
    # when model output is 3D. Adjust if model output logic changes.
    # For quantile loss, y_true is (B, H, 1), y_pred is (B, H, Q) -> handled by loss fn
    y = dummy_data["y"] # Shape (B, H, 1)

    try:
        model.compile(optimizer='adam', loss=loss_fn)
        history = model.fit(
            inputs, y, epochs=1, batch_size=2, verbose=0
        )
    except Exception as e:
        pytest.fail(f"Model compile/fit failed (quantiles={use_quantiles}). Error: {e}")

    assert history is not None
    assert 'loss' in history.history
    print(f"Compile/Fit OK: quantiles={use_quantiles}")

def test_tft_serialization(dummy_data):
    """Test model get_config and from_config."""
    quantiles=[0.1, 0.5, 0.9]
    model = TemporalFusionTransformer(
        dynamic_input_dim=dummy_data["D_dyn"],
        static_input_dim=dummy_data["D_stat"],
        future_input_dim=dummy_data["D_fut"],
        forecast_horizon=dummy_data["H"],
        quantiles=quantiles,
        hidden_units=16,
        num_heads=2,
        dropout_rate=0.05,
        activation='gelu',
        use_batch_norm=True,
        num_lstm_layers=1,
        lstm_units=20
    )
    # Prepare inputs to allow building the model before get_config
    inputs = [
        dummy_data["X_dynamic"],
        dummy_data["X_future"], 
        dummy_data["X_static"],
    ]
    _ = model(inputs) # Run a forward pass to build

    try:
        config = model.get_config()
        rebuilt_model = TemporalFusionTransformer.from_config(config)
    except Exception as e:
        pytest.fail(f"Model serialization/deserialization failed. Error: {e}")

    assert isinstance(rebuilt_model, TemporalFusionTransformer)
    # Check a few key config parameters match
    assert rebuilt_model.dynamic_input_dim == model.dynamic_input_dim
    assert rebuilt_model.static_input_dim == model.static_input_dim
    assert rebuilt_model.future_input_dim == model.future_input_dim
    assert rebuilt_model.forecast_horizon == model.forecast_horizon
    assert rebuilt_model.quantiles == model.quantiles
    assert rebuilt_model.num_heads == model.num_heads
    assert rebuilt_model.lstm_units == model.lstm_units

    # Optional: Check output shape consistency
    # Need to build the rebuilt model too
    try:
        preds_original = model(inputs)
        preds_rebuilt = rebuilt_model(inputs)
        assert preds_original.shape == preds_rebuilt.shape
    except Exception as e:
         pytest.fail(f"Prediction shape mismatch after from_config. Error: {e}")

    print("Serialization (get_config/from_config) OK")
    
if __name__=='__main__': 
    pytest.main( [__file__])
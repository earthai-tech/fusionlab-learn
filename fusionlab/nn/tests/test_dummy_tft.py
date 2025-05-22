# test_dummy_tft.py

import pytest
import numpy as np

from pathlib import Path
import warnings

# --- Attempt to import DummyTFT and dependencies ---
try:
    import tensorflow as tf
    # Assuming DummyTFT is in fusionlab.nn.transformers
    from fusionlab.nn.transformers import DummyTFT
    from fusionlab.nn.losses import combined_quantile_loss
    # Import validator to understand its expected input for DummyTFT's call
    from fusionlab.nn._tensor_validation import validate_model_inputs
    from fusionlab.nn import KERAS_BACKEND
    from fusionlab.api.bunch import Bunch # If used by model or helpers
    FUSIONLAB_INSTALLED = True
except ImportError as e:
    print(f"Skipping DummyTFT tests due to import error: {e}")
    FUSIONLAB_INSTALLED = False
    class DummyTFT: pass # Dummy for collection
    def combined_quantile_loss(q): return "mse" # Dummy

pytestmark = pytest.mark.skipif(
    not FUSIONLAB_INSTALLED,
    reason="fusionlab.nn.transformers.DummyTFT or deps not found"
)

# --- Test Fixtures ---
# Define base dimensions for dummy data
B, T_past, H_out = 8, 12, 6 # Batch, Past Timesteps, Output Horizon
D_s, D_d, D_o = 2, 3, 1    # Static, Dynamic, Output Dims

@pytest.fixture(scope="module")
def dummy_tft_model_config():
    """
    Provides base configuration for DummyTFT.
    Requires static_input_dim and dynamic_input_dim.
    """
    return {
        "static_input_dim": D_s,
        "dynamic_input_dim": D_d,
        "forecast_horizon": H_out,
        "output_dim": D_o,
        # Minimal other params for faster testing
        "hidden_units": 8,
        "num_heads": 1,
        "num_lstm_layers": 1,
        "lstm_units": 8, # Can be int, list, or None
        "dropout_rate": 0.0,
        "activation": 'relu',
        "use_batch_norm": False,
    }

@pytest.fixture(scope="module")
def dummy_tft_input_data(dummy_tft_model_config):
    """
    Provides dummy data for DummyTFT (static and dynamic only).
    """
    cfg = dummy_tft_model_config
    # X_static: (Batch, NumStaticFeatures)
    X_static = np.random.rand(
        B, cfg["static_input_dim"]).astype(np.float32)
    # X_dynamic: (Batch, PastTimeSteps, NumDynamicFeatures)
    X_dynamic = np.random.rand(
        B, T_past, cfg["dynamic_input_dim"]).astype(np.float32)
    # y_target: (Batch, ForecastHorizon, OutputDim)
    y_target = np.random.rand(
        B, cfg["forecast_horizon"], cfg["output_dim"]
        ).astype(np.float32)

    return {
        "X_static": tf.constant(X_static),
        "X_dynamic": tf.constant(X_dynamic),
        "y_target": tf.constant(y_target),
    }

# --- Test Functions ---

def test_dummy_tft_instantiation(dummy_tft_model_config):
    """Test basic DummyTFT instantiation with required args."""
    config = dummy_tft_model_config.copy()
    try:
        model = DummyTFT(**config)
        assert isinstance(model, tf.keras.Model)
        assert model.static_input_dim == config["static_input_dim"]
        assert model.dynamic_input_dim == config["dynamic_input_dim"]
        assert model.quantiles is None # Default
        assert model.future_input_dim is None # Explicitly set in DummyTFT
        # Check for some key internal layers
        assert hasattr(model, 'static_var_sel')
        assert hasattr(model, 'dynamic_var_sel')
        assert hasattr(model, 'temporal_attention')
        assert hasattr(model, 'output_projection_layer') # For point
    except Exception as e:
        pytest.fail(f"DummyTFT instantiation failed. Error: {e}")
    print("DummyTFT Instantiation (Required Args): OK")

@pytest.mark.skip("Skip the user warnings.")
def test_dummy_tft_instantiation_with_future_dim_warning(
    dummy_tft_model_config
    ):
    """
    Test DummyTFT instantiation when future_input_dim is passed.
    It should accept it but warn and ignore it.
    """
    config = dummy_tft_model_config.copy()
    config["future_input_dim"] = 5 # Pass a dummy future_input_dim

    with pytest.warns(UserWarning) as record:
        model = DummyTFT(**config)
    
    assert isinstance(model, tf.keras.Model)
    assert model.static_input_dim == config["static_input_dim"]
    assert model.dynamic_input_dim == config["dynamic_input_dim"]
    # Crucially, internal future_input_dim should be None
    assert model.future_input_dim is None
    assert model._future_input_dim_config == 5 # Check stored original
    
    # Check that the warning was issued by param_deprecated_message
    assert len(record) >= 1
    assert "future_input_dim' parameter is accepted by DummyTFT" \
           in str(record[0].message)
    print("DummyTFT future_input_dim warning and handling: OK")


@pytest.mark.parametrize("use_quantiles", [False, True])
def test_dummy_tft_call_and_output_shape(
    dummy_tft_model_config, dummy_tft_input_data, use_quantiles
    ):
    """Test DummyTFT call for point and quantile forecasts."""
    config = dummy_tft_model_config.copy()
    data = dummy_tft_input_data
    
    quantiles_list = [0.2, 0.5, 0.8] if use_quantiles else None
    config["quantiles"] = quantiles_list

    model = DummyTFT(**config)

    # DummyTFT expects inputs as [static, dynamic]
    inputs_list = [data["X_static"], data["X_dynamic"]]

    try:
        # Build model by calling it (if not built in __init__)
        predictions = model(inputs_list, training=False)
    except Exception as e:
        pytest.fail(
            f"DummyTFT call failed (Quantiles={use_quantiles}). Error: {e}"
            )

    # Check output shape
    # Point: (B, H, O)
    # Quantile (O=1): (B, H, Q)
    # Quantile (O>1): (B, H, Q, O)
    if use_quantiles:
        if config["output_dim"] == 1:
            expected_shape = (
                B, config["forecast_horizon"], len(quantiles_list)
                )
        else:
            expected_shape = (
                B, config["forecast_horizon"],
                len(quantiles_list), config["output_dim"]
                )
    else: # Point forecast
        expected_shape = (
            B, config["forecast_horizon"], config["output_dim"]
            )

    assert predictions.shape == expected_shape, \
        (f"Output shape mismatch. Expected {expected_shape}, "
         f"got {predictions.shape}")
    print(f"DummyTFT Call OK: Quantiles={use_quantiles}, "
          f"Output Shape={predictions.shape}")


def test_dummy_tft_call_input_errors(
    dummy_tft_model_config, dummy_tft_input_data
    ):
    """Test DummyTFT call raises ValueError for incorrect inputs."""
    model = DummyTFT(**dummy_tft_model_config)
    data = dummy_tft_input_data

    # Only one input
    with pytest.raises(ValueError, match="DummyTFT expects inputs as a list/tuple of 2"):
        model([data["X_static"]], training=False)
    # Too many inputs
    with pytest.raises(ValueError, match="DummyTFT expects inputs as a list/tuple of 2"):
        model([data["X_static"], data["X_dynamic"], data["X_dynamic"]], training=False)

    # Static is None
    with pytest.raises(ValueError, match="Static and Dynamic inputs must be provided"):
        model([None, data["X_dynamic"]], training=False)
    # Dynamic is None
    with pytest.raises(ValueError, match="Static and Dynamic inputs must be provided"):
        model([data["X_static"], None], training=False)
    print("DummyTFT Call Input Errors (count/None): OK")


@pytest.mark.parametrize("use_quantiles", [False, True])
def test_dummy_tft_compile_and_fit(
    dummy_tft_model_config, dummy_tft_input_data, use_quantiles
    ):
    """Test compile and a single training step for DummyTFT."""
    config = dummy_tft_model_config.copy()
    data = dummy_tft_input_data
    quantiles_list = [0.2, 0.5, 0.8] if use_quantiles else None
    config["quantiles"] = quantiles_list

    model = DummyTFT(**config)
    inputs_list = [data["X_static"], data["X_dynamic"]]
    y_target = data["y_target"] # Shape (B, H, O)

    # For quantile loss, y_true (B,H,O) is compared with y_pred (B,H,Q*O) or (B,H,Q,O)
    # The combined_quantile_loss handles broadcasting of y_true.

    loss_fn = combined_quantile_loss(quantiles_list) if use_quantiles else "mse"

    try:
        model.compile(optimizer='adam', loss=loss_fn)
        # Dummy call to build the model before fit, if not already built
        # This helps catch build-time errors earlier.
        _ = model(inputs_list)

        history = model.fit(
            inputs_list, y_target,
            epochs=1, batch_size=4, # Small batch for small data
            verbose=0
        )
    except Exception as e:
        pytest.fail(
            f"DummyTFT compile/fit failed (Quantiles={use_quantiles}). Error: {e}"
            )
    assert history is not None and 'loss' in history.history
    print(f"DummyTFT Compile/Fit OK: Quantiles={use_quantiles}")


def test_dummy_tft_serialization(dummy_tft_model_config, dummy_tft_input_data):
    """Test DummyTFT get_config and from_config."""
    config = dummy_tft_model_config.copy()
    config["quantiles"] = [0.1, 0.9] # Non-default
    config["hidden_units"] = 24    # Non-default
    config["output_dim"] = 2       # Test multi-output
    config["future_input_dim"] = 5 # Test passing this, should be ignored by logic

    model1 = DummyTFT(**config)
    inputs_list = [
        dummy_tft_input_data["X_static"],
        dummy_tft_input_data["X_dynamic"]
        ]
    # Adjust target for output_dim=2
    y_target_multi_dim = np.random.rand(
        B, config["forecast_horizon"], config["output_dim"]
    ).astype(np.float32)

    _ = model1(inputs_list) # Call to build

    try:
        retrieved_config = model1.get_config()
        # Check key parameters are saved
        assert retrieved_config['static_input_dim'] == config['static_input_dim']
        assert retrieved_config['dynamic_input_dim'] == config['dynamic_input_dim']
        assert retrieved_config['hidden_units'] == config['hidden_units']
        assert [ round(a, 1) for a in retrieved_config['quantiles']] == config['quantiles']
        assert retrieved_config['output_dim'] == config['output_dim']
        # Check that future_input_dim from init is stored for API consistency
        assert retrieved_config['future_input_dim'] == None # 

        model2 = DummyTFT.from_config(retrieved_config)
    except Exception as e:
        pytest.fail(f"DummyTFT serialization/deserialization failed. Error: {e}")

    assert isinstance(model2, DummyTFT)
    # Verify internal future_input_dim is None after from_config
    assert model2.future_input_dim is None

    # Check output shape consistency
    try:
        # Prepare inputs for model2 if its input_signature is different
        # (though from_config should restore it correctly)
        output_original = model1(inputs_list, training=False)
        output_rebuilt = model2(inputs_list, training=False)
        assert output_original.shape == output_rebuilt.shape
    except Exception as e:
         pytest.fail(
             f"Output shape mismatch after from_config. Error: {e}"
             )
    print("DummyTFT Serialization OK")

# Allows running the tests directly if needed
if __name__=='__main__':
     pytest.main([__file__])


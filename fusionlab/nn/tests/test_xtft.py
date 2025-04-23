# test_xtft.py

import pytest
import numpy as np

import os

# --- Attempt to import XTFT and dependencies ---
try:
    import tensorflow as tf
    from fusionlab.nn._xtft import XTFT
    # Import necessary components for context/checks if needed
    from fusionlab.nn.components import (
         GatedResidualNetwork, VariableSelectionNetwork # Min dependencies
    )
    from fusionlab.nn.losses import (
        combined_quantile_loss, combined_total_loss,
        prediction_based_loss
    )
    from fusionlab.nn.components import AnomalyLoss # Needed for combined_total_loss test
    # Import validator if needed for specific checks
    from fusionlab.nn._tensor_validation import validate_xtft_inputs
    # Check backend availability
    from fusionlab.nn import KERAS_BACKEND
except ImportError as e:
    print(f"Skipping XTFT tests due to import error: {e}")
    KERAS_BACKEND = False
# --- End Imports ---

# Skip all tests if TF/Keras backend is not available
pytestmark = pytest.mark.skipif(
    not KERAS_BACKEND, reason="TensorFlow/Keras backend not available"
)

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def xtft_base_config():
    """Provides base configuration dimensions for XTFT."""
    return {
        "static_input_dim": 4,
        "dynamic_input_dim": 6,
        "future_input_dim": 3,
        "forecast_horizon": 5,
        "output_dim": 1,
        # Keep other hyperparams small for faster testing
        "embed_dim": 8,
        "hidden_units": 8,
        "attention_units": 8,
        "lstm_units": 8,
        "num_heads": 1,
        "max_window_size": 10, # Should be >= T_past used in data
        "memory_size": 10,
        "dropout_rate": 0.0,
        "activation": 'relu',
        "use_batch_norm": False,
        "use_residuals": False,
    }

@pytest.fixture(scope="module")
def dummy_xtft_data(xtft_base_config):
    """Provides dummy data for XTFT tests."""
    B, H = 4, xtft_base_config["forecast_horizon"]
    # Time steps for dynamic input (lookback)
    T_past = xtft_base_config["max_window_size"] # Match max_window_size
    D_dyn = xtft_base_config["dynamic_input_dim"]
    D_stat = xtft_base_config["static_input_dim"]
    D_fut = xtft_base_config["future_input_dim"]

    # Future input needs to cover the time steps processed by VSN/LSTM etc.
    # In the current XTFT call logic, it seems concatenated with dynamic,
    # so it might need T_past length, similar to TFT revision.
    # Let's assume T_future = T_past for input combination step.
    T_future = T_past

    X_static = np.random.rand(B, D_stat).astype(np.float32)
    X_dynamic = np.random.rand(B, T_past, D_dyn).astype(np.float32)
    X_future = np.random.rand(B, T_future, D_fut).astype(np.float32)
    y = np.random.rand(B, H, xtft_base_config["output_dim"]).astype(np.float32)
    # Dummy anomaly scores (matching target shape for loss functions)
    anomaly_scores = np.random.rand(B, H, xtft_base_config["output_dim"]
                                    ).astype(np.float32) * 0.1


    return {
        "B": B, "T_past": T_past, "H": H, "T_future": T_future,
        "D_dyn": D_dyn, "D_stat": D_stat, "D_fut": D_fut,
        "X_static": tf.constant(X_static),
        "X_dynamic": tf.constant(X_dynamic),
        "X_future": tf.constant(X_future),
        "y": tf.constant(y),
        "anomaly_scores": tf.constant(anomaly_scores) # For from_config test
    }

# --- Test Functions ---

def test_xtft_instantiation_required(xtft_base_config):
    """Test XTFT instantiation with only required args."""
    config = xtft_base_config.copy()
    # Keep only required args: static, dynamic, future dims
    minimal_config = {
        k: config[k] for k in
        ["static_input_dim", "dynamic_input_dim", "future_input_dim"]
    }
    try:
        model = XTFT(**minimal_config) # Should use defaults for others
        assert isinstance(model, tf.keras.Model)
        assert model.quantiles is None # Default is point forecast
    except Exception as e:
        pytest.fail(f"XTFT minimal instantiation failed. Error: {e}")
    print("XTFT Instantiation (Required Args) OK")

@pytest.mark.parametrize("use_quantiles", [False, True])
@pytest.mark.parametrize("use_scales", [False, True])
@pytest.mark.parametrize("anomaly_strat", [None, "feature_based"])
def test_xtft_call_and_output_shape(
    xtft_base_config, dummy_xtft_data, use_quantiles, use_scales, anomaly_strat
):
    """Test XTFT call with various configurations."""
    config = xtft_base_config.copy()
    data = dummy_xtft_data
    B, H, O = data["B"], data["H"], config["output_dim"]

    config["quantiles"] = [0.25, 0.5, 0.75] if use_quantiles else None
    config["scales"] = [5, 10] if use_scales else None # Example scales
    config["anomaly_detection_strategy"] = anomaly_strat
    num_outputs = len(config["quantiles"]) if use_quantiles else O

    try:
        model = XTFT(**config)
    except Exception as e:
         pytest.fail(f"XTFT instantiation failed for test params."
                     f" Error: {e}")

    # Prepare inputs in the standard order: [static, dynamic, future]
    inputs = [data["X_static"], data["X_dynamic"], data["X_future"]]

    # Perform forward pass
    try:
        # Build model first if needed (call usually handles it)
        # model.build(input_shape=[i.shape for i in inputs])
        predictions = model(inputs, training=False)
    except Exception as e:
        pytest.fail(f"XTFT call failed (Q={use_quantiles}, Scales={use_scales},"
                    f" Anomaly={anomaly_strat}). Error: {e}")

    # Check output shape
    # QuantileDistributionModeling outputs (B, H, Q) or (B, H, O)
    expected_shape = (B, H, num_outputs)
    assert predictions.shape == expected_shape, \
        f"Output shape mismatch. Expected {expected_shape}, got {predictions.shape}"

    print(f"XTFT Call OK: Q={use_quantiles}, Scales={use_scales},"
          f" Anomaly={anomaly_strat}, Output Shape={predictions.shape}")

def test_xtft_call_wrong_input_number(xtft_base_config, dummy_xtft_data):
    """Test call raises ValueError if not exactly 3 inputs."""
    model = XTFT(**xtft_base_config)
    data = dummy_xtft_data
    bad_inputs = [data["X_static"], data["X_dynamic"]] # Only 2 inputs
    # Check if internal validate_xtft_inputs raises the error
    with pytest.raises(ValueError): # Might be specific error message
        _ = model(bad_inputs, training=False)
    print("XTFT Call Input Number Check OK")

# --- Training Tests ---
# Separate tests for different compile/loss scenarios

def test_xtft_compile_fit_point(xtft_base_config, dummy_xtft_data):
    """Test compile/fit for point forecast (MSE)."""
    config = xtft_base_config.copy()
    config["quantiles"] = None
    model = XTFT(**config)
    data = dummy_xtft_data
    inputs = [data["X_static"], data["X_dynamic"], data["X_future"]]
    y = data["y"]

    try:
        model.compile(optimizer='adam') # Should default loss to mse
        history = model.fit(inputs, y, epochs=1, batch_size=2, verbose=0)
    except Exception as e:
        pytest.fail(f"XTFT compile/fit failed (Point forecast). Error: {e}")
    assert history is not None and 'loss' in history.history
    print("XTFT Compile/Fit OK: Point Forecast")

def test_xtft_compile_fit_quantile(xtft_base_config, dummy_xtft_data):
    """Test compile/fit for quantile forecast."""
    config = xtft_base_config.copy()
    quantiles = [0.1, 0.5, 0.9]
    config["quantiles"] = quantiles
    model = XTFT(**config)
    data = dummy_xtft_data
    inputs = [data["X_static"], data["X_dynamic"], data["X_future"]]
    y = data["y"]

    try:
        model.compile(optimizer='adam') # Should default loss to quantile
        assert isinstance(model.loss, combined_quantile_loss(quantiles).__class__)
        history = model.fit(inputs, y, epochs=1, batch_size=2, verbose=0)
    except Exception as e:
        pytest.fail(f"XTFT compile/fit failed (Quantile forecast). Error: {e}")
    assert history is not None and 'loss' in history.history
    print("XTFT Compile/Fit OK: Quantile Forecast")

def test_xtft_compile_fit_anomaly_from_config(xtft_base_config, dummy_xtft_data):
    """Test compile/fit for anomaly strategy 'from_config'."""
    config = xtft_base_config.copy()
    quantiles = [0.1, 0.5, 0.9]
    config["quantiles"] = quantiles
    config["anomaly_detection_strategy"] = "from_config"
    config["anomaly_loss_weight"] = 0.1
    # Provide dummy scores via anomaly_config
    dummy_scores = dummy_xtft_data["anomaly_scores"]
    config["anomaly_config"] = {"anomaly_scores": dummy_scores}

    model = XTFT(**config)
    data = dummy_xtft_data
    inputs = [data["X_static"], data["X_dynamic"], data["X_future"]]
    # Target y shape needs to be compatible with loss function (e.g., (B, H, 1))
    y = data["y"]

    try:
        # Compile should use combined_total_loss implicitly or explicitly
        # Explicit check might require importing combined_total_loss factory
        loss_fn = combined_total_loss(
             quantiles=quantiles,
             anomaly_layer=AnomalyLoss(weight=config["anomaly_loss_weight"]),
             anomaly_scores=tf.constant(dummy_scores[:2], dtype=tf.float32) # Pass subset matching batch size=2
             )
        # Note: Passing scores to loss factory is tricky for variable batches.
        # The model's compile/train_step needs to handle score alignment.
        # For testing, we rely on the model's internal compile logic.
        model.compile(optimizer='adam') # Rely on internal compile logic
        # If model.compile doesn't auto-detect combined_total_loss:
        # model.compile(optimizer='adam', loss=loss_fn) # Pass explicitly if needed

        # Fit requires only x and y, loss handles anomaly scores
        inputs_2 = [ ip[:2] for ip in inputs ]
        history = model.fit(
            inputs_2, y[:2], epochs=1, batch_size=2, verbose=0
            ) # Use subset for matching scores if loss needs it

    except Exception as e:
        pytest.fail(f"XTFT compile/fit failed (Anomaly='from_config'). Error: {e}")
    assert history is not None and 'loss' in history.history
    print("XTFT Compile/Fit OK: Anomaly='from_config'")

def test_xtft_compile_fit_anomaly_prediction_based(xtft_base_config, dummy_xtft_data):
    """Test compile/fit for anomaly strategy 'prediction_based'."""
    config = xtft_base_config.copy()
    quantiles = [0.1, 0.5, 0.9]
    config["quantiles"] = quantiles
    config["anomaly_detection_strategy"] = "prediction_based"
    config["anomaly_loss_weight"] = 0.1

    model = XTFT(**config)
    data = dummy_xtft_data
    inputs = [data["X_static"], data["X_dynamic"], data["X_future"]]
    y = data["y"]

    try:
        # Compile should use prediction_based_loss implicitly
        model.compile(optimizer='adam') # Rely on internal compile logic
        # Check if the correct loss function was set (optional)
        assert isinstance(model.loss, prediction_based_loss(
            quantiles=quantiles, anomaly_loss_weight=config["anomaly_loss_weight"]
            ).__class__)

        # Fit uses custom train_step defined in XTFT
        history = model.fit(inputs, y, epochs=1, batch_size=2, verbose=0)
    except Exception as e:
        pytest.fail(f"XTFT compile/fit failed (Anomaly='prediction_based'). Error: {e}")
    assert history is not None and 'loss' in history.history
    print("XTFT Compile/Fit OK: Anomaly='prediction_based'")


def test_xtft_serialization(xtft_base_config, dummy_xtft_data):
    """Test XTFT get_config and from_config."""
    config = xtft_base_config.copy()
    config["quantiles"] = [0.1, 0.9] # Non-default
    config["embed_dim"] = 12 # Non-default
    config["scales"] = [5, 10] # Non-default

    model = XTFT(**config)
    # Build model
    inputs = [dummy_xtft_data["X_static"], dummy_xtft_data["X_dynamic"],
              dummy_xtft_data["X_future"]]
    _ = model(inputs)

    try:
        retrieved_config = model.get_config()
        # Check key params
        assert retrieved_config['static_input_dim'] == config['static_input_dim']
        assert retrieved_config['embed_dim'] == config['embed_dim']
        assert retrieved_config['quantiles'] == config['quantiles']
        assert retrieved_config['scales'] == config['scales']

        rebuilt_model = XTFT.from_config(retrieved_config)
    except Exception as e:
        pytest.fail(f"XTFT serialization/deserialization failed. Error: {e}")

    assert isinstance(rebuilt_model, XTFT)
    # Check output shape consistency
    try:
        output_original = model(inputs)
        output_rebuilt = rebuilt_model(inputs)
        assert output_original.shape == output_rebuilt.shape
    except Exception as e:
         pytest.fail(f"Output shape mismatch after from_config. Error: {e}")

    print("XTFT Serialization OK")

# Allows running the tests directly if needed
if __name__=='__main__':
     pytest.main([__file__])
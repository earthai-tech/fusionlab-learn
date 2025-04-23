
import pytest
import numpy as np

import os

try:
    import tensorflow as tf
    # Import SuperXTFT and its base class XTFT
    from fusionlab.nn._xtft import SuperXTFT, XTFT
    # Import necessary components used internally if needed for checks
    from fusionlab.nn.components import (
        VariableSelectionNetwork, GatedResidualNetwork # Min dependencies
    )
    from fusionlab.nn.losses import (
        combined_quantile_loss, combined_total_loss,
        prediction_based_loss
    )
    from fusionlab.nn.components import AnomalyLoss
    # Import validator if needed for specific checks
    from fusionlab.nn._tensor_validation import validate_xtft_inputs
    # Check backend availability
    from fusionlab.nn import KERAS_BACKEND
except ImportError as e:
    print(f"Skipping SuperXTFT tests due to import error: {e}")
    KERAS_BACKEND = False
# --- End Imports ---

# Skip all tests if TF/Keras backend is not available
pytestmark = pytest.mark.skipif(
    not KERAS_BACKEND, reason="TensorFlow/Keras backend not available"
)

# --- Test Fixtures (Reuse from XTFT tests) ---

@pytest.fixture(scope="module")
def xtft_base_config():
    """Provides base configuration dimensions (reused for SuperXTFT)."""
    return {
        "static_input_dim": 4,
        "dynamic_input_dim": 6,
        "future_input_dim": 3,
        "forecast_horizon": 5,
        "output_dim": 1,
        "embed_dim": 8, "hidden_units": 8, "attention_units": 8,
        "lstm_units": 8, "num_heads": 1, "max_window_size": 10,
        "memory_size": 10, "dropout_rate": 0.0, "activation": 'relu',
        "use_batch_norm": False, "use_residuals": False,
        "final_agg": 'last', # Added from SuperXTFT init defaults
        "multi_scale_agg": 'auto', # Added from SuperXTFT init defaults
    }

@pytest.fixture(scope="module")
def dummy_xtft_data(xtft_base_config):
    """Provides dummy data (reused for SuperXTFT tests)."""
    B, H = 4, xtft_base_config["forecast_horizon"]
    T_past = xtft_base_config["max_window_size"]
    D_dyn = xtft_base_config["dynamic_input_dim"]
    D_stat = xtft_base_config["static_input_dim"]
    D_fut = xtft_base_config["future_input_dim"]
    T_future = T_past # Assume future inputs match past length

    X_static = np.random.rand(B, D_stat).astype(np.float32)
    X_dynamic = np.random.rand(B, T_past, D_dyn).astype(np.float32)
    X_future = np.random.rand(B, T_future, D_fut).astype(np.float32)
    y = np.random.rand(B, H, xtft_base_config["output_dim"]).astype(np.float32)
    anomaly_scores = np.random.rand(
        B, H, xtft_base_config["output_dim"]).astype(np.float32) * 0.1

    return {
        "B": B, "T_past": T_past, "H": H, "T_future": T_future,
        "D_dyn": D_dyn, "D_stat": D_stat, "D_fut": D_fut,
        "X_static": tf.constant(X_static),
        "X_dynamic": tf.constant(X_dynamic),
        "X_future": tf.constant(X_future),
        "y": tf.constant(y),
        "anomaly_scores": tf.constant(anomaly_scores)
    }

# --- Test Functions for SuperXTFT ---

def test_superxtft_instantiation(xtft_base_config):
    """Test SuperXTFT instantiation."""
    config = xtft_base_config.copy()
    try:
        model = SuperXTFT(**config)
        assert isinstance(model, SuperXTFT)
        assert isinstance(model, XTFT) # Check inheritance
        # Check if specific SuperXTFT layers were added
        assert hasattr(model, 'variable_selection_static')
        assert isinstance(model.variable_selection_static, VariableSelectionNetwork)
        assert hasattr(model, 'variable_selection_dynamic')
        assert isinstance(model.variable_selection_dynamic, VariableSelectionNetwork)
        assert hasattr(model, 'variable_future_covariate')
        assert isinstance(model.variable_future_covariate, VariableSelectionNetwork)
        assert hasattr(model, 'grn_attention_hierarchical')
        assert isinstance(model.grn_attention_hierarchical, GatedResidualNetwork)
        assert hasattr(model, 'grn_decoder')
        assert isinstance(model.grn_decoder, GatedResidualNetwork)
    except Exception as e:
        pytest.fail(f"SuperXTFT instantiation failed. Error: {e}")
    print("SuperXTFT Instantiation OK")

@pytest.mark.parametrize("use_quantiles", [False, True])
@pytest.mark.parametrize("use_scales", [False, True])
@pytest.mark.parametrize("anomaly_strat", [None, "feature_based"])
def test_superxtft_call_and_output_shape(
    xtft_base_config, dummy_xtft_data, use_quantiles, use_scales, anomaly_strat
):
    """Test SuperXTFT call with various configurations."""
    config = xtft_base_config.copy()
    data = dummy_xtft_data
    B, H, O = data["B"], data["H"], config["output_dim"]

    config["quantiles"] = [0.25, 0.5, 0.75] if use_quantiles else None
    config["scales"] = [5, 10] if use_scales else None
    config["anomaly_detection_strategy"] = anomaly_strat
    # Determine expected output features based on quantiles/output_dim
    num_outputs = len(config["quantiles"]) if use_quantiles and O == 1 else O

    try:
        model = SuperXTFT(**config)
    except Exception as e:
         pytest.fail(f"SuperXTFT instantiation failed for test params."
                     f" Error: {e}")

    # Prepare inputs in the standard order: [static, dynamic, future]
    inputs = [data["X_static"], data["X_dynamic"], data["X_future"]]

    # Perform forward pass using SuperXTFT's call method
    try:
        predictions = model(inputs, training=False)
    except Exception as e:
        pytest.fail(
            f"SuperXTFT call failed (Q={use_quantiles}, Scales={use_scales},"
            f" Anomaly={anomaly_strat}). Error: {e}"
            )

    # Check output shape (should match XTFT's expected output)
    # Assuming final squeeze logic exists for univariate quantile case
    expected_shape = (B, H, num_outputs)
    assert predictions.shape == expected_shape, \
        f"Output shape mismatch. Expected {expected_shape}, got {predictions.shape}"

    print(f"SuperXTFT Call OK: Q={use_quantiles}, Scales={use_scales},"
          f" Anomaly={anomaly_strat}, Output Shape={predictions.shape}")

def test_superxtft_call_wrong_input_number(xtft_base_config, dummy_xtft_data):
    """Test SuperXTFT call raises ValueError if not exactly 3 inputs."""
    model = SuperXTFT(**xtft_base_config)
    data = dummy_xtft_data
    bad_inputs = [data["X_static"], data["X_dynamic"]] # Only 2 inputs
    # Check if internal validate_xtft_inputs raises the error
    with pytest.raises(ValueError):
        _ = model(bad_inputs, training=False)
    print("SuperXTFT Call Input Number Check OK")

# --- Training Tests (Uses inherited compile/train_step from XTFT) ---

@pytest.mark.parametrize("use_quantiles", [False, True])
def test_superxtft_compile_and_fit(
    xtft_base_config, dummy_xtft_data, use_quantiles
):
    """Test compile and a single training step for SuperXTFT."""
    config = xtft_base_config.copy()
    quantiles = [0.2, 0.5, 0.8] if use_quantiles else None
    config["quantiles"] = quantiles
    loss_fn = None # Use model's default compile logic from XTFT

    model = SuperXTFT(**config)
    data = dummy_xtft_data
    inputs = [data["X_static"], data["X_dynamic"], data["X_future"]]
    y = data["y"]

    try:
        # Use the inherited compile method
        model.compile(optimizer='adam', loss=loss_fn)
        # Fit for one step - tests if overridden call works with training
        history = model.fit(
            inputs, y, epochs=1, batch_size=2, verbose=0
            )
    except Exception as e:
        pytest.fail(
            f"SuperXTFT compile/fit failed (quantiles={use_quantiles})."
            f" Error: {e}"
            )

    assert history is not None
    assert 'loss' in history.history
    print(f"SuperXTFT Compile/Fit OK: quantiles={use_quantiles}")

# Add anomaly strategy fit tests if needed, similar to XTFT tests

def test_superxtft_serialization(xtft_base_config, dummy_xtft_data):
    """Test SuperXTFT get_config and from_config."""
    config = xtft_base_config.copy()
    config["quantiles"] = [0.1, 0.9]
    config["memory_size"] = 75 # Non-default

    model = SuperXTFT(**config)
    inputs = [dummy_xtft_data["X_static"], dummy_xtft_data["X_dynamic"],
              dummy_xtft_data["X_future"]]
    _ = model(inputs) # Build model

    try:
        # get_config should be inherited from XTFT
        retrieved_config = model.get_config()
        # Check XTFT params are saved
        assert retrieved_config['static_input_dim'] == config['static_input_dim']
        assert retrieved_config['quantiles'] == config['quantiles']
        assert retrieved_config['memory_size'] == config['memory_size']

        # Rebuild using the SuperXTFT class method
        rebuilt_model = SuperXTFT.from_config(retrieved_config)
    except Exception as e:
        pytest.fail(f"SuperXTFT serialization/deserialization failed. Error: {e}")

    assert isinstance(rebuilt_model, SuperXTFT) # Check correct type
    # Check output shape consistency
    try:
        output_original = model(inputs)
        output_rebuilt = rebuilt_model(inputs)
        assert output_original.shape == output_rebuilt.shape
    except Exception as e:
         pytest.fail(f"Output shape mismatch after from_config. Error: {e}")

    print("SuperXTFT Serialization OK")

# Allows running the tests directly if needed
if __name__=='__main__':
     pytest.main([__file__])
# test_forecast_tuner.py

import pytest
import numpy as np
import tensorflow as tf
import os
import shutil # For cleaning up tuner directories
from pathlib import Path

# --- Attempt to import tuner functions and dependencies ---
try:
    from fusionlab.nn.forecast_tuner import xtft_tuner, tft_tuner
    from fusionlab.nn.transformers import (
        XTFT, SuperXTFT, TemporalFusionTransformer
    )
    from fusionlab.nn.losses import combined_quantile_loss
    from fusionlab.core.io import _get_valid_kwargs 
    # Import Keras Tuner to check tuner object type
    import keras_tuner as kt
    # Check if TF backend is available
    from fusionlab.nn import KERAS_BACKEND
    FUSIONLAB_INSTALLED = True
    HAS_KT = True # Keras Tuner imported successfully
except ImportError as e:
    print(f"Skipping forecast_tuner tests due to import error: {e}")
    FUSIONLAB_INSTALLED = False
    HAS_KT = False
    # Dummy classes for pytest collection if imports fail
    class XTFT: pass
    class SuperXTFT: pass
    class TemporalFusionTransformer: pass
    def xtft_tuner(*args, **kwargs): raise ImportError("xtft_tuner not found")
    def tft_tuner(*args, **kwargs): raise ImportError("tft_tuner not found")
    class kt: # Dummy kt module
        class Tuner: pass
# --- End Imports ---

# Skip all tests if Keras Tuner or fusionlab components are not available
pytestmark = pytest.mark.skipif(
    not (FUSIONLAB_INSTALLED and HAS_KT),
    reason="Keras Tuner or fusionlab components not found"
)

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def tuner_shared_config():
    """Provides base configuration dimensions shared by tuner tests."""
    return {
        "static_input_dim": 2,
        "dynamic_input_dim": 3,
        "future_input_dim": 2,
        "forecast_horizon": 3, # H
        "output_dim": 1,
        # For model builder inside tuner
        "embed_dim": 8,
        "hidden_units": 8,
        "attention_units": 8,
        "lstm_units": 8,
        "num_heads": 1,
        "max_window_size": 6, # T_past
        "memory_size": 10,
    }

@pytest.fixture(scope="module")
def dummy_tuner_data(tuner_shared_config):
    """Provides dummy data for tuner function tests."""
    cfg = tuner_shared_config
    B, H = 8, cfg["forecast_horizon"] # Small batch for faster fit
    T_past = cfg["max_window_size"]   # Lookback period
    D_stat = cfg["static_input_dim"]
    D_dyn = cfg["dynamic_input_dim"]
    D_fut = cfg["future_input_dim"]

    # Future input should span T_past + H for full utility in models
    T_future_total = T_past + H

    X_static = np.random.rand(B, D_stat).astype(np.float32)
    X_dynamic = np.random.rand(B, T_past, D_dyn).astype(np.float32)
    X_future = np.random.rand(B, T_future_total, D_fut).astype(np.float32)
    y = np.random.rand(B, H, cfg["output_dim"]).astype(np.float32)

    return {
        "X_static": tf.constant(X_static),
        "X_dynamic": tf.constant(X_dynamic),
        "X_future": tf.constant(X_future), # Full future span
        "y": tf.constant(y),
        "forecast_horizon": H, # Pass H for case_info
        # Pass input dims for case_info if model builder needs them
        "static_input_dim": D_stat,
        "dynamic_input_dim": D_dyn,
        "future_input_dim": D_fut,
        "output_dim": cfg["output_dim"]
    }

@pytest.fixture
def temp_tuner_dir(tmp_path_factory):
    """Create a temporary directory for tuner results."""
    # tmp_path_factory is a pytest fixture
    return tmp_path_factory.mktemp("tuner_run_")

# --- Helper Function ---
def _run_tuner_and_asserts(
    tuner_func, model_class, config, data, temp_dir,
    use_quantiles=False, model_name_arg="xtft", tuner_type_arg='random'
    ):
    """Helper to run tuner and perform common assertions."""
    project_name = f"{model_name_arg}_{tuner_type_arg}_test"
    if use_quantiles:
        project_name += "_quantile"
        quantiles = [0.2, 0.5, 0.8]
    else:
        project_name += "_point"
        quantiles = None

    # Minimal search space for speed
    param_space = {
        'hidden_units': [config['hidden_units']], # Fix to one value
        'num_heads': [config['num_heads']],       # Fix to one value
        'learning_rate': [1e-3]                   # Fix to one value
    }
    # Case info for the model builder
    case_info = {
        'quantiles': quantiles,
        'forecast_horizons': data['forecast_horizon'], # Correct key
        'static_input_dim': data['static_input_dim'],
        'dynamic_input_dim': data['dynamic_input_dim'],
        'future_input_dim': data['future_input_dim'],
        'output_dim': data['output_dim'],
        'verbose_build': 0 # Suppress model builder logs
    }

    inputs_list = [data["X_static"], data["X_dynamic"], data["X_future"]]

    try:
        best_hps, best_model, tuner_obj = tuner_func(
            inputs=inputs_list,
            y=data["y"],
            param_space=param_space,
            forecast_horizon=data['forecast_horizon'], # For tuner func itself
            quantiles=quantiles,                      # For tuner func itself
            case_info=case_info,                      # For model builder
            max_trials=1,       # Minimal trials
            objective='val_loss',
            epochs=1,           # Minimal epochs for final train
            batch_sizes=[4],    # Single small batch size
            validation_split=0.5, # Use 50% of small data for val
            tuner_dir=str(temp_dir),
            project_name=project_name,
            tuner_type=tuner_type_arg,
            model_name=model_name_arg, # Critical for _model_builder_factory
            verbose=0# Suppress tuner logs during test run
        )
    except Exception as e:
        pytest.fail(
            f"{tuner_func.__name__} failed for {project_name}. Error: {e}"
            )

    assert isinstance(best_hps, dict)
    assert best_model is not None, "Tuner did not return a best model."
    assert isinstance(best_model, model_class)
    assert isinstance(tuner_obj, kt.Tuner) # Check it's a Keras Tuner object

    # Check if model can predict
    try:
        _ = best_model.predict(inputs_list, verbose=0)
    except Exception as e:
        pytest.fail(f"Best model from {tuner_func.__name__} failed to predict."
                    f" Error: {e}")

    # Check if tuner directory was created
    assert (Path(temp_dir) / project_name).exists()
    #Clean up (optional, tmp_path_factory usually handles it)
    shutil.rmtree(temp_dir / project_name, ignore_errors=True)


# --- Tests for xtft_tuner ---

# @pytest.mark.parametrize("use_quantiles", [False, True])
# @pytest.mark.parametrize("tuner_type", ["random", "bayesian"])
# def test_xtft_tuner_runs(
#     tuner_shared_config, dummy_tuner_data, temp_tuner_dir,
#     use_quantiles, tuner_type
# ):
#     """Test xtft_tuner runs for point and quantile forecasts."""
#     print(f"\nTesting xtft_tuner: Quantiles={use_quantiles}, Tuner={tuner_type}")
#     _run_tuner_and_asserts(
#         xtft_tuner, XTFT, tuner_shared_config, dummy_tuner_data,
#         temp_tuner_dir, use_quantiles=use_quantiles,
#         model_name_arg="xtft", tuner_type_arg=tuner_type
#     )

# def test_xtft_tuner_for_superxtft(
#     tuner_shared_config, dummy_tuner_data, temp_tuner_dir
# ):
#     """Test xtft_tuner can build and tune SuperXTFT."""
#     print("\nTesting xtft_tuner for SuperXTFT")
#     _run_tuner_and_asserts(
#         xtft_tuner, SuperXTFT, tuner_shared_config, dummy_tuner_data,
#         temp_tuner_dir, use_quantiles=False, # Point forecast for simplicity
#         model_name_arg="superxtft", tuner_type_arg='random'
#     )

# def test_xtft_tuner_invalid_inputs(dummy_tuner_data, temp_tuner_dir):
#     """Test xtft_tuner error handling for invalid inputs."""
#     data = dummy_tuner_data
#     # Missing y
#     with pytest.raises(ValueError): # y is now required
#         xtft_tuner(inputs=[data["X_static"], data["X_dynamic"], data["X_future"]],
#                     y=None, forecast_horizon=data["forecast_horizon"])
#     # Inputs not a list of 3
#     with pytest.raises(ValueError, match="Expected inputs to contain exactly 3 elements"):
#         xtft_tuner(inputs=[data["X_static"], data["X_dynamic"]],
#                     y=data["y"], forecast_horizon=data["forecast_horizon"])
#     print("xtft_tuner invalid inputs: OK")


# # --- Tests for tft_tuner ---

@pytest.mark.parametrize("use_quantiles", [False, True])
@pytest.mark.parametrize("tuner_type", ["random", "bayesian"])
def test_tft_tuner_runs(
    tuner_shared_config, dummy_tuner_data, temp_tuner_dir,
    use_quantiles, tuner_type
):
    """Test tft_tuner runs for point and quantile forecasts."""
    print(f"\nTesting tft_tuner: Quantiles={use_quantiles}, Tuner={tuner_type}")
    # tft_tuner calls xtft_tuner with model_name="tft"
    _run_tuner_and_asserts(
        tft_tuner, TemporalFusionTransformer, tuner_shared_config,
        dummy_tuner_data, temp_tuner_dir, use_quantiles=use_quantiles,
        model_name_arg="tft", tuner_type_arg=tuner_type
    )

# Allows running the tests directly if needed
if __name__=='__main__':
     pytest.main([__file__])

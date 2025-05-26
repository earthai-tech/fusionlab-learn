# test_forecast_tuner.py

import pytest
import numpy as np

import os
import shutil # For cleaning up tuner directories
from pathlib import Path

# --- Attempt to import tuner functions and dependencies ---
try:
    import tensorflow as tf
    from fusionlab.nn.forecast_tuner import xtft_tuner, tft_tuner
    from fusionlab.nn.utils import prepare_model_inputs 
    from fusionlab.nn.transformers import (
        XTFT, SuperXTFT,
        TemporalFusionTransformer as TFTFlexible, # Alias for clarity
        TFT as TFTStricter # Alias for the stricter TFT
    )
    from fusionlab.nn.losses import combined_quantile_loss
    # from fusionlab.core.io import _get_valid_kwargs # Not used in this test file
    import keras_tuner as kt
    from fusionlab.nn import KERAS_BACKEND
    FUSIONLAB_INSTALLED = True
    HAS_KT = True
except ImportError as e:
    print(f"Skipping forecast_tuner tests due to import error: {e}")
    FUSIONLAB_INSTALLED = False
    HAS_KT = False
    class XTFT: pass
    class SuperXTFT: pass
    class TFTFlexible: pass # Use alias
    class TFTStricter: pass # Use alias
    def xtft_tuner(*args, **kwargs): raise ImportError("xtft_tuner not found")
    def tft_tuner(*args, **kwargs): raise ImportError("tft_tuner not found")
    class kt:
        class Tuner: pass
    
# --- End Imports ---
# XXX TO OPTIMIZE later : SKIP for Now; 
# HAS_KT =False 

pytestmark = pytest.mark.skipif(
    not (FUSIONLAB_INSTALLED and HAS_KT),
    reason="Keras Tuner or fusionlab components not found"
)

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def tuner_shared_config():
    """Base config dimensions for tuner tests (model HPs)."""
    return {
        # These are for the _model_builder_factory to pick from param_space
        # if not overridden. Actual model will get dims from data.
        "embed_dim": 8,
        "hidden_units": 8,
        "attention_units": 8,
        "lstm_units": 8,
        "num_heads": 1,
        "num_lstm_layers": 1, # For TFTs
        "max_window_size": 6,
        "memory_size": 10,
        "dropout_rate": 0.0,
        "recurrent_dropout_rate": 0.0,
        "activation": 'relu',
        "use_batch_norm": False,
        "use_residuals": False, # For XTFT
        "final_agg": "last",    # For XTFT
        "multi_scale_agg": "last", # For XTFT
        "scales": None, # For XTFT
    }

@pytest.fixture(scope="module")
def dummy_tuner_data(tuner_shared_config):
    """Dummy data for tuner tests. Shapes match typical inputs."""
    cfg = tuner_shared_config
    B, H_out = 8, 3 # Batch, Model Output Horizon
    T_past = 6      # Lookback period (e.g., model's max_window_size)
    D_stat, D_dyn, D_fut = 2, 3, 2 # Feature dimensions

    # Future input should span T_past + H_out for full utility
    T_future_total = T_past + H_out

    X_static = np.random.rand(B, D_stat).astype(np.float32)
    X_dynamic = np.random.rand(B, T_past, D_dyn).astype(np.float32)
    X_future = np.random.rand(B, T_future_total, D_fut).astype(np.float32)
    y = np.random.rand(B, H_out, 1).astype(np.float32) # OutputDim=1

    return {
        "X_static": tf.constant(X_static),
        "X_dynamic": tf.constant(X_dynamic),
        "X_future": tf.constant(X_future),
        "y": tf.constant(y),
        "forecast_horizon": H_out, # For tuner function and case_info
        "static_input_dim": D_stat, # For case_info
        "dynamic_input_dim": D_dyn, # For case_info
        "future_input_dim": D_fut,  # For case_info
        "output_dim": 1             # For case_info
    }

@pytest.fixture
def temp_tuner_dir(tmp_path_factory):
    """Create a temporary directory for tuner results."""
    return tmp_path_factory.mktemp("tuner_run_")

# --- Helper Function ---
def _run_tuner_and_asserts(
    tuner_func, model_class_expected, # Pass expected class type
    base_model_config, # From tuner_shared_config
    data_dict, # From dummy_tuner_data
    temp_dir_path,
    use_quantiles=False,
    model_name_arg="xtft", # This is passed to tuner func
    tuner_type_arg='random'
    ):
    """Helper to run tuner and perform common assertions."""
    project_name = f"{model_name_arg}_{tuner_type_arg}_test"
    quantiles = [0.2, 0.5, 0.8] if use_quantiles else None
    if use_quantiles: project_name += "_quantile"
    else: project_name += "_point"

    # Minimal search space for speed, using fixed values from base_model_config
    param_space = {
        'hidden_units': [base_model_config['hidden_units']],
        'num_heads': [base_model_config['num_heads']],
        'learning_rate': [1e-3]
    }
    # Add model-specific HPs to search space if they are in base_model_config
    if model_name_arg in ["xtft", "superxtft", "super_xtft"]:
        param_space['embed_dim'] = [base_model_config['embed_dim']]
        param_space['lstm_units'] = [base_model_config['lstm_units']]
    elif model_name_arg in ["tft", "tft_flex"]:
        param_space['num_lstm_layers'] = [base_model_config['num_lstm_layers']]
        param_space['lstm_units'] = [base_model_config['lstm_units']]


    # Case info for the model builder (fixed params for this run)
    # Must include all necessary args for the specific model_name
    case_info = {
        'quantiles': quantiles,
        'forecast_horizon': data_dict['forecast_horizon'],
        'static_input_dim': data_dict['static_input_dim'],
        'dynamic_input_dim': data_dict['dynamic_input_dim'],
        'future_input_dim': data_dict['future_input_dim'],
        'output_dim': data_dict['output_dim'],
        'verbose_build': 0 # Suppress model builder logs
    }
    # Add other fixed HPs from base_model_config to case_info
    # so they are passed to the model builder if not in param_space
    for k, v in base_model_config.items():
        if k not in param_space and k not in case_info: # Avoid overriding
            case_info[k] = v


    inputs_list = [data_dict["X_static"], data_dict["X_dynamic"], data_dict["X_future"]]
    # For tft_flex, test with None for static/future if model_builder handles it
    if model_name_arg == "tft_flex" and not use_quantiles: # Example: point forecast no static/future
        inputs_list_flex = [None, data_dict["X_dynamic"], None]
        case_info_flex = case_info.copy()
        # Using full inputs for now, builder handles flexibility.
        # we can use the prepare_model_inputs for consistent preparations: 
            # transforming in strict mode, the None into a zeros tensors
        case_info_flex['static_input_dim'] = None
        case_info_flex['future_input_dim'] = None
        
        inputs_list_flex = prepare_model_inputs(
            static_input= inputs_list_flex[0], 
            dynamic_input= inputs_list_flex[1] ,
            future_input =inputs_list_flex[-1],
            model_type = 'strict',
          )
        
        
        current_inputs = inputs_list_flex
        current_case_info = case_info_flex
    else:
        current_inputs = inputs_list
        current_case_info = case_info


    try:
        best_hps, best_model, tuner_obj = tuner_func(
            inputs=current_inputs,
            y=data_dict["y"],
            param_space=param_space,
            forecast_horizon=data_dict['forecast_horizon'],
            quantiles=quantiles,
            case_info=current_case_info, # Pass the prepared case_info
            max_trials=1,
            objective='val_loss',
            epochs=1,
            batch_sizes=[4],
            validation_split=0.5,
            tuner_dir=str(temp_dir_path),
            project_name=project_name,
            tuner_type=tuner_type_arg,
            model_name=model_name_arg, # Pass model_name to tuner
            verbose=7
        )
    except Exception as e:
        pytest.fail(
            f"{tuner_func.__name__} failed for {project_name}. Error: {e}"
            )

    assert best_hps is not None, "Tuner did not return best_hps."
    assert isinstance(best_hps, dict)
    assert best_model is not None, "Tuner did not return a best model."
    assert isinstance(best_model, model_class_expected)
    assert isinstance(tuner_obj, kt.Tuner)

    try:
        _ = best_model.predict(current_inputs, verbose=0)
    except Exception as e:
        pytest.fail(f"Best model from {tuner_func.__name__} failed to predict."
                    f" Error: {e}")

    assert (Path(temp_dir_path) / project_name).exists()
    # Clean up tuner project directory for this test run
    shutil.rmtree(temp_dir_path / project_name, ignore_errors=True)


# --- Tests for xtft_tuner ---

@pytest.mark.parametrize("use_quantiles", [False, True])
@pytest.mark.parametrize("tuner_type", ["random", "bayesian"])
def test_xtft_tuner_for_xtft_runs(
    tuner_shared_config, dummy_tuner_data, temp_tuner_dir,
    use_quantiles, tuner_type
):
    """Test xtft_tuner for XTFT model."""
    print(f"\nTesting xtft_tuner for XTFT: Q={use_quantiles}, Tuner={tuner_type}")
    _run_tuner_and_asserts(
        xtft_tuner, XTFT, tuner_shared_config, dummy_tuner_data,
        temp_tuner_dir, use_quantiles=use_quantiles,
        model_name_arg="xtft", tuner_type_arg=tuner_type
    )

def test_xtft_tuner_for_superxtft_runs(
    tuner_shared_config, dummy_tuner_data, temp_tuner_dir
):
    
    """Test xtft_tuner for SuperXTFT model."""
    print("\nTesting xtft_tuner for SuperXTFT")
    _run_tuner_and_asserts(
        xtft_tuner, SuperXTFT, tuner_shared_config, dummy_tuner_data,
        temp_tuner_dir, use_quantiles=False,
        model_name_arg="superxtft", tuner_type_arg='random'
    )

def test_xtft_tuner_invalid_inputs(dummy_tuner_data):
    """Test xtft_tuner error handling for invalid inputs."""
    data = dummy_tuner_data
    with pytest.raises(TypeError): # y is required
        xtft_tuner(inputs=[data["X_static"], data["X_dynamic"], data["X_future"]],
                   y=None, forecast_horizon=data["forecast_horizon"])
    with pytest.raises(ValueError, match="Inputs must be a list/tuple of 3 elements"):
        xtft_tuner(inputs=[data["X_static"], data["X_dynamic"]],
                   y=data["y"], forecast_horizon=data["forecast_horizon"])
    print("xtft_tuner invalid inputs: OK")


# --- Tests for tft_tuner ---

@pytest.mark.parametrize("use_quantiles", [False, True])
@pytest.mark.parametrize("model_name_to_test", ["tft", "tft_flex"]) # Test both TFT variants
@pytest.mark.parametrize("tuner_type", ["random", "bayesian"])
def test_tft_tuner_variants_run(
    tuner_shared_config, dummy_tuner_data, temp_tuner_dir,
    use_quantiles, model_name_to_test, tuner_type
):
    """Test tft_tuner for stricter TFT and flexible TemporalFusionTransformer."""
    print(f"\nTesting tft_tuner for {model_name_to_test}: Q={use_quantiles}, Tuner={tuner_type}")

    expected_model_class = TFTStricter if model_name_to_test == "tft" else TFTFlexible

    # For tft_flex, test with potentially None static/future inputs
    current_dummy_data = dummy_tuner_data.copy()
    if model_name_to_test == "tft_flex":
        # Example: test tft_flex with only dynamic inputs
        # This requires _model_builder_factory to handle None for X_static/X_future
        # and pass appropriate None or derived dims to TFTFlexible constructor.
        # For simplicity in this test, we'll still pass all 3 from dummy_tuner_data,
        # but the _model_builder_factory logic for 'tft_flex' should correctly
        # handle passing None for static_input_dim / future_input_dim if
        # corresponding X_static_val / X_future_val are None.
        pass 

    # Using full inputs for now, builder handles flexibility.
    # we can use the prepare_model_inputs for consistent preparations: 
        # transforming in strict mode, the None into a zeros tensors
 
    X_static, X_dynamic, X_future = prepare_model_inputs(
        static_input= current_dummy_data['X_static'], 
        dynamic_input= current_dummy_data["X_dynamic"] ,
        future_input =current_dummy_data['X_future'],
        model_type = 'strict',
      )
    # # update dummy data 
    # current_dummy_data['X_static'] = X_static
    # current_dummy_data['X_dynamic'] = X_dynamic
    # current_dummy_data['X_future'] = X_future
    
    _run_tuner_and_asserts(
        tft_tuner, expected_model_class, tuner_shared_config,
        current_dummy_data, temp_tuner_dir, use_quantiles=use_quantiles,
        model_name_arg=model_name_to_test, tuner_type_arg=tuner_type
    )

# Allows running the tests directly if needed
if __name__=='__main__':
     pytest.main([__file__])

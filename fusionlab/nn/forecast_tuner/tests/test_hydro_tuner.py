# file: fusionlab/nn/forecast_tuner/tests/test_hydro_tuner.py
#XXX TO TEST 
import pytest
import numpy as np
import os
from typing import Dict, Any

try:
    import tensorflow as tf
    from fusionlab.nn.pinn.models import PIHALNet, TransFlowSubsNet
    from fusionlab.nn.forecast_tuner._hydro_tuner import HydroTuner
    KERAS_BACKEND = True
except ImportError:
    KERAS_BACKEND = False

# Skip all tests if backend is not available
pytestmark = pytest.mark.skipif(
    not KERAS_BACKEND, reason="TensorFlow/Keras backend not available"
)

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def common_dimensions() -> Dict[str, int]:
    """Provides common data dimensions for tests."""
    return {
        "n_samples": 64, "past_steps": 10, "horizon": 5,
        "static_dim": 4, "dynamic_dim": 6, "future_dim": 3,
        "subs_dim": 1, "gwl_dim": 1
    }

@pytest.fixture(scope="module")
def dummy_data(common_dimensions: Dict[str, int]) -> Dict[str, Dict]:
    """Provides a standard set of dummy NumPy arrays for inputs and targets."""
    d = common_dimensions
    inputs = {
        "coords": np.random.rand(d["n_samples"], d["horizon"], 3).astype(np.float32),
        "static_features": np.random.rand(d["n_samples"], d["static_dim"]).astype(np.float32),
        "dynamic_features": np.random.rand(d["n_samples"], d["past_steps"], d["dynamic_dim"]).astype(np.float32),
        "future_features": np.random.rand(d["n_samples"], d["horizon"], d["future_dim"]).astype(np.float32),
    }
    targets = {
        "subsidence": np.random.rand(d["n_samples"], d["horizon"], d["subs_dim"]).astype(np.float32),
        "gwl": np.random.rand(d["n_samples"], d["horizon"], d["gwl_dim"]).astype(np.float32)
    }
    return {"inputs": inputs, "targets": targets}

@pytest.fixture(scope="module")
def fixed_params(common_dimensions: Dict[str, int]) -> Dict[str, Any]:
    """Provides the corresponding fixed parameters for the dummy_data."""
    d = common_dimensions
    return {
        "static_input_dim": d["static_dim"],
        "dynamic_input_dim": d["dynamic_dim"],
        "future_input_dim": d["future_dim"],
        "output_subsidence_dim": d["subs_dim"],
        "output_gwl_dim": d["gwl_dim"],
        "forecast_horizon": d["horizon"],
        "max_window_size": d["past_steps"] # Link to data shape
    }

@pytest.fixture
def minimal_search_space() -> Dict[str, Any]:
    """Provides a very small search space for quick test runs."""
    return {
        "learning_rate": [1e-3],
        "dropout_rate": [0.1],
        "embed_dim": [16],
    }

# --- Test Cases ---

def test_hydrotuner_init(fixed_params, minimal_search_space):
    """Tests successful instantiation of the HydroTuner."""
    tuner = HydroTuner(
        model_name_or_cls=PIHALNet,
        fixed_params=fixed_params,
        search_space=minimal_search_space,
        project_name="test_init"
    )
    assert tuner.model_class == PIHALNet
    assert tuner.fixed_params["forecast_horizon"] == 5
    assert "learning_rate" in tuner.search_space

def test_init_with_invalid_model_name(fixed_params, minimal_search_space):
    """Tests that tuner raises an error for an unknown model string."""
    with pytest.raises(ValueError, match="Unknown model name 'InvalidModel'"):
        HydroTuner(
            model_name_or_cls="InvalidModel",
            fixed_params=fixed_params,
            search_space=minimal_search_space
        )

def test_init_with_missing_fixed_params(minimal_search_space):
    """Tests that tuner raises an error if essential fixed params are missing."""
    incomplete_params = {"static_input_dim": 2} # Missing many keys
    with pytest.raises(ValueError, match="Missing required key"):
        HydroTuner(
            model_name_or_cls=PIHALNet,
            fixed_params=incomplete_params,
            search_space=minimal_search_space
        )

def test_param_space_deprecation(fixed_params):
    """Tests that using the legacy 'param_space' raises a FutureWarning."""
    with pytest.warns(FutureWarning, match="`param_space` argument is deprecated"):
        tuner = HydroTuner(
            model_name_or_cls=PIHALNet,
            fixed_params=fixed_params,
            param_space={"learning_rate": [1e-3]} # Using legacy param
        )
    # Ensure the value was correctly transferred
    assert "learning_rate" in tuner.search_space

def test_create_method_infers_dims(dummy_data, minimal_search_space):
    """Tests the .create() factory method for correct dimension inference."""
    tuner = HydroTuner.create(
        model_name_or_cls=PIHALNet,
        inputs_data=dummy_data["inputs"],
        targets_data=dummy_data["targets"],
        search_space=minimal_search_space
    )
    # Check if inferred dimensions match the data shapes
    assert tuner.fixed_params["dynamic_input_dim"] == dummy_data["inputs"]["dynamic_features"].shape[-1]
    assert tuner.fixed_params["output_gwl_dim"] == dummy_data["targets"]["gwl"].shape[-1]
    assert tuner.fixed_params["forecast_horizon"] == dummy_data["targets"]["gwl"].shape[1]

def test_create_with_override(dummy_data, minimal_search_space):
    """Tests that `fixed_params` in .create() override inferred values."""
    manual_overrides = {"forecast_horizon": 99, "activation": "gelu"}
    tuner = HydroTuner.create(
        model_name_or_cls=PIHALNet,
        inputs_data=dummy_data["inputs"],
        targets_data=dummy_data["targets"],
        search_space=minimal_search_space,
        fixed_params=manual_overrides
    )
    assert tuner.fixed_params["forecast_horizon"] == 99
    assert tuner.fixed_params["activation"] == "gelu"

@pytest.mark.parametrize("model_cls", [PIHALNet, TransFlowSubsNet])
def test_end_to_end_run(
    model_cls, dummy_data, minimal_search_space, tmp_path
):
    """
    Performs a minimal end-to-end run test for both supported models.
    This implicitly tests the `build` and `run` methods.
    """
    project_dir = tmp_path / model_cls.__name__
    
    tuner = HydroTuner.create(
        model_name_or_cls=model_cls,
        inputs_data=dummy_data["inputs"],
        targets_data=dummy_data["targets"],
        search_space=minimal_search_space,
        max_trials=1,
        executions_per_trial=1,
        project_name="end_to_end_test",
        directory=str(project_dir)
    )

    best_model, best_hps, tuner_instance = tuner.run(
        inputs=dummy_data["inputs"],
        y=dummy_data["targets"],
        validation_data=(dummy_data["inputs"], dummy_data["targets"]),
        epochs=1,
        batch_size=16
    )

    assert best_model is not None
    assert isinstance(best_model, model_cls)
    assert best_hps is not None
    assert "learning_rate" in best_hps.values
    assert tuner_instance is not None
    assert os.path.exists(project_dir / "end_to_end_test")

def test_build_with_model_specific_hps(fixed_params, tmp_path):
    """
    Tests that the build method correctly handles hyperparameters specific
    to TransFlowSubsNet.
    """
    project_dir = tmp_path / "TransFlowSpecific"
    
    # Define a search space that includes HPs only used by TransFlowSubsNet
    transflow_search_space = {
        "learning_rate": [1e-3],
        "K": ["learnable", 1e-4],
        "lambda_gw": {"type": "float", "min_value": 0.1, "max_value": 1.0}
    }
    
    tuner = HydroTuner(
        model_name_or_cls=TransFlowSubsNet,
        fixed_params=fixed_params,
        search_space=transflow_search_space,
        max_trials=1,
        project_name="tf_specific_test",
        directory=str(project_dir)
    )
    
    # Run a minimal search to trigger the build method
    try:
        tuner.run(
            inputs=dummy_data(common_dimensions())["inputs"],
            y=dummy_data(common_dimensions())["targets"],
            epochs=1
        )
    except Exception as e:
        pytest.fail(
            "HydroTuner.run failed for TransFlowSubsNet with specific HPs: "
            f"{e}"
        )
        
if __name__ =='__main__': # pragma : no cover 
    pytest.main( [__file__,  "--maxfail=1 ", "--disable-warnings",  "-q"])
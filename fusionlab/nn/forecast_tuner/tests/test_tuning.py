import pytest
import numpy as np

from unittest.mock import patch, MagicMock

from typing import Dict, Tuple 
try:
    import tensorflow as tf
    import keras_tuner as kt # Renamed from kt to avoid conflict with local kt var
    
    from fusionlab.nn.pinn.models import PiHALNet
    from fusionlab.nn.forecast_tuner._pihal_tuner import PIHALTuner, DEFAULT_PIHALNET_FIXED_PARAMS
    from fusionlab.nn.forecast_tuner._base_tuner import PINNTunerBase
    from fusionlab.utils.generic_utils import rename_dict_keys
    from fusionlab.core.checks import exist_features

    FUSIONLAB_AVAILABLE = True
    HAS_KT_LIB = True # Renamed from HAS_KT to avoid conflict
except ImportError as e:
    print(f"Could not import fusionlab PINN components for tuning tests: {e}")
    FUSIONLAB_AVAILABLE = False
    HAS_KT_LIB = False

         
pytestmark = pytest.mark.skipif(
    not FUSIONLAB_AVAILABLE,
    reason="fusionlab.nn.pinn.models.PiHALNet or dependencies not found"
)

# --- Test Parameters ---
BATCH_SIZE_TUNER = 2
T_PAST_TUNER = 8
T_HORIZON_TUNER = 4
S_DIM_TUNER = 3
D_DIM_TUNER = 5
F_DIM_TUNER = 2
OUT_S_DIM_TUNER = 1
OUT_G_DIM_TUNER = 1

# --- Helper to generate raw NumPy data ---
def generate_raw_numpy_data_for_tuner(
    batch_size=BATCH_SIZE_TUNER,
    t_past=T_PAST_TUNER, t_horizon=T_HORIZON_TUNER,
    s_dim=S_DIM_TUNER, d_dim=D_DIM_TUNER, f_dim=F_DIM_TUNER,
    out_s_dim=OUT_S_DIM_TUNER, out_g_dim=OUT_G_DIM_TUNER
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    inputs_np = {}
    inputs_np['coords'] = np.random.rand(batch_size, t_horizon, 3).astype(np.float32)
    inputs_np['dynamic_features'] = np.random.rand(batch_size, t_past, d_dim).astype(np.float32)
    
    if s_dim > 0:
        inputs_np['static_features'] = np.random.rand(batch_size, s_dim).astype(np.float32)
    else: # Must provide key even if empty, for _infer_dims_and_prepare_fixed_params
        inputs_np['static_features'] = np.zeros(
            (batch_size, 0), dtype=np.float32)
        
    if f_dim > 0:
        # Assuming future_features for PiHALNet.call (via run_halnet_core)
        # are the full span that align_temporal_dimensions will slice from.
        # However, prepare_pinn_data_sequences outputs future_features of length T_HORIZON.
        # The _infer_dims logic needs to be consistent with what PiHALNet.call expects.
        # For PIHALTuner.create, inputs_data is what prepare_pinn_data_sequences outputs.
        inputs_np['future_features'] = np.random.rand(
            batch_size, t_horizon, f_dim).astype(np.float32)
    else:
        inputs_np['future_features'] = np.zeros(
            (batch_size, t_horizon, 0), dtype=np.float32)
        
    targets_np = {
        'subsidence': np.random.rand(batch_size, t_horizon, out_s_dim).astype(np.float32),
        'gwl': np.random.rand(batch_size, t_horizon, out_g_dim).astype(np.float32)
    }
    # Keys for targets_data in _infer_dims_and_prepare_fixed_params are 'subs_pred', 'gwl_pred'
    # after rename_dict_keys. So, generate_raw_numpy_data_for_tuner should use these.
    targets_np_renamed = {
        'subs_pred': targets_np['subsidence'],
        'gwl_pred': targets_np['gwl']
    }
    return inputs_np, targets_np_renamed


@pytest.mark.skipif(
    not FUSIONLAB_AVAILABLE or not HAS_KT_LIB, 
    reason="PiHALNet or KerasTuner not available")
def test_pihaltuner_create_with_data_inference():
    """Tests PIHALTuner.create() with data inference for fixed_model_params."""
    raw_inputs, raw_targets = generate_raw_numpy_data_for_tuner(
        s_dim=S_DIM_TUNER, d_dim=D_DIM_TUNER, f_dim=F_DIM_TUNER
    )
    tuner = PIHALTuner.create(
        inputs_data=raw_inputs,
        targets_data=raw_targets,
        forecast_horizon=T_HORIZON_TUNER,
        quantiles=None,
        objective='val_loss',
        max_trials=1, # Minimal for instantiation test
        project_name="test_create_infer"
    )
    assert isinstance(tuner, PIHALTuner)
    assert tuner.fixed_model_params["static_input_dim"] == S_DIM_TUNER
    assert tuner.fixed_model_params["dynamic_input_dim"] == D_DIM_TUNER
    assert tuner.fixed_model_params["future_input_dim"] == F_DIM_TUNER
    assert tuner.fixed_model_params["output_subsidence_dim"] == OUT_S_DIM_TUNER
    assert tuner.fixed_model_params["output_gwl_dim"] == OUT_G_DIM_TUNER
    assert tuner.fixed_model_params["forecast_horizon"] == T_HORIZON_TUNER
    assert tuner.fixed_model_params["quantiles"] is None

@pytest.mark.skipif(
    not FUSIONLAB_AVAILABLE or not HAS_KT_LIB,
    reason="PiHALNet or KerasTuner not available"
   )
def test_pihaltuner_create_with_explicit_fixed_params():
    """Tests PIHALTuner.create() with explicitly provided fixed_model_params."""
    explicit_fixed = {
        "static_input_dim": 2, "dynamic_input_dim": 6, "future_input_dim": 1,
        "output_subsidence_dim": 1, "output_gwl_dim": 1, "forecast_horizon": 5,
        "quantiles": [0.5]
    }
    tuner = PIHALTuner.create(
        fixed_model_params=explicit_fixed, # Use override
        objective='val_loss',
        max_trials=1,
        project_name="test_create_explicit"
    )
    assert isinstance(tuner, PIHALTuner)
    for key, val in explicit_fixed.items():
        assert tuner.fixed_model_params[key] == val

@pytest.mark.skipif(
    not FUSIONLAB_AVAILABLE or not HAS_KT_LIB, 
    reason="PiHALNet or KerasTuner not available"
)
def test_pihaltuner_init_direct_with_fixed_params():
    """Tests direct PIHALTuner instantiation with fixed_model_params."""
    fixed_params = DEFAULT_PIHALNET_FIXED_PARAMS.copy()
    fixed_params.update({
        "static_input_dim": S_DIM_TUNER,
        "dynamic_input_dim": D_DIM_TUNER,
        "future_input_dim": F_DIM_TUNER,
        "output_subsidence_dim": OUT_S_DIM_TUNER,
        "output_gwl_dim": OUT_G_DIM_TUNER, 
        "forecast_horizon": T_HORIZON_TUNER,
    })
    tuner = PIHALTuner(
        fixed_model_params=fixed_params,
        objective='val_loss', max_trials=1,
        project_name="test_init_direct"
    )
    assert tuner.fixed_model_params["dynamic_input_dim"] == D_DIM_TUNER

@pytest.mark.skipif(
    not FUSIONLAB_AVAILABLE or not HAS_KT_LIB, 
    reason="PiHALNet or KerasTuner not available"
  )
def test_pihaltuner_build_method_from_create():
    """Tests the build method after instantiation via `create`."""
    raw_inputs, raw_targets = generate_raw_numpy_data_for_tuner()
    tuner = PIHALTuner.create(
        inputs_data=raw_inputs, targets_data=raw_targets,
        forecast_horizon=T_HORIZON_TUNER,
        objective='val_loss', max_trials=1,
        project_name="test_build_via_create"
    )
    hp = kt.HyperParameters() # Keras Tuner provides this
    model = tuner.build(hp)
    assert isinstance(model, PiHALNet)
    assert model.optimizer is not None

@pytest.mark.skipif(
    not FUSIONLAB_AVAILABLE or not HAS_KT_LIB, 
    reason="PiHALNet or KerasTuner not available"
 )
def test_pihaltuner_run_calls_search(mocker):
    """Tests that PIHALTuner.runt() correctly calls super().search()."""
    fixed_params = DEFAULT_PIHALNET_FIXED_PARAMS.copy()
    fixed_params.update({
        "static_input_dim": S_DIM_TUNER,
        "dynamic_input_dim": D_DIM_TUNER,
        "future_input_dim": F_DIM_TUNER, 
        "output_subsidence_dim": OUT_S_DIM_TUNER,
        "output_gwl_dim": OUT_G_DIM_TUNER,
        "forecast_horizon": T_HORIZON_TUNER,
    })
    tuner = PIHALTuner(
        fixed_model_params=fixed_params,
        objective='val_loss', max_trials=1,
        project_name="test_run_calls_search"
    )
    
    raw_inputs_np, raw_targets_np = generate_raw_numpy_data_for_tuner()

    # Mock the super().search() method
    mocked_base_search = mocker.patch.object(PINNTunerBase, 'search', autospec=True)
    mocked_base_search.return_value = (None, None, None) # Mock its return

    tuner.run(
        inputs=raw_inputs_np,
        y=raw_targets_np,
        # forecast_horizon and quantiles are now part of fixed_model_params
        epochs=1,
        batch_size=BATCH_SIZE_TUNER,
        verbose=0
    )
    mocked_base_search.assert_called_once()
    # Check if train_dataset passed to search is a tf.data.Dataset
    args, kwargs = mocked_base_search.call_args
    assert 'train_data' in kwargs
    assert isinstance(kwargs['train_data'], tf.data.Dataset)


@pytest.mark.slow
@pytest.mark.skipif(
    not FUSIONLAB_AVAILABLE or not HAS_KT_LIB,
    reason="PiHALNet or KerasTuner not available"
 )
def test_pihaltuner_run_short_search_smoke_test():
    """Smoke test for a minimal PIHALTuner.run() run."""
    raw_inputs_np, raw_targets_np = generate_raw_numpy_data_for_tuner(
        s_dim=S_DIM_TUNER, d_dim=D_DIM_TUNER, f_dim=F_DIM_TUNER
    )
    # Use create method for this end-to-end test
    tuner = PIHALTuner.create(
        inputs_data=raw_inputs_np,
        targets_data=raw_targets_np,
        forecast_horizon=T_HORIZON_TUNER,
        quantiles=None, # Simpler for smoke test
        objective='val_total_loss',
        max_trials=1, executions_per_trial=1,
        project_name="pihal_smoke_e2e", directory="tuner_smoke_results_e2e",
        tuner_type='randomsearch', overwrite_tuner=True, seed=42
    )

    # Create dummy validation data
    val_inputs_np, val_targets_np = generate_raw_numpy_data_for_tuner(
        batch_size=1, s_dim=S_DIM_TUNER, d_dim=D_DIM_TUNER, f_dim=F_DIM_TUNER
    )
    
    try:
        best_model, best_hps, _ = tuner.run(
            inputs=raw_inputs_np, # Data for creating tf.data.Dataset
            y=raw_targets_np,
            validation_data=(val_inputs_np, val_targets_np),
            # forecast_horizon & quantiles already set via .create()
            epochs=1, batch_size=BATCH_SIZE_TUNER, verbose=0
        )
        assert best_model is not None
        assert best_hps is not None
        assert isinstance(best_model, PiHALNet)
    except Exception as e:
        import traceback
        traceback.print_exc()
        pytest.fail(f"PIHALTuner smoke test (run method) failed: {e}")

if __name__ == '__main__':
    if FUSIONLAB_AVAILABLE and HAS_KT_LIB:
        pytest.main([__file__, "-vv"]) # Run all tests
        # pytest.main([__file__, "-vv", "-m slow"]) # To include slow tests
    else:
        print("Skipping PIHALTuner tests: fusionlab components or keras-tuner not available.")
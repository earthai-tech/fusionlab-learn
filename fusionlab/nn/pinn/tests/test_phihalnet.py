import pytest
import numpy as np
from typing import Dict 
# Attempt to import PIHALNet and its dependencies
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input as KerasInput # Alias to avoid confusion
    from tensorflow.keras.models import Model as KerasFunctionalModel # Alias

    from fusionlab.nn.pinn.models import PIHALNet
    from fusionlab.nn.pinn.op import (
        process_pinn_inputs, 
        compute_consolidation_residual
    )
    from fusionlab.nn._tensor_validation import validate_model_inputs 
    # Import other components PIHALNet uses if they are not automatically
    # registered or if needed for type checking/mocking (usually not needed for tests)
    FUSIONLAB_AVAILABLE = True
except ImportError as e:
    print(f"Could not import PIHALNet or dependencies: {e}")
    FUSIONLAB_AVAILABLE = False
    # Dummy PIHALNet for test collection if imports fail
    class PIHALNet(tf.keras.Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Store key params to avoid attribute errors in test structure
            self.output_subsidence_dim = kwargs.get('output_subsidence_dim', 1)
            self.output_gwl_dim = kwargs.get('output_gwl_dim', 1)
            self.forecast_horizon = kwargs.get('forecast_horizon', 1)
            self.quantiles = kwargs.get('quantiles', None)
        def call(self, inputs, training=False):
            # Simplified dummy output for placeholder
            batch_size = tf.shape(inputs['coords'])[0]
            h = self.forecast_horizon
            s_dim, g_dim = self.output_subsidence_dim, self.output_gwl_dim
            q_len = len(self.quantiles) if self.quantiles else 0

            s_shape = (batch_size, h, q_len, s_dim) if q_len else (batch_size, h, s_dim)
            g_shape = (batch_size, h, q_len, g_dim) if q_len else (batch_size, h, g_dim)
            
            return {
                "subs_pred": tf.zeros(s_shape, dtype=tf.float32),
                "gwl_pred": tf.zeros(g_shape, dtype=tf.float32),
                "pde_residual": tf.zeros((batch_size, h -1 if h > 1 else 1, s_dim), dtype=tf.float32)
            }
        def get_pinn_coefficient_C(self): return tf.constant(0.01)

pytestmark = pytest.mark.skipif(
    not FUSIONLAB_AVAILABLE,
    reason="fusionlab.nn.pinn.models.PIHALNet or dependencies not found"
)
# --- Test Parameters ---
BATCH_SIZE = 2 # Keep small for tests
T_PAST = 8     # Lookback window
T_HORIZON = 4  # Forecast horizon

S_DIM_DEF = 3  # Static features
D_DIM_DEF = 5  # Dynamic features
F_DIM_DEF = 2  # Future known features

OUT_S_DIM = 1
OUT_G_DIM = 1

EMBED_DIM_TEST = 16
HIDDEN_UNITS_TEST = 16
LSTM_UNITS_TEST = 16
ATTN_UNITS_TEST = 16
NUM_HEADS_TEST = 1
VSN_UNITS_TEST = 16

# --- Helper Function to Generate Dummy Input Dictionaries ---
def generate_pinn_input_dict(
    batch_size=BATCH_SIZE,
    t_past=T_PAST,
    t_horizon=T_HORIZON,
    s_dim=S_DIM_DEF,
    d_dim=D_DIM_DEF,
    f_dim=F_DIM_DEF
) -> Dict[str, tf.Tensor]:
    """Generates a dictionary of dummy input tensors for PIHALNet."""
    inputs = {}
    # Coords: (batch, horizon, 3) for [t, x, y] for the forecast window
    # The 't' here is the time for which predictions are made (for PDE residual)
    coords_t = np.random.rand(batch_size, t_horizon, 1).astype(np.float32)
    coords_x = np.random.rand(batch_size, t_horizon, 1).astype(np.float32)
    coords_y = np.random.rand(batch_size, t_horizon, 1).astype(np.float32)
    inputs['coords'] = tf.constant(np.concatenate([coords_t, coords_x, coords_y], axis=-1))

    if s_dim > 0:
        inputs['static_features'] = tf.constant(
            np.random.rand(batch_size, s_dim).astype(np.float32)
        )
    else: # Model's validate_model_inputs expects this structure if strict
        inputs['static_features'] = tf.constant(
            np.zeros((batch_size, 0)).astype(np.float32)
        )
        
    inputs['dynamic_features'] = tf.constant(
        np.random.rand(batch_size, t_past, d_dim).astype(np.float32)
    )
    
    if f_dim > 0:
        # Future known covariates for the entire span (past + horizon)
        # as align_temporal_dimensions will slice it.
        # Or, if it's only for the horizon, adjust length.
        # PIHALNet's align_temporal_dimensions uses future_input for embedding path
        # and also needs it for the VSN path if use_vsn=True.
        # Let's assume future_features are provided for T_PAST + T_HORIZON
        inputs['future_features'] = tf.constant(
            np.random.rand(batch_size, t_past + t_horizon, f_dim).astype(np.float32)
        )
    else:
        inputs['future_features'] = tf.constant(
            np.zeros((batch_size, t_past + t_horizon, 0)).astype(np.float32)
        )
        
    return inputs

# In your test file: fusionlab/nn/pinn/tests/test_phihalnet.py

def generate_pinn_target_dict(
    batch_size=BATCH_SIZE,
    t_horizon=T_HORIZON,
    out_s_dim=OUT_S_DIM,
    out_g_dim=OUT_G_DIM,
    quantiles=None # This parameter is not used for y_true shape
) -> Dict[str, tf.Tensor]:
    """Generates a dictionary of dummy target tensors for PIHALNet."""
    targets = {}
    # y_true (targets) should NOT have a quantile dimension.
    # Shape: (Batch, Horizon, Output_Dim_Per_Target)
    targets['subs_pred'] = tf.constant(
        np.random.rand(batch_size, t_horizon, out_s_dim).astype(np.float32)
    )
    targets['gwl_pred'] = tf.constant(
        np.random.rand(batch_size, t_horizon, out_g_dim).astype(np.float32)
    )
    return targets

# def generate_pinn_target_dict(
#     batch_size=BATCH_SIZE,
#     t_horizon=T_HORIZON,
#     out_s_dim=OUT_S_DIM,
#     out_g_dim=OUT_G_DIM,
#     quantiles=None
# ) -> Dict[str, tf.Tensor]:
#     """Generates a dictionary of dummy target tensors for PIHALNet."""
#     targets = {}
#     # CORRECTED: y_true should NOT have a quantile dimension.
#     # Its shape is (Batch, Horizon, Output_Dim_Per_Target)
    
#     targets['subsidence'] = tf.constant(
#         np.random.rand(batch_size, t_horizon, out_s_dim).astype(np.float32)
#     )
#     targets['gwl'] = tf.constant(
#         np.random.rand(batch_size, t_horizon, out_g_dim).astype(np.float32)
#     )
#     if quantiles:
#         num_q = len(quantiles)
#         targets['subs_pred'] = tf.constant(
#             np.random.rand(batch_size, t_horizon, num_q, out_s_dim).astype(np.float32)
#         )
#         targets['gwl_pred'] = tf.constant(
#             np.random.rand(batch_size, t_horizon, num_q, out_g_dim).astype(np.float32)
#         )
#     else:
#         targets['subs_pred'] = tf.constant(
#             np.random.rand(batch_size, t_horizon, out_s_dim).astype(np.float32)
#         )
#         targets['gwl_pred'] = tf.constant(
#             np.random.rand(batch_size, t_horizon, out_g_dim).astype(np.float32)
#         )
#     return targets

# --- Pytest Test Functions ---
@pytest.mark.skipif(not FUSIONLAB_AVAILABLE, reason="fusionlab PIHALNet not available")
@pytest.mark.parametrize("use_vsn_config", [True, False])
@pytest.mark.parametrize("pinn_c_config", ['learnable', 0.05, None])
@pytest.mark.parametrize("quantiles_config", [None, [0.25, 0.5, 0.75]])
def test_pihalnet_instantiation(use_vsn_config, pinn_c_config, quantiles_config):
    """Tests PIHALNet instantiation with various configurations."""
    model = PIHALNet(
        static_input_dim=S_DIM_DEF,
        dynamic_input_dim=D_DIM_DEF,
        future_input_dim=F_DIM_DEF,
        output_subsidence_dim=OUT_S_DIM,
        output_gwl_dim=OUT_G_DIM,
        embed_dim=EMBED_DIM_TEST,
        hidden_units=HIDDEN_UNITS_TEST,
        lstm_units=LSTM_UNITS_TEST,
        attention_units=ATTN_UNITS_TEST,
        num_heads=NUM_HEADS_TEST,
        forecast_horizon=T_HORIZON,
        quantiles=quantiles_config,
        pinn_coefficient_C=pinn_c_config,
        use_vsn=use_vsn_config,
        vsn_units=VSN_UNITS_TEST if use_vsn_config else None
    )
    assert isinstance(model, PIHALNet)
    assert model.use_vsn == use_vsn_config
    assert model.pinn_coefficient_C_config == pinn_c_config
    if quantiles_config:
        assert model.quantiles == quantiles_config
    else: # set_default_params might set a default
        assert model.quantiles is None or isinstance(model.quantiles, list)

    # Test get_pinn_coefficient_C
    c_val = model.get_pinn_coefficient_C()
    assert isinstance(c_val, tf.Tensor)
    if isinstance(pinn_c_config, float):
        np.testing.assert_allclose(c_val.numpy(), pinn_c_config, atol=1e-6)

@pytest.mark.skipif(not FUSIONLAB_AVAILABLE, reason="fusionlab PIHALNet not available")
@pytest.mark.parametrize("s_dim, d_dim, f_dim", [
    (S_DIM_DEF, D_DIM_DEF, F_DIM_DEF), # All present
    (0, D_DIM_DEF, F_DIM_DEF),         # No static
    (S_DIM_DEF, D_DIM_DEF, 0),         # No future known
    (0, D_DIM_DEF, 0),                 # Dynamic only
])
@pytest.mark.parametrize("quantiles_config", [None, [0.1, 0.5, 0.9]])
@pytest.mark.parametrize("use_vsn_flag", [True, False])
def test_pihalnet_forward_pass_shapes(s_dim, d_dim, f_dim, quantiles_config, use_vsn_flag):
    """Tests PIHALNet forward pass and output shapes."""
    model = PIHALNet(
        static_input_dim=s_dim,
        dynamic_input_dim=d_dim,
        future_input_dim=f_dim,
        output_subsidence_dim=OUT_S_DIM,
        output_gwl_dim=OUT_G_DIM,
        embed_dim=EMBED_DIM_TEST,
        hidden_units=HIDDEN_UNITS_TEST,
        lstm_units=LSTM_UNITS_TEST,
        attention_units=ATTN_UNITS_TEST,
        num_heads=NUM_HEADS_TEST,
        forecast_horizon=T_HORIZON,
        quantiles=quantiles_config,
        use_vsn=use_vsn_flag,
        vsn_units=VSN_UNITS_TEST if use_vsn_flag else None,
        dropout_rate=0.0, # Disable dropout for deterministic shape tests
        pinn_coefficient_C=0.01 # Fixed C
    )
    
    dummy_inputs_dict = generate_pinn_input_dict(
        s_dim=s_dim, d_dim=d_dim, f_dim=f_dim
    )
    
    # Build the model by calling it (subclassed model)
    outputs_dict = model(dummy_inputs_dict, training=False)

    assert "subs_pred" in outputs_dict
    assert "gwl_pred" in outputs_dict
    assert "pde_residual" in outputs_dict

    # Expected shapes
    num_q = len(quantiles_config) if quantiles_config else 0
    
    expected_subs_shape = [BATCH_SIZE, T_HORIZON]
    if num_q > 0:
        expected_subs_shape.append(num_q)
    expected_subs_shape.append(OUT_S_DIM)
    assert outputs_dict["subs_pred"].shape.as_list() == expected_subs_shape

    expected_gwl_shape = [BATCH_SIZE, T_HORIZON]
    if num_q > 0:
        expected_gwl_shape.append(num_q)
    expected_gwl_shape.append(OUT_G_DIM)
    assert outputs_dict["gwl_pred"].shape.as_list() == expected_gwl_shape
    
    # PDE residual shape: (Batch, Horizon-1, Output_Dim_of_Mean_Pred)
    # If horizon=1, residual is zeros_like mean_pred.
    # s_pred_mean_for_pde has OUT_S_DIM as last dim.
    expected_pde_shape = [BATCH_SIZE, T_HORIZON - 1 if T_HORIZON > 1 else T_HORIZON, OUT_S_DIM]
    assert outputs_dict["pde_residual"].shape.as_list() == expected_pde_shape

@pytest.mark.skipif(not FUSIONLAB_AVAILABLE, reason="fusionlab PIHALNet not available")
def test_pihalnet_compilation_and_train_step():
    """Tests model compilation and a single training step."""
    s_dim, d_dim, f_dim = S_DIM_DEF, D_DIM_DEF, F_DIM_DEF
    quantiles = [0.1, 0.5, 0.9]
    
    model = PIHALNet(
        static_input_dim=s_dim,
        dynamic_input_dim=d_dim,
        future_input_dim=f_dim,
        output_subsidence_dim=OUT_S_DIM,
        output_gwl_dim=OUT_G_DIM,
        embed_dim=EMBED_DIM_TEST,
        hidden_units=HIDDEN_UNITS_TEST,
        lstm_units=LSTM_UNITS_TEST,
        attention_units=ATTN_UNITS_TEST,
        num_heads=NUM_HEADS_TEST,
        forecast_horizon=T_HORIZON,
        quantiles=quantiles,
        pinn_coefficient_C='learnable', # Test learnable C
        dropout_rate=0.0
    )

    dummy_inputs_dict = generate_pinn_input_dict(
        s_dim=s_dim, d_dim=d_dim, f_dim=f_dim)
    dummy_targets_dict = generate_pinn_target_dict(quantiles=quantiles)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={ # Loss for data terms
            'subs_pred': tf.keras.losses.MeanSquaredError(), # Example
            'gwl_pred': tf.keras.losses.MeanSquaredError()  # Example
        },
        metrics={ # Metrics for data terms
            'subs_pred': ['mae'],
            'gwl_pred': ['mae']
        },
        loss_weights={'subs_pred': 1.0, 'gwl_pred': 0.8}, # Example weights
        lambda_pde=0.1 # Weight for the physics loss
    )
    
    # Create a tf.data.Dataset for training
    # Inside test_pihalnet_compilation_and_train_step
    # ... after generating dummy_inputs_dict and dummy_targets_dict ...
    print("--- Input Shapes for Dataset ---")
    for key, val in dummy_inputs_dict.items():
        if hasattr(val, 'shape'):
            print(f"Input '{key}': {val.shape}")
        else:
            print(f"Input '{key}': {type(val)}") # Should not be None here
    
    print("--- Target Shapes for Dataset ---")
    for key, val in dummy_targets_dict.items():
        if hasattr(val, 'shape'):
            print(f"Target '{key}': {val.shape}")
        else:
            print(f"Target '{key}': {type(val)}")
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (dummy_inputs_dict, dummy_targets_dict)
    ).batch(BATCH_SIZE)

    # Perform one training step
    history = model.fit(dataset, epochs=1, verbose=0)

    # assert 'loss' not in history.history # total_loss is not 'loss' by default
    assert 'total_loss' in history.history
    assert 'data_loss' in history.history
    assert 'physics_loss' in history.history
    assert 'subs_pred_mae' in history.history # Check one of the compiled metrics
    assert model.get_pinn_coefficient_C().numpy() is not None # Check C is accessible

@pytest.mark.skipif(not FUSIONLAB_AVAILABLE, reason="fusionlab PIHALNet not available")
def test_pihalnet_serialization_config():
    """Tests get_config and from_config for PIHALNet."""
    config_orig = {
        "static_input_dim": S_DIM_DEF,
        "dynamic_input_dim": D_DIM_DEF,
        "future_input_dim": F_DIM_DEF,
        "output_subsidence_dim": OUT_S_DIM,
        "output_gwl_dim": OUT_G_DIM,
        "embed_dim": EMBED_DIM_TEST,
        "hidden_units": HIDDEN_UNITS_TEST,
        "lstm_units": LSTM_UNITS_TEST,
        "attention_units": ATTN_UNITS_TEST,
        "num_heads": NUM_HEADS_TEST,
        "dropout_rate": 0.1,
        "forecast_horizon": T_HORIZON,
        "quantiles": [0.2, 0.8],
        "max_window_size": 15,
        "memory_size": 50,
        "scales": [1, 3],
        "multi_scale_agg": "average",
        "final_agg": "mean", # Should be 'average' for consistency
        "activation": "gelu",
        "use_residuals": False,
        "use_batch_norm": True,
        "pinn_coefficient_C": 0.02,
        "use_vsn": False,
        "vsn_units": None,
        "name": "TestPIHALNetSerialization"
    }
    # Correct final_agg if needed
    if config_orig["final_agg"] == "mean": config_orig["final_agg"] = "average"

    model1 = PIHALNet(**config_orig)
    
    # Build the model by calling it with symbolic inputs
    s_in_spec = KerasInput(shape=(S_DIM_DEF,), name="s_in_cfg") if S_DIM_DEF > 0 else None
    d_in_spec = KerasInput(shape=(T_PAST, D_DIM_DEF), name="d_in_cfg")
    # f_in_spec = KerasInput(shape=(T_HORIZON, F_DIM_DEF), name="f_in_cfg") if F_DIM_DEF > 0 else None
    f_in_spec = KerasInput(shape=(T_PAST + T_HORIZON, F_DIM_DEF), name="f_in_cfg_full_future") if F_DIM_DEF > 0 else None
    coords_in_spec = KerasInput(shape=(T_HORIZON, 3), name="coords_in_cfg")

    call_inputs_dict = {'coords': coords_in_spec, 'dynamic_features': d_in_spec}
    if S_DIM_DEF > 0:
        call_inputs_dict['static_features'] = s_in_spec
    if F_DIM_DEF > 0:
        call_inputs_dict['future_features'] = f_in_spec
    
    _ = model1(call_inputs_dict) # Build model

    config_retrieved = model1.get_config()
    
    # Check that all original __init__ params are in the retrieved config
    for key, value in config_orig.items():
        if key == "name": continue # Keras handles name separately
        assert key in config_retrieved, f"Key '{key}' missing in get_config()"
        # Special handling for quantiles and scales as they are processed by set_default_params
        if key == "quantiles":
            assert config_retrieved[key] == value, f"Config mismatch for {key}"
        elif key == "scales":
              assert config_retrieved[key] == value, f"Config mismatch for {key}"
        elif key == "multi_scale_agg": # multi_scale_agg_mode is saved
            assert config_retrieved["multi_scale_agg"] == value, f"Config mismatch for {key}"
        # This config retrieved can by passed since since 
        # vsn_units, if None, is auto selected
        elif not callable(value): # Skip direct comparison for callables
            assert config_retrieved[key] == value if value is not None\
                else config_orig["hidden_units"], f"Config mismatch for {key}"

    # Test reconstruction
    try:
        # Ensure all custom objects are known to Keras or passed via custom_objects
        model2 = PIHALNet.from_config(config_retrieved)
        _ = model2(call_inputs_dict) # Call reconstructed model
        assert isinstance(model2, PIHALNet)
        print()
        assert model2.name == model1.name # Name should be preserved
    except Exception as e:
        pytest.fail(f"PIHALNet.from_config failed: {e}\nConfig was: {config_retrieved}")


# This allows running the test file directly using `python path/to/test_models.py`
if __name__ == '__main__':
    # This will run all tests in the current file
    # Ensure FUSIONLAB_AVAILABLE is True or tests will be skipped
    if FUSIONLAB_AVAILABLE:
        pytest.main([__file__, "-vv"]) # Add -vv for more verbose output
    else:
        print("Skipping tests as fusionlab components could not be imported.")
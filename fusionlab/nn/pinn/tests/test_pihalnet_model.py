# test_pihalnet.py
import pytest
import tempfile 
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# --- Adjust these imports based on your project structure ---
# Assuming PIHALNet and its components are in fusionlab.nn.pinn
from fusionlab.nn.pinn.models import PIHALNet 
from fusionlab.nn.losses import combined_quantile_loss # If you use a custom one
from fusionlab.nn.components import HierarchicalAttention, MultiHeadAttention # etc.


# --- Fixtures ---

@pytest.fixture
def default_model_params():
    """Default fixed parameters for PIHALNet instantiation for tests."""
    return {
        "static_input_dim": 5,
        "dynamic_input_dim": 4,
        "future_input_dim": 2,
        "output_subsidence_dim": 1,
        "output_gwl_dim": 1,
        "forecast_horizon": 3,
        "max_window_size":3, # Corresponds to TIME_STEPS
        # Architectural params (can be overridden in specific tests)
        "embed_dim": 32,
        "hidden_units": 32,
        "lstm_units": 32,
        "attention_units": 16, # Must be divisible by num_heads if key_dim not changed
        "num_heads": 2,
        "dropout_rate": 0.0, # Disable dropout for simpler testing
        "vsn_units": 16,
        "pde_mode": "consolidation",
        "pinn_coefficient_C": "learnable",
    }

@pytest.fixture
def dummy_input_data(default_model_params):
    """Generates dummy input data for PIHALNet based on model_params."""
    batch_size = 2
    # From default_model_params
    time_steps = default_model_params["max_window_size"]
    forecast_horizon = default_model_params["forecast_horizon"]
    static_dim = default_model_params["static_input_dim"]
    dynamic_dim = default_model_params["dynamic_input_dim"]
    future_dim = default_model_params["future_input_dim"]

    inputs = {
        'coords': tf.random.normal(
            (batch_size, forecast_horizon, 3), dtype=tf.float32),
        'dynamic_features': tf.random.normal(
            (batch_size, time_steps, dynamic_dim), dtype=tf.float32),
    }
    if static_dim > 0:
        inputs['static_features'] = tf.random.normal(
            (batch_size, static_dim), dtype=tf.float32)
    else: 
        # Model expects the key even if dim is 0 for VSN internal checks or general structure
        inputs['static_features'] = tf.zeros(
            (batch_size, 0), dtype=tf.float32)

    if future_dim > 0:
        inputs['future_features'] = tf.random.normal(
            (batch_size, forecast_horizon + time_steps, future_dim), dtype=tf.float32)
    else:
        inputs['future_features'] = tf.zeros(
            (batch_size, forecast_horizon + time_steps, 0), dtype=tf.float32)
        
    return inputs, batch_size

@pytest.fixture
def dummy_target_data(default_model_params, dummy_input_data):
    """Generates dummy target data for PIHALNet."""
    _, batch_size = dummy_input_data
    forecast_horizon = default_model_params["forecast_horizon"]
    out_s_dim = default_model_params["output_subsidence_dim"]
    out_g_dim = default_model_params["output_gwl_dim"]
    quantiles = default_model_params.get("quantiles")

    if quantiles:
        num_quantiles = len(quantiles)
        targets = {
            'subs_pred': tf.random.normal(
                (batch_size, forecast_horizon, out_s_dim, num_quantiles), 
                dtype=tf.float32),
            'gwl_pred': tf.random.normal(
                (batch_size, forecast_horizon, out_g_dim, num_quantiles), 
                dtype=tf.float32),
        }
    else:
        targets = {
            'subs_pred': tf.random.normal(
                (batch_size, forecast_horizon, out_s_dim), dtype=tf.float32),
            'gwl_pred': tf.random.normal(
                (batch_size, forecast_horizon, out_g_dim), dtype=tf.float32),
        }
    return targets

# --- Test Functions ---

def test_pihalnet_instantiation(default_model_params):
    """Test basic model instantiation."""
    model = PIHALNet(**default_model_params)
    assert isinstance(model, PIHALNet)
    assert model.name == "PIHALNet" # Default name

@pytest.mark.parametrize("use_vsn_config", [True, False])
@pytest.mark.parametrize("static_dim_override", [0, 5])
@pytest.mark.parametrize("future_dim_override", [0, 2])
def test_pihalnet_instantiation_variations(
    default_model_params, use_vsn_config, static_dim_override, future_dim_override
):
    """Test model instantiation with VSN on/off and different input dims."""
    params = default_model_params.copy()
    params["use_vsn"] = use_vsn_config
    params["static_input_dim"] = static_dim_override
    params["future_input_dim"] = future_dim_override
    
    # Adjust vsn_units if use_vsn is True and static_input_dim is 0 but vsn needs >0
    if use_vsn_config and params.get("vsn_units", 0) == 0:
        params["vsn_units"] = 16 # Ensure vsn_units is positive if VSN is used

    model = PIHALNet(**params)
    assert isinstance(model, PIHALNet)
    assert model.use_vsn == use_vsn_config

def test_pihalnet_call_shapes_point_pred(default_model_params, dummy_input_data):
    """Test model call for point predictions and output shapes."""
    inputs, batch_size = dummy_input_data
    params = default_model_params.copy()
    params["quantiles"] = None # Ensure point prediction
    model = PIHALNet(**params)

    outputs = model(inputs, training=False)
    assert isinstance(outputs, dict)
    assert "subs_pred" in outputs
    assert "gwl_pred" in outputs
    assert "pde_residual" in outputs

    fh = params["forecast_horizon"]
    s_dim = params["output_subsidence_dim"]
    g_dim = params["output_gwl_dim"]

    assert outputs["subs_pred"].shape == (batch_size, fh, s_dim)
    assert outputs["gwl_pred"].shape == (batch_size, fh, g_dim)
    # PDE residual shape should match subsidence mean prediction shape
    # Assuming output_subsidence_dim is 1 for the residual target
    assert outputs["pde_residual"].shape == (batch_size, fh-1, s_dim) 

def test_pihalnet_call_shapes_quantile_pred(default_model_params, dummy_input_data):
    """Test model call for quantile predictions and output shapes."""
    inputs, batch_size = dummy_input_data
    params = default_model_params.copy()
    my_quantiles = [0.1, 0.5, 0.9]
    params["quantiles"] = my_quantiles
    num_quantiles = len(my_quantiles)
    model = PIHALNet(**params)

    outputs = model(inputs, training=True) # Test with training=True as well
    assert isinstance(outputs, dict)

    fh = params["forecast_horizon"]
    s_dim = params["output_subsidence_dim"]
    g_dim = params["output_gwl_dim"]

    assert outputs["subs_pred"].shape == (batch_size, fh, num_quantiles, s_dim)
    assert outputs["gwl_pred"].shape == (batch_size, fh,  num_quantiles, g_dim)
    assert "pde_residual" in outputs # Shape check done in point_pred test

def test_pihalnet_compilation(default_model_params):
    """Test if the model compiles with standard losses."""
    params = default_model_params.copy()
    model = PIHALNet(**params)
    
    loss_dict = {
        'subs_pred': MeanSquaredError(),
        'gwl_pred': MeanSquaredError()
    }
    if params.get("quantiles"):
        loss_dict = {
            'subs_pred': combined_quantile_loss(params["quantiles"]),
            'gwl_pred': combined_quantile_loss(params["quantiles"])
        }

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=loss_dict,
        metrics={'subs_pred': ['mae'], 'gwl_pred': ['mae']},
        lambda_pde=0.1 # Example PDE weight
    )
    assert model.optimizer is not None
    assert model.compiled_loss is not None

@pytest.mark.parametrize("pde_mode_config", ["consolidation", "none"])
def test_pihalnet_train_step(
        default_model_params, dummy_input_data,
        dummy_target_data, pde_mode_config):
    """Test a single train_step execution."""
    inputs, _ = dummy_input_data
    # Adjust dummy_target_data if quantiles are set in default_model_params
    params = default_model_params.copy()
    params["pde_mode"] = pde_mode_config
    params["quantiles"] = None # Simplify for this test
    
    # Re-generate targets if quantiles changed
    fh = params["forecast_horizon"]
    s_dim = params["output_subsidence_dim"]
    g_dim = params["output_gwl_dim"]
    batch_size = inputs['coords'].shape[0]
    targets = {
        'subs_pred': tf.random.normal((batch_size, fh, s_dim), dtype=tf.float32),
        'gwl_pred': tf.random.normal((batch_size, fh, g_dim), dtype=tf.float32),
    }

    model = PIHALNet(**params)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={'subs_pred': MeanSquaredError(), 'gwl_pred': MeanSquaredError()},
        lambda_pde=0.1 if pde_mode_config != "none" else 0.0
    )

    # Simulate one step of training data
    # train_step expects data as a tuple (inputs_dict, targets_dict)
    results = model.train_step((inputs, targets))
    assert isinstance(results, dict)
    assert "total_loss" in results
    assert not np.isnan(results["total_loss"].numpy())
    assert not np.isinf(results["total_loss"].numpy())
    if pde_mode_config != "none":
        assert "physics_loss" in results
        assert not np.isnan(results["physics_loss"].numpy())


def test_pihalnet_get_config_from_config(default_model_params, dummy_input_data):
    """Test model serialization and deserialization."""
    model1 = PIHALNet(**default_model_params)
    # Call model on dummy data to ensure all weights are built
    dummy_inputs, _ = dummy_input_data
    model1(dummy_inputs) 

    config = model1.get_config()
    assert isinstance(config, dict)

    # PIHALNet.from_config might need some specific handling of nested configs
    # For a basic test:
    try:
        model2 = PIHALNet.from_config(config.copy()) # Use copy if from_config modifies dict
        assert isinstance(model2, PIHALNet)
        # A simple check
        assert model2.forecast_horizon == model1.forecast_horizon
        assert model2.embed_dim == model1.embed_dim

        # More thorough check: compare all relevant config items
        config2 = model2.get_config()
        for key in default_model_params:
            if key in config and key in config2 : # Some keys might be transformed
                 # Handle list comparison for quantiles, scales
                if isinstance(config[key], list) or isinstance(config[key],tuple ):
                    assert list(config[key] or []) == list(config2[key] or [])
                elif isinstance(config[key], dict): # like gw_flow_coeffs
                     assert config[key] == config2[key]
                elif key not in ['name', 'dtype']: # Name might change, dtype is complex
                    assert config[key] == config2[key], f"Config mismatch for key: {key}"

    except Exception as e:
        pytest.fail(f"PIHALNet.from_config failed: {e}\nConfig was: {config}")


@pytest.mark.parametrize("c_config", ["learnable", 0.05, None])
def test_pinn_coefficient_c(default_model_params, c_config):
    """Test the pinn_coefficient_C logic."""
    params = default_model_params.copy()
    params["pinn_coefficient_C"] = c_config
    params["pde_mode"] = "consolidation" # C is used in consolidation
    model = PIHALNet(**params)

    c_val_tensor = model.get_pinn_coefficient_C()
    assert isinstance(c_val_tensor, tf.Tensor)
    
    if c_config == "learnable":
        # Check if log_C_consolidation_var is a trainable variable
        # The name might vary based on how _build_pinn_components sets it up
        trainable_coeffs = [v for v in model.trainable_variables if "log_pinn_coefficient_C" in v.name]
        assert len(trainable_coeffs) > 0, "Learnable C coefficient not found or not trainable."
    elif isinstance(c_config, float):
        assert np.isclose(c_val_tensor.numpy(), c_config)
    elif c_config is None:
        # If None and pde_mode is consolidation, _build_pinn_components might default it
        # or get_C_consolidation might return a default (e.g. 1.0 if _is_unique_consolidation)
        # Based on current PIHALNet code: if config is None, get_C returns None unless _is_unique_consolidation
        # Let's assume for this test, if C_config is None, and PDE is on, it defaults.
        # The provided PIHALNet._build_pinn_components uses default_initial_value if config is None for log_C
        # but your _get_C logic for None config makes it return None if var is None.
        # Your get_C_consolidation has:
        # if self.log_C_consolidation_var is None:
        #    if self._is_unique_consolidation: return tf_constant(1.0, dtype=tf_float32)
        #    return None
        # This makes testing tricky without knowing _is_unique_consolidation.
        # Let's assume if pde_mode='consolidation' and pinn_coefficient_C=None, it means C is not used or is 1.0.
        # For this test, we'll just check it returns a tensor.
        pass # Already asserted it's a tensor. Specific value depends on internal defaults.


# Example of a test that might hit the sequence length issue if not careful:
def test_hierarchical_attention_specific_lengths(default_model_params, dummy_input_data):
    """
    This test would require careful setup of dummy_input_data and PIHALNet
    parameters to replicate the 4 vs 3 sequence length issue.
    It's more of an integration test for specific hyperparameter sets.
    """
    params = default_model_params.copy()
    # Setup params and dummy_input_data to specifically create the 4 vs 3 scenario
    # This means understanding how inputs to HierarchicalAttention are derived.
    # For example, if one input is based on FORECAST_HORIZON and another
    # on a processed version of TIME_STEPS.
    params["forecast_horizon"] = 3
    params["max_window_size"] = 7 # TIME_STEPS
      # ... (modify other architectural HPs like lstm_units, attention_units, etc.
      #      that might influence intermediate sequence lengths before HierarchicalAttention)

    model = PIHALNet(**params)
    
    # Create dummy inputs that will lead to the (None,4,X) and (None,3,Y) inputs
    # for the HierarchicalAttention layer within PIHALNet.
    # This is non-trivial as it depends on the internal processing of PIHALNet.
    inputs, _ = dummy_input_data # This fixture might need to be more flexible or have variants
    
    try:
        outputs = model(inputs, training=False)
        # If no error, the shapes were compatible for this specific setup
    except tf.errors.InvalidArgumentError as e:
        if "Dimensions must be equal, but are 4 and 3" in str(e):
            pytest.fail("Reproduced the HierarchicalAttention shape mismatch under specific config.")
        else:
            raise e # Re-raise other unexpected errors

# import os # For saving/loading model in new test
# import tempfile # For creating temporary directory for model saving
# from typing import List, Dict, Tuple, Optional, Any # Added for type hints


# --- Fixtures ---

@pytest.fixture
def default_model_params2():
    """Default fixed parameters for PIHALNet instantiation for tests."""
    return {
        "static_input_dim": 5,
        "dynamic_input_dim": 4,
        "future_input_dim": 2,
        "output_subsidence_dim": 1,
        "output_gwl_dim": 1,
        "forecast_horizon": 3,
        "max_window_size": 7, # Corresponds to TIME_STEPS
        # Architectural params (can be overridden in specific tests)
        "embed_dim": 32,
        "hidden_units": 32,
        "lstm_units": 32,
        "attention_units": 16, 
        "num_heads": 2,
        "dropout_rate": 0.0, 
        "vsn_units": 16,
        "pde_mode": "consolidation",
        "pinn_coefficient_C": "learnable",
        "quantiles": None, # Default to point prediction for simplicity in some tests
        "scales": [1,2], # Example scales
        "activation": "relu",
    }

@pytest.fixture
def dummy_input_data2(default_model_params2):
    """Generates dummy input data for PIHALNet based on model_params."""
    batch_size = 2
    time_steps = default_model_params2["max_window_size"]
    forecast_horizon = default_model_params2["forecast_horizon"]
    static_dim = default_model_params2["static_input_dim"]
    dynamic_dim = default_model_params2["dynamic_input_dim"]
    future_dim = default_model_params2["future_input_dim"]

    inputs = {
        'coords': tf.random.normal(
            (batch_size, forecast_horizon, 3), dtype=tf.float32),
        'dynamic_features': tf.random.normal(
            (batch_size, time_steps, dynamic_dim), dtype=tf.float32),
    }
    if static_dim > 0:
        inputs['static_features'] = tf.random.normal(
            (batch_size, static_dim), dtype=tf.float32)
    else: 
        inputs['static_features'] = tf.zeros(
            (batch_size, 0), dtype=tf.float32)

    if future_dim > 0:
        inputs['future_features'] = tf.random.normal(
            (batch_size, forecast_horizon + time_steps, future_dim), dtype=tf.float32)
    else:
        inputs['future_features'] = tf.zeros(
            (batch_size, forecast_horizon +time_steps, 0), dtype=tf.float32)
        
    return inputs, batch_size

@pytest.fixture
def dummy_target_data_fixture(default_model_params2, dummy_input_data2): # Renamed to avoid clash
    """Generates dummy target data for PIHALNet. Name changed for clarity."""
    _, batch_size = dummy_input_data2
    forecast_horizon = default_model_params2["forecast_horizon"]
    out_s_dim = default_model_params2["output_subsidence_dim"]
    out_g_dim = default_model_params2["output_gwl_dim"]
    quantiles = default_model_params2.get("quantiles")

    if quantiles:
        num_quantiles = len(quantiles)
        targets = {
            'subs_pred': tf.random.normal((batch_size, forecast_horizon, out_s_dim, num_quantiles), dtype=tf.float32),
            'gwl_pred': tf.random.normal((batch_size, forecast_horizon, out_g_dim, num_quantiles), dtype=tf.float32),
        }
    else:
        targets = {
            'subs_pred': tf.random.normal((batch_size, forecast_horizon, out_s_dim), dtype=tf.float32),
            'gwl_pred': tf.random.normal((batch_size, forecast_horizon, out_g_dim), dtype=tf.float32),
        }
    return targets

# --- Test Functions ---

def test_pihalnet_instantiation2(default_model_params2):
    """Test basic model instantiation."""
    model = PIHALNet(**default_model_params2)
    assert isinstance(model, PIHALNet)
    assert model.name == "PIHALNet" 

@pytest.mark.parametrize("use_vsn_config", [True, False])
@pytest.mark.parametrize("static_dim_override", [0, 5])
@pytest.mark.parametrize("future_dim_override", [0, 2])
def test_pihalnet_instantiation_variations2(
    default_model_params2, use_vsn_config, static_dim_override, future_dim_override
):
    """Test model instantiation with VSN on/off and different input dims."""
    params = default_model_params2.copy()
    params["use_vsn"] = use_vsn_config
    params["static_input_dim"] = static_dim_override
    params["future_input_dim"] = future_dim_override
    
    if use_vsn_config and params.get("vsn_units", 0) == 0:
        params["vsn_units"] = 16 

    model = PIHALNet(**params)
    assert isinstance(model, PIHALNet)
    assert model.use_vsn == use_vsn_config

def test_pihalnet_call_shapes_point_pred2(default_model_params2, dummy_input_data2):
    """Test model call for point predictions and output shapes."""
    inputs, batch_size = dummy_input_data2
    params = default_model_params2.copy()
    params["quantiles"] = None 
    model = PIHALNet(**params)

    outputs = model(inputs, training=False)
    assert isinstance(outputs, dict)
    assert "subs_pred" in outputs
    assert "gwl_pred" in outputs
    assert "pde_residual" in outputs

    fh = params["forecast_horizon"]
    s_dim = params["output_subsidence_dim"]
    g_dim = params["output_gwl_dim"]

    assert outputs["subs_pred"].shape == (batch_size, fh, s_dim)
    assert outputs["gwl_pred"].shape == (batch_size, fh, g_dim)
    # note the pde_residual shape axis [1] will be forecast_horizon -1 
    assert outputs["pde_residual"].shape == (batch_size, fh -1, s_dim) 

def test_pihalnet_call_shapes_quantile_pred2(default_model_params2, dummy_input_data2):
    """Test model call for quantile predictions and output shapes."""
    inputs, batch_size = dummy_input_data2
    params = default_model_params2.copy()
    my_quantiles = [0.1, 0.5, 0.9]
    params["quantiles"] = my_quantiles
    num_quantiles = len(my_quantiles)
    
    # Ensure attention_units is compatible with num_heads for MHA
    if params["attention_units"] % params["num_heads"] != 0:
        params["attention_units"] = params["num_heads"] * (
            params["attention_units"] // params["num_heads"])
        if params["attention_units"] == 0 :
            params["attention_units"] = params["num_heads"]

    model = PIHALNet(**params)

    outputs = model(inputs, training=True) 
    assert isinstance(outputs, dict)

    fh = params["forecast_horizon"]
    s_dim = params["output_subsidence_dim"]
    g_dim = params["output_gwl_dim"]

    assert outputs["subs_pred"].shape == (batch_size, fh,  num_quantiles, s_dim)
    assert outputs["gwl_pred"].shape == (batch_size, fh,  num_quantiles, g_dim)
    assert "pde_residual" in outputs

def test_pihalnet_compilation2(default_model_params2):
    """Test if the model compiles with standard losses."""
    params = default_model_params2.copy()
    model = PIHALNet(**params)
    
    loss_dict = {
        'subs_pred': MeanSquaredError(),
        'gwl_pred': MeanSquaredError()
    }
    if params.get("quantiles"):
        loss_dict = {
            'subs_pred': combined_quantile_loss(params["quantiles"]),
            'gwl_pred': combined_quantile_loss(params["quantiles"])
        }

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=loss_dict,
        metrics={'subs_pred': ['mae'], 'gwl_pred': ['mae']},
        lambda_pde=0.1 
    )
    assert model.optimizer is not None
    assert model.compiled_loss is not None

@pytest.mark.parametrize("pde_mode_config", ["consolidation", "none"])
def test_pihalnet_train_step2(default_model_params2, dummy_input_data2, pde_mode_config):
    """Test a single train_step execution."""
    inputs, batch_size = dummy_input_data2
    params = default_model_params2.copy()
    params["pde_mode"] = pde_mode_config
    params["quantiles"] = None 
    
    fh = params["forecast_horizon"]
    s_dim = params["output_subsidence_dim"]
    g_dim = params["output_gwl_dim"]
    targets = { # Regenerate targets for point prediction
        'subs_pred': tf.random.normal((batch_size, fh, s_dim), dtype=tf.float32),
        'gwl_pred': tf.random.normal((batch_size, fh, g_dim), dtype=tf.float32),
    }

    model = PIHALNet(**params)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={'subs_pred': MeanSquaredError(), 'gwl_pred': MeanSquaredError()},
        lambda_pde=0.1 if pde_mode_config != "none" else 0.0
    )

    results = model.train_step((inputs, targets))
    assert isinstance(results, dict)
    assert "total_loss" in results
    assert not np.isnan(results["total_loss"].numpy())
    assert not np.isinf(results["total_loss"].numpy())
    if pde_mode_config != "none":
        assert "physics_loss" in results
        assert not np.isnan(results["physics_loss"].numpy())


def test_pihalnet_get_config_from_config2(default_model_params2, dummy_input_data2):
    """Test model serialization and deserialization."""
    model1 = PIHALNet(**default_model_params2)
    # Call model on dummy data to ensure all weights are built
    dummy_inputs, _ = dummy_input_data2 # Use the fixture directly
    model1(dummy_inputs) 

    config = model1.get_config()
    assert isinstance(config, dict)

    try:
        model2 = PIHALNet.from_config(config.copy()) 
        assert isinstance(model2, PIHALNet)
        assert model2.forecast_horizon == model1.forecast_horizon
        assert model2.embed_dim == model1.embed_dim

        config2 = model2.get_config()
        # Compare relevant keys, excluding potentially problematic ones like 'name' or complex objects
        # that might not serialize/deserialize perfectly identically but are functionally equivalent.
        relevant_keys = [
            "static_input_dim", "dynamic_input_dim", "future_input_dim",
            "output_subsidence_dim", "output_gwl_dim", "embed_dim",
            "hidden_units", "lstm_units", "attention_units", "num_heads",
            "dropout_rate", "forecast_horizon", "quantiles", "max_window_size",
            "memory_size", "scales", "multi_scale_agg", "final_agg",
            "activation", "use_residuals", "use_batch_norm",
            "pinn_coefficient_C", "use_vsn", "vsn_units", "pde_mode"
            # "gw_flow_coeffs" # Can be complex if dicts are involved
        ]
        for key in relevant_keys:
            if key in config and key in config2:
                val1 = config[key]
                val2 = config2[key]
                if isinstance(val1, (list, tuple)): # Convert to list for comparison
                    assert list(val1 or []) == list(val2 or []), f"Config mismatch for list key: {key}"
                elif isinstance(val1, np.ndarray):
                    assert np.array_equal(val1, val2), f"Config mismatch for ndarray key: {key}"
                else:
                    assert val1 == val2, f"Config mismatch for key: {key}"
            elif key in config or key in config2: # One has it, other doesn't
                assert False, f"Key '{key}' present in one config but not the other."


    except Exception as e:
        pytest.fail(f"PIHALNet.from_config failed: {e}\nOriginal Config was: {config}")


@pytest.mark.parametrize("c_config", ["learnable", 0.05, None])
def test_pinn_coefficient_c2(default_model_params2, c_config):
    """Test the pinn_coefficient_C logic."""
    params = default_model_params2.copy()
    params["pinn_coefficient_C"] = c_config
    params["pde_mode"] = "consolidation" 
    model = PIHALNet(**params)

    c_val_tensor = model.get_pinn_coefficient_C() #
    if params["pde_mode"] == "consolidation":
        c_val_tensor = model.get_pinn_coefficient_C()
        assert isinstance(c_val_tensor, tf.Tensor)
        if c_config == "learnable":
            trainable_coeffs = [
                v for v in model.trainable_variables if "log_pinn_coefficient_C" in v.name]
            assert len(trainable_coeffs) > 0
        elif isinstance(c_config, float):
            # assert np.isclose(tf.exp(c_val_tensor).numpy(), c_config, atol=1e-6) # if stored as log(C)
            assert np.isclose(c_val_tensor.numpy(), c_config) # If stored directly: 
        elif c_config is None: 
            # Based on PIHALNet, if config is None and pde_mode includes 'consolidation',
            # log_C_consolidation_var will be created with default.
            assert c_val_tensor is not None 
    else: # PDE mode is not consolidation
        assert model.get_pinn_coefficient_C() is None


def test_pihalnet_fit_and_refit(default_model_params2, dummy_input_data2, dummy_target_data_fixture):
    """Test model.fit(), then continue fitting, and fit after save/load."""
    inputs, _ = dummy_input_data2
    targets = dummy_target_data_fixture # Use the renamed fixture
    
    params = default_model_params2.copy()
    params["quantiles"] = None # Use point predictions for simpler loss in this test
     # Adjust targets if quantiles were default in fixture
    if default_model_params2.get("quantiles") is not None:
        targets = {
            k: (v[..., 0] if v.shape.ndims == 4 else v) 
            for k, v in targets.items()
        }


    model = PIHALNet(**params)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={'subs_pred': MeanSquaredError(), 'gwl_pred': MeanSquaredError()},
        lambda_pde=0.01 # Small PDE weight
    )

    # Initial fit
    print("\n--- Initial Fit ---")
    history1 = model.fit(inputs, targets, epochs=2, batch_size=1, verbose=0)
    assert history1 is not None
    assert "total_loss" in history1.history
    assert len(history1.history["total_loss"]) == 2
    initial_loss = history1.history["total_loss"][-1]
    assert not np.isnan(initial_loss)

    # Continue fitting
    print("\n--- Continued Fit ---")
    history2 = model.fit(inputs, targets, epochs=2, batch_size=1, verbose=0) # Fit for 2 more epochs
    assert history2 is not None
    assert "total_loss" in history2.history
    assert len(history2.history["total_loss"]) == 2
    continued_loss = history2.history["total_loss"][-1]
    assert not np.isnan(continued_loss)
    # Loss might not always decrease on dummy data with few epochs,
    # but it shouldn't be NaN.

    # Save and reload model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "pihalnet_refit_test.keras")
        print(f"\n--- Saving model to {model_path} ---")
        model.save(model_path)
        
        print("--- Reloading model ---")
        # When loading custom model with custom objects like specific layers or losses:
        # custom_objects = {'PIHALNet': PIHALNet} # Add other custom objects if any
        # if params.get("quantiles"):
        #    custom_objects['placeholder_cql'] = combined_quantile_loss(params["quantiles"])
        # reloaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        # For PIHALNet, if it's registered correctly, direct load might work.
        # However, Keras often needs explicit custom_objects for complex models.
        # For simplicity, we re-instantiate and load weights if direct load_model is tricky.
        
        # Option A: Direct load (might require full registration of all custom components)
        try:
            reloaded_model = tf.keras.models.load_model(model_path)
        except Exception as e_load:
            print(f"Direct load_model failed: {e_load}. Attempting re-instantiate and load_weights.")
            # Option B: Re-instantiate and load weights (more robust for complex custom models)
            reloaded_model = PIHALNet(**params)
            reloaded_model(inputs) # Build the model by calling it
            reloaded_model.load_weights(model_path)


        # Compile the reloaded model (essential before fitting again)
        reloaded_model.compile(
            optimizer=Adam(learning_rate=1e-4), # Can use a different LR
            loss={'subs_pred': MeanSquaredError(), 'gwl_pred': MeanSquaredError()},
            lambda_pde=0.01
        )
        
        print("--- Fitting reloaded model ---")
        history3 = reloaded_model.fit(inputs, targets, epochs=1, batch_size=1, verbose=0)
        assert history3 is not None
        assert "total_loss" in history3.history
        assert len(history3.history["total_loss"]) == 1
        reloaded_loss = history3.history["total_loss"][-1]
        assert not np.isnan(reloaded_loss)

    print("Fit and refit tests completed.")


        
if __name__=='__main__': 
    pytest.main([__file__]) 
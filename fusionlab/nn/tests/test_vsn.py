# -*- coding: utf-8 -*-
# test_variable_selection_network.py
# (Place in your tests directory, e.g., fusionlab/nn/tests/)

import pytest
import numpy as np
import tensorflow as tf

# --- Attempt to import VSN and dependencies ---
try:
    from fusionlab.nn.components import (
        VariableSelectionNetwork,
        GatedResidualNetwork # Needed dependency for VSN
        )
    # Import loss if testing training
    from fusionlab.nn.losses import combined_quantile_loss
    # Check if TF backend is available
    from fusionlab.nn import KERAS_BACKEND
except ImportError as e:
    print(f"Skipping VSN tests due to import error: {e}")
    KERAS_BACKEND = False
# --- End Imports ---

# Skip all tests in this file if TensorFlow/Keras backend is not available
pytestmark = pytest.mark.skipif(
    not KERAS_BACKEND, reason="TensorFlow/Keras backend not available"
)

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def vsn_config():
    """Provides default configuration for VSN."""
    return {
        "num_inputs": 5,  # Number of distinct input variables (N)
        "units": 8,       # Output embedding dimension per variable
        "dropout_rate": 0.0,
        "activation": 'relu',
        "use_batch_norm": False,
    }

@pytest.fixture(scope="module")
def vsn_dummy_data(vsn_config):
    """Provides dummy data for VSN tests."""
    B, T = 4, 10       # Batch size, Time steps
    N = vsn_config["num_inputs"]
    F = 1              # Feature dimension per variable (usually 1)
    U = vsn_config["units"] # Units for context matches GRN units

    # Non-Time-Distributed Input (Batch, NumVars, Features)
    X_static_3d = np.random.rand(B, N, F).astype(np.float32)
    # Non-Time-Distributed Input (Batch, NumVars) - should be expanded
    X_static_2d = np.random.rand(B, N).astype(np.float32)

    # Time-Distributed Input (Batch, Time, NumVars, Features)
    X_dynamic_4d = np.random.rand(B, T, N, F).astype(np.float32)
    # Time-Distributed Input (Batch, Time, NumVars) - should be expanded
    X_dynamic_3d = np.random.rand(B, T, N).astype(np.float32)

    # Context (must be projectable to 'units')
    context = np.random.rand(B, U).astype(np.float32) # Example static context

    return {
        "B": B, "T": T, "N": N, "F": F, "U": U,
        "X_static_3d": X_static_3d,
        "X_static_2d": X_static_2d,
        "X_dynamic_4d": X_dynamic_4d,
        "X_dynamic_3d": X_dynamic_3d,
        "context": context
    }

# --- Test Functions ---

def test_vsn_instantiation(vsn_config):
    """Test basic VSN instantiation."""
    try:
        vsn = VariableSelectionNetwork(**vsn_config)
        assert isinstance(vsn, tf.keras.layers.Layer)
        assert len(vsn.single_variable_grns) == vsn_config["num_inputs"]
    except Exception as e:
        pytest.fail(f"VSN instantiation failed. Error: {e}")
    print("VSN Instantiation OK")

@pytest.mark.parametrize(
    "use_time_distributed, input_tensor_key, use_context",
    [
        (False, "X_static_3d", False), # Static, 3D input, no context
        (False, "X_static_2d", False), # Static, 2D input, no context
        (False, "X_static_3d", True),  # Static, 3D input, with context
        (False, "X_static_2d", True),  # Static, 2D input, with context
        (True, "X_dynamic_4d", False), # Dynamic, 4D input, no context
        (True, "X_dynamic_3d", False), # Dynamic, 3D input, no context
        (True, "X_dynamic_4d", True),  # Dynamic, 4D input, with context
        (True, "X_dynamic_3d", True),  # Dynamic, 3D input, with context
    ]
)
def test_vsn_call_and_output_shape(
    vsn_config, vsn_dummy_data, use_time_distributed, input_tensor_key, use_context
    ):
    """Test VSN call with different modes and inputs."""
    config = vsn_config.copy()
    config["use_time_distributed"] = use_time_distributed
    vsn = VariableSelectionNetwork(**config)

    input_tensor = vsn_dummy_data[input_tensor_key]
    context_tensor = vsn_dummy_data["context"] if use_context else None
    B = vsn_dummy_data["B"]
    T = vsn_dummy_data["T"]
    U = vsn_config["units"]

    # Perform forward pass
    try:
        # Build layer explicitly or let first call handle it
        # vsn.build(input_tensor.shape) # Optional explicit build
        outputs = vsn(input_tensor, context=context_tensor, training=False)
    except Exception as e:
        pytest.fail(f"VSN call failed for mode={use_time_distributed}, "
                    f"input_key={input_tensor_key}, context={use_context}."
                    f" Error: {e}")

    # Check output shape
    if use_time_distributed:
        expected_shape = (B, T, U)
    else:
        expected_shape = (B, U)
    assert outputs.shape == expected_shape, \
        f"Output shape mismatch. Expected {expected_shape}, got {outputs.shape}"

    # Check variable importances
    assert hasattr(vsn, 'variable_importances_')
    importances = vsn.variable_importances_
    assert importances is not None
    N = vsn_config["num_inputs"]
    if use_time_distributed:
        expected_importance_shape = (B, T, N)
    else:
        expected_importance_shape = (B, N)
    assert importances.shape == expected_importance_shape, \
        f"Importance shape mismatch. Expected {expected_importance_shape}, " \
        f"got {importances.shape}"
    # Check that weights sum to 1 (approximately) across variables
    sum_axis = -1 # The last axis holds the weights per variable
    sums = tf.reduce_sum(importances, axis=sum_axis).numpy()
    assert np.allclose(sums, 1.0), \
        f"Importance weights do not sum to 1. Sums: {sums}"

    print(f"VSN Call OK: TD={use_time_distributed}, "
          f"InputKey={input_tensor_key}, Context={use_context}, "
          f"Output Shape={outputs.shape}")


@pytest.mark.parametrize("use_time_distributed", [False, True])
def test_vsn_minimal_train_step(vsn_config, vsn_dummy_data, use_time_distributed):
    """Test if VSN works within a minimal trainable model."""
    config = vsn_config.copy()
    config["use_time_distributed"] = use_time_distributed
    vsn = VariableSelectionNetwork(**config)

    if use_time_distributed:
        input_tensor = vsn_dummy_data["X_dynamic_4d"]
        # Dummy target shape: (B, T, U) matching VSN output
        dummy_target = np.random.rand(
            vsn_dummy_data["B"], vsn_dummy_data["T"], vsn_config["units"]
            ).astype(np.float32)
    else:
        input_tensor = vsn_dummy_data["X_static_3d"]
        # Dummy target shape: (B, U) matching VSN output
        dummy_target = np.random.rand(
            vsn_dummy_data["B"], vsn_config["units"]
            ).astype(np.float32)

    context_tensor = vsn_dummy_data["context"]

    # Create a simple model wrapping the VSN
    class TestModel(tf.keras.Model):
        def __init__(self, vsn_layer):
            super().__init__()
            self.vsn = vsn_layer
            # Add a dummy dense layer just to ensure output is used
            self.dense = tf.keras.layers.Dense(vsn_layer.units)

        def call(self, inputs, training=False):
            # Assumes inputs = [main_input, context] or [main_input]
            if isinstance(inputs, (list, tuple)):
                main_input, context = inputs[0], inputs[1]
            else: # Only main input provided
                 main_input = inputs
                 context=None
            vsn_output = self.vsn(main_input, context=context, training=training)
            # Pass through another layer to ensure gradient computation
            final_output = self.dense(vsn_output)
            return final_output

    test_model = TestModel(vsn)

    # Prepare inputs for the wrapper model
    model_inputs = [input_tensor, context_tensor]

    try:
        test_model.compile(optimizer='adam', loss='mse')
        history = test_model.fit(
            model_inputs, dummy_target, epochs=1, batch_size=2, verbose=0
            )
    except Exception as e:
        pytest.fail(f"VSN minimal train step failed (TD={use_time_distributed})."
                    f" Error: {e}")

    assert history is not None
    assert 'loss' in history.history
    print(f"VSN Minimal Train Step OK: TD={use_time_distributed}")


def test_vsn_serialization(vsn_config, vsn_dummy_data):
    """Test VSN get_config and from_config."""
    config = vsn_config.copy()
    config["use_time_distributed"] = True # Example config
    vsn = VariableSelectionNetwork(**config)

    # Build the layer first (needed for get_config to be complete)
    input_tensor = vsn_dummy_data["X_dynamic_4d"]
    context_tensor = vsn_dummy_data["context"]
    _ = vsn(input_tensor, context=context_tensor) # Call once to build

    try:
        retrieved_config = vsn.get_config()
        # Check a few key parameters were saved
        assert retrieved_config['num_inputs'] == config['num_inputs']
        assert retrieved_config['units'] == config['units']
        assert retrieved_config['activation'] == config['activation']
        assert retrieved_config['use_time_distributed'] == config['use_time_distributed']

        rebuilt_vsn = VariableSelectionNetwork.from_config(retrieved_config)
    except Exception as e:
        pytest.fail(f"VSN serialization/deserialization failed. Error: {e}")

    assert isinstance(rebuilt_vsn, VariableSelectionNetwork)
    # Check output shape consistency
    try:
        output_original = vsn(input_tensor, context=context_tensor)
        # Rebuild the loaded layer by calling it
        output_rebuilt = rebuilt_vsn(input_tensor, context=context_tensor)
        assert output_original.shape == output_rebuilt.shape
    except Exception as e:
         pytest.fail(f"Output shape mismatch after from_config. Error: {e}")

    print("VSN Serialization OK")
    
if __name__=='__main__': 
    pytest.main([__file__])
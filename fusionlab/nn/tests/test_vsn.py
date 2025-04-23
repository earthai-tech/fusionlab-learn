# test_variable_selection_network.py

import pytest
import numpy as np

# --- Attempt to import VSN and dependencies ---
try:
    import tensorflow as tf
    from fusionlab.nn.components import (
        VariableSelectionNetwork,
        GatedResidualNetwork # VSN depends on GRN: Refesh fixed.
        )
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
        "units": 8,       # Output embedding dimension
        "dropout_rate": 0.0,
        "activation": 'relu', # Activation for internal GRNs
        "use_batch_norm": False,
    }

@pytest.fixture(scope="module")
def vsn_dummy_data(vsn_config):
    """Provides dummy data for VSN tests."""
    B, T = 4, 10       # Batch size, Time steps
    N = vsn_config["num_inputs"]
    F = 1              # Feature dimension per variable (VSN adds this if needed)
    U = vsn_config["units"] # Units for context should match GRN units

    # Non-Time-Distributed Input (Batch, NumVars, Features)
    X_static_3d = np.random.rand(B, N, F).astype(np.float32)
    # Non-Time-Distributed Input (Batch, NumVars) - VSN should expand
    X_static_2d = np.random.rand(B, N).astype(np.float32)

    # Time-Distributed Input (Batch, Time, NumVars, Features)
    X_dynamic_4d = np.random.rand(B, T, N, F).astype(np.float32)
    # Time-Distributed Input (Batch, Time, NumVars) - VSN should expand
    X_dynamic_3d = np.random.rand(B, T, N).astype(np.float32)

    # Context (Batch, Units) - Example static context, pre-projected
    context = np.random.rand(B, U).astype(np.float32)

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
        # Check internal layers are created
        assert len(vsn.single_variable_grns) == vsn_config["num_inputs"]
        assert isinstance(vsn.variable_importance_dense, tf.keras.layers.Dense)
        assert isinstance(vsn.softmax, tf.keras.layers.Layer) # Softmax or TD(Softmax)
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
    """Test VSN call with different modes, inputs, and context."""
    config = vsn_config.copy()
    config["use_time_distributed"] = use_time_distributed
    vsn = VariableSelectionNetwork(**config)

    input_tensor = vsn_dummy_data[input_tensor_key]
    context_tensor = vsn_dummy_data["context"] if use_context else None
    B = vsn_dummy_data["B"]
    T = vsn_dummy_data["T"]
    U = vsn_config["units"] # Output dimension is always units

    # Perform forward pass
    try:
        # Build layer explicitly or let first call handle it
        # vsn.build(input_tensor.shape) # Optional explicit build
        outputs = vsn(input_tensor, context=context_tensor, training=False)
    except Exception as e:
        pytest.fail(
            f"VSN call failed for TD={use_time_distributed}, "
            f"InputKey={input_tensor_key}, Context={use_context}."
            f"\nInput Shape: {input_tensor.shape}, "
            f"Context Shape: {context_tensor.shape if context_tensor is not None else None}"
            f"\nError: {e}"
            )

    # Check output shape
    if use_time_distributed:
        expected_shape = (B, T, U)
    else:
        expected_shape = (B, U)
    assert outputs.shape == expected_shape, \
        f"Output shape mismatch. Expected {expected_shape}, got {outputs.shape}"

    # Check variable importances were stored and have correct shape
    assert hasattr(vsn, 'variable_importances_')
    importances = vsn.variable_importances_
    assert importances is not None
    N = vsn_config["num_inputs"]
    # Importance weights shape should be (B, [T,] N, 1) before softmax axis reduction
    # The softmax layer used (axis=-2) results in (B, [T,] N, 1)
    # Let's check this shape directly
    if use_time_distributed:
        expected_importance_shape = (B, T, N, 1)
    else:
        expected_importance_shape = (B, N, 1)
    assert importances.shape == expected_importance_shape, \
        (f"Importance shape mismatch. Expected {expected_importance_shape}, "
          f"got {importances.shape}")

    # Check that weights sum to 1 (approximately) across variables (axis=-2)
    sum_axis = -2 # The 'N' dimension where softmax was applied
    sums = tf.reduce_sum(importances, axis=sum_axis).numpy()
    # After summing across N, shape is (B, [T,] 1), sums should be close to 1
    assert np.allclose(sums, 1.0), \
        f"Importance weights do not sum to 1 along axis {sum_axis}. Sums: {sums}"

    print(f"VSN Call OK: TD={use_time_distributed}, "
          f"InputKey={input_tensor_key}, Context={use_context}, "
          f"Output Shape={outputs.shape}")


@pytest.mark.parametrize("use_time_distributed", [False, True])
@pytest.mark.parametrize("use_context", [False, True])
def test_vsn_minimal_train_step(
    vsn_config, vsn_dummy_data, use_time_distributed, use_context
    ):
    """Test if VSN works within a minimal trainable model."""
    config = vsn_config.copy()
    config["use_time_distributed"] = use_time_distributed
    vsn = VariableSelectionNetwork(**config)

    # Select appropriate input based on TD flag
    if use_time_distributed:
        input_tensor = vsn_dummy_data["X_dynamic_4d"] # Use 4D input
        dummy_target = np.random.rand(
            vsn_dummy_data["B"], vsn_dummy_data["T"], vsn_config["units"]
            ).astype(np.float32)
    else:
        input_tensor = vsn_dummy_data["X_static_3d"] # Use 3D input
        dummy_target = np.random.rand(
            vsn_dummy_data["B"], vsn_config["units"]
            ).astype(np.float32)

    context_tensor = vsn_dummy_data["context"] if use_context else None

    # --- FIX: Explicitly build the VSN layer ---
    # This will also trigger the build methods of internal GRNs
    try:
         vsn.build(input_tensor.shape)
         # Also build context projection if context is used
         if use_context:
              # Check if context_projection exists (created lazily in VSN build)
              if vsn.context_projection and not vsn.context_projection.built:
                   vsn.context_projection.build(context_tensor.shape)
         print(
             f"VSN built explicitly for TD={use_time_distributed},"
             f" Context={use_context}")
    except Exception as e:
         pytest.fail(f"Explicit VSN build failed. Error: {e}")
    # --- END FIX ---

    # Create a simple model wrapping the VSN
    class TestModel(tf.keras.Model):
         def __init__(self, vsn_layer):
            super().__init__()
            self.vsn = vsn_layer
            self.dense = tf.keras.layers.Dense(vsn_layer.units)

         def call(self, inputs, training=False):
            if isinstance(inputs, (list, tuple)):
                 main_input, context_input = inputs
            else:
                 main_input, context_input = inputs, None
            vsn_output = self.vsn(
                main_input, context=context_input, training=training
                )
            return self.dense(vsn_output)

    test_model = TestModel(vsn)
    model_inputs = [
        input_tensor, context_tensor] if use_context else input_tensor

    try:
        test_model.compile(optimizer='adam', loss='mse')
        history = test_model.fit(
            model_inputs, dummy_target, epochs=1, batch_size=2, verbose=0
            )
    except Exception as e:
        pytest.fail(f"VSN minimal train step failed (TD={use_time_distributed}, "
                    f"Context={use_context}). Error: {e}")

    assert history is not None
    assert 'loss' in history.history
    print(f"VSN Minimal Train Step OK: TD={use_time_distributed}, "
          f"Context={use_context}")


def test_vsn_serialization(vsn_config, vsn_dummy_data):
    """Test VSN get_config and from_config."""
    config = vsn_config.copy()
    config["use_time_distributed"] = True # Test with non-default
    config["activation"] = 'gelu'      # Test with non-default

    vsn = VariableSelectionNetwork(**config)

    # Build the layer first
    input_tensor = vsn_dummy_data["X_dynamic_4d"]
    context_tensor = vsn_dummy_data["context"]
    _ = vsn(input_tensor, context=context_tensor) # Call once to build

    try:
        retrieved_config = vsn.get_config()
        # Check key parameters saved correctly
        assert retrieved_config['num_inputs'] == config['num_inputs']
        assert retrieved_config['units'] == config['units']
        assert retrieved_config['activation'] == config['activation'] # String
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

# Allows running the tests directly if needed
if __name__=='__main__':
     pytest.main([__file__])
# test_gated_residual_network.py (Revised)

import pytest
import numpy as np
import tensorflow as tf

# --- Attempt to import GRN and dependencies ---
try:
    from fusionlab.nn.components import GatedResidualNetwork
    # Activation layer no longer needed for GRN internal use test
    from fusionlab.nn import KERAS_BACKEND
except ImportError as e:
    print(f"Skipping GRN tests due to import error: {e}")
    KERAS_BACKEND = False
# --- End Imports ---

pytestmark = pytest.mark.skipif(
    not KERAS_BACKEND, reason="TensorFlow/Keras backend not available"
)

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def grn_config():
    """Provides default configuration for GRN."""
    return {
        "units": 16,
        "dropout_rate": 0.0,
        "activation": 'relu', # Test with a standard activation
        "use_batch_norm": False,
        "output_activation": None,
    }

@pytest.fixture(scope="module")
def dummy_grn_data(grn_config):
    """Provides dummy data for GRN tests."""
    B, T, F_in = 4, 10, 8 # Batch, TimeSteps, InputFeatures
    U = grn_config["units"]

    x_non_td = np.random.rand(B, F_in).astype(np.float32)
    x_td = np.random.rand(B, T, F_in).astype(np.float32)
    context = np.random.rand(B, U).astype(np.float32) # Context matches units

    return {
        "B": B, "T": T, "F_in": F_in, "U": U,
        "x_non_td": x_non_td, "x_td": x_td, "context": context
    }

# --- Test Functions ---

def test_grn_instantiation(grn_config):
    """Test revised GRN instantiation."""
    try:
        grn = GatedResidualNetwork(**grn_config)
        assert isinstance(grn, tf.keras.layers.Layer)
        # Check if activation function object was created correctly
        assert callable(grn.activation_fn)
        assert grn.activation_fn == tf.keras.activations.get('relu')
        assert grn.output_activation_fn is None
    except Exception as e:
        pytest.fail(f"GRN instantiation failed. Error: {e}")
    print("GRN Instantiation OK")

@pytest.mark.parametrize(
    "is_td_input, use_context, input_feature_match",
    [
        (False, False, True),  # Non-TD input, No Context, F_in == U
        (False, False, False), # Non-TD input, No Context, F_in != U
        (False, True, True),   # Non-TD input, With Context, F_in == U
        (False, True, False),  # Non-TD input, With Context, F_in != U
        (True, False, True),   # TD input, No Context, F_in == U
        (True, False, False),  # TD input, No Context, F_in != U
        (True, True, True),    # TD input, With Context, F_in == U
        (True, True, False)    # TD input, With Context, F_in != U
    ]
)
def test_grn_call_and_output_shape(
    grn_config, dummy_grn_data, is_td_input, use_context, input_feature_match
):
    """Test revised GRN call with various configurations."""
    config = grn_config.copy()
    F_in = config["units"] if input_feature_match else config["units"] + 5
    B = dummy_grn_data["B"]
    T = dummy_grn_data["T"]
    U = config["units"]

    # Select input tensor based on time distribution flag
    if is_td_input:
        x = np.random.rand(B, T, F_in).astype(np.float32)
        expected_shape = (B, T, U)
    else:
        x = np.random.rand(B, F_in).astype(np.float32)
        expected_shape = (B, U)

    context = dummy_grn_data["context"] if use_context else None

    try:
        grn = GatedResidualNetwork(**config)
        grn.build(x.shape) # Build explicitly

        # Check projection layer creation
        if F_in != U:
            assert grn.projection is not None, "Projection missing"
        else:
            assert grn.projection is None, "Projection created unnecessarily"

        # Perform forward pass
        outputs = grn(x, context=context, training=False)
    except Exception as e:
        pytest.fail(
            f"GRN call failed for TD Input={is_td_input}, "
            f"Context={use_context}, F_in!=Units={not input_feature_match}."
            f"\nInput Shape: {x.shape}, Context Shape: {context.shape if context is not None else None}"
            f"\nError: {e}"
        )

    # Check output shape
    assert outputs.shape == expected_shape, \
        f"Output shape mismatch. Expected {expected_shape}, got {outputs.shape}"

    print(f"GRN Call OK: TD Input={is_td_input}, Context={use_context},"
          f" F_in!=Units={not input_feature_match}, Output Shape={outputs.shape}")


@pytest.mark.parametrize("use_context", [False, True])
@pytest.mark.parametrize("use_batch_norm", [False, True])
def test_grn_minimal_train_step(
    grn_config, dummy_grn_data, use_context, use_batch_norm
):
    """Test if revised GRN runs within a minimal trainable model."""
    config = grn_config.copy()
    config["use_batch_norm"] = use_batch_norm # Test BN flag

    grn = GatedResidualNetwork(**config)

    x = dummy_grn_data["x_non_td"]
    context = dummy_grn_data["context"] if use_context else None
    dummy_target = np.random.rand(
        dummy_grn_data["B"], grn_config["units"]
        ).astype(np.float32)

    # Simple wrapper model
    class TestModel(tf.keras.Model):
        def __init__(self, grn_layer):
            super().__init__()
            self.grn = grn_layer
            self.dense = tf.keras.layers.Dense(grn_layer.units)

        def call(self, inputs, training=False):
            if isinstance(inputs, (list, tuple)):
                 x_in, ctx_in = inputs
            else:
                 x_in, ctx_in = inputs, None
            grn_out = self.grn(x_in, context=ctx_in, training=training)
            return self.dense(grn_out)

    test_model = TestModel(grn)
    model_inputs = [x, context] if use_context else x

    try:
        test_model.compile(optimizer='adam', loss='mse')
        history = test_model.fit(
            model_inputs, dummy_target, epochs=1, batch_size=2, verbose=0
            )
    except Exception as e:
        pytest.fail(f"GRN minimal train step failed (Context={use_context}, "
                    f"BN={use_batch_norm}). Error: {e}")

    assert history is not None
    assert 'loss' in history.history
    print(f"GRN Minimal Train Step OK: Context={use_context}, BN={use_batch_norm}")


def test_grn_serialization(grn_config, dummy_grn_data):
    """Test revised GRN get_config and from_config."""
    config = grn_config.copy()
    config["dropout_rate"] = 0.15
    config["activation"] = "gelu"
    config["output_activation"] = "linear"
    config["use_batch_norm"] = True

    grn = GatedResidualNetwork(**config)
    x = dummy_grn_data["x_non_td"] # Need input to build
    _ = grn(x) # Call once to build projection if needed

    try:
        retrieved_config = grn.get_config()
        # Check important params (activations are strings)
        assert retrieved_config['activation'] == config['activation']
        assert retrieved_config['output_activation'] == config['output_activation']
        assert 'use_time_distributed' not in retrieved_config # Should be removed

        rebuilt_grn = GatedResidualNetwork.from_config(retrieved_config)
    except Exception as e:
        pytest.fail(f"GRN serialization/deserialization failed. Error: {e}")

    assert isinstance(rebuilt_grn, GatedResidualNetwork)
    # Check output shape consistency
    context = dummy_grn_data["context"]
    try:
        output_original = grn(x, context=context)
        output_rebuilt = rebuilt_grn(x, context=context)
        assert output_original.shape == output_rebuilt.shape
    except Exception as e:
         pytest.fail(f"Output shape mismatch after from_config. Error: {e}")

    print("GRN Serialization OK")

if __name__=='__main__':
     pytest.main([__file__])
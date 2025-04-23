# test_temporal_attention_layer.py

import pytest
import numpy as np

try:
    import tensorflow as tf
    from fusionlab.nn.components import (
        TemporalAttentionLayer,
        GatedResidualNetwork 
        )
    # Check if TF backend is available
    from fusionlab.nn import KERAS_BACKEND
except ImportError as e:
    print(f"Skipping TAL tests due to import error: {e}")
    KERAS_BACKEND = False
# --- End Imports ---

# Skip all tests in this file if TensorFlow/Keras backend is not available
pytestmark = pytest.mark.skipif(
    not KERAS_BACKEND, reason="TensorFlow/Keras backend not available"
)

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def tal_config():
    """Provides default configuration for TemporalAttentionLayer."""
    return {
        "units": 16,       # Output/hidden dimension
        "num_heads": 4,    # Number of attention heads
        "dropout_rate": 0.0,
        "activation": 'relu',
        "use_batch_norm": False, # In internal GRNs
    }

@pytest.fixture(scope="module")
def dummy_tal_data(tal_config):
    """Provides dummy data for TAL tests."""
    B, T, U = 4, 10, tal_config["units"] # Batch, TimeSteps, Units

    # Main time-distributed input
    inputs_td = np.random.rand(B, T, U).astype(np.float32)
    # Context vector (Non-time-distributed, matches units)
    context = np.random.rand(B, U).astype(np.float32)
    # Dummy target for training test
    dummy_target = np.random.rand(B, T, U).astype(np.float32)

    return {
        "B": B, "T": T, "U": U,
        "inputs_td": tf.constant(inputs_td),
        "context": tf.constant(context),
        "dummy_target": tf.constant(dummy_target)
    }

# --- Test Functions ---

def test_tal_instantiation(tal_config):
    """Test basic TAL instantiation."""
    try:
        tal = TemporalAttentionLayer(**tal_config)
        assert isinstance(tal, tf.keras.layers.Layer)
        # Check internal layers are created
        assert hasattr(tal, 'multi_head_attention')
        assert hasattr(tal, 'context_grn')
        assert hasattr(tal, 'output_grn')
        assert hasattr(tal, 'dropout')
        assert hasattr(tal, 'layer_norm1')
        assert isinstance(tal.context_grn, GatedResidualNetwork)
        assert isinstance(tal.output_grn, GatedResidualNetwork)
    except Exception as e:
        pytest.fail(f"TAL instantiation failed. Error: {e}")
    print("TAL Instantiation OK")

@pytest.mark.parametrize("use_context", [False, True])
@pytest.mark.parametrize("training_flag", [False, True])
def test_tal_call_and_output_shape(
    tal_config, dummy_tal_data, use_context, training_flag
):
    """Test TAL call with and without context, and training flag."""
    config = tal_config.copy()
    # Add dropout for testing training flag difference
    if training_flag:
        config["dropout_rate"] = 0.1
    tal = TemporalAttentionLayer(**config)

    inputs_td = dummy_tal_data["inputs_td"]
    context_tensor = dummy_tal_data["context"] if use_context else None
    B = dummy_tal_data["B"]
    T = dummy_tal_data["T"]
    U = config["units"]

    # Perform forward pass
    try:
        # Build layer explicitly (optional, call builds it)
        # tal.build(inputs_td.shape)
        outputs = tal(inputs_td, context_vector=context_tensor,
                      training=training_flag)
    except Exception as e:
        pytest.fail(
            f"TAL call failed for Context={use_context}, "
            f"Training={training_flag}. "
            f"\nInput Shape: {inputs_td.shape}, "
            f"Context Shape: {context_tensor.shape if context_tensor is not None else None}"
            f"\nError: {e}"
            )

    # Check output shape
    expected_shape = (B, T, U)
    assert outputs.shape == expected_shape, \
        f"Output shape mismatch. Expected {expected_shape}, got {outputs.shape}"

    # Check output type
    assert isinstance(outputs, tf.Tensor)

    print(f"TAL Call OK: Context={use_context}, Training={training_flag},"
          f" Output Shape={outputs.shape}")

@pytest.mark.parametrize("use_context", [False, True])
def test_tal_minimal_train_step(tal_config, dummy_tal_data, use_context):
    """Test if TAL works within a minimal trainable model."""
    config = tal_config.copy()
    tal = TemporalAttentionLayer(**config)

    inputs_td = dummy_tal_data["inputs_td"]
    context_tensor = dummy_tal_data["context"] if use_context else None
    dummy_target = dummy_tal_data["dummy_target"] # Matches output shape

    # Create a simple model wrapping the TAL
    class TestModel(tf.keras.Model):
        def __init__(self, tal_layer):
            super().__init__()
            self.tal = tal_layer
            # Add a final dense layer to ensure output is used in loss
            self.dense = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(tal_layer.units)
                )

        def call(self, inputs, training=False):
            # Handle inputs list [main_input, context] or just main_input
            if isinstance(inputs, (list, tuple)):
                 main_input, ctx_input = inputs
            else:
                 main_input, ctx_input = inputs, None
            tal_out = self.tal(
                main_input, context_vector=ctx_input, training=training
                )
            return self.dense(tal_out) # Pass through Dense

    test_model = TestModel(tal)
    model_inputs = [inputs_td, context_tensor] if use_context else inputs_td

    try:
        test_model.compile(optimizer='adam', loss='mse')
        history = test_model.fit(
            model_inputs, dummy_target, epochs=1, batch_size=2, verbose=0
            )
    except Exception as e:
        pytest.fail(f"TAL minimal train step failed (Context={use_context})."
                    f" Error: {e}")

    assert history is not None
    assert 'loss' in history.history
    print(f"TAL Minimal Train Step OK: Context={use_context}")


def test_tal_serialization(tal_config, dummy_tal_data):
    """Test TAL get_config and from_config."""
    config = tal_config.copy()
    config["dropout_rate"] = 0.15 # Non-default
    config["activation"] = 'gelu' # Non-default
    config["use_batch_norm"] = True # Non-default

    tal = TemporalAttentionLayer(**config)

    # Build the layer first
    inputs_td = dummy_tal_data["inputs_td"]
    context_tensor = dummy_tal_data["context"]
    _ = tal(inputs_td, context_vector=context_tensor) # Call once to build

    try:
        retrieved_config = tal.get_config()
        # Check key parameters saved correctly
        assert retrieved_config['units'] == config['units']
        assert retrieved_config['num_heads'] == config['num_heads']
        assert retrieved_config['dropout_rate'] == config['dropout_rate']
        assert retrieved_config['activation'] == config['activation'] # String
        assert retrieved_config['use_batch_norm'] == config['use_batch_norm']

        rebuilt_tal = TemporalAttentionLayer.from_config(retrieved_config)
    except Exception as e:
        pytest.fail(f"TAL serialization/deserialization failed. Error: {e}")

    assert isinstance(rebuilt_tal, TemporalAttentionLayer)
    # Check output shape consistency
    try:
        output_original = tal(inputs_td, context_vector=context_tensor)
        output_rebuilt = rebuilt_tal(inputs_td, context_vector=context_tensor)
        assert output_original.shape == output_rebuilt.shape
    except Exception as e:
         pytest.fail(f"Output shape mismatch after from_config. Error: {e}")

    print("TAL Serialization OK")

# Allows running the tests directly if needed
if __name__=='__main__':
     pytest.main([__file__])

import pytest
import numpy as np

try:
    # import tensorflow as tf
    import tensorflow as tf
    from tensorflow.keras.layers import Layer, Input
    from tensorflow.keras.models import Model
    from fusionlab.nn.components import PositionalEncoding 
    from fusionlab.nn import KERAS_BACKEND
except ImportError as e:
    print(f"Skipping combine_temporal_inputs_for_lstm tests due to"
          f" import error: {e}")
    KERAS_BACKEND = False
# --- End Imports ---

# Skip all tests in this file if TensorFlow/Keras backend is not available
pytestmark = pytest.mark.skipif(
    not KERAS_BACKEND, reason="TensorFlow/Keras backend not available"
)


# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def dummy_input_tensor() -> tf.Tensor:
    """Provides a standard input tensor for tests."""
    # Use a fixed seed for reproducibility in tests
    tf.random.set_seed(42)
    return tf.random.normal(
        (4, 50, 128)
    ) # (batch_size, sequence_length, feature_dim)

# --- Test Cases ---

def test_pe_output_shape(dummy_input_tensor):
    """
    Tests that the output shape is identical to the input shape.
    """
    pe_layer = PositionalEncoding(max_length=100)
    output = pe_layer(dummy_input_tensor)
    assert output.shape == dummy_input_tensor.shape

def test_pe_values_are_added(dummy_input_tensor):
    """
    Tests that the output is different from the input, confirming
    that the encoding was successfully added.
    """
    pe_layer = PositionalEncoding(max_length=100)
    output = pe_layer(dummy_input_tensor)
    
    # The sum of absolute differences should be greater than zero,
    # unless by some astronomical coincidence the input is the exact
    # negative of the encoding. This check is practically reliable.
    assert tf.reduce_sum(tf.abs(output - dummy_input_tensor)) > 1e-6

def test_pe_values_are_bounded():
    """
    Tests that the generated encoding values are within the expected
    [-1, 1] range of sine and cosine.
    """
    pe_layer = PositionalEncoding(max_length=100)
    # Build the layer to create the positional_encoding matrix
    pe_layer.build(input_shape=(None, None, 128))
    
    encoding_matrix = pe_layer.positional_encoding
    assert tf.reduce_min(encoding_matrix) >= -1.0
    assert tf.reduce_max(encoding_matrix) <= 1.0

def test_pe_serialization(dummy_input_tensor):
    """Tests the get_config and from_config methods for serialization."""
    pe_layer1 = PositionalEncoding(max_length=1234)
    # Call the layer to build it before getting the config
    output1 = pe_layer1(dummy_input_tensor)
    
    config = pe_layer1.get_config()
    assert config['max_length'] == 1234
    
    pe_layer2 = PositionalEncoding.from_config(config)
    output2 = pe_layer2(dummy_input_tensor)

    # The outputs should be identical if the layers are configured the same.
    np.testing.assert_allclose(output1.numpy(), output2.numpy(), rtol=1e-6)

def test_pe_call_with_training_flag(dummy_input_tensor):
    """
    Tests that the layer's call method correctly accepts the
    `training` argument for Keras API compatibility, even if unused.
    """
    pe_layer = PositionalEncoding(max_length=100)
    try:
        # This call should succeed without a TypeError
        _ = pe_layer(dummy_input_tensor, training=True)
        _ = pe_layer(dummy_input_tensor, training=False)
    except TypeError:
        pytest.fail(
            "PositionalEncoding.call() raised a TypeError when passed "
            "the 'training' argument."
        )

def test_pe_in_keras_model(dummy_input_tensor):
    """
    Tests the layer's integration into a simple Keras functional model.
    """
    inputs = Input(shape=(50, 128))
    outputs = PositionalEncoding(max_length=100)(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    
    # Ensure the model can be called without error
    model_output = model(dummy_input_tensor)
    assert model_output.shape == dummy_input_tensor.shape

@pytest.mark.parametrize("seq_len", [10, 100, 500])
def test_pe_with_variable_sequence_lengths(seq_len):
    """
    Tests that the layer correctly handles different input sequence lengths
    by slicing its pre-computed encoding matrix.
    """
    batch_size = 2
    feature_dim = 64
    max_len = 1000
    
    input_tensor = tf.random.normal((batch_size, seq_len, feature_dim))
    pe_layer = PositionalEncoding(max_length=max_len)
    
    output = pe_layer(input_tensor)
    
    # The output shape's sequence length must match the input's
    assert output.shape[1] == seq_len
    
    # The encoding added should be a slice of the full matrix
    encoding_only = pe_layer(tf.zeros_like(input_tensor))
    full_encoding_matrix = pe_layer.positional_encoding
    
    # Check if the applied encoding matches the corresponding slice
    # of the full matrix.
    expected_slice = full_encoding_matrix[:, :seq_len, :]
    # --- FIX: Compare tensors with compatible shapes ---
    # The `encoding_only` tensor has a batch size of 2, while the
    # `expected_slice` has a batch size of 1. We compare the content
    # of the first element of the batch, which should be identical
    # to the (squeezed) expected slice due to broadcasting.
    np.testing.assert_allclose(
        encoding_only.numpy()[0], # Shape: (seq_len, feature_dim)
        np.squeeze(expected_slice.numpy(), axis=0), # Shape: (seq_len, feature_dim)
        rtol=1e-6
    )

if __name__=='__main__': 
    pytest.main( [__file__, '-vv'])
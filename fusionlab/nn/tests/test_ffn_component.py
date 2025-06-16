
import pytest
import numpy as np
import os

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, save_model, load_model
    # Assuming the class is in this location
    from fusionlab.nn.components import PositionwiseFeedForward
    KERAS_BACKEND = True
except ImportError:
    KERAS_BACKEND = False

# Skip all tests if backend is not available
pytestmark = pytest.mark.skipif(
    not KERAS_BACKEND, reason="TensorFlow/Keras backend not available"
)

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def layer_params() -> dict:
    """Provides a standard set of initialization parameters."""
    return {
        "embed_dim": 128,
        "ffn_dim": 512,
        "activation": "relu",
        "dropout_rate": 0.1,
    }

@pytest.fixture(scope="module")
def input_data() -> tf.Tensor:
    """Provides a standard dummy input tensor."""
    # (batch_size, sequence_length, embed_dim)
    return tf.random.normal((32, 50, 128))

# --- Test Cases ---

def test_initialization(layer_params):
    """
    Tests if the layer can be instantiated correctly and if its
    internal components are created.
    """
    ffn_layer = PositionwiseFeedForward(**layer_params)
    
    assert isinstance(ffn_layer, tf.keras.layers.Layer)
    assert ffn_layer.embed_dim == layer_params["embed_dim"]
    assert ffn_layer.ffn_dim == layer_params["ffn_dim"]
    assert ffn_layer.dropout.rate == layer_params["dropout_rate"]
    
    # Check that internal layers are Dense layers
    assert isinstance(ffn_layer.dense_1, tf.keras.layers.Dense)
    assert isinstance(ffn_layer.dense_2, tf.keras.layers.Dense)
    
    # Check that the dimensions of internal layers are correct
    assert ffn_layer.dense_1.units == layer_params["ffn_dim"]
    assert ffn_layer.dense_2.units == layer_params["embed_dim"]

def test_output_shape(layer_params, input_data):
    """
    Tests that the layer's output shape is identical to the input shape.
    """
    ffn_layer = PositionwiseFeedForward(**layer_params)
    output_tensor = ffn_layer(input_data)
    
    assert output_tensor.shape == input_data.shape

@pytest.mark.parametrize("activation_fn", ["relu", "gelu", "tanh"])
def test_different_activations(layer_params, input_data, activation_fn):
    """
    Tests that the layer can be successfully initialized and called with
    various activation functions.
    """
    params = layer_params.copy()
    params["activation"] = activation_fn
    
    try:
        ffn_layer = PositionwiseFeedForward(**params)
        output_tensor = ffn_layer(input_data)
        assert output_tensor.shape == input_data.shape
    except Exception as e:
        pytest.fail(
            f"FFN layer failed to initialize or execute with "
            f"activation='{activation_fn}'. Error: {e}"
        )

def test_call_in_training_vs_inference_mode(layer_params, input_data):
    """
    Tests that the layer behaves differently in training vs. inference
    modes due to dropout.
    """
    # Use a non-zero dropout rate for this test
    params = layer_params.copy()
    params["dropout_rate"] = 0.5 
    ffn_layer = PositionwiseFeedForward(**params)
    
    # In inference mode, output should be deterministic
    output_inference_1 = ffn_layer(input_data, training=False)
    output_inference_2 = ffn_layer(input_data, training=False)
    np.testing.assert_allclose(
        output_inference_1.numpy(), output_inference_2.numpy()
    )

    # In training mode, dropout is active, output should differ from inference
    output_training = ffn_layer(input_data, training=True)
    # Check if the training output is different from the inference output
    assert not np.allclose(
        output_inference_1.numpy(), output_training.numpy()
    )

def test_serialization(layer_params, input_data, tmp_path):
    """
    Tests that a model containing the custom layer can be saved and
    reloaded successfully. This validates the get_config method and
    Keras serialization.
    """
    # 1. Create a simple model using the custom layer
    model = Sequential([
        tf.keras.layers.InputLayer(input_shape=(50, 128)),
        PositionwiseFeedForward(**layer_params)
    ])
    
    # 2. Get predictions from the original model
    original_predictions = model.predict(input_data)
    
    # 3. Save the model to a temporary directory
    model_path = os.path.join(tmp_path, "ffn_model.keras")
    model.save(model_path)
    
    # 4. Load the model back
    # No `custom_objects` needed due to @register_keras_serializable
    reloaded_model = load_model(model_path)
    
    # 5. Get predictions from the reloaded model
    reloaded_predictions = reloaded_model.predict(input_data)
    
    # 6. Assert that the predictions are identical
    np.testing.assert_allclose(
        original_predictions, reloaded_predictions, atol=1e-6
    )

    # 7. Check if the config was saved and restored correctly
    assert reloaded_model.layers[0].embed_dim == layer_params["embed_dim"]
    assert reloaded_model.layers[0].ffn_dim == layer_params["ffn_dim"]

if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
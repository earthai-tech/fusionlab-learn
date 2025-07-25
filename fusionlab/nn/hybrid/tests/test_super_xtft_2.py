import pytest
import numpy as np 
tf = pytest.importorskip("tensorflow", reason="TF required")

import tensorflow as tf
from fusionlab.nn.hybrid._super_xtft import SuperXTFT

# Mock input dimensions and hyperparameters for the test
@pytest.fixture
def super_xtft_model():
    model = SuperXTFT(
        static_input_dim=10,
        dynamic_input_dim=32,
        future_input_dim=8,
        embed_dim=32,
        forecast_horizon=7,
        quantiles=[0.1, 0.5, 0.9],
        max_window_size=10,
        memory_size=100,
        num_heads=4,
        dropout_rate=0.1,
        output_dim=1,
        attention_units=32,
        hidden_units=64,
        lstm_units=64,
        scales='auto',
        multi_scale_agg='auto',
        activation='relu',
        use_residuals=True,
        use_batch_norm=False,
        final_agg='last',
        architecture_config={
            'encoder_type': 'hybrid',
            'decoder_attention_stack': ['cross', 'hierarchical', 'memory'],
            'feature_processing': 'vsn',
        },
    )
    return model

# Test: Initialization and architecture config
def test_superxtft_initialization(super_xtft_model):
    assert super_xtft_model.encoder_type == 'hybrid'
    assert 'cross' in super_xtft_model.architecture_config['decoder_attention_stack']
    assert 'hierarchical' in super_xtft_model.architecture_config['decoder_attention_stack']
    assert 'memory' in super_xtft_model.architecture_config['decoder_attention_stack']
    assert super_xtft_model.feature_processing == 'vsn'

# Test: Forward pass and output shape
def test_superxtft_forward_pass(super_xtft_model):
    # Create dummy data
    static_input = tf.random.normal((8, 10))  # (batch_size, static_input_dim)
    dynamic_input = tf.random.normal((8, 5, 32))  # (batch_size, t_past, dynamic_input_dim)
    future_input = tf.random.normal((8, 7, 8))   # (batch_size, t_fut, future_input_dim)
    
    inputs = [static_input, dynamic_input, future_input]
    
    # Forward pass
    predictions = super_xtft_model(inputs, training=True)

    # Check if the output shape is correct
    # Expected shape: (batch_size, forecast_horizon, num_quantiles)
    assert predictions.shape == (8, 7, 3)  # Shape (batch_size, forecast_horizon, quantiles)

# Test: Attention mechanisms behavior
def test_superxtft_attention_mechanisms(super_xtft_model):
    # Ensure that attention mechanisms are correctly applied
    # Dummy data
    static_input = tf.random.normal((8, 10))
    dynamic_input = tf.random.normal((8, 5, 32))  # (batch_size, t_past, dynamic_input_dim)
    future_input = tf.random.normal((8, 7, 8))   # (batch_size, t_fut, future_input_dim)
    inputs = [static_input, dynamic_input, future_input]
    
    # Perform the forward pass
    super_xtft_model(inputs, training=True)

    # Check if the model applies attention correctly
    # This would normally be done by inspecting logs or intermediate outputs. 
    # For now, we will just verify no exceptions are thrown when applying the stack.
    assert True  # If no exception was thrown, the test passes

# Test: Dense vs VSN feature processing
@pytest.mark.parametrize("feature_processing, expected_output", [
    ('vsn', 'vsn'),
    ('dense', 'dense')
])
def test_feature_processing(super_xtft_model, feature_processing, expected_output):
    # Set the feature_processing configuration to either 'vsn' or 'dense'
    super_xtft_model.architecture_config['feature_processing'] = feature_processing
    
    # Apply the configuration
    super_xtft_model._build_components()

    # Check that the architecture configuration matches the expected feature processing
    assert super_xtft_model.feature_processing == expected_output

# Test: Decoder attention stack - Ensure correct attention layers are used
def test_decoder_attention_stack(super_xtft_model):
    # Check the attention stack is applied correctly based on architecture config
    attention_stack = super_xtft_model.architecture_config['decoder_attention_stack']
    assert len(attention_stack) == 3
    assert 'cross' in attention_stack
    assert 'hierarchical' in attention_stack
    assert 'memory' in attention_stack

    # Create dummy data
    static_input = tf.random.normal((8, 10))  # (batch_size, static_input_dim)
    dynamic_input = tf.random.normal((8, 5, 32))  # (batch_size, t_past, dynamic_input_dim)
    future_input = tf.random.normal((8, 7, 8))   # (batch_size, t_fut, future_input_dim)
    inputs = [static_input, dynamic_input, future_input]

    # Perform a forward pass
    super_xtft_model(inputs, training=True)

    # Ensure no errors are thrown during the forward pass for different attention mechanisms
    assert True

# Test: Feature-based anomaly scores
def test_superxtft_feature_based_anomaly_scores(super_xtft_model):
    b, t_past, t_fut = 3, 6, 2
    x_static = np.random.rand(b, 2).astype("float32")
    x_dyn = np.random.rand(b, t_past, 3).astype("float32")
    x_fut = np.random.rand(b, t_fut, 1).astype("float32")

    model = SuperXTFT(
        static_input_dim=2,
        dynamic_input_dim=3,
        future_input_dim=1,
        forecast_horizon=t_fut,
        quantiles=[0.5],
        anomaly_detection_strategy="feature_based",
        anomaly_loss_weight=0.5,
    )
    _ = model([x_static, x_dyn, x_fut], training=True)

    assert model.anomaly_scores is not None

    model.compile(optimizer="adam")
    y = np.random.rand(b, t_fut, 1).astype("float32")
    model.fit([x_static, x_dyn, x_fut], y,
              epochs=1, batch_size=1, verbose=0)

    y_pred = model.predict([x_static, x_dyn, x_fut], verbose=0)
    assert y_pred.shape[0] == b


if __name__ == '__main__': 
    pytest.main([__file__])

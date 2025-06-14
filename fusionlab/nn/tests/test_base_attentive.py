import pytest

try:
    import tensorflow as tf
    from fusionlab.nn._base_attentive import BaseAttentive
    FUSIONLAB_AVAILABLE = True
except ImportError as e:
    print(f"Could not import PIHALNet or dependencies: {e}")
    FUSIONLAB_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FUSIONLAB_AVAILABLE,
    reason="fusionlab.nn.pinn.models.PIHALNet or dependencies not found"
)

@pytest.fixture
def example_inputs():
    return {
        "static_features": tf.random.normal([32, 4]),  # B × S
        "dynamic_features": tf.random.normal([32, 10, 8]),  # B × T × D
        "future_features": tf.random.normal([32, 24, 6]),  # B × H × F
        "coords": tf.zeros([32, 24, 3])  # Dummy (t, x, y)
    }

# Test for model initialization
def test_base_attentive_initialization():
    model = BaseAttentive(
        static_input_dim=4,
        dynamic_input_dim=8,
        future_input_dim=6,
        output_dim=2,
        forecast_horizon=24,
        quantiles=[0.1, 0.5, 0.9],
        scales=[1, 3],
        multi_scale_agg="concat",
        final_agg="last",
        attention_units=64,
        num_heads=8,
        dropout_rate=0.15,
    )
    
    # Test that model initializes correctly
    assert model.static_input_dim == 4
    assert model.dynamic_input_dim == 8
    assert model.future_input_dim == 6
    assert model.output_dim == 2
    assert model.forecast_horizon == 24
    assert model.quantiles == [0.1, 0.5, 0.9]
    assert model.scales == [1, 3]
    assert model.multi_scale_agg_mode == "concat"
    assert model.final_agg == "last"
    assert model.attention_units == 64
    assert model.num_heads == 8
    assert model.dropout_rate == 0.15

# Test for `get_config` method
def test_get_config():
    model = BaseAttentive(
        static_input_dim=4,
        dynamic_input_dim=8,
        future_input_dim=6,
        output_dim=2,
        forecast_horizon=24,
        quantiles=[0.1, 0.5, 0.9],
        scales=[1, 3],
        multi_scale_agg="concat",
        final_agg="last",
        attention_units=64,
        num_heads=8,
        dropout_rate=0.15,
    )
    
    config = model.get_config()
    
    # Test that the config dictionary contains the correct values
    assert config["static_input_dim"] == 4
    assert config["dynamic_input_dim"] == 8
    assert config["future_input_dim"] == 6
    assert config["output_dim"] == 2
    assert config["forecast_horizon"] == 24
    assert config["quantiles"] == [0.1, 0.5, 0.9]
    assert config["scales"] == [1, 3]
    assert config["multi_scale_agg"] == "concat"
    assert config["final_agg"] == "last"
    assert config["attention_units"] == 64
    assert config["num_heads"] == 8
    assert config["dropout_rate"] == 0.15

# Test for the `apply_dtw` parameter behavior
@pytest.mark.parametrize("apply_dtw", [True, False])
def test_apply_dtw(apply_dtw):
    model = BaseAttentive(
        static_input_dim=4,
        dynamic_input_dim=8,
        future_input_dim=6,
        output_dim=2,
        forecast_horizon=24,
        apply_dtw=apply_dtw
    )
    
    # Check the value of `apply_dtw`
    assert model.apply_dtw == apply_dtw

# Test for the `objective` parameter (hybrid vs transformer)
@pytest.mark.parametrize("objective", ["hybrid", "transformer"])
def test_mode_parameter(objective):
    model = BaseAttentive(
        static_input_dim=4,
        dynamic_input_dim=8,
        future_input_dim=6,
        output_dim=2,
        forecast_horizon=24,
        objective=objective
    )
    
    # Test that the objective is correctly set
    assert model.architecture_config['encoder_type']== objective

# Test for the `att_levels` functionality (cross, hierarchical, memory)
@pytest.mark.parametrize(
    "att_levels", [None, "use_all", "cross", "hierarchical", "memory"])
def test_attention_levels(att_levels):
    model = BaseAttentive(
        static_input_dim=4,
        dynamic_input_dim=8,
        future_input_dim=6,
        output_dim=2,
        forecast_horizon=24,
        attention_levels=att_levels
    )
    
    # Validate that attention levels are properly assigned
    if att_levels is None or att_levels == "use_all":
        assert model.architecture_config['decoder_attention_stack'] == ["cross", "hierarchical", "memory"]
    else:
        assert model.architecture_config['decoder_attention_stack'] == [att_levels]
        
# Test for the hybrid and transformer architecture setting
@pytest.mark.parametrize("objective", ["hybrid", "transformer"])
def test_architecture_setting(objective):
    model = BaseAttentive(
        static_input_dim=4,
        dynamic_input_dim=8,
        future_input_dim=6,
        output_dim=2,
        forecast_horizon=24,
        objective=objective
    )
    
    # Ensure the architecture is correctly set
    assert model.architecture_config['encoder_type'] == objective

# Test for model's `call` method with example inputs
def test_call(example_inputs):
    model = BaseAttentive(
        static_input_dim=4,
        dynamic_input_dim=8,
        future_input_dim=6,
        output_dim=2,
        forecast_horizon=24,
        quantiles=[0.1, 0.5, 0.9],
        scales=[1, 3],
        multi_scale_agg="concat",
        final_agg="last",
        attention_units=64,
        num_heads=8,
        dropout_rate=0.15,
    )
    
    # Run the forward pass
    inputs = [ example_inputs.get('static_features'),
              example_inputs.get('dynamic_features'),
              example_inputs.get('future_features'),
              ]
    outputs = model(inputs, training=True)
    
    # Ensure the model outputs a tensor with the expected shape
    assert outputs.shape == (32, 24, 3, 2)  # B × H × Q × output_dim

# Test for `fit` method
@pytest.mark.parametrize("objective", ["hybrid", "transformer"])
@pytest.mark.parametrize(
    "attention_levels", [None, "cross", 'hier', 'memory']
  )
def test_fit(example_inputs, objective, attention_levels):
    model = BaseAttentive(
        static_input_dim=4,
        dynamic_input_dim=8,
        future_input_dim=6,
        output_dim=2,
        forecast_horizon=24,
        quantiles=[0.1, 0.5, 0.9],
        scales=[1, 3],
        multi_scale_agg="concat",
        final_agg="last",
        attention_units=64,
        num_heads=8,
        dropout_rate=0.15,
        objective = objective, 
        attention_levels= attention_levels, 
    )
    inputs = [ example_inputs.get('static_features'),
              example_inputs.get('dynamic_features'),
              example_inputs.get('future_features'),
              ]
    model.compile(optimizer="adam", loss="mse")
    
    # Create dummy targets for fitting
    y_dummy = tf.random.normal([32, 24, 3, 2])  # B × H × Q × output_dim
    
    # Fit the model
    model.fit(inputs, y_dummy, epochs=2, batch_size=32)
    
    # Test that the model runs without errors
    assert True

# Test for `compile` method
def test_compile():
    model = BaseAttentive(
        static_input_dim=4,
        dynamic_input_dim=8,
        future_input_dim=6,
        output_dim=2,
        forecast_horizon=24,
        quantiles=[0.1, 0.5, 0.9],
        scales=[1, 3],
        multi_scale_agg="concat",
        final_agg="last",
        attention_units=64,
        num_heads=8,
        dropout_rate=0.15,
    )
    
    # Compile the model
    model.compile(optimizer="adam", loss="mse")
    
    # Ensure the model compiles without error
    assert model.optimizer is not None
    assert model.loss == "mse"

if __name__ =='__main__': # pragma : no cover 
    pytest.main( [__file__,  "--maxfail=1 ", "--disable-warnings",  "-q"])

import pytest
import numpy as np


# Attempt to import from the library structure
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    from fusionlab.nn._transformers import (
        TimeSeriesTransformer,
        TransformerEncoderLayer, 
        TransformerDecoderLayer, 
        create_causal_mask,
        prepare_model_inputs 
    )
    from fusionlab.nn.components import (
        PositionalEncodingTF,
        QuantileDistributionModeling,
        GatedResidualNetwork
    )
    from fusionlab.api.property import NNLearner # If needed for specific checks

    FUSIONLAB_AVAILABLE = True
except ImportError as e:
    print(f"Could not import fusionlab components, tests might be limited: {e}")
    FUSIONLAB_AVAILABLE = False
    # Define dummy classes if imports fail, to allow pytest to collect tests
    # This is a fallback for environments where fusionlab isn't installed/discoverable
    # but the core logic is being tested. For actual library testing, imports must work.
    class Layer: pass
    class Model: pass
    class NNLearner: pass
    class TimeSeriesTransformer(Model, NNLearner): # type: ignore
        def __init__(self, *args, **kwargs): pass
        def call(self, *args, **kwargs): pass
    class PositionalEncodingTF(Layer): pass # type: ignore
    class QuantileDistributionModeling(Layer): pass # type: ignore
    class GatedResidualNetwork(Layer): pass # type: ignore
    def prepare_model_inputs(*args, **kwargs): return [None,None,None]
    def create_causal_mask(*args, **kwargs): return None


# Test Parameters
BATCH_SIZE = 4
T_PAST = 20         # Sequence length for dynamic historical inputs
T_HORIZON = 10      # Forecast horizon (and length of decoder input sequence)
S_DIM_DEF = 5       # Default static feature dimension
D_DIM_DEF = 8       # Default dynamic feature dimension
F_DIM_DEF = 3       # Default known future feature dimension (for decoder input)
OUTPUT_DIM_DEF = 1  # Default number of target variables to forecast
EMBED_DIM = 32
NUM_HEADS = 2
FFN_DIM = 64
NUM_ENC_LAYERS = 1 # Keep small for faster tests
NUM_DEC_LAYERS = 1 # Keep small for faster tests
MAX_SEQ_LEN_ENCODER = T_PAST + 10
MAX_SEQ_LEN_DECODER = T_HORIZON + 10

# Helper to generate dummy data
def generate_dummy_data(
    s_dim=S_DIM_DEF, d_dim=D_DIM_DEF, f_dim=F_DIM_DEF,
    t_past=T_PAST, t_horizon=T_HORIZON, batch_size=BATCH_SIZE
):
    static_data = np.random.rand(batch_size, s_dim).astype(np.float32) if s_dim > 0 else None
    dynamic_data = np.random.rand(batch_size, t_past, d_dim).astype(np.float32)
    # Future input for the decoder should match the forecast horizon length
    future_data = np.random.rand(batch_size, t_horizon, f_dim).astype(np.float32) if f_dim > 0 else None
    
    inputs = []
    if static_data is not None:
        inputs.append(static_data)
    inputs.append(dynamic_data)
    if future_data is not None:
        inputs.append(future_data)
    
    # For target data (y_true for training test)
    y_shape = (batch_size, t_horizon, OUTPUT_DIM_DEF) # For point forecast
    # Adjust y_shape if quantiles are used, e.g. (batch_size, t_horizon, num_quantiles) for O_dim=1
    y_true = np.random.rand(*y_shape).astype(np.float32)

    return inputs, y_true

@pytest.mark.skipif(not FUSIONLAB_AVAILABLE, reason="fusionlab components not available")
class TestTimeSeriesTransformer:

    def get_model_config(self, static_dim, dynamic_dim, future_dim, **kwargs):
        config = {
            "static_input_dim": static_dim,
            "dynamic_input_dim": dynamic_dim,
            "future_input_dim": future_dim,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "ffn_dim": FFN_DIM,
            "num_encoder_layers": NUM_ENC_LAYERS,
            "num_decoder_layers": NUM_DEC_LAYERS,
            "output_dim": OUTPUT_DIM_DEF,
            "dropout_rate": 0.0,
            "input_dropout_rate": 0.0,
            "max_seq_len_encoder": MAX_SEQ_LEN_ENCODER,
            "max_seq_len_decoder": MAX_SEQ_LEN_DECODER, # Should match T_HORIZON for PE
            "forecast_horizon": T_HORIZON,
            "quantiles": None,
            "use_grn_for_static": False,
            "static_integration_mode": 'add_to_decoder_input',
        }
        config.update(kwargs)
        return config

    @pytest.mark.parametrize("s_dim, d_dim, f_dim", [
        (S_DIM_DEF, D_DIM_DEF, F_DIM_DEF), # All present
        (0, D_DIM_DEF, F_DIM_DEF),         # No static
        (S_DIM_DEF, D_DIM_DEF, 0),         # No future known covariates
        (0, D_DIM_DEF, 0),                 # Dynamic only
    ])
    def test_model_instantiation(self, s_dim, d_dim, f_dim):
        """Tests model instantiation with various input dimension configurations."""
        config = self.get_model_config(s_dim, d_dim, f_dim)
        model = TimeSeriesTransformer(**config)
        assert isinstance(model, TimeSeriesTransformer)
        assert model.static_input_dim == s_dim
        assert model.dynamic_input_dim == d_dim
        assert model.future_input_dim == f_dim

    def test_model_instantiation_with_quantiles(self):
        """Tests model instantiation with quantiles."""
        quantiles = [0.1, 0.5, 0.9]
        config = self.get_model_config(S_DIM_DEF, D_DIM_DEF, F_DIM_DEF, quantiles=quantiles)
        model = TimeSeriesTransformer(**config)
        assert model.quantiles == quantiles
        assert model.quantile_modeling is not None

    @pytest.mark.parametrize("integration_mode", ['add_to_encoder_input', 'add_to_decoder_input', 'none'])
    def test_model_instantiation_static_integration(self, integration_mode):
        """Tests different static integration modes."""
        config = self.get_model_config(
            S_DIM_DEF, D_DIM_DEF, F_DIM_DEF, 
            static_integration_mode=integration_mode)
        model = TimeSeriesTransformer(**config)
        assert model.static_integration_mode == integration_mode

    def test_model_instantiation_use_grn_for_static(self):
        """Tests using GRN for static feature processing."""
        config = self.get_model_config(
            S_DIM_DEF, D_DIM_DEF, F_DIM_DEF, 
            use_grn_for_static=True)
        model = TimeSeriesTransformer(**config)
        assert model.use_grn_for_static
        if S_DIM_DEF > 0:
            assert isinstance(
                model.static_processor, GatedResidualNetwork)

    # --- Forward Pass Tests ---
    @pytest.mark.parametrize("s_dim, d_dim, f_dim, quantiles_config, expected_output_last_dim", [
        (S_DIM_DEF, D_DIM_DEF, F_DIM_DEF, None, OUTPUT_DIM_DEF),  # All features, point
        (S_DIM_DEF, D_DIM_DEF, F_DIM_DEF, [0.1, 0.5, 0.9], 3),   # All features, quantile (3 quantiles)
        (0, D_DIM_DEF, F_DIM_DEF, None, OUTPUT_DIM_DEF),          # No static, point
        (S_DIM_DEF, D_DIM_DEF, 0, None, OUTPUT_DIM_DEF),          # No future known, point
        (0, D_DIM_DEF, 0, None, OUTPUT_DIM_DEF),                  # Dynamic only, point
        (0, D_DIM_DEF, 0, [0.25, 0.75], 2),                       # Dynamic only, quantile (2 quantiles)
    ])
    def test_forward_pass_shapes(
            self, s_dim, d_dim, f_dim, quantiles_config, expected_output_last_dim):
        """Tests the forward pass and output shapes for various configurations."""
        model_config = self.get_model_config(
            s_dim, d_dim, f_dim, quantiles=quantiles_config)
        model = TimeSeriesTransformer(**model_config)
        
        dummy_inputs, _ = generate_dummy_data(
            s_dim, d_dim, f_dim)
        
        # Build the model by calling it once if it's a subclassed Model
        # Or use Keras Input layers for a Functional API model for testing
        
        # Construct input tensors based on config
        input_specs = []
        if s_dim > 0:
            input_specs.append(tf.keras.Input(shape=(s_dim,), name="static_input_spec"))
        input_specs.append(tf.keras.Input(shape=(T_PAST, d_dim), name="dynamic_input_spec"))
        if f_dim > 0:
            input_specs.append(tf.keras.Input(shape=(T_HORIZON, f_dim), name="future_input_spec"))
        
        # Create a functional model wrapper for testing build and call
        test_functional_model = tf.keras.Model(inputs=input_specs, outputs=model(input_specs))
        
        # Now call with actual data
        predictions = test_functional_model(dummy_inputs)

        expected_shape = (BATCH_SIZE, T_HORIZON, expected_output_last_dim)
        if quantiles_config and OUTPUT_DIM_DEF > 1 : # (B, H, O, Q)
             expected_shape = (BATCH_SIZE, T_HORIZON, OUTPUT_DIM_DEF, expected_output_last_dim)
        
        assert predictions.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {predictions.shape}"

    def test_model_compilation(self):
        """Tests if the model compiles with a standard optimizer and loss."""
        config = self.get_model_config(S_DIM_DEF, D_DIM_DEF, F_DIM_DEF)
        model = TimeSeriesTransformer(**config)
        model.compile(optimizer='adam', loss='mse')
        assert model.optimizer is not None
        assert model.loss == 'mse' # Keras stores the string if string is passed

    def test_model_train_step(self):
        """Tests a single training step."""
        s_dim, d_dim, f_dim = S_DIM_DEF, D_DIM_DEF, F_DIM_DEF
        quantiles = [0.1, 0.5, 0.9]
        num_quantiles = len(quantiles)

        config = self.get_model_config(s_dim, d_dim, f_dim, quantiles=quantiles)
        model = TimeSeriesTransformer(**config)

        # Use a quantile loss for this test
        def simple_quantile_loss(y_true, y_pred):
            q_error = y_true - y_pred # y_true: (B,H,1), y_pred: (B,H,Q)
            q = tf.constant(quantiles, dtype=tf.float32)
            q = tf.reshape(q, [1, 1, num_quantiles])
            return tf.reduce_mean(tf.maximum(q * q_error, (q - 1) * q_error))

        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=simple_quantile_loss)

        dummy_model_inputs, y_true_raw = generate_dummy_data(s_dim, d_dim, f_dim)
        
        # Adjust y_true shape for quantile loss: (B, H, 1) if O_dim=1
        # and model output is (B, H, Q)
        if OUTPUT_DIM_DEF == 1 and quantiles:
            y_true_for_loss = np.expand_dims(y_true_raw, axis=-1) # (B, H, O_dim=1, 1 for broadcasting)
                                                                  # or (B,H,1) if y_true_raw is (B,H)
                                                                  # Assuming y_true_raw is (B,H,O_dim)
                                                                  # and output is (B,H,Q) for O_dim=1
                                                                  # Loss expects y_true (B,H,1), y_pred (B,H,Q)
            if y_true_for_loss.shape[-1] !=1: # if original y_true_raw had O_dim >1
                 y_true_for_loss= y_true_for_loss[...,:1] # Take first output dim for loss calc

        else:
            y_true_for_loss = y_true_raw

        history = model.fit(dummy_model_inputs, y_true_for_loss, batch_size=BATCH_SIZE, epochs=1, verbose=0)
        assert 'loss' in history.history
        assert history.history['loss'][0] is not None

    # Test for get_config and from_config (basic check)
    def test_model_serialization_config(self):
        """Tests get_config and from_config model serialization."""
        config_orig = self.get_model_config(S_DIM_DEF, D_DIM_DEF, F_DIM_DEF, quantiles=[0.2, 0.8])
        model1 = TimeSeriesTransformer(**config_orig)
        
        # Build the model (e.g., by creating dummy Input tensors and calling)
        s_in = tf.keras.Input(shape=(S_DIM_DEF,)) if S_DIM_DEF > 0 else None
        d_in = tf.keras.Input(shape=(T_PAST, D_DIM_DEF))
        f_in = tf.keras.Input(shape=(T_HORIZON, F_DIM_DEF)) if F_DIM_DEF > 0 else None
        
        call_inputs = []
        if s_in is not None: call_inputs.append(s_in)
        call_inputs.append(d_in)
        if f_in is not None: call_inputs.append(f_in)
        
        _ = model1(call_inputs) # Build model

        config = model1.get_config()
        
        # Ensure all __init__ params are in config
        for key in config_orig:
            if key not in ["name", "kwargs"]: # name is handled by Model itself
                 assert key in config, f"Key '{key}' missing in get_config()"
                 # Basic check, more detailed value checks can be added
                 if isinstance(config_orig[key], list): # like quantiles
                     assert config[key] == config_orig[key] 
                 elif not callable(config_orig[key]): # skip callables like activations for now
                     assert config[key] == config_orig[key]


        # Test reconstruction (requires all custom layers to be registered)
        try:
            model2 = TimeSeriesTransformer.from_config(config)
            # Call the reconstructed model to ensure it builds
            _ = model2(call_inputs)
            assert isinstance(model2, TimeSeriesTransformer)
        except Exception as e:
            pytest.fail(f"TimeSeriesTransformer.from_config failed: {e}")
            
if __name__=="__main__": 
    pytest.main([__file__])
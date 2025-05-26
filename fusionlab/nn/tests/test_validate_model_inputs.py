# test_validate_model_inputs.py

import re
import pytest
import numpy as np
import warnings

# --- Attempt to import function and dependencies ---
try:
    import tensorflow as tf
    from fusionlab.nn._tensor_validation import validate_model_inputs
    # For type hinting if KERAS_BACKEND is not fully set up for tests
    if hasattr(tf, 'Tensor'): # Check if tf.Tensor exists
        Tensor = tf.Tensor
    else: # Fallback for environments where tf.Tensor might not be typical
        class Tensor: pass
    FUSIONLAB_INSTALLED = True
except ImportError as e:
    print(f"Skipping validate_model_inputs tests: {e}")
    FUSIONLAB_INSTALLED = False
    class Tensor: pass # Dummy for collection
    def validate_model_inputs(*args, **kwargs):
        raise ImportError("validate_model_inputs not found")
# --- End Imports ---
pytestmark = pytest.mark.skipif(
    not FUSIONLAB_INSTALLED,
    reason="fusionlab.nn._tensor_validation.validate_model_inputs not found"
)

# --- Test Fixtures ---
B, T_past, H_out = 8, 12, 6 # Batch, Past Timesteps, Output Horizon
D_s, D_d, D_f, D_o = 2, 3, 2, 1 # Static, Dynamic, Future, Output Dims
T_future_span_ok = T_past + H_out # Future covers lookback + horizon
T_future_span_min = T_past      # Future only covers lookback

@pytest.fixture
def valid_static_data():
    return tf.random.normal((B, D_s), dtype=tf.float32)

@pytest.fixture
def valid_dynamic_data():
    return tf.random.normal((B, T_past, D_d), dtype=tf.float32)

@pytest.fixture
def valid_future_data_full_span():
    return tf.random.normal((B, T_future_span_ok, D_f), dtype=tf.float32)

@pytest.fixture
def valid_future_data_min_span():
    return tf.random.normal((B, T_future_span_min, D_f), dtype=tf.float32)

# --- Test Functions ---

def test_return_order_and_type(
    valid_static_data, valid_dynamic_data, valid_future_data_full_span
    ):
    """Test return order (S, D, F) and types."""
    s_p, d_p, f_p = validate_model_inputs(
        inputs=[valid_static_data, valid_dynamic_data,
                valid_future_data_full_span],
        static_input_dim=D_s, dynamic_input_dim=D_d,
        future_covariate_dim=D_f, forecast_horizon=H_out,
        mode='strict', verbose=0
    )
    assert isinstance(s_p, Tensor)
    assert isinstance(d_p, Tensor)
    assert isinstance(f_p, Tensor)
    assert s_p.dtype == tf.float32
    assert d_p.dtype == tf.float32
    assert f_p.dtype == tf.float32
    print("Return order and type: OK")

@pytest.mark.parametrize("mode_to_test", ['strict', 'soft'])
def test_basic_valid_inputs(
    valid_static_data, valid_dynamic_data, valid_future_data_full_span,
    mode_to_test
    ):
    """Test with all valid inputs for strict and soft modes."""
    try:
        s_p, d_p, f_p = validate_model_inputs(
            inputs=[valid_static_data, valid_dynamic_data,
                    valid_future_data_full_span],
            static_input_dim=D_s, dynamic_input_dim=D_d,
            future_covariate_dim=D_f, forecast_horizon=H_out,
            mode=mode_to_test, verbose=0
        )
        assert s_p is not None and d_p is not None and f_p is not None
        assert s_p.shape == (B, D_s)
        assert d_p.shape == (B, T_past, D_d)
        assert f_p.shape == (B, T_future_span_ok, D_f)
    except Exception as e:
        pytest.fail(f"Validation failed for mode='{mode_to_test}' "
                    f"with valid inputs: {e}")
    print(f"Basic valid inputs (mode='{mode_to_test}'): OK")

def test_deep_check_deprecation_mapping(
    valid_static_data, valid_dynamic_data, valid_future_data_min_span
    ):
    """Test deep_check deprecation and mapping to mode."""
    with pytest.warns(DeprecationWarning,):
        # deep_check=True should map to mode='strict'
        s_p, d_p, f_p = validate_model_inputs(
            inputs=[valid_static_data, valid_dynamic_data,
                    valid_future_data_min_span],
            static_input_dim=D_s, dynamic_input_dim=D_d,
            future_covariate_dim=D_f, forecast_horizon=H_out,
            deep_check=True, verbose=0
        )
        assert s_p is not None # Strict checks should pass

    with pytest.warns(DeprecationWarning,):
        # deep_check=False should map to mode='soft'
        # Test a case that would fail strict but pass soft (e.g., no feat_dim)
        s_p, d_p, f_p = validate_model_inputs(
            inputs=[valid_static_data, valid_dynamic_data,
                    valid_future_data_min_span],
            # No *_input_dim provided, should pass in soft mode
            forecast_horizon=H_out,
            deep_check=False, model_name='tft_flex', # Ensure soft mode logic
            verbose=0
        )
        assert s_p is not None
    print("deep_check deprecation mapping: OK")

@pytest.mark.parametrize("input_to_none_idx", [0, 1, 2]) # static, dynamic, future
def test_tft_flex_soft_mode_with_none_inputs(
    valid_static_data, valid_dynamic_data, valid_future_data_min_span,
    input_to_none_idx
    ):
    """Test tft_flex in soft mode with optional inputs being None."""
    inputs = [valid_static_data, valid_dynamic_data, valid_future_data_min_span]
    inputs[input_to_none_idx] = None

    # Only dynamic is truly required by TFTFlexible constructor
    if input_to_none_idx == 1 and inputs[1] is None: # If dynamic is None
        with pytest.raises(ValueError, match=re.escape( 
                "Parameter 'dynamic_p' is required and cannot be None."
                " Please provide a valid dynamic input.")):
             validate_model_inputs(
                inputs=inputs,
                dynamic_input_dim=D_d, # Still specify it as required by model
                model_name='tft_flex', mode='soft', verbose=0
            )
        return # Test ends here if dynamic is None

    s_p, d_p, f_p = validate_model_inputs(
        inputs=inputs,
        # Provide dims for non-None inputs for this test
        static_input_dim=D_s if inputs[0] is not None else None,
        dynamic_input_dim=D_d if inputs[1] is not None else None,
        future_covariate_dim=D_f if inputs[2] is not None else None,
        model_name='tft_flex', mode='soft', verbose=0
    )

    if input_to_none_idx == 0: assert s_p is None
    else: assert s_p is not None or inputs[0] is None
    if input_to_none_idx == 1: assert d_p is None # Should not happen due to check above
    else: assert d_p is not None
    if input_to_none_idx == 2: assert f_p is None
    else: assert f_p is not None or inputs[2] is None
    print(f"tft_flex soft mode with input {input_to_none_idx}=None: OK")


# --- Rank Validation Tests ---
def test_rank_validation_static(valid_dynamic_data, valid_future_data_min_span):
    """Test rank validation for static input."""
    # Correct 2D static
    validate_model_inputs(
        inputs=[tf.random.normal((B, D_s)), valid_dynamic_data,
                valid_future_data_min_span],
        static_input_dim=D_s, dynamic_input_dim=D_d,
        future_covariate_dim=D_f, mode='strict', verbose=0
    )
    # Incorrect 3D static
    with pytest.raises(tf.errors.InvalidArgumentError, match="Static input must be 2D"):
        validate_model_inputs(
            inputs=[tf.random.normal((B, T_past, D_s)), valid_dynamic_data,
                    valid_future_data_min_span],
            static_input_dim=D_s, dynamic_input_dim=D_d,
            future_covariate_dim=D_f, mode='strict', verbose=0
        )
    print("Rank validation static: OK")

def test_rank_validation_dynamic(valid_static_data, valid_future_data_min_span):
    """Test rank validation for dynamic input."""
    # Incorrect 2D dynamic
    with pytest.raises(tf.errors.InvalidArgumentError, match="Dynamic input must be 3D"):
        validate_model_inputs(
            inputs=[valid_static_data, tf.random.normal((B, D_d)),
                    valid_future_data_min_span],
            static_input_dim=D_s, dynamic_input_dim=D_d,
            future_covariate_dim=D_f, mode='strict', verbose=0
        )
    print("Rank validation dynamic: OK")

def test_rank_validation_future(valid_static_data, valid_dynamic_data):
    """Test rank validation for future input."""
    # Incorrect 2D future
    with pytest.raises(tf.errors.InvalidArgumentError, match="Future input must be 3D"):
        validate_model_inputs(
            inputs=[valid_static_data, valid_dynamic_data,
                    tf.random.normal((B, D_f))],
            static_input_dim=D_s, dynamic_input_dim=D_d,
            future_covariate_dim=D_f, mode='strict', verbose=0
        )
    print("Rank validation future: OK")

# --- Feature Dimension Validation Tests (Strict Mode) ---
def test_feature_dim_validation_strict(
    valid_static_data, valid_dynamic_data, valid_future_data_min_span
    ):
    """Test feature dimension validation in strict mode."""
    # Correct dims
    validate_model_inputs(
        inputs=[valid_static_data, valid_dynamic_data, valid_future_data_min_span],
        static_input_dim=D_s, dynamic_input_dim=D_d,
        future_covariate_dim=D_f, mode='strict', verbose=0
    )
    # Incorrect static_input_dim
    with pytest.raises(tf.errors.InvalidArgumentError,):# match="Static input last dimension mismatch"):
        validate_model_inputs(
            inputs=[valid_static_data, valid_dynamic_data, valid_future_data_min_span],
            static_input_dim=D_s + 1, dynamic_input_dim=D_d,
            future_covariate_dim=D_f, mode='strict', verbose=0
        )
    # Incorrect dynamic_input_dim
    with pytest.raises(
            tf.errors.InvalidArgumentError):# match="Dynamic input last dimension mismatch"):
        validate_model_inputs(
            inputs=[valid_static_data, valid_dynamic_data, valid_future_data_min_span],
            static_input_dim=D_s, dynamic_input_dim=D_d + 1,
            future_covariate_dim=D_f, mode='strict', verbose=0
        )
    print("Feature dimension validation (strict): OK")

# --- Batch Size Consistency Tests ---
def test_batch_size_consistency(
    valid_static_data, valid_dynamic_data, valid_future_data_min_span):
    """Test batch size consistency checks."""
    # Consistent
    validate_model_inputs(
        inputs=[valid_static_data, valid_dynamic_data, valid_future_data_min_span],
        static_input_dim=D_s, dynamic_input_dim=D_d,
        future_covariate_dim=D_f, mode='strict', verbose=0
    )
    # Inconsistent
    static_wrong_batch = tf.random.normal((B + 1, D_s))
    with pytest.raises(tf.errors.InvalidArgumentError, match="Inconsistent batch sizes"):
        validate_model_inputs(
            inputs=[static_wrong_batch, valid_dynamic_data, valid_future_data_min_span],
            static_input_dim=D_s, dynamic_input_dim=D_d,
            future_covariate_dim=D_f, mode='strict', verbose=0
        )
    print("Batch size consistency: OK")

# --- Time Dimension Consistency Tests ---
def test_time_dim_consistency(valid_static_data, valid_dynamic_data):
    """Test time dimension consistency (dynamic vs future)."""
    # Future span >= Dynamic span (OK)
    future_ok = tf.random.normal((B, T_past + 2, D_f)) # T_span_fut = T_past + 2
    validate_model_inputs(
        inputs=[valid_static_data, valid_dynamic_data, future_ok],
        static_input_dim=D_s, dynamic_input_dim=D_d,
        future_covariate_dim=D_f, mode='strict', verbose=0
    )
    # Future span < Dynamic span (Error)
    future_short = tf.random.normal((B, T_past - 1, D_f))
    with pytest.raises(tf.errors.InvalidArgumentError,
                       match=re.escape("Future input time span must be >= dynamic input")):
        validate_model_inputs(
            inputs=[valid_static_data, valid_dynamic_data, future_short],
            static_input_dim=D_s, dynamic_input_dim=D_d,
            future_covariate_dim=D_f, mode='strict', verbose=0
        )
    print("Time dimension consistency (dyn vs fut): OK")

def test_future_span_warning(
    valid_static_data, valid_dynamic_data, valid_future_data_min_span
    ):
    """Test warning for future span < dynamic_lookback + horizon."""
    # Here, valid_future_data_min_span has T_past steps.
    # If forecast_horizon > 0, then T_past < T_past + forecast_horizon.
    with pytest.warns(UserWarning, match="Future input time span .* is less than"):
        validate_model_inputs(
            inputs=[valid_static_data, valid_dynamic_data,
                    valid_future_data_min_span], # Future T = T_past
            static_input_dim=D_s, dynamic_input_dim=D_d,
            future_covariate_dim=D_f,
            forecast_horizon=H_out, # H_out > 0
            mode='strict', verbose=1 # Ensure warning is captured
        )
    print("Future span warning: OK")

# Allows running the tests directly if needed
if __name__=='__main__':
     pytest.main([__file__])


# test_prepare_model_inputs.py
import pytest
import numpy as np

import warnings
from typing import List, Optional, Union

# --- Attempt to import function and dependencies ---
try:
    import tensorflow as tf
    # Assuming prepare_model_inputs is in fusionlab.nn.utils
    # or fusionlab.nn._tensor_validation based on your project structure.
    # Adjust the import path as necessary.
    from fusionlab.nn.utils import prepare_model_inputs
    # For type hinting if KERAS_BACKEND is not fully set up for tests
    if hasattr(tf, 'Tensor'):
        Tensor = tf.Tensor
    else: # Fallback for environments where tf.Tensor might not be typical
        class Tensor: pass
    FUSIONLAB_INSTALLED = True
except ImportError as e:
    print(f"Skipping prepare_model_inputs tests: {e}")
    FUSIONLAB_INSTALLED = False
    class Tensor: pass # Dummy for collection
    def prepare_model_inputs(*args, **kwargs):
        raise ImportError("prepare_model_inputs not found")
# --- End Imports ---

pytestmark = pytest.mark.skipif(
    not FUSIONLAB_INSTALLED,
    reason="fusionlab.nn.utils.prepare_model_inputs not found"
)

# --- Test Fixtures and Constants ---
B, T_DYN, H_OUT = 4, 10, 3 # Batch, DynamicTimeSteps, Horizon
D_S, D_D, D_F = 2, 5, 3    # Static, Dynamic, Future Dims

@pytest.fixture
def dynamic_input_3d() -> tf.Tensor:
    """Valid 3D dynamic input."""
    return tf.random.normal((B, T_DYN, D_D), dtype=tf.float32)

@pytest.fixture
def dynamic_input_2d() -> tf.Tensor:
    """Valid 2D dynamic input (Batch, Features)."""
    return tf.random.normal((B, D_D), dtype=tf.float32)

@pytest.fixture
def static_input_valid() -> tf.Tensor:
    """Valid 2D static input."""
    return tf.random.normal((B, D_S), dtype=tf.float32)

@pytest.fixture
def future_input_valid() -> tf.Tensor:
    """Valid 3D future input."""
    # Future span matching dynamic + horizon
    return tf.random.normal((B, T_DYN + H_OUT, D_F), dtype=tf.float32)

@pytest.fixture
def future_input_short_span() -> tf.Tensor:
    """Valid 3D future input with time span matching dynamic_input."""
    return tf.random.normal((B, T_DYN, D_F), dtype=tf.float32)


# --- Test Functions ---

def test_return_structure_and_type(
    dynamic_input_3d, static_input_valid, future_input_valid
    ):
    """Test return is a list of 3, types, and float32 dtype."""
    result_list = prepare_model_inputs(
        dynamic_input=dynamic_input_3d,
        static_input=static_input_valid,
        future_input=future_input_valid,
        model_type='strict' # Does not affect return type structure
    )
    assert isinstance(result_list, list)
    assert len(result_list) == 3
    for item in result_list:
        assert isinstance(item, Tensor) # Should be tf.Tensor
        assert item.dtype == tf.float32
    print("Return structure and type: OK")

# --- Tests for 'strict' mode ---
def test_strict_mode_all_inputs_provided(
    dynamic_input_3d, static_input_valid, future_input_valid
    ):
    """Strict mode: all inputs provided, should pass through."""
    s_p, d_p, f_p = prepare_model_inputs(
        dynamic_input=dynamic_input_3d,
        static_input=static_input_valid,
        future_input=future_input_valid,
        model_type='strict', verbose=0
    )
    assert tf.reduce_all(tf.equal(s_p, static_input_valid))
    assert tf.reduce_all(tf.equal(d_p, dynamic_input_3d))
    assert tf.reduce_all(tf.equal(f_p, future_input_valid))
    print("Strict mode (all inputs): OK")

def test_strict_mode_static_none(dynamic_input_3d, future_input_valid):
    """Strict mode: static is None, dummy static (B,0) created."""
    s_p, d_p, f_p = prepare_model_inputs(
        dynamic_input=dynamic_input_3d,
        static_input=None,
        future_input=future_input_valid,
        model_type='strict', verbose=1
    )
    assert s_p is not None
    assert s_p.shape.as_list() == [B, 0]
    assert tf.reduce_all(tf.equal(d_p, dynamic_input_3d))
    assert tf.reduce_all(tf.equal(f_p, future_input_valid))
    print("Strict mode (static=None): OK")

def test_strict_mode_future_none_no_horizon(dynamic_input_3d, static_input_valid):
    """Strict mode: future is None, no horizon, dummy future (B,T_dyn,0) created."""
    s_p, d_p, f_p = prepare_model_inputs(
        dynamic_input=dynamic_input_3d,
        static_input=static_input_valid,
        future_input=None,
        model_type='strict', forecast_horizon=None, verbose=1
    )
    assert f_p is not None
    assert f_p.shape.as_list() == [B, T_DYN, 0]
    print("Strict mode (future=None, no horizon): OK")

def test_strict_mode_future_none_with_horizon(dynamic_input_3d, static_input_valid):
    """Strict mode: future is None, with horizon, dummy future (B,T_dyn+H,0) created."""
    s_p, d_p, f_p = prepare_model_inputs(
        dynamic_input=dynamic_input_3d,
        static_input=static_input_valid,
        future_input=None,
        model_type='strict', forecast_horizon=H_OUT, verbose=1
    )
    assert f_p is not None
    assert f_p.shape.as_list() == [B, T_DYN + H_OUT, 0]
    print("Strict mode (future=None, with horizon): OK")

def test_strict_mode_static_and_future_none(dynamic_input_3d):
    """Strict mode: static and future are None, dummies created."""
    s_p, d_p, f_p = prepare_model_inputs(
        dynamic_input=dynamic_input_3d,
        static_input=None,
        future_input=None,
        model_type='strict', forecast_horizon=H_OUT, verbose=1
    )
    assert s_p is not None and s_p.shape.as_list() == [B, 0]
    assert d_p is not None
    assert f_p is not None and f_p.shape.as_list() == [B, T_DYN + H_OUT, 0]
    print("Strict mode (static=None, future=None): OK")

# --- Tests for 'flexible' mode ---
def test_flexible_mode_all_inputs_provided(
    dynamic_input_3d, static_input_valid, future_input_valid
    ):
    """Flexible mode: all inputs provided, should pass through."""
    s_p, d_p, f_p = prepare_model_inputs(
        dynamic_input=dynamic_input_3d,
        static_input=static_input_valid,
        future_input=future_input_valid,
        model_type='flexible', verbose=0
    )
    assert tf.reduce_all(tf.equal(s_p, static_input_valid))
    assert tf.reduce_all(tf.equal(d_p, dynamic_input_3d))
    assert tf.reduce_all(tf.equal(f_p, future_input_valid))
    print("Flexible mode (all inputs): OK")

def test_flexible_mode_static_none(dynamic_input_3d, future_input_valid):
    """Flexible mode: static is None, should return None for static."""
    s_p, d_p, f_p = prepare_model_inputs(
        dynamic_input=dynamic_input_3d,
        static_input=None,
        future_input=future_input_valid,
        model_type='flexible', verbose=1
    )
    assert s_p is None
    assert tf.reduce_all(tf.equal(d_p, dynamic_input_3d))
    assert tf.reduce_all(tf.equal(f_p, future_input_valid))
    print("Flexible mode (static=None): OK")

def test_flexible_mode_future_none(dynamic_input_3d, static_input_valid):
    """Flexible mode: future is None, should return None for future."""
    s_p, d_p, f_p = prepare_model_inputs(
        dynamic_input=dynamic_input_3d,
        static_input=static_input_valid,
        future_input=None,
        model_type='flexible', verbose=1
    )
    assert tf.reduce_all(tf.equal(s_p, static_input_valid))
    assert tf.reduce_all(tf.equal(d_p, dynamic_input_3d))
    assert f_p is None
    print("Flexible mode (future=None): OK")

def test_flexible_mode_static_and_future_none(dynamic_input_3d):
    """Flexible mode: static and future are None, returns Nones."""
    s_p, d_p, f_p = prepare_model_inputs(
        dynamic_input=dynamic_input_3d,
        static_input=None,
        future_input=None,
        model_type='flexible', verbose=1
    )
    assert s_p is None
    assert d_p is not None
    assert f_p is None
    print("Flexible mode (static=None, future=None): OK")

# --- Error Handling and Edge Cases ---
def test_error_dynamic_input_none():
    """Test ValueError if dynamic_input is None."""
    with pytest.raises(ValueError, match="`dynamic_input` is required"):
        prepare_model_inputs(dynamic_input=None)
    print("Error dynamic_input=None: OK")

def test_error_invalid_ranks(
    dynamic_input_3d, static_input_valid, future_input_valid
    ):
    """Test ValueError for incorrect input ranks."""
    # Static not 2D
    with pytest.raises(ValueError, match="static_input, if provided, must be 2D"):
        prepare_model_inputs(dynamic_input_3d, static_input=tf.random.normal((B, D_S, 1)))
    # Dynamic not 2D/3D (e.g. 1D)
    with pytest.raises(ValueError, match="dynamic_input must be at least 2D"):
        prepare_model_inputs(dynamic_input=tf.random.normal((B,)))
    # Future not 3D
    with pytest.raises(ValueError, match="future_input, if provided, must be 3D"):
        prepare_model_inputs(dynamic_input_3d, future_input=tf.random.normal((B, D_F)))
    print("Error invalid ranks: OK")

def test_warning_dynamic_2d_input(dynamic_input_2d):
    """Test warning if dynamic_input is 2D."""
    with pytest.warns(UserWarning, match="dynamic_input was 2D"):
        s_p, d_p, f_p = prepare_model_inputs(
            dynamic_input=dynamic_input_2d, model_type='flexible'
            )
    assert d_p is not None
    # Check if past_time_steps was set to 1 for dummy future in strict mode
    s_p_strict, _, f_p_strict = prepare_model_inputs(
        dynamic_input=dynamic_input_2d, model_type='strict', 
        forecast_horizon=H_OUT
    )
    assert f_p_strict.shape[1] == 1 + H_OUT # T_past (1) + H_OUT
    print("Warning dynamic_input 2D: OK")

@pytest.mark.skip ("The error to catch is handled with the @validate_params decorators." 
                   " The ValueError would not be reached... ")
def test_error_invalid_model_type(dynamic_input_3d):
    """Test ValueError for invalid model_type."""
    with pytest.raises(ValueError, match="Invalid `model_type`"):
        prepare_model_inputs(dynamic_input_3d, model_type='unknown')
    print("Error invalid model_type: OK")

def test_numpy_inputs(dynamic_input_3d):
    """Test if NumPy array inputs are correctly converted to tf.Tensor."""
    dyn_np = np.random.rand(B, T_DYN, D_D).astype(np.float32)
    stat_np = np.random.rand(B, D_S).astype(np.float32)
    fut_np = np.random.rand(B, T_DYN + H_OUT, D_F).astype(np.float32)

    s_p, d_p, f_p = prepare_model_inputs(
        dynamic_input=dyn_np, static_input=stat_np, future_input=fut_np,
        model_type='strict'
    )
    assert isinstance(s_p, Tensor) and s_p.dtype == tf.float32
    assert isinstance(d_p, Tensor) and d_p.dtype == tf.float32
    assert isinstance(f_p, Tensor) and f_p.dtype == tf.float32
    print("NumPy inputs conversion: OK")

# Allows running the tests directly if needed
if __name__=='__main__':
     pytest.main([__file__])


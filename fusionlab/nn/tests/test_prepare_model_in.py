import pytest
import numpy as np

try: 
    import tensorflow as tf
    from fusionlab.nn.utils import prepare_model_inputs_in
    from sklearn.utils._param_validation import InvalidParameterError 
    from fusionlab.nn import KERAS_BACKEND 
except Exception as e : 
    print(f"Skipping combine_temporal_inputs_for_lstm tests due to"
          f" import error: {e}")
    KERAS_BACKEND = False

pytestmark = pytest.mark.skipif(
    not KERAS_BACKEND, reason="TensorFlow/Keras backend not available"
)

def make_dummy_dynamic(batch=2, time=3, features=4):
    # 3D array: (batch, time, features)
    return np.random.randn(batch, time, features).astype(np.float32)

def make_dummy_static(batch=2, features=5):
    # 2D array: (batch, features)
    return np.random.randn(batch, features).astype(np.float32)

def make_dummy_future(batch=2, horizon=6, features=3):
    # 3D array: (batch, horizon, features)
    return np.random.randn(batch, horizon, features).astype(np.float32)

def test_dynamic_none_raises():
    with pytest.raises(ValueError) as exc:
        prepare_model_inputs_in(None)
    assert "`dynamic_input` is required" in str(exc.value)

def test_strict_mode_dummy_static_and_future_with_horizon():
    dyn = make_dummy_dynamic(batch=4, time=7, features=2)
    # both static and future omitted
    s, d, f = prepare_model_inputs_in(
        dyn,
        static_input=None,
        future_input=None,
        model_type="strict",
        forecast_horizon=5,
        verbose=0
    )
    # static dummy should have shape (batch, 0)
    assert isinstance(s, tf.Tensor)
    assert tuple(s.shape.as_list()) == (4, 0)
    # dynamic should pass through
    assert isinstance(d, tf.Tensor)
    assert tuple(d.shape.as_list()) == (4, 7, 2)
    # future dummy length = past_time_steps + forecast_horizon = 7 + 5 = 12
    assert isinstance(f, tf.Tensor)
    assert tuple(f.shape.as_list()) == (4, 12, 0)

def test_strict_mode_dummy_future_without_horizon():
    dyn = make_dummy_dynamic(batch=3, time=8, features=1)
    # provide a real static, omit future
    static = make_dummy_static(batch=3, features=10)
    s, d, f = prepare_model_inputs_in(
        dyn,
        static_input=static,
        future_input=None,
        model_type="strict",
        forecast_horizon=None,
        verbose=0
    )
    # static should pass through unchanged
    assert isinstance(s, tf.Tensor)
    tf.debugging.assert_equal(s, tf.convert_to_tensor(static))
    # future dummy length should equal past_time_steps = 8
    assert tuple(f.shape.as_list()) == (3, 8, 0)

def test_flexible_mode_pass_through():
    dyn = make_dummy_dynamic(batch=5, time=4, features=3)
    static = make_dummy_static(batch=5, features=2)
    future = make_dummy_future(batch=5, horizon=6, features=7)
    s, d, f = prepare_model_inputs_in(
        dyn,
        static_input=static,
        future_input=future,
        model_type="flexible",
        forecast_horizon=2,
        verbose=0
    )
    # everything should be exactly what was passed in
    tf.debugging.assert_equal(s, tf.convert_to_tensor(static))
    tf.debugging.assert_equal(d, tf.convert_to_tensor(dyn))
    tf.debugging.assert_equal(f, tf.convert_to_tensor(future))

@pytest.mark.parametrize("bad_horizon", [-1, -10])
def test_invalid_forecast_horizon_raises(bad_horizon):
    dyn = make_dummy_dynamic(batch=1, time=2, features=2)
    with pytest.raises(InvalidParameterError):
        prepare_model_inputs_in(
            dyn,
            static_input=None,
            future_input=None,
            model_type="strict",
            forecast_horizon=bad_horizon
        )

def test_invalid_model_type_raises():
    dyn = make_dummy_dynamic(batch=2, time=2, features=2)
    with pytest.raises(ValueError) as exc:
        prepare_model_inputs_in(
            dyn,
            static_input=None,
            future_input=None,
            model_type="not_a_mode",
            forecast_horizon=None
        )
    assert "The 'model_type' parameter" in str(exc.value)
    
if __name__ =='__main__': # pragma : no cover 
    pytest.main( [__file__,  "--maxfail=1 ", "--disable-warnings",  "-q"])
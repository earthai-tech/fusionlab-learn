import pytest
import numpy as np

try: 
    import tensorflow as tf
    from fusionlab.nn.pinn.utils import extract_txy 
    from fusionlab.nn import KERAS_BACKEND 
except Exception as e : 
    print(f"Skipping combine_temporal_inputs_for_lstm tests due to"
          f" import error: {e}")
    KERAS_BACKEND = False

pytestmark = pytest.mark.skipif(
    not KERAS_BACKEND, reason="TensorFlow/Keras backend not available"
)
# -- Fixtures ------------------------------------------------------------

@pytest.fixture(params=[2, 3])
def sample_coords(request):
    """Generate a small random 2D or 3D coords tensor/array."""
    ndim = request.param
    shape = (4, 3) if ndim == 2 else (4, 2, 3)
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    return data

# -- Tests for array/tensor inputs --------------------------------------

def test_array_input_default(sample_coords):
    coords = sample_coords
    t, x, y = extract_txy(coords)

    # Check shapes: same rank, last dim = 1
    assert t.shape[:-1] == coords.shape[:-1]
    assert t.shape[-1] == 1
    # Values correspond to slices
    np.testing.assert_array_equal(
        t.numpy() if hasattr(t, 'numpy') else t,
        coords[..., 0:1]
    )
    np.testing.assert_array_equal(
        x.numpy() if hasattr(x, 'numpy') else x,
        coords[..., 1:2]
    )
    np.testing.assert_array_equal(
        y.numpy() if hasattr(y, 'numpy') else y,
        coords[..., 2:3]
    )

def test_array_input_expect_dim_2d_raises_for_3d():
    arr3 = np.zeros((5, 2, 3))
    with pytest.raises(ValueError):
        extract_txy(arr3, expect_dim='2d')

def test_array_input_expect_dim_3d_expands_2d():
    arr2 = np.zeros((7, 3))
    t, x, y = extract_txy(arr2, expect_dim='3d')
    # should expand middle axis
    assert t.shape == (7, 1, 1)

def test_array_input_expect_dim_3d_only_raises_for_2d():
    arr2 = np.zeros((7, 3))
    with pytest.raises(ValueError):
        extract_txy(arr2, expect_dim='3d_only')

# -- Tests for dict inputs -----------------------------------------------

def test_dict_coords_key(sample_coords):
    coords = sample_coords
    inp = {'coords': coords}
    t, x, y = extract_txy(inp)
    # same checks as for array
    assert t.shape[:-1] == coords.shape[:-1]
    assert x.shape == t.shape
    assert y.shape == t.shape

def test_dict_separate_keys_2d():
    # t, x, y as separate 2D arrays
    t_arr = np.arange(8).reshape((4, 2))
    x_arr = t_arr + 10
    y_arr = t_arr + 20
    inp = {'t': t_arr, 'x': x_arr, 'y': y_arr}
    tout, xout, yout = extract_txy(inp)
    np.testing.assert_array_equal(
        tout.numpy() if hasattr(tout, 'numpy') else tout, t_arr#[..., :1]
    )
    np.testing.assert_array_equal(
        xout.numpy() if hasattr(xout, 'numpy') else xout, x_arr#[..., :1].astype(float)
    )
    np.testing.assert_array_equal(
        yout.numpy() if hasattr(yout, 'numpy') else yout, y_arr#[..., :1]
    )

def test_dict_missing_keys_raises():
    with pytest.raises(ValueError):
        extract_txy({'foo': np.zeros((3,3))})

# -- Tests for invalid types ---------------------------------------------

def test_invalid_input_type():
    with pytest.raises(TypeError):
        extract_txy(12345)

def test_bad_slice_dimension():
    # last dim < 3
    arr = np.zeros((2,2))
    with pytest.raises(ValueError):
        extract_txy(arr)

# -- Tests for custom slice map ------------------------------------------

def test_custom_slice_map():
    # create array where t/x/y are in reversed order
    coords = np.stack([
        np.ones((3,3))*2,  # placeholder
        np.arange(9).reshape(3,3),  # t
        np.arange(9).reshape(3,3)+10,  # x
        np.arange(9).reshape(3,3)+20   # y
    ], axis=-1)
    # slice map tells where t,x,y live
    smap = {'t':1, 'x':2, 'y':3}
    t, x, y = extract_txy(coords, coord_slice_map=smap)
    np.testing.assert_array_equal(
        t.numpy() if hasattr(t, 'numpy') else t, coords[...,1:2]
    )
    np.testing.assert_array_equal(
        x.numpy() if hasattr(x, 'numpy') else x, coords[...,2:3]
    )
    np.testing.assert_array_equal(
        y.numpy() if hasattr(y, 'numpy') else y, coords[...,3:4]
    )

if __name__ =='__main__': # pragma : no cover 
    pytest.main( [__file__,  "--maxfail=1 ", "--disable-warnings",  "-q"])

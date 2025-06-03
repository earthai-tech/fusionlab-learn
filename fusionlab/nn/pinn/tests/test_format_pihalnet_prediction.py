import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from unittest.mock import MagicMock # For mocking model and scalers
from unittest.mock import patch

try:
    from fusionlab.nn.pinn.utils import format_pihalnet_predictions
    from fusionlab.nn import KERAS_BACKEND, KERAS_DEPS # For Tensor type hint
    if KERAS_BACKEND:
        import tensorflow as tf
        Tensor = KERAS_DEPS.Tensor
        Model = KERAS_DEPS.Model
    else:
        class Tensor: pass # Dummy
        class Model: pass # Dummy
        tf = None

    FUSIONLAB_AVAILABLE = True
except ImportError as e:
    print(f"Could not import fusionlab PINN components for formatting tests: {e}")
    FUSIONLAB_AVAILABLE = False
    class Tensor: pass # Dummy
    class Model: pass # Dummy
    tf = None # type: ignore
    def format_pihalnet_predictions(*args, **kwargs):
        return pd.DataFrame()

HAS_COVERAGE_SCORE =False 
try: 
    from fusionlab.metrics import coverage_score # Noqa
    HAS_COVERAGE_SCORE =True 
except : 
    pass 
# --- Test Parameters ---
B = 2   # Batch size for tests
H = 3   # Forecast Horizon
O_S = 1 # Output dimension for subsidence
O_G = 1 # Output dimension for GWL

# --- Test Fixtures ---

@pytest.fixture
def dummy_pihalnet_outputs_point():
    """Generates dummy PIHALNet point forecast outputs."""
    return {
        "subs_pred": tf.constant(np.random.rand(B, H, O_S), dtype=tf.float32) \
                     if tf else np.random.rand(B, H, O_S).astype(np.float32),
        "gwl_pred": tf.constant(np.random.rand(B, H, O_G), dtype=tf.float32) \
                    if tf else np.random.rand(B, H, O_G).astype(np.float32),
        "pde_residual": tf.constant(np.random.rand(B, H -1 if H > 1 else H , O_S), dtype=tf.float32) \
                        if tf else np.random.rand(B, H-1 if H > 1 else H, O_S).astype(np.float32)
    }

@pytest.fixture
def dummy_pihalnet_outputs_quantile(quantiles=[0.1, 0.5, 0.9]):
    """Generates dummy PIHALNet quantile forecast outputs."""
    num_q = len(quantiles)
    return {
        "subs_pred": tf.constant(np.random.rand(B, H, num_q, O_S), dtype=tf.float32) \
                     if tf else np.random.rand(B, H, num_q, O_S).astype(np.float32),
        "gwl_pred": tf.constant(np.random.rand(B, H, num_q, O_G), dtype=tf.float32) \
                    if tf else np.random.rand(B, H, num_q, O_G).astype(np.float32),
        "pde_residual": tf.constant(np.random.rand(B, H -1 if H > 1 else H, O_S), dtype=tf.float32) \
                        if tf else np.random.rand(B, H-1 if H > 1 else H, O_S).astype(np.float32)
    }

@pytest.fixture
def dummy_y_true_dict():
    """Generates dummy true target values."""
    return {
        'subsidence': np.random.rand(B, H, O_S).astype(np.float32),
        'gwl': np.random.rand(B, H, O_G).astype(np.float32)
    }

@pytest.fixture
def dummy_model_inputs():
    """Generates dummy model inputs, including coordinates."""
    return {
        'coords': np.random.rand(B, H, 3).astype(np.float32), # t, x, y
        # Add other features if format_pihalnet_predictions uses them
    }

@pytest.fixture
def dummy_ids_dataframe():
    return pd.DataFrame({
        'site_id': [f'Site_{i+1}' for i in range(B)],
        'region': [f'Region_{ (i % 2) + 1}' for i in range(B)]
    })

@pytest.fixture
def dummy_ids_numpy():
    return np.array([[100 + i, 2000 + i] for i in range(B)], dtype=np.float32)


# --- Pytest Test Functions ---

@pytest.mark.skipif(not FUSIONLAB_AVAILABLE, reason="fusionlab components not available")
def test_format_point_forecasts_all_outputs(
    dummy_pihalnet_outputs_point, dummy_y_true_dict, dummy_model_inputs
    ):
    """Test formatting for point forecasts with all outputs and info."""
    df = format_pihalnet_predictions(
        pihalnet_outputs=dummy_pihalnet_outputs_point,
        y_true_dict=dummy_y_true_dict,
        model_inputs=dummy_model_inputs,
        forecast_horizon=H,
        output_dims={'subs_pred': O_S, 'gwl_pred': O_G},
        include_coords_in_df=True,
        verbose=1
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == B * H
    expected_cols = [
        'sample_idx', 'forecast_step',
        'coord_t', 'coord_x', 'coord_y',
        'subsidence_pred', 'subsidence_actual',
        'gwl_pred', 'gwl_actual'
    ]
    for col in expected_cols:
        assert col in df.columns

@pytest.mark.skipif(not FUSIONLAB_AVAILABLE, reason="fusionlab components not available")
def test_format_quantile_forecasts(
    quantiles=[0.1, 0.5, 0.9] # Define fixture or pass directly
    ):
    """Test formatting for quantile forecasts."""
    # Generate quantile outputs within the test
    num_q = len(quantiles)
    outputs_quantile = {
        "subs_pred": np.random.rand(B, H, num_q, O_S).astype(np.float32),
        "gwl_pred": np.random.rand(B, H, num_q, O_G).astype(np.float32)
    }
    y_true = {
        'subsidence': np.random.rand(B, H, O_S).astype(np.float32),
        'gwl': np.random.rand(B, H, O_G).astype(np.float32)
    }

    df = format_pihalnet_predictions(
        pihalnet_outputs=outputs_quantile,
        y_true_dict=y_true,
        quantiles=quantiles,
        forecast_horizon=H,
        output_dims={'subs_pred': O_S, 'gwl_pred': O_G},
        verbose=1
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == B * H
    assert 'subsidence_q10' in df.columns
    assert 'subsidence_q50' in df.columns
    assert 'subsidence_q90' in df.columns
    assert 'gwl_q50' in df.columns
    assert 'subsidence_actual' in df.columns

@pytest.mark.skipif(not FUSIONLAB_AVAILABLE, reason="fusionlab components not available")
def test_format_no_gwl_output(dummy_pihalnet_outputs_point):
    """Test formatting when include_gwl_in_df is False."""
    df = format_pihalnet_predictions(
        pihalnet_outputs=dummy_pihalnet_outputs_point,
        include_gwl_in_df=False,
        forecast_horizon=H,
        output_dims={'subs_pred': O_S, 'gwl_pred': O_G} # Still provide full output_dims
    )
    assert 'gwl_pred' not in df.columns
    assert 'subsidence_pred' in df.columns

@pytest.mark.skipif(not FUSIONLAB_AVAILABLE, reason="fusionlab components not available")
def test_format_no_coords_output(dummy_pihalnet_outputs_point):
    """Test formatting when include_coords_in_df is False."""
    df = format_pihalnet_predictions(
        pihalnet_outputs=dummy_pihalnet_outputs_point,
        include_coords_in_df=False,
        forecast_horizon=H,
        model_inputs=None # No coords to add
    )
    assert 'coord_t' not in df.columns
    assert 'coord_x' not in df.columns
    assert 'coord_y' not in df.columns

@pytest.mark.skipif(not FUSIONLAB_AVAILABLE, reason="fusionlab components not available")
def test_format_with_ids_dataframe(
    dummy_pihalnet_outputs_point, dummy_ids_dataframe
    ):
    """Test adding static IDs from a DataFrame."""
    df = format_pihalnet_predictions(
        pihalnet_outputs=dummy_pihalnet_outputs_point,
        ids_data_array=dummy_ids_dataframe,
        ids_cols=['site_id', 'region'],
        forecast_horizon=H
    )
    assert 'site_id' in df.columns
    assert 'region' in df.columns
    assert len(df) == B * H
    assert df['site_id'].iloc[0] == 'Site_1'
    assert df['site_id'].iloc[H] == 'Site_1' # Check repetition

@pytest.mark.skipif(not FUSIONLAB_AVAILABLE, reason="fusionlab components not available")
def test_format_with_ids_numpy(
    dummy_pihalnet_outputs_point, dummy_ids_numpy
    ):
    """Test adding static IDs from a NumPy array."""
    df = format_pihalnet_predictions(
        pihalnet_outputs=dummy_pihalnet_outputs_point,
        ids_data_array=dummy_ids_numpy,
        ids_cols=['id_val1', 'id_val2'], # Provide names for numpy cols
        # ids_cols_indices=[0,1] # Not needed if ids_data_array is already sliced
        forecast_horizon=H
    )
    assert 'id_val1' in df.columns
    assert 'id_val2' in df.columns
    assert df['id_val1'].iloc[0] == 100.0

@pytest.mark.skipif(not FUSIONLAB_AVAILABLE , reason="Keras or Tuner not available")
def test_format_model_prediction_generation(dummy_model_inputs):
    """Test generating predictions if outputs are not provided."""
    # Mock PIHALNet model
    mock_model = MagicMock(spec=Model) # Use spec from Keras if TF available
    
    # Define what model.predict should return (matching pihalnet_outputs structure)
    mock_model.predict.return_value = {
        "subs_pred": np.random.rand(B, H, O_S).astype(np.float32),
        "gwl_pred": np.random.rand(B, H, O_G).astype(np.float32),
        "pde_residual": np.random.rand(B, H-1 if H>1 else H, O_S).astype(np.float32)
    }

    df = format_pihalnet_predictions(
        model=mock_model,
        model_inputs=dummy_model_inputs,
        forecast_horizon=H
    )
    mock_model.predict.assert_called_once_with(dummy_model_inputs, verbose=0)
    assert 'subsidence_pred' in df.columns
    assert len(df) == B * H

@pytest.mark.skipif(not FUSIONLAB_AVAILABLE, reason="fusionlab components not available")
def test_format_missing_predictions_and_model():
    """Test error when neither predictions nor model is provided."""
    with pytest.raises(ValueError, match="If 'pihalnet_outputs' is None, both 'model'"):
        format_pihalnet_predictions(forecast_horizon=H)

@pytest.mark.skipif(not FUSIONLAB_AVAILABLE, reason="fusionlab components not available")
def test_format_with_inverse_scaling_and_coverage(dummy_pihalnet_outputs_quantile):
    """Test inverse scaling and coverage score (conceptual)."""
    # This test is more complex due to mocking scalers and coverage_score
    
    quantiles = [0.1, 0.5, 0.9] # Must match dummy_pihalnet_outputs_quantile
    y_true = {
        'subsidence': np.random.rand(B, H, O_S).astype(np.float32) * 10, # Unscaled
        'gwl': np.random.rand(B, H, O_G).astype(np.float32) * 5        # Unscaled
    }
    
    # Mock scalers
    mock_scaler_subs = MagicMock()
    mock_scaler_subs.inverse_transform.side_effect = lambda x: x * 10 # Simple unscaling
    mock_scaler_gwl = MagicMock()
    mock_scaler_gwl.inverse_transform.side_effect = lambda x: x * 5

    scaler_info_dict = {
        'subsidence': {'scaler': mock_scaler_subs, 
                       'all_features': ['subsidence'], 'idx': 0},
        'gwl': {'scaler': mock_scaler_gwl,
                'all_features': ['gwl'], 'idx': 0}
    }

    # Mock coverage_score if its import is conditional
    if not HAS_COVERAGE_SCORE:
        coverage_patch_target = 'fusionlab.metrics.coverage_score' 
    else: 
        coverage_patch_target="fusionlab.nn.pinn.utils.coverage_score"
        # Adjust if coverage_score is imported differently in your utils

    with patch(coverage_patch_target, return_value=0.85) as mock_cov_score:
        df = format_pihalnet_predictions(
            pihalnet_outputs=dummy_pihalnet_outputs_quantile,
            y_true_dict=y_true,
            quantiles=quantiles,
            forecast_horizon=H,
            target_mapping={'subs_pred': 'subsidence', 'gwl_pred': 'gwl'},
            scaler_info=scaler_info_dict,
            evaluate_coverage=True,
            coverage_quantile_indices=(0, 2) # Using q10 and q90
        )
        
    
    assert 'subsidence_q50' in df.columns
    # Check if inverse transform was called
    mock_scaler_subs.inverse_transform.assert_called()
    mock_scaler_gwl.inverse_transform.assert_called()
    
    if HAS_COVERAGE_SCORE or hasattr(mock_cov_score, 'assert_called'): # Check if patched
        mock_cov_score.assert_called()  
        # mock_cov_score.assert_called_once()
        # You could also add df.attrs check if you store coverage score there

# This allows running the test file directly
if __name__ == '__main__':
    if FUSIONLAB_AVAILABLE:
        # Construct the command to run pytest on this file
        # You might need to adjust the path depending on where you run from
        # and how your project is structured for pytest discovery.
        pytest_args = [__file__, "-vv"]
        # Example: add -k "test_name" to run a specific test
        # pytest_args.extend(["-k", "test_format_with_inverse_scaling_and_coverage"])
        pytest.main(pytest_args)
    else:
        print("Skipping format_pihalnet_predictions tests: "
              "fusionlab components not available.")
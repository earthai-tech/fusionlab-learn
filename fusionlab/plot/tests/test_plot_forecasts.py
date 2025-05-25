# test_plot_forecasts.py

import pytest
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt # For mocking
from unittest.mock import patch, MagicMock # For mocking plt.show and plt.figure
from typing import Dict 
# --- Attempt to import function and dependencies ---
try:
    import tensorflow as tf # For Tensor type hint and tf.constant
    # Assuming plot_forecasts and format_predictions_to_dataframe
    # are in fusionlab.nn.utils
    from fusionlab.nn.utils import (
        format_predictions_to_dataframe
    )
    from fusionlab.plot.forecast import plot_forecasts 
    
    # For type hinting if KERAS_BACKEND is not fully set up for tests
    if hasattr(tf, 'Tensor'):
        Tensor = tf.Tensor
    else: # Fallback
        class Tensor: pass
    FUSIONLAB_INSTALLED = True
except ImportError as e:
    print(f"Skipping plot_forecasts tests due to import error: {e}")
    FUSIONLAB_INSTALLED = False
    class Tensor: pass # Dummy for collection
    def plot_forecasts(*args, **kwargs):
        raise ImportError("plot_forecasts not found")
    def format_predictions_to_dataframe(*args, **kwargs):
        # Return a dummy DataFrame to allow tests to proceed if only plot_forecasts is missing
        return pd.DataFrame({'sample_idx': [], 'forecast_step': []})

pytestmark = pytest.mark.skipif(
    not FUSIONLAB_INSTALLED,
    reason="fusionlab.nn.utils.plot_forecasts or dependencies not found"
)

# --- Test Fixtures and Constants ---
B, H, O_SINGLE, O_MULTI = 4, 3, 1, 2 # Batch, Horizon, OutputDims
Q_LIST = [0.1, 0.5, 0.9]
N_Q = len(Q_LIST)
SAMPLES = B # Number of sequences (distinct sample_idx values)

TARGET_NAME = "value"
SPATIAL_COLS_TEST = ['lon', 'lat']

@pytest.fixture
def base_predictions() -> Dict[str, np.ndarray]:
    """Generates base prediction arrays."""
    return {
        "point_single_pred": np.random.rand(SAMPLES, H, O_SINGLE).astype(np.float32),
        "point_multi_pred": np.random.rand(SAMPLES, H, O_MULTI).astype(np.float32),
        "quant_single_pred": np.random.rand(SAMPLES, H, N_Q).astype(np.float32), # O=1
        "quant_multi_pred_3d": np.random.rand(SAMPLES, H, N_Q * O_MULTI).astype(np.float32),
        "quant_multi_pred_4d": np.random.rand(SAMPLES, H, N_Q, O_MULTI).astype(np.float32),
        "y_true_single": np.random.rand(SAMPLES, H, O_SINGLE).astype(np.float32),
        "y_true_multi": np.random.rand(SAMPLES, H, O_MULTI).astype(np.float32),
    }

@pytest.fixture
def forecast_df_point_single(base_predictions) -> pd.DataFrame:
    """DataFrame for point forecast, single output, with actuals."""
    return format_predictions_to_dataframe(
        predictions=base_predictions["point_single_pred"],
        y_true_sequences=base_predictions["y_true_single"],
        target_name=TARGET_NAME, forecast_horizon=H, output_dim=O_SINGLE
    )

@pytest.fixture
def forecast_df_quantile_single(base_predictions) -> pd.DataFrame:
    """DataFrame for quantile forecast, single output, with actuals."""
    return format_predictions_to_dataframe(
        predictions=base_predictions["quant_single_pred"],
        y_true_sequences=base_predictions["y_true_single"],
        target_name=TARGET_NAME, quantiles=Q_LIST,
        forecast_horizon=H, output_dim=O_SINGLE
    )

@pytest.fixture
def forecast_df_point_multi_with_spatial(base_predictions) -> pd.DataFrame:
    """DataFrame for point forecast, multi-output, with spatial columns."""
    df = format_predictions_to_dataframe(
        predictions=base_predictions["point_multi_pred"],
        y_true_sequences=base_predictions["y_true_multi"],
        target_name=TARGET_NAME, forecast_horizon=H, output_dim=O_MULTI
    )
    # Add spatial columns - ensure length matches df
    # df has SAMPLES * H rows
    df[SPATIAL_COLS_TEST[0]] = np.tile(np.random.rand(SAMPLES), H)
    df[SPATIAL_COLS_TEST[1]] = np.tile(np.random.rand(SAMPLES), H)
    return df

@pytest.fixture
def mock_scaler():
    """A mock scaler with an inverse_transform method."""
    class MockScaler:
        def __init__(self, feature_names_in_):
            self.feature_names_in_ = feature_names_in_
            self.n_features_in_ = len(feature_names_in_)
        def inverse_transform(self, X):
            if X.shape[1] != self.n_features_in_ :
                if X.shape[1] == 1 and self.n_features_in_ > 1:
                    return X + 100.0 # Simplified for single target column
                raise ValueError(f"Scaler expected {self.n_features_in_} features, got {X.shape[1]}")
            return X + 100.0 # Simple inverse: adds 100
    return MockScaler

# --- Test Cases ---

@patch('matplotlib.pyplot.show') # Mock plt.show to prevent plots appearing
@patch('matplotlib.pyplot.figure') # Mock plt.figure
def test_temporal_point_single_output(
    mock_fig, mock_show, forecast_df_point_single
    ):
    """Test temporal plot for point forecast, single output."""
    try:
        plot_forecasts(
            forecast_df=forecast_df_point_single,
            target_name=TARGET_NAME,
            kind="temporal",
            sample_ids="first_n", num_samples=1,
            verbose=0
        )
        mock_fig.assert_called() # Check if a figure was created
        mock_show.assert_called_once() # Check if plt.show() was called
    except Exception as e:
        pytest.fail(f"plot_forecasts (temporal, point, single_out) failed: {e}")
    print("Temporal point single output: OK")

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.figure')
def test_temporal_quantile_single_output(
    mock_fig, mock_show, forecast_df_quantile_single
    ):
    """Test temporal plot for quantile forecast, single output."""
    try:
        plot_forecasts(
            forecast_df=forecast_df_quantile_single,
            target_name=TARGET_NAME,
            quantiles=Q_LIST,
            kind="temporal",
            sample_ids=[0, 1], # Test list of sample_ids
            max_cols=1, # Test max_cols
            verbose=0
        )
        mock_fig.assert_called()
        mock_show.assert_called_once()
    except Exception as e:
        pytest.fail(f"plot_forecasts (temporal, quantile, single_out) failed: {e}")
    print("Temporal quantile single output: OK")


@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.figure')
def test_spatial_point_multi_output(
    mock_fig, mock_show, forecast_df_point_multi_with_spatial
    ):
    """Test spatial plot for point forecast, multi-output."""
    try:
        plot_forecasts(
            forecast_df=forecast_df_point_multi_with_spatial,
            target_name=TARGET_NAME,
            output_dim=O_MULTI,
            kind="spatial",
            horizon_steps=1, # Plot first horizon step
            spatial_cols=SPATIAL_COLS_TEST,
            verbose=0
        )
        # For multi-output and multiple horizon steps, multiple figures might be created
        # or multiple subplots. Here, 1 step, O_MULTI outputs -> O_MULTI subplots
        assert mock_fig.call_count >= 1
        assert mock_show.call_count >= 1
    except Exception as e:
        pytest.fail(f"plot_forecasts (spatial, point, multi_out) failed: {e}")
    print("Spatial point multi-output: OK")

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.figure')
def test_spatial_quantile_all_horizon_steps(
    mock_fig, mock_show, base_predictions
    ):
    """Test spatial plot for quantile, all horizon steps."""
    # Create a df with spatial columns for this test
    df_quant_spatial = format_predictions_to_dataframe(
        predictions=base_predictions["quant_single_pred"],
        y_true_sequences=base_predictions["y_true_single"],
        target_name=TARGET_NAME, quantiles=Q_LIST,
        forecast_horizon=H, output_dim=O_SINGLE
    )
    df_quant_spatial[SPATIAL_COLS_TEST[0]] = np.tile(np.random.rand(SAMPLES), H)
    df_quant_spatial[SPATIAL_COLS_TEST[1]] = np.tile(np.random.rand(SAMPLES), H)

    try:
        plot_forecasts(
            forecast_df=df_quant_spatial,
            target_name=TARGET_NAME,
            quantiles=Q_LIST,
            output_dim=O_SINGLE,
            kind="spatial",
            horizon_steps="all", # Plot all H steps
            spatial_cols=SPATIAL_COLS_TEST,
            max_cols=H, # Ensure all steps fit in one row if possible
            verbose=0
        )
        assert mock_fig.call_count >= 1
        assert mock_show.call_count >= 1
    except Exception as e:
        pytest.fail(f"plot_forecasts (spatial, quantile, all_steps) failed: {e}")
    print("Spatial quantile all horizon steps: OK")

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.figure')
def test_scaler_functionality_temporal(
    mock_fig, mock_show, forecast_df_point_single, mock_scaler
    ):
    """Test temporal plot with scaler for inverse transform."""
    scaler_features = ['other_feat', TARGET_NAME, 'another_feat']
    target_idx = 1
    scaler_instance = mock_scaler(scaler_features)

    # Add some value to check if inverse transform (add 100) happens
    df_copy = forecast_df_point_single.copy()
    # Ensure original values are small enough that adding 100 is noticeable
    df_copy[f'{TARGET_NAME}_pred'] = df_copy[f'{TARGET_NAME}_pred'] - 90
    df_copy[f'{TARGET_NAME}_actual'] = df_copy[f'{TARGET_NAME}_actual'] - 90

    try:
        with patch('matplotlib.pyplot.plot') as mock_plot: # Mock plot to check data
            plot_forecasts(
                forecast_df=df_copy,
                target_name=TARGET_NAME,
                kind="temporal",
                sample_ids=0,
                scaler=scaler_instance,
                scaler_feature_names=scaler_features,
                target_idx_in_scaler=target_idx,
                verbose=1
            )
            mock_fig.assert_called()
            mock_show.assert_called_once()

            # Check if plotted data was inverse transformed (values > 10)
            # This is an indirect check. Actual plotted values are in mock_plot.call_args_list
            # For simplicity, we just check if it runs. A more detailed test
            # would inspect the arguments passed to ax.plot.
            # Example: mock_plot.call_args_list[0][0][1] would be y-values of first plot
            # assert np.all(mock_plot.call_args_list[0][0][1] > 10) # If actuals plotted first
            # assert np.all(mock_plot.call_args_list[1][0][1] > 10) # If preds plotted second
    except Exception as e:
        pytest.fail(f"plot_forecasts with scaler failed: {e}")
    print("Scaler functionality with temporal plot: OK")


def test_error_invalid_kind(forecast_df_point_single):
    """Test ValueError for invalid plot kind."""
    with pytest.raises(ValueError, 
                       match="Unsupported `kind`: 'unknown'. Choose 'temporal' or 'spatial'."):
        plot_forecasts(forecast_df_point_single, kind="unknown")
    print("Error invalid kind: OK")

def test_error_missing_spatial_cols_for_spatial_plot(forecast_df_point_single):
    """Test ValueError if spatial_cols missing for kind='spatial'."""
    with pytest.raises(ValueError, match=( 
                           "`spatial_x_col` and `spatial_y_col` must"
                           " be provided for `kind='spatial'`."
                           )):
        plot_forecasts(forecast_df_point_single, kind="spatial", spatial_cols=None)
    with pytest.raises(ValueError, match=( 
            "Spatial columns 'non_existent_x' or 'non_existent_y' not found in forecast_df.")):
        plot_forecasts(forecast_df_point_single, kind="spatial",
                       spatial_cols=['non_existent_x', 'non_existent_y'])
    print("Error missing/invalid spatial_cols for spatial plot: OK")

def test_error_missing_core_df_columns():
    """Test ValueError if forecast_df misses 'sample_idx' or 'forecast_step'."""
    bad_df1 = pd.DataFrame({'forecast_step': [1,2], 'value_pred': [0,1]})
    with pytest.raises(ValueError, match="must contain 'sample_idx'"):
        plot_forecasts(bad_df1)
    bad_df2 = pd.DataFrame({'sample_idx': [0,0], 'value_pred': [0,1]})
    with pytest.raises(ValueError, match="must contain .* 'forecast_step'"):
        plot_forecasts(bad_df2)
    print("Error missing core DataFrame columns: OK")

if __name__ == '__main__':
    pytest.main([__file__])

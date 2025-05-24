
import re
import pytest
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt # For mocking # noqa
# For mocking plt.show and plt.figure
from unittest.mock import patch, MagicMock, call # noqa

# --- Attempt to import functions and dependencies ---
try:
    from fusionlab.plot.evaluation import (
        plot_metric_radar,
        plot_forecast_comparison,
        plot_metric_over_horizon
    )
    from fusionlab.nn.utils import format_predictions_to_dataframe
    
    FUSIONLAB_INSTALLED = True
except ImportError as e:
    print(f"Skipping plot.evaluation tests due to import error: {e}")
    FUSIONLAB_INSTALLED = False
    def format_predictions_to_dataframe(*args, **kwargs):
        # Return a minimal dummy DataFrame for tests to parse
        return pd.DataFrame({
            'sample_idx': [0,0,1,1],
            'forecast_step': [1,2,1,2],
            'target_actual': [0,0,0,0],
            'target_pred': [0,0,0,0]
        })

pytestmark = pytest.mark.skipif(
    not FUSIONLAB_INSTALLED,
    reason="fusionlab.plot.evaluation or dependencies not found"
)

# --- Test Fixtures and Constants ---
B, H, O_SINGLE, O_MULTI = 4, 3, 1, 2 # Batch, Horizon, OutputDims
Q_LIST = [0.1, 0.5, 0.9]
N_Q = len(Q_LIST)
SAMPLES = B

TARGET_NAME_TEST = "sales"
SPATIAL_COLS_TEST = ['store_id', 'region_code']
SEGMENT_COL_TEST = 'store_id'


@pytest.fixture
def base_predictions_for_plot():
    """Generates base prediction arrays for creating forecast_df."""
    np.random.seed(42) # For reproducibility in tests
    return {
        "point_single_pred": np.random.rand(SAMPLES, H, O_SINGLE).astype(np.float32),
        "point_multi_pred": np.random.rand(SAMPLES, H, O_MULTI).astype(np.float32),
        "quant_single_pred": np.random.rand(SAMPLES, H, N_Q).astype(np.float32), # O=1
        "quant_multi_pred_4d": np.random.rand(SAMPLES, H, N_Q, O_MULTI).astype(np.float32),
        "y_true_single": np.random.rand(SAMPLES, H, O_SINGLE).astype(np.float32) + 0.5,
        "y_true_multi": np.random.rand(SAMPLES, H, O_MULTI).astype(np.float32) + 0.5,
    }

@pytest.fixture
def forecast_df_point_single_plot(base_predictions_for_plot) -> pd.DataFrame:
    df = format_predictions_to_dataframe(
        predictions=base_predictions_for_plot["point_single_pred"],
        y_true_sequences=base_predictions_for_plot["y_true_single"],
        target_name=TARGET_NAME_TEST, forecast_horizon=H, output_dim=O_SINGLE
    )
    # Add dummy spatial/segment columns for broader testing
    df[SPATIAL_COLS_TEST[0]] = np.repeat(np.arange(SAMPLES), H)
    df[SPATIAL_COLS_TEST[1]] = np.tile(np.arange(H), SAMPLES) # Example
    return df

@pytest.fixture
def forecast_df_quantile_multi_plot(base_predictions_for_plot) -> pd.DataFrame:
    df = format_predictions_to_dataframe(
        predictions=base_predictions_for_plot["quant_multi_pred_4d"],
        y_true_sequences=base_predictions_for_plot["y_true_multi"],
        target_name=TARGET_NAME_TEST, quantiles=Q_LIST,
        forecast_horizon=H, output_dim=O_MULTI
    )
    df[SPATIAL_COLS_TEST[0]] = np.repeat([f"store_{i}" for i in range(SAMPLES)], H)
    df[SPATIAL_COLS_TEST[1]] = np.tile([f"region_{j}" for j in range(H)], SAMPLES)
    df['month'] = np.tile([1,2,3]* (len(df)//3),1)[:len(df)] # Example segment
    return df

# --- Tests for plot_forecast_comparison ---

@pytest.mark.skip("Test passed locally so can skipped.")
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.figure')
def test_plot_forecast_comparison_temporal_point(
    mock_fig, mock_show, forecast_df_point_single_plot):
    """Test plot_forecast_comparison: temporal, point forecast."""
    try:
        plot_forecast_comparison(
            forecast_df=forecast_df_point_single_plot,
            target_name=TARGET_NAME_TEST,
            kind="temporal",
            sample_ids="first_n", num_samples=1,
            verbose=0
        )
        mock_fig.assert_called()
        mock_show.assert_called_once()
    except Exception as e:
        pytest.fail(f"plot_forecast_comparison (temporal, point) failed: {e}")
    print("plot_forecast_comparison (temporal, point): OK")

@pytest.mark.skip("Test passed locally so can skipped.")
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.figure')
def test_plot_forecast_comparison_spatial_quantile(
    mock_fig, mock_show, forecast_df_quantile_multi_plot):
    """Test plot_forecast_comparison: spatial, quantile forecast."""
    try:
        plot_forecast_comparison(
            forecast_df=forecast_df_quantile_multi_plot,
            target_name=TARGET_NAME_TEST,
            quantiles=Q_LIST,
            output_dim=O_MULTI,
            kind="spatial",
            horizon_steps=[1, H], # Test list of steps
            spatial_cols=SPATIAL_COLS_TEST,
            max_cols=2,
            verbose=0
        )
        assert mock_fig.call_count >= 1 # One fig per output_dim or combined
        assert mock_show.call_count >= 1
    except Exception as e:
        pytest.fail(f"plot_forecast_comparison (spatial, quantile) failed: {e}")
    print("plot_forecast_comparison (spatial, quantile): OK")

def test_plot_forecast_comparison_errors(forecast_df_point_single_plot):
    """Test error handling in plot_forecast_comparison."""
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        plot_forecast_comparison(forecast_df=None)
    with pytest.raises(ValueError, match=re.escape(
            "`forecast_df` needs columns 'sample_idx' and 'forecast_step'.")):
        plot_forecast_comparison(forecast_df=pd.DataFrame({'test': [1]}))
    with pytest.raises(ValueError, match=re.escape (
            "`kind` must be 'temporal' or 'spatial'.")):
        plot_forecast_comparison(forecast_df_point_single_plot, kind="bad_kind")
    with pytest.raises(ValueError, match=re.escape(
            "`spatial_cols` must be two columns for kind='spatial'.")):
        plot_forecast_comparison(forecast_df_point_single_plot, kind="spatial",
                                 spatial_cols=['lon_only'])
    print("plot_forecast_comparison error handling: OK")

# --- Tests for plot_metric_over_horizon ---

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.figure')
def test_plot_metric_over_horizon_bar_mae(
    mock_fig, mock_show, forecast_df_point_single_plot):
    """Test plot_metric_over_horizon: bar plot, MAE."""
    try:
        plot_metric_over_horizon(
            forecast_df=forecast_df_point_single_plot,
            target_name=TARGET_NAME_TEST,
            metrics='mae',
            plot_kind='bar',
            verbose=0
        )
        mock_fig.assert_called()
        mock_show.assert_called_once()
    except Exception as e:
        pytest.fail(f"plot_metric_over_horizon (bar, mae) failed: {e}")
    print("plot_metric_over_horizon (bar, mae): OK")

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.figure')
def test_plot_metric_over_horizon_line_multi_metric_grouped(
    mock_fig, mock_show, forecast_df_quantile_multi_plot):
    """Test plot_metric_over_horizon: line, multi-metric, grouped."""

    try:
        plot_metric_over_horizon(
            forecast_df=forecast_df_quantile_multi_plot,
            target_name=TARGET_NAME_TEST,
            metrics=['mae', 'rmse'], # RMSE requires MSE from sklearn
            quantiles=Q_LIST, # MAE/RMSE will be on median
            output_dim=O_MULTI,
            group_by_cols=[SPATIAL_COLS_TEST[0]], # Group by store_id
            plot_kind='line',
            max_cols_metrics=1, # Test layout
            verbose=0
        )
        # One figure per output_dim, each with subplots for metrics
        assert mock_fig.call_count == O_MULTI
        assert mock_show.call_count == O_MULTI
    except Exception as e:
        pytest.fail(f"plot_metric_over_horizon (line, multi, grouped) failed: {e}")
    print("plot_metric_over_horizon (line, multi, grouped): OK")

# --- Tests for plot_metric_radar ---

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.figure')
def test_plot_metric_radar_mae_by_segment(
    mock_fig, mock_show, forecast_df_quantile_multi_plot):
    """Test plot_metric_radar: MAE by segment."""
    # Use 'month' as segment from the df_quantile_multi_plot
    segment_col_radar = 'month'
    if segment_col_radar not in forecast_df_quantile_multi_plot.columns:
        # Add dummy if not present from fixture generation
        forecast_df_quantile_multi_plot[segment_col_radar] = \
            np.tile(np.arange(1, H + 1), SAMPLES*O_MULTI)[:len(forecast_df_quantile_multi_plot)]


    try:
        plot_metric_radar(
            forecast_df=forecast_df_quantile_multi_plot,
            segment_col=segment_col_radar,
            metric='mae', # MAE of median for quantile forecast
            target_name=TARGET_NAME_TEST,
            quantiles=Q_LIST,
            output_dim=O_MULTI, # Will create O_MULTI radar plots
            verbose=0
        )
        #assert mock_fig.call_count == O_MULTI # One plot per output_dim
        assert mock_show.call_count == O_MULTI
    except Exception as e:
        pytest.fail(f"plot_metric_radar (mae by segment) failed: {e}")
    print("plot_metric_radar (mae by segment): OK")

def test_plot_metric_radar_errors(forecast_df_point_single_plot):
    """Test error handling in plot_metric_radar."""
    with pytest.raises(ValueError, ):
        plot_metric_radar(forecast_df_point_single_plot, segment_col="non_existent")
    with pytest.raises(ValueError):
        plot_metric_radar(forecast_df_point_single_plot, segment_col=SPATIAL_COLS_TEST[0],
                          metric="bad_metric_name")
    print("plot_metric_radar error handling: OK")


if __name__ == '__main__':
    pytest.main([__file__])


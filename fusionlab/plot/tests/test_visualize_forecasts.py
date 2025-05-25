# test_visualize_forecasts.py
import re
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # For mocking
from unittest.mock import patch, MagicMock

# --- Attempt to import function and dependencies ---
try:
    from fusionlab.plot.forecast import visualize_forecasts
    # For creating dummy DataFrames, assuming format_predictions_to_dataframe
    # is a good way to get a compatible structure, or create manually.
    # For this test, we'll create DataFrames manually to match visualize_forecasts' needs.
    FUSIONLAB_INSTALLED = True
except ImportError as e:
    print(f"Skipping visualize_forecasts tests due to import error: {e}")
    FUSIONLAB_INSTALLED = False
    def visualize_forecasts(*args, **kwargs):
        raise ImportError("visualize_forecasts not found")

pytestmark = pytest.mark.skipif(
    not FUSIONLAB_INSTALLED,
    reason="fusionlab.plot.forecast.visualize_forecasts not found"
)

# --- Test Fixtures and Constants ---
N_SAMPLES_PER_PERIOD = 10
DT_COL_TEST = 'year' # As per example, could be 'Date'
TARGET_NAME_TEST_VIZ = 'subsidence'
SPATIAL_X_TEST = 'longitude'
SPATIAL_Y_TEST = 'latitude'

@pytest.fixture
def forecast_df_for_viz() -> pd.DataFrame:
    """
    Creates a dummy forecast_df suitable for visualize_forecasts.
    Includes multiple periods.
    """
    data = []
    for year in [2022, 2023]:
        for i in range(N_SAMPLES_PER_PERIOD):
            data.append({
                DT_COL_TEST: year,
                SPATIAL_X_TEST: 113.0 + np.random.rand() * 0.1,
                SPATIAL_Y_TEST: 22.5 + np.random.rand() * 0.1,
                f'{TARGET_NAME_TEST_VIZ}_q50': np.random.rand() * 10, # For quantile mode
                f'{TARGET_NAME_TEST_VIZ}_pred': np.random.rand() * 10, # For point mode
                # Actuals might also be in this df if test_data is None
                f'{TARGET_NAME_TEST_VIZ}': np.random.rand() * 10 + 1
            })
    return pd.DataFrame(data)

@pytest.fixture
def test_data_for_viz() -> pd.DataFrame:
    """
    Creates a dummy test_data DataFrame for visualize_forecasts.
    """
    data = []
    for year in [2022, 2023, 2024]: # Include an extra year not in forecast_df
        for i in range(N_SAMPLES_PER_PERIOD):
            data.append({
                DT_COL_TEST: year,
                SPATIAL_X_TEST: 113.0 + np.random.rand() * 0.1,
                SPATIAL_Y_TEST: 22.5 + np.random.rand() * 0.1,
                TARGET_NAME_TEST_VIZ: np.random.rand() * 10 + 1.5 # Actual values
            })
    return pd.DataFrame(data)

# --- Test Cases for visualize_forecasts ---

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.figure')
def test_visualize_spatial_quantile_with_test_data(
    mock_fig, mock_show, forecast_df_for_viz, test_data_for_viz):
    """Test spatial quantile plot with separate test_data."""
    try:
        visualize_forecasts(
            forecast_df=forecast_df_for_viz,
            dt_col=DT_COL_TEST,
            tname=TARGET_NAME_TEST_VIZ,
            test_data=test_data_for_viz,
            eval_periods=[2022, 2023], # Periods present in both
            mode="quantile",
            kind="spatial",
            x=SPATIAL_X_TEST, y=SPATIAL_Y_TEST,
            verbose=0
        )
        mock_fig.assert_called()
        mock_show.assert_called_once()
        # Check if subplots were created (2 periods * 2 plots/period = 4)
        # mock_fig.return_value.subplots.assert_called_with(4, 2) # Example
    except Exception as e:
        pytest.fail(f"visualize_forecasts (spatial, quantile, test_data) failed: {e}")
    print("Visualize spatial quantile with test_data: OK")

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.figure')
def test_visualize_spatial_point_no_test_data(
    mock_fig, mock_show, forecast_df_for_viz):
    """Test spatial point plot without separate test_data."""
    # Actuals should come from forecast_df itself if test_data is None
    # Ensure forecast_df has the actual column
    df = forecast_df_for_viz.copy()
    df[TARGET_NAME_TEST_VIZ] = df[f'{TARGET_NAME_TEST_VIZ}_pred'] + \
                               np.random.randn(len(df)) # Dummy actuals

    try:
        visualize_forecasts(
            forecast_df=df,
            dt_col=DT_COL_TEST,
            tname=TARGET_NAME_TEST_VIZ,
            test_data=None, # Key part of this test
            eval_periods=[2022],
            mode="point",
            kind="spatial",
            x=SPATIAL_X_TEST, y=SPATIAL_Y_TEST,
            max_cols=1,
            verbose=0
        )
        mock_fig.assert_called()
        mock_show.assert_called_once()
    except Exception as e:
        pytest.fail(f"visualize_forecasts (spatial, point, no test_data) failed: {e}")
    print("Visualize spatial point no test_data: OK")

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.figure')
def test_visualize_non_spatial_plot(
    mock_fig, mock_show, forecast_df_for_viz):
    """Test non-spatial plot kind."""
    df = forecast_df_for_viz.copy()
    # For non-spatial, x and y are just regular columns
    df['time_step_plot'] = np.tile(np.arange(N_SAMPLES_PER_PERIOD), 2)

    try:
        visualize_forecasts(
            forecast_df=df,
            dt_col=DT_COL_TEST,
            tname=TARGET_NAME_TEST_VIZ,
            eval_periods=[2022],
            mode="point",
            kind="non-spatial", # Key change
            x='time_step_plot', y=f'{TARGET_NAME_TEST_VIZ}_pred',
            verbose=0
        )
        mock_fig.assert_called()
        mock_show.assert_called_once()
    except Exception as e:
        pytest.fail(f"visualize_forecasts (non-spatial) failed: {e}")
    print("Visualize non-spatial plot: OK")

def test_visualize_error_missing_columns(forecast_df_for_viz):
    """Test ValueErrors for missing essential columns."""
    df_no_dt = forecast_df_for_viz.drop(columns=[DT_COL_TEST])
    with pytest.raises(KeyError): # filter_by_period will raise KeyError
        visualize_forecasts(df_no_dt, dt_col=DT_COL_TEST, tname=TARGET_NAME_TEST_VIZ)

    df_no_x = forecast_df_for_viz.drop(columns=[SPATIAL_X_TEST])
    with pytest.raises(ValueError, match=re.escape (
            "The following spatial_cols are not present in the dataframe: {'longitude'}"
            )):
        visualize_forecasts(df_no_x, dt_col=DT_COL_TEST, tname=TARGET_NAME_TEST_VIZ,
                            kind="spatial", x=SPATIAL_X_TEST, y=SPATIAL_Y_TEST)
    print("Visualize error missing columns: OK")

def test_visualize_error_invalid_mode_kind(forecast_df_for_viz):
    """Test ValueErrors for invalid mode or kind."""
    with pytest.raises(ValueError, match="Mode must be either 'quantile' or 'point'"):
        visualize_forecasts(forecast_df_for_viz, dt_col=DT_COL_TEST,
                            tname=TARGET_NAME_TEST_VIZ, mode="bad_mode")
    # Kind validation happens inside if kind=="spatial" or else,
    # so a generic error might not be raised if x,y are not provided for non-spatial
    # This depends on how strictly 'kind' is validated if not 'spatial'.
    # The provided code for visualize_forecasts doesn't explicitly error on unknown 'kind'
    # if x and y are provided for non-spatial.
    print("Visualize error invalid mode: OK")


@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.figure')
def test_visualize_eval_periods_auto_selection(
    mock_fig, mock_show, forecast_df_for_viz, test_data_for_viz):
    """Test auto-selection of eval_periods when None."""
    try:
        visualize_forecasts(
            forecast_df=forecast_df_for_viz,
            dt_col=DT_COL_TEST,
            tname=TARGET_NAME_TEST_VIZ,
            test_data=test_data_for_viz,
            eval_periods=None, # Auto-select
            mode="quantile",
            kind="spatial",
            verbose=1 # To see print about auto-selection
        )
        mock_fig.assert_called()
        mock_show.assert_called_once()
    except Exception as e:
        pytest.fail(f"visualize_forecasts (auto eval_periods) failed: {e}")
    print("Visualize auto eval_periods: OK")


if __name__ == '__main__':
    pytest.main([__file__])

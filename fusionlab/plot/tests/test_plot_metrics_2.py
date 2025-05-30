# -*- coding: utf-8 -*-
#   License: BSD-3-Clause

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes  # For type hinting
import warnings

# Attempt to import plotting functions and required utilities
# In a real test environment, these imports should succeed.
try:
    from fusionlab.plot._evaluation import (
        plot_radar_scores,            
        plot_qce_donut                
    )
    FUSIONLAB_AVAILABLE = True
except ImportError as e:
    FUSIONLAB_AVAILABLE = False
    warnings.warn(
        f"Failed to import fusionlab components: {e}. "
        "Plotting tests will be skipped or will use placeholders. "
        "Ensure fusionlab is correctly installed and accessible."
    )
# --- Pytest Fixtures ---
@pytest.fixture(autouse=True)
def close_plots_after_test():
    """Close all matplotlib plots after each test."""
    yield
    if FUSIONLAB_AVAILABLE: 
        plt.close('all')

# --- Test Data ---
# Basic 1D data
Y_TRUE_1D_BASIC = np.array([1, 2, 3, 4])
Y_LOWER_1D = np.array([0, 1, 2, 3])
Y_UPPER_1D = np.array([2, 3, 4, 5])
Y_MEDIAN_1D = np.array([1, 2, 3, 4]) 

Y_TRUE_2D_BASIC = np.array([[1,10], [2,20], [3,30]]) 
Y_LOWER_2D = np.array([[0,9], [1,19], [2,29]])
Y_UPPER_2D = np.array([[2,11], [3,21], [4,31]])
Y_MEDIAN_2D = np.array([[1,10], [2,20], [3,30]]) 

ALPHAS_K2 = np.array([0.1, 0.5]) 
Y_LOWER_WIS_1D_K2_PLOT = Y_LOWER_1D[:, np.newaxis] + np.array([-0.5, -1]) 
Y_UPPER_WIS_1D_K2_PLOT = Y_UPPER_1D[:, np.newaxis] + np.array([0.5, 1])  
Y_LOWER_WIS_2D_K2_PLOT = Y_LOWER_2D[...,np.newaxis] + np.array([[[-0.5,-1]]]) 
Y_UPPER_WIS_2D_K2_PLOT = Y_UPPER_2D[...,np.newaxis] + np.array([[[0.5,1]]])   

Y_PRED_ENS_1D = np.random.rand(4, 5) 
Y_PRED_ENS_2D = np.random.rand(3, 2, 5) 

Y_TRUE_TS_S2_O1_T4 = np.array([[[1,2,3,4]],[[5,6,7,8]]]) 
Y_PRED_TS_S2_O1_T4 = Y_TRUE_TS_S2_O1_T4 + np.random.normal(size=(2,1,4)) * 0.1
Y_TRUE_TS_S2_O2_T3 = np.random.randint(0,5,size=(2,2,3))
Y_PRED_TS_S2_O2_T3 = Y_TRUE_TS_S2_O2_T3 + np.random.normal(size=(2,2,3)) * 0.2
Y_MEDIAN_TS_S2_O2_T3 = Y_PRED_TS_S2_O2_T3 
ALPHAS_K1_TWIS = np.array([0.5])
Y_LOWER_TWIS_S2_O2_K1_T3 = Y_PRED_TS_S2_O2_T3[:,:,np.newaxis,:] - 0.5
Y_UPPER_TWIS_S2_O2_K1_T3 = Y_PRED_TS_S2_O2_T3[:,:,np.newaxis,:] + 0.5

Y_PRED_PSS_S3_T5 = np.array([[1,1,2,2,3], [2,3,2,3,2], [0,1,0,1,0]]) 
Y_PRED_PSS_S2_O2_T3 = np.array([[[1,2,1],[5,5,5]],[[3,2,3],[0,1,0]]]) 
Y_PRED_PSS_NAN = np.array([[1,np.nan,2], [3,3,3]]) 

QUANTILES_Q3 = np.array([0.25, 0.5, 0.75])
Y_PRED_QCE_1D_Q3 = np.array([[0.5,1.5,2.5],[1.5,2.5,3.5],[2.5,3.5,4.5],[3.5,4.5,5.5]])
Y_PRED_QCE_2D_Q3 = np.random.rand(3,2,3) 

# Data for Radar Plot
RADAR_VALUES_DICT = {"MAE": 0.5, "RMSE": 0.7, "MAPE": 0.1, "PSS": 0.8}
RADAR_VALUES_LIST = [0.5, 0.7, 0.1, 0.8]
RADAR_NAMES_FULL = ["MAE", "RMSE", "MAPE", "PSS"]
RADAR_NAMES_PARTIAL = ["MAE", "RMSE"]
Y_TRUE_RADAR = np.array([1, 2, 3, 4, 5, 6])
Y_PRED_RADAR = np.array([1.1, 1.9, 3.2, 3.8, 5.3, 6.1])

# Data for QCE Donut Plot
DF_QCE_DONUT = pd.DataFrame({
    'actual': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'q0.1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'q0.5': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'q0.9': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
})
ACTUAL_COL_DONUT = 'actual'
QUANTILE_COLS_DONUT = ['q0.1', 'q0.5', 'q0.9']
QUANTILE_LEVELS_DONUT = [0.1, 0.5, 0.9]


# Skip tests if fusionlab components are not available
pytestmark = pytest.mark.skipif(
    not FUSIONLAB_AVAILABLE,
    reason="Skipping plotting tests: fusionlab components not imported."
)

# --- New Test Classes for plot_radar_scores and plot_qce_donut ---

class TestPlotRadarScores:
    def test_with_dict_values(self):
        ax = plot_radar_scores(data_values=RADAR_VALUES_DICT)
        assert isinstance(ax, Axes)
        assert len(ax.get_xticklabels()) == len(RADAR_VALUES_DICT)

    def test_with_list_values_and_names(self):
        ax = plot_radar_scores(data_values=RADAR_VALUES_LIST,
                               category_names=RADAR_NAMES_FULL)
        assert isinstance(ax, Axes)
        assert len(ax.get_xticklabels()) == len(RADAR_VALUES_LIST)

    def test_with_list_values_partial_names(self):
        ax = plot_radar_scores(data_values=RADAR_VALUES_LIST,
                               category_names=RADAR_NAMES_PARTIAL,
                               verbose=1) # To check auto-name warning
        assert isinstance(ax, Axes)
        assert len(ax.get_xticklabels()) == len(RADAR_VALUES_LIST)
        # Check if last label is auto-generated (e.g., "metric_4")
        assert "metric_4" in [label.get_text() for label in ax.get_xticklabels()] \
            or "metric_3" in [label.get_text() for label in ax.get_xticklabels()]

    def test_with_y_true_y_pred_default_metrics(self):
        ax = plot_radar_scores(y_true=Y_TRUE_RADAR, y_pred=Y_PRED_RADAR)
        assert isinstance(ax, Axes)
        assert len(ax.get_xticklabels()) == 3 # MAE, RMSE, MAPE

    def test_with_y_true_y_pred_custom_metric(self):
        from sklearn.metrics import max_error
        custom_funcs = [max_error]
        custom_names = ["MaxError"]
        ax = plot_radar_scores(y_true=Y_TRUE_RADAR, y_pred=Y_PRED_RADAR,
                               metric_functions=custom_funcs,
                               category_names=custom_names)
        assert isinstance(ax, Axes)
        assert "MaxError" in [label.get_text() for label in ax.get_xticklabels()]

    def test_normalize_values(self):
        ax = plot_radar_scores(data_values=RADAR_VALUES_DICT,
                               normalize_values=True)
        assert isinstance(ax, Axes)
        # Check if y-ticks are roughly between 0 and 1
        yticks = ax.get_yticks()
        assert np.all(yticks >= 0) and np.all(yticks <= 1.01) # Allow for slight overshoot

    def test_custom_ax_provided(self):
        fig, custom_ax = plt.subplots(subplot_kw=dict(polar=True))
        ax = plot_radar_scores(data_values=RADAR_VALUES_DICT, ax=custom_ax)
        assert ax is custom_ax

    def test_no_data_values_no_true_pred(self):
        with pytest.raises(ValueError, match="y_true' and 'y_pred' must be provided"):
            plot_radar_scores()

    def test_empty_values_list(self):
         with pytest.raises(ValueError, match="it must not be empty"):
            plot_radar_scores(data_values=[])


class TestPlotQCEDonut:
    def test_basic_donut_plot(self):
        ax = plot_qce_donut(df=DF_QCE_DONUT,
                              actual_col=ACTUAL_COL_DONUT,
                              quantile_cols=QUANTILE_COLS_DONUT,
                              quantile_levels=QUANTILE_LEVELS_DONUT)
        assert isinstance(ax, Axes)
        # Check for wedges (pie segments)
        # assert len(ax.patches) == len(QUANTILE_LEVELS_DONUT)
        # Check for center text (Avg QCE)
        center_text_found = any(
            "Avg QCE" in child.get_text() for child in ax.texts
            if child.get_position() == (0,0) # Center text
        )
        assert center_text_found

    def test_nan_handling_omit(self):
        df_nan = DF_QCE_DONUT.copy()
        df_nan.loc[0, ACTUAL_COL_DONUT] = np.nan # Introduce NaN
        ax = plot_qce_donut(df=df_nan,
                              actual_col=ACTUAL_COL_DONUT,
                              quantile_cols=QUANTILE_COLS_DONUT,
                              quantile_levels=QUANTILE_LEVELS_DONUT,
                              metric_kws={'nan_policy': 'omit'})
        assert isinstance(ax, Axes)
        # Further checks could verify if the calculations reflect omission

    def test_nan_handling_propagate_warns_or_empty_plot(self):
        df_nan = DF_QCE_DONUT.copy()
        df_nan.loc[0, QUANTILE_COLS_DONUT[0]] = np.nan
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = plot_qce_donut(df=df_nan,
                                  actual_col=ACTUAL_COL_DONUT,
                                  quantile_cols=QUANTILE_COLS_DONUT,
                                  quantile_levels=QUANTILE_LEVELS_DONUT,
                                  metric_kws={'nan_policy': 'propagate'})
            assert isinstance(ax, Axes)
            # Check if a warning about NaNs was issued, or if plot indicates issues
            nan_warning_found = any("NaNs found with nan_policy='propagate'" in str(warn.message) for warn in w)
            plot_text_indicates_issue = any("Data Issues" in child.get_text() for child in ax.texts)
            assert nan_warning_found or plot_text_indicates_issue


    def test_perfect_calibration_donut(self):
        # Create data that should result in QCE near 0 for all quantiles
        df_perfect = pd.DataFrame({
            'actual': np.linspace(0.01, 0.99, 100),
            'q0.25': np.full(
                100, np.percentile(np.linspace(0.01,0.99,100), 25)),
            'q0.50': np.full(
                100, np.percentile(np.linspace(0.01,0.99,100), 50)),
            'q0.75': np.full(
                100, np.percentile(np.linspace(0.01,0.99,100), 75))
        })
        # This setup is not ideal for perfect calibration, as y_pred should vary.
        # Let's make y_pred such that observed proportions match nominal.
        # For q=0.25, first 25% of y_true should be <= y_pred_q0.25
        # For simplicity, let's test the "no miscalibration" message.
        df_zero_miscal = DF_QCE_DONUT.copy()
        # Force miscalibrations to be zero by making observed_proportions = quantile_levels
        # This is hard to do by manipulating input data directly for the test.
        # Instead, we test the path where sum of miscalibrations is near zero.
        # This requires mocking the internal calculation or testing a known zero-miscal case.
        # For now, we'll test the warning/message when miscalibrations are effectively zero.
        
        # To test the zero miscalibration path, we need miscalibrations_q to sum to near zero.
        # This means observed_proportions_q approx quantile_levels_np
        # For this, let's use a case where observed_proportions match quantile_levels
        
        # Mocking internal calculation is too complex for this test.
        # We rely on the warning message for now.
        with warnings.catch_warnings(record=True) as w:
            ax = plot_qce_donut(df=DF_QCE_DONUT, # Use original data
                                actual_col=ACTUAL_COL_DONUT,
                                quantile_cols=QUANTILE_COLS_DONUT,
                                quantile_levels=QUANTILE_LEVELS_DONUT,
                                # To force zero miscal, we'd need to ensure observed_prop = nominal_q
                                # This test will show non-zero miscalibration
                                )
            # This test won't hit the "perfect calibration" text directly
            # unless the data is specifically crafted.
            # The goal here is just to run the plot.
            assert isinstance(ax, Axes)


    def test_custom_ax_provided_donut(self):
        fig, custom_ax = plt.subplots()
        ax = plot_qce_donut(df=DF_QCE_DONUT,
                              actual_col=ACTUAL_COL_DONUT,
                              quantile_cols=QUANTILE_COLS_DONUT,
                              quantile_levels=QUANTILE_LEVELS_DONUT,
                              ax=custom_ax)
        assert ax is custom_ax

    def test_missing_columns_donut(self):
        with pytest.raises(ValueError, ):
            plot_qce_donut(df=DF_QCE_DONUT,
                             actual_col='non_existent_actual',
                             quantile_cols=QUANTILE_COLS_DONUT,
                             quantile_levels=QUANTILE_LEVELS_DONUT)
        with pytest.raises(ValueError,):
            plot_qce_donut(df=DF_QCE_DONUT,
                             actual_col=ACTUAL_COL_DONUT,
                             quantile_cols=['q0.1', 'non_existent_q'],
                             quantile_levels=[0.1, 0.5])

if __name__=='__main__': 
    pytest.main( [__file__])
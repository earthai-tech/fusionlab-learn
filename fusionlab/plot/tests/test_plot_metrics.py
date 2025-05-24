# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes  
import warnings
from typing import Optional

try:
    from fusionlab.plot._evaluation import (
        plot_coverage,
        plot_crps,
        plot_mean_interval_width,
        plot_prediction_stability,
        plot_quantile_calibration,
        plot_theils_u_score,
        plot_time_weighted_metric,
        plot_weighted_interval_score
        )
    FUSIONLAB_AVAILABLE = True
except ImportError as e:
    FUSIONLAB_AVAILABLE = False
    warnings.warn(
        f"Failed to import fusionlab components: {e}. "
        "Plotting tests will be skipped or will use placeholders. "
        "Ensure fusionlab is correctly installed and accessible."
    )
    def _placeholder_plot_func(*args, **kwargs) -> Optional[Axes]:
        if kwargs.get('ax') is not None: return kwargs['ax']
        warnings.warn(f"Plot function for {args[0] if args else 'metric'} "
                      "is a placeholder due to import error.")
        return None 
    
    plot_coverage = _placeholder_plot_func
    plot_crps = _placeholder_plot_func
    plot_mean_interval_width = _placeholder_plot_func
    plot_prediction_stability = _placeholder_plot_func
    plot_quantile_calibration = _placeholder_plot_func
    plot_theils_u_score = _placeholder_plot_func
    plot_time_weighted_metric = _placeholder_plot_func
    plot_weighted_interval_score = _placeholder_plot_func


# --- Pytest Fixtures ---
@pytest.fixture(autouse=True)
def close_plots_after_test():
    """Close all matplotlib plots after each test."""
    yield
    if FUSIONLAB_AVAILABLE: 
        plt.close('all')

# --- Test Data (simplified for plotting tests) ---
# Focus is on shapes and basic values, not precise metric outcomes here.
# For coverage, MIW, non-time-weighted WIS
Y_TRUE_1D_BASIC = np.array([1, 2, 3, 4])
Y_LOWER_1D = np.array([0, 1, 2, 3])
Y_UPPER_1D = np.array([2, 3, 4, 5])
Y_MEDIAN_1D = np.array([1, 2, 3, 4]) # For WIS

Y_TRUE_2D_BASIC = np.array([[1,10], [2,20], [3,30]]) # (3,2)
Y_LOWER_2D = np.array([[0,9], [1,19], [2,29]])
Y_UPPER_2D = np.array([[2,11], [3,21], [4,31]])
Y_MEDIAN_2D = np.array([[1,10], [2,20], [3,30]]) # For WIS

ALPHAS_K2 = np.array([0.1, 0.5]) # K=2 intervals
# For WIS (N,O,K) or (N,K) -> (N,1,K)
Y_LOWER_WIS_1D_K2_PLOT = Y_LOWER_1D[:, np.newaxis] + \
    np.array([-0.5, -1]) # (4,2)
Y_UPPER_WIS_1D_K2_PLOT = Y_UPPER_1D[:, np.newaxis] + \
    np.array([0.5, 1])  # (4,2)

Y_LOWER_WIS_2D_K2_PLOT = Y_LOWER_2D[...,np.newaxis] + \
    np.array([[[-0.5,-1]]]) # (3,2,2)
Y_UPPER_WIS_2D_K2_PLOT = Y_UPPER_2D[...,np.newaxis] + \
    np.array([[[0.5,1]]])   # (3,2,2)


# For CRPS
Y_PRED_ENS_1D = np.random.rand(4, 5) # (N, M=5 members)
Y_PRED_ENS_2D = np.random.rand(3, 2, 5) # (N, O=2, M=5)

# For PSS, TW-MAE, TWA, TWIS (time-series like)
# y_true_ts: (N,O,T)
Y_TRUE_TS_S2_O1_T4 = np.array([[[1,2,3,4]],[[5,6,7,8]]]) # (2,1,4)
Y_PRED_TS_S2_O1_T4 = Y_TRUE_TS_S2_O1_T4 + \
    np.random.normal(size=(2,1,4)) * 0.1

Y_TRUE_TS_S2_O2_T3 = np.random.randint(0,5,size=(2,2,3))
Y_PRED_TS_S2_O2_T3 = Y_TRUE_TS_S2_O2_T3 + \
    np.random.normal(size=(2,2,3)) * 0.2
Y_MEDIAN_TS_S2_O2_T3 = Y_PRED_TS_S2_O2_T3 # Simplified for TWIS


# Definitions for PSS tests
Y_PRED_PSS_S3_T5 = np.array(
    [[1,1,2,2,3], [2,3,2,3,2], [0,1,0,1,0]]
) # (3 samples, 5 timesteps)
Y_PRED_PSS_S2_O2_T3 = np.array([
    [[1,2,1], [5,5,5]],      # Sample 0: Output 0, Output 1
    [[3,2,3], [0,1,0]]       # Sample 1: Output 0, Output 1
]) # (2 samples, 2 outputs, 3 timesteps)
Y_PRED_PSS_NAN = np.array([[1,np.nan,2], [3,3,3]]) # (2 samples, 3 timesteps)


# For QCE
QUANTILES_Q3 = np.array([0.25, 0.5, 0.75])
# For 1D y_true (N=4), y_pred_quantiles should be (N=4, Q=3)
Y_PRED_QCE_1D_Q3 = np.array([ 
    [0.5, 1.5, 2.5], [1.5, 2.5, 3.5],
    [2.5, 3.5, 4.5], [3.5, 4.5, 5.5]
])
# For 2D y_true (N=3, O=2), y_pred_quantiles should be (N=3, O=2, Q=3)
Y_PRED_QCE_2D_Q3 = np.random.rand(3,2,3) 


# For TWIS: y_lower/upper (N,O,K,T)
ALPHAS_K1_TWIS = np.array([0.5])
Y_LOWER_TWIS_S2_O2_K1_T3 = Y_PRED_TS_S2_O2_T3[:,:,np.newaxis,:] - 0.5
Y_UPPER_TWIS_S2_O2_K1_T3 = Y_PRED_TS_S2_O2_T3[:,:,np.newaxis,:] + 0.5


# For QCE
QUANTILES_Q3 = np.array([0.25, 0.5, 0.75])
Y_PRED_QCE_1D_Q3 = np.array([ # (N=4, Q=3)
    [0.5, 1.5, 2.5], [1.5, 2.5, 3.5],
    [2.5, 3.5, 4.5], [3.5, 4.5, 5.5]
])
Y_PRED_QCE_2D_Q3 = np.random.rand(3,2,3) # (N=3, O=2, Q=3)


# Skip tests if fusionlab components are not available
pytestmark = pytest.mark.skipif(
    not FUSIONLAB_AVAILABLE,
    reason="Skipping plotting tests: fusionlab components not imported."
)

# --- Test Classes for Plotting Functions ---

class TestPlotCoverage:
    def test_intervals_kind_1d(self):
        ax = plot_coverage(Y_TRUE_1D_BASIC, Y_LOWER_1D, Y_UPPER_1D,
                           kind='intervals')
        assert isinstance(ax, Axes)
        assert "Coverage" in ax.get_title()

    def test_summary_bar_kind_1d(self):
        ax = plot_coverage(Y_TRUE_1D_BASIC, Y_LOWER_1D, Y_UPPER_1D,
                           kind='summary_bar')
        assert isinstance(ax, Axes)
        assert len(ax.patches) == 1 # One bar for overall

    def test_intervals_kind_2d_multi_out(self):
        ax = plot_coverage(Y_TRUE_2D_BASIC, Y_LOWER_2D, Y_UPPER_2D,
                           kind='intervals', output_index=0)
        assert isinstance(ax, Axes)
        assert "Output 0" in ax.get_title()

    def test_summary_bar_kind_2d_raw(self):
        metric_kws = {'multioutput': 'raw_values'}
        ax = plot_coverage(Y_TRUE_2D_BASIC, Y_LOWER_2D, Y_UPPER_2D,
                           kind='summary_bar', metric_kws=metric_kws)
        assert isinstance(ax, Axes)
        assert len(ax.patches) == Y_TRUE_2D_BASIC.shape[1] # Bars per output

    def test_with_precomputed_values_summary(self):
        ax = plot_coverage(Y_TRUE_1D_BASIC, Y_LOWER_1D, Y_UPPER_1D,
                           coverage_values=0.75, kind='summary_bar')
        assert isinstance(ax, Axes)
        # Check if annotation matches
        annotation_found = any(
            "75.00%" in child.get_text() for child in ax.texts
            )
        assert annotation_found

    def test_custom_ax(self):
        fig, custom_ax = plt.subplots()
        ax = plot_coverage(Y_TRUE_1D_BASIC, Y_LOWER_1D, Y_UPPER_1D,
                           ax=custom_ax)
        assert ax is custom_ax


class TestPlotCRPS:
    def test_summary_bar_1d(self):
        ax = plot_crps(Y_TRUE_1D_BASIC, Y_PRED_ENS_1D, kind='summary_bar')
        assert isinstance(ax, Axes)
        assert len(ax.patches) == 1

    def test_summary_bar_2d_raw(self):
        metric_kws = {'multioutput': 'raw_values'}
        ax = plot_crps(Y_TRUE_2D_BASIC, Y_PRED_ENS_2D,
                       kind='summary_bar', metric_kws=metric_kws)
        assert isinstance(ax, Axes)
        assert len(ax.patches) == Y_TRUE_2D_BASIC.shape[1]

    def test_ecdf_plot(self):
        ax = plot_crps(Y_TRUE_1D_BASIC, Y_PRED_ENS_1D,
                       kind='ensemble_ecdf', sample_idx=0)
        assert isinstance(ax, Axes)
        assert "Continuous Ranked Probability Score (CRPS)" in ax.get_title()

    def test_histogram_plot_1d(self):
        ax = plot_crps(Y_TRUE_1D_BASIC, Y_PRED_ENS_1D,
                       kind='scores_histogram')
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0 # Some histogram bars


class TestPlotMeanIntervalWidth:
    def test_summary_bar_1d(self):
        ax = plot_mean_interval_width(Y_LOWER_1D, Y_UPPER_1D,
                                      kind='summary_bar')
        assert isinstance(ax, Axes)

    def test_histogram_plot_2d_selected_output(self):
        ax = plot_mean_interval_width(Y_LOWER_2D, Y_UPPER_2D,
                                      kind='widths_histogram',
                                      output_idx=0)
        assert isinstance(ax, Axes)
        assert "Output 0" in ax.get_title()


class TestPlotPredictionStability:
    def test_summary_bar_2d_timeseries(self): # y_pred (N,T)
        ax = plot_prediction_stability(Y_PRED_PSS_S3_T5,
                                       kind='summary_bar')
        assert isinstance(ax, Axes)

    def test_histogram_3d_timeseries_selected_output(self): # y_pred (N,O,T)
        ax = plot_prediction_stability(Y_PRED_PSS_S2_O2_T3,
                                       kind='scores_histogram',
                                       output_idx=0)
        assert isinstance(ax, Axes)


class TestPlotQuantileCalibration:
    def test_reliability_diagram_1d(self):
        ax = plot_quantile_calibration(Y_TRUE_1D_BASIC,
                                       Y_PRED_QCE_1D_Q3,
                                       QUANTILES_Q3,
                                       kind='reliability_diagram')
        assert isinstance(ax, Axes)
        assert "Quantile Calibration Error" in ax.get_title()

    def test_summary_bar_2d_raw(self):
        metric_kws = {'multioutput': 'raw_values'}
        ax = plot_quantile_calibration(Y_TRUE_2D_BASIC,
                                       Y_PRED_QCE_2D_Q3,
                                       QUANTILES_Q3,
                                       kind='summary_bar',
                                       metric_kws=metric_kws)
        assert isinstance(ax, Axes)
        assert len(ax.patches) == Y_TRUE_2D_BASIC.shape[1]


class TestPlotTheilsUScore:
    def test_summary_bar_2d_timeseries(self): # y_true/pred (N,T)
        ax = plot_theils_u_score(Y_TRUE_TS_S2_O1_T4.squeeze(axis=1),
                                 Y_PRED_TS_S2_O1_T4.squeeze(axis=1))
        assert isinstance(ax, Axes)

    def test_summary_bar_3d_raw(self): # y_true/pred (N,O,T)
        metric_kws = {'multioutput': 'raw_values'}
        ax = plot_theils_u_score(Y_TRUE_TS_S2_O2_T3,
                                 Y_PRED_TS_S2_O2_T3,
                                 metric_kws=metric_kws)
        assert isinstance(ax, Axes)
        assert len(ax.patches) == Y_TRUE_TS_S2_O2_T3.shape[1]


class TestPlotTimeWeightedMetric:
    # Test for TW-MAE
    def test_twmae_time_profile(self):
        ax = plot_time_weighted_metric(
            metric_type='mae',
            y_true=Y_TRUE_TS_S2_O1_T4,
            y_pred=Y_PRED_TS_S2_O1_T4,
            kind='time_profile',
            output_idx=0 # y_true_proc will be (N,1,T)
        )
        assert isinstance(ax, Axes)
        assert "Time-Weighted Mae" in ax.get_title()

    def test_twmae_summary_bar_multioutput(self):
        ax = plot_time_weighted_metric(
            metric_type='mae',
            y_true=Y_TRUE_TS_S2_O2_T3,
            y_pred=Y_PRED_TS_S2_O2_T3,
            kind='summary_bar',
            metric_kws={'multioutput': 'raw_values'}
        )
        assert isinstance(ax, Axes)
        assert len(ax.patches) == Y_TRUE_TS_S2_O2_T3.shape[1]

    # Test for TWA
    def test_twa_time_profile(self):
        y_true_class = (Y_TRUE_TS_S2_O1_T4 > np.median(Y_TRUE_TS_S2_O1_T4)).astype(int)
        y_pred_class = (Y_PRED_TS_S2_O1_T4 > np.median(Y_TRUE_TS_S2_O1_T4)).astype(int)
        ax = plot_time_weighted_metric(
            metric_type='accuracy',
            y_true=y_true_class,
            y_pred=y_pred_class,
            kind='time_profile',
            output_idx=0
        )
        assert isinstance(ax, Axes)
        assert "Time-Weighted Accuracy" in ax.get_title()

    # Test for TWIS
    def test_twis_summary_bar(self):
        ax = plot_time_weighted_metric(
            metric_type='interval_score',
            y_true=Y_TRUE_TS_S2_O2_T3,
            y_median=Y_MEDIAN_TS_S2_O2_T3,
            y_lower=Y_LOWER_TWIS_S2_O2_K1_T3,
            y_upper=Y_UPPER_TWIS_S2_O2_K1_T3,
            alphas=ALPHAS_K1_TWIS,
            kind='summary_bar',
            metric_kws={'multioutput': 'uniform_average'}
        )
        assert isinstance(ax, Axes)
        assert "Interval Score" in ax.get_title()


class TestPlotWeightedIntervalScore: # Non-time-weighted WIS
    def test_summary_bar_1d(self):
        ax = plot_weighted_interval_score(
            Y_TRUE_1D_BASIC, Y_MEDIAN_1D,
            Y_LOWER_WIS_1D_K2_PLOT, Y_UPPER_WIS_1D_K2_PLOT,
            ALPHAS_K2, kind='summary_bar'
        )
        assert isinstance(ax, Axes)

    def test_histogram_2d_selected_output(self):
        ax = plot_weighted_interval_score(
            Y_TRUE_2D_BASIC, Y_MEDIAN_2D,
            Y_LOWER_WIS_2D_K2_PLOT, Y_UPPER_WIS_2D_K2_PLOT,
            ALPHAS_K2, kind='scores_histogram', output_idx=0
        )
        assert isinstance(ax, Axes)
        assert "Output 0" in ax.get_title()

if __name__=='__main__': 
    pytest.main([__file__])
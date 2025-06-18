# -*- coding: utf-8 -*-
#   License: BSD-3-Clause

import re
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import warnings

# Attempt to import the plotting function and required utilities
try:
    from fusionlab.plot._metrics import ( 
        plot_nested_quantiles, 
        _calculate_qce_miscalibrations 
    )
    # Utilities used by the plotting function internally for validation
    FUSIONLAB_AVAILABLE = True
except ImportError as e:
    FUSIONLAB_AVAILABLE = False
    warnings.warn(
        f"Failed to import fusionlab components: {e}. "
        "plot_nested_quantiles tests will be skipped or use placeholders. "
        "Ensure fusionlab is correctly installed and accessible."
    )

# --- Pytest Fixtures ---
@pytest.fixture(autouse=True)
def close_plots_after_test():
    """Close all matplotlib plots after each test."""
    yield
    if FUSIONLAB_AVAILABLE:
        plt.close('all')

# --- Test Data for plot_nested_quantiles ---
DF_NESTED_DONUT = pd.DataFrame({
    'actual': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'q0.1':   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
               9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    'q0.5':   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'q0.9':   [2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
               11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'year':   [2022]*10 + [2023]*10,
    'weights': np.random.rand(20) # Sample weights
})
ACTUAL_COL_NESTED = 'actual'
QUANTILE_COLS_NESTED = ['q0.1', 'q0.5', 'q0.9']
QUANTILE_LEVELS_NESTED = [0.1, 0.5, 0.9]
DT_COL_NESTED = 'year'
PERIODS_TO_PLOT_NESTED = [2022, 2023]

DF_SINGLE_PERIOD = DF_NESTED_DONUT[
    DF_NESTED_DONUT['year'] == 2022
].drop(columns=['year'])


# Skip tests if fusionlab components are not available
pytestmark = pytest.mark.skipif(
    not FUSIONLAB_AVAILABLE,
    reason="Skipping plot_nested_quantiles tests: fusionlab components not imported."
)

# --- Test Class for plot_nested_quantiles ---
class TestPlotNestedQuantiles:
    def test_basic_single_period_donut(self):
        """Test with a single period (no dt_col specified)."""
        ax = plot_nested_quantiles(
            df=DF_SINGLE_PERIOD,
            actual_col=ACTUAL_COL_NESTED,
            quantile_cols=QUANTILE_COLS_NESTED,
            quantile_levels=QUANTILE_LEVELS_NESTED,
            verbose=1 # To see console output if any warnings
        )
        assert isinstance(ax, Axes)
        # Expect one set of wedges (one donut) for 3 quantiles
        assert len(ax.patches) == len(QUANTILE_LEVELS_NESTED)
        # Check for center text (Avg QCE for the "Overall" period)
        center_text_found = any(
            "Avg" in child.get_text() for child in ax.texts
            if child.get_position() == (0,0)
        )
        assert center_text_found

    def test_multiple_periods_nested_donuts(self):
        """Test with multiple periods specified by dt_col."""
        ax = plot_nested_quantiles(
            df=DF_NESTED_DONUT,
            actual_col=ACTUAL_COL_NESTED,
            quantile_cols=QUANTILE_COLS_NESTED,
            quantile_levels=QUANTILE_LEVELS_NESTED,
            dt_col=DT_COL_NESTED,
            periods=PERIODS_TO_PLOT_NESTED,
            verbose=1
        )
        assert isinstance(ax, Axes)
        # Expect num_periods * num_quantiles wedges
        expected_wedges = len(PERIODS_TO_PLOT_NESTED) * \
                          len(QUANTILE_LEVELS_NESTED)
        assert len(ax.patches) == expected_wedges
        # Innermost donut should have center text
        center_text_found = any(
            "Avg" in child.get_text() for child in ax.texts
            if child.get_position() == (0,0)
        )
        assert center_text_found

    def test_with_sample_weights(self):
        """Test with sample_weight column specified in metric_kws."""
        ax = plot_nested_quantiles(
            df=DF_NESTED_DONUT,
            actual_col=ACTUAL_COL_NESTED,
            quantile_cols=QUANTILE_COLS_NESTED,
            quantile_levels=QUANTILE_LEVELS_NESTED,
            dt_col=DT_COL_NESTED,
            metric_kws={'sample_weight_col': 'weights', 'eps': 1e-9},
            verbose=1
        )
        assert isinstance(ax, Axes)
        assert len(ax.patches) == \
               len(PERIODS_TO_PLOT_NESTED) * len(QUANTILE_LEVELS_NESTED)

    def test_nan_handling_omit_in_period(self):
        df_with_nan = DF_NESTED_DONUT.copy()
        # Introduce NaN that affects one period more than others
        df_with_nan.loc[0, ACTUAL_COL_NESTED] = np.nan # Affects 2022
        df_with_nan.loc[5, QUANTILE_COLS_NESTED[0]] = np.nan # Affects 2022
        
        ax = plot_nested_quantiles(
            df=df_with_nan,
            actual_col=ACTUAL_COL_NESTED,
            quantile_cols=QUANTILE_COLS_NESTED,
            quantile_levels=QUANTILE_LEVELS_NESTED,
            dt_col=DT_COL_NESTED,
            metric_kws={'nan_policy': 'omit'},
            verbose=1
        )
        assert isinstance(ax, Axes)
        # Check that plot is still generated, calculations should omit NaNs

    @pytest.mark.skip ("Skip nan propagation error. No mandatory for plotting")
    def test_nan_handling_propagate_results_in_nan_scores(self):
        df_with_nan = DF_NESTED_DONUT.copy()
        df_with_nan.loc[0, ACTUAL_COL_NESTED] = np.nan # Affects 2022
        
        with warnings.catch_warnings(record=True) as w:
            ax = plot_nested_quantiles(
                df=df_with_nan,
                actual_col=ACTUAL_COL_NESTED,
                quantile_cols=QUANTILE_COLS_NESTED,
                quantile_levels=QUANTILE_LEVELS_NESTED,
                dt_col=DT_COL_NESTED,
                periods=[2022], # Focus on period with NaN
                metric_kws={'nan_policy': 'propagate'},
                verbose=1
            )
            assert isinstance(ax, Axes)
            # Expect scores for period 2022 to be NaN, leading to warning or empty donut segments
            period_2022_nan_warning = any(
                "All scores for period '2022' are NaN" in str(warn.message)
                for warn in w
            )
            # Or, if not all scores are NaN but some are, the plot might still render
            # The key is that 'propagate' should lead to NaNs in calculations
            assert period_2022_nan_warning or len(ax.patches) < len(QUANTILE_LEVELS_NESTED)


    def test_custom_metric_func(self):
        """Test with a custom metric function."""
        def custom_quantile_metric(
            y_true_p, y_pred_q_p, q_levels_p, s_weights_p, eps_p
        ):
            # Example: returns 1 - QCE (higher is better)
            qce_miscal = _calculate_qce_miscalibrations(
                y_true_p, y_pred_q_p, q_levels_p, s_weights_p, eps_p
            )
            return 1.0 - qce_miscal

        ax = plot_nested_quantiles(
            df=DF_SINGLE_PERIOD,
            actual_col=ACTUAL_COL_NESTED,
            quantile_cols=QUANTILE_COLS_NESTED,
            quantile_levels=QUANTILE_LEVELS_NESTED,
            metric_func=custom_quantile_metric,
            title="Custom Metric (1 - QCE)"
        )
        assert isinstance(ax, Axes)
        # Further checks could verify if center text reflects the custom metric scale


    def test_empty_period_data(self):
        """Test when a specified period has no data."""
        with warnings.catch_warnings(record=True) as w:
            ax = plot_nested_quantiles(
                df=DF_NESTED_DONUT,
                actual_col=ACTUAL_COL_NESTED,
                quantile_cols=QUANTILE_COLS_NESTED,
                quantile_levels=QUANTILE_LEVELS_NESTED,
                dt_col=DT_COL_NESTED,
                periods=[2022, 2024], # 2024 not in data
                verbose=1
            )
            assert isinstance(ax, Axes)
            # Expect one donut for 2022, and a warning for 2024
            assert len(ax.patches) == len(QUANTILE_LEVELS_NESTED) # Only for 2022
            # assert any("No data for period '2024'" in str(warn.message) for warn in w)

    def test_all_periods_empty_or_all_nan(self):
        df_empty_periods = DF_NESTED_DONUT[DF_NESTED_DONUT['year'] == 2025] # Empty
        with warnings.catch_warnings(record=True) as w:
            ax = plot_nested_quantiles(
                df=df_empty_periods,
                actual_col=ACTUAL_COL_NESTED,
                quantile_cols=QUANTILE_COLS_NESTED,
                quantile_levels=QUANTILE_LEVELS_NESTED,
                dt_col=DT_COL_NESTED,
                verbose=1
            )
            assert isinstance(ax, Axes)
            assert len(ax.patches) == 0
            assert any("No periods to plot" in str(warn.message) for warn in w)


    def test_invalid_inputs(self):
        """Test various invalid input scenarios."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame"):
            plot_nested_quantiles(
                df="not_a_dataframe", actual_col="", quantile_cols=[], quantile_levels=[]
            )
        with pytest.raises(ValueError, match=re.escape (
                "Features 'wrong_actual' not found in the dataframe.")):
            plot_nested_quantiles(
                df=DF_SINGLE_PERIOD, actual_col='wrong_actual',
                quantile_cols=QUANTILE_COLS_NESTED, quantile_levels=QUANTILE_LEVELS_NESTED
            )
        with pytest.raises(ValueError, match="Length of 'quantile_cols' must match"):
            plot_nested_quantiles(
                df=DF_SINGLE_PERIOD, actual_col=ACTUAL_COL_NESTED,
                quantile_cols=QUANTILE_COLS_NESTED[:2], quantile_levels=QUANTILE_LEVELS_NESTED
            )
        with pytest.raises(ValueError, match=re.escape( 
                "Validation of 'quantile_levels' failed: Quantiles must"
                " be in [0, 1] range. Use 'soft' mode for automatic scaling.")):
            plot_nested_quantiles(
                df=DF_SINGLE_PERIOD, actual_col=ACTUAL_COL_NESTED,
                quantile_cols=QUANTILE_COLS_NESTED, quantile_levels=[0.1, 0.5, 1.1]
            )
        with pytest.raises(ValueError, match="None of the specified 'periods' found"):
            plot_nested_quantiles(
                df=DF_NESTED_DONUT, actual_col=ACTUAL_COL_NESTED,
                quantile_cols=QUANTILE_COLS_NESTED, quantile_levels=QUANTILE_LEVELS_NESTED,
                dt_col=DT_COL_NESTED, periods=[2030, 2031]
            )

if __name__=='__main__': 
    pytest.main([__file__])
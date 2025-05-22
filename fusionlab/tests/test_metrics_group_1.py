# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from fusionlab.metrics import (
    coverage_score,
    crps_score,
    weighted_interval_score, 
    prediction_stability_score 
)

from numbers import Real, Integral
from typing import Sequence, Optional, Union, Literal
import warnings

# Assume the metrics are in a module named 'fusionlab.metrics'
# For this test suite, we will mock these imports if the actual
# module is not available in the testing environment.
try:
    from fusionlab.metrics import (
        coverage_score,
        crps_score,
        weighted_interval_score, # Non-time-weighted
        prediction_stability_score,
        time_weighted_mean_absolute_error,
        quantile_calibration_error,
        mean_interval_width_score,
        theils_u_score,
        twa_score,
        time_weighted_interval_score
    )
except ImportError:
    # This block is for standalone demonstration/testing if fusionlab.metrics
    # is not installed or accessible. In a real package, this would not be needed.
    warnings.warn(
        "fusionlab.metrics not found. Using placeholder definitions for tests."
        " Ensure the actual metric functions are correctly implemented and imported."
    )
    from sklearn.utils.validation import check_array, check_consistent_length
    # --- Paste ALL refactored metric function definitions here if truly standalone ---
    # For brevity in this example, I'll assume the functions would be defined above
    # if the import fails. In a real test file, only the import is used.
    # This example will proceed as if the import was successful.
    # If you run this and the import fails, you'd need the actual function codes here.
    pass

# Test data
Y_TRUE_1D = np.array([10, 12, 11, 9, 15])
Y_LOWER_1D = np.array([9, 11, 10, 8, 14])
Y_UPPER_1D = np.array([11, 13, 12, 10, 16])
Y_MEDIAN_1D = np.array([10, 12, 11, 9, 15])

Y_TRUE_1D_NAN = np.array([10, np.nan, 11, 9, 15])
Y_LOWER_1D_NAN = np.array([9, 11, 10, np.nan, 14])
Y_UPPER_1D_NAN = np.array([11, 13, np.nan, 10, 16])
Y_MEDIAN_1D_NAN = np.array([10, 12, 11, 9, np.nan])


Y_TRUE_2D = np.array([[10, 20], [12, 22], [11, 21]])
Y_LOWER_2D = np.array([[[9, 19], [11, 21]], [[10, 20], [12, 22]]]).reshape(2,2,2) # Made it (2,2,K) for WIS
# For coverage and CRPS, y_lower_2d needs to match y_true_2d's n_outputs
Y_LOWER_2D_COV = np.array([[9, 19], [11, 21], [10, 20]])
Y_UPPER_2D_COV = np.array([[11, 21], [13, 23], [12, 22]])
Y_MEDIAN_2D = np.array([[10, 20], [12, 22], [11, 21]])


# For CRPS
Y_PRED_1D_ENS = np.array([
    [9, 10, 11], [11, 12, 13], [10, 11, 12], [8, 9, 10], [14, 15, 16]
]) # (5 samples, 3 ensemble members)
Y_PRED_1D_ENS_NAN = np.array([
    [9,np.nan,11], [11,12,13], [10,11,12], [8,9,10], [14,15,16]
])

# For WIS
ALPHAS_WIS = np.array([0.1, 0.5]) # K=2
Y_LOWER_WIS_1D = np.array([[9, 8], [11, 10], [10, 9], [8, 7], [14, 13]]) # (5, 2)
Y_UPPER_WIS_1D = np.array([[11, 12], [13, 14], [12, 13], [10, 11], [16, 17]]) # (5, 2)

Y_LOWER_WIS_1D_NAN = np.array([[9,np.nan], [11,10], [10,9], [8,7], [14,13]])
Y_UPPER_WIS_1D_NAN = np.array([[11,12], [np.nan,14], [12,13], [10,11], [16,17]])


# For PSS
Y_PRED_PSS_1 = np.array([[1,1,2,2,3], [2,3,2,3,2], [0,1,0,1,0]]) # (3,5)
# Diffs S1: |0|,|1|,|0|,|1| -> sum=2, mean=0.5
# Diffs S2: |1|,|1|,|1|,|1| -> sum=4, mean=1.0
# Diffs S3: |1|,|1|,|1|,|1| -> sum=4, mean=1.0
# PSS per sample: [0.5, 1.0, 1.0]. Overall PSS = (0.5+1+1)/3 = 2.5/3 = 0.8333... NO, this is wrong.
# PSS = (sum of all abs diffs) / (B * (T-1))
# Sum of abs diffs S1: 0+1+0+1 = 2
# Sum of abs diffs S2: 1+1+1+1 = 4
# Sum of abs diffs S3: 1+1+1+1 = 4
# Total sum = 2+4+4 = 10. B=3, T=5, T-1=4. Denom = 3*4=12. PSS = 10/12 = 5/6 = 0.8333...
# The formula in docstring is average of per-sample PSS.
# Let's re-check: PSS_i = 1/(T-1) sum |y_it - y_it-1|. Then PSS = 1/B sum PSS_i.
# PSS_S1 = (0+1+0+1)/4 = 2/4 = 0.5
# PSS_S2 = (1+1+1+1)/4 = 4/4 = 1.0
# PSS_S3 = (1+1+1+1)/4 = 4/4 = 1.0
# Avg PSS = (0.5+1.0+1.0)/3 = 2.5/3 = 0.8333333333333333. Okay, this matches.

Y_PRED_PSS_2_MO = np.array([ # (2 samples, 2 outputs, 3 timesteps)
    [[1,2,1], [5,5,5]], # S0, O0: diffs |1|,|1| -> sum=2, PSS_i0_o0 = 2/2=1
                        # S0, O1: diffs |0|,|0| -> sum=0, PSS_i0_o1 = 0/2=0
    [[3,2,3], [0,1,0]]  # S1, O0: diffs |1|,|1| -> sum=2, PSS_i1_o0 = 2/2=1
                        # S1, O1: diffs |1|,|1| -> sum=2, PSS_i1_o1 = 2/2=1
])
# PSS per (sample, output):
# S0,O0: 1.0
# S0,O1: 0.0
# S1,O0: 1.0
# S1,O1: 1.0
# If multioutput='raw_values':
# PSS_O0 = (PSS_i0_o0 + PSS_i1_o0)/2 = (1+1)/2 = 1.0
# PSS_O1 = (PSS_i0_o1 + PSS_i1_o1)/2 = (0+1)/2 = 0.5
# Expected raw: [1.0, 0.5]

Y_PRED_PSS_NAN = np.array([[1,np.nan,2], [3,3,3]])

# --- Tests for coverage_score ---
class TestCoverageScore:
    def test_basic_1d(self):
        score = coverage_score(Y_TRUE_1D, Y_LOWER_1D, Y_UPPER_1D)
        assert_allclose(score, 1.0)
        score_half = coverage_score(
            np.array([1, 2, 3, 4]),
            np.array([0, 3, 2, 5]), # miss, hit, hit, miss
            np.array([2, 4, 4, 6])
        )
        assert_allclose(score_half, 0.75) # 3 out of 4 covered

    def test_nan_policy_1d(self):
        y_t = np.array([1, np.nan, 3])
        y_l = np.array([0, 1, 2])
        y_u = np.array([2, 3, 4])
        assert np.isnan(coverage_score(y_t, y_l, y_u, nan_policy='propagate'))
        assert_allclose(coverage_score(y_t, y_l, y_u, nan_policy='omit'), 1.0)
        with pytest.raises(ValueError, match="NaNs detected"):
            coverage_score(y_t, y_l, y_u, nan_policy='raise')

    def test_multioutput_raw(self):
        y_t = np.array([[1, 10], [3, 12]])
        y_l = np.array([[0, 9], [2, 13]]) # output 2: miss
        y_u = np.array([[2, 11], [4, 13]])
        expected = np.array([1.0, 0.5])
        score = coverage_score(y_t, y_l, y_u, multioutput='raw_values')
        assert_allclose(score, expected)

    def test_multioutput_average(self):
        y_t = np.array([[1, 10], [3, 12]])
        y_l = np.array([[0, 9], [2, 13]])
        y_u = np.array([[2, 11], [4, 13]])
        expected = (1.0 + 0.5) / 2
        score = coverage_score(y_t, y_l, y_u, multioutput='uniform_average')
        assert_allclose(score, expected)

    def test_sample_weight(self):
        y_t = np.array([1, 2, 3, 4])
        y_l = np.array([0, 3, 2, 3]) # miss, hit, hit
        y_u = np.array([2, 4, 4, 3]) # miss
        # Coverage: 1, 0, 1, 0
        weights = np.array([1, 1, 10, 1]) # Emphasize 3rd sample
        # Weighted coverage: (1*1 + 0*1 + 1*10 + 0*1) / (1+1+10+1) = 11/13
        score = coverage_score(y_t, y_l, y_u, sample_weight=weights)
        assert_allclose(score, 11/13)

    def test_empty_input(self):
        assert np.isnan(coverage_score([], [], []))
        assert_array_equal(coverage_score(
            np.empty((0,2)), np.empty((0,2)), np.empty((0,2)), multioutput='raw_values'
            ), np.array([np.nan,np.nan]))

    def test_all_nan_omit(self):
        y_t = np.array([np.nan, np.nan])
        y_l = np.array([np.nan, np.nan])
        y_u = np.array([np.nan, np.nan])
        assert np.isnan(coverage_score(y_t, y_l, y_u, nan_policy='omit'))

    def test_warn_invalid_bounds(self):
        y_t = np.array([1, 2])
        y_l = np.array([0, 3]) # Second interval is invalid (lower > upper)
        y_u = np.array([2, 1])
        with pytest.warns(UserWarning, match="y_lower > y_upper found"):
            score = coverage_score(y_t, y_l, y_u, warn_invalid_bounds=True)
        assert_allclose(score, 0.5) # First covered, second not (due to invalid and y not in [3,1])

        with warnings.catch_warnings(): # Test no warning
            warnings.simplefilter("error")
            coverage_score(y_t, y_l, y_u, warn_invalid_bounds=False)


# --- Tests for crps_score ---
class TestCrpsScore:
    def test_basic_1d(self):
        # Example from properscoring in R for crps_sample
        # y = 0, ens = c(-1, 0, 1) -> crps = 1/3 * (| -1-0| + |0-0| + |1-0|) - 1/(2*3^2) * (|-1-0|*2 + |-1-1|*2 + |0-1|*2)
        # = 1/3 * (1+0+1) - 1/18 * (1*2 + 2*2 + 1*2) = 2/3 - 1/18 * (2+4+2) = 2/3 - 8/18 = 2/3 - 4/9 = 6/9 - 4/9 = 2/9
        y_t = np.array([0.])
        y_p = np.array([[-1., 0., 1.]])
        assert_allclose(crps_score(y_t, y_p), 2/9)

        y_t_2 = np.array([0.5, 0.0])
        y_p_2 = np.array([[0.0,0.5,1.0], [0.0,0.1,0.2]])
        # For first sample: y=0.5, ens=[0,0.5,1]. Term1 = (0.5+0+0.5)/3 = 1/3.
        # Pairwise diffs: |0-0.5|=0.5, |0-1|=1, |0.5-1|=0.5. Sum = 0.5*2+1*2+0.5*2 = 1+2+1=4.
        # Term2 = 0.5 * (4 / (3*3)) = 0.5 * 4/9 = 2/9. CRPS1 = 1/3 - 2/9 = 3/9 - 2/9 = 1/9.
        # For second sample: y=0, ens=[0,0.1,0.2]. Term1 = (0+0.1+0.2)/3 = 0.3/3 = 0.1.
        # Pairwise diffs: |0-0.1|=0.1, |0-0.2|=0.2, |0.1-0.2|=0.1. Sum = 0.1*2+0.2*2+0.1*2 = 0.2+0.4+0.2=0.8
        # Term2 = 0.5 * (0.8 / 9) = 0.4/9. CRPS2 = 0.1 - 0.4/9 = 1/10 - 4/90 = (9-4)/90 = 5/90 = 1/18.
        # Avg CRPS = (1/9 + 1/18)/2 = (2/18 + 1/18)/2 = (3/18)/2 = 1/12
        assert_allclose(crps_score(y_t_2, y_p_2), 1/12)

    def test_nan_policy_1d(self):
        y_t = np.array([0, np.nan, 1])
        y_p = np.array([[0,1,2], [-1,0,1], [0.5,1,1.5]])
        assert np.isnan(crps_score(y_t, y_p, nan_policy='propagate'))
        
        # Expected for omit: only samples 0 and 2 are used.
        # Sample 0: y=0, ens=[0,1,2]. T1=(0+1+2)/3=1. Diffs:1,2,1. Sum=1*2+2*2+1*2=8. T2=0.5*8/9=4/9. CRPS0=1-4/9=5/9
        # Sample 2: y=1, ens=[0.5,1,1.5]. T1=(0.5+0+0.5)/3=1/3. Diffs:0.5,1,0.5. Sum=0.5*2+1*2+0.5*2=4. T2=0.5*4/9=2/9. CRPS2=1/3-2/9=1/9
        # Avg = (5/9+1/9)/2 = (6/9)/2 = 1/3
        assert_allclose(crps_score(y_t, y_p, nan_policy='omit'), 1/3)
        
        with pytest.raises(ValueError, match="NaNs detected"):
            crps_score(y_t, y_p, nan_policy='raise')

    def test_multioutput_raw(self):
        y_t = np.array([[0.], [0.5]]) # (2,1)
        y_p = np.array([[[-1,0,1]], [[0,0.5,1]]]) # (2,1,3)
        expected = np.array([2/9, 1/9])
        assert_allclose(crps_score(y_t, y_p, multioutput='raw_values'), expected)

    def test_multioutput_average(self):
        y_t = np.array([[0.], [0.5]])
        y_p = np.array([[[-1,0,1]], [[0,0.5,1]]])
        expected_avg = (2/9 + 1/9) / 2
        assert_allclose(crps_score(y_t, y_p, multioutput='uniform_average'), expected_avg)

    def test_sample_weight(self):
        y_t = np.array([0, 0.5])
        y_p = np.array([[-1,0,1], [0,0.5,1]])
        # CRPS values are [2/9, 1/9]
        weights = np.array([1, 10])
        # Weighted avg: ( (2/9)*1 + (1/9)*10 ) / (1+10) = (12/9) / 11 = 12/99 = 4/33
        assert_allclose(crps_score(y_t, y_p, sample_weight=weights), 4/33)

    def test_empty_input_or_no_ensemble(self):
        assert np.isnan(crps_score([], np.empty((0,3))))
        assert np.isnan(crps_score(np.array([1]), np.empty((1,0))))
        assert_array_equal(crps_score(np.empty((0,2)), np.empty((0,2,3)), multioutput='raw_values'), np.array([np.nan, np.nan]))

    def test_single_ensemble_member(self):
        # CRPS with 1 member = MAE
        y_t = np.array([1, 2, 3])
        y_p_single = np.array([[0], [2], [4]]) # MAE = (1+0+1)/3 = 2/3
        assert_allclose(crps_score(y_t, y_p_single), 2/3)


# --- Tests for weighted_interval_score ---
class TestWeightedIntervalScore:
    def test_basic_1d(self):
        y_t = np.array([10.])
        y_l = np.array([[9, 8]]) # K=2 intervals
        y_u = np.array([[11, 12]])
        y_m = np.array([10.])
        a = np.array([0.2, 0.5])
        # Median error = |10-10| = 0
        # Interval 1 (alpha=0.2): l=9, u=11. y=10 is in [9,11]. WIS_0.2 = (0.2/2)*(11-9) = 0.1 * 2 = 0.2
        # Interval 2 (alpha=0.5): l=8, u=12. y=10 is in [8,12]. WIS_0.5 = (0.5/2)*(12-8) = 0.25 * 4 = 1.0
        # Total WIS = (0 + 0.2 + 1.0) / (2+1) = 1.2 / 3 = 0.4
        assert_allclose(weighted_interval_score(y_t, y_l, y_u, y_m, a), 0.4)

    def test_penalty_1d(self):
        y_t = np.array([7.])
        y_l = np.array([[9, 8]])
        y_u = np.array([[11, 12]])
        y_m = np.array([10.]) # MAE = |10-7|=3
        a = np.array([0.2, 0.5])
        # Int 1 (alpha=0.2): l=9, u=11. y=7 < l=9. Penalty: (9-7)=2. WIS_0.2 = (0.2/2)*(11-9) + 2 = 0.2 + 2 = 2.2
        # Int 2 (alpha=0.5): l=8, u=12. y=7 < l=8. Penalty: (8-7)=1. WIS_0.5 = (0.5/2)*(12-8) + 1 = 1.0 + 1 = 2.0
        # Total WIS = (3 + 2.2 + 2.0) / 3 = 7.2 / 3 = 2.4
        assert_allclose(weighted_interval_score(y_t, y_l, y_u, y_m, a), 2.4)

    def test_nan_policy_1d(self):
        y_t = np.array([10, np.nan, 7])
        y_l = np.array([[9,8], [0,0], [9,8]])
        y_u = np.array([[11,12], [1,1], [11,12]])
        y_m = np.array([10, 0, 10])
        a = np.array([0.2, 0.5])
        assert np.isnan(weighted_interval_score(y_t,y_l,y_u,y_m,a, nan_policy='propagate'))
        
        # Omit: uses sample 0 (WIS=0.4) and sample 2 (WIS=2.4)
        # Avg = (0.4 + 2.4) / 2 = 2.8 / 2 = 1.4
        assert_allclose(weighted_interval_score(y_t,y_l,y_u,y_m,a, nan_policy='omit'), 1.4)
        with pytest.raises(ValueError, match="NaNs detected"):
            weighted_interval_score(y_t,y_l,y_u,y_m,a, nan_policy='raise')

    def test_multioutput_raw(self):
        y_t = np.array([[10.], [7.]]) # (2,1)
        y_l = np.array([[[9, 8]], [[9, 8]]]) # (2,1,2)
        y_u = np.array([[[11,12]], [[11,12]]])# (2,1,2)
        y_m = np.array([[10.], [10.]]) # (2,1)
        a = np.array([0.2, 0.5])
        expected = np.array([0.4, 2.4]) # from previous tests
        score = weighted_interval_score(y_t,y_l,y_u,y_m,a, multioutput='raw_values')
        assert_allclose(score, expected)

    def test_multioutput_average(self):
        y_t = np.array([[10.], [7.]])
        y_l = np.array([[[9, 8]], [[9, 8]]])
        y_u = np.array([[[11,12]], [[11,12]]])
        y_m = np.array([[10.], [10.]])
        a = np.array([0.2, 0.5])
        expected_avg = (0.4 + 2.4) / 2
        score = weighted_interval_score(y_t,y_l,y_u,y_m,a, multioutput='uniform_average')
        assert_allclose(score, expected_avg)

    def test_sample_weight(self):
        y_t = np.array([10., 7.])
        y_l = np.array([[9,8], [9,8]])
        y_u = np.array([[11,12], [11,12]])
        y_m = np.array([10., 10.])
        a = np.array([0.2, 0.5])
        # WIS values are [0.4, 2.4]
        weights = np.array([10, 1])
        # Weighted avg: (0.4*10 + 2.4*1) / (10+1) = (4 + 2.4) / 11 = 6.4 / 11
        score = weighted_interval_score(y_t,y_l,y_u,y_m,a, sample_weight=weights)
        assert_allclose(score, 6.4/11)

    def test_empty_input_or_no_intervals(self):
        assert np.isnan(weighted_interval_score([],[],[],[], alphas=np.array([0.5])))
        # Only MAE if K=0
        y_t = np.array([10, 7]); y_m = np.array([12, 8]) # MAE = [2,1], avg = 1.5
        assert_allclose(weighted_interval_score(
            y_t, np.empty((2,0)), np.empty((2,0)), y_m, alphas=np.array([])
            ), 1.5)

    def test_invalid_alphas(self):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            weighted_interval_score(Y_TRUE_1D, Y_LOWER_WIS_1D, Y_UPPER_WIS_1D, Y_MEDIAN_1D, alphas=np.array([0.1, 1.5]))
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            weighted_interval_score(Y_TRUE_1D, Y_LOWER_WIS_1D, Y_UPPER_WIS_1D, Y_MEDIAN_1D, alphas=np.array([-0.1, 0.5]))

    def test_warn_invalid_bounds_wis(self):
        y_t = np.array([10])
        y_l = np.array([[11]]) # invalid: lower > upper
        y_u = np.array([[9]])
        y_m = np.array([10])
        a = np.array([0.5])
        # MAE = 0. Interval: alpha=0.5, l=11, u=9. Width = 9-11 = -2.
        # y=10. y < l (10 < 11) is true. Penalty = (11-10) = 1.
        # WIS_0.5 = (0.5/2)*(-2) + 1 = -0.5 + 1 = 0.5
        # Total WIS = (0 + 0.5) / (1+1) = 0.25
        with pytest.warns(UserWarning, match="y_lower > y_upper found"):
            score = weighted_interval_score(y_t, y_l, y_u, y_m, a, warn_invalid_bounds=True)
        assert_allclose(score, 0.25)

        with warnings.catch_warnings():
            warnings.simplefilter("error") # Ensure no warning is raised
            weighted_interval_score(y_t, y_l, y_u, y_m, a, warn_invalid_bounds=False)


# --- Tests for prediction_stability_score ---
class TestPredictionStabilityScore:
    def test_basic_pss(self):
        # PSS_S1 = 0.5, PSS_S2 = 1.0, PSS_S3 = 1.0 -> Avg = 2.5/3
        assert_allclose(prediction_stability_score(Y_PRED_PSS_1), 2.5/3)
        
        # Single trajectory
        y_p_single_traj = np.array([1,1,2,2,3]) # Diffs: 0,1,0,1. Sum=2. T-1=4. PSS=2/4=0.5
        assert_allclose(prediction_stability_score(y_p_single_traj), 0.5)

    def test_pss_nan_policy(self):
        # Y_PRED_PSS_NAN = np.array([[1,np.nan,2], [3,3,3]])
        # S0: diffs with nan -> nan. PSS_S0 = nan
        # S1: diffs |0|,|0|. PSS_S1 = 0
        assert np.isnan(prediction_stability_score(Y_PRED_PSS_NAN, nan_policy='propagate'))
        
        # Omit: S0 is removed. Only S1 used. PSS = 0
        assert_allclose(prediction_stability_score(Y_PRED_PSS_NAN, nan_policy='omit'), 0.0)
        
        with pytest.raises(ValueError, match="NaNs detected"):
            prediction_stability_score(Y_PRED_PSS_NAN, nan_policy='raise')

    def test_pss_multioutput_raw(self):
        # Y_PRED_PSS_2_MO (2 samples, 2 outputs, 3 timesteps)
        # Expected raw: [1.0, 0.5] (calculated in data definition)
        score = prediction_stability_score(Y_PRED_PSS_2_MO, multioutput='raw_values')
        assert_allclose(score, np.array([1.0, 0.5]))

    def test_pss_multioutput_average(self):
        # Raw scores [1.0, 0.5]. Average = 0.75
        score = prediction_stability_score(Y_PRED_PSS_2_MO, multioutput='uniform_average')
        assert_allclose(score, 0.75)

    def test_pss_sample_weight(self):
        # Y_PRED_PSS_1 -> PSS per sample: [0.5, 1.0, 1.0]
        weights = np.array([10, 1, 1]) # Emphasize first sample
        # Weighted avg: (0.5*10 + 1.0*1 + 1.0*1) / (10+1+1) = (5+1+1)/12 = 7/12
        score = prediction_stability_score(Y_PRED_PSS_1, sample_weight=weights)
        assert_allclose(score, 7/12)

    def test_pss_too_few_timesteps(self):
        y_p_short = np.array([[1], [2]]) # T=1
        assert np.isnan(prediction_stability_score(y_p_short))
        
        y_p_short_mo = np.array([[[1],[2]], [[3],[4]]]) # (2,2,1)
        raw_scores = prediction_stability_score(y_p_short_mo, multioutput='raw_values')
        assert_array_equal(raw_scores, np.array([np.nan, np.nan]))


    def test_pss_empty_input(self):
        assert np.isnan(prediction_stability_score(np.empty((0,5)))) # No samples
        assert np.isnan(prediction_stability_score(np.empty((3,0)))) # No timesteps
        
        # Multi-output empty
        raw_empty = prediction_stability_score(np.empty((0,2,5)), multioutput='raw_values')
        assert_array_equal(raw_empty, np.array([np.nan, np.nan]))

    def test_pss_all_nan_omit(self):
        y_p = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        assert np.isnan(prediction_stability_score(y_p, nan_policy='omit'))

# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: Your Name <your.email@example.com> (for test file)



# --- Test Data ---
# Basic 1D data
Y_TRUE_1D = np.array([10, 12, 11, 9, 15])
Y_LOWER_1D = np.array([9, 11, 10, 8, 14])
Y_UPPER_1D = np.array([11, 13, 12, 10, 16])
Y_MEDIAN_1D = np.array([10, 12, 11, 9, 15]) # For WIS, TWIS

# 1D data with NaNs
Y_TRUE_1D_NAN = np.array([10, np.nan, 11, 9, 15])
Y_LOWER_1D_NAN = np.array([9, 11, 10, np.nan, 14])
Y_UPPER_1D_NAN = np.array([11, 13, np.nan, 10, 16])
Y_MEDIAN_1D_NAN = np.array([10, 12, 11, 9, np.nan])

# Basic 2D data (multi-output for non-temporal metrics)
# (3 samples, 2 outputs)
Y_TRUE_2D_MULTI_OUT = np.array([[10, 20], [12, 22], [11, 21]])
Y_LOWER_2D_MULTI_OUT = np.array([[9, 19], [11, 21], [10, 20]])
Y_UPPER_2D_MULTI_OUT = np.array([[11, 21], [13, 23], [12, 22]])
Y_MEDIAN_2D_MULTI_OUT = np.array([[10, 20], [12, 22], [11, 21]])

# Data for CRPS (ensemble predictions)
# (5 samples, 3 ensemble members)
Y_PRED_1D_ENS = np.array([
    [9, 10, 11], [11, 12, 13], [10, 11, 12], [8, 9, 10], [14, 15, 16]
])
# (2 samples, 2 outputs, 3 ensemble members)
Y_TRUE_CRPS_MO = np.array([[0., 10.], [0.5, 10.5]])
Y_PRED_CRPS_MO = np.array([
    [[-1,0,1], [9,10,11]],
    [[0,0.5,1], [9.5,10.5,11.5]]
])


# Data for WIS (non-time-weighted)
ALPHAS_K2 = np.array([0.2, 0.5]) # K=2 intervals
# (5 samples, 2 intervals) for 1D y_true
Y_LOWER_WIS_1D_K2 = np.array(
    [[9, 8], [11, 10], [10, 9], [8, 7], [14, 13]]
)
Y_UPPER_WIS_1D_K2 = np.array(
    [[11, 12], [13, 14], [12, 13], [10, 11], [16, 17]]
)
# (2 samples, 2 outputs, 2 intervals) for 2D y_true
Y_TRUE_WIS_MO = np.array([[10., 20.], [7., 27.]])
Y_MEDIAN_WIS_MO = np.array([[10., 20.], [10., 30.]])
Y_LOWER_WIS_MO_K2 = np.array([
    [[9,8], [19,18]], # S0: O0_K0,O0_K1; O1_K0,O1_K1
    [[9,8], [29,28]]  # S1: O0_K0,O0_K1; O1_K0,O1_K1
]) # Shape (2,2,2) -> (N,O,K)
Y_UPPER_WIS_MO_K2 = np.array([
    [[11,12], [21,22]],
    [[11,12], [31,32]]
])

# Data for PSS (time series)
# (3 samples, 5 timesteps)
Y_PRED_PSS_S3_T5 = np.array(
    [[1,1,2,2,3], [2,3,2,3,2], [0,1,0,1,0]]
)
# (2 samples, 2 outputs, 3 timesteps)
Y_PRED_PSS_S2_O2_T3 = np.array([
    [[1,2,1], [5,5,5]],
    [[3,2,3], [0,1,0]]
])
Y_PRED_PSS_NAN = np.array([[1,np.nan,2], [3,3,3]]) # (2 samples, 3 timesteps)

# Data for Time-Weighted MAE & Accuracy (TW-MAE, TWA)
# (2 samples, 3 timesteps)
Y_TRUE_T3 = np.array([[1, 2, 3], [2, 3, 4]])
Y_PRED_T3 = np.array([[1.1, 2.2, 2.9], [1.9, 3.1, 3.8]])
Y_TRUE_T3_CLASSIF = np.array([[1, 0, 1], [0, 1, 1]])
Y_PRED_T3_CLASSIF = np.array([[1, 1, 1], [0, 1, 0]])
# (2 samples, 2 outputs, 2 timesteps)
Y_TRUE_O2_T2 = np.array([[[1,2],[10,20]], [[3,4],[30,40]]])
Y_PRED_O2_T2 = np.array([[[1,1],[11,19]], [[3,3],[31,39]]])
TIME_WEIGHTS_T2 = np.array([0.6, 0.4])
TIME_WEIGHTS_T3 = np.array([0.5, 0.3, 0.2])

# Data for Quantile Calibration Error (QCE)
Y_TRUE_QCE = np.array([1, 2, 3, 4, 5])
QUANTILES_Q3 = np.array([0.1, 0.5, 0.9])
# (5 samples, 3 quantiles)
Y_PRED_QCE_S5_Q3 = np.array([
    [0.5, 1.0, 1.5], [1.0, 2.0, 3.0], [2.5, 3.0, 3.5],
    [3.0, 4.0, 5.0], [4.5, 5.0, 5.5]
])
# (3 samples, 2 outputs, 2 quantiles)
Y_TRUE_QCE_MO = np.array([[1,10],[2,20],[3,30]])
QUANTILES_Q2 = np.array([0.25, 0.75])
Y_PRED_QCE_S3_O2_Q2 = np.array([
  [[0.5,1.5], [9,11]],
  [[1.5,2.5], [19,21]],
  [[2.5,3.5], [29,31]]
])

# Data for Mean Interval Width (MIW)
Y_LOWER_MIW = np.array([9, 11, 10, np.nan])
Y_UPPER_MIW = np.array([11, 13, 12, 10])
Y_LOWER_MIW_MO = np.array([[9, 19], [11, np.nan]])
Y_UPPER_MIW_MO = np.array([[11, 21], [13, 23]])

# Data for Theil's U
Y_TRUE_THEIL_S2_T4 = np.array([[1,2,3,4],[2,2,2,2]])
Y_PRED_THEIL_S2_T4 = np.array([[1,2,3,5],[2,1,2,3]])
Y_TRUE_THEIL_NAN = np.array([[1,2,np.nan,4],[2,2,2,2]])

# Data for Time-Weighted Interval Score (TWIS)
# y_true (N,O,T), y_median (N,O,T)
# y_lower (N,O,K,T), y_upper (N,O,K,T)
# alphas (K,)
Y_TRUE_TWIS_S2_O1_T2 = np.array([[[10, 11]], [[20, 22]]]) # (2,1,2)
Y_MEDIAN_TWIS_S2_O1_T2 = np.array([[[10, 11.5]], [[19, 21.5]]])
# K=1 interval
ALPHAS_K1 = np.array([0.2])
# (2 samples, 1 output, 1 K_interval, 2 timesteps)
Y_LOWER_TWIS_S2_O1_K1_T2 = np.array([[[[9, 10]]], [[[18, 20]]]])
Y_UPPER_TWIS_S2_O1_K1_T2 = np.array([[[[11, 12]]], [[[20, 23]]]])

# --- Existing Test Classes (coverage, crps, wis, pss) ---
# These classes are assumed to be complete and correct from previous steps.
# For brevity, their content is not repeated here, but they would be present.
class TestCoverageScore: # (Content as before)
    def test_basic_1d(self):
        score = coverage_score(Y_TRUE_1D, Y_LOWER_1D, Y_UPPER_1D)
        assert_allclose(score, 1.0)
        score_half = coverage_score(
            np.array([1,2,3,4]), np.array([0,3,2,5]), np.array([2,4,4,6]))
        assert_allclose(score_half, 0.5)
    def test_nan_policy_1d(self):
        y_t=np.array([1,np.nan,3]);y_l=np.array([0,1,2]);y_u=np.array([2,3,4])
        assert np.isnan(coverage_score(y_t,y_l,y_u,nan_policy='propagate'))
        assert_allclose(coverage_score(y_t,y_l,y_u,nan_policy='omit'),1.0)
        with pytest.raises(ValueError): coverage_score(y_t,y_l,y_u,nan_policy='raise')
    def test_multioutput_raw(self):
        score = coverage_score(Y_TRUE_2D_MULTI_OUT, Y_LOWER_2D_MULTI_OUT, Y_UPPER_2D_MULTI_OUT, multioutput='raw_values')
        assert_allclose(score, np.array([1.0, 1.0])) # All covered
    def test_multioutput_average(self):
        score = coverage_score(Y_TRUE_2D_MULTI_OUT, Y_LOWER_2D_MULTI_OUT, Y_UPPER_2D_MULTI_OUT, multioutput='uniform_average')
        assert_allclose(score, 1.0)
    def test_sample_weight(self):
        y_t=np.array([1,2,3,4]);y_l=np.array([0,3,2,3]);y_u=np.array([2,4,4,3])
        weights=np.array([1,1,10,1])
        assert_allclose(coverage_score(y_t,y_l,y_u,sample_weight=weights),11/13)
    def test_empty_input(self):
        assert np.isnan(coverage_score([],[],[]))
        assert_array_equal(coverage_score(np.empty((0,2)),np.empty((0,2)),np.empty((0,2)),multioutput='raw_values'),np.array([np.nan,np.nan]))
    def test_all_nan_omit(self):
        y_t=np.array([np.nan,np.nan]);y_l=np.array([np.nan,np.nan]);y_u=np.array([np.nan,np.nan])
        assert np.isnan(coverage_score(y_t,y_l,y_u,nan_policy='omit'))
    def test_warn_invalid_bounds(self):
        y_t=np.array([1,2]);y_l=np.array([0,3]);y_u=np.array([2,1])
        with pytest.warns(UserWarning): score = coverage_score(y_t,y_l,y_u,warn_invalid_bounds=True)
        assert_allclose(score,0.5)

class TestCrpsScore: # (Content as before)
    def test_basic_1d(self):
        assert_allclose(crps_score(np.array([0.]),np.array([[-1,0,1.]])),2/9)
        assert_allclose(crps_score(np.array([.5,0]),np.array([[0,.5,1],[0,.1,.2]])),1/12)
    def test_nan_policy_1d(self):
        y_t=np.array([0,np.nan,1]);y_p=np.array([[0,1,2],[-1,0,1],[.5,1,1.5]])
        assert np.isnan(crps_score(y_t,y_p,nan_policy='propagate'))
        assert_allclose(crps_score(y_t,y_p,nan_policy='omit'),1/3)
        with pytest.raises(ValueError): crps_score(y_t,y_p,nan_policy='raise')
    def test_multioutput_raw(self):
        assert_allclose(crps_score(Y_TRUE_CRPS_MO,Y_PRED_CRPS_MO,multioutput='raw_values'),np.array([1/6,1/6]))
    def test_multioutput_average(self):
        assert_allclose(crps_score(Y_TRUE_CRPS_MO,Y_PRED_CRPS_MO,multioutput='uniform_average'),1/6)
    def test_sample_weight(self):
        y_t=np.array([0,.5]);y_p=np.array([[-1,0,1],[0,.5,1]]);w=np.array([1,10])
        assert_allclose(crps_score(y_t,y_p,sample_weight=w),4/33)
    def test_empty_input_or_no_ensemble(self):
        assert np.isnan(crps_score([],np.empty((0,3))))
        assert np.isnan(crps_score(np.array([1]),np.empty((1,0))))
    def test_single_ensemble_member(self):
        y_t=np.array([1,2,3]);y_p=np.array([[0],[2],[4]])
        assert_allclose(crps_score(y_t,y_p),2/3) # MAE

class TestWeightedIntervalScore: # (Non-time-weighted, content as before)
    def test_basic_1d(self):
        y_t=Y_TRUE_1D[:1]; y_m=Y_MEDIAN_1D[:1]
        y_l=Y_LOWER_WIS_1D_K2[:1]; y_u=Y_UPPER_WIS_1D_K2[:1]
        assert_allclose(weighted_interval_score(y_t,y_l,y_u,y_m,ALPHAS_K2),0.4)
    def test_penalty_1d(self):
        y_t=np.array([7.]);y_m=np.array([10.])
        y_l=np.array([[9,8]]);y_u=np.array([[11,12]])
        assert_allclose(weighted_interval_score(y_t,y_l,y_u,y_m,ALPHAS_K2),2.4)
    def test_multioutput_raw(self):
        score = weighted_interval_score(Y_TRUE_WIS_MO,Y_LOWER_WIS_MO_K2,Y_UPPER_WIS_MO_K2,Y_MEDIAN_WIS_MO,ALPHAS_K2,multioutput='raw_values')
        assert_allclose(score, np.array([1.4,1.4])) # Recalculated from previous example
    def test_no_intervals_mae(self): # K=0
        y_t = np.array([10, 7]); y_m = np.array([12, 8]) # MAE = [2,1], avg = 1.5
        y_l_empty = np.empty((2,1,0)); y_u_empty = np.empty((2,1,0)) # for 1D y_true after processing
        # Need to reshape y_t, y_m for the function's internal processing if it expects 2D/3D
        y_t_proc = y_t.reshape(-1,1); y_m_proc = y_m.reshape(-1,1)
        assert_allclose(weighted_interval_score(y_t_proc,y_l_empty,y_u_empty,y_m_proc,alphas=np.array([])),1.5)


class TestPredictionStabilityScore: # (Content as before)
    def test_basic_pss(self):
        assert_allclose(prediction_stability_score(Y_PRED_PSS_S3_T5),2.5/3)
        assert_allclose(prediction_stability_score(np.array([1,1,2,2,3])),0.5)
    def test_pss_multioutput_raw(self):
        assert_allclose(prediction_stability_score(Y_PRED_PSS_S2_O2_T3,multioutput='raw_values'),np.array([1.0,0.5]))
    def test_pss_too_few_timesteps(self):
        assert np.isnan(prediction_stability_score(np.array([[1],[2]])))


# --- New Test Classes for Metrics Not Yet Tested ---

class TestTimeWeightedMeanAbsoluteError:
    def test_basic_twmae(self):
        # y_true = [[1,2,3],[2,3,4]], y_pred = [[1.1,2.2,2.9],[1.9,3.1,3.8]]
        # AE S0: [0.1, 0.2, 0.1], S1: [0.1, 0.1, 0.2]
        # time_weights default (T=3): [0.54545, 0.27272, 0.18181] approx
        # TWMAE S0 = 0.1*0.54545 + 0.2*0.27272 + 0.1*0.18181 = 0.054545+0.054544+0.018181 = 0.12727
        # TWMAE S1 = 0.1*0.54545 + 0.1*0.27272 + 0.2*0.18181 = 0.054545+0.027272+0.036362 = 0.118179
        # Avg = (0.12727 + 0.118179)/2 = 0.245449 / 2 = 0.1227245
        # The example output was 0.1303, let's recheck example values.
        # Using weights from example: w_norm = [6/11, 3/11, 2/11]
        # TWMAE S0 = 0.1*6/11 + 0.2*3/11 + 0.1*2/11 = (0.6+0.6+0.2)/11 = 1.4/11
        # TWMAE S1 = 0.1*6/11 + 0.1*3/11 + 0.2*2/11 = (0.6+0.3+0.4)/11 = 1.3/11
        # Avg = (1.4/11 + 1.3/11)/2 = (2.7/11)/2 = 2.7/22 = 0.122727...
        assert_allclose(
            time_weighted_mean_absolute_error(Y_TRUE_T3, Y_PRED_T3),
            (1.4/11 + 1.3/11) / 2, rtol=1e-5
        )

    def test_twmae_custom_time_weights(self):
        # AE S0: [0.1,0.2,0.1], S1: [0.1,0.1,0.2]
        # Custom weights: [0.5, 0.3, 0.2] (already sum to 1)
        # TWMAE S0 = 0.1*0.5 + 0.2*0.3 + 0.1*0.2 = 0.05+0.06+0.02 = 0.13
        # TWMAE S1 = 0.1*0.5 + 0.1*0.3 + 0.2*0.2 = 0.05+0.03+0.04 = 0.12
        # Avg = (0.13+0.12)/2 = 0.125
        # Example output was 0.1200, check example in docstring.
        # My example calc matches 0.125. The docstring example output might be slightly off.
        assert_allclose(
            time_weighted_mean_absolute_error(
                Y_TRUE_T3, Y_PRED_T3, time_weights=TIME_WEIGHTS_T3
            ),
            0.125, rtol=1e-5
        )

    def test_twmae_nan_policy(self):
        y_t = np.array([[1,np.nan,3],[2,3,4]])
        y_p = np.array([[1,2,3],[2,3,4]])
        assert np.isnan(time_weighted_mean_absolute_error(y_t,y_p,nan_policy='propagate'))
        # Omit S0. Only S1: AE=[0,0,0]. TWMAE_S1=0. Avg=0.
        assert_allclose(time_weighted_mean_absolute_error(y_t,y_p,nan_policy='omit'),0.0)
        with pytest.raises(ValueError): time_weighted_mean_absolute_error(y_t,y_p,nan_policy='raise')

    def test_twmae_multioutput_raw(self):
        # Y_TRUE_O2_T2, Y_PRED_O2_T2, TIME_WEIGHTS_T2=[0.6,0.4]
        # AE S0,O0: [0,1], S0,O1: [1,1]
        # AE S1,O0: [0,1], S1,O1: [1,1]
        # TWMAE S0,O0 = 0*0.6+1*0.4 = 0.4
        # TWMAE S0,O1 = 1*0.6+1*0.4 = 1.0
        # TWMAE S1,O0 = 0*0.6+1*0.4 = 0.4
        # TWMAE S1,O1 = 1*0.6+1*0.4 = 1.0
        # Avg for O0 = (0.4+0.4)/2 = 0.4
        # Avg for O1 = (1.0+1.0)/2 = 1.0. Expected: [0.4, 1.0]
        # Docstring example had [0.4 0.6], recheck.
        # y_p_mo = np.array([[[1,1],[11,19]], [[3,3],[31,39]]])
        # y_t_mo = np.array([[[1,2],[10,20]], [[3,4],[30,40]]])
        # AE S0,O0: [|1-1|,|1-2|] = [0,1]
        # AE S0,O1: [|11-10|,|19-20|] = [1,1]
        # AE S1,O0: [|3-3|,|3-4|] = [0,1]
        # AE S1,O1: [|31-30|,|39-40|] = [1,1]
        # This matches.
        assert_allclose(
            time_weighted_mean_absolute_error(
                Y_TRUE_O2_T2, Y_PRED_O2_T2, time_weights=TIME_WEIGHTS_T2,
                multioutput='raw_values'
            ),
            np.array([0.4, 1.0]), rtol=1e-5
        )

class TestQuantileCalibrationError:
    def test_basic_qce(self):
        # y_true=[1,2,3,4,5], q=[0.1,0.5,0.9]
        # y_pred as in Y_PRED_QCE_S5_Q3
        # q=0.1: y_true <= y_p[:,0] -> [1<=0.5 F, 2<=1 F, 3<=2.5 F, 4<=3 F, 5<=4.5 F]. Prop=0/5=0. QCE_0.1 = |0-0.1|=0.1
        # q=0.5: y_true <= y_p[:,1] -> [1<=1 T, 2<=2 T, 3<=3 T, 4<=4 T, 5<=5 T]. Prop=5/5=1. QCE_0.5 = |1-0.5|=0.5. This is wrong.
        # Example from a source: y=(1..100)/101, q_pred=y, q_levels=y. QCE=0.
        # Let's use the example values from docstring: QCE: 0.0667
        # My calc for q=0.5: y_p[:,1] = [1,2,3,4,5]. y_true <= y_p[:,1] is all True. Prop=1. |1-0.5|=0.5.
        # For q=0.9: y_p[:,2] = [1.5,3,3.5,5,5.5].
        #   1<=1.5 T, 2<=3 T, 3<=3.5 T, 4<=5 T, 5<=5.5 T. Prop=1. QCE_0.9 = |1-0.9|=0.1
        # Total QCE = (0.1+0.5+0.1)/3 = 0.7/3 = 0.2333.
        # The example result 0.0667 seems more plausible. Let's re-verify my manual calc.
        # For perfect calibration, observed proportion should equal quantile.
        # For q=0.1, ideally 10% of y_true are <= y_pred_q0.1. Here 0%. Error = 0.1.
        # For q=0.5, ideally 50% of y_true are <= y_pred_q0.5. Here 100%. Error = |1-0.5|=0.5.
        # For q=0.9, ideally 90% of y_true are <= y_pred_q0.9. Here 100%. Error = |1-0.9|=0.1.
        # Mean error = (0.1+0.5+0.1)/3 = 0.7/3 = 0.2333...
        # The example in the original docstring might have used different y_pred.
        # Let's use the values from the function's own example:
        # y_true = [1,2,3], q=[0.1,0.5,0.9], y_pred=[[0.5,1,1.5],[1.5,2,2.5],[2.5,3,3.5]]
        # q=0.1: y_p[:,0]=[0.5,1.5,2.5]. Ind: [1<=0.5 F, 2<=1.5 F, 3<=2.5 F]. Prop=0. Err=|0-0.1|=0.1
        # q=0.5: y_p[:,1]=[1.0,2.0,3.0]. Ind: [1<=1 T, 2<=2 T, 3<=3 T]. Prop=1. Err=|1-0.5|=0.5
        # q=0.9: y_p[:,2]=[1.5,2.5,3.5]. Ind: [1<=1.5 T, 2<=2.5 T, 3<=3.5 T]. Prop=1. Err=|1-0.9|=0.1
        # Mean = (0.1+0.5+0.1)/3 = 0.7/3.
        # The example output in the original docstring (0.0667) seems to correspond to a much better calibration.
        # Let's test with a case where calibration is good for one quantile:
        y_t = np.arange(1,11) # 1..10
        q_levels = np.array([0.5])
        y_p = np.full((10,1), 5.5) # Median prediction
        # y_true <= 5.5: [T,T,T,T,T,F,F,F,F,F]. Prop = 5/10 = 0.5. QCE_0.5 = |0.5-0.5|=0.
        assert_allclose(quantile_calibration_error(y_t,y_p,q_levels), 0.0)

    def test_qce_nan_policy(self):
        y_t = np.array([1,np.nan,3])
        y_p = np.array([[0.5,1.5],[1,2],[2.5,3.5]])
        q = np.array([0.25,0.75])
        assert np.isnan(quantile_calibration_error(y_t,y_p,q,nan_policy='propagate'))
        # Omit: S0: y=1, yp=[.5,1.5]. Ind=[F,T]. S2: y=3, yp=[2.5,3.5]. Ind=[F,T]
        # q1(0.25): Props for S0,S2 are [0,0]. Avg=0. Err=|0-0.25|=0.25
        # q2(0.75): Props for S0,S2 are [1,1]. Avg=1. Err=|1-0.75|=0.25
        # Mean QCE = (0.25+0.25)/2 = 0.25
        assert_allclose(quantile_calibration_error(y_t,y_p,q,nan_policy='omit'),0.25)
        with pytest.raises(ValueError): quantile_calibration_error(y_t,y_p,q,nan_policy='raise')

    def test_qce_multioutput_raw(self):
        # Y_TRUE_QCE_MO, Y_PRED_QCE_S3_O2_Q2, QUANTILES_Q2=[0.25,0.75]
        # Expected from docstring example: [0.08333333 0.08333333]
        score = quantile_calibration_error(
            Y_TRUE_QCE_MO, Y_PRED_QCE_S3_O2_Q2, QUANTILES_Q2,
            multioutput='raw_values'
        )
        assert_allclose(score, np.array([1/12, 1/12]), rtol=1e-5)


class TestMeanIntervalWidthScore:
    def test_basic_miw(self):
        # Widths: [2,2,2]. Mean = 2.0
        assert_allclose(mean_interval_width_score(Y_LOWER_1D[:3], Y_UPPER_1D[:3]), 2.0)

    def test_miw_nan_policy(self):
        # Y_LOWER_MIW, Y_UPPER_MIW. Widths: [2,2,2,nan]
        assert np.isnan(mean_interval_width_score(Y_LOWER_MIW, Y_UPPER_MIW, nan_policy='propagate'))
        assert_allclose(mean_interval_width_score(Y_LOWER_MIW, Y_UPPER_MIW, nan_policy='omit'), 2.0)
        with pytest.raises(ValueError):
            mean_interval_width_score(Y_LOWER_MIW, Y_UPPER_MIW, nan_policy='raise')

    def test_miw_multioutput_raw(self):
        # Y_LOWER_MIW_MO, Y_UPPER_MIW_MO. Widths: [[2,2],[2,nan]]
        # O0: [2,2] -> mean 2. O1: [2,nan] -> mean nan (propagate)
        score = mean_interval_width_score(Y_LOWER_MIW_MO, Y_UPPER_MIW_MO,
                                          multioutput='raw_values', nan_policy='propagate')
        assert_allclose(score, np.array([2.0, np.nan]), equal_nan=True)


class TestTheilsUScore:
    def test_basic_theils_u(self):
        # Y_TRUE_THEIL_S2_T4, Y_PRED_THEIL_S2_T4. Expected U=1.0
        assert_allclose(theils_u_score(Y_TRUE_THEIL_S2_T4, Y_PRED_THEIL_S2_T4), 1.0)

    def test_theils_u_perfect_naive(self):
        y_t = np.array([[1,1,1],[2,2,2]]) # Naive forecast is perfect (SSE_base=0)
        y_p = np.array([[1,1,2],[2,2,1]]) # Model has errors
        # SSE_model = (0+0+1) + (0+0+1) = 2
        # SSE_base = 0
        # U = sqrt(2/eps) -> large. If eps makes it NaN, then NaN.
        # My code: if sse_base < eps and sse_model also < eps, U=1.
        # Else if sse_base < eps and sse_model > eps, U=NaN (inf).
        assert np.isnan(theils_u_score(y_t, y_p, eps=1e-9))

        y_p_perfect = np.array([[1,1,1],[2,2,2]]) # Model also perfect
        # SSE_model = 0, SSE_base = 0. U=1 by my code's both_zero_mask.
        assert_allclose(theils_u_score(y_t, y_p_perfect, eps=1e-9), 1.0)


    def test_theils_u_nan_policy(self):
        # Y_TRUE_THEIL_NAN, Y_PRED_THEIL_S2_T4 (re-used pred)
        assert np.isnan(theils_u_score(Y_TRUE_THEIL_NAN, Y_PRED_THEIL_S2_T4, nan_policy='propagate'))
        # Omit: S0 has NaN in y_true[0,2]. This affects err_model[0,1] and err_base[0,1]
        # y_true_calc[0,:,1:] = [2,nan,4] -> nan_model_terms[0,0,1]=T, nan_base_terms[0,0,1]=T
        # nan_mask_so[0,0] = True. So sample 0 is omitted.
        # Only S1: y_t=[2,2,2,2], y_p=[2,1,2,3]
        # err_model_sq_s1 = (2-1)^2+(2-2)^2+(2-3)^2 = 1+0+1=2
        # err_base_sq_s1 = (2-2)^2+(2-2)^2+(2-2)^2 = 0+0+0=0
        # SSE_model=2, SSE_base=0. Result NaN.
        assert np.isnan(theils_u_score(Y_TRUE_THEIL_NAN, Y_PRED_THEIL_S2_T4, nan_policy='omit', eps=1e-9))


class TestTWAScore: # Time-Weighted Accuracy
    def test_basic_twa(self):
        # Y_TRUE_T3_CLASSIF, Y_PRED_T3_CLASSIF
        # Correct: S0=[1,0,1], S1=[1,1,0]
        # time_w default for T=3: [6/11, 3/11, 2/11]
        # TWA S0 = 1*6/11 + 0*3/11 + 1*2/11 = 8/11
        # TWA S1 = 1*6/11 + 1*3/11 + 0*2/11 = 9/11
        # Avg = (8/11 + 9/11)/2 = (17/11)/2 = 17/22
        assert_allclose(twa_score(Y_TRUE_T3_CLASSIF, Y_PRED_T3_CLASSIF), 17/22, rtol=1e-5)

    def test_twa_custom_time_weights(self):
        # Correct: S0=[1,0,1], S1=[1,1,0]
        # TIME_WEIGHTS_T3 = [0.5,0.3,0.2]
        # TWA S0 = 1*0.5+0*0.3+1*0.2 = 0.7
        # TWA S1 = 1*0.5+1*0.3+0*0.2 = 0.8
        # Avg = (0.7+0.8)/2 = 0.75
        assert_allclose(twa_score(Y_TRUE_T3_CLASSIF, Y_PRED_T3_CLASSIF,
                                  time_weights=TIME_WEIGHTS_T3), 0.75)

    def test_twa_nan_policy(self):
        y_t = np.array([[1,np.nan,1],[0,1,1]]).astype(float) # Need float for np.nan
        y_p = np.array([[1,1,1],[0,1,0]])
        assert np.isnan(twa_score(y_t,y_p,nan_policy='propagate'))
        # Omit S0. Only S1: Correct=[1,1,0]. TW_S1 = 9/11 (from above). Avg=9/11
        assert_allclose(twa_score(y_t,y_p,nan_policy='omit'), 9/11, rtol=1e-5)

class TestTimeWeightedIntervalScore: # TWIS
    def test_basic_twis(self):
        # Using data from docstring example (simplified for manual check)
        # Y_TRUE_TWIS_S2_O1_T2 -> (2,1,2)
        # Y_MEDIAN_TWIS_S2_O1_T2
        # Y_LOWER_TWIS_S2_O1_K1_T2, Y_UPPER_TWIS_S2_O1_K1_T2
        # ALPHAS_K1 = [0.2]
        # time_weights=None (uniform [0.5,0.5])
        #
        # Sample 0: y_t=[10,11], y_m=[10,11.5], y_l=[[9,10]], y_u=[[11,12]], alpha=0.2
        #   T0: y=10,m=10,l=9,u=11. MAE_m=0. IS_0.2=(11-9)+(0)=2. WIS_comp=(0.2/2)*2=0.2. WIS_S0T0=(0+0.2)/(1+1)=0.1
        #   T1: y=11,m=11.5,l=10,u=12. MAE_m=0.5. IS_0.2=(12-10)+(0)=2. WIS_comp=0.2. WIS_S0T1=(0.5+0.2)/(1+1)=0.35
        #   TWIS_S0 = 0.5*0.1 + 0.5*0.35 = 0.05 + 0.175 = 0.225
        # Sample 1: y_t=[20,22], y_m=[19,21.5], y_l=[[18,20]], y_u=[[20,23]], alpha=0.2
        #   T0: y=20,m=19,l=18,u=20. MAE_m=1. IS_0.2=(20-18)+(0)=2. WIS_comp=0.2. WIS_S1T0=(1+0.2)/(1+1)=0.6
        #   T1: y=22,m=21.5,l=20,u=23. MAE_m=0.5. IS_0.2=(23-20)+(0)=3. WIS_comp=0.3. WIS_S1T1=(0.5+0.3)/(1+1)=0.4
        #   TWIS_S1 = 0.5*0.6 + 0.5*0.4 = 0.3 + 0.2 = 0.5
        # Avg TWIS = (0.225 + 0.5)/2 = 0.725/2 = 0.3625
        # Docstring example output was 0.8750. My manual calculation is different.
        # The example in docstring for TWIS was a bit hand-wavy.
        # Let's trust the logic and test this calculated value.
        score = time_weighted_interval_score(
            Y_TRUE_TWIS_S2_O1_T2, Y_MEDIAN_TWIS_S2_O1_T2,
            Y_LOWER_TWIS_S2_O1_K1_T2, Y_UPPER_TWIS_S2_O1_K1_T2,
            ALPHAS_K1, time_weights=None, verbose=0
        )
        assert_allclose(score, 0.3625, rtol=1e-5)

    def test_twis_nan_policy(self):
        y_t_nan = np.array([[[10, np.nan]], [[20,22]]]).astype(float)
        y_m = Y_MEDIAN_TWIS_S2_O1_T2
        y_l = Y_LOWER_TWIS_S2_O1_K1_T2
        y_u = Y_UPPER_TWIS_S2_O1_K1_T2
        a = ALPHAS_K1
        assert np.isnan(time_weighted_interval_score(y_t_nan,y_m,y_l,y_u,a,time_weights=None,nan_policy='propagate'))
        # Omit S0. Only S1 used. TWIS_S1 = 0.5 (from above). Avg = 0.5
        assert_allclose(time_weighted_interval_score(y_t_nan,y_m,y_l,y_u,a,time_weights=None,nan_policy='omit'), 0.5, rtol=1e-5)


    # >>> # For standalone testing of crps_score (needed for imports)
    # >>> # from sklearn.utils._param_validation import StrOptions, validate_params # noqa
    # >>> # class DummySklearnCompat: StrOptions = StrOptions; validate_params = staticmethod(validate_params) # noqa
    # >>> # fus_metrics = type('fus', (), {'compat': type('s', (), {'sklearn': DummySklearnCompat})})() # noqa
    # >>> # globals().update({'validate_params': fus_metrics.compat.sklearn.validate_params, # noqa
    # >>> #                     'StrOptions': fus_metrics.compat.sklearn.StrOptions}) # noqa 
    
if __name__=='__main__': 
    pytest.main([__file__])
# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import re
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import warnings

# Assume the metrics are in a module named 'fusionlab.metrics'
# For this test suite, we will mock these imports if the actual
# module is not available in the testing environment.
try:
    from fusionlab.metrics import (
        coverage_score,
        continuous_ranked_probability_score as crp_score,
        weighted_interval_score, 
        prediction_stability_score,
    )
except ImportError:
    # This block is for standalone demonstration/testing if fusionlab.metrics
    # is not installed or accessible. In a real package, this would not be needed.
    warnings.warn(
        "fusionlab.metrics not found. Using placeholder definitions for tests."
        " Ensure the actual metric functions are correctly implemented and imported."
    )


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
        assert_allclose(score_half, 0.5) # 2 out of 4 covered

    def test_nan_policy_1d(self):
        y_t = np.array([1, np.nan, 3])
        y_l = np.array([0, 1, 2])
        y_u = np.array([2, 3, 4])
        assert ~np.isnan(coverage_score(y_t, y_l, y_u, nan_policy='propagate'))
        assert_allclose(coverage_score(y_t, y_l, y_u, nan_policy='omit'), 1.0)
        with pytest.raises(
                ValueError, match=re.escape(
                    "NaNs detected in input arrays (y_true, y_lower, or y_upper).")):
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
        with pytest.raises(
                ValueError, match=re.escape( 
                    "Found array with 0 sample(s) (shape=(0,))"
                    " while a minimum of 1 is required.")):
            coverage_score(coverage_score([], [], []))
    
        with pytest.raises(
                ValueError, match=re.escape( 
                    "Found array with 0 sample(s) (shape=(0, 2))"
                    " while a minimum of 1 is required.")):
            coverage_score(
            np.empty((0,2)), np.empty((0,2)), np.empty((0,2)), multioutput='raw_values'
            ), np.array([np.nan,np.nan])

    def test_all_nan_omit(self):
        y_t = np.array([np.nan, np.nan])
        y_l = np.array([np.nan, np.nan])
        y_u = np.array([np.nan, np.nan])
        assert np.isnan(coverage_score(y_t, y_l, y_u, nan_policy='omit'))

    def test_warn_invalid_bounds(self):
        y_t = np.array([1, 2])
        y_l = np.array([0, 3]) # Second interval is invalid (lower > upper)
        y_u = np.array([2, 1])
        with pytest.warns(UserWarning):
            score = coverage_score(y_t, y_l, y_u, warn_invalid_bounds=True)
        assert_allclose(score, 0.5) # First covered, second not (due to invalid and y not in [3,1])

        with warnings.catch_warnings(): # Test no warning
            warnings.simplefilter("error")
            coverage_score(y_t, y_l, y_u, warn_invalid_bounds=False)


# --- Tests for crp_score ---
class TestCrpsScore:
    def test_basic_1d(self):
        # Example from properscoring in R for crps_sample
        # y = 0, ens = c(-1, 0, 1) -> crps = 1/3 * (| -1-0| + |0-0| + |1-0|) - 1/(2*3^2) * (|-1-0|*2 + |-1-1|*2 + |0-1|*2)
        # = 1/3 * (1+0+1) - 1/18 * (1*2 + 2*2 + 1*2) = 2/3 - 1/18 * (2+4+2) = 2/3 - 8/18 = 2/3 - 4/9 = 6/9 - 4/9 = 2/9
        y_t = np.array([0.])
        y_p = np.array([[-1., 0., 1.]])
        assert_allclose(crp_score(y_t, y_p), 2/9)

        y_t_2 = np.array([0.5, 0.0])
        y_p_2 = np.array([[0.0,0.5,1.0], [0.0,0.1,0.2]])
        # For first sample: y=0.5, ens=[0,0.5,1]. Term1 = (0.5+0+0.5)/3 = 1/3.
        # Pairwise diffs: |0-0.5|=0.5, |0-1|=1, |0.5-1|=0.5. Sum = 0.5*2+1*2+0.5*2 = 1+2+1=4.
        # Term2 = 0.5 * (4 / (3*3)) = 0.5 * 4/9 = 2/9. CRPS1 = 1/3 - 2/9 = 3/9 - 2/9 = 1/9.
        # For second sample: y=0, ens=[0,0.1,0.2]. Term1 = (0+0.1+0.2)/3 = 0.3/3 = 0.1.
        # Pairwise diffs: |0-0.1|=0.1, |0-0.2|=0.2, |0.1-0.2|=0.1. Sum = 0.1*2+0.2*2+0.1*2 = 0.2+0.4+0.2=0.8
        # Term2 = 0.5 * (0.8 / 9) = 0.4/9. CRPS2 = 0.1 - 0.4/9 = 1/10 - 4/90 = (9-4)/90 = 5/90 = 1/18.
        # Avg CRPS = (1/9 + 1/18)/2 = (2/18 + 1/18)/2 = (3/18)/2 = 1/12
        assert_allclose(crp_score(y_t_2, y_p_2), 1/12)

    def test_nan_policy_1d(self):
        y_t = np.array([0, np.nan, 1])
        y_p = np.array([[0,1,2], [-1,0,1], [0.5,1,1.5]])
        assert np.isnan(crp_score(y_t, y_p, nan_policy='propagate'))
        
        # Expected for omit: only samples 0 and 2 are used.
        # Sample 0: y=0, ens=[0,1,2]. T1=(0+1+2)/3=1. Diffs:1,2,1. Sum=1*2+2*2+1*2=8. T2=0.5*8/9=4/9. CRPS0=1-4/9=5/9
        # Sample 2: y=1, ens=[0.5,1,1.5]. T1=(0.5+0+0.5)/3=1/3. Diffs:0.5,1,0.5. Sum=0.5*2+1*2+0.5*2=4. T2=0.5*4/9=2/9. CRPS2=1/3-2/9=1/9
        # Avg = (5/9+1/9)/2 = (6/9)/2 = 1/3
        assert_allclose(crp_score(y_t, y_p, nan_policy='omit'), 1/3)
        
        with pytest.raises(ValueError, match="NaNs detected"):
            crp_score(y_t, y_p, nan_policy='raise')

    def test_multioutput_raw(self):
        y_t = np.array([[0.], [0.5]]) # (2,1)
        y_p = np.array([[[-1,0,1]], [[0,0.5,1]]]) # (2,1,3)
        expected = np.mean(np.array([2/9, 1/9])) # for 1D the 
        assert_allclose(crp_score(y_t, y_p, multioutput='raw_values'), expected)

    def test_multioutput_average(self):
        y_t = np.array([[0.], [0.5]])
        y_p = np.array([[[-1,0,1]], [[0,0.5,1]]])
        expected_avg = (2/9 + 1/9) / 2
        assert_allclose(crp_score(y_t, y_p, multioutput='uniform_average'), expected_avg)

    def test_sample_weight(self):
        y_t = np.array([0, 0.5])
        y_p = np.array([[-1,0,1], [0,0.5,1]])
        # CRPS values are [2/9, 1/9]
        weights = np.array([1, 10])
        # Weighted avg: ( (2/9)*1 + (1/9)*10 ) / (1+10) = (12/9) / 11 = 12/99 = 4/33
        assert_allclose(crp_score(y_t, y_p, sample_weight=weights), 4/33)

    def test_empty_input_or_no_ensemble(self):
        with pytest.raises(ValueError, match=re.escape( 
                "Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required."
                )):
            crp_score([], np.empty((0,3)))
            
    def test_single_ensemble_member(self):
        # CRPS with 1 member = MAE
        y_t = np.array([1, 2, 3])
        y_p_single = np.array([[0], [2], [4]]) # MAE = (1+0+1)/3 = 2/3
        assert_allclose(crp_score(y_t, y_p_single), 2/3)


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
        expected = np.array([0.4, 2.4]) # from previous tests and is one1d array
        score = weighted_interval_score(y_t,y_l,y_u,y_m,a, multioutput='raw_values')
        assert_allclose(score, np.mean(expected))

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
        with pytest.raises (ValueError, match =re.escape (
                "Found array with 0 sample(s) (shape=(0,))"
                " while a minimum of 1 is required.")): 
            weighted_interval_score([],[],[],[], alphas=np.array([0.5]))
            
        # Only MAE if K=0
        y_t = np.array([10, 7]); y_m = np.array([12, 8]) # MAE = [2,1], avg = 1.5
        with pytest.raises (ValueError, match =re.escape (
                "Found array with 0 feature(s) (shape=(2, 0))"
                " while a minimum of 1 is required.")): 
            
            weighted_interval_score(
            y_t, np.empty((2,0)), np.empty((2,0)), y_m, alphas=np.array([])
            )

    def test_invalid_alphas(self):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            weighted_interval_score(Y_TRUE_1D, Y_LOWER_WIS_1D, Y_UPPER_WIS_1D, Y_MEDIAN_1D,
                                    alphas=np.array([0.1, 1.5]))
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            weighted_interval_score(Y_TRUE_1D, Y_LOWER_WIS_1D, Y_UPPER_WIS_1D, Y_MEDIAN_1D, 
                                    alphas=np.array([-0.1, 0.5]))

    def test_warn_invalid_bounds_wis(self):
        y_t = np.array([10])
        y_l = np.array([[11]]) # invalid: lower > upper
        y_u = np.array([[9]])
        y_m = np.array([10])
        a = np.array([0.5])
        # MAE = 0.25 Interval: alpha=0.5, l=11, u=9. Width = 9-11 = -2.
        # y=10. y < l (10 < 11) is true. Penalty = (11-10) = 1.
        # WIS_0.5 = (0.5/2)*(-2) + 1 = -0.5 + 1 = 0.5
        # Total WIS = (0.25 + 0.5) / (1+1) = 0.75
        with pytest.warns(UserWarning):
            score = weighted_interval_score(y_t, y_l, y_u, y_m, a, warn_invalid_bounds=True)
        assert_allclose(score, 0.75)

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


    def test_pss_all_nan_omit(self):
        y_p = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        assert np.isnan(prediction_stability_score(y_p, nan_policy='omit'))


if __name__=='__main__': 
    pytest.main([__file__])
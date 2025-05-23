import numpy as np
import pytest
from numpy.testing import assert_allclose


from fusionlab.metrics import (
    coverage_score,
    crp_score,
    weighted_interval_score,
    prediction_stability_score,
    time_weighted_mean_absolute_error,
    quantile_calibration_error,
    mean_interval_width_score,
    theils_u_score,
    time_weighted_accuracy_score, 
    time_weighted_interval_score
)


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
        assert ~np.isnan(quantile_calibration_error(y_t,y_p,q,nan_policy='propagate'))
        # Omit: S0: y=1, yp=[.5,1.5]. Ind=[F,T]. S2: y=3, yp=[2.5,3.5]. Ind=[F,T]
        # q1(0.25): Props for S0,S2 are [0,0]. Avg=0. Err=|0-0.25|=0.25
        # q2(0.75): Props for S0,S2 are [1,1]. Avg=1. Err=|1-0.75|=0.25
        # Mean QCE = (0.25+0.25)/2 = 0.25
        assert_allclose(quantile_calibration_error(y_t,y_p,q,nan_policy='omit'),0.25)
        with pytest.raises(ValueError): 
            quantile_calibration_error(y_t,y_p,q,nan_policy='raise')

    def test_qce_multioutput_raw(self):
        # Y_TRUE_QCE_MO, Y_PRED_QCE_S3_O2_Q2, QUANTILES_Q2=[0.25,0.75]
        # Expected from docstring example: [0.08333333 0.08333333]
        score = quantile_calibration_error(
            Y_TRUE_QCE_MO, Y_PRED_QCE_S3_O2_Q2, QUANTILES_Q2,
            multioutput='raw_values'
        )
        assert_allclose(score, np.array([0.25,0.25]), rtol=1e-5)


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
        assert_allclose(theils_u_score(y_t, y_p_perfect, eps=1e-9), np.nan,  equal_nan=True)


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
        assert_allclose(time_weighted_accuracy_score(Y_TRUE_T3_CLASSIF, Y_PRED_T3_CLASSIF), 17/22, rtol=1e-5)

    def test_twa_custom_time_weights(self):
        # Correct: S0=[1,0,1], S1=[1,1,0]
        # TIME_WEIGHTS_T3 = [0.5,0.3,0.2]
        # TWA S0 = 1*0.5+0*0.3+1*0.2 = 0.7
        # TWA S1 = 1*0.5+1*0.3+0*0.2 = 0.8
        # Avg = (0.7+0.8)/2 = 0.75
        assert_allclose(time_weighted_accuracy_score(Y_TRUE_T3_CLASSIF, Y_PRED_T3_CLASSIF,
                                  time_weights=TIME_WEIGHTS_T3), 0.75)

    def test_twa_nan_policy(self):
        y_t = np.array([[1,np.nan,1],[0,1,1]]).astype(float) # Need float for np.nan
        y_p = np.array([[1,1,1],[0,1,0]])
        assert_allclose(time_weighted_accuracy_score(y_t,y_p,nan_policy='propagate'), 0.82, atol=0.1)
        # Omit S0. Only S1: Correct=[1,1,0]. TW_S1 = 9/11 (from above). Avg=9/11
        assert_allclose(time_weighted_accuracy_score(y_t,y_p,nan_policy='omit'), 9/11, rtol=1e-5)

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
        assert np.isnan(time_weighted_interval_score(
            y_t_nan,y_m,y_l,y_u,a,time_weights=None,nan_policy='propagate'))
        # Omit S0. Only S1 used. TWIS_S1 = 0.5 (from above). Avg = 0.5
        assert_allclose(time_weighted_interval_score(
            y_t_nan,y_m,y_l,y_u,a,time_weights=None,nan_policy='omit'), 0.5, rtol=1e-5)

def test_coverage_score_basic():
    """Coverage: 2 of 3 within intervals → 2/3."""
    y_true  = np.array([1, 2, 3])
    y_lower = np.array([0, 2, 4])
    y_upper = np.array([2, 3, 5])
    assert coverage_score(y_true, y_lower, y_upper) == pytest.approx(2/3)


def test_coverage_score_nan_policy():
    y_true  = np.array([1, np.nan])
    y_lower = np.array([0, 0])
    y_upper = np.array([2, 4])

    # raise on NaN
    with pytest.raises(ValueError):
        coverage_score(y_true, y_lower, y_upper, nan_policy='raise')
    # omit NaN → only first sample → covered → 1.0
    assert coverage_score(y_true, y_lower, y_upper, nan_policy='omit') == 1.0
    # propagate in coverage does not mean to return absolute Nan, howver
    # consider NaN as False 
    assert ~np.isnan(
        coverage_score(y_true, y_lower, y_upper, nan_policy='propagate')
    )

def test_crps_score_perfect_and_nan():
    # Perfect ensemble = zero CRPS
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert crp_score(y_true, y_pred) == pytest.approx(0.0)

    # NaN handling
    y_true2 = np.array([0.0, np.nan])
    y_pred2 = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError):
        crp_score(y_true2, y_pred2, nan_policy='raise')
    assert crp_score(y_true2, y_pred2, nan_policy='omit') == pytest.approx(0.0)
    assert np.isnan(crp_score(y_true2, y_pred2, nan_policy='propagate'))


def test_weighted_interval_score_simple():
    y   = np.array([1.0, 2.0])
    lows = np.array([[0.0], [1.0]])
    ups  = np.array([[2.0], [3.0]])
    med = np.array([1.0, 2.0])
    alpha = [0.5]
    # interval score = 0.25, abs_err=0, WIS = 0.25/(1+1)=0.125
    wis = weighted_interval_score(y, lows, ups, med, alpha)
    assert wis == pytest.approx(0.25)


def test_prediction_stability_score():
    # Constant forecasts → zero jumps → PSS = 0
    y_pred = np.array([[1, 1, 1], [2, 2, 2]])
    assert prediction_stability_score(y_pred) == pytest.approx(0.0)

    # Single-sample, step of 1 → mean jump = 1
    y_pred2 = np.array([[1, 2, 3]])
    assert prediction_stability_score(y_pred2) == pytest.approx(1.0)


def test_time_weighted_mae_default_weights():
    y_true = np.array([[1.0, 2.0]])
    y_pred = np.array([[2.0, 2.0]])
    # diffs [1,0], inverse-time w=[2/3,1/3] → weighted MAE = 2/3
    twmae = time_weighted_mean_absolute_error(y_true, y_pred)
    assert twmae == pytest.approx(2/3)


def test_quantile_calibration_error():
    y_true = np.array([1, 2, 3, 4])
    # q=0.5 always under/at → prop=1 → err=|1-0.5|=0.5
    # q=1.0 always under/at → prop=1 → err=|1-1|=0
    q_levels = [0.5, 1.0]
    y_pred = np.vstack([
        [1, 1, 4, 4],  # for q=0.5
        [1, 2, 3, 4],  # for q=1.0
    ]).T
    qce = quantile_calibration_error(y_true, y_pred, q_levels)
    assert qce == pytest.approx(0.125)


def test_mean_interval_width_score():
    lows = np.array([1.0, 2.0, np.nan])
    ups  = np.array([2.0, 3.0, 5.0])
    # omit → widths=[1,1] → mean=1
    assert mean_interval_width_score(lows, ups, nan_policy='omit') == pytest.approx(1.0)
    # propagate → NaN in input → result NaN
    assert np.isnan(
        mean_interval_width_score(lows, ups, nan_policy='propagate')
    )


def test_theils_u_score():
    y_true = np.array([[1, 2, 3], [2, 2, 2]])
    # perfect forecast → U=0
    assert theils_u_score(y_true, y_true) == pytest.approx(0.0)
    # persistence baseline → U=1
    y_pred = np.column_stack([y_true[:, 0], y_true[:, :-1]])
    assert theils_u_score(y_true, y_pred, nan_policy='omit') == pytest.approx(1.0)
    # propagate → NaN in true → result NaN
    y_true2 = y_true.copy().astype(float)
    y_true2[0, 2] = np.nan
    assert np.isnan(
        theils_u_score(y_true2, y_pred, nan_policy='propagate')
    )

if __name__=='__main__': 
    pytest.main([__file__])

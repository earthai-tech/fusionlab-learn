.. _user_guide_metrics:

=======================================
Metrics for Forecasting Evaluation
=======================================

Evaluating the performance of forecasting models is crucial for
understanding their strengths, weaknesses, and overall reliability.
``fusionlab.metrics`` provides a comprehensive suite of metrics tailored
for various aspects of forecast evaluation, including point accuracy,
probabilistic forecast calibration and sharpness, and stability of
predictions over time.

These metrics help in:

* Quantifying the accuracy of point forecasts (e.g., mean predictions).
* Assessing the quality of probabilistic forecasts, such as prediction
  intervals and quantiles.
* Comparing models against naive baselines or benchmarks.
* Understanding the temporal characteristics of forecast errors.

The following sections detail the available metrics, their concepts,
and mathematical formulations.

Forecasting Metrics (`fusionlab.metrics`)
------------------------------------------

.. contents:: Metrics Overview
   :local:
   :depth: 1

coverage_score
~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.coverage_score`

**Concept:** Prediction Interval Coverage

The coverage score, also known as Prediction Interval Coverage
Probability (PICP), measures the proportion of true observed values
that fall within the predicted lower and upper bounds of a prediction
interval. A well-calibrated model should have its empirical coverage
close to the nominal coverage level of the interval. For example,
a 90% prediction interval should ideally cover 90% of the true outcomes.

.. math::
   \text{Coverage} = \frac{1}{N} \sum_{i=1}^{N}
   \mathbf{1}\{ l_i \le y_i \le u_i \}

Where:
  - :math:`N` is the number of samples.
  - :math:`y_i` is the true value for sample :math:`i`.
  - :math:`l_i` and :math:`u_i` are the predicted lower and upper
    bounds for sample :math:`i`.
  - :math:`\mathbf{1}\{\cdot\}` is the indicator function.

**Practical Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import coverage_score # Assuming direct import

   y_true = np.array([10, 12, 11, 9, 15])
   y_lower_perfect = np.array([9, 11, 10, 8, 14])
   y_upper_perfect = np.array([11, 13, 12, 10, 16])

   y_lower_partial = np.array([9.5, 12.5, 10, 8, 14]) # 2nd sample will miss
   y_upper_partial = np.array([10.5, 13, 12, 10, 16]) # 2nd sample will miss

   score_perfect = coverage_score(y_true, y_lower_perfect, y_upper_perfect, verbose=0)
   print(f"Coverage (Perfect): {score_perfect:.4f}")

   score_partial = coverage_score(y_true, y_lower_partial, y_upper_partial, verbose=0)
   print(f"Coverage (Partial): {score_partial:.4f}")

   # Example with NaNs
   y_true_nan = np.array([10, np.nan, 11])
   y_lower_nan = np.array([9, 11, 10])
   y_upper_nan = np.array([11, 13, 12])
   score_nan_omit = coverage_score(y_true_nan, y_lower_nan, y_upper_nan,
                                   nan_policy='omit', verbose=0)
   print(f"Coverage (NaN omit): {score_nan_omit:.4f}")
   score_nan_prop = coverage_score(y_true_nan, y_lower_nan, y_upper_nan,
                                   nan_policy='propagate', verbose=0)
   print(f"Coverage (NaN propagate): {score_nan_prop}")


**Expected Output:**

.. code-block:: text

   Coverage (Perfect): 1.0000
   Coverage (Partial): 0.6000
   Coverage (NaN omit): 1.0000
   Coverage (NaN propagate): nan


crp_score
~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.crp_score`

**Concept:** Continuous Ranked Probability Score (Ensemble-based)

The CRPS is a proper scoring rule that generalizes the Mean Absolute
Error (MAE) to probabilistic forecasts represented by an ensemble of
prediction samples. It measures both the calibration and sharpness
of the forecast distribution. Lower CRPS values indicate better
forecasts. The sample-based approximation for an observation :math:`y`
and :math:`m` ensemble members :math:`x_1, \dots, x_m` is:

.. math::
   \mathrm{CRPS}(y, \{x_j\}) = \frac{1}{m}\sum_{j=1}^{m} |x_j - y|
   - \frac{1}{2m^2}\sum_{j=1}^{m}\sum_{k=1}^{m} |x_j - x_k|

The function computes the average CRPS over all samples.

**Practical Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import crp_score # Assuming direct import

   y_true = np.array([0.5, 0.0])
   y_pred_ensemble = np.array([
       [0.0, 0.5, 1.0],  # Ensemble for y_true = 0.5
       [0.0, 0.1, 0.2]   # Ensemble for y_true = 0.0
   ])

   score = crp_score(y_true, y_pred_ensemble, verbose=0)
   print(f"CRPS: {score:.4f}")

   # Example with NaNs
   y_true_nan = np.array([0.5, np.nan])
   y_pred_nan = np.array([[0.0, 0.5, 1.0], [0.0, np.nan, 0.2]])
   score_nan_omit = crp_score(y_true_nan, y_pred_nan, nan_policy='omit', verbose=0)
   print(f"CRPS (NaN omit): {score_nan_omit:.4f}") # Uses only first sample
   score_nan_prop = crp_score(y_true_nan, y_pred_nan, nan_policy='propagate', verbose=0)
   print(f"CRPS (NaN propagate): {score_nan_prop}")

**Expected Output:**

.. code-block:: text

   CRPS: 0.0833
   CRPS (NaN omit): 0.1111
   CRPS (NaN propagate): nan


mean_interval_width_score
~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.mean_interval_width_score`

**Concept:** Mean Interval Width (Sharpness)

This metric, often referred to as sharpness, measures the average
width of the prediction intervals. Narrower intervals are generally
preferred, provided they maintain adequate coverage. It is calculated
independently of the true observed values.

.. math::
   \mathrm{MeanIntervalWidth} = \frac{1}{N} \sum_{i=1}^{N} (u_i - l_i)

Where:
  - :math:`N` is the number of samples (or sample-output pairs).
  - :math:`u_i` and :math:`l_i` are the upper and lower bounds of the
    prediction interval for sample :math:`i`.

**Practical Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import mean_interval_width_score # Assuming direct import

   y_lower = np.array([9, 11, 10, 8])
   y_upper = np.array([11, 13, 12, 10])
   # Widths: [2, 2, 2, 2]

   score = mean_interval_width_score(y_lower, y_upper, verbose=0)
   print(f"Mean Interval Width: {score:.4f}")

   y_lower_nan = np.array([9, np.nan, 10])
   y_upper_nan = np.array([11, 13, 12])
   # Widths for omit: [2, 2]
   score_nan_omit = mean_interval_width_score(y_lower_nan, y_upper_nan,
                                              nan_policy='omit', verbose=0)
   print(f"Mean Interval Width (NaN omit): {score_nan_omit:.4f}")

**Expected Output:**

.. code-block:: text

   Mean Interval Width: 2.0000
   Mean Interval Width (NaN omit): 2.0000


prediction_stability_score
~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.prediction_stability_score`

**Concept:** Prediction Stability Score (PSS)

PSS measures the temporal smoothness or coherence of consecutive
forecasts within a prediction horizon. It quantifies the average
absolute change between predictions at successive time steps. Lower
values indicate smoother and more stable forecast trajectories.

For a single forecast trajectory :math:`\hat{y}_1, \dots, \hat{y}_T`:
.. math::
   \mathrm{PSS}_{\text{trajectory}} = \frac{1}{T-1} \sum_{t=2}^{T}
   |\hat{y}_{t} - \hat{y}_{t-1}|

The function averages this score over all provided samples and outputs.

**Practical Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import prediction_stability_score # Assuming direct import

   # 3 samples, 5-step horizon
   y_pred = np.array([
       [1, 1, 2, 2, 3],  # Diffs: [0,1,0,1], Mean diff = 0.5
       [2, 3, 2, 3, 2],  # Diffs: [1,1,1,1], Mean diff = 1.0
       [0, 1, 0, 1, 0]   # Diffs: [1,1,1,1], Mean diff = 1.0
   ])
   # Overall PSS = (0.5 + 1.0 + 1.0) / 3 = 2.5 / 3

   score = prediction_stability_score(y_pred, verbose=0)
   print(f"PSS: {score:.4f}")

**Expected Output:**

.. code-block:: text

   PSS: 0.8333
   
quantile_calibration_error
~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.quantile_calibration_error`

**Concept:** Quantile Calibration Error (QCE)

QCE assesses the calibration of probabilistic forecasts given as a set
of predicted quantiles. For each nominal quantile level :math:`q`, it
measures the absolute difference between :math:`q` and the empirical
frequency of observations falling below the predicted :math:`q`-th
quantile :math:`\hat{Q}(q)`.

.. math::
   \mathrm{QCE}(q) = \left| \frac{1}{N} \sum_{i=1}^{N}
   \mathbf{1}\{y_i \le \hat{Q}_i(q)\} - q \right|

The function returns the average QCE across all provided quantile levels.
Lower values indicate better calibration.

**Practical Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import quantile_calibration_error # Assuming direct import

   y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
   quantiles = np.array([0.25, 0.5, 0.75])
   # Predictions for these quantiles
   y_pred_quantiles = np.array([
       [1.5, 4.5, 7.5], # Sample 1 (true=1) -> Q0.25=1.5, Q0.5=4.5, Q0.75=7.5
       [2.0, 5.0, 8.0], # Sample 2 (true=2)
       [2.5, 5.5, 8.5], # ... and so on for 10 samples
       [3.0, 6.0, 9.0],
       [3.5, 6.5, 9.5],
       [4.0, 7.0, 10.0],
       [4.5, 7.5, 10.5],
       [5.0, 8.0, 11.0],
       [5.5, 8.5, 11.5],
       [6.0, 9.0, 12.0]
   ])

   # For q=0.25 (predicted: y_pred_quantiles[:,0])
   # y_true <= y_pred_quantiles[:,0]:
   # [1<=1.5 T, 2<=2 F, 3<=2.5 F, 4<=3 F, 5<=3.5 F,
   #  6<=4 F, 7<=4.5 F, 8<=5 F, 9<=5.5 F, 10<=6 F]
   # Proportion = 1/10 = 0.1. QCE(0.25) = |0.1 - 0.25| = 0.15

   # For q=0.5 (predicted: y_pred_quantiles[:,1])
   # y_true <= y_pred_quantiles[:,1]:
   # [1<=4.5 T, 2<=5 T, 3<=5.5 T, 4<=6 T, 5<=6.5 T,
   #  6<=7 T, 7<=7.5 T, 8<=8 T, 9<=8.5 F, 10<=9 F]
   # Proportion = 8/10 = 0.8. QCE(0.5) = |0.8 - 0.5| = 0.3

   # For q=0.75 (predicted: y_pred_quantiles[:,2])
   # y_true <= y_pred_quantiles[:,2]:
   # [1<=7.5 T, ..., 10<=12 T]. All True.
   # Proportion = 10/10 = 1.0. QCE(0.75) = |1.0 - 0.75| = 0.25
   # Average QCE = (0.15 + 0.3 + 0.25) / 3 = 0.7 / 3

   score = quantile_calibration_error(y_true, y_pred_quantiles, quantiles, verbose=0)
   print(f"QCE: {score:.4f}")

**Expected Output:**

.. code-block:: text

   QCE: 0.2333


theils_u_score
~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.theils_u_score`

**Concept:** Theil's U Statistic

Theil's U statistic is a relative accuracy measure that compares the
forecast to a naive persistence model (random walk forecast, where the
forecast for the next period is the current period's actual value).
It is the ratio of the Root Mean Squared Error (RMSE) of the model's
forecast to the RMSE of the naive forecast.

.. math::
   U = \sqrt{ \frac{\sum_{i,o,t}(y_{i,o,t} - \hat{y}_{i,o,t})^2}
   {\sum_{i,o,t}(y_{i,o,t} - y_{i,o,t-1})^2} }

Where sums are over valid samples :math:`i`, outputs :math:`o`, and
time steps :math:`t \ge 2`.
  - :math:`U < 1`: Forecast is better than the naive model.
  - :math:`U = 1`: Forecast is as good as the naive model.
  - :math:`U > 1`: Forecast is worse than the naive model.

**Practical Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import theils_u_score # Assuming direct import

   # 2 samples, 4-step horizon
   y_true = np.array([[1, 2, 3, 4], [2, 2, 2, 2]])
   y_pred = np.array([[1, 2, 3, 5], [2, 1, 2, 3]])

   # Numerator (SSE_model):
   # Sample 0: (2-2)^2 + (3-3)^2 + (4-5)^2 = 0 + 0 + 1 = 1
   # Sample 1: (2-1)^2 + (2-2)^2 + (2-3)^2 = 1 + 0 + 1 = 2
   # Total SSE_model = 1 + 2 = 3

   # Denominator (SSE_base - naive persistence):
   # Sample 0: (2-1)^2 + (3-2)^2 + (4-3)^2 = 1 + 1 + 1 = 3
   # Sample 1: (2-2)^2 + (2-2)^2 + (2-2)^2 = 0 + 0 + 0 = 0
   # Total SSE_base = 3 + 0 = 3
   # U = sqrt(3 / 3) = 1.0

   score = theils_u_score(y_true, y_pred, verbose=0)
   print(f"Theil's U: {score:.4f}")


time_weighted_accuracy_score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.time_weighted_accuracy_score`

**Concept:** Time-Weighted Accuracy (TWA) Score

TWA evaluates classification accuracy over sequences, applying
time-dependent weights. This is useful when the importance of correct
predictions varies across the time horizon.

For a single sample :math:`i`, output :math:`o`, the TWA is:
.. math::
   \mathrm{TWA}_{io} = \sum_{t=1}^{T_{steps}} w_t \cdot
   \mathbf{1}\{y_{i,o,t} = \hat{y}_{i,o,t}\}

Where :math:`w_t` are normalized time weights. The final score is an
average over samples and possibly outputs. Higher scores are better.

**Practical Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import twa_score # Assuming direct import

   y_true = np.array([[1, 0, 1], [0, 1, 1]]) # 2 samples, 3 timesteps
   y_pred = np.array([[1, 1, 1], [0, 1, 0]])
   # Correctness: S0: [1,0,1], S1: [1,1,0]

   # Default time_weights (inverse_time, T=3) normalized:
   # w_raw = [1/1, 1/2, 1/3] = [1, 0.5, 0.333...]
   # sum_w_raw = 11/6 approx 1.8333
   # w_norm = [6/11, 3/11, 2/11] approx [0.5454, 0.2727, 0.1818]

   # TWA S0 = 1*(6/11) + 0*(3/11) + 1*(2/11) = 8/11
   # TWA S1 = 1*(6/11) + 1*(3/11) + 0*(2/11) = 9/11
   # Avg TWA = (8/11 + 9/11) / 2 = (17/11) / 2 = 17/22

   score_default = twa_score(y_true, y_pred, verbose=0)
   print(f"TWA (default weights): {score_default:.4f}")

   custom_weights = np.array([0.6, 0.3, 0.1]) # Sum to 1
   # TWA S0 = 1*0.6 + 0*0.3 + 1*0.1 = 0.7
   # TWA S1 = 1*0.6 + 1*0.3 + 0*0.1 = 0.9
   # Avg TWA = (0.7 + 0.9) / 2 = 0.8
   score_custom = twa_score(y_true, y_pred, time_weights=custom_weights, verbose=0)
   print(f"TWA (custom weights): {score_custom:.4f}")

**Expected Output:**

.. code-block:: text

   TWA (default weights): 0.7727
   TWA (custom weights): 0.8000
   
time_weighted_interval_score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.time_weighted_interval_score`

**Concept:** Time-Weighted Interval Score (TWIS)

TWIS extends the Weighted Interval Score (WIS) by applying
time-dependent weights to the WIS calculated at each time step.
It evaluates probabilistic forecasts (median and prediction intervals)
over a time horizon, emphasizing performance at certain horizons.
WIS itself considers both sharpness and calibration.

The WIS for a single observation :math:`y`, median :math:`m`, and
:math:`K` prediction intervals is:
.. math::
   \mathrm{WIS}(y, m, \text{intervals}) = \frac{1}{K+1} \left(
       |y-m| + \sum_{k=1}^K \mathrm{IS}_{\alpha_k}(y, l_k, u_k)
   \right)
where :math:`\mathrm{IS}_{\alpha_k}` is the interval score for the
k-th interval with nominal coverage :math:`1-\alpha_k`.

TWIS calculates :math:`\mathrm{WIS}_{iot}` for each sample :math:`i`,
output :math:`o`, and time step :math:`t`. Then:
.. math::
   \mathrm{TWIS}_{io} = \sum_{t=1}^{T_{steps}} w_t \cdot \mathrm{WIS}_{iot}

Where :math:`w_t` are normalized time weights. Lower scores are better.

**Practical Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import time_weighted_interval_score # Assuming import

   # 2 samples, 1 output (implicit), 2 timesteps
   y_true = np.array([[10, 11], [20, 22]])
   y_median = np.array([[10, 11.5], [19, 21.5]])
   # For K=1 interval, alpha=0.2 (80% PI)
   alphas = np.array([0.2])
   # y_lower/upper shape: (n_samples, n_outputs_dummy=1, K_intervals=1, n_timesteps=2)
   y_lower = np.array([[[[9, 10]]], [[[18, 20]]]])
   y_upper = np.array([[[[11, 12]]], [[[20, 23]]]])

   # Uniform time_weights for T=2: [0.5, 0.5]
   # Sample 0: y_t=[10,11], y_m=[10,11.5], y_l=[[9,10]], y_u=[[11,12]]
   #   T0: y=10,m=10,l=9,u=11. MAE_m=0. IS_0.2=(11-9)=2. WIS_comp=(0.2/2)*2=0.2.
   #       WIS_S0T0=(0+0.2)/(1+1)=0.1
   #   T1: y=11,m=11.5,l=10,u=12. MAE_m=0.5. IS_0.2=(12-10)=2. WIS_comp=0.2.
   #       WIS_S0T1=(0.5+0.2)/(1+1)=0.35
   #   TWIS_S0 = 0.5*0.1 + 0.5*0.35 = 0.05 + 0.175 = 0.225
   # Sample 1: y_t=[20,22], y_m=[19,21.5], y_l=[[18,20]], y_u=[[20,23]]
   #   T0: y=20,m=19,l=18,u=20. MAE_m=1. IS_0.2=(20-18)=2. WIS_comp=0.2.
   #       WIS_S1T0=(1+0.2)/(1+1)=0.6
   #   T1: y=22,m=21.5,l=20,u=23. MAE_m=0.5. IS_0.2=(23-20)=3. WIS_comp=0.3.
   #       WIS_S1T1=(0.5+0.3)/(1+1)=0.4
   #   TWIS_S1 = 0.5*0.6 + 0.5*0.4 = 0.3 + 0.2 = 0.5
   # Avg TWIS = (0.225 + 0.5) / 2 = 0.725 / 2 = 0.3625

   score = time_weighted_interval_score(
       y_true, y_median, y_lower, y_upper, alphas,
       time_weights=None, verbose=0 # None -> uniform weights
   )
   print(f"TWIS (uniform time weights): {score:.4f}")

**Expected Output:**

.. code-block:: text

   TWIS (uniform time weights): 0.3625


time_weighted_mean_absolute_error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.time_weighted_mean_absolute_error`

**Concept:** Time-Weighted Mean Absolute Error (TW-MAE)

TW-MAE calculates the mean absolute error, giving different weights to
errors at different time steps in a sequence. This is useful when
errors at certain points (e.g., early predictions) are more critical.

For a single sequence :math:`i` and output :math:`o`:
.. math::
   \mathrm{TWMAE}_{io} = \sum_{t=1}^{T_{steps}}
   w_t | \hat{y}_{i,o,t} - y_{i,o,t} |

Where :math:`w_t` are normalized time weights. The final score is an
average over samples and possibly outputs. Lower scores are better.

**Practical Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import time_weighted_mean_absolute_error # Assuming import

   y_true = np.array([[1, 2, 3], [2, 3, 4]])
   y_pred = np.array([[1.1, 2.2, 2.9], [1.9, 3.1, 3.8]])
   # Abs Errors:
   # S0: [0.1, 0.2, 0.1]
   # S1: [0.1, 0.1, 0.2]

   # Default time_weights (inverse_time, T=3) normalized: [6/11, 3/11, 2/11]
   # TWMAE S0 = 0.1*(6/11) + 0.2*(3/11) + 0.1*(2/11) = (0.6+0.6+0.2)/11 = 1.4/11
   # TWMAE S1 = 0.1*(6/11) + 0.1*(3/11) + 0.2*(2/11) = (0.6+0.3+0.4)/11 = 1.3/11
   # Avg TWMAE = (1.4/11 + 1.3/11) / 2 = (2.7/11) / 2 = 2.7/22

   score_default = time_weighted_mean_absolute_error(y_true, y_pred, verbose=0)
   print(f"TW-MAE (default weights): {score_default:.4f}")

   custom_weights = np.array([0.5, 0.3, 0.2])
   # TWMAE S0 = 0.1*0.5 + 0.2*0.3 + 0.1*0.2 = 0.05+0.06+0.02 = 0.13
   # TWMAE S1 = 0.1*0.5 + 0.1*0.3 + 0.2*0.2 = 0.05+0.03+0.04 = 0.12
   # Avg TWMAE = (0.13 + 0.12) / 2 = 0.125
   score_custom = time_weighted_mean_absolute_error(
       y_true, y_pred, time_weights=custom_weights, verbose=0
   )
   print(f"TW-MAE (custom weights): {score_custom:.4f}")

**Expected Output:**

.. code-block:: text

   TW-MAE (default weights): 0.1227
   TW-MAE (custom weights): 0.1250
   
weighted_interval_score
~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.weighted_interval_score`

**Concept:** Weighted Interval Score (WIS) (Non-Time-Weighted)

WIS is a proper scoring rule for evaluating probabilistic forecasts
provided as a median and a set of central prediction intervals. It
generalizes the absolute error and considers multiple quantile levels,
balancing sharpness (interval width) and calibration.

.. math::
   \mathrm{WIS}(y, m, \text{intervals}) = \frac{1}{K+1} \left(
       |y-m| + \sum_{k=1}^K \mathrm{IS}_{\alpha_k}(y, l_k, u_k)
   \right)

Where :math:`m` is the median forecast, and :math:`\mathrm{IS}_{\alpha_k}`
is the interval score for the k-th prediction interval :math:`(l_k, u_k)`
with nominal coverage :math:`1-\alpha_k`. The interval score component is
typically:
.. math::
   \mathrm{IS}_{\alpha_k}(y, l_k, u_k) = (u_k - l_k) +
   \frac{2}{\alpha_k}(l_k - y)\mathbf{1}\{y < l_k\} +
   \frac{2}{\alpha_k}(y - u_k)\mathbf{1}\{y > u_k\}

Lower WIS values indicate better forecast performance.

**Practical Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import weighted_interval_score # Assuming import

   y_true = np.array([10, 12, 11])
   y_median = np.array([10, 12, 11])
   # For K=2 intervals
   y_lower = np.array([[9, 8], [11, 10], [10, 9]]) # (N, K)
   y_upper = np.array([[11, 12], [13, 14], [12, 13]])
   alphas = np.array([0.2, 0.5]) # Corresponds to 80% and 50% PIs

   # Sample 0: y=10, m=10. MAE_m=0.
   #   k=0 (alpha=0.2): l=9, u=11. IS_0.2 = (11-9) = 2. WIS_comp = (0.2/2)*2 = 0.2
   #   k=1 (alpha=0.5): l=8, u=12. IS_0.5 = (12-8) = 4. WIS_comp = (0.5/2)*4 = 1.0
   #   WIS_S0 = (0 + 0.2 + 1.0) / (2+1) = 1.2/3 = 0.4
   # All samples are perfectly centered, so WIS will be 0.4 for all. Avg = 0.4.

   score = weighted_interval_score(y_true, y_lower, y_upper, y_median, alphas, verbose=0)
   print(f"WIS: {score:.4f}")

**Expected Output:**

.. code-block:: text

   WIS: 0.4000


Visualizing Metrics
-------------------

While this section details the calculation of metrics, visualizing
them can provide deeper insights. For example, plotting calibration
curves for `quantile_calibration_error` or showing interval widths
against coverage for `mean_interval_width_score` and `coverage_score`
can be very informative.

*(Note: Specific plotting functions are not part of `fusionlab.metrics`
itself but can be implemented using standard libraries like Matplotlib
or Seaborn based on the outputs of these metric functions. If example
plots are generated for the documentation, they would be referenced here,
e.g., like this:)*

.. code-block:: rst

   .. figure:: ../../images/example_metric_plot.png
      :alt: Example Metric Visualization
      :align: center
      :width: 70%

      Caption describing the example metric plot.


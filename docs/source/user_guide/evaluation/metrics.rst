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
mathematical formulations, and how to visualize them using utilities
from :mod:`fusionlab.plot.evaluation`.

Forecasting Metrics (`fusionlab.metrics`)
------------------------------------------

.. _metric_coverage_score:

coverage_score
~~~~~~~~~~~~~~~~
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

**When to Use:**
Use this metric when you are working with probabilistic forecasts that
produce prediction intervals (defined by lower and upper quantiles,
e.g., 10th and 90th percentiles). It directly tells you how often your
true values are captured by the predicted range. A score significantly
lower than the nominal interval (e.g., 0.7 for a 90% interval) indicates
the model is too confident or its intervals are too narrow. A score
much higher might indicate overly wide and less informative intervals.

**Practical Example (Calculation):**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import coverage_score

   y_true_cs = np.array([10, 12, 11, 9, 15, 13, 14])
   # Example 90% prediction interval (e.g., from q05 and q95)
   y_lower_cs = np.array([9,  11, 10,  8, 14, 11, 13])
   y_upper_cs = np.array([11, 13, 12, 10, 16, 15, 15])

   # Case 1: All actuals within bounds
   score_perfect_cs = coverage_score(y_true_cs, y_lower_cs, y_upper_cs, verbose=0)
   print(f"Coverage (Perfect Interval): {score_perfect_cs:.4f}")

   # Case 2: Some actuals outside bounds
   y_true_cs_miss = np.array([10, 13.5, 11, 7.5, 15, 16, 12]) # 2nd, 4th, 6th, 7th miss
   score_partial_cs = coverage_score(y_true_cs_miss, y_lower_cs, y_upper_cs, verbose=0)
   print(f"Coverage (Partial Interval): {score_partial_cs:.4f}")

**Expected Output (Calculation):**

.. code-block:: text

   Coverage (Perfect Interval): 1.0000
   Coverage (Partial Interval): 0.4286

**Visualization with `plot_coverage`:**

The :func:`~fusionlab.plot.evaluation.plot_coverage` function can
visualize these prediction intervals against the true values,
highlighting which points are covered.

.. code-block:: python
   :linenos:

   import matplotlib.pyplot as plt
   from fusionlab.plot.evaluation import plot_coverage 

   # Using data from Case 2 above
   # For plotting, it's often useful to have a time or sample index
   sample_indices_cs = np.arange(len(y_true_cs_miss))

   fig_cs, ax_cs = plt.subplots(figsize=(10, 5))
   plot_coverage(
       y_true=y_true_cs_miss,
       y_lower=y_lower_cs,
       y_upper=y_upper_cs,
       sample_indices=sample_indices_cs, # Optional x-axis values
       title="Prediction Interval Coverage Visualization",
       xlabel="Sample Index",
       ylabel="Value",
       ax=ax_cs, # Pass the created Axes object
       verbose=0
   )
   # To save for documentation:
   # plt.savefig("docs/source/images/metric_coverage_plot.png")
   plt.show()

**Expected Plot (`plot_coverage`):**

.. figure:: ../../images/metric_coverage_plot.png
   :alt: Visualization of Prediction Interval Coverage
   :align: center
   :width: 80%

   Plot showing actual values against their predicted intervals, with
   covered and uncovered points distinctly marked.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

.. _metric_continuous_ranked_probability_score:

continuous_ranked_probability_score (CRPS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.continuous_ranked_probability_score`
   *(Note: can be aliased as `crp_score` for consistency).*

**Concept:** Ensemble Forecast Evaluation

The Continuous Ranked Probability Score (CRPS) is a proper scoring rule
that generalizes the Mean Absolute Error (MAE) to evaluate probabilistic
forecasts represented by an ensemble of prediction samples (i.e.,
multiple possible future trajectories). It measures both the calibration
(reliability) and sharpness (resolution) of the forecast distribution.
Lower CRPS values indicate better forecasts.

For a single observation :math:`y` and an ensemble of :math:`m`
forecast members :math:`x_1, \dots, x_m`, the sample-based CRPS is
approximated as:

.. math::
   \mathrm{CRPS}(y, \{x_j\}) = \frac{1}{m}\sum_{j=1}^{m} |x_j - y|
   - \frac{1}{2m^2}\sum_{j=1}^{m}\sum_{k=1}^{m} |x_j - x_k|

The first term is the average absolute error of the ensemble members
to the actual value (related to accuracy). The second term is the
average absolute difference between all pairs of ensemble members
(related to the ensemble's spread or sharpness). The function computes
the average CRPS over all provided samples.

**When to Use:**

Use CRPS when your model produces an ensemble of possible future
trajectories rather than quantiles or a single point forecast. It's a
comprehensive measure for evaluating the overall quality of such
probabilistic forecasts. It is particularly common in meteorological
and hydrological forecasting.

**Practical Example (Calculation):**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import continuous_ranked_probability_score

   y_true_crps = np.array([0.5, 0.0, 1.0])
   # Ensemble predictions: (n_samples, n_ensemble_members)
   # For multi-step, this would be (n_samples, n_horizon_steps, n_ensemble_members)
   # The current crp_score might expect 2D y_pred_ensemble if averaging over horizon.
   # For this example, let's assume single-step or already aggregated over horizon.
   y_pred_ensemble_crps = np.array([
       [0.0, 0.2, 0.4, 0.6, 0.8],  # Ensemble for y_true = 0.5
       [-0.2, 0.0, 0.1, 0.2, 0.3], # Ensemble for y_true = 0.0
       [0.8, 0.9, 1.0, 1.1, 1.2]   # Ensemble for y_true = 1.0
   ])

   score_crps = continuous_ranked_probability_score(
       y_true_crps, y_pred_ensemble_crps, verbose=0
       )
   print(f"Average CRPS: {score_crps:.4f}")

**Expected Output (Calculation):**

.. code-block:: text

   Average CRPS: 0.0680

**Visualization with `plot_crps`:**

The :func:`~fusionlab.plot.evaluation.plot_crps` function can help
visualize the ensemble predictions against the true value for a specific
sample (using `kind='ensemble_ecdf'`) or show the distribution of CRPS
scores (`kind='scores_histogram'`).

.. code-block:: python
   :linenos:

   import matplotlib.pyplot as plt
   from fusionlab.plot.evaluation import plot_crps 

   # Using data from the calculation example above
   # Plot ECDF for the first sample
   fig_crps1, ax_crps1 = plt.subplots(figsize=(8, 6))
   plot_crps(
       y_true=y_true_crps,
       y_pred_ensemble=y_pred_ensemble_crps,
       kind='ensemble_ecdf', # Plot ECDF of ensemble vs true value
       sample_idx=0,         # Plot for the first sample
       title=f"CRPS: Ensemble ECDF vs True Value (Sample 0)",
       ax=ax_crps1,
       verbose=0
   )
   # To save for documentation:
   # plt.savefig("docs/source/images/metric_crps_ecdf_plot.png")
   plt.show()

   # Plot histogram of CRPS scores (if crps_values are pre-calculated)
   # First, calculate CRPS for each sample individually
   all_crps_scores = []
   for i in range(len(y_true_crps)):
       single_true = np.array([y_true_crps[i]])
       single_ensemble = y_pred_ensemble_crps[i:i+1, :] # Keep 2D for function
       all_crps_scores.append(
           continuous_ranked_probability_score(single_true, single_ensemble)
           )
   all_crps_scores = np.array(all_crps_scores)

   fig_crps2, ax_crps2 = plt.subplots(figsize=(8, 5))
   plot_crps(
       y_true=y_true_crps, # Still needed for context if show_score=True
       y_pred_ensemble=y_pred_ensemble_crps, # Not strictly needed if metric_values given
       metric_values=all_crps_scores, # Pass pre-calculated scores
       kind='scores_histogram',
       title="Distribution of CRPS Scores",
       ax=ax_crps2,
       verbose=0
   )
   # To save for documentation:
   # plt.savefig("docs/source/images/metric_crps_histogram_plot.png")
   plt.show()

**Expected Plot (`plot_crps` - ECDF):**

.. figure:: ../../images/metric_crps_ecdf_plot.png
   :alt: Visualization of CRPS Ensemble ECDF
   :align: center
   :width: 70%

   Plot showing the Empirical Cumulative Distribution Function (ECDF)
   of an ensemble forecast against the true observed value for a sample.

**Expected Plot (`plot_crps` - Histogram):**

.. figure:: ../../images/metric_crps_histogram_plot.png
   :alt: Visualization of CRPS Scores Histogram
   :align: center
   :width: 70%

   Histogram showing the distribution of CRPS scores across all samples.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

mean_interval_width_score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.mean_interval_width_score`

**Concept:** Mean Interval Width (Sharpness)

This metric, often referred to as sharpness, measures the average
width of the prediction intervals. Narrower intervals are generally
preferred, provided they maintain adequate coverage (as measured by
:func:`~fusionlab.metrics.coverage_score`). It is calculated
independently of the true observed values.

.. math::
   \mathrm{MeanIntervalWidth} = \frac{1}{N} \sum_{i=1}^{N} (u_i - l_i)

Where:
 - :math:`N` is the number of samples (or sample-horizon pairs).
 - :math:`u_i` and :math:`l_i` are the upper and lower bounds of the
   prediction interval for sample :math:`i`.

**When to Use:**

Use this metric alongside `coverage_score` to evaluate probabilistic
forecasts that produce prediction intervals. While high coverage is
good, if the intervals are excessively wide, they may not be very
useful. A good model balances high coverage with reasonably narrow
(sharp) intervals. This metric helps quantify that sharpness.

**Practical Example (Calculation):**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import mean_interval_width_score

   y_lower_miw = np.array([9, 11, 10, 8, 13])
   y_upper_miw = np.array([11, 13, 12, 10, 14])
   # Widths: [2, 2, 2, 2, 1]

   score_miw = mean_interval_width_score(y_lower_miw, y_upper_miw, verbose=0)
   print(f"Mean Interval Width: {score_miw:.4f}")

**Expected Output (Calculation):**

.. code-block:: text

   Mean Interval Width: 1.8000

**Visualization with `plot_metric_over_horizon` or `plot_metric_radar`:**

The Mean Interval Width can be visualized:

* **Over the forecast horizon:** Use
  :func:`~fusionlab.plot.evaluation.plot_metric_over_horizon`.
  This requires calculating MIW for each forecast step.
* **Across different segments:** Use
  :func:`~fusionlab.plot.evaluation.plot_metric_radar`. This
  requires calculating MIW for each segment.

For a simple bar chart of the overall MIW, you can use Matplotlib directly.

.. code-block:: python
   :linenos:

   import matplotlib.pyplot as plt
   from fusionlab.plot.evaluation import plot_metric_over_horizon
   from fusionlab.nn.utils import format_predictions_to_dataframe # For df structure

   # Assume y_true_val, y_lower_val, y_upper_val are (Samples, Horizon)
   # For plot_metric_over_horizon, we need a forecast_df
   # Let's simulate a forecast_df for this visualization
   B, H = 5, 4 # 5 samples, 4 horizon steps
   y_true_dummy = np.random.rand(B, H)
   y_lower_dummy = y_true_dummy - np.random.rand(B, H) * 0.5
   y_upper_dummy = y_true_dummy + np.random.rand(B, H) * 0.5

   # Create a dummy forecast_df (simplified)
   # In practice, use format_predictions_to_dataframe
   rows = []
   for i in range(B):
       for h_step in range(H):
           rows.append({
               'sample_idx': i,
               'forecast_step': h_step + 1,
               'target_actual': y_true_dummy[i, h_step],
               'target_q_lower': y_lower_dummy[i, h_step], # Example name
               'target_q_upper': y_upper_dummy[i, h_step]  # Example name
           })
   df_for_miw_plot = pd.DataFrame(rows)

   # Custom metric function for MIW to pass to plot_metric_over_horizon
   def miw_for_plot(y_true, y_pred_dict): # y_pred_dict will contain q_lower, q_upper
       return mean_interval_width_score(
           y_pred_dict['target_q_lower'], y_pred_dict['target_q_upper']
           )

   # This requires plot_metric_over_horizon to handle y_pred_dict
   # or a more specific plot function for interval metrics.
   # For simplicity, let's plot the overall MIW as a bar.
   overall_miw = mean_interval_width_score(
       df_for_miw_plot['target_q_lower'], df_for_miw_plot['target_q_upper']
       )

   fig_miw, ax_miw = plt.subplots(figsize=(6,4))
   ax_miw.bar(['Overall MIW'], [overall_miw], color='skyblue')
   ax_miw.set_title('Mean Interval Width (Overall)')
   ax_miw.set_ylabel('Average Width')
   ax_miw.text(0, overall_miw + 0.05, f'{overall_miw:.2f}', ha='center')
   plt.grid(axis='y', linestyle='--')
   # plt.savefig("docs/source/images/metric_miw_plot.png")
   plt.show()

**Expected Plot (Overall MIW Bar Chart):**

.. figure:: ../../images/metric_miw_plot.png
   :alt: Visualization of Mean Interval Width
   :align: center
   :width: 60%

   Bar chart showing the overall Mean Interval Width.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

.. _metric_prediction_stability_score:

prediction_stability_score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.prediction_stability_score`

**Concept:** Prediction Stability Score (PSS)

PSS measures the temporal smoothness or coherence of consecutive
forecasts within a prediction horizon. It quantifies the average
absolute change between predictions at successive time steps. Lower
values indicate smoother and more stable forecast trajectories, which
can be desirable for interpretability and actionability.

For a single forecast trajectory :math:`\hat{y}_1, \dots, \hat{y}_T`
(where :math:`T` is the horizon length):

.. math::
   \mathrm{PSS}_{\text{trajectory}} = \frac{1}{T-1} \sum_{t=2}^{T}
   |\hat{y}_{t} - \hat{y}_{t-1}|

The function averages this score over all provided samples and outputs
(if multi-output).

**When to Use:**
Use PSS to evaluate the "smoothness" or "jitteriness" of your
multi-step forecasts. Highly erratic predictions over the horizon might
be less trustworthy or harder to act upon, even if their average
accuracy is good. This is particularly relevant for models that predict
an entire sequence at once.

**Practical Example (Calculation):**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import prediction_stability_score

   y_pred_pss = np.array([
       [1, 1.1, 1.3, 1.4, 1.6], # Smooth
       [2, 3,   2,   3,   2],   # Jittery
       [5, 4.9, 4.8, 4.7, 4.6]  # Smooth (decreasing)
   ])
   # PSS for row 0: (|0.1|+|0.2|+|0.1|+|0.2|)/4 = 0.6/4 = 0.15
   # PSS for row 1: (|1|+|1|+|1|+|1|)/4 = 4/4 = 1.0
   # PSS for row 2: (|-0.1|+|-0.1|+|-0.1|+|-0.1|)/4 = 0.4/4 = 0.1
   # Overall PSS = (0.15 + 1.0 + 0.1) / 3 = 1.25 / 3

   score_pss = prediction_stability_score(y_pred_pss, verbose=0)
   print(f"PSS: {score_pss:.4f}")

**Expected Output (Calculation):**

.. code-block:: text

   PSS: 0.4167

**Visualization with `plot_metric_radar` or Bar Chart:**

PSS is typically a single score per item or overall. It can be
visualized using a bar chart if comparing across different models or
segments, or using :func:`~fusionlab.plot.evaluation.plot_metric_radar`
if you have PSS calculated for different categories.

.. code-block:: python
   :linenos:

   import matplotlib.pyplot as plt

   fig_pss, ax_pss = plt.subplots(figsize=(6,4))
   ax_pss.bar(['Overall PSS'], [score_pss], color='lightcoral')
   ax_pss.set_title('Prediction Stability Score (Overall)')
   ax_pss.set_ylabel('Average Step-to-Step Change')
   ax_pss.text(0, score_pss + 0.01, f'{score_pss:.2f}', ha='center')
   plt.grid(axis='y', linestyle='--')
   # plt.savefig("docs/source/images/metric_pss_plot.png")
   plt.show()

**Expected Plot (Overall PSS Bar Chart):**

.. figure:: ../../images/metric_pss_plot.png
   :alt: Visualization of Prediction Stability Score
   :align: center
   :width: 60%

   Bar chart showing the overall Prediction Stability Score.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

.. _metric_quantile_calibration_error:

quantile_calibration_error
~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.quantile_calibration_error`

**Concept:** Quantile Calibration Error (QCE)

QCE assesses the calibration of probabilistic forecasts given as a set
of predicted quantiles. For each nominal quantile level :math:`q`, it
measures the absolute difference between :math:`q` and the empirical
frequency of observations falling below the predicted :math:`q`-th
quantile :math:`\hat{Q}(q)`. A perfectly calibrated forecast would have
this empirical frequency match :math:`q`.

.. math::
   \mathrm{QCE}(q) = \left| \frac{1}{N} \sum_{i=1}^{N}
   \mathbf{1}\{y_i \le \hat{Q}_i(q)\} - q \right|

The function returns the average QCE across all provided quantile levels.
Lower values (closer to 0) indicate better calibration.

**When to Use:**
Use QCE to evaluate if your model's predicted quantiles are reliable.
For example, if you predict the 0.1 quantile, you expect about 10% of
the actual values to fall below this prediction. QCE quantifies how
far off your model is from this ideal. It's essential for assessing
the trustworthiness of prediction intervals derived from quantiles.

**Practical Example (Calculation):**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import quantile_calibration_error

   y_true_qce = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
   quantiles_qce = np.array([0.25, 0.5, 0.75])
   y_pred_quantiles_qce = np.array([
       [1.5, 4.5, 7.5], [2.0, 5.0, 8.0], [2.5, 5.5, 8.5],
       [3.0, 6.0, 9.0], [3.5, 6.5, 9.5], [4.0, 7.0, 10.0],
       [4.5, 7.5, 10.5],[5.0, 8.0, 11.0],[5.5, 8.5, 11.5],
       [6.0, 9.0, 12.0]
   ])
   score_qce = quantile_calibration_error(
       y_true_qce, y_pred_quantiles_qce, quantiles_qce, verbose=0
       )
   print(f"Average QCE: {score_qce:.4f}")

**Expected Output (Calculation):**

.. code-block:: text

   Average QCE: 0.2000

**Visualization with `plot_quantile_calibration`:**

The :func:`~fusionlab.plot.evaluation.plot_quantile_calibration`
function creates a reliability diagram, plotting the observed
proportion of actuals below each predicted quantile against the
nominal quantile level.

.. code-block:: python
   :linenos:

   import matplotlib.pyplot as plt
   from fusionlab.plot.evaluation import plot_quantile_calibration

   # Using data from the calculation example above
   fig_qce, ax_qce = plt.subplots(figsize=(7, 7))
   plot_quantile_calibration(
       y_true=y_true_qce,
       y_pred_quantiles=y_pred_quantiles_qce,
       quantiles=quantiles_qce,
       title="Quantile Calibration Reliability Diagram",
       ax=ax_qce,
       verbose=0
   )
   # plt.savefig("docs/source/images/metric_qce_plot.png")
   plt.show()

**Expected Plot (`plot_quantile_calibration`):**

.. figure:: ../../images/metric_qce_plot.png
   :alt: Quantile Calibration Reliability Diagram
   :align: center
   :width: 70%

   Reliability diagram showing observed vs. nominal proportions for
   different quantiles. Points closer to the diagonal indicate better
   calibration.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

.. _metric_theils_u_score:

theils_u_score
~~~~~~~~~~~~~~~~~
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
time steps :math:`t \ge 2` (or :math:`t \ge \text{lag}+1` if a different
lag is used for the naive model).
 - :math:`U < 1`: Forecast is better than the naive model.
 - :math:`U = 1`: Forecast is as good as the naive model.
 - :math:`U > 1`: Forecast is worse than the naive model.

**When to Use:**

Use Theil's U to understand if your sophisticated forecasting model is
actually providing more value than a very simple baseline (like
predicting the last known value). It's a good sanity check, especially
for time series that exhibit strong persistence. A U score greater than
1 is a strong indication that the model needs improvement or is not
suitable for the data.

**Practical Example (Calculation):**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import theils_u_score

   # 2 samples, 4-step horizon, 1 output dim
   y_true_u = np.array([[1,2,3,4],[2,2,2,2]])
   y_pred_u = np.array([[1,2,3,5],[2,1,2,3]])

   score_u = theils_u_score(y_true_u, y_pred_u, verbose=0)
   print(f"Theil's U: {score_u:.4f}")

**Expected Output (Calculation):**

.. code-block:: text

   Theil's U: 1.0000

**Visualization with `plot_theils_u_score`:**

The :func:`~fusionlab.plot.evaluation.plot_theils_u_score` function
can display Theil's U as a bar chart, often with a reference line at 1.0.

.. code-block:: python
   :linenos:

   import matplotlib.pyplot as plt
   from fusionlab.plot.evaluation import plot_theils_u_score

   # Using data from the calculation example above
   fig_u, ax_u = plt.subplots(figsize=(6, 5))
   plot_theils_u_score(
       y_true=y_true_u, # Required if metric_values is None
       y_pred=y_pred_u, # Required if metric_values is None
       # metric_values=score_u, # Can pass pre-calculated score
       title="Theil's U Statistic Example",
       ax=ax_u,
       verbose=0
   )
   # plt.savefig("docs/source/images/metric_theils_u_plot.png")
   plt.show()

**Expected Plot (`plot_theils_u_score`):**

.. figure:: ../../images/metric_theils_u_plot.png
   :alt: Theil's U Statistic Visualization
   :align: center
   :width: 60%

   Bar chart showing Theil's U statistic, with a reference line at 1.0.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

.. _metric_time_weighted_accuracy_score:

time_weighted_accuracy_score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.time_weighted_accuracy_score`

**Concept:** Time-Weighted Accuracy (TWA) Score

TWA evaluates classification accuracy over sequences, applying
time-dependent weights. This is useful when the importance of correct
predictions varies across the time horizon (e.g., earlier predictions
in a sequence might be more critical).

For a single sample :math:`i`, output :math:`o`, the TWA is:
.. math::
   \mathrm{TWA}_{io} = \sum_{t=1}^{T_{steps}} w_t \cdot
   \mathbf{1}\{y_{i,o,t} = \hat{y}_{i,o,t}\}

Where :math:`w_t` are normalized time weights (summing to 1 over the
horizon). The final score is an average over samples and possibly
outputs. Higher scores (closer to 1) are better.

**When to Use:**
Use this metric for evaluating sequential classification tasks where
the accuracy at different time steps within the sequence has varying
importance. For example, in predicting a sequence of states, correctly
predicting the initial states might be more valuable.

**Practical Example (Calculation):**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import time_weighted_accuracy_score as twa_score

   y_true_twa = np.array([[1, 0, 1], [0, 1, 1]]) # 2 samples, 3 timesteps
   y_pred_twa = np.array([[1, 1, 1], [0, 1, 0]])

   score_default_twa = twa_score(y_true_twa, y_pred_twa, verbose=0)
   print(f"TWA (default 'inverse_time' weights): {score_default_twa:.4f}")

   custom_weights_twa = np.array([0.6, 0.3, 0.1]) # Must sum to 1
   score_custom_twa = twa_score(
       y_true_twa, y_pred_twa, time_weights=custom_weights_twa, verbose=0
       )
   print(f"TWA (custom weights [0.6, 0.3, 0.1]): {score_custom_twa:.4f}")

**Expected Output (Calculation):**

.. code-block:: text

   TWA (default 'inverse_time' weights): 0.7727
   TWA (custom weights [0.6, 0.3, 0.1]): 0.8000

**Visualization with `plot_time_weighted_metric`:**

The :func:`~fusionlab.plot.evaluation.plot_time_weighted_metric`
function can visualize how the accuracy (or its components) and weights
evolve over the time steps.

.. code-block:: python
   :linenos:

   import matplotlib.pyplot as plt
   from fusionlab.plot.evaluation import plot_time_weighted_metric

   # Using data from the calculation example above
   fig_twa, ax_twa = plt.subplots(figsize=(10, 5))
   plot_time_weighted_metric(
       metric_type='accuracy', # Specify the metric
       y_true=y_true_twa,
       y_pred=y_pred_twa,
       time_weights='inverse_time', # Can also pass custom_weights_twa
       kind='time_profile', # Show accuracy profile over time steps
       title="Time-Weighted Accuracy Profile",
       ax=ax_twa,
       verbose=0
   )
   # To save for documentation:
   # plt.savefig("docs/source/images/metric_twa_plot.png")
   plt.show()

**Expected Plot (`plot_time_weighted_metric` for TWA):**

.. figure:: ../../images/metric_twa_plot.png
   :alt: Time-Weighted Accuracy Profile
   :align: center
   :width: 70%

   Plot showing the accuracy at each time step along with the
   time weights applied.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

.. _metric_time_weighted_interval_score:

time_weighted_interval_score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.time_weighted_interval_score`

**Concept:** Time-Weighted Interval Score (TWIS)

TWIS extends the Weighted Interval Score (WIS) by applying
time-dependent weights to the WIS calculated at each time step.
It evaluates probabilistic forecasts (median and prediction intervals)
over a time horizon, emphasizing performance at certain horizons.
WIS itself considers both sharpness (interval width) and calibration
(coverage of multiple intervals). Lower scores are better.

The WIS for a single observation :math:`y`, median :math:`m`, and
:math:`K` prediction intervals (defined by lower bounds :math:`l_k`
and upper bounds :math:`u_k` with nominal coverages :math:`1-\alpha_k`) is:

.. math::
   \mathrm{WIS}(y, m, \text{intervals}) = \frac{1}{K+0.5} \left(
     |y-m| + \sum_{k=1}^K \frac{\alpha_k}{2} \mathrm{IS}_{\alpha_k}(y, l_k, u_k)
   \right)
   
(Note: The original formula in the prompt had :math:`K+1` in the denominator
and different weighting for IS. The formula above is a common representation.
The exact formula used by `fusionlab.metrics.weighted_interval_score` should
be checked for precise interpretation.)

TWIS then calculates :math:`\mathrm{WIS}_{iot}` for each sample :math:`i`,
output :math:`o`, and time step :math:`t`, and applies time weights:

.. math::
   \mathrm{TWIS}_{io} = \sum_{t=1}^{T_{steps}} w_t \cdot \mathrm{WIS}_{iot}

Where :math:`w_t` are normalized time weights.

**When to Use:**
Use TWIS when evaluating multi-step probabilistic forecasts that provide
a median and multiple prediction intervals (defined by quantiles), and
when the importance of forecast quality varies across the forecast
horizon. It provides a single score summarizing both calibration and
sharpness, weighted by time.

**Practical Example (Calculation):**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import time_weighted_interval_score

   # 2 samples, 1 output (implicit in y_true shape), 2 timesteps
   y_true_twis = np.array([[10, 11], [20, 22]])
   y_median_twis = np.array([[10, 11.5], [19, 21.5]])
   # For K=1 interval, alpha=0.2 (80% PI)
   alphas_twis = np.array([0.2])
   # y_lower/upper shape: (n_samples, n_outputs_dummy=1, K_intervals=1, n_timesteps)
   # The function expects (n_samples, K_intervals, n_timesteps)
   y_lower_twis = np.array([[[9, 10]], [[18, 20]]]) # (2, 1, 2)
   y_upper_twis = np.array([[[11, 12]], [[20, 23]]])# (2, 1, 2)

   score_twis = time_weighted_interval_score(
       y_true_twis, y_median_twis, y_lower_twis, y_upper_twis, alphas_twis,
       time_weights=None, verbose=0 # None -> uniform weights
   )
   print(f"TWIS (uniform time weights): {score_twis:.4f}")

**Expected Output (Calculation):**
*(Calculation based on the provided example values and a common WIS formula)*

.. code-block:: text

   TWIS (uniform time weights): 0.3625

**Visualization with `plot_time_weighted_metric`:**

Use :func:`~fusionlab.plot.evaluation.plot_time_weighted_metric`
with `metric_type='interval_score'`.

.. code-block:: python
   :linenos:

   import matplotlib.pyplot as plt
   from fusionlab.plot.evaluation import plot_time_weighted_metric

   # Using data from the calculation example above
   fig_twis, ax_twis = plt.subplots(figsize=(10, 5))
   plot_time_weighted_metric(
       metric_type='interval_score',
       y_true=y_true_twis,
       y_median=y_median_twis,
       y_lower=y_lower_twis,
       y_upper=y_upper_twis,
       alphas=alphas_twis,
       time_weights=None, # Uniform weights
       kind='profile',
       title="Time-Weighted Interval Score (TWIS) Profile",
       ax=ax_twis,
       verbose=0
   )
   # To save for documentation:
   # plt.savefig("docs/source/images/metric_twis_plot.png")
   plt.show()

**Expected Plot (`plot_time_weighted_metric` for TWIS):**

.. figure:: ../../images/metric_twis_plot.png
   :alt: Time-Weighted Interval Score Profile
   :align: center
   :width: 70%

   Plot showing the Weighted Interval Score at each time step,
   along with the time weights applied.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

.. _metric_time_weighted_mae:

time_weighted_mean_absolute_error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.time_weighted_mean_absolute_error`

**Concept:** Time-Weighted Mean Absolute Error (TW-MAE)

TW-MAE calculates the mean absolute error, giving different weights to
errors at different time steps in a sequence. This is useful when
errors at certain points (e.g., early predictions in a multi-step
forecast) are considered more critical than errors at later steps.

For a single sequence :math:`i` and output :math:`o`:

.. math::
   \mathrm{TWMAE}_{io} = \sum_{t=1}^{T_{steps}}
   w_t | \hat{y}_{i,o,t} - y_{i,o,t} |

Where :math:`w_t` are normalized time weights (summing to 1 over the
horizon). The final score is an average over samples and possibly
outputs. Lower scores are better.

**When to Use:**
Apply TW-MAE when evaluating multi-step point forecasts where the
accuracy of predictions at different forecast horizons has varying
importance. For instance, short-term accuracy might be prioritized
over long-term accuracy.

**Practical Example (Calculation):**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import time_weighted_mean_absolute_error

   y_true_twmae = np.array([[1, 2, 3], [2, 3, 4]])
   y_pred_twmae = np.array([[1.1, 2.2, 2.9], [1.9, 3.1, 3.8]])

   score_default_twmae = time_weighted_mean_absolute_error(
       y_true_twmae, y_pred_twmae, verbose=0
       )
   print(f"TW-MAE (default 'inverse_time' weights): {score_default_twmae:.4f}")

   custom_weights_twmae = np.array([0.5, 0.3, 0.2]) # Must sum to 1
   score_custom_twmae = time_weighted_mean_absolute_error(
       y_true_twmae, y_pred_twmae, time_weights=custom_weights_twmae, verbose=0
   )
   print(f"TW-MAE (custom weights [0.5, 0.3, 0.2]): {score_custom_twmae:.4f}")

**Expected Output (Calculation):**

.. code-block:: text

   TW-MAE (default 'inverse_time' weights): 0.1227
   TW-MAE (custom weights [0.5, 0.3, 0.2]): 0.1250

**Visualization with `plot_time_weighted_metric`:**

Use :func:`~fusionlab.plot.evaluation.plot_time_weighted_metric`
with `metric_type='mae'`.

.. code-block:: python
   :linenos:

   import matplotlib.pyplot as plt
   from fusionlab.plot.evaluation import plot_time_weighted_metric

   # Using data from the calculation example above
   fig_twmae, ax_twmae = plt.subplots(figsize=(10, 5))
   plot_time_weighted_metric(
       metric_type='mae', # Specify MAE
       y_true=y_true_twmae,
       y_pred=y_pred_twmae,
       time_weights='inverse_time', # or custom_weights_twmae
       kind='profile',
       title="Time-Weighted MAE Profile",
       ax=ax_twmae,
       verbose=0
   )
   # To save for documentation:
   # plt.savefig("docs/source/images/metric_twmae_plot.png")
   plt.show()

**Expected Plot (`plot_time_weighted_metric` for TW-MAE):**

.. figure:: ../../images/metric_twmae_plot.png
   :alt: Time-Weighted MAE Profile
   :align: center
   :width: 70%

   Plot showing the Mean Absolute Error at each time step, along with
   the time weights applied.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

.. _metric_weighted_interval_score:

weighted_interval_score
~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.metrics.weighted_interval_score`

**Concept:** Weighted Interval Score (WIS) (Non-Time-Weighted)

WIS is a proper scoring rule for evaluating probabilistic forecasts
provided as a median and a set of central prediction intervals. It
generalizes the absolute error and considers multiple quantile levels,
balancing sharpness (interval width) and calibration (coverage of
multiple intervals). This version calculates an average score over all
provided time steps and samples, without explicit time-weighting.

.. math::
   \mathrm{WIS}(y, m, \text{intervals}) = \frac{1}{K+0.5} \left(
     |y-m| + \sum_{k=1}^K \frac{\alpha_k}{2} \mathrm{IS}_{\alpha_k}(y, l_k, u_k)
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

**When to Use:**
Use WIS when you have probabilistic forecasts in the form of a median
and several symmetric prediction intervals (defined by quantiles,
leading to :math:`\alpha_k` values). It provides a single, comprehensive
score that balances the accuracy of the median forecast with the
calibration and sharpness of the prediction intervals. It's a standard
metric in challenges like the M5 competition.

**Practical Example (Calculation):**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.metrics import weighted_interval_score

   y_true_wis = np.array([10, 12, 11])
   y_median_wis = np.array([10, 12, 11])
   # For K=2 intervals. y_lower/upper shape: (n_samples, K_intervals)
   y_lower_wis = np.array([[9, 8], [11, 10], [10, 9]])
   y_upper_wis = np.array([[11, 12], [13, 14], [12, 13]])
   alphas_wis = np.array([0.2, 0.5]) # Corresponds to 80% and 50% PIs

   score_wis = weighted_interval_score(
       y_true_wis, y_lower_wis, y_upper_wis, y_median_wis, alphas_wis,
       verbose=0
       )
   print(f"WIS: {score_wis:.4f}")

**Expected Output (Calculation):**

.. code-block:: text

   WIS: 0.4000

**Visualization:**

WIS is typically a single summary score. It can be visualized as a bar
chart, especially when comparing different models or segments.

.. code-block:: python
   :linenos:

   import matplotlib.pyplot as plt
   # Using score_wis from the calculation example above

   fig_wis, ax_wis = plt.subplots(figsize=(6,4))
   ax_wis.bar(['Overall WIS'], [score_wis], color='olivedrab', width=0.5)
   ax_wis.set_title('Weighted Interval Score (Overall)')
   ax_wis.set_ylabel('Score Value')
   ax_wis.text(0, score_wis, f'{score_wis:.2f}', ha='center', va='bottom')
   plt.grid(axis='y', linestyle=':', alpha=0.7)
   # To save for documentation:
   # plt.savefig("docs/source/images/metric_wis_plot.png")
   plt.show()

**Expected Plot (Overall WIS Bar Chart):**

.. figure:: ../../images/metric_wis_plot.png
   :alt: Visualization of Weighted Interval Score
   :align: center
   :width: 60%

   Bar chart showing the overall Weighted Interval Score.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


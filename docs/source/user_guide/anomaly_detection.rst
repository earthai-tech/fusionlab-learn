.. _user_guide_anomaly_detection:

===================
Anomaly Detection
===================

Anomaly detection involves identifying data points, events, or
observations that deviate significantly from the expected or normal
behavior within a dataset. In the context of time series, this could
mean detecting sudden spikes or drops, unusual patterns, or periods
where the data behaves differently from the norm.

Incorporating anomaly detection into forecasting workflows can:

* Improve model robustness by identifying or down-weighting unusual
  data points during training.
* Provide insights into data quality issues or real-world events
  impacting the time series.
* Help understand when and why a forecasting model might be struggling
  (e.g., high prediction errors coinciding with detected anomalies).

``fusionlab`` provides components and integrates strategies (especially
within :class:`~fusionlab.nn.XTFT`) to leverage anomaly information.

Anomaly Detection Components (`fusionlab.nn.anomaly_detection`)
----------------------------------------------------------------

These are neural network layers designed specifically for anomaly
detection tasks, often intended to be used within or alongside
forecasting models.

.. _lstm_autoencoder_anomaly:

LSTMAutoencoderAnomaly
~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.anomaly_detection.LSTMAutoencoderAnomaly`

**Concept:** Reconstruction-Based Anomaly Detection

This layer implements an LSTM-based autoencoder. The core idea is
to train the model to reconstruct "normal" time series sequences accurately.
It learns a compressed representation (encoding) of typical patterns
and then attempts to rebuild the original sequence (decoding) from that
representation.

.. math::
   \mathbf{z} = \text{Encoder}_{LSTM}(\mathbf{X}) \quad \rightarrow \quad \mathbf{\hat{X}} = \text{Decoder}_{LSTM}(\mathbf{z})

Anomalous sequences, which do not conform to the patterns learned
from normal data, are expected to have a higher **reconstruction
error** (the difference between the original input :math:`\mathbf{X}` and
the reconstructed output :math:`\mathbf{\hat{X}}`).

**How it Works:**

* Takes an input sequence (Batch, TimeSteps, Features).
* The encoder LSTM processes the sequence and produces a latent
  vector (typically the final hidden state).
* The decoder LSTM takes this latent vector (repeated across time)
  and generates the reconstructed sequence.
* Returns the reconstructed sequence :math:`\mathbf{\hat{X}}`. The output
  shape depends on the `n_repeats` and `n_features` parameters (see
  API reference for details).

**Usage:**

1.  **Training:** Train the autoencoder typically on data assumed to
    be *normal*, minimizing a reconstruction loss like Mean Squared
    Error (MSE) between the input and the output. This is an
    *unsupervised* approach as it doesn't require anomaly labels.
2.  **Scoring:** After training, feed new (or training/validation)
    sequences into the autoencoder. Calculate the reconstruction error
    for each sequence (e.g., using the layer's
    `.compute_reconstruction_error()` method which calculates MSE per
    sample).
3.  **Detection:** Use the reconstruction error as an anomaly score.
    Sequences with errors exceeding a predefined threshold (determined
    based on validation data or domain knowledge) can be flagged as
    anomalous.

**Integration:** The anomaly scores derived from the reconstruction error
could potentially be used as input for the `'from_config'` strategy in
:class:`~fusionlab.nn.XTFT` by pre-calculating them.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   # Assuming LSTMAutoencoderAnomaly is importable
   from fusionlab.nn.anomaly_detection import LSTMAutoencoderAnomaly

   # Config
   batch_size = 4
   time_steps = 20
   features = 5
   latent_dim = 8  # Size of internal compressed representation
   lstm_units = 16 # Units in LSTM layers

   # Dummy input sequence
   dummy_input = tf.random.normal((batch_size, time_steps, features))

   # Instantiate the layer (using enhanced version parameters)
   lstm_ae_layer = LSTMAutoencoderAnomaly(
       latent_dim=latent_dim,
       lstm_units=lstm_units,
       num_encoder_layers=1, # Example: 1 encoder layer
       num_decoder_layers=1, # Example: 1 decoder layer
       n_features=features,  # Reconstruct original feature count
       n_repeats=time_steps, # Reconstruct original time step count
       activation='tanh'
   )

   # Apply the layer to get reconstructions
   reconstructions = lstm_ae_layer(dummy_input)

   # Compute reconstruction error (MSE per sample)
   recon_error = lstm_ae_layer.compute_reconstruction_error(
       dummy_input, reconstructions
   )

   print(f"Input shape: {dummy_input.shape}")
   print(f"Reconstruction shape: {reconstructions.shape}")
   print(f"Reconstruction Error shape (per sample): {recon_error.shape}")
   # Expected shapes: (4, 20, 5), (4, 20, 5), (4,)


.. _sequence_anomaly_score_layer:

SequenceAnomalyScoreLayer
~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.anomaly_detection.SequenceAnomalyScoreLayer`

**Concept:** Feature-Based Anomaly Scoring

This layer learns to directly predict an anomaly score from a set of
input features. These input features are typically learned representations
extracted from a time series by preceding layers in a larger model (e.g.,
the final hidden state of an LSTM, the output of attention layers, or
an aggregated feature vector).

**How it Works:**

* Takes input features (typically Batch, Features).
* Passes these features through one or more internal Dense layers
  with non-linear activations and optional dropout/normalization.
* A final Dense layer with a single output neuron produces the scalar
  anomaly score for each input sample. The activation of this final
  layer (e.g., 'linear' for unbounded score, 'sigmoid' for 0-1 score)
  determines the score's range.

**Usage:**

1.  **Integration:** Add this layer near the end of a larger neural
    network architecture (like a modified XTFT or a custom model). It
    takes informative features from the network as input.
2.  **Training:** Training requires a loss function that incorporates
    this anomaly score output. This could involve supervised training
    with anomaly labels or unsupervised/semi-supervised integration
    with a primary task loss (e.g., forecasting).
3.  **Detection:** Use the output score directly. Higher scores indicate
    a higher likelihood of the input features representing an anomaly,
    as interpreted by the trained layer. Apply thresholding as needed.

**Integration:** This type of layer aligns conceptually with the
`'feature_based'` anomaly detection strategy mentioned in relation to
:class:`~fusionlab.nn.XTFT`, where anomaly scores are computed internally
from learned features.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.anomaly_detection import SequenceAnomalyScoreLayer

   # Config
   batch_size = 4
   feature_dim = 32 # Dimension of features input to this layer

   # Dummy input features (e.g., output from previous layers)
   learned_features = tf.random.normal((batch_size, feature_dim))

   # Instantiate the layer
   anomaly_scorer = SequenceAnomalyScoreLayer(
       hidden_units=[16, 8], # Example: 2 hidden layers
       activation='relu',
       dropout_rate=0.1,
       final_activation='linear' # Output unbounded score
   )

   # Apply the layer
   anomaly_scores = anomaly_scorer(learned_features, training=False)

   print(f"Input features shape: {learned_features.shape}")
   print(f"Output anomaly scores shape: {anomaly_scores.shape}")
   # Expected: (4, 32), (4, 1)


.. _prediction_error_anomaly_score:

PredictionErrorAnomalyScore
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.anomaly_detection.PredictionErrorAnomalyScore`

**Concept:** Prediction-Error-Based Anomaly Scoring

This layer quantifies the discrepancy between ground truth (`y_true`)
and model predictions (`y_pred`) for time series, aggregating the
error across time and features to produce a single anomaly score per
sequence.

**Functionality:**

1. Takes input as a list `[y_true, y_pred]`, where both tensors
   typically have shape :math:`(B, T, F)`.
2. Calculates the element-wise error based on the specified
   `error_metric` ('mae' or 'mse').

   .. math::
      \text{MAE}_t = \frac{1}{F} \sum_{f=1}^F |y_{true; t,f} - y_{pred; t,f}|
      \; \text{ or } \;
      \text{MSE}_t = \frac{1}{F} \sum_{f=1}^F (y_{true; t,f} - y_{pred; t,f})^2

3. Aggregates these per-step errors across the time dimension :math:`T`
   using the specified `aggregation` method ('mean' or 'max').
4. Returns a scalar anomaly score for each sequence in the batch
   (shape :math:`(B, 1)`).

**Usage Context:** Designed to be used when paired ground truth and
predictions are available. It directly links the anomaly score to the
model's predictive performance on a sequence. The output score can be
used in a custom loss function or training step (similar to the logic
in :func:`~fusionlab.nn.losses.prediction_based_loss`) to penalize
large prediction deviations, thereby implicitly identifying anomalies.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.anomaly_detection import PredictionErrorAnomalyScore

   # Config
   batch_size = 4
   time_steps = 10
   features = 1

   # Dummy true and predicted sequences
   y_true = tf.random.normal((batch_size, time_steps, features))
   # Simulate predictions with some noise
   y_pred = y_true + tf.random.normal(tf.shape(y_true), stddev=0.5)

   # Instantiate the layer (MAE, max aggregation)
   error_scorer = PredictionErrorAnomalyScore(
       error_metric='mae',
       aggregation='max'
   )

   # Calculate scores
   anomaly_scores = error_scorer([y_true, y_pred])

   print(f"Input y_true shape: {y_true.shape}")
   print(f"Input y_pred shape: {y_pred.shape}")
   print(f"Output anomaly scores shape: {anomaly_scores.shape}")
   # Expected: (4, 10, 1), (4, 10, 1), (4, 1)


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


Using Anomaly Detection with XTFT
-----------------------------------

The :class:`~fusionlab.nn.XTFT` model provides specific parameters to
integrate anomaly detection during training:

* ``anomaly_detection_strategy``: Can be set to ``'prediction_based'``
  (derives scores from prediction errors using
  :func:`~fusionlab.nn.losses.prediction_based_loss`), potentially
  ``'feature_based'`` (using internal layers like
  :class:`SequenceAnomalyScoreLayer`), or implies ``'from_config'`` logic
  when used with specific combined losses like
  :func:`~fusionlab.nn.losses.combined_total_loss`.
* ``anomaly_loss_weight``: Controls the relative importance of the
  anomaly objective compared to the main forecasting objective in the
  loss function.
* ``anomaly_config``: A dictionary potentially used to pass pre-computed
  scores (for ``'from_config'`` logic) or configure internal anomaly
  components.

Refer to the :doc:`/user_guide/examples/xtft_with_anomaly_detection`
example for practical implementations of the `'from_config'` (via
combined loss) and `'prediction_based'` strategies.
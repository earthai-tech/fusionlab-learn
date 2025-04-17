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
-------------------------------------------------------------

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
error** (the difference between the original input $\mathbf{X}$ and
the reconstructed output $\mathbf{\hat{X}}$).

**How it Works:**
* Takes an input sequence (Batch, TimeSteps, Features).
* The encoder LSTM processes the sequence and produces a latent
    vector (typically the final hidden state).
* The decoder LSTM takes this latent vector (repeated across time)
    and generates the reconstructed sequence.
* Returns the reconstructed sequence $\mathbf{\hat{X}}$ (same shape
    as input).

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
    with non-linear activations and optional dropout.
* A final Dense layer with a single output neuron produces the scalar
    anomaly score for each input sample. The activation of this final
    layer ('linear' for unbounded score, 'sigmoid' for 0-1 score)
    determines the score's range.

**Usage:**
1.  **Integration:** Add this layer near the end of a larger neural
    network architecture (like a modified XTFT or a custom model). It
    takes informative features from the network as input.
2.  **Training:** Training requires a loss function that incorporates
    this anomaly score output. This could involve:
    * **Supervised:** If anomaly labels are available, train to predict
        high scores for anomalies and low scores for normal data (e.g.,
        using binary cross-entropy if output is sigmoid).
    * **Unsupervised/Semi-supervised:** Integrate the score into a
        combined loss function alongside the main task's loss (e.g.,
        forecasting loss). For example, penalize the model if it produces
        high anomaly scores for data points that are otherwise well-predicted
        or reconstructed, or use it within contrastive learning frameworks.
3.  **Detection:** Use the output score directly. Higher scores indicate
    a higher likelihood of the input features representing an anomaly,
    as interpreted by the trained layer. Apply thresholding as needed.

**Integration:** This type of layer aligns conceptually with the
`'feature_based'` anomaly detection strategy mentioned in relation to
:class:`~fusionlab.nn.XTFT`, where anomaly scores are computed internally
from learned features.

Using Anomaly Detection with XTFT
---------------------------------

The :class:`~fusionlab.nn.XTFT` model provides specific parameters to
integrate anomaly detection during training:

* `anomaly_detection_strategy`: Can be set to `'prediction_based'`
    (derives scores from prediction errors using
    :func:`~fusionlab.nn.losses.prediction_based_loss`) or potentially
    `'feature_based'` (using internal layers like
    :class:`SequenceAnomalyScoreLayer`) or implies `'from_config'` logic
    when used with specific combined losses like
    :func:`~fusionlab.nn.losses.combined_total_loss`.
* `anomaly_loss_weight`: Controls the relative importance of the
    anomaly objective compared to the main forecasting objective in the
    loss function.
* `anomaly_config`: A dictionary potentially used to pass pre-computed
    scores (for `'from_config'` logic) or configure internal anomaly
    components.

Refer to the :doc:`/user_guide/examples/xtft_with_anomaly_detection`
example for practical implementations of the `'from_config'` (via
combined loss) and `'prediction_based'` strategies.
.. _gallery_anomaly_detection_examples:

================================
Anomaly Detection Examples
================================

This section of the gallery provides practical examples of using the
anomaly detection components available in ``fusionlab-learn``. These
components can be used to build standalone anomaly detection models or
be integrated into larger forecasting frameworks like
:class:`~fusionlab.nn.XTFT` to make them anomaly-aware.

We will cover:

1.  Using :class:`~fusionlab.nn.anomaly_detection.LSTMAutoencoderAnomaly`
    for reconstruction-based anomaly detection.
2.  Conceptual usage of
    :class:`~fusionlab.nn.anomaly_detection.SequenceAnomalyScoreLayer`
    for feature-based anomaly scoring.
3.  Using
    :class:`~fusionlab.nn.anomaly_detection.PredictionErrorAnomalyScore`
    to derive anomaly scores from prediction errors.


Prerequisites
-------------

Ensure you have ``fusionlab-learn`` and its dependencies installed.
For some examples, `matplotlib` and `scikit-learn` are also useful.

.. code-block:: bash

   pip install fusionlab-learn matplotlib scikit-learn

Example 1: LSTM Autoencoder for Anomaly Detection
----------------------------------------------------
The :class:`~fusionlab.nn.anomaly_detection.LSTMAutoencoderAnomaly`
learns to reconstruct normal time series sequences. Anomalous sequences
are expected to have higher reconstruction errors.

Workflow:
~~~~~~~~~
1. Generate synthetic time series data with injected anomalies.
2. Preprocess the data (scaling, sequence creation).
3. Define, compile, and train the `LSTMAutoencoderAnomaly` model.
4. Calculate reconstruction errors on all sequences.
5. Identify anomalies by thresholding the reconstruction errors.
6. Visualize the results.

Step 1.1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from sklearn.preprocessing import StandardScaler
   import warnings
   import os

   # FusionLab imports
   from fusionlab.nn.anomaly_detection import LSTMAutoencoderAnomaly
   from fusionlab.nn.utils import create_sequences # For sequence prep

   # Suppress warnings and TF logs
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   if hasattr(tf, 'autograph'):
       tf.autograph.set_verbosity(0)

   output_dir_ad = "./anomaly_detection_gallery_output"
   os.makedirs(output_dir_ad, exist_ok=True)
   print("Libraries imported for LSTM Autoencoder example.")

Step 1.2: Generate Data with Anomalies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   # Create a sine wave with some noise
   time = np.arange(0, 200, 0.5)
   signal = np.sin(time * 0.1) + np.random.normal(0, 0.1, len(time))

   # Inject anomalies
   signal_with_anomalies = signal.copy()
   signal_with_anomalies[50:60] += 2.5  # Spike up
   signal_with_anomalies[150:155] -= 2.0 # Dip down

   df_ad = pd.DataFrame({'Timestamp': time, 'Value': signal_with_anomalies})
   print(f"Generated data shape: {df_ad.shape}")

   # Visualize the data
   plt.figure(figsize=(12, 4))
   plt.plot(df_ad['Timestamp'], df_ad['Value'], label='Signal with Anomalies')
   plt.title('Synthetic Time Series with Injected Anomalies')
   plt.xlabel('Time'); plt.ylabel('Value')
   plt.legend(); plt.grid(True)
   # plt.savefig(os.path.join(output_dir_ad, "ad_data_with_anomalies.png"))
   plt.show()

.. figure:: ../../images/gallery_anomaly_detection_ad_data_with_anomalies.png
   :alt: LSTM Autoencoder Anomaly Detection
   :align: center
   :width: 90%


Step 1.3: Preprocessing and Sequence Creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   scaler_ad = StandardScaler()
   df_ad['ScaledValue'] = scaler_ad.fit_transform(df_ad[['Value']])

   sequence_length_ad = 20 # Length of input sequences for autoencoder
   X_sequences, _ = create_sequences(
       df_ad[['ScaledValue']],
       sequence_length=sequence_length_ad,
       target_col='ScaledValue',
       forecast_horizon=0,
       drop_last=False,
       verbose=0
   )
   y_sequences = X_sequences.copy()

   X_train_ad = X_sequences.reshape(
       X_sequences.shape[0], sequence_length_ad, 1
       ).astype(np.float32)
   y_train_ad = y_sequences.reshape(
       y_sequences.shape[0], sequence_length_ad, 1
       ).astype(np.float32)

   print(f"\nTraining sequences (X) shape: {X_train_ad.shape}")
   print(f"Target sequences (y) shape: {y_train_ad.shape}")

Step 1.4: Define, Compile, and Train LSTM Autoencoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `LSTMAutoencoderAnomaly` model is defined. We ensure that if a
bidirectional encoder or a bottleneck dense layer changes the dimension
of the encoder's final state, it's projected to match the decoder's
LSTM units before being used as an initial state.

.. code-block:: python
   :linenos:

   latent_dim_ad = 8
   lstm_units_ad = 16 # Keep this consistent with test that failed
   n_features_ad = 1

   autoencoder_model = LSTMAutoencoderAnomaly(
       latent_dim=latent_dim_ad,
       lstm_units=lstm_units_ad, # Decoder LSTMs will have this many units
       num_encoder_layers=1,
       num_decoder_layers=1,
       n_features=n_features_ad,
       n_repeats=sequence_length_ad,
       use_bidirectional_encoder=True, # This was True in the failing test
       use_bottleneck_dense=False,    # This was False in the failing test
       name="lstm_autoencoder_anomaly_detector"
   )

   autoencoder_model.compile(optimizer='adam', loss='mse')
   print("\nLSTM Autoencoder model compiled.")

   print("Training LSTM Autoencoder...")
   # Build the model with the input shape before fitting
   # This ensures all layers, including conditional ones in build, are created.
   autoencoder_model.build(input_shape=(None, sequence_length_ad, n_features_ad))
   # autoencoder_model.summary() # Optional: view model structure

   history_ad = autoencoder_model.fit(
       X_train_ad, y_train_ad,
       epochs=20,
       batch_size=32,
       shuffle=True,
       verbose=0
   )
   print("Training complete.")
   if history_ad and history_ad.history.get('loss'):
       print(f"Final training loss: {history_ad.history['loss'][-1]:.4f}")

Step 1.5: Calculate Reconstruction Errors (Anomaly Scores)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   reconstructions_ad = autoencoder_model.predict(X_train_ad, verbose=0)
   reconstruction_errors = autoencoder_model.compute_reconstruction_error(
       X_train_ad, reconstructions_ad
   )
   print(f"\nReconstruction errors shape: {reconstruction_errors.shape}")

   anomaly_scores_ts = np.full(len(df_ad), np.nan)
   for i, error_val in enumerate(reconstruction_errors):
       if i + sequence_length_ad -1 < len(anomaly_scores_ts): # Boundary check
           anomaly_scores_ts[i + sequence_length_ad - 1] = error_val

Step 1.6: Identify Anomalies and Visualize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   # Filter out NaNs from reconstruction_errors before calculating percentile
   valid_errors = reconstruction_errors[~np.isnan(reconstruction_errors)]
   if len(valid_errors) > 0:
       threshold_ad = np.percentile(valid_errors, 95)
       print(f"Anomaly threshold (95th percentile of errors): {threshold_ad:.4f}")
       anomalous_indices = np.where(reconstruction_errors > threshold_ad)[0]
       anomalous_time_points = [
           idx + sequence_length_ad - 1 for idx in anomalous_indices
           if idx + sequence_length_ad - 1 < len(df_ad) # Boundary check
           ]
   else:
       print("No valid reconstruction errors to calculate threshold.")
       threshold_ad = np.inf # Set to infinity if no errors
       anomalous_time_points = []


   plt.figure(figsize=(12, 6))
   plt.subplot(2, 1, 1)
   plt.plot(df_ad['Timestamp'], df_ad['Value'], label='Original Signal')
   if anomalous_time_points:
       plt.scatter(df_ad['Timestamp'].iloc[anomalous_time_points],
                   df_ad['Value'].iloc[anomalous_time_points],
                   color='red', label='Detected Anomalies', marker='o', s=50, zorder=5)
   plt.title('Signal with Detected Anomalies (LSTM Autoencoder)')
   plt.ylabel('Value'); plt.legend(); plt.grid(True)

   plt.subplot(2, 1, 2)
   plt.plot(df_ad['Timestamp'], anomaly_scores_ts,
            label='Reconstruction Error (Anomaly Score)', color='orange')
   if np.isfinite(threshold_ad): # Only plot threshold if it's finite
       plt.axhline(threshold_ad, color='red', linestyle='--',
                   label=f'Anomaly Threshold ({threshold_ad:.2f})')
   plt.title('Anomaly Scores Over Time')
   plt.xlabel('Time'); plt.ylabel('Reconstruction Error (MSE)')
   plt.legend(); plt.grid(True)
   plt.tight_layout()
   # plt.savefig(os.path.join(output_dir_ad, "ad_lstm_ae_results.png"))
   plt.show()

**Example Output Plot (LSTM Autoencoder):**

.. figure:: ../../images/gallery_lstm_autoencoder_anomaly.png
   :alt: LSTM Autoencoder Anomaly Detection
   :align: center
   :width: 90%

   Top: Original signal with detected anomalies. Bottom: Reconstruction
   error over time with the anomaly threshold.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


Example 2: Using SequenceAnomalyScoreLayer (Conceptual)
-------------------------------------------------------
The :class:`~fusionlab.nn.anomaly_detection.SequenceAnomalyScoreLayer`
is designed to be integrated into a larger model. It takes learned
features (e.g., from an encoder or attention layers) as input and
outputs a scalar anomaly score. Training this layer requires a custom
setup with an appropriate loss function, not shown in this isolated
example.

Step 2.1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.anomaly_detection import SequenceAnomalyScoreLayer
   print("\nLibraries imported for SequenceAnomalyScoreLayer example.")


Step 2.2: Instantiate and Use the Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   # Assume 'learned_features' is the output of a preceding layer
   # (e.g., aggregated output of XTFT's attention/LSTM blocks)
   # Shape: (Batch, FeatureDim)
   batch_size_sas = 16
   feature_dim_sas = 64
   learned_features_sas = tf.random.normal(
       (batch_size_sas, feature_dim_sas), dtype=tf.float32
       )

   # Instantiate the scoring layer
   anomaly_scorer_layer = SequenceAnomalyScoreLayer(
       hidden_units=32, # Hidden units in the scorer's internal MLP
       activation='relu',
       dropout_rate=0.1,
       final_activation='linear' # Output an unbounded score
   )

   # Pass features through the layer to get scores
   # (This is typically done within a main model's call method)
   anomaly_scores_output = anomaly_scorer_layer(
       learned_features_sas, training=False
       )

   print(f"\nInput features shape: {learned_features_sas.shape}")
   print(f"Output anomaly scores shape: {anomaly_scores_output.shape}")
   # Expected output: (Batch, 1) -> (16, 1)

.. note::
   The `SequenceAnomalyScoreLayer` needs to be trained as part of a
   larger model. The loss function would guide what these scores
   represent (e.g., using anomaly labels if available, or incorporating
   it into an unsupervised objective). This example only shows the
   forward pass. Refer to the XTFT `'feature_based'` strategy
   discussion in the :doc:`User Guide </user_guide/anomaly_detection>`
   for conceptual integration.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Example 3: Using PredictionErrorAnomalyScore
---------------------------------------------
The :class:`~fusionlab.nn.anomaly_detection.PredictionErrorAnomalyScore`
layer calculates an anomaly score based directly on the difference
(error) between true values and a model's predicted values for a sequence.

Step 3.1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np # For a more visible error injection
   from fusionlab.nn.anomaly_detection import PredictionErrorAnomalyScore
   print("\nLibraries imported for PredictionErrorAnomalyScore example.")

Step 3.2: Instantiate and Use the Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   # Config
   batch_size_peas = 4
   time_steps_peas = 10
   features_peas = 1 # Univariate example

   # Dummy true values (e.g., from a dataset)
   y_true_peas = tf.random.normal(
       (batch_size_peas, time_steps_peas, features_peas), dtype=tf.float32
       )
   # Dummy predicted values (e.g., from a forecasting model)
   # Add some noise to simulate prediction errors
   y_pred_peas_np = y_true_peas.numpy() + np.random.normal(
       scale=0.6, size=y_true_peas.shape
       ).astype(np.float32)
   # Inject a larger error for one sample to see difference in 'max' aggregation
   y_pred_peas_np[1, 5, 0] += 5.0 # Add large error to sample 1, step 5
   y_pred_peas = tf.constant(y_pred_peas_np)


   # --- Instantiate with MAE and Mean Aggregation ---
   error_scorer_mean = PredictionErrorAnomalyScore(
       error_metric='mae',     # Use Mean Absolute Error
       aggregation='mean'    # Average errors across time steps
   )
   # Calculate scores (average error per sequence)
   anomaly_scores_mean = error_scorer_mean([y_true_peas, y_pred_peas])

   # --- Instantiate with MAE and Max Aggregation ---
   error_scorer_max = PredictionErrorAnomalyScore(
       error_metric='mae',     # Use Mean Absolute Error
       aggregation='max'     # Take max error across time steps
   )
   # Calculate scores (max error per sequence)
   anomaly_scores_max = error_scorer_max([y_true_peas, y_pred_peas])

   print(f"\nInput y_true shape: {y_true_peas.shape}")
   print(f"Input y_pred shape: {y_pred_peas.shape}")
   print("\n--- MAE + Mean Aggregation ---")
   print(f"Output anomaly scores shape: {anomaly_scores_mean.shape}")
   print(f"Example Scores (Mean Error per sequence): \n"
         f"{anomaly_scores_mean.numpy().flatten()}")
   print("\n--- MAE + Max Aggregation ---")
   print(f"Output anomaly scores shape: {anomaly_scores_max.shape}")
   print(f"Example Scores (Max Error per sequence): \n"
         f"{anomaly_scores_max.numpy().flatten()}")
   # Expected output shapes for scores: (Batch, 1) -> (4, 1)

.. note::
   The scores from `PredictionErrorAnomalyScore` can be used to
   construct a loss term that penalizes large prediction deviations,
   aligning with the `'prediction_based'` anomaly detection strategy
   in :class:`~fusionlab.nn.XTFT`.



.. _example_ad_components:

============================
Anomaly Detection Examples
============================

This page provides examples on how to use the specialized anomaly
detection components available in
``fusionlab.nn.anomaly_detection``. These components can be used
independently for unsupervised anomaly detection or integrated into
larger models.

See the :doc:`../anomaly_detection` section in the User Guide for
more conceptual details.

Example 1: Unsupervised Anomaly Detection with LSTM Autoencoder
---------------------------------------------------------------

This example demonstrates using the
:class:`~fusionlab.nn.anomaly_detection.LSTMAutoencoderAnomaly`
layer to detect anomalies in a time series based on reconstruction
error. We train the autoencoder on the sequence data and identify
anomalies as points where the reconstruction error is high.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from sklearn.preprocessing import StandardScaler

   # Assuming fusionlab components are importable
   from fusionlab.nn.anomaly_detection import LSTMAutoencoderAnomaly

   # Suppress warnings and TF logs for cleaner output
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)

   # 1. Generate Synthetic Data with Anomalies
   # -----------------------------------------
   np.random.seed(42)
   time = np.arange(0, 200, 0.5)
   # Normal sine wave
   normal_data = np.sin(time * 0.5) + np.random.normal(0, 0.1, len(time))
   # Inject anomalies
   anomalous_data = normal_data.copy()
   anomalous_data[50:60] += 2.0  # Add a positive spike
   anomalous_data[150:155] -= 1.5 # Add a negative dip

   df = pd.DataFrame({'Value': anomalous_data, 'Time': time})
   print("Generated data shape:", df.shape)

   plt.figure(figsize=(12, 3))
   plt.plot(df['Time'], df['Value'], label='Data with Anomalies')
   plt.title("Synthetic Data")
   plt.xlabel("Time")
   plt.ylabel("Value")
   plt.legend()
   plt.grid(True)
   plt.show()

   # 2. Preprocessing: Scaling
   # -------------------------
   scaler = StandardScaler()
   df['Value_Scaled'] = scaler.fit_transform(df[['Value']])
   print("Data scaled.")

   # 3. Create Sequences for Autoencoder
   # -----------------------------------
   # Autoencoders are trained to reconstruct input, so X=y
   sequence_length = 20 # Length of sequences to reconstruct
   data_scaled = df['Value_Scaled'].values

   sequences = []
   # Simple sequential windowing
   for i in range(len(data_scaled) - sequence_length + 1):
       sequences.append(data_scaled[i:i + sequence_length])

   sequences = np.array(sequences).astype(np.float32)
   # Reshape for LSTM: (NumSamples, TimeSteps, Features=1)
   sequences = sequences.reshape(sequences.shape[0], sequence_length, 1)
   print(f"Created sequences shape: {sequences.shape}")

   # For unsupervised learning, we typically train on all data,
   # assuming anomalies are rare. Alternatively, train only on known
   # 'normal' data if available. Here, we use all sequences.
   X_train = sequences
   y_train = sequences # Target is the same as input

   # 4. Define LSTM Autoencoder Model
   # --------------------------------
   latent_dim = 8
   lstm_units = 16

   lstm_ae_model = LSTMAutoencoderAnomaly(
       latent_dim=latent_dim,
       lstm_units=lstm_units,
       activation='linear' # Use linear for potentially unbounded scaled data
   )

   # 5. Compile and Train the Autoencoder
   # ------------------------------------
   lstm_ae_model.compile(optimizer='adam', loss='mse')
   print("Autoencoder compiled. Starting training...")

   history = lstm_ae_model.fit(
       X_train, y_train,
       epochs=20, # Train longer for better reconstruction
       batch_size=16,
       shuffle=True, # Shuffle sequences for training
       verbose=0
   )
   print("Training finished.")
   print(f"Final training loss (MSE): {history.history['loss'][-1]:.4f}")

   # 6. Calculate Reconstruction Errors (Anomaly Scores)
   # -------------------------------------------------
   print("Calculating reconstruction errors...")
   # Use the model's helper method to get MSE per sequence
   # Note: This gives one error value PER SEQUENCE (window)
   reconstruction_errors = lstm_ae_model.compute_reconstruction_error(
       sequences # Pass all sequences
   ).numpy()
   print(f"Reconstruction errors shape: {reconstruction_errors.shape}")

   # Simple way to map sequence error back to original time points
   # (Assign error of a sequence to its last point)
   errors_mapped = np.full(len(df), np.nan)
   for i in range(len(reconstruction_errors)):
       # Assign error to the end point of the sequence window
       errors_mapped[i + sequence_length - 1] = reconstruction_errors[i]

   df['ReconstructionError'] = errors_mapped

   # 7. Detect Anomalies using a Threshold
   # -------------------------------------
   # Define threshold (e.g., based on error distribution percentile)
   threshold = np.nanpercentile(df['ReconstructionError'], 95) # Flag top 5% error
   df['Is_Anomaly'] = df['ReconstructionError'] > threshold
   print(f"Anomaly threshold (95th percentile error): {threshold:.4f}")
   print(f"Number of points flagged as anomalies: {df['Is_Anomaly'].sum()}")

   # 8. Visualize Results
   # --------------------
   fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

   # Plot original data and detected anomalies
   ax[0].plot(df['Time'], df['Value'], label='Original Data', zorder=1)
   anomalies = df[df['Is_Anomaly']]
   ax[0].scatter(anomalies['Time'], anomalies['Value'], color='red',
                 label='Detected Anomaly', zorder=5, s=50) # Larger marker
   ax[0].set_ylabel('Value')
   ax[0].set_title('Time Series with Detected Anomalies')
   ax[0].legend()
   ax[0].grid(True)

   # Plot reconstruction error and threshold
   ax[1].plot(df['Time'], df['ReconstructionError'],
              label='Reconstruction Error (MSE per Sequence)', color='orange')
   ax[1].axhline(threshold, color='red', linestyle='--',
                 label=f'Threshold ({threshold:.4f})')
   ax[1].set_ylabel('Reconstruction Error (MSE)')
   ax[1].set_xlabel('Time')
   ax[1].set_title('Reconstruction Error and Anomaly Threshold')
   ax[1].legend()
   ax[1].grid(True)

   plt.tight_layout()
   plt.show()


Example 2: Using SequenceAnomalyScoreLayer (Conceptual)
-------------------------------------------------------

This layer is designed to be integrated into a larger model, taking
learned features as input and outputting a scalar anomaly score. Here's
how you might instantiate and use it conceptually. Training requires
a custom setup not shown in this isolated example.

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.anomaly_detection import SequenceAnomalyScoreLayer

   # Assume 'learned_features' is the output of a preceding layer
   # (e.g., aggregated output of XTFT's attention/LSTM blocks)
   # Shape: (Batch, FeatureDim)
   batch_size = 16
   feature_dim = 64
   learned_features = tf.random.normal((batch_size, feature_dim))

   # Instantiate the scoring layer
   anomaly_scorer = SequenceAnomalyScoreLayer(
       output_units=32, # Hidden units in the scorer MLP
       activation='relu',
       dropout_rate=0.1,
       final_activation='linear' # Output an unbounded score
   )

   # Pass features through the layer to get scores
   # (Typically done within the main model's call method)
   anomaly_scores = anomaly_scorer(learned_features)

   print("Input features shape:", learned_features.shape)
   print("Output anomaly scores shape:", anomaly_scores.shape)
   # Expected: (Batch, 1) -> (16, 1)


Example 3: Using PredictionErrorAnomalyScore
---------------------------------------------

This layer calculates an anomaly score based directly on the difference
between true values and predicted values for a sequence.

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.anomaly_detection import PredictionErrorAnomalyScore

   # Config
   batch_size = 4
   time_steps = 10
   features = 1

   # Dummy true values (e.g., from dataset)
   y_true = tf.random.normal((batch_size, time_steps, features))
   # Dummy predicted values (e.g., output from a forecasting model)
   # Add some noise to simulate prediction errors
   y_pred = y_true + tf.random.normal(tf.shape(y_true), stddev=0.6)
   # Add a larger error for one sample to see difference in 'max' aggregation
   y_pred = tf.tensor_scatter_nd_update(
       y_pred, [[1, 5, 0]], [y_pred[1, 5, 0] + 5.0] # Add large error to sample 1, step 5
   )

   # --- Instantiate with MAE and Mean Aggregation ---
   error_scorer_mean = PredictionErrorAnomalyScore(
       error_metric='mae',
       aggregation='mean'
   )
   # Calculate scores (average error per sequence)
   anomaly_scores_mean = error_scorer_mean([y_true, y_pred])

   # --- Instantiate with MAE and Max Aggregation ---
   error_scorer_max = PredictionErrorAnomalyScore(
       error_metric='mae',
       aggregation='max'
   )
   # Calculate scores (max error per sequence)
   anomaly_scores_max = error_scorer_max([y_true, y_pred])


   print(f"Input y_true shape: {y_true.shape}")
   print(f"Input y_pred shape: {y_pred.shape}")
   print("\n--- MAE + Mean Aggregation ---")
   print(f"Output anomaly scores shape: {anomaly_scores_mean.shape}")
   print(f"Example Scores (Mean Error): \n{anomaly_scores_mean.numpy()}")
   print("\n--- MAE + Max Aggregation ---")
   print(f"Output anomaly scores shape: {anomaly_scores_max.shape}")
   print(f"Example Scores (Max Error): \n{anomaly_scores_max.numpy()}")
   # Expected output shapes: (4, 10, 1), (4, 10, 1), (4, 1), (4, 1)
   # Note how scores differ based on aggregation, esp. for sample 1


.. topic:: Explanations

    **Example 1: LSTM Autoencoder**

    1.  **Data Generation & Viz:** We create a simple sine wave and
        manually add two anomalous periods (a spike and a dip) to have
        clear irregularities to detect. We plot the data first.
    2.  **Preprocessing:** The 'Value' column is scaled using
        `StandardScaler`. Scaling is important for neural network
        stability and performance.
    3.  **Sequence Preparation:** We create overlapping sequences of a fixed
        `sequence_length`. For an autoencoder, the input and target are
        the same sequence, as the goal is reconstruction. The data is
        reshaped to `(Samples, TimeSteps, Features=1)` for the LSTM input.
    4.  **Model Definition:** We instantiate the
        :class:`~fusionlab.nn.anomaly_detection.LSTMAutoencoderAnomaly`
        layer, specifying the size of the compressed representation
        (`latent_dim`) and the number of units in the LSTM layers
        (`lstm_units`).
    5.  **Compile & Train:** The model is compiled with 'adam' optimizer
        and 'mse' loss, suitable for a reconstruction task. We train the
        model on *all* sequences (`X_train`, `y_train` are identical)
        in an unsupervised manner. The model learns to minimize the
        reconstruction error for the patterns present in the data.
    6.  **Anomaly Score Calculation:** We feed *all* sequences through the
        *trained* autoencoder. We use the layer's built-in
        `.compute_reconstruction_error()` method, which calculates the
        Mean Squared Error between each input sequence and its
        reconstruction. This error serves as our anomaly score for each
        sequence window. We then map these sequence errors back to
        individual time points (here, simplistically assigning the error
        to the end of the window).
    7.  **Thresholding:** A simple thresholding strategy is applied. We
        calculate a threshold based on a high percentile (e.g., 95th) of
        the reconstruction errors. Sequence windows whose errors exceed
        this threshold are flagged as anomalous time points. More
        sophisticated thresholding methods exist.
    8.  **Visualization:** We plot the original data with the detected
        anomalies highlighted, and separately plot the reconstruction
        error over time along with the calculated threshold, showing how
        the error spikes during the anomalous periods.

    **Example 2: SequenceAnomalyScoreLayer**

    1.  **Concept:** This layer doesn't perform detection on its own but
        acts as a scoring head within a larger network.
    2.  **Instantiation:** We show how to create an instance, specifying
        hidden units and activations.
    3.  **Usage:** We demonstrate passing hypothetical `learned_features`
        (which would come from other layers in a real model) through the
        scorer to get anomaly scores.
    4.  **Training Note:** The crucial point is that this layer needs to
        be *trained* as part of a larger model with an appropriate loss
        function that guides the meaning of the score (e.g., using anomaly
        labels if available, or incorporating it into an unsupervised/
        semi-supervised objective). This isolated example only shows the
        forward pass. Refer to the XTFT `'feature_based'` strategy discussion
        in the :doc:`../anomaly_detection` guide for conceptual integration.
        
   **Example 3: PredictionErrorAnomalyScore**

   1.  **Concept:** This layer directly quantifies the difference between
       actual (`y_true`) and predicted (`y_pred`) sequences.
   2.  **Data:** We create dummy `y_true` and `y_pred` tensors, adding
       some noise to `y_pred` to simulate prediction errors.
   3.  **Instantiation:** We show how to create the layer, specifying the
       `error_metric` ('mae' or 'mse') and the `aggregation` method
       ('mean' or 'max') for combining errors across time steps.
   4.  **Usage:** The layer is called with a list containing the true and
       predicted tensors: `layer([y_true, y_pred])`.
   5.  **Output:** It returns a single score per sequence in the batch
       (shape `(Batch, 1)`), representing either the average or maximum
       prediction error for that sequence. This score can be used in
       loss functions similar to the `'prediction_based'` strategy in XTFT.
.. _quickstart:

============
Quickstart
============

This guide provides a minimal example to get you started with
training a basic forecasting model using ``fusionlab``.

Prerequisites
--------------

Make sure you have installed ``fusionlab`` and its core
dependencies, including TensorFlow. If not, please follow the
:ref:`installation` guide first.

Steps
-----

Let's train a simple `TemporalFusionTransformer` for point
forecasting using only dynamic (past) inputs.

1. **Import Libraries**
   We need TensorFlow, NumPy, and the model class.

   .. code-block:: python

      import tensorflow as tf
      import numpy as np
      from fusionlab.nn import TemporalFusionTransformer

      # Optional: Suppress TensorFlow warnings for cleaner output
      tf.get_logger().setLevel('ERROR')
      tf.autograph.set_verbosity(0)


2. **Prepare Dummy Data**
   We'll create some random data simulating dynamic features and
   a target variable.

   .. code-block:: python

      # Define data dimensions
      batch_size = 16
      num_past_timesteps = 20  # Length of historical input sequence
      dynamic_feature_dim = 3  # Number of dynamic features
      forecast_horizon = 5     # Number of steps to predict

      # Generate random dynamic (past) input data
      # Shape: (batch_size, num_past_timesteps, dynamic_feature_dim)
      X_dynamic = np.random.rand(
          batch_size, num_past_timesteps, dynamic_feature_dim
      ).astype(np.float32)

      # Generate random target data (what we want to predict)
      # Shape: (batch_size, forecast_horizon, 1) -> Point forecast (1 value per step)
      y_target = np.random.rand(
          batch_size, forecast_horizon, 1
      ).astype(np.float32)

      print(f"Dynamic Input Shape: {X_dynamic.shape}")
      print(f"Target Output Shape: {y_target.shape}")


3. **Instantiate the Model**
   Create an instance of `TemporalFusionTransformer`. Since we are
   only using dynamic inputs, we only need to specify
   `dynamic_input_dim`. We also set the `forecast_horizon`.
   We omit `quantiles` for point forecasting.

   .. code-block:: python

      model = TemporalFusionTransformer(
          dynamic_input_dim=dynamic_feature_dim,
          forecast_horizon=forecast_horizon,
          # Using default values for other parameters like:
          # static_input_dim=None,
          # future_input_dim=None,
          # hidden_units=32,
          # num_heads=4,
          # quantiles=None, # Default is point forecast
          # etc.
      )

      # Optional: Build the model by passing a sample input shape or data
      # This is needed before summary() or plotting can work.
      # Note: Input must be a tuple, even with only one element.
      model.build(input_shape=[(None, num_past_timesteps, dynamic_feature_dim)])
      model.summary()


4. **Compile the Model**
   Specify the optimizer and loss function. For point forecasting,
   Mean Squared Error ('mse') is a common choice.

   .. code-block:: python

      model.compile(optimizer='adam', loss='mse')


5. **Train the Model**
   Fit the model to the dummy data for a few epochs.

   .. code-block:: python

      print("\nTraining the model...")
      history = model.fit(
          x=(X_dynamic,), # Input must be a tuple
          y=y_target,
          epochs=3,       # Use few epochs for a quick demo
          batch_size=4,
          verbose=1       # Show progress
      )
      print("Training complete.")


6. **Make Predictions**
   Use the trained model to generate forecasts on new data (or the
   same data in this example).

   .. code-block:: python

      print("\nMaking predictions...")
      # Use the same input data for prediction in this example
      predictions = model.predict((X_dynamic,))

      print(f"Predictions output shape: {predictions.shape}")
      # Expected shape: (batch_size, forecast_horizon, 1)


Conclusion
------------

This quickstart demonstrated the basic workflow: preparing data,
instantiating a model, compiling it, training it, and making
predictions.

For more advanced use cases involving static/future features,
quantile forecasts, anomaly detection, or the XTFT model, please
refer to the :doc:`User Guide </user_guide/index>` and the
:doc:`API Reference </api>`.
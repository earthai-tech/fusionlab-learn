.. _user_guide_losses:

================
Loss Functions
================

``fusionlab`` provides several custom loss functions tailored for
advanced time series forecasting tasks, particularly those involving
probabilistic (quantile) predictions and integrated anomaly
detection. These functions are designed to be compatible with the
Keras API (e.g., ``model.compile(loss=...)``).

Understanding these losses is key to training models like
:class:`~fusionlab.nn.TemporalFusionTransformer` and
:class:`~fusionlab.nn.XTFT` effectively, especially when dealing
with uncertainty estimation or anomaly-aware training strategies.

Quantile Loss Functions
-------------------------

These functions are used when the goal is to predict specific
quantiles of the target distribution, enabling probabilistic
forecasts. They typically return a callable loss function suitable
for Keras `compile`.

.. _losses_quantile_loss: 

quantile_loss
~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.losses.quantile_loss`

**Purpose:** Creates a Keras-compatible loss function that computes
the quantile (pinball) loss for a **single**, specified quantile :math:`q`.

**Functionality:**
Takes a single quantile :math:`q` (between 0 and 1) as input and returns
a function `loss_fn(y_true, y_pred)`. This returned function
calculates the pinball loss:

.. math::
   L_q(y_{true}, y_{pred}) = \text{mean}(\max(q \cdot error, (q - 1) \cdot error))

where :math:`error = y_{true} - y_{pred}`. The mean is typically taken
across all dimensions except the last feature dimension.

**Usage Context:** Useful when you need to train a model to predict
only one specific quantile of the target distribution. Pass the
result of this function to `model.compile`. For example:
``model.compile(loss=quantile_loss(q=0.75))`` would train the model
to predict the 75th percentile.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np
   from fusionlab.nn.losses import quantile_loss

   # Config
   batch_size = 4
   horizon = 6
   output_dim = 1
   quantile_to_predict = 0.75

   # Dummy true values and predictions for a SINGLE quantile/output
   y_true = tf.random.normal((batch_size, horizon, output_dim))
   y_pred = y_true + tf.random.normal(tf.shape(y_true), stddev=0.5)

   # Create the loss function for the specific quantile
   loss_fn_q75 = quantile_loss(q=quantile_to_predict)

   # Calculate the loss
   loss_value = loss_fn_q75(y_true, y_pred)

   print(f"y_true shape: {y_true.shape}")
   print(f"y_pred shape (single quantile): {y_pred.shape}")
   print(f"Calculated Loss for q={quantile_to_predict}: {loss_value.numpy():.4f}")


.. _losses_quantile_loss_multi:

quantile_loss_multi
~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.losses.quantile_loss_multi`

**Purpose:** Creates a Keras-compatible loss function that computes
the *average* quantile (pinball) loss across a **list** of specified
quantiles.

**Functionality:**
Takes a list of `quantiles` (e.g., `[0.1, 0.5, 0.9]`) as input and
returns a function `loss_fn(y_true, y_pred)`. The model's prediction
`y_pred` is expected to have a final dimension matching the number of
quantiles. The returned function calculates the pinball loss :math:`L_q`
(as defined in :func:`quantile_loss`) for *each* quantile :math:`q`
and corresponding prediction slice, then computes the *average* of these
individual quantile losses.

.. math::
   L_{multi} = \frac{1}{|Q|} \sum_{q \in Q} L_q(y_{true}, \hat{y}_{pred, q})

where :math:`Q` is the set of specified `quantiles`.

**Usage Context:** Intended for training models that output predictions
for multiple quantiles simultaneously. The model's output layer should
typically have a final dimension whose size equals the number of
quantiles. This function provides one way to achieve multi-quantile
training. Ensure the model output shape is compatible.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np
   from fusionlab.nn.losses import quantile_loss_multi

   # Config
   batch_size = 4
   horizon = 6
   output_dim = 1 # Univariate target
   quantiles = [0.1, 0.5, 0.9]
   num_quantiles = len(quantiles)

   # Dummy true values (B, H, O=1)
   y_true = tf.random.normal((batch_size, horizon, output_dim))
   # Dummy predicted quantiles (B, H, Q) - Assuming O=1 is squeezed
   y_pred_multi_q = tf.random.normal((batch_size, horizon, num_quantiles))

   # Create the loss function
   loss_fn_multi = quantile_loss_multi(quantiles=quantiles)

   # Calculate the loss
   loss_value = loss_fn_multi(y_true, y_pred_multi_q)

   print(f"y_true shape: {y_true.shape}")
   print(f"y_pred shape (multi-quantile): {y_pred_multi_q.shape}")
   print(f"Calculated Multi-Quantile Loss: {loss_value.numpy():.4f}")

.. _losses_combined_quantile_loss: 

combined_quantile_loss
~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.losses.combined_quantile_loss`

**Purpose:** Creates a Keras-compatible loss function that
calculates the mean quantile loss (pinball loss) averaged across
multiple specified quantiles. This is the primary recommended loss
function for multi-quantile forecasting in ``fusionlab``.

**Functionality:**
This function takes a list of target `quantiles` (e.g.,
`[0.1, 0.5, 0.9]`) and returns *another function*
`loss_fn(y_true, y_pred)` suitable for Keras. The returned
`loss_fn` performs the following calculation:

1. Calculates the prediction error: :math:`error = y_{true} - y_{pred}`.
   Note that :math:`y_{true}` (shape :math:`(B, H, O)`) is typically
   expanded and broadcasted internally to match the shape of
   :math:`y_{pred}` which includes the quantile dimension
   (e.g., shape :math:`(B, H, Q)` or :math:`(B, H, Q, O)`).
2. For each specified quantile :math:`q` in the `quantiles` list, it
   computes the pinball loss:

   .. math::
      \text{Loss}_q(error) = \max(q \cdot error, (q - 1) \cdot error)

3. It averages the loss across all dimensions (batch B, horizon H,
   quantiles Q, output O if present).

**Usage Context:** This is the standard loss function to use with
`model.compile` when training a model (like TFT or XTFT) that is
configured to output predictions for multiple quantiles. The use of
`@register_keras_serializable` within the factory ensures models
compiled with this loss can often be saved and loaded correctly.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np
   from fusionlab.nn.losses import combined_quantile_loss

   # Config
   batch_size = 4
   horizon = 6
   output_dim = 1 # Univariate target
   quantiles = [0.1, 0.5, 0.9]
   num_quantiles = len(quantiles)

   # Dummy true values (B, H, O=1)
   y_true = tf.random.normal((batch_size, horizon, output_dim))
   # Dummy predicted quantiles (B, H, Q) - Assuming O=1 is squeezed
   y_pred_multi_q = tf.random.normal((batch_size, horizon, num_quantiles))

   # Create the loss function using the factory
   loss_fn_combined = combined_quantile_loss(quantiles=quantiles)

   # Calculate the loss
   loss_value = loss_fn_combined(y_true, y_pred_multi_q)

   print(f"y_true shape: {y_true.shape}")
   print(f"y_pred shape (multi-quantile): {y_pred_multi_q.shape}")
   print(f"Calculated Combined Quantile Loss: {loss_value.numpy():.4f}")

   # Typical compilation:
   # model.compile(optimizer='adam', loss=loss_fn_combined)

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


Anomaly & Combined Loss Functions
-----------------------------------

These functions integrate anomaly detection signals into the training
objective, often combining them with a primary forecasting loss like
the quantile loss. They typically return callable functions suitable
for Keras `compile`.

.. _losses_anomaly_loss: 

anomaly_loss
~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.losses.anomaly_loss`

**Purpose:** Creates a Keras-compatible loss function based on
**fixed**, pre-provided anomaly scores. This allows incorporating an
anomaly penalty into the total loss where the anomaly scores
themselves are static inputs *captured when the loss function is
created*.

**Functionality:**
Takes a tensor of `anomaly_scores` and an `anomaly_loss_weight`
during initialization. It returns a Keras loss function
:math:`loss\_fn(y_{true}, y_{pred})`. Crucially, this returned function
**ignores** :math:`y_{true}` and :math:`y_{pred}` and computes the
loss *only* based on the `anomaly_scores` provided when the loss
function was created:

.. math::
   L_{anomaly} = w_{anomaly} \cdot \text{mean}(\text{anomaly\_scores}^2)

where :math:`w_{anomaly}` is the `anomaly_loss_weight`.

**Usage Context:** This function differs significantly from the
:class:`~fusionlab.nn.components.AnomalyLoss` *layer* (which processes
dynamic scores). This function captures scores *at definition time*.
It might be used in specific scenarios where anomaly scores are fixed
throughout training and treated purely as an additional static penalty
term. Its direct use might be less common than using the `AnomalyLoss`
layer within combined loss strategies like
:func:`~fusionlab.nn.losses.combined_total_loss`.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np
   from fusionlab.nn.losses import anomaly_loss

   # Config
   batch_size = 4
   horizon = 6
   output_dim = 1
   anomaly_weight = 0.1

   # Dummy anomaly scores (fixed for the loss function)
   # Shape needs to be considered carefully based on how mean is taken
   dummy_scores = tf.constant(
       np.random.rand(batch_size, horizon, output_dim) * 0.5,
       dtype=tf.float32
   )

   # Create the loss function, capturing the scores
   loss_fn_anomaly = anomaly_loss(
       anomaly_scores=dummy_scores,
       anomaly_loss_weight=anomaly_weight
   )

   # Dummy y_true/y_pred (ignored by this specific loss function)
   y_true = tf.random.normal((batch_size, horizon, output_dim))
   y_pred = tf.random.normal((batch_size, horizon, output_dim))

   # Calculate the loss (depends only on captured scores and weight)
   loss_value = loss_fn_anomaly(y_true, y_pred)

   print(f"Captured anomaly scores shape: {dummy_scores.shape}")
   print(f"Calculated Anomaly Loss (fixed scores): {loss_value.numpy():.4f}")


.. _losses_prediction_based_loss: 

prediction_based_loss
~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.losses.prediction_based_loss`

**Purpose:** Creates a Keras-compatible loss function specifically
for the `'prediction_based'` anomaly detection strategy used in
:class:`~fusionlab.nn.XTFT`. This strategy defines anomalies based
on the magnitude of prediction errors.

**Functionality:**
This function takes optional `quantiles` and an `anomaly_loss_weight`
and returns a Keras loss function :math:`loss\_fn(y_{true}, y_{pred})`.
The returned :math:`loss\_fn` computes two components internally:

1.  **Prediction Loss (:math:`L_{pred}`):**
    * If `quantiles` are provided: Standard quantile loss based on
      :func:`combined_quantile_loss`.
    * If `quantiles` is `None`: Standard Mean Squared Error (MSE).
      :math:`L_{pred} = \text{mean}((y_{true} - y_{pred})^2)`.
2.  **Anomaly Loss (:math:`L_{anomaly}`):**
    * Calculates prediction error :math:`|y_{true} - y_{pred}|`. If
      predicting quantiles, the error relative to the median
      (or average across quantiles) might be used.
    * Anomaly loss is the mean squared value of these errors:
      :math:`L_{anomaly} = \text{mean}(\text{error}^2)`.
3.  **Total Loss:** Weighted sum:

    .. math::
       L_{total} = L_{pred} + w_{anomaly} \cdot L_{anomaly}

    where :math:`w_{anomaly}` is the `anomaly_loss_weight`.

**Usage Context:** This function should be used to create the loss
for `model.compile` *only* when using the `'prediction_based'`
anomaly detection strategy in :class:`~fusionlab.nn.XTFT`. It allows
the model to simultaneously minimize forecasting error and penalize
large prediction errors (treating them as anomalies).

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np
   from fusionlab.nn.losses import prediction_based_loss

   # Config
   batch_size = 4
   horizon = 6
   output_dim = 1
   quantiles = [0.1, 0.5, 0.9] # Example for quantile mode
   num_quantiles = len(quantiles)
   anomaly_weight = 0.05

   # Create the loss function for quantile + prediction-based anomaly
   loss_fn_pred_based = prediction_based_loss(
       quantiles=quantiles,
       anomaly_loss_weight=anomaly_weight
   )

   # Dummy true values (B, H, O=1)
   y_true = tf.random.normal((batch_size, horizon, output_dim))
   # Dummy predicted quantiles (B, H, Q)
   y_pred_quantiles = tf.random.normal((batch_size, horizon, num_quantiles))

   # Calculate the combined loss
   loss_value = loss_fn_pred_based(y_true, y_pred_quantiles)

   print(f"y_true shape: {y_true.shape}")
   print(f"y_pred shape (multi-quantile): {y_pred_quantiles.shape}")
   print(f"Calculated Prediction-Based Loss: {loss_value.numpy():.4f}")

   # Example for point forecast mode
   loss_fn_point = prediction_based_loss(quantiles=None, anomaly_loss_weight=0.1)
   y_pred_point = tf.random.normal((batch_size, horizon, output_dim))
   loss_value_point = loss_fn_point(y_true, y_pred_point)
   print(f"\nCalculated Prediction-Based Loss (Point): {loss_value_point.numpy():.4f}")

.. _losses_combined_total_loss: 

combined_total_loss
~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.losses.combined_total_loss`

**Purpose:** Creates a Keras-compatible loss function that combines
a standard quantile loss with an anomaly loss derived from
**pre-computed** or **externally provided** anomaly scores captured
at the time of loss creation. This is primarily used for the
`'from_config'` anomaly detection strategy.

**Functionality:**
This function takes the `quantiles` list, an instance of the
:class:`~fusionlab.nn.components.AnomalyLoss` layer (`anomaly_layer`),
and a tensor of fixed `anomaly_scores` as input. It returns a Keras
loss function :math:`loss\_fn(y_{true}, y_{pred})`. The returned
:math:`loss\_fn` computes:

1.  **Quantile Loss (:math:`L_{quantile}`):** Calculated using the internal
    :func:`combined_quantile_loss` logic based on `quantiles`,
    :math:`y_{true}`, and :math:`y_{pred}`.
2.  **Anomaly Loss (:math:`L_{anomaly}`):** Calculated by calling the
    provided `anomaly_layer` with the *fixed* `anomaly_scores` tensor
    that was passed during the creation of this loss function.
    Typically: :math:`L_{anomaly} = w \cdot \text{mean}(\text{anomaly\_scores}^2)`.
3.  **Total Loss:** :math:`L_{total} = L_{quantile} + L_{anomaly}`

**Usage Context:** Used to create the loss for `model.compile` when
using the `'from_config'` anomaly detection strategy in
:class:`~fusionlab.nn.XTFT`. Requires providing the `anomaly_scores`
tensor when *creating* the loss function. *(Note: Aligning these
fixed scores with training batches within `model.fit` can be complex;
using `model.add_loss` in a custom `train_step` might be more robust
for `'from_config'`)*.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np
   # Assuming relevant functions/classes are importable
   from fusionlab.nn.losses import combined_total_loss
   from fusionlab.nn.components import AnomalyLoss

   # Config
   batch_size = 4
   horizon = 6
   output_dim = 1
   quantiles = [0.1, 0.5, 0.9]
   num_quantiles = len(quantiles)
   anomaly_weight = 0.05

   # 1. Instantiate the anomaly loss layer component
   anomaly_loss_layer = AnomalyLoss(weight=anomaly_weight)

   # 2. Provide FIXED anomaly scores (e.g., for training data)
   #    Shape needs careful handling based on loss implementation
   #    Assuming (B, H, O) or compatible shape for demo
   dummy_scores_train = tf.constant(
       np.random.rand(batch_size, horizon, output_dim) * 0.2,
       dtype=tf.float32
   )

   # 3. Create the combined loss function, capturing scores
   loss_fn_total = combined_total_loss(
       quantiles=quantiles,
       anomaly_layer=anomaly_loss_layer,
       anomaly_scores=dummy_scores_train # Pass fixed scores
   )

   # 4. Dummy data for calculation demo
   y_true = tf.random.normal((batch_size, horizon, output_dim))
   y_pred_quantiles = tf.random.normal((batch_size, horizon, num_quantiles))

   # 5. Calculate the loss
   #    Uses y_true/y_pred for quantile part, uses captured scores for anomaly part
   loss_value = loss_fn_total(y_true, y_pred_quantiles)

   print(f"Captured anomaly scores shape: {dummy_scores_train.shape}")
   print(f"y_true shape: {y_true.shape}")
   print(f"y_pred shape: {y_pred_quantiles.shape}")
   print(f"Calculated Combined Total Loss: {loss_value.numpy():.4f}")


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">
   
.. _losses_function_wrappers:
 
Loss Function Wrappers/Factories
-----------------------------------

These functions help in constructing or wrapping loss components for
use with Keras.

.. _losses_objective_loss: 

objective_loss
~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.losses.objective_loss`

**Purpose:** To create a standard Keras-compatible loss function
signature :math:`loss(y_{true}, y_{pred})` from a pre-configured
:class:`~fusionlab.nn.components.MultiObjectiveLoss` layer instance,
potentially incorporating fixed `anomaly_scores` captured at creation time.

**Functionality:**
This function acts as a bridge or factory. It takes an instantiated
`multi_obj_loss` layer (which internally holds other loss layers like
`AdaptiveQuantileLoss` and `AnomalyLoss`) and optional fixed
`anomaly_scores`. It returns a standard Keras loss function
`_loss_fn(y_true, y_pred)`. When Keras calls `_loss_fn`, it internally
invokes the `multi_obj_loss` layer's `call` method, passing along
`y_true`, `y_pred`, and the captured `anomaly_scores` in a way the
`MultiObjectiveLoss` layer expects (requiring careful design of the
`MultiObjectiveLoss.call` signature or data format).

**Usage Context:** Provides a way to package a configured
:class:`~fusionlab.nn.components.MultiObjectiveLoss` layer and potentially
fixed anomaly scores into the standard `loss(y_true, y_pred)` format
expected by `model.compile`. This might be used to simplify compilation
when dealing with multi-task objectives managed by the
`MultiObjectiveLoss` layer, particularly for strategies like `'from_config'`
where scores are fixed.

**Code Example (Instantiation):**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.losses import objective_loss
   from fusionlab.nn.components import (
       MultiObjectiveLoss, AdaptiveQuantileLoss, AnomalyLoss
   )

   # Config
   quantiles = [0.1, 0.5, 0.9]
   anomaly_weight = 0.05
   batch_size, horizon, output_dim = 4, 6, 1

   # 1. Instantiate individual loss components
   quantile_loss_comp = AdaptiveQuantileLoss(quantiles=quantiles)
   anomaly_loss_comp = AnomalyLoss(weight=anomaly_weight)

   # 2. Instantiate the multi-objective loss layer
   multi_loss_layer = MultiObjectiveLoss(
       quantile_loss_fn=quantile_loss_comp,
       anomaly_loss_fn=anomaly_loss_comp
   )

   # 3. Provide FIXED anomaly scores (if needed by multi_loss_layer's logic)
   dummy_scores = tf.constant(
       np.random.rand(batch_size, horizon, output_dim) * 0.2,
       dtype=tf.float32
   )

   # 4. Create the Keras-compatible loss function using the factory
   keras_loss_fn = objective_loss(
       multi_obj_loss=multi_loss_layer,
       anomaly_scores=dummy_scores # Pass scores to be captured by the wrapper
   )

   print("Keras-compatible objective_loss function created.")
   # Now use this in compile:
   # model.compile(optimizer='adam', loss=keras_loss_fn)
   # Note: model.fit needs to provide y_true/y_pred in a format
   # that the internal MultiObjectiveLoss understands.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">
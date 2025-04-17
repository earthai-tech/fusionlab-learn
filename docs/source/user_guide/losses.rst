.. _user_guide_losses:

================
Loss Functions
================

``fusionlab`` provides several custom loss functions tailored for
advanced time series forecasting tasks, particularly those involving
probabilistic (quantile) predictions and integrated anomaly
detection. These functions are designed to be compatible with the
Keras API (`model.compile(loss=...)`).

Understanding these losses is key to training models like
:class:`~fusionlab.nn.TemporalFusionTransformer` and
:class:`~fusionlab.nn.XTFT` effectively, especially when dealing
with uncertainty estimation or anomaly-aware training strategies.

Quantile Loss Functions
-----------------------

These functions are used when the goal is to predict specific
quantiles of the target distribution, enabling probabilistic
forecasts. They typically return a callable function suitable for
Keras.

quantile_loss
~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.losses.quantile_loss`

**Purpose:** Creates a Keras-compatible loss function that computes
the quantile (pinball) loss for a *single*, specified quantile `q`.

**Functionality:**
Takes a single quantile `q` (between 0 and 1) as input and returns
a function `loss_fn(y_true, y_pred)`. This returned function
calculates the pinball loss:

.. math::
   L_q(y_{true}, y_{pred}) = \text{mean}(\max(q \cdot error, (q - 1) \cdot error))

where $error = y_{true} - y_{pred}$. The mean is typically taken
across the batch and any other dimensions except the feature
dimension specified by Keras backend `K.mean(..., axis=-1)`.

**Usage Context:** Useful when you need to train a model to predict
only one specific quantile of the target distribution. Pass the
result of this function to `model.compile`. For example:
`model.compile(loss=quantile_loss(q=0.75))` would train the model
to predict the 75th percentile.

quantile_loss_multi
~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.losses.quantile_loss_multi`

**Purpose:** Creates a Keras-compatible loss function that computes
the *average* quantile (pinball) loss across a *list* of specified
quantiles.

**Functionality:**
Takes a list of `quantiles` as input and returns a function
`loss_fn(y_true, y_pred)`. The returned function calculates the
pinball loss (as defined in :func:`quantile_loss`) for *each*
quantile $q$ in the provided list and then computes the *average*
of these individual quantile losses.

.. math::
   L_{multi} = \frac{1}{|Q|} \sum_{q \in Q} L_q(y_{true}, y_{pred})

where $Q$ is the set of specified `quantiles`.

**Usage Context:** Intended for training models that output predictions
for multiple quantiles simultaneously. The model's output layer should
typically have a dimension corresponding to the number of quantiles.
This function provides an alternative way to achieve multi-quantile
training compared to :func:`combined_quantile_loss`, with potentially
minor differences in internal calculation (e.g., order of averaging).
Ensure the model output shape is compatible with multi-quantile loss
calculation.

combined_quantile_loss
~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.losses.combined_quantile_loss`

**Purpose:** Creates a Keras-compatible loss function that
calculates the mean quantile loss (also known as pinball loss)
averaged across multiple specified quantiles.

**Functionality:**
This function takes a list of target `quantiles` (e.g.,
`[0.1, 0.5, 0.9]`) and returns *another function*
`loss_fn(y_true, y_pred)` suitable for Keras. The returned
`loss_fn` performs the following calculation:

1.  Calculates the prediction error: $error = y_{true} - y_{pred}$.
    Note that $y_{true}$ (shape `(B, H, O)`) is typically expanded
    and broadcasted to match the shape of $y_{pred}$ which includes
    the quantile dimension (shape `(B, H, Q, O)`).
2.  For each specified quantile $q$ in the `quantiles` list, it
    computes the pinball loss:

    .. math::
       \text{Loss}_q(error) = \max(q \cdot error, (q - 1) \cdot error)

3.  It averages the loss for each quantile across the batch (B),
    horizon (H), and output (O) dimensions.
4.  Finally, it averages the losses obtained for all individual
    quantiles.

**Usage Context:** This is the standard loss function to use with
`model.compile` when training a model (like TFT or XTFT) that is
configured to output predictions for multiple quantiles. The use of
`@register_keras_serializable` ensures that models compiled with
this loss can be saved and loaded correctly.


Anomaly & Combined Loss Functions
---------------------------------

These functions integrate anomaly detection signals into the training
objective, often combining them with a primary forecasting loss like
the quantile loss.


anomaly_loss
~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.losses.anomaly_loss`

**Purpose:** Creates a Keras-compatible loss function based on
*fixed*, pre-provided anomaly scores. This allows incorporating an
anomaly penalty into the total loss where the anomaly scores
themselves are static inputs to the loss function.

**Functionality:**
Takes a tensor of `anomaly_scores` and an `anomaly_loss_weight`
during initialization. It returns a Keras loss function
`loss_fn(y_true, y_pred)`. Crucially, this returned function
**ignores** `y_true` and `y_pred` and computes the loss *only*
based on the `anomaly_scores` provided when the loss function was
created:

.. math::
   L_{anomaly} = w_{anomaly} \cdot \text{mean}(\text{anomaly\_scores}^2)

where $w_{anomaly}$ is the `anomaly_loss_weight`.

**Usage Context:** This function differs significantly from the
:class:`~fusionlab.nn.components.AnomalyLoss` *layer*. The layer
takes anomaly scores dynamically during the forward pass, while this
function captures the scores at the time the loss function is
defined. It might be used in specific scenarios where anomaly scores
are fixed throughout training and treated purely as an additional
static penalty term during compilation. Its direct use might be less
common than using the `AnomalyLoss` layer within combined loss
strategies.


prediction_based_loss
~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.losses.prediction_based_loss`

**Purpose:** Creates a Keras-compatible loss function specifically
for the `'prediction_based'` anomaly detection strategy used in
:class:`~fusionlab.nn.XTFT`. This strategy defines anomalies based
on the magnitude of prediction errors.

**Functionality:**
This function takes optional `quantiles` and an `anomaly_loss_weight`
and returns a Keras loss function `loss_fn(y_true, y_pred)`. The
returned `loss_fn` computes two components:

1.  **Prediction Loss ($L_{pred}$):**
    * If `quantiles` are provided, this is the standard quantile loss
      calculated using :func:`combined_quantile_loss`.
    * If `quantiles` is `None`, this is the standard Mean Squared
      Error (MSE): $L_{pred} = \text{mean}((y_{true} - y_{pred})^2)$.
2.  **Anomaly Loss ($L_{anomaly}$):**
    * Calculates the prediction error $|y_{true} - y_{pred}|$. If
      predicting quantiles, the error is averaged across the
      quantile dimension first.
    * The anomaly loss is defined as the mean squared value of these
      (potentially averaged) prediction errors:
      $L_{anomaly} = \text{mean}(\text{error}^2)$.
3.  **Total Loss:** The final loss is a weighted sum:

    .. math::
       L_{total} = L_{pred} + w_{anomaly} \cdot L_{anomaly}

    where $w_{anomaly}$ is the `anomaly_loss_weight`.

**Usage Context:** This function should be used to create the loss
for `model.compile` *only* when using the `'prediction_based'`
anomaly detection strategy in :class:`~fusionlab.nn.XTFT`. It allows
the model to simultaneously minimize forecasting error and penalize
large prediction errors (which are treated as anomalies).

combined_total_loss
~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.losses.combined_total_loss`

**Purpose:** Creates a Keras-compatible loss function that combines
a standard quantile loss with an anomaly loss derived from
*pre-computed* or *externally provided* anomaly scores. This is
primarily used for the `'from_config'` anomaly detection strategy.

**Functionality:**
This function takes the `quantiles` list, an instance of the
:class:`~fusionlab.nn.components.AnomalyLoss` layer (`anomaly_layer`),
and a tensor of `anomaly_scores` as input. It returns a Keras loss
function `loss_fn(y_true, y_pred)`. The returned `loss_fn` computes:

1.  **Quantile Loss ($L_{quantile}$):** Calculated using the internal
    :func:`combined_quantile_loss` function based on the provided
    `quantiles`, $y_{true}$, and $y_{pred}$.
2.  **Anomaly Loss ($L_{anomaly}$):** Calculated by calling the
    provided `anomaly_layer` with the externally supplied
    `anomaly_scores`. Typically, this computes:
    $L_{anomaly} = w \cdot \text{mean}(\text{anomaly\_scores}^2)$,
    where $w$ is the weight configured within the `anomaly_layer`.
3.  **Total Loss:** The final loss is the sum:

    .. math::
       L_{total} = L_{quantile} + L_{anomaly}

**Usage Context:** Used to create the loss for `model.compile` when
using the `'from_config'` anomaly detection strategy in
:class:`~fusionlab.nn.XTFT`. It requires you to provide the anomaly
scores tensor when *creating* the loss function.


Loss Function Wrappers/Factories
--------------------------------

These functions help in constructing or wrapping loss components for
use with Keras.

objective_loss
~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.losses.objective_loss`

**Purpose:** To create a Keras-compatible loss function signature
`loss(y_true, y_pred)` from a pre-configured
:class:`~fusionlab.nn.components.MultiObjectiveLoss` layer instance,
optionally incorporating fixed `anomaly_scores`.

**Functionality:**
This function essentially acts as a bridge. It takes an instantiated
`multi_obj_loss` layer and optional `anomaly_scores`. It returns
a function `_loss_fn(y_true, y_pred)` that, when called by Keras,
internally calls the `multi_obj_loss` layer's `call` method, passing
it `y_true`, `y_pred`, and the `anomaly_scores` (if they were
provided when `objective_loss` was called).

**Usage Context:** Provides a convenient way to package a configured
:class:`~fusionlab.nn.components.MultiObjectiveLoss` layer (which might
combine quantile and anomaly losses internally) and fixed anomaly
scores (e.g., for the `'from_config'` strategy) into the standard
`loss(y_true, y_pred)` format expected by `model.compile`. It ensures
the resulting loss function is serializable by Keras.







.. _transflow_subnet_guide:

===============================================================
TransFlowSubsNet: A Physics-Informed Hybrid Forecasting Model
===============================================================

:API Reference: :class:`~fusionlab.nn.pinn.TransFlowSubsNet`

The ``TransFlowSubsNet`` is an advanced, hybrid model designed to
solve a complex, coupled problem: simultaneously forecasting
**land subsidence** (:math:`s`) and **groundwater levels** (:math:`h`).
It achieves this by integrating a powerful, data-driven forecasting
engine with the fundamental physical laws that govern these phenomena.

This model inherits its core architecture from the
:class:`~fusionlab.nn.models.BaseAttentive` class, giving it the
ability to process rich static, dynamic, and future-known features.
However, its key innovation lies in its custom training loop, which
penalizes predictions that violate the principles of groundwater flow
and aquifer-system consolidation.

The Hybrid Loss Function: Fusing Data and Physics
----------------------------------------------------
``TransFlowSubsNet`` is trained by minimizing a composite loss
function, which is a weighted sum of a data-fidelity term and two
physics-based residual terms. This ensures that the model not only
fits the observed data but also produces physically plausible results.

The total loss, :math:`\mathcal{L}_{total}`, is defined as:

.. math::
   \mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_{gw} \mathcal{L}_{gw} + \lambda_{c} \mathcal{L}_{c}

* **Data Loss (:math:`\mathcal{L}_{data}`):** This is a standard
    supervised loss (e.g., Mean Squared Error) that measures the
    discrepancy between the model's forecasts
    (:math:`\hat{s}, \hat{h}`) and the true, observed data
    (:math:`s_{true}, h_{true}`).

* **Groundwater Flow Loss (:math:`\mathcal{L}_{gw}`):** The mean
    squared residual of the transient groundwater flow equation. This
    term forces the predicted hydraulic head :math:`\hat{h}` to
    conserve mass.

* **Consolidation Loss (:math:`\mathcal{L}_{c}`):** The mean
    squared residual of the consolidation equation. This term forces
    the predicted subsidence rate :math:`\frac{\partial \hat{s}}{\partial t}`
    to be consistent with changes in the hydraulic head field.

The weights :math:`\lambda_{gw}` and :math:`\lambda_{c}` are
hyperparameters set during ``.compile()`` that control the influence
of each physical constraint.

Governing Physical Equations
-------------------------------
The two physics-loss terms are derived from the following PDEs.

**Physics I: Transient Groundwater Flow**
********************************************
This component is governed by a form of the diffusion equation, which
describes how the hydraulic head :math:`h` changes in time and space.
It enforces the conservation of mass within the aquifer.

.. math::
   S_s \frac{\partial h}{\partial t} - K \left( \frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2} \right) - Q = 0

Here, :math:`K` is the hydraulic conductivity, :math:`S_s` is the
specific storage, and :math:`Q` is a source/sink term.

**Physics II: Aquifer-System Consolidation**
***********************************************
This component links the rate of land subsidence (:math:`s`) to
changes in the hydraulic head (:math:`h`), based on Terzaghi's
principle of effective stress. As head decreases, water is squeezed
from the fine-grained layers of the aquifer system, causing them to
compact and the land surface to subside. The model uses a simplified
relationship where the rate of subsidence is proportional to the
divergence of groundwater flow.

.. math::
   \frac{\partial s}{\partial t} - C \left( \frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2} \right) = 0

Here, :math:`C` is a learnable or fixed consolidation coefficient
that encapsulates the mechanical properties of the aquifer.

Architectural Workflow
~~~~~~~~~~~~~~~~~~~~~~~~
``TransFlowSubsNet`` employs a multi-stage process within its
custom ``train_step``.

1.  **Data-Driven Core Prediction:**
    First, the model uses the full power of its inherited
    ``BaseAttentive`` architecture (VSN, Encoder, Attention Stack)
    to process the static, dynamic, and future features. This
    produces initial, purely data-driven predictions,
    :math:`s_{net}` and :math:`h_{net}`.

2.  **Coordinate-Based Corrections:**
    A unique feature of ``TransFlowSubsNet`` is the use of two small,
    independent MLPs that provide learnable, additive corrections based
    on the spatio-temporal coordinates :math:`(t, x, y)`. This allows
    the model to capture localized physical behaviors that the main,
    global network might miss.
    .. math::
       h_{final} = h_{net} + f_{\theta_h}(t, x, y) \\
       s_{final} = s_{net} + f_{\theta_s}(t, x, y)

3.  **Loss Calculation and Optimization:**
    The custom ``train_step`` then orchestrates the final loss
    calculation.
    * It computes the **data loss** by comparing the final predictions
        to the true targets.
    * It uses ``tf.GradientTape`` to compute the required first and
        second-order derivatives of the *corrected* predictions,
        :math:`s_{final}` and :math:`h_{final}`.
    * It calculates the two **physics losses** using these derivatives.
    * Finally, it computes the total weighted loss and applies gradients
        to update all trainable parameters, including the main network,
        the coordinate-correction MLPs, and any learnable physical
        coefficients (:math:`K`, :math:`S_s`, :math:`C`).

Key Configuration Parameters
-------------------------------
In addition to the parameters from ``BaseAttentive``, ``TransFlowSubsNet``
introduces several key arguments for its physics-informed components:

* **`output_subsidence_dim`**, **`output_gwl_dim`**: Specify the
    number of output variables for subsidence and groundwater level,
    respectively.
* **`pde_mode`**: A string (``'both'``, ``'gw_flow'``,
    ``'consolidation'``, or ``'none'``) that controls which physics
    losses are active during training.
* **`K`**, **`Ss`**, **`Q`**, **`pinn_coefficient_C`**: These
    parameters define the physical coefficients in the PDEs. Each can
    be set as a fixed `float` or as a `Learnable` object to be
    inferred by the model.
* **`gw_flow_coeffs`**: A convenience dictionary to set `K`, `Ss`,
    and `Q` at once.

Complete Usage Example
-------------------------
This example shows how to set up and train the ``TransFlowSubsNet`` model.
Note that this model requires both **feature inputs** (for the data-
driven core) and **coordinate inputs** (for the physics loss).

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.pinn import TransFlowSubsNet

   # 1. Define Model Parameters
   BATCH_SIZE = 16
   PAST_STEPS = 12
   HORIZON = 6

   # 2. Prepare Dummy Input Data
   # Feature-based inputs for the BaseAttentive core
   static_features = tf.random.normal([BATCH_SIZE, 3])
   dynamic_features = tf.random.normal([BATCH_SIZE, PAST_STEPS, 8])
   future_features = tf.random.normal([BATCH_SIZE, HORIZON, 4])
   
   # Coordinate inputs for the PINN component (must match horizon)
   coords = tf.random.normal([BATCH_SIZE, HORIZON, 3]) # (t, x, y)

   # The full input is a dictionary
   inputs = {
       "static_features": static_features,
       "dynamic_features": dynamic_features,
       "future_features": future_features,
       "coords": coords,
   }

   # Prepare dummy target data
   true_subsidence = tf.random.normal([BATCH_SIZE, HORIZON, 1])
   true_gwl = tf.random.normal([BATCH_SIZE, HORIZON, 1])
   targets = {
       "subs_pred": true_subsidence,
       "gwl_pred": true_gwl
   }

   # 3. Instantiate the Model
   model = TransFlowSubsNet(
       static_input_dim=3,
       dynamic_input_dim=8,
       future_input_dim=4,
       output_subsidence_dim=1,
       output_gwl_dim=1,
       forecast_horizon=HORIZON,
       max_window_size=PAST_STEPS,
       mode='pihal_like', # Future features only used in decoder
       pde_mode='both',  # Activate both physics losses
       K='learnable',    # Infer hydraulic conductivity
       Ss=1e-5           # Use a fixed specific storage
   )

   # 4. Compile the model with data loss and physics weights
   model.compile(
       optimizer='adam',
       loss={'subs_pred': 'mse', 'gwl_pred': 'mse'}, # Data losses
       lambda_gw=1.0,      # Weight for groundwater physics loss
       lambda_cons=0.5     # Weight for consolidation physics loss
   )

   # 5. Train the model
   print("Starting TransFlowSubsNet training...")
   history = model.fit(inputs, targets, epochs=3, verbose=1)
   print("Training complete.")


**Expected Output:**

.. code-block:: text

   Starting TransFlowSubsNet training...
   Epoch 1/3
   1/1 [==============================] - 37s 37s/step - loss: 5.3391 - gwl_pred_loss: 2.6204 - subs_pred_loss: 2.7187 - total_loss: 5.3454 - data_loss: 5.3391 - consolidation_loss: 0.0127 - gw_flow_loss: 2.2860e-11
   Epoch 2/3
   1/1 [==============================] - 0s 34ms/step - loss: 2.8532 - gwl_pred_loss: 1.4979 - subs_pred_loss: 1.3553 - total_loss: 2.8584 - data_loss: 2.8532 - consolidation_loss: 0.0103 - gw_flow_loss: 4.1065e-07
   Epoch 3/3
   1/1 [==============================] - 0s 20ms/step - loss: 2.1304 - gwl_pred_loss: 1.0753 - subs_pred_loss: 1.0551 - total_loss: 2.1345 - data_loss: 2.1304 - consolidation_loss: 0.0083 - gw_flow_loss: 9.8899e-09
   Training complete.
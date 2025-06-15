.. _transflow_subnet_guide:

===============================================================
TransFlowSubsNet: A Physics-Informed Hybrid Forecasting Model
===============================================================

:API Reference: :class:`~fusionlab.nn.pinn._transflow_subsnet.TransFlowSubsNet`

Forecasting land subsidence is a critical challenge in hydrogeology,
requiring an understanding of both historical data trends and the
complex interplay of underlying physical processes. The
``TransFlowSubsNet`` is the flagship model in ``fusionlab-learn`` for
tackling this problem. It is an advanced, hybrid architecture designed
to simultaneously forecast **land subsidence** (:math:`s`) and
**groundwater levels** (:math:`h`).

The model's power comes from its dual nature. First, it inherits its
core architecture from the powerful and flexible
:class:`~fusionlab.nn.models.BaseAttentive` class. This gives it a
state-of-the-art **data-driven engine** capable of learning complex
temporal patterns from a rich set of static, dynamic past, and known
future features.

Second, its key innovation is a **physics-informed** custom training
loop. This component acts as a strong regularizer, penalizing any
predictions that violate the fundamental physical laws of groundwater
flow and aquifer-system consolidation, ensuring the model's forecasts
are not just accurate, but also physically plausible.

The Hybrid Loss Function: Fusing Data and Physics
----------------------------------------------------
At the heart of ``TransFlowSubsNet`` is its multi-objective, composite
loss function. The model is trained to achieve two goals at once:
**data fidelity** (matching historical observations) and **physical
consistency** (obeying the governing equations). The loss function is
the mathematical tool used to balance these two often-competing
objectives.

The total loss, :math:`\mathcal{L}_{total}`, is a weighted sum of a
data-fidelity term and two physics-based residual terms:

.. math::
   \mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_{gw} \mathcal{L}_{gw} + \lambda_{c} \mathcal{L}_{c}

Each component serves a distinct purpose:

* **Data Loss** (:math:`\mathcal{L}_{data}`)
  This is the supervised learning component of the loss. It answers
  the question: *"How well do my predictions match the measured data?"*
  It is typically a standard metric like Mean Squared Error (for point
  forecasts) or a Quantile Loss (for probabilistic forecasts) that
  measures the discrepancy between the model's forecasts
  (:math:`\hat{s}, \hat{h}`) and the true, observed data
  (:math:`s_{true}, h_{true}`).

* **Groundwater Flow Loss** (:math:`\mathcal{L}_{gw}`)
  This is the fluid dynamics constraint. It answers the question:
  *"Does my predicted change in groundwater level respect the
  conservation of mass?"* It is calculated as the mean squared error
  of the transient groundwater flow PDE residual. By minimizing this
  term, the model is forced to learn a physically realistic hydraulic
  head field.

* **Consolidation Loss** (:math:`\mathcal{L}_{c}`)
  This is the geomechanical constraint. It answers the question:
  *"Is my predicted rate of land subsidence physically consistent with
  how water is moving through the soil?"* It is calculated as the mean
  squared error of the consolidation PDE residual, linking the two
  target variables together through the laws of physics.

The weights :math:`\lambda_{gw}` and :math:`\lambda_{c}` are crucial
hyperparameters set during the ``.compile()`` step. They act as tuning
knobs that control the influence of each physical constraint on the
final solution. Higher values force stricter adherence to physics, which
is useful in noisy or data-scarce scenarios, while lower values allow
the model to fit the training data more closely.

Governing Physical Equations
----------------------------
The two physics-loss terms are derived from the following fundamental
PDEs of hydrogeology and soil mechanics. The model uses automatic
differentiation to calculate the terms of these equations from its own
outputs and enforces them as soft constraints during training.

**Physics I: Transient Groundwater Flow**
*****************************************
This component is governed by the **2D transient groundwater flow
equation**, a cornerstone of hydrogeology derived from Darcy's Law and
the principle of conservation of mass. In simple terms, it states that
the rate at which water accumulates in or is depleted from a point in
the aquifer must equal the net rate of water flowing through it, plus
any water being added or removed.

The equation is expressed as:

.. math::
   S_s \frac{\partial h}{\partial t} - K \left( \frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2} \right) - Q = 0

This can be written more compactly using the **Laplacian operator**,
:math:`\nabla^2`, which represents the divergence of the gradient:

.. math::
   \underbrace{S_s \frac{\partial h}{\partial t}}_{\text{Change in Storage}} - \underbrace{K \nabla^2 h}_{\text{Net Subsurface Flow}} - \underbrace{Q}_{\text{Sources/Sinks}} = 0

Let's break down each term:

* :math:`\frac{\partial h}{\partial t}` **(The Transient Term):**
  This is the rate of change of the **hydraulic head** (groundwater
  level) over time. It describes how quickly the water table is rising
  or falling at a given point.

* :math:`\nabla^2 h` **(The Laplacian Term):**
  The Laplacian, :math:`\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2}`,
  describes the net "curvature" of the water table. A non-zero
  Laplacian indicates a net flow of water. For example, at the center
  of a cone of depression caused by pumping, the Laplacian is large and
  negative, indicating that water is flowing into that point from all
  directions.

* :math:`S_s` **(Specific Storage):**
  This crucial parameter represents the aquifer's elasticity. It is
  defined as the volume of water that a unit volume of the aquifer
  releases from storage per unit decline in hydraulic head. It
  accounts for both the compressibility of water and the compaction
  of the aquifer's porous skeleton.

* :math:`K` **(Hydraulic Conductivity):**
  This parameter measures the ability of the porous medium (sand,
  silt, clay) to transmit water. A high :math:`K` (like in sand)
  allows water to move easily, while a low :math:`K` (like in clay)
  impedes flow.

* :math:`Q` **(Source/Sink Term):**
  This term accounts for any water being artificially added (e.g.,
  via injection wells) or removed (e.g., via pumping wells) from the
  aquifer system.

**Physics II: Aquifer-System Consolidation**
********************************************
This component physically links the two target variables, subsidence
(:math:`s`) and hydraulic head (:math:`h`), based on **Terzaghi's
principle of effective stress**.

The principle states that the total stress on an aquifer is borne by
both the rock/soil skeleton (effective stress) and the water pressure
(pore pressure). When groundwater is extracted, the pore pressure
decreases. To maintain equilibrium, the effective stress on the soil
skeleton must increase. This increased load, particularly on fine-grained
clay and silt layers (aquitards), causes them to slowly compact. The
sum of this compaction over all compressible layers manifests at the
surface as **land subsidence**.

``TransFlowSubsNet`` models this relationship with the equation:

.. math::
   \frac{\partial s}{\partial t} - C \left( \frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2} \right) = 0

Which can be written as:

.. math::
   \underbrace{\frac{\partial s}{\partial t}}_{\text{Rate of Subsidence}} = \underbrace{C \nabla^2 h}_{\text{Rate of Water Volume Change}}

This elegant equation states that the **rate of subsidence** at a point
is directly proportional to the net outflow of water from that point
(represented by the Laplacian of the head).

* :math:`C` (Consolidation Coefficient):
  This is a lumped physical parameter that represents the combined
  geomechanical properties of the aquifer system, including the
  compressibility and thickness of its clay layers. Since this value
  is often unknown and varies spatially, ``TransFlowSubsNet`` is
  designed to treat :math:`C` as a **learnable parameter**, allowing
  it to be discovered from the observational data. This is a key
  feature for performing inverse modeling.

Architectural Workflow: A Deep Dive
------------------------------------
The key innovation of ``TransFlowSubsNet`` is its custom training step,
which seamlessly fuses a powerful data-driven forecasting engine with
the governing laws of physics. The workflow within a single training
step can be broken down into the following stages:

**1. Data-Driven Core Prediction (** ``BaseAttentive`` **Engine)**
********************************************************************
First, the model processes the **feature-based inputs**:
`static_features`, `dynamic_features`, and `future_features`. These
are fed into the inherited :class:`~fusionlab.nn.models.BaseAttentive`
architecture, which executes its full pipeline:

* **Feature Selection:** Optional :class:`~fusionlab.nn.components.VariableSelectionNetwork`
  (VSN) layers learn the importance of each input variable.
* **Temporal Encoding:** A :class:`~fusionlab.nn.components.MultiScaleLSTM`
  or a pure Transformer encoder processes the historical sequence to
  capture temporal dependencies at various resolutions.
* **Attention Fusion:** A stack of attention mechanisms, including
  cross-attention, fuses the historical context with information from
  known future features.

The output of this powerful data-driven core is a set of initial,
purely statistical predictions for subsidence and groundwater level,
which we can denote as :math:`s_{net}` and :math:`h_{net}`. These
represent the model's "best guess" based solely on the patterns and
correlations present in the feature data.

**2. Coordinate-Based Physics Correction**
******************************************
A unique feature of ``TransFlowSubsNet`` is its use of dedicated,
coordinate-based correction networks. While the main `BaseAttentive`
network excels at learning global trends from features, it may
struggle to capture fine-grained physical behaviors that depend purely
on the continuous spatio-temporal coordinates :math:`(t, x, y)`.

To address this, the `coords` tensor from the input data is fed into
two small, independent Multi-Layer Perceptrons (MLPs), :math:`f_{\theta_h}`
and :math:`f_{\theta_s}`. These networks learn a "correction field"—a
localized, additive adjustment that helps the final prediction better
satisfy the physical equations.

The final predictions, which will be used for both the data and
physics loss calculations, are a combination of the data-driven
forecast and the learned physical correction:

.. math::
   h_{final} = h_{net} + \Delta h = h_{net} + f_{\theta_h}(t, x, y) \\
   s_{final} = s_{net} + \Delta s = s_{net} + f_{\theta_s}(t, x, y)

This architecture allows the model to learn a global, feature-based
trend and then add a localized, coordinate-based physical adjustment
on top, combining the strengths of both approaches.

**3. Composite Loss Calculation and Optimization**
**************************************************
This is the final stage where data fidelity and physical consistency
are unified into a single training objective.

* **A. Data Fidelity Loss:** The first component,
  :math:`\mathcal{L}_{data}`, is calculated. The final predictions
  (:math:`s_{final}`, :math:`h_{final}`) are compared against the
  ground-truth targets (:math:`s_{true}`, :math:`h_{true}`) from the
  dataset using a standard loss function (e.g., MSE or Quantile Loss).

* **B. Physics Residual Calculation:** The physics module is now activated.
  Using TensorFlow's ``GradientTape``, the model computes the partial
  derivatives of the **final corrected predictions** (:math:`s_{final}`,
  :math:`h_{final}`) with respect to the coordinate inputs
  (:math:`t, x, y`). These derivatives, such as
  :math:`\frac{\partial s_{final}}{\partial t}` and
  :math:`\frac{\partial^2 h_{final}}{\partial x^2}`, are then plugged
  into the governing equations to calculate the physics residuals,
  :math:`\mathcal{R}_{gw}` and :math:`\mathcal{R}_{c}`, for every
  point in the batch.

* **C. End-to-End Backpropagation:** The total composite loss,
  :math:`\mathcal{L}_{total}`, is assembled from the data and physics
  components. The gradients of this single loss value are then
  calculated with respect to **all** trainable parameters in the
  entire system, including:
  
  * The weights of the main `BaseAttentive` network.
  * The weights of the two coordinate-correction MLPs.
  * Any learnable physical coefficients (:math:`K`, :math:`S_s`, :math:`C`).

The optimizer then applies these gradients in a single step, updating
the entire model to become better at both fitting the data and
respecting the laws of physics simultaneously.

Key Configuration Parameters for Physics
----------------------------------------
While ``TransFlowSubsNet`` inherits its core data-driven configuration
from the :class:`~fusionlab.nn.models.BaseAttentive` class (e.g.,
`architecture_config`), it introduces a dedicated set of parameters
to control its unique physics-informed behavior. These arguments allow
you to define the exact physical problem the model should solve.

**output_subsidence_dim** and **output_gwl_dim**
*************************************************
These integer parameters define the shape of the model's two primary
output heads.

* ``output_subsidence_dim`` (int, default=1)
* ``output_gwl_dim`` (int, default=1)

While typically set to `1` for predicting a single subsidence and a
single GWL value per forecast, they can be set to values greater than
one. This enables the model to predict, for example, subsidence at
multiple surface locations or groundwater levels in multiple aquifer
layers simultaneously for each input sample.

**pde_mode**
************
This crucial string parameter acts as a switch to determine which
physical laws are enforced as "soft constraints" during training. It
gives you precise control over the physics component of the loss
function.

* ``'both'`` (Default): This is the most powerful mode. The model is
  constrained by **both** the groundwater flow and consolidation
  PDEs, creating a fully coupled physical system. This is the
  recommended setting for most applications.
* ``'gw_flow'``: In this mode, only the groundwater flow loss
  (:math:`\mathcal{L}_{gw}`) is active. The model is forced to
  produce a physically plausible hydraulic head field, but the
  subsidence prediction becomes a purely data-driven output,
  uncoupled from the head prediction's physics.
* ``'consolidation'``: Here, only the consolidation loss
  (:math:`\mathcal{L}_{c}`) is active. This forces the relationship
  between the rate of subsidence and the change in head to be
  physically consistent. This mode is similar in scope to the
  legacy ``PIHALNet`` model.
* ``'none'``: This mode **disables all physics losses**. The model
  behaves as a purely data-driven, dual-output forecasting engine,
  similar to ``HALNet``. This is extremely useful for **ablation
  studies** to precisely quantify the performance gain achieved by
  incorporating the physics constraints.

**Defining Physical Coefficients (** ``K``, ``Ss``, ``Q``, ``pinn_coefficient_C`` **)**
******************************************************************************************
These parameters allow you to inject domain knowledge into the model
or, more powerfully, to have the model perform **inverse modeling**
by discovering the parameter values from the data. Each can be
specified in multiple ways:

* **As a** ``float``: Use this when the physical parameter is known and
  should be treated as a fixed constant in the PDE calculation (e.g.,
  ``Ss=1e-5``).
* **As the string** ``'learnable'``: This is the key to parameter
  discovery. It instructs the model to create a trainable variable
  for this coefficient. The model will then learn the optimal value
  for the coefficient by minimizing the total composite loss, effectively
  finding the parameter value that best explains the observed data
  while respecting the laws of physics.
* **As a** ``Learnable`` object: For more control, you can use a
  `Learnable` helper class (e.g., ``K=LearnableK(initial_value=1e-4)``)
  to set the initial guess for a learnable parameter.

**gw_flow_coeffs**
******************
This is a convenience dictionary that allows you to set the `K`, `Ss`,
and `Q` parameters for the groundwater flow equation in a single,
organized place. Values provided in this dictionary will override any
values passed to the individual `K`, `Ss`, or `Q` arguments.

.. code-block:: python

   # Example of using the dictionary
   gw_coeffs = {
       'K': 'learnable', # Discover K
       'Ss': 1e-5,       # Use a fixed value for Ss
       'Q': 0.0          # Assume no sources/sinks
   }

   model = TransFlowSubsNet(
       # ... other parameters
       gw_flow_coeffs=gw_coeffs
   )
   
Complete Usage Example
-------------------------
This example demonstrates the complete, end-to-end workflow for setting
up and training the ``TransFlowSubsNet`` model.

A key takeaway is how the input data is structured. The model requires
a **single dictionary** that contains both the standard feature-based
tensors (`static_features`, `dynamic_features`, `future_features`) for
the data-driven core, and a special `coords` tensor containing the
spatio-temporal coordinates for the physics-loss calculations.

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.pinn import TransFlowSubsNet

   # 1. Define Model & Data Dimensions
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

   # The full input is a dictionary containing all data types
   inputs = {
       "static_features": static_features,
       "dynamic_features": dynamic_features,
       "future_features": future_features,
       "coords": coords,
   }

   # Prepare a dictionary of dummy target data
   true_subsidence = tf.random.normal([BATCH_SIZE, HORIZON, 1])
   true_gwl = tf.random.normal([BATCH_SIZE, HORIZON, 1])
   targets = {
       "subs_pred": true_subsidence,
       "gwl_pred": true_gwl
   }

   # 3. Instantiate the Model
   # We configure both the data-driven aspects (dims, mode) and the
   # physics-informed settings (pde_mode, K, Ss).
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
       K='learnable',    # Ask the model to infer hydraulic conductivity
       Ss=1e-5           # Use a fixed specific storage
   )

   # 4. Compile the model with the composite loss function
   # We specify the data loss for each output and the weights for each
   # physics loss component.
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

The training log clearly shows all the components of the composite loss
being tracked for each epoch, including the total loss, the combined data
loss, and the individual, unweighted physics losses.

.. code-block:: text

   Starting TransFlowSubsNet training...
   Epoch 1/3
   1/1 [==============================] - 37s 37s/step - loss: 5.3391 - gwl_pred_loss: 2.6204 - subs_pred_loss: 2.7187 - total_loss: 5.3454 - data_loss: 5.3391 - consolidation_loss: 0.0127 - gw_flow_loss: 2.2860e-11
   Epoch 2/3
   1/1 [==============================] - 0s 34ms/step - loss: 2.8532 - gwl_pred_loss: 1.4979 - subs_pred_loss: 1.3553 - total_loss: 2.8584 - data_loss: 2.8532 - consolidation_loss: 0.0103 - gw_flow_loss: 4.1065e-07
   Epoch 3/3
   1/1 [==============================] - 0s 20ms/step - loss: 2.1304 - gwl_pred_loss: 1.0753 - subs_pred_loss: 1.0551 - total_loss: 2.1345 - data_loss: 2.1304 - consolidation_loss: 0.0083 - gw_flow_loss: 9.8899e-09
   Training complete.

Next Steps
----------

.. note::

   Now that you are familiar with the architecture and features of
   the ``TransFlowSubsNet`` model, you can put it into practice.

   Proceed to the exercises for a hands-on guide:
   :doc:`../../exercises/exercise_transflow_subnet`
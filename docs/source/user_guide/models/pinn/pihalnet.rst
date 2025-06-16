.. _pihalnet_guide:

=======================================================
Hybrid Physics-Data Models: PiHALNet & PIHALNet
=======================================================

This guide delves into the ``PiHALNet`` family of models, a suite
of sophisticated hybrid architectures designed for complex, coupled
geophysical forecasting. These models uniquely combine a powerful,
data-driven deep learning engine with the physical laws of soil
mechanics and groundwater flow.

The primary goal of these models is to produce forecasts for land
subsidence and groundwater levels that are not only accurate with
respect to observational data but are also physically consistent.
This page details the two evolutionary versions of this concept,
highlighting their shared principles and key differences.


Common Key Features
-------------------
Both versions of ``PiHALNet`` are built upon the same core
principles, offering a powerful set of common features that make
them uniquely suited for complex geophysical forecasting tasks.

* **Hybrid Data-Physics Architecture**
  The models integrate a powerful data-driven forecasting core
  (the Hybrid Attentive LSTM Network, or HALNet, engine) with a
  physics-informed module. This means they learn from both
  observational data and the governing physical laws, with the PDE
  residual being a key component of the loss function.

* **Coupled Multi-Target Prediction**
  They are designed to simultaneously forecast multiple, physically
  linked variables. The primary use case is predicting land
  subsidence (:math:`s`) and groundwater levels (:math:`h`) in a
  coupled manner.

* **Advanced Input Handling**
  The architecture natively processes three distinct types of time
  series inputs:
  
  * **Static features:** Time-invariant metadata (e.g., sensor
    location, soil type).
  * **Dynamic past features:** Time-varying data observed up to the
    present (e.g., historical rainfall, past measurements).
  * **Known future features:** Time-varying data known in advance
    (e.g., day of the week, scheduled pumping).
  This is enhanced by an optional
  :class:`~fusionlab.nn.components.VariableSelectionNetwork` (VSN)
  for intelligent, learnable feature selection.

* **Sophisticated Temporal Processing**
  To capture complex time-dependencies, the models employ:
  
  * A :class:`~fusionlab.nn.components.MultiScaleLSTM` that processes
    the input sequence at various user-defined temporal
    resolutions, capturing both short-term and long-term patterns.
  * A rich suite of attention mechanisms that work together to fuse
    information from all sources, including
    :class:`~fusionlab.nn.components.CrossAttention` (for
    encoder-decoder interaction) and various self-attention layers
    ike :class:`~fusionlab.nn.components.HierarchicalAttention`.

* **Flexible Physics-Informed Constraints**
  The physics module is highly configurable:
    
  * The specific physical law to enforce can be selected via the
    ``pde_mode`` parameter, with the primary focus being on
    ``'consolidation'``.
  * Key physical coefficients in the PDEs (like the consolidation
    coefficient :math:`C` or hydraulic conductivity :math:`K`) can
    be either **fixed** as known constants or made **learnable**.
    This allows the model to perform *parameter inversion*—
    discovering the values of physical constants directly from the
    observational data.

* **Probabilistic Forecasting for Uncertainty**
  The models can produce probabilistic forecasts to quantify
  prediction uncertainty. By specifying a list of ``quantiles``, the
  :class:`~fusionlab.nn.components.QuantileDistributionModeling` head
  is activated, generating prediction intervals alongside the point
  forecast.

* **Multi-Horizon Output Structure**
  Using a :class:`~fusionlab.nn.components.MultiDecoder`, the models
  generate predictions for each step in the forecast horizon in a
  sequence-to-sequence manner, making them true multi-step-ahead
  forecasters.
    

Physical Formulation and Hybrid Loss
---------------------------------------
The power of the ``PIHALNet`` family lies in its **hybrid formulation**,
which forces the data-driven predictions to conform to physical laws.
This is achieved by integrating the governing equations of groundwater
hydrology and soil mechanics directly into the model's training
objective.

Governing Equations
~~~~~~~~~~~~~~~~~~~
The models are designed to understand and simulate two coupled
physical processes. The specific equations activated during training
depend on the ``pde_mode`` setting.

**1. Transient Groundwater Flow**

This equation, a form of the diffusion equation, enforces the
conservation of mass for groundwater moving through a porous medium. It
describes how the hydraulic head :math:`h` changes in time and space.
The 2D residual form used by the model is:

.. math::
   \mathcal{R}_{gw} = S_s \frac{\partial h}{\partial t} - K \left( \frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2} \right) - Q

Here, :math:`K` is the hydraulic conductivity, :math:`S_s` is the
specific storage, and :math:`Q` is a source/sink term.

**2. Aquifer-System Consolidation**

This principle links the rate of land subsidence (:math:`s`) to
changes in the hydraulic head field. As the head :math:`h` declines,
pressure within the aquifer system changes, causing fine-grained
clay layers to compact, which results in subsidence at the surface.
The residual form of this relationship is:

.. math::
   \mathcal{R}_{c} = \frac{\partial s}{\partial t} - C \left( \frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2} \right)

Here, :math:`C` is the consolidation coefficient, a parameter that
encapsulates the mechanical properties of the aquifer system.

Operational Workflow: From Data to Physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The model's custom ``train_step`` seamlessly integrates the data-driven
and physics-informed components in a three-stage process:

**1. Data-Driven Prediction**

First, the model acts as a powerful data-driven forecaster. It uses
its internal ``BaseAttentive`` engine to process the rich set of
static, dynamic, and future features. This stage produces initial,
highly accurate "mean" predictions for the target variables, denoted
as :math:`\bar{s}_{net}` and :math:`\bar{h}_{net}`. These predictions
are used to calculate the data-fidelity portion of the loss.

**2. Physics Residual Calculation**

Next, the physics module is activated. The model takes the mean
predictions (:math:`\bar{s}_{net}`, :math:`\bar{h}_{net}`) and their
corresponding spatio-temporal coordinates (:math:`t, x, y`). Using
TensorFlow's ``GradientTape`` for automatic differentiation, it
computes all the necessary derivatives (e.g.,
:math:`\frac{\partial \bar{s}_{net}}{\partial t}`,
:math:`\frac{\partial^2 \bar{h}_{net}}{\partial x^2}`). These derivatives
are then plugged into the governing equations to calculate the
physics residuals, :math:`\mathcal{R}_{gw}` and/or :math:`\mathcal{R}_{c}`.

**3. Composite Loss Function**

Finally, the total loss function, :math:`\mathcal{L}_{total}`, is
assembled as a weighted sum of the data and physics components.

.. math::
   \mathcal{L}_{total} = \mathcal{L}_{data} + \sum_{i \in \{gw, c\}} \lambda_{i} \mathcal{L}_{physics, i}

* :math:`\mathcal{L}_{data}`: This is the supervised loss (e.g.,
  Mean Squared Error or a Quantile Loss) calculated between the
  model's final forecast and the true observational data.
* :math:`\mathcal{L}_{physics, i}`: This is the Mean Squared Error
  of a specific PDE residual (e.g., :math:`\text{mean}(\mathcal{R}_c^2)`).
  It quantifies how much the predictions violate that physical law.
* :math:`\lambda_{i}`: These are user-defined hyperparameters
  (e.g., ``lambda_gw``, ``lambda_cons``) passed to ``.compile()`` that
  control the influence of each physical constraint on the total loss.

This composite loss is then used to update all trainable parameters in
the model, ensuring that the network learns to be accurate to both the
data and the underlying physics simultaneously.


Architectural & Feature Differences
------------------------------------------
While both models in the ``PIHALNet`` family aim to solve the same
problem, they represent a significant evolution in software design
and capability. Understanding their differences is key to leveraging
the full power of the library.

The Legacy `PiHALNet`
~~~~~~~~~~~~~~~~~~~~~
The original ``PiHALNet`` is a monolithic, self-contained class that
inherits directly from ``tf.keras.Model``. Its data-driven components,
such as the LSTMs and attention layers, were implemented specifically
for its own use case. While effective, this design meant that the
architecture was relatively rigid. Configuration was handled via a
long list of parameters in the ``__init__`` method, and the sequence
of internal operations (like the application of attention) was largely
fixed.

The Modern `PIHALNet` (BaseAttentive-based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The modern ``PIHALNet`` represents a paradigm shift towards
modularity and flexibility.

* **Inheritance from BaseAttentive:** Its most important feature is
  that it inherits from the :class:`~fusionlab.nn.models.BaseAttentive`
  class. It does not reinvent the data-driven forecasting engine;
  instead, it inherits a powerful, tested, and highly configurable
  one. This means any improvements to ``BaseAttentive`` are
  immediately available to ``PIHALNet``.

* **Smart Configuration:** Architectural choices are no longer
  controlled by numerous, disconnected parameters. Instead, they
  are defined in a single, clean ``architecture_config``
  dictionary. This allows for clear and explicit control over key
  components like the ``encoder_type`` ('hybrid' vs. 'transformer')
  or ``feature_processing`` ('vsn' vs. 'dense'). This makes
  experimenting with different architectures trivial.

* **Modular Attention Stack:** The sequence of attention mechanisms
  in the decoder is no longer hardcoded. It is now controlled by the
  ``decoder_attention_stack`` key in the configuration dictionary,
  allowing the user to easily add, remove, or reorder attention
  layers (e.g., `['cross', 'hierarchical']`) to tailor the model
  to a specific problem.

In essence, the modern design separates the **"what"** (the physics
of subsidence and groundwater flow, handled by ``PIHALNet``) from the
**"how"** (the data-driven sequence processing, handled by
``BaseAttentive``).

**Comparison Summary**

.. list-table:: Comparison of PiHALNet Model Versions
   :widths: 20 40 40
   :header-rows: 1

   * - Feature
     - `PiHALNet` (Legacy)
     - `PIHALNet` (Modern, `BaseAttentive`-based)
   * - **Base Class**
     - Inherits directly from `tf.keras.Model`.
     - Inherits from the powerful and modular
       :class:`~fusionlab.nn.models.BaseAttentive` class.
   * - **Core Architecture**
     - Data-driven components are implemented internally and are
       specific to this class.
     - Leverages the full, tested, and highly-configurable
       `BaseAttentive` engine.
   * - **Configuration**
     - Primarily configured via a long list of individual ``__init__``
       parameters.
     - Uses the modern ``architecture_config`` dictionary for clear,
       flexible control over internal structure.
   * - **Attention Mechanism**
     - The sequence of attention layers is largely hardcoded within the
       `call` method.
     - The decoder's attention stack is fully configurable via the
       ``decoder_attention_stack`` key in the config.
   * - **Feature Selection**
     - Control over VSNs is a simple boolean flag (`use_vsn`).
     - Controlled via the ``feature_processing`` key, allowing easy
       switching between `'vsn'` and `'dense'`.
       
For all new projects, the modern, ``BaseAttentive``-based **PIHALNet**
is the recommended choice due to its modularity,
configurability, and alignment with the latest architectural patterns
in the library. The legacy version is maintained for backward
compatibility.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


PIHALNet (Modern, BaseAttentive-based)
-----------------------------------------
:API Reference: :class:`~fusionlab.nn.pinn.models.PIHALNet`

The modern ``PIHALNet`` is a powerful and flexible implementation built
upon the modular :class:`~fusionlab.nn.models.BaseAttentive`
architecture. It combines a state-of-the-art data-driven forecasting
engine with physics-based regularization, making it the recommended
choice for all new projects.

This version inherits all the advanced features of its parent class,
including the smart configuration system, which allows for precise
control over the model's internal structure.

Usage Example: Standard Hybrid Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example demonstrates a typical use case for ``PIHALNet``, where
we use the default hybrid architecture (LSTM + Attention) and configure
it to learn the physical consolidation coefficient :math:`C` from data.

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.pinn import PIHALNet
   from fusionlab.params import LearnableC

   # 1. Define Model & Data Dimensions
   BATCH_SIZE = 16
   PAST_STEPS = 10
   HORIZON = 5
   STATIC_DIM, DYNAMIC_DIM, FUTURE_DIM = 4, 6, 3

   # 2. Prepare Dummy Input Data
   # Feature-based inputs for the data-driven core
   static_features = tf.random.normal([BATCH_SIZE, STATIC_DIM])
   dynamic_features = tf.random.normal([BATCH_SIZE, PAST_STEPS, DYNAMIC_DIM])
   # For 'pihal_like' mode, future features span the horizon
   future_features = tf.random.normal([BATCH_SIZE, HORIZON, FUTURE_DIM])

   # Coordinate inputs for the PINN module
   coords = tf.random.normal([BATCH_SIZE, HORIZON, 3]) # (t, x, y)

   # Assemble the full input dictionary
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
   model = PIHALNet(
       static_input_dim=STATIC_DIM,
       dynamic_input_dim=DYNAMIC_DIM,
       future_input_dim=FUTURE_DIM,
       output_subsidence_dim=1,
       output_gwl_dim=1,
       forecast_horizon=HORIZON,
       max_window_size=PAST_STEPS,
       mode='pihal_like',
       # Ask the model to discover the consolidation coefficient
       pinn_coefficient_C=LearnableC(initial_value=0.01),
   )

   # 4. Compile the model with data losses and a physics weight
   model.compile(
       optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
       loss={'subs_pred': 'mse', 'gwl_pred': 'mse'},
       lambda_physics=0.1 # Weight for the consolidation loss
   )

   # 5. Display the model summary
   model.summary(line_length=110)

Advanced Configuration Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example demonstrates the power and flexibility of the smart
configuration system. We will create a `PIHALNet` variant that uses a
pure **transformer** encoder and a simplified attention stack in the
decoder, showcasing how easily the internal architecture can be modified.

.. code-block:: python
   :linenos:

   # 1. Define a custom architecture using the config dictionary
   transformer_pinn_config = {
       'encoder_type': 'transformer',
       'decoder_attention_stack': ['cross', 'hierarchical'], # Simpler stack
       'feature_processing': 'dense' # Use dense layers instead of VSN
   }

   # 2. Instantiate the model with the custom architecture
   tfmr_pinn_model = PIHALNet(
       static_input_dim=STATIC_DIM,
       dynamic_input_dim=DYNAMIC_DIM,
       future_input_dim=FUTURE_DIM,
       output_subsidence_dim=1,
       output_gwl_dim=1,
       forecast_horizon=HORIZON,
       max_window_size=PAST_STEPS,
       mode='pihal_like',
       pinn_coefficient_C=0.05, # Use a fixed physical constant
       architecture_config=transformer_pinn_config # Pass the config
   )

   # 3. Compile the model as before
   tfmr_pinn_model.compile(
       optimizer='adam',
       loss='mae', # Use a different data loss
       lambda_physics=0.2
   )

   # 4. Train for a single step to demonstrate it works
   print("\nTraining a Transformer-based PIHALNet for one step...")
   history = tfmr_pinn_model.fit(
       inputs, targets, epochs=1, verbose=1
   )
   print("\nTraining step complete.")


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


PiHALNet (Legacy Version)
---------------------------
:API Reference: :class:`~fusionlab.nn.pinn.models.legacy.PiHALNet`

This section documents the original, legacy version of ``PiHALNet``. It
is maintained primarily for backward compatibility. For all new
projects, using the modern, :class:`~fusionlab.nn.pinn.models.PIHALNet`
(which inherits from ``BaseAttentive``) is strongly recommended due to
its superior flexibility and modularity.

The legacy ``PiHALNet`` is a self-contained, monolithic class that
implements its data-driven components (LSTMs, attention) internally.
Its architecture is configured via a long list of individual parameters
in its constructor, making it less flexible than the modern version's
smart configuration system.

Usage Example
~~~~~~~~~~~~~~~
The instantiation and compilation process is similar to the modern
version, but it relies on direct keyword arguments like ``objective``
and ``attention_levels`` instead of the ``architecture_config``
dictionary.

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.pinn.models.legacy import PiHALNet

   # 1. Define Model & Data Dimensions
   BATCH_SIZE = 16
   PAST_STEPS = 10
   HORIZON = 5
   STATIC_DIM, DYNAMIC_DIM, FUTURE_DIM = 4, 6, 3

   # 2. Prepare Dummy Input Data (same as modern version)
   inputs = {
       "static_features": tf.random.normal([BATCH_SIZE, STATIC_DIM]),
       "dynamic_features": tf.random.normal([BATCH_SIZE, PAST_STEPS, DYNAMIC_DIM]),
       "future_features": tf.random.normal([BATCH_SIZE, HORIZON, FUTURE_DIM]),
       "coords": tf.random.normal([BATCH_SIZE, HORIZON, 3]),
   }
   targets = {
       "subs_pred": tf.random.normal([BATCH_SIZE, HORIZON, 1]),
       "gwl_pred": tf.random.normal([BATCH_SIZE, HORIZON, 1])
   }

   # 3. Instantiate the Legacy Model
   # Note the direct use of parameters like `objective`
   legacy_model = PiHALNet(
       static_input_dim=STATIC_DIM,
       dynamic_input_dim=DYNAMIC_DIM,
       future_input_dim=FUTURE_DIM,
       output_subsidence_dim=1,
       output_gwl_dim=1,
       forecast_horizon=HORIZON,
       max_window_size=PAST_STEPS,
       objective='hybrid', # Configured directly
       pinn_coefficient_C='learnable'
   )

   # 4. Compile and train as usual
   legacy_model.compile(
       optimizer='adam',
       loss='mse',
       lambda_physics=0.1
   )
   print("Successfully instantiated and compiled the legacy PiHALNet model.")

Next Steps
------------

.. note::

   Now that you are familiar with the architecture and features of
   the ``PIHALNet`` models, you can put them into practice.

   Proceed to the exercises for a hands-on guide:
   :doc:`../../exercises/exercise_pihalnet`
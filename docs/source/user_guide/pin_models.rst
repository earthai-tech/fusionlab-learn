.. _pinn_models_guide:

=========================================
Physics-Informed Neural Networks (PINNs)
===========================================

This section of the user guide delves into the Physics-Informed Neural Networks (PINNs)
available within the ``fusionlab`` library. These models uniquely combine
data-driven deep learning architectures with physical laws, expressed as
Partial Differential Equations (PDEs), to produce forecasts that are not
only accurate but also physically consistent.

The primary PINN model currently featured is PIHALNet, designed for complex
spatio-temporal forecasting tasks like land subsidence, where understanding
and respecting the underlying geohydrological processes is crucial.

.. toctree::
   :hidden:

PIHALNet (Physics-Informed Hybrid Attentive LSTM Network)
-------------------------------------------------------------
:API Reference: :class:`~fusionlab.nn.pinn.models.PIHALNet`

The ``PIHALNet`` model is a sophisticated hybrid architecture tailored for
multi-horizon probabilistic forecasting of coupled geophysical phenomena,
such as land subsidence and groundwater level changes. It leverages the
strengths of deep learning for pattern recognition from data while
constraining its predictions with physical knowledge.

**Key Features:**

* **Hybrid Architecture:** Integrates a data-driven forecasting core (HALNet - Hybrid Attentive LSTM Network) with a physics-informed module that incorporates PDE residuals into the loss function.
* **Dual-Target Prediction:** Simultaneously predicts multiple related variables (e.g., subsidence and groundwater levels).
* **Input Handling:** Accepts static, dynamic (past observed), and future known inputs. It can utilize:
    * :class:`~fusionlab.nn.components.VariableSelectionNetwork` (VSN) for intelligent feature selection and embedding of different input types.
    * If VSNs are not used, :class:`~fusionlab.nn.components.MultiModalEmbedding` can process raw dynamic and future inputs.
* **Advanced Temporal Processing:**
    * :class:`~fusionlab.nn.components.MultiScaleLSTM` captures temporal dependencies at various user-defined scales.
    * A suite of attention mechanisms enhances contextual understanding:
        * :class:`~fusionlab.nn.components.HierarchicalAttention`
        * :class:`~fusionlab.nn.components.CrossAttention`
        * :class:`~fusionlab.nn.components.MemoryAugmentedAttention`
        * :class:`~fusionlab.nn.components.MultiResolutionAttentionFusion`
* **Physics-Informed Constraints:**
    * Supports different PDE modes (via ``pde_mode``), such as 'consolidation' (Terzaghi's theory), 'gw_flow' (groundwater flow equation), or 'both' (coupled).
    * Physical coefficients (e.g., consolidation coefficient `C`, hydraulic conductivity `K`, specific storage `Ss`) can be:
        * **Learnable:** Treated as trainable parameters, allowing the model to discover them from data (default for some).
        * **Fixed:** Specified as known constants.
    * The PDE residual is computed based on model outputs and incorporated into the total loss function, weighted by ``lambda_pde``.
* **Probabilistic Forecasting:** Employs :class:`~fusionlab.nn.components.QuantileDistributionModeling` to output forecasts for specified ``quantiles``, enabling uncertainty estimation. Point forecasts are produced if ``quantiles`` is ``None``.
* **Flexible Output Structure:** Uses a :class:`~fusionlab.nn.components.MultiDecoder` to generate horizon-specific predictions.

**When to Use PIHALNet:**

``PIHALNet`` is particularly well-suited for:

* Forecasting spatio-temporal phenomena governed by known (or partially known) physical laws.
* Problems where data might be sparse or noisy, and incorporating physical constraints can improve model robustness and generalization.
* Situations requiring predictions for coupled physical processes (e.g., subsidence and groundwater).
* Tasks where discovering or refining physical parameters from observational data is of interest.
* Generating probabilistic forecasts to quantify prediction uncertainty.

Formulation
~~~~~~~~~~~~~

PIHALNet's operation involves two main conceptual parts:

1.  **Data-Driven Forecasting (HALNet Core):**

    This part processes the input features (static, dynamic, future) to produce initial forecasts for the target variables (e.g., subsidence :math:`s` and groundwater level :math:`h`).
    * **Input Processing:** Inputs are optionally processed by Variable Selection Networks (VSNs) and Gated Residual Networks (GRNs) or by a MultiModalEmbedding layer. Positional encoding is added.
    * **Temporal Encoding:** The :class:`~fusionlab.nn.components.MultiScaleLSTM` processes dynamic features.
    * **Attention Mechanisms:** A series of attention layers (Hierarchical, Cross, Memory-Augmented, Multi-Resolution Fusion) refine and integrate features from different sources and contexts.
    * **Decoding:** The :class:`~fusionlab.nn.components.MultiDecoder` generates horizon-specific outputs.
    * **Output Layer:** :class:`~fusionlab.nn.components.QuantileDistributionModeling` produces the final data-driven predictions (:math:`\hat{s}_{data}, \hat{h}_{data}`), potentially across multiple quantiles. The mean of these predictions (:math:`\bar{s}_{data}, \bar{h}_{data}`) is also available for PDE calculation.

2.  **Physics-Informed Module:**

    * **PDE Residual Calculation:** The mean predictions from the data-driven core (:math:`\bar{s}_{data}, \bar{h}_{data}`) and the input coordinates (:math:`t, x, y`) are used to compute the residual of the specified PDE(s).
        * For **consolidation** (Terzaghi's 1D theory, often applied vertically):
            .. math::
                \mathcal{R}_{cons} = \frac{\partial \bar{s}_{data}}{\partial t} - C \frac{\partial^2 \bar{s}_{data}}{\partial z^2}
            (Note: PIHALNet typically uses :math:`\bar{h}_{data}` as a proxy for pore pressure in the consolidation equation if coupled, or directly predicts :math:`s` and uses its derivatives if only subsidence PDE is active). The exact formulation for the consolidation residual within PIHALNet, especially how :math:`s` and :math:`h` are coupled, would be detailed in its internal `compute_consolidation_residual` function. A common simplified form might involve relating :math:`\frac{\partial h}{\partial t}` to :math:`C \nabla^2 h` or a similar expression involving subsidence :math:`s`.
            The provided `PIHALNet.call` method implies a form like:
            .. math::
                \mathcal{R}_{cons} = \frac{\partial \bar{s}_{data}}{\partial t} - C \cdot f(\bar{s}_{data}, \bar{h}_{data}, \nabla^2 \bar{s}_{data}, \nabla^2 \bar{h}_{data}, ...)
            For the provided example in `PIHALNet.call` which uses `compute_consolidation_residual(s_pred, h_pred, time_steps, C)`, the typical 1D Terzaghi consolidation related to head change is:
            .. math::
                 \frac{\partial h}{\partial t} = C_v \frac{\partial^2 h}{\partial z^2}
            So, the residual would be :math:`\mathcal{R}_{cv} = \frac{\partial \bar{h}_{data}}{\partial t} - C_v \frac{\partial^2 \bar{h}_{data}}{\partial z^2}`. If subsidence `s` is directly related to `h` (e.g. :math:`s = \alpha h`), then derivatives of `s` can be used. PIHALNet aims to be flexible.
        * For **groundwater flow** (2D horizontal, isotropic):
            .. math::
                \mathcal{R}_{gw} = S_s \frac{\partial \bar{h}_{data}}{\partial t} - K \left( \frac{\partial^2 \bar{h}_{data}}{\partial x^2} + \frac{\partial^2 \bar{h}_{data}}{\partial y^2} \right) + Q
            This residual would be computed by a component like :class:`~fusionlab.nn.pinn.base.GroundwaterFlowPDEResidual`.
    * The model stores these residuals as :math:`\text{pde_residual}` in its output dictionary.

3.  **Loss Function:**
    The total loss function during training is a weighted sum of the data fidelity loss (:math:`\mathcal{L}_{data}`) and the physics loss (:math:`\mathcal{L}_{physics}`):
    .. math::
        \mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_{PDE} \mathcal{L}_{physics}
    * :math:`\mathcal{L}_{data}`: Calculated using the specified loss functions (e.g., Mean Squared Error or a quantile loss like :func:`~fusionlab.nn.losses.combined_quantile_loss`) between the model's final predictions (:math:`\hat{s}_{data}, \hat{h}_{data}`) and the true target values.
    * :math:`\mathcal{L}_{physics}`: Typically the mean squared error of the :math:`\text{pde_residual}` (e.g., :math:`\frac{1}{N} \sum (\mathcal{R})^2`), encouraging the model to satisfy the PDE.
    * :math:`\lambda_{PDE}`: A hyperparameter controlling the influence of the physics-based loss term.

**Code Example (Instantiation):**

.. code-block:: python
   :linenos:

   import numpy as np
   import tensorflow as tf
   from fusionlab.nn.pinn.models import PIHALNet

   # Example Configuration (dimensions must match prepared data)
   static_dim, dynamic_dim, future_dim = 5, 4, 2
   horizon = 3
   time_steps = 7 # Corresponds to max_window_size

   my_quantiles = [0.1, 0.5, 0.9]
   # Or my_quantiles = None for point predictions

   # These fixed parameters are typically inferred from data by PIHALTuner
   # or set manually if using PIHALNet directly.
   fixed_params = {
       "static_input_dim": static_dim,
       "dynamic_input_dim": dynamic_dim,
       "future_input_dim": future_dim,
       "output_subsidence_dim": 1,
       "output_gwl_dim": 1,
       "forecast_horizon": horizon,
       "quantiles": my_quantiles,
       "max_window_size": time_steps, # Important for internal consistency
       "pde_mode": "consolidation", # or 'gw_flow', 'both', 'none'
       "pinn_coefficient_C": "learnable", # For consolidation
       "gw_flow_coeffs": { # For 'gw_flow' or 'both' modes
           'K': 'learnable', # Hydraulic conductivity
           'Ss': 1e-5,       # Specific storage (fixed)
           'Q': 0.0          # Source/sink term (fixed)
       }
   }

   # Architectural hyperparameters (these would be tuned by PIHALTuner)
   arch_params = {
       "embed_dim": 64,
       "hidden_units": 64,
       "lstm_units": 64,
       "attention_units": 32,
       "num_heads": 4,
       "dropout_rate": 0.1,
       "vsn_units": 32,
       "use_vsn": True,
       "scales": [1, 2], # For MultiScaleLSTM
       "memory_size": 50
   }

   # Instantiate PIHALNet
   pihalnet_model = PIHALNet(**fixed_params, **arch_params)

   # Example dummy input data (shapes must match expected sequence format)
   batch_size = 2
   dummy_inputs = {
       'coords': tf.random.normal((batch_size, horizon, 3)), # (t,x,y) for horizon
       'static_features': tf.random.normal((batch_size, static_dim)) if static_dim > 0 else tf.zeros((batch_size, 0)),
       'dynamic_features': tf.random.normal((batch_size, time_steps, dynamic_dim)),
       'future_features': tf.random.normal((batch_size, horizon, future_dim)) if future_dim > 0 else tf.zeros((batch_size, horizon, 0)),
   }

   # Build the model by calling it (or use model.build())
   # model_outputs = pihalnet_model(dummy_inputs)

   # Compile the model (example)
   # Loss dictionary keys MUST match output keys of PIHALNet's call method
   # which are 'subs_pred' and 'gwl_pred' for data loss part.
   # The 'pde_residual' is handled in train_step.
   # from fusionlab.nn.losses import combined_quantile_loss # if using
   from tensorflow.keras.losses import MeanSquaredError
   from tensorflow.keras.optimizers import Adam

   loss_fns = {}
   if my_quantiles:
       # loss_fns['subs_pred'] = combined_quantile_loss(my_quantiles)
       # loss_fns['gwl_pred'] = combined_quantile_loss(my_quantiles)
       pass # Assuming combined_quantile_loss is defined elsewhere
   else:
       loss_fns['subs_pred'] = MeanSquaredError(name="subs_mse")
       loss_fns['gwl_pred'] = MeanSquaredError(name="gwl_mse")
    
   # Ensure loss_fns is populated if quantiles are used
   if not loss_fns: # Fallback if combined_quantile_loss placeholder not active
        loss_fns['subs_pred'] = MeanSquaredError(name="subs_mse_fallback")
        loss_fns['gwl_pred'] = MeanSquaredError(name="gwl_mse_fallback")


   pihalnet_model.compile(
       optimizer=Adam(learning_rate=1e-3),
       loss=loss_fns,
       metrics={'subs_pred': ['mae'], 'gwl_pred': ['mae']},
       lambda_pde=0.1 # Weight for the physics loss component
   )

   pihalnet_model.summary(line_length=110)


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


PINN Utilities
--------------

These are base classes and utilities that support the construction and
operation of Physics-Informed Neural Networks within ``fusionlab``.

GWResidualCalculator
~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.pinn.base.GWResidualCalculator`

The ``GWResidualCalculator`` is a helper class designed to manage the
physical coefficients required for groundwater flow PDE (Partial Differential
Equation) calculations. It is not a Keras layer itself but is intended to be
used within Keras layers or models (like the
:class:`~fusionlab.nn.pinn.base.GroundwaterFlowPDEResidual` layer)
that need to access these coefficients.

**Key Features:**

* **Coefficient Management:** Handles the initialization and retrieval of key
    groundwater flow parameters:
    * **K (Hydraulic Conductivity):** Can be learnable or fixed.
    * **Ss (Specific Storage):** Can be learnable or fixed.
    * **Q (Source/Sink Term):** Can be learnable or fixed.
* **Learnable Parameters:** If a coefficient is set to ``'learnable'``,
    the class creates a ``tf.Variable`` for it, allowing the coefficient
    to be optimized during model training.
    * `K` and `Ss` are typically positive, so they are managed in log-space
        internally if learnable (i.e., :math:`\log(K)` and :math:`\log(S_s)`
        are learned) to ensure positivity when :math:`\exp()` is applied.
* **Fixed Parameters:** Coefficients can also be set to fixed floating-point
    values.
* **Tensor Output:** Getter methods (``get_K()``, ``get_Ss()``, ``get_Q()``)
    return the coefficients as TensorFlow tensors, suitable for use in PDE
    computations within a TensorFlow graph.
* **Name Scoping:** Allows a ``name_prefix`` for created ``tf.Variable``s,
    helping to avoid name collisions in complex models.

**When to Use:**

Use ``GWResidualCalculator`` when you need a structured way to define, store,
and access physical parameters for a groundwater flow PDE within a PINN:

* When building a custom Keras layer (like
    :class:`~fusionlab.nn.pinn.base.GroundwaterFlowPDEResidual`) that computes
    the PDE residual.
* Inside a Keras model's ``call`` or ``train_step`` method if you are
    directly implementing the PDE residual calculation there.
* To centralize the definition of these physical parameters, making it
    easier to switch between learnable and fixed values or to experiment
    with different initializations.

**Code Example (Instantiation and Usage):**

.. code-block:: python
   :linenos:

   from fusionlab.nn.pinn.base import GWResidualCalculator
   import tensorflow as tf

   # Example 1: All learnable coefficients
   gw_config1 = {'K': 'learnable', 'Ss': 'learnable', 'Q': 'learnable'}
   calculator1 = GWResidualCalculator(gw_flow_coeffs=gw_config1, name_prefix="my_model_gw")
   
   K1 = calculator1.get_K()
   Ss1 = calculator1.get_Ss()
   Q1 = calculator1.get_Q()
   # print(f"K1 (learnable): {K1}, Ss1 (learnable): {Ss1}, Q1 (learnable): {Q1}")
   # print(f"Calculator 1 Trainable Variables: {calculator1.trainable_variables}")


   # Example 2: Mixed fixed and learnable coefficients
   gw_config2 = {'K': 1.5e-4, 'Ss': 'learnable', 'Q': 0.001}
   calculator2 = GWResidualCalculator(
       gw_flow_coeffs=gw_config2,
       default_Ss=2e-5 # Initial value if Ss is learnable
   )
   K2 = calculator2.get_K() # tf.Tensor (constant)
   Ss2 = calculator2.get_Ss() # tf.Tensor (from tf.Variable)
   # print(f"K2 (fixed): {K2.numpy()}, Ss2 (learnable): {Ss2}")


GroundwaterFlowPDEResidual (Keras Layer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.pinn.base.GroundwaterFlowPDEResidual`

The ``GroundwaterFlowPDEResidual`` is a ``tf.keras.layers.Layer`` specifically
designed to compute the residual of the 2D groundwater flow equation. It
leverages automatic differentiation (via ``tf.GradientTape``) to calculate
the necessary partial derivatives of the predicted hydraulic head (:math:`h`).

The standard 2D isotropic groundwater flow equation is:

.. math::
    S_s \frac{\partial h}{\partial t} - K \left( \frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2} \right) - Q = \mathcal{R}

where :math:`\mathcal{R}` is the residual that the PINN aims to minimize.

**Key Features:**

* **Keras Layer:** Integrates seamlessly into Keras model architectures.
* **Automatic Differentiation:** Computes first and second-order partial
    derivatives of the input hydraulic head prediction (:math:`h_{pred}`)
    with respect to time (:math:`t`) and spatial coordinates (:math:`x, y`).
* **Coefficient Management:** Internally uses an instance of
    :class:`~fusionlab.nn.pinn.base.GWResidualCalculator` to manage the
    physical coefficients :math:`S_s`, :math:`K`, and :math:`Q`. These
    coefficients can be learnable or fixed, as configured through the
    ``gw_flow_coeffs_config`` parameter.
* **Input Structure:** The ``call`` method expects a tuple of four tensors:
    `(h_pred, t_coords, x_coords, y_coords)`. It's crucial that `h_pred`
    is differentiable with respect to `t_coords`, `x_coords`, and `y_coords`
    within the TensorFlow graph.

**When to Use:**

Incorporate this layer into your PINN model when you need to enforce the
2D groundwater flow equation as a physics-based constraint:

* As part of your model's ``call`` method, where you compute predictions
    and then calculate the PDE residual using these predictions and their
    corresponding coordinates.
* The output of this layer (the PDE residual) would then be used in the
    physics part of your composite loss function during training.

**Code Example (Usage within a conceptual model):**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.pinn.base import GroundwaterFlowPDEResidual

   # Assume `head_prediction_network` is a Keras model that takes
   # (t,x,y) coordinates (or features derived from them) and outputs h_pred.
   # class HeadPredictor(tf.keras.Model):
   # ... (definition of a model that outputs h_pred) ...
   # head_predictor = HeadPredictor()

   # Define configuration for the groundwater flow coefficients
   gw_coeffs_config = {
       'K': 'learnable',  # Hydraulic conductivity
       'Ss': 1e-5,        # Specific storage (fixed)
       'Q': 0.0           # Source/sink term (fixed at zero)
   }

   # Instantiate the PDE residual layer
   gw_pde_layer = GroundwaterFlowPDEResidual(
       gw_flow_coeffs_config=gw_coeffs_config,
       name="gw_pde_loss_calculator"
   )

   # Example dummy inputs for a PINN
   batch_size = 4
   num_collocation_points = 100 # Points where PDE is evaluated
   dummy_t = tf.random.uniform((batch_size, num_collocation_points, 1), dtype=tf.float32)
   dummy_x = tf.random.uniform((batch_size, num_collocation_points, 1), dtype=tf.float32)
   dummy_y = tf.random.uniform((batch_size, num_collocation_points, 1), dtype=tf.float32)

   # In a real PINN, h_pred would come from your neural network,
   # and it must be differentiable with respect to t, x, y.
   # For this example, let's simulate h_pred using a simple function of t,x,y
   # so that gradients are non-zero.
   
   # This part simulates how PIHALNet might generate h_pred (e.g., gwl_pred_mean)
   # using a sub-network, ensuring it's part of the gradient tape.
   # We need a tape here to ensure h_pred is differentiable when passed to gw_pde_layer
   # (though gw_pde_layer uses its own internal tapes for the actual PDE derivatives).

   class MySimpleNet(tf.keras.Model): # A simple model for demonstration
       def __init__(self):
           super().__init__()
           self.dense = tf.keras.layers.Dense(1)
       def call(self, t, x, y):
           coords_combined = tf.concat([t, x, y], axis=-1)
           return self.dense(coords_combined)

   simple_h_net = MySimpleNet()

   with tf.GradientTape(watch_accessed_variables=False) as outer_tape:
       outer_tape.watch(dummy_t)
       outer_tape.watch(dummy_x)
       outer_tape.watch(dummy_y)
       # Simulate h_pred being generated by a network that takes t,x,y
       # This is crucial for AD to work inside the gw_pde_layer
       simulated_h_pred = simple_h_net(dummy_t, dummy_x, dummy_y)

   # Now, call the PDE residual layer
   # Inputs: (h_pred, t_coords, x_coords, y_coords)
   pde_residuals = gw_pde_layer((simulated_h_pred, dummy_t, dummy_x, dummy_y))

   # print(f"Shape of simulated_h_pred: {simulated_h_pred.shape}")
   # print(f"Shape of PDE residuals: {pde_residuals.shape}")
   # print(f"Example PDE residual values: {pde_residuals[0, :3, 0].numpy()}")

   # Trainable variables from the PDE layer (includes learnable K)
   # print("Trainable variables in gw_pde_layer:", [v.name for v in gw_pde_layer.trainable_variables])

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

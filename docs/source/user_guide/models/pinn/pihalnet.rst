.. _pinn_models_guide:

===========================================
Physics-Informed Neural Networks (PINNs)
===========================================

This section of the user guide delves into the Physics-Informed
Neural Networks (PINNs) available within the ``fusionlab`` library.
These models uniquely combine data-driven deep learning architectures
with physical laws, expressed as Partial Differential Equations (PDEs),
to produce forecasts that are not only accurate but also physically
consistent.

The primary PINN model currently featured is PIHALNet, designed for
complex spatio-temporal forecasting tasks like land subsidence,
where understanding and respecting the underlying geohydrological
processes is crucial.

.. toctree::
   :hidden:

PIHALNet (Physics-Informed Hybrid Attentive LSTM Network)
-----------------------------------------------------------
:API Reference: :class:`~fusionlab.nn.pinn.models.PIHALNet`

The ``PIHALNet`` model is a sophisticated hybrid architecture tailored
for multi-horizon probabilistic forecasting of coupled geophysical
phenomena, such as land subsidence and groundwater level changes.
It leverages the strengths of deep learning for pattern recognition
from data while constraining its predictions with physical knowledge.

**Key Features:**

* **Hybrid Architecture:** Integrates a data-driven forecasting core
  (HALNet - Hybrid Attentive LSTM Network) with a physics-informed
  module that incorporates PDE residuals into the loss function.
* **Dual-Target Prediction:** Simultaneously predicts multiple related
  variables (e.g., subsidence and groundwater levels).
* **Input Handling:** Accepts static, dynamic (past observed), and
  future known inputs. It can utilize:
    * :class:`~fusionlab.nn.components.VariableSelectionNetwork` (VSN)
      for intelligent feature selection and embedding of different
      input types.
    * If VSNs are not used,
      :class:`~fusionlab.nn.components.MultiModalEmbedding` can
      process raw dynamic and future inputs.
* **Advanced Temporal Processing:**
    * :class:`~fusionlab.nn.components.MultiScaleLSTM` captures
      temporal dependencies at various user-defined scales.
    * A suite of attention mechanisms enhances contextual understanding:
        * :class:`~fusionlab.nn.components.HierarchicalAttention`
        * :class:`~fusionlab.nn.components.CrossAttention`
        * :class:`~fusionlab.nn.components.MemoryAugmentedAttention`
        * :class:`~fusionlab.nn.components.MultiResolutionAttentionFusion`
* **Physics-Informed Constraints:**
    * Supports different PDE modes (via ``pde_mode``), such as
      'consolidation' (Terzaghi's theory), 'gw_flow' (groundwater
      flow equation), or 'both' (coupled).
    * Physical coefficients (e.g., consolidation coefficient `C`,
      hydraulic conductivity `K`, specific storage `Ss`) can be:
        * **Learnable:** Treated as trainable parameters, allowing
          the model to discover them from data (default for some).
        * **Fixed:** Specified as known constants.
    * The PDE residual is computed based on model outputs and
      incorporated into the total loss function, weighted by
      ``lambda_pde``.
* **Probabilistic Forecasting:** Employs
  :class:`~fusionlab.nn.components.QuantileDistributionModeling`
  to output forecasts for specified ``quantiles``, enabling
  uncertainty estimation. Point forecasts are produced if
  ``quantiles`` is ``None``.
* **Flexible Output Structure:** Uses a
  :class:`~fusionlab.nn.components.MultiDecoder` to generate
  horizon-specific predictions.

**When to Use PIHALNet:**

``PIHALNet`` is particularly well-suited for:

* Forecasting spatio-temporal phenomena governed by known (or
  partially known) physical laws.
* Problems where data might be sparse or noisy, and incorporating
  physical constraints can improve model robustness and generalization.
* Situations requiring predictions for coupled physical processes
  (e.g., subsidence and groundwater).
* Tasks where discovering or refining physical parameters from
  observational data is of interest.
* Generating probabilistic forecasts to quantify prediction uncertainty.

Formulation
~~~~~~~~~~~~~

PIHALNet's operation involves two main conceptual parts:

1.  **Data-Driven Forecasting (HALNet Core):**
    This part processes the input features (static, dynamic, future)
    to produce initial forecasts for the target variables (e.g.,
    subsidence :math:`s` and groundwater level :math:`h`).

    * **Input Processing:** Inputs are optionally processed by
      Variable Selection Networks (VSNs) and Gated Residual
      Networks (GRNs) or by a MultiModalEmbedding layer.
      Positional encoding is added.
    * **Temporal Encoding:** The
      :class:`~fusionlab.nn.components.MultiScaleLSTM` processes
      dynamic features.
    * **Attention Mechanisms:** A series of attention layers
      (Hierarchical, Cross, Memory-Augmented, Multi-Resolution
      Fusion) refine and integrate features from different
      sources and contexts.
    * **Decoding:** The
      :class:`~fusionlab.nn.components.MultiDecoder` generates
      horizon-specific outputs.
    * **Output Layer:**
      :class:`~fusionlab.nn.components.QuantileDistributionModeling`
      produces the final data-driven predictions
      (:math:`\hat{s}_{data}, \hat{h}_{data}`), potentially across
      multiple quantiles. The mean of these predictions
      (:math:`\bar{s}_{data}, \bar{h}_{data}`) is also available for
      PDE calculation.

2.  **Physics-Informed Module:**
    * **PDE Residual Calculation:** The mean predictions from the
      data-driven core (:math:`\bar{s}_{data}, \bar{h}_{data}`) and
      the input coordinates (:math:`t, x, y`) are used to compute
      the residual of the specified PDE(s).
        * For **consolidation** (Terzaghi's 1D theory):
          A common simplified form relates head change to its
          second spatial derivative.
            .. math::
                \mathcal{R}_{cv} = \frac{\partial \bar{h}_{data}}{\partial t} - C_v \frac{\partial^2 \bar{h}_{data}}{\partial z^2}
          ``PIHALNet`` offers a flexible implementation where this
          can be adapted.
        * For **groundwater flow** (2D horizontal, isotropic):
            .. math::
                \mathcal{R}_{gw} = S_s \frac{\partial \bar{h}_{data}}{\partial t} - K \left( \frac{\partial^2 \bar{h}_{data}}{\partial x^2} + \frac{\partial^2 \bar{h}_{data}}{\partial y^2} \right) + Q
          This residual would be computed by a component like
          :class:`~fusionlab.nn.pinn.base.GroundwaterFlowPDEResidual`.
    * The model stores these residuals as :math:`\text{pde_residual}`
      in its output dictionary.

3.  **Loss Function:**
    The total loss function during training is a weighted sum of the
    data fidelity loss (:math:`\mathcal{L}_{data}`) and the physics
    loss (:math:`\mathcal{L}_{physics}`):

    .. math::
        \mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_{PDE} \mathcal{L}_{physics}

    * :math:`\mathcal{L}_{data}`: Calculated using specified loss
      functions (e.g., MSE or a quantile loss like
      :func:`~fusionlab.nn.losses.combined_quantile_loss`).
    * :math:`\mathcal{L}_{physics}`: Typically the mean squared error
      of the :math:`\text{pde_residual}`, e.g.,
      :math:`\frac{1}{N} \sum (\mathcal{R})^2`.
    * :math:`\lambda_{PDE}`: A hyperparameter controlling the
      influence of the physics-based loss term.

**Code Example (Instantiation):**

.. code-block:: python
   :linenos:

   import numpy as np
   import tensorflow as tf
   from fusionlab.nn.pinn.models import PIHALNet

   # Example Configuration
   static_dim, dynamic_dim, future_dim = 5, 4, 2
   horizon = 3
   time_steps = 7

   my_quantiles = [0.1, 0.5, 0.9]

   # Fixed parameters are typically inferred from data
   fixed_params = {
       "static_input_dim": static_dim,
       "dynamic_input_dim": dynamic_dim,
       "future_input_dim": future_dim,
       "output_subsidence_dim": 1,
       "output_gwl_dim": 1,
       "forecast_horizon": horizon,
       "quantiles": my_quantiles,
       "max_window_size": time_steps,
       "pde_mode": "consolidation",
       "pinn_coefficient_C": "learnable",
       "gw_flow_coeffs": {'K': 'learnable', 'Ss': 1e-5}
   }

   # Architectural hyperparameters (tuned by PIHALTuner)
   arch_params = {
       "embed_dim": 64, "hidden_units": 64, "lstm_units": 64,
       "attention_units": 32, "num_heads": 4, "dropout_rate": 0.1,
       "vsn_units": 32, "use_vsn": True, "scales": [1, 2],
       "memory_size": 50
   }

   # Instantiate PIHALNet
   pihalnet_model = PIHALNet(**fixed_params, **arch_params)

   # Example dummy input data
   batch_size = 2
   dummy_inputs = {
       'coords': tf.random.normal((batch_size, horizon, 3)),
       'static_features': tf.random.normal((batch_size, static_dim)),
       'dynamic_features': tf.random.normal((batch_size, time_steps,
                                              dynamic_dim)),
       'future_features': tf.random.normal((batch_size, horizon,
                                             future_dim)),
   }

   # Compile the model
   from tensorflow.keras.losses import MeanSquaredError
   from tensorflow.keras.optimizers import Adam

   # A real quantile loss function would be used here
   loss_fns = {'subs_pred': MeanSquaredError(), 'gwl_pred': MeanSquaredError()}

   pihalnet_model.compile(
       optimizer=Adam(learning_rate=1e-3),
       loss=loss_fns,
       metrics={'subs_pred': ['mae'], 'gwl_pred': ['mae']},
       lambda_pde=0.1
   )

   pihalnet_model.summary(line_length=110)


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


PINN Utilities
----------------

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




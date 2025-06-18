# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

""" Physics-informed for Groundwater flow modeling."""

from __future__ import annotations
from typing import Optional, Union, Dict,List, Tuple, Any  

from ..._fusionlog import fusionlog 
from ...api.property import NNLearner
from ...core.handlers import columns_manager 
from ...params import (
    LearnableK, LearnableSs, LearnableQ, resolve_physical_param
)
from ...utils.deps_utils import ensure_pkg 

from .. import KERAS_DEPS, KERAS_BACKEND, dependency_message
from .._tensor_validation import validate_tensors
from ..comp_utils import resolve_gw_coeffs 
from .utils import extract_txy_in 

Tensor = KERAS_DEPS.Tensor 
GradientTape =KERAS_DEPS.GradientTape
Dense =KERAS_DEPS.Dense 
Model =KERAS_DEPS.Model 
InputLayer= KERAS_DEPS.InputLayer
Sequential =KERAS_DEPS.Sequential 
Adam =KERAS_DEPS.Adam

tf_concat =KERAS_DEPS.concat
tf_reduce_mean=KERAS_DEPS.reduce_mean
tf_square =KERAS_DEPS.square
tf_shape =KERAS_DEPS.shape 
tf_reshape = KERAS_DEPS.reshape 
    

DEP_MSG = dependency_message('nn.pinn') 

logger = fusionlog().get_fusionlab_logger(__name__)

__all__=['PiTGWFlow']


class PiTGWFlow(Model, NNLearner):
    """Physics-Informed Transient Groundwater Flow.

    A self-contained Physics-Informed Neural Network (PINN) designed
    to solve the 2D/3D transient groundwater flow equation. This
    model leverages a simple Multi-Layer Perceptron (MLP) to
    approximate the hydraulic head :math:`h` as a continuous
    function of time and space, :math:`h = h(t, x, y)`.

    The model is trained by minimizing the residual of the governing
    PDE at a set of collocation points, rather than by fitting to
    observed data. The core of the model is its custom training and
    evaluation steps, which compute the necessary spatial and
    temporal derivatives to enforce the physical law.

    The governing equation solved by this PINN is:

    .. math::
       S_s \\frac{\partial h}{\partial t} - K \\left(
       \\frac{\partial^2 h}{\partial x^2} +
       \\frac{\partial^2 h}{\partial y^2} \\right) - Q = 0

    Where :math:`K` is the hydraulic conductivity, :math:`S_s` is
    the specific storage, and :math:`Q` is a source/sink term. These
    physical parameters can be set as fixed constants or as
    trainable variables.

    Parameters
    ----------
    hidden_units : int or list of int, optional
        Defines the architecture of the internal MLP.
        - If an **int**, three hidden layers of that size are created.
        - If a **list**, its first three values are used as the sizes
          for the three hidden layers.
        Defaults to ``[32, 32, 32]``.

    activation : str, default 'tanh'
        The activation function to use in the hidden layers of the
        MLP. 'tanh' is often recommended for PINNs.

    learning_rate : float, default 1e-3
        The learning rate for the Adam optimizer used to train the
        network weights and any learnable physical parameters.

    K : float or LearnableK, default 1.0
        The hydraulic conductivity. Can be provided as a fixed Python
        float or as a ``LearnableK`` instance to be optimized during
        training.

    Ss : float or LearnableSs, default 1e-4
        The specific storage. Can be provided as a fixed Python
        float or as a ``LearnableSs`` instance to be optimized during
        training.

    Q : float or LearnableQ, default 0.0
        The volumetric source/sink term. Can be provided as a fixed
        Python float or as a ``LearnableQ`` instance to be optimized
        during training.

    gw_coeffs_config : dict, optional
        A dictionary to conveniently configure physical parameters. If
        provided, its values will supersede the individual ``K``,
        ``Ss``, and ``Q`` arguments. For example::

            gw_coeffs_config = {'K': LearnableK(0.5), 'Ss': 1e-5}

    name : str, default 'PiTGWFlow'
        The name of the Keras model.

    **kwargs
        Additional keyword arguments passed to the parent
        ``tf.keras.Model`` constructor.

    Notes
    -----
    This model is a pure physics solver, meaning its loss function is
    derived entirely from the PDE residual. It does not perform data
    fitting in the traditional sense. Consequently, the ``y_true``
    component of a dataset provided to ``fit()`` or ``evaluate()``
    is ignored.

    The model implements custom ``train_step`` and ``test_step`` methods.
    Within these steps, ``tf.GradientTape`` is used to calculate the
    first and second-order derivatives of the network's output (:math:`h`)
    with respect to its inputs (:math:`t, x, y`). This is necessary
    to compute the PDE residual, which forms the loss.

    When a physical parameter is made learnable (e.g., by passing
    ``K=LearnableK(...)``), it is automatically added to the model's
    list of trainable variables and is updated via backpropagation
    to minimize the physics loss.

    See Also
    --------
    fusionlab.nn.pinn.TransFlowSubsNet : A more complex, hybrid
        data-physics model that couples groundwater flow with land
        subsidence and is trained on both physical laws and observed
        data.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from fusionlab.nn.pinn import PiTGWFlow
    >>> from fusionlab.params import LearnableK, LearnableSs
    ...
    >>> # 1. Instantiate the model with a learnable K and a fixed Ss
    >>> model = PiTGWFlow(
    ...     hidden_units=[64, 64, 64],
    ...     K=LearnableK(initial_value=0.5),
    ...     Ss=1e-5,
    ...     Q=0.0
    ... )
    ...
    >>> # 2. Create a dataset of collocation points (t, x, y)
    >>> n_points = 1000
    >>> coords = {
    ...     "t": tf.random.uniform((n_points, 1)),
    ...     "x": tf.random.uniform((n_points, 1)),
    ...     "y": tf.random.uniform((n_points, 1)),
    ... }
    >>> # Dummy targets are needed for the Keras API, but are ignored
    >>> dummy_y = tf.zeros((n_points, 1))
    >>> dataset = tf.data.Dataset.from_tensor_slices((coords, dummy_y)).batch(32)
    ...
    >>> # 3. Compile and train the model
    >>> # The optimizer is taken from the model instance.
    >>> model.compile()
    >>> history = model.fit(dataset, epochs=10)
    ...
    >>> # The history will contain the physics-based loss
    >>> print(list(history.history.keys()))
    ['pde_loss']

    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        hidden_units: Optional[Union[int, List[int]]] = None,
        activation: str = "tanh",
        learning_rate: float = 1e-3,
        K: Union[float, LearnableK] = 1.0,
        Ss: Union[float, LearnableSs] = 1e-4,
        Q: Union[float, LearnableQ] = 0.0,
        gw_coeffs_config : Dict[str, Any]=None, 
        name: str = "PiTGWFlow",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        # --- Store configuration and physical parameters ---
        if hidden_units is None:
            hidden_units = [32, 32, 32]

        self.hidden_units_config = hidden_units
        if isinstance(self.hidden_units_config, (float, int)):
            self.hidden_units = [int(self.hidden_units_config)] * 3
        else:
            temp_units = columns_manager(self.hidden_units_config)
            self.hidden_units = (temp_units * 3)[:3]

        self.activation = activation
        self.learning_rate = learning_rate
        self.gw_coeffs_config = gw_coeffs_config 
        
        K, Ss, Q = resolve_gw_coeffs( 
            gw_flow_coeffs= gw_coeffs_config, 
            K=K, Ss =Ss, Q =Q, 
            param_status="fixed", 
            ) # collected the fixed values even Learnable parameters 
        # The learnable param should be create with resolve_physical_param 
        
        self.K_config = K
        self.Ss_config = Ss
        self.Q_config = Q

        # Create tf.Variables for learnable parameters
        self.K = resolve_physical_param(
            self.K_config, name="param_K", status="learnable")
        self.Ss = resolve_physical_param(
            self.Ss_config, name="param_Ss", status="learnable")
        self.Q = resolve_physical_param(
            self.Q_config, name="param_Q", status="learnable")

        # --- Build the neural network (MLP) ---
        layers = [InputLayer(input_shape=(3,))]
        for units, name in zip (self.hidden_units, ['K', 'Ss', 'Q']):
            layers.append(Dense(units, activation=self.activation,
                                name=f"param_{name}"))
        layers.append(Dense(1, activation="linear", name="h_pred"))
        self.coord_mlp = Sequential(layers, name="GWFlow_MLP")

        # Optimizer
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def call(
        self,
        inputs: Union[Tensor, Dict[str, Tensor]],
        training: bool = False
    ) -> Tensor:
        """Performs the forward pass of the network.
    
        This method maps the input spatio-temporal coordinates
        :math:`(t, x, y)` to the predicted hydraulic head :math:`h`.
        It is designed to be flexible, accepting coordinates in several
        formats.
    
        Parameters
        ----------
        inputs : tf.Tensor or dict
            The input coordinates. Supported formats include:
            - A dictionary with keys ``'t'``, ``'x'``, and ``'y'``,
              where each value is a tensor of shape
              ``(batch, time_steps, 1)``.
            - A single concatenated tensor of shape
              ``(batch, time_steps, 3)``.
        training : bool, optional
            Indicates whether the model is in training mode. This is
            passed to internal Keras layers. Defaults to ``False``.
    
        Returns
        -------
        tf.Tensor
            A tensor of shape ``(batch, time_steps, 1)`` containing
            the predicted hydraulic head :math:`h` at each input
            coordinate.
    
        """
        t_coords, x_coords, y_coords = extract_txy_in(inputs,  verbose=0)
        validate_tensors(t_coords, x_coords, y_coords, last_dim=1, check_N=True)

        coords_tensor = tf_concat([t_coords, x_coords, y_coords], axis=-1)
        original_shape = tf_shape(coords_tensor)
        num_features = original_shape[-1]
        
        # Reshape from (batch, time, 3) to (batch * time, 3) for MLP
        reshaped_coords = tf_reshape(coords_tensor, [-1, num_features])
        h_pred_flat = self.coord_mlp(reshaped_coords, training=training)

        # Reshape output back to (batch, time, 1)
        output_shape = [original_shape[0], original_shape[1], 1]
        return tf_reshape(h_pred_flat, output_shape)

    def train_step(
        self, data: Tuple[Dict[str, Tensor], Any]
        ) -> Dict[str, Tensor]:
        """Defines the logic for a single optimization step.
    
        This method overrides the default Keras training behavior to
        implement the physics-informed loss. In each step, it computes
        the residual of the governing PDE and uses it as the loss to
        update all trainable variables, including both the network
        weights and any learnable physical parameters.
    
        The process involves:
        1.  Computing derivatives of the output :math:`h` with respect
            to inputs :math:`(t, x, y)` using ``tf.GradientTape``.
        2.  Assembling the PDE residual using these derivatives.
        3.  Calculating the mean squared error of the residual.
        4.  Applying gradients to update the model's variables.
    
        Parameters
        ----------
        data : tuple
            A tuple of ``(inputs, targets)`` as provided by a
            ``tf.data.Dataset``. The ``inputs`` are a dictionary of
            coordinate tensors, while the ``targets`` are ignored as
            the loss is unsupervised.
    
        Returns
        -------
        dict
            A dictionary mapping the metric name ``'pde_loss'`` to its
            scalar value for this training step.
    
        """
        """Custom training step to minimize the PDE residual."""
        coords_batch, y_true = data  # y_true is unused, for compatibility
        
        with GradientTape(persistent=True) as tape:
            # Extract coordinates and watch them for gradient computation
            t, x, y = extract_txy_in(coords_batch)
            tape.watch([t, x, y])
            
            # Combine coordinates for the forward pass
            coords_for_model = tf_concat([t, x, y], axis=-1)
            
            # --- FORWARD PASS ---
            # h_pred must be computed inside the tape's context
            h_pred = self(coords_for_model, training=True)
            tape.watch(h_pred)
            
            # --- PDE RESIDUAL CALCULATION ---
            # First-order derivatives
            dh_dt = tape.gradient(h_pred, t)
            dh_dx = tape.gradient(h_pred, x)
            dh_dy = tape.gradient(h_pred, y)

            # Second-order derivatives
            d2h_dx2 = tape.gradient(dh_dx, x)
            d2h_dy2 = tape.gradient(dh_dy, y)
            
            # Validate gradients
            if any(g is None for g in [dh_dt, d2h_dx2, d2h_dy2]):
                raise ValueError(
                    "Failed to compute one or more PDE gradients. "
                    "Ensure t, x, y inputs influence the model output."
                )
            
            # Assemble the PDE residual
            laplacian_h = d2h_dx2 + d2h_dy2
            residual = (
                self.Ss.get_value() * dh_dt) - (
                    self.K.get_value() * laplacian_h) - self.Q.get_value() 
            
            # The loss is the mean of the squared residuals.
            pde_loss = tf_reduce_mean(tf_square(residual))

        # --- APPLY GRADIENTS ---
        trainable_vars = self.trainable_variables
        grads = tape.gradient(pde_loss, trainable_vars)
        del tape  # Clean up the persistent tape
        
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        # --- METRICS & RETURN ---
        if self.compiled_metrics:
            self.compiled_metrics.update_state(y_true, residual)
            
        results = {"pde_loss": pde_loss}
        for m in self.metrics:
            results[m.name] = m.result()
            
        return results

    def test_step(
        self, data: Tuple[Dict[str, Tensor], Any]
    ) -> Dict[str, Tensor]:
        """Defines the logic for a single evaluation step.
    
        This method overrides the default Keras evaluation behavior to
        report the model's physics fidelity. It calculates the PDE
        residual on the validation or test data but does not perform
        any weight updates. This is crucial for assessing how well the
        model adheres to the physical laws on unseen collocation points.
    
        Parameters
        ----------
        data : tuple
            A tuple of ``(inputs, targets)`` from the validation or
            test dataset. The ``inputs`` are a dictionary of coordinate
            tensors; the ``targets`` are ignored.
    
        Returns
        -------
        dict
            A dictionary mapping the metric name ``'pde_loss'`` to its
            scalar value for the evaluation step.
    
        """
        coords_batch, y_true = data
        
        with GradientTape(persistent=True) as tape:
            t, x, y = extract_txy_in(coords_batch)
            tape.watch([t, x, y])
            
            coords_for_model = tf_concat([t, x, y], axis=-1)
            h_pred = self(coords_for_model, training=False)
            tape.watch(h_pred)
            
            dh_dt = tape.gradient(h_pred, t)
            dh_dx = tape.gradient(h_pred, x)
            dh_dy = tape.gradient(h_pred, y)
            
            d2h_dx2 = tape.gradient(dh_dx, x)
            d2h_dy2 = tape.gradient(dh_dy, y)
            
        del tape

        if any(g is None for g in [dh_dt, d2h_dx2, d2h_dy2]):
             raise ValueError("Failed to compute gradients during validation.")
             
        laplacian_h = d2h_dx2 + d2h_dy2
        residual = (self.Ss.get_value() * dh_dt) - (
            self.K.get_value() * laplacian_h) - self.Q.get_value()
        pde_loss = tf_reduce_mean(tf_square(residual))
        
        if self.compiled_metrics:
            self.compiled_metrics.update_state(y_true, residual)
            
        results = {"pde_loss": pde_loss}
        for m in self.metrics:
            results[m.name] = m.result()
            
        return results

    def get_config(self) -> dict:
        """Returns the serializable configuration of the model.
    
        This method allows the model to be saved and re-instantiated
        without losing its architectural and parameter setup. It captures
        all initialization arguments, including the configuration of the
        MLP and the physical parameters.
    
        Returns
        -------
        dict
            A dictionary containing all constructor parameters needed to
            recreate the model instance. Learnable parameter objects
            (e.g., ``LearnableK``) are serialized as part of this config.
    
        """
        base_config = super().get_config()
        config = {
            "hidden_units": self.hidden_units_config,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "K": self.K_config,
            "Ss": self.Ss_config,
            "Q": self.Q_config,
            "gw_coeffs_config": self.gw_coeffs_config 
        }
        base_config.update(config)
        return base_config

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "PiTGWFlow":
        """Creates a model instance from its configuration dictionary.
    
        This class method is the counterpart to ``get_config`` and is
        used by Keras's loading utilities to reconstruct a model from
        its saved configuration.
    
        Parameters
        ----------
        config : dict
            The configuration dictionary, typically generated by
            ``get_config()``.
        custom_objects : dict, optional
            A dictionary mapping names to custom classes or functions.
            Keras uses this to deserialize custom objects like
            ``LearnableK``.
    
        Returns
        -------
        PiTGWFlow
            A new instance of the ``PiTGWFlow`` model.
    
        """
        return cls(**config)
    

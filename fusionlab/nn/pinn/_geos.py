# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause
# File fusionlab/nn/pinn/_geos.py 

from __future__ import annotations

from typing import Optional, Union, Dict,List, Tuple, Any  

from ..._fusionlog import fusionlog 
from ...api.property import NNLearner, BaseClass  
from ...core.handlers import columns_manager 

from ...params import ( 
    LearnableK, 
    LearnableSs, 
    LearnableQ, 
    resolve_physical_param
)
from ...utils.deps_utils import ensure_pkg 

from .. import KERAS_DEPS, KERAS_BACKEND, dependency_message
from .._tensor_validation import validate_tensors
from ..utils import squeeze_last_dim_if 
from .op import compute_gw_flow_residual #noqa
from .utils import extract_txy_in 

logger = fusionlog().get_fusionlab_logger(__name__)

if KERAS_BACKEND: 
    Variable =KERAS_DEPS.Variable 
    Tensor = KERAS_DEPS.Tensor 
    GradientTape =KERAS_DEPS.GradientTape
    Dense =KERAS_DEPS.Dense 
    Layer =KERAS_DEPS.Layer
    Model =KERAS_DEPS.Model 
    InputLayer= KERAS_DEPS.InputLayer
    Sequential =KERAS_DEPS.Sequential 
    Adam =KERAS_DEPS.Adam
    
    tf_log = KERAS_DEPS.log 
    tf_cast =KERAS_DEPS.cast 
    tf_float32 = KERAS_DEPS.float32 
    tf_abs =KERAS_DEPS.abs 
    tf_exp =KERAS_DEPS.exp 
    tf_constant =KERAS_DEPS.constant 
    tf_zeros_like = KERAS_DEPS.zeros_like
    tf_random = KERAS_DEPS.random 
    tf_concat =KERAS_DEPS.concat
    tf_reduce_mean=KERAS_DEPS.reduce_mean
    tf_square =KERAS_DEPS.square
    tf_shape =KERAS_DEPS.shape 
    tf_reshape = KERAS_DEPS.reshape 
    

DEP_MSG = dependency_message('nn.pinn._geos') 

class PITGWFlow(Model, NNLearner):
    """
    Physic Informed Transient Grounwwater Flow.
    
    A self-contained PINN for 2D/3D transient groundwater flow.

    This model uses a simple Multi-Layer Perceptron (MLP) to approximate
    the hydraulic head `h` as a function of time and space, `h = h(t, x, y)`.
    It includes a custom training step to minimize the residual of the
    2D transient groundwater flow PDE.

    The physical parameters K (Hydraulic Conductivity), Ss (Specific
    Storage), and Q (Source/Sink term) can be provided as fixed values
    or as learnable parameters.

    Usage
    -----
    >>> from fusionlab.nn.pinn._geos import GWFlowPINN
    >>> from fusionlab.params import LearnableK, LearnableSs
    >>> import tensorflow as tf

    >>> # Initialize the PINN with learnable K and Ss
    >>> model = GWFlowPINN(
    ...     hidden_units=[32, 32, 32],
    ...     learning_rate=1e-3,
    ...     K=LearnableK(initial_value=0.5),
    ...     Ss=LearnableSs(initial_value=1e-4),
    ...     Q=0.0  # Fixed Q
    ... )
    >>> # Generate some collocation points (where PDE is enforced)
    >>> n_points = 100
    >>> coords_dict = {
    ...     "t": tf.random.uniform((n_points, 1)),
    ...     "x": tf.random.uniform((n_points, 1)),
    ...     "y": tf.random.uniform((n_points, 1)),
    ... }
    >>> # Dummy targets (required by Keras fit, but not used in this PINN's loss)
    >>> dummy_y = tf.zeros((n_points, 1))
    >>>
    >>> # Train the model to minimize the PDE residual
    >>> # model.fit(coords_dict, dummy_y, epochs=10, batch_size=32)
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
        name: str = "PITGWFlow",
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
            temp_units= columns_manager (self.hidden_units_config)
            self.hidden_units = (temp_units * 3)[:3]
            
        self.activation = activation
        self.learning_rate = learning_rate
        
        # Store original param configurations for serialization
        self.K_config = K
        self.Ss_config = Ss
        self.Q_config = Q

        # Create tf.Variables or tf.Constants as direct model
        # attributes. This is CRITICAL for Keras to track them as
        # trainable_variables if they are learnable.
        self.K = resolve_physical_param(
            self.K_config, name="param_K")
        self.Ss = resolve_physical_param(
            self.Ss_config, name="param_Ss")
        self.Q = resolve_physical_param(
            self.Q_config, name="param_Q")

        # --- Build the neural network (MLP) ---
        layers = [InputLayer(input_shape=(3,))]
        for units, name in zip (
                self.hidden_units, ["param_K", "param_Ss", "Param_Q"]):
            layers.append(Dense(units, activation=self.activation, name=name))
        layers.append(Dense(1, activation="linear", name="h_pred"))
        self.net = Sequential(layers, name="GWFlow_MLP")

        # Optimizer for training the network and learnable params
        self.optimizer = Adam(learning_rate=self.learning_rate)


    def call(
        self,
        inputs: Union[Tensor, Dict[str, Tensor]],
        training: bool = False
    ) -> Tensor:
        """
        Forward pass of the network: (t, x, y) -> h_pred.

        This method is designed to be flexible and can accept coordinates
        in several formats.

        Parameters
        ----------
        inputs : Union[tf.Tensor, Dict[str, tf.Tensor]]
            A tensor or dictionary representing the coordinates.
            Supported formats:
            - A single 2D tensor of shape `(batch, 3)` for `[t, x, y]`.
            - A single 3D tensor of shape `(batch, time_steps, 3)`.
            - A dictionary with a 'coords' key holding one of the above tensors.
            - A dictionary with 't', 'x', 'y' keys, each a tensor of shape
              `(batch, 1)` or `(batch, time_steps, 1)`.

        training : bool
            Whether the model is in training mode (passed to internal layers).

        Returns
        -------
        tf.Tensor
            A tensor containing predicted hydraulic head values. The shape
            will match the input time dimension, e.g., `(batch, 1)` or
            `(batch, time_steps, 1)`.
        """
        # Standardize various input formats into a single tensor of shape (N, 3)
        t_coords, x_coords, y_coords = extract_txy_in(
            inputs, 
            verbose=0
        )
        original_shape = tf_shape(t_coords) # To reshape output later if needed
        
        # validate tensor to check whether they all as the same shape 
        validate_tensors(
            t_coords, x_coords, y_coords, 
            last_dim =1, check_N = True
        )
        # Concatenate into the (N, 3) format expected by the MLP
        coords_tensor = tf_concat([t_coords, x_coords, y_coords], axis=-1)
        
        # If there's a time dimension, flatten for MLP processing
        # if len(original_shape) > 2: # e.g., (batch, time_steps, 1)
        #     num_features = tf_shape(coords_tensor)[-1]
        #     reshaped_coords = tf_reshape(coords_tensor, [-1, num_features])
        #     h_pred_flat = self.net(reshaped_coords, training=training)
        #     # Reshape output back to include the time dimension
        #     output_shape = [original_shape[0], original_shape[1], 1]
        #     return tf_reshape(h_pred_flat, output_shape)
        
        # # For 2D inputs (batch, 3)
        # return self.net(coords_tensor, training=training)
    
        # --- FIX: Use tf.rank and correctly reshape for the MLP ---
        # The MLP (self.net) expects a 2D input of shape (num_points, 3).
        # We must flatten the batch and time dimensions together.
        original_shape = tf_shape(coords_tensor)
        num_features = tf_shape(coords_tensor)[-1]
        batch_size, time_steps = original_shape[0], original_shape[1]
        
        # Reshape from (batch, time, 3) to (batch * time, 3)
        reshaped_coords = tf_reshape(coords_tensor, [-1, num_features])
        
        # Get predictions from the network
        h_pred_flat = self.net(reshaped_coords, training=training)
        
        # Reshape output back to include the original batch and time dimensions
        # Final shape: (batch, time, 1)
        output_shape = [batch_size, time_steps, 1]
        
        return squeeze_last_dim_if (
            tf_reshape(h_pred_flat, output_shape), 
            output_dims=1
            )


    def compute_residual(
        self,
        coords: Dict[str, Tensor],
        h_pred: Tensor
    ) -> Tensor:
        """
        Computes the 2D transient groundwater flow residual.
        
        R = Ss * ∂h/∂t - K * (∂²h/∂x² + ∂²h/∂y²) - Q
        """
        print(coords)
        # A persistent tape is needed to calculate gradients of
        # gradients (i.e., second-order derivatives).
        with GradientTape(persistent=True) as tape:
            tape.watch(coords['t'])
            tape.watch(coords['x'])
            tape.watch(coords['y'])
            
            # Use a nested tape for first-order derivatives.
            with GradientTape(persistent=True) as inner_tape:
                inner_tape.watch([coords['t'], coords['x'], coords['y']])
                # The prediction 'h_pred' is passed in. We must ensure
                # the tape can trace its origin to the coordinates.
                # However, for gradient calculation, we need to treat
                # h_pred as the target to differentiate.
                
            # First-order derivatives of h_pred wrt coordinates.
            dh_dt = tape.gradient(h_pred, coords['t'])
            dh_dx = tape.gradient(h_pred, coords['x'])
            dh_dy = tape.gradient(h_pred, coords['y'])
            
        # Second-order spatial derivatives.
        d2h_dx2 = tape.gradient(dh_dx, coords['x'])
        d2h_dy2 = tape.gradient(dh_dy, coords['y'])
        
        # Clean up the tape from memory once done.
        del tape

        # Validate that gradients were computed successfully.
        if any(g is None for g in [dh_dt, dh_dx, dh_dy, d2h_dx2, d2h_dy2]):
             raise ValueError(
                "Failed to compute one or more first-order gradients. "
                "Ensure t, x, y are inputs to the model and influence its "
                "hydraulic head prediction."
            )
            
        # Assemble the PDE residual using the physical coefficients.
        laplacian_h = d2h_dx2 + d2h_dy2
        residual = (self.Ss * dh_dt) - (self.K * laplacian_h) - self.Q
        return residual

    def train_step(
        self, data: Tuple[Dict[str, Tensor], Any]
    ) -> Dict[str, Tensor]:
        """Custom training step to minimize the PDE residual."""
        coords_batch, _ = data
        
        with GradientTape() as tape:
            # The tape automatically watches the model's trainable
            # variables (network weights and learnable K, Ss, Q).
            
            # To compute the PDE residual, we need the model's
            # prediction AND its derivatives wrt inputs. This all
            # needs to happen inside this tape context.
            h_pred = self(coords_batch, training=True)
            
            residual = self.compute_residual(
                coords=coords_batch, h_pred=h_pred
            )
            
            # The loss is the mean of the squared residuals.
            pde_loss = tf_reduce_mean(tf_square(residual))

        # Compute gradients of the loss w.r.t. all trainable variables.
        grads = tape.gradient(pde_loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables)
        )
        
        # Update compiled metrics (if any).
        if self.compiled_metrics is not None:
            self.compiled_metrics.update_state(_, residual)

        results = {"pde_loss": pde_loss}
        if self.metrics:
            for m in self.metrics:
                results[m.name] = m.result()
        return results

    def test_step(
        self, data: Tuple[Dict[str, Tensor], Any]
    ) -> Dict[str, Tensor]:
        """Custom validation step to evaluate the PDE residual."""
        coords_batch, _ = data
        h_pred = self(coords_batch, training=False)
        residual = self.compute_residual(
            coords=coords_batch, h_pred=h_pred
        )
        pde_loss = tf_reduce_mean(tf_square(residual))
        
        if self.compiled_metrics is not None:
            self.compiled_metrics.update_state(_, residual)
        
        results = {"pde_loss": pde_loss}
        if self.metrics:
            for m in self.metrics:
                results[m.name] = m.result()
        return results
  
        
    def get_config(self) -> dict:
        """Returns the configuration of the model for serialization."""
        base_config = super().get_config()
        config = {
            "hidden_units": self.hidden_units_config,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            # Use the config objects for serialization
            "K": self.K_config,
            "Ss": self.Ss_config,
            "Q": self.Q_config,
        }
        base_config.update(config)
        return base_config
    
    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "PITGWFlow":
        """Reconstructs a model from its config."""
        # Keras automatically handles deserialization of registered
        # custom objects (like LearnableK) if they are passed directly.
        return cls(**config)




class GWResidual:
    """
    Manages and provides physical coefficients (K, Ss, Q) for groundwater
    flow PDE calculations.

    This class initializes coefficients based on provided configurations,
    allowing them to be 'learnable' (as Variables) or fixed values.
    K (Hydraulic Conductivity) and Ss (Specific Storage) are typically
    positive and are handled in log-space if learnable to ensure positivity.
    Q (Source/Sink term) can be positive or negative.

    Parameters
    ----------
    gw_flow_coeffs : Optional[Dict[str, Union[str, float, None]]], default None
        A dictionary configuring the groundwater flow coefficients.
        Expected keys are 'K', 'Ss', 'Q'.
        For each key, the value can be:
        - 'learnable': The coefficient will be a trainable `Variable`.
        - float: The coefficient will be a fixed value.
        - None: The coefficient is considered not active or will use a
                default (typically 0 for Q). If K or Ss are essential for a PDE
                and set to None, their respective get methods will return None,
                which should be handled by the PDE computation logic.
        Example:
        `{'K': 'learnable', 'Ss': 1e-5, 'Q': 0.0}`
    default_K : float, default 1e-4
        Default initial value for K if 'learnable'.
    default_Ss : float, default 1e-5
        Default initial value for Ss if 'learnable'.
    default_Q : float, default 0.0
        Default initial value for Q if 'learnable' or if Q config is None.
    name_prefix : str, default "gw_flow_coeffs"
        A prefix for the names of `Variable`s created for learnable
        coefficients, to help with identification in a larger model.

    Attributes
    ----------
    log_K_var : Union[Variable, Tensor, None]
        Stores log(K) if K is learnable, or a constant tensor of log(K_fixed),
        or None. K is retrieved via `get_K()`.
    log_Ss_var : Union[Variable, Tensor, None]
        Stores log(Ss) if Ss is learnable, or a constant tensor of log(Ss_fixed),
        or None. Ss is retrieved via `get_Ss()`.
    Q_var : Union[Variable, Tensor, None]
        Stores Q if learnable, or a constant tensor of Q_fixed.
        If Q config is None, it defaults to a constant 0.0.
        Q is retrieved via `get_Q()`.
    """
    def __init__(
        self,
        gw_flow_coeffs: Optional[Dict[str, Union[str, float, None]]] = None,
        default_K: float = 1e-4,
        default_Ss: float = 1e-5,
        default_Q: float = 0.0,
        name_prefix: str = "gw_flow_coeffs" # Changed default prefix
    ):
        self.gw_flow_coeffs_config = gw_flow_coeffs if gw_flow_coeffs is not None else {}
        self.name_prefix = name_prefix

        # Initialize coefficients
        # If a coeff is not in config, it might be treated as 'None' by _initialize_coefficient
        # or could default to 'learnable' if explicitly desired.
        # Current _initialize_coefficient handles 'None' by returning None.
        k_config = self.gw_flow_coeffs_config.get('K') # No default to 'learnable' here
        ss_config = self.gw_flow_coeffs_config.get('Ss')
        q_config = self.gw_flow_coeffs_config.get('Q') # Q will default to 0.0 if None in _initialize

        self.log_K_var = self._initialize_coefficient(
            coeff_name='K',
            config_value=k_config,
            default_initial_value=default_K,
            is_log_space=True,
            is_positive_only=True
        )
        self.log_Ss_var = self._initialize_coefficient(
            coeff_name='Ss',
            config_value=ss_config,
            default_initial_value=default_Ss,
            is_log_space=True,
            is_positive_only=True
        )
        self.Q_var = self._initialize_coefficient(
            coeff_name='Q',
            config_value=q_config,
            default_initial_value=default_Q,
            is_log_space=False,
            is_positive_only=False, # Q can be negative (sink)
            default_if_none=0.0 # Q defaults to 0 if config is None
        )

    def _initialize_coefficient(
        self,
        coeff_name: str,
        config_value: Union[str, float, None],
        default_initial_value: float,
        is_log_space: bool,
        is_positive_only: bool,
        default_if_none: Optional[float] = None # For Q to default to 0.0
    ) -> Union[Variable, Tensor, None]:
        """
        Helper to create a Variable or Tensor for a coefficient.
        """
        variable_name = f"{self.name_prefix}_{coeff_name}"
        if is_log_space:
            variable_name = f"log_{variable_name}"

        if config_value is None and default_if_none is not None:
            config_value = default_if_none # e.g., Q becomes fixed 0.0
            logger.info(
                f"Coefficient '{coeff_name}' not configured, defaulting to "
                f"fixed value: {default_if_none}."
            )

        if config_value == 'learnable':
            initial_val_for_var = default_initial_value
            if is_log_space:
                if initial_val_for_var <= 0:
                    logger.warning(
                        f"Initial value for log-space learnable coefficient "
                        f"'{coeff_name}' ({initial_val_for_var}) must be positive. "
                        f"Using small epsilon (1e-9) instead."
                    )
                    initial_val_for_var = 1e-9 # Ensure positivity for log
                initial_val_for_var = tf_log(
                    tf_cast(initial_val_for_var, dtype=tf_float32))
            
            return Variable(
                initial_value=initial_val_for_var,
                trainable=True,
                name=variable_name,
                dtype=tf_float32
            )
        elif isinstance(config_value, (float, int)):
            val_to_store = float(config_value)
            if is_log_space:
                if is_positive_only and val_to_store <= 0:
                    logger.warning(
                        f"Fixed coefficient '{coeff_name}' ({val_to_store}) "
                        f"is expected to be positive but is not. "
                        f"Using log(abs(value) + epsilon) for stability."
                    )
                    val_to_store = tf_log(tf_abs(val_to_store) + 1e-9) # Use abs and epsilon
                elif val_to_store <=0: # is_log_space but not necessarily positive_only (though unusual for K, Ss)
                     raise ValueError (
                         f"Fixed coefficient '{coeff_name}' ({val_to_store}) "
                         f"cannot be non-positive if stored in log-space."
                     )
                else:
                    val_to_store = tf_log(tf_cast(val_to_store, dtype=tf_float32))
            return tf_constant(val_to_store, dtype=tf_float32)
        elif config_value is None:
            # This path is now only taken if default_if_none was also None (e.g., for K, Ss)
            logger.info(
                f"Coefficient '{coeff_name}' is not configured (None) and has no "
                f"automatic default. It will be None."
            )
            return None
        else:
            raise ValueError(
                f"Invalid configuration for coefficient '{coeff_name}': {config_value}. "
                "Expected 'learnable', a number, or None."
            )

    def get_K(self) -> Optional[Tensor]:
        """
        Returns Hydraulic Conductivity K as a positive TensorFlow tensor.
        Returns None if K was not configured to have a value.
        """
        if self.log_K_var is None:
            return None
        return tf_exp(self.log_K_var)

    def get_Ss(self) -> Optional[Tensor]:
        """
        Returns Specific Storage Ss as a positive TensorFlow tensor.
        Returns None if Ss was not configured to have a value.
        """
        if self.log_Ss_var is None:
            return None
        return tf_exp(self.log_Ss_var)

    def get_Q(self) -> Optional[Tensor]:
        """
        Returns Source/Sink term Q as a TensorFlow tensor.
        Returns None if Q was not configured to have a value (and no default_if_none).
        Q_var will be a tf_constant(0.0) if q_config was None and default_if_none=0.0 was used.
        """
        # _initialize_coefficient now ensures Q_var is a Tensor or Variable if not None
        return self.Q_var


    def get_all_coeffs(self) -> Dict[str, Optional[Tensor]]:
        """
        Returns a dictionary of all configured groundwater flow coefficients.
        Values can be None if a coefficient was not configured.
        """
        return {
            "K": self.get_K(),
            "Ss": self.get_Ss(),
            "Q": self.get_Q()
        }
    
    @property
    def trainable_variables(self) -> List[Variable]:
        """Returns a list of trainable Variables managed by this calculator."""
        variables = []
        if isinstance(self.log_K_var, Variable):
            variables.append(self.log_K_var)
        if isinstance(self.log_Ss_var, Variable):
            variables.append(self.log_Ss_var)
        if isinstance(self.Q_var, Variable):
            variables.append(self.Q_var)
        return variables

    def __repr__(self) -> str:
        # Using .numpy() can cause issues if called during graph construction.
        # Better to represent based on variable type for learnable, or config for fixed.
        def get_val_repr(var, config_val, is_log):
            if isinstance(var, Variable):
                return 'learnable'
            elif var is not None: # Tensor (fixed)
                if is_log:
                    try: return f"{tf_exp(var).numpy():.2e}" # Show actual value
                    except: return "fixed_log_value"
                else:
                    try: return f"{var.numpy()}"
                    except: return "fixed_value"
            return 'None'

        k_repr = get_val_repr(self.log_K_var, self.gw_flow_coeffs_config.get('K'), True)
        ss_repr = get_val_repr(self.log_Ss_var, self.gw_flow_coeffs_config.get('Ss'), True)
        q_repr = get_val_repr(self.Q_var, self.gw_flow_coeffs_config.get('Q'), False)

        return (
            f"{self.__class__.__name__}("
            f"K={k_repr}, Ss={ss_repr}, Q={q_repr})"
        )

class GroundwaterFlowPDEResidual(Layer):
    """
    A Keras Layer to compute the 2D groundwater flow PDE residual.

    The PDE is: Ss * ∂h/∂t - Kx * ∂²h/∂x² - Ky * ∂²h/∂y² - Q = 0.
    For simplicity, this implementation assumes isotropic hydraulic
    conductivity (Kx = Ky = K).

    This layer uses automatic differentiation via `GradientTape` to
    compute the necessary derivatives of the predicted hydraulic head `h`.
    The physical coefficients (Ss, K, Q) are managed by an internal
    `GWResidualCalculator` instance.

    Parameters
    ----------
    gw_flow_coeffs_config : Optional[Dict[str, Union[str, float, None]]], default None
        Configuration dictionary for `GWResidualCalculator` to define
        Ss, K, and Q. See `GWResidualCalculator` for details.
    name : str, default "groundwater_pde_residual"
        Name of the layer.
    **kwargs :
        Additional keyword arguments passed to `Layer`.

    Call Arguments
    --------------
    inputs : Tuple[Tensor, Tensor, Tensor, Tensor]
        A tuple containing:
        - `h_pred` (Tensor): Predicted hydraulic head at collocation points.
          Shape: `(batch_size, num_points, 1)`. It is crucial that `h_pred`
          is a differentiable function of `t_coords`, `x_coords`, `y_coords`
          within the TensorFlow graph, meaning `t,x,y` were inputs to the
          network part that produced `h_pred`.
        - `t_coords` (Tensor): Temporal coordinates of collocation points.
          Shape: `(batch_size, num_points, 1)`.
        - `x_coords` (Tensor): Spatial x-coordinates.
          Shape: `(batch_size, num_points, 1)`.
        - `y_coords` (Tensor): Spatial y-coordinates.
          Shape: `(batch_size, num_points, 1)`.

    Returns
    -------
    Tensor
        The computed PDE residual at each collocation point.
        Shape: `(batch_size, num_points, 1)`.
    """
    def __init__(
        self,
        gw_flow_coeffs_config: Optional[Dict[str, Union[str, float, None]]] = None,
        name: str = "groundwater_pde_residual",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.gw_coeffs_config = gw_flow_coeffs_config
        # The calculator will create its own Variables if 'learnable'
        # These will be automatically tracked by this Keras Layer because
        # the calculator instance is an attribute of this Layer.
        self.coeff_calculator = GWResidual(
            gw_flow_coeffs=self.gw_coeffs_config,
            name_prefix=f"{self.name}_coeffs" # Ensure unique variable names
        )
        # Expose the calculator's trainable variables to this layer's properties
        # This makes them part of this layer's trainable_variables.
        # Note: Keras layers usually discover Variables in attributes.
        # Explicitly adding them to _trainable_weights is more robust if they
        # are in a plain Python object that is an attribute.
        # However, GWResidualCalculator initializes Variables which Keras tracks.

    def call(
        self,
        inputs: Tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> Tensor:
        """
        Computes the groundwater flow PDE residual.
        inputs = (h_pred, t_coords, x_coords, y_coords)
        """
        if not (isinstance(inputs, tuple) and len(inputs) == 4):
            raise ValueError(
                "Inputs must be a tuple of four tensors: "
                "(h_pred, t_coords, x_coords, y_coords)."
            )
        h_pred, t_coords, x_coords, y_coords = inputs

        # Ensure inputs are float32 for gradient computation
        t_coords = tf_cast(t_coords, tf_float32)
        x_coords = tf_cast(x_coords, tf_float32)
        y_coords = tf_cast(y_coords, tf_float32)
        h_pred = tf_cast(h_pred, tf_float32) # h_pred should already be from model

        # Compute derivatives using GradientTape
        # It's crucial that h_pred is a result of operations on t_coords,
        # x_coords, y_coords that the tape is watching.
        # This means if h_pred is passed directly, the model that produced h_pred
        # must have been called with t_coords, x_coords, y_coords as watched inputs.
        
        with GradientTape(
                persistent=True, 
                watch_accessed_variables=False
                ) as tape2:
            tape2.watch(x_coords)
            tape2.watch(y_coords)
            with GradientTape(
                    persistent=True, watch_accessed_variables=False
                    ) as tape1:
                tape1.watch(t_coords)
                tape1.watch(x_coords)
                tape1.watch(y_coords)
                # Re-evaluate h_pred if it's from a sub-model taking t,x,y
                # For now, assume h_pred is already the tensor to differentiate.
                # This implies the model part producing h_pred was called with
                # these specific t_coords, x_coords, y_coords.
                
                # If h_pred is truly a function of these specific coordinate instances:
                # (This is a strong assumption on how PIHALNet would use this layer.
                # PIHALNet's main `call` would compute `h_pred` using its features,
                # and then pass `h_pred` and the *original input coordinates* here.
                # For AD to work, `h_pred` must be differentiable wrt these *input* coords.)

                # To make h_pred differentiable wrt the *current instances* of t,x,y
                # that the tape is watching, h_pred ideally should be re-calculated
                # using a model that explicitly takes these t,x,y as inputs here.
                # For now, let's assume h_pred is passed in such a way its graph
                # retains differentiability w.r.t. the t,x,y inputs it originated from.

                # Get first-order derivatives
                dh_dt = tape1.gradient(h_pred, t_coords)
                dh_dx = tape1.gradient(h_pred, x_coords)
                dh_dy = tape1.gradient(h_pred, y_coords)

        # Handle potential None gradients (if h_pred doesn't depend on a coordinate)
        dh_dt = dh_dt if dh_dt is not None else tf_zeros_like(h_pred)
        dh_dx = dh_dx if dh_dx is not None else tf_zeros_like(h_pred)
        dh_dy = dh_dy if dh_dy is not None else tf_zeros_like(h_pred)
        
        # Get second-order derivatives
        d2h_dx2 = tape2.gradient(dh_dx, x_coords)
        d2h_dy2 = tape2.gradient(dh_dy, y_coords)
        
        del tape1 # Delete tapes once done
        del tape2

        d2h_dx2 = d2h_dx2 if d2h_dx2 is not None else tf_zeros_like(h_pred)
        d2h_dy2 = d2h_dy2 if d2h_dy2 is not None else tf_zeros_like(h_pred)

        # Retrieve physical coefficients
        Ss = self.coeff_calculator.get_Ss()
        K = self.coeff_calculator.get_K() # Assuming isotropic Kx=Ky=K
        Q = self.coeff_calculator.get_Q()

        # Handle cases where coefficients might be None (not configured)
        if Ss is None:
            logger.warning(
                f"Specific Storage (Ss) is None in {self.name}."
                " Term Ss*dh/dt will be zero.")
            Ss = tf_constant(0.0, dtype=tf_float32)
        if K is None:
            logger.warning(
                f"Hydraulic Conductivity (K) is None in {self.name}."
                " Laplacian term will be zero.")
            K = tf_constant(0.0, dtype=tf_float32)
        if Q is None: # Q defaults to 0.0 in GWResidualCalculator if config is None
            Q = tf_constant(0.0, dtype=tf_float32)


        # Assemble the PDE residual: Ss * ∂h/∂t - K * (∂²h/∂x² + ∂²h/∂y²) - Q = 0
        # The residual is what we want to drive to zero.
        term1 = Ss * dh_dt
        term2_spatial_laplacian = K * (d2h_dx2 + d2h_dy2)
        
        pde_residual = term1 - term2_spatial_laplacian - Q
        
        return pde_residual

    def get_config(self):
        config = super().get_config()
        config.update({
            "gw_flow_coeffs_config": self.gw_coeffs_config,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # GWResidualCalculator is created internally, config only needs its setup.
        return cls(**config)


class GWResidualCalculator (BaseClass):
    """
    Manages and provides physical coefficients (K, Ss, Q) for groundwater
    flow PDE calculations.

    This class initializes coefficients based on provided configurations,
    allowing them to be 'learnable' (as Variables) or fixed values.
    K (Hydraulic Conductivity) and Ss (Specific Storage) are typically
    positive and are handled in log-space if learnable to ensure positivity.
    Q (Source/Sink term) can be positive or negative.

    Parameters
    ----------
    gw_flow_coeffs : Optional[Dict[str, Union[str, float, None]]], default None
        A dictionary configuring the groundwater flow coefficients.
        Expected keys are 'K', 'Ss', 'Q'.
        For each key, the value can be:
        - 'learnable': The coefficient will be a trainable `Variable`.
        - float: The coefficient will be a fixed value.
        - None: The coefficient is considered not active or will use a
                default (typically 0 for Q, or might raise error if essential
                and None for K, Ss).
        Example:
        `{'K': 'learnable', 'Ss': 1e-5, 'Q': 0.0}`
    default_K : float, default 1e-4
        Default initial value for K if 'learnable' and no specific initial
        value is provided through a more complex config (future extension).
    default_Ss : float, default 1e-5
        Default initial value for Ss if 'learnable'.
    default_Q : float, default 0.0
        Default initial value for Q if 'learnable'.
    name_prefix : str, default "gw_flow"
        A prefix for the names of `Variable`s created for learnable
        coefficients, to help with identification in a larger model.

    Attributes
    ----------
    log_K_var : Union[Variable, Tensor, None]
        Stores log(K) if K is learnable, or a constant tensor of log(K_fixed),
        or None. K is retrieved via `get_K()`.
    log_Ss_var : Union[Variable, Tensor, None]
        Stores log(Ss) if Ss is learnable, or a constant tensor of log(Ss_fixed),
        or None. Ss is retrieved via `get_Ss()`.
    Q_var : Union[Variable, Tensor, None]
        Stores Q if learnable, or a constant tensor of Q_fixed, or None.
        Q is retrieved via `get_Q()`.
    """
    def __init__(
        self,
        gw_flow_coeffs: Optional[Dict[str, Union[str, float, None]]] = None,
        default_K: float = 1e-4,      # m/s (typical range for aquifers)
        default_Ss: float = 1e-5,     # 1/m (typical range for aquifers)
        default_Q: float = 0.0,       # m/s (source/sink per unit volume)
        name_prefix: str = "gw_flow"
    ):
        self.gw_flow_coeffs_config = gw_flow_coeffs if gw_flow_coeffs is not None else {}
        self.name_prefix = name_prefix

        # Initialize coefficients
        k_config = self.gw_flow_coeffs_config.get('K', 'learnable') # Default to learnable if not specified
        ss_config = self.gw_flow_coeffs_config.get('Ss', 'learnable')
        q_config = self.gw_flow_coeffs_config.get('Q', 0.0) # Default Q to 0.0 if not specified

        self.log_K_var = self._initialize_coefficient(
            coeff_name='K',
            config_value=k_config,
            default_initial_value=default_K,
            is_log_space=True,
            is_positive_only=True
        )
        self.log_Ss_var = self._initialize_coefficient(
            coeff_name='Ss',
            config_value=ss_config,
            default_initial_value=default_Ss,
            is_log_space=True,
            is_positive_only=True
        )
        self.Q_var = self._initialize_coefficient(
            coeff_name='Q',
            config_value=q_config,
            default_initial_value=default_Q,
            is_log_space=False,
            is_positive_only=False # Q can be negative (sink)
        )

    def _initialize_coefficient(
        self,
        coeff_name: str,
        config_value: Union[str, float, None],
        default_initial_value: float,
        is_log_space: bool,
        is_positive_only: bool
    ) -> Union[Variable, Tensor, None]:
        """
        Helper to create a Variable or Tensor for a coefficient.
        """
        variable_name = f"{self.name_prefix}_{coeff_name}"
        if is_log_space:
            variable_name = f"log_{variable_name}"

        if config_value == 'learnable':
            initial_val = default_initial_value
            if is_log_space:
                if initial_val <= 0:
                    logger.warning(
                        f"Initial value for log-space learnable coefficient "
                        f"'{coeff_name}' ({initial_val}) must be positive. "
                        f"Using small epsilon (1e-9) instead."
                    )
                    initial_val = 1e-9 # Ensure positivity for log
                initial_val = tf_log(tf_cast(initial_val, dtype=tf_float32))
            
            return Variable(
                initial_value=initial_val,
                trainable=True,
                name=variable_name,
                dtype=tf_float32
            )
        elif isinstance(config_value, (float, int)):
            val_to_store = float(config_value)
            if is_log_space:
                if is_positive_only and val_to_store <= 0:
                    logger.warning(
                        f"Fixed coefficient '{coeff_name}' ({val_to_store}) "
                        f"is expected to be positive but is not. "
                        f"Using log(abs(value) + epsilon) for stability."
                    )
                    val_to_store = tf_log(tf_abs(val_to_store) + 1e-9)
                elif val_to_store <=0 : # not positive_only but log_space
                     raise ValueError (
                         f"Fixed coefficient '{coeff_name}' ({val_to_store}) "
                         f"cannot be non-positive in log-space."
                     )
                else:
                    val_to_store = tf_log(tf_cast(val_to_store, dtype=tf_float32))
            return tf_constant(val_to_store, dtype=tf_float32)
        elif config_value is None:
            logger.info(
                f"Coefficient '{coeff_name}' is not configured (None). "
                "It will not be available unless a default is implicitly handled."
            )
            return None
        else:
            raise ValueError(
                f"Invalid configuration for coefficient '{coeff_name}': {config_value}. "
                "Expected 'learnable', a number, or None."
            )

    def get_K(self) -> Optional[Tensor]:
        """
        Returns the Hydraulic Conductivity K as a positive TensorFlow tensor.
        Returns None if K was not configured.
        """
        if self.log_K_var is None:
            return None
        return tf_exp(self.log_K_var)

    def get_Ss(self) -> Optional[Tensor]:
        """
        Returns the Specific Storage Ss as a positive TensorFlow tensor.
        Returns None if Ss was not configured.
        """
        if self.log_Ss_var is None:
            return None
        return tf_exp(self.log_Ss_var)

    def get_Q(self) -> Optional[Tensor]:
        """
        Returns the Source/Sink term Q as a TensorFlow tensor.
        Returns None if Q was not configured.
        """
        if self.Q_var is None:
            return None
        # If Q_var is a Variable or Tensor, it's returned directly.
        # If it was stored as a float (e.g., if config was a fixed number and not log-space),
        # ensure it's a tensor for consistency, though _initialize_coefficient
        # should already handle this by returning tf_constant for fixed values.
        if isinstance(self.Q_var, (float, int)): # Should ideally not happen with current _initialize
            return tf_constant(float(self.Q_var), dtype=tf_float32)
        return self.Q_var

    def get_all_coeffs(self) -> Dict[str, Optional[Tensor]]:
        """
        Returns a dictionary of all configured groundwater flow coefficients.
        """
        return {
            "K": self.get_K(),
            "Ss": self.get_Ss(),
            "Q": self.get_Q()
        }

    def __repr__(self) -> str:
        k_val = self.get_K()
        ss_val = self.get_Ss()
        q_val = self.get_Q()
        return (
            f"{self.__class__.__name__}("
            f"K={'learnable' if isinstance(self.log_K_var, Variable) else (k_val.numpy() if k_val is not None else None)}, "
            f"Ss={'learnable' if isinstance(self.log_Ss_var, Variable) else (ss_val.numpy() if ss_val is not None else None)}, "
            f"Q={'learnable' if isinstance(self.Q_var, Variable) else (q_val.numpy() if q_val is not None else None)}"
            ")"
        )


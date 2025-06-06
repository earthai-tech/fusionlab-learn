# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

from __future__ import annotations

from typing import Optional, Union, Dict,List, Tuple  

from ..._fusionlog import fusionlog 
from ...api.property import NNLearner, BaseClass  
from ...params import ( 
    LearnableK, 
    LearnableSs, 
    LearnableQ, 
    resolve_physical_param
)
from ...utils.deps_utils import ensure_pkg 
from .. import KERAS_DEPS, KERAS_BACKEND, dependency_message
from .op import compute_gw_flow_residual 
from .utils import extract_txy 

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
    

DEP_MSG = dependency_message('nn.pinn.base') 

class GWFlowPINN(Model, NNLearner):
    """
    A small PINN for 2D transient groundwater‐flow:
      h = h(t, x, y) predicted by an MLP, plus a PDE‐residual method.

    Usage:
      >>> from fusionlab.nn.pinn.base import GWFlowPINN
      >> model = GWFlowPINN(
      ...     hidden_units=[32, 32],
      ...     learning_rate=1e-3,
      ...     K=LearnableK(initial_value=0.5),
      ...     Ss=LearnableSs(initial_value=1e-4),
      ...     Q=LearnableQ(initial_value=0.0)
      ... )
      >> # Build on a dummy input batch to initialize weights:
      >> dummy = tf.random.normal((1, 3))  # shape = (batch, [t,x,y])
      >> _ = model(dummy)
      >> # Now compute residual on real collocation points:
      >> coords = {
      ...     "t": tf.Variable(tf.random.uniform((16,1)), trainable=True),
      ...     "x": tf.Variable(tf.random.uniform((16,1)), trainable=True),
      ...     "y": tf.Variable(tf.random.uniform((16,1)), trainable=True)
      ... }
      >> residual = model.compute_residual(coords=coords)
      >>> from fusionlab.params import ( 
          LearnableK, 
          LearnableSs, 
          LearnableQ, 
          resolve_physical_param
      )
      >>> from fusionlab.nn.pinn.base import GWFlowPINN
      >>> model = GWFlowPINN(
      ...     hidden_units=[32, 32],
      ...     learning_rate=1e-3,
      ...     K=LearnableK(initial_value=0.5),
      ...     Ss=LearnableSs(initial_value=1e-4),
      ...     Q=LearnableQ(initial_value=0.0)
      ... )
      >>> dummy = tf.random.normal((1, 3))  # shape = (batch, [t,x,y])

      >>> _ = model(dummy)
      >>> coords = {
      ...     "t": tf.Variable(tf.random.uniform((16,1)), trainable=True),
      ...     "x": tf.Variable(tf.random.uniform((16,1)), trainable=True),
      ...     "y": tf.Variable(tf.random.uniform((16,1)), trainable=True)
      ... }

      residual = model.compute_residual(coords=coords)
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)   
    def __init__(
        self,
        hidden_units: Optional[list[int]] = None,
        activation: str = "tanh",
        learning_rate: float = 1e-3,
        K: Union[float, LearnableK] = 1.0,
        Ss: Union[float, LearnableSs] = 1e-4,
        Q: Union[float, LearnableQ] = 0.0,
        name: str = "GWFlowPINN",
        **kwargs
    ):
        super().__init__(**kwargs)

        # Store physical parameters (float or Learnable)
        self.K_param = K
        self.Ss_param = Ss
        self.Q_param = Q

        if hidden_units is None:
            hidden_units = [32, 32]

        # Build a small MLP to predict h = h(t,x,y)
        self._layers_list = []
        # We do not use an InputLayer explicitly because we will
        # build when the first call comes in. Still, for clarity:
        self._layers_list.append(InputLayer(input_shape=(3,)))

        for units in hidden_units:
            self._layers_list.append(
                Dense(units, activation=activation)
            )

        # Final output layer: single scalar prediction h
        self._layers_list.append(
            Dense(1, activation="linear", name="h_pred"))

        # Create a tf.keras.Sequential internally
        self.net = Sequential(
            self._layers_list, name="GWFlow_Net")

        # Optimizer for training the network parameters
        self.optimizer = Adam(
            learning_rate=learning_rate
        )

        # Compile with a dummy loss—actual training
        # will minimize the PDE residual
        self.compile(
            optimizer=self.optimizer,
            loss="mse",  # placeholder (we’ll override in train_step)
            metrics=[]
        )

    def call(self, inputs:Tensor, training: bool = False) -> Tensor:
        """
        Forward pass of the network: (t, x, y) → h_pred.

        Parameters
        ----------
        inputs : tf.Tensor
            A 2D tensor of shape (batch_size, 3), where each row is
            [t, x, y]. Dtype must be float32 or compatible.
        training : bool
            Whether the network is in training mode (unused here but required
            by the Keras Model API).

        Returns
        -------
        tf.Tensor
            A tensor of shape (batch_size, 1) containing predicted
            hydraulic head values.
        """
        # Basic shape‐check: inputs must be (batch, 3)
        if inputs.ndim != 2 or inputs.shape[-1] != 3:
            raise ValueError(
                f"`inputs` must be a 2D tensor of shape (batch,3). "
                f"Received shape: {inputs.shape}"
            )
        return self.net(inputs, training=training)

    def compute_residual(
        self,
        coords: Dict[str, Tensor],
        K: Union[float, Tensor, LearnableK, None] = None,
        Ss: Union[float, Tensor, LearnableSs, None] = None,
        Q: Union[float, Tensor, LearnableQ, None] = None,
        h_pred: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute the 2D transient groundwater‐flow residual:
          R = K*(h_xx + h_yy) + Q – Ss * (∂h/∂t)

        Parameters
        ----------
        coords : dict
            Must contain keys 't', 'x', and 'y'. Each value is a tf.Tensor
            (shape = (batch_size, 1)) that is watched by GradientTape.
        K : float, tf.Tensor, or LearnableK, optional
            Hydraulic conductivity or a learnable variable. Defaults to
            the instance’s `self.K_param`.
        Ss : float, tf.Tensor, or LearnableSs, optional
            Specific storage coefficient. Defaults to `self.Ss_param`.
        Q : float, tf.Tensor, or LearnableQ, optional
            Source/sink term. Defaults to `self.Q_param`.
        h_pred : tf.Tensor, optional
            If provided, skip calling the model and use this precomputed
            head prediction of shape (batch_size, 1). Otherwise, the method
            will call `self(...)` internally.

        Returns
        -------
        tf.Tensor
            The PDE residual at each sample; shape = (batch_size, 1).

        Raises
        ------
        ValueError
            If coords does not contain 't', 'x', or 'y', or if gradients
            cannot be computed (e.g. because coords were not watched).
        """
        
        # Validate coords dictionary
        if isinstance (coords, dict): 
            if 'coords' in coords : 
                t, x, y = extract_txy(inputs = coords )
            else: 
                if not all(k in coords for k in ("t", "x", "y")):
                    raise ValueError("`coords` must contain keys 't', 'x', and 'y'.")
        
                t = coords["t"]
                x = coords["x"]
                y = coords["y"]
        else: 
            # extract analyways if tensor is given 
            t, x, y = extract_txy(inputs = coords ) 

        # Determine physical parameters: use passed or defaults
        K_val = resolve_physical_param(
            K) if K is not None else resolve_physical_param(self.K_param)
        Ss_val = resolve_physical_param(
            Ss) if Ss is not None else resolve_physical_param(self.Ss_param)
        Q_val = resolve_physical_param(
            Q) if Q is not None else resolve_physical_param(self.Q_param)

        # Use the helper from fusionlab to compute the residual
        return compute_gw_flow_residual(
            model=self,
            coords={"t": t, "x": x, "y": y},
            K=K_val,
            Ss=Ss_val,
            Q=Q_val,
            h_pred=h_pred
        )

    def train_step(
        self,
        data: tuple[Dict[str, Tensor], Tensor]
    ) -> Dict[str, Tensor]:
        """
        Custom training step that minimizes PDE residual MSE at collocation points.

        Expects `data` to be a tuple (coords_dict, dummy_targets). We ignore
        dummy_targets because PINNs often have separate data‐loss terms. Here
        we only drive PDE residual → 0.

        Parameters
        ----------
        data : (inputs, _)
            - inputs: a dict {"t": tf.Tensor, "x": tf.Tensor, "y": tf.Tensor}
              each of shape (batch_size, 1).
            - _: ignored placeholder.

        Returns
        -------
        metrics : dict
            Dictionary with key "pde_loss" = mean squared PDE residual.
        """
        coords_batch, _ = data

        with GradientTape() as tape:
            # Compute the PDE residual for this batch using instance parameters
            R = self.compute_residual(
                coords=coords_batch,
                K=coords_batch.get("K", None),
                Ss=coords_batch.get("Ss", None),
                Q=coords_batch.get("Q", None),
                h_pred=None
            )
            # PDE‐loss = MSE( R, 0 )
            pde_loss = tf_reduce_mean(tf_square(R))

        # Compute gradients w.r.t. network trainable variables
        grads = tape.gradient(pde_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"pde_loss": pde_loss}

    def test_step(
        self,
        data: tuple[Dict[str,Tensor], Tensor]
    ) -> Dict[str, Tensor]:
        """
        Custom validation/inference step. Computes PDE‐residual on validation coords.

        Parameters
        ----------
        data : (coords_dict, _)
            coords_dict: same as in train_step.

        Returns
        -------
        metrics : dict
            {"pde_loss": MSE of residual on val batch}
        """
        coords_batch, _ = data
        R = self.compute_residual(
            coords=coords_batch,
            K=coords_batch.get("K", None),
            Ss=coords_batch.get("Ss", None),
            Q=coords_batch.get("Q", None),
            h_pred=None
        )
        val_loss = tf_reduce_mean(tf_square(R))
        return {"pde_loss": val_loss}
    
    def get_config(self) -> dict:
        """
        Returns the configuration of the GWFlowPINN for serialization.
    
        The returned dict contains all arguments necessary to reconstruct
        this instance via `from_config`.  Physical parameters (K, Ss, Q)
        are stored either as raw floats or, if they are learnable instances,
        with their `initial_value` and a flag indicating learnability.
        """
        base_config = super().get_config()
        # Serialize hidden_units, activation, learning_rate
        config = {
            "hidden_units": self._layers_list[1].units
                             if self._layers_list and isinstance(
                                 self._layers_list[1], Dense
                             ) else None,
            "activation": self._layers_list[1].activation.__name__
                           if self._layers_list and isinstance(
                               self._layers_list[1], Dense
                           ) else None,
            "learning_rate": float(self.optimizer.learning_rate.numpy()),
        }
    
        # Helper to serialize a physical parameter (float or Learnable*)
        def _serialize_param(param, cls_name: str):
            if hasattr(param, "initial_value"):
                return {
                    "learnable": True,
                    "initial_value": float(param.initial_value),
                    "class": cls_name
                }
            else:
                return {"learnable": False, "initial_value": float(param)}
    
        config["K"] = _serialize_param(self.K_param, "LearnableK")
        config["Ss"] = _serialize_param(self.Ss_param, "LearnableSs")
        config["Q"] = _serialize_param(self.Q_param, "LearnableQ")
    
        # Include name (Model base class already serializes name)
        config["name"] = self.name
    
        base_config.update(config)
        return base_config
    
    @classmethod
    def from_config(cls, config: dict) -> "GWFlowPINN":
        """
        Reconstructs a GWFlowPINN instance from its `config` dictionary.
    
        Expects exactly the format produced by `get_config`.  Re‐instantiates
        any learnable physical parameters as LearnableK, LearnableSs, or LearnableQ.
        """
        # Pop fields specific to this class
        hidden_units = config.pop("hidden_units", None)
        activation = config.pop("activation", "tanh")
        learning_rate = config.pop("learning_rate", 1e-3)
    
        def _deserialize_param(param_dict, learnable_cls):
            if param_dict.get("learnable", False):
                return learnable_cls(initial_value=param_dict["initial_value"])
            else:
                return float(param_dict["initial_value"])
    
        K_dict = config.pop("K", {"learnable": False, "initial_value": 1.0})
        Ss_dict = config.pop("Ss", {"learnable": False, "initial_value": 1e-4})
        Q_dict = config.pop("Q", {"learnable": False, "initial_value": 0.0})
    
        K_param = _deserialize_param(K_dict, LearnableK)
        Ss_param = _deserialize_param(Ss_dict, LearnableSs)
        Q_param = _deserialize_param(Q_dict, LearnableQ)
    
        # The remaining keys in config are passed to super().from_config
        name = config.pop("name", "GWFlowPINN")
        # Build the instance
        instance = cls(
            hidden_units=hidden_units,
            activation=activation,
            learning_rate=learning_rate,
            K=K_param,
            Ss=Ss_param,
            Q=Q_param,
            name=name,
            **config
        )
        return instance


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


 
from typing import Dict, Optional, Any , Union,  TYPE_CHECKING 
from typing import Tuple, List 

from ..._fusionlog import fusionlog 
from ...api.docs import DocstringComponents, _pinn_tuner_common_params 
from ...utils.generic_utils import ( 
    vlog, rename_dict_keys, cast_multiple_bool_params  )
from ...core.handlers import _get_valid_kwargs 

from .. import KERAS_BACKEND, KERAS_DEPS 

from ._tuning import PINNTunerBase 
from .models import PIHALNet 
from .utils import  ( # noqa
    prepare_pinn_data_sequences, 
    check_required_input_keys
)
import numpy as np 

try:
    import keras_tuner as kt
    HAS_KT = True
except ImportError:
    # fallback *only* for runtime
    class _DummyTuner:  
        pass
    # minimal fake module
    class _DummyKT: 
        Tuner = _DummyTuner

    kt = _DummyKT()  # type: ignore[misc]

# ---- for static type‑checkers ----
if TYPE_CHECKING:
    # mypy / pyright will see the real names
    import keras_tuner as kt  # noqa: F811  (shadowing on purpose)

if KERAS_BACKEND: 
    Model =KERAS_DEPS.Model 
    Adam =KERAS_DEPS.Adam
    MeanSquaredError =KERAS_DEPS.MeanSquaredError
    MeanAbsoluteError =KERAS_DEPS.MeanAbsoluteError
    Callback =KERAS_DEPS.Callback 
    Dataset =KERAS_DEPS.Dataset 
    AUTOTUNE =KERAS_DEPS.AUTOTUNE
    
logger = fusionlog().get_fusionlab_logger(__name__) 

# Wrap into a DocstringComponents object once
_pinn_tuner_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_pinn_tuner_common_params)
)


DEFAULT_PIHALNET_FIXED_PARAMS = {
    # "static_input_dim": 0,
    # "dynamic_input_dim": 1, # [Must be > 0]
    # "future_input_dim": 0,
    "output_subsidence_dim": 1,
    "output_gwl_dim": 1,
    "forecast_horizon": 1,
    "quantiles": None,
    "max_window_size": 10,
    "memory_size": 100,
    "scales": [1],
    "multi_scale_agg": 'last',
    "final_agg": 'last',
    "use_residuals": True,
    "use_batch_norm": False,
    "use_vsn": True, 
    "vsn_units": 32, 
    "activation": "relu", 
    "pde_mode": "consolidation",
    "pinn_coefficient_C": "learnable",
    "gw_flow_coeffs": None,
    "loss_weights": {
        'data_loss': 1.0,
        'total_loss': 0.8, 
        'physics_loss':0.8 
        } 
}

# Default case info, can be updated in fit()
DEFAULT_PIHAL_CASE_INFO = {
    "description": "PIHALNet {} forecast",
    "forecast_horizon": 1,
    "quantiles": None,
    "output_subsidence_dim": 1,
    "output_gwl_dim": 1,
    # "static_input_dim": 0,
    # "dynamic_input_dim": 0,
    # "future_input_dim": 0,
}


class PIHALTuner(PINNTunerBase):
    def __init__(
        self,
        fixed_model_params: Dict[str, Any], 
        param_space: Optional[Dict[str, Any]] = None, 
        objective: Union[str, kt.Objective] = 'val_total_loss',
        max_trials: int = 20,
        project_name: str = "PIHALNet_Tuning",
        executions_per_trial: int =1, 
        tuner_type: str ='randomsearch', 
        seed: int =None, 
        overwrite_tuner: bool=True, 
        directory: str = "pihalnet_tuner_results",
        **tuner_kwargs
    ):
        super().__init__( 
            objective=objective,
            max_trials=max_trials,
            project_name=project_name,
            directory=directory,
            executions_per_trial=executions_per_trial, 
            tuner_type=tuner_type, 
            seed=seed, 
            overwrite_tuner=overwrite_tuner, 
            **tuner_kwargs
        )
        # Store fixed_model_params and 
        # param_space_config directly on the instance
        self.fixed_model_params = fixed_model_params
        self.param_space = param_space or {}
        
        # Validate required fixed parameters after super init
        required_fixed = [
            "static_input_dim", "dynamic_input_dim", "future_input_dim",
            "output_subsidence_dim", "output_gwl_dim", "forecast_horizon"
        ]
        for req_param in required_fixed:
            if req_param not in self.fixed_model_params:
                raise ValueError(
                    f"Missing required key in `fixed_model_params`: '{req_param}'"
                )
        self._current_run_case_info = {} # Will be populated by fit if needed

    @staticmethod
    def _infer_dims_and_prepare_fixed_params(
        inputs_data: Optional[Dict[str, np.ndarray]] = None,
        targets_data: Optional[Dict[str, np.ndarray]] = None,
        user_provided_fixed_params: Optional[Dict[str, Any]] = None,
        forecast_horizon: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        default_params: Dict = DEFAULT_PIHALNET_FIXED_PARAMS,
        verbose: int = 0
    ) -> Dict[str, Any]:
        """
        Infers or sets fixed model parameters required by PIHALNet.
        Priority: user_provided_fixed_params > inferred_from_data > defaults.
        """
        final_fixed_params = default_params.copy()
        source_log = "defaults"

        # Rename keys in targets_data if necessary
        if targets_data:
            _, targets_data = check_required_input_keys(
                None, targets_data, 
                message=( 
                    "Target keys 'subs_pred' and 'gwl_pred'"
                    " are required in 'y' data." )
            )
            targets_data = rename_dict_keys(
                targets_data.copy(), # Work on a copy
                param_to_rename={"subsidence": 'subs_pred', "gwl": 'gwl_pred'}
            )
            # Check required target keys after renaming

        inferred_params = {}
        if inputs_data and targets_data:
            source_log = "inferred from data"
            # Check required input keys
            check_required_input_keys(inputs_data, targets_data)
         
            inferred_params['static_input_dim'] = \
                inputs_data['static_features'].shape[-1] \
                if inputs_data.get('static_features') is not None and \
                   inputs_data['static_features'].ndim == 2 else 0
            
            inferred_params['dynamic_input_dim'] = \
                inputs_data['dynamic_features'].shape[-1]
                
            inferred_params['future_input_dim'] = \
                inputs_data['future_features'].shape[-1] \
                if inputs_data.get('future_features') is not None and \
                   inputs_data['future_features'].ndim == 3 else 0
            
            inferred_params['output_subsidence_dim'] = targets_data['subs_pred'].shape[-1]
            inferred_params['output_gwl_dim'] = targets_data['gwl_pred'].shape[-1]
            
            # If forecast_horizon is not given, try to infer from targets
            if forecast_horizon is None:
                if targets_data['subs_pred'].ndim >=2: # (B,H,...) or (H,...)
                    fh_idx = 1 if targets_data['subs_pred'].ndim >1 else 0
                    forecast_horizon = targets_data['subs_pred'].shape[fh_idx]
                    vlog(
                        f"Inferred forecast_horizon={forecast_horizon}"
                        " from target shapes.",
                         verbose=verbose, level=3)
                else: # Fallback to default if cannot infer
                    forecast_horizon = default_params.get('forecast_horizon', 1)
                    vlog(
                        "Cannot infer forecast_horizon,"
                        f" using default={forecast_horizon}.",
                         verbose=verbose, level=2, )

            inferred_params['forecast_horizon'] = forecast_horizon
            inferred_params['quantiles'] = quantiles # Use passed quantiles or None
            
            final_fixed_params.update(inferred_params)

        # Override with explicitly provided fixed_params if any
        if user_provided_fixed_params:
            final_fixed_params.update(user_provided_fixed_params)
            source_log = ( 
                "user_provided_fixed_params" if not inputs_data 
                else "inferred_from_data & user_override"
            )
            # Ensure explicitly passed forecast_horizon and quantiles are respected
            if (
                    'forecast_horizon' in user_provided_fixed_params 
                    and forecast_horizon is not None 
                    and user_provided_fixed_params[
                        'forecast_horizon'] != forecast_horizon
                ) :
                logger.warning(
                    "Mismatch in forecast_horizon: "
                    "provided fixed_params override explicit arg.")
            elif forecast_horizon is not None:
                 final_fixed_params['forecast_horizon'] = forecast_horizon
                 
            if (
                    'quantiles' in user_provided_fixed_params 
                    and quantiles is not None 
                    and user_provided_fixed_params[
                        'quantiles'] != quantiles
                ):
                logger.warning(
                    "Mismatch in quantiles: provided "
                    "fixed_params override explicit arg.")
            elif quantiles is not None:
                 final_fixed_params['quantiles'] = quantiles


        vlog(f"Final fixed_model_params determined from: {source_log}",
             verbose=verbose, level=2)
        if verbose >=3:
            for k,v_ in final_fixed_params.items():
                vlog(f"  Fixed Param -> {k}: {v_}", verbose=verbose, level=3, )
                
        return final_fixed_params

    @classmethod
    def create(
        cls,
        # Option 1: Provide full fixed_model_params dict
        fixed_model_params: Optional[Dict[str, Any]] = None,
        # Option 2: Provide data to infer from
        inputs_data: Optional[Dict[str, np.ndarray]] = None,
        targets_data: Optional[Dict[str, np.ndarray]] = None,
        # Explicit args to guide inference or override defaults
        forecast_horizon: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        # Tuner configurations
        objective: Union[str, kt.Objective] = 'val_total_loss',
        max_trials: int = 20,
        project_name: str = "PIHALNet_Tuning_From_Config",
        directory: str = "pihalnet_tuner_results",
        param_space: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        **tuner_init_kwargs
    ) -> 'PIHALTuner':
        """
        Creates a PIHALTuner instance.

        Fixed model parameters for PIHALNet are determined with the
        following priority:
        1. Values in `fixed_model_params` if provided.
        2. Inferred from `inputs_data` and `targets_data` if provided.
        3. Default values from `DEFAULT_PIHALNET_FIXED_PARAMS`.

        Args:
            fixed_model_params (Dict, optional): A complete dictionary
                of fixed parameters for PIHALNet.
            inputs_data (Dict[str, np.ndarray], optional): Dictionary of
                NumPy input arrays to infer dimensions from.
            targets_data (Dict[str, np.ndarray], optional): Dictionary of
                NumPy target arrays to infer dimensions from.
            forecast_horizon (int, optional): Explicitly set forecast horizon.
                Used if inferring from data or if not in `fixed_model_params`.
            quantiles (List[float], optional): Explicitly set quantiles.
            objective, max_trials, ... : Tuner configuration parameters.
            verbose (int): Logging verbosity.
            **tuner_init_kwargs : Additional keyword arguments for
                                 `PINNTunerBase`.

        Returns:
            PIHALTuner: An instance of PIHALTuner.
        """
        vlog("Creating PIHALTuner instance via from_config_and_data...",
             verbose=verbose, level=1)
             
        
        actual_fixed_params = cls._infer_dims_and_prepare_fixed_params(
            inputs_data=inputs_data,
            targets_data=targets_data,
            user_provided_fixed_params=fixed_model_params,
            forecast_horizon=forecast_horizon,
            quantiles=quantiles,
            default_params=DEFAULT_PIHALNET_FIXED_PARAMS.copy(),
            verbose=verbose
        )

        return cls(
            fixed_model_params=actual_fixed_params,
            param_space=param_space,
            objective=objective,
            max_trials=max_trials,
            project_name=project_name,
            directory=directory,
            **tuner_init_kwargs
        )
    
    def build(self, hp: kt.HyperParameters) -> Model:
        """
        Builds and compiles PIHALNet. `self.fixed_model_params` must be populated first.
        """
        if not self.fixed_model_params or 'dynamic_input_dim' not in self.fixed_model_params:
            raise RuntimeError(
                "`fixed_model_params` (with inferred dimensions) must be set "
                "by calling `fit()` before the tuner calls `build()`."
            )
            
        # --- Architectural HPs ---
        embed_dim_hp = self._get_hp_int(hp, 'embed_dim', 16, 64, step=16)
        hidden_units_hp = self._get_hp_int(hp, 'hidden_units', 32, 128, step=32)
        lstm_units_hp = self._get_hp_int(hp, 'lstm_units', 32, 128, step=32)
        attention_units_hp = self._get_hp_int(hp, 'attention_units', 32, 128, step=32)
        num_heads_hp = self._get_hp_choice(hp, 'num_heads', [1, 2, 4])
        dropout_rate_hp = self._get_hp_float(hp, 'dropout_rate', 0.0, 0.3)
        activation_hp = self._get_hp_choice(hp, 'activation', ['relu', 'gelu'])
        use_vsn_hp = self._get_hp_choice(hp, 'use_vsn', [True, False])
        vsn_units_hp = self._get_hp_int(
            hp, 'vsn_units', 
            max(16, hidden_units_hp // 4), hidden_units_hp, 
            max(16, hidden_units_hp // 4)
        ) if use_vsn_hp else None
        
        # --- PINN HPs ---
        # Defaulting to 'consolidation' as 'gw_flow' is complex for current PIHALNet
        pde_mode_hp = self._get_hp_choice(hp, 'pde_mode', ['consolidation', 'none'])
        pinn_c_type = self._get_hp_choice(
            hp, 'pinn_coefficient_C_type', ['learnable', 'fixed']
        )
        pinn_c_value_hp = 'learnable' if pinn_c_type == 'learnable' else \
            self._get_hp_float(hp, 'pinn_coefficient_C_value', 1e-4, 1e-1, sampling='log')
        
        lambda_pde_hp = self._get_hp_float(hp, 'lambda_pde', 1, 10.0, sampling='log')
        learning_rate_hp = self._get_hp_choice(hp, 'learning_rate', [1e-3, 5e-4, 1e-4])

        cast_multiple_bool_params (
            self.fixed_model_params, 
            bool_params_to_cast= [('use_vsn', False), ('use_residuals', True)], 
        )
    
        model_params = {
            **self.fixed_model_params,
            "embed_dim": embed_dim_hp, 
            "hidden_units": hidden_units_hp,
            "lstm_units": lstm_units_hp,
            "attention_units": attention_units_hp,
            "num_heads": num_heads_hp, 
            "dropout_rate": dropout_rate_hp,
            "activation": activation_hp, 
            "use_vsn": use_vsn_hp,
            "vsn_units": vsn_units_hp, 
            "pde_mode": pde_mode_hp,
            "pinn_coefficient_C": pinn_c_value_hp, 
            "gw_flow_coeffs": None,
            # Defaults for other PIHALNet params if not in fixed_model_params
            "max_window_size": self.fixed_model_params.get('max_window_size', 10),
            "memory_size": self.fixed_model_params.get('memory_size', 100),
            "scales": self.fixed_model_params.get('scales', [1]),
            "multi_scale_agg": self.fixed_model_params.get('multi_scale_agg', 'last'),
            "final_agg": self.fixed_model_params.get('final_agg', 'last'),
            "use_residuals": self.fixed_model_params.get('use_residuals', True),
            "use_batch_norm": self.fixed_model_params.get('use_batch_norm', False),
        }
        model_params = _get_valid_kwargs(PIHALNet, model_params)
        model = PIHALNet(**model_params)

        loss_dict = {
            'subs_pred': MeanSquaredError(name='subs_data_loss'),
            'gwl_pred': MeanSquaredError(name='gwl_data_loss')
        }
        metrics_dict = {
            'subs_pred': [MeanAbsoluteError(name='subs_mae')],
            'gwl_pred': [MeanAbsoluteError(name='gwl_mae')]
        }
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate_hp),
            loss=loss_dict, metrics=metrics_dict,
            loss_weights=self.fixed_model_params.get(
                'loss_weights', {'subs_pred': 1.0, 'gwl_pred': 0.8}
            ),
            lambda_pde=lambda_pde_hp
        )
        return model

    def run(
        self,
        inputs: Dict[str, np.ndarray], 
        y: Dict[str, np.ndarray],  
        validation_data: Optional[
            Tuple[Dict[str, np.ndarray], Dict[
                str, np.ndarray]]] = None,
        # forecast_horizon & quantiles are now part of self.fixed_model_params
        # if PIHALTuner was instantiated via create_with_config.
        # However, if user created PIHALTuner() directly and then calls fit(),
        # these are still needed here to populate self.fixed_model_params.
        forecast_horizon: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        epochs: int = 10, 
        batch_size: int = 32,
        callbacks: Optional[List[Callback]] = None,
        verbose: int = 1, 
        case_info: Optional[Dict[str, Any]] = None, 
        **search_kwargs
    ):
        """
        Prepares data if needed, ensures fixed parameters are set, 
        and runs hyperparameter search.
        """
        vlog(f"PIHALTuner: Executing `fit` for project: {self.project_name}",
             verbose=verbose, level=1, )

        # If fixed_model_params were not provided at __init__ or are incomplete,
        # infer them now using the data and explicit args to fit.
        # This allows PIHALTuner() then tuner.fit(data...) workflow.
        if not self.fixed_model_params or not all(
            k in self.fixed_model_params for k in [
                "static_input_dim", "dynamic_input_dim", "future_input_dim",
                "output_subsidence_dim", "output_gwl_dim", "forecast_horizon"
            ]
        ):
            vlog("`fixed_model_params` not fully set at init, "
                 "inferring from `fit` data.", verbose=verbose, level=2)
            
            # Use forecast_horizon & quantiles passed to fit for inference
            fh_for_inference = ( 
                forecast_horizon if forecast_horizon is not None 
                else self.fixed_model_params.get('forecast_horizon')
                )# fallback if already partly set
            q_for_inference = ( 
                quantiles if quantiles is not None 
                else self.fixed_model_params.get('quantiles')
                )

            inferred_params = self._infer_dims_and_prepare_fixed_params(
                inputs_data=inputs,
                targets_data=y,
                user_provided_fixed_params=self.fixed_model_params, # Can be empty
                forecast_horizon=fh_for_inference,
                quantiles=q_for_inference,
                default_params=DEFAULT_PIHALNET_FIXED_PARAMS.copy(),
                verbose=verbose
            )
            self.fixed_model_params.update(inferred_params) # Update instance's dict

        vlog("Final fixed model parameters for PIHALNet build:",
             verbose=verbose, level=2 )
        
        if verbose >=3:
            for k,v_ in self.fixed_model_params.items():
                vlog(f"  {k}: {v_}", 
                     verbose=verbose, level=3, 
            )
        # for consisteny ,recheck and apply rename 
        if y is not None: 
            # Check required target keys after renaming
            check_required_input_keys(None, y=y)
            y = rename_dict_keys(
                y.copy(), # Work on a copy
                param_to_rename={"subsidence": 'subs_pred', "gwl": 'gwl_pred'}
            )
            
        # Prepare tf.data.Dataset
        targets_for_dataset = {
            'subs_pred': y['subs_pred'],
            'gwl_pred': y['gwl_pred']
        } # Assuming y keys are already correct
        train_dataset = Dataset.from_tensor_slices(
            (inputs, targets_for_dataset)
        ).batch(batch_size).prefetch(AUTOTUNE)
        
        val_dataset = None
        if validation_data:
            val_inputs_dict, val_targets_dict = validation_data
            
            # Check required target keys after renaming
            
            check_required_input_keys(None, y=val_targets_dict)
            val_targets_dict = rename_dict_keys(
                val_targets_dict.copy(), # Work on a copy
                param_to_rename={"subsidence": 'subs_pred', "gwl": 'gwl_pred'}
            )
            
            val_targets_for_dataset = {
                'subs_pred': val_targets_dict['subs_pred'],
                'gwl_pred': val_targets_dict['gwl_pred']
            }
            val_dataset = Dataset.from_tensor_slices(
                (val_inputs_dict, val_targets_for_dataset)
            ).batch(batch_size).prefetch(AUTOTUNE)
        
        self._current_run_case_info = DEFAULT_PIHAL_CASE_INFO.copy()
        self._current_run_case_info.update(self.fixed_model_params) # Use the final one
        self._current_run_case_info["description"] = \
            self._current_run_case_info["description"].format(
                "Quantile" if self.fixed_model_params.get(
                    'quantiles') else "Point"
            )
        if case_info:
            self._current_run_case_info.update(case_info)
        
        return super().search( 
            train_data=train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=verbose,
            **search_kwargs
        )

    def _get_hp_choice(self, hp, name, default_choices, **kwargs):
        return hp.Choice(name, self.param_space.get(name, default_choices), **kwargs)

    def _get_hp_int(self, hp, name, default_min, default_max, step=1, **kwargs):
        config = self.param_space.get(name, {})
        return hp.Int(
            name,
            min_value=config.get('min_value', default_min),
            max_value=config.get('max_value', default_max),
            step=config.get('step', step), **kwargs
        )

    def _get_hp_float(self, hp, name, default_min, default_max, default_sampling=None, **kwargs):
        config = self.param_space.get(name, {})
        return hp.Float(
            name,
            min_value=config.get('min_value', default_min),
            max_value=config.get('max_value', default_max),
            sampling=config.get(
                'sampling', kwargs.pop('sampling', default_sampling)),
            **kwargs
        )


PIHALTuner.__doc__ = """
Hyperparameter tuner for the PIHALNet model, which jointly predicts
land subsidence and groundwater level (GWL) via a physics-informed
neural network (PINN) framework.

PIHALTuner leverages Keras Tuner (e.g., RandomSearch or
BayesianOptimization) to search over architectural and PINN-specific
hyperparameters—embedding dimension, hidden units, LSTM layers,
attention heads, dropout, activation functions, PDE coefficients,
learning rates, etc.—while keeping the core model dimensions fixed.
Fixed dimensions (input/output dims, forecast horizon, quantiles)
are passed in `fixed_model_params`, ensuring that only the desired
hyperparameters vary during tuning.

Objective
~~~~~~~~~
Minimize the combined validation loss for subsidence and GWL:
.. math::

   \\theta^* \\;=\\; \\arg\\min_{{\\theta\\in\\Theta}} \\Bigl[
   L_{{\\text{{val}}}}^{{\\text{{subs}}}}\\bigl(f_{{\\theta}}(X),y^{{\\text{{subs}}}}\\bigr)
   + \\lambda_{{\\text{{gwl}}}} \\,
   L_{{\\text{{val}}}}^{{\\text{{gwl}}}}\\bigl(f_{{\\theta}}(X),y^{{\\text{{gwl}}}}\\bigr)
   \\Bigr]

where :math:`\\Theta` is the joint hyperparameter space, and
:math:`\\lambda_{{\\text{{gwl}}}}` can be tuned via `loss_weights` in
`fixed_model_params`.

Parameters
----------
{params.base.fixed_model_params}
{params.base.param_space}
{params.base.objective}
{params.base.max_trials}
{params.base.project_name}
{params.base.directory}
{params.base.executions_per_trial}
{params.base.tuner_type}
{params.base.seed}
{params.base.overwrite_tuner}
{params.base.tuner_kwargs}

verbose : int, default ``1``
    Controls console logging verbosity. ``0``=silent, ``1``=high-level,
    ``2``=detailed, ``>=3``=debug. Higher levels print inferred dims,
    override warnings, and per-epoch metrics.

Methods
-------
fit(inputs, y, validation_data=None, forecast_horizon=None,
    quantiles=None, epochs=10, batch_size=32, callbacks=None,
    verbose=1, case_info=None, **search_kwargs)
    Prepares data, infers/finalizes `fixed_model_params`, builds
    a tf.data.Dataset, and runs the Keras Tuner `search()` method.

Other Parameters (fit method)
-----------------------------
inputs : dict[str, np.ndarray]
    Dictionary of NumPy arrays for model inputs. Must contain:
    ``"coords"``, ``"static_features"``, ``"dynamic_features"``.
    Optionally, ``"future_features"`` if forecasting with exogenous
    variables. Shapes:
      - ``coords``: (batch_size, 2)
      - ``static_features``: (batch_size, static_dim)
      - ``dynamic_features``: (batch_size, time_steps, dynamic_dim)
      - ``future_features``: (batch_size, time_steps, future_dim)

y : dict[str, np.ndarray]
    Dictionary of NumPy arrays for targets. Must contain either
    ``"subsidence"`` or ``"subs_pred"`` (renamed to ``subs_pred``),
    and either ``"gwl"`` or ``"gwl_pred"`` (renamed to ``gwl_pred``).
    Shapes typically: (batch_size, time_steps, 1) for multi-horizon
    or (batch_size, 1) for point forecasts.

validation_data : tuple, optional
    A tuple ``(val_inputs_dict, val_targets_dict)`` analogous to
    ``inputs`` and ``y``. Used for early stopping and objective
    evaluation during search.

forecast_horizon : int, optional
    Horizon length for multi-step forecasting. If not provided in
    ``fixed_model_params``, PIHALTuner will attempt to infer from
    the second dimension of ``y['subs_pred']`` or ``y['gwl_pred']``.

quantiles : list[float], optional
    List of quantiles (e.g., [0.1, 0.5, 0.9]) for probabilistic PINN
    training. If not given, defaults to those already in
    ``fixed_model_params``, or omitted if none.

epochs : int, default ``10``
    Number of training epochs for each model built during the search.

batch_size : int, default ``32``
    Batch size for converting NumPy arrays to tf.data.Dataset.

callbacks : list[tf.keras.callbacks.Callback], optional
    List of Keras callbacks (e.g., `EarlyStopping`) active during both
    the search and refit phases. If None, a sensible default
    `EarlyStopping` on validation loss is applied.

case_info : dict[str, Any], optional
    Dictionary of metadata to include in the tuner’s run case info.
    Used for logging/descriptions. Keys such as ``"description"`` may
    be formatted with “Point” or “Quantile” based on whether
    ``quantiles`` is provided.

**search_kwargs : Any
    Additional keyword arguments forwarded to Keras Tuner’s `search()`
    method (e.g., ``tuner.search(train_data=..., validation_data=...,...)``).

Returns
-------
(model, best_hps, tuner_oracle) : tuple
    - **model** (`tf.keras.Model`): The best‐performing PIHALNet model
      retrained on the full training set with the champion hyperparameters.
    - **best_hps** (`keras_tuner.HyperParameters`): The winning hyperparameter
      configuration.
    - **tuner_oracle** (`keras_tuner.Oracle`): The underlying Keras Tuner
      object containing search history and trial results.

Examples
--------
# 1) Create and run a tuning session using raw NumPy data:
>>> import numpy as np
>>> from fusionlab.nn.pinn.tuning import PIHALTuner
>>> B, T, Sdim, Ddim, Fdim, O = 128, 12, 5, 3, 2, 1
>>> rng = np.random.default_rng(123)
>>> inputs = {{
...     "coords":       rng.normal(size=(B, 2)).astype("float32"),
...     "static_features":  rng.normal(size=(B, Sdim)).astype("float32"),
...     "dynamic_features": rng.normal(size=(B, T, Ddim)).astype("float32"),
...     "future_features":  rng.normal(size=(B, T, Fdim)).astype("float32"),
... }}
>>> targets = {{
...     "subsidence": rng.normal(size=(B, T, O)).astype("float32"),
...     "gwl":        rng.normal(size=(B, T, O)).astype("float32"),
... }}
>>> fixed_params = {{
...     "static_input_dim":    Sdim,
...     "dynamic_input_dim":   Ddim,
...     "future_input_dim":    Fdim,
...     "output_subsidence_dim": O,
...     "output_gwl_dim":      O,
...     "forecast_horizon":    T,
...     "quantiles":           [0.1, 0.5, 0.9],
... }}
>>> tuner = PIHALTuner(
...     fixed_model_params=fixed_params,
...     param_space={{
...         "learning_rate": {{
...             "min_value": 1e-4, "max_value": 1e-2, "sampling": "log"
...         }},
...         "dropout_rate": {{"min_value": 0.0, "max_value": 0.5}},
...         "embed_dim": {{"min_value": 16, "max_value": 64, "step": 16}},
...     }},
...     objective="val_total_loss",
...     max_trials=5,
...     tuner_type="bayesianoptimization",
...     verbose=2,
... )
>>> best_model, best_hps, oracle = tuner.fit(
...     inputs=inputs,
...     y=targets,
...     validation_data=(inputs, targets),
...     epochs=3,
...     batch_size=32,
...     callbacks=None,
...     verbose=2,
... )

# 2) Alternative: Use `create()` to infer dimensions automatically:
>>> tuner2 = PIHALTuner.create(
...     inputs_data=inputs,
...     targets_data=targets,
...     forecast_horizon=None,        # will be inferred as T=12
...     quantiles=[0.05, 0.95],
...     param_space={{"learning_rate": lambda hp: hp.Float("lr", 1e-5, 1e-3, sampling="log")}},
...     objective="val_total_loss",
...     max_trials=3,
...     directory="my_tuner_dir",
...     verbose=1,
... )
>>> best_model2, best_hps2, oracle2 = tuner2.fit(
...     inputs=inputs,
...     y=targets,
...     validation_data=(inputs, targets),
...     epochs=3,
...     batch_size=16,
... )

See Also
--------
PINNTunerBase
    Base class for PINN hyperparameter tuning; implements `.search()`.
fusionlab.nn.pinn.models.PIHALNet
    The core physics-informed neural network architecture being tuned.

References
----------
.. [1] Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).
       *Physics-informed neural networks: A deep learning framework
       for solving forward and inverse problems involving nonlinear
       partial differential equations*. Journal of Computational Physics,
       378, 686–707.

.. [2] Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P.,
       Wang, S., & Yang, L. (2021). *Physics-informed machine learning*.
       Nature Reviews Physics, 3(6), 422–440.

.. [3] Heng, M.H., Chen, W., & Smith, E.C. (2022). *Joint modeling of
       land subsidence and groundwater levels with PINNs*. Environmental
       Modelling & Software, 150, 105347.
""".format(params=_pinn_tuner_docs)

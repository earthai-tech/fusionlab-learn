 
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
        'subs_pred': 1.0,
        'gwl_pred': 0.8
    }
}

# Default case info, can be updated in fit()
DEFAULT_PIHAL_CASE_INFO = {
    "description": "PIHALNet {} forecast",
}

class PIHALTuner(PINNTunerBase):
    def __init__(
        self,
        fixed_model_params: Dict[str, Any], 
        param_space: Optional[Dict[str, Any]] = None, 
        objective: Union[str, kt.Objective] = 'val_loss',
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

        self.fixed_model_params = fixed_model_params
        self.param_space = param_space or {}

        required_fixed = [
            "static_input_dim", 
            "dynamic_input_dim", 
            "future_input_dim",
            "output_subsidence_dim", 
            "output_gwl_dim", 
            "forecast_horizon"
        ]
        for req_param in required_fixed:
            if req_param not in self.fixed_model_params:
                raise ValueError(
                    "Missing required key in"
                    f" `fixed_model_params`: '{req_param}'"
                )
        self._current_run_case_info = {} 


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
        Infers or finalizes fixed model parameters for PIHALNet.
    
        Priority of sources:
          1. `user_provided_fixed_params` (if not None)
          2. Inference from `inputs_data` and `targets_data`
          3. `default_params`
    
    
        If `inputs_data` and `targets_data` are given, this method:
          - Validates required keys in both dictionaries.
          - Infers input dimensions:
              * `static_input_dim` from `static_features`.
              * `dynamic_input_dim` from `dynamic_features`.
              * `future_input_dim` from `future_features`.
          - Infers output dimensions:
              * `output_subsidence_dim` from `targets_data["subs_pred"]`.
              * `output_gwl_dim` from `targets_data["gwl_pred"]`.
          - Infers `forecast_horizon` if not provided, based on target array shapes.
          - Assigns `quantiles` to the inferred or provided list.
    
        Finally, any keys in `user_provided_fixed_params` will override
        inferred or default values. Warnings are emitted if explicit
        values conflict with inferred ones.
    
    
        Parameters
        ----------
        inputs_data : dict of np.ndarray, optional
            Input arrays keyed by layer names (e.g. "static_features",
            "dynamic_features", "future_features"). Used to infer dims.
    
        targets_data : dict of np.ndarray, optional
            Target arrays keyed by "subs_pred" and "gwl_pred". Used to infer
            output dimensions and `forecast_horizon`.
    
        user_provided_fixed_params : dict, optional
            Explicit fixed parameters to override inference (e.g.,
            precomputed dimensions, forecast_horizon, quantiles).
    
        forecast_horizon : int, optional
            Number of steps ahead to predict. If missing, inferred from
            `targets_data["subs_pred"]` shape.
    
        quantiles : list of float, optional
            Quantiles for probabilistic forecasts. If missing, remains None
            or taken from `user_provided_fixed_params`.
    
        default_params : dict
            Default parameter values to use when neither inference nor
            user overrides provide a key. Typically `DEFAULT_PIHALNET_FIXED_PARAMS`.
    
        verbose : int, default=0
            Controls logging verbosity of inference steps.
    
        Returns
        -------
        final_fixed_params : dict
            Fully populated dictionary of fixed parameters for PIHALNet,
            combining defaults, inferred values, and any user overrides.
    
        """
        final_fixed_params = default_params.copy()
        source_log = "defaults"
    
        # Rename keys in targets_data if necessary
        if targets_data:
            _, targets_data = check_required_input_keys(
                None, targets_data,
                message=(
                    "Target keys 'subs_pred' and 'gwl_pred'"
                    " are required in 'y' data."
                )
            )
            targets_data = rename_dict_keys(
                targets_data.copy(),
                param_to_rename={"subsidence": "subs_pred", "gwl": "gwl_pred"}
            )
            # Check required target keys after renaming
    
        inferred_params = {}
        if inputs_data and targets_data:
            source_log = "inferred from data"
            # Check required input keys
            check_required_input_keys(inputs_data, targets_data)
    
            inferred_params["static_input_dim"] = (
                inputs_data["static_features"].shape[-1]
                if inputs_data.get("static_features") is not None and
                   inputs_data["static_features"].ndim == 2 else 0
            )
    
            inferred_params["dynamic_input_dim"] = (
                inputs_data["dynamic_features"].shape[-1]
            )
    
            inferred_params["future_input_dim"] = (
                inputs_data["future_features"].shape[-1]
                if inputs_data.get("future_features") is not None and
                   inputs_data["future_features"].ndim == 3 else 0
            )
    
            inferred_params["output_subsidence_dim"] = (
                targets_data["subs_pred"].shape[-1]
            )
            inferred_params["output_gwl_dim"] = (
                targets_data["gwl_pred"].shape[-1]
            )
    
            # If forecast_horizon is not given, infer from targets
            if forecast_horizon is None:
                if targets_data["subs_pred"].ndim >= 2:
                    # (batch, horizon, ...) or (horizon, ...)
                    fh_idx = 1 if targets_data["subs_pred"].ndim > 1 else 0
                    forecast_horizon = (
                        targets_data["subs_pred"].shape[fh_idx]
                    )
                    vlog(
                        f"Inferred forecast_horizon={forecast_horizon}"
                        " from target shapes.",
                        verbose=verbose, level=3
                    )
                else:
                    # Fallback to default if cannot infer
                    forecast_horizon = default_params.get(
                        "forecast_horizon", 1
                    )
                    vlog(
                        "Cannot infer forecast_horizon,"
                        f" using default={forecast_horizon}.",
                        verbose=verbose, level=2,
                    )
    
            inferred_params["forecast_horizon"] = forecast_horizon
            inferred_params["quantiles"] = quantiles  # Use passed quantiles
    
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
                "forecast_horizon" in user_provided_fixed_params and
                forecast_horizon is not None and
                user_provided_fixed_params["forecast_horizon"] != forecast_horizon
            ):
                logger.warning(
                    "Mismatch in forecast_horizon: "
                    "provided fixed_params override explicit arg."
                )
            elif forecast_horizon is not None:
                final_fixed_params["forecast_horizon"] = forecast_horizon
    
            if (
                "quantiles" in user_provided_fixed_params and
                quantiles is not None and
                user_provided_fixed_params["quantiles"] != quantiles
            ):
                logger.warning(
                    "Mismatch in quantiles: provided "
                    "fixed_params override explicit arg."
                )
            elif quantiles is not None:
                final_fixed_params["quantiles"] = quantiles
    
        vlog(
            f"Final fixed_model_params determined from: {source_log}",
            verbose=verbose, level=2
        )
        if verbose >= 3:
            for k, v_ in final_fixed_params.items():
                vlog(f"  Fixed Param -> {k}: {v_}", verbose=verbose, level=3)
    
        return final_fixed_params

    @classmethod
    def create(
        cls,
        fixed_model_params: Optional[Dict[str, Any]] = None,
        inputs_data: Optional[Dict[str, np.ndarray]] = None,
        targets_data: Optional[Dict[str, np.ndarray]] = None,
        forecast_horizon: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        objective: Union[str, kt.Objective] = 'val_loss',
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
    
        After computing the final `fixed_model_params`, this method
        instantiates and returns a `PIHALTuner` with all required
        configuration for hyperparameter search.
    
        Parameters
        ----------
        fixed_model_params : Dict[str, Any], optional
            A complete dictionary of fixed parameters for PIHALNet.
            These typically include input/output dimensions (static,
            dynamic, future), output dimensions for subsidence and GWL,
            forecast_horizon, quantiles, etc. If not provided, inference
            occurs using `inputs_data` and `targets_data`, or defaults.
    
        inputs_data : Dict[str, np.ndarray], optional
            Dictionary of NumPy arrays for model inputs. Used to infer
            dimensions if `fixed_model_params` is incomplete.
    
        targets_data : Dict[str, np.ndarray], optional
            Dictionary of NumPy arrays for model targets. Used to infer
            output dimensions if `fixed_model_params` is incomplete.
    
        forecast_horizon : int, optional
            Explicitly set forecast horizon. Used during inference if not
            already in `fixed_model_params`.
    
        quantiles : List[float], optional
            Explicitly set quantiles for probabilistic forecasts. Used
            during inference if not already in `fixed_model_params`.
    
        objective : str or keras_tuner.Objective, optional
            The optimization metric name (e.g. "val_loss" or "val_total_loss")
            that the tuner should optimize. Defaults to "val_loss".
    
        max_trials : int, optional
            The maximum number of hyperparameter combinations (trials) to
            explore. Defaults to 20.
    
        project_name : str, optional
            Name of the tuner project folder under `directory`. Defaults
            to "PIHALNet_Tuning_From_Config".
    
        directory : str, optional
            Root directory where Keras Tuner stores results for this
            project. Defaults to "pihalnet_tuner_results".
    
        param_space : Dict[str, Any], optional
            A mapping from hyperparameter names to search-space definitions
            understood by Keras Tuner (e.g. hp.Choice, hp.Int, hp.Float).
            When None, the tuner will use the built-in default space
            defined in `PIHALTuner.build()`.
    
        verbose : int, default=0
            Logging verbosity level. `0` = silent; `1` = info; `>=2` = debug.
    
        **tuner_init_kwargs : Any
            Additional keyword arguments forwarded to the `PINNTunerBase`
            constructor (e.g. `executions_per_trial`, `tuner_type`, `seed`,
            `overwrite_tuner`, etc.).
    
        Returns
        -------
        PIHALTuner
            An instance of `PIHALTuner` with `fixed_model_params` fully
            populated and ready to run hyperparameter search.
    
        Examples
        --------
        >>> # Example: infer dimensions from data and create the tuner
        >>> inputs = {
        ...     "coords": np.random.rand(100, 2, 3),
        ...     "static_features": np.random.rand(100, 5),
        ...     "dynamic_features": np.random.rand(100, 2, 4),
        ...     "future_features": np.random.rand(100, 2, 1)
        ... }
        >>> targets = {
        ...     "subsidence": np.random.rand(100, 2, 1),
        ...     "gwl": np.random.rand(100, 2, 1)
        ... }
        >>> tuner = PIHALTuner.create(
        ...     inputs_data=inputs,
        ...     targets_data=targets,
        ...     forecast_horizon=2,
        ...     quantiles=[0.1, 0.5, 0.9],
        ...     max_trials=5,
        ...     project_name="zh_config_test",
        ...     tuner_type="bayesian",
        ...     verbose=1
        ... )
    
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
        Builds and compiles a PIHALNet model given a set of hyperparameters.
    
        This method assumes that `self.fixed_model_params` has already been
        populated (either in `__init__` or via a prior `run()`/`fit()` call).
        It will:
    
          1. Verify that all required fixed parameters (e.g.,
             `"dynamic_input_dim"`) exist.
          2. Extract architecture‐related hyperparameters from `hp` (e.g.,
             `embed_dim`, `hidden_units`, LSTM units, attention units, etc.).
          3. Extract PINN‐specific hyperparameters (e.g., `pde_mode`, whether
             to learn or fix coefficient :math:`C`, PDE weight
             `lambda_pde`, learning rate).
          4. Merge the hyperparameters with `self.fixed_model_params`,
             discarding any unexpected keys via `_get_valid_kwargs`.
          5. Instantiate `PIHALNet(**model_params)`.
          6. Compile with an Adam optimizer (clipping gradients at norm = 1.0),
             two separate MSE losses (`subs_pred` vs. `gwl_pred`), and their
             corresponding MAE metrics. Loss weights default to
             `{"subs_pred": 1.0, "gwl_pred": 0.8}` if not provided in
             `fixed_model_params`.
          7. Return the compiled `tf.keras.Model`.
    
        Parameters
        ----------
        hp : keras_tuner.HyperParameters
            A `HyperParameters` instance provided by Keras Tuner containing
            values for:
              - **embed_dim** (int between 16 and 64, step 16)
              - **hidden_units** (int between 32 and 128, step 32)
              - **lstm_units** (int between 32 and 128, step 32)
              - **attention_units** (int between 32 and 128, step 32)
              - **num_heads** (choice among [1, 2, 4])
              - **dropout_rate** (float between 0.0 and 0.3)
              - **activation** (choice among ["relu","gelu"])
              - **use_vsn** (boolean)
              - **vsn_units** (int between `max(16, hidden_units//4)` and
                `hidden_units` if `use_vsn=True`)
              - **pde_mode** (choice among ["consolidation","none"])
              - **pinn_coefficient_C_type** (choice among
                ["learnable","fixed"])
              - **pinn_coefficient_C_value** (float between 1e–5 and 1e–1 if
                `pinn_coefficient_C_type="fixed"`)
              - **lambda_pde** (float between 0.01 and 1.0)
              - **learning_rate** (choice among [1e–3, 5e–4, 1e–4])
    
        Returns
        -------
        tf.keras.Model
            A compiled `PIHALNet` instance, ready for training. The model’s
            `compile()` call uses:
              - **optimizer**: `Adam(learning_rate=<chosen>, clipnorm=1.0)`
              - **loss**: `{'subs_pred': MSE(name="subs_data_loss"),
                'gwl_pred': MSE(name="gwl_data_loss")}`
              - **metrics**: `{'subs_pred': [MAE(name="subs_mae")],
                'gwl_pred': [MAE(name="gwl_mae")]}`
              - **loss_weights**: as given in
                `self.fixed_model_params["loss_weights"]` or
                `{"subs_pred": 1.0, "gwl_pred": 0.8}` by default
              - **lambda_pde**: the PDE‐weight hyperparameter
    
        Raises
        ------
        RuntimeError
            If `self.fixed_model_params` is empty or missing
            `"dynamic_input_dim"`, indicating that the tuner has not yet
            inferred or been given fixed dimensions.
        """

        if not self.fixed_model_params or \
           'dynamic_input_dim' not in self.fixed_model_params:
            raise RuntimeError(
                "`fixed_model_params` (with inferred dimensions) must be set "
                "by calling `fit()` before the tuner calls `build()`."
            )

        # --- Architectural HPs ---
        embed_dim_hp = self._get_hp_int(
            hp, 'embed_dim', 16, 64, step=16
        )
        hidden_units_hp = self._get_hp_int(
            hp, 'hidden_units', 32, 128, step=32
        )
        lstm_units_hp = self._get_hp_int(
            hp, 'lstm_units', 32, 128, step=32
        )
        attention_units_hp = self._get_hp_int(
            hp, 'attention_units', 32, 128, step=32
        )
        num_heads_hp = self._get_hp_choice(
            hp, 'num_heads', [1, 2, 4]
        )
        dropout_rate_hp = self._get_hp_float(
            hp, 'dropout_rate', 0.0, 0.3
        )
        activation_hp = self._get_hp_choice(
            hp, 'activation', ['relu', 'gelu']
        )
        use_vsn_hp = self._get_hp_choice(
            hp, 'use_vsn', [True, False]
        )
        vsn_units_hp = (
            self._get_hp_int(
                hp, 'vsn_units',
                max(16, hidden_units_hp // 4),
                hidden_units_hp,
                max(16, hidden_units_hp // 4)
            ) if use_vsn_hp else None
        )

        # --- PINN HPs ---
        # Defaulting to 'consolidation' as 'gw_flow' is complex for current PIHALNet
        pde_mode_hp = self._get_hp_choice(
            hp, 'pde_mode', ['consolidation', 'none']
        )
        pinn_c_type = self._get_hp_choice(
            hp, 'pinn_coefficient_C_type', ['learnable', 'fixed']
        )
        pinn_c_value_hp = (
            'learnable' if pinn_c_type == 'learnable' else
            self._get_hp_float(
                hp, 'pinn_coefficient_C_value',
                1e-5, 1e-1, sampling='linear'
            )
        )

        lambda_pde_hp = self._get_hp_float(
            hp, 'lambda_pde', 0.01, 1, sampling='linear'
        )
        learning_rate_hp = self._get_hp_choice(
            hp, 'learning_rate', [1e-3, 5e-4, 1e-4]
        )

        cast_multiple_bool_params(
            self.fixed_model_params,
            bool_params_to_cast=[('use_vsn', False),
                                 ('use_residuals', True)],
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
            # Defaults for other PIHALNet params
            # if not in fixed_model_params
            "max_window_size": self.fixed_model_params.get(
                'max_window_size', 10
            ),
            "memory_size": self.fixed_model_params.get(
                'memory_size', 100
            ),
            "scales": self.fixed_model_params.get('scales', [1]),
            "multi_scale_agg": self.fixed_model_params.get(
                'multi_scale_agg', 'last'
            ),
            "final_agg": self.fixed_model_params.get(
                'final_agg', 'last'
            ),
            "use_residuals": self.fixed_model_params.get(
                'use_residuals', True
            ),
            "use_batch_norm": self.fixed_model_params.get(
                'use_batch_norm', False
            ),
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

        opt = Adam(learning_rate=learning_rate_hp, clipnorm=1.0)
        model.compile(
            optimizer=opt,
            loss=loss_dict,
            metrics=metrics_dict,
            loss_weights=self.fixed_model_params.get(
                'loss_weights',
                {'subs_pred': 1.0, 'gwl_pred': 0.8}
            ),
            lambda_pde=lambda_pde_hp
        )
        return model

    def run(
        self,
        inputs: Dict[str, np.ndarray],
        y: Dict[str, np.ndarray],
        validation_data: Optional[
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        ] = None,
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
    
        Steps:
          1. Logs a message indicating the start of `fit()` for this tuner.
          2. If `self.fixed_model_params` is empty, or missing any of
             `["static_input_dim","dynamic_input_dim","future_input_dim",
             "output_subsidence_dim","output_gwl_dim","forecast_horizon"]`,
             call `_infer_dims_and_prepare_fixed_params(...)`:
               - Uses `inputs`, `y`, plus any explicitly given
                 `forecast_horizon` or `quantiles` to fill in missing
                 dimensions (static, dynamic, future, output dims) and other
                 defaults from `DEFAULT_PIHALNET_FIXED_PARAMS`.
               - Updates `self.fixed_model_params` in‐place.
          3. Logs (at debug/verbose ≥3) all final entries in
             `self.fixed_model_params`.
          4. Renames target keys:
               - Validates that `y` contains keys “subsidence” and “gwl”
                 (or already “subs_pred”, “gwl_pred”) via
                 `check_required_input_keys`.
               - Calls
                 `rename_dict_keys(y, {"subsidence": "subs_pred", "gwl":
                 "gwl_pred"})`.
          5. Constructs `tf.data.Dataset` for training:
               - `train_dataset = Dataset.from_tensor_slices((inputs,
                 {'subs_pred': …,'gwl_pred': …}))`
               - Batch by `batch_size` and `.prefetch(AUTOTUNE)`.
          6. If `validation_data` is provided:
               - Unpack into `(val_inputs, val_targets)`.
               - Rename `val_targets` similarly.
               - Create
                 `val_dataset = Dataset.from_tensor_slices((val_inputs,
                 {'subs_pred': …,'gwl_pred': …}))` with same
                 batching/prefetch.
          7. Prepare `self._current_run_case_info` by copying
             `DEFAULT_PIHAL_CASE_INFO`, updating with
             `self.fixed_model_params`, and formatting the
             `"description"` field (inserting “Quantile” vs. “Point”
             depending on `quantiles`). If the user passed `case_info`,
             merge those keys last.
          8. Call `super().search(...)` with:
               - `train_data=train_dataset`
               - `validation_data=val_dataset`
               - `epochs=epochs`
               - `callbacks=callbacks`
               - `verbose=verbose`
               - Any additional `**search_kwargs` (e.g., `max_trials`,
                 `project_name`, `directory`, etc., are already set on the
                 tuner object)
          9. Return whatever `super().search(...)` returns (typically best
             model, best HP, and tuner instance).
    
        Parameters
        ----------
        inputs : Dict[str, np.ndarray]
            A dictionary of NumPy arrays for model inputs. Keys must match
            the input‐layer names expected by `PIHALNet` (e.g., `"coords"`,
            `"static_features"`, `"dynamic_features"`, `"future_features"`,
            etc.).
        y : Dict[str, np.ndarray]
            A dictionary of NumPy arrays for targets. Expected keys:
              - `"subsidence"` and `"gwl"` if not already renamed, or
              - `"subs_pred"` and `"gwl_pred"` if already in model format.
            Values must be arrays of shape `(batch_size, ..., 1)` matching
            the output dimensions.
        validation_data : Optional[
            Tuple[Dict[str, np.ndarray],Dict[str, np.ndarray]]
        ], default=None
            If provided, a tuple `(val_inputs, val_targets)`, with the same
            key conventions as `inputs` and `y`. If `None`, no validation
            set is used.
        forecast_horizon : Optional[int], default=None
            The number of time‐steps ahead the model predicts. Used only if
            `self.fixed_model_params` still lacks `"forecast_horizon"` and
            must be inferred.
        quantiles : Optional[List[float]], default=None
            If using quantile predictions, the list of quantiles. Used only
            if `self.fixed_model_params` lacks `"quantiles"` and must be
            inferred.
        epochs : int, default=10
            The maximum number of epochs to train each trial during the
            search.
        batch_size : int, default=32
            Batch size for both training and validation datasets.
        callbacks : Optional[List[tf.keras.callbacks.Callback]],
            default=None
            List of Keras callbacks (e.g., `EarlyStopping`) to apply during
            each trial. If `None`, no additional callbacks are used.
        verbose : int, default=1
            Verbosity mode. `0` = silent; `1` = progress bars; `≥2` = debug/log.
        case_info : Optional[Dict[str, Any]], default=None
            Additional metadata (strings, numbers) to merge into
            `self._current_run_case_info`, which is ultimately saved in the
            tuner’s summary JSON. Common keys include `"description"`,
            `"run_id"`, etc.
        **search_kwargs : Any
            Additional keyword arguments forwarded to `KerasTuner.search(...)`,
            such as `max_trials`, `project_name`, `directory`, etc. All other
            tuning configuration (e.g., `tuner_type`, `executions_per_trial`)
            should already be set on the tuner instance.
    
        Returns
        -------
        Any
            The return value of `super().search(...)`, which for Keras Tuner
            is typically a tuple `(best_model, best_hyperparameters,
            tuner_instance)`.
    
        Raises
        ------
        RuntimeError
            If `self.fixed_model_params` cannot be inferred (e.g. missing
            critical dimensions and no `forecast_horizon`/`quantiles`
            provided).
    
        ValueError
            If required target keys are missing in `y` (after attempting to
            rename).
    
        Notes
        -----
        - This method replaces the usual `fit(...)` interface; users should
          call `PIHALTuner.run(...)` (or `tuner.fit(...)` if aliased)
          instead of directly calling `search(...)`.
        - After this returns, `self.best_hps_`, `self.best_model_`, etc. are
          populated and `self._save_tuning_summary()` is called internally.
    
        Examples
        --------
        >>> # Suppose `inputs_train` and `targets_train` are dicts of NumPy
        >>> # arrays:
        >>> tuner = PIHALTuner(
        ...     fixed_model_params={'static_input_dim': 5,
        ...                         'dynamic_input_dim': 4,
        ...                         'future_input_dim': 1,
        ...                         'output_subsidence_dim': 1,
        ...                         'output_gwl_dim': 1,
        ...                         'forecast_horizon': 3},
        ...     max_trials=10,
        ...     project_name="zh_tuning",
        ...     tuner_type="bayesian"
        ... )
        >>> best_model, best_hps, tuner_obj = tuner.run(
        ...     inputs=inputs_train,
        ...     y={'subsidence': subs_arr, 'gwl': gwl_arr},
        ...     validation_data=(inputs_val,
        ...         {'subsidence': subs_val, 'gwl': gwl_val}),
        ...     epochs=20,
        ...     batch_size=64,
        ...     callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)],
        ...     verbose=2
        ... )
        """

        vlog(
            f"PIHALTuner: Executing `fit` for project: "
            f"{self.project_name}",
            verbose=verbose, level=1
        )

        # If fixed_model_params were not provided at __init__ or are incomplete,
        # infer them now using the data and explicit args to fit.
        # This allows PIHALTuner() then tuner.fit(data...) workflow.
        if not self.fixed_model_params or not all(
            k in self.fixed_model_params for k in [
                "static_input_dim", "dynamic_input_dim",
                "future_input_dim", "output_subsidence_dim",
                "output_gwl_dim", "forecast_horizon"
            ]
        ):
            vlog(
                "`fixed_model_params` not fully set at init, "
                "inferring from `fit` data.",
                verbose=verbose, level=2
            )

            # Use forecast_horizon & quantiles passed to fit for inference
            fh_for_inference = (
                forecast_horizon
                if forecast_horizon is not None
                else self.fixed_model_params.get('forecast_horizon')
            )
            q_for_inference = (
                quantiles
                if quantiles is not None
                else self.fixed_model_params.get('quantiles')
            )

            inferred_params = self._infer_dims_and_prepare_fixed_params(
                inputs_data=inputs,
                targets_data=y,
                user_provided_fixed_params=self.fixed_model_params,  # Can be empty
                forecast_horizon=fh_for_inference,
                quantiles=q_for_inference,
                default_params=DEFAULT_PIHALNET_FIXED_PARAMS.copy(),
                verbose=verbose
            )
            self.fixed_model_params.update(inferred_params)

        vlog(
            "Final fixed model parameters for PIHALNet build:",
            verbose=verbose, level=2
        )
        if verbose >= 3:
            for k, v_ in self.fixed_model_params.items():
                vlog(f"  {k}: {v_}", verbose=verbose, level=3)

        # for consistency, recheck and apply rename
        if y is not None:
            # Check required target keys after renaming
            check_required_input_keys(None, y=y)
            y = rename_dict_keys(
                y.copy(),  # Work on a copy
                param_to_rename={
                    "subsidence": 'subs_pred',
                    "gwl": 'gwl_pred'
                }
            )

        # Prepare tf.data.Dataset
        targets_for_dataset = {
            'subs_pred': y['subs_pred'],
            'gwl_pred': y['gwl_pred']
        }
        train_dataset = Dataset.from_tensor_slices(
            (inputs, targets_for_dataset)
        ).batch(batch_size).prefetch(AUTOTUNE)

        val_dataset = None
        if validation_data:
            val_inputs_dict, val_targets_dict = validation_data

            # Check required target keys after renaming
            check_required_input_keys(None, y=val_targets_dict)
            val_targets_dict = rename_dict_keys(
                val_targets_dict.copy(),  # Work on a copy
                param_to_rename={
                    "subsidence": 'subs_pred',
                    "gwl": 'gwl_pred'
                }
            )

            val_targets_for_dataset = {
                'subs_pred': val_targets_dict['subs_pred'],
                'gwl_pred': val_targets_dict['gwl_pred']
            }
            val_dataset = Dataset.from_tensor_slices(
                (val_inputs_dict, val_targets_for_dataset)
            ).batch(batch_size).prefetch(AUTOTUNE)

        self._current_run_case_info = DEFAULT_PIHAL_CASE_INFO.copy()
        self._current_run_case_info.update(
            self.fixed_model_params
        )  # Use the final one
        self._current_run_case_info[
            "description"
        ] = self._current_run_case_info["description"].format(
            "Quantile"
            if self.fixed_model_params.get('quantiles') else "Point"
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
        return hp.Choice(
            name,
            self.param_space.get(name, default_choices),
            **kwargs
        )

    def _get_hp_int(
        self, hp, name, default_min, default_max, step=1, **kwargs
    ):
        config = self.param_space.get(name, {})
        return hp.Int(
            name,
            min_value=config.get('min_value', default_min),
            max_value=config.get('max_value', default_max),
            step=config.get('step', step),
            **kwargs
        )

    def _get_hp_float(
        self, hp, name, default_min, default_max,
        default_sampling=None, **kwargs
    ):
        config = self.param_space.get(name, {})
        return hp.Float(
            name,
            min_value=config.get('min_value', default_min),
            max_value=config.get('max_value', default_max),
            sampling=config.get(
                'sampling', kwargs.pop('sampling', default_sampling)
            ),
            **kwargs
        )

    # def build(self, hp: kt.HyperParameters) -> Model:
    #     """
    #     Builds and compiles PIHALNet. `self.fixed_model_params` must be populated first.
    #     """
    #     if not self.fixed_model_params or 'dynamic_input_dim' not in self.fixed_model_params:
    #         raise RuntimeError(
    #             "`fixed_model_params` (with inferred dimensions) must be set "
    #             "by calling `fit()` before the tuner calls `build()`."
    #         )
            
    #     # --- Architectural HPs ---
    #     embed_dim_hp = self._get_hp_int(hp, 'embed_dim', 16, 64, step=16)
    #     hidden_units_hp = self._get_hp_int(hp, 'hidden_units', 32, 128, step=32)
    #     lstm_units_hp = self._get_hp_int(hp, 'lstm_units', 32, 128, step=32)
    #     attention_units_hp = self._get_hp_int(hp, 'attention_units', 32, 128, step=32)
    #     num_heads_hp = self._get_hp_choice(hp, 'num_heads', [1, 2, 4])
    #     dropout_rate_hp = self._get_hp_float(hp, 'dropout_rate', 0.0, 0.3)
    #     activation_hp = self._get_hp_choice(hp, 'activation', ['relu', 'gelu'])
    #     use_vsn_hp = self._get_hp_choice(hp, 'use_vsn', [True, False])
    #     vsn_units_hp = self._get_hp_int(
    #         hp, 'vsn_units', 
    #         max(16, hidden_units_hp // 4), hidden_units_hp, 
    #         max(16, hidden_units_hp // 4)
    #     ) if use_vsn_hp else None
        
    #     # --- PINN HPs ---
    #     # Defaulting to 'consolidation' as 'gw_flow' is complex for current PIHALNet
    #     pde_mode_hp = self._get_hp_choice(hp, 'pde_mode', ['consolidation', 'none'])
    #     pinn_c_type = self._get_hp_choice(
    #         hp, 'pinn_coefficient_C_type', ['learnable', 'fixed']
    #     )
    #     pinn_c_value_hp = 'learnable' if pinn_c_type == 'learnable' else \
    #         self._get_hp_float(hp, 'pinn_coefficient_C_value', 1e-5, 1e-1, sampling='linear')
        
    #     lambda_pde_hp = self._get_hp_float(hp, 'lambda_pde', 0.01, 1, sampling='linear')
    #     learning_rate_hp = self._get_hp_choice(hp, 'learning_rate', [1e-3, 5e-4, 1e-4])

    #     cast_multiple_bool_params (
    #         self.fixed_model_params, 
    #         bool_params_to_cast= [('use_vsn', False), ('use_residuals', True)], 
    #     )
    
    #     model_params = {
    #         **self.fixed_model_params,
    #         "embed_dim": embed_dim_hp, 
    #         "hidden_units": hidden_units_hp,
    #         "lstm_units": lstm_units_hp,
    #         "attention_units": attention_units_hp,
    #         "num_heads": num_heads_hp, 
    #         "dropout_rate": dropout_rate_hp,
    #         "activation": activation_hp, 
    #         "use_vsn": use_vsn_hp,
    #         "vsn_units": vsn_units_hp, 
    #         "pde_mode": pde_mode_hp,
    #         "pinn_coefficient_C": pinn_c_value_hp, 
    #         "gw_flow_coeffs": None,
    #         # Defaults for other PIHALNet params if not in fixed_model_params
    #         "max_window_size": self.fixed_model_params.get('max_window_size', 10),
    #         "memory_size": self.fixed_model_params.get('memory_size', 100),
    #         "scales": self.fixed_model_params.get('scales', [1]),
    #         "multi_scale_agg": self.fixed_model_params.get('multi_scale_agg', 'last'),
    #         "final_agg": self.fixed_model_params.get('final_agg', 'last'),
    #         "use_residuals": self.fixed_model_params.get('use_residuals', True),
    #         "use_batch_norm": self.fixed_model_params.get('use_batch_norm', False),
    #     }
    #     model_params = _get_valid_kwargs(PIHALNet, model_params)
    #     model = PIHALNet(**model_params)

    #     loss_dict = {
    #         'subs_pred': MeanSquaredError(name='subs_data_loss'),
    #         'gwl_pred': MeanSquaredError(name='gwl_data_loss')
    #     }
    #     metrics_dict = {
    #         'subs_pred': [MeanAbsoluteError(name='subs_mae')],
    #         'gwl_pred': [MeanAbsoluteError(name='gwl_mae')]
    #     }
        
    #     opt = Adam(learning_rate=learning_rate_hp, clipnorm=1.0)
    #     model.compile(
    #         optimizer=opt,
    #         loss=loss_dict, 
    #         metrics=metrics_dict,
    #         loss_weights=self.fixed_model_params.get(
    #             'loss_weights', {'subs_pred': 1.0, 'gwl_pred': 0.8}
    #         ),
    #         lambda_pde=lambda_pde_hp
    #     )
    #     return model

    # def run(
    #     self,
    #     inputs: Dict[str, np.ndarray], 
    #     y: Dict[str, np.ndarray],  
    #     validation_data: Optional[
    #         Tuple[Dict[str, np.ndarray], Dict[
    #             str, np.ndarray]]] = None,
    #     # forecast_horizon & quantiles are now part of self.fixed_model_params
    #     # if PIHALTuner was instantiated via create_with_config.
    #     # However, if user created PIHALTuner() directly and then calls fit(),
    #     # these are still needed here to populate self.fixed_model_params.
    #     forecast_horizon: Optional[int] = None,
    #     quantiles: Optional[List[float]] = None,
    #     epochs: int = 10, 
    #     batch_size: int = 32,
    #     callbacks: Optional[List[Callback]] = None,
    #     verbose: int = 1, 
    #     case_info: Optional[Dict[str, Any]] = None, 
    #     **search_kwargs
    # ):
    #     """
    #     Prepares data if needed, ensures fixed parameters are set, 
    #     and runs hyperparameter search.
    #     """
    #     vlog(f"PIHALTuner: Executing `fit` for project: {self.project_name}",
    #          verbose=verbose, level=1, )

    #     # If fixed_model_params were not provided at __init__ or are incomplete,
    #     # infer them now using the data and explicit args to fit.
    #     # This allows PIHALTuner() then tuner.fit(data...) workflow.
    #     if not self.fixed_model_params or not all(
    #         k in self.fixed_model_params for k in [
    #             "static_input_dim", "dynamic_input_dim", "future_input_dim",
    #             "output_subsidence_dim", "output_gwl_dim", "forecast_horizon"
    #         ]
    #     ):
    #         vlog("`fixed_model_params` not fully set at init, "
    #              "inferring from `fit` data.", verbose=verbose, level=2)
            
    #         # Use forecast_horizon & quantiles passed to fit for inference
    #         fh_for_inference = ( 
    #             forecast_horizon if forecast_horizon is not None 
    #             else self.fixed_model_params.get('forecast_horizon')
    #             )# fallback if already partly set
    #         q_for_inference = ( 
    #             quantiles if quantiles is not None 
    #             else self.fixed_model_params.get('quantiles')
    #             )

    #         inferred_params = self._infer_dims_and_prepare_fixed_params(
    #             inputs_data=inputs,
    #             targets_data=y,
    #             user_provided_fixed_params=self.fixed_model_params, # Can be empty
    #             forecast_horizon=fh_for_inference,
    #             quantiles=q_for_inference,
    #             default_params=DEFAULT_PIHALNET_FIXED_PARAMS.copy(),
    #             verbose=verbose
    #         )
    #         self.fixed_model_params.update(inferred_params) # Update instance's dict

    #     vlog("Final fixed model parameters for PIHALNet build:",
    #          verbose=verbose, level=2 )
        
    #     if verbose >=3:
    #         for k,v_ in self.fixed_model_params.items():
    #             vlog(f"  {k}: {v_}", 
    #                  verbose=verbose, level=3, 
    #         )
    #     # for consisteny ,recheck and apply rename 
    #     if y is not None: 
    #         # Check required target keys after renaming
    #         check_required_input_keys(None, y=y)
    #         y = rename_dict_keys(
    #             y.copy(), # Work on a copy
    #             param_to_rename={"subsidence": 'subs_pred', "gwl": 'gwl_pred'}
    #         )
            
    #     # Prepare tf.data.Dataset
    #     targets_for_dataset = {
    #         'subs_pred': y['subs_pred'],
    #         'gwl_pred': y['gwl_pred']
    #     } # Assuming y keys are already correct
    #     train_dataset = Dataset.from_tensor_slices(
    #         (inputs, targets_for_dataset)
    #     ).batch(batch_size).prefetch(AUTOTUNE)
        
    #     val_dataset = None
    #     if validation_data:
    #         val_inputs_dict, val_targets_dict = validation_data
            
    #         # Check required target keys after renaming
            
    #         check_required_input_keys(None, y=val_targets_dict)
    #         val_targets_dict = rename_dict_keys(
    #             val_targets_dict.copy(), # Work on a copy
    #             param_to_rename={"subsidence": 'subs_pred', "gwl": 'gwl_pred'}
    #         )
            
    #         val_targets_for_dataset = {
    #             'subs_pred': val_targets_dict['subs_pred'],
    #             'gwl_pred': val_targets_dict['gwl_pred']
    #         }
    #         val_dataset = Dataset.from_tensor_slices(
    #             (val_inputs_dict, val_targets_for_dataset)
    #         ).batch(batch_size).prefetch(AUTOTUNE)
        
    #     self._current_run_case_info = DEFAULT_PIHAL_CASE_INFO.copy()
    #     self._current_run_case_info.update(self.fixed_model_params) # Use the final one
    #     self._current_run_case_info["description"] = \
    #         self._current_run_case_info["description"].format(
    #             "Quantile" if self.fixed_model_params.get(
    #                 'quantiles') else "Point"
    #         )
    #     if case_info:
    #         self._current_run_case_info.update(case_info)
        
    #     return super().search( 
    #         train_data=train_dataset,
    #         epochs=epochs,
    #         validation_data=val_dataset,
    #         callbacks=callbacks,
    #         verbose=verbose,
    #         **search_kwargs
    #     )

    # def _get_hp_choice(self, hp, name, default_choices, **kwargs):
    #     return hp.Choice(name, self.param_space.get(name, default_choices), **kwargs)

    # def _get_hp_int(self, hp, name, default_min, default_max, step=1, **kwargs):
    #     config = self.param_space.get(name, {})
    #     return hp.Int(
    #         name,
    #         min_value=config.get('min_value', default_min),
    #         max_value=config.get('max_value', default_max),
    #         step=config.get('step', step), **kwargs
    #     )

    # def _get_hp_float(self, hp, name, default_min, default_max, default_sampling=None, **kwargs):
    #     config = self.param_space.get(name, {})
    #     return hp.Float(
    #         name,
    #         min_value=config.get('min_value', default_min),
    #         max_value=config.get('max_value', default_max),
    #         sampling=config.get(
    #             'sampling', kwargs.pop('sampling', default_sampling)),
    #         **kwargs
    #     )


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

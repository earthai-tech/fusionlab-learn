# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import warnings
import logging 
from typing import Dict, Optional, Any , Union, Type
from typing import Tuple, List, Callable 

import numpy as np 

from ..._fusionlog import fusionlog 
from ...api.docs import DocstringComponents, _pinn_tuner_common_params 
from ...utils.generic_utils import ( 
    vlog, rename_dict_keys, cast_multiple_bool_params
   )
from ...core.handlers import _get_valid_kwargs 

from ..pinn.models import PIHALNet , TransFlowSubsNet 
from ..pinn.utils import  ( # noqa
    prepare_pinn_data_sequences, 
    check_required_input_keys
)
from .. import KERAS_DEPS 

from . import KT_DEPS
from ._base_tuner import PINNTunerBase 

HyperParameters = KT_DEPS.HyperParameters
Objective = KT_DEPS.Objective
Tuner = KT_DEPS.Tuner 

AUTOTUNE = KERAS_DEPS.AUTOTUNE
Model =KERAS_DEPS.Model 
Adam =KERAS_DEPS.Adam
MeanSquaredError =KERAS_DEPS.MeanSquaredError
MeanAbsoluteError =KERAS_DEPS.MeanAbsoluteError
Callback =KERAS_DEPS.Callback 
Dataset =KERAS_DEPS.Dataset 

logger = fusionlog().get_fusionlab_logger(__name__) 

_pinn_tuner_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_pinn_tuner_common_params)
)

# --- 1. Common Default Parameters 
# for all BaseAttentive-based models ---
DEFAULT_COMMON_FIXED_PARAMS = {
    "output_subsidence_dim": 1,
    "output_gwl_dim": 1,
    "quantiles": None,
    "max_window_size": 10,
    "memory_size": 100,
    "scales": [1],
    "multi_scale_agg": 'last',
    "final_agg": 'last',
    "use_residuals": True,
    "use_batch_norm": False,
    "activation": "relu",
    "architecture_config": { # Default architecture
        'encoder_type': 'hybrid',
        'decoder_attention_stack': ['cross', 'hierarchical', 'memory'],
        'feature_processing': 'vsn'
    }
}
# --- 2. Model-Specific Default Parameters ---
# Inherits from common and adds/overrides specific settings.
DEFAULT_PIHALNET_PARAMS = {
    **DEFAULT_COMMON_FIXED_PARAMS,
    "pde_mode": "consolidation",
    "pinn_coefficient_C": "learnable",
    # PIHALNet does not use these by default
    "gw_flow_coeffs": None,
    "loss_weights": {
        'subs_pred': 1.0,
        'gwl_pred': 0.8
    }
}

DEFAULT_TRANSFLOW_PARAMS = {
    **DEFAULT_COMMON_FIXED_PARAMS,
    "pde_mode": "both", # Uses both physical laws by default
    "pinn_coefficient_C": "learnable",
    # TransFlowSubsNet uses these by default
    "K": "learnable",
    "Ss": "learnable",
    "Q": 0.0,
    "loss_weights": {
        'subs_pred': 1.0,
        'gwl_pred': 1.0
    }
}

# Default case info for logging and summaries
DEFAULT_CASE_INFO = {
    "description": "{model_name} {forecast_type} forecast",
}

class HydroTuner(PINNTunerBase):
    """A robust and flexible hyperparameter tuner for
    hydrogeological PINN models.

    This class provides a unified interface to perform hyperparameter
    optimization for complex physics-informed models like ``PIHALNet`` and
    ``TransFlowSubsNet``, using ``keras-tuner`` as the backend.

    It is designed to be highly flexible, separating the fixed,
    data-dependent parameters from the tunable hyperparameters, which
    are defined in a user-provided search space. The tuner dynamically
    constructs and compiles the specified model for each trial based
    on these configurations.

    The recommended way to instantiate this class is through the
    ``.create()`` factory method, which can automatically infer data
    dimensions.

    Parameters
    ----------
    model_name_or_cls : str or Type[Model]
        The specific model to be tuned. This can be provided as a
        string identifier (e.g., ``'PIHALNet'``, ``'TransFlowSubsNet'``)
        or as the model class object itself.
    fixed_params : dict
        A dictionary containing all parameters that are **not** to be
        tuned. This must include all data-dependent dimensions required
        by the model's constructor, such as:
        - ``static_input_dim``
        - ``dynamic_input_dim``
        - ``future_input_dim``
        - ``output_subsidence_dim``
        - ``output_gwl_dim``
        - ``forecast_horizon``
        It can also include other fixed settings like ``quantiles`` or
        ``mode``.
    search_space : dict, optional
        A dictionary defining the hyperparameter search space. The keys
        are the names of the parameters to tune, and the values define
        their search range or choices.
        
        - For discrete choices, provide a list:``{'num_heads': [2, 4, 8]}``
        - For ranges, provide a dictionary with a ``type`` key:
          ``{'dropout_rate': {'type': 'float', 'min_value': 0.1, 'max_value': 0.4}}``
          Supported types are ``'int'``, ``'float'``, ``'bool'``, and ``'choice'``.
    objective : str or keras_tuner.Objective, default='val_loss'
        The metric to optimize. The direction (min/max) is inferred
        automatically if the name contains "loss".
    max_trials : int, default=10
        The total number of hyperparameter combinations to test.
    project_name : str, default="HydroTuner_Project"
        The name for the tuning project. Results for each trial will
        be stored in a subdirectory within the ``directory``.
    directory : str, default="hydrotuner_results"
        The root directory where tuning results are saved.
    executions_per_trial : int, default=1
        The number of times to train a model with the same set of
        hyperparameters. The final score is averaged.
    tuner_type : {'randomsearch', 'bayesianoptimization', 'hyperband'}, default='randomsearch'
        The search algorithm to use.
    seed : int, optional
        A random seed for reproducibility of the search process.
    overwrite : bool, default=True
        If ``True``, any existing results in the ``project_name``
        directory will be overwritten. Set to ``False`` to resume a
        previous search.
    param_space : dict, optional
        **[DEPRECATED]** Use ``search_space`` instead. This parameter
        is maintained for backward compatibility and will issue a
        ``FutureWarning`` if used.
    **kwargs
        Additional keyword arguments passed to the parent
        :class:`~fusionlab.nn.forecast_tuner.PINNTunerBase` constructor.

    Notes
    -----
    The ``HydroTuner`` separates the model's configuration into two distinct
    parts passed at initialization:

    1.  **`fixed_params`**: Defines the static properties of a given
        modeling problem, primarily the shapes of the data. These do
        not change during the tuning process.
    2.  **`search_space`**: Defines the experiment. This dictionary contains
        all the architectural, physical, and optimization parameters
        that you want to find the optimal values for.

    This design allows the ``build`` method to be completely generic.
    It constructs the ``model_params`` by combining the ``fixed_params``
    with values sampled from the ``search_space`` for each trial, making
    the tuner easily adaptable to new models in the future.

    See Also
    --------
    HydroTuner.create : The recommended factory method for creating a tuner
                        instance by inferring dimensions from data.
    HydroTuner.build : The method that constructs a model for a single trial.
    HydroTuner.run : The main method to start the hyperparameter search.
    fusionlab.nn.pinn.models.PIHALNet : One of the target models for this tuner.
    fusionlab.nn.pinn.models.TransFlowSubsNet : The other primary target model.

    Examples
    --------
    >>> from fusionlab.nn.forecast_tuner import HydroTuner
    >>> from fusionlab.nn.pinn.models import TransFlowSubsNet

    >>> # 1. Define fixed parameters (e.g., from data shapes)
    >>> fixed_params = {
    ...     "static_input_dim": 5,
    ...     "dynamic_input_dim": 8,
    ...     "future_input_dim": 3,
    ...     "output_subsidence_dim": 1,
    ...     "output_gwl_dim": 1,
    ...     "forecast_horizon": 7,
    ...     "quantiles": [0.1, 0.5, 0.9]
    ... }

    >>> # 2. Define the hyperparameter search space
    >>> search_space = {
    ...     # Architectural HPs
    ...     "embed_dim": [32, 64],
    ...     "num_heads": [2, 4],
    ...     "dropout_rate": {"type": "float", "min_value": 0.1, "max_value": 0.3},
    ...     # Physics HPs for TransFlowSubsNet
    ...     "K": ["learnable", 1e-4],
    ...     # Compile-time HPs
    ...     "learning_rate": [1e-3, 5e-4],
    ...     "lambda_gw": {"type": "float", "min_value": 0.5, "max_value": 1.5}
    ... }

    >>> # 3. Instantiate the tuner
    >>> tuner = HydroTuner(
    ...     model_name_or_cls=TransFlowSubsNet,
    ...     fixed_params=fixed_params,
    ...     search_space=search_space,
    ...     objective="val_loss",
    ...     max_trials=20,
    ...     project_name="TransFlowSubsNet_Optimization"
    ... )
    >>> print(f"Tuner configured for model: {tuner.model_class.__name__}")
    Tuner configured for model: TransFlowSubsNet
    """

    def __init__(
        self,
        model_name_or_cls: Union[str, Type[Model]],
        fixed_params: Dict[str, Any],
        search_space: Optional[Dict[str, Any]] = None,
        # Keras-Tuner specific arguments with defaults
        objective: Union[str, 'Objective'] = 'val_loss',
        max_trials: int = 10,
        project_name: str = "HydroTuner_Project",
        directory: str = "hydrotuner_results",
        executions_per_trial: int = 1,
        tuner_type: str = 'randomsearch',
        seed: Optional[int] = None,
        overwrite: bool = True,
        # Legacy parameter for backward compatibility
        param_space: Optional[Dict[str, Any]] = None,
        _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
        **kwargs
    ):
        self._logger = _logger or print 
        # 1. Initialize the parent Keras Tuner HyperModel
        # We explicitly pass all tuner-related arguments to the superclass.
        super().__init__(
            objective=objective,
            max_trials=max_trials,
            project_name=project_name,
            directory=directory,
            executions_per_trial=executions_per_trial,
            tuner_type=tuner_type,
            seed=seed,
            overwrite_tuner=overwrite, 
            _logger= self._logger, 
            **kwargs
        )

        # 2. Handle legacy `param_space` with a deprecation warning
        if param_space is not None:
            if search_space is not None:
                raise ValueError(
                    "Cannot provide both `search_space` and the deprecated"
                    " `param_space`. Please use `search_space` only."
                )
            warnings.warn(
                "The `param_space` argument is deprecated and will be removed"
                " in a future version. Please use `search_space` instead.",
                FutureWarning
            )
            # Transfer value for backward compatibility
            self.search_space = param_space
        else:
            self.search_space = search_space or {}

        # 3. Resolve the model class from string or direct class object
        if isinstance(model_name_or_cls, str):
            # A registry for known, tunable models
            model_map = {
                "PIHALNet": PIHALNet,
                "TransFlowSubsNet": TransFlowSubsNet
            }
            self.model_class = model_map.get(model_name_or_cls)
            if self.model_class is None:
                raise ValueError(
                    f"Unknown model name '{model_name_or_cls}'. "
                    f"Supported names: {list(model_map.keys())}"
                )
        else:
            # Assume a class object was passed directly
            self.model_class = model_name_or_cls

        # 4. Store fixed parameters and validate required keys
        self.fixed_params = fixed_params
        required_keys = [
            "static_input_dim", "dynamic_input_dim", "future_input_dim",
            "output_subsidence_dim", "output_gwl_dim", "forecast_horizon"
        ]
        
        for key in required_keys:
            if key not in self.fixed_params:
                raise ValueError(
                    f"The `fixed_params` dictionary is missing the required"
                    f" key: '{key}'"
                )

    def _create_hyperparameter(
        self, hp: 'HyperParameters', name: str, definition: Any
    ) -> Union[int, float, str, bool]:
        """Dynamically creates a hyperparameter from its definition.
    
        This helper interprets a flexible definition from the
        `search_space` dictionary and uses the appropriate Keras Tuner
        ``hp`` method to create a searchable hyperparameter.
    
        Parameters
        ----------
        hp : keras_tuner.HyperParameters
            The `HyperParameters` object provided by Keras Tuner within
            the `build` method.
        name : str
            The name of the hyperparameter (e.g., "learning_rate").
        definition : Any
            The search space definition for this hyperparameter. It can be:
            - A ``list`` of discrete values (e.g., ``[16, 32, 64]``).
            - A ``dict`` specifying the type and range, e.g.,
              ``{'type': 'float', 'min_value': 0.1, 'max_value': 0.5}``.
              Supported types: 'int', 'float', 'choice', 'bool'.
    
        Returns
        -------
        Union[int, float, str, bool]
            The hyperparameter object (e.g., ``hp.Int(...)``) that Keras
            Tuner can use to sample values.
    
        Raises
        ------
        TypeError
            If the `definition` format is not a supported list or dict.
    
        Notes
        -----
        This method is a core utility used internally by the ``build``
        method to translate the user-friendly `search_space` dictionary
        into concrete Keras Tuner search objects.
        """
        # If the definition is a list, create a Choice hyperparameter.
        if isinstance(definition, list):
            return hp.Choice(name, definition)
    
        # If the definition is a dictionary, parse its type and arguments.
        if isinstance(definition, dict):
            hp_type = definition.get("type", "float")
            # Exclude 'type' key from kwargs passed to Keras Tuner
            hp_kwargs = {k: v for k, v in definition.items() if k != 'type'}
    
            if hp_type == 'int':
                return hp.Int(name, **hp_kwargs)
            elif hp_type == 'float':
                return hp.Float(name, **hp_kwargs)
            elif hp_type == 'choice':
                return hp.Choice(name, **hp_kwargs)
            elif hp_type == 'bool':
                return hp.Boolean(name, **hp_kwargs)
    
        raise TypeError(
            f"Unsupported hyperparameter definition for '{name}': {definition}"
        )
    
    @classmethod
    def create(
        cls,
        model_name_or_cls: Union[str, Type['Model']],
        inputs_data: Dict[str, np.ndarray],
        targets_data: Dict[str, np.ndarray],
        search_space: Dict[str, Any],
        fixed_params: Optional[Dict[str, Any]] = None,
        **tuner_kwargs
    ) -> "HydroTuner":
        """Primary factory method to create and configure a HydroTuner.
    
        This classmethod simplifies the tuner setup by automatically
        inferring essential data-dependent parameters (like input/output
        dimensions and forecast horizon) directly from the provided
        NumPy data arrays. It then intelligently merges these inferred
        parameters with model-specific defaults and any user-provided
        overrides.
    
        Parameters
        ----------
        model_name_or_cls : str or Model Class
            The model to be tuned. Can be a string identifier like
            ``'PIHALNet'`` or ``'TransFlowSubsNet'``, or the model
            class object itself.
        inputs_data : dict of np.ndarray
            A dictionary of input NumPy arrays, keyed by feature type
            (e.g., ``{'static_features': ..., 'dynamic_features': ...}``).
        targets_data : dict of np.ndarray
            A dictionary of target NumPy arrays, keyed by target name
            (e.g., ``{'subsidence': ..., 'gwl': ...}``). The keys will
            be automatically standardized.
        search_space : dict
            The hyperparameter search space definition. Keys are parameter
            names, and values are their search definitions (lists or
            dictionaries).
        fixed_params : dict, optional
            A dictionary of parameters to manually set. These will
            override any inferred values or defaults, giving the user
            final control.
        **tuner_kwargs
            Standard arguments for the Keras Tuner backend, such as
            ``objective``, ``max_trials``, and ``project_name``.
    
        Returns
        -------
        HydroTuner
            An initialized ``HydroTuner`` instance, fully configured and
            ready for the ``.run()`` method to be called.
            
        See Also
        --------
        _infer_and_merge_params : The internal static method that handles
                                  the parameter inference and merging logic.
    
        Examples
        --------
        >>> # Example for tuning a TransFlowSubsNet
        >>> search_space_tf = {
        ...     "learning_rate": [1e-3, 5e-4],
        ...     "dropout_rate": {"type": "float", "min_value": 0.1, "max_value": 0.4},
        ...     "K": ["learnable", 1e-4]
        ... }
        >>> tuner = HydroTuner.create(
        ...     model_name_or_cls="TransFlowSubsNet",
        ...     inputs_data=my_inputs,
        ...     targets_data=my_targets,
        ...     search_space=search_space_tf,
        ...     fixed_params={"forecast_horizon": 7}, # Override inferred horizon
        ...     max_trials=20,
        ...     project_name="MyTransFlowTuning"
        ... )
        """
        # Resolve model name to fetch the correct default parameter set
        model_name = (
            model_name_or_cls if isinstance(model_name_or_cls, str)
            else model_name_or_cls.__name__
        )
    
        # Standardize target keys (e.g., 'subsidence' -> 'subs_pred')
        # This ensures consistency before dimension inference.
        renamed_targets = rename_dict_keys(
            targets_data.copy(),
            param_to_rename={"subsidence": "subs_pred", "gwl": "gwl_pred"}
        )
    
        # Infer and merge all fixed parameters into a single config dict
        final_fixed_params = cls._infer_and_merge_params(
            model_name=model_name,
            inputs_data=inputs_data,
            targets_data=renamed_targets,
            user_fixed_params=fixed_params
        )
    
        # Instantiate the tuner with the complete, finalized configuration
        return cls(
            model_name_or_cls=model_name_or_cls,
            fixed_params=final_fixed_params,
            search_space=search_space,
            **tuner_kwargs
        )

    def build(self, hp: 'HyperParameters') -> 'Model':
        """Builds and compiles a model for a single hyperparameter trial.
    
        This method is called internally by the Keras Tuner for each trial
        in the search process. It dynamically constructs a model instance
        (e.g., ``PIHALNet`` or ``TransFlowSubsNet``) using the combination of
        fixed parameters and the hyperparameters sampled for the current
        trial.
    
        The process involves:
        1. Iterating through the user-defined ``search_space``.
        2. Sampling a value for each hyperparameter using the ``hp`` object.
        3. Separating model initialization arguments from model compilation
           arguments.
        4. Instantiating the model specified by ``self.model_class``.
        5. Compiling the model with the appropriate optimizer, losses, and
           physics-loss weights (:math:`\lambda`).
    
        Parameters
        ----------
        hp : keras_tuner.HyperParameters
            The ``HyperParameters`` object for the current trial, provided
            by the Keras Tuner framework. It is used to sample values for
            each hyperparameter defined in the ``search_space``.
    
        Returns
        -------
        tf.keras.Model
            A fully compiled Keras model instance, ready to be trained for
            the current trial.
    
        Notes
        -----
        This method should not be called directly by the user. It is part
        of the Keras Tuner ``HyperModel`` API and is invoked by the
        tuner's ``search()`` loop. Its generic design allows it to build
        any model whose parameters are defined in the ``search_space``.
        """
        # 1. Initialize parameters with the fixed, non-tunable values.
        model_init_params = self.fixed_params.copy()
        compile_hps = {}
    
        # 2. Sample and assign hyperparameters from the search space.
        for name, definition in self.search_space.items():
            # Sample the value for the current trial using our helper.
            sampled_value = self._create_hyperparameter(hp, name, definition)
    
            # Separate compile-time HPs from model __init__ HPs.
            # This makes the tuner agnostic to the model's compile signature.
            if name in ['learning_rate', 'lambda_gw', 'lambda_cons', 'lambda_physics']:
                compile_hps[name] = sampled_value
            else:
                model_init_params[name] = sampled_value
    
        # Cast HPs that *must* be integers back to int
        for _k in ("embed_dim", "hidden_units", "lstm_units",
                   "attention_units", "vsn_units", "num_heads"):
            if _k in model_init_params:
                model_init_params[_k] = int(model_init_params[_k])

        # 3. Instantiate the specified model class.
        # We filter the dictionary to only pass valid arguments to the model's
        # constructor, preventing errors from extraneous keys.
        valid_init_params = _get_valid_kwargs(
            self.model_class.__init__, model_init_params
        )
        cast_multiple_bool_params(
            valid_init_params,
            bool_params_to_cast=[('use_vsn', False),('use_residuals', True)],
        )
        model = self.model_class(**valid_init_params)
    
        # 4. Prepare and compile the model.
        # Define standard losses and metrics for the dual-output models.
        loss_dict = {
            'subs_pred': MeanSquaredError(name='subs_data_loss'),
            'gwl_pred': MeanSquaredError(name='gwl_data_loss')
        }
        metrics_dict = {
            'subs_pred': [MeanAbsoluteError(name='subs_mae')],
            'gwl_pred': [MeanAbsoluteError(name='gwl_mae')]
        }
    
        optimizer = Adam(
            learning_rate=compile_hps.pop('learning_rate', 1e-3),
            clipnorm=1.0
        )
        
        # Filter the remaining compile HPs (lambdas) to only pass valid
        # ones to the model's compile method. This automatically handles
        # the difference between PIHALNet and TransFlowSubsNet.
        valid_compile_params = _get_valid_kwargs(
            model.compile, compile_hps
        )
    
        model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            metrics=metrics_dict,
            # Default loss weights can be in fixed_params
            loss_weights=self.fixed_params.get(
                'loss_weights', {'subs_pred': 1.0, 'gwl_pred': 0.8}
            ),
            **valid_compile_params
        )
    
        return model
    
    def run(
        self,
        inputs: Dict[str, np.ndarray],
        y: Dict[str, np.ndarray],
        validation_data: Optional[
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        ] = None,
        epochs: int = 10,
        batch_size: int = 32,
        callbacks: Optional[List['Callback']] = None,
        case_info: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        **search_kwargs
    ) -> Tuple[Optional['Model'], Optional['HyperParameters'], Optional['Tuner']]:
        """Executes the end-to-end hyperparameter search workflow.
    
        This is the primary method to start the tuning process. It
        orchestrates data preparation, dimension inference (if needed),
        and the Keras Tuner search loop.
    
        Parameters
        ----------
        inputs : dict of np.ndarray
            A dictionary of NumPy arrays for model inputs. Must contain
            keys like ``"static_features"``, ``"dynamic_features"``, etc.,
            matching the model's requirements.
        y : dict of np.ndarray
            A dictionary of NumPy arrays for targets. It expects keys like
            ``"subsidence"`` and ``"gwl"``, which are automatically
            renamed to match the model's output layers.
        validation_data : tuple, optional
            A tuple ``(val_inputs, val_targets)`` with the same dictionary
            structure as ``inputs`` and ``y``. Used for the validation
            step during tuning. Default is ``None``.
        epochs : int, default=10
            The number of epochs to train each model trial.
        batch_size : int, default=32
            The batch size for creating ``tf.data.Dataset`` objects from the
            input NumPy arrays.
        callbacks : list of keras.callbacks.Callback, optional
            A list of Keras callbacks to use during the search. A default
            ``EarlyStopping`` callback is added automatically if not
            provided. Default is ``None``.
        case_info : dict, optional
            A dictionary of additional metadata to save in the final
            tuning summary JSON file.
        verbose : int, default=1
            Verbosity level for logging during the tuning process.
        **search_kwargs
            Additional keyword arguments forwarded directly to the Keras
            Tuner ``search()`` method.
    
        Returns
        -------
        tuple of (Model | None, HyperParameters | None, Tuner | None)
            A tuple containing:
            - The best model found, retrained on the full dataset.
            - The best hyperparameter configuration object.
            - The Keras Tuner instance containing the search history.
    
        Raises
        ------
        ValueError
            If required keys are missing from the `inputs` or `y`
            dictionaries.
        """
        vlog(
            f"HydroTuner: Starting run for project '{self.project_name}'...",
            verbose=verbose, level=1, logger=self._logger
        )
    
        # 1. Infer and finalize fixed parameters if not fully provided at init.
        # This allows for a flexible workflow where dimensions are determined
        # from the data just before the run starts.
        required_keys = [
            "static_input_dim", "dynamic_input_dim", "future_input_dim",
            "output_subsidence_dim", "output_gwl_dim", "forecast_horizon"
        ]
        if not all(k in self.fixed_params for k in required_keys):
            vlog(
                "Inferring missing fixed parameters from provided data.",
                verbose=verbose, level=2, logger=self._logger
            )
            self.fixed_params = self._infer_and_merge_params(
                model_name=self.model_class.__name__,
                inputs_data=inputs,
                targets_data=y,
                user_fixed_params=self.fixed_params
            )
    
        # 2. Standardize target dictionary keys for model compatibility.
        y = rename_dict_keys(
            y.copy(),
            param_to_rename={"subsidence": "subs_pred", "gwl": "gwl_pred"}
        )
        # Ensure all required keys are present after potential renaming
        check_required_input_keys(inputs, y)
    
        # 3. Prepare tf.data.Dataset for training.
        train_dataset = Dataset.from_tensor_slices((inputs, y)).batch(
            batch_size).prefetch(AUTOTUNE)
    
        # 4. Prepare validation dataset if provided.
        val_dataset = None
        if validation_data:
            val_inputs, val_targets = validation_data
            val_targets = rename_dict_keys(
                val_targets.copy(),
                param_to_rename={"subsidence": "subs_pred", "gwl": "gwl_pred"}
            )
            check_required_input_keys(val_inputs, val_targets)
            val_dataset = Dataset.from_tensor_slices(
                (val_inputs, val_targets)
            ).batch(batch_size).prefetch(AUTOTUNE)
    
        # 5. Prepare metadata for logging and summary.
        self._current_run_case_info = DEFAULT_CASE_INFO.copy()
        self._current_run_case_info.update(self.fixed_params)
        forecast_type = "Quantile" if self.fixed_params.get('quantiles') else "Point"
        self._current_run_case_info["description"] = self._current_run_case_info[
            "description"
        ].format(model_name=self.model_class.__name__, forecast_type=forecast_type)
        if case_info:
            self._current_run_case_info.update(case_info)
    
        # 6. Execute the search by calling the parent's search method.
        vlog("Handing off to Keras Tuner search...", verbose=verbose, level=2, 
             logger=self._logger )
        return super().search(
            train_data=train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=verbose,
            **search_kwargs
        )

    @staticmethod
    def _infer_and_merge_params(
        model_name: str,
        inputs_data: Dict[str, np.ndarray],
        targets_data: Dict[str, np.ndarray],
        user_fixed_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Infers data dimensions and merges with defaults and user overrides.
    
        This static helper is the core of the tuner's factory pattern. It
        determines the final set of "fixed" (non-tunable) parameters
        required to instantiate a model.
    
        The merging follows a strict order of precedence:
        1. Base values are taken from a model-specific default dictionary
           (e.g., ``DEFAULT_PIHALNET_PARAMS``).
        2. Data-dependent dimensions (e.g., ``dynamic_input_dim``) are
           inferred from the shapes of the provided ``inputs_data`` and
           ``targets_data`` arrays, overriding the defaults.
        3. Any parameters explicitly provided in ``user_fixed_params`` will
           override all other values, giving the user the final say.
    
        Parameters
        ----------
        model_name : str
            The name of the model ('PIHALNet' or 'TransFlowSubsNet') used
            to select the appropriate set of default parameters.
        inputs_data : dict of np.ndarray
            A dictionary of input NumPy arrays, keyed by feature type
            (e.g., ``{'static_features': ..., 'dynamic_features': ...}``).
        targets_data : dict of np.ndarray
            A dictionary of target NumPy arrays, keyed by standardized
            target name (e.g., ``{'subs_pred': ..., 'gwl_pred': ...}``).
        user_fixed_params : dict, optional
            A dictionary of parameters to manually set. These values take
            the highest priority. Default is ``None``.
    
        Returns
        -------
        dict
            A single, complete dictionary of fixed parameters ready to be
            used for instantiating the ``HydroTuner``.
    
        Raises
        ------
        ValueError
            If required keys (like 'dynamic_features' or 'subs_pred') are
            missing from the input data dictionaries.
        """
        # 1. Select the correct default parameter set based on model name
        if model_name == 'PIHALNet':
            final_params = DEFAULT_PIHALNET_PARAMS.copy()
        elif model_name == 'TransFlowSubsNet':
            final_params = DEFAULT_TRANSFLOW_PARAMS.copy()
        else:
            # Fallback for future models to avoid errors
            final_params = {}
            warnings.warn(
                f"No default parameter set found for model '{model_name}'."
                " Relying solely on inference and user-provided parameters."
            )
    
        # 2. Infer data-dependent dimensions from the shapes of the arrays
        inferred_params = {}
    
        # This check ensures all required arrays are present before access
        check_required_input_keys(inputs_data, targets_data)
    
        # Infer input dimensions
        if "static_features" in inputs_data:
            inferred_params["static_input_dim"] = \
                inputs_data["static_features"].shape[-1]
        else:
            inferred_params["static_input_dim"] = 0
    
        inferred_params["dynamic_input_dim"] = \
            inputs_data["dynamic_features"].shape[-1]
    
        if "future_features" in inputs_data:
            inferred_params["future_input_dim"] = \
                inputs_data["future_features"].shape[-1]
        else:
            inferred_params["future_input_dim"] = 0
    
        # Infer output dimensions and forecast horizon
        inferred_params["output_subsidence_dim"] = \
            targets_data["subs_pred"].shape[-1]
        inferred_params["output_gwl_dim"] = \
            targets_data["gwl_pred"].shape[-1]
        inferred_params["forecast_horizon"] = \
            targets_data["subs_pred"].shape[1]
    
        # 3. Merge parameters with the correct precedence
        # Start with defaults, update with inferred values, then apply
        # user overrides.
        final_params.update(inferred_params)
        if user_fixed_params:
            final_params.update(user_fixed_params)
    
        return final_params
    
    def _get_hp_choice(self, hp, name, default_choices, **kwargs):
        return hp.Choice(
            name,
            self.param_space.get(name, default_choices),
            **kwargs
        )

    def _parse_hp_config(
        self,
        hp,
        name,
        default_min,
        default_max,
        default_step_or_sampling,
        hp_type
    ):
        """
        Helper to interpret `param_space[name]` which may be:
          - A list of explicit values (use hp.Choice).
          - A dict with keys 'min_value', 'max_value', and for ints 'step', for
            floats 'sampling'.
          - None or other (fallback to defaults).
        """
        config = self.search_space.get(name, None)
    
        # If user provided a list of discrete values, use Choice
        if isinstance(config, list):
            return hp.Choice(name, config)
    
        # If user provided a dict with min/max settings
        if isinstance(config, dict):
            min_val = config.get('min_value', default_min)
            max_val = config.get('max_value', default_max)
    
            if hp_type == 'int':
                step_val = config.get('step', default_step_or_sampling)
                return hp.Int(
                    name,
                    min_value=min_val,
                    max_value=max_val,
                    step=step_val
                )
            # hp_type == 'float'
            sampling_val = config.get(
                'sampling',
                default_step_or_sampling
            )
            return hp.Float(
                name,
                min_value=min_val,
                max_value=max_val,
                sampling=sampling_val
            )
    
        # Fallback: no config or unexpected type, use defaults
        if hp_type == 'int':
            return hp.Int(
                name,
                min_value=default_min,
                max_value=default_max,
                step=default_step_or_sampling
            )
        return hp.Float(
            name,
            min_value=default_min,
            max_value=default_max,
            sampling=default_step_or_sampling
        )

    def _get_hp_int(
            self,
            hp,
            name,
            default_min,
            default_max,
            step=1,
            **kwargs
        ):
        """
        Retrieves or creates an integer hyperparameter.  The user may define in
        `param_space[name]` either:
          - A list of discrete integer values → uses hp.Choice
          - A dict with 'min_value', 'max_value', 'step'
          - None → fallback to default_min, default_max, step
        """
        return self._parse_hp_config(
            hp,
            name,
            default_min,
            default_max,
            step,
            hp_type='int'
        )
    
    
    def _get_hp_float(
            self,
            hp,
            name,
            default_min,
            default_max,
            default_sampling=None,
            **kwargs
        ):
        """
        Retrieves or creates a float hyperparameter.  The user may define in
        `param_space[name]` either:
          - A list of discrete float values → uses hp.Choice
          - A dict with 'min_value', 'max_value', 'sampling'
          - None → fallback to default_min, default_max, default_sampling
        """
        return self._parse_hp_config(
            hp,
            name,
            default_min,
            default_max,
            default_sampling,
            hp_type='float'
        )
    
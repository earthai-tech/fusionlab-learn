# -*- coding: utf-8 -*-
# File: fusionlab/nn/forecast_tuner/hal_tuner.py
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

"""
Provides a dedicated Keras Tuner class for hyperparameter
optimization of the HALNet model.
"""
from __future__ import annotations
from typing import Dict, Optional, Any, Union, List, Tuple

import numpy as np 

from ...api.docs import DocstringComponents, _pinn_tuner_common_params 
from ..._fusionlog import fusionlog
from ...core.handlers import _get_valid_kwargs
from ...utils.generic_utils import ( 
    vlog,  cast_multiple_bool_params
   )
from ._base_tuner import PINNTunerBase
from ..models import HALNet
from ..losses import combined_quantile_loss

from .. import KERAS_DEPS
from . import KT_DEPS

HyperParameters = KT_DEPS.HyperParameters
Objective = KT_DEPS.Objective

Model = KERAS_DEPS.Model
Callback = KERAS_DEPS.Callback
Dataset = KERAS_DEPS.Dataset 
Adam = KERAS_DEPS.Adam
EarlyStopping = KERAS_DEPS.EarlyStopping
AUTOTUNE =KERAS_DEPS.AUTOTUNE 
    
Model = KERAS_DEPS.Model
Adam = KERAS_DEPS.Adam
MeanSquaredError = KERAS_DEPS.MeanSquaredError
MeanAbsoluteError = KERAS_DEPS.MeanAbsoluteError
Dataset = KERAS_DEPS.Dataset
AUTOTUNE = KERAS_DEPS.AUTOTUNE
Tensor = KERAS_DEPS.Tensor


# Wrap into a DocstringComponents object once
_pinn_tuner_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_pinn_tuner_common_params)
)

logger = fusionlog().get_fusionlab_logger(__name__)

__all__ = ["HALTuner"]

# Default fixed parameters for HALNet, used when they cannot be inferred.
DEFAULT_HALNET_FIXED_PARAMS = {
    "output_dim": 1,
    "forecast_horizon": 1,
    "quantiles": None,
    "max_window_size": 10,
    "memory_size": 100,
    "scales": [1],
    "multi_scale_agg": 'last',
    "final_agg": 'last',
    "use_residuals": True,
    "use_vsn": True,
    "vsn_units": 32,
    "activation": "relu",
    "mode": "tft_like", # Default operational mode
}

class HALTuner(PINNTunerBase):
    """
    A Keras Tuner for hyperparameter optimization of the HALNet model.

    This class inherits from `PINNTunerBase` and implements the `build`
    method to construct and compile `HALNet` instances with different
    hyperparameter combinations.

    Parameters
    ----------
    fixed_model_params : Dict[str, Any]
        A dictionary of parameters for the `HALNet` model that remain
        constant during tuning. This must include `static_input_dim`,
        `dynamic_input_dim`, and `future_input_dim`.

    param_space : Dict[str, Any], optional
        A dictionary defining the hyperparameter search space. If not
        provided, a default search space within the `build` method
        is used.

    objective : str or keras_tuner.Objective, default 'val_loss'
        The metric to optimize during the search.

    max_trials : int, default 20
        The maximum number of hyperparameter combinations to test.

    project_name : str, default "HALNet_Tuning"
        The name for the tuning project.

    **tuner_kwargs : Any
        Additional keyword arguments passed to the `PINNTunerBase` and
        underlying Keras Tuner constructor (e.g., `tuner_type`, `seed`).
    """
    def __init__(
        self,
        fixed_model_params: Dict[str, Any],
        param_space: Optional[Dict[str, Any]] = None,
        objective: Union[str, Objective] = 'val_loss',
        max_trials: int = 20,
        directory: str = "hal_tuner_results",
        executions_per_trial: int = 1,
        tuner_type: str = 'randomsearch',
        seed: Optional[int] = None,
        overwrite_tuner: bool = True,
        project_name: str = "HALNet_Tuning",
        **tuner_kwargs
    ):
        super().__init__(
            objective=objective,
            max_trials=max_trials,
            project_name=project_name,
            directory = directory,
            executions_per_trial= executions_per_trial,
            tuner_type= tuner_type,
            seed= seed,
            overwrite_tuner= overwrite_tuner,
            **tuner_kwargs
        )
        self.fixed_model_params = fixed_model_params
        self.param_space = param_space or {}
        
        if 'search_space' in tuner_kwargs: 
            self.param_space = tuner_kwargs.pop('search_space')
            
        # Validate that essential data-dependent dimensions are provided.
        required_fixed = [
            "static_input_dim", "dynamic_input_dim", "future_input_dim",
            "output_dim", "forecast_horizon"
        ]
        for req_param in required_fixed:
            if req_param not in self.fixed_model_params:
                raise ValueError(
                    "Missing required key in `fixed_model_params`: "
                    f"'{req_param}'"
                )
        vlog(f"{self.__class__.__name__} initialized for project "
             f"'{self.project_name}'.", level=1,
             verbose=self.tuner_kwargs.get('verbose', 1))

    def build(self, hp: HyperParameters) -> Model:
        """
        Builds and compiles a HALNet model for a given trial.
        """
        verbose = self.tuner_kwargs.get('verbose', 1)
        # Try to retrieve the trial‑id if it exists
        trial_id = getattr(getattr(hp, "trial", None), "trial_id", "manual")
        vlog(f"Building trial {trial_id}...", level=2,
             verbose=verbose)
        
        # --- Sample Architectural Hyperparameters ---
        embed_dim = self._get_hp_int(hp, 'embed_dim', 16, 64, 16)
        hidden_units = self._get_hp_int(hp, 'hidden_units', 32, 128, 32)
        lstm_units = self._get_hp_int(hp, 'lstm_units', 32, 128, 32)
        attention_units = self._get_hp_int(
            hp, 'attention_units', 16, 64, 16
        )
        num_heads = self._get_hp_choice(hp, 'num_heads', [2, 4])
        dropout_rate = self._get_hp_float(
            hp, 'dropout_rate', 0.0, 0.3, step=0.1
        )
        use_vsn = self._get_hp_choice(hp, 'use_vsn', [True, False])

        # --- Sample Training Hyperparameters ---
        learning_rate = self._get_hp_choice(
            hp, 'learning_rate', [1e-3, 5e-4, 1e-4]
        )
        
        cast_multiple_bool_params(
            self.fixed_model_params,
            bool_params_to_cast=[('use_vsn', False),
                                 ('use_residuals', True)],
        )
        # Merge fixed and hyper-parameters
        model_params = {
            **self.fixed_model_params,
            "embed_dim": embed_dim,
            "hidden_units": hidden_units,
            "lstm_units": lstm_units,
            "attention_units": attention_units,
            "num_heads": num_heads,
            "dropout_rate": dropout_rate,
            "use_vsn": use_vsn,
        }

        # Build the HALNet model
        model = HALNet(**_get_valid_kwargs(HALNet, model_params))
        
        # Determine the loss function based on quantiles
        if self.fixed_model_params.get("quantiles"):
            loss = combined_quantile_loss(
                self.fixed_model_params["quantiles"])
        else:
            loss = MeanSquaredError()

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=[MeanAbsoluteError()]
        )
        vlog(f"Trial {trial_id} model built and compiled.",
             level=3, verbose=verbose)
        
        return model

    @classmethod
    def create(
        cls,
        inputs_data: List[np.ndarray],
        targets_data: np.ndarray,
        fixed_model_params: Optional[Dict[str, Any]] = None,
        verbose: int =1, 
        **kwargs
    ) -> "HALTuner":
        """
        A factory method to create a HALTuner instance by inferring
        dimensions from data.
        """
        vlog("Creating HALTuner instance from data...", level=1,
             verbose=verbose)
        
        fixed_params = fixed_model_params or {}
        
        # Infer dimensions from data shapes
        inferred_params = {
            "static_input_dim": inputs_data[0].shape[-1],
            "dynamic_input_dim": inputs_data[1].shape[-1],
            "future_input_dim": inputs_data[2].shape[-1],
            "output_dim": targets_data.shape[-1],
            "forecast_horizon": targets_data.shape[1],
            "max_window_size": inputs_data[1].shape[1]
        }
        vlog(f"Inferred dimensions from data: {inferred_params}",
             level=2, verbose=verbose)
        # User-provided params override inferred ones
        final_fixed_params = {**inferred_params, **fixed_params}
        
        return cls(fixed_model_params=final_fixed_params, **kwargs)

    def run(
        self,
        inputs: List[np.ndarray],
        y: np.ndarray,
        validation_data: Optional[Tuple[List[np.ndarray], np.ndarray]] = None,
        epochs: int = 10,
        batch_size: int = 32,
        **search_kwargs
    ):
        """
        A user-friendly wrapper around the `search` method. It creates
        tf.data.Dataset objects before initiating the search.
        """
        verbose = search_kwargs.get('verbose', 1)
        vlog("Preparing tf.data.Dataset objects for tuning...", level=1,
             verbose=verbose)
        
        # ------------------------------------------------------------------
        # 1. Wrap numpy arrays so tf.data can slice them sample‑wise
        # ------------------------------------------------------------------
        # `inputs` is a list like [static, dynamic, future].  We convert it
        # to a *tuple* so that `from_tensor_slices` interprets it as
        #   (x1, x2, x3)  paired with  y,
        # producing elements of the form  ((x1_i, x2_i, x3_i), y_i).
        # A Python list would be treated as a single ragged object and
        # trigger "non‑rectangular sequence" errors.
        train_dataset = (
            Dataset.from_tensor_slices((tuple(inputs), y))
                   .batch(batch_size)
                   .prefetch(AUTOTUNE)
        )
        vlog(f"Training dataset created with {len(y)} samples.",
             level=2, verbose=verbose)
        
        val_dataset = None
        if validation_data is not None:
            val_inputs, val_y = validation_data
            val_dataset = (
                Dataset.from_tensor_slices((tuple(val_inputs), val_y))
                       .batch(batch_size)
                       .prefetch(AUTOTUNE)
            )
            vlog(f"Validation dataset created with {len(val_y)} samples.",
                 level=2, verbose=verbose)


        # train_dataset = Dataset.from_tensor_slices((inputs, y))
        # train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)

        # vlog(f"Training dataset created with {len(y)} samples.",
        #      level=2, verbose=verbose)
        
        # val_dataset = None
        # if validation_data:
        #     val_inputs, val_y = validation_data
        #     val_dataset = Dataset.from_tensor_slices((val_inputs, val_y))
        #     val_dataset = val_dataset.batch(batch_size).prefetch(AUTOTUNE)
        #     vlog(f"Validation dataset created with {len(val_y)} samples.",
        #          level=2, verbose=verbose)


        return super().search(
            train_data=train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            **search_kwargs
        )

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
        config = self.param_space.get(name, None)
    
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
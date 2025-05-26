# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Hyperparameter tuning module for Temporal Fusion Transformer 
(TFT) and Extreme TFT models.

This module provides tuners for optimizing the 
hyperparameters of neural network models for time-series
forecasting, including `XTFT`, `SuperXTFT`, and 
`TemporalFusionTransformer`. It utilizes Keras Tuner (`kt`) to 
perform Bayesian and Random Search optimization.

Key Functions:
- `xtft_tuner`: Tunes XTFT and SuperXTFT models.
- `tft_tuner`: Tunes Temporal Fusion Transformer (TFT) models.

Dependencies:
- TensorFlow/Keras for deep learning models.
- Keras Tuner for hyperparameter optimization.
"""

import os
import warnings
import json
from numbers import Real, Integral
from typing import (
    Union, Dict, Any, Optional, Callable, List, Tuple, 
    TYPE_CHECKING
)

import numpy as np

from .__init__ import config
from ..api.docs import _tuner_common_params, DocstringComponents 
from ..api.summary import ResultSummary 
from ..compat.sklearn import validate_params, Interval
from ..core.checks import (
    check_params, check_non_emptiness
    )
from ..core.handlers import param_deprecated_message
from ..core.io import _get_valid_kwargs 
from ..utils.deps_utils import ensure_pkg
from ..utils.generic_utils import vlog
from ._tensor_validation import validate_model_inputs

from . import KERAS_DEPS, KERAS_BACKEND, dependency_message
from .losses import combined_quantile_loss
from ._forecast_tuner import( 
    CASE_INFO, DEFAULT_PS,
    BaseTuner
)
from .transformers import (
    XTFT,
    SuperXTFT,
    TemporalFusionTransformer as TFTFlexible, 
    TFT as TFTStricter 
)
HAS_KT = False
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
    Adam = KERAS_DEPS.Adam
    Model = KERAS_DEPS.Model
    Tensor = KERAS_DEPS.Tensor
    EarlyStopping = KERAS_DEPS.EarlyStopping
    tf_convert_to_tensor = KERAS_DEPS.convert_to_tensor
    tf_float32 = KERAS_DEPS.float32
    tf_zeros = KERAS_DEPS.zeros 
    tf_shape =KERAS_DEPS.shape 
    
else:
    class Model: pass
    class Tensor: pass
    class Adam: pass
    class EarlyStopping: pass

DEP_MSG = dependency_message('nn.forecast_tuner')

_tuner_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_tuner_common_params)
)

__all__ = ['XTFTTuner', 'TFTTuner', 'xtft_tuner', 'tft_tuner']


class XTFTTuner(BaseTuner):
    _DEFAULT_MODEL_NAME: str = "xtft"
    _SUPPORTED_XTFT_MODELS: List[str] = [
        "xtft",
        "superxtft",
        "super_xtft",
    ]

    def __init__(  # noqa: D401
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        param_space: Optional[Dict[str, Any]] = None,
        max_trials: int = 10,
        objective: str = "val_loss",
        epochs: int = 10,
        batch_sizes: List[int] = [32],
        validation_split: float = 0.2,
        tuner_dir: Optional[str] = None,
        project_name: Optional[str] = None,
        tuner_type: str = "random",
        callbacks: Optional[List[Callable]] = None,
        model_builder: Optional[Callable] = None,
        verbose: int = 1,
        **kws: Any,
    ):
        model_name_lower = model_name.lower()
        if model_name_lower not in self._SUPPORTED_XTFT_MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Choose from {self._SUPPORTED_XTFT_MODELS}."
            )

        super().__init__(
            model_name=model_name_lower,
            param_space=param_space,
            max_trials=max_trials,
            objective=objective,
            epochs=epochs,
            batch_sizes=batch_sizes,
            validation_split=validation_split,
            tuner_dir=tuner_dir,
            project_name=project_name,
            tuner_type=tuner_type,
            callbacks=callbacks,
            model_builder=model_builder,
            verbose=verbose,
            **kws,
        )

XTFTTuner.__doc__ = """
Keras‑Tuner wrapper dedicated to Extreme Temporal Fusion Transformer
families (XTFT, SuperXTFT).

The tuner searches a composite hyper‑parameter space covering
embedding size, window length, memory depth, attention width, LSTM
layers, residual routing, and optimiser learning rate.  Each batch
size listed in *batch_sizes* triggers an independent search followed
by a *refit* phase that prolongs training on the best trial.

Mathematically the optimisation objective is

.. math::

   \\theta^* \\;=\\; \\arg\\min_{{\\theta\\,\\in\\,\\Theta}}
   \\;\\; L_{{\\text{{val}}}}\\bigl(f_{{\\theta}}(\\boldsymbol X),\\,
   \\boldsymbol y\\bigr),

where :math:`\\Theta` denotes the hyper‑parameter space and
:math:`L_{{\\text{{val}}}}` the validation loss.

Supported model aliases
~~~~~~~~~~~~~~~~~~~~~~~
- ``"xtft"``
- ``"superxtft"``
- ``"super_xtft"``

Parameters
----------
{params.base.model_name}
{params.base.param_space}
{params.base.max_trials}
{params.base.objective}
{params.base.epochs}
{params.base.batch_sizes}
{params.base.validation_split}
{params.base.tuner_dir}
{params.base.project_name}
{params.base.tuner_type}
{params.base.callbacks}
{params.base.model_builder}
{params.base.verbose}

**kws : Any
    Extra keyword arguments forwarded verbatim to
    :class:`~fusionlab.nn._forecast_tuner.BaseTuner`.

Examples
--------
Create synthetic data, instantiate the tuner, and run a full search.

>>> import numpy as np, tensorflow as tf
>>> from fusionlab.forecast_tuner import XTFTTuner
>>> B, F, Ns, Nd, Nf, O = 128, 6, 3, 4, 2, 1
>>> rng = np.random.default_rng(42)
>>> X_static  = rng.normal(size=(B, Ns)).astype("float32")
>>> X_dynamic = rng.normal(size=(B, F, Nd)).astype("float32")
>>> X_future  = rng.normal(size=(B, F, Nf)).astype("float32")
>>> y         = rng.normal(size=(B, F, O)).astype("float32")
>>>
>>> tuner = XTFTTuner(
...     max_trials=5,
...     epochs=50,
...     batch_sizes=[16, 32],
...     tuner_type="bayesian",
...     verbose=2,
... )
>>> best_hps, best_model, kt_obj = tuner.fit(
...     inputs=[X_static, X_dynamic, X_future],
...     y=y,
...     forecast_horizon=F,
...     quantiles=[0.1, 0.5, 0.9],
... )
>>> print(best_hps)

See Also
--------
TFTTuner
    Analogue tuner class for Temporal Fusion Transformer variants.
fusionlab.nn.transformers.XTFT
    Reference implementation of the model being tuned.

References
----------
.. [1] McKinney, W. (2010). "Data Structures for Statistical Computing 
       in Python". Proceedings of the 9th Python in Science Conference.
.. [2] Van der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). "The 
       NumPy Array: A Structure for Efficient Numerical Computation". 
       Computing in Science & Engineering, 13(2), 22-30.
       
""".format(params=_tuner_docs)

class TFTTuner(BaseTuner):
    _DEFAULT_MODEL_NAME: str = "tft"
    _SUPPORTED_TFT_MODELS: List[str] = ["tft", "tft_flex"]

    def __init__(  # noqa: D401
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        param_space: Optional[Dict[str, Any]] = None,
        max_trials: int = 10,
        objective: str = "val_loss",
        epochs: int = 10,
        batch_sizes: List[int] = [32],
        validation_split: float = 0.2,
        tuner_dir: Optional[str] = None,
        project_name: Optional[str] = None,
        tuner_type: str = "random",
        callbacks: Optional[List[Callable]] = None,
        model_builder: Optional[Callable] = None,
        verbose: int = 1,
        **kws: Any,
    ):
        model_name_lower = model_name.lower()
        if model_name_lower not in self._SUPPORTED_TFT_MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Choose from {self._SUPPORTED_TFT_MODELS}."
            )

        super().__init__(
            model_name=model_name_lower,
            param_space=param_space,
            max_trials=max_trials,
            objective=objective,
            epochs=epochs,
            batch_sizes=batch_sizes,
            validation_split=validation_split,
            tuner_dir=tuner_dir,
            project_name=project_name,
            tuner_type=tuner_type,
            callbacks=callbacks,
            model_builder=model_builder,
            verbose=verbose,
            **kws,
        )

TFTTuner.__doc__ = """
Hyper‑parameter optimiser for Temporal Fusion Transformer (TFT)
architectures – both the stricter TFT and the flexible
TemporalFusionTransformer (``'tft_flex'``).

The tuner explores LSTM depth, attention heads, hidden size,
drop‑out, batch‑norm toggles, and optimiser learning rate.  Every
candidate batch size launches its own search loop followed by a
refit on the champion trial, yielding a single best model across
all batch sizes.

Objective
~~~~~~~~~
.. math::

   \\theta^* \\;=\\; \\arg\\min_{{\\theta\\in\\Theta}}\\;
   L_{{\\text{{val}}}}\\bigl(f_{{\\theta}}(\\mathbf X),\\,\\mathbf y\\bigr)

with :math:`\\Theta` the joint hyper‑parameter space and
:math:`L_{{\\text{{val}}}}` the validation loss.

Supported model aliases
~~~~~~~~~~~~~~~~~~~~~~~
- ``"tft"``
- ``"tft_flex"``

Parameters
----------
{params.base.model_name}
{params.base.param_space}
{params.base.max_trials}
{params.base.objective}
{params.base.epochs}
{params.base.batch_sizes}
{params.base.validation_split}
{params.base.tuner_dir}
{params.base.project_name}
{params.base.tuner_type}
{params.base.callbacks}
{params.base.model_builder}
{params.base.verbose}

**kws : Any
    Extra keyword arguments forwarded to the base tuner.

Example
-------
Create synthetic data, instantiate the tuner, and run a full search.

>>> import numpy as np, tensorflow as tf
>>> from fusionlab.forecast_tuner import TFTTuner
>>> B, F, Ns, Nd, Nf, O = 128, 6, 3, 4, 2, 1
>>> rng = np.random.default_rng(42)
>>> X_static  = rng.normal(size=(B, Ns)).astype("float32")
>>> X_dynamic = rng.normal(size=(B, F, Nd)).astype("float32")
>>> X_future  = rng.normal(size=(B, F, Nf)).astype("float32")
>>> y         = rng.normal(size=(B, F, O)).astype("float32")
>>>
>>> tuner = TFTTuner(
...     model_name="tft_flex",
...     max_trials=4,
...     epochs=30,
...     batch_sizes=[32, 64],
...     tuner_type="bayesian",
...     verbose=2,
... )
>>> best_hps, best_model, kt_obj = tuner.fit(
...     inputs=[X_static, X_dynamic, X_future],
...     y=y,
...     forecast_horizon=F,
... )
>>> print("Best learning‑rate:",
...       f"{{best_hps['learning_rate']:.3g}}")

See Also
--------
XTFTTuner
    Companion tuner for Extreme TFT variants.
fusionlab.nn.transformers.TFT
    Strict reference implementation.
fusionlab.nn.transformers.TemporalFusionTransformer
    Flexible implementation accepting missing input blocks.

References
----------
.. [1] Lim, B. & Zohren, S. (2021). *Temporal Fusion Transformers
       for Interpretable Multi‑horizon Time Series Forecasting*,
       Int. J. Forecasting, 37(4), 1748‑1764.
.. [2] Ouali, Y. et al. (2022). *Improving Temporal Fusion
       Transformers with Hierarchical Context Modelling*,
       arXiv:2202.01176.
""".format(params=_tuner_docs)




@ensure_pkg(
    'keras_tuner',
    extra="'keras_tuner' is required for model tuning.",
    auto_install=config.INSTALL_DEPS,
    use_conda=config.USE_CONDA
)
@param_deprecated_message(
    conditions_params_mappings=[
        {
            'param': 'tuner_type',
            'condition': lambda v: v not in {'bayesian', 'random'},
            'message': (
                "Tuner type supports 'bayesian' or 'random'. "
                "Defaulting to 'random'."
            ),
            'default': "random"
        }
    ],
    warning_category=UserWarning
)
@check_params({
    'tuner_dir': Optional[str],
    'project_name': Optional[str]
    })
@validate_params({
    'inputs': ['array-like'],
    'y': ['array-like'],
    'param_space': [dict, None],
    'forecast_horizon': [Interval(Integral, 1, None, closed="left")],
    'quantiles': ['array-like', None],
    'case_info': [dict, None],
    'max_trials': [Interval(Integral, 1, None, closed ='left')],
    'objective': [str],
    'epochs': [Interval(Integral, 1, None, closed ='left')],
    'batch_sizes': ['array-like'],
    'validation_split': [Interval(Real, 0, 1, closed='neither')],
    'tuner_type': [str],
    'model_name': [str],
    })
@check_non_emptiness 
def xtft_tuner(
    inputs: List[Optional[Union[np.ndarray, Tensor]]],
    y: Union[np.ndarray, Tensor],
    param_space: Optional[Dict[str, Any]] = None,
    forecast_horizon: int = 1,
    quantiles: Optional[List[float]] = None,
    case_info: Optional[Dict[str, Any]] = None,
    max_trials: int = 10,
    objective: str = 'val_loss',
    epochs: int = 10, # Epochs for final training of best model
    batch_sizes: List[int] = [32],
    validation_split: float = 0.2,
    tuner_dir: Optional[str] = None,
    project_name: Optional[str] = None,
    tuner_type: str = 'random',
    callbacks: Optional[List[Callable]] = None,
    model_builder: Optional[Callable] = None,
    model_name: str = "xtft",
    verbose: int = 1, 
    **kws
) -> Tuple[Optional[Dict], Optional[Model], Optional[kt.Tuner]]:
    """
    Fine-tunes XTFT, SuperXTFT, or TFT (stricter) models.

    """
    if not HAS_KT: # Check if Keras Tuner was imported
        raise ImportError(
            "keras_tuner is not installed. Please install it via "
            "`pip install keras-tuner` to use this tuning function."
            )

    # Determine loss function based on quantiles
    current_loss_fn_obj: Union[str, Callable] = "mse"
    if quantiles is not None:
        current_loss_fn_obj = combined_quantile_loss(quantiles)

    # Prepare case_info for the model builder
    run_case_info = CASE_INFO.copy() # Start with global defaults
    run_case_info.update({
        'description': run_case_info['description'].format(
            "Quantile" if quantiles is not None else "Point"),
        'forecast_horizon': forecast_horizon, # Use correct key
        'quantiles': quantiles,
        'output_dim': y.shape[-1] if y.ndim == 3 else 1,
        'verbose_build': verbose >= 3 # Pass high verbosity to builder
    })
    if case_info: # User-provided case_info overrides
        run_case_info.update(case_info)

    # Prepare parameter space for the tuner
    current_default_ps = DEFAULT_PS.copy()
    current_default_ps['loss'] = current_loss_fn_obj

    def get_param_space_value(name: str, default_override=None):
        user_ps = param_space or {}
        # User's space -> current defaults (with dynamic loss) -> hardcoded default
        return user_ps.get(name, current_default_ps.get(name, default_override))

    vlog(f"Starting {model_name.upper()} {tuner_type.upper()} tune...",
         level=1, verbose=verbose)

    # Validate model_name
    valid_model_names = {
        "xtft", 'superxtft', 'super_xtft', 'tft', 'tft_flex'
        }
    model_name_lower = model_name.lower()
    if model_name_lower not in valid_model_names:
        raise ValueError(
            f"Unsupported model_name: '{model_name}'. Must be one of "
            f"{valid_model_names}."
        )

    # --- Input Validation ---
    # User provides inputs as [Static, Dynamic, Future]
    if model_name !='tft_flex':
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
            raise ValueError(
                "Inputs must be a list/tuple of 3 elements: "
                "[X_static, X_dynamic, X_future]."
                )
    # Ensure y is a tensor for shape operations
    y_tensor = tf_convert_to_tensor(y, dtype=tf_float32)
          
    # `validate_tft_inputs` expects [S, D, F] and returns (D_p, F_p, S_p)
    # This specific order is for the internal structure of TFT model.
    # The flexible `TemporalFusionTransformer` handles Nones internally.
    # --- Prepare inputs_for_validator (always a list of 3) ---
    s_input_raw: Optional[Union[np.ndarray, Tensor]] = None
    d_input_raw: Optional[Union[np.ndarray, Tensor]] = None
    f_input_raw: Optional[Union[np.ndarray, Tensor]] = None

    if model_name_lower == 'tft_flex':
        if not isinstance(inputs, (list, tuple)): # Single tensor
            d_input_raw = inputs # Assumed dynamic
        elif len(inputs) == 1:
            d_input_raw = inputs[0] # Assumed dynamic
        elif len(inputs) == 2:
            # `validate_model_inputs` with `mode='soft'` will infer roles
            # We pass it as [in0, in1, None]
            # For deriving initial dims, we need to make a temp assumption
            # or let validate_model_inputs handle it entirely.
            # For now, pass as is to validator, it will sort it out.
            s_input_raw, d_input_raw, f_input_raw = (
                inputs[0], inputs[1], None) # Temp assignment
        elif len(inputs) == 3:
            s_input_raw, d_input_raw, f_input_raw = inputs
        else:
            raise ValueError(
                "For 'tft_flex', inputs must be a single tensor or a "
                "list/tuple of 1, 2, or 3 elements."
            )
    else: # Stricter models (xtft, superxtft, tft)
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
            raise ValueError(
                f"Model '{model_name}' requires inputs as a list/tuple of "
                "3 elements: [X_static, X_dynamic, X_future]."
            )
        s_input_raw, d_input_raw, f_input_raw = inputs
        if s_input_raw is None or d_input_raw is None or f_input_raw is None:
             raise ValueError(
                f"Model '{model_name}' requires all three inputs "
                "(static, dynamic, future) to be non-None."
            )
    # At this point, s_input_raw, d_input_raw, f_input_raw are set
    # (some might be None only if model_name_lower == 'tft_flex')
    # Get initial feature dimensions for the validator
    s_dim_val = s_input_raw.shape[-1] if s_input_raw is not None else None
    d_dim_val = d_input_raw.shape[-1] if d_input_raw is not None else None
    f_dim_val = f_input_raw.shape[-1] if f_input_raw is not None else None

    if d_input_raw is None or d_dim_val is None: # Dynamic is always essential
        raise ValueError("Dynamic input is missing or has no features.")

    validation_mode = 'soft' if model_name_lower == 'tft_flex' else 'strict'

    # `validate_model_inputs` expects [S,D,F] and returns (S,D,F)
    X_static, X_dynamic, X_future = validate_model_inputs(
        inputs=[s_input_raw, d_input_raw, f_input_raw],
        static_input_dim=s_dim_val,
        dynamic_input_dim=d_dim_val,
        future_covariate_dim=f_dim_val,
        forecast_horizon=forecast_horizon,
        error="raise",
        mode=validation_mode,
        model_name=model_name_lower,
        verbose=verbose >= 4
    )
    # X_static, X_dynamic, X_future are now validated.
    vlog(
          "Parameters check sucessfully passed. ",
          level=4, verbose=verbose
      )
        
    # --- Inside xtft_tuner function ---
    # Prepare inputs for Keras Tuner's fit method.
    # Replace None with dummy tensors with 0 features.
    inputs_for_fit_list = []
    
    # Determine a reference batch size and time steps from dynamic_input
    # (which is guaranteed to be non-None for all relevant models).
    ref_batch_size = tf_shape(X_dynamic)[0]
    ref_dyn_time_steps = tf_shape(X_dynamic)[1]

    if X_static is not None:
        inputs_for_fit_list.append(X_static)
    else:
        # Create dummy 2D static tensor: (Batch, 0 Features)
        dummy_static = tf_zeros((ref_batch_size, 0), dtype=tf_float32)
        inputs_for_fit_list.append(dummy_static)
        if verbose >= 2:
            vlog(f"  Using dummy static input for fit: {dummy_static.shape}",
                 level=3, verbose=verbose)

    inputs_for_fit_list.append(X_dynamic) # Dynamic is always present

    if X_future is not None:
        inputs_for_fit_list.append(X_future)
    else:
        # Create dummy 3D future tensor: (Batch, Time, 0 Features)
        # The time dimension for future dummy should match what the model
        # might expect if future_input was present.
        # For TFTFlexible, it might align with dynamic or dynamic + horizon.
        # Using dynamic's time steps + forecast_horizon is a safe default length.
        future_time_span_for_dummy = ref_dyn_time_steps
        if run_case_info.get("forecast_horizon") is not None:
            future_time_span_for_dummy += run_case_info["forecast_horizon"]
        
        dummy_future = tf_zeros(
            (ref_batch_size, future_time_span_for_dummy, 0),
            dtype=tf_float32
        )
        inputs_for_fit_list.append(dummy_future)
        if verbose >= 2:
            vlog(f"  Using dummy future input for fit: {dummy_future.shape}",
                 level=3, verbose=verbose)

    inputs_for_fit = inputs_for_fit_list
    # inputs_for_fit is now always a list of 3 actual tensors.

    # Update case_info with actual input dimensions for the builder
    # These will be 0 if dummy tensors were used.
    run_case_info['static_input_dim'] = inputs_for_fit[0].shape[-1]
    run_case_info['dynamic_input_dim'] = inputs_for_fit[1].shape[-1]
    run_case_info['future_input_dim'] = inputs_for_fit[2].shape[-1]

    vlog(f"  Final input dims for model builder: "
         f"S={run_case_info['static_input_dim']}, "
         f"D={run_case_info['dynamic_input_dim']}, "
         f"F={run_case_info['future_input_dim']}",
         level=3, verbose=verbose)


    vlog("Parameters and inputs checked/validated successfully.",
         level=2, verbose=verbose)

    # --- Define model_builder ---
    actual_model_builder = model_builder
    if actual_model_builder is None:
        vlog("Using default _model_builder_factory.",
             level=2, verbose=verbose)
        actual_model_builder = lambda hp: _model_builder_factory(
            hp, model_name_lower, # Pass lowercase model_name
            X_static, X_dynamic, X_future, # Pass validated S, D, F
            run_case_info,
            get_param_space_value
        )

    # --- Callbacks ---
    actual_callbacks = callbacks
    if actual_callbacks is None:
        vlog("Setting default EarlyStopping callback.",
             level=2, verbose=verbose)
        actual_callbacks = [
            EarlyStopping(
                monitor=get_param_space_value('monitor', 'val_loss'),
                patience=get_param_space_value('patience', 10),
                restore_best_weights=True
            )
        ]

    # --- Tuner Setup ---
    tuner_dir = tuner_dir or os.path.join(
        os.getcwd(), "fusionlab_tuning_results"
        )
    project_name = project_name or (
        f"{model_name.upper()}_Tune_"
        f"{run_case_info.get('description', '').replace(' ', '_')}"
        )

    vlog(f"Initializing Keras Tuner: {tuner_type.upper()} "
         f"for {project_name}", level=1, verbose=verbose)
    common_tuner_args = {
        "hypermodel": actual_model_builder,
        "objective": objective,
        "max_trials": max_trials,
        "directory": tuner_dir,
        "project_name": project_name,
        "overwrite": True # Start fresh each time for tests
    }
    if tuner_type == "bayesian":
        tuner = kt.BayesianOptimization(**common_tuner_args)
    elif tuner_type == "random":
        tuner = kt.RandomSearch(**common_tuner_args)
    else:
        raise ValueError(f"Unsupported tuner_type: {tuner_type}")
    vlog("Tuner initialized.", level=2, verbose=verbose)

    # --- Tuning Loop ---
    overall_best_model: Optional[Model] = None
    overall_best_hps_dict: Optional[Dict] = None
    overall_best_val_loss = np.inf
    overall_best_batch_size: Optional[int] = None
    tuning_log_data: List[Dict] = []

    for current_batch_size in batch_sizes:
        vlog(f"--- Tuning with Batch Size: {current_batch_size} ---",
             level=1, verbose=verbose)
        try:
            tuner.search(
                x=inputs_for_fit, # Use [S, D, F] order
                y=y_tensor,
                epochs=epochs, # Epochs for each trial in search
                batch_size=current_batch_size,
                validation_split=validation_split,
                callbacks=actual_callbacks,
                verbose=verbose >= 2 # Tuner's own verbosity
            )
            current_best_trial_hps = tuner.get_best_hyperparameters(
                num_trials=1)[0]
            
            vlog(f"  Best HPs for batch {current_batch_size}: "
                 f"{current_best_trial_hps.values}",
                 level=2, verbose=verbose)
            
            current_best_model = tuner.hypermodel.build(
                current_best_trial_hps
                )
            vlog(f"  Training best model for batch {current_batch_size} "
                 f"for {epochs} epochs...", level=2, verbose=verbose)
            history = current_best_model.fit(
                x=inputs_for_fit, y=y_tensor, epochs=epochs,
                batch_size=current_batch_size,
                validation_split=validation_split,
                callbacks=actual_callbacks, verbose=verbose >= 2
            )
            current_model_val_loss = min(history.history['val_loss'])
            vlog(
                f"  Batch Size {current_batch_size}: Final val_loss = "
                f"{current_model_val_loss:.4f}",
                level=1, verbose=verbose
            )
            trial_info = {
                "batch_size": current_batch_size,
                "best_val_loss_for_batch": current_model_val_loss,
                "hyperparameters": current_best_trial_hps.values
            }
            tuning_log_data.append(trial_info)

            if current_model_val_loss < overall_best_val_loss:
                overall_best_val_loss = current_model_val_loss
                overall_best_hps_dict = current_best_trial_hps.values.copy()
                overall_best_model = current_best_model
                overall_best_batch_size = current_batch_size
                if overall_best_hps_dict is not None:
                    overall_best_hps_dict['batch_size'] = (
                        overall_best_batch_size
                        )
        except Exception as e:
            vlog(f"Tuning failed for batch size {current_batch_size}."
                 f" Error: {e}", level=0, verbose=verbose)
            warnings.warn(
                f"Tuning for batch {current_batch_size} failed: {e}"
                )
            continue # Try next batch size

    if overall_best_model is None:
        vlog("Hyperparameter tuning failed for all batch sizes.",
             level=0, verbose=verbose)
        return None, None, tuner

    # Log final results
    if overall_best_hps_dict is not None:
        tuning_log_data.append({
            'overall_best_batch_size': overall_best_batch_size,
            'overall_best_val_loss': overall_best_val_loss,
            'overall_best_hyperparameters': overall_best_hps_dict
        })
    log_file_path = os.path.join(
        tuner_dir, f"{project_name}_tuning_summary.json"
        )
    try:
        with open(log_file_path, "w") as f:
            # Use default=str for non-serializable Keras HP values
            json.dump(tuning_log_data, f, indent=4, default=str)
        vlog(f"Full tuning summary saved to {log_file_path}",
             level=1, verbose=verbose)
    except Exception as e:
        warnings.warn(f"Could not save tuning summary log: {e}")

    vlog("--- Overall Best ---", level=1, verbose=verbose)
    vlog(f"Best Batch Size: {overall_best_batch_size}",
         level=1, verbose=verbose)
    
    
    
    summary = ResultSummary(
        'BestHyperParameters').add_results(overall_best_hps_dict)
    vlog(f"Best Hyperparameters:\n {summary}",
         level=1, verbose=verbose
         )
    
    vlog(f"Best Validation Loss: {overall_best_val_loss:.4f}",
         level=1, verbose=verbose)

    return overall_best_hps_dict, overall_best_model, tuner

def tft_tuner( 
    inputs: List[Optional[Union[np.ndarray, Tensor]]],
    y: Union[np.ndarray, Tensor],
    param_space: Optional[Dict[str, Any]] = None,
    forecast_horizon: int = 1,
    quantiles: Optional[List[float]] = None,
    case_info: Optional[Dict[str, Any]] = None,
    max_trials: int = 10,
    objective: str = 'val_loss',
    epochs: int = 10, # Reduced default for faster example
    batch_sizes: List[int] = [32],
    validation_split: float = 0.2,
    tuner_dir: Optional[str] = None,
    project_name: Optional[str] = None,
    tuner_type: str = 'random',
    callbacks: Optional[List[Callable]] = None,
    model_builder: Optional[Callable] = None,
    model_name: str = "tft", # Default to stricter 'tft'
    verbose: int = 1
) -> tuple:
    """
    Fine-tunes TemporalFusionTransformer (flexible via model_name='tft_flex')
    or TFT (stricter via model_name='tft') models.
    """
    # Ensure model_name is one of the TFT variants
    if model_name.lower() not in ["tft", "tft_flex"]:
        warnings.warn(
            f"model_name '{model_name}' for tft_tuner is unusual. "
            "Expected 'tft' or 'tft_flex'. Proceeding."
            )
    return xtft_tuner(
        inputs=inputs, y=y, param_space=param_space,
        forecast_horizon=forecast_horizon, quantiles=quantiles,
        case_info=case_info, max_trials=max_trials,
        objective=objective, epochs=epochs, batch_sizes=batch_sizes,
        validation_split=validation_split, tuner_dir=tuner_dir,
        project_name=project_name, tuner_type=tuner_type,
        callbacks=callbacks, model_builder=model_builder,
        model_name=model_name, # Pass along specified tft or tft_flex
        verbose=verbose
    )

def _model_builder_factory(
    hp: "kt.HyperParameters",
    model_name_lower: str, # Expects lowercase
    # These are the validated inputs in S, D, F order
    X_static_val: Optional[Tensor],
    X_dynamic_val: Tensor,
    X_future_val: Optional[Tensor],
    case_info_param: Dict[str, Any],
    get_param_space_func: Callable
    ):
    """
    Builds and compiles a model instance for Keras Tuner.
    (Full docstring omitted for brevity)
    """
    # --- Base parameters common to most models ---
    params = {
        "forecast_horizon": case_info_param.get("forecast_horizon"),
        "quantiles": case_info_param.get("quantiles"),
        "output_dim": case_info_param.get("output_dim", 1),
        "hidden_units": hp.Choice(
            'hidden_units',
            get_param_space_func('hidden_units', [32, 64]) # Default
            ),
        "num_heads": hp.Choice(
            'num_heads',
            get_param_space_func('num_heads', [2, 4])
            ),
        "dropout_rate": hp.Choice(
            'dropout_rate',
            get_param_space_func('dropout_rate', [0.0, 0.1])
            ),
        "activation": hp.Choice( # Pass string, model __init__ handles it
            'activation',
            get_param_space_func('activation', ["relu", "gelu"])
            ),
        "use_batch_norm": hp.Choice(
            'use_batch_norm',
             get_param_space_func('use_batch_norm', [True, False]), 
             default=False, 
        ),
    }

    # --- Add input dimensions from validated data ---
    # Dynamic is always required for all these models
    params["dynamic_input_dim"] = X_dynamic_val.shape[-1]

    # Static and Future dimensions depend on whether they were provided
    if X_static_val is not None:
        params["static_input_dim"] = X_static_val.shape[-1]
    # else: static_input_dim remains unset or will be None if model allows

    if X_future_val is not None:
        params["future_input_dim"] = X_future_val.shape[-1]
    # else: future_input_dim remains unset or will be None
    
    # --- Model-specific parameters and class selection ---
    if model_name_lower in ["xtft", "superxtft", "super_xtft"]:
        # These models require all three input dimensions
        if params.get("static_input_dim") is None or \
           params.get("future_input_dim") is None:
            raise ValueError(
                f"{model_name_lower.upper()} requires static and future inputs."
                " Corresponding X_static or X_future was None."
            )
        params.update({
            "embed_dim": hp.Choice(
                'embed_dim', get_param_space_func('embed_dim', [16, 32])),
            "max_window_size": hp.Choice(
                'max_window_size', get_param_space_func('max_window_size', [5, 10])),
            "memory_size": hp.Choice(
                'memory_size', get_param_space_func('memory_size', [50, 100])),
            "lstm_units": hp.Choice(
                'lstm_units', get_param_space_func('lstm_units', [32, 64])),
            "attention_units": hp.Choice(
                'attention_units', get_param_space_func('attention_units', [32, 64])),
            "recurrent_dropout_rate": hp.Choice(
                'recurrent_dropout_rate',
                get_param_space_func('recurrent_dropout_rate', [0.0, 0.1])),
            "use_residuals": hp.Choice(
                'use_residuals',
                get_param_space_func('use_residuals', [False, True]), 
                default=True 
                ),
            "final_agg": hp.Choice(
                'final_agg',
                get_param_space_func('final_agg', ['last', 'average'])),
            "multi_scale_agg": hp.Choice(
                'multi_scale_agg',
                get_param_space_func('multi_scale_agg', ['last', 'average'])),
            # Handle scales string to actual list/None mapping
               "scales": _map_scales_choice(hp.Choice(
                   'scales_options', # Use the string choice name
                   get_param_space_func('scales_options', ['default_scales', 'no_scales'])
                   ))
        })
        model_class = SuperXTFT if model_name_lower in [
            "super_xtft", "superxtft"] else XTFT

    elif model_name_lower == "tft": # Stricter TFT
        # Stricter TFT also requires all three input dimensions
        if params.get("static_input_dim") is None or \
           params.get("future_input_dim") is None:
            raise ValueError(
                "Stricter TFT model requires static_input_dim and "
                "future_input_dim. Corresponding X_static or X_future was None."
            )
        params.update({
            "num_lstm_layers": hp.Choice(
                'num_lstm_layers',
                get_param_space_func('num_lstm_layers', [1, 2])),
            "lstm_units": hp.Choice( # Can be int or list
                'lstm_units',
                get_param_space_func('lstm_units', [32, 64])),
            "recurrent_dropout_rate": hp.Choice(
                'recurrent_dropout_rate',
                get_param_space_func('recurrent_dropout_rate', [0.0, 0.1]))
        })
        model_class = TFTStricter # Use aliased stricter TFT

    elif model_name_lower == "tft_flex": # Flexible TemporalFusionTransformer
        # This model only *requires* dynamic_input_dim.
        # static_input_dim and future_input_dim are optional.
        # We need to ensure only valid params are passed.
        tft_flex_params = {
            "dynamic_input_dim": params["dynamic_input_dim"]
            }
        # Add optional dims if they were provided (i.e., X_static/X_future were not None)
        if "static_input_dim" in params:
            tft_flex_params["static_input_dim"] = params["static_input_dim"]
        if "future_input_dim" in params:
            tft_flex_params["future_input_dim"] = params["future_input_dim"]

        # Add other common params it accepts (from base `params` dict)
        for common_param in ["forecast_horizon", "quantiles", "output_dim",
                             "hidden_units", "num_heads", "dropout_rate",
                             "activation", "use_batch_norm"]:
            if common_param in params: # Check if defined by HP or case_info
                tft_flex_params[common_param] = params[common_param]

        # LSTM specific params for TemporalFusionTransformer
        tft_flex_params["num_lstm_layers"] = hp.Choice(
            'num_lstm_layers',
            get_param_space_func('num_lstm_layers', [1, 2]))
        tft_flex_params["lstm_units"] = hp.Choice( # Can be int or list
            'lstm_units',
            get_param_space_func('lstm_units', [32, 64]))
        # Note: TemporalFusionTransformer (flexible) does not take
        # recurrent_dropout_rate in its __init__ as per its signature.

        params = tft_flex_params # Override params dict for this model
        model_class = TFTFlexible # Use aliased flexible TFT
    else:
        # This case should ideally be caught by model_name validation earlier
        raise ValueError(
            f"Unsupported model_name for tuning factory: {model_name_lower}")

    # --- Explicitly cast boolean HPs before model instantiation ---
    # These are parameters that the models expect as Python bool
    # then apply the transformation inplace via _cast_hp_to_bool
    # to change param dict inplace. 
    # Usage:
    bool_params_to_cast = [('use_batch_norm', False), ('use_residuals', True)]
    cast_multiple_bool_params(params, bool_params_to_cast)
                 
    # for safetly # get the valid params of model class 
    params = _get_valid_kwargs (model_class, params )
  
    # Instantiate model
    # For tft_flex, some *_input_dim might be None, which is fine for its __init__

    model = model_class(**params)

    # Compile model
    learning_rate = hp.Choice(
        'learning_rate',
        get_param_space_func('learning_rate', [1e-3, 5e-4])
        )
    # Loss should be a callable function or a string Keras recognizes
    loss_to_use = get_param_space_func('loss', 'mse')

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss_to_use
    )
    vlog(f"{model_name_lower.upper()} model built and compiled for trial.",
         level=3, verbose=case_info_param.get("verbose_build", 0))

    return model

def _map_scales_choice(scales_choice_str: str) -> Optional[List[int]]:
    """Maps string choice for scales to actual list or None."""
    if scales_choice_str == 'default_scales':
        return [1, 3, 7]
    elif scales_choice_str == 'alt_scales':
        return [1, 5, 10]
    elif scales_choice_str == 'no_scales':
        return None
    return None # Default fallback

def _cast_hp_to_bool(
    params: Dict[str, Any],
    param_name: str,
    default_value: bool = False # Default if param not in hp choices
) -> None:
    """
    Casts a hyperparameter value in the params dict to boolean.
    Keras Tuner might return 0 or 1 for boolean choices.
    This helper ensures it's a Python bool before model instantiation.
    Modifies `params` in-place.
    """
    if param_name in params:
        value = params[param_name]
        if isinstance(value, (int, float)): # Catches 0, 1, 0.0, 1.0
            params[param_name] = bool(value)
        elif not isinstance(value, bool):
            # If it's something else, it might be an issue with
            # param_space definition or how Keras Tuner sampled it.
            # For safety, default or warn.
            warnings.warn(
                f"Hyperparameter '{param_name}' received unexpected value "
                f"'{value}' (type: {type(value)}). Expected bool or 0/1. "
                f"Defaulting to False. Please check param_space definition."
            )
            params[param_name] = default_value
    # If param_name not in params from hp, it might be a fixed value
    # from case_info, which should already be bool. Or it might not be
    # applicable to the current model. No action needed here.

# Optimized casting for multiple boolean parameters
def cast_multiple_bool_params(
        params: Dict[str, Any], 
        bool_params_to_cast: List[Tuple[str, bool]]
        ) -> None:
    """
    Casts a list of boolean hyperparameters to ensure they are Python booleans.
    
    Args:
        params: Dictionary of hyperparameters.
        bool_params_to_cast: List of tuples (param_name, default_value) for boolean params.
    """
    for param_name, default_value in bool_params_to_cast:
        _cast_hp_to_bool(params, param_name, default_value)

xtft_tuner.__doc__+=r"""\
The function sets up a hyperparameter tuning workflow for the XTFT model, 
leveraging Keras Tuner's Bayesian Optimization to search over a defined 
hyperparameter space. The function accepts input tensors for static, dynamic,
and future features along with the target output, and returns the best 
hyperparameter configuration, the corresponding trained model, and the 
tuner instance.

The hyperparameter search is formulated as:

.. math::
   \min_{\theta \in \Theta} \; L\bigl(\theta; \mathbf{X}, y\bigr)

where :math:`\Theta` is the hyperparameter space and 
:math:`L(\theta; \mathbf{X}, y)` is the validation loss computed over the 
training data.

Parameters
----------
inputs : List[Union[np.ndarray, Tensor]]
    A list containing three input tensors:
      - ``X_static``: static features with shape (``B, N_s``)
      - ``X_dynamic``: dynamic features with shape (``B, F, N_d``)
      - ``X_future``: future features with shape (``B, F, N_f``)
    Here, :math:`B` is the batch size, :math:`N_s` is the number of static 
    features, :math:`F` is the forecast horizon, :math:`N_d` is the number of 
    dynamic features, and :math:`N_f` is the number of future features.
y                : np.ndarray
    The target output tensor with shape (``B, F, O``), where 
    :math:`O` is the output dimension.
param_space      : Dict[str, Any], optional
    A dictionary specifying custom hyperparameter ranges. If not provided, a 
    default parameter space is used.
forecast_horizon : int, default=1
    The number of future steps to forecast. This should be consistent with 
    the forecast horizon in the dynamic and future inputs.
quantiles        : List[float], optional
    A list of quantile values for quantile forecasting (e.g., 
    ``[0.1, 0.5, 0.9]``). If not provided, default quantiles are used based on 
    the case configuration.
case_info        : Dict[str, Any], optional
    A dictionary containing case-specific configuration parameters (such as 
    forecast horizon and quantiles) to configure the XTFT model.
max_trials       : int, default=10
    Maximum number of hyperparameter tuning trials to perform.
objective        : str, default='val_loss'
    The performance metric to optimize (e.g., ``"val_loss"``).
epochs           : int, default=50
    The number of training epochs for each tuning trial.
batch_sizes      : List[int], default=[16, 32, 64]
    A list of batch sizes to explore during tuning.
validation_split : float, default=0.2
    Fraction of training data used as the validation set.
tuner_dir        : Optional[str], default=None
    Directory in which tuner results and logs will be saved. A default 
    directory is used if not provided.
project_name     : Optional[str], default=None
    Name for the tuning project. If not provided, a name is generated based 
    on the case description.
tuner_type       : str, default='bayesian'
    The type of tuner to use. Currently, only Bayesian Optimization is 
    supported.
callbacks        : Optional[list], default=None
    A list of Keras callbacks to use during tuning. If not provided, a default 
    EarlyStopping callback is applied.
model_builder    : Optional[Callable], default=None
    A callable that builds and compiles the XTFT model. If omitted, a default 
    model builder is used which defines a hyperparameter search space over 
    key model parameters.
verbose          : int, default=1
    Verbosity level controlling logging output. Values range from 1 (minimal) 
    to 7 (very detailed).

Returns
-------
tuple
    A tuple containing:
      - dict: The best hyperparameters found.
      - tf.keras.Model: The best trained XTFT model.
      - kt.Tuner: The tuner instance used for hyperparameter search.

Examples
--------
>>> from fusionlab.nn.forecast_tuner import xtft_tuner
>>> # Assume preprocessed inputs: X_static, X_dynamic, X_future, and y
>>> best_hps, best_model, tuner = xtft_tuner(
...     inputs=[X_static, X_dynamic, X_future],
...     y=y,
...     forecast_horizon=4,
...     quantiles=[0.1, 0.5, 0.9],
...     case_info={"description": "Quantile Forecast",
...                "forecast_horizon": 4,
...                "quantiles": [0.1, 0.5, 0.9]},
...     max_trials=5,
...     epochs=50,
...     batch_sizes=[16, 32],
...     validation_split=0.2,
...     tuner_dir="tuning_results",
...     project_name="XTFT_Tuning_Case",
...     verbose=5
... )
>>> print("Best hyperparameters:", best_hps)
>>> best_model.summary()

Notes
-----
The function first validates and converts input tensors to 
``float32`` for numerical stability via 
:func:`validate_minimal_inputs`. It then defines a hyperparameter search 
space (defaulting to a predefined space if ``param_space`` is not provided) 
and iterates over the specified batch sizes. For each batch size, the tuner 
trains the model for a set number of epochs and selects the best model based 
on the validation loss. The final best hyperparameters, trained model, and 
tuner instance are returned.

See Also
--------
:func:`validate_minimal_inputs` : Validates input tensor dimensions.
:class:`kt.BayesianOptimization` : Keras Tuner class for Bayesian optimization.
:class:`tensorflow.keras.optimizers.Adam` : Optimizer used for model training.
XTFT : The transformer model used for forecasting. 

References
----------
.. [1] McKinney, W. (2010). "Data Structures for Statistical Computing 
       in Python". Proceedings of the 9th Python in Science Conference.
.. [2] Van der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). "The 
       NumPy Array: A Structure for Efficient Numerical Computation". 
       Computing in Science & Engineering, 13(2), 22-30.
"""

tft_tuner.__doc__+=r"""\
This function is a wrapper around :func:`xtft_tuner` that explicitly 
sets the model type to ``"tft"`` and configures hyperparameter tuning 
for Temporal Fusion Transformer (TFT) models. It leverages Bayesian 
Optimization (or another tuner type if specified) to search over a 
defined hyperparameter space. The tuning process is formulated as:

.. math::
   \min_{\theta \in \Theta} \; L\bigl(\theta; \mathbf{X}, y\bigr)

where :math:`\Theta` is the hyperparameter space and 
:math:`L(\theta; \mathbf{X}, y)` is the loss (e.g., validation loss) 
computed over the training data.

Parameters
----------
inputs           : List[Union[np.ndarray, Tensor]]
    A list containing three input arrays:
      - ``X_static`` with shape ``(B, N_s)``,
      - ``X_dynamic`` with shape ``(B, F, N_d)``, and
      - ``X_future`` with shape ``(B, F, N_f)``.
    Here, :math:`B` denotes the batch size, :math:`N_s` is the number 
    of static features, :math:`F` is the forecast horizon, :math:`N_d` is 
    the number of dynamic features, and :math:`N_f` is the number of future 
    features.
y                : Optional[np.ndarray], default=None
    The target output with shape ``(B, F, O)`` (if provided), where 
    :math:`O` is the output dimension.
param_space      : Optional[Dict[str, Any]], default=None
    A dictionary defining the hyperparameter search space. If omitted, 
    a default parameter space is used.
forecast_horizon : int, default=1
    The expected number of future steps to forecast.
quantiles        : Optional[List[float]], default=None
    A list of quantile values for quantile regression (e.g., 
    ``[0.1, 0.5, 0.9]``). Ignored in point forecasting mode.
case_info        : Optional[Dict[str, Any]], default=None
    A dictionary containing additional configuration details (e.g., 
    forecast horizon, quantiles) used to configure the model.
max_trials       : int, default=10
    Maximum number of hyperparameter tuning trials to perform.
objective        : str, default='val_loss'
    The performance metric (objective) to minimize during tuning.
epochs           : int, default=50
    Number of training epochs per tuning trial.
batch_sizes      : List[int], default=[16, 32, 64]
    A list of batch sizes to explore during tuning.
validation_split : float, default=0.2
    Fraction of the training data to use for validation.
tuner_dir        : Optional[str], default=None
    Directory where tuner results and logs are stored. If not provided, 
    a default directory is used.
project_name     : Optional[str], default=None
    The name of the tuning project. If omitted, a project name is generated 
    based on the case information.
tuner_type       : str, default='bayesian'
    The tuner type to use (e.g., ``"bayesian"`` or ``"random"``).
callbacks        : Optional[List], default=None
    A list of Keras callbacks to use during tuning. If not provided, a 
    default EarlyStopping callback is applied.
model_builder    : Optional[Callable], default=None
    A function that builds and compiles the TFT model. If omitted, a default 
    builder is used.
verbose          : int, default=1
    Verbosity level for logging output. Higher values (from 1 up to 7) 
    yield more detailed debug messages.

Returns
-------
tuple
    A tuple containing:
      - dict: The best hyperparameters found.
      - tf.keras.Model: The best trained TFT model.
      - kt.Tuner: The tuner instance used for hyperparameter search.

Examples
--------
>>> from fusionlab.nn.forecast_tuner import tft_tuner
>>> best_hps, best_model, tuner = tft_tuner(
...     inputs=[X_static, X_dynamic, X_future],
...     y=y,
...     forecast_horizon=4,
...     quantiles=[0.1, 0.5, 0.9],
...     case_info={"description": "TFT Point Forecast",
...                "forecast_horizon": 4,
...                "quantiles": None},
...     max_trials=5,
...     epochs=50,
...     batch_sizes=[16, 32],
...     validation_split=0.2,
...     tuner_dir="tuning_results",
...     project_name="TFT_Tuning",
...     tuner_type="bayesian",
...     verbose=5
... )
>>> print("Best Hyperparameters:", best_hps)
>>> best_model.summary()

Notes
-----
This function is a thin wrapper around :func:`xtft_tuner` that sets the 
``model_name`` parameter to ``"tft"``. It validates input tensors using 
:func:`validate_minimal_inputs` and constructs a default model builder if 
none is provided. Hyperparameter tuning is performed over multiple batch sizes 
to identify the configuration that minimizes the validation loss.

See Also
--------
xtft_tuner          : Function for tuning XTFT models.
validate_minimal_inputs : Validates the dimensions of input tensors.
kt.BayesianOptimization : Keras Tuner's Bayesian Optimization class.
tensorflow.keras.optimizers.Adam : Optimizer used for model training.

References
----------
.. [1] McKinney, W. (2010). "Data Structures for Statistical Computing 
       in Python". Proceedings of the 9th Python in Science Conference.
.. [2] Van der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). "The 
       NumPy Array: A Structure for Efficient Numerical Computation". 
       Computing in Science & Engineering, 13(2), 22-30.
"""




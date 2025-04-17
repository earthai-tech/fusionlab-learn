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
from typing import Union, Dict, Any, Optional, Callable, List
from typing import TYPE_CHECKING # noqa

import numpy as np

from ..compat.sklearn import validate_params, Interval 
from ..core.checks import assert_ratio, check_params, check_non_emptiness
from ..core.handlers import param_deprecated_message
from ..utils.deps_utils import ensure_pkg 
from ..utils.generic_utils import vlog
from ..utils.validator import validate_positive_integer, parameter_validator

from .__init__ import config 
from ._tensor_validation import validate_minimal_inputs 
from . import KERAS_DEPS, KERAS_BACKEND, dependency_message
from .keras_validator import validate_keras_model 
from .losses import combined_quantile_loss 
from .transformers import XTFT, SuperXTFT, TemporalFusionTransformer 

HAS_KT=False
try: 
    import keras_tuner as kt
    HAS_KT=True
except : 
    kt=None 
    pass 

if KERAS_BACKEND:
    Adam = KERAS_DEPS.Adam
    Model=KERAS_DEPS.Model
    Tensor=KERAS_DEPS.Tensor 
    EarlyStopping=KERAS_DEPS.EarlyStopping
    
DEP_MSG = dependency_message('nn.forecast_tuner') 

CASE_INFO = { 
    "description": "{} forecast", 
    "forecast_horizons": 1,
    "quantiles": None,
   }

DEFAULT_PS = {
    'embed_dim'         : [16, 32, 64],
    'max_window_size'   : [3, 5, 10],
    'memory_size'       : [50, 100, 200],
    'num_heads'         : [2, 4, 8],
    'dropout_rate'      : [0.1, 0.2, 0.3],
    'lstm_units'        : [32, 64, 128],
    'attention_units'   : [32, 64, 128],
    'hidden_units'      : [32, 64, 128],
    'learning_rate'     : [0.0001, 0.001, 0.01],
    'monitor'           : 'val_loss',
    'patience'          : 5,
    }

__all__ = ['xtft_tuner', 'tft_tuner']


@ensure_pkg (
    'keras_tuner',
    extra="'keras_tuner' is required for forecasting model to tune.", 
    auto_install= config.INSTALL_DEPS,
    use_conda = config.USE_CONDA 
)
@param_deprecated_message(
    conditions_params_mappings=[
        {
            'param': 'tuner_type',
            'condition': lambda v: v not in {'bayesian', 'random'}, 
            'message': (
                "Forecast Tuner currently support bayesian and randomized"
                " optimizations. Falling back to randomized optimization"
                " instead."
            ),
            'default': "random"
        }
    ],
    warning_category=UserWarning
)
@check_params({ 
    'tuner_dir':Optional[str], 
    'project_name': Optional[str]
    })
@validate_params({ 
    'inputs': ['array-like'], 
    'y': ['array-like', None], 
    'param_space': [dict, None], 
    'forecast_horizon': [Interval(Integral, 1, None, closed="left")], 
    'quantiles': ['array-like', None], 
    'max_trials': [Interval(Integral, 1, None, closed ='left')], 
    'epochs': [Interval(Integral, 1, None, closed ='left')], 
    'batch_sizes': ['array-like'], 
    'validation_split': [Interval (Real, 0, 1, closed='neither'), str],
    })
@check_non_emptiness 
def xtft_tuner(
    inputs: List[Union [ np.ndarray, Tensor]],
    y: np.ndarray,
    param_space: Dict[str, Any] = None,
    forecast_horizon: int = 1,
    quantiles: List[float] = None,
    case_info: Dict[str, Any] = None,
    max_trials: int = 10,
    objective: str = 'val_loss',
    epochs: int = 50,
    batch_sizes: List[int] = [16, 32, 64],
    validation_split: float = 0.2,
    tuner_dir: Optional[str] = None,
    project_name: Optional[str] = None,
    tuner_type: str = 'random',
    callbacks: Optional[list] = None,
    model_builder: Optional[Callable] = None,
    model_name="xtft",
    verbose: int = 3
) -> tuple:
    
    loss_fn="mse"
    if quantiles is not None: 
        loss_fn = combined_quantile_loss (quantiles)
        
    # Update case info.
    CASE_INFO.update ({ 
        'description':CASE_INFO['description'].format(
            "Quantile" if quantiles is not None else "Point"), 
        'forecast_horizon': forecast_horizon, 
        'quantiles': quantiles, 
        }
    )
    case_info = case_info or CASE_INFO 
    
    DEFAULT_PS.update({
    'loss' : loss_fn if case_info.get(
        "quantiles", quantiles) else 'mse', 
    })
    
    # Define default parameter space.
    vlog(
        f"Start {model_name.upper()} {tuner_type.upper()} tune ... ",
        level=3
    )
    # Local helper to retrieve parameter space values.
    def get_param_space(name):
        ps = param_space or {}
        return ps.pop(name, DEFAULT_PS.get(name))
    
    vlog(
        "Check parameters... ",
        level=4
    )
    forecast_horizon= validate_positive_integer(
        forecast_horizon, "forecast_horizon", 
   )
    validation_split = assert_ratio(
        validation_split, bounds =(0, 1,), 
        exclude_values= [0], 
        name ="validation split"
    )
    model_name = parameter_validator(
        "model_name", target_strs= {"xtft", 'superxtft', 'super_xtft'}, 
        )(model_name)
    
    # Validate input type.
    if not isinstance(inputs, (list, tuple)):
        raise ValueError("Inputs must be provided as a list or tuple.")
    inputs = list(inputs)  # Ensure consistency.

    # Ensure there are exactly 3 elements: X_static, X_dynamic, X_future.
    if len(inputs) != 3:
        raise ValueError("Expected inputs to contain exactly 3 elements: "
                         "[X_static, X_dynamic, X_future].")

    # Validate input shapes using the minimal inputs validator.
    # This returns X_static, X_dynamic, X_future, and optionally y.
    validated = validate_minimal_inputs(
        *inputs, y=y, forecast_horizon=forecast_horizon, deep_check=True
    )
    if y is None:
        X_static, X_dynamic, X_future = validated
    else:
        X_static, X_dynamic, X_future, y = validated
    vlog(
        "Parameters check sucessfully passed. ",
        level=4
    )
    
    # Define default model builder if none is provided.
    if model_builder is None:
        vlog(
            "Set default model builder... ",
            level=3
        )
        # Define model builder if none is provided
        model_builder = lambda hp: _model_builder_factory(
            hp, model_name, *inputs, case_info, get_param_space
        )
  
    vlog(
        "Is model builder a keras model object?",
        level=5
    )
    model_builder = validate_keras_model(
        model_builder, 
        raise_exception= True 
        )
    
    vlog(
        "YES",
        level=5
    )
    
    # Set default callbacks if none provided.
    if callbacks is None:
        vlog(
            "Set default callbacks... ",
            level=3
        )
        callbacks = [
            EarlyStopping(
                monitor = get_param_space('monitor'),
                patience = get_param_space('patience'),
                restore_best_weights = True
            )
        ]
        vlog(
            "Callbacks set successfully. ",
            level=3
        )
    # Set default tuner directory and project name.
    tuner_dir = tuner_dir or os.path.join(os.getcwd(), "tuning_results")
    project_name = project_name or ( 
        f"{model_name.upper()}_Tuning_{case_info.get('description', '')}"
        )

    # Initialize the tuner based on the selected strategy.
    vlog(f"Initialize the Keras Tuner using {tuner_type.upper()}...", 
         level=3)
    
    if tuner_type == "bayesian":
        tuner = kt.BayesianOptimization(
            model_builder,
            objective=objective,
            max_trials=max_trials,
            directory=tuner_dir,
            project_name=project_name
        )
    elif tuner_type == "random":
        tuner = kt.RandomSearch(
            model_builder,
            objective=objective,
            max_trials=max_trials,
            directory=tuner_dir,
            project_name=project_name
        )

    vlog("Tuner initialization completed.", level=3)

    # Variables to store the best model and hyperparameters.
    best_model   = None
    best_hps     = None
    best_val_loss= np.inf
    best_batch   = None
    tuning_results= []
    # Iterate over each batch size and tune.
    vlog("Run hyperparameter tuning...", level=3)
    for bs in batch_sizes:
        vlog(f"Tuning with batch size: {bs}", level=4)
        try:
            tuner.search(
                x = [X_static, X_dynamic, X_future],
                y = y,
                epochs = epochs,
                batch_size = bs,
                validation_split = validation_split,
                callbacks = callbacks,
                verbose = verbose
            )
            # Retrieve best hyperparameters for current batch size.
            vlog(
                f"Retrieve best hyperparameters for current batch: {bs}",
                level=5
            )
            current_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            # Build and train model using these hyperparameters.
            model = tuner.hypermodel.build(current_hps)
            history = model.fit(
                x = [X_static, X_dynamic, X_future],
                y = y,
                epochs = epochs,
                batch_size = bs,
                validation_split = validation_split,
                callbacks = callbacks,
                verbose = verbose
            )
            current_val_loss = min(history.history['val_loss'])
            vlog(
                f"Batch Size {bs}: Best val_loss = {current_val_loss:.4f}", 
                level=3
            )
            
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_hps     = current_hps.values
                best_model   = model
                best_batch   = bs
                
            tuning_results.append({
                "batch_size": bs,
                "best_val_loss": current_val_loss,
                "hyperparameters": current_hps.values
            })
        except Exception as e:
            vlog(
                "Tuning failed for batch size {bs}",
                level=1
            )
            warnings.warn(f"Tuning failed for batch size {bs}: {e}")
            continue

    if best_model is None:
        raise RuntimeError("Hyperparameter tuning failed for all batch sizes.")
    
    # Save tuning results to a text file
    tuning_results.append({
        'best_batch': best_batch, 
        'best_hps': best_hps 
        })
    
    log_file =  os.path.join(
        tuner_dir, f"{model_name.upper()}_tuning_results.txt")
    
    with open(log_file, "w") as f:
        json.dump(tuning_results, f, indent=4)

    vlog(f"Tuning results saved to {log_file}", level=3)
    
    vlog(f"Best Batch Size: {best_batch}", level=3)
    vlog(f"Best Hyperparameters: {best_hps}", level=3)

    return best_hps, best_model, tuner


def tft_tuner(
    inputs: List[Union[np.ndarray, Tensor]],
    y: Optional[np.ndarray] = None,
    param_space: Optional[Dict[str, Any]] = None,
    forecast_horizon: int = 1,
    quantiles: Optional[List[float]] = None,
    case_info: Optional[Dict[str, Any]] = None,
    max_trials: int = 10,
    objective: str = 'val_loss',
    epochs: int = 50,
    batch_sizes: List[int] = [16, 32, 64],
    validation_split: float = 0.2,
    tuner_dir: Optional[str] = None,
    project_name: Optional[str] = None,
    tuner_type: str = 'bayesian',
    callbacks: Optional[List] = None,
    model_builder: Optional[Callable] = None,
    verbose: int = 1
) -> tuple:

    return xtft_tuner(
        inputs=inputs,
        y=y,
        param_space=param_space,
        forecast_horizon=forecast_horizon,
        quantiles=quantiles,
        case_info=case_info,
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
        model_name="tft",  
        verbose=verbose
    )

def _model_builder_factory(
    hp: "kt.HyperParameters", 
    model_name: str, 
    X_static, 
    X_dynamic,
    X_future, 
    case_info, 
    get_param_space
    ):
    """
    Generic model builder that dynamically assigns the correct model class
    based on the model_name and configures hyperparameters.

    Parameters:
    - hp: Keras tuner HyperParameters object.
    - model_name: str, one of ['xtft', 'super_xtft', 'superxtft', 'tft'].
    - X_static, X_dynamic, X_future: input feature tensors.
    - case_info: Dictionary containing forecast settings.
    - get_param_space: Function to retrieve hyperparameter values.

    Returns:
    - A compiled model instance.
    """
    
    params = {
        "dynamic_input_dim": X_dynamic.shape[2],
        "static_input_dim": X_static.shape[1] if X_static is not None else None,
        "future_input_dim": X_future.shape[2] if X_future is not None else None,
        "forecast_horizon": case_info.get("forecast_horizon", 1),
        "quantiles": case_info.get("quantiles", None),
        "hidden_units": hp.Choice(
            'hidden_units', get_param_space('hidden_units')),
        "num_heads": hp.Choice(
            'num_heads', get_param_space('num_heads')),
        "dropout_rate": hp.Choice(
            'dropout_rate', get_param_space('dropout_rate')),
        "activation": hp.Choice(
            'activation', ["elu", "relu", "gelu"]), #"tanh", "sigmoid", "linear",
        "use_batch_norm": hp.Boolean(
            'use_batch_norm'),
    }

    # Adjust parameters based on model type
    if model_name.lower() in ["xtft", "super_xtft", "superxtft"]:
        params.update({
            "embed_dim": hp.Choice(
                'embed_dim', get_param_space('embed_dim')),
            "max_window_size": hp.Choice(
                'max_window_size', get_param_space('max_window_size')),
            "memory_size": hp.Choice(
                'memory_size', get_param_space('memory_size')),
            "lstm_units": hp.Choice(
                'lstm_units', get_param_space('lstm_units')),
            "attention_units": hp.Choice(
                'attention_units', get_param_space('attention_units'))
        })
        model_class = SuperXTFT if model_name.lower() in [
            "super_xtft", "superxtft"] else XTFT

    elif model_name.lower() == "tft":
        params.update({
            "num_lstm_layers": hp.Choice(
                'num_lstm_layers', [1, 2, 3]),
            "lstm_units": hp.Choice(
                'lstm_units', get_param_space('lstm_units')
                )
        })
        model_class = TemporalFusionTransformer
    else:
        raise ValueError(
            f"Unsupported model type: {model_name}")

    # Instantiate model
    model = model_class(**params)

    # Compile model
    model.compile(
        optimizer=Adam(hp.Choice(
            'learning_rate', get_param_space('learning_rate'))),
        loss=get_param_space(
            'loss')
    )

    vlog(f"{model_name.upper()} model builder successfully set.", level=3)

    return model

xtft_tuner.__doc__=r"""\
Fine-tune the XTFT forecasting model using Keras Tuner with Bayesian 
or RandomSearch Optimization.

This function sets up a hyperparameter tuning workflow for the XTFT model, 
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
>>> from gofast.nn.forecast_tuner import xtft_tuner
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

tft_tuner.__doc__=r"""\
Fine-tune Temporal Fusion Transformer (TFT) models using Keras Tuner.

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
>>> from gofast.nn.forecast_tuner import tft_tuner
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


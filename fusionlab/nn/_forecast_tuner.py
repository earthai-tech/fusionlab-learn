# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Base hyperparameter tuner (docstring shortened for brevity).
"""

import os
import warnings
import json
from numbers import Real, Integral
from typing import (
    Union,
    Dict,
    Any,
    Optional,
    Callable,
    List,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np

from ..api.docs import DocstringComponents, _tuner_common_params
from ..api.summary import ResultSummary
from ..compat.sklearn import validate_params, Interval
from ..core.checks import (
    check_params,
)
from ..core.io import _get_valid_kwargs
from ..utils.deps_utils import ensure_pkg
from ..utils.generic_utils import vlog
from ..utils.validator import validate_positive_integer
from ._tensor_validation import validate_model_inputs
from . import (
    KERAS_DEPS,
    KERAS_BACKEND,
)
from .__init__ import config
from .losses import combined_quantile_loss
# from .keras_validator import validate_keras_model
from .transformers import (
    XTFT,
    SuperXTFT,
    TemporalFusionTransformer as TFTFlexible,
    TFT as TFTStricter,
)

HAS_KT = False
try:
    import keras_tuner as kt

    HAS_KT = True
except ImportError:
    class _DummyTuner:
        pass

    class _DummyKT:
        Tuner = _DummyTuner

    kt = _DummyKT()

if TYPE_CHECKING:
    import keras_tuner as kt  # type: ignore
    import tensorflow as tf

    Tensor = tf.Tensor
    Model = tf.keras.Model
else:
    class Tensor:
        pass

    class Model:
        pass


if KERAS_BACKEND:
    Adam = KERAS_DEPS.Adam
    EarlyStopping = KERAS_DEPS.EarlyStopping
    tf_convert_to_tensor = KERAS_DEPS.convert_to_tensor
    tf_float32 = KERAS_DEPS.float32
    tf_zeros = KERAS_DEPS.zeros
    tf_shape = KERAS_DEPS.shape
else:
    class Adam:
        pass

    class EarlyStopping:
        pass

    tf_convert_to_tensor = (
        lambda x, dtype: np.asarray(x, dtype=str(dtype).split(".")[-1])
    )
    tf_float32 = np.float32  # type: ignore
    tf_zeros = np.zeros
    tf_shape = np.shape

_tuner_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_tuner_common_params)
)
    
CASE_INFO = {
    "description": "{} forecast",
    "forecast_horizon": 1,
    "quantiles": None,
    "output_dim": 1,
    "static_input_dim": None,
    "dynamic_input_dim": None,
    "future_input_dim": None,
    "verbose_build": 0,
}

DEFAULT_PS = {
    "embed_dim": [16, 32, 64],
    "max_window_size": [3, 5, 10],
    "memory_size": [50, 100, 200],
    "num_heads": [2, 4, 8],
    "dropout_rate": [0.0, 0.1, 0.2, 0.3],
    "recurrent_dropout_rate": [0.0, 0.1, 0.2],
    "lstm_units": [32, 64, 128],
    "attention_units": [32, 64, 128],
    "hidden_units": [32, 64, 128],
    "num_lstm_layers": [1, 2],
    "activation": ["relu", "gelu"],
    "use_batch_norm": [False, True],
    "use_residuals": [True, False],
    "final_agg": ["last", "average"],
    "multi_scale_agg": ["last", "average"],
    "scales_options": [
        "default_scales",
        "alt_scales",
        "no_scales",
    ],
    "learning_rate": [1e-3, 1e-4, 5e-4],
    "monitor": "val_loss",
    "patience": 10,
}



class BaseTuner:
    @ensure_pkg(
        "keras_tuner",
        extra="'keras_tuner' is required for model tuning.",
        auto_install=config.INSTALL_DEPS,
        use_conda=config.USE_CONDA,
    )
    @check_params(
        {
            "tuner_dir": Optional[str],
            "project_name": Optional[str],
        }
    )
    @validate_params(
        {
            "param_space": [dict, None],
            "max_trials": [Interval(Integral, 1, None, closed="left")],
            "objective": [str],
            "epochs": [Interval(Integral, 1, None, closed="left")],
            "batch_sizes": [list, tuple],
            "validation_split": [
                Interval(Real, 0, 1, closed="neither")
            ],
            "tuner_type": [str],
            "model_name": [str],
            "verbose": [Integral],
        }
    )
    def __init__(
        self,
        model_name: str,
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
        **kws,
    ):
        if not HAS_KT:
            raise ImportError(
                "keras_tuner is not installed. Please run "
                "`pip install keras-tuner` to use this tuning class."
            )

        self.model_name = model_name
        self.param_space = param_space or {}
        self.max_trials = max_trials
        self.objective = objective
        self.epochs = epochs
        self.search_epochs = kws.pop("search_epochs", epochs)
        self.batch_sizes = batch_sizes
        self.validation_split = validation_split
        self.tuner_dir = tuner_dir
        self.project_name = project_name
        self.tuner_type = self._validate_tuner_type(tuner_type)
        self.callbacks = callbacks
        self.custom_model_builder = model_builder
        self.verbose = verbose
        self.kws = kws

        self.best_hps_: Optional[Dict[str, Any]] = None
        self.best_model_: Optional[Model] = None
        self.tuner_: Optional[kt.Tuner] = None
        self.tuning_log_: List[Dict[str, Any]] = []
        self._run_case_info: Dict[str, Any] = {}

    def _validate_tuner_type(self, tuner_type: str) -> str:
        """Validate tuner_type and fallback to 'random' if invalid."""
        valid_types = {"bayesian", "random"}
        tt_lower = tuner_type.lower()
        if tt_lower not in valid_types:
            warnings.warn(
                f"Unsupported tuner type: '{tuner_type}'. "
                f"Supported types: {valid_types}. "
                "Defaulting to 'random'.",
                UserWarning,
            )
            return "random"
        return tt_lower

    def _get_param_space_value(
        self,
        name: str,
        default_override: Any = None,
    ) -> Any:
        """
        Fetch hyperparameter value from user space,
        DEFAULT_PS, or default_override.
        """
        if name in self.param_space:
            return self.param_space[name]
        if name in DEFAULT_PS:
            return DEFAULT_PS[name]
        if default_override is not None:
            return default_override
        raise KeyError(
            f"Hyperparameter '{name}' not found in param_space, "
            "DEFAULT_PS, or default override."
        )

    @staticmethod
    def _map_scales_choice(
        scales_choice_str: str,
    ) -> Optional[List[int]]:
        """Convert scale choice string to list or None."""
        if scales_choice_str == "default_scales":
            return [1, 3, 7]
        if scales_choice_str == "alt_scales":
            return [1, 5, 10]
        if scales_choice_str == "no_scales":
            return None
        return None

    @staticmethod
    def _cast_hp_to_bool(
        params: Dict[str, Any],
        param_name: str,
        default_value: bool = False,
    ) -> None:
        """Cast hyperparameter to bool (in‑place)."""
        if param_name not in params:
            return
        value = params[param_name]
        if isinstance(value, (int, float)):
            params[param_name] = bool(value)
        elif not isinstance(value, bool):
            warnings.warn(
                f"Hyperparameter '{param_name}' got '{value}' "
                f"(type {type(value)}). Expected bool or 0/1. "
                f"Defaulting to {default_value}.",
            )
            params[param_name] = default_value

    @staticmethod
    def cast_multiple_bool_params(
        params: Dict[str, Any],
        bool_params_to_cast: List[Tuple[str, bool]],
    ) -> None:
        """Cast several boolean hyperparameters at once."""
        for param_name, default in bool_params_to_cast:
            BaseTuner._cast_hp_to_bool(params, param_name, default)

    def _prepare_inputs(
        self,
        inputs: List[Optional[Union[np.ndarray, Tensor]]],
        y: Union[np.ndarray, Tensor],
        forecast_horizon: int,
    ) -> Tuple[
        List[Tensor],
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        """
        Validate & convert inputs, returning tensors ready for KT.fit.
        """
        model_name_lower = self.model_name.lower()
        y_tensor = tf_convert_to_tensor(y, dtype=tf_float32)
        s_input_raw, d_input_raw, f_input_raw = None, None, None

        if model_name_lower == "tft_flex":
            if not isinstance(inputs, (list, tuple)):
                d_input_raw = inputs
            elif len(inputs) == 1:
                d_input_raw = inputs[0]
            elif len(inputs) == 2:
                s_input_raw, d_input_raw = inputs
            elif len(inputs) == 3:
                s_input_raw, d_input_raw, f_input_raw = inputs
            else:
                raise ValueError(
                    "For 'tft_flex', inputs must have 1‑3 elements."
                )
        else:
            if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
                raise ValueError(
                    f"Model '{self.model_name}' expects "
                    "inputs=[X_static, X_dynamic, X_future]."
                )
            s_input_raw, d_input_raw, f_input_raw = inputs
            if (
                s_input_raw is None
                or d_input_raw is None
                or f_input_raw is None
            ):
                raise ValueError(
                    f"Model '{self.model_name}' needs all three "
                    "inputs (static, dynamic, future) non‑None."
                )

        if d_input_raw is None:
            raise ValueError(
                "Dynamic input (X_dynamic) is mandatory and missing."
            )

        s_dim_val = (
            s_input_raw.shape[-1] if s_input_raw is not None else None
        )
        d_dim_val = d_input_raw.shape[-1]
        f_dim_val = (
            f_input_raw.shape[-1] if f_input_raw is not None else None
        )

        validation_mode = (
            "soft" if model_name_lower == "tft_flex" else "strict"
        )

        (
            X_static_val,
            X_dynamic_val,
            X_future_val,
        ) = validate_model_inputs(
            inputs=[s_input_raw, d_input_raw, f_input_raw],
            static_input_dim=s_dim_val,
            dynamic_input_dim=d_dim_val,
            future_covariate_dim=f_dim_val,
            forecast_horizon=forecast_horizon,
            error="raise",
            mode=validation_mode,
            model_name=model_name_lower,
            verbose=self.verbose >= 4,
        )

        inputs_for_fit: List[Tensor] = []
        ref_batch = tf_shape(X_dynamic_val)[0]
        ref_steps = tf_shape(X_dynamic_val)[1]

        if X_static_val is not None:
            inputs_for_fit.append(X_static_val)
        else:
            dummy_static = tf_zeros((ref_batch, 0), dtype=tf_float32)
            inputs_for_fit.append(dummy_static)
            vlog(
                f"Using dummy static input: {dummy_static.shape}",
                level=3,
                verbose=self.verbose,
            )

        inputs_for_fit.append(X_dynamic_val)

        if X_future_val is not None:
            inputs_for_fit.append(X_future_val)
        else:
            fh = self._run_case_info.get(
                "forecast_horizon", forecast_horizon
            )
            future_span = ref_steps + fh
            dummy_future = tf_zeros(
                (ref_batch, future_span, 0),
                dtype=tf_float32,
            )
            inputs_for_fit.append(dummy_future)
            vlog(
                f"Using dummy future input: {dummy_future.shape}",
                level=3,
                verbose=self.verbose,
            )

        self._run_case_info["static_input_dim"] = inputs_for_fit[0].shape[-1]
        self._run_case_info["dynamic_input_dim"] = inputs_for_fit[1].shape[-1]
        self._run_case_info["future_input_dim"] = inputs_for_fit[2].shape[-1]

        vlog(
            "Final input dims ‑ "
            f"S={self._run_case_info['static_input_dim']}, "
            f"D={self._run_case_info['dynamic_input_dim']}, "
            f"F={self._run_case_info['future_input_dim']}",
            level=3,
            verbose=self.verbose,
        )

        return (
            inputs_for_fit,
            y_tensor,
            X_static_val,
            X_dynamic_val,
            X_future_val,
        )
    
    def _model_builder_factory(
        self,
        hp: kt.HyperParameters,
        # Pass validated (non-dummy) tensors to determine actual
        # feature dimensions
        X_static_validated: Optional[Tensor],
        X_dynamic_validated: Tensor,
        X_future_validated: Optional[Tensor],
    ) -> Model:
        """
        Builds and compiles a model instance for Keras Tuner.
        This method uses `self.model_name` to select the model class and
        `self._run_case_info` for configuration.
        """
        model_name_lower = self.model_name.lower()
        case_info_param = self._run_case_info  # Use prepared case_info

        # --- Base parameters common to most models ---
        params = {
            "forecast_horizon": case_info_param.get("forecast_horizon"),
            "quantiles": case_info_param.get("quantiles"),
            "output_dim": case_info_param.get("output_dim", 1),
            "hidden_units": hp.Choice(
                "hidden_units",
                self._get_param_space_value("hidden_units", [32, 64]),
            ),
            "num_heads": hp.Choice(
                "num_heads",
                self._get_param_space_value("num_heads", [2, 4]),
            ),
            "dropout_rate": hp.Choice(
                "dropout_rate",
                self._get_param_space_value("dropout_rate", [0.0, 0.1]),
            ),
            "activation": hp.Choice(
                "activation",
                self._get_param_space_value(
                    "activation",
                    ["relu", "gelu"],
                ),
            ),
            "use_batch_norm": hp.Choice(
                "use_batch_norm",
                self._get_param_space_value(
                    "use_batch_norm",
                    [True, False],
                ),
                # Keras Tuner needs a default for boolean if not in choices
                default=False,
            ),
        }

        # --- Add input dimensions from validated data ---
        params["dynamic_input_dim"] = X_dynamic_validated.shape[-1]
        if X_static_validated is not None:
            params["static_input_dim"] = X_static_validated.shape[-1]
        if X_future_validated is not None:
            params["future_input_dim"] = X_future_validated.shape[-1]

        # --- Model‑specific parameters and class selection ---
        model_class: type[Model]
        if model_name_lower in ["xtft", "superxtft", "super_xtft"]:
            if (
                params.get("static_input_dim") is None
                or params.get("future_input_dim") is None
            ):
                raise ValueError(
                    f"{model_name_lower.upper()} requires static and "
                    "future inputs. Ensure X_static and X_future are "
                    "provided and have features."
                )
            params.update(
                {
                    "embed_dim": hp.Choice(
                        "embed_dim",
                        self._get_param_space_value("embed_dim"),
                    ),
                    "max_window_size": hp.Choice(
                        "max_window_size",
                        self._get_param_space_value("max_window_size"),
                    ),
                    "memory_size": hp.Choice(
                        "memory_size",
                        self._get_param_space_value("memory_size"),
                    ),
                    "lstm_units": hp.Choice(
                        "lstm_units",
                        self._get_param_space_value("lstm_units"),
                    ),
                    "attention_units": hp.Choice(
                        "attention_units",
                        self._get_param_space_value("attention_units"),
                    ),
                    "recurrent_dropout_rate": hp.Choice(
                        "recurrent_dropout_rate",
                        self._get_param_space_value(
                            "recurrent_dropout_rate"
                        ),
                    ),
                    "use_residuals": hp.Choice(
                        "use_residuals",
                        self._get_param_space_value("use_residuals"),
                        default=True,
                    ),
                    "final_agg": hp.Choice(
                        "final_agg",
                        self._get_param_space_value("final_agg"),
                    ),
                    "multi_scale_agg": hp.Choice(
                        "multi_scale_agg",
                        self._get_param_space_value("multi_scale_agg"),
                    ),
                    "scales": self._map_scales_choice(
                        hp.Choice(
                            "scales_options",
                            self._get_param_space_value(
                                "scales_options"
                            ),
                        )
                    ),
                }
            )
            model_class = (
                SuperXTFT
                if model_name_lower in ["super_xtft", "superxtft"]
                else XTFT
            )

        elif model_name_lower == "tft":  # Stricter TFT
            if (
                params.get("static_input_dim") is None
                or params.get("future_input_dim") is None
            ):
                raise ValueError(
                    "Stricter TFT model requires static_input_dim and "
                    "future_input_dim. Ensure X_static and X_future are "
                    "provided and have features."
                )
            params.update(
                {
                    "num_lstm_layers": hp.Choice(
                        "num_lstm_layers",
                        self._get_param_space_value("num_lstm_layers"),
                    ),
                    "lstm_units": hp.Choice(
                        "lstm_units",
                        self._get_param_space_value("lstm_units"),
                    ),
                    "recurrent_dropout_rate": hp.Choice(
                        "recurrent_dropout_rate",
                        self._get_param_space_value(
                            "recurrent_dropout_rate"
                        ),
                    ),
                }
            )
            model_class = TFTStricter

        elif model_name_lower == "tft_flex":  # Flexible TFT
            tft_flex_params = {
                "dynamic_input_dim": params["dynamic_input_dim"]
            }
            if "static_input_dim" in params:
                tft_flex_params["static_input_dim"] = params[
                    "static_input_dim"
                ]
            if "future_input_dim" in params:
                tft_flex_params["future_input_dim"] = params[
                    "future_input_dim"
                ]
            for common in [
                "forecast_horizon",
                "quantiles",
                "output_dim",
                "hidden_units",
                "num_heads",
                "dropout_rate",
                "activation",
                "use_batch_norm",
            ]:
                if common in params:
                    tft_flex_params[common] = params[common]

            tft_flex_params["num_lstm_layers"] = hp.Choice(
                "num_lstm_layers",
                self._get_param_space_value("num_lstm_layers"),
            )
            tft_flex_params["lstm_units"] = hp.Choice(
                "lstm_units",
                self._get_param_space_value("lstm_units"),
            )
            params = tft_flex_params
            model_class = TFTFlexible
        else:
            raise ValueError(
                f"Unsupported model_name for tuning factory: "
                f"{model_name_lower}"
            )

        # Cast boolean HPs that Keras Tuner might return as 0/1
        bool_params_to_cast = [
            ("use_batch_norm", False),
            ("use_residuals", True),
        ]
        self.cast_multiple_bool_params(params, bool_params_to_cast)

        # Keep only kwargs accepted by model __init__
        valid_model_params = _get_valid_kwargs(
            model_class.__init__,
            params,
        )

        model = model_class(**valid_model_params)

        learning_rate = hp.Choice(
            "learning_rate",
            self._get_param_space_value(
                "learning_rate",
                [1e-3, 5e-4],
            ),
        )
        loss_to_use = self._get_param_space_value("loss", "mse")

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss_to_use,
        )
        vlog(
            f"{model_name_lower.upper()} model built and compiled for "
            "trial.",
            level=3,
            verbose=case_info_param.get("verbose_build", 0),
        )
        return model

    def fit(
        self,
        inputs: List[Optional[Union[np.ndarray, Tensor]]],
        y: Union[np.ndarray, Tensor],
        forecast_horizon: int = 1,
        quantiles: Optional[List[float]] = None,
        case_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[
        Optional[Dict[str, Any]],
        Optional[Model],
        Optional[kt.Tuner],
    ]:
        """
        Execute the complete tuning workflow.

        The routine proceeds as follows:

        1.  **Register loss** (point or quantile) inside *param_space* so
            that the model factory can pick it up.
        2.  **Validate & cast** inputs via
            :func:`~fusionlab.nn._tensor_validation.validate_model_inputs`.
        3.  **Instantiate** a Keras Tuner (random or Bayesian) and run
            :pymeth:`~keras_tuner.Tuner.search`.
        4.  For every *batch_size* **refit** the winning trial for
            *epochs* epochs.
        5.  **Select** the global best model / hyper‑parameters and store
            a JSON summary.

        Parameters
        ----------
        inputs : list[np.ndarray | Tensor]
            A list containing three input tensors:
              - ``X_static``: static features with shape (``B, N_s``)
              - ``X_dynamic``: dynamic features with shape (``B, F, N_d``)
              - ``X_future``: future features with shape (``B, F, N_f``)
            Here, :math:`B` is the batch size, :math:`N_s` is the number
            of static features, :math:`F` is the forecast horizon, 
            :math:`N_d` is the number of dynamic features, and 
            :math:`N_f` is the number of future features.
            
            Ordered list ``[X_static, X_dynamic, X_future]``.  Models
            labelled ``*_flex`` may accept *None* for absent blocks.
            
        y : np.ndarray | Tensor
            Target array shaped ``(B, F, O)`` or ``(B, O)`` for
            single‑step forecasting.
        forecast_horizon : int, default ``1``
            Number of future steps the model must forecast.
        quantiles : list[float], optional
            Quantile levels for probabilistic losses (e.g.
            ``[0.1, 0.5, 0.9]``).  When *None* a point loss is used.
        case_info : dict, optional
            Extra metadata merged into ``_run_case_info`` – useful to tag
            runs or override default description strings.

        Returns
        -------
        tuple
            ``(best_hps, best_model, tuner_instance)`` where *best_hps* is
            a ``dict`` (or *None* if tuning failed), *best_model* the
            trained :class:`tf.keras.Model`, and *tuner_instance* the
            configured :class:`keras_tuner.Tuner`.

        Raises
        ------
        ValueError
            If mandatory input blocks are missing or dimensions mismatch.
        ImportError
            When *keras_tuner* is unavailable and auto‑install is disabled.
        """
        forecast_horizon= validate_positive_integer(
            forecast_horizon, "forecast_horizon", 
       )
        # --- 1. Initial Setup ---
        current_loss_fn_obj = (
            combined_quantile_loss(quantiles)
            if quantiles is not None
            else "mse"
        )
        self.param_space["loss"] = current_loss_fn_obj

        self._run_case_info = CASE_INFO.copy()
        self._run_case_info.update(
            {
                "description": self._run_case_info[
                    "description"
                ].format("Quantile" if quantiles else "Point"),
                "forecast_horizon": forecast_horizon,
                "quantiles": quantiles,
                "output_dim": (
                    y.shape[-1]
                    if hasattr(y, "ndim") and y.ndim == 3
                    else 1
                ),
                "verbose_build": self.verbose >= 3,
            }
        )
        if case_info:
            self._run_case_info.update(case_info)

        vlog(
            f"Starting {self.model_name.upper()} "
            f"{self.tuner_type.upper()} tune...",
            level=1,
            verbose=self.verbose,
        )

        # --- 2. Prepare Inputs ---
        (
            inputs_for_fit,
            y_tensor,
            X_s_val,
            X_d_val,
            X_f_val,
        ) = self._prepare_inputs(inputs, y, forecast_horizon)
        vlog("Inputs prepared and validated.", level=2, verbose=self.verbose)

        # --- 3. Model Builder Setup ---
        actual_model_builder = self.custom_model_builder
        if actual_model_builder is None:
            vlog(
                "Using default internal _model_builder_factory.",
                level=2,
                verbose=self.verbose,
            )
            actual_model_builder = (
                lambda hp: self._model_builder_factory(
                    hp,
                    X_s_val,
                    X_d_val,
                    X_f_val,
                )
            )

        # --- 4. Callbacks Setup ---
        actual_callbacks = self.callbacks
        if actual_callbacks is None:
            vlog(
                "Setting default EarlyStopping callback.",
                level=2,
                verbose=self.verbose,
            )
            actual_callbacks = [
                EarlyStopping(
                    monitor=self._get_param_space_value(
                        "monitor",
                        "val_loss",
                    ),
                    patience=self._get_param_space_value(
                        "patience",
                        10,
                    ),
                    restore_best_weights=True,
                )
            ]

        # --- 5. Tuner Initialization ---
        tuner_dir = self.tuner_dir or os.path.join(
            os.getcwd(),
            "fusionlab_tuning_results",
        )
        project_name = self.project_name or (
            f"{self.model_name.upper()}_Tune_"
            f"{self._run_case_info.get('description','').replace(' ','_')}"
        )

        common_tuner_args = {
            "hypermodel": actual_model_builder,
            "objective": self.objective,
            "max_trials": self.max_trials,
            "directory": tuner_dir,
            "project_name": project_name,
            "overwrite": True,
        }
        if self.tuner_type == "bayesian":
            self.tuner_ = kt.BayesianOptimization(**common_tuner_args)
        else:
            self.tuner_ = kt.RandomSearch(**common_tuner_args)
        vlog(
            f"Keras Tuner initialized: {self.tuner_type.upper()} "
            f"for {project_name}",
            level=1,
            verbose=self.verbose,
        )

        # --- 6. Tuning Loop ---
        self.best_hps_ = None
        self.best_model_ = None
        overall_best_val_loss = np.inf
        overall_best_batch_size = None
        self.tuning_log_ = []

        for current_batch_size in self.batch_sizes:
            vlog(
                f"--- Tuning with Batch Size: {current_batch_size} ---",
                level=1,
                verbose=self.verbose,
            )
            try:
                self.tuner_.search(
                    x=inputs_for_fit,
                    y=y_tensor,
                    epochs=self.search_epochs,
                    batch_size=current_batch_size,
                    validation_split=self.validation_split,
                    callbacks=actual_callbacks,
                    verbose=self.verbose >= 2,
                )
                current_best_trial_hps = (
                    self.tuner_.get_best_hyperparameters(
                        num_trials=1
                    )[0]
                )
                vlog(
                    "  Best HPs for batch "
                    f"{current_batch_size} (search phase): "
                    f"{current_best_trial_hps.values}",
                    level=2,
                    verbose=self.verbose,
                )

                current_best_model = (
                    self.tuner_.hypermodel.build(
                        current_best_trial_hps
                    )
                )
                vlog(
                    "  Training best model for batch "
                    f"{current_batch_size} "
                    f"for {self.epochs} epochs...",
                    level=2,
                    verbose=self.verbose,
                )
                history = current_best_model.fit(
                    x=inputs_for_fit,
                    y=y_tensor,
                    epochs=self.epochs,
                    batch_size=current_batch_size,
                    validation_split=self.validation_split,
                    callbacks=actual_callbacks,
                    verbose=self.verbose >= 2,
                )
                current_model_val_loss = min(
                    history.history["val_loss"]
                )
                vlog(
                    f"  Batch Size {current_batch_size}: "
                    f"Final val_loss = "
                    f"{current_model_val_loss:.4f}",
                    level=1,
                    verbose=self.verbose,
                )

                trial_info = {
                    "batch_size": current_batch_size,
                    "best_val_loss_for_batch": current_model_val_loss,
                    "hyperparameters": (
                        current_best_trial_hps.values
                    ),
                }
                self.tuning_log_.append(trial_info)

                if current_model_val_loss < overall_best_val_loss:
                    overall_best_val_loss = current_model_val_loss
                    self.best_hps_ = (
                        current_best_trial_hps.values.copy()
                    )
                    self.best_model_ = current_best_model
                    overall_best_batch_size = current_batch_size
                    if self.best_hps_ is not None:
                        self.best_hps_[
                            "batch_size"
                        ] = overall_best_batch_size

            except Exception as e:
                vlog(
                    "Tuning failed for batch size "
                    f"{current_batch_size}. Error: {e}",
                    level=0,
                    verbose=self.verbose,
                )
                warnings.warn(
                    f"Tuning for batch {current_batch_size} failed: {e}"
                )
                continue

        # --- 7. Finalize and Log ---
        if self.best_model_ is None:
            vlog(
                "Hyperparameter tuning failed for all batch sizes.",
                level=0,
                verbose=self.verbose,
            )
            return None, None, self.tuner_

        self._save_log(
            tuner_dir,
            project_name,
            overall_best_batch_size,
            overall_best_val_loss,
        )

        vlog("--- Overall Best ---", level=1, verbose=self.verbose)
        vlog(
            f"Best Batch Size: {overall_best_batch_size}",
            level=1,
            verbose=self.verbose,
        )
        if self.best_hps_ is not None:
            summary = ResultSummary(
                "BestHyperParameters"
            ).add_results(self.best_hps_)
            vlog(
                f"Best Hyperparameters:\n {summary}",
                level=1,
                verbose=self.verbose,
            )
        vlog(
            f"Best Validation Loss: "
            f"{overall_best_val_loss:.4f}",
            level=1,
            verbose=self.verbose,
        )

        return self.best_hps_, self.best_model_, self.tuner_

    def _save_log(
        self,
        tuner_dir: str,
        project_name: str,
        overall_best_batch_size: Optional[int],
        overall_best_val_loss: float,
    ) -> None:
        """Saves the tuning summary to a JSON file."""
        if self.best_hps_ is not None:
            self.tuning_log_.append(
                {
                    "overall_best_batch_size": (
                        overall_best_batch_size
                    ),
                    "overall_best_val_loss": overall_best_val_loss,
                    "overall_best_hyperparameters": self.best_hps_,
                }
            )

        log_file_path = os.path.join(
            tuner_dir,
            f"{project_name}_tuning_summary.json",
        )
        try:
            os.makedirs(tuner_dir, exist_ok=True)
            with open(log_file_path, "w") as f:
                json.dump(
                    self.tuning_log_,
                    f,
                    indent=4,
                    default=str,
                )
            vlog(
                f"Full tuning summary saved to {log_file_path}",
                level=1,
                verbose=self.verbose,
            )
        except Exception as e:
            warnings.warn(
                "Could not save tuning summary log to "
                f"{log_file_path}: {e}"
            )

# ------------------------------------------------------------------
# Shared documentation components for tuner classes are defined in
# `_tuner_docs` 
# ------------------------------------------------------------------

BaseTuner.__doc__ = (
    """
    Base class for hyperparameter tuning of time-series forecasting models.

    This class provides the core framework for setting up and running
    Keras Tuner optimization, including input preparation, model building,
    and results logging. Specific tuner implementations for different
    model architectures (e.g., XTFT, TFT) should inherit from this class.
    
    Universal hyper‑parameter tuning scaffold that powers both
    :class:`XTFTTuner` and :class:`TFTTuner`.

    The class bundles:

    * **Input validation** – converts NumPy / Tensor inputs to
      ``float32`` and enforces dimensional consistency via
      :func:`fusionlab.nn._tensor_validation.validate_model_inputs`.
    * **Model factory** – builds a compiled model from a
      :class:`keras_tuner.HyperParameters` instance.
    * **Batch‑size sweep** – runs an independent Keras Tuner search
      for every value in *batch_sizes*, then refits the champion
      trial for additional epochs.
    * **Result logging** – stores trial summaries and a final
      JSON report under *tuner_dir/project_name*.

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

    Attributes
    ----------
    best_hps_ : dict | None
        Mapping of the best hyper‑parameters discovered.
    best_model_ : tf.keras.Model | None
        Fully trained model achieving *overall_best_val_loss*.
    tuner_ : keras_tuner.Tuner | None
        Underlying Keras Tuner object.
    tuning_log_ : list[dict]
        Chronological list of per‑batch results – ultimately persisted
        to ``<tuner_dir>/<project_name>_tuning_summary.json``.
    """
).format(params=_tuner_docs)


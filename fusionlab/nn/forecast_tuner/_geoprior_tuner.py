# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import (
    Dict, Optional, Any, Union, Type, Tuple, List, Callable
)
import logging
import numpy as np

from ..._fusionlog import fusionlog
from ...core.handlers import _get_valid_kwargs
from ...utils.generic_utils import (
    vlog, rename_dict_keys, cast_multiple_bool_params
)
from ..pinn.models import GeoPriorSubsNet
from ..pinn.utils import check_required_input_keys
from .. import KERAS_DEPS
from . import KT_DEPS
from ._base_tuner import PINNTunerBase

HyperParameters = KT_DEPS.HyperParameters
Objective = KT_DEPS.Objective
Tuner = KT_DEPS.Tuner

AUTOTUNE = KERAS_DEPS.AUTOTUNE
Model = KERAS_DEPS.Model
Adam = KERAS_DEPS.Adam
MeanSquaredError = KERAS_DEPS.MeanSquaredError
MeanAbsoluteError = KERAS_DEPS.MeanAbsoluteError
Callback = KERAS_DEPS.Callback
Dataset = KERAS_DEPS.Dataset

tf_float32 = KERAS_DEPS.float32
tf_const = KERAS_DEPS.constant
tf_cast = KERAS_DEPS.cast
tf_reduce_mean = KERAS_DEPS.reduce_mean
tf_max = KERAS_DEPS.maximum
tf_expand = KERAS_DEPS.expand_dims

logger = fusionlog().get_fusionlab_logger(__name__)

# ------------------------------
# Defaults specialized to model
# ------------------------------

_DEFAULT_COMMON = {
    "output_subsidence_dim": 1,
    "output_gwl_dim": 1,
    "quantiles": None,
    "max_window_size": 10,
    "memory_size": 100,
    "scales": [1],
    "multi_scale_agg": "last",
    "final_agg": "last",
    "use_residuals": True,
    "use_batch_norm": False,
    "activation": "relu",
    "architecture_config": {
        "encoder_type": "hybrid",
        "decoder_attention_stack": [
            "cross", "hierarchical", "memory"
        ],
        "feature_processing": "vsn",
    },
    "loss_weights": {"subs_pred": 1.0, "gwl_pred": 1.0},
}

_DEFAULT_GEOPRIOR = {
    **_DEFAULT_COMMON,
    "pde_mode": "both",
    "mv": 1e-7,
    "kappa": 1.0,
    "gamma_w": 9810.0,
    "h_ref": 0.0,
    "use_effective_h": False,
    "hd_factor": 1.0,
    "kappa_mode": "bar",
    "scale_pde_residuals": True,
    "scaling_kwargs": {},
}

# Compile-only HPs
_COMPILE_ONLY = {
    "learning_rate",
    "lambda_gw",
    "lambda_cons",
    "lambda_prior",
    "lambda_smooth",
    "lambda_mv",
    "mv_lr_mult",
    "kappa_lr_mult",
}

_INT_FIELDS = (
    "embed_dim", "hidden_units", "lstm_units",
    "attention_units", "vsn_units", "num_heads",
)

_BOOL_CAST = [
    ("use_vsn", True),
    ("use_residuals", True),
    ("use_batch_norm", False),
    ("use_effective_h", False),
    ("scale_pde_residuals", True),
]


class GeoPriorTuner(PINNTunerBase):
    r"""
    Specialized tuner for ``GeoPriorSubsNet`` models.

    This class provides a flexible hyperparameter tuner for the
    physics-informed ``GeoPriorSubsNet``. It builds on
    ``PINNTunerBase`` and uses keras-tuner backends to search
    architectural, physics, and compile-time spaces. Fixed,
    data-dependent parameters are separated from the search
    space, and GeoPrior input checks are enforced during
    ``run`` and ``create``.

    The recommended entry point is ``GeoPriorTuner.create``,
    which infers data dimensions from NumPy arrays, merges
    them with robust defaults, and applies user overrides.

    Parameters
    ----------
    fixed_params : dict
        Non-tunable configuration passed to the model
        ``__init__``. Include data-shape keys and stable
        flags, for example:
            
            - ``static_input_dim`` : int
            - ``dynamic_input_dim`` : int
            - ``future_input_dim`` : int
            - ``output_subsidence_dim`` : int
            - ``output_gwl_dim`` : int
            - ``forecast_horizon`` : int
            - optional flags (e.g., ``use_batch_norm``)
            - physics toggles (e.g., ``pde_mode``)
            
    search_space : dict, optional
        Hyperparameter definitions. Each entry is either a
        list of discrete choices or a dict with a typed
        range, e.g.:
            
            - list: ``{"embed_dim": [32, 64, 96]}``
            - dict: ``{"dropout_rate": {"type": "float",
              "min_value": 0.1, "max_value": 0.4}}``
            
        Supported types are ``int``, ``float``, ``choice``,
        and ``bool``. Items are routed to model ``__init__``
        or to ``compile`` (see Notes).
        
    objective : str or keras_tuner.Objective, default "val_loss"
        Metric to optimize. If the string contains "loss",
        direction is inferred as "min".
    max_trials : int, default 10
        Maximum number of trials evaluated by the tuner.
    project_name : str, default "GeoPriorTuner_Project"
        Project name used for directory layout.
    directory : str, default "geoprior_tuner_results"
        Root directory where tuner artifacts are saved.
    executions_per_trial : int, default 1
        Number of repeated trainings per hyperparameter set.
    tuner_type : {"randomsearch", "bayesianoptimization",
                  "hyperband"}, default "randomsearch"
        Search algorithm used by keras-tuner.
    seed : int, optional
        Random seed for reproducibility.
    overwrite : bool, default True
        If True, existing project results are overwritten.
    _logger : logging.Logger or callable, optional
        Logger or print-like callable for progress lines.
    **kwargs
        Forwarded to ``PINNTunerBase`` for advanced control.

    Attributes
    ----------
    model_class : Type[tf.keras.Model]
        Bound to ``GeoPriorSubsNet``.
    fixed_params : dict
        Finalized, non-tunable model configuration.
    search_space : dict
        User-provided hyperparameter search definitions.
    best_hps_ : keras_tuner.HyperParameters or None
        Best hyperparameters found by the search.
    best_model_ : tf.keras.Model or None
        Model built with the best hyperparameters.
    tuner_ : keras_tuner.Tuner or None
        Underlying tuner instance after ``run`` or ``search``.
    tuning_summary_ : dict
        Compact summary saved under the project directory.

    Notes
    -----
    Required inputs. ``GeoPriorSubsNet`` expects the keys in
    ``inputs``:
        - ``coords`` : spatiotemporal coordinates
        - ``dynamic_features`` : time-varying covariates
        - ``H_field`` : soil thickness field
    The helper canonicalizes ``H_field`` from common aliases,
    e.g., ``soil_thickness``, ``soil thickness``, ``h_field``.

    Targets are canonicalized to ``subs_pred`` and
    ``gwl_pred`` from ``subsidence`` and ``gwl``. This is
    handled internally before data pipelines are built.

    Compile-only hyperparameters. These search keys are not
    passed to ``__init__`` and are routed to ``compile``:
        
        - ``learning_rate``
        - ``lambda_gw``, ``lambda_cons``, ``lambda_prior``,
          ``lambda_smooth``, ``lambda_mv``
        - ``mv_lr_mult``, ``kappa_lr_mult``

    Losses and metrics. By default the supervised heads use
    mean squared error with mean absolute error metrics. If
    ``fixed_params["quantiles"]`` is set, a pinball loss can
    be injected via a user loss factory at compile time.

    Physics objectives. GeoPrior adds residuals consistent
    with consolidation and groundwater flow:
        
        - :math:`R_gw = Ss * d(h)/dt - div(K * grad(h)) - Q`
        - :math:`R_cons = d(s)/dt - (s_eq - s) / tau`
      with :math:`s_eq = m_v * gamma_w * (h_ref - h) * H`.
      Weights are controlled by the compile-time lambdas.

    Typical search groups:
        - Architecture: ``embed_dim``, ``hidden_units``,
          ``lstm_units``, ``attention_units``, ``num_heads``,
          ``dropout_rate``, ``vsn_units``, ``use_vsn``,
          ``use_batch_norm``.
        - Physics: ``pde_mode``, ``mv``, ``kappa``,
          ``use_effective_h``, ``hd_factor``, ``kappa_mode``,
          ``scale_pde_residuals``.
        - Optimization: ``learning_rate`` and lambda weights.

    Workflow. ``run`` builds ``tf.data`` pipelines from NumPy
    inputs, applies key canonicalization, validates GeoPrior
    requirements, and delegates to ``search``. The best HPs
    and a built model are returned and stored on the class.

    Examples
    --------
    Create from arrays and tune a small space.

    >>> from fusionlab.nn.forecast_tuner import GeoPriorTuner
    >>> fixed = {"forecast_horizon": 7}
    >>> space = {
    ...   "embed_dim": [32, 64],
    ...   "num_heads": [2, 4],
    ...   "dropout_rate": {"type": "float",
    ...     "min_value": 0.1, "max_value": 0.3},
    ...   "learning_rate": [1e-3, 5e-4],
    ...   "lambda_gw": {"type": "float",
    ...     "min_value": 0.5, "max_value": 1.5},
    ... }
    >>> tuner = GeoPriorTuner.create(
    ...   inputs_data=inputs_np,
    ...   targets_data=targets_np,
    ...   search_space=space,
    ...   fixed_params=fixed,
    ...   max_trials=20,
    ...   project_name="GeoPrior_HP_Search",
    ... )
    >>> best_model, best_hps, kt = tuner.run(
    ...   inputs=inputs_np,
    ...   y=targets_np,
    ...   validation_data=(val_inputs_np, val_targets_np),
    ...   epochs=30,
    ...   batch_size=32,
    ... )

    See Also
    --------
    PINNTunerBase
        Base hypermodel with the generic ``search`` routine.
    GeoPriorSubsNet
        Target model with physics residuals and priors.
    HydroTuner
        Generic PINN tuner for HAL and TransFlow models.

    References
    ----------
    .. [1] Keras Tuner. https://keras.io/keras_tuner/
    .. [2] Terzaghi, K. Theory of Consolidation, 1943.
    .. [3] Bear, J. Dynamics of Fluids in Porous Media, 1972.
    """

    def __init__(
        self,
        fixed_params: Dict[str, Any],
        search_space: Optional[Dict[str, Any]] = None,
        objective: Union[str, "Objective"] = "val_loss",
        max_trials: int = 10,
        project_name: str = "GeoPriorTuner_Project",
        directory: str = "geoprior_tuner_results",
        executions_per_trial: int = 1,
        tuner_type: str = "randomsearch",
        seed: Optional[int] = None,
        overwrite: bool = True,
        _logger: Optional[Union[
            Callable[[str], None], "logging.Logger"
        ]] = None,
        **kwargs,
    ):
        self._logger = _logger or print
        super().__init__(
            objective=objective,
            max_trials=max_trials,
            project_name=project_name,
            directory=directory,
            executions_per_trial=executions_per_trial,
            tuner_type=tuner_type,
            seed=seed,
            overwrite_tuner=overwrite,
            _logger=self._logger,
            **kwargs,
        )
        self.fixed_params = dict(fixed_params or {})
        self.search_space = dict(search_space or {})
        self.model_class: Type[Model] = GeoPriorSubsNet

    @classmethod
    def create(
        cls,
        inputs_data: Dict[str, np.ndarray],
        targets_data: Dict[str, np.ndarray],
        search_space: Dict[str, Any],
        fixed_params: Optional[Dict[str, Any]] = None,
        **tuner_kwargs,
    ) -> "GeoPriorTuner":
        t_std = rename_dict_keys(
            targets_data.copy(),
            param_to_rename={
                "subsidence": "subs_pred",
                "gwl": "gwl_pred",
            },
        )
        final_fixed = cls._infer_and_merge_params(
            inputs_data=inputs_data,
            targets_data=t_std,
            user_fixed_params=fixed_params,
        )
        return cls(
            fixed_params=final_fixed,
            search_space=search_space,
            **tuner_kwargs,
        )

    def _create_hyperparameter(
        self, hp: "HyperParameters", name: str, definition: Any
    ) -> Union[int, float, str, bool]:
        if isinstance(definition, list):
            return hp.Choice(name, definition)
        if isinstance(definition, dict):
            hp_type = definition.get("type", "float")
            kw = {k: v for k, v in definition.items() if k != "type"}
            if hp_type == "int":
                return hp.Int(name, **kw)
            if hp_type == "float":
                return hp.Float(name, **kw)
            if hp_type == "choice":
                return hp.Choice(name, **kw)
            if hp_type == "bool":
                return hp.Boolean(name, **kw)
        raise TypeError(
            f"Unsupported HP def for '{name}': {definition}"
        )

    # ----------------------------
    # Build per-trial model
    # ----------------------------
    def build(self, hp: "HyperParameters") -> "Model":
        init_params = dict(self.fixed_params)
        compile_hps: Dict[str, Any] = {}
        for name, spec in self.search_space.items():
            val = self._create_hyperparameter(hp, name, spec)
            if name in _COMPILE_ONLY:
                compile_hps[name] = val
            else:
                init_params[name] = val

        for k in _INT_FIELDS:
            if k in init_params:
                init_params[k] = int(init_params[k])

        valid_init = _get_valid_kwargs(
            self.model_class.__init__,
            init_params,
            error="ignore",
        )
        cast_multiple_bool_params(
            valid_init,
            bool_params_to_cast=_BOOL_CAST,
        )
        model = self.model_class(**valid_init)

        # choose losses: pinball if quantiles set
        quantiles = valid_init.get("quantiles")
        if quantiles:
            pinball = _pinball_factory(quantiles)
            loss = {
                "subs_pred": pinball,
                "gwl_pred": pinball,
            }
            metrics = {}   # avoid MAE on Q-dim tensors
        else:
            loss = {
                "subs_pred": MeanSquaredError(name="subs_data_loss"),
                "gwl_pred": MeanSquaredError(name="gwl_data_loss"),
            }
            metrics = {
                "subs_pred": [MeanAbsoluteError(name="subs_mae")],
                "gwl_pred": [MeanAbsoluteError(name="gwl_mae")],
            }

        lr = compile_hps.pop("learning_rate", 1e-3)
        optimizer = Adam(learning_rate=lr, clipnorm=1.0, clipvalue=0.5)

        valid_compile = _get_valid_kwargs(
            model.compile,
            compile_hps,
            error="ignore",
        )
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics or None,
            loss_weights=self.fixed_params.get(
                "loss_weights",
                {"subs_pred": 1.0, "gwl_pred": 1.0},
            ),
            **valid_compile,
        )
        return model

    # ----------------------------
    # Run end-to-end search
    # ----------------------------
    def run(
        self,
        inputs: Dict[str, np.ndarray],
        y: Dict[str, np.ndarray],
        validation_data: Optional[
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        ] = None,
        epochs: int = 10,
        batch_size: int = 32,
        callbacks: Optional[List["Callback"]] = None,
        case_info: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        **search_kwargs,
    ) -> Tuple[
        Optional["Model"],
        Optional["HyperParameters"],
        Optional["Tuner"],
    ]:
        vlog(
            "GeoPriorTuner: starting run...",
            verbose=verbose,
            level=1,
            logger=self._logger,
        )

        # fast fail for H_field presence
        _require_h_field(inputs, "inputs")
        if validation_data:
            _require_h_field(validation_data[0], "val_inputs")

        req = [
            "static_input_dim", "dynamic_input_dim",
            "future_input_dim", "output_subsidence_dim",
            "output_gwl_dim", "forecast_horizon",
        ]
        if not all(k in self.fixed_params for k in req):
            vlog(
                "Inferring fixed params from data...",
                verbose=verbose,
                level=2,
                logger=self._logger,
            )
            y_std = rename_dict_keys(
                y.copy(),
                param_to_rename={
                    "subsidence": "subs_pred",
                    "gwl": "gwl_pred",
                },
            )
            self.fixed_params = self._infer_and_merge_params(
                inputs_data=inputs,
                targets_data=y_std,
                user_fixed_params=self.fixed_params,
            )

        inputs, y = check_required_input_keys(
            inputs, y, model_name="GeoPrior"
        )

        train_ds = (
            Dataset.from_tensor_slices((inputs, y))
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )

        val_ds = None
        if validation_data:
            vx, vy = validation_data
            vx, vy = check_required_input_keys(
                vx, vy, model_name="GeoPrior"
            )
            
            val_ds = (
                Dataset.from_tensor_slices((vx, vy))
                .batch(batch_size)
                .prefetch(AUTOTUNE)
            )

        metric_kind = (
            "Quantile" if self.fixed_params.get("quantiles")
            else "Point"
        )
        self._current_run_case_info = {
            "description": (
                f"GeoPriorSubsNet {metric_kind} forecast"
            ),
        }
        self._current_run_case_info.update(self.fixed_params)
        if case_info:
            self._current_run_case_info.update(case_info)

        vlog(
            "Delegating to base.search()...",
            verbose=verbose,
            level=2,
            logger=self._logger,
        )
        return super().search(
            train_data=train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=verbose,
            **search_kwargs,
        )

    @staticmethod
    def _infer_and_merge_params(
        inputs_data: Dict[str, np.ndarray],
        targets_data: Dict[str, np.ndarray],
        user_fixed_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        final = dict(_DEFAULT_GEOPRIOR)

        inputs_data, targets_data = check_required_input_keys(
            inputs_data, targets_data,  
            model_name="GeoPrior"
        )

        inferred: Dict[str, Any] = {}
        if "static_features" in inputs_data:
            inferred["static_input_dim"] = (
                inputs_data["static_features"].shape[-1]
            )
        else:
            inferred["static_input_dim"] = 0

        inferred["dynamic_input_dim"] = (
            inputs_data["dynamic_features"].shape[-1]
        )

        if "future_features" in inputs_data:
            inferred["future_input_dim"] = (
                inputs_data["future_features"].shape[-1]
            )
        else:
            inferred["future_input_dim"] = 0

        inferred["output_subsidence_dim"] = (
            targets_data["subs_pred"].shape[-1]
        )
        inferred["output_gwl_dim"] = (
            targets_data["gwl_pred"].shape[-1]
        )
        inferred["forecast_horizon"] = (
            targets_data["subs_pred"].shape[1]
        )

        final.update(inferred)
        if user_fixed_params:
            final.update(user_fixed_params)
            
        return final

def _require_h_field(d: Dict[str, Any], tag: str) -> None:
    if ("H_field" not in d) and ("soil_thickness" not in d):
        raise ValueError(
            f"{tag} must contain 'H_field' or 'soil_thickness'."
        )


def _pinball_factory(qs: List[float]):
    q_base = tf_const(qs, dtype=tf_float32)  # shape (Q,)

    def loss(y_true, y_pred):
        yt = tf_cast(y_true, tf_float32)
        yp = tf_cast(y_pred, tf_float32)

        # --- Normalize shapes ---
        # y_pred: either (B,H,Q) or (B,H,Q,1)
        if yp.shape.rank == 3:
            # want y_true as (B,H,1) so it broadcasts across Q
            if yt.shape.rank == 2:               # (B,H) -> (B,H,1)
                yt = tf_expand(yt, axis=-1)
            # if yt is already (B,H,1), leave it
            q = q_base[None, None, :]            # (1,1,Q) -> matches (B,H,Q)

        elif yp.shape.rank == 4:
            # want y_true as (B,H,1,1) to match (B,H,Q,1)
            if yt.shape.rank == 2:               # (B,H) -> (B,H,1)
                yt = tf_expand(yt, axis=-1)
            if yt.shape.rank == 3:               # (B,H,1) -> (B,H,1,1)
                yt = tf_expand(yt, axis=2)
            q = q_base[None, None, :, None]      # (1,1,Q,1) -> matches (B,H,Q,1)

        else:
            raise ValueError("y_pred must be rank 3 (B,H,Q) or rank 4 (B,H,Q,1).")

        # --- Pinball loss ---
        err = yt - yp
        pin = tf_max(q * err, (q - 1.0) * err)
        return tf_reduce_mean(pin)

    return loss


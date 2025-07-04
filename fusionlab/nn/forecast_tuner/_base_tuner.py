# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Base classes and utilities for hyperparameter tuning of PINN models.
"""
import os
import json
import warnings
import logging 
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from ..._fusionlog import fusionlog 
from ...api.docs import DocstringComponents
from ...api.docs import _pinn_tuner_common_params
from ...api.property import BaseClass
from ...utils.generic_utils import vlog, rename_dict_keys  
from ...utils.deps_utils import ensure_pkg
 
from .. import KERAS_DEPS
from .._config import Config 
from . import KT_DEPS, HAS_KT

HyperModel = KT_DEPS.HyperModel
Tuner = KT_DEPS.Tuner
RandomSearch = KT_DEPS.RandomSearch
KT_DEPS.BayesianOptimization
BayesianOptimization = KT_DEPS.Hyperband
HyperParameters = KT_DEPS.HyperParameters
Objective = KT_DEPS.Objective
Hyperband = KT_DEPS.Hyperband

Model = KERAS_DEPS.Model
Callback = KERAS_DEPS.Callback
Dataset = KERAS_DEPS.Dataset 
Adam = KERAS_DEPS.Adam
EarlyStopping = KERAS_DEPS.EarlyStopping
AUTOTUNE =KERAS_DEPS.AUTOTUNE 
    
_pinn_tuner_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_pinn_tuner_common_params)
)

logger = fusionlog().get_fusionlab_logger(__name__)

class PINNTunerBase(HyperModel, BaseClass):
    @ensure_pkg(
        "keras_tuner",
        extra="'keras_tuner' is required for model tuning.",
        auto_install=Config.INSTALL_DEPS, 
        use_conda=Config.USE_CONDA
    )
    def __init__(
        self,
        objective: Union[str, Objective] = 'val_loss',
        max_trials: int = 10,
        project_name: str = "PINN_Tuning",
        directory: str = "pinn_tuner_results",
        executions_per_trial: int = 1,
        tuner_type: str = 'randomsearch',
        seed: Optional[int] = None,
        overwrite_tuner: bool = True,
        _logger: Optional[Union[logging.Logger, Callable[[str], None]]] = None,
        **tuner_kwargs
    ):
        if not HAS_KT:
            raise ImportError(
                "keras_tuner is not installed. Please run "
                "`pip install keras-tuner` to use this tuning class."
            )
        super().__init__() 
        
        self.objective = objective
        self.max_trials = max_trials
        self.project_name = project_name
        self.directory = directory
        self.executions_per_trial = executions_per_trial
        self.tuner_type = self._validate_tuner_type(tuner_type)
        self.seed = seed
        self.overwrite_tuner = overwrite_tuner
        self.tuner_kwargs = tuner_kwargs
        self._logger = _logger or print 

        self.best_hps_: Optional[HyperParameters] = None
        self.best_model_: Optional[Model] = None
        self.tuner_: Optional[Tuner] = None
        
        self.tuning_summary_: Dict[str, Any] = {}
        self.fixed_model_params: Dict[str, Any] = {}
        self.param_space_config: Dict[str, Any] = {}
        
        if isinstance(self.objective, str):
            # Default: any metric name containing "loss" is minimized
            direction = "min" if "loss" in self.objective else "max"
            self.objective = Objective(self.objective, direction=direction)


    def _validate_tuner_type(self, tuner_type: str) -> str:
        valid_types = {"randomsearch", "bayesianoptimization", "hyperband"}
        tt_lower = tuner_type.lower()
        # Allow partial match for "random"
        if "random" in tt_lower: 
            tt_lower = "randomsearch"
        if "bayesian" in tt_lower:
            tt_lower = "bayesianoptimization"
            
        if tt_lower not in valid_types:
            warnings.warn(
                f"Unsupported tuner type: '{tuner_type}'. "
                f"Supported types: {valid_types}. "
                "Defaulting to 'randomsearch'.", UserWarning
            )
            return "randomsearch"
        
        return tt_lower

    def build(self, hp: HyperParameters) -> Model:
        """
        Builds and compiles the Keras model with hyperparameters.

        This method **must be overridden** by subclasses (e.g., PIHALTuner)
        to define the specific model architecture (like PIHALNet), sample
        hyperparameters using the `hp` object based on
        `self.param_space`, and compile the model.

        Args:
            hp (kt.HyperParameters): Keras Tuner HyperParameters object.

        Returns:
            tf.keras.Model: The compiled Keras model.
        """
        raise NotImplementedError(
            "Subclasses must implement the `build(hp)` method."
        )

    def search(
        self,
        train_data: Dataset,
        epochs: int,
        validation_data: Optional[Dataset] = None,
        callbacks: Optional[List[Callback]] = None,
        verbose: int = 1, 
        patience: int = 10,
        **additional_search_kwargs
    ) -> Tuple[Optional[Model], Optional[HyperParameters], Optional[Tuner]]:
        """
        Performs the hyperparameter search using Keras Tuner.

        Args:
            train_data (tf.data.Dataset): Training dataset. Must yield
                tuples of `(inputs_dict, targets_dict)` compatible with
                the model's `train_step`.
            epochs (int): Number of epochs to train each model during a trial.
            validation_data (tf.data.Dataset, optional): Validation dataset.
            callbacks (List[tf.keras.callbacks.Callback], optional):
                List of Keras callbacks for the search phase.
            verbose (int, default 1): Verbosity level for Keras Tuner search.
            **additional_search_kwargs: Additional keyword arguments passed to
                the Keras Tuner's `search()` method.

        Returns:
            Tuple[Optional[tf.keras.Model], Optional[kt.HyperParameters], Optional[kt.Tuner]]:
                - The best model instance built with the best hyperparameters.
                - The best HyperParameters object found.
                - The Keras Tuner instance.
                Returns (None, None, self.tuner_) if search encounters issues or
                no best HPs are found.
        """
        tuner_verbose = additional_search_kwargs.pop('tuner_verbose', 1)
        # ------------------------------------------------------------------
        # Rename target‑dict keys *only if* each element’s target component
        # is a Python dict produced by PIHALNet.  For HALNet the target is
        # already a Tensor, so we leave it unchanged.
        # ------------------------------------------------------------------
        def _maybe_rename_targets(tgts):
            # tgts is either a dict of tensors or a single/tuple Tensor
            return (
                rename_dict_keys(
                    tgts,
                    param_to_rename={"subsidence": "subs_pred",
                                     "gwl": "gwl_pred"},
                )
                if isinstance(tgts, dict)
                else tgts
            )
        # STEP 1: If train_data is not None, wrap it so that any target dict
        #          inside gets its keys renamed.  We assume each element of
        #          train_data is (input_dict, target_dict).
        
        if train_data is not None:
            train_data = train_data.map(
                lambda in_dict, tgts: (in_dict, _maybe_rename_targets(tgts)),
                num_parallel_calls=AUTOTUNE,
            )
        # STEP 2: Do the same for validation_data, if provided.
        if validation_data is not None:
            validation_data = validation_data.map(
                lambda in_dict, tgts: (in_dict, _maybe_rename_targets(tgts)),
                num_parallel_calls=AUTOTUNE,
            )
 
        tuner_class_map = {
            'randomsearch': RandomSearch,
            'bayesianoptimization': BayesianOptimization,
            'hyperband': Hyperband
        }
        TunerClass = tuner_class_map[self.tuner_type]
        
        tuner_params = {
            "hypermodel": self,
            "objective": self.objective,
            "executions_per_trial": self.executions_per_trial,
            "directory": self.directory,
            "project_name": self.project_name,
            "seed": self.seed,
            "overwrite": self.overwrite_tuner,
            **self.tuner_kwargs
        }

        if self.tuner_type == 'hyperband':
            tuner_params['max_epochs'] = self.tuner_kwargs.get('max_epochs', epochs)
            tuner_params['factor'] = self.tuner_kwargs.get('factor', 3)
            if 'max_trials' in tuner_params: del tuner_params['max_trials']
        else:
            tuner_params['max_trials'] = self.max_trials

        self.tuner_ = TunerClass(**tuner_params)
        
        vlog(
            f"Starting hyperparameter search with {self.tuner_type.upper()}...",
            verbose=verbose, level=1, logger= self._logger, 
        )
        vlog(f"  Project: {self.project_name} (in {self.directory}/)",
             verbose=verbose, level=2,  logger= self._logger,)
        vlog(f"  Objective: {self.objective}", verbose=verbose, 
             level=2, logger= self._logger,
             )
        vlog(f"  Epochs per trial: {epochs}", verbose=verbose, level=2, 
             logger= self._logger 
             )
 
        search_callbacks = callbacks or []
        if not any(isinstance(cb, EarlyStopping) for cb in search_callbacks):
            # Objective name for monitor:
            monitor_objective = self.objective
            if not isinstance(self.objective, str) and hasattr(self.objective, 'name'):
                monitor_objective = self.objective.name
            
            early_stopping_search = EarlyStopping(
                monitor=str(monitor_objective), # Ensure it's a string
                patience=patience,
                verbose=1 if verbose >=2 else 0, # Keras verbose mapping
                restore_best_weights=True
            )
            search_callbacks.append(early_stopping_search)

            vlog("  Added default EarlyStopping callback for search.",
                 verbose=verbose, level=2, logger= self._logger)

        self.tuner_.search(
            train_data,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=search_callbacks,
            verbose= tuner_verbose, # 1 if verbose >=1 else 0, # Keras tuner verbose
            **additional_search_kwargs
        )
        
        vlog(
            "\nHyperparameter search complete.", 
            verbose=verbose, level=1, logger= self._logger, 
        )
        try:
            self.tuner_.results_summary(num_trials=10)
        except Exception as e:
            logger.warning(f"Could not display Keras Tuner results_summary: {e}")

        try:
            best_hps_list = self.tuner_.get_best_hyperparameters(num_trials=1)
            if not best_hps_list:
                logger.error("Keras Tuner found no best hyperparameters.")
                self.best_hps_ = None; self.best_model_ = None
            else:
                self.best_hps_ = best_hps_list[0]
                vlog("\n--- Best Hyperparameters Found ---", 
                     verbose=verbose, level=1, logger= self._logger,)
                for hp_name, hp_value in self.best_hps_.values.items():
                    vlog(f"  {hp_name}: {hp_value}",
                         verbose=verbose, level=2, logger= self._logger, )
                
                vlog("\nBuilding model with best hyperparameters...",
                     verbose=verbose, level=1, logger= self._logger, )
                try: 
                    self.best_model_ = self.tuner_.hypermodel.build(
                        self.best_hps_)
                except: 
                   self.best_model_ = self.tuner_.get_best_models(
                       num_models=1)[0] # Alternative
                
        except Exception as e:
            logger.error(f"Error retrieving or building best model: {e}")
            self.best_hps_ = None; self.best_model_ = None
        
        self._save_tuning_summary(verbose=verbose)
        
        return self.best_model_, self.best_hps_, self.tuner_

    def _save_tuning_summary(self, verbose: int = 1):
        if self.tuner_ is None or self.best_hps_ is None:
            vlog("No tuner or best HPs found to save summary.",
                 verbose=verbose, level=2, logger= self._logger,)
            return

        summary_data = {
            "project_name": self.project_name,
            "tuner_type": self.tuner_type,
            "objective": self.objective if isinstance(self.objective, str) \
                         else getattr(self.objective, 'name', str(self.objective)),
            "best_hyperparameters": self.best_hps_.values if self.best_hps_ else None,
        }
        try:
            best_trial = self.tuner_.oracle.get_best_trials(1)[0]
            summary_data["best_score"] = best_trial.score
            summary_data["best_trial_id"] = best_trial.trial_id
        except: 
            summary_data["best_score"] = "N/A"

        self.tuning_summary_ = summary_data
        log_file_path = os.path.join(
            self.directory, self.project_name, "tuning_summary.json",
        )
        try:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            with open(log_file_path, "w") as f:
                json.dump(summary_data, f, indent=4, default=str)
            vlog(f"Tuning summary saved to {log_file_path}",
                 verbose=verbose, level=1, logger= self._logger,)
        except Exception as e:
            logger.warning(
                f"Could not save tuning summary log to {log_file_path}: {e}"
            )
            

PINNTunerBase.__doc__=(
    r"""
    Base class for hyperparameter tuning of Physics‐Informed Neural
    Networks (PINNs) like PIHALNet, using Keras Tuner.
    
    This class should be inherited by specific model tuners (e.g.,
    ``PIHALTuner``). The subclass must implement the
    ``build(self, hp)`` method, which defines how the Keras model is
    constructed and compiled with a given set of hyperparameters.
    
    The ``PINNTunerBase`` provides a ``search`` method to orchestrate
    the tuning process.
    
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
    
    Attributes
    ----------
    best_hps_ : dict | None
        Mapping of the best hyper-parameters discovered during tuning.
    best_model_ : tf.keras.Model | None
        Fully trained model achieving the best validation objective.
    tuner_ : keras_tuner.Tuner | None
        Underlying Keras Tuner instance used for trials.
    tuning_log_ : list[dict]
        Chronological list of trial results, ultimately saved to
        ``<directory>/<project_name>_tuning_summary.json``.
    """
    ).format(params=_pinn_tuner_docs)
    
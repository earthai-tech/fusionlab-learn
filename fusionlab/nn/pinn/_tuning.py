# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com> & Gemini AI

"""
Base classes and utilities for hyperparameter tuning of PINN models.
"""
import os
import json
import warnings
from typing import Dict, Any, Optional, List, Tuple, Union

# import numpy as np

from ..._fusionlog import fusionlog 
from ...api.property import BaseClass
from ...utils.generic_utils import vlog 
from ...utils.deps_utils import ensure_pkg
from .. import KERAS_BACKEND, KERAS_DEPS, config  

# Keras Tuner is an optional dependency
HAS_KT = False
try:
    import keras_tuner as kt
    HAS_KT = True
except ImportError:
    # Define dummy kt.HyperModel for type hinting and structure
    # if keras_tuner is not installed. The ensure_pkg decorator
    # will handle the runtime error if it's actually used.
    class HyperModel:
        def __init__(self, *args, **kwargs): pass
        def build(self, hp): raise NotImplementedError()

    class Tuner: # Dummy base for kt.Tuner
        def __init__(self, *args, **kwargs): pass
        def search(self, *args, **kwargs): pass
        def results_summary(self, *args, **kwargs): pass
        def get_best_hyperparameters(self, *args, **kwargs): 
            return [kt.HyperParameters()]
        def get_best_models(self, *args, **kwargs): 
            return [None]

    class RandomSearch(Tuner): pass
    class BayesianOptimization(Tuner): pass
    class Hyperband(Tuner): pass
    class HyperParameters: pass # Dummy
    class Objective: pass # Dummy

    # Assign to kt namespace
    kt = type("KT", (), {
        "HyperModel": HyperModel,
        "Tuner": Tuner,
        "RandomSearch": RandomSearch,
        "BayesianOptimization": BayesianOptimization,
        "Hyperband": Hyperband,
        "HyperParameters": HyperParameters,
        "Objective": Objective
    })() 


if KERAS_BACKEND:
    Model = KERAS_DEPS.Model
    Callback = KERAS_DEPS.Callback
    Dataset = KERAS_DEPS.Dataset 
    Adam = KERAS_DEPS.Adam
    EarlyStopping = KERAS_DEPS.EarlyStopping
    
else:
    class Model: pass # Dummy
    class Callback: pass # Dummy
    class Adam: pass # Dummy
    class EarlyStopping: pass # Dummy


logger = fusionlog().get_fusionlab_logger(__name__)

class PINNTunerBase(kt.HyperModel):
    """
    Base class for hyperparameter tuning of Physics-Informed Neural
    Networks (PINNs) like PIHALNet, using Keras Tuner.

    This class should be inherited by specific model tuners (e.g.,
    `PIHALTuner`). The subclass must implement the `build(self, hp)`
    method, which defines how the Keras model is constructed and
    compiled with a given set of hyperparameters.

    The `PINNTunerBase` provides a `search` method to orchestrate the
    tuning process.

    Parameters
    ----------
    fixed_model_params : Dict[str, Any]
        A dictionary of parameters that are fixed for the model during
        this tuning session (e.g., input/output dimensions,
        forecast_horizon). These are passed to the model's constructor.
    param_space_config : Dict[str, Any], optional
        A dictionary defining the hyperparameter search space. Keys are
        parameter names, and values define how they are sampled by
        Keras Tuner (e.g., using `hp.Choice`, `hp.Int`, `hp.Float`).
        If None, the subclass's `build` method must define the full space.
    objective : Union[str, kt.Objective]
        The objective to optimize (e.g., 'val_loss',
        'val_total_loss', or a custom `kt.Objective` instance).
    max_trials : int
        The maximum number of hyperparameter combinations to try.
    project_name : str
        Name of the Keras Tuner project. Results will be stored in
        `directory/project_name`.
    directory : str, default "pinn_tuner_results"
        Directory to store Keras Tuner results.
    executions_per_trial : int, default 1
        Number of models to build and fit for each trial for robustness.
    tuner_type : str, default 'randomsearch'
        Type of Keras Tuner to use. Options: 'randomsearch',
        'bayesianoptimization', 'hyperband'.
    seed : int, optional
        Random seed for the tuner and hyperparameter sampling for
        reproducibility.
    overwrite_tuner : bool, default True
        Whether to overwrite a previous tuning run with the same
        `project_name` in the same `directory`.
    **tuner_kwargs : Any
        Additional keyword arguments to pass to the specific Keras Tuner
        constructor (e.g., `beta` for BayesianOptimization,
        `hyperband_iterations` for Hyperband).
    """
    @ensure_pkg(
        "keras_tuner",
        extra="'keras_tuner' is required for model tuning.",
        auto_install=config.INSTALL_DEPS, # Assuming config is available
        use_conda=config.USE_CONDA
    )
    def __init__(
        self,
        objective: Union[str, kt.Objective] = 'val_total_loss',
        max_trials: int = 10,
        project_name: str = "PINN_Tuning",
        directory: str = "pinn_tuner_results",
        executions_per_trial: int = 1,
        tuner_type: str = 'randomsearch',
        seed: Optional[int] = None,
        overwrite_tuner: bool = True,
        **tuner_kwargs
    ):
        if not HAS_KT:
            raise ImportError(
                "keras_tuner is not installed. Please run "
                "`pip install keras-tuner` to use this tuning class."
            )
        super().__init__() # kt.HyperModel __init__
        
        self.objective = objective
        self.max_trials = max_trials
        self.project_name = project_name
        self.directory = directory
        self.executions_per_trial = executions_per_trial
        self.tuner_type = self._validate_tuner_type(tuner_type)
        self.seed = seed
        self.overwrite_tuner = overwrite_tuner
        self.tuner_kwargs = tuner_kwargs

        self.best_hps_: Optional[kt.HyperParameters] = None
        self.best_model_: Optional[Model] = None
        self.tuner_: Optional[kt.Tuner] = None
        
        self.tuning_summary_: Dict[str, Any] = {}
        self.fixed_model_params: Dict[str, Any] = {}
        self.param_space_config: Dict[str, Any] = {}
        
        if isinstance(self.objective, str):
            # Default: any metric name containing "loss" is minimized
            direction = "min" if "loss" in self.objective else "max"
            self.objective = kt.Objective(self.objective, direction=direction)


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

    def build(self, hp: kt.HyperParameters) -> Model:
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
        early_stopping_patience: int = 10,
        **additional_search_kwargs
    ) -> Tuple[Optional[Model], Optional[kt.HyperParameters], Optional[kt.Tuner]]:
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
        tuner_class_map = {
            'randomsearch': kt.RandomSearch,
            'bayesianoptimization': kt.BayesianOptimization,
            'hyperband': kt.Hyperband
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
            verbose=verbose, level=1
        )
        vlog(f"  Project: {self.project_name} (in {self.directory}/)",
             verbose=verbose, level=2)
        vlog(f"  Objective: {self.objective}", verbose=verbose, level=2)
        vlog(f"  Epochs per trial: {epochs}", verbose=verbose, level=2)
 
        search_callbacks = callbacks or []
        if not any(isinstance(cb, EarlyStopping) for cb in search_callbacks):
            # Objective name for monitor:
            monitor_objective = self.objective
            if not isinstance(self.objective, str) and hasattr(self.objective, 'name'):
                monitor_objective = self.objective.name
            
            early_stopping_search = EarlyStopping(
                monitor=str(monitor_objective), # Ensure it's a string
                patience=early_stopping_patience,
                verbose=1 if verbose >=2 else 0, # Keras verbose mapping
                restore_best_weights=True
            )
            search_callbacks.append(early_stopping_search)

            vlog("  Added default EarlyStopping callback for search.",
                 verbose=verbose, level=2, )
            
        self.tuner_.search(
            train_data,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=search_callbacks,
            verbose=1 if verbose >=1 else 0, # Keras tuner verbose
            **additional_search_kwargs
        )
        
        vlog(
            "\nHyperparameter search complete.", 
            verbose=verbose, level=1 
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
                     verbose=verbose, level=1,)
                for hp_name, hp_value in self.best_hps_.values.items():
                    vlog(f"  {hp_name}: {hp_value}",
                         verbose=verbose, level=2, )
                
                vlog("\nBuilding model with best hyperparameters...",
                     verbose=verbose, level=1, )
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
                 verbose=verbose, level=2)
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
                 verbose=verbose, level=1, )
        except Exception as e:
            logger.warning(
                f"Could not save tuning summary log to {log_file_path}: {e}"
            )
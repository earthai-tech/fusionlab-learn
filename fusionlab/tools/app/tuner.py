# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

""" Tuner App """

from __future__ import annotations

import time
import json 
import numpy as np
import pandas as pd 

from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple

from ...nn import KERAS_DEPS 
from ...nn.forecast_tuner import HydroTuner
from ...nn.pinn.models import PIHALNet, TransFlowSubsNet 
from ...registry import ManifestRegistry, _update_manifest
from ...utils.generic_utils import ensure_directory_exists
from .config import SubsConfig
from .processing import DataProcessor, SequenceGenerator
from .utils import ( 
    TunerProgressCallback, 
    GuiTunerProgress, 
    StopCheckCallback, 
    safe_model_loader,
    TunerProgressCallback
)

Callback = KERAS_DEPS.Callback
Dataset  = KERAS_DEPS.Dataset
AUTOTUNE = KERAS_DEPS.AUTOTUNE
EarlyStopping = KERAS_DEPS.EarlyStopping 
Model =KERAS_DEPS.Model 

class TunerApp:
    def __init__(
        self,
        cfg: SubsConfig,
        search_space: Optional[Dict[str, Any]] = None,
        *,
        log_callback: Optional[Callable[[str], None]] = None,
        tuner_kwargs: Optional[Dict[str, Any]] = None,
        manifest_path: Optional[str | Path] = None,  
        progress_manager = None,   #  ← NEW
        edited_df: Optional[pd.DataFrame] = None,            #  ← NEW (optional)
    ):
        self.cfg = cfg
        self.log = log_callback or print
        self.search_space = search_space or {}
        self._tuner_kwargs = tuner_kwargs or {}
        # self._trial_info = trial_info 
        
        self._pm         = progress_manager                  
        self._edited_df  = edited_df     
        
        # Nute keras tuner verbose and let the Progress 
        # bar handle it.
        self._tuner_verbose = 0 
                
        # ── Manifest bootstrap ──────────────────────────────────────────
        reg = ManifestRegistry(
            log_callback=self.log, manifest_kind='tuning')

        if manifest_path:
            # user passed an existing one ⇒ import into registry
            self._manifest_path = reg.import_manifest(manifest_path)
        elif cfg.registry_path:
            # config already created one via cfg.to_json()
            self._manifest_path = cfg.registry_path
        else:
            # nothing yet ⇒ create a brand-new run dir & manifest
            run_dir = reg.new_run_dir(city=cfg.city_name,
                                      model=cfg.model_name)
            self._manifest_path = (
                run_dir / reg._manifest_filename
            )
            # prime it with the current cfg
            cfg.to_json(manifest_kind='tuning')  # writes inside run_dir

        self._run_dir = self._manifest_path.parent                 # ← NEW

        # keep references for later
        self._processor: DataProcessor | None = None
        self._seq_gen: SequenceGenerator | None = None
        self._tuner: HydroTuner | None = None
        self._best_model = None
        self._best_hps   = None

    def _tick(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log(f"[{ts}] {msg}")

    def _run(
        self,
        *,
        stop_check: Optional[Callable[[], bool]] = None,
        callbacks: Optional[list[Callback]] = None,
    ):
        """End-to-end hyper-parameter search."""
        self.stop_check = stop_check or (lambda: False)

        self._prepare_progress_manager()
        processed_df = self._preprocess_data()
        train_ds, val_ds = self._generate_sequences(processed_df)
        self._build_fixed_params(train_ds, val_ds)
        
        self._pm.start_step("Tuning…")
        search_callbacks = self._build_callbacks(callbacks)
        self._run_tuner(train_ds, val_ds, search_callbacks)
        self._persist_results()
        # Finish the progress bar after tuning completes
        self._pm.finish_step("Tuning ✓")

        return self._best_model, self._best_hps, self._tuner

    def _prepare_progress_manager(self):
        self._pct = lambda p: self._pm.update(p, 100)
        self._pm.reset()
        #  self.cfg.progress_callback = self._pct
        
    def _preprocess_data(self) -> pd.DataFrame:
        if self.stop_check():
            raise InterruptedError("Tuning cancelled by user.")

        self._pm.start_step("Pre-processing…")
        self._processor = DataProcessor(
            self.cfg, self.log, raw_df=self._edited_df
        )
        df = self._processor.run(stop_check=self.stop_check)
        self._pm.finish_step("Pre-processing ✓")
        return df

    def _generate_sequences(self, processed_df) -> Tuple[Dataset, Dataset]:
        if self.stop_check():
            raise InterruptedError("Aborted during sequence generation.")

        self._pm.start_step("Sequencing…")

        self._seq_gen = SequenceGenerator(self.cfg, self.log)
        train_ds, val_ds = self._seq_gen.run(
            processed_df,
            static_features_encoded=getattr(
                self._processor, "static_features_encoded", []
            ),
            stop_check=self.stop_check,
        )
        self._pm.finish_step("Sequencing ✓")
        return train_ds, val_ds

    def _build_fixed_params(self, train_ds, val_ds):
        inputs = self._seq_gen.inputs_train
        targets = self._seq_gen.targets_train
        self._fixed_params = {
            "static_input_dim": inputs.get(
                "static_features", np.zeros((0, 0))
            ).shape[-1],
            "dynamic_input_dim": inputs["dynamic_features"].shape[-1],
            "future_input_dim": inputs.get(
                "future_features", np.zeros((0, 0, 0))
            ).shape[-1],
            "output_subsidence_dim": targets["subsidence"].shape[-1],
            "output_gwl_dim": targets["gwl"].shape[-1],
            "forecast_horizon": targets["subsidence"].shape[1],
            "quantiles": self.cfg.quantiles,
            "pde_mode": self.cfg.pde_mode,
            "pinn_coefficient_C": self.cfg.pinn_coeff_c,
            "lambda_cons": self.cfg.lambda_cons,
            "lambda_gw": self.cfg.lambda_gw,
        }

        _update_manifest(
            self._run_dir,
            section="tuner",
            item={
                "search_space": self.search_space,
                "fixed_params": self._fixed_params,
                "tuner_kwargs": self._tuner_kwargs,
            },
            manifest_kind="tuning",
        )

        self._tuner = HydroTuner(
            model_name_or_cls=self.cfg.model_name,
            fixed_params=self._fixed_params,
            search_space=self.search_space,
            _logger=self._tick,
            **self._tuner_kwargs,
        )

        self._train_tf = train_ds.prefetch(AUTOTUNE)
        self._val_tf = val_ds.prefetch(AUTOTUNE)

    def _build_callbacks(
            self, extra: Optional[list[Callback]]) -> list[Callback]:
    
        patience = self._tuner_kwargs.pop("patience", 8)
        self._num_cpus = self._tuner_kwargs.pop ('num_cpus', 1)
        
        self._tuner = HydroTuner(
            model_name_or_cls=self.cfg.model_name,
            fixed_params=self._fixed_params,
            search_space=self.search_space,
            _logger=self._tick,
            **self._tuner_kwargs,
        )
        
        cb = list(extra or [])

        stop_cb = StopCheckCallback(
            stop_check_fn=self.stop_check, log_callback=self.log
        )
        cb.append(stop_cb)

        if not any(isinstance(x, EarlyStopping) for x in cb):
            patience = self._tuner_kwargs.pop("patience", 8)
            objective = (
                self._tuner.objective
                if isinstance(self._tuner.objective, str)
                else getattr(self._tuner.objective, "name", str(self._tuner.objective))
            )
            cb.append(
                EarlyStopping(
                    monitor=objective,
                    patience=patience,
                    restore_best_weights=True,
                    # verbose=self.cfg.verbose,
                )
            )

        max_trials = self._tuner_kwargs.get("max_trials", 10)
        
        # Custom callback for progress tracking during the tuning phase
        def _pct_tuning(percent: int) -> None:
            # Regular progress bar update
            self._pm.update(percent, 100)
            
            # Derive trial and epoch indices from the percentage
            trial = int(percent // (100 / max_trials)) + 1
            epoch = int((percent % (100 // max_trials)) / (
                100 // (max_trials * self.cfg.epochs))) + 1
            
            # Set the context for trial and epoch in the progress manager
            self._pm.set_trial_context(trial=trial, total=max_trials)
            self._pm.set_epoch_context(epoch=epoch, total=self.cfg.epochs)
    
        # Assign the custom progress callback to the `tuner`'s search process
        self.cfg.progress_callback = _pct_tuning
        
        # If a progress callback is provided, link it to the update function for tuning
        if callable(getattr(self.cfg, "progress_callback", None)):
            # Create and configure the GuiTunerProgress callback
            tuner_progress_callback = GuiTunerProgress(
                total_trials=max_trials,
                total_epochs=self.cfg.epochs,
                update_fn=self.cfg.progress_callback,  # Connect progress callback
                epoch_level=True,  # Set this to False if you want per-trial updates
            )
            
            cb.append(tuner_progress_callback)
        else: 
            self._tuner_verbose = self.cfg.verbose 
            
        # cb.append(
        #     TunerProgressCallback(
        #         progress_manager=self._pm,
        #         total_trials=max_trials,
        #         total_epochs=self.cfg.epochs,
        #         log=self.log,  # Custom log callback
        #         trial_batch_level=True, 
        #     )
        # )
        return cb
    
    def _run_tuner(self, train_ds, val_ds, callbacks):
        """Runs the hyperparameter tuning process with live 
        *Trial x/N* and *Epoch x/N* prefixes."""
        if self.stop_check():
            raise InterruptedError("Model-parameter setting cancelled.")
   
        # Run the tuner with the customized callback
        self._best_model, self._best_hps, self._tuner = self._tuner.search(
            train_data=self._train_tf,
            validation_data=self._val_tf,
            epochs=self.cfg.epochs,
            callbacks=callbacks,
            verbose = self.cfg.verbose, 
            tuner_verbose=self._tuner_verbose,
        )
    
    def _persist_results(self):
        out_dir = self.config.run_output_path # / "tuner_results"
        ensure_directory_exists(out_dir)

        (out_dir / "best_hyperparameters.json").write_text(
            json.dumps(self._best_hps.values, indent=2)
        )

        fmt = (self.cfg.save_format or "keras").lower()
        if fmt == "tf":
            best_path = out_dir / f"{self.cfg.model_name}_best"
            self._best_model.save(best_path, save_format="tf")
        elif fmt == "weights":
            best_path = out_dir / f"{self.cfg.model_name}.best.weights.h5"
            self._best_model.save_weights(best_path)
        else:
            best_path = out_dir / f"{self.cfg.model_name}_best.keras"
            self._best_model.save(best_path)

        _update_manifest(
            run_dir=self._run_dir,
            section="tuner_results",
            item={
                "best_hyperparameters": self._best_hps.values,
                "best_model": best_path.name,
                "save_format": fmt,
            },
            manifest_kind="tuning",
        )
        self._tick(f"Search complete → {best_path}")
    

    def run(
        self,
        *,
        stop_check: Optional[Callable[[], bool]] = None,
        callbacks: Optional[list[Callback]] = None,
    ):
        """End-to-end hyper-parameter search."""
        self.stop_check = stop_check or (lambda: False)

        self._prepare_progress_manager()
        processed_df = self._preprocess_data()
        train_ds, val_ds = self._generate_sequences(processed_df)
        
        self._pm.start_step("Tuning…")
        self._build_fixed_params(train_ds, val_ds)
        search_callbacks = self._build_callbacks(callbacks)
        self._run_tuner(train_ds, val_ds, search_callbacks)
        self._persist_results()
        
        self._pm.finish_step("Tuning ✓")

        return self._best_model, self._best_hps, self._tuner

    @staticmethod
    def build_model_from_manifest(
        manifest_path: str | Path,
        *,
        custom_objects: Optional[dict] = None,
        log: Callable[[str], None] = print,
    ):
        """
        Convenience loader for **inference**.

        Given a *tuner_run_manifest.json* (or its directory), rebuilds the
        tuned model and loads its weights / full file – so callers don’t
        need to duplicate the model-reconstruction logic.
        """
        manifest_path = Path(manifest_path)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "tuner_run_manifest.json"

        data = json.loads(manifest_path.read_text("utf-8"))
        cfg_sec   = data.get("tuner", {})          # fixed + user cfg
        result_sec= data.get("tuner_results", {})

        fixed_params   = cfg_sec.get("fixed_params", {})
        best_hps_vals  = result_sec.get("best_hyperparameters", {})
        save_fmt       = result_sec.get("save_format", "keras")
        best_model_rel = result_sec.get("best_model")
        input_shapes   = result_sec.get("input_shapes")

        run_dir  = manifest_path.parent
        model_fp = run_dir / "tuner_results" / best_model_rel

        # 1. merge fixed params + best HPs
        merged = fixed_params.copy()
        merged.update(best_hps_vals)

        # 2. decide model class
        model_cls = TransFlowSubsNet if merged.get("pde_mode", "both") == "both" \
                    else PIHALNet

        # 3. reconstruction helper
        def _build() -> Model:
            return model_cls(**merged)

        # 4. load with universal helper
        model = safe_model_loader(
            model_fp,
            build_fn=_build if save_fmt == "weights" else None,
            custom_objects=custom_objects,
            log=log,
            model_cfg={"config": _build().get_config(),
                       "input_shapes": input_shapes} if save_fmt == "weights" else {},
        )
        return model

    # properties -------------------------------------------------------------
    @property
    def tuner(self):      return self._tuner
    @property
    def best_model(self): return self._best_model
    @property
    def best_hps(self):   return self._best_hps

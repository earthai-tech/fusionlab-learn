# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

""" Tuner App """

from __future__ import annotations

import time
import json 
import numpy as np

from pathlib import Path
from typing import Dict, Any, Optional, Callable

from fusionlab.nn import KERAS_DEPS 
from fusionlab.nn.forecast_tuner import HydroTuner
from fusionlab.nn.pinn.models import PIHALNet, TransFlowSubsNet 
from fusionlab.registry import ManifestRegistry, _update_manifest
from fusionlab.tools.app.config import SubsConfig
from fusionlab.tools.app.processing import DataProcessor, SequenceGenerator

from fusionlab.utils.generic_utils import ensure_directory_exists
from fusionlab.tools.app.utils import safe_model_loader

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
        manifest_path: Optional[str | Path] = None,                    # ← NEW
    ):
        self.cfg = cfg
        self.log = log_callback or print
        self.search_space = search_space or {}
        self._tuner_kwargs = tuner_kwargs or {}

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

    def run(
        self,
        *,
        stop_check: Optional[Callable[[], bool]] = None,
        callbacks: Optional[list[Callback]] = None,
    ):
    
        # 1 ▸ Data processing ------------------------------------------------
        self._processor = DataProcessor(self.cfg, self.log)
        processed_df = self._processor.run(stop_check=stop_check)
        if stop_check and stop_check():
            raise InterruptedError("Aborted during preprocessing")

        # 2 ▸ Sequence generation -------------------------------------------
        self._seq_gen = SequenceGenerator(self.cfg, self.log)
        train_ds, val_ds = self._seq_gen.run(
            processed_df,
            static_features_encoded=getattr(
                self._processor, "static_features_encoded", []
            ),
            stop_check=stop_check,
        )
        if stop_check and stop_check():
            raise InterruptedError("Aborted during sequence generation")

        # 3 ▸ Build fixed-param dict for HydroTuner --------------------------
        inputs  = self._seq_gen.inputs_train
        targets = self._seq_gen.targets_train

        fixed_params = {
            "static_input_dim": inputs.get(
                "static_features", np.zeros((0, 0))).shape[-1],
            "dynamic_input_dim": inputs["dynamic_features"].shape[-1],
            "future_input_dim": inputs.get(
                "future_features", np.zeros((0, 0, 0))).shape[-1],
            "output_subsidence_dim": targets["subsidence"].shape[-1],
            "output_gwl_dim": targets["gwl"].shape[-1],
            "forecast_horizon": targets["subsidence"].shape[1],
            "quantiles": self.cfg.quantiles,
            # cfg-driven physics flags
            "pde_mode": self.cfg.pde_mode,
            "pinn_coefficient_C": self.cfg.pinn_coeff_c,
            "lambda_cons": self.cfg.lambda_cons,
            "lambda_gw":  self.cfg.lambda_gw,
        }

        # initial manifest update – store search config --
        _update_manifest(
            self._run_dir,
            section="tuner",
            item={
                "search_space": self.search_space,
                "fixed_params": fixed_params,
                "tuner_kwargs": self._tuner_kwargs,
            },
            manifest_kind='tuning', 
        )
        patience = self._tuner_kwargs.pop("patience", 8)
        self._num_cpus = self._tuner_kwargs.pop ('num_cpus', 1)
        # 4 ▸ Instantiate HydroTuner ---------------------
        self._tuner = HydroTuner(
            model_name_or_cls=self.cfg.model_name,
            fixed_params=fixed_params,
            search_space=self.search_space,
            **self._tuner_kwargs,
        )

        # 5 ▸ Datasets for tuner.search() ----------------
        train_tf = train_ds.prefetch(AUTOTUNE)
        val_tf   = val_ds.prefetch(AUTOTUNE)

        # 6 ▸ Run search --------------------------------
        monitor_name = (
            self._tuner.objective
            if isinstance(self._tuner.objective, str)
            else getattr(self._tuner.objective, "name", str(self._tuner.objective))
        )
                
        early_stop = EarlyStopping(
            monitor=str(monitor_name), 
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
        
        self._tick("TunerApp ▸ hyper-parameter search …")

        self._tuner.search(
            train_data=train_tf,
            validation_data=val_tf,
            epochs=self.cfg.epochs,
            callbacks=[early_stop, *(callbacks or [])],
            verbose=self.cfg.fit_verbose,
        )

        # 7 ▸ Persist results ------------------------------------------------
        self._best_hps   = self._tuner.get_best_hyperparameters(1)[0]
        self._best_model = self._tuner.get_best_models(1)[0]

        out_dir = self._run_dir / "tuner_results"
        ensure_directory_exists(out_dir)

        hps_path = out_dir / "best_hyperparameters.json"
        hps_path.write_text(json.dumps(self._best_hps.values, indent=2))

        # ── save best artefacts using user-selected `cfg.save_format` ─
        save_fmt = (self.cfg.save_format or "keras").lower()
        out_dir  = self._run_dir / "tuner_results"
        ensure_directory_exists(out_dir)

        if save_fmt == "tf":
            model_path = out_dir / f"{self.cfg.model_name}_best"
            self._best_model.save(model_path, save_format="tf")
        elif save_fmt == "weights":
            model_path = out_dir / f"{self.cfg.model_name}.best.weights.h5"
            self._best_model.save_weights(model_path)
        else:                                              # default “keras”
            model_path = out_dir / f"{self.cfg.model_name}_best.keras"
            self._best_model.save(model_path)

        # when weights-only, preserve input-shape dictionary for reload
        if save_fmt == "weights":
            input_shapes = {
                "coords":            list(self._seq_gen.inputs_train["coords"].shape),
                "static_features":   list(self._seq_gen.inputs_train
                                          .get("static_features",
                                               np.zeros((0, 0))).shape),
                "dynamic_features":  list(self._seq_gen.inputs_train
                                          ["dynamic_features"].shape),
                "future_features":   list(self._seq_gen.inputs_train
                                          .get("future_features",
                                               np.zeros((0, 0, 0))).shape),
            }
        else:
            input_shapes = None
        
        # update manifest with results & artefacts ----------
        _update_manifest(
            run_dir=self._run_dir,
            section="tuner_results",
            item={
                "best_hyperparameters": self._best_hps.values,
                "best_model": model_path.name,
                "save_format": save_fmt,
                "input_shapes": input_shapes,
            },
            manifest_kind="tuning",
        )

        self._tick(f"Search complete → {model_path}")
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
    
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

""" Tuner App """

from __future__ import annotations

import time
import json 
import numpy as np

from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple, List

from ...core.handlers import _get_valid_kwargs 
from ...nn import KERAS_DEPS 
from ...nn.forecast_tuner import HydroTuner
from ...nn.pinn.models import PIHALNet, TransFlowSubsNet 
from ...nn.pinn.op import extract_physical_parameters 
from ...registry import  _update_manifest, _resolve_manifest 
from ...utils.generic_utils import ensure_directory_exists
from .config import SubsConfig
from .utils import ( 
    StopCheckCallback, 
    safe_model_loader,
    TunerProgressCallback
)

Callback = KERAS_DEPS.Callback
Dataset  = KERAS_DEPS.Dataset
AUTOTUNE = KERAS_DEPS.AUTOTUNE
EarlyStopping = KERAS_DEPS.EarlyStopping 
Model =KERAS_DEPS.Model 
experimental =KERAS_DEPS.experimental 


class TunerApp:
    """
    Thin wrapper around :class:`~fusionlab.nn.forecast_tuner.HydroTuner`.

    * It **does not** prepare data – datasets are injected via
      :meth:`set_data_context`.
    * It orchestrates *fixed-param extraction → search → persistence* while
      driving a :class:`ProgressManager` for the GUI.
    """

    def __init__(
        self,
        cfg: SubsConfig,                                        # SubsConfig
        search_space: Optional[Dict[str, Any]] = None,
        *,
        log_callback:   Optional[Callable[[str], None]] = None,
        tuner_kwargs:   Optional[Dict[str, Any]] = None,
        manifest_path:  Optional[str | Path] = None,
        progress_manager=None,
    ):
        # user-supplied & bookkeeping
        self.cfg            = cfg
        self.log            = log_callback or print
        self.search_space   = search_space or {}
        self._tuner_kwargs  = tuner_kwargs or {}
        self._pm            = progress_manager        # may be None in CLI
        self._tuner_verbose = 0

        # data placeholders – filled by set_data_context()
        self._processor = None
        self._seq_gen   = None
        self._train_tf  = None
        self._val_tf    = None

        # outcome placeholders
        self._tuner       = None
        self._best_model  = None
        self._best_hps    = None
        self._fixed_params: Dict[str, Any] = {}
        
        self._manifest_path = _resolve_manifest(
            cfg            = cfg,
            manifest_path  = manifest_path,
            log_cb         = self.log, 
            manifest_kind="tuning"
        )
        self._run_dir = self._manifest_path.parent

    def _tick(self, msg: str):
        self.log(f"[{time.strftime('%H:%M:%S')}] {msg}")

    @property
    def stop_check(self) -> Callable[[], bool]:
        # set by execute() – defaults to “never stop”
        return getattr(self, "_stop_check", lambda: False)

    def set_data_context(
        self,
        *,
        processor,             # DataProcessor
        seq_gen,               # SequenceGenerator
        train_ds: Dataset,
        val_ds:   Dataset,
    ):
        """Inject ready datasets & helpers prepared by the caller."""
        self._processor = processor
        self._seq_gen   = seq_gen
        self._train_tf  = train_ds.prefetch(AUTOTUNE)
        self._val_tf    = val_ds.prefetch(AUTOTUNE)

    def execute(
        self,
        *,
        stop_check: Callable[[], bool] = lambda: False,
        extra_callbacks: Optional[List[Callback]] = None,
    ) -> Tuple[Model, Dict[str, Any], HydroTuner]:
        """
        Main entry point: build fixed params → search → persist results.

        Returns ``(best_model, best_hyperparameters, tuner)``.
        """
        self._pm.start_step("Preparing")
        # guard: data must have been injected
        if self._train_tf is None or self._val_tf is None:
            raise RuntimeError("Data context not set – "
                               "call set_data_context() first.")

        self._stop_check = stop_check
        self._prepare_progress_manager()
        
        self._pm.finish_step("Preparing")

        self._build_fixed_params()
        callbacks = self._build_callbacks(extra_callbacks)
        self._run_tuner(callbacks)
        self._persist_results()

        return self._best_model, self._best_hps, self._tuner


    def _prepare_progress_manager(self):
        if not self._pm:
            return
        # self._pct = lambda p: self._pm.update(p, 100)
  
    def _build_fixed_params(self):
        if self.stop_check():
            raise InterruptedError("Tuning cancelled by user.")

        if self._pm:
            self._pm.start_step("Building")

        # Take one batch to infer shapes
        sample_inputs, _ = next(iter(self._train_tf))
        if isinstance(sample_inputs, (list, tuple)):
            keys = list(self._seq_gen.inputs_train.keys())
            sample_inputs = {k: t for k, t in zip(keys, sample_inputs)}

        input_shapes = {k: list(v.shape) for k, v in sample_inputs.items()}

        inputs  = self._seq_gen.inputs_train
        targets = self._seq_gen.targets_train
        self._fixed_params = dict(
            static_input_dim   = inputs.get("static_features",
                                            np.zeros((0, 0))).shape[-1],
            dynamic_input_dim  = inputs["dynamic_features"].shape[-1],
            future_input_dim   = inputs.get("future_features",
                                            np.zeros((0, 0, 0))).shape[-1],
            output_subsidence_dim = targets["subsidence"].shape[-1],
            output_gwl_dim        = targets["gwl"].shape[-1],
            forecast_horizon      = targets["subsidence"].shape[1],
            quantiles             = self.cfg.quantiles,
            pde_mode              = self.cfg.pde_mode,
            pinn_coefficient_C    = self.cfg.pinn_coeff_c,
            lambda_cons           = self.cfg.lambda_cons,
            lambda_gw             = self.cfg.lambda_gw,
        )

        _update_manifest(
            self._run_dir,
            section="tuner",
            item=dict(
                search_space = self.search_space,
                fixed_params = self._fixed_params,
                tuner_kwargs = self._tuner_kwargs,
                input_shapes = input_shapes,
            ),
            manifest_kind="tuning",
        )

    def _build_callbacks(
        self,
        extra: Optional[List[Callback]],
    ) -> List[Callback]:
        if self.stop_check():
            raise InterruptedError("Callback building cancelled.")

        patience = self._tuner_kwargs.pop("patience", 8)
        self._num_cpus = self._tuner_kwargs.pop ('num_cpus', 1)

        self._tuner = HydroTuner(
            model_name_or_cls = self.cfg.model_name,
            fixed_params      = self._fixed_params,
            search_space      = self.search_space,
            directory         = self.cfg.run_output_path,
            _logger           = self._tick,
            **self._tuner_kwargs,
        )

        cb: List[Callback] = list(extra or [])

        cb.append(
            StopCheckCallback(
                stop_check_fn=self.stop_check,
                log_callback = self.log,
            )
        )

        if not any(isinstance(x, EarlyStopping) for x in cb):
            monitor = (self._tuner.objective
                       if isinstance(self._tuner.objective, str)
                       else getattr(self._tuner.objective, "name",
                                    str(self._tuner.objective)))
            cb.append(
                EarlyStopping(
                    monitor             = monitor,
                    patience            = patience,
                    restore_best_weights=True,
                    verbose             = self._tuner_verbose,
                )
            )

        # Progress-bar callback 
        if self._pm and callable(getattr(self.cfg, "progress_callback", None)):
            bpe = experimental.cardinality(self._train_tf).numpy()
            cb.append(
                TunerProgressCallback(
                    total_trials      = self._tuner_kwargs.get("max_trials", 10),
                    total_epochs      = self.cfg.epochs,
                    batches_per_epoch = bpe,
                    progress_manager  = self._pm,
                    epoch_level       = True,
                    batch_level = True,
                    log               = self.log,
                )
            )

        if self._pm:
            self._pm.finish_step("Build")
            self._pm.reset("0 %")
        return cb

    def _run_tuner(self, callbacks: List[Callback]):
        if self.stop_check():
            raise InterruptedError("Parameter search cancelled.")

        if self._pm:
            self._pm.start_step("Tuning")
        try:
            self._best_model, self._best_hps, self._tuner = (
                self._tuner.search(
                    train_data     = self._train_tf,
                    validation_data= self._val_tf,
                    epochs         = self.cfg.epochs,
                    callbacks      = callbacks,
                    verbose        = self.cfg.verbose,
                    tuner_verbose  = self._tuner_verbose,
                )
            )
        finally:
            if self._pm:
                self._pm.finish_step("Tuning")

  
    def _persist_results(self):
        if self._pm:
            self._pm.reset()
            self._pm.start_step("Persisting")

        out_dir = Path(self.cfg.run_output_path)
        ensure_directory_exists(out_dir)

        # Try building from manifest shapes first 
        try:
            manifest = json.loads(self._manifest_path.read_text())
            shapes = manifest["tuner"].get("input_shapes", {})
            self._best_model.build(shapes)
        except Exception as e:
            self.log(
                    f"[Warning] Failed to build model from manifest shapes "
                    f"(falling back to sample-based build): {e}"
                )
            # ── Fallback: run a single batch through the model 
            try:
                sample_inputs, _ = next(iter(self._train_tf))
                if isinstance(sample_inputs, (tuple, list)):
                    keys = list(self._seq_gen.inputs_train.keys())
                    sample_inputs = {
                        k: t for k, t in zip(keys, sample_inputs)
                    }
                _ = self._best_model(sample_inputs, training=False)
            except Exception as ee:
                self.log(
                    f"[Warning] Sample-based build also failed; model "
                    f"may not be fully built: {ee}"
                )
                
        # hyper-params
        (out_dir / "best_hyperparameters.json").write_text(
            json.dumps(self._best_hps.values, indent=2)
        )

        # model save
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

        # manifest update
        _update_manifest(
            run_dir       = self._run_dir,
            section       = "tuner_results",
            item          = dict(
                best_hyperparameters = self._best_hps.values,
                best_model           = best_path.name,
                save_format          = fmt,
            ),
            manifest_kind = "tuning",
        )

        # physical parameters
        try:
            phys_path = out_dir / f"{self.cfg.model_name}.tune.physical_parameters.csv"
            extract_physical_parameters(self._best_model, filename=phys_path)
        except Exception as err:
            self.log(f"[WARN] Could not export physical parameters: {err}")

        self._tick(f"Search complete → {best_path}")

        if self._pm:
            self._pm.finish_step("Save")

    @staticmethod
    def build_model_from_manifest(
        manifest_path: str | Path,
        *,
        custom_objects: Optional[dict] = None,
        log: Callable[[str], None] = print,
    ) -> Model:
        """Reconstruct tuned model (for inference) from a manifest."""
        manifest_path = Path(manifest_path)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "tuner_run_manifest.json"

        data         = json.loads(manifest_path.read_text("utf-8"))
        config       = data.get('configuration', {})
        cfg_sec      = data.get("tuner", {})
        result_sec   = data.get("tuner_results", {})

        fixed_params = cfg_sec.get("fixed_params", {})
        best_hps     = result_sec.get("best_hyperparameters", {})
        save_fmt     = result_sec.get("save_format", "keras")
        model_rel    = result_sec.get("best_model")
        input_shapes = result_sec.get("input_shapes")

        run_dir  = Path(config["run_output_path"]).resolve()
        model_fp = run_dir / model_rel

        merged = {**fixed_params, **best_hps}
        model_cls = TransFlowSubsNet if merged.get(
            "pde_mode", "both") == "both" else PIHALNet
        
        valid_kws  = _get_valid_kwargs(
            model_cls, merged, error="ignore")

        def _build():
            return model_cls(**valid_kws)

        return safe_model_loader(
            model_fp,
            build_fn=_build if save_fmt == "weights" else None,
            custom_objects=custom_objects,
            log=log,
            model_cfg=dict(
                config       = _build().get_config(),
                input_shapes = input_shapes,
            ) if save_fmt == "weights" else {},
        )

    @property
    def tuner(self):      return self._tuner
    @property
    def best_model(self): return self._best_model
    @property
    def best_hps(self):   return self._best_hps

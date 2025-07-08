# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Provides high-level utility functions for the GUI application,
such as a robust model loader that can handle multiple Keras/TF
saving formats.
"""
from __future__ import annotations 

import os
import time 
from numbers import Integral, Real
import json
from typing import Callable, Optional, Any, Dict, List 
import subprocess, sys, platform
from pathlib import Path
import warnings 

try:
    import ast                                
    _AST_AVAILABLE = True
except Exception:                            
    _AST_AVAILABLE = False

try:
    from fusionlab.params import (
        LearnableK, LearnableSs, LearnableQ,
        LearnableC, FixedC, DisabledC,
    )
    from fusionlab.nn import KERAS_DEPS
    from fusionlab.nn.models import TransFlowSubsNet, PIHALNet
except ImportError as e:
    raise ImportError(
        "This utility requires the `fusionlab` library and its"
        f" dependencies to be installed. Error: {e}"
    )

_ALLOWED_KEYS = {
    "min_value", "max_value", "step",          
    "sampling",                                
}

# Define custom objects needed for model deserialization
_CUSTOM_OBJECTS = {
    "LearnableK": LearnableK, "LearnableSs": LearnableSs,
    "LearnableQ": LearnableQ, "LearnableC":  LearnableC,
    "FixedC":     FixedC,     "DisabledC":   DisabledC,
}

_LEARNABLE_TYPES = (LearnableK, LearnableSs, LearnableQ)

Callback =KERAS_DEPS.Callback 
load_model = KERAS_DEPS.load_model
Model = KERAS_DEPS.Model            # type alias
custom_object_scope = KERAS_DEPS.custom_object_scope
deserialize_keras_object = KERAS_DEPS.deserialize_keras_object
serialize_keras_object =KERAS_DEPS.serialize_keras_object

 
class StopCheckCallback(Callback):
    """
    A Keras callback to gracefully interrupt training *or* tuning.

    If `stop_check_fn()` returns True, we:
      1. set model.stop_training = True   (stops the current .fit())
      2. raise InterruptedError          (bubbles out of tuner.search())
    """
    def __init__(
        self,
        stop_check_fn: Callable[[], bool],
        log_callback: Optional[Callable[[str], None]] = None
    ):
        super().__init__()
        self.stop_check = stop_check_fn
        self.log = log_callback or print
        self.was_stopped = False

    def _maybe_stop(self):
        if self.stop_check is not None and self.stop_check():
            if not self.was_stopped:
                self.log("  [Callback] Interruption requested. Halting process…")
                self.was_stopped = True
            # 1) stop the current fit loop
            self.model.stop_training = True
            # 2) abort tuner.search() as well
            # raise InterruptedError("Stop requested by user")

    def on_epoch_end(self, epoch, logs=None):
        self._maybe_stop()

    def on_train_batch_end(self, batch, logs=None):
        self._maybe_stop()

    def on_test_batch_end(self, batch, logs=None):
        self._maybe_stop()

    def on_test_end(self, logs=None):
        self._maybe_stop()

    def on_train_end(self, logs=None):
        # final check in case it was very fast
        self._maybe_stop()

    def __deepcopy__(self, memo):
        # KerasTuner deep-copies callbacks per trial; signals/threads
        # cannot be deep-copied, so return self.
        return self

class TunerProgressCallback(Callback):
    """
    Drive one ProgressManager bar across:
      total_trials × total_epochs × batches_per_epoch
    
    We detect each new trial via `on_train_begin`, since
    Keras-Tuner runs one model.fit() per trial.
    """

    def __init__(
        self,
        total_trials: int,
        total_epochs: int,
        batches_per_epoch: int,
        progress_manager,
        *,
        epoch_level: bool = True,
        batch_level: bool = False,
        log: Callable[[str], None] = print,
    ):
        super().__init__()

        if total_trials <= 0 or total_epochs <= 0 or batches_per_epoch <= 0:
            raise ValueError(
                "total_trials, total_epochs and"
                " batches_per_epoch must all be > 0")

        self.total_trials       = total_trials
        self.total_epochs       = total_epochs
        self.batches_per_epoch  = batches_per_epoch
        self.pm                 = progress_manager
        self.epoch_level        = epoch_level
        self.batch_level        = batch_level
        self.log                = log

        # state:
        self.current_trial = 0            # 1-based
        self.global_batch  = 0            # 0-based count of batches done
        self.global_total  = (
            total_trials * total_epochs * batches_per_epoch
        )
            
    def on_train_begin(self, logs=None):
        """
        Called once at the start of each trial's model.fit().
        We increment the trial count here.
        """
        self.current_trial += 1

        # reset the ETA timer & bar style
        self.pm.start_step("Tuning")

        # position the global counter at the start of this trial
        self.global_batch = (self.current_trial - 1) * (
            self.total_epochs * self.batches_per_epoch
        )

        # update the prefix
        self.pm.set_trial_context(
            trial=self.current_trial,
            total=self.total_trials
        )
        self.log(f"Trial {self.current_trial}/{self.total_trials} started...")

    def on_epoch_end(self, epoch, logs=None):
        if not self.epoch_level:
            return

        # jump the global counter to the end of this epoch
        self.global_batch = (
            (self.current_trial - 1) * self.total_epochs
            + (epoch + 1)
        ) * self.batches_per_epoch

        # update the epoch prefix
        self.pm.set_epoch_context(
            epoch=epoch + 1,
            total=self.total_epochs
        )
        self._update_bar()

    def on_train_batch_end(self, batch, logs=None):
        if not self.batch_level:
            return

        # advance one batch
        self.global_batch += 1

        # recompute which epoch we're in
        epoch = (
            self.global_batch // self.batches_per_epoch
        ) - (self.current_trial - 1) * self.total_epochs

        self.pm.set_epoch_context(
            epoch=epoch + 1,
            total=self.total_epochs
        )
        self._update_bar()

    def on_train_end(self, logs=None):
        """
        Called at the end of each trial's model.fit().
        If this was the last trial, snap to 100% and finish.
        Otherwise, just log "trial done" and leave the bar at
        its current position ready for the next `on_train_begin`.
        """
        if self.current_trial < self.total_trials:
            self._update_bar()
            self.log(f"Trial {self.current_trial}/{self.total_trials} completed.")
        else:
            # final finish
            self.pm.update(current=self.global_total, total=self.global_total)
            self.pm.finish_step("Tuning")
            self.log("Hyperparameter tuning completed!")

    def _update_bar(self):
        pct = (self.global_batch / self.global_total) * 100
        self.pm.update(current=self.global_batch, total=self.global_total)
        self.log(
            f"Trial {self.current_trial}/{self.total_trials} – "
            f"Global batch {self.global_batch} – "
            f"Progress: {pct:.2f}%"
        )

    def __deepcopy__(self, memo):
        # Keras-Tuner deep-copies callbacks per trial —
        # Qt signals can't be copied, so just return self.
        return self

class PerTrialTunerProgress(Callback):
    def __init__(
        self,
        total_trials: int,
        total_epochs: int,
        progress_manager, 
        *,
        log: Callable[[str], None] = print,
    ):
        super().__init__()
        self.total_trials     = total_trials
        self.total_epochs     = total_epochs
        self.pm               = progress_manager
        self.log              = log
        self.trial_idx        = 0

    def on_trial_begin(self, trial):
        # new trial: reset the bar for this trial’s epochs
        self.trial_idx = trial.trial_id
        self.pm.start_step(
            name=f"Trial {self.trial_idx+1}/{self.total_trials}",
            total=self.total_epochs,
        )
        self.pm.set_trial_context(
            trial=self.trial_idx+1,
            total=self.total_trials,
        )
        self.log(f"Trial {self.trial_idx+1}/"
                 f"{self.total_trials} started.")

    def on_epoch_end(self, epoch, logs=None):
        # epoch is 0-based → convert to 1-based
        ep = epoch + 1
        # update the bar for this trial
        self.pm.update(current=ep, total=self.total_epochs)
        self.pm.set_epoch_context(epoch=ep, total=self.total_epochs)
        pct = ep / self.total_epochs * 100
        self.log(
            f"Trial {self.trial_idx+1}/{self.total_trials} - "
            f"Epoch {ep}/{self.total_epochs} - "
            f"Progress: {pct:.2f}%"
        )

    def on_trial_end(self, trial, logs=None):
        # ensure we hit 100% and finish this trial
        self.pm.finish_step(f"Trial {self.trial_idx+1} done")
        self.log(f"Trial {self.trial_idx+1}/{self.total_trials} finished.")

    def on_train_end(self, logs=None):
        # after all trials
        self.log("Hyperparameter tuning completed!")
        
    def __deepcopy__(self, memo):
        return self

class UnifiedTunerProgress(TunerProgressCallback):
    def on_batch_end(self, batch, logs=None):
        super().on_batch_end(batch, logs)
        # optionally also update trial context here

    def on_epoch_end(self, epoch, logs=None):
        # compute a *global* percentage:
        global_step = (self.trial_idx * self.total_epochs) + (epoch + 1)
        global_total = self.total_trials * self.total_epochs
        pct = int(global_step / global_total * 100)
        self.progress_manager.update(current=global_step, total=global_total)
        self.progress_manager.set_trial_context(trial=self.trial_idx+1,
                                                total=self.total_trials)
        self.progress_manager.set_epoch_context(epoch=epoch+1,
                                                total=self.total_epochs)
        self.log(f"[Global] {pct}%")

class _GuiProgress(Callback):
    """
    Emit percentage updates while `model.fit` runs.

    Parameters
    ----------
    total_epochs : int
        Epochs you pass to `model.fit`.
    batches_per_epoch : int
        Length of the training dataset (`len(ds)`).
        Needed only for *batch-level* granularity.
    update_fn : Callable[[int], None]
        Function that receives an **int 0-100**.
        Examples: `my_qprogressbar.setValue`, `signal.emit`.
    epoch_level : bool, default=True
        If True, update once per epoch; otherwise per batch.
    """
    def __init__(
        self,
        total_epochs: int,
        batches_per_epoch: int,
        update_fn: Callable[[int], None],
        *,
        epoch_level: bool = True,
    ):
        super().__init__()
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.update_fn = update_fn
        self.epoch_level = epoch_level
        self._seen_batches = 0

    # epoch-level -
    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        # if self.epoch_level:
        #     pct = int((epoch + 1) / self.total_epochs * 100)
        #     self.update_fn(pct)
        if self.epoch_level:
            pct = int((ep * 100) // self.total_epochs)
            self.update_fn(pct)

    # batch-level 
    def on_train_batch_end(self, batch, logs=None):
        # if not self.epoch_level:
        #     self._seen_batches += 1
        #     total_batches = self.total_epochs * self.batches_per_epoch
        #     pct = math.floor(self._seen_batches / total_batches * 100)
        #     self.update_fn(pct)
        if not self.epoch_level:
            self._seen_batches += 1
            total_batches = self.total_epochs * self.batches_per_epoch
            pct =(self._seen_batches * 100) // total_batches
            self.update_fn(pct)

    def __deepcopy__(self, memo):
        """
        Keras-Tuner deep-copies the callbacks list for every trial.
        Qt signal objects cannot be copied, so we just return *self*.
        That is perfectly fine because the tuner resets internal
        callback state at the start of each trial.
        """
        return self

class GuiProgress(Callback):
    """
    Emit 0–100% updates while `model.fit` runs, either per‐epoch or
    per‐batch.  All ETA, labels, prefixes, etc. are handled downstream
    by your ProgressManager via the progress_callback.
    """
    def __init__(
        self,
        total_epochs: int,
        batches_per_epoch: int,
        update_fn: Callable[[int], None],
        *,
        epoch_level: bool = True,
    ):
        super().__init__()
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.update_fn = update_fn
        self.epoch_level = epoch_level
        self._seen_batches = 0

    def on_epoch_end(self, epoch, logs=None):
        if not self.epoch_level:
            return
        # epoch is zero‐based; we want (1…total_epochs)
        ep = epoch + 1
        pct = int(ep * 100 // self.total_epochs)
        self.update_fn(pct)

    def on_train_batch_end(self, batch, logs=None):
        if self.epoch_level:
            return
        # count across all epochs
        self._seen_batches += 1
        total_batches = self.total_epochs * self.batches_per_epoch
        pct = int(self._seen_batches * 100 // total_batches)
        self.update_fn(pct)

    def __deepcopy__(self, memo):
        # Keras‐Tuner will copy callbacks; Qt signals can't be
        # so just return self
        return self

class GuiTunerProgress(Callback):
    """
    Emit percentage updates while KerasTuner runs the hyperparameter search.

    Parameters
    ----------
    total_trials : int
        Number of trials in the hyperparameter search.
    
    total_epochs : int
        Number of epochs per trial.
    
    update_fn : Callable[[int], None]
        Function that receives an **int 0-100** to update the progress bar.
        Examples: `my_qprogressbar.setValue`, `signal.emit`.
    
    epoch_level : bool, default=True
        If True, updates once per epoch; otherwise updates per trial.
    """
    def __init__(
        self,
        total_trials: int,
        total_epochs: int,
        update_fn: Callable[[int], None],
        *,
        epoch_level: bool = True,
    ):
        super().__init__()
        self.total_trials = total_trials
        self.total_epochs = total_epochs
        self.update_fn = update_fn
        self.epoch_level = epoch_level
        self._seen_epochs = 0
        self._seen_trials = 0

    def on_epoch_end(self, epoch, logs=None):
        """Track progress at the end of each epoch."""
        if self.epoch_level:
            pct = int((epoch + 1) / self.total_epochs * 100)
            self.update_fn(pct)

    def on_trial_end(self, trial, logs=None):
        """Track progress at the end of each trial."""
        self._seen_trials += 1
        pct = int(self._seen_trials / self.total_trials * 100)
        self.update_fn(pct)

    def on_train_end(self, logs=None):
        """Final update to finish the progress bar."""
        self.update_fn(100)

    def __deepcopy__(self, memo):
        """
        Keras-Tuner deep-copies the callbacks list for every trial.
        Qt signal objects cannot be copied, so we just return *self*.
        That is perfectly fine because the tuner resets internal
        callback state at the start of each trial.
        """
        return self

class GUILoggerCallback(Callback):
    """
    Streams textual log lines *and* a global 0-100 % progress value
    back to the Qt GUI.

    Parameters
    ----------
    log_sig   : QtCore.pyqtSignal(str)
    prog_sig  : QtCore.pyqtSignal(int)
    max_trials: int        # value from `HydroTuner(..., max_trials=N)`
    epochs    : int        # same epochs you pass to tuner.search()
    """
    def __init__(self, *, log_sig, prog_sig,
                 trial_sig, max_trials: int, epochs: int):
        super().__init__()
        def _mk_emit(obj):
            """Return a 0-cost callable whether *obj* is a Qt-signal or func."""
            return obj.emit if hasattr(obj, "emit") else obj

        self._log   = _mk_emit(log_sig)
        self._prog  = _mk_emit(prog_sig)
        self._trial = _mk_emit(trial_sig)
        
        self._max_trials = max_trials
        self._epochs     = epochs
        self._trial_idx  = 0
        self._t0_trial   = None            # wall-clock start of current trial
     
        self._trial(0, self._max_trials, "--:--")


    # Keras-Tuner hooks 
    def on_trial_begin(self, trial):
        try:
            idx = int(trial.trial_id)
        except Exception:                             # fallback – should not happen
            idx = 0
        self._trial_idx = idx + 1
        # (re-)start the per-trial stopwatch for ETA calculation
        self._t0_trial = time.time()
        
        self._log(f"\n── Trial {self._trial_idx}/{self._max_trials} …")
        self._trial(self._trial_idx, self._max_trials, "--:--")
        self._prog(int((self._trial_idx-1) * 100 / self._max_trials))  
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self._t0_trial is None:          # extra belt-and-braces
            self._t0_trial = time.time()
        
        self._log(
            f"      epoch {epoch + 1:>3d} → "
            f"loss={logs.get('loss', 0):.4f} | "
            f"val_loss={logs.get('val_loss', 0):.4f}"
        )
                
        epochs_done = epoch + 1
        elapsed     = time.time() - self._t0_trial
        eta_secs    = (elapsed / epochs_done) * (self._epochs - epochs_done)
        eta_str     = time.strftime("%M:%S", time.gmtime(max(0, eta_secs)))
        self._trial(self._trial_idx, self._max_trials, eta_str)
       
        percent = ((self._trial_idx-1) +
                   epochs_done / self._epochs) * 100 / self._max_trials
        self._prog(int(percent))

    def on_trial_end(self, trial):
        """
        Ensure the progress bar snaps exactly to the next tick when a trial
        finishes.  This matters for very short trials (e.g. 1-epoch) and for
        the final 100 % update at the end of the search.
        """
        # Final ETA for this trial is always zero
        self._trial(self._trial_idx, self._max_trials, "00:00")
        self._prog(int(self._trial_idx * 100 / self._max_trials))
        
    def __deepcopy__(self, memo):
        """
        Keras-Tuner deep-copies the callbacks list for every trial.
        Qt signal objects cannot be copied, so we just return *self*.
        That is perfectly fine because the tuner resets internal
        callback state at the start of each trial.
        """
        return self

# ----functions utils -----------------

def _detect_gpu() -> str | None:
    """
    Very light-weight GPU detection.  Returns a short string or None.
      • Tries NVIDIA via `nvidia-smi` 
      • Falls back to Apple Metal (macOS) or ROCm (linux-AMD) heuristics
    """
    # NVIDIA 
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name",
                                       "--format=csv,noheader"],
                                       stderr=subprocess.DEVNULL,
                                       timeout=2, encoding="utf-8")
        return out.splitlines()[0].strip()
    except Exception:
        pass
    # Apple Metal (M-series) 
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "Apple-Silicon GPU"
    # Other heuristics could go here …
    return None

    
def locate_and_load_manifest(
    manifest_path: Optional[str | os.PathLike] = None,
    validation_data_path: Optional[str | os.PathLike] = None,
    log: Callable[[str], None] = print
) -> Dict[str, Any]:
    """
    Locates and loads a run_manifest.json file.

    This function provides a robust way to find the correct manifest
    file for an inference run. It can be given a direct path to the
    manifest or a path to a new data file, from which it will
    heuristically search for the most recent, relevant manifest.

    Parameters
    ----------
    manifest_path : str or pathlib.Path, optional
        A direct path to the `run_manifest.json` file. If provided,
        this path is used directly.
    validation_data_path : str or pathlib.Path, optional
        A path to the new data file for prediction. This is used as
        the starting point for a heuristic search if `manifest_path`
        is not provided.
    log : callable, default=print
        A logging function to output status messages during the search.

    Returns
    -------
    dict
        The parsed content of the found JSON manifest file.

    Raises
    ------
    ValueError
        If neither `manifest_path` nor `validation_data_path` is provided.
    FileNotFoundError
        If no manifest file can be found from the given paths.
    """
    if not manifest_path and not validation_data_path:
        raise ValueError(
            "Must provide either a `manifest_path` or a "
            "`validation_data_path` to find the run manifest."
        )

    if manifest_path:
        found_manifest_path = Path(manifest_path)
    else:
        # If no direct path, search heuristically from the data path
        found_manifest_path = __locate_manifest(
            Path(validation_data_path), log=log
        )

    if not found_manifest_path or not found_manifest_path.exists():
        raise FileNotFoundError(
            f"Could not find a valid `run_manifest.json` at or near the "
            f"provided path: {manifest_path or validation_data_path}"
        )

    log(f"Loading configuration from manifest: {found_manifest_path}")
    return json.loads(found_manifest_path.read_text("utf-8"))

def _find_manifest_in(dir_: Path) -> List[Path]:
    """Return every run_manifest.json inside *dir_/**_run/ sub-folders."""
    return list(dir_.glob("*_run/run_manifest.json"))

def __locate_manifest(csv_path: Path, max_up: int = 3) -> Optional[Path]:
    """
    Heuristic search for the `run_manifest.json` that matches *csv_path*.

    *   check the CSV folder itself;
    *   check *_run/ sub-folders right under it;
    *   walk **up to `max_up` parent levels**, at each step:
          – look for a `run_manifest.json` next to the folder,
          – look in any *_run/ sub-folder,
          – look inside a sibling *results_pinn/* directory (common default).
    Return the most **recent** hit (mtime) or *None*.
    """
    here = csv_path.parent

    for lvl in range(max_up + 1):
        probe = here if lvl == 0 else csv_path.parents[lvl]

        # 1) same folder
        direct = probe / "run_manifest.json"
        if direct.exists():
            return direct

        # 2) any *_run/ below this folder
        hits = _find_manifest_in(probe)
        if hits:
            return max(hits, key=lambda p: p.stat().st_mtime)

        # 3) common results dir (e.g.  <probe>/results_pinn/*_run/)
        res_dir = probe / "results_pinn"
        if res_dir.is_dir():
            hits = _find_manifest_in(res_dir)
            if hits:
                return max(hits, key=lambda p: p.stat().st_mtime)

    # nothing found
    return None


def _rebuild_from_arch_cfg(arch_cfg: dict) -> Model:
    """
    Rebuilds an un-compiled model instance from an architecture config dict.
    
    This helper de-serializes any custom `Learnable` parameter objects
    and instantiates the correct model class (`TransFlowSubsNet` or
    `PIHALNet`) from the configuration.
    """
    # 1. De-serialize nested Learnable parameter objects
    for key in ("K", "Ss", "Q", "pinn_coefficient_C"):
        param_config = arch_cfg.get(key)
        if isinstance(param_config, dict) and "class_name" in param_config:
            arch_cfg[key] = deserialize_keras_object(
                param_config, custom_objects=_CUSTOM_OBJECTS
            )

    # 2. Decide which concrete model class to instantiate
    cls_name = arch_cfg.get("name", "TransFlowSubsNet")
    ModelCls = {
        "TransFlowSubsNet": TransFlowSubsNet,
        "PIHALNet": PIHALNet
    }.get(cls_name)
    
    if ModelCls is None:
        raise ValueError(f"Unknown model class '{cls_name}' in manifest config.")

    # 3. Build and return the un-compiled model from its config
    return ModelCls.from_config(arch_cfg)

def safe_model_loader(
    model_path: str | os.PathLike,
    *,
    build_fn: Optional[Callable[[], Model]] = None,
    custom_objects: Optional[dict] = None,
    log: Callable[[str], None] = print,
    model_cfg: Optional[Dict[str, Any]] = None,
) -> Model:
    """Loads a Keras/TF model saved in various formats.

    This function provides a single, robust entry point for loading models
    saved as:
    - The modern Keras v3 format (``.keras``)
    - The legacy HDF5 format (``.h5``)
    - The TensorFlow SavedModel directory format
    - Weights-only files (e.g., ``.weights.h5``)

    For weights-only files, this function requires a build function or an
    architecture configuration from a manifest to reconstruct the model
    before loading the weights.

    Parameters
    ----------
    model_path : str or pathlib.Path
        The path to the model file or directory.
    build_fn : callable, optional
        A callable function that returns a fresh, un-compiled model
        instance. Required if `model_path` points to a weights-only
        file and `arch_cfg` is not provided.
    custom_objects : dict, optional
        A dictionary mapping names to custom classes or functions required
        for loading the model. Merged with default PINN custom objects.
    log : callable, default=print
        A logging function to output status messages.
    model_cfg : dict, optional
        The trained model configuration. It encompasses the architecture 
         configuration dictionary, typically from a manifest
        file. Used with `build_fn` to reconstruct a model for
        weights-only loading.

    Returns
    -------
    keras.Model
        The loaded Keras model.

    Raises
    ------
    IOError
        If the model path does not exist or if loading fails for other reasons.
    ValueError
        If a weights-only file is provided without a way to reconstruct the
        model (i.e., missing `build_fn` or `arch_cfg`).
    """
    path = Path(model_path)
    if not path.exists():
        raise IOError(f"[safe_model_loader] Path does not exist: {path}")

    # Combine user custom objects with the default ones for PINN params
    final_custom_objects = _CUSTOM_OBJECTS.copy()
    if custom_objects:
        final_custom_objects.update(custom_objects)
    
    arch_cfg = model_cfg.get("config")
    try:
        # --- Case 1: Full Model (directory or .keras/.h5 file) ---
        if path.is_dir() or (path.suffix in {".keras", ".h5"} and not
                             path.name.endswith(".weights.h5")):
            log(f"[loader] Reading full model from: {path.name}")
            with custom_object_scope(final_custom_objects):
                model = load_model(path)
            log("  Model loaded successfully.")
            return model

        # --- Case 2: Weights-only file ---
        elif path.name.endswith((".weights.h5", ".weights.keras")):
            log(f"[loader] Loading weights from: {path.name}")
            
            # Rebuild the model architecture first
            if build_fn:
                model = build_fn()
                log("  Rebuilding model from provided build_fn...")
            elif arch_cfg:
                model = _rebuild_from_arch_cfg(arch_cfg)
                log("  Rebuilding model from manifest architecture config...")
            else:
                raise ValueError(
                    "A `build_fn` or `arch_cfg` must be provided to load a "
                    "weights-only file."
                )

            # Build the model to create its weights before loading
            if not model.built and arch_cfg and "input_shapes" in model_cfg:
                log("  Building model with input shapes from manifest...")
                try:
                    # The `input_shapes` from the manifest will be a dict
                    model.build(model_cfg["input_shapes"])
                except Exception as e:
                    log(f"  [Warning] Failed to build model with manifest"
                        f" shapes: {e}. Model will build on first call.")

            # Load the weights into the reconstructed architecture
            model.load_weights(str(path))
            log("  Weights loaded successfully into reconstructed model.")
            return model

        else:
            raise ValueError(f"Unknown model format for path: {path}")

    except Exception as err:
        raise IOError(
            f"[safe_model_loader] Failed to load model from {path}. "
            f"Ensure custom objects are registered or provided. Error: {err}"
        ) from err

def json_ready(obj: Any, *, mode: str = "literal") -> Any:
    """
    Recursively walk *obj* and return a clone that **json.dumps**
    can handle.

    Parameters
    ----------
    obj   : any Python object (dict / list / scalar / …)
    mode  : "literal" | "config"
        * literal – replace Learnable… objects with the same value you
          originally passed to the model (i.e. ``'learnable'`` /
          ``'fixed'`` **or** the float) so the JSON is compact.
        * config  – call ``.get_config()`` on each Learnable… object and
          store that dict instead (useful if you want to rebuild them
          later with ``.from_config``).

    Returns
    -------
    jsonable_obj : an isomorphic structure containing only JSON-safe
                   data types (dict / list / str / int / float / bool /
                   None).

    Notes
    -----
    •  If *obj* already is JSON-safe it is returned unchanged.  
    •  Unknown custom classes will raise ``TypeError`` – add your own
       handler if you need more.
    """
    # 1) primitives — nothing to do
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # 2) Learnable…  --------------------------------------------
    if isinstance(obj, _LEARNABLE_TYPES):
        if mode == "literal":
            # Re-use the original “initial value”
            return obj.initial_value
        elif mode == "config":
            return serialize_keras_object(obj)# obj.get_config()
        else:
            raise ValueError("mode must be 'literal' or 'config'")

    # 3) collections — walk recursively
    if isinstance(obj, dict):
        return {k: json_ready(v, mode=mode) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [json_ready(v, mode=mode) for v in obj]

    # 4) anything else → unsupported
    raise TypeError(f"Object of type {type(obj).__name__} "
                    "is not JSON serialisable")


def _safe_eval(txt: str):
    """
    Return the Python literal contained in *txt* without using eval().

    * First choice  : ast.literal_eval  (fast, safe, built-in)
    * Fallback      : json.loads       (more restrictive – strings must use "")
    """
    if _AST_AVAILABLE:
        return ast.literal_eval(txt)
    return json.loads(txt)

def _is_number(x):               # (int OR float, but NOT bool)
    return isinstance(x, (Integral, Real)) and not isinstance(x, bool)

def _convert_if_path(obj):
    """Turn any pathlib.Path into its string form so json.dumps will work."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _convert_if_path(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(_convert_if_path(v) for v in obj)
    return obj

def parse_search_space(text: str) -> dict:
    """
    Safely parse the user-supplied Python-like dict and return a clean,
    JSON-serialisable dictionary that HydroTuner can consume.

    Raises ValueError / TypeError with descriptive messages on bad input.
    """
    try:
        space = _safe_eval(text.strip())
    except Exception as err:
        raise ValueError(f"Not valid Python literal: {err}") from None

    if not isinstance(space, dict):
        raise TypeError("The top-level object must be a dictionary")

    cleaned = {}
    for hp_name, spec in space.items():

        #  Case A : list / tuple → hp.Choice 
        if isinstance(spec, (list, tuple)):
            if not spec:
                raise ValueError(f"{hp_name}: choice list cannot be empty")
            cleaned[hp_name] = [_convert_if_path(v) for v in spec]
            continue

        #  Case B : {min_value, max_value, …} 
        if isinstance(spec, dict):
            if not _ALLOWED_KEYS.issuperset(spec):
                extra = set(spec) - _ALLOWED_KEYS
                raise ValueError(
                    f"{hp_name}: unknown keys in range spec: {extra}"
                )
            lo, hi = spec.get("min_value"), spec.get("max_value")
            if lo is None or hi is None or not (_is_number(lo) and _is_number(hi)):
                raise TypeError(f"{hp_name}: min/max must be numbers")
            if lo >= hi:
                raise ValueError(f"{hp_name}: min_value must be < max_value")
            step = spec.get("step")
            if step is not None and (not _is_number(step) or step <= 0):
                raise ValueError(f"{hp_name}: step must be a positive number")

            # everything OK – deep-copy & convert Paths to str just in case
            cleaned[hp_name] = _convert_if_path(spec)
            continue

        # ---------- Anything else ------------------------------------------------
        raise TypeError(
            f"{hp_name}: expected list/tuple for choices or dict "
            "{min_value, max_value, …} for a range; got {type(spec).__name__}"
        )

    return cleaned

def get_safe_output_dir(
    base_dir: str,
    run_type: str,
    *,
    force: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Create and return a safe, structured output directory without nesting.

    If `base_dir` is already inside a `.fusionlab_runs` tree, this
    function will treat it as the safe directory and return it directly,
    avoiding duplicate `.fusionlab_runs` subfolders.

    Parameters
    ----------
    base_dir : str
        The user-specified root output directory (might already be under
        `.fusionlab_runs`).
    run_type : str
        The type of workflow, e.g. 'training' or 'tuning'.
    force : bool, default=False
        If True and `base_dir` is not yet under `.fusionlab_runs`, create
        a fresh timestamped root.  Otherwise reuse existing.
    log : callable, optional
        Optional logger (e.g., print) to notify about reuse or creation.

    Returns
    -------
    pathlib.Path
        The path to the safe output directory for this run_type.
    """
    base = Path(base_dir)

    # --- already safe? ---
    if ".fusionlab_runs" in base.parts:
        # base_dir is already inside the branded folder → reuse
        if log:
            log(f"Reusing existing safe output dir: {base}")
        base.mkdir(parents=True, exist_ok=True)
        return base

    # --- not yet under .fusionlab_runs: build it ---
    safe_root = base / ".fusionlab_runs"

    # if safe root exists and force==False, reuse it
    if safe_root.exists() and not force:
        if log:
            log(f"Using existing '.fusionlab_runs' directory at {safe_root}")
    elif force and safe_root.exists():
        # optionally, you could timestamp here to force a fresh root
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        safe_root = base / f".fusionlab_runs_{ts}"
        if log:
            log(f"Force-creating new safe root: {safe_root}")

    # subdir for this run type
    type_dir = safe_root / f"{run_type}_results"
    type_dir.mkdir(parents=True, exist_ok=True)

    if log:
        log(f"Created safe output dir: {type_dir}")

    return type_dir

def inspect_run_type_from_manifest(
    path: str | os.PathLike,
    delegate: str = 'raise'
) -> str:
    """Inspects a path to determine the run type from a manifest.

    This function is to the input `path` being either a direct
    path to a manifest file or a path to a run directory. It checks
    for the presence of specific manifest files to determine if a run
    was from a 'tuning' or 'training' workflow.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to inspect. This can be a directory containing a
        manifest file or a direct path to a `run_manifest.json` or
        `tuner_run_manifest.json`.
    delegate : {'raise', 'warn', 'ignore', 'training'}, default='raise'
        The behavior to adopt if no manifest file is found at the path.
        - 'raise': Raises a FileNotFoundError.
        - 'warn': Issues a warning and defaults to 'training'.
        - 'ignore': Silently defaults to 'training'.
        - 'training': Explicitly defaults to 'training' without a warning.

    Returns
    -------
    str
        Returns 'tuning' if a tuner manifest is found or if the manifest
        content indicates a tuning run. Otherwise, returns 'training'.

    Raises
    ------
    FileNotFoundError
        If no manifest can be found and `delegate` is set to 'raise'.
    ValueError
        If the found manifest file is corrupted or not valid JSON.
    """
    path_obj = Path(path)
    manifest_path: Optional[Path] = None

    # --- Step 1: Find the correct manifest file ---
    if path_obj.is_dir():
        # If a directory is provided, search inside it.
        # Prioritize the tuner manifest if it exists.
        tuner_manifest = path_obj / "tuner_run_manifest.json"
        run_manifest = path_obj / "run_manifest.json"
        if tuner_manifest.exists():
            manifest_path = tuner_manifest
        elif run_manifest.exists():
            manifest_path = run_manifest
    elif path_obj.is_file():
        # If a file is provided, use it directly.
        manifest_path = path_obj

    # --- Step 2: Handle cases where no manifest is found ---
    if not manifest_path or not manifest_path.exists():
        msg = f"No manifest file found at or in the directory: {path}"
        if delegate == 'raise':
            raise FileNotFoundError(msg)
        if delegate == 'warn':
            warnings.warn(msg + ". Defaulting to 'training' mode.")
        # For 'warn', 'ignore', or 'training', we default to training.
        return 'training'

    # --- Step 3: Inspect the manifest content to determine run type ---
    try:
        manifest_data = json.loads(manifest_path.read_text("utf-8"))
        # The presence of these keys is a definitive sign of a tuning run.
        if "tuner" in manifest_data or "tuner_results" in manifest_data:
            return 'tuning'
        else:
            return 'training'
        
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse manifest file at {manifest_path}. "
            f"It may be corrupted. Error: {e}"
        )

def log_tuning_params(
    cfg_dict: Dict[str, Any],
    log_fn: Callable[[str], None] = print
) -> None:
    """
    Nicely prints a two-column ASCII table of all the tuning parameters
    in `cfg_dict`, using `log_fn` for each line (e.g. self._log).

    It flattens three sections:
      • fixed_params
      • sequence_params
      • tuner_settings
    """
    sections = [
        ("Search space", cfg_dict.get ("search_space", {})), 
        ("Fixed params", cfg_dict.get("fixed_params", {})),
        ("Sequence params", cfg_dict.get("sequence_params", {})),
        ("Tuner settings", cfg_dict.get("tuner_settings", {})),
    ]

    # gather all rows
    rows = []
    for title, sub in sections:
        if not sub:
            continue
        rows.append((title, ""))  # section header
        for k, v in sub.items():
            rows.append((f"  {k}", v))

    if not rows:
        log_fn("⚠ No tuning parameters to display.")
        return

    # compute column widths
    col1w = max(len(str(r[0])) for r in rows) + 1
    col2w = max(len(str(r[1])) for r in rows) + 1

    sep = "+" + "-" * (col1w + 2) + "+" + "-" * (col2w + 2) + "+"
    log_fn(sep)
    log_fn(f"| {'Parameter'.ljust(col1w)} | {'Value'.ljust(col2w)} |")
    log_fn(sep)
    for name, val in rows:
        if val == "" and not name.startswith("  "):
            # section header
            header = name.upper()
            log_fn(f"| {header.center(col1w)} | {' '.ljust(col2w)} |")
            log_fn(sep)
        else:
            log_fn(f"| {name.ljust(col1w)} | {str(val).ljust(col2w)} |")
    log_fn(sep)


def get_workflow_status(
    cfg: Any,
    default: str = "trained"
) -> str:
    """
    Derive a normalized “status” keyword from a config-like object.

    Looks for either `cfg.run_type` or `cfg.mode` and maps common
    values to human-friendly status strings:

      - training  → trained
      - tuning    → tuned
      - inference → inferred

    If neither attribute is present or its value is unrecognized,
    returns `default`.

    Examples
    --------
    >>> class C: run_type = "training"
    >>> get_workflow_status(C())
    'trained'

    >>> class C: run_type = "tuning"
    >>> get_workflow_status(C())
    'tuned'

    >>> class C: mode = "inference"
    >>> get_workflow_status(C(), default="done")
    'inferred'

    >>> class C: pass
    >>> get_workflow_status(C(), default="idle")
    'idle'
    """
    # try run_type first, then mode
    raw = getattr(cfg, "run_type", None) or getattr(cfg, "mode", None)
    if raw is None:
        return default

    key = str(raw).strip().lower()
    if key in ("training", "train"):
        return "trained"
    if key in ("tuning", "tune"):
        return "tuned"
    if key in ("inference", "infer"):
        return "inferred"
    # fallback to default for unrecognized
    return default

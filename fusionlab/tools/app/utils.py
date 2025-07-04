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
import math 
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


class _StopCheckCallback(Callback):
    """A Keras callback to gracefully interrupt training or tuning.

    This callback is designed to be used within a Keras `.fit()` or
    Keras Tuner `.search()` loop. At the end of each epoch, it calls a
    provided `stop_check` function. If this function returns `True`,
    the callback sets `model.stop_training = True`, which tells Keras
    to halt the current process cleanly.

    This provides a robust mechanism for a GUI's "Stop" button to
    interrupt a long-running backend task.

    Parameters
    ----------
    stop_check_fn : callable
        A function with no arguments that returns `True` if an
        interruption has been requested, and `False` otherwise. This is
        typically connected to a `QThread.isInterruptionRequested`.
    log_callback : callable, optional
        A logging function to output a message when the interruption
        is triggered. Defaults to `print`.

    Attributes
    ----------
    was_stopped : bool
        A flag that is set to `True` if the training process was
        successfully stopped by this callback. This can be checked
        after the `.fit()` or `.search()` call completes.

    Examples
    --------
    >>> from fusionlab.tools.app.utils import StopCheckCallback
    >>> stop_requested = False
    >>> def check_if_stopped():
    ...     # In a real app, this would check a thread's state.
    ...     return stop_requested
    ...
    >>> stop_callback = StopCheckCallback(check_if_stopped)
    >>> # model.fit(..., callbacks=[stop_callback])
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

    def on_epoch_end(self, epoch, logs=None):
        """
        Called by Keras at the end of each training epoch.
        """
        if self.stop_check():
            self.model.stop_training = True
            self.was_stopped = True
            self.log("  [Callback] Interruption requested. Halting process...")

    def __deepcopy__(self, memo):
        return self

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
        if self.stop_check():
            if not self.was_stopped:
                self.log("  [Callback] Interruption requested. Halting process…")
                self.was_stopped = True
            # 1) stop the current fit loop
            self.model.stop_training = True
            # 2) abort tuner.search() as well
            raise InterruptedError("Stop requested by user")

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
    def __init__(
        self,
        total_trials: int,
        total_epochs: int,
        batches_per_epoch: int,
        progress_manager,
        *,
        epoch_level=True,
        trial_batch_level=False,
        log=print
    ):
        super().__init__()
        self.total_trials       = total_trials
        self.total_epochs       = total_epochs
        self.batches_per_epoch  = batches_per_epoch
        self.progress_manager   = progress_manager
        self.epoch_level        = epoch_level
        self.trial_batch_level  = trial_batch_level
        self.log                = log

        self.trial_idx    = 0
        self.global_batch = 0
        self.global_total = (total_trials
                             * total_epochs
                             * batches_per_epoch)

    def on_trial_begin(self, trial):
        self.trial_idx    = trial.trial_id
        self.global_batch = self.trial_idx * (
            self.total_epochs * self.batches_per_epoch
        )
        self.progress_manager.set_trial_context(
            trial=self.trial_idx + 1,
            total=self.total_trials
        )
        self.log(f"Trial {self.trial_idx+1}/"
                 f"{self.total_trials} started.")

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_level:
            # bump global_batch to end‐of‐this‐epoch
            self.global_batch = (
                self.trial_idx * self.total_epochs
                + (epoch + 1)
            ) * self.batches_per_epoch
            self._apply_context_and_update()

    def on_batch_end(self, batch, logs=None):
        if self.trial_batch_level:
            # increment one local batch
            self.global_batch += 1

            # re-apply trial + epoch context before every update
            epoch = (self.global_batch // self.batches_per_epoch) \
                    - (self.trial_idx * self.total_epochs)
            self.progress_manager.set_trial_context(
                trial=self.trial_idx + 1,
                total=self.total_trials
            )
            self.progress_manager.set_epoch_context(
                epoch=epoch + 1,
                total=self.total_epochs
            )

            self._apply_context_and_update()

    def _apply_context_and_update(self):
        """Helper to call ProgressManager.update with correct globals."""
        self.progress_manager.update(
            current=self.global_batch,
            total=self.global_total
        )
        pct = self.global_batch / self.global_total * 100
        self.log(f"Trial {self.trial_idx+1}/{self.total_trials} - "
                 f"Global batch {self.global_batch} – "
                 f"Progress: {pct:.2f}%")

    def on_trial_end(self, trial):
        self._apply_context_and_update()
        self.log(f"Trial {self.trial_idx+1}/{self.total_trials} done.")

    def on_train_end(self, logs=None):
        self.progress_manager.finish_step("Done")
        self.log("Tuning completed!")

    def __deepcopy__(self, memo):
        return self

class TunerProgressCallback0(Callback):
    """
    A Keras Callback to track the progress of the hyperparameter tuning process
    using KerasTuner. It updates the progress bar during training and tuning
    trials, ensuring smooth updates for each step (trial, epoch, or batch).
    
    Parameters
    ----------
    total_trials : int
        Total number of trials to run in the tuning process.
    
    total_epochs : int
        Total number of epochs in each trial.
    
    progress_manager : ProgressManager
        The ProgressManager instance for managing and updating the progress bar.
    
    epoch_level : bool, default=True
        If True, updates the progress per epoch, else updates per batch.
    
    trial_batch_level : bool, default=False
        If True, updates progress bar per batch for each trial.
        
    log : logging.Logger or callable, optional
        A logger or callable (e.g., print) to capture progress messages.
    """
    
    def __init__(
        self, total_trials, total_epochs, progress_manager, 
        epoch_level=True, trial_batch_level=False, log=None
    ):
        super().__init__()
        self.total_trials = total_trials
        self.total_epochs = total_epochs
        self.progress_manager = progress_manager
        self.epoch_level = epoch_level
        self.trial_batch_level = trial_batch_level
        self.log = log or print
        self.trial_idx = 0
        self.batch_idx = 0
        self.epochs_per_trial = total_epochs
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Update progress at the end of each epoch. This is triggered during each
        epoch of training within a trial.
        """
        if self.epoch_level:
            pct = (
                (self.trial_idx * self.total_epochs + epoch + 1) / 
                (self.total_trials * self.total_epochs) * 100
            )
            self.progress_manager.update(current=epoch + 1, total=self.total_epochs)
            self.log(f"Trial {self.trial_idx + 1}/{self.total_trials} - "
                     f"Epoch {epoch + 1}/{self.total_epochs} - Progress: {pct:.2f}%")

    def on_batch_end(self, batch, logs=None):
        """
        Optionally, update progress per batch (useful for fine-grained tracking).
        """
        if self.trial_batch_level:
            pct = (
                (self.trial_idx * self.total_epochs * self.epochs_per_trial + 
                 self.batch_idx + 1) / 
                (self.total_trials * self.total_epochs * self.epochs_per_trial) * 100
            )
            self.progress_manager.update(current=batch + 1, total=self.total_epochs)
            self.log(f"Trial {self.trial_idx + 1}/{self.total_trials} - "
                     f"Batch {batch + 1} - Progress: {pct:.2f}%")
            self.batch_idx += 1  # Update batch index

    def on_trial_begin(self, trial):
        """
        Initialize and log trial start.
        """
        self.trial_idx = trial.trial_id
        self.batch_idx = 0
        self.progress_manager.set_trial_context(trial=self.trial_idx + 1, 
                                                 total=self.total_trials)
        self.log(f"Trial {self.trial_idx + 1}/{self.total_trials} started.")

    def on_trial_end(self, trial):
        """
        Finalize progress and log when a trial finishes.
        """
        self.log(f"Trial {self.trial_idx + 1}/{self.total_trials} finished.")
        pct = (self.trial_idx + 1) / self.total_trials * 100
        self.progress_manager.update(
            current=self.total_epochs, total=self.total_epochs)
        self.log(f"Trial progress: {pct:.2f}% complete.")

    def on_train_end(self, logs=None):
        """
        Log completion of training.
        """
        self.log("Hyperparameter tuning completed!")
        self.progress_manager.finish_step("Done")


    def __deepcopy__(self, memo):
        """
        Keras-Tuner deep-copies the callbacks list for each trial.
        Qt signal objects cannot be copied, so we return *self* to avoid 
        duplication. This is fine as the tuner resets internal callback 
        state at the start of each trial.
        """
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

class _TunerProgressCallback(Callback):
    """
    A Keras Callback to track the progress of the hyperparameter tuning process
    using KerasTuner. It updates the progress bar during training and tuning
    trials, ensuring smooth updates for each step (trial, epoch, or batch).
    
    Parameters
    ----------
    total_trials : int
        Total number of trials to run in the tuning process.
    
    total_epochs : int
        Total number of epochs in each trial.
    
    update_fn : Callable[[int], None]
        A function that receives an integer percentage (0 to 100) to update a progress
        bar or any other UI component.
    
    epoch_level : bool, default=True
        If True, updates the progress per epoch, else updates per batch.
    
    trial_batch_level : bool, default=False
        If True, updates progress bar per batch for each trial.
        
    logger : logging.Logger, optional
        A logger to capture progress messages.
    
    Attributes
    ----------
    trial_idx : int
        Current trial index being executed in the hyperparameter search.
    """

    def __init__(
        self, 
        total_trials,
        total_epochs, 
        update_fn,
        epoch_level=True,
        trial_batch_level=False, 
        log_callback=None
        ):
        super().__init__()
        self.total_trials = total_trials
        self.total_epochs = total_epochs
        self.update_fn = update_fn
        self.epoch_level = epoch_level
        self.trial_batch_level = trial_batch_level
        self.log = log_callback or print 
        self.trial_idx = 0
        self.batch_idx = 0
        self.epochs_per_trial = total_epochs
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Update progress at the end of each epoch. This is triggered during each
        epoch of training within a trial.
        """
        if self.epoch_level:
            pct = (self.trial_idx * self.total_epochs + epoch + 1) / (
                self.total_trials * self.total_epochs) * 100
            self.update_fn(int(pct))
            self.log(
                f"Trial {self.trial_idx + 1}/{self.total_trials}"
                " - Epoch {epoch + 1}/{self.total_epochs} - Progress: {pct:.2f}%")

    def on_batch_end(self, batch, logs=None):
        """
        Optionally, update progress per batch (useful for fine-grained tracking).
        """
        if self.trial_batch_level:
            pct = (self.trial_idx * self.total_epochs * self.epochs_per_trial + self.batch_idx + 1
                   ) / (self.total_trials * self.total_epochs * self.epochs_per_trial) * 100
            self.update_fn(int(pct))
            self.log(f"Trial {self.trial_idx + 1}/{self.total_trials}"
                             " - Batch {batch + 1} - Progress: {pct:.2f}%")
            
    def on_trial_begin(self, trial):
        """
        Initialize and log trial start.
        """
        self.trial_idx = trial.trial_id
        self.batch_idx = 0
        self.log(f"Trial {self.trial_idx + 1}/{self.total_trials} started.")

    def on_trial_end(self, trial):
        """
        Finalize progress and log when a trial finishes.
        """
        self.log(f"Trial {self.trial_idx + 1}/{self.total_trials} finished.")
        pct = (self.trial_idx + 1) / self.total_trials * 100
        self.update_fn(int(pct))
        self.log(f"Tuning progress: {pct:.2f}% complete.")

    def on_train_end(self):
        """
        Log completion of training.
        """
        self.log("Hyperparameter tuning completed!")
        self.update_fn(100)


class GuiProgress(Callback):
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
        if self.epoch_level:
            pct = int((epoch + 1) / self.total_epochs * 100)
            self.update_fn(pct)

    # batch-level 
    def on_train_batch_end(self, batch, logs=None):
        if not self.epoch_level:
            self._seen_batches += 1
            total_batches = self.total_epochs * self.batches_per_epoch
            pct = math.floor(self._seen_batches / total_batches * 100)
            self.update_fn(pct)

    def __deepcopy__(self, memo):
        """
        Keras-Tuner deep-copies the callbacks list for every trial.
        Qt signal objects cannot be copied, so we just return *self*.
        That is perfectly fine because the tuner resets internal
        callback state at the start of each trial.
        """
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

# class _TunerProgressCallback(Callback):
#     """A Keras callback to provide detailed progress during tuning.

#     This callback integrates with the `ProgressManager` to update the
#     GUI's progress bar and contextual text label during a Keras Tuner
#     search. It tracks the overall progress across all trials and updates
#     the label with the current trial number and ETA.

#     Parameters
#     ----------
#     progress_manager : ProgressManager
#         The central progress management object from the main GUI.
#     max_trials : int
#         The total number of trials in the tuning search.
#     epochs_per_trial : int
#         The maximum number of epochs for each individual trial.
#     """
#     def __init__(
#         self,
#         progress_manager, 
#         max_trials: int,
#         epochs_per_trial: int,
#     ):
#         super().__init__()
#         self.pm = progress_manager
#         self.max_trials = max_trials
#         self.epochs_per_trial = epochs_per_trial
#         self.current_trial = 0
#         self.total_epochs_in_search = self.max_trials * self.epochs_per_trial

#     def on_trial_begin(self, trial):
#         """Called by Keras Tuner at the beginning of a new trial."""
#         # Keras Tuner trial_id is a string, but we want a simple number
#         self.current_trial += 1
#         # Use the progress manager to set the contextual text
#         self.pm.set_trial_context(
#             trial=self.current_trial, total=self.max_trials
#         )

#     def on_epoch_end(self, epoch, logs=None):
#         """Called by Keras at the end of each epoch within a trial."""
#         # Calculate the total number of epochs completed so far across all trials
#         epochs_in_past_trials = (self.current_trial - 1) * self.epochs_per_trial
#         total_epochs_so_far = epochs_in_past_trials + (epoch + 1)
        
#         # Update the progress manager with the overall progress
#         self.pm.update(total_epochs_so_far, self.total_epochs_in_search)
        
#     def __deepcopy__(self, memo):
#         """
#         Keras-Tuner deep-copies the callbacks list for every trial.
#         Qt signal objects cannot be copied, so we just return *self*.
#         That is perfectly fine because the tuner resets internal
#         callback state at the start of each trial.
#         """
#         return self
    
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
    run_type: str
) -> Path:
    """Creates and returns a safe, structured output directory.

    This helper prevents the application from overwriting user data
    by creating a dedicated, uniquely named subdirectory for all
    its outputs.

    Args:
        base_dir (str): The user-specified root output directory.
        run_type (str): The type of workflow, e.g., 'training' or 'tuning'.

    Returns:
        pathlib.Path: The path to the safe, type-specific output directory.
    """
    # Create a main, branded directory to avoid conflicts.
    # Using a dot prefix makes it hidden on Linux/macOS.
    safe_root = Path(base_dir) / ".fusionlab_runs"
    
    # Create a subdirectory for the specific run type.
    type_specific_dir = safe_root / f"{run_type}_results"
    
    # Ensure the full path exists.
    type_specific_dir.mkdir(parents=True, exist_ok=True)
    
    return type_specific_dir

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

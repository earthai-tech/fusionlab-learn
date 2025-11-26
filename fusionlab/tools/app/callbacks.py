# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Provides high-level utility functions for the GUI application,
such as a robust model loader that can handle multiple Keras/TF
saving formats.
"""
from __future__ import annotations 


from typing import Callable, Optional


try:
    from fusionlab.nn import KERAS_DEPS
except ImportError as e:
    raise ImportError(
        "This utility requires the `fusionlab` library and its"
        f" dependencies to be installed. Error: {e}"
    )

_ALLOWED_KEYS = {
    "min_value", "max_value", "step",          
    "sampling",                                
}

Callback =KERAS_DEPS.Callback 

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
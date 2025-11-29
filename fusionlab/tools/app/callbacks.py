# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
GUI-aware Keras callbacks for GeoPrior / FusionLab apps.

This module centralizes small callbacks that integrate Keras training
loops with Qt-based GUIs:

- GuiProgress:
    Converts batch/epoch progress into 0–100% percentages and forwards
    them to a GUI progress callback (e.g. ProgressManager.update).

- GuiEpochLogger:
    Logs per-epoch metrics (loss, val_loss, etc.) via a provided logger
    function instead of printing to stdout.

- GuiEarlyStopping:
    Drop-in replacement for keras.callbacks.EarlyStopping that sends a
    concise summary message (stopped epoch, best metric) to a logger.
"""

from __future__ import annotations

import sys, io, contextlib
import math
from typing import Callable, Mapping, Any, Sequence, Optional, Dict, List

from .. import KERAS_DEPS

_ALLOWED_KEYS = {
    "min_value", "max_value", "step",          
    "sampling",                                
}

Callback =KERAS_DEPS.Callback 
EarlyStopping =KERAS_DEPS.EarlyStopping

LogFn = Callable[[str], None]
PctUpdateFn = Callable[[int], None]


class GuiMetricLogger(Callback):
    """
    Grouped metric logger for GUI.

    At the end of each epoch, it prints a compact, grouped summary of
    selected metrics into the provided log function (e.g. GUI log box).

    Parameters
    ----------
    metric_groups : dict
        Mapping "Group name" -> list of metric names as they appear in
        Keras logs (e.g. ["loss", "val_loss"] or
        ["subs_pred_mae", "val_subs_pred_mae"]).
    log_fn : callable
        Function `f(msg: str) -> None` used to display logs
        (in your GUI, this is the same `log` you pass to run_training).
    total_epochs : int, optional
        If given, the callback will print `Epoch i/N`. Otherwise it
        prints `Epoch i` only.
    precision : int, optional
        Number of decimal places for metric values.
    """

    def __init__(
        self,
        metric_groups: Mapping[str, List[str]],
        log_fn: Callable[[str], None],
        total_epochs: Optional[int] = None,
        precision: int = 4,
    ) -> None:
        super().__init__()
        self.metric_groups = dict(metric_groups)
        self.log_fn = log_fn
        self.total_epochs = total_epochs
        self.precision = int(precision)

    def _fmt_val(self, v) -> str:
        try:
            # Floats / ints → fixed precision
            if isinstance(v, (float, int)):
                return f"{v:.{self.precision}f}"
            # Anything else → str fallback
            return str(v)
        except Exception:
            return str(v)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        logs = logs or {}

        # Header -------------------------------------------------------
        epoch_idx = epoch + 1
        if self.total_epochs is not None and self.total_epochs > 0:
            header = f"[Metrics] Epoch {epoch_idx}/{self.total_epochs}"
        else:
            header = f"[Metrics] Epoch {epoch_idx}"
        lines = [header]

        # Precompute padding for group names for nicer alignment
        pad = 0
        if self.metric_groups:
            pad = max(len(name) for name in self.metric_groups)

        # One block per group ------------------------------------------
        for group_name, metric_names in self.metric_groups.items():
            # Keep only metrics that actually appear in logs
            present = [m for m in metric_names if m in logs]
            if not present:
                continue

            train_parts: List[str] = []
            val_parts: List[str] = []

            for m in present:
                val = self._fmt_val(logs.get(m))
                # heuristics: val_* → validation side, else training side
                if m.startswith("val_"):
                    val_parts.append(f"{m}={val}")
                else:
                    train_parts.append(f"{m}={val}")

            if not train_parts and not val_parts:
                continue

            gname = group_name.ljust(pad)
            if train_parts and val_parts:
                line = (
                    f"  {gname} : "
                    f"train({', '.join(train_parts)}) | "
                    f"val({', '.join(val_parts)})"
                )
            elif train_parts:
                line = f"  {gname} : train({', '.join(train_parts)})"
            else:
                line = f"  {gname} : val({', '.join(val_parts)})"

            lines.append(line)

        # Only log if at least one group had metrics
        if len(lines) > 1:
            msg = "\n".join(lines)
            try:
                self.log_fn(msg)
            except Exception:
                # Never let logging crash training
                pass

    def __deepcopy__(self, memo):
        # KerasTuner deep-copies callbacks per trial; signals/threads
        # cannot be deep-copied, so return self.
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

class StopTrainingOnSignal(Callback):
    """Keras callback that uses a `stop_check` callable to interrupt fit."""

    def __init__(
        self,
        stop_check: Callable[[], bool],
        logger: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.stop_check = stop_check
        self._log = logger or (lambda msg: print(msg, flush=True))

    def on_batch_end(self, batch, logs=None):
        if self.stop_check and self.stop_check():
            self._log("[Stop] stop_check() returned True – stopping training.")
            self.model.stop_training = True

    def _maybe_stop(self):
        if self.stop_check is not None and self.stop_check():
            if not self.was_stopped:
                self._log("[Callback] Interruption requested. Halting process…")
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
        # Keras‐Tuner will copy callbacks; Qt signals can't be
        # so just return self
        return self


class GuiProgbarLogger(Callback):
    """
    Text logger that mimics Keras' ASCII progbar, but sends lines to a
    GUI logger function instead of stdout.

    It does NOT draw an in-place bar; it prints compact lines like:
        Epoch 3/50 [=======>.....] 42/150 - loss=0.1234 - val_loss=...
    """

    def __init__(
        self,
        total_epochs: int,
        batches_per_epoch: int,
        log_fn: Callable[[str], None],
        *,
        every_n_batches: int = 10,
        bar_width: int = 20,
    ) -> None:
        super().__init__()
        self.total_epochs = max(1, int(total_epochs))
        self.batches_per_epoch = max(1, int(batches_per_epoch))
        self.log_fn = log_fn
        self.every_n_batches = max(1, int(every_n_batches))
        self.bar_width = max(10, int(bar_width))

        self._epoch = 0
        self._batch = 0

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        # Keras is 0-based for epoch index
        self._epoch = epoch
        self._batch = 0
        self.log_fn(
            f"[Train] Epoch {epoch + 1}/{self.total_epochs} started..."
        )

    def on_train_batch_end(self, batch: int, logs=None) -> None:
        logs = logs or {}
        self._batch = batch + 1  # 1-based
        if self._batch % self.every_n_batches != 0 and self._batch != self.batches_per_epoch:
            return

        frac = self._batch / float(self.batches_per_epoch)
        frac = max(0.0, min(1.0, frac))

        bar = self._make_bar(frac)
        loss = logs.get("loss")
        loss_str = f"{loss:.4f}" if isinstance(loss, (float, int)) else str(loss)

        msg = (
            f"Epoch {self._epoch + 1}/{self.total_epochs} "
            f"{bar} {self._batch}/{self.batches_per_epoch} "
            f"- loss={loss_str}"
        )
        self.log_fn(msg)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        parts = [
            f"[Train] Epoch {epoch + 1}/{self.total_epochs} finished.",
        ]
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        self.log_fn(" ".join(parts))
        
    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _make_bar(self, frac: float) -> str:
        filled = int(math.floor(frac * self.bar_width))
        filled = max(0, min(self.bar_width, filled))
        # leave room for '>'
        if filled == self.bar_width:
            bar = "=" * self.bar_width
        else:
            bar = "=" * filled + ">" + "." * (self.bar_width - filled - 1)
        return f"[{bar}]"

    def __deepcopy__(self, memo):
        # Keras‐Tuner will copy callbacks; Qt signals can't be
        # so just return self
        return self
    

class GuiStdout(io.TextIOBase):
    def __init__(self, log_fn):
        self.log_fn = log_fn
        self._orig = sys.stdout

    def write(self, s):
        # Keras often writes partial lines, so we buffer or just forward raw
        if s.strip():
            self.log_fn(s.rstrip("\n"))
        return len(s)

    def flush(self):
        pass

@contextlib.contextmanager
def capture_keras_stdout(log_fn):
    
    # with capture_keras_stdout(log):
    # history = subs_model_inst.fit(
    #     train_dataset,
    #     validation_data=val_dataset,
    #     epochs=EPOCHS,
    #     callbacks=callbacks_without_GuiProgress,
    #     verbose=1,   # KERAS bar
    # )

    orig = sys.stdout
    sys.stdout = GuiStdout(log_fn)
    try:
        yield
    finally:
        sys.stdout = orig

class _GuiProgress(Callback):
    """
    Map Keras training progress to a GUI progress callback.

    Parameters
    ----------
    total_epochs : int
        Total number of epochs scheduled for training.

    batches_per_epoch : int
        Number of batches per epoch. Used to compute fine-grained
        progress when ``epoch_level=False``.

    update_fn : callable
        Function taking an integer percentage 0–100, e.g.
        ``update_fn(42)`` meaning "42% done". In the GeoPrior GUI this
        is typically a thin wrapper around ProgressManager that also
        updates the status text.

    epoch_level : bool, default=False
        If ``True``, update progress once per epoch (0, 1/NE, 2/NE,
        ...). If ``False``, update after every batch using the total
        number of batches (smoother progress bar).
    """

    def __init__(
        self,
        total_epochs: int,
        batches_per_epoch: int,
        update_fn: PctUpdateFn,
        epoch_level: bool = False,
    ) -> None:
        super().__init__()
        self.total_epochs = max(1, int(total_epochs))
        self.batches_per_epoch = max(1, int(batches_per_epoch))
        self.update_fn = update_fn
        self.epoch_level = bool(epoch_level)

        self._seen_batches: int = 0
        self._last_pct: int = -1  # ensure 0 is emitted on begin

    # ------------------------------------------------------------------
    # Keras hooks
    # ------------------------------------------------------------------
    def on_train_begin(self, logs: Mapping[str, Any] | None = None) -> None:
        self._seen_batches = 0
        self._last_pct = -1
        self._emit_pct(0)

    def on_batch_end(
        self, batch: int, logs: Mapping[str, Any] | None = None
    ) -> None:
        if self.epoch_level:
            # Defer updates to on_epoch_end in epoch-level mode
            return

        self._seen_batches += 1
        total_batches = self.total_epochs * self.batches_per_epoch
        if total_batches <= 0:
            return

        frac = min(1.0, max(0.0, self._seen_batches / float(total_batches)))
        pct = int(round(frac * 100))
        self._emit_pct(pct)

    def on_epoch_end(
        self, epoch: int, logs: Mapping[str, Any] | None = None
    ) -> None:
        if not self.epoch_level:
            return

        # epoch is 0-based, progress is (epoch+1)/total_epochs
        frac = min(1.0, max(0.0, (epoch + 1) / float(self.total_epochs)))
        pct = int(round(frac * 100))
        self._emit_pct(pct)

    def on_train_end(
        self, logs: Mapping[str, Any] | None = None
    ) -> None:
        # Ensure we end at 100%
        self._emit_pct(100)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _emit_pct(self, pct: int) -> None:
        pct = max(0, min(100, int(pct)))
        if pct == self._last_pct:
            return
        self._last_pct = pct
        try:
            self.update_fn(pct)
        except Exception:
            # Never break training because the GUI failed to update.
            pass
        
    def __deepcopy__(self, memo):
        # Keras‐Tuner will copy callbacks; Qt signals can't be
        # so just return self
        return self

class GuiEpochLogger(Callback):
    """
    Log per-epoch metrics via a logger instead of stdout.

    Parameters
    ----------
    log_fn : callable
        Function taking a string message (e.g. Qt status logger).

    keys : sequence of str, optional
        Preferred order of metric keys in the log line. Any metrics
        present in `logs` but not in `keys` will be appended afterwards.
        Defaults to ("loss", "val_loss").
    """

    def __init__(
        self,
        log_fn: LogFn,
        keys: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self._log = log_fn or (lambda msg: None)
        self._keys = list(keys) if keys is not None else ["loss", "val_loss"]

    def on_train_begin(self, logs: Mapping[str, Any] | None = None) -> None:
        self._log("[Training] Started model fitting...")

    def on_epoch_end(
        self, epoch: int, logs: Mapping[str, Any] | None = None
    ) -> None:
        logs = logs or {}
        # Order metrics: preferred keys first, then everything else.
        ordered_keys: list[str] = []
        for k in self._keys:
            if k in logs and k not in ordered_keys:
                ordered_keys.append(k)
        for k in logs.keys():
            if k not in ordered_keys:
                ordered_keys.append(k)

        parts: list[str] = []
        for k in ordered_keys:
            v = logs.get(k)
            if v is None:
                continue
            try:
                parts.append(f"{k}={float(v):.4f}")
            except (TypeError, ValueError):
                parts.append(f"{k}={v}")

        msg = f"Epoch {epoch + 1}: " + ", ".join(parts) if parts else \
            f"Epoch {epoch + 1}"
        self._log(msg)

    def on_train_end(self, logs: Mapping[str, Any] | None = None) -> None:
        self._log("[Training] Finished model fitting.")

    def __deepcopy__(self, memo):
        # Keras‐Tuner will copy callbacks; Qt signals can't be
        # so just return self
        return self

class GuiEarlyStopping(EarlyStopping):
    """
    EarlyStopping that logs a summary message to a GUI logger.

    Parameters
    ----------
    log_fn : callable, optional
        Logger function receiving a final summary message at the end
        of training. If ``None``, no additional logging is performed.

    **kwargs :
        All additional keyword arguments are forwarded to
        :class:`tensorflow.keras.callbacks.EarlyStopping`.
    """

    def __init__(
        self,
        *args: Any,
        log_fn: LogFn | None = None,
        **kwargs: Any,
    ) -> None:
        self._log = log_fn or (lambda msg: None)
        super().__init__(*args, **kwargs)

    def on_train_end(self, logs: Mapping[str, Any] | None = None) -> None:
        super().on_train_end(logs)

        # The base class sets self.stopped_epoch when a stop happens, and
        # tracks the best value of the monitored quantity in self.best.
        if getattr(self, "stopped_epoch", 0) > 0:
            # +1 because Keras epochs are 0-based internally.
            try:
                best_val = float(self.best)
                best_str = f"{best_val:.4f}"
            except Exception:
                best_str = str(getattr(self, "best", "unknown"))

            self._log(
                f"[EarlyStopping] Stopped at epoch "
                f"{self.stopped_epoch + 1}; "
                f"best {self.monitor} = {best_str}"
            )
        else:
            # Training ran for all epochs without triggering early stop.
            if hasattr(self, "best"):
                try:
                    best_val = float(self.best)
                    best_str = f"{best_val:.4f}"
                except Exception:
                    best_str = str(self.best)
                self._log(
                    f"[EarlyStopping] Completed all epochs; "
                    f"best {self.monitor} = {best_str}"
                )
                
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
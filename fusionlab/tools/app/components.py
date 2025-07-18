# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>
"""
Centralised, thread‑safe controller for the (ETA‑aware) progress bar used
throughout the Mini‑Subsidence GUI and CLI tools.

* Provide a single, reusable API for **all** workflows (training, tuning, 
  inference, visualisation …).
* Compute and display **ETA** based on wall‑clock time.
* Show optional *context prefixes* such as the current **Epoch** or **Trial**.
* Remain **thread‑safe** – long‑running worker threads may call
  :py:meth:`update` directly; GUI updates always happen in the Qt main thread
  via queued signals.

"""
from __future__ import annotations

from typing import Callable , Optional, List 
import time
import errno
from enum import Enum, auto
from pathlib import Path
from datetime import datetime
import traceback, textwrap, os
import shutil

from PyQt5.QtCore import ( 
    QObject, 
    pyqtSignal, 
    Qt, 
    QThread, 
    QMetaObject, 
    Q_ARG, 
    QEvent,
    pyqtSlot, 
)
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QDialog, 
    QTextEdit,
    QVBoxLayout, 
    QHBoxLayout, 
    QFileDialog, 
    QApplication, 
    QPushButton, 
    QMessageBox, 
    QProgressBar, 
    QLabel, 
    QMainWindow,
    QWidget, 
    QFrame,
    QCheckBox, 
    QPlainTextEdit
)

from ...nn import KERAS_DEPS  
from ...registry import ManifestRegistry, _locate_manifest
from .styles import ( 
    LOG_STYLES, 
    PRIMARY, 
    INFERENCE_OFF, 
    SECONDARY
)

Callback = KERAS_DEPS.Callback 


__all__ = [
    "ProgressManager", "WorkerController",
    "ErrorManager", "ExitController", 
    "ModeSwitch", "TunerProgress", 
    "TuningProgress", "ResetController", 
    "Mode", "ModeManager", "ManifestManager", 
    "LogManager", 
  ]


class ProgressManager(QObject):
    """Thread‑safe progress‑bar wrapper with ETA & contextual prefixes."""

    # Internal *queued* signals – always executed on the GUI thread.
    # They are emitted from worker threads, so *QueuedConnection* is used
    # automatically by PyQt when the sender != receiver thread.

    _sig_set_val   = pyqtSignal(int)   # → QProgressBar.setValue(int)
    _sig_set_fmt   = pyqtSignal(str)   # → QProgressBar.setFormat(str)
    _sig_set_label = pyqtSignal(str)   # → QLabel.setText(str)
    _sig_set_pct   = pyqtSignal(str)   # NEW – percentage label text
    
    def __init__(
        self,
        progress_bar: QProgressBar,
        text_label: QLabel,
        pct_label:  QLabel | None = None,
    ) -> None:
        super().__init__(progress_bar)  # QObject needs a parent
        self._bar   = progress_bar
        self._lbl   = text_label
        self._pct_lbl    = pct_label 

        # Wire signals to the widgets – *direct* since widgets live in the
        # same thread as *self* (GUI thread).
        self._sig_set_val.connect(self._bar.setValue, Qt.QueuedConnection)
        self._sig_set_fmt.connect(self._bar.setFormat, Qt.QueuedConnection)
        self._sig_set_label.connect(self._lbl.setText, Qt.QueuedConnection)

        if self._pct_lbl is not None:                           # NEW
           self._sig_set_pct.connect(
               self._pct_lbl.setText, Qt.QueuedConnection)


        # Internal state ------------------------------------------------
        # self._step_start_t: float | None = None
        # self._context_prefix: str = ""  # e.g. "Epoch 3/50 – "
        # self.reset()
        # --- context fields instead of one single prefix ---
        self._trial_prefix = ""
        self._epoch_prefix = ""
        self._step_start_t = None
        self.reset()
        

    def start_step(self, name: str, total: int | None = None) -> None:
        """Begin a new logical step (pre‑processing, training, …).

        Parameters
        ----------
        name : str
            Human‑readable name shown in the *label* until first ETA update.
        total : int or None, optional
            If known, you may pass the total number of iterations so the
            first ETA can be estimated immediately.  Leave *None* for
            unknown totals or when the step itself manages percentages.
        """
        self._trial_prefix = ""
        self._epoch_prefix = ""
        
        self._step_start_t = time.time()
        # self._context_prefix = ""  # clear any lingering epoch/trial text

        # Indeterminate bar if *total* is unknown; switches to % later.
        if total is None or total <= 0:
            self._sig_set_fmt.emit("")          # defaults to busy
            self._sig_set_val.emit(0)
        else:
            self._sig_set_fmt.emit("%p%")
            self._sig_set_val.emit(0)
        self._sig_set_label.emit(f"{name}…")

    def set_epoch_context(self, *, epoch: int, total: int) -> None:
        """Remember epoch info but don’t stomp trial info."""
        epoch = min(epoch, total)
        self._epoch_prefix = f"Epoch {epoch}/{total} – "

    def set_trial_context(self, *, trial: int, total: int) -> None:
        """Remember trial info but don’t stomp epoch info."""
        # if trial > total: trial =total 
        # self._context_prefix = f"Trial {trial}/{total} – "
        trial = min(trial, total)
        self._trial_prefix = f"Trial {trial}/{total} – "
        

    def update(self, current: int, total: int) -> None:
        """
        Update progress (thread-safe), combining trial and epoch contexts,
        computing ETA, and emitting UI signals.
    
        Parameters
        ----------
        current : int
            Completed work count.
        total : int
            Total work count; if <= 0, bar is busy.
        """
        # Busy mode – animate bar indefinitely
        if total <= 0:
            self._sig_set_fmt.emit("")  
            self._sig_set_val.emit(0)  
            prefix = self._trial_prefix + self._epoch_prefix
            self._sig_set_label.emit(prefix + "Working…")
            return
    
        # Compute fraction completed and percentage
        frac = max(0.0, min(1.0, current / total))
        pct = int(frac * 100)
    
        # Emit progress value and optional pct label
        self._sig_set_val.emit(pct)
        if self._pct_lbl is not None:
            self._sig_set_pct.emit(f"{pct:3d} %") # right-hand label
    
        # Compute ETA
        if self._step_start_t is None:
            eta_str = "--:--"
        else:
            elapsed = time.time() - self._step_start_t
            eta = (
                (elapsed / frac) - elapsed
                if frac > 1e-6
                else -1.0
            )
            eta_str = self._format_time(eta)
    
        # Combine trial+epoch prefixes and emit final label
        prefix = self._trial_prefix + self._epoch_prefix
        self._sig_set_label.emit(f"{prefix}ETA: {eta_str}")

    def finish_step(self, msg: str = "Done") -> None:
        """Mark the step complete – sets bar to 100 %."""
        self._sig_set_val.emit(100)
        self._sig_set_fmt.emit("%p%")
        fmsg = f"{msg} ✓" if str(msg).lower() !='done' else msg 
        self._sig_set_label.emit(fmsg)
        if self._pct_lbl is not None:               
            self._sig_set_pct.emit("100 %")

    def reset(self, status ="Idle") -> None:
        """Return the bar to an idle state (0 % and blank label)."""
        self._step_start_t = None
        self._context_prefix = ""
        self._sig_set_fmt.emit(f"{status}")
        self._sig_set_val.emit(0)
        self._sig_set_label.emit("")
        if self._pct_lbl is not None:               
            self._sig_set_pct.emit(f"{status}")

 
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Return *HH:MM:SS* (or *MM:SS* if < 1 h)."""
        if seconds < 0 or seconds == float("inf"):
            return "--:--"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class TunerProgress(Callback):
    """
    Drive ``ProgressManager`` during a HydroTuner search.

    Parameters
    ----------
    pm : ProgressManager
        Shared instance that owns the GUI bar + labels.
    max_trials : int
        The ``max_trials`` argument you pass to ``HydroTuner``.
    epochs : int
        Epochs per trial (same value you pass to ``tuner.search``).
    trial_prefix : str, default="Tuning"
        Short text shown while the tuner is active.
    """

    def __init__(
        self,
        pm: "ProgressManager",
        *,
        max_trials: int,
        epochs: int,
        trial_prefix: str = "Tuning",
    ) -> None:
        super().__init__()
        self._pm          = pm
        self._max_trials  = max_trials
        self._epochs      = epochs
        self._trial_prefix= trial_prefix

        # internal bookkeeping
        self._trial_idx   = 0           # 1-based
        self._epoch_idx   = 0
        self._t0_trial    = None        # wall-clock per trial

        # one logical “step” for the whole search -------------------
        self._pm.start_step(trial_prefix, total=max_trials * epochs)

    # Keras-Tuner forwards these Keras hooks to every trial model ----
    def on_trial_begin(self, trial):
        self._trial_idx  = trial.trial_id + 1   # 1 … N
        self._epoch_idx  = 0
        if self._t0_trial is None:
            # Either on_trial_begin() wasn’t called yet (warm-start) or
            # something unexpected happened – initialise it *now*.
            self._t0_trial = time.time()

        self._pm.set_trial_context(
            trial=self._trial_idx, total=self._max_trials
        )

    def on_epoch_end(self, epoch, logs=None):
        self._epoch_idx = epoch + 1
        if self._t0_trial is None:
            # Either on_trial_begin() wasn’t called yet (warm-start) or
            # something unexpected happened – initialise it *now*.
            self._t0_trial = time.time()

        # ---- ETA (within *trial*) ---------------------------------
        elapsed  = time.time() - self._t0_trial
        eta_secs = ((elapsed / self._epoch_idx) *
                    (self._epochs - self._epoch_idx))
        eta_secs = max(0, eta_secs)

        # ---- global percent (all trials) --------------------------
        done_epochs  = ((self._trial_idx - 1) * self._epochs) + self._epoch_idx
        total_epochs = self._max_trials * self._epochs
        self._pm.update(done_epochs, total_epochs)   # << bar + % + ETA text

    def on_trial_end(self, trial):
        # Snap to next integer % in case the last epoch didn't land exactly
        done_epochs  = self._trial_idx * self._epochs
        total_epochs = self._max_trials * self._epochs
        self._pm.update(done_epochs, total_epochs)

    def on_search_end(self, logs=None):
        self._pm.finish_step(f"{self._trial_prefix} ✓")
        
    def __deepcopy__(self, memo):
        """
        Keras-Tuner deep-copies the callbacks list for every trial.
        Qt signal objects cannot be copied, so we just return *self*.
        That is perfectly fine because the tuner resets internal
        callback state at the start of each trial.
        """
        return self
    
class TuningProgress(Callback):
    """Custom callback to update progress during hyperparameter tuning."""
    
    def __init__(self, pm: ProgressManager, max_trials: int, epochs: int):
        """
        Initialize the progress update for tuning.
        
        Parameters
        ----------
        pm : ProgressManager
            The shared instance that controls the progress bar.
        max_trials : int
            The total number of trials in the tuning process.
        epochs : int
            The number of epochs per trial.
        """
        self._pm = pm
        self.max_trials = max_trials
        self.epochs = epochs
        self.current_trial = 0
        self.current_epoch = 0
        self.start_time = None  # To calculate ETA for each trial

    def on_trial_begin(self, trial):
        """Called at the beginning of each trial."""
        self.current_trial = trial.trial_id + 1  # 1-based index
        self.current_epoch = 0
        self.start_time = time.time()  # Reset the timer at the start of each trial
        
        # Update progress bar with the trial context
        self._pm.set_trial_context(trial=self.current_trial, total=self.max_trials)
        self._update_progress_bar()

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        self.current_epoch = epoch + 1  # 1-based index
        self._update_progress_bar()

    def on_trial_end(self, trial):
        """Called at the end of each trial."""
        self._update_progress_bar()
    
    def _update_progress_bar(self):
        """Update the progress bar with current trial, epoch, and ETA."""
        # elapsed_time = time.time() - self.start_time if self.start_time else 0
        # eta = (elapsed_time / self.current_epoch) * (
        #     self.epochs - self.current_epoch) if self.current_epoch > 0 else 0
        # eta_str = self._format_eta(eta)

        # Calculate global progress
        completed_epochs = (self.current_trial - 1) * self.epochs + self.current_epoch
        total_epochs = self.max_trials * self.epochs
        # progress = (completed_epochs / total_epochs) * 100
        
        # Update the progress bar
        self._pm.update(completed_epochs, total_epochs)
        
        # Update the trial and ETA information
        self._pm.set_trial_context(
            trial=self.current_trial, total=self.max_trials)
        # self._pm.set_epoch_context(
        #     epoch=self.current_epoch, total=self.epochs)
        # self._pm.set_label(
        #     f"Trial {self.current_trial}/{self.max_trials} - ETA: {eta_str}")
        
    @staticmethod
    def _format_eta(seconds: float) -> str:
        """Format ETA to HH:MM:SS."""
        if seconds < 0 or seconds == float("inf"):
            return "--:--"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    
    def __deepcopy__(self, memo):
        """
        Keras-Tuner deep-copies the callbacks list for every trial.
        Qt signal objects cannot be copied, so we just return *self*.
        That is perfectly fine because the tuner resets internal
        callback state at the start of each trial.
        """
        return self


class WorkerController(QObject):
    """
    Central Stop-button handler with Yes/No confirmation.

    • Keeps a reference to the *current* QThread worker  
    • Shows a modal *“Stop workflow?”* dialog  
    • Safely interrupts the worker (requestInterruption)  
    • Disables / re-enables the button automatically  
    • Emits *stopped* when the thread has actually finished
    """

    stopped = pyqtSignal()

    def __init__(
        self,
        stop_button: QPushButton,
        *,
        parent,                               # needed for QMessageBox
        log_fn:     Callable[[str], None] | None = None,
        status_fn:  Callable[[str], None] | None = None,
        confirm:    bool = True,                  # turn dialog on/off
    ) -> None:
        super().__init__(stop_button)
        self._btn         = stop_button
        self._parent_gui  = parent
        self._log         = log_fn or (lambda *_: None)
        self._set_status  = status_fn or (lambda *_: None)
        self._confirm     = confirm

        self._worker: QThread | None = None

        self._btn.setEnabled(False)
        self._btn.clicked.connect(self._on_stop)

    def bind(self, worker: QThread | None) -> None:
        """Call each time you launch a new QThread."""
        if self._worker is not None:
            try:
                self._worker.finished.disconnect(self._on_finished)
            except TypeError:
                pass
            # error_occurred is optional → guard with hasattr
            if hasattr(self._worker, "error_occurred"):
                try:
                    self._worker.error_occurred.disconnect(self._on_error)
                except TypeError:
                    pass

        self._worker = worker

        if worker is None:
            self._btn.setEnabled(False)
            return

        worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        if hasattr(worker, "error_occurred"):
            worker.error_occurred.connect(
                self._on_error, Qt.QueuedConnection)
            
        self._btn.setEnabled(True)

    # slots
    @pyqtSlot()
    def _on_finished(self):
        self._cleanup()
    
    @pyqtSlot(str)
    def _on_error(self, _msg: str):
        """Called when the worker emits *error_occurred*."""
        self._log("⚠ workflow aborted due to an error")
        self._cleanup()
        
    def _on_stop(self):
        if self._worker is None or not self._worker.isRunning():
            self._btn.setEnabled(False)
            return

        # optional confirmation box 
        if self._confirm:
            reply = QMessageBox.question(
                self._parent_gui,
                "Confirm Stop",
                "Are you sure you want to stop the current workflow?\n"
                "This action cannot be undone.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return  # user cancelled

        # proceed with interruption 
        self._log("⏹ stop requested — will interrupt at next safe point")
        self._set_status("⏹ Stopping…")
        self._worker.requestInterruption()
        self._btn.setEnabled(False)

    def _cleanup(self):
       """Common path for both normal end and error."""
       self._btn.setEnabled(False)
       self._set_status("⚪ Idle")
       self._worker = None
       self.stopped.emit()
       
    def is_busy(self) -> bool:
        """
        True while a worker thread is bound *and* still running.
        ExitController queries this before quitting.
        """
        return bool(self._worker and self._worker.isRunning())
       

class ErrorManager(QObject):
    """
    Centralised error-popup helper.

    • report(msg, exc=None)  → shows a modal box, logs & emits *handled*
    """
    handled = pyqtSignal()            # emitted after the dialog is closed

    def __init__(
        self,
        parent_gui,                   # used as dialog parent
        log_fn : Callable[[str], None],
        *,
        max_width = 580,
        default_title: str = "Workflow Error",
    ):
        super().__init__(parent_gui)
        self._parent  = parent_gui
        self._log     = log_fn
        self._last_msg: str | None = None
        self._max_w   = max_width
        self._default_ttl  = default_title 

    # API
    def report(self, message: str, exc: Optional[BaseException] = None):
        """Slot-friendly wrapper; safe to call from any thread."""
        # Avoid spamming identical messages multiple times in the log
        if message == self._last_msg:
            return
        self._last_msg = message
        txt = textwrap.fill(message, 80)
        if exc:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            txt += "\n\n" + tb
        self._log(f"❌ {self._default_ttl}: {message}")
        # QApplication.invokeMethod(
        #     self, "_popup", Qt.QueuedConnection, txt
        # )
        QMetaObject.invokeMethod(
           self,
           "_popup",
           Qt.QueuedConnection,
           Q_ARG(str, txt)
       )

    @pyqtSlot(str)
    def _popup(self, full_text: str):
        """Runs in the GUI thread – create & exec_() the modal dialog."""
        dlg = QDialog(self._parent)
        dlg.setObjectName("errorDialog") 
        dlg.setWindowTitle(f"{self._default_ttl}")
        dlg.setModal(True)
        # dlg.setMinimumWidth(self._max_w)

        L = QVBoxLayout(dlg)
        # — icon + short text —
        top = QHBoxLayout()
        icon_lbl = QLabel("🛑")
        icon_lbl.setStyleSheet("font-size:26px;")
        top.addWidget(icon_lbl, 0, Qt.AlignTop)

        msg_lbl = QLabel("<b>An error occurred.</b><br>"
                         "You can copy the details to the clipboard or save "
                         "them to a file.")
        msg_lbl.setWordWrap(True)
        top.addWidget(msg_lbl, 1)
        L.addLayout(top)

        # — details pane —
        txt = QTextEdit(full_text)
        txt.setReadOnly(True)
        txt.setMinimumHeight(160)
        L.addWidget(txt)

        # — buttons —
        row = QHBoxLayout()
        row.addStretch(1)

        def _copy():
            try:
                QApplication.clipboard().setText(full_text)
                self._log("ℹ traceback copied to clipboard")
            except Exception as err:
                self._log(f"⚠ clipboard copy failed ({err})")

        def _save():
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            suggest = f"error_{ts}.txt"
            path, _ = QFileDialog.getSaveFileName(
                dlg, "Save error log", suggest, "Text files (*.txt)"
            )
            if path:
                try:
                    with open(path, "w", encoding="utf-8") as fp:
                        fp.write(full_text)
                    self._log(f"📝 traceback saved to {os.path.abspath(path)}")
                except Exception as err:
                    self._log(f"⚠ could not save file ({err})")

        for label, slot in (
            ("Copy", _copy),
            ("Save…", _save),
            ("Close", dlg.accept),
        ):
            b = QPushButton(label); b.clicked.connect(slot); row.addWidget(b)

        L.addLayout(row)
        # ------------------------------------------------------------------
        # *** let Qt size the dialog automatically ***
        # ------------------------------------------------------------------
        dlg.adjustSize()                       # sizeHint from the layouts
        # cap width to something reasonable (optional)
        if dlg.width() > self._max_w:
            dlg.resize(self._max_w, dlg.height())
     
        dlg.exec_()
        self.handled.emit()

class ExitController(QObject):
    """
    Reusable *Quit* button / close-event helper.

    Parameters
    ----------
    quit_button : QPushButton
        The button the user clicks to quit.  If you pass *None* the
        controller only handles the window's title-bar close button.
    parent_gui : QMainWindow
        Main window – used as parent for modal QMessageBoxes **and**
        to install an event-filter that intercepts closeEvent().
    worker_ctl : WorkerController | None
        If given, we check `worker_ctl.is_busy()` to warn the user that a
        workflow is still running.
    pre_quit_hook : Callable[[], None] | None
        Optional clean-up function executed *after* the user confirms but
        *before* we call ``QApplication.quit()``.
    """

    def __init__(
        self,
        quit_button:   QPushButton | None,
        *,
        parent:    QMainWindow,
        worker_ctl:    "WorkerController | None" = None,
        pre_quit_hook: Callable[[], None] | None = None,
        log_fn:        Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._wnd   = parent
        self._busy_fn     = worker_ctl.is_busy if worker_ctl else (lambda: False)
        self._stop_worker   = worker_ctl._on_stop if worker_ctl else (
            lambda: None)
        self._pre   = pre_quit_hook or (lambda: None)
        self._log   = log_fn or (lambda *_: None)
        
        self._wc          = worker_ctl 
        
        # 1) optional push-button
        if quit_button:
            quit_button.clicked.connect(self._maybe_quit)

        # 2) intercept title-bar close
        self._wnd.installEventFilter(self)

    # -----------------------------------------------------------------
    # centralised confirmation dialog
    def _maybe_quit(self):
        busy = self._busy_fn()  
        if busy:
            ttl = "Stop workflow & quit?"
            txt = ("A training / inference run is still in progress.\n\n"
                   " •  Stop the workflow and quit the application\n"
                   " •  or cancel and let it finish")
        else:
            ttl = "Quit application?"
            txt = "Do you really want to quit Fusionlab-learn?"

        reply = QMessageBox.question(
            self._wnd, ttl, txt,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return                                          # user cancelled

        if busy and self._wc:                                   # graceful stop
            self._stop_worker()                                 # ask WorkerController
            conn_type = (
                Qt.SingleShotConnection         # PyQt ≥ 5.15
                 if hasattr(Qt, "SingleShotConnection")
                 else Qt.QueuedConnection
            )       # fallback
            self._wc.stopped.connect(self._final_quit, conn_type)
        else: 
            self._final_quit()

    def _final_quit(self):
        self._pre()               # optional user clean-up
        self._log("⇧ quitting…")
        QApplication.quit()
        
    #   QOBJECT EVENT-FILTER → catch title-bar close 
    def eventFilter(self, watched, ev):
        if watched is self._wnd and ev.type() == QEvent.Close:
            ev.ignore()            # we’ll handle it
            self._maybe_quit()
            return True
        return super().eventFilter(watched, ev)


class ModeSwitch(QObject):
    """
    State helper that tints a button while a given QThread
    is running and reverts it automatically when the thread finishes.
    Optionally updates the button text as well.
    """
    def __init__(
        self,
        *,
        button: QPushButton,
        tint: str,
        tooltip_running: str,
        tooltip_idle: str,
        text_running: str = None,
        text_idle:   str = None,
    ) -> None:
        super().__init__(button)
        self._btn          = button
        self._tint         = tint
        self._tt_run       = tooltip_running
        self._tt_idle      = tooltip_idle
        self._text_run     = text_running
        self._text_idle    = text_idle

    def bind(self, worker: Optional[QThread]) -> None:
        """Call every time you launch **or** stop a tuning thread."""
        # detach previous
        try:
            self._worker.finished.disconnect(self._revert)
        except (AttributeError, TypeError):
            pass

        self._worker = worker
        if worker is None:
            return self._revert()

        # activate running look
        self._btn.setStyleSheet(f"background:{self._tint}; color:white;")
        self._btn.setToolTip(self._tt_run)
        if self._text_run is not None:
            self._btn.setText(self._text_run)

        worker.finished.connect(self._revert, Qt.QueuedConnection)

    def _revert(self):
        """Return to the idle look."""
        self._btn.setStyleSheet("")       # reset QSS        
        self._btn.setToolTip(self._tt_idle)
        if self._text_idle is not None:
            self._btn.setText(self._text_idle)


class ResetController(QObject):
    """
    Encapsulates all logic around the 'Reset' action:
      • confirmation prompt
      • cache cleaning vs. archiving
      • button enable/disable, styling, tooltip
      • emits `reset_done` so others can hook in
    """

    reset_done = pyqtSignal()

    def __init__(
        self,
        reset_button: QPushButton,
        *,
        cache_dir: Optional[Path] = None,
        mode: str = "clean",       # "clean" or "archive"
        primary_color: str = "#2E3191",
        disabled_color: str = "#cccccc",
        parent: Optional[QWidget]=None, 
    ):
        super().__init__(parent)

        # determine cache_dir if not provided
        if cache_dir is None:
            registry = ManifestRegistry(session_only=False)
            # persistent_root already ends in '/runs'
            cache_dir = registry.persistent_root

        self._btn       = reset_button
        self._cache_dir = Path(cache_dir)
        self.mode       = mode
        self.primary    = primary_color
        self.disabled   = disabled_color

        # initial state: disabled until idle
        self._btn.setEnabled(False)
        self._btn.setStyleSheet(f"background:{self.disabled};")
        self._btn.setToolTip("Nothing to reset")
        self._btn.clicked.connect(self._on_clicked)

    def enable(self):
        """Enable the Reset button (when GUI is idle)."""
        self._btn.setEnabled(True)
        self._btn.setStyleSheet(f"background:{self.primary}; color:white;")
        verb = "clean cache" if self.mode=="clean" else "archive cache"
        self._btn.setToolTip(f"Reset UI & {verb}")

    def disable(self):
        """Disable the Reset button (when workflow is running)."""
        self._btn.setEnabled(False)
        self._btn.setStyleSheet(f"background:{self.disabled};")
        self._btn.setToolTip("Cannot reset while running")

    def _on_clicked(self):
        """Ask for confirmation, then clean or archive the cache."""
        verb = "archive" if self.mode=="archive" else "clean"
        reply = QMessageBox.question(
            None,
            "Confirm Reset",
            f"Are you sure you want to reset the UI and {verb}?\n"
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        if self._cache_dir.exists():
            if self.mode == "clean":
                self._clean_cache()
            else:
                self._archive_cache()

        # notify listeners to clear forms, logs, progress, etc.
        self.reset_done.emit()

    def _clean_cache(self):
        """Delete all files and folders under the cache directory."""
        for child in self._cache_dir.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            except Exception:
                pass

    def _archive_cache(self):
        """
        Move the cache directory to an archive, then recreate it.
        e.g. 'runs' → 'runs_archive_20250705_150023'
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = self._cache_dir.parent / f"{self._cache_dir.name}_archive_{ts}"
        try:
            shutil.move(str(self._cache_dir), str(archive))
        except Exception:
            pass
        self._cache_dir.mkdir(parents=True, exist_ok=True)

class LogManager(QObject):
    """
    Central log buffer + widget + on-disk dumping with UI 
    trimming and optional collapse mode.

    Modes:
      - "default": full append, trimmed to _max_ui_lines
      - "collapse": identical consecutive lines are
        collapsed with a count suffix
    """
    line_appended = pyqtSignal(str)

    def __init__(
        self,
        log_widget: QPlainTextEdit,
        mode: str = 'collapse', #"default",
        *,
        cache_limit: int = 10_000,
        ui_limit: int = 500,
        log_dir_name: str = "_log",
        parent: Optional[QObject] = None
    ):
        super().__init__(parent)
        self._widget = log_widget
        self._widget.setObjectName("logWidget")
        # apply the CSS we defined above
        self._widget.setStyleSheet(LOG_STYLES)
        
        self._cache: list[str] = []
        self._limit = cache_limit
        self._ui_limit = ui_limit
        self._log_dir = log_dir_name
        self.mode = mode
        # For collapse mode
        self._last_line: Optional[str] = None
        self._repeat_count = 0

    def append(self, msg: str) -> None:
        """Add one line (no timestamp), keep scroll at bottom."""
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        # 1) full in-memory cache
        self._cache.append(line)
        if len(self._cache) > self._limit:
            self._cache.pop(0)

        # 2) UI display
        if self.mode == "collapse":
            if line == self._last_line:
                self._repeat_count += 1
                display = f"{line}  (×{self._repeat_count})"
                cursor = self._widget.textCursor()
                cursor.movePosition(cursor.End)
                cursor.select(cursor.BlockUnderCursor)
                cursor.removeSelectedText()
                cursor.insertText(display)
            else:
                self._last_line = line
                self._repeat_count = 1
                self._widget.appendPlainText(line)
        else:
            self._widget.appendPlainText(line)

        # 3) UI trimming
        doc = self._widget.document()
        blocks = doc.blockCount()
        if blocks > self._ui_limit:
            cursor = self._widget.textCursor()
            cursor.movePosition(cursor.Start)
            for _ in range(blocks - self._ui_limit):
                cursor.select(cursor.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()
            self._widget.setTextCursor(cursor)

        # 4) scroll to bottom
        sb = self._widget.verticalScrollBar()
        sb.setValue(sb.maximum())
        
        # # *also* ensure the cursor is at the end and visible
        c = self._widget.textCursor()
        c.movePosition(QTextCursor.End)
        self._widget.setTextCursor(c)
        self._widget.ensureCursorVisible()

        # 5) signal
        self.line_appended.emit(line)

    def save_cache(self, run_output_path: str) -> Path:
        """
        Dump the cached lines to disk under
        <run_output_path>/<self._log_dir>/gui_log_<timestamp>.txt
        Returns the file path.
        """
        out_dir = Path(run_output_path) / self._log_dir
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        fname = out_dir / f"gui_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(fname, "w", encoding="utf-8") as fp:
            fp.write("\n".join(self._cache))
        return fname

    def clear(self) -> None:
        """Clear both UI and in-memory cache."""
        self._cache.clear()
        self._last_line = None
        self._repeat_count = 0
        self._widget.clear()

class _LogManager(QObject):
    """Central log buffer + widget + on-disk dumping."""
    # emitted whenever a new line is appended
    line_appended = pyqtSignal(str)

    def __init__(
        self,
        log_widget: QTextEdit,
        *,
        cache_limit: int = 10_000,
        log_dir_name: str = "_log",
        parent: Optional[QObject] = None
    ):
        super().__init__(parent)
        self._widget     = log_widget
        self._widget.setObjectName("logWidget")
        # apply the CSS we defined above
        self._widget.setStyleSheet(LOG_STYLES)

        self._cache      = []        # type: list[str]
        self._limit      = cache_limit
        self._log_dir    = log_dir_name

    def append(self, msg: str) -> None:
        """Add one line (no timestamp), keep scroll at bottom."""
        ts   = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._cache.append(line)
        if len(self._cache) > self._limit:
            self._cache.pop(0)
    
        # push to QTextEdit
        self._widget.append(line)
    
        # scroll to bottom via scrollbar
        sb = self._widget.verticalScrollBar()
        sb.setValue(sb.maximum())
    
        # *also* ensure the cursor is at the end and visible
        c = self._widget.textCursor()
        c.movePosition(QTextCursor.End)
        self._widget.setTextCursor(c)
        self._widget.ensureCursorVisible()
    
        # signal for anyone else listening
        self.line_appended.emit(line)

    def save_cache(self, run_output_path: str) -> Path:
        """
        Dump the cached lines to disk under
        <run_output_path>/<self._log_dir>/gui_log_<timestamp>.txt
        Returns the file path.
        """
        out_dir = Path(run_output_path) / self._log_dir
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        fname = out_dir / f"gui_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(fname, "w", encoding="utf-8") as fp:
            fp.write("\n".join(self._cache))
        return fname

class Mode(Enum):
    TRAIN   = auto()
    INFER   = auto()
    TUNER   = auto()
    DRY_RUN = auto()


class ModeManager(QObject):
    """
    Centralizes UI state across different application modes:
      • TRAIN   — full training workflow
      • INFER   — inference on existing model
      • TUNER   — hyperparameter tuning
      • DRY_RUN — no-op, UI only
    """
    mode_changed = pyqtSignal(Mode)

    def __init__(
        self,
        *,
        run_button:    QPushButton,
        tune_button:   QPushButton,
        infer_button:  QPushButton,
        stop_button:   QPushButton,
        panels:        List[QFrame],
        mode_badge: QLabel,
        parent:        Optional[QObject] = None
    ):
        super().__init__(parent)
        self._run   = run_button
        self._tune  = tune_button
        self._infer = infer_button
        self._stop  = stop_button
        self._panels = panels
        self._badge = mode_badge
        self.mode   = Mode.TRAIN

        # initialize hooks
        self._infer.clicked.connect(self.toggle_inference)
        self._tune.clicked.connect(self.toggle_tuning)

        # Initialize UI to TRAIN mode
        self._apply_mode()

    def toggle_inference(self):
        if self.mode == Mode.INFER:
            self.set_mode(Mode.TRAIN)
        else:
            self.set_mode(Mode.INFER)

    def toggle_tuning(self):
        if self.mode == Mode.TUNER:
            self.set_mode(Mode.TRAIN)
        else:
            self.set_mode(Mode.TUNER)

    def set_mode(self, m: Mode):
        self.mode = m
        self._apply_mode()
        self.mode_changed.emit(m)

    def enable_infer(self, allowed: bool):
        self._infer.setEnabled(allowed)
        if not allowed and self.mode == Mode.INFER:
            self.set_mode(Mode.TRAIN)

    def enable_tune(self, allowed: bool):
        self._tune.setEnabled(allowed)
        if not allowed and self.mode == Mode.TUNER:
            self.set_mode(Mode.TRAIN)

    def _apply_mode(self):
        """Update widget states, labels, styles,
        and grey-out titles on INFER."""
        # Reset panels
        for p in self._panels:
            p.setEnabled(True)
            p.setProperty("inferenceMode", False)
            p.style().unpolish(p); p.style().polish(p)
            # reset title color
            title = p.findChild(QLabel, 'cardTitle')
            if title:
                title.setStyleSheet("")

        self._infer.setEnabled(True)
        self._stop.setEnabled(False)

        if self.mode == Mode.TRAIN:
            self._run.setText("Run")
            self._run.setToolTip("Launch training workflow")
            self._run.setStyleSheet(
                f"background-color: {PRIMARY}; color: white;")

            self._infer.setText("Inference")
            self._infer.setToolTip("Switch to inference mode")
            self._infer.setStyleSheet(
                f"background-color: {INFERENCE_OFF}; color: white;")

        elif self.mode == Mode.INFER:
            for p in self._panels:
                p.setEnabled(False)
                p.setProperty("inferenceMode", True)
                p.style().unpolish(p); p.style().polish(p)
                # grey-out title label inside panel
                title = p.findChild(QLabel, 'cardTitle')
                if title:
                    title.setStyleSheet("color: #888888;")

            self._run.setText("Infer")
            self._run.setToolTip("Launch inference workflow")
            self._run.setStyleSheet(
                f"background-color: {PRIMARY}; color: white;")

            self._infer.setText("Training")
            self._infer.setToolTip("Switch back to training mode")
            self._infer.setStyleSheet(
                f"background-color: {INFERENCE_OFF}; color: white;")

        elif self.mode == Mode.TUNER:
            # for p in self._panels:
            #     p.setEnabled(False)
            self._run.setText("Tune")
            self._run.setToolTip("Launch hyper-parameter tuning")
            self._run.setStyleSheet("background-color: gray; color: white;")
            self._infer.setEnabled(False)
            self._stop.setEnabled(True)

        elif self.mode == Mode.DRY_RUN:
            self._run.setText("Dry Run")
            self._run.setToolTip("Execute UI-only dry run")
            self._run.setStyleSheet("background-color: teal; color: white;")
            self._infer.setEnabled(False)
            self._stop.setEnabled(False)

        # finally, update the badge
        self._update_mode_badge(self.mode)
   
    def _update_mode_badge(self, mode: Mode):
        """Update the little mode badge’s text, color & hover tooltip."""
        mapping = {
            Mode.TRAIN:   (
                "TRAINING", PRIMARY,   
                "You are in TRAINING mode: run new models."),
            Mode.INFER:   (
                "INFER", "#00aa00", # 
                "You are in INFERENCE mode: apply existing models."),
            Mode.TUNER:   (
                "TUNING", SECONDARY,      #"#888888"
                "You are in TUNING mode: search for best hyperparameters."),
            Mode.DRY_RUN: (
                "DRY RUN", "teal",        
                "You are in DRY-RUN mode: no real execution."),
        }
        text, color, tip = mapping[mode]
        self._badge.setText(text)
        self._badge.setStyleSheet(f"""
            background-color: {color};
            color: white;
            border-radius: 4px;
            padding: 4px 6px;
            font-size: 10px;
        """)
        self._badge.setToolTip(tip)

    
class ManifestManager(QObject):
    """
    Manage detection and UI integration for training/inference manifests.

    - Uses the registry to locate both run_manifest.json and tuner_run_manifest.json
    - Updates the infer-button's enabled state and styling
    - Warns the user if inference is attempted without any available model
    - When multiple manifests exist, prompts the user to choose one
    """
    def __init__(
        self,
        infer_button: QPushButton,
        *,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._infer_btn = infer_button
        self._locate = _locate_manifest
        self._primary = PRIMARY
        self._disabled = INFERENCE_OFF

        # guard clicks
        self._infer_btn.clicked.connect(self._on_infer_clicked)
        # initial detection
        self.refresh()

    def refresh(self) -> None:
        """
        Re-scan for manifests using the registry helper.
        """
        manifest, tuner = self._locate(locate_both=True)
        self._run_manifest = Path(manifest) if manifest else None
        self._tuner_manifest = Path(tuner) if tuner else None

        allowed = bool(self._run_manifest or self._tuner_manifest)
        self._infer_btn.setEnabled(allowed)
        color = self._primary if allowed else self._disabled
        self._infer_btn.setStyleSheet(
            f"background-color: {color}; color: white;")

    def pick_manifest(self, theme ='light') -> Optional[str]:
        """
        If both run and tuner manifests exist, show choice dialog.
        Returns the chosen manifest path or None if cancelled.
        """
        self.refresh() 
        if not (self._run_manifest or self._tuner_manifest):
            return None
        if self._run_manifest and not self._tuner_manifest:
            return str(self._run_manifest)
        if self._tuner_manifest and not self._run_manifest:
            return str(self._tuner_manifest)
        
        # both exist: ask the user
        from .dialog import ModelChoiceDialog
        
        dlg = ModelChoiceDialog(
            theme=theme, parent=self._infer_btn.parentWidget())
        choice = dlg.choice()
        if choice == 'train':
            return str(self._run_manifest)
        if choice == 'tuned': 
            return str(self._tuner_manifest)
        return None

    def _on_infer_clicked(self):
        if not (self._run_manifest or self._tuner_manifest):
            QMessageBox.warning(
                self._infer_btn.parentWidget(),
                "No Model Available",
                "No trained or tuned model found. Please run training first."
            )
 
class DryRunController(QObject):
    """
    Wire a QCheckBox into ModeManager to switch in and out of DRY_RUN.
    """
    def __init__(
        self,
        checkbox: QCheckBox,
        mode_mgr: ModeManager,
        *,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._cb      = checkbox
        self._mode_mgr = mode_mgr

        # when user toggles the box, enter or leave DRY_RUN
        self._cb.stateChanged.connect(self._on_state_change)

        # initial state is unchecked → TRAIN
        self._cb.setChecked(False)

    def _on_state_change(self, state: int):
        if state == Qt.Checked:
            self._mode_mgr.set_mode(Mode.DRY_RUN)
        else:
            # if we're in DRY_RUN, go back to TRAIN
            if self._mode_mgr.mode == Mode.DRY_RUN:
                self._mode_mgr.set_mode(Mode.TRAIN)

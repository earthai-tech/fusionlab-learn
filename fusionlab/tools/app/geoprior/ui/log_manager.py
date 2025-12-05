from __future__ import annotations

import time
import errno
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import (
    QObject,
    pyqtSignal,
    Qt,
)
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QProgressBar,
    QLabel,
    QPlainTextEdit,
)

from ..styles import LOG_STYLES


class LogManager(QObject):
    """
    Central log buffer + widget + on-disk dumping with UI
    trimming and optional collapse mode.

    Modes
    -----
    - "default": full append, trimmed to `_ui_limit`
    - "collapse": identical consecutive lines are
      collapsed with a count suffix.
    """

    line_appended = pyqtSignal(str)

    def __init__(
        self,
        log_widget: QPlainTextEdit,
        mode: str = "collapse",
        *,
        cache_limit: int = 10_000,
        ui_limit: int = 500,
        log_dir_name: str = "_log",
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._widget = log_widget
        self._widget.setObjectName("logWidget")
        # Styling is already applied at window level, but this is harmless:
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
        """Add one line (no timestamp in `msg`), keep scroll at bottom."""
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
                cursor.movePosition(QTextCursor.End)
                cursor.select(QTextCursor.BlockUnderCursor)
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
            cursor.movePosition(QTextCursor.Start)
            for _ in range(blocks - self._ui_limit):
                cursor.select(QTextCursor.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()
            self._widget.setTextCursor(cursor)

        # 4) scroll to bottom
        sb = self._widget.verticalScrollBar()
        sb.setValue(sb.maximum())
        c = self._widget.textCursor()
        c.movePosition(QTextCursor.End)
        self._widget.setTextCursor(c)
        self._widget.ensureCursorVisible()

        # 5) signal
        self.line_appended.emit(line)

    def save_cache(self, run_output_path: str | Path) -> Path:
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

        if self._pct_lbl is not None:                           
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

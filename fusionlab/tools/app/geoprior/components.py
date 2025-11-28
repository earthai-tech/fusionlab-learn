from typing import Callable , Optional
import time
import errno

from pathlib import Path
from datetime import datetime
import traceback, textwrap, os

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

from PyQt5.QtWidgets import (
    QWidget,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QHBoxLayout,
    QStackedWidget,
    QLabel,
    QDialog, 
    QTextEdit,
    QVBoxLayout, 
    QFileDialog, 
    QApplication, 
    QPushButton, 
    QMessageBox, 
    QMainWindow,
    QPlainTextEdit
)
from PyQt5.QtGui import QTextCursor
from .styles import LOG_STYLES 

class RangeListEditor(QWidget):
    """
    Small helper widget to edit either a float *range* (min/max)
    or a discrete *list* of values.

    It understands NATCOM `TUNER_SEARCH_SPACE` formats:

    - range: {"type": "float", "min_value": ..., "max_value": ..., "sampling": ...}
    - list:  [v1, v2, v3]

    Parameters
    ----------
    show_sampling : bool, default True
        If False, the "Sampling" controls are hidden and no "sampling"
        key is emitted.
    spin_width : int or None, default None
        If given, both Min/Max spinboxes are forced to this width.
        Use it in the Scalars & losses dialog for nice vertical
        alignment; keep None elsewhere (Dropout, HD factor).
    """

    def __init__(
        self,
        parent=None,
        *,
        min_allowed: float = 0.0,
        max_allowed: float = 1.0,
        decimals: int = 6,
        show_sampling: bool = True,
        spin_width: int | None = None,
    ) -> None:
        super().__init__(parent)

        self._show_sampling = show_sampling
        self._spin_width = spin_width

        # sampling == None  -> "linear" (default)
        # sampling == "log" -> log-scale search
        self._sampling: str | None = None

        # --- Mode selector: Range / List ---
        self.mode = QComboBox(self)
        self.mode.addItems(["Range", "List"])

        # --- Stacked editor: [range page | list page] ---
        self.stack = QStackedWidget(self)

        # Range page
        range_page = QWidget(self)
        range_layout = QHBoxLayout(range_page)
        range_layout.setContentsMargins(0, 0, 0, 0)

        self.min_sb = QDoubleSpinBox(range_page)
        self.max_sb = QDoubleSpinBox(range_page)

        for sb in (self.min_sb, self.max_sb):
            sb.setDecimals(decimals)
            sb.setRange(min_allowed, max_allowed)
            step = (max_allowed - min_allowed) / 100.0 or 0.01
            sb.setSingleStep(step)
            if self._spin_width is not None:
                sb.setMinimumWidth(self._spin_width)
                sb.setMaximumWidth(self._spin_width)

        range_layout.addWidget(QLabel("Min:", range_page))
        range_layout.addWidget(self.min_sb)
        range_layout.addWidget(QLabel("Max:", range_page))
        range_layout.addWidget(self.max_sb)

        self.stack.addWidget(range_page)

        # List page
        list_page = QWidget(self)
        list_layout = QHBoxLayout(list_page)
        list_layout.setContentsMargins(0, 0, 0, 0)

        self.list_edit = QLineEdit(list_page)
        self.list_edit.setPlaceholderText("e.g. 0.05, 0.10, 0.20")
        list_layout.addWidget(self.list_edit)

        self.stack.addWidget(list_page)

        # --- optional Sampling selector ---
        self.sampling_cb: QComboBox | None = None
        self._sampling_label: QLabel | None = None
        if self._show_sampling:
            self._sampling_label = QLabel("Sampling:", self)
            self.sampling_cb = QComboBox(self)
            self.sampling_cb.addItem("Linear")   # underlying: None
            self.sampling_cb.addItem("Log")      # underlying: "log"
            self.sampling_cb.currentIndexChanged.connect(
                self._on_sampling_changed
            )

        # Root layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.mode)
        layout.addWidget(self.stack)
        if self._show_sampling:
            layout.addWidget(self._sampling_label)
            layout.addWidget(self.sampling_cb)
        layout.addStretch(1)

        self.mode.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.mode.setCurrentIndex(0)
        self.stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Internal helpers for sampling
    # ------------------------------------------------------------------
    def _set_sampling_from_string(self, sampling: str | None) -> None:
        """Update self._sampling and combo box from a raw string."""
        if not self._show_sampling or self.sampling_cb is None:
            self._sampling = None
            return

        s = (sampling or "").lower()
        if s == "log":
            self._sampling = "log"
            idx = self.sampling_cb.findText("Log")
            if idx >= 0:
                self.sampling_cb.setCurrentIndex(idx)
        else:
            self._sampling = None
            idx = self.sampling_cb.findText("Linear")
            if idx >= 0:
                self.sampling_cb.setCurrentIndex(idx)

    def _on_sampling_changed(self, index: int) -> None:
        """Callback when user changes the combo box."""
        if not self._show_sampling or self.sampling_cb is None:
            return
        text = self.sampling_cb.itemText(index).lower()
        self._sampling = "log" if text == "log" else None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def set_defaults(
        self,
        min_value: float,
        max_value: float,
        sampling: str | None = None,
    ) -> None:
        """Set default range and optional sampling ('log' or 'linear')."""
        lo = float(min(min_value, max_value))
        hi = float(max(min_value, max_value))
        self.min_sb.setRange(
            min(self.min_sb.minimum(), lo),
            max(self.min_sb.maximum(), hi),
        )
        self.max_sb.setRange(
            min(self.max_sb.minimum(), lo),
            max(self.max_sb.maximum(), hi),
        )
        self.min_sb.setValue(lo)
        self.max_sb.setValue(hi)
        if self._show_sampling:
            self._set_sampling_from_string(sampling)
        else:
            self._sampling = None

    def _defaults_from_value(self, value) -> tuple[float, float, str | None]:
        """Infer sensible (min, max, sampling) from a tuner-space value."""
        if isinstance(value, dict):
            return (
                float(value.get("min_value", 0.0)),
                float(value.get("max_value", 1.0)),
                value.get("sampling"),
            )
        if isinstance(value, (list, tuple)) and value:
            vals = [float(v) for v in value]
            return min(vals), max(vals), None
        return 0.0, 1.0, None

    def from_search_space_value(self, value, default_value=None) -> None:
        """Populate the widget from a `TUNER_SEARCH_SPACE` entry."""
        if default_value is None:
            default_value = value

        dmin, dmax, dsamp = self._defaults_from_value(default_value)

        if isinstance(value, dict) and "min_value" in value and "max_value" in value:
            # Range mode
            self.mode.setCurrentIndex(0)
            self.stack.setCurrentIndex(0)
            self.min_sb.setValue(float(value.get("min_value", dmin)))
            self.max_sb.setValue(float(value.get("max_value", dmax)))
            if self._show_sampling:
                self._set_sampling_from_string(value.get("sampling", dsamp))
            else:
                self._sampling = None
        elif isinstance(value, (list, tuple)) and value:
            # List mode
            self.mode.setCurrentIndex(1)
            self.stack.setCurrentIndex(1)
            self.list_edit.setText(", ".join(str(v) for v in value))
            if self._show_sampling:
                self._set_sampling_from_string(dsamp)
            else:
                self._sampling = None
        else:
            # Fallback: use defaults as a range
            self.mode.setCurrentIndex(0)
            self.stack.setCurrentIndex(0)
            self.min_sb.setValue(dmin)
            self.max_sb.setValue(dmax)
            if self._show_sampling:
                self._set_sampling_from_string(dsamp)
            else:
                self._sampling = None

    def _parse_list_text(self) -> list[float]:
        """Parse comma/semicolon separated floats from the list editor."""
        text = self.list_edit.text().strip()
        if not text:
            return []
        vals: list[float] = []
        for tok in text.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                vals.append(float(tok))
            except ValueError:
                continue
        return vals

    def to_search_space_value(self):
        """
        Return a value suitable for `TUNER_SEARCH_SPACE`:

        - If mode == 'List' and we have valid entries -> [v1, v2, ...]
        - Otherwise -> {"type": "float", "min_value": ..., "max_value": ..., "sampling": ...}
        """
        if self.mode.currentIndex() == 1:
            vals = self._parse_list_text()
            if vals:
                return vals  # discrete grid

        lo = float(min(self.min_sb.value(), self.max_sb.value()))
        hi = float(max(self.min_sb.value(), self.max_sb.value()))
        d = {
            "type": "float",
            "min_value": lo,
            "max_value": hi,
        }
        if self._show_sampling and self._sampling:
            d["sampling"] = self._sampling
        return d


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
        worker_ctl= None,
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

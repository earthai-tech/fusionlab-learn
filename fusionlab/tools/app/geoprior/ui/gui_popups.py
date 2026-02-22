
from __future__ import annotations 
import shutil, os

from PyQt5.QtCore import Qt # , QPointF
from PyQt5.QtGui import QPixmap, QGuiApplication, QKeySequence, QPainter
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, 
    QHBoxLayout, 
    QLabel, 
    QPushButton,
    QFileDialog,
    QMessageBox, 
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem, 
    QToolButton, QWidget, 
    QShortcut, 
    QTableWidget, 
    QAbstractItemView, 
    QTableWidgetItem, 
    QHeaderView
)

class Stage1ChoiceDialog(QDialog):
    """
    Let the user decide how to handle Stage-1 before training.

    API
    ---
    decision, summary = Stage1ChoiceDialog.ask(
        parent, city, runs_for_city, all_runs
    )

    decision in {"reuse", "rebuild", "cancel"}.
    summary is the selected Stage1Summary (for reuse) or None.
    """

    def __init__(self, parent, city, runs_for_city, all_runs):
        super().__init__(parent)
        self._runs = runs_for_city
        self._all_runs = all_runs
        self.decision = "cancel"
        self.selected_summary = None

        self.setWindowTitle(f"Stage-1 runs for {city}")
        self.resize(700, 400)

        layout = QVBoxLayout(self)

        # --- table with existing runs (you already have this) ---
        self.table = QTableWidget(len(self._runs), 5, self)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.table.setHorizontalHeaderLabels(
            ["City", "Timestamp", "T", "H (years)", "Status"]
        )

        for row, s in enumerate(self._runs):
            self.table.setItem(row, 0, QTableWidgetItem(s.city))
            self.table.setItem(row, 1, QTableWidgetItem(s.timestamp))
            self.table.setItem(row, 2, QTableWidgetItem(str(s.time_steps)))
            self.table.setItem(row, 3, QTableWidgetItem(str(s.horizon_years)))
            status = "OK" if s.is_complete else "Incomplete"
            if not s.config_match:
                status += " (config mismatch)"
            self.table.setItem(row, 4, QTableWidgetItem(status))

        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        layout.addWidget(self.table)

        # --- diff label (NEW) ---
        self.diff_label = QLabel("")
        self.diff_label.setObjectName("diffLabel")
        self.diff_label.setStyleSheet(
            "color:#d98a00; font-weight:600;"
        )
        layout.addWidget(self.diff_label)

        # --- buttons ---
        btn_row = QHBoxLayout()
        self.reuse_btn = QPushButton("Reuse selected Stage-1")
        self.rebuild_btn = QPushButton("Rebuild Stage-1")
        self.cancel_btn = QPushButton("Cancel")

        btn_row.addWidget(self.reuse_btn)
        btn_row.addWidget(self.rebuild_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.cancel_btn)

        layout.addLayout(btn_row)

        # connections
        self.reuse_btn.clicked.connect(self._accept_reuse)
        self.rebuild_btn.clicked.connect(self._accept_rebuild)
        self.cancel_btn.clicked.connect(self.reject)

        self.table.selectionModel().currentRowChanged.connect(
            self._on_row_changed
        )

        if self._runs:
            self.table.selectRow(len(self._runs) - 1)

    @classmethod
    def ask(cls, parent, city, runs_for_city, all_runs):
        dlg = cls(parent, city, runs_for_city, all_runs)
        result = dlg.exec_()
        if result != QDialog.Accepted:
            return "cancel", None
        return dlg.decision, dlg.selected_summary

    def _on_row_changed(self, current, _previous):
        row = current.row()
        if row < 0 or row >= len(self._runs):
            self.diff_label.setText("")
            return

        summary = self._runs[row]
        if summary.diff_fields:
            self.diff_label.setText(
                "⚠ changed: " + ", ".join(summary.diff_fields)
            )
        else:
            self.diff_label.setText(
                "✓ config matches current GUI setup."
            )

    def _accept_reuse(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self._runs):
            QMessageBox.warning(
                self,
                "No selection",
                "Please select a Stage-1 run to reuse.",
            )
            return

        self.decision = "reuse"
        self.selected_summary = self._runs[row]
        self.accept()

    def _accept_rebuild(self):
        self.decision = "rebuild"
        self.selected_summary = None
        self.accept()

class ImagePopup(QDialog):
    """Simple full-screen preview (no extra actions)."""
    def __init__(self, png_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(png_path))
        self.resize(900, 600)

        vbox = QVBoxLayout(self)

        lbl = QLabel(alignment=Qt.AlignCenter)
        lbl.setPixmap(
            QPixmap(png_path).scaled(
                880, 560, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        vbox.addWidget(lbl, 1)

        btn = QPushButton("Close", clicked=self.accept)
        vbox.addWidget(btn, alignment=Qt.AlignCenter)

class _ZoomableView(QGraphicsView):
    """Mouse-wheel zoom + hand-drag panning."""
    _ZOOM_STEP = 1.25

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setRenderHints(
            self.renderHints() | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    # wheel = zoom 
    def wheelEvent(self, ev):
        factor = self._ZOOM_STEP if ev.angleDelta().y() > 0 else 1 / self._ZOOM_STEP
        self.scale(factor, factor)

    # double-click = fit 
    def mouseDoubleClickEvent(self, ev):
        self.fitInView(self.scene().itemsBoundingRect(),
                        Qt.KeepAspectRatio)
        super().mouseDoubleClickEvent(ev)


class ImagePreviewDialog(QDialog):
    """
    Zoomable preview pane with
      • mouse-wheel zoom
      • hand-drag panning
      • floating ‘+ / – / fit’ bar (top-left, translucent)
      • Save-as / Copy / Close buttons (bottom-right)
    """
    _ZOOM = 1.25     # used by buttons

    def __init__(self, png_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(png_path))
        self.setMinimumSize(920, 680)
        self.png_path = png_path

        # central graphics view 
        self.view  = _ZoomableView()
        # self.view.setRenderHints(
        #         self.view.renderHints()
        #         | QPainter.SmoothPixmapTransform
        #         | QPainter.Antialiasing
        # )
                    
        scene      = QGraphicsScene(self)
        pix        = QPixmap(png_path)

        pix = QPixmap(png_path)
        if pix.isNull():
            QMessageBox.warning(self, "Preview", "Cannot load image.")
        else:
            # Hi-DPI screens → make the pixmap report *native* resolution
            pix.setDevicePixelRatio(self.devicePixelRatioF())
        item = QGraphicsPixmapItem(pix)

        scene.addItem(item)
        self.view.setScene(scene)
        self.view.fitInView(item, Qt.KeepAspectRatio)

        # main layout 
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.addWidget(self.view, 1)

        # FLOATING ZOOM BAR 
        overlay = QWidget(self.view)
        overlay.setAttribute(Qt.WA_StyledBackground, True)
        overlay.setStyleSheet("""
            QWidget { background: rgba(255,255,255,120); border-radius:6px; }
            QToolButton {
                background: transparent;
                border: 1px solid rgba(0,0,0,40);
                border-radius: 4px;
                padding: 0 6px;
                font-weight: bold;
            }
            QToolButton:hover {
                background: rgba(242,134,32,140);          /* SECONDARY */
                color: white;
                border: 1px solid rgba(242,134,32,180);
            }
        """)
        ol = QHBoxLayout(overlay); ol.setContentsMargins(4, 2, 4, 2)

        tb_minus = QToolButton(); tb_minus.setText("–")
        tb_plus  = QToolButton(); tb_plus.setText("+")
        tb_fit = QToolButton(); tb_fit.setText("⤢")
        for b in (tb_minus, tb_plus, tb_fit):
            ol.addWidget(b)

        overlay.move(10, 10)     # 10 px from top–left
        overlay.raise_()

        # zoom button actions
        tb_plus .clicked.connect(
            lambda: self.view.scale(self._ZOOM, self._ZOOM))
        tb_minus.clicked.connect(
            lambda: self.view.scale(1 / self._ZOOM, 1 / self._ZOOM))
        tb_fit  .clicked.connect(
            lambda: self.view.fitInView(item, Qt.KeepAspectRatio))

        # BOTTOM-RIGHT ACTION BAR 
        row = QHBoxLayout(); row.addStretch(1)

        btn_save  = QPushButton("Save as…")
        btn_copy  = QPushButton("Copy to clipboard")
        btn_close = QPushButton("Close")

        row.addWidget(btn_save)
        row.addWidget(btn_copy)
        row.addWidget(btn_close)
        vbox.addLayout(row)

        # connections 
        btn_close.clicked.connect(self.accept)
        btn_save .clicked.connect(self._save_as)
        btn_copy .clicked.connect(self._copy_clipboard)

        # handy shortcuts
        btn_plus_sc  = QShortcut(QKeySequence("Ctrl++"), self)
        btn_plus_sc.activated.connect(tb_plus.click)
        btn_minus_sc = QShortcut(QKeySequence("Ctrl+-"), self)
        btn_minus_sc.activated.connect(tb_minus.click)
        btn_fit_sc   = QShortcut(QKeySequence("Ctrl+0"),  self)
        btn_fit_sc.activated.connect(tb_fit.click)

    def _save_as(self):
        dst, _ = QFileDialog.getSaveFileName(
            self, "Save figure as…", self.png_path,
            "PNG image (*.png);;PDF file (*.pdf);;All files (*)")
        if dst:
            try:
                shutil.copyfile(self.png_path, dst)
            except Exception as e:
                QMessageBox.warning(self, "Save error", str(e))

    def _copy_clipboard(self):
        pix = QPixmap(self.png_path)
        if pix.isNull():
            QMessageBox.information(self, "Clipboard",
                                    "Could not load image for copying.")
            return
        QGuiApplication.clipboard().setPixmap(pix)



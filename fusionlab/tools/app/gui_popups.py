
from __future__ import annotations 

from PyQt5.QtCore   import Qt # , QPointF
from PyQt5.QtGui    import QPixmap, QGuiApplication, QKeySequence, QPainter
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QToolButton, QWidget, QShortcut 
)
import shutil, os

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

class _ImagePreviewDialog(QDialog):
    """
    Small preview that pops up whenever the visualiser saves a figure.
    Comes with “Save as…” and “Copy to clipboard” actions.
    """
    def __init__(self, png_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(png_path))
        self.setMinimumSize(600, 400)

        vbox = QVBoxLayout(self)

        pix = QPixmap(png_path)
        lbl = QLabel(alignment=Qt.AlignCenter)
        lbl.setPixmap(
            pix.scaled(580, 340, Qt.KeepAspectRatio,
                       Qt.SmoothTransformation)
        )
        vbox.addWidget(lbl, 1)

        # button row 
        row = QHBoxLayout(); vbox.addLayout(row)

        self.btn_save  = QPushButton("Save as…")
        self.btn_copy  = QPushButton("Copy to clipboard")
        self.btn_close = QPushButton("Close")

        row.addStretch(1)
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_copy)
        row.addWidget(self.btn_close)

        # connections 
        self.png_path = png_path
        self.btn_close.clicked.connect(self.close)
        self.btn_save .clicked.connect(self._save_as)
        self.btn_copy.clicked.connect(self._copy_clipboard)

    def _save_as(self):
        dst, _ = QFileDialog.getSaveFileName(
            self,
            "Save figure as…",
            self.png_path,
            "PNG image (*.png);;PDF file (*.pdf);;All files (*)",
        )
        if dst:
            try:
                shutil.copyfile(self.png_path, dst)
            except Exception as err:
                QMessageBox.warning(self, "Save error", str(err))

    def _copy_clipboard(self):
        pix = QPixmap(self.png_path)
        if pix.isNull():
            QMessageBox.information(
                self, "Clipboard", "Could not load image for copying."
            )
            return
        QGuiApplication.clipboard().setPixmap(pix)


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


#  Image preview dialog with floating zoom bar
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
        self.setMinimumSize(720, 480)
        self.png_path = png_path

        # central graphics view 
        self.view  = _ZoomableView()
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

    # 
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



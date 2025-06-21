# gui_popups.py
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox
)
from PyQt5.QtGui  import QPixmap, QGuiApplication
from PyQt5.QtCore import Qt
import os, shutil

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


class ImagePreviewDialog(QDialog):
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
            pix.scaled(580, 340, Qt.KeepAspectRatio, Qt.SmoothTransformation)
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

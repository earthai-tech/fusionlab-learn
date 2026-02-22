from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QSplitter, QVBoxLayout, QWidget

from .navigator import XferNavigator
from .run_center import XferRunCenter
from .preview import XferRunPreviewPanel


class XferRunWorkspace(QWidget):
    view_clicked = pyqtSignal()

    def __init__(self, *, store, make_card, parent=None):
        super().__init__(parent)
        self._s = store
        self._make_card = make_card
        self._build_ui()
        self._wire()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal, self)
        root.addWidget(self.splitter, 1)

        self.nav = XferNavigator(
            store=self._s,
            make_card=self._make_card,
            parent=self,
        )
        self.splitter.addWidget(self.nav)

        self.center = XferRunCenter(
            store=self._s,
            make_card=self._make_card,
            parent=self,
        )
        self.splitter.addWidget(self.center)

        self.preview = XferRunPreviewPanel(
            store=self._s,
            parent=self,
        )
        self.splitter.addWidget(self.preview)

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)

        self._alias_from_center()

    def _wire(self) -> None:
        self.btn_make_view.clicked.connect(
            self.view_clicked.emit
        )
        self.nav.clicked.connect(self.center.goto_card)


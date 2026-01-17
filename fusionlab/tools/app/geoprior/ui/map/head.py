# geoprior/ui/map/head.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.map.head

Map head bar (H + H2).

Includes:
- engine selector
- coord mode selector
- focus + analytics toggles
- X/Y/Z column pickers
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Sequence

from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFontMetrics
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy
)

@dataclass
class XYZ:
    x: str = ""
    y: str = ""
    z: str = ""

class ElideLabel(QLabel):
    """
    QLabel that keeps a full text and shows
    an elided version depending on width.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__("", parent)
        self._full = str(text or "")
        sp = QSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        self.setSizePolicy(sp)
        self.set_full_text(self._full)

    def set_full_text(self, text: str) -> None:
        self._full = str(text or "")
        self._apply_elide()

    def full_text(self) -> str:
        return str(self._full)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_elide()

    def _apply_elide(self) -> None:
        fm: QFontMetrics = self.fontMetrics()
        w = max(10, int(self.width()))
        self.setText(
            fm.elidedText(
                self._full,
                Qt.ElideRight,
                w,
            )
        )


class ColumnPicker(QWidget):
    """
    Compact picker: label + readonly field + menu button.
    """

    changed = pyqtSignal(str)

    def __init__(
        self,
        *,
        label: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._cols: List[str] = []

        self.lb = QLabel(label, self)
        self.edit = QLineEdit(self)
        self.edit.setReadOnly(True)

        self.btn = QToolButton(self)
        self.btn.setText("▼")
        self.btn.setToolTip("Pick column")

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.lb.setMinimumWidth(16)
        self.edit.setMinimumWidth(140)
        self.btn.setFixedWidth(26)

        row.addWidget(self.lb)
        row.addWidget(self.edit, 1)
        row.addWidget(self.btn)

        self.btn.clicked.connect(self._open_menu)


    def set_columns(self, cols: Sequence[str]) -> None:
        self._cols = list(cols or [])

    def set_value(self, value: str) -> None:
        self.edit.setText(str(value or ""))

    def value(self) -> str:
        return self.edit.text().strip()

    def _open_menu(self) -> None:
        menu = QMenu(self)
        menu.setToolTipsVisible(True)

        if not self._cols:
            act = menu.addAction("(no columns)")
            act.setEnabled(False)
        else:
            for c in self._cols:
                act = menu.addAction(str(c))
                act.triggered.connect(
                    lambda _=False, v=c: self._set(v),
                )

        menu.exec_(self.btn.mapToGlobal(
            self.btn.rect().bottomLeft(),
        ))

    def _set(self, v: str) -> None:
        self.set_value(v)
        self.changed.emit(str(v))


class MapHeadBar(QWidget):
    """
    Top bar for MapTab (engine + selectors).
    """

    engine_changed = pyqtSignal(str)
    coord_mode_changed = pyqtSignal(str)
    focus_toggled = pyqtSignal(bool)
    analytics_toggled = pyqtSignal(bool)

    x_changed = pyqtSignal(str)
    y_changed = pyqtSignal(str)
    z_changed = pyqtSignal(str)
    fit_clicked = pyqtSignal()
    swap_xy_clicked = pyqtSignal()
    reset_mapping_clicked = pyqtSignal()


    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._cols: List[str] = []

        self.cmb_engine = QComboBox(self)
        self.cmb_engine.addItems([
            "leaflet",
            "maplibre",
            "google",
        ])

        self.cmb_coord = QComboBox(self)
        self.cmb_coord.addItems([
            "lonlat",
            "utm",
            "epsg",
        ])

        self.chk_focus = QCheckBox("Focus", self)
        self.chk_analytics = QCheckBox("Analytics", self)

        self.pk_x = ColumnPicker(label="X:", parent=self)
        self.pk_y = ColumnPicker(label="Y:", parent=self)
        self.pk_z = ColumnPicker(label="Z:", parent=self)
        
        # Context pill (City / Model / Stage)
        self.lb_city = QLabel("", self)
        self.lb_city.setObjectName("MapCityPill")
        self.lb_city.setVisible(False)
        
        self.lb_active = ElideLabel("", self)
        self.lb_active.setObjectName("MapActiveLabel")
        self.lb_active.setVisible(False)
        
        self.btn_swap = QToolButton(self)
        self.btn_swap.setObjectName("MapSwapButton")
        self.btn_swap.setToolTip("Swap X and Y")
        self.btn_swap.setAutoRaise(True)
        self.btn_swap.setText("⇄")
        self.btn_swap.setFixedSize(28, 26)
        
        self.btn_reset = QToolButton(self)
        self.btn_reset.setObjectName("MapResetButton")
        self.btn_reset.setToolTip("Reset X/Y/Z mapping")
        self.btn_reset.setAutoRaise(True)
        self.btn_reset.setText("↺")
        self.btn_reset.setFixedSize(28, 26)


        # Fit-to-data (map icon)
        self.btn_fit = QToolButton(self)
        self.btn_fit.setObjectName("MapFitButton")
        self.btn_fit.setToolTip("Zoom to plotted points")
        self.btn_fit.setAutoRaise(True)
        self.btn_fit.setIcon(self._load_icon("map_icon.svg"))
        self.btn_fit.setIconSize(QSize(18, 18))
        
        
        self._build_ui()
        self._connect()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(10)

        left = QWidget(self)
        l = QHBoxLayout(left)
        
        ctx = QWidget(self)
        cv = QVBoxLayout(ctx)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(2)

        cv.addWidget(self.lb_city, 0)
        cv.addWidget(self.lb_active, 0)

        l.addWidget(ctx, 0)
        l.addSpacing(8)

        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(10)

        l.addWidget(QLabel("Engine:", self))
        l.addWidget(self.cmb_engine)
        l.addSpacing(8)

        l.addWidget(QLabel("Coords:", self))
        l.addWidget(self.cmb_coord)
        l.addSpacing(8)

        l.addWidget(self.chk_focus)
        l.addWidget(self.chk_analytics)
        l.addStretch(1)

        right = QWidget(self)
        r = QVBoxLayout(right)
        r.setContentsMargins(0, 0, 0, 0)
        r.setSpacing(6)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        row.addWidget(self.pk_x)
        row.addWidget(self.btn_swap)
        row.addWidget(self.pk_y)
        row.addWidget(self.pk_z)
        row.addWidget(self.btn_fit)
        row.addWidget(self.btn_reset)

        r.addLayout(row)

        root.addWidget(left, 1)
        root.addWidget(right, 0)
        
        self._apply_styles()

    def _connect(self) -> None:
        self.cmb_engine.currentTextChanged.connect(
            self.engine_changed,
        )
        self.cmb_coord.currentTextChanged.connect(
            self.coord_mode_changed,
        )
        self.chk_focus.toggled.connect(
            self.focus_toggled,
        )
        self.chk_analytics.toggled.connect(
            self.analytics_toggled,
        )

        self.pk_x.changed.connect(self.x_changed)
        self.pk_y.changed.connect(self.y_changed)
        self.pk_z.changed.connect(self.z_changed)
        self.btn_fit.clicked.connect(self.fit_clicked)
        self.btn_swap.clicked.connect(
            self.swap_xy_clicked,
        )
        self.btn_reset.clicked.connect(
            self.reset_mapping_clicked,
        )
        
    def _apply_styles(self) -> None:
        qss = "\n".join([
            "QLabel#MapCityPill {",
            "  padding: 3px 10px;",
            "  border-radius: 10px;",
            "  font-weight: 600;",
            "  border: 1px solid rgba(46,49,145,0.35);",
            "  background: rgba(46,49,145,0.10);",
            "}",
            "QLabel#MapActiveLabel {",
            "  padding-left: 2px;",
            "  font-size: 11px;",
            "  color: rgba(0,0,0,0.65);",
            "}",
            "QToolButton#MapFitButton,",
            "QToolButton#MapSwapButton,",
            "QToolButton#MapResetButton {",
            "  border-radius: 8px;",
            "  padding: 4px;",
            "}",
            "QToolButton#MapFitButton:hover,",
            "QToolButton#MapSwapButton:hover,",
            "QToolButton#MapResetButton:hover {",
            "  background: rgba(46,49,145,0.10);",
            "}",
        ])
        self.setStyleSheet(qss)

    # -----------------------------
    # External setters
    # -----------------------------
    def set_available_columns(
        self,
        cols: Sequence[str],
    ) -> None:
        self._cols = list(cols or [])
        self.pk_x.set_columns(self._cols)
        self.pk_y.set_columns(self._cols)
        self.pk_z.set_columns(self._cols)

    def set_engine(self, engine: str) -> None:
        self._set_combo(self.cmb_engine, engine)

    def set_coord_mode(self, mode: str) -> None:
        self._set_combo(self.cmb_coord, mode)

    def set_focus_checked(self, checked: bool) -> None:
        self.chk_focus.blockSignals(True)
        self.chk_focus.setChecked(bool(checked))
        self.chk_focus.blockSignals(False)

    def set_analytics_checked(self, checked: bool) -> None:
        self.chk_analytics.blockSignals(True)
        self.chk_analytics.setChecked(bool(checked))
        self.chk_analytics.blockSignals(False)

    def set_xyz(self, *, x: str, y: str, z: str) -> None:
        self.pk_x.set_value(x)
        self.pk_y.set_value(y)
        self.pk_z.set_value(z)
        
    def set_active_dataset(
        self,
        name: str,
        *,
        tooltip: str = "",
    ) -> None:
        name = (name or "").strip()
        if not name:
            self.lb_active.setVisible(False)
            self.lb_active.set_full_text("")
            self.lb_active.setToolTip("")
            return

        self.lb_active.setVisible(True)
        self.lb_active.set_full_text(name)
        self.lb_active.setToolTip(tooltip or name)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _set_combo(self, cmb: QComboBox, v: str) -> None:
        v = str(v or "").strip()
        if not v:
            return
        idx = cmb.findText(v, Qt.MatchFixedString)
        if idx >= 0:
            cmb.blockSignals(True)
            cmb.setCurrentIndex(idx)
            cmb.blockSignals(False)

    def set_city_badge(
        self,
        *,
        city: str = "",
        model: str = "",
        stage: str = "",
    ) -> None:
        city = (city or "").strip()
        model = (model or "").strip()
        stage = (stage or "").strip()

        if not city:
            self.lb_city.setVisible(False)
            self.lb_city.setText("")
            return

        # Compact text like: "nansha · GeoPriorSubsNet · stage1"
        parts = [city]
        if model:
            parts.append(model)
        if stage:
            parts.append(stage)

        self.lb_city.setText(" · ".join(parts))
        self.lb_city.setVisible(True)
        self.lb_city.setToolTip(self.lb_city.text())

    def _load_icon(self, filename: str) -> QIcon:
        # geoprior/ui/map/head.py -> go up to geoprior/ then icons/
        here = Path(__file__).resolve()
        pkg = here.parents[2]  # .../geoprior
        icon_path = pkg / "icons" / filename
        if icon_path.exists():
            return QIcon(str(icon_path))
        return QIcon()  # fallback

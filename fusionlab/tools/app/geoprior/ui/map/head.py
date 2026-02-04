# geoprior/ui/map/head.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.map.head

Map head bar (H + H2).

Includes
--------
- engine selector
- coord mode selector
- focus + analytics toggles
- X/Y/Z column pickers

Premium touches
-----------------
- small "Map" brand (icon + title)
- status chip (datasets + mapping validity)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QIcon, QFontMetrics, QIntValidator
from PyQt5.QtWidgets import (
    QActionGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QStyle, 
    # QFontMetrics,
    # QIntValidator,
)

from ..icon_utils import try_icon

@dataclass
class XYZ:
    x: str = ""
    y: str = ""
    z: str = ""


class ElideLabel(QLabel):
    """QLabel that elides text based on width."""

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
    """Compact picker: label + field + menu button."""

    changed = pyqtSignal(str)

    def __init__(
        self,
        *,
        label: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._cols: List[str] = []

        self.setObjectName("mapColPicker")
        sp = QSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        self.setSizePolicy(sp)

        self.lb = QLabel(label, self)
        self.lb.setObjectName("mapColLabel")

        self.edit = QLineEdit(self)
        self.edit.setObjectName("mapColEdit")
        self.edit.setReadOnly(True)
        self.edit.setFocusPolicy(Qt.NoFocus)
        self.edit.setMinimumHeight(28)
        self.edit.setPlaceholderText("—")

        self.btn = QToolButton(self)
        self.btn.setObjectName("miniAction")
        self.btn.setProperty("role", "mapHead")
        self.btn.setText("▾")
        self.btn.setToolTip("Pick column")
        self.btn.setAutoRaise(True)
        self.btn.setFixedWidth(28)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.lb.setMinimumWidth(18)
        self.edit.setMinimumWidth(132)

        row.addWidget(self.lb, 0)
        row.addWidget(self.edit, 1)
        row.addWidget(self.btn, 0)

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
            for col in self._cols:
                act = menu.addAction(str(col))
                act.triggered.connect(
                    lambda _=False, v=col: self._set(v)
                )

        pos = self.btn.mapToGlobal(
            self.btn.rect().bottomLeft()
        )
        menu.exec_(pos)

    def _set(self, v: str) -> None:
        self.set_value(v)
        self.changed.emit(str(v))


class MapHeadBar(QWidget):
    """Top bar for MapTab (engine + selectors)."""

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

    basemap_changed = pyqtSignal(str)
    grid_toggled = pyqtSignal(bool)
    legend_toggled = pyqtSignal(bool)

    export_requested = pyqtSignal(str)
    bookmark_requested = pyqtSignal(str)
    measure_mode_changed = pyqtSignal(str)
    
    epsg_changed = pyqtSignal(int)
    clear_map_requested = pyqtSignal()

    data_toggled = pyqtSignal(bool)
    view_toggled = pyqtSignal(bool)


    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._cols: List[str] = []
        self._n_sets = 0

        self.card = QFrame(self)
        self.card.setObjectName("mapHeadCard")
        self.card.setFrameShape(QFrame.NoFrame)

        # Brand (icon + title)
        self.lb_icon = QLabel(self.card)
        self.lb_icon.setObjectName("mapHeadIcon")
        self._set_brand_icon()

        self.lb_title = QLabel("Map", self.card)
        self.lb_title.setObjectName("mapHeadTitle")

        # Status chip
        self.lb_status = QLabel("", self.card)
        self.lb_status.setObjectName("mapHeadStatus")
        self.lb_status.setVisible(False)
        self.lb_status.setProperty("state", "info")

        # Context pill + dataset
        self.lb_city = QLabel("", self.card)
        self.lb_city.setObjectName("mapHeadPill")
        self.lb_city.setVisible(False)

        self.lb_active = ElideLabel("", self.card)
        self.lb_active.setObjectName("mapHeadDataset")
        self.lb_active.setVisible(False)

        # Engine / coords
        self.cmb_engine = QComboBox(self.card)
        self.cmb_engine.setObjectName("mapHeadCombo")
        self.cmb_engine.addItems(
            [
                "leaflet",
                "maplibre",
                "google",
            ]
        )
        self.cmb_engine.setMinimumHeight(30)

        self.cmb_coord = QComboBox(self.card)
        self.cmb_coord.setObjectName("mapHeadCombo")
        self.cmb_coord.addItems(
            [
                "lonlat",
                "utm",
                "epsg",
            ]
        )
        self.cmb_coord.setMinimumHeight(30)
        
        # EPSG input (stable slot; enabled only in "epsg" mode)
        self._epsg_wrap = QWidget(self.card)
        self._epsg_wrap.setObjectName("mapHeadEpsgWrap")
        self._epsg_wrap.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )

        ew = QHBoxLayout(self._epsg_wrap)
        ew.setContentsMargins(0, 0, 0, 0)
        ew.setSpacing(4)

        self.lb_epsg = QLabel("EPSG", self._epsg_wrap)
        self.lb_epsg.setObjectName("mapHeadKey")

        self.ed_epsg = QLineEdit(self._epsg_wrap)
        self.ed_epsg.setObjectName("mapHeadEpsgEdit")
        self.ed_epsg.setAlignment(Qt.AlignCenter)
        self.ed_epsg.setMinimumHeight(30)
        self.ed_epsg.setFixedWidth(72)
        self.ed_epsg.setPlaceholderText("4326")
        self.ed_epsg.setToolTip(
            "Source EPSG for X/Y columns"
        )

        vld = QIntValidator(1, 999999, self.ed_epsg)
        self.ed_epsg.setValidator(vld)

        ew.addWidget(self.lb_epsg, 0)
        ew.addWidget(self.ed_epsg, 0)

        # Always reserve the slot (stable layout).
        self._epsg_wrap.setVisible(True)
        self._epsg_wrap.setEnabled(False)

        # Keep the wrapper compact (prevents "too wide" EPSG block).
        lbw = self.lb_epsg.sizeHint().width()
        # spacing(4) + edit(72) + tiny slack(6)
        self._epsg_wrap.setFixedWidth(lbw + 4 + 72 + 6)

        # Toggle pills
        self.btn_focus = QToolButton(self.card)
        self.btn_focus.setObjectName("mapHeadToggle")
        self.btn_focus.setText("Focus")
        self.btn_focus.setCheckable(True)
        self.btn_focus.setAutoRaise(True)
        self.btn_focus.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.btn_focus.setIcon(
            try_icon(
                "focus.svg",
                fallback=self.style().standardIcon(
                    QStyle.SP_DialogApplyButton
                ),
            )
        )
        self.btn_focus.setIconSize(QSize(16, 16))
        self.btn_focus.setMinimumHeight(28)

        self.btn_analytics = QToolButton(self.card)
        self.btn_analytics.setObjectName("mapHeadToggle")
        self.btn_analytics.setText("Analytics")
        self.btn_analytics.setCheckable(True)
        self.btn_analytics.setAutoRaise(True)
        self.btn_analytics.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.btn_analytics.setIcon(
            try_icon(
                "analytics.svg",
                fallback=self.style().standardIcon(
                    QStyle.SP_ComputerIcon
                ),
            )
        )
        self.btn_analytics.setIconSize(QSize(16, 16))
        self.btn_analytics.setMinimumHeight(28)

        # Data / View (mini pill toggles)
        self.btn_data = QToolButton(self.card)
        self.btn_data.setObjectName("mapHeadToggle")
        self.btn_data.setProperty("variant", "mini")
        self.btn_data.setText("Data")
        self.btn_data.setCheckable(True)
        self.btn_data.setAutoRaise(True)
        self.btn_data.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.btn_data.setToolTip("Toggle data")
        self.btn_data.setIcon(
            try_icon(
                "data-panel.svg",
                fallback=self.style().standardIcon(
                    QStyle.SP_FileDialogContentsView
                ),
            )
        )
        self.btn_data.setIconSize(QSize(16, 16))
        self.btn_data.setMinimumHeight(28)

        self.btn_view = QToolButton(self.card)
        self.btn_view.setObjectName("mapHeadToggle")
        self.btn_view.setProperty("variant", "mini")
        self.btn_view.setText("View")
        self.btn_view.setCheckable(True)
        self.btn_view.setAutoRaise(True)
        self.btn_view.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.btn_view.setToolTip("Toggle view options")
        self.btn_view.setIcon(
            try_icon(
                "view-panel.svg",
                fallback=self.style().standardIcon(
                    QStyle.SP_FileDialogDetailedView
                ),
            )
        )
        self.btn_view.setIconSize(QSize(16, 16))
        self.btn_view.setMinimumHeight(28)

        self._bm_acts = {}
        self._ms_acts = {}
        self._bm_names = []

        self.btn_more = QToolButton(self.card)
        self.btn_more.setObjectName("mapHeadMore")
        self.btn_more.setToolTip("More map actions")
        self.btn_more.setAutoRaise(True)
        self.btn_more.setPopupMode(
            QToolButton.InstantPopup
        )
        # ico = self.style().standardIcon(
        #     QStyle.SP_TitleBarMenuButton
        # )
        # self.btn_more.setIcon(ico)
        self.btn_more.setIcon(
            try_icon(
                "more.svg",
                fallback=self.style().standardIcon(
                    QStyle.SP_TitleBarMenuButton
                ),
            )
        )
        self.btn_more.setIconSize(QSize(16, 16))
        self.btn_more.setText("")
        self.btn_more.setFixedSize(30, 30)

        self._more_menu = self._build_more_menu()
        self.btn_more.setMenu(self._more_menu)

        # Mapping pickers
        self.pk_x = ColumnPicker(label="X:", parent=self.card)
        self.pk_y = ColumnPicker(label="Y:", parent=self.card)
        self.pk_z = ColumnPicker(label="Z:", parent=self.card)

        # Actions
        self.btn_swap = QToolButton(self.card)
        self.btn_swap.setObjectName("miniAction")
        self.btn_swap.setProperty("role", "mapHead")
        self.btn_swap.setToolTip("Swap X and Y")
        self.btn_swap.setAutoRaise(True)
        self.btn_swap.setText("⇄")
        self.btn_swap.setFixedSize(30, 30)

        # Map actions (fit/clear/reset) moved to the
        # hover tooltab and the "More" menu.

        self._build_ui()
        self._connect()
        self._update_status()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        card_l = QVBoxLayout(self.card)
        card_l.setContentsMargins(12, 10, 12, 10)
        card_l.setSpacing(8)

        # Row 0: brand + ctx (left) | status + toggles
        row0 = QWidget(self.card)
        r0 = QHBoxLayout(row0)
        r0.setContentsMargins(0, 0, 0, 0)
        r0.setSpacing(10)

        left = QWidget(row0)
        lv = QHBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)

        brand = QWidget(left)
        bv = QHBoxLayout(brand)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(6)
        bv.addWidget(self.lb_icon, 0)
        bv.addWidget(self.lb_title, 0)

        ctx = QWidget(left)
        cv = QVBoxLayout(ctx)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(2)
        cv.addWidget(self.lb_city, 0)
        cv.addWidget(self.lb_active, 0)

        lv.addWidget(brand, 0)
        lv.addWidget(ctx, 1)

        right = QWidget(row0)
        rv = QHBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(8)
        rv.addWidget(self.lb_status, 0)
        rv.addWidget(self.btn_data, 0)
        rv.addWidget(self.btn_view, 0)
        rv.addWidget(self.btn_focus, 0)
        rv.addWidget(self.btn_analytics, 0)
        rv.addWidget(self.btn_more, 0)
        
        r0.addWidget(left, 1)
        r0.addWidget(right, 0)

        # Row 1: controls | mapping
        row1 = QWidget(self.card)
        r1 = QHBoxLayout(row1)
        r1.setContentsMargins(0, 0, 0, 0)
        r1.setSpacing(10)

        # ---- Controls group (single row, stable EPSG slot)
        ctrl = QFrame(row1)
        ctrl.setObjectName("mapHeadGroup")
        ctrl.setSizePolicy(
            QSizePolicy.Minimum,
            QSizePolicy.Fixed,
        )

        c1 = QHBoxLayout(ctrl)
        c1.setContentsMargins(10, 8, 10, 8)
        c1.setSpacing(8)

        lb_engine = QLabel("Engine", ctrl)
        lb_engine.setObjectName("mapHeadKey")

        lb_coord = QLabel("Coords", ctrl)
        lb_coord.setObjectName("mapHeadKey")

        c1.addWidget(lb_engine, 0)
        c1.addWidget(self.cmb_engine, 0)
        c1.addSpacing(6)
        c1.addWidget(lb_coord, 0)
        c1.addWidget(self.cmb_coord, 0)
        c1.addSpacing(6)
        c1.addWidget(self._epsg_wrap, 0)

        # No stretch here: ctrl stays "minimum needed"
        # so Mapping gets the width.
        # (Do not addStretch(1) in ctrl)

        # ---- Mapping group (gets stretch, won't collapse)
        mapping = QFrame(row1)
        mapping.setObjectName("mapHeadGroup")
        mapping.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )

        m1 = QHBoxLayout(mapping)
        m1.setContentsMargins(10, 8, 10, 8)
        m1.setSpacing(8)

        title = QLabel("Mapping", mapping)
        title.setObjectName("mapHeadKey")

        # Ensure pickers keep usable width.
        for pk in (self.pk_x, self.pk_y, self.pk_z):
            pk.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Fixed,
            )
            pk.setMinimumWidth(170)

        m1.addWidget(title, 0)
        m1.addWidget(self.pk_x, 1)
        m1.addWidget(self.btn_swap, 0)
        m1.addWidget(self.pk_y, 1)
        m1.addWidget(self.pk_z, 1)
        m1.addSpacing(2)

        # Critical stretch: Mapping expands, ctrl doesn't.
        r1.addWidget(ctrl, 0)
        r1.addWidget(mapping, 1)

        card_l.addWidget(row0, 0)
        card_l.addWidget(row1, 0)

        root.addWidget(self.card, 1)


    def _connect(self) -> None:
        self.cmb_engine.currentTextChanged.connect(
            self.engine_changed
        )
        self.cmb_coord.currentTextChanged.connect(
            self._on_coord_changed
        )
        self.ed_epsg.editingFinished.connect(
            self._emit_epsg
        )

        self.btn_focus.toggled.connect(self.focus_toggled)
        self.btn_analytics.toggled.connect(
            self.analytics_toggled
        )

        self.pk_x.changed.connect(self.x_changed)
        self.pk_y.changed.connect(self.y_changed)
        self.pk_z.changed.connect(self.z_changed)

        self.pk_x.changed.connect(
            lambda _v: self._update_status(),
        )
        self.pk_y.changed.connect(
            lambda _v: self._update_status(),
        )
        self.pk_z.changed.connect(
            lambda _v: self._update_status(),
        )

        self.btn_swap.clicked.connect(self.swap_xy_clicked)
        self.btn_data.toggled.connect(self.data_toggled)
        self.btn_view.toggled.connect(self.view_toggled)

    # -----------------------------
    # External setters
    # -----------------------------

    def _clear_icon(self) -> QIcon:
        sp = getattr(QStyle, "SP_TrashIcon", None)
        if sp is None:
            sp = QStyle.SP_DialogDiscardButton
        return self.style().standardIcon(sp)

    def set_epsg(self, epsg: int) -> None:
        try:
            v = int(epsg)
        except Exception:
            return
        if v <= 0:
            return
        self.ed_epsg.blockSignals(True)
        self.ed_epsg.setText(str(v))
        self.ed_epsg.blockSignals(False)

    def _on_coord_changed(self, mode: str) -> None:
        m = str(mode or "").strip().lower()
        self._epsg_wrap.setEnabled(m == "epsg")
        self.coord_mode_changed.emit(str(m))

    def _emit_epsg(self) -> None:
        txt = self.ed_epsg.text().strip()
        if not txt:
            return
        try:
            v = int(float(txt))
        except Exception:
            return
        if v <= 0:
            return
        self.epsg_changed.emit(int(v))

    def set_available_columns(
        self,
        cols: Sequence[str],
    ) -> None:
        self._cols = list(cols or [])
        self.pk_x.set_columns(self._cols)
        self.pk_y.set_columns(self._cols)
        self.pk_z.set_columns(self._cols)
        self._update_status()

    def set_engine(self, engine: str) -> None:
        self._set_combo(self.cmb_engine, engine)

    def set_coord_mode(self, mode: str) -> None:
        self._set_combo(self.cmb_coord, mode)
        m = str(mode or "").strip().lower()
        self._epsg_wrap.setEnabled(m == "epsg")

    def set_focus_checked(self, checked: bool) -> None:
        self.btn_focus.blockSignals(True)
        self.btn_focus.setChecked(bool(checked))
        self.btn_focus.blockSignals(False)
        
    def set_data_checked(self, checked: bool) -> None:
        self.btn_data.blockSignals(True)
        self.btn_data.setChecked(bool(checked))
        self.btn_data.blockSignals(False)
    
    def set_view_checked(self, checked: bool) -> None:
        self.btn_view.blockSignals(True)
        self.btn_view.setChecked(bool(checked))
        self.btn_view.blockSignals(False)

    def set_analytics_checked(self, checked: bool) -> None:
        self.btn_analytics.blockSignals(True)
        self.btn_analytics.setChecked(bool(checked))
        self.btn_analytics.blockSignals(False)

    def set_xyz(self, *, x: str, y: str, z: str) -> None:
        self.pk_x.set_value(x)
        self.pk_y.set_value(y)
        self.pk_z.set_value(z)
        self._update_status()

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

        parts = [city]
        if model:
            parts.append(model)
        if stage:
            parts.append(stage)

        self.lb_city.setText(" · ".join(parts))
        self.lb_city.setVisible(True)
        self.lb_city.setToolTip(self.lb_city.text())

    def _build_more_menu(self) -> QMenu:
        menu = QMenu(self)

        self._add_basemap_menu(menu)
        menu.addSeparator()

        self._act_grid = menu.addAction("Grid")
        self._act_grid.setCheckable(True)
        self._act_grid.toggled.connect(
            self.grid_toggled
        )

        self._act_leg = menu.addAction("Legend")
        self._act_leg.setCheckable(True)
        self._act_leg.toggled.connect(
            self.legend_toggled
        )

        menu.addSeparator()
        self._add_export_menu(menu)

        menu.addSeparator()
        self._add_bookmarks_menu(menu)

        menu.addSeparator()
        self._add_measure_menu(menu)

        menu.addSeparator()

        act = menu.addAction("Zoom to points")
        act.triggered.connect(self.fit_clicked.emit)

        act = menu.addAction("Reset X/Y/Z mapping")
        act.triggered.connect(
            self.reset_mapping_clicked.emit
        )

        act = menu.addAction("Clear map")
        act.triggered.connect(
            self.clear_map_requested.emit
        )

        return menu

    def _add_basemap_menu(self, root: QMenu) -> None:
        sub = root.addMenu("Basemap")

        grp = QActionGroup(sub)
        grp.setExclusive(True)

        items = [
            ("streets", "Streets"),
            ("sat", "Satellite"),
            ("terrain", "Terrain"),
            ("light", "Light"),
            ("dark", "Dark"),
        ]

        for key, label in items:
            act = sub.addAction(label)
            act.setCheckable(True)
            act.setData(key)
            grp.addAction(act)
            self._bm_acts[str(key)] = act

            act.triggered.connect(
                lambda _=False, k=key:
                self.basemap_changed.emit(str(k))
            )

    def _add_export_menu(self, root: QMenu) -> None:
        sub = root.addMenu("Export")

        a1 = sub.addAction("Snapshot PNG")
        a1.triggered.connect(
            lambda: self.export_requested.emit("png")
        )

        a2 = sub.addAction("Export visible (CSV)")
        a2.triggered.connect(
            lambda: self.export_requested.emit("csv_vis")
        )

        a3 = sub.addAction("Export selection (CSV)")
        a3.triggered.connect(
            lambda: self.export_requested.emit("csv_sel")
        )

    def _add_bookmarks_menu(self, root: QMenu) -> None:
        sub = root.addMenu("Bookmarks")

        a1 = sub.addAction("Add bookmark…")
        a1.triggered.connect(
            lambda: self.bookmark_requested.emit("add")
        )

        a2 = sub.addAction("Clear bookmarks")
        a2.triggered.connect(
            lambda: self.bookmark_requested.emit("clear")
        )

        sub.addSeparator()

        self._bm_go = sub.addMenu("Go to")
        self._bm_go.aboutToShow.connect(
            self._refresh_bm_go
        )

    def _refresh_bm_go(self) -> None:
        self._bm_go.clear()

        names = list(self._bm_names or [])
        names = [str(x).strip() for x in names]
        names = [x for x in names if x]

        if not names:
            a0 = self._bm_go.addAction("(none)")
            a0.setEnabled(False)
            return

        for name in names:
            act = self._bm_go.addAction(name)
            act.triggered.connect(
                lambda _=False, n=name:
                self.bookmark_requested.emit(
                    "goto:" + str(n)
                )
            )

    def _add_measure_menu(self, root: QMenu) -> None:
        sub = root.addMenu("Measure")

        grp = QActionGroup(sub)
        grp.setExclusive(True)

        items = [
            ("off", "Off"),
            ("dist", "Distance"),
            ("area", "Area"),
        ]

        for key, label in items:
            act = sub.addAction(label)
            act.setCheckable(True)
            act.setData(key)
            grp.addAction(act)
            self._ms_acts[str(key)] = act

            act.triggered.connect(
                lambda _=False, k=key:
                self.measure_mode_changed.emit(str(k))
            )

        sub.addSeparator()
        a0 = sub.addAction("Clear")
        a0.triggered.connect(
            lambda: self.measure_mode_changed.emit("clear")
        )

    def set_dataset_count(self, n: int) -> None:
        self._n_sets = max(0, int(n))
        self._update_status()

    def set_basemap(self, key: str) -> None:
        key = str(key or "").strip()
        act = self._bm_acts.get(key, None)
        if not act:
            return
        act.blockSignals(True)
        act.setChecked(True)
        act.blockSignals(False)

    def set_grid_checked(self, checked: bool) -> None:
        if not hasattr(self, "_act_grid"):
            return
        self._act_grid.blockSignals(True)
        self._act_grid.setChecked(bool(checked))
        self._act_grid.blockSignals(False)

    def set_legend_checked(self, checked: bool) -> None:
        if not hasattr(self, "_act_leg"):
            return
        self._act_leg.blockSignals(True)
        self._act_leg.setChecked(bool(checked))
        self._act_leg.blockSignals(False)

    def set_bookmarks(self, names: Sequence[str]) -> None:
        self._bm_names = list(names or [])

    def set_measure_mode(self, mode: str) -> None:
        mode = str(mode or "").strip()
        act = self._ms_acts.get(mode, None)
        if not act:
            return
        act.blockSignals(True)
        act.setChecked(True)
        act.blockSignals(False)

    # -----------------------------
    # Status logic
    # -----------------------------
    def _mapping_ok(self) -> bool:
        x = self.pk_x.value()
        y = self.pk_y.value()
        if not x or not y:
            return False

        cols = list(self._cols or [])
        if cols and (x not in cols or y not in cols):
            return False

        return True

    def _update_status(self) -> None:
        n = int(self._n_sets)
        x = self.pk_x.value()
        y = self.pk_y.value()

        have_map = bool(x or y or self.pk_z.value())
        have_data = n > 0

        if not have_data and not have_map:
            self.lb_status.setVisible(False)
            self.lb_status.setText("")
            return

        map_ok = self._mapping_ok()

        if have_data:
            left = f"{n} datasets"
        else:
            left = "No data"

        right = "mapping OK" if map_ok else "pick X/Y"
        text = f"{left} · {right}"

        if have_data and map_ok:
            state = "ok"
        elif have_data and not map_ok:
            state = "warn"
        else:
            state = "info"

        self.lb_status.setProperty("state", state)
        self.lb_status.setText(text)
        self.lb_status.setVisible(True)
        self.lb_status.style().unpolish(self.lb_status)
        self.lb_status.style().polish(self.lb_status)

    def _set_brand_icon(self) -> None:
        ico = self._load_icon("map.svg")
        if ico.isNull():
            ico = self._load_icon("map_icon.svg")
        if ico.isNull():
            self.lb_icon.setVisible(False)
            return

        self.lb_icon.setVisible(True)
        self.lb_icon.setPixmap(ico.pixmap(16, 16))

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

    def _load_icon(self, filename: str) -> QIcon:
        ico = try_icon(str(filename or ""))
        if not ico.isNull():
            return ico
        return QIcon()

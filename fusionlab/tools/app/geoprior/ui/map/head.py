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

from PyQt5.QtCore import Qt, QSize, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QIcon,
    QFontMetrics,
    QIntValidator,
    QColor,
    QFont,
    QCursor,
)
from PyQt5.QtWidgets import (
    QActionGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QWidgetAction,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QStyle,
    QCompleter,
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
        
class ClickableFrame(QFrame):
    clicked = pyqtSignal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

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
    analytics_requested = pyqtSignal(str)

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
        # Analytics menu (dropdown)
        self._ana_acts = {}
        self._ana_group = QActionGroup(self)
        self._ana_group.setExclusive(True)

        self._ana_menu = self._build_analytics_menu()
        self.btn_analytics.setMenu(self._ana_menu)
        self.btn_analytics.setPopupMode(
            QToolButton.InstantPopup
        )
        # Dynamic label state (default tab)
        self._ana_mode = "selection"
        self._update_ana_button_text()

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

        # Mapping chip (dot + button inside)
        self._map_state = "info"
        self._pulse_on = False
        
        self.w_mapchip = ClickableFrame(self.card)
        self.w_mapchip.setObjectName("mapHeadMapChip")
        self.w_mapchip.setProperty("state", "info")
        
        wl = QHBoxLayout(self.w_mapchip)
        wl.setContentsMargins(10, 0, 10, 0)
        wl.setSpacing(8)
        
        self.lb_mapdot = QLabel(self.w_mapchip)
        self.lb_mapdot.setObjectName("mapHeadMapDot")
        self.lb_mapdot.setFixedSize(8, 8)
        self.lb_mapdot.setProperty("state", "info")
        self.lb_mapdot.setProperty("pulse", "0")
        
        self.lb_mapico = QLabel(self.w_mapchip)
        self.lb_mapico.setObjectName("mapHeadMapIco")
        ico = try_icon("mapping.svg")
        if not ico.isNull():
            self.lb_mapico.setPixmap(ico.pixmap(16, 16))
        
        # Key/Value segments (pro look)
        def mk_key(txt: str) -> QLabel:
            lb = QLabel(txt, self.w_mapchip)
            lb.setObjectName("mapHeadMapKey")
            return lb
        
        def mk_sep() -> QLabel:
            lb = QLabel("·", self.w_mapchip)
            lb.setObjectName("mapHeadMapSep")
            return lb
        
        self.lb_xk = mk_key("X")
        self.lb_xv = ElideLabel("—", self.w_mapchip)
        self.lb_xv.setObjectName("mapHeadMapVal")
        
        self.lb_yk = mk_key("Y")
        self.lb_yv = ElideLabel("—", self.w_mapchip)
        self.lb_yv.setObjectName("mapHeadMapVal")
        
        self.lb_zk = mk_key("Z")
        self.lb_zv = ElideLabel("—", self.w_mapchip)
        self.lb_zv.setObjectName("mapHeadMapVal")
        
        # Swap inside capsule (so it feels like 1 control)
        self.btn_swap = QToolButton(self.w_mapchip)
        self.btn_swap.setObjectName("miniAction")
        self.btn_swap.setProperty("role", "mapHead")
        self.btn_swap.setToolTip("Swap X and Y")
        self.btn_swap.setAutoRaise(True)
        self.btn_swap.setIcon(
            try_icon(
                "swap.svg",
                fallback=self.style().standardIcon(
                    QStyle.SP_BrowserReload
                ),
            )
        )
        self.btn_swap.setIconSize(QSize(15, 15))
        self.btn_swap.setText("")
        self.btn_swap.setFixedSize(28, 28)
        
        # Drop button (small), but capsule click also opens menu
        self.btn_mapdrop = QToolButton(self.w_mapchip)
        self.btn_mapdrop.setObjectName("miniAction")
        self.btn_mapdrop.setProperty("role", "mapHead")
        self.btn_mapdrop.setToolTip("Edit mapping")
        self.btn_mapdrop.setAutoRaise(True)
        
        self.btn_mapdrop.setText("▾")
        self.btn_mapdrop.setFixedSize(28, 28)
        
        self._map_menu = self._build_mapping_menu()
        self._map_hover_timer = QTimer(self)
        self._map_hover_timer.setInterval(260)
        self._map_hover_timer.timeout.connect(
            self._map_menu_autoclose_tick
        )
        
        self._map_close_timer = QTimer(self)
        self._map_close_timer.setSingleShot(True)
        self._map_close_timer.setInterval(260)
        self._map_close_timer.timeout.connect(
            self._map_menu.close
        )
        
        self._hook_map_menu_autoclose()

        wl.addWidget(self.lb_mapdot, 0, Qt.AlignVCenter)
        wl.addWidget(self.lb_mapico, 0, Qt.AlignVCenter)
        wl.addWidget(self.lb_xk, 0)
        wl.addWidget(self.lb_xv, 1)
        
        wl.addWidget(self.btn_swap, 0)
        
        wl.addWidget(self.lb_yk, 0)
        wl.addWidget(self.lb_yv, 1)
        
        wl.addWidget(mk_sep(), 0)
        
        wl.addWidget(self.lb_zk, 0)
        wl.addWidget(self.lb_zv, 1)
        
        wl.addStretch(1)
        wl.addWidget(self.btn_mapdrop, 0)
        
        # Pulse timer (warn)
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(520)
        self._pulse_timer.timeout.connect(self._tick_pulse)
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

        # title = QLabel("Mapping", mapping)
        # title.setObjectName("mapHeadKey")
        mapping.setProperty("variant", "plain")

        m1.addWidget(self.w_mapchip, 1)
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

        # User selection from dropdown
        self.cmb_map_x.activated[int].connect(
            lambda _i: self._commit_map("x")
        )
        self.cmb_map_y.activated[int].connect(
            lambda _i: self._commit_map("y")
        )
        self.cmb_map_z.activated[int].connect(
            lambda _i: self._commit_map("z")
        )
        
        # User typed a value then pressed Enter / left field
        for w, cmb in (
            ("x", self.cmb_map_x),
            ("y", self.cmb_map_y),
            ("z", self.cmb_map_z),
        ):
            le = cmb.lineEdit()
            if le is not None:
                le.editingFinished.connect(
                    lambda ww=w: self._commit_map(ww)
                )
        self.btn_swap.clicked.connect(self.swap_xy_clicked)
        self.btn_data.toggled.connect(self.data_toggled)
        self.btn_view.toggled.connect(self.view_toggled)

        self.w_mapchip.clicked.connect(self._popup_map_menu)
        self.btn_mapdrop.clicked.connect(self._popup_map_menu)
        
    # -----------------------------
    # External setters
    # -----------------------------
    def _hook_map_menu_autoclose(self) -> None:
        """
        Auto-dismiss mapping popover when the cursor leaves it.
        Keeps it open while combo dropdowns / completer popups
        are visible.
        """
        try:
            self._map_menu.aboutToShow.connect(
                self._map_hover_timer.start
            )
            self._map_menu.aboutToHide.connect(
                self._map_hover_timer.stop
            )
            self._map_menu.aboutToHide.connect(
                self._map_close_timer.stop
            )
        except Exception:
            return
    
    
    def _map_menu_autoclose_tick(self) -> None:
        if not self._map_menu.isVisible():
            self._map_close_timer.stop()
            return
    
        if self._map_any_child_popup_open():
            self._map_close_timer.stop()
            return
    
        if self._map_menu_contains_cursor():
            self._map_close_timer.stop()
            return
    
        if not self._map_close_timer.isActive():
            self._map_close_timer.start()
    
    
    def _map_menu_contains_cursor(self) -> bool:
        pos = QCursor.pos()
    
        # Main menu window geometry
        if self._map_menu.frameGeometry().contains(pos):
            return True
    
        # Combo dropdown / completer popups (separate windows)
        for g in self._map_child_popup_geoms():
            if g.contains(pos):
                return True
    
        return False
    
    
    def _map_any_child_popup_open(self) -> bool:
        # If any dropdown is visible, do not auto-close.
        for cmb in (self.cmb_map_x, self.cmb_map_y, self.cmb_map_z):
            try:
                vw = cmb.view()
                if vw is not None:
                    win = vw.window()
                    if win is not None and win.isVisible():
                        return True
            except Exception:
                pass
    
            try:
                comp = cmb.completer()
                if isinstance(comp, QCompleter):
                    pv = comp.popup()
                    if pv is not None and pv.isVisible():
                        return True
            except Exception:
                pass
    
        return False
    
    
    def _map_child_popup_geoms(self):
        geoms = []
    
        for cmb in (self.cmb_map_x, self.cmb_map_y, self.cmb_map_z):
            try:
                vw = cmb.view()
                if vw is not None:
                    win = vw.window()
                    if win is not None and win.isVisible():
                        geoms.append(win.frameGeometry())
            except Exception:
                pass
    
            try:
                comp = cmb.completer()
                if isinstance(comp, QCompleter):
                    pv = comp.popup()
                    if pv is not None and pv.isVisible():
                        geoms.append(pv.frameGeometry())
            except Exception:
                pass
    
        return geoms

    def _popup_map_menu(self) -> None:
        self._sync_map_flow()
        pos = self.w_mapchip.mapToGlobal(
            self.w_mapchip.rect().bottomLeft()
        )
        self._map_menu.popup(pos)
        self._map_menu.setFocus(Qt.PopupFocusReason)
    
    def _tick_pulse(self) -> None:
        if str(self._map_state) != "warn":
            return
        self._pulse_on = not bool(self._pulse_on)
        self.lb_mapdot.setProperty(
            "pulse",
            "1" if self._pulse_on else "0",
        )
        self.lb_mapdot.style().unpolish(self.lb_mapdot)
        self.lb_mapdot.style().polish(self.lb_mapdot)
    
    
    def _set_pulse_enabled(self, on: bool) -> None:
        if on:
            if not self._pulse_timer.isActive():
                self._pulse_on = False
                self.lb_mapdot.setProperty("pulse", "0")
                self._pulse_timer.start()
        else:
            if self._pulse_timer.isActive():
                self._pulse_timer.stop()
            self._pulse_on = False
            self.lb_mapdot.setProperty("pulse", "0")
            self.lb_mapdot.style().unpolish(self.lb_mapdot)
            self.lb_mapdot.style().polish(self.lb_mapdot)
            
            
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

    def _show_combo_start(self, cmb: QComboBox) -> None:
        le = cmb.lineEdit()
        if le is None:
            return
        le.setCursorPosition(0)
        le.deselect()
    
    def set_available_columns(
        self,
        cols: Sequence[str],
    ) -> None:

        self._cols = list(cols or [])
        self._fill_map_combo(self.cmb_map_x, self._cols)
        self._fill_map_combo(self.cmb_map_y, self._cols)
        self._fill_map_combo(self.cmb_map_z, self._cols)
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
        was = self.btn_analytics.blockSignals(True)
        self.btn_analytics.setChecked(bool(checked))
        self.btn_analytics.blockSignals(was)
        self._update_ana_button_text()

    def set_xyz(self, *, x: str, y: str, z: str) -> None:
        # self.pk_x.set_value(x)
        # self.pk_y.set_value(y)
        # self.pk_z.set_value(z)
        # self._update_status()
        self._set_map_value(self.cmb_map_x, x)
        self._set_map_value(self.cmb_map_y, y)
        self._set_map_value(self.cmb_map_z, z)
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

    # -----------------------------
    # Analytics dropdown
    # -----------------------------
    def _ana_label(self, mode: str) -> str:
        m = str(mode or "").strip().lower()
        lab = {
            "selection": "Selection",
            "spatial": "Spatial",
            "sharpness": "Sharpness",
            "reliability": "Reliability",
            "inspector": "Inspector",
        }
        if not m:
            return ""
        return lab.get(m, m[:1].upper() + m[1:])
    
    def _update_ana_button_text(self) -> None:
        base = "Analytics"
        if not bool(self.btn_analytics.isChecked()):
            self.btn_analytics.setText(base)
            self.btn_analytics.setToolTip(base)
            return
    
        mode = getattr(self, "_ana_mode", "")
        label = self._ana_label(mode)
        if not label:
            self.btn_analytics.setText(base)
            self.btn_analytics.setToolTip(base)
            return
    
        txt = f"{base} · {label}"
        self.btn_analytics.setText(txt)
        self.btn_analytics.setToolTip(txt)
    
    def _build_analytics_menu(self) -> QMenu:
        menu = QMenu(self)
        menu.setToolTipsVisible(True)

        def add_item(key: str, label: str) -> None:
            act = menu.addAction(str(label))
            act.setCheckable(True)
            act.setData(str(key))
            self._ana_group.addAction(act)
            self._ana_acts[str(key)] = act

            act.triggered.connect(
                lambda _=False, k=key:
                self._emit_analytics_mode(str(k))
            )

        add_item("selection", "Selection")
        add_item("spatial", "Spatial")
        add_item("sharpness", "Sharpness")
        add_item("reliability", "Reliability")
        add_item("inspector", "Inspector")

        menu.addSeparator()

        a_hide = menu.addAction("Hide analytics")
        a_hide.triggered.connect(
            lambda: self._hide_analytics()
        )

        return menu

    def _emit_analytics_mode(self, mode: str) -> None:
        m = str(mode or "").strip().lower()
        if not m:
            return
    
        self._ana_mode = m
        self.set_analytics_mode(m)
    
        # visual indicator (checked) without re-entrancy
        was = self.btn_analytics.blockSignals(True)
        self.btn_analytics.setChecked(True)
        self.btn_analytics.blockSignals(was)
    
        # keep old pipeline working
        self.analytics_toggled.emit(True)
    
        # tell MapTab which view to open
        self.analytics_requested.emit(str(m))

    def _hide_analytics(self) -> None:
        was = self.btn_analytics.blockSignals(True)
        self.btn_analytics.setChecked(False)
        self.btn_analytics.blockSignals(was)
    
        self.analytics_toggled.emit(False)
        self._update_ana_button_text()

    def set_analytics_mode(self, mode: str) -> None:
        m = str(mode or "").strip().lower()
    
        if m:
            self._ana_mode = m
            act = self._ana_acts.get(m, None)
            if act is not None:
                act.blockSignals(True)
                act.setChecked(True)
                act.blockSignals(False)
    
        self._update_ana_button_text()

    # -----------------------------
    # Mapping popover (premium)
    # -----------------------------
    def _build_mapping_menu(self) -> QMenu:
        menu = QMenu(self)
        menu.setObjectName("mapMapMenu")
        menu.setToolTipsVisible(True)
        menu.setAttribute(Qt.WA_TranslucentBackground, True)
        menu.setWindowFlags(
            menu.windowFlags()
            | Qt.FramelessWindowHint
            | Qt.NoDropShadowWindowHint
        )
    
        pop = QFrame(menu)
        pop.setObjectName("mapMapPopover")
        pop.setFrameShape(QFrame.NoFrame)
        pop.setMinimumWidth(360)
    
        lay = QVBoxLayout(pop)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(8)
    
        hdr = QWidget(pop)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
    
        lb = QLabel("Mapping", hdr)
        lb.setObjectName("mapMapTitle")
    
        self.lb_map_hint = QLabel("", hdr)
        self.lb_map_hint.setObjectName("mapMapHint")
        self.lb_map_hint.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
    
        hl.addWidget(lb, 0)
        hl.addWidget(self.lb_map_hint, 1)
        lay.addWidget(hdr, 0)
    
        row = QWidget(pop)
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        
        lx = QLabel("X", row); lx.setObjectName("mapMapKey")
        ly = QLabel("Y", row); ly.setObjectName("mapMapKey")
        lz = QLabel("Z", row); lz.setObjectName("mapMapKey")
        
        self.cmb_map_x = self._make_map_combo(row)
        self.cmb_map_y = self._make_map_combo(row)
        self.cmb_map_z = self._make_map_combo(row)
        
        self.cmb_map_x.setToolTip("X / longitude column")
        self.cmb_map_y.setToolTip("Y / latitude column")
        self.cmb_map_z.setToolTip("Value / color column")
        
        bswap = QToolButton(row)
        bswap.setObjectName("miniAction")
        bswap.setProperty("role", "mapHead")
        bswap.setAutoRaise(True)
        bswap.setToolTip("Swap X and Y")
        bswap.setIcon(
            try_icon(
                "swap.svg",
                fallback=self.style().standardIcon(
                    QStyle.SP_BrowserReload
                ),
            )
        )
        bswap.setIconSize(QSize(15, 15))
        bswap.setFixedSize(28, 28)
        bswap.clicked.connect(self.swap_xy_clicked.emit)

        rl.addWidget(lx, 0)
        rl.addWidget(self.cmb_map_x, 1)
        
        rl.addWidget(bswap, 0)
        
        rl.addWidget(ly, 0)
        rl.addWidget(self.cmb_map_y, 1)
        
        rl.addSpacing(6)
        
        rl.addWidget(lz, 0)
        rl.addWidget(self.cmb_map_z, 1)
        
        lay.addWidget(row, 0)
            
        actw = QWidget(pop)
        al = QHBoxLayout(actw)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(8)
        
        breset = QToolButton(actw)
        breset.setObjectName("miniAction")
        breset.setProperty("role", "mapHead")
        breset.setAutoRaise(True)
        breset.setToolTip("Reset X/Y/Z mapping")
        breset.setIcon(
            try_icon(
                "reset.svg",
                fallback=self._clear_icon(),
            )
        )
        breset.setIconSize(QSize(15, 15))
        breset.setFixedSize(28, 28)
        breset.clicked.connect(
            self.reset_mapping_clicked.emit
        )
        
        bok = QToolButton(actw)
        bok.setObjectName("miniAction")
        bok.setProperty("role", "mapHead")
        bok.setProperty("accent", "true")
        bok.setAutoRaise(True)
        bok.setToolTip("OK")
        bok.setIcon(
            try_icon(
                "ok.svg",
                fallback=self.style().standardIcon(
                    QStyle.SP_DialogApplyButton
                ),
            )
        )
        bok.setIconSize(QSize(15, 15))
        bok.setFixedSize(28, 28)
        bok.clicked.connect(menu.close)
        
        al.addStretch(1)
        al.addWidget(breset, 0)
        al.addWidget(bok, 0)
        
        lay.addWidget(actw, 0)
    
        wa = QWidgetAction(menu)
        wa.setDefaultWidget(pop)
        menu.addAction(wa)
        
        self._sync_map_flow()
        
        return menu

    def _sync_map_flow(self) -> None:
        x = self._map_val(self.cmb_map_x)
        y = self._map_val(self.cmb_map_y)
    
        self.cmb_map_y.setEnabled(bool(x))
        self.cmb_map_z.setEnabled(bool(x and y))
    
    def _make_map_combo(self, parent: QWidget) -> QComboBox:
        cmb = QComboBox(parent)
        cmb.setObjectName("mapMapCombo")
        cmb.setEditable(True)
        cmb.setInsertPolicy(QComboBox.NoInsert)
        cmb.setMinimumHeight(30)
        cmb.setMinimumWidth(140)
        cmb.setMaximumWidth(240)
        
        view = cmb.view()
        view.setMinimumWidth(260)
        view.setMaximumWidth(320)
    
        comp = cmb.completer()
        if isinstance(comp, QCompleter):
            comp.setCaseSensitivity(Qt.CaseInsensitive)
            comp.setFilterMode(Qt.MatchContains)
    
        self._fill_map_combo(cmb, [])
        return cmb

    def _mapping_full(self) -> bool:
        x = self._map_val(self.cmb_map_x)
        y = self._map_val(self.cmb_map_y)
        z = self._map_val(self.cmb_map_z)
    
        if not (x and y and z):
            return False
    
        cols = list(self._cols or [])
        if cols and (
            (x not in cols)
            or (y not in cols)
            or (z not in cols)
        ):
            return False
    
        return True
    
    
    def _commit_map(self, which: str) -> None:
        self._emit_map_xyz(which)
    
        w = str(which or "").strip().lower()
    
        # Step-by-step UX: X -> focus Y, Y -> focus Z
        if w == "x":
            try:
                self.cmb_map_y.setFocus()
            except Exception:
                pass
            return
    
        if w == "y":
            try:
                self.cmb_map_z.setFocus()
            except Exception:
                pass
            return
    
        # Close like the old menu once Z is chosen
        if w == "z" and self._mapping_full():
            QTimer.singleShot(0, self._map_menu.close)

    def _map_val(self, cmb: QComboBox) -> str:
        v = cmb.currentData()
        if not v:
            v = cmb.currentText()
        return str(v or "").strip()

    def _fill_map_combo(
        self,
        cmb: QComboBox,
        cols: Sequence[str],
    ) -> None:
        cur = self._map_val(cmb)
        items = [str(c) for c in (cols or [])]
    
        cmb.blockSignals(True)
        cmb.clear()
        cmb.addItem("—", "")
        for c in items:
            cmb.addItem(c, c)
    
        if cur and cur not in items:
            warn = self.style().standardIcon(
                QStyle.SP_MessageBoxWarning
            )
            cmb.insertItem(1, cur, cur)
            cmb.setItemIcon(1, warn)
        
            cmb.setItemData(
                1,
                "Missing column in current dataset",
                Qt.ToolTipRole,
            )
        
            f = QFont(cmb.font())
            f.setItalic(True)
            cmb.setItemData(1, f, Qt.FontRole)
        
            cmb.setItemData(
                1,
                QColor(180, 120, 30),
                Qt.ForegroundRole,
            )
        
            cmb.setCurrentIndex(1)
            self._show_combo_start(cmb)
            
        elif cur:
            idx = cmb.findData(cur)
            if idx >= 0:
                cmb.setCurrentIndex(idx)
        else:
            cmb.setCurrentIndex(0)
    
        cmb.blockSignals(False)

    
    def _set_map_value(self, cmb: QComboBox, v: str) -> None:
        v = str(v or "").strip()
        if not v:
            cmb.blockSignals(True)
            cmb.setCurrentIndex(0)
            cmb.blockSignals(False)
            return
    
        idx = cmb.findData(v)
        if idx < 0:
            cmb.blockSignals(True)
            # cmb.insertItem(1, f"{v} [—]", v)
            warn = self.style().standardIcon(QStyle.SP_MessageBoxWarning)
            cmb.insertItem(1, v, v)
            cmb.setItemIcon(1, warn)
            cmb.setItemData(
                1,
                "Missing column in current dataset",
                Qt.ToolTipRole,
            )
            f = QFont(cmb.font())
            f.setItalic(True)
            cmb.setItemData(1, f, Qt.FontRole)
            cmb.setItemData(1, QColor(180, 120, 30), Qt.ForegroundRole)

            cmb.setCurrentIndex(1)
            self._show_combo_start(cmb)
            cmb.blockSignals(False)
            return
    
        cmb.blockSignals(True)
        cmb.setCurrentIndex(idx)
        self._show_combo_start(cmb)
        cmb.blockSignals(False)


    def _emit_map_xyz(self, which: str) -> None:
        w = str(which or "").strip().lower()
        if w == "x":
            self.x_changed.emit(self._map_val(self.cmb_map_x))
        elif w == "y":
            self.y_changed.emit(self._map_val(self.cmb_map_y))
        elif w == "z":
            self.z_changed.emit(self._map_val(self.cmb_map_z))
            
        self._sync_map_flow()
        self._update_status()
        
    
    def _short_col(self, v: str, n: int = 14) -> str:
        s = str(v or "").strip()
        if not s:
            return "—"
        if len(s) <= int(n):
            return s
        return s[: int(n) - 1] + "…"

    def _update_mapping_pill(self, map_ok: bool) -> None:
        x = self._map_val(self.cmb_map_x)
        y = self._map_val(self.cmb_map_y)
        z = self._map_val(self.cmb_map_z)
    
        sx = self._short_col(x)
        sy = self._short_col(y)
        sz = self._short_col(z)
    
        if map_ok:
            state = "ok"
            hint = "OK"
        elif x or y or z:
            state = "warn"
            hint = "Pick X/Y"
        else:
            state = "info"
            hint = "Not set"
    
        self._map_state = state
    
        # Pro display: keys bold, values normal
        self.lb_xv.set_full_text(sx)
        self.lb_yv.set_full_text(sy)
        self.lb_zv.set_full_text(sz)
    
        tip = (
            "Mapping columns\n"
            f"X: {x or '—'}\n"
            f"Y: {y or '—'}\n"
            f"Z: {z or '—'}"
        )
        self.w_mapchip.setToolTip(tip)
    
        # State drives capsule + dot
        self.w_mapchip.setProperty("state", state)
        self.lb_mapdot.setProperty("state", state)
    
        if hasattr(self, "lb_map_hint"):
            self.lb_map_hint.setText(str(hint))
    
        for w in (self.w_mapchip, self.lb_mapdot):
            w.style().unpolish(w)
            w.style().polish(w)
    
        self._set_pulse_enabled(state == "warn")
        

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
        x = self._map_val(self.cmb_map_x)
        y = self._map_val(self.cmb_map_y)
        if not x or not y:
            return False
    
        cols = list(self._cols or [])
        if cols and (x not in cols or y not in cols):
            return False
    
        return True

    def _update_status(self) -> None:
        n = int(self._n_sets)
    
        x = self._map_val(self.cmb_map_x)
        y = self._map_val(self.cmb_map_y)
        z = self._map_val(self.cmb_map_z)
    
        have_map = bool(x or y or z)
        have_data = n > 0
    
        if not have_data and not have_map:
            self.lb_status.setVisible(False)
            self.lb_status.setText("")
            self._update_mapping_pill(False)
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
    
        self._update_mapping_pill(map_ok)

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

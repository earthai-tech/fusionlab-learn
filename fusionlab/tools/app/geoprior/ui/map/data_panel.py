# geoprior/ui/map/data_panel.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.map.data_panel

Auto-hide data/navigation panel (A).

Features (now)
--------------
- Data source: Auto | Manual
- Auto: scan results_root for city/jobs/files
- Manual: import CSVs into results_root/_datasets
- Emits selected file paths (UI selection only)

Notes
-----
We persist UI-only state via store._extra:
- map.data_source      : "auto" | "manual"
- map.manual_files     : list[str]
- map.selected_files   : list[str]
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Set
import tempfile

from PyQt5.QtCore import (
    QEasingCurve,
    QEvent,
    QObject,
    QPropertyAnimation,
    QTimer,
    Qt,
    pyqtSignal,
)

from PyQt5.QtWidgets import (
    QFrame,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    # QPushButton,
    QStackedWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QGroupBox,
    # QSplitter,
    QGridLayout,
    QLineEdit,
    QStyle, 
    QScrollArea,
    QFormLayout,
    QHeaderView,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QLayout,

)

from ...config.store import GeoConfigStore
from ..icon_utils import try_icon
from .utils import (
    scan_results_root,
    MapCity,
    MapFile,
    MapJob,
    unique_str,
    load_forecast_meta,
    ForecastMeta,
)

ROLE_PATH = Qt.UserRole
ROLE_KIND = Qt.UserRole + 1

class AutoHidePanel(QFrame):
    """
    Base auto-hide panel with a handle and pin mode.

    Emits width_changed while animating, so the parent
    QSplitter can reallocate space to the map.
    """

    width_changed = pyqtSignal(int)
    pinned_changed = pyqtSignal(bool)

    def __init__(
        self,
        *,
        title: str,
        side: str,
        expanded_w: int = 300,
        handle_w: int = 22,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._title = str(title or "")
        self._side = str(side or "left").lower()

        self._expanded_w = int(expanded_w)
        self._handle_w = int(handle_w)

        self._pinned = False
        self._expanded = True
        self._hover_enabled = True

        self._anim = QPropertyAnimation(
            self,
            b"maximumWidth",
        )
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(
            self._on_anim_width,
        )
        self._anim.finished.connect(self._on_anim_done)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._on_hide_timer)

        self._build_ui()
        self.expand(immediate=True)

    # -----------------------------
    # Public API
    # -----------------------------
    def handle_width(self) -> int:
        return int(self._handle_w)

    def expanded_width(self) -> int:
        return int(self._expanded_w)

    def toggle_pinned(self) -> None:
        self.set_pinned(not self._pinned)

    def set_pinned(self, pinned: bool) -> None:
        self._pinned = bool(pinned)
        self._update_pin_ui()

        if self._pinned:
            self.setMinimumWidth(self._expanded_w)
            self.expand()
        else:
            self.setMinimumWidth(self._handle_w)
            if not self.underMouse():
                self.collapse()

        self.pinned_changed.emit(self._pinned)

    def is_pinned(self) -> bool:
        return bool(self._pinned)

    def set_hover_enabled(self, enabled: bool) -> None:
        self._hover_enabled = bool(enabled)
        if not self._hover_enabled:
            self._hide_timer.stop()

    def expand(self, *, immediate: bool = False) -> None:
        self._hide_timer.stop()
        self._expanded = True

        self._content.setVisible(True)

        if self._pinned:
            self.setMinimumWidth(self._expanded_w)
        else:
            self.setMinimumWidth(self._handle_w)

        if immediate:
            self._stop_anim()
            self.setMaximumWidth(self._expanded_w)
            self.width_changed.emit(self._expanded_w)
            return

        self._animate_to(self._expanded_w)

    def collapse(
        self,
        *,
        immediate: bool = False,
        force: bool = False,
    ) -> None:
        if self._pinned and not force:
            return

        self._expanded = False
        self.setMinimumWidth(self._handle_w)

        if immediate:
            self._stop_anim()
            self._content.setVisible(False)
            self.setMaximumWidth(self._handle_w)
            self.width_changed.emit(self._handle_w)
            return

        self._animate_to(self._handle_w)

    def is_expanded(self) -> bool:
        return bool(self._expanded)

    # -----------------------------
    # Qt events
    # -----------------------------
    def enterEvent(self, event: QEvent) -> None:
        super().enterEvent(event)
        if not self._hover_enabled:
            return
        self.expand()

    def leaveEvent(self, event: QEvent) -> None:
        super().leaveEvent(event)
        if not self._hover_enabled:
            return
        if self._pinned:
            return
        self._hide_timer.start(260)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self._btn_handle:
            if event.type() == QEvent.MouseButtonDblClick:
                self.toggle_pinned()
                return True
        return super().eventFilter(obj, event)

    # -----------------------------
    # Internals
    # -----------------------------
    def _build_ui(self) -> None:
        self.setObjectName("AutoHidePanel")
        self.setFrameShape(QFrame.NoFrame)

        self._handle = QWidget(self)
        hl = QVBoxLayout(self._handle)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)

        self._btn_handle = QToolButton(self._handle)
        self._btn_handle.setText("≡")
        self._btn_handle.setToolTip(self._title)
        self._btn_handle.clicked.connect(self.expand)
        self._btn_handle.installEventFilter(self)

        hl.addWidget(self._btn_handle, 0)
        hl.addStretch(1)

        self._content = QWidget(self)
        cl = QVBoxLayout(self._content)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(10)

        top = QWidget(self._content)
        tl = QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(8)

        self._lb = QLabel(self._title, top)

        # Pin as small icon toggle
        self._btn_pin = QToolButton(top)
        self._btn_pin.setCheckable(True)
        self._btn_pin.toggled.connect(self.set_pinned)
        self._btn_pin.setToolButtonStyle(
            Qt.ToolButtonTextOnly,
        )

        tl.addWidget(self._lb, 1)
        tl.addWidget(self._btn_pin, 0)
        cl.addWidget(top)

        self.body = QWidget(self._content)
        cl.addWidget(self.body, 1)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if self._side == "right":
            root.addWidget(self._content, 1)
            root.addWidget(self._handle, 0)
        else:
            root.addWidget(self._handle, 0)
            root.addWidget(self._content, 1)

        self.setMinimumWidth(self._handle_w)
        self.setMaximumWidth(self._expanded_w)
        self._update_pin_ui()

    def _update_pin_ui(self) -> None:
        self._btn_pin.blockSignals(True)
        self._btn_pin.setChecked(self._pinned)
        self._btn_pin.blockSignals(False)

        if self._pinned:
            self._btn_pin.setText("📌")
            self._btn_pin.setToolTip("Pinned")
            self._btn_pin.setStyleSheet(
                "QToolButton{"
                "font-weight:600;"
                "padding:2px 8px;"
                "border-radius:9px;"
                "border:1px solid "
                "rgba(46,49,145,0.60);"
                "background:rgba(46,49,145,0.12);"
                "}"
            )
        else:
            self._btn_pin.setText("📌")
            self._btn_pin.setToolTip("Pin panel")
            self._btn_pin.setStyleSheet("")

    def _animate_to(self, w: int) -> None:
        w = int(max(self._handle_w, w))
        self._stop_anim()
        self._anim.setStartValue(self.maximumWidth())
        self._anim.setEndValue(w)
        self._anim.start()

    def _stop_anim(self) -> None:
        if self._anim.state() == QPropertyAnimation.Running:
            self._anim.stop()

    def _on_anim_width(self, v) -> None:
        try:
            w = int(v)
        except Exception:
            w = int(self.maximumWidth())
        self.width_changed.emit(w)

    def _on_anim_done(self) -> None:
        if not self._expanded:
            self._content.setVisible(False)
            self.setMaximumWidth(self._handle_w)
            self.width_changed.emit(self._handle_w)

    def _on_hide_timer(self) -> None:
        if self._pinned:
            return
        if self.underMouse():
            return
        self.collapse()


class AutoHideDataPanel(AutoHidePanel):
    """
    Left data panel (A).
    """

    selection_changed = pyqtSignal(object)
    columns_changed = pyqtSignal(object)
    active_changed = pyqtSignal(object)
    sampling_changed = pyqtSignal(object)


    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            title="Data",
            side="left",
            expanded_w=320,
            parent=parent,
        )
        self.store = store

        self._selected: Set[str] = set()
        self._cities: List[MapCity] = []
        self._active_path: Optional[Path] = None
        self._active_meta: Optional[ForecastMeta] = None
        self._tmp = tempfile.TemporaryDirectory(
            prefix="fusionlab_map_",
        )
        self._manual_src = {}

        self._build_body()
        self._sync_from_store()
        self._refresh_auto()
        
    def _tmp_root(self) -> Path:
        return Path(self._tmp.name)
    
    def closeEvent(self, event) -> None:
        try:
            self._tmp.cleanup()
        except Exception:
            pass
        super().closeEvent(event)

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_body(self) -> None:
        outer = QVBoxLayout(self.body)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
    
        # ---- Scroll wrapper (key change) ----
        self.scroll = QScrollArea(self.body)
        self.scroll.setObjectName("mapDataScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        outer.addWidget(self.scroll, 1)
    
        host = QWidget(self.scroll)
        host.setObjectName("mapDataHost")
        self.scroll.setWidget(host)
    
        root = QVBoxLayout(host)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        root.setSizeConstraint(QLayout.SetMinimumSize)
        
        self.scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
    
        # -----------------------------
        # Toolbar card (Source + Refresh)
        # -----------------------------
        bar = QFrame(host)
        bar.setObjectName("mapPanelCard")
        bar.setProperty("role", "toolbar")
        bar_l = QHBoxLayout(bar)
        bar_l.setContentsMargins(10, 8, 10, 8)
        bar_l.setSpacing(8)
    
        lb_src = QLabel("Source", bar)
        lb_src.setObjectName("mapFieldLabel")
    
        self.cmb_source = QComboBox(bar)
        self.cmb_source.addItems(["Auto", "Manual"])
        self.cmb_source.setMinimumHeight(28)
    
        self.btn_refresh = QToolButton(bar)
        self.btn_refresh.setObjectName("miniAction")
        self.btn_refresh.setAutoRaise(True)
        self.btn_refresh.setToolTip("Refresh")
        self.btn_refresh.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
    
        bar_l.addWidget(lb_src, 0)
        bar_l.addWidget(self.cmb_source, 1)
        bar_l.addWidget(self.btn_refresh, 0)
    
        root.addWidget(bar, 0)
    
        # -----------------------------
        # Datasets card (Filter + Tree stack)
        # -----------------------------
        ds = QFrame(host)
        ds.setObjectName("mapPanelCard")
        ds.setProperty("role", "datasets")
        ds.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
        ds_l = QVBoxLayout(ds)
        ds_l.setContentsMargins(10, 10, 10, 10)
        ds_l.setSpacing(8)
    
        title_row = QWidget(ds)
        tr = QHBoxLayout(title_row)
        tr.setContentsMargins(0, 0, 0, 0)
        tr.setSpacing(8)
    
        lb = QLabel("Datasets", title_row)
        lb.setObjectName("mapSectionTitle")
    
        self.lb_count = QLabel("", title_row)
        self.lb_count.setObjectName("mapCountChip")
    
        tr.addWidget(lb, 1)
        tr.addWidget(self.lb_count, 0)
        ds_l.addWidget(title_row, 0)
    
        self.ed_filter = QLineEdit(ds)
        self.ed_filter.setObjectName("mapSearch")
        self.ed_filter.setPlaceholderText("Filter datasets…")
        self.ed_filter.setClearButtonEnabled(True)
        # Leading filter icon (SVG first, fallback to Qt standard icon)
        ico = try_icon("filter.svg")
        if ico is None:
            ico = self.style().standardIcon(
                QStyle.SP_FileDialogContentsView
            )
        
        act = self.ed_filter.addAction(
            ico,
            QLineEdit.LeadingPosition,
        )
        act.setToolTip("Filter datasets")
        act.triggered.connect(self.ed_filter.setFocus)

        ds_l.addWidget(self.ed_filter, 0)
    
        self.stack = QStackedWidget(ds)
        self.page_auto = self._build_auto_page(self.stack)
        self.page_manual = self._build_manual_page(self.stack)
        self.stack.addWidget(self.page_auto)
        self.stack.addWidget(self.page_manual)
        ds_l.addWidget(self.stack, 1)
    
        root.addWidget(ds, 1)

        # -----------------------------
        # Sampling + Details
        # One shared horizontal scroll
        # -----------------------------
        self._bottom_scroll = QScrollArea(host)
        self._bottom_scroll.setObjectName(
            "mapDataBottomScroll"
        )
        self._bottom_scroll.setFrameShape(QFrame.NoFrame)
        self._bottom_scroll.setWidgetResizable(True)
        self._bottom_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self._bottom_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        root.addWidget(self._bottom_scroll, 0)
        
        bottom = QWidget(self._bottom_scroll)
        bottom.setObjectName("mapDataBottomHost")
        self._bottom_scroll.setWidget(bottom)
        
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)
        bl.setSizeConstraint(QLayout.SetMinimumSize)
        
        self.sampling = self._build_sampling_box(bottom)
        self.details = self._build_details_box(bottom)
        
        bl.addWidget(self.sampling, 0)
        bl.addWidget(self.details, 0)
        
        min_w = max(
            self.sampling.sizeHint().width(),
            self.details.sizeHint().width(),
        )
        bottom.setMinimumWidth(min_w)
        self._bottom_scroll.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        QTimer.singleShot(0, self._fix_bottom_scroll_height)

        # ---- Signals ----
        self.cmb_source.currentIndexChanged.connect(self._on_source_changed)
        self.btn_refresh.clicked.connect(self._on_refresh)
        self.ed_filter.textChanged.connect(self._on_filter_changed)
    
        # default view
        self.stack.setCurrentWidget(self.page_auto)
        self._details_set_empty()
        
    def _fix_bottom_scroll_height(self) -> None:
        s = getattr(self, "_bottom_scroll", None)
        if s is None:
            return
    
        w = s.widget()
        if w is None:
            return
    
        h = int(w.sizeHint().height())
        if h <= 0:
            return
    
        s.setMinimumHeight(h)
        s.setMaximumHeight(h)
        
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_bottom_scroll"):
            QTimer.singleShot(
                0,
                self._fix_bottom_scroll_height,
            )

    def _build_sampling_box(self, parent: QWidget) -> QFrame:
        box = QFrame(parent)
        box.setObjectName("mapPanelCard")
        box.setProperty("role", "sampling")

        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        head = QWidget(box)
        hl = QHBoxLayout(head)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)

        title = QLabel("Sampling for visualization", head)
        title.setObjectName("mapSectionTitle")

        self.lb_samp = QLabel("", head)
        self.lb_samp.setObjectName("mapCountChip")

        hl.addWidget(title, 1)
        hl.addWidget(self.lb_samp, 0)
        v.addWidget(head, 0)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        v.addLayout(grid)

        self.cmb_samp_mode = QComboBox(box)
        self.cmb_samp_mode.setMinimumHeight(28)
        self.cmb_samp_mode.addItem("Auto", "auto")
        self.cmb_samp_mode.addItem("Off", "off")
        self.cmb_samp_mode.addItem("Always", "always")
        self.cmb_samp_mode.setToolTip(
            "Auto: only sample when dataset is huge."
        )

        self.cmb_samp_method = QComboBox(box)
        self.cmb_samp_method.setMinimumHeight(28)
        self.cmb_samp_method.addItem("Grid (spatial)", "grid")
        self.cmb_samp_method.addItem("Random", "random")
        self.cmb_samp_method.setToolTip(
            "Grid keeps spatial coverage."
        )

        self.sp_samp_max = QSpinBox(box)
        self.sp_samp_max.setRange(2000, 5000000)
        self.sp_samp_max.setSingleStep(2000)
        self.sp_samp_max.setMinimumHeight(28)

        self.sp_samp_seed = QSpinBox(box)
        self.sp_samp_seed.setRange(0, 999999)
        self.sp_samp_seed.setMinimumHeight(28)

        self.sp_samp_cell = QDoubleSpinBox(box)
        self.sp_samp_cell.setRange(0.05, 200.0)
        self.sp_samp_cell.setDecimals(2)
        self.sp_samp_cell.setSingleStep(0.25)
        self.sp_samp_cell.setMinimumHeight(28)
        self.sp_samp_cell.setToolTip(
            "Cell size (km) for Grid sampling."
        )

        self.sp_samp_per = QSpinBox(box)
        self.sp_samp_per.setRange(1, 1000)
        self.sp_samp_per.setSingleStep(5)
        self.sp_samp_per.setMinimumHeight(28)
        self.sp_samp_per.setToolTip(
            "Max points per spatial cell."
        )

        self.chk_samp_hot = QCheckBox(
            "Apply same sample to hotspots",
            box,
        )

        grid.addWidget(QLabel("Mode", box), 0, 0)
        grid.addWidget(self.cmb_samp_mode, 0, 1)

        grid.addWidget(QLabel("Method", box), 1, 0)
        grid.addWidget(self.cmb_samp_method, 1, 1)

        grid.addWidget(QLabel("Max points", box), 2, 0)
        grid.addWidget(self.sp_samp_max, 2, 1)

        grid.addWidget(QLabel("Seed", box), 3, 0)
        grid.addWidget(self.sp_samp_seed, 3, 1)

        grid.addWidget(QLabel("Cell (km)", box), 4, 0)
        grid.addWidget(self.sp_samp_cell, 4, 1)

        grid.addWidget(QLabel("Max / cell", box), 5, 0)
        grid.addWidget(self.sp_samp_per, 5, 1)

        v.addWidget(self.chk_samp_hot, 0)

        hint = QLabel(
            "Used by Map points, Analytics, and Hotspots.",
            box,
        )
        hint.setWordWrap(True)
        hint.setObjectName("mapFieldHint")
        v.addWidget(hint, 0)

        self.cmb_samp_mode.currentIndexChanged.connect(
            self._on_sampling_ui_changed,
        )
        self.cmb_samp_method.currentIndexChanged.connect(
            self._on_sampling_ui_changed,
        )
        self.sp_samp_max.valueChanged.connect(
            self._on_sampling_ui_changed,
        )
        self.sp_samp_seed.valueChanged.connect(
            self._on_sampling_ui_changed,
        )
        self.sp_samp_cell.valueChanged.connect(
            self._on_sampling_ui_changed,
        )
        self.sp_samp_per.valueChanged.connect(
            self._on_sampling_ui_changed,
        )
        self.chk_samp_hot.toggled.connect(
            self._on_sampling_ui_changed,
        )

        self._sync_sampling_from_store()
        return box

    def _sync_sampling_from_store(self) -> None:
        mode = str(self.store.get(
            "map.sampling.mode",
            "auto",
        ) or "auto").strip().lower()

        method = str(self.store.get(
            "map.sampling.method",
            "grid",
        ) or "grid").strip().lower()

        try:
            maxp = int(self.store.get(
                "map.sampling.max_points",
                80000,
            ))
        except Exception:
            maxp = 80000

        try:
            seed = int(self.store.get(
                "map.sampling.seed",
                0,
            ))
        except Exception:
            seed = 0

        try:
            cell = float(self.store.get(
                "map.sampling.cell_km",
                1.0,
            ))
        except Exception:
            cell = 1.0

        try:
            per = int(self.store.get(
                "map.sampling.max_per_cell",
                50,
            ))
        except Exception:
            per = 50

        hot = bool(self.store.get(
            "map.sampling.apply_hotspots",
            True,
        ))

        for w in (
            self.cmb_samp_mode,
            self.cmb_samp_method,
            self.sp_samp_max,
            self.sp_samp_seed,
            self.sp_samp_cell,
            self.sp_samp_per,
        ):
            w.blockSignals(True)

        self.chk_samp_hot.blockSignals(True)

        try:
            self._set_combo_data(self.cmb_samp_mode, mode)
            self._set_combo_data(self.cmb_samp_method, method)

            self.sp_samp_max.setValue(max(1, maxp))
            self.sp_samp_seed.setValue(max(0, seed))
            self.sp_samp_cell.setValue(max(0.001, cell))
            self.sp_samp_per.setValue(max(1, per))
            self.chk_samp_hot.setChecked(bool(hot))

            chip = f"{mode}/{method}  ≤{maxp}"
            self.lb_samp.setText(chip)
        finally:
            for w in (
                self.cmb_samp_mode,
                self.cmb_samp_method,
                self.sp_samp_max,
                self.sp_samp_seed,
                self.sp_samp_cell,
                self.sp_samp_per,
            ):
                w.blockSignals(False)
            self.chk_samp_hot.blockSignals(False)

    def _on_sampling_ui_changed(self) -> None:
        mode = str(self.cmb_samp_mode.currentData() or "auto")
        method = str(self.cmb_samp_method.currentData() or "grid")

        snap = {
            "mode": mode,
            "method": method,
            "max_points": int(self.sp_samp_max.value()),
            "seed": int(self.sp_samp_seed.value()),
            "cell_km": float(self.sp_samp_cell.value()),
            "max_per_cell": int(self.sp_samp_per.value()),
            "apply_hotspots": bool(self.chk_samp_hot.isChecked()),
        }

        with self.store.batch():
            self.store.set("map.sampling.mode", snap["mode"])
            self.store.set("map.sampling.method", snap["method"])
            self.store.set("map.sampling.max_points", snap["max_points"])
            self.store.set("map.sampling.seed", snap["seed"])
            self.store.set("map.sampling.cell_km", snap["cell_km"])
            self.store.set(
                "map.sampling.max_per_cell",
                snap["max_per_cell"],
            )
            self.store.set(
                "map.sampling.apply_hotspots",
                snap["apply_hotspots"],
            )

        self.lb_samp.setText(
            f"{snap['mode']}/{snap['method']}  ≤{snap['max_points']}"
        )
        self.sampling_changed.emit(dict(snap))


    def _build_auto_page(self, parent: QWidget) -> QWidget:
        w = QWidget(parent)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
    
        self.tree = QTreeWidget(w)
        self.tree.setObjectName("mapTree")
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Item", "Info"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setIndentation(14)
        self.tree.setRootIsDecorated(True)
        self.tree.setAnimated(True)
        self.tree.setSelectionBehavior(QTreeWidget.SelectRows)
        self.tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.tree.setAllColumnsShowFocus(False)
        self.tree.setExpandsOnDoubleClick(True)
        self.tree.setSortingEnabled(False)
    
        hdr = self.tree.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setHighlightSections(False)
        hdr.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    
        # signals
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.itemSelectionChanged.connect(self._on_active_item_changed)
    
        lay.addWidget(self.tree, 1)
        return w
    
    
    def _build_manual_page(self, parent: QWidget) -> QWidget:
        w = QWidget(parent)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
    
        self.tree_manual = QTreeWidget(w)
        self.tree_manual.setObjectName("mapTree")
        self.tree_manual.setColumnCount(2)
        self.tree_manual.setHeaderLabels(["Dataset", "Path"])
        self.tree_manual.setAlternatingRowColors(True)
        self.tree_manual.setUniformRowHeights(True)
        self.tree_manual.setIndentation(14)
        self.tree_manual.setRootIsDecorated(False)
        self.tree_manual.setAnimated(True)
        self.tree_manual.setSelectionBehavior(QTreeWidget.SelectRows)
        self.tree_manual.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.tree_manual.setAllColumnsShowFocus(False)
        self.tree_manual.setExpandsOnDoubleClick(False)
        self.tree_manual.setSortingEnabled(False)
    
        hdr = self.tree_manual.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setHighlightSections(False)
        hdr.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    
        self.tree_manual.itemChanged.connect(self._on_manual_item_changed)
        self.tree_manual.itemSelectionChanged.connect(self._on_active_item_changed)
    
        # ---- Footer actions (modern mini buttons) ----
        btns = QWidget(w)
        bl = QHBoxLayout(btns)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)
    
        self.btn_add = QToolButton(btns)
        self.btn_add.setObjectName("miniAction")
        self.btn_add.setAutoRaise(True)
        self.btn_add.setToolTip("Add CSV files")
        self.btn_add.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
    
        self.btn_add_dir = QToolButton(btns)
        self.btn_add_dir.setObjectName("miniAction")
        self.btn_add_dir.setAutoRaise(True)
        self.btn_add_dir.setToolTip("Add folder")
        self.btn_add_dir.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
    
        self.btn_remove = QToolButton(btns)
        self.btn_remove.setObjectName("miniAction")
        self.btn_remove.setAutoRaise(True)
        self.btn_remove.setToolTip("Remove selected")
        self.btn_remove.setIcon(
            self.style().standardIcon(
                QStyle.SP_TrashIcon
                if hasattr(QStyle, "SP_TrashIcon")
                else QStyle.SP_DialogDiscardButton
            )
        )
    
        bl.addWidget(self.btn_add, 0)
        bl.addWidget(self.btn_add_dir, 0)
        bl.addWidget(self.btn_remove, 0)
        bl.addStretch(1)
    
        lay.addWidget(self.tree_manual, 1)
        lay.addWidget(btns, 0)
    
        self.btn_add.clicked.connect(self._on_add_csv)
        self.btn_add_dir.clicked.connect(self._on_add_folder)
        self.btn_remove.clicked.connect(self._on_remove_manual)
    
        return w


    def _build_details_box(self, parent: QWidget) -> QFrame:
        box = QFrame(parent)
        box.setObjectName("mapPanelCard")
        box.setProperty("role", "details")
    
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)
    
        # Header row
        head = QWidget(box)
        hl = QHBoxLayout(head)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
    
        title = QLabel("Forecast details", head)
        title.setObjectName("mapSectionTitle")
    
        self.btn_guess = QToolButton(head)
        self.btn_guess.setObjectName("miniAction")
        self.btn_guess.setAutoRaise(True)
        self.btn_guess.setToolTip("Guess mapping")
        self.btn_guess.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
    
        self.btn_check = QToolButton(head)
        self.btn_check.setObjectName("miniAction")
        self.btn_check.setAutoRaise(True)
        self.btn_check.setToolTip("Validate mapping")
        self.btn_check.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
    
        hl.addWidget(title, 1)
        hl.addWidget(self.btn_guess, 0)
        hl.addWidget(self.btn_check, 0)
        v.addWidget(head, 0)
    
        # Form layout (single-column = no "Time c" truncation)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        v.addLayout(form)
    
        self.ed_active = QLineEdit(box)
        self.ed_active.setReadOnly(True)
        self.ed_active.setMinimumHeight(28)
        self.ed_active.setObjectName("mapActiveEdit")
        form.addRow("Active", self.ed_active)
    
        self.cmb_tcol = QComboBox(box)
        self.cmb_scol = QComboBox(box)
        self.cmb_vcol = QComboBox(box)
        self.cmb_tval = QComboBox(box)
    
        for cmb in (self.cmb_tcol, self.cmb_scol, self.cmb_vcol, self.cmb_tval):
            cmb.setMinimumHeight(28)
    
        self.cmb_tcol.setToolTip("Time column")
        self.cmb_scol.setToolTip("Step column (optional)")
        self.cmb_vcol.setToolTip("Value column")
        self.cmb_tval.setToolTip("Time filter")
    
        form.addRow("Time column", self.cmb_tcol)
        form.addRow("Step column", self.cmb_scol)
        form.addRow("Value", self.cmb_vcol)
        form.addRow("Time filter", self.cmb_tval)
    
        self.lb_status = QLabel("", box)
        self.lb_status.setObjectName("mapStatusChip")
        self.lb_status.setWordWrap(True)
        v.addWidget(self.lb_status, 0)
    
        # Signals stay the same
        self.btn_guess.clicked.connect(self._on_guess)
        self.btn_check.clicked.connect(self._on_validate)
    
        self.cmb_tcol.currentIndexChanged.connect(self._on_tcol_changed)
        self.cmb_scol.currentIndexChanged.connect(self._on_scol_changed)
        self.cmb_vcol.currentIndexChanged.connect(self._on_vcol_changed)
        self.cmb_tval.currentIndexChanged.connect(self._on_tval_changed)
    
        self._details_set_empty()
        return box


    def _build_details_panel(self, parent: QWidget) -> QWidget:
        box = QGroupBox("Forecast details", parent)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
    
        self.lb_meta = QLabel(
            "Select a forecast CSV to see details.",
            box,
        )
        self.lb_meta.setWordWrap(True)
    
        self.cmb_time = QComboBox(box)
        self.cmb_value = QComboBox(box)
    
        lay.addWidget(self.lb_meta, 0)
        lay.addWidget(QLabel("Time:", box), 0)
        lay.addWidget(self.cmb_time, 0)
        lay.addWidget(QLabel("Value:", box), 0)
        lay.addWidget(self.cmb_value, 0)
        lay.addStretch(1)
    
        self.cmb_time.currentIndexChanged.connect(
            self._on_details_changed,
        )
        self.cmb_value.currentIndexChanged.connect(
            self._on_details_changed,
        )
    
        return box

    # -------------------------------------------------
    # Store sync
    # -------------------------------------------------
    def _on_filter_changed(self, text: str) -> None:
        text = (text or "").strip().lower()
    
        src = str(self.store.get("map.data_source", "auto")).strip().lower()
        tree = self.tree_manual if src == "manual" else self.tree
    
        def _match_item(it: QTreeWidgetItem) -> bool:
            hay = " ".join([
                str(it.text(0) or ""),
                str(it.text(1) or ""),
            ]).lower()
            return (text in hay) if text else True
    
        def _recurse(it: QTreeWidgetItem) -> bool:
            any_child = False
            for k in range(it.childCount()):
                any_child = _recurse(it.child(k)) or any_child
    
            show = _match_item(it) or any_child
            it.setHidden(not show)
            return show
    
        tree.blockSignals(True)
        try:
            for i in range(tree.topLevelItemCount()):
                _recurse(tree.topLevelItem(i))
        finally:
            tree.blockSignals(False)


    def _sync_from_store(self) -> None:
        src = str(self.store.get("map.data_source", "auto"))
        src = src.strip().lower()

        if src == "manual":
            self.cmb_source.setCurrentIndex(1)
            self.stack.setCurrentWidget(self.page_manual)
        else:
            self.cmb_source.setCurrentIndex(0)
            self.stack.setCurrentWidget(self.page_auto)

        sel = self.store.get("map.selected_files", [])
        paths = unique_str(sel or [])
        self._selected = set(paths)

        self._load_manual_list()
        self._sync_sampling_from_store()


    def _emit_selection(self) -> None:
        paths = sorted(self._selected)
        self.store.set("map.selected_files", paths)
        self.selection_changed.emit(paths)
     
    def _on_guess(self) -> None:
        meta = self._active_meta
        if meta is None:
            return

        t = str(meta.time_col or "")
        s = str(meta.step_col or "")
        v = self._guess_value_col(meta)

        self._set_combo_data(self.cmb_tcol, t)
        self._set_combo_data(self.cmb_scol, s)
        self._set_combo_data(self.cmb_vcol, v)
        self._set_combo_data(self.cmb_tval, "")

        with self.store.batch():
            self.store.set("map.time_col", t)
            self.store.set("map.step_col", s)
            self.store.set("map.value_col", v)
            self.store.set("map.time_value", "")

        self._update_details_status(meta)

    def _on_validate(self) -> None:
        ok, msg = self._validate_details()
        if ok:
            self.lb_status.setText(f"✓ {msg}")
        else:
            self.lb_status.setText(f"⚠ {msg}")


    def _on_active_item_changed(self) -> None:
        src = str(self.store.get("map.data_source", "auto"))
        src = src.strip().lower()
    
        if src == "manual":
            it = self.tree_manual.currentItem()
        else:
            it = self.tree.currentItem()
    
        if it is None:
            return
    
        p = it.data(0, ROLE_PATH)
        if not p:
            return
    
        self._set_active_file(str(p))
    
    def _set_active_file(self, path: Path) -> None:
        p = Path(path).expanduser()
        if not p.exists():
            return

        self._active_path = p
        self.store.set("map.active_file", str(p))
        self.ed_active.setText(p.name)
        self.ed_active.setToolTip(str(p))

        cm = self._colmap_from_store()
        try:
            meta = load_forecast_meta(p, colmap=cm)
        except Exception as exc:
            self._active_meta = None
            self._details_set_error(str(exc))
            return

        self._active_meta = meta
        self._details_apply_meta(meta)
        self.active_changed.emit(str(p))
        self.columns_changed.emit(list(meta.cols))
        
        # self.split.setSizes([650, 250])
        # if want smaller 
        # self.split.setSizes([700, 220])


    def _colmap_from_store(self) -> dict:
        return {
            "time": str(self.store.get("map.time_col", "")),
            "step": str(self.store.get("map.step_col", "")),
        }
    
    # def _set_active_file(self, path: str) -> None:
    #     self.store.set("map.active_file", str(path))
    
    #     cm = self._colmap_from_store()
    #     meta = load_forecast_meta(
    #         Path(path),
    #         colmap=cm,
    #     )

    #     self._fill_details(meta)
    
    #     # Expand bottom details pane.
    #     self.split.setSizes([650, 250])
        
    # def _colmap_from_store(self) -> dict:
    #     return {
    #         "x": str(self.store.get("map.x_col", "")),
    #         "y": str(self.store.get("map.y_col", "")),
    #         "time": str(self.store.get("map.time_col", "")),
    #         "step": str(self.store.get("map.step_col", "")),
    #     }
    def _details_set_empty(self) -> None:
        self.ed_active.setText("")
        self.lb_status.setText("No dataset selected.")
        self.cmb_tcol.clear()
        self.cmb_scol.clear()
        self.cmb_vcol.clear()
        self.cmb_tval.clear()

        self.cmb_tcol.addItem("(none)", "")
        self.cmb_scol.addItem("(none)", "")
        self.cmb_vcol.addItem("(none)", "")
        self.cmb_tval.addItem("All", "")

    def _details_set_error(self, msg: str) -> None:
        self.lb_status.setText(f"⚠ {msg}")

    def _details_apply_meta(self, meta: ForecastMeta) -> None:
        cols = list(meta.cols)

        self.cmb_tcol.blockSignals(True)
        self.cmb_scol.blockSignals(True)
        self.cmb_vcol.blockSignals(True)
        self.cmb_tval.blockSignals(True)

        try:
            self.cmb_tcol.clear()
            self.cmb_scol.clear()
            self.cmb_vcol.clear()
            self.cmb_tval.clear()

            self.cmb_tcol.addItem("(none)", "")
            for c in cols:
                self.cmb_tcol.addItem(c, c)

            self.cmb_scol.addItem("(none)", "")
            for c in cols:
                self.cmb_scol.addItem(c, c)

            self._fill_value_items(meta)
            self._fill_time_values(meta)

            self._apply_defaults_from_store(meta)
            self._update_details_status(meta)
        finally:
            self.cmb_tcol.blockSignals(False)
            self.cmb_scol.blockSignals(False)
            self.cmb_vcol.blockSignals(False)
            self.cmb_tval.blockSignals(False)

    def _fill_value_items(self, meta: ForecastMeta) -> None:
        self.cmb_vcol.addItem("(none)", "")
        for label, col in meta.value_items:
            self.cmb_vcol.addItem(str(label), str(col))

    def _fill_time_values(self, meta: ForecastMeta) -> None:
        self.cmb_tval.addItem("All", "")
        for v in meta.time_values:
            self.cmb_tval.addItem(str(v), int(v))
            
    def _validate_details(self) -> tuple[bool, str]:
        if self._active_path is None:
            return False, "No active dataset."

        meta = self._active_meta
        if meta is None:
            return False, "No metadata."

        tcol = str(self.cmb_tcol.currentData() or "")
        scol = str(self.cmb_scol.currentData() or "")
        vcol = str(self.cmb_vcol.currentData() or "")
        tval = self.cmb_tval.currentData()

        cols = set(meta.cols)

        if tcol and tcol not in cols:
            return False, "Time col not found."
        if scol and scol not in cols:
            return False, "Step col not found."
        if vcol and vcol not in cols:
            return False, "Value col not found."
        if not vcol:
            return False, "Pick a value column."

        try:
            ok = _quick_csv_check(
                path=self._active_path,
                tcol=tcol,
                scol=scol,
                vcol=vcol,
            )
        except Exception as exc:
            return False, str(exc)

        if not ok:
            return False, "Columns not numeric / empty."

        if tval not in ("", None):
            try:
                _ = int(tval)
            except Exception:
                return False, "Time value invalid."

        return True, "Mapping looks good."
    
    def _fill_details(self, meta) -> None:
        lines = [
            f"Rows: {meta.n_rows}",
            f"Points: {meta.n_points}",
            f"Years: {meta.year_min} → {meta.year_max}",
            f"Steps: {meta.step_min} → {meta.step_max}",
            f"Quantiles: {meta.quantile_label}",
            f"Unit: {meta.unit or '-'}",
        ]
        self.lb_meta.setText("\n".join(lines))
    
        self.cmb_time.blockSignals(True)
        self.cmb_value.blockSignals(True)
        try:
            self.cmb_time.clear()
            for v in meta.time_values:
                self.cmb_time.addItem(str(v))
    
            self.cmb_value.clear()
            for label, col in meta.value_items:
                self.cmb_value.addItem(label, col)
        finally:
            self.cmb_time.blockSignals(False)
            self.cmb_value.blockSignals(False)
    
        self._on_details_changed()
    
    
    def _on_details_changed(self) -> None:
        if self.cmb_time.count() > 0:
            self.store.set(
                "map.time_value",
                self.cmb_time.currentText(),
            )
    
        if self.cmb_value.count() > 0:
            col = self.cmb_value.currentData()
            self.store.set("map.value_col", str(col))
            
    def _apply_defaults_from_store(
        self,
        meta: ForecastMeta,
    ) -> None:
        t0 = str(self.store.get("map.time_col", "")).strip()
        s0 = str(self.store.get("map.step_col", "")).strip()
        v0 = str(self.store.get("map.value_col", "")).strip()
        tv = str(self.store.get("map.time_value", "")).strip()

        if not t0:
            t0 = str(meta.time_col or "")
        if not s0:
            s0 = str(meta.step_col or "")

        if not v0:
            v0 = self._guess_value_col(meta)

        self._set_combo_data(self.cmb_tcol, t0)
        self._set_combo_data(self.cmb_scol, s0)
        self._set_combo_data(self.cmb_vcol, v0)

        if tv:
            try:
                self._set_combo_data(self.cmb_tval, int(tv))
            except Exception:
                self._set_combo_data(self.cmb_tval, "")

    def _guess_value_col(self, meta: ForecastMeta) -> str:
        for _, c in meta.value_items:
            if c.lower().endswith("_q50"):
                return str(c)
        for _, c in meta.value_items:
            if c.lower().endswith("_pred"):
                return str(c)
        if meta.value_items:
            return str(meta.value_items[0][1])
        return ""

    def _set_combo_data(
        self,
        cmb: QComboBox,
        val,
    ) -> None:
        for i in range(cmb.count()):
            if cmb.itemData(i) == val:
                cmb.setCurrentIndex(i)
                return

    # -------------------------------------------------
    # Auto mode
    # -------------------------------------------------
    def _refresh_auto(self) -> None:
        self.tree.blockSignals(True)
        try:
            self.tree.clear()

            root = self._results_root()
            if not root:
                self._add_info_item(
                    self.tree,
                    "No results_root configured.",
                    "",
                )
                return

            self._cities = scan_results_root(root)
            if not self._cities:
                self._add_info_item(
                    self.tree,
                    "No forecast CSV found.",
                    str(root),
                )
                return

            for city in self._cities:
                self._add_city_item(city)
        finally:
            self.tree.blockSignals(False)

    def _add_city_item(self, city: MapCity) -> None:
        it = QTreeWidgetItem(self.tree)
        it.setText(0, city.city)
        it.setText(1, f"{city.model} · {city.stage}")
        it.setFlags(it.flags() | Qt.ItemIsEnabled)

        for job in city.jobs:
            self._add_job_item(it, job)

        it.setExpanded(True)

    def _add_job_item(self, parent: QTreeWidgetItem, job: MapJob) -> None:
        it = QTreeWidgetItem(parent)
        it.setText(0, f"{job.kind} · {job.job_id}")
        it.setText(1, job.root.name)
        it.setFlags(it.flags() | Qt.ItemIsEnabled)
        it.setExpanded(True)

        for f in job.files:
            self._add_file_item(it, f)

    def _add_file_item(self, parent: QTreeWidgetItem, f: MapFile) -> None:
        it = QTreeWidgetItem(parent)
        it.setText(0, f.display)
        it.setText(1, f.path.name)

        it.setData(0, ROLE_PATH, str(f.path))
        it.setData(0, ROLE_KIND, str(f.kind))

        it.setFlags(
            it.flags()
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsSelectable
            | Qt.ItemIsEnabled
        )

        p = str(f.path)
        chk = Qt.Checked if p in self._selected else Qt.Unchecked
        it.setCheckState(0, chk)

    def _on_item_changed(self, item: QTreeWidgetItem, col: int) -> None:
        p = item.data(0, ROLE_PATH)
        if not p:
            return

        if item.checkState(0) == Qt.Checked:
            self._selected.add(str(p))
        else:
            self._selected.discard(str(p))

        self._emit_selection()

    def _on_item_double_clicked(
        self,
        item: QTreeWidgetItem,
        col: int,
    ) -> None:
        p = item.data(0, ROLE_PATH)
        if not p:
            return

        if item.checkState(0) == Qt.Checked:
            item.setCheckState(0, Qt.Unchecked)
        else:
            item.setCheckState(0, Qt.Checked)


    # -------------------------------------------------
    # Manual mode
    # -------------------------------------------------
    def _datasets_root(self) -> Optional[Path]:
        root = self._results_root()
        if not root:
            return None
        return root / "_datasets"

    def _load_manual_list(self) -> None:
        self.tree_manual.blockSignals(True)
        try:
            self.tree_manual.clear()

            files = self.store.get("map.manual_files", [])
            paths = [Path(p) for p in unique_str(files or [])]

            if not paths:
                self._add_info_item(
                    self.tree_manual,
                    "No manual datasets yet.",
                    "Use 'Add CSV'.",
                )
                return

            for p in paths:
                it = QTreeWidgetItem(self.tree_manual)
                it.setText(0, p.name)
                orig = self._manual_src.get(str(p), str(p))
                it.setText(1, str(orig))


                it.setData(0, ROLE_PATH, str(p))
                it.setFlags(
                    it.flags()
                    | Qt.ItemIsUserCheckable
                    | Qt.ItemIsSelectable
                    | Qt.ItemIsEnabled
                )

                chk = (
                    Qt.Checked
                    if str(p) in self._selected
                    else Qt.Unchecked
                )
                it.setCheckState(0, chk)
        finally:
            self.tree_manual.blockSignals(False)

    def _on_add_csv(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CSV files",
            str(Path.home()),
            "CSV files (*.csv);;All files (*)",
        )
        if not paths:
            return
    
        kept = self._import_manual_paths(paths)
        if not kept:
            return
    
        old = self.store.get("map.manual_files", [])
        merged = unique_str(list(old or []) + kept)
    
        with self.store.batch():
            self.store.set("map.manual_files", merged)
            self.store.set("map.data_source", "manual")
    
        self._sync_from_store()

    def _on_add_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self,
            "Select folder",
            str(Path.home()),
        )
        if not d:
            return
    
        root = Path(d)
        csvs = sorted(root.rglob("*.csv"))
    
        if not csvs:
            return
    
        names = [p.name.lower() for p in csvs]
        has_forecast = any("forecast" in n for n in names)
    
        if has_forecast:
            csvs = [p for p in csvs
                    if "forecast" in p.name.lower()]
    
        kept = self._import_manual_paths(
            [str(p) for p in csvs],
        )
        if not kept:
            return
    
        old = self.store.get("map.manual_files", [])
        merged = unique_str(list(old or []) + kept)
    
        with self.store.batch():
            self.store.set("map.manual_files", merged)
            self.store.set("map.data_source", "manual")
    
        self._sync_from_store()
        
    def _import_manual_paths(self, paths) -> List[str]:
        kept: List[str] = []
        tmp = self._tmp_root()
        tmp.mkdir(parents=True, exist_ok=True)
    
        for p in paths:
            src = Path(p).expanduser()
            if not src.exists():
                continue
    
            dst = tmp / src.name
            if dst.exists():
                dst = tmp / f"{src.stem}__1{src.suffix}"
    
            try:
                shutil.copy2(str(src), str(dst))
            except Exception:
                continue
    
            self._manual_src[str(dst)] = str(src)
            kept.append(str(dst))
    
        return kept

    def _on_remove_manual(self) -> None:
        items = self.tree_manual.selectedItems()
        if not items:
            return

        remove = set()
        for it in items:
            p = it.data(0, ROLE_PATH)
            if p:
                remove.add(str(p))

        old = unique_str(self.store.get("map.manual_files", []) or [])
        kept = [p for p in old if p not in remove]

        self.store.set("map.manual_files", kept)

        for p in remove:
            self._selected.discard(p)
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
            self._manual_src.pop(p, None)

        self._load_manual_list()
        self._emit_selection()

    def _on_manual_item_changed(
        self,
        item: QTreeWidgetItem,
        col: int,
    ) -> None:
        p = item.data(0, ROLE_PATH)
        if not p:
            return

        if item.checkState(0) == Qt.Checked:
            self._selected.add(str(p))
        else:
            self._selected.discard(str(p))

        self._emit_selection()

    # -------------------------------------------------
    # Top controls
    # -------------------------------------------------
    def _on_source_changed(self, idx: int) -> None:
        src = "manual" if idx == 1 else "auto"
        self.store.set("map.data_source", src)

        if src == "auto":
            self.stack.setCurrentWidget(self.page_auto)
            self._refresh_auto()
        else:
            self.stack.setCurrentWidget(self.page_manual)
            self._load_manual_list()

    def _on_refresh(self) -> None:
        src = str(self.store.get("map.data_source", "auto"))
        if src.strip().lower() == "manual":
            self._load_manual_list()
        else:
            self._refresh_auto()
            
    def _on_tcol_changed(self, _idx: int) -> None:
        v = str(self.cmb_tcol.currentData() or "")
        self.store.set("map.time_col", v)

    def _on_scol_changed(self, _idx: int) -> None:
        v = str(self.cmb_scol.currentData() or "")
        self.store.set("map.step_col", v)

    def _on_vcol_changed(self, _idx: int) -> None:
        v = str(self.cmb_vcol.currentData() or "")
        self.store.set("map.value_col", v)

    def _on_tval_changed(self, _idx: int) -> None:
        v = self.cmb_tval.currentData()
        if v in ("", None):
            self.store.set("map.time_value", "")
            return
        try:
            self.store.set("map.time_value", str(int(v)))
        except Exception:
            self.store.set("map.time_value", "")

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def _results_root(self) -> Optional[Path]:
        rr = getattr(self.store.cfg, "results_root", None)
        if rr is None:
            return None
        try:
            p = Path(rr).expanduser()
        except Exception:
            return None
        if p.exists() and p.is_dir():
            return p
        return p

    def _add_info_item(
        self,
        tree: QTreeWidget,
        title: str,
        info: str,
    ) -> None:
        it = QTreeWidgetItem(tree)
        it.setText(0, str(title))
        it.setText(1, str(info))
        it.setFlags(it.flags() | Qt.ItemIsEnabled)
        
    def _update_details_status(self, meta: ForecastMeta) -> None:
        u = meta.unit or "-"
        yrs = meta.quantile_label or "-"
        msg = (
            f"rows={meta.n_rows}, "
            f"points={meta.n_points}, "
            f"unit={u}, "
            f"vars={yrs}"
        )
        self.lb_status.setText(msg)

def _quick_csv_check(
    *,
    path: Path,
    tcol: str,
    scol: str,
    vcol: str,
) -> bool:
    import pandas as pd

    use = [c for c in [tcol, scol, vcol] if c]
    if not use:
        return False

    df = pd.read_csv(
        Path(path),
        usecols=use,
        nrows=300,
    )
    if not len(df):
        return False

    for c in [tcol, scol, vcol]:
        if not c:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 3:
            return False

    return True

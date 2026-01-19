# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer_tab

Transferability tab (cross-city transfer matrix).

Now:
- Top cards stay here (cities/splits + results/view).
- Mini-tabs row: [Map] [Options] + Run button.
- Stacked pages:
  - XferMapPage
  - XferOptionsPanel
- Advanced options live in XferOptionsPanel (store-backed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..config.store import GeoConfigStore
from .xfer.keys import K_VIEW_MODE, K_MAP_EXPANDED
from .xfer.map.page import XferMapPage
from .xfer.options.panel import XferOptionsPanel

def _blocked(w: QWidget) -> QSignalBlocker:
    return QSignalBlocker(w)


class XferTab(QWidget):
    """Transferability tab widget (thin host)."""

    run_clicked = pyqtSignal()
    view_clicked = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        make_card: Callable[[str], tuple[QWidget, QVBoxLayout]],
        make_run_button: Callable[[str], QToolButton],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store
        self._map_expanded = False
        
        self._make_card = make_card
        self._make_run_button = make_run_button

        self._build_ui()
        self._connect_signals()
        self._sync_from_store()
        self._apply_view_mode_from_store()
        self._apply_map_expanded_from_store()
        self._update_map_from_store()

        self._s.config_changed.connect(self._on_store_changed)

    # -------------------------------------------------
    # Public helpers
    # -------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """
        Collect a controller-friendly snapshot.

        Base fields come from this tab.
        Advanced fields come from XferOptionsPanel.
        """
        st: Dict[str, Any] = {
            "city_a": self.ed_city_a.text().strip(),
            "city_b": self.ed_city_b.text().strip(),
            "results_root": (
                self.ed_results_root.text().strip() or None
            ),
            "splits": self._get_checked_splits(),
            "calib_modes": self._get_checked_calib(),
            "batch_size": int(self.sp_batch.value()),
            "rescale_to_source": bool(
                self.chk_rescale.isChecked()
            ),
            "view_kind": str(
                self.cmb_view_kind.currentData() or "calib_panel"
            ),
            "view_split": str(
                self.cmb_view_split.currentData() or "val"
            ),
        }

        # Merge advanced/store-backed options.
        try:
            st.update(self.options_panel.get_state())
        except Exception:
            pass

        return st

    def set_last_output(self, out_dir: Optional[str]) -> None:
        txt = out_dir or "No transfer run yet."
        self.lbl_last_out.setText(txt)
        self.btn_make_view.setVisible(bool(out_dir))

    def set_has_result(self, has: bool) -> None:
        self.btn_make_view.setVisible(bool(has))
        self.btn_make_view.setEnabled(bool(has))

    def set_view_enabled(self, enabled: bool) -> None:
        self.btn_make_view.setEnabled(bool(enabled))

    def set_run_enabled(self, enabled: bool) -> None:
        self.btn_run_xfer.setEnabled(bool(enabled))

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # --------------------------
        # Top cards row
        # --------------------------
        # row = QHBoxLayout()
        # row.setSpacing(10)
        # row.setAlignment(Qt.AlignTop)

        # c1, b1 = self._make_card("Cities & splits")
        # self._build_cities_box(b1)
        # row.addWidget(c1, 1, Qt.AlignTop)

        # c2, b2 = self._make_card("Results & view")
        # self._build_results_box(b2)
        # row.addWidget(c2, 1, Qt.AlignTop)

        # root.addLayout(row)
        
        self._cards_row = QWidget(self)
        row = QHBoxLayout(self._cards_row)
        row.setSpacing(10)
        row.setAlignment(Qt.AlignTop)

        c1, b1 = self._make_card("Cities & splits")
        self._build_cities_box(b1)
        row.addWidget(c1, 1, Qt.AlignTop)

        c2, b2 = self._make_card("Results & view")
        self._build_results_box(b2)
        row.addWidget(c2, 1, Qt.AlignTop)

        root.addWidget(self._cards_row)

        # Dock slot used when map is expanded
        self._map_dock = QWidget(self)
        self._map_dock.setVisible(False)

        self._map_dock_l = QVBoxLayout(self._map_dock)
        self._map_dock_l.setContentsMargins(0, 0, 0, 0)
        self._map_dock_l.setSpacing(0)

        root.addWidget(self._map_dock)

        # --------------------------
        # Mini-tabs row + Run
        # --------------------------
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)

        self.btn_mode_map = QToolButton(self)
        self.btn_mode_map.setText("Map")
        self.btn_mode_map.setCheckable(True)

        self.btn_mode_opts = QToolButton(self)
        self.btn_mode_opts.setText("Options")
        self.btn_mode_opts.setCheckable(True)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self.btn_mode_map, 0)
        self._mode_group.addButton(self.btn_mode_opts, 1)

        mode_row.addWidget(self.btn_mode_map)
        mode_row.addWidget(self.btn_mode_opts)
        mode_row.addStretch(1)

        self.btn_run_xfer = self._make_run_button(
            "Run transfer matrix"
        )
        mode_row.addWidget(self.btn_run_xfer)

        root.addLayout(mode_row)

        # --------------------------
        # Pages: Map + Options
        # --------------------------
        self._stack = QStackedWidget(self)

        self.map_page = XferMapPage(store=self._s, parent=self)
        self._stack.addWidget(self.map_page)

        self.options_panel = XferOptionsPanel(
            store=self._s,
            parent=self,
        )
        self._stack.addWidget(self.options_panel)

        root.addWidget(self._stack, 1)

    def _build_cities_box(self, box: QVBoxLayout) -> None:
        self.ed_city_a = QLineEdit(self)
        self.ed_city_a.setPlaceholderText("nansha")

        self.ed_city_b = QLineEdit(self)
        self.ed_city_b.setPlaceholderText("zhongshan")

        self.chk_split_train = QCheckBox("train", self)
        self.chk_split_val = QCheckBox("val", self)
        self.chk_split_test = QCheckBox("test", self)

        self.chk_cal_none = QCheckBox("none", self)
        self.chk_cal_source = QCheckBox("source", self)
        self.chk_cal_target = QCheckBox("target", self)

        self.sp_batch = QSpinBox(self)
        self.sp_batch.setRange(1, 2048)

        self.chk_rescale = QCheckBox(
            "Rescale target city to source domain",
            self,
        )

        g = QGridLayout()
        r = 0

        g.addWidget(QLabel("City A (source):"), r, 0)
        g.addWidget(self.ed_city_a, r, 1)
        r += 1

        g.addWidget(QLabel("City B (target):"), r, 0)
        g.addWidget(self.ed_city_b, r, 1)
        r += 1

        g.addWidget(QLabel("Splits:"), r, 0)
        srow = QHBoxLayout()
        srow.addWidget(self.chk_split_train)
        srow.addWidget(self.chk_split_val)
        srow.addWidget(self.chk_split_test)
        srow.addStretch(1)
        g.addLayout(srow, r, 1)
        r += 1

        g.addWidget(QLabel("Calibration:"), r, 0)
        crow = QHBoxLayout()
        crow.addWidget(self.chk_cal_none)
        crow.addWidget(self.chk_cal_source)
        crow.addWidget(self.chk_cal_target)
        crow.addStretch(1)
        g.addLayout(crow, r, 1)
        r += 1

        g.addWidget(QLabel("Batch size:"), r, 0)
        g.addWidget(self.sp_batch, r, 1)
        r += 1

        g.addWidget(self.chk_rescale, r, 0, 1, 2)

        box.addLayout(g)

    def _build_results_box(self, box: QVBoxLayout) -> None:
        self.ed_results_root = QLineEdit(self)
        self.btn_browse_root = QPushButton("Browse…", self)

        self.lbl_last_out = QLabel("No transfer run yet.", self)
        self.lbl_last_out.setObjectName("xferLastOutLabel")

        self.cmb_view_kind = QComboBox(self)
        self.cmb_view_kind.addItem(
            "Calibration vs error (scatter panel)",
            "calib_panel",
        )
        self.cmb_view_kind.addItem(
            "Per-horizon MAE + cov/sharp (summary)",
            "summary_panel",
        )

        self.cmb_view_split = QComboBox(self)
        self.cmb_view_split.addItem("Validation (val)", "val")
        self.cmb_view_split.addItem("Test (test)", "test")

        self.btn_make_view = QPushButton(
            "Make view figure…",
            self,
        )
        self.btn_make_view.setVisible(False)

        g = QGridLayout()
        r = 0

        g.addWidget(QLabel("Results root:"), r, 0)
        g.addWidget(self.ed_results_root, r, 1)
        g.addWidget(self.btn_browse_root, r, 2)
        r += 1

        g.addWidget(QLabel("Last output folder:"), r, 0)
        g.addWidget(self.lbl_last_out, r, 1, 1, 2)
        r += 1

        g.addWidget(QLabel("View type:"), r, 0)
        g.addWidget(self.cmb_view_kind, r, 1, 1, 2)
        r += 1

        g.addWidget(QLabel("View split:"), r, 0)
        g.addWidget(self.cmb_view_split, r, 1, 1, 2)
        r += 1

        g.addWidget(self.btn_make_view, r, 0, 1, 3)

        box.addLayout(g)

    # -------------------------------------------------
    # Wiring
    # -------------------------------------------------
    def _connect_signals(self) -> None:
        self.btn_run_xfer.clicked.connect(self.run_clicked)
        self.btn_make_view.clicked.connect(self.view_clicked)

        self.btn_browse_root.clicked.connect(self._on_browse_root)

        for w in (self.ed_city_a, self.ed_city_b, self.ed_results_root):
            w.editingFinished.connect(self._push_to_store)

        for cb in (
            self.chk_split_train,
            self.chk_split_val,
            self.chk_split_test,
            self.chk_cal_none,
            self.chk_cal_source,
            self.chk_cal_target,
            self.chk_rescale,
        ):
            cb.toggled.connect(self._push_to_store)

        self.sp_batch.valueChanged.connect(self._push_to_store)
        self.cmb_view_kind.currentIndexChanged.connect(
            self._push_to_store
        )
        self.cmb_view_split.currentIndexChanged.connect(
            self._push_to_store
        )

        self._mode_group.buttonClicked.connect(self._on_mode_changed)
        self.map_page.request_open_options.connect(self._show_options)
        self.map_page.toolbar.request_expand.connect(
            self._on_map_expand
        )


    def _on_browse_root(self) -> None:
        start = self.ed_results_root.text().strip()
        start = start or str(self._s.get("results_root", ""))
        root = QFileDialog.getExistingDirectory(
            self,
            "Select results root (Stage-1/2/xfer)",
            start,
        )
        if not root:
            return

        rp = Path(root).expanduser()
        self._s.set("results_root", str(rp))
        self.ed_results_root.setText(str(rp))

    # -------------------------------------------------
    # View mode
    # -------------------------------------------------
    def _apply_view_mode_from_store(self) -> None:
        mode = self._s.get(K_VIEW_MODE, "map")
        mode = str(mode or "map").strip().lower()
        self._set_view_mode(mode)

    def _set_view_mode(self, mode: str) -> None:
        m = str(mode or "map").strip().lower()
        if m == "options":
            self.btn_mode_opts.setChecked(True)
            self._stack.setCurrentIndex(1)
            return

        self.btn_mode_map.setChecked(True)
        self._stack.setCurrentIndex(0)

    def _on_mode_changed(self) -> None:
        if self._map_expanded:
            self._set_map_expanded(False, persist=False)

        idx = self._mode_group.checkedId()
        mode = "options" if idx == 1 else "map"
        self._s.set(K_VIEW_MODE, mode)
        self._set_view_mode(mode)

        if mode == "map":
            self._apply_map_expanded_from_store()
            self._update_map_from_store()
            
        if mode != "map" and self._map_expanded:
            self._set_map_expanded(False)

    def _show_options(self) -> None:
        if self._map_expanded:
            self._set_map_expanded(False)

        self._s.set(K_VIEW_MODE, "options")
        self._set_view_mode("options")
        
    def _on_map_expand(self, on: bool) -> None:
        # Only meaningful when Map page is shown
        if self._stack.currentIndex() != 0:
            self.map_page.toolbar.set_expanded(False)
            return
        self._set_map_expanded(bool(on))

    def _set_map_expanded(
        self,
        on: bool,
        *,
        persist: bool = True,
    ) -> None:
        on = bool(on)
        if on == bool(self._map_expanded):
            # still ensure button reflects state
            self.map_page.toolbar.set_expanded(on)
            return
    
        tb = self.map_page.toolbar
        self._map_expanded = on
    
        if on:
            self.map_page.take_toolbar()
            self._map_dock_l.addWidget(tb)
    
            self._cards_row.setVisible(False)
            self._map_dock.setVisible(True)
    
            tb.set_expanded(True)
        else:
            self._map_dock_l.removeWidget(tb)
            tb.setParent(None)
            self.map_page.restore_toolbar()
    
            self._map_dock.setVisible(False)
            self._cards_row.setVisible(True)
    
            tb.set_expanded(False)
    
        if persist:
            # store "preference", not "current visible layout"
            cur = bool(self._s.get(K_MAP_EXPANDED, False))
            if cur != on:
                self._s.set(K_MAP_EXPANDED, on)

    def _apply_map_expanded_from_store(self) -> None:
        want = bool(self._s.get(K_MAP_EXPANDED, False))
    
        # If we're on map page: apply preference.
        if self._stack.currentIndex() == 0:
            self._set_map_expanded(want, persist=False)
            return
    
        # Not on map page: keep layout collapsed,
        # but keep preference intact.
        self._set_map_expanded(False, persist=False)

    # -------------------------------------------------
    # Store I/O
    # -------------------------------------------------
    def _on_store_changed(self, keys: object) -> None:
        try:
            changed = set(keys or [])
        except Exception:
            changed = set()
        if not changed:
            return

        base = {
            "results_root",
            "xfer.city_a",
            "xfer.city_b",
            "xfer.splits",
            "xfer.calib_modes",
            "xfer.batch_size",
            "xfer.rescale_to_source",
            "xfer.view_kind",
            "xfer.view_split",
            K_VIEW_MODE,
        }
        if changed & base:
            self._sync_from_store(keys=changed)

        if changed & {K_MAP_EXPANDED, K_VIEW_MODE}:
            self._apply_map_expanded_from_store()

        if changed & {
            "xfer.city_a",
            "xfer.city_b",
            "xfer.city_a_lat",
            "xfer.city_a_lon",
            "xfer.city_b_lat",
            "xfer.city_b_lon",
        }:
            self._update_map_from_store()

    def _sync_from_store(
        self,
        keys: Optional[set[str]] = None,
    ) -> None:
        s = self._s
        keys = keys or set()

        def wants(k: str) -> bool:
            return (not keys) or (k in keys)

        if wants("results_root"):
            v = s.get("results_root", "")
            if v:
                with _blocked(self.ed_results_root):
                    self.ed_results_root.setText(str(v))

        if wants("xfer.city_a"):
            v = s.get("xfer.city_a", "")
            if v:
                with _blocked(self.ed_city_a):
                    self.ed_city_a.setText(str(v))

        if wants("xfer.city_b"):
            v = s.get("xfer.city_b", "")
            if v:
                with _blocked(self.ed_city_b):
                    self.ed_city_b.setText(str(v))

        if wants("xfer.batch_size"):
            v = s.get("xfer.batch_size", None)
            if v is not None:
                with _blocked(self.sp_batch):
                    self.sp_batch.setValue(int(v))

        if wants("xfer.rescale_to_source"):
            v = bool(s.get("xfer.rescale_to_source", False))
            with _blocked(self.chk_rescale):
                self.chk_rescale.setChecked(v)

        if wants("xfer.splits"):
            self._sync_splits(
                s.get("xfer.splits", ("val", "test"))
            )

        if wants("xfer.calib_modes"):
            self._sync_calib_modes(
                s.get(
                    "xfer.calib_modes",
                    ("none", "source", "target"),
                )
            )

        if wants("xfer.view_kind"):
            v = s.get("xfer.view_kind", "calib_panel")
            self._set_combo_data(self.cmb_view_kind, v)

        if wants("xfer.view_split"):
            v = s.get("xfer.view_split", "val")
            self._set_combo_data(self.cmb_view_split, v)

        if wants(K_VIEW_MODE):
            self._apply_view_mode_from_store()

    def _push_to_store(self) -> None:
        s = self._s

        s.set("results_root", self.ed_results_root.text().strip())
        s.set("xfer.city_a", self.ed_city_a.text().strip())
        s.set("xfer.city_b", self.ed_city_b.text().strip())

        s.set("xfer.splits", self._get_checked_splits())
        s.set("xfer.calib_modes", self._get_checked_calib())

        s.set("xfer.batch_size", int(self.sp_batch.value()))
        s.set(
            "xfer.rescale_to_source",
            bool(self.chk_rescale.isChecked()),
        )

        s.set("xfer.view_kind", str(self.cmb_view_kind.currentData()))
        s.set("xfer.view_split", str(self.cmb_view_split.currentData()))

        mode = "options" if self._stack.currentIndex() == 1 else "map"
        s.set(K_VIEW_MODE, mode)

    # -------------------------------------------------
    # Small helpers
    # -------------------------------------------------
    def _get_checked_splits(self) -> List[str]:
        out: List[str] = []
        if self.chk_split_train.isChecked():
            out.append("train")
        if self.chk_split_val.isChecked():
            out.append("val")
        if self.chk_split_test.isChecked():
            out.append("test")
        return out

    def _sync_splits(self, splits: Sequence[str]) -> None:
        ss = set(splits or [])
        with _blocked(self.chk_split_train):
            self.chk_split_train.setChecked("train" in ss)
        with _blocked(self.chk_split_val):
            self.chk_split_val.setChecked("val" in ss)
        with _blocked(self.chk_split_test):
            self.chk_split_test.setChecked("test" in ss)

    def _get_checked_calib(self) -> List[str]:
        out: List[str] = []
        if self.chk_cal_none.isChecked():
            out.append("none")
        if self.chk_cal_source.isChecked():
            out.append("source")
        if self.chk_cal_target.isChecked():
            out.append("target")
        return out

    def _sync_calib_modes(self, modes: Sequence[str]) -> None:
        mm = set(modes or [])
        with _blocked(self.chk_cal_none):
            self.chk_cal_none.setChecked("none" in mm)
        with _blocked(self.chk_cal_source):
            self.chk_cal_source.setChecked("source" in mm)
        with _blocked(self.chk_cal_target):
            self.chk_cal_target.setChecked("target" in mm)

    def _set_combo_data(self, cmb: QComboBox, data: Any) -> None:
        for i in range(cmb.count()):
            if cmb.itemData(i) == data:
                with _blocked(cmb):
                    cmb.setCurrentIndex(i)
                return

    def _as_float(self, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        try:
            return float(v)
        except Exception:
            return None

    def _coords_complete_from_store(self) -> bool:
        s = self._s
        lat_a = self._as_float(s.get("xfer.city_a_lat", None))
        lon_a = self._as_float(s.get("xfer.city_a_lon", None))
        lat_b = self._as_float(s.get("xfer.city_b_lat", None))
        lon_b = self._as_float(s.get("xfer.city_b_lon", None))

        if None in (lat_a, lon_a, lat_b, lon_b):
            return False

        if not (-90.0 <= lat_a <= 90.0 and -180.0 <= lon_a <= 180.0):
            return False
        if not (-90.0 <= lat_b <= 90.0 and -180.0 <= lon_b <= 180.0):
            return False

        return True

    def _update_map_from_store(self) -> None:
        s = self._s
        a = str(s.get("xfer.city_a", "") or "").strip()
        b = str(s.get("xfer.city_b", "") or "").strip()
    
        lat_a = self._as_float(s.get("xfer.city_a_lat", None))
        lon_a = self._as_float(s.get("xfer.city_a_lon", None))
        lat_b = self._as_float(s.get("xfer.city_b_lat", None))
        lon_b = self._as_float(s.get("xfer.city_b_lon", None))
    
        # Only centroids depend on these coords.
        # Do NOT clear layers when coords are missing.
        if None in (lat_a, lon_a, lat_b, lon_b):
            self.map_page.clear_centroids()
            return
    
        try:
            self.map_page.set_centroids(
                a or "City A",
                float(lat_a),
                float(lon_a),
                b or "City B",
                float(lat_b),
                float(lon_b),
            )
        except Exception:
            self.map_page.clear_centroids()


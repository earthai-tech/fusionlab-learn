# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.xfer_tab

Transferability tab (cross-city transfer matrix).

v3.2 goals
----------
- UI is centralized in this module (app.py stays clean).
- Store is the single source of truth (GeoConfigStore).
- Advanced knobs are store-backed (xfer.* keys).
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
)

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QButtonGroup,
    QStackedWidget,
)

from ..config.store import GeoConfigStore
from .map_view import MapView, CityPoint

def _blocked(w: QWidget):
    return QSignalBlocker(w)


def _parse_float_list(text: str) -> Optional[List[float]]:
    raw = (text or "").strip()
    if not raw:
        return None

    bits = [b.strip() for b in raw.split(",")]
    out: List[float] = []
    for b in bits:
        if not b:
            continue
        try:
            out.append(float(b))
        except Exception:
            return None

    return out or None


class XferTab(QWidget):
    """Transferability tab widget."""

    run_clicked = pyqtSignal()
    view_clicked = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        make_card: Callable[
            [str],
            tuple[QWidget, QVBoxLayout],
        ],
        make_run_button: Callable[[str], QToolButton],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._make_card = make_card
        self._make_run_button = make_run_button

        self._build_ui()
        self._connect_signals()
        self._sync_from_store()
        self._apply_view_mode_from_store()
        self._update_map_from_store()


        store.config_changed.connect(self._on_store_changed)

    # -------------------------------------------------
    # Public helpers
    # -------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Collect a controller-friendly snapshot."""
        return {
            "city_a": self.xfer_city_a.text().strip(),
            "city_b": self.xfer_city_b.text().strip(),
            "results_root": (
                self.xfer_results_root.text().strip()
                or None
            ),
            "splits": self._get_checked_splits(),
            "calib_modes": self._get_checked_calib(),
            "batch_size": int(self.sp_xfer_batch.value()),
            "rescale_to_source": bool(
                self.chk_xfer_rescale.isChecked()
            ),
            "quantiles_override": self.get_quantiles(),
            "write_json": bool(
                self.chk_xfer_json.isChecked()
            ),
            "write_csv": bool(
                self.chk_xfer_csv.isChecked()
            ),
            "prefer_tuned": bool(
                self.chk_xfer_prefer_tuned.isChecked()
            ),
            "align_policy": str(
                self.cmb_xfer_align.currentData()
                or "align_by_name_pad"
            ),
            "interval_target": float(
                self.sp_xfer_interval.value()
            ),
            "load_endpoint": str(
                self.cmb_xfer_endpoint.currentData()
                or "serve"
            ),
            "export_physics_payload": bool(
                self.chk_xfer_phys_payload.isChecked()
            ),
            "export_physical_parameters_csv": bool(
                self.chk_xfer_phys_csv.isChecked()
            ),
            "write_eval_future_csv": bool(
                self.chk_xfer_eval_future.isChecked()
            ),
            "view_kind": str(
                self.cmb_xfer_view.currentData()
                or "calib_panel"
            ),
            "view_split": str(
                self.cmb_xfer_view_split.currentData()
                or "val"
            ),
        }

    def set_last_output(self, out_dir: Optional[str]) -> None:
        txt = out_dir or "No transfer run yet."
        self.lbl_xfer_last_out.setText(txt)
        self.btn_xfer_view.setVisible(bool(out_dir))

    def set_view_enabled(self, enabled: bool) -> None:
        self.btn_xfer_view.setEnabled(bool(enabled))

    def set_run_enabled(self, enabled: bool) -> None:
        self.btn_run_xfer.setEnabled(bool(enabled))

    def get_quantiles(self) -> Optional[Sequence[float]]:
        return _parse_float_list(
            self.ed_xfer_quantiles.text()
        )

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(10)
        row.setAlignment(Qt.AlignTop)

        cities_card, cities_box = self._make_card(
            "Cities & splits"
        )
        self._build_cities_box(cities_box)
        row.addWidget(cities_card, 1, Qt.AlignTop)

        res_card, res_box = self._make_card(
            "Results & view"
        )
        self._build_results_box(res_box)
        row.addWidget(res_card, 1, Qt.AlignTop)

        root.addLayout(row)

        # ---------------------------------------------
        # Mini tabs: [ Map ] [ Options ]
        # ---------------------------------------------
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        
        self.btn_mode_map = QToolButton()
        self.btn_mode_map.setText("Map")
        self.btn_mode_map.setCheckable(True)
        
        self.btn_mode_opts = QToolButton()
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
        
        # ---------------------------------------------
        # Pages: Map + Options
        # ---------------------------------------------
        self._stack = QStackedWidget()
        
        self.map_view = MapView(parent=self)
        self._stack.addWidget(self.map_view)
        
        self.advanced_box = self._build_advanced_box()
        adv_scroll = QScrollArea()
        adv_scroll.setWidgetResizable(True)
        adv_scroll.setFrameShape(QScrollArea.NoFrame)
        adv_scroll.setWidget(self.advanced_box)
        self._stack.addWidget(adv_scroll)
        
        root.addWidget(self._stack, 1)

    def _build_cities_box(self, box: QVBoxLayout) -> None:
        self.xfer_city_a = QLineEdit()
        self.xfer_city_a.setPlaceholderText("nansha")

        self.xfer_city_b = QLineEdit()
        self.xfer_city_b.setPlaceholderText("zhongshan")

        self.chk_xfer_split_train = QCheckBox("train")
        self.chk_xfer_split_val = QCheckBox("val")
        self.chk_xfer_split_test = QCheckBox("test")

        self.chk_xfer_cal_none = QCheckBox("none")
        self.chk_xfer_cal_source = QCheckBox("source")
        self.chk_xfer_cal_target = QCheckBox("target")

        self.sp_xfer_batch = QSpinBox()
        self.sp_xfer_batch.setRange(1, 2048)

        self.chk_xfer_rescale = QCheckBox(
            "Rescale target city to source domain"
        )

        grid = QGridLayout()
        r = 0

        grid.addWidget(QLabel("City A (source):"), r, 0)
        grid.addWidget(self.xfer_city_a, r, 1)
        r += 1

        grid.addWidget(QLabel("City B (target):"), r, 0)
        grid.addWidget(self.xfer_city_b, r, 1)
        r += 1

        grid.addWidget(QLabel("Splits:"), r, 0)
        splits = QHBoxLayout()
        splits.addWidget(self.chk_xfer_split_train)
        splits.addWidget(self.chk_xfer_split_val)
        splits.addWidget(self.chk_xfer_split_test)
        splits.addStretch(1)
        grid.addLayout(splits, r, 1)
        r += 1

        grid.addWidget(QLabel("Calibration:"), r, 0)
        cal = QHBoxLayout()
        cal.addWidget(self.chk_xfer_cal_none)
        cal.addWidget(self.chk_xfer_cal_source)
        cal.addWidget(self.chk_xfer_cal_target)
        cal.addStretch(1)
        grid.addLayout(cal, r, 1)
        r += 1

        grid.addWidget(QLabel("Batch size:"), r, 0)
        grid.addWidget(self.sp_xfer_batch, r, 1)
        r += 1

        grid.addWidget(self.chk_xfer_rescale, r, 0, 1, 2)

        box.addLayout(grid)

    def _build_results_box(self, box: QVBoxLayout) -> None:
        self.xfer_results_root = QLineEdit()
        self.xfer_results_root_btn = QPushButton("Browse…")

        self.lbl_xfer_last_out = QLabel(
            "No transfer run yet."
        )
        self.lbl_xfer_last_out.setObjectName(
            "xferLastOutLabel"
        )

        self.cmb_xfer_view = QComboBox()
        self.cmb_xfer_view.addItem(
            "Calibration vs error (scatter panel)",
            "calib_panel",
        )
        self.cmb_xfer_view.addItem(
            "Per-horizon MAE + cov/sharp (summary)",
            "summary_panel",
        )

        self.cmb_xfer_view_split = QComboBox()
        self.cmb_xfer_view_split.addItem(
            "Validation (val)",
            "val",
        )
        self.cmb_xfer_view_split.addItem(
            "Test (test)",
            "test",
        )

        self.btn_xfer_view = QPushButton(
            "Make view figure…"
        )
        self.btn_xfer_view.setVisible(False)

        grid = QGridLayout()
        r = 0

        grid.addWidget(QLabel("Results root:"), r, 0)
        grid.addWidget(self.xfer_results_root, r, 1)
        grid.addWidget(self.xfer_results_root_btn, r, 2)
        r += 1

        grid.addWidget(QLabel("Last output folder:"), r, 0)
        grid.addWidget(self.lbl_xfer_last_out, r, 1, 1, 2)
        r += 1

        grid.addWidget(QLabel("View type:"), r, 0)
        grid.addWidget(self.cmb_xfer_view, r, 1, 1, 2)
        r += 1

        grid.addWidget(QLabel("View split:"), r, 0)
        grid.addWidget(self.cmb_xfer_view_split, r, 1, 1, 2)
        r += 1

        grid.addWidget(self.btn_xfer_view, r, 0, 1, 3)

        box.addLayout(grid)

    def _build_advanced_box(self) -> QWidget:
        grp = QGroupBox("Advanced")
        lay = QGridLayout(grp)
        r = 0

        self.ed_xfer_quantiles = QLineEdit()
        self.ed_xfer_quantiles.setPlaceholderText(
            "0.1,0.5,0.9"
        )

        self.chk_xfer_json = QCheckBox("Write JSON")
        self.chk_xfer_csv = QCheckBox("Write CSV")

        row0 = QHBoxLayout()
        row0.addWidget(self.chk_xfer_json)
        row0.addWidget(self.chk_xfer_csv)
        row0.addStretch(1)

        lay.addWidget(QLabel("Quantiles override:"), r, 0)
        lay.addWidget(self.ed_xfer_quantiles, r, 1)
        r += 1
        lay.addLayout(row0, r, 1)
        r += 1

        self.chk_xfer_prefer_tuned = QCheckBox(
            "Prefer tuned calibrator if available"
        )
        lay.addWidget(self.chk_xfer_prefer_tuned, r, 0, 1, 2)
        r += 1

        self.cmb_xfer_align = QComboBox()
        self.cmb_xfer_align.addItem(
            "Align by name (pad)",
            "align_by_name_pad",
        )
        self.cmb_xfer_align.addItem(
            "Strict (same columns)",
            "strict",
        )

        lay.addWidget(QLabel("Align policy:"), r, 0)
        lay.addWidget(self.cmb_xfer_align, r, 1)
        r += 1

        self.sp_xfer_interval = QDoubleSpinBox()
        self.sp_xfer_interval.setRange(0.10, 0.99)
        self.sp_xfer_interval.setSingleStep(0.01)
        self.sp_xfer_interval.setDecimals(2)

        lay.addWidget(QLabel("Interval target:"), r, 0)
        lay.addWidget(self.sp_xfer_interval, r, 1)
        r += 1

        self.cmb_xfer_endpoint = QComboBox()
        self.cmb_xfer_endpoint.addItem("serve", "serve")
        self.cmb_xfer_endpoint.addItem("export", "export")

        lay.addWidget(QLabel("Load endpoint:"), r, 0)
        lay.addWidget(self.cmb_xfer_endpoint, r, 1)
        r += 1

        self.chk_xfer_phys_payload = QCheckBox(
            "Export physics payload"
        )
        self.chk_xfer_phys_csv = QCheckBox(
            "Export physical parameters CSV"
        )
        self.chk_xfer_eval_future = QCheckBox(
            "Write eval_future.csv"
        )

        lay.addWidget(self.chk_xfer_phys_payload, r, 0, 1, 2)
        r += 1
        lay.addWidget(self.chk_xfer_phys_csv, r, 0, 1, 2)
        r += 1
        lay.addWidget(self.chk_xfer_eval_future, r, 0, 1, 2)
        r += 1

        lay.setRowStretch(r, 1)
        return grp

    # -------------------------------------------------
    # Wiring
    # -------------------------------------------------
    def _connect_signals(self) -> None:
        self.btn_run_xfer.clicked.connect(self.run_clicked)
        self.btn_xfer_view.clicked.connect(self.view_clicked)

        self.xfer_results_root_btn.clicked.connect(
            self._on_browse_root
        )

        for w in (
            self.xfer_city_a,
            self.xfer_city_b,
            self.xfer_results_root,
            self.ed_xfer_quantiles,
        ):
            w.editingFinished.connect(self._push_to_store)

        for cb in (
            self.chk_xfer_split_train,
            self.chk_xfer_split_val,
            self.chk_xfer_split_test,
            self.chk_xfer_cal_none,
            self.chk_xfer_cal_source,
            self.chk_xfer_cal_target,
            self.chk_xfer_rescale,
            self.chk_xfer_json,
            self.chk_xfer_csv,
            self.chk_xfer_prefer_tuned,
            self.chk_xfer_phys_payload,
            self.chk_xfer_phys_csv,
            self.chk_xfer_eval_future,
        ):
            cb.toggled.connect(self._push_to_store)

        self.sp_xfer_batch.valueChanged.connect(
            self._push_to_store
        )
        self.sp_xfer_interval.valueChanged.connect(
            self._push_to_store
        )

        self.cmb_xfer_align.currentIndexChanged.connect(
            self._push_to_store
        )
        self.cmb_xfer_endpoint.currentIndexChanged.connect(
            self._push_to_store
        )
        self.cmb_xfer_view.currentIndexChanged.connect(
            self._push_to_store
        )
        self.cmb_xfer_view_split.currentIndexChanged.connect(
            self._push_to_store
        )
        self._mode_group.buttonClicked.connect(
            self._on_mode_changed
        )
        
        self.map_view.request_open_options.connect(
            self._show_options
        )
        self.map_view.request_fit.connect(self._on_fit_requested)

    def _on_fit_requested(self) -> None:
        if not self._coords_complete_from_store():
            self._show_options()
            return
        self._update_map_from_store()
        self.map_view.fit_to_cities()


    def _on_browse_root(self) -> None:
        start = self.xfer_results_root.text().strip()
        start = start or str(
            self._store.get("results_root", "")
        )
        root = QFileDialog.getExistingDirectory(
            self,
            "Select results root (Stage-1/2/xfer)",
            start,
        )
        if not root:
            return

        root_path = Path(root).expanduser()
        self._store.set("results_root", str(root_path))
        self.xfer_results_root.setText(str(root_path))

    def _apply_view_mode_from_store(self) -> None:
        mode = self._store.get("xfer.view_mode", "map")
        mode = (mode or "map").strip().lower()
        self._set_view_mode(mode)
    
    def _set_view_mode(self, mode: str) -> None:
        mode = (mode or "map").strip().lower()
        if mode == "options":
            self.btn_mode_opts.setChecked(True)
            self._stack.setCurrentIndex(1)
            return
    
        self.btn_mode_map.setChecked(True)
        self._stack.setCurrentIndex(0)
        
    def _on_mode_changed(self) -> None:
        idx = self._mode_group.checkedId()
        mode = "options" if idx == 1 else "map"
        self._store.set("xfer.view_mode", mode)
        self._set_view_mode(mode)
    
        if mode == "map":
            self._update_map_from_store()

    def _show_options(self) -> None:
        self._store.set("xfer.view_mode", "options")
        self._set_view_mode("options")

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
        self._sync_from_store(keys=changed)
        if changed & {
            "xfer.city_a",
            "xfer.city_b",
            "xfer.city_a_lat",
            "xfer.city_a_lon",
            "xfer.city_b_lat",
            "xfer.city_b_lon",
        }:
            self._update_map_from_store()

    # Your MapView.set_cities(...) signature depends on your
    # implementation. If it expects CityPoint objects, adapt
    # accordingly

    def _update_map_from_store(self) -> None:
        s = self._store
    
        a = (s.get("xfer.city_a", "") or "").strip()
        b = (s.get("xfer.city_b", "") or "").strip()
    
        lat_a = s.get("xfer.city_a_lat", None)
        lon_a = s.get("xfer.city_a_lon", None)
        lat_b = s.get("xfer.city_b_lat", None)
        lon_b = s.get("xfer.city_b_lon", None)
    
        # If coords are missing, just clear (MapView must be safe; see section 2)
        if None in (lat_a, lon_a, lat_b, lon_b):
            self.map_view.clear()
            return
    
        try:
            src = CityPoint(a or "City A", float(lat_a), float(lon_a))
            tgt = CityPoint(b or "City B", float(lat_b), float(lon_b))
            self.map_view.set_cities(src, tgt)
        except Exception:
            self.map_view.clear()

    def _sync_from_store(
        self,
        keys: Optional[set[str]] = None,
    ) -> None:
        s = self._store
        keys = keys or set()

        def wants(k: str) -> bool:
            return (not keys) or (k in keys)

        if wants("results_root"):
            v = s.get("results_root", "")
            if v:
                with _blocked(self.xfer_results_root):
                    self.xfer_results_root.setText(str(v))

        if wants("xfer.city_a"):
            v = s.get("xfer.city_a", "")
            if v:
                with _blocked(self.xfer_city_a):
                    self.xfer_city_a.setText(str(v))

        if wants("xfer.city_b"):
            v = s.get("xfer.city_b", "")
            if v:
                with _blocked(self.xfer_city_b):
                    self.xfer_city_b.setText(str(v))

        if wants("xfer.batch_size"):
            v = s.get("xfer.batch_size", None)
            if v is not None:
                with _blocked(self.sp_xfer_batch):
                    self.sp_xfer_batch.setValue(int(v))

        if wants("xfer.rescale_to_source"):
            v = bool(
                s.get("xfer.rescale_to_source", False)
            )
            with _blocked(self.chk_xfer_rescale):
                self.chk_xfer_rescale.setChecked(v)

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

        if wants("xfer.write_json"):
            v = bool(s.get("xfer.write_json", True))
            with _blocked(self.chk_xfer_json):
                self.chk_xfer_json.setChecked(v)

        if wants("xfer.write_csv"):
            v = bool(s.get("xfer.write_csv", True))
            with _blocked(self.chk_xfer_csv):
                self.chk_xfer_csv.setChecked(v)

        if wants("xfer.quantiles_override"):
            q = s.get("xfer.quantiles_override", None)
            if q:
                txt = ",".join(str(x) for x in q)
                with _blocked(self.ed_xfer_quantiles):
                    self.ed_xfer_quantiles.setText(txt)

        if wants("xfer.prefer_tuned"):
            v = bool(s.get("xfer.prefer_tuned", True))
            with _blocked(self.chk_xfer_prefer_tuned):
                self.chk_xfer_prefer_tuned.setChecked(v)

        if wants("xfer.align_policy"):
            v = s.get(
                "xfer.align_policy",
                "align_by_name_pad",
            )
            self._set_combo_data(self.cmb_xfer_align, v)

        if wants("xfer.interval_target"):
            v = float(s.get("xfer.interval_target", 0.80))
            with _blocked(self.sp_xfer_interval):
                self.sp_xfer_interval.setValue(v)

        if wants("xfer.load_endpoint"):
            v = s.get("xfer.load_endpoint", "serve")
            self._set_combo_data(self.cmb_xfer_endpoint, v)

        if wants("xfer.export_physics_payload"):
            v = bool(
                s.get("xfer.export_physics_payload", True)
            )
            with _blocked(self.chk_xfer_phys_payload):
                self.chk_xfer_phys_payload.setChecked(v)

        if wants("xfer.export_physical_parameters_csv"):
            v = bool(
                s.get(
                    "xfer.export_physical_parameters_csv",
                    True,
                )
            )
            with _blocked(self.chk_xfer_phys_csv):
                self.chk_xfer_phys_csv.setChecked(v)

        if wants("xfer.write_eval_future_csv"):
            v = bool(
                s.get("xfer.write_eval_future_csv", True)
            )
            with _blocked(self.chk_xfer_eval_future):
                self.chk_xfer_eval_future.setChecked(v)

        if wants("xfer.view_kind"):
            v = s.get("xfer.view_kind", "calib_panel")
            self._set_combo_data(self.cmb_xfer_view, v)

        if wants("xfer.view_split"):
            v = s.get("xfer.view_split", "val")
            self._set_combo_data(self.cmb_xfer_view_split, v)
            
        if wants("xfer.view_mode"):
            self._apply_view_mode_from_store()


    def _push_to_store(self) -> None:
        s = self._store

        s.set(
            "results_root",
            self.xfer_results_root.text().strip(),
        )
        s.set(
            "xfer.city_a",
            self.xfer_city_a.text().strip(),
        )
        s.set(
            "xfer.city_b",
            self.xfer_city_b.text().strip(),
        )

        s.set("xfer.splits", self._get_checked_splits())
        s.set("xfer.calib_modes", self._get_checked_calib())

        s.set(
            "xfer.batch_size",
            int(self.sp_xfer_batch.value()),
        )
        s.set(
            "xfer.rescale_to_source",
            bool(self.chk_xfer_rescale.isChecked()),
        )

        s.set(
            "xfer.write_json",
            bool(self.chk_xfer_json.isChecked()),
        )
        s.set(
            "xfer.write_csv",
            bool(self.chk_xfer_csv.isChecked()),
        )

        s.set(
            "xfer.quantiles_override",
            self.get_quantiles(),
        )

        s.set(
            "xfer.prefer_tuned",
            bool(self.chk_xfer_prefer_tuned.isChecked()),
        )

        s.set(
            "xfer.align_policy",
            str(self.cmb_xfer_align.currentData()),
        )
        s.set(
            "xfer.interval_target",
            float(self.sp_xfer_interval.value()),
        )
        s.set(
            "xfer.load_endpoint",
            str(self.cmb_xfer_endpoint.currentData()),
        )

        s.set(
            "xfer.export_physics_payload",
            bool(self.chk_xfer_phys_payload.isChecked()),
        )
        s.set(
            "xfer.export_physical_parameters_csv",
            bool(self.chk_xfer_phys_csv.isChecked()),
        )
        s.set(
            "xfer.write_eval_future_csv",
            bool(self.chk_xfer_eval_future.isChecked()),
        )

        s.set(
            "xfer.view_kind",
            str(self.cmb_xfer_view.currentData()),
        )
        s.set(
            "xfer.view_split",
            str(self.cmb_xfer_view_split.currentData()),
        )
        mode = "map"
        if self._stack.currentIndex() == 1:
            mode = "options"
        s.set("xfer.view_mode", mode)


    # -------------------------------------------------
    # Small helpers
    # -------------------------------------------------
    def _get_checked_splits(self) -> List[str]:
        out: List[str] = []
        if self.chk_xfer_split_train.isChecked():
            out.append("train")
        if self.chk_xfer_split_val.isChecked():
            out.append("val")
        if self.chk_xfer_split_test.isChecked():
            out.append("test")
        return out

    def _sync_splits(self, splits: Sequence[str]) -> None:
        s = set(splits or [])
        with _blocked(self.chk_xfer_split_train):
            self.chk_xfer_split_train.setChecked("train" in s)
        with _blocked(self.chk_xfer_split_val):
            self.chk_xfer_split_val.setChecked("val" in s)
        with _blocked(self.chk_xfer_split_test):
            self.chk_xfer_split_test.setChecked("test" in s)

    def _get_checked_calib(self) -> List[str]:
        out: List[str] = []
        if self.chk_xfer_cal_none.isChecked():
            out.append("none")
        if self.chk_xfer_cal_source.isChecked():
            out.append("source")
        if self.chk_xfer_cal_target.isChecked():
            out.append("target")
        return out

    def _sync_calib_modes(self, modes: Sequence[str]) -> None:
        m = set(modes or [])
        with _blocked(self.chk_xfer_cal_none):
            self.chk_xfer_cal_none.setChecked("none" in m)
        with _blocked(self.chk_xfer_cal_source):
            self.chk_xfer_cal_source.setChecked("source" in m)
        with _blocked(self.chk_xfer_cal_target):
            self.chk_xfer_cal_target.setChecked("target" in m)

    def _set_combo_data(
        self,
        cmb: QComboBox,
        data: Any,
    ) -> None:
        for i in range(cmb.count()):
            if cmb.itemData(i) == data:
                with _blocked(cmb):
                    cmb.setCurrentIndex(i)
                return

    def set_has_result(self, has: bool) -> None:
        self.btn_xfer_view.setVisible(bool(has))
        self.btn_xfer_view.setEnabled(bool(has))
        
    def _as_float(self, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
        try:
            return float(v)
        except Exception:
            return None

    def _coords_complete_from_store(self) -> bool:
        """
        Return True if both cities have valid lat/lon in the store.
        """
        s = self._store

        lat_a = self._as_float(s.get("xfer.city_a_lat", None))
        lon_a = self._as_float(s.get("xfer.city_a_lon", None))
        lat_b = self._as_float(s.get("xfer.city_b_lat", None))
        lon_b = self._as_float(s.get("xfer.city_b_lon", None))

        if None in (lat_a, lon_a, lat_b, lon_b):
            return False

        # basic geographic bounds
        if not (-90.0 <= lat_a <= 90.0 and -180.0 <= lon_a <= 180.0):
            return False
        if not (-90.0 <= lat_b <= 90.0 and -180.0 <= lon_b <= 180.0):
            return False

        return True

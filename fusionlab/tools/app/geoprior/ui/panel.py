# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QSignalBlocker
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QComboBox, 
    QPlainTextEdit, 
    QDoubleSpinBox
)

from ..config.geoprior_config import GeoPriorConfig
from ..config.store import GeoConfigStore
from ..dialogs.feature_dialog import FeatureConfigDialog
from ..dialogs.architecture_dialog import ArchitectureConfigDialog

from .bindings import Binder
from .schema import default_fields, fields_for_section

Section = Tuple[str, str]

def _as_str_path(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, Path):
        return str(v)
    return str(v)


class ConfigCenterPanel(QWidget):
    """
    Experiment Setup tab (Config Center).

    This is the main "settings drawer" UI:
    - sticky header (profiles, dirty badge, actions)
    - left navigation
    - right scroll area with cards
    """

    def __init__(
        self,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.binder = Binder(store)

        self._sections: List[Section] = self._default_sections()
        self._section_to_widget: Dict[str, QWidget] = {}
        self._schema = default_fields()
        self._dataset_columns: List[str] = []
        self._col_combos: Dict[str, QComboBox] = {}

        self._build_ui()
        self._wire_store()

        self._apply_filter("")
        self._refresh_summary()

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        header = self._build_header()
        root.addWidget(header)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)

        left = self._build_nav()
        right = self._build_scroll()

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        splitter.setSizes([230, 9999])

        root.addWidget(splitter, 1)

    def _build_header(self) -> QWidget:
        w = QWidget(self)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.btn_load = QPushButton("Load", w)
        self.btn_save = QPushButton("Save", w)
        self.btn_reset = QPushButton("Reset", w)

        self.btn_apply = QPushButton("Apply", w)
        self.btn_diff = QPushButton("Show diff", w)

        self.chk_lock = QCheckBox("Lock for run", w)

        self.lbl_dirty = QLabel("0 changes", w)
        self.lbl_dirty.setObjectName("dirtyBadge")

        self.search = QLineEdit(w)
        self.search.setPlaceholderText("Search sections...")
        self.search.setClearButtonEnabled(True)

        lay.addWidget(self.btn_load)
        lay.addWidget(self.btn_save)
        lay.addWidget(self.btn_reset)

        sep = QFrame(w)
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        lay.addWidget(sep)

        lay.addWidget(self.lbl_dirty)
        lay.addWidget(self.btn_apply)
        lay.addWidget(self.btn_diff)
        lay.addWidget(self.chk_lock)

        lay.addStretch(1)
        lay.addWidget(self.search, 2)

        self.btn_load.clicked.connect(self._on_load)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_diff.clicked.connect(self._on_show_diff)
        self.search.textChanged.connect(self._apply_filter)

        return w

    def _build_nav(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.nav = QListWidget(w)
        self.nav.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        self.nav.setAlternatingRowColors(True)

        for sec_id, title in self._sections:
            it = QListWidgetItem(title, self.nav)
            it.setData(Qt.UserRole, sec_id)

        self.nav.currentItemChanged.connect(
            self._on_nav_changed
        )

        lay.addWidget(self.nav, 1)
        return w

    def _build_scroll(self) -> QWidget:
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)

        self.content = QWidget(self.scroll)
        self.content_lay = QVBoxLayout(self.content)
        self.content_lay.setContentsMargins(0, 0, 0, 0)
        self.content_lay.setSpacing(10)

        self.scroll.setWidget(self.content)

        self._add_cards()

        spacer = QWidget(self.content)
        spacer.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self.content_lay.addWidget(spacer, 1)

        return self.scroll

    def _add_cards(self) -> None:
        self._add_summary_card()
        self._add_project_paths_card()
        self._add_time_window_card()
        
        self._add_data_semantics_card() 
        self._add_coords_crs_card()
        self._add_features_card()
        self._add_censoring_card()
        self._add_scaling_card()
        self._add_arch_card()

        # XXX TODO: Placeholders for now (we fill them gradually)
        self._add_placeholder("train", "Training basics")
        self._add_placeholder("physics", "Physics & constraints")
        self._add_placeholder("prob", "Probabilistic outputs")
        self._add_placeholder("tuning", "Tuning")
        self._add_placeholder("device", "Device & runtime")
        self._add_placeholder("ui", "UI preferences")

        # Select the first visible section
        if self.nav.count() > 0:
            self.nav.setCurrentRow(0)

    # ------------------------------------------------------------------
    # Cards
    # ------------------------------------------------------------------
    def _add_card(self, sec_id: str, title: str) -> QGroupBox:
        box = QGroupBox(title, self.content)
        box.setObjectName(f"cfgCard_{sec_id}")

        lay = QGridLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)

        self.content_lay.addWidget(box)
        self._section_to_widget[sec_id] = box
        return box

    def _add_summary_card(self) -> None:
        box = self._add_card("summary", "Summary")
        lay = box.layout()

        self._sum_labels: Dict[str, QLabel] = {}

        fields = [
            ("Model", "model_name"),
            ("Mode", "mode"),
            ("Strategy", "training_strategy"),
            ("PDE mode", "pde_mode"),
            ("Train end", "train_end_year"),
            ("Forecast start", "forecast_start_year"),
            ("Horizon", "forecast_horizon_years"),
            ("Time steps", "time_steps"),
            ("Epochs", "epochs"),
            ("Batch", "batch_size"),
            ("LR", "learning_rate"),
            ("Results", "results_root"),
            ("Forecast end", "__forecast_end"),
        ]

        r = 0
        for label, key in fields:
            lab = QLabel(f"{label}:", box)
            val = QLabel("-", box)
            val.setTextInteractionFlags(
                Qt.TextSelectableByMouse
            )
            lay.addWidget(lab, r, 0)
            lay.addWidget(val, r, 1)
            self._sum_labels[key] = val
            r += 1

    def _add_project_paths_card(self) -> None:
        box = self._add_card("paths", "Project & paths")
        lay = box.layout()

        self.ed_city = QLineEdit(box)

        self.ed_dataset = QLineEdit(box)
        self.ed_dataset.setReadOnly(True)
        self.btn_dataset = QPushButton("Browse...", box)

        self.ed_results = QLineEdit(box)
        self.btn_results = QPushButton("Browse...", box)

        # Row 0: city
        lay.addWidget(QLabel("City:", box), 0, 0)
        lay.addWidget(self.ed_city, 0, 1, 1, 2)

        # Row 1: dataset
        lay.addWidget(QLabel("Dataset:", box), 1, 0)
        lay.addWidget(self.ed_dataset, 1, 1)
        lay.addWidget(self.btn_dataset, 1, 2)

        # Row 2: results
        lay.addWidget(QLabel("Results root:", box), 2, 0)
        lay.addWidget(self.ed_results, 2, 1)
        lay.addWidget(self.btn_results, 2, 2)

        # Advanced expander
        self.stage1_exp = self._make_expander(
            "Stage-1 reuse policy",
            parent=box,
        )
        lay.addWidget(self.stage1_exp["btn"], 3, 0, 1, 3)
        lay.addWidget(self.stage1_exp["body"], 4, 0, 1, 3)

        adv = self.stage1_exp["body_lay"]

        self.chk_clean = QCheckBox("Clean Stage-1 dir", box)
        self.chk_reuse = QCheckBox(
            "Auto reuse if match",
            box,
        )
        self.chk_force = QCheckBox(
            "Force rebuild if mismatch",
            box,
        )
        self.ed_audit = QLineEdit(box)

        adv.addWidget(self.chk_clean, 0, 0, 1, 2)
        adv.addWidget(self.chk_reuse, 1, 0, 1, 2)
        adv.addWidget(self.chk_force, 2, 0, 1, 2)

        adv.addWidget(QLabel("Audit stages:", box), 3, 0)
        adv.addWidget(self.ed_audit, 3, 1)

        # Binds
        self.binder.bind_line_edit(
            "city",
            self.ed_city,
        )
        self.binder.bind_line_edit(
            "dataset_path",
            self.ed_dataset,
            from_store=_as_str_path,
            on="editingFinished",
        )
        self.binder.bind_line_edit(
            "results_root",
            self.ed_results,
            from_store=_as_str_path,
            on="editingFinished",
        )

        self.binder.bind_checkbox(
            "clean_stage1_dir",
            self.chk_clean,
        )
        self.binder.bind_checkbox(
            "stage1_auto_reuse_if_match",
            self.chk_reuse,
        )
        self.binder.bind_checkbox(
            "stage1_force_rebuild_if_mismatch",
            self.chk_force,
        )
        self.binder.bind_line_edit(
            "audit_stages",
            self.ed_audit,
            on="editingFinished",
        )

        # Browse buttons
        self.btn_dataset.clicked.connect(
            self._browse_dataset
        )
        self.btn_results.clicked.connect(
            self._browse_results_root
        )

    def _add_time_window_card(self) -> None:
        box = self._add_card("time", "Time window & forecast")
        lay = box.layout()

        self.sp_train_end = QSpinBox(box)
        self.sp_train_end.setRange(1900, 2200)

        self.sp_fc_start = QSpinBox(box)
        self.sp_fc_start.setRange(1900, 2200)

        self.sp_horizon = QSpinBox(box)
        self.sp_horizon.setRange(1, 100)

        self.sp_steps = QSpinBox(box)
        self.sp_steps.setRange(1, 500)

        self.chk_future_npz = QCheckBox(
            "Build future NPZ",
            box,
        )

        lay.addWidget(QLabel("Train end year:", box), 0, 0)
        lay.addWidget(self.sp_train_end, 0, 1)

        lay.addWidget(QLabel("Forecast start:", box), 1, 0)
        lay.addWidget(self.sp_fc_start, 1, 1)

        lay.addWidget(QLabel("Horizon (years):", box), 2, 0)
        lay.addWidget(self.sp_horizon, 2, 1)

        lay.addWidget(QLabel("Time steps:", box), 3, 0)
        lay.addWidget(self.sp_steps, 3, 1)

        lay.addWidget(self.chk_future_npz, 4, 0, 1, 2)

        self.binder.bind_spin_box(
            "train_end_year",
            self.sp_train_end,
        )
        self.binder.bind_spin_box(
            "forecast_start_year",
            self.sp_fc_start,
        )
        self.binder.bind_spin_box(
            "forecast_horizon_years",
            self.sp_horizon,
        )
        self.binder.bind_spin_box(
            "time_steps",
            self.sp_steps,
        )
        self.binder.bind_checkbox(
            "build_future_npz",
            self.chk_future_npz,
        )

    def _add_placeholder(self, sec_id: str, title: str) -> None:
        box = self._add_card(sec_id, title)
        lay = box.layout()
        msg = QLabel("Coming soon...", box)
        msg.setStyleSheet("color: #888;")
        lay.addWidget(msg, 0, 0)

    # ------------------------------------------------------------------
    # Expander helper
    # ------------------------------------------------------------------
    def _make_expander(
        self,
        title: str,
        *,
        parent: QWidget,
    ) -> Dict[str, Any]:
        btn = QToolButton(parent)
        btn.setText(title)
        btn.setCheckable(True)
        btn.setChecked(False)
        btn.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        btn.setArrowType(Qt.RightArrow)

        body = QWidget(parent)
        body.setVisible(False)

        body_lay = QGridLayout(body)
        body_lay.setContentsMargins(8, 6, 8, 6)
        body_lay.setHorizontalSpacing(10)
        body_lay.setVerticalSpacing(6)

        def _toggle(on: bool) -> None:
            body.setVisible(bool(on))
            btn.setArrowType(
                Qt.DownArrow if on else Qt.RightArrow
            )

        btn.toggled.connect(_toggle)

        return {
            "btn": btn,
            "body": body,
            "body_lay": body_lay,
        }

    # ------------------------------------------------------------------
    # Store wiring
    # ------------------------------------------------------------------
    def _wire_store(self) -> None:
        self.store.dirty_changed.connect(
            self._on_dirty_changed
        )
        self.store.config_changed.connect(
            self._on_config_changed
        )
        self.store.config_replaced.connect(
            lambda _cfg: self._refresh_summary()
        )

        # initialize dirty badge
        self._on_dirty_changed(self.store.overrides_count())

    def _on_dirty_changed(self, n: int) -> None:
        self.lbl_dirty.setText(f"{int(n)} changes")

    def _on_config_changed(self, _keys: object) -> None:
        self._refresh_summary()

    # ------------------------------------------------------------------
    # Summary refresh
    # ------------------------------------------------------------------
    def _refresh_summary(self) -> None:
        cfg = self.store.cfg

        def _set(key: str, val: Any) -> None:
            lab = self._sum_labels.get(key)
            if lab is None:
                return
            lab.setText(_as_str_path(val))

        _set("model_name", cfg.model_name)
        _set("mode", cfg.mode)
        _set("training_strategy", cfg.training_strategy)
        _set("pde_mode", cfg.pde_mode)

        _set("train_end_year", cfg.train_end_year)
        _set("forecast_start_year", cfg.forecast_start_year)
        _set("forecast_horizon_years", cfg.forecast_horizon_years)
        _set("time_steps", cfg.time_steps)

        _set("epochs", cfg.epochs)
        _set("batch_size", cfg.batch_size)
        _set("learning_rate", cfg.learning_rate)
        _set("results_root", cfg.results_root)

        end_year = (
            cfg.forecast_start_year
            + cfg.forecast_horizon_years
            - 1
        )
        _set("__forecast_end", end_year)

    # ------------------------------------------------------------------
    # Nav behavior
    # ------------------------------------------------------------------
    def _default_sections(self) -> List[Section]:
        return [
            ("summary", "Summary"),
            ("paths", "Project & paths"),
            ("time", "Time window & forecast"),
            ("data_semantics", "Data columns & semantics"),
            ("coords", "Coordinates & CRS"),
            ("features", "Feature registry"),
            ("censoring", "Censoring & H-field"),
            ("scaling", "Scaling & units"),
            ("arch", "Model architecture"),
            ("train", "Training basics"),
            ("physics", "Physics & constraints"),
            ("prob", "Probabilistic outputs"),
            ("tuning", "Tuning"),
            ("device", "Device & runtime"),
            ("ui", "UI preferences"),
        ]

    def _apply_filter(self, text: str) -> None:
        q = (text or "").strip().lower()

        for i in range(self.nav.count()):
            it = self.nav.item(i)
            title = (it.text() or "").lower()
            it.setHidden(bool(q) and q not in title)

        # keep a valid selection
        cur = self.nav.currentItem()
        if cur is not None and cur.isHidden():
            for i in range(self.nav.count()):
                it = self.nav.item(i)
                if not it.isHidden():
                    self.nav.setCurrentItem(it)
                    break

    def _on_nav_changed(
        self,
        cur: Optional[QListWidgetItem],
        _prev: Optional[QListWidgetItem],
    ) -> None:
        if cur is None:
            return
        sec_id = cur.data(Qt.UserRole)
        w = self._section_to_widget.get(sec_id)
        if w is None:
            return
        self.scroll.ensureWidgetVisible(w, 0, 20)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_apply(self) -> None:
        # Force a "broadcast refresh" across tabs.
        keys = set(self.store.cfg.__dataclass_fields__.keys())
        self.store.config_changed.emit(keys)

    def _on_show_diff(self) -> None:
        data = self.store.snapshot_overrides()
        self._show_json_dialog(
            title="Config overrides (diff)",
            payload=data,
        )

    def _on_save(self) -> None:
        path, _flt = QFileDialog.getSaveFileName(
            self,
            "Save config snapshot",
            "",
            "JSON (*.json)",
        )
        if not path:
            return
        payload = self.store.cfg.as_dict()
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            self.store.error_raised.emit(str(exc))

    def _on_load(self) -> None:
        path, _flt = QFileDialog.getOpenFileName(
            self,
            "Load config snapshot",
            "",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError("Snapshot must be a dict.")
            with self.store.batch():
                self.store.patch(payload)
        except Exception as exc:
            self.store.error_raised.emit(str(exc))

    def _on_reset(self) -> None:
        # Reset to defaults (your current GUI behavior).
        try:
            cfg = GeoPriorConfig.from_defaults()
            self.store.replace_config(cfg)
        except Exception as exc:
            self.store.error_raised.emit(str(exc))

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------
    def _browse_dataset(self) -> None:
        path, _flt = QFileDialog.getOpenFileName(
            self,
            "Select dataset file",
            "",
            "CSV (*.csv);;All files (*.*)",
        )
        if not path:
            return
        self.store.patch({"dataset_path": path})

    def _browse_results_root(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select results root",
            "",
        )
        if not path:
            return
        self.store.patch({"results_root": path})

    # ------------------------------------------------------------------
    # Dialog helper
    # ------------------------------------------------------------------
    def _show_json_dialog(
        self,
        *,
        title: str,
        payload: Dict[str, Any],
    ) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(820, 520)

        lay = QVBoxLayout(dlg)

        txt = QTextEdit(dlg)
        txt.setReadOnly(True)

        try:
            pretty = json.dumps(payload, indent=2)
        except Exception:
            pretty = str(payload)

        txt.setPlainText(pretty)
        lay.addWidget(txt, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        btn_close = QPushButton("Close", dlg)
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)

        lay.addLayout(btn_row)

        dlg.exec_()
        
    def set_dataset_columns(self, columns: List[str]) -> None:
        cols = [str(c) for c in (columns or [])]
        self._dataset_columns = cols
    
        for key, cmb in self._col_combos.items():
            spec = self._schema.get(key)
            none_text = None if spec is None else spec.none_text
            self._set_combo_columns(
                cmb,
                cols,
                none_text=none_text,
            )
    
        self.binder.refresh_keys(set(self._col_combos.keys()))
    
    
    def _set_combo_columns(
        self,
        cmb: QComboBox,
        cols: List[str],
        *,
        none_text: Optional[str],
    ) -> None:
        with QSignalBlocker(cmb):
            cur = cmb.currentText()
            cmb.clear()
    
            if none_text is not None:
                cmb.addItem(str(none_text), None)
    
            for c in cols:
                cmb.addItem(str(c), str(c))
    
            # keep current text if possible
            if cur:
                idx = cmb.findText(cur)
                if idx >= 0:
                    cmb.setCurrentIndex(idx)
                elif cmb.isEditable():
                    cmb.setEditText(cur)
                        
    def _add_data_semantics_card(self) -> None:
        box = self._add_card(
            "data_semantics",
            "Data columns & semantics",
        )
        lay = box.layout()
    
        # 4-column grid: (label, widget) x 2
        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)
    
        fs = default_fields()
    
        def _h(text: str) -> QLabel:
            lab = QLabel(f"<b>{text}</b>", box)
            lab.setTextFormat(Qt.RichText)
            return lab
    
        def _lab(spec_key: str) -> QLabel:
            s = fs[spec_key]
            w = QLabel(f"{s.label}:", box)
            if s.tooltip:
                w.setToolTip(s.tooltip)
            return w
    
        def _col_combo(spec_key: str) -> QComboBox:
            s = fs[spec_key]
            cmb = QComboBox(box)
            cmb.setEditable(True)
            if cmb.lineEdit() is not None:
                cmb.lineEdit().setPlaceholderText(
                    s.placeholder or ""
                )
            if s.tooltip:
                cmb.setToolTip(s.tooltip)
    
            self._set_combo_columns(
                cmb,
                self._dataset_columns,
                none_text=s.none_text,
            )
    
            self._col_combos[spec_key] = cmb
    
            self.binder.bind_combo(
                spec_key,
                cmb,
                editable=True,
                none_text=s.none_text,
                use_item_data=False,
            )
            return cmb
    
        def _enum_combo(spec_key: str) -> QComboBox:
            s = fs[spec_key]
            cmb = QComboBox(box)
            if s.tooltip:
                cmb.setToolTip(s.tooltip)
    
            items = s.items if s.items is not None else []
            self.binder.bind_combo(
                spec_key,
                cmb,
                items=items,
                editable=False,
                none_text=s.none_text,
                use_item_data=True,
            )
            return cmb
    
        r = 0
        lay.addWidget(_h("Columns"), r, 0, 1, 4)
        r += 1
    
        # Row 1: time + subs
        lay.addWidget(_lab("time_col"), r, 0)
        lay.addWidget(_col_combo("time_col"), r, 1)
        lay.addWidget(_lab("subs_col"), r, 2)
        lay.addWidget(_col_combo("subs_col"), r, 3)
        r += 1
    
        # Row 2: lon + lat
        lay.addWidget(_lab("lon_col"), r, 0)
        lay.addWidget(_col_combo("lon_col"), r, 1)
        lay.addWidget(_lab("lat_col"), r, 2)
        lay.addWidget(_col_combo("lat_col"), r, 3)
        r += 1
    
        # Row 3: gwl + h_field
        lay.addWidget(_lab("gwl_col"), r, 0)
        lay.addWidget(_col_combo("gwl_col"), r, 1)
        lay.addWidget(_lab("h_field_col"), r, 2)
        lay.addWidget(_col_combo("h_field_col"), r, 3)
        r += 1
    
        # Row 4: z_surf + include flag
        lay.addWidget(_lab("z_surf_col"), r, 0)
        lay.addWidget(_col_combo("z_surf_col"), r, 1)
    
        chk_z = QCheckBox(
            fs["include_z_surf_as_static"].label,
            box,
        )
        tip = fs["include_z_surf_as_static"].tooltip
        if tip:
            chk_z.setToolTip(tip)
        self.binder.bind_checkbox(
            "include_z_surf_as_static",
            chk_z,
        )
        lay.addWidget(chk_z, r, 2, 1, 2)
        r += 1
    
        lay.addWidget(_h("Semantics"), r, 0, 1, 4)
        r += 1
    
        # Row 1: mode + subsidence kind
        lay.addWidget(_lab("mode"), r, 0)
        lay.addWidget(_enum_combo("mode"), r, 1)
        lay.addWidget(_lab("subsidence_kind"), r, 2)
        lay.addWidget(_enum_combo("subsidence_kind"), r, 3)
        r += 1
    
        # Row 2: gwl kind + sign
        lay.addWidget(_lab("gwl_kind"), r, 0)
        lay.addWidget(_enum_combo("gwl_kind"), r, 1)
        lay.addWidget(_lab("gwl_sign"), r, 2)
        lay.addWidget(_enum_combo("gwl_sign"), r, 3)
        r += 1
    
        # Row 3: head proxy + head name
        chk_head = QCheckBox(
            fs["use_head_proxy"].label,
            box,
        )
        tip = fs["use_head_proxy"].tooltip
        if tip:
            chk_head.setToolTip(tip)
        self.binder.bind_checkbox(
            "use_head_proxy",
            chk_head,
        )
        lay.addWidget(chk_head, r, 0, 1, 2)
    
        ed_head = QLineEdit(box)
        tip = fs["head_col"].tooltip
        if tip:
            ed_head.setToolTip(tip)
        self.binder.bind_line_edit(
            "head_col",
            ed_head,
            on="editingFinished",
        )
        lay.addWidget(_lab("head_col"), r, 2)
        lay.addWidget(ed_head, r, 3)
        r += 1
    
        # Advanced: optional gwl_dyn_index
        exp = self._make_expander("Advanced", parent=box)
        lay.addWidget(exp["btn"], r, 0, 1, 4)
        r += 1
        lay.addWidget(exp["body"], r, 0, 1, 4)
        r += 1
    
        body = exp["body_lay"]
        s = fs["gwl_dyn_index"]
    
        lab = QLabel(f"{s.label}:", box)
        enable = QCheckBox("Set", box)
        sp = QSpinBox(box)
        sp.setRange(-999999, 999999)
    
        if s.tooltip:
            lab.setToolTip(s.tooltip)
            enable.setToolTip(s.tooltip)
            sp.setToolTip(s.tooltip)
    
        body.addWidget(lab, 0, 0)
    
        row = QWidget(box)
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.addWidget(enable)
        row_lay.addWidget(sp, 1)
        body.addWidget(row, 0, 1)
    
        self.binder.bind_optional_spin_box(
            "gwl_dyn_index",
            sp,
            enable,
        )
            
    def _add_coords_crs_card(self) -> None:
        box = self._add_card(
            "coords",
            "Coordinates & CRS",
        )
        lay = box.layout()
    
        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)
    
        fs = default_fields()
    
        def _lab(spec_key: str) -> QLabel:
            s = fs[spec_key]
            w = QLabel(f"{s.label}:", box)
            if s.tooltip:
                w.setToolTip(s.tooltip)
            return w
    
        r = 0
    
        # Row 1: coord mode
        s = fs["coord_mode"]
        cmb = QComboBox(box)
        items = s.items if s.items is not None else []
        self.binder.bind_combo(
            "coord_mode",
            cmb,
            items=items,
            editable=False,
            none_text=s.none_text,
            use_item_data=True,
        )
        lay.addWidget(_lab("coord_mode"), r, 0)
        lay.addWidget(cmb, r, 1, 1, 3)
        r += 1
    
        # Row 2: epsg pair
        sp_src = QSpinBox(box)
        sp_src.setRange(0, 999999)
        self.binder.bind_spin_box("coord_src_epsg", sp_src)
    
        sp_utm = QSpinBox(box)
        sp_utm.setRange(0, 999999)
        self.binder.bind_spin_box("utm_epsg", sp_utm)
    
        lay.addWidget(_lab("coord_src_epsg"), r, 0)
        lay.addWidget(sp_src, r, 1)
        lay.addWidget(_lab("utm_epsg"), r, 2)
        lay.addWidget(sp_utm, r, 3)
        r += 1
    
        # Advanced expander
        exp = self._make_expander("Advanced", parent=box)
        lay.addWidget(exp["btn"], r, 0, 1, 4)
        r += 1
        lay.addWidget(exp["body"], r, 0, 1, 4)
        r += 1
    
        body = exp["body_lay"]
    
        chk_norm = QCheckBox(fs["normalize_coords"].label, box)
        chk_raw = QCheckBox(fs["keep_coords_raw"].label, box)
        chk_shift = QCheckBox(fs["shift_raw_coords"].label, box)
    
        for k, w in (
            ("normalize_coords", chk_norm),
            ("keep_coords_raw", chk_raw),
            ("shift_raw_coords", chk_shift),
        ):
            tip = fs[k].tooltip
            if tip:
                w.setToolTip(tip)
    
        self.binder.bind_checkbox("normalize_coords", chk_norm)
        self.binder.bind_checkbox("keep_coords_raw", chk_raw)
        self.binder.bind_checkbox("shift_raw_coords", chk_shift)
    
        body.addWidget(chk_norm, 0, 0)
        body.addWidget(chk_raw, 0, 1)
        body.addWidget(chk_shift, 1, 0)

    def _add_features_card(self) -> None:
        box = self._add_card("features", "Feature registry")
        lay = box.layout()
    
        # ----------------------------
        # Driver groups (3 columns)
        # ----------------------------
        grp_drivers = QGroupBox("Driver groups", box)
        g = QGridLayout(grp_drivers)
        g.setContentsMargins(8, 8, 8, 8)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)
    
        def _mk_list_editor(ph: str) -> QPlainTextEdit:
            ed = QPlainTextEdit(grp_drivers)
            ed.setPlaceholderText(ph)
            ed.setMinimumHeight(80)
            return ed
    
        ed_dyn = _mk_list_editor("One feature per line…")
        ed_sta = _mk_list_editor("One feature per line…")
        ed_fut = _mk_list_editor("One feature per line…")
    
        g.addWidget(QLabel("Dynamic"), 0, 0)
        g.addWidget(QLabel("Static"), 0, 1)
        g.addWidget(QLabel("Future"), 0, 2)
    
        g.addWidget(ed_dyn, 1, 0)
        g.addWidget(ed_sta, 1, 1)
        g.addWidget(ed_fut, 1, 2)
    
        self.binder.bind_list_text("dynamic_driver_features", ed_dyn)
        self.binder.bind_list_text("static_driver_features", ed_sta)
        self.binder.bind_list_text("future_driver_features", ed_fut)
    
        lay.addWidget(grp_drivers, 0, 0, 1, 4)
    
        # ----------------------------
        # Optional registries
        # ----------------------------
        grp_opt = QGroupBox("Optional registries (stage-1)", box)
        o = QGridLayout(grp_opt)
        o.setContentsMargins(8, 8, 8, 8)
        o.setHorizontalSpacing(8)
        o.setVerticalSpacing(6)
    
        ed_opt_num = QPlainTextEdit(grp_opt)
        ed_opt_num.setPlaceholderText(
            "List-of-lists. One group per line, comma-separated.\n"
            "Example:\n"
            "rain, pumping\n"
            "temp, humidity"
        )
        ed_opt_num.setMinimumHeight(110)
    
        ed_opt_cat = QPlainTextEdit(grp_opt)
        ed_opt_cat.setPlaceholderText(
            "Optional categorical registry.\n"
            "JSON recommended (list, list-of-lists, or strings)."
        )
        ed_opt_cat.setMinimumHeight(90)
    
        ed_norm = QPlainTextEdit(grp_opt)
        ed_norm.setPlaceholderText("Already normalized features…")
        ed_norm.setMinimumHeight(90)
    
        o.addWidget(QLabel("Optional numeric (groups)"), 0, 0)
        o.addWidget(ed_opt_num, 1, 0, 1, 2)
    
        o.addWidget(QLabel("Optional categorical"), 0, 2)
        o.addWidget(ed_opt_cat, 1, 2)
    
        o.addWidget(QLabel("Already normalized"), 2, 0)
        o.addWidget(ed_norm, 3, 0, 1, 2)
    
        self.binder.bind_list2_text("optional_numeric_features", ed_opt_num)
        self.binder.bind_json_text("optional_categorical_features", ed_opt_cat)
        self.binder.bind_list_text("already_normalized_features", ed_norm)
    
        lay.addWidget(grp_opt, 1, 0, 1, 4)
    
        # ----------------------------
        # Advanced: naming overrides
        # ----------------------------
        exp = self._make_expander("Advanced: feature naming overrides", parent=box)
        lay.addWidget(exp["btn"], 2, 0, 1, 4)
        lay.addWidget(exp["body"], 3, 0, 1, 4)
    
        body = exp["body_lay"]
    
        chk_dyn = QCheckBox("Override dynamic_feature_names", box)
        ed_dyn_names = QPlainTextEdit(box)
        ed_dyn_names.setMinimumHeight(70)
    
        chk_fut = QCheckBox("Override future_feature_names", box)
        ed_fut_names = QPlainTextEdit(box)
        ed_fut_names.setMinimumHeight(70)
    
        body.addWidget(chk_dyn, 0, 0)
        body.addWidget(ed_dyn_names, 1, 0)
        body.addWidget(chk_fut, 0, 1)
        body.addWidget(ed_fut_names, 1, 1)
    
        self.binder.bind_optional_list_text(
            "dynamic_feature_names",
            ed_dyn_names,
            chk_dyn,
        )
        self.binder.bind_optional_list_text(
            "future_feature_names",
            ed_fut_names,
            chk_fut,
        )
    
        # ----------------------------
        # Buttons row
        # ----------------------------
        row = QWidget(box)
        row_l = QHBoxLayout(row)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.addStretch(1)
    
        btn_open = QPushButton("Open Feature Config…", box)
        btn_open.clicked.connect(self._open_feature_config_dialog)
        row_l.addWidget(btn_open)
    
        lay.addWidget(row, 4, 0, 1, 4)

    def _open_feature_config_dialog(self) -> None:
        # FeatureConfigDialog expects base_cfg + overrides (dict)
        base_cfg = getattr(self.store.cfg, "_base_cfg", {}) or {}
        overrides = self.store.snapshot_overrides()
    
        # Provide columns to dialog (optional)
        df = None
        try:
            import pandas as pd
            if self._dataset_columns:
                df = pd.DataFrame(columns=self._dataset_columns)
        except Exception:
            df = None
    
        csv_path = ""
        try:
            if self.store.cfg.dataset_path:
                csv_path = str(self.store.cfg.dataset_path)
        except Exception:
            pass
    
        dlg = FeatureConfigDialog(
            csv_path=csv_path,
            base_cfg=base_cfg,
            current_overrides=overrides,
            parent=self,
            df=df,
        )
    
        if dlg.exec_() != dlg.Accepted:
            return
    
        ov = dlg.get_overrides() or {}
        mapped = {}
    
        MAP = {
            "DYNAMIC_DRIVER_FEATURES": "dynamic_driver_features",
            "STATIC_DRIVER_FEATURES": "static_driver_features",
            "FUTURE_DRIVER_FEATURES": "future_driver_features",
            "OPTIONAL_NUMERIC_FEATURES_REGISTRY": "optional_numeric_features",
            "OPTIONAL_NUMERIC_FEATURES": "optional_numeric_features",
            "OPTIONAL_CATEGORICAL_FEATURES_REGISTRY": "optional_categorical_features",
            "OPTIONAL_CATEGORICAL_FEATURES": "optional_categorical_features",
            "ALREADY_NORMALIZED_FEATURES": "already_normalized_features",
            "INCLUDE_CENSOR_FLAGS_AS_FUTURE":"include_censor_flags_as_future", 
        }
    
        for k, v in ov.items():
            if k in MAP:
                mapped[MAP[k]] = v
    
        if mapped:
            self.store.patch(mapped)
            
    def _add_censoring_card(self) -> None:
        box = self._add_card("censoring", "Censoring & H-field")
        lay = box.layout()
    
        lbl = QLabel("Censoring specs (JSON list of dicts):", box)
        lay.addWidget(lbl, 0, 0, 1, 4)
    
        ed = QPlainTextEdit(box)
        ed.setMinimumHeight(120)
        ed.setPlaceholderText(
            "[\n"
            "  {\n"
            '    "col": "H_eff",\n'
            '    "cap": 20.0,\n'
            '    "direction": "down",\n'
            '    "eff_mode": "clip",\n'
            '    "flag_threshold": 0.0\n'
            "  }\n"
            "]"
        )
        self.binder.bind_json_text(
            "censoring_specs",
            ed,
            default_on_empty=[],
        )
        lay.addWidget(ed, 1, 0, 1, 4)
    
        row = QWidget(box)
        row_l = QHBoxLayout(row)
        row_l.setContentsMargins(0, 0, 0, 0)
    
        chk_dyn = QCheckBox(
            "Include censor flags as dynamic",
            box,
        )
        chk_fut = QCheckBox(
            "Include censor flags as future",
            box,
        )
        chk_eff = QCheckBox(
            "Use effective H-field",
            box,
        )
    
        self.binder.bind_checkbox(
            "include_censor_flags_as_dynamic",
            chk_dyn,
        )
        self.binder.bind_checkbox(
            "include_censor_flags_as_future",
            chk_fut,
        )
        self.binder.bind_checkbox(
            "use_effective_h_field",
            chk_eff,
        )
    
        row_l.addWidget(chk_dyn)
        row_l.addWidget(chk_fut)
        row_l.addWidget(chk_eff)
        row_l.addStretch(1)
    
        lay.addWidget(row, 2, 0, 1, 4)
        
    def _browse_scaling_kwargs_json(self) -> None:
        path, _flt = QFileDialog.getOpenFileName(
            self,
            "Select scaling kwargs JSON",
            "",
            "JSON (*.json);;All files (*.*)",
        )
        if not path:
            return
        self.store.patch({"scaling_kwargs_json_path": path})
            
    def _add_scaling_card(self) -> None:
        box = self._add_card("scaling", "Scaling & units")
        lay = box.layout()
    
        def _dspin(
            *,
            minimum: float = -1e12,
            maximum: float = 1e12,
            decimals: int = 12,
            step: float = 1e-3,
        ) -> QDoubleSpinBox:
            sp = QDoubleSpinBox(box)
            sp.setRange(minimum, maximum)
            sp.setDecimals(decimals)
            sp.setSingleStep(step)
            return sp
    
        # ------------------------------------------------------------
        # Row: Stage-1 scaling toggles (inline)
        # ------------------------------------------------------------
        row0 = QWidget(box)
        r0 = QHBoxLayout(row0)
        r0.setContentsMargins(0, 0, 0, 0)
    
        chk_h = QCheckBox("Scale H-field", box)
        chk_g = QCheckBox("Scale GWL", box)
        chk_z = QCheckBox("Scale z_surf", box)
    
        self.binder.bind_checkbox("scale_h_field", chk_h)
        self.binder.bind_checkbox("scale_gwl", chk_g)
        self.binder.bind_checkbox("scale_z_surf", chk_z)
    
        r0.addWidget(chk_h)
        r0.addWidget(chk_g)
        r0.addWidget(chk_z)
        r0.addStretch(1)
    
        lay.addWidget(row0, 0, 0, 1, 4)
    
        # ------------------------------------------------------------
        # 4-column grid for the rest
        # ------------------------------------------------------------
        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)
    
        r = 1
    
        # Row: time_units + scaling_error_policy
        cmb_time = QComboBox(box)
        self.binder.bind_combo(
            "time_units",
            cmb_time,
            items=[
                ("year", "year"),
                ("day", "day"),
                ("second", "second"),
            ],
            editable=False,
            use_item_data=True,
        )
    
        cmb_policy = QComboBox(box)
        self.binder.bind_combo(
            "scaling_error_policy",
            cmb_policy,
            items=[
                ("raise", "raise"),
                ("warn", "warn"),
                ("ignore", "ignore"),
            ],
            editable=False,
            use_item_data=True,
        )
    
        lay.addWidget(QLabel("Time units:", box), r, 0)
        lay.addWidget(cmb_time, r, 1)
        lay.addWidget(QLabel("Error policy:", box), r, 2)
        lay.addWidget(cmb_policy, r, 3)
        r += 1
    
        # Row: auto_si_affine_from_stage1 (inline)
        chk_auto = QCheckBox(
            "Auto SI affine from Stage-1",
            box,
        )
        self.binder.bind_checkbox(
            "auto_si_affine_from_stage1",
            chk_auto,
        )
        lay.addWidget(chk_auto, r, 0, 1, 4)
        r += 1
    
        # ------------------------------------------------------------
        # Units + affine mapping
        # ------------------------------------------------------------
        sp_sub_u = _dspin(minimum=0.0, maximum=1e12)
        sp_sub_s = _dspin()
        sp_sub_b = _dspin(decimals=6, step=1e-2)
    
        self.binder.bind_double_spin_box(
            "subs_unit_to_si",
            sp_sub_u,
        )
        self.binder.bind_double_spin_box(
            "subs_scale_si",
            sp_sub_s,
        )
        self.binder.bind_double_spin_box(
            "subs_bias_si",
            sp_sub_b,
        )
    
        sp_head_u = _dspin(minimum=0.0, maximum=1e12)
        self.binder.bind_double_spin_box(
            "head_unit_to_si",
            sp_head_u,
        )
    
        # Optional head_scale_si / head_bias_si
        head_scale_enable = QCheckBox("Set", box)
        head_bias_enable = QCheckBox("Set", box)
    
        sp_head_s = _dspin()
        sp_head_b = _dspin(decimals=6, step=1e-2)
    
        self.binder.bind_optional_double_spin_box(
            "head_scale_si",
            sp_head_s,
            head_scale_enable,
        )
        self.binder.bind_optional_double_spin_box(
            "head_bias_si",
            sp_head_b,
            head_bias_enable,
        )
    
        sp_th_u = _dspin(minimum=0.0, maximum=1e12)
        sp_z_u = _dspin(minimum=0.0, maximum=1e12)
        sp_hmin = _dspin(minimum=-1e9, maximum=1e9, decimals=6)
    
        self.binder.bind_double_spin_box(
            "thickness_unit_to_si",
            sp_th_u,
        )
        self.binder.bind_double_spin_box(
            "z_surf_unit_to_si",
            sp_z_u,
        )
        self.binder.bind_double_spin_box(
            "h_field_min_si",
            sp_hmin,
        )
    
        # Row: subsidence mapping (unit/scale/bias)
        lay.addWidget(QLabel("Subs unit->SI:", box), r, 0)
        lay.addWidget(sp_sub_u, r, 1)
        lay.addWidget(QLabel("Head unit->SI:", box), r, 2)
        lay.addWidget(sp_head_u, r, 3)
        r += 1
    
        lay.addWidget(QLabel("Subs scale (SI):", box), r, 0)
        lay.addWidget(sp_sub_s, r, 1)
        lay.addWidget(QLabel("Thickness unit->SI:", box), r, 2)
        lay.addWidget(sp_th_u, r, 3)
        r += 1
    
        lay.addWidget(QLabel("Subs bias (SI):", box), r, 0)
        lay.addWidget(sp_sub_b, r, 1)
        lay.addWidget(QLabel("z_surf unit->SI:", box), r, 2)
        lay.addWidget(sp_z_u, r, 3)
        r += 1
    
        # Row: optional head scale/bias (two mini rows)
        w_scale = QWidget(box)
        wl1 = QHBoxLayout(w_scale)
        wl1.setContentsMargins(0, 0, 0, 0)
        wl1.addWidget(head_scale_enable)
        wl1.addWidget(sp_head_s, 1)
    
        w_bias = QWidget(box)
        wl2 = QHBoxLayout(w_bias)
        wl2.setContentsMargins(0, 0, 0, 0)
        wl2.addWidget(head_bias_enable)
        wl2.addWidget(sp_head_b, 1)
    
        lay.addWidget(QLabel("Head scale (opt):", box), r, 0)
        lay.addWidget(w_scale, r, 1)
        lay.addWidget(QLabel("Head bias (opt):", box), r, 2)
        lay.addWidget(w_bias, r, 3)
        r += 1
    
        lay.addWidget(QLabel("H-field min (SI):", box), r, 0)
        lay.addWidget(sp_hmin, r, 1)
    
        sp_dt = _dspin(minimum=0.0, maximum=1e9, decimals=6, step=1e-2)
        self.binder.bind_double_spin_box("dt_min_units", sp_dt)
    
        lay.addWidget(QLabel("dt_min_units:", box), r, 2)
        lay.addWidget(sp_dt, r, 3)
        r += 1
    
        # ------------------------------------------------------------
        # scaling_kwargs_json_path (line + browse)
        # ------------------------------------------------------------
        ed_path = QLineEdit(box)
        ed_path.setPlaceholderText("scaling_kwargs.json")
        self.binder.bind_line_edit(
            "scaling_kwargs_json_path",
            ed_path,
            on="editingFinished",
        )
    
        btn = QPushButton("Browse...", box)
        btn.clicked.connect(self._browse_scaling_kwargs_json)
    
        rowp = QWidget(box)
        rp = QHBoxLayout(rowp)
        rp.setContentsMargins(0, 0, 0, 0)
        rp.addWidget(ed_path, 1)
        rp.addWidget(btn, 0)
    
        lay.addWidget(QLabel("Scaling kwargs JSON:", box), r, 0)
        lay.addWidget(rowp, r, 1, 1, 3)
    
    def _add_arch_card(self) -> None:

        box = self._add_card("arch", "Model architecture")
        lay = box.layout()
    
        # specs = fields_for_section("arch")
    
        # --- helpers ---
        def add_row(r: int, label: str, w: QWidget, col_span: int = 2):
            lay.addWidget(QLabel(label + ":", box), r, 0)
            lay.addWidget(w, r, 1, 1, col_span)
    
        r = 0
    
        # Model name (editable combo feels modern)
        cmb_model = QComboBox(box)
        cmb_model.setEditable(True)
        cmb_model.addItems(["GeoPriorSubsNet"])
        self.binder.bind_combo(
            "model_name",
            cmb_model,
            editable=True,
            use_item_data=False,
        )
        add_row(r, "Model name", cmb_model)
        r += 1
    
        # Attention levels (list editor)
        # use QPlainTextEdit for clean UX (one per line)
        ed_att = QPlainTextEdit(box)
        ed_att.setPlaceholderText("e.g.\n1\n2\n3")
        ed_att.setFixedHeight(70)
        self.binder.bind_list_text(
            "attention_levels",
            ed_att,
            item_type=int,
        )
        add_row(r, "Attention levels", ed_att)
        r += 1
    
        # Grid for numeric knobs (2 per row)
        def make_spin(minv=0, maxv=4096):
            sp = QSpinBox(box)
            sp.setRange(minv, maxv)
            return sp
    
        def make_dspin(minv=0.0, maxv=1.0, step=0.01):
            sp = QDoubleSpinBox(box)
            sp.setDecimals(4)
            sp.setRange(minv, maxv)
            sp.setSingleStep(step)
            return sp
    
        sp_embed = make_spin()
        sp_hidden = make_spin()
        sp_lstm = make_spin()
        sp_att = make_spin()
        sp_heads = make_spin(1, 16)
        sp_dropout = make_dspin(0.0, 0.9, 0.01)
    
        self.binder.bind_spin_box("embed_dim", sp_embed)
        self.binder.bind_spin_box("hidden_units", sp_hidden)
        self.binder.bind_spin_box("lstm_units", sp_lstm)
        self.binder.bind_spin_box("attention_units", sp_att)
        self.binder.bind_spin_box("num_heads", sp_heads)
        self.binder.bind_double_spin_box("dropout_rate", sp_dropout)
    
        # Layout: two fields per row
        row = QWidget(box)
        row_l = QGridLayout(row)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setHorizontalSpacing(10)
    
        row_l.addWidget(QLabel("Embed dim:"), 0, 0)
        row_l.addWidget(sp_embed, 0, 1)
        row_l.addWidget(QLabel("Hidden units:"), 0, 2)
        row_l.addWidget(sp_hidden, 0, 3)
    
        row_l.addWidget(QLabel("LSTM units:"), 1, 0)
        row_l.addWidget(sp_lstm, 1, 1)
        row_l.addWidget(QLabel("Attention units:"), 1, 2)
        row_l.addWidget(sp_att, 1, 3)
    
        row_l.addWidget(QLabel("Heads:"), 2, 0)
        row_l.addWidget(sp_heads, 2, 1)
        row_l.addWidget(QLabel("Dropout:"), 2, 2)
        row_l.addWidget(sp_dropout, 2, 3)

        lay.addWidget(row, r, 0, 1, 3)
        r += 1
        
        # ------------------------------------------------------------
        # Advanced (memory_size, scales)
        # ------------------------------------------------------------
        exp = self._make_expander("Advanced", parent=box)
        lay.addWidget(exp["btn"], r, 0, 1, 3)
        r += 1
        lay.addWidget(exp["body"], r, 0, 1, 3)
        r += 1
        
        adv = exp["body_lay"]
        
        # memory_size (advanced)
        sp_mem = make_spin(1, 512)
        self.binder.bind_spin_box("memory_size", sp_mem)
        
        adv.addWidget(QLabel("Memory size:", box), 0, 0)
        adv.addWidget(sp_mem, 0, 1)
        
        # scales (advanced)
        ed_scales = QPlainTextEdit(box)
        ed_scales.setPlaceholderText("e.g.\n1.0\n2.0")
        ed_scales.setFixedHeight(70)
        
        self.binder.bind_list_text(
            "scales",
            ed_scales,
            item_type=float,
        )
        
        adv.addWidget(QLabel("Scales:", box), 1, 0)
        adv.addWidget(ed_scales, 1, 1)

        # Checkboxes in one line (compact UX)
        row_chk = QWidget(box)
        chk_l = QHBoxLayout(row_chk)
        chk_l.setContentsMargins(0, 0, 0, 0)
    
        chk_res = QCheckBox("Residuals", box)
        chk_bn = QCheckBox("Batch norm", box)
        chk_vsn = QCheckBox("VSN", box)
    
        chk_l.addWidget(chk_res)
        chk_l.addWidget(chk_bn)
        chk_l.addWidget(chk_vsn)
        chk_l.addStretch(1)
    
        self.binder.bind_checkbox("use_residuals", chk_res)
        self.binder.bind_checkbox("use_batch_norm", chk_bn)
        self.binder.bind_checkbox("use_vsn", chk_vsn)
    
        lay.addWidget(QLabel("Flags:"), r, 0)
        lay.addWidget(row_chk, r, 1, 1, 2)
        r += 1
    
        # VSN units (only enabled when VSN checked)
        sp_vsn = make_spin(0, 1024)
        self.binder.bind_spin_box("vsn_units", sp_vsn)
        add_row(r, "VSN units", sp_vsn)
        r += 1
    
        def _toggle_vsn(on: bool):
            sp_vsn.setEnabled(bool(on))
    
        chk_vsn.toggled.connect(_toggle_vsn)
        _toggle_vsn(chk_vsn.isChecked())
    
        # Button row
        btn_row = QWidget(box)
        btn_l = QHBoxLayout(btn_row)
        btn_l.setContentsMargins(0, 0, 0, 0)
    
        btn_arch = QPushButton("Open Architecture Config…", box)
        btn_arch.setCursor(Qt.PointingHandCursor)
        btn_l.addWidget(btn_arch)
        btn_l.addStretch(1)
    
        lay.addWidget(btn_row, r, 0, 1, 3)
    
        # Dialog wiring: apply overrides -> cfg fields
        def _open_arch_dialog():
            # cfg = self.store.cfg
    
            base = {}  # you can optionally pass cfg._base_cfg here
            cur = {}   # optional: pass cfg.arch_overrides
    
            dlg = ArchitectureConfigDialog(
                base_cfg=base,
                current_overrides=cur,
                parent=self,
            )
            if dlg.exec_() != QDialog.Accepted:
                return
    
            overrides = dlg.get_overrides()
    
            # Map NAT keys -> GeoPriorConfig fields
            mapping = {
                "ATTENTION_LEVELS": "attention_levels",
                "EMBED_DIM": "embed_dim",
                "HIDDEN_UNITS": "hidden_units",
                "LSTM_UNITS": "lstm_units",
                "ATTENTION_UNITS": "attention_units",
                "NUMBER_HEADS": "number_heads",
                "DROPOUT_RATE": "dropout_rate",
                "MEMORY_SIZE": "memory_size",
                "SCALES": "scales",
                "USE_RESIDUALS": "use_residuals",
                "USE_BATCH_NORM": "use_batch_norm",
                "USE_VSN": "use_vsn",
                "VSN_UNITS": "vsn_units",
            }
    
            patch = {}
            for k, v in (overrides or {}).items():
                if k in mapping:
                    patch[mapping[k]] = v
    
            if patch:
                self.store.patch(patch)
    
        btn_arch.clicked.connect(_open_arch_dialog)

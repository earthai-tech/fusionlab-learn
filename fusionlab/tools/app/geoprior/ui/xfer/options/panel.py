# geoprior/ui/xfer/options/panel.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer.options.panel

Advanced options panel for XferTab.

- Store-backed (xfer.* keys)
- Self-contained scroll + responsive columns
- Reset-to-defaults
- Reusable section widgets (for RunCenter later)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from PyQt5.QtCore import QEvent, QObject, QSignalBlocker, Qt
from PyQt5.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ....config.store import GeoConfigStore


def _blocked(w: QWidget) -> QSignalBlocker:
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


ADV_DEFAULTS: Dict[str, Any] = {
    "xfer.quantiles_override": None,
    "xfer.write_json": True,
    "xfer.write_csv": True,
    "xfer.prefer_tuned": True,
    "xfer.align_policy": "align_by_name_pad",
    "xfer.allow_reorder_dynamic": None,
    "xfer.allow_reorder_future": None,
    "xfer.interval_target": 0.80,
    "xfer.load_endpoint": "serve",
    "xfer.export_physics_payload": True,
    "xfer.export_physical_parameters_csv": True,
    "xfer.write_eval_future_csv": True,
    "xfer.rescale_modes": None,
    "xfer.strategies": None,
    "xfer.warm_split": "train",
    "xfer.warm_samples": 20000,
    "xfer.warm_frac": 0.0,
    "xfer.warm_epochs": 3,
    "xfer.warm_lr": 1e-4,
    "xfer.warm_seed": 0,
}


_MISSING = object()


def ensure_adv_defaults(store: GeoConfigStore) -> None:
    """
    Initialise missing xfer.* advanced keys only.

    Uses a sentinel so existing values (even None) are not
    treated as "missing".
    """
    s = store
    with s.batch():
        for k, v in ADV_DEFAULTS.items():
            if s.get(k, _MISSING) is _MISSING:
                s.set(k, v)


class OutputsAlignmentSection(QWidget):
    """
    Store-backed section widget:

    - quantiles override (+ chip)
    - output formats (json/csv)
    - prefer tuned
    - alignment policy (+ optional reorder)
    - interval target
    - load endpoint
    - export toggles
    """

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store

        ensure_adv_defaults(self._s)

        self._build_ui()
        self._wire()
        self._sync_from_store(keys=set())

        self._s.config_changed.connect(self._on_store_changed)

    # -------------------------
    # Public
    # -------------------------
    def get_quantiles(self) -> Optional[Sequence[float]]:
        return _parse_float_list(self.ed_xfer_quantiles.text())

    def get_state(self) -> Dict[str, Any]:
        return {
            "quantiles_override": self.get_quantiles(),
            "write_json": bool(self.chk_xfer_json.isChecked()),
            "write_csv": bool(self.chk_xfer_csv.isChecked()),
            "prefer_tuned": bool(
                self.chk_xfer_prefer_tuned.isChecked()
            ),
            "align_policy": str(
                self.cmb_xfer_align.currentData()
                or "align_by_name_pad"
            ),
            "allow_reorder_dynamic": self._opt_bool_from_combo(
                self.cmb_xfer_allow_re_dyn
            ),
            "allow_reorder_future": self._opt_bool_from_combo(
                self.cmb_xfer_allow_re_fut
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
        }

    def adv_keys(self) -> set[str]:
        return {
            "xfer.quantiles_override",
            "xfer.write_json",
            "xfer.write_csv",
            "xfer.prefer_tuned",
            "xfer.align_policy",
            "xfer.allow_reorder_dynamic",
            "xfer.allow_reorder_future",
            "xfer.interval_target",
            "xfer.load_endpoint",
            "xfer.export_physics_payload",
            "xfer.export_physical_parameters_csv",
            "xfer.write_eval_future_csv",
        }

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        root.addLayout(grid)

        # Quantiles
        self.ed_xfer_quantiles = QLineEdit()
        self.ed_xfer_quantiles.setPlaceholderText(
            "0.1,0.5,0.9"
        )

        self.lbl_xfer_q_chip = QLabel("AUTO")
        self.lbl_xfer_q_chip.setObjectName("xferAdvChip")

        self.btn_xfer_q_clear = QToolButton()
        self.btn_xfer_q_clear.setObjectName("miniAction")
        self.btn_xfer_q_clear.setIcon(
            self.style().standardIcon(
                QStyle.SP_DialogCloseButton
            )
        )
        self.btn_xfer_q_clear.setToolTip(
            "Clear quantiles override"
        )

        q_body = QWidget()
        q_row = QHBoxLayout(q_body)
        q_row.setContentsMargins(0, 0, 0, 0)
        q_row.setSpacing(6)
        q_row.addWidget(self.ed_xfer_quantiles, 1)
        q_row.addWidget(self.lbl_xfer_q_chip)
        q_row.addWidget(self.btn_xfer_q_clear)

        f_q = self._field(
            "Quantiles override",
            q_body,
            "Override output quantiles for eval/export.\n"
            "Leave empty to use model defaults.",
        )
        grid.addWidget(f_q, 0, 0, 1, 2)

        # Formats
        self.chk_xfer_json = QCheckBox("Write JSON")
        self.chk_xfer_csv = QCheckBox("Write CSV")

        io_body = QWidget()
        io = QHBoxLayout(io_body)
        io.setContentsMargins(0, 0, 0, 0)
        io.setSpacing(12)
        io.addWidget(self.chk_xfer_json)
        io.addWidget(self.chk_xfer_csv)
        io.addStretch(1)

        f_io = self._field(
            "Formats",
            io_body,
            "CSV is easier to audit.\n"
            "JSON enables richer summary panels.",
        )
        grid.addWidget(f_io, 1, 0)

        # Prefer tuned
        self.chk_xfer_prefer_tuned = QCheckBox(
            "Prefer tuned calibrator if available"
        )

        pt_body = QWidget()
        pt = QVBoxLayout(pt_body)
        pt.setContentsMargins(0, 0, 0, 0)
        pt.setSpacing(0)
        pt.addWidget(self.chk_xfer_prefer_tuned)

        f_pt = self._field(
            "Calibration",
            pt_body,
            "If tuning artifacts exist, reuse them\n"
            "instead of fitting a fresh calibrator.",
        )
        grid.addWidget(f_pt, 1, 1)

        # Align policy
        self.cmb_xfer_align = QComboBox()
        self.cmb_xfer_align.addItem(
            "Align by name (pad)",
            "align_by_name_pad",
        )
        self.cmb_xfer_align.addItem(
            "Strict (same columns)",
            "strict",
        )

        f_align = self._field(
            "Align policy",
            self.cmb_xfer_align,
            "How to align feature columns across cities.\n"
            "Strict = fail fast when mismatched.",
        )
        grid.addWidget(f_align, 2, 0)

        # Reorder dynamic
        self.cmb_xfer_allow_re_dyn = QComboBox()
        self._fill_opt_bool_combo(self.cmb_xfer_allow_re_dyn)

        f_rdyn = self._field(
            "Reorder dynamic",
            self.cmb_xfer_allow_re_dyn,
            "Auto = follow align policy.\n"
            "Allow = reorder if names match.\n"
            "Block = treat reorder as mismatch.",
        )
        grid.addWidget(f_rdyn, 2, 1)

        # Reorder future
        self.cmb_xfer_allow_re_fut = QComboBox()
        self._fill_opt_bool_combo(self.cmb_xfer_allow_re_fut)

        f_rfut = self._field(
            "Reorder future",
            self.cmb_xfer_allow_re_fut,
            "Same as above, for future-known inputs.",
        )
        grid.addWidget(f_rfut, 3, 0)

        # Interval target
        self.sp_xfer_interval = QDoubleSpinBox()
        self.sp_xfer_interval.setRange(0.10, 0.99)
        self.sp_xfer_interval.setSingleStep(0.01)
        self.sp_xfer_interval.setDecimals(2)

        f_int = self._field(
            "Interval target",
            self.sp_xfer_interval,
            "Controls calibration coverage.",
        )
        grid.addWidget(f_int, 3, 1)

        # Load endpoint
        self.cmb_xfer_endpoint = QComboBox()
        self.cmb_xfer_endpoint.addItem("serve", "serve")
        self.cmb_xfer_endpoint.addItem("export", "export")

        f_end = self._field(
            "Load endpoint",
            self.cmb_xfer_endpoint,
            "Choose where to load artifacts from.",
        )
        grid.addWidget(f_end, 4, 0)

        # Exports
        self.chk_xfer_phys_payload = QCheckBox(
            "Export physics payload"
        )
        self.chk_xfer_phys_csv = QCheckBox(
            "Export physical parameters CSV"
        )
        self.chk_xfer_eval_future = QCheckBox(
            "Write future evaluation CSV"
        )

        ex_body = QWidget()
        ex = QGridLayout(ex_body)
        ex.setContentsMargins(0, 0, 0, 0)
        ex.setHorizontalSpacing(12)
        ex.setVerticalSpacing(8)
        ex.addWidget(self.chk_xfer_phys_payload, 0, 0)
        ex.addWidget(self.chk_xfer_phys_csv, 0, 1)
        ex.addWidget(
            self.chk_xfer_eval_future,
            1,
            0,
            1,
            2,
        )

        f_ex = self._field(
            "Exports",
            ex_body,
            "Export physics/closures for audits.\n"
            "Write extra CSVs when available.",
        )
        grid.addWidget(f_ex, 4, 1)

        self._update_quantiles_chip()

    # -------------------------
    # Wiring
    # -------------------------
    def _wire(self) -> None:
        self.btn_xfer_q_clear.clicked.connect(
            lambda: self.ed_xfer_quantiles.setText("")
        )
        self.btn_xfer_q_clear.clicked.connect(
            self._update_quantiles_chip
        )
        self.btn_xfer_q_clear.clicked.connect(
            self._push_to_store
        )

        self.ed_xfer_quantiles.editingFinished.connect(
            self._update_quantiles_chip
        )
        self.ed_xfer_quantiles.editingFinished.connect(
            self._push_to_store
        )

        for cb in (
            self.chk_xfer_json,
            self.chk_xfer_csv,
            self.chk_xfer_prefer_tuned,
            self.chk_xfer_phys_payload,
            self.chk_xfer_phys_csv,
            self.chk_xfer_eval_future,
        ):
            cb.toggled.connect(self._push_to_store)

        self.cmb_xfer_align.currentIndexChanged.connect(
            self._push_to_store
        )
        self.cmb_xfer_allow_re_dyn.currentIndexChanged.connect(
            self._push_to_store
        )
        self.cmb_xfer_allow_re_fut.currentIndexChanged.connect(
            self._push_to_store
        )
        self.sp_xfer_interval.valueChanged.connect(
            self._push_to_store
        )
        self.cmb_xfer_endpoint.currentIndexChanged.connect(
            self._push_to_store
        )

    # -------------------------
    # Store I/O
    # -------------------------
    def _on_store_changed(self, keys: object) -> None:
        try:
            changed = set(keys or [])
        except Exception:
            changed = set()
        if not changed:
            return

        if not (changed & self.adv_keys()):
            return

        self._sync_from_store(keys=changed)

    def _sync_from_store(self, *, keys: set[str]) -> None:
        s = self._s

        def wants(k: str) -> bool:
            return (not keys) or (k in keys)

        if wants("xfer.quantiles_override"):
            q = s.get("xfer.quantiles_override", None)
            txt = ""
            if q:
                try:
                    txt = ",".join(str(x) for x in q)
                except Exception:
                    txt = ""
            with _blocked(self.ed_xfer_quantiles):
                self.ed_xfer_quantiles.setText(txt)

        if wants("xfer.write_json"):
            v = bool(s.get("xfer.write_json", True))
            with _blocked(self.chk_xfer_json):
                self.chk_xfer_json.setChecked(v)

        if wants("xfer.write_csv"):
            v = bool(s.get("xfer.write_csv", True))
            with _blocked(self.chk_xfer_csv):
                self.chk_xfer_csv.setChecked(v)

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

        if wants("xfer.allow_reorder_dynamic"):
            v = s.get("xfer.allow_reorder_dynamic", None)
            self._set_opt_bool_combo(
                self.cmb_xfer_allow_re_dyn,
                v,
            )

        if wants("xfer.allow_reorder_future"):
            v = s.get("xfer.allow_reorder_future", None)
            self._set_opt_bool_combo(
                self.cmb_xfer_allow_re_fut,
                v,
            )

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

        self._update_quantiles_chip()

    def _push_to_store(self) -> None:
        s = self._s
        with s.batch():
            q, ok, _txt = self._quantiles_ui_status()
            raw = (self.ed_xfer_quantiles.text() or "").strip()
            if ok or (not raw):
                s.set("xfer.quantiles_override", q)

            s.set(
                "xfer.write_json",
                bool(self.chk_xfer_json.isChecked()),
            )
            s.set(
                "xfer.write_csv",
                bool(self.chk_xfer_csv.isChecked()),
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
                "xfer.allow_reorder_dynamic",
                self._opt_bool_from_combo(
                    self.cmb_xfer_allow_re_dyn
                ),
            )
            s.set(
                "xfer.allow_reorder_future",
                self._opt_bool_from_combo(
                    self.cmb_xfer_allow_re_fut
                ),
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

    # -------------------------
    # Small UI helpers
    # -------------------------
    def _make_help_btn(self, tip: str) -> QToolButton:
        b = QToolButton()
        b.setObjectName("miniAction")
        b.setAutoRaise(True)
        b.setIcon(
            self.style().standardIcon(
                QStyle.SP_MessageBoxInformation
            )
        )
        b.setToolTip(tip)
        b.setCursor(Qt.WhatsThisCursor)
        b.setFixedSize(22, 22)
        return b

    def _field(
        self,
        title: str,
        body: QWidget,
        tip: str,
        *,
        right: Optional[QWidget] = None,
    ) -> QFrame:
        f = QFrame()
        f.setObjectName("xferField")

        v = QVBoxLayout(f)
        v.setContentsMargins(10, 8, 10, 10)
        v.setSpacing(6)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)

        lbl = QLabel(title)
        lbl.setObjectName("xferFieldTitle")

        top.addWidget(lbl)
        top.addStretch(1)

        if right is not None:
            top.addWidget(right)

        top.addWidget(self._make_help_btn(tip))

        v.addLayout(top)
        v.addWidget(body)
        return f

    def _fill_opt_bool_combo(self, cmb: QComboBox) -> None:
        cmb.clear()
        cmb.addItem("Auto", None)
        cmb.addItem("Allow", True)
        cmb.addItem("Block", False)

    def _opt_bool_from_combo(self, cmb: QComboBox) -> Optional[bool]:
        return cmb.currentData()

    def _set_opt_bool_combo(
        self,
        cmb: QComboBox,
        v: Optional[bool],
    ) -> None:
        for i in range(cmb.count()):
            if cmb.itemData(i) == v:
                with _blocked(cmb):
                    cmb.setCurrentIndex(i)
                return

    def _set_combo_data(self, cmb: QComboBox, data: Any) -> None:
        for i in range(cmb.count()):
            if cmb.itemData(i) == data:
                with _blocked(cmb):
                    cmb.setCurrentIndex(i)
                return

    def _quantiles_ui_status(
        self,
    ) -> Tuple[Optional[List[float]], bool, str]:
        raw = (self.ed_xfer_quantiles.text() or "").strip()
        if not raw:
            return None, True, "AUTO"

        q = _parse_float_list(raw)
        if q is None:
            return None, False, "INVALID"

        return list(q), True, "OK"

    def _update_quantiles_chip(self) -> None:
        _q, ok, txt = self._quantiles_ui_status()
        self.lbl_xfer_q_chip.setText(txt)
        self.lbl_xfer_q_chip.setProperty("ok", bool(ok))
        self.lbl_xfer_q_chip.style().unpolish(
            self.lbl_xfer_q_chip
        )
        self.lbl_xfer_q_chip.style().polish(
            self.lbl_xfer_q_chip
        )


class StrategyWarmStartSection(QWidget):
    """
    Store-backed section widget:

    - strategies: baseline/xfer/warm
    - rescale modes: as_is/strict
    - warm-start settings (enabled only if warm strategy)
    """

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store

        ensure_adv_defaults(self._s)

        self._build_ui()
        self._wire()
        self._sync_from_store(keys=set())

        self._s.config_changed.connect(self._on_store_changed)

    def get_state(self) -> Dict[str, Any]:
        return {
            "strategies": self._get_checked_strategies(),
            "rescale_modes": self._get_checked_rescale_modes(),
            "warm_split": str(
                self.cmb_xfer_warm_split.currentData()
                or "train"
            ),
            "warm_samples": int(self.sp_xfer_warm_samples.value()),
            "warm_frac": float(self.sp_xfer_warm_frac.value()),
            "warm_epochs": int(self.sp_xfer_warm_epochs.value()),
            "warm_lr": float(self.sp_xfer_warm_lr.value()),
            "warm_seed": int(self.sp_xfer_warm_seed.value()),
        }

    def adv_keys(self) -> set[str]:
        return {
            "xfer.strategies",
            "xfer.rescale_modes",
            "xfer.warm_split",
            "xfer.warm_samples",
            "xfer.warm_frac",
            "xfer.warm_epochs",
            "xfer.warm_lr",
            "xfer.warm_seed",
        }

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        root.addLayout(grid)

        self.chk_xfer_strat_baseline = QCheckBox(
            "In-domain baselines (A→A, B→B)"
        )
        self.chk_xfer_strat_xfer = QCheckBox(
            "Zero-shot transfer (A→B, B→A)"
        )
        self.chk_xfer_strat_warm = QCheckBox(
            "Warm-start fine-tune (A→(few B)→B)"
        )

        st_body = QWidget()
        st = QVBoxLayout(st_body)
        st.setContentsMargins(0, 0, 0, 0)
        st.setSpacing(6)
        st.addWidget(self.chk_xfer_strat_baseline)
        st.addWidget(self.chk_xfer_strat_xfer)
        st.addWidget(self.chk_xfer_strat_warm)

        f_st = self._field(
            "Strategies",
            st_body,
            "Pick which comparisons to run.\n"
            "Warm-start unlocks the panel below.",
        )
        grid.addWidget(f_st, 0, 0, 1, 2)

        self.chk_xfer_rmode_as_is = QCheckBox("as-is")
        self.chk_xfer_rmode_strict = QCheckBox("strict")

        rm_body = QWidget()
        rm = QHBoxLayout(rm_body)
        rm.setContentsMargins(0, 0, 0, 0)
        rm.setSpacing(12)
        rm.addWidget(self.chk_xfer_rmode_as_is)
        rm.addWidget(self.chk_xfer_rmode_strict)
        rm.addStretch(1)

        f_rm = self._field(
            "Rescale variants",
            rm_body,
            "Run multiple rescale variants for audits.\n"
            "Strict keeps scalers consistent.",
        )
        grid.addWidget(f_rm, 1, 0)

        self.cmb_xfer_warm_split = QComboBox()
        self.cmb_xfer_warm_split.addItem("train", "train")
        self.cmb_xfer_warm_split.addItem("val", "val")
        self.cmb_xfer_warm_split.addItem("test", "test")

        self.sp_xfer_warm_samples = QSpinBox()
        self.sp_xfer_warm_samples.setRange(1, 1_000_000)

        self.sp_xfer_warm_frac = QDoubleSpinBox()
        self.sp_xfer_warm_frac.setRange(0.0, 1.0)
        self.sp_xfer_warm_frac.setSingleStep(0.05)
        self.sp_xfer_warm_frac.setDecimals(2)

        self.sp_xfer_warm_epochs = QSpinBox()
        self.sp_xfer_warm_epochs.setRange(1, 10_000)

        self.sp_xfer_warm_lr = QDoubleSpinBox()
        self.sp_xfer_warm_lr.setRange(1e-8, 1.0)
        self.sp_xfer_warm_lr.setDecimals(8)
        self.sp_xfer_warm_lr.setSingleStep(1e-4)

        self.sp_xfer_warm_seed = QSpinBox()
        self.sp_xfer_warm_seed.setRange(0, 2_147_483_647)

        self._warm_box = QGroupBox("Warm-start settings")
        self._warm_box.setObjectName("xferWarmBox")

        warm = QGridLayout(self._warm_box)
        warm.setContentsMargins(8, 8, 8, 8)
        warm.setHorizontalSpacing(10)
        warm.setVerticalSpacing(8)

        warm.addWidget(QLabel("Warm split:"), 0, 0)
        warm.addWidget(self.cmb_xfer_warm_split, 0, 1)
        warm.addWidget(QLabel("Warm epochs:"), 0, 2)
        warm.addWidget(self.sp_xfer_warm_epochs, 0, 3)

        warm.addWidget(QLabel("Warm samples:"), 1, 0)
        warm.addWidget(self.sp_xfer_warm_samples, 1, 1)
        warm.addWidget(QLabel("Warm lr:"), 1, 2)
        warm.addWidget(self.sp_xfer_warm_lr, 1, 3)

        warm.addWidget(QLabel("Warm frac:"), 2, 0)
        warm.addWidget(self.sp_xfer_warm_frac, 2, 1)
        warm.addWidget(QLabel("Warm seed:"), 2, 2)
        warm.addWidget(self.sp_xfer_warm_seed, 2, 3)

        warm.setColumnStretch(1, 1)
        warm.setColumnStretch(3, 1)

        f_warm = self._field(
            "Warm-start",
            self._warm_box,
            "Settings used when warm-start is enabled.",
        )
        grid.addWidget(f_warm, 1, 1)

        self._update_warm_enabled()

    def _wire(self) -> None:
        for cb in (
            self.chk_xfer_strat_baseline,
            self.chk_xfer_strat_xfer,
            self.chk_xfer_strat_warm,
            self.chk_xfer_rmode_as_is,
            self.chk_xfer_rmode_strict,
        ):
            cb.toggled.connect(self._push_to_store)

        self.chk_xfer_strat_warm.toggled.connect(
            lambda _v: self._update_warm_enabled()
        )

        for w in (
            self.cmb_xfer_warm_split,
            self.sp_xfer_warm_samples,
            self.sp_xfer_warm_frac,
            self.sp_xfer_warm_epochs,
            self.sp_xfer_warm_lr,
            self.sp_xfer_warm_seed,
        ):
            if isinstance(w, QComboBox):
                w.currentIndexChanged.connect(
                    self._push_to_store
                )
            else:
                w.valueChanged.connect(  # type: ignore[attr-defined]
                    self._push_to_store
                )

    def _on_store_changed(self, keys: object) -> None:
        try:
            changed = set(keys or [])
        except Exception:
            changed = set()
        if not changed:
            return

        if not (changed & self.adv_keys()):
            return

        self._sync_from_store(keys=changed)

    def _sync_from_store(self, *, keys: set[str]) -> None:
        s = self._s

        def wants(k: str) -> bool:
            return (not keys) or (k in keys)

        if wants("xfer.strategies"):
            self._sync_strategies(
                s.get("xfer.strategies", None)
            )

        if wants("xfer.rescale_modes"):
            self._sync_rescale_modes(
                s.get("xfer.rescale_modes", None)
            )

        if wants("xfer.warm_split"):
            v = s.get("xfer.warm_split", "train")
            self._set_combo_data(self.cmb_xfer_warm_split, v)

        if wants("xfer.warm_samples"):
            v = s.get("xfer.warm_samples", 20000)
            with _blocked(self.sp_xfer_warm_samples):
                self.sp_xfer_warm_samples.setValue(int(v))

        if wants("xfer.warm_frac"):
            v = s.get("xfer.warm_frac", 0.0)
            with _blocked(self.sp_xfer_warm_frac):
                self.sp_xfer_warm_frac.setValue(float(v))

        if wants("xfer.warm_epochs"):
            v = s.get("xfer.warm_epochs", 3)
            with _blocked(self.sp_xfer_warm_epochs):
                self.sp_xfer_warm_epochs.setValue(int(v))

        if wants("xfer.warm_lr"):
            v = s.get("xfer.warm_lr", 1e-4)
            with _blocked(self.sp_xfer_warm_lr):
                self.sp_xfer_warm_lr.setValue(float(v))

        if wants("xfer.warm_seed"):
            v = s.get("xfer.warm_seed", 0)
            with _blocked(self.sp_xfer_warm_seed):
                self.sp_xfer_warm_seed.setValue(int(v))

        self._update_warm_enabled()

    def _push_to_store(self) -> None:
        s = self._s
        with s.batch():
            s.set(
                "xfer.strategies",
                self._get_checked_strategies(),
            )
            s.set(
                "xfer.rescale_modes",
                self._get_checked_rescale_modes(),
            )

            s.set(
                "xfer.warm_split",
                str(self.cmb_xfer_warm_split.currentData()),
            )
            s.set(
                "xfer.warm_samples",
                int(self.sp_xfer_warm_samples.value()),
            )
            s.set(
                "xfer.warm_frac",
                float(self.sp_xfer_warm_frac.value()),
            )
            s.set(
                "xfer.warm_epochs",
                int(self.sp_xfer_warm_epochs.value()),
            )
            s.set(
                "xfer.warm_lr",
                float(self.sp_xfer_warm_lr.value()),
            )
            s.set(
                "xfer.warm_seed",
                int(self.sp_xfer_warm_seed.value()),
            )

    def _get_checked_strategies(self) -> Optional[List[str]]:
        out: List[str] = []
        if self.chk_xfer_strat_baseline.isChecked():
            out.append("baseline")
        if self.chk_xfer_strat_xfer.isChecked():
            out.append("xfer")
        if self.chk_xfer_strat_warm.isChecked():
            out.append("warm")
        return out or None

    def _sync_strategies(self, v: Any) -> None:
        ss = set(v or [])
        with _blocked(self.chk_xfer_strat_baseline):
            self.chk_xfer_strat_baseline.setChecked(
                "baseline" in ss
            )
        with _blocked(self.chk_xfer_strat_xfer):
            self.chk_xfer_strat_xfer.setChecked("xfer" in ss)
        with _blocked(self.chk_xfer_strat_warm):
            self.chk_xfer_strat_warm.setChecked("warm" in ss)
        self._update_warm_enabled()

    def _get_checked_rescale_modes(self) -> Optional[List[str]]:
        out: List[str] = []
        if self.chk_xfer_rmode_as_is.isChecked():
            out.append("as_is")
        if self.chk_xfer_rmode_strict.isChecked():
            out.append("strict")
        return out or None

    def _sync_rescale_modes(self, v: Any) -> None:
        ss = set(v or [])
        with _blocked(self.chk_xfer_rmode_as_is):
            self.chk_xfer_rmode_as_is.setChecked("as_is" in ss)
        with _blocked(self.chk_xfer_rmode_strict):
            self.chk_xfer_rmode_strict.setChecked("strict" in ss)

    def _update_warm_enabled(self) -> None:
        on = bool(self.chk_xfer_strat_warm.isChecked())
        self._warm_box.setEnabled(on)
        for w in (
            self.cmb_xfer_warm_split,
            self.sp_xfer_warm_samples,
            self.sp_xfer_warm_frac,
            self.sp_xfer_warm_epochs,
            self.sp_xfer_warm_lr,
            self.sp_xfer_warm_seed,
        ):
            w.setEnabled(on)

    def _make_help_btn(self, tip: str) -> QToolButton:
        b = QToolButton()
        b.setObjectName("miniAction")
        b.setAutoRaise(True)
        b.setIcon(
            self.style().standardIcon(
                QStyle.SP_MessageBoxInformation
            )
        )
        b.setToolTip(tip)
        b.setCursor(Qt.WhatsThisCursor)
        b.setFixedSize(22, 22)
        return b

    def _field(
        self,
        title: str,
        body: QWidget,
        tip: str,
        *,
        right: Optional[QWidget] = None,
    ) -> QFrame:
        f = QFrame()
        f.setObjectName("xferField")

        v = QVBoxLayout(f)
        v.setContentsMargins(10, 8, 10, 10)
        v.setSpacing(6)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)

        lbl = QLabel(title)
        lbl.setObjectName("xferFieldTitle")

        top.addWidget(lbl)
        top.addStretch(1)

        if right is not None:
            top.addWidget(right)

        top.addWidget(self._make_help_btn(tip))

        v.addLayout(top)
        v.addWidget(body)
        return f

    def _set_combo_data(self, cmb: QComboBox, data: Any) -> None:
        for i in range(cmb.count()):
            if cmb.itemData(i) == data:
                with _blocked(cmb):
                    cmb.setCurrentIndex(i)
                return


class XferOptionsPanel(QWidget):
    """
    Advanced options panel (scrollable).

    XferTab can embed this directly:
        self.options_panel = XferOptionsPanel(store=store)

    Internally composed of two reusable section widgets.
    """

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store

        ensure_adv_defaults(self._s)

        self._build_ui()
        self._update_adv_layout(self._scroll.viewport().width())

    # -------------------------------------------------
    # Public
    # -------------------------------------------------
    def get_quantiles(self) -> Optional[Sequence[float]]:
        return self.outputs.get_quantiles()

    def get_state(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out.update(self.outputs.get_state())
        out.update(self.strategy.get_state())
        return out

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("xferAdvScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        self._scroll.horizontalScrollBar().setSingleStep(24)
        self._scroll.horizontalScrollBar().setPageStep(240)

        self._content = self._build_advanced_box()
        self._scroll.setWidget(self._content)

        self._scroll.viewport().installEventFilter(self)
        root.addWidget(self._scroll, 1)

    def _build_advanced_box(self) -> QWidget:
        outer = QWidget()
        lay = QVBoxLayout(outer)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(10)

        # Header row: title + reset
        top = QHBoxLayout()
        top.setSpacing(10)

        tcol = QVBoxLayout()
        tcol.setContentsMargins(0, 0, 0, 0)
        tcol.setSpacing(2)

        ttl = QLabel("Advanced options")
        ttl.setObjectName("xferAdvTitle")

        sub = QLabel(
            "Fine control for audits and transfer semantics."
        )
        sub.setObjectName("setupCardSubtitle")

        tcol.addWidget(ttl)
        tcol.addWidget(sub)
        top.addLayout(tcol, 1)

        self.btn_xfer_adv_reset = QToolButton()
        self.btn_xfer_adv_reset.setObjectName("miniAction")
        self.btn_xfer_adv_reset.setIcon(
            self.style().standardIcon(QStyle.SP_BrowserReload)
        )
        self.btn_xfer_adv_reset.setToolTip(
            "Reset Advanced options to defaults."
        )
        self.btn_xfer_adv_reset.clicked.connect(
            self._on_adv_reset
        )
        top.addWidget(self.btn_xfer_adv_reset)

        lay.addLayout(top)

        # Responsive columns
        self._adv_cols = QBoxLayout(QBoxLayout.LeftToRight)
        self._adv_cols.setContentsMargins(0, 0, 0, 0)
        self._adv_cols.setSpacing(12)

        self._adv_left = QWidget()
        left = QVBoxLayout(self._adv_left)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(10)
        left.setAlignment(Qt.AlignTop)

        self._adv_right = QWidget()
        right = QVBoxLayout(self._adv_right)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        right.setAlignment(Qt.AlignTop)

        self._adv_cols.addWidget(self._adv_left, 1)
        self._adv_cols.addWidget(self._adv_right, 1)
        lay.addLayout(self._adv_cols)

        # Sections (reusable widgets inside)
        self.outputs = OutputsAlignmentSection(store=self._s)
        self.strategy = StrategyWarmStartSection(store=self._s)

        left.addWidget(
            self._wrap_section(
                "Outputs & alignment",
                QStyle.SP_FileDialogDetailedView,
                self.outputs,
            )
        )
        right.addWidget(
            self._wrap_section(
                "Strategies & warm-start",
                QStyle.SP_ArrowRight,
                self.strategy,
            )
        )

        left.addStretch(1)
        right.addStretch(1)

        # Compatibility aliases (optional, safe)
        self._alias_widgets_for_compat()

        return outer

    def _alias_widgets_for_compat(self) -> None:
        """
        Preserve the old attribute names so nothing breaks.
        """
        o = self.outputs
        s = self.strategy

        self.ed_xfer_quantiles = o.ed_xfer_quantiles
        self.lbl_xfer_q_chip = o.lbl_xfer_q_chip
        self.btn_xfer_q_clear = o.btn_xfer_q_clear

        self.chk_xfer_json = o.chk_xfer_json
        self.chk_xfer_csv = o.chk_xfer_csv
        self.chk_xfer_prefer_tuned = o.chk_xfer_prefer_tuned

        self.cmb_xfer_align = o.cmb_xfer_align
        self.cmb_xfer_allow_re_dyn = o.cmb_xfer_allow_re_dyn
        self.cmb_xfer_allow_re_fut = o.cmb_xfer_allow_re_fut
        self.sp_xfer_interval = o.sp_xfer_interval
        self.cmb_xfer_endpoint = o.cmb_xfer_endpoint

        self.chk_xfer_phys_payload = o.chk_xfer_phys_payload
        self.chk_xfer_phys_csv = o.chk_xfer_phys_csv
        self.chk_xfer_eval_future = o.chk_xfer_eval_future

        self.chk_xfer_strat_baseline = s.chk_xfer_strat_baseline
        self.chk_xfer_strat_xfer = s.chk_xfer_strat_xfer
        self.chk_xfer_strat_warm = s.chk_xfer_strat_warm

        self.chk_xfer_rmode_as_is = s.chk_xfer_rmode_as_is
        self.chk_xfer_rmode_strict = s.chk_xfer_rmode_strict

        self.cmb_xfer_warm_split = s.cmb_xfer_warm_split
        self.sp_xfer_warm_samples = s.sp_xfer_warm_samples
        self.sp_xfer_warm_frac = s.sp_xfer_warm_frac
        self.sp_xfer_warm_epochs = s.sp_xfer_warm_epochs
        self.sp_xfer_warm_lr = s.sp_xfer_warm_lr
        self.sp_xfer_warm_seed = s.sp_xfer_warm_seed

    def _wrap_section(
        self,
        title: str,
        icon: QStyle.StandardPixmap,
        body: QWidget,
    ) -> QFrame:
        frame = QFrame()
        frame.setObjectName("xferAdvSection")

        v = QVBoxLayout(frame)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        header = QToolButton()
        header.setObjectName("xferAdvToggle")
        header.setCheckable(True)
        header.setChecked(True)
        header.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        header.setIcon(self.style().standardIcon(icon))
        header.setText(title)
        header.setArrowType(Qt.DownArrow)

        def _toggle(on: bool) -> None:
            body.setVisible(bool(on))
            header.setArrowType(
                Qt.DownArrow if on else Qt.RightArrow
            )

        header.toggled.connect(_toggle)

        v.addWidget(header)
        v.addWidget(body)

        return frame

    # -------------------------------------------------
    # Qt events
    # -------------------------------------------------
    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:
        if obj is self._scroll.viewport():
            if ev.type() == QEvent.Resize:
                self._update_adv_layout(
                    self._scroll.viewport().width()
                )
        return super().eventFilter(obj, ev)

    def _update_adv_layout(self, w: int) -> None:
        bp = 1020
        stacked = int(w) < int(bp)

        if stacked:
            self._adv_cols.setDirection(QBoxLayout.TopToBottom)
            self._scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarAlwaysOff
            )
            return

        self._adv_cols.setDirection(QBoxLayout.LeftToRight)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )

    # -------------------------------------------------
    # Actions
    # -------------------------------------------------
    def _on_adv_reset(self) -> None:
        s = self._s
        with s.batch():
            for k, v in ADV_DEFAULTS.items():
                s.set(k, v)

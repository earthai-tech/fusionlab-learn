# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config.prior_schema import FieldKey
from ..config.store import GeoConfigStore
from ..config.geoprior_config import default_tuner_search_space


class ModelParamsDialog(QDialog):
    """
    Store-driven dialog for extra model-level tuning parameters.

    Covers:
    - memory_size (choices)
    - scales (list-of-list of ints)
    - attention_levels (list-of-list of strings)
    - use_batch_norm / use_residuals / use_vsn (boolean HPs)
    - batch_size, epochs (choices)

    Notes
    -----
    - Reads tuner_search_space from GeoConfigStore.
    - Writes only on OK (merge into tuner_search_space).
    - Cancel does not touch the store.
    """

    _KEYS = (
        "memory_size",
        "scales",
        "attention_levels",
        "use_batch_norm",
        "use_residuals",
        "use_vsn",
        "batch_size",
        "epochs",
    )

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store

        self.setWindowTitle("Model-level tuning parameters")
        self.setModal(True)

        self._scale_combos: List[List[int]] = []
        self._attn_combos: List[List[str]] = []

        root = QVBoxLayout(self)
        root.setSpacing(10)

        # -------------------------
        # 1) Memory + training card
        # -------------------------
        mem_group = QGroupBox("Core sizes")
        mem_layout = QGridLayout()
        r = 0

        self.ed_memory_size = QLineEdit()
        self.ed_memory_size.setPlaceholderText("e.g. 50, 100")

        self.ed_batch_size = QLineEdit()
        self.ed_batch_size.setPlaceholderText("e.g. 32, 64")

        self.ed_epochs = QLineEdit()
        self.ed_epochs.setPlaceholderText("e.g. 30, 50")

        mem_layout.addWidget(QLabel("Memory size (choices):"), r, 0)
        mem_layout.addWidget(self.ed_memory_size, r, 1)
        r += 1

        mem_layout.addWidget(QLabel("Batch size (choices):"), r, 0)
        mem_layout.addWidget(self.ed_batch_size, r, 1)
        r += 1

        mem_layout.addWidget(QLabel("Epochs (choices):"), r, 0)
        mem_layout.addWidget(self.ed_epochs, r, 1)

        mem_group.setLayout(mem_layout)
        root.addWidget(mem_group)

        # -------------------------
        # 2) Multi-scale card
        # -------------------------
        scales_group = QGroupBox("Multi-scale configuration")
        scales_layout = QVBoxLayout()

        top_row = QHBoxLayout()
        self.ed_scales_combo = QLineEdit()
        self.ed_scales_combo.setPlaceholderText("Scale combo, e.g. 1, 2")
        self.btn_scales_add = QPushButton("Add scale")
        self.btn_scales_clear = QPushButton("Clear")

        top_row.addWidget(QLabel("Scales:"))
        top_row.addWidget(self.ed_scales_combo, 1)
        top_row.addWidget(self.btn_scales_add)
        top_row.addWidget(self.btn_scales_clear)

        self.list_scales = QListWidget()
        self.list_scales.setSelectionMode(QListWidget.ExtendedSelection)

        scales_layout.addLayout(top_row)
        scales_layout.addWidget(QLabel("Current scale combos:"))
        scales_layout.addWidget(self.list_scales, 1)

        scales_group.setLayout(scales_layout)
        root.addWidget(scales_group)

        # -------------------------
        # 3) Attention-level card
        # -------------------------
        att_group = QGroupBox("Attention levels")
        att_layout = QVBoxLayout()

        cb_row = QHBoxLayout()
        self.chk_att_cross = QCheckBox("cross")
        self.chk_att_hier = QCheckBox("hierarchical")
        self.chk_att_mem = QCheckBox("memory")
        cb_row.addWidget(QLabel("Select levels then add:"))
        cb_row.addWidget(self.chk_att_cross)
        cb_row.addWidget(self.chk_att_hier)
        cb_row.addWidget(self.chk_att_mem)
        cb_row.addStretch(1)

        add_row = QHBoxLayout()
        self.btn_att_add = QPushButton("Add level")
        self.btn_att_clear = QPushButton("Clear")
        add_row.addWidget(self.btn_att_add)
        add_row.addWidget(self.btn_att_clear)
        add_row.addStretch(1)

        self.list_att = QListWidget()
        self.list_att.setSelectionMode(QListWidget.ExtendedSelection)

        att_layout.addLayout(cb_row)
        att_layout.addLayout(add_row)
        att_layout.addWidget(QLabel("Current attention combos:"))
        att_layout.addWidget(self.list_att, 1)

        att_group.setLayout(att_layout)
        root.addWidget(att_group)

        # -------------------------
        # 4) Boolean toggles
        # -------------------------
        bool_group = QGroupBox("Boolean hyperparameters")
        bool_layout = QVBoxLayout()

        self.chk_use_batch_norm = QCheckBox(
            "Tune 'use batch norm' as boolean HP"
        )
        self.chk_use_residuals = QCheckBox(
            "Tune 'use residuals' as boolean HP"
        )
        self.chk_use_vsn = QCheckBox("Tune 'use vsn' as boolean HP")

        bool_layout.addWidget(self.chk_use_batch_norm)
        bool_layout.addWidget(self.chk_use_residuals)
        bool_layout.addWidget(self.chk_use_vsn)
        bool_group.setLayout(bool_layout)
        root.addWidget(bool_group)

        # -------------------------
        # Buttons
        # -------------------------
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_reset = QPushButton("Reset to defaults")
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")
        btn_row.addWidget(self.btn_reset)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_ok)
        root.addLayout(btn_row)

        # Signals
        self.btn_scales_add.clicked.connect(self._on_add_scale_combo)
        self.btn_scales_clear.clicked.connect(self._on_clear_scales)
        self.btn_att_add.clicked.connect(self._on_add_att_combo)
        self.btn_att_clear.clicked.connect(self._on_clear_att)

        self.btn_ok.clicked.connect(self._on_ok)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_reset.clicked.connect(self._on_reset_clicked)

        # init from store
        self._defaults = self._get_defaults()
        self._space0 = self._get_space_merged(self._defaults)

        self.load_from_space(self._space0, self._defaults)
        self._init_norm = self._norm_fragment(
            self.to_search_space_fragment()
        )

    # --------------------------------------------------------------
    # Store helpers
    # --------------------------------------------------------------
    def _get_defaults(self) -> Dict[str, Any]:
        off = self._store.get_value(
            FieldKey("offset_mode"),
            default="mul",
        )
        try:
            return default_tuner_search_space(offset_mode=str(off))
        except Exception:
            return default_tuner_search_space()

    def _get_space_merged(
        self,
        defaults: Dict[str, Any],
    ) -> Dict[str, Any]:
        cur = self._store.get_value(
            FieldKey("tuner_search_space"),
            default={},
        )
        if not isinstance(cur, dict):
            cur = {}
        merged = dict(defaults)
        merged.update(dict(cur))
        return merged

    def _apply_fragment(self, frag: Dict[str, Any]) -> None:
        if not frag:
            return
        with self._store.batch():
            self._store.merge_dict_field(
                "tuner_search_space",
                dict(frag),
                replace=False,
            )

    # --------------------------------------------------------------
    # Small helpers (parsing)
    # --------------------------------------------------------------
    def _parse_int_list(
        self,
        text: str,
        fallback: List[int],
    ) -> List[int]:
        vals: List[int] = []
        for tok in text.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                vals.append(int(tok))
            except Exception:
                continue
        return vals or list(fallback)

    # ---- scales ----
    def _on_add_scale_combo(self) -> None:
        combo = self._parse_int_list(self.ed_scales_combo.text(), [])
        if not combo:
            return
        if combo not in self._scale_combos:
            self._scale_combos.append(combo)
            item = QListWidgetItem(", ".join(str(v) for v in combo))
            self.list_scales.addItem(item)
        self.ed_scales_combo.clear()

    def _on_clear_scales(self) -> None:
        self._scale_combos = []
        self.list_scales.clear()

    # ---- attention ----
    def _current_att_levels(self) -> List[str]:
        levels: List[str] = []
        if self.chk_att_cross.isChecked():
            levels.append("cross")
        if self.chk_att_hier.isChecked():
            levels.append("hierarchical")
        if self.chk_att_mem.isChecked():
            levels.append("memory")
        return levels

    def _on_add_att_combo(self) -> None:
        levels = self._current_att_levels()
        if not levels:
            return
        if levels not in self._attn_combos:
            self._attn_combos.append(levels)
            item = QListWidgetItem(" + ".join(levels))
            self.list_att.addItem(item)

    def _on_clear_att(self) -> None:
        self._attn_combos = []
        self.list_att.clear()
        self.chk_att_cross.setChecked(False)
        self.chk_att_hier.setChecked(False)
        self.chk_att_mem.setChecked(False)

    # --------------------------------------------------------------
    # Public API (kept for compatibility)
    # --------------------------------------------------------------
    def load_from_space(
        self,
        space: Dict[str, Any],
        defaults: Dict[str, Any],
    ) -> None:
        """Populate widgets from a TUNER_SEARCH_SPACE dict."""

        def _get(name: str, local_default: Any) -> Any:
            if name in space:
                return space[name]
            if name in defaults:
                return defaults[name]
            return local_default

        # memory_size / batch_size / epochs
        mem_spec = _get("memory_size", [50])
        mem_vals = mem_spec if isinstance(mem_spec, list) else [mem_spec]
        self.ed_memory_size.setText(
            ", ".join(str(int(v)) for v in mem_vals)
        )

        bs_spec = _get("batch_size", [32])
        bs_vals = bs_spec if isinstance(bs_spec, list) else [bs_spec]
        self.ed_batch_size.setText(
            ", ".join(str(int(v)) for v in bs_vals)
        )

        ep_spec = _get("epochs", [50])
        ep_vals = ep_spec if isinstance(ep_spec, list) else [ep_spec]
        self.ed_epochs.setText(", ".join(str(int(v)) for v in ep_vals))

        # scales (list[list[int]])
        scales_spec = _get("scales", [[1, 2]])
        self._scale_combos = []
        self.list_scales.clear()
        if isinstance(scales_spec, list):
            for combo in scales_spec:
                if isinstance(combo, (list, tuple)):
                    ints = [int(v) for v in combo]
                else:
                    ints = [int(scales_spec)]
                self._scale_combos.append(ints)
                self.list_scales.addItem(", ".join(str(v) for v in ints))

        # attention_levels
        att_spec = _get(
            "attention_levels",
            [["cross", "hierarchical", "memory"]],
        )
        self._attn_combos = []
        self.list_att.clear()
        if isinstance(att_spec, list):
            is_flat = all(isinstance(a, str) for a in att_spec)
            combos = [att_spec] if is_flat else att_spec
            for combo in combos:
                levels = [str(l) for l in combo]
                self._attn_combos.append(levels)
                self.list_att.addItem(" + ".join(levels))

        # boolean HPs – treat dict(type='bool') as "tuned"
        def _is_bool_hp(val: Any) -> bool:
            return isinstance(val, dict) and (
                str(val.get("type", "")).lower() in ("bool", "boolean")
            )

        self.chk_use_batch_norm.setChecked(
            _is_bool_hp(_get("use_batch_norm", {}))
        )
        self.chk_use_residuals.setChecked(
            _is_bool_hp(_get("use_residuals", {}))
        )
        self.chk_use_vsn.setChecked(_is_bool_hp(_get("use_vsn", {})))

    def to_search_space_fragment(self) -> Dict[str, Any]:
        """Convert dialog state into a partial TUNER_SEARCH_SPACE dict."""
        fragment: Dict[str, Any] = {}

        fragment["memory_size"] = self._parse_int_list(
            self.ed_memory_size.text(),
            [50],
        )
        fragment["batch_size"] = self._parse_int_list(
            self.ed_batch_size.text(),
            [32],
        )
        fragment["epochs"] = self._parse_int_list(
            self.ed_epochs.text(),
            [50],
        )

        if self._scale_combos:
            fragment["scales"] = [list(c) for c in self._scale_combos]
        else:
            fragment["scales"] = []

        if self._attn_combos:
            fragment["attention_levels"] = [list(c) for c in self._attn_combos]
        else:
            fragment["attention_levels"] = []

        if self.chk_use_batch_norm.isChecked():
            fragment["use_batch_norm"] = {"type": "bool"}
        else:
            fragment["use_batch_norm"] = {}

        if self.chk_use_residuals.isChecked():
            fragment["use_residuals"] = {"type": "bool"}
        else:
            fragment["use_residuals"] = {}

        if self.chk_use_vsn.isChecked():
            fragment["use_vsn"] = {"type": "bool"}
        else:
            fragment["use_vsn"] = {}

        return fragment

    # --------------------------------------------------------------
    # Reset / OK
    # --------------------------------------------------------------
    def _on_reset_clicked(self) -> None:
        self.load_from_space(self._defaults, self._defaults)

    def _on_ok(self) -> None:
        try:
            frag = self.to_search_space_fragment()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid values",
                str(exc) or "Check your entries.",
            )
            return

        new_norm = self._norm_fragment(frag)
        patch: Dict[str, Any] = {}

        for k in self._KEYS:
            if self._init_norm.get(k) != new_norm.get(k):
                patch[k] = frag.get(k)

        if patch:
            self._apply_fragment(patch)

        self.accept()

    # --------------------------------------------------------------
    # Normalization (diff detection)
    # --------------------------------------------------------------
    @staticmethod
    def _norm_fragment(frag: Dict[str, Any]) -> Dict[str, Tuple]:
        out: Dict[str, Tuple] = {}
        for k, v in (frag or {}).items():
            out[k] = ModelParamsDialog._norm_value(v)
        return out

    @staticmethod
    def _norm_value(v: Any) -> Tuple:
        if isinstance(v, dict):
            t = str(v.get("type", "")).lower()
            if t in ("bool", "boolean"):
                return ("bool",)
            items = tuple(sorted((str(x), str(v[x])) for x in v))
            return ("dict", items)

        if isinstance(v, (list, tuple)):
            # list of ints OR list-of-lists
            if v and isinstance(v[0], (list, tuple)):
                return ("list2", tuple(tuple(x) for x in v))
            return ("list", tuple(v))

        return ("other", str(v))

    # --------------------------------------------------------------
    # Public entry
    # --------------------------------------------------------------
    @classmethod
    def edit(
        cls,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> bool:
        dlg = cls(store=store, parent=parent)
        return dlg.exec_() == QDialog.Accepted

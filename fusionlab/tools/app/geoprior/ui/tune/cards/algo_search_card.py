# geoprior/ui/tune/cards/algo_search_card.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Algorithm & objective card (Tune tab).

Uses the shared app-wide card factory (make_card)
for consistent styling with Train tab.

- Algorithm is stored in cfg._meta["tuner_algo"]
  for compatibility.
- Objective and trials use schema keys (FieldKey).
- Direction uses UI-only extra key "tune.direction".
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ....config.prior_schema import FieldKey
from ....config.store import GeoConfigStore
from ...icon_utils import try_icon

__all__ = ["TuneAlgoSearchCard"]

MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]

_META_KEY = "_meta"
_META_ALGO = "tuner_algo"

_DIR_KEY = "tune.direction"  # "min" | "max"

_TRIALS_FK = FieldKey("tuner_max_trials")
_OBJ_FK = FieldKey("tuner_objective")

_ALLOWED_ALGOS = ("random", "bayesian", "hyperband")

_OBJ_ITEMS = [
    ("Auto", ""),
    ("Validation loss", "val_loss"),
    ("Validation MAE", "val_mae"),
    ("Validation RMSE", "val_rmse"),
    ("Stability score (PSS)", "val_PSS"),
    ("Time-weighted accuracy (TWA)", "val_TWA"),
]


class TuneAlgoSearchCard(QWidget):
    """
    Algorithm & objective card (expand/collapse).

    Signals
    -------
    changed:
        Emitted after writing to the store.

    edit_toggled:
        Emitted when user expands/collapses the card.
    """

    changed = pyqtSignal()
    edit_toggled = pyqtSignal(bool)

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        make_card: MakeCardFn,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self._make_card = make_card
        self._expanded = False

        self._build_ui()
        self._wire()
        self.refresh_from_store()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._frame, body = self._make_card(
            "Algorithm & objective"
        )
        root.addWidget(self._frame)

        # Header: summary + Edit (always visible)
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)

        self.lbl_sum = QLabel("")
        self.lbl_sum.setObjectName("sumLine")
        self.lbl_sum.setWordWrap(True)

        self.btn_edit = QToolButton(self._frame)
        self.btn_edit.setObjectName("disclosure")
        self.btn_edit.setCursor(Qt.PointingHandCursor)
        self.btn_edit.setAutoRaise(True)
        self.btn_edit.setCheckable(True)
        self.btn_edit.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.btn_edit.setText("Edit")
        self._set_edit_icon(expanded=False)

        hdr.addWidget(self.lbl_sum, 1)
        hdr.addWidget(self.btn_edit, 0)
        body.addLayout(hdr)

        # Details (collapsed by default)
        self.details = QWidget(self._frame)
        self.details.setObjectName("drawer")
        self.details.setVisible(False)

        dlay = QVBoxLayout(self.details)
        dlay.setContentsMargins(0, 6, 0, 0)
        dlay.setSpacing(10)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        self.cmb_algo = QComboBox(self.details)
        self.cmb_algo.addItem("Bayesian", "bayesian")
        self.cmb_algo.addItem("Random", "random")
        self.cmb_algo.addItem("Hyperband", "hyperband")

        self.cmb_obj = QComboBox(self.details)
        for lab, key in _OBJ_ITEMS:
            self.cmb_obj.addItem(lab, key)

        self.cmb_dir = QComboBox(self.details)
        self.cmb_dir.addItem("Minimize", "min")
        self.cmb_dir.addItem("Maximize", "max")

        self.sp_trials = QSpinBox(self.details)
        self.sp_trials.setRange(1, 5000)
        self.sp_trials.setMinimumWidth(110)

        form.addRow("Search algorithm:", self.cmb_algo)
        form.addRow("Objective:", self.cmb_obj)
        form.addRow("Direction:", self.cmb_dir)
        form.addRow("Max trials:", self.sp_trials)

        dlay.addLayout(form)
        body.addWidget(self.details)

    def _wire(self) -> None:
        self.btn_edit.toggled.connect(self._on_toggle)

        self.cmb_algo.currentIndexChanged.connect(
            lambda _=0: self._on_change()
        )
        self.cmb_obj.currentIndexChanged.connect(
            lambda _=0: self._on_change()
        )
        self.cmb_dir.currentIndexChanged.connect(
            lambda _=0: self._on_change()
        )
        self.sp_trials.valueChanged.connect(
            lambda _=0: self._on_change()
        )

    def _set_edit_icon(self, *, expanded: bool) -> None:
        name = "chev_down.svg" if expanded else "chev_right.svg"
        ic = try_icon(name)
        if ic is not None:
            self.btn_edit.setIcon(ic)
        self.btn_edit.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )

    def _on_toggle(self, on: bool) -> None:
        self._expanded = bool(on)
        self.details.setVisible(self._expanded)
        self._set_edit_icon(expanded=self._expanded)
        self.edit_toggled.emit(bool(on))

    # -----------------------------------------------------------------
    # Store reads
    # -----------------------------------------------------------------
    def refresh_from_store(self) -> None:
        algo = self._get_meta_str(_META_ALGO, "bayesian")
        if algo not in _ALLOWED_ALGOS:
            algo = "bayesian"

        obj = self._get_fk_str(_OBJ_FK, "")
        direction = str(self._store.get(_DIR_KEY, "min") or "")
        if direction not in {"min", "max"}:
            direction = "min"

        trials = self._get_fk_int(_TRIALS_FK, 20)

        with QSignalBlocker(self.cmb_algo):
            j = self.cmb_algo.findData(algo)
            self.cmb_algo.setCurrentIndex(0 if j < 0 else j)

        with QSignalBlocker(self.cmb_obj):
            j = self.cmb_obj.findData(obj)
            self.cmb_obj.setCurrentIndex(0 if j < 0 else j)

        with QSignalBlocker(self.cmb_dir):
            j = self.cmb_dir.findData(direction)
            self.cmb_dir.setCurrentIndex(0 if j < 0 else j)

        with QSignalBlocker(self.sp_trials):
            self.sp_trials.setValue(int(trials))

        self._update_summary()

    # -----------------------------------------------------------------
    # Store writes
    # -----------------------------------------------------------------
    def _on_change(self) -> None:
        algo = str(self.cmb_algo.currentData() or "")
        algo = (algo or "bayesian").strip().lower()
        if algo not in _ALLOWED_ALGOS:
            algo = "bayesian"

        obj = str(self.cmb_obj.currentData() or "").strip()

        direction = str(self.cmb_dir.currentData() or "min")
        direction = direction.strip().lower()
        if direction not in {"min", "max"}:
            direction = "min"

        trials = int(self.sp_trials.value())

        with self._store.batch():
            self._patch_meta({_META_ALGO: algo})

            try:
                self._store.set_value_by_key(_OBJ_FK, obj)
            except Exception:
                pass

            self._store.set(_DIR_KEY, direction)

            try:
                self._store.set_value_by_key(_TRIALS_FK, trials)
            except Exception:
                pass

        self._update_summary()
        self.changed.emit()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    def _update_summary(self) -> None:
        algo = self._get_meta_str(_META_ALGO, "bayesian")
        if algo not in _ALLOWED_ALGOS:
            algo = "bayesian"

        obj = self._get_fk_str(_OBJ_FK, "")
        direction = str(self._store.get(_DIR_KEY, "min") or "")
        if direction not in {"min", "max"}:
            direction = "min"

        trials = self._get_fk_int(_TRIALS_FK, 20)

        obj_s = obj if obj else "auto"
        dir_s = "minimize" if direction == "min" else "maximize"

        self.lbl_sum.setText(
            f"{algo} · {obj_s} · {dir_s} · "
            f"{int(trials)} trials"
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _get_meta(self) -> Dict[str, Any]:
        try:
            cur = getattr(self._store.cfg, _META_KEY, None)
        except Exception:
            cur = None
        return dict(cur) if isinstance(cur, dict) else {}

    def _get_meta_str(self, key: str, default: str) -> str:
        meta = self._get_meta()
        v = str(meta.get(key, default) or default)
        return v.strip().lower()

    def _patch_meta(self, patch: Dict[str, Any]) -> None:
        if not patch:
            return
        try:
            self._store.merge_dict_field(
                _META_KEY,
                dict(patch),
                replace=False,
            )
        except Exception:
            pass

    @staticmethod
    def _coerce_int(obj: Any, default: int) -> int:
        try:
            return int(obj)
        except Exception:
            return int(default)

    def _get_fk_str(self, fk: FieldKey, default: str) -> str:
        try:
            v = self._store.get_value(fk, default=default)
        except Exception:
            v = default
        return str(v or "").strip()

    def _get_fk_int(self, fk: FieldKey, default: int) -> int:
        try:
            v = self._store.get_value(fk, default=default)
        except Exception:
            v = default
        return self._coerce_int(v, default)

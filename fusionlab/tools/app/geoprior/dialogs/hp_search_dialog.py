# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
hp_search_dialog.py

Store-driven dialog for tuning "Search algo..." settings.

This dialog stores UI-only preferences in cfg._meta:
- tuner_algo
- tuner_objective
- tuner_direction
- tuner_seed
- tuner_executions_per_trial
- tuner_max_failed_trials
- tuner_max_retries_per_trial
- tuner_hyperband_factor
- tuner_hyperband_iterations

Writes are applied only on OK.
Cancel does not touch the store.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
)

from ..config.store import GeoConfigStore


_ALLOWED_ALGOS = ("random", "bayesian", "hyperband")
_ALLOWED_DIRS = ("min", "max")


class SearchAlgoDialog(QDialog):
    """
    Edit tuning algorithm preferences (UI-only) via the store.

    Notes
    -----
    - Values are stored under cfg._meta.
    - OK writes to store, Cancel does nothing.
    """

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QDialog] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self.setWindowTitle("Search algorithm")

        meta = self._get_meta()

        algo0 = str(meta.get("tuner_algo", "bayesian") or "bayesian")
        obj0 = str(meta.get("tuner_objective", "val_loss") or "val_loss")
        dir0 = str(meta.get("tuner_direction", "min") or "min")

        seed0 = self._coerce_int(meta.get("tuner_seed", 0), 0)
        exe0 = self._coerce_int(
            meta.get("tuner_executions_per_trial", 1),
            1,
        )
        fail0 = self._coerce_int(
            meta.get("tuner_max_failed_trials", 3),
            3,
        )
        retry0 = self._coerce_int(
            meta.get("tuner_max_retries_per_trial", 0),
            0,
        )

        hb_factor0 = self._coerce_int(
            meta.get("tuner_hyperband_factor", 3),
            3,
        )
        hb_iter0 = self._coerce_int(
            meta.get("tuner_hyperband_iterations", 1),
            1,
        )

        self._init = {
            "tuner_algo": algo0,
            "tuner_objective": obj0,
            "tuner_direction": dir0,
            "tuner_seed": seed0,
            "tuner_executions_per_trial": exe0,
            "tuner_max_failed_trials": fail0,
            "tuner_max_retries_per_trial": retry0,
            "tuner_hyperband_factor": hb_factor0,
            "tuner_hyperband_iterations": hb_iter0,
        }

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        card, grid = self._make_card("Tuning search settings")

        self.cb_algo = QComboBox()
        self.cb_algo.addItems(list(_ALLOWED_ALGOS))
        self.cb_algo.setCurrentText(
            algo0 if algo0 in _ALLOWED_ALGOS else "bayesian",
        )
        self.cb_algo.currentTextChanged.connect(
            self._on_algo_changed,
        )

        self.le_objective = QLineEdit()
        self.le_objective.setText(obj0)
        self.le_objective.setPlaceholderText("e.g. val_loss")

        self.cb_dir = QComboBox()
        self.cb_dir.addItems(list(_ALLOWED_DIRS))
        self.cb_dir.setCurrentText(dir0 if dir0 in _ALLOWED_DIRS else "min")

        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 2**31 - 1)
        self.sp_seed.setValue(int(seed0))

        self.sp_exec = QSpinBox()
        self.sp_exec.setRange(1, 100)
        self.sp_exec.setValue(int(exe0))

        self.sp_fail = QSpinBox()
        self.sp_fail.setRange(0, 999)
        self.sp_fail.setValue(int(fail0))

        self.sp_retry = QSpinBox()
        self.sp_retry.setRange(0, 999)
        self.sp_retry.setValue(int(retry0))

        self.sp_hb_factor = QSpinBox()
        self.sp_hb_factor.setRange(2, 20)
        self.sp_hb_factor.setValue(int(hb_factor0))

        self.sp_hb_iters = QSpinBox()
        self.sp_hb_iters.setRange(1, 50)
        self.sp_hb_iters.setValue(int(hb_iter0))

        r = 0
        grid.addWidget(self._lbl("Search algo:"), r, 0)
        grid.addWidget(self.cb_algo, r, 1)
        r += 1

        grid.addWidget(self._lbl("Objective:"), r, 0)
        grid.addWidget(self.le_objective, r, 1)
        r += 1

        grid.addWidget(self._lbl("Direction:"), r, 0)
        grid.addWidget(self.cb_dir, r, 1)
        r += 1

        grid.addWidget(self._lbl("Seed:"), r, 0)
        grid.addWidget(self.sp_seed, r, 1)
        r += 1

        grid.addWidget(self._lbl("Exec / trial:"), r, 0)
        grid.addWidget(self.sp_exec, r, 1)
        r += 1

        grid.addWidget(self._lbl("Max failed:"), r, 0)
        grid.addWidget(self.sp_fail, r, 1)
        r += 1

        grid.addWidget(self._lbl("Max retries:"), r, 0)
        grid.addWidget(self.sp_retry, r, 1)
        r += 1

        grid.addWidget(self._lbl("HB factor:"), r, 0)
        grid.addWidget(self.sp_hb_factor, r, 1)
        r += 1

        grid.addWidget(self._lbl("HB iterations:"), r, 0)
        grid.addWidget(self.sp_hb_iters, r, 1)

        layout.addWidget(card)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        ok_btn = btns.button(QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setDefault(True)

        self._on_algo_changed(self.cb_algo.currentText())

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _make_card(
        self,
        title: str,
    ) -> Tuple[QFrame, QGridLayout]:
        frame = QFrame()
        frame.setObjectName("card")

        outer = QVBoxLayout(frame)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(8)

        ttl = QLabel(title)
        ttl.setObjectName("cardTitle")
        outer.addWidget(ttl)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        outer.addLayout(grid)

        return frame, grid

    def _lbl(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return lab

    def _on_algo_changed(self, algo: str) -> None:
        algo = str(algo or "").strip().lower()
        is_hb = algo == "hyperband"
        self.sp_hb_factor.setEnabled(is_hb)
        self.sp_hb_iters.setEnabled(is_hb)

    # -----------------------------------------------------------------
    # Store helpers
    # -----------------------------------------------------------------
    def _get_meta(self) -> Dict[str, Any]:
        try:
            cur = getattr(self._store.cfg, "_meta", None)
        except Exception:
            cur = None
        if not isinstance(cur, dict):
            return {}
        return dict(cur)

    def _patch_meta(self, patch: Dict[str, Any]) -> None:
        if not patch:
            return
        with self._store.batch():
            self._store.merge_dict_field(
                "_meta",
                dict(patch),
                replace=False,
            )

    # -----------------------------------------------------------------
    # OK
    # -----------------------------------------------------------------
    def _on_accept(self) -> None:
        algo = str(self.cb_algo.currentText() or "bayesian")
        algo = algo.strip().lower()

        if algo not in _ALLOWED_ALGOS:
            QMessageBox.warning(
                self,
                "Invalid algorithm",
                "Select a supported algorithm.",
            )
            return

        obj = str(self.le_objective.text() or "").strip()
        if not obj:
            QMessageBox.warning(
                self,
                "Invalid objective",
                "Objective cannot be empty.",
            )
            return

        direction = str(self.cb_dir.currentText() or "min")
        direction = direction.strip().lower()
        if direction not in _ALLOWED_DIRS:
            QMessageBox.warning(
                self,
                "Invalid direction",
                "Direction must be 'min' or 'max'.",
            )
            return

        seed = int(self.sp_seed.value())
        exe = int(self.sp_exec.value())
        fail = int(self.sp_fail.value())
        retry = int(self.sp_retry.value())

        hb_factor = int(self.sp_hb_factor.value())
        hb_iters = int(self.sp_hb_iters.value())

        patch: Dict[str, Any] = {}

        def ch(k: str, v: Any) -> None:
            if self._init.get(k) != v:
                patch[k] = v

        ch("tuner_algo", algo)
        ch("tuner_objective", obj)
        ch("tuner_direction", direction)
        ch("tuner_seed", seed)
        ch("tuner_executions_per_trial", exe)
        ch("tuner_max_failed_trials", fail)
        ch("tuner_max_retries_per_trial", retry)

        if algo == "hyperband":
            ch("tuner_hyperband_factor", hb_factor)
            ch("tuner_hyperband_iterations", hb_iters)

        if patch:
            self._patch_meta(patch)

        self.accept()

    # -----------------------------------------------------------------
    # Utils
    # -----------------------------------------------------------------
    @staticmethod
    def _coerce_int(obj: Any, default: int) -> int:
        try:
            return int(obj)
        except Exception:
            return int(default)

    # -----------------------------------------------------------------
    # Public entry
    # -----------------------------------------------------------------
    @classmethod
    def edit(
        cls,
        *,
        store: GeoConfigStore,
        parent: Optional[QDialog] = None,
    ) -> bool:
        dlg = cls(store=store, parent=parent)
        return dlg.exec_() == QDialog.Accepted

    @staticmethod
    def snapshot(store: GeoConfigStore) -> Dict[str, Any]:
        keys = (
            "tuner_algo",
            "tuner_objective",
            "tuner_direction",
            "tuner_seed",
            "tuner_executions_per_trial",
            "tuner_max_failed_trials",
            "tuner_max_retries_per_trial",
            "tuner_hyperband_factor",
            "tuner_hyperband_iterations",
        )
        try:
            meta = getattr(store.cfg, "_meta", None)
        except Exception:
            meta = None

        if not isinstance(meta, dict):
            meta = {}

        snap: Dict[str, Any] = {}
        for k in keys:
            snap[k] = meta.get(k)
        return snap

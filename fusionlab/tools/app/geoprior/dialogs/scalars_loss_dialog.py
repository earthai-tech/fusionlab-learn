# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QGridLayout,
    QWidget,
)

from ..config.prior_schema import FieldKey
from ..config.store import GeoConfigStore
from ..config.geoprior_config import default_tuner_search_space


class ScalarsLossDialog(QDialog):
    """
    Store-driven Scalars & loss weights dialog.

    - Reads current tuner_search_space from GeoConfigStore.
    - Writes only on OK.
    - Cancel does not touch the store.
    """

    _KEYS = (
        "mv",
        "kappa",
        "learning_rate",
        "lambda_gw",
        "lambda_cons",
        "lambda_prior",
        "lambda_smooth",
        "lambda_mv",
        "mv_lr_mult",
        "kappa_lr_mult",
    )

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        range_editor_cls: type,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._range_editor_cls = range_editor_cls

        self.setWindowTitle("Scalars & loss weights")
        self.setModal(True)

        main = QVBoxLayout(self)

        info = QLabel(
            "Configure physical scalars and loss weights used "
            "during Stage-2 hyperparameter tuning."
        )
        info.setWordWrap(True)
        main.addWidget(info)

        grid = QGridLayout()
        row = 0

        def add_row(label: str, editor: QWidget) -> None:
            nonlocal row
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(editor, row, 1)
            row += 1

        spin_w = 130

        # Editors (same as your previous layout)
        self.ed_mv = self._range_editor_cls(
            min_allowed=1e-9,
            max_allowed=1e-5,
            decimals=8,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_kappa = self._range_editor_cls(
            min_allowed=0.0,
            max_allowed=10.0,
            decimals=6,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_lr = self._range_editor_cls(
            min_allowed=1e-6,
            max_allowed=1e-3,
            decimals=8,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_lgw = self._range_editor_cls(
            min_allowed=0.0,
            max_allowed=1.0,
            decimals=6,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_lcons = self._range_editor_cls(
            min_allowed=0.0,
            max_allowed=1.0,
            decimals=6,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_lprior = self._range_editor_cls(
            min_allowed=0.0,
            max_allowed=1.0,
            decimals=6,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_lsmooth = self._range_editor_cls(
            min_allowed=0.0,
            max_allowed=1.0,
            decimals=6,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_lmv = self._range_editor_cls(
            min_allowed=0.0,
            max_allowed=1.0,
            decimals=6,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_mv_lr = self._range_editor_cls(
            min_allowed=0.0,
            max_allowed=20.0,
            decimals=3,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_kappa_lr = self._range_editor_cls(
            min_allowed=0.0,
            max_allowed=20.0,
            decimals=3,
            show_sampling=True,
            spin_width=spin_w,
        )

        add_row("Storage coefficient mᵥ:", self.ed_mv)
        add_row("Consolidation factor κ:", self.ed_kappa)
        add_row("Learning rate:", self.ed_lr)
        add_row("λ (GW loss):", self.ed_lgw)
        add_row("λ (consolidation loss):", self.ed_lcons)
        add_row("λ (prior term):", self.ed_lprior)
        add_row("λ (smoothness):", self.ed_lsmooth)
        add_row("λ (mᵥ regularizer):", self.ed_lmv)
        add_row("LR multiplier for mᵥ:", self.ed_mv_lr)
        add_row("LR multiplier for κ:", self.ed_kappa_lr)

        main.addLayout(grid)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self.btn_reset = QPushButton("Reset to defaults")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_ok = QPushButton("OK")

        btn_row.addWidget(self.btn_reset)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_ok)
        main.addLayout(btn_row)

        self.btn_ok.clicked.connect(self._on_ok)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_reset.clicked.connect(self._on_reset_clicked)

        # load current store values
        self._defaults = self._get_defaults()
        self._space0 = self._get_space_merged(self._defaults)

        self.load_from_space(self._space0, self._defaults)
        self._init_norm = self._norm_fragment(
            self.to_search_space_fragment()
        )

    # ------------------------------------------------------------------
    # Store helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Reset / load / save
    # ------------------------------------------------------------------
    def _on_reset_clicked(self) -> None:
        self.load_from_space(self._defaults, self._defaults)

    def load_from_space(
        self,
        space: Dict[str, Any],
        defaults: Dict[str, Any],
    ) -> None:
        def _get(name: str) -> Any:
            return space.get(name, defaults.get(name))

        self.ed_mv.from_search_space_value(
            _get("mv"), defaults["mv"]
        )
        self.ed_kappa.from_search_space_value(
            _get("kappa"), defaults["kappa"]
        )
        self.ed_lr.from_search_space_value(
            _get("learning_rate"), defaults["learning_rate"]
        )
        self.ed_lgw.from_search_space_value(
            _get("lambda_gw"), defaults["lambda_gw"]
        )
        self.ed_lcons.from_search_space_value(
            _get("lambda_cons"), defaults["lambda_cons"]
        )
        self.ed_lprior.from_search_space_value(
            _get("lambda_prior"), defaults["lambda_prior"]
        )
        self.ed_lsmooth.from_search_space_value(
            _get("lambda_smooth"), defaults["lambda_smooth"]
        )
        self.ed_lmv.from_search_space_value(
            _get("lambda_mv"), defaults["lambda_mv"]
        )
        self.ed_mv_lr.from_search_space_value(
            _get("mv_lr_mult"), defaults["mv_lr_mult"]
        )
        self.ed_kappa_lr.from_search_space_value(
            _get("kappa_lr_mult"), defaults["kappa_lr_mult"]
        )

    def to_search_space_fragment(self) -> Dict[str, Any]:
        return {
            "mv": self.ed_mv.to_search_space_value(),
            "kappa": self.ed_kappa.to_search_space_value(),
            "learning_rate": self.ed_lr.to_search_space_value(),
            "lambda_gw": self.ed_lgw.to_search_space_value(),
            "lambda_cons": self.ed_lcons.to_search_space_value(),
            "lambda_prior": self.ed_lprior.to_search_space_value(),
            "lambda_smooth": self.ed_lsmooth.to_search_space_value(),
            "lambda_mv": self.ed_lmv.to_search_space_value(),
            "mv_lr_mult": self.ed_mv_lr.to_search_space_value(),
            "kappa_lr_mult": self.ed_kappa_lr.to_search_space_value(),
        }

    # ------------------------------------------------------------------
    # OK
    # ------------------------------------------------------------------
    def _on_ok(self) -> None:
        try:
            frag = self.to_search_space_fragment()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid values",
                str(exc) or "Check your ranges.",
            )
            return

        # Only patch changed keys
        new_norm = self._norm_fragment(frag)
        patch: Dict[str, Any] = {}

        for k in self._KEYS:
            if self._init_norm.get(k) != new_norm.get(k):
                patch[k] = frag.get(k)

        if patch:
            self._apply_fragment(patch)

        self.accept()

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    @staticmethod
    def _norm_fragment(frag: Dict[str, Any]) -> Dict[str, Tuple]:
        out: Dict[str, Tuple] = {}
        for k, v in (frag or {}).items():
            out[k] = ScalarsLossDialog._norm_value(v)
        return out

    @staticmethod
    def _norm_value(v: Any) -> Tuple:
        if isinstance(v, dict):
            t = str(v.get("type", "")).lower()
            if t in ("float", "int", "range"):
                return (
                    t,
                    float(v.get("min_value", v.get("min", 0.0))),
                    float(v.get("max_value", v.get("max", 0.0))),
                    float(v.get("step", 0.0) or 0.0),
                    str(v.get("sampling", "") or ""),
                )
            if t == "choice":
                vals = tuple(v.get("values", []) or [])
                return ("choice", vals)
            if t in ("bool", "boolean"):
                return ("bool",)
            items = tuple(sorted((str(x), str(v[x])) for x in v))
            return ("dict", items)

        if isinstance(v, (list, tuple)):
            return ("list", tuple(v))

        return ("other", str(v))

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    @classmethod
    def edit(
        cls,
        *,
        store: GeoConfigStore,
        range_editor_cls: type,
        parent: Optional[QWidget] = None,
    ) -> bool:
        dlg = cls(
            store=store,
            range_editor_cls=range_editor_cls,
            parent=parent,
        )
        return dlg.exec_() == QDialog.Accepted

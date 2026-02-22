# geoprior/ui/tune/cards/search_space_card.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Search space card (Tune tab).

- Uses the shared app card factory (make_card) for
  consistent Train-like styling.
- Expands inline (same card) on Edit.
- Provides a compact "essentials" HP editor for PINN
  tuning: core architecture + dropout + key λ weights.

Store contract (best-effort)
----------------------------
- Search space dict lives in FieldKey("tuner_search_space").
- Keys we touch (if present / created):
  - embed_dim, hidden_units, lstm_units, attention_units,
    num_heads, vsn_units
  - dropout_rate
  - lambda_cons, lambda_gw, lambda_prior,
    lambda_mv, lambda_q

Notes
-----
- This card writes to the store on edit-finish (safe).
- The reset button only resets THIS card's fields to the
  card defaults (does not perform global "reset tune").
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy
)

from ....config.prior_schema import FieldKey
from ....config.store import GeoConfigStore
from ....utils.components import RangeListEditor
from ...icon_utils import try_icon

__all__ = ["TuneSearchSpaceCard"]

MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]

_SPACE_FK = FieldKey("tuner_search_space")


class TuneSearchSpaceCard(QWidget):
    """
    Search space (quick essentials).

    Signals
    -------
    edit_toggled(bool)
        Emitted when the card expands/collapses.

    reset_requested()
        Emitted when user clicks "Reset essentials".

    changed()
        Emitted after writing to the store.
    """

    edit_toggled = pyqtSignal(bool)
    reset_requested = pyqtSignal()
    changed = pyqtSignal()

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
    # UI helpers
    # -----------------------------------------------------------------
    def _row(self, label: str, editor: QWidget) -> QWidget:
        w = QWidget(self.details)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        lab = QLabel(label, w)
        lab.setMinimumWidth(80)

        lay.addWidget(lab, 0)
        lay.addWidget(editor, 1)
        return w

    def _mk_lambda(self) -> RangeListEditor:
        ed = RangeListEditor(
            self.details,
            min_allowed=0.0,
            max_allowed=10.0,
            decimals=4,
            show_sampling=False,
        )
        try:
            ed.set_defaults(0.0, 1.0, sampling=None)
        except Exception:
            pass
        return ed

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._frame, body = self._make_card("Search space")
        self._frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        root.addWidget(self._frame)

        # Summary + Edit on the SAME row (consistent with other cards)
        sum_row = QWidget(self._frame)
        sum_l = QHBoxLayout(sum_row)
        sum_l.setContentsMargins(0, 0, 0, 0)
        sum_l.setSpacing(8)

        self.lbl_sum = QLabel("—", self._frame)
        self.lbl_sum.setObjectName("sumLine")
        self.lbl_sum.setWordWrap(True)

        self.btn_edit = QToolButton(self._frame)
        self.btn_edit.setObjectName("disclosure")
        self.btn_edit.setCursor(Qt.PointingHandCursor)
        self.btn_edit.setAutoRaise(True)
        self.btn_edit.setCheckable(True)
        self.btn_edit.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_edit.setText("Edit")
        self._set_edit_icon(expanded=False)

        sum_l.addWidget(self.lbl_sum, 1)
        sum_l.addWidget(self.btn_edit, 0)

        body.addWidget(sum_row)

        # Help line
        self.lbl_help = QLabel(
            "Quick essentials the tuner may explore. "
            "For full builders, use Advanced."
        )
        self.lbl_help.setObjectName("helpText")
        self.lbl_help.setWordWrap(True)
        body.addWidget(self.lbl_help)

        # Drawer (collapsed)
        self.details = QWidget(self._frame)
        self.details.setObjectName("drawer")
        self.details.setVisible(False)

        d = QVBoxLayout(self.details)
        d.setContentsMargins(0, 6, 0, 0)
        d.setSpacing(10)

        # -------------------------
        # Architecture essentials
        # -------------------------
        self.hp_embed_dim = QLineEdit(self.details)
        self.hp_hidden_units = QLineEdit(self.details)
        self.hp_lstm_units = QLineEdit(self.details)
        self.hp_attention_units = QLineEdit(self.details)
        self.hp_num_heads = QLineEdit(self.details)
        self.hp_vsn_units = QLineEdit(self.details)

        self.hp_embed_dim.setPlaceholderText("32, 48, 64")
        self.hp_hidden_units.setPlaceholderText("64, 96, 128")
        self.hp_lstm_units.setPlaceholderText("64, 96")
        self.hp_attention_units.setPlaceholderText("32, 48")
        self.hp_num_heads.setPlaceholderText("2, 4")
        self.hp_vsn_units.setPlaceholderText("24, 32, 40")

        self.hp_dropout = RangeListEditor(
            self.details,
            min_allowed=0.0,
            max_allowed=1.0,
            decimals=4,
            show_sampling=False,
        )
        try:
            self.hp_dropout.set_defaults(0.05, 0.20, sampling=None)
        except Exception:
            pass

        d.addWidget(self._row("Embed dim:", self.hp_embed_dim))
        d.addWidget(self._row("Hidden units:", self.hp_hidden_units))
        d.addWidget(self._row("LSTM units:", self.hp_lstm_units))
        d.addWidget(self._row("Attention units:", self.hp_attention_units))
        d.addWidget(self._row("Attention heads:", self.hp_num_heads))
        d.addWidget(self._row("VSN units:", self.hp_vsn_units))
        d.addWidget(self._row("Dropout:", self.hp_dropout))

        # -------------------------
        # Minimal λ weights (RangeListEditor, consistent)
        # -------------------------
        self.hp_lc = self._mk_lambda()  # λc
        self.hp_lg = self._mk_lambda()  # λgw
        self.hp_lp = self._mk_lambda()  # λp
        self.hp_lm = self._mk_lambda()  # λmv
        self.hp_lq = self._mk_lambda()  # λq

        d.addWidget(self._row("λc (cons):", self.hp_lc))
        d.addWidget(self._row("λgw:", self.hp_lg))
        d.addWidget(self._row("λp (prior):", self.hp_lp))
        d.addWidget(self._row("λmv:", self.hp_lm))
        d.addWidget(self._row("λq:", self.hp_lq))

        # Reset (card-local)
        self.btn_reset = QPushButton("Reset essentials", self.details)
        self.btn_reset.setObjectName("miniAction")
        ic = try_icon("reset.svg")
        if ic is not None:
            self.btn_reset.setIcon(ic)
        self.btn_reset.setMinimumHeight(28)
        d.addWidget(self.btn_reset, 0, Qt.AlignRight)

        body.addWidget(self.details)

    def _set_edit_icon(self, *, expanded: bool) -> None:
        name = "chev_down.svg" if expanded else "chev_right.svg"
        ic = try_icon(name)
        if ic is not None:
            self.btn_edit.setIcon(ic)
        self.btn_edit.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )

    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.btn_edit.toggled.connect(self._on_toggle)
        self.btn_reset.clicked.connect(self._on_reset_essentials)

        for ed in (
            self.hp_embed_dim,
            self.hp_hidden_units,
            self.hp_lstm_units,
            self.hp_attention_units,
            self.hp_num_heads,
            self.hp_vsn_units,
        ):
            ed.editingFinished.connect(self._commit)

        for ed in (
            self.hp_dropout,
            self.hp_lc,
            self.hp_lg,
            self.hp_lp,
            self.hp_lm,
            self.hp_lq,
        ):
            self._connect_range_editor(ed)

    def _connect_range_editor(self, editor: QWidget) -> None:
        for sig_name in (
            "changed",
            "valueChanged",
            "sig_changed",
            "signalChanged",
        ):
            sig = getattr(editor, sig_name, None)
            if sig is None:
                continue
            if hasattr(sig, "connect"):
                try:
                    sig.connect(self._commit)
                    return
                except Exception:
                    continue

    # -----------------------------------------------------------------
    # Expand / collapse
    # -----------------------------------------------------------------
    def _on_toggle(self, on: bool) -> None:
        self._expanded = bool(on)
        self.details.setVisible(self._expanded)
        self._set_edit_icon(expanded=self._expanded)
        self.edit_toggled.emit(bool(on))

    def set_expanded(self, on: bool) -> None:
        with QSignalBlocker(self.btn_edit):
            self.btn_edit.setChecked(bool(on))
        self._on_toggle(bool(on))

    def is_expanded(self) -> bool:
        return bool(self._expanded)

    # -----------------------------------------------------------------
    # Store I/O
    # -----------------------------------------------------------------
    def refresh_from_store(self) -> None:
        space = self._get_space()

        with QSignalBlocker(self.hp_embed_dim):
            self.hp_embed_dim.setText(
                self._ints_to_csv(space.get("embed_dim", []))
            )
        with QSignalBlocker(self.hp_hidden_units):
            self.hp_hidden_units.setText(
                self._ints_to_csv(space.get("hidden_units", []))
            )
        with QSignalBlocker(self.hp_lstm_units):
            self.hp_lstm_units.setText(
                self._ints_to_csv(space.get("lstm_units", []))
            )
        with QSignalBlocker(self.hp_attention_units):
            self.hp_attention_units.setText(
                self._ints_to_csv(space.get("attention_units", []))
            )
        with QSignalBlocker(self.hp_num_heads):
            self.hp_num_heads.setText(
                self._ints_to_csv(space.get("num_heads", []))
            )
        with QSignalBlocker(self.hp_vsn_units):
            self.hp_vsn_units.setText(
                self._ints_to_csv(space.get("vsn_units", []))
            )

        try:
            with QSignalBlocker(self.hp_dropout):
                self.hp_dropout.from_search_space_value(
                    space.get("dropout_rate"),
                    space.get("dropout_rate"),
                )
        except Exception:
            pass

        for key, editor in (
            ("lambda_cons", self.hp_lc),
            ("lambda_gw", self.hp_lg),
            ("lambda_prior", self.hp_lp),
            ("lambda_mv", self.hp_lm),
            ("lambda_q", self.hp_lq),
        ):
            try:
                with QSignalBlocker(editor):
                    editor.from_search_space_value(
                        space.get(key),
                        space.get(key),
                    )
            except Exception:
                pass

        self._update_summary(space)

    def _commit(self) -> None:
        cur = self._get_space()
        patch: Dict[str, Any] = dict(cur)

        patch["embed_dim"] = self._csv_to_ints(self.hp_embed_dim.text())
        patch["hidden_units"] = self._csv_to_ints(
            self.hp_hidden_units.text()
        )
        patch["lstm_units"] = self._csv_to_ints(self.hp_lstm_units.text())
        patch["attention_units"] = self._csv_to_ints(
            self.hp_attention_units.text()
        )
        patch["num_heads"] = self._csv_to_ints(self.hp_num_heads.text())
        patch["vsn_units"] = self._csv_to_ints(self.hp_vsn_units.text())

        patch["dropout_rate"] = self._range_to_space_value(
            self.hp_dropout,
            fallback=patch.get("dropout_rate"),
        )

        patch["lambda_cons"] = self._range_to_space_value(
            self.hp_lc,
            fallback=patch.get("lambda_cons"),
        )
        patch["lambda_gw"] = self._range_to_space_value(
            self.hp_lg,
            fallback=patch.get("lambda_gw"),
        )
        patch["lambda_prior"] = self._range_to_space_value(
            self.hp_lp,
            fallback=patch.get("lambda_prior"),
        )
        patch["lambda_mv"] = self._range_to_space_value(
            self.hp_lm,
            fallback=patch.get("lambda_mv"),
        )
        patch["lambda_q"] = self._range_to_space_value(
            self.hp_lq,
            fallback=patch.get("lambda_q"),
        )

        try:
            self._store.set_value_by_key(_SPACE_FK, patch)
        except Exception:
            return

        self._update_summary(patch)
        self.changed.emit()

    def _get_space(self) -> Dict[str, Any]:
        try:
            obj = self._store.get_value(_SPACE_FK, default={})
        except Exception:
            obj = {}
        return dict(obj) if isinstance(obj, dict) else {}

    # -----------------------------------------------------------------
    # Reset (card-local)
    # -----------------------------------------------------------------
    def _on_reset_essentials(self) -> None:
        with QSignalBlocker(self.hp_embed_dim):
            self.hp_embed_dim.setText("32, 48, 64")
        with QSignalBlocker(self.hp_hidden_units):
            self.hp_hidden_units.setText("64, 96, 128")
        with QSignalBlocker(self.hp_lstm_units):
            self.hp_lstm_units.setText("64, 96")
        with QSignalBlocker(self.hp_attention_units):
            self.hp_attention_units.setText("32, 48")
        with QSignalBlocker(self.hp_num_heads):
            self.hp_num_heads.setText("2, 4")
        with QSignalBlocker(self.hp_vsn_units):
            self.hp_vsn_units.setText("24, 32, 40")

        try:
            with QSignalBlocker(self.hp_dropout):
                self.hp_dropout.set_defaults(0.05, 0.20, sampling=None)
        except Exception:
            pass

        # Reset λ to defaults (fixed values)
        defaults = {
            "lc": 1.0,     # lambda_cons
            "lg": 0.1,     # lambda_gw
            "lp": 0.5,     # lambda_prior
            "lm": 0.01,    # lambda_mv
            "lq": 1e-5,    # lambda_q (NOT 0)
        }
        
        pairs = [
            ("lc", getattr(self, "hp_lc", None)),
            ("lg", getattr(self, "hp_lg", None)),
            ("lp", getattr(self, "hp_lp", None)),
            ("lm", getattr(self, "hp_lm", None)),
            ("lq", getattr(self, "hp_lq", None)),
        ]
        
        for key, ed in pairs:
            if ed is None:
                continue
            v = float(defaults[key])
            try:
                with QSignalBlocker(ed):
                    # fixed value => min=max=default
                    ed.set_defaults(v, v, sampling=None)
            except Exception:
                pass

        self.reset_requested.emit()
        self._commit()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    def _update_summary(self, space: Dict[str, Any]) -> None:
        def n(key: str) -> int:
            v = space.get(key)
            return len(v) if isinstance(v, (list, tuple)) else 0

        self.lbl_sum.setText(
            f"embed={n('embed_dim')} hidden={n('hidden_units')} "
            f"lstm={n('lstm_units')} attn={n('attention_units')} "
            f"heads={n('num_heads')} vsn={n('vsn_units')} · "
            "dropout · 5λ"
        )

    # -----------------------------------------------------------------
    # Small helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _ints_to_csv(vals: Any) -> str:
        try:
            return ", ".join(str(int(v)) for v in vals)
        except Exception:
            return ""

    @staticmethod
    def _csv_to_ints(text: str) -> List[int]:
        s = str(text or "").strip()
        if not s:
            return []
        out: List[int] = []
        for part in s.split(","):
            p = part.strip()
            if not p:
                continue
            try:
                out.append(int(float(p)))
            except Exception:
                continue
        return out

    @staticmethod
    def _range_to_space_value(editor: Any, fallback: Any = None) -> Any:
        for nm in ("to_search_space_value", "search_space_value", "value"):
            fn = getattr(editor, nm, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    continue
        return fallback

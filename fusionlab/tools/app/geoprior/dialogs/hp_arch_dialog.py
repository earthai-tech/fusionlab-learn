# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
hp_arch_dialog.py

Store-driven dialog for "More architecture HP".

Edits the architecture-related entries inside:
    GeoPriorConfig.tuner_search_space

Writes are applied only on OK.
Cancel does not touch the store.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from ..config.store import GeoConfigStore
from ..config.prior_schema import FieldKey
from ..config.geoprior_config import default_tuner_search_space


class _FloatRangeListEditor(QWidget):
    """
    Small self-contained "Range / List" editor for float HP specs.

    Supported formats:
    - Range dict:
        {"type":"float","min_value":0.05,"max_value":0.25,"step":0.05}
    - List:
        [0.05, 0.10, 0.15]
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Range", "List"])

        # Range widgets
        self.sp_min = QDoubleSpinBox()
        self.sp_min.setDecimals(6)
        self.sp_min.setRange(-1e9, 1e9)

        self.sp_max = QDoubleSpinBox()
        self.sp_max.setDecimals(6)
        self.sp_max.setRange(-1e9, 1e9)

        self.sp_step = QDoubleSpinBox()
        self.sp_step.setDecimals(6)
        self.sp_step.setRange(0.0, 1e9)

        # List widget
        self.le_list = QLineEdit()
        self.le_list.setPlaceholderText("e.g. 0.05, 0.10, 0.15")

        lay.addWidget(self.cb_mode)
        lay.addWidget(QLabel("Min:"))
        lay.addWidget(self.sp_min)
        lay.addWidget(QLabel("Max:"))
        lay.addWidget(self.sp_max)
        lay.addWidget(QLabel("Step:"))
        lay.addWidget(self.sp_step)
        lay.addWidget(self.le_list, 1)

        self.cb_mode.currentTextChanged.connect(self._sync_mode)
        self._sync_mode(self.cb_mode.currentText())

    def _sync_mode(self, mode: str) -> None:
        is_range = str(mode).strip().lower() == "range"
        self.sp_min.setVisible(is_range)
        self.sp_max.setVisible(is_range)
        self.sp_step.setVisible(is_range)
        # labels are the 3 QLabel we inserted after combobox
        # indices: 1,3,5 in layout order, but simplest is:
        for w in self.findChildren(QLabel):
            txt = (w.text() or "").strip().lower()
            if txt in {"min:", "max:", "step:"}:
                w.setVisible(is_range)
        self.le_list.setVisible(not is_range)

    def from_search_space_value(
        self,
        value: Any,
        default: Any,
    ) -> None:
        v = value if value is not None else default

        if isinstance(v, dict):
            # range dict
            self.cb_mode.setCurrentText("Range")
            self.sp_min.setValue(float(v.get("min_value", 0.0)))
            self.sp_max.setValue(float(v.get("max_value", 0.0)))
            self.sp_step.setValue(float(v.get("step", 0.0)))
            self.le_list.setText("")
            return

        if isinstance(v, (list, tuple)):
            self.cb_mode.setCurrentText("List")
            parts = []
            for x in v:
                try:
                    parts.append(self._fmt_float(float(x)))
                except Exception:
                    continue
            self.le_list.setText(", ".join(parts))
            # keep range boxes reasonable defaults
            d = default if isinstance(default, dict) else {}
            self.sp_min.setValue(float(d.get("min_value", 0.0)))
            self.sp_max.setValue(float(d.get("max_value", 0.0)))
            self.sp_step.setValue(float(d.get("step", 0.0)))
            return

        # fallback to default
        self.from_search_space_value(default, default)

    def to_search_space_value(self) -> Any:
        mode = str(self.cb_mode.currentText() or "Range").lower()
        if mode == "list":
            xs = self._parse_float_list(self.le_list.text())
            return xs

        mn = float(self.sp_min.value())
        mx = float(self.sp_max.value())
        st = float(self.sp_step.value())
        return {
            "type": "float",
            "min_value": mn,
            "max_value": mx,
            "step": st,
        }

    @staticmethod
    def _fmt_float(x: float) -> str:
        s = f"{float(x):.6f}"
        s = s.rstrip("0").rstrip(".")
        return s or "0"

    @staticmethod
    def _parse_float_list(text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return []
        out: List[float] = []
        for part in text.replace(";", ",").split(","):
            s = part.strip()
            if not s:
                continue
            out.append(float(s))
        return out

    @staticmethod
    def norm_value(v: Any) -> Tuple:
        if isinstance(v, dict):
            return (
                "range",
                float(v.get("min_value", 0.0)),
                float(v.get("max_value", 0.0)),
                float(v.get("step", 0.0)),
            )
        if isinstance(v, (list, tuple)):
            vals = []
            for x in v:
                try:
                    vals.append(float(x))
                except Exception:
                    continue
            return ("list", tuple(vals))
        return ("other", repr(v))


class ArchHPDialog(QDialog):
    """
    Edit architecture hyperparameters inside tuner_search_space.

    Notes
    -----
    - Writes are applied only on OK.
    - Cancel does not touch the store.
    """

    _ARCH_KEYS = (
        "embed_dim",
        "hidden_units",
        "lstm_units",
        "attention_units",
        "num_heads",
        "vsn_units",
        "dropout_rate",
        "max_window_size",
        "memory_size",
        "attention_levels",
        "scales",
    )

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QDialog] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self.setWindowTitle("Architecture hyperparameters")

        defaults = default_tuner_search_space()
        space = self._coerce_space(
            self._get_space(defaults),
            defaults,
        )

        # snapshot initial values (normalized)
        self._initial: Dict[str, Any] = {}
        for k in self._ARCH_KEYS:
            self._initial[k] = space.get(k, defaults.get(k))

        self._initial_dropout_norm = _FloatRangeListEditor.norm_value(
            self._initial.get("dropout_rate"),
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # ==================================================
        # Card: Core architecture lists
        # ==================================================
        card1, grid1 = self._make_card("Core architecture search")

        self.le_embed = QLineEdit()
        self.le_hidden = QLineEdit()
        self.le_lstm = QLineEdit()
        self.le_att_units = QLineEdit()
        self.le_heads = QLineEdit()
        self.le_vsn = QLineEdit()

        self.le_embed.setPlaceholderText("e.g. 32, 48, 64")
        self.le_hidden.setPlaceholderText("e.g. 64, 96, 128")
        self.le_lstm.setPlaceholderText("e.g. 64, 96")
        self.le_att_units.setPlaceholderText("e.g. 32, 48")
        self.le_heads.setPlaceholderText("e.g. 2, 4")
        self.le_vsn.setPlaceholderText("e.g. 24, 32, 40")

        self.le_embed.setText(self._fmt_int_list(
            self._coerce_int_list(space.get("embed_dim"), defaults["embed_dim"])
        ))
        self.le_hidden.setText(self._fmt_int_list(
            self._coerce_int_list(
                space.get("hidden_units"),
                defaults["hidden_units"],
            )
        ))
        self.le_lstm.setText(self._fmt_int_list(
            self._coerce_int_list(
                space.get("lstm_units"),
                defaults["lstm_units"],
            )
        ))
        self.le_att_units.setText(self._fmt_int_list(
            self._coerce_int_list(
                space.get("attention_units"),
                defaults["attention_units"],
            )
        ))
        self.le_heads.setText(self._fmt_int_list(
            self._coerce_int_list(space.get("num_heads"), defaults["num_heads"])
        ))
        self.le_vsn.setText(self._fmt_int_list(
            self._coerce_int_list(space.get("vsn_units"), defaults["vsn_units"])
        ))

        r = 0
        grid1.addWidget(self._lbl("Embedding dim:"), r, 0)
        grid1.addWidget(self.le_embed, r, 1)
        r += 1

        grid1.addWidget(self._lbl("Hidden units:"), r, 0)
        grid1.addWidget(self.le_hidden, r, 1)
        r += 1

        grid1.addWidget(self._lbl("LSTM units:"), r, 0)
        grid1.addWidget(self.le_lstm, r, 1)
        r += 1

        grid1.addWidget(self._lbl("Attention units:"), r, 0)
        grid1.addWidget(self.le_att_units, r, 1)
        r += 1

        grid1.addWidget(self._lbl("Attention heads:"), r, 0)
        grid1.addWidget(self.le_heads, r, 1)
        r += 1

        grid1.addWidget(self._lbl("VSN units:"), r, 0)
        grid1.addWidget(self.le_vsn, r, 1)

        layout.addWidget(card1)

        # ==================================================
        # Card: Dropout editor
        # ==================================================
        card2, grid2 = self._make_card("Dropout search")
        self.dropout_editor = _FloatRangeListEditor()
        self.dropout_editor.from_search_space_value(
            space.get("dropout_rate"),
            defaults.get("dropout_rate"),
        )
        grid2.addWidget(self._lbl("Dropout rate:"), 0, 0)
        grid2.addWidget(self.dropout_editor, 0, 1)
        layout.addWidget(card2)

        # ==================================================
        # Card: Sequence / memory + attention stack variants
        # ==================================================
        card3, grid3 = self._make_card("Sequence + attention variants")

        self.le_max_window = QLineEdit()
        self.le_mem_size = QLineEdit()
        self.le_att_levels = QLineEdit()
        self.le_scales = QLineEdit()

        self.le_max_window.setPlaceholderText("e.g. 8, 10, 12")
        self.le_mem_size.setPlaceholderText("e.g. 50, 100")
        self.le_att_levels.setPlaceholderText(
            "e.g. cross, memory, hierarchical; cross, hierarchical"
        )
        self.le_scales.setPlaceholderText("e.g. 1, 2; 1, 2, 4")

        self.le_max_window.setText(self._fmt_int_list(
            self._coerce_int_list(
                space.get("max_window_size"),
                defaults.get("max_window_size", [8, 10, 12]),
            )
        ))
        self.le_mem_size.setText(self._fmt_int_list(
            self._coerce_int_list(
                space.get("memory_size"),
                defaults.get("memory_size", [50, 100]),
            )
        ))

        att0 = self._coerce_groups(
            space.get("attention_levels"),
            defaults.get("attention_levels", [["cross", "memory"]]),
        )
        sc0 = self._coerce_int_groups(
            space.get("scales"),
            defaults.get("scales", [[1, 2]]),
        )

        self.le_att_levels.setText(self._fmt_groups(att0))
        self.le_scales.setText(self._fmt_int_groups(sc0))

        rr = 0
        grid3.addWidget(self._lbl("Max window size:"), rr, 0)
        grid3.addWidget(self.le_max_window, rr, 1)
        rr += 1

        grid3.addWidget(self._lbl("Memory size:"), rr, 0)
        grid3.addWidget(self.le_mem_size, rr, 1)
        rr += 1

        grid3.addWidget(self._lbl("Attention levels:"), rr, 0)
        grid3.addWidget(self.le_att_levels, rr, 1)
        rr += 1

        grid3.addWidget(self._lbl("Scales:"), rr, 0)
        grid3.addWidget(self.le_scales, rr, 1)

        layout.addWidget(card3)

        # ----------------------------
        # Buttons
        # ----------------------------
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        ok_btn = btns.button(QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setDefault(True)

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
        keys = list(ArchHPDialog._ARCH_KEYS)
        snap: Dict[str, Any] = {}
        try:
            space = store.get_value(FieldKey("tuner_search_space"))
        except Exception:
            space = None
        if not isinstance(space, dict):
            space = {}
        for k in keys:
            snap[k] = space.get(k)
        return snap

    # -----------------------------------------------------------------
    # UI helpers
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

    # -----------------------------------------------------------------
    # Store helpers
    # -----------------------------------------------------------------
    def _get_space(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        try:
            cur = self._store.get_value(FieldKey("tuner_search_space"))
        except Exception:
            cur = None
        if not isinstance(cur, dict):
            return dict(defaults)
        merged = dict(defaults)
        merged.update(dict(cur))
        return merged

    @staticmethod
    def _coerce_space(
        space: Any,
        defaults: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(space, dict):
            return dict(defaults)
        merged = dict(defaults)
        merged.update(dict(space))
        return merged

    def _apply_fragment(self, frag: Dict[str, Any]) -> None:
        if not frag:
            return
        with self._store.batch():
            # safest: replace dict object via merge_dict_field
            self._store.merge_dict_field(
                "tuner_search_space",
                dict(frag),
                replace=False,
            )

    # -----------------------------------------------------------------
    # Parsing / formatting
    # -----------------------------------------------------------------
    @staticmethod
    def _fmt_int_list(vals: Sequence[int]) -> str:
        return ", ".join(str(int(x)) for x in (vals or []))

    @staticmethod
    def _parse_int_list_strict(text: str) -> List[int]:
        text = (text or "").strip()
        if not text:
            return []
        out: List[int] = []
        for tok in text.replace(";", ",").split(","):
            s = tok.strip()
            if not s:
                continue
            try:
                out.append(int(s))
            except Exception as exc:
                raise ValueError(f"Invalid int: {s!r}") from exc
        return out

    @staticmethod
    def _coerce_int_list(obj: Any, fallback: List[int]) -> List[int]:
        if isinstance(obj, int):
            return [int(obj)]
        if isinstance(obj, (list, tuple)):
            out: List[int] = []
            for x in obj:
                try:
                    out.append(int(x))
                except Exception:
                    continue
            return out or list(fallback)
        return list(fallback)

    @staticmethod
    def _parse_groups(text: str) -> List[List[str]]:
        """
        Parse: "a, b; c, d" -> [["a","b"],["c","d"]]
        """
        text = (text or "").strip()
        if not text:
            return []
        groups: List[List[str]] = []
        for chunk in text.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            items: List[str] = []
            for tok in chunk.split(","):
                t = tok.strip()
                if t:
                    items.append(t)
            if items:
                groups.append(items)
        return groups

    @staticmethod
    def _coerce_groups(obj: Any, fallback: Any) -> List[List[str]]:
        fb = obj if obj is not None else fallback

        # list[str] -> one group
        if isinstance(fb, (list, tuple)) and fb and all(
            isinstance(x, str) for x in fb
        ):
            return [list(fb)]  # type: ignore[arg-type]

        # list[list[str]]
        if isinstance(fb, (list, tuple)):
            groups: List[List[str]] = []
            for g in fb:
                if isinstance(g, (list, tuple)):
                    items = []
                    for x in g:
                        if isinstance(x, str) and x.strip():
                            items.append(x.strip())
                    if items:
                        groups.append(items)
            return groups or [["cross", "memory"]]

        return [["cross", "memory"]]

    @staticmethod
    def _fmt_groups(groups: List[List[str]]) -> str:
        parts: List[str] = []
        for g in groups or []:
            parts.append(", ".join(str(x) for x in g))
        return "; ".join(parts)

    @staticmethod
    def _parse_int_groups(text: str) -> List[List[int]]:
        """
        Parse: "1, 2; 1, 2, 4" -> [[1,2],[1,2,4]]
        """
        text = (text or "").strip()
        if not text:
            return []
        groups: List[List[int]] = []
        for chunk in text.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            vals: List[int] = []
            for tok in chunk.split(","):
                s = tok.strip()
                if not s:
                    continue
                try:
                    vals.append(int(s))
                except Exception as exc:
                    raise ValueError(f"Invalid int: {s!r}") from exc
            if vals:
                groups.append(vals)
        return groups

    @staticmethod
    def _coerce_int_groups(obj: Any, fallback: Any) -> List[List[int]]:
        fb = obj if obj is not None else fallback

        # list[int] -> one group
        if isinstance(fb, (list, tuple)) and fb and all(
            isinstance(x, int) for x in fb
        ):
            return [list(fb)]  # type: ignore[arg-type]

        # list[list[int]]
        if isinstance(fb, (list, tuple)):
            groups: List[List[int]] = []
            for g in fb:
                if isinstance(g, (list, tuple)):
                    vals = []
                    for x in g:
                        try:
                            vals.append(int(x))
                        except Exception:
                            continue
                    if vals:
                        groups.append(vals)
            return groups or [[1, 2]]

        return [[1, 2]]

    @staticmethod
    def _fmt_int_groups(groups: List[List[int]]) -> str:
        parts: List[str] = []
        for g in groups or []:
            parts.append(", ".join(str(int(x)) for x in g))
        return "; ".join(parts)

    # -----------------------------------------------------------------
    # OK / apply
    # -----------------------------------------------------------------
    def _on_accept(self) -> None:
        defaults = default_tuner_search_space()

        try:
            embed = self._parse_int_list_strict(self.le_embed.text())
            hidden = self._parse_int_list_strict(self.le_hidden.text())
            lstm = self._parse_int_list_strict(self.le_lstm.text())
            attu = self._parse_int_list_strict(self.le_att_units.text())
            heads = self._parse_int_list_strict(self.le_heads.text())
            vsn = self._parse_int_list_strict(self.le_vsn.text())

            maxw = self._parse_int_list_strict(self.le_max_window.text())
            mems = self._parse_int_list_strict(self.le_mem_size.text())

            att_levels = self._parse_groups(self.le_att_levels.text())
            scales = self._parse_int_groups(self.le_scales.text())
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid value",
                str(exc) or "Please check your inputs.",
            )
            return

        # If user leaves empty, fall back to defaults.
        if not embed:
            embed = self._coerce_int_list(None, defaults["embed_dim"])
        if not hidden:
            hidden = self._coerce_int_list(None, defaults["hidden_units"])
        if not lstm:
            lstm = self._coerce_int_list(None, defaults["lstm_units"])
        if not attu:
            attu = self._coerce_int_list(None, defaults["attention_units"])
        if not heads:
            heads = self._coerce_int_list(None, defaults["num_heads"])
        if not vsn:
            vsn = self._coerce_int_list(None, defaults["vsn_units"])

        if not maxw:
            maxw = self._coerce_int_list(
                None,
                defaults.get("max_window_size", [8, 10, 12]),
            )
        if not mems:
            mems = self._coerce_int_list(
                None,
                defaults.get("memory_size", [50, 100]),
            )

        if not att_levels:
            att_levels = self._coerce_groups(
                None,
                defaults.get("attention_levels", [["cross", "memory"]]),
            )
        if not scales:
            scales = self._coerce_int_groups(
                None,
                defaults.get("scales", [[1, 2]]),
            )

        drop = self.dropout_editor.to_search_space_value()
        drop_norm = _FloatRangeListEditor.norm_value(drop)

        # Basic validation
        for name, arr in [
            ("embed_dim", embed),
            ("hidden_units", hidden),
            ("lstm_units", lstm),
            ("attention_units", attu),
            ("num_heads", heads),
            ("vsn_units", vsn),
            ("max_window_size", maxw),
            ("memory_size", mems),
        ]:
            if not arr:
                QMessageBox.warning(
                    self,
                    "Invalid value",
                    f"{name} cannot be empty.",
                )
                return
            if any(int(x) <= 0 for x in arr):
                QMessageBox.warning(
                    self,
                    "Invalid value",
                    f"{name} must be > 0.",
                )
                return

        # Build fragment only for changed keys
        frag: Dict[str, Any] = {}

        def _changed(k: str, newv: Any) -> bool:
            return self._initial.get(k) != newv

        if _changed("embed_dim", embed):
            frag["embed_dim"] = embed
        if _changed("hidden_units", hidden):
            frag["hidden_units"] = hidden
        if _changed("lstm_units", lstm):
            frag["lstm_units"] = lstm
        if _changed("attention_units", attu):
            frag["attention_units"] = attu
        if _changed("num_heads", heads):
            frag["num_heads"] = heads
        if _changed("vsn_units", vsn):
            frag["vsn_units"] = vsn

        if drop_norm != self._initial_dropout_norm:
            frag["dropout_rate"] = drop

        if _changed("max_window_size", maxw):
            frag["max_window_size"] = maxw
        if _changed("memory_size", mems):
            frag["memory_size"] = mems
        if _changed("attention_levels", att_levels):
            frag["attention_levels"] = att_levels
        if _changed("scales", scales):
            frag["scales"] = scales

        if frag:
            self._apply_fragment(frag)

        self.accept()

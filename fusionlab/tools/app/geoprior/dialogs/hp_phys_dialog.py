# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
hp_phys_dialog.py

Store-driven dialog for "More physics HP".

Edits physics-related entries inside:
    GeoPriorConfig.tuner_search_space

Writes are applied only on OK.
Cancel does not touch the store.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from ..config.store import GeoConfigStore
from ..config.prior_schema import FieldKey
from ..config.geoprior_config import default_tuner_search_space


_ALLOWED_PDE = {"off", "gw", "cons", "both"}
_ALLOWED_KAPPA_MODE = {"bar", "kb"}


class _BoolHpEditor(QWidget):
    """
    Bool HP editor.

    - tune=True  -> {"type":"bool"}
    - tune=False -> [fixed_bool]
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        row = QWidget()
        row_lay = QGridLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setHorizontalSpacing(8)

        self.ck_tune = QCheckBox("Tune as boolean HP")
        self.cb_fixed = QComboBox()
        self.cb_fixed.addItems(["True", "False"])

        row_lay.addWidget(self.ck_tune, 0, 0)
        row_lay.addWidget(QLabel("Fixed:"), 0, 1)
        row_lay.addWidget(self.cb_fixed, 0, 2)
        row_lay.setColumnStretch(3, 1)

        lay.addWidget(row)

        self.ck_tune.toggled.connect(self._sync)
        self._sync(self.ck_tune.isChecked())

    def _sync(self, on: bool) -> None:
        self.cb_fixed.setEnabled(not bool(on))

    def from_value(self, v: Any, default: Any) -> None:
        vv = v if v is not None else default

        if isinstance(vv, dict) and vv.get("type") == "bool":
            self.ck_tune.setChecked(True)
            self.cb_fixed.setCurrentText("True")
            return

        fixed = True
        if isinstance(vv, (list, tuple)) and vv:
            fixed = bool(vv[0])
        elif isinstance(vv, bool):
            fixed = bool(vv)

        self.ck_tune.setChecked(False)
        self.cb_fixed.setCurrentText("True" if fixed else "False")

    def to_value(self) -> Any:
        if bool(self.ck_tune.isChecked()):
            return {"type": "bool"}
        fixed = self.cb_fixed.currentText().strip().lower() == "true"
        return [bool(fixed)]

    @staticmethod
    def norm_value(v: Any) -> Tuple:
        if isinstance(v, dict) and v.get("type") == "bool":
            return ("tune",)
        if isinstance(v, (list, tuple)) and v:
            return ("fixed", bool(v[0]))
        if isinstance(v, bool):
            return ("fixed", bool(v))
        return ("other", repr(v))


class _FloatSpecEditor(QWidget):
    """
    Float HP editor supporting:
    - Range dict:
        {"type":"float","min_value":...,"max_value":...,
         "step":..., "sampling":"log"}
    - List:
        [0.1, 0.2, 0.3]
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setHorizontalSpacing(8)
        lay.setVerticalSpacing(6)

        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Range", "List"])

        self.sp_min = QDoubleSpinBox()
        self.sp_min.setDecimals(12)
        self.sp_min.setRange(-1e18, 1e18)

        self.sp_max = QDoubleSpinBox()
        self.sp_max.setDecimals(12)
        self.sp_max.setRange(-1e18, 1e18)

        self.sp_step = QDoubleSpinBox()
        self.sp_step.setDecimals(12)
        self.sp_step.setRange(0.0, 1e18)

        self.cb_sampling = QComboBox()
        self.cb_sampling.addItems(["linear", "log"])

        self.le_list = QLineEdit()
        self.le_list.setPlaceholderText("e.g. 0.05, 0.10, 0.15")

        lay.addWidget(self.cb_mode, 0, 0)
        lay.addWidget(QLabel("Min:"), 0, 1)
        lay.addWidget(self.sp_min, 0, 2)
        lay.addWidget(QLabel("Max:"), 0, 3)
        lay.addWidget(self.sp_max, 0, 4)

        lay.addWidget(QLabel("Step:"), 1, 1)
        lay.addWidget(self.sp_step, 1, 2)
        lay.addWidget(QLabel("Sampling:"), 1, 3)
        lay.addWidget(self.cb_sampling, 1, 4)

        lay.addWidget(self.le_list, 0, 1, 1, 4)

        lay.setColumnStretch(5, 1)

        self.cb_mode.currentTextChanged.connect(self._sync_mode)
        self._sync_mode(self.cb_mode.currentText())

    def _sync_mode(self, mode: str) -> None:
        is_range = str(mode).strip().lower() == "range"

        self.sp_min.setVisible(is_range)
        self.sp_max.setVisible(is_range)
        self.sp_step.setVisible(is_range)
        self.cb_sampling.setVisible(is_range)

        for w in self.findChildren(QLabel):
            t = (w.text() or "").strip().lower()
            if t in {"min:", "max:", "step:", "sampling:"}:
                w.setVisible(is_range)

        self.le_list.setVisible(not is_range)

    def from_value(self, v: Any, default: Any) -> None:
        vv = v if v is not None else default

        if isinstance(vv, dict) and vv.get("type") == "float":
            self.cb_mode.setCurrentText("Range")
            self.sp_min.setValue(float(vv.get("min_value", 0.0)))
            self.sp_max.setValue(float(vv.get("max_value", 0.0)))
            self.sp_step.setValue(float(vv.get("step", 0.0)))
            samp = str(vv.get("sampling", "linear") or "linear")
            samp = samp.strip().lower()
            self.cb_sampling.setCurrentText("log" if samp == "log" else "linear")
            self.le_list.setText("")
            return

        if isinstance(vv, (list, tuple)):
            self.cb_mode.setCurrentText("List")
            parts: List[str] = []
            for x in vv:
                try:
                    parts.append(self._fmt_float(float(x)))
                except Exception:
                    continue
            self.le_list.setText(", ".join(parts))
            self._seed_range_defaults(default)
            return

        self._seed_range_defaults(default)

    def _seed_range_defaults(self, default: Any) -> None:
        self.cb_mode.setCurrentText("Range")

        if isinstance(default, dict):
            self.sp_min.setValue(float(default.get("min_value", 0.0)))
            self.sp_max.setValue(float(default.get("max_value", 0.0)))
            self.sp_step.setValue(float(default.get("step", 0.0)))
            samp = str(default.get("sampling", "linear") or "linear")
            samp = samp.strip().lower()
            self.cb_sampling.setCurrentText("log" if samp == "log" else "linear")
        else:
            self.sp_min.setValue(0.0)
            self.sp_max.setValue(1.0)
            self.sp_step.setValue(0.0)
            self.cb_sampling.setCurrentText("linear")

        self.le_list.setText("")

    def to_value(self) -> Any:
        mode = str(self.cb_mode.currentText() or "Range")
        mode = mode.strip().lower()

        if mode == "list":
            return self._parse_float_list(self.le_list.text())

        mn = float(self.sp_min.value())
        mx = float(self.sp_max.value())
        st = float(self.sp_step.value())
        samp = str(self.cb_sampling.currentText() or "linear")
        samp = samp.strip().lower()

        out: Dict[str, Any] = {
            "type": "float",
            "min_value": mn,
            "max_value": mx,
        }
        if st > 0.0:
            out["step"] = st
        if samp == "log":
            out["sampling"] = "log"
        return out

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
    def _fmt_float(x: float) -> str:
        s = f"{float(x):.12f}"
        s = s.rstrip("0").rstrip(".")
        return s or "0"

    @staticmethod
    def norm_value(v: Any) -> Tuple:
        if isinstance(v, dict) and v.get("type") == "float":
            return (
                "range",
                float(v.get("min_value", 0.0)),
                float(v.get("max_value", 0.0)),
                float(v.get("step", 0.0)),
                str(v.get("sampling", "linear") or "linear"),
            )
        if isinstance(v, (list, tuple)):
            vals: List[float] = []
            for x in v:
                try:
                    vals.append(float(x))
                except Exception:
                    continue
            return ("list", tuple(vals))
        return ("other", repr(v))


class _ChoiceEditor(QWidget):
    """
    Choice HP editor for numeric values:
    - {"type":"choice","values":[...]}
    - or fixed list: [v1, v2, ...]
    """

    def __init__(
        self,
        *,
        placeholder: str = "e.g. 0.0, 1e-4, 1e-3",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.le = QLineEdit()
        self.le.setPlaceholderText(placeholder)
        lay.addWidget(self.le)

    def from_value(self, v: Any, default: Any) -> None:
        vv = v if v is not None else default

        vals: List[Any] = []
        if isinstance(vv, dict) and vv.get("type") == "choice":
            vals = list(vv.get("values", []) or [])
        elif isinstance(vv, (list, tuple)):
            vals = list(vv)
        else:
            vals = []

        parts: List[str] = []
        for x in vals:
            parts.append(str(x))
        self.le.setText(", ".join(parts))

    def to_value(self) -> Any:
        vals = self._parse_tokens(self.le.text())
        return {"type": "choice", "values": vals}

    @staticmethod
    def _parse_tokens(text: str) -> List[float]:
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
        if isinstance(v, dict) and v.get("type") == "choice":
            return ("choice", tuple(v.get("values", []) or []))
        if isinstance(v, (list, tuple)):
            return ("fixed", tuple(v))
        return ("other", repr(v))


class PhysHPDialog(QDialog):
    """
    Edit physics hyperparameters inside tuner_search_space.

    Notes
    -----
    - Writes are applied only on OK.
    - Cancel does not touch the store.
    """

    _PHYS_KEYS = (
        "pde_mode",
        "kappa_mode",
        "scale_pde_residuals",
        "hd_factor",
        "mv",
        "kappa",
        "lambda_offset",
        "lambda_gw",
        "lambda_cons",
        "lambda_prior",
        "lambda_smooth",
        "lambda_bounds",
        "lambda_mv",
        "lambda_q",
        "mv_lr_mult",
        "kappa_lr_mult",
        "scale_mv_with_offset",
        "scale_q_with_offset",
    )

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QDialog] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self.setWindowTitle("Physics hyperparameters")

        offset_mode = str(self._get("offset_mode", "mul") or "mul")
        defaults = default_tuner_search_space(
            offset_mode=offset_mode,
        )

        space = self._coerce_space(
            self._get_space(defaults),
            defaults,
        )

        self._initial: Dict[str, Any] = {}
        for k in self._PHYS_KEYS:
            self._initial[k] = space.get(k, defaults.get(k))

        self._init_norm: Dict[str, Tuple] = {
            "scale_pde_residuals": _BoolHpEditor.norm_value(
                self._initial.get("scale_pde_residuals"),
            ),
            "hd_factor": _FloatSpecEditor.norm_value(
                self._initial.get("hd_factor"),
            ),
            "mv": _FloatSpecEditor.norm_value(
                self._initial.get("mv"),
            ),
            "kappa": _FloatSpecEditor.norm_value(
                self._initial.get("kappa"),
            ),
            "lambda_offset": _FloatSpecEditor.norm_value(
                self._initial.get("lambda_offset"),
            ),
            "lambda_gw": _FloatSpecEditor.norm_value(
                self._initial.get("lambda_gw"),
            ),
            "lambda_cons": _FloatSpecEditor.norm_value(
                self._initial.get("lambda_cons"),
            ),
            "lambda_prior": _FloatSpecEditor.norm_value(
                self._initial.get("lambda_prior"),
            ),
            "lambda_smooth": _FloatSpecEditor.norm_value(
                self._initial.get("lambda_smooth"),
            ),
            "lambda_bounds": _FloatSpecEditor.norm_value(
                self._initial.get("lambda_bounds"),
            ),
            "lambda_mv": _FloatSpecEditor.norm_value(
                self._initial.get("lambda_mv"),
            ),
            "lambda_q": _ChoiceEditor.norm_value(
                self._initial.get("lambda_q"),
            ),
            "mv_lr_mult": _FloatSpecEditor.norm_value(
                self._initial.get("mv_lr_mult"),
            ),
            "kappa_lr_mult": _FloatSpecEditor.norm_value(
                self._initial.get("kappa_lr_mult"),
            ),
            "scale_mv_with_offset": _ChoiceEditor.norm_value(
                self._initial.get("scale_mv_with_offset"),
            ),
            "scale_q_with_offset": _ChoiceEditor.norm_value(
                self._initial.get("scale_q_with_offset"),
            ),
        }

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # ==================================================
        # Card 1: Physics switches
        # ==================================================
        card1, grid1 = self._make_card("Physics switches")

        self.le_pde = QLineEdit()
        self.le_pde.setPlaceholderText("e.g. both")
        self.le_pde.setText(
            self._fmt_str_list(
                self._coerce_str_list(
                    space.get("pde_mode"),
                    defaults.get("pde_mode", ["both"]),
                )
            )
        )

        self.le_kmode = QLineEdit()
        self.le_kmode.setPlaceholderText("e.g. bar, kb")
        self.le_kmode.setText(
            self._fmt_str_list(
                self._coerce_str_list(
                    space.get("kappa_mode"),
                    defaults.get("kappa_mode", ["kb"]),
                )
            )
        )

        self.scale_pde_editor = _BoolHpEditor()
        self.scale_pde_editor.from_value(
            space.get("scale_pde_residuals"),
            defaults.get("scale_pde_residuals"),
        )

        self.hd_editor = _FloatSpecEditor()
        self.hd_editor.from_value(
            space.get("hd_factor"),
            defaults.get("hd_factor"),
        )

        r = 0
        grid1.addWidget(self._lbl("PDE mode(s):"), r, 0)
        grid1.addWidget(self.le_pde, r, 1)
        r += 1

        grid1.addWidget(self._lbl("kappa mode(s):"), r, 0)
        grid1.addWidget(self.le_kmode, r, 1)
        r += 1

        grid1.addWidget(self._lbl("Scale PDE residuals:"), r, 0)
        grid1.addWidget(self.scale_pde_editor, r, 1)
        r += 1

        grid1.addWidget(self._lbl("HD factor:"), r, 0)
        grid1.addWidget(self.hd_editor, r, 1)

        layout.addWidget(card1)

        # ==================================================
        # Card 2: Learnable scalar initials + offset coupling
        # ==================================================
        card2, grid2 = self._make_card(
            "Learnable scalars and offset coupling",
        )

        self.mv_editor = _FloatSpecEditor()
        self.mv_editor.from_value(
            space.get("mv"),
            defaults.get("mv"),
        )

        self.kappa_editor = _FloatSpecEditor()
        self.kappa_editor.from_value(
            space.get("kappa"),
            defaults.get("kappa"),
        )

        self.lam_off_editor = _FloatSpecEditor()
        self.lam_off_editor.from_value(
            space.get("lambda_offset"),
            defaults.get("lambda_offset"),
        )

        self.mv_lr_editor = _FloatSpecEditor()
        self.mv_lr_editor.from_value(
            space.get("mv_lr_mult"),
            defaults.get("mv_lr_mult"),
        )

        self.k_lr_editor = _FloatSpecEditor()
        self.k_lr_editor.from_value(
            space.get("kappa_lr_mult"),
            defaults.get("kappa_lr_mult"),
        )

        self.ck_mv_scale = _ChoiceEditor(
            placeholder="e.g. 0, 1  (False/True as 0/1)",
        )
        self.ck_mv_scale.from_value(
            space.get("scale_mv_with_offset"),
            defaults.get("scale_mv_with_offset"),
        )

        self.ck_q_scale = _ChoiceEditor(
            placeholder="e.g. 0, 1  (False/True as 0/1)",
        )
        self.ck_q_scale.from_value(
            space.get("scale_q_with_offset"),
            defaults.get("scale_q_with_offset"),
        )

        rr = 0
        grid2.addWidget(self._lbl("mv init:"), rr, 0)
        grid2.addWidget(self.mv_editor, rr, 1)
        rr += 1

        grid2.addWidget(self._lbl("kappa init:"), rr, 0)
        grid2.addWidget(self.kappa_editor, rr, 1)
        rr += 1

        grid2.addWidget(self._lbl("lambda offset:"), rr, 0)
        grid2.addWidget(self.lam_off_editor, rr, 1)
        rr += 1

        grid2.addWidget(self._lbl("mv lr mult:"), rr, 0)
        grid2.addWidget(self.mv_lr_editor, rr, 1)
        rr += 1

        grid2.addWidget(self._lbl("kappa lr mult:"), rr, 0)
        grid2.addWidget(self.k_lr_editor, rr, 1)
        rr += 1

        grid2.addWidget(self._lbl("scale mv w/ offset:"), rr, 0)
        grid2.addWidget(self.ck_mv_scale, rr, 1)
        rr += 1

        grid2.addWidget(self._lbl("scale q w/ offset:"), rr, 0)
        grid2.addWidget(self.ck_q_scale, rr, 1)

        layout.addWidget(card2)

        # ==================================================
        # Card 3: Physics loss weights
        # ==================================================
        card3, grid3 = self._make_card("Physics loss weights")

        self.lgw_editor = _FloatSpecEditor()
        self.lgw_editor.from_value(
            space.get("lambda_gw"),
            defaults.get("lambda_gw"),
        )

        self.lcons_editor = _FloatSpecEditor()
        self.lcons_editor.from_value(
            space.get("lambda_cons"),
            defaults.get("lambda_cons"),
        )

        self.lprior_editor = _FloatSpecEditor()
        self.lprior_editor.from_value(
            space.get("lambda_prior"),
            defaults.get("lambda_prior"),
        )

        self.lsmooth_editor = _FloatSpecEditor()
        self.lsmooth_editor.from_value(
            space.get("lambda_smooth"),
            defaults.get("lambda_smooth"),
        )

        self.lbounds_editor = _FloatSpecEditor()
        self.lbounds_editor.from_value(
            space.get("lambda_bounds"),
            defaults.get("lambda_bounds"),
        )

        self.lmv_editor = _FloatSpecEditor()
        self.lmv_editor.from_value(
            space.get("lambda_mv"),
            defaults.get("lambda_mv"),
        )

        self.lq_editor = _ChoiceEditor()
        self.lq_editor.from_value(
            space.get("lambda_q"),
            defaults.get("lambda_q"),
        )

        tr = 0
        grid3.addWidget(self._lbl("λ gw:"), tr, 0)
        grid3.addWidget(self.lgw_editor, tr, 1)
        tr += 1

        grid3.addWidget(self._lbl("λ cons:"), tr, 0)
        grid3.addWidget(self.lcons_editor, tr, 1)
        tr += 1

        grid3.addWidget(self._lbl("λ prior:"), tr, 0)
        grid3.addWidget(self.lprior_editor, tr, 1)
        tr += 1

        grid3.addWidget(self._lbl("λ smooth:"), tr, 0)
        grid3.addWidget(self.lsmooth_editor, tr, 1)
        tr += 1

        grid3.addWidget(self._lbl("λ bounds:"), tr, 0)
        grid3.addWidget(self.lbounds_editor, tr, 1)
        tr += 1

        grid3.addWidget(self._lbl("λ mv:"), tr, 0)
        grid3.addWidget(self.lmv_editor, tr, 1)
        tr += 1

        grid3.addWidget(self._lbl("λ q:"), tr, 0)
        grid3.addWidget(self.lq_editor, tr, 1)

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
    def _get(self, key: str, default: Any) -> Any:
        try:
            return self._store.get_value(FieldKey(key))
        except Exception:
            return default

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
    def _coerce_space(space: Any, defaults: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(space, dict):
            return dict(defaults)
        merged = dict(defaults)
        merged.update(dict(space))
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

    # -----------------------------------------------------------------
    # Parsing helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _coerce_str_list(v: Any, fallback: List[str]) -> List[str]:
        if isinstance(v, str):
            s = v.strip()
            return [s] if s else list(fallback)
        if isinstance(v, (list, tuple)):
            out: List[str] = []
            for x in v:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())
            return out or list(fallback)
        return list(fallback)

    @staticmethod
    def _parse_str_list(text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        out: List[str] = []
        for part in text.replace(";", ",").split(","):
            s = part.strip()
            if s:
                out.append(s)
        return out

    @staticmethod
    def _fmt_str_list(xs: List[str]) -> str:
        return ", ".join(xs or [])

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    def _validate_str_choices(
        self,
        *,
        name: str,
        vals: List[str],
        allowed: set,
    ) -> Optional[str]:
        bad = [v for v in vals if v not in allowed]
        if bad:
            return f"{name}: invalid values: {bad}"
        return None

    def _validate_float_spec(
        self,
        *,
        name: str,
        spec: Any,
    ) -> Optional[str]:
        if isinstance(spec, dict) and spec.get("type") == "float":
            mn = float(spec.get("min_value", 0.0))
            mx = float(spec.get("max_value", 0.0))
            if not (mn < mx):
                return f"{name}: min_value must be < max_value"
            samp = str(spec.get("sampling", "linear") or "linear")
            samp = samp.strip().lower()
            if samp == "log" and (mn <= 0.0 or mx <= 0.0):
                return f"{name}: log sampling needs min/max > 0"
            step = float(spec.get("step", 0.0) or 0.0)
            if step < 0.0:
                return f"{name}: step must be >= 0"
            return None

        if isinstance(spec, (list, tuple)):
            if not spec:
                return f"{name}: empty list"
            return None

        return f"{name}: unsupported spec"

    # -----------------------------------------------------------------
    # OK / apply
    # -----------------------------------------------------------------
    def _on_accept(self) -> None:
        defaults = default_tuner_search_space(
            offset_mode=str(self._get("offset_mode", "mul") or "mul"),
        )

        pde = self._parse_str_list(self.le_pde.text())
        kmode = self._parse_str_list(self.le_kmode.text())

        if not pde:
            pde = self._coerce_str_list(
                None,
                defaults.get("pde_mode", ["both"]),
            )
        if not kmode:
            kmode = self._coerce_str_list(
                None,
                defaults.get("kappa_mode", ["kb"]),
            )

        err = self._validate_str_choices(
            name="pde_mode",
            vals=pde,
            allowed=_ALLOWED_PDE,
        )
        if err:
            QMessageBox.warning(self, "Invalid PDE mode", err)
            return

        err = self._validate_str_choices(
            name="kappa_mode",
            vals=kmode,
            allowed=_ALLOWED_KAPPA_MODE,
        )
        if err:
            QMessageBox.warning(self, "Invalid kappa mode", err)
            return

        scale_pde = self.scale_pde_editor.to_value()
        hd = self.hd_editor.to_value()
        mv = self.mv_editor.to_value()
        kap = self.kappa_editor.to_value()
        lam_off = self.lam_off_editor.to_value()

        mv_lr = self.mv_lr_editor.to_value()
        k_lr = self.k_lr_editor.to_value()

        lgw = self.lgw_editor.to_value()
        lcons = self.lcons_editor.to_value()
        lprior = self.lprior_editor.to_value()
        lsmooth = self.lsmooth_editor.to_value()
        lbounds = self.lbounds_editor.to_value()
        lmv = self.lmv_editor.to_value()

        lq = self.lq_editor.to_value()
        mv_scale = self.ck_mv_scale.to_value()
        q_scale = self.ck_q_scale.to_value()

        # Validate float specs
        for nm, sp in [
            ("hd_factor", hd),
            ("mv", mv),
            ("kappa", kap),
            ("lambda_offset", lam_off),
            ("mv_lr_mult", mv_lr),
            ("kappa_lr_mult", k_lr),
            ("lambda_gw", lgw),
            ("lambda_cons", lcons),
            ("lambda_prior", lprior),
            ("lambda_smooth", lsmooth),
            ("lambda_bounds", lbounds),
            ("lambda_mv", lmv),
        ]:
            e = self._validate_float_spec(name=nm, spec=sp)
            if e:
                QMessageBox.warning(self, "Invalid range", e)
                return

        # Validate choice lists (lambda_q + bool choices)
        lq_vals = list(lq.get("values", []) or [])
        if any(float(x) < 0.0 for x in lq_vals):
            QMessageBox.warning(
                self,
                "Invalid lambda_q",
                "lambda_q values must be >= 0",
            )
            return

        # Bool choices are represented as floats in editor:
        # user can enter "0, 1" or "1" etc.
        for nm, ch in [
            ("scale_mv_with_offset", mv_scale),
            ("scale_q_with_offset", q_scale),
        ]:
            vals = list(ch.get("values", []) or [])
            for v in vals:
                if float(v) not in (0.0, 1.0):
                    QMessageBox.warning(
                        self,
                        "Invalid choice",
                        f"{nm}: use 0/1 for False/True",
                    )
                    return

        frag: Dict[str, Any] = {}

        if self._initial.get("pde_mode") != pde:
            frag["pde_mode"] = pde
        if self._initial.get("kappa_mode") != kmode:
            frag["kappa_mode"] = kmode

        if (
            _BoolHpEditor.norm_value(scale_pde)
            != self._init_norm["scale_pde_residuals"]
        ):
            frag["scale_pde_residuals"] = scale_pde

        if (
            _FloatSpecEditor.norm_value(hd)
            != self._init_norm["hd_factor"]
        ):
            frag["hd_factor"] = hd

        if (
            _FloatSpecEditor.norm_value(mv)
            != self._init_norm["mv"]
        ):
            frag["mv"] = mv

        if (
            _FloatSpecEditor.norm_value(kap)
            != self._init_norm["kappa"]
        ):
            frag["kappa"] = kap

        if (
            _FloatSpecEditor.norm_value(lam_off)
            != self._init_norm["lambda_offset"]
        ):
            frag["lambda_offset"] = lam_off

        if (
            _FloatSpecEditor.norm_value(mv_lr)
            != self._init_norm["mv_lr_mult"]
        ):
            frag["mv_lr_mult"] = mv_lr

        if (
            _FloatSpecEditor.norm_value(k_lr)
            != self._init_norm["kappa_lr_mult"]
        ):
            frag["kappa_lr_mult"] = k_lr

        for key, editor_val, norm_fn in [
            ("lambda_gw", lgw, _FloatSpecEditor.norm_value),
            ("lambda_cons", lcons, _FloatSpecEditor.norm_value),
            ("lambda_prior", lprior, _FloatSpecEditor.norm_value),
            ("lambda_smooth", lsmooth, _FloatSpecEditor.norm_value),
            ("lambda_bounds", lbounds, _FloatSpecEditor.norm_value),
            ("lambda_mv", lmv, _FloatSpecEditor.norm_value),
        ]:
            if norm_fn(editor_val) != self._init_norm[key]:
                frag[key] = editor_val

        if (
            _ChoiceEditor.norm_value(lq)
            != self._init_norm["lambda_q"]
        ):
            frag["lambda_q"] = lq

        if (
            _ChoiceEditor.norm_value(mv_scale)
            != self._init_norm["scale_mv_with_offset"]
        ):
            frag["scale_mv_with_offset"] = mv_scale

        if (
            _ChoiceEditor.norm_value(q_scale)
            != self._init_norm["scale_q_with_offset"]
        ):
            frag["scale_q_with_offset"] = q_scale

        if frag:
            self._apply_fragment(frag)

        self.accept()

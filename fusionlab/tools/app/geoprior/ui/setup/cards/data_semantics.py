# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.cards.data_semantics

Data columns & semantics.

Modern UX:
- Column mapping on the left (clean pickers).
- Semantics on the right (mode + kinds + sign).
- Advisory banner when pihal_like is selected.
- Advanced optional settings in an expander.

Store-driven via Binder.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, QSignalBlocker
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ....config.store import GeoConfigStore
from ..schema import default_fields
from ..bindings import Binder
from .base import CardBase

class _Expander(QWidget):
    def __init__(
        self,
        title: str,
        *,
        parent: QWidget,
    ) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self.btn = QToolButton(self)
        self.btn.setCheckable(True)
        self.btn.setChecked(False)
        self.btn.setText(str(title))
        self.btn.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.btn.setArrowType(Qt.RightArrow)

        self.body = QWidget(self)
        self.body.setVisible(False)

        self.body_l = QGridLayout(self.body)
        self.body_l.setContentsMargins(8, 6, 8, 6)
        self.body_l.setHorizontalSpacing(10)
        self.body_l.setVerticalSpacing(6)

        self.btn.toggled.connect(self._toggle)

        root.addWidget(self.btn, 0)
        root.addWidget(self.body, 0)

    def _toggle(self, on: bool) -> None:
        self.body.setVisible(bool(on))
        self.btn.setArrowType(
            Qt.DownArrow if on else Qt.RightArrow
        )


def _set_combo_columns(
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

        if cur:
            idx = cmb.findText(cur)
            if idx >= 0:
                cmb.setCurrentIndex(idx)
            elif cmb.isEditable():
                cmb.setEditText(cur)


class DataSemanticsCard(CardBase):
    """Data columns & semantics (store-driven)."""

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        binder: Binder,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            section_id="data_semantics",
            title="Data columns & semantics",
            subtitle=(
                "Map dataset columns to required roles "
                "and choose meaning conventions."
            ),
            parent=parent,
        )

        self.store = store
        self.binder = binder

        self._fs = default_fields()
        self._cols: List[str] = []
        self._col_combos: Dict[str, QComboBox] = {}

        self._build()
        self._wire()
        self.refresh()

    # -----------------------------------------------------------------
    # External API (panel pushes dataset columns here)
    # -----------------------------------------------------------------
    def set_dataset_columns(self, cols: List[str]) -> None:
        self._cols = [str(c) for c in (cols or [])]

        for k, cmb in self._col_combos.items():
            spec = self._fs.get(k)
            none_text = None if spec is None else spec.none_text
            _set_combo_columns(
                cmb,
                self._cols,
                none_text=none_text,
            )

        self.binder.refresh_keys(set(self._col_combos.keys()))
        self.refresh()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build(self) -> None:
        body = self.body_layout()

        grid = QWidget(self)
        g = QGridLayout(grid)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10)

        g.setColumnStretch(0, 1)
        g.setColumnStretch(1, 1)

        self.grp_cols = self._build_cols_group(grid)
        self.grp_sem = self._build_sem_group(grid)

        g.addWidget(self.grp_cols, 0, 0)
        g.addWidget(self.grp_sem, 0, 1)

        body.addWidget(grid, 0)

    def _build_cols_group(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Columns", parent)
        lay = QGridLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)
        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)

        r = 0

        lay.addWidget(self._hint(
            "Pick the dataset columns used by the model."
        ), r, 0, 1, 4)
        r += 1

        lay.addWidget(self._lab("time_col", box), r, 0)
        lay.addWidget(self._col_combo("time_col", box), r, 1)
        lay.addWidget(self._lab("subs_col", box), r, 2)
        lay.addWidget(self._col_combo("subs_col", box), r, 3)
        r += 1

        lay.addWidget(self._lab("lon_col", box), r, 0)
        lay.addWidget(self._col_combo("lon_col", box), r, 1)
        lay.addWidget(self._lab("lat_col", box), r, 2)
        lay.addWidget(self._col_combo("lat_col", box), r, 3)
        r += 1

        lay.addWidget(self._lab("gwl_col", box), r, 0)
        lay.addWidget(self._col_combo("gwl_col", box), r, 1)
        lay.addWidget(self._lab("h_field_col", box), r, 2)
        lay.addWidget(self._col_combo("h_field_col", box), r, 3)
        r += 1

        lay.addWidget(self._lab("z_surf_col", box), r, 0)
        lay.addWidget(self._col_combo("z_surf_col", box), r, 1)

        chk = QCheckBox(
            self._fs["include_z_surf_as_static"].label,
            box,
        )
        tip = self._fs["include_z_surf_as_static"].tooltip
        if tip:
            chk.setToolTip(tip)

        self.binder.bind_checkbox(
            "include_z_surf_as_static",
            chk,
        )
        lay.addWidget(chk, r, 2, 1, 2)
        r += 1

        return box

    def _build_sem_group(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Semantics", parent)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        self.banner = QLabel("", box)
        self.banner.setWordWrap(True)
        self.banner.setObjectName("modeBanner")

        self._banner_style(box)
        lay.addWidget(self.banner, 0)

        form = QWidget(box)
        f = QGridLayout(form)
        f.setContentsMargins(0, 0, 0, 0)
        f.setHorizontalSpacing(10)
        f.setVerticalSpacing(8)
        f.setColumnStretch(1, 1)
        f.setColumnStretch(3, 1)

        self.cmb_mode = self._enum_combo("mode", box)
        self.cmb_sub_kind = self._enum_combo(
            "subsidence_kind",
            box,
        )

        self.cmb_gwl_kind = self._enum_combo("gwl_kind", box)
        self.cmb_gwl_sign = self._enum_combo("gwl_sign", box)

        r = 0
        f.addWidget(self._lab("mode", box), r, 0)
        f.addWidget(self.cmb_mode, r, 1)
        f.addWidget(self._lab("subsidence_kind", box), r, 2)
        f.addWidget(self.cmb_sub_kind, r, 3)
        r += 1

        f.addWidget(self._lab("gwl_kind", box), r, 0)
        f.addWidget(self.cmb_gwl_kind, r, 1)
        f.addWidget(self._lab("gwl_sign", box), r, 2)
        f.addWidget(self.cmb_gwl_sign, r, 3)
        r += 1

        lay.addWidget(form, 0)

        row = QWidget(box)
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)

        chk = QCheckBox(self._fs["use_head_proxy"].label, box)
        tip = self._fs["use_head_proxy"].tooltip
        if tip:
            chk.setToolTip(tip)

        self.binder.bind_checkbox("use_head_proxy", chk)

        ed = QLineEdit(box)
        ed.setPlaceholderText(self._fs["head_col"].placeholder or "")
        tip = self._fs["head_col"].tooltip
        if tip:
            ed.setToolTip(tip)

        self.binder.bind_line_edit(
            "head_col",
            ed,
            on="editingFinished",
        )

        rl.addWidget(chk, 1)
        rl.addWidget(QLabel("Head col:", box), 0)
        rl.addWidget(ed, 1)

        lay.addWidget(row, 0)

        self.exp = _Expander("Advanced", parent=box)

        s = self._fs["gwl_dyn_index"]
        lab = QLabel(f"{s.label}:", self.exp.body)

        self.chk_dyn = QCheckBox("Set", self.exp.body)
        self.sp_dyn = QSpinBox(self.exp.body)
        self.sp_dyn.setRange(-999999, 999999)

        tip = s.tooltip
        if tip:
            lab.setToolTip(tip)
            self.chk_dyn.setToolTip(tip)
            self.sp_dyn.setToolTip(tip)

        self.exp.body_l.addWidget(lab, 0, 0)

        row2 = QWidget(self.exp.body)
        r2 = QHBoxLayout(row2)
        r2.setContentsMargins(0, 0, 0, 0)
        r2.setSpacing(6)

        r2.addWidget(self.chk_dyn, 0)
        r2.addWidget(self.sp_dyn, 1)

        self.exp.body_l.addWidget(row2, 0, 1)

        self.binder.bind_optional_spin_box(
            "gwl_dyn_index",
            self.sp_dyn,
            self.chk_dyn,
        )

        lay.addWidget(self.exp, 0)

        return box

    def _banner_style(self, parent: QWidget) -> None:
        parent.setStyleSheet(
            "\n".join(
                [
                    "QLabel#modeBanner {",
                    "  padding: 8px 10px;",
                    "  border-radius: 10px;",
                    "  border: 1px solid",
                    "    rgba(0,0,0,0.10);",
                    "  background: rgba(0,0,0,0.03);",
                    "  color: rgba(30,30,30,0.80);",
                    "  font-size: 11px;",
                    "}",
                ]
            )
        )

    def _hint(self, text: str) -> QLabel:
        lab = QLabel(str(text), self)
        lab.setWordWrap(True)
        lab.setObjectName("dsHint")
        lab.setStyleSheet(
            "\n".join(
                [
                    "QLabel#dsHint {",
                    "  color: rgba(30,30,30,0.72);",
                    "  font-size: 11px;",
                    "}",
                ]
            )
        )
        return lab

    def _lab(self, key: str, parent: QWidget) -> QLabel:
        s = self._fs[key]
        w = QLabel(f"{s.label}:", parent)
        if s.tooltip:
            w.setToolTip(s.tooltip)
        return w

    def _col_combo(self, key: str, parent: QWidget) -> QComboBox:
        s = self._fs[key]

        cmb = QComboBox(parent)
        cmb.setEditable(True)

        le = cmb.lineEdit()
        if le is not None:
            le.setPlaceholderText(s.placeholder or "")

        if s.tooltip:
            cmb.setToolTip(s.tooltip)

        _set_combo_columns(
            cmb,
            self._cols,
            none_text=s.none_text,
        )

        self._col_combos[key] = cmb

        self.binder.bind_combo(
            key,
            cmb,
            editable=True,
            none_text=s.none_text,
            use_item_data=False,
        )
        return cmb

    def _enum_combo(self, key: str, parent: QWidget) -> QComboBox:
        s = self._fs[key]
        cmb = QComboBox(parent)
        if s.tooltip:
            cmb.setToolTip(s.tooltip)

        items = s.items if s.items is not None else []
        self.binder.bind_combo(
            key,
            cmb,
            items=items,
            editable=False,
            none_text=s.none_text,
            use_item_data=True,
        )
        return cmb

    # -----------------------------------------------------------------
    # Wiring / refresh
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.cmb_mode.currentIndexChanged.connect(
            self._on_mode_changed,
        )

        self.store.config_changed.connect(
            lambda _k: self.refresh(),
        )
        self.store.config_replaced.connect(
            lambda _cfg: self.refresh(),
        )

        self._on_mode_changed()

    def refresh(self) -> None:
        self._sync_mode_banner()

    def _on_mode_changed(self) -> None:
        self._sync_mode_banner()

    def _sync_mode_banner(self) -> None:
        mode = str(self.cmb_mode.currentData() or "")
        mode = mode.strip().lower()

        if not mode:
            mode = str(self.cmb_mode.currentText() or "")
            mode = mode.strip().lower()

        # Default guidance
        msg = (
            "Recommended: tft_like. It supports the "
            "Deep Prior Network architecture."
        )
        accent = "ok"
        title = "Mode"

        if mode == "pihal_like":
            msg = (
                "pihal_like is not available for the "
                "Deep Prior Network architecture. "
                "Use tft_like instead."
            )
            accent = "warn"
            title = "Not supported"

        self.banner.setText(f"<b>{title}:</b> {msg}")
        self.badge(
            "mode",
            text="tft_like" if mode != "pihal_like" else "warn",
            accent=accent,
            tip=msg,
        )

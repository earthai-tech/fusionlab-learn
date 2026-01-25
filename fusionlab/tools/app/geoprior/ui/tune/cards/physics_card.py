# geoprior/ui/tune/cards/physics_card.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QStyle,
)

from ....config.prior_schema import FieldKey
from ....config.store import GeoConfigStore
from ....utils.components import RangeListEditor
from ...icon_utils import try_icon

MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]

__all__ = ["TunePhysicsCard"]


class TunePhysicsCard(QWidget):
    """
    Expandable Tune card: Physics switches.

    Header:
      - title
      - summary line (objectName="sumLine")
      - Edit toggle (objectName="disclosure")

    Body (expand/collapse inside same card):
      - PDE modes (comma list)
      - kappa mode (comma list)
      - tune scale_pde_residuals as bool
      - HD factor range + gear details button
    """

    edit_toggled = pyqtSignal(bool)
    hd_details_clicked = pyqtSignal()

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

        self._writing = False
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
        
        self._frame, body = self._make_card("Physics switches")
        self._frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        root.addWidget(self._frame)
        
        # -------------------------
        # Header row + Summary line
        # -------------------------
        sum_row = QWidget(self._frame)
        sum_l = QHBoxLayout(sum_row)
        sum_l.setContentsMargins(0, 0, 0, 0)
        sum_l.setSpacing(8)
        
        self.lbl_sum = QLabel("mode=?  kappa=?  hd=?", self._frame)
        self.lbl_sum.setObjectName("sumLine")
        self.lbl_sum.setWordWrap(True)
        
        self.btn_edit = QToolButton(self._frame)
        self.btn_edit.setObjectName("disclosure")
        self.btn_edit.setCursor(Qt.PointingHandCursor)
        self.btn_edit.setCheckable(True)
        self.btn_edit.setAutoRaise(True)
        self.btn_edit.setText("Edit")
        self.btn_edit.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self._set_edit_icon(expanded=False)

        sum_l.addWidget(self.lbl_sum, 1)
        sum_l.addWidget(self.btn_edit, 0)
        
        body.addWidget(sum_row)

        # -------------------------
        # Details body (collapsible)
        # -------------------------
        self.details = QWidget(self._frame)
        self.details.setObjectName("drawer")
        self.details.setVisible(False)

        body_l = QVBoxLayout(self.details)
        body_l.setContentsMargins(0, 4, 0, 0)
        body_l.setSpacing(8)

        self.hp_pde_mode = QLineEdit(self.details)
        self.hp_kappa_mode = QLineEdit(self.details)

        self.hp_pde_mode.setPlaceholderText(
            "both, gw, cons"
        )
        self.hp_kappa_mode.setPlaceholderText(
            "bar, kb"
        )

        self.hp_scale_pde_bool = QCheckBox(
            "Tune 'scale PDE residuals' as boolean HP",
            self.details,
        )

        self.hp_hd = RangeListEditor(
            self.details,
            min_allowed=0.0,
            max_allowed=2.0,
            decimals=3,
            show_sampling=False,
        )
        self.hp_hd.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )

        self.btn_hd_details = QToolButton(self.details)
        self.btn_hd_details.setObjectName("miniAction")
        self.btn_hd_details.setAutoRaise(True)
        self.btn_hd_details.setCursor(Qt.PointingHandCursor)
        self.btn_hd_details.setFocusPolicy(Qt.NoFocus)
        self.btn_hd_details.setToolTip(
            "Advanced HD factor + scale PDE residuals\n"
            "• step / sampling / list\n"
            "• fixed True/False for scale PDE"
        )

        ico = try_icon("settings.svg")
        if ico is None or ico.isNull():
            ico = QIcon.fromTheme("settings")
        if ico.isNull():
            ico = self.style().standardIcon(
                QStyle.SP_FileDialogDetailedView
            )
        self.btn_hd_details.setIcon(ico)
        self.btn_hd_details.setIconSize(
            self.btn_hd_details.iconSize()
        )
        self.btn_hd_details.setFixedSize(30, 26)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)

        r = 0
        grid.addWidget(QLabel("PDE modes:"), r, 0)
        grid.addWidget(self.hp_pde_mode, r, 1, 1, 2)
        r += 1

        grid.addWidget(QLabel("k mode (bar/kb):"), r, 0)
        grid.addWidget(self.hp_kappa_mode, r, 1, 1, 2)
        r += 1

        grid.addWidget(self.hp_scale_pde_bool, r, 0, 1, 3)
        r += 1

        grid.addWidget(QLabel("HD factor:"), r, 0)
        grid.addWidget(self.hp_hd, r, 1)
        grid.addWidget(
            self.btn_hd_details,
            r,
            2,
            alignment=Qt.AlignRight,
        )

        body_l.addLayout(grid)
        body.addWidget(self.details)

    # -----------------------------------------------------------------
    # Edit toggle helpers (shared convention)
    # -----------------------------------------------------------------
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
    # Wiring
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.btn_edit.toggled.connect(self._on_toggle)

        self.hp_pde_mode.editingFinished.connect(
            self._commit_to_store
        )
        self.hp_kappa_mode.editingFinished.connect(
            self._commit_to_store
        )
        self.hp_scale_pde_bool.toggled.connect(
            lambda _=None: self._commit_to_store()
        )

        self._connect_range_editor(self.hp_hd)

        self.btn_hd_details.clicked.connect(
            lambda *_: self.hd_details_clicked.emit()
        )


    # -----------------------------------------------------------------
    # Store helpers
    # -----------------------------------------------------------------
    def _get_space(self) -> Dict[str, Any]:
        try:
            v = self._store.get_value(
                FieldKey("tuner_search_space"),
                default={},
            )
        except Exception:
            v = {}
        if isinstance(v, dict):
            return dict(v)
        return {}

    def _set_space(self, space: Dict[str, Any]) -> None:
        try:
            self._store.merge_dict_field(
                "tuner_search_space",
                dict(space),
                replace=True,
            )
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def refresh_from_store(self) -> None:
        self._writing = True
        try:
            space = self._get_space()

            pde = space.get("pde_mode", [])
            kap = space.get("kappa_mode", [])
            hd = space.get("hd_factor", None)
            sc = space.get("scale_pde_residuals", None)

            self.hp_pde_mode.setText(
                _to_csv(pde)
            )
            self.hp_kappa_mode.setText(
                _to_csv(kap)
            )

            is_bool = (
                isinstance(sc, dict)
                and str(sc.get("type", "")).lower()
                == "bool"
            )
            self.hp_scale_pde_bool.setChecked(
                bool(is_bool)
            )

            self.hp_hd.from_search_space_value(hd, hd)

            self.lbl_sum.setText(
                _physics_summary(space)
            )
        finally:
            self._writing = False

    # -----------------------------------------------------------------
    # Commit
    # -----------------------------------------------------------------
    def _commit_to_store(self) -> None:
        if self._writing:
            return

        base = self._get_space()
        new = dict(base)

        pde = _csv_to_list(self.hp_pde_mode.text())
        kap = _csv_to_list(self.hp_kappa_mode.text())

        if pde:
            new["pde_mode"] = pde
        else:
            new.pop("pde_mode", None)

        if kap:
            new["kappa_mode"] = kap
        else:
            new.pop("kappa_mode", None)

        hd = self.hp_hd.to_search_space_value()
        if hd is not None:
            new["hd_factor"] = hd

        if self.hp_scale_pde_bool.isChecked():
            new["scale_pde_residuals"] = {"type": "bool"}
        else:
            new.pop("scale_pde_residuals", None)

        self._writing = True
        try:
            self._set_space(new)
        finally:
            self._writing = False

        self.refresh_from_store()

    # -----------------------------------------------------------------
    # RangeListEditor bridge
    # -----------------------------------------------------------------
    def _connect_range_editor(self, editor: QWidget) -> None:
        for nm in (
            "changed",
            "valueChanged",
            "sig_changed",
            "signalChanged",
        ):
            sig = getattr(editor, nm, None)
            if sig is None:
                continue
            if hasattr(sig, "connect"):
                try:
                    sig.connect(self._commit_to_store)
                    return
                except Exception:
                    continue


def _csv_to_list(text: str) -> list[str]:
    parts = [p.strip() for p in (text or "").split(",")]
    return [p for p in parts if p]


def _to_csv(v: Any) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    if v is None:
        return ""
    return str(v)


def _fmt_hd(hd: Any) -> str:
    if isinstance(hd, dict):
        t = str(hd.get("type", "")).lower()
        if t in ("float", "int", "range"):
            mn = hd.get("min_value", hd.get("min"))
            mx = hd.get("max_value", hd.get("max"))
            if mn is not None and mx is not None:
                return f"{mn}–{mx}"
    return "off"


def _physics_summary(space: Dict[str, Any]) -> str:
    pde = _to_csv(space.get("pde_mode", ""))
    kap = _to_csv(space.get("kappa_mode", ""))
    hd = _fmt_hd(space.get("hd_factor", None))

    sc = space.get("scale_pde_residuals", None)
    is_bool = (
        isinstance(sc, dict)
        and str(sc.get("type", "")).lower()
        == "bool"
    )

    pde_s = pde if pde else "—"
    kap_s = kap if kap else "—"
    sc_s = "on" if is_bool else "off"

    return (
        f"pde={pde_s}  "
        f"kappa={kap_s}  "
        f"hd={hd}  "
        f"scale_bool={sc_s}"
    )

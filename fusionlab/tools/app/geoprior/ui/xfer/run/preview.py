# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer.run.preview

Xfer Run Preview panel.

Shows:
- Resolved plan (human readable)
- Readiness + warnings (explicit)
- Details tree (deep view, last)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStyle,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...icon_utils import try_icon
from ....config.store import GeoConfigStore

__all__ = ["XferRunPreview"]


def _exists(p: str) -> bool:
    try:
        return bool(p) and os.path.exists(p)
    except Exception:
        return False


def _abspath(p: str) -> str:
    try:
        return os.path.abspath(p)
    except Exception:
        return p


def _dir_of(p: str) -> str:
    p = str(p or "").strip()
    if not p:
        return ""
    ap = _abspath(p)
    if os.path.isdir(ap):
        return ap
    return os.path.dirname(ap)


def _open_path(p: str) -> None:
    p = str(p or "").strip()
    if not _exists(p):
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(p))


class _Chip(QLabel):
    """
    Small status chip.

    Style hook:
    - objectName: inferChip
    - property: kind = ok|warn|err|off|info
    """

    def __init__(
        self,
        text: str,
        *,
        kind: str = "off",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(text, parent)
        self.setObjectName("inferChip")
        self.setProperty("kind", kind)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(
            QSizePolicy.Minimum,
            QSizePolicy.Fixed,
        )

    def set_kind(self, kind: str) -> None:
        self.setProperty("kind", str(kind))
        self.style().unpolish(self)
        self.style().polish(self)


class XferRunPreview(QFrame):
    """
    Right-side preview widget (content only).

    Place inside a "Run preview" card body.

    Notes
    -----
    "Compute" is intentionally NOT shown here anymore
    (already displayed in the navigator panel).

    Public API
    ----------
    - refresh_all()
    - set_plan_text(text)  (optional override)
    - status_label()       (for bottom bar)
    """

    toast = pyqtSignal(str)

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        show_compute: bool = False,  # kept for compat; ignored
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store
        self._plan_text = ""

        self.setObjectName("runPreviewPanel")
        self.setFrameShape(QFrame.NoFrame)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._build_ui()
        self._wire_store()
        self.refresh_all()

    # -------------------------------------------------
    # Icons
    # -------------------------------------------------
    def _std_icon(self, sp: QStyle.StandardPixmap) -> Any:
        return self.style().standardIcon(sp)

    def _set_icon(
        self,
        btn: QToolButton,
        name: str,
        fallback: QStyle.StandardPixmap,
    ) -> None:
        ic = try_icon(name)
        if ic is None:
            ic = self._std_icon(fallback)
        btn.setIcon(ic)

    def _mk_icon_btn(
        self,
        tip: str,
        icon_name: str,
        fallback: QStyle.StandardPixmap,
    ) -> QToolButton:
        b = QToolButton(self)
        b.setAutoRaise(True)
        b.setToolButtonStyle(Qt.ToolButtonIconOnly)
        b.setToolTip(tip)
        self._set_icon(b, icon_name, fallback)
        return b

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        # =========================================
        # Header row: title + chips + copy
        # =========================================
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)

        t = QLabel("Run preview", self)
        t.setObjectName("subTitle")

        self.chip_ready = _Chip(
            "—",
            kind="off",
            parent=self,
        )
        self.chip_warn = _Chip(
            "0 warn",
            kind="off",
            parent=self,
        )

        self.btn_copy = self._mk_icon_btn(
            "Copy resolved plan",
            "copy.svg",
            QStyle.SP_DialogSaveButton,
        )

        hdr.addWidget(t, 0)
        hdr.addSpacing(6)
        hdr.addWidget(self.chip_ready, 0)
        hdr.addWidget(self.chip_warn, 0)
        hdr.addStretch(1)
        hdr.addWidget(self.btn_copy, 0)
        root.addLayout(hdr)

        # Hint line
        self.lbl_hint = QLabel(
            "Checks readiness and shows the resolved setup.",
            self,
        )
        self.lbl_hint.setObjectName("sumLine")
        self.lbl_hint.setWordWrap(True)
        root.addWidget(self.lbl_hint, 0)

        # =========================================
        # Resolved plan: title + actions
        # (moved here like inference)
        # =========================================
        plan_hdr = QHBoxLayout()
        plan_hdr.setContentsMargins(0, 0, 0, 0)
        plan_hdr.setSpacing(8)

        plan_t = QLabel("Resolved plan", self)
        plan_t.setObjectName("subTitle")

        self.btn_open_root = self._mk_icon_btn(
            "Open results root",
            "folder_open.svg",
            QStyle.SP_DirOpenIcon,
        )
        self.btn_open_last = self._mk_icon_btn(
            "Open last output folder",
            "folder.svg",
            QStyle.SP_DirIcon,
        )
        self.btn_refresh = self._mk_icon_btn(
            "Refresh preview",
            "refresh.svg",
            QStyle.SP_BrowserReload,
        )

        plan_hdr.addWidget(plan_t, 0)
        plan_hdr.addStretch(1)
        plan_hdr.addWidget(self.btn_open_root, 0)
        plan_hdr.addWidget(self.btn_open_last, 0)
        plan_hdr.addWidget(self.btn_refresh, 0)
        root.addLayout(plan_hdr)

        # Plan text
        self.lbl_plan = QLabel("", self)
        self.lbl_plan.setObjectName("runPlanText")
        self.lbl_plan.setWordWrap(True)
        self.lbl_plan.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        root.addWidget(self.lbl_plan, 0)

        # =========================================
        # Warnings list (compact, before tree)
        # =========================================
        self.warn_box = QWidget(self)
        self.warn_box.setObjectName("xferWarnBox")

        wb = QVBoxLayout(self.warn_box)
        wb.setContentsMargins(0, 0, 0, 0)
        wb.setSpacing(4)

        self.lbl_warn_title = QLabel(
            "Warnings",
            self.warn_box,
        )
        self.lbl_warn_title.setObjectName("subTitle")

        self.lbl_warns = QLabel("", self.warn_box)
        self.lbl_warns.setObjectName("sumLine")
        self.lbl_warns.setWordWrap(True)
        self.lbl_warns.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        wb.addWidget(self.lbl_warn_title, 0)
        wb.addWidget(self.lbl_warns, 0)

        self.warn_box.setVisible(False)
        root.addWidget(self.warn_box, 0)

        # =========================================
        # Details tree (LAST)
        # =========================================
        self.tree = QTreeWidget(self)
        self.tree.setObjectName("xferPreviewTree")
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Item", "Value"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setIndentation(14)
        root.addWidget(self.tree, 1)

        # Status label for bottom bar
        self._lbl_status = QLabel("—", self)
        self._lbl_status.setObjectName("sumLine")

        # Wiring
        self.btn_copy.clicked.connect(self._on_copy)
        self.btn_refresh.clicked.connect(self.refresh_all)
        self.btn_open_root.clicked.connect(self._on_open_root)
        self.btn_open_last.clicked.connect(self._on_open_last)


    def _wire_store(self) -> None:
        sig = getattr(self._s, "config_changed", None)
        if sig is None:
            sig = getattr(self._s, "changed", None)
        if sig is None:
            return
        try:
            sig.connect(lambda _k: self.refresh_all())
        except Exception:
            return

    # -------------------------------------------------
    # Public
    # -------------------------------------------------
    def status_label(self) -> QLabel:
        return self._lbl_status

    def set_plan_text(self, text: str) -> None:
        self._plan_text = str(text or "")
        self.lbl_plan.setText(self._plan_text)

    def refresh_all(self) -> None:
        plan, status, warns = _xfer_preview_snapshot(self._s)
        self._render(plan, status, warns)

    # -------------------------------------------------
    # Render
    # -------------------------------------------------
    def _render(
        self,
        plan: Dict[str, Any],
        status: str,
        warns: List[str],
    ) -> None:
        self._render_plan(plan, status, warns)
        self._render_warns(warns)
        self._render_tree(plan, status, warns)

    def _render_plan(
        self,
        plan: Dict[str, Any],
        status: str,
        warns: List[str],
    ) -> None:
        a = str(plan.get("city_a") or "").strip() or "—"
        b = str(plan.get("city_b") or "").strip() or "—"

        rr = str(plan.get("results_root") or "").strip() or "—"
        bs = str(plan.get("batch_size") or 0)

        splits = plan.get("splits") or []
        splits_s = ", ".join(map(str, splits)) or "—"

        calib = plan.get("calib_modes") or []
        calib_s = ", ".join(map(str, calib)) or "—"

        ok = str(status).strip().lower() == "ok"
        ready_s = "OK" if ok else "Needs attention"

        lines: List[str] = []
        lines.append(f"City A → City B:  {a} → {b}")
        lines.append(f"Results root:    {rr}")
        lines.append(f"Splits:          {splits_s}")
        lines.append(f"Calibration:     {calib_s}")
        lines.append(f"Batch size:      {bs}")
        lines.append(f"Readiness:       {ready_s}")

        txt = "\n".join(lines)
        self._plan_text = txt
        self.lbl_plan.setText(txt)

        self._set_ready(ok, len(warns))

    def _render_warns(self, warns: List[str]) -> None:
        if not warns:
            self.warn_box.setVisible(False)
            self.lbl_warns.setText("")
            return

        # compact bullet list (no table yet)
        txt = "\n".join(f"• {w}" for w in warns)
        self.lbl_warns.setText(txt)
        self.warn_box.setVisible(True)

    def _set_ready(self, ok: bool, n_warn: int) -> None:
        if ok:
            self._lbl_status.setText("OK")
            self.chip_ready.setText("OK")
            self.chip_ready.set_kind("ok")

            self.chip_warn.setText("0 warn")
            self.chip_warn.set_kind("off")
            return

        self._lbl_status.setText("Needs attention")

        n = max(1, int(n_warn))
        self.chip_ready.setText("Fix")
        self.chip_ready.set_kind("warn")

        self.chip_warn.setText(f"{n} warn")
        self.chip_warn.set_kind("warn")

    def _render_tree(
        self,
        plan: Dict[str, Any],
        status: str,
        warns: List[str],
    ) -> None:
        self.tree.clear()

        ready = "OK" if str(status) == "ok" else "Needs attention"

        sec = self._section("Readiness")
        self._kv(sec, "Status", ready)
        self._kv(sec, "Warnings", str(len(warns)))

        sec = self._section("Inputs")
        self._kv(sec, "City A", str(plan.get("city_a") or ""))
        self._kv(sec, "City B", str(plan.get("city_b") or ""))
        self._kv(sec, "Results root", str(plan.get("results_root") or ""))
        self._kv(sec, "Splits", _fmt_list(plan.get("splits")))
        self._kv(sec, "Calibration", _fmt_list(plan.get("calib_modes")))
        self._kv(sec, "Batch size", str(plan.get("batch_size") or 0))
        self._kv(
            sec,
            "Rescale to source",
            str(bool(plan.get("rescale_to_source"))),
        )

        sec = self._section("Strategy & warm-start")
        self._kv(sec, "Strategies", _fmt_list(plan.get("strategies")))
        self._kv(sec, "Rescale modes", _fmt_list(plan.get("rescale_modes")))
        self._kv(sec, "Warm split", str(plan.get("warm_split") or ""))
        self._kv(sec, "Warm samples", str(plan.get("warm_samples") or 0))
        self._kv(sec, "Warm frac", str(plan.get("warm_frac") or 0.0))
        self._kv(sec, "Warm epochs", str(plan.get("warm_epochs") or 0))
        self._kv(sec, "Warm lr", str(plan.get("warm_lr") or 0.0))
        self._kv(sec, "Warm seed", str(plan.get("warm_seed") or 0))

        sec = self._section("Outputs & alignment")
        self._kv(sec, "Write JSON", str(bool(plan.get("write_json"))))
        self._kv(sec, "Write CSV", str(bool(plan.get("write_csv"))))
        self._kv(sec, "Prefer tuned", str(bool(plan.get("prefer_tuned"))))
        self._kv(sec, "Align policy", str(plan.get("align_policy") or ""))
        self._kv(
            sec,
            "Reorder dynamic",
            _fmt_opt_bool(plan.get("allow_reorder_dynamic")),
        )
        self._kv(
            sec,
            "Reorder future",
            _fmt_opt_bool(plan.get("allow_reorder_future")),
        )
        self._kv(
            sec,
            "Interval target",
            str(plan.get("interval_target")),
        )
        self._kv(
            sec,
            "Endpoint",
            str(plan.get("load_endpoint") or ""),
        )
        self._kv(
            sec,
            "Quantiles override",
            _fmt_list(plan.get("quantiles_override")),
        )

        sec = self._section("Results & view")
        self._kv(sec, "View kind", str(plan.get("view_kind") or ""))
        self._kv(sec, "View split", str(plan.get("view_split") or ""))

        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setExpanded(True)

    # -------------------------------------------------
    # Actions
    # -------------------------------------------------
    def _on_copy(self) -> None:
        txt = (self._plan_text or "").strip()
        if not txt:
            txt = (self.lbl_plan.text() or "").strip()
        if not txt:
            self.toast.emit("Nothing to copy.")
            return
        QApplication.clipboard().setText(txt)
        self.toast.emit("Run plan copied.")

    def _on_open_root(self) -> None:
        rr = str(self._s.get("results_root", "") or "").strip()
        if rr:
            _open_path(rr)

    def _on_open_last(self) -> None:
        last = str(self._s.get("xfer.last_output", "") or "").strip()
        if not last:
            # fallback: open results root
            rr = str(self._s.get("results_root", "") or "").strip()
            _open_path(rr)
            return
        _open_path(_dir_of(last))

    # -------------------------------------------------
    # Tree helpers
    # -------------------------------------------------
    def _section(self, title: str) -> QTreeWidgetItem:
        it = QTreeWidgetItem([title, ""])
        it.setFirstColumnSpanned(True)

        f = QFont(it.font(0))
        f.setBold(True)
        it.setFont(0, f)

        self.tree.addTopLevelItem(it)
        it.setExpanded(True)
        return it

    def _kv(
        self,
        parent: QTreeWidgetItem,
        key: str,
        val: str,
    ) -> None:
        QTreeWidgetItem(parent, [key, val])


# -------------------------------------------------
# Snapshot logic
# -------------------------------------------------
def _xfer_preview_snapshot(
    s: GeoConfigStore,
) -> Tuple[Dict[str, Any], str, List[str]]:
    plan: Dict[str, Any] = {}

    plan["results_root"] = _s_str(s.get("results_root", None))
    plan["city_a"] = _s_str(s.get("xfer.city_a", None))
    plan["city_b"] = _s_str(s.get("xfer.city_b", None))

    plan["splits"] = list(s.get("xfer.splits", []) or [])
    plan["calib_modes"] = list(s.get("xfer.calib_modes", []) or [])

    plan["batch_size"] = int(s.get("xfer.batch_size", 0) or 0)
    plan["rescale_to_source"] = bool(
        s.get("xfer.rescale_to_source", False)
    )

    plan["quantiles_override"] = s.get("xfer.quantiles_override", None)
    plan["write_json"] = bool(s.get("xfer.write_json", True))
    plan["write_csv"] = bool(s.get("xfer.write_csv", True))
    plan["prefer_tuned"] = bool(s.get("xfer.prefer_tuned", True))
    plan["align_policy"] = _s_str(
        s.get("xfer.align_policy", "align_by_name_pad")
    )
    plan["allow_reorder_dynamic"] = s.get(
        "xfer.allow_reorder_dynamic", None
    )
    plan["allow_reorder_future"] = s.get(
        "xfer.allow_reorder_future", None
    )
    plan["interval_target"] = float(
        s.get("xfer.interval_target", 0.80) or 0.80
    )
    plan["load_endpoint"] = _s_str(s.get("xfer.load_endpoint", "serve"))

    plan["strategies"] = s.get("xfer.strategies", None)
    plan["rescale_modes"] = s.get("xfer.rescale_modes", None)

    plan["warm_split"] = _s_str(s.get("xfer.warm_split", "train"))
    plan["warm_samples"] = int(s.get("xfer.warm_samples", 20000) or 0)
    plan["warm_frac"] = float(s.get("xfer.warm_frac", 0.0) or 0.0)
    plan["warm_epochs"] = int(s.get("xfer.warm_epochs", 3) or 0)
    plan["warm_lr"] = float(s.get("xfer.warm_lr", 1e-4) or 0.0)
    plan["warm_seed"] = int(s.get("xfer.warm_seed", 0) or 0)

    plan["view_kind"] = _s_str(s.get("xfer.view_kind", "calib_panel"))
    plan["view_split"] = _s_str(s.get("xfer.view_split", "val"))

    warns: List[str] = []
    status = "ok"

    if not plan["city_a"]:
        warns.append("Missing City A (xfer.city_a).")
    if not plan["city_b"]:
        warns.append("Missing City B (xfer.city_b).")
    if not plan["results_root"]:
        warns.append("Missing results_root (Stage-1/2/xfer).")

    splits = plan.get("splits") or []
    if not splits:
        warns.append("No splits selected (train/val/test).")

    calib = plan.get("calib_modes") or []
    if not calib:
        warns.append("No calibration mode selected.")

    if int(plan.get("batch_size") or 0) <= 0:
        warns.append("Batch size must be > 0.")

    strats = plan.get("strategies") or []
    if not strats:
        warns.append("No transfer strategies selected.")
    else:
        if "warm" in set(strats):
            if int(plan.get("warm_samples") or 0) <= 0:
                warns.append("Warm: warm_samples <= 0.")
            if int(plan.get("warm_epochs") or 0) <= 0:
                warns.append("Warm: warm_epochs <= 0.")
            if float(plan.get("warm_lr") or 0.0) <= 0.0:
                warns.append("Warm: warm_lr <= 0.")

    if warns:
        status = "warn"

    return plan, status, warns


def _s_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v or "").strip()


def _fmt_list(v: Any) -> str:
    if not v:
        return "—"
    try:
        return ", ".join(str(x) for x in v)
    except Exception:
        return "—"


def _fmt_opt_bool(v: Any) -> str:
    if v is None:
        return "Auto"
    if bool(v) is True:
        return "Allow"
    return "Block"

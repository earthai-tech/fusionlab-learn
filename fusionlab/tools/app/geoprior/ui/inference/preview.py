# geoprior/ui/inference/preview.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import os
from typing import Dict, Optional

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
    QHeaderView
)

from ..icon_utils import try_icon
from ...config.store import GeoConfigStore
from .plan import build_plan_text
from .runtime_snapshot import InferRuntimeSnapshot


__all__ = ["InferencePreviewPanel"]


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


def _open_path(p: str) -> None:
    p = str(p or "").strip()
    if not _exists(p):
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(p))


def _dir_of(p: str) -> str:
    p = str(p or "").strip()
    if not p:
        return ""
    ap = _abspath(p)
    if os.path.isdir(ap):
        return ap
    return os.path.dirname(ap)


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


class InferencePreviewPanel(QFrame):
    """
    Right [D]: resolved plan + readiness summary.

    Owns:
    - runtime validation (paths)
    - emits InferRuntimeSnapshot

    UI (Train-like)
    ---------------
    - Top readiness strip (chips + quick actions)
    - "Resolved plan" text (selectable) + copy
    - Details table (QTreeWidget) as deep view
    """

    runtime_snapshot_changed = pyqtSignal(object)
    toast = pyqtSignal(str)

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self._last_outputs: Dict[str, str] = {}

        self.setObjectName("runPreviewPanel")
        self.setFrameShape(QFrame.NoFrame)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._build_ui()
        self._wire_store()

    # ---------------------------------------------------------
    # Icons
    # ---------------------------------------------------------
    def _std_icon(self, sp: QStyle.StandardPixmap):
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

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
    
        # ======================================================
        # Header row: "Run preview" + chips + copy
        # ======================================================
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)
    
        t = QLabel("Run preview")
        t.setObjectName("subTitle")
    
        self.chip_ready = _Chip("—", kind="off", parent=self)
        self.chip_warn = _Chip("0 warn", kind="off", parent=self)
    
        self.btn_copy_plan = self._mk_icon_btn(
            "Copy resolved plan",
            "copy.svg",
            QStyle.SP_DialogSaveButton,
        )
    
        hdr.addWidget(t, 0)
        hdr.addSpacing(6)
        hdr.addWidget(self.chip_ready, 0)
        hdr.addWidget(self.chip_warn, 0)
        hdr.addStretch(1)
        hdr.addWidget(self.btn_copy_plan, 0)
        root.addLayout(hdr)
    
        # Small hint line (optional, modern + calm)
        self.lbl_hint = QLabel(
            "Checks runtime artifacts and shows resolved run setup.",
            self,
        )
        self.lbl_hint.setObjectName("sumLine")
        self.lbl_hint.setWordWrap(True)
        root.addWidget(self.lbl_hint, 0)
    
        # ======================================================
        # Resolved plan: title + runtime actions
        # ======================================================
        plan_hdr = QHBoxLayout()
        plan_hdr.setContentsMargins(0, 0, 0, 0)
        plan_hdr.setSpacing(8)
    
        plan_t = QLabel("Resolved plan")
        plan_t.setObjectName("subTitle")
    
        self.btn_open_model = self._mk_icon_btn(
            "Open model folder",
            "folder_open.svg",
            QStyle.SP_DirOpenIcon,
        )
        self.btn_open_outputs = self._mk_icon_btn(
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
        plan_hdr.addWidget(self.btn_open_model, 0)
        plan_hdr.addWidget(self.btn_open_outputs, 0)
        plan_hdr.addWidget(self.btn_refresh, 0)
        root.addLayout(plan_hdr)
    
        self.lbl_plan = QLabel("")
        self.lbl_plan.setObjectName("runPlanText")
        self.lbl_plan.setWordWrap(True)
        self.lbl_plan.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        root.addWidget(self.lbl_plan, 0)
    
        # ======================================================
        # Deep details table
        # ======================================================
        self._tree = QTreeWidget(self)
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Item", "Value"])
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setIndentation(14)

        # Fill remaining space (removes "space below").
        self._tree.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        # Horizontal scroll when long values (paths) appear.
        self._tree.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        self._tree.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        self._tree.setTextElideMode(Qt.ElideNone)

        hdr = self._tree.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(
            0,
            QHeaderView.ResizeToContents,
        )
        hdr.setSectionResizeMode(
            1,
            QHeaderView.ResizeToContents,
        )

        root.addWidget(self._tree, 1)
        root.setStretchFactor(self._tree, 1)

        # Status label (used by bottom bar)
        self._lbl_status = QLabel("—", self)
        self._lbl_status.setObjectName("sumLine")
    
        # Wiring
        self.btn_copy_plan.clicked.connect(self._on_copy)
        self.btn_open_model.clicked.connect(self._on_open_model)
        self.btn_open_outputs.clicked.connect(self.open_last_outputs)
        self.btn_refresh.clicked.connect(self.refresh)
    
        self.refresh()
    
    def _resolved_plan_text(
        self,
        *,
        dataset_key: str,
        use_future: bool,
        batch: int,
        model_path: str,
        manifest_path: str,
        inputs_npz: str,
        targets_npz: str,
        calib_path: str,
        use_source_calib: bool,
        fit_calib: bool,
        include_gwl: bool,
        plots: bool,
        run_dir: str,
    ) -> str:
        ds = str(dataset_key or "—")
        fut = "ON" if use_future else "OFF"
    
        model_s = _abspath(model_path) if model_path else "(empty)"
        mani_s = _abspath(manifest_path) if manifest_path else "(auto)"
    
        inp_s = _abspath(inputs_npz) if inputs_npz else "(none)"
        tgt_s = _abspath(targets_npz) if targets_npz else "(none)"
        cal_s = _abspath(calib_path) if calib_path else "(none)"
    
        rd = _abspath(run_dir) if run_dir else "(unknown)"
        gwl = "ON" if include_gwl else "OFF"
        pl = "ON" if plots else "OFF"
    
        src = "ON" if use_source_calib else "OFF"
        fit = "ON" if fit_calib else "OFF"
    
        lines = []
        lines.append(f"• Dataset: {ds}  (future={fut})")
        lines.append(f"• Batch size: {int(batch)}")
        lines.append(f"• Model: {model_s}")
        lines.append(f"• Stage-1 manifest: {mani_s}")
        lines.append(f"• Run dir: {rd}")
    
        if ds == "custom" and not use_future:
            lines.append(f"• Inputs NPZ: {inp_s}")
            lines.append(f"• Targets NPZ: {tgt_s}")
    
        lines.append(
            f"• Calibration: source={src}, fit={fit}, file={cal_s}"
        )
        lines.append(f"• Outputs: GWL={gwl}, plots={pl}")
    
        return "\n".join(lines)
    
    
    def _on_copy(self) -> None:
        txt = (self.lbl_plan.text() or "").strip()
        if not txt:
            self.toast.emit("Nothing to copy.")
            return
        QApplication.clipboard().setText(txt)
        self.toast.emit("Resolved plan copied.")


    def _wire_store(self) -> None:
        sig = getattr(self._store, "config_changed", None)
        if sig is None:
            sig = getattr(self._store, "changed", None)
        if sig is None:
            return
        try:
            sig.connect(lambda _k: self.refresh())
        except Exception:
            return

    # ---------------------------------------------------------
    # Public
    # ---------------------------------------------------------
    def status_label(self) -> QLabel:
        return self._lbl_status

    def plan_summary(self) -> str:
        return build_plan_text(self._store)

    def set_last_outputs(self, out: Dict[str, str]) -> None:
        d = dict(out or {})
        if "forecast_csv" in d and "csv_future_path" not in d:
            d["csv_future_path"] = d.get("forecast_csv", "")
        self._last_outputs = d
        self.refresh()

    def last_outputs(self) -> Dict[str, str]:
        return dict(self._last_outputs)

    def last_run_dir(self) -> str:
        return str(self._last_outputs.get("run_dir", "") or "")

    def open_last_outputs(self) -> None:
        p = self.last_run_dir().strip()
        if _exists(p):
            _open_path(p)

    def copy_plan_to_clipboard(self) -> None:
        txt = (self.plan_summary() or "").strip()
        if not txt:
            self.toast.emit("Nothing to copy.")
            return
        QApplication.clipboard().setText(txt)
        self.toast.emit("Run plan copied.")

    # ---------------------------------------------------------
    # Refresh / compute
    # ---------------------------------------------------------
    def refresh(self) -> None:
        tab = self.parent()
        while tab is not None and not hasattr(tab, "center"):
            tab = tab.parent()

        if tab is None:
            self._tree.clear()
            self.lbl_plan.setText("")
            self._set_ready(False, 0)
            return

        c = tab.center  # InferenceCenterPanel

        model_p = c.inf_model_edit.text().strip()
        mani_p = c.inf_manifest_edit.text().strip()

        dkey = c.cmb_inf_dataset.currentData() or "test"
        use_future = c.chk_inf_use_future.isChecked()

        inputs_p = c.inf_inputs_edit.text().strip()
        targets_p = c.inf_targets_edit.text().strip()

        cov = float(c.sp_inf_cov.value())
        mode = c.cmb_calib_mode.currentData() or "none"
        temp = float(c.sp_calib_temp.value())

        src_cal = c.chk_inf_use_source_calib.isChecked()
        fit_cal = c.chk_inf_fit_calib.isChecked()
        cal_p = c.inf_calib_edit.text().strip()

        inc_gwl = c.chk_inf_include_gwl.isChecked()
        plots = c.chk_inf_plots.isChecked()
        bsz = int(c.sp_inf_batch.value())

        run_dir = _dir_of(model_p)

        warn: list[str] = []

        if not model_p:
            warn.append("- Missing model path.")
        elif not _exists(model_p):
            warn.append("- Model path does not exist.")

        if (dkey == "custom") and (not use_future):
            if not inputs_p:
                warn.append("- Custom inputs NPZ is required.")
            elif not _exists(inputs_p):
                warn.append("- Inputs NPZ does not exist.")

            if targets_p and (not _exists(targets_p)):
                warn.append("- Targets NPZ does not exist.")

        if mani_p and (not _exists(mani_p)):
            warn.append("- Manifest path does not exist.")

        if cal_p and (not _exists(cal_p)):
            warn.append("- Calibrator file does not exist.")

        # Plan text (store-driven)
        txt = self._resolved_plan_text(
            dataset_key=str(dkey),
            use_future=bool(use_future),
            batch=int(bsz),
            model_path=str(model_p),
            manifest_path=str(mani_p),
            inputs_npz=str(inputs_p),
            targets_npz=str(targets_p),
            calib_path=str(cal_p),
            use_source_calib=bool(src_cal),
            fit_calib=bool(fit_cal),
            include_gwl=bool(inc_gwl),
            plots=bool(plots),
            run_dir=str(run_dir),
        )
        self.lbl_plan.setText(txt)

        # Chips + status
        ok = not bool(warn)
        self._set_ready(ok, len(warn))

        # Build tree
        self._tree.clear()

        ready = "OK" if ok else "Needs attention"

        sec = self._section("Readiness")
        self._kv(sec, "Status", ready)
        self._kv(sec, "Warnings", str(len(warn)))

        sec = self._section("Inputs")
        self._kv(sec, "Dataset", f"{dkey} (future={use_future})")
        self._kv(sec, "Batch size", str(bsz))
        self._kv(sec, "Model", _abspath(model_p) or "(empty)")
        self._kv(sec, "Stage-1 manifest", _abspath(mani_p) or "(auto)")
        self._kv(sec, "Run dir", run_dir or "(unknown)")

        sec = self._section("Uncertainty (store)")
        self._kv(sec, "Interval level", f"{cov:.3f}")
        self._kv(sec, "Mode", str(mode))
        self._kv(sec, "Temperature", f"{temp:.3f}")
        self._kv(sec, "Cross penalty", f"{c.sp_cross_pen.value():.4f}")
        self._kv(
            sec,
            "Cross margin",
            f"{c.sp_cross_margin.value():.4f}",
        )

        sec = self._section("Calibration (runtime)")
        self._kv(sec, "Use source", str(src_cal))
        self._kv(sec, "Fit on val", str(fit_cal))
        self._kv(sec, "Explicit file", _abspath(cal_p) or "(none)")

        sec = self._section("Outputs (runtime)")
        self._kv(sec, "Include GWL", str(inc_gwl))
        self._kv(sec, "Generate plots", str(plots))

        if warn:
            sec = self._section("Warnings")
            for w in warn:
                self._kv(sec, "", w)

        snap = InferRuntimeSnapshot(
            mode="forecast" if use_future else "evaluate",
            dataset_key=str(dkey),
            use_future=bool(use_future),
            model_path=str(model_p),
            manifest_path=str(mani_p),
            inputs_npz=str(inputs_p),
            targets_npz=str(targets_p),
            has_targets=bool(targets_p) and _exists(targets_p),
            warnings=list(warn),
            last_run_dir=self.last_run_dir(),
            last_eval_csv=str(
                self._last_outputs.get("csv_eval_path", "")
            ),
            last_future_csv=str(
                self._last_outputs.get("csv_future_path", "")
            ),
            last_json=str(
                self._last_outputs.get(
                    "inference_summary_json",
                    "",
                )
            ),
        )
        self.runtime_snapshot_changed.emit(snap)
        
        try:
            self._tree.resizeColumnToContents(0)
            self._tree.resizeColumnToContents(1)
        except Exception:
            pass

    def _set_ready(self, ok: bool, n_warn: int) -> None:
        if ok:
            self._lbl_status.setText("OK")
            self._set_chip(self.chip_ready, "OK", "ok")
            self._set_chip(self.chip_warn, "0 warn", "off")
            return
    
        self._lbl_status.setText("Needs attention")
    
        # "Fix" should look warn; count chip also warn
        n = max(1, int(n_warn))
        self._set_chip(self.chip_ready, "Fix", "warn")
        self._set_chip(self.chip_warn, f"{n} warn", "warn")

    def _set_chip(self, chip: QLabel, text: str, kind: str) -> None:
        chip.setText(str(text))
        chip.setProperty("kind", str(kind))
        chip.style().unpolish(chip)
        chip.style().polish(chip)

    # ---------------------------------------------------------
    # Actions
    # ---------------------------------------------------------

    def _on_open_model(self) -> None:
        tab = self.parent()
        while tab is not None and not hasattr(tab, "center"):
            tab = tab.parent()
        if tab is None:
            return
        c = tab.center
        model_p = c.inf_model_edit.text().strip()
        _open_path(_dir_of(model_p))

    # ---------------------------------------------------------
    # Tree helpers
    # ---------------------------------------------------------
    def _section(self, title: str) -> QTreeWidgetItem:
        it = QTreeWidgetItem([title, ""])
        it.setFirstColumnSpanned(True)

        f = QFont(it.font(0))
        f.setBold(True)
        it.setFont(0, f)

        self._tree.addTopLevelItem(it)
        it.setExpanded(True)
        return it

    def _kv(
        self,
        parent: QTreeWidgetItem,
        key: str,
        val: str,
    ) -> None:
        QTreeWidgetItem(parent, [key, val])

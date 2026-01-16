# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.setup.cards.training_basics

Training basics card (v3.2 store-driven).

- Core optimisation knobs (epochs / batch / lr)
- Runtime toggles (SavedModel, in-memory, debug)
- Auditing selector (all/off/custom list)
- Stage-1 workflow (expander)
- Device options button (dialog with OK/Cancel rollback)
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt5.QtCore import Qt, QSignalBlocker
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .base import CardBase
from ..bindings import Binder
from ....config.store import GeoConfigStore
from ....device_options import DeviceOptionsWidget


class _Expander:
    def __init__(
        self,
        title: str,
        *,
        parent: QWidget,
    ) -> None:
        self.btn = QToolButton(parent)
        self.btn.setText(str(title))
        self.btn.setCheckable(True)
        self.btn.setChecked(False)
        self.btn.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.btn.setArrowType(Qt.RightArrow)

        self.body = QWidget(parent)
        self.body.setVisible(False)

        self.body_lay = QGridLayout(self.body)
        self.body_lay.setContentsMargins(8, 6, 8, 6)
        self.body_lay.setHorizontalSpacing(10)
        self.body_lay.setVerticalSpacing(6)

        def _toggle(on: bool) -> None:
            self.body.setVisible(bool(on))
            self.btn.setArrowType(
                Qt.DownArrow if on else Qt.RightArrow
            )

        self.btn.toggled.connect(_toggle)


class TrainingBasicsCard(CardBase):
    """
    Modern Training basics card.

    Parameters
    ----------
    store : GeoConfigStore
        Single source of truth.
    binder : Binder
        Widget <-> store bindings.
    parent : QWidget, optional
        Parent widget.
    """

    _DEVICE_KEYS = (
        "tf_device_mode",
        "tf_gpu_allow_growth",
        "tf_intra_threads",
        "tf_inter_threads",
        "tf_gpu_memory_limit_mb",
    )

    _BADGE_KEYS = (
        "epochs",
        "batch_size",
        "learning_rate",
        "debug",
        "use_tf_savedmodel",
        "use_in_memory_model",
    )

    _AUDIT_KEY = "audit_stages"

    _STAGE1_KEYS = (
        "clean_stage1_dir",
        "stage1_auto_reuse_if_match",
        "stage1_force_rebuild_if_mismatch",
        "build_future_npz",
    )

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        binder: Binder,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            section_id="train",
            title="Training basics",
            subtitle=(
                "Core optimisation loop, runtime switches, "
                "and Stage-1 workflow behaviour."
            ),
            parent=parent,
        )
        self.store = store
        self.binder = binder

        self._build()
        self._wire()
        self._refresh_all()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build(self) -> None:
        body = self.body_layout()

        hint = QLabel(
            "Tip: these settings affect Train and Tune. "
            "Device options apply to TF execution.",
            self,
        )
        hint.setWordWrap(True)
        hint.setObjectName("setupCardSubtitle")
        body.addWidget(hint, 0)

        top = QWidget(self)
        top_lay = QGridLayout(top)
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setHorizontalSpacing(10)
        top_lay.setVerticalSpacing(8)

        body.addWidget(top, 0)

        self.grp_opt = QGroupBox("Optimization", top)
        opt_l = QGridLayout(self.grp_opt)
        opt_l.setContentsMargins(10, 10, 10, 10)
        opt_l.setHorizontalSpacing(10)
        opt_l.setVerticalSpacing(8)

        self.grp_run = QGroupBox("Runtime & outputs", top)
        run_l = QGridLayout(self.grp_run)
        run_l.setContentsMargins(10, 10, 10, 10)
        run_l.setHorizontalSpacing(10)
        run_l.setVerticalSpacing(8)

        top_lay.addWidget(self.grp_opt, 0, 0)
        top_lay.addWidget(self.grp_run, 0, 1)
        top_lay.setColumnStretch(0, 1)
        top_lay.setColumnStretch(1, 1)

        # ---- Optimization fields
        self.sp_epochs = QSpinBox(self.grp_opt)
        self.sp_epochs.setRange(1, 100000)
        self.sp_epochs.setSingleStep(5)

        self.sp_batch = QSpinBox(self.grp_opt)
        self.sp_batch.setRange(1, 8192)
        self.sp_batch.setSingleStep(8)

        self.sp_lr = QDoubleSpinBox(self.grp_opt)
        self.sp_lr.setRange(1e-10, 10.0)
        self.sp_lr.setDecimals(10)
        self.sp_lr.setSingleStep(1e-4)

        opt_l.addWidget(QLabel("Epochs:", self.grp_opt), 0, 0)
        opt_l.addWidget(self.sp_epochs, 0, 1)

        opt_l.addWidget(QLabel("Batch size:", self.grp_opt), 1, 0)
        opt_l.addWidget(self.sp_batch, 1, 1)

        opt_l.addWidget(QLabel("Learning rate:", self.grp_opt), 2, 0)
        opt_l.addWidget(self.sp_lr, 2, 1)

        lr_note = QLabel(
            "Use small LR for stability; "
            "1e-3 to 1e-4 is typical.",
            self.grp_opt,
        )
        lr_note.setWordWrap(True)
        lr_note.setObjectName("setupCardSubtitle")
        opt_l.addWidget(lr_note, 3, 0, 1, 2)

        # ---- Runtime toggles (2x2 grid)
        self.chk_eval = QCheckBox(
            "Evaluate after training",
            self.grp_run,
        )
        self.chk_saved = QCheckBox(
            "Export TF SavedModel",
            self.grp_run,
        )
        self.chk_mem = QCheckBox(
            "Keep model in memory",
            self.grp_run,
        )
        self.chk_debug = QCheckBox(
            "Debug (verbose logs)",
            self.grp_run,
        )

        run_l.addWidget(self.chk_eval, 0, 0)
        run_l.addWidget(self.chk_saved, 0, 1)
        run_l.addWidget(self.chk_mem, 1, 0)
        run_l.addWidget(self.chk_debug, 1, 1)

        # ---- Audit mode + custom list
        self.cmb_audit = QComboBox(self.grp_run)
        self.cmb_audit.addItem("All stages", "*")
        self.cmb_audit.addItem("Off", "")
        self.cmb_audit.addItem("Custom list", "__custom__")

        self.ed_audit = QPlainTextEdit(self.grp_run)
        self.ed_audit.setPlaceholderText(
            "One stage per line\n"
            "e.g.\n"
            "stage1\n"
            "stage2"
        )
        self.ed_audit.setFixedHeight(76)

        run_l.addWidget(QLabel("Auditing:", self.grp_run), 2, 0)
        run_l.addWidget(self.cmb_audit, 2, 1)
        run_l.addWidget(self.ed_audit, 3, 0, 1, 2)

        # ---- Device preview + button
        self.lbl_device = QLabel("", self.grp_run)
        self.lbl_device.setWordWrap(True)
        self.lbl_device.setObjectName("setupCardSubtitle")
        run_l.addWidget(self.lbl_device, 4, 0, 1, 2)

        row_btn = QWidget(self.grp_run)
        row_btn_l = QHBoxLayout(row_btn)
        row_btn_l.setContentsMargins(0, 0, 0, 0)
        row_btn_l.setSpacing(8)

        self.btn_device = QPushButton("Device options…", row_btn)
        self.btn_device.setObjectName("miniAction")
        self.btn_device.setCursor(Qt.PointingHandCursor)
        self.btn_device.setIcon(
            self.style().standardIcon(QStyle.SP_ComputerIcon)
        )

        row_btn_l.addWidget(self.btn_device, 0)
        row_btn_l.addStretch(1)

        run_l.addWidget(row_btn, 5, 0, 1, 2)

        # ---- Stage-1 workflow (expander)
        self.exp_stage1 = _Expander(
            "Stage-1 workflow",
            parent=self,
        )
        body.addWidget(self.exp_stage1.btn, 0)
        body.addWidget(self.exp_stage1.body, 0)

        s1 = self.exp_stage1.body_lay

        self.chk_clean = QCheckBox(
            "Clean Stage-1 directory before run",
            self.exp_stage1.body,
        )
        self.chk_reuse = QCheckBox(
            "Auto reuse if config matches",
            self.exp_stage1.body,
        )
        self.chk_force = QCheckBox(
            "Force rebuild if mismatch",
            self.exp_stage1.body,
        )
        self.chk_future = QCheckBox(
            "Pre-build future_* NPZ (Stage-3)",
            self.exp_stage1.body,
        )

        s1.addWidget(self.chk_clean, 0, 0, 1, 2)
        s1.addWidget(self.chk_reuse, 1, 0, 1, 2)
        s1.addWidget(self.chk_force, 2, 0, 1, 2)
        s1.addWidget(self.chk_future, 3, 0, 1, 2)

        s1_note = QLabel(
            "These options control how Stage-1 prepares and "
            "reuses preprocessing artifacts between runs.",
            self.exp_stage1.body,
        )
        s1_note.setWordWrap(True)
        s1_note.setObjectName("setupCardSubtitle")
        s1.addWidget(s1_note, 4, 0, 1, 2)

        # ---- Header action (compact)
        self.act_device = self.add_action(
            text="Device…",
            tip="Open device / threading options",
            icon=QStyle.SP_ComputerIcon,
        )

        # ---- Bindings (store-backed)
        self.binder.bind_spin_box("epochs", self.sp_epochs)
        self.binder.bind_spin_box("batch_size", self.sp_batch)
        self.binder.bind_double_spin_box("learning_rate", self.sp_lr)

        self.binder.bind_checkbox(
            "evaluate_training",
            self.chk_eval,
        )
        self.binder.bind_checkbox(
            "use_tf_savedmodel",
            self.chk_saved,
        )
        self.binder.bind_checkbox(
            "use_in_memory_model",
            self.chk_mem,
        )
        self.binder.bind_checkbox("debug", self.chk_debug)

        self.binder.bind_checkbox(
            "clean_stage1_dir",
            self.chk_clean,
        )
        self.binder.bind_checkbox(
            "stage1_auto_reuse_if_match",
            self.chk_reuse,
        )
        self.binder.bind_checkbox(
            "stage1_force_rebuild_if_mismatch",
            self.chk_force,
        )
        self.binder.bind_checkbox(
            "build_future_npz",
            self.chk_future,
        )

    # -----------------------------------------------------------------
    # Wiring / refresh
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.btn_device.clicked.connect(
            self._open_device_dialog
        )
        self.act_device.clicked.connect(
            self._open_device_dialog
        )

        self.cmb_audit.currentIndexChanged.connect(
            self._commit_audit_mode
        )
        self.ed_audit.textChanged.connect(
            self._commit_audit_text
        )

        self.store.config_changed.connect(
            self._on_store_changed
        )
        self.store.config_replaced.connect(
            self._on_store_replaced
        )

    def _on_store_replaced(self, _cfg: object) -> None:
        self._refresh_all()

    def _on_store_changed(self, keys_obj: object) -> None:
        try:
            keys = set(keys_obj or [])
        except Exception:
            keys = set()

        watch = set(self._BADGE_KEYS)
        watch.add(self._AUDIT_KEY)
        watch |= set(self._DEVICE_KEYS)
        watch |= set(self._STAGE1_KEYS)

        if keys & watch:
            self._sync_badges()
            self._sync_device_preview()
            self._sync_audit_controls()

    def _refresh_all(self) -> None:
        self._sync_badges()
        self._sync_device_preview()
        self._sync_audit_controls()

    # -----------------------------------------------------------------
    # Badges / preview
    # -----------------------------------------------------------------
    def _sync_badges(self) -> None:
        cfg = self.store.cfg

        self.badge(
            "epochs",
            text=f"E {int(cfg.epochs)}",
            tip="Epochs",
        )
        self.badge(
            "batch",
            text=f"B {int(cfg.batch_size)}",
            tip="Batch size",
        )

        lr = float(cfg.learning_rate)
        lr_txt = self._fmt_lr(lr)
        self.badge(
            "lr",
            text=f"LR {lr_txt}",
            tip="Learning rate",
        )

        lab_dbg = self.badge(
            "dbg",
            text="Debug",
            accent="warn",
            tip="Verbose logs enabled",
        )
        lab_dbg.setVisible(bool(cfg.debug))

        lab_sm = self.badge(
            "sm",
            text="SavedModel",
            tip="TF SavedModel export",
        )
        lab_sm.setVisible(bool(cfg.use_tf_savedmodel))

        lab_mem = self.badge(
            "mem",
            text="In-mem",
            tip="Keep model object in memory",
        )
        lab_mem.setVisible(bool(cfg.use_in_memory_model))

    def _sync_device_preview(self) -> None:
        cfg = self.store.cfg

        mode = (cfg.tf_device_mode or "auto").strip()
        intra = cfg.tf_intra_threads
        inter = cfg.tf_inter_threads
        lim = cfg.tf_gpu_memory_limit_mb
        growth = bool(cfg.tf_gpu_allow_growth)

        thr = []
        if intra is not None:
            thr.append(f"intra={int(intra)}")
        if inter is not None:
            thr.append(f"inter={int(inter)}")
        thr_txt = ", ".join(thr) if thr else "default"

        gpu = []
        gpu.append("growth" if growth else "no-growth")
        if lim is not None:
            gpu.append(f"cap={int(lim)}MB")
        gpu_txt = ", ".join(gpu)

        self.lbl_device.setText(
            f"Device: {mode}  ·  Threads: {thr_txt}  ·  GPU: {gpu_txt}"
        )

    @staticmethod
    def _fmt_lr(v: float) -> str:
        if v == 0.0:
            return "0"
        if (abs(v) < 1e-3) or (abs(v) >= 1.0):
            return f"{v:.1e}"
        return f"{v:.4f}".rstrip("0").rstrip(".")

    # -----------------------------------------------------------------
    # Audit controls
    # -----------------------------------------------------------------
    def _sync_audit_controls(self) -> None:
        val = self.store.get(self._AUDIT_KEY, "*")

        # Normalize modes:
        #   "*" -> all
        #   ""/None/[] -> off
        #   list/tuple -> custom
        is_all = (val == "*")
        is_off = (val is None) or (val == "") or (val == [])
        is_custom = isinstance(val, (list, tuple))

        with QSignalBlocker(self.cmb_audit):
            if is_all:
                self._set_combo_data(self.cmb_audit, "*")
            elif is_off:
                self._set_combo_data(self.cmb_audit, "")
            else:
                self._set_combo_data(self.cmb_audit, "__custom__")

        custom_on = (
            self.cmb_audit.currentData() == "__custom__"
        )

        self.ed_audit.setEnabled(bool(custom_on))

        if is_custom:
            lines = [str(x).strip() for x in val]
            txt = "\n".join([x for x in lines if x])
        else:
            txt = ""

        with QSignalBlocker(self.ed_audit):
            self.ed_audit.setPlainText(txt)

    def _commit_audit_mode(self, _idx: int) -> None:
        data = self.cmb_audit.currentData()

        if data == "*":
            self.store.patch({self._AUDIT_KEY: "*"})
            return

        if data == "":
            self.store.patch({self._AUDIT_KEY: ""})
            return

        # custom
        self._commit_audit_text()

    def _commit_audit_text(self) -> None:
        if self.cmb_audit.currentData() != "__custom__":
            return

        raw = self.ed_audit.toPlainText() or ""
        items = [
            ln.strip()
            for ln in raw.splitlines()
            if ln.strip()
        ]
        self.store.patch({self._AUDIT_KEY: items})

    @staticmethod
    def _set_combo_data(cmb: QComboBox, data: Any) -> None:
        for i in range(cmb.count()):
            if cmb.itemData(i) == data:
                cmb.setCurrentIndex(i)
                return

    # -----------------------------------------------------------------
    # Device dialog
    # -----------------------------------------------------------------
    def _open_device_dialog(self) -> None:
        snap = {
            k: self.store.get(k, None)
            for k in self._DEVICE_KEYS
        }

        dlg = QDialog(self)
        dlg.setWindowTitle("Device && runtime")
        dlg.setModal(True)

        root = QVBoxLayout(dlg)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        note = QLabel(
            "These settings control TensorFlow device selection "
            "and threading. Changes apply when you run Train/Tune.",
            dlg,
        )
        note.setWordWrap(True)
        note.setObjectName("setupCardSubtitle")
        root.addWidget(note, 0)

        w = DeviceOptionsWidget(store=self.store, parent=dlg)
        root.addWidget(w, 1)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel,
            dlg,
        )
        root.addWidget(btns, 0)

        def _accept() -> None:
            dlg.accept()

        def _reject() -> None:
            # Roll back changes made while dialog was open.
            self.store.patch(dict(snap))
            dlg.reject()

        btns.accepted.connect(_accept)
        btns.rejected.connect(_reject)

        dlg.resize(560, 420)
        dlg.exec_()

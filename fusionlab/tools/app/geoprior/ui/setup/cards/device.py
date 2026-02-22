# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.cards.device

Device & runtime.

Modern UX goals
---------------
- Quick controls in-card (no heavy TF probing on tab open).
- One-click "Advanced device options" (opens full widget).
- Clear runtime toggles with safe constraints.
- Badges summarise effective selection.
"""

from __future__ import annotations

from typing import Optional, Set

from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from PyQt5.QtWidgets import QStyle
from ....config.store import GeoConfigStore 
from ..bindings import Binder
from .base import CardBase

class DeviceRuntimeCard(CardBase):
    """Device & runtime (store-driven)."""

    _WATCH: Set[str] = {
        "tf_device_mode",
        "tf_gpu_allow_growth",
        "tf_intra_threads",
        "tf_inter_threads",
        "tf_gpu_memory_limit_mb",
        "use_tf_savedmodel",
        "use_in_memory_model",
        "debug",
    }

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        binder: Binder,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            section_id="device",
            title="Device & runtime",
            subtitle=(
                "Pick CPU/GPU strategy and runtime behavior. "
                "Advanced diagnostics are available on demand."
            ),
            parent=parent,
        )
        self.store = store
        self.binder = binder

        self._build()
        self._wire()
        self.refresh()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build(self) -> None:
        body = self.body_layout()

        # Header actions
        btn_adv = self.add_action(
            text="Advanced…",
            tip="Open full device options + diagnostics",
            icon=QStyle.SP_ComputerIcon,
        )
        btn_adv.clicked.connect(self._open_device_dialog)

        grid = QWidget(self)
        g = QGridLayout(grid)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10)
        g.setColumnStretch(0, 1)
        g.setColumnStretch(1, 1)

        g.addWidget(self._build_device_box(grid), 0, 0)
        g.addWidget(self._build_runtime_box(grid), 0, 1)

        body.addWidget(grid, 0)

    def _build_device_box(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Execution device", parent)
        lay = QGridLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)
        lay.setColumnStretch(1, 1)

        r = 0

        # Device mode
        self.cmb_mode = QComboBox(box)
        items = [
            ("Auto (GPU if available)", "auto"),
            ("CPU only", "cpu"),
            ("GPU only", "gpu"),
        ]
        self.binder.bind_combo(
            "tf_device_mode",
            self.cmb_mode,
            items=items,
            editable=False,
            use_item_data=True,
        )

        lay.addWidget(QLabel("Mode:", box), r, 0)
        lay.addWidget(self.cmb_mode, r, 1)
        r += 1

        # Threads (single override toggle -> 2 fields)
        self.chk_threads = QCheckBox(
            "Override CPU threads",
            box,
        )
        self.chk_threads.setToolTip(
            "When enabled, set intra/inter threads.\n"
            "Applied at run start (may require restart)."
        )

        row = QWidget(box)
        row_l = QHBoxLayout(row)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setSpacing(8)

        self.sp_intra = QSpinBox(row)
        self.sp_intra.setRange(1, 512)
        self.sp_intra.setToolTip("Intra-op threads (>= 1).")

        self.sp_inter = QSpinBox(row)
        self.sp_inter.setRange(1, 512)
        self.sp_inter.setToolTip("Inter-op threads (>= 1).")

        row_l.addWidget(QLabel("Intra:", row), 0)
        row_l.addWidget(self.sp_intra, 0)
        row_l.addSpacing(8)
        row_l.addWidget(QLabel("Inter:", row), 0)
        row_l.addWidget(self.sp_inter, 0)
        row_l.addStretch(1)

        lay.addWidget(self.chk_threads, r, 0, 1, 2)
        r += 1
        lay.addWidget(row, r, 0, 1, 2)
        r += 1

        # GPU memory controls
        mem = QGroupBox("GPU memory", box)
        m = QGridLayout(mem)
        m.setContentsMargins(10, 10, 10, 10)
        m.setHorizontalSpacing(10)
        m.setVerticalSpacing(8)
        m.setColumnStretch(1, 1)

        self.chk_growth = QCheckBox("Allow memory growth", mem)
        self.chk_growth.setToolTip(
            "Allocate GPU memory on demand.\n"
            "Recommended for shared GPUs."
        )
        self.binder.bind_checkbox(
            "tf_gpu_allow_growth",
            self.chk_growth,
        )

        self.chk_cap = QCheckBox("Cap memory (MB)", mem)
        self.sp_cap = QSpinBox(mem)
        self.sp_cap.setRange(256, 1_000_000)
        self.sp_cap.setSingleStep(256)

        self.binder.bind_optional_spin_box(
            "tf_gpu_memory_limit_mb",
            self.sp_cap,
            self.chk_cap,
        )

        m.addWidget(self.chk_growth, 0, 0, 1, 2)
        m.addWidget(self.chk_cap, 1, 0)
        m.addWidget(self.sp_cap, 1, 1)

        lay.addWidget(mem, r, 0, 1, 2)
        r += 1

        return box

    def _build_runtime_box(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Runtime behavior", parent)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.chk_inmem = QCheckBox(
            "Keep model in memory (interactive)",
            box,
        )
        self.chk_inmem.setToolTip(
            "Useful for GUI inference and fast iteration.\n"
            "Disable when exporting or memory is constrained."
        )

        self.chk_saved = QCheckBox(
            "Export as TensorFlow SavedModel",
            box,
        )
        self.chk_saved.setToolTip(
            "If enabled, we prefer SavedModel export.\n"
            "This disables in-memory model caching."
        )

        self.chk_debug = QCheckBox("Debug mode", box)
        self.chk_debug.setToolTip(
            "More logs and checks (slower)."
        )

        # runtime: we keep explicit handlers
        self.chk_inmem.toggled.connect(self._commit_inmem)
        self.chk_saved.toggled.connect(self._commit_savedmodel)
        self.binder.bind_checkbox("debug", self.chk_debug)

        lay.addWidget(self.chk_inmem, 0)
        lay.addWidget(self.chk_saved, 0)

        line = QLabel(
            "Tip: If you force GPU but no GPU exists, "
            "the run may fail. Use Auto to be safe.",
            box,
        )
        line.setWordWrap(True)
        line.setStyleSheet("color: rgba(30,30,30,0.72);")
        lay.addWidget(line, 0)

        lay.addWidget(self.chk_debug, 0)
        lay.addStretch(1)

        return box

    # -----------------------------------------------------------------
    # Store wiring
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.chk_threads.toggled.connect(self._commit_threads)
        self.sp_intra.valueChanged.connect(self._commit_threads)
        self.sp_inter.valueChanged.connect(self._commit_threads)

        self.store.config_changed.connect(self._on_changed)
        self.store.config_replaced.connect(lambda *_a: self.refresh())

    def _on_changed(self, keys: object) -> None:
        if not isinstance(keys, (set, list, tuple)):
            self.refresh()
            return
        ks = {str(k) for k in keys}
        if not ks or (ks & self._WATCH):
            self.refresh()

    # -----------------------------------------------------------------
    # Commits (custom logic)
    # -----------------------------------------------------------------
    def _commit_threads(self, *_a: object) -> None:
        if not self.chk_threads.isChecked():
            self.store.patch(
                {
                    "tf_intra_threads": None,
                    "tf_inter_threads": None,
                }
            )
            self._apply_threads_enabled(False)
            return

        self.store.patch(
            {
                "tf_intra_threads": int(self.sp_intra.value()),
                "tf_inter_threads": int(self.sp_inter.value()),
            }
        )
        self._apply_threads_enabled(True)

    def _commit_inmem(self, on: bool) -> None:
        # If SavedModel is on, in-memory is forced off.
        if self.chk_saved.isChecked():
            with QSignalBlocker(self.chk_inmem):
                self.chk_inmem.setChecked(False)
            self.store.patch({"use_in_memory_model": False})
            self._apply_runtime_guard()
            return

        self.store.patch({"use_in_memory_model": bool(on)})
        self._apply_runtime_guard()

    def _commit_savedmodel(self, on: bool) -> None:
        if on:
            self.store.patch(
                {
                    "use_tf_savedmodel": True,
                    "use_in_memory_model": False,
                }
            )
        else:
            self.store.patch({"use_tf_savedmodel": False})
        self._apply_runtime_guard()
        self.refresh()

    # -----------------------------------------------------------------
    # Dialog
    # -----------------------------------------------------------------
    def _open_device_dialog(self) -> None:
        # Imported on demand to avoid heavy TF probing
        # during Setup tab construction.
        from ....device_options import DeviceOptionsWidget

        dlg = QDialog(self)
        dlg.setWindowTitle("Device options")
        dlg.resize(880, 460)

        root = QVBoxLayout(dlg)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        w = DeviceOptionsWidget(store=self.store, parent=dlg)
        root.addWidget(w, 1)

        btns = QDialogButtonBox(QDialogButtonBox.Close, dlg)
        btns.rejected.connect(dlg.reject)
        root.addWidget(btns, 0)

        dlg.exec_()

    # -----------------------------------------------------------------
    # Refresh
    # -----------------------------------------------------------------
    def refresh(self) -> None:
        cfg = self.store.cfg

        # Threads state
        has_threads = (
            cfg.tf_intra_threads is not None
            or cfg.tf_inter_threads is not None
        )
        with QSignalBlocker(self.chk_threads):
            self.chk_threads.setChecked(bool(has_threads))

        if cfg.tf_intra_threads is not None:
            with QSignalBlocker(self.sp_intra):
                self.sp_intra.setValue(int(cfg.tf_intra_threads))
        if cfg.tf_inter_threads is not None:
            with QSignalBlocker(self.sp_inter):
                self.sp_inter.setValue(int(cfg.tf_inter_threads))

        self._apply_threads_enabled(bool(has_threads))

        # Runtime toggles
        with QSignalBlocker(self.chk_inmem):
            self.chk_inmem.setChecked(bool(cfg.use_in_memory_model))
        with QSignalBlocker(self.chk_saved):
            self.chk_saved.setChecked(bool(cfg.use_tf_savedmodel))

        self._apply_runtime_guard()
        self._refresh_badges()

    def _apply_threads_enabled(self, on: bool) -> None:
        self.sp_intra.setEnabled(bool(on))
        self.sp_inter.setEnabled(bool(on))

    def _apply_runtime_guard(self) -> None:
        saved = bool(self.chk_saved.isChecked())
        self.chk_inmem.setEnabled(not saved)
        if saved:
            self.chk_inmem.setChecked(False)

    def _refresh_badges(self) -> None:
        cfg = self.store.cfg
        mode = (cfg.tf_device_mode or "auto").lower().strip()

        mode_txt = {"auto": "Auto", "cpu": "CPU", "gpu": "GPU"}.get(
            mode,
            "Auto",
        )
        accent = "warn" if mode == "gpu" else ""
        self.badge(
            "backend",
            text=f"Backend: {mode_txt}",
            accent=accent,
            tip="Execution target at run start",
        )

        if (
            cfg.tf_intra_threads is None
            and cfg.tf_inter_threads is None
        ):
            th = "Threads: auto"
        else:
            th = (
                f"Threads: {cfg.tf_intra_threads or '?'}"
                f"/{cfg.tf_inter_threads or '?'}"
            )
        self.badge("threads", text=th)

        cap = cfg.tf_gpu_memory_limit_mb
        if cap is None:
            gpu = "GPU mem: growth" if cfg.tf_gpu_allow_growth else "GPU mem: fixed"
        else:
            gpu = f"GPU mem: {int(cap)}MB"
        self.badge("gpu", text=gpu)

        rt = []
        if cfg.use_tf_savedmodel:
            rt.append("SavedModel")
        if cfg.use_in_memory_model and not cfg.use_tf_savedmodel:
            rt.append("In-memory")
        if cfg.debug:
            rt.append("Debug")

        rt_txt = "Runtime: " + (", ".join(rt) if rt else "default")
        rt_acc = "warn" if cfg.use_tf_savedmodel else ""
        self.badge(
            "runtime",
            text=rt_txt,
            accent=rt_acc,
        )

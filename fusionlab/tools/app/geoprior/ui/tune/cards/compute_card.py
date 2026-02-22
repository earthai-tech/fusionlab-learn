# geoprior/ui/tune/cards/compute_card.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Compute & parallelism card (Tune tab).

- Uses shared make_card() factory for styling.
- Expands inline (same card) on Edit.
- Store-driven; robust against small store API shifts.
- Does NOT duplicate runtime/compute diagnostics
  (already shown in the navigator panel).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ....config.prior_schema import FieldKey
from ....config.store import GeoConfigStore
from ...icon_utils import try_icon

__all__ = ["TuneComputeCard"]

MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]

# UI-only store key (safe + non-schema)
_WORKERS_KEY = "tune.workers"


class TuneComputeCard(QWidget):
    """
    Compute & parallelism card with inline expansion.

    Controls (store keys)
    --------------------
    - FieldKey("tf_device_mode")         : "auto"|"cpu"|"gpu"
    - FieldKey("tf_intra_threads")       : int|None
    - FieldKey("tf_inter_threads")       : int|None
    - FieldKey("tf_gpu_allow_growth")    : bool
    - FieldKey("tf_gpu_memory_limit_mb") : int|None
    - store[_WORKERS_KEY]                : int (UI-only)
    """

    changed = pyqtSignal()
    edit_toggled = pyqtSignal(bool)

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

        self._frame, body = self._make_card("Compute & parallelism")
        root.addWidget(self._frame)

        # Summary + Edit (same row)
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

        self.lbl_help = QLabel(
            "Configure device selection, CPU threads, GPU memory "
            "behavior, and parallel workers.",
            self._frame,
        )
        self.lbl_help.setObjectName("helpText")
        self.lbl_help.setWordWrap(True)
        body.addWidget(self.lbl_help)

        # Drawer (collapsed)
        self.details = QWidget(self._frame)
        self.details.setObjectName("drawer")
        self.details.setVisible(False)

        dlay = QVBoxLayout(self.details)
        dlay.setContentsMargins(0, 6, 0, 0)
        dlay.setSpacing(10)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)
        grid.setColumnStretch(3, 1)

        r = 0

        # Device mode
        grid.addWidget(QLabel("Device:"), r, 0, alignment=Qt.AlignRight)
        self.cmb_device = QComboBox(self.details)
        self.cmb_device.addItem("Auto", "auto")
        self.cmb_device.addItem("CPU", "cpu")
        self.cmb_device.addItem("GPU", "gpu")
        self.cmb_device.setToolTip(
            "Auto uses GPU if available else CPU.\n"
            "CPU forces CPU only.\n"
            "GPU forces GPU only (if present)."
        )
        grid.addWidget(self.cmb_device, r, 1, 1, 3)
        r += 1

        # Workers
        grid.addWidget(QLabel("Workers:"), r, 0, alignment=Qt.AlignRight)
        self.sp_workers = QSpinBox(self.details)
        self.sp_workers.setRange(0, 64)
        self.sp_workers.setToolTip(
            "Parallel workers for tuning / data pipeline.\n"
            "0 means 'auto' / single-threaded fallback."
        )
        grid.addWidget(self.sp_workers, r, 1, 1, 3)
        r += 1

        # Thread override
        self.chk_threads = QCheckBox("Override CPU threads", self.details)
        self.chk_threads.setToolTip(
            "When unchecked, TF chooses thread counts.\n"
            "When checked, intra/inter values are stored."
        )
        grid.addWidget(self.chk_threads, r, 1, 1, 3)
        r += 1

        # Intra / Inter
        grid.addWidget(QLabel("Intra-op:"), r, 0, alignment=Qt.AlignRight)
        self.sp_intra = QSpinBox(self.details)
        self.sp_intra.setRange(1, 512)
        grid.addWidget(self.sp_intra, r, 1)

        grid.addWidget(QLabel("Inter-op:"), r, 2, alignment=Qt.AlignRight)
        self.sp_inter = QSpinBox(self.details)
        self.sp_inter.setRange(1, 512)
        grid.addWidget(self.sp_inter, r, 3)
        r += 1

        # GPU memory controls
        self.chk_growth = QCheckBox("GPU memory growth", self.details)
        self.chk_growth.setToolTip("If enabled, allocate GPU memory on demand.")
        grid.addWidget(self.chk_growth, r, 1, 1, 3)
        r += 1

        grid.addWidget(QLabel("GPU limit:"), r, 0, alignment=Qt.AlignRight)
        self.sp_gpu_mb = QSpinBox(self.details)
        self.sp_gpu_mb.setRange(0, 1_000_000)
        self.sp_gpu_mb.setSuffix(" MB")
        self.sp_gpu_mb.setToolTip(
            "0 means 'no explicit cap'.\n"
            "Otherwise sets a per-process GPU memory limit."
        )
        grid.addWidget(self.sp_gpu_mb, r, 1, 1, 3)

        dlay.addLayout(grid)
        body.addWidget(self.details)

        self._apply_enabled_state()

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

        self.cmb_device.currentIndexChanged.connect(lambda _=0: self._commit())
        self.sp_workers.valueChanged.connect(lambda _=0: self._commit())

        self.chk_threads.toggled.connect(self._on_threads_toggle)
        self.sp_intra.valueChanged.connect(lambda _=0: self._commit())
        self.sp_inter.valueChanged.connect(lambda _=0: self._commit())

        self.chk_growth.toggled.connect(lambda _=0: self._commit())
        self.sp_gpu_mb.valueChanged.connect(lambda _=0: self._commit())

        for nm in ("config_changed", "config_replaced"):
            sig = getattr(self._store, nm, None)
            if sig is not None and hasattr(sig, "connect"):
                try:
                    sig.connect(self.refresh_from_store)
                except Exception:
                    pass

    # -----------------------------------------------------------------
    # Store <-> UI
    # -----------------------------------------------------------------
    def refresh_from_store(self, *_: Any) -> None:
        self._writing = True
        try:
            mode = self._get_fk("tf_device_mode", "auto")
            mode = str(mode or "auto").strip().lower()
            if mode not in {"auto", "cpu", "gpu"}:
                mode = "auto"

            workers = int(self._get_extra(_WORKERS_KEY, 0) or 0)

            intra = self._as_int_or_none(self._get_fk("tf_intra_threads", None))
            inter = self._as_int_or_none(self._get_fk("tf_inter_threads", None))

            allow_growth = bool(self._get_fk("tf_gpu_allow_growth", True))
            lim = self._as_int_or_none(self._get_fk("tf_gpu_memory_limit_mb", None))

            with QSignalBlocker(self.cmb_device):
                j = self.cmb_device.findData(mode)
                self.cmb_device.setCurrentIndex(max(j, 0))

            with QSignalBlocker(self.sp_workers):
                self.sp_workers.setValue(max(0, workers))

            has_threads = (intra is not None) or (inter is not None)
            with QSignalBlocker(self.chk_threads):
                self.chk_threads.setChecked(bool(has_threads))

            with QSignalBlocker(self.sp_intra):
                self.sp_intra.setValue(int(intra or 1))
            with QSignalBlocker(self.sp_inter):
                self.sp_inter.setValue(int(inter or 1))

            with QSignalBlocker(self.chk_growth):
                self.chk_growth.setChecked(bool(allow_growth))

            with QSignalBlocker(self.sp_gpu_mb):
                self.sp_gpu_mb.setValue(int(lim or 0))

            self._apply_enabled_state()
            self._refresh_summary()
        finally:
            self._writing = False

    def _commit(self) -> None:
        if self._writing:
            return

        mode = str(self.cmb_device.currentData() or "auto").strip().lower()
        if mode not in {"auto", "cpu", "gpu"}:
            mode = "auto"

        workers = int(self.sp_workers.value())

        if self.chk_threads.isChecked():
            intra = int(self.sp_intra.value())
            inter = int(self.sp_inter.value())
        else:
            intra = None
            inter = None

        allow_growth = bool(self.chk_growth.isChecked())
        lim_mb = int(self.sp_gpu_mb.value())
        lim_mb = None if lim_mb <= 0 else int(lim_mb)

        self._writing = True
        try:
            self._set_fk("tf_device_mode", mode)
            self._set_extra(_WORKERS_KEY, workers)

            self._set_fk("tf_intra_threads", intra)
            self._set_fk("tf_inter_threads", inter)

            self._set_fk("tf_gpu_allow_growth", allow_growth)
            self._set_fk("tf_gpu_memory_limit_mb", lim_mb)
        finally:
            self._writing = False

        self._apply_enabled_state()
        self._refresh_summary()
        self.changed.emit()

    # -----------------------------------------------------------------
    # UI helpers
    # -----------------------------------------------------------------
    def _on_threads_toggle(self, _: bool) -> None:
        self._apply_enabled_state()
        self._commit()

    def _apply_enabled_state(self) -> None:
        th_on = bool(self.chk_threads.isChecked())
        self.sp_intra.setEnabled(th_on)
        self.sp_inter.setEnabled(th_on)

    def _refresh_summary(self) -> None:
        mode = str(self.cmb_device.currentData() or "auto")
        workers = int(self.sp_workers.value())

        intra = "auto"
        inter = "auto"
        if self.chk_threads.isChecked():
            intra = str(int(self.sp_intra.value()))
            inter = str(int(self.sp_inter.value()))

        lim = int(self.sp_gpu_mb.value())
        lim_s = "auto" if lim <= 0 else f"{lim}MB"

        grow = "on" if self.chk_growth.isChecked() else "off"

        self.lbl_sum.setText(
            f"device={mode} · workers={workers} · "
            f"intra={intra} inter={inter} · "
            f"gpu={grow} limit={lim_s}"
        )

    # -----------------------------------------------------------------
    # Store helpers (robust)
    # -----------------------------------------------------------------
    def _get_fk(self, name: str, default: Any) -> Any:
        fk = FieldKey(str(name))
        try:
            return self._store.get_value(fk, default=default)
        except Exception:
            return default

    def _set_fk(self, name: str, value: Any) -> None:
        fk = FieldKey(str(name))
        try:
            self._store.set_value_by_key(fk, value)
            return
        except Exception:
            pass
        try:
            self._store.patch_fields({str(name): value})
        except Exception:
            pass

    def _get_extra(self, key: str, default: Any) -> Any:
        try:
            return self._store.get(key, default)
        except Exception:
            return default

    def _set_extra(self, key: str, value: Any) -> None:
        try:
            self._store.set(key, value)
        except Exception:
            pass

    @staticmethod
    def _as_int_or_none(v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            iv = int(v)
        except Exception:
            return None
        if iv <= 0:
            return None
        return iv
